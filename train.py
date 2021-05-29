import numpy as np
import torch
import os
import argparse
from load_scannet import ScannetDataset, SceneFiles
from models.superglue import SuperGlue
from models.superpoint import SuperPoint
from utils import ground_truth_matches
from multiprocessing import Pool


def loss_fn(P, match_indices0, match_indices1, match_weights0, match_weights1):
    bs, ft, _ = P.shape

    l0 = -P.reshape(bs*ft, ft)[range(bs*ft), match_indices0.reshape(bs*ft)]
    l1 = -P.transpose(1,2).reshape(bs*ft, ft)[range(bs*ft), match_indices1.reshape(bs*ft)]

    loss = torch.dot(l0, match_weights0.reshape(bs*ft)) + torch.dot(l1, match_weights1.reshape(bs*ft))

    return loss / bs


def process_keypoints(kpts0, kpts1, depth0, depth1, K, T_0to1):
    MI, MJ, WI, WJ = ground_truth_matches(kpts0.cpu().numpy(), kpts1.cpu().numpy(),
            K.numpy(), K.numpy(), T_0to1.numpy(), depth0.numpy(), depth1.numpy())
    return {
        'match_indices0': torch.tensor(MI),
        'match_indices1': torch.tensor(MJ),
        'match_weights0': torch.tensor(WI),
        'match_weights1': torch.tensor(WJ)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SuperGlue training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'scans_path', type=str,
        help='Path to the directory of scans')
    parser.add_argument(
        'pairs_path', type=str,
        help='Path to training pair files')
    parser.add_argument(
        '--num_epochs', type=int, default=500,
        help='Number of epochs')
    parser.add_argument(
        '--num_iters', type=int, default=900000,
        help='Break after this many iterations')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size')
    parser.add_argument(
        '--num_threads', type=int, default=0,
        help='Number of threads for the DataLoader and for computing groundtruth')
    parser.add_argument(
        '--num_batches_per_optimizer_step', type=int, default=1,
        help='Number of batches processed before running the optimization step')
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='Learning rate')
    parser.add_argument(
        '--lr_decay', type=float, default=0.999992,
        help='Exponential learning rate decay')
    parser.add_argument(
        '--lr_decay_iter', type=int, default=100000,
        help='Decay the learning rate after this iteration')
    parser.add_argument(
        '--checkpoint_dir', type=str, default="dump_weights",
        help='Weights are saved periodically to this directory')
    parser.add_argument(
        '--save_every_n_epochs', type=int, default=1,
        help='Weights are saved after every n epochs')
    opt = parser.parse_args()


    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)

    scene_files = SceneFiles(opt.pairs_path, opt.scans_path)
    scannet = ScannetDataset(scene_files)
    trainloader = torch.utils.data.DataLoader(dataset=scannet, shuffle=False,
            batch_size=opt.batch_size, num_workers=opt.num_threads, drop_last=True)

    superglue = SuperGlue({}).train().cuda()
    superpoint = SuperPoint({
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 400
    }).eval().cuda()

    optimizer = torch.optim.Adam(params=superglue.parameters(), lr=opt.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=opt.lr_decay)
    pool = Pool(max(opt.num_threads, 1))
    iteration = 0

    for epoch in range(opt.num_epochs):
        for i, data in enumerate(trainloader):

            # detect superpoint keypoints
            with torch.no_grad():
                p1 = superpoint({'image': data['image0'].cuda()})
                p2 = superpoint({'image': data['image1'].cuda()})

            # compute groundtruth matches
            processed = []
            params = zip(p1['keypoints'], p2['keypoints'],
                    data['depth0'], data['depth1'], data['intrinsics'], data['T_0to1'])
            if opt.num_threads == 0:
                for p in params:
                    processed.append(process_keypoints(*p))
            else:
                # TODO - untested with more than one cpu. not sure if faster
                processed = pool.starmap(process_keypoints, params)
            inputs = {}
            for k in processed[0]:
                inputs[k] = torch.stack([e[k] for e in processed], axis=0)
            inputs['keypoints0'] = torch.stack(p1['keypoints'], axis=0)
            inputs['descriptors0'] = torch.stack(p1['descriptors'], axis=0)
            inputs['scores0'] = torch.stack(p1['scores'], axis=0)
            inputs['keypoints1'] = torch.stack(p2['keypoints'], axis=0)
            inputs['descriptors1'] = torch.stack(p2['descriptors'], axis=0)
            inputs['scores1'] = torch.stack(p2['scores'], axis=0)

            # superglue forward and backward pass
            if i % opt.num_batches_per_optimizer_step == 0:
                optimizer.zero_grad()
            P = superglue(inputs)
            loss = loss_fn(P, inputs["match_indices0"].cuda(), inputs["match_indices1"].cuda(),
                    inputs["match_weights0"].cuda(), inputs["match_weights1"].cuda())
            loss.backward()
            if i % opt.num_batches_per_optimizer_step == opt.num_batches_per_optimizer_step - 1:
                optimizer.step()
                iteration += 1

            # update learning rate
            if iteration > opt.lr_decay_iter and i % opt.num_batches_per_optimizer_step == 0:
                lr_scheduler.step()
            
            print("Epoch:", epoch, "Batch:", i, "Loss:", loss)

        if epoch % opt.save_every_n_epochs == 0:
            torch.save(superglue.state_dict(), "{}/superglue_weights_{:04d}.pth".format(opt.checkpoint_dir, epoch))
        if iteration == opt.num_iters:
            torch.save(superglue.state_dict(), "{}/superglue_weights_{:04d}.pth".format(opt.checkpoint_dir, epoch))
            break
        scannet.resample(scene_files)
