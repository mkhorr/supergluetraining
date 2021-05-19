import numpy as np
import torch
import psutil
import os
import time
from load_scannet import ScannetDataset, SceneFiles
from models.superglue import SuperGlue
from models.superpoint import SuperPoint
from utils import ground_truth_matches
#from multiprocessing import Pool
from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


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

    out_dir = "dump_weights"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    scene_files = SceneFiles("../first_scene/training_pairs", "../first_scene/scannet")
    scannet = ScannetDataset(scene_files)
    trainloader = torch.utils.data.DataLoader(dataset=scannet, shuffle=False, batch_size=24, num_workers=psutil.cpu_count(logical=False), drop_last=True)

    superglue = SuperGlue({}).train().cuda()
    superpoint = SuperPoint({
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 400
    }).eval().cuda()

    optimizer = torch.optim.Adam(params=superglue.parameters(), lr=2e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999992)

    pool = Pool(psutil.cpu_count(logical=False))
    iteration = 0

    for epoch in range(410):

        for i, data in enumerate(trainloader):

            # detect superpoint keypoints
            with torch.no_grad():
                p1 = superpoint({'image': data['image0'].cuda()})
                p2 = superpoint({'image': data['image1'].cuda()})

            # compute groundtruth matches
            processed = pool.starmap(process_keypoints,
                zip(p1['keypoints'], p2['keypoints'],
                    data['depth0'], data['depth1'], data['intrinsics'], data['T_0to1']))
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
            if i % 8 == 0:
                optimizer.zero_grad()
            P = superglue(inputs)
            loss = loss_fn(P, inputs["match_indices0"].cuda(), inputs["match_indices1"].cuda(),
                    inputs["match_weights0"].cuda(), inputs["match_weights1"].cuda())
            loss.backward()
            if i % 8 == 7:
                optimizer.step()

            print("Epoch:", epoch, "Batch:", i, "Loss:", loss)

            if epoch > 50 and i % 8 == 0: #iteration >= 100000:
                lr_scheduler.step()
            iteration += 1

        torch.save(superglue.state_dict(), "{}/superglue_weights_{:04d}.pth".format(out_dir, epoch))
        if iteration == 900000:
            break
        scannet.resample(scene_files)
    
        



