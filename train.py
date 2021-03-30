import numpy as np
import numba as nb
import torch
import psutil
from load_scannet import ScannetDataset, SceneFiles
from models.superglue import SuperGlue
from models.superpoint import SuperPoint
from multiprocessing import Pool


NUM_KEYPOINTS = 400

def loss_fn(P, match_indices0, match_indices1, match_weights0, match_weights1, epoch):
    bs, ft, _ = P.shape

    l0 = -P.reshape(bs*ft, ft)[range(bs*ft), match_indices0.reshape(bs*ft)]
    l1 = -P.transpose(1,2).reshape(bs*ft, ft)[range(bs*ft), match_indices1.reshape(bs*ft)]

    loss = torch.dot(l0, match_weights0.reshape(bs*ft)) + torch.dot(l1, match_weights1.reshape(bs*ft))
    loss = torch.sum(l0) + torch.sum(l1)

    return loss / bs


def process_keypoints(kpts0, des0, scores0, kpts1, des1, scores1, depth0, depth1, K, T_0to1):

    MI, MJ, WI, WJ = ground_truth_matches(kpts0.cpu().numpy(), kpts1.cpu().numpy(),
            K.numpy(), K.numpy(), T_0to1.numpy(), depth0.numpy(), depth1.numpy())

    # padding tensors with 0 to handle varying numbers of keypoints
    if kpts0.shape[1] != NUM_KEYPOINTS:
        excess = NUM_KEYPOINTS - kpts0.shape[0]
        kpts0 = torch.cat((kpts0, torch.cuda.FloatTensor(excess, 2).fill_(0)), 0)
        des0 = torch.cat((des0, torch.cuda.FloatTensor(256, excess).fill_(0)), 1)
        scores0 = torch.cat((scores0, torch.cuda.FloatTensor(excess).fill_(0)), 0)
    if kpts1.shape[1] != NUM_KEYPOINTS:
        excess = NUM_KEYPOINTS - kpts1.shape[0]
        kpts1 = torch.cat((kpts1, torch.cuda.FloatTensor(excess, 2).fill_(0)), 0)
        des1 = torch.cat((des1, torch.cuda.FloatTensor(256, excess).fill_(0)), 1)
        scores1 = torch.cat((scores1, torch.cuda.FloatTensor(excess).fill_(0)), 0)
    
    return {
        'keypoints0': kpts0,
        'keypoints1': kpts1,
        'descriptors0': des0,
        'descriptors1': des1,
        'scores0': scores0,
        'scores1': scores1,
        'match_indices0': torch.tensor(MI).cuda(),
        'match_indices1': torch.tensor(MJ).cuda(),
        'match_weights0': torch.tensor(WI).cuda(),
        'match_weights1': torch.tensor(WJ).cuda()
    }
    

def move_to_cpu(p):
    tmp = p['scores']
    p['scores'] = []
    for i in range(len(p['keypoints'])):
        p['keypoints'][i] = p['keypoints'][i].cpu()
        p['descriptors'][i] = p['descriptors'][i].cpu()
        p['scores'].append(tmp[i].cpu())


#todo - implement differently, verify carefully, and make it work with cuda tensors
@nb.njit
def ground_truth_matches(kpts0, kpts1, K0, K1, T0to1, depth0, depth1):
    
    m = kpts0.shape[0]
    n = kpts1.shape[0]

    depths0 = np.full(fill_value=np.nan,shape=(m,1), dtype=np.float32)
    for i in range(m):
        depths0[i][0] = depth0[np.int32(kpts0[i][1]), np.int32(kpts0[i][0])]
    kpts0to1 = (K1 @ T0to1 @ np.linalg.inv(K0)) @ np.concatenate((kpts0 * depths0, depths0, np.ones((m,1), dtype=np.float32)), axis=1).T
    depths0to1 = np.expand_dims(kpts0to1.T[:, 2], 1)
    kpts0to1 = (kpts0to1.T / depths0to1)[:,:2]

    depths1 = np.full(fill_value=np.nan,shape=(n,1), dtype=np.float32)
    for i in range(n):
        depths1[i][0] = depth1[np.int32(kpts1[i][1]), np.int32(kpts1[i][0])]
    kpts1to0 = (K0 @ np.linalg.inv(T0to1) @ np.linalg.inv(K1)) @ np.concatenate((kpts1 * depths1, depths1, np.ones((n,1), dtype=np.float32)), axis=1).T
    depths1to0 = np.expand_dims(kpts1to0.T[:, 2], 1)
    kpts1to0 = (kpts1to0.T / depths1to0)[:,:2]

    match_indices0 = np.full((401,1), -1, dtype=np.int64)
    match_indices1 = np.full((401,1), -1, dtype=np.int64)
    match_weights0 = np.full((401,1), 0, dtype=np.float32)
    match_weights1 = np.full((401,1), 0, dtype=np.float32)
    match_cnt = 0
    for i in range(m):
        errs = np.sum((kpts1to0 - kpts0[i])**2, axis=1) + np.sum((kpts1 - kpts0to1[i])**2, axis=1)
        j = np.argmin(errs)
        if errs[j] <= 50 and depths1[j][0] != 0 and abs(depths0to1[i][0] - depths1[j][0]) / depths1[j][0] <= .1:
            if match_indices1[j] == -1:#TODO
                match_indices0[i] = j
                match_indices1[j] = i
            match_cnt += 1

    if match_cnt == 0:
        return match_indices0, match_indices1, match_weights0, match_weights1

    matched_weight = 2 * match_cnt / (m + n)
    unmatched_weight = .5 / (1 - matched_weight)
    matched_weight = .5 / matched_weight
    for i in range(m):
        match_weights0[i] = unmatched_weight if match_indices0[i] == -1 else matched_weight
    for j in range(n):
        match_weights1[j] = unmatched_weight if match_indices1[j] == -1 else matched_weight
    return match_indices0, match_indices1, match_weights0, match_weights1




if __name__ == "__main__":

    scene_files = SceneFiles("D:/training_pairs", "D:/scannet/raw/scans")
    out_dir = "dump_weights"
    scannet = ScannetDataset(scene_files)
    trainloader = torch.utils.data.DataLoader(dataset=scannet, shuffle=False, batch_size=64, num_workers=psutil.cpu_count(logical=False)-1)

    superglue = SuperGlue({}).train().cuda()
    superpoint = SuperPoint({
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': NUM_KEYPOINTS
    }).eval().cuda()

    optimizer = torch.optim.Adam(params=superglue.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999992)

    pool = Pool(psutil.cpu_count(logical=False)-1)
    iteration = 0

    for epoch in range(5000):

        for i, data in enumerate(trainloader):

            with torch.no_grad():
                p1 = superpoint({'image': data['image0'].cuda()})
                p2 = superpoint({'image': data['image1'].cuda()})

            # TODO - remove on linux
            #move_to_cpu(p1)
            #move_to_cpu(p2)

            # TODO - maybe define a custom transformation for this?
            processed = pool.starmap(process_keypoints,
                zip(p1['keypoints'], p1['descriptors'], p1['scores'],
                    p2['keypoints'], p2['descriptors'], p2['scores'],
                    data['depth0'], data['depth1'], data['intrinsics'], data['T_0to1']))
            inputs = {}
            for k in processed[0]:
                inputs[k] = torch.stack([e[k] for e in processed], axis=0)

            optimizer.zero_grad()
            P = superglue(inputs)
            loss = loss_fn(P, inputs["match_indices0"], inputs["match_indices1"],
                    inputs["match_weights0"], inputs["match_weights1"], epoch)
            loss.backward()
            optimizer.step()

            print(epoch, i, loss)
            if iteration >= 100000:
                lr_scheduler.step()
            iteration += 1

        torch.save(superglue.state_dict(), "{}/superglue_weights_{:04d}.pth".format(out_dir, epoch))
        if iteration == 900000:
            break
        scannet.resample(scene_files)
    
        



