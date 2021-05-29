import numpy as np
import cv2
from extract_sens import SceneInfo
from PIL import Image
from multiprocessing import Pool
import os
import time
import numba as nb
import argparse
import psutil

@nb.njit
def relative_depth_error(d, d_expected):
    return abs(d - d_expected) / d_expected

@nb.njit
def frustum_intersect(w, h, K, K_inv, E1, E2):

    z_far = np.diag(np.array([20.0, 20.0, 20.0, 1.0]))
    W1 = E1 @ z_far @ K_inv
    W2 = E2 @ z_far @ K_inv

    # frustum points
    p1 = np.empty((5, 3))
    p1[0] = (W1 @ np.array([0, 0, 0, 1.0]).astype(np.float64))[:3]
    p1[1] = (W1 @ np.array([0, 0, 1, 1.0]).astype(np.float64))[:3]
    p1[2] = (W1 @ np.array([w, 0, 1, 1.0]).astype(np.float64))[:3]
    p1[3] = (W1 @ np.array([w, h, 1, 1.0]).astype(np.float64))[:3]
    p1[4] = (W1 @ np.array([0, h, 1, 1.0]).astype(np.float64))[:3]
    
    p2 = np.empty((5, 3))
    p2[0] = (W2 @ np.array([0, 0, 0, 1]).astype(np.float64))[:3]
    p2[1] = (W2 @ np.array([0, 0, 1, 1]).astype(np.float64))[:3]
    p2[2] = (W2 @ np.array([w, 0, 1, 1]).astype(np.float64))[:3]
    p2[3] = (W2 @ np.array([w, h, 1, 1]).astype(np.float64))[:3]
    p2[4] = (W2 @ np.array([0, h, 1, 1]).astype(np.float64))[:3]

    # points wrt camera center
    pc1 = p1 - p1[0]
    pc2 = p2 - p2[0]

    # plane normals
    n1 = np.empty((5, 3))
    n1[0] = np.cross(pc1[4], pc1[1])
    n1[1] = np.cross(pc1[2], pc1[3])
    n1[2] = np.cross(pc1[1], pc1[2])
    n1[3] = np.cross(pc1[3], pc1[4])
    n1[4] = pc1[1] + (pc1[3] - pc1[1]) / 2

    proj_p1_to_n1 = n1 @ p1.T
    proj_p2_to_n1 = n1 @ p2.T
    for i in range(proj_p1_to_n1.shape[0]):
        if not points_overlap(proj_p1_to_n1[i], proj_p2_to_n1[i]):
            return False

    n2 = np.empty((5, 3))
    n2[0] = np.cross(pc2[4], pc2[1])
    n2[1] = np.cross(pc2[2], pc2[3])
    n2[2] = np.cross(pc2[1], pc2[2])
    n2[3] = np.cross(pc2[3], pc2[4])
    n2[4] = pc2[1] + (pc2[3] - pc2[1]) / 2

    proj_p1_to_n2 = n2 @ p1.T
    proj_p2_to_n2 = n2 @ p2.T
    for i in range(proj_p1_to_n2.shape[0]):
        if not points_overlap(proj_p1_to_n2[i], proj_p2_to_n2[i]):
            return False

    for i in range(n1.shape[0]):
        for j in range(n2.shape[0]):
            n = np.cross(n1[i], n2[j])
            if not points_overlap(p1 @ n, p2 @ n):
                return False

    return True

@nb.njit
def points_overlap(p1, p2):
    return not (max(p1) < min(p2) or max(p2) < min(p1))

class ScanProcessor:

    def __init__(self, data_dir, scanId, min_score=.3, frame_skip=1, width=640, height=480, stride=10):
        self.scan_dir = data_dir + "/" + scanId
        self.scene_info = SceneInfo(self.scan_dir + "/_info.txt")
        self.scene_info.resize_depth(width, height)
        self.K = self.scene_info.calibration_depth_intrinsic
        self.K_inv = np.linalg.inv(self.K)
        self.depth_images = {}
        self.poses = {}
        self.inv_poses = {}
        self.scores = []
        self.frame_skip = frame_skip
        self.min_score = min_score
        self.width = width
        self.height = height
        self.stride = stride

        R,C = np.mgrid[:height//stride,:width//stride] * stride
        R = R.ravel()
        C = C.ravel()
        num_pixels = (height // stride) * (width // stride)
        ones = np.ones(num_pixels)
        self.ones = ones
        self.coords = np.column_stack((C, R, ones, ones))


    def compute_overlap(self, frame1_idx, frame2_idx):
        score = self.compute_overlap_score(self.width, self.height, self.stride, self.coords,
                self.ones, self.depth_images[frame1_idx], self.depth_images[frame2_idx],
                self.poses[frame1_idx], self.poses[frame2_idx], self.inv_poses[frame1_idx],
                self.inv_poses[frame2_idx], self.K, self.K_inv)
        if score >= self.min_score:
            self.scores.append((frame1_idx, frame2_idx, score))

    def load_depth_and_poses(self):
        for i in range(0, self.scene_info.num_frames, self.frame_skip):
            img = cv2.imread("{}/frame-{:06d}.depth.png".format(self.scan_dir, i), -1)
            if img.shape[0] != self.height or img.shape[1] != self.width:
                img = cv2.resize(img, (self.width, self.height))
            self.depth_images[i] = img / 1000.0
            self.poses[i] = np.loadtxt("{}/frame-{:06d}.pose.txt".format(self.scan_dir, i))
            self.inv_poses[i] = np.linalg.inv(self.poses[i])

    def compute_overlaps(self):
        # TODO - try multprocessing on linux - wasn't speeding up on windows
        for i in range(0, self.scene_info.num_frames):
            for j in range(i + 1, self.scene_info.num_frames):
                self.compute_overlap(i, j)
        
        self.scores.sort()

    def write_scores(self, outfilepath):
        to_save = np.array(self.scores)
        np.save(outfilepath, to_save)
    
    @staticmethod
    @nb.njit
    def compute_overlap_score(w, h, stride, coords, ones, frame1_depth, frame2_depth,
            frame1_pose, frame2_pose, frame1_pose_inv, frame2_pose_inv, K, K_inv):

        if not frustum_intersect(w, h, K, K_inv, frame1_pose, frame2_pose):
            return 0.0

        M21 = frame2_pose_inv @ frame1_pose
        M12 = frame1_pose_inv @ frame2_pose

        depths1 = frame1_depth[::stride,::stride].ravel()
        depths2 = frame2_depth[::stride,::stride].ravel()

        coords1 = coords.T * depths1
        coords1[3] = ones
        coords2 = coords.T * depths2
        coords2[3] = ones

        points1 = (K @ M21 @ K_inv) @ coords1
        a_pixels_in_b = 0
        for p in points1.T:
            d = p[2]
            x = int(p[0] / d)
            y = int(p[1] / d)
            if d <= 1e-9 or x < 0 or x >= w or y < 0 or y >= h:
                continue
            d_expected = frame2_depth[y][x]
            if d_expected <= 1e-9 or relative_depth_error(d, d_expected) > .1:
                continue
            a_pixels_in_b += 1

        points2 = (K @ M12 @ K_inv) @ coords2
        b_pixels_in_a = 0
        for p in points2.T:
            d = p[2]
            x = int(p[0] / d)
            y = int(p[1] / d)
            if d <= 1e-9 or x < 0 or x >= w or y < 0 or y >= h:
                continue
            d_expected = frame1_depth[y][x]
            if d_expected <= 1e-9 or relative_depth_error(d, d_expected) > .1:
                continue
            b_pixels_in_a += 1

        num_pixels = (h // stride) * (w // stride)
        a_pixels_in_b /= num_pixels
        b_pixels_in_a /= num_pixels
        return (a_pixels_in_b + b_pixels_in_a) / 2


def run(opt, scanId):
    if scanId.startswith("."):
        return

    print("Computing overlaps for " + scanId)
        
    info_file = "{}/{}/_info.txt".format(opt.data_dir, scanId)
    overlap_file = "{}/{}/{}_overlap_scores.npy".format(opt.data_dir, scanId, scanId)
    if not opt.overwrite_overlap_files and os.path.exists(overlap_file):
        return
    if not os.path.exists(info_file):
        return

    scan = ScanProcessor(opt.data_dir, scanId, min_score=opt.min_score, frame_skip=opt.frame_skip,
            width=opt.width, height=opt.height, stride=opt.stride)
    scan.load_depth_and_poses()
    scan.compute_overlaps()
    scan.write_scores(overlap_file)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Compute overlap scores for Scannet pairs (the number of pixels of A visible in B and vice versa). '
                'Scores are written to <scanId>_overlap_scores.npy in each scan folder. '
                'Each row of this matrix is in the form [frame1idx, frame2idx, overlap_score].',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_dir', type=str,
        help='Path to the directory of scans')
    parser.add_argument(
        '--min_score', type=float, default=.3,
        help='Pairs with scores below this value will not be written')
    parser.add_argument(
        '--frame_skip', type=int, default=1,
        help='Skips every N frames in the sequence')
    parser.add_argument(
        '--width', type=int, default=64,
        help='Resizes depth image to this width')
    parser.add_argument(
        '--height', type=int, default=48,
        help='Resizes depth image to this height')
    parser.add_argument(
        '--stride', type=int, default=1,
        help='Skips every N pixels in both row and column directions when computing overlap')
    parser.add_argument(
        '--num_processes', type=int, default=psutil.cpu_count(logical=False),
        help='Number of scans to be processed simultaneously')
    parser.add_argument(
        '--overwrite_overlap_files', action='store_true',
        help='Overwrites existing <scanId>_overlap_scores.npy files')
    opt = parser.parse_args()

    start = time.time()

    with Pool(opt.num_processes) as p:
        dirs = os.listdir(opt.data_dir)
        p.starmap(run, zip([opt]*len(dirs), dirs))

    end = time.time()
    print("DONE: " + str(end - start) + " seconds")


        
