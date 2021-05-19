import numpy as np
import torch
import os
import cv2
import mmap
import random
import struct

from torch.utils.data import Dataset
from models.utils import SceneInfo

#random.seed(3)

# separating these objects so that the data loader doesnt attempt to copy them to other processes
class SceneFiles():

    def __init__(self, train_path, scan_path):
        self.scan_path = scan_path
        self.train_path = train_path

        self.scenefiles = {}
        for filename in os.listdir(train_path):
            if filename.startswith("scene") and len(filename) == 9:
                filepath = train_path + "/" + filename
                f = open(filepath, "rb")
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self.scenefiles[filename] = mm
        self.num_scenes = len(self.scenefiles)
        self.intrinsics = {}
        for scan_id in os.listdir(scan_path):
            info = SceneInfo("{}/{}/_info.txt".format(scan_path, scan_id))
            self.intrinsics[scan_id] = info.calibration_color_intrinsic.astype(np.float32)


class ScannetDataset(Dataset):

    ENTRY_SIZE_BYTES = 69
    SAMPLES_PER_SCENE = 200

    def __init__(self, scene_files):
        self.samples = []
        self.resample(scene_files)
        

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        [img_path1, img_path2, depth_path1, depth_path2, K, T_1to2] = self.samples[idx]
        img1 = torch.from_numpy(cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)) / 255
        img2 = torch.from_numpy(cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)) / 255
        depth1 = cv2.imread(depth_path1, -1) / np.float32(1000.0)
        depth2 = cv2.imread(depth_path2, -1) / np.float32(1000.0)

        return {
            'image0': img1.float()[None],
            'image1': img2.float()[None],
            'depth0': depth1,
            'depth1': depth2,
            'intrinsics': K,
            'T_0to1': T_1to2
        }
        

    def resample(self, scene_files):
        self.samples = []
        for scene_id, f in scene_files.scenefiles.items():
            for i in random.sample(range(f.size()//ScannetDataset.ENTRY_SIZE_BYTES), ScannetDataset.SAMPLES_PER_SCENE):
                byte_offset = i * ScannetDataset.ENTRY_SIZE_BYTES
                bytez = f[byte_offset:byte_offset + ScannetDataset.ENTRY_SIZE_BYTES]
                scan_id, frame1_idx, frame2_idx = struct.unpack("=BHH", bytez[:5])
                T_1to2 = np.frombuffer(bytez[5:], dtype="f").reshape((4,4))
                img_path1 = "{}/{}_{:02d}/frame-{:06d}.color.jpg".format(scene_files.scan_path, scene_id, scan_id, frame1_idx)
                img_path2 = "{}/{}_{:02d}/frame-{:06d}.color.jpg".format(scene_files.scan_path, scene_id, scan_id, frame2_idx)
                depth_path1 = "{}/{}_{:02d}/frame-{:06d}.depth.png".format(scene_files.scan_path, scene_id, scan_id, frame1_idx)
                depth_path2 = "{}/{}_{:02d}/frame-{:06d}.depth.png".format(scene_files.scan_path, scene_id, scan_id, frame2_idx)
                K = scene_files.intrinsics["{}_{:02d}".format(scene_id, scan_id)]
                self.samples.append([img_path1, img_path2, depth_path1, depth_path2, K, T_1to2])
        random.shuffle(self.samples)




        
