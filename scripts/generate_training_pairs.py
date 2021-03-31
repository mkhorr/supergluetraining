import os
import numpy as np
from shutil import copyfile
import random
import struct
import time
import mmap


# converting score files into sampleable binary files of training pairs with relative poses

input_dir = "~/scannet/raw/scans"
output_dir = "~/training_pairs"

for scene_id in range(707):

    scene_outfile = open("{}/scene0{:03d}".format(output_dir, scene_id), "wb")

    for scene_sub_id in range(100):
        scan_id = "scene0{:03d}_{:02d}".format(scene_id, scene_sub_id)
        scan_path = "{}/{}".format(input_dir, scan_id)
        if not os.path.exists(scan_path):
            break

        # loading poses
        poses = {}
        for f in os.listdir(scan_path):
            if f.endswith(".pose.txt"):
                frame_idx = int(f[6:12])
                poses[frame_idx] = np.loadtxt("{}/{}".format(scan_path, f))

        # filtering and writing scored pairs
        scorefile = "{}/{}_overlap_scores.npy".format(scan_path, scan_id)
        scores = np.load(scorefile)
        for row in scores:
            if row[2] >= .4 and row[2] <= .8:
                frame1 = round(row[0])
                frame2 = round(row[1])
                M = np.linalg.inv(poses[frame2]) @ poses[frame1]
                bytez = struct.pack("=BHH", scene_sub_id, frame1, frame2) + M.astype('f').tobytes()
                scene_outfile.write(bytez)

    scene_outfile.close()
    print("Done proccessing scene", scene_id)
