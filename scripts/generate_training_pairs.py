import os
import numpy as np
import struct
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert overlap_scores.npy files into sampleable binary files of training pairs with relative poses.\n'
                'Each entry has the following byte format: \n'
                '1 byte unsigned int - scene subId (ie. the Nth scan in a scene)\n'
                '2 byte unsigned int - index of first frame\n'
                '2 byte unsigned int - index of second frame\n'
                '64 byte 4x4 float matrix - relative pose from frame 1 to frame 2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'input_dir', type=str,
        help='Path to the directory of scans')
    parser.add_argument(
        'output_dir', type=str,
        help='Output directory for training pair files')
    parser.add_argument(
        '--min_score', type=float, default=.4,
        help='Include training pairs with overlap scores above this value')
    parser.add_argument(
        '--max_score', type=float, default=.8,
        help='Include training pairs with overlap scores below this value')
    opt = parser.parse_args()

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    for scene_id in range(707):

        outfile_path = "{}/scene0{:03d}".format(opt.output_dir, scene_id)
        scene_outfile = None

        first_scan = True
        for scene_sub_id in range(100):
            scan_id = "scene0{:03d}_{:02d}".format(scene_id, scene_sub_id)
            scan_path = "{}/{}".format(opt.input_dir, scan_id)
            if not os.path.exists(scan_path):
                break
            if first_scan:
                first_scan = False
                scene_outfile = open(outfile_path, "wb")

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
                if row[2] >= opt.min_score and row[2] <= opt.max_score:
                    frame1 = round(row[0])
                    frame2 = round(row[1])
                    M = np.linalg.inv(poses[frame2]) @ poses[frame1]
                    bytez = struct.pack("=BHH", scene_sub_id, frame1, frame2) + M.astype('f').tobytes()
                    scene_outfile.write(bytez)

        if scene_outfile is not None:
            scene_outfile.close()
            print("Done proccessing scene", scene_id)
