import os 
import sys 
from PIL import Image
from multiprocessing.pool import ThreadPool
import subprocess
import numpy as np
import time
import psutil
import argparse

import time

opt = None

class SceneInfo:

    def __init__(self, infofile):
        items = {}
        with open(infofile) as file:
            for line in file:
                k, v = [x.strip() for x in line.partition("=")[::2]]
                items[k] = v
    
        self.version_number = int(items["m_versionNumber"])
        self.sensor_name = items["m_sensorName"]
        self.color_width = int(items["m_colorWidth"])
        self.color_height = int(items["m_colorHeight"])
        self.depth_width = int(items["m_depthWidth"])
        self.depth_height = int(items["m_depthHeight"])
        self.depth_shift = int(items["m_depthShift"])
        self.calibration_color_intrinsic = np.fromstring(items["m_calibrationColorIntrinsic"], sep=" ").reshape((4,4))
        self.calibration_color_extrinsic = np.fromstring(items["m_calibrationColorExtrinsic"], sep=" ").reshape((4,4))
        self.calibration_depth_intrinsic = np.fromstring(items["m_calibrationDepthIntrinsic"], sep=" ").reshape((4,4))
        self.calibration_depth_extrinsic = np.fromstring(items["m_calibrationDepthExtrinsic"], sep=" ").reshape((4,4))
        self.num_frames = int(items["m_frames.size"])
    
    def write(self, outfile):
        with open(outfile, 'w') as file:
            file.write("m_versionNumber = " + str(self.version_number) + "\n")
            file.write("m_sensorName = " + self.sensor_name + "\n")
            file.write("m_colorWidth = " + str(self.color_width) + "\n")
            file.write("m_colorHeight = " + str(self.color_height) + "\n")
            file.write("m_depthWidth = " + str(self.depth_width) + "\n")
            file.write("m_depthHeight = " + str(self.depth_height) + "\n")
            file.write("m_depthShift = " + str(self.depth_shift) + "\n")
            file.write("m_calibrationColorIntrinsic = " + mat4_str(self.calibration_color_intrinsic) + "\n")
            file.write("m_calibrationColorExtrinsic = " + mat4_str(self.calibration_color_extrinsic) + "\n")
            file.write("m_calibrationDepthIntrinsic = " + mat4_str(self.calibration_depth_intrinsic) + "\n")
            file.write("m_calibrationDepthExtrinsic = " + mat4_str(self.calibration_depth_extrinsic) + "\n")
            file.write("m_frames.size = " + str(self.num_frames) + "\n")
    
    def resize_color(self, w, h):
        scale_x = w / self.color_width
        scale_y = h / self.color_height
        self.calibration_color_intrinsic = np.diag([scale_x, scale_y, 1, 1]) @ self.calibration_color_intrinsic
        self.color_width = w
        self.color_height = h
    
    def resize_depth(self, w, h):
        scale_x = w / self.depth_width
        scale_y = h / self.depth_height
        self.calibration_depth_intrinsic = np.diag([scale_x, scale_y, 1, 1]) @ self.calibration_depth_intrinsic
        self.depth_width = w
        self.depth_height = h


def mat4_str(mat4):
    s = ""
    for i in range(4):
        for j in range(4):
            s += str(mat4[i][j]) + " "
    return s

def mat3_str(mat3):
    s = ""
    for i in range(3):
        for j in range(3):
            s += str(mat3[i][j]) + " "
    return s


def convert_pgms(input_path):
    def convert_to_png(filename):
        if filename.startswith("."):
            return
        # convert .pgm depth frames to .png
        if filename.lower().endswith(".depth.pgm"):
            if opt.convert_pgm_to_png:
                filepath = input_path + "/" + filename
                img = Image.open(filepath)
                img.save(filepath[0:-4] + ".png", "png")
                os.remove(filepath)
        # reduce color image sizes to 640x480 
        elif filename.lower().endswith(".color.jpg"):
            if opt.resize_color_img:
                filepath = input_path + "/" + filename
                img = Image.open(filepath)
                w, h = img.size
                if w != opt.color_width or h != opt.color_height:
                    img = img.resize((opt.color_width, opt.color_height))
                    img.save(filepath)

    # todo - manage threads and processes better
    with ThreadPool(processes=opt.num_threads_png) as p:
        p.map(convert_to_png, os.listdir(input_path))
    
    # rewrite color intrinsic matrix
    if opt.resize_color_img:
        info_file = input_path + "/_info.txt"
        if os.path.exists(info_file):
            info = SceneInfo(info_file)
            info.resize_color(opt.color_width, opt.color_height)
            info.write(info_file)


def process_sens(scan_id):
    if scan_id.startswith("."):
        return
    print("processing " + scan_id)
    scanpath = opt.input_dir + "/" + scan_id
    sensfile = scanpath + "/" + scan_id + ".sens"
    outpath = opt.output_dir + "/" + scan_id
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # use C++ sens reader to extract sensor frames
    if os.path.exists(sensfile):
        subprocess.call([opt.sens_reader_path.replace("/", os.path.sep),
                sensfile.replace("/", os.path.sep), outpath.replace("/", os.path.sep)])

    # convert depth images to pngs, and resize color images
    convert_pgms(outpath)

    if opt.remove_sens_file and os.path.exists(sensfile):
        os.remove(sensfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Convenience script to extract the .sens files for each scan, '
                'convert depth images from pgm to png, '
                'and resize color images to 480x640.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'sens_reader_path', type=str,
        help='Path to the SensReader executable provided by ScanNet: '
                'https://github.com/ScanNet/ScanNet/tree/master/SensReader/c%2B%2B')
    parser.add_argument(
        'input_dir', type=str,
        help='Path to the directory of scans')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for extracted frames. Defaults to input_dir.')
    parser.add_argument(
        '--convert_pgm_to_png', action='store_true',
        help='Convert depth images from .pgm to .png')
    parser.add_argument(
        '--remove_sens_file', action='store_true',
        help='Delete the original .sens files after extracting')
    parser.add_argument(
        '--resize_color_img', action='store_true',
        help='Resize the color images after extracting them')
    parser.add_argument(
        '--color_width', type=int, default=640,
        help='Resize the color images to this width')
    parser.add_argument(
        '--color_height', type=int, default=480,
        help='Resize the color images to this height')
    parser.add_argument(
        '--num_threads_sens', type=int, default=psutil.cpu_count(logical=False),
        help='Number of scan extraction threads')
    parser.add_argument(
        '--num_threads_png', type=int, default=psutil.cpu_count(logical=False),
        help='Number of image conversion threads')
    opt = parser.parse_args()

    if opt.output_dir == None:
        opt.output_dir = opt.input_dir
        
    # todo - manage threads and processes better
    with ThreadPool(processes=opt.num_threads_sens) as p:
        p.map(process_sens, os.listdir(opt.input_dir))
