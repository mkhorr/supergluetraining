import os 
import sys 
from PIL import Image
from multiprocessing.pool import ThreadPool
import subprocess
import numpy as np
import time

# todo - make these commandline args
sens_reader_path = "~/ScanNet-master/SensReader/c++/x64/Release/sens.exe"
input_dir = "~/scannet/raw/scans"
output_dir = "~/tmp"
convert_pgm_to_png = True
remove_sens_file = False
resize_color_img = False
color_width = 640
color_height = 480

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


# for some reason, there are two .txt files in each scan directory with similar information
class SceneInformation:
    def __init__(self, infofile):
        items = {}
        with open(infofile) as file:
            for line in file:
                k, v = [x.strip() for x in line.partition("=")[::2]]
                items[k] = v

        self.axis_alignment = np.fromstring(items["axisAlignment"], sep=" ").reshape((4,4))
        self.color_height = int(items["colorHeight"])
        if "colorToDepthExtrinsics" in items:
            self.color_to_depth_extrinsics = np.fromstring(items["colorToDepthExtrinsics"], sep=" ").reshape((4,4))
        else:
            self.color_to_depth_extrinsics = None
        self.color_width = int(items["colorWidth"])
        self.depth_height = int(items["depthHeight"])
        self.depth_width = int(items["depthWidth"])
        self.fx_color = float(items["fx_color"])
        self.fx_depth = float(items["fx_depth"])
        self.fy_color = float(items["fy_color"])
        self.fy_depth = float(items["fy_depth"])
        self.mx_color = float(items["mx_color"])
        self.mx_depth = float(items["mx_depth"])
        self.my_color = float(items["my_color"])
        self.my_depth = float(items["my_depth"])
        self.num_color_frames = int(items["numColorFrames"])
        self.num_depth_frames = int(items["numDepthFrames"])
        self.num_IMU_measurements = int(items["numIMUmeasurements"])
        self.scene_type = items["sceneType"]

    
    def write(self, outfile):
        with open(outfile, 'w') as file:
            file.write("axisAlignment = " + mat4_str(self.axis_alignment) + "\n")
            file.write("colorHeight = " + str(self.color_height) + "\n")
            if self.color_to_depth_extrinsics is not None:
                file.write("colorToDepthExtrinsics = " + mat4_str(self.color_to_depth_extrinsics) + "\n")
            file.write("colorWidth = " + str(self.color_width) + "\n")
            file.write("depthHeight = " + str(self.depth_height) + "\n")
            file.write("depthWidth = " + str(self.depth_width) + "\n")
            file.write("fx_color = " + str(self.fx_color) + "\n")
            file.write("fx_depth = " + str(self.fx_depth) + "\n")
            file.write("fy_color = " + str(self.fy_color) + "\n")
            file.write("fy_depth = " + str(self.fy_depth) + "\n")
            file.write("mx_color = " + str(self.mx_color) + "\n")
            file.write("mx_depth = " + str(self.mx_depth) + "\n")
            file.write("my_color = " + str(self.my_color) + "\n")
            file.write("my_depth = " + str(self.my_depth) + "\n")
            file.write("numColorFrames = " + str(self.num_color_frames) + "\n")
            file.write("numDepthFrames = " + str(self.num_depth_frames) + "\n")
            file.write("numIMUmeasurements = " + str(self.num_IMU_measurements) + "\n")
            file.write("sceneType = " + self.scene_type + "\n")
    
    def resize_color(self, w, h):
        scale_x = w / self.color_width
        scale_y = h / self.color_height
        self.color_width = w
        self.color_height = h
        self.fx_color *= scale_x
        self.fy_color *= scale_y
        self.mx_color *= scale_x
        self.my_color *= scale_y


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
        # convert .pgm depth frames to .png
        if filename.lower().endswith(".depth.pgm"):
            if convert_pgm_to_png:
                filepath = input_path + "/" + filename
                img = Image.open(filepath)
                img.save(filepath[0:-4] + ".png", "png")
                os.remove(filepath)
        # reduce color image sizes to 640x480 
        elif filename.lower().endswith(".color.jpg"):
            if resize_color_img:
                filepath = input_path + "/" + filename
                img = Image.open(filepath)
                w, h = img.size
                if w != color_width or h != color_height:
                    img = img.resize((color_width, color_height))
                    img.save(filepath)

    # todo - manage threads and processes better
    with ThreadPool(processes=12) as p:
        p.map(convert_to_png, os.listdir(input_path))
    
    # rewrite color intrinsic matrix
    if resize_color_img:
        info_file = input_path + "/_info.txt"
        if os.path.exists(info_file):
            info = SceneInfo(info_file)
            info.resize_color(color_width, color_height)
            info.write(info_file)

        # note - not using this txt file, so ignore it for now
        #other_info_file = input_path + "/" + os.path.basename(input_path) + ".txt"
        #if os.path.exists(other_info_file):
        #    info = SceneInformation(other_info_file)
        #    info.resize_color(color_width, color_height)
        #    info.write(other_info_file)


def process_sens(scan_id):
    print("processing " + scan_id)
    scanpath = input_dir + "/" + scan_id
    sensfile = scanpath + "/" + scan_id + ".sens"
    outpath = output_dir + "/" + scan_id
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # use C++ sens reader to extract sensor frames
    if os.path.exists(sensfile):
        subprocess.call([sens_reader_path.replace("/", os.path.sep),
                sensfile.replace("/", os.path.sep), outpath.replace("/", os.path.sep)])

    # convert depth images to pngs, and resize color images
    convert_pgms(outpath)

    if remove_sens_file and os.path.exists(sensfile):
        os.remove(sensfile)



if __name__ == '__main__':
    return #todo
    start = time.time()

    # todo - manage threads and processes better
    with ThreadPool(processes=12) as p:
        p.map(process_sens, os.listdir(input_dir))
