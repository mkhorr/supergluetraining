
Download the following three scans to a directory (for example ~/scannet):

http://kaldir.vc.in.tum.de/scannet/v1/scans/scene0000_00/scene0000_00.sens
http://kaldir.vc.in.tum.de/scannet/v1/scans/scene0000_01/scene0000_01.sens
http://kaldir.vc.in.tum.de/scannet/v1/scans/scene0000_02/scene0000_02.sens

change the following variables to point to this directory:

scripts/extract_sens.py
	input_dir = "~/scannet"
	output_dir = "~/scannet"
	sens_reader_path = ***

scripts/compute_overlap_scores.py
	data_dir = "~/scannet"

scripts/generate_training_pairs.py
	input_dir = "~/scannet"
	output_dir = "~/training_pairs"

train.py
	scene_files = SceneFiles("~/training_pairs", "~/scannet")

*** you need to point this variable to ScanNet's sens reader executable which you can compile from source here:

https://github.com/ScanNet/ScanNet/tree/master/SensReader/c%2B%2B




Then, run the following:

./extract_sens.py

This will extract the frame data (color image, depth image, pose) from the .sens file. It will also resize the color images, convert the depth images from .pgm to .png, and delete the .sens file to save space (you can turn these off by setting the variables at the top of the file)

./compute_overlap_scores.py

This will compute the overlap scores for all the pairs of images in each scan, and will save the results in <scan_path>/<scan_id>_overlap_scores.npy

./generate_training_pairs.py

This will read all the .npy files, filter for entries with scores between [.4, .8], compute the relative pose for the image pair, combine the entries for scans that come from the same scene, and save the results to a binary file per scene. Each entry is a fixed number of bytes so it's easy to randomly sample from the file (since it won't fit into memory).

./train.py

This will run the training. Right now, I'm just saving the state_dict every epoch (in ./dump_weights/).




Evaluating the model:

To evaluate the model, it's probably easiest to run the superglue demo in "evaluation mode" (see https://github.com/magicleap/SuperGluePretrainedNetwork for more details):

./match_pairs.py --eval --input_pairs <input.txt>

The input is a txt file where each line is in the followng format:

path_image_A path_image_B exif_rotationA exif_rotationB [KA_0 ... KA_8] [KB_0 ... KB_8] [T_AB_0 ... T_AB_15]

You can look at assets/scannet_sample_pairs_with_gt.txt as an example which is the default file if you don't specify --input_pairs. You can generate a file like this for validation by reading and removing entries from the binary training files. Look at ScannetDataset.resample in load_scannet.py for example code that reads from the binary files. exif_rotationA and exif_rotationB can be set to 0, KA == KB since the pairs come from the same camera, and I think I saved K as a 4x4 matrix, so you would need to do K[:3,:3]. And obviously, you would have to load the state_dict that you want to test when you run the evaluation.




