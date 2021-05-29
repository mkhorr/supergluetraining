# SuperGlue Indoor Training Code

This repository attempts to reproduce the training code for the indoor pose estimation experiment for the [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) network.

## Preprocessing

Download the [ScanNet](https://github.com/ScanNet/ScanNet) sensor data after requesting access. The preprocessing scripts assume the following directory structure:

```
scans
│
└───<scanId_1>
│   │   <scanId_1>.sens
│   │   ...
│   
└───<scanId_2>
│   │   <scanId_2>.sens
│   │   ...
...
```


To extract the frame data (color images, depth images, poses) from the .sens files, build the [SensReader](https://github.com/ScanNet/ScanNet/tree/master/SensReader/c%2B%2B) provided by ScanNet. Run the following convenience script to extract the .sens files in parallel, resize the color images to 640x480, convert the depth images from .pgm to .png, and optionally delete the .sens file to save space. On Windows, remove the call to getchar() in main.cpp before building the SensReader in order to prevent halting for user input.


```bash
python extract_sens.py <path_to_sens_reader_executable> <path_to_scans> --convert_pgm_to_png --resize_color_img
```

In the indoor experiment, SuperGlue is trained on image pairs with an overlap score between [0.4, 0.8]. Run the following script to compute these overlap scores for each scan. The results will be saved to <scan_path>/<scan_id>_overlap_scores.npy where each entry has the form [frame1_idx, frame2_idx, overlap_score]. Since this script can be time consuming, --width, --height, and --stride can be used to reduce the number of pixels used in the calculation, but this can change the training set slightly. 


```bash
python compute_overlap_scores.py <path_to_extracted_scans>
```

The following script will generate training pairs from the .npy files. It will filter for entries with scores between [.4, .8], compute the relative pose for the image pair, combine the entries for scans that come from the same scene, and save the results to a binary file per scene. Each entry is a fixed number of bytes to simplify randomly sampling from the file.


```bash
python generate_training_pairs.py <path_to_extracted_scans> <path_to_training_pairs>
```



## Training

To train the model, run the following. Use -h for more info.

```bash
python train.py <path_to_extracted_scans> <path_to_training_pairs>
```

## Evaluation

To evaluate the model, run the following:

```bash
match_pairs.py --eval --input_pairs <input.txt> --weights <path_to_weights>
```

The input is a txt file where each line is in the following format:

```
path_image_A path_image_B exif_rotationA exif_rotationB [KA_0 ... KA_8] [KB_0 ... KB_8] [T_AB_0 ... T_AB_15]
```

If you want to convert the binary training pairs to this format, there are some functions in utils.py
