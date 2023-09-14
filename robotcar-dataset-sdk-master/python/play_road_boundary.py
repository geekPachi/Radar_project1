################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Daniele De Martini (daniele@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################



        # Code Explaination 

# The script you provided is a Python program for processing and visualizing images and masks from a specified directory. It uses the argparse module for command-line argument parsing. Let's break down the script step by step:

# Import Statements:

# argparse: Imported for parsing command-line arguments.
# cv2 (OpenCV): Used for image processing and displaying images.
# pathlib.Path: Used to work with file paths and directories in a more Pythonic way.
# tqdm: A library for adding progress bars to loops.
# numpy (as np): Imported for numerical operations.
# datetime (as dt): Imported to work with date and time.
# Functions from an external module road_boundary are imported for loading road boundary images and masks.
# Argument Parsing:

# The script starts by defining a command-line interface using the argparse.ArgumentParser class.
# It adds several command-line arguments, including the directory containing images ('trial'), camera ID ('camera_id'), data type ('type'), mask type ('masks_id'), a flag for saving a video ('save_video'), and the directory for saving images ('save_dir').
# The argparse.parse_args() method is used to parse the provided command-line arguments. The parsed arguments are stored in the args object.
# File Paths and Assertions:

# File paths for images and masks are constructed based on the provided arguments (args.trial, args.type, args.camera_id, and args.masks_id) using the pathlib.Path class.
# Assertions are used to check if the specified image and mask paths exist. If they don't, assertion errors are raised.
# Image and Mask Processing Loop:

# The script enters a loop that iterates over pairs of image and mask files, which are sorted based on their filenames.
# For each iteration:
# Images and masks are loaded using functions from the road_boundary module (not provided in the script).
# The mask is dilated using a 5x5 kernel to enhance its visibility.
# Depending on the args.masks_id argument, different color overlays are applied to the image. If args.masks_id is 'mask', yellow is used; otherwise, cyan and red are used for specific mask values.
# Images are converted from RGB to BGR color format for OpenCV compatibility.
# The resulting image is displayed using cv2.imshow(). The display window is updated continuously, and the loop breaks if the 'q' key is pressed.
# If the args.save_video flag is set and video initialization is not done (initialised is False), a video writer is created with the specified parameters (e.g., MPEG codec, frames per second, framesize), and a video frame is saved as a JPEG image.
# Subsequent frames are written to the video file using the video writer.
# Video Release:

# After the loop ends, the video writer is released using out.release(). This finalizes the video file.
# Overall, this script is used to visualize images and overlay masks, and it can optionally save the resulting frames as a video. The user can specify various options and paths through command-line arguments. The processing logic involving the 'road_boundary' module is not provided in the script, and it relies on external functions to load and process images and masks.


import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np
from pathlib import WindowsPath

from datetime import datetime as dt
from road_boundary import load_road_boundary_image, load_road_boundary_mask


YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
RED = (255, 0, 0)


parser = argparse.ArgumentParser(description='Play back images from a given directory')

parser.add_argument('trial', type=str, help='Directory containing images.')
parser.add_argument('--camera_id', type=str, default='left', choices=['left', 'right'], help='(optional) Camera ID to display')
parser.add_argument('--type', type=str, default='uncurated', choices=['uncurated', 'curated'], help='(optional) Curated vs uncurated')
parser.add_argument('--masks_id', type=str, default='mask', choices=['mask', 'mask_classified'], help='(optional) Masks type to overlay')

parser.add_argument('--save_video',  action='store_true', help='Flag for saving a video')
parser.add_argument('--save_dir',  type=str, help='Where to save the images')

args = parser.parse_args()

path_str = str(args.trial)
trail_path_str = path_str[6:]

image_path = Path(trail_path_str) / args.type / args.camera_id / 'rgb'
#path_str = str(args.trial)
#new_path_str = path_str[6:]
#image_path = WindowsPath(new_path_str)
#image_path = image_path[4:]
mask_path = Path(trail_path_str) / args.type / args.camera_id / args.masks_id
print("Mask path is ", mask_path)
#mask_path = WindowsPath(new_path_str)

assert image_path.exists(), f'Image path {image_path} does not exist'
assert mask_path.exists(), f'Mask path {mask_path} does not exist'

images = sorted(image_path.glob('*.png'))
masks = sorted(mask_path.glob('*.png'))

fname = f"{trail_path_str}_{args.camera_id}_{args.type}_{args.masks_id}"

initialised = False
for image, mask in tqdm(zip(images, masks)):
    image = load_road_boundary_image(str(image))
    mask = load_road_boundary_image(str(mask))

    kernel = np.ones((5, 5), 'uint8')
    mask = cv2.dilate(mask, kernel, iterations=1)

    if args.masks_id == 'mask':
        image[mask > 0] = YELLOW
    else:
        image[mask == 1] = CYAN
        image[mask == 2] = RED

    image_ = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Video', image_)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if args.save_video:
        if not initialised:
            framesize = (image.shape[1], image.shape[0])
            out = cv2.VideoWriter(str(Path(args.save_dir) / f'{fname}.avi'), \
                                  cv2.VideoWriter_fourcc(*'MPEG'),
                                  20, framesize, True)
            initialised = True

            cv2.imwrite(str(Path(args.save_dir) / f'{fname}.jpg'), image_)

        out.write(image_)

out.release()


# This code processes a sequence of images and, if specified, saves them as a video. Let's break it down step by step:

# initialised = False: This variable is used to keep track of whether the video writer (out) has been initialized. It's initially set to False.

# for image, mask in tqdm(zip(images, masks)):

# This loop iterates through pairs of images and masks obtained from the images and masks lists. zip(images, masks) combines the two lists element-wise, and tqdm adds a progress bar to the loop for visualization.
# image = load_road_boundary_image(str(image)) and mask = load_road_boundary_image(str(mask)): These lines load image and mask files using the load_road_boundary_image function. The str(image) and str(mask) convert the Path objects to strings, which are required by the loading function.

# kernel = np.ones((5, 5), 'uint8'): This line defines a 5x5 square-shaped kernel used for dilation. It's often used for morphological operations on binary images.

# mask = cv2.dilate(mask, kernel, iterations=1): Here, the mask image is dilated using the specified kernel. This can help in filling gaps and making the mask more robust for further processing.

# if args.masks_id == 'mask':

# This conditional block checks the value of args.masks_id, which determines how the mask should be applied to the image.
# If args.masks_id is equal to 'mask', it means you want to overlay the mask on the image using a yellow color.
# If it's not equal to 'mask', it means you want to overlay the mask using cyan for class 1 and red for class 2.
# image_ = cv2.cvtColor(image, cv2.COLOR_RGB2BGR): This line converts the processed image from the RGB color space to the BGR color space. OpenCV often works with images in BGR format.

# cv2.imshow('Video', image_): This displays the processed image with the window title "Video." It updates the window with each frame.

# if cv2.waitKey(1) & 0xFF == ord('q'):

# This line waits for a key event and checks if the key pressed is 'q' (quit). If 'q' is pressed, the loop breaks, and the program continues.
# if args.save_video:

# This conditional block checks if the args.save_video flag is set to True, which means you want to save the processed frames as a video.
# if not initialised:
# This block checks if the initialised flag is still False, indicating that the video writer (out) hasn't been initialized yet.
# Video Writer Initialization:

# framesize = (image.shape[1], image.shape[0]): This line determines the size (width and height) of the video frames based on the shape of the processed image.
# out = cv2.VideoWriter(str(Path(args.save_dir) / f'{fname}.avi'), cv2.VideoWriter_fourcc(*'MPEG'), 20, framesize, True): Here, a video writer (out) is initialized with the following parameters:
# Output file path: It's based on the args.save_dir and fname to create a filename for the video.
# FourCC codec: 'MPEG' specifies the codec to be used for video encoding.
# Frames per second (FPS): Set to 20 FPS.
# Framesize: The width and height of video frames.
# True specifies that the video writer should write in color (as opposed to grayscale).
# out.write(image_): If the video writer has been initialized, this line writes the processed image frame to the video.

# out.release(): Once all frames have been processed and written to the video, the video writer is released, finalizing the video file.