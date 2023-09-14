################################################################################
#
#  Copyright (c) 2017 University of Oxford
#  Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################


        # Code Explaination
  
# This Python script is a command-line tool for playing back images from a specified directory. It provides options for image scaling and undistortion using camera models. Let's break down the script step by step:

# Argument Parsing:

# The script starts by importing the argparse library to parse command-line arguments.
# argparse.ArgumentParser is used to define the command-line interface of the script.
# Three arguments are defined:
# 'dir': A required argument representing the directory containing the images.
# '--models_dir': An optional argument for specifying the directory containing camera models. If provided, the images will be undistorted using these models before display.
# '--scale': An optional argument for specifying a scaling factor to resize the images before display (default is 1.0).
# argparse.parse_args() parses the command-line arguments and stores them in the args object.
# Camera Type Detection:

# The script uses regular expressions (re) to determine the camera type based on the provided image directory. It searches for patterns like "stereo" or "mono_(left|right|rear)" in the directory name.
# The detected camera type is stored in the camera variable.
# Timestamps File Path:

# The script constructs the path to the timestamps file based on the detected camera type (camera). It checks two possible paths for the timestamps file and raises an IOError if neither path leads to an existing file.
# Camera Model Loading:

# If the --models_dir argument is provided, the script attempts to load a camera model using the CameraModel class. This model will be used for undistorting the images.
# Image Playback Loop:

# The script enters a loop that reads the timestamps file line by line.
# For each line, it splits the line into tokens, where the first token represents the timestamp in microseconds, and the second token represents the chunk number.
# It converts the timestamp to a datetime object (datetime.utcfromtimestamp) and stores it in the datetime variable.
# It constructs the filename of the image based on the timestamp by appending '.png' to the timestamp and joining it with the image directory path.
# If the image file does not exist, it prints a message indicating that the chunk is not found and continues to the next line in the timestamps file.
# If the image file exists, it:
# Updates the current_chunk variable with the current chunk number.
# Loads the image using the load_image function, optionally undistorting it using the camera model.
# Displays the image using plt.imshow.
# Adds an x-axis label (plt.xlabel) indicating the datetime.
# Removes x and y ticks (plt.xticks([]) and plt.yticks([])).
# Pauses for a short duration (0.01 seconds) to display the image.
# This script is designed for visualizing a sequence of images from a specified directory. It can optionally apply undistortion if camera models are available and resize images based on the provided scale factor. The script can be useful for inspecting image datasets or recorded image sequences.



import argparse
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime as dt
from image import load_image
from camera_model import CameraModel

parser = argparse.ArgumentParser(description='Play back images from a given directory')

parser.add_argument('dir', type=str, help='Directory containing images.')
parser.add_argument('--models_dir', type=str, default=None, help='(optional) Directory containing camera model. If supplied, images will be undistorted before display')
parser.add_argument('--scale', type=float, default=1.0, help='(optional) factor by which to scale images before display')

args = parser.parse_args()

camera = re.search('(stereo|mono_(left|right|rear))', args.dir).group(0)

#timestamps_path = os.path.join(os.path.join(args.dir, os.pardir, camera + '.timestamps'))
timestamps_path = "C:/Radar_project/Radar_Datasets/large_dataset/2019-01-10-14-36-48-radar-oxford-10k-partial/mono_left.timestamps"
if not os.path.isfile(timestamps_path):
  timestamps_path = os.path.join(args.dir, os.pardir, os.pardir, camera + '.timestamps')
  if not os.path.isfile(timestamps_path):
      raise IOError("Could not find timestamps file")

model = None
if args.models_dir:
    model = CameraModel(args.models_dir, args.dir)

current_chunk = 0
timestamps_file = open(timestamps_path)
for line in timestamps_file:
    tokens = line.split()
    datetime = dt.utcfromtimestamp(int(tokens[0])/1000000)
    chunk = int(tokens[1])

    filename = os.path.join(args.dir, tokens[0] + '.png')
    filename = filename.replace("\\","/")
    filename = filename[4:]
    # print("Filename is ", filename)
    # print("current_chunk is ",current_chunk)
    if not os.path.isfile(filename):
        if chunk != current_chunk:
            print("Chunk " + str(chunk) + " not found")
            current_chunk = chunk
        continue

    current_chunk = chunk

    img = load_image(filename, model)
    plt.imshow(img)
    plt.xlabel(datetime)
    plt.xticks([])
    plt.yticks([])
    plt.pause(0.001)
