################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Dan Barnes (dbarnes@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################


        # Explaination of code


# This Python script appears to be a radar data visualization tool. It loads radar data from a specified directory, processes it, and displays it as a combined polar and Cartesian representation. Let's break down the script step by step:

# Import Statements:

# argparse is imported to parse command-line arguments.
# os is imported for operating system-related functions.
# radar functions are imported from a module (not provided in the script), which is likely responsible for loading and processing radar data.
# numpy is imported for numerical operations.
# cv2 (OpenCV) is imported for computer vision and image processing.
# Argument Parsing:

# The script starts by defining a command-line interface using the argparse.ArgumentParser class.
# It adds a single required argument, 'dir', representing the directory containing radar data.
# The argparse.parse_args() method is used to parse the provided command-line arguments. The parsed arguments are stored in the args object.
# Timestamps File Path:

# A hardcoded path to a timestamps file is assigned to the variable timestamps_path.
# This path appears to point to a specific file on the Windows file system, but it could be modified to use the args.dir argument.
# Timestamps File Check:

# The script checks if the timestamps file specified by timestamps_path exists using os.path.isfile().
# If the file doesn't exist, it raises an IOError indicating that the timestamps file could not be found.
# Cartesian Visualization Setup:

# The script defines several variables related to Cartesian visualization:
# cart_resolution: Resolution of the Cartesian form of the radar scan in meters per pixel.
# cart_pixel_width: Cartesian visualization size in pixels (for both height and width).
# interpolate_crossover: A boolean flag indicating whether to interpolate crossover regions.
# title: A title for the radar visualization example.
# Processing Radar Data:

# The script reads radar timestamps from the timestamps file using np.loadtxt(). The timestamps are stored in the radar_timestamps array.
# It then enters a loop to process each radar timestamp.
# Inside the loop:
# It constructs the filename of the radar data file based on the current timestamp.
# It checks if the radar data file exists using os.path.isfile(). If the file doesn't exist, it raises a FileNotFoundError.
# It loads radar data using functions from the radar module (not provided in the script).
# It converts polar radar data to Cartesian coordinates using the radar_polar_to_cartesian function. This results in cart_img, the Cartesian representation of radar data.
# It combines the polar and Cartesian representations for visualization, resizing the polar data to match the height of the Cartesian image.
# Finally, it displays the combined visualization using OpenCV's cv2.imshow(), waits for a key press (using cv2.waitKey()), and repeats the process for the next timestamp.
# This script is designed to visualize radar data in both polar and Cartesian formats, allowing users to inspect radar scans and their corresponding Cartesian representations. It can be a valuable tool for radar data analysis and debugging.










import argparse
# '''argparse module is a built-in Python library that allows you to parse 
# command-line arguments in a structured and user-friendly way.''' 


import os
# '''The os module, short for "operating system," provides a way to interact 
# with the underlying operating system on which your Python code is running.''' 
import re

from radar import load_radar, radar_polar_to_cartesian

'''used to convert radar data from polar coordinates to Cartesian coordinates '''


import numpy as np

import cv2

# ''' OpenCV, which stands for "Open Source Computer Vision Library," is a 
# popular and powerful open-source computer vision and image processing library.'''



parser = argparse.ArgumentParser(description='Play back radar data from a given directory')

parser.add_argument('dir', type=str, help='Directory containing radar data.')

args = parser.parse_args()




# '''The ArgumentParser class is provided by the argparse module and 
# is used to define and manage the command-line interface of your Python script.

# This line adds a command-line argument to your script using the add_argument method of the ArgumentParser object (parser).

# After defining the argument(s) with add_argument, you use parser.parse_args() to parse the command-line arguments provided by the user.'''






#timestamps_path = os.path.join(os.path.join(args.dir, 'radar.timestamps'))
timestamps_path = "C:/Radar_project/Radar_Datasets/large_dataset/2019-01-10-14-36-48-radar-oxford-10k-partial/radar.timestamps"
print("timestamps_path is ", timestamps_path)
if not os.path.isfile(timestamps_path):
    raise IOError("Could not find timestamps file")
    
    
''' you are assigning a file path to the variable timestamps_path. 
This path is hardcoded and points to a specific file on the Windows file system.'''


    
    



# Cartesian Visualsation Setup
# Resolution of the cartesian form of the radar scan in metres per pixel
cart_resolution = .25
# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 501  # pixels
interpolate_crossover = True

title = "Radar Visualisation Example"

radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
for radar_timestamp in radar_timestamps:
    filename = os.path.join(args.dir, str(radar_timestamp) + '.png')
    filename = filename.replace("\\","/")
    filename = filename[4:]
    print("File name is ", filename)
    #img = cv2.imread(filename)
    #cv2.imshow("Image",img)
    #filename = filename.replace("//","//")
    #print("File name is ", filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError("Could not find radar example: {}".format(filename))

    timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(filename)
    cart_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                        interpolate_crossover)

    # Combine polar and cartesian for visualisation
    # The raw polar data is resized to the height of the cartesian representation
    downsample_rate = 4
    fft_data_vis = fft_data[:, ::downsample_rate]
    resize_factor = float(cart_img.shape[0]) / float(fft_data_vis.shape[0])
    fft_data_vis = cv2.resize(fft_data_vis, (0, 0), None, resize_factor, resize_factor)
    vis = cv2.hconcat((fft_data_vis, fft_data_vis[:, :10] * 0 + 1, cart_img))

    print("Type of vis ",type(vis))
    cv2.imshow(title, vis * 3.)  # The data is doubled to improve visualisation
    cv2.waitKey(100)
