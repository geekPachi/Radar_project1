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




        #Code Explainization


# import os: Imports the os module, which provides functions for interacting with the operating system, such as file and directory operations.

# import re: Imports the re module, which provides support for regular expressions. It's used for pattern matching.

# import numpy as np: Imports the numpy library as np, which is a popular library for numerical computations in Python.

# import matplotlib.pyplot as plt: Imports the pyplot submodule of the matplotlib library, which is used for creating visualizations and plots.

# import argparse: Imports the argparse module, which is used for parsing command-line arguments in a structured way.

# from build_pointcloud import build_pointcloud: Imports the build_pointcloud function from a custom module named build_pointcloud. This function is used to create a point cloud from LIDAR data.

# from transform import build_se3_transform: Imports the build_se3_transform function from a custom module named transform. This function is used to build a 3D transformation matrix.

# from image import load_image: Imports the load_image function from a custom module named image. This function is used to load an image.

# from camera_model import CameraModel: Imports the CameraModel class from a custom module named camera_model. This class is used to handle camera models.

# parser = argparse.ArgumentParser(...): Creates an ArgumentParser object named parser for defining and parsing command-line arguments. It also provides a description of the script's purpose.

# parser.add_argument('--image_dir', type=str, help='Directory containing images'): Adds a command-line argument named --image_dir, which specifies the directory containing images. It expects a string value.

# parser.add_argument('--laser_dir', type=str, help='Directory containing LIDAR scans'): Adds a command-line argument named --laser_dir, which specifies the directory containing LIDAR scans. It expects a string value.

# parser.add_argument('--poses_file', type=str, help='File containing either INS or VO poses'): Adds a command-line argument named --poses_file, which specifies the file containing either INS or VO poses. It expects a string value.

# parser.add_argument('--models_dir', type=str, help='Directory containing camera models'): Adds a command-line argument named --models_dir, which specifies the directory containing camera models. It expects a string value.

# parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics'): Adds a command-line argument named --extrinsics_dir, which specifies the directory containing sensor extrinsics. It expects a string value.

# parser.add_argument('--image_idx', type=int, help='Index of image to display'): Adds a command-line argument named --image_idx, which specifies the index of the image to display. It expects an integer value.

# args = parser.parse_args(): Parses the command-line arguments and stores the values in the args object.

# The rest of the code uses the parsed command-line arguments to perform various operations, such as loading LIDAR data, camera models, and image data. It also creates a point cloud, projects it onto the image, and displays the result using Matplotlib's pyplot. The code involves working with 3D transformations, file paths, and visualization functions to project LIDAR data into a camera image.



import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
parser.add_argument('--image_dir', type=str, help='Directory containing images')
parser.add_argument('--laser_dir', type=str, help='Directory containing LIDAR scans')
parser.add_argument('--poses_file', type=str, help='File containing either INS or VO poses')
parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
parser.add_argument('--image_idx', type=int, help='Index of image to display')

args = parser.parse_args()

model = CameraModel(args.models_dir, args.image_dir)

extrinsics_path = os.path.join(args.extrinsics_dir, model.camera + '.txt')
with open(extrinsics_path) as extrinsics_file:
    extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

G_camera_vehicle = build_se3_transform(extrinsics)
G_camera_posesource = None

poses_type = re.search('(vo|ins|rtk)\.csv', args.poses_file).group(1)
if poses_type in ['ins', 'rtk']:
    with open(os.path.join(args.extrinsics_dir, 'ins.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
else:
    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle


timestamps_path = os.path.join(args.image_dir, os.pardir, model.camera + '.timestamps')
if not os.path.isfile(timestamps_path):
    timestamps_path = os.path.join(args.image_dir, os.pardir, os.pardir, model.camera + '.timestamps')

timestamp = 0
with open(timestamps_path) as timestamps_file:
    for i, line in enumerate(timestamps_file):
        if i == args.image_idx:
            timestamp = int(line.split(' ')[0])

pointcloud, reflectance = build_pointcloud(args.laser_dir, args.poses_file, args.extrinsics_dir,
                                           timestamp - 1e7, timestamp + 1e7, timestamp)

pointcloud = np.dot(G_camera_posesource, pointcloud)

image_path = os.path.join(args.image_dir, str(timestamp) + '.png')
print("Image path is ", image_path)
print("Timestamp is ", timestamp)
image = load_image(image_path, model)

uv, depth = model.project(pointcloud, image.shape)

plt.imshow(image)
plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
plt.xlim(0, image.shape[1])
plt.ylim(image.shape[0], 0)
plt.xticks([])
plt.yticks([])
plt.show()
