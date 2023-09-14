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


# Here's an explanation of each line in the provided code:

# import argparse: Imports the argparse module, which is used for parsing command-line arguments.

# from argparse import RawTextHelpFormatter: Imports the RawTextHelpFormatter class from the argparse module. This class is used to format the help message for command-line arguments with multi-line descriptions.

# import os: Imports the os module, which provides functions for interacting with the operating system, such as file and directory operations.

# from velodyne import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud: Imports functions from the velodyne module. These functions are used to load Velodyne LiDAR data and process it.

# import numpy as np: Imports the numpy library as np, which is a popular library for numerical computations in Python.

# import cv2: Imports the OpenCV library, which is used for computer vision and image processing tasks.

# from matplotlib.cm import get_cmap: Imports the get_cmap function from the matplotlib.cm module. This function is used to retrieve a color map for visualizations.

# from scipy import interpolate: Imports the interpolate module from SciPy, which provides functions for interpolation and curve fitting.

# import open3d: Imports the open3d library, which is used for 3D data visualization and processing.

# from transform import build_se3_transform: Imports the build_se3_transform function from a custom module called transform. This function is used to build a 3D transformation matrix.

# mode_flag_help = """..."""": Defines a multi-line string mode_flag_help that provides help text explaining the available modes for the --mode command-line argument.

# parser = argparse.ArgumentParser(...): Creates an ArgumentParser object named parser for defining and parsing command-line arguments. The RawTextHelpFormatter is used as the formatter for the help message.

# parser.add_argument('--mode', default="raw_interp", type=str, help=mode_flag_help): Adds a --mode command-line argument to the parser. It accepts a string value, has a default value of "raw_interp," and includes the help message from the mode_flag_help variable.

# parser.add_argument('--scale', default=2., type=float, help="Scale visualisations by this amount"): Adds a --scale command-line argument to the parser. It accepts a float value, has a default value of 2.0, and includes a brief help message.

# parser.add_argument('dir', type=str, help='Directory containing velodyne data.'): Adds a positional argument named dir to the parser. It accepts a string value and represents the directory containing Velodyne data.

# args = parser.parse_args(): Parses the command-line arguments and stores the values in the args object.

# The rest of the code defines a main function that processes Velodyne LiDAR data based on the parsed command-line arguments and visualization mode (--mode). The code performs various operations, including loading Velodyne data, visualizing it using Open3D or OpenCV, and handling different modes based on the user's choice.

# The if __name__ == "__main__": block ensures that the main function is executed when the script is run as the main program.



import argparse
from argparse import RawTextHelpFormatter
import os
from velodyne import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud
import numpy as np
import cv2
from matplotlib.cm import get_cmap
from scipy import interpolate
import open3d
from transform import build_se3_transform

mode_flag_help = """Mode to run in, one of: (raw|raw_interp|raw_ptcld|bin_ptcld)
- 'raw' - visualise the raw velodyne data (intensities and ranges)
- 'raw_interp' - visualise the raw velodyne data (intensities and ranges) interpolated to consistent azimuth angles
                 between scans.
- 'raw_ptcld' - visualise the raw velodyne data converted to a pointcloud (converts files of the form
                <timestamp>.png to pointcloud)
- 'bin_ptcld' - visualise the precomputed velodyne pointclouds (files of the form <timestamp>.bin). This is
                approximately 2x faster than running the conversion from raw data `raw_ptcld` at the cost of
                approximately 8x the storage space.
"""

parser = argparse.ArgumentParser(description='Play back velodyne data from a given directory',
                                 formatter_class=RawTextHelpFormatter)
parser.add_argument('--mode', default="raw_interp", type=str, help=mode_flag_help)
parser.add_argument('--scale', default=2., type=float, help="Scale visualisations by this amount")
parser.add_argument('dir', type=str, help='Directory containing velodyne data.')

args = parser.parse_args()


def main():
    velodyne_dir = args.dir
    if velodyne_dir[-1] == '/':
        velodyne_dir = velodyne_dir[:-1]

    velodyne_sensor = os.path.basename(velodyne_dir)
    if velodyne_sensor not in ["velodyne_left", "velodyne_right"]:
        raise ValueError("Velodyne directory not valid: {}".format(velodyne_dir))

    timestamps_path = "C:/Radar_project/Radar_Datasets/large_dataset/2019-01-10-14-36-48-radar-oxford-10k-partial/velodyne_left.timestamps"
    if not os.path.isfile(timestamps_path):
        raise IOError("Could not find timestamps file: {}".format(timestamps_path))

    title = "Velodyne Visualisation Example"
    extension = ".bin" if args.mode == "bin_ptcld" else ".png"
    velodyne_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
    #velodyne_timestamps = velodyne_timestamps.replace("\\","/")
    #velodyne_timestamps = velodyne_timestamps[4:]


    colourmap = (get_cmap("viridis")(np.linspace(0, 1, 255))[:, :3] * 255).astype(np.uint8)[:, ::-1]
    interp_angles = np.mod(np.linspace(np.pi, 3 * np.pi, 720), 2 * np.pi)
    vis = None

    for velodyne_timestamp in velodyne_timestamps:

        filename = os.path.join(args.dir, str(velodyne_timestamp) + extension)
        filename = filename.replace("\\","/")
        filename = filename[4:]

        if args.mode == "bin_ptcld":
            ptcld = load_velodyne_binary(filename)
        else:
            ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(filename)

            if args.mode == "raw_ptcld":
                ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)
            elif args.mode == "raw_interp":
                intensities = interpolate.interp1d(angles[0], intensities, bounds_error=False)(interp_angles)
                ranges = interpolate.interp1d(angles[0], ranges, bounds_error=False)(interp_angles)
                intensities[np.isnan(intensities)] = 0
                ranges[np.isnan(ranges)] = 0

        if '_ptcld' in args.mode:
            # Pointcloud Visualisation using Open3D
            if vis is None:
                vis = open3d.Visualizer()
                vis.create_window(window_name=title)
                pcd = open3d.geometry.PointCloud()
                # initialise the geometry pre loop
                pcd.points = open3d.utility.Vector3dVector(ptcld[:3].transpose().astype(np.float64))
                pcd.colors = open3d.utility.Vector3dVector(np.tile(ptcld[3:].transpose(), (1, 3)).astype(np.float64))
                # Rotate pointcloud to align displayed coordinate frame colouring
                pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))
                vis.add_geometry(pcd)
                render_option = vis.get_render_option()
                render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
                render_option.point_color_option = open3d.PointColorOption.ZCoordinate
                coordinate_frame = open3d.geometry.create_mesh_coordinate_frame()
                vis.add_geometry(coordinate_frame)
                view_control = vis.get_view_control()
                params = view_control.convert_to_pinhole_camera_parameters()
                params.extrinsic = build_se3_transform([0, 3, 10, 0, -np.pi * 0.42, -np.pi / 2])
                view_control.convert_from_pinhole_camera_parameters(params)

            pcd.points = open3d.utility.Vector3dVector(ptcld[:3].transpose().astype(np.float64))
            pcd.colors = open3d.utility.Vector3dVector(
                np.tile(ptcld[3:].transpose(), (1, 3)).astype(np.float64) / 40)
            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()

        else:
            # Ranges and Intensities visualisation using OpenCV
            intensities_vis = colourmap[np.clip((intensities * 4).astype(int), 0, colourmap.shape[0] - 1)]
            ranges_vis = colourmap[np.clip((ranges * 4).astype(int), 0, colourmap.shape[0] - 1)]
            visualisation = np.concatenate((intensities_vis, ranges_vis), 0)
            visualisation = cv2.resize(visualisation, None, fy=6 * args.scale, fx=args.scale,
                                       interpolation=cv2.INTER_NEAREST)
            print("Type of vis ",type(visualisation))
            cv2.imshow(title, visualisation)
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
