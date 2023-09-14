################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################



        # Explaination of Code 

# This Python script appears to be a part of a larger program for building and visualizing a point cloud from LIDAR (Light Detection and Ranging) data along with odometry information. 
# Here's an explanation of the script:

# Importing Libraries: The script starts by importing several Python libraries, including os, re (regular expressions), numpy, and functions from custom modules (transform, interpolate_poses, velodyne, and open3d).
# These libraries are used for various purposes like file operations, data manipulation, and visualization.

# build_pointcloud Function: This function is the main focus of the script. It is responsible for building a point cloud by combining LIDAR scans with odometry information. Here's what it does:

# Takes several arguments: lidar_dir (directory containing LIDAR scans), poses_file (file containing pose information), extrinsics_dir (directory containing extrinsic calibrations), start_time, end_time (timestamps defining the time window for data inclusion), and origin_time (timestamp for the origin frame).

# Determines the LIDAR type (e.g., 'lms_front', 'lms_rear', 'ldmrs', 'velodyne_left', 'velodyne_right') from the lidar_dir.

# Reads timestamps from a file named <lidar_type>.timestamps in the LIDAR directory. It filters timestamps within the specified time window.

# Reads extrinsic calibration information from a file and builds a transformation matrix.

# Depending on the type of odometry data (VO - Visual Odometry, INS - Inertial Navigation System, RTK - Real-Time Kinematics), it adjusts the transformation matrix.

# Iterates through the filtered timestamps, reads LIDAR scan data from binary files, and applies transformations to align the scans.

# Concatenates the transformed scan points into a point cloud, considering reflectance values (if available).

# If no scan files are found within the specified time window, it raises an IOError.

# Command-Line Argument Parsing: The script also has a command-line interface using the argparse library. Users can provide command-line arguments --poses_file, --extrinsics_dir, and --laser_dir to specify the relevant files and directories.

# Visualization Using Open3D: After building the point cloud, the script uses the open3d library for visualization. It sets up a 3D visualization window, specifies the background color, assigns colors to points based on reflectance values, and adds the point cloud to the visualization.

# The point cloud is transformed to match the coordinate frame for better visualization.

# It also specifies the initial camera parameters for the viewpoint.

# The visualization is run using vis.run().

# This script is designed to be a part of a larger system for processing and visualizing LIDAR data with odometry information. It provides a command-line interface for specifying input files and directories and uses the open3d library to display the resulting point cloud.

import os
import re
import numpy as np

from transform import build_se3_transform
from interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from velodyne import load_velodyne_raw, load_velodyne_binary, velodyne_raw_to_pointcloud

from open3d.visualization import VisualizerWithKeyCallback
import open3d as o3d





def build_pointcloud(lidar_dir, poses_file, extrinsics_dir, start_time, end_time, origin_time=-1):
    """Builds a pointcloud by combining multiple LIDAR scans with odometry information.

    Args:
        lidar_dir (str): Directory containing LIDAR scans.
        poses_file (str): Path to a file containing pose information. Can be VO or INS data.
        extrinsics_dir (str): Directory containing extrinsic calibrations.
        start_time (int): UNIX timestamp of the start of the window over which to build the pointcloud.
        end_time (int): UNIX timestamp of the end of the window over which to build the pointcloud.
        origin_time (int): UNIX timestamp of origin frame. Pointcloud coordinates are relative to this frame.

    Returns:
        numpy.ndarray: 3xn array of (x, y, z) coordinates of pointcloud
        numpy.array: array of n reflectance values or None if no reflectance values are recorded (LDMRS)

    Raises:
        ValueError: if specified window doesn't contain any laser scans.
        IOError: if scan files are not found.

    """
    if origin_time < 0:
        origin_time = start_time

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', lidar_dir).group(0)
    timestamps_path = os.path.join(lidar_dir, os.pardir, lidar + '.timestamps')

    timestamps = []
    with open(timestamps_path) as timestamps_file:
        for line in timestamps_file:
            timestamp = int(line.split(' ')[0])
            if start_time <= timestamp <= end_time:
                timestamps.append(timestamp)

    if len(timestamps) == 0:
        raise ValueError("No LIDAR data in the given time bracket.")

    with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
    G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])

    poses_type = re.search('(vo|ins|rtk)\.csv', poses_file).group(1)

    if poses_type in ['ins', 'rtk']:
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser)

        poses = interpolate_ins_poses(poses_file, timestamps, origin_time, use_rtk=(poses_type == 'rtk'))
    else:
        # sensor is VO, which is located at the main vehicle frame
        poses = interpolate_vo_poses(poses_file, timestamps, origin_time)

    pointcloud = np.array([[0], [0], [0], [0]])
    if lidar == 'ldmrs':
        reflectance = None
    else:
        reflectance = np.empty((0))

    for i in range(0, len(poses)):
        scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.bin')
        if "velodyne" not in lidar:
            if not os.path.isfile(scan_path):
                continue

            scan_file = open(scan_path)
            scan = np.fromfile(scan_file, np.double)
            scan_file.close()

            scan = scan.reshape((len(scan) // 3, 3)).transpose()

            if lidar != 'ldmrs':
                # LMS scans are tuples of (x, y, reflectance)
                reflectance = np.concatenate((reflectance, np.ravel(scan[2, :])))
                scan[2, :] = np.zeros((1, scan.shape[1]))
        else:
            if os.path.isfile(scan_path):
                ptcld = load_velodyne_binary(scan_path)
            else:
                scan_path = os.path.join(lidar_dir, str(timestamps[i]) + '.png')
                if not os.path.isfile(scan_path):
                    continue
                ranges, intensities, angles, approximate_timestamps = load_velodyne_raw(scan_path)
                ptcld = velodyne_raw_to_pointcloud(ranges, intensities, angles)

            reflectance = np.concatenate((reflectance, ptcld[3]))
            scan = ptcld[:3]

        scan = np.dot(np.dot(poses[i], G_posesource_laser), np.vstack([scan, np.ones((1, scan.shape[1]))]))
        pointcloud = np.hstack([pointcloud, scan])

    pointcloud = pointcloud[:, 1:]
    if pointcloud.shape[1] == 0:
        raise IOError("Could not find scan files for given time range in directory " + lidar_dir)

    return pointcloud, reflectance


if __name__ == "__main__":
    import argparse
    import open3d

    parser = argparse.ArgumentParser(description='Build and display a pointcloud')
    parser.add_argument('--poses_file', type=str, default=None, help='File containing relative or absolute poses')
    parser.add_argument('--extrinsics_dir', type=str, default=None,
                        help='Directory containing extrinsic calibrations')
    parser.add_argument('--laser_dir', type=str, default=None, help='Directory containing LIDAR data')

    args = parser.parse_args()

    lidar = re.search('(lms_front|lms_rear|ldmrs|velodyne_left|velodyne_right)', args.laser_dir).group(0)
    timestamps_path = os.path.join(args.laser_dir, os.pardir, lidar + '.timestamps')
    with open(timestamps_path) as timestamps_file:
        start_time = int(next(timestamps_file).split(' ')[0])

    end_time = start_time + 2e7

    pointcloud, reflectance = build_pointcloud(args.laser_dir, args.poses_file,
                                               args.extrinsics_dir, start_time, end_time)

    if reflectance is not None:
        colours = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min())
        colours = 1 / (1 + np.exp(-10 * (colours - colours.mean())))
    else:
        colours = 'gray'

    # Pointcloud Visualisation using Open3D
    # Create a visualizer instance
    vis = VisualizerWithKeyCallback()

    # Create a window
    vis.create_window(window_name=os.path.basename(__file__))

    # Access render options
    render_option = vis.get_render_option()

    # Set the background color
    render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)

    # Set the point color option (corrected)
    render_option.point_color_option = o3d.visualization.PointColorOption.ZCoordinate

    # Create a coordinate frame (corrected)
    coordinate_frame = o3d.geometry.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        -np.ascontiguousarray(pointcloud[[1, 0, 2]].transpose().astype(np.float64)))

    # Set colors for the point cloud
    pcd.colors = o3d.utility.Vector3dVector(np.tile(colours[:, np.newaxis], (1, 3)).astype(np.float64))

    # Rotate point cloud to align with the displayed coordinate frame
    pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    # Access view control parameters
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()

    # Set extrinsic parameters
    params.extrinsic = build_se3_transform([0, 3, 10, 0, -np.pi * 0.42, -np.pi / 2])

    # Update view control with the new parameters
    view_control.convert_from_pinhole_camera_parameters(params)

    # Run the visualizer
    vis.run()


