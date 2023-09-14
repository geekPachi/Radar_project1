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
###############################################################################


        # Code Explaination 



# from typing import AnyStr, Tuple: Import necessary type hints for function parameter and return value annotations.

# import numpy as np: Import the numpy library as np, which is used for numerical computations.

# import cv2: Import the OpenCV library, which is used for image processing and computer vision tasks.

# The load_radar function is defined, which takes an example_path as input and returns multiple values as a tuple. This function is used to decode a single Oxford Radar RobotCar Dataset radar example.

# Inside the load_radar function, a docstring describes the function's purpose and the format of its return values.

# radar_resolution is defined as a hard-coded configuration value. It represents the resolution of the polar radar data in meters per pixel.

# encoder_size is defined as a constant value of 5600.

# raw_example_data is assigned the result of reading the image data from the example_path using OpenCV's cv2.imread function. The image is read in grayscale mode.

# timestamps, azimuths, valid, and fft_data are extracted from the raw_example_data using array slicing and data type conversions.

# The function returns a tuple containing the extracted values: timestamps, azimuths, valid, fft_data, and radar_resolution.

# The radar_polar_to_cartesian function is defined, which converts a polar radar scan to Cartesian coordinates. It takes several input parameters and returns a Cartesian radar image.

# The function's docstring describes the purpose and usage of the function, as well as the format of its return value.

# The function calculates the cart_min_range based on whether cart_pixel_width is even or odd, which determines the range of Cartesian coordinates.

# It creates a grid of Cartesian coordinates (X and Y) using np.meshgrid, which represents the pixel locations in Cartesian space.

# The function calculates sample_range as the Euclidean distance from the origin for each pixel and sample_angle as the azimuth angle for each pixel.

# It calculates sample_u and sample_v to represent the coordinates in the radar data space based on the provided parameters.

# The function clips the sample_u values to ensure they are not below 0, which corresponds to the minimum sensor reading range.

# If interpolate_crossover is True, the function adds an extra azimuthal scan to interpolate between the end and start azimuth of the radar scan.

# polar_to_cart_warp is created as a stack of sample_u and sample_v arrays to define the warp transformation.

# The function uses OpenCV's cv2.remap function to perform the interpolation from polar radar data to Cartesian radar data.

# The result is stored in cart_img, which is then returned as the function's output.

# This code is responsible for loading and processing radar data from a specific format and converting it into a Cartesian representation for further analysis or visualization.




from typing import AnyStr, Tuple
import numpy as np
import cv2


def load_radar(example_path: AnyStr) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Decode a single Oxford Radar RobotCar Dataset radar example
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset Example png
    Returns:
        timestamps (np.ndarray): Timestamp for each azimuth in int64 (UNIX time)
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        valid (np.ndarray) Mask of whether azimuth data is an original sensor reading or interpolated from adjacent
            azimuths
        fft_data (np.ndarray): Radar power readings along each azimuth
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
    """
    # Hard coded configuration to simplify parsing code
    radar_resolution = np.array([0.0432], np.float32)
    encoder_size = 5600

    raw_example_data = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
    timestamps = raw_example_data[:, :8].copy().view(np.int64)
    azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    valid = raw_example_data[:, 10:11] == 255
    fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255.

    return timestamps, azimuths, valid, fft_data, radar_resolution


def radar_polar_to_cartesian(azimuths: np.ndarray, fft_data: np.ndarray, radar_resolution: float,
                             cart_resolution: float, cart_pixel_width: int, interpolate_crossover=True) -> np.ndarray:
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_size (int): Width and height of the returned square cartesian output (pixels). Please see the Notes
            below for a full explanation of how this is used.
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    Notes:
        After using the warping grid the output radar cartesian is defined as as follows where
        X and Y are the `real` world locations of the pixels in metres:
         If 'cart_pixel_width' is odd:
                        +------ Y = -1 * cart_resolution (m)
                        |+----- Y =  0 (m) at centre pixel
                        ||+---- Y =  1 * cart_resolution (m)
                        |||+--- Y =  2 * cart_resolution (m)
                        |||| +- Y =  cart_pixel_width // 2 * cart_resolution (m) (at last pixel)
                        |||| +-----------+
                        vvvv             v
         +---------------+---------------+
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         +---------------+---------------+ <-- X = 0 (m) at centre pixel
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         +---------------+---------------+
         <------------------------------->
             cart_pixel_width (pixels)
         If 'cart_pixel_width' is even:
                        +------ Y = -0.5 * cart_resolution (m)
                        |+----- Y =  0.5 * cart_resolution (m)
                        ||+---- Y =  1.5 * cart_resolution (m)
                        |||+--- Y =  2.5 * cart_resolution (m)
                        |||| +- Y =  (cart_pixel_width / 2 - 0.5) * cart_resolution (m) (at last pixel)
                        |||| +----------+
                        vvvv            v
         +------------------------------+
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         +------------------------------+
         <------------------------------>
             cart_pixel_width (pixels)
    """
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
    Y, X = np.meshgrid(coords, -coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

    # Interpolate Radar Data Coordinates
    azimuth_step = azimuths[1] - azimuths[0]
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
    return cart_img
