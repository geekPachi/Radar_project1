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
###############################################################################



# This Python script defines a function called load_image that loads and processes images from a dataset. It utilizes various libraries for image handling, including re for regular expressions, PIL (Python Imaging Library) for image loading, colour_demosaicing for demosaicing, and numpy for numerical operations. Here's an explanation of the script and its components:

# Import Statements:

# import re: This module allows for the use of regular expressions.
# from PIL import Image: This module provides functionality for opening and manipulating images.
# from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic: This imports a specific demosaicing function from the colour_demosaicing library and renames it as demosaic for convenience.
# import numpy as np: This imports the NumPy library with the alias np.
# Constants:

# BAYER_STEREO and BAYER_MONO: These constants define Bayer patterns used for demosaicing. Bayer patterns specify the arrangement of color filter elements on an image sensor.
# load_image Function:

# This function loads and processes an image from a given file path.

# Arguments:

# image_path (str): The path to the image file.
# model (optional): An instance of a camera model (from the camera_model module, not defined in this script). If provided, the function uses the model to undistort the image. It is optional and can be set to None.
# debayer (optional): A boolean flag that determines whether to demosaic the image. By default, it is set to True.
# Image Demosaicing:

# The script determines the Bayer pattern (pattern) based on the camera type (stereo or mono).
# It opens the image using PIL and, if debayer is True, it applies the demosaicing algorithm using the specified Bayer pattern.
# If a model is provided, it further undistorts the image using the camera model's undistort method.
# Return Value:

# The function returns the processed image as a NumPy array with data type np.uint8.
# Overall, the load_image function is designed to handle loading, demosaicing, and optionally undistorting images based on a camera model. It provides flexibility for various image processing tasks and is commonly used in computer vision and image analysis applications, especially when dealing with raw or calibrated image data.

import re
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import numpy as np

BAYER_STEREO = 'gbrg'
BAYER_MONO = 'rggb'


def load_image(image_path, model=None, debayer=True):
    """Loads and rectifies an image from file.

    Args:
        image_path (str): path to an image from the dataset.
        model (camera_model.CameraModel): if supplied, model will be used to undistort image.

    Returns:
        numpy.ndarray: demosaiced and optionally undistorted image

    """
    if model:
        camera = model.camera
    else:
        camera = re.search('(stereo|mono_(left|right|rear))', image_path).group(0)
    if camera == 'stereo':
        pattern = BAYER_STEREO
    else:
        pattern = BAYER_MONO

    img = Image.open(image_path)
    if debayer:
        img = demosaic(img, pattern)
    if model:
        img = model.undistort(img)

    return np.array(img).astype(np.uint8)

