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


# import numpy as np: Import the numpy library as np, which is commonly used for numerical operations and array manipulation.

# from PIL import Image: Import the Image module from the Python Imaging Library (PIL), which is used for working with images in various formats.

# The load_road_boundary_image function is defined, which takes an image_path as input and returns a NumPy array representing the image.

# Inside the load_road_boundary_image function, Image.open(image_path) is used to open the image located at image_path and load it into a PIL Image object. This allows for easy manipulation and conversion of the image.

# The function returns the image data as a NumPy array by converting the Image object to an array using np.array(img).

# .astype(np.uint8) is used to cast the array data type to uint8, which represents an 8-bit unsigned integer. This is a common data type for image data, where pixel values typically range from 0 to 255.

# The load_road_boundary_mask function is defined, which takes mask_path and model as input parameters and returns the result of calling load_road_boundary_image(mask_path).

# Inside the load_road_boundary_mask function, load_road_boundary_image(mask_path) is called to load the mask image. This function essentially delegates the loading of the mask image to the load_road_boundary_image function.

# These functions are designed to load image and mask data, which is commonly used in computer vision and image processing tasks. The PIL library is used for opening and manipulating images, and numpy is used for working with the image data in a numerical format.




import numpy as np
from PIL import Image


def load_road_boundary_image(image_path):
    img = Image.open(image_path)
    return np.array(img).astype(np.uint8)


def load_road_boundary_mask(mask_path, model):
    return load_road_boundary_image(mask_path)
