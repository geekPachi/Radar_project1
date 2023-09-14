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


        # Explaination of below code



# import numpy as np: Import the numpy library as np, which is commonly used for numerical operations and array manipulation.

# import numpy.matlib as matlib: Import the numpy.matlib module as matlib, which provides matrix manipulation functions.

# from math import sin, cos, atan2, sqrt: Import specific math functions sin, cos, atan2, and sqrt. These functions are used for trigonometric calculations.

# MATRIX_MATCH_TOLERANCE = 1e-4: Define a constant MATRIX_MATCH_TOLERANCE with a value of 1e-4, which represents a tolerance value used for matrix comparisons.

# def build_se3_transform(xyzrpy):: Define a function build_se3_transform that takes a list xyzrpy as input. This function is used to create an SE3 homogeneous transformation matrix from translation and Euler angles.

# Inside the build_se3_transform function, a check is performed to ensure that the length of the xyzrpy list is 6. If not, a ValueError is raised, indicating that the function requires exactly six values.

# se3 = matlib.identity(4): Create a 4x4 identity matrix se3 using matlib.

# se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6]): Assign the top-left 3x3 submatrix of se3 by calling the euler_to_so3 function on the elements of xyzrpy from index 3 to 5 (Euler angles).

# se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose(): Assign the top-right 3x1 submatrix of se3 with the translation values from xyzrpy, converted to a column matrix.

# Return the resulting se3 matrix as the output of the function.

# def euler_to_so3(rpy):: Define a function euler_to_so3 that takes a list rpy (Euler angles in radians) as input. This function is used to convert Euler angles to an SO3 rotation matrix.

# Inside the euler_to_so3 function, a check is performed to ensure that the length of the rpy list is 3. If not, a ValueError is raised, indicating that the function requires exactly three values.

# Compute the individual rotation matrices for rotations around the X, Y, and Z axes using the provided Euler angles.

# Multiply these individual rotation matrices together to obtain the final SO3 rotation matrix R_zyx.

# Return the resulting R_zyx matrix as the output of the function.

# def so3_to_euler(so3):: Define a function so3_to_euler that takes an SO3 rotation matrix so3 as input. This function is used to convert an SO3 rotation matrix to Euler angles.

# Inside the so3_to_euler function, a check is performed to ensure that the so3 matrix is 3x3. If not, a ValueError is raised.

# Calculate the roll and yaw angles using the elements of the so3 matrix.

# Compute the pitch angle with two possible values and select the one that matches the input so3 matrix most closely.

# Return a matrix containing the computed Euler angles.

# def so3_to_quaternion(so3):: Define a function so3_to_quaternion that takes an SO3 rotation matrix so3 as input. This function is used to convert an SO3 rotation matrix to a quaternion.

# Inside the so3_to_quaternion function, a check is performed to ensure that the so3 matrix is 3x3. If not, a ValueError is raised.

# Calculate the elements of the quaternion based on the elements of the so3 matrix, following a specific formula.

# Return the resulting quaternion as a NumPy array [w, x, y, z].

# def se3_to_components(se3):: Define a function se3_to_components that takes an SE3 transformation matrix se3 as input. This function is used to convert an SE3 matrix into linear translation and Euler angles.

# Inside the se3_to_components function, a check is performed to ensure that the se3 matrix is 4x4. If not, a ValueError is raised.

# Create an empty NumPy array xyzrpy to store the translation and Euler angles.

# Extract the translation components from the se3 matrix and store them in the first three elements of xyzrpy.

# Extract the rotation components using the `so





import numpy as np
import numpy.matlib as matlib
from math import sin, cos, atan2, sqrt

MATRIX_MATCH_TOLERANCE = 1e-4


def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.

    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix

    Raises:
        ValueError: if `len(xyzrpy) != 6`

    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx


def so3_to_euler(so3):
    """Converts an SO3 rotation matrix to Euler angles

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of Euler angles (size 3)

    Raises:
        ValueError: if so3 is not 3x3
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")
    roll = atan2(so3[2, 1], so3[2, 2])
    yaw = atan2(so3[1, 0], so3[0, 0])
    denom = sqrt(so3[0, 0] ** 2 + so3[1, 0] ** 2)
    pitch_poss = [atan2(-so3[2, 0], denom), atan2(-so3[2, 0], -denom)]

    R = euler_to_so3((roll, pitch_poss[0], yaw))

    if (so3 - R).sum() < MATRIX_MATCH_TOLERANCE:
        return np.matrix([roll, pitch_poss[0], yaw])
    else:
        R = euler_to_so3((roll, pitch_poss[1], yaw))
        if (so3 - R).sum() > MATRIX_MATCH_TOLERANCE:
            raise ValueError("Could not find valid pitch angle")
        return np.matrix([roll, pitch_poss[1], yaw])


def so3_to_quaternion(so3):
    """Converts an SO3 rotation matrix to a quaternion

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.ndarray: quaternion [w, x, y, z]

    Raises:
        ValueError: if so3 is not 3x3
    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")

    R_xx = so3[0, 0]
    R_xy = so3[0, 1]
    R_xz = so3[0, 2]
    R_yx = so3[1, 0]
    R_yy = so3[1, 1]
    R_yz = so3[1, 2]
    R_zx = so3[2, 0]
    R_zy = so3[2, 1]
    R_zz = so3[2, 2]

    try:
        w = sqrt(so3.trace() + 1) / 2
    except(ValueError):
        # w is non-real
        w = 0

    # Due to numerical precision the value passed to `sqrt` may be a negative of the order 1e-15.
    # To avoid this error we clip these values to a minimum value of 0.
    x = sqrt(max(1 + R_xx - R_yy - R_zz, 0)) / 2
    y = sqrt(max(1 + R_yy - R_xx - R_zz, 0)) / 2
    z = sqrt(max(1 + R_zz - R_yy - R_xx, 0)) / 2

    max_index = max(range(4), key=[w, x, y, z].__getitem__)

    if max_index == 0:
        x = (R_zy - R_yz) / (4 * w)
        y = (R_xz - R_zx) / (4 * w)
        z = (R_yx - R_xy) / (4 * w)
    elif max_index == 1:
        w = (R_zy - R_yz) / (4 * x)
        y = (R_xy + R_yx) / (4 * x)
        z = (R_zx + R_xz) / (4 * x)
    elif max_index == 2:
        w = (R_xz - R_zx) / (4 * y)
        x = (R_xy + R_yx) / (4 * y)
        z = (R_yz + R_zy) / (4 * y)
    elif max_index == 3:
        w = (R_yx - R_xy) / (4 * z)
        x = (R_zx + R_xz) / (4 * z)
        y = (R_yz + R_zy) / (4 * z)

    return np.array([w, x, y, z])


def se3_to_components(se3):
    """Converts an SE3 rotation matrix to linear translation and Euler angles

    Args:
        se3: 4x4 transformation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of [x, y, z, roll, pitch, yaw]

    Raises:
        ValueError: if se3 is not 4x4
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if se3.shape != (4, 4):
        raise ValueError("SE3 transform must be a 4x4 matrix")
    xyzrpy = np.empty(6)
    xyzrpy[0:3] = se3[0:3, 3].transpose()
    xyzrpy[3:6] = so3_to_euler(se3[0:3, 0:3])
    return xyzrpy
