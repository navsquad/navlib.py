'''
|================================ navlib/attitude/quaternion.py ===================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/attitude/quaternion.py                                                         |
|  @brief    Attitude conversion from quaternions. All rotations assume right-hand                 |
|            coordinate frames with the order. Assumes euler angles in the order 'roll-pitch-yaw'  |
|            and DCMs with the order of 'ZYX'.                                                     |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit

# === QUAT2EULER ===
# Converts quaternion to corresponding euler angles (roll-pitch-yaw)
#
# INPUTS:
#   q     4x1     quaternion
#
# OUTPUTS:
#   e     3x1     RPY euler angles [radians]
#
@njit(cache=True, fastmath=True)
def quat2euler(q):
  w, x, y, z = q
  e = np.array([np.arctan2(2*(w*x + y*z), (w*w - x*x - y*y + z*z)), \
                np.arcsin(-2*(-w*y + x*z)), \
                np.arctan2(2*(w*z + x*y), (w*w + x*x - y*y - z*z))], \
               dtype=np.double)
  return e


# === QUAT2DCM ===
# Converts quaternion to corresponding 'XYZ' DCM
#
# INPUTS:
#   q     4x1     quaternion
#
# OUTPUTS:
#   C     3x3     'XYZ' direction cosine matrix
#
@njit(cache=True, fastmath=True)
def quat2dcm(q):
  w, x, y, z = q
  C = np.array([[w*w + x*x - y*y - z*z,         2*(x*y - w*z),          2*(w*y + x*z)], \
                [        2*(w*z + x*y), w*w - x*x + y*y - z*z,          2*(y*z - w*x)], \
                [        2*(x*z - w*y),         2*(y*z + w*x),  w*w - x*x - y*y + z*z]], \
               dtype=np.double)
  return C
