'''
|==================================== navlib/attitude/dcm.py ======================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/attitude/dcm.py                                                                |
|  @brief    Attitude conversion from direction cosine matrices. All rotations assume right-hand   |
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
from .euler import euler2quat

# === DCM2EULER === 
# Converts 'ZYX' DCM matrix into corresponding euler angles (roll-pitch-yaw)
#
# INPUTS:
#   C     3x3     'XYZ' direction cosine matrix
#
# OUTPUTS:
#   e     3x1   euler angles [rad]
#
@njit(cache=True, fastmath=True)
def dcm2euler(C):
  e = np.array([np.arctan2(C[1,2], C[2,2]), \
                np.arcsin(-C[0,2]), \
                np.arctan2(C[0,1], C[0,0])], 
               dtype=np.double)
  return e


# === DCM2QUAT === 
# Converts DCM matrix into corresponding quaternion
#
# INPUTS:
#   C     3x3     'XYZ' direction cosine matrix
#
# OUTPUTS:
#   q     4x1   quaternion
#
@njit(cache=True, fastmath=True)
def dcm2quat(C):
  q = euler2quat(dcm2euler(C))
  return q
