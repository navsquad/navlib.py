'''
|=================================== navlib/attitude/skew.py ======================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/attitude/skew.py                                                               |
|  @brief    Attitude skew symmetric form utilities. All rotations assume right-hand               |
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

# === SKEW ===
# Converts vector into its skew symmetric form
#
# INPUTS:
#   v     3x1     vector
#
# OUTPUTS:
#   M     3x3     skew symmetric form of input vector
#
@njit(cache=True, fastmath=True)
def skew(v):
  M = np.array([[  0.0, -v[2],  v[1]], \
                [ v[2],   0.0, -v[0]], \
                [-v[1],  v[0],   0.0]], \
       dtype=np.double)
  return M


# === DESKEW ===
# Converts skew symmetric form into its respective vector
#
# INPUTS:
#   M     3x3     skew symmetric form of vector
#
# OUTPUTS:
#   v     3x1     output vector
#
@njit(cache=True, fastmath=True)
def deskew(M):
  v = np.array([M[2,1], M[0,2], M[1,0]], dtype=np.double)
  return v
