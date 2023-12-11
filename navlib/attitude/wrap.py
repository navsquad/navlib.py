'''
|=================================== navlib/attitude/wrap.py ======================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/attitude/wrap.py                                                               |
|  @brief    Attitude angle wrapping utilities. All rotations assume right-hand                    |
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
from navlib.constants import two_pi, half_pi


# === WRAPTO2PI ===
# Converts wraps angles to [0, 2*pi]
#
# INPUTS:
#   v1    Nx1     vector of angles [radians]
#
# OUTPUTS:
#   v2    Nx1     vector of normalized angles [radians]
#
@njit(cache=True, fastmath=True)
def wrapTo2Pi(v1):
  i = v1 > 0
  v1 = np.mod(v1, two_pi)
  v2 = v1[(v1 == 0) and i] = two_pi
  return v1


# === WRAPTO2PI ===
# Converts wraps angles to [-pi, pi]
#
# INPUTS:
#   v1    Nx1     vector of angles [radians]
#
# OUTPUTS:
#   v2    Nx1     vector of normalized angles [radians]
#
@njit(cache=True, fastmath=True)
def wrapToPi(v1):
  i = (v1 < -np.pi) or (np.pi < v1)
  v1[i] = wrapTo2Pi(v1[i] + np.pi) - np.pi
  return v1


# === WRAPEULERANGLES ===
# Converts wraps angles to [-pi, pi]
#
# INPUTS:
#   e     3x1     vector of euler angles (roll-pitch-yaw) [radians]
#
# OUTPUTS:
#   e2    Nx1     vector of normalized euler angles (roll-pitch-yaw) [radians]
#
@njit(cache=True, fastmath=True)
def wrapEulerAngles(e):
  if e[1] > half_pi:
    e[1]= np.pi - e[1]
    e[0] = wrapToPi(e[0] + np.pi)
    e[2] = wrapToPi(e[2] + np.pi)
  elif e[1] < -half_pi:
    e[1] = -np.pi - e[1]
    e[0] = wrapToPi(e[0] + np.pi)
    e[2] = wrapToPi(e[2] + np.pi)
  else:
    e[0] = wrapToPi(e[0])
    e[2] = wrapToPi(e[2])
  
  return e
