'''
|================================== navlib/gravity/rotations.py ===================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/gravity/rotations.py                                                           |
|  @brief    Calculations for common Earth rotation rates used for navigation frame.               |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from navlib.attitude.skew import skew
from navlib.constants import w_ie
from .radii_of_curvature import radiiOfCurvature


# === EARTHRATE ===
# Rotation rate of the earth relative to the 'NAV' frame
#
# INPUTS:
#   lla     3x1   Latitude, longitude, height [rad, rad, m]
#
# OUTPUTS:
#   W_ie_n  3x3   Skew symmetric form of the earth'r rotation in the 'NAV' frame
#
@njit(cache=True, fastmath=True)
def earthRate(lla):
  sinPhi = np.sin(lla[0])
  cosPhi = np.cos(lla[0])
  return skew(np.array([w_ie*cosPhi, 0.0, w_ie*sinPhi], dtype=np.double))
  

# === TRANSPORTRATE ===
# Transport rate of the 'ECEF' frame relative to the 'NAV' frame
#
# INPUTS:
#   lla     3x1   Latitude, longitude, height [rad, rad, m]
#   v_nb_n  3x1   Velocity in the 'NED' coordinate system
#
# OUTPUTS:
#   W_en_n  3x3   Skew symmetric form of the earth'r rotation in the 'NAV' frame
#
@njit(cache=True, fastmath=True)
def transportRate(lla, v_nb_n):
  phi, lam, h = lla
  vn, ve, vd = v_nb_n
  Re, Rn, r_es_e = radiiOfCurvature(phi)
  return skew(np.array([ ve / (Re + h), \
                        -vn / (Rn + h), \
                        -ve * np.tan(phi) / (Re + h)], \
                       dtype=np.double))
  

# === CORIOLIS ===
# Coriolis effect perceived in the nave frame
#
# INPUTS:
#   lla     3x1   Latitude, longitude, height [rad, rad, m]
#   v_nb_n  3x1   Velocity in the 'NED' coordinate system
#
# OUTPUTS:
#   cor     3x1   Coriolis effect
#
@njit(cache=True, fastmath=True)
def coriolis(lla, v_nb_n):
  W_ie_n = earthRate(lla)
  W_en_n = transportRate(lla,  v_nb_n)
  return (W_en_n + 2*W_ie_n) @ v_nb_n