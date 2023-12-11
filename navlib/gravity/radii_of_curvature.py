'''
|============================= navlib/gravity/radii_of_curvature.py ===============================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/gravity/radii_of_curvature.py                                                  |
|  @brief    Calculations for common Earth radii used for navigation.                              |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from navlib.constants import Ro, e2

# === TRANSVERSERADIUS ===
# Calculates the transverse radius relative to user latitude
#
# INPUTS:
#   phi     3x1     Latitude [rad]
#
# OUTPUTS:
#   Re      double  Earth's transverse radius at Latitude
#
@njit(cache=True, fastmath=True)
def transverseRadius(phi):
  sinPhi2 = np.sin(phi)**2
  t = 1 - e2*sinPhi2
  return Ro / np.sqrt(t)


# === MERIDIANRADIUS ===
# Calculates the meridian radius relative to user latitude
#
# INPUTS:
#   phi     3x1     Latitude [rad]
#
# OUTPUTS:
#   Rn      double  Earth's meridian radius at Latitude
#
@njit(cache=True, fastmath=True)
def meridianRadius(phi):
  sinPhi2 = np.sin(phi)**2
  t = 1 - e2*sinPhi2
  return Ro * (1 - e2) / (t**1.5)


# === GEOCENTRICRADIUS ===
# Calculates the geocentric radius relative to the user latitude
#
# INPUTS:
#   phi     3x1     Latitude [rad]
#
# OUTPUTS:
#   r_es_e  double  Earth's geocentric radius at Latitude
#
@njit(cache=True, fastmath=True)
def geocentricRadius(phi):
  sinPhi2 = np.sin(phi)**2
  cosPhi2 = np.cos(phi)**2
  t = 1 - e2*sinPhi2
  Re = Ro / np.sqrt(t)
  return Re * np.sqrt(cosPhi2 + (1 - e2)**2 * sinPhi2)


# === RADIIOFCURVATURE ===
# Calculates the transverse, meridian, and geocentric radii or curvature
#
# INPUTS:
#   phi     3x1     Latitude [rad]
#
# OUTPUTS:
#   Re      double  Earth's transverse radius at Latitude
#   Rn      double  Earth's meridian radius at Latitude
#   r_es_e  double  Earth's geocentric radius at Latitude
#
@njit(cache=True, fastmath=True)
def radiiOfCurvature(phi):
  sinPhi2 = np.sin(phi)**2
  cosPhi2 = np.cos(phi)**2
  t = 1 - e2*sinPhi2
  Re = Ro / np.sqrt(t)
  Rn = Ro * (1- e2) / (t**1.5)
  r_es_e = Re * np.sqrt(cosPhi2 + (1 - e2)**2 * sinPhi2)
  return Re, Rn, r_es_e
  