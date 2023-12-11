'''
|=================================== navlib/gravity/gravity.py ====================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/gravity/gravity.py                                                             |
|  @brief    Calculations for Earth gravity rates in different frames.                             |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from scipy import linalg
from numba import njit
from navlib.constants import e2, Ro, Rp, f, mu, w_ie, J2
from navlib.coordinates.dcm import ned2ecefDcm
from navlib.coordinates.position import ecef2lla


# === SOMIGLIANA ===
# Calculates the somilgiana model to calculate reference gravity
#
# INPUTS:
#   lla     3x1       Latitude, longitude, height [rad, rad, m]
#
# OUTPUTS:
#   g0      double    somigliana gravity
#
@njit(cache=True, fastmath=True)
def somigliana(lla):
  sinPhi2 = np.sin(lla[0])**2
  return 9.7803253359 * ((1 + 0.001931853*sinPhi2) / np.sqrt(1 - e2*sinPhi2))


# === NEDG ===
# Calculates gravity in the 'NED' frame
#
# INPUTS:
#   lla     3x1       Latitude, longitude, height [rad, rad, m]
#
# OUTPUTS:
#   g       3x1       'NED' gravity
#
@njit(cache=True, fastmath=True)
def nedg(lla):
  phi, lam, h = lla
  sinPhi2 = np.sin(phi)**2
  g0 = 9.7803253359 * ((1 + 0.001931853*sinPhi2) / np.sqrt(1 - e2*sinPhi2))
  return np.array([-8.08e-9 * h * np.sin(2*phi), \
                    0.0, \
                    g0 * (1 - (2 / Ro) * (1 + f * (1 - 2 * sinPhi2) + \
                    (w_ie**2 * Ro**2 * Rp / mu)) * h + \
                    (3 * h**2 / Ro**2))
                  ])
  
  
# === GRAVITYECEF ===
# Calculates gravity in the 'ECEF' frame
#
# INPUTS:
#   lla     3x1       Latitude, longitude, height [rad, rad, m]
#
# OUTPUTS:
#   g       3x1       'ECEF' gravity
#   gamma   3x1       'ECEF' gravitational acceleration
#
@njit(cache=True, fastmath=True)
def ecefg(r_eb_e):
  x,y,z = r_eb_e
  mag_r = np.linalg.norm(r_eb_e)
  if mag_r == 0.0:
    gamma = np.zeros(3)
    g = np.zeros(3)
  else:
    zeta = 5 * (z / mag_r)**2
    M = np.array([(1.0 - zeta) * x, \
                  (1.0 - zeta) * y, \
                  (3.0 - zeta) * z], 
                 dtype=np.double)
    gamma = -mu / mag_r**3 * (r_eb_e + 1.5 * J2 * (Ro / mag_r)**2 * M)
    g = gamma + w_ie*w_ie * np.array([x, y, 0.0], dtype=np.double)
  return g, gamma


# === NED2ECEFG ===
# Calculates gravity in the 'NAV' frame and rotates it to the 'ECEF' frame
#
# INPUTS:
#   lla     3x1       Latitude, longitude, height [rad, rad, m]
#
# OUTPUTS:
#   g       3x1       'ECEF' gravity
#   gamma   3x1       'ECEF' gravitational acceleration
#
@njit(cache=True, fastmath=True)
def ned2ecefg(r_eb_e):
  lla = ecef2lla(r_eb_e)
  g_ned = nedg(lla)
  C_n_e = ned2ecefDcm(lla)
  g_ecef = C_n_e @ g_ned
  gamma = g_ecef - w_ie*w_ie * np.array([r_eb_e[0], r_eb_e[1], 0.0], dtype=np.double)
  return g_ecef, gamma
