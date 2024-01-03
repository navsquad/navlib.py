'''
|================================ navlib/coordinates/position.py ==================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/coordinate/position.py                                                         |
|  @brief    Common position coordinate frame transformations.                                     |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from .dcm import *
from navlib.constants import Ro, e2

#--------------------------------------------------------------------------------------------------#
# === LLA2ECI ===
# Latitude-Longitude-Height to Earth-Centered-Inertial coordinates
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   xyz     3x1     ECI x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def lla2eci(lla, dt):
  xyz = lla2ecef(lla)
  return ecef2eciDcm(dt) @ xyz


# === LLA2ECEF ===
# Latitude-Longitude-Height to Earth-Centered-Earth-Fixed coordinates
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   xyz     3x1     ECEF x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def lla2ecef(lla):
  sinPhi, sinLam = np.sin(lla[:2])
  cosPhi, cosLam = np.cos(lla[:2])
  h = lla[2]
  
  Re = Ro / np.sqrt(1 - e2*sinPhi*sinPhi)
  x = (Re + h) * cosPhi*cosLam
  y = (Re + h) * cosPhi*sinLam
  z = (Re * (1 - e2) + h) * sinPhi
  
  return np.array([x,y,z], dtype=np.double)


# === LLA2NED ===
# Latitude-Longitude-Height to North-East-Down coordinates
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   ned     3x1     NED x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def lla2ned(lla, lla0):
  C_e_n = ecef2nedDcm(lla0)
  xyz0 = lla2ecef(lla0)
  xyz = lla2ecef(lla)
  return C_e_n @ (xyz - xyz0)


# === LLA2ENU ===
# Latitude-Longitude-Height to East-North-Up coordinates
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   enu     3x1     ENU x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def lla2enu(lla, lla0):
  C_e_n = ecef2enuDcm(lla0)
  xyz0 = lla2ecef(lla0)
  xyz = lla2ecef(lla)
  return C_e_n @ (xyz - xyz0)


# === LLA2AER ===
# Converts Latitude-Longitude-Height to Azimuth-Elevation-Range coordinates
#
# INPUTS:
#   lla_t  3x1   target LLA coordinates
#   lla_r  3x1   reference LLA coordinates
#
# OUTPUTS:
#   aer     3x1   relative AER from reference to target
#
@njit(cache=True, fastmath=True)
def lla2aer(lla_t, lla_r):
  return enu2aer(lla2enu(lla_t, lla_r), ecef2enu(lla_r, lla_r))


#--------------------------------------------------------------------------------------------------#
# === ECI2ECEF ===
# Earth-Centered-Inertial to Earth-Centered-Earth-Fixed coordinates
#
# INPUTS:
#   xyz     3x1     ECI x,y,z coordinates [m]
#   dt      double  time [s]
#
# OUTPUTS:
#   xyz     3x1     ECEF x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def eci2ecef(xyz, dt):
  C_i_e = eci2ecefDcm(dt)
  return C_i_e @ xyz


# === ECI2LLA ===
# Earth-Centered-Inertial to Latitude-Longitude-Height coordinates
#
# INPUTS:
#   xyz     3x1     ECI x,y,z coordinates [m]
#   dt      double  time [s]
#
# OUTPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
@njit(cache=True, fastmath=True)
def eci2lla(xyz, dt):
  xyz = eci2ecef(xyz, dt)
  return ecef2lla(xyz)


# === ECI2NED ===
# Earth-Centered-Inertial to North-East-Down coordinates
#
# INPUTS:
#   ned     3x1     NED x,y,z coordinates [m]
#   lla0    3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
@njit(cache=True, fastmath=True)
def eci2ned(xyz, lla0, dt):
  xyz = eci2ecef(xyz, dt)
  return ecef2ned(xyz, lla0)


# === ECI2ENU ===
# Earth-Centered-Inertial to East-North-Up coordinates
#
# INPUTS:
#   eny     3x1     ENU x,y,z coordinates [m]
#   lla0    3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
@njit(cache=True, fastmath=True)
def eci2enu(xyz, lla0, dt):
  xyz = eci2ecef(xyz, dt)
  return ecef2enu(xyz, lla0)


# === ECI2AER ===
# Converts Earth-Centered-Inertial to Azimuth-Elevation-Range coordinates
#
# INPUTS:
#   eci_t   3x1   target ECI coordinates
#   eci_r   3x1   reference ECI coordinates
#
# OUTPUTS:
#   aer     3x1   relative AER from reference to target
#
@njit(cache=True, fastmath=True)
def eci2aer(eci_t, eci_r, dt):
  lla0 = eci2lla(eci_r, dt)
  return enu2aer(eci2enu(eci_t, lla0, dt), eci2enu(eci_r, lla0, dt))


#--------------------------------------------------------------------------------------------------#
# === ECEF2ECI ===
# Earth-Centered-Earth-Fixed to Earth-Centered-Inertial coordinates
#
# INPUTS:
#   xyz     3x1     ECEF x,y,z coordinates [m]
#   dt      double  time [s]
#
# OUTPUTS:
#   xyz     3x1     ECI x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def ecef2eci(xyz, dt):
  C_e_i = ecef2eciDcm(dt)
  return C_e_i @ xyz


# === ECEF2LLA ===
# Earth-Centered-Earth-Fixed to Latitude-Longitude-Height coordinates
#  - (Groves Appendix C) Borkowski closed form exact solution
#
# INPUTS:
#   xyz     3x1     ECEF x,y,z [m]
#
# OUTPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
@njit(cache=True, fastmath=True)
def ecef2lla(xyz):
  x,y,z = xyz
  
  beta = np.hypot(x, y)                                                           # (Groves C.18)
  a = np.sqrt(1 - e2) * np.abs(z)
  b = e2 * Ro
  E = (a - b) / beta                                                              # (Groves C.29)
  F = (a + b) / beta                                                              # (Groves C.30)
  P = 4/3 * (E*F + 1)                                                             # (Groves C.31)
  Q = 2 * (E*E - F*F)                                                             # (Groves C.32)
  D = P*P*P + Q*Q                                                                   # (Groves C.33)
  V = (np.sqrt(D) - Q)**(1/3) - (np.sqrt(D) + Q)**(1/3)                             # (Groves C.34)
  G = 0.5 * (np.sqrt(E*E + V) + E)                                                # (Groves C.35)
  T = np.sqrt( G*G + ((F - V*G) / (2*G - E)) ) - G                                # (Groves C.36)
  phi = np.sign(z) * np.arctan( (1 - T*T) / (2*T*np.sqrt(1 - e2)) )               # (Groves C.37)
  h = (beta - Ro*T)*np.cos(phi) + (z - np.sign(z)*Ro*np.sqrt(1 - e2))*np.sin(phi) # (Groves C.38)

  # combine lla
  lamb = np.arctan2(y, x)
  lla = np.array([phi, lamb, h], dtype=np.double)
  
  return np.array([phi, lamb, h], dtype=np.double)


# === ECEF2NED ===
# Earth-Centered-Earth-Fixed to North-East-Down coordinates
#
# INPUTS:
#   xyz     3x1     ECEF x,y,z [m]
#
# OUTPUTS:
#   ned     3x1     NED x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def ecef2ned(xyz, lla0):
  C_e_n = ecef2nedDcm(lla0)
  xyz0 = lla2ecef(lla0)
  return C_e_n @ (xyz - xyz0)


# === ECEF2ENU ===
# Earth-Centered-Earth-Fixed to East-North-Up coordinates
#
# INPUTS:
#   xyz     3x1     ECEF x,y,z [m]
#
# OUTPUTS:
#   enu     3x1     ENU x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def ecef2enu(xyz, lla0):
  C_e_n = ecef2enuDcm(lla0)
  xyz0 = lla2ecef(lla0)
  return C_e_n @ (xyz - xyz0)


# === ECEF2AER ===
# Converts Earth-Centered-Earth-Fixed to Azimuth-Elevation-Range coordinates
#
# INPUTS:
#   ecef_t  3x1   target ECEF coordinates
#   ecef_r  3x1   reference ECEF coordinates
#
# OUTPUTS:
#   aer     3x1   relative AER from reference to target
#
@njit(cache=True, fastmath=True)
def ecef2aer(ecef_t, ecef_r):
  lla0 = ecef2lla(ecef_r)
  return enu2aer(ecef2enu(ecef_t, lla0), ecef2enu(ecef_r, lla0))


#--------------------------------------------------------------------------------------------------#
# === NED2ECI ===
# North-East-Down to Earth-Centered-Inertial coordinates
#
# INPUTS:
#   ned     3x1     NED x,y,z [m]
#
# OUTPUTS:
#   xyz     3x1     ECI x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def ned2eci(ned, lla0, dt):
  xyz = ned2ecef(ned, lla0)
  C_e_i = ecef2eciDcm(dt)
  return C_e_i @ xyz


# === NED2ECEF ===
# North-East-Down to Earth-Centered-Earth-Fixed coordinates
#
# INPUTS:
#   ned     3x1     NED x,y,z [m]
#
# OUTPUTS:
#   xyz     3x1     ECI x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def ned2ecef(ned, lla0):
  C_n_e = ned2ecefDcm(lla0)
  xyz = lla2ecef(lla0)
  return xyz + C_n_e @ ned
  

# === NED2LLA ===
# North-East-Down to Latitude-Longitude-Height coordinates
#
# INPUTS:
#   ned     3x1     NED x,y,z [m]
#
# OUTPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
@njit(cache=True, fastmath=True)
def ned2lla(ned, lla0):
  xyz = ned2ecef(ned, lla0)
  return ecef2lla(xyz)


# === NED2AER ===
# Converts North-East-Down to Azimuth-Elevation-Range coordinates
#
# INPUTS:
#   ned_t   3x1   target NED coordinates
#   ned_r   3x1   reference NED coordinates
#
# OUTPUTS:
#   aer     3x1   relative AER from reference to target
#
@njit(cache=True, fastmath=True)
def ned2aer(ned_t, ned_r):
  dn, de, dd = ned_t - ned_r

  r = np.hypot(de, dn)
  az = np.mod(np.arctan2(de, dn), 2*np.pi)
  el = np.arctan2(-dd, r)
  rng = np.hypot(r, -dd)
  
  return np.array([az, el, rng], dtype=np.double)


#--------------------------------------------------------------------------------------------------#
# === ENU2ECI ===
# East-North-Up to Earth-Centered-Inertial coordinates
#
# INPUTS:
#   enu     3x1     ENU x,y,z [m]
#
# OUTPUTS:
#   xyz     3x1     ECI x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def enu2eci(enu, lla0, dt):
  xyz = enu2ecef(enu, lla0)
  C_e_i = ecef2eciDcm(dt)
  return C_e_i @ xyz


# === ENU2ECEF ===
# East-North-Up to Earth-Centered-Earth-Fixed coordinates
#
# INPUTS:
#   enu     3x1     ENU x,y,z [m]
#
# OUTPUTS:
#   xyz     3x1     ECI x,y,z coordinates [m]
#
@njit(cache=True, fastmath=True)
def enu2ecef(enu, lla0):
  C_n_e = enu2ecefDcm(lla0)
  xyz = lla2ecef(lla0)
  return xyz + C_n_e @ enu
  

# === ENU2LLA ===
# East-North-Up to Latitude-Longitude-Height coordinates
#
# INPUTS:
#   enu     3x1     ENU x,y,z [m]
#
# OUTPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
@njit(cache=True, fastmath=True)
def enu2lla(enu, lla0):
  xyz = enu2ecef(enu, lla0)
  return ecef2lla(xyz)


# === ENU2AER ===
# Converts East-North-Up to Azimuth-Elevation-Range coordinates
#
# INPUTS:
#   enu_t   3x1   target ENU coordinates
#   enu_r   3x1   reference ENU coordinates
#
# OUTPUTS:
#   aer     3x1   relative AER from reference to target
#
@njit(cache=True, fastmath=True)
def enu2aer(enu_t, enu_r):
  de, dn, du = enu_t - enu_r

  r = np.hypot(de, dn)
  az = np.mod(np.arctan2(de, dn), 2*np.pi)
  el = np.arctan2(du, r)
  rng = np.hypot(r, du)
  
  return np.array([az, el, rng], dtype=np.double)
