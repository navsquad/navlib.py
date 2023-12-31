'''
|=================================== navlib/coordinates/dcm.py ====================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/coordinate/tangent.py                                                          |
|  @brief    Common coordinate frame transformation matrices.                                      |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from navlib.constants import w_ie

#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFDCM ===
# Earth-Centered-Inertial to Earth-Centered-Earth-Fixed direction cosine matrix
#
# INPUTS:
#   dt      double  time [s]
#
# OUTPUTS:
#   C_i_e   3x3     ECI->ECEF direction cosine matrix
#
@njit(cache=True, fastmath=True)
def eci2ecefDcm(dt):
  sin_wie = np.sin(w_ie*dt)
  cos_wie = np.cos(w_ie*dt)
  # Groves 2.145
  C_i_e = np.array([[ cos_wie, sin_wie, 0.0], \
                    [-sin_wie, cos_wie, 0.0], \
                    [     0.0,     0.0, 1.0]], \
                   dtype=np.double)
  return C_i_e


# === ECI2NEDDCM ===
# Earth-Centered-Inertial to North-East-Down direction cosine matrix
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   C_i_n   3x3     ECI->NAV direction cosine matrix
#
@njit(cache=True, fastmath=True)
def eci2nedDcm(lla, dt):
  sinPhi = np.sin(lla[0])
  cosPhi = np.cos(lla[0])
  sin_lam_wie = np.sin(lla[1] + w_ie*dt)
  cos_lam_wie = np.cos(lla[1] + w_ie*dt)
  # Groves 2.154
  C_i_n = np.array([[-sinPhi*cos_lam_wie, -sinPhi*sin_lam_wie,  cosPhi], \
                    [       -sin_lam_wie,         cos_lam_wie,     0.0], \
                    [-cosPhi*cos_lam_wie, -cosPhi*sin_lam_wie, -sinPhi]], 
                   dtype=np.double)
  return C_i_n


# === ECI2NEDDCM ===
# Earth-Centered-Inertial to East-North-Up direction cosine matrix
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   C_i_n   3x3     ECI->NAV direction cosine matrix
#
@njit(cache=True, fastmath=True)
def eci2enuDcm(lla, dt):
  sinPhi = np.sin(lla[0])
  cosPhi = np.cos(lla[0])
  sin_lam_wie = np.sin(lla[1] + w_ie*dt)
  cos_lam_wie = np.cos(lla[1] + w_ie*dt)
  C_i_n = np.array([[       -sin_lam_wie,         cos_lam_wie,    0.0], \
                    [-sinPhi*cos_lam_wie, -sinPhi*sin_lam_wie, cosPhi], \
                    [ cosPhi*cos_lam_wie,  cosPhi*sin_lam_wie, sinPhi]], 
                   dtype=np.double)
  return C_i_n


#--------------------------------------------------------------------------------------------------#
# === ECEF2ECIDCM ===
# Earth-Centered-Earth-Fixed to Earth-Centered-Inertial direction cosine matrix
#
# INPUTS:
#   dt      double  time [s]
#
# OUTPUTS:
#   C_e_i   3x3     ECEF->ECI direction cosine matrix
#
@njit(cache=True, fastmath=True)
def ecef2eciDcm(dt):
  sin_wie = np.sin(w_ie*dt)
  cos_wie = np.cos(w_ie*dt)
  # Groves 2.145
  C_e_i = np.array([[cos_wie, -sin_wie, 0.0], \
                    [sin_wie,  cos_wie, 0.0], \
                    [    0.0,      0.0, 1.0]], \
                   dtype=np.double)
  return C_e_i


# === ECEF2NEDDCM ===
# Earth-Centered-Earth-Fixed to North-East-Down direction cosine matrix
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
# OUTPUTS:
#   C_e_n   3x3     ECEF->NAV direction cosine matrix
#
@njit(cache=True, fastmath=True)
def ecef2nedDcm(lla):
  sinPhi, sinLam = np.sin(lla[:2])
  cosPhi, cosLam = np.cos(lla[:2])
  # Groves 2.150
  C_e_n = np.array([[-sinPhi*cosLam, -sinPhi*sinLam,  cosPhi], \
                    [       -sinLam,         cosLam,     0.0], \
                    [-cosPhi*cosLam, -cosPhi*sinLam, -sinPhi]], \
                   dtype=np.double)
  return C_e_n


# === ECEF2ENUDCM ===
# Earth-Centered-Earth-Fixed to East-North-Up direction cosine matrix
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
# OUTPUTS:
#   C_e_n   3x3     ECEF->NAV direction cosine matrix
#
@njit(cache=True, fastmath=True)
def ecef2enuDcm(lla):
  sinPhi, sinLam = np.sin(lla[:2])
  cosPhi, cosLam = np.cos(lla[:2])
  C_e_n = np.array([[       -sinLam,         cosLam,    0.0], \
                    [-sinPhi*cosLam, -sinPhi*sinLam, cosPhi], \
                    [ cosPhi*cosLam,  cosPhi*sinLam, sinPhi]], \
                   dtype=np.double)
  return C_e_n


#--------------------------------------------------------------------------------------------------#
# === NED2ECIDCM ===
# North-East-Down to Earth-Centered-Inertial direction cosine matrix
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
# OUTPUTS:
#   C_e_i   3x3     NAV->ECI direction cosine matrix
#
@njit(cache=True, fastmath=True)
def ned2eciDcm(lla, dt):
  sinPhi = np.sin(lla[0])
  cosPhi = np.cos(lla[0])
  sin_lam_wie = np.sin(lla[1] + w_ie*dt)
  cos_lam_wie = np.cos(lla[1] + w_ie*dt)
  # Groves 2.154
  C_n_i = np.array([[-sinPhi*cos_lam_wie, -sin_lam_wie, -cosPhi*cos_lam_wie], \
                    [-sinPhi*sin_lam_wie,  cos_lam_wie, -cosPhi*sin_lam_wie], \
                    [             cosPhi,          0.0,             -sinPhi]], \
                   dtype=np.double)
  return C_n_i


# === NED2ECEFDCM ===
# North-East-Down to Earth-Centered-Earth-Fixed direction cosine matrix
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
# OUTPUTS:
#   C_e_n   3x3     NAV->ECEF direction cosine matrix
#
@njit(cache=True, fastmath=True)
def ned2ecefDcm(lla):
  sinPhi, sinLam = np.sin(lla[:2])
  cosPhi, cosLam = np.cos(lla[:2])
  # Groves 2.150
  C_n_e = np.array([[-sinPhi*cosLam, -sinLam, -cosPhi*cosLam], \
                    [-sinPhi*sinLam,  cosLam, -cosPhi*sinLam], \
                    [        cosPhi,     0.0,        -sinPhi]], \
                   dtype=np.double)
  return C_n_e


#--------------------------------------------------------------------------------------------------#
# === ENU2ECIDCM ===
# East-North-Up to Earth-Centered-Inertial direction cosine matrix
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
# OUTPUTS:
#   C_e_i   3x3     NAV->ECI direction cosine matrix
#
@njit(cache=True, fastmath=True)
def enu2eciDcm(lla, dt):
  sinPhi = np.sin(lla[0])
  cosPhi = np.cos(lla[0])
  sin_lam_wie = np.sin(lla[1] + w_ie*dt)
  cos_lam_wie = np.cos(lla[1] + w_ie*dt)
  # Groves 2.154
  C_n_i = np.array([[-sin_lam_wie, -sinPhi*cos_lam_wie, cosPhi*cos_lam_wie], \
                    [ cos_lam_wie, -sinPhi*sin_lam_wie, cosPhi*sin_lam_wie], \
                    [         0.0,              cosPhi,             sinPhi]], \
                   dtype=np.double)
  return C_n_i


# === ENU2ECEFDCM ===
# East-North-Up to Earth-Centered-Earth-Fixed direction cosine matrix
#
# INPUTS:
#   lla     3x1     Geodetic Latitude, Longitude, Height [rad, rad, m]
#
# OUTPUTS:
#   C_e_n   3x3     NAV->ECEF direction cosine matrix
#
@njit(cache=True, fastmath=True)
def enu2ecefDcm(lla):
  sinPhi, sinLam = np.sin(lla[:2])
  cosPhi, cosLam = np.cos(lla[:2])
  C_n_e = np.array([[-sinLam, -cosLam*sinPhi, cosLam*cosPhi], \
                    [ cosLam, -sinLam*sinPhi, sinLam*cosPhi], \
                    [    0.0,         cosPhi,        sinPhi]], \
                   dtype=np.double)
  return C_n_e
