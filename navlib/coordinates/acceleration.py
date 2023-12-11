'''
|=============================== navlib/coordinates/acceleration.py ===============================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/coordinate/acceleration.py                                                     |
|  @brief    Common acceleration coordinate frame transformations.                                 |
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
from .position import *
from .velocity import *
from navlib.constants import OMEGA_ie, w_ie
from navlib.attitude.skew import skew


#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFA ===
# Converts Earth-Centered-Inertial to Earth-Centered-Earth-Fixed acceleration
#
# INPUTS:
#   r_ib_i  3x1     ECI x,y,z position [m]
#   v_ib_i  3x1     ECI x,y,z velocity [m/s]
#   a_ib_i  3x1     ECI x,y,z acceleration [m/s]
#   dt      double  time [s]
#
# OUTPUTS:
#   a_ib_i  3x1     ECEF x,y,z acceleration [m/s]
#
@njit(cache=True, fastmath=True)
def eci2ecefa(r_ib_i, v_ib_i, a_ib_i, dt):
  C_i_e = eci2ecefDcm(dt)
  return C_i_e @ (a_ib_i - 2 @ OMEGA_ie @ v_ib_i + OMEGA_ie @ OMEGA_ie @ r_ib_i)


# === ECI2NEDA ===
# Converts Earth-Centered-Inertial to North-East-Down acceleration
#
# INPUTS:
#   r_ib_i  3x1     ECI x,y,z position [m]
#   v_ib_i  3x1     ECI x,y,z velocity [m/s]
#   a_ib_i  3x1     ECI x,y,z acceleration [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   a_nb_n  3x1     NED x,y,z acceleration [m/s]
#
@njit(cache=True, fastmath=True)
def eci2neda(r_ib_i, v_ib_i, a_ib_i, lla0, dt):
  C_i_n = eci2nedDcm(lla0, dt)
  return C_i_n @ (a_ib_i + 2 @ OMEGA_ie @ v_ib_i + OMEGA_ie @ OMEGA_ie @ r_ib_i)


# === ECI2ENUA ===
# Converts Earth-Centered-Inertial to East-North-Up acceleration
#
# INPUTS:
#   r_ib_i  3x1     ECI x,y,z position [m]
#   v_ib_i  3x1     ECI x,y,z velocity [m/s]
#   a_ib_i  3x1     ECI x,y,z acceleration [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   a_nb_n  3x1     NED x,y,z acceleration [m/s]
#
@njit(cache=True, fastmath=True)
def eci2enua(r_ib_i, v_ib_i, a_ib_i, lla0, dt):
  C_i_n = eci2enuDcm(lla0, dt)
  return C_i_n @ (a_ib_i + 2 @ OMEGA_ie @ v_ib_i + OMEGA_ie @ OMEGA_ie @ r_ib_i)


#--------------------------------------------------------------------------------------------------#
# === ECEF2ECIA ===
# Converts Earth-Centered-Earth-Fixed to Earth-Centered-Inertial acceleration
#
# INPUTS:
#   r_eb_e  3x1     ECEF x,y,z position [m]
#   v_eb_e  3x1     ECEF x,y,z velocity [m/s]
#   a_eb_e  3x1     ECEF x,y,z acceleration [m/s]
#   dt      double  time [s]
#
# OUTPUTS:
#   a_ib_i  3x1    ECEF x,y,z acceleration [m/s]
#
@njit(cache=True, fastmath=True)
def ecef2ecia(r_eb_e, v_eb_e, a_eb_e, dt):
  C_i_e = eci2ecefDcm(dt)
  return C_i_e @ (a_eb_e - 2 @ OMEGA_ie @ v_eb_e + OMEGA_ie @ OMEGA_ie @ r_eb_e)


# === ECEF2NEDA ===
# Converts Earth-Centered-Earth-Fixed to North-East-Down acceleration
#
# INPUTS:
#   a_eb_e  3x1     ECEF x,y,z acceleration [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#
# OUTPUTS:
#   a_nb_n  3x1    NED x,y,z acceleration [m/s]
#
@njit(cache=True, fastmath=True)
def ecef2neda(a_eb_e, lla0):
  C_e_n = ecef2nedDcm(lla0)
  return C_e_n @ a_eb_e


# === ECEF2ENUA ===
# Converts Earth-Centered-Earth-Fixed to East-North-Up acceleration
#
# INPUTS:
#   a_eb_e  3x1     ECEF x,y,z acceleration [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#
# OUTPUTS:
#   a_nb_n  3x1    NED x,y,z acceleration [m/s]
#
@njit(cache=True, fastmath=True)
def ecef2neda(a_eb_e, lla0):
  C_e_n = ecef2enuDcm(lla0)
  return C_e_n @ a_eb_e


#--------------------------------------------------------------------------------------------------#
# === NED2ECIA ===
# Converts North-East-Down to East-Centered-Inertial acceleration
#
# INPUTS:
#   r_nb_n  3x1     NED x,y,z position [m/s]
#   v_nb_n  3x1     NED x,y,z velocity [m/s]
#   a_nb_n  3x1     NED x,y,z acceleration [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   a_ib_i  3x1    NED x,y,z acceleration [m/s]
#
@njit(cache=True, fastmath=True)
def ned2ecia(r_nb_n, v_nb_n, a_nb_n, lla0, dt):
  r_eb_e = ned2ecef(r_nb_n, lla0)
  C_n_e = ned2ecefDcm(lla0)
  OMEGA_ie_n = skew(np.array([w_ie*np.cos(lla0[0]), 0.0, -w_ie*np.sin(lla0[0])]))
  C_n_i = ned2eciDcm(lla0, dt)
  C_e_i = ecef2eciDcm(lla0, dt)
  a_eb_n = C_n_e @ a_nb_n
  v_eb_n = C_n_e @ v_nb_n
  return C_n_i @ (a_eb_n + 2 @ OMEGA_ie_n @ v_eb_n) + C_e_i @ OMEGA_ie @ OMEGA_ie @ r_eb_e


# === NED2ECEFA ===
# Converts North-East-Down to East-Centered-Earth-Fixed acceleration
#
# INPUTS:
#   a_nb_n  3x1     NED x,y,z acceleration [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#
# OUTPUTS:
#   a_eb_e  3x1    ECEF x,y,z acceleration [m/s]
#
@njit(cache=True, fastmath=True)
def ned2ecefa(a_nb_n, lla0):
  C_n_e = ned2ecefDcm(lla0)
  return C_n_e @ a_nb_n


#--------------------------------------------------------------------------------------------------#
# === ENU2ECIA ===
# Converts East-North-Up to East-Centered-Inertial acceleration
#
# INPUTS:
#   r_nb_n  3x1     ENU x,y,z position [m/s]
#   v_nb_n  3x1     ENU x,y,z velocity [m/s]
#   a_nb_n  3x1     ENU x,y,z acceleration [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   a_ib_i  3x1     ECI x,y,z acceleration [m/s]
#
@njit(cache=True, fastmath=True)
def enu2ecia(r_nb_n, v_nb_n, a_nb_n, lla0, dt):
  r_eb_e = enu2ecef(r_nb_n, lla0)
  C_n_e = enu2ecefDcm(lla0)
  OMEGA_ie_n = skew(np.array([w_ie*np.cos(lla0[0]), 0.0, -w_ie*np.sin(lla0[0])]))
  C_n_i = enu2eciDcm(lla0, dt)
  C_e_i = ecef2eciDcm(lla0, dt)
  a_eb_n = C_n_e @ a_nb_n
  v_eb_n = C_n_e @ v_nb_n
  return C_n_i @ (a_eb_n + 2 @ OMEGA_ie_n @ v_eb_n) + C_e_i @ OMEGA_ie @ OMEGA_ie @ r_eb_e


# === ENU2ECEFA ===
# Converts East-North-Up to East-Centered-Earth-Fixed acceleration
#
# INPUTS:
#   a_nb_n  3x1     ENU x,y,z acceleration [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#
# OUTPUTS:
#   a_eb_e  3x1     ECEF x,y,z acceleration [m/s]
#
@njit(cache=True, fastmath=True)
def enu2ecefa(a_nb_n, lla0):
  C_n_e = enu2ecefDcm(lla0)
  return C_n_e @ a_nb_n
