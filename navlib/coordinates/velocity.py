'''
|================================ navlib/coordinates/velcoity.py ==================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/coordinate/velocity.py                                                         |
|  @brief    Common velocity coordinate frame transformations.                                     |
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
from navlib.constants import OMEGA_ie

#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFV ===
# Converts Earth-Centered-Inertial to Earth-Centered-Earth-Fixed velocity
#
# INPUTS:
#   r_ib_i  3x1     ECI x,y,z position [m]
#   v_ib_i  3x1     ECI x,y,z velocity [m/s]
#   dt      double  time [s]
#
# OUTPUTS:
#   v_eb_e  3x1     ECEF x,y,z velocity [m/s]
#
@njit(cache=True, fastmath=True)
def eci2ecefv(r_ib_i, v_ib_i, dt):
  C_i_e = eci2ecefDcm(dt)
  return C_i_e @ (v_ib_i - OMEGA_ie @ r_ib_i)


# === ECI2NEDV ===
# Converts Earth-Centered-Inertial to North-East-Down velocity
#
# INPUTS:
#   r_ib_i  3x1     ECI x,y,z position [m]
#   v_ib_i  3x1     ECI x,y,z velocity [m/s]
#   dt      double  time [s]
#
# OUTPUTS:
#   v_nb_e  3x1     NED x,y,z velocity [m/s]
#
@njit(cache=True, fastmath=True)
def eci2nedv(r_ib_i, v_ib_i, lla0, dt):
  C_i_n = eci2nedDcm(lla0, dt)
  return C_i_n @ (v_ib_i - OMEGA_ie @ r_ib_i)


# === ECI2ENUV ===
# Converts Earth-Centered-Inertial to East-North-Up velocity
#
# INPUTS:
#   r_ib_i  3x1     ECI x,y,z position [m]
#   v_ib_i  3x1     ECI x,y,z velocity [m/s]
#   dt      double  time [s]
#
# OUTPUTS:
#   v_nb_e  3x1     ENU x,y,z velocity [m/s]
#
@njit(cache=True, fastmath=True)
def eci2enuv(r_ib_i, v_ib_i, lla0, dt):
  C_i_n = eci2enuDcm(lla0, dt)
  return C_i_n @ (v_ib_i - OMEGA_ie @ r_ib_i)


#--------------------------------------------------------------------------------------------------#
# === ECEF2ECIV ===
# Converts Earth-Centered-Earth-Fixed to Earth-Centered-Inertial acceleration
#
# INPUTS:
#   r_eb_e  3x1     ECEF x,y,z position [m]
#   v_eb_e  3x1     ECEF x,y,z velocity [m/s]
#   dt      double  time [s]
#
# OUTPUTS:
#   a_ib_i  3x1    ECEF x,y,z velocity [m/s]
#
@njit(cache=True, fastmath=True)
def ecef2eciv(r_eb_e, v_eb_e, dt):
  C_e_i = ecef2eciDcm(dt)
  return C_e_i @ (v_eb_e - OMEGA_ie @ r_eb_e)


# === ECEF2NEDV ===
# Converts Earth-Centered-Earth-Fixed to North-East-Down acceleration
#
# INPUTS:
#   v_eb_e  3x1     ECEF x,y,z velocity [m/s]
#
# OUTPUTS:
#   a_nb_n  3x1    NED x,y,z velocity [m/s]
#
@njit(cache=True, fastmath=True)
def ecef2nedv(v_eb_e, lla0):
  C_e_n = ecef2nedDcm(lla0)
  return C_e_n @ v_eb_e


# === ECEF2ENUV ===
# Converts Earth-Centered-Earth-Fixed to East-North-Up acceleration
#
# INPUTS:
#   v_eb_e  3x1     ECEF x,y,z velocity [m/s]
#
# OUTPUTS:
#   v_nb_n  3x1    NED x,y,z velocity [m/s]
#
@njit(cache=True, fastmath=True)
def ecef2enuv(v_eb_e, lla0):
  C_e_n = ecef2enuDcm(lla0)
  return C_e_n @ v_eb_e


#--------------------------------------------------------------------------------------------------#
# === NED2ECIV ===
# Converts North-East-Down to East-Centered-Inertial velocity
#
# INPUTS:
#   r_nb_n  3x1     NED x,y,z position [m/s]
#   v_nb_n  3x1     NED x,y,z velocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   v_ib_i  3x1     ECI x,y,z velocity [m/s]
#
@njit(cache=True, fastmath=True)
def ned2eciv(ned, v_ned, lla0, dt):
  C_n_i = ned2eciDcm(lla0, dt)
  C_e_i = ecef2eciDcm(dt)
  xyz = ned2ecef(ned, lla0)
  return C_n_i @ v_ned + C_e_i @ OMEGA_ie @ xyz


# === NED2ECEFV ===
# Converts North-East-Down to East-Centered-Earth-Fixed velocity
#
# INPUTS:
#   v_nb_n  3x1     NED x,y,z velocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#
# OUTPUTS:
#   v_eb_e  3x1     ECEF x,y,z velocity [m/s]
#
@njit(cache=True, fastmath=True)
def ned2ecefv(v_ned, lla0):
  C_n_e = ned2ecefDcm(lla0)
  return C_n_e @ v_ned


#--------------------------------------------------------------------------------------------------#
# === ENU2ECIV ===
# Converts East-North-Up to East-Centered-Inertial velocity
#
# INPUTS:
#   r_nb_n  3x1     ENU x,y,z position [m/s]
#   v_nb_n  3x1     ENU x,y,z velocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   v_ib_i  3x1     ECI x,y,z velocity [m/s]
#
@njit(cache=True, fastmath=True)
def enu2eciv(r_nb_n, v_nb_n, lla0, dt):
  C_n_i = enu2eciDcm(lla0, dt)
  C_e_i = ecef2eciDcm(dt)
  r_eb_e = enu2ecef(r_nb_n, lla0)
  return C_n_i @ v_nb_n + C_e_i @ OMEGA_ie @ r_eb_e


# === ENU2ECEFV ===
# Converts East-North-Up to East-Centered-Earth-Fixed velocity
#
# INPUTS:
#   v_nb_n  3x1     ENU x,y,z velocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#
# OUTPUTS:
#   v_eb_e  3x1     ECEF x,y,z velocity [m/s]
#
@njit(cache=True, fastmath=True)
def enu2ecefv(v_nb_n, lla0):
  C_n_e = enu2ecefDcm(lla0)
  return C_n_e @ v_nb_n
