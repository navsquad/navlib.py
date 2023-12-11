'''
|============================= navlib/coordinates/angular_velocity.py =============================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/coordinate/angular_velocity.py                                                 |
|  @brief    Common angular velocity coordinate frame transformations.                             |
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
from navlib.constants import omega_ie, e2, Ro
from navlib.attitude.skew import skew


#--------------------------------------------------------------------------------------------------#
# === ECI2ECEFW ===
# Converts Earth-Centered-Inertial to Earth-Centered-Earth-Fixed acceleration
#
# INPUTS:
#   w_ib_i  3x1     ECI x,y,z acceleration [m/s]
#   dt      double  time [s]
#
# OUTPUTS:
#   w_ib_i  3x1     ECEF x,y,z acceleration [m/s]
#
def eci2ecefw(w_ib_i, dt):
  C_i_e = eci2ecefDcm(dt)
  return C_i_e @ (w_ib_i + omega_ie)


# === ECI2NEDW ===
# Converts Earth-Centered-Inertial to North-East-Down angular velocity
#
# INPUTS:
#   w_ib_i  3x1     ECI x,y,z angular velocity [m/s]
#   r_ib_i  3x1     ECI x,y,z position [m]
#   v_ib_i  3x1     ECI x,y,z velocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   w_nb_n  3x1     NED x,y,z angular velocity [m/s]
#
def eci2nedw(w_ib_i, r_ib_i, v_ib_i, lla0, dt):
  C_i_n = eci2nedDcm(lla0, dt)
  
  vn, ve, vd = eci2nedv(r_ib_i, v_ib_i, lla0, dt)
  phi, lam, h = lla0
  sinPhi = np.sin(phi)
  
  trans = 1 - e2 * sinPhi*sinPhi
  Re = Ro / np.sqrt(trans)
  Rn = Ro * (1- e2) / trans**1.5
  w_en_n = np.array([ ve / (Re + h), \
                     -vn / (Rn + h), \
                     -ve * np.tan(phi) / (Re + h)], 
                    dtype=np.double)
  
  return C_i_n @ (w_ib_i - omega_ie) - w_en_n


# === ECI2ENUW ===
# Converts Earth-Centered-Inertial to East-North-Up angular velocity
#
# INPUTS:
#   w_ib_i  3x1     ECI x,y,z angular velocity [m/s]
#   r_ib_i  3x1     ECI x,y,z position [m]
#   v_ib_i  3x1     ECI x,y,z velocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   w_nb_n  3x1     NED x,y,z angular velocity [m/s]
#
def eci2enuw(w_ib_i, r_ib_i, v_ib_i, lla0, dt):
  C_i_n = eci2enuDcm(lla0, dt)
  
  ve, vn, vu = eci2enuv(r_ib_i, v_ib_i, lla0, dt)
  phi, lam, h = lla0
  sinPhi = np.sin(phi)
  
  trans = 1 - e2 * sinPhi*sinPhi
  Re = Ro / np.sqrt(trans)
  Rn = Ro * (1- e2) / trans**1.5
  w_en_n = np.array([ ve / (Re + h), \
                     -vn / (Rn + h), \
                     -ve * np.tan(phi) / (Re + h)], 
                    dtype=np.double)
  
  return C_i_n @ (w_ib_i - omega_ie) - w_en_n


#--------------------------------------------------------------------------------------------------#
# === ECEF2ECIW ===
# Converts Earth-Centered-Earth-Fixed to Earth-Centered-Inertial angular velocity
#
# INPUTS:
#   w_eb_e  3x1     ECEF x,y,z angular velocity [m/s]
#   dt      double  time [s]
#
# OUTPUTS:
#   w_ib_i  3x1     ECEF x,y,z angular velocity [m/s]
#
def ecef2eciw(w_eb_e, dt):
  C_e_i = ecef2eciDcm(dt)
  return C_e_i @ (w_eb_e + omega_ie)

# === ECEF2NED ===
# Converts Earth-Centered-Earth-Fixed to North-East-Down angular velocity
#
# INPUTS:
#   w_eb_e  3x1     CEF x,y,z angular velocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#
# OUTPUTS:
#   w_nb_n  3x1     NED x,y,z angular velocity [m/s]
#
def ecef2nedw(w_eb_e, lla0):
  C_e_n = ecef2nedDcm(lla0)
  return C_e_n @ w_eb_e


# === ECEF2NEDW ===
# Converts Earth-Centered-Earth-Fixed to East-North-Up angular velocity
#
# INPUTS:
#   w_eb_e  3x1     ECEF x,y,z angular velocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#
# OUTPUTS:
#   w_nb_n  3x1     ENU x,y,z angular velocity [m/s]
#
def ecef2enuw(w_eb_e, lla0):
  C_e_n = ecef2enuDcm(lla0)
  return C_e_n @ w_eb_e


#--------------------------------------------------------------------------------------------------#
# === NED2ECIW ===
# Converts North-East-Down to East-Centered-Inertial angular velocity
#
# INPUTS:
#   w_nb_n  3x1     NED x,y,z angular velocity [m/s]
#   v_nb_n  3x1     NED x,y,z velocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   a_ib_i  3x1     NED x,y,z angular velocity [m/s]
#
def ned2eciw(w_nb_n, v_nb_n, lla0, dt):
  C_n_i = ned2eciDcm(lla0, dt)
  
  vn, ve, vd = v_nb_n
  phi, lam, h = lla0
  sinPhi = np.sin(phi)
  
  trans = 1 - e2 * sinPhi*sinPhi
  Re = Ro / np.sqrt(trans)
  Rn = Ro * (1- e2) / trans**1.5
  w_en_n = np.array([ ve / (Re + h), \
                     -vn / (Rn + h), \
                     -ve * np.tan(phi) / (Re + h)], 
                    dtype=np.double)
  
  return C_n_i @ (w_nb_n + w_en_n) + omega_ie


# === NED2ECEFW ===
# Converts North-East-Down to East-Centered-Earth-Fixed angular velocity
#
# INPUTS:
#   w_nb_n  3x1     NED x,y,z angular velocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#
# OUTPUTS:
#   w_eb_e  3x1     ECEF x,y,z angular velocity [m/s]
#
def ned2ecefww(w_eb_e, lla0):
  C_n_e = ned2ecefDcm(lla0)
  return C_n_e @ w_eb_e


#--------------------------------------------------------------------------------------------------#
# === ENU2ECIW ===
# Converts East-North-Up to East-Centered-Inertial angular velocity
#
# INPUTS:
#   w_nb_n  3x1     ENU x,y,z angular velocity [m/s]
#   v_nb_n  3x1     ENU x,y,z velocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#   dt      double  time [s]
#
# OUTPUTS:
#   w_ib_i  3x1     ECI x,y,z angular velocity [m/s]
#
def enu2eciw(w_nb_n, v_nb_n, lla0, dt):
  C_n_i = enu2eciDcm(lla0, dt)
  
  ve, vn, vu = v_nb_n
  phi, lam, h = lla0
  sinPhi = np.sin(phi)
  
  trans = 1 - e2 * sinPhi*sinPhi
  Re = Ro / np.sqrt(trans)
  Rn = Ro * (1- e2) / trans**1.5
  w_en_n = np.array([ ve / (Re + h), \
                     -vn / (Rn + h), \
                     -ve * np.tan(phi) / (Re + h)], 
                    dtype=np.double)
  
  return C_n_i @ (w_nb_n + w_en_n) + omega_ie


# === ENU2ECEFW ===
# Converts East-North-Up to East-Centered-Earth-Fixed angular velocity
#
# INPUTS:
#   w_nb_n  3x1     ENU x,y,z angular valocity [m/s]
#   lla0    3x1     LLA [rad, rad, m]
#
# OUTPUTS:
#   w_eb_e  3x1     ECEF x,y,z angular velocity [m/s]
#
def enu2ecefw(w_eb_e, lla0):
  C_n_e = enu2ecefDcm(lla0)
  return C_n_e @ w_eb_e


