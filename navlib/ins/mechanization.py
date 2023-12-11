'''
|================================== navlib/ins/mechanization.py ===================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/ins/mechanization.py                                                           |
|  @brief    IMU mechanization                                                                     |
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
from navlib.constants import w_ie, OMEGA_ie, I3
from navlib.gravity.gravity import ned2ecefg, nedg
from navlib.gravity.rotations import earthRate, transportRate
from navlib.gravity.radii_of_curvature import radiiOfCurvature


# === MECHANIZATION_ECEF ===
# Mechanizes body frame imu measurements into the ECEF frame attitude/velocity/position
# INPUTS:
#   w_ib_b    3x1     corrected imu angular velocity
#   f_ib_b    3x1     corrected imu specific force
#   C_old     3x3     old ecef attitude dcm
#   V_old     3x1     old ecef velocity [m/s]
#   P_old     3x1     old ecef position [m]
#   dt        double  delta time
# OUTPUTS:
#   C_new     3x3     new ecef attitude dcm
#   V_new     3x1     new ecef velocity
#   P_new     3x1     new ecef position
#
@njit(cache=True, fastmath=True)
def mechanization_ecef(f_ib_b, w_ib_b, C_old, V_old, P_old, dt):
  # (Groves 2.145) determine earth rotation over update interval
  alpha_ie = w_ie * dt
  sin_aie = np.sin(alpha_ie)
  cos_aie = np.cos(alpha_ie)
  C_earth = np.array([[ cos_aie, sin_aie, 0.0], \
                      [-sin_aie, cos_aie, 0.0], \
                      [     0.0,     0.0, 1.0]])
  
  # attitude increment
  alpha = w_ib_b * dt
  alpha_n = np.linalg.norm(alpha) # norm of alpha
  Alpha = skew(alpha)             # skew symmetric form of alpha
  
  # (Groves 5.73) obtain dcm from new->old attitude
  sina = np.sin(alpha_n)
  cosa = np.cos(alpha_n)
  if alpha_n > 1.0e-8:
    C_new_old = I3 + (sina / alpha_n * Alpha) + \
                      ((1 - cosa) / alpha_n**2 * Alpha @ Alpha)
  else:
    C_new_old = I3 + Alpha
  
  # (Groves 5.75) attitude update
  C_new = C_earth @ C_old @ C_new_old

  # (Groves 5.84/5.85) calculate average body-to-ECEF dcm
  Alpha_ie = skew(np.array([0.0, 0.0, alpha_ie]))
  if alpha_n > 1.0e-8:
    C_avg = C_old @ (I3 + ((1 - cosa) / alpha_n**2 \
                  * Alpha) + ((1 - sina / alpha_n) / alpha_n**2 \
                  * Alpha @ Alpha)) - (0.5 * Alpha_ie @ C_old)
  else:
    C_avg = C_old - 0.5 * Alpha_ie @ C_old
    
  # (Groves 5.85) specific force transformation body-to-ECEF
  f_ib_e = C_avg @ f_ib_b

  # (Groves 5.36) velocity update
  gravity,_ = ned2ecefg(P_old) # use this because imu is simulated this way
  V_new = V_old + dt * (f_ib_e + gravity - 2.0*OMEGA_ie@V_old)

  # (Groves 5.38) position update
  P_new = P_old + 0.5 * dt * (V_new + V_old)
  
  return C_new, V_new, P_new


# === MECHANIZATION_NED ===
# Mechanizes body frame imu measurements into the NED frame attitude/velocity/position
# INPUTS:
#   w_ib_b    3x1     corrected imu angular velocity
#   f_ib_b    3x1     corrected imu specific force
#   C_old     3x3     old NED attitude dcm
#   V_old     3x1     old NED velocity [m/s]
#   P_old     3x1     old NED position (lla) [rad, rad, m]
#   dt        double  delta time
# OUTPUTS:
#   C_new     3x3     new NED attitude dcm
#   V_new     3x1     new NED velocity
#   P_new     3x1     new NED position
#
def mechanization_ned(w_ib_b, f_ib_b, C_old, V_old, P_old, dt):
  # determine earth's rotation rate and transport rate of ECEF frame w.r.t NAV frame
  Rn_old, Re_old, _ = radiiOfCurvature(P_old[0])
  W_en_n = skew(np.array([ V_old[1] / (Re_old + P_old[2]), \
                          -V_old[0] / (Rn_old + P_old[2]), \
                          -V_old[1] * np.tan(P_old[0]) / (Re_old + P_old[2])], \
                         dtype=np.double))
  W_ie_n = earthRate(P_old)
  
  # attitude increment
  alpha = w_ib_b * dt
  Alpha = skew(alpha)
  alpha_n = np.linalg.norm(alpha)
  
  # (Groves 5.83/84/86) Specific force transformation
  if alpha_n > 1.0e-8:
    C_avg = C_old @ (I3 + (1 - np.cos(alpha_n)) / alpha_n*alpha_n \
                  * Alpha + (1 - np.sin(alpha_n) / alpha_n) / alpha_n*alpha_n \
                  * Alpha @ Alpha) - 0.5 * (W_en_n + W_ie_n) @ C_old
  else:
    C_avg = C_old - 0.5 * (W_en_n + W_ie_n) @ C_old
  f_ib_n = C_avg @ f_ib_b
  
  # (Groves 5.54) velocity update
  g = nedg(P_old)
  V_new  = V_old + (f_ib_n + g - (W_en_n + 2*W_ie_n) @ V_old) * dt
  
  # (Groves 5.56) position update
  h = P_old[2] - 0.5 * (V_old[2] + V_new[2]) * dt
  phi = P_old[0] + 0.5 * dt * (V_old[0] / (Rn_old + P_old[2]) + V_new[0] / (Rn_old + h))
  Rn_new, Re_new, _ = radiiOfCurvature(phi)
  lam = P_old[1] + 0.5 * dt * (V_old[1] / ((Re_old + P_old[2]) * np.cos(P_old[0])) \
                 + V_new[1] / ((Re_new + h) * np.cos(phi)))
  P_new = np.array([phi, lam, h], dtype=np.double)
  
  # New transport rate
  W_en_n_new = skew(np.array([ V_new[1] / (Re_new + h), \
                              -V_new[0] / (Rn_new + h), \
                              -V_new[1] * np.tan(phi) / (Re_new + h)], \
                             dtype=np.double))
  
  # (Groves 5.73/76/77)
  if alpha_n > 1.e-8:
    C_bar = I3 + np.sin(alpha_n) / alpha_n * Alpha \
               + (1 - np.cos(alpha_n)) / alpha_n*alpha_n * Alpha @ Alpha
  else:
    C_bar = I3 + Alpha
  C_new = (I3 - (W_ie_n + 0.5*W_en_n_new + 0.5*W_en_n)  * dt) @ C_old @ C_bar
  
  return C_new, V_new, P_new
  