'''
|================================ navlib/gnss/kf_measurement.py ===================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/gnss/kf_measurement.py                                                         |
|  @brief    GNSS measurement domain kalman filter.                                                |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from navlib.constants import I3, Z3

I6 = np.eye(6, dtype=np.double)


# === PREDICT ===
# GNSS position/velocity Kalman filter prediction step (position domain)
#
# INPUTS:
#   kf_x    6x1     state +
#   kf_P    6x6     state covariance +
#   S_a     double  PSD of acceleration [m^2/s^3]
#   dt      double  integration time [s]
#
# OUTPUTS:
#   kf_x    6x1     state -
#   kf_P    6x6     state covariance -
#
@njit(cache=True, fastmath=True)
def predict(kf_x, kf_P, S_a, dt):
  # state transition and process noise matrices
  A, Q = __gen_transition(S_a, dt)
  
  # Kalman prediction
  kf_x = A @ kf_x
  kf_P = A @ kf_P @ A.T + Q
  return kf_x, kf_P


# === CORRECT ===
# GNSS position/velocity Kalman filter correction step (position domain)
#
# INPUTS:
#   kf_x      6x1     state -
#   kf_P      6x6     state covariance -
#   gnss_p    3x1     GNSS position measurement [m]
#   gnss_v    3x1     GNSS velocity measurement [m/s]
#   S_pos     3x1     GNSS position variance [m^2]
#   S_vel     3x1     GNSS velocity variance [m^2]
#   dt        double  integration time [s]
#
# OUTPUTS:
#   kf_x      6x1     state +
#   kf_P      6x6     state covariance +
#
@njit(cache=True, fastmath=True)
def correct(kf_x, kf_P, gnss_p, gnss_v, S_pos, S_vel):
  # measurement innovation
  dy = np.concatenate((gnss_p - kf_x[:3], \
                       gnss_v - kf_x[3:]))
  
  # observation matrix
  C = I6
  
  # observation covariance matrix
  R = np.diag(np.concatenate((S_pos, S_vel)))
  
  # Kalman update
  PCt = kf_P @ C.T
  L = PCt @ np.linalg.inv(C @ kf_P @ C.T + R)
  kf_P = (I6 - L @ C) @ kf_P
  kf_x = kf_x + L @ dy
  return kf_x, kf_P


# === __GEN_TRANSITION ===
# Generates state transition matrix and process noise covariance for a GNSS position/velocity 
# Kalman filter (position domain)
#
# INPUTS:
#   S_a     double  PSD of acceleration [m^2/s^3]
#   dt      double  integration time [s]
#
# OUTPUTS:
#   A       6x6     state transition matrix
#   Q       6x6     process noise covariance
#
@njit(cache=True, fastmath=True)
def __gen_transition(S_a, dt):
  # (Groves 9.148) state transition matrix
  A12 = dt * I3
  A = np.vstack((np.hstack((I3, A12)), \
                 np.hstack((Z3,  I3))), \
                )
  
  # (Groves 9.152) process noise covariance
  q11 = 1/3 * S_a * dt**3 * I3
  q12 = 1/2 * S_a * dt**2 * I3
  q22 = S_a * dt * I3
  Q = np.vstack((np.hstack((q11, q12)), \
                 np.hstack((q12, q22))), \
                )
  
  return A, Q
