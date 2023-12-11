'''
|=================================== navlib/gnss/kf_position.py ===================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/gnss/kf_position.py                                                            |
|  @brief    GNSS position domain kalman filter.                                                   |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from numba import njit
from navlib.constants import c, w_ie, OMEGA_ie, I3, Z3, Z32, Z23

I8 = np.eye(8, dtype=np.double)


# === PREDICT ===
# GNSS pseudorange/pseudorange-rate Kalman filter prediction step (measurement domain)
#
# INPUTS:
#   kf_x    8x1     state +
#   kf_P    8x8     state covariance +
#   S_a     double  PSD of acceleration [m^2/s^3]
#   S_p     double  PSD of clock phase [m^2/s]
#   S_f     double  PSD of clock frequency [m^2/s^3]
#   dt      double  integration time [s]
#
# OUTPUTS:
#   kf_x    8x1     state -
#   kf_P    8x8     state covariance -
#
@njit(cache=True, fastmath=True)
def predict(kf_x, kf_P, S_a, S_p, S_f, dt):
  # state transition and process noise matrices
  A, Q = __gen_transition(S_a, S_p, S_f, dt)
  
  # Kalman prediction
  kf_x = A @ kf_x
  kf_P = A @ kf_P @ A.T + Q
  return kf_x, kf_P


# === CORRECT ===
# GNSS pseudorange/pseudorange-rate Kalman filter correction step (measurement domain)
#
# INPUTS:
#   kf_x      8x1     state -
#   kf_P      8x8     state covariance -
#   sv_p      Nx1     GNSS satellite position [m]
#   sv_v      Nx1     GNSS satellite velocity [m/s]
#   psr       Nx1     pseudorange measurements [m]
#   psrdot    Nx1     pseudorange-rate measurements [m/s]
#   S_psr     Nx1     pseudorange variance [m^2]
#   S_psrdot  Nx1     pseudorange-rate variance [m^2]
#
# OUTPUTS:
#   kf_x      8x1     state +
#   kf_P      8x8     state covariance +
#
@njit(cache=True, fastmath=True)
def correct(kf_x, kf_P, sv_p, sv_v, psr, psrdot, S_psr, S_psrdot):
  # generate geometry matrix and estimated measurement
  C, psr_hat, psrdot_hat = __gen_estimate(kf_x, sv_p, sv_v)
  
  # (Groves 9.159) innovation
  dy = np.concatenate((psr - psr_hat, \
                       psrdot - psrdot_hat))
  
  # observation covariance matrix
  R = np.diag(np.concatenate((S_psr, S_psrdot)))
  
  # Kalman update
  PCt = kf_P @ C.T
  L = PCt @ np.linalg.inv(C @ kf_P @ C.T + R)
  kf_P = (I8 - L @ C) @ kf_P
  kf_x = kf_x + L @ dy
  return kf_x, kf_P


# === __GEN_TRANSITION ===
# Generates state transition matrix and process noise covariance for a GNSS psuedorange/
# pseudorange-rate Kalman filter (measurement domain)
#
# INPUTS:
#   S_a     double  PSD of acceleration [m^2/s^3]
#   S_p     double  PSD of clock phase [m^2/s]
#   S_f     double  PSD of clock frequency [m^2/s^3]
#   dt      double  integration time [s]
#
# OUTPUTS:
#   A       8x8     state transition matrix
#   Q       8x8     process noise covariance
#
@njit(cache=True, fastmath=True)
def __gen_transition(S_a, S_p, S_f, dt):
  # (Groves 9.148) state transition matrix
  A12 = dt * I3
  A33 = np.array([[1.0,  dt], \
                  [0.0, 1.0]], \
                 dtype=np.double)
  A = np.vstack((np.hstack(( I3, A12, Z32)), \
                 np.hstack(( Z3,  I3, Z32)), \
                 np.hstack((Z23, Z23, A33))), \
               )
  
  # (Groves 9.152) process noise covariance
  q11 = 1/3 * S_a * dt**3 * I3
  q12 = 1/2 * S_a * dt**2 * I3
  q22 = S_a * dt * I3
  q33 = np.array([[S_p*dt + 1/3*S_f*dt**3, 1/2*S_f*dt**2], \
                  [         1/2*S_f*dt**2,        S_f*dt]],\
                 dtype=np.double)
  Q = np.vstack((np.hstack((  q11,   q12, Z32)), \
                 np.hstack((  q12,   q22, Z32)), \
                 np.hstack((Z32.T, Z32.T, q33))), \
               )
  
  return A, Q


# === __GEN_ESTIMATE ===
# Generates the observation matrix and pseudorange/pseudorange-rate measurements
#
# INPUTS:
#   kf_x      8x1     state -
#   sv_p      Nx1     GNSS satellite position [m]
#   sv_v      Nx1     GNSS satellite velocity [m/s]
#
# OUTPUTS:
#   C         2Nx8    observation matrix
#   psr       Nx1     pseudorange measurement estimates [m]
#   psrdot    Nx1     pseudorange-rate measurement estimates [m/s]
#
@njit(cache=True, fastmath=True)
def __gen_estimate(kf_x, sv_p, sv_v):
  # number of total measurements
  N = sv_p.shape[0]
  
  # Preallocate observation matrix
  C = np.zeros((2*N, 8), dtype=np.double)
  C[:N,6] = np.ones(N, dtype=np.double)
  C[N:,7] = np.ones(N, dtype=np.double)
  
  # Preallocate measurement vector
  psr = np.zeros(N, dtype=np.double)
  psrdot = np.zeros(N, dtype=np.double)
  
  # loop through each satellite
  for i in np.arange(N):
    # # (Groves 8.36) approximate ECEF frame rotation during transit time
    # dt = np.linalg.norm(sv_p[i,:] - kf_x[:3]) / c
    # C_e_i = np.array([[     1.0, w_ie*dt, 0.0], \
    #                   [-w_ie*dt,     1.0, 0.0], \
    #                   [     0.0,     0.0, 1.0]], \
    #                  dtype=np.double)
    
    # (Groves 9.165) predicted pseudorange and unit vector
    # dr = C_e_i @ sv_p[i,:] - kf_x[:3]
    dr = sv_p[i,:] - kf_x[:3]
    raw_range = np.linalg.norm(dr)
    Ur = dr / raw_range
    
    # (Groves 9.165) predicted pseudorange-rate and doppler position vector
    # dv = (C_e_i @ (sv_v[i,:] + OMEGA_ie@sv_p[i,:]) - (kf_x[3:6] + OMEGA_ie@kf_x[:3]))
    dv = sv_v[i,:] - kf_x[3:6]
    raw_range_rate = Ur @ dv
    Uv = np.cross(Ur, np.cross(Ur, dv/raw_range))
    
    # (Groves 9.163) add to geometry matrix and estimated measurement vector
    C[i,:3] = -Ur
    # C[i+N,:3] = -Uv
    C[i+N,3:6] = -Ur
    psr[i] = raw_range + kf_x[6]
    psrdot[i] = raw_range_rate + kf_x[7]
    
  return C, psr, psrdot
  
