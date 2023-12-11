'''
|================================== navlib/gnss-ins/tightly.py ====================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/gnss-ins/kf_tightly.py                                                         |
|  @brief    Tightly coupled GNSS-INS kalman filter                                                |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

# TODO: NAV frame implementation

import numpy as np
from numba import njit
from navlib.attitude.skew import skew
from navlib.coordinates.position import ecef2lla
from navlib.gravity.gravity import ned2ecefg
from navlib.gravity.radii_of_curvature import geocentricRadius
from navlib.constants import I3, Z3, Z32, Z23, OMEGA_ie, w_ie, c 

I17 = np.eye(17)


# === PREDICT ===
# Loosely coupled GNSS-INS prediction step
#
# INPUTS:
#   kf_x      15x1    state -
#   kf_P      15x15   state covariance -
#   f_ib_b    3x1     IMU specific force measurement [m/s**2]
#   C_b_e     3x3     DCM from body to ECEF frame
#   r_eb_e    3x1     IMU ECEF position solution [m]
#   v_eb_e    3x1     IMU ECEF velocity solution [m/s]
#   S_rg      double  IMU gyro bias drift PSD
#   S_ra      double  IMU accel bias drift PSD
#   S_bgd     double  IMU gyro random walk PSD
#   S_bad     double  IMU accel random walk PSD
#   S_p       double  PSD of clock phase [m**2/s]
#   S_f       double  PSD of clock frequency [m**2/s**3]
#
# OUTPUTS:
#   kf_x    15x1      state +
#   kf_P    15x15     state covariance +
#
@njit(cache=True, fastmath=True)
def predict(kf_x, kf_P, f_ib_b, C_b_e, r_eb_e, S_rg, S_ra, S_bgd, S_bad, S_p, S_f, dt):
  # state transition and process noise matrices
  A, Q = __gen_transition(f_ib_b, C_b_e, r_eb_e, S_rg, S_ra, S_bgd, S_bad, S_p, S_f, dt)
  
  # Kalman Prediction
  kf_P = A @ kf_P @ A.T + Q
  kf_x = A @ kf_x
  return kf_x, kf_P


# === CORRECT ===
# Loosely coupled GNSS-INS correction step
#
# INPUTS:
#   kf_x      15x1    state -
#   kf_P      15x15   state covariance -
#   sv_p      Nx1     GNSS satellite position [m]
#   sv_v      Nx1     GNSS satellite velocity [m/s]
#   psr       Nx1     pseudorange measurements [m]
#   psrdot    Nx1     pseudorange-rate measurements [m/s]
#   S_psr     Nx1     pseudorange variance [m**2]
#   S_psrdot  Nx1     pseudorange-rate variance [m**2]
#   w_ib_b    3x1     IMU angular velocity measurement [rad/s]
#   C_b_e     3x3     DCM from body to ECEF frame
#   r_eb_e    3x1     IMU ECEF position solution [m]
#   v_eb_e    3x1     IMU ECEF velocity solution [m/s]
#   L_ba_b    3x1     body frame lever arm from IMU to RCVR [m]
#
# OUTPUTS:
#   kf_x    15x1      state +
#   kf_P    15x15     state covariance +
#
@njit(cache=True, fastmath=True)
def correct(kf_x, kf_P, \
            sv_p, sv_v, psr, psrdot, S_psr, S_psrdot, \
            w_ib_b, C_b_e, r_eb_e, v_eb_e, L_ba_b):
  # generate geometry matrix and estimated measurement
  C, psr_hat, psrdot_hat = __gen_estimate(sv_p, sv_v, \
                                          w_ib_b, C_b_e, r_eb_e, v_eb_e, L_ba_b, \
                                          kf_x[15], kf_x[16])
  
  # (Groves 14.119) innovation
  dy = np.concatenate((psr - psr_hat, \
                       psrdot - psrdot_hat))
  
  # observation covariance matrix
  R = np.diag(np.concatenate((S_psr, S_psrdot)))
  
  # Kalman update
  PCt = kf_P @ C.T
  L = PCt @ np.linalg.inv(C @ kf_P @ C.T + R)
  kf_P = (I17 - L @ C) @ kf_P
  kf_x = kf_x + L @ dy
  return kf_x, kf_P


# === __GEN_TRANSITION ===
# Generates the state transition matrix and process noise covariance
#
# INPUTS:
#   f_ib_b    3x1     IMU specific force measurement [m/s**2]
#   C_b_e     3x3     DCM from body to ECEF frame
#   r_eb_e    3x1     IMU ECEF position solution [m]
#   v_eb_e    3x1     IMU ECEF velocity solution [m/s]
#   S_rg      double  IMU gyro bias drift PSD
#   S_ra      double  IMU accel bias drift PSD
#   S_bgd     double  IMU gyro random walk PSD
#   S_bad     double  IMU accel random walk PSD
#   S_p       double  PSD of clock phase [m**2/s]
#   S_f       double  PSD of clock frequency [m**2/s**3]
#
# OUTPUTS:
#   A         15x15   state transition matrix
#   Q         15x15   process noise covariance
#
@njit(cache=True, fastmath=True)
def __gen_transition(f_ib_b, C_b_e, r_eb_e, S_rg, S_ra, S_bgd, S_bad, S_p, S_f, dt):
  # radii of curvature and gravity
  lla = ecef2lla(r_eb_e)
  r_es_e = geocentricRadius(lla[0])
  _,gamma = ned2ecefg(r_eb_e)
  
  # (Groves 14.18/87) state transition matrix discretization
  f21 = -skew(C_b_e @ f_ib_b)
  
  # (Groves Appendix I)
  F11 = I3 - OMEGA_ie*dt
  F15 = C_b_e*dt - 1/2*OMEGA_ie@C_b_e*dt**2
  F21 = f21*dt - 1/2*f21@OMEGA_ie*dt**2 - OMEGA_ie@f21*dt**2
  F22 = I3 - 2*OMEGA_ie*dt
  F23 = np.outer(-(2 * gamma / r_es_e), (r_eb_e / np.linalg.norm(r_eb_e))) * dt
  F24 = C_b_e*dt - OMEGA_ie@C_b_e*dt**2
  F25 = 1/2*f21@C_b_e*dt**2 - 1/6*f21@OMEGA_ie@C_b_e*dt**3 - 1/3*OMEGA_ie@f21@C_b_e*dt**3
  F31 = 1/2*f21*dt**2 - 1/6*f21@OMEGA_ie*dt**3 - 1/3*OMEGA_ie@f21*dt**3
  F32 = I3*dt - OMEGA_ie*dt**2
  F34 = 1/2*C_b_e*dt**2 - 1/3*OMEGA_ie@C_b_e*dt**3
  F35 = 1/6*f21@C_b_e*dt**3
  F66 = np.array([[1.0,  dt], \
                  [0.0, 1.0]], \
                 dtype=np.double)
  
  A = np.vstack((np.hstack((F11,  Z3,  Z3,  Z3, F15, Z32)), \
                 np.hstack((F21, F22, F23, F24, F25, Z32)), \
                 np.hstack((F31, F32,  I3, F34, F35, Z32)), \
                 np.hstack(( Z3,  Z3,  Z3,  I3,  Z3, Z32)), \
                 np.hstack(( Z3,  Z3,  Z3,  Z3,  I3, Z32)), \
                 np.hstack((Z23, Z23, Z23, Z23, Z23, F66))), \
               )
  
  # (Groves 14.80/88) process noise covariance
  Q11 = (S_rg*dt + 1/3*S_bgd*dt**3) * I3
  Q21 = (1/2*S_rg*dt**2 + 1/4*S_bgd*dt**4) * f21
  Q22 = (S_ra*dt + 1/3*S_bad*dt**3) * I3 + (1/3*S_rg*dt**3 + 1/5*S_bgd*dt**5) * (f21@f21.T)
  Q31 = (1/3*S_rg*dt**3 + 1/5*S_bgd*dt**5) * f21
  Q32 = (1/2*S_ra*dt**2 + 1/4*S_bad*dt**4) * I3 + (1/4*S_rg*dt**4 + 1/6*S_bgd*dt**6) * (f21@f21.T)
  Q33 = (1/3*S_ra*dt**3 + 1/5*S_bad*dt**5) * I3 + (1/5*S_rg*dt**5 + 1/7*S_bgd*dt**7) * (f21@f21.T)
  Q34 = 1/3*S_bad*dt**3*C_b_e
  Q35 = 1/4*S_bgd*dt**4*f21@C_b_e
  Q15 = 1/2*S_bgd*dt**2*C_b_e
  Q24 = 1/2*S_bad*dt**2*C_b_e
  Q25 = 1/3*S_bgd*dt**3*f21@C_b_e
  Q44 = S_bad*dt*I3
  Q55 = S_bgd*dt*I3
  Q66 = np.array([[S_p*dt + 1/3*S_f*dt**3, 1/2*S_f*dt**2], \
                  [         1/2*S_f*dt**2,        S_f*dt]],\
                 dtype=np.double)
  # Q66 = np.array([[       S_f*dt,          1/2*S_f*dt**2], \
  #                 [1/2*S_f*dt**2, S_p*dt + 1/3*S_f*dt**3]],\
  #                dtype=np.double)
  
  Q = np.vstack((np.hstack((Q11, Q21.T, Q31.T,  Z3, Q15, Z32)), \
                 np.hstack((Q21,   Q22, Q32.T, Q24, Q25, Z32)), \
                 np.hstack((Q31,   Q32,   Q33, Q34, Q35, Z32)), \
                 np.hstack(( Z3,   Q24, Q34.T, Q44,  Z3, Z32)), \
                 np.hstack((Q15, Q25.T, Q35.T,  Z3, Q55, Z32)), \
                 np.hstack((Z23,   Z23,   Z23, Z23, Z23, Q66))), \
               )
  
  return A, Q


# === __GEN_ESTIMATE ===
# Generates the observation matrix and pseudorange/pseudorange-rate measurements
#
# INPUTS:
#   sv_p      Nx1     GNSS satellite position [m]
#   sv_v      Nx1     GNSS satellite velocity [m/s]
#   w_ib_b    3x1     IMU angular velocity measurement [rad/s]
#   C_b_e     3x3     DCM from body to ECEF frame
#   r_eb_e    3x1     IMU ECEF position solution [m]
#   v_eb_e    3x1     IMU ECEF velocity solution [m/s]
#   L_ba_b    3x1     body frame lever arm from IMU to RCVR [m]
#   b         double  clock bias estimate
#   bdot      double  clock drift estimate
#
# OUTPUTS:
#   C         2Nx8    observation matrix
#   psr       Nx1     pseudorange measurement estimates [m]
#   psrdot    Nx1     pseudorange-rate measurement estimates [m/s]
#
@njit(cache=True, fastmath=True)
def __gen_estimate(sv_p, sv_v, w_ib_b, C_b_e, r_eb_e, v_eb_e, L_ba_b, b, bdot):
  # number of total measurements
  N = sv_p.shape[0]
  
  # Preallocate observation matrix
  C = np.zeros((2*N, 17), dtype=np.double)
  C[:N,15] = np.ones(N, dtype=np.double)
  C[N:,16] = np.ones(N, dtype=np.double)
  
  # Preallocate measurement vector
  psr = np.zeros(N, dtype=np.double)
  psrdot = np.zeros(N, dtype=np.double)
  
  # (Groves 14.121) lever arm corrections
  r_eb_e = r_eb_e + C_b_e @ L_ba_b
  v_eb_e = v_eb_e + C_b_e @ skew(w_ib_b) @ L_ba_b + OMEGA_ie @ C_b_e @ L_ba_b
  
  # loop through each satellite
  for i in np.arange(N):
    # # (Groves 8.36) approximate ECEF frame rotation during transit time
    # dt = np.linalg.norm(sv_p[i,:] - r_eb_e) / c
    # C_e_i = np.array([[     1.0, w_ie*dt, 0.0], \
    #                   [-w_ie*dt,     1.0, 0.0], \
    #                   [     0.0,     0.0, 1.0]], \
    #                  dtype=np.double)
    
    # (Groves 9.165) predicted pseudorange and unit vector
    # dr = C_e_i @ sv_p[i,:] - r_eb_e
    dr = sv_p[i,:] - r_eb_e
    raw_range = np.linalg.norm(dr)
    Ur = dr / raw_range
    
    # (Groves 9.165) predicted pseudorange-rate and doppler position vector
    # dv = (C_e_i @ (sv_v[i,:] + OMEGA_ie@sv_p[i,:]) - (v_eb_e + OMEGA_ie@r_eb_e))
    dv = sv_v[i,:] - v_eb_e
    raw_range_rate = Ur @ dv
    # Uv = np.cross(Up, np.cross(Up, dv/raw_range))
    
    # (Groves 14.126) add to geometry matrix and estimated measurement vector
    C[i,6:9] = Ur
    C[i+N,3:6] = Ur
    # C[i+N,6:9] = Uv
    psr[i] = raw_range + b
    psrdot[i] = raw_range_rate + bdot
    
  return C, psr, psrdot
