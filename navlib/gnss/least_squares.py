'''
|================================== navlib/gnss/least_squares.py ==================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/gnss/least_squares.py                                                          |
|  @brief    GNSS least squares solvers.                                                           |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from navlib.constants import Z3, Z32, Z23
from numba import njit

Z_4 = np.zeros(4, dtype=np.double)
Z_3 = np.zeros(3, dtype=np.double)
I = np.ones(4, dtype=np.double)


# === CALCPOS ===
# GNSS position iterative least squares solver
#
# INPUTS:
#   sv_pos  Nx3   satellite ECEF positions [m]
#   psr     Nx3   pseudorange measurements [m]
#   W       NxN   pseudorange variances [m^2]
#   x       4x1   initial ECEF position estimate [m]
#
# OUTPUTS:
#   x       4x1   ECEF position estimate [m]
#   H       4x4   Geometry matrix
#   P       4x4   Estimate covariance matrix
#
@njit(cache=True, fastmath=True)
def calcPos(sv_pos: np.ndarray,
            psr: np.ndarray,
            W: np.ndarray, 
            x: np.ndarray = Z_4):
  dx = I
  H = np.zeros((psr.size,4), dtype=np.double)
  H[:,3] = np.ones(psr.size, dtype=np.double)
  dy = np.zeros(psr.size, dtype=np.double)

  # while np.linalg.norm(dx) > 1e-6:
  for _ in np.arange(10):
    for i in np.arange(psr.size):
      dr = sv_pos[i,:] - x[:3]
      r = np.linalg.norm(dr)
      H[i,:3] = -dr / r
      dy[i] = psr[i] - (r + x[3])

    P = np.linalg.inv(H.T @ W @ H)
    dx = P @ H.T @ W @ dy
    x += dx
    if np.linalg.norm(dx[:3]) < 1e-6:
      break

  return x, H, P


# === CALCVEL ===
# GNSS velocity least squares solver
#
# INPUTS:
#   sv_vel  Nx3   satellite ECEF velocities [m/s]
#   psrdot  Nx3   pseudorange-rate measurements [m/s]
#   H       4x4   Geometry matrix from position solution
#   W       NxN   pseudorange-rate variances [m^2/s^2]
#
# OUTPUTS:
#   x       4x1   ECEF velocity estimate [m]
#   P       4x4   Estimate covariance matrix
#
@njit(cache=True, fastmath=True)
def calcVel(sv_vel: np.ndarray, 
            psr_dot: np.ndarray, 
            H: np.ndarray, 
            W: np.ndarray):
  y = psr_dot - np.sum((-H[:,:3])*sv_vel, 1)
  P = np.linalg.inv(H.T @ W @ H)
  x = P @ H.T @ W @ y
  return x, P


# === CALCPOSVEL ===
# GNSS position and velocity least squares solver
#
# INPUTS:
#   sv_pos  Nx3   satellite ECEF positions [m]
#   sv_vel  Nx3   satellite ECEF velocities [m/s]
#   psr     Nx3   pseudorange measurements [m]
#   psrdot  Nx3   pseudorange-rate measurements [m/s]
#   W       NxN   pseudorange-rate variances [m^2/s^2]
#   x       4x1   initial ECEF position estimate [m]
#
# OUTPUTS:
#   x       8x1   ECEF position and velocity estimate [m]
#   P       8x8   Estimate covariance matrix
#   DOP     4X4   Estimate Dilution Of Precision matrix
#
@njit(cache=True, fastmath=True)
def calcPosVel(sv_pos: np.ndarray, 
               sv_vel: np.ndarray, 
               psr: np.ndarray, 
               psr_dot:np.ndarray, 
               W: np.ndarray, 
               x: np.ndarray = Z_4):
  x_pos, H, P_pos = calcPos(sv_pos, psr, W, x)
  x_vel, P_vel = calcVel(sv_vel, psr_dot, H, W)
  x = np.hstack((x_pos[:3], x_vel[:3], np.array([x_pos[3], x_vel[3]])))
  P = np.vstack((np.hstack((P_pos[:3,:3], Z3, np.vstack((P_pos[:3,3], Z_3)).T )), \
                 np.hstack((Z3, P_vel[:3,:3], np.vstack((Z_3, P_vel[:3,3])).T )), \
                 np.hstack((np.vstack((P_pos[3,:3], Z_3)), 
                            np.vstack((Z_3, P_vel[3,:3])), 
                            np.array([[P_pos[3,3], 0.0], [0.0, P_vel[3,3]]])
                           ))
                ))
  DOP = np.linalg.inv(H.T @ H)
  return x, P, DOP
  