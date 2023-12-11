'''
|====================================== navlib/gnss/rcvr.py =======================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/gnss/rcvr.py                                                                   |
|  @brief    RCVR class (*dictionary*) type.                                                       |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

# TODO: Add receiver class (*dictionary*)

import numpy as np

# Position Domain (bad gps)
bad_gps = {
  'stdp': np.array([30.0, 30.0, 34.0]),          # [m]
  'stdv': np.array([0.03, 0.03, 0.05]),       # [m/s]
  'accel_psd': np.array([20.0, 20.0, 20.0]),  # [m^2/s^3]
  'psr_std': 30.0,                             # [m]
  'psrdot_std': 0.03,                         # [m/s]
  # 'clkp_psd': 0.01,                           # [m^2/s]
  # 'clkf_psd': 0.04,                           # [m^2/s^3]
  'clkp_psd': 0.1, 
  'clkf_psd': 0.4, 
  'freq': 10.0,                               # [Hz]
}

def RCVR(stdp: np.ndarray,
         stdv: np.ndarray,
         accel_psd: np.ndarray,
         freq: np.double):
  rcvr_ = {
    'stdp': stdp,
    'stdv': stdv,
    'accel_psd': accel_psd,
    'freq': freq
  }