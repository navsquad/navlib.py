'''
|======================================= navlib/ins/imu.py ========================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/ins/imu.py                                                                     |
|  @brief    IMU class (*dictionary*) type.                                                        |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from navlib.constants import G, D2R, G2T, FT2M


# === Default IMUs ===
# Tactical Grade (Honeywell HG1700)
# https://aerospace.honeywell.com/content/dam/aerobt/en/documents/landing-pages/brochures/N61-1619-000-001-HG1700InertialMeasurementUnit-bro.pdf
hg1700 = {
  'vrw': np.array([0.65, 0.65, 0.65]) * FT2M,   # m/s/root(hr)
  'arw': np.array([0.125, 0.125, 0.125]),       # deg/root(hr)
  'vrrw': np.array([0.0, 0.0, 0.0]),            # m/s/root(hr)/s
  'arrw': np.array([0.0, 0.0, 0.0]),            # deg/root(hr)/s
  'ab_sta': np.array([0.0, 0.0, 0.0]),          # mg
  'gb_sta': np.array([0.0, 0.0, 0.0]),          # deg/hr
  'ab_dyn': np.array([0.58, 0.58, 0.58]),       # mg
  'gb_dyn': np.array([0.017, 0.017, 0.017]),    # deg/hr
  'ab_corr': np.array([100.0, 100.0, 100.0]),   # s
  'gb_corr': np.array([100.0, 100.0, 100.0]),   # s
  'mag_psd': np.array([0.0, 0.0, 0.0]),         # mGauss/root(Hz)
  'freq': 100.0
}

# Industrial Grade (VectorNav VN100)
#https://www.vectornav.com/docs/default-source/datasheets/vn-100-datasheet-rev2.pdf?sfvrsn=8e35fd12_10
vn100 = {
  'vrw': np.array([0.14, 0.14, 0.14]) * 0.001 * G * 60, # m/s/root(hr)
  'arw': np.array([3.5e-3, 3.5e-3, 3.5e-3]) * 60,       # deg/root(hr)
  'vrrw': np.array([0.0, 0.0, 0.0]),                    # m/s/root(hr)/s
  'arrw': np.array([0.0, 0.0, 0.0]),                    # deg/root(hr)/s
  'ab_sta': np.array([0.0, 0.0, 0.0]),                  # mg
  'gb_sta': np.array([0.0, 0.0, 0.0]),                  # deg/hr
  'ab_dyn': np.array([0.04, 0.04, 0.04]),               # mg
  'gb_dyn': np.array([5.0, 5.0, 5.0]),                  # deg/hr
  'ab_corr': np.array([100.0, 100.0, 100.0]),           # s
  'gb_corr': np.array([100.0, 100.0, 100.0]),           # s
  'mag_psd': np.array([140.0, 140.0, 140.0]),           # mGauss/root(Hz)
  'freq': 100.0
}


# === IMU ===
# Creates an IMU object and corrects to SI units
#
# INPUTS:
#   vrw       3x1     velocity random walks [m/s/root(hour)]
#   arw       3x1     angle random walks [deg/root(hour)]
#   vrrw      3x1     velocity rate random walks [deg/root(hour)/s]
#   arrw      3x1     angle rate random walks [deg/root(hour)/s]
#   ab_sta    3x1     accel static biases [mg]
#   gb_sta    3x1     gyro static biases [deg/s]
#   ab_dyn    3x1     accel dynamic biases [mg]
#   gb_dyn    3x1     gyro dynamic biases[deg/s]
#   ab_corr   3x1     accel correlation times [s]
#   gb_corr   3x1     gyro correlation times [s]
#   mag_psd   3x1     magnetometer noise density [mgauss/root(Hz)]
#   freq      double  IMU frequency [Hz]
#
# OUTPUTS:
#   imu       dict    imu object
#
def IMU(vrw: np.ndarray, 
        arw: np.ndarray, 
        vrrw: np.ndarray, 
        arrw: np.ndarray, 
        ab_sta: np.ndarray, 
        gb_sta: np.ndarray, 
        ab_dyn: np.ndarray, 
        gb_dyn: np.ndarray, 
        ab_corr: np.ndarray, 
        gb_corr: np.ndarray, 
        mag_psd: np.ndarray, 
        freq: np.double,
        correct: bool=True):
  # design imu
  imu_ = {
    'vrw': vrw,
    'arw': arw,
    'vrrw': vrrw,
    'arrw': arrw,
    'ab_sta': ab_sta,
    'gb_sta': gb_sta,
    'ab_dyn': ab_dyn,
    'gb_dyn': gb_dyn,
    'ab_corr': ab_corr,
    'gb_corr': gb_corr,
    'mag_psd': mag_psd,
    'freq': freq
  }
  # fix si errors
  if correct:
    imu_ = fix_si_errors(imu_)
  return imu_


# === FIX_SI_ERRORS ===
# Corrects IMU errors from spec-sheets into SI units
#
# INPUTS:
#   IMU object with the following fields
#     vrw       3x1     velocity random walks [m/s/root(hour)]
#     arw       3x1     angle random walks [deg/root(hour)]
#     vrrw      3x1     velocity rate random walks [m/s/root(hour)/s]
#     arrw      3x1     angle rate random walks [deg/root(hour)/s]
#     ab_sta    3x1     accel static biases [mg]
#     gb_sta    3x1     gyro static biases [deg/s]
#     ab_dyn    3x1     accel dynamic biases [mg]
#     gb_dyn    3x1     gyro dynamic biases[deg/s]
#     ab_corr   3x1     accel correlation times [s]
#     gb_corr   3x1     gyro correlation times [s]
#     mag_psd   3x1     magnetometer noise density [mgauss/root(Hz)]
#     freq      double  IMU frequency [Hz]
#
# OUTPUTS:
#   imu         dict    imu object with the following fields
#     vrw       3x1     velocity random walks [m/s^2/root(Hz)]
#     arw       3x1     angle random walks [rad/s/root(Hz)]
#     vrrw      3x1     velocity rate random walks [m/s^3/root(Hz)]
#     arrw      3x1     angle rate random walks [rad/s^2/root(Hz)]
#     ab_sta    3x1     accel static biases [m/s^2]
#     gb_sta    3x1     gyro static biases [rad/s]
#     ab_dyn    3x1     accel dynamic biases [m/s^2]
#     gb_dyn    3x1     gyro dynamic biases[rad/s]
#     ab_corr   3x1     accel correlation times [s]
#     gb_corr   3x1     gyro correlation times [s]
#     ab_psd    3x1     acc dynamic bias root-PSD [m/s^2/root(Hz)]
#     gb_psd    3x1     gyro dynamic bias root-PSD [rad/s/root(Hz)]
#     mag_psd   3x1     magnetometer noise density [tesla]
#     freq      double  IMU frequency [Hz]
#
def fix_si_errors(*args):
  if len(args) > 1:
    imu['vrw'] = args[0]
    imu['arw'] = args[1]
    imu['vrrw'] = args[2]
    imu['arrw'] = args[3]
    imu['ab_sta'] = args[4]
    imu['gb_sta'] = args[5]
    imu['ab_dyn'] = args[6]
    imu['gb_dyn'] = args[7]
    imu['ab_corr'] = args[8]
    imu['gb_corr'] = args[9]
    imu['mag_psd'] = args[10]
    imu['freq'] = args[11]
  else:
    imu = args[0]
    
  imu_si = {}
  
  # root-PSD noise
  imu_si['vrw'] = (imu['vrw'] / 60)             # m/s/root(hour) -> m/s^2/root(Hz)
  imu_si['arw'] = (imu['arw'] / 60) * D2R;      # deg/root(hour) -> rad/s/root(Hz)

  #  root-PSD rate noise
  imu_si['vrrw'] = (imu['vrrw'] / 60) 
  imu_si['arrw'] = (imu['arrw'] / 60) * D2R;    # deg/root(hour) -> rad/s/root(Hz)

  # Dynamic bias
  imu_si['ab_dyn'] = imu['ab_dyn'] * 0.001 * G; # mg -> m/s^2
  imu_si['gb_dyn'] = imu['gb_dyn'] * D2R;       # deg/s -> rad/s;

  # Correlation time
  imu_si['ab_corr'] = imu['ab_corr']
  imu_si['gb_corr'] = imu['gb_corr']

  # Dynamic bias root-PSD
  if (np.any(np.isinf(imu['ab_corr']))):
      imu_si['ab_psd'] = imu_si['ab_dyn']       # m/s^2 (approximation)
  else:
      imu_si['ab_psd'] = imu_si['ab_dyn'] / np.sqrt(imu['ab_corr']) # m/s^2/root(Hz)

  if (np.any(np.isinf(imu['gb_corr']))):
      imu_si['gb_psd'] = imu_si['gb_dyn']       # rad/s (approximation)
  else:
      imu_si['gb_psd'] = imu_si['gb_dyn'] / np.sqrt(imu['gb_corr']) # rad/s/root(Hz)

  # time 
  dt = 1.0 / imu['freq']

  # Static bias
  imu_si['ab_sta'] = imu['ab_sta'] * 0.001 * G;    # mg -> m/s^2
  imu_si['gb_sta'] = imu['gb_sta'] * D2R;          # deg/s -> rad/s

  # Standard deviation
  imu_si['a_std']   = imu_si['vrw'] / np.sqrt(dt); # m/s^2/root(Hz) -> m/s^2
  imu_si['g_std']   = imu_si['arw'] / np.sqrt(dt); # rad/s/root(Hz) -> rad/s

  # MAG
  imu_si['mag_std'] = (imu['mag_psd'] * 1e-3) / np.sqrt(dt) * G2T # mGauss/root(Hz) -> Tesla

  imu_si['freq'] = imu['freq']
  
  return imu_si
    

