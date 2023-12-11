'''
|================================ navlib/examples/test_filters.py =================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/coordinate/test_filter.py                                                      |
|  @brief    Test GNSS and GNSS-INS filters from 'gnss' and 'gnss-ins' folders. Implements in the  |
|            ECEF frame.                                                                           |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['dark_background'])

from navlib.constants import I3, lla_deg2rad, lla_rad2deg
from navlib.ins.imu import fix_si_errors, vn100
from navlib.gnss.rcvr import bad_gps
from navlib.coordinates.dcm import ned2ecefDcm, ecef2nedDcm
from navlib.coordinates.position import ecef2lla, ecef2ned, lla2ecef, lla2ned
from navlib.coordinates.velocity import ned2ecefv, ecef2nedv
from navlib.attitude.euler import euler2dcm
from navlib.attitude.dcm import dcm2euler
from navlib.attitude.skew import skew

from navlib.ins.mechanization import mechanization_ecef
from navlib.gnss.least_squares import calcPosVel
import navlib.gnss_ins.kf_loosely as loose
import navlib.gnss_ins.kf_tightly as tight
import navlib.gnss.kf_position as kf_pos
import navlib.gnss.kf_measurement as kf_meas


# === test_ls ===
def test_ls(init_p, init_v, sv_pos, sv_vel, psr, psrdot, gps_info):
  # initialize output
  lla_out = np.zeros((sv_pos.shape[0],3), dtype=np.double)
  ned_p_out = np.zeros((sv_pos.shape[0],3), dtype=np.double)
  ned_v_out = np.zeros((sv_pos.shape[0],3), dtype=np.double)
  ecef_pv_out = np.zeros((sv_pos.shape[0],6), dtype=np.double)
  
  lla_out[0,:] = init_p * lla_rad2deg
  ned_v_out[0,:] = init_v
  
  x = np.zeros(8, dtype=np.double)
  for i in np.arange(sv_pos.shape[0]-1):
    sv_p = np.zeros((int(sv_pos[i,:].size/3), 3))
    sv_v = np.zeros((int(sv_pos[i,:].size/3), 3))
    for k in np.arange(int(sv_pos[i,:].size/3)):
      kk = k*3
      sv_p[k,:] = sv_pos[i,kk:kk+3]
      sv_v[k,:] = sv_vel[i,kk:kk+3]
    x,_,_ = calcPosVel(sv_p, sv_v, psr[i,:], psrdot[i,:], np.diag(np.ones(psr[i,:].shape)), x[:4])
    
    # append output
    lla_out[i,:] = ecef2lla(x[:3])
    ned_p_out[i,:] = ecef2ned(x[:3], lla0)
    ned_v_out[i,:] = ecef2nedv(x[3:6], lla0)
    lla_out[i,:] = lla_out[i,:] * lla_rad2deg
    ecef_pv_out[i,:] = x[:6]
    
  return lla_out, ned_p_out, ned_v_out, ecef_pv_out


# === test_mechanization ===
def test_mechanization(init_p, init_v, init_a, f_ib_b, w_ib_b, imu_info):
  # initialize output
  lla_out = np.zeros(f_ib_b.shape, dtype=np.double)
  ned_p_out = np.zeros(f_ib_b.shape, dtype=np.double)
  ned_v_out = np.zeros(f_ib_b.shape, dtype=np.double)
  rpy_out = np.zeros(f_ib_b.shape, dtype=np.double)
  
  lla_out[0,:] = init_p * lla_rad2deg
  ned_v_out[0,:] = init_v
  rpy_out[0,:] = np.rad2deg(init_a)
  
  # initial ecef position, velocity and attitude
  X = lla2ecef(init_p)
  V = ned2ecefv(init_v, init_p)
  C_b_n = euler2dcm(init_a).T
  C = ned2ecefDcm(init_p) @ C_b_n
  
  dt = 1 / imu_info['freq']
  
  # run simulation
  j = 1
  for i in np.arange(1,f_ib_b.shape[0]):
    # apply bias correction
    wb = w_ib_b[i,:] - imu_info['gb_sta']
    fb = f_ib_b[i,:] - imu_info['ab_sta']
    
    # mechanize imu (ecef frame)
    C, V, X = mechanization_ecef(fb, wb, C, V, X, dt)
      
    # append output
    lla_out[i,:] = ecef2lla(X)
    ned_p_out[i,:] = ecef2ned(X, init_p)
    ned_v_out[i,:] = ecef2nedv(V, init_p)
    C_e_n = ecef2nedDcm(lla_out[i,:])
    rpy_out[i,:] = np.rad2deg(dcm2euler((C_e_n @ C).T))
    lla_out[i,:] = lla_out[i,:] * lla_rad2deg
    
  # finished
  return lla_out, ned_p_out, ned_v_out, rpy_out


# === test_gps_pos_domain_filter
def test_gps_pos_domain_filter(init_p, init_v, gps_meas, gps_info):
  # initialize output
  lla_out = np.zeros((gps_meas.shape[0],3), dtype=np.double)
  ned_p_out = np.zeros((gps_meas.shape[0],3), dtype=np.double)
  ned_v_out = np.zeros((gps_meas.shape[0],3), dtype=np.double)
  
  lla_out[0,:] = init_p * lla_rad2deg
  ned_v_out[0,:] = init_v
  
  # initialize kalman filter
  kf_x = np.concatenate((lla2ecef(init_p), ned2ecefv(init_v, init_p)))
  kf_P = np.diag([5.0,5.0,7.0, 0.05,0.05,0.05])
  S_a = gps_info['accel_psd']**2
  S_pos = gps_info['stdp']**2
  S_vel = gps_info['stdv']**2
  
  dt = 1 / gps_info['freq']
  
  # run simulation
  for i in np.arange(1,gps_meas.shape[0]-1):
    # time update (prediction)
    kf_x, kf_P = kf_pos.predict(kf_x, kf_P, S_a, dt)
    
    # measurement update (correction)
    kf_x, kf_P = kf_pos.correct(kf_x, kf_P, gps_meas[i,:3], gps_meas[i,3:], S_pos, S_vel)
    
    # append output
    lla_out[i,:] = ecef2lla(kf_x[:3])
    ned_p_out[i,:] = ecef2ned(kf_x[:3], init_p)
    ned_v_out[i,:] = ecef2nedv(kf_x[3:], init_p)
    lla_out[i,:] = lla_out[i,:] * lla_rad2deg
    
  # finished
  return lla_out, ned_p_out, ned_v_out


# === test_gps_meas_domain_filter
def test_gps_meas_domain_filter(init_p, init_v, sv_pos, sv_vel, psr, psrdot, gps_info):
  # initialize output
  lla_out = np.zeros((sv_pos.shape[0],3), dtype=np.double)
  ned_p_out = np.zeros((sv_pos.shape[0],3), dtype=np.double)
  ned_v_out = np.zeros((sv_pos.shape[0],3), dtype=np.double)
  
  lla_out[0,:] = init_p * lla_rad2deg
  ned_v_out[0,:] = init_v
  
  # initialize kalman filter
  kf_x = np.concatenate((lla2ecef(init_p), ned2ecefv(init_v, init_p), np.array([0.0,0.0])))
  kf_P = np.diag([5.0,5.0,7.0, 0.05,0.05,0.05, 1.0,0.1])
  S_a = gps_info['accel_psd']**2
  S_psr = np.repeat(gps_info['psr_std']**2, psr.shape[1])
  S_psrdot = np.repeat(gps_info['psrdot_std']**2, psrdot.shape[1])
  S_p = gps_info['clkp_psd']**2
  S_f = gps_info['clkf_psd']**2
  
  dt = 1 / gps_info['freq']
  
  # run simulation
  for i in np.arange(1,sv_pos.shape[0]-1):
    # time update (prediction)
    kf_x, kf_P = kf_meas.predict(kf_x, kf_P, S_a, S_p, S_f, dt)
    
    # measurement update (correction)
    sv_p = np.zeros((int(sv_pos[i,:].size/3), 3))
    sv_v = np.zeros((int(sv_pos[i,:].size/3), 3))
    for k in np.arange(int(sv_pos[i,:].size/3)):
      kk = k*3
      sv_p[k,:] = sv_pos[i,kk:kk+3]
      sv_v[k,:] = sv_vel[i,kk:kk+3]
    kf_x, kf_P = kf_meas.correct(kf_x, kf_P, sv_p, sv_v, psr[i,:], psrdot[i,:], S_psr, S_psrdot)
    
    # append output
    lla_out[i,:] = ecef2lla(kf_x[:3])
    ned_p_out[i,:] = ecef2ned(kf_x[:3], init_p)
    ned_v_out[i,:] = ecef2nedv(kf_x[3:6], init_p)
    lla_out[i,:] = lla_out[i,:] * lla_rad2deg
    
  # finished
  return lla_out, ned_p_out, ned_v_out


# === test_loosely_coupled ===
def test_loosely_coupled(init_p, init_v, init_a, f_ib_b, w_ib_b, gps_meas, gps_info, imu_info):
  # initialize output
  lla_out = np.zeros(f_ib_b.shape, dtype=np.double)
  ned_p_out = np.zeros(f_ib_b.shape, dtype=np.double)
  ned_v_out = np.zeros(f_ib_b.shape, dtype=np.double)
  rpy_out = np.zeros(f_ib_b.shape, dtype=np.double)
  
  lla_out[0,:] = init_p * lla_rad2deg
  ned_v_out[0,:] = init_v
  rpy_out[0,:] = np.rad2deg(init_a)
  
  # initial ecef position, velocity and attitude
  X = lla2ecef(init_p)
  V = ned2ecefv(init_v, init_p)
  C_b_n = euler2dcm(init_a).T
  C = ned2ecefDcm(init_p) @ C_b_n
  lever_arm = np.zeros(3, dtype=np.double)
  
  # initialize kalman filter
  kf_x = np.zeros(15)
  kf_P = np.diag([0.017,0.017,0.035, 0.05,0.05,0.05, 5.0,5.0,7.0, 0.1,0.1,0.1, 0.001,0.001,0.001])
  b_gyr = np.zeros(3, dtype=np.double)
  b_acc = np.zeros(3, dtype=np.double)
  S_rg = imu_info['gb_psd']**2
  S_ra = imu_info['ab_psd']**2
  S_bgd = imu_info['arw']**2
  S_bad = imu_info['vrw']**2
  S_pos = gps_info['stdp']**2
  S_vel = gps_info['stdv']**2
  
  f_update = np.round(imu_info['freq'] / gps_info['freq'])
  dt = 1 / imu_info['freq']
  
  # run simulation
  j = 1
  for i in np.arange(1,f_ib_b.shape[0]):
    # apply bias correction
    wb = w_ib_b[i,:] - b_gyr - imu_info['gb_sta']
    fb = f_ib_b[i,:] - b_acc - imu_info['ab_sta']
    
    # mechanize imu (ecef frame)
    C, V, X = mechanization_ecef(fb, wb, C, V, X, dt)
    
    # time update (prediction)
    kf_x, kf_P = loose.predict(kf_x, kf_P, fb, C, X, S_rg, S_ra, S_bgd, S_bad, dt)
    
    # determine if GPS update has occurred
    if (np.mod(i, f_update) == 0.0) and (i != f_ib_b.shape[0]-1):
      # measurement update (correction)
      kf_x, kf_P = loose.correct(kf_x, kf_P, \
                                 gps_meas[j,:3], gps_meas[j,3:], \
                                 wb, C, X, V, lever_arm, \
                                 S_pos, S_vel)
      j += 1
      
      # apply error state corrections
      C = (I3 - skew(kf_x[:3])) @ C
      V = V - kf_x[3:6]
      X = X - kf_x[6:9]
      b_acc = b_acc + kf_x[9:12]
      b_gyr = b_gyr + kf_x[12:15]
      kf_x = np.zeros(15, dtype=np.double)
      
    # append output
    lla_out[i,:] = ecef2lla(X)
    ned_p_out[i,:] = ecef2ned(X, init_p)
    ned_v_out[i,:] = ecef2nedv(V, init_p)
    C_e_n = ecef2nedDcm(lla_out[i,:])
    rpy_out[i,:] = np.rad2deg(dcm2euler((C_e_n @ C).T))
    lla_out[i,:] = lla_out[i,:] * lla_rad2deg
    
  # finished
  return lla_out, ned_p_out, ned_v_out, rpy_out


# === test_tightly_coupled ===
def test_tightly_coupled(init_p, init_v, init_a, f_ib_b, w_ib_b, sv_pos, sv_vel, psr, psrdot, gps_info, imu_info):
  # initialize output
  lla_out = np.zeros(f_ib_b.shape, dtype=np.double)
  ned_p_out = np.zeros(f_ib_b.shape, dtype=np.double)
  ned_v_out = np.zeros(f_ib_b.shape, dtype=np.double)
  rpy_out = np.zeros(f_ib_b.shape, dtype=np.double)
  
  lla_out[0,:] = init_p * lla_rad2deg
  ned_v_out[0,:] = init_v
  rpy_out[0,:] = np.rad2deg(init_a)
  
  # initial ecef position, velocity and attitude
  X = lla2ecef(init_p)
  V = ned2ecefv(init_v, init_p)
  C_b_n = euler2dcm(init_a).T
  C = ned2ecefDcm(init_p) @ C_b_n
  lever_arm = np.zeros(3, dtype=np.double)
  
  # initialize kalman filter
  kf_x = np.zeros(17)
  kf_P = np.diag([0.017,0.017,0.035, 0.05,0.05,0.05, 5.0,5.0,7.0, 0.1,0.1,0.1, 0.001,0.001,0.001, 0.5,0.01])
  b_gyr = np.zeros(3, dtype=np.double)
  b_acc = np.zeros(3, dtype=np.double)
  S_rg = imu_info['gb_psd']**2
  S_ra = imu_info['ab_psd']**2
  S_bgd = imu_info['arw']**2
  S_bad = imu_info['vrw']**2
  S_psr = np.repeat(gps_info['psr_std']**2, psr.shape[1])
  S_psrdot = np.repeat(gps_info['psrdot_std']**2, psrdot.shape[1])
  S_p = gps_info['clkp_psd']**2
  S_f = gps_info['clkf_psd']**2
  
  f_update = np.round(imu_info['freq'] / gps_info['freq'])
  dt = 1 / imu_info['freq']
  
  # run simulation
  j = 1
  for i in np.arange(1,f_ib_b.shape[0]):
    # apply bias correction
    wb = w_ib_b[i,:] - b_gyr - imu_info['gb_sta']
    fb = f_ib_b[i,:] - b_acc - imu_info['ab_sta']
    
    # mechanize imu (ecef frame)
    C, V, X = mechanization_ecef(fb, wb, C, V, X, dt)
    
    # time update (prediction)
    kf_x, kf_P = tight.predict(kf_x, kf_P, fb, C, X, S_rg, S_ra, S_bgd, S_bad, S_p, S_f, dt)
    
    # determine if GPS update has occurred
    if (np.mod(i, f_update) == 0.0) and (i != f_ib_b.shape[0]-1):
      sv_p = np.zeros((int(sv_pos[j,:].size/3), 3))
      sv_v = np.zeros((int(sv_pos[j,:].size/3), 3))
      for k in np.arange(int(sv_pos[j,:].size/3)):
        kk = k*3
        sv_p[k,:] = sv_pos[j,kk:kk+3]
        sv_v[k,:] = sv_vel[j,kk:kk+3]
        
      # measurement update (correction)
      kf_x, kf_P = tight.correct(kf_x, kf_P, \
                                  sv_p, sv_v, psr[j,:], psrdot[j,:], S_psr, S_psrdot, \
                                  wb, C, X, V, lever_arm)
      j += 1
      
      # apply error state corrections
      C = (I3 - skew(kf_x[:3])) @ C
      V = V - kf_x[3:6]
      X = X - kf_x[6:9]
      b_acc = b_acc + kf_x[9:12]
      b_gyr = b_gyr + kf_x[12:15]
      kf_x[:15] = np.zeros(15, dtype=np.double)
  
    # append output
    lla_out[i,:] = ecef2lla(X)
    ned_p_out[i,:] = ecef2ned(X, init_p)
    ned_v_out[i,:] = ecef2nedv(V, init_p)
    C_e_n = ecef2nedDcm(lla_out[i,:])
    rpy_out[i,:] = np.rad2deg(dcm2euler((C_e_n @ C).T))
    lla_out[i,:] = lla_out[i,:] * lla_rad2deg
    
  # finished
  return lla_out, ned_p_out, ned_v_out, rpy_out


if __name__ == '__main__':
  G = 9.80665
  # VectorNav VN100 (better than average imu) -> convert units
  imu = fix_si_errors(vn100)

  # load data
  print(os.path.abspath(''))
  # data_path = os.path.abspath('..//examples//example_data//')
  data_path = os.path.abspath('.//navlib//examples//example_data//')
  
  ref_lla = np.genfromtxt(data_path+'//ref_pos.csv', delimiter=",", skip_header=1)
  ref_vel = np.genfromtxt(data_path+'//ref_vel.csv', delimiter=",", skip_header=1)
  ref_rpy = np.fliplr(np.genfromtxt(data_path+'//ref_att_euler.csv', delimiter=",", skip_header=1))
  lla0 = ref_lla[0,:] * lla_deg2rad
  
  accel = np.genfromtxt(data_path+'//accel-0.csv', delimiter=",", skip_header=1)
  gyro = np.deg2rad(np.genfromtxt(data_path+'//gyro-0.csv', delimiter=",", skip_header=1))
  sv_pos = np.genfromtxt(data_path+'//svpos-0.csv', delimiter=",", skip_header=0)
  sv_vel = np.genfromtxt(data_path+'//svvel-0.csv', delimiter=",", skip_header=0)
  psr = np.genfromtxt(data_path+'//ranges-0.csv', delimiter=",", skip_header=0)
  psrdot = np.genfromtxt(data_path+'//rangerates-0.csv', delimiter=",", skip_header=0)
  
  # gps_pv = np.genfromtxt(data_path+'//gps-0.csv', delimiter=",", skip_header=1)
  # gps_pv_ecef = np.zeros(gps_pv.shape)
  # gps_p_ned = np.zeros((gps_pv.shape[0],3))
  # for i in np.arange(gps_pv.shape[0]):
  #   gps_pv_ecef[i,3:] = ned2ecefv(gps_pv[i,3:], gps_pv[i,:3] * lla_deg2rad)
  #   gps_pv_ecef[i,:3] = lla2ecef(gps_pv[i,:3] * lla_deg2rad)
  #   gps_p_ned[i,:] = lla2ned(gps_pv[i,:3] * lla_deg2rad, lla0)
  
  # reference lla
  ref_ned = np.zeros(ref_lla.shape, dtype=np.double)
  for i in np.arange(ref_lla.shape[0]):
    ref_ned[i,:] = lla2ned(ref_lla[i,:] * lla_deg2rad, lla0)
    
  # test least squares
  print('testing GNSS least squares...')
  lla_ls, ned_p_ls, ned_v_ls, ecef_pv_ls = test_ls( \
    lla0, ref_vel[0,:], sv_pos, sv_vel, psr, psrdot, bad_gps)
  
  # test mechanization
  print('testing IMU mechanization...')
  lla_mech, ned_p_mech, ned_v_mech, rpy_mech = test_mechanization( \
    lla0, ref_vel[0,:], np.deg2rad(ref_rpy[0,:]), accel, gyro, imu)
  
  # test position domain filter
  print('testing GNSS position domain filter...')
  lla_pos_dom, ned_p_pos_dom, ned_v_pos_dom = test_gps_pos_domain_filter( \
    lla0, ref_vel[0,:], ecef_pv_ls, bad_gps)
  
  # test measurement domain filter
  print('testing GNSS measurement domain filter...')
  lla_meas_dom, ned_p_meas_dom, ned_v_meas_dom = test_gps_meas_domain_filter( \
    lla0, ref_vel[0,:], sv_pos, sv_vel, psr, psrdot, bad_gps)
  
  # test loosely coupled
  print('testing loosely coupled GNSS-INS integration...')
  lla_loose, ned_p_loose, ned_v_loose, rpy_loose = test_loosely_coupled( \
    lla0, ref_vel[0,:], np.deg2rad(ref_rpy[0,:]), \
    accel, gyro, ecef_pv_ls, bad_gps, imu)
  
  # test tightly coupled
  print('testing tightly coupled GNSS-INS integration...')
  lla_tight, ned_p_tight, ned_v_tight, rpy_tight = test_tightly_coupled( \
    lla0, ref_vel[0,:], np.deg2rad(ref_rpy[0,:]), \
    accel, gyro, sv_pos, sv_vel, psr, psrdot, bad_gps, imu)
  
  # finished
  plt.figure()
  plt.plot(ref_ned[:,1], ref_ned[:,0], color='silver', linewidth=2, label='Ref.')
  plt.plot(ned_p_ls[:,1], ned_p_ls[:,0], 'x', color='yellowgreen', linewidth=1, markersize=5, label='LS')
  plt.plot(ned_p_mech[:,1], ned_p_mech[:,0], '--', color='red', linewidth=2, label='Mech.')
  plt.plot(ned_p_pos_dom[:,1], ned_p_pos_dom[:,0], '.', color='aquamarine', linewidth=3, label='GNSS-Pos')
  plt.plot(ned_p_meas_dom[:,1], ned_p_meas_dom[:,0], '.', color='pink', linewidth=3, label='GNSS-Meas')
  plt.plot(ned_p_loose[:,1], ned_p_loose[:,0], '.', color='deepskyblue', linewidth=3, label='Loose')
  plt.plot(ned_p_tight[:,1], ned_p_tight[:,0], '.', color='magenta', linewidth=3, label='Tight')
  plt.legend()
  plt.title('NED Position')
  plt.grid(which='both', visible=True)
  
  plt.figure()
  plt.plot(ref_rpy[:,2], '*', color='silver', linewidth=2, label='Ref.')
  plt.plot(rpy_mech[:,2], '.', color='red', linewidth=3, label='Mech.')
  plt.plot(rpy_loose[:,2], '.', color='deepskyblue', linewidth=3, label='Loose')
  plt.plot(rpy_tight[:,2], '.', color='magenta', linewidth=3, label='Tight')
  plt.legend()
  plt.title('Heading')
  
  plt.show()
  
  print('done!')