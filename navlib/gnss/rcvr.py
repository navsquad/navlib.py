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
import navlib.gnss.kf_measurement as kfm 
import navlib.gnss.kf_position as kfp 
import navlib.gnss.least_squares as ls

# Poor gps, vehicle mounted approximation, 10 Hz update rate
bad_gps = {
  'std_pos': np.array([30.0, 30.0, 30.0]),  # [m]
  'std_vel': np.array([0.05, 0.05, 0.05]),  # [m/s]
  'std_range': 30.0,                        # [m]
  'std_rate': 0.05,                         # [m/s]
  'psd_acc': np.array([20.0, 20.0, 20.0]),  # [m^2/s^3]
  'psd_clkp': 0.1,                          # [m^2/s]
  'psd_clkf': 0.4,                          # [m^2/s^3]
  'freq': 10.0,                             # [Hz]
}


class RCVR:
  # === __INIT__ ===
  # constructor
  #
  # INPUTS:
  #   type    str   type of receiver used
  #                   - 'pos': position domain
  #                   - 'meas': measurement domain
  #   kwargs  dict  receiver specifications, defaults to "bad_gps"
  #                   - 'std_pos': position error standard deviation [m]
  #                   - 'std_vel': velocity error standard deviation [m/s]
  #                   - 'std_range': pseudorange error standard deviation [m]
  #                   - 'std_rate': pseudorange-rate error standard deviation [m/s]
  #                   - 'psd_acc': acceleration induced error power spectral density [m^2/s^3]
  #                   - 'psd_clkp': clock phase induced error power spectral density [m^2/s]
  #                   - 'psd_clkf': clock frequency induced error power spectral density [m^2/s^3]
  #                   - 'freq': receiver update frequency [Hz]
  #
  def __init__(self, type='meas', **kwargs):
    # update type
    self.type = type.lower()

    # check keyword arguments
    self.std_pos = bad_gps['std_pos']
    self.std_vel = bad_gps['std_vel']
    self.std_range = bad_gps['std_range']
    self.std_rate = bad_gps['std_rate']
    self.psd_acc = bad_gps['psd_acc']
    self.psd_clkp = bad_gps['psd_clkp']
    self.psd_clkf = bad_gps['psd_clkf']
    self.freq = bad_gps['freq']
    self.dt = 1 / self.freq
    self.__parse_rcvr_kwargs(kwargs)


# ------------------------------------------------------------------------------------------------ #
  # === SET_TYPE ===
  # change the type of receiver used
  #
  # INPUTS:
  #   type      type of receiver used
  #               - 'pos': position domain
  #               - 'meas': measurement domain
  #
  def _set_type(self, type):
    # update type
    self.type = type.lower()


  # === SET_POS_ERROR ===
  # change the standard deviation of position error
  #
  # INPUTS:
  #   err   3x1   position error [m]
  #
  def _set_pos_error(self, err):
    # update error
    self.std_pos = err


  # === SET_VEL_ERROR ===
  # change the standard deviation of velocity error
  #
  # INPUTS:
  #   err   3x1   velocity error [m/s]
  #
  def _set_vel_error(self, err):
    # update error
    self.std_vel = err


  # === SET_ACCEL_PSD ===
  # change the power spectral density of acceleration induced error
  #
  # INPUTS:
  #   psd   double  acceleration induced error PSD [m^2/s^3]
  #
  def _set_accel_psd(self, psd):
    # update psd
    self.psd_acc = psd


  # === _SET_CLKP_PSD ===
  # change the power spectral density of clock phase induced error
  #
  # INPUTS:
  #   psd   double  clock phase induced error PSD [m^2/s]
  #
  def _set_clkp_psd(self, psd):
    # update psd
    self.psd_clkp = psd

  
  # === _SET_CLKF_PSD ===
  # change the power spectral density of clock frequency induced error
  #
  # INPUTS:
  #   psd   double  clock frequency induced error PSD [m^2/s^3]
  #
  def _set_clkf_psd(self, psd):
    # update psd
    self.psd_clkf = psd


  # === _SET_FREQ ===
  # change the receiver update frequency
  #
  # INPUTS:
  #   freq  double  update rate [Hz]
  #
  def _set_freq(self, freq):
    # update freq
    self.freq = freq
    self.dt = 1 / self.freq


# ------------------------------------------------------------------------------------------------ #
  # === _SET_RANGE_ERROR ===
  # change the standard deviation of pseudorange error
  #
  # INPUTS:
  #   err   double or Nx1   pseudorange error [m]
  #
  def _set_range_error(self, err):
    # update error
    self.std_range = err


  # === _SET_RATE_ERROR ===
  # change the standard deviation of pseudorange-rate error
  #
  # INPUTS:
  #   err   double or Nx1   pseudorange-rate error [m/s]
  #
  def set_rate_error(self, err):
    # update error
    self.std_rate = err


  # === SET_RANGES ===
  # set psuedoranges to be used in next estimation/filter
  # 
  # INPUTS:
  #   ranges  Nx1   vector of pseudoranges
  #
  def _set_ranges(self, ranges):
    # update ranges
    self.ranges = ranges


  # === _SET_RATES ===
  # set psuedoranges-rates to be used in next estimation/filter
  # 
  # INPUTS:
  #   rates   Nx1   vector of pseudorange-rates
  #
  def _set_rates(self, rates):
    # update rates
    self.rates = rates


  # === _SET_SATELLITE_POS ===
  # set satellite position to be used in next estimation/filter
  # 
  # INPUTS:
  #   pos     3x1   vector of ECEF position coordinates
  #
  def _set_satellite_pos(self, pos):
    # update position
    self.sv_pos = pos


  # === _SET_SATELLITE_VEL ===
  # set satellite velocity to be used in next estimation/filter
  # 
  # INPUTS:
  #   vel     3x1   vector of ECEF velocity coordinates
  #
  def _set_satellite_vel(self, vel):
    # update velocity
    self.sv_vel = vel


  # === _SET_POSITION ===
  # set position to be used in next estimation/filter
  # 
  # INPUTS:
  #   pos     3x1   vector of ECEF position coordinates
  #
  def _set_position(self, pos):
    # update position
    self.pos = pos


  # === _SET_VELOCITY ===
  # set velocity to be used in next estimation/filter
  # 
  # INPUTS:
  #   vel     3x1   vector of ECEF velocity coordinates
  #
  def _set_velocity(self, vel):
    # update velocity
    self.vel = vel


# ------------------------------------------------------------------------------------------------ #
  # === KALMAN_FILTER ===
  # run kalman filter to smooth position/velocity estimation
  #
  def kalman_filter(self, **kwargs):
    # parse keyword arguments
    self.__parse_rcvr_kwargs(kwargs)

    # measurement domain update
    if self.type == 'meas':
      # predict
      self.kf_x, self.kf_P = \
      kfm.predict(self.kf_x, self.kf_P, self.psd_acc, self.psd_clkp, self.psd_clkf, self.dt)
      
      # correct
      self.kf_x, self.kf_P = \
      kfm.correct(self.kf_x, self.kf_P, 
                  self.sv_pos, self.sv_vel, self.ranges, self.rates, self.std_range, self.std_rate)
      
    # position domain update
    elif self.type == 'pos':
      # predict
      self.kf_x, self.kf_P = \
      kfp.predict(self.kf_x, self.kf_P, self.psd_acc, self.dt)

      # correct
      self.kf_x, self.kf_P = \
      kfp.correct(self.kf_x, self.kf_P, self.pos, self.vel, self.std_pos, self.std_vel)

  
  # === LEAST_SQUARES ===
  # run least squares to estimate position/velocity
  #
  def least_squares(self, **kwargs):
    # parse keyword arguments
    self.__parse_rcvr_kwargs(kwargs)

    # run least squares
    x, P, _ = ls.calcPosVel(self.sv_pos, self.sv_vel, self.ranges, self.rates, self.std_range)
    
    # update kalman filter states
    if self.type == 'meas':
      self.kf_x = x
      self.kf_P = P
    elif self.type == 'pos':
      self.kf_x = x[:6]
      self.kf_P = x[:6,:6]



# ------------------------------------------------------------------------------------------------ #
  # === __PARSE_RCVR_KWARGS ===
  # parse all keyword arguments
  #
  # INPUTS:
  #   kwargs  dict  receiver specifications, defaults to "bad_gps"
  #                   - 'std_pos': position error standard deviation [m]
  #                   - 'std_vel': velocity error standard deviation [m/s]
  #                   - 'std_range': pseudorange error standard deviation [m]
  #                   - 'std_rate': pseudorange-rate error standard deviation [m/s]
  #                   - 'psd_acc': acceleration induced error power spectral density [m^2/s^3]
  #                   - 'psd_clkp': clock phase induced error power spectral density [m^2/s]
  #                   - 'psd_clkf': clock frequency induced error power spectral density [m^2/s^3]
  #                   - 'freq': receiver update frequency [Hz]
  #                   - 'sv_pos': satellite ecef position [m]
  #                   - 'sv_vel': satellite ecef velocity [m/s]
  #                   - 'ranges': pseudorange measuremtns [m]
  #                   - 'rates': pseudorange-rate measurements [m/s]
  #                   - 'pos': user ecef position measurement [m]
  #                   - 'vel': user ecef velocity measurement [m/s]
  #
  def __parse_rcvr_kwargs(self, **kwargs):
    # check keyword arguments
    if 'type' in kwargs:
      self.type = kwargs['type']

    if 'std_pos' in kwargs:
      self.std_pos = kwargs['std_pos']
      
    if 'std_vel' in kwargs:
      self.std_vel = kwargs['std_vel']

    if 'std_range' in kwargs:
      self.std_range = kwargs['std_range']

    if 'std_rate' in kwargs:
      self.std_rate = kwargs['std_rate']

    if 'psd_acc' in kwargs:
      self.psd_acc = kwargs['psd_acc']

    if 'psd_clkp' in kwargs:
      self.psd_clkp = kwargs['psd_clkp']

    if 'psd_clkf' in kwargs:
      self.psd_clkf = kwargs['psd_clkf']

    if 'freq' in kwargs:
      self.freq = kwargs['freq']
    self.dt = 1 / self.freq

    if 'sv_pos' in kwargs:
      self.sv_pos = kwargs['sv_pos']

    if 'sv_vel' in kwargs:
      self.sv_vel = kwargs['sv_vel']

    if 'pos' in kwargs:
      self.pos = kwargs['pos']

    if 'vel' in kwargs:
      self.vel = kwargs['vel']

    if 'ranges' in kwargs:
      self.ranges = kwargs['ranges']

    if 'rates' in kwargs:
      self.rates = kwargs['rates']
