'''
|===================================== navlib/constants.py ========================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/constants.py                                                                   |
|  @brief    Common navigational constants.                                                        |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from navlib.attitude.skew import skew

# Matrices
I3 = np.eye(3, dtype=np.double)
Z3 = np.zeros((3,3), dtype=np.double)
Z32 = np.zeros((3,2), dtype=np.double)
Z23 = Z32.T

# Pi values
two_pi = 2*np.pi
half_pi = 0.5*np.pi
R2D = 180.0 / np.pi
D2R = np.pi / 180.0

# WGS84 Constants
Ro = 6378137.0      # Equatorial radius (semi-major axis)
Rp = 6356752.31425  # Polar radius (semi-major axis)
e = 0.0818191908425 # WGS84 eccentricity
e2 = e*e            # WGS84 eccentricity squared
mu = 3.986004418e14 # wgs84 earth gravitational constant
f = (Ro - Rp) / Ro  # wgs84 flattening
J2 = 1.082627E-3;   # wgs84 earth second gravitational constant

# Earth rotational constant
w_ie = 7.292115e-5
omega_ie = np.array([0.0, 0.0, w_ie], dtype=np.double)
OMEGA_ie = skew(omega_ie)

# LLA Unit conversions
lla_rad2deg = np.array([R2D, R2D, 1.0], dtype=np.double)
lla_deg2rad = np.array([D2R, D2R, 1.0], dtype=np.double)

# Speed of light
c = 299792458
F_L1 = 1575.42e6                # GPS L1 frequency [Hz]
F_L2 = 1227.60e6                # GPS L2 frequency [Hz]
F_L5 = 1176.45e6                # GPS L5 frequency [Hz]
L_L1 = c / F_L1                 # GPS L1 wavelength [m]
L_L2 = c / F_L2                 # GPS L2 wavelength [m]
L_L5 = c / F_L5                 # GPS L5 wavelength [m]

# IMU conversions
G = 9.80665       # Earth's gravity constant [m/s^2]
G2T = 1e-4        # Gauss to Tesla
FT2M = 0.3048     # Feet to meters