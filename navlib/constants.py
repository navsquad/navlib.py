'''
|================================ navlib/coordinates/position.py ==================================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/coordinate/tangent.py                                                          |
|  @brief    Common coordinate frame transformation matrices.                                      |
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
rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0

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
lla_rad2deg = np.array([rad2deg, rad2deg, 1.0], dtype=np.double)
lla_deg2rad = np.array([deg2rad, deg2rad, 1.0], dtype=np.double)

# Speed of light
c = 299792458