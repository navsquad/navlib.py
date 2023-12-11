'''
|============================== navlib/examples/test_coordinates.py ===============================|
|                                                                                                  |
|  Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be        |
|  super sad and unfortunate for me. Proprietary and confidential.                                 |
|                                                                                                  |
|--------------------------------------------------------------------------------------------------| 
|                                                                                                  |
|  @file     navlib/coordinate/test_coordinates.py                                                 |
|  @brief    Test coordinate transformations from 'coordinates' folder.                            |
|  @ref      Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems           |
|              - (2013) Paul D. Groves                                                             |
|  @author   Daniel Sturdivant <sturdivant20@gmail.com>                                            | 
|  @date     December 2023                                                                         |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
from navlib.constants import lla_rad2deg, lla_deg2rad
from navlib.coordinates.position import *
from navlib.coordinates.velocity import *

def test_position(init_lla_deg, lla0_deg, dt):
  # convert lla to radians
  init_lla_rad = init_lla_deg * lla_deg2rad
  lla0_rad = lla0_deg * lla_deg2rad
  
  # convert to ecef
  ecef1 = lla2ecef(init_lla_rad)
  
  # convert to eci
  eci1 = ecef2eci(ecef1, dt)
  
  # convert to ned
  ned1 = eci2ned(eci1, lla0_rad, dt)
  
  # convert to lla
  temp_lla_rad = ned2lla(ned1, lla0_rad)
  temp_lla_deg = temp_lla_rad * lla_rad2deg
  
  # convert to enu
  enu2 = lla2enu(temp_lla_rad, lla0_rad)
  
  # convert to eci
  eci2 = enu2eci(enu2, lla0_rad, dt)
  
  # convert to ecef
  ecef2 = eci2ecef(eci2, dt)
  
  # convert to lla
  last_lla_rad = ecef2lla(ecef2)
  last_lla_deg = last_lla_rad * lla_rad2deg
  
  # print results
  print("Position Tests:")
  print(f"Initial LLA: [{init_lla_deg[0]:.6f}, {init_lla_deg[1]:.6f}, {init_lla_deg[2]:.1f}]")
  print(f"Middle LLA: [{temp_lla_deg[0]:.6f}, {temp_lla_deg[1]:.6f}, {temp_lla_deg[2]:.1f}]")
  print(f"Final LLA: [{last_lla_deg[0]:.6f}, {last_lla_deg[1]:.6f}, {last_lla_deg[2]:.1f}]\n")
  
  
def test_velocity(init_ned_pos, init_ned_vel, lla0_deg, dt):
  lla0_rad = lla0_deg * lla_deg2rad
  
  # convert to ecef
  ecef1 = ned2ecefv(init_ned_vel, lla0_rad)
  
  # convert to enu
  enu_p1 = np.array([init_ned_pos[1], init_ned_pos[0], -init_ned_pos[2]], dtype=np.double)
  enu_v1 = ecef2enuv(ecef1, lla0_rad)
  
  # convert to eci
  eci_p1 = enu2eci(enu_p1, lla0_rad, dt)
  eci_v1 = enu2eciv(enu_p1, enu_v1, lla0_rad, dt)
  
  # convert to ned
  temp_ned_vel = eci2nedv(eci_p1, eci_v1, lla0_rad, dt)
  
  # print results
  print("Velocity Tests:")
  print(f"Initial NED Velocity: [{init_ned_vel[0]:.6f}, {init_ned_vel[1]:.6f}, {init_ned_vel[2]:.1f}]")
  print(f"Final NED Velocity: [{temp_ned_vel[0]:.6f}, {temp_ned_vel[1]:.6f}, {temp_ned_vel[2]:.1f}]\n")
  
  
if __name__ == '__main__':
  lla = np.array([32.2, -85.5, 250.0], dtype=np.double)
  lla0 = np.array([32.1, -85.4, 250.0], dtype=np.double)
  dt = 1.0
  test_position(lla, lla0, dt)
  
  ned_p = np.array([100.0, 25.0, -10.0], dtype=np.double)
  ned_v = np.array([1.0, 0.5, 0.0], dtype=np.double)
  test_velocity(ned_p, ned_v, lla0, dt)
  