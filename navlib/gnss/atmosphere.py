#========================================= atmosphere.py ==========================================#
#                                                                                                  #
#   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       #
#   super sad and unfortunate for me. Proprietary and confidential.                                #
#                                                                                                  #
# ------------------------------------------------------------------------------------------------ #
#                                                                                                  #
#   @file                                                                                          #
#   @brief    Common atmospheric corrections                                                       #
#   @author   Daniel Sturdivant <sturdivant20@gmail.com> <dfs0012@auburn.edu>                      #
#   @date     November 2023                                                                        #
#                                                                                                  #
#==================================================================================================#

import numpy as np
from numba import njit
from navlib.constants import R2D, F_L1, c

# TODO: test to see if these work

# === klobuchar ===
# the Klobuchar ionosphere correction model (Groves appendix G.7.2)
# @njit(cache=True, fastmath=True)
def klobuchar(alpha: np.ndarray, 
              beta: np.ndarray, 
              dt: np.double, 
              el: np.double, 
              az: np.double, 
              lla: np.ndarray, 
              carr_freq: np.double, 
              unit: str='degrees'):
  # unpack vectors
  a0, a1, a2, a3 = alpha
  b0, b1, b2, b3 = beta

  unit = unit.lower()
  lat, lon, h = lla
  if unit == 'degrees' or unit == 'd':
    lat /= R2D
    lon /= R2D
    az /= R2D
    el /= R2D

  PSI = (0.1352 / (el + 0.3456)) - 0.06912        # Earth centered angle (G.45)
  LI = np.mod(lat + PSI*np.cos(az), 2.614)        # Sub-ionospheric latitude (G.46)
  if LI > 1.307: LI -= 1.307
  LAMB = lon + PSI*(np.sin(az) / np.cos(LI))      # Sub-ionospheric longitude (G.47)
  LM = (LI + 0.201*np.cos(LAMB - 5.080)) / np.pi  # Geo-magnetic latitude (G.48)
  T = np.mod(dt, 86400) + 1.375e4*LAMB            # Sub-ionospheric time point (G.49)

  ALPHA = a0 + a1*LM + a2*LM**2 + a3*LM**3
  BETA = b0 + b1*LM + b2*LM**2 + b3*LM**3
  XS = 2*np.pi*(T - 50400) / BETA                 # G.51

  # G.50
  if np.abs(XS) > 1.57:
    err = 5e-9 * (1 + 0.516*(1.67 - el)**3) * c
  else:
    err = (5e-9 + ALPHA*(1 - XS**2/2 + XS**4/24)) * (1 + 0.516*(1.67 - el)**3) * c

  return (F_L1 / carr_freq)**2 * err              # G.52


# === stanag ===
# the NATO Standardization agreement (STANAG) troposphere model (Groves Ch. 9.3.2)
# @njit(cache=True, fastmath=True)
def stanag(lla: np.ndarray, el: np.double, unit: str='degrees'):
  _,_,h = lla
  unit = unit.lower()
  if unit == 'degrees' or unit == 'd':
    el /= R2D

  # Groves 9.90
  if h < 1000:
    err = 2.464 - 3.248e-4*h + 2.2395e-8*h*h
  elif h < 9000:
    err = 2.284*np.exp(-0.1226*(1e-3*h - 1)) - 0.122
  else:
    err = 0.7374*np.exp(1.2816 - 1.424e-4*h)

  # Groves 9.91
  return err / (np.sin(el) + (0.00143 / (np.tan(el) + 0.0455)))


# === waastropo ===
# Initial WAAS tropospheric delay model (Groves appendix G.7.4)
# @njit(cache=True, fastmath=True)
def waastropo(lla: np.ndarray, el: np.double, day: np.int32, unit: str='degrees'):
  lat, lon, h = lla
  unit = unit.lower()
  if unit == 'degrees' or unit == 'd':
    lat /= R2D
    lon /= R2D
    el /= R2D

  if lat < 0:
    dN = 3.61e-3*h*np.cos(2*np.pi*(day-335)/365) \
       + (0.1*np.cos(2*np.pi*(day-30)/365) - 0.8225)*np.abs(lat)          # Groves G.57
  else:
    dN = 3.61e-3*h*np.cos(2*np.pi*(day-152)/365) \
       + (0.1*np.cos(2*np.pi*(day-213)/365) - 0.8225)*np.abs(lat)         # Groves G.58
    
  if h > 1500:
    err = 2.484*(1 + 1.5363e-3*np.exp(-2.133e-4*h)*dN) * \
          np.exp(-1.509e-4*h) / np.sin(el + 6.11e-3)                      # Groves G.56
  else:
    err = 2.506*(1 + 1.25e-3*dN) * (1 - 1.264e-4*h) / np.sin(el + 6.11e-3)# Groves G.55

  return err