import numpy as np 

MAGNET_M = 342.86                 # A·m^2 
MU0      = 4 * np.pi * 1e-7
test1 = 1
I3 = np.eye(3)
magnetisation = 8e3
LENGTH  = 0.054                     # m
RADIUS      = 0.00054                   # m
A_val   = np.pi * RADIUS**2            # m^2
I_val  = np.pi * RADIUS**4 / 4.0      # m^4
E_val  = 3.8e6   
