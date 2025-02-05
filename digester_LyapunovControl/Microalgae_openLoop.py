"""Implemented by Federico Zocco
    Last update: 5 February 2025 
    
Implementation of the microalgae-compartment in open loop as detailed in [1],
i.e., without a controller.

References:
    [1] Zocco, F., García, J. and Haddad, W.M., 2025. Circular 
    microalgae-based carbon control for net zero. arXiv 
    preprint: arXiv:2502.02382.
    [2] Vatcheva, I., De Jong, H., Bernard, O. and Mars, N.J., 2006. Experiment 
    selection for the discrimination of semi-quantitative models of dynamical systems. 
    Artificial Intelligence, 170(4-5), pp.472-506.
    [3] Bernard, O., 2011. Hurdles and challenges for modelling and control of 
    microalgae for CO2 mitigation and biofuel production. Journal of Process 
    Control, 21(10), pp.1378-1389.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


#######################Simulator settings#####################
# Simulation time:
t_final = 15

# Model parameters (from Fig. 8 in [2] and from [3]):
mu_ALG = 1.2 # 1/day
K_sI = 0 # value not found, so at the moment it is set as 0 
K_iI = 295  
K_S = 0.0012
Y = 0.5  
T_h = 1/0.45 # it is 1/D, with D taken from Table 1 in [2]
S_in = 100

# Model parameters added by this paper (i.e., [1]): 
K_CO2 = 0.3 # 0 < K_CO2 < 1; value chosen since not found



# Initial conditions:
I_bar = 50 # value chosen as a "frequent" value in Fig. 7 of [3]
X_ALG_ini = 26 # from Table 1 in [2]
S_ini = 0 # from Table 1 in [2] 
X_ini = np.array([X_ALG_ini, S_ini])  

# Inputs (with open loop):
I = I_bar
################################################################

# Equations in state space form:
def microalgae_openLoop(X, t=0): 
    mu = mu_ALG*(I/(I+K_sI+(I**2/K_iI)))*(X[1]/(X[1]+K_S))
    rho = (1/Y)*mu 
    
    return np.array([mu*X[0] - (1/T_h)*X[0],
        (1/T_h)*(S_in - X[1]) - rho*X[0]])

# Numerical solution:
t = np.linspace(0, t_final, 1000)
X, infodict = integrate.odeint(microalgae_openLoop, X_ini, t,
mxstep = 1000, full_output = True)
X_ALG, S = X.T


# Absorption of CO2:
m_dot23 = K_CO2*(1/Y)*mu_ALG*(I/(I+K_sI+(I**2/K_iI)))*(S/(S+K_S)) # i.e., K_CO2*rho


# Plots
fig = plt.figure(figsize=(10, 10))
plt.plot(t, X_ALG, 'r-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (d)", fontsize=35)
plt.ylabel(r"Algal biomass, $X_{ALG} \, \left(\frac{\mu m^3}{L}\right)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, S, 'k-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (d)", fontsize=35)
plt.ylabel(r"Nutrient, $S \, \left(\frac{\mu mol}{L}\right)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, m_dot23, 'b-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (d)", fontsize=35)
plt.ylabel(r"$CO_2$ flow, $\dot{m}_{2,3} \, \left(\frac{\mu mol}{\mu m^3d}\right)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)