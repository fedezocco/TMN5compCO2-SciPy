"""
Implemented by Federico Zocco
    Last update: 5 February 2025

Initial-condition-dependent-controller (ICDC) proposed in [1].
References:
    [1] Zocco, F., García, J. and Haddad, W.M., 2025. Circular 
    microalgae-based carbon control for net zero. arXiv 
    preprint: arXiv:2502.02382. 
    [2] Campos-Rodríguez, A., García-Sandoval, J.P., 
    González-Álvarez, V. and González-Álvarez, A., 2019. 
    Hybrid cascade control for a class of nonlinear 
    dynamical systems. Journal of Process Control, 76, 
    pp.141-154.
    [3] Bernard, O., Hadj‐Sadok, Z., Dochain, D., Genovesi,
    A. and Steyer, J.P., 2001. Dynamical model development
    and parameter identification for an anaerobic 
    wastewater treatment process. Biotechnology and 
    bioengineering, 75(4), pp.424-438.
"""

import numpy as np
import math
from scipy import integrate
import matplotlib.pyplot as plt


#######################Simulator setting#####################
# Simulation time:
t_final = 10

# Model parameters (from Table 2 in [2]):
mu_max1 = 1.2 
mu_max2 = 0.744 
K_S1 = 7.1 
K_S2 = 9.28 
K_I2 = 16 
alpha = 0.5 
k_1 = 42.14 
k_2 = 116.5 
k_3 = 268 
S_1in = 30 
S_2in = 750
k_6 = 453 # value from [3], not [2]
K_a = 1.5*10**(-5) # from Bernard et al. 
B_in = 0
pH_in = 4.42 # from Table II in Bernard et al. (I choose the minimum value to align with the assumption of low pH used to neglect B_in in equation (11))
Z_in = B_in + (K_a*S_2in)/(K_a + 10**(-pH_in)) # equation (11) in Bernard et al. 
C_in = 40 # chosen by me as I do not find its value (the only value I didn't find) 
k_4 = 50.6 # from Table V in Bernard et al.
k_5 = 343.6 # from Table V in Bernard et al.
K_H = 16 # from Bernard et al.
P_T = 1 # from Bernard et al.
k_La = 19.8 # from Table III in Bernard et al.


# Model parameter added by this paper (i.e., [1]):
f_r = 0.15 # released fraction of CO_2


# Values at equilibrium SS6 of [2], p. 10 (only D_bar needs to be set):
D_bar = 0.5 # 0.05 < D < 1.2 for operational stability; interval took from [2], p. 11
S1_bar = (alpha*D_bar*K_S1) / (mu_max1-alpha*D_bar)
S2_bar = (((K_I2**2)*(mu_max2-alpha*D_bar)) / (2*alpha*D_bar)) - math.sqrt((((K_I2**2)*(mu_max2-alpha*D_bar))/(2*alpha*D_bar))**2 - K_S2*(K_I2**2))  
X1_bar = (S_1in-S1_bar) / (alpha*k_1)  
X2_bar = (k_2*(S_1in-S1_bar) + k_1*(S_2in-S2_bar)) / (alpha*k_1*k_3)
Z_bar = Z_in
# Added to write C_bar compactly:
Psi_bar = C_in-Z_bar+S2_bar+k_4*alpha*X1_bar+k_5*alpha*X2_bar # equation (46.5) in Bernard et al.
omega_bar = K_H*P_T+Psi_bar+((k_La+D_bar)/(k_La))*k_6*alpha*X2_bar # equation (47.5) in Bernard et al.
PC_bar = (omega_bar-math.sqrt(omega_bar**2-4*K_H*P_T*Psi_bar)) / (2*K_H) # equation (48) in Bernard et al.
CO2_bar = (1/(k_La+D_bar))*(k_La*K_H*PC_bar+D_bar*Psi_bar) # equation (46) in Bernard et al.
C_bar = CO2_bar - S2_bar + Z_bar # equation (19) in Bernard et al. 


# Initial conditions:
x1_tilde_ini = -1
x2_tilde_ini = 0.5
x3_tilde_ini = 1.0
x4_tilde_ini = 1.5
x5_tilde_ini = 0.8
x6_tilde_ini = -0.5
X_tilde_ini = np.array([x1_tilde_ini, x2_tilde_ini, x3_tilde_ini, x4_tilde_ini, x5_tilde_ini, x6_tilde_ini])


# Controller tuning:
T_MAX = 3.5
q = T_MAX/(1+T_MAX)  
p = (1/2) * ((np.sum((X_tilde_ini)**2))**(1/(1+T_MAX))) / ((T_MAX/(1+T_MAX))**2)
################################################################

# Equations in affine state space form (six-state model):
def digester_closedLoop(X_tilde, t): 
    
    mu1_tilde = mu_max1*((X_tilde[1]+S1_bar) / (X_tilde[1]+S1_bar+K_S1))
    mu2_tilde = mu_max2*((X_tilde[3]+S2_bar) / (X_tilde[3]+S2_bar+K_S2+(((X_tilde[3]+S2_bar)/K_I2)**2)))  
    Phi_tilde = X_tilde[5]+C_bar+X_tilde[3]+S2_bar-X_tilde[4]-Z_bar+K_H*P_T+(k_6/k_La)*mu2_tilde*(X_tilde[2]+X2_bar)  
    PC_tilde = (Phi_tilde - math.sqrt(Phi_tilde**2 - 4*K_H*P_T*(X_tilde[5]*C_bar+X_tilde[3]+S2_bar-X_tilde[4]-Z_bar)))/(2*K_H)
    qC_tilde = k_La*(X_tilde[5]+C_bar+X_tilde[3]+S2_bar-X_tilde[4]-Z_bar-K_H*PC_tilde)
    
    f1 = mu1_tilde*(X_tilde[0]+X1_bar) - alpha*D_bar*(X_tilde[0]+X1_bar) 
    f2 = -k_1*mu1_tilde*(X_tilde[0]+X1_bar) + D_bar*(S_1in-X_tilde[1]-S1_bar)
    f3 = mu2_tilde*(X_tilde[2]+X2_bar) - alpha*D_bar*(X_tilde[2]+X2_bar)
    f4 = k_2*mu1_tilde*(X_tilde[0]+X1_bar) - k_3*mu2_tilde*(X_tilde[2]+X2_bar) + D_bar*(S_2in-X_tilde[3]-S2_bar)                     
    f5 = D_bar*(Z_in-X_tilde[4]-Z_bar)
    f6 = D_bar*(C_in-X_tilde[5]-C_bar)-qC_tilde+k_4*mu1_tilde*(X_tilde[0]+X1_bar)+k_5*mu2_tilde*(X_tilde[2]+X2_bar)
    
    f = np.array([[f1],[f2],[f3],[f4],[f5],[f6]]) 
    
    G11 = - alpha*(X_tilde[0]+X1_bar)
    G22 = S_1in - X_tilde[1] - S1_bar 
    G33 = - alpha*(X_tilde[2]+X2_bar)
    G44 = S_2in - X_tilde[3] - S2_bar
    G55 = Z_in - X_tilde[4] - Z_bar
    G66 = C_in - X_tilde[5] - C_bar
    
    G = np.array([[G11, 0, 0, 0, 0, 0],[0, G22, 0, 0, 0, 0],[0, 0, G33, 0, 0, 0],[0, 0, 0, G44, 0, 0],[0, 0, 0, 0, G55, 0],[0, 0, 0, 0, 0, G66]]) 
    
    
    # Control law:
    Vprime = (2*p*q*(X_tilde[0]**2+X_tilde[1]**2+X_tilde[2]**2+X_tilde[3]**2+X_tilde[4]**2+X_tilde[5]**2)**(q-1))*np.array([[X_tilde[0], X_tilde[1], X_tilde[2], X_tilde[3], X_tilde[4], X_tilde[5]]])
    G_inv = np.linalg.inv(G)
    Phi_controller = - (1/2)*G_inv.dot((2*f + Vprime.transpose()))     
                             
    return (f + G.dot(Phi_controller)).flatten()    


# Numerical solution:
t = np.linspace(0, t_final, 1000)
X_tilde, infodict = integrate.odeint(digester_closedLoop, X_tilde_ini, t,
mxstep=1000, full_output = True)
x1_tilde, x2_tilde, x3_tilde, x4_tilde, x5_tilde, x6_tilde = X_tilde.T


# CO_2 production (called q_C in Bernard et al.):
mu2_tilde = mu_max2*((x4_tilde+S2_bar) / (x4_tilde+S2_bar+K_S2+(((x4_tilde+S2_bar)/K_I2)**2)))    
Phi_tilde = x6_tilde+C_bar+x4_tilde+S2_bar-x5_tilde-Z_bar+K_H*P_T+(k_6/k_La)*mu2_tilde*(x3_tilde+X2_bar)
PC_tilde = (Phi_tilde - np.sqrt(Phi_tilde**2 - 4*K_H*P_T*(x6_tilde*C_bar+x4_tilde+S2_bar-x5_tilde-Z_bar)))/(2*K_H)
qC_tilde = f_r*k_La*(x6_tilde+C_bar+x4_tilde+S2_bar-x5_tilde-Z_bar-K_H*PC_tilde)  
m_dot12 = qC_tilde 


# Plots
fig = plt.figure(figsize=(10, 10))
plt.plot(t, x1_tilde, 'r-', label = 'Acidogenic bacteria', linewidth=6)
plt.plot(t, x2_tilde, 'b-', label = 'Organic substrate', linewidth=6)
plt.plot(t, x3_tilde, 'g-', label = 'Methanogenic bacteria', linewidth=6)
plt.plot(t, x4_tilde, 'k-', label = 'Volatile fatty acids', linewidth=6)
plt.plot(t, x5_tilde, 'b--', label = 'Total alkalinity', linewidth=6)
plt.plot(t, x6_tilde, 'r--', label = 'Total inorganic carbon', linewidth=6)
plt.grid()
plt.legend(loc='best', prop={'size': 27})
plt.xlabel(r"Time, $t$ (d)", fontsize=35)
plt.ylabel(r"Translated state, $\tilde{\mathbf{x}}$", fontsize=35)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)


fig = plt.figure(figsize=(10, 10))
plt.plot(t, m_dot12, 'm-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (d)", fontsize=35)
plt.ylabel(r"$CO_2$ flow, $\dot{m}_{1,2} \, \left(\frac{mmol}{Ld}\right)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
  