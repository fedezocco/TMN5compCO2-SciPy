"""Implemented by Federico Zocco
    Last update: 5 February 2025 
    
Implementation of the dynamics of the atmospheric CO_2 modelled
in [1].

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

import matplotlib.pyplot as plt

# Import CO_2 dynamics of digester and microalgae: 
from Digester6States_ICDC import m_dot12, t
from Microalgae_openLoop import m_dot23  


# CO_2 dynamics in the atmosphere: 
dm2_dt = m_dot12 - m_dot23 # NOTE: these two quantities have the same 
# measurement unit since mmol/(Ld) = \mumol/(\mum^3 d)

# Plots:
fig = plt.figure(figsize=(10, 10))
plt.plot(t, dm2_dt, 'k-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (d)", fontsize=35)
plt.ylabel(r"$CO_2$ accumul. rate, $\frac{dm_2}{dt} \, \left(\frac{mmol}{Ld}\right)$", fontsize=35)  
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)