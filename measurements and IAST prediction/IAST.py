"""
IAST prediction of hydrogen co-adsorption

Soprtion pressures taken from: 
https://pure.tudelft.nl/ws/portalfiles/portal/154160336/RUPTURA_simulation_code_for_breakthrough_ideal_adsorption_solution_theory_computations_and_fitting_of_isotherm_models.pdf
"""

import numpy as np
from scipy import optimize
import math
import matplotlib.pyplot as plt

R = 8.314 # ideal gas constant J/(molK)

# fluid phase mole fractions
# 1: water, 2: hydrogen
y_1 = 777e-6
y_2 = 1-y_1
T = 20+273
p = 30*1e5

# Langmuir-Freundlich Parameters from FitParameters.py
molmasse_H20 = 18/(10**(3))    # kg/mol
molmasse_H2 = 2.01588/(10**(3))
q_sat_1 = 0.3284/molmasse_H20  # [kg/kg]*molmasse = [mol/kg]
B0 = 4.37e-6                   # [1/Pa^n]
B1 = 27482.81 
n1 = 0.3896
n2 = 53.76        
n_LF = n1 + n2/T
b_LF = B0*math.exp(B1/(R*T))

# Calculate sorption pressure and isotherm for water
def Langmuir_sp(psi):
    return ((math.exp(n_LF*psi/q_sat_1)-1)/b_LF)**(1/n_LF)

def Langmuir_q(p_1):
    return q_sat_1*(b_LF*p_1**n_LF)/(1+b_LF*p_1**n_LF)

# Sips parameter for hydrogen (from Streb and Mazotti)
q_sat_2 = 5.013      # mol/kg
B_1 = 1.034*1e-9     # in 1/Pa: 
B_2 = 9.453          # kJ/mol
R_Sips = 8.314/1e3   # kJ/mol/K
b = B_1*math.exp(B_2/(R_Sips*T))
s = 1.006

# Calculate sorption pressure and isotherm for hydrogen
def Sips_sp(psi):
    return 1/b*(math.exp(psi/((1/s)*q_sat_2))-1)**(1/s)

def Sips_q(p_2):
    return q_sat_2*(b*p_2)**s/(1+(b*p_2)**s)

# Calculate root to find reduced grand potential psi
def root_function(psi):
    return (y_1*p)/Langmuir_sp(psi) + (y_2*p)/Sips_sp(psi) - 1

psi = optimize.fsolve(root_function, x0=2)
print("Grand potential:", psi)

# Sorption pressures
p_1 = Langmuir_sp(psi)
p_2 = Sips_sp(psi)

# Adsorbed mole fractions
x_1 = y_1*p/p_1
x_2 = y_2*p/p_2

# Total adsorbed amount
q_T = 1/(x_1/Langmuir_q(p_1) + x_2/Sips_q(p_2))

# Seperate adsorbed amount in kg/mol
q_1 = x_1*q_T
q_2 = x_2*q_T

# Calculate IAST prediction as a function of pressure
p_vector = np.arange(0.01,30,0.01)*1e5
q_1_list = []
q_2_list = []

for i in range(len(p_vector)):
    p = p_vector[i]
    psi = optimize.fsolve(root_function, x0=2)
    # Sorption pressures
    p_1 = Langmuir_sp(psi)
    p_2 = Sips_sp(psi)

    # Adsorbed mole fractions
    x_1 = y_1*p/p_1
    x_2 = y_2*p/p_2

    # Total adsorbed amount
    q_T = 1/(x_1/Langmuir_q(p_1) + x_2/Sips_q(p_2))

    # Seperate adsorbed amount in kg/mol
    q_1 = x_1*q_T
    q_2 = x_2*q_T
    
    q_1_list.append(q_1)
    q_2_list.append(q_2)

langmuir_vector = Langmuir_q(y_1*p_vector)
Sips_vector = Sips_q(y_2*p_vector)

fig, ax = plt.subplots()
ax.plot(y_1*p_vector, np.array(q_1_list)*molmasse_H20, label='IAST water component', color = 'blue')
ax.plot(y_1*p_vector, langmuir_vector*molmasse_H20, label='Pure water component', linestyle='--', color = 'orange')
plt.ylabel("adsorbed amount [kg/kg]")
plt.xlabel("pressure (Pa)")
plt.grid(True)
plt.legend(loc="best")
plt.show()

fig, ax = plt.subplots()
ax.plot(y_2*p_vector, np.array(q_2_list)*molmasse_H2, label='IAST hydrogen component', color = 'blue')
ax.plot(y_2*p_vector, Sips_vector*molmasse_H2, label='Pure hydrogen component', linestyle='--', color = 'orange')
plt.ylabel("adsorbed amount [kg/kg]")
plt.xlabel("pressure (Pa)")
plt.grid(True)
plt.legend(loc="best")

plt.show()

print("Maximum absolute deviation water: ", max(abs(np.array(q_1_list)*molmasse_H20-langmuir_vector*molmasse_H20)))
print("Maximum value of hydrogen co-adsorption: ", max(q_2_list))