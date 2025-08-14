# Hydrogen-Adsorption
## Balance Equations
This repository presents code used by the paper "Simulation of Hydrogen Drying via Adsorption in Offshore Hydrogen Production" (doi is followed).
The file BalanceEquations.py is the main file in which the balance equations of an adsorption process through Finite-Difference Method using explicit discretization are solved.
It includes the mass balance equation for the solid and gas phase, an energy balance equation and the Ergun-equation for pressure loss calculation. Furthermore it includes parameters for the Langmuir-Freundlich isotherm, 
which is based on our measurement data.
This file incorporates one "mass_flow_xms.npy", where x stands for the mean wind velocity in m/s. The values in the .npy files are the sum of the mass flows in mol/s coming from 3x5 MW PEM-electrolyzers.
It contains mass flows for a simulation period of three hours with a discretization step of dt_IWES = 0.01 sec.

## Folder "measurements and IAST prediction"
The folder "measurements and IAST prediction" holds all the supplementary files. 
"measurements and IAST prediction/PlotIsobars.py" shows our measurement data for adsorption isobars at pressures 1223 Pa and 23339 Pa for $T \in [24.91, 34.86, 44.86, 54.88, 64.89, 74.86, 84.86, 94.86, 104.87, 114.85, 124.82, 134.83, 144.79]$.
"measurements and IAST prediction/FitParameters.py" fits the parameters to the Langmuir-Freundlich (LF) isotherm with measurement data and support points form modified potential theory and compares the resulting curve to LF curves from existing literature (Gaeini).
"measurements and IAST prediction/IAST.py" predicts the influence of hydrogen co-adsorption with the Ideal Adsorption Solution Theory (IAST). 
"measurements and IAST prediction/individual_calculation_output.py" holds all support points obtained from modified potential theory.
