# Hydrogen-Adsorption
## BalanceEquations.py
This repository presents the simulation code used in the paper with the title *Simulation of Hydrogen Drying via Adsorption in Offshore Hydrogen Production* (doi is followed).
The file *BalanceEquations.py* is the main file in which the balance equations of an adsorption process are solved using the Finite-Difference Method and explicit discretization.
It includes the mass balance equation for the solid and gas phase, the energy balance equation and the Ergun-equation for pressure loss calculation. Furthermore it includes parameters for the Langmuir-Freundlich isotherm, 
which are fitted based on our measurement data.

The *BalanceEquations.py* file reads one *mass_flow_xms.npy* datafile at a time, where x stands for the mean wind velocity in m/s. The values in the .npy files are the sum of the mass flows in mol/s exiting 3 x 5MW PEM-electrolyzers.
The .npy files contain mass flows for a simulation period of 3 hours with a discretization step of dt_IWES = 0.01 sec. They are being cut to 1 hour.

In order to calculate the adsorption loading the file *BalanceEquations.py* simply has to be executed. Prior to that, a .npy with the desired wind speed has to be chosen (other .npy files need to be commented out).

## Folder "measurements and IAST prediction"
The folder "measurements and IAST prediction" holds all relevant supplementary files. 

*measurements and IAST prediction/PlotIsobars.py* shows our measurement data for adsorption isobars at pressures 1223 Pa and 23339 Pa for $T \in$ {24.91, 34.86, 44.86, 54.88, 64.89, 74.86, 84.86, 94.86, 104.87, 114.85, 124.82, 134.83, 144.79}.

*measurements and IAST prediction/FitParameters.py* fits the parameters to the Langmuir-Freundlich (LF) isotherm with the measurement data and support points form modified potential theory. This file also compares the resulting curve to a LF curve from existing literature (M.Gaeini).

*measurements and IAST prediction/IAST.py* predicts the influence of hydrogen co-adsorption with the Ideal Adsorption Solution Theory (IAST). 

*measurements and IAST prediction/individual_calculation_output.py* holds all support points obtained from modified potential theory.
