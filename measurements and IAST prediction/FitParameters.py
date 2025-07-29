"""
Fit parameters with measurement data and support points form modified potential theory 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

R = 8.314                  # ideal gas constant [J/(mol*K)]
M_H2O = 0.018              # molar mass of water [kg/mol]

# import support points form modified potential theory
xlsx_file = pd.ExcelFile("individual_calculation_output.xlsx", engine="openpyxl")
print(xlsx_file.sheet_names)

df_sheet1 = xlsx_file.parse("T1")
df_sheet2 = xlsx_file.parse("T2")
df_sheet3 = xlsx_file.parse("T3")

# erase first row
df_sheet1 = df_sheet1.iloc[1:]
df_sheet2 = df_sheet2.iloc[1:]
df_sheet3 = df_sheet3.iloc[1:]

p = np.array([1223, 2339]) # pressure in Pa
T = np.array([24.91, 34.86, 44.86, 54.88, 64.89, 74.86, 84.86, 94.86, 104.87, 114.85, 124.82, 134.83, 144.79])
grid1, grid2 = np.meshgrid(T+273.15, p)
data = np.stack((grid2.flatten(), grid1.flatten()), axis=-1)

p_add = np.arange(100, 2600, 100)
for T in [25, 115, 145]:
    T_add = np.ones(25)*(T+273.15)
    combined = np.vstack((p_add, T_add))
    combined = combined.T
    data = np.vstack((data, combined))

W_curve_fit = np.array([307.3629153, 294.2579607, 280.5618051, 267.506117300, 253.604683600, 239.227004700, 225.132609700, 203.237646400, 181.43711100, 158.462384800, 137.696451200, 119.34458800, 102.749905600,
               327.113355500, 313.211807700, 300.3736092, 288.007554300, 274.615100400, 261.464876600, 247.65365190, 228.2834503000, 211.787165900, 191.755963400, 169.737652400, 149.037237800, 126.36613700])

for sheet in [df_sheet1, df_sheet2, df_sheet3]:
    W_curve_fit = np.hstack((W_curve_fit, sheet.iloc[:,2].to_numpy()))

W_curve_fit = W_curve_fit/1e3

# Langmuir-Freundlich with parameters from Gaeini 
def langmuir_freundlich_gaeini(T, p):
    X_sat = 0.3240  # kg/kg = 0.018 kg/mol*18 mol/kg
    b0 = 0.000308   # [1/Pa^n]
    deltaE = 18016
    n1 = -0.3615
    n2 = 274.23
    
    b_LF = b0*np.exp(deltaE/(R*T))
    n_LF = n1 + n2/T
    X_gl = (X_sat*b_LF*p**n_LF)/(1+b_LF*p**n_LF)
    return X_gl

# Determine new LF parameters
def langmuir_freundlich(data, b0, deltaE, n1, n2, X_0LF):
    b_LF = b0*np.exp(deltaE/(R*data[:,1]))
    n_LF = n1 + n2/data[:,1]
    X_gl = (X_0LF*b_LF*data[:,0]**n_LF)/(1+b_LF*data[:,0]**n_LF)
    return X_gl

def model_func_wrapper_lf(x_flat, b0, deltaE, n1, n2, X_0LF):
    x = x_flat.reshape(-1, 2)
    return langmuir_freundlich(x, b0, deltaE, n1, n2, X_0LF)

data_flat = data.flatten()

# starting conditions from Gaeinis parameters
X_sat = 0.3240
B0 = 0.000308
B1 = 18016
n1 = -0.3615
n2 = 274.23

p1 = [B0, B1, n1, n2, X_sat]
popt1, pcov1 = curve_fit(model_func_wrapper_lf, data_flat, W_curve_fit, p0=p1)

# Get the optimized parameters
B0_new, B1_new, n1_new, n2_new, X_sat_new = popt1

print(f"Optimized parameters: B0={B0_new}, B1={B1_new}, n1={n1_new}, n2={n2_new}, X_sat={X_sat_new}")

# plot pressure against loading
p_range = np.linspace(0, 2500, 100)

# Temperature 25°C
plt.figure()
T_test = 25+273.15
lf_gaeini = langmuir_freundlich_gaeini(T_test, p_range)
test = np.vstack((p_range, np.ones_like(p_range)*T_test))
lf = langmuir_freundlich(test.T, B0_new, B1_new, n1_new, n2_new, X_sat_new)
plt.plot(p_range, lf, label = 'new fitted LF Isotherm')

plt.scatter(df_sheet1.iloc[:, 0], df_sheet1.iloc[:, 2]/1e3, s=10, label='points from modified potential theory')
plt.plot(p_range, lf_gaeini, label = 'LF Isotherm from Gaeini', color = 'green')
plt.scatter(np.array([1223, 2339]), np.array([307.3629153, 327.113355500])/1e3, label='measured points', marker='s', s=10, zorder=3)

plt.xlabel('pressure [Pa]')
plt.ylabel('$X_{s,eq}$ [kg/kg]')
plt.title("25°C")
plt.grid(True)
plt.legend()
plt.show()

# Temperature 115°C
plt.figure()
T_test = 115+273.15
lf_gaeini = langmuir_freundlich_gaeini(T_test, p_range)
test = np.vstack((p_range, np.ones_like(p_range)*T_test))
lf = langmuir_freundlich(test.T, B0_new, B1_new, n1_new, n2_new, X_sat_new)
plt.plot(p_range, lf, label = 'new fitted LF Isotherm')

plt.scatter(df_sheet2.iloc[:, 0], df_sheet2.iloc[:, 2]/1e3, s=10, label='points from modified potential theory')
plt.plot(p_range, lf_gaeini, label = 'LF Isotherm from Gaeini', color = 'green')
plt.scatter(np.array([1223, 2339]), np.array([158.462384800, 191.755963400])/1e3, label='measured points', marker='s', s=10, zorder=3)
plt.xlabel('pressure [Pa]')
plt.ylabel('$X_{s,eq}$ [kg/kg]')
plt.title("115°C")
plt.grid(True)
plt.legend()
plt.show()

# Temperature 145°C
plt.figure()
T_test = 145+273.15
lf_gaeini = langmuir_freundlich_gaeini(T_test, p_range)
test = np.vstack((p_range, np.ones_like(p_range)*T_test))
lf = langmuir_freundlich(test.T, B0_new, B1_new, n1_new, n2_new, X_sat_new)
plt.plot(p_range, lf, label = 'new fitted LF Isotherm')

plt.scatter(df_sheet3.iloc[:, 0], df_sheet3.iloc[:, 2]/1e3, s=10, label='points from modified potential theory')
plt.plot(p_range, lf_gaeini, label = 'LF Isotherm from Gaeini', color = 'green')
plt.scatter(np.array([1223, 2339]), np.array([102.749905600, 126.36613700])/1e3, label='measured points', marker='s', s=10, zorder=3)
plt.xlabel('pressure [Pa]')
plt.ylabel('$X_{s,eq}$ [kg/kg]')
plt.title("145°C")
plt.grid(True)
plt.legend()
plt.show()