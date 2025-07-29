"""
Explicit discretisation of the balance equations using the finite difference method
Gradients for variation in mass flow were adjusted,
concentration, initial temperature are assumed to be constant
Pressure can be either fixed or adapted

Important note on the model: when dz becomes smaller, you can see that X_s becomes larger. This is not wrong.
The reason for this is that water is absorbed more quickly over time in a smaller section (because it is loaded first).
If this section becomes larger, the load is distributed over a larger section and a kind of average is formed. 
A larger section is loaded more slowly. With fine simulation, the aim should be to select dz as small as necessary 
to give the best possible forecasts and not violate the CFL condition.
For coarse simulation (how long does a load take), a large dz could be sufficient.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt
import time

plot_graph = True          # if true, plot graphs
p_ergun_qu = False         # if true, calculate pressure loss with Ergun equation

N = 5                      # number of sections (without boundary sections)
t_end = 60*60              # total simulation time [s] -> #minutes*60s
t_end_minutes = t_end/60
dt = 0.001                 # time step [s]
dt_IWES = 0.01             # time step of wind data (we discretise more finely than IWES)
time_steps = int(t_end/dt)

## column paramete
d_column = 0.3             # diameter of column [m]
dz = 0.02                  # height of one section [m] 

M_H2 = 0.002016            # molar mass of hydrogen [kg/mol]
M_H2O = 0.018              # molar mass of water [kg/mol]
diff_vol_H2 = 6.12         # diffusion volume (VDI Wärmeatlas)
diff_vol_H2O = 13.1        # diffusion volume
R = 8.314                  # ideal gas constant [J/(mol*K)]

## ---------------------------------##
## --boundary conditions for z = 0--##
## ---------------------------------##         
c_A_in = 777e-6                # mole fraction of water in feed, constant
p_in = 30*1e5                  # total pressure at column entrance [Pa], constant
p_w_in = p_in*c_A_in           # partial pressure of water at entrance [Pa]
T_in_ads = 273.15+20           # gas temperature at entrance [K]
T_init = 273.15+20

# convert mole fraction to mass fraction
X_f_in = c_A_in*M_H2O/(c_A_in*M_H2O+(1-c_A_in)*M_H2) # [kg/kg]

## property values of water
rho_w = 998.16             # density of water at 20°C [kg/m3] (from NIST)
cp_w  = 4184               # specific heat capacity of water at 30 bar and 20°C

## property values of hydrogen
rho_H2 = 2.4376                                   # hydrogen density at 20°C, 30 bar [kg/m^3]
rho_f_in = 1/(X_f_in/rho_w+(1-X_f_in)/rho_H2)     # hydrogen density in feed [kg/m^3]
lambda_f = 0.186                                  # thermal conductivy hydrogen [J/(s*m*K)], at 20°C, 30 bar
cp_f = 14369                                      # specific heat capacity of hydrogen [J/(kg*K)], at 20°C, 30 bar

# Density in electrolyzer tank assumed to be constant, mass flow varies
# Choose between three documents which represent mass flows around wind velocity mean of 6 m/s, 10 m/s and 16 m/s
sum_ELY_massflow_mol_old = np.load('mass_flow_6ms.npy')
#sum_ELY_massflow_mol_old = np.load('mass_flow_10ms.npy')
#sum_ELY_massflow_mol_old = np.load('mass_flow_16ms.npy')

# Cut negative mass values, problem of IWES model at the beginning
# Cut model to one hour simulation time (instead of 3h) 
sum_ELY_massflow_mol = np.where(sum_ELY_massflow_mol_old < 0, 0, sum_ELY_massflow_mol_old)
sum_ELY_massflow_mol = sum_ELY_massflow_mol[0:360000]

mass_flow_kgs = sum_ELY_massflow_mol*(c_A_in*M_H2O+(1-c_A_in)*M_H2)        ## [kg/s]
mass_flow_m3s = mass_flow_kgs/rho_f_in                                     ## [m^3/s]
u_0_array_old = mass_flow_m3s/(np.pi*(d_column/2)**2)                      ## [m/s]

# interpolate, as IWES time step is larger than our time step
x = np.linspace(0, time_steps, u_0_array_old.size)
interpolated_function = interp1d(x, u_0_array_old, kind='linear')
new_x = np.linspace(x.min(), x.max(), time_steps)
u_0_array = interpolated_function(new_x)

# Check if CFL condition is satisfied
if u_0_array.any() < dz/dt: 
    print("CFL is satisfied.")
else: 
    print("CFL not satisfied.")

## property values of zeolite 13X
d_p = 0.002             # particle diameter [m]
d_p_macro = 300e-9      # macropore diameter [m] (Mette)
tort = 4                # tortuosity (Mette)
rho_s = 1150            # solid particle density of dry adsorbent [kg/m^3]
eps = 0.4               # bed porosity
eps_p = 0.6             # particle porosity
lambda_s = 0.4          # thermal conductivy of zeolite [J/(s*m*K)]
cp_s = 880              # specific heat capacity of zeolite [J/(kg*K)]

## ---------------------------------##
## ------Pressure-loss-Ergun--------##
## ---------------------------------##
eta_f = 0.882*10e-5      # dynamic viscosity of hydrogen
u_0_matrix = np.full((N+2, time_steps), np.nan)
for i in range(0,N+2):
    u_0_matrix[i,i:] = u_0_array[:time_steps-(i)]

# Replace all NaN with u_leer[0]
u_0_matrix[np.isnan(u_0_matrix)] = u_0_array[0]

# In our case, p=3e6 is sufficient, as only very small pressure losses due to low flow velocity
p_ergun = 3e6*np.ones((N+2, time_steps)) 

if p_ergun_qu == True:
    # effective viscosity of hydrogen
    eta_eff = eta_f*2.0*np.exp(3.5e-3*u_0_matrix*d_p*rho_f_in/eta_f)

    # Calculate pressure loss at matrix multiplication, long calculation times
    dp = dz*(150*(1-eps)**2/eps**3*(eta_eff/d_p**2)*u_0_matrix+1.75*(1-eps)/eps**3*rho_f_in/d_p*u_0_matrix**2)

    p_ergun[0,:] = p_in
    for t in range(time_steps):
        for i in range(1, N+2):
            p_ergun[i,t] = p_ergun[i-1,t]-dp[i-1,t]


## ---------------------------------##
## -----------Isotherm--------------##
## ---------------------------------##
X_0LF = 0.3284
b0 = 4.37e-6
b1 = 27482.8
n1 = 0.3896
n2 = 53.76

def langmuir_freundlich(T,p_H2O):
    b_LF = b0*np.exp(b1/(R*T))
    n_LF = n1 + n2/T
    X_gl = (X_0LF*b_LF*p_H2O**n_LF)/(1+b_LF*p_H2O**n_LF)
    delta_X_gl = (X_0LF*b_LF*n_LF*p_H2O**(n_LF-1))/(1+b_LF*p_H2O**n_LF)**2
    return X_gl, delta_X_gl

E = 1192250        # characteristic energy [J/kg]
n = 1.55           # Exponent - equilibrium approach

# Wagner equations constants, vaporization enthalpy of water
p_c = 220.64*1e5          # critical pressure of water [Pa]
T_c = 647.25              # critical temperature of water [K] 
A_V=2872019; B_V=0.28184; C_V=-0.109110; D_V=0.147096; E_V=0.044874; 

# specific heat capacity cp_ads
interp_func = UnivariateSpline(np.array([200,275,400,450]),np.array([800,1000,4100,4100]))

# thermal coefficient of water as interpolated version
beta_x = np.array([0.01, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250])+273.15
beta_y = np.array([-0.0000855, 0.0000821, 0.0002066, 0.0003056, 0.0003890, 0.0004624, 0.0005288, 0.0005900, 0.0006473, 0.0007019, 0.0007547, 0.001024, 0.001372, 0.001955])
beta_func = UnivariateSpline(beta_x, beta_y)
# In line 231, you can choose interpolated version with beta_func
beta = 0.207e-3

# effective axial thermal conductivity
Pe_lambda = u_0_matrix*d_p*rho_f_in*cp_f/lambda_f
k_s = lambda_s/lambda_f
B = 1.25*((1-eps)/eps)**(10/9)
C = 1-B/k_s
k_c = 2/C*(B/(C**2)*(k_s-1)/k_s*np.log(k_s/B)-(B+1)/2-(B-1)/C)
lambda_bed = lambda_f*(1-np.sqrt(1-eps) + np.sqrt(1-eps)*k_c)
L_ax = lambda_bed + Pe_lambda/2*lambda_f

## ---------------------------------##
## --boundary condition for t = 0---##
## ---------------------------------##

# start with very small loading, assume almost full regeneration
X_s = 1e-8*np.ones((N+2, time_steps)) 

# water loading in fluid [kg/kg]
X_f = c_A_in*np.ones((N+2, time_steps))
T = T_init*np.ones((N+2, time_steps))
p_H2O = np.ones((N+2, time_steps))
p_H2O[:,0] = p_ergun[:,0]*X_f[:,0]/(X_f[:,0]+(1-X_f[:,0])*M_H2O/M_H2)
# Alternative to p_H2O[:,0] = p_ergun[:,0]*(X_f[:,0]/M_H2O)/(X_f[:,0]/M_H2O+(1-X_f[:,0])/M_H2).
# Equations are almost the same for X_f approaching 0

start_time = time.time()
modul = 1/dt

# Solve balance equations
for i in range(0, time_steps-1):    
    if i/modul % 60 == 0:
         print("Minute ", (i/(modul*60)))

    X_gl, delta_X_gl = langmuir_freundlich(T[:,i], p_H2O[:,i])
    
    ## ---------------------------------##
    ## ---mass-balance, solid phase-----##
    ## ---------------------------------##
    
    # free gas diffusion, molar mass in g/mol und pressure p in bar
    D_12 = (1e-4*0.00143*T[:,i]**(1.75)*(1/(M_H2*1000)+1/(M_H2O*1000))**0.5)/(p_ergun[:,i]/1e5*np.sqrt(2)*(diff_vol_H2O**(1/3) + diff_vol_H2**(1/3))**2) # [m^2/s]
    # Knudsen-diffusion
    D_Kn = (4/3)*d_p_macro*np.sqrt(R*T[:,i]/(2*np.pi*M_H2O))
    D_ges = 1/(1/D_Kn + 1/D_12)
    alpha = rho_s*R*T[:,i]/(eps_p*M_H2O)*delta_X_gl
    delta_eff = (D_ges/tort)/(1+alpha)
    k_LDF = 15*delta_eff/((0.5*d_p)**2)
    
    X_s[:,i+1] = (X_s[:,i]+dt*k_LDF*X_gl)/(1+dt*k_LDF)
    
    ## ---------------------------------##
    ## ----mass-balance, gas phase------##
    ## ---------------------------------##
    
    # calculate axial disperision
    d_bed = D_12*(1-np.sqrt(1-eps))
    Pe_d = u_0_matrix[:,i]*d_p/D_12
    D_ax = d_bed + (Pe_d/2)*D_12

    X_f[1:-1,i+1] = X_f[1:-1,i] + dt/eps*(-(u_0_matrix[1:-1,i]*X_f[1:-1,i]-u_0_matrix[0:-2,i]*X_f[0:-2,i])/dz
                                      +(D_ax[1:-1]-D_ax[0:-2])/dz*(X_f[1:-1,i]-X_f[0:-2,i])/dz + D_ax[1:-1]*(X_f[2:,i]-2*X_f[1:-1,i]+X_f[0:-2,i])/(dz**2)
                                      -rho_s/rho_f_in*(1-eps)*k_LDF[1:-1]*(X_gl[1:-1]-X_s[1:-1,i+1]))
    
    X_f[0,i+1] = X_f_in
    X_f[N+1,i+1] = (2/3)*(-1/2*X_f[N-1,i]+2*X_f[N,i])
    
    ## ---------------------------------##
    ## ---------energy balance----------##
    ## ---------------------------------##
    T_r = T[:,i]/T_c
    h_v = A_V*(1-T_r)**(B_V+C_V*T_r+D_V*T_r**2+E_V*T_r**3)
    # Choose either the interpolated or fixed thermal expansion coefficient beta
    # beta = beta_func(T[:,i])
    h_ads = h_v + E*np.log(X_0LF/X_s[:,i+1])**(1/n)+E*beta*T[:,i]/n*(np.log(X_0LF/X_s[:,i+1]))**(-(n-1/n))
    cp_ads = interp_func(T[:,i])
    dhdX = -E/(n*X_s[:,i+1])*np.log(X_0LF/X_s[:,i+1])**(1/n-1) + E*beta*T[:,i]*(n-1)/(n**2*X_s[:,i+1])*np.log(X_0LF/X_s[:,i+1])**(-2+1/n)
    
    C_1 = rho_f_in*eps*cp_f + rho_s*(1-eps)*(cp_s + X_s[:,i+1]*cp_ads)
    C_2 = rho_s*(1-eps)*k_LDF*(X_gl-X_s[:,i+1])*(h_ads+X_s[:,i+1]*dhdX)
      
    T[1:-1,i+1] = T[1:-1,i] + (dt/C_1[1:-1])*(-rho_f_in*cp_f*(u_0_matrix[1:-1,i]*T[1:-1,i]-u_0_matrix[0:-2,i]*T[0:-2,i])/dz 
                                              - rho_f_in*cp_w*(u_0_matrix[1:-1,i]*T[1:-1,i]*X_f[1:-1,i]-u_0_matrix[0:-2,i]*T[0:-2,i]*X_f[0:-2,i])/dz
                                              + (L_ax[1:-1,i]-L_ax[0:-2,i])/dz*(T[1:-1,i]-T[0:-2,i])/dz + L_ax[1:-1,i]*(T[2:,i]-2*T[1:-1,i]+T[0:-2,i])/(dz**2)
                                              +C_2[1:-1])
    T[0,i+1] = T_in_ads
    T[N+1,i+1] = (2/3)*(-1/2*T[N-1,i]+2*T[N,i])
    
    p_H2O[:,i+1] = p_ergun[:,i+1]*X_f[:,i+1]/(X_f[:,i+1]+(1-X_f[:,i+1])*M_H2O/M_H2)
    

if plot_graph == True:
    # Plot windprofile data
    ticks = np.arange(0, np.floor(t_end_minutes)+1,10)
    labels = ["%d" % (x) for x in ticks]
    xticks = ticks*60*(1/dt_IWES)

    fig, ax = plt.subplots()
    ax.plot(sum_ELY_massflow_mol)
    plt.xlabel('time [min]')
    plt.ylabel('sum of molar mass flows [mol/s]')
    ax.set_xticks(xticks, labels=labels)
    plt.show()
    
    xticks = ticks*60*(1/dt)
    # water loading in adsorbent X_s
    fig, ax = plt.subplots()
    ax.plot(X_s[1,:], label = 'section 1') 
    ax.plot(X_s[2,:], label = 'section 2') 
    ax.plot(X_s[N,:], label = 'section 5')   
    plt.xlabel('time [min]')
    plt.ylabel('$X_s$ [kg/kg]')
    ax.set_xticks(xticks, labels=labels)
    plt.legend(loc = 'lower right')
    plt.show()
    
    # water loading in fluid X_f
    fig, ax = plt.subplots()
    ax.plot(X_f[1,:], label = 'section 1') 
    ax.plot(X_f[2,:], label = 'section 2') 
    ax.plot(X_f[N,:], label = 'section 5')   
    plt.xlabel('time [min]')
    plt.ylabel('$X_f$ [kg/kg]')
    ax.set_xticks(xticks, labels=labels)
    plt.legend()
    plt.show()
    
    # temperature T
    fig, ax = plt.subplots()
    ax.plot(T[1,:], label = 'section 1') 
    ax.plot(T[2,:], label = 'section 2') 
    ax.plot(T[N,:], label = 'section 5') 
    plt.xlabel('time [min]')
    plt.ylabel('$T$ [K]')
    ax.set_xticks(xticks, labels=labels)
    plt.legend()
    plt.show()
    end = time.time()