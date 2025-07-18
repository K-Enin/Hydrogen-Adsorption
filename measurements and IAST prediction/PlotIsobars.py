"""
Measurement data for adsorption isobars at pressures 1223 Pa and 23339 Pa
"""

import numpy as np
import matplotlib.pyplot as plt

T = np.array([24.91, 34.86, 44.86, 54.88, 64.89, 74.86, 84.86, 94.86, 104.87, 114.85, 124.82, 134.83, 144.79]) 
W_plot = [[307.3629153, 327.113355500],[294.2579607, 313.211807700], 
     [280.5618051, 300.3736092], [267.506117300, 288.007554300], 
     [253.604683600, 274.615100400], [239.227004700, 261.464876600], 
     [225.132609700, 247.653651900], [203.237646400, 228.283450300], 
     [181.43711100, 211.787165900], [158.462384800, 191.755963400], 
     [137.696451200, 169.737652400], [119.34458800, 149.037237800], 
     [102.749905600, 126.36613700]] 

z_values = [item[0] for item in W_plot]
y_values = [item[1] for item in W_plot]

# Plot
plt.scatter(T, z_values, label='1223 Pa')
plt.scatter(T, y_values, label='2339 Pa', marker='^')
plt.xlabel('temperature (C°)')
plt.ylabel('adsorbed amount of water (mg/g)')
plt.grid(True)
plt.legend()
plt.savefig("high_resolution_water.png", dpi=300) 
plt.show()
