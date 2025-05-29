import numpy as np
import matplotlib.pyplot as plt
import os
from v0_multi_constants import (RESULTS_DIR)

# Output directory (optional, for saving)
output_dir = os.path.join(os.path.dirname(__file__), RESULTS_DIR)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# To load later:
data = np.load(os.path.join(output_dir, 'trajectory_data.npz'))
t_arr = data['t_arr']
y_arr = data['y_arr']
e_num = int(data['e_num'])

# Compute position modulus for each electron at each time step
position_modulus = []
for i in range(e_num):
    # Extract x, y, z for electron i at all times
    x = y_arr[3*i, :]
    y = y_arr[3*i+1, :]
    z = y_arr[3*i+2, :]
    modulus = np.sqrt(x**2 + y**2 + z**2)
    position_modulus.append(modulus)

# Style settings (copied from pot_alpha.py)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['font.size'] = 26
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 1
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['text.usetex'] = False

# Plot evolution of position modulus
plt.figure(figsize=(10, 6))
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1, 1))]
for i, modulus in enumerate(position_modulus):
    plt.plot(
        t_arr, modulus, label=f'Electron {i+1}',
        linestyle=linestyles[i % len(linestyles)]
    )
plt.xlabel(r'$t$ (a.u.)')
plt.ylabel(r'$|\vec{r}_i|$ (a.u.)')
# plt.ylim(-1, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'position_modulus_vs_time.svg'))
plt.show()
