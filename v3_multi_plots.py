import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from matplotlib import animation
from matplotlib.colors import to_rgba
from v3_multi_constants import (RESULTS_DIR)
import matplotlib.patches as patches

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

output_dir = os.path.join(os.path.dirname(__file__), RESULTS_DIR)
csv_file = os.path.join(output_dir, 'trajectory_data.csv')

# --- Parse CSV ---
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    lines = list(reader)

# Find t_arr and y_arr blocks
t_arr = None
y_arr = []
e_num = None
for idx, row in enumerate(lines):
    if row and row[0].strip() == 'e_num':
        e_num = int(row[1])
    if row and row[0].strip() == 't_arr':
        t_arr = np.array([float(x) for x in lines[idx+1] if x.strip() != ''])
    if row and row[0].startswith('y['):
        # All y_arr rows are consecutive after this
        y_start = idx
        break

# Read y_arr
y_arr = []
for row in lines[y_start:]:
    if not row or not row[0].startswith('y['):
        break
    y_arr.append([float(x) for x in row[1:]])
y_arr = np.array(y_arr)  # shape: (6*e_num, n_times)

n_times = y_arr.shape[1]

# --- Plot modulus of position vs time (_r) ---
plt.figure(figsize=(12, 8))
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1, 1))]
for i in range(e_num):
    r_vec = y_arr[3*i:3*(i+1), :]
    r_mod = np.linalg.norm(r_vec, axis=0)
    plt.plot(
        t_arr, r_mod, label=f'Electron {i+1}',
        linestyle=linestyles[i % len(linestyles)]
    )
plt.xlabel(r'$t$ (a.u.)')
plt.ylabel(r'$|\vec{r}_i|$ (a.u.)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'position_modulus_vs_time_r.svg'))
plt.show()

# --- Plot modulus of momentum vs time (_e) ---
plt.figure(figsize=(12, 8))
for i in range(e_num):
    p_vec = y_arr[3*e_num + 3*i:3*e_num + 3*(i+1), :]
    p_mod = np.linalg.norm(p_vec, axis=0)
    plt.plot(
        t_arr, p_mod, label=f'Electron {i+1}',
        linestyle=linestyles[i % len(linestyles)]
    )
plt.xlabel(r'$t$ (a.u.)')
plt.ylabel(r'$|\vec{p}_i|$ (a.u.)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'momentum_modulus_vs_time_e.svg'))
plt.show()

# --- 3D Trajectory Animation (_gif) ---
frame_step = 1
frames = range(0, len(t_arr), frame_step)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
lines = [ax.plot([], [], [], label=f'Electron {i+1}')[0] for i in range(e_num)]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
current_markers = [
    ax.plot([], [], [], marker='o', markersize=16, color=to_rgba(colors[i % len(colors)], 0.9),
            markeredgecolor='black', linestyle='None', zorder=5)[0]
    for i in range(e_num)
]
inf_lim = -5.2
sup_lim = 5.2
ax.set_xlim(inf_lim, sup_lim)
ax.set_ylim(inf_lim, sup_lim)
ax.set_zlim(inf_lim, sup_lim)
ax.set_xlabel('x (a.u.)', labelpad=18)
ax.set_ylabel('y (a.u.)', labelpad=24)
ax.set_zlabel('z (a.u.)', labelpad=30)
ax.tick_params(axis='z', pad=12)
ax.zaxis.label.set_verticalalignment('bottom')
ax.zaxis.set_label_coords(1.05, 0.5)
ax.xaxis.set_label_coords(0.5, -0.12)
ax.yaxis.set_label_coords(-0.15, 0.5)
ax.legend()

# Add a time bar above the plot
bar_ax = fig.add_axes([0.15, 0.92, 0.7, 0.03])
bar_ax.set_xlim(t_arr[0], t_arr[-1])
bar_ax.set_ylim(0, 1)
bar_ax.axis('off')
bar_patch = patches.Rectangle((t_arr[0], 0), 0, 1, color='royalblue')
bar_ax.add_patch(bar_patch)
time_text = bar_ax.text(
    0.5 * (t_arr[0] + t_arr[-1]), -1, '', va='center', ha='center', fontsize=18, color='black'
)

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    for marker in current_markers:
        marker.set_data([], [])
        marker.set_3d_properties([])
    bar_patch.set_width(0)
    time_text.set_text('')
    return lines + current_markers + [bar_patch, time_text]

def animate(frame_idx):
    frame = frames[frame_idx]
    for i, line in enumerate(lines):
        x = y_arr[3*i, :frame]
        y = y_arr[3*i+1, :frame]
        z = y_arr[3*i+2, :frame]
        line.set_data(x, y)
        line.set_3d_properties(z)
        if frame > 0:
            current_markers[i].set_data([x[-1]], [y[-1]])
            current_markers[i].set_3d_properties([z[-1]])
        else:
            current_markers[i].set_data([], [])
            current_markers[i].set_3d_properties([])
    bar_patch.set_width(t_arr[frame] - t_arr[0])
    time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
    return lines + current_markers + [bar_patch, time_text]

ani = animation.FuncAnimation(
    fig, animate, frames=len(frames), init_func=init,
    interval=40, blit=True
)
gif_path = os.path.join(output_dir, 'trajectory_evolution.gif')
ani.save(gif_path, writer='pillow', fps=25)
plt.show()

print("All plots saved in", output_dir)

# --- Energy plot from y_arr and parameters in CSV ---

# First, parse parameters from the CSV file (if not already loaded)
params = {}
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if row and row[0] in ['e_num', 'MU', 'ZZ', 'XI_H', 'ALPHA_H', 'XI_P', 'ALPHA_P']:
            params[row[0]] = float(row[1])
        if row and row[0].startswith('t_arr'):
            break

e_num = int(params['e_num'])
MU = params['MU']
ZZ = params['ZZ']
XI_H = params['XI_H']
ALPHA_H = params['ALPHA_H']
XI_P = params['XI_P']
ALPHA_P = params['ALPHA_P']

# Now parse t_arr and y_arr from the CSV
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    lines = list(reader)

# Find t_arr and y_arr blocks
for idx, row in enumerate(lines):
    if row and row[0].strip() == 't_arr':
        t_arr = np.array([float(x) for x in lines[idx+1] if x.strip() != ''])
    if row and row[0].startswith('y['):
        y_start = idx
        break

y_arr = []
for row in lines[y_start:]:
    if not row or not row[0].startswith('y['):
        break
    y_arr.append([float(x) for x in row[1:]])
y_arr = np.array(y_arr)  # shape: (6*e_num, n_times)
n_times = y_arr.shape[1]

# Energy calculation function
epsilon = 1e-18
def calculate_total_energy(state, MU, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, e_num):
    r_e_flat = state[:3 * e_num]
    p_e_flat = state[3 * e_num:]
    r_electrons = r_e_flat.reshape((e_num, 3))
    p_electrons = p_e_flat.reshape((e_num, 3))
    kinetic_energy = 0
    potential_energy_en = 0
    potential_energy_ee = 0
    potential_energy_heisenberg = 0
    potential_energy_pauli = 0
    e_spin = np.zeros(e_num, dtype=int)
    for ii in range(e_num):
        e_spin[ii] = ii % 2  # 0 for odd, 1 for even

    for ii in range(e_num):
        ri = r_electrons[ii]
        pi = p_electrons[ii]
        ri_norm = np.linalg.norm(ri)
        pi_norm = np.linalg.norm(pi)

        kinetic_energy += np.sum(pi**2) / (2 * MU)
        potential_energy_en += -ZZ / (ri_norm + epsilon)

        for mm in range(ii + 1, e_num):
            r_im = ri - r_electrons[mm]
            r_im_norm = np.linalg.norm(r_im)
            potential_energy_ee += 1.0 / (r_im_norm + epsilon)
            if e_spin[ii] == e_spin[mm]:
                p_im = (pi - p_electrons[mm]) / 2
                p_im_norm = np.linalg.norm(p_im)
                uu_p = (r_im_norm * p_im_norm / XI_P)**2
                pauli_arg_exp_p = uu_p**2
                if pauli_arg_exp_p > 100 and ALPHA_P * (1 - pauli_arg_exp_p) < -300:
                    exp_pauli = 0.0
                elif ALPHA_P * (1 - pauli_arg_exp_p) > 300:
                    exp_pauli = np.exp(300)
                else:
                    exp_pauli = np.exp(ALPHA_P * (1 - pauli_arg_exp_p))
                potential_energy_pauli += (XI_P**2 / (2 * ALPHA_P * r_im_norm**2)) * exp_pauli

        # Heisenberg potential
        if (ri_norm > epsilon and pi_norm > epsilon and np.abs(XI_H) > epsilon):
            uu = (ri_norm * pi_norm / XI_H)**2
            hei_arg_exp = uu**2
            if hei_arg_exp > 100 and ALPHA_H * (1 - hei_arg_exp) < -300:
                exp_hei_val = 0.0
            elif ALPHA_H * (1 - hei_arg_exp) > 300:
                exp_hei_val = np.exp(300)
            else:
                exp_hei_val = np.exp(ALPHA_H * (1 - hei_arg_exp))
            potential_energy_heisenberg += (
                (XI_H**2 / (4 * ALPHA_H * ri_norm**2 * MU)) *
                exp_hei_val
            )

    total_energy = (kinetic_energy + potential_energy_en +
                    potential_energy_ee + potential_energy_heisenberg +
                    potential_energy_pauli)
    return total_energy

# Compute energies
energies = []
for i_step in range(n_times):
    current_state = y_arr[:, i_step]
    E_tot = calculate_total_energy(
        current_state, MU, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, e_num
    )
    energies.append(E_tot)
energies = np.array(energies)
E0 = energies[0]
relative_energy_error = ((energies - E0) / (np.abs(E0) + epsilon))

# Plot total energy and relative error
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_arr, energies, label='Total Energy')
plt.xlabel('Time (a.u.)')
plt.ylabel('Energy (a.u.)')
plt.title('Total Energy vs. Time')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_arr, relative_energy_error, label='Relative Energy Error')
plt.xlabel('Time (a.u.)')
plt.ylabel('(E(t) - E(0)) / E(0)')
plt.title('Relative Energy Error vs. Time')
plt.yscale('symlog', linthresh=1e-10)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'energy_vs_time.svg'))
plt.show()

# --- Energy components plot ---
ke_list, pe_en_list, pe_ee_list, pe_h_list, pe_p_list = [], [], [], [], []
e_spin = np.zeros(e_num, dtype=int)
for i in range(e_num):
    e_spin[i] = i % 2  # 0 for odd, 1 for even

def calculate_total_energy_components(state, MU, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, e_num, e_spin):
    r_e_flat = state[:3 * e_num]
    p_e_flat = state[3 * e_num:]
    r_electrons = r_e_flat.reshape((e_num, 3))
    p_electrons = p_e_flat.reshape((e_num, 3))
    epsilon = 1e-18
    kinetic_energy = 0
    potential_energy_en = 0
    potential_energy_ee = 0
    potential_energy_heisenberg = 0
    potential_energy_pauli = 0

    for ii in range(e_num):
        ri = r_electrons[ii]
        pi = p_electrons[ii]
        ri_norm = np.linalg.norm(ri)
        pi_norm = np.linalg.norm(pi)
        kinetic_energy += np.sum(pi**2) / (2 * MU)
        potential_energy_en += -ZZ / (ri_norm + epsilon)
        for mm in range(ii + 1, e_num):
            r_im = ri - r_electrons[mm]
            r_im_norm = np.linalg.norm(r_im)
            potential_energy_ee += 1.0 / (r_im_norm + epsilon)
            if e_spin[ii] == e_spin[mm]:
                p_im = (pi - p_electrons[mm]) / 2
                p_im_norm = np.linalg.norm(p_im)
                uu_p = (r_im_norm * p_im_norm / XI_P)**2
                pauli_arg_exp_p = uu_p**2
                if pauli_arg_exp_p > 100 and ALPHA_P * (1 - pauli_arg_exp_p) < -300:
                    exp_pauli = 0.0
                elif ALPHA_P * (1 - pauli_arg_exp_p) > 300:
                    exp_pauli = np.exp(300)
                else:
                    exp_pauli = np.exp(ALPHA_P * (1 - pauli_arg_exp_p))
                potential_energy_pauli += (XI_P**2 / (2 * ALPHA_P * r_im_norm**2)) * exp_pauli
        # Heisenberg potential
        if (ri_norm > epsilon and pi_norm > epsilon and np.abs(XI_H) > epsilon):
            uu = (ri_norm * pi_norm / XI_H)**2
            hei_arg_exp = uu**2
            if hei_arg_exp > 100 and ALPHA_H * (1 - hei_arg_exp) < -300:
                exp_hei_val = 0.0
            elif ALPHA_H * (1 - hei_arg_exp) > 300:
                exp_hei_val = np.exp(300)
            else:
                exp_hei_val = np.exp(ALPHA_H * (1 - hei_arg_exp))
            potential_energy_heisenberg += (
                (XI_H**2 / (4 * ALPHA_H * ri_norm**2 * MU)) *
                exp_hei_val
            )
    return (kinetic_energy, potential_energy_en, potential_energy_ee, potential_energy_heisenberg, potential_energy_pauli)

for i_step in range(n_times):
    current_state = y_arr[:, i_step]
    ke, pe_en, pe_ee, pe_h, pe_p = calculate_total_energy_components(
        current_state, MU, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, e_num, e_spin
    )
    ke_list.append(ke)
    pe_en_list.append(pe_en)
    pe_ee_list.append(pe_ee)
    pe_h_list.append(pe_h)
    pe_p_list.append(pe_p)

plt.figure(figsize=(10,7))
plt.plot(t_arr, ke_list, label='Kinetic E')
plt.plot(t_arr, pe_en_list, label='PE e-N')
plt.plot(t_arr, pe_ee_list, label='PE e-e')
plt.plot(t_arr, pe_h_list, label='PE Heisenberg')
plt.plot(t_arr, pe_p_list, label='PE Pauli')
plt.plot(t_arr, energies, label='Total E', linestyle='--', color='black')
plt.xlabel('Time (a.u.)')
plt.ylabel('Energy Components (a.u.)')
plt.legend()
plt.grid(True)
plt.title('Energy Components vs. Time')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'energy_components_vs_time.svg'))
plt.show()