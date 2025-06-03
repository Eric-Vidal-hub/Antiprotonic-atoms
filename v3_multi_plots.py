import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from matplotlib import animation
from matplotlib.colors import to_rgba
from v3_multi_constants import (
    RESULTS_DIR, PLOT_POSITION, PLOT_MOMENTUM, PLOT_ENERGY, PLOT_COMPONENTS,
    PLOT_GIF, N_FRAMES, FPS
)
import matplotlib.patches as patches


# --- Energy calculation and plots ---
def calculate_energies(state, MU, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, e_num):
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
    e_spin = np.zeros(e_num, dtype=int)
    for ii in range(e_num):
        e_spin[ii] = ii % 2
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
                if (pauli_arg_exp_p > 100 and
                    ALPHA_P * (1 - pauli_arg_exp_p) < -300):
                    exp_pauli = 0.0
                elif ALPHA_P * (1 - pauli_arg_exp_p) > 300:
                    exp_pauli = np.exp(300)
                else:
                    exp_pauli = np.exp(ALPHA_P * (1 - pauli_arg_exp_p))
                potential_energy_pauli += (
                    (XI_P**2 / (2 * ALPHA_P * r_im_norm**2)) * exp_pauli
                )
        if (ri_norm > epsilon and pi_norm > epsilon and
            np.abs(XI_H) > epsilon):
            uu = (ri_norm * pi_norm / XI_H)**2
            hei_arg_exp = uu**2
            if (hei_arg_exp > 100 and
                ALPHA_H * (1 - hei_arg_exp) < -300):
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
    return (total_energy, kinetic_energy, potential_energy_en,
            potential_energy_ee, potential_energy_heisenberg,
            potential_energy_pauli)


plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['font.size'] = 26
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['legend.fontsize'] = 16
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

# %% --- Parse CSV ---
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    lines = list(reader)

t_arr = None
y_arr = []
e_num = None
for idx, row in enumerate(lines):
    if row and row[0].strip() == 'e_num':
        e_num = int(row[1])
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
y_arr = np.array(y_arr)
n_times = y_arr.shape[1]

# --- Plot modulus of position vs time (_r) ---
if PLOT_POSITION:
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
    plt.ylabel(r'$r_i$ (a.u.)')
    plt.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_modulus_vs_time_r.svg'))

# --- Plot modulus of momentum vs time (_e) ---
if PLOT_MOMENTUM:
    plt.figure(figsize=(12, 8))
    for i in range(e_num):
        p_vec = y_arr[3*e_num + 3*i:3*e_num + 3*(i+1), :]
        p_mod = np.linalg.norm(p_vec, axis=0)
        plt.plot(
            t_arr, p_mod, label=f'Electron {i+1}',
            linestyle=linestyles[i % len(linestyles)]
        )
    plt.xlabel(r'$t$ (a.u.)')
    plt.ylabel(r'$p_i$ (a.u.)')
    plt.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'momentum_modulus_vs_time_e.svg'))


# %% --- Energy plot from y_arr and parameters in CSV ---
params = {}
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if row and row[0] in [
            'e_num', 'MU', 'ZZ', 'XI_H', 'ALPHA_H', 'XI_P', 'ALPHA_P']:
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
# --- Compute all energies/components once ---
energies = []
ke_list, pe_en_list, pe_ee_list, pe_h_list, pe_p_list = [], [], [], [], []
for i_step in range(n_times):
    current_state = y_arr[:, i_step]
    E, ke, pe_en, pe_ee, pe_h, pe_p = calculate_energies(
        current_state, MU, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, e_num
    )
    energies.append(E)
    ke_list.append(ke)
    pe_en_list.append(pe_en)
    pe_ee_list.append(pe_ee)
    pe_h_list.append(pe_h)
    pe_p_list.append(pe_p)
energies = np.array(energies)
E0 = energies[0]
relative_energy_error = np.abs((energies - E0) / (np.abs(E0) + 1e-18))

# --- Energy plot ---
if PLOT_ENERGY:
    plt.close('all')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(t_arr, energies, label='Total energy')
    plt.xlabel(r'$t$ (a.u.)')
    plt.ylabel(r'$E(t)$ (a.u.)')
    plt.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(t_arr, relative_energy_error, label='|Relative energy error|')
    plt.xlabel('$t$ (a.u.)')
    plt.ylabel('|E(t) - E(0)| / |E(0)|')
    plt.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.yscale('symlog', linthresh=1e-10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_vs_time.svg'))


# --- Energy components plot ---
if PLOT_COMPONENTS:
    plt.close('all')
    plt.figure(figsize=(10, 7))
    plt.plot(t_arr, ke_list, label='Kinetic', linestyle='-', color='tab:blue')
    plt.plot(t_arr, pe_en_list, label=r'$e^-$-nucleus', linestyle='--', color='tab:orange')
    plt.plot(t_arr, pe_ee_list, label=r'$e^-$-$e^-$', linestyle='-.', color='tab:green')
    plt.plot(t_arr, pe_h_list, label='Heisenberg', linestyle=':', color='tab:red')
    plt.plot(t_arr, pe_p_list, label='Pauli', linestyle=(0, (3, 1, 1, 1)), color='tab:purple')
    plt.plot(t_arr, energies, label='Total', linestyle='-', color='gray', linewidth=3)
    plt.xlabel(r'$t$ (a.u.)')
    plt.ylabel(r'$E_i(t)$ (a.u.)')
    plt.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_components_vs_time.svg'))


# %% --- 3D Trajectory Animation (_gif) with time bar ---
if PLOT_GIF:
    # Use a smaller frame_step for smoother animation
    frame_step = max(1, len(t_arr) // N_FRAMES)  # adjust as needed
    frames = range(0, len(t_arr), frame_step)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    lines = [ax.plot([], [], [], label=f'Electron {i+1}')[0]
             for i in range(e_num)]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    current_markers = [
        ax.plot([], [], [], marker='o', markersize=16,
                color=to_rgba(colors[i % len(colors)], 0.9),
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
        0.5 * (t_arr[0] + t_arr[-1]), -1, '', va='center', ha='center',
        fontsize=18, color='black'
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
        interval=N_FRAMES / FPS, blit=True  # 60 FPS for smoothness
    )
    gif_path = os.path.join(output_dir, 'trajectory_evolution.gif')
    ani.save(gif_path, writer='pillow', fps=FPS)  # Save at 60 FPS for smooth playback

# %% --- Compute per-electron averages and deviations ---
r_means, r_stds, p_means, p_stds = [], [], [], []
for i in range(e_num):
    r_vec = y_arr[3*i:3*(i+1), :]
    r_mod = np.linalg.norm(r_vec, axis=0)
    r_means.append(np.mean(r_mod))
    r_stds.append(np.std(r_mod))
    p_vec = y_arr[3*e_num + 3*i:3*e_num + 3*(i+1), :]
    p_mod = np.linalg.norm(p_vec, axis=0)
    p_means.append(np.mean(p_mod))
    p_stds.append(np.std(p_mod))

# --- Compute atom energy average and std ---
E_mean = np.mean(energies)
E_std = np.std(energies)
# --- Compute component energy average and std ---
ke_mean = np.mean(ke_list)
ke_std = np.std(ke_list)
pe_en_mean = np.mean(pe_en_list)
pe_en_std = np.std(pe_en_list)
pe_ee_mean = np.mean(pe_ee_list)
pe_ee_std = np.std(pe_ee_list)
pe_h_mean = np.mean(pe_h_list)
pe_h_std = np.std(pe_h_list)
pe_p_mean = np.mean(pe_p_list)
pe_p_std = np.std(pe_p_list)

# --- Print results ---
for i in range(e_num):
    print(f"Electron {i+1}: <|r|> = {r_means[i]:.8f}, std = {r_stds[i]:.8e}, "
          f"<|p|> = {p_means[i]:.8f}, std = {p_stds[i]:.8e}")
# Print component energies
print(f"Component Energies (mean ± std):")
print(f"Kinetic: {ke_mean:.8f} ± {ke_std:.8e}")
print(f"Electron-Nucleus: {pe_en_mean:.8f} ± {pe_en_std:.8e}")
print(f"Electron-Electron: {pe_ee_mean:.8f} ± {pe_ee_std:.8e}")
print(f"Heisenberg: {pe_h_mean:.8f} ± {pe_h_std:.8e}")
print(f"Pauli: {pe_p_mean:.8f} ± {pe_p_std:.8e}")
print(f"Total Energy: {E_mean:.8f} ± {E_std:.8e}")

# --- Compute total time ---
total_time = t_arr[-1] - t_arr[0]
print(f"Total time: {total_time:.8f} a.u.")

# --- Write to CSV ---
# Write a header describing the element and time interval
# List of elements by electron number (Z)
elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
element = elements[e_num - 1] if 1 <= e_num <= len(elements) else f"Z{e_num}"
time_interval = total_time
csv_out = os.path.join(
    output_dir, f"{element}_averages_{int(round(time_interval))}au.csv"
)
with open(csv_out, "w", newline="") as f:
    f.write(f"# Element: {element}, Time interval: {time_interval:.2f} a.u.\n")
with open(csv_out, "w", newline="") as f:
    f.write("electron,r_mean,r_std,p_mean,p_std\n")
    for i in range(e_num):
        f.write(f"{i+1},{r_means[i]},{r_stds[i]},{p_means[i]},{p_stds[i]}\n")
    f.write("atom,E_mean,E_std\n")
    f.write(f"total_time,{total_time}\n")
    f.write("component,mean,std\n")
    f.write(f"Kinetic,{ke_mean},{ke_std}\n")
    f.write(f"Electron-Nucleus,{pe_en_mean},{pe_en_std}\n")
    f.write(f"Electron-Electron,{pe_ee_mean},{pe_ee_std}\n")
    f.write(f"Heisenberg,{pe_h_mean},{pe_h_std}\n")
    f.write(f"Pauli,{pe_p_mean},{pe_p_std}\n")
    f.write(f"Total Energy,{E_mean},{E_std}\n")
print(f"Per-electron averages and deviations written to {csv_out}")
