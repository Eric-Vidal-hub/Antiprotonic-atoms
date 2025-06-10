import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from matplotlib import animation
from matplotlib.colors import to_rgba
from v0_trajectory_constants_HPC import (
    RESULTS_DIR, FILENAME, PLOT_POSITION, PLOT_MOMENTUM, PLOT_ENERGY, PLOT_COMPONENTS,
    PLOT_GIF, N_FRAMES, FPS, PARTICLE_ENERGIES, M_PBAR
)
import matplotlib.patches as patches



def calculate_energies(state, M_STAR, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, e_num):
    """
    Compute total and component energies for a given state vector,
    matching the logic in v0_trajectory_run_HPC.py.
    Returns: E, ke, pe_en, pe_ee, pe_h, pe_p
    """
    # Unpack state vector
    # [re1_x, re1_y, re1_z, ..., pe1_x, pe1_y, pe1_z, ..., rpbar_x, rpbar_y, rpbar_z, ppbar_x, ppbar_y, ppbar_z]
    r_e = [state[3*i:3*(i+1)] for i in range(e_num)]
    p_e = [state[3*e_num + 3*i:3*e_num + 3*(i+1)] for i in range(e_num)]
    r_pbar = state[-6:-3]
    p_pbar = state[-3:]

    # Kinetic energies
    kin_e = np.sum([np.dot(p, p)/2.0 for p in p_e])
    kin_pbar = np.dot(p_pbar, p_pbar)/(2*M_STAR)
    ke = kin_e + kin_pbar

    # Electron-nucleus
    pe_en = np.sum([-ZZ / (np.linalg.norm(r) + 1e-18) for r in r_e])

    # Electron-electron
    pe_ee = 0.0
    pauli_pot = 0.0
    for i in range(e_num):
        for j in range(i+1, e_num):
            delta_r = np.linalg.norm(r_e[i] - r_e[j])
            pe_ee += 1.0 / (delta_r + 1e-18)
            # Pauli term for identical spins (assuming alternating spins)
            if (i % 2) == (j % 2):
                delta_p = np.linalg.norm(p_e[i] - p_e[j])
                uu_p = (delta_r * delta_p / XI_P)**2
                pauli_arg_exp_p = uu_p**2
                if pauli_arg_exp_p > 100 and ALPHA_P * (1 - pauli_arg_exp_p) < -300:
                    exp_pauli = 0.0
                elif ALPHA_P * (1 - pauli_arg_exp_p) > 300:
                    exp_pauli = np.exp(300)
                else:
                    exp_pauli = np.exp(ALPHA_P * (1 - pauli_arg_exp_p))
                pauli_pot += (XI_P**2 / (2 * ALPHA_P * (delta_r + 1e-18)**2)) * exp_pauli

    # Electron-antiproton
    pair_pot_pbar = 0.0
    for i in range(e_num):
        pair_pot_pbar += 1.0 / (np.linalg.norm(r_e[i] - r_pbar) + 1e-18)

    # Heisenberg terms
    pe_h = 0.0
    for i in range(e_num):
        r_mod = np.linalg.norm(r_e[i])
        p_mod = np.linalg.norm(p_e[i])
        pe_h += (XI_H**2 / (4 * ALPHA_H * (r_mod**2 + 1e-18))) * np.exp(ALPHA_H * (1 - (r_mod * p_mod / XI_H)**4))
    r_pbar_mod = np.linalg.norm(r_pbar)
    p_pbar_mod = np.linalg.norm(p_pbar)
    heisenberg_pbar = (XI_H**2 / (4 * ALPHA_H * M_STAR * (r_pbar_mod**2 + 1e-18))) * np.exp(ALPHA_H * (1 - (r_pbar_mod * p_pbar_mod / XI_H)**4))
    pe_h += heisenberg_pbar

    # Antiproton-nucleus
    nuc_pbar = -ZZ / (r_pbar_mod + 1e-18)

    # Total energy
    E = ke + pe_en + pe_ee + pe_h + pauli_pot + pair_pot_pbar + nuc_pbar

    return E, ke, pe_en, pe_ee, pe_h, pauli_pot



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
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# --- Find and read the capture file for parameters ---
capture_files = [f for f in os.listdir(output_dir) if f.startswith('capture_') and f.endswith('.csv')]
if not capture_files:
    raise FileNotFoundError("No capture_{ID}.csv file found in the results directory.")
capture_file = os.path.join(output_dir, capture_files[0])
with open(capture_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header_cap = next(reader)
    header_cap = [h.strip() for h in header_cap]
    row_cap = next(reader)
params = {key: float(val) for key, val in zip(header_cap, row_cap)}

# Extract parameters for the run
e_num = int(params['e_num'])
ZZ = float(params['p_num'])
ALPHA_H = params.get('ALPHA_H', 5.0)
XI_H = params.get('XI_H', 1.0 / (1 + 1 / (2 * ALPHA_H))**0.5)
ALPHA_P = params.get('ALPHA_P', 5.0)
XI_P = params.get('XI_P', 2.767 / (1 + 1 / (2 * ALPHA_P))**0.5)
M_STAR = 1/((1/M_PBAR)+(1/(ZZ*M_PBAR)))

# --- Find and read the trajectory file for time-dependent data ---
traj_files = [f for f in os.listdir(output_dir) if f.startswith('trajectory_') and f.endswith('.csv')]
if not traj_files:
    raise FileNotFoundError("No trajectory_{ID}.csv file found in the results directory.")
csv_file = os.path.join(output_dir, traj_files[0])
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    data = np.array([[float(x) for x in row] for row in reader])

# Identify columns and build arrays as before
time_idx = header.index('time')
n_times = data.shape[0]
t_arr = data[:, time_idx]

# Electron positions and momenta
r_e_arr = np.zeros((e_num, 3, n_times))
p_e_arr = np.zeros((e_num, 3, n_times))
for i in range(e_num):
    r_e_arr[i, 0, :] = data[:, header.index(f're{i+1}_x')]
    r_e_arr[i, 1, :] = data[:, header.index(f're{i+1}_y')]
    r_e_arr[i, 2, :] = data[:, header.index(f're{i+1}_z')]
    p_e_arr[i, 0, :] = data[:, header.index(f'pe{i+1}_x')]
    p_e_arr[i, 1, :] = data[:, header.index(f'pe{i+1}_y')]
    p_e_arr[i, 2, :] = data[:, header.index(f'pe{i+1}_z')]

# Antiproton position and momentum
r_pbar_arr = np.stack([data[:, header.index('rpbar_x')],
                       data[:, header.index('rpbar_y')],
                       data[:, header.index('rpbar_z')]])
p_pbar_arr = np.stack([data[:, header.index('ppbar_x')],
                       data[:, header.index('ppbar_y')],
                       data[:, header.index('ppbar_z')]])

# --- Build y_arr: shape (n_vars, n_times)
# The order should match your ODE integration: [r_e1, r_e2, ..., p_e1, p_e2, ..., r_pbar, p_pbar]
# For 2 electrons: [re1_x, re1_y, re1_z, re2_x, re2_y, re2_z, pe1_x, pe1_y, pe1_z, pe2_x, pe2_y, pe2_z, rpbar_x, rpbar_y, rpbar_z, ppbar_x, ppbar_y, ppbar_z]
n_vars = data.shape[1] - 1  # subtract time column
y_arr = data[:, 1:].T  # shape (n_vars, n_times)

# --- Plot modulus of position vs time (_r) ---
if PLOT_POSITION:
    plt.figure(figsize=(12, 8))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1, 1))]
    for i in range(e_num):
        r_mod = np.linalg.norm(r_e_arr[i, :, :], axis=0)
        plt.plot(
            t_arr, r_mod, label=f'Electron {i+1}',
            linestyle=linestyles[i % len(linestyles)]
        )
    r_pbar_mod = np.linalg.norm(r_pbar_arr, axis=0)
    plt.plot(t_arr, r_pbar_mod, label='Antiproton', color='black', linestyle='-')
    plt.xlabel(r'$t$ (a.u.)')
    plt.ylabel(r'$|\vec{r}_i|$ (a.u.)')
    plt.legend()
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.ylim(np.min(r_pbar_mod)*0.9, np.max(r_pbar_mod)*1.1)
    plt.savefig(os.path.join(plots_dir, f'{FILENAME}_position_modulus_vs_time_r.svg'))

# --- Plot modulus of momentum vs time (_e) ---
if PLOT_MOMENTUM:
    plt.figure(figsize=(12, 8))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (1, 1))]
    for i in range(e_num):
        p_mod = np.linalg.norm(p_e_arr[i, :, :], axis=0)
        plt.plot(
            t_arr, p_mod, label=f'Electron {i+1}',
            linestyle=linestyles[i % len(linestyles)]
        )
    p_pbar_mod = np.linalg.norm(p_pbar_arr, axis=0)
    plt.plot(t_arr, p_pbar_mod, label='Antiproton', color='black', linestyle='-')
    plt.xlabel(r'$t$ (a.u.)')
    plt.ylabel(r'$|\vec{p}_i|$ (a.u.)')
    plt.legend()
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.ylim(np.min(p_mod)*0.9, np.max(p_mod)*2)
    plt.savefig(os.path.join(plots_dir, f'{FILENAME}_momentum_modulus_vs_time_e.svg'))

# --- Compute all energies/components once ---
energies = []
ke_list, pe_en_list, pe_ee_list, pe_h_list, pe_p_list = [], [], [], [], []
for i_step in range(n_times):
    current_state = y_arr[:, i_step]
    E, ke, pe_en, pe_ee, pe_h, pe_p = calculate_energies(
        current_state, M_STAR, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, e_num
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
    plt.ylim(-2,2)

    plt.subplot(1, 2, 2)
    plt.plot(t_arr, relative_energy_error, label='|Relative energy error|')
    plt.xlabel('$t$ (a.u.)')
    plt.ylabel('|E(t) - E(0)| / |E(0)|')
    plt.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.yscale('symlog', linthresh=1e-10)
    plt.ylim(0, 1)   # top=np.max(relative_energy_error)*1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{FILENAME}_energy_vs_time.svg'))

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
    plt.ylim(-2,2)
    # plt.ylim(np.min(energies)*0.95, np.max(energies)*1.05)
    plt.savefig(os.path.join(plots_dir, f'{FILENAME}_energy_components_vs_time.svg'))

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

if PARTICLE_ENERGIES:
    # --- Compute antiproton energy evolution ---
    r_pbar_vec = y_arr[-6:-3, :]
    p_pbar_vec = y_arr[-3:, :]
    r_pbar_mod = np.linalg.norm(r_pbar_vec, axis=0)
    p_pbar_mod = np.linalg.norm(p_pbar_vec, axis=0)
    kin_pbar = p_pbar_mod**2 / (2 * M_STAR)  # Use correct mass if different
    nuc_pbar = -ZZ / (r_pbar_mod + 1e-18)
    # Electron-antiproton interaction
    pair_pot_pbar = np.zeros_like(r_pbar_mod)
    for i in range(e_num):
        r_e_vec = y_arr[3*i:3*(i+1), :]
        pair_pot_pbar += 1.0 / (np.linalg.norm(r_e_vec - r_pbar_vec, axis=0) + 1e-18)
    # Heisenberg term (optional, if you want to include)
    heisenberg_pbar = (XI_H**2 / (4 * ALPHA_H * (r_pbar_mod**2 + 1e-18) * M_STAR)) * \
        np.exp(ALPHA_H * (1 - (r_pbar_mod * p_pbar_mod / XI_H)**4))

    E_pbar = kin_pbar + nuc_pbar + pair_pot_pbar + heisenberg_pbar
    print(f"Antiproton INITIAL energy: {E_pbar[0]:.8f} a.u. at t = {t_arr[0]:.2f} a.u.")
    print(f"Antiproton FINAL energy: {E_pbar[-1]:.8f} a.u. at t = {t_arr[-1]:.2f} a.u.")
    # compute the average energy of the antiproton during the last 500 time steps before the end of the simulation
    if n_times > 500:
        avg_pbar_energy = np.mean(E_pbar[-500:])
        std_pbar_energy = np.std(E_pbar[-500:])
    else:
        avg_pbar_energy = np.mean(E_pbar)
        std_pbar_energy = np.std(E_pbar)
    print(f"Antiproton AVERAGE energy (last 500 steps): {avg_pbar_energy:.8f} ± {std_pbar_energy:.8e} a.u.")

    # --- Compute electron energies (with antiproton interaction) ---
    electron_energies = np.zeros((e_num, n_times))
    for i in range(e_num):
        r_e_vec = y_arr[3*i:3*(i+1), :]
        p_e_vec = y_arr[3*e_num + 3*i:3*e_num + 3*(i+1), :]
        r_e_mod = np.linalg.norm(r_e_vec, axis=0)
        p_e_mod = np.linalg.norm(p_e_vec, axis=0)
        kin_e = p_e_mod**2 / 2.0
        nuc_e = -ZZ / (r_e_mod + 1e-18)
        # Electron-antiproton interaction
        r_epbar = np.linalg.norm(r_e_vec - r_pbar_vec, axis=0)
        pot_epbar = 1.0 / (r_epbar + 1e-18)
        # Heisenberg term (optional)
        heisenberg_e = (XI_H**2 / (4 * ALPHA_H * (r_e_mod**2 + 1e-18))) * \
            np.exp(ALPHA_H * (1 - (r_e_mod * p_e_mod / XI_H)**4))
        # Electron-electron and Pauli terms can be added as needed
        electron_energies[i, :] = kin_e + nuc_e + pot_epbar + heisenberg_e

    # %% --- Plot all individual particle energies together ---
    plt.figure(figsize=(12, 8))
    for i in range(e_num):
        plt.plot(t_arr, electron_energies[i], label=f'Electron {i+1}')
    plt.plot(t_arr, E_pbar, label='Antiproton', color='black', linestyle='-')
    plt.xlabel(r'$t$ (a.u.)')
    plt.ylabel(r'Particle energy (a.u.)')
    plt.legend()
    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.ylim(-5,2)
    plt.savefig(os.path.join(plots_dir, f'{FILENAME}_all_particle_energies_vs_time.svg'))

# %% --- 3D Trajectory Animation (_gif) with time bar and antiproton ---
if PLOT_GIF:
    frame_step = max(1, len(t_arr) // N_FRAMES)
    frames = range(0, len(t_arr), frame_step)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    lines = [ax.plot([], [], [], label=f'Electron {i+1}')[0]
             for i in range(e_num)]
    # Add antiproton line
    pbar_line = ax.plot([], [], [], label='Antiproton', color='black', linewidth=2)[0]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    current_markers = [
        ax.plot([], [], [], marker='o', markersize=16,
                color=to_rgba(colors[i % len(colors)], 0.9),
                markeredgecolor='black', linestyle='None', zorder=5)[0]
        for i in range(e_num)
    ]
    # Antiproton marker
    pbar_marker = ax.plot([], [], [], marker='o', markersize=18,
                          color='black', markeredgecolor='yellow', linestyle='None', zorder=6)[0]
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
        pbar_line.set_data([], [])
        pbar_line.set_3d_properties([])
        pbar_marker.set_data([], [])
        pbar_marker.set_3d_properties([])
        bar_patch.set_width(0)
        time_text.set_text('')
        return lines + [pbar_line] + current_markers + [pbar_marker, bar_patch, time_text]

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
        # Antiproton line and marker
        x_pbar = y_arr[-6, :frame]
        y_pbar = y_arr[-5, :frame]
        z_pbar = y_arr[-4, :frame]
        pbar_line.set_data(x_pbar, y_pbar)
        pbar_line.set_3d_properties(z_pbar)
        if frame > 0:
            pbar_marker.set_data([x_pbar[-1]], [y_pbar[-1]])
            pbar_marker.set_3d_properties([z_pbar[-1]])
        else:
            pbar_marker.set_data([], [])
            pbar_marker.set_3d_properties([])
        bar_patch.set_width(t_arr[frame] - t_arr[0])
        time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
        return lines + [pbar_line] + current_markers + [pbar_marker, bar_patch, time_text]

    ani = animation.FuncAnimation(
        fig, animate, frames=len(frames), init_func=init,
        interval=N_FRAMES / FPS, blit=True
    )
    gif_path = os.path.join(plots_dir, f'{FILENAME}_trajectory_evolution.gif')
    ani.save(gif_path, writer='pillow', fps=FPS)
