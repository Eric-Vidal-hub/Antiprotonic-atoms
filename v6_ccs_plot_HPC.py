import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from v6_ccs_FMD_constants_HPC import (RESULTS_DIR, N_TRAJ, THRESH_1, THRESH_2,
                                      B1, B2, B3, BMAX_0, AUTO_BMAX, FILENAME)


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

PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_bmax(E0):
    if AUTO_BMAX:
        if E0 > THRESH_1:
            return B1
        elif E0 > THRESH_2:
            return B2
        else:
            return B3
    else:
        return BMAX_0

# --- Plot 1: Capture cross sections vs. Energy ---
cross_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'cross_sections_E0_*.csv')))
if not cross_files:
    print(f"No cross sections files found in {RESULTS_DIR}")
else:
    cross_list = []
    for filepath in cross_files:
        df = pd.read_csv(filepath)
        cross_list.append(df)
    cross_all = pd.concat(cross_list, ignore_index=True)
    cross_all = cross_all.drop_duplicates(subset=['Energy']).sort_values('Energy')

    cross_all['BMAX'] = cross_all['Energy'].apply(get_bmax)
    cross_all['NR'] = (cross_all['Sigma_total'] / (np.pi * cross_all['BMAX']**2)) * N_TRAJ
    cross_all['Ntot'] = N_TRAJ

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_err = cross_all['Sigma_total'] * np.sqrt(
            (cross_all['Ntot'] - cross_all['NR']) / (cross_all['Ntot'] * cross_all['NR'])
        )
        sigma_err = np.nan_to_num(sigma_err, nan=0.0, posinf=0.0, neginf=0.0)

    MARKERSIZE = 8

    plt.figure()
    plt.errorbar(
        cross_all['Energy'], cross_all['Sigma_total'], yerr=sigma_err,
        fmt='o-', label='Total', capsize=5, color='tab:blue', ecolor='black', markersize=MARKERSIZE
    )
    if 'He' in FILENAME:
        plt.plot(cross_all['Energy'], cross_all['Sigma_single'], 's-', label='Single', color='tab:orange', markersize=MARKERSIZE)
        plt.plot(cross_all['Energy'], cross_all['Sigma_double'], '^-', label='Double', color='tab:green', markersize=MARKERSIZE)
        # --- Cohen Plot (PRA 62) data ---
        cohen_csv = os.path.join(os.path.dirname(__file__), "cohen_pbar_he_capture_data_with_errors.csv")
        if os.path.exists(cohen_csv):
            cohen_df = pd.read_csv(cohen_csv)
            plt.errorbar(
                cohen_df["Energy_au"],
                cohen_df["Sigma_Total_Capture_a02"],
                yerr=cohen_df["Error_Sigma_Cap_a02"],
                fmt='o', capsize=5, color='black', ecolor='gray',
                label='Cohen PRA 62'
            )
        plt.xlim(0.05, 1.5)
        plt.ylim(0, 25)
    else:
        plt.plot(cross_all['Energy'], cross_all['Sigma_partial'], 's-', label='Partial', color='tab:orange', markersize=MARKERSIZE)
        plt.plot(cross_all['Energy'], cross_all['Sigma_full'], '^-', label='Full', color='tab:green', markersize=MARKERSIZE)
    plt.xlabel(r'$E_{0}$ (a.u.)')
    plt.ylabel(r'$\sigma_{cap}$ (a₀²)')
    plt.legend()
    plt.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{os.path.basename(RESULTS_DIR)}_cross_sections_vs_energy.svg'))

    # --- Add Cohen (PRA 62) data ---
    cohen_csv = os.path.join(os.path.dirname(__file__), "cohen_pbar_he_capture_data_with_errors.csv")
    if os.path.exists(cohen_csv):
        cohen_df = pd.read_csv(cohen_csv)
        plt.errorbar(
            cohen_df["Energy_au"],
            cohen_df["Sigma_Total_Capture_a02"],
            yerr=cohen_df["Error_Sigma_Cap_a02"],
            fmt='o', capsize=5, color='black', ecolor='gray',
            label='Cohen PRA 62'
        )
    else:
        print(f"Cohen data file not found: {cohen_csv}")

# --- Plot 2: Initial (L_initial, E_initial) Distribution ---
init_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'initial_states_E0_*.csv')))
if not init_files:
    print(f"No initial_states files found in {RESULTS_DIR}")
else:
    init_list = []
    for filepath in init_files:
        df = pd.read_csv(filepath)
        init_list.append(df)
    init_all = pd.concat(init_list, ignore_index=True)
    plt.figure()
    for cap_type in init_all['type'].unique():
        subset = init_all[init_all['type'] == cap_type]
        plt.scatter(subset['L_initial'], subset['E_initial'], label=cap_type, alpha=0.6)  # Swapped axes
    plt.xlabel(r'$L_{0}$ (a.u.)')  # Updated label
    plt.ylabel(r'$E_{0}$ (a.u.)')  # Updated label
    plt.legend()
    plt.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{os.path.basename(RESULTS_DIR)}_initial_states_scatter.svg'))

# --- Plot 3: Final (L_final, E_final) Distribution ---
final_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'final_states_E0_*.csv')))
if not final_files:
    print(f"No final_states files found in {RESULTS_DIR}")
else:
    final_list = []
    for filepath in final_files:
        df = pd.read_csv(filepath)
        final_list.append(df)
    final_all = pd.concat(final_list, ignore_index=True)
    plt.figure()
    for cap_type in final_all['type'].unique():
        subset = final_all[final_all['type'] == cap_type]
        plt.scatter(subset['L_final'], subset['E_final'], label=cap_type, alpha=0.6)  # Swapped axes
    plt.xlabel(r'$L_{f}$ (a.u.)')  # Updated label
    plt.ylabel(r'$E_{f}$ (a.u.)')  # Updated label
    plt.legend()
    plt.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{os.path.basename(RESULTS_DIR)}_final_states_scatter.svg'))
