import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from v5_ccs_FMD_constants_HPC import (RESULTS_DIR)


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
    plt.figure()
    plt.plot(cross_all['Energy'], cross_all['Sigma_total'], 'o-', label='Total')
    plt.plot(cross_all['Energy'], cross_all['Sigma_partial'], 's-', label='Partial')
    plt.plot(cross_all['Energy'], cross_all['Sigma_full'], '^-', label='Full')
    # plt.plot(cross_all['Energy'], cross_all['Sigma_single'], 's-', label='Single')
    # plt.plot(cross_all['Energy'], cross_all['Sigma_double'], '^-', label='Double')
    plt.xlabel(r'$E_{0}$ (a.u.)')
    plt.ylabel(r'$\sigma_{cap}$ (a₀²)')
    plt.legend()
    plt.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.show()

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
    plt.show()

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
    plt.ylabel(r'$L_{f}$ (a.u.)')  # Updated label
    plt.legend()
    plt.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.show()
