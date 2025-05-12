import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

results_dir = 'HPC_dynamics_R0_2.0_Ntraj_100'

# --- Plot 1: Capture cross sections vs. Energy ---
cross_files = sorted(glob.glob(os.path.join(results_dir, 'cross_sections_E0_*.csv')))
if not cross_files:
    print(f"No cross sections files found in {results_dir}")
else:
    cross_list = []
    for filepath in cross_files:
        df = pd.read_csv(filepath)
        cross_list.append(df)
    cross_all = pd.concat(cross_list, ignore_index=True)
    cross_all = cross_all.drop_duplicates(subset=['Energy']).sort_values('Energy')
    plt.figure()
    plt.plot(cross_all['Energy'], cross_all['Sigma_total'], 'o-', label='Total')
    plt.plot(cross_all['Energy'], cross_all['Sigma_single'], 's-', label='Single')
    plt.plot(cross_all['Energy'], cross_all['Sigma_double'], '^-', label='Double')
    plt.xlabel('Initial Energy (a.u.)')
    plt.ylabel('Capture Cross Section (a₀²)')
    plt.title('Capture Cross Sections vs. Energy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Plot 2: Initial (E_initial, L_initial) Distribution ---
init_files = sorted(glob.glob(os.path.join(results_dir, 'initial_states_E0_*.csv')))
if not init_files:
    print(f"No initial_states files found in {results_dir}")
else:
    init_list = []
    for filepath in init_files:
        df = pd.read_csv(filepath)
        init_list.append(df)
    init_all = pd.concat(init_list, ignore_index=True)
    plt.figure()
    for cap_type in init_all['type'].unique():
        subset = init_all[init_all['type'] == cap_type]
        plt.scatter(subset['E_initial'], subset['L_initial'], label=cap_type, alpha=0.6)
    plt.xlabel('Initial Energy (a.u.)')
    plt.ylabel('Initial Angular Momentum L (a.u.)')
    plt.title('Initial (E, L) Distribution by Capture Type')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Plot 3: Final (E_final, L_final) Distribution ---
final_files = sorted(glob.glob(os.path.join(results_dir, 'final_states_E0_*.csv')))
if not final_files:
    print(f"No final_states files found in {results_dir}")
else:
    final_list = []
    for filepath in final_files:
        df = pd.read_csv(filepath)
        final_list.append(df)
    final_all = pd.concat(final_list, ignore_index=True)
    plt.figure()
    for cap_type in final_all['type'].unique():
        subset = final_all[final_all['type'] == cap_type]
        plt.scatter(subset['E_final'], subset['L_final'], label=cap_type, alpha=0.6)
    plt.xlabel('Final Energy (a.u.)')
    plt.ylabel('Final Angular Momentum L (a.u.)')
    plt.title('Final (E, L) Distribution by Capture Type')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
