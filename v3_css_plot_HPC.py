import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from v3_ccs_FMD_constants_HPC import (RESULTS_DIR)


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
    plt.plot(cross_all['Energy'], cross_all['Sigma_single'], 's-', label='Single')
    plt.plot(cross_all['Energy'], cross_all['Sigma_double'], '^-', label='Double')
    plt.xlabel('Initial Energy (a.u.)')
    plt.ylabel('Capture Cross Section (a₀²)')
    plt.title('Capture Cross Sections vs. Energy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# # --- Plot 2: Initial (L_initial, E_initial) Distribution ---
# init_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'initial_states_E0_*.csv')))
# if not init_files:
#     print(f"No initial_states files found in {RESULTS_DIR}")
# else:
#     init_list = []
#     for filepath in init_files:
#         df = pd.read_csv(filepath)
#         init_list.append(df)
#     init_all = pd.concat(init_list, ignore_index=True)
#     plt.figure()
#     for cap_type in init_all['type'].unique():
#         subset = init_all[init_all['type'] == cap_type]
#         plt.scatter(subset['L_initial'], subset['E_initial'], label=cap_type, alpha=0.6)  # Swapped axes
#     plt.xlabel('Initial Angular Momentum L (a.u.)')  # Updated label
#     plt.ylabel('Initial Energy (a.u.)')  # Updated label
#     plt.title('Initial (L, E) Distribution by Capture Type')  # Updated title
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # --- Plot 3: Final (L_final, E_final) Distribution ---
# final_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'final_states_E0_*.csv')))
# if not final_files:
#     print(f"No final_states files found in {RESULTS_DIR}")
# else:
#     final_list = []
#     for filepath in final_files:
#         df = pd.read_csv(filepath)
#         final_list.append(df)
#     final_all = pd.concat(final_list, ignore_index=True)
#     plt.figure()
#     for cap_type in final_all['type'].unique():
#         subset = final_all[final_all['type'] == cap_type]
#         plt.scatter(subset['L_final'], subset['E_final'], label=cap_type, alpha=0.6)  # Swapped axes
#     plt.xlabel('Final Angular Momentum L (a.u.)')  # Updated label
#     plt.ylabel('Final Energy (a.u.)')  # Updated label
#     plt.title('Final (L, E) Distribution by Capture Type')  # Updated title
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
