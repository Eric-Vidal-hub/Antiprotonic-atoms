import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

# --- User: set the two results directories to compare ---
XPBAR = 10
N_TRAJ = 200

LABEL_1 = r'$^6$Li'
FILENAME_1 = '03_Li_03e'
RESULTS_DIR_1 = 'CCS_{}_R0_{:.1f}_Ntraj_{:d}_HPC'.format(FILENAME_1, XPBAR, int(N_TRAJ))

LABEL_2 = r'$^6$Li$^+$'
FILENAME_2 = '03_Li_02e'
RESULTS_DIR_2 = 'CCS_{}_R0_{:.1f}_Ntraj_{:d}_HPC'.format(FILENAME_2, XPBAR, int(N_TRAJ))

# --- Find and load cross section files ---def get_bmax(E0, AUTO_BMAX, THRESH_1, THRESH_2, B1, B2, B3, BMAX_0):
def get_bmax(E0, AUTO_BMAX, THRESH_1, THRESH_2, B1, B2, B3, BMAX_0):
    if AUTO_BMAX:
        if E0 > THRESH_1:
            return B1
        elif E0 > THRESH_2:
            return B2
        else:
            return B3
    else:
        return BMAX_0

def load_cross_section(results_dir):
    cross_files = sorted(glob.glob(os.path.join(results_dir, 'cross_sections_E0_*.csv')))
    if not cross_files:
        raise FileNotFoundError(f"No cross_sections_E0_*.csv found in {results_dir}")
    cross_list = [pd.read_csv(f) for f in cross_files]
    cross_all = pd.concat(cross_list, ignore_index=True)
    cross_all = cross_all.drop_duplicates(subset=['Energy']).sort_values('Energy')
    # Try to load constants if available
    try:
        from v6_ccs_FMD_constants_HPC import N_TRAJ, THRESH_1, THRESH_2, B1, B2, B3, BMAX_0, AUTO_BMAX
    except ImportError:
        # Set defaults if not available
        N_TRAJ = 1000
        THRESH_1, THRESH_2, B1, B2, B3, BMAX_0, AUTO_BMAX = 1, 0.5, 2, 1.5, 1, 2, False
    cross_all['BMAX'] = cross_all['Energy'].apply(lambda E: get_bmax(E, AUTO_BMAX, THRESH_1, THRESH_2, B1, B2, B3, BMAX_0))
    cross_all['NR'] = (cross_all['Sigma_total'] / (np.pi * cross_all['BMAX']**2)) * N_TRAJ
    cross_all['Ntot'] = N_TRAJ
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_err = cross_all['Sigma_total'] * np.sqrt(
            (cross_all['Ntot'] - cross_all['NR']) / (cross_all['Ntot'] * cross_all['NR'])
        )
        sigma_err = np.nan_to_num(sigma_err, nan=0.0, posinf=0.0, neginf=0.0)
    cross_all['sigma_err'] = sigma_err
    return cross_all

cross1 = load_cross_section(RESULTS_DIR_1)
cross2 = load_cross_section(RESULTS_DIR_2)


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

# --- Plot ---
plt.figure(figsize=(12, 8))
plt.errorbar(
    cross1['Energy'], cross1['Sigma_total'], yerr=cross1['sigma_err'],
    fmt='o-', label=LABEL_1, capsize=5, markersize=10, ecolor='black'
)
plt.errorbar(
    cross2['Energy'], cross2['Sigma_total'], yerr=cross2['sigma_err'],
    fmt='s-', label=LABEL_2, capsize=5, markersize=10, ecolor='black'
)
plt.xlabel(r'$E_{0}$ (a.u.)')
plt.ylabel(r'$\sigma_{cap}$ (a₀²)')
plt.legend()
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig('compare_cross_sections.svg')
plt.show()
