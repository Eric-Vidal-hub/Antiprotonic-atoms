# Physical constants
# Proton mass (a.u.)
M_PBAR = 1836.152672  # antiproton mass (a.u.)
ALPHA = 5             # Hardness parameter
XI_H = 1.000          # Tuning parameter for the Heisenberg potential
XI_P = 2.767          # Tuning parameter for the Pauli potential

# Scaling parameters according to alpha
XI_H /= (1 + 1 / (2 * ALPHA))**0.5
XI_P /= (1 + 1 / (2 * ALPHA))**0.5

print(f"XI_H: {XI_H}, XI_P: {XI_P}")

# Simulation parameters
XPBAR = 3.0           # Initial distance of antiproton (a.u.)
N_TRAJ = 100          # Number of trajectories per energy
T_MAX = 25000.0       # Maximum simulation time (a.u.)
MIN_E = 0.1           # Minimum initial energy (a.u.)
MAX_E = 3.0           # Maximum initial energy (a.u.)
N_STEP = 16           # Number of energy steps
BMAX_0 = 3.0            # Maximum impact parameter (a.u.)
# If True, it determines B_MAX based on initial energy
AUTO_BMAX = False
THRESH_1 = 2.3      # energy threshold for stepping b_max
THRESH_2 = 1.2
B1, B2, B3 = 1.0, 2.0, 3.0  # impact parameters (a.u.)

# If True, the code will save the first capture trajectory for each energy
TRAJ_SAVED = True

# LOADING THE GS ATOM
# Define the directory and file name
DIRECTORY_ATOM = '/scratch/vym17xaj/HPC_results_gs_with_alpha_modifying/' \
                + '02_He_02e.csv'

# Load RESULTS FOR PLOTTING from the CSV file in the directory:
RESULTS_DIR = 'HPC_dynamics_R0_{:.1f}_Ntraj_{:d}'.format(XPBAR, int(N_TRAJ))