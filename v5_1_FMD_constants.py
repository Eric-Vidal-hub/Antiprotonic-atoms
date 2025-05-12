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
MIN_E = 0.5           # Minimum initial energy (a.u.)
MAX_E = 3.0           # Maximum initial energy (a.u.)
N_STEP = 5           # Number of energy steps
N_TRAJ = 5          # Number of trajectories per energy
T_MAX = 25000.0       # Maximum simulation time (a.u.)
BMAX = 3.0            # Maximum impact parameter (a.u.)
XPBAR = 2.0          # Initial distance of antiproton (a.u.)

# DIRECTORY TO SAVE THE RESULTS
DIRECTORY_PBAR = 'dynamics_R0_2.0/'

# LOADING THE GS ATOM
# Define the directory and file name
DIRECTORY_ATOM = 'HPC_results_gs_with_alpha_modifying/'
DIRECTORY_ATOM += '02_He_02e.csv'
