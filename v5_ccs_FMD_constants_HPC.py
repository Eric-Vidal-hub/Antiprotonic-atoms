# Physical constants
# Proton mass (a.u.)
M_PBAR = 1836.152672  # antiproton mass (a.u.)
ALPHA_H = 5             # Hardness parameter HEISENBERG potential
ALPHA_P = 5             # Hardness parameter PAULI potential
XI_H = 1.000          # Tuning parameter for the Heisenberg potential
XI_P = 2.767          # Tuning parameter for the Pauli potential

# Scaling parameters according to alpha
XI_H /= (1 + 1 / (2 * ALPHA_H))**0.5
XI_P /= (1 + 1 / (2 * ALPHA_P))**0.5

print(f"XI_H: {XI_H}, XI_P: {XI_P}")

# Simulation parameters
XPBAR = 10.0           # Initial distance of antiproton (a.u.)
N_TRAJ = 200          # Number of trajectories per energy
T_MAX = 25000.0       # Maximum simulation time (a.u.)
MIN_E = 0.01           # Minimum initial energy (a.u.)
MAX_E = 1.5           # Maximum initial energy (a.u.)
N_STEP = 16           # Number of energy steps
BMAX_0 = 3.0            # Maximum impact parameter (a.u.)
# If True, it determines B_MAX based on initial energy
AUTO_BMAX = True
THRESH_1 = 2.3      # energy threshold for stepping b_max
THRESH_2 = 1.2
B1, B2, B3 = 1.0, 2.0, 3.0  # impact parameters (a.u.)

# LOADING THE GS ATOM
# Define the directory and file name
# FILENAME = '02_He_02e'
FILENAME = '03_Li_03e'
DIRECTORY_ATOM = 'GS_alpha_HPC/' \
                + FILENAME + '.csv'
# FILENAME = '02_He_03e'

# DIRECTORY_ATOM = 'GS_alpha_anions_HPC/' \
                # + FILENAME + '.csv'
# FILENAME = '02_He_01e'
# FILENAME = '03_Li_02e'
# DIRECTORY_ATOM = 'GS_alpha_pos_ions_HPC/' \
#                 + FILENAME + '.csv'

# Load RESULTS FOR PLOTTING from the CSV file in the directory:
RESULTS_DIR = 'CCS_{}_R0_{:.1f}_Ntraj_{:d}_HPC'.format(FILENAME, XPBAR, int(N_TRAJ))
