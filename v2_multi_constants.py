# Physical constants
# Proton mass (a.u.)
from re import A


M_PBAR = 1836.152672  # antiproton mass (a.u.)
ALPHA_H = 5             # Hardness parameter for Heisenberg potential
ALPHA_P = 5             # Hardness parameter for Pauli potential
XI_H = 1.000          # Tuning parameter for the Heisenberg potential
XI_P = 2.767          # Tuning parameter for the Pauli potential

# Scaling parameters according to alpha
XI_H /= (1 + 1 / (2 * ALPHA_H))**0.5
XI_P /= (1 + 1 / (2 * ALPHA_P))**0.5

print(f"XI_H: {XI_H}, XI_P: {XI_P}")

# Simulation parameters
T_MAX = 5.0       # Maximum simulation time (a.u.)
N_STEP = 100           # Number of energy steps

# LOADING THE GS ATOM
# Define the directory and file name
FILENAME = '03_Li_03e'
DIRECTORY_ATOM = 'GS_alpha_HPC/' \
                + FILENAME + '.csv'

# Load RESULTS FOR PLOTTING from the CSV file in the directory:
RESULTS_DIR = 'EVO_{}_TIME_{:d}_RND'.format(FILENAME, int(T_MAX))
