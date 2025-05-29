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
T_MAX = 5.0       # Maximum simulation time (a.u.)
N_STEP = 100           # Number of energy steps

# LOADING THE GS ATOM
# Define the directory and file name
FILENAME = '02_He_02e'
DIRECTORY_ATOM = 'HPC_results_gs_with_alpha_modifying/' \
                + FILENAME + '.csv'

# Load RESULTS FOR PLOTTING from the CSV file in the directory:
RESULTS_DIR = 'EVO_{}_TIME_{:d}'.format(FILENAME, int(T_MAX))
