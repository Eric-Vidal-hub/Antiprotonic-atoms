# Physical constants
# Proton mass (a.u.)
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
T_MAX = 50.0       # Maximum simulation time (a.u.)
N_STEP = 1000       # Number of evaluation time steps

# LOADING THE GS ATOM
# Define the directory and file name
# FILENAME = '02_He_02e'
# FILENAME = '03_Li_03e'
# DIRECTORY_ATOM = 'GS_alpha_HPC/' \
#                 + FILENAME + '.csv'
# FILENAME = '02_He_03e'

# DIRECTORY_ATOM = 'GS_alpha_anions_HPC/' \
                # + FILENAME + '.csv'
# FILENAME = '02_He_01e'
FILENAME = '03_Li_02e'
DIRECTORY_ATOM = 'GS_alpha_pos_ions_HPC/' \
                + FILENAME + '.csv'

# Load RESULTS FOR PLOTTING from the CSV file in the directory:
RESULTS_DIR = 'EVO_{}_TIME_{:d}_RND'.format(FILENAME, int(T_MAX))

# Plot control flags
PLOT_POSITION = True
PLOT_MOMENTUM = True
PLOT_ENERGY = True
PLOT_COMPONENTS = True
PLOT_GIF = True
N_FRAMES = N_STEP  # Number of frames for the GIF
FPS = 30  # Frames per second for the GIF
