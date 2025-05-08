# Physical constants
# Proton mass (a.u.)
M_PBAR = 1836.152672  # antiproton mass (a.u.)
ALPHA = 5             # Hardness parameter
XI_H = 1.000          # Tuning parameter for the Heisenberg potential
XI_P = 2.767          # Tuning parameter for the Pauli potential

# Scaling parameters according to alpha
# XI_H /= (1 + ALPHA / 2)**0.5
# XI_P /= (1 + ALPHA / 2)**0.5

print(f"XI_H: {XI_H}, XI_P: {XI_P}")

# Simulation parameters
MIN_E = 0.1           # Minimum initial energy (a.u.)
MAX_E = 3.0           # Maximum initial energy (a.u.)
N_STEP = 16           # Number of energy steps
N_TRAJ = 100          # Number of trajectories per energy
T_MAX = 25000.0       # Maximum simulation time (a.u.)
BMAX = 3.0            # Maximum impact parameter (a.u.)
XPBAR = 50.0          # Initial distance of antiproton (a.u.)