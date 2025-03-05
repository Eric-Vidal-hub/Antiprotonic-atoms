from scipy.optimize import minimize
import numpy as np


# import gs_functions as gs
# INITIAL VARIABLES (dimensionless)
zz = 2      # Number of protons/electrons
alpha = 5   # Hardness parameter for the soft-Coulomb potential
xi_h = 1.000    # Heisenberg constraint parameter
xi_p = 2.767    # Pauli exclusion constraint parameter

# INITIAL CONDITIONS (Atomic units: hbar = m = e = 4pi eps0 = 1)
seed = 1234
rng = np.random.default_rng(seed)

# Generate initial positions and momenta in spherical coordinates
r0 = rng.uniform(1, 1.2, zz)    # Initial radial positions
theta0 = rng.uniform(0, np.pi, zz)  # Initial polar angles
phi0 = rng.uniform(0, 2 * np.pi, zz)  # Initial azimuthal angles

# Generate initial momenta in spherical coordinates
p0 = rng.uniform(1, 1.2, zz)    # Initial radial momenta
theta_p0 = rng.uniform(0, np.pi, zz)  # Initial polar angles for momenta
phi_p0 = rng.uniform(0, 2 * np.pi, zz)  # Initial azimuthal angles for momenta

# Initial configuration
xx = np.concatenate((r0, theta0, phi0, p0, theta_p0, phi_p0))

# Generate electron spins for arbitrary zz
e_spin = np.array([1 if i % 2 == 0 else -1 for i in range(zz)])
print("Electron spins:", e_spin)

# EFFECTIVE FMD HAMILTONIAN (a.u.)
def hamiltonian(xx):
    rr = np.zeros(zz)
    theta = np.zeros(zz)
    phi = np.zeros(zz)
    pp = np.zeros(zz)
    theta_p = np.zeros(zz)
    phi_p = np.zeros(zz)
    for i in range(zz):
        rr[i] = xx[i]
        theta[i] = xx[zz + i]
        phi[i] = xx[2 * zz + i]
        pp[i] = xx[3 * zz + i]
        theta_p[i] = xx[4 * zz + i]
        phi_p[i] = xx[5 * zz + i]

    # ONE-BODY POTENTIALS
    kin_pot = np.sum(0.5 * pp**2)   # Kinetic energy
    nuc_pot = -np.sum(zz / rr)      # Nuclear-e Coulomb potential
    # Heisenberg constraint potential to enforce: r_i * p_i >= xi_H.
    heisen_pot = np.sum(xi_h**2 * np.exp(alpha * (1 - (rr * pp / xi_h)**4))
                        / (4 * alpha * rr**2))

    # TWO-BODY POTENTIALS for multi-e systems
    h_multi = 0.0
    if zz > 1:
        pauli_pot = 0.0     # Pauli exclusion constraint potential
        pair_pot = 0.0      # Coulomb potential between electrons
        for i in range(zz):
            for j in range(i + 1, zz):
                #print(i, j)
                delta_r = np.abs(rr[i] - rr[j])
                pair_pot += 1.0 / delta_r
                # For identical elecrons
                if e_spin[i] == e_spin[j]:
                    print("Identical electrons")
                    delta_p = 0.5 * np.abs(pp[i] - pp[j])
                    pauli_pot += (xi_p**2 / (4 * alpha * delta_r**2)) * np.exp(alpha * (1 - (delta_r * delta_p / xi_p)**4))
        h_multi = pair_pot + pauli_pot
    # TOTAL ENERGY
    return kin_pot + nuc_pot + heisen_pot + h_multi


# Use a simple optimization routine to minimize the Hamiltonian.
# List of optimizers

optimizers = [
    'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 
    'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'
]

result = minimize(hamiltonian, xx, method='BFGS')
print("Optimal configuration (r, p):", result.x)
print("Ground state energy:", result.fun)

# # Use each optimizer to minimize the Hamiltonian
# for method in optimizers:
#     result = minimize(hamiltonian, xx, method=method)
#     print(f"Optimizer: {method}")
#     print("Optimal configuration (r, p):", result.x)
#     print("Ground state energy:", result.fun)
#     print()
