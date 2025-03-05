from scipy.optimize import minimize
import numpy as np


# import gs_functions as gs
# INITIAL VARIABLES (dimensionless)
e_num = 3      # Number of protons/electrons
alpha = 5   # Hardness parameter for the soft-Coulomb potential
xi_h = 1.000    # Heisenberg constraint parameter
xi_p = 2.767    # Pauli exclusion constraint parameter

# INITIAL CONDITIONS (Atomic units: hbar = m = e = 4pi eps0 = 1)
seed = 1234
rng = np.random.default_rng(seed)

# Generate electron spins for arbitrary e_num
e_spin = np.array([1 if i % 2 == 0 else -1 for i in range(e_num)])
print("Electron spins:", e_spin)

# Generate initial positions and momenta in spherical coordinates
half_e_num = e_num // 2 + e_num % 2  # Half the number of electrons, rounded up if e_num is odd
r0 = rng.uniform(1, 1.2, half_e_num)    # Initial radial positions
theta0 = rng.uniform(0, np.pi, half_e_num)  # Initial polar angles
phi0 = rng.uniform(0, 2 * np.pi, half_e_num)  # Initial azimuthal angles

# Mirror the other half
r0 = np.concatenate((r0, r0[:e_num - half_e_num]))
theta0 = np.concatenate((theta0, np.pi - theta0[:e_num - half_e_num]))
phi0 = np.concatenate((phi0, phi0[:e_num - half_e_num] + np.pi))

# Generate initial momenta in spherical coordinates
p0 = rng.uniform(1, 1.2, half_e_num)    # Initial radial momenta
theta_p0 = rng.uniform(0, np.pi, half_e_num)  # Initial polar angles for momenta
phi_p0 = rng.uniform(0, 2 * np.pi, half_e_num)  # Initial azimuthal angles for momenta

# Mirror the other half
p0 = np.concatenate((p0, p0[:e_num - half_e_num]))
theta_p0 = np.concatenate((theta_p0, np.pi - theta_p0[:e_num - half_e_num]))
phi_p0 = np.concatenate((phi_p0, phi_p0[:e_num - half_e_num] + np.pi))

# Initial configuration
xx = np.concatenate((r0, theta0, phi0, p0, theta_p0, phi_p0))

# EFFECTIVE FMD HAMILTONIAN (a.u.)
def hamiltonian(xx):
    rr = np.zeros(e_num)
    theta = np.zeros(e_num)
    phi = np.zeros(e_num)
    pp = np.zeros(e_num)
    theta_p = np.zeros(e_num)
    phi_p = np.zeros(e_num)
    for i in range(e_num):
        rr[i] = xx[i]
        theta[i] = xx[e_num + i]
        phi[i] = xx[2 * e_num + i]
        pp[i] = xx[3 * e_num + i]
        theta_p[i] = xx[4 * e_num + i]
        phi_p[i] = xx[5 * e_num + i]

    # ONE-BODY POTENTIALS
    kin_pot = np.sum(0.5 * pp**2)   # Kinetic energy
    nuc_pot = -np.sum(e_num / rr)      # Nuclear-e Coulomb potential
    # Heisenberg constraint potential to enforce: r_i * p_i >= xi_H.
    heisen_pot = np.sum(xi_h**2 * np.exp(alpha * (1 - (rr * pp / xi_h)**4))
                        / (4 * alpha * rr**2))

    # TWO-BODY POTENTIALS for multi-e systems
    h_multi = 0.0
    if e_num > 1:
        pauli_pot = 0.0     # Pauli exclusion constraint potential
        pair_pot = 0.0      # Coulomb potential between electrons
        # Convert spherical coordinates to Cartesian coordinates
        xx = rr * np.sin(theta) * np.cos(phi)
        yy = rr * np.sin(theta) * np.sin(phi)
        zz = rr * np.cos(theta)

        px = pp * np.sin(theta_p) * np.cos(phi_p)
        py = pp * np.sin(theta_p) * np.sin(phi_p)
        pz = pp * np.cos(theta_p)
        for i in range(e_num):
            for j in range(i + 1, e_num):
                #print(i, j)
                delta_r = np.sqrt((xx[i] - xx[j])**2 + (yy[i] - yy[j])**2 + (zz[i] - zz[j])**2)
                # Coulomb potential between electrons
                pair_pot += 1.0 / delta_r
                # For identical elecrons
                if e_spin[i] == e_spin[j]:
                    print("Identical electrons")
                    delta_p = 0.5 * np.sqrt((px[i] - px[j])**2 + (py[i] - py[j])**2 + (pz[i] - pz[j])**2)
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

# # Use each optimizer to minimize the Hamiltonian
# for method in optimizers:
#     print(f"Optimizer: {method}")
result = minimize(hamiltonian, xx, method='BFGS')
print("Optimal configuration (r, p):", result.x)
print("Ground state energy:", result.fun)
