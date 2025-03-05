import numpy as np


# Define a penalty potential to enforce the Heisenberg constraint: r * p ~ 1.
def heisenberg_potential(r, p, gamma=10.0):
    # The penalty increases quadratically when r*p deviates from 1.
    return gamma * (r * p - 1.0)**2

# (For multi-electron systems, one could define a similar function for the Pauli exclusion constraint.)
def pauli_potential(r_i, r_j, p_i, p_j, beta=10.0):
    # Penalize if two electrons (with the same spin) are too similar in phase space.
    delta_r = np.abs(r_i - r_j)
    delta_p = 0.5 * np.abs(p_i - p_j)
    return beta * (delta_r * delta_p - 1.0)**2

# Define the Hamiltonian for a single electron in a Coulomb potential
# with a Heisenberg constraint. (Atomic units are used: hbar = m = e = 1.)
def hamiltonian(x):
    # x[0] is the radial position (r) and x[1] is the momentum (p).
    r, p = x[0], x[1]
    
    # Avoid division by zero.
    if r <= 0:
        return 1e6
    
    # Kinetic energy: T = p^2 / 2
    T = 0.5 * p**2
    # Coulomb potential: V = -1 / r (for the hydrogen atom)
    V_coulomb = -1.0 / r
    # Heisenberg constraint potential
    V_heisenberg = heisenberg_potential(r, p)
    
    # Total energy
    return T + V_coulomb + V_heisenberg
