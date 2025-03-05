import GS_functions as gs
from scipy.optimize import minimize

# Initial guess for the configuration: [r, p]
x0 = [1.0, 1.0]

# Use a simple optimization routine to minimize the Hamiltonian.
result = minimize(gs.hamiltonian, x0, method='Nelder-Mead')

print("Optimal configuration (r, p):", result.x)
print("Ground state energy:", result.fun)
