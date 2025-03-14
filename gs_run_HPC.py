import csv
import numpy as np
from scipy.optimize import fmin_bfgs
from progressbar import progressbar
import time


class HamiltonianOptimizer:
    def __init__(self, alpha, xi_h, xi_p, seed=1234, optimizers=['BFGS']):
        self.alpha = alpha
        self.xi_h = xi_h
        self.xi_p = xi_p
        self.seed = seed
        self.optimizers = optimizers
        self.rng = np.random.default_rng(seed)

    def generate_initial_config(self, e_num):
        half_e_num = e_num // 2 + e_num % 2  # Rounded up half number of electrons
        r0 = self.rng.uniform(1, 1.2, half_e_num).astype(np.float64)  # Ini radial positions
        theta0 = self.rng.uniform(0, np.pi, half_e_num).astype(np.float64)  # Ini polar angles
        phi0 = self.rng.uniform(0, 2 * np.pi, half_e_num).astype(np.float64)  # Ini azimuthal angles
        # Mirror the other half
        r0 = np.concatenate((r0, r0[:e_num - half_e_num]))
        theta0 = np.concatenate((theta0, np.pi - theta0[:e_num - half_e_num]))
        phi0 = np.concatenate((phi0, phi0[:e_num - half_e_num] + np.pi))

        p0 = self.rng.uniform(1, 1.2, half_e_num).astype(np.float64)  # Ini radial momenta
        theta_p0 = self.rng.uniform(0, np.pi, half_e_num).astype(np.float64)  # Ini polar angles for p
        phi_p0 = self.rng.uniform(0, 2 * np.pi, half_e_num).astype(np.float64)  # Ini azimuthal angles for p
        # Mirror the other half
        p0 = np.concatenate((p0, p0[:e_num - half_e_num]))
        theta_p0 = np.concatenate((theta_p0, np.pi - theta_p0[:e_num - half_e_num]))
        phi_p0 = np.concatenate((phi_p0, phi_p0[:e_num - half_e_num] + np.pi))

        # Initial configuration
        return np.concatenate((r0, theta0, phi0, p0, theta_p0, phi_p0))

    # def generate_initial_config(self, e_num):
    #     half_e_num = e_num // 2 + e_num % 2  # Rounded up half number of electrons
    #     r0 = self.rng.uniform(1, 1.2, half_e_num) * (np.arange(1, half_e_num + 1) ** 2 / e_num)
    #     theta0 = self.rng.uniform(0, np.pi, half_e_num).astype(np.float64)  # Ini polar angles
    #     phi0 = self.rng.uniform(0, 2 * np.pi, half_e_num).astype(np.float64)  # Ini azimuthal angles
    #     # Mirror the other half
    #     r0 = np.concatenate((r0, r0[:e_num - half_e_num]))
    #     theta0 = np.concatenate((theta0, np.pi - theta0[:e_num - half_e_num]))
    #     phi0 = np.concatenate((phi0, phi0[:e_num - half_e_num] + np.pi))

    #     p0 = r0  # Ini momenta for p
    #     theta_p0 = theta0  # Ini polar angles for p same as theta0
    #     phi_p0 = phi0  # Ini azimuthal angles for p same as phi0

    #     print('r0 for e_num {}: {}'.format(e_num, r0))
    #     # Initial configuration
    #     return np.concatenate((r0, theta0, phi0, p0, theta_p0, phi_p0))

    def convert_to_cartesian(self, rr, theta, phi, pp, theta_p, phi_p):
        x_coord = rr * np.sin(theta) * np.cos(phi)
        y_coord = rr * np.sin(theta) * np.sin(phi)
        z_coord = rr * np.cos(theta)

        px = pp * np.sin(theta_p) * np.cos(phi_p)
        py = pp * np.sin(theta_p) * np.sin(phi_p)
        pz = pp * np.cos(theta_p)

        return x_coord, y_coord, z_coord, px, py, pz

    def one_body_potentials(self, rr, pp):
        kin_pot = np.sum(0.5 * pp**2)  # Kinetic energy
        nuc_pot = -np.sum(len(rr) / rr)  # Nuclear-e Coulomb potential
        heisen_pot = np.sum(
            self.xi_h**2 * np.exp(self.alpha * (1 - (rr * pp / self.xi_h)**4))
            / (4 * self.alpha * rr**2)
        )
        return kin_pot, nuc_pot, heisen_pot

    def two_body_potentials(self, e_num, e_spin, x_coord, y_coord, z_coord, px, py, pz):
        pair_pot = 0.0  # Coulomb potential between electrons
        pauli_pot = 0.0  # Pauli exclusion constraint potential
        if e_num > 1:
            for i in range(e_num):
                for j in range(i + 1, e_num):
                    delta_r = np.sqrt(
                        (x_coord[i] - x_coord[j])**2 +
                        (y_coord[i] - y_coord[j])**2 +
                        (z_coord[i] - z_coord[j])**2
                    )
                    # Coulomb potential for electron pairs
                    pair_pot += 1.0 / delta_r
                    # For identical electrons
                    if e_spin[i] == e_spin[j]:
                        delta_p = 0.5 * np.sqrt(
                            (px[i] - px[j])**2 +
                            (py[i] - py[j])**2 +
                            (pz[i] - pz[j])**2
                        )
                        pauli_pot += (
                            self.xi_p**2 / (4 * self.alpha * delta_r**2)
                        ) * np.exp(
                            self.alpha * (1 - (delta_r * delta_p / self.xi_p)**4)
                        )
        return pair_pot, pauli_pot

    def hamiltonian(self, xx, e_num, e_spin):
        rr = np.zeros(e_num, dtype=np.float64)
        theta = np.zeros(e_num, dtype=np.float64)
        phi = np.zeros(e_num, dtype=np.float64)
        pp = np.zeros(e_num, dtype=np.float64)
        theta_p = np.zeros(e_num, dtype=np.float64)
        phi_p = np.zeros(e_num, dtype=np.float64)
        for i in range(e_num):
            rr[i] = xx[i]
            theta[i] = xx[e_num + i]
            phi[i] = xx[2 * e_num + i]
            pp[i] = xx[3 * e_num + i]
            theta_p[i] = xx[4 * e_num + i]
            phi_p[i] = xx[5 * e_num + i]

        x_coord, y_coord, z_coord, px, py, pz = self.convert_to_cartesian(
            rr, theta, phi, pp, theta_p, phi_p
        )

        kin_pot, nuc_pot, heisen_pot = self.one_body_potentials(rr, pp)
        pair_pot, pauli_pot = self.two_body_potentials(e_num, e_spin, x_coord, y_coord, z_coord, px, py, pz)

        total_energy = kin_pot + nuc_pot + heisen_pot + pair_pot + pauli_pot
        return total_energy

    def hamiltonian_components(self, xx, e_num, e_spin):
        rr = np.zeros(e_num, dtype=np.float64)
        theta = np.zeros(e_num, dtype=np.float64)
        phi = np.zeros(e_num, dtype=np.float64)
        pp = np.zeros(e_num, dtype=np.float64)
        theta_p = np.zeros(e_num, dtype=np.float64)
        phi_p = np.zeros(e_num, dtype=np.float64)
        for i in range(e_num):
            rr[i] = xx[i]
            theta[i] = xx[e_num + i]
            phi[i] = xx[2 * e_num + i]
            pp[i] = xx[3 * e_num + i]
            theta_p[i] = xx[4 * e_num + i]
            phi_p[i] = xx[5 * e_num + i]

        x_coord, y_coord, z_coord, px, py, pz = self.convert_to_cartesian(
            rr, theta, phi, pp, theta_p, phi_p
        )

        kin_pot, nuc_pot, heisen_pot = self.one_body_potentials(rr, pp)
        pair_pot, pauli_pot = self.two_body_potentials(e_num, e_spin, x_coord, y_coord, z_coord, px, py, pz)

        return kin_pot, nuc_pot, heisen_pot, pair_pot, pauli_pot  # Return individual components


# # INITIAL CONFIGURATION VALUES
# alpha = 5
# xi_h = 1.000
# xi_p = 2.767
# # # Scaling parameters according to alpha
# # xi_h = xi_h / np.sqrt(1 + 1 / (2 * alpha))
# # xi_p = xi_p / np.sqrt(1 + 1 / (2 * alpha))
# # print(f'For alpha = {alpha}: xi_h = {xi_h:.3f}, xi_p = {xi_p:.3f}')

# # Number of electrons range to study
# e_ini = 1
# e_fin = 14
# # POSSIBLE OPTIMIZERS, ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
# # optimizers = ['BFGS', 'trust-constr'] # Powell and SLSQP are very fast but do not convey a proper electron structure and convergence (SLSQP is the fastest)
# # optimizers = ['trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']

# optimizer = HamiltonianOptimizer(alpha, xi_h, xi_p)
# e_num_values = list(range(e_ini, e_fin + 1))

# # Open the CSV file to write the results
# output_filename = f'results_alpha_{alpha}_xi_h_{xi_h:.3f}_xi_p_{xi_p:.3f}_e_{e_ini}_to_{e_fin}.csv'
# with open(output_filename, 'w', newline='') as csvfile:
#     fieldnames = ['e_num', 'message', 'time_taken', 'ground_state_energy', 'kinetic_energy', 'nuclear_potential', 'heisenberg_potential', 'pair_potential', 'pauli_potential', 'optimal_configuration']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()

#     for e_num in progressbar(e_num_values):
#         # Generate electron spins for arbitrary e_num
#         e_spin = np.array([1 if i % 2 == 0 else -1 for i in range(e_num)])

#         # Generate initial configuration
#         ini_config = optimizer.generate_initial_config(e_num)
#         start_time = time.time()
        
#         result = fmin_bfgs(
#             optimizer.hamiltonian, ini_config, args=(e_num, e_spin), gtol=1e-3, full_output=True, disp=False, retall=False
#         )
#         optimal_config, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = result
#         converged = warnflag == 0

#         if converged:
#             message = 'Converged'
#         else:
#             message = 'NOT Converged'

#         end_time = time.time()
#         time_taken = end_time - start_time

#         # Extract individual components of the Hamiltonian
#         kin_pot, nuc_pot, heisen_pot, pair_pot, pauli_pot = optimizer.hamiltonian_components(optimal_config, e_num, e_spin)
#         ground_state_energy = kin_pot + nuc_pot + heisen_pot + pair_pot + pauli_pot

#         writer.writerow({
#             'e_num': e_num,
#             'ground_state_energy': f'{ground_state_energy:.3f}',
#             'kinetic_energy': f'{kin_pot:.3f}',
#             'nuclear_potential': f'{nuc_pot:.3f}',
#             'heisenberg_potential': f'{heisen_pot:.3f}',
#             'pair_potential': f'{pair_pot:.3f}',
#             'pauli_potential': f'{pauli_pot:.3f}',
#             'time_taken': f'{time_taken:.3f}',
#             'message': message,
#             'optimal_configuration': np.array2string(optimal_config)
#         })


# INITIAL CONFIGURATION VALUES
alpha = 5
xi_h = 1.000
xi_p = 2.767
# # Scaling parameters according to alpha
# xi_h = xi_h / np.sqrt(1 + 1 / (2 * alpha))
# xi_p = xi_p / np.sqrt(1 + 1 / (2 * alpha))
# print(f'For alpha = {alpha}: xi_h = {xi_h:.3f}, xi_p = {xi_p:.3f}')
# Number of electrons range to study
e_ini = 1
e_fin = 14
# POSSIBLE OPTIMIZERS, ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
# optimizers = ['BFGS', 'trust-constr'] # Powell and SLSQP are very fast but do not convey a proper electron structure and convergence (SLSQP is the fastest)
# optimizers = ['trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']

optimizer = HamiltonianOptimizer(alpha, xi_h, xi_p)
e_num_values = list(range(e_ini, e_fin + 1))

# Open the CSV file to write the results
output_filename = f'results_alpha_{alpha}_xi_h_{xi_h:.3f}_xi_p_{xi_p:.3f}_e_{e_ini}_to_{e_fin}.csv'
with open(output_filename, 'w', newline='') as csvfile:
    fieldnames = ['e_num', 'message', 'time_taken', 'ground_state_energy', 'kinetic_energy', 'nuclear_potential', 'heisenberg_potential', 'pair_potential', 'pauli_potential', 'optimal_configuration']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for e_num in progressbar(e_num_values):
        # Generate electron spins for arbitrary e_num
        e_spin = np.array([1 if i % 2 == 0 else -1 for i in range(e_num)])

        # Generate initial configuration
        ini_config = optimizer.generate_initial_config(e_num)
        start_time = time.time()
        
        maxiter = 6
        iter_num = 0
        converged = False
        positive = np.all(ini_config[:e_num] > 0) # and np.all(ini_config[3*e_num:4*e_num] > 0)
        while not converged or not positive:
            if iter_num >= maxiter:
                message = f'NOT Converged after {maxiter} iterations'
                break
            # if there is any negative r0 and p0 change to absolute value
            if not positive:
                ini_config[:e_num] = np.abs(ini_config[:e_num])
                # ini_config[3*e_num:4*e_num] = np.abs(ini_config[3*e_num:4*e_num])
            result = fmin_bfgs(
            optimizer.hamiltonian, ini_config, args=(e_num, e_spin), gtol=1e-4, full_output=True, disp=False, retall=False
            )
            optimal_config, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = result
            converged = warnflag == 0
            ini_config = optimal_config
            
            positive = np.all(ini_config[:e_num] > 0) # and np.all(ini_config[3*e_num:4*e_num] > 0)
            iter_num += 1

        if converged:
            message = 'Converged after {} iterations'.format(iter_num)

        end_time = time.time()
        time_taken = end_time - start_time

        # Extract individual components of the Hamiltonian
        kin_pot, nuc_pot, heisen_pot, pair_pot, pauli_pot = optimizer.hamiltonian_components(optimal_config, e_num, e_spin)
        ground_state_energy = kin_pot + nuc_pot + heisen_pot + pair_pot + pauli_pot

        writer.writerow({
            'e_num': e_num,
            'ground_state_energy': f'{ground_state_energy:.3f}',
            'kinetic_energy': f'{kin_pot:.3f}',
            'nuclear_potential': f'{nuc_pot:.3f}',
            'heisenberg_potential': f'{heisen_pot:.3f}',
            'pair_potential': f'{pair_pot:.3f}',
            'pauli_potential': f'{pauli_pot:.3f}',
            'time_taken': f'{time_taken:.3f}',
            'message': message,
            'optimal_configuration': np.array2string(optimal_config)
        })
