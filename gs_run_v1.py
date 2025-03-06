import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class HamiltonianOptimizer:
    def __init__(self, alpha, xi_h, xi_p, seed=1234):
        self.alpha = alpha
        self.xi_h = xi_h
        self.xi_p = xi_p
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.optimizers = [
            'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'SLSQP'
        ]

    def generate_initial_config(self, e_num):
        half_e_num = e_num // 2 + e_num % 2  # Rounded up half number of electrons
        r0 = self.rng.uniform(1, 1.2, half_e_num)  # Ini radial positions
        theta0 = self.rng.uniform(0, np.pi, half_e_num)  # Ini polar angles
        phi0 = self.rng.uniform(0, 2 * np.pi, half_e_num)  # Ini azimuthal angles
        # Mirror the other half
        r0 = np.concatenate((r0, r0[:e_num - half_e_num]))
        theta0 = np.concatenate((theta0, np.pi - theta0[:e_num - half_e_num]))
        phi0 = np.concatenate((phi0, phi0[:e_num - half_e_num] + np.pi))

        p0 = self.rng.uniform(1, 1.2, half_e_num)  # Ini radial momenta
        theta_p0 = self.rng.uniform(0, np.pi, half_e_num)  # Ini polar angles for p
        phi_p0 = self.rng.uniform(0, 2 * np.pi, half_e_num)  # Ini azimuthal angles for p
        # Mirror the other half
        p0 = np.concatenate((p0, p0[:e_num - half_e_num]))
        theta_p0 = np.concatenate((theta_p0, np.pi - theta_p0[:e_num - half_e_num]))
        phi_p0 = np.concatenate((phi_p0, phi_p0[:e_num - half_e_num] + np.pi))

        # Initial configuration
        return np.concatenate((r0, theta0, phi0, p0, theta_p0, phi_p0))

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
        h_multi = 0.0
        if e_num > 1:
            pauli_pot = 0.0  # Pauli exclusion constraint potential
            pair_pot = 0.0  # Coulomb potential between electrons
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
            h_multi = pair_pot + pauli_pot
        return h_multi

    def hamiltonian(self, xx, e_num, e_spin):
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

        x_coord, y_coord, z_coord, px, py, pz = self.convert_to_cartesian(
            rr, theta, phi, pp, theta_p, phi_p
        )

        kin_pot, nuc_pot, heisen_pot = self.one_body_potentials(rr, pp)
        h_multi = self.two_body_potentials(e_num, e_spin, x_coord, y_coord, z_coord, px, py, pz)

        return kin_pot + nuc_pot + heisen_pot + h_multi  # TOTAL ENERGY

    def plot_results(self):
        # Read the results from the CSV file
        df = pd.read_csv('results.csv')

        # Plot the ground state energy for each e_num and optimizer
        plt.figure(figsize=(10, 6))
        for optimizer in df['optimizer'].unique():
            subset = df[df['optimizer'] == optimizer]
            plt.plot(subset['e_num'], subset['ground_state_energy'], label=optimizer)

        plt.xlabel('Number of Electrons (e_num)')
        plt.ylabel('Ground State Energy')
        plt.legend()
        plt.grid(True)
        plt.savefig('ground_state_energy_plot.png')
        plt.show()


def optimize(optimizer, e_num_values):
    # Open the CSV file to write the results
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['e_num', 'optimizer', 'optimal_configuration', 'ground_state_energy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for e_num in e_num_values:
            # Generate electron spins for arbitrary e_num
            e_spin = np.array([1 if i % 2 == 0 else -1 for i in range(e_num)])

            # Generate initial configuration
            ini_config = optimizer.generate_initial_config(e_num)

            # Use each optimizer to minimize the Hamiltonian
            for method in optimizer.optimizers:
                result = minimize(
                    optimizer.hamiltonian, ini_config, args=(e_num, e_spin), method=method
                )
                writer.writerow({
                    'e_num': e_num,
                    'optimizer': method,
                    'optimal_configuration': np.array2string(result.x),
                    'ground_state_energy': result.fun
                })


# Example usage
optimizer = HamiltonianOptimizer(alpha=5, xi_h=1.000, xi_p=2.767)
e_num_values = [2, 3, 4]
optimize(optimizer, e_num_values)
optimizer.plot_results()
