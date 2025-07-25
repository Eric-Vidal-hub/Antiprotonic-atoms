"""
Author: Eric Vidal Marcos
Date: 27-03-2025
Project: GS study using the FMD semi-classical model with V_H and V_P.

This module contains the HamiltonianOptimizer class and the main script
to run the Hamiltonian optimization for a given set of parameters.

The HamiltonianOptimizer class is used to optimize the Hamiltonian for
a given set of parameters, including the number of electrons, scaling
parameters for the Heisenberg and Pauli potentials, and the seed for
the random number generator.

The main script initializes the HamiltonianOptimizer with the specified
parameters, generates initial configurations for a range of electron
numbers, and performs the optimization using the BFGS algorithm. The
results are written to a CSV file.
"""

import sys
import os
import csv
import time
import numpy as np
from scipy.optimize import fmin_bfgs
from v5_gs_constants_HPC import (
    ALPHA_H, ALPHA_P, XI_H, XI_P, XI_H_RYD, MAXITER, GTOL, ELEMENTS_LIST,
    DIFF_P_E, RESULTS_DIR, P_NUM
)


class HamiltonianOptimizer:
    """Optimizes the Hamiltonian of a system of electrons.

    This class calculates the total Hamiltonian of a system of electrons
    interacting with a nucleus, including kinetic energy, Coulomb
    interactions, and additional Heisenberg and Pauli potentials. It uses
    optimization algorithms to find the ground state energy and
    configuration.
    """
    def __init__(self, alpha_h, alpha_p, xi_h, xi_p, xi_h_ryd=None, seed=1234, optimizers=None):
        """Initializes the HamiltonianOptimizer.

        Initializes the HamiltonianOptimizer with the given parameters for
        the Heisenberg and Pauli potentials, random number generator seed,
        and optimization algorithms.

        Args:
            alpha: Hardness parameter.
            xi_h: Tuning parameter for the Heisenberg potential (V_H).
            xi_p: Tuning parameter for the Pauli potential (V_P).
            xi_h_ryd: Optional tuning parameter for the Heisenberg potential
                      for the last electron (in Rydberg units).
            seed: Seed for the random number generator.
            optimizers: List of optimization algorithms to use.
        """

        if optimizers is None:
            optimizers = ['BFGS']
        self.alpha_h = alpha_h
        self.alpha_p = alpha_p
        self.xi_h = xi_h
        self.xi_p = xi_p
        self.xi_h_ryd = xi_h_ryd
        self.seed = seed
        self.optimizers = optimizers
        self.rng = np.random.default_rng(seed)

    def generate_initial_config(self, e_num):
        """Generates an initial configuration for the electrons.

        The initial configuration is generated by randomly distributing the
        electrons in a spherical shell and assigning them random momenta.
        The configuration is mirrored to ensure symmetry.

        Args:
            e_num: The number of electrons.

        Returns:
            A numpy array representing the initial configuration.
        """

        half_e_num = e_num // 2 + e_num % 2  # Rounded up half num_e
        r0 = self.rng.uniform(1, 1.2, half_e_num) * (
            np.arange(1, half_e_num + 1) ** 2 / e_num
        )
        theta0 = self.rng.uniform(0, np.pi, half_e_num).astype(np.float64)
        phi0 = self.rng.uniform(0, 2 * np.pi, half_e_num).astype(np.float64)
        # Mirror the other half
        r0 = np.concatenate((r0, r0[:e_num - half_e_num]))
        theta0 = np.concatenate((theta0, np.pi - theta0[:e_num - half_e_num]))
        phi0 = np.concatenate((phi0, phi0[:e_num - half_e_num] + np.pi))

        p0 = r0             # Ini momenta for p
        theta_p0 = theta0   # Ini polar angles for p same as theta0
        phi_p0 = phi0       # Ini azimuthal angles for p same as phi0

        return np.concatenate((p0, theta_p0, phi_p0, p0, theta_p0, phi_p0))

    def convert_to_cartesian(self, rr, theta, phi, pp, theta_p, phi_p):
        """Converts spherical coordinates to Cartesian coordinates.

        Converts the given spherical coordinates (r, theta, phi) and momenta
        (p, theta_p, phi_p) to Cartesian coordinates (x, y, z) and momenta
        (px, py, pz).

        Args:
            rr: Radial distance.
            theta: Polar angle.
            phi: Azimuthal angle.
            pp: Momentum magnitude.
            theta_p: Polar angle of momentum.
            phi_p: Azimuthal angle of momentum.

        Returns:
            A tuple containing the Cartesian coordinates (x, y, z, px, py, pz).
        """

        x_coord = rr * np.sin(theta) * np.cos(phi)
        y_coord = rr * np.sin(theta) * np.sin(phi)
        z_coord = rr * np.cos(theta)

        px = pp * np.sin(theta_p) * np.cos(phi_p)
        py = pp * np.sin(theta_p) * np.sin(phi_p)
        pz = pp * np.cos(theta_p)

        return x_coord, y_coord, z_coord, px, py, pz

    def one_body_potentials(self, rr, pp, p_num):
        """Calculates the one-body potentials, using xi_h_ryd for the last electron if set.

        Calculates the kinetic energy, nuclear-electron Coulomb potential,
        and Heisenberg potential for the given radial distances and momenta.

        Args:
            rr: Radial distances of the electrons.
            pp: Momenta of the electrons.

        Returns:
            A tuple containing the kinetic, nuclear, and Heisenberg potentials.
        """

        xi_h_array = np.full_like(rr, self.xi_h)
        if self.xi_h_ryd is not None and len(rr) > 1 and self.xi_h_ryd != self.xi_h:
            xi_h_array[-1] = self.xi_h_ryd
        kin_pot = np.sum(0.5 * pp ** 2)     # Kinetic energy
        nuc_pot = -np.sum(p_num / rr)       # Nuclear-e Coulomb potential
        heisen_pot = np.sum(
            xi_h_array ** 2 * np.exp(self.alpha_h *
                                    (1 - (rr * pp / xi_h_array) ** 4)) /
            (4 * self.alpha_h * rr ** 2)
        )
        return kin_pot, nuc_pot, heisen_pot

    def two_body_potentials(self, e_num, e_spin,
                            x_coord, y_coord, z_coord,
                            px, py, pz):
        """Calculates the two-body potentials.

        Calculates the electron-electron Coulomb potential and the Pauli
        exclusion constraint potential for the given electron coordinates
        and momenta.

        Args:
            e_num: The number of electrons.
            e_spin: Array of electron spins.
            x_coord: x-coordinates of the electrons.
            y_coord: y-coordinates of the electrons.
            z_coord: z-coordinates of the electrons.
            px: x-component of the momenta.
            py: y-component of the momenta.
            pz: z-component of the momenta.

        Returns:
            A tuple containing the pair potential and the Pauli potential.
        """

        pair_pot = 0.0  # Coulomb potential between electrons
        pauli_pot = 0.0  # Pauli exclusion constraint potential
        if e_num > 1:
            for i in range(e_num):
                for j in range(i + 1, e_num):
                    delta_r = np.sqrt(
                        (x_coord[i] - x_coord[j]) ** 2 +
                        (y_coord[i] - y_coord[j]) ** 2 +
                        (z_coord[i] - z_coord[j]) ** 2
                    )
                    # Coulomb potential for electron pairs
                    pair_pot += 1.0 / delta_r
                    # For identical electrons
                    if e_spin[i] == e_spin[j]:
                        delta_p = 0.5 * np.sqrt(
                            (px[i] - px[j]) ** 2 +
                            (py[i] - py[j]) ** 2 +
                            (pz[i] - pz[j]) ** 2
                        )
                        pauli_pot += (
                            self.xi_p ** 2 / (2 * self.alpha_p * delta_r ** 2)
                        ) * np.exp(
                            self.alpha_p * (1 - (delta_r * delta_p
                                               / self.xi_p) ** 4)
                        )
        return pair_pot, pauli_pot

    def hamiltonian(self, xx, e_num, e_spin, p_num):
        """Calculates the total Hamiltonian.

        Calculates the total Hamiltonian of the system for the given
        configuration, number of electrons, and electron spins.

        Args:
            xx: The configuration vector.
            e_num: The number of electrons.
            e_spin: Array of electron spins.

        Returns:
            The total energy of the system.
        """

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

        kin_pot, nuc_pot, heisen_pot = self.one_body_potentials(rr, pp, p_num)
        pair_pot, pauli_pot = self.two_body_potentials(
            e_num, e_spin, x_coord, y_coord, z_coord, px, py, pz
        )

        return kin_pot + nuc_pot + heisen_pot + pair_pot + pauli_pot

    def hamiltonian_components(self, xx, e_num, e_spin, p_num):
        """Calculates individual components of the Hamiltonian.

        Calculates the kinetic, nuclear, Heisenberg, pair, and Pauli potentials
        separately for the given configuration, number of electrons, and spins.

        Args:
            xx: The configuration vector.
            e_num: The number of electrons.
            e_spin: Array of electron spins.

        Returns:
            A tuple containing the kinetic, nuclear, Heisenberg, pair,
            and Pauli potentials.
        """

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

        kin_pot, nuc_pot, heisen_pot = self.one_body_potentials(rr, pp, p_num)
        pair_pot, pauli_pot = self.two_body_potentials(
            e_num, e_spin, x_coord, y_coord, z_coord, px, py, pz
        )

        return kin_pot, nuc_pot, heisen_pot, pair_pot, pauli_pot


# DIRECTORY TO SAVE THE RESULTS
directory = RESULTS_DIR
dif_p_e = DIFF_P_E

path = os.path.abspath(directory)  # Ensure absolute path
if not os.path.exists(path):
    print('Directory not found.')
    print('Create Directory...')
    try:
        os.makedirs(path)  # Use makedirs to create intermediate directories if needed
    except FileExistsError:
        print("Directory was already created by a different process!")
else:
    print('Directory exists!')

# INITIAL CONFIGURATION VALUES
p_num = P_NUM      # Number of protons
e_num = p_num + dif_p_e     # Number of electrons

# OPTIMIZATION PARAMETERS
iter_num = 0
converged = False

# Generate electron spins for arbitrary e_num
e_spin = np.array([1 if i % 2 == 0 else -1 for i in range(e_num)])

# Check if the proton number is valid
if 1 <= p_num <= len(ELEMENTS_LIST):
    element = ELEMENTS_LIST[p_num - 1]
else:
    raise ValueError(f"Invalid proton number {p_num}. It must be between 1 and {len(ELEMENTS_LIST)}.")

# Initialize the HamiltonianOptimizer
optimizer = HamiltonianOptimizer(ALPHA_H, ALPHA_P, XI_H, XI_P, xi_h_ryd=XI_H_RYD)

# Try to pick the optimized configuration of a previous element plus one rnd e
previous_element_filename = os.path.join(
    path, f'{p_num - 1:02d}_{ELEMENTS_LIST[p_num - 2]}_{e_num - 1:02d}e.csv'
)

if os.path.exists(previous_element_filename):
    with open(previous_element_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        if rows:
            last_row = rows[-1]
            try:
                prev_config = np.fromstring(
                    last_row['optimal_configuration'].strip('[]'), sep=' '
                )
                # Add one random electron to the configuration
                random_electron = optimizer.generate_initial_config(1)
                ini_config = np.concatenate((prev_config, random_electron))
            except Exception as e:
                print(f"Error loading previous configuration: {e}")
                print("Generating configuration from scratch.")
                ini_config = optimizer.generate_initial_config(e_num)
        else:
            print("Previous configuration file is empty. Generating from scratch.")
            ini_config = optimizer.generate_initial_config(e_num)
else:
    print("Previous configuration file not found. Generating from scratch.")
    ini_config = optimizer.generate_initial_config(e_num)

# Check if all variables in the initial configuration have positive values
positive = (
    np.all(ini_config[:e_num] > 0) and
    np.all(ini_config[3 * e_num:4 * e_num] > 0)
)

# Value Error if positive is False
if not positive:
    ini_config[:e_num] = np.abs(ini_config[:e_num])
    ini_config[3 * e_num:4 * e_num] = np.abs(
        ini_config[3 * e_num:4 * e_num]
    )

element = ELEMENTS_LIST[p_num - 1]
output_filename = os.path.join(
    path,
    f'{p_num:02d}_{element}_{e_num:02d}e_XI_H_{XI_H:.4f}_XI_P_{XI_P:.4f}_ALPHA_H_{ALPHA_H:.1f}_ALPHA_P_{ALPHA_P:.1f}.csv'
)

# Open the CSV file to write the results
with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        'p_num', 'e_num', 'message', 'time_taken', 'ground_state_energy',
        'kinetic_energy', 'nuclear_potential', 'heisenberg_potential',
        'pair_potential', 'pauli_potential', 'optimal_configuration'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # OPTIMIZATION
    start_time = time.time()
    while not converged or not positive:
        if iter_num >= MAXITER:
            message = f'NOT Converged after {MAXITER} iterations'
            break
        # if there is any negative r0 and p0 change to absolute value
        if not positive:
            ini_config[:e_num] = np.abs(ini_config[:e_num])
            ini_config[3 * e_num:4 * e_num] = np.abs(
                ini_config[3 * e_num:4 * e_num]
            )
            ini_config[:e_num] = np.where(
                ini_config[:e_num] > 50, 1, ini_config[:e_num]
            )
            ini_config[3 * e_num:4 * e_num] = np.where(
                ini_config[3 * e_num:4 * e_num] > 50, 1,
                ini_config[3 * e_num:4 * e_num]
            )
        result = fmin_bfgs(
            optimizer.hamiltonian, ini_config, args=(e_num, e_spin, p_num),
            gtol=GTOL, full_output=True, disp=False, retall=False
        )
        optimal_config, fopt, gopt, Bopt, func_calls, grad_calls, \
            warnflag = result
        converged = warnflag == 0
        ini_config = optimal_config

        positive = (
            np.all(ini_config[:e_num] > 0) and
            np.all(ini_config[3 * e_num:4 * e_num] > 0)
        )
        iter_num += 1

    if converged:
        message = f'Converged after {iter_num} iterations'

    end_time = time.time()
    time_taken = end_time - start_time

    # Extract individual components of the Hamiltonian
    kin_pot, nuc_pot, heisen_pot, pair_pot, pauli_pot = optimizer.\
        hamiltonian_components(optimal_config, e_num, e_spin, p_num)
    ground_state_energy = (kin_pot + nuc_pot + heisen_pot +
                           pair_pot + pauli_pot)

    writer.writerow({
        'p_num': p_num,
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

    print(f"Results for element {element} written to: {output_filename}")
