import numpy as np
from scipy.integrate import solve_ivp
import os
import csv
import time
import glob
from v3_multi_constants import (M_PBAR, ALPHA_H, XI_H, ALPHA_P, XI_P, XI_H_RYD, T_MAX,
                                N_STEP, DIRECTORY_ATOM, RESULTS_DIR)


start_time = time.time()
# %% FUNCTIONS
def hamiltonian_equations(t, state, MU, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, E_SPIN):
    """
    Generalized force computation for an arbitrary number of electrons.
    Given state vector y = [r1(3), r2(3),
                            p1(3), p2(3)].
    It returns derivatives dy/dt according to Hamilton's equations.
    """
    # Unpack state vector into shaped arrays for easier handling
    num_electrons = len(state) // 6     # Calculate the number of electrons
    r_e_flat = state[:3 * num_electrons]
    p_e_flat = state[3 * num_electrons:]

    r_electrons = r_e_flat.reshape((num_electrons, 3))
    p_electrons = p_e_flat.reshape((num_electrons, 3))

    dr_dt_electrons_flat = np.zeros(3 * num_electrons)
    dp_dt_electrons_flat = np.zeros(3 * num_electrons)

    # Small constant to prevent division by zero or extremely small norms
    epsilon = 1e-18

    # --- Forces and derivatives for Electrons ---
    for kk in range(num_electrons):
        ri = r_electrons[kk]
        pi = p_electrons[kk]
        ri_norm = np.linalg.norm(ri)
        pi_norm = np.linalg.norm(pi)

        # Heisenberg core contribution (ensure this part is numerically stable)
        v_hei = 0.0
        # Guard against issues if ri_norm or pi_norm is zero, or XI_H is zero
        if ri_norm > epsilon and pi_norm > epsilon and np.abs(XI_H) > epsilon:
            # The exponent term can become very large negatively or positively.
            # np.exp can overflow or underflow.
            uu = (ri_norm * pi_norm / XI_H)**2
            hei_arg_exp = uu**2
            # Cap the argument to prevent overflows or underflows
            # If hei_arg_exp is very small, exp_hei ~ exp(ALPHA)
            if hei_arg_exp > 100 and ALPHA_H * (1 - hei_arg_exp) < -300:
                exp_hei = 0.0
            elif ALPHA_H * (1-hei_arg_exp) > 300:  # exp(300) overflows
                exp_hei = np.exp(300)
            else:
                exp_hei = np.exp(ALPHA_H * (1 - hei_arg_exp))
            v_hei = (XI_H**2 / (4 * ALPHA_H * ri_norm**2 * MU)) * exp_hei

        # Pauli exclusion principle contribution
        v_pauli_rr = np.zeros(3)
        for ii in range(num_electrons):
            for mm in range(ii + 1, num_electrons):
                if E_SPIN[ii] == E_SPIN[mm]:
                    if kk == ii or kk == mm:
                        r_im = (2 * ri - r_electrons[mm] - r_electrons[ii])
                        p_im = (2 * pi - p_electrons[mm] - p_electrons[ii]) / 2
                        r_im_norm = np.linalg.norm(r_im)
                        p_im_norm = np.linalg.norm(p_im)
                        uu_p = (r_im_norm * p_im_norm / XI_P)**2
                        hei_arg_exp_p = uu_p**2
                        # Cap the argument to prevent overflows or underflows
                        if hei_arg_exp_p > 100 and ALPHA_P * (1 - hei_arg_exp_p) < -300:
                            exp_pauli = 0.0
                        elif ALPHA_P * (1 - hei_arg_exp_p) > 300:  # exp(300) overflows
                            exp_pauli = np.exp(300)
                        else:
                            exp_pauli = np.exp(ALPHA_P * (1 - hei_arg_exp_p))

                        v_pauli_rr -= p_im * uu_p * exp_pauli


        # Time derivative of r_i: dr_i/dt = dH/dp_i
        dri_dt = pi * (1 - (1 / MU) * uu * exp_hei) + v_pauli_rr
        dr_dt_electrons_flat[3*kk:3*(kk+1)] = dri_dt

        # --- Forces for dp_i/dt = -dV/dr_i ---
        f_en = -ZZ / (ri_norm**3 + epsilon)

        r_ij = ri - r_electrons
        norm_r_ij = np.linalg.norm(r_ij, axis=1) + epsilon
        mask = np.arange(num_electrons) != kk
        f_ee_sum = np.sum(r_ij[mask] / norm_r_ij[mask][:, None]**3, axis=0)

        # Heisenberg potential force
        f_heisenberg_p = 2 * (v_hei / (ri_norm**2 + epsilon)) * (1 + 2 * ALPHA_H * hei_arg_exp)

        # Pauli exclusion principle contribution
        v_pauli_pp = np.zeros(3)
        for ii in range(num_electrons):
            for mm in range(ii + 1, num_electrons):
                if kk == ii or kk == mm:
                    r_im = (2 * ri - r_electrons[mm] - r_electrons[ii])
                    r_im_norm = np.linalg.norm(r_im)
                    # Ensure r_im_norm is not zero to avoid division by zero
                    factor = (r_im / (r_im_norm**2 + epsilon))
                    if E_SPIN[ii] == E_SPIN[mm]:
                        p_im = (2 * pi - p_electrons[mm] - p_electrons[ii]) / 2
                        p_im_norm = np.linalg.norm(p_im)
                        uu_p = (r_im_norm * p_im_norm / XI_P)**2
                        hei_arg_exp_p = uu_p**2
                        # Cap the argument to prevent overflows or underflows
                        if hei_arg_exp_p > 100 and ALPHA_P * (1 - hei_arg_exp_p) < -300:
                            exp_pauli = 0.0
                        elif ALPHA_P * (1 - hei_arg_exp_p) > 300:
                            exp_pauli = np.exp(300)
                        else:
                            exp_pauli = np.exp(ALPHA_P * (1 - hei_arg_exp_p))
                        v_pauli_term = (XI_P**2 / (2 * ALPHA_P * r_im_norm**2)) * exp_pauli
                        v_pauli_pp += factor * 2 * v_pauli_term * (1 + 2 * ALPHA_P * hei_arg_exp_p)

        # T-DER of p_i: dp_i/dt = -dH/dr_i
        dp_dt_electrons_flat[3*kk:3*(kk+1)] = ri * (f_en + f_heisenberg_p) + f_ee_sum + v_pauli_pp

    # Assemble the full derivative vector in the correct order
    derivatives = np.concatenate([
        dr_dt_electrons_flat,
        dp_dt_electrons_flat
    ])
    # print(f"t: {t}, derivatives: {derivatives}")

    return derivatives


def convert_to_cartesian(rr, theta, phi, pp, theta_p, phi_p):
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


# %% SIMULATION
# Output directory (optional, for saving)
output_dir = os.path.join(os.path.dirname(__file__), RESULTS_DIR)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %% LOADING THE GS ATOM
# Find any file in GS_fitting/ that starts with '02_He_02e'
atom_files = glob.glob(os.path.join('GS_fitting', '02_He_02e*'))
if not atom_files:
    raise FileNotFoundError("No file starting with '02_He_02e' found in GS_fitting/")
DIRECTORY_ATOM = atom_files[0]

# Read the CSV file using the csv module
MULTI_DATA = []
with open(DIRECTORY_ATOM, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        MULTI_DATA.append(row)

# Ensure the expected columns exist
required_col = ['p_num', 'e_num', 'optimal_configuration']
if not all(col in MULTI_DATA[0] for col in required_col):
    raise KeyError(f"Missing required columns in the CSV file: {required_col}")

# Expected config: r0_1, r0_i, theta_r_1, theta_r_i, phi_r_1, phi_r_i, p0_1,
# p0_i, theta_p_1, theta_p_i, phi_p_1, phi_p_i
# Convert to numpy arrays
for row in MULTI_DATA:
    p_num = int(row['p_num'])
    e_num = int(row['e_num'])
    optimal_config = np.fromstring(row['optimal_configuration'].strip('[]'),
                                   sep=' ')
    r0 = optimal_config[:e_num]
    theta_r = optimal_config[e_num:2*e_num]
    phi_r = optimal_config[2*e_num:3*e_num]
    p0 = optimal_config[3*e_num:4*e_num]
    theta_p = optimal_config[4*e_num:5*e_num]
    phi_p = optimal_config[5*e_num:6*e_num]

# Atomic number Z, number of protons
ZZ = p_num
print(f"Number of protons (Z): {ZZ}, Number of electrons: {e_num}")

# %% ATOM RANDOM ORIENTATION
# Randomize the angles
theta_rnd = np.pi * np.random.random()
phi_rnd = 2 * np.pi * np.random.random()

# Convert to Cartesian coordinates
rx, ry, rz, px, py, pz = convert_to_cartesian(
    r0, theta_r + theta_rnd, phi_r + phi_rnd,
    p0, theta_p + theta_rnd, phi_p + phi_rnd)

# %% INITIAL STATE VECTOR
# coordinates per particle: re1(3), re2(3),
# momenta per particle: pe1(3), pe2(3),
y0 = np.concatenate(
    [np.column_stack((rx, ry, rz)).flatten(),
        np.column_stack((px, py, pz)).flatten()]
    )

# Time span for integration
t_span = [0, T_MAX]
t_eval = np.linspace(t_span[0], t_span[1], N_STEP)

# Parameters
MU = 1 / (1 + (1 / (2 * ZZ * M_PBAR)))  # Reduced mass (a.u.)

print(f"Reduced mass E- ATOM MU: {MU}, Heisenberg parameter XI_H: {XI_H}")

# electrons spin, 0 for odd, 1 for even
e_spin = np.zeros(e_num, dtype=int)
# Set the spin of electrons based on their index
for i in range(e_num):
    e_spin[i] = i % 2  # 0 for odd, 1 for even
# Print the electron spins
print(f"Electron spins: {e_spin}")
# Ensure the initial state vector is a 1D array
y0 = np.array(y0).flatten()
# Print the initial state vector
print(f"Initial state vector (y0): {y0}")
# Print the initial position and momentum of electrons
print(f"Initial position of electrons: {y0[:3 * e_num]}")
print(f"Initial momentum of electrons: {y0[3 * e_num:]}")


# %% INTEGRATION
start_time = time.time()
sol = solve_ivp(
    hamiltonian_equations, t_span, y0, args=(MU, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, e_spin),
    t_eval=t_eval, dense_output=True, method='DOP853', rtol=1e-6, atol=1e-8)
end_time = time.time()

print(f"Simulation time: {end_time - start_time:.2f} seconds")
# SOLUTION
yf = sol.y[:, -1]

# Extract initial position and momentum of electrons
rf_e = yf[:3 * e_num]
pf_e = yf[3 * e_num:]
# Ensure the vectors are 1D
rf_e = np.array(rf_e).flatten()
pf_e = np.array(pf_e).flatten()

print(f"Final position of electrons: {rf_e}")
print(f"Final momentum of electrons: {pf_e}")

# Get the time array and solution array
t_arr = sol.t
y_arr = sol.y  # shape: (6*e_num, len(t_arr))

# --- Save the same data to a CSV file ---
csv_file = os.path.join(output_dir, 'trajectory_data.csv')
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write scalar parameters
    writer.writerow(['e_num', e_num])
    writer.writerow(['MU', MU])
    writer.writerow(['ZZ', ZZ])
    writer.writerow(['XI_H', XI_H])
    writer.writerow(['ALPHA_H', ALPHA_H])
    writer.writerow(['XI_P', XI_P])
    writer.writerow(['ALPHA_P', ALPHA_P])
    writer.writerow(['T_MAX', T_MAX])
    writer.writerow([])  # Blank line

    # Write t_arr
    writer.writerow(['t_arr'])
    writer.writerow(t_arr)
    writer.writerow([])

    # Write y_arr (each row: variable index, then values over time)
    writer.writerow(['y_arr (each row: variable index, then values over time)'])
    for i, row in enumerate(y_arr):
        writer.writerow([f'y[{i}]'] + list(row))

# --- Save time, modulus of position and momentum for each electron in a new CSV for plotting ---
csv_plot_file = os.path.join(output_dir, 'trajectory_data_plot.csv')
with open(csv_plot_file, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = ['time'] + [f'r_e{i+1}' for i in range(e_num)] + [f'p_e{i+1}' for i in range(e_num)]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for idx, t in enumerate(t_arr):
        row = {'time': t}
        # Position modulus for each electron
        for i in range(e_num):
            r_vec = y_arr[3*i:3*(i+1), idx]
            row[f'r_e{i+1}'] = np.linalg.norm(r_vec)
        # Momentum modulus for each electron
        for i in range(e_num):
            p_vec = y_arr[3*e_num + 3*i:3*e_num + 3*(i+1), idx]
            row[f'p_e{i+1}'] = np.linalg.norm(p_vec)
        writer.writerow(row)
