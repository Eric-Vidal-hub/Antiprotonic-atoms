import numpy as np
from scipy.integrate import solve_ivp
import os
import csv
import time


def hamiltonian_equations(t, state, M_STAR, ZZ, XI_H, ALPHA):
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
    for ii in range(num_electrons):
        ri = r_electrons[ii]
        pi = p_electrons[ii]
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
            if hei_arg_exp > 100 and ALPHA * (1 - hei_arg_exp) < -300:
                exp_hei = 0.0
            elif ALPHA * (1-hei_arg_exp) > 300:  # exp(300) overflows
                exp_hei = np.exp(300)
            else:
                exp_hei = np.exp(ALPHA * (1 - hei_arg_exp))
            v_hei = (XI_H**2 / (4 * ALPHA * ri_norm**2 * M_STAR)) * exp_hei

        # Time derivative of r_i: dr_i/dt = dH/dp_i
        dri_dt = pi * (1 - (1 / M_STAR) * uu * exp_hei)
        dr_dt_electrons_flat[3*ii:3*(ii+1)] = dri_dt

        # --- Forces for dp_i/dt = -dV/dr_i ---
        f_en = -ZZ / (ri_norm**3 + epsilon)

        f_ee_sum = np.zeros(3)
        for jj in range(num_electrons):
            if ii != jj:
                r_ij = ri - r_electrons[jj]
                norm_r_ij = np.linalg.norm(r_ij)
                f_ee_sum += np.abs(r_ij) / (norm_r_ij**3 + epsilon)

        f_heisenberg_p = 2 * (v_hei / (ri_norm**2 + epsilon)) * (1 + 2 * ALPHA * hei_arg_exp)

        dp_dt_electrons_flat[3*ii:3*(ii+1)] = pi * (f_en + f_heisenberg_p) + f_ee_sum

    # Assemble the full derivative vector in the correct order
    derivatives = np.concatenate([
        dr_dt_electrons_flat,
        dp_dt_electrons_flat
    ])

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
output_dir = os.path.join(os.path.dirname(__file__), 'He_atom_evo_output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# LOADING THE GS ATOM
# Read the CSV file using the csv module
helium_data = []
DIRECTORY_ATOM = os.path.join('HPC_results_gs_with_previous_z_as_ic', '02_He_02e.csv')
with open(DIRECTORY_ATOM, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        helium_data.append(row)

# Ensure the expected columns exist
required_col = ['p_num', 'e_num', 'optimal_configuration']
if not all(col in helium_data[0] for col in required_col):
    raise KeyError(f"Missing required columns in the CSV file: {required_col}")

# Expected config: r0_1, r0_i, theta_r_1, theta_r_i, phi_r_1, phi_r_i, p0_1,
# p0_i, theta_p_1, theta_p_i, phi_p_1, phi_p_i
# Convert to numpy arrays
for row in helium_data:
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


# ATOM RANDOM ORIENTATION
# Randomize the angles
# theta_rnd = np.pi * np.random.random()
# phi_rnd = 2 * np.pi * np.random.random()
theta_rnd = 0
phi_rnd = 0
# Convert to Cartesian coordinates
rx, ry, rz, px, py, pz = convert_to_cartesian(
    r0, theta_r + theta_rnd, phi_r + phi_rnd,
    p0, theta_p + theta_rnd, phi_p + phi_rnd)

# INITIAL STATE VECTOR
# coordinates per particle: re1(3), re2(3),
# momenta per particle: pe1(3), pe2(3),
y0 = np.concatenate(
    [np.column_stack((rx, ry, rz)).flatten(),
        np.column_stack((px, py, pz)).flatten()]
    )

# Time span for integration
t_span = [0, 150]
# t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Parameters
M_PBAR = 1836.152672  # antiproton mass (a.u.)
M_STAR = 1 / (1 + (1 / (2 * ZZ * M_PBAR)))  # Reduced mass (a.u.)
ALPHA = 5.0
XI_H = 1.0
XI_H /= (1 + 1 / (2 * ALPHA))**0.5

print(f"Reduced mass M_STAR: {M_STAR}, Heisenberg parameter XI_H: {XI_H}")
# Number of electrons


# INTEGRATION
start_time = time.time()
sol = solve_ivp(
    hamiltonian_equations, t_span, y0, args=(M_STAR, ZZ, XI_H, ALPHA),
    method='DOP853', rtol=1e-6, atol=1e-8)
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

# Save t_arr and y_arr to a .npz file
np.savez(os.path.join(output_dir, 'trajectory_data.npz'),
         t_arr=t_arr, y_arr=y_arr, e_num=e_num)
