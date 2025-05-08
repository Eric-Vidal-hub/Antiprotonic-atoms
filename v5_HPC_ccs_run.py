"""
Semiclassical Monte Carlo simulation of He + antiproton capture
using the FMD method (Beck et al., Phys. Rev. A 48, 2779 (1993)).

Produces:
 - Capture cross sections vs. initial energy (total, single, double).
    Outputs cross_sections.csv
 - Example trajectory radii vs. time.
    Outputs trajectory_example.csv
 - Initial (E,L) distribution for capture events.
    Outputs initial_states.csv
 - Final (E,L) distribution after capture.
    Outputs final_states.csv

Reads:
 - csv_file: Ground-state system to extract coordinates and momenta

Dependencies:
    numpy, pandas, scipy
"""
import sys
import os
import numpy as np
import csv
from scipy.integrate import solve_ivp


# %% FUNCTIONS
def compute_forces(t, state):
    """
    Generalized force computation for an arbitrary number of electrons.
    Given state vector y = [r1(3), r2(3), ...,
                            p1(3), p2(3), ..., r_pbar(3), p_pbar(3)].
    It returns derivatives dy/dt according to Hamilton's equations.
    """
    # Unpack state vector into shaped arrays for easier handling
    num_electrons = (len(state) - 6) // 6  # Calculate the number of electrons
    r_e_flat = state[:3 * num_electrons]
    p_e_flat = state[3 * num_electrons : 6 * num_electrons]
    r_pbar = state[-6:-3]
    p_pbar = state[-3:]

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
            hei_arg_exp = (ri_norm * pi_norm / XI_H)**4
            # Cap the argument to prevent overflow if needed, though for large positive hei_arg_exp,
            # (1 - hei_arg_exp) becomes large negative, leading to exp_hei -> 0.
            # If hei_arg_exp is very small, (1 - hei_arg_exp) ~ 1, exp_hei ~ exp(ALPHA)
            if hei_arg_exp > 100 and ALPHA * (1 - hei_arg_exp) < -700: # exp(-700) is ~0
                exp_hei = 0.0
            elif ALPHA * (1-hei_arg_exp) > 700: # exp(700) overflows
                exp_hei = np.exp(700) # Cap to a very large number
            else:
                exp_hei = np.exp(ALPHA * (1 - hei_arg_exp))
            v_hei = ri_norm * pi_norm**3 * exp_hei / (XI_H**3) # Removed epsilon from XI_H**3, assuming XI_H is well-defined

        # Time derivative of r_i: dr_i/dt = dH/dp_i
        # Assuming H contains p_i^2/2m and the Heisenberg term contributes -v_hei * r_i to dH/dp_i
        dri_dt = pi - v_hei * ri  # Check signs if m_e is not 1
        dr_dt_electrons_flat[3 * ii : 3*(ii+1)] = dri_dt

        # --- Forces for dp_i/dt = -dV/dr_i ---
        # 1. Electron-nucleus interaction (attractive potential V = -ZZ/||ri||)
        # Force F_en = -grad(V) = -ZZ * ri / ||ri||^3
        f_en = -ZZ * ri / (ri_norm**3 + epsilon)

        # 2. Electron-electron interaction (repulsive potential V_ij = 1/||ri-rj||)
        # Force on i from j: F_ij = -grad_i(V_ij) = (ri - rj) / ||ri - rj||^3
        f_ee_sum = np.zeros(3)
        for jj in range(num_electrons):
            if ii == jj:
                continue
            r_ij = ri - r_electrons[jj]
            norm_r_ij = np.linalg.norm(r_ij)
            f_ee_sum += r_ij / (norm_r_ij**3 + epsilon)
        
        # 3. Electron-antiproton interaction (attractive potential V_ipbar = -1/||ri-rpbar||)
        # Force on i from pbar: F_ipbar = -grad_i(V_ipbar) = -(ri - r_pbar) / ||ri - r_pbar||^3
        r_ipbar = ri - r_pbar
        norm_r_ipbar = np.linalg.norm(r_ipbar)
        f_epbar = -r_ipbar / (norm_r_ipbar**3 + epsilon)
        
        # Heisenberg force component on momentum: -dV_H/dr_i
        # Assuming the Heisenberg term contributes +v_hei * pi to -dV_H/dr_i
        f_heisenberg_p = v_hei * pi

        dp_dt_electrons_flat[3*ii : 3*(ii+1)] = f_en + f_ee_sum + f_epbar + f_heisenberg_p

    # --- Forces and derivatives for Antiproton ---
    # Time derivative of R_pbar: dR_pbar/dt = dH/dP_pbar = P_pbar / M_STAR
    dR_pbar_dt = p_pbar / M_STAR

    # Forces for dP_pbar/dt = -dV/dR_pbar
    # 1. Antiproton-nucleus interaction (attractive V = -ZZ/||R_pbar||)
    # Force F_pbar_nuc = -grad(V) = -ZZ * r_pbar / ||R_pbar||^3
    norm_r_pbar = np.linalg.norm(r_pbar)
    f_pbar_nuc = -ZZ * r_pbar / (norm_r_pbar**3 + epsilon)

    # 2. Antiproton-electron interaction (attractive V_pbar_e = sum_i -1/||R_pbar-ri||)
    # Force on pbar from electron i: F_pbar_ei = -grad_Rpbar(V_pbar_ei)
    # F_pbar_ei = -(r_pbar - ri) / ||R_pbar - ri||^3 = (ri - r_pbar) / ||ri - r_pbar||^3
    f_pbar_e_sum = np.zeros(3)
    for ii in range(num_electrons):
        # r_ipbar was ri - r_pbar. So (r_electrons[i] - r_pbar) is correct here.
        # norm_r_ipbar was norm(ri - r_pbar)
        # Need to recalculate for each electron interaction if not already available
        vec_ei_to_pbar = r_pbar - r_electrons[ii] # Vector from electron i to antiproton
        norm_vec_ei_to_pbar = np.linalg.norm(vec_ei_to_pbar)
        # Force on antiproton from electron i is attractive, pointing towards electron i.
        f_pbar_e_sum += -vec_ei_to_pbar / (norm_vec_ei_to_pbar**3 + epsilon) # Force on pbar is towards electron

    dP_pbar_dt = f_pbar_nuc + f_pbar_e_sum

    # Assemble the full derivative vector in the correct order
    derivatives = np.concatenate([
        dr_dt_electrons_flat,
        dp_dt_electrons_flat,
        dR_pbar_dt,
        dP_pbar_dt
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
# DIRECTORY TO SAVE THE RESULTS
DIRECTORY_PBAR = sys.argv[1]
ID = int(sys.argv[2])

path = DIRECTORY_PBAR
if not os.path.exists(path):
    print('Directory not found.')
    print('Create Directory...')
    try:
        os.mkdir(path)
    except FileExistsError:
        print("Directory was already created by a different process!")
else:
    print('Directory exists!')

# PARAMETERS (a.u.)
# initial antiproton energies (a.u.)
MIN_E = 0.1
MAX_E = 3.0
N_STEP = 16
ENERGIES = np.linspace(MIN_E, MAX_E, N_STEP)  # initial energies (a.u.)
N_TRAJ = 100       # trajectories per energy
T_MAX = 25000.0     # max time (a.u.)
# THRESH_1 = 2.3      # energy threshold for stepping b_max
# THRESH_2 = 1.2
# B1, B2, B3 = 1.0, 2.0, 3.0  # impact parameters (a.u.)
BMAX = 3.0      # impact parameter (a.u.)
XPBAR = 50.0     # initial distance of antiproton (a.u.)
# (away from nucleus)

# Physical constants
M_PBAR = 1836.152672  # antiproton mass (a.u.)

# FMD parameters
ALPHA = 5       # Hardness parameter
XI_H = 1.000    # Tuning parameter for the Heisenberg potential
XI_P = 2.767    # Tuning parameter for the Pauli potential
# Scaling parameters according to alpha
XI_H /= np.sqrt(1 + 1 / (2 * ALPHA))
XI_P /= np.sqrt(1 + 1 / (2 * ALPHA))

# Define the directory and file name
DIRECTORY_ATOM = '/scratch/vym17xaj/HPC_results_gs_with_alpha_modifying/'
FILE_NAME = '02_He_02e.csv'

# Read the CSV file using the csv module
helium_data = []
with open(DIRECTORY_ATOM + FILE_NAME, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        helium_data.append(row)

# Ensure the expected columns exist
required_col = ['p_num', 'e_num', 'optimal_configuration']
if not all(col in helium_data[0] for col in required_col):
    raise KeyError(f"Missing required columns in the CSV file: {required_col}")

# Process the data
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
M_STAR = M_PBAR / (1 + (1 / (2 * ZZ)))  # Reduced mass (a.u.)

# STORAGE PARAMETERS
cross_data = []
initial_states = []
final_states = []
traj_saved = False

# DYNAMIC SIMULATION
E0 = ENERGIES[ID]
# Initialize counters
n_double = 0
n_single = 0

# # Determine b_max based on initial energy
# if E0 > THRESH_1:
#     BMAX = B1
# elif E0 > THRESH_2:
#     BMAX = B2
# else:
#     BMAX = B3

for ii in range(N_TRAJ):
    # ATOM RANDOM ORIENTATION
    # Randomize the angles
    theta_rnd = np.pi * np.random.random()
    phi_rnd = 2 * np.pi * np.random.random()
    # Convert to Cartesian coordinates
    rx, ry, rz, px, py, pz = convert_to_cartesian(
        r0, theta_r + theta_rnd, phi_r + phi_rnd,
        p0, theta_p + theta_rnd, phi_p + phi_rnd)

    # ANTIPROTON INITIALIZATION
    # Random impact parameter uniform in area
    # b = np.sqrt(np.random.random() * BMAX / np.pi)
    b = np.sqrt(np.random.random()) * BMAX
    angle = 2 * np.pi * np.random.random()
    # Launch antiproton far away along +x with offset in y
    r0_pbar = np.array([-XPBAR, b * np.cos(angle), b * np.sin(angle)])
    # initial momentum vector
    p0_pbar = np.array([np.sqrt(2 * E0 * M_STAR), 0.0, 0.0])
    # Record initial and final (E,L)
    L_init = np.linalg.norm(np.cross(r0_pbar, p0_pbar))

    # INITIAL STATE VECTOR
    # coordinates per particle: re1(3), re2(3), ..., rpbar(3),
    # momenta per particle: pe1(3), pe2(3), ..., ppbar(3)
    y0 = np.concatenate(
        [np.column_stack((rx, ry, rz)).flatten(),
            np.column_stack((px, py, pz)).flatten(), r0_pbar, p0_pbar]
        )

    # INTEGRATION
    sol = solve_ivp(
        compute_forces,
        (0.0, T_MAX), y0,
        method='RK45', rtol=1e-6, atol=1e-9
    )

    # SOLUTION
    yf = sol.y[:, -1]

    # Extract initial position and momentum of electrons
    rf_e = yf[:3 * e_num]
    pf_e = yf[3 * e_num:-6]
    # Ensure the vectors are 1D
    rf_e = np.array(rf_e).flatten()
    pf_e = np.array(pf_e).flatten()

    # Extract final position and momentum of the antiproton
    rf_pbar = yf[-6:-3]
    pf_pbar = yf[-3:]
    # Ensure the vectors are 1D
    rf_pbar = np.array(rf_pbar).flatten()
    pf_pbar = np.array(pf_pbar).flatten()

    # COMPUTE PBAR FINAL ENERGY AND ANGULAR MOMENTUM
    # Final ang mom of the antiproton
    Lf_pbar = np.linalg.norm(np.cross(rf_pbar, pf_pbar))

    # Final energy of the antiproton
    # One body potentials
    norm_rf_pbar = np.linalg.norm(rf_pbar)
    norm_pf_pbar = np.linalg.norm(pf_pbar)
    kin_pbar = norm_pf_pbar**2 / (2 * M_STAR)
    nuc_pbar = -ZZ / np.linalg.norm(rf_pbar)
    # heisenberg_pbar = (
    #     (XI_H / (4 * ALPHA * norm_rf_pbar**2)) *
    #     np.exp(ALPHA * (1 - (norm_rf_pbar * norm_pf_pbar / XI_H)**4))
    # )
    # Two body potentials
    pair_pot = 0.0  # Coulomb potential between electrons
    # pauli_pot = 0.0  # Pauli exclusion constraint potential
    if e_num > 1:
        for ii in range(e_num):
            # Electron position and momentum
            rf_ei = rf_e[3 * ii:3 * (ii + 1)]
            delta_r = np.linalg.norm(rf_pbar - rf_ei)
            # Coulomb potential for electron pairs
            pair_pot += 1.0 / delta_r
            # For identical electrons
            # if e_spin[i] == pbar_spin:
            #     pf_ei = pf_e[3 * i:3 * (i + 1)]
            #     delta_p = np.linalg.norm(pf_pbar - pf_ei)
            #     pauli_pot += (
            #         self.xi_p ** 2 / (4 * self.alpha * delta_r ** 2)
            #     ) * np.exp(
            #         self.alpha * (1 - (delta_r * delta_p
            #                             / self.xi_p) ** 4)
            #     )
    Ef_pbar = kin_pbar + nuc_pbar + pair_pot
    # + heisenberg_pbar + pauli_pot

    # COMPUTE ELECTRONS FINAL ENERGY
    bound_electrons = []
    E_electrons = []
    for ii in range(e_num):
        # Final position and momentum of the i-th electron
        rf_ei = rf_e[3 * ii:3 * (ii + 1)]
        pf_ei = pf_e[3 * ii:3 * (ii + 1)]

        # Final energy of the i-th electron
        norm_rf_ei = np.linalg.norm(rf_ei)
        norm_pf_ei = np.linalg.norm(pf_ei)
        kin_ei = norm_pf_ei**2 / 2.0
        nuc_ei = -ZZ / norm_rf_ei
        heisenberg_ei = (
            (XI_H / (4 * ALPHA * norm_rf_ei**2)) *
            np.exp(ALPHA * (1 - (norm_rf_ei *
                                 norm_pf_ei / XI_H)**4))
        )
        # Two body potentials
        pair_pot = 0.0
        # pauli_pot = 0.0  # Pauli exclusion constraint potential
        if e_num > 1:
            for j in range(e_num):
                if ii != j:
                    rf_ej = rf_e[3 * j:3 * (j + 1)]
                    delta_r = np.linalg.norm(rf_ei - rf_ej)
                    pair_pot += 1.0 / delta_r
                    # For identical electrons
                    # if e_spin[i] == e_spin[j]:
                    #     pf_ej = pf_e[3 * j:3 * (j + 1)]
                    #     delta_p = np.linalg.norm(pf_ei - pf_ej)
                    #     pauli_pot += (
                    #         self.xi_p ** 2
                    #         / (4 * self.alpha * delta_r ** 2)
                    #     ) * np.exp(
                    #         self.alpha * (1 - (delta_r * delta_p
                    #                             / self.xi_p) ** 4)
                    #     )
        Ef_ei = kin_ei + nuc_ei + heisenberg_ei + pair_pot  # + pauli_pot
        E_electrons.append(Ef_ei)

        # CAPTURE CLASSIFICATION
        bound_electrons.append(Ef_ei < 0)

    bound_p = Ef_pbar < 0     # Antiproton bound if Ef < 0

    # Classify capture
    if bound_p and any(bound_electrons):
        cap_type = 'pbar_electrons_' + str(len(bound_electrons))
        n_single += 1
    elif bound_p:
        cap_type = 'double'
        n_double += 1
    else:
        cap_type = 'none'

    # Save the first capture trajectory
    if ((cap_type != 'none') and (not traj_saved)):
        times = sol.t

        # Extract radial distance of the antiproton
        r_p = np.linalg.norm(sol.y[-6:-3, :], axis=0)

        # Extract radial distances of the electrons
        electron_radial_distances = {}
        for ii in range(e_num):
            r_e = sol.y[3 * ii:3 * (ii + 1), :]  # Positions of the i-th electron
            r_e_modulus = np.linalg.norm(r_e, axis=0)  # Radial distance
            electron_radial_distances[f'r_e{ii+1}'] = r_e_modulus

        # Combine all trajectory data into a single dictionary
        trajectory_data = []
        for idx, t in enumerate(times):
            row = {'time': t, 'r_p': r_p[idx]}
            for key, value in electron_radial_distances.items():
                row[key] = value[idx]
            trajectory_data.append(row)

        # Save the trajectory data to a CSV file
        trajectory_file = os.path.join(DIRECTORY_PBAR,
                                       'trajectory_example_{ii}_E0_{E0}.csv')
        with open(trajectory_file, mode='w', newline='', encoding='utf-8') as file:
            fieldnames = ['time', 'r_p'] + [f'r_e{i+1}' for i in range(e_num)]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(trajectory_data)

        traj_saved = True

    initial_states.append((E0, L_init, cap_type))
    final_states.append((Ef_pbar, E_electrons, Lf_pbar, cap_type))

# COMPUTE CROSS SECTIONS
sigma_tot = np.pi * BMAX**2 * (n_double + n_single) / N_TRAJ
sigma_sng = np.pi * BMAX**2 * n_single / N_TRAJ
sigma_dbl = np.pi * BMAX**2 * n_double / N_TRAJ
cross_data.append((E0, sigma_tot, sigma_sng, sigma_dbl))

# Save the combined data to a single CSV file
output_file = os.path.join(DIRECTORY_PBAR, f'ini_e_{E0}_R0_{XPBAR}.csv')
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = [
        'Energy', 'Sigma_total', 'Sigma_single', 'Sigma_double',
        'E_initial', 'L_initial', 'Type_initial',
        'E_final', 'E_electrons', 'L_final', 'Type_final'
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for ii, (cross, initial, final) in enumerate(zip(cross_data, initial_states, final_states)):
        writer.writerow({
            'Energy': cross[0],
            'Sigma_total': cross[1],
            'Sigma_single': cross[2],
            'Sigma_double': cross[3],
            'E_initial': initial[0],
            'L_initial': initial[1],
            'Type_initial': initial[2],
            'E_final': final[0],
            'E_electrons': final[1],
            'L_final': final[2],
            'Type_final': final[3],
        })
