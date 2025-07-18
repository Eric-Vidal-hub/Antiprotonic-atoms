"""
Semiclassical Monte Carlo simulation of He + antiproton capture
using the FMD method (Beck et al., Phys. Rev. A 48, 2779 (1993)).

Produces:
 - Capture cross sections vs. initial energy (total, partial, full).
    Outputs cross_sections.csv
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
from v0_trajectory_constants_HPC import (M_PBAR, ALPHA_H, XI_H, ALPHA_P, XI_P,
                                         MIN_E, MAX_E, E_STEP, T_STEP, T_MAX,
                                         BMAX_0, XPBAR, DIRECTORY_ATOM, B1,
                                         B2, B3, AUTO_BMAX, THRESH_1, THRESH_2,
                                         N_CHECK_MAX, T_MEAN)
import time


start_time = time.time()
# %% FUNCTIONS
def compute_forces(t, state, M_STAR, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, E_SPIN):
    """
    Generalized force computation for an arbitrary number of electrons.
    Given state vector y = [r1(3), r2(3), ...,
                            p1(3), p2(3), ..., r_pbar(3), p_pbar(3)].
    It returns derivatives dy/dt according to Hamilton's equations.
    """
    # Unpack state vector into shaped arrays for easier handling
    num_electrons = (len(state) - 6) // 6  # Calculate the number of electrons
    r_e_flat = state[:3 * num_electrons]
    p_e_flat = state[3 * num_electrons:6 * num_electrons]
    r_pbar = state[-6:-3]
    p_pbar = state[-3:]

    r_electrons = r_e_flat.reshape((num_electrons, 3))
    p_electrons = p_e_flat.reshape((num_electrons, 3))

    dr_dt_electrons_flat = np.zeros(3 * num_electrons)
    dp_dt_electrons_flat = np.zeros(3 * num_electrons)

    # Small constant to prevent division by zero or extremely small norms
    epsilon = 1e-18

    # Initialize force of all electrons on the antiproton
    f_pbar_e_sum = np.zeros(3)

    # --- FORCES AND DERIVATIVES FOR ELECTRONS ---
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
            v_hei = (XI_H**2 / (4 * ALPHA_H * ri_norm**2)) * exp_hei

        # --- PAULI EXCLUSION PRINCIPLE CONTRIBUTION ---
        v_pauli_rr = np.zeros(3)
        v_pauli_pp = np.zeros(3)
        for ii in range(num_electrons):
            for mm in range(ii + 1, num_electrons):
                if E_SPIN[ii] == E_SPIN[mm]:
                    # For dr/dt (Pauli force in momentum space)
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
                        elif ALPHA_P * (1 - hei_arg_exp_p) > 300:
                            exp_pauli = np.exp(300)
                        else:
                            exp_pauli = np.exp(ALPHA_P * (1 - hei_arg_exp_p))
                        v_pauli_rr -= p_im * uu_p * exp_pauli

                    # For dp/dt (Pauli force in position space)
                    if kk == ii or kk == mm:
                        r_im = (2 * ri - r_electrons[mm] - r_electrons[ii])
                        r_im_norm = np.linalg.norm(r_im)
                        factor = (r_im / (r_im_norm**2 + epsilon))
                        p_im = (2 * pi - p_electrons[mm] - p_electrons[ii]) / 2
                        p_im_norm = np.linalg.norm(p_im)
                        uu_p = (r_im_norm * p_im_norm / XI_P)**2
                        hei_arg_exp_p = uu_p**2
                        if hei_arg_exp_p > 100 and ALPHA_P * (1 - hei_arg_exp_p) < -300:
                            exp_pauli = 0.0
                        elif ALPHA_P * (1 - hei_arg_exp_p) > 300:
                            exp_pauli = np.exp(300)
                        else:
                            exp_pauli = np.exp(ALPHA_P * (1 - hei_arg_exp_p))
                        v_pauli_term = (XI_P**2 / (2 * ALPHA_P * r_im_norm**2)) * exp_pauli
                        v_pauli_pp += factor * 2 * v_pauli_term * (1 + 2 * ALPHA_P * hei_arg_exp_p)

        # T-DER of r_i: dr_i/dt = dH/dp_i
        dri_dt = pi * (1 - uu * exp_hei) + v_pauli_rr
        dr_dt_electrons_flat[3*kk:3*(kk+1)] = dri_dt

        # --- FORCES ON E MOMENTA ---
        # Force on electron i from the nucleus
        f_en = -ZZ / (ri_norm**3 + epsilon)

        # Force on electron i from antiproton
        f_epbar = (ri - r_pbar) / (np.linalg.norm(r_pbar - ri)**3 + epsilon)

        # Force on electron i from other electrons
        f_ee_sum = np.zeros(3)
        for ii in range(num_electrons):
            for mm in range(ii + 1, num_electrons):
                if kk == ii:
                    r_ij = ri - r_electrons[mm]
                    norm_r_ij = np.linalg.norm(r_ij)
                    f_ee_sum += r_ij / (norm_r_ij**3 + epsilon)
                elif kk == mm:
                    r_ij = ri - r_electrons[ii]
                    norm_r_ij = np.linalg.norm(r_ij)
                    f_ee_sum += r_ij / (norm_r_ij**3 + epsilon)

        # Heisenberg core contribution for electron i
        f_heisenberg_p = (2 * v_hei / (ri_norm**2 + epsilon)) * (
            1 + 2 * ALPHA_H * hei_arg_exp
        )
        # T-DER p_i: dp_i/dt = -dH/dr_i
        dp_dt_electrons_flat[3*kk:3*(kk+1)] = (
            ri * (f_en + f_heisenberg_p) + f_ee_sum + f_epbar + v_pauli_pp
        )

        # --- FORCES AND DERIVATIVES FOR PBAR ---
        # Force of all electrons on the antiproton
        f_pbar_e_sum -= f_epbar

    # ITERATION ON ELECTRONS FINISHED
    r_pbar_norm = np.linalg.norm(r_pbar)
    p_pbar_norm = np.linalg.norm(p_pbar)
    # Heisenberg core contribution for antiproton
    v_hei_pbar = 0.0
    # Guard against issues if r_pbar_norm or p_pbar_norm is zero, or XI_H is zero
    if r_pbar_norm > epsilon and p_pbar_norm > epsilon and np.abs(XI_H) > epsilon:
        uu_pbar = (r_pbar_norm * p_pbar_norm / XI_H)**2
        hei_arg_exp_pbar = uu_pbar**2
        # Cap the argument to prevent overflows or underflows
        if hei_arg_exp_pbar > 100 and ALPHA_H * (1 - hei_arg_exp_pbar) < -300:
            exp_hei_pbar = 0.0
        elif ALPHA_H * (1-hei_arg_exp_pbar) > 300:  # exp(300) overflows
            exp_hei_pbar = np.exp(300)
        else:
            exp_hei_pbar = np.exp(ALPHA_H * (1 - hei_arg_exp_pbar))
        v_hei_pbar = (XI_H**2 / (4 * ALPHA_H * r_pbar_norm**2 * M_STAR)) * exp_hei_pbar
    # Time derivative of r_pbar: dr_pbar/dt = dH/dp_pbar
    dR_pbar_dt = (p_pbar / M_STAR) * (1 - uu_pbar * exp_hei_pbar)

    # --- FORCES ON PBAR MOMENTUM ---
    # Force on antiproton from nucleus (attractive V = -ZZ/||R_pbar||)
    f_pbar_nuc = -ZZ / (r_pbar_norm**3 + epsilon)

    # Heisenberg core contribution for antiproton
    f_hei_p_pbar = (2 * v_hei_pbar / (r_pbar_norm**2 + epsilon)) * (
        1 + 2 * ALPHA_H * hei_arg_exp_pbar
    )

    # T-DER P: dP/dt = -dH/dR
    dP_pbar_dt = r_pbar * (f_pbar_nuc + f_hei_p_pbar) + f_pbar_e_sum

    # Assemble FULL DER vector
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

if not os.path.exists(DIRECTORY_PBAR):
    print('Directory not found.')
    print('Create Directory...')
    try:
        os.mkdir(DIRECTORY_PBAR)
    except FileExistsError:
        print("Directory was already created by a different process!")
else:
    print('Directory exists!')

# %% LOADING THE GS ATOM
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

# electrons spin, 0 for odd, 1 for even
e_spin = np.zeros(e_num, dtype=int)
# Set the spin of electrons based on their index
for i in range(e_num):
    e_spin[i] = i % 2  # 0 for odd, 1 for even

# Atomic number Z, number of protons
ZZ = p_num
M_STAR = M_PBAR / (1 + (1 / (2 * ZZ)))  # Reduced mass (a.u.)

# %% DYNAMIC SIMULATION
ENERGIES = np.linspace(MIN_E, MAX_E, E_STEP)  # Initial energies (a.u.)
E0 = ENERGIES[ID]

if AUTO_BMAX:
    # Determine b_max based on initial energy
    if E0 > THRESH_1:
        BMAX = B1
    elif E0 > THRESH_2:
        BMAX = B2
    else:
        BMAX = B3
else:
    BMAX = BMAX_0

CAPTURE = True  # Flag to control capture simulation
N_CHECK = 0  # Counter for processed trajectories
print(f"Running simulation for E0 = {E0:.3f} a.u. with ID {ID}...")
while CAPTURE:
    np.random.seed(N_CHECK)
    # %% ATOM RANDOM ORIENTATION
    # Randomize the angles
    theta_rnd = np.pi * np.random.random()
    phi_rnd = 2 * np.pi * np.random.random()

    # Convert to Cartesian coordinates
    rx, ry, rz, px, py, pz = convert_to_cartesian(
        r0, theta_r + theta_rnd, phi_r + phi_rnd,
        p0, theta_p + theta_rnd, phi_p + phi_rnd)

    # %% ANTIPROTON INITIALIZATION
    # Random impact parameter uniform in area
    BB = np.sqrt(np.random.random()) * BMAX
    angle = 2 * np.pi * np.random.random()
    # Launch antiproton far away along +x with offset in y
    r0_pbar = np.array([-XPBAR, BB * np.cos(angle), BB * np.sin(angle)])
    # initial momentum vector
    p0_pbar = np.array([np.sqrt(2 * E0 * M_STAR), 0.0, 0.0])

    # INITIAL STATE VECTOR
    # coordinates per particle: re1(3), re2(3), ..., rpbar(3),
    # momenta per particle: pe1(3), pe2(3), ..., ppbar(3)
    y0 = np.concatenate(
        [np.column_stack((rx, ry, rz)).flatten(),
            np.column_stack((px, py, pz)).flatten(), r0_pbar, p0_pbar]
        )
    
    # Time span for integration
    t_span = [0, T_MAX]
    t_eval = np.linspace(t_span[0], t_span[1], T_STEP)

    # %% INTEGRATION
    sol = solve_ivp(
        compute_forces,
        t_span, y0, args=(M_STAR, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, e_spin),
        t_eval=t_eval, dense_output=True, method='DOP853', rtol=1e-6, atol=1e-8
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

    # %% COMPUTE PBAR FINAL ENERGY
    # One body potentials
    norm_rf_pbar = np.linalg.norm(rf_pbar)
    norm_pf_pbar = np.linalg.norm(pf_pbar)
    kin_pbar = norm_pf_pbar**2 / (2 * M_STAR)
    nuc_pbar = -ZZ / np.linalg.norm(rf_pbar)
    heisenberg_pbar = (
        (XI_H**2 / (4 * ALPHA_H * M_STAR * norm_rf_pbar**2)) *
        np.exp(ALPHA_H * (1 - (norm_rf_pbar * norm_pf_pbar / XI_H)**4))
    )
    # Two body potentials
    pair_pot_pbar = 0.0  # Coulomb potential between electrons

    # %% COMPUTE ELECTRONS FINAL ENERGY and pbar term
    bound_electrons = []
    E_electrons = []

    for ii in range(e_num):
        rf_ei = rf_e[3 * ii:3 * (ii + 1)]
        pf_ei = pf_e[3 * ii:3 * (ii + 1)]

        norm_rf_ei = np.linalg.norm(rf_ei)
        norm_pf_ei = np.linalg.norm(pf_ei)
        kin_ei = norm_pf_ei**2 / 2.0
        nuc_ei = -ZZ / norm_rf_ei
        heisenberg_ei = (
            (XI_H**2 / (4 * ALPHA_H * norm_rf_ei**2)) *
            np.exp(ALPHA_H * (1 - (norm_rf_ei * norm_pf_ei / XI_H)**4))
        )

        # Electron-antiproton Coulomb
        ei_pbar = np.linalg.norm(rf_ei - rf_pbar)
        pot_pbar_ei = 1.0 / ei_pbar
        pair_pot_pbar += pot_pbar_ei

        # Electron-electron Coulomb and Pauli (sum only for j > ii to avoid double-counting)
        pair_pot_ei = 0
        pauli_pot = 0.0
        if e_num > 1:
            for j in range(ii + 1, e_num):
                rf_ej = rf_e[3 * j:3 * (j + 1)]
                delta_r = np.linalg.norm(rf_ei - rf_ej)
                pair_pot_ei += 1.0 / delta_r
                # Pauli term for identical spins
                if e_spin[ii] == e_spin[j]:
                    pf_ej = pf_e[3 * j:3 * (j + 1)]
                    delta_p = np.linalg.norm(pf_ei - pf_ej)
                    uu_p = (delta_r * delta_p / XI_P)**2
                    pauli_arg_exp_p = uu_p**2
                    if pauli_arg_exp_p > 100 and ALPHA_P * (1 - pauli_arg_exp_p) < -300:
                        exp_pauli = 0.0
                    elif ALPHA_P * (1 - pauli_arg_exp_p) > 300:
                        exp_pauli = np.exp(300)
                    else:
                        exp_pauli = np.exp(ALPHA_P * (1 - pauli_arg_exp_p))
                    pauli_pot += (XI_P**2 / (4 * ALPHA_P * delta_r**2)) * exp_pauli

        Ef_ei = kin_ei + nuc_ei + heisenberg_ei + pair_pot_ei + pot_pbar_ei + pauli_pot
        E_electrons.append(Ef_ei)

    # %% CAPTURE CLASSIFICATION
    # Final energy of the antiproton
    Ef_pbar = kin_pbar + nuc_pbar + pair_pot_pbar + heisenberg_pbar
    bound_p = Ef_pbar < 0     # Antiproton bound if Ef < 0

    if bound_p and any(e < 0 for e in E_electrons):
        CAPTURE = False
        print(f"Processed {N_CHECK} trajectories for E0 = {E0:.3f} a.u. ")
        end_time = time.time()
        print(f"Total simulation time: {end_time - start_time:.2f} seconds")

        # Save in a CSV file the positions and momenta of the antiproton and electrons
        with open(os.path.join(DIRECTORY_PBAR, f'capture_{ID}.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            header = ['p_num', 'e_num', 'E0', 'BMAX', 'rx', 'ry', 'rz',
                      'px', 'py', 'pz', 'rf_pbar_x', 'rf_pbar_y',
                      'rf_pbar_z', 'pf_pbar_x', 'pf_pbar_y',
                      'pf_pbar_z'] + [f'E_electron_{i}' for i in range(e_num)]
            writer.writerow(header)
            # Write data
            row_data = [p_num, e_num, E0, BMAX] + \
                       rf_e.tolist() + pf_e.tolist() + \
                       rf_pbar.tolist() + pf_pbar.tolist() + \
                       E_electrons
            writer.writerow(row_data)
    print(f"E_f of antiproton: {Ef_pbar:.3f} a.u. for {N_CHECK} check.")
    N_CHECK += 1
    if N_CHECK >= N_CHECK_MAX:
        print(f"Maximum number of checks ({N_CHECK_MAX}) reached for E0 = {E0:.3f} a.u.")
        CAPTURE = False

    # After integration (sol = solve_ivp(...))
    # Save all time steps if the antiproton is bound
    if bound_p:
        CAPTURE = False
        print(f"Processed {N_CHECK} trajectories for E0 = {E0:.3f} a.u. ")
        end_time = time.time()
        print(f"Total simulation time: {end_time - start_time:.2f} seconds")

        # Save positions and momenta at every time step
        with open(os.path.join(DIRECTORY_PBAR, f'trajectory_{ID}.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            # Header: time, electron positions/momenta, antiproton position/momentum
            header = ['time']
            for i in range(e_num):
                header += [f're{i+1}_x', f're{i+1}_y', f're{i+1}_z']
            for i in range(e_num):
                header += [f'pe{i+1}_x', f'pe{i+1}_y', f'pe{i+1}_z']
            header += ['rpbar_x', 'rpbar_y', 'rpbar_z', 'ppbar_x', 'ppbar_y', 'ppbar_z']
            writer.writerow(header)

            # Write data for each time step
            for idx, t in enumerate(sol.t):
                row = [t]
                # Electron positions
                for i in range(e_num):
                    row += sol.y[3*i:3*(i+1), idx].tolist()
                # Electron momenta
                for i in range(e_num):
                    row += sol.y[3*e_num + 3*i:3*e_num + 3*(i+1), idx].tolist()
                # Antiproton position and momentum
                row += sol.y[-6:-3, idx].tolist()  # rpbar
                row += sol.y[-3:, idx].tolist()    # ppbar
                writer.writerow(row)

# --- Robust antiproton bound check: median over last T_MEAN steps ---
n_times = sol.y.shape[1]
last_n = min(T_MEAN, n_times)
E_pbar_time = []

for idx in range(n_times - last_n, n_times):
    y = sol.y[:, idx]
    pf_pbar = y[-3:]
    rf_pbar = y[-6:-3]
    norm_rf_pbar = np.linalg.norm(rf_pbar)
    norm_pf_pbar = np.linalg.norm(pf_pbar)
    kin_pbar = norm_pf_pbar**2 / (2 * M_STAR)
    nuc_pbar = -ZZ / norm_rf_pbar
    heisenberg_pbar = (
        (XI_H**2 / (4 * ALPHA_H * M_STAR * norm_rf_pbar**2)) *
        np.exp(ALPHA_H * (1 - (norm_rf_pbar * norm_pf_pbar / XI_H)**4))
    )
    # Electron-antiproton Coulomb
    pair_pot_pbar = 0.0
    for ii in range(e_num):
        rf_ei = y[3 * ii:3 * (ii + 1)]
        pair_pot_pbar += 1.0 / (np.linalg.norm(rf_ei - rf_pbar))
    Ef_pbar = kin_pbar + nuc_pbar + pair_pot_pbar + heisenberg_pbar
    E_pbar_time.append(Ef_pbar)

med_Ef_pbar = np.median(E_pbar_time)
bound_p_median = med_Ef_pbar < 0

if bound_p_median:
    CAPTURE = False
    print(f"Processed {N_CHECK} trajectories for E0 = {E0:.3f} a.u. ")
    end_time = time.time()
    print(f"Total simulation time: {end_time - start_time:.2f} seconds")

    # Save positions and momenta at every time step
    with open(os.path.join(DIRECTORY_PBAR, f'trajectory_{ID}.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header: time, electron positions/momenta, antiproton position/momentum
        header = ['time']
        for i in range(e_num):
            header += [f're{i+1}_x', f're{i+1}_y', f're{i+1}_z']
        for i in range(e_num):
            header += [f'pe{i+1}_x', f'pe{i+1}_y', f'pe{i+1}_z']
        header += ['rpbar_x', 'rpbar_y', 'rpbar_z', 'ppbar_x', 'ppbar_y', 'ppbar_z']
        writer.writerow(header)

        # Write data for each time step
        for idx, t in enumerate(sol.t):
            row = [t]
            # Electron positions
            for i in range(e_num):
                row += sol.y[3*i:3*(i+1), idx].tolist()
            # Electron momenta
            for i in range(e_num):
                row += sol.y[3*e_num + 3*i:3*e_num + 3*(i+1), idx].tolist()
            # Antiproton position and momentum
            row += sol.y[-6:-3, idx].tolist()  # rpbar
            row += sol.y[-3:, idx].tolist()    # ppbar
            writer.writerow(row)
