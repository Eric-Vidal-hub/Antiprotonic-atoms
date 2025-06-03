import os
import numpy as np
import csv
from scipy.integrate import solve_ivp
import time

# Import constants and potentially shared functions if they were in a separate file
from v0_trajectory_constants_single import (
    M_PBAR, M_ELECTRON, ALPHA, XI_H, XI_P,
    E0_PBAR, XPBAR_INIT, BB_PBAR, T_MAX_SIM, ATOM_GS_CSV_PATH,
    CAPTURE_DISTANCE_THRESHOLD, CAPTURE_ENERGY_THRESHOLD,
    OUTPUT_DIR_SINGLE_TRAJ, OUTPUT_FILENAME_PREFIX
)

# --- Helper Functions (can be moved to a shared utility file) ---
def convert_spherical_to_cartesian_config(r_sph, theta_sph, phi_sph, p_sph, theta_p_sph, phi_p_sph):
    """ Converts arrays of spherical coords to Cartesian for multiple particles. """
    e_num = len(r_sph)
    r_cart = np.zeros((e_num, 3))
    p_cart = np.zeros((e_num, 3))
    for i in range(e_num):
        r_cart[i, 0] = r_sph[i] * np.sin(theta_sph[i]) * np.cos(phi_sph[i])
        r_cart[i, 1] = r_sph[i] * np.sin(theta_sph[i]) * np.sin(phi_sph[i])
        r_cart[i, 2] = r_sph[i] * np.cos(theta_sph[i])
        
        p_cart[i, 0] = p_sph[i] * np.sin(theta_p_sph[i]) * np.cos(phi_p_sph[i])
        p_cart[i, 1] = p_sph[i] * np.sin(theta_p_sph[i]) * np.sin(phi_p_sph[i])
        p_cart[i, 2] = p_sph[i] * np.cos(theta_p_sph[i])
    return r_cart, p_cart

def load_atom_gs(csv_filepath):
    gs_data_row = None
    with open(csv_filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        try:
            gs_data_row = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file {csv_filepath} is empty.")

    p_num = int(gs_data_row['p_num'])
    e_num = int(gs_data_row['e_num'])
    optimal_config_flat = np.fromstring(
        gs_data_row['optimal_configuration'].strip('[]'), sep=' '
    )
    # Config is: r_e_sph, theta_e_sph, phi_e_sph, p_e_sph, theta_pe_sph, phi_pe_sph
    r_e_s = optimal_config_flat[0*e_num : 1*e_num]
    th_e_s = optimal_config_flat[1*e_num : 2*e_num]
    phi_e_s = optimal_config_flat[2*e_num : 3*e_num]
    p_e_s = optimal_config_flat[3*e_num : 4*e_num]
    th_pe_s = optimal_config_flat[4*e_num : 5*e_num]
    phi_pe_s = optimal_config_flat[5*e_num : 6*e_num]
    
    r_elec_cart, p_elec_cart = convert_spherical_to_cartesian_config(
        r_e_s, th_e_s, phi_e_s, p_e_s, th_pe_s, phi_pe_s
    )
    return p_num, e_num, r_elec_cart, p_elec_cart

# --- Hamiltonian Equations of Motion (compute_forces) ---
def compute_forces(t, state, p_num_nucleus, num_electrons,
                   m_elec, m_pbar, xi_h_val, alpha_val, xi_p_val): # alpha_p assumed same as alpha_val
    """
    Forces for electrons and one antiproton.
    state: [r_e1, r_e2, ..., p_e1, p_e2, ..., r_pbar, p_pbar] (all Cartesian 3D vectors flattened)
    """
    r_e_flat = state[:3 * num_electrons]
    p_e_flat = state[3 * num_electrons : 6 * num_electrons]
    r_pbar = state[6 * num_electrons : 6 * num_electrons + 3]
    p_pbar = state[6 * num_electrons + 3 :]

    r_electrons = r_e_flat.reshape((num_electrons, 3))
    p_electrons = p_e_flat.reshape((num_electrons, 3))

    dr_dt_electrons_flat = np.zeros(3 * num_electrons)
    dp_dt_electrons_flat = np.zeros(3 * num_electrons)
    epsilon = 1e-18
    ZZ = p_num_nucleus

    # --- FOR ELECTRONS kk ---
    for kk in range(num_electrons):
        ri = r_electrons[kk]
        pi = p_electrons[kk]
        ri_norm = np.linalg.norm(ri)
        pi_norm = np.linalg.norm(pi)

        # Heisenberg e-N
        v_hei_en = 0.0
        uu_en = 0.0
        exp_hei_en = 0.0
        if ri_norm > epsilon and pi_norm > epsilon and abs(xi_h_val) > epsilon:
            uu_en = (ri_norm * pi_norm / xi_h_val)**2
            hei_arg_exp_en = uu_en**2
            # Capping
            exp_val_en = alpha_val * (1 - hei_arg_exp_en)
            if hei_arg_exp_en > 100 and exp_val_en < -300: exp_hei_en = 0.0
            elif exp_val_en > 300: exp_hei_en = np.exp(300)
            else: exp_hei_en = np.exp(exp_val_en)
            v_hei_en = (xi_h_val**2 / (4 * alpha_val * ri_norm**2 * m_elec)) * exp_hei_en # Using m_elec

        # dr_i/dt (Electron)
        # This form implies H_e = p_e^2/(2m_e) + V_H_e(r_e,p_e) where V_H_e gives the second term in dr/dt
        # So, ∂V_H_e/∂p_e = - p_e/m_e * (uu_en * exp_hei_en)
        dri_dt = (pi / m_elec) * (1 - uu_en * exp_hei_en) # Simpler FMD form for dr/dt from V_H
        
        # Pauli e-e contributions to dr_i/dt
        v_pauli_contrib_dr = np.zeros(3)
        # (Assuming E_SPIN is handled if this function is used in a context with spins)
        # For this single trajectory, we'd need to define spins for the target atom
        # If no Pauli, this term is zero. For Li, 2 spins same, 1 different.
        # For now, let's assume Pauli between e0-e2 if Li is e0,e1,e2 and spins are [0,1,0]
        # This part needs careful implementation based on how your FULL Hamiltonian has Pauli term.
        # The version from v2_multi_evo.py was complex.
        # A simpler Pauli derivative for dr_k/dt for pair (k,j):
        # ∂V_pauli_kj / ∂p_k = ( (p_k-p_j)/2 / m_elec_reduced_pauli ) * factor_from_pauli_exp
        # For now, omitting direct v_pauli_rr for simplicity, assuming dominant term is from V_H.
        # If you add it, ensure it's ∂V_Pauli_kk,sum_j / ∂p_k
        
        dr_dt_electrons_flat[3*kk : 3*(kk+1)] = dri_dt # + v_pauli_contrib_dr

        # dp_i/dt (Electron)
        f_en_coul = -ZZ * ri / (ri_norm**3 + epsilon)
        f_epbar_coul = (ri - r_pbar) / (np.linalg.norm(ri - r_pbar)**3 + epsilon) # Force on e_k BY pbar (Repulsive)

        f_ee_coul_sum = np.zeros(3)
        for jj in range(num_electrons):
            if kk == jj: continue
            r_kj = ri - r_electrons[jj]
            f_ee_coul_sum += r_kj / (np.linalg.norm(r_kj)**3 + epsilon)

        f_hei_en_force_scalar = (2 * v_hei_en / (ri_norm**2 + epsilon)) * (1 + 2 * alpha_val * hei_arg_exp_en)
        
        # Pauli e-e contributions to dp_i/dt
        v_pauli_contrib_dp = np.zeros(3)
        # Similar to dr/dt, this needs careful derivation.
        # -∂V_pauli_kj / ∂r_k = factor_pauli_exp * (r_k-r_j)/norm * scalar_terms
        # For now, omitting direct v_pauli_pp.

        dp_dt_electrons_flat[3*kk : 3*(kk+1)] = f_en_coul + f_epbar_coul + f_ee_coul_sum + \
                                               (ri * f_hei_en_force_scalar) # + v_pauli_contrib_dp
    
    # --- FOR ANTIPROTON ---
    r_pbar_norm = np.linalg.norm(r_pbar)
    p_pbar_norm = np.linalg.norm(p_pbar)

    # Heisenberg pbar-N (assuming similar form to e-N but with m_pbar)
    v_hei_pbar_n = 0.0
    uu_pbar_n = 0.0
    exp_hei_pbar_n = 0.0
    if r_pbar_norm > epsilon and p_pbar_norm > epsilon and abs(xi_h_val) > epsilon: # Use same XI_H, ALPHA for pbar-N
        uu_pbar_n = (r_pbar_norm * p_pbar_norm / xi_h_val)**2
        hei_arg_exp_pbar_n = uu_pbar_n**2
        exp_val_pn = alpha_val * (1 - hei_arg_exp_pbar_n)
        if hei_arg_exp_pbar_n > 100 and exp_val_pn < -300: exp_hei_pbar_n = 0.0
        elif exp_val_pn > 300: exp_hei_pbar_n = np.exp(300)
        else: exp_hei_pbar_n = np.exp(exp_val_pn)
        # Potential term uses M_PBAR in denominator
        v_hei_pbar_n = (xi_h_val**2 / (4 * alpha_val * r_pbar_norm**2 * m_pbar)) * exp_hei_pbar_n

    # dr_pbar/dt
    # Similar logic for dr/dt as for electrons, using m_pbar
    dR_pbar_dt = (p_pbar / m_pbar) * (1 - uu_pbar_n * exp_hei_pbar_n)

    # dp_pbar/dt
    f_pbar_nuc_coul = -ZZ * r_pbar / (r_pbar_norm**3 + epsilon) # Attractive to nucleus
    
    f_pbar_elec_coul_sum = np.zeros(3)
    for kk in range(num_electrons):
        r_pbark = r_pbar - r_electrons[kk] # Vector from e_k to pbar
        # Force on pbar BY e_k is - (r_pbark / norm^3) (Attractive because charges -1,-1 -> +1 potential)
        # Wait, pbar(-e) and electron(-e) REPEL. Force on pbar from electron k is + r_pbark / norm^3
        f_pbar_elec_coul_sum += r_pbark / (np.linalg.norm(r_pbark)**3 + epsilon)

    f_hei_pbar_n_force_scalar = (2 * v_hei_pbar_n / (r_pbar_norm**2 + epsilon)) * (1 + 2 * alpha_val * hei_arg_exp_pbar_n)
    
    dP_pbar_dt = f_pbar_nuc_coul + f_pbar_elec_coul_sum + (r_pbar * f_hei_pbar_n_force_scalar)

    derivatives = np.concatenate([
        dr_dt_electrons_flat, dp_dt_electrons_flat,
        dR_pbar_dt, dP_pbar_dt
    ])
    return derivatives

# --- Capture Event Definition ---
# This needs to be callable by solve_ivp's 'events' argument
def check_capture_event(t, state, p_num_nucleus, num_electrons, m_pbar, xi_h_val, alpha_val):
    """
    Checks if the antiproton is captured.
    Capture: pbar_dist_to_nucleus < threshold AND pbar_energy_rel_to_nucleus < threshold
    """
    r_pbar = state[6*num_electrons : 6*num_electrons+3]
    p_pbar = state[6*num_electrons+3 :]
    r_electrons = state[:3*num_electrons].reshape((num_electrons, 3))
    
    dist_pbar_nucleus = np.linalg.norm(r_pbar)
    
    # Calculate pbar energy (KE_pbar + V_pbar_N_Coul + sum(V_pbar_e_Coul) + V_pbar_N_Heis)
    ke_pbar = np.sum(p_pbar**2) / (2 * m_pbar)
    V_pbar_N_coul = -p_num_nucleus / (dist_pbar_nucleus + 1e-18)
    
    V_pbar_E_coul_sum = 0
    for i in range(num_electrons):
        V_pbar_E_coul_sum += 1.0 / (np.linalg.norm(r_pbar - r_electrons[i]) + 1e-18) # Repulsive

    # Heisenberg pbar-N
    V_pbar_N_heis = 0.0
    p_pbar_norm = np.linalg.norm(p_pbar)
    if dist_pbar_nucleus > 1e-18 and p_pbar_norm > 1e-18 and abs(xi_h_val) > 1e-18:
        uu_pn = (dist_pbar_nucleus * p_pbar_norm / xi_h_val)**2
        arg_exp_pn = uu_pn**2
        exp_val = alpha_val * (1 - arg_exp_pn)
        if arg_exp_pn > 100 and exp_val < -300: exp_h_pn = 0.0
        elif exp_val > 300: exp_h_pn = np.exp(300)
        else: exp_h_pn = np.exp(exp_val)
        V_pbar_N_heis = (xi_h_val**2 / (4 * alpha_val * dist_pbar_nucleus**2 * m_pbar)) * exp_h_pn
        
    pbar_energy_relative = ke_pbar + V_pbar_N_coul + V_pbar_E_coul_sum + V_pbar_N_heis
    
    # Event function: value should go from positive to negative (or vice-versa) at event
    # We want to stop if (dist < D_thresh AND energy < E_thresh)
    # Let's define event as: (CAPTURE_ENERGY_THRESHOLD - pbar_energy_relative)
    # This goes positive when pbar_energy_relative < CAPTURE_ENERGY_THRESHOLD.
    # And also dist_pbar_nucleus - CAPTURE_DISTANCE_THRESHOLD (goes negative when close)
    # For combined event, it's trickier. Let's use a simpler distance for now, or energy.
    
    # Stop if pbar energy is negative AND it's close
    if pbar_energy_relative < CAPTURE_ENERGY_THRESHOLD and dist_pbar_nucleus < CAPTURE_DISTANCE_THRESHOLD :
        return 0 # Event occurs when function is zero
    return 1 # Otherwise, no event
check_capture_event.terminal = True # Stop integration if event occurs
check_capture_event.direction = -1 # Event when value goes from positive to negative (or just hits zero)


# --- Main Simulation ---
if __name__ == "__main__":
    print("Starting single trajectory simulation for antiproton capture...")

    # Load Atom Ground State
    try:
        ZZ_nucleus, num_e, r_elec_init_cart, p_elec_init_cart = load_atom_gs(ATOM_GS_CSV_PATH)
        print(f"Loaded target atom: Z={ZZ_nucleus}, N_e={num_e}")
    except Exception as e:
        print(f"Error loading atom ground state from {ATOM_GS_CSV_PATH}: {e}")
        exit()

    # Initial Antiproton State
    # For simplicity, impact parameter BB_PBAR is in the y-direction, pbar approaches along -x
    r0_pbar = np.array([-XPBAR_INIT, BB_PBAR, 0.0])
    # Initial momentum (all in +x direction)
    # M_STAR_PBAR_SYSTEM for initial KE calculation (reduced mass pbar + TargetAtomAsWhole)
    # This M_STAR is only for setting initial p0_pbar from E0_PBAR.
    # The dynamics use M_PBAR for the antiproton.
    mass_target_atom_approx = ZZ_nucleus * M_PBAR # Approx mass of nucleus for reduced mass calc
    m_reduced_pbar_atom = (M_PBAR * mass_target_atom_approx) / (M_PBAR + mass_target_atom_approx)
    
    p0_pbar_magnitude = np.sqrt(2 * E0_PBAR * m_reduced_pbar_atom)
    p0_pbar = np.array([p0_pbar_magnitude, 0.0, 0.0])
    print(f"Antiproton: E0={E0_PBAR:.2f} au, B={BB_PBAR:.2f} au, R0_pbar={r0_pbar}, P0_pbar_mag={p0_pbar_magnitude:.2f} au")

    # Initial State Vector y0 for solve_ivp
    y0 = np.concatenate([
        r_elec_init_cart.flatten(),
        p_elec_init_cart.flatten(),
        r0_pbar,
        p0_pbar
    ])

    # Arguments for compute_forces and event function
    force_args = (ZZ_nucleus, num_e, M_ELECTRON, M_PBAR, XI_H, ALPHA, XI_P) # Assuming ALPHA_P=ALPHA
    event_args = (ZZ_nucleus, num_e, M_PBAR, XI_H, ALPHA)


    # Integration
    print(f"Starting integration up to T_MAX = {T_MAX_SIM:.1f} a.u. ...")
    start_time = time.time()
    sol = solve_ivp(
        compute_forces,
        (0.0, T_MAX_SIM),
        y0,
        args=force_args,
        method='DOP853', # Good general purpose, high order
        rtol=1e-7, # Tighter tolerance might be needed
        atol=1e-9,
        dense_output=True,
        events=check_capture_event # Add the event function
    )
    end_time = time.time()
    print(f"Integration finished in {end_time - start_time:.2f} seconds. Status: {sol.message}")

    if sol.status == 1: # Event triggered
        print("Capture event detected!")
        capture_time = sol.t_events[0][0]
        capture_state = sol.sol(capture_time) # Get state at exact event time
        print(f"  Capture occurred at t = {capture_time:.2f} a.u.")
        # Truncate solution arrays to the point of capture for saving
        t_to_save = sol.t[sol.t <= capture_time]
        y_to_save = sol.y[:, sol.t <= capture_time]
        if len(t_to_save) == 0 or t_to_save[-1] < capture_time : # Ensure event time is included
             t_to_save = np.append(t_to_save, capture_time)
             y_to_save = np.column_stack((y_to_save, capture_state))

    elif sol.status == 0: # Integration completed full T_MAX
        print("Integration completed full T_MAX. No capture event detected by criteria.")
        capture_time = T_MAX_SIM
        capture_state = sol.y[:, -1]
        t_to_save = sol.t
        y_to_save = sol.y
    else: # Solver failed
        print(f"Solver failed with status {sol.status}: {sol.message}")
        # Save what we have
        capture_time = sol.t[-1] if len(sol.t) > 0 else 0
        capture_state = sol.y[:, -1] if sol.y.shape[1] > 0 else y0
        t_to_save = sol.t
        y_to_save = sol.y


    # Prepare data for saving
    os.makedirs(OUTPUT_DIR_SINGLE_TRAJ, exist_ok=True)
    filename = f"{OUTPUT_FILENAME_PREFIX}_E0_{E0_PBAR:.2f}_B_{BB_PBAR:.2f}.npz"
    output_path = os.path.join(OUTPUT_DIR_SINGLE_TRAJ, filename)

    saved_data = {
        'time_array': t_to_save,
        'state_array': y_to_save, # Shape: (N_coords, N_timesteps)
        'num_electrons': num_e,
        'p_num_nucleus': ZZ_nucleus,
        'initial_E0_pbar': E0_PBAR,
        'initial_BB_pbar': BB_PBAR,
        'initial_XPBAR': XPBAR_INIT,
        'T_MAX_simulated': t_to_save[-1] if len(t_to_save) > 0 else 0,
        'capture_time': capture_time if sol.status == 1 else None,
        'capture_state': capture_state if sol.status == 1 else None,
        'final_status_message': sol.message,
        'FMD_ALPHA': ALPHA,
        'FMD_XI_H': XI_H, # This is the scaled one
        'FMD_XI_P': XI_P  # This is the scaled one
    }

    np.savez_compressed(output_path, **saved_data)
    print(f"Trajectory data saved to: {output_path}")

    # Example of how to check final pbar energy (if needed for analysis)
    final_pbar_r = capture_state[6*num_e : 6*num_e+3]
    final_pbar_p = capture_state[6*num_e+3 :]
    final_electrons_r = capture_state[:3*num_e].reshape((num_e,3))

    # Recalculate final pbar energy (simplified for printing)
    final_ke_pbar = np.sum(final_pbar_p**2) / (2 * M_PBAR)
    final_V_pbar_N = -ZZ_nucleus / (np.linalg.norm(final_pbar_r) + 1e-18)
    final_V_pbar_E = 0
    for i in range(num_e):
        final_V_pbar_E += 1.0 / (np.linalg.norm(final_pbar_r - final_electrons_r[i]) + 1e-18)
    # Simplified: not including Heisenberg here for brevity, add if needed
    final_pbar_energy_approx = final_ke_pbar + final_V_pbar_N + final_V_pbar_E
    print(f"Approximate final pbar energy (Coulomb + KE): {final_pbar_energy_approx:.3f} a.u.")
    if sol.status == 1:
         print(f"  Distance to nucleus at capture: {np.linalg.norm(final_pbar_r):.3f} a.u.")