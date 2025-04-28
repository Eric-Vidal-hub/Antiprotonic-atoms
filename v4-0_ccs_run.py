"""
Author: Eric Vidal Marcos
Date: 26-03-2025
Project: Antiproton-Helium Collision Cross Section Scan

This script simulates pbar-He collisions over ranges of impact parameters
and initial energies to estimate energy loss and capture cross sections.
It uses the semi-classical model including the Heisenberg core potential.

Based on Beck, Wilets, Alberg, Phys. Rev. A 48, 2779 (1993).
"""

import sys
import os
import csv
import time
import numpy as np
from scipy.integrate import solve_ivp

# --- Constants (Atomic Units) ---
M_PROTON = 1836.15267343
M_ALPHA = 4 * M_PROTON
M_PBAR = M_PROTON
M_ELECTRON = 1.0
Mstar = 1 / (1 / M_ALPHA + 1 / M_PROTON)  # Reduced mass

Q_E = -1.0
Q_PBAR = -1.0
Q_ALPHA = 2.0

# --- Parameters for Heisenberg Potential ---
ALPHA_H = 5.0
XI_H = 1.000

# --- Simulation Parameters ---
GROUND_STATE_FILE = 'HPC_results_gs_with_alpha_modifying/02_He_02e.csv'  # Input ground state file
T_VALUES_AU = [3.0, 1.5, 0.5, 0.1]  # Initial pbar kinetic energies to scan
B_MAX = 3.5         # Maximum impact parameter to scan
B_STEP = 0.1        # Step size for impact parameter scan
T_MAX_SIM = 400.0     # Max simulation time (might need adjustment per T)
INITIAL_PBAR_DIST = 50.0 # How far away pbar starts
R_CAPTURE_THRESHOLD = 10.0 # Max final distance for pbar to be considered captured

# --- Small epsilon to prevent division by zero ---
EPSILON = 1e-9

# --- Helper Functions ---

def parse_optimal_config(config_string):
    numbers = config_string.replace('[', '').replace(']', '').replace('\n', '').split()
    return np.array([float(n) for n in numbers])

def convert_spherical_to_cartesian(rr, theta, phi, pp, theta_p, phi_p):
    x = rr * np.sin(theta) * np.cos(phi)
    y = rr * np.sin(theta) * np.sin(phi)
    z = rr * np.cos(theta)
    pos_cart = np.stack((x, y, z), axis=-1)
    px = pp * np.sin(theta_p) * np.cos(phi_p)
    py = pp * np.sin(theta_p) * np.sin(phi_p)
    pz = pp * np.cos(theta_p)
    mom_cart = np.stack((px, py, pz), axis=-1)
    return pos_cart, mom_cart

# --- Equations of Motion (Copy from collision_simulation.py) ---

def derivatives(t, y, m_pbar_reduced, alpha_h, xi_h):
    num_electrons = 2
    dim = 3
    pos_idx_end = num_electrons * dim
    mom_idx_start = pos_idx_end + dim
    pbar_pos_idx_start = num_electrons*dim
    pbar_mom_idx_start = mom_idx_start + num_electrons*dim
    r_e = y[:pos_idx_end].reshape((num_electrons, dim))
    R_p = y[pbar_pos_idx_start:mom_idx_start]
    p_e = y[mom_idx_start:pbar_mom_idx_start].reshape((num_electrons, dim))
    P_p = y[pbar_mom_idx_start:]
    dydt = np.zeros_like(y)
    dr_dt = np.zeros_like(r_e)
    for i in range(num_electrons):
        ri = np.linalg.norm(r_e[i]) + EPSILON
        pi = np.linalg.norm(p_e[i]) + EPSILON
        ri_vec = r_e[i]
        pi_vec = p_e[i]
        dr_dt_ke = p_e[i] / M_ELECTRON
        term_in_exp = 1 - (ri * pi / xi_h)**4
        exp_term = np.exp(alpha_h * term_in_exp)
        common_factor_vh = (xi_h**2) / (4 * alpha_h * ri**2)
        dVH_dpi_mag = common_factor_vh * exp_term * alpha_h * (-4 * (ri / xi_h)**4 * pi**3)
        dVH_dp_vec = dVH_dpi_mag * (pi_vec / pi)
        dr_dt[i] = dr_dt_ke + dVH_dp_vec
    dydt[:pos_idx_end] = dr_dt.flatten()
    dydt[pbar_pos_idx_start:mom_idx_start] = P_p / m_pbar_reduced
    F_e = np.zeros_like(p_e)
    F_p = np.zeros_like(P_p)
    for i in range(num_electrons):
        ri = np.linalg.norm(r_e[i]) + EPSILON
        pi = np.linalg.norm(p_e[i]) + EPSILON
        ri_vec = r_e[i]
        pi_vec = p_e[i]
        F_e[i] += Q_E * Q_ALPHA * ri_vec / ri**3
        r_ie_p = r_e[i] - R_p
        r_ie_p_mag = np.linalg.norm(r_ie_p) + EPSILON
        F_e[i] += Q_E * Q_PBAR * r_ie_p / r_ie_p_mag**3
        for j in range(num_electrons):
            if i == j: continue
            r_ij = r_e[i] - r_e[j]
            r_ij_mag = np.linalg.norm(r_ij) + EPSILON
            F_e[i] += Q_E * Q_E * r_ij / r_ij_mag**3
        term_in_exp = 1 - (ri * pi / xi_h)**4
        exp_term = np.exp(alpha_h * term_in_exp)
        common_factor_vh = (xi_h**2) / (4 * alpha_h)
        dVH_dr_scalar = common_factor_vh * (
              (exp_term * alpha_h * (-4 * (pi/xi_h)**4 * ri**3) / ri**2)
            + (exp_term * (-2 / ri**3))
        )
        dVH_dr_vec = dVH_dr_scalar * (ri_vec / ri)
        F_e[i] -= dVH_dr_vec
    Rp_mag = np.linalg.norm(R_p) + EPSILON
    F_p += Q_PBAR * Q_ALPHA * R_p / Rp_mag**3
    for i in range(num_electrons):
        r_p_ie = R_p - r_e[i]
        r_p_ie_mag = np.linalg.norm(r_p_ie) + EPSILON
        F_p += Q_PBAR * Q_E * r_p_ie / r_p_ie_mag**3
    dydt[mom_idx_start:pbar_mom_idx_start] = F_e.flatten()
    dydt[pbar_mom_idx_start:] = F_p
    return dydt

# --- Simulation Setup ---

def setup_initial_state(gs_optimal_config_sph, T_initial_au, impact_param_b_au):
    """Sets up y0 from ground state config, T, and b."""
    e_num = 2 # Hardcoded for Helium
    rr_e = gs_optimal_config_sph[:e_num]
    theta_e = gs_optimal_config_sph[e_num : 2*e_num]
    phi_e = gs_optimal_config_sph[2*e_num : 3*e_num]
    pp_e = gs_optimal_config_sph[3*e_num : 4*e_num]
    theta_p_e = gs_optimal_config_sph[4*e_num : 5*e_num]
    phi_p_e = gs_optimal_config_sph[5*e_num : 6*e_num]

    pos_e_cart, mom_e_cart = convert_spherical_to_cartesian(
        rr_e, theta_e, phi_e, pp_e, theta_p_e, phi_p_e
    )

    R0 = np.array([-INITIAL_PBAR_DIST, impact_param_b_au, 0.0])
    P0_mag = np.sqrt(2.0 * Mstar * T_initial_au)
    P0 = np.array([P0_mag, 0.0, 0.0])

    y0 = np.concatenate([pos_e_cart.flatten(), R0, mom_e_cart.flatten(), P0])
    return y0

# --- Simulation Execution ---

def run_simulation(y0, t_max):
    """Runs the ODE solver for a single trajectory."""
    sol = solve_ivp(
        derivatives, [0, t_max], y0,
        args=(Mstar, ALPHA_H, XI_H),
        method='DOP853', dense_output=False, # Don't need dense output here
        rtol=1e-6, atol=1e-8 # Slightly looser tolerance for speed
    )
    if sol.status != 0:
        print(f"Warning: ODE solver failed! Status {sol.status}: {sol.message}")
        return None, False # Indicate failure
    return sol.y[:, -1], True # Return final state y_f and success

# --- Analysis ---

def analyze_final_state(y_f, T_initial_au):
    """Analyzes the final state y_f to determine outcome and energy loss."""
    e_num = 2
    dim = 3
    pos_idx_end = e_num * dim
    mom_idx_start = pos_idx_end + dim
    pbar_pos_idx_start = e_num*dim
    pbar_mom_idx_start = mom_idx_start + e_num*dim

    r_e_f = y_f[:pos_idx_end].reshape((e_num, dim))
    R_p_f = y_f[pbar_pos_idx_start:mom_idx_start]
    p_e_f = y_f[mom_idx_start:pbar_mom_idx_start].reshape((e_num, dim))
    P_p_f = y_f[pbar_mom_idx_start:]

    # Calculate final pbar kinetic energy
    T_final_au = np.sum(P_p_f**2) / (2.0 * Mstar)
    delta_E = T_initial_au - T_final_au

    # --- Check for Capture ---
    Rp_f_mag = np.linalg.norm(R_p_f)
    # Energy of pbar relative to nucleus ONLY
    E_p_rel_nucleus = T_final_au + Q_PBAR * Q_ALPHA / (Rp_f_mag + EPSILON)

    is_captured = (E_p_rel_nucleus < 0) and (Rp_f_mag < R_CAPTURE_THRESHOLD)

    outcome = "scattered"
    if is_captured:
        # Check ionization state of electrons relative to (alpha + pbar)
        ionized_count = 0
        for i in range(e_num):
            ke_e = np.sum(p_e_f[i]**2) / (2.0 * M_ELECTRON)
            pe_alpha_e = Q_E * Q_ALPHA / (np.linalg.norm(r_e_f[i]) + EPSILON)
            pe_pbar_e = Q_E * Q_PBAR / (np.linalg.norm(r_e_f[i] - R_p_f) + EPSILON)
            E_e_total = ke_e + pe_alpha_e + pe_pbar_e
            if E_e_total > 0:
                ionized_count += 1

        if ionized_count == 2:
            outcome = "capture_DI" # Double Ionization
        elif ionized_count == 1:
            outcome = "capture_SI" # Single Ionization
        else:
            outcome = "capture_NI" # No Ionization (transient alpha e2 pbar) - treat as SI? or separate?

    return outcome, delta_E

# --- Main Loop ---
if __name__ == "__main__":
    print("--- pbar-He Cross Section Scan ---")
    if not os.path.exists(GROUND_STATE_FILE):
        print(f"Error: Ground state file not found: {GROUND_STATE_FILE}", file=sys.stderr)
        sys.exit(1)

    # Load ground state once
    print(f"Loading ground state from: {GROUND_STATE_FILE}")
    with open(GROUND_STATE_FILE, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        gs_data = next(reader)
    gs_optimal_config = parse_optimal_config(gs_data['optimal_configuration'])
    print("Ground state loaded.")

    # Prepare impact parameter scan values
    # Start from delta_b/2 to represent center of first annulus
    b_values = np.arange(B_STEP / 2.0, B_MAX, B_STEP)
    num_b_steps = len(b_values)
    delta_b = B_STEP # Annulus width

    results_summary = {} # Store results keyed by T

    total_sim_start_time = time.time()

    # Loop over initial energies
    for T_init in T_VALUES_AU:
        print(f"\n--- Simulating for T_initial = {T_init:.3f} a.u. ---")
        results_T = {
            'b': [],
            'outcome': [],
            'delta_E': []
        }
        b_max_DI = 0.0
        b_max_SI = 0.0
        b_max_capture = 0.0 # Max b for any capture (DI or SI)

        sim_count_T = 0
        time_start_T = time.time()

        # Loop over impact parameters
        for b in b_values:
            sim_count_T += 1
            print(f"  Running b = {b:.3f} a.u. ({sim_count_T}/{num_b_steps})... ", end="")
            time_start_b = time.time()

            y0 = setup_initial_state(gs_optimal_config, T_init, b)
            y_f, success = run_simulation(y0, T_MAX_SIM)

            outcome = "failed"
            delta_E = np.nan
            if success:
                outcome, delta_E = analyze_final_state(y_f, T_init)
            time_end_b = time.time()
            print(f"Outcome: {outcome}, Delta E: {delta_E:+.3f} ({time_end_b - time_start_b:.1f}s)")

            results_T['b'].append(b)
            results_T['outcome'].append(outcome)
            results_T['delta_E'].append(delta_E)

            # Update max b for cross section estimate
            if outcome == "capture_DI":
                b_max_DI = b # Keep track of largest b leading to DI
                b_max_capture = b
            elif outcome == "capture_SI":
                 # Assume SI occurs outside DI region
                if b > b_max_DI:
                    b_max_SI = b # Keep track of largest b leading to SI (outside DI)
                b_max_capture = b # Update max capture b regardless


        time_end_T = time.time()

        # --- Calculate Cross Sections for this T ---
        # Approximate using the midpoint rule: sigma = pi * b_max^2
        # where b_max is approximated as b_last + delta_b/2

        effective_b_max_DI = b_max_DI + delta_b / 2.0
        effective_b_max_SI = b_max_SI + delta_b / 2.0 # Max b for SI (potentially including DI range)
        effective_b_max_capture = b_max_capture + delta_b / 2.0

        sigma_DI = np.pi * effective_b_max_DI**2
        # Sigma_SI is the annulus between DI and SI boundaries
        sigma_SI = np.pi * (effective_b_max_SI**2 - effective_b_max_DI**2)
        if sigma_SI < 0: sigma_SI = 0 # Ensure non-negative if SI doesn't extend beyond DI

        sigma_capture_total = np.pi * effective_b_max_capture**2
        # Consistency check: sigma_capture_total should be approx sigma_DI + sigma_SI
        # This method relies on SI happening concentrically outside DI.

        # Store results
        results_summary[T_init] = {
            'details': results_T,
            'sigma_DI': sigma_DI,
            'sigma_SI': sigma_SI,
            'sigma_capture_total': sigma_capture_total,
            'time_taken': time_end_T - time_start_T
        }

        print(f"\n--- Results for T_initial = {T_init:.3f} a.u. ---")
        print(f"  Time taken: {results_summary[T_init]['time_taken']:.1f}s")
        print(f"  Approximate Max b for DI: {effective_b_max_DI:.3f} => sigma_DI = {sigma_DI:.3f} a.u.")
        print(f"  Approximate Max b for SI (outside DI): {effective_b_max_SI:.3f} => sigma_SI = {sigma_SI:.3f} a.u.")
        print(f"  Approximate Total Capture Cross Section: {sigma_capture_total:.3f} a.u.")

        # Note: Energy loss cross section requires more analysis (binning delta_E weighted by 2*pi*b*db)
        # You can access the raw data via results_summary[T_init]['details']['delta_E']
        # and results_summary[T_init]['details']['b']

    total_sim_end_time = time.time()
    print(f"\nTotal simulation time: {total_sim_end_time - total_sim_start_time:.1f}s")

    # --- Optional: Save summary results ---
    # with open('cross_section_summary.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['T_initial_au', 'sigma_DI_au', 'sigma_SI_au', 'sigma_capture_total_au', 'time_s'])
    #     for T_init, results in results_summary.items():
    #         writer.writerow([T_init, results['sigma_DI'], results['sigma_SI'], results['sigma_capture_total'], results['time_taken']])

    print("\nScan complete.")
