# he_antiproton_capture.py
"""
Semiclassical Monte Carlo simulation of He + antiproton capture
using the FMD method (Beck et al., Phys. Rev. A 48, 2779 (1993)).

Produces:
 - Capture cross sections vs. initial energy (total, single, double).  Outputs cross_sections.csv
 - Example trajectory radii vs. time.  Outputs trajectory_example.csv
 - Initial (E,L) distribution for capture events.  Outputs initial_states.csv
 - Final (E,L) distribution after capture.  Outputs final_states.csv

Reads:
 - 02_He_02e.csv: Helium ground-state two-electron coordinates and momenta

Usage:
    python he_antiproton_capture.py

Dependencies:
    numpy, pandas, scipy
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

# Simulation parameters
ENERGIES = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]   # initial antiproton energies (a.u.)
N_TRAJ = 1000         # trajectories per energy
T_MAX = 25000.0       # max time (a.u.)
THRESH_1 = 2.3        # energy threshold for stepping b_max
THRESH_2 = 1.2
B1, B2, B3 = 1.0, 2.0, 3.0  # impact parameters (a.u.)

# Load helium electron config
helium_df = pd.read_csv('02_He_02e.csv')
# Expect columns: r1_x, r1_y, r1_z, p1_x, p1_y, p1_z, r2_x, ... p2_z
# Convert to numpy arrays:
r1_init = helium_df[['r1_x','r1_y','r1_z']].iloc[0].values
p1_init = helium_df[['p1_x','p1_y','p1_z']].iloc[0].values
r2_init = helium_df[['r2_x','r2_y','r2_z']].iloc[0].values
p2_init = helium_df[['p2_x','p2_y','p2_z']].iloc[0].values

# Physical constants
M_PBAR = 1836.152672  # antiproton mass (a.u.)

# FMD potential parameters (Heisenberg core)
alpha = 5.0
xi_h = 1.0/np.sqrt(1+1/(2*alpha))
# (more parameters can be set here as needed)

# Placeholder functions for forces -- to be implemented

def compute_forces(state):
    """
    Given state vector y = [r1(3), r2(3), r_p(3), p1(3), p2(3), p_p(3)],
    returns derivatives dy/dt according to Hamilton's equations.
    """
    # Unpack
    r1 = state[0:3]; r2 = state[3:6]; rp = state[6:9]
    p1 = state[9:12]; p2 = state[12:15]; pp = state[15:18]

    # Initialize derivatives
    dr1_dt = p1         # electron mass = 1 a.u.
    dr2_dt = p2
    drp_dt = pp / M_PBAR

    # Compute forces: -dV/dr for all interactions + Heisenberg core
    # Coulomb terms: electron-nucleus, electron-electron, electron-antiproton, antiproton-nucleus
    # and Heisenberg core on electrons
    # TODO: fill in exact force expressions
    F1 = np.zeros(3)
    F2 = np.zeros(3)
    Fp = np.zeros(3)

    # dp/dt = force
    dp1_dt = F1
    dp2_dt = F2
    dpp_dt = Fp

    return np.concatenate([dr1_dt, dr2_dt, drp_dt, dp1_dt, dp2_dt, dpp_dt])


def classify_capture(final_state):
    """
    Decide if the trajectory leads to single or double capture.
    Returns 'none', 'single', or 'double'.
    """
    r1, r2, rp, p1, p2, pp = (
        final_state[0:3], final_state[3:6], final_state[6:9],
        final_state[9:12], final_state[12:15], final_state[15:18]
    )
    # Compute electron energies: kinetic + Coulomb potential
    E1 = 0.5*np.dot(p1,p1) - 2.0/np.linalg.norm(r1)
    E2 = 0.5*np.dot(p2,p2) - 2.0/np.linalg.norm(r2)
    # Compute antiproton energy
    E_p = 0.5*np.dot(pp,pp)/M_PBAR - 2.0/np.linalg.norm(rp)

    bound1 = (E1 < 0)
    bound2 = (E2 < 0)
    bound_p = (E_p < 0)

    if bound_p and bound1 and bound2:
        return 'double'
    elif bound_p and (bound1 ^ bound2):
        return 'single'
    else:
        return 'none'

# Prepare results storage
cross_data = []
initial_states = []
final_states = []
traj_saved = False

for E0 in ENERGIES:
    n_cap = 0; n_single = 0; n_double = 0

    # Determine b_max based on initial energy
    if E0 > THRESH_1:
        bmax = B1
    elif E0 > THRESH_2:
        bmax = B2
    else:
        bmax = B3

    for i in range(N_TRAJ):
        # Random impact parameter uniform in area
        b = np.sqrt(np.random.random()) * bmax
        theta = 2*np.pi*np.random.random()
        # Launch antiproton far away along +x with offset in y
        rp0 = np.array([-100.0, b*np.cos(theta), b*np.sin(theta)])
        vp_mag = np.sqrt(2*E0/M_PBAR)
        vp0 = np.array([vp_mag, 0.0, 0.0])

        # Initial state vector
        y0 = np.concatenate([r1_init, r2_init, rp0, p1_init, p2_init, vp0*M_PBAR])

        # Integrate
        sol = solve_ivp(
            compute_forces,
            (0.0, T_MAX), y0,
            method='RK45', rtol=1e-6, atol=1e-9
        )

        yf = sol.y[:, -1]
        cap_type = classify_capture(yf)
        if cap_type != 'none':
            n_cap += 1
            if cap_type == 'single':
                n_single += 1
            else:
                n_double += 1
            # Record first capturing trajectory for radii
            if not traj_saved:
                # compute radii time series and save
                times = sol.t
                r_e1 = np.linalg.norm(sol.y[0:3, :], axis=0)
                r_e2 = np.linalg.norm(sol.y[3:6, :], axis=0)
                r_p  = np.linalg.norm(sol.y[6:9, :], axis=0)
                df_traj = pd.DataFrame({'time': times,
                                        'r_e1': r_e1,
                                        'r_e2': r_e2,
                                        'r_p': r_p})
                df_traj.to_csv('trajectory_example.csv', index=False)
                traj_saved = True

        # Record initial and final (E,L)
        # Initial L and E
        L_init = np.linalg.norm(np.cross(rp0, vp0*M_PBAR))
        initial_states.append((E0, L_init, cap_type))
        # Final L and E
        rpf = yf[6:9]; vpf = yf[15:18]/M_PBAR
        E_pf = 0.5*np.dot(vpf, vpf)*M_PBAR - 2.0/np.linalg.norm(rpf)
        L_pf = np.linalg.norm(np.cross(rpf, yf[15:18]))
        final_states.append((E_pf, L_pf, cap_type))

    # Compute cross sections
    sigma_tot = np.pi*bmax**2 * (n_cap     / N_TRAJ)
    sigma_sng = np.pi*bmax**2 * (n_single  / N_TRAJ)
    sigma_dbl = np.pi*bmax**2 * (n_double  / N_TRAJ)
    cross_data.append((E0, sigma_tot, sigma_sng, sigma_dbl))

# Save CSVs
pd.DataFrame(cross_data, columns=['Energy','Sigma_total','Sigma_single','Sigma_double']) \
  .to_csv('cross_sections.csv', index=False)
pd.DataFrame(initial_states, columns=['E_initial','L_initial','type']) \
  .to_csv('initial_states.csv', index=False)
pd.DataFrame(final_states,   columns=['E_final','L_final','type'])   \
  .to_csv('final_states.csv', index=False)

print("Simulation completed. CSV files written:")
print(" - cross_sections.csv")
print(" - trajectory_example.csv")
print(" - initial_states.csv")
print(" - final_states.csv")
