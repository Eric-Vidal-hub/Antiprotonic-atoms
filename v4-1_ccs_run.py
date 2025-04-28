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
 - 02_He_02e.csv: Helium ground-state two-electron coordinates and momenta

Dependencies:
    numpy, pandas, scipy
"""
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from tqdm import tqdm  # Import tqdm for the progress bar


#%% FUNCTIONS
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


def compute_forces(t, state):
    """
    Generalized force computation for an arbitrary number of electrons.
    Given state vector y = [r1(3), r2(3), ..., r_p(3), p1(3), p2(3), ..., p_p(3)],
    returns derivatives dy/dt according to Hamilton's equations.
    """
    # Unpack state vector
    # r1, r2, ..., rp: electron positions (3D vectors)
    num_electrons = (len(state) - 6) // 6  # Calculate the number of electrons
    # -6 for the antiproton position and momentum
    r_electrons = [state[3 * i:3 * (i + 1)] for i in range(num_electrons)]
    p_electrons = [state[3 * (num_electrons + i):3 * (num_electrons + i + 1)] for i in range(num_electrons)]
    rp = state[3 * num_electrons:3 * num_electrons + 3]
    pp = state[3 * (2 * num_electrons):3 * (2 * num_electrons) + 3]

    # Initialize derivatives
    dr_electrons_dt = []
    dp_electrons_dt = []
    drp_dt = pp / M_PBAR

    # Compute forces for each electron
    for i in range(num_electrons):
        ri = r_electrons[i]
        pi = p_electrons[i]
        ri_norm = np.linalg.norm(ri)
        pi_norm = np.linalg.norm(pi)

        # Heisenberg core factors
        Xi = (ri_norm * pi_norm / XI_H)
        E_hi = np.exp(ALPHA * (1 - Xi**4))

        # dr/dt from p + Heisenberg momentum term
        dVdpi_mag = -(ri_norm**2 * pi_norm**3 / XI_H**2) * E_hi
        dri_dt = pi + (dVdpi_mag * (pi / pi_norm))

        # Coulomb forces
        # Electron-nucleus: F = +2 r/r^3
        Fi_nuc = 2 * ri / ri_norm**3

        # Electron-electron interactions
        Fi_ee = np.zeros(3)
        for j in range(num_electrons):
            if i != j:
                rj = r_electrons[j]
                rij = ri - rj
                norm_ij = np.linalg.norm(rij)
                Fi_ee += -rij / norm_ij**3

        # Electron-antiproton: F = - (r_i - r_p)/|r_i-r_p|^3
        rip = ri - rp
        norm_ip = np.linalg.norm(rip)
        Fi_ep = -rip / norm_ip**3

        # Heisenberg core radial force
        Fhi_rad = (XI_H**2 * E_hi / (2 * ALPHA * ri_norm**3) +
                   (ri_norm * pi_norm**4 / XI_H**2) * E_hi)
        Fhi = Fhi_rad * (ri / ri_norm)

        # Total dp/dt for this electron
        dpi_dt = Fi_nuc + Fi_ee + Fi_ep + Fhi

        # Append results
        dr_electrons_dt.append(dri_dt)
        dp_electrons_dt.append(dpi_dt)

    # Forces on antiproton
    # Antiproton-nucleus: F = +2 r_p/r_p^3
    Fp_nuc = 2 * rp / np.linalg.norm(rp)**3

    # Antiproton-electron interactions
    Fp_e = np.zeros(3)
    for ri in r_electrons:
        rip = ri - rp
        norm_ip = np.linalg.norm(rip)
        Fp_e += rip / norm_ip**3

    dpp_dt = Fp_nuc + Fp_e

    # Combine derivatives
    derivatives = []
    for dr, dp in zip(dr_electrons_dt, dp_electrons_dt):
        derivatives.extend(dr)
        derivatives.extend(dp)
    derivatives.extend(drp_dt)
    derivatives.extend(dpp_dt)

    return np.array(derivatives)

def classify_capture(final_state, num_electrons):
    """
    Decide if the trajectory leads to single or double capture.
    Returns 'none', 'single', or 'double'.
    """
    r_electrons = [final_state[3 * i:3 * (i + 1)] for i in range(num_electrons)]
    p_electrons = [final_state[3 * (num_electrons + i):3 * (num_electrons + i + 1)] for i in range(num_electrons)]
    rp = final_state[3 * num_electrons:3 * num_electrons + 3]
    pp = final_state[3 * (2 * num_electrons):3 * (2 * num_electrons) + 3]

    # Compute energies
    bound_electrons = []
    for r, p in zip(r_electrons, p_electrons):
        E = 0.5 * np.dot(p, p) - 2.0 / np.linalg.norm(r)
        bound_electrons.append(E < 0)

    E_p = 0.5 * np.dot(pp, pp) / M_PBAR - 2.0 / np.linalg.norm(rp)
    bound_p = (E_p < 0)

    if bound_p and all(bound_electrons):
        return 'double'
    elif bound_p and any(bound_electrons):
        return 'single'
    else:
        return 'none'


#%% SIMULATION

# Simulation parameters
# initial antiproton energies (a.u.)
ENERGIES = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
N_TRAJ = 1000         # trajectories per energy
T_MAX = 25000.0       # max time (a.u.)
THRESH_1 = 2.3        # energy threshold for stepping b_max
THRESH_2 = 1.2
B1, B2, B3 = 1.0, 2.0, 3.0  # impact parameters (a.u.)

# Physical constants
M_PBAR = 1836.152672  # antiproton mass (a.u.)

# FMD parameters
ALPHA = 5       # Hardness parameter
XI_H = 1.000    # Tuning parameter for the Heisenberg potential
XI_P = 2.767    # Tuning parameter for the Pauli potential
# SCALING PARAMETERS according to alpha
XI_H /= np.sqrt(1 + 1 / (2 * ALPHA))
XI_P /= np.sqrt(1 + 1 / (2 * ALPHA))

# Load helium electron config
helium_df = pd.read_csv('HPC_results_gs_with_alpha_modifying/02_He_02e.csv')

# Ensure the expected columns exist
required_columns = ['e_num', 'optimal_configuration']
if not all(col in helium_df.columns for col in required_columns):
    raise KeyError(f"Missing required columns in the CSV file: {required_columns}")

# Expect columns: r0_1, r0_i, theta_r_1, theta_r_i, phi_r_1, phi_r_i, p0_1,
# p0_i, theta_p_1, theta_p_i, phi_p_1, phi_p_i
# Convert to numpy arrays
for _, row in helium_df.iterrows():
    e_num = int(row['e_num'])
    optimal_config = np.fromstring(row['optimal_configuration'].strip('[]'), sep=' ')
    r0 = optimal_config[:e_num]
    theta_r = optimal_config[e_num:2*e_num]
    phi_r = optimal_config[2*e_num:3*e_num]
    p0 = optimal_config[3*e_num:4*e_num]
    theta_p = optimal_config[4*e_num:5*e_num]
    phi_p = optimal_config[5*e_num:6*e_num]

print(f"Loaded {e_num} electrons with initial positions and momenta.")
# Initial positions and momenta of electrons
print(f"Initial position of electrons: {r0}")

# Convert to Cartesian coordinates
rx, ry, rz, px, py, pz = convert_to_cartesian(
    r0, theta_r, phi_r, p0, theta_p, phi_p)

# STORAGE PARAMETERS
cross_data = []
initial_states = []
final_states = []
traj_saved = False


# DYNAMIC SIMULATION
for E0 in ENERGIES:
    # Initialize counters
    n_cap = 0
    n_single = 0
    n_double = 0

    # Determine b_max based on initial energy
    if E0 > THRESH_1:
        bmax = B1
    elif E0 > THRESH_2:
        bmax = B2
    else:
        bmax = B3

    for i in tqdm(range(N_TRAJ), desc=f"Processing trajectories for E0={E0}"):
        # Random impact parameter uniform in area
        b = np.sqrt(np.random.random()) * bmax
        theta = 2 * np.pi * np.random.random()
        # Launch antiproton far away along +x with offset in y
        rpbar0 = np.array([-100.0, b * np.cos(theta), b * np.sin(theta)])
        vpbar_mag = np.sqrt(2 * E0 / M_PBAR)   # initial velocity magnitude
        vpbar0 = np.array([vpbar_mag, 0.0, 0.0])  # initial velocity vector

        # Initial state vector
        y0 = np.concatenate(
                [np.array([rx, ry, rz]).flatten(), rpbar0,
                 np.array([px, py, pz]).flatten(), vpbar0 * M_PBAR]
            )

        # print("Initial state vector:", y0)
        # print("Length of y0:", len(y0))

        # Integrate
        sol = solve_ivp(
            compute_forces,
            (0.0, T_MAX), y0,
            method='RK45', rtol=1e-6, atol=1e-9
        )

        # print("Integration successful:", sol.success)
        # print("Length of sol.y:", len(sol.y))
        # print("Shape of sol.y:", sol.y.shape)
        # print("Vector of sol.y:", sol.y)

        yf = sol.y[:, -1]
        cap_type = classify_capture(yf, e_num)
        if cap_type != 'none':
            n_cap += 1
            if cap_type == 'single':
                n_single += 1
            else:
                n_double += 1
            if not traj_saved:
                times = sol.t
                r_p = np.linalg.norm(sol.y[3 * e_num:3 * e_num + 3, :], axis=0)
                df_traj = pd.DataFrame({'time': times, 'r_p': r_p})
                df_traj.to_csv('trajectory_example.csv', index=False)
                traj_saved = True

        # Record initial and final (E,L)
        L_init = np.linalg.norm(np.cross(rpbar0, vpbar0 * M_PBAR))
        initial_states.append((E0, L_init, cap_type))

        # print("Length of yf:", len(yf))
        # print("Slice for vpf:", 3 * (2 * e_num), "to", 3 * (2 * e_num) + 3)

        # Extract final position and momentum of the antiproton
        rpf = yf[3 * e_num:3 * e_num + 3]  # Final position of the antiproton
        vpf = yf[3 * (2 * e_num):3 * (2 * e_num) + 3] / M_PBAR  # Final velocity of the antiproton

        # Ensure the vectors are 1D with 3 elements
        rpf = np.array(rpf).flatten()
        vpf = np.array(vpf).flatten()

        # Compute final energy and angular momentum
        E_pf = 0.5 * np.dot(vpf, vpf) * M_PBAR - 2.0 / np.linalg.norm(rpf)
        # print("rpf shape:", rpf.shape)
        # print("vpf shape:", vpf.shape)
        L_pf = np.linalg.norm(np.cross(rpf, vpf * M_PBAR))

        final_states.append((E_pf, L_pf, cap_type))

        # Compute cross sections
        sigma_tot = np.pi * bmax**2 * (n_cap / N_TRAJ)
        sigma_sng = np.pi * bmax**2 * (n_single / N_TRAJ)
        sigma_dbl = np.pi * bmax**2 * (n_double / N_TRAJ)
        cross_data.append((E0, sigma_tot, sigma_sng, sigma_dbl))

# Save CSVs
pd.DataFrame(cross_data, columns=['Energy', 'Sigma_total', 'Sigma_single', 'Sigma_double']) \
  .to_csv('cross_sections.csv', index=False)
pd.DataFrame(initial_states, columns=['E_initial', 'L_initial', 'type']) \
  .to_csv('initial_states.csv', index=False)
pd.DataFrame(final_states, columns=['E_final', 'L_final', 'type']) \
  .to_csv('final_states.csv', index=False)

print("Simulation completed. CSV files written:")
print(" - cross_sections.csv")
print(" - trajectory_example.csv")
print(" - initial_states.csv")
print(" - final_states.csv")
