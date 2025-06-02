import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Style settings (copied from pot_alpha.py)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 1
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['text.usetex'] = False

# --- Constants from your v3_ccs_FMD_constants.py or similar ---
M_PBAR_MASS = 1836.152672 # Antiproton mass (used if Heisenberg involves mass)
ALPHA_FMD = 5.0          # Hardness parameter (assuming same for pbar-nucleus Heisenberg)
XI_H_FMD_BASE = 1.0      # Base Tuning parameter for the Heisenberg potential
# Scaled XI_H for pbar-nucleus (if you use the same scaling as for electrons)
XI_H_PBAR_SCALED = XI_H_FMD_BASE / np.sqrt(1 + 1 / (2 * ALPHA_FMD))

# --- Function to load Ground State Configuration ---
def load_gs_configuration(csv_filepath):
    """Loads the ground state configuration from a CSV file."""
    gs_data_row = None
    with open(csv_filepath, mode='r') as file:
        reader = csv.DictReader(file)
        try:
            gs_data_row = next(reader) # Assuming one relevant row (e.g., He)
        except StopIteration:
            raise ValueError(f"CSV file {csv_filepath} is empty or has no data rows.")

    p_num = int(gs_data_row['p_num'])
    e_num = int(gs_data_row['e_num'])
    optimal_config_spherical_flat = np.fromstring(
        gs_data_row['optimal_configuration'].strip('[]'), sep=' '
    )

    # Convert spherical to Cartesian for electrons
    # (Using a simplified conversion here, assuming your full conversion is available elsewhere)
    rr_s = optimal_config_spherical_flat[0*e_num : 1*e_num]
    theta_s = optimal_config_spherical_flat[1*e_num : 2*e_num]
    phi_s = optimal_config_spherical_flat[2*e_num : 3*e_num]
    # Momenta are not needed for static electron positions

    r_electrons_cart = np.zeros((e_num, 3))
    for i in range(e_num):
        r_electrons_cart[i, 0] = rr_s[i] * np.sin(theta_s[i]) * np.cos(phi_s[i])
        r_electrons_cart[i, 1] = rr_s[i] * np.sin(theta_s[i]) * np.sin(phi_s[i])
        r_electrons_cart[i, 2] = rr_s[i] * np.cos(theta_s[i])
        
    return p_num, e_num, r_electrons_cart

# --- Function to Calculate Potential for Antiproton ---
def calculate_pbar_potential(r_pbar_pos, p_num_nucleus, r_electrons_cart, 
                             p_pbar_norm_for_heisenberg=None, # Optional momentum for V_H
                             alpha_h=ALPHA_FMD, xi_h_scaled=XI_H_PBAR_SCALED, m_pbar=M_PBAR_MASS):
    """Calculates potential energy components for the antiproton."""
    epsilon = 1e-18
    ZZ_nucleus = p_num_nucleus

    # 1. Antiproton - Nucleus Coulomb Potential (attractive)
    # Nucleus is at origin (0,0,0)
    dist_pbar_nucleus = np.linalg.norm(r_pbar_pos)
    V_pbar_nucleus_coulomb = -ZZ_nucleus / (dist_pbar_nucleus + epsilon)

    # 2. Antiproton - Electrons Coulomb Potential (sum over electrons, attractive)
    V_pbar_electrons_coulomb = 0
    for r_e_pos in r_electrons_cart:
        dist_pbar_electron = np.linalg.norm(r_pbar_pos - r_e_pos)
        V_pbar_electrons_coulomb += -1.0 / (dist_pbar_electron + epsilon) # q_pbar = -1, q_e = -1 => V ~ (-1)*(-1)/r = 1/r ? No, pbar is -1, e is -1. Potential between pbar and e is q_pbar*q_e / r = (-1)*(-1)/r = +1/r.
                                                                        # Wait, your force f_epbar = (ri - r_pbar) / norm³ means V = -1/norm. 
                                                                        # This is if electron is -1 and pbar is +1.
                                                                        # Antiproton is -1, electron is -1. Standard Coulomb is q1*q2/r.
                                                                        # So for pbar(-e) and electron(-e), V = (-e)*(-e)/r = +e^2/r. Attractive potential is negative.
                                                                        # Pbar (-charge) and Electron (-charge) REPEL. Potential is +1/r.
                                                                        # Pbar (-charge) and Nucleus (+Zcharge) ATTRACT. Potential is -Z/r.

    # Let's re-verify charge conventions for potential:
    # V_pbar-nucleus: pbar (-1) and nucleus (+Z). Potential = (-1)*(Z)/r = -Z/r. (Correct)
    # V_pbar-electron: pbar (-1) and electron (-1). Potential = (-1)*(-1)/r = +1/r. (REPULSIVE)
    V_pbar_electrons_coulomb = 0
    for r_e_pos in r_electrons_cart:
        dist_pbar_electron = np.linalg.norm(r_pbar_pos - r_e_pos)
        V_pbar_electrons_coulomb += +1.0 / (dist_pbar_electron + epsilon) # Repulsive

    # 3. Antiproton - Nucleus Heisenberg Potential (Optional)
    # This is similar to the electron-nucleus Heisenberg but for pbar-nucleus.
    # It depends on p_pbar_norm. If p_pbar_norm_for_heisenberg is None, we skip it.
    V_pbar_nucleus_heisenberg = 0
    if p_pbar_norm_for_heisenberg is not None:
        r_pbar_norm = dist_pbar_nucleus # already calculated
        p_p_norm = p_pbar_norm_for_heisenberg
        
        if r_pbar_norm > epsilon and p_p_norm > epsilon and np.abs(xi_h_scaled) > epsilon:
            uu_h = (r_pbar_norm * p_p_norm / xi_h_scaled)**2
            arg_exp_h_base = uu_h**2 
            
            exp_arg_val_h = alpha_h * (1 - arg_exp_h_base)
            # Capping from your dynamics code
            if arg_exp_h_base > 100 and exp_arg_val_h < -300:
                exp_hei_val = 0.0
            elif exp_arg_val_h > 300:
                exp_hei_val = np.exp(300)
            else:
                exp_hei_val = np.exp(exp_arg_val_h)
            
            # Using the form from your dynamics code for v_hei_pbar, with M_PBAR
            # V_H_pbar = (XI_H_scaled^2 / (4 * ALPHA_H * r_pbar_norm^2 * M_PBAR_mass)) * exp_hei_val
            # Note: The mass term M_PBAR here makes V_H very small if M_PBAR is large.
            # Often, for FMD potentials, the ħ²/ (2 * mass) is absorbed into definition of XI_H or prefactor.
            # Let's use the form that appears in your v3_ccs_run for v_hei_pbar, assuming M_STAR there was effectively M_PBAR
            # The potential in the GS code (v3-2) does not have MU in Heisenberg denominator
            # Let's use that simpler form: (XI_H^2 / (4*ALPHA*r^2)) * exp(...)
            V_pbar_nucleus_heisenberg = (xi_h_scaled**2 / (4 * alpha_h * r_pbar_norm**2)) * exp_hei_val


    V_total = V_pbar_nucleus_coulomb + V_pbar_electrons_coulomb + V_pbar_nucleus_heisenberg
    
    return V_total, V_pbar_nucleus_coulomb, V_pbar_electrons_coulomb, V_pbar_nucleus_heisenberg

# --- Main Script ---
# Define file path for Helium ground state
# This should be the path to your '02_He_02e.csv'
HELIUM_GS_FILE = 'GS_alpha_HPC/02_He_02e.csv' # MODIFY AS NEEDED

if not os.path.exists(HELIUM_GS_FILE):
    print(f"Error: Helium ground state file not found at {HELIUM_GS_FILE}")
    print("Please provide the correct path to '02_He_02e.csv'.")
    exit()

p_num_he, e_num_he, r_electrons_he_cart = load_gs_configuration(HELIUM_GS_FILE)
print(f"Loaded Helium (Z={p_num_he}, N_e={e_num_he}) ground state.")
print(f"Electron Cartesian positions (a.u.):\n{r_electrons_he_cart}")

# Define range of XPBAR values (initial -x coordinate of antiproton)
xpbar_values = np.linspace(-14.5, 5.0, 250)
xpbar_values = xpbar_values[np.abs(xpbar_values) > 1e-6]

# --- Scenario 1: Static potential (assume p_pbar_norm = 0 for Heisenberg calc) ---
p_pbar_norm_static = 0.0 
# Note: if p_pbar_norm = 0, uu_h = 0, arg_exp_h_base = 0.
# exp_arg_val_h = alpha_h. V_H becomes (xi_h_scaled**2 / (4 * alpha_h * r_pbar_norm**2)) * exp(alpha_h)
# This is a purely 1/r^2 repulsive potential if p_pbar_norm=0.

potentials_static_p0 = {
    'total': [], 'pbar_nuc_coul': [], 'pbar_e_coul': [], 'pbar_nuc_heis': []
}

for xp_val in xpbar_values:
    r_pbar_current = np.array([-xp_val, 0.0, 0.0]) # Approaching along -x axis
    V_t, V_pn, V_pe, V_ph = calculate_pbar_potential(
        r_pbar_current, p_num_he, r_electrons_he_cart, 
        p_pbar_norm_for_heisenberg=p_pbar_norm_static
    )
    potentials_static_p0['total'].append(V_t)
    potentials_static_p0['pbar_nuc_coul'].append(V_pn)
    potentials_static_p0['pbar_e_coul'].append(V_pe)
    potentials_static_p0['pbar_nuc_heis'].append(V_ph)

# --- Scenario 2: Potential for a typical initial p_pbar_norm ---
# From your v3_ccs_run.py, p0_pbar = np.array([np.sqrt(2 * E0 * M_STAR), 0.0, 0.0])
# Let's pick a typical E0, e.g., E0 = 1.0 a.u.
E0_typical = 1.0 # a.u.
# M_STAR from v3_ccs_run.py: M_PBAR / (1 + (1 / (2 * ZZ)))
# For He, ZZ=2. M_STAR_pbar_sys = M_PBAR_MASS / (1 + (1 / 4)) = M_PBAR_MASS / 1.25
M_STAR_PBAR_SYSTEM = M_PBAR_MASS / (1 + (1.0 / (2.0 * p_num_he))) 
p_pbar_norm_typical = np.sqrt(2 * E0_typical * M_STAR_PBAR_SYSTEM)
print(f"\nFor Scenario 2, using E0={E0_typical} a.u., M_STAR_pbar_sys={M_STAR_PBAR_SYSTEM:.2f} a.u., p_pbar_norm_typical={p_pbar_norm_typical:.2f} a.u.")

potentials_typical_p = {
    'total': [], 'pbar_nuc_coul': [], 'pbar_e_coul': [], 'pbar_nuc_heis': []
}
for xp_val in xpbar_values:
    r_pbar_current = np.array([-xp_val, 0.0, 0.0])
    V_t, V_pn, V_pe, V_ph = calculate_pbar_potential(
        r_pbar_current, p_num_he, r_electrons_he_cart,
        p_pbar_norm_for_heisenberg=p_pbar_norm_typical
    )
    potentials_typical_p['total'].append(V_t)
    potentials_typical_p['pbar_nuc_coul'].append(V_pn)
    potentials_typical_p['pbar_e_coul'].append(V_pe)
    potentials_typical_p['pbar_nuc_heis'].append(V_ph)


# --- Plotting ---
# plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for aesthetics

fig, ax = plt.subplots()

# Plot for Scenario 1 (Typical initial p_pbar_norm for V_H, b=0)
ax.plot(xpbar_values, potentials_typical_p['total'], label='Total', color='gray', linewidth=3)
ax.plot(xpbar_values, potentials_typical_p['pbar_nuc_coul'], label=r'$\bar{p}$-Nucleus (Coulomb)', linestyle='--')
ax.plot(xpbar_values, potentials_typical_p['pbar_e_coul'], label=r'$\bar{p}$-Electrons (Coulomb)', linestyle=':')
ax.plot(xpbar_values, potentials_typical_p['pbar_nuc_heis'], label=r'$\bar{p}$-Nucleus (Heisenberg)', linestyle='-.')
ax.set_xlabel(r'$r$ (a.u.)')
ax.set_ylabel(r'$V_{\bar{p}}$ (a.u.)')
# ax.set_title(rf'He-$\bar{{p}}$, $E_0$={E0_typical} a.u.')
ax.legend()
ax.set_ylim(-p_num_he*3, p_num_he*3) # Adjust ylim
ax.set_xlim(-14.5, 5.0)  # Adjust xlim to focus on negative r

# --- Additions for Antiproton Representation ---
# Add a ball for the antiproton at the starting position (e.g., r = -14, y = 1)
pbar_x = -10
pbar_y = 0.0
ax.plot(pbar_x, pbar_y, 'o', color='purple', markersize=18, zorder=10)
ax.text(pbar_x, pbar_y-0.5, r'$\bar{p}$', fontsize=28, ha='center', va='top')

# Add a right arrow for the momentum (E0) starting at the pbar position
arrow_dx = 3.0  # length of the arrow in x
arrow_dy = 0.0  # no y component
ax.annotate(
    '', xy=(pbar_x + arrow_dx, pbar_y), xytext=(pbar_x, pbar_y),
    arrowprops=dict(arrowstyle='->', color='black', lw=3), zorder=11
)
# Label for E0 next to the arrow
ax.text(pbar_x + arrow_dx + 0.2, pbar_y - 0.5, r'$E_0$', fontsize=28, ha='center', va='top')

# --- Additions for Atom Representation ---
# Add a ball for the atom core
atom_x = 0.0
atom_y = 0.0
ax.plot(atom_x, atom_y, 'o', color='royalblue', markersize=20, zorder=10)
ax.text(atom_x-1.1, atom_y+0.9, r'$^4_2$He', fontsize=24, ha='center', va='top')
# Add electrons as small circles around the nucleus
for r_e_pos in r_electrons_he_cart:
    ax.plot(r_e_pos[0], r_e_pos[1], 'o', color='orange', markersize=10, zorder=10)

# Plot parameters
plt.tick_params(
    axis='both', which='both', direction='in', top=True, right=True
)
plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('pbar_pot_ini.svg')
plt.close()

# --- Scenario 3: Potential for varying impact parameters ---
# Let's say impact parameters b = [0.0, 1.0, 2.0, 3.0] (in a.u.)
impact_parameters = np.array([0.0, 1.0, 2.0, 3.0])
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=impact_parameters.min(), vmax=impact_parameters.max())
colors = cmap(np.linspace(0, 1, len(impact_parameters)))

fig, ax = plt.subplots()

for b, color in zip(impact_parameters, colors):
    potentials = []
    for xp_val in xpbar_values:
        r_pbar_current = np.array([-xp_val, b, 0.0])  # Impact parameter in y
        V_t, V_pn, V_pe, V_ph = calculate_pbar_potential(
            r_pbar_current, p_num_he, r_electrons_he_cart,
            p_pbar_norm_for_heisenberg=p_pbar_norm_typical
        )
        potentials.append(np.abs(V_t))  # Take absolute value here
    ax.plot(
        xpbar_values, potentials, color=color, linewidth=2,
        label=f'$b$={b:.2f}'
    )

ax.set_xlabel(r'$r$ (a.u.)')
ax.set_ylabel(r'$|V_{\bar{p}, Total}|$ (a.u.)')
# ax.set_title(rf'He-$\bar{{p}}$, $E_0$={E0_typical} a.u.')
ax.set_yscale('log')
ax.set_ylim(1e-6, None)
ax.set_xlim(-14.0, 5.0)

plt.tick_params(
    axis='both', which='both', direction='in', top=True, right=True
)
plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)

# Colorbar for impact parameter
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r'$b$ (a.u.)')
cbar.ax.tick_params(
    axis='y', which='both', direction='in', length=12, width=2,
    right=True, left=True, labelsize=16
)

# --- Inset plot for impact parameter geometry ---
inset_ax = inset_axes(ax, width="40%", height="40%", loc='upper left', borderpad=2)
theta = np.linspace(0, 2 * np.pi, 100)
for b, color in zip(impact_parameters, colors):
    x = b * np.cos(theta)
    y = b * np.sin(theta)
    inset_ax.plot(x, y, color=color, linewidth=2)
    inset_ax.plot(0, b, 'o', color=color, markersize=8)
inset_ax.plot(0, 0, 'ko', markersize=10, label='Nucleus')
inset_ax.set_aspect('equal')
inset_ax.set_xlim(-impact_parameters.max()-0.5, impact_parameters.max()+0.5)
inset_ax.set_ylim(-impact_parameters.max()-0.5, impact_parameters.max()+0.5)
tick_vals = np.arange(-impact_parameters.max(), impact_parameters.max()+1, 1)
inset_ax.set_xticks(tick_vals)
inset_ax.set_yticks(tick_vals)
inset_ax.set_xlabel(r'$x$ (a.u.)', fontsize=14)
inset_ax.set_ylabel(r'$y$ (a.u.)', fontsize=14)
inset_ax.tick_params(axis='both', which='both', direction='in', labelsize=12, length=5, width=1.2)
inset_ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.7)

# Draw a double arrow for b between (0,0) and (0,3)
inset_ax.annotate(
    '', xy=(0, 3), xytext=(0, 0),
    arrowprops=dict(arrowstyle='<->', color='black', lw=2)
)
# Place the label "b" next to the arrow, at (0.2, 1.5)
inset_ax.text(0.2, 1.5, r'$b$', fontsize=18, va='center', ha='left')

plt.tight_layout()
plt.savefig('pbar_pot_b.svg')
plt.close()
