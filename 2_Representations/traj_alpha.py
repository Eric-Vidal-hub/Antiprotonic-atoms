import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp

# Set default font size and line width
plt.rcParams['font.size'] = 26
plt.rcParams['lines.linewidth'] = 3
mpl.rc('text', usetex=False)


def hamiltonian_equations(t, y, m_param, xi_param, hbar_param, alpha_param):
    r, p = y
    # Hamiltonian equations of motion
    # r = position, p = momentum
    # precalculated terms
    u_val = (p * r / (hbar_param * xi_param))**4
    exp_term = np.exp(alpha_param * (1 - u_val))

    # dr/dt = ∂H/∂p
    # Original H = p^2/m + (xi^2*hbar^2 / (m*r^2*2*alpha)) * exp(-alpha*(u-1))
    # dHdp_term1 = 2*p/m_param
    # dHdp_term2_coeff = (xi_param**2 * hbar_param**2) / (m_param * r**2 * 2 * alpha_param)
    # d_exp_du = exp_term * (-alpha_param)
    # du_dp = 2 * (p * r**2) / (hbar_param**2 * xi_param**2)
    # dr_dt = dHdp_term1 + dHdp_term2_coeff * d_exp_du * du_dp
    # This is equivalent to the simpler form:
    dr_dt = (p / m_param) * (2 - exp_term)

    # dp/dt = -∂H/∂r
    # dHdr_term_pot_r_cubed = -2 * (xi_param**2 * hbar_param**2) / (m_param * r**3 * 2 * alpha_param) * exp_term
    # d_exp_du_for_r = exp_term * (-alpha_param)
    # du_dr = 2 * (p**2 * r) / (hbar_param**2 * xi_param**2)
    # dHdr_term_pot_exp_deriv = (xi_param**2 * hbar_param**2) / (m_param * r**2 * 2 * alpha_param) * d_exp_du_for_r * du_dr
    # dp_dt = -(dHdr_term_pot_r_cubed + dHdr_term_pot_exp_deriv)
    # This is equivalent to the simpler form:
    factor1 = 1 / (m_param * r)
    factor2 = (xi_param**2 * hbar_param**2) / (alpha_param * r**2) + p**2
    dp_dt = factor1 * exp_term * factor2
    
    return [dr_dt, dp_dt]


# Parameters
m = 1.0
xi = 1.0
hbar = 1.0

alphas_to_plot = [1.0, 1.5, 2.5]
colors = [plt.cm.viridis((a-1)/(2.5-1)) for a in alphas_to_plot]
markers = ['s', 'o', '^']
linestyles = ['-', '--', ':']

plt.figure(figsize=(12, 8))

# Initial conditions:
# We need to choose an energy. The plot shows different trajectories,
# which implies different energies or different r_min for p=0.
# Since the shape only depends on alpha, we can pick a consistent way to start.
# Let's start near a turning point where p is small.
# If p_init = 0, then dr/dt = 0, which is not good for integration start.
# Let's choose an r_min (closest approach where p would be 0 if no oscillation)
# and start slightly away from it with a small p.

# To make the "loops" visible, the energy needs to be in a certain range.
# The energy is H = p^2/m + Vp. If p=0 at r_min, E = Vp(r_min, 0)
# Vp(r,0) = (xi^2*hbar^2 / (m*r^2*2*alpha)) * np.exp(alpha)
# Let E_scaled = E * m / (xi^2*hbar^2) = 1/(2*alpha*r_min_scaled^2) * exp(alpha)
# where r_min_scaled = r_min / (some length scale, not directly used here)

# Let's try starting all trajectories such that if they reached p=0 without oscillation,
# it would be at some r_min. Let's aim for r_min ~ 0.5 in arbitrary units.
# This sets an energy for each alpha.

# It's easier to pick an initial r (e.g., r_start_large) and p (e.g., p_start_negative)
# that defines a suitable energy to see the loops.

# The paper (page 345) implies they find r_0 (where p=0) from f(0) = x_0 * y_E.
# And x_c (closest approach) from f'(x_c*y_c)+1 = 0, implying x_c*y_c=1.
# This is for the analytic consideration of H = m^-1 [y + x^-1 f(xy)].
# This might not directly translate to initial conditions for the ODEs of H = p^2/m + V_p.

# Let's choose an energy E. The trajectories will be contours of H(r,p) = E.
# We need to pick an E that results in the oscillatory behavior.
# From Fig 1, the "potential well" depth for Vp (if it were only r-dependent) is significant.
# The oscillation happens when kinetic energy can be converted to potential and back in a way
# that r decreases, p goes to 0, r increases, p reverses, etc.

# Let's try to pick initial conditions that are "coming in from infinity"
r_init_large = 3.0  # "Large" r in arbitrary units
# We need a p_init that gives enough energy to see the loop.
# The energy H = p_init^2/m + Vp(r_init_large, p_init)
# Vp at large r with finite p will be small. So E ~ p_init^2/m.
# Let's set an energy E. E.g., E = 1.0 (arbitrary units, assuming m, xi, hbar = 1)
# Then p_init ~ sqrt(m*E). Let p_init = -sqrt(m*E_target) for incoming.
E_target = 0.8  # Arbitrary energy units, adjust this to get good loops

# Time span for integration
t_span = [0, 10]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

for i, (alpha_val, color, marker, ls) in enumerate(zip(alphas_to_plot, colors, markers, linestyles)):
    # For this alpha, Vp(r,0) = (1 / (2*alpha_val*r^2)) * exp(alpha_val)
    # If E_target = Vp(r_min_eff, 0), then
    # r_min_eff_sq = exp(alpha_val) / (2*alpha_val*E_target)
    # r_min_eff = np.sqrt(r_min_eff_sq)
    # print(f"Alpha: {alpha_val}, Effective r_min for E={E_target}: {r_min_eff:.2f}")
    
    # Heuristic p_initial based on E_target
    # This assumes Vp is small at r_init_large, which might not be true if p_init is large.
    p_initial = -np.sqrt(1.0 * E_target)    # Incoming, m=1
    
    # Adjust p_initial if Vp(r_init_large, p_initial) is too large or complex to estimate
    # Let's try a fixed p_initial that is known to produce loops for alpha=2.5
    if alpha_val == 2.5:
        p_initial_val = -0.85     # Tuned by trial and error for alpha=2.5
    elif alpha_val == 1.5:
        p_initial_val = -0.75     # Needs tuning
    else:   # alpha_val == 1.0
        p_initial_val = -0.65     # Needs tuning
      
    y0 = [r_init_large, p_initial_val]

    sol = solve_ivp(hamiltonian_equations, t_span, y0, args=(m, xi, hbar, alpha_val),
                    dense_output=True, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)

    r_traj = sol.y[0]
    p_traj = sol.y[1]

    # Filter out parts where r might become negative or too small causing instability
    valid_indices = r_traj > 0.01   # Adjust threshold as needed
    
    # Plot trajectory
    plt.plot(r_traj[valid_indices], p_traj[valid_indices],
             color=color, linestyle=ls, linewidth=3, label=f'$\alpha={alpha_val}$')

    # Add markers
    plt.plot(r_traj[valid_indices][::50], p_traj[valid_indices][::50],
             marker=marker, markersize=10, fillstyle='none', color=color, linestyle='None')


# Text labels for alpha values (adjust as needed)
textfontsize = 20
plt.text(1.5, 0.4, r'$\alpha=1.0$', fontsize=textfontsize, color=colors[0])
plt.text(1.0, 0.6, r'$\alpha=1.5$', fontsize=textfontsize, color=colors[1])
plt.text(0.6, 0.8, r'$\alpha=2.5$', fontsize=textfontsize, color=colors[2])

# Labels
plt.xlabel(r'Position (arb. units)', fontsize=26)
plt.ylabel(r'Momentum (arb. units)', fontsize=26)

# Set limits and aspect
plt.xlim(0, 2.5)
plt.ylim(-1.0, 1.0)
plt.gca().set_aspect('auto', adjustable='box')

# Ticks and formatting
plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5], fontsize=36)
plt.yticks([-1.0, -0.5, 0, 0.5, 1.0], fontsize=36)
ax = plt.gca()
xlabels = ["" if tick == 0 else f"{tick:.1f}" for tick in ax.get_xticks()]
ylabels = ["" if tick == 0 else f"{tick:.1f}" for tick in ax.get_yticks()]
ax.set_xticklabels(xlabels, fontsize=36)
ax.set_yticklabels(ylabels, fontsize=36)
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=36)
plt.minorticks_on()

plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.show()
