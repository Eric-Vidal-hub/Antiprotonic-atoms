import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import solve_ivp
import os
from scipy.signal import argrelextrema


# Output directory (optional, for saving)
output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Style settings (copied from pot_alpha.py)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 36
plt.rcParams['legend.fontsize'] = 26
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


def hamiltonian_equations(t, yy, m_param, xi_param, alpha_param):
    rr, pp = yy
    # Hamiltonian equations of motion
    # r = position, p = momentum
    # precalculated terms
    u_val = pp * rr / xi_param
    exp_term = np.exp(alpha_param * (1 - u_val**4))

    # Original H = p^2/m + (xi^2 / (4*alpha * m*r^2)) * exp(alpha*(1-u^4))
    # dr/dt = ∂H/∂p
    dr_dt = (pp / m_param) * (2 - u_val**2 * exp_term)

    # dp/dt = -∂H/∂r
    dp_dt = (1 / (2 * alpha_param) + u_val**4) * (xi_param**2 / (m_param * rr**3)) * exp_term
    return [dr_dt, dp_dt]


# Parameters
m = 1.0
xi = 1.0

alphas_to_plot = np.linspace(2.5, 4.5, 7)
cmap = plt.cm.viridis
norm = plt.Normalize(alphas_to_plot.min(), alphas_to_plot.max())
colors = [cmap(norm(a)) for a in alphas_to_plot]
markers = ['s', 'o', '^', 'D', 'v', '>', '<']
linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 1))]

fig, ax = plt.subplots()

# Initial conditions:
# We need to choose an energy. The plot shows different trajectories,
# which implies different energies or different x_c (closest distance) for p=0.
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

for i, (alpha_val, color) in enumerate(zip(alphas_to_plot, colors)):
    p_initial_val = -0.85   # + 0.2 * (alpha_val - alphas_to_plot[0])
    y0 = [r_init_large, p_initial_val]
    sol = solve_ivp(
        hamiltonian_equations, t_span, y0, args=(m, xi, alpha_val),
        dense_output=True, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)
    r_traj = sol.y[0]
    p_traj = sol.y[1]
    valid_indices = r_traj > 0.01
    ax.plot(
        r_traj[valid_indices], p_traj[valid_indices],
        color=color, linewidth=3
    )

    # For the highest alpha, add vertical lines and labels
    if i == len(alphas_to_plot) - 1:
        # Closest approach (minimum r)
        r_c = r_traj[valid_indices][np.argmin(r_traj[valid_indices])]
        p_c = p_traj[valid_indices][np.argmin(r_traj[valid_indices])]
        # Black vertical line at r_c
        ax.axvline(r_c, color='black', linestyle='--', linewidth=2)
        # Red cross at (r_c, p_c)
        ax.plot(r_c, p_c, 'x', color='red', markersize=14, markeredgewidth=3)
        # Red label for r_c
        ax.text(
            r_c - 0.08, p_c, r'$r_c$', color='red', fontsize=28,
            ha='center', va='bottom'
        )

        # Red cross at (r_c, -p_c)
        ax.plot(r_c, -p_c, 'x', color='red', markersize=14, markeredgewidth=3)
        # Red label for r_c
        ax.text(
            r_c - 0.08, -p_c, r'$r_c$', color='red', fontsize=28,
            ha='center', va='bottom'
        )


        # Where p crosses zero (find first crossing)
        sign_change = np.where(np.diff(np.sign(p_traj[valid_indices])))[0]
        if len(sign_change) > 0:
            idx0 = sign_change[0]
            # Linear interpolation for more accurate r_0
            p1, p2 = p_traj[valid_indices][idx0], p_traj[valid_indices][idx0+1]
            r1, r2 = r_traj[valid_indices][idx0], r_traj[valid_indices][idx0+1]
            r_0 = r1 - p1 * (r2 - r1) / (p2 - p1)
            # Interpolated p should be zero, but for plotting, use 0
            ax.plot(r_0, 0, 'x', color='royalblue', markersize=14, markeredgewidth=3)
            # Blue label for r_0
            ax.text(
                r_0 + 0.08, 0.08, r'$r_0$', color='royalblue', fontsize=28,
                ha='center', va='bottom'
            )

ax.set_xlabel(r'$r$ (a. u.)')
ax.set_ylabel(r'$p$ (a. u.)')
ax.set_xlim(0.8, 3.0)
ax.set_ylim(-1.0, 1.0)
ax.set_aspect('auto', adjustable='box')

plt.tick_params(
    axis='both', which='both', direction='in', top=True, right=True
)
plt.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)

# Colorbar for alpha
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r'$\alpha$')
cbar.ax.tick_params(
    axis='y', which='both', direction='in', length=12, width=2,
    right=True, left=True, labelsize=26
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'traj_alpha.svg'))
plt.show()
