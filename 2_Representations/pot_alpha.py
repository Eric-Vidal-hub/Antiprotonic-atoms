import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


# Directory to save the output figure
output_dir = os.path.join(os.path.dirname(__file__), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set the style for the plot
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
# Set the style for the grid
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 1
# Set the style for the ticks
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# Set the style for the axes
plt.rcParams['axes.linewidth'] = 2
# Set the style for the figure
plt.rcParams['figure.facecolor'] = 'white'
# Set the style for the text
plt.rcParams['text.usetex'] = False


# Scaled relative momentum X = pr / (ħξ)
X = np.linspace(0, 1.6, 400)
ke_scaled = X**2
alphas = np.linspace(3.4, 4.6, 7)
cmap = plt.cm.viridis
norm = plt.Normalize(alphas.min(), alphas.max())

fig, ax = plt.subplots()

ax.plot(X, ke_scaled, color='black', label=r'$p^2/m$')

for alpha in alphas:
    color = cmap(norm(alpha))
    vp_scaled = (1 / (4 * alpha)) * np.exp(alpha * (1 - X**4))
    total_scaled = ke_scaled + vp_scaled
    ax.plot(X, vp_scaled, color=color, linestyle=':', alpha=0.8)
    ax.plot(
        X, total_scaled, color=color, linestyle='-', alpha=0.8,
        label=f'Total ($\\alpha$={alpha:.2f})' if alpha == alphas[0] else None
    )

textfontsize = 28
ax.text(0.5, 0.5, r'$p^2/\mu$', fontsize=textfontsize, ha='left', color='black')
ax.text(
    0.4, 1.7, r'$v_p$', fontsize=textfontsize, ha='left',
    color=cmap(norm(alphas[0]))
)
ax.text(
    0.15, 1.8, r'Total', fontsize=textfontsize, ha='center',
    color=cmap(norm(alphas[0]))
)

ax.set_xlabel(r'$pr/\xi$')
ax.set_ylabel(r'Energy $\cdot \; \mu r^2/\xi^2$')
ax.set_xlim(0, 1.2)
# ax.set_ylim(0, 4.5)
ax.set_aspect('auto', adjustable='box')

plt.tick_params(
    axis='both', which='both', direction='in', top=True, right=True
)

plt.grid(
    True, which='both', linestyle='--', linewidth=1, alpha=0.5
)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r'$\alpha$')
cbar.ax.tick_params(
    axis='y', which='both', direction='in', length=12, width=2,
    right=True, left=True, labelsize=26
)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pot_alpha.svg'))
