import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set default font size
plt.rcParams['font.size'] = 26

# Set the style of the plots
mpl.rc('text', usetex=False)

# Scaled relative momentum X = pr / (ħξ)
X = np.linspace(0, 1.6, 400)

# Kinetic energy is the same for all curves
ke_scaled = X**2

# Alphas to iterate over
alphas = np.linspace(3.4, 4.6, 7)  # 7 steps from 3.5 to 4.5

# Colormap for fading effect
cmap = plt.cm.viridis
norm = plt.Normalize(alphas.min(), alphas.max())

plt.figure(figsize=(12, 8))

# Plot kinetic energy (same for all)
plt.plot(X, ke_scaled, color='black', linewidth=2, label=r'$p^2/m$')

# Plot for each alpha
for alpha in alphas:
    color = cmap(norm(alpha))
    vp_scaled = (1 / (4 * alpha)) * np.exp(alpha * (1 - X**4))
    total_scaled = ke_scaled + vp_scaled
    plt.plot(X, vp_scaled, color=color, linestyle=':', linewidth=1, alpha=0.8)
    plt.plot(X, total_scaled, color=color, linestyle='-', linewidth=1.5, alpha=0.8,
             label=f'Total ($\\alpha$={alpha:.2f})' if alpha == alphas[0] else None)

# Relocated text labels for the new x/y limits (previous positions)
textfontsize = 20
plt.text(0.55, 0.5, r'$p^2/m$', fontsize=textfontsize, ha='left', color='black')
plt.text(0.4, 1.7, r'$v_p$', fontsize=textfontsize, ha='left', color=cmap(norm(alphas[0])))
plt.text(0.2, 2.3, r'Total', fontsize=textfontsize, ha='center', color=cmap(norm(alphas[0])))

# Labels
plt.xlabel(r'$pr/\hbar\xi$')
plt.ylabel(r'Energy $\cdot \; mr^2/(\hbar^2\xi^2)$')

# Set new x limit and keep y limit
plt.xlim(0, 1.2)
plt.ylim(0, 4.5)
plt.gca().set_aspect('auto', adjustable='box')

# Add a single zero at the bottom left corner, slightly offset to avoid overlap
plt.text(-0.08, -0.22, "0.0", ha='left', va='bottom')

# Set ticks as before
plt.xticks(np.arange(0, 1.3, 0.2))
plt.yticks(np.arange(0, 4.6, 0.5))

# Remove the first (zero) tick label from both axes, and always show one decimal
ax = plt.gca()
xlabels = ["" if tick == 0 else f"{tick:.1f}" for tick in ax.get_xticks()]
ylabels = ["" if tick == 0 else f"{tick:.1f}" for tick in ax.get_yticks()]
ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)

# Ticks inwards and on all sides, with minor ticks
plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)
plt.minorticks_on()

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

# Colorbar for alpha
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, pad=0.02)
cbar.set_label(r'$\alpha$')

plt.tight_layout()
plt.show()