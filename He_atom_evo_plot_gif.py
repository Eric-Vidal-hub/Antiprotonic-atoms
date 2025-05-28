import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import animation
import matplotlib.patches as patches
from matplotlib.colors import to_rgba

# Output directory (optional, for saving)
output_dir = os.path.join(os.path.dirname(__file__), 'He_atom_evo_output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# To load later:
data = np.load(os.path.join(output_dir, 'trajectory_data.npz'))
t_arr = data['t_arr']
y_arr = data['y_arr']
e_num = int(data['e_num'])

# Compute position modulus for each electron at each time step
position_modulus = []
for i in range(e_num):
    # Extract x, y, z for electron i at all times
    x = y_arr[3*i, :]
    y = y_arr[3*i+1, :]
    z = y_arr[3*i+2, :]
    modulus = np.sqrt(x**2 + y**2 + z**2)
    position_modulus.append(modulus)

# Style settings (copied from pot_alpha.py)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['font.size'] = 26
plt.rcParams['axes.labelsize'] = 26
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

# Animation settings
frame_step = 150  # Adjust this for smoother/faster animation
# Create a 3D animation of electron trajectories
frames = range(0, len(t_arr), frame_step)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Prepare lines and current position markers for each electron
lines = [ax.plot([], [], [], label=f'Electron {i+1}')[0] for i in range(e_num)]
# Use a darker color for the current position marker
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
current_markers = [
    ax.plot([], [], [], marker='o', markersize=16, color=to_rgba(colors[i % len(colors)], 0.9),
            markeredgecolor='black', linestyle='None', zorder=5)[0]
    for i in range(e_num)
]

# Set axis limits (adjust as needed)
inf_lim = -1.2
sup_lim = 1.2
ax.set_xlim(inf_lim, sup_lim)
ax.set_ylim(inf_lim, sup_lim)
ax.set_zlim(inf_lim, sup_lim)
ax.set_xlabel('x (a.u.)', labelpad=18)
ax.set_ylabel('y (a.u.)', labelpad=24)
ax.set_zlabel('z (a.u.)', labelpad=30)
ax.tick_params(axis='z', pad=12)  # Increase pad from default (~6) to 12 or more
ax.zaxis.label.set_verticalalignment('bottom')
ax.zaxis.set_label_coords(1.05, 0.5)  # Push z label to the right

# Optional: Tweak x and y label coords for balance
ax.xaxis.set_label_coords(0.5, -0.12)  # Slightly below center
ax.yaxis.set_label_coords(-0.15, 0.5)  # Slightly left of center
ax.legend()

# Add a time bar above the plot
bar_ax = fig.add_axes([0.15, 0.92, 0.7, 0.03])  # y-position moved near top
bar_ax.set_xlim(t_arr[0], t_arr[-1])
bar_ax.set_ylim(0, 1)
bar_ax.axis('off')
bar_patch = patches.Rectangle((t_arr[0], 0), 0, 1, color='royalblue')
bar_ax.add_patch(bar_patch)
# Place the time label at the center of the bar
time_text = bar_ax.text(
    0.4 * (t_arr[0] + t_arr[-1]), -1, '', va='center', ha='center', fontsize=18, color='black'
)

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    for marker in current_markers:
        marker.set_data([], [])
        marker.set_3d_properties([])
    bar_patch.set_width(0)
    time_text.set_text('')
    return lines + current_markers + [bar_patch, time_text]

def animate(frame_idx):
    frame = frames[frame_idx]
    for i, line in enumerate(lines):
        x = y_arr[3*i, :frame]
        y = y_arr[3*i+1, :frame]
        z = y_arr[3*i+2, :frame]
        line.set_data(x, y)
        line.set_3d_properties(z)
        # Update current position marker
        if frame > 0:
            current_markers[i].set_data([x[-1]], [y[-1]])
            current_markers[i].set_3d_properties([z[-1]])
        else:
            current_markers[i].set_data([], [])
            current_markers[i].set_3d_properties([])
    bar_patch.set_width(t_arr[frame] - t_arr[0])
    time_text.set_text(f't = {t_arr[frame]:.2f} a.u.')
    return lines + current_markers + [bar_patch, time_text]

ani = animation.FuncAnimation(
    fig, animate, frames=len(frames), init_func=init,
    interval=40, blit=True
)

# To save the animation as mp4 (requires ffmpeg):
ani.save(os.path.join(output_dir, 'trajectory_evolution.gif'), writer='pillow', fps=25)

plt.show()