import pandas as pd
import matplotlib.pyplot as plt

# Load data
cross_df = pd.read_csv('cross_sections.csv')
initial_df = pd.read_csv('initial_states.csv')
final_df = pd.read_csv('final_states.csv')

# Plot 1: Capture cross sections vs. Energy
plt.figure()
plt.plot(cross_df['Energy'], cross_df['Sigma_total'], label='Total', marker='o')  # Added marker
plt.plot(cross_df['Energy'], cross_df['Sigma_single'], label='Single Capture', marker='s')  # Added marker
plt.plot(cross_df['Energy'], cross_df['Sigma_double'], label='Double Capture', marker='^')  # Added marker
plt.xlabel('Initial Antiproton Energy (a.u.)')
plt.ylabel('Capture Cross Section (a.u.)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Initial States (L_initial vs. E_initial)
plt.figure()
# Exclude 'none' type
filtered_initial_df = initial_df[initial_df['type'] != 'none']
for capture_type in filtered_initial_df['type'].unique():
    subset = filtered_initial_df[filtered_initial_df['type'] == capture_type]
    plt.scatter(subset['L_initial'], subset['E_initial'], label=capture_type, alpha=0.7)  # Swapped axes
plt.xlabel('Initial Angular Momentum L (a.u.)')  # Updated label
plt.ylabel('Initial Antiproton Energy (a.u.)')  # Updated label
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Final States (L_final vs. E_final)
plt.figure()
# Exclude 'none' type
filtered_final_df = final_df[final_df['type'] != 'none']
for capture_type in filtered_final_df['type'].unique():
    subset = filtered_final_df[filtered_final_df['type'] == capture_type]
    plt.scatter(subset['L_final'], subset['E_final'], label=capture_type, alpha=0.7)  # Swapped axes
plt.xlabel('Final Angular Momentum L (a.u.)')  # Updated label
plt.ylabel('Final Antiproton Energy (a.u.)')  # Updated label
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
