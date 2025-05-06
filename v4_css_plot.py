import pandas as pd
import matplotlib.pyplot as plt

# Load data
cross_df = pd.read_csv('cross_sections.csv')
initial_df = pd.read_csv('initial_states.csv')
final_df = pd.read_csv('final_states.csv')

# Plot 1: Capture cross sections vs. Energy
plt.figure()
plt.plot(cross_df['Energy'], cross_df['Sigma_total'], label='Total')
plt.plot(cross_df['Energy'], cross_df['Sigma_single'], label='Single Capture')
plt.plot(cross_df['Energy'], cross_df['Sigma_double'], label='Double Capture')
plt.xlabel('Initial Antiproton Energy (a.u.)')
plt.ylabel('Capture Cross Section (a₀²)')
plt.title('Capture Cross Sections vs. Energy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Initial States (E_initial vs. L_initial)
plt.figure()
for capture_type in initial_df['type'].unique():
    subset = initial_df[initial_df['type'] == capture_type]
    plt.scatter(subset['E_initial'], subset['L_initial'], label=capture_type, alpha=0.7)
plt.xlabel('Initial Antiproton Energy (a.u.)')
plt.ylabel('Initial Angular Momentum L (a.u.)')
plt.title('Initial (E, L) Distribution by Capture Type')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Final States (E_final vs. L_final)
plt.figure()
for capture_type in final_df['type'].unique():
    subset = final_df[final_df['type'] == capture_type]
    plt.scatter(subset['E_final'], subset['L_final'], label=capture_type, alpha=0.7)
plt.xlabel('Final Antiproton Energy (a.u.)')
plt.ylabel('Final Angular Momentum L (a.u.)')
plt.title('Final (E, L) Distribution by Capture Type')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
