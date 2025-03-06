import pandas as pd
import matplotlib.pyplot as plt


# Read the results from the CSV file
df = pd.read_csv('results.csv')

# Plot the ground state energy for each e_num and optimizer
plt.figure(figsize=(10, 6))
for optimizer in df['optimizer'].unique():
    subset = df[df['optimizer'] == optimizer]
    plt.plot(subset['e_num'], subset['ground_state_energy'], label=optimizer)

plt.xlabel('Number of Electrons (e_num)')
plt.ylabel('Ground State Energy')
plt.legend()
plt.grid(True)
plt.show()