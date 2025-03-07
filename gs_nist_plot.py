import os
import re
import pandas as pd
import matplotlib.pyplot as plt


# Read the results from the CSV file
df = pd.read_csv('results.csv')

# Directory containing the files
directory = 'c:/Users/propietario/Documents/Antiprotonic-atoms/LDA/neutrals'

# Extract Etot values and atomic numbers from the files
minus_etot_values = []
atomic_numbers = []

for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        atomic_number = int(re.findall(r'\d+', filename)[0])
        with open(os.path.join(directory, filename), 'r') as file:
            for line in file:
                if "Etot" in line:
                    etot = float(line.split('=')[1].strip())
                    minus_etot_values.append(-etot)
                    atomic_numbers.append(atomic_number)
                    break

# Plot Etot vs Atomic Number
labelfontsize = 18
tickfontsize = 14
plt.figure(figsize=(10, 6))
# NIST data
plt.plot(atomic_numbers, minus_etot_values, 'o', markersize=3, linestyle='None', label='NIST-LDA')

# Calculated data
for optimizer in df['optimizer'].unique():
    subset = df[df['optimizer'] == optimizer]
    e_num_values = subset['e_num'].values
    ground_state_energies = -subset['ground_state_energy'].values
    plt.plot(e_num_values, ground_state_energies, 'o', markersize=3, linestyle='None', label=optimizer)

plt.xlabel(r'$Z$', fontsize=labelfontsize)
plt.ylabel(r'$-E_{GS}$ (a.u.)', fontsize=labelfontsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.xlim(0, 5)
plt.ylim(1e-1, 1e2)
# plt.yscale('log')
plt.grid(True)
plt.legend()
# plt.savefig('E_GS_vs_Z.png')
plt.show()