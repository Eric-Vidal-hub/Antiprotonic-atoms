import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the results from the CSV file
df = pd.read_csv('results.csv')

# Directory containing the files
directory = 'c:/Users/propietario/Documents/Antiprotonic-atoms/LDA/neutrals'

# Extract Etot values and atomic numbers from the files
nist_etot_values = []
atomic_numbers = []

for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        atomic_number = int(re.findall(r'\d+', filename)[0])
        with open(os.path.join(directory, filename), 'r') as file:
            for line in file:
                if "Etot" in line:
                    etot = float(line.split('=')[1].strip())
                    nist_etot_values.append(-etot)
                    atomic_numbers.append(atomic_number)
                    break

# Create a dictionary for NIST data for easy lookup
nist_data = dict(zip(atomic_numbers, nist_etot_values))

# Plot error comparison
labelfontsize = 18
tickfontsize = 14
plt.figure(figsize=(10, 6))

# Calculated data
for optimizer in df['optimizer'].unique():
    subset = df[df['optimizer'] == optimizer]
    e_num_values = subset['e_num'].values
    ground_state_energies = -subset['ground_state_energy'].values
    
    # Calculate errors
    errors = []
    for e_num, gs_energy in zip(e_num_values, ground_state_energies):
        if e_num in nist_data:
            error = 100 * np.abs((gs_energy - nist_data[e_num]) / nist_data[e_num])
            errors.append(error)
        else:
            errors.append(None)  # for cases where NIST data is not available
    
    # Filter out None values
    filtered_e_num_values = [e_num for e_num, error in zip(e_num_values, errors) if error is not None]
    filtered_errors = [error for error in errors if error is not None]
    
    plt.plot(filtered_e_num_values, filtered_errors, 'o', markersize=3, linestyle='None', label=optimizer)

plt.xlabel(r'$Z$', fontsize=labelfontsize)
plt.ylabel(r'$E_{GS}$ Relative Error (%)', fontsize=labelfontsize)
plt.xticks(fontsize=tickfontsize)
plt.yticks(fontsize=tickfontsize)
plt.grid(True)
plt.legend()
plt.savefig('fig_rel-error_vs_z.png')
plt.show()