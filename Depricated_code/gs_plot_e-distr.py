import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the results from the CSV file
df = pd.read_csv('results.csv')

# Initialize dictionaries to store r0 and p0 values for each optimizer
r0_values = {}
p0_values = {}
e_num_values = {}

for optimizer in df['optimizer'].unique():
    subset = df[df['optimizer'] == optimizer]
    
    r0_values[optimizer] = []
    p0_values[optimizer] = []
    e_num_values[optimizer] = []

    for index, row in subset.iterrows():
        e_num = int(row['e_num'])
        optimal_config = np.fromstring(row['optimal_configuration'].strip('[]'), sep=' ')
        r0 = optimal_config[:e_num]
        p0 = optimal_config[3 * e_num:4 * e_num]
        
        # Store the values
        r0_values[optimizer].append((e_num, r0))
        p0_values[optimizer].append((e_num, p0))
        e_num_values[optimizer].append(e_num)

# Plot r0 and p0 for each optimizer
for optimizer in df['optimizer'].unique():
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    for e_num, r0 in r0_values[optimizer]:
        for i, value in enumerate(r0):
            marker = '^' if i % 2 == 0 else 'v'
            axs[0].plot(e_num, value, marker=marker, linestyle='None', label=f'e_num={e_num}' if i == 0 else "")
    axs[0].set_ylabel('r0')
    axs[0].set_title(f'Optimal r0 for {optimizer}')
    axs[0].grid(True)
    axs[0].legend()

    for e_num, p0 in p0_values[optimizer]:
        for i, value in enumerate(p0):
            marker = '^' if i % 2 == 0 else 'v'
            axs[1].plot(e_num, value, marker=marker, linestyle='None', label=f'e_num={e_num}' if i == 0 else "")
    axs[1].set_xlabel('e_num')
    axs[1].set_ylabel('p0')
    axs[1].set_title(f'Optimal p0 for {optimizer}')
    axs[1].grid(True)
    axs[1].legend()

    plt.savefig(f'fig_optimal_r0_p0_{optimizer}.png')
    plt.close(fig)  # Close the figure to avoid displaying it
