import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('results.csv')

for optimizer in df['optimizer'].unique():
    subset = df[df['optimizer'] == optimizer]
    for index, row in subset.iterrows():
        e_num = int(row['e_num'])
        optimal_config = np.fromstring(row['optimal_configuration'].strip('[]'), sep=' ')
        r0 = optimal_config[:e_num]
        p0 = optimal_config[3 * e_num:4 * e_num]

        # Plot r0
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, e_num + 1), r0, 'o-', label=f'{optimizer} - r0')
        plt.xlabel('Electron Index')
        plt.ylabel('r0')
        plt.title(f'Optimal r0 for {optimizer} (e_num={e_num})')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'optimal_r0_{optimizer}_e_num_{e_num}.png')
        plt.show()

        # Plot p0
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, e_num + 1), p0, 'o-', label=f'{optimizer} - p0')
        plt.xlabel('Electron Index')
        plt.ylabel('p0')
        plt.title(f'Optimal p0 for {optimizer} (e_num={e_num})')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'optimal_p0_{optimizer}_e_num_{e_num}.png')
        plt.show()
