import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from progressbar import progressbar

# Function to read NIST data
def read_nist_data(directory):
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

    return dict(zip(atomic_numbers, nist_etot_values))

# Function to plot ground state energy for each e_num and optimizer
def plot_results(df, output_dir):
    plt.figure(figsize=(10, 6))
    for optimizer in progressbar(df['optimizer'].unique(), prefix='Plotting ground state energy: '):
        subset = df[df['optimizer'] == optimizer]
        plt.plot(subset['e_num'], np.abs(subset['ground_state_energy'].astype(float)), 'o', label=optimizer)

    plt.xlabel('Number of Electrons (e_num)')
    plt.ylabel(r'$-E_{GS}$ (a.u.)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'fig_gs_e_optimizers.png'))
    plt.close()  # Close the figure to avoid displaying it

# Function to plot electron distribution
def plot_electron_distribution(df, output_dir):
    r0_values = {}
    p0_values = {}
    e_num_values = {}
    time_taken_values = {}

    for optimizer in df['optimizer'].unique():
        subset = df[df['optimizer'] == optimizer]
        
        r0_values[optimizer] = []
        p0_values[optimizer] = []
        e_num_values[optimizer] = []
        time_taken_values[optimizer] = []

        for index, row in subset.iterrows():
            e_num = int(row['e_num'])
            optimal_config = np.fromstring(row['optimal_configuration'].strip('[]'), sep=' ')
            r0 = optimal_config[:e_num]
            p0 = optimal_config[3 * e_num:4 * e_num]
            time_taken = row['time_taken']
            
            # Store the values
            r0_values[optimizer].append((e_num, r0))
            p0_values[optimizer].append((e_num, p0))
            e_num_values[optimizer].append(e_num)
            time_taken_values[optimizer].append(time_taken)

    # Plot r0 and p0 for each optimizer
    for optimizer in progressbar(df['optimizer'].unique(), prefix='Plotting electron distribution: '):
        fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

        for e_num, r0 in r0_values[optimizer]:
            for i, value in enumerate(r0):
                marker = '^' if i % 2 == 0 else 'v'
                axs[0].plot(e_num, value, marker=marker, linestyle='None')
        axs[0].set_ylabel('r0')
        axs[0].set_title(f'Optimal r0 for {optimizer} (Time taken: {sum(time_taken_values[optimizer]):.2f} s)')
        axs[0].grid(True)
        axs[0].set_yscale('log')

        for e_num, p0 in p0_values[optimizer]:
            for i, value in enumerate(p0):
                marker = '^' if i % 2 == 0 else 'v'
                axs[1].plot(e_num, value, marker=marker, linestyle='None', label=f'e_num={e_num}' if i == 0 else "")
        axs[1].set_xlabel('e_num')
        axs[1].set_ylabel('p0')
        axs[1].set_title(f'Optimal p0 for {optimizer} (Time taken: {sum(time_taken_values[optimizer]):.2f} s)')
        axs[1].set_yscale('log')
        axs[1].grid(True)
        axs[1].legend()

        plt.savefig(os.path.join(output_dir, f'fig_optimal_r0_p0_{optimizer}.png'))
        plt.close(fig)  # Close the figure to avoid displaying it

# Function to plot ground state energy
def plot_ground_state_energy(df, nist_data, output_dir):
    labelfontsize = 18
    tickfontsize = 14
    plt.figure(figsize=(10, 6))
    # NIST data
    atomic_numbers = list(nist_data.keys())
    minus_etot_values = list(nist_data.values())
    plt.plot(atomic_numbers, minus_etot_values, 'o', markersize=3, linestyle='None', label='NIST-LDA')

    # Calculated data
    for optimizer in progressbar(df['optimizer'].unique(), prefix='Plotting ground state energy vs Z: '):
        subset = df[df['optimizer'] == optimizer]
        e_num_values = subset['e_num'].values
        ground_state_energies = -subset['ground_state_energy'].values
        plt.plot(e_num_values, ground_state_energies, 'o', markersize=3, linestyle='None', label=optimizer)

    plt.xlabel(r'$Z$', fontsize=labelfontsize)
    plt.ylabel(r'$-E_{GS}$ (a.u.)', fontsize=labelfontsize)
    plt.xticks(fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fig_gs_e_vs_z.png'))
    plt.close()  # Close the figure to avoid displaying it

# Function to plot relative error
def plot_relative_error(df, nist_data, output_dir):
    labelfontsize = 18
    tickfontsize = 14
    plt.figure(figsize=(10, 6))

    # Calculated data
    for optimizer in progressbar(df['optimizer'].unique(), prefix='Plotting relative error: '):
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
    plt.savefig(os.path.join(output_dir, 'fig_rel-error_vs_z.png'))
    plt.close()  # Close the figure to avoid displaying it


# %%
# Main function to call the plotting functions
def main(file_name, output_dir, nist_directory='c:/Users/propietario/Documents/Antiprotonic-atoms/LDA/neutrals'):
    # Read the results from the CSV file
    df = pd.read_csv(file_name)

    # Output directory for plots
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot ground state energy for each e_num and optimizer
    plot_results(df, output_dir)

    # Plot electron distribution
    plot_electron_distribution(df, output_dir)

    # Directory containing the NIST data files
    nist_data = read_nist_data(nist_directory)

    # Plot ground state energy
    plot_ground_state_energy(df, nist_data, output_dir)

    # Plot relative error
    plot_relative_error(df, nist_data, output_dir)


# Default values
# INITIAL CONFIGURATION VALUES
alpha = 5
xi_h = 1.000
xi_p = 2.767
# # Scaling parameters according to alpha
# xi_h = xi_h / np.sqrt(1 + 1 / (2 * alpha))
# xi_p = xi_p / np.sqrt(1 + 1 / (2 * alpha))
# print(f'For alpha = {alpha}: xi_h = {xi_h:.3f}, xi_p = {xi_p:.3f}')

# Number of electrons range to study
e_ini = 1
e_fin = 14

# Open the CSV file to write the results
file_name = f'results_alpha_{alpha}_xi_h_{xi_h:.3f}_xi_p_{xi_p:.3f}_e_{e_ini}_to_{e_fin}.csv'
output_dir = f'Plots_e_{e_ini}_to_{e_fin}_precision_1e-3'
# nist_directory = 'c:/Users/propietario/Documents/Antiprotonic-atoms/LDA/neutrals'

# Call the main function
if __name__ == "__main__":
    main(file_name, output_dir)
