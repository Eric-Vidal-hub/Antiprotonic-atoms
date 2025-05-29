"""
Author: Eric Vidal Marcos
Date: 14-03-2025
Project: Plotting for the GS study using the FMD semi-classical model
with V_H and V_P potentials.

This module contains functions to read data, plot results, and compare
calculated ground state energies with NIST data. It includes functions
to plot ground state energy, electron distribution, and relative error.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Global font sizes
labelfontsize = 18
tickfontsize = 14


def read_nist_data(directory):
    """
    Read NIST data from files in the specified directory.

    :param directory: Directory containing the NIST data files.
    :return: Dictionary with atomic numbers as keys and Etot values as values.
    """
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


def plot_results(df, output_dir):
    """
    Plot ground state energy for each e_num.

    :param df: DataFrame containing the results.
    :param output_dir: Directory to save the plots.
    """
    plt.figure(figsize=(10, 6))
    plt.title('BFGS simulation', fontsize=labelfontsize)
    plt.plot(df['e_num'], np.abs(df['ground_state_energy'].astype(float)), 'o')
    plt.xlabel('Z', fontsize=labelfontsize)
    plt.ylabel(r'$-E_{GS}$ (a.u.)', fontsize=labelfontsize)
    plt.xticks(fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'fig_gs_e.svg'))
    plt.close()  # Close the figure to avoid displaying it


def plot_electron_distribution(df, output_dir):
    """
    Plot electron distribution for each e_num.

    :param df: DataFrame containing the results.
    :param output_dir: Directory to save the plots.
    """
    r0_values = []
    p0_values = []
    e_num_values = []
    time_taken_values = []
    markersize = 10
    markeredgewidth = 1.5

    for _, row in df.iterrows():
        e_num = int(row['e_num'])
        optimal_config = np.fromstring(
            row['optimal_configuration'].strip('[]'), sep=' '
        )
        r0 = optimal_config[:e_num]
        p0 = optimal_config[3 * e_num:4 * e_num]
        time_taken = row['time_taken']

        # Store the values
        r0_values.append((e_num, r0))
        p0_values.append((e_num, p0))
        e_num_values.append(e_num)
        time_taken_values.append(time_taken)

    # Plot r0 and p0
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=False)

    for e_num, r0 in r0_values:
        for i, value in enumerate(r0):
            marker = '^' if i % 2 == 0 else 'v'
            axs[0].plot(e_num, value, marker=marker, linestyle='None',
                        markersize=markersize, markeredgecolor='black',
                        markerfacecolor='none',
                        markeredgewidth=markeredgewidth,)
    axs[0].set_xlabel(r'$Z$', fontsize=labelfontsize)
    axs[0].set_ylabel(r'$r_0$', fontsize=labelfontsize)
    axs[0].set_title(f'Time taken: {sum(time_taken_values):.2f} s',
                     fontsize=labelfontsize)
    axs[0].grid(True)
    axs[0].set_yscale('log')
    axs[0].tick_params(axis='both', which='major', labelsize=tickfontsize)

    for e_num, p0 in p0_values:
        for i, value in enumerate(p0):
            marker = '^' if i % 2 == 0 else 'v'
            axs[1].plot(e_num, value, marker=marker, linestyle='None',
                        markersize=markersize, markeredgecolor='black',
                        markerfacecolor='none',
                        markeredgewidth=markeredgewidth)
    axs[1].set_xlabel(r'$Z$', fontsize=labelfontsize)
    axs[1].set_ylabel(r'$p_0$', fontsize=labelfontsize)
    axs[1].set_yscale('log')
    axs[1].grid(True)
    axs[1].tick_params(axis='both', which='major', labelsize=tickfontsize)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_optimal_r0_p0.svg'))
    plt.close(fig)  # Close the figure to avoid displaying it


def plot_ground_state_energy(df, nist_data, output_dir):
    """
    Plot ground state energy and compare with NIST data.

    :param df: DataFrame containing the results.
    :param nist_data: Dictionary with NIST data.
    :param output_dir: Directory to save the plots.
    """
    plt.figure(figsize=(10, 6))
    # NIST data
    atomic_numbers = list(nist_data.keys())
    minus_etot_values = list(nist_data.values())
    plt.plot(atomic_numbers, minus_etot_values, 'o', markersize=5,
             linestyle='None', label='NIST-LDA', markeredgecolor='orange',
             markerfacecolor='orange')

    # Calculated data
    e_num_values = df['e_num'].values
    ground_state_energies = -df['ground_state_energy'].values
    plt.plot(e_num_values, ground_state_energies, 'o', markersize=5,
             linestyle='None', label='BFGS', markeredgecolor='darkblue',
             markerfacecolor='none')

    plt.xlabel(r'$Z$', fontsize=labelfontsize)
    plt.ylabel(r'$-E_{GS}$ (a.u.)', fontsize=labelfontsize)
    plt.xticks(fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fig_gs_e_vs_z.svg'))
    plt.close()  # Close the figure to avoid displaying it


def plot_relative_error(df, nist_data, output_dir):
    """
    Plot relative error of ground state energy compared to NIST data.

    :param df: DataFrame containing the results.
    :param nist_data: Dictionary with NIST data.
    :param output_dir: Directory to save the plots.
    """
    plt.figure(figsize=(10, 6))

    # Calculated data
    e_num_values = df['e_num'].values
    ground_state_energies = -df['ground_state_energy'].values

    # Calculate errors
    errors = []
    for e_num, gs_energy in zip(e_num_values, ground_state_energies):
        if e_num in nist_data:
            error = 100 * np.abs((gs_energy - nist_data[e_num]) /
                                 nist_data[e_num])
            errors.append(error)
        else:
            errors.append(None)  # for cases where NIST data is not available

    # Filter out None values
    filtered_e_num_values = [e_num for e_num, error in zip(e_num_values,
                                                           errors)
                             if error is not None]
    filtered_errors = [error for error in errors if error is not None]

    plt.plot(filtered_e_num_values, filtered_errors, 'o', markersize=7,
             linestyle='None')

    plt.xlabel(r'$Z$', fontsize=labelfontsize)
    plt.ylabel(r'$|\frac{E_{GS, BFGS} - E_{GS, NIST}}{E_{GS, NIST}}|$ (%)',
               fontsize=labelfontsize)
    plt.xticks(fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_rel-error_vs_z.svg'))
    plt.close()  # Close the figure to avoid displaying it


def main(file_name, output_dir,
         nist_directory='c:/Users/propietario/Documents/Antiprotonic-atoms/'
                        'LDA/neutrals'):
    """
    Main function to read data, plot results, and compare with NIST data.

    :param file_name: CSV file containing the results.
    :param output_dir: Directory to save the plots.
    :param nist_directory: Directory containing the NIST data files.
    """
    # Read the results from the CSV file
    df = pd.read_csv(file_name)

    # Output directory for plots
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot ground state energy for each e_num
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
gtol = 1e-4
# # Scaling parameters according to alpha
# xi_h = xi_h / np.sqrt(1 + 1 / (2 * alpha))
# xi_p = xi_p / np.sqrt(1 + 1 / (2 * alpha))
# print(f'For alpha = {alpha}: xi_h = {xi_h:.3f}, xi_p = {xi_p:.3f}')

# Number of electrons range to study
e_ini = 1
e_fin = 14

# Open the CSV file to write the results
file_name = (f'results_alpha_{alpha}_xi_h_{xi_h:.3f}_xi_p_{xi_p:.3f}_'
             f'e_{e_ini}_to_{e_fin}.csv')
output_dir = (f'Plots_e_{e_ini}_to_{e_fin}_gtol_{gtol:.1e}_optimal_conditions')

# Call the main function
if __name__ == "__main__":
    main(file_name, output_dir)
