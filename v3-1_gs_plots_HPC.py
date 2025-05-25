"""
Author: Eric Vidal Marcos
Date: 26-03-2025
Project: Plotting for the GS study using the FMD semi-classical model
with V_H and V_P potentials.

This module contains functions to read data, plot results, and compare
calculated ground state energies with NIST data. It includes functions
to plot ground state energy, electron distribution, and relative error.
"""

import os
import re
from matplotlib import markers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Global font sizes
labelfontsize = 24
tickfontsize = 24
ticklength = 5
tickwidth = 2


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


def plot_combined_results(aggregated_df, nist_data, output_dir):
    """
    Plot combined ground state energy for all files and compare with NIST data.

    :param aggregated_df: Aggregated DataFrame containing the results.
    :param nist_data: Dictionary with NIST data.
    :param output_dir: Directory to save the plots.
    """
    markersize = 9
    markeredgewidth = 2  # Increased from previous value

    plt.figure(figsize=(10, 6))
    # Plot NIST data
    atomic_numbers = list(nist_data.keys())
    minus_etot_values = list(nist_data.values())
    plt.plot(atomic_numbers[:52], minus_etot_values[:52], 'o', markersize=markersize,
             linestyle='None', label='NIST-LDA', markeredgecolor='orange',
             markerfacecolor='orange', markeredgewidth=markeredgewidth)
    
    # Plot results from files
    plt.plot(aggregated_df['e_num'],
             np.abs(aggregated_df['ground_state_energy']),
             'o', markersize=markersize, linestyle='None', label='BFGS',
             markeredgecolor='darkblue', markerfacecolor='none', markeredgewidth=markeredgewidth)

    plt.xlabel(r'$Z$', fontsize=labelfontsize)
    plt.ylabel(r'$-E_{GS}$ (a.u.)', fontsize=labelfontsize)
    plt.xlim(0, 53)
    plt.xticks(fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.tick_params(axis='both', which='both', direction='in', length=ticklength, width=tickwidth, top=True, right=True)
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fontsize=18, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_fig_gs_e_vs_nist.svg'))
    plt.close()


def plot_electron_distribution(aggregated_df, output_dir):
    """
    Plot electron distribution for all files with up triangles for spin up
    and down triangles for spin down. The distribution is plotted in log scale.

    :param aggregated_df: Aggregated DataFrame containing the results.
    :param output_dir: Directory to save the plots.
    """
    markersize = 6
    markeredgewidth = 1.5
    plt.figure(figsize=(10, 6))
    for _, group in aggregated_df.groupby('file_name'):
        for _, row in group.iterrows():
            e_num = int(row['e_num'])
            optimal_config = np.fromstring(row['optimal_configuration']
                                           .strip('[]'), sep=' ')
            r0 = optimal_config[:e_num]

            for i, r in enumerate(r0):
                if i % 2 == 0:  # Even index -> Spin down
                    plt.plot(e_num, r, '^', linestyle='None',
                             markersize=markersize, markeredgecolor='black',
                             markerfacecolor='none',
                             markeredgewidth=markeredgewidth)
                else:  # Odd index -> Spin up
                    plt.plot(e_num, r, 'v', linestyle='None',
                             markersize=markersize, markeredgecolor='black',
                             markerfacecolor='none',
                             markeredgewidth=markeredgewidth)

    plt.xlabel(r'$Z$', fontsize=labelfontsize)
    plt.ylabel(r'$r_0$', fontsize=labelfontsize)
    plt.yscale('log')
    plt.xticks(fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.tick_params(axis='both', which='both', direction='in', length=ticklength, width=tickwidth, top=True, right=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'electron_distribution.svg'))
    plt.close()


def plot_momenta_distribution(aggregated_df, output_dir):
    """
    Plot momenta distribution for all files with up triangles for spin up
    and down triangles for spin down. The distribution is plotted in log scale.

    :param aggregated_df: Aggregated DataFrame containing the results.
    :param output_dir: Directory to save the plots.
    """
    markersize = 6
    markeredgewidth = 1.5
    plt.figure(figsize=(10, 6))
    for _, group in aggregated_df.groupby('file_name'):
        for _, row in group.iterrows():
            e_num = int(row['e_num'])
            optimal_config = np.fromstring(row['optimal_configuration']
                                           .strip('[]'), sep=' ')
            p0 = optimal_config[3*e_num:4*e_num]

            for i, p in enumerate(p0):
                if i % 2 == 0:  # Even index -> Spin down
                    plt.plot(e_num, p, '^', linestyle='None',
                             markersize=markersize, markeredgecolor='black',
                             markerfacecolor='none',
                             markeredgewidth=markeredgewidth)
                else:  # Odd index -> Spin up
                    plt.plot(e_num, p, 'v', linestyle='None',
                             markersize=markersize, markeredgecolor='black',
                             markerfacecolor='none',
                             markeredgewidth=markeredgewidth)

    plt.xlabel(r'$Z$', fontsize=labelfontsize)
    plt.ylabel(r'$p_0$', fontsize=labelfontsize)
    plt.yscale('log')
    plt.xticks(fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.tick_params(axis='both', which='both', direction='in', length=ticklength, width=tickwidth, top=True, right=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'momenta_distribution.svg'))
    plt.close()


def plot_relative_error(aggregated_df, nist_data, output_dir):
    """
    Plot relative error of ground state energy compared to NIST data.

    :param aggregated_df: Aggregated DataFrame containing the results.
    :param nist_data: Dictionary with NIST data.
    :param output_dir: Directory to save the plots.
    """
    plt.figure(figsize=(10, 6))
    # Plot relative error for results
    for _, group in aggregated_df.groupby('file_name'):
        errors = []
        for _, row in group.iterrows():
            e_num = int(row['e_num'])
            gs_energy = -row['ground_state_energy']
            if e_num in nist_data:
                error = 100 * np.abs((gs_energy - nist_data[e_num]) /
                                     nist_data[e_num])
                errors.append((e_num, error))

        if errors:
            e_nums, rel_errors = zip(*errors)
            plt.plot(e_nums, rel_errors, 'o', markersize=7,
                     linestyle='None', color='blue')

    plt.xlabel(r'$Z$', fontsize=labelfontsize)
    plt.ylabel(r'$|\frac{E_{GS, BFGS} -E_{GS, NIST}}'
               r'{E_{GS, NIST}}|$ (%)', fontsize=labelfontsize)
    plt.xticks(fontsize=tickfontsize)
    plt.yticks(fontsize=tickfontsize)
    plt.tick_params(axis='both', which='both', direction='in', length=ticklength, width=tickwidth, top=True, right=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_relative_error.svg'))
    plt.close()


def main(hpc_results_dir, output_dir, start_file=None, end_file=None,
         nist_directory='c:/Users/propietario/Documents/Antiprotonic-atoms/'
                        'LDA/neutrals'):
    """
    Main function to read all files in the HPC_results folder, aggregate data,
    and plot combined results.

    :param hpc_results_dir: Directory containing the HPC results CSV files.
    :param output_dir: Directory to save the plots.
    :param start_file: Name of the first file to include in the plots.
    :param end_file: Name of the last file to include in the plots.
    :param nist_directory: Directory containing the NIST data files.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read NIST data
    nist_data = read_nist_data(nist_directory)

    # Aggregate data from all CSV files
    aggregated_data = []
    files = sorted(os.listdir(hpc_results_dir))
    if start_file and end_file:
        # Filter files based on the specified range
        if start_file in files and end_file in files:
            start_index = files.index(start_file)
            end_index = files.index(end_file) + 1
            files = files[start_index:end_index]
        else:
            print("Specified file range is invalid.")
            return

    for filename in files:
        if filename.endswith('.csv'):
            file_path = os.path.join(hpc_results_dir, filename)
            try:
                # Attempt to read the CSV file
                df = pd.read_csv(file_path)
                if df.empty:
                    print(f"Skipping empty file: {filename}")
                    continue
                df['file_name'] = filename  # Add a column to track
                aggregated_data.append(df)
            except pd.errors.EmptyDataError:
                print(f"Skipping invalid or empty file: {filename}")
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    # Combine all data into a single DataFrame
    if aggregated_data:
        aggregated_df = pd.concat(aggregated_data, ignore_index=True)

        # Plot combined ground state energy with NIST data
        plot_combined_results(aggregated_df, nist_data, output_dir)

        # Plot electron position distribution
        plot_electron_distribution(aggregated_df, output_dir)

        # Plot electron momenta distribution
        plot_momenta_distribution(aggregated_df, output_dir)

        # Plot relative error
        plot_relative_error(aggregated_df, nist_data, output_dir)
    else:
        print("No valid data files found in the HPC_results directory.")


# Define the directories
hpc_results_dir = ('c:/Users/propietario/Documents/Antiprotonic-atoms/'
                   'HPC_results_gs_with_previous_z_as_ic')
output_dir = 'c:/Users/propietario/Documents/Antiprotonic-atoms/Plots'

# Specify the file range (optional)
start_file = '01_H_01e.csv'
end_file = '52_Te_52e.csv'

# Call the main function
if __name__ == "__main__":
    main(hpc_results_dir, output_dir, start_file, end_file)
