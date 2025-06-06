import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# --- Configuration ---
# Base directories for the ground state data
DIR_NEUTRALS = 'GS_alpha_HPC'
DIR_ANIONS = 'GS_alpha_anions_HPC'    # Z protons, Z+1 electrons
DIR_CATIONS = 'GS_alpha_pos_ions_HPC' # Z protons, Z-1 electrons

# Output directory for plots
OUTPUT_DIR = 'electron_binding_analysis_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of elements (atomic number Z) to process
# Adjust this list based on the data you have computed
ELEMENTS_Z = list(range(1, 19)) # H (1) to Ar (18)
ELEMENT_SYMBOLS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                   "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]

# --- Helper Function to Load Ground State Energy ---
def load_gs_energy(directory, p_num, e_num, element_symbol):
    """Loads ground state energy from a specific CSV file."""
    filename = f'{p_num:02d}_{element_symbol}_{e_num:02d}e.csv'
    filepath = os.path.join(directory, filename)
    
    if not os.path.exists(filepath):
        # print(f"Warning: File not found - {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            row = next(reader) # Assuming one relevant data row
            return float(row['ground_state_energy'])
    except Exception as e:
        print(f"Warning: Error reading or parsing {filepath}: {e}")
        return None

# --- Main Processing ---
delta_E_anion = []    # E_neutral - E_anion
delta_E_cation = []   # E_neutral - E_cation
processed_symbols_anion = []
processed_symbols_cation = []

for Z_idx, Z in enumerate(ELEMENTS_Z):
    p_num = Z
    element_symbol = ELEMENT_SYMBOLS[Z_idx]

    # --- For Anion Comparison ---
    e_num_neutral = p_num
    e_num_anion = p_num + 1
    
    E_neutral_for_anion = load_gs_energy(DIR_NEUTRALS, p_num, e_num_neutral, element_symbol)
    E_anion = load_gs_energy(DIR_ANIONS, p_num, e_num_anion, element_symbol)
    
    if E_neutral_for_anion is not None and E_anion is not None:
        diff = E_neutral_for_anion - E_anion
        delta_E_anion.append(diff)
        processed_symbols_anion.append(element_symbol)
        print(f"{element_symbol}: E_neutral = {E_neutral_for_anion:.3f}, E_anion = {E_anion:.3f}, E_N - E_A = {diff:.3f}")
    else:
        print(f"Skipping anion comparison for {element_symbol} (Z={Z}) due to missing data.")

    # --- For Cation Comparison ---
    e_num_neutral_for_cation = p_num # Should be the same as E_neutral_for_anion if p_num is the same
    e_num_cation = p_num - 1
    
    if e_num_cation >= 0: # Cation must have non-negative electrons (H+ has 0e)
        E_neutral_for_cation = load_gs_energy(DIR_NEUTRALS, p_num, e_num_neutral_for_cation, element_symbol)
        E_cation = load_gs_energy(DIR_CATIONS, p_num, e_num_cation, element_symbol)

        if E_neutral_for_cation is not None and E_cation is not None:
            diff = E_neutral_for_cation - E_cation
            delta_E_cation.append(diff)
            processed_symbols_cation.append(element_symbol)
            print(f"{element_symbol}: E_neutral = {E_neutral_for_cation:.3f}, E_cation = {E_cation:.3f}, E_N - E_C = {diff:.3f}")
        else:
            print(f"Skipping cation comparison for {element_symbol} (Z={Z}) due to missing data for cation or neutral.")
    else:
        print(f"Skipping cation for {element_symbol} (Z={Z}) as it would have < 0 electrons.")


# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 12


# Plot 1: E_neutral - E_anion
if delta_E_anion:
    fig1, ax1 = plt.subplots(figsize=(15, 7))
    colors_anion = ['green' if val > 0 else 'red' for val in delta_E_anion]
    bars1 = ax1.bar(processed_symbols_anion, delta_E_anion, color=colors_anion)
    ax1.set_xlabel('Element (Z)')
    ax1.set_ylabel(r'$E_{neutral}(Z) - E_{anion}(Z, e=Z+1)$ [a.u.]')
    ax1.set_title('Energy Difference: Neutral vs. Anion (FMD Model)')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (0.05 * np.sign(yval) if yval != 0 else 0.05), 
                 f'{yval:.3f}', ha='center', va='bottom' if yval >=0 else 'top', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'deltaE_neutral_vs_anion.png'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'deltaE_neutral_vs_anion.svg'))
    print(f"Plot 'deltaE_neutral_vs_anion.png' saved to {OUTPUT_DIR}")
    plt.show()

# Plot 2: E_neutral - E_cation
if delta_E_cation:
    fig2, ax2 = plt.subplots(figsize=(15, 7))
    colors_cation = ['green' if val > 0 else 'red' for val in delta_E_cation]
    bars2 = ax2.bar(processed_symbols_cation, delta_E_cation, color=colors_cation)
    ax2.set_xlabel('Element (Z)')
    ax2.set_ylabel(r'$E_{neutral}(Z) - E_{cation}(Z, e=Z-1)$ [a.u.]')
    ax2.set_title('Energy Difference: Neutral vs. Cation (FMD Model)')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (0.05 * np.sign(yval) if yval != 0 else 0.05), 
                 f'{yval:.3f}', ha='center', va='bottom' if yval >=0 else 'top', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'deltaE_neutral_vs_cation.png'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'deltaE_neutral_vs_cation.svg'))
    print(f"Plot 'deltaE_neutral_vs_cation.png' saved to {OUTPUT_DIR}")
    plt.show()

if not delta_E_anion and not delta_E_cation:
    print("No data was processed to generate plots. Check file paths and data availability.")
