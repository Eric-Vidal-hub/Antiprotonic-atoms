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


# --- Matplotlib style to match v3_multi_plots.py ---
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['font.size'] = 26
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 1
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['figure.facecolor'] = 'white'

# --- Plot 1: E_neutral - E_anion ---
if delta_E_anion:
    plt.close('all')
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    colors_anion = ['green' if val > 0 else 'red' for val in delta_E_anion]
    # Use markers instead of bars
    ax1.plot(processed_symbols_anion, delta_E_anion, 
             marker='o', linestyle='None', markersize=12, 
             color='black', markerfacecolor='none')
    for i, val in enumerate(delta_E_anion):
        ax1.plot(processed_symbols_anion[i], val, marker='o', 
                 markersize=12, color=colors_anion[i])
    ax1.set_xlabel('Element (Z)')
    ax1.set_ylabel(r'$E_{\mathrm{neutral}} - E_{\mathrm{anion}}$ (a.u.)')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    ax1.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'deltaE_neutral_vs_anion.png'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'deltaE_neutral_vs_anion.svg'))
    print(f"Plot 'deltaE_neutral_vs_anion.png' saved to {OUTPUT_DIR}")

# --- Plot 2: E_neutral - E_cation ---
if delta_E_cation:
    plt.close('all')
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    colors_cation = ['green' if val > 0 else 'red' for val in delta_E_cation]
    ax2.plot(processed_symbols_cation, delta_E_cation, 
             marker='o', linestyle='None', markersize=12, 
             color='black', markerfacecolor='none')
    for i, val in enumerate(delta_E_cation):
        ax2.plot(processed_symbols_cation[i], val, marker='o', 
                 markersize=12, color=colors_cation[i])
    ax2.set_xlabel('Element (Z)')
    ax2.set_ylabel(r'$E_{\mathrm{neutral}} - E_{\mathrm{cation}}$ (a.u.)')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.tick_params(
        axis='both', which='both', direction='in', top=True, right=True
    )
    ax2.grid(True, which='both', linestyle='--', linewidth=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'deltaE_neutral_vs_cation.png'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'deltaE_neutral_vs_cation.svg'))
    print(f"Plot 'deltaE_neutral_vs_cation.png' saved to {OUTPUT_DIR}")

if not delta_E_anion and not delta_E_cation:
    print("No data was processed to generate plots. Check file paths and data availability.")
