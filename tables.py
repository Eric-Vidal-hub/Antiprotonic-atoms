import numpy as np
import csv
import os

import numpy as np
import csv
import os

# --- Helper Functions (same as before) ---

def load_fmd_gs_data(filepath):
    """Loads FMD ground state data from a single-row CSV."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found - {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        try:
            row = next(reader)
            data = {
                'p_num': int(row['p_num']),
                'e_num': int(row['e_num']),
                'ground_state_energy': float(row['ground_state_energy']),
                'kinetic_energy': float(row['kinetic_energy']),
                'nuclear_potential': float(row['nuclear_potential']),
                'heisenberg_potential': float(row['heisenberg_potential']),
                'pair_potential': float(row['pair_potential']),
                'pauli_potential': float(row['pauli_potential']),
                'optimal_configuration': np.fromstring(row['optimal_configuration'].strip('[]'), sep=' ')
            }
            return data
        except StopIteration:
            print(f"Warning: File is empty or has no data rows - {filepath}")
            return None
        except KeyError as e:
            print(f"Warning: Missing expected column {e} in {filepath}")
            return None
        except ValueError as e:
            print(f"Warning: Error parsing data in {filepath}: {e}")
            return None

def get_spherical_coords_from_config(config_flat, e_num):
    """Extracts r and p magnitudes from the flat optimal_configuration."""
    if len(config_flat) != 6 * e_num:
        # Try to handle if only r and p mags are stored (less ideal but for flexibility)
        if len(config_flat) == 2 * e_num:
            print("Warning: Configuration seems to contain only r and p magnitudes.")
            r_magnitudes = config_flat[:e_num]
            p_magnitudes = config_flat[e_num:]
            return r_magnitudes, p_magnitudes
        raise ValueError(f"Configuration length {len(config_flat)} does not match 6*e_num={6*e_num} or 2*e_num={2*e_num}")
    
    r_magnitudes = config_flat[0*e_num : 1*e_num]
    p_magnitudes = config_flat[3*e_num : 4*e_num]
    return r_magnitudes, p_magnitudes

def calculate_moments(r_mags, p_mags):
    """Calculates <1/r>, <r>, <r^2>^1/2, <p>, <p^2>^1/2."""
    if len(r_mags) == 0:
        return [np.nan] * 5
    r_mags_safe = np.maximum(r_mags, 1e-9)
    avg_inv_r = np.mean(1.0 / r_mags_safe)
    avg_r = np.mean(r_mags)
    rms_r = np.sqrt(np.mean(r_mags**2))
    avg_p = np.mean(p_mags)
    rms_p_val = np.sqrt(np.mean(p_mags**2)) # This is <p^2>^1/2
    return avg_inv_r, avg_r, rms_r, avg_p, rms_p_val

# --- Hardcoded Data from Kirschbaum & Wilets (1980) Paper ---

PAPER_DATA = {
    "TableI_Neon": { # Moments for Neon
        "HF": { # Ref. 22 (implicitly standard HF values)
            "<1/r>": 3.111,
            "<r>": 0.789,
            "<r^2>^1/2": 0.968, # Paper might list <r^2>, check carefully. Assuming <r^2>^1/2 from context.
            "<p>": None,      # Not listed for HF
            "<p^2>": 5.070    # Paper lists <p^2>
        }
    },
    "TableII_Argon": { # Moments for Argon
        "HF": { # Ref. 22
            "<1/r>": 3.873,
            "<r>": 0.893,
            "<r^2>^1/2": 1.203,
            "<p>": None,      # Not listed for HF
            "<p^2>^1/2": 7.258 # Paper lists <p^2>^1/2
        }
    },
    "TableIII_Neon_Energy": { # Energy contributions for Neon (Classical values from paper for comparison)
        "Classical_Paper": {"Ek": 133.5, "Ep": -291.1, "Vh": 8.9, "Vp": 3.1, "Total": -145.5}
    },
    "TableIV_Argon_Energy": { # Energy contributions for Argon
        "Classical_Paper": {"Ek": 520.2, "Ep": -1132.0, "Vh": 28.4, "Vp": 17.4, "Total": -566.0} # Typo in paper's Ep for Ar, should be sum
    },
    "TableV_Energies": { # Summary of (-) energies
        # System: [Classical_Paper, TF_Paper, HF_Paper, Experimental_Paper]
        "H-": [0.5625, None, None, 0.5277], # TF not applicable or not given for H-
        "He": [3.0625, 3.8740, 2.8359, 2.8900], # Note: paper has 2.8359 for HF, exp is often ~2.903
                                               # Their experimental for He is 2.8900 from table caption.
                                               # My previous assumption of 2.903 was from general knowledge.
                                               # Sticking to paper's values: HF 2.8539, Exp 2.8900
                                               # Correcting HF for He from paper table: 2.8539 is likely typo in my previous.
                                               # Paper table V has He: Class 3.0625, TF 3.8740, HF 2.8359, Exp 2.8900. OK.
        "Li": [7.938, 9.978, 7.432, 7.44],     # Assuming neutral Li+ is Z=3, e=2. Paper usually means neutral Li.
                                               # Paper states Li values, not Li+.
        "Ne": [145.5, 165.6, 128.5, None],    # Experimental not listed for Ne/Ar
        "Ar": [566.0, 652.7, 526.8, None]
    }
}
# Correcting Ep for Argon Table IV as per paper's total energy logic
# Ep = Total - Ek - Vh - Vp = -566.0 - 520.2 - 28.4 - 17.4 = -1132.0. This is what the paper has.

# --- Main Script ---
BASE_GS_DIR = 'GS_feedback_HPC'
OUTPUT_TABLES_DIR = 'kirschbaum_wilets_reproduced_tables_with_paper_data'
os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)

systems_to_process = {
    "H-": ('01_H_02e.csv', 'H-'),
    "He": ('02_He_02e.csv', 'He'),
    "Li": ('03_Li_03e.csv', 'Li'),
    "Ne": ('10_Ne_10e.csv', 'Ne'),
    "Ar": ('18_Ar_18e.csv', 'Ar')
}
all_system_data_fmd = {}
for label, (fname, _) in systems_to_process.items():
    filepath = os.path.join(BASE_GS_DIR, fname)
    data = load_fmd_gs_data(filepath)
    if data:
        r_mags, p_mags = get_spherical_coords_from_config(data['optimal_configuration'], data['e_num'])
        data['moments'] = calculate_moments(r_mags, p_mags)
        all_system_data_fmd[label] = data
    else:
        print(f"Skipping {label} due to missing or invalid FMD data file.")

# --- Table I: Some moments for neon ---
if "Ne" in all_system_data_fmd:
    ne_fmd = all_system_data_fmd["Ne"]
    ne_moments_fmd = ne_fmd['moments']
    ne_hf_paper = PAPER_DATA["TableI_Neon"]["HF"]
    
    table1_header = ['Moment', 'Classical (FMD)', 'HF (Paper Ref.22)']
    table1_data_rows = [
        ["<1/r>", f"{ne_moments_fmd[0]:.3f}", f"{ne_hf_paper['<1/r>']:.3f}"],
        ["<r>", f"{ne_moments_fmd[1]:.3f}", f"{ne_hf_paper['<r>']:.3f}"],
        ["<r^2>^1/2", f"{ne_moments_fmd[2]:.3f}", f"{ne_hf_paper['<r^2>^1/2']:.3f}"],
        ["<p>", f"{ne_moments_fmd[3]:.3f}", "N/A" if ne_hf_paper['<p>'] is None else f"{ne_hf_paper['<p>']:.3f}"],
        ["<p^2>", f"{ne_moments_fmd[4]**2:.3f}", f"{ne_hf_paper['<p^2>']:.3f}"] # FMD rms_p^2, Paper <p^2>
    ]
    with open(os.path.join(OUTPUT_TABLES_DIR, 'Table_I_Neon_Moments.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(table1_header)
        writer.writerows(table1_data_rows)
    print("Generated Table_I_Neon_Moments.csv")

# --- Table II: Some moments for argon ---
if "Ar" in all_system_data_fmd:
    ar_fmd = all_system_data_fmd["Ar"]
    ar_moments_fmd = ar_fmd['moments']
    ar_hf_paper = PAPER_DATA["TableII_Argon"]["HF"]

    table2_header = ['Moment', 'Classical (FMD)', 'HF (Paper Ref.22)']
    table2_data_rows = [
        ["<1/r>", f"{ar_moments_fmd[0]:.3f}", f"{ar_hf_paper['<1/r>']:.3f}"],
        ["<r>", f"{ar_moments_fmd[1]:.3f}", f"{ar_hf_paper['<r>']:.3f}"],
        ["<r^2>^1/2", f"{ar_moments_fmd[2]:.3f}", f"{ar_hf_paper['<r^2>^1/2']:.3f}"],
        ["<p>", f"{ar_moments_fmd[3]:.3f}", "N/A" if ar_hf_paper['<p>'] is None else f"{ar_hf_paper['<p>']:.3f}"],
        ["<p^2>^1/2", f"{ar_moments_fmd[4]:.3f}", f"{ar_hf_paper['<p^2>^1/2']:.3f}"]
    ]
    with open(os.path.join(OUTPUT_TABLES_DIR, 'Table_II_Argon_Moments.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(table2_header)
        writer.writerows(table2_data_rows)
    print("Generated Table_II_Argon_Moments.csv")

# --- Table III: Contributions to the total energy for neon ---
if "Ne" in all_system_data_fmd:
    ne_fmd = all_system_data_fmd["Ne"]
    ne_classical_paper = PAPER_DATA["TableIII_Neon_Energy"]["Classical_Paper"]
    table3_header = ['Component', 'Ek', 'Ep', 'Vh', 'Vp', 'Total']
    table3_data_rows = [
        ["Classical (FMD)",
            f"{ne_fmd['kinetic_energy']:.1f}",
            f"{ne_fmd['nuclear_potential'] + ne_fmd['pair_potential']:.1f}",
            f"{ne_fmd['heisenberg_potential']:.1f}",
            f"{ne_fmd['pauli_potential']:.1f}",
            f"{ne_fmd['ground_state_energy']:.1f}"],
        ["Classical (Paper)", # For direct comparison
            f"{ne_classical_paper['Ek']:.1f}",
            f"{ne_classical_paper['Ep']:.1f}",
            f"{ne_classical_paper['Vh']:.1f}",
            f"{ne_classical_paper['Vp']:.1f}",
            f"{ne_classical_paper['Total']:.1f}"]
    ]
    with open(os.path.join(OUTPUT_TABLES_DIR, 'Table_III_Neon_Energy_Contributions.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        # Transpose for better CSV layout matching the paper's table structure
        writer.writerow(table3_header)
        # writer.writerows(table3_data_rows) # This writes rows as is
        # To match paper layout (components as rows):
        transposed_data = list(map(list, zip(*([row[0] for row in table3_data_rows], [row[1] for row in table3_data_rows], [row[2] for row in table3_data_rows],
                                              [row[3] for row in table3_data_rows], [row[4] for row in table3_data_rows], [row[5] for row in table3_data_rows]))))
        # This is getting messy for CSV. Let's keep it simple rows for CSV, format in spreadsheet.
        # Or write components as columns and methods as rows
        csv_output_table3 = [table3_header]
        for i in range(len(table3_header)): # Iterate through components
            row_to_write = [table3_header[i]] + [table3_data_rows[j][i] for j in range(len(table3_data_rows))]
            if i==0: row_to_write = ["Source"] + [r[0] for r in table3_data_rows] # Special for first column
            csv_output_table3.append(row_to_write)
        
        # Simpler CSV output:
        writer.writerow(['Source', 'Ek', 'Ep', 'Vh', 'Vp', 'Total'])
        writer.writerow(['Classical (FMD)'] + table3_data_rows[0][1:])
        writer.writerow(['Classical (Paper)'] + table3_data_rows[1][1:])

    print("Generated Table_III_Neon_Energy_Contributions.csv")


# --- Table IV: Contributions to the total energy for argon ---
if "Ar" in all_system_data_fmd:
    ar_fmd = all_system_data_fmd["Ar"]
    ar_classical_paper = PAPER_DATA["TableIV_Argon_Energy"]["Classical_Paper"]
    table4_header = ['Source', 'Ek', 'Ep', 'Vh', 'Vp', 'Total']
    table4_data_rows = [
        ["Classical (FMD)",
            f"{ar_fmd['kinetic_energy']:.1f}",
            f"{ar_fmd['nuclear_potential'] + ar_fmd['pair_potential']:.1f}",
            f"{ar_fmd['heisenberg_potential']:.1f}",
            f"{ar_fmd['pauli_potential']:.1f}",
            f"{ar_fmd['ground_state_energy']:.1f}"],
        ["Classical (Paper)",
            f"{ar_classical_paper['Ek']:.1f}",
            f"{ar_classical_paper['Ep']:.1f}",
            f"{ar_classical_paper['Vh']:.1f}",
            f"{ar_classical_paper['Vp']:.1f}",
            f"{ar_classical_paper['Total']:.1f}"]
    ]
    with open(os.path.join(OUTPUT_TABLES_DIR, 'Table_IV_Argon_Energy_Contributions.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(table4_header)
        for row in table4_data_rows:
            writer.writerow(row)
    print("Generated Table_IV_Argon_Energy_Contributions.csv")

# --- Table V: Summary of the (-) energies of completed systems ---
table5_header = ['System', 'Classical (FMD)', 'Classical (Paper)', 'TF (Paper)', 'HF (Paper)', 'Experimental (Paper)']
table5_data_rows = []
table_v_order = ["H-", "He", "Li", "Ne", "Ar"]

for system_label_key in table_v_order: # Use the key for all_system_data_fmd
    system_display_label = systems_to_process[system_label_key][1] # Get display label like "H-"
    
    fmd_energy_str = "N/A"
    if system_label_key in all_system_data_fmd:
        energy_to_report_fmd = -all_system_data_fmd[system_label_key]['ground_state_energy']
        fmd_energy_str = f"{energy_to_report_fmd:.4f}"
        
    paper_energies = PAPER_DATA["TableV_Energies"].get(system_display_label, [None]*4) # Get energies for display label

    classical_paper_str = f"{paper_energies[0]:.4f}" if paper_energies[0] is not None else "N/A"
    tf_paper_str = f"{paper_energies[1]:.4f}" if paper_energies[1] is not None else "N/A"
    hf_paper_str = f"{paper_energies[2]:.3f}" if paper_energies[2] is not None else "N/A" # HF is .3f in paper
    if system_display_label == "He" and paper_energies[2] is not None: # Special case for He HF from paper
        hf_paper_str = f"{paper_energies[2]:.4f}"
    experimental_paper_str = f"{paper_energies[3]:.4f}" if paper_energies[3] is not None else "N/A"

    table5_data_rows.append([
        system_display_label,
        fmd_energy_str,
        classical_paper_str,
        tf_paper_str,
        hf_paper_str,
        experimental_paper_str
    ])

with open(os.path.join(OUTPUT_TABLES_DIR, 'Table_V_System_Energies.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(table5_header)
    writer.writerows(table5_data_rows)
print("Generated Table_V_System_Energies.csv")

print(f"\nAll table CSV files saved in: {OUTPUT_TABLES_DIR}")