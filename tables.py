import numpy as np
import csv
import os

# --- Helper Functions ---

def load_fmd_gs_data(filepath):
    """Loads FMD ground state data from a single-row CSV."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found - {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        try:
            row = next(reader) # Assuming only one data row per file relevant here
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
        raise ValueError(f"Configuration length {len(config_flat)} does not match 6*e_num={6*e_num}")
    
    r_magnitudes = config_flat[0*e_num : 1*e_num]
    # theta_r = config_flat[1*e_num : 2*e_num]
    # phi_r = config_flat[2*e_num : 3*e_num]
    p_magnitudes = config_flat[3*e_num : 4*e_num]
    # theta_p = config_flat[4*e_num : 5*e_num]
    # phi_p = config_flat[5*e_num : 6*e_num]
    return r_magnitudes, p_magnitudes

def calculate_moments(r_mags, p_mags):
    """Calculates <1/r>, <r>, <r^2>^1/2, <p>, <p^2>^1/2."""
    if len(r_mags) == 0: # Avoid division by zero for e_num=0 (though unlikely here)
        return [np.nan] * 5

    # Ensure r_mags are positive for 1/r
    r_mags_safe = np.maximum(r_mags, 1e-9) # Avoid division by zero if any r is exactly 0

    avg_inv_r = np.mean(1.0 / r_mags_safe)
    avg_r = np.mean(r_mags)
    rms_r = np.sqrt(np.mean(r_mags**2))
    avg_p = np.mean(p_mags)
    rms_p = np.sqrt(np.mean(p_mags**2)) # This is <p^2>^1/2, not sqrt(<p>^2)
    
    return avg_inv_r, avg_r, rms_r, avg_p, rms_p

# --- Main Script ---

# Define the base directory where your GS CSV files are located
BASE_GS_DIR = 'GS_feedback_HPC' # MODIFY AS NEEDED (e.g., the path from your v3-2_gs_run_HPC.py)

# Define output directory for the new tables
OUTPUT_TABLES_DIR = 'kirschbaum_wilets_reproduced_tables'
os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)

# List of atoms/ions and their corresponding FMD data filenames
# (filename, system_label_for_table)
systems_to_process = {
    "H-": ('01_H_02e.csv', 'H-'),        # Assuming H- is 1 proton, 2 electrons
    "H":  ('01_H_01e.csv', 'H'),         # For Table V (if you want to compare with their H)
    "He": ('02_He_02e.csv', 'He'),
    "Li": ('03_Li_03e.csv', 'Li'),       # Assuming neutral Li
    "Ne": ('10_Ne_10e.csv', 'Ne'),
    "Ar": ('18_Ar_18e.csv', 'Ar')
}

# --- Data Collection ---
all_system_data = {}
for label, (fname, _) in systems_to_process.items():
    filepath = os.path.join(BASE_GS_DIR, fname)
    data = load_fmd_gs_data(filepath)
    if data:
        r_mags, p_mags = get_spherical_coords_from_config(data['optimal_configuration'], data['e_num'])
        data['moments'] = calculate_moments(r_mags, p_mags)
        all_system_data[label] = data
    else:
        print(f"Skipping {label} due to missing or invalid data file.")

# --- Table I: Some moments for neon ---
if "Ne" in all_system_data:
    ne_data = all_system_data["Ne"]
    ne_moments = ne_data['moments']
    table1_data = [
        # Moment, Classical (Your FMD), HF (Ref. 22 from paper - for comparison if you add it)
        ["<1/r>", f"{ne_moments[0]:.3f}"],
        ["<r>", f"{ne_moments[1]:.3f}"],
        ["<r^2>^1/2", f"{ne_moments[2]:.3f}"],
        ["<p>", f"{ne_moments[3]:.3f}"],
        ["<p^2>^1/2", f"{ne_moments[4]:.3f}"] # Paper uses <p^2> not <p^2>^1/2 for the last one in table
                                        # but their table heading says <p^2>. Let's assume they mean magnitude of <p^2>
                                        # Or actually, their table just says <p^2>. Let's use rms_p^2
                                        # Rereading paper: Table I has <p^2>. My rms_p is <p^2>^1/2. So I need rms_p**2
    ]
    # Correcting the last moment for Table I to be <p^2>
    table1_data[-1] = ["<p^2>", f"{ne_moments[4]**2:.3f}"]


    with open(os.path.join(OUTPUT_TABLES_DIR, 'Table_I_Neon_Moments.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Moment', 'Classical (FMD)']) # Add HF column if you have that data
        writer.writerows(table1_data)
    print("Generated Table_I_Neon_Moments.csv")

# --- Table II: Some moments for argon ---
if "Ar" in all_system_data:
    ar_data = all_system_data["Ar"]
    ar_moments = ar_data['moments']
    table2_data = [
        ["<1/r>", f"{ar_moments[0]:.3f}"],
        ["<r>", f"{ar_moments[1]:.3f}"],
        ["<r^2>^1/2", f"{ar_moments[2]:.3f}"],
        ["<p>", f"{ar_moments[3]:.3f}"],
        ["<p^2>^1/2", f"{ar_moments[4]:.3f}"] # Paper table seems to use <p^2>^1/2 here
    ]
    with open(os.path.join(OUTPUT_TABLES_DIR, 'Table_II_Argon_Moments.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Moment', 'Classical (FMD)'])
        writer.writerows(table2_data)
    print("Generated Table_II_Argon_Moments.csv")

# --- Table III: Contributions to the total energy for neon ---
if "Ne" in all_system_data:
    ne_data = all_system_data["Ne"]
    table3_data = [
        # Ek, Ep (V_Nuc + V_ee), Vh, Vp, Total
        [
            f"{ne_data['kinetic_energy']:.1f}",
            f"{ne_data['nuclear_potential'] + ne_data['pair_potential']:.1f}", # Ep is sum of attractive and repulsive Coulomb
            f"{ne_data['heisenberg_potential']:.1f}",
            f"{ne_data['pauli_potential']:.1f}",
            f"{ne_data['ground_state_energy']:.1f}"
        ]
    ]
    with open(os.path.join(OUTPUT_TABLES_DIR, 'Table_III_Neon_Energy_Contributions.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ek', 'Ep', 'Vh', 'Vp', 'Total'])
        writer.writerows(table3_data)
    print("Generated Table_III_Neon_Energy_Contributions.csv")

# --- Table IV: Contributions to the total energy for argon ---
if "Ar" in all_system_data:
    ar_data = all_system_data["Ar"]
    table4_data = [
        [
            f"{ar_data['kinetic_energy']:.1f}",
            f"{ar_data['nuclear_potential'] + ar_data['pair_potential']:.1f}",
            f"{ar_data['heisenberg_potential']:.1f}",
            f"{ar_data['pauli_potential']:.1f}",
            f"{ar_data['ground_state_energy']:.1f}"
        ]
    ]
    with open(os.path.join(OUTPUT_TABLES_DIR, 'Table_IV_Argon_Energy_Contributions.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Ek', 'Ep', 'Vh', 'Vp', 'Total'])
        writer.writerows(table4_data)
    print("Generated Table_IV_Argon_Energy_Contributions.csv")


# --- Table V: Summary of the (-) energies of completed systems ---
table5_data = []
# Order from paper: H-, He, Li, Ne, Ar
table_v_order = ["H-", "He", "Li", "Ne", "Ar"] 
# Add "H" if you want to include neutral hydrogen comparison
# if "H" in all_system_data: table_v_order.insert(1, "H") 

for system_label in table_v_order:
    if system_label in all_system_data:
        sys_data = all_system_data[system_label]
        # Paper reports negative energies, so multiply by -1 if your GS energy is already negative
        # Or just take the value if it's already correct.
        # Your example CSV has ground_state_energy: -8.038
        # The table shows positive values for the summary of (-) energies.
        # So, if your energy is -X, report X.
        energy_to_report = -sys_data['ground_state_energy'] 
        table5_data.append(
            [systems_to_process[system_label][1], f"{energy_to_report:.4f}"] # System label, Classical (FMD)
        )
    else:
        table5_data.append([systems_to_process[system_label][1], "N/A"])


with open(os.path.join(OUTPUT_TABLES_DIR, 'Table_V_System_Energies.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    # Add TF, HF, Experimental columns if you collect that data elsewhere for comparison
    writer.writerow(['System', 'Classical (FMD)']) 
    writer.writerows(table5_data)
print("Generated Table_V_System_Energies.csv")

print(f"\nAll table CSV files saved in: {OUTPUT_TABLES_DIR}")