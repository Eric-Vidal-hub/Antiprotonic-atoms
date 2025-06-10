import csv
import pandas as pd # For reading and plotting convenience
import matplotlib.pyplot as plt
import numpy as np # For isnan if needed, though pandas handles it

data_string = """
 pb+He  FMD   J.S. Cohen, Phys. Rev. A 62, 022512 (2000)                            
 0.0100  7.7760E+01  1.1300E+00
 0.1000  1.8200E+01  2.8000E-01
 0.2000  9.5500E+00  1.4000E-01
 0.3000  6.9000E+00  9.0000E-02
 0.4000  5.3400E+00  6.0000E-02
 0.5000  4.5600E+00  7.0000E-02
 0.6000  3.9400E+00  6.0000E-02
 0.7000  3.5400E+00  5.0000E-02
 0.8000  3.2700E+00  5.0000E-02
 0.9000  3.0500E+00  5.0000E-02
 1.0000  2.9100E+00  5.0000E-02
 1.1000  1.7600E+00  7.0000E-02
 1.2000  9.5000E-01  6.0000E-02
 1.5000  3.8000E-01  5.0000E-02
 2.0000  1.0000E-01  3.0000E-02
 2.5000  1.0000E-02  1.0000E-02
"""

# Output CSV filename
csv_filename = "cohen_pbar_he_capture_data_with_errors.csv"

# Split the string into lines and filter out empty lines or header
lines = [line.strip() for line in data_string.strip().split('\n')]
data_lines = [line for line in lines if line and not line.startswith("pb+He")]

# Prepare data for CSV
# Columns are: Energy (a.u.), Sigma_Total_Capture (a0^2), Error_Sigma_Total_Capture (a0^2)
header = ["Energy_au", "Sigma_Total_Capture_a02", "Error_Sigma_Cap_a02"]
rows_for_csv = []

for line in data_lines:
    parts = line.split()
    if len(parts) == 3:
        try:
            energy = float(parts[0])
            sigma_cap = float(parts[1])
            error_sigma_cap = float(parts[2]) # Now interpreted as error
            rows_for_csv.append([energy, sigma_cap, error_sigma_cap])
        except ValueError:
            print(f"Skipping line due to parsing error: {line}")
    else:
        print(f"Skipping malformed line: {line}")

# Write to CSV
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)
    csvwriter.writerows(rows_for_csv)

print(f"Data successfully written to {csv_filename}")

# --- Example: How to read and plot this data with pandas and error bars ---
try:
    df = pd.read_csv(csv_filename)
    print("\n--- Data from CSV with Errors ---")
    print(df)

    plt.figure(figsize=(10, 6))
    
    # Plotting with error bars
    plt.errorbar(
        df["Energy_au"], 
        df["Sigma_Total_Capture_a02"], 
        yerr=df["Error_Sigma_Cap_a02"], 
        fmt='o',           # Marker style
        capsize=5,         # Error bar cap size
        label='Total Capture (Cohen PRA 62)',
        color='tab:blue',
        ecolor='gray'      # Color of error bars
    )
    
    plt.xlabel("Antiproton Energy (a.u.)")
    plt.ylabel("Cross Section ($a_0^2$)")
    plt.title("pbar + He Total Capture Cross Section (Data from J.S. Cohen, PRA 62, 022512)")
    plt.legend()
    plt.ylim(0, 25)  # Adjust y-limits based on data
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("cohen_pbar_he_plot_with_errors.png")
    plt.show()
    
except FileNotFoundError:
    print(f"\nCSV file {csv_filename} not found. Plotting skipped.")
except Exception as e:
    print(f"\nError reading or plotting CSV: {e}")