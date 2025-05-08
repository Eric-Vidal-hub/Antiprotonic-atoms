import csv
import matplotlib.pyplot as plt


# Filepath to the trajectory file
file_path = 'HPC_dynamics/trajectory_example.csv'

# Initialize data storage
time = []
r_p = []
electron_distances = {}

# Read the CSV file
with open(file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Store time and antiproton radial distance
        time.append(float(row['time']))
        r_p.append(float(row['r_p']))

        # Store electron radial distances
        for key in row:
            if key.startswith('r_e'):  # Electron radial distances
                if key not in electron_distances:
                    electron_distances[key] = []
                electron_distances[key].append(float(row[key]))

# Plot the antiproton radial distance
plt.figure()
plt.plot(time, r_p, label='Antiproton', color='blue')
plt.xlabel('Time (a.u.)')
plt.ylabel('Radial Distance (a.u.)')
plt.ylim(0, 90)  # Set y-axis limit to 20 a.u.
plt.grid(True)

# Plot the electron radial distances
for key, values in electron_distances.items():
    plt.plot(time, values, label=key)

# Add legend and show the plot
plt.legend()
plt.tight_layout()
plt.show()