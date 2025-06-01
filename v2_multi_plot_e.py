import numpy as np
import matplotlib.pyplot as plt
import os

from v2_multi_constants import (RESULTS_DIR)

def calculate_total_energy(state, MU, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, E_SPIN):
    num_electrons = len(state) // 6
    r_e_flat = state[:3 * num_electrons]
    p_e_flat = state[3 * num_electrons:]
    r_electrons = r_e_flat.reshape((num_electrons, 3))
    p_electrons = p_e_flat.reshape((num_electrons, 3))
    epsilon = 1e-18

    kinetic_energy = 0
    potential_energy_en = 0
    potential_energy_ee = 0
    potential_energy_heisenberg = 0
    potential_energy_pauli = 0

    for ii in range(num_electrons):
        ri = r_electrons[ii]
        pi = p_electrons[ii]
        ri_norm = np.linalg.norm(ri)
        pi_norm = np.linalg.norm(pi)

        kinetic_energy += np.sum(pi**2) / (2 * MU)

        potential_energy_en += -ZZ / (ri_norm + epsilon)

        for mm in range(ii + 1, num_electrons):
            r_im = ri - r_electrons[mm]
            r_im_norm = np.linalg.norm(r_im)
            potential_energy_ee += 1.0 / (r_im_norm + epsilon)
            if E_SPIN[ii] == E_SPIN[mm]:
                p_im = (pi - p_electrons[mm]) / 2
                p_im_norm = np.linalg.norm(p_im)
                uu_p = (r_im_norm * p_im_norm / XI_P)**2
                pauli_arg_exp_p = uu_p**2
                # Cap the argument to prevent overflows or underflows
                if pauli_arg_exp_p > 100 and ALPHA_P * (1 - pauli_arg_exp_p) < -300:
                    exp_pauli = 0.0
                elif ALPHA_P * (1 - pauli_arg_exp_p) > 300:  # exp(300) overflows
                    exp_pauli = np.exp(300)
                else:
                    exp_pauli = np.exp(ALPHA_P * (1 - pauli_arg_exp_p))
                potential_energy_pauli += (XI_P**2 / (4 * ALPHA_P * r_im_norm**2)) * exp_pauli

        # Heisenberg potential
        if (ri_norm > epsilon and pi_norm > epsilon and
                np.abs(XI_H) > epsilon):
            uu = (ri_norm * pi_norm / XI_H)**2
            hei_arg_exp = uu**2
            if hei_arg_exp > 100 and ALPHA_H * (1 - hei_arg_exp) < -300:
                exp_hei_val = 0.0
            elif ALPHA_H * (1 - hei_arg_exp) > 300:
                exp_hei_val = np.exp(300)
            else:
                exp_hei_val = np.exp(ALPHA_H * (1 - hei_arg_exp))
            potential_energy_heisenberg += (
                (XI_H**2 / (4 * ALPHA_H * ri_norm**2 * MU)) *
                exp_hei_val
            )
        

    total_energy = (kinetic_energy + potential_energy_en +
                    potential_energy_ee + potential_energy_heisenberg +
                    potential_energy_pauli)
    return (total_energy, kinetic_energy, potential_energy_en,
            potential_energy_ee, potential_energy_heisenberg, potential_energy_pauli)


# Output directory (optional, for saving)
output_dir = os.path.join(os.path.dirname(__file__), RESULTS_DIR)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# To load later:
data = np.load(os.path.join(output_dir, 'trajectory_data.npz'))
t_arr = data['t_arr']
y_arr = data['y_arr']
e_num = int(data['e_num'])
MU = data['MU']
ZZ = data['ZZ']
XI_H = data['XI_H']
ALPHA_H = data['ALPHA_H']
XI_P = data['XI_P']
ALPHA_P = data['ALPHA_P']
# electrons spin, 0 for odd, 1 for even
e_spin = np.zeros(e_num, dtype=int)
# Set the spin of electrons based on their index
for i in range(e_num):
    e_spin[i] = i % 2  # 0 for odd, 1 for even

epsilon = 1e-18  # Small constant to prevent division by zero

# After sol = solve_ivp(...)
energies = []
ke_list, pe_en_list, pe_ee_list, pe_h_list, pe_p_list = [], [], [], [], []
initial_energy_components = calculate_total_energy(
    y_arr[:, 0], MU, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, E_SPIN=e_spin
)
E0 = initial_energy_components[0]
print(f"Initial Total Energy: {E0:.6e}")
print(f"  KE: {initial_energy_components[1]:.6e}, "
      f"PE_eN: {initial_energy_components[2]:.6e}, "
      f"PE_ee: {initial_energy_components[3]:.6e}, "
      f"PE_H: {initial_energy_components[4]:.6e}",
      f"PE_P: {initial_energy_components[5]:.6e}")

for i_step in range(len(t_arr)):
    current_state = y_arr[:, i_step]
    E_tot, ke, pe_en, pe_ee, pe_h, pe_p = calculate_total_energy(
        current_state, MU, ZZ, XI_H, ALPHA_H, XI_P, ALPHA_P, E_SPIN=e_spin
    )
    energies.append(E_tot)
    ke_list.append(ke)
    pe_en_list.append(pe_en)
    pe_ee_list.append(pe_ee)
    pe_h_list.append(pe_h)
    pe_p_list.append(pe_p)

energies = np.array(energies)
relative_energy_error = ((energies - E0) / (np.abs(E0) + epsilon))

# Style settings (copied from pot_alpha.py)
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['font.size'] = 26
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['legend.fontsize'] = 20
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
plt.rcParams['text.usetex'] = False

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t_arr, energies, label='Total Energy')
plt.xlabel('Time (a.u.)')
plt.ylabel('Energy (a.u.)')
plt.title('Total Energy vs. Time')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_arr, relative_energy_error, label='Relative Energy Error')
plt.xlabel('Time (a.u.)')
plt.ylabel('(E(t) - E(0)) / E(0)')
plt.title('Relative Energy Error vs. Time')
plt.yscale('symlog', linthresh=1e-10) # Use symlog for better view of small errors
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'energy_vs_time.svg'))
plt.show()

# Optional: Plot individual energy components
plt.figure(figsize=(10,7))
plt.plot(t_arr, ke_list, label='Kinetic E')
plt.plot(t_arr, pe_en_list, label='PE e-N')
plt.plot(t_arr, pe_ee_list, label='PE e-e')
plt.plot(t_arr, pe_h_list, label='PE Heisenberg')
plt.plot(t_arr, pe_p_list, label='PE Pauli')
plt.plot(t_arr, energies, label='Total E', linestyle='--', color='black')
plt.xlabel('Time (a.u.)')
plt.ylabel('Energy Components (a.u.)')
plt.legend()
plt.grid(True)
plt.title('Energy Components vs. Time')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'energy_components_vs_time.svg'))
plt.show()
