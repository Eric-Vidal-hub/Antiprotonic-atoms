# Simulation of Antiproton Capture by Atomic Systems using Fermion Molecular Dynamics

This repository contains the Python source code and associated analysis scripts for the Master's Thesis titled _"Simulation of the antiproton capture by atomic systems"_. The project was conducted as part of the AEgIS Collaboration at CERN.

The primary goal of this work is to simulate the capture process of low-energy antiprotons by various multi-electronic atomic systems (neutral atoms, cations, and anions). The simulations are used to calculate capture cross-sections, analyze particle trajectories, and study the underlying dynamics of energy exchange, providing valuable insights for experiments at CERN's Antimatter Factory.

## Core Methodology: Fermion Molecular Dynamics (FMD)

The simulations are based on the **Fermion Molecular Dynamics (FMD)** model, a semiclassical approach designed to study the dynamics of N-body systems where quantum effects are important but a full quantum mechanical treatment is computationally intractable.

The FMD model, originally proposed for nuclear collisions by Wilets et al. and later extended to atomic systems by Kirschbaum and Wilets, retains essential quantum features through the use of effective, momentum-dependent pseudopotentials. These potentials enforce key quantum principles:
1.  **Heisenberg Uncertainty Principle:** A repulsive potential between each electron and the nucleus prevents atomic collapse. A similar potential is applied to the antiproton-nucleus interaction.
2.  **Pauli Exclusion Principle:** A repulsive potential acts between electrons of identical spin, preventing them from occupying the same phase space volume.

This approach allows for the simulation of complex processes like multiple ionization and particle rearrangement while keeping track of the trajectory of every particle in the system.

## Repository Structure

The repository is organized into simulation scripts, analysis/plotting scripts, and directories for storing results.

```
.
├── GS_alpha_HPC/                  # OUTPUT: Ground state data for neutral atoms
├── GS_alpha_anions_HPC/           # OUTPUT: Ground state data for anions (Z, e=Z+1)
├── GS_alpha_pos_ions_HPC/         # OUTPUT: Ground state data for cations (Z, e=Z-1)
├── CCS_.../                       # OUTPUT: Cross-section simulation results
├── capture_trajectory_details/    # OUTPUT: Detailed data from single capture runs
└── PBAR_ATOM_EVOLUTION_RUNS/      # OUTPUT: Data from pbar-atom evolution simulations
|
├── v0_...py                       # Scripts for single trajectory analysis and plotting
├── v2_...py                       # Scripts for multi-electron atom evolution (no pbar)
├── v3-2_gs_run_HPC.py             # SCRIPT: Ground state optimization
├── v5_ccs_run_HPC.py              # SCRIPT: Monte Carlo capture cross-section calculation
├── v6_...py                       # (Not present, development version)
├── v7_pbar_atom_evo_run.py        # SCRIPT: Simulates a single pbar-atom trajectory in detail
├── v7_pbar_atom_plots.py          # SCRIPT: Plots results from v7_pbar_atom_evo_run.py
├── plot_electron_binding_differences.py # SCRIPT: Analysis of IP and EA
└── reproduce_kw_tables...py       # SCRIPT: Formats results to match Kirschbaum & Wilets (1980) paper
```

## Workflow & Usage

The project workflow is divided into three main stages: Ground State Calculation, Dynamical Simulation, and Analysis/Plotting.

### 1. Prerequisites & Setup

Ensure you have Python 3.x installed along with the necessary libraries.

```bash
# Clone the repository
git clone https://github.com/Eric-Vidal-hub/Antiprotonic-atoms.git
cd Antiprotonic-atoms

# Install required packages
pip install numpy scipy pandas matplotlib tqdm Pillow
```

### 2. Ground State Calculation

Before any dynamical simulation, the ground state configuration of the target atom/ion must be calculated. This is done by minimizing the FMD Hamiltonian.

-   **Script:** `v3-2_gs_run_HPC.py`
-   **Function:** Finds the minimum energy configuration (positions and momenta of electrons) for a given atom or ion.
-   **Usage:** The script is designed for HPC and takes command-line arguments.
    ```bash
    # Example: Calculate ground state for Helium (Z=2, e=2) and save to GS_alpha_HPC
    # python v3-2_gs_run_HPC.py <output_directory> <proton_number> <electron_number_offset>
    python v3-2_gs_run_HPC.py ./GS_alpha_HPC 0 2 
    # This would run for p_num=2, e_num=2+0=2.

    # Example: Calculate ground state for Li+ (Z=3, e=2)
    python v3-2_gs_run_HPC.py ./GS_alpha_pos_ions_HPC -1 3
    # This would run for p_num=3, e_num=3-1=2.
    ```
-   **Output:** A `.csv` file (e.g., `02_He_02e.csv`) containing the optimal configuration.

### 3. Dynamical Simulations

Once a ground state is computed, you can simulate the antiproton collision.

#### A) Capture Cross-Section Calculation (Monte Carlo)

This script runs many trajectories over a range of energies to calculate the capture cross-section.

-   **Script:** `v5_ccs_run_HPC.py`
-   **Function:** Simulates thousands of antiproton-atom collisions with randomized impact parameters to determine capture probabilities.
-   **Usage:** Configure the target atom, energy range, and other parameters in `v5_ccs_FMD_constants_HPC.py`. The script is parallelized and intended for HPC. It takes arguments for the output directory and the energy step ID.
    ```bash
    # Example: Run the 3rd energy step for a configured target
    python v5_ccs_run_HPC.py ./CCS_He_results 3
    ```
-   **Output:** `.csv` files containing cross-section data and final state information for each energy.

#### B) Single Trajectory Simulation & Capture Analysis

This script simulates a single, specific trajectory in high detail until capture occurs or time runs out.

-   **Script:** `v7_pbar_atom_evo_run.py`
-   **Function:** Propagates the full system (antiproton + target atom) and saves the complete state vector at many time steps. Useful for visualizing the capture mechanism.
-   **Usage:** Configure the target atom, antiproton initial energy (`E0`), and impact parameter (`BB`) in `v7_pbar_atom_evo_constants.py`. Then run the main script.
    ```bash
    python v7_pbar_atom_evo_run.py
    ```
-   **Output:** A detailed `.npz` file with the full numerical data and a human-readable `.csv` file with particle positions and momenta over time.

### 4. Analysis and Plotting

After running the simulations, use the plotting scripts to visualize the results.

-   **Scripts:** `v5_css_plot_HPC.py`, `v7_pbar_atom_plots.py`, `plot_electron_binding_differences.py`, `reproduce_kw_tables...py`.
-   **Function:** These scripts read the `.csv` or `.npz` files generated by the simulation steps and create plots of cross-sections, particle trajectories, energy evolution, animations (GIFs), and summary tables.
-   **Usage:** Configure the relevant `...constants.py` file to point to the correct results directory, then run the plotting script.
    ```bash
    # Example: Plot the results from a cross-section run
    # (after configuring v5_ccs_FMD_constants_HPC.py to point to the right RESULTS_DIR)
    python v5_css_plot_HPC.py
    
    # Example: Plot the results from a single trajectory run
    # (after configuring v7_pbar_atom_evo_constants.py)
    python v7_pbar_atom_plots.py
    ```

## Citation

If you use this code or results from this work, please cite the corresponding Master's Thesis:

> E. Vidal Marcos, "Simulation of the antiproton capture by atomic systems", Master's Thesis, Aarhus University and CERN (2025).

A BibTeX entry can be formatted as:
```bibtex
@mastersthesis{VidalMarcos2025,
  author  = {Vidal Marcos, Eric},
  title   = {Simulation of the antiproton capture by atomic systems},
  school  = {Aarhus University and CERN},
  year    = {2025},
  month   = {June}
}
```

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for details.
