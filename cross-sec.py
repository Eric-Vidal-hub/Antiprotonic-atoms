import numpy as np
import matplotlib.pyplot as plt

def electron_ionization_cross_section(E, B=13.6):
    """
    Calculate the electron-impact ionization cross section for hydrogen 
    using the BEB model.
    
    Parameters:
      E : array_like
          Electron energies in eV.
      B : float, optional
          Ionization energy in eV (default is 13.6 eV for hydrogen).
    
    Returns:
      sigma : array_like
          Cross section in cm^2.
    """
    # Bohr radius in cm
    a0 = 5.29177e-9  # cm
    # Prefactor S = 4*pi*a0^2
    S = 4 * np.pi * a0**2

    sigma = np.zeros_like(E)
    mask = E >= B  # Ionization occurs only for E >= B
    t = E[mask] / B
    sigma[mask] = S * (np.log(t)/t) * (1 - 1/t + 1/(2*t**2))
    return sigma

def positron_ionization_cross_section(E, B=13.6):
    """
    Calculate an approximate positron-impact ionization cross section for hydrogen.
    
    This model omits the exchange term (1/(2*t^2)) that is present in the electron BEB model.
    
    Parameters:
      E : array_like
          Positron energies in eV.
      B : float, optional
          Ionization energy in eV (default is 13.6 eV for hydrogen).
    
    Returns:
      sigma : array_like
          Cross section in cm^2.
    """
    # Bohr radius in cm
    a0 = 5.29177e-9  # cm
    # Prefactor S = 4*pi*a0^2
    S = 4 * np.pi * a0**2

    sigma = np.zeros_like(E)
    mask = E >= B  # Ionization occurs only for E >= B
    t = E[mask] / B
    sigma[mask] = S * (np.log(t)/t) * (1 - 1/t)
    return sigma

def main():
    # Define an energy range from 10 to 1000 eV
    E = np.linspace(10, 1000, 1000)
    
    # Compute the cross sections
    sigma_e = electron_ionization_cross_section(E)
    sigma_p = positron_ionization_cross_section(E)
    
    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(E, sigma_e, label='Electron Impact (BEB Model)', lw=2)
    plt.plot(E, sigma_p, label='Positron Impact (Modified BEB)', lw=2, linestyle='--')
    plt.xlabel('Incident Particle Energy (eV)', fontsize=12)
    plt.ylabel('Ionization Cross Section (cm²)', fontsize=12)
    plt.title('Hydrogen Ionization Cross Section', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()
