import numpy as np
from scipy.fft import fft, ifft
import sys
import os


directory = sys.argv[1]
id = int(sys.argv[2])

path = directory
if os.path.exists(path) != True:
    print('Directory not found.')
    print('Create Directory...')
    try:
        os.mkdir(path)
    except:
        print("Directory was already created by a different demon!")
else:
    print('Directory exists!')

# PARAMETERS DEFINITION
minBeta, maxBeta, numBeta = -2, 3, 50
minalpha, maxalpha, numalpha = 1, 2, 10
betas = np.logspace(minBeta, maxBeta, numBeta)
alphas = np.linspace(minalpha, maxalpha, numalpha)
maxIterations, threshold, energy_check = 1e8, 1e-10, 2000


beta = betas[id % numBeta]
alpha = alphas[int(id/numBeta)]


# Constants definition
aa, numPoints, numBosons, m_bosons, gamma = 0.1, 2**8, 40, 1, 0.4
gg, m_impurity, unitlessL = gamma * numBosons / \
    (m_bosons * numPoints), alpha * m_bosons, np.sqrt(2 * gamma) * numBosons
# Grid and potential definition
gridLength, xx, dx = numPoints, np.arange(numPoints), 1
kc, dk, kk = 1 / 2, 1 / \
    numPoints, np.linspace(-1 / 2, 1 / 2 - 1 / numPoints, numPoints)
vx_WELL, vx_HO = np.zeros_like(xx), (xx - numPoints / 2.)**2 * 0.000000005
vx_WELL[:1], vx_WELL[-1:] = 1000, 1000
vxx = vx_WELL  # Well potential selected

# Pre-calculate pp and constant parts of the exponentials outside the loop
dtau = -1j
pp = 2 * np.pi * np.linspace(-kc, kc - dk, numPoints)
exp_factor_impurity = np.exp(-1j * dtau * pp**2 / (2.0 * m_impurity))
exp_factor_bosons = np.exp(-1j * dtau * pp**2 / (2.0 * m_bosons))


g_ib = beta * gg
psi0x, phi0x = np.sin(xx * np.pi / numPoints), np.concatenate((np.tanh(xx[:numPoints // 2] * numBosons * np.sqrt(
    gamma) / numPoints), -np.tanh((xx[numPoints // 2:] - numPoints) * numBosons * np.sqrt(gamma) / numPoints)))
psi0x, phi0x = psi0x / np.sqrt(np.sum(np.abs(psi0x)**2)), phi0x / \
    np.sqrt(np.sum(np.abs(phi0x)**2)) * np.sqrt(numBosons)
energy, count = np.inf, 0

while count < maxIterations:
    abs_psi0x_sq, abs_phi0x_sq = np.abs(psi0x)**2, np.abs(phi0x)**2
    psi1x, phi1x = np.exp(-(1j * dtau / 2.0) * (vxx + g_ib * abs_phi0x_sq)) * psi0x, np.exp(-(
        1j * dtau / 2.0) * (vxx + g_ib * abs_psi0x_sq + gg * abs_phi0x_sq)) * phi0x
    psi1p, phi1p = np.fft.fftshift(
        fft(psi1x)) * dx, np.fft.fftshift(fft(phi1x)) * dx

    # Use pre-calculated parts of the exponentials
    psi2p, phi2p = exp_factor_impurity * psi1p, exp_factor_bosons * phi1p

    psi2x, phi2x = ifft(np.fft.ifftshift(psi2p)) / \
        dx, ifft(np.fft.ifftshift(phi2p)) / dx
    psifx1, phifx1 = np.exp(-(1j * dtau / 2.0) * (vxx + g_ib * np.abs(phi0x - phi1x + phi2x)**2)) * psi2x, np.exp(-(
        1j * dtau / 2.0) * (vxx + g_ib * np.abs(psi0x - psi1x + psi2x)**2 + gg * np.abs(phi0x - phi1x + phi2x)**2)) * phi2x
    psi0x, phi0x = psifx1 / np.sqrt(np.sum(np.abs(psifx1)**2)), phifx1 / \
        np.sqrt(np.sum(np.abs(phifx1)**2)) * np.sqrt(numBosons)
    count += 1
    """
    if count % energy_check == 0:
        e_impurity, e_bosons = np.sum(-0.5 / m_impurity * (np.roll(psi0x, 1) - 2 * psi0x + np.roll(psi0x, -1)) / dx**2 + g_ib * np.abs(psi0x)**2 * np.abs(
            phi0x)**2) * dx, np.sum(-0.5 / m_bosons * (np.roll(phi0x, 1) - 2 * phi0x + np.roll(phi0x, -1)) / dx**2 + gg / 2 * np.abs(phi0x)**4) * dx
        process_energy = np.abs(e_impurity + e_bosons)
        if np.abs(energy - process_energy) < threshold:
            # print(f'Energy has converged for beta={beta:.2f}')
            break
        energy = process_energy
    """
        
xx_plot, psi0x_plot, phi0x_plot = xx * unitlessL / numPoints - unitlessL / \
    2., (np.abs(psi0x))**2 * numPoints / \
    numBosons, (np.abs(phi0x))**2 * numPoints / numBosons
np.savez(path+"/final_densities_Well_id_"+str(id), xx_plot=xx_plot,
         psi0x_plot=psi0x_plot, phi0x_plot=phi0x_plot)
