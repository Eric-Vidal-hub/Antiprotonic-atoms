import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import matplotlib as mpl
import sys
import os
from progressbar import progressbar

# Changing the style of the Plots
##########################################################################
sns.set_style("ticks")                                          # IDK
sns.set_context("paper",font_scale=1.8)                         # Changing font style and size
mpl.rc('text', usetex=False)                                    # IDK
mpl.rcParams['mathtext.fontset'] = 'cm'                         # Changes unit in which sizes of plots is calculated
plt.rcParams['figure.figsize'] = (7.50*1.5, 5*2)                      # Sets figure size to 7.50cm by 5.00cm
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#77b6ba",  # Adds custom color cycler with RPTU Brand colors
                                                    "#ffa252",
                                                    "#26d07c",
                                                    "#e31b4c",
                                                    "#4c3575",
                                                    "#042c58",
                                                    "#d13896",
                                                    "#507289",
                                                    "#006b6b",
                                                    "#6ab2e7"])
plt.rcParams['lines.linewidth'] = 2                             # Increases the linewidth of the graphs

# For executing the file one have to call the python environment,
# path to code, files directory name, initial id, and final id
# EXAMPLE:
# C:/Users/propietario/AppData/Local/Programs/Python/Python310/python.exe
# argv[0] = c:/Users/propietario/Documents/Hiwi-work/plotting_HPC.py
# argv[1] = final_densities_Well_id_
# argv[2] = 0 (int)
# argv[3] = 500 (int)
if len(sys.argv) < 4:
    raise (ValueError("Programm needs to be called as: python " + sys.argv[0]
                      + " <directory> <id_min> <id_max>"))
directory = sys.argv[1]
id_min = int(sys.argv[2])   # Initial id str to int
id_max = int(sys.argv[3])   # Final id str to int

# Creates a Directory called Plot-Impurity-Variance in the folder
# previously specified
##########################################################################
path = os.getcwd() + "/" + directory + '/Plot-Impurity-Variance'            # path where plots should be saved
if not os.path.exists(path):                                                # check if folder already exists
    print('Directory not found.')
    print('Create Directory...')
    try:
        print(path)
        os.mkdir(path)                                                      # try to create folder (may fail if programms running in parallel already created it)
    except OSError:
        print("Directory was already created by a different programm!")     # catch error
else:
    print('Directory exists!')

# Function to calculate variance
##########################################################################

# <X>
def exp_position_impurity(xx, psi):
    psi_norm = psi/np.sum(psi)      #normalize psi to 1 so it's a probability distribution
    return np.sum(xx * psi_norm)

# <X^2>
def exp_position_squared_impurity(xx, psi):
    psi_norm = psi/np.sum(psi)      #normalize psi to 1 so it's a probability distribution
    return np.sum(xx**2 * psi_norm)

# Delta X
def variance_impurity(xx, psi):
    X = exp_position_impurity(xx, psi)
    X_squared = exp_position_squared_impurity(xx, psi)
    return np.sqrt(X_squared - X**2)

# Expected variation of Impurity for g_IB = 0
def variance_gib_0(xx):
    psi = np.cos(xx*np.pi/(xx[-1]-xx[0]))**2
    return variance_impurity(xx, psi)

# Analytic approximation for self trapped polaron
##########################################################################

def TF_approx(gamma, alpha, beta, xx_plot, psi0x_plot):
    # TF approx
    alpha_TF = 1 / alpha
    zeta = (beta) ** 2 * gamma / alpha_TF  # Self-trapping parameter | 2*beta because g in paper is g/2 for us
    lamb = 2 / zeta                      # Localization length
    # TF approx Impurity wavefunction
    xi = 1 / (np.sqrt(2 * lamb) * np.cosh(xx_plot / lamb))
    # Analytic density
    return xi ** 2 / np.sum(xi**2) * np.sum(psi0x_plot)

def TF_approx_var(gamma, alpha, beta, xx_plot, psi0x_plot):
    psi0_ana = TF_approx(gamma, alpha, beta, xx_plot, psi0x_plot)
    return variance_impurity(xx_plot, psi0_ana)


# Specify values which have been simulated
##########################################################################
minBeta, maxBeta, numBeta = -2, 3, 50
minalpha, maxalpha, numalpha = 1, 2, 10
betas = np.logspace(minBeta, maxBeta, numBeta)
alphas = np.linspace(minalpha, maxalpha, numalpha)
gamma = 0.4


# Load in Data, calculate variance and mean positon of impurity and store
# it in a matrix for plotting
##########################################################################
var_imp_list = np.zeros((len(alphas), len(betas)))      # set matrix to save variance
mean_pos_list = np.zeros((len(alphas), len(betas)))     # set matrix to save mean position
var_imp_self_trap_ana = np.zeros((len(alphas), len(betas))) # set matrix to save variance of impurity from analytic formular
xx_plot = 0

for my_id in progressbar(range(id_min, id_max)):
    beta = betas[my_id % numBeta]
    alpha = alphas[int(my_id/numBeta)]

    # Try to load the variables for plotting psi0x and phi0x
    # (may fail if data has not yet been calculated)
    try:
        data = np.load(directory+"/final_densities_Well_id_"+str(my_id)+".npz")
    except OSError:
        print(my_id, "has not yet been calculated!")
    else:
        xx_plot = data['xx_plot']
        psi0x_plot = data['psi0x_plot']
        phi0x_plot = data['phi0x_plot']

        # Save variance values in matrices at the corresponding indices
        var_imp_list[int(my_id/numBeta), my_id % numBeta] = variance_impurity(xx_plot, psi0x_plot) / np.sqrt(2*gamma)
        mean_pos_list[int(my_id/numBeta), my_id % numBeta] = abs(exp_position_impurity(xx_plot, psi0x_plot)) / np.sqrt(2*gamma)
        var_imp_self_trap_ana[int(my_id/numBeta), my_id % numBeta] = TF_approx_var(gamma, alpha, beta, xx_plot, psi0x_plot) / np.sqrt(2*gamma)

betas_DMRG = np.load("betas.npy")
var_imp_list_DMRG = np.load("var_imp.npy") / np.sqrt(2*gamma)
mean_pos_list_DMRG = abs(np.load("pos_imp.npy")) / np.sqrt(2*gamma)

alpha_id_cut1 = 1
alpha_id_cut2 = 7

DMRG_color = "#77b6ba"
SSM_color = "#ffa252"
grey = "#e31b4c"

# Plot in 3D Colorplot
##########################################################################
# Create grid of all alpha and beta values
# alpha_grid first row (alpha_0, alpha_0,...), second row (alpha_1, alpha_1,...), ...
# beta_grid first row (beta_0, beta_1,...), second row (beta_0, beta_1,...), ...
alpha_grid, beta_grid = np.meshgrid(alphas, betas * gamma, indexing="ij") 
alpha_grid_DMRG, beta_grid_DMRG = np.meshgrid(alphas, betas_DMRG * gamma, indexing="ij")   

# Variance
##########################################################################

fig = plt.figure(1)
# set up subplot grid
mpl.gridspec.GridSpec(2,2)  # Grid of figure. Here (2,2) so one can have maximaly 4 subplots

var_max = np.amax(np.array([var_imp_list,var_imp_list_DMRG]))
var_min = np.amin(np.array([var_imp_list,var_imp_list_DMRG]))

ax0 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
im0 = ax0.pcolor(beta_grid_DMRG, alpha_grid_DMRG, var_imp_list_DMRG,cmap='viridis',rasterized=True, vmin=var_min, vmax=var_max)         # Plot Variance Matrix where each entry gets a alpha and beta value from the alpha/beta grids
#cbar0 = plt.colorbar(im0)
#cbar0.set_label("$\Delta \hat{X}_\psi/\\xi$")
ax0.axhline(y=alphas[alpha_id_cut1],color=grey)
ax0.axhline(y=alphas[alpha_id_cut2],ls="--",color=grey)
ax0.set_xscale("log")
ax0.set_xlabel("$g_\mathrm{IB}\\frac{m}{n_0}$")
ax0.set_ylabel("$M/m$")
ax0.text(.99, .95, "DMRG", ha='right', va='top', bbox=dict(facecolor='white', alpha=1), transform=ax0.transAxes)    #places a small text box in the upper right corner of the plot
ax0.set_title("a.)",loc='left')

ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1, sharey=ax0)
im1 = ax1.pcolor(beta_grid, alpha_grid, var_imp_list, cmap='viridis',rasterized=True, vmin=var_min, vmax=var_max)         # Plot Variance Matrix where each entry gets a alpha and beta value from the alpha/beta grids
#cbar1 = plt.colorbar(im1)
#cbar1.set_label("$\Delta \hat{X}_\psi/\\xi$")
ax1.axhline(y=alphas[alpha_id_cut1],color=grey)
ax1.axhline(y=alphas[alpha_id_cut2],ls="--",color=grey)
ax1.set_xscale("log")
ax1.set_xlabel("$g_\mathrm{IB}\\frac{m}{n_0}$")
#ax1.set_ylabel("$M/m$")
plt.tick_params('y', labelleft=False)
ax1.text(.99, .95, "DMF", ha='right', va='top', bbox=dict(facecolor='white', alpha=1), transform=ax1.transAxes)
ax1.set_title("b.)",loc='left')

ax2 = plt.subplot2grid((2,2), (1,0), colspan=2, rowspan=1)  # creates subplot in the lower left corner which spans over 1 row but 2 columns so it spans over the entire lower row
ax2.plot(betas_DMRG, var_imp_list_DMRG[alpha_id_cut1],label="DMRG",color=DMRG_color)
ax2.plot(betas_DMRG, var_imp_list_DMRG[alpha_id_cut2],ls="--",color=DMRG_color)
ax2.plot(betas, var_imp_list[alpha_id_cut1],label="DMF",color=SSM_color)
ax2.plot(betas, var_imp_list[alpha_id_cut2],ls="--",color=SSM_color)
ax2.plot(betas, var_imp_self_trap_ana[alpha_id_cut1], label="Analytic", color="#26d07c")
#ax2.plot(betas, var_imp_self_trap_ana[alpha_id_cut2], ls="--", color="#26d07c")
ax2.axhline(y=variance_gib_0(xx_plot) / np.sqrt(2 * gamma),ls="-.",color="#7f7f7f",label="$\Delta \hat{X}_\psi(g_\mathrm{IB}=0)$")
ax2.set_xscale("log")
ax2.set_xlabel("$g_\mathrm{IB}\\frac{m}{n_0}$")
ax2.set_ylabel("$\Delta \hat{X}_\psi\cdot n_0$")
ax2.set_title("c.)",loc='left')
ax2.legend()
ax2.set_xlim(xmax=10**2)
ax2.set_ylim(ymax=10)

fig.tight_layout()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.585, 0.025, 0.362])
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label("$\Delta \hat{X}_\psi\cdot n_0$")

ax2_bbox = ax2.get_position()
ax2_bbox = ax2_bbox.get_points().flatten()
ax2.set_position(mpl.transforms.Bbox.from_extents(ax2_bbox[0], ax2_bbox[1], 0.96275108, ax2_bbox[3]))

plt.savefig(path + "/variance_comparison.pdf", bbox_inches='tight')
plt.close()

# Mean position
##########################################################################

fig = plt.figure(1)
# set up subplot grid
mpl.gridspec.GridSpec(2,2)  # Grid of figure. Here (2,2) so one can have maximaly 4 subplots

var_max = np.amax(np.array([mean_pos_list,mean_pos_list_DMRG]))
var_min = np.amin(np.array([mean_pos_list,mean_pos_list_DMRG]))

ax0 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
im0 = ax0.pcolor(beta_grid_DMRG, alpha_grid_DMRG, mean_pos_list_DMRG,cmap='viridis_r',rasterized=True, vmin=var_min, vmax=var_max)         # Plot Variance Matrix where each entry gets a alpha and beta value from the alpha/beta grids
#cbar0 = plt.colorbar(im0)
#cbar0.set_label("$|<\psi|\hat{X}|\psi>|/\\xi$")
ax0.axhline(y=alphas[alpha_id_cut1],color=grey)
ax0.axhline(y=alphas[alpha_id_cut2],ls="--",color=grey)
ax0.set_xscale("log")
ax0.set_xlabel("$g_\mathrm{IB}\\frac{m}{n_0}$")
ax0.set_ylabel("$M/m$")
ax0.text(.99, .95, "DMRG", ha='right', va='top', bbox=dict(facecolor='white', alpha=1), transform=ax0.transAxes)    #places a small text box in the upper right corner of the plot
ax0.set_title("a.)",loc='left')

ax1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1, sharey=ax0)
im1 = ax1.pcolor(beta_grid, alpha_grid, mean_pos_list, cmap='viridis_r',rasterized=True, vmin=var_min, vmax=var_max)         # Plot Variance Matrix where each entry gets a alpha and beta value from the alpha/beta grids
#cbar1 = plt.colorbar(im1)
#cbar1.set_label("$|<\psi|\hat{X}|\psi>|/\\xi$")
ax1.axhline(y=alphas[alpha_id_cut1],color=grey)
ax1.axhline(y=alphas[alpha_id_cut2],ls="--",color=grey)
ax1.set_xscale("log")
ax1.set_xlabel("$g_\mathrm{IB}\\frac{m}{n_0}$")
#ax1.set_ylabel("$M/m$")
plt.tick_params('y', labelleft=False)
ax1.text(.99, .95, "DMF", ha='right', va='top', bbox=dict(facecolor='white', alpha=1), transform=ax1.transAxes)
ax1.set_title("b.)",loc='left')

ax2 = plt.subplot2grid((2,2), (1,0), colspan=2, rowspan=1)  # creates subplot in the lower left corner which spans over 1 row but 2 columns so it spans over the entire lower row
ax2.plot(betas_DMRG, mean_pos_list_DMRG[alpha_id_cut1],label="DMRG",color=DMRG_color)
ax2.plot(betas_DMRG, mean_pos_list_DMRG[alpha_id_cut2],ls="--",color=DMRG_color)
ax2.plot(betas, mean_pos_list[alpha_id_cut1],label="DMF",color=SSM_color)
ax2.plot(betas, mean_pos_list[alpha_id_cut2],ls="--",color=SSM_color)
#ax2.axhline(y=variance_gib_0(xx_plot),ls="-.",color="#7f7f7f")
ax2.set_xscale("log")
ax2.set_xlabel("$g_\mathrm{IB}\\frac{m}{n_0}$")
ax2.set_ylabel("$|<\psi|\hat{X}|\psi>|\cdot n_0$")
ax2.set_title("c.)",loc='left')
ax2.legend()
ax2.set_xlim(xmax=10**2)

fig.tight_layout()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.585, 0.025, 0.362])
cbar = fig.colorbar(im1, cax=cbar_ax)
cbar.set_label("$|<\psi|\hat{X}|\psi>|\cdot n_0$")

ax2_bbox = ax2.get_position()
ax2_bbox = ax2_bbox.get_points().flatten()
ax2.set_position(mpl.transforms.Bbox.from_extents(ax2_bbox[0], ax2_bbox[1], 0.96275108, ax2_bbox[3]))

plt.savefig(path + "/mean_pos_comparison.pdf", bbox_inches='tight')
plt.close()
