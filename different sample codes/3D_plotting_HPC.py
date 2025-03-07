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
plt.rcParams['figure.figsize'] = (7.50, 5)                      # Sets figure size to 7.50cm by 5.00cm
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


# Specify values which have been simulated
##########################################################################
minBeta, maxBeta, numBeta = -2, 3, 50
minalpha, maxalpha, numalpha = 1, 2, 10
betas = np.logspace(minBeta, maxBeta, numBeta)
alphas = np.linspace(minalpha, maxalpha, numalpha)


# Load in Data, calculate variance and mean positon of impurity and store
# it in a matrix for plotting
##########################################################################
var_imp_list = np.zeros((len(alphas), len(betas)))      # set matrix to save variance
mean_pos_list = np.zeros((len(alphas), len(betas)))     # set matrix to save mean position

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

        # Save variance and mean position values in matrices at the corresponding indices
        var_imp_list[int(my_id/numBeta), my_id % numBeta] = variance_impurity(xx_plot, psi0x_plot)
        mean_pos_list[int(my_id/numBeta), my_id % numBeta] = exp_position_impurity(xx_plot, psi0x_plot)

try:
    transition_line = np.load("translition_line.npy").transpose()
except:
    print("No DMRG transition line data found!")
    print("continue without...")


# Plot in 3D Colorplot
##########################################################################
# Create grid of all alpha and beta values
# alpha_grid first row (alpha_0, alpha_0,...), second row (alpha_1, alpha_1,...), ...
# beta_grid first row (beta_0, beta_1,...), second row (beta_0, beta_1,...), ...
alpha_grid, beta_grid = np.meshgrid(alphas, betas, indexing="ij")   

# Plot Variance
im = plt.pcolor(beta_grid, alpha_grid, var_imp_list,cmap='viridis')         # Plot Variance Matrix where each entry gets a alpha and beta value from the alpha/beta grids
cbar = plt.colorbar(im)
cbar.set_label("$\Delta \hat{X}_\psi/\\xi$")
try:
    plt.plot(transition_line[1],transition_line[0],color="#e31b4c",label="DMRG phase change")
    plt.legend()
except:
    x = 1
plt.xscale("log")
plt.xlabel("$g_\mathrm{IB}/g$")
plt.ylabel("$M/m$")
plt.savefig(path + "/variance.png", bbox_inches='tight')
plt.close()

# Plot mean position
im = plt.pcolor(beta_grid, alpha_grid, abs(mean_pos_list),cmap='viridis_r')   # Plot mean position Matrix where each entry gets a alpha and beta value from the alpha/beta grids
cbar = plt.colorbar(im)
cbar.set_label("$|<\psi|\hat{X}|\psi>|/\\xi$")
try:
    plt.plot(transition_line[1],transition_line[0],color="#e31b4c",label="DMRG phase change")
    plt.legend()
except:
    x = 1
plt.xscale("log")
plt.xlabel("$g_\mathrm{IB}/g$")
plt.ylabel("$M/m$")
plt.savefig(path + "/mean_pos.png", bbox_inches='tight')
plt.close()