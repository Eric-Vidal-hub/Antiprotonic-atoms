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
sns.set_style("ticks")                      # Set ticks style
sns.set_context("paper",font_scale=1.8)     # Changing font style and size
mpl.rc('text', usetex=True)                 # Sets LaTeX as text renderer
mpl.rcParams['mathtext.fontset'] = 'cm'     # Changes size unit
plt.rcParams['figure.figsize'] = (7.50, 5)  # Sets figure size
# Adds custom color cycler with RPTU Brand colors
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#77b6ba",
                                                    "#ffa252",
                                                    "#26d07c",
                                                    "#e31b4c",
                                                    "#4c3575",
                                                    "#042c58",
                                                    "#d13896",
                                                    "#507289",
                                                    "#006b6b",
                                                    "#6ab2e7"])
plt.rcParams['lines.linewidth'] = 2     # Increases the linewidth of the graphs


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

# Creates a Directory called /Plot-Density in the folder
# previously specified
##########################################################################
path = os.getcwd() + "/" + directory + '/Plot-Density'  # path where plots should be saved
if not os.path.exists(path):        # check if folder already exists
    print('Directory not found.')
    print('Create Directory...')
    try:
        os.mkdir(path)              # try to create folder (may fail if programms running in parallel already created it)
    except OSError:
        print("Directory was already created by a different programm!")     # catch error
else:
    print('Directory exists!')

minBeta, maxBeta, numBeta = -2, 3, 50
minalpha, maxalpha, numalpha = 1, 2, 10
betas = np.logspace(minBeta, maxBeta, numBeta)
alphas = np.linspace(minalpha, maxalpha, numalpha)

for my_id in progressbar(range(id_min, id_max)):
    beta = betas[my_id % numBeta]
    alpha = alphas[int(my_id/numBeta)]

    # Try to load the variables for plotting psi0x and phi0x
    # (may fail if data has not yet been calculated)
    try:
        data = np.load(directory+"/final_densities_Well_id_"+str(my_id)+".npz")
    except OSError:
        print(id, "has not yet been calculated!")
    else:
        xx_plot = data['xx_plot']
        psi0x_plot = data['psi0x_plot']
        phi0x_plot = data['phi0x_plot']

        plt.figure(1)
        plt.plot(xx_plot, psi0x_plot, label=r'$|\psi(x)|^2 \; /n_0$')
        plt.plot(xx_plot, phi0x_plot, label=r'$|\phi(x)|^2 \; /n_0$')
        plt.xlabel(r'$x \; /\xi$')
        plt.ylabel('Densities')
        plt.title(r'Final densities for $\beta$ = %.2f, $M/m$ = %.2f' % (beta, alpha))
        plt.legend()
        plt.tight_layout()
        plt.savefig(path+"/final_densities_Well_id_"+str(my_id))
        plt.close()
