# Physical and model constants
ALPHA = 5
XI_H = 1.000
XI_H_RYD = 1000
XI_P = 2.767

# Scaling parameters according to alpha
XI_H /= (1 + 1 / (2 * ALPHA))**0.5
XI_P /= (1 + 1 / (2 * ALPHA))**0.5

MAXITER = 20
GTOL = 1e-4

# Directory and file settings
ELEMENTS_LIST = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U'
]

START_FILE = '01_H_01e.csv'
END_FILE = '38_Sr_38e.csv'

# Default directories (can be overridden by sys.argv in scripts)
# RESULTS_DIR = 'GS_feedback_HPC'
# RESULTS_DIR = 'GS_alpha_HPC'
# RESULTS_DIR = 'GS_alpha_anions_HPC'
# RESULTS_DIR = 'GS_alpha_pos_ions_HPC'
RESULTS_DIR = 'GS_alpha_neutral_ryd_HPC'
DEFAULT_RESULTS_DIR = 'c:/Users/propietario/Documents/Antiprotonic-atoms/' + RESULTS_DIR
DEFAULT_PLOTS_DIR = DEFAULT_RESULTS_DIR + '/plots'
DEFAULT_NIST_DIR = 'c:/Users/propietario/Documents/Antiprotonic-atoms/LDA/neutrals'