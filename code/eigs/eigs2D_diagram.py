from tqdm import trange
import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn/")
from dyn_ultis import *
sys.path.append("./code/eigs/")
from eigs_ultis import *
from eigs_params import *

file_name_list = ['d_II','j_EE','j_EI','alpha','j_IE','j_II','sigma_IE','sigma_II']
changed_params_latex_list = [r'$d_{II}$', r'$\bar{g}_{EE}$', r'$\bar{g}_{EI}$', r'$\alpha', r'$\bar{g}_{IE}$',
                             r'$\bar{g}_{II}$', r'$g_{IE}$', r'$g_{II}$']
changed_params_list = ['d_II','g_bar_EE','g_bar_EI','alpha','g_bar_IE','g_bar_II','g_IE','g_II']
generate_func_list = [generate_params_eigs2D_d_II, generate_params_eigs2D_g_bar_EE, generate_params_eigs2D_g_bar_EI,
                      generate_params_eigs2D_alpha, generate_params_eigs2D_g_bar_IE, generate_params_eigs2D_g_bar_II, 
                      generate_params_eigs2D_g_IE, generate_params_eigs2D_g_II]

for plot_n in trange(len(file_name_list)):
    plot_eigs2D_diagram(file_name_list[plot_n], changed_params_list[plot_n], changed_params_latex_list[plot_n], generate_func_list[plot_n])

