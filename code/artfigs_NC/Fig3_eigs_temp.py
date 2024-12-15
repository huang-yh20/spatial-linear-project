import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
from tqdm import trange
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import os
import sys
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/phase/")
sys.path.append("./code/artfigs_NC/")
from dyn_ultis import *
from spatial_ultis import *
from artfigs_NC_params import *
from artfigs_NC_ultis import *

file_name_list = ['tanh_global','tanh_osc','tanh_bump','tanh_wave','tanh_chaos','tanh_stable']
p_net_eigs_list = [generate_params_phase_d_II_g_bar_II_L(10,17),generate_params_phase_d_II_g_bar_II_L(5,5),generate_params_phase_d_II_g_bar_II_L(3,12),
              generate_params_phase_d_II_g_bar_II_L(17,7), generate_params_phase_g_d_II_L_chaos(15,9),generate_params_phase_d_II_g_bar_II_L(10,10)]

if __name__ == "__main__":
    args = sys.argv[1:]
trial = int(args[0])

p_net = p_net_eigs_list[trial]
file_name = file_name_list[trial]

J = generate_net_sparse(p_net, dim=2)
J = J.toarray()
eigs, eig_V = np.linalg.eig(J)
np.save(r"./data/artfigs_NC_variousdyn_eigs_"+file_name+"eigs.npy", eigs)
np.save(r"./data/artfigs_NC_variousdyn_eigs_"+file_name+"eigV.npy", eig_V)