import os 
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
from scipy.ndimage import convolve
from tqdm import trange
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from typing import NamedTuple, Union, Callable, List
import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn/")
from dyn_ultis import *
sys.path.append("./code/eigs/")
from eigs_ultis import *
from eigs_params import *

file_name_old_list = ['d_II','j_EE','j_EI','alpha','j_IE','j_II','sigma_IE','sigma_II']
file_name_new_list = ['d_II','g_bar_EE','g_bar_EI','alpha','g_bar_IE','g_bar_II','g_IE','g_II']
generate_func_list = [generate_params_eigs2D_d_II, generate_params_eigs2D_g_bar_EE, generate_params_eigs2D_g_bar_EI,
                      generate_params_eigs2D_alpha, generate_params_eigs2D_g_bar_IE, generate_params_eigs2D_g_bar_II, 
                      generate_params_eigs2D_g_IE, generate_params_eigs2D_g_II]
repeat_num = 10

if __name__ == "__main__":
    args = sys.argv[1:]
file_name_n, param_n = int(args[0]), int(args[1])

#TEMP 
for repeat_trial in range(repeat_num):
    if os.path.exists(r'./data/' +file_name_old_list[file_name_n] + str(repeat_trial) + str(param_n) + 'eig.npy'):
        eigs = np.load(r'./data/' +file_name_old_list[file_name_n] + str(repeat_trial) + str(param_n) + 'eig.npy')
        np.save(r'./data/eigs_' +file_name_new_list[param_n] + '_' + str(param_n) + '_' + str(repeat_trial) + 'eig.npy', eigs)
        os.remove(r'./data/' +file_name_old_list[file_name_n] + str(repeat_trial) + str(param_n) + 'eig.npy')
    else:
        generate_eigs_pramas = generate_func_list[param_n]
        p_net = generate_eigs_pramas(param_n)

        dist_list = calc_dist(p_net, dim = 2)
        J = generate_net(p_net, dist_list)
        eigs, eig_V = np.linalg.eig(J)
        np.save(r'./data/eigs_' +file_name_new_list[param_n] + '_' + str(param_n) + '_' + str(repeat_trial) + 'eig.npy', eigs)
        