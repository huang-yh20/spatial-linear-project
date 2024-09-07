import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import os
import sys
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/phase/")
sys.path.append("./code/artfigs/")
from spatial_ultis import *
from dyn_ultis import *
from phase_params import *
from artfigs_ulits import *

def generate_params_spatial_effect(trial:int):
    rescale = 1600
    N_E, N_I = 22500, 5625
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
    J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -8/rescale, -8/rescale
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 8/rescale, 8/rescale, -16/rescale, -16/rescale
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.1

    d_list = [1, 0.2, 0.1, 0.05]
    d_EE, d_IE, d_EI, d_II = d_list[trial], d_list[trial], d_list[trial], d_list[trial]
    

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net

trial_num = 4
for trial in range(trial_num):
    p_net = generate_params_spatial_effect(trial)
    calc_eigs_bool = False
    if os.path.exists(r"data/artfigs_spatialeffect_eigs_"+str(trial)+"eigs.npy") and (not calc_eigs_bool):
        eigs = np.load(r"data/artfigs_spatialeffect_eigs_"+str(trial)+"eigs.npy")
        eig_V = np.load(r"data/artfigs_spatialeffect_eigs_"+str(trial)+"eigV.npy")
    else:
        dist_list = calc_dist(p_net, dim = 1)
        J = generate_net(p_net, dist_list)
        eigs, eig_V = np.linalg.eig(J)
        np.save(r"data/artfigs_spatialeffect_eigs_"+str(trial)+"eigs.npy", eigs)
        np.save(r"data/artfigs_spatialeffect_eigs_"+str(trial)+"eigV.npy", eig_V)
    artfigs_plot_eigs(eigs)
    plt.savefig(r"figs/artfigs_spatialeffect_eigs_"+str(trial)+".png")
    plt.close()


    temp_plot_pred(p_net, dim=1)
    artfigs_plot_eigs(eigs)
    plt.savefig(r"figs/artfigs_spatialeffect_eigs_withpred"+str(trial)+".png")
    plt.close()
