import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import sys
import os
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/phase/")
sys.path.append("./code/artfigs/")
from spatial_ultis import *
from dyn_ultis import *
from phase_params import *
from dyn_params import *
from artfigs_ulits import *

p_net = generate_params_dyn_global(3)

calc_eigs_bool = False
if os.path.exists("./data/artfigs_eigsandeigV1D_eigs.npy") and (not calc_eigs_bool):
    eigs = np.load("./data/artfigs_eigsandeigV1D_eigs.npy")
    eig_V = np.load("./data/artfigs_eigsandeigV1D_eigV.npy")
else:
    dist_list = calc_dist(p_net, dim = 1)
    J = generate_net(p_net, dist_list)
    eigs, eig_V = np.linalg.eig(J)
    np.save("./data/artfigs_eigsandeigV1D_eigs.npy", eigs)
    np.save("./data/artfigs_eigsandeigV1D_eigV.npy", eig_V)

#eigV ploted
outlier_plot_num = 21
bulk_plot_num = 1

#outlier eigs ploted indice
outlier_eig_indice = find_points(eigs, 100+0j, outlier_plot_num)

#bulk eigs ploted indice
bulk_eig_indice = find_points(eigs, 0.1+0j, bulk_plot_num)

artfigs_plot_eigs(eigs)
plt.savefig(r"figs/artfigs_eigsandeigV1D_eigs.png")
plt.close()

temp_plot_pred(p_net, dim=1)
artfigs_plot_eigs(eigs)
plt.savefig(r"figs/artfigs_eigsandeigV1D_eigs_withpred.png")
plt.close()

eigs_indice = outlier_eig_indice + bulk_eig_indice
for eig_trial in range(len(eigs_indice)):
    eig_index = eigs_indice[eig_trial]
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0,1,p_net.N_E), eig_V[0:p_net.N_E, eig_index],'r.',markersize=0.5,label='Exc.')
    ax.plot(np.linspace(0,1,p_net.N_I), eig_V[p_net.N_E:p_net.N_E+p_net.N_I, eig_index],'b.',markersize=0.5,label='Inh.')
    ax.plot(np.linspace(0,1,p_net.N_E), np.linspace(0,0,p_net.N_E), "k--")
    ax.set_xlabel("Location", fontsize=15)
    ax.set_ylabel("Activity", fontsize=15)
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylim((-np.max(eig_V),np.max(eig_V)))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.legend(loc = 'upper right',fontsize=15)
    plt.savefig(r"figs/artfigs_eigsandeigV1D_eigV_"+str(eig_trial)+".png")
    plt.close()








