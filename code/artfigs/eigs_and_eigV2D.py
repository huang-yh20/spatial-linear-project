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
from dyn_params import *
from artfigs_ulits import *

#generate network params
rescale = 1600
N_E, N_I = 22500, 5625
conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -8/rescale, -8/rescale
sigma_EE, sigma_IE, sigma_EI, sigma_II = 8/rescale, 8/rescale, -16/rescale, -16/rescale
d_EE, d_IE, d_EI, d_II = 0.1,0.1,0.1,0.1

p_net = Network_Params(N_E = N_E, N_I = N_I,
    N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
    d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
    g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
    g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
    )

#calc eigs
calc_eigs_bool = False
if os.path.exists("./data/artfigs_eigsandeigV2D_eigs.npy") and (not calc_eigs_bool):
    eigs = np.load("./data/artfigs_eigsandeigV2D_eigs.npy")
    eig_V = np.load("./data/artfigs_eigsandeigV2D_eigV.npy")
else:
    dist_list = calc_dist(p_net, dim = 2)
    J = generate_net(p_net, dist_list)
    eigs, eig_V = np.linalg.eig(J)
    np.save("./data/artfigs_eigsandeigV2D_eigs.npy", eigs)
    np.save("./data/artfigs_eigsandeigV2D_eigV.npy", eig_V)

#eigV ploted
outlier_plot_num = 21
bulk_plot_num = 1

#outlier eigs ploted indice
outlier_eig_indice = find_points(eigs, 100+0j, outlier_plot_num)

#bulk eigs ploted indice
bulk_eig_indice = find_points(eigs, 0.1+0j, bulk_plot_num)

artfigs_plot_eigs(eigs,axvline=False)
plt.tight_layout()
plt.savefig(r"figs/artfigs_eigsandeigV2D_eigs.png")
plt.close()


temp_plot_pred(p_net, dim=2)
artfigs_plot_eigs(eigs)
plt.tight_layout()
plt.savefig(r"figs/artfigs_eigsandeigV2D_eigs_withpred.png")
plt.close()

eigs_indice = outlier_eig_indice + bulk_eig_indice
for eig_trial in range(len(eigs_indice)):
    eig_index = eigs_indice[eig_trial]
    fig, ax = plt.subplots()
    scale_max = np.max((eig_V.real)[0:p_net.N_E, eig_index])
    norm = mcolors.TwoSlopeNorm(vmin=-scale_max, vcenter=0, vmax=scale_max)
    eigV_imag = eig_V[0:p_net.N_E, eig_index].reshape((int(np.ceil(np.sqrt(p_net.N_E))),int(np.ceil(np.sqrt(p_net.N_E)))))
    img = ax.imshow(eigV_imag.real, cmap=plt.cm.RdBu, norm=norm, origin='upper', aspect=1)
    ax.set_xlabel("Location", fontsize=15)
    ax.set_ylabel("Location", fontsize=15)
    # 设置 x 和 y 轴的刻度
    ticks = [0, int(np.ceil(np.sqrt(p_net.N_E)))]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    # 设置 x 和 y 轴的刻度标签
    ax.set_xticklabels([0, 1])
    ax.set_yticklabels([0, 1])
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    cb = fig.colorbar(img, ax=ax, extend='both')
    cb.locator = MaxNLocator(nbins=5)
    cb.update_ticks()
    plt.tight_layout()
    plt.savefig(r"figs/artfigs_eigsandeigV2D_eigV_"+str(eig_trial)+".png")
    plt.close()


