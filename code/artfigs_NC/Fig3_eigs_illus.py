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

#eigV ploted
outlier_plot_num = 21
bulk_plot_num = 1

#calc eigs
calc_eigs_bool = False
if os.path.exists("./data/artfigs_NC_illus_eigs.npy") and (not calc_eigs_bool):
    eigs = np.load("./data/artfigs_NC_illus_eigs.npy")
    eigV_list = np.load("./data/artfigs_NC_illus_eigV.npy")
else:
    J = generate_net_sparse(p_net, dim=2)
    J = J.toarray()
    eigs, eig_V = np.linalg.eig(J)
    np.save("./data/artfigs_NC_illus_eigs.npy", eigs)

    #eigs ploted indice
    outlier_eig_indice = find_points(eigs, 100+0j, outlier_plot_num)
    bulk_eig_indice = find_points(eigs, 0.1+0j, bulk_plot_num)
    eigs_indice = outlier_eig_indice + bulk_eig_indice

    eigV_list = []
    for index in eigs_indice:
        eigV_list.append(eig_V[:,index])
    np.save("./data/artfigs_NC_illus_eigV.npy", eigV_list)


temp_plot_pred(p_net, dim=2)
radius = calc_pred_radius(p_net,dim=2)
artfigs_plot_eigs(eigs, axvline=False, radius_transparent=radius)
plt.tight_layout()
plt.savefig(r"figs/artfigs_NC_eigs_illus.png")
plt.close()

scale_max = np.abs(np.max((eigV_list[0].real)))
for eig_trial, eig_V_plot in enumerate(eigV_list):
    fig, ax = plt.subplots()
    norm = mcolors.TwoSlopeNorm(vmin=-scale_max, vcenter=0, vmax=scale_max)
    eigV_imag = eig_V_plot[0:p_net.N_E].reshape((int(np.ceil(np.sqrt(p_net.N_E))),int(np.ceil(np.sqrt(p_net.N_E)))))
    img = ax.imshow(eigV_imag.real, cmap=plt.cm.RdBu, norm=norm, origin='upper', aspect=1)

    ax.set_xlabel("Location", fontsize=30)
    ax.set_ylabel("Location", fontsize=30)
    ticks = [0, int(np.ceil(np.sqrt(p_net.N_E)))]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([0, 1])
    ax.set_yticklabels([0, 1])
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    cb = fig.colorbar(img, ax=ax, extend='both')
    cb.locator = MaxNLocator(nbins=3)
    cb.ax.tick_params(labelsize=20)
    cb.set_label("Real Part",fontsize=20,loc='center')
    cb.update_ticks()

    plt.tight_layout()
    plt.savefig(r"figs/artfigs_NC_eigV_illus_"+str(eig_trial)+".svg")
    plt.close()


