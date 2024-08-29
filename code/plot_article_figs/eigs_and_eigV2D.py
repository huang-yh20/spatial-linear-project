import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import sys
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/phase/")
from spatial_ultis import *
from dyn_ultis import *
from phase_params import *

#generate network params
rescale = 1600
N_E, N_I = 22500, 5625
conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -8/rescale, -8/rescale
sigma_EE, sigma_IE, sigma_EI, sigma_II = 8/rescale, 8/rescale, -16/rescale, -16/rescale
d_EE, d_IE, d_EI, d_II = 0.05, 0.05, 0.05, 0.05, 0.05

p_net = Network_Params(N_E = N_E, N_I = N_I,
    N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
    d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
    g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
    g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
    )

#calc eigs
dist_list = calc_dist(p_net, dim = 2)
J = generate_net(p_net, dist_list)
eigs, eig_V = np.linalg.eig(J)

#eigV ploted
outlier_plot_num = 21
bulk_plot_num = 1

#outlier eigs ploted indice
outlier_eig_indice = find_points(eigs, 100+0j, outlier_plot_num)

#bulk eigs ploted indice
bulk_eig_indice = find_points(eigs, 0.1+0j, bulk_plot_num)

real_part = np.real(eigs)
imag_part = np.imag(eigs)
plt.scatter(real_part, imag_part, s=3, c='none', marker='o', edgecolors='k')
ax = plt.gca()
ax.set_xlabel("$Re(\\lambda)$", fontsize=15)
ax.tick_params(axis='x', labelsize=15)  # 控制x轴刻度的字体大小
ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) 
ax.set_ylabel("$Im(\\lambda)$", fontsize=15)
ax.tick_params(axis='y', labelsize=15)  # 控制y轴刻度的字体大小
ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
ax.set_aspect('equal') 
plt.savefig(r"figs/artfigs_eigsandeigV2D_eigs.png")
plt.close()

plt.scatter(real_part, imag_part, s=3, c='none', marker='o', edgecolors='k')
temp_plot_pred(p_net, dim=2)
ax = plt.gca()
ax.set_xlabel("$Re(\\lambda)$", fontsize=15)
ax.tick_params(axis='x', labelsize=15)  # 控制x轴刻度的字体大小
ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) 
ax.set_ylabel("$Im(\\lambda)$", fontsize=15)
ax.tick_params(axis='y', labelsize=15)  # 控制y轴刻度的字体大小
ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
ax.set_aspect('equal') 
plt.savefig(r"figs/artfigs_eigsandeigV2D_eigs_withpred.png")
plt.close()

eigs_indice = outlier_eig_indice + bulk_eig_indice
for eig_trial in range(len(eigs_indice)):
    eig_index = eigs_indice[eig_trial]
    fig, ax = plt.subplots()
    scale_max = np.max(eig_V[0:p_net.N_E,:])
    norm = mcolors.TwoSlopeNorm(vmin=-scale_max, vcenter=0, vmax=scale_max)
    eigV_imag = eig_V[0:p_net.N_E, eig_index].reshape((int(np.ceil(np.sqrt(p_net.N_E))),int(np.ceil(np.sqrt(p_net.N_E)))))
    img = ax.imshow(eigV_imag, cmap=plt.cm.RdBu, norm=norm, origin='upper', aspect=1)
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
    plt.savefig(r"figs/artfigs_eigsandeigV2D_eigV_"+str(eig_trial)+".png")
    plt.close()


