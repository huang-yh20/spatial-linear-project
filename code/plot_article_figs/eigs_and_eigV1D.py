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
from dyn_params import *

p_net = generate_params_dyn_global(3)
dist_list = calc_dist(p_net, dim = 1)
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
plt.savefig(r"figs/artfigs_eigsandeigV1D_eigs.png")
plt.close()

plt.scatter(real_part, imag_part, s=3, c='none', marker='o', edgecolors='k')
temp_plot_pred(p_net, dim=1)
ax = plt.gca()
ax.set_xlabel("$Re(\\lambda)$", fontsize=15)
ax.tick_params(axis='x', labelsize=15)  # 控制x轴刻度的字体大小
ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) 
ax.set_ylabel("$Im(\\lambda)$", fontsize=15)
ax.tick_params(axis='y', labelsize=15)  # 控制y轴刻度的字体大小
ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
ax.set_aspect('equal') 
plt.savefig(r"figs/artfigs_eigsandeigV1D_eigs_withpred.png")
plt.close()

eigs_indice = outlier_eig_indice + bulk_eig_indice
for eig_trial in range(len(eigs_indice)):
    eig_index = eigs_indice[eig_trial]
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0,1,p_net.N_E), eig_V[0:p_net.N_E, eig_index],'r-',markersize=0.5,label='Exc.')
    ax.plot(np.linspace(0,1,p_net.N_I), eig_V[p_net.N_E:p_net.N_E+p_net.N_I, eig_index],'b-',markersize=0.5,label='Inh.')
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








