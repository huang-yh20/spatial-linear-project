import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
from tqdm import trange
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

trial_num = 41
t_show = 20
p_simul = Simul_Params(T = 40, t_step=100, record_step=10, activation_func=['tanh','linear'])
file_name_list = ["alpha"]
generate_params_func_list = [generate_params1p_phase_alpha]
trial_plot_lists = [[10,20,30]]

for file_trial in range(file_name_list):
    file_name = file_name_list[file_trial]
    generate_params_func = generate_params_func_list[file_trial]
    trial_plot_list = trial_plot_lists[file_trial]
    for plot_trial in range(len(trial_plot_list)):
        p_net = generate_params_func(plot_trial,trial_num)
        dist_list = calc_dist(p_net, dim = 2)
        J = generate_net(p_net, dist_list)
        eigs, eig_V = np.linalg.eig(J)

        #plot eigs
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
        plt.savefig(r"figs/artfigs_phasediagacti1p_"+file_name+"eigs_"+str(plot_trial)+".png")
        plt.close()

        #plot dyn_imag
        record_x = np.load(r'./data/'+'phase_dynrec_'+file_name+'_'+str(plot_trial)+'_0.npy')
        scale_max = np.max(record_x)
        record_x_img = (record_x[:,0:p_net.N_E]).reshape(np.shape(record_x)[0],int(np.ceil(np.sqrt(p_net.N_E))), int(np.ceil(np.sqrt(p_net.N_E))))
        
        step_show = int(t_show * p_simul.t_step/p_simul.record_step)
        fig, ax = plt.subplots()         
        norm = mcolors.TwoSlopeNorm(vmin=-scale_max, vcenter=0, vmax=scale_max)
        img = ax.imshow(record_x_img[step_show,:,:], cmap=plt.cm.RdBu, norm=norm, origin='upper', aspect=1)
        ax.set_xlabel("Location", fontsize=15)
        ax.set_ylabel("Location", fontsize=15)

        ticks = [0, int(np.ceil(np.sqrt(p_net.N_E)))]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.set_xticklabels([0, 1])
        ax.set_yticklabels([0, 1])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        cb = fig.colorbar(img, ax=ax, extend='both')
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()

        plt.savefig(r"./figs/artfigs_phasediagacti1p_"+file_name+"dynimag_"+str(plot_trial)+".png")

        #plot dynimag with eigs
        scale_max = np.max(record_x)
        record_x_img = (record_x[:,0:p_net.N_E]).reshape(np.shape(record_x)[0],int(np.ceil(np.sqrt(p_net.N_E))), int(np.ceil(np.sqrt(p_net.N_E))))
        
        step_show = int(t_show * p_simul.t_step/p_simul.record_step)
        fig, ax = plt.subplots()         
        norm = mcolors.TwoSlopeNorm(vmin=-scale_max, vcenter=0, vmax=scale_max)
        img = ax.imshow(record_x_img[step_show,:,:], cmap=plt.cm.RdBu, norm=norm, origin='upper', aspect=1)
        ax.set_xlabel("Location", fontsize=15)
        ax.set_ylabel("Location", fontsize=15)

        ticks = [0, int(np.ceil(np.sqrt(p_net.N_E)))]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.set_xticklabels([0, 1])
        ax.set_yticklabels([0, 1])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        cb = fig.colorbar(img, ax=ax, extend='both')
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()

        ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper left')        
        real_part = np.real(eigs)
        imag_part = np.imag(eigs)
        plt.scatter(real_part, imag_part, s=3, c='none', marker='o', edgecolors='k')
        ax = plt.gca()
        #ax_inset.set_xlabel("$Re(\\lambda)$", fontsize=15)
        #ax_inset.tick_params(axis='x', labelsize=15)  # 控制x轴刻度的字体大小
        ax_inset.xaxis.set_major_locator(MaxNLocator(nbins=3)) 
        #ax_inset.set_ylabel("$Im(\\lambda)$", fontsize=15)
        #ax_inset.tick_params(axis='y', labelsize=15)  # 控制y轴刻度的字体大小
        ax_inset.yaxis.set_major_locator(MaxNLocator(nbins=3)) 
        ax_inset.set_aspect('equal') 
        plt.savefig(r"figs/artfigs_phasediagacti1p_"+file_name+"dynimagwitheigs_"+str(plot_trial)+".png")
        plt.close()        