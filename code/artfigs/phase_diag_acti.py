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
sys.path.append("./code/artfigs/")
from spatial_ultis import *
from dyn_ultis import *
from phase_params import *
from dyn_params import *
from artfigs_ulits import *

trial_num = 21
t_show = 20
p_simul = Simul_Params(T = 40, t_step=100, record_step=10, activation_func=['tanh','linear'])
file_name_list = ["g_bar_E_I_0", "g_bar_IE_d_II"]
generate_params_func_list = [generate_params_phase_g_bar_E_I_0, generate_params_phase_g_bar_IE_d_II]
trial_plot_lists = [[(4,5), (4,18), (4,11), (13,8)], [(10,5), (10,15), (18,15), (3,11), (3,14)]]

for file_trial in range(len(file_name_list)):
    file_name = file_name_list[file_trial]
    generate_params_func = generate_params_func_list[file_trial]
    trial_plot_list = trial_plot_lists[file_trial]
    for plot_trial in range(len(trial_plot_list)):
        p_net = generate_params_func(plot_trial[0],plot_trial[1],trial_num)

        calc_eigs_bool = False
        if os.path.exists(r"./data/artfigs_phasediagacti_"+file_name+"eigs_"+str(plot_trial[0])+"_"+str(plot_trial[1])+"eigs.npy") and (not calc_eigs_bool):
            eigs = np.load(r"./data/artfigs_phasediagacti_"+file_name+"eigs_"+str(plot_trial[0])+"_"+str(plot_trial[1])+"eigs.npy")
            eig_V = np.load(r"./data/artfigs_phasediagacti_"+file_name+"eigs_"+str(plot_trial[0])+"_"+str(plot_trial[1])+"eigV.npy")
        else:
            dist_list = calc_dist(p_net, dim = 2)
            J = generate_net(p_net, dist_list)
            eigs, eig_V = np.linalg.eig(J)
            np.save(r"./data/artfigs_phasediagacti_"+file_name+"eigs_"+str(plot_trial[0])+"_"+str(plot_trial[1])+"eigs.npy", eigs)
            np.save(r"./data/artfigs_phasediagacti_"+file_name+"eigs_"+str(plot_trial[0])+"_"+str(plot_trial[1])+"eigV.npy", eig_V)

        #plot eigs
        artfigs_plot_eigs(eigs)
        plt.savefig(r"figs/artfigs_phasediagacti_"+file_name+"eigs_"+str(plot_trial[0])+"_"+str(plot_trial[1])+".png")
        plt.close()

        #plot dyn_imag
        record_x = np.load(r'./data/'+'phase_dynrec_'+file_name+'_'+str(plot_trial[0])+'_'+str(plot_trial[1])+'_0.npy')
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

        plt.savefig(r"./figs/artfigs_phasediagacti_"+file_name+"dynimag_"+str(plot_trial[0])+"_"+str(plot_trial[1])+".png")

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
        artfigs_plot_eigs(eigs, ax=ax_inset)
        plt.savefig(r"figs/artfigs_phasediagacti_"+file_name+"dynimagwitheigs_"+str(plot_trial[0])+"_"+str(plot_trial[1])+".png")
        plt.close()        





