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

file_name_list = ['thres_stable','thres_bump','thres_osc','thres_wave','thres_chaos']
generate_params_func = generate_params_phase_d_II_g_bar_II_thres_L
trial_params_list = [(10,10), (2,8), (10,2), (20,5),(15,20)] #TODO
repeat_num = 1

p_simul_thres = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['thres_linear','thres_powerlaw'], external_input="DC_noise",tau_m=20.0)
t_show_onset, t_show_step, t_show_num, t_step_onset = 1500, 80, 6, 400
t_dynt_onset, t_dynt_end = 1700, 2000
moran_radius = 5
exc_plot_num, inh_plot_num = 4, 1

activation_func_list = [activation_func_dict[p_simul_thres.activation_func[0]], activation_func_dict[p_simul_thres.activation_func[1]]]

for trial_plot in trange(len(file_name_list)):
    file_name = file_name_list[trial_plot]
    trial_params = trial_params_list[trial_plot]
    p_net = generate_params_func(*trial_params)


    #plot dyn of neurons
    record_x = np.load(r"./data/artfigs_NC_"+'d_II_g_bar_II_thres_L'+'_'+str(trial_params[0])+'_'+str(trial_params[1])+'_'+ str(0)+r'.npy')
    plot_exc_neurons_list = list(np.random.randint(0, p_net.N_E, size=exc_plot_num))
    plot_inh_neurons_list = list(np.random.randint(p_net.N_E, p_net.N_E + p_net.N_I, size=inh_plot_num))

    plt.plot(np.linspace(t_dynt_onset, t_dynt_end, int((t_dynt_end-t_dynt_onset)*p_simul_thres.t_step/p_simul_thres.record_step)), record_x[int(t_dynt_onset*p_simul_thres.t_step/p_simul_thres.record_step):int(t_dynt_end*p_simul_thres.t_step/p_simul_thres.record_step), plot_exc_neurons_list[0]], color='red', label='Exc.')
    for neuron_index in plot_exc_neurons_list[1::]:
        plt.plot(np.linspace(t_dynt_onset, t_dynt_end, int((t_dynt_end-t_dynt_onset)*p_simul_thres.t_step/p_simul_thres.record_step)), record_x[int(t_dynt_onset*p_simul_thres.t_step/p_simul_thres.record_step):int(t_dynt_end*p_simul_thres.t_step/p_simul_thres.record_step), neuron_index], color='red')
    plt.plot(np.linspace(t_dynt_onset, t_dynt_end, int((t_dynt_end-t_dynt_onset)*p_simul_thres.t_step/p_simul_thres.record_step)), record_x[int(t_dynt_onset*p_simul_thres.t_step/p_simul_thres.record_step):int(t_dynt_end*p_simul_thres.t_step/p_simul_thres.record_step), plot_inh_neurons_list[0]], color='blue', label='Inh.')
    for neuron_index in plot_inh_neurons_list[1::]:
        plt.plot(np.linspace(t_dynt_onset, t_dynt_end, int((t_dynt_end-t_dynt_onset)*p_simul_thres.t_step/p_simul_thres.record_step)), record_x[int(t_dynt_onset*p_simul_thres.t_step/p_simul_thres.record_step):int(t_dynt_end*p_simul_thres.t_step/p_simul_thres.record_step), neuron_index], color='blue')  
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel("Time", fontsize=15)
    ax.tick_params(axis='x', labelsize=15)  
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) 
    ax.set_ylabel("Activity", fontsize=15)
    ax.tick_params(axis='y', labelsize=15)  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
    plt.tight_layout()
    plt.savefig(r"./figs/artfigs_NC_d_II_g_bar_II_thres_L_dynt_"+file_name+".png")
    plt.close()
    

    #plot dynimag
    record_x = np.load(r"./data/artfigs_NC_"+'d_II_g_bar_II_thres_L'+'_'+str(trial_params[0])+'_'+str(trial_params[1])+'_'+ str(0)+r'.npy')
    record_x = activation_func_list[0](record_x)
    scale_max = np.max(record_x)
    record_x_img = (record_x[:,0:p_net.N_E]).reshape(np.shape(record_x)[0],int(np.ceil(np.sqrt(p_net.N_E))), int(np.ceil(np.sqrt(p_net.N_E))))
    for trial_show in range(t_show_num):
        step_show = int((t_show_step * trial_show + t_show_onset) * p_simul_thres.t_step/p_simul_thres.record_step)
        fig, ax = plt.subplots()         
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5*scale_max ,vmax=scale_max)
        img = ax.imshow(record_x_img[step_show,:,:], norm=norm, origin='upper', aspect=1)
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
        cb.ax.set_title("Firing Rate")
        cb.update_ticks()
        plt.tight_layout()
        plt.savefig(r"./figs/artfigs_NC_d_II_g_bar_II_thres_L_dynimag_"+file_name+"_"+str(trial_show)+".png")
        plt.close()

    record_x = np.load(r"./data/artfigs_NC_"+'d_II_g_bar_II_thres_L'+'_'+str(trial_params[0])+'_'+str(trial_params[1])+'_'+ str(0)+r'.npy')
    record_x = activation_func_list[0](record_x)   
    product_gif(record_x, p_net, p_simul_thres, file_name = r"./figs/artfigs_NC_d_II_g_bar_II_thres_L_dyngif_"+file_name+"_"+str(trial_show)+".gif", dim=2, filter = False)