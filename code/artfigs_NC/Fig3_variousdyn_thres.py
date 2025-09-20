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

file_name_0 = 'homo_d_II_g_bar_II_thres_L'
file_name_list = ['thres_stable_indiscale','thres_bump_indiscale','thres_osc_indiscale','thres_wave_indiscale','thres_chaos_indiscale']
generate_params_func = generate_params_phase_d_II_g_bar_II_thres_L
trial_params_list = [(10,10), (1,8), (10,2), (18,3), (10,19)] #TODO
repeat_num = 1

p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['thres_linear','thres_powerlaw'], external_input="DC_noise",tau_m=20.0)
t_show_onset, t_show_step, t_show_num, t_step_onset = 1680, 50, 6, 400
t_dynt_onset, t_dynt_end = 1680, 1980 
moran_radius = 5
exc_plot_num, inh_plot_num = 4, 1

activation_func_list = [activation_func_dict[p_simul.activation_func[0]], activation_func_dict[p_simul.activation_func[1]]]

for trial_plot in trange(len(file_name_list)):
    #TEMP: for better illus of chaos
    if trial_plot == 4:
        t_dynt_onset, t_dynt_end = 1000, 2000
    else:
        t_dynt_onset, t_dynt_end = 1680, 1980 

    file_name = file_name_list[trial_plot]
    trial_params = trial_params_list[trial_plot]
    p_net = generate_params_func(*trial_params)


    #plot dyn of neurons
    record_x = np.load(r"./data/artfigs_NC_"+file_name_0+'_'+str(trial_params[0])+'_'+str(trial_params[1])+'_'+ str(0)+r'.npy')
    activated_x_E = activation_func_list[0](record_x[:, 0:p_net.N_E])
    activated_x_I = activation_func_list[0](record_x[:, p_net.N_E:p_net.N_E+p_net.N_I])
    record_x = np.concatenate((activated_x_E,activated_x_I),axis=1)
    plot_exc_neurons_list = list(np.random.randint(0, p_net.N_E, size=exc_plot_num))
    plot_inh_neurons_list = list(np.random.randint(p_net.N_E, p_net.N_E + p_net.N_I, size=inh_plot_num))

    plt.plot(np.linspace(0, t_dynt_end-t_dynt_onset, int((t_dynt_end-t_dynt_onset)*p_simul.t_step/p_simul.record_step)), record_x[int(t_dynt_onset*p_simul.t_step/p_simul.record_step):int(t_dynt_end*p_simul.t_step/p_simul.record_step), plot_exc_neurons_list[0]], color='red', label='Exc.')
    for neuron_index in plot_exc_neurons_list[1::]:
        plt.plot(np.linspace(0, t_dynt_end-t_dynt_onset, int((t_dynt_end-t_dynt_onset)*p_simul.t_step/p_simul.record_step)), record_x[int(t_dynt_onset*p_simul.t_step/p_simul.record_step):int(t_dynt_end*p_simul.t_step/p_simul.record_step), neuron_index], color='red')
    plt.plot(np.linspace(0, t_dynt_end-t_dynt_onset, int((t_dynt_end-t_dynt_onset)*p_simul.t_step/p_simul.record_step)), record_x[int(t_dynt_onset*p_simul.t_step/p_simul.record_step):int(t_dynt_end*p_simul.t_step/p_simul.record_step), plot_inh_neurons_list[0]], color='blue', label='Inh.')
    for neuron_index in plot_inh_neurons_list[1::]:
        plt.plot(np.linspace(0, t_dynt_end-t_dynt_onset, int((t_dynt_end-t_dynt_onset)*p_simul.t_step/p_simul.record_step)), record_x[int(t_dynt_onset*p_simul.t_step/p_simul.record_step):int(t_dynt_end*p_simul.t_step/p_simul.record_step), neuron_index], color='blue')

    lg = plt.legend(loc = 'upper right',fontsize=25)
    ax = plt.gca()
    ax.set_xlabel("Time(ms)", fontsize=25)
    ax.tick_params(axis='x', labelsize=25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_ylabel("Firing Rate(Hz)", fontsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


    if trial_plot != 0:
        plt.xlabel("")
        plt.ylabel("")
        lg.remove()
    plt.tight_layout()
    plt.savefig(r"./figs/artfigs_NC_variousdyn_"+"dynt_"+file_name+".svg",bbox_inches='tight')
    plt.close()


    #plot dynimag
    record_x = np.load(r"./data/artfigs_NC_"+file_name_0+'_'+str(trial_params[0])+'_'+str(trial_params[1])+'_'+ str(0)+r'.npy')
    record_x = activation_func_list[0](record_x)
    # scale_min, scale_max = 0, np.minimum(np.max(record_x),300)
    # 下面是为了async弄的
    scale_min, scale_max = np.min(record_x[500:,0:p_net.N_E]), np.minimum(np.max(record_x[500:,0:p_net.N_E]),300)
    record_x_img = (record_x[:,0:p_net.N_E]).reshape(np.shape(record_x)[0],int(np.ceil(np.sqrt(p_net.N_E))), int(np.ceil(np.sqrt(p_net.N_E))))
    for trial_show in range(t_show_num):
        step_show = int((t_show_step * trial_show + t_show_onset) * p_simul.t_step/p_simul.record_step)
        fig, ax = plt.subplots()
        norm = mcolors.TwoSlopeNorm(vmin=scale_min, vcenter=0.5*(scale_min + scale_max) ,vmax=scale_max)
        img = ax.imshow(record_x_img[step_show,:,:], norm=norm, origin='upper', aspect=1)
        ax.set_xlabel("Neuron location (X)", fontsize=25)
        ax.set_ylabel("Neuron location (Y)", fontsize=25)

        ticks = [0, int(np.ceil(np.sqrt(p_net.N_E)))]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.set_xticklabels([0, 1])
        ax.set_yticklabels([0, 1])
        ax.tick_params(axis='x', labelsize=30)
        ax.tick_params(axis='y', labelsize=30)
        cb = fig.colorbar(img, ax=ax, extend='both')
        cb.locator = MaxNLocator(nbins=3)
        cb.ax.set_title("Firing Rate(Hz)", fontsize=20)
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_size(30)
            cb.update_ticks()
        if trial_plot != 0 or trial_show != 0:
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("")
            plt.ylabel("")
        if trial_show != 2:
            cb.remove()
        plt.tight_layout()
        plt.savefig(r"./figs/artfigs_NC_variousdyn_"+"dynimag_"+file_name+"_"+str(trial_show)+".svg", bbox_inches='tight')
        plt.close()

    # record_x = np.load(r"./data/artfigs_NC_"+file_name_0+'_'+str(trial_params[0])+'_'+str(trial_params[1])+'_'+ str(0)+r'.npy')
    # product_gif(record_x, p_net, p_simul, file_name = r"artfigs_NC_"+file_name_0+"_dyngif_"+file_name+"_"+str(trial_show)+".gif", dim=2, filter = False)
