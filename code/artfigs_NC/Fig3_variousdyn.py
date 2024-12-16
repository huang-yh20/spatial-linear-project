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
from spatial_ultis import *
from dyn_ultis import *
from phase_params import *
from dyn_params import *
from artfigs_NC_ultis import *
from artfigs_NC_params import *

file_name_list = ['tanh_global','tanh_osc','tanh_bump','tanh_wave','tanh_chaos','tanh_stable']
p_net_eigs_list = [generate_params_phase_d_II_g_bar_II_L(10,17),generate_params_phase_d_II_g_bar_II_L(5,5),generate_params_phase_d_II_g_bar_II_L(3,12),
              generate_params_phase_d_II_g_bar_II_L(17,7), generate_params_phase_g_d_II_L_chaos(15,9),generate_params_phase_d_II_g_bar_II_L(10,10)]
p_net_order_list = [generate_params_phase_d_II_g_bar_II_S(10,17),generate_params_phase_d_II_g_bar_II_S(5,5),generate_params_phase_d_II_g_bar_II_S(3,12),
              generate_params_phase_d_II_g_bar_II_S(17,7), generate_params_phase_g_d_II_L_chaos(15,9),generate_params_phase_d_II_g_bar_II_S(10,10)]
order_file_name_list = ["d_II_g_bar_II_S_10_17","d_II_g_bar_II_S_5_5","d_II_g_bar_II_S_3_12","d_II_g_bar_II_S_17_7",'g_d_II_L_chaos_15_9',"d_II_g_bar_II_S_10_10"]
show_file_name_list = ["d_II_g_bar_II_L_10_17","d_II_g_bar_II_L_5_5","d_II_g_bar_II_L_3_12","d_II_g_bar_II_L_17_7",'g_d_II_L_chaos_15_9',"d_II_g_bar_II_L_10_10"]
repeat_num = 5

p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['tanh','tanh_high'], external_input="noise",tau_m=20.0)
t_show_onset, t_show_step, t_show_num, t_step_onset = 1400, 80, 6, 500
t_dynt_onset, t_dynt_end = 1700, 2000
moran_radius = 5

calc_orderprams_bool = True
exc_plot_num, inh_plot_num = 4, 1

for trial_plot in trange(len(file_name_list)):
    file_name = file_name_list[trial_plot]
    p_net = p_net_order_list[trial_plot]

    #calc orderprams
    orderparams_name = ['Std.', 'Local Sync.', "Moran's Index", 'Osc. Index']

    mean_acti_all, mean_localsync_all, moran_all, freq_index_all = [], [], [], []
    weight_matrix = np.ones((2*moran_radius+1, 2*moran_radius+1))
    weight_matrix = weight_matrix[np.newaxis, :, :]
    for repeat_trial in range(repeat_num):
        record_x = np.load(r"./data/artfigs_NC_"+order_file_name_list[trial_plot]+'_'+str(repeat_trial)+r'.npy')
        activated_x = np.tanh(record_x) #TEMP
        #mean acti
        mean_acti_all.append(np.mean(np.std(activated_x[t_step_onset::,0:p_net.N_E])))
        
        #local sync.
        activated_x_cut = activated_x[t_step_onset::,0:p_net.N_E]
        activated_x_E_2d = activated_x_cut[:,0:p_net.N_E].reshape((np.shape(activated_x_cut)[0], int(np.sqrt(p_net.N_E)), int(np.sqrt(p_net.N_E))))
        local_sum = convolve(activated_x_E_2d, weight_matrix, mode='wrap')
        local_abs_sum = convolve(np.abs(activated_x_E_2d), weight_matrix, mode='wrap')
        mean_localsync_all.append(np.abs(local_sum/(local_abs_sum +1e-9)))

        #moran index
        activated_x_E = activated_x[t_step_onset::, 0:p_net.N_E]
        centralized_activated_x_E = activated_x_E - np.mean(activated_x_E, axis=1)[:,np.newaxis]
        centralized_activated_x_E_2d = centralized_activated_x_E.reshape((np.shape(centralized_activated_x_E)[0], int(np.sqrt(p_net.N_E)), int(np.sqrt(p_net.N_E))))
        local_sum = convolve(centralized_activated_x_E_2d, weight_matrix, mode='wrap')
        numerator = np.sum(centralized_activated_x_E_2d * local_sum, axis=(1,2))
        denominator = np.sum(centralized_activated_x_E_2d ** 2, axis=(1,2))
        moran_index_time = (1 / np.sum(weight_matrix)) * (numerator / (denominator + 1e-5))
        moran_index = np.mean(moran_index_time)
        moran_all.append(moran_index)

        #freq index
        freq_max_eps = 5
        sp_activated_x = np.abs(np.fft.fft(activated_x[t_step_onset::,:], axis=0))
        freq_sp = np.fft.fftfreq(np.shape(activated_x[t_step_onset::,:])[0], 1/(p_simul.t_step/p_simul.record_step))
        sp_mean = np.mean(sp_activated_x, axis=1)
        freq_len = np.shape(sp_mean)[0]//2
        freq_max = np.argmax(sp_mean[0:freq_len])
        if freq_max > freq_max_eps:
            freq_max_left = np.argmin(sp_mean[0:freq_max])
            freq_max_right = freq_max + (freq_max - freq_max_left)
            freq_index = np.sum(sp_mean[freq_max_left:freq_max_right]**2)/np.sum(sp_mean[0:freq_len]**2)
        else: 
            freq_index = 0
        freq_index_all.append(freq_index)

    mean_acti, std_acti = np.mean(mean_acti_all), np.std(mean_acti_all)
    mean_localsync, std_localsync = np.mean(mean_localsync_all), np.std(mean_localsync_all)
    mean_moran, std_moran = np.mean(moran_all), np.std(moran_all)
    mean_freq_inex, std_freq_index = np.mean(freq_index_all), np.std(freq_index_all)

    orderprams_mean_array = np.array([mean_acti,mean_localsync,mean_moran,mean_freq_inex])
    orderprams_std_array = np.array([std_acti,std_localsync,std_moran,std_freq_index])

    plt.bar(orderparams_name, orderprams_mean_array, yerr=orderprams_std_array, capsize=15, width=0.5, facecolor='none', edgecolor='black', error_kw={'ecolor': 'black'})
    ax = plt.gca()
    ax.set_ylabel("Value", fontsize=15)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim((0, 1))
    ax.tick_params(axis='y', labelsize=15)

    plt.savefig("./figs/artfigs_NC_variousdyn_orderparams_"+file_name+".png")
    plt.close()

    #calc eigs
    calc_eigs_bool = False

    p_net = p_net_eigs_list[trial_plot]
    if os.path.exists(r"./data/artfigs_NC_variousdyn_eigs_"+file_name+"eigs.npy") and (not calc_eigs_bool):
        eigs = np.load(r"./data/artfigs_NC_variousdyn_eigs_"+file_name+"eigs.npy")
        eig_V = np.load(r"./data/artfigs_NC_variousdyn_eigs_"+file_name+"eigV.npy")
    else:
        J = generate_net_sparse(p_net, dim=2)
        J = J.toarray()
        eigs, eig_V = np.linalg.eig(J)
        np.save(r"./data/artfigs_NC_variousdyn_eigs_"+file_name+"eigs.npy", eigs)
        np.save(r"./data/artfigs_NC_variousdyn_eigs_"+file_name+"eigV.npy", eig_V)
    
    real_part = np.real(eigs)
    imag_part = np.imag(eigs)

    #plot eigs
    fig, ax = plt.subplots()
    artfigs_plot_eigs(eigs)

    #plot eigV of largest eigs
    largest_eigs_index = np.argmax(real_part)
    plt.scatter([real_part[largest_eigs_index]],[imag_part[largest_eigs_index]],s=30,c='r',marker='^')
        
    ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper left')
    scale_max = np.abs(np.max((eig_V.real)[0:p_net.N_E, largest_eigs_index]))
    norm = mcolors.TwoSlopeNorm(vmin=-scale_max, vcenter=0, vmax=scale_max)
    eigV_imag = eig_V[0:p_net.N_E, largest_eigs_index].reshape((int(np.ceil(np.sqrt(p_net.N_E))),int(np.ceil(np.sqrt(p_net.N_E)))))
    img = ax_inset.imshow(eigV_imag.real, cmap=plt.cm.RdBu, norm=norm, origin='upper', aspect=1)
    # ax_inset.set_xlabel("Location", fontsize=15)
    # ax_inset.set_ylabel("Location", fontsize=15)

    ticks = [0, int(np.ceil(np.sqrt(p_net.N_E)))]
    ax_inset.set_xticks(ticks)
    ax_inset.set_yticks(ticks)
    
    ax_inset.set_xticklabels([0, 1])
    ax_inset.set_yticklabels([0, 1])
    # ax_inset.tick_params(ax_insetis='x', labelsize=15)
    # ax_inset.tick_params(ax_insetis='y', labelsize=15)
    plt.tight_layout()
    plt.savefig(r"./figs/artfigs_NC_variousdyn_eigs_"+file_name+".png")
    plt.close()

    #plot dyn of neurons
    p_net = p_net_eigs_list[trial_plot]
    record_x = np.load(r"./data/artfigs_NC_"+show_file_name_list[trial_plot]+'_'+str(0)+r'.npy')
    plot_exc_neurons_list = list(np.random.randint(0, p_net.N_E, size=exc_plot_num))
    plot_inh_neurons_list = list(np.random.randint(p_net.N_E, p_net.N_E + p_net.N_I, size=inh_plot_num))

    plt.plot(np.linspace(t_dynt_onset, t_dynt_end, int((t_dynt_end-t_dynt_onset)*p_simul.t_step/p_simul.record_step)), record_x[int(t_dynt_onset*p_simul.t_step/p_simul.record_step):int(t_dynt_end*p_simul.t_step/p_simul.record_step), plot_exc_neurons_list[0]], color='red', label='Exc.')
    for neuron_index in plot_exc_neurons_list[1::]:
        plt.plot(np.linspace(t_dynt_onset, t_dynt_end, int((t_dynt_end-t_dynt_onset)*p_simul.t_step/p_simul.record_step)), record_x[int(t_dynt_onset*p_simul.t_step/p_simul.record_step):int(t_dynt_end*p_simul.t_step/p_simul.record_step), neuron_index], color='red')
    plt.plot(np.linspace(t_dynt_onset, t_dynt_end, int((t_dynt_end-t_dynt_onset)*p_simul.t_step/p_simul.record_step)), record_x[int(t_dynt_onset*p_simul.t_step/p_simul.record_step):int(t_dynt_end*p_simul.t_step/p_simul.record_step), plot_inh_neurons_list[0]], color='blue', label='Inh.')
    for neuron_index in plot_inh_neurons_list[1::]:
        plt.plot(np.linspace(t_dynt_onset, t_dynt_end, int((t_dynt_end-t_dynt_onset)*p_simul.t_step/p_simul.record_step)), record_x[int(t_dynt_onset*p_simul.t_step/p_simul.record_step):int(t_dynt_end*p_simul.t_step/p_simul.record_step), neuron_index], color='blue')  
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel("Time", fontsize=15)
    ax.tick_params(axis='x', labelsize=15)  
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) 
    ax.set_ylabel("Activity", fontsize=15)
    ax.tick_params(axis='y', labelsize=15)  
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
    plt.tight_layout()
    plt.savefig(r"./figs/artfigs_NC_variousdyn_dynt_"+file_name+".png")
    plt.close()
    

    #plot dynimag
    p_net = p_net_eigs_list[trial_plot]
    record_x = np.load(r"./data/artfigs_NC_"+show_file_name_list[trial_plot]+'_'+str(0)+r'.npy')
    record_x = activation_func(record_x)
    #scale_max = np.max(record_x)
    scale_max = 1
    record_x_img = (record_x[:,0:p_net.N_E]).reshape(np.shape(record_x)[0],int(np.ceil(np.sqrt(p_net.N_E))), int(np.ceil(np.sqrt(p_net.N_E))))
    for trial_show in range(t_show_num):
        step_show = int((t_show_step * trial_show + t_show_onset) * p_simul.t_step/p_simul.record_step)
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
        cb.ax.set_title("Firing Rate")
        cb.update_ticks()
        plt.tight_layout()
        plt.savefig(r"./figs/artfigs_NC_variousdyn_dynimag_"+file_name+"_"+str(trial_show)+".png")
        plt.close()
            
        
        

        





    

