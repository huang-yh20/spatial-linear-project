import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
from scipy.ndimage import convolve
from tqdm import trange
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from typing import NamedTuple, Union, Callable, List
import imageio
import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn/")
from dyn_ultis import *

#注意我很多坐标轴都设成0到1，所以发放率最大也要是1
def plot_phase_diagram_axis_default(changed_params:str, changed_params_latex:str, generate_phase_params:callable, trial_num:int = 21):
    showed_ticks = [0, (trial_num-1)//2, trial_num-1]
    ylabels = [(generate_phase_params(ticks,0,trial_num))._asdict()[changed_params[0]] for ticks in showed_ticks]
    xlabels = [(generate_phase_params(0,ticks,trial_num))._asdict()[changed_params[1]] for ticks in showed_ticks]

    plt.ylabel(changed_params_latex[0],fontsize=15)
    plt.xlabel(changed_params_latex[1],fontsize=15)
    plt.yticks(ticks=showed_ticks, labels=ylabels,fontsize=15)
    plt.xticks(ticks=showed_ticks, labels=xlabels,fontsize=15)

def plot_phase_diagram(file_name:str, changed_params:str, changed_params_latex:str, generate_phase_params:callable, p_simul:Simul_Params, trial_num: int = 21, repeat_num:int = 1, plot_phase_diagram_axis: Callable = plot_phase_diagram_axis_default):
    p_net = generate_phase_params(0, 0, trial_num)
    activation_func_dict = {"linear": activation_func_linear, "tanh":activation_func_tanh, "rectified_linear_lowthres":activation_func_rectified_linear_lowthres, "rectified_linear_highthres":activation_func_rectified_linear_highthres}
    if type(p_simul.activation_func) == str:
        activation_func_list = [activation_func_dict[p_simul.activation_func], activation_func_dict[p_simul.activation_func]]
    elif type(p_simul.activation_func) == list:
        activation_func_list = [activation_func_dict[p_simul.activation_func[0]], activation_func_dict[p_simul.activation_func[1]]]

    #this function can only be used in calculating the record_x
    def calc_activated_x(x):
        activated_x_E = activation_func_list[0](x[:,0:p_net.N_E])
        activated_x_I = activation_func_list[1](x[:,p_net.N_E:p_net.N_E+p_net.N_I])
        activated_x = np.concatenate((activated_x_E,activated_x_I), axis=1)
        return activated_x
    
    #calculate the phase boundary
    def plot_phase_boundary(radius_list, max_real_list, wavenum_diagram, freq_diagram, trial_num_theo, trial_num=21, plot_list=[True,True,True,True]):
        eps = 0.001
        x = np.linspace(0, trial_num-1, trial_num_theo)
        y = np.linspace(0, trial_num-1, trial_num_theo)
        X, Y = np.meshgrid(x, y)
        if plot_list[0]:
            plt.contour(X, Y, radius_list, levels=[1], colors='gray', linestyles='--')
        if plot_list[1]:
            plt.contour(X, Y, max_real_list, levels=[1], colors='black', linestyles='--')
        if plot_list[2]:
            plt.contour(X, Y, wavenum_diagram, levels=[eps], colors='blue', linestyles='--')
        if plot_list[3]:
            plt.contour(X, Y, freq_diagram, levels=[eps], colors='red', linestyles='--')

    t_step_onset = int(p_simul.t_step/p_simul.record_step) * 1
    trial_num_theo = 21
    moran_radius = 5

    radius_list = np.zeros((trial_num_theo, trial_num_theo))
    max_real_list = np.zeros((trial_num_theo, trial_num_theo))
    phase_diagram = np.full((trial_num_theo, trial_num_theo), np.nan)
    wavenum_diagram = np.zeros((trial_num_theo, trial_num_theo))
    freq_diagram = np.zeros((trial_num_theo, trial_num_theo))
    for trial1 in trange(trial_num_theo):
        for trial2 in range(trial_num_theo):
            p_net = generate_phase_params(trial1, trial2, trial_num_theo)
            radius = calc_pred_radius(p_net, dim=2)
            radius_list[trial1, trial2] = radius
            lambda_list_pred_select,label_list_pred_select = calc_pred_outliers(p_net, dim=2)
            real_part_pred_select = np.real(lambda_list_pred_select)
            imag_part_pred_select = np.imag(lambda_list_pred_select)
            if len(lambda_list_pred_select) != 0:
                max_real_list[trial1, trial2] = real_part_pred_select[np.argmax(real_part_pred_select)]
            else: 
                max_real_list[trial1, trial2] = 0
            if radius >= 1:
                phase_diagram[trial1, trial2] = 0.5
            elif len(lambda_list_pred_select) != 0:
                max_real_index = np.argmax(real_part_pred_select)
                wavenum = label_list_pred_select[max_real_index]
                wavenum_diagram[trial1, trial2] = np.sqrt(wavenum[1]**2 + wavenum[2]**2)
                freq_diagram[trial1, trial2] = np.abs(imag_part_pred_select[max_real_index])/(2*np.pi)
                if (real_part_pred_select[max_real_index] < 1):
                    phase_diagram[trial1, trial2] = 1
            else:
                phase_diagram[trial1, trial2] = 1

    plt.imshow(wavenum_diagram, origin='lower', cmap='viridis')
    cb = plt.colorbar()
    cb.locator = MaxNLocator(nbins=5)
    cb.ax.tick_params(labelsize=15)
    cb.update_ticks()
    norm = mcolors.Normalize(vmin=0, vmax=1)
    plt.imshow(phase_diagram, cmap='gray', norm=norm, origin='lower')
    plot_phase_diagram_axis(changed_params, changed_params_latex, generate_phase_params, trial_num_theo)

    plt.tight_layout()
    plt.savefig("./figs/phase_"+file_name+"_wavenum.png")
    plt.close()

    plt.imshow(freq_diagram, origin='lower', cmap='viridis')
    cb = plt.colorbar()
    cb.locator = MaxNLocator(nbins=5)
    cb.ax.tick_params(labelsize=15)
    cb.update_ticks()
    norm = mcolors.Normalize(vmin=0, vmax=1)
    plt.imshow(phase_diagram, cmap='gray', norm=norm, origin='lower')
    plot_phase_diagram_axis(changed_params, changed_params_latex, generate_phase_params, trial_num_theo)

    plt.tight_layout()
    plt.savefig("./figs/phase_"+file_name+"_freq_theo.png")
    plt.close()


    phase_diagram = np.full((trial_num_theo, trial_num_theo), np.nan)
    wavenum_diagram = np.zeros((trial_num_theo, trial_num_theo))
    freq_diagram = np.zeros((trial_num_theo, trial_num_theo))
    for trial1 in trange(trial_num_theo):
        for trial2 in range(trial_num_theo):
            p_net = generate_phase_params(trial1, trial2, trial_num_theo)
            radius = calc_pred_radius(p_net, dim=2)
            if radius >= 1:
                phase_diagram[trial1, trial2] = 0.5
            else:
                max_lambda, wavenum = calc_max_theoried_lambda(p_net, dim=2)
                wavenum_diagram[trial1, trial2] = np.sqrt(wavenum[1]**2 + wavenum[2]**2)
                freq_diagram[trial1, trial2] = np.abs(np.imag(max_lambda))/(2*np.pi)
                if (np.real(max_lambda) < 1):
                    phase_diagram[trial1, trial2] = 1

    plt.imshow(wavenum_diagram, origin='lower', cmap='viridis')
    cb = plt.colorbar()
    cb.locator = MaxNLocator(nbins=5)
    cb.ax.tick_params(labelsize=15)
    cb.update_ticks()
    norm = mcolors.Normalize(vmin=0, vmax=1)
    plt.imshow(phase_diagram, cmap='gray', norm=norm, origin='lower')
    plot_phase_diagram_axis(changed_params, changed_params_latex, generate_phase_params, trial_num_theo)

    plt.tight_layout()
    plt.savefig("./figs/phase_"+file_name+"_conn_wavenum.png")
    cb.remove()
    plt.close()

    plt.imshow(freq_diagram, origin='lower', cmap='viridis')
    cb = plt.colorbar()
    cb.locator = MaxNLocator(nbins=5)
    cb.ax.tick_params(labelsize=15)
    cb.update_ticks()
    norm = mcolors.Normalize(vmin=0, vmax=1)
    plt.imshow(phase_diagram, cmap='gray', norm=norm, origin='lower')
    plot_phase_diagram_axis(changed_params, changed_params_latex, generate_phase_params, trial_num_theo)

    plt.tight_layout()
    plt.savefig("./figs/phase_"+file_name+"_conn_freq.png")
    cb.remove()
    plt.close()

    #magnitude of neural activity
    mean_acti_all_repeat = np.zeros((repeat_num, trial_num, trial_num))
    for repeat_trial in range(repeat_num):
        for trial1 in trange(trial_num):
            for trial2 in range(trial_num):
                record_x = np.load(r"./data/phase_dynrec_"+file_name+'_'+str(trial1)+'_'+str(trial2)+'_'+str(repeat_trial)+r'.npy')
                activated_x = calc_activated_x(record_x)
                mean_acti_all_repeat[repeat_trial, trial1, trial2] = np.mean(np.abs(activated_x[t_step_onset::,0:p_net.N_E]))
    mean_acti = np.mean(mean_acti_all_repeat, axis=0)
    plt.imshow(mean_acti, origin='lower', cmap='viridis', vmin=0, vmax=1)
    cb = plt.colorbar()
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['0', '0.5', '1'])
    cb.ax.tick_params(labelsize=15)
    cb.update_ticks()

    plot_phase_diagram_axis(changed_params, changed_params_latex, generate_phase_params, trial_num)
    plot_phase_boundary(radius_list, max_real_list, wavenum_diagram, freq_diagram, trial_num_theo, trial_num, plot_list=[True,True,False,False])
    plt.tight_layout()
    plt.savefig("./figs/phase_"+file_name+"_acti.png")
    plt.close()

    #local sync
    mean_sync_all = np.zeros((repeat_num, trial_num, trial_num))
    weight_matrix = np.ones((2*moran_radius+1, 2*moran_radius+1))
    weight_matrix = weight_matrix[np.newaxis, :, :]
    for trial1 in trange(trial_num):
        for trial2 in range(trial_num):  
            for repeat_trial in range(repeat_num):         
                record_x = np.load(r"./data/phase_dynrec_"+file_name+'_'+str(trial1)+'_'+str(trial2)+'_'+str(repeat_trial)+r'.npy')
                activated_x = calc_activated_x(record_x)[t_step_onset::, 0:p_net.N_E]
                activated_x_E_2d = activated_x.reshape((np.shape(activated_x)[0], int(np.sqrt(p_net.N_E)), int(np.sqrt(p_net.N_E))))
                local_sum = convolve(activated_x_E_2d, weight_matrix, mode='wrap')
                local_abs_sum = convolve(np.abs(activated_x_E_2d), weight_matrix, mode='wrap')
            mean_sync_all[repeat_trial, trial1, trial2] = np.mean(np.abs(local_sum/(local_abs_sum +1e-9)))
    mean_sync = np.mean(mean_sync_all, axis=0)
    plt.imshow(mean_sync, origin='lower', cmap='viridis', vmin=0, vmax=1)
    cb = plt.colorbar()
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['0', '0.5', '1'])
    cb.ax.tick_params(labelsize=15)
    cb.update_ticks()

    plot_phase_diagram_axis(changed_params, changed_params_latex, generate_phase_params, trial_num)
    plot_phase_boundary(radius_list, max_real_list, wavenum_diagram, freq_diagram, trial_num_theo, trial_num, plot_list=[True,True,False,False])
    plt.tight_layout()
    plt.savefig("./figs/phase_"+file_name+"_local_sync.png")
    plt.close()

    #global sync
    mean_sync = np.zeros((trial_num, trial_num))
    for trial1 in trange(trial_num):
        for trial2 in range(trial_num):
            mean_sync_one_trial = [] 
            for repeat_trial in range(repeat_num):              
                record_x = np.load(r"./data/phase_dynrec_"+file_name+'_'+str(trial1)+'_'+str(trial2)+'_'+str(repeat_trial)+r'.npy')
                activated_x = calc_activated_x(record_x)
                mean_sync_one_trial.append(np.mean(np.abs(np.sum(activated_x, axis=1))/np.sum(np.abs(activated_x), axis=1)))
            mean_sync[trial1,trial2] = np.mean(np.abs(np.array(mean_sync_one_trial)))

    plt.imshow(mean_sync, origin='lower', cmap='viridis', vmin=0, vmax=1)
    cb = plt.colorbar()
    cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(['0', '0.5', '1'])
    cb.ax.tick_params(labelsize=15)
    cb.update_ticks()

    plot_phase_diagram_axis(changed_params, changed_params_latex, generate_phase_params, trial_num)
    plot_phase_boundary(radius_list, max_real_list, wavenum_diagram, freq_diagram, trial_num_theo, trial_num, plot_list=[True,True,True,False])
    plt.tight_layout()
    plt.savefig("./figs/phase_"+file_name+"_global_sync.png")
    plt.close()

    #simul freq
    mean_freq = np.zeros((trial_num, trial_num))
    for trial1 in trange(trial_num):
        for trial2 in range(trial_num):
            freq_list = []
            for repeat_trial in range(repeat_num):
                record_x = np.load(r"./data/phase_dynrec_"+file_name+'_'+str(trial1)+'_'+str(trial2)+'_'+str(repeat_trial)+r'.npy')
                activated_x = calc_activated_x(record_x)                   
                sp_activated_x = np.abs(np.fft.fft(activated_x[t_step_onset::,:], axis=0))
                freq_sp = np.fft.fftfreq(np.shape(activated_x[t_step_onset::,:])[0], 1/(p_simul.t_step/p_simul.record_step))
                sp_mean = np.mean(sp_activated_x, axis=1)
                freq_list.append(np.abs(freq_sp[np.argmax(sp_mean)]))

            if freq_list.count(0) >= (0.5 * repeat_num):
                mean_freq[trial1, trial2] = 0
            else:
                mean_freq[trial1, trial2] = np.mean(np.array(freq_list))  * (len(mean_freq)/(len(mean_freq) - freq_list.count(0)))
    
    plt.imshow(mean_freq, origin='lower', cmap='viridis', vmin=0)
    cb = plt.colorbar()
    cb.locator = MaxNLocator(nbins=5)
    cb.ax.tick_params(labelsize=15)
    cb.update_ticks()

    plot_phase_diagram_axis(changed_params, changed_params_latex, generate_phase_params, trial_num)
    plot_phase_boundary(radius_list, max_real_list, wavenum_diagram, freq_diagram, trial_num_theo, trial_num, plot_list=[True,True,False,True])
    plt.tight_layout()
    plt.savefig("./figs/phase_"+file_name+"_freq_exp.png")
    plt.close()

    #simul wavenum
    mean_wavenum = np.zeros((trial_num, trial_num))
    for trial1 in trange(trial_num):
        for trial2 in range(trial_num):
            wavenum_list = []
            for repeat_trial in range(repeat_num):
                record_x = np.load(r"./data/phase_dynrec_"+file_name+'_'+str(trial1)+'_'+str(trial2)+'_'+str(repeat_trial)+r'.npy')
                activated_x = calc_activated_x(record_x)    
                activated_x_2d = activated_x[t_step_onset::,0:p_net.N_E].reshape((np.shape(activated_x)[0] - t_step_onset, int(np.sqrt(p_net.N_E)), int(np.sqrt(p_net.N_E))))
                sp_activated_x = np.abs(np.fft.fft2(activated_x_2d))
                sp_mean = np.mean(sp_activated_x, axis=0)
                sp_mean = sp_mean[0:int(p_net.N_E//2), 0:int(p_net.N_E//2)]
                max_wavenum_tuple = np.where(sp_mean == np.max(sp_mean))
                max_wavenum = np.sqrt(max_wavenum_tuple[0][0]**2 + max_wavenum_tuple[1][0]**2)
                wavenum_list.append(max_wavenum)
            if wavenum_list.count(0) >= (0.5 * repeat_num):
                mean_wavenum[trial1, trial2] = 0
            else:
                mean_wavenum[trial1, trial2] = np.mean(np.array(wavenum_list)) * (len(mean_wavenum)/(len(mean_wavenum) - wavenum_list.count(0)))
    
    plt.imshow(mean_wavenum, origin='lower', cmap='viridis', vmin=0)
    cb = plt.colorbar()
    cb.locator = MaxNLocator(nbins=5)
    cb.ax.tick_params(labelsize=15)
    cb.update_ticks()

    plot_phase_diagram_axis(changed_params, changed_params_latex, generate_phase_params, trial_num)
    plot_phase_boundary(radius_list, max_real_list, wavenum_diagram, freq_diagram, trial_num_theo, trial_num, plot_list=[True,True,True,False])
    plt.tight_layout()
    plt.savefig("./figs/phase_"+file_name+"_wavenum_exp.png")
    plt.close()

    mean_moran = np.zeros((trial_num, trial_num))
    weight_matrix = np.ones((2*moran_radius+1, 2*moran_radius+1))
    weight_matrix = weight_matrix[np.newaxis, :, :]
    for trial1 in trange(trial_num):
        for trial2 in range(trial_num):
            moran_list = []
            for repeat_trial in range(repeat_num):
                record_x = np.load(r"./data/phase_dynrec_"+file_name+'_'+str(trial1)+'_'+str(trial2)+'_'+str(repeat_trial)+r'.npy')
                activated_x_E = (calc_activated_x(record_x))[t_step_onset::,0:p_net.N_E]
                centralized_activated_x_E = activated_x_E - np.mean(activated_x_E, axis=1)[:,np.newaxis]
                centralized_activated_x_E_2d = centralized_activated_x_E.reshape((np.shape(centralized_activated_x_E)[0], int(np.sqrt(p_net.N_E)), int(np.sqrt(p_net.N_E))))
                local_sum = convolve(centralized_activated_x_E_2d, weight_matrix, mode='wrap')
                numerator = np.sum(centralized_activated_x_E_2d * local_sum, axis=(1,2))
                denominator = np.sum(centralized_activated_x_E_2d ** 2, axis=(1,2))
                moran_index_time = (1 / np.sum(weight_matrix)) * (numerator / denominator)
                moran_index = np.mean(moran_index_time)
                moran_list.append(moran_index)
            mean_moran[trial1, trial2] = np.mean(np.array(moran_list))

    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    plt.imshow(mean_moran, origin='lower', norm=norm, cmap=plt.cm.RdBu)
    cb = plt.colorbar()
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels(['-1', '0', '1'])
    cb.ax.tick_params(labelsize=15)
    cb.update_ticks()

    plot_phase_diagram_axis(changed_params, changed_params_latex, generate_phase_params, trial_num)
    plot_phase_boundary(radius_list, max_real_list, wavenum_diagram, freq_diagram, trial_num_theo, trial_num, plot_list=[True,True,True,False])
    plt.tight_layout()
    plt.savefig("./figs/phase_"+file_name+"_moran.png")
    plt.close()
   

    


def plot_phase_diagram1p(file_name:str, changed_params_value:tuple, changed_params_latex:str, generate_phase_params1p:callable, p_simul:Simul_Params, trial_num: int = 21, repeat_num:int = 1):
    p_net = generate_phase_params1p(0, trial_num)
    activation_func_dict = {"linear": activation_func_linear, "tanh":activation_func_tanh, "rectified_linear_lowthres":activation_func_rectified_linear_lowthres, "rectified_linear_highthres":activation_func_rectified_linear_highthres}
    if type(p_simul.activation_func) == str:
        activation_func_list = [activation_func_dict[p_simul.activation_func], activation_func_dict[p_simul.activation_func]]
    elif type(p_simul.activation_func) == list:
        activation_func_list = [activation_func_dict[p_simul.activation_func[0]], activation_func_dict[p_simul.activation_func[1]]]

    #this function can only be used in calculating the record_x
    def calc_activated_x(x):
        activated_x_E = activation_func_list[0](x[:,0:p_net.N_E])
        activated_x_I = activation_func_list[1](x[:,p_net.N_E:p_net.N_E+p_net.N_I])
        activated_x = np.concatenate((activated_x_E,activated_x_I), axis=1)
        return activated_x
    
    t_step_onset = int(p_simul.t_step/p_simul.record_step) * 1
    trial_num_theo = 101
    moran_radius = 5



    radius_list = []
    max_real_list = []
    for trial in range(trial_num_theo):
        p_net = generate_phase_params1p(trial, trial_num_theo)
        radius_list.append(calc_pred_radius(p_net,dim=2)) 
        lambda_list_pred_select,label_list_pred_select = calc_pred_outliers(p_net, dim=2)
        real_part_pred_select = np.real(lambda_list_pred_select)
        if len(lambda_list_pred_select) != 0:
            max_real_list.append(real_part_pred_select[np.argmax(real_part_pred_select)])
        else: 
            max_real_list.append(np.nan)
    #find params that radius=1 
    params_value_chaos_trans_list = [] 
    for trial in range(trial_num_theo-1):
        if (radius_list[trial] - 1) * (radius_list[trial+1] - 1) <= 0:
            params_value_chaos_trans = changed_params_value[0] * ((trial_num_theo-1-trial)/(trial_num_theo-1)) + changed_params_value[1] * ((trial)/(trial_num_theo-1))
            params_value_chaos_trans_list.append(params_value_chaos_trans)
    for params_value_chaos_trans in params_value_chaos_trans_list:
        plt.axvline(params_value_chaos_trans, linestyle='--', color='gray')
    #find params that radius = real(max(lambda))
    params_value_chaos_trans_list = []
    for trial in range(trial_num_theo-1):
        if (((radius_list[trial] - max_real_list[trial]) * (radius_list[trial+1] - max_real_list[trial+1]) <= 0) or (np.isnan(max_real_list[trial]) ^  np.isnan(max_real_list[trial+1]))) and (radius_list[trial] >= 1):
            params_value_chaos_trans = changed_params_value[0] * ((trial_num_theo-1-trial)/(trial_num_theo-1)) + changed_params_value[1] * ((trial)/(trial_num_theo-1))
            params_value_chaos_trans_list.append(params_value_chaos_trans)
    for params_value_chaos_trans in params_value_chaos_trans_list:
        plt.axvline(params_value_chaos_trans, linestyle='--', color='blue')


    #calc local_sync, moran_index, mean_acti, todo:freq
    mean_sync_all, moran_index_all, mean_acti_all = np.zeros((repeat_num, trial_num)), np.zeros((repeat_num, trial_num)), np.zeros((repeat_num, trial_num))
    weight_matrix = np.ones((2*moran_radius+1, 2*moran_radius+1))
    weight_matrix = weight_matrix[np.newaxis, :, :]
    for trial in trange(trial_num):
        for repeat_trial in range(repeat_num):
            record_x = np.load(r"./data/phase_dynrec_"+file_name+'_'+str(trial)+'_'+str(repeat_trial)+r'.npy')
            activated_x = calc_activated_x(record_x)[t_step_onset::,0:p_net.N_E]

            #local sync
            activated_x_E_2d = activated_x.reshape((np.shape(activated_x)[0], int(np.sqrt(p_net.N_E)), int(np.sqrt(p_net.N_E))))
            local_sum = convolve(activated_x_E_2d, weight_matrix, mode='wrap')
            local_abs_sum = convolve(np.abs(activated_x_E_2d), weight_matrix, mode='wrap')
            mean_sync_all[repeat_trial, trial] = np.mean(np.abs(local_sum/(local_abs_sum +1e-9)))

            #mean acti
            mean_acti_all[repeat_trial, trial] = np.mean(np.abs(activated_x))

            #moran_index
            centralized_activated_x_E = activated_x - np.mean(activated_x, axis=1)[:,np.newaxis]
            centralized_activated_x_E_2d = centralized_activated_x_E.reshape((np.shape(centralized_activated_x_E)[0], int(np.sqrt(p_net.N_E)), int(np.sqrt(p_net.N_E))))
            local_sum = convolve(centralized_activated_x_E_2d, weight_matrix, mode='wrap')
            numerator = np.sum(centralized_activated_x_E_2d * local_sum, axis=(1,2))
            denominator = np.sum(centralized_activated_x_E_2d ** 2, axis=(1,2))
            moran_index_time = (1 / np.sum(weight_matrix)) * (numerator / denominator)
            moran_index_all[repeat_trial, trial] = np.mean(moran_index_time)
    
    mean_sync, std_sync = np.mean(mean_sync_all, axis=0), np.std(mean_sync_all, axis=0)
    mean_acti, std_acti = np.mean(mean_acti_all, axis=0), np.std(mean_acti_all, axis=0)
    mean_moran, std_moran = np.mean(moran_index_all, axis=0), np.std(moran_index_all, axis=0)

    changed_params_value_list = np.linspace(changed_params_value[0], changed_params_value[1], trial_num)
    plt.errorbar(changed_params_value_list, mean_sync, std_sync, label = 'Local Sync.')
    plt.errorbar(changed_params_value_list, mean_moran, std_moran, label = 'Moran Index')
    plt.errorbar(changed_params_value_list, mean_acti, std_acti, label = 'Magnitude of Neural Activity')

    plt.xlabel(changed_params_latex,fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.xlabel(changed_params_latex, fontsize=15)
    plt.legend()
    
    plt.savefig("./figs/phase_1p_"+file_name+".png")

        




    