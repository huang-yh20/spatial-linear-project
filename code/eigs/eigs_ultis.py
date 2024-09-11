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
import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn/")
from dyn_ultis import *

def plot_eigs2D_diagram(file_name:str, changed_params:str, changed_params_latex:str, generate_eigs_params:callable, trial_num: int = 6, repeat_num:int = 10):
    trial_num_theo = 100
    label_plot_list = [(1,0,0),(1,1,0),(1,0,1),(1,1,1),(1,2,0)]
    c_list = ['r','g','b','y','c','m']

    #TEMP
    if file_name == 'alpha':
        changed_param_theo_list = np.linspace(0.2, 0.7, trial_num_theo)
        changed_param_exp_list = np.linspace(0.2, 0.7, trial_num)
    else:
        changed_param_exp_list = [(generate_eigs_params(trial, trial_num))._asdict()[changed_params] for trial in range(trial_num)]
        changed_param_theo_list = [(generate_eigs_params(trial, trial_num_theo))._asdict()[changed_params] for trial in range(trial_num_theo)]
    

    for plot_line in ['real','imag']:
        for label_n in range(len(label_plot_list)): 
            label = label_plot_list[label_n]
            real_part_pred_line, imag_part_pred_line = [], []
            for param_n in range(trial_num_theo):
                p_net = generate_eigs_params(param_n, trial_num_theo)
                lambda_list_pred_select,label_list_pred_select = calc_pred_outliers(p_net,dim=2)
                try:
                    lambda_pred = lambda_list_pred_select[label_list_pred_select.index(label)]
                except ValueError:
                    lambda_pred = np.nan
                real_part_pred_line.append(np.real(lambda_pred))
                imag_part_pred_line.append(np.imag(lambda_pred))
            if plot_line == 'real':
                plt.plot(changed_param_theo_list, real_part_pred_line,c=c_list[label_n],linewidth=1)
            else:
                plt.plot(changed_param_theo_list, imag_part_pred_line,c=c_list[label_n],linewidth=1)


        for label_n in range(len(label_plot_list)):
            label = label_plot_list[label_n]
            real_part_exp_mean, real_part_exp_std = [], []
            imag_part_exp_mean, imag_part_exp_std = [], []
            for param_n in range(trial_num):
                p_net = generate_eigs_params(param_n, trial_num)
                lambda_list_pred_select,label_list_pred_select = calc_pred_outliers(p_net,dim=2)

                finded_points_list = []
                for repeat_trail in range(repeat_num):
                    eigs = np.load(r'./data/' +file_name+ str(repeat_trail) + str(param_n) + 'eig.npy')
                    try:
                        pred_label_index = label_list_pred_select.index(label)
                        added_points_index = find_points(eigs, lambda_list_pred_select[pred_label_index], degenerate_num(label))
                        finded_points_list += [eigs[index] for index in added_points_index]
                    except ValueError:
                        pass
                real_part_exp_mean.append(np.mean(np.real(np.array(finded_points_list))))
                imag_part_exp_mean.append(np.mean(np.imag(np.array(finded_points_list))))
                real_part_exp_std.append(np.std(np.real(np.array(finded_points_list))))
                imag_part_exp_std.append(np.std(np.imag(np.array(finded_points_list))))
            if plot_line == 'real':
                plt.errorbar(changed_param_exp_list, real_part_exp_mean, yerr=real_part_exp_std, fmt=c_list[label_n]+'o',markersize=1, capsize=4)
                plt.ylabel("$Re(\\lambda)$",fontsize=15)
            else: 
                plt.errorbar(changed_param_exp_list, imag_part_exp_mean, yerr=imag_part_exp_std, fmt=c_list[label_n]+'o',markersize=1, capsize=4)
                plt.ylabel("$Im(\\lambda)$",fontsize=15)
            plt.legend(loc="lower right", labels=[label[1::] for label in label_plot_list])

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) 
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
        plt.xlabel(changed_params_latex,fontsize=15)
        plt.ylabel("$Re(\\lambda)$",fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if plot_line == 'real':
            plt.savefig(r'./figs/eigs_'+changed_params+'_real_outliers.png', bbox_inches='tight')
        else:
            plt.savefig(r'./figs/eigs_'+changed_params+'_imag_outliers.png', bbox_inches='tight')
        plt.close()
    
    #以下是理论预测
    R_pred_line = []
    for param_n in range(trial_num_theo):
        p_net = generate_eigs_params(param_n, trial_num_theo)
        radius = calc_pred_radius(p_net, dim=2)
        R_pred_line.append(radius)
    plt.plot(np.array(changed_param_theo_list), R_pred_line,c='k',linewidth=1)
    #以下是实际实验结果
    R_exp_line, R_exp_sigma = [], []
    for param_n in range(trial_num):
        R_list_one_par = []
        for repeat_trail in range(repeat_num):
            p_net = generate_eigs_params(param_n)
            lambda_list_pred_select,label_list_pred_select = calc_pred_outliers(p_net,dim=2)
            
            eigs = np.load(r'./data/' +file_name+ str(repeat_trail) + str(param_n) + 'eig.npy')
            eigs_select = eigs.copy()  
            eigs_select = get_eigs_diskpart(eigs_select, lambda_list_pred_select, label_list_pred_select)
            R_list_one_par.append(np.max(np.abs(eigs_select)))
        R_exp_line.append(np.mean(np.array(R_list_one_par)))
        R_exp_sigma.append(np.std(np.array(R_list_one_par)))

    plt.errorbar(np.array(changed_param_exp_list), R_exp_line, yerr=R_exp_sigma, fmt='ko', capsize=5)        


    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) 
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
    plt.xlabel(changed_params_latex,fontsize=15)
    plt.ylabel("$r$",fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.savefig(r'./figs/eigs_'+changed_params+'_radius.jpg', bbox_inches='tight')
    plt.close()