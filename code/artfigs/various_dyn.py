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

file_name_list = ['2dglobal','2dosc','2dbump','2dwave','2dchaos','2dwave']
generate_params_list = [generate_params_dyn_global_new, generate_params_dyn_osc_new, generate_params_dyn_bump_new, generate_params_dyn_wave_new, generate_params_dyn_chaos_new, generate_params_dyn_wave_new]
trial_params_list = [3,3,3,3,3,0]

p_simul_tanhlinear = Simul_Params(T = 40, t_step=100, record_step=10, activation_func=['tanh','linear'])
t_show_onset, t_show_step, t_show_num = 10, 4, 6

for trial_plot in trange(len(file_name_list)):
    file_name = file_name_list[trial_plot]
    generate_params_func = generate_params_list[trial_plot]
    trial_params = trial_params_list[trial_plot]

    p_net = generate_params_func(trial_params)
    calc_eigs_bool = False
    if os.path.exists(r"./data/artfigs_variousdyn_eigs_"+str(trial_plot)+"eigs.npy") and (not calc_eigs_bool):
        eigs = np.load(r"./data/artfigs_variousdyn_eigs_"+str(trial_plot)+"eigs.npy")
        eig_V = np.load(r"./data/artfigs_variousdyn_eigs_"+str(trial_plot)+"eigV.npy")
    else:
        dist_list = calc_dist(p_net, dim = 2)
        J = generate_net(p_net, dist_list)
        eigs, eig_V = np.linalg.eig(J)
        np.save(r"./data/artfigs_variousdyn_eigs_"+str(trial_plot)+"eigs.npy", eigs)
        np.save(r"./data/artfigs_variousdyn_eigs_"+str(trial_plot)+"eigV.npy", eig_V)
    
    real_part = np.real(eigs)
    imag_part = np.imag(eigs)

    #plot eigs
    fig, ax = plt.subplots()
    artfigs_plot_eigs(eigs)

    #plot eigV of largest eigs
    largest_eigs_index = np.argmax(real_part)
    plt.scatter([real_part[largest_eigs_index]],[imag_part[largest_eigs_index]],s=30,c='r',marker='^')
        
    ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper left')
    scale_max = np.max((eig_V.real)[0:p_net.N_E,:])
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
    plt.savefig(r"./figs/artfigs_variousdyn_eigs_"+str(trial_plot)+".png")
    plt.close()

    record_x = np.load(r'./data/'+'dyn_record_'+file_name+'_'+'tanhlinear'+str(trial_params)+'.npy')
    record_x = activation_func(record_x)
    #scale_max = np.max(record_x)
    scale_max = 1
    record_x_img = (record_x[:,0:p_net.N_E]).reshape(np.shape(record_x)[0],int(np.ceil(np.sqrt(p_net.N_E))), int(np.ceil(np.sqrt(p_net.N_E))))
    for trial_show in range(t_show_num):
        step_show = int((t_show_step * trial_show + t_show_onset) * p_simul_tanhlinear.t_step/p_simul_tanhlinear.record_step)
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
        plt.tight_layout()
        plt.savefig(r"./figs/artfigs_variousdyn_dynimag_"+str(trial_plot)+"_"+str(trial_show)+".png")
        plt.close()
            
        
        

        





    



'''
# 创建主图
fig, ax = plt.subplots(figsize=(8, 6))

# 在主图上绘制数据
data = [1, 2, 3, 4, 5]
ax.plot(data, label='Feature Distribution')
ax.set_title('Main Plot')
ax.legend()

# 创建插入的坐标轴
ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper right')

# 在小图上绘制数据
inset_data = [0.1, 0.2, 0.3, 0.4, 0.5]
ax_inset.plot(inset_data, color='red')
ax_inset.set_title('Inset Plot')

plt.show()
'''