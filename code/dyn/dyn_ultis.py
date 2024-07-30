import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
from scipy.ndimage import gaussian_filter,gaussian_filter1d
from tqdm import trange
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import imageio
from typing import NamedTuple, Union, Callable
import sys
sys.path.append("./code/")
from spatial_ultis import *


class Simul_Params(NamedTuple):
    T: Union[int, float]
    t_step: int
    record_step: int
    activation_func: str = "linear"
    external_input: str = "noise" 

#以下是激活函数
def activation_func_tanh(x):
    max_firing_rate = 1
    return max_firing_rate*np.tanh(x/max_firing_rate)
def activation_func_linear(x):
    return x

#以下是外界输入，返回一个tuple，代表非噪声项和噪声项
def external_input_noise(t, p_net:Network_Params):
    return (0, 0.1*np.random.randn(p_net.N_E+p_net.N_I))

def dyn_simul(p_net:Network_Params, p_simul:Simul_Params, dim=1):
    if p_simul.activation_func == "linear":
        activation_func = activation_func_linear
    elif p_simul.activation_func == "tanh":
        activation_func = activation_func_tanh

    external_input = external_input_noise

    dist_list = calc_dist(p_net, dim = dim)
    J = generate_net(p_net, dist_list)
    J_spa = spa.csr_matrix(J)
    x = np.zeros((p_net.N_E+p_net.N_I,))
    
    record_x = []
    for step in trange(p_simul.T*p_simul.t_step):
        external_input_tuple = external_input(step/p_simul.t_step, p_net)
        x *= np.exp(-1/p_simul.t_step)
        x += J_spa @ activation_func(x)/ p_simul.t_step + external_input_tuple[0] / p_simul.t_step + external_input_tuple[1] * np.sqrt(1/p_simul.t_step) 
        if step % p_simul.record_step == 0:
            record_x.append(x.copy())
    record_x = np.array(record_x)

    return record_x

def product_gif(record_x: list, p_net:Network_Params = None, p_simul: Simul_Params = None, file_name: str = 'try', dim=1, filter = False):
    p_net = p_net if p_net != None else generate_params_default(2)
    p_simul = p_simul if p_simul != None else Simul_Params(T = 40, t_step=100, record_step=10, activation_func='linear')
    
    if p_simul.activation_func == "linear":
        activation_func = activation_func_linear
    elif p_simul.activation_func == "tanh":
        activation_func = activation_func_tanh
    record_x = activation_func(record_x)

    frames = []
    scale_max = np.max(record_x)
    if dim == 1:
        if filter:
            x_smoothed_E = gaussian_filter1d(record_x[:,0:p_net.N_E],200)
            x_smoothed_I = gaussian_filter1d(record_x[:,p_net.N_E:p_net.N_E+p_net.N_I],50)
            scale_max = np.maximum(np.max(x_smoothed_E),np.max(x_smoothed_I))
        else:
            scale_max = np.max(record_x)
        for step in trange(np.shape(record_x)[0]):
            fig, ax = plt.subplots()
            # 绘制数据
            if filter == False: 
                ax.plot(np.linspace(0,1,p_net.N_E), record_x[step,0:p_net.N_E],'ro',markersize=0.5,label='Exc.')
                ax.plot(np.linspace(0,1,p_net.N_I), record_x[step,p_net.N_E:p_net.N_E+p_net.N_I],'bo',markersize=0.5,label='Inh.')
            else:
                ax.plot(np.linspace(0,1,p_net.N_E), x_smoothed_E,'r-',markersize=0.5,label='Exc.')
                ax.plot(np.linspace(0,1,p_net.N_I), x_smoothed_I,'b-',markersize=0.5,label='Inh.')
            ax.plot(np.linspace(0,1,p_net.N_E), np.linspace(0,0,p_net.N_E), "k--")
            
            # 设置标签和刻度
            ax.set_xlabel("Location", fontsize=15)
            ax.set_ylabel("Activity", fontsize=15)
            ax.set_xticks([0, 0.5, 1])
            ax.set_ylim((-scale_max,scale_max))
            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            # 添加图例
            ax.legend(loc = 'upper right',fontsize=15)
            
            # 保存当前帧
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            
            # 关闭当前图表
            plt.close(fig)

    if dim == 2:
        record_x = np.pad(record_x[:,0:p_net.N_E], ((0,0),(0,int(np.ceil(np.sqrt(p_net.N_E))**2 - p_net.N_E))))
        record_x_img = record_x.reshape(np.shape(record_x)[0],int(np.ceil(np.sqrt(p_net.N_E))), int(np.ceil(np.sqrt(p_net.N_E))))
        if filter:
            x_imag_smoothed = []
            for step in range(np.shape(record_x)[0]):
                x_imag_smoothed.append(gaussian_filter(record_x_img[step,:,:],10))
            x_imag_smoothed = np.array(x_imag_smoothed)
            scale_max = np.max(x_imag_smoothed)
        else:
            scale_max = np.max(record_x)
        for step in trange(np.shape(record_x)[0]):
            fig, ax = plt.subplots()
            
            norm = mcolors.TwoSlopeNorm(vmin=-scale_max, vcenter=0, vmax=scale_max)
            if filter == False:
                #img = ax.imshow(record_x[step][0:int(np.ceil(np.sqrt(p_net.N_E)))**2].reshape((int(np.ceil(np.sqrt(p_net.N_E))), int(np.ceil(np.sqrt(p_net.N_E))))), cmap=plt.cm.RdBu, norm=norm, origin='upper', aspect=1)
                img = ax.imshow(record_x_img[step,:,:], cmap=plt.cm.RdBu, norm=norm, origin='upper', aspect=1)
            else:
                img = ax.imshow(x_imag_smoothed[step,:,:], cmap=plt.cm.RdBu, norm=norm, origin='upper', aspect=1)
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
            
            # 保存当前帧
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
            
            # 关闭当前图表
            plt.close(fig)

    # 将所有帧保存为一个GIF
    imageio.mimsave("./figs/"+file_name+".gif", frames, duration=0.05)

def product_dyn_simul_gif(file_name, generate_params:Callable, dim=1):
    p_simul_linear = Simul_Params(T = 40, t_step=100, record_step=10, activation_func='linear')
    p_simul_tanh = Simul_Params(T = 40, t_step=100, record_step=10, activation_func='tanh')
    p_simul_list = [('linear',p_simul_linear), ('tanh',p_simul_tanh)]
    trial_num_list = [3,5]

    #calculate xlim and ylim
    fig_lim_x_min, fig_lim_y_min, fig_lim_x_max, fig_lim_y_max= -1.1, -1.1, 1.1, 1.1
    for trial in range(max(trial_num_list)):
        p_net = generate_params(trial)
        radius = calc_pred_radius(p_net,dim=dim)
        lambda_list_pred_select, label_list_pred_select= calc_pred_outliers(p_net, dim=2)
        if len(lambda_list_pred_select) == 0:
            lambda_list_pred_select = [0]
        real_part_pred_select, imag_part_pred_select = np.real(np.array(lambda_list_pred_select)), np.imag(np.array(lambda_list_pred_select))
        fig_lim_x_min = min([fig_lim_x_min, -radius-0.1, np.min(real_part_pred_select)-0.1])
        fig_lim_y_min = min([fig_lim_y_min, -radius-0.1, np.min(imag_part_pred_select)-0.1])
        fig_lim_x_max = max([fig_lim_x_max, radius+0.1, np.max(real_part_pred_select)+0.1])
        fig_lim_y_max = max([fig_lim_y_max, radius+0.1, np.max(imag_part_pred_select)+0.1])
    fig_lim = max([fig_lim_x_max - fig_lim_x_min, fig_lim_y_max - fig_lim_y_min])
    x_center, y_center = 0.5*(fig_lim_x_max + fig_lim_x_min), 0.5*(fig_lim_y_max + fig_lim_y_min)
    fig_lim_x_min, fig_lim_x_max = x_center - 0.5 * fig_lim, x_center + 0.5 * fig_lim
    fig_lim_y_min, fig_lim_y_max = y_center - 0.5 * fig_lim, y_center + 0.5 * fig_lim

    for simul_num in range(len(p_simul_list)):
        activation_func_str, p_simul  = p_simul_list[simul_num]
        for trial in trange(trial_num_list[simul_num]):
            p_net = generate_params(trial)
            dist_list = calc_dist(p_net, dim=dim)
            J = generate_net(p_net, dist_list)

            eigs, eig_V = np.linalg.eig(J)
            
            radius = calc_pred_radius(p_net,dim=dim)
            x_dots = np.linspace(-radius, radius, 200)
            y_dots = np.sqrt(radius**2 - x_dots**2)
            
            plt.plot(x_dots, y_dots, c='lightcoral', linewidth=1)
            plt.plot(x_dots, -y_dots, c='lightcoral', linewidth=1)

            lambda_list_pred_select, label_list_pred_select= calc_pred_outliers(p_net, dim=2)
            real_part_pred_select, imag_part_pred_select = np.real(np.array(lambda_list_pred_select)), np.imag(np.array(lambda_list_pred_select))
            plt.scatter(real_part_pred_select, imag_part_pred_select, s=30, c='lightcoral', marker='x')
            real_part = np.real(eigs)
            imag_part = np.imag(eigs)
            plt.scatter(real_part, imag_part, s=3, c='none', marker='o', edgecolors='k')

            plt.plot(np.ones(100), np.linspace(fig_lim_y_min, fig_lim_y_max, 100), color='gray', linestyle='--')
            
            ax = plt.gca()
            ax.set_xlim((fig_lim_x_min, fig_lim_x_max))
            ax.set_xlabel("$Re(\\lambda)$", fontsize=15)
            ax.tick_params(axis='x', labelsize=15)  # 控制x轴刻度的字体大小
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) 
            ax.set_ylim((fig_lim_y_min, fig_lim_y_max))
            ax.set_ylabel("$Im(\\lambda)$", fontsize=15)
            ax.tick_params(axis='y', labelsize=15)  # 控制y轴刻度的字体大小
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
            ax.set_aspect('equal') 

            plt.tight_layout()
            plt.savefig(r'./figs/'+'dyn_eigs_'+file_name+'_'+activation_func_str+str(trial), bbox_inches='tight')
            plt.close()

            record_x = dyn_simul(p_net, p_simul, dim=dim)
            np.save(r'./data/'+'dyn_record_'+file_name+'_'+activation_func_str+str(trial), record_x)
            product_gif(record_x, p_net, p_simul, 'dyn_gif_'+file_name+'_'+activation_func_str+str(trial), dim=dim)

def record_to_gif():
    p_simul_linear = Simul_Params(T = 40, t_step=100, record_step=10, activation_func='linear')
    p_simul_tanh = Simul_Params(T = 40, t_step=100, record_step=10, activation_func='tanh')
    p_simul_list = [('linear',p_simul_linear), ('tanh',p_simul_tanh)]
    
    file_name_list = ['global','bump','osc','wave','chaos']
    trial_num_list = [3,5]
    for file_name in file_name_list:
        for simul_num in range(len(p_simul_list)):
            activation_func_str, p_simul = p_simul_list[simul_num]
            for trial in range(trial_num_list[simul_num]):
                record_x = np.load(r'./data/'+'dyn_record_'+file_name+'_'+activation_func_str+str(trial)+'.npy')
                product_gif(record_x, p_net=None, p_simul = p_simul, file_name='dyn_gif_'+file_name+'_'+activation_func_str+str(trial), dim=2)
                product_gif(record_x, p_net=None, p_simul = p_simul, file_name='dyn_gif_smoothed_'+file_name+'_'+activation_func_str+str(trial), dim=2, filter=True)