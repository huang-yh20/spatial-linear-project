import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter,gaussian_filter1d
from tqdm import trange
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import imageio
from typing import NamedTuple, Union, Callable, List
import sys
sys.path.append("./code/")
from spatial_ultis import *

guess_num = 10 #for finding fixed points

class Simul_Params(NamedTuple):
    T: Union[int, float]
    t_step: int
    record_step: int
    activation_func: Union[str, List[str]] = "linear"
    external_input: str = "noise" 
    tau_m:float = 1.0

#以下是激活函数
def activation_func_tanh(x):
    max_firing_rate = 1
    return max_firing_rate*(np.tanh(x/max_firing_rate).astype(np.float32))
def activation_func_tanh_high(x):
    max_firing_rate = 5
    return max_firing_rate*(np.tanh(x/max_firing_rate).astype(np.float32))
def activation_func_linear(x):
    return x
def activation_func_rectified_linear_lowthres(x):
    max_firing_rate = 1
    return np.minimum(max_firing_rate, np.maximum(-max_firing_rate, x))
def activation_func_rectified_linear_highthres(x):
    max_firing_rate = 5
    return np.minimum(max_firing_rate, np.maximum(-max_firing_rate, x))
def activation_func_thres_linear(x):
    return 10.0 * np.maximum(0, x)
def activation_func_thres_powerlaw(x):
    return 2.0 * ((np.maximum(0, x))**2)
def activation_func_shifted_sigmoid(x):
    max_rate, V_th, sigma = 10, 20, 5
    return max_rate/(1 + np.exp(-(x - V_th)/sigma))

activation_func_dict = {"linear": activation_func_linear, "tanh":activation_func_tanh, "tanh_high":activation_func_tanh_high, "rectified_linear_lowthres":activation_func_rectified_linear_lowthres, "rectified_linear_highthres":activation_func_rectified_linear_highthres,
                        "thres_linear": activation_func_thres_linear, "thres_powerlaw": activation_func_thres_powerlaw, "shifted_sigmoid":activation_func_shifted_sigmoid}

#以下是外界输入，返回一个tuple，代表非噪声项和噪声项
def external_input_noise(t, p_net:Network_Params):
    return (np.zeros(p_net.N_E+p_net.N_I), 0.1*(np.random.randn(p_net.N_E+p_net.N_I).astype(np.float32)))

def external_input_DC_noise(t, p_net:Network_Params):
    return (np.ones(p_net.N_E+p_net.N_I), 0.1*(np.random.randn(p_net.N_E+p_net.N_I).astype(np.float32)))
external_input_dict = {"noise": external_input_noise, "DC_noise": external_input_DC_noise}

def dyn_simul(p_net:Network_Params, p_simul:Simul_Params, dim=1, homo_fix_point=False):
    if type(p_simul.activation_func) == str:
        activation_func_list = [activation_func_dict[p_simul.activation_func], activation_func_dict[p_simul.activation_func]]
    elif type(p_simul.activation_func) == list:
        activation_func_list = [activation_func_dict[p_simul.activation_func[0]], activation_func_dict[p_simul.activation_func[1]]]

    external_input = external_input_dict[p_simul.external_input]

    # dist_list = calc_dist(p_net, dim = dim)
    # J = (generate_net(p_net, dist_list)).astype(np.float32)
    # J_spa = spa.csr_matrix(J)
    J_spa = generate_net_sparse(p_net, dim = dim, homo_fix_point=homo_fix_point)
    # x = np.zeros((p_net.N_E+p_net.N_I,))
    # x = np.random.randn(p_net.N_E+p_net.N_I) * 1 #TEMP
    x = np.random.uniform(0, 10, p_net.N_E+p_net.N_I)
    
    record_x = []
    for step in trange(p_simul.T*p_simul.t_step):
        external_input_tuple = external_input(step/p_simul.t_step, p_net)
        x *= np.exp(-1/(p_simul.t_step * p_simul.tau_m)).astype(np.float32)
        activated_x_E = activation_func_list[0](x[0:p_net.N_E])
        activated_x_I = activation_func_list[1](x[p_net.N_E:p_net.N_E+p_net.N_I])
        activated_x = np.concatenate((activated_x_E,activated_x_I))
        x += (J_spa @ activated_x/ p_simul.t_step + external_input_tuple[0] / p_simul.t_step + external_input_tuple[1] * np.sqrt(1/p_simul.t_step).astype(np.float32))/p_simul.tau_m
        if step % p_simul.record_step == 0:
            record_x.append(x.copy())
    record_x = np.array(record_x,dtype=np.float32)

    return record_x

def product_gif(record_x: list, p_net:Network_Params = None, p_simul: Simul_Params = None, file_name: str = 'try', dim=1, filter = False):
    p_net = p_net if p_net != None else generate_params_default(2)
    p_simul = p_simul if p_simul != None else Simul_Params(T = 40, t_step=100, record_step=10, activation_func='linear')
    
    if type(p_simul.activation_func) == str:
        activation_func_list = [activation_func_dict[p_simul.activation_func], activation_func_dict[p_simul.activation_func]]
    elif type(p_simul.activation_func) == list:
        activation_func_list = [activation_func_dict[p_simul.activation_func[0]], activation_func_dict[p_simul.activation_func[1]]]

    activated_x_E = activation_func_list[0](record_x[:,0:p_net.N_E])
    activated_x_I = activation_func_list[1](record_x[:,p_net.N_E:p_net.N_E+p_net.N_I])
    record_x = np.concatenate((activated_x_E,activated_x_I),axis=1)

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
                ax.plot(np.linspace(0,1,p_net.N_E), x_smoothed_E,'r.',markersize=0.5,label='Exc.')
                ax.plot(np.linspace(0,1,p_net.N_I), x_smoothed_I,'b.',markersize=0.5,label='Inh.')
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
            #TEMP
            scale_max = np.minimum(np.max(record_x),300)
            scale_min = np.min(record_x)
        for step in trange(np.shape(record_x)[0]):
            fig, ax = plt.subplots()
            #TEMP
            norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.5*scale_max, vmax=scale_max)
            #TEMP: cmap=plt.cm.RdBu
            if filter == False:
                #img = ax.imshow(record_x[step][0:int(np.ceil(np.sqrt(p_net.N_E)))**2].reshape((int(np.ceil(np.sqrt(p_net.N_E))), int(np.ceil(np.sqrt(p_net.N_E))))), cmap=plt.cm.RdBu, norm=norm, origin='upper', aspect=1)
                img = ax.imshow(record_x_img[step,:,:], norm=norm, origin='upper', aspect=1)
            else:
                img = ax.imshow(x_imag_smoothed[step,:,:], norm=norm, origin='upper', aspect=1)
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
    p_simul_tanhlinear = Simul_Params(T = 40, t_step=100, record_step=10, activation_func=['tanh','linear'])
    p_simul_list = [('linear',p_simul_linear), ('tanhlinear',p_simul_tanhlinear)]
    trial_num_list = [0,5] #only simul nonlinear case

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
            product_gif(record_x, p_net, p_simul, 'dyn_gif_'+file_name+'_'+activation_func_str+str(trial), dim=dim, filter=False)
            product_gif(record_x, p_net, p_simul, 'dyn_gif_smoothed_'+file_name+'_'+activation_func_str+str(trial), dim=dim, filter=True)

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


def find_dyn_fix_point(p_net: Network_Params, p_simul: Simul_Params):
    if type(p_simul.activation_func) == str:
        activation_func_list = [activation_func_dict[p_simul.activation_func], activation_func_dict[p_simul.activation_func]]
    elif type(p_simul.activation_func) == list:
        activation_func_list = [activation_func_dict[p_simul.activation_func[0]], activation_func_dict[p_simul.activation_func[1]]]
    external_input = external_input_dict[p_simul.external_input]

    def df(x):
        dx_dt = np.array([0.0,0.0])
        dx_dt[0] = -x[0] + p_net.g_bar_EE * activation_func_list[0](x[0]) + p_net.g_bar_EI * p_net.N_I / p_net.N_E * activation_func_list[1](x[1]) + np.mean(external_input(0, p_net)[0][0:p_net.N_E])
        dx_dt[1] = -x[1] + p_net.g_bar_IE * activation_func_list[0](x[0]) * p_net.N_E / p_net.N_I + p_net.g_bar_II * activation_func_list[1](x[1]) + np.mean(external_input(0, p_net)[0][p_net.N_E:p_net.N_E+p_net.N_I])
        return dx_dt

    #TEMP
    if type(p_simul.activation_func) == str:
        activation_func_type_list = [p_simul.activation_func]
    else:
        activation_func_type_list = p_simul.activation_func

    fixed_points = []
    while len(fixed_points) == 0:
        if all(activation_func in ['tanh','tanh_high','linear'] for activation_func in activation_func_type_list):
            return [np.array([0,0])]
        else:
            initial_guesses = [np.random.uniform(0,10,2) for _ in range(guess_num)]
        for guess in initial_guesses:
            point = fsolve(df, guess)
            if np.sqrt((np.array(df(point)) ** 2).sum()) < 1e-3:
                if not any(np.allclose(point, fp, atol=1e-3) for fp in fixed_points):
                    fixed_points.append(point)
    return fixed_points

#TODO 完整的理论推导，以下只是暂时的

def gauss_int(mu, sigma_2, f:Callable):
    def f_int(x,sigma):
        return (1 / (np.sqrt(2 * np.pi) * np.abs(sigma))) * np.exp(-((x - mu) ** 2) / (2 * np.abs(sigma) ** 2)) * f(x)
    if sigma_2 <= 0:
        return f(mu)
    else:
        sigma = np.sqrt(sigma_2)
        return quad(lambda x:f_int(x,sigma), -np.inf, np.inf)[0]
    

def find_dyn_fix_point_with_variance(p_net: Network_Params, p_simul: Simul_Params, dim=2):
    if type(p_simul.activation_func) == str:
        activation_func_list = [activation_func_dict[p_simul.activation_func], activation_func_dict[p_simul.activation_func]]
    elif type(p_simul.activation_func) == list:
        activation_func_list = [activation_func_dict[p_simul.activation_func[0]], activation_func_dict[p_simul.activation_func[1]]]
    external_input = external_input_dict[p_simul.external_input]        
        
    #x定义为前两个是E跟I的均值，而后两个则是方差
    def df(x):
        dx_dt = np.array([0.0,0.0,0.0,0.0])
        dx_dt[0] = -x[0] + p_net.g_bar_EE * gauss_int(x[0], x[2], activation_func_list[0]) + p_net.g_bar_EI * p_net.N_I / p_net.N_E * gauss_int(x[1], x[3], activation_func_list[1]) + np.mean(external_input(0, p_net)[0][0:p_net.N_E])
        dx_dt[1] = -x[1] + p_net.g_bar_IE * gauss_int(x[0], x[2], activation_func_list[0]) * p_net.N_E / p_net.N_I + p_net.g_bar_II * gauss_int(x[1], x[3], activation_func_list[1]) + np.mean(external_input(0, p_net)[0][p_net.N_E:p_net.N_E+p_net.N_I])
        
        J_EE, J_EI, J_IE, J_II = p_net.g_bar_EE/p_net.N_EE, p_net.g_bar_EI/p_net.N_EI, p_net.g_bar_IE/p_net.N_IE, p_net.g_bar_II/p_net.N_II
        sigma_EE, sigma_EI, sigma_IE, sigma_II = p_net.g_EE/np.sqrt(p_net.N_EE), p_net.g_EI/np.sqrt(p_net.N_EI), p_net.g_IE/np.sqrt(p_net.N_IE), p_net.g_II/np.sqrt(p_net.N_II)  
        if dim == 1:
            sigma_eff_EE = p_net.N_EE * (p_net.N_E/p_net.N_E) * ((1 - p_net.N_EE/(2 * np.sqrt(np.pi)*p_net.d_EE*(p_net.N_E))) * J_EE **2 + sigma_EE**2)
            sigma_eff_IE = p_net.N_IE * (p_net.N_E/p_net.N_I) * ((1 - p_net.N_IE/(2 * np.sqrt(np.pi)*p_net.d_IE*(p_net.N_I))) * J_IE **2 + sigma_IE**2)
            sigma_eff_EI = p_net.N_EI * (p_net.N_I/p_net.N_E) * ((1 - p_net.N_EI/(2 * np.sqrt(np.pi)*p_net.d_EI*(p_net.N_E))) * J_EI **2 + sigma_EI**2)
            sigma_eff_II = p_net.N_II * (p_net.N_I/p_net.N_I) * ((1 - p_net.N_II/(2 * np.sqrt(np.pi)*p_net.d_II*(p_net.N_I))) * J_II **2 + sigma_II**2)
        elif dim == 2:
            sigma_eff_EE = p_net.N_EE * (p_net.N_E/p_net.N_E) * ((1 - p_net.N_EE/(4 * np.pi * p_net.d_EE**2 * p_net.N_E)) * J_EE **2 + sigma_EE**2)
            sigma_eff_IE = p_net.N_IE * (p_net.N_E/p_net.N_I) * ((1 - p_net.N_IE/(4 * np.pi * p_net.d_IE**2 * p_net.N_I)) * J_IE **2 + sigma_IE**2)
            sigma_eff_EI = p_net.N_EI * (p_net.N_I/p_net.N_E) * ((1 - p_net.N_EI/(4 * np.pi * p_net.d_EI**2 * p_net.N_E)) * J_EI **2 + sigma_EI**2)
            sigma_eff_II = p_net.N_II * (p_net.N_I/p_net.N_I) * ((1 - p_net.N_II/(4 * np.pi * p_net.d_II**2 * p_net.N_I)) * J_II **2 + sigma_II**2)
        else:
            print('dimenstion error')
            return None       
        
        #TODO 这边存噪声强度没得给出
        dx_dt[2] = -x[2] + sigma_eff_EE * gauss_int(x[0], x[2], lambda x:activation_func_list[0](x)**2) + sigma_eff_EI * gauss_int(x[1], x[3], lambda x:activation_func_list[1](x)**2) + 0.01
        dx_dt[3] = -x[3] + sigma_eff_IE * gauss_int(x[0], x[2], lambda x:activation_func_list[0](x)**2) + sigma_eff_II * gauss_int(x[1], x[3], lambda x:activation_func_list[1](x)**2) + 0.01
        return dx_dt
    
    # initial_guesses = [np.random.uniform(low=0, high=5, size=(4,)) for _ in range(50)]

    fixed_points = []
    while len(fixed_points) == 0:
        initial_guesses = [np.random.uniform(low=0, high=10, size=(4,)) for _ in range(guess_num)] #TEMP
        for guess in initial_guesses:
            point = fsolve(df, guess)
            if np.sqrt((np.array(df(point)) ** 2).sum()) < 1e-3:
                if not any(np.allclose(point, fp, atol=1e-3) for fp in fixed_points):
                    fixed_points.append(point)
    return fixed_points

#TEMP 固定均值为0
def find_dyn_fix_point_with_variance_tanh(p_net: Network_Params, p_simul: Simul_Params, dim=2):
    if type(p_simul.activation_func) == str:
        activation_func_list = [activation_func_dict[p_simul.activation_func], activation_func_dict[p_simul.activation_func]]
    elif type(p_simul.activation_func) == list:
        activation_func_list = [activation_func_dict[p_simul.activation_func[0]], activation_func_dict[p_simul.activation_func[1]]]
    external_input = external_input_dict[p_simul.external_input]        
        
    #x定义为前两个是E跟I的均值，而后两个则是方差
    def df(x):
        dx_dt = np.array([0.0,0.0])

        J_EE, J_EI, J_IE, J_II = p_net.g_bar_EE/p_net.N_EE, p_net.g_bar_EI/p_net.N_EI, p_net.g_bar_IE/p_net.N_IE, p_net.g_bar_II/p_net.N_II
        sigma_EE, sigma_EI, sigma_IE, sigma_II = p_net.g_EE/np.sqrt(p_net.N_EE), p_net.g_EI/np.sqrt(p_net.N_EI), p_net.g_IE/np.sqrt(p_net.N_IE), p_net.g_II/np.sqrt(p_net.N_II)  
        if dim == 1:
            sigma_eff_EE = p_net.N_EE * (p_net.N_E/p_net.N_E) * ((1 - p_net.N_EE/(2 * np.sqrt(np.pi)*p_net.d_EE*(p_net.N_E))) * J_EE **2 + sigma_EE**2)
            sigma_eff_IE = p_net.N_IE * (p_net.N_E/p_net.N_I) * ((1 - p_net.N_IE/(2 * np.sqrt(np.pi)*p_net.d_IE*(p_net.N_I))) * J_IE **2 + sigma_IE**2)
            sigma_eff_EI = p_net.N_EI * (p_net.N_I/p_net.N_E) * ((1 - p_net.N_EI/(2 * np.sqrt(np.pi)*p_net.d_EI*(p_net.N_E))) * J_EI **2 + sigma_EI**2)
            sigma_eff_II = p_net.N_II * (p_net.N_I/p_net.N_I) * ((1 - p_net.N_II/(2 * np.sqrt(np.pi)*p_net.d_II*(p_net.N_I))) * J_II **2 + sigma_II**2)
        elif dim == 2:
            sigma_eff_EE = p_net.N_EE * (p_net.N_E/p_net.N_E) * ((1 - p_net.N_EE/(4 * np.pi * p_net.d_EE**2 * p_net.N_E)) * J_EE **2 + sigma_EE**2)
            sigma_eff_IE = p_net.N_IE * (p_net.N_E/p_net.N_I) * ((1 - p_net.N_IE/(4 * np.pi * p_net.d_IE**2 * p_net.N_I)) * J_IE **2 + sigma_IE**2)
            sigma_eff_EI = p_net.N_EI * (p_net.N_I/p_net.N_E) * ((1 - p_net.N_EI/(4 * np.pi * p_net.d_EI**2 * p_net.N_E)) * J_EI **2 + sigma_EI**2)
            sigma_eff_II = p_net.N_II * (p_net.N_I/p_net.N_I) * ((1 - p_net.N_II/(4 * np.pi * p_net.d_II**2 * p_net.N_I)) * J_II **2 + sigma_II**2)
        else:
            print('dimenstion error')
            return None       
        
        #TODO 这边存噪声强度没得给出
        dx_dt[0] = -x[0] + sigma_eff_EE * gauss_int(0, x[0], lambda x:activation_func_list[0](x)**2) + sigma_eff_EI * gauss_int(0, x[1], lambda x:activation_func_list[1](x)**2) + 0.01
        dx_dt[1] = -x[1] + sigma_eff_IE * gauss_int(0, x[0], lambda x:activation_func_list[0](x)**2) + sigma_eff_II * gauss_int(0, x[1], lambda x:activation_func_list[1](x)**2) + 0.01
        return dx_dt
    
    # initial_guesses = [np.random.uniform(low=0, high=5, size=(4,)) for _ in range(50)]

    fixed_points = []
    while len(fixed_points) == 0:
        guess = np.random.uniform(low=0, high=10, size=(2,))
        point = fsolve(df, guess)
        if np.sqrt((np.array(df(point)) ** 2).sum()) < 1e-3:
            if not any(np.allclose(point, fp, atol=1e-3) for fp in fixed_points):
                fixed_point = [0.0, 0.0, point[0], point[1]]
                fixed_points.append(fixed_point)
    return fixed_points




def calc_eff_p_net(p_net: Network_Params, p_simul: Simul_Params):
    if type(p_simul.activation_func) == str:
        activation_func_list = [activation_func_dict[p_simul.activation_func], activation_func_dict[p_simul.activation_func]]
    elif type(p_simul.activation_func) == list:
        activation_func_list = [activation_func_dict[p_simul.activation_func[0]], activation_func_dict[p_simul.activation_func[1]]]
    external_input = external_input_noise    
    eps = 1e-5

    fixed_points = find_dyn_fix_point(p_net, p_simul)

    #TODO: select the correct fix point
    fixed_points = sorted(fixed_points, key=lambda x: x[0])
    if len(fixed_points) == 1:
        fixed_point = fixed_points[0]
    elif len(fixed_points) == 2:
        fixed_point = fixed_points[1]
    elif len(fixed_points) == 3:
        fixed_point = fixed_points[1]
    else:
        print("fixed points number error!", len(fixed_points))

    d_phi_list = [0.0, 0.0]
    d_phi_list[0] = (activation_func_list[0](fixed_point[0] + eps) - activation_func_list[0](fixed_point[0] - eps))/(2 * eps)
    d_phi_list[1] = (activation_func_list[1](fixed_point[1] + eps) - activation_func_list[1](fixed_point[1] - eps))/(2 * eps)

    p_net_eff = Network_Params(N_E = p_net.N_E, N_I = p_net.N_I,
        N_EE = p_net.N_EE, N_IE = p_net.N_IE, N_EI = p_net.N_EI, N_II = p_net.N_II,
        d_EE = p_net.d_EE, d_IE = p_net.d_IE, d_EI = p_net.d_EI, d_II = p_net.d_II,
        g_bar_EE = p_net.g_bar_EE * d_phi_list[0], g_bar_EI = p_net.g_bar_EI * d_phi_list[1], g_bar_IE = p_net.g_bar_IE * d_phi_list[0], g_bar_II = p_net.g_bar_II * d_phi_list[1],
        g_EE = p_net.g_EE * d_phi_list[0], g_EI = p_net.g_EI * d_phi_list[1], g_IE = p_net.g_IE * d_phi_list[0], g_II = p_net.g_II * d_phi_list[1])
    
    return p_net_eff

#TEMP 只是用来计算混沌到底是不是稳定的
def calc_eff_p_net_with_variance(p_net: Network_Params, p_simul: Simul_Params):
    if type(p_simul.activation_func) == str:
        activation_func_list = [activation_func_dict[p_simul.activation_func], activation_func_dict[p_simul.activation_func]]
    elif type(p_simul.activation_func) == list:
        activation_func_list = [activation_func_dict[p_simul.activation_func[0]], activation_func_dict[p_simul.activation_func[1]]]
    external_input = external_input_noise    
    eps_d = 1e-5

    def d_phi_E(x):
        return (activation_func_list[0](x+eps_d) - activation_func_list[0](x-eps_d))/(2*eps_d)
    def d_phi_I(x):
        return (activation_func_list[1](x+eps_d) - activation_func_list[1](x-eps_d))/(2*eps_d)
    
    if type(p_simul.activation_func) == str:
        activation_func_type_list = [p_simul.activation_func]
    else:
        activation_func_type_list = p_simul.activation_func
    if all(activation_func in ['tanh','tanh_high','linear'] for activation_func in activation_func_type_list):
        fixed_points = find_dyn_fix_point_with_variance_tanh(p_net, p_simul)
    else:
        fixed_points = find_dyn_fix_point_with_variance(p_net, p_simul)

    #TODO: select the correct fix point
    fixed_points = sorted(fixed_points, key=lambda x: x[0])
    if len(fixed_points) == 1:
        fixed_point = fixed_points[0]
    elif len(fixed_points) == 2:
        fixed_point = fixed_points[1]
    elif len(fixed_points) == 3:
        fixed_point = fixed_points[1]
    else:
        print("fixed points number error!", len(fixed_points))

    d_phi_E_mean = gauss_int(fixed_point[0], fixed_point[2], d_phi_E)
    d_phi_I_mean = gauss_int(fixed_point[1], fixed_point[3], d_phi_I)

    

    p_net_eff = Network_Params(N_E = p_net.N_E, N_I = p_net.N_I,
        N_EE = p_net.N_EE, N_IE = p_net.N_IE, N_EI = p_net.N_EI, N_II = p_net.N_II,
        d_EE = p_net.d_EE, d_IE = p_net.d_IE, d_EI = p_net.d_EI, d_II = p_net.d_II,
        g_bar_EE = p_net.g_bar_EE * d_phi_E_mean, g_bar_EI = p_net.g_bar_EI * d_phi_E_mean, g_bar_IE = p_net.g_bar_IE * d_phi_I_mean, g_bar_II = p_net.g_bar_II * d_phi_I_mean,
        g_EE = p_net.g_EE * d_phi_E_mean, g_EI = p_net.g_EI * d_phi_I_mean, g_IE = p_net.g_IE * d_phi_E_mean, g_II = p_net.g_II * d_phi_I_mean)
    
    return p_net_eff

#这个是真正正确的关于混沌稳定性的结果

def calc_eff_perturb_matrix(p_net:Network_Params, p_simul:Simul_Params, n:int, dim:int = 2):

    fixed_points = find_dyn_fix_point_with_variance(p_net, p_simul)

    #TODO: select the correct fix point
    fixed_points = sorted(fixed_points, key=lambda x: x[0])
    if len(fixed_points) == 1:
        fixed_point = fixed_points[0]
    elif len(fixed_points) == 2:
        fixed_point = fixed_points[1]
    elif len(fixed_points) == 3:
        fixed_point = fixed_points[1]
    else:
        print("fixed points number error!", len(fixed_points))

    if type(p_simul.activation_func) == str:
        activation_func_list = [activation_func_dict[p_simul.activation_func], activation_func_dict[p_simul.activation_func]]
    elif type(p_simul.activation_func) == list:
        activation_func_list = [activation_func_dict[p_simul.activation_func[0]], activation_func_dict[p_simul.activation_func[1]]]
    external_input = external_input_noise    
    eps_d = 1e-5

    def d_phi_E(x):
        return (activation_func_list[0](x+eps_d) - activation_func_list[0](x-eps_d))/(2*eps_d)
    def d_phi_I(x):
        return (activation_func_list[1](x+eps_d) - activation_func_list[1](x-eps_d))/(2*eps_d)
    def dd_phi_E(x):
        return (d_phi_E(x+eps_d) - d_phi_E(x-eps_d))/(2*eps_d)
    def dd_phi_I(x):
        return (d_phi_I(x+eps_d) - d_phi_I(x-eps_d))/(2*eps_d)   

    d_phi_E_mean = gauss_int(fixed_point[0], fixed_point[2], d_phi_E)
    d_phi_I_mean = gauss_int(fixed_point[1], fixed_point[3], d_phi_I)
    d_phi_mean = np.array([d_phi_E_mean, d_phi_I_mean])     

    dd_phi_E_mean = gauss_int(fixed_point[0], fixed_point[2], dd_phi_E)
    dd_phi_I_mean = gauss_int(fixed_point[1], fixed_point[3], dd_phi_I)
    dd_phi_mean = np.array([dd_phi_E_mean, dd_phi_I_mean])    

    d_phi_E_square_mean = gauss_int(fixed_point[0], fixed_point[2], lambda x: d_phi_E(x) ** 2)
    d_phi_I_square_mean = gauss_int(fixed_point[1], fixed_point[3], lambda x: d_phi_I(x) ** 2)   
    d_phi_square_mean = np.array([d_phi_E_square_mean, d_phi_I_square_mean])

    phi_dphi_E_mean = gauss_int(fixed_point[0], fixed_point[2], lambda x: activation_func_list[0](x) * d_phi_E(x))
    phi_dphi_I_mean = gauss_int(fixed_point[1], fixed_point[3], lambda x: activation_func_list[1](x) * d_phi_I(x))
    phi_dphi_mean = np.array([phi_dphi_E_mean, phi_dphi_I_mean])   

    phi_ddphi_E_mean = gauss_int(fixed_point[0], fixed_point[2], lambda x: activation_func_list[0](x) * dd_phi_E(x))
    phi_ddphi_I_mean = gauss_int(fixed_point[1], fixed_point[3], lambda x: activation_func_list[1](x) * dd_phi_I(x))
    phi_ddphi_mean = np.array([phi_ddphi_E_mean, phi_ddphi_I_mean])
    
    d = np.array([[p_net.d_EE, p_net.d_EI],
                  [p_net.d_IE, p_net.d_II]])
    g_bar = np.array([[p_net.g_bar_EE, p_net.g_bar_EI],
                      [p_net.g_bar_IE, p_net.g_bar_II]])

    J_EE, J_EI, J_IE, J_II = p_net.g_bar_EE/p_net.N_EE, p_net.g_bar_EI/p_net.N_EI, p_net.g_bar_IE/p_net.N_IE, p_net.g_bar_II/p_net.N_II
    sigma_EE, sigma_EI, sigma_IE, sigma_II = p_net.g_EE/np.sqrt(p_net.N_EE), p_net.g_EI/np.sqrt(p_net.N_EI), p_net.g_IE/np.sqrt(p_net.N_IE), p_net.g_II/np.sqrt(p_net.N_II)  
    if dim == 1:
        sigma_eff_EE = p_net.N_EE * (p_net.N_E/p_net.N_E) * ((1 - p_net.N_EE/(2 * np.sqrt(np.pi)*p_net.d_EE*(p_net.N_E))) * J_EE **2 + sigma_EE**2)
        sigma_eff_IE = p_net.N_IE * (p_net.N_E/p_net.N_I) * ((1 - p_net.N_IE/(2 * np.sqrt(np.pi)*p_net.d_IE*(p_net.N_I))) * J_IE **2 + sigma_IE**2)
        sigma_eff_EI = p_net.N_EI * (p_net.N_I/p_net.N_E) * ((1 - p_net.N_EI/(2 * np.sqrt(np.pi)*p_net.d_EI*(p_net.N_E))) * J_EI **2 + sigma_EI**2)
        sigma_eff_II = p_net.N_II * (p_net.N_I/p_net.N_I) * ((1 - p_net.N_II/(2 * np.sqrt(np.pi)*p_net.d_II*(p_net.N_I))) * J_II **2 + sigma_II**2)
    elif dim == 2:
        sigma_eff_EE = p_net.N_EE * (p_net.N_E/p_net.N_E) * ((1 - p_net.N_EE/(4 * np.pi * p_net.d_EE**2 * p_net.N_E)) * J_EE **2 + sigma_EE**2)
        sigma_eff_IE = p_net.N_IE * (p_net.N_E/p_net.N_I) * ((1 - p_net.N_IE/(4 * np.pi * p_net.d_IE**2 * p_net.N_I)) * J_IE **2 + sigma_IE**2)
        sigma_eff_EI = p_net.N_EI * (p_net.N_I/p_net.N_E) * ((1 - p_net.N_EI/(4 * np.pi * p_net.d_EI**2 * p_net.N_E)) * J_EI **2 + sigma_EI**2)
        sigma_eff_II = p_net.N_II * (p_net.N_I/p_net.N_I) * ((1 - p_net.N_II/(4 * np.pi * p_net.d_II**2 * p_net.N_I)) * J_II **2 + sigma_II**2)
    else:
        print('dimenstion error')
        return None 
    
    g = np.array([[sigma_eff_EE, sigma_eff_EI],
                  [sigma_eff_IE, sigma_eff_II]])
    
    if dim == 1:
        spa_F = np.array([[p_net.N_EE/(np.sqrt(2*np.pi) * p_net.N_E * p_net.d_EE), p_net.N_EI/(np.sqrt(2*np.pi) * p_net.N_E * p_net.d_EI)],
                          [p_net.N_IE/(np.sqrt(2*np.pi) * p_net.N_I * p_net.d_IE), p_net.N_II/(np.sqrt(2*np.pi) * p_net.N_I * p_net.d_II)]])
    elif dim == 2:
        spa_F = np.array([[p_net.N_EE/((4*np.pi) * p_net.N_E * p_net.d_EE**2), p_net.N_EI/((4*np.pi) * p_net.N_E * p_net.d_EI**2)],
                          [p_net.N_IE/((4*np.pi) * p_net.N_I * p_net.d_IE**2), p_net.N_II/((4*np.pi) * p_net.N_I * p_net.d_II**2)]])   
        
    k = 2 * np.pi * n
    
    E_g_bar = 2 * ( (((g ** 2) * np.exp(-k**2 * d**2 / 2)).dot(np.diag(phi_dphi_mean))).dot(g_bar * np.exp(-k**2 * d**2 / 2)) \
                   - (g ** 2) * np.exp(-k**2 * d**2 / 4) * spa_F).dot(np.diag(phi_dphi_mean)).dot(g_bar * np.exp(-k**2 * d**2 / 2) )
    A_g_bar = np.diag(d_phi_mean).dot(g_bar) * np.exp(-d**2 * k **2 /2) + np.diag(dd_phi_mean).dot(0.5 *E_g_bar)
    D = (g ** 2 * np.exp(- d**2 * k**2 / 2)).dot(np.diag(d_phi_square_mean + phi_ddphi_mean))
    B = 0.5 * np.diag(dd_phi_mean).dot(D)

    perturb_matrix = np.block([[A_g_bar, B],
                               [E_g_bar, D]])
    
    return perturb_matrix
    




