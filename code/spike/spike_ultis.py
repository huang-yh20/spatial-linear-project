import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
from scipy.integrate import quad
from scipy.optimize import fsolve
from typing import NamedTuple, Union, Callable, List
from tqdm import trange
from typing import NamedTuple, Union, Callable, List
import sys
sys.path.append("./code/")
from spatial_ultis import *


class Spike_Simul_Params(NamedTuple):
    T: Union[int, float]
    t_step: int
    tau_r:float = 0.5
    tau_m:float = 20.0
    Delta:float = 0.55
    V_th:float = 20
    V_r:float = 10
    input_ext:List = [0.0, 0.0]


def simulate_lif_network(p_net:Network_Params, s_params:Spike_Simul_Params):
    J_spa = generate_net_sparse(p_net, dim=2, homo_fix_point=True)

    N_total = p_net.N_E + p_net.N_I  # 总神经元数量
    steps = int(s_params.T / s_params.t_step)  # 总的时间步数
    x = np.zeros(N_total)  # 初始膜电位
    spike_train = np.zeros((N_total, steps))  # 用于记录脉冲序列

    refractory_timer = np.zeros(N_total)  # 不应期计时器
    tau_m_inv = 1.0 / s_params.tau_m

    for t in range(steps):
        # 注意我此处的定义相当于是乘以了一个tau_m
        ext_input = np.concatenate([
            np.random.poisson(s_params.input_ext[0], p_net.N_E),
            np.random.poisson(s_params.input_ext[1], p_net.N_I)
        ])
        
        if t > (s_params.Delta/s_params.t_step + 1): 
            dv = (-x * tau_m_inv + J_spa.dot(spike_train[:,t-int(s_params.Delta/s_params.t_step)]) + ext_input)  * s_params.t_step
            x += dv
        else:
            dv = (-x * tau_m_inv + ext_input)  * s_params.t_step
            x += dv
        # 发放脉冲和复位条件
        spiking_neurons = (x >= s_params.V_th) & (refractory_timer <= 0)
        x[spiking_neurons] = s_params.V_r  # 发放脉冲后复位
        spike_train[spiking_neurons, t] = 1  # 记录脉冲

        # 更新不应期计时器
        refractory_timer[spiking_neurons] = s_params.tau_r / s_params.t_step
        refractory_timer[refractory_timer > 0] -= 1

    return x, spike_train




def F(mu, sigma_2, p_simul:Spike_Simul_Params):
    def inner_integrand(v):
        return np.exp(-v**2)

    def outer_integrand(u):
        inner_integral, _ = quad(inner_integrand, -np.inf, u)
        return np.exp(u**2) * inner_integral

    lower_limit = (p_simul.V_r - mu) / np.sqrt(sigma_2)
    upper_limit = (p_simul.V_th - mu) / np.sqrt(sigma_2)

    outer_integral, _ = quad(outer_integrand, lower_limit, upper_limit)

    result = 1 / (p_simul.tau_r + 2 * p_simul.tau_m * outer_integral)
    return result

def find_dyn_fix_point_spike(p_net: Network_Params, p_simul:Spike_Simul_Params):
    
    def d_firing_rate(firing_rate:List):
        mu, sigma_2, d_firing_rate = np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])
        mu[0] = p_simul.tau_m * (p_simul.input_ext[0] + p_net.g_bar_EE * firing_rate[0] + p_net.g_bar_EI * p_net.N_I / p_net.N_E * firing_rate[1])
        mu[1] = p_simul.tau_m * (p_simul.input_ext[1] + p_net.g_bar_IE * firing_rate[0] * p_net.N_E / p_net.N_I + p_net.g_bar_II * firing_rate[1])
        sigma_2[0] = p_simul.tau_m * (p_simul.input_ext[0] + (p_net.g_bar_EE**2/p_net.N_EE + p_net.g_EE**2) * firing_rate[0] + (p_net.g_bar_EI**2/p_net.N_EI + p_net.g_EI**2) * p_net.N_I/p_net.N_E * firing_rate[1])
        sigma_2[1] = p_simul.tau_m * (p_simul.input_ext[1] + (p_net.g_bar_IE**2/p_net.N_IE + p_net.g_IE**2) * p_net.N_E/p_net.N_I  * firing_rate[0] + (p_net.g_bar_II**2/p_net.N_II + p_net.g_II**2) * firing_rate[1])
        d_firing_rate[0] = - firing_rate[0] + F(mu[0], sigma_2[0], p_simul)
        d_firing_rate[1] = - firing_rate[1] + F(mu[1], sigma_2[1], p_simul)
        return d_firing_rate
        
    initial_guesses = [np.random.randn(2) for _ in range(200)]

    fixed_points = []
    for guess in initial_guesses:
        point = fsolve(d_firing_rate, guess)
        if np.sqrt((np.array(d_firing_rate(point)) ** 2).sum()) < 1e-3:
            if not any(np.allclose(point, fp, atol=1e-3) for fp in fixed_points):
                fixed_points.append(point)
    return fixed_points

def calc_eff_p_net_spike(p_net: Network_Params, p_simul:Spike_Simul_Params):
  
    eps = 1e-4

    fixed_points = find_dyn_fix_point_spike(p_net,p_simul)

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

    mu, sigma_2 = np.array([0.0, 0.0]), np.array([0.0, 0.0])
    mu[0] = p_simul.tau_m * (p_simul.input_ext[0] + p_net.g_bar_EE * fixed_point[0] + p_net.g_bar_EI * p_net.N_I / p_net.N_E * fixed_point[1])
    mu[1] = p_simul.tau_m * (p_simul.input_ext[1] + p_net.g_bar_IE * fixed_point[0] * p_net.N_E / p_net.N_I + p_net.g_bar_II * fixed_point[1])
    sigma_2[0] = p_simul.tau_m * (p_simul.input_ext[0] + (p_net.g_bar_EE**2/p_net.N_EE + p_net.g_EE**2) * fixed_point[0] + (p_net.g_bar_EI**2/p_net.N_EI + p_net.g_EI**2) * p_net.N_I/p_net.N_E * fixed_point[1])
    sigma_2[1] = p_simul.tau_m * (p_simul.input_ext[1] + (p_net.g_bar_IE**2/p_net.N_IE + p_net.g_IE**2) * p_net.N_E/p_net.N_I  * fixed_point[0] + (p_net.g_bar_II**2/p_net.N_II + p_net.g_II**2) * fixed_point[1])

    gamma_mu, gamma_sigma_2 = [0.0, 0.0], [0.0, 0.0]
    gamma_mu[0] = (F(mu[0] + eps, sigma_2[0],p_simul) - F(mu[0] - eps, sigma_2[0],p_simul))/(2 * eps)
    gamma_mu[1] = (F(mu[1] + eps, sigma_2[1],p_simul) - F(mu[1] - eps, sigma_2[1],p_simul))/(2 * eps)
    gamma_sigma_2[0] = (F(mu[0], sigma_2[0]  + eps,p_simul) - F(mu[0], sigma_2[0] - eps,p_simul))/(2 * eps)
    gamma_sigma_2[1] = (F(mu[1], sigma_2[1]  + eps,p_simul) - F(mu[1], sigma_2[1] - eps,p_simul))/(2 * eps)

    g_bar_EE_eff = p_simul.tau_m * (p_net.g_bar_EE * gamma_mu[0] + (p_net.g_bar_EE ** 2/p_net.N_EE + p_net.g_EE ** 2) * gamma_sigma_2[0])
    g_bar_EI_eff = p_simul.tau_m * (p_net.g_bar_EI * gamma_mu[0] + (p_net.g_bar_EI ** 2/p_net.N_EI + p_net.g_EI ** 2) * gamma_sigma_2[0])
    g_bar_IE_eff = p_simul.tau_m * (p_net.g_bar_IE * gamma_mu[1] + (p_net.g_bar_IE ** 2/p_net.N_IE + p_net.g_IE ** 2) * gamma_sigma_2[1])
    g_bar_II_eff = p_simul.tau_m * (p_net.g_bar_II * gamma_mu[1] + (p_net.g_bar_II ** 2/p_net.N_II + p_net.g_II ** 2) * gamma_sigma_2[1])

    g_EE_eff = p_simul.tau_m * p_net.g_EE * np.sqrt((gamma_mu[0] + 2 * p_net.g_bar_EE/p_net.N_EE * gamma_sigma_2[0])**2 + 2 * (p_net.g_EE**2 / p_net.N_EE) * gamma_sigma_2[0])
    g_EI_eff = p_simul.tau_m * p_net.g_EI * np.sqrt((gamma_mu[0] + 2 * p_net.g_bar_EI/p_net.N_EI * gamma_sigma_2[0])**2 + 2 * (p_net.g_EI**2 / p_net.N_EI) * gamma_sigma_2[0])
    g_IE_eff = p_simul.tau_m * p_net.g_IE * np.sqrt((gamma_mu[1] + 2 * p_net.g_bar_IE/p_net.N_IE * gamma_sigma_2[1])**2 + 2 * (p_net.g_IE**2 / p_net.N_IE) * gamma_sigma_2[1])
    g_II_eff = p_simul.tau_m * p_net.g_II * np.sqrt((gamma_mu[1] + 2 * p_net.g_bar_II/p_net.N_II * gamma_sigma_2[1])**2 + 2 * (p_net.g_II**2 / p_net.N_II) * gamma_sigma_2[1])


    p_net_eff = Network_Params(N_E = p_net.N_E, N_I = p_net.N_I,
        N_EE = p_net.N_EE, N_IE = p_net.N_IE, N_EI = p_net.N_EI, N_II = p_net.N_II,
        d_EE = p_net.d_EE, d_IE = p_net.d_IE, d_EI = p_net.d_EI, d_II = p_net.d_II,
        g_bar_EE = g_bar_EE_eff, g_bar_EI = g_bar_EI_eff, g_bar_IE = g_bar_IE_eff, g_bar_II = g_bar_II_eff,
        g_EE = g_EE_eff, g_EI = g_EI_eff, g_IE = g_IE_eff, g_II = g_II_eff)
    
    return p_net_eff


