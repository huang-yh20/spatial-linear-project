import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
from tqdm import trange
import matplotlib.colors as mcolors
import imageio
import sys
sys.path.append("./exp/code/")
from spatial_ultis import *
from dyn_ultis import *

def generate_params_dyn_global(trial:int):
    rescale = 1600
    N_E, N_I = 22500, 5625
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
    J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -8/rescale, -8/rescale
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 8/rescale, 8/rescale, -16/rescale, -16/rescale
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.1

    rescale_list = list(rescale * np.array([1/0.5, 1/0.75, 1, 1/1.25, 1/1.5]))
    rescale = rescale_list[trial]
    J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -8/rescale, -8/rescale

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net

def generate_params_dyn_bump(trial:int):
    rescale = 1400
    N_E, N_I = 22500, 5625
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
    J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -18/rescale, -18/rescale
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 8/rescale, 8/rescale, 20/rescale, 20/rescale
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.2, 0.2

    rescale_list = list(rescale * np.array([1/0.5, 1/0.75, 1, 1/1.25, 1/1.5]))
    rescale = rescale_list[trial]
    J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -18/rescale, -18/rescale

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net

def generate_params_dyn_osc(trial:int):
    rescale = 1600
    N_E, N_I = 22500, 5625
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
    J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -8/rescale, 0/rescale
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 8/rescale, 8/rescale, 16/rescale, 0/rescale
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.1     

    rescale_list = list(rescale * np.array([1/0.5, 1/0.75, 1, 1/1.25, 1/1.5]))
    rescale = rescale_list[trial]
    J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -8/rescale, 0/rescale

    
    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net

def generate_params_dyn_wave(trial:int):
    rescale = 830
    rescale_list = list(rescale * [1/0.5, 1/0.75, 1, 1/1.25, 1/1.5])
    N_E, N_I = 22500, 5625
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
    J_EE, J_IE, J_EI, J_II = 5/rescale, 9/rescale, -9/rescale, -14/rescale
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 5/rescale, 9/rescale, 9/rescale, 14/rescale
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.16
    
    rescale_list = list(rescale * np.array([1/0.5, 1/0.75, 1, 1/1.25, 1/1.5]))
    rescale = rescale_list[trial]
    J_EE, J_IE, J_EI, J_II = 5/rescale, 9/rescale, -9/rescale, -14/rescale

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net

def generate_params_dyn_chaos(trial:int):
    #chaos
    rescale = 1600
    sigma_rescale = 5.5
    N_E, N_I = 22500, 5625
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
    J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -8/rescale, 0/rescale
    sigma_EE, sigma_IE, sigma_EI, sigma_II = tuple(np.array([8/rescale, 8/rescale, 16/rescale, 0/rescale]) * sigma_rescale)
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.1   

    sigma_rescale_list = list(np.array([0.2,0.6,1,1.4,1.8]) * sigma_rescale)
    sigma_rescale = sigma_rescale_list[trial]
    sigma_EE, sigma_IE, sigma_EI, sigma_II = tuple(np.array([8/rescale, 8/rescale, 16/rescale, 0/rescale]) * sigma_rescale)
    
    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net
    
def generate_params_dyn_try(trial:int):
    rescale = 207.5
    N_E, N_I = 4900, 1225
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 200, 50, 200, 50
    J_EE, J_IE, J_EI, J_II = 5/rescale, 9/rescale, -9/rescale, -14/rescale
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 0,0,0,0
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.16

    rescale_list = list(rescale * np.array([1/0.5, 1/0.75, 1, 1/1.25, 1/1.5]))
    rescale = rescale_list[trial]
    J_EE, J_IE, J_EI, J_II = 5/rescale, 9/rescale, -9/rescale, -14/rescale

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net