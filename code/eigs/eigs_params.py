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

def generate_params_eigs2D_d_II(trial:int, trial_num:int = 6):
    rescale = 1250
    N_E, N_I = 22500, 5625
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
    J_EE, J_IE, J_EI, J_II = 5/rescale, 9/rescale, -9/rescale, -12/rescale
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 10/rescale, 18/rescale, 18/rescale, 24/rescale
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.2
    d_II_list = list(np.linspace(0.1, 0.3, trial_num))
    
    d_II = d_II_list[trial]

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net

def generate_params_eigs2D_g_bar_EE(trial:int, trial_num:int = 6):
    rescale = 1250
    N_E, N_I = 22500, 5625
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
    J_EE, J_IE, J_EI, J_II = 5/rescale, 9/rescale, -9/rescale, -12/rescale
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 10/rescale, 18/rescale, 18/rescale, 24/rescale
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.2
    j_EE_list = np.linspace(0, 6.25, trial_num)
    
    J_EE = j_EE_list[trial]/rescale

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net


def generate_params_eigs2D_g_bar_EI(trial:int, trial_num:int = 6):
    rescale = 1250
    N_E, N_I = 22500, 5625
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
    J_EE, J_IE, J_EI, J_II = 5/rescale, 9/rescale, -9/rescale, -12/rescale
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 10/rescale, 18/rescale, 18/rescale, 24/rescale
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.2
    j_EI_list = np.linspace(0, -20, trial_num)
    
    J_EI = j_EI_list[trial]/rescale

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net


def generate_params_eigs2D_alpha(trial:int, trial_num:int = 6):
    rescale = 1
    N_E, N_I = 22500, 5625
    d_EE, d_IE, d_EI, d_II = 0.2, 0.2, 0.2, 0.2
    alpha = 0.5
    j_EE, j_IE, j_EI, j_II = 2, 3, -12, -18
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 2*np.pi*(d_EE**2)*N_E*alpha,2*np.pi*(d_IE**2)*N_I*alpha,2*np.pi*(d_EI**2)*N_E*alpha,2*np.pi*(d_II**2)*N_I*alpha
    J_EE, J_IE, J_EI, J_II = j_EE/(conn_NEE*rescale), j_IE/(conn_NIE*rescale), j_EI/(conn_NEI*rescale), j_II/(conn_NII*rescale)
    sigma_EE, sigma_IE, sigma_EI, sigma_II = J_EE, J_IE, J_EI, J_II
    alpha_list = np.linspace(0.2, 0.7, trial_num)
    
    alpha = alpha_list[trial]
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 2*np.pi*(d_EE**2)*N_E*alpha,2*np.pi*(d_IE**2)*N_I*alpha,2*np.pi*(d_EI**2)*N_E*alpha,2*np.pi*(d_II**2)*N_I*alpha
    
    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net


def generate_params_eigs2D_g_bar_IE(trial:int, trial_num:int = 6):
    rescale = 1
    N_E, N_I = 22500, 5625
    d_EE, d_IE, d_EI, d_II = 0.2, 0.2, 0.2, 0.2
    alpha = 0.2
    j_EE, j_IE, j_EI, j_II = 2, 3, -12, -18
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 2*np.pi*(d_EE**2)*N_E*alpha,2*np.pi*(d_IE**2)*N_I*alpha,2*np.pi*(d_EI**2)*N_E*alpha,2*np.pi*(d_II**2)*N_I*alpha
    J_EE, J_IE, J_EI, J_II = j_EE/(conn_NEE*rescale), j_IE/(conn_NIE*rescale), j_EI/(conn_NEI*rescale), j_II/(conn_NII*rescale)
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 0,0,0,0
    j_IE_list = np.linspace(0, 50, trial_num)

    J_IE = j_IE_list[trial]/(conn_NIE*rescale)

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net   

def generate_params_eigs2D_g_bar_II(trial:int, trial_num:int = 6):
    rescale = 1
    N_E, N_I = 22500, 5625
    d_EE, d_IE, d_EI, d_II = 0.2, 0.2, 0.2, 0.2
    alpha = 0.2
    j_EE, j_IE, j_EI, j_II = 2, 3, -12, -18
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 2*np.pi*(d_EE**2)*N_E*alpha,2*np.pi*(d_IE**2)*N_I*alpha,2*np.pi*(d_EI**2)*N_E*alpha,2*np.pi*(d_II**2)*N_I*alpha
    J_EE, J_IE, J_EI, J_II = j_EE/(conn_NEE*rescale), j_IE/(conn_NIE*rescale), j_EI/(conn_NEI*rescale), j_II/(conn_NII*rescale)
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 0,0,0,0
    j_II_list = np.linspace(-5, -30, trial_num)

    J_II = j_II_list[trial]/(conn_NII*rescale)

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net      

def generate_params_eigs2D_g_IE(trial:int, trial_num:int = 6):
    rescale = 1
    N_E, N_I = 22500, 5625
    d_EE, d_IE, d_EI, d_II = 0.2, 0.2, 0.2, 0.2
    alpha = 0.5
    j_EE, j_IE, j_EI, j_II = 2, 3, -12, -18
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 2*np.pi*(d_EE**2)*N_E*alpha,2*np.pi*(d_IE**2)*N_I*alpha,2*np.pi*(d_EI**2)*N_E*alpha,2*np.pi*(d_II**2)*N_I*alpha
    J_EE, J_IE, J_EI, J_II = j_EE/(conn_NEE*rescale), j_IE/(conn_NIE*rescale), j_EI/(conn_NEI*rescale), j_II/(conn_NII*rescale)
    sigma_EE, sigma_IE, sigma_EI, sigma_II = J_EE, J_IE, J_EI, J_II
    sigma_IE_list = np.linspace(0, 50, trial_num) * sigma_IE

    sigma_IE = sigma_IE_list[trial]

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net      


def generate_params_eigs2D_g_II(trial:int, trial_num:int = 6):
    rescale = 1
    N_E, N_I = 22500, 5625
    d_EE, d_IE, d_EI, d_II = 0.2, 0.2, 0.2, 0.2
    alpha = 0.5
    j_EE, j_IE, j_EI, j_II = 2, 3, -12, -18
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 2*np.pi*(d_EE**2)*N_E*alpha,2*np.pi*(d_IE**2)*N_I*alpha,2*np.pi*(d_EI**2)*N_E*alpha,2*np.pi*(d_II**2)*N_I*alpha
    J_EE, J_IE, J_EI, J_II = j_EE/(conn_NEE*rescale), j_IE/(conn_NIE*rescale), j_EI/(conn_NEI*rescale), j_II/(conn_NII*rescale)
    sigma_EE, sigma_IE, sigma_EI, sigma_II = J_EE, J_IE, J_EI, J_II
    sigma_II_list = np.linspace(0, 2, trial_num) * sigma_II

    sigma_II = sigma_II_list[trial]

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net      





