import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
from tqdm import trange
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import imageio
import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn/")
from dyn_ultis import *

def generate_params_phase_d_II_g_bar_II(trial1:int, trial2:int, trial_num:int = 21):
    trial_num = trial_num

    N_E, N_I = 22500, 5625
    alpha = 0.8
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 5.5, 5, -5, -4.25
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.1
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_EE, g_EI, g_IE, g_II = 0.1,0.1,0.1,0.1

    d_II_list = np.linspace(0.02,0.20,trial_num)
    d_II = d_II_list[trial1]
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))

    g_bar_II_list = list(np.linspace(-1, -7, trial_num))
    g_bar_II = g_bar_II_list[trial2]

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    
    return p_net  

def generate_params_phase_d_II_g_bar_II_L(trial1:int, trial2:int, trial_num:int = 21):
    trial_num = trial_num

    N_E, N_I = 22500*4, 5625*4
    alpha = 0.8
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 5.5, 5, -5, -4.25
    d_EE, d_IE, d_EI, d_II = 0.05, 0.05, 0.05, 0.05
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_EE, g_EI, g_IE, g_II = 0.1,0.1,0.1,0.1

    d_II_list = np.linspace(0.01,0.1,trial_num)
    d_II = d_II_list[trial1]
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))

    g_bar_II_list = list(np.linspace(-1, -7, trial_num))
    g_bar_II = g_bar_II_list[trial2]

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    
    return p_net  