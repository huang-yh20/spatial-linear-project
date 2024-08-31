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

def generate_params_phase_d_II_g_bar_EE(trial1:int, trial2:int, trial_num:int = 21):
    trial_num = trial_num

    rescale = 160
    N_E, N_I = 22500, 5625
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.05, 0.05, 0.05, 0.07
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 5.0, 2.81, -11.25, -4.38
    g_EE, g_EI, g_IE, g_II = 0.1, 0.1, 0.1, 0.1

    d_II_list = np.linspace(0.06,0.07,trial_num)
    d_II = d_II_list[trial1]

    g_bar_EE_list = list(np.linspace(5.0, 7.5, trial_num))
    g_bar_EE = g_bar_EE_list[trial2]

    
    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net   

def generate_params_phase_g_bar_E_I_0(trial1:int, trial2:int, trial_num:int = 21):
    trial_num = trial_num

    rescale = 160
    N_E, N_I = 22500, 5625
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.05, 0.05, 0.05, 0.05
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_EE, g_EI, g_IE, g_II = 0.1, 0.1, 0.1, 0.1
    
    g_E_list = np.linspace(0,20,trial_num)
    g_I_list = np.linspace(0,-10,trial_num)
    g_E, g_I = g_E_list[trial1], g_I_list[trial2]


    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_E, g_bar_EI = g_E, g_bar_IE = g_I, g_bar_II = g_I,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net   

def generate_params_phase_g_bar_E_I_1(trial1:int, trial2:int, trial_num:int = 21):
    trial_num = trial_num

    rescale = 160
    N_E, N_I = 22500, 5625
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.05, 0.05, 0.1, 0.1
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_EE, g_EI, g_IE, g_II = 0.1, 0.1, 0.1, 0.1
    
    g_E_list = np.linspace(0,20,trial_num)
    g_I_list = np.linspace(0,-10,trial_num)
    g_E, g_I = g_E_list[trial1], g_I_list[trial2]


    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_E, g_bar_EI = g_E, g_bar_IE = g_I, g_bar_II = g_I,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net   

def generate_params_phase_g_bar_E_I_2(trial1:int, trial2:int, trial_num:int = 21):
    trial_num = trial_num

    rescale = 160
    N_E, N_I = 22500, 5625
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.05, 0.05
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_EE, g_EI, g_IE, g_II = 0.1, 0.1, 0.1, 0.1
    
    g_E_list = np.linspace(0,20,trial_num)
    g_I_list = np.linspace(0,-10,trial_num)
    g_E, g_I = g_E_list[trial1], g_I_list[trial2]


    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_E, g_bar_EI = g_E, g_bar_IE = g_I, g_bar_II = g_I,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net   

def generate_params_phase_g_bar_IE_d_II(trial1:int, trial2:int, trial_num:int = 21):
    trial_num = trial_num

    N_E, N_I = 22500, 5625
    alpha = 0.8
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 5.5, 9, -9, -4.25
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.16
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_EE, g_EI, g_IE, g_II = 0.1,0.1,0.1,0.1

    d_II_list = np.linspace(0.01,0.10,trial_num)*2
    d_II = d_II_list[trial1]
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    

    g_bar_IE_list = list(np.linspace(0,9,trial_num))
    g_bar_IE, g_bar_EI = g_bar_IE_list[trial2], -g_bar_IE_list[trial2]


    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net 

def generate_params_phase_try(trial1:int, trial2:int, trial_num:int = 21):
    trial_num = trial_num

    rescale = 160
    N_E, N_I = 4900, 1225
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.14
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 5.0, 2.81, -11.25, -4.38
    g_EE, g_EI, g_IE, g_II = 0.1, 0.1, 0.1, 0.1

    d_II_list = np.linspace(0.06,0.07,trial_num)*2
    d_II = d_II_list[trial1]

    g_bar_EE_list = list(np.linspace(5.0, 7.5, trial_num))
    g_bar_EE = g_bar_EE_list[trial2]

    
    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net  

def generate_params1p_phase_alpha(trial:int, trial_num:int = 21):
    rescale = 160
    N_E, N_I = 22500, 5625
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.05, 0.05, 0.05, 0.065
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 6.25, 2.81, -11.25, -4.38
    g_EE, g_EI, g_IE, g_II = 0.1, 0.1, 0.1, 0.1

    alpha_list = np.linspace(0.2, 0.32, trial_num)
    alpha = alpha_list[trial]
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    
    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net   

def generate_params1p_phase_alpha_try(trial:int, trial_num:int = 21):
    rescale = 160
    N_E, N_I = 4900, 1225
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.13
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 6.25, 2.81, -11.25, -4.38
    g_EE, g_EI, g_IE, g_II = 0.1, 0.1, 0.1, 0.1

    alpha_list = np.linspace(0.2, 0.4, trial_num)
    alpha = alpha_list[trial]
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    
    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net   

def generate_params1p_phase_eta_try(trial:int, trial_num:int = 21):
    rescale = 160
    N_E, N_I = 4900, 1225
    alpha = 0.8
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.13
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 6.25, 2.81, -11.25, -4.38
    g_EE, g_EI, g_IE, g_II = 0.1, 0.1, 0.1, 0.1

    eta_list = np.linspace(0, 0.3, trial_num)
    eta = eta_list[trial]
    g_EE, g_EI, g_IE, g_II = tuple(eta * np.array([g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II]))
    
    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net

def generate_params1p_phase_eta(trial:int, trial_num:int = 21):
    rescale = 160
    N_E, N_I = 22500, 5625
    alpha = 0.8
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.13
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 6.25, 2.81, -11.25, -4.38
    g_EE, g_EI, g_IE, g_II = 0.1, 0.1, 0.1, 0.1

    eta_list = np.linspace(0, 0.3, trial_num)
    eta = eta_list[trial]
    g_EE, g_EI, g_IE, g_II = tuple(eta * np.array([g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II]))
    
    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net

def generate_params1p_phase_alpha_wave(trial:int, trial_num:int = 21):
    rescale = 160
    N_E, N_I = 22500, 5625
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.05, 0.05, 0.05, 0.08
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 6.25, 2.81, -11.25, -4.38
    g_EE, g_EI, g_IE, g_II = 0.3, 0.3, 0.3, 0.3

    alpha_list = np.linspace(0.1, 0.4, trial_num)
    alpha = alpha_list[trial]
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    
    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net  

def generate_params1p_phase_alpha_E_I_0(trial:int, trial_num:int = 21):
    N_E, N_I = 22500, 5625
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.05, 0.05, 0.05, 0.05
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 5, 5, -5, -5
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_EE, g_EI, g_IE, g_II = 0.1, 0.1, 0.1, 0.1
    
    alpha_list = np.linspace(0.1, 0.5, trial_num)
    alpha = alpha_list[trial]
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net   

def generate_params1p_phase_alpha_E_I_1(trial:int, trial_num:int = 21):
    N_E, N_I = 22500, 5625
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.05, 0.05, 0.05, 0.05
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 6.3, 6.3, -5, -5
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_EE, g_EI, g_IE, g_II = 0.2, 0.2, 0.2, 0.2
    
    alpha_list = np.linspace(0.1, 0.5, trial_num)
    alpha = alpha_list[trial]
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net  

def generate_params1p_phase_alpha_E_I_2(trial:int, trial_num:int = 21):
    N_E, N_I = 22500, 5625
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.05, 0.05, 0.05, 0.05
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 6.5, 6.5, -5, -5
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_EE, g_EI, g_IE, g_II = 0.2, 0.2, 0.2, 0.2
    
    alpha_list = np.linspace(0.1, 0.5, trial_num)
    alpha = alpha_list[trial]
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net   