import numpy as np
import sys
sys.path.append("./code/")
from spatial_ultis import *
sys.path.append("./code/spike/")
from spike_ultis import *

def generate_params_dyn_try(trial:int):
    N_E, N_I = 22500, 5625
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
    J_EE, J_IE, J_EI, J_II = 0.25, 0.25, 0.25, 0.25
    sigma_EE, sigma_IE, sigma_EI, sigma_II =0,0,0,0
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.1


    p_net = Network_Params(N_E = 22500, N_I = 5625,
        N_EE = 800, N_IE = 200, N_EI = 800, N_II = 200,
        d_EE = 0.1, d_IE = 0.1, d_EI = 0.1, d_II = 0.1,
        g_bar_EE = 0.25, g_bar_EI = -1, g_bar_IE = 0.25, g_bar_II = -1,
        g_EE = 0, g_EI = 0, g_IE = 0, g_II = 0
        )
    return p_net