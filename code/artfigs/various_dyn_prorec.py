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

T, t_step, record_step = 100, 100, 10
file_name_list = ['2dglobal','2dosc','2dbump','2dwave','2dchaos','stable']
generate_params_list = [generate_params_dyn_global_new, generate_params_dyn_osc_new, generate_params_dyn_bump_new, generate_params_dyn_wave_new, generate_params_dyn_chaos_new, generate_params_dyn_wave_new]
trial_params_list = [3,3,3,3,3,0]

if __name__ == "__main__":
    args = sys.argv[1:]
repeat_trial = int(args[0])

p_simul = Simul_Params(T = T, t_step=t_step, record_step=record_step, activation_func=['tanh','linear'])
for generate_params_trial in range(len(generate_params_list)):
    generate_params_func = generate_params_list[generate_params_trial]
    p_net = generate_params_func(trial_params_list[generate_params_trial])
    record_x = dyn_simul(p_net, p_simul, dim=2)
    np.save(r"./data/artfigs_dynrec_"+file_name_list[generate_params_trial]+'_'+str(repeat_trial)+r'.npy',record_x)


    
