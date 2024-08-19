import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn/")
from dyn_ultis import *
sys.path.append("./code/phase/")
from phase_ultis import *
from phase_params import *

changed_params = ['g_bar_EE', 'g_bar_II']
changed_params_latex = [r'$\bar{g}_{EE}$', r'$\bar{g}_{II}$']
p_simul = Simul_Params(T = 40, t_step=100, record_step=10, activation_func=['tanh','linear'])
plot_phase_diagram("g_bar_E_I_1", changed_params, changed_params_latex, generate_params_phase_g_bar_E_I_1, p_simul)