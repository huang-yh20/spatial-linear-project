import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn/")
from dyn_ultis import *
sys.path.append("./code/phase/")
from phase_ultis import *
from phase_params import *

changed_params = ['d_II', 'g_bar_EE']
changed_params_latex = [r'$d_{II}$', r'$\bar{g}_{EE}$']
p_simul = Simul_Params(T = 40, t_step=100, record_step=10, activation_func=['tanh','linear'])
plot_phase_diagram("try", changed_params, changed_params_latex, generate_params_phase_try, p_simul)