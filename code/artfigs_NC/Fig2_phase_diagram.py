import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn")
from dyn_ultis import *
sys.path.append("./code/artfigs_NC")
from artfigs_NC.artfigs_NC_ultis import *
from artfigs_NC_params import *

# changed_params = ['d_II', 'g_bar_II']
# changed_params_latex = [r'$d_{II}$', r'$\bar{g}_{II}$']
# p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['thres_linear','thres_powerlaw'], external_input="DC_noise",tau_m=20.0)
# plot_phase_diagram_new("homo_d_II_g_bar_II_thres_L", changed_params, changed_params_latex, generate_params_phase_d_II_g_bar_II_thres_L, p_simul, repeat_num=1)

changed_params = ['d_II', 'g_bar_II']
changed_params_latex = [r'$d_{II}$', r'$\bar{g}_{II}$']
p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['tanh','linear'], external_input="noise",tau_m=20.0)
plot_phase_diagram_new("d_II_g_bar_II_L", changed_params, changed_params_latex, generate_params_phase_d_II_g_bar_II_L, p_simul, repeat_num=1)