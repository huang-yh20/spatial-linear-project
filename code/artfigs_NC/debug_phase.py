import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn")
from dyn_ultis import *
sys.path.append("./code/artfigs_NC")
from artfigs_NC.artfigs_NC_ultis import *
from artfigs_NC_params import *

# changed_params = ['g_bar_IE', 'g_bar_EE']
# changed_params_latex = [r'$\bar{g}_{IE}$', r'$\bar{g}_{EE}$']
# p_simul = Simul_Params(T = 2000, t_step=25, record_step=10, activation_func=['tanh','tanh_high'], external_input="noise",tau_m=1.0)
# plot_phase_diagram_new("debug_wspa_0.04", changed_params, changed_params_latex, generate_params_phase_wspa_g_bar_IE_g_bar_EE_L, p_simul, repeat_num=20)

changed_params = ['g_bar_IE', 'g_bar_EE']
changed_params_latex = [r'$\bar{g}_{IE}/\bar{g}_I$', r'$\bar{g}_{EE}/\bar{g}_I$']
changed_params_values = [[0.8, 1.8], [0.7, 1.7]]
p_simul = Simul_Params(T = 2000, t_step=100, record_step=10, activation_func=['tanh','tanh_high'], external_input="noise",tau_m=1.0)
plot_phase_diagram_new("debug_wospa_0.01", changed_params, changed_params_latex, generate_params_phase_wospa_g_bar_IE_g_bar_EE_L, p_simul, repeat_num=20, changed_params_values=changed_params_values)