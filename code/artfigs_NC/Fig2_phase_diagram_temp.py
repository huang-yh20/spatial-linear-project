import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn")
from dyn_ultis import *
sys.path.append("./code/artfigs_NC")
from artfigs_NC.artfigs_NC_ultis import *
from artfigs_NC_params import *

changed_params = ['d_II', 'g_bar_II']
changed_params_latex = [r'$d_{II}$', r'$\bar{g}_{II}$']
p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['thres_linear','thres_powerlaw'], external_input="DC_noise",tau_m=20.0)
plot_phase_diagram_new("d_II_g_bar_II_thres_L", changed_params, changed_params_latex, generate_params_phase_d_II_g_bar_II_thres_L, p_simul, repeat_num=1)

changed_params = ['g_II', 'd_II']
changed_params_latex = [r'$\sigma_{\alpha\beta}$', r'$d_{II}$']
p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['tanh','tanh_high'], external_input="noise",tau_m=20.0)
marker_list = [(15,9)]
plot_phase_diagram_new("g_d_II_L_chaos", changed_params, changed_params_latex, generate_params_phase_g_d_II_L_chaos, p_simul, repeat_num=5, marker_list=marker_list)

changed_params = ['g_bar_IE', 'g_bar_EE']
changed_params_latex = [r'$|\bar{g}_{IE}/\bar{g}_I|$', r'$|\bar{g}_{EE}/\bar{g}_I$|']
changed_params_values = [[1.0, 1.8], [0.7, 1.1]]
marker_list = [(5,15),(18,15)]
p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['tanh','tanh_high'], external_input="noise",tau_m=20.0)
plot_phase_diagram_new("wspa_g_bar_IE_g_bar_EE", changed_params, changed_params_latex, generate_params_phase_wspa_g_bar_IE_g_bar_EE_L, p_simul, repeat_num=5, changed_params_values=changed_params_values, marker_list=marker_list)

changed_params = ['N_II', 'd_II']
changed_params_latex = [r'$k^{out}_{\alpha\beta}/(2 \pi {d_{\alpha\beta}}^2 N_{\alpha})$', r'$d_{II}$']
changed_params_values = [[0.1, 0.7], [0.05, 0.12]]
p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['tanh','tanh_high'], external_input="noise",tau_m=20.0)
plot_phase_diagram_new("alpha_d_II_L_chaos", changed_params, changed_params_latex, generate_params_phase_alpha_d_II_L_chaos, p_simul, repeat_num=5, changed_params_values=changed_params_values)

