import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn/")
from dyn_ultis import *
sys.path.append("./code/phase/")
from phase_ultis import *
from phase_params import *

if __name__ == "__main__":
    args = sys.argv[1:]
trial = int(args[0])

if trial == 0:
    changed_params = ['d_II', 'g_bar_EE']
    changed_params_latex = [r'$d_{II}$', r'$\bar{g}_{EE}$']
    p_simul = Simul_Params(T = 100, t_step=100, record_step=10, activation_func=['tanh','linear'])
    plot_phase_diagram("d_II_g_bar_EE", changed_params, changed_params_latex, generate_params_phase_d_II_g_bar_EE, p_simul, repeat_num=5)
if trial == 1:
    changed_params = ['g_bar_EE', 'g_bar_II']
    changed_params_latex = [r'$\bar{g}_{E}$', r'$\bar{g}_{I}$']
    p_simul = Simul_Params(T = 100, t_step=100, record_step=10, activation_func=['tanh','linear'])
    plot_phase_diagram("g_bar_E_I_0", changed_params, changed_params_latex, generate_params_phase_g_bar_E_I_0, p_simul, repeat_num=5)
if trial == 2:
    changed_params = ['g_bar_EE', 'g_bar_II']
    changed_params_latex = [r'$\bar{g}_{E}$', r'$\bar{g}_{I}$']
    p_simul = Simul_Params(T = 100, t_step=100, record_step=10, activation_func=['tanh','linear'])
    plot_phase_diagram("g_bar_E_I_1", changed_params, changed_params_latex, generate_params_phase_g_bar_E_I_1, p_simul)
if trial == 3:
    changed_params = ['g_bar_EE', 'g_bar_II']
    changed_params_latex = [r'$\bar{g}_{E}$', r'$\bar{g}_{I}$']
    p_simul = Simul_Params(T = 100, t_step=100, record_step=10, activation_func=['tanh','linear'])
    plot_phase_diagram("g_bar_E_I_2", changed_params, changed_params_latex, generate_params_phase_g_bar_E_I_2, p_simul)
if trial == 4:
    changed_params = ['d_II', 'g_bar_IE']
    changed_params_latex = [r'$d_{II}$', r'$\bar{g}_{IE}$']
    p_simul = Simul_Params(T = 100, t_step=100, record_step=10, activation_func=['tanh','linear'])
    plot_phase_diagram("d_II_g_bar_IE", changed_params, changed_params_latex, generate_params_phase_d_II_g_bar_IE, p_simul, repeat_num=5)

