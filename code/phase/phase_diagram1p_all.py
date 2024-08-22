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

p_simul = Simul_Params(T = 100, t_step=100, record_step=10, activation_func=['tanh','linear'])

if trial == 1:
    plot_phase_diagram1p("alpha_wave", (0.1,0.4), r"$\alpha$", generate_params1p_phase_alpha_wave, p_simul, trial_num = 41, repeat_num = 5)

if trial == 2:
    plot_phase_diagram1p("alpha_E_I_0", (0.1,0.4), r"$\alpha$", generate_params1p_phase_alpha_E_I_0, p_simul, trial_num = 41, repeat_num = 5)

if trial == 3:
    plot_phase_diagram1p("alpha_E_I_1", (0.16, 0.32), r"$\alpha$", generate_params1p_phase_alpha_E_I_1, p_simul, trial_num = 41, repeat_num = 5)

if trial == 4:
    plot_phase_diagram1p("alpha_E_I_2", (0.12, 0.36), r"$\alpha$", generate_params1p_phase_alpha_E_I_2, p_simul, trial_num = 41, repeat_num = 5)

if trial == 5:
    plot_phase_diagram1p("eta", (0, 0.3), r"$\eta$", generate_params1p_phase_eta, p_simul, trial_num = 41, repeat_num = 5)

if trial == 6:
    plot_phase_diagram1p("alpha", (0.2, 0.4), r"$\eta$", generate_params1p_phase_alpha, p_simul, trial_num = 41, repeat_num = 5)

