import sys
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/phase/")
from spatial_ultis import *
from dyn_ultis import *
from phase_ultis import *
from phase_params import *

p_simul = Simul_Params(T = 100, t_step=100, record_step=10, activation_func=['tanh','linear'])
plot_phase_diagram1p("alpha_try", (0.2,0.4), r"$\alpha$", generate_params1p_phase_alpha_try, p_simul, trial_num = 21, repeat_num = 1)