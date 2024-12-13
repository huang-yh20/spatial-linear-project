import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn")
from dyn_ultis import *
sys.path.append("./code/artfigs_NC")
from artfigs_NC.artfigs_NC_ultis import *
from artfigs_NC_params import *

generate_phase_params1p = generate_params_phase_wave_thres_L_1p
p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['thres_linear','thres_powerlaw'], external_input="DC_noise",tau_m=20.0)
plot_phase_diagram1p_new("wave_thres_L_1p", (0.1, 0.4), "\alpha", generate_phase_params1p, p_simul, trial_num = 41, repeat_num = 10)