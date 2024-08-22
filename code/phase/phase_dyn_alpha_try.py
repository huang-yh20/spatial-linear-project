import sys
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/phase/")
from spatial_ultis import *
from dyn_ultis import *
from phase_params import *

T, t_step, record_step = 100, 100, 10
repeat_num = 1
trial_num = 21
file_name = 'alpha_try'

for trial in trange(trial_num):
    for repeat_trial in range(repeat_num):
            p_simul = Simul_Params(T = T, t_step=t_step, record_step=record_step, activation_func=['tanh','linear'])
            p_net = generate_params1p_phase_alpha_try(trial, trial_num)
            record_x = dyn_simul(p_net, p_simul, dim=2)
            np.save(r"./data/phase_dynrec_"+file_name+'_'+str(trial)+ '_'+ str(repeat_trial)+r'.npy',record_x)
