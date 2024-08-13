import sys
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/phase/")
from spatial_ultis import *
from dyn_ultis import *
from phase_params import *

file_name = 'try'
generate_params_phase_func = generate_params_phase_try
T, t_step, record_step = 40, 100, 10
repeat_num = 1

if __name__ == "__main__":
    args = sys.argv[1:]
trial1, trial2 = int(args[0]), int(args[1])

for repeat_trial in range(repeat_num):
    p_simul = Simul_Params(T = T, t_step=t_step, record_step=record_step, activation_func=['tanh','linear'])
    p_net = generate_params_phase_func(trial1, trial2)

    record_x = dyn_simul(p_net, p_simul, dim=2)
    np.save(r"./data/phase_dynrec_"+file_name+'_'+str(trial1)+'_'+str(trial2)+'_'+ str(repeat_trial)+r'.npy',record_x)

