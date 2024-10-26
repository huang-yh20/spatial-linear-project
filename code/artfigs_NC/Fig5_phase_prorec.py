import sys
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/artfigs_NC/")
from spatial_ultis import *
from dyn_ultis import *
from artfigs_NC_params import *
from artfigs_NC_ultis import *

T, t_step, record_step = 200, 100, 10
repeat_num = 5
trial_num = 21

if __name__ == "__main__":
    args = sys.argv[1:]
trial1, trial2 = int(args[0]), int(args[1])

for repeat_trial in range(0,repeat_num):
    p_simul = Simul_Params(T = T, t_step=t_step, record_step=record_step, activation_func=['tanh','linear'])
    p_net = generate_params_phase_d_II_g_bar_II(trial1, trial2, trial_num)
    record_x = dyn_simul(p_net, p_simul, dim=2)
    np.save(r"./data/artfigs_NC_"+'d_II_g_bar_II'+'_'+str(trial1)+'_'+str(trial2)+'_'+ str(repeat_trial)+r'.npy',record_x)
