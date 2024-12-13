import sys
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/artfigs_NC/")
from spatial_ultis import *
from dyn_ultis import *
from artfigs_NC_params import *

repeat_num = 10
trial_num = 41

if __name__ == "__main__":
    args = sys.argv[1:]
trial = int(args[0])

for repeat_trial in range(0,repeat_num):
    p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['thres_linear','thres_powerlaw'], external_input="DC_noise",tau_m=20.0)
    p_net = generate_params_phase_wave_thres_L_1p(trial, trial_num)
    record_x = dyn_simul(p_net, p_simul, dim=2, homo_fix_point=True)
    np.save(r"./data/artfigs_NC_"+'wave_thres_L_1p'+'_'+str(trial)+'_'+str(repeat_trial)+r'.npy',record_x)    

