import sys
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/artfigs_NC/")
from spatial_ultis import *
from dyn_ultis import *
from artfigs_NC_params import *

repeat_num = 1
trial_num = 21

if __name__ == "__main__":
    args = sys.argv[1:]
trial1, trial2 = int(args[0]), int(args[1])

for repeat_trial in range(0,repeat_num):
    # p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['thres_linear','thres_powerlaw'], external_input="DC_noise",tau_m=20.0)
    # p_net = generate_params_phase_d_II_g_bar_II_thres_L(trial1, trial2, trial_num)
    # record_x = dyn_simul(p_net, p_simul, dim=2, homo_fix_point=True)
    # np.save(r"./data/artfigs_NC_"+'homo_d_II_g_bar_II_thres_L'+'_'+str(trial1)+'_'+str(trial2)+'_'+str(repeat_trial)+r'.npy',record_x)

    p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['tanh','tanh_high'], external_input="noise",tau_m=20.0)
    p_net = generate_params_phase_d_II_g_bar_II_L(trial1, trial2, trial_num)
    record_x = dyn_simul(p_net, p_simul, dim=2, homo_fix_point=False)
    np.save(r"./data/artfigs_NC_"+'d_II_g_bar_II_L'+'_'+str(trial1)+'_'+str(trial2)+'_'+str(repeat_trial)+r'.npy',record_x)    
