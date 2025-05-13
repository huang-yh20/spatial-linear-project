import sys
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/artfigs_NC/")
from spatial_ultis import *
from dyn_ultis import *
from artfigs_NC_params import *


if __name__ == "__main__":
    args = sys.argv[1:]
trial, repeat_trial = int(args[0]), int(args[1])

file_name_list = ['wospa_g_bar_IE_g_bar_EE_10_5','wospa_g_bar_IE_g_bar_EE_18_15', 'wspa_g_bar_IE_g_bar_EE_5_15', 'wospa_g_bar_IE_g_bar_EE_18_15']
p_net_list = [generate_params_phase_wospa_g_bar_IE_g_bar_EE_L(10, 5), generate_params_phase_wospa_g_bar_IE_g_bar_EE_L(5, 15), generate_params_phase_wospa_g_bar_IE_g_bar_EE_L(18, 15),\
              generate_params_phase_wspa_g_bar_IE_g_bar_EE_L(5, 15), generate_params_phase_wspa_g_bar_IE_g_bar_EE_L(18, 15)]

p_simul = Simul_Params(T = 2000, t_step=5, record_step=10, activation_func=['tanh','tanh_high'], external_input="noise",tau_m=20.0)
p_net = p_net_list[trial]
record_x = dyn_simul(p_net, p_simul, dim=2, homo_fix_point=False)
np.save(r"./data/artfigs_NC_"+ file_name_list[trial]+'_'+str(repeat_trial)+r'.npy',record_x)


