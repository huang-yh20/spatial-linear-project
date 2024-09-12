import sys
sys.path.append("./code/dyn/")
sys.path.append("./code/")
sys.path.append("./code/phase/")
from spatial_ultis import *
from dyn_ultis import *
from phase_params import *

T, t_step, record_step = 100, 100, 10
repeat_num = 5
trial_num = 41

if __name__ == "__main__":
    args = sys.argv[1:]
trial = int(args[0]) 

for repeat_trial in range(repeat_num):
    # p_simul = Simul_Params(T = T, t_step=t_step, record_step=record_step, activation_func=['tanh','linear'])
    # p_net = generate_params1p_phase_alpha_wave(trial, trial_num)
    # record_x = dyn_simul(p_net, p_simul, dim=2)
    # np.save(r"./data/phase_dynrec_"+'alpha_wave'+'_'+str(trial) + '_'+ str(repeat_trial)+r'.npy',record_x)

    # p_simul = Simul_Params(T = T, t_step=t_step, record_step=record_step, activation_func=['tanh','linear'])
    # p_net = generate_params1p_phase_alpha_E_I_0(trial, trial_num)
    # record_x = dyn_simul(p_net, p_simul, dim=2)
    # np.save(r"./data/phase_dynrec_"+'alpha_E_I_0'+'_'+str(trial)+ '_'+ str(repeat_trial)+r'.npy',record_x)

    # p_simul = Simul_Params(T = T, t_step=t_step, record_step=record_step, activation_func=['tanh','linear'])
    # p_net = generate_params1p_phase_alpha_E_I_1(trial, trial_num)
    # record_x = dyn_simul(p_net, p_simul, dim=2)
    # np.save(r"./data/phase_dynrec_"+'alpha_E_I_1'+'_'+str(trial)+ '_'+ str(repeat_trial)+r'.npy',record_x)

    # p_simul = Simul_Params(T = T, t_step=t_step, record_step=record_step, activation_func=['tanh','linear'])
    # p_net = generate_params1p_phase_alpha_E_I_2(trial, trial_num)
    # record_x = dyn_simul(p_net, p_simul, dim=2)
    # np.save(r"./data/phase_dynrec_"+'alpha_E_I_2'+'_'+str(trial)+ '_'+ str(repeat_trial)+r'.npy',record_x)

    p_simul = Simul_Params(T = T, t_step=t_step, record_step=record_step, activation_func=['tanh','linear'])
    p_net = generate_params1p_phase_eta(trial, trial_num)
    record_x = dyn_simul(p_net, p_simul, dim=2)
    np.save(r"./data/phase_dynrec_"+'eta'+'_'+str(trial)+ '_'+ str(repeat_trial)+r'.npy',record_x)

    p_simul = Simul_Params(T = T, t_step=t_step, record_step=record_step, activation_func=['tanh','linear'])
    p_net = generate_params1p_phase_alpha(trial, trial_num)
    record_x = dyn_simul(p_net, p_simul, dim=2)
    np.save(r"./data/phase_dynrec_"+'alpha'+'_'+str(trial)+ '_'+ str(repeat_trial)+r'.npy',record_x)
