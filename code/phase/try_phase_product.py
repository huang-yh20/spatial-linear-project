import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
from tqdm import trange
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import imageio
import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn/")
from dyn_ultis import *

#少神经元，用于测试
def generate_params_phase_try(trial1:int, trial2:int, trial_num=21):
    trial_num = trial_num

    rescale = 160
    N_E, N_I = 4900, 1225
    alpha = 0.5
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.14
    conn_NEE, conn_NIE, conn_NEI, conn_NII = tuple(alpha * np.array([2*np.pi * N_E * d_EE **2, 2*np.pi * N_I * d_IE **2, 2*np.pi * N_E * d_EI **2,2*np.pi * N_I * d_II **2]))
    g_bar_EE, g_bar_IE, g_bar_EI, g_bar_II = 5.0, 2.81, -11.25, -4.38
    g_EE, g_EI, g_IE, g_II = 0.1, 0.1, 0.1, 0.1

    d_II_list = np.linspace(0.06,0.07,trial_num)*2
    d_II = d_II_list[trial1]

    g_bar_EE_list = list(np.linspace(5.0, 7.5, trial_num))
    g_bar_EE = g_bar_EE_list[trial2]

    
    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = g_bar_EE, g_bar_EI = g_bar_EI, g_bar_IE = g_bar_IE, g_bar_II = g_bar_II,
        g_EE = g_EE, g_EI = g_EI, g_IE = g_IE, g_II = g_II
        )
    return p_net  

generate_phase_params = generate_params_phase_try
file_name = 'try'
trial_num = 21
changed_parmas = ['d_II', 'g_bar_EE']
changed_parmas_latex = [r'$d_{II}$', r'$\bar{g}_{EE}$']
showed_ticks = [0, (trial_num-1)//2, trial_num-1]
ylabels = [(generate_phase_params(ticks,0))._asdict()[changed_parmas[0]] for ticks in showed_ticks]
xlabels = [(generate_phase_params(0,ticks))._asdict()[changed_parmas[1]] for ticks in showed_ticks]

phase_diagram = np.full((trial_num, trial_num), np.nan)
wavenum_diagram = np.zeros((trial_num, trial_num))
freq_diagram = np.zeros((trial_num, trial_num))
for trail1 in trange(trial_num):
    for trail2 in range(trial_num):
        p_net = generate_params_phase_try(trail1, trail2)
        radius = calc_pred_radius(p_net, dim=2)
        lambda_list_pred_select,label_list_pred_select = calc_pred_outliers(p_net, dim=2)
        real_part_pred_select = np.real(lambda_list_pred_select)
        imag_part_pred_select = np.imag(lambda_list_pred_select)
        if radius >= 1:
            phase_diagram[trail1, trail2] = 0.5
        elif len(lambda_list_pred_select) != 0:
            max_real_index = np.argmax(real_part_pred_select)
            wavenum = label_list_pred_select[max_real_index]
            wavenum_diagram[trail1, trail2] = np.sqrt(wavenum[1]**2 + wavenum[2]**2)
            freq_diagram[trail1, trail2] = np.abs(imag_part_pred_select[max_real_index])/(2*np.pi)
            if (real_part_pred_select[max_real_index] < 1):
                phase_diagram[trail1, trail2] = 1
        else:
            phase_diagram[trail1, trail2] = 1

plt.imshow(wavenum_diagram, origin='lower')
cb = plt.colorbar()
cb.locator = MaxNLocator(nbins=5)
cb.ax.tick_params(labelsize=15)
cb.update_ticks()
norm = mcolors.Normalize(vmin=0, vmax=1)
plt.imshow(phase_diagram, cmap='gray', norm=norm, origin='lower')

plt.ylabel(changed_parmas_latex[0],fontsize=15)
plt.xlabel(changed_parmas_latex[1],fontsize=15)
plt.yticks(ticks=showed_ticks, labels=ylabels,fontsize=15)
plt.xticks(ticks=showed_ticks, labels=xlabels,fontsize=15)

plt.tight_layout()
plt.savefig("./figs/phase_"+file_name+"_wavenum.png")
cb.remove()
plt.close()

plt.imshow(freq_diagram, origin='lower')
cb = plt.colorbar()
cb.locator = MaxNLocator(nbins=5)
cb.ax.tick_params(labelsize=15)
cb.update_ticks()
norm = mcolors.Normalize(vmin=0, vmax=1)
plt.imshow(phase_diagram, cmap='gray', norm=norm, origin='lower')

plt.ylabel(changed_parmas_latex[0],fontsize=15)
plt.xlabel(changed_parmas_latex[1],fontsize=15)
plt.yticks(ticks=showed_ticks, labels=ylabels,fontsize=15)
plt.xticks(ticks=showed_ticks, labels=xlabels,fontsize=15)

plt.tight_layout()
plt.savefig("./figs/phase_"+file_name+"_freq.png")
cb.remove()
plt.close()


phase_diagram = np.full((trial_num, trial_num), np.nan)
wavenum_diagram = np.zeros((trial_num, trial_num))
freq_diagram = np.zeros((trial_num, trial_num))
for trail1 in trange(trial_num):
    for trail2 in range(trial_num):
        p_net = generate_params_phase_try(trail1, trail2)
        radius = calc_pred_radius(p_net, dim=2)
        if radius >= 1:
            phase_diagram[trail1, trail2] = 0.5
        else:
            max_lambda, wavenum = calc_max_theoried_lambda(p_net, dim=2)
            wavenum_diagram[trail1, trail2] = np.sqrt(wavenum[1]**2 + wavenum[2]**2)
            freq_diagram[trail1, trail2] = np.abs(np.imag(max_lambda))/(2*np.pi)
            if (np.real(max_lambda) < 1):
                phase_diagram[trail1, trail2] = 1

plt.imshow(wavenum_diagram, origin='lower')
cb = plt.colorbar()
cb.locator = MaxNLocator(nbins=5)
cb.ax.tick_params(labelsize=15)
cb.update_ticks()
norm = mcolors.Normalize(vmin=0, vmax=1)
plt.imshow(phase_diagram, cmap='gray', norm=norm, origin='lower')

plt.ylabel(changed_parmas_latex[0],fontsize=15)
plt.xlabel(changed_parmas_latex[1],fontsize=15)
plt.yticks(ticks=showed_ticks, labels=ylabels,fontsize=15)
plt.xticks(ticks=showed_ticks, labels=xlabels,fontsize=15)

plt.tight_layout()
plt.savefig("./figs/phase_"+file_name+"_conn_wavenum.png")
cb.remove()
plt.close()

plt.imshow(freq_diagram, origin='lower')
cb = plt.colorbar()
cb.locator = MaxNLocator(nbins=5)
cb.ax.tick_params(labelsize=15)
cb.update_ticks()
norm = mcolors.Normalize(vmin=0, vmax=1)
plt.imshow(phase_diagram, cmap='gray', norm=norm, origin='lower')

plt.ylabel(changed_parmas_latex[0],fontsize=15)
plt.xlabel(changed_parmas_latex[1],fontsize=15)
plt.yticks(ticks=showed_ticks, labels=ylabels,fontsize=15)
plt.xticks(ticks=showed_ticks, labels=xlabels,fontsize=15)

plt.tight_layout()
plt.savefig("./figs/phase_"+file_name+"_conn_freq.png")
cb.remove()
plt.close()




