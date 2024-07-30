import sys
sys.path.append("./code/")
sys.path.append("./code/dyn/")
from spatial_ultis import *
from dyn_ultis import *


rescale = 475
N_E, N_I = 4900, 1225
conn_NEE, conn_NIE, conn_NEI, conn_NII = 200, 50, 200, 50
J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -18/rescale, -18/rescale
sigma_EE, sigma_IE, sigma_EI, sigma_II = 0,0,0,0
d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.2, 0.2

rescale_list = list(rescale * np.array([1/0.5, 1/0.75, 1, 1/1.25, 1/1.5]))
rescale = rescale_list[4]
J_EE, J_IE, J_EI, J_II = 4/rescale, 4/rescale, -18/rescale, -18/rescale

p_net = Network_Params(N_E = N_E, N_I = N_I,
    N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
    d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
    g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
    g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
    )
p_simul = Simul_Params(T = 40, t_step=100, record_step=10, activation_func='tanh')

record_x = dyn_simul(p_net, p_simul, dim=2)
product_gif(record_x, p_net, p_simul, "debug", 2)