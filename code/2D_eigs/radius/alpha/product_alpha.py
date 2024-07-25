import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
import sys
sys.path.append("./exp/code/")
from spatial_ultis import *

rescale = 1
N_E, N_I = 22500, 5625
d_EE, d_IE, d_EI, d_II = 0.2, 0.2, 0.2, 0.2
alpha = 0.5
j_EE, j_IE, j_EI, j_II = 2, 3, -12, -18
conn_NEE, conn_NIE, conn_NEI, conn_NII = 2*np.pi*(d_EE**2)*N_E*alpha,2*np.pi*(d_IE**2)*N_I*alpha,2*np.pi*(d_EI**2)*N_E*alpha,2*np.pi*(d_II**2)*N_I*alpha
J_EE, J_IE, J_EI, J_II = j_EE/(conn_NEE*rescale), j_IE/(conn_NIE*rescale), j_EI/(conn_NEI*rescale), j_II/(conn_NII*rescale)
sigma_EE, sigma_IE, sigma_EI, sigma_II = J_EE, J_IE, J_EI, J_II
alpha_list = [0.2,0.3,0.4,0.5,0.6,0.7]

def wrapped_Guassian(x, sigma):
    num = int(np.floor(5 * np.max(sigma))) + 1
    sum = 0
    for n in range(-num, num + 1):
        sum += np.exp(-(x+n)**2/(2 * sigma**2))/(np.sqrt(2 * np.pi) * sigma)
    return sum

repeat_num = 10
for num in range(repeat_num):
    for alpha_n in range(len(alpha_list)):
        alpha = alpha_list[alpha_n]
        conn_NEE, conn_NIE, conn_NEI, conn_NII = 2*np.sqrt(np.pi)*d_EE*N_E*alpha,2*np.sqrt(np.pi)*d_IE*N_I*alpha,2*np.sqrt(np.pi)*d_EI*N_E*alpha,2*np.sqrt(np.pi)*d_II*N_I*alpha
        J_EE, J_IE, J_EI, J_II = j_EE/(conn_NEE*rescale), j_IE/(conn_NIE*rescale), j_EI/(conn_NEI*rescale), j_II/(conn_NII*rescale)
        
        params = Network_Params(N_E = N_E, N_I = N_I,
            N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
            d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
            g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
            g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
            )

        dist_list = calc_dist(params, dim = 2)

        J = generate_net(params, dist_list) 

        J_spa = spa.csr_matrix(J)
        eigs, eig_V = spalin.eigs(J_spa,eig_num)
        np.save(r'./data/' + 'alpha' + str(num) + str(alpha_n) + 'eig.npy', eigs)
