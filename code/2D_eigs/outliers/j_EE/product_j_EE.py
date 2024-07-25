import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
import sys
sys.path.append("./exp/code/")
from spatial_ultis import *

rescale = 1250
N_E, N_I = 22500, 5625
conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
J_EE, J_IE, J_EI, J_II = 5/rescale, 9/rescale, -9/rescale, -12/rescale
sigma_EE, sigma_IE, sigma_EI, sigma_II = 10/rescale, 18/rescale, 18/rescale, 24/rescale
d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.2
j_EE_list = [0,1.25,2.5,3.75,5.0,6.25]

def wrapped_Guassian(x, sigma):
    num = int(np.floor(5 * np.max(sigma))) + 1
    sum = 0
    for n in range(-num, num + 1):
        sum += np.exp(-(x+n)**2/(2 * sigma**2))/(np.sqrt(2 * np.pi) * sigma)
    return sum

repeat_num = 10
for num in range(repeat_num):
    for j_n in range(len(j_EE_list)):
        J_EE = j_EE_list[j_n]/rescale
        params = Network_Params(N_E = N_E, N_I = N_I,
            N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
            d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
            g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
            g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
            )

        dist_list = calc_dist(params, dim = 2)

        J = generate_net(params, dist_list)  

        J_spa = spa.csr_matrix(J)
        eigs, eig_V = spalin.eigs(J_spa,k=eig_num)
        np.save(r'./data/' +'j_EE'+ str(num) + str(j_n) + 'eig.npy', eigs)
