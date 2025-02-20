import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import scipy.sparse as spa
import scipy.sparse.linalg as spalin

from collections import namedtuple

#定义一些参数
eig_num = 500
#定义网络所需要的参数
Network_Params = namedtuple('Network_Params', ["N_E", "N_I",
    "d_EE", "d_IE", "d_EI", "d_II",
    "N_EE", "N_IE", "N_EI", "N_II",
    "g_bar_EE", "g_bar_IE", "g_bar_EI", "g_bar_II",
    "g_EE", "g_IE", "g_EI", "g_II"])

def generate_params_default(trial:int):
    rescale = 830
    rescale_list = list(rescale * [1/0.5, 1/0.75, 1, 1/1.25, 1/1.5])
    N_E, N_I = 22500, 5625
    conn_NEE, conn_NIE, conn_NEI, conn_NII = 800, 200, 800, 200
    J_EE, J_IE, J_EI, J_II = 5/rescale, 9/rescale, -9/rescale, -14/rescale
    sigma_EE, sigma_IE, sigma_EI, sigma_II = 5/rescale, 9/rescale, 9/rescale, 14/rescale
    d_EE, d_IE, d_EI, d_II = 0.1, 0.1, 0.1, 0.16
    
    rescale_list = list(rescale * np.array([1/0.5, 1/0.75, 1, 1/1.25, 1/1.5]))
    rescale = rescale_list[trial]
    J_EE, J_IE, J_EI, J_II = 5/rescale, 9/rescale, -9/rescale, -14/rescale

    p_net = Network_Params(N_E = N_E, N_I = N_I,
        N_EE = conn_NEE, N_IE = conn_NIE, N_EI = conn_NEI, N_II = conn_NII,
        d_EE = d_EE, d_IE = d_IE, d_EI = d_EI, d_II = d_II,
        g_bar_EE = J_EE * conn_NEE, g_bar_EI = J_EI * conn_NEI, g_bar_IE = J_IE * conn_NIE, g_bar_II = J_II * conn_NII,
        g_EE = sigma_EE * np.sqrt(conn_NEE), g_EI = sigma_EI * np.sqrt(conn_NEI), g_IE = sigma_IE * np.sqrt(conn_NIE), g_II = sigma_II * np.sqrt(conn_NII)
        )
    return p_net

def wrapped_Guassian(x, sigma):
    num = int(np.floor(5 * np.max(sigma))) + 1
    sum = 0
    for n in range(-num, num + 1):
        sum += np.exp(-(x+n)**2/(2 * sigma**2))/(np.sqrt(2 * np.pi) * sigma)
    return sum

#计算不同神经元之间的距离
def calc_dist(p:Network_Params, dim = 1):
    if dim == 1:
        loc_E = np.linspace(0, 1, p.N_E)
        loc_I = np.linspace(0, 1, p.N_I)
        loc = np.hstack((loc_E, loc_I))
        dist = np.abs(loc.reshape((p.N_E + p.N_I, 1)) - loc.reshape((1, p.N_E + p.N_I)))
        return (dist,)
    elif dim == 2:
        loc_E_x = (np.ones((int(np.ceil(np.sqrt(p.N_E))),1)).dot(np.linspace(0, 1, int(np.ceil(np.sqrt(p.N_E)))).reshape((1,-1)))).reshape(-1)
        loc_E_y = (np.ones((int(np.ceil(np.sqrt(p.N_E))),1)).dot(np.linspace(0, 1, int(np.ceil(np.sqrt(p.N_E)))).reshape((1,-1)))).T.reshape(-1)
        loc_I_x = (np.ones((int(np.ceil(np.sqrt(p.N_I))),1)).dot(np.linspace(0, 1, int(np.ceil(np.sqrt(p.N_I)))).reshape((1,-1)))).reshape(-1)
        loc_I_y = (np.ones((int(np.ceil(np.sqrt(p.N_I))),1)).dot(np.linspace(0, 1, int(np.ceil(np.sqrt(p.N_I)))).reshape((1,-1)))).T.reshape(-1)
        loc_x = np.hstack((loc_E_x[0:p.N_E], loc_I_x[0:p.N_I])) 
        loc_y = np.hstack((loc_E_y[0:p.N_E], loc_I_y[0:p.N_I])) 
        dist_x = loc_x.reshape((p.N_E + p.N_I, 1)) - loc_x.reshape((1, p.N_E + p.N_I))
        dist_y = loc_y.reshape((p.N_E + p.N_I, 1)) - loc_y.reshape((1, p.N_E + p.N_I))
        return (dist_x,dist_y)
    else: 
        print('dimension error!')

#产生神经元之间的连接矩阵
def generate_net(p:Network_Params, dist_list):
    #读取神经元之间的距离
    d_list = np.block([[np.ones((p.N_E, p.N_E)) * p.d_EE, np.ones((p.N_E, p.N_I)) * p.d_EI],
                [np.ones((p.N_I, p.N_E)) * p.d_IE, np.ones((p.N_I, p.N_I)) * p.d_II]]) 
    dist_coef = np.ones((p.N_E+p.N_I,p.N_E+p.N_I))
    for dist in dist_list:
        dist_coef *= wrapped_Guassian(dist, d_list)
    
    conn_prob = np.block([[np.ones(shape=(p.N_E, p.N_E))*p.N_EE/p.N_E, np.ones(shape=(p.N_E, p.N_I))*p.N_EI/p.N_E],
            [np.ones(shape=(p.N_I, p.N_E))*p.N_IE/p.N_I, np.ones(shape=(p.N_I, p.N_I))*p.N_II/p.N_I]])*dist_coef
    conn_syna = np.random.binomial(1, conn_prob, size=(p.N_E+p.N_I, p.N_E+p.N_I))

    J_EE, J_EI, J_IE, J_II = p.g_bar_EE/p.N_EE, p.g_bar_EI/p.N_EI, p.g_bar_IE/p.N_IE, p.g_bar_II/p.N_II
    sigma_EE, sigma_EI, sigma_IE, sigma_II = p.g_EE/np.sqrt(p.N_EE), p.g_EI/np.sqrt(p.N_EI), p.g_IE/np.sqrt(p.N_IE), p.g_II/np.sqrt(p.N_II)
    J_mean = np.block([[np.ones(shape=(p.N_E, p.N_E))*J_EE, np.ones(shape=(p.N_E, p.N_I))*J_EI],
            [np.ones(shape=(p.N_I, p.N_E))*J_IE, np.ones(shape=(p.N_I, p.N_I))*J_II]])
    J_var = np.block([[np.random.randn(p.N_E, p.N_E)*sigma_EE, np.random.randn(p.N_E, p.N_I)*sigma_EI],
            [np.random.randn(p.N_I, p.N_E)*sigma_IE, np.random.randn(p.N_I, p.N_I)*sigma_II]])
    J = (J_mean + J_var) * conn_syna 
    return J

#产生神经场论的连接矩阵
def generate_field(p:Network_Params, dist_list):
    #读取神经元之间的距离
    d_list = np.block([[np.ones((p.N_E, p.N_E)) * p.d_EE, np.ones((p.N_E, p.N_I)) * p.d_EI],
                [np.ones((p.N_I, p.N_E)) * p.d_IE, np.ones((p.N_I, p.N_I)) * p.d_II]]) 
    dist_coef = np.ones((p.N_E+p.N_I,p.N_E+p.N_I))
    for dist in dist_list:
        dist_coef *= wrapped_Guassian(dist, d_list)
    conn_prob = np.block([[np.ones(shape=(p.N_E, p.N_E))*p.N_EE/p.N_E, np.ones(shape=(p.N_E, p.N_I))*p.N_EI/p.N_E],
            [np.ones(shape=(p.N_I, p.N_E))*p.N_IE/p.N_I, np.ones(shape=(p.N_I, p.N_I))*p.N_II/p.N_I]])*dist_coef
    J_EE, J_EI, J_IE, J_II = p.g_bar_EE/p.N_EE, p.g_bar_EI/p.N_EI, p.g_bar_IE/p.N_IE, p.g_bar_II/p.N_II
    J_mean = np.block([[np.ones(shape=(p.N_E, p.N_E))*J_EE, np.ones(shape=(p.N_E, p.N_I))*J_EI],
            [np.ones(shape=(p.N_I, p.N_E))*J_IE, np.ones(shape=(p.N_I, p.N_I))*J_II]])
    J = J_mean * conn_prob
    return J

#直接产生稀疏矩阵，目的是减少产生过程中的内存占用
def generate_net_sparse(p:Network_Params, dim=2, homo_fix_point = False):
    def calc_dist_one_neuron(neuron_index:int , p:Network_Params, dim = 2):
        if dim == 1:
            loc_E = np.linspace(0, 1, p.N_E)
            loc_I = np.linspace(0, 1, p.N_I)
            loc = np.hstack((loc_E, loc_I))
            dist = np.abs(loc[neuron_index] - loc)
            return (dist,)
        elif dim == 2:
            loc_E_x = (np.ones((int(np.ceil(np.sqrt(p.N_E))),1)).dot(np.linspace(0, 1, int(np.ceil(np.sqrt(p.N_E)))).reshape((1,-1)))).reshape(-1)
            loc_E_y = (np.ones((int(np.ceil(np.sqrt(p.N_E))),1)).dot(np.linspace(0, 1, int(np.ceil(np.sqrt(p.N_E)))).reshape((1,-1)))).T.reshape(-1)
            loc_I_x = (np.ones((int(np.ceil(np.sqrt(p.N_I))),1)).dot(np.linspace(0, 1, int(np.ceil(np.sqrt(p.N_I)))).reshape((1,-1)))).reshape(-1)
            loc_I_y = (np.ones((int(np.ceil(np.sqrt(p.N_I))),1)).dot(np.linspace(0, 1, int(np.ceil(np.sqrt(p.N_I)))).reshape((1,-1)))).T.reshape(-1)
            loc_x = np.hstack((loc_E_x[0:p.N_E], loc_I_x[0:p.N_I])) 
            loc_y = np.hstack((loc_E_y[0:p.N_E], loc_I_y[0:p.N_I])) 
            dist_x = np.abs(loc_x[neuron_index] - loc_x)
            dist_y = np.abs(loc_y[neuron_index] - loc_y)
            return (dist_x,dist_y)
        else: 
            print('dimension error!')
        
    J_EE, J_EI, J_IE, J_II = p.g_bar_EE/p.N_EE, p.g_bar_EI/p.N_EI, p.g_bar_IE/p.N_IE, p.g_bar_II/p.N_II
    sigma_EE, sigma_EI, sigma_IE, sigma_II = p.g_EE/np.sqrt(p.N_EE), p.g_EI/np.sqrt(p.N_EI), p.g_IE/np.sqrt(p.N_IE), p.g_II/np.sqrt(p.N_II)

    J_coo_row_list, J_coo_col_list, J_coo_data_list = [], [], []

    if not homo_fix_point:
        #一列一列地构建这个稀疏矩阵，neuron_index表示应当是哪一列
        d_list = np.block([np.ones(p.N_E) * p.d_EE, np.ones(p.N_I) * p.d_IE])
        for neuron_index in range(p.N_E):
            dist_one_neuron = calc_dist_one_neuron(neuron_index, p, dim=dim)
            dist_coef = np.ones(p.N_E+p.N_I)
            for dist in dist_one_neuron:
                dist_coef *= wrapped_Guassian(dist, d_list)
            conn_prob = np.block([np.ones(shape=(p.N_E,))*p.N_EE/p.N_E, np.ones(shape=(p.N_I,))*p.N_IE/p.N_I]) * dist_coef
            conn_syna = np.random.binomial(1, conn_prob, size=(p.N_E+p.N_I,))

            J_coo_row_list += list(np.where(conn_syna > 0)[0])
            J_coo_col_list += [neuron_index] * np.sum(conn_syna)
            J_coo_data_list += list(np.ones(shape=(np.sum(conn_syna[0:p.N_E]),))*J_EE + np.random.randn(np.sum(conn_syna[0:p.N_E]))*sigma_EE)
            J_coo_data_list += list(np.ones(shape=(np.sum(conn_syna[p.N_E:p.N_E+p.N_I]),))*J_IE + np.random.randn(np.sum(conn_syna[p.N_E:p.N_E+p.N_I]))*sigma_IE)
        
        d_list = np.block([np.ones(p.N_E) * p.d_EI, np.ones(p.N_I) * p.d_II])
        for neuron_index in range(p.N_E, p.N_E + p.N_I):
            dist_one_neuron = calc_dist_one_neuron(neuron_index, p, dim=dim)
            dist_coef = np.ones(p.N_E+p.N_I)
            for dist in dist_one_neuron:
                dist_coef *= wrapped_Guassian(dist, d_list)
            conn_prob = np.block([np.ones(shape=(p.N_E,))*p.N_EI/p.N_E, np.ones(shape=(p.N_I,))*p.N_II/p.N_I]) * dist_coef
            conn_syna = np.random.binomial(1, conn_prob, size=(p.N_E+p.N_I,))

            J_coo_row_list += list(np.where(conn_syna > 0)[0])
            J_coo_col_list += [neuron_index] * np.sum(conn_syna)
            J_coo_data_list += list(np.ones(shape=(np.sum(conn_syna[0:p.N_E]),))*J_EI + np.random.randn(np.sum(conn_syna[0:p.N_E]))*sigma_EI)
            J_coo_data_list += list(np.ones(shape=(np.sum(conn_syna[p.N_E:p.N_E+p.N_I]),))*J_II + np.random.randn(np.sum(conn_syna[p.N_E:p.N_E+p.N_I]))*sigma_II)
        
    else:
        #一列一列地构建这个稀疏矩阵，neuron_index表示应当是哪一列
        d_list = np.block([np.ones(p.N_E) * p.d_EE, np.ones(p.N_I) * p.d_EI])
        for neuron_index in range(p.N_E):
            dist_one_neuron = calc_dist_one_neuron(neuron_index, p, dim=dim)
            dist_coef = np.ones(p.N_E+p.N_I)
            for dist in dist_one_neuron:
                dist_coef *= wrapped_Guassian(dist, d_list)

            conn_prob = dist_coef[0:p.N_E]
            conn_prob = conn_prob/np.sum(conn_prob)
            conn_syna_EE = np.random.choice(list(np.arange(0, p.N_E)), int(p.N_EE), replace=False, p=conn_prob)
            J_coo_row_list += [neuron_index] * len(list(conn_syna_EE))
            J_coo_col_list += list(conn_syna_EE)
            data_syna_EE = J_EE + sigma_EE * np.random.randn(len(list(conn_syna_EE)))
            data_syna_EE = data_syna_EE - (np.mean(data_syna_EE) - J_EE)
            J_coo_data_list += list(data_syna_EE)

            conn_prob = dist_coef[p.N_E:p.N_E+p.N_I]
            conn_prob = conn_prob/np.sum(conn_prob)
            conn_syna_EI = np.random.choice(list(np.arange(p.N_E, p.N_E+p.N_I)), int(p.N_EI*p.N_I/p.N_E), replace=False, p=conn_prob)
            J_coo_row_list += [neuron_index] * len(list(conn_syna_EI))
            J_coo_col_list += list(conn_syna_EI)
            data_syna_EI = J_EI + sigma_EI * np.random.randn(len(list(conn_syna_EI)))
            data_syna_EI = data_syna_EI - (np.mean(data_syna_EI) - J_EI)
            J_coo_data_list += list(data_syna_EI)

        d_list = np.block([np.ones(p.N_E) * p.d_IE, np.ones(p.N_I) * p.d_II])
        for neuron_index in range(p.N_E, p.N_E + p.N_I):
            dist_one_neuron = calc_dist_one_neuron(neuron_index, p, dim=dim)
            dist_coef = np.ones(p.N_E+p.N_I)
            for dist in dist_one_neuron:
                dist_coef *= wrapped_Guassian(dist, d_list)

            conn_prob = dist_coef[0:p.N_E]
            conn_prob = conn_prob/np.sum(conn_prob)
            conn_syna_IE = np.random.choice(list(np.arange(0, p.N_E)), int(p.N_IE*p.N_E/p.N_I), replace=False, p=conn_prob)
            J_coo_row_list += [neuron_index] * len(list(conn_syna_IE))
            J_coo_col_list += list(conn_syna_IE)
            data_syna_IE = J_IE + sigma_IE * np.random.randn(len(list(conn_syna_IE)))
            data_syna_IE = data_syna_IE - (np.mean(data_syna_IE) - J_IE)
            J_coo_data_list += list(data_syna_IE)

            conn_prob = dist_coef[p.N_E:p.N_E+p.N_I]
            conn_prob = conn_prob/np.sum(conn_prob)
            conn_syna_II = np.random.choice(list(np.arange(p.N_E, p.N_E+p.N_I)), int(p.N_II), replace=False, p=conn_prob)
            J_coo_row_list += [neuron_index] * len(list(conn_syna_II))
            J_coo_col_list += list(conn_syna_II)
            data_syna_II = J_II + sigma_II * np.random.randn(len(list(conn_syna_II)))
            data_syna_II = data_syna_II - (np.mean(data_syna_II) - J_II)
            J_coo_data_list += list(data_syna_II)
        
    J_spa = spa.coo_matrix((J_coo_data_list, (J_coo_row_list, J_coo_col_list)), shape=(p.N_E+p.N_I, p.N_E+p.N_I))
    return J_spa
            
#理论预测圆形部分半径
def calc_pred_radius(p:Network_Params, dim = 1):
    J_EE, J_EI, J_IE, J_II = p.g_bar_EE/p.N_EE, p.g_bar_EI/p.N_EI, p.g_bar_IE/p.N_IE, p.g_bar_II/p.N_II
    sigma_EE, sigma_EI, sigma_IE, sigma_II = p.g_EE/np.sqrt(p.N_EE), p.g_EI/np.sqrt(p.N_EI), p.g_IE/np.sqrt(p.N_IE), p.g_II/np.sqrt(p.N_II)
    
    if dim == 1:
        sigma_eff_EE = p.N_EE * np.sqrt(p.N_E/p.N_E) * ((1 - p.N_EE/(2 * np.sqrt(np.pi)*p.d_EE*(p.N_E))) * J_EE **2 + sigma_EE**2)
        sigma_eff_IE = p.N_IE * np.sqrt(p.N_E/p.N_I) * ((1 - p.N_IE/(2 * np.sqrt(np.pi)*p.d_IE*(p.N_I))) * J_IE **2 + sigma_IE**2)
        sigma_eff_EI = p.N_EI * np.sqrt(p.N_I/p.N_E) * ((1 - p.N_EI/(2 * np.sqrt(np.pi)*p.d_EI*(p.N_E))) * J_EI **2 + sigma_EI**2)
        sigma_eff_II = p.N_II * np.sqrt(p.N_I/p.N_I) * ((1 - p.N_II/(2 * np.sqrt(np.pi)*p.d_II*(p.N_I))) * J_II **2 + sigma_II**2)
    elif dim == 2:
        sigma_eff_EE = p.N_EE * np.sqrt(p.N_E/p.N_E) * ((1 - p.N_EE/(4 * np.pi * p.d_EE**2 * p.N_E)) * J_EE **2 + sigma_EE**2)
        sigma_eff_IE = p.N_IE * np.sqrt(p.N_E/p.N_I) * ((1 - p.N_IE/(4 * np.pi * p.d_IE**2 * p.N_I)) * J_IE **2 + sigma_IE**2)
        sigma_eff_EI = p.N_EI * np.sqrt(p.N_I/p.N_E) * ((1 - p.N_EI/(4 * np.pi * p.d_EI**2 * p.N_E)) * J_EI **2 + sigma_EI**2)
        sigma_eff_II = p.N_II * np.sqrt(p.N_I/p.N_I) * ((1 - p.N_II/(4 * np.pi * p.d_II**2 * p.N_I)) * J_II **2 + sigma_II**2)
    else:
        print('dimenstion error')
        return None
    sigma_eff = np.array([[sigma_eff_EE, sigma_eff_EI],
                        [sigma_eff_IE, sigma_eff_II]])
    eigs_sigma, eig_V_sigma = np.linalg.eig(sigma_eff)
    real_part_sigma = np.real(eigs_sigma)
    imag_part_sigma = np.imag(eigs_sigma)
    radius = np.sqrt(np.max(real_part_sigma))
    return radius

#理论预测离群点位置
def calc_pred_outliers(p:Network_Params, dim = 1, radius_filter = True):
    radius = calc_pred_radius(p,dim)

    lambda_list_pred = []
    label_list_pred = []

    if dim == 1:
        #n_max = (p.N_E+p.N_I-2)//4
        n_max = 40 #这个是临时的，但是是够用的
        for n in range(n_max):
            eig_num = 1 if n == 0 else 2
            k = 2 * np.pi * n
            g_eff_EE_n = np.exp(-(k*p.d_EE)**2/2) * p.g_bar_EE
            g_eff_IE_n = np.exp(-(k*p.d_IE)**2/2) * p.g_bar_IE
            g_eff_EI_n = np.exp(-(k*p.d_EI)**2/2) * p.g_bar_EI
            g_eff_II_n = np.exp(-(k*p.d_II)**2/2) * p.g_bar_II
            lambda_list_pred += [0.5*(g_eff_EE_n+g_eff_II_n+np.emath.sqrt((g_eff_EE_n-g_eff_II_n)**2+4*g_eff_IE_n*g_eff_EI_n))] * eig_num
            label_list_pred += [(0,n)] * eig_num
            lambda_list_pred += [0.5*(g_eff_EE_n+g_eff_II_n-np.emath.sqrt((g_eff_EE_n-g_eff_II_n)**2+4*g_eff_IE_n*g_eff_EI_n))] * eig_num
            label_list_pred += [(1,n)] * eig_num
    elif dim == 2:
        #n_max = int(np.sqrt(p.N_E+p.N_I)//2)
        n_max = 40 #这个是临时的，但是是够用的
        for n_x in range(n_max):
            for n_y in range(n_max):
                eig_num = 1 if (n_x == 0 and n_y ==0) else 2
                k = 2 * np.pi * np.sqrt(n_x **2 + n_y ** 2)
                g_eff_EE_n = np.exp(-(k*p.d_EE)**2/2) * p.g_bar_EE
                g_eff_IE_n = np.exp(-(k*p.d_IE)**2/2) * p.g_bar_IE
                g_eff_EI_n = np.exp(-(k*p.d_EI)**2/2) * p.g_bar_EI
                g_eff_II_n = np.exp(-(k*p.d_II)**2/2) * p.g_bar_II
                lambda_list_pred += [0.5*(g_eff_EE_n+g_eff_II_n+np.emath.sqrt((g_eff_EE_n-g_eff_II_n)**2+4*g_eff_IE_n*g_eff_EI_n))] * eig_num
                label_list_pred += [(0,n_x,n_y)] * eig_num
                lambda_list_pred += [0.5*(g_eff_EE_n+g_eff_II_n-np.emath.sqrt((g_eff_EE_n-g_eff_II_n)**2+4*g_eff_IE_n*g_eff_EI_n))] * eig_num
                label_list_pred += [(1,n_x,n_y)] * eig_num

    lambda_list_pred_select = []
    label_list_pred_select = []
    for i in range(len(lambda_list_pred)):
        if (np.abs(lambda_list_pred[i]) > radius) or (not radius_filter):
            lambda_list_pred_select.append(lambda_list_pred[i])
            label_list_pred_select.append(label_list_pred[i])
    
    return (lambda_list_pred_select,label_list_pred_select)

#这个函数用于返回最大的理论特征以及它的标签，需要注意这个函数没有过滤掉小于半径的部分
#希望能通过并行的方式大大加快速度
def calc_max_theoried_lambda(p:Network_Params, dim = 1):
    n_max, n_num = 40, 500

    radius = calc_pred_radius(p,dim)

    if dim == 1:
        n_list = np.linspace(0, n_max, n_num)
        k_list = 2 * np.pi * n_list
    elif dim == 2:
        n_list = np.linspace(0, n_max, n_num)
        n_list_x, n_list_y = np.meshgrid(n_list, n_list)
        n_list_abs = np.sqrt(n_list_x ** 2 + n_list_y ** 2)
        k_list = 2 * np.pi * n_list_abs
        
    g_eff_EE_n = np.exp(-(k_list*p.d_EE)**2/2) * p.g_bar_EE
    g_eff_IE_n = np.exp(-(k_list*p.d_IE)**2/2) * p.g_bar_IE
    g_eff_EI_n = np.exp(-(k_list*p.d_EI)**2/2) * p.g_bar_EI
    g_eff_II_n = np.exp(-(k_list*p.d_II)**2/2) * p.g_bar_II
    lambda_list_pred_pos = 0.5*(g_eff_EE_n+g_eff_II_n+np.emath.sqrt((g_eff_EE_n-g_eff_II_n)**2+4*g_eff_IE_n*g_eff_EI_n))
    lambda_list_pred_neg = 0.5*(g_eff_EE_n+g_eff_II_n-np.emath.sqrt((g_eff_EE_n-g_eff_II_n)**2+4*g_eff_IE_n*g_eff_EI_n))
    
    if dim == 1:
        max_lambda_pos, max_label_pos = lambda_list_pred_pos[np.argmax(np.real(max_lambda_pos))], n_list[np.argmax(np.real(max_lambda_pos))]
        max_lambda_neg, max_label_neg = lambda_list_pred_neg[np.argmax(np.real(max_lambda_neg))], n_list[np.argmax(np.real(max_lambda_neg))]
        max_lambda, max_label = (max_lambda_pos, max_label_pos) if (np.real(max_lambda_pos) > np.real(max_lambda_neg)) else (max_lambda_neg, max_label_neg)
    elif dim == 2:
        max_n_x_pos_loc, max_n_y_pos_loc = np.where(np.real(lambda_list_pred_pos) == np.max(np.real(lambda_list_pred_pos)))
        max_n_x_pos, max_n_y_pos = n_list[max_n_x_pos_loc[0]], n_list[max_n_y_pos_loc[0]]
        max_lambda_pos = lambda_list_pred_pos[max_n_x_pos_loc[0], max_n_y_pos_loc[0]]
        max_n_x_neg_loc, max_n_y_neg_loc = np.where(np.real(lambda_list_pred_neg) == np.max(np.real(lambda_list_pred_neg)))
        max_n_x_neg, max_n_y_neg = n_list[max_n_x_neg_loc[0]], n_list[max_n_y_neg_loc[0]]
        max_lambda_neg = lambda_list_pred_neg[max_n_x_neg_loc[0], max_n_y_neg_loc[0]]       
        max_lambda, max_label = (max_lambda_pos, (0, max_n_x_pos, max_n_y_pos)) if (np.real(max_lambda_pos) > np.real(max_lambda_neg)) else (max_lambda_neg, (1, max_n_x_neg, max_n_y_neg))
    return (max_lambda, max_label)
    


#这个函数用于返回圆形部分特征根
def get_eigs_diskpart(eigs:list, lambda_list_pred_select:list, label_list_pred_select:list):
    eigs = list(eigs)
    for lambda_n in range(len(lambda_list_pred_select)):
        eig_num = 1 if all(x == 0 for x in label_list_pred_select[lambda_n][1::]) else 2
        for clear_num in range(eig_num):
            points_dist_list = list(np.abs(np.array(eigs) - lambda_list_pred_select[lambda_n]))
            nearest_point = np.argmin(points_dist_list)
            eigs.pop(nearest_point)
    return np.array(eigs)

#这个函数用于找到距离最近的若干特征根，返回这些点的索引
def find_points(points, target, find_num):
    points_dist_list = list(np.abs(np.array(points)- np.array(target)))
    finded_points_indice = []
    for num in range(find_num):
        finded_points_indice.append(np.argmin(points_dist_list))
        points_dist_list[np.argmin(points_dist_list)] = np.inf
    return finded_points_indice

#计算不同模式的简并重数，以下的函数只是部分正确，需要修改
def degenerate_num(label):
    return 1 if all(x == 0 for x in label[1::]) else 2 * (len(label) - 1)

def temp_plot_pred(p_net:Network_Params, dim=1):
    radius = calc_pred_radius(p_net,dim=dim)
    lambda_list_pred_select, label_list_pred_select  = calc_pred_outliers(p_net, dim=dim)
    real_part_pred_select = np.real(lambda_list_pred_select)
    imag_part_pred_select = np.imag(lambda_list_pred_select)

    x_dots = np.linspace(-radius, radius, 200)
    y_dots = np.sqrt(radius**2 - x_dots**2)

    plt.plot(x_dots, y_dots, c='lightcoral', linewidth=1)
    plt.plot(x_dots, -y_dots, c='lightcoral', linewidth=1)
    plt.scatter(real_part_pred_select, imag_part_pred_select, s=10, c='lightcoral', marker='x')
    plt.axis("equal")

#这是一个临时版本，只能处理恰好是整数的状态
def find_neibour(p_net: Network_Params, loc:int, radius:int = 6, dim:int = 1):
    if dim == 1:
        neuro_num = np.arange(0, p_net.N_E)
        neibour_loc_list = np.arange(loc - radius, loc + radius + 1) % p_net.N_E
    if dim == 2:
        neuro_num = (np.arange(0, p_net.N_E)).reshape(int(np.ceil(np.sqrt(p_net.N_E))), int(np.ceil(np.sqrt(p_net.N_E))))
        loc_x, loc_y = loc % int(np.ceil(np.sqrt(p_net.N_E))), loc // int(np.ceil(np.sqrt(p_net.N_E)))
        neibour_loc_x = np.arange(loc_x - radius, loc_x + radius + 1) % int(np.ceil(np.sqrt(p_net.N_E)))
        neibour_loc_y = np.arange(loc_y - radius, loc_y + radius + 1) % int(np.ceil(np.sqrt(p_net.N_E)))
        neibour_loc_list = neuro_num[np.ix_(neibour_loc_y, neibour_loc_x)].reshape(-1)
    return neibour_loc_list