import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
from scipy.ndimage import convolve
from tqdm import trange
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from typing import NamedTuple, Union, Callable, List
import imageio
import sys
sys.path.append("./code")
from spatial_ultis import *
sys.path.append("./code/dyn/")
from dyn_ultis import *

activation_func = activation_func_tanh

def eigs_axislim_default(eigs:np.ndarray):
    real_part = np.real(eigs)
    imag_part = np.imag(eigs)
    fig_lim = max(np.max(real_part) - np.min(real_part), np.max(imag_part) - np.min(imag_part)) + 0.2
    fig_lim = max(2.2, fig_lim)
    x_center, y_center = 0.5 * (np.max(real_part) + np.min(real_part)), 0.5 * (np.max(imag_part) + np.min(imag_part))
    return ((x_center - 0.5*fig_lim, x_center + 0.5*fig_lim), (y_center - 0.5*fig_lim, y_center + 0.5*fig_lim))


def artfigs_plot_eigs(eigs:np.ndarray, ax = None, eigs_axislim: Callable = eigs_axislim_default):
    xlim, ylim = eigs_axislim(eigs)
    if ax == None:
        ax = plt. gca()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$Re(\\lambda)$", fontsize=15)
        ax.tick_params(axis='x', labelsize=15)  
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4)) 
        ax.set_ylabel("$Im(\\lambda)$", fontsize=15)
        ax.tick_params(axis='y', labelsize=15)  
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4)) 
        ax.set_aspect('equal') 
    else: 
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        #ax.set_xlabel("$Re(\\lambda)$")
        ax.tick_params(axis='x')  
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2)) 
        #ax.set_ylabel("$Im(\\lambda)$")
        ax.tick_params(axis='y')  
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2)) 
        ax.set_aspect('equal') 

    real_part = np.real(eigs)
    imag_part = np.imag(eigs)
    ax.scatter(real_part, imag_part, s=3, c='none', marker='o', edgecolors='k')
    ax.axvline(x=1,c='gray',ls='--')

