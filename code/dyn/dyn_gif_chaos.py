import sys
sys.path.append("./code/")
from spatial_ultis import *
from dyn_ultis import *
from dyn_params import *

#product_dyn_simul_gif('chaos',generate_params_dyn_chaos,dim=2)
product_dyn_simul_gif('2dchaos',generate_params_dyn_chaos_new,dim=2)