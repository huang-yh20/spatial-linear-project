import sys
sys.path.append("./code/")
from spatial_ultis import *
from dyn_ultis import *
from dyn_params import *

#product_dyn_simul_gif('global',generate_params_dyn_global,dim=2)
product_dyn_simul_gif('2dglobal',generate_params_dyn_global_new,dim=2)