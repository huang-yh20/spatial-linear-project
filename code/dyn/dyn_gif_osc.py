import sys
sys.path.append("./code/")
from spatial_ultis import *
from dyn_ultis import *
from dyn_params import *

#product_dyn_simul_gif('osc',generate_params_dyn_osc,dim=2)
product_dyn_simul_gif('2dosc',generate_params_dyn_osc_new,dim=2)