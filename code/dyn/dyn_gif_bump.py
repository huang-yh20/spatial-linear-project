import sys
sys.path.append("./code/")
from spatial_ultis import *
from dyn_ultis import *
from dyn_params import *

#product_dyn_simul_gif('bump',generate_params_dyn_bump,dim=2)
product_dyn_simul_gif('2dbump',generate_params_dyn_bump_new,dim=2)