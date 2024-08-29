import sys
sys.path.append("./code/")
from spatial_ultis import *
from dyn_ultis import *
from dyn_params import *

#product_dyn_simul_gif('wave',generate_params_dyn_wave,dim=2)
product_dyn_simul_gif('2dwave',generate_params_dyn_wave_new,dim=2)