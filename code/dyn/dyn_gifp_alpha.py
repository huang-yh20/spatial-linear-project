import sys
sys.path.append("./code/")
from spatial_ultis import *
from dyn_ultis import *
from dyn_params import *

product_dyn_simul_gif('params_alpha',generate_params_dyn_params_alpha,dim=2)