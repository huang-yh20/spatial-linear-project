import sys
sys.path.append("./exp/code/")
from spatial_ultis import *
from dyn_ultis import *
from dyn_params import *

product_dyn_simul_gif('global',generate_params_dyn_global,dim=2)