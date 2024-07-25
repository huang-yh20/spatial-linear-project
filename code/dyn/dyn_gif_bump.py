import sys
sys.path.append("./exp/code/")
from spatial_ultis import *
from dyn_ultis import *
from dyn_params import *

product_dyn_simul_gif('bump',generate_params_dyn_bump,dim=2)