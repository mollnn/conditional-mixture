

from m3dpg_run_common import *
from m3dpg_texopt_scenes import *

import copy
import time
import matplotlib.pyplot as plt
import myplotters as mpl
import os
import numpy as np


if __name__ == '__main__':

    scene_idx = 13
    scene_config = scenes[scene_idx]

    groups = {}
    groups["ref"] = {'integrator': 'prb', 'spp_per_pass': 24}
    

    
    for group_key, group_value in groups.items():
        config = gen_default_config(scene_config, group_key, group_value)
        
        
        
        

 
        t_start = time.time()
        dispatch_backward(group_key, config, prefer_console=True, prefer_monitor=False)
        t_delta = time.time() - t_start

        print('duration {}: {:.2f}s'.format(group_key, t_delta))
