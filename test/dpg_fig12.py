

from m3dpg_run_common import *
from m3dpg_texopt_scenes import *

import copy
import time
import matplotlib.pyplot as plt
import myplotters as mpl
import os
import numpy as np


scene_config = {
    'name': 'veach-ajar-test',
    'path': '../scenes/veach-ajar/scene_v3_tex_newpos.xml',
    'key': 'LandscapeBSDF.brdf_0.reflectance.data',
    'iterations': 6,
    'warmup-iterations': 8,
    'learning_rate': 0.01,
    'momentum': 0.0,
    'target_bounce': -1,
    'decay_rate': 0.0,
    'cv_alpha': 0.0,
    'fill_value': 0.5,
    'spp_per_pass': 16,
    'spp_per_pass_ad': 2,
    'gt_spp': 10240,
    'res': 400,
    'fit_iter_max': 64,  
}

if __name__ == '__main__':

    groups = {}

    groups["ref"] = {'integrator': 'prb', 'spp_per_pass': 40}
    groups["p_optimal"] = {'integrator': 'prb_basic_mod', 'enable_prb_pro': True, 'spatialFilter_Primal':'nearest'} 



    for group_key, group_value in groups.items():
        extra_params = {}
        extra_params['tex'] = 'textures/' + 'landscape-with-a-lake-640p.jpg'
        config = gen_default_config(scene_config, group_key, group_value, **extra_params)  

        t_start = time.time()
        
        dispatch_backward(group_key, config, True, False)
        t_delta = time.time() - t_start

        print('duration {}: {:.2f}s'.format(group_key, t_delta))
