import traceback
from multiprocessing import Process, Queue
import os
import sys
import time
import copy
import m3dpg_workers as worker
from enum import Enum
import drjit as dr
import numpy as np
import myplotters as mpl
from m3test_common import *
import mitsuba as mi

class ERenderingMode(Enum):
    EForward = 1
    EBackward = 2

def get_output_dir_for_scene(mode: ERenderingMode, scene_name):
    output_dir_base = 'results/m3dpg_texopt' if mode == ERenderingMode.EBackward else 'results/m3dpg_oneopt'
    return os.path.join(output_dir_base, scene_name)

def get_output_dir_for_group(mode: ERenderingMode, scene_name, group_id):
    return os.path.join(get_output_dir_for_scene(mode, scene_name), group_id)


def get_primal_loss(scene_config: dict, groups_config: dict):
    
    mi.set_variant('llvm_ad_rgb')
    loss = {}
    mode = ERenderingMode.EBackward
    iterations, res, spp = scene_config['iterations'], scene_config['res'], scene_config['gt_spp']
    scene_name = scene_config['name']
    gt_file = worker.get_primal_gt_path(get_output_dir_for_scene(mode, scene_name), res, spp)
    gt = mi.TensorXf(mpl.readexr(gt_file))
    for group_id in groups_config:
        loss[group_id] = np.zeros(shape=iterations, dtype=np.float64)
        results_dir = get_output_dir_for_group(mode, scene_name, group_id)
        for i in range(iterations):
            result = mpl.readexr(os.path.join(results_dir, f'primal_{i}.exr'))
            clamp_value = 10
            loss[group_id][i] = mse(dr.minimum(clamp_value, mi.TensorXf(result)), dr.minimum(clamp_value, gt))[0]
    return loss


def get_param_loss(scene_config: dict, groups_config: dict):
    
    mi.set_variant('llvm_ad_rgb')
    loss = {}
    mode = ERenderingMode.EBackward
    iterations = scene_config['iterations']
    scene_name, key = scene_config['name'], scene_config['key']
    if '_extra' in scene_config:
        scene = mi.load_file(scene_config['path'], **scene_config['_extra']) 
    else:
        scene = mi.load_file(scene_config['path'])
    param = mi.traverse(scene)
    gt = param[key]
    for group_id in groups_config:
        loss[group_id] = np.zeros(shape=iterations, dtype=np.float64)
        results_dir = get_output_dir_for_group(mode, scene_name, group_id)
        for i in range(iterations):
            result = mpl.readexr(os.path.join(results_dir, f'param_{i}.exr'))
            if 'normal' in key:
                loss[group_id][i] = mse(decode(gt), decode(mi.TensorXf(result)))[0]
            else:
                loss[group_id][i] = mse(gt, mi.TensorXf(result))[0]
    return loss



def resolve_cmd(queue: Queue, mode: ERenderingMode, group_id, config: dict):
    def resolve_cmd_impl(queue: Queue, mode: ERenderingMode, group_id, config: dict):
        assert type(config) == dict
        mini_requirements = ['integrator', 'path', 'name', 'iterations', 'res', 'spp_per_pass']
        for req in mini_requirements:
            assert req in config
        assert 'key' in config or 'mesh_key' in config
        
        

        integrator = config['integrator']
        scene_path, scene_name, param_key = config['path'], config['name'], (config['key'] if 'key' in config else None)
        res, spp_per_pass = config['res'], config['spp_per_pass']
        spp_per_pass_ad = config['spp_per_pass_ad'] if 'spp_per_pass_ad' in config else spp_per_pass
        iterations = config['iterations']

        output_dir = get_output_dir_for_group(mode, scene_name, group_id)
        os.makedirs(output_dir, exist_ok=True)

        
        if 'mesh_key' in config:
            assert 'mesh_transform' in config
            transform_config = config['mesh_transform']
            if 'rotate_axis' in transform_config and 'rotate_angle' in transform_config:
                r = transform_config['rotate_axis']
                assert type(r) == list and len(r) == 3
                r = np.array(r, dtype=np.float64)
                assert np.sum(np.abs(r)) > 0
                r /= np.linalg.norm(r)

                assert type(transform_config['rotate_angle']) == int or type(transform_config['rotate_angle']) == float
                transform_config['rotate_angle'] = float(transform_config['rotate_angle'])
                assert transform_config['rotate_angle'] != float(0)
            if 'translate' in transform_config:
                t = transform_config['translate']
                assert type(t) == list and len(t) == 3
                t = np.array(t, dtype=np.float64)
                assert np.sum(np.abs(t)) > 0

            assert integrator != 'srb' 
            assert 'reparam' in integrator or integrator == 'prb_basic_mod' or integrator == 'fd'
            

        dr.set_flag(dr.JitFlag.KernelHistory, 1)
        if mode == ERenderingMode.EForward:
            if integrator == 'prb_basic_mod':
                config['target_fuse'] = 4
                worker.forward_ours(queue, output_dir, scene_name, scene_path, param_key, integrator, iterations, res, spp_per_pass, spp_per_pass_ad, config)
            elif integrator == 'srb':
                config['target_fuse'] = 5
                worker.forward_ours(queue, output_dir, scene_name, scene_path, param_key, integrator, iterations, res, spp_per_pass, spp_per_pass_ad, config)
            else:
                worker.forward_mts(queue, output_dir, scene_name, scene_path, param_key, integrator, iterations, res, spp_per_pass, spp_per_pass_ad, config)
        else:
            assert 'gt_spp' in config
            worker.primal_gt(queue, get_output_dir_for_scene(mode, scene_name), scene_path, config)

            if integrator == 'prb_basic_mod':
                config['target_fuse'] = 4
                worker.backward_ours(queue, output_dir, scene_name, scene_path, param_key, integrator, iterations, res, spp_per_pass, spp_per_pass_ad, config)
            elif integrator == 'srb':
                config['target_fuse'] = 5
                worker.backward_ours(queue, output_dir, scene_name, scene_path, param_key, integrator, iterations, res, spp_per_pass, spp_per_pass_ad, config)
            else:
                worker.backward_mts(queue, output_dir, scene_name, scene_path, param_key, integrator, iterations, res, spp_per_pass, spp_per_pass_ad, config)

    if queue is not None: 
        try: 
            resolve_cmd_impl(queue, mode, group_id, config)
        except:
            traceback.print_exc()
            queue.put('?')
        finally:
            queue.put('DONE')
    else:
        try: 
            resolve_cmd_impl(queue, mode, group_id, config)
        except:
            traceback.print_exc()

class RedirectWrapper:
    
    def __init__(self, filename, target_io):
        self.prev_fd = None
        self.prev_io = None
        self.file = open(filename, 'w')
        self.target_io = target_io
        self.is_canceled = False

    def __enter__(self):
        self.prev_io = self.target_io
        self.prev_fd = os.dup(self.target_io.fileno())
        os.close(self.target_io.fileno())
        os.dup2(self.file.fileno(), self.target_io.fileno())
        self.target_io = self.file
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if not self.is_canceled:
            os.dup2(self.prev_fd, self.prev_io.fileno())
            self.target_io = self.prev_io 
            os.close(self.prev_fd)
    
    def cancel(self):
        os.dup2(self.prev_fd, self.prev_io.fileno())
        self.target_io = self.prev_io 
        os.close(self.prev_fd)
        self.is_canceled = True
        

def dispatch(mode: ERenderingMode, group_id, config: dict, prefer_console: bool):
    print(LOGI(), 'start run {}'.format(config['name'] + '_' + group_id), LEND)
    if prefer_console:
        resolve_cmd(None, mode, group_id, config)
    else:
        ret = None
        with RedirectWrapper('log_out_{}.txt'.format(config['name'] + '_' + group_id), sys.stdout) as f,\
            RedirectWrapper('log_err_{}.txt'.format(config['name'] + '_' + group_id), sys.stderr):
            queue = Queue()
            process = Process(target=resolve_cmd, args=(queue, mode, group_id, config))
            process.daemon = True
            process.start()
            f.cancel()
            while True:
                try:
                    msg = queue.get(timeout=60)  
                except:
                    print('wait too long')
                    process.kill()
                    break
                if msg == '?':
                    print(f'\033[31m ERROR in {group_id} !\033[0m')
                    
                    break
                elif msg == 'DONE':
                    break
                print(msg)
            process.join(timeout=60) 
            ret = process.exitcode
        assert ret is not None
        if ret != 0:
            print(f'\033[31m ERROR in {group_id} !\033[0m')
    print('finish run {}'.format(config['name'] + '_' + group_id))

def dispatch_forward(group_id, config: dict, prefer_console: bool = False, prefer_monitor: bool = False):
    config['_prefer_monitor'] = prefer_monitor
    dispatch(ERenderingMode.EForward, group_id, config, prefer_console)

def dispatch_backward(group_id, config: dict, prefer_console: bool = False, prefer_monitor: bool = False):
    config['_prefer_monitor'] = prefer_monitor
    dispatch(ERenderingMode.EBackward, group_id, config, prefer_console)

def gen_default_config(scene_config, group_key, group_value, **kwargs):
    config =  copy.deepcopy(scene_config)
    for k, v in group_base.items():
        config[k] = v
    for k, v in group_value.items():
        config[k] = v
    config['_id'] = group_key
    config['_extra'] = {}
    if len(kwargs) > 0:
        assert len(kwargs) == 1 
        k, v = next(iter(kwargs.items()))
        name = os.path.splitext(os.path.basename(v))[0]
        config['name'] += '-' + name 
        config['_extra'][k] = v

    return config

group_base = {
    "target_super": 1, 
    "target_fuse": 4, 
                                                                
                                                                
                                                                
    

    
    
    
    
    "bsdf_roughness_threshold" : 0.2,
    "guiding_frac" : 0.75,

    "enable_product_sampling" : False,
    "guiding_mis_weight_ad": 0.5,   
    "enable_prb_pro" : True,  
    "enable_srb_pro" : True,  

    "enable_positivization" : False, 
    
    "change_seed" : False,
    "enable_ssguiding" : False,

    "init_rr_prob": 0.5,
    "uniform_rr_prob": 0.5
}





if __name__ == '__main__':  

    
    scene_config = { 
      'name': 'test',
      'path': '../scenes/optcaustic/new.xml',
      'key': 'target.normalmap.data',
      'iterations': 3,  
      'learning_rate': 0.001,
      'momentum': 0.0,
      'target_bounce': -1,  
      'decay_rate': 0.9,  
      'cv_alpha': 0.0,  
      'fill_value': 0.5,
      'spp_per_pass': 4,
      'spp_per_pass_ad': 4,
      'gt_spp': 512,
      'res': 128,
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    groups = {}
    
    groups["baseline"] = {'integrator': 'prb_basic', 'spp_per_pass' : 16}
    groups["our_prb"] = {'integrator': 'prb_basic_mod'}
    

    
    for file in os.listdir(os.path.join(os.path.dirname(scene_config['path']), 'textures/')):
        if not file.endswith('.exr'):
            continue
        extra_params = {}
        extra_params['normalmap_file'] = 'textures/' + file 
        

            
        for group_key, group_value in groups.items():
            config = gen_default_config(scene_config, group_key, group_value, **extra_params) 

            t_start = time.time()
            
            dispatch_backward(group_key, config, True, True)
            t_delta = time.time() - t_start

            print('duration {}: {:.2f}s'.format(group_key, t_delta))
