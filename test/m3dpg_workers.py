import os
from labcommon import *
from config import * 
import numpy as np
from threading import Thread
import drjit as dr
import mitsuba as mi
from m3test_common import *
import shutil
from multiprocessing import Queue
import json
import hashlib
import sys 
sys.path.append("..") 
from utils.webmonitor import client as monitor_client


LTIME = lambda: time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

LOGD = lambda: LTIME() + " mts3"
LOGI = lambda: LTIME() + " mts3"
LOGT = lambda: LTIME() + " mts3"
LOGK = lambda: LTIME() + " mts3"
LOGW = lambda: LTIME() + " mts3"
LOGE = lambda: LTIME() + " mts3"
LEND = ""




          


















def make_mts1_cmd_str(TESTID, spp_per_pass, bbox, config, single_thread=False):
    test_name = "dummy"
    exr_filename = "results/%s/%s.exr" % (TESTID, test_name)
    cmd = MITSUBA1_EXE + " "
    cmd += "../scenes/ppg_mts1_dummy.xml" + " "
    cmd += "-o %s " % exr_filename
    cmd += "-Dspp=%d " % 999999999
    cmd += "-DbudgetType=%s " % "spp"
    cmd += "-Dsppass=%d " % spp_per_pass
    cmd += "-Dseq=%s " % "uniform"
    cmd += "-Dadt=%d " % config["target_fuse"]
    cmd += "-Dbbox_sx=%f " % (bbox.extents()[0] / 2 * 1.00001)
    cmd += "-Dbbox_sy=%f " % (bbox.extents()[1] / 2 * 1.00001)
    cmd += "-Dbbox_sz=%f " % (bbox.extents()[2] / 2 * 1.00001)
    cmd += "-Dbbox_cx=%f " % bbox.center()[0]
    cmd += "-Dbbox_cy=%f " % bbox.center()[1]
    cmd += "-Dbbox_cz=%f " % bbox.center()[2]
    cmd += "-Ddecay_rate=%f " % config['decay_rate']
    cmd += "-Dguiding_mis_weight_ad=%f " % config["guiding_mis_weight_ad"]
    if 'spatialFilter_Primal' in config:
        cmd += "-DspatialFilter_Primal=%s " % config["spatialFilter_Primal"]
    if 'spatialFilter_Adjoint' in config: 
        cmd += "-DspatialFilter_Adjoint=%s " % config["spatialFilter_Adjoint"]
    if single_thread == True:
        cmd += "-p 1 "
    return cmd

def mts1_block_me():
    
    try_remove("guiding_turn3.txt")
    os.system("echo 123 *> guiding_turn1.txt") 
    print("mts3: alarm and wait mts1")
    print(LOGD(), "con", "mts3 sleep", LEND)
    while True:
        if os.path.exists("guiding_turn3.txt"): 
            break
        time.sleep(0.01)
    print(LOGD(), "con", "mts3 wakeup", LEND)

def mts1_kill_me():
    
    try_remove("guiding_turn3.txt")
    os.system("echo 123 *> guiding_turn1end.txt") 
    print(LOGW(), "con", "kill mts1", LEND)


def notify(queue, msg):
    print(msg)
    if queue is not None:
        queue.put(msg)

def need_multi_pass(res, spp):
    return 2**32//res//res//2 < spp

def do_high_spp_rendering(queue, scene, res, params, spp, msg, forward_key=None):
    print(spp)
    image = None
    image_grad = None
    need_forward = forward_key is not None
    max_spp_per_iter = 2**32//res//res //2
    accum_spp = 0
    num_iter = (spp + max_spp_per_iter - 1) // max_spp_per_iter
    for i in range(num_iter):
        notify(queue, f'{msg} -- pass {i}')
        sppi = min(max_spp_per_iter, spp - accum_spp)
        accum_spp += sppi

        time0 = time.time()
        if params is not None:
            cur = mi.render(scene, params, spp=sppi, seed=i)  * sppi
        else:
            cur = mi.render(scene, spp=sppi, seed=i) * sppi
        notify(queue, f'iter{i} primal use {time.time() - time0:2f}')

        if need_forward:
            time0 = time.time()
            dr.forward(params[forward_key])
            cur_grad = dr.grad(cur) 
            notify(queue, f'iter{i} forward use {time.time() - time0:2f}')

        if i == 0:
            image = cur
            if need_forward:
                image_grad = cur_grad
        else:
            image += cur
            if need_forward:
                image_grad += cur_grad 
    image /= spp
    if need_forward:
        return image, image_grad / spp
    else:
        return image

def save_time(output_dir, times):
    np.savetxt(os.path.join(output_dir, 'times.txt'), times, delimiter=' ', fmt='%.6e') 


def apply_transformation_forward(params, config, initial_vertex_positions, value, with_grad=True):
    if with_grad:
        assert dr.grad_enabled(value)
    trafo = mi.Transform4f() 
    transform_config = config['mesh_transform']
    
    if 'rotate_axis' in transform_config:
        trafo = mi.Transform4f.rotate(axis=transform_config['rotate_axis'], angle=value*np.sign(transform_config['rotate_angle']))
    if 'translate' in transform_config:
        sgn = np.sign(transform_config['translate'])
        trafo = mi.Transform4f.translate([value*sgn[0], value*sgn[1], value*sgn[2]]) @ trafo
    params[config['mesh_key']] = dr.ravel(trafo @ initial_vertex_positions)
    params.update()

def forward_mts(queue: Queue, output_dir, scene_name, scene_path, param_key, integrator, iteratons, res, spp_per_pass, spp_per_pass_ad, config: dict):
    mi.set_variant('cuda_ad_rgb')
    np.random.seed(0)
    print("========================== Priaml")
    key = param_key
    mesh_key = config['mesh_key'] if 'mesh_key' in config else None
    spp = max(spp_per_pass, spp_per_pass_ad) 
    
    
    

    time0 = time.time()
     
    
    if integrator == 'prb' and need_multi_pass(res, spp):
        scene = mi.load_file(scene_path, res=res, integrator='prb', **config['_extra'])
        params = mi.traverse(scene)
        dr.enable_grad(params[key])
        params.update()
        image_ours, grad_image_ours = do_high_spp_rendering(queue, scene, res, params, spp, 'forward', key)
    elif integrator=='fd':
        scene = mi.load_file(scene_path, res=res, integrator='prb', **config['_extra'])
        params = mi.traverse(scene)
        
        eps = 1e-3  
        if key is not None:
            params[key] -= eps
            params[key] = dr.clamp(params[key], 0.001, 1.0)
            params.update()
        if mesh_key is not None:
            initial_vertex_positions = dr.unravel(mi.Point3f, params[config['mesh_key']])
            apply_transformation_forward(params, config, initial_vertex_positions, -eps, with_grad=False)

        image_ours = do_high_spp_rendering(queue, scene, res, params, spp, 'forward 1', None)
        
        if key is not None:
            params[key] += 2 * eps 
            params[key] = dr.clamp(params[key], 0.001, 1.0)
            params.update()
        if mesh_key is not None:
            apply_transformation_forward(params, config, initial_vertex_positions, +eps, with_grad=False)
               
        image_ours2 = do_high_spp_rendering(queue, scene, res, params, spp, 'forward 2', None)
            
        grad_image_ours = (image_ours2 - image_ours) / (eps * 2)
    else:
        
        scene = mi.load_file(scene_path, res=res, integrator=integrator, **config['_extra'])
        params = mi.traverse(scene)
        if key is not None:
            if type(key) == list:
                for k in key:
                    dr.enable_grad(params[k])
            else:
                dr.enable_grad(params[key])
            params.update()
        if mesh_key is not None:
            initial_vertex_positions = dr.unravel(mi.Point3f, params[config['mesh_key']])
            theta = mi.Float(0)
            dr.enable_grad(theta)
            apply_transformation_forward(params, config, initial_vertex_positions, theta)
            dr.forward(theta, dr.ADFlag.ClearEdges)
        time0 = time.time()
        image_ours = mi.render(scene, params, spp=spp_per_pass, spp_grad=spp_per_pass_ad, seed=0)
        print("=================== time usage %.3f" % (time.time() - time0))
        print("========================== Diff")
        time0 = time.time()
        if key is not None:
            if type(key) == list:
                for k in key:
                    dr.set_grad(params[k], 1)
                grad_image_ours = dr.forward_to(image_ours)
            else:
                dr.forward(params[key])
                grad_image_ours = dr.grad(image_ours)
                
        if mesh_key is not None:
            grad_image_ours = dr.forward_to(image_ours)
            del theta
        print("=================== time usage %.3f" % (time.time() - time0))

    totaltime = get_elapsed_execution_time()
    msg = f'MtsTotalTime { time.time() - time0:.2f}s KernelTotalTime {totaltime/1000:.2f}s '
    print(LOGI(), msg, LEND)
    if queue is not None:
        queue.put(msg)

    
    mi.util.write_bitmap(os.path.join(output_dir, f'primal_{integrator}.exr'), image_ours)
    mi.util.write_bitmap(os.path.join(output_dir, f'deriv_{integrator}.exr'), grad_image_ours)

    times = np.ones(shape=(iteratons,1), dtype=np.float64) * totaltime / 1e3
    save_time(output_dir, times)

def forward_ours(queue: Queue, output_dir, scene_name, scene_path, param_key, integrator, iteratons, res, spp_per_pass, spp_per_pass_ad, config: dict):
    mi.set_variant('llvm_ad_rgb')
    np.random.seed(0)

    try_remove('plot_sample_primal.txt')
    try_remove('plot_sample_adjoint_primal.txt')
    try_remove('plot_sample_adjoint.txt')
    try_remove('plot_sample_primal_2.txt')
    try_remove('plot_sample_adjoint_primal_2.txt')
    try_remove('plot_sample_adjoint_2.txt')


    print(LOGT(), "[main]", "start rendering", scene_name, scene_path, param_key, res, spp_per_pass, config, LEND)

    use_decompse_adjoint = True if config["target_fuse"]>=3 else False

    scene = mi.load_file(scene_path, res=res, integrator=integrator, **config['_extra'])
    
    
    
    
    
    
    

    
    bbox = scene.bbox()
    print(bbox.center(), bbox.extents())

    
    try_remove("guiding_turn3.txt")
    try_remove("guiding_turn1.txt")
    try_remove("guiding_log.txt") 
    try_remove("guiding_log2.txt") 
    try_remove("guiding_log_ad_pos.txt") 
    try_remove("guiding_log_ad_neg.txt") 
    try_remove("guiding_log_mix.txt") 
    try_remove("guiding_samples.txt")  
    try_remove("guiding_samples_adjoint.txt") 

    write_float_to_file("guiding_clock.txt", 0)
    TESTID = 'm3test05'
    def run_command(cmd):
        my_run_cmd(TESTID, cmd, "dummy", instant=True)

    cmd = make_mts1_cmd_str(TESTID, spp_per_pass, bbox, config)
    try_remove("guiding_turn1.txt")

    th = Thread(target = run_command, args = (cmd,))  
    th.start()
    print(LOGD(), "con", "mts1 proc start", LEND)
    mts1_block_me()

    try_remove("guiding_log.txt") 
    try_remove("guiding_log2.txt")
    try_remove("guiding_log_ad_pos.txt") 
    try_remove("guiding_log_ad_neg.txt")
    try_remove("guiding_log_mix.txt")

    time00 = time.time()
    total_time = 0

    times = np.zeros(shape=(iteratons,1), dtype=np.float64)

    scene.integrator().set_config(**config)

    params = mi.traverse(scene)
    key = param_key
    mesh_key = config['mesh_key'] if 'mesh_key' in config else None

    if key is not None:
        if type(key) == list:
            for k in key:
                dr.enable_grad(params[k])
        else:
            dr.enable_grad(params[key])
        params.update()
    if mesh_key is not None:
        initial_vertex_positions = dr.unravel(mi.Point3f, params[config['mesh_key']])

    

    
    

    for i in range(iteratons):
        print(LOGI(), "[main]", "Iteration", i, "=======================", LEND)
        print(LOGI(), "[main]", "Primal phase", LEND)

        
        
        
        

        
        scene.integrator().set_config(is_primal_phase=True, enable_positivization=False, enable_ssguiding=False)
        scene.integrator().outputfile_prefix = os.path.join(output_dir, f'iter{i}_')

        if i == 0:
            try_remove("guiding_log.txt") 
            try_remove("guiding_log2.txt") 
            try_remove("guiding_log_ad_pos.txt") 
            try_remove("guiding_log_ad_neg.txt")
            try_remove("guiding_log_mix.txt")

        time0 = time.time()
        if mesh_key is not None:
            theta = mi.Float(0)
            dr.enable_grad(theta)
            apply_transformation_forward(params, config, initial_vertex_positions, theta)
            dr.forward(theta, dr.ADFlag.ClearEdges)

        image_ours = mi.render(scene, params, spp=spp_per_pass, spp_grad=spp_per_pass_ad, seed=i)
        print(LOGT(), "[main]", "time usage %.3f" % (time.time() - time0), LEND)
        os.system("copy guiding_samples.txt guiding_samples_primal.txt >nul")

        scene.integrator().set_config(is_primal_phase=False, enable_positivization=config['enable_positivization'], enable_ssguiding=config['enable_ssguiding'])
        
        print(LOGI(), "[main]", "Adjoint phase (forward)", LEND)
        time0 = time.time()
        
        if key is not None:
            if type(key) == list:
                if i < iteratons - 1:
                    for k in key:  
                        dr.set_grad(params[k], 1)
                    grad_image_ours = dr.forward_to(image_ours)
                else:
                    
                    dr.forward(params[key[0]], flags=dr.ADFlag.ClearVertices)
                    grad_image_ours = dr.grad(image_ours)
                    mi.util.write_bitmap(os.path.join(output_dir, f'deriv_{i}_key1.exr'), grad_image_ours)
                    dr.set_grad(image_ours, 0)
                    
                    dr.forward(params[key[1]])
                    grad_image_ours = dr.grad(image_ours)
                    mi.util.write_bitmap(os.path.join(output_dir, f'deriv_{i}_key2.exr'), grad_image_ours)
            else:
                dr.forward(params[key])
                grad_image_ours = dr.grad(image_ours)
        
        if mesh_key is not None:
            grad_image_ours = dr.forward_to(image_ours)
            del theta
        print(LOGI(), "[main]", "Adjoint phase (grad)", LEND)
        print(LOGT(), "[main]", "time usage %.3f" % (time.time() - time0), LEND)
        
        total_time += get_elapsed_execution_time()
        times[i] = total_time / 1e3
        msg = f'iter {i} OurTotalTime { time.time() - time00:.2f}s KernelTotalTime {total_time/1000:.2f}s '
        print(LOGI(), msg, LEND)
        if queue is not None:
            queue.put(msg)
        if config['_prefer_monitor'] == True:
            monitor_client.send_post(scene_name, config['_id'], f'iter_{i}')

        
        os.system("copy /b guiding_samples_primal.txt+guiding_samples.txt guiding_samples_merge.txt >nul")
        os.system("copy guiding_samples_merge.txt guiding_samples.txt >nul")

        if use_decompse_adjoint and os.path.exists("guiding_samples_adjoint.txt"): 
            print('adjoint samples save success')

        
        mts1_block_me()

        
        mi.util.write_bitmap(os.path.join(output_dir, f'primal_{i}.exr'), image_ours)
        mi.util.write_bitmap(os.path.join(output_dir, f'deriv_{i}.exr'), grad_image_ours)

        
        
        
        
        
        
        
        os.system("copy guiding_log_mix.txt results\\m3dpg_oneopt\\guiding_log_mix_%d.txt" % i)
        
        
        

        print(LOGT(), "[main]", "m3dpg_oneopt wallclock ", time.time() - time00, LEND)

    mts1_kill_me()
    print(LOGI(), "con", "waiting mts1 to die", LEND)
    th.join()
    try_remove("guiding_turn1end.txt")

    print(LOGT(), "[main]", f"our forward {integrator} total time usage ", time.time() - time00, LEND)
    save_time(output_dir, times)
    if config['_prefer_monitor'] == True:
        monitor_client.send_post(scene_name, config['_id'], f'done')

def get_primal_gt_path(output_dir, res, spp):
    return os.path.join(os.path.abspath(output_dir), f'image_r{res}_s{spp}.exr')

image_gt = None
clamp_value = 1e9

def loss_func(image):
    return mse(dr.minimum(clamp_value, image), dr.minimum(clamp_value, image_gt))

def primal_gt(queue: Queue, output_dir, scene_path, config):
    param_key = config['key']
    res = config['res']
    spp = config['gt_spp']
    output_name = get_primal_gt_path(output_dir, res, spp)

    assert os.path.exists(scene_path)

    
    hashfilename = os.path.join(os.path.join(output_dir, '..'), 'scene_gt_hash.json')  

    scene_xml_file_hash = ''
    with open(scene_path, 'rb') as scene_file:
        scene_xml_file_hash = hashlib.md5(scene_file.read()).hexdigest() + f'_r{res}_s{spp}'  
    gt_image_file_hash = ''
    
    record = {}
    if os.path.exists(hashfilename):
        with open(hashfilename, 'r') as f:
            record = json.load(f)

    msg = ''
    if scene_xml_file_hash in record:
        hash_value = record[scene_xml_file_hash]
        if os.path.exists(output_name):  
            with open(output_name, 'rb') as f:
                gt_image_file_hash = hashlib.md5(f.read()).hexdigest()
                if gt_image_file_hash == hash_value[0]:
                    notify(queue, f'gt {output_name} already exist in {output_dir}')
                    
                else:
                    msg = 'current gt img not match scene, regenerate' 
                    
                    
        else:
            msg = 'gt img not found'
    else:
        msg = 'detect a new scene or modified scene'
    
    if len(msg) > 0 or len(config['_extra']) > 0:
        
        notify(queue, msg)
        notify(queue, "========================== gt start")
        mi.set_variant('cuda_ad_rgb')
        time0 = time.time()
        scene = mi.load_file(scene_path, res=res, integrator='prb', **config['_extra'])
        image = do_high_spp_rendering(queue, scene, res, None, spp, 'primal gt', None)
        notify(queue, "=================== gt done, time usage %.3f" % (time.time() - time0))

        mi.util.write_bitmap(output_name, image, write_async=False) 
        notify(queue, f'save gt {output_name} in {output_dir}')

        with open(output_name, 'rb') as f:
            gt_image_file_hash = hashlib.md5(f.read()).hexdigest()

        
        params = mi.traverse(scene)
        param_ref =  mi.TensorXf(params[param_key])
        params_init = np.full(np.array(param_ref.shape).tolist(), fill_value=config['fill_value'], dtype=np.float32)
        if 'normal' in param_key: 
            params_init[:,:,0]=0.5
            params_init[:,:,1]=0.5
            params_init[:,:,2]=1
        params[param_key] = mi.TensorXf(params_init)
        params.update()
        image_init = do_high_spp_rendering(queue, scene, res, params, spp, 'primal initial gt', None)
        mi.util.write_bitmap(os.path.join(output_dir, f'image_initial_r{res}_s{spp}.exr'), image_init, write_async=False)
        mi.util.write_bitmap(os.path.join(output_dir, 'param_initial.exr'), params_init, write_async=False)
        mi.util.write_bitmap(os.path.join(output_dir, 'param_ref.exr'), param_ref, write_async=False)
        dr.kernel_history_clear()

    record[scene_xml_file_hash] = (gt_image_file_hash, str(os.path.abspath(scene_path)))
    with open(hashfilename, 'w') as f:
        json.dump(record, f)


def backward_mts(queue: Queue, output_dir, scene_name, scene_path, param_key, integrator, iteratons, res, spp_per_pass, spp_per_pass_ad, config: dict):
    
    mi.set_variant('llvm_ad_rgb')
    np.random.seed(0)
    image_gt_path = get_primal_gt_path(os.path.join(output_dir, '..'), res, config['gt_spp'])
    global image_gt
    image_gt = mi.Bitmap(image_gt_path)
    image_gt = mi.TensorXf(image_gt)

    scene = mi.load_file(scene_path, res=res, integrator=integrator, **config['_extra'])
    key = param_key
    params = mi.traverse(scene)
    param_ref = mi.TensorXf(params[key])
    param_shape = np.array(params[key].shape)

    param_initial = np.full(param_shape.tolist(), fill_value=config['fill_value'], dtype=np.float32)
    if param_shape[2] == 4:  
        param_initial[:,:,3] = 1
        param_ref[:,:,3] = 1
    if 'normal' in key: 
        param_initial[:,:,0]=0.5
        param_initial[:,:,1]=0.5
        param_initial[:,:,2]=1
    params[key] = mi.TensorXf(param_initial)
    params.update()

    opt = mi.ad.Adam(lr=config['learning_rate'], beta_1=config['momentum'] if 'momentum' in config else 0) 
    opt[key] = params[key]
    params.update(opt)

    time00 = time.time()
    total_time = 0
    times = np.zeros(shape=(iteratons,1), dtype=np.float64)

    for it in range(iteratons):
        image = mi.render(scene, params, spp=spp_per_pass, spp_grad=spp_per_pass_ad, seed=it)
        loss = loss_func(image)
        dr.backward(loss)
        grad = dr.grad(opt[key])
        mi.util.write_bitmap(os.path.join(output_dir, f'primal_{it}.exr'), image)
        mi.util.write_bitmap(os.path.join(output_dir, f'deriv_{it}.exr'), grad)
        opt.step()
        if 'normal' in key:
            opt[key] = normalize(opt[key]) 
        else:
            opt[key] = dr.clamp(opt[key], 0.001, 1.0)
        params.update(opt)
        output_file = os.path.join(output_dir, f'param_{it}.exr')
        mi.util.write_bitmap(output_file, opt[key])

        if 'normal' in key:
            ploss = mse(decode(param_ref), decode(opt[key]))
        else:
            ploss = mse(param_ref, opt[key])
        print(loss, ploss)

        total_time += get_elapsed_execution_time()
        times[it] = total_time / 1e3
        msg = f'iter {it} MtsTotalTime { time.time() - time00:.2f}s KernelTotalTime {total_time/1000:.2f}s '
        if queue is not None:
            queue.put(msg)
        print(LOGI(), msg, LEND)
  
    print(LOGT(), f"[GT] mts {integrator} total time usage ", time.time() - time00, LEND)
    save_time(output_dir, times)

def backward_ours(queue: Queue, output_dir, scene_name, scene_path, param_key, integrator, iteratons, res, spp_per_pass, spp_per_pass_ad, config: dict):
    mi.set_variant('llvm_ad_rgb')
    np.random.seed(0)
    image_gt_path = get_primal_gt_path(os.path.join(output_dir, '..'), res, config['gt_spp'])
    global image_gt
    image_gt = mi.Bitmap(image_gt_path)
    image_gt = mi.TensorXf(image_gt)
    
    try_remove('guiding_film.txt')
    try_remove('plot_sample_primal.txt')
    try_remove('plot_sample_adjoint_primal.txt')
    try_remove('plot_sample_adjoint.txt')
    try_remove('plot_sample_primal_2.txt')
    try_remove('plot_sample_adjoint_primal_2.txt')
    try_remove('plot_sample_adjoint_2.txt')
    
    scene = mi.load_file(scene_path, res=res, integrator=integrator, **config['_extra'])
    key = param_key
    params = mi.traverse(scene)
    param_ref = mi.TensorXf(params[key])
    param_shape = np.array(params[key].shape)

    param_initial = np.full(param_shape.tolist(), fill_value=config['fill_value'], dtype=np.float32)
    if param_shape[2] == 4:  
        param_initial[:,:,3] = 1
        param_ref[:,:,3] = 1
    if 'normal' in key: 
        param_initial[:,:,0]=0.5
        param_initial[:,:,1]=0.5
        param_initial[:,:,2]=1
    params[key] = mi.TensorXf(param_initial)
    params.update()
    scene.integrator().param_key = key  
    scene.integrator().set_config(**config)

    is_base_color = 'base_color' in key

    if 'learning_rate' in config.keys():
        learning_rate = config['learning_rate']
    elif is_base_color:
        learning_rate = 0.1
    else:
        learning_rate = 0.01

    opt = mi.ad.Adam(lr=learning_rate, beta_1=config['momentum'] if 'momentum' in config else 0) 
    opt[key] = params[key]
    params.update(opt)

    bbox = scene.bbox()
    print(bbox.center(), bbox.extents())

    try_remove("guiding_turn3.txt")
    try_remove("guiding_turn1.txt")
    try_remove("guiding_log_mix.txt") 
    try_remove("guiding_samples.txt") 
    try_remove("guiding_samples_adjoint.txt") 

    write_float_to_file("guiding_clock.txt", 0)
    TESTID = 'm3test08'
    def run_command(cmd):
        my_run_cmd(TESTID, cmd, "dummy", instant=True)

    cmd = make_mts1_cmd_str(TESTID, spp_per_pass, bbox, config)

    try_remove("guiding_turn1.txt")
    th = Thread(target=run_command, args=(cmd,))
    th.start()
    print(LOGD(), "con", "mts1 proc start", LEND)

    mts1_block_me()
    time.sleep(0.1)
    try_remove("guiding_log_mix.txt") 

    time00 = time.time()

    total_time = 0
    times = np.zeros(shape=(iteratons,1), dtype=np.float64)
    paramlosses = []

    use_warmup = 'warmup-iterations' in config.keys()
    opt_begin_iter = 0 if use_warmup == False else config['warmup-iterations']

    fit_iter_max = config['fit_iter_max'] if 'fit_iter_max' in config.keys() else 99999

    image = None

    for it in range(iteratons):
        print(LOGK(), "[main]", "======== Iteration", it, "================================", LEND)

        scene.integrator().set_config(opt_iter=it, is_primal_phase=True, enable_ssguiding=False, enable_positivization=False, require_record = True if it < fit_iter_max else False)
        scene.integrator().outputfile_prefix = os.path.join(output_dir, f'iter{it}_')

        if it == 0:
            try_remove("guiding_log_mix.txt")

        print(LOGI(), "[main]", "Primal phase", LEND)
        time0 = time.time()
        
        image = mi.render(scene, params, spp=spp_per_pass, spp_grad=spp_per_pass_ad, seed=it)
        os.system("copy guiding_samples.txt guiding_samples_primal.txt >nul")
        loss = loss_func(image)
        scene.integrator().set_config(opt_iter = it, is_primal_phase=False, enable_ssguiding=config["enable_ssguiding"], enable_positivization=config['enable_positivization'], require_record = True if it < fit_iter_max else False)

        print(LOGI(), "[main]", "Adjoint phase", LEND)
        dr.backward(loss)
        os.system("copy /b guiding_samples_primal.txt+guiding_samples.txt guiding_samples_merge.txt >nul")
        os.system("copy guiding_samples_merge.txt guiding_samples.txt >nul")

        grad = dr.grad(opt[key])
        mi.util.write_bitmap(os.path.join(output_dir, f'primal_{it}.exr'), image)
        mi.util.write_bitmap(os.path.join(output_dir, f'deriv_{it}.exr'), grad)

        if it >= opt_begin_iter:
            opt.step()
            if 'normal' in key:
                opt[key] = normalize(opt[key]) 
            else:
                opt[key] = dr.clamp(opt[key], 0.001, 1.0)
            params.update(opt)
        else:
            dr.set_grad(opt[key], 0)

        output_file = os.path.join(output_dir, f'param_{it}.exr')
        mi.util.write_bitmap(output_file, opt[key])
 
        
        
        
        
        if it < fit_iter_max:
            mts1_block_me()

        if 'normal' in key:
            ploss = mse(decode(param_ref), decode(opt[key]))
        else:
            ploss = mse(param_ref, opt[key])
        paramlosses.append(ploss[0])

        if config['_prefer_monitor'] == True:
            monitor_client.send_post(scene_name, config['_id'], f'iter_{it}')
        total_time += get_elapsed_execution_time()
        times[it] = total_time / 1e3
        msg = f'iter {it} OurTotalTime { time.time() - time00:.2f}s KernelTotalTime {total_time/1000:.2f}s '
        print(LOGI(), msg, LEND)
        if queue is not None:
            queue.put(msg)
        if it % 5 == 0:
            print(LOGK(), "paramlosses (first 5):", ", ".join(str(x)[1:8] for x in paramlosses[:5]), LEND)
            print(LOGK(), "paramlosses (jmp 10):", ", ".join(str(x)[1:8] for x in paramlosses[::10]), LEND)
            print(LOGK(), "paramlosses (last 5):", ", ".join(str(x)[1:8] for x in paramlosses[-5:]), LEND)

    mts1_kill_me()
    print(LOGI(), "con", "waiting mts1 to die", LEND)
    th.join()
    try_remove("guiding_turn1end.txt")

    print(LOGT(), "[main]", "total time usage ", time.time() - time00, LEND)
    save_time(output_dir, times)
    if config['_prefer_monitor'] == True:
        monitor_client.send_post(scene_name, config['_id'], f'done')
