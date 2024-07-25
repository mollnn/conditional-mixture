from __future__ import annotations as __annotations__
from matplotlib import cm, pyplot as plt 

import mitsuba as mi
import drjit as dr
import gc
import numpy as np
import os
import itertools
import time
import scipy.ndimage

swap_weight = 0.95

def canonical_to_dir(p: mi.Point2f) -> mi.Vector3f:
	cos_theta = mi.Float(p[1] * 2 - 1)
	phi = mi.Float(2 * 3.14159 * p[0])
	sin_theta = dr.sqrt(1 - cos_theta * cos_theta)
	sin_phi = dr.sin(phi)
	cos_phi = dr.cos(phi)
	return mi.Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)

def fmod(x, y):
	return x - dr.floor(x / y) * y

def dir_to_canonical(d: mi.Vector3f) -> mi.Point2f:
	cos_theta = dr.minimum(dr.maximum(d[2], -1.0), 1.0)
	phi = dr.atan2(d.y, d.x)
	phi = fmod(phi, 2.0 * 3.14159)
	phi[phi < 0] += 2.0 * 3.14159
	return mi.Point2f(phi / (2 * 3.14159), (cos_theta + 1) / 2)

def dbsdf_dalpha_finite(bsdf, ctx, si, wo, alpha):
    delta = 0.001
    val0 = bsdf.eval_with_roughness(ctx, si, wo, alpha)
    val1 = bsdf.eval_with_roughness(ctx, si, wo, alpha + delta)
    result = dr.mean((val1 - val0) / delta)
    
    return result

def prob_film(w, h, result_film):
	result_film = np.abs(np.nan_to_num(result_film))
	p_film = np.sum(result_film, axis = -1)
	p_film = np.abs(p_film)
	p_film = scipy.ndimage.gaussian_filter(p_film, sigma=5)
	p_film /= np.sum(p_film)
	p_film += 1 / w / h * 1
	p_film /= np.sum(p_film)
	return p_film


def make_film_distr(w, h, spp, p: np.ndarray): 
    t_begin = time.time()
    p1 = p.flatten()
    p1 = np.nan_to_num(p1, nan=0, posinf=0, neginf=0)
    p1 += 1e-18
    p1 /= sum(p1)
    idx = np.random.choice(len(p1), w * h * spp, p=p1)
    pos = mi.Vector2i(np.array([(x//w, x%w) for x in idx]))        
    t_end = time.time()
    print("    ssguidewr uses %.3f sec" % (t_end - t_begin))
    return pos

def get_positiv_str(postiv_iter):
    if postiv_iter < 0:
        return 'PositivOff'
    elif postiv_iter == 0:
        return 'PositivPos'
    else:
        assert postiv_iter == 1
        return 'PositivNeg'


class MySDTree:
    def __init__(self, type_id, mixture_count=1):  
        t_begin = time.time()
        self.mix_dtree = mixture_count > 1
        if self.mix_dtree:
            assert type_id == 4
        self.mixture_count = mixture_count
        choice = ["guiding_log.txt", "guiding_log2.txt", "guiding_log_ad_pos.txt", "guiding_log_ad_neg.txt", "guiding_log_mix.txt"]

        fp = open(choice[type_id])
        t = 0
        lines = fp.readlines()
        fp.close()
        
        a = [list(map(float, i.strip().split())) for i in lines]
        line_id = 0
        stree_size = int(a[line_id][0])
        self.stree_info = np.zeros((stree_size, 4), dtype=np.float32)
        self.stree_offset = np.zeros((stree_size, 3), dtype=np.float32)
        self.stree_size = np.zeros((stree_size, 3), dtype=np.float32)
        self.dtree_ch_list = []
        self.dtree_val_list = []
        self.dtree_sw_list = []
        line_id += 1
        offset = 0

        dtree_val_size_unit = 4 * self.mixture_count
        dtree_sw_size_unit = 2 
        offset0, size0 = None, None
        while True:
            if int(a[line_id][0]) == -1:
                line_id += 1
                break
            snode_id = int(a[line_id][0])
            snode_info = list(map(int, a[line_id][1:4])) + [offset]
            snode_offset = list(map(float, a[line_id][4:7]))
            snode_size = list(map(float, a[line_id][7:10]))
            self.stree_info[snode_id] = np.array(snode_info, dtype=np.float32)
            self.stree_offset[snode_id] = np.array(snode_offset, dtype=np.float32)
            self.stree_size[snode_id] = np.array(snode_size, dtype=np.float32)
            if offset0 == None:
                offset0 = snode_offset
                size0 = snode_size
            self.stree_offset[snode_id] = (self.stree_offset[snode_id] - offset0) / size0
            self.stree_size[snode_id] = self.stree_size[snode_id] / size0
            line_id += 1
            dtree_size = int(a[line_id][0])
            dtree_sw = float(a[line_id][1])
            offset += dtree_size
            line_id += 1
            
            self.dtree_ch = np.zeros((dtree_size, 4), dtype=np.float32)
            self.dtree_val = np.zeros((dtree_size, dtree_val_size_unit), dtype=np.float32)
            self.dtree_sw = np.ones((dtree_size, dtree_sw_size_unit), dtype=np.float32) * dtree_sw
            if dtree_sw_size_unit == 2:
                self.dtree_sw[:,1] = float(a[line_id-1][2])

            while True:
                if int(a[line_id][0]) == -1:
                    line_id += 1
                    break
                dnode_id = int(a[line_id][0])
                dnode_ch = [i - dnode_id for i in list(map(int, a[line_id][1:5]))]
                dnode_val = a[line_id][5:5+dtree_val_size_unit]
                self.dtree_ch[dnode_id] = np.array(dnode_ch)
                self.dtree_val[dnode_id] = np.array(dnode_val)
                line_id += 1

            self.dtree_ch_list.append(self.dtree_ch)
            self.dtree_val_list.append(self.dtree_val)
            self.dtree_sw_list.append(self.dtree_sw)
        lut_list = []
        while line_id < len(a):
            ln = list(map(int, a[line_id]))
            if len(ln) == 0:
                break
            ln = np.array(ln)
            lut_list.append(ln)
            line_id += 1

        self.stree_lut_n = int(len(lut_list) ** 0.5)
        assert self.stree_lut_n * self.stree_lut_n == len(lut_list)
        self.stree_lut = np.concatenate(lut_list)

        self.dtree_ch = np.concatenate(self.dtree_ch_list)
        self.dtree_val = np.concatenate(self.dtree_val_list)
        self.dtree_sw = np.concatenate(self.dtree_sw_list)
        
        STREE_PAD = 10000
        DTREE_PAD = 1000000

        while self.stree_info.shape[0] > STREE_PAD:
            STREE_PAD *= 3
        while self.dtree_ch.shape[0] > DTREE_PAD:
            DTREE_PAD *= 3

        self.stree_info = np.pad(self.stree_info, ((0, STREE_PAD - len(self.stree_info)), (0,0)))
        self.stree_offset = np.pad(self.stree_offset, ((0, STREE_PAD - len(self.stree_offset)), (0,0)))
        self.stree_size = np.pad(self.stree_size, ((0, STREE_PAD - len(self.stree_size)), (0,0)))
        self.dtree_ch = np.pad(self.dtree_ch, ((0, DTREE_PAD - len(self.dtree_ch)), (0,0)))
        self.dtree_val = np.pad(self.dtree_val, ((0, DTREE_PAD - len(self.dtree_val)), (0,0)))
        self.dtree_sw = np.pad(self.dtree_sw, ((0, DTREE_PAD - len(self.dtree_sw)), (0,0)))

        self.nst = len(self.stree_info)
        self.ndt = len(self.dtree_ch)

        self.stree_info = mi.Texture1f(mi.TensorXf(self.stree_info.reshape(-1, 4)))
        self.stree_offset = mi.Texture1f(mi.TensorXf(self.stree_offset.reshape(-1, 3)))
        self.stree_size = mi.Texture1f(mi.TensorXf(self.stree_size.reshape(-1, 3)))
        self.stree_lut = mi.Texture3f(mi.TensorXf(self.stree_lut.reshape(self.stree_lut_n, self.stree_lut_n, self.stree_lut_n, 1)))
        self.dtree_ch = mi.Texture1f(mi.TensorXf(self.dtree_ch.reshape(-1, 4)))
        self.dtree_val = mi.Texture1f(mi.TensorXf(self.dtree_val.reshape(-1, dtree_val_size_unit)))
        self.dtree_sw = mi.Texture1f(mi.TensorXf(self.dtree_sw.reshape(-1, dtree_sw_size_unit)))

        self.guiding_mis_weight_ad = 0.5
        
        self.grid_resolution = 4
        self.enable_positivization = False
        self.enable_positivization_pm = False
        self.enable_positivization_ps = False
        self.positivization_iter = -1
        self.enable_prb_pro = False
        self.enable_srb_pro = False
        self.record_ratio = False
        self.resolution = 256 
        self.spp = 2
        t_end = time.time()
        print("    loadsdtre uses %.3f sec" % (t_end - t_begin))

    def query_stree(self, query_pos: mi.Point3f) -> mi.Float:
        query_pos_relative_ = query_pos * 0 

        query_pos_relative = query_pos * 1
        query_snode_id = query_pos[0] * 0 

        
        query_snode_id = self.stree_lut.eval_fetch(query_pos_relative + mi.Vector3f(0.5 / self.stree_lut_n))[0][0]
        query_snode_offset = mi.Vector3f(self.stree_offset.eval_fetch(query_snode_id / self.nst)[1])
        query_snode_size = mi.Vector3f(self.stree_size.eval_fetch(query_snode_id / self.nst)[1])
        query_pos_relative = (query_pos_relative - query_snode_offset) / query_snode_size

        axis = mi.Float(0)
        notleaf = mi.Bool(True)
        query_dnode_id = mi.Float(0)
        count = mi.Int(0)
        
        loop = mi.Loop(name="query_stree_loop", state=lambda: (query_pos_relative, query_pos_relative_, query_snode_id, notleaf, axis, query_dnode_id, count))
        
        while loop(notleaf):
        
            query_snode_id = dr.round(query_snode_id)

            info_res = self.stree_info.eval_fetch((query_snode_id) / self.nst)[1]

            
            notleaf[info_res[1] < 0.5] &= False
            axis = info_res[0]

            count += 1
            notleaf[count > 512] &= False

            query_pos_relative_ = query_pos_relative * 1
            query_pos_relative_[0][notleaf & (dr.abs(axis - 0) < 0.5) & (query_pos_relative[0] < 0.5)] = query_pos_relative[0] * 2
            query_pos_relative_[0][notleaf & (dr.abs(axis - 0) < 0.5) & (query_pos_relative[0] >= 0.5)] = query_pos_relative[0] * 2 - 1
            query_pos_relative_[1][notleaf & (dr.abs(axis - 1) < 0.5) & (query_pos_relative[1] < 0.5)] = query_pos_relative[1] * 2
            query_pos_relative_[1][notleaf & (dr.abs(axis - 1) < 0.5) & (query_pos_relative[1] >= 0.5)] = query_pos_relative[1] * 2 - 1
            query_pos_relative_[2][notleaf & (dr.abs(axis - 2) < 0.5) & (query_pos_relative[2] < 0.5)] = query_pos_relative[2] * 2
            query_pos_relative_[2][notleaf & (dr.abs(axis - 2) < 0.5) & (query_pos_relative[2] >= 0.5)] = query_pos_relative[2] * 2 - 1

            query_snode_id[notleaf & (dr.abs(axis - 0) < 0.5) & (query_pos_relative[0] < 0.5)] = info_res[1]
            query_snode_id[notleaf & (dr.abs(axis - 0) < 0.5) & (query_pos_relative[0] >= 0.5)] = info_res[2]
            query_snode_id[notleaf & (dr.abs(axis - 1) < 0.5) & (query_pos_relative[1] < 0.5)] = info_res[1]
            query_snode_id[notleaf & (dr.abs(axis - 1) < 0.5) & (query_pos_relative[1] >= 0.5)] = info_res[2]
            query_snode_id[notleaf & (dr.abs(axis - 2) < 0.5) & (query_pos_relative[2] < 0.5)] = info_res[1]
            query_snode_id[notleaf & (dr.abs(axis - 2) < 0.5) & (query_pos_relative[2] >= 0.5)] = info_res[2]

            query_pos_relative = query_pos_relative_ * 1

        query_dnode_id = self.stree_info.eval_fetch((query_snode_id) / self.nst)[1][3]
        
        
        return query_dnode_id

    def query_stree_wrapped(self, query_pos: mi.Point3f, bbx: mi.BoundingBox3f) -> mi.Float:
        xc = bbx.center()
        xs = bbx.extents()
        x0 = xc - xs / 2
        ans = self.query_stree((query_pos - x0) / xs)
        
        return ans
    

    
    
    
    
    def sample_dtree(self, rng: mi.Sampler, sample_dnode_id_: mi.Float, sample_method: int, bsdf: mi.BSDFPtr, si: mi.SurfaceInteraction3f, bsdf_ctx: mi.BSDFContext, alpha: mi.Float, β: mi.Float, β_df: mi.Float, record_outputname: str) -> mi.Point2f:
        sample_dnode_id = sample_dnode_id_ * 1.0
        sample_base = sample_dnode_id * 1.0
        sample_ans = mi.Point2f(0, 0)
        sample_notleaf = mi.Bool(True)
        sample_notleaf_last = mi.Bool(True)
        sample_child_id = mi.Float(0)
        sample_val_0 = mi.Float(0)
        sample_val_1 = mi.Float(0)
        sample_val_2 = mi.Float(0)
        sample_val_3 = mi.Float(0)
        sample_val_sum = mi.Float(0)
        sample_rnd = mi.Float(0)
        sample_rnd_src = rng
        count = mi.Int(0)
        
        coord_base = mi.Point2f(0, 0)

        sample_idx_scale = self.mixture_count

        
        
        
        enable_brute_force_product = type(sample_method)==int and sample_method == 2
        
        assert self.mix_dtree 
        assert not (enable_brute_force_product and self.enable_prb_pro)
        assert not (self.enable_prb_pro and self.enable_srb_pro)

        
        
        if self.mixture_count == 5:
            
            mask_eq3 = dr.eq(sample_method, 3)
            mask_eq4 = dr.eq(sample_method, 4)

            gw_m_pos = 1
            gw_m_neg = 1
            gw_s_pos = 1
            gw_s_neg = 1
            if self.enable_positivization:
                iter = self.positivization_iter
                assert iter == 0 or iter == 1
                if self.enable_positivization_pm:
                    if iter == 0:
                        gw_m_pos = 1 * swap_weight
                        gw_m_neg = 1 * (1 - swap_weight)
                    else:
                        gw_m_pos = 1 * (1 - swap_weight)
                        gw_m_neg = 1 * swap_weight
                if self.enable_positivization_ps:
                    if iter == 0:
                        gw_s_pos = 1 * swap_weight
                        gw_s_neg = 1 * (1 - swap_weight)
                    else:
                        gw_s_pos = 1 * (1 - swap_weight)
                        gw_s_neg = 1 * swap_weight
            elif self.enable_srb_pro:
                
                root_val = self.dtree_val.eval_fetch((sample_dnode_id) / self.ndt)[1]
                sw = self.dtree_sw.eval_fetch((sample_dnode_id) / self.ndt)[1] 
                
                gw_m_pos, gw_m_neg = self.get_pos_neg_ratio(
                    root_val[0 * sample_idx_scale + 2] + root_val[1 * sample_idx_scale + 2] + root_val[2 * sample_idx_scale + 2] + root_val[3 * sample_idx_scale + 2],
                    root_val[0 * sample_idx_scale + 3] + root_val[1 * sample_idx_scale + 3] + root_val[2 * sample_idx_scale + 3] + root_val[3 * sample_idx_scale + 3],
                    sw[1]
                )

                
                gw_s_pos, gw_s_neg = self.get_pos_neg_ratio(
                    root_val[0 * sample_idx_scale + 0] + root_val[1 * sample_idx_scale + 0] + root_val[2 * sample_idx_scale + 0] + root_val[3 * sample_idx_scale + 0],
                    root_val[0 * sample_idx_scale + 1] + root_val[1 * sample_idx_scale + 1] + root_val[2 * sample_idx_scale + 1] + root_val[3 * sample_idx_scale + 1],
                    sw[0]
                )

            if self.record_ratio:
                spp = self.spp
                w, h = self.resolution, self.resolution
                idx = dr.arange(dtype=mi.UInt, start=0, stop=w*h*spp, step=1) // spp
                val_result = mi.TensorXf(0, shape=(w, h, 1))
                dr.scatter_reduce(dr.ReduceOp.Add, val_result.array, gw_m_pos / spp, idx)
                mi.util.write_bitmap(f'{record_outputname}_{get_positiv_str(self.positivization_iter)}_ratio_for_pM.exr', val_result)
                val_result = mi.TensorXf(0, shape=(w, h, 1))
                dr.scatter_reduce(dr.ReduceOp.Add, val_result.array, gw_s_pos / spp, idx)
                mi.util.write_bitmap(f'{record_outputname}_{get_positiv_str(self.positivization_iter)}_ratio_for_pS.exr', val_result)

        else:
            assert self.mixture_count == 4
            if sample_method != 1:
                
                gw1 = 1 - self.guiding_mis_weight_ad   
                gw2 = self.guiding_mis_weight_ad

                if not enable_brute_force_product:
                    root_val = self.dtree_val.eval_fetch((sample_dnode_id) / self.ndt)[1]
                    root_val_0 = mi.Vector4f(root_val[0:4])
                    root_val_1 = mi.Vector4f(root_val[4:8])
                    root_val_2 = mi.Vector4f(root_val[8:12])
                    root_val_3 = mi.Vector4f(root_val[12:16])
                    root_val_sum = root_val_0 + root_val_1 + root_val_2 + root_val_3 + 1e-18
                    sw = self.dtree_sw.eval_fetch((sample_dnode_id) / self.ndt)[1] 

                if self.enable_prb_pro:
                    gw1, gw2 = self.get_weight_pro(root_val, sw, β, β_df)  

                ratio = 0.5  
                if self.enable_positivization:                
                    ratio = swap_weight if self.positivization_iter == 0 else 1 - swap_weight
                elif self.enable_prb_pro:
                    ratio, _ = self.get_pos_neg_ratio(
                        root_val_0[1] + root_val_1[1] + root_val_2[1] + root_val_3[1],
                        root_val_0[2] + root_val_1[2] + root_val_2[2] + root_val_3[2],
                        sw[1]
                    )
                gw21 = gw2 * ratio
                gw22 = gw2 - gw21
                if self.record_ratio:
                    spp = self.spp
                    w, h = self.resolution, self.resolution
                    idx = dr.arange(dtype=mi.UInt, start=0, stop=w*h*spp, step=1) // spp
                    val_result = mi.TensorXf(0, shape=(w, h, 1))
                    dr.scatter_reduce(dr.ReduceOp.Add, val_result.array, ratio / spp, idx)
                    mi.util.write_bitmap(f'{record_outputname}_{get_positiv_str(self.positivization_iter)}_ratio_for_gw21gw22.exr', val_result)
                    
                    res = (gw2 + 1e-9) / (gw1 + gw2 + 1e-9)
                    val_result = mi.TensorXf(0, shape=(w, h, 1))
                    dr.scatter_reduce(dr.ReduceOp.Add, val_result.array, res / spp, idx)
                    mi.util.write_bitmap(f'{record_outputname}_{get_positiv_str(self.positivization_iter)}_gw2_div_gw12.exr', val_result)
                    

        sample_child_pdf = mi.Vector3f(1)

        i = mi.Float(0)
        power2_i = mi.Float(1)
        def eval_grid(resolution, offset, eval_lambda):
            
            margin = power2_i / resolution / 2
            start = margin
            stop = power2_i - margin
            step = (stop - start) / (resolution - 1 if resolution > 1 else 1)
            pos = [start + step * i for i in range(resolution)]
            ans = 0
            for a, b in itertools.product(pos, pos):
                p = mi.Point2f(a, b)
                ans += eval_lambda(canonical_to_dir(p + offset))
            return ans / (resolution * resolution)

        loop = mi.Loop(name="sample_dtree_loop", state=lambda: (count, sample_dnode_id, sample_base, sample_ans, sample_notleaf, sample_notleaf_last, 
            i, power2_i, sample_child_id, sample_val_0, sample_val_1, sample_val_2, sample_val_3, sample_val_sum, sample_rnd, sample_rnd_src,
            sample_child_pdf, coord_base))
        loop.set_max_iterations(10)
        while loop(sample_notleaf):
        
            count += 1
            sample_notleaf[count > 10] &= False
            sample_dnode_id = dr.round(sample_dnode_id)
            
            power2_i *= 0.5
            val_res = self.dtree_val.eval_fetch((sample_dnode_id) / self.ndt)[1]

            if self.mixture_count == 4:
                
                if sample_method == 1:
                    
                    sample_val_0 = val_res[0 * sample_idx_scale + 0]
                    sample_val_1 = val_res[1 * sample_idx_scale + 0]
                    sample_val_2 = val_res[2 * sample_idx_scale + 0]
                    sample_val_3 = val_res[3 * sample_idx_scale + 0]
                else:
                    if enable_brute_force_product:
                        bsdf_lambda = lambda wo: dr.mean(bsdf.eval(bsdf_ctx, si, si.to_local(wo)))
                        bsdf_val_0 = eval_grid(self.grid_resolution, coord_base, bsdf_lambda)
                        bsdf_val_1 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(0, power2_i), bsdf_lambda)
                        bsdf_val_2 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(power2_i, 0), bsdf_lambda)
                        bsdf_val_3 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(power2_i), bsdf_lambda)

                        dbsdf_lambda = lambda wo: dbsdf_dalpha_finite(bsdf, bsdf_ctx, si, si.to_local(wo), alpha)
                        dbsdf_val_0 = eval_grid(self.grid_resolution, coord_base, dbsdf_lambda)
                        dbsdf_val_1 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(0, power2_i), dbsdf_lambda)
                        dbsdf_val_2 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(power2_i, 0), dbsdf_lambda)
                        dbsdf_val_3 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(power2_i), dbsdf_lambda)

                        dbsdf_val_pos_0 = dr.select(dbsdf_val_0 > 0, dbsdf_val_0, 0)
                        dbsdf_val_neg_0 = dr.select(dbsdf_val_0 < 0, dbsdf_val_0, 0)
                        dbsdf_val_pos_1 = dr.select(dbsdf_val_1 > 0, dbsdf_val_1, 0)
                        dbsdf_val_neg_1 = dr.select(dbsdf_val_1 < 0, dbsdf_val_1, 0)
                        dbsdf_val_pos_2 = dr.select(dbsdf_val_2 > 0, dbsdf_val_2, 0)
                        dbsdf_val_neg_2 = dr.select(dbsdf_val_2 < 0, dbsdf_val_2, 0)
                        dbsdf_val_pos_3 = dr.select(dbsdf_val_3 > 0, dbsdf_val_3, 0)
                        dbsdf_val_neg_3 = dr.select(dbsdf_val_3 < 0, dbsdf_val_3, 0)
                        sample_val_0_prm = val_res[0 * sample_idx_scale + 0] * bsdf_val_0
                        sample_val_1_prm = val_res[1 * sample_idx_scale + 0] * bsdf_val_1
                        sample_val_2_prm = val_res[2 * sample_idx_scale + 0] * bsdf_val_2
                        sample_val_3_prm = val_res[3 * sample_idx_scale + 0] * bsdf_val_3
                        sample_val_0_pos = val_res[0 * sample_idx_scale + 0] * dbsdf_val_pos_0 
                        sample_val_1_pos = val_res[1 * sample_idx_scale + 0] * dbsdf_val_pos_1
                        sample_val_2_pos = val_res[2 * sample_idx_scale + 0] * dbsdf_val_pos_2
                        sample_val_3_pos = val_res[3 * sample_idx_scale + 0] * dbsdf_val_pos_3
                        sample_val_0_neg = -val_res[0 * sample_idx_scale + 0] * dbsdf_val_neg_0 
                        sample_val_1_neg = -val_res[1 * sample_idx_scale + 0] * dbsdf_val_neg_1
                        sample_val_2_neg = -val_res[2 * sample_idx_scale + 0] * dbsdf_val_neg_2
                        sample_val_3_neg = -val_res[3 * sample_idx_scale + 0] * dbsdf_val_neg_3

                        sum_primal = sample_val_0_prm + sample_val_1_prm + sample_val_2_prm + sample_val_3_prm
                        sum_pos = sample_val_0_pos + sample_val_1_pos + sample_val_2_pos + sample_val_3_pos
                        sum_neg = sample_val_0_neg + sample_val_1_neg + sample_val_2_neg + sample_val_3_neg

                        inv_sum_primal = dr.select(dr.neq(sum_primal, 0), dr.rcp(sum_primal), 0) 
                        inv_sum_pos = dr.select(dr.neq(sum_pos, 0), dr.rcp(sum_pos), 0) 
                        inv_sum_neg = dr.select(dr.neq(sum_neg, 0), dr.rcp(sum_neg), 0) 

                        sample_val_0_prm = sample_child_pdf[0] * sample_val_0_prm * inv_sum_primal
                        sample_val_1_prm = sample_child_pdf[0] * sample_val_1_prm * inv_sum_primal
                        sample_val_2_prm = sample_child_pdf[0] * sample_val_2_prm * inv_sum_primal
                        sample_val_3_prm = sample_child_pdf[0] * sample_val_3_prm * inv_sum_primal
                        sample_val_0_pos = sample_child_pdf[1] * sample_val_0_pos * inv_sum_pos
                        sample_val_1_pos = sample_child_pdf[1] * sample_val_1_pos * inv_sum_pos
                        sample_val_2_pos = sample_child_pdf[1] * sample_val_2_pos * inv_sum_pos
                        sample_val_3_pos = sample_child_pdf[1] * sample_val_3_pos * inv_sum_pos
                        sample_val_0_neg = sample_child_pdf[2] * sample_val_0_neg * inv_sum_neg
                        sample_val_1_neg = sample_child_pdf[2] * sample_val_1_neg * inv_sum_neg
                        sample_val_2_neg = sample_child_pdf[2] * sample_val_2_neg * inv_sum_neg
                        sample_val_3_neg = sample_child_pdf[2] * sample_val_3_neg * inv_sum_neg
                    else:
                        
                        
                        sample_val_0_prm = val_res[0 * sample_idx_scale + 0] / root_val_sum[0]
                        sample_val_1_prm = val_res[1 * sample_idx_scale + 0] / root_val_sum[0]
                        sample_val_2_prm = val_res[2 * sample_idx_scale + 0] / root_val_sum[0]
                        sample_val_3_prm = val_res[3 * sample_idx_scale + 0] / root_val_sum[0]
                        sample_val_0_pos = val_res[0 * sample_idx_scale + 1] / root_val_sum[1]
                        sample_val_1_pos = val_res[1 * sample_idx_scale + 1] / root_val_sum[1]
                        sample_val_2_pos = val_res[2 * sample_idx_scale + 1] / root_val_sum[1]
                        sample_val_3_pos = val_res[3 * sample_idx_scale + 1] / root_val_sum[1]
                        sample_val_0_neg = val_res[0 * sample_idx_scale + 2] / root_val_sum[2]
                        sample_val_1_neg = val_res[1 * sample_idx_scale + 2] / root_val_sum[2]
                        sample_val_2_neg = val_res[2 * sample_idx_scale + 2] / root_val_sum[2]
                        sample_val_3_neg = val_res[3 * sample_idx_scale + 2] / root_val_sum[2]

                    sample_val_0 = sample_val_0_prm * gw1\
                                 + sample_val_0_pos * gw21\
                                 + sample_val_0_neg * gw22 
                    sample_val_1 = sample_val_1_prm * gw1\
                                 + sample_val_1_pos * gw21\
                                 + sample_val_1_neg * gw22 
                    sample_val_2 = sample_val_2_prm * gw1\
                                 + sample_val_2_pos * gw21\
                                 + sample_val_2_neg * gw22 
                    sample_val_3 = sample_val_3_prm * gw1\
                                 + sample_val_3_pos * gw21\
                                 + sample_val_3_neg * gw22 
            else:
                assert self.mixture_count == 5
                
                
                sample_val_0 = dr.select(mask_eq3,
                                    val_res[0 * sample_idx_scale + 4],
                                    dr.select(mask_eq4, 
                                        val_res[0 * sample_idx_scale + 2] * gw_m_pos + val_res[0 * sample_idx_scale + 3] * gw_m_neg, 
                                        val_res[0 * sample_idx_scale + 0] * gw_s_pos + val_res[0 * sample_idx_scale + 1] * gw_s_neg)) 
                sample_val_1 = dr.select(mask_eq3,
                                    val_res[1 * sample_idx_scale + 4],
                                    dr.select(mask_eq4, 
                                        val_res[1 * sample_idx_scale + 2] * gw_m_pos + val_res[1 * sample_idx_scale + 3] * gw_m_neg, 
                                        val_res[1 * sample_idx_scale + 0] * gw_s_pos + val_res[1 * sample_idx_scale + 1] * gw_s_neg)) 
                sample_val_2 = dr.select(mask_eq3,
                                    val_res[2 * sample_idx_scale + 4],
                                    dr.select(mask_eq4, 
                                        val_res[2 * sample_idx_scale + 2] * gw_m_pos + val_res[2 * sample_idx_scale + 3] * gw_m_neg, 
                                        val_res[2 * sample_idx_scale + 0] * gw_s_pos + val_res[2 * sample_idx_scale + 1] * gw_s_neg)) 
                sample_val_3 = dr.select(mask_eq3,
                                    val_res[3 * sample_idx_scale + 4],
                                    dr.select(mask_eq4, 
                                        val_res[3 * sample_idx_scale + 2] * gw_m_pos + val_res[3 * sample_idx_scale + 3] * gw_m_neg, 
                                        val_res[3 * sample_idx_scale + 0] * gw_s_pos + val_res[3 * sample_idx_scale + 1] * gw_s_neg)) 
        

            sample_val_sum = sample_val_0 + sample_val_1 + sample_val_2 + sample_val_3

            sample_rnd = sample_rnd_src.next_1d() * sample_val_sum
            sample_child_id *= 0
            sample_child_id[sample_rnd >= sample_val_0] = 1
            sample_child_id[sample_rnd >= sample_val_0 + sample_val_1] = 2
            sample_child_id[sample_rnd >= sample_val_0 + sample_val_1 + sample_val_2] = 3

            sample_ans[1][sample_notleaf & (abs(sample_child_id - 1) < 0.5)] += power2_i
            sample_ans[1][sample_notleaf & (abs(sample_child_id - 3) < 0.5)] += power2_i
            sample_ans[0][sample_notleaf & (abs(sample_child_id - 2) < 0.5)] += power2_i
            sample_ans[0][sample_notleaf & (abs(sample_child_id - 3) < 0.5)] += power2_i

            coord_base[1][sample_notleaf & (abs(sample_child_id - 1) < 0.5)] += power2_i
            coord_base[1][sample_notleaf & (abs(sample_child_id - 3) < 0.5)] += power2_i
            coord_base[0][sample_notleaf & (abs(sample_child_id - 2) < 0.5)] += power2_i
            coord_base[0][sample_notleaf & (abs(sample_child_id - 3) < 0.5)] += power2_i

            ch_res = self.dtree_ch.eval_fetch((sample_dnode_id) / self.ndt)[1]
            for j in range(4):
                sample_notleaf[(abs(sample_child_id - j) < 0.5) & (dr.abs(sample_dnode_id + ch_res[j]) < sample_base + 0.5)] &= False 
                sample_dnode_id[(abs(sample_child_id - j) < 0.5) & sample_notleaf & sample_notleaf_last] = sample_dnode_id + ch_res[j]
            
            if enable_brute_force_product:
                sample_child_pdf[0][sample_notleaf & (abs(sample_child_id - 0) < 0.5)] *=  sample_val_0_prm * inv_sum_primal
                sample_child_pdf[0][sample_notleaf & (abs(sample_child_id - 1) < 0.5)] *=  sample_val_1_prm * inv_sum_primal
                sample_child_pdf[0][sample_notleaf & (abs(sample_child_id - 2) < 0.5)] *=  sample_val_2_prm * inv_sum_primal
                sample_child_pdf[0][sample_notleaf & (abs(sample_child_id - 3) < 0.5)] *=  sample_val_3_prm * inv_sum_primal
                sample_child_pdf[1][sample_notleaf & (abs(sample_child_id - 0) < 0.5)] *=  sample_val_0_pos * inv_sum_pos
                sample_child_pdf[1][sample_notleaf & (abs(sample_child_id - 1) < 0.5)] *=  sample_val_1_pos * inv_sum_pos
                sample_child_pdf[1][sample_notleaf & (abs(sample_child_id - 2) < 0.5)] *=  sample_val_2_pos * inv_sum_pos
                sample_child_pdf[1][sample_notleaf & (abs(sample_child_id - 3) < 0.5)] *=  sample_val_3_pos * inv_sum_pos
                sample_child_pdf[2][sample_notleaf & (abs(sample_child_id - 0) < 0.5)] *=  sample_val_0_neg * inv_sum_neg
                sample_child_pdf[2][sample_notleaf & (abs(sample_child_id - 1) < 0.5)] *=  sample_val_1_neg * inv_sum_neg
                sample_child_pdf[2][sample_notleaf & (abs(sample_child_id - 2) < 0.5)] *=  sample_val_2_neg * inv_sum_neg
                sample_child_pdf[2][sample_notleaf & (abs(sample_child_id - 3) < 0.5)] *=  sample_val_3_neg * inv_sum_neg

            sample_ans[0][sample_notleaf_last & ~sample_notleaf] += sample_rnd_src.next_1d() * power2_i
            sample_ans[1][sample_notleaf_last & ~sample_notleaf] += sample_rnd_src.next_1d() * power2_i
            sample_notleaf_last &= sample_notleaf
            i += 1
        
        
        return sample_ans

    def sample_dtree_wrapped(self, rng: mi.Sampler, sample_dnode_id: mi.Float, sample_method: int, bsdf: mi.BSDFPtr, si: mi.SurfaceInteraction3f, bsdf_ctx: mi.BSDFContext, alpha: mi.Float, β: mi.Float, β_df: mi.Float, record_outputname: str) -> mi.Vector3f:
        ans = canonical_to_dir(self.sample_dtree(rng, sample_dnode_id, sample_method, bsdf, si, bsdf_ctx, alpha, β, β_df, record_outputname))
        
        return ans
            
    def query_dtree(self, query_col_: mi.Point2f, query_dnode_id_: mi.Float, sample_method: int, bsdf: mi.BSDFPtr, si: mi.SurfaceInteraction3f, bsdf_ctx: mi.BSDFContext, alpha: mi.Float, β: mi.Float, β_df: mi.Float):
        
        query_col = query_col_ * 1
        query_dnode_id = query_dnode_id_ * 1

        query_ans = query_col[0] * 0 if self.mixture_count != 4 else mi.Vector4f(query_col[0] * 0)
        query_child_id = query_col[0] * 0 
        query_col_relative = mi.Point2f(query_col)
        query_base = query_dnode_id * 1
        notleaf = mi.Bool(True)
        notleaf_active = mi.Bool(True)

        coord_base = mi.Point2f(0, 0)
        sample_child_pdf = mi.Vector3f(1) 
        chain_pdf = mi.Float(1)   
        power2_i = mi.Float(1)

        enable_brute_force_product = type(sample_method)==int and sample_method == 2
        
        assert self.mix_dtree

        
        if self.mixture_count == 5:
            
            idx_scale = 5
            mask_eq3 = dr.eq(sample_method, 3)
            mask_eq4 = dr.eq(sample_method, 4)

            gw_m_pos = 1
            gw_m_neg = 1
            gw_s_pos = 1
            gw_s_neg = 1
            if self.enable_positivization:
                iter = self.positivization_iter
                assert iter == 0 or iter == 1
                if self.enable_positivization_pm:
                    if iter == 0:
                        gw_m_pos = 1 * swap_weight
                        gw_m_neg = 1 * (1 - swap_weight)
                    else:
                        gw_m_pos = 1 * (1 - swap_weight)
                        gw_m_neg = 1 * swap_weight
                if self.enable_positivization_ps:
                    if iter == 0:
                        gw_s_pos = 1 * swap_weight
                        gw_s_neg = 1 * (1 - swap_weight)
                    else:
                        gw_s_pos = 1 * (1 - swap_weight)
                        gw_s_neg = 1 * swap_weight
            elif self.enable_srb_pro:
                
                root_val = self.dtree_val.eval_fetch((query_dnode_id) / self.ndt)[1]
                sw = self.dtree_sw.eval_fetch((query_dnode_id) / self.ndt)[1] 
                
                gw_m_pos, gw_m_neg = self.get_pos_neg_ratio(
                    root_val[0 * idx_scale + 2] + root_val[1 * idx_scale + 2] + root_val[2 * idx_scale + 2] + root_val[3 * idx_scale + 2],
                    root_val[0 * idx_scale + 3] + root_val[1 * idx_scale + 3] + root_val[2 * idx_scale + 3] + root_val[3 * idx_scale + 3],
                    sw[1]
                )
                
                gw_s_pos, gw_s_neg = self.get_pos_neg_ratio(
                    root_val[0 * idx_scale + 0] + root_val[1 * idx_scale + 0] + root_val[2 * idx_scale + 0] + root_val[3 * idx_scale + 0],
                    root_val[0 * idx_scale + 1] + root_val[1 * idx_scale + 1] + root_val[2 * idx_scale + 1] + root_val[3 * idx_scale + 1],
                    sw[0]
                )
        else:
            assert self.mixture_count == 4
            if sample_method != 1:
                
                gw1 = 1 - self.guiding_mis_weight_ad   
                gw2 = self.guiding_mis_weight_ad

                if not enable_brute_force_product:
                    root_val = self.dtree_val.eval_fetch((query_dnode_id) / self.ndt)[1]
                    root_val_0 = mi.Vector4f(root_val[0:4])
                    root_val_1 = mi.Vector4f(root_val[4:8])
                    root_val_2 = mi.Vector4f(root_val[8:12])
                    root_val_3 = mi.Vector4f(root_val[12:16])
                    root_val_sum = root_val_0 + root_val_1 + root_val_2 + root_val_3 + 1e-18
                    sw = self.dtree_sw.eval_fetch((query_dnode_id) / self.ndt)[1] 

                if self.enable_prb_pro:
                    gw1, gw2 = self.get_weight_pro(root_val, sw, β, β_df)  

                ratio = 0.5
                if self.enable_positivization:                
                    ratio = swap_weight if self.positivization_iter == 0 else 1 - swap_weight
                elif self.enable_prb_pro:
                    ratio, _ = self.get_pos_neg_ratio(
                        root_val_0[1] + root_val_1[1] + root_val_2[1] + root_val_3[1],
                        root_val_0[2] + root_val_1[2] + root_val_2[2] + root_val_3[2],
                        sw[1]
                    )
                gw21 = gw2 * ratio
                gw22 = gw2 - gw21
                       
    
        i = mi.Float(0)
        count = mi.Int(0)

        def eval_grid(resolution, offset, eval_lambda):
            margin = power2_i / resolution / 2
            start = margin
            stop = power2_i - margin
            step = (stop - start) / (resolution - 1 if resolution > 1 else 1)
            pos = [start + step * i for i in range(resolution)]
            ans = 0
            for a, b in itertools.product(pos, pos):
                p = mi.Point2f(a, b)
                ans += eval_lambda(canonical_to_dir(p + offset))
            return ans / (resolution * resolution)

        loop = mi.Loop(name="query_dtree_loop", state=lambda: (count, query_col, query_dnode_id, query_ans,\
                                                                query_child_id, query_col_relative, query_base, notleaf, notleaf_active, i,\
                                                                coord_base, sample_child_pdf, power2_i, chain_pdf ))
        loop.set_max_iterations(10)
        while loop(notleaf):
        
            count += 1
            notleaf[count > 10] &= False
            query_dnode_id = dr.round(query_dnode_id)
            query_child_id = mi.Float(0)
            query_child_id[(query_col_relative[0] < 0.5) & (query_col_relative[1] >= 0.5)] = 1
            query_child_id[(query_col_relative[0] >= 0.5) & (query_col_relative[1] < 0.5)] = 2
            query_child_id[(query_col_relative[0] >= 0.5) & (query_col_relative[1] >= 0.5)] = 3

            query_col_relative = query_col_relative * 2
            query_col_relative[0][query_col_relative[0] >= 1] -= 1
            query_col_relative[1][query_col_relative[1] >= 1] -= 1

            ch_res = self.dtree_ch.eval_fetch((query_dnode_id) / self.ndt)[1]
            val_res = self.dtree_val.eval_fetch((query_dnode_id) / self.ndt)[1]
            power2_i *= 0.5

            if self.mixture_count == 4:
                
                if enable_brute_force_product:
                    
                    sample_idx_scale = 4
                
                    bsdf_lambda = lambda wo: dr.mean(bsdf.eval(bsdf_ctx, si, si.to_local(wo)))
                    bsdf_val_0 = eval_grid(self.grid_resolution, coord_base, bsdf_lambda)
                    bsdf_val_1 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(0, power2_i), bsdf_lambda)
                    bsdf_val_2 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(power2_i, 0), bsdf_lambda)
                    bsdf_val_3 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(power2_i), bsdf_lambda)

                    dbsdf_lambda = lambda wo: dbsdf_dalpha_finite(bsdf, bsdf_ctx, si, si.to_local(wo), alpha)
                    dbsdf_val_0 = eval_grid(self.grid_resolution, coord_base, dbsdf_lambda)
                    dbsdf_val_1 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(0, power2_i), dbsdf_lambda)
                    dbsdf_val_2 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(power2_i, 0), dbsdf_lambda)
                    dbsdf_val_3 = eval_grid(self.grid_resolution, coord_base + mi.Point2f(power2_i), dbsdf_lambda)

                    dbsdf_val_pos_0 = dr.select(dbsdf_val_0 > 0, dbsdf_val_0, 0)
                    dbsdf_val_neg_0 = dr.select(dbsdf_val_0 < 0, dbsdf_val_0, 0)
                    dbsdf_val_pos_1 = dr.select(dbsdf_val_1 > 0, dbsdf_val_1, 0)
                    dbsdf_val_neg_1 = dr.select(dbsdf_val_1 < 0, dbsdf_val_1, 0)
                    dbsdf_val_pos_2 = dr.select(dbsdf_val_2 > 0, dbsdf_val_2, 0)
                    dbsdf_val_neg_2 = dr.select(dbsdf_val_2 < 0, dbsdf_val_2, 0)
                    dbsdf_val_pos_3 = dr.select(dbsdf_val_3 > 0, dbsdf_val_3, 0)
                    dbsdf_val_neg_3 = dr.select(dbsdf_val_3 < 0, dbsdf_val_3, 0)
                    sample_val_0_prm = val_res[0 * sample_idx_scale + 0] * bsdf_val_0
                    sample_val_1_prm = val_res[1 * sample_idx_scale + 0] * bsdf_val_1
                    sample_val_2_prm = val_res[2 * sample_idx_scale + 0] * bsdf_val_2
                    sample_val_3_prm = val_res[3 * sample_idx_scale + 0] * bsdf_val_3

                    sample_val_0_pos = val_res[0 * sample_idx_scale + 0] * dbsdf_val_pos_0 
                    sample_val_1_pos = val_res[1 * sample_idx_scale + 0] * dbsdf_val_pos_1
                    sample_val_2_pos = val_res[2 * sample_idx_scale + 0] * dbsdf_val_pos_2
                    sample_val_3_pos = val_res[3 * sample_idx_scale + 0] * dbsdf_val_pos_3

                    sample_val_0_neg = -val_res[0 * sample_idx_scale + 0] * dbsdf_val_neg_0 
                    sample_val_1_neg = -val_res[1 * sample_idx_scale + 0] * dbsdf_val_neg_1
                    sample_val_2_neg = -val_res[2 * sample_idx_scale + 0] * dbsdf_val_neg_2
                    sample_val_3_neg = -val_res[3 * sample_idx_scale + 0] * dbsdf_val_neg_3

                    sum_primal = sample_val_0_prm + sample_val_1_prm + sample_val_2_prm + sample_val_3_prm
                    sum_pos = sample_val_0_pos + sample_val_1_pos + sample_val_2_pos + sample_val_3_pos
                    sum_neg = sample_val_0_neg + sample_val_1_neg + sample_val_2_neg + sample_val_3_neg

                    inv_sum_primal = dr.select(dr.neq(sum_primal, 0), dr.rcp(sum_primal), 0) 
                    inv_sum_pos = dr.select(dr.neq(sum_pos, 0), dr.rcp(sum_pos), 0) 
                    inv_sum_neg = dr.select(dr.neq(sum_neg, 0), dr.rcp(sum_neg), 0) 

                    sample_val_0 = sample_child_pdf[0] * sample_val_0_prm * inv_sum_primal * gw1\
                        +  sample_child_pdf[1] * sample_val_0_pos * inv_sum_pos * gw21\
                        +  sample_child_pdf[2] * sample_val_0_neg * inv_sum_neg * gw22 
                    sample_val_1 = sample_child_pdf[0] * sample_val_1_prm * inv_sum_primal * gw1\
                        +  sample_child_pdf[1] * sample_val_1_pos * inv_sum_pos * gw21\
                        +  sample_child_pdf[2] * sample_val_1_neg * inv_sum_neg * gw22 
                    sample_val_2 = sample_child_pdf[0] * sample_val_2_prm * inv_sum_primal * gw1\
                        +  sample_child_pdf[1] * sample_val_2_pos * inv_sum_pos * gw21\
                        +  sample_child_pdf[2] * sample_val_2_neg * inv_sum_neg * gw22 
                    sample_val_3 = sample_child_pdf[0] * sample_val_3_prm * inv_sum_primal * gw1\
                        +  sample_child_pdf[1] * sample_val_3_pos * inv_sum_pos * gw21\
                        +  sample_child_pdf[2] * sample_val_3_neg * inv_sum_neg * gw22 

                    primal_val_arr = [sample_val_0_prm, sample_val_1_prm, sample_val_2_prm, sample_val_3_prm]
                    pos_val_arr = [sample_val_0_pos, sample_val_1_pos, sample_val_2_pos, sample_val_3_pos]
                    neg_val_arr = [sample_val_0_neg, sample_val_1_neg, sample_val_2_neg, sample_val_3_neg]
                    pdf_val_arr = [sample_val_0, sample_val_1, sample_val_2, sample_val_3]

                    sample_val_sum = sample_val_0 + sample_val_1 + sample_val_2 + sample_val_3
                    inv_sample_val_sum = dr.select(dr.neq(sample_val_sum, 0), dr.rcp(sample_val_sum), 0) 

            coord_base[1][notleaf & (abs(query_child_id - 1) < 0.5)] += power2_i
            coord_base[1][notleaf & (abs(query_child_id - 3) < 0.5)] += power2_i
            coord_base[0][notleaf & (abs(query_child_id - 2) < 0.5)] += power2_i
            coord_base[0][notleaf & (abs(query_child_id - 3) < 0.5)] += power2_i

            for j in range(4): 
                notleaf[(abs(query_child_id - j) < 0.5) & (dr.abs(query_dnode_id + ch_res[j]) < query_base + 0.5)] &= False 
                if not self.mix_dtree:
                    query_ans[(abs(query_child_id - j) < 0.5) & notleaf_active] = val_res[j] * dr.power(4, i + 1)
                else:
                    if self.mixture_count == 4:
                        
                        if sample_method == 1:
                            
                            query_ans[0][(abs(query_child_id - j) < 0.5) & notleaf_active] = val_res[j*4+0] * dr.power(4, i + 1)
                        else:
                            if enable_brute_force_product:
                                
                                query_ans[0][(abs(query_child_id - j) < 0.5) & notleaf_active] = sample_child_pdf[0] * primal_val_arr[j] * inv_sum_primal * chain_pdf 
                                query_ans[1][(abs(query_child_id - j) < 0.5) & notleaf_active] = pos_val_arr[j]
                                query_ans[2][(abs(query_child_id - j) < 0.5) & notleaf_active] = neg_val_arr[j]
                                
                                query_ans[3][(abs(query_child_id - j) < 0.5) & notleaf_active] = pdf_val_arr[j] * inv_sample_val_sum * chain_pdf * 4
                            else:
                                
                                query_ans[3][(abs(query_child_id - j) < 0.5) & notleaf_active] =\
                                    ( val_res[j * 4 + 0] / root_val_sum[0] * gw1\
                                    + val_res[j * 4 + 1] / root_val_sum[1] * gw21\
                                    + val_res[j * 4 + 2] / root_val_sum[2] * gw22)\
                                    * dr.power(4, i + 1) 
                                
                    else:
                        
                        assert self.mixture_count == 5 
                        query_ans[(abs(query_child_id - j) < 0.5) & notleaf_active] =\
                                dr.select(mask_eq3,
                                    val_res[j * idx_scale + 4],
                                    dr.select(mask_eq4, 
                                        val_res[j * idx_scale + 2] * gw_m_pos + val_res[j * idx_scale + 3] * gw_m_neg, 
                                        val_res[j * idx_scale + 0] * gw_s_pos + val_res[j * idx_scale + 1] * gw_s_neg)) \
                                * dr.power(4, i + 1)
                query_dnode_id[(abs(query_child_id - j) < 0.5) & notleaf & notleaf_active] = query_dnode_id + ch_res[j]
            
            if enable_brute_force_product:
                sample_child_pdf[0][notleaf & (abs(query_child_id - 0) < 0.5)] *=  sample_val_0_prm * inv_sum_primal
                sample_child_pdf[0][notleaf & (abs(query_child_id - 1) < 0.5)] *=  sample_val_1_prm * inv_sum_primal
                sample_child_pdf[0][notleaf & (abs(query_child_id - 2) < 0.5)] *=  sample_val_2_prm * inv_sum_primal
                sample_child_pdf[0][notleaf & (abs(query_child_id - 3) < 0.5)] *=  sample_val_3_prm * inv_sum_primal
                sample_child_pdf[1][notleaf & (abs(query_child_id - 0) < 0.5)] *=  sample_val_0_pos * inv_sum_pos
                sample_child_pdf[1][notleaf & (abs(query_child_id - 1) < 0.5)] *=  sample_val_1_pos * inv_sum_pos
                sample_child_pdf[1][notleaf & (abs(query_child_id - 2) < 0.5)] *=  sample_val_2_pos * inv_sum_pos
                sample_child_pdf[1][notleaf & (abs(query_child_id - 3) < 0.5)] *=  sample_val_3_pos * inv_sum_pos
                sample_child_pdf[2][notleaf & (abs(query_child_id - 0) < 0.5)] *=  sample_val_0_neg * inv_sum_neg
                sample_child_pdf[2][notleaf & (abs(query_child_id - 1) < 0.5)] *=  sample_val_1_neg * inv_sum_neg
                sample_child_pdf[2][notleaf & (abs(query_child_id - 2) < 0.5)] *=  sample_val_2_neg * inv_sum_neg
                sample_child_pdf[2][notleaf & (abs(query_child_id - 3) < 0.5)] *=  sample_val_3_neg * inv_sum_neg
                chain_pdf[notleaf & (abs(query_child_id - 0) < 0.5)] *= sample_val_0 * inv_sample_val_sum * 4
                chain_pdf[notleaf & (abs(query_child_id - 1) < 0.5)] *= sample_val_1 * inv_sample_val_sum * 4
                chain_pdf[notleaf & (abs(query_child_id - 2) < 0.5)] *= sample_val_2 * inv_sample_val_sum * 4
                chain_pdf[notleaf & (abs(query_child_id - 3) < 0.5)] *= sample_val_3 * inv_sample_val_sum * 4

            notleaf_active &= notleaf

            i += 1
        
        
        return query_ans

    def query_dtree_wrapped(self, query_col: mi.Vector3f, query_dnode_id: mi.Float, sample_method: int, bsdf: mi.BSDFPtr, si: mi.SurfaceInteraction3f, bsdf_ctx: mi.BSDFContext, alpha: mi.Float, β: mi.Float, β_df: mi.Float):
        if (type(sample_method)==int and sample_method==2):
            ans = self.query_dtree(dir_to_canonical(query_col), query_dnode_id, sample_method, bsdf, si, bsdf_ctx, alpha, β, β_df) / (4 * 3.14159)
        else:
            root_val_sum = self.query_dtree_sum_wrapped(query_dnode_id, sample_method, β, β_df)
            ans = self.query_dtree(dir_to_canonical(query_col), query_dnode_id, sample_method, bsdf, si, bsdf_ctx, alpha, β, β_df) / root_val_sum / (4 * 3.14159)
        return ans

    def query_dtree_sum_wrapped(self, query_dnode_id: mi.Float, sample_method, β: mi.Float, β_df: mi.Float): 
        x = self.dtree_val.eval_fetch((query_dnode_id) / self.ndt)[1]
        if not self.mix_dtree:
            root_val_0 = x[0]
            root_val_1 = x[1]
            root_val_2 = x[2]
            root_val_3 = x[3]  
            
        else:
            enable_brute_force_product = type(sample_method)==int and sample_method == 2
            if self.mixture_count == 4:
                
                root_val_0 = mi.Vector4f(x[0:4])
                root_val_1 = mi.Vector4f(x[4:8])
                root_val_2 = mi.Vector4f(x[8:12])
                root_val_3 = mi.Vector4f(x[12:16])
                if sample_method != 1:
                    gw1 = 1 - self.guiding_mis_weight_ad   
                    gw2 = self.guiding_mis_weight_ad

                    if not enable_brute_force_product:
                        sw = self.dtree_sw.eval_fetch((query_dnode_id) / self.ndt)[1] 

                    if self.enable_prb_pro:
                        gw1, gw2 = self.get_weight_pro(x, sw, β, β_df)  
             
                    ratio = 0.5
                    if self.enable_positivization:                
                        ratio = swap_weight if self.positivization_iter == 0 else 1 - swap_weight
                    elif self.enable_prb_pro:
                        ratio, _ = self.get_pos_neg_ratio(
                            root_val_0[1] + root_val_1[1] + root_val_2[1] + root_val_3[1],
                            root_val_0[2] + root_val_1[2] + root_val_2[2] + root_val_3[2],
                            sw[1]
                        )
                    gw21 = gw2 * ratio
                    gw22 = gw2 - gw21

                    
                    idx_scale = 4
                    root_val_sum = root_val_0 + root_val_1 + root_val_2 + root_val_3 + 1e-18
                    root_val_0 = x[0 * idx_scale + 0] / root_val_sum[0] * gw1\
                        + x[0 * idx_scale + 1] / root_val_sum[1] * gw21\
                        + x[0 * idx_scale + 2] / root_val_sum[2] * gw22 
                    root_val_1 = x[1 * idx_scale + 0] / root_val_sum[0] * gw1\
                        + x[1 * idx_scale + 1] / root_val_sum[1] * gw21\
                        + x[1 * idx_scale + 2] / root_val_sum[2] * gw22 
                    root_val_2 = x[2 * idx_scale + 0] / root_val_sum[0] * gw1\
                        + x[2 * idx_scale + 1] / root_val_sum[1] * gw21\
                        + x[2 * idx_scale + 2] / root_val_sum[2] * gw22 
                    root_val_3 = x[3 * idx_scale + 0] / root_val_sum[0] * gw1\
                        + x[3 * idx_scale + 1] / root_val_sum[1] * gw21\
                        + x[3 * idx_scale + 2] / root_val_sum[2] * gw22 
                    
                    
            else:
                
                assert self.mixture_count == 5
                idx_scale = self.mixture_count
                gw_m_pos = 1
                gw_m_neg = 1
                gw_s_pos = 1
                gw_s_neg = 1
                if self.enable_positivization:
                    iter = self.positivization_iter
                    assert iter == 0 or iter == 1
                    if self.enable_positivization_pm:
                        if iter == 0:
                            gw_m_pos = 1 * swap_weight
                            gw_m_neg = 1 * (1 - swap_weight)
                        else:
                            gw_m_pos = 1 * (1 - swap_weight)
                            gw_m_neg = 1 * swap_weight
                    if self.enable_positivization_ps:
                        if iter == 0:
                            gw_s_pos = 1 * swap_weight
                            gw_s_neg = 1 * (1 - swap_weight)
                        else:
                            gw_s_pos = 1 * (1 - swap_weight)
                            gw_s_neg = 1 * swap_weight
                elif self.enable_srb_pro:
                    
                    root_val = x
                    sw = self.dtree_sw.eval_fetch((query_dnode_id) / self.ndt)[1] 
                    
                    gw_m_pos, gw_m_neg = self.get_pos_neg_ratio(
                        root_val[0 * idx_scale + 2] + root_val[1 * idx_scale + 2] + root_val[2 * idx_scale + 2] + root_val[3 * idx_scale + 2],
                        root_val[0 * idx_scale + 3] + root_val[1 * idx_scale + 3] + root_val[2 * idx_scale + 3] + root_val[3 * idx_scale + 3],
                        sw[1]
                    )

                    
                    gw_s_pos, gw_s_neg = self.get_pos_neg_ratio(
                        root_val[0 * idx_scale + 0] + root_val[1 * idx_scale + 0] + root_val[2 * idx_scale + 0] + root_val[3 * idx_scale + 0],
                        root_val[0 * idx_scale + 1] + root_val[1 * idx_scale + 1] + root_val[2 * idx_scale + 1] + root_val[3 * idx_scale + 1],
                        sw[0]
                    )                
                    
                mask_eq3 = dr.eq(sample_method, 3)
                mask_eq4 = dr.eq(sample_method, 4)
                root_val_0 = dr.select(mask_eq3,
                                x[0 * idx_scale + 4],
                                dr.select(mask_eq4, 
                                    x[0 * idx_scale + 2] * gw_m_pos + x[0 * idx_scale + 3] * gw_m_neg, 
                                    x[0 * idx_scale + 0] * gw_s_pos + x[0 * idx_scale + 1] * gw_s_neg))
                root_val_1 = dr.select(mask_eq3,
                                x[1 * idx_scale + 4],
                                dr.select(mask_eq4, 
                                    x[1 * idx_scale + 2] * gw_m_pos + x[1 * idx_scale + 3] * gw_m_neg, 
                                    x[1 * idx_scale + 0] * gw_s_pos + x[1 * idx_scale + 1] * gw_s_neg))
                root_val_2 = dr.select(mask_eq3,
                                x[2 * idx_scale + 4],
                                dr.select(mask_eq4, 
                                    x[2 * idx_scale + 2] * gw_m_pos + x[2 * idx_scale + 3] * gw_m_neg, 
                                    x[2 * idx_scale + 0] * gw_s_pos + x[2 * idx_scale + 1] * gw_s_neg))
                root_val_3 = dr.select(mask_eq3,
                                x[3 * idx_scale + 4],
                                dr.select(mask_eq4, 
                                    x[3 * idx_scale + 2] * gw_m_pos + x[3 * idx_scale + 3] * gw_m_neg, 
                                    x[3 * idx_scale + 0] * gw_s_pos + x[3 * idx_scale + 1] * gw_s_neg))
            
        root_val_sum = root_val_0 + root_val_1 + root_val_2 + root_val_3 + 1e-18
        ans = root_val_sum

        
        return ans

    def query_dtree_stats_weight(self, query_dnode_id: mi.Float) -> mi.Float:
        ans = self.dtree_sw.eval_fetch((query_dnode_id) / self.ndt)[1][0]
        
        return ans
    
    def get_rr_prob(self, query_dnode_id: mi.Float) -> mi.Float:
        
        val = self.dtree_val.eval_fetch((query_dnode_id) / self.ndt)[1] 
        sw = self.dtree_sw.eval_fetch((query_dnode_id) / self.ndt)[1] 

        idx_scale = 5
        es_val = val[0 * idx_scale + 0] + val[0 * idx_scale + 1]\
                + val[1 * idx_scale + 0] + val[1 * idx_scale + 1]\
                + val[2 * idx_scale + 0] + val[2 * idx_scale + 1]\
                + val[3 * idx_scale + 0] + val[3 * idx_scale + 1]
        em_val = val[0 * idx_scale + 2] + val[0 * idx_scale + 3]\
                + val[1 * idx_scale + 2] + val[1 * idx_scale + 3]\
                + val[2 * idx_scale + 2] + val[2 * idx_scale + 3]\
                + val[3 * idx_scale + 2] + val[3 * idx_scale + 3]

        es_sw = sw[0]
        em_sw = sw[1]

        es = dr.select(dr.neq(es_sw, 0), es_val / es_sw, 0)
        em = dr.select(dr.neq(em_sw, 0), em_val / em_sw, 0)
        prob = (es + 1e-9) / (es + em + 2e-9)
        return prob

    def get_weight_pro(self, dtree_val, sw_val, β, β_df):
        assert self.mixture_count == 4
        idx_scale = 4
        sample_val_0_prm = dtree_val[0 * idx_scale + 0]
        sample_val_1_prm = dtree_val[1 * idx_scale + 0]
        sample_val_2_prm = dtree_val[2 * idx_scale + 0]
        sample_val_3_prm = dtree_val[3 * idx_scale + 0]
        sample_val_0_pos = dtree_val[0 * idx_scale + 1] 
        sample_val_1_pos = dtree_val[1 * idx_scale + 1]
        sample_val_2_pos = dtree_val[2 * idx_scale + 1]
        sample_val_3_pos = dtree_val[3 * idx_scale + 1]
        sample_val_0_neg = dtree_val[0 * idx_scale + 2] 
        sample_val_1_neg = dtree_val[1 * idx_scale + 2]
        sample_val_2_neg = dtree_val[2 * idx_scale + 2]
        sample_val_3_neg = dtree_val[3 * idx_scale + 2]

        sum_primal = sample_val_0_prm + sample_val_1_prm + sample_val_2_prm + sample_val_3_prm
        sum_pos = sample_val_0_pos + sample_val_1_pos + sample_val_2_pos + sample_val_3_pos
        sum_neg = sample_val_0_neg + sample_val_1_neg + sample_val_2_neg + sample_val_3_neg

        esem_val = sum_pos + sum_neg 
        el_val = sum_primal

        esem = dr.select(dr.neq(sw_val[1], 0), esem_val / sw_val[1], 0)
        el = dr.select(dr.neq(sw_val[0], 0), el_val / sw_val[0], 0)

        gw1 = β_df * el
        gw2 = β * esem

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        return gw1, gw2
    
    def get_pos_neg_ratio(self, dtree_pos_sum, dtree_neg_sum, sw_sum):
        pos_mean = dr.select(dr.neq(sw_sum, 0), dtree_pos_sum / sw_sum, 0)
        neg_mean = dr.select(dr.neq(sw_sum, 0), dtree_neg_sum / sw_sum, 0)
        ratio = (pos_mean + 1e-9) / (pos_mean + neg_mean + 2e-9)
        return ratio, 1 - ratio


class ADIntegrator(mi.CppADIntegrator):
    """
    Abstract base class of numerous differentiable integrators in Mitsuba

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (Default: 6)
     * - rr_depth
       - |int|
       - Specifies the path depth, at which the implementation will begin to use
         the *russian roulette* path termination criterion. For example, if set to
         1, then path generation many randomly cease after encountering directly
         visible surfaces. (Default: 5)
    """

    def __init__(self, props = mi.Properties()):
        super().__init__(props)

        max_depth = props.get('max_depth', 6)
        if max_depth < 0 and max_depth != -1:
            raise Exception("\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        
        self.max_depth = max_depth if max_depth != -1 else 0xffffffff

        self.rr_depth = props.get('rr_depth', 5)
        if self.rr_depth <= 0:
            raise Exception("\"rr_depth\" must be set to a value greater than zero!")

        
        self.sample_border_warning = True

        
        self.sdtree_primal = None
        self.sdtree_adjoint = None
        self.sdtree_ad_pos = None
        self.sdtree_ad_neg = None

        self.sdtree_mix = None

        self.enable_ssguiding = False 
        self.guiding_films = [None, None]
        self.guiding_film_idx = -1


    def restore_config(self):
        self.sdtree_primal = None
        self.sdtree_adjoint = None
        self.sdtree_ad_pos = None
        self.sdtree_ad_neg = None

        self.sdtree_mix = None

        self.enable_ssguiding = False 
        self.guiding_films = [None, None]
        self.guiding_film_idx = -1

    def aovs(self):
        return []

    def to_string(self):
        return f'{type(self).__name__}[max_depth = {self.max_depth},' \
               f' rr_depth = { self.rr_depth }]'

    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:

        self.sdtree_primal = None

        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        
        with dr.suspend_grad():
            
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aovs()
            )

            
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)

            
            L, valid, state = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                reparam=None,
                active=mi.Bool(True)
            )

            
            block = film.create_block()

            
            block.set_coalesce(block.coalesce() and spp >= 4)

            
            ADIntegrator._splat_to_block(block, film, pos, 
                                         value=L*weight,
                                         weight=1.0,
                                         alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                                         wavelengths=ray.wavelengths)

            
            del sampler, ray, weight, pos, L, valid
            gc.collect()

            
            film.put_block(block)
            self.primal_image = film.develop()

            return self.primal_image

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        
        with dr.suspend_grad():
            
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            
            
            
            if hasattr(self, 'reparam'):
                reparam = _ReparamWrapper(
                    scene=scene,
                    params=params,
                    reparam=self.reparam,
                    wavefront_size=sampler.wavefront_size(),
                    seed=seed
                )
            else:
                reparam = None

            
            
            ray, weight, pos, det = self.sample_rays(scene, sensor,
                                                     sampler, reparam)

            with dr.resume_grad():
                L, valid, _ = self.sample(
                    mode=dr.ADMode.Forward,
                    scene=scene,
                    sampler=sampler,
                    ray=ray,
                    reparam=reparam,
                    active=mi.Bool(True)
                )

                block = film.create_block()
                
                block.set_coalesce(block.coalesce() and spp >= 4)

                
                
                
                
                
                ADIntegrator._splat_to_block(block, film, pos,
                                             value=L*weight*det,
                                             weight=det,
                                             alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                                             wavelengths=ray.wavelengths)

                
                film.put_block(block)
                result_img = film.develop()

                dr.forward_to(result_img)

        return dr.grad(result_img)

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        
        with dr.suspend_grad():
            
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            
            
            
            if hasattr(self, 'reparam'):
                reparam = _ReparamWrapper(
                    scene=scene,
                    params=params,
                    reparam=self.reparam,
                    wavefront_size=sampler.wavefront_size(),
                    seed=seed
                )
            else:
                reparam = None

            
            
            ray, weight, pos, det = self.sample_rays(scene, sensor,
                                                     sampler, reparam)

            with dr.resume_grad():
                L, valid, _ = self.sample(
                    mode=dr.ADMode.Backward,
                    scene=scene,
                    sampler=sampler,
                    ray=ray,
                    reparam=reparam,
                    active=mi.Bool(True)
                )

                
                block = film.create_block()

                
                block.set_coalesce(block.coalesce() and spp >= 4)

                
                ADIntegrator._splat_to_block(block, film, pos,
                                value=L*weight*det,
                                weight=det,
                                alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                                wavelengths=ray.wavelengths)

                film.put_block(block)

                del valid
                gc.collect()

                
                dr.schedule(block.tensor())
                image = film.develop()

                
                
                dr.set_grad(image, grad_in)
                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(mi.Float, dr.ADMode.Backward)

            
            del ray, weight, pos, block, sampler
            gc.collect()

            
            dr.eval()


    def update_film_distr(self, idx, w, h, spp, result_film):
        if self.enable_ssguiding:
            assert idx == 0 or idx == 1
            p_film = prob_film(w, h, result_film)
            self.guiding_films[idx] = make_film_distr(w, h, spp, p_film)

    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        reparam: Callable[[mi.Ray3f, mi.UInt32, mi.Bool],
                          Tuple[mi.Vector3f, mi.Float]] = None
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
        """
        Sample a 2D grid of primary rays for a given sensor

        Returns a tuple containing

        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous 2D image-space positions associated with each ray

        When a reparameterization function is provided via the 'reparam'
        argument, it will be applied to the returned image-space position (i.e.
        the sample positions will be moving). The other two return values
        remain detached.
        """


        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()

        
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        original_idx = mi.UInt32(idx)

        
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        
        pos = mi.Vector2i()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(-film_size[0], pos.y, idx)

        
        if self.enable_ssguiding:
            assert self.guiding_film_idx == 0 or self.guiding_film_idx == 1
            if self.guiding_films[self.guiding_film_idx] is not None:
                t_begin_ss = time.time()
                pos = self.guiding_films[self.guiding_film_idx] 
                t_end_ss = time.time()
                print("    ssguiderd uses %.3f sec" % (t_end_ss - t_begin_ss))

        if film.sample_border():
            pos -= border_size

        pos += mi.Vector2i(film.crop_offset())

        
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        
        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time_ = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time_ += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time_,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )

        reparam_det = 1.0

        if reparam is not None:
            if rfilter.is_box_filter():
                raise Exception(
                    "ADIntegrator detected the potential for image-space "
                    "motion due to differentiable shape or camera pose "
                    "parameters. This is, however, incompatible with the box "
                    "reconstruction filter that is currently used. Please "
                    "specify a smooth reconstruction filter in your scene "
                    "description (e.g. 'gaussian', which is actually the "
                    "default)")

            
            if not film.sample_border() and self.sample_border_warning:
                self.sample_border_warning = True

                mi.Log(mi.LogLevel.Warn,
                    "ADIntegrator detected the potential for image-space "
                    "motion due to differentiable shape or camera pose "
                    "parameters. To correctly account for shapes entering "
                    "or leaving the viewport, it is recommended that you set "
                    "the film's 'sample_border' parameter to True.")

            with dr.resume_grad():
                
                reparam_d, reparam_det = reparam(ray=dr.detach(ray),
                                                 depth=mi.UInt32(0))

                
                
                if dr.grad_enabled(ray.o):
                    reparam_d, _ = reparam(ray=ray, depth=mi.UInt32(0))

                
                
                it = dr.zeros(mi.Interaction3f)
                it.p = ray.o + reparam_d
                ds, _ = sensor.sample_direction(it, aperture_sample)

                
                pos_f = ds.uv + film.crop_offset()

        
        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

        return ray, weight, splatting_pos, reparam_det

    def prepare(self,
                sensor: mi.Sensor,
                seed: int = 0,
                spp: int = 0,
                aovs: list = []):
        """
        Given a sensor and a desired number of samples per pixel, this function
        computes the necessary number of Monte Carlo samples and then suitably
        seeds the sampler underlying the sensor.

        Returns the created sampler and the final number of samples per pixel
        (which may differ from the requested amount depending on the type of
        ``Sampler`` being used)

        Parameter ``sensor`` (``int``, ``mi.Sensor``):
            Specify a sensor to render the scene from a different viewpoint.

        Parameter ``seed` (``int``)
            This parameter controls the initialization of the random number
            generator during the primal rendering step. It is crucial that you
            specify different seeds (e.g., an increasing sequence) if subsequent
            calls should produce statistically independent images (e.g. to
            de-correlate gradient-based optimization steps).

        Parameter ``spp`` (``int``):
            Optional parameter to override the number of samples per pixel for the
            primal rendering step. The value provided within the original scene
            specification takes precedence if ``spp=0``.
        """

        film = sensor.film()
        sampler = sensor.sampler().clone()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        film_size = film.crop_size()

        if film.sample_border():
            film_size += 2 * film.rfilter().border_size()

        wavefront_size = dr.prod(film_size) * spp

        if wavefront_size > 2**32:
            raise Exception(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size)

        sampler.seed(seed, wavefront_size)
        film.prepare(aovs)

        return sampler, spp

    def _splat_to_block(block: mi.ImageBlock,
                       film: mi.Film,
                       pos: mi.Point2f,
                       value: mi.Spectrum,
                       weight: mi.Float,
                       alpha: mi.Float,
                       wavelengths: mi.Spectrum):
        '''Helper function to splat values to a imageblock'''
        if (dr.all(mi.has_flag(film.flags(), mi.FilmFlags.Special))):
            aovs = film.prepare_sample(value, wavelengths,
                                        block.channel_count(),
                                        weight=weight,
                                        alpha=alpha)
            block.put(pos, aovs)
            del aovs
        else:
            block.put(
                pos=pos,
                wavelengths=wavelengths,
                value=value,
                weight=weight,
                alpha=alpha
            )

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               depth: mi.UInt32,
               δL: Optional[mi.Spectrum],
               state_in: Any,
               reparam: Optional[
                   Callable[[mi.Ray3f, mi.UInt32, mi.Bool],
                            Tuple[mi.Vector3f, mi.Float]]],
               active: mi.Bool) -> Tuple[mi.Spectrum, mi.Bool]:
        """
        This function does the main work of differentiable rendering and
        remains unimplemented here. It is provided by subclasses of the
        ``RBIntegrator`` interface.

        In those concrete implementations, the function performs a Monte Carlo
        random walk, implementing a number of different behaviors depending on
        the ``mode`` argument. For example in primal mode (``mode ==
        drjit.ADMode.Primal``), it behaves like a normal rendering algorithm
        and estimates the radiance incident along ``ray``.

        In forward mode (``mode == drjit.ADMode.Forward``), it estimates the
        derivative of the incident radiance for a set of scene parameters being
        differentiated. (This requires that these parameters are attached to
        the AD graph and have gradients specified via ``dr.set_grad()``)

        In backward mode (``mode == drjit.ADMode.Backward``), it takes adjoint
        radiance ``δL`` and accumulates it into differentiable scene parameters.

        You are normally *not* expected to directly call this function. Instead,
        use ``mi.render()`` , which performs various necessary
        setup steps to correctly use the functionality provided here.

        The parameters of this function are as follows:

        Parameter ``mode`` (``drjit.ADMode``)
            Specifies whether the rendering algorithm should run in primal or
            forward/backward derivative propagation mode

        Parameter ``scene`` (``mi.Scene``):
            Reference to the scene being rendered in a differentiable manner.

        Parameter ``sampler`` (``mi.Sampler``):
            A pre-seeded sample generator

        Parameter ``depth`` (``mi.UInt32``):
            Path depth of `ray` (typically set to zero). This is mainly useful
            for forward/backward differentiable rendering phases that need to
            obtain an incident radiance estimate. In this case, they may
            recursively invoke ``sample(mode=dr.ADMode.Primal)`` with a nonzero
            depth.

        Parameter ``δL`` (``mi.Spectrum``):
            When back-propagating gradients (``mode == drjit.ADMode.Backward``)
            the ``δL`` parameter should specify the adjoint radiance associated
            with each ray. Otherwise, it must be set to ``None``.

        Parameter ``state_in`` (``Any``):
            The primal phase of ``sample()`` returns a state vector as part of
            its return value. The forward/backward differential phases expect
            that this state vector is provided to them via this argument. When
            invoked in primal mode, it should be set to ``None``.

        Parameter ``reparam`` (see above):
            If provided, this callable takes a ray and a mask of active SIMD
            lanes and returns a reparameterized ray and Jacobian determinant.
            The implementation of the ``sample`` function should then use it to
            correctly account for visibility-induced discontinuities during
            differentiation.

        Parameter ``active`` (``mi.Bool``):
            This mask array can optionally be used to indicate that some of
            the rays are disabled.

        The function returns a tuple ``(spec, valid, state_out)`` where

        Output ``spec`` (``mi.Spectrum``):
            Specifies the estimated radiance and differential radiance in
            primal and forward mode, respectively.

        Output ``valid`` (``mi.Bool``):
            Indicates whether the rays intersected a surface, which can be used
            to compute an alpha channel.
        """

        raise Exception('RBIntegrator does not provide the sample() method. '
                        'It should be implemented by subclasses that '
                        'specialize the abstract RBIntegrator interface.')


class RBIntegrator(ADIntegrator):
    """
    Abstract base class of radiative-backpropagation style differentiable
    integrators.
    """

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        """
        Evaluates the forward-mode derivative of the rendering step.

        Forward-mode differentiation propagates gradients from scene parameters
        through the simulation, producing a *gradient image* (i.e., the derivative
        of the rendered image with respect to those scene parameters). The gradient
        image is very helpful for debugging, for example to inspect the gradient
        variance or visualize the region of influence of a scene parameter. It is
        not particularly useful for simultaneous optimization of many parameters,
        since multiple differentiation passes are needed to obtain separate
        derivatives for each scene parameter. See ``Integrator.render_backward()``
        for an efficient way of obtaining all parameter derivatives at once, or
        simply use the ``mi.render()`` abstraction that hides both
        ``Integrator.render_forward()`` and ``Integrator.render_backward()`` behind
        a unified interface.

        Before calling this function, you must first enable gradient tracking and
        furthermore associate concrete input gradients with one or more scene
        parameters, or the function will just return a zero-valued gradient image.
        This is typically done by invoking ``dr.enable_grad()`` and
        ``dr.set_grad()`` on elements of the ``SceneParameters`` data structure
        that can be obtained obtained via a call to
        ``mi.traverse()``.

        Parameter ``scene`` (``mi.Scene``):
            The scene to be rendered differentially.

        Parameter ``params``:
           An arbitrary container of scene parameters that should receive
           gradients. Typically this will be an instance of type
           ``mi.SceneParameters`` obtained via ``mi.traverse()``. However, it
           could also be a Python list/dict/object tree (DrJit will traverse it
           to find all parameters). Gradient tracking must be explicitly enabled
           for each of these parameters using ``dr.enable_grad(params['parameter_name'])``
           (i.e. ``render_forward()`` will not do this for you). Furthermore,
           ``dr.set_grad(...)`` must be used to associate specific gradient values
           with each parameter.

        Parameter ``sensor`` (``int``, ``mi.Sensor``):
            Specify a sensor or a (sensor index) to render the scene from a
            different viewpoint. By default, the first sensor within the scene
            description (index 0) will take precedence.

        Parameter ``seed` (``int``)
            This parameter controls the initialization of the random number
            generator. It is crucial that you specify different seeds (e.g., an
            increasing sequence) if subsequent calls should produce statistically
            independent images (e.g. to de-correlate gradient-based optimization
            steps).

        Parameter ``spp`` (``int``):
            Optional parameter to override the number of samples per pixel for the
            differential rendering step. The value provided within the original
            scene specification takes precedence if ``spp=0``.
        """

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        
        
        with dr.suspend_grad():
            
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            
            
            
            if hasattr(self, 'reparam'):
                reparam = _ReparamWrapper(
                    scene=scene,
                    params=params,
                    reparam=self.reparam,
                    wavefront_size=sampler.wavefront_size(),
                    seed=seed
                )
            else:
                reparam = None

            
            
            ray, weight, pos, det = self.sample_rays(scene, sensor,
                                                     sampler, reparam)

            
            L, valid, state_out = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler.clone(),
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                reparam=None,
                active=mi.Bool(True)
            )

            
            δL, valid_2, state_out_2 = self.sample(
                mode=dr.ADMode.Forward,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=state_out,
                reparam=reparam,
                active=mi.Bool(True)
            )


            
            
            
            
            
            
            
            
            
            

            sample_pos_deriv = None 

            with dr.resume_grad():
                if dr.grad_enabled(pos):
                    sample_pos_deriv = film.create_block()

                    
                    sample_pos_deriv.set_coalesce(sample_pos_deriv.coalesce() and spp >= 4)

                    
                    ADIntegrator._splat_to_block(sample_pos_deriv, film, pos,
                                                 value=L*weight*det,
                                                 weight=det,
                                                 alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                                                 wavelengths=ray.wavelengths)

                    
                    tensor = sample_pos_deriv.tensor()
                    dr.forward_to(tensor, flags=dr.ADFlag.ClearInterior | dr.ADFlag.ClearEdges)

                    dr.schedule(tensor, dr.grad(tensor))

                    
                    dr.disable_grad(pos)
                    del tensor

            
            block = film.create_block()

            
            block.set_coalesce(block.coalesce() and spp >= 4)

            
            ADIntegrator._splat_to_block(block, film, pos,
                                         value=δL * weight,
                                         weight=1.0,
                                         alpha=dr.select(valid_2, mi.Float(1), mi.Float(0)),
                                         wavelengths=ray.wavelengths)

            
            film.put_block(block)

            
            del sampler, ray, weight, pos, L, valid, δL, valid_2, params, \
                state_out, state_out_2, block

            gc.collect()

            result_grad = film.develop()

            
            if sample_pos_deriv is not None:
                with dr.resume_grad():
                    film.clear()
                    
                    film.put_block(sample_pos_deriv)
                    reparam_result = film.develop()
                    dr.forward_to(reparam_result)
                    result_grad += dr.grad(reparam_result)

        return result_grad

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:
        """
        Evaluates the reverse-mode derivative of the rendering step.

        Reverse-mode differentiation transforms image-space gradients into scene
        parameter gradients, enabling simultaneous optimization of scenes with
        millions of free parameters. The function is invoked with an input
        *gradient image* (``grad_in``) and transforms and accumulates these into
        the gradient arrays of scene parameters that previously had gradient
        tracking enabled.

        Before calling this function, you must first enable gradient tracking for
        one or more scene parameters, or the function will not do anything. This is
        typically done by invoking ``dr.enable_grad()`` on elements of the
        ``SceneParameters`` data structure that can be obtained obtained via a call
        to ``mi.traverse()``. Use ``dr.grad()`` to query the
        resulting gradients of these parameters once ``render_backward()`` returns.

        Parameter ``scene`` (``mi.Scene``):
            The scene to be rendered differentially.

        Parameter ``params``:
           An arbitrary container of scene parameters that should receive
           gradients. Typically this will be an instance of type
           ``mi.SceneParameters`` obtained via ``mi.traverse()``. However, it
           could also be a Python list/dict/object tree (DrJit will traverse it
           to find all parameters). Gradient tracking must be explicitly enabled
           for each of these parameters using ``dr.enable_grad(params['parameter_name'])``
           (i.e. ``render_backward()`` will not do this for you).

        Parameter ``grad_in`` (``mi.TensorXf``):
            Gradient image that should be back-propagated.

        Parameter ``sensor`` (``int``, ``mi.Sensor``):
            Specify a sensor or a (sensor index) to render the scene from a
            different viewpoint. By default, the first sensor within the scene
            description (index 0) will take precedence.

        Parameter ``seed` (``int``)
            This parameter controls the initialization of the random number
            generator. It is crucial that you specify different seeds (e.g., an
            increasing sequence) if subsequent calls should produce statistically
            independent images (e.g. to de-correlate gradient-based optimization
            steps).

        Parameter ``spp`` (``int``):
            Optional parameter to override the number of samples per pixel for the
            differential rendering step. The value provided within the original
            scene specification takes precedence if ``spp=0``.
        """

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        
        with dr.suspend_grad():
            
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            
            
            
            if hasattr(self, 'reparam'):
                reparam = _ReparamWrapper(
                    scene=scene,
                    params=params,
                    reparam=self.reparam,
                    wavefront_size=sampler.wavefront_size(),
                    seed=seed
                )
            else:
                reparam = None

            
            
            ray, weight, pos, det = self.sample_rays(scene, sensor,
                                                     sampler, reparam)

            
            
            with dr.resume_grad():
                with dr.suspend_grad(pos, det, ray, weight):
                    L = dr.full(mi.Spectrum, 1.0, dr.width(ray))
                    dr.enable_grad(L)

                    RBIntegrator._splatting_and_backward_gradient_image(film, pos, ray, spp,
                        value=L * weight,
                        weight=1.0,
                        alpha=1.0,
                        grad_in=grad_in
                    )

                    δL = dr.grad(L)

            
            film.clear()

            
            L, valid, state_out = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler.clone(),
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                reparam=None,
                active=mi.Bool(True)
            )

            
            L_2, valid_2, state_out_2 = self.sample(
                mode=dr.ADMode.Backward,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=δL,
                state_in=state_out,
                reparam=reparam,
                active=mi.Bool(True)
            )

            
            if reparam is not None:
                with dr.resume_grad():
                    
                    
                    
                    
                    
                    RBIntegrator._splatting_and_backward_gradient_image(film, pos, ray, spp,
                        value=L * weight * det,
                        weight=det,
                        alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                        grad_in=grad_in
                    )

            
            del L_2, valid_2, state_out, state_out_2, δL, \
                ray, weight, pos, sampler

            gc.collect()

            
            dr.eval()

    
    
    def _splatting_and_backward_gradient_image(film, pos, ray, spp,
                                                value: mi.Spectrum,
                                                weight: mi.Float,
                                                alpha: mi.Float,
                                                grad_in):
        '''
        Backward propagation of the gradient image through the sample
        splatting and weight division steps.
        '''

        
        block = film.create_block()

        
        block.set_coalesce(block.coalesce() and spp >= 4)     

        ADIntegrator._splat_to_block(block, film, pos,
                                    value=value,
                                    weight=weight,
                                    alpha=alpha,
                                    wavelengths=ray.wavelengths)

        film.put_block(block)

        
        
        
        gc.collect()

        image = film.develop()

        
        
        dr.set_grad(image, grad_in)
        dr.enqueue(dr.ADMode.Backward, image)
        dr.traverse(mi.Float, dr.ADMode.Backward)





def render_forward(self: mi.Integrator,
                   scene: mi.Scene,
                   params: Any,
                   sensor: Union[int, mi.Sensor] = 0,
                   seed: int = 0,
                   spp: int = 0) -> mi.TensorXf:
    """
    Evaluates the forward-mode derivative of the rendering step.

    Forward-mode differentiation propagates gradients from scene parameters
    through the simulation, producing a *gradient image* (i.e., the derivative
    of the rendered image with respect to those scene parameters). The gradient
    image is very helpful for debugging, for example to inspect the gradient
    variance or visualize the region of influence of a scene parameter. It is
    not particularly useful for simultaneous optimization of many parameters,
    since multiple differentiation passes are needed to obtain separate
    derivatives for each scene parameter. See ``Integrator.render_backward()``
    for an efficient way of obtaining all parameter derivatives at once, or
    simply use the ``mi.render()`` abstraction that hides both
    ``Integrator.render_forward()`` and ``Integrator.render_backward()`` behind
    a unified interface.

    Before calling this function, you must first enable gradient tracking and
    furthermore associate concrete input gradients with one or more scene
    parameters, or the function will just return a zero-valued gradient image.
    This is typically done by invoking ``dr.enable_grad()`` and
    ``dr.set_grad()`` on elements of the ``SceneParameters`` data structure
    that can be obtained obtained via a call to ``mi.traverse()``.

    Note the default implementation of this functionality relies on naive
    automatic differentiation (AD), which records a computation graph of the
    primal rendering step that is subsequently traversed to propagate
    derivatives. This tends to be relatively inefficient due to the need to
    track intermediate program state. In particular, it means that
    differentiation of nontrivial scenes at high sample counts will often run
    out of memory. Integrators like ``rb`` (Radiative Backpropagation) and
    ``prb`` (Path Replay Backpropagation) that are specifically designed for
    differentiation can be significantly more efficient.

    Parameter ``scene`` (``mi.Scene``):
        The scene to be rendered differentially.

    Parameter ``params``:
       An arbitrary container of scene parameters that should receive
       gradients. Typically this will be an instance of type
       ``mi.SceneParameters`` obtained via ``mi.traverse()``. However, it could
       also be a Python list/dict/object tree (DrJit will traverse it to find
       all parameters). Gradient tracking must be explicitly enabled for each of
       these parameters using ``dr.enable_grad(params['parameter_name'])`` (i.e.
       ``render_forward()`` will not do this for you). Furthermore,
       ``dr.set_grad(...)`` must be used to associate specific gradient values
       with each parameter.

    Parameter ``sensor`` (``int``, ``mi.Sensor``):
        Specify a sensor or a (sensor index) to render the scene from a
        different viewpoint. By default, the first sensor within the scene
        description (index 0) will take precedence.

    Parameter ``seed` (``int``)
        This parameter controls the initialization of the random number
        generator. It is crucial that you specify different seeds (e.g., an
        increasing sequence) if subsequent calls should produce statistically
        independent images (e.g. to de-correlate gradient-based optimization
        steps).

    Parameter ``spp`` (``int``):
        Optional parameter to override the number of samples per pixel for the
        differential rendering step. The value provided within the original
        scene specification takes precedence if ``spp=0``.
    """

    
    
    
    
    

    
    with dr.scoped_set_flag(dr.JitFlag.LoopRecord, False):
        image = self.render(
            scene=scene,
            sensor=sensor,
            seed=seed,
            spp=spp,
            develop=True,
            evaluate=False
        )

        
        
        dr.forward_to(image)

        return dr.grad(image)

def render_backward(self: mi.Integrator,
                    scene: mi.Scene,
                    params: Any,
                    grad_in: mi.TensorXf,
                    sensor: Union[int, mi.Sensor] = 0,
                    seed: int = 0,
                    spp: int = 0) -> None:
    """
    Evaluates the reverse-mode derivative of the rendering step.

    Reverse-mode differentiation transforms image-space gradients into scene
    parameter gradients, enabling simultaneous optimization of scenes with
    millions of free parameters. The function is invoked with an input
    *gradient image* (``grad_in``) and transforms and accumulates these into
    the gradient arrays of scene parameters that previously had gradient
    tracking enabled.

    Before calling this function, you must first enable gradient tracking for
    one or more scene parameters, or the function will not do anything. This is
    typically done by invoking ``dr.enable_grad()`` on elements of the
    ``SceneParameters`` data structure that can be obtained obtained via a call
    to ``mi.traverse()``. Use ``dr.grad()`` to query the resulting gradients of
    these parameters once ``render_backward()`` returns.

    Note the default implementation of this functionality relies on naive
    automatic differentiation (AD), which records a computation graph of the
    primal rendering step that is subsequently traversed to propagate
    derivatives. This tends to be relatively inefficient due to the need to
    track intermediate program state. In particular, it means that
    differentiation of nontrivial scenes at high sample counts will often run
    out of memory. Integrators like ``rb`` (Radiative Backpropagation) and
    ``prb`` (Path Replay Backpropagation) that are specifically designed for
    differentiation can be significantly more efficient.

    Parameter ``scene`` (``mi.Scene``):
        The scene to be rendered differentially.

    Parameter ``params``:
       An arbitrary container of scene parameters that should receive
       gradients. Typically this will be an instance of type
       ``mi.SceneParameters`` obtained via ``mi.traverse()``. However, it could
       also be a Python list/dict/object tree (DrJit will traverse it to find
       all parameters). Gradient tracking must be explicitly enabled for each of
       these parameters using ``dr.enable_grad(params['parameter_name'])`` (i.e.
       ``render_backward()`` will not do this for you).

    Parameter ``grad_in`` (``mi.TensorXf``):
        Gradient image that should be back-propagated.

    Parameter ``sensor`` (``int``, ``mi.Sensor``):
        Specify a sensor or a (sensor index) to render the scene from a
        different viewpoint. By default, the first sensor within the scene
        description (index 0) will take precedence.

    Parameter ``seed` (``int``)
        This parameter controls the initialization of the random number
        generator. It is crucial that you specify different seeds (e.g., an
        increasing sequence) if subsequent calls should produce statistically
        independent images (e.g. to de-correlate gradient-based optimization
        steps).

    Parameter ``spp`` (``int``):
        Optional parameter to override the number of samples per pixel for the
        differential rendering step. The value provided within the original
        scene specification takes precedence if ``spp=0``.
    """

    
    with dr.scoped_set_flag(dr.JitFlag.LoopRecord, False):
        image = self.render(
            scene=scene,
            sensor=sensor,
            seed=seed,
            spp=spp,
            develop=True,
            evaluate=False
        )

        
        dr.backward_from(image * grad_in)


mi.Integrator.render_backward = render_backward
mi.Integrator.render_forward = render_forward

del render_backward
del render_forward



class _ReparamWrapper:
    """
    This class is an implementation detail of ``ADIntegrator``, which performs
    necessary initialization steps and subsequently wraps a reparameterization
    technique. It serves the following important purposes:

    1. Ensuring the availability of uncorrelated random variates.
    2. Connecting reparameterization calls to relevant shape-related
       variables in the AD graph.
    3. Exposing the underlying RNG state to recorded loops.
    """

    
    
    DRJIT_STRUCT = { 'rng' : mi.PCG32 }

    def __init__(self,
                 scene : mi.Scene,
                 params: Any,
                 reparam: Callable[
                     [mi.Scene, mi.PCG32, Any,
                      mi.Ray3f, mi.UInt32, mi.Bool],
                     Tuple[mi.Vector3f, mi.Float]],
                 wavefront_size : int,
                 seed : int):

        self.scene = scene
        self.params = params
        self.reparam = reparam

        
        
        
        if isinstance(params, mi.SceneParameters):
            params = params.copy()
            params.keep(
                [
                    k for k in params.keys() \
                        if (params.flags(k) & mi.ParamFlags.Discontinuous) != 0
                ]
            )

        
        
        

        idx = dr.arange(mi.UInt32, wavefront_size)
        tmp = dr.opaque(mi.UInt32, 0xffffffff ^ seed)
        v0, v1 = mi.sample_tea_32(tmp, idx)
        self.rng = mi.PCG32(initstate=v0, initseq=v1)

    def __call__(self,
                 ray: mi.Ray3f,
                 depth: mi.UInt32,
                 active: Union[mi.Bool, bool] = True
    ) -> Tuple[mi.Vector3f, mi.Float]:
        """
        This function takes a ray, a path depth value (to potentially disable
        reparameterizations after a certain number of bounces) and a boolean
        active mask as input and returns the reparameterized ray direction and
        the Jacobian determinant of the change of variables.
        """
        return self.reparam(self.scene, self.rng, self.params, ray,
                            depth, active)






def mis_weight(pdf_a, pdf_b):
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    b2 = dr.sqr(pdf_b)
    w = a2 / (a2 + b2)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))
