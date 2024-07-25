from __future__ import annotations 

import drjit as dr
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import time 
from .common import ADIntegrator, RBIntegrator, MySDTree, prob_film, _ReparamWrapper
import os 
import gc
import pprint
import matplotlib.cm as cm
import scipy.ndimage



def save_tmp_samples():
	os.system("copy guiding_samples.txt guiding_samples_0.txt >nul")
	os.system("copy guiding_samples_adjoint.txt guiding_samples_adjoint_0.txt >nul")
def merge_tmp_samples():
	os.system("copy /b guiding_samples_0.txt+guiding_samples.txt guiding_samples_1.txt >nul")
	os.system("copy /b guiding_samples_adjoint_0.txt+guiding_samples_adjoint.txt guiding_samples_adjoint_1.txt >nul")
	os.system("copy guiding_samples_1.txt guiding_samples.txt >nul")
	os.system("copy guiding_samples_adjoint_1.txt guiding_samples_adjoint.txt >nul")








data_swap_time = 0
rendering_time = 0
class BasicPRBModIntegrator(RBIntegrator):

	def __init__(self, props = mi.Properties()):
		super().__init__(props)

		self.param_key = None
		self.texture_id = None  
		self.grad_out = None
		self.target_channel_one = False

		self.target_func_full = True 
		self.sdtree_type = 4  
		self.is_primal_phase = True
		self.target_bounce = 0
		self.cv_alpha = 0.0
		self.dtree_sample_method = 0  
		self.guiding_mis_weight_ad = 0.5
		self.enable_positivization = False
		self.positivization_iter = -1
		self.change_seed = False
		self.bsdf_roughness_threshold = 0.2
		self.guiding_frac = 0.5

		self.val_backward = None 
		self.opt_iter = 0
		self.require_record = True
		self.require_load_sdtree = True

		self.record_ratio = False  
		self.outputfile_prefix = ''

		self.guide_primal_phase_only = False

		self.need_reparam = False 
		
		self.reparam_max_depth = self.max_depth
		self.reparam_rays = 16
		self.reparam_kappa = 1e5
		self.reparam_exp = 3.0
		self.reparam_antithetic = False
		self.reparam_unroll = False

	def restore_config(self):	
		super().restore_config()
		self.target_func_full = True
		self.sdtree_type = 4 
		self.is_primal_phase = True
		self.target_bounce = 0
		self.cv_alpha = 0.0
		self.dtree_sample_method = 0
		self.guiding_mis_weight_ad = 0.5
		self.enable_positivization = False
		self.positivization_iter = -1
		self.change_seed = False
		self.bsdf_roughness_threshold = 0.2
		self.guiding_frac = 0.5
		self.opt_iter = 0
		self.require_record = True
		self.record_ratio = False
		self.outputfile_prefix = ''
		self.guide_primal_phase_only = False
		self.need_reparam = False
		self.reparam_max_depth = self.max_depth
		self.reparam_rays = 16
		self.reparam_kappa = 1e5
		self.reparam_exp = 3.0
		self.reparam_antithetic = False
		self.reparam_unroll = False
	
	
	
	def set_config(self, **kwargs):
		def handle_conversion(key: str, value):
			if key == 'enable_product_sampling':
				key = 'dtree_sample_method'
				assert type(value) == bool
				value = 2 if value == True else 0
			elif key == 'enable_prb_pro':
				key = 'dtree_sample_method' 
				assert type(value) == bool
				value = 6 if value == True else 0
			elif key == 'target_super':
				key = 'target_func_full' 
				assert type(value) == int  
				value = bool(value)
			elif key == 'target_fuse':
				key = 'sdtree_type' 
			return key, value
		
		for (key, value) in kwargs.items():
			key, value = handle_conversion(key, value)
			if hasattr(self, key):
				assert type(value) == type(getattr(self, key))
				setattr(self, key, value)
			else:
				if key == 'mesh_key':
					self.need_reparam = True
					continue
				print(f'[integrator cfg] Ignore unknown parameter: {key}={value}')

	def print_config(self):
		print('====== integrator config begin==================')
		
		pprint.pprint(self.__dict__, indent=2, sort_dicts=False)
		print('====== integrator config end ===================')

	def sample(self,
			   mode: dr.ADMode,
			   scene: mi.Scene,
			   sampler: mi.Sampler, 
			   ray: mi.Ray3f,
			   δL: Optional[mi.Spectrum],
			   state_in: Optional[mi.Spectrum],
			   active: mi.Bool,
			   params: Optional[mi.SceneParameters] = None,
			   **kwargs 
	) -> Tuple[mi.Spectrum,
			   mi.Bool, mi.Spectrum]:
		
		primal = mode == dr.ADMode.Primal
		bsdf_ctx = mi.BSDFContext()
		ray = mi.Ray3f(ray)
		depth = mi.UInt32(0)                          
		L = mi.Spectrum(0 if primal else state_in)    
		δL = mi.Spectrum(δL if δL is not None else 0) 
		β = mi.Spectrum(1)                            
		active = mi.Bool(active)                      
		δrec = mi.Float(0)

		
		assert(self.sdtree_type == 4)
		
		wall_clock = time.time()
		
		apply_to_single_bounce = self.target_bounce >= 0  
		

		depth_lim = self.max_depth  
		assert depth_lim >= 1
		record_size_delta = dr.shape(ray.o)[1]  
		record_level = (1 if apply_to_single_bounce else depth_lim)
		record_size = record_size_delta * record_level
		if self.sdtree_mix is not None:
			self.sdtree_mix.guiding_mis_weight_ad = self.guiding_mis_weight_ad 
			self.sdtree_mix.enable_prb_pro = not self.is_primal_phase and self.dtree_sample_method == 6
			self.sdtree_mix.spp = sampler.sample_count()
		
		outputname = self.outputfile_prefix + 'ourprb_' + str(mode).split('.')[1] + '_depth'

		
		
		
		
		
		
		record_pos = [dr.zeros(dr.llvm.ad.Array3f, record_size_delta) for _ in range(record_level)] 
		record_dir = [dr.zeros(dr.llvm.ad.Array3f, record_size_delta) for _ in range(record_level)]
		record_beta = [dr.zeros(dr.llvm.ad.Array3f, record_size_delta) for _ in range(record_level)]
		record_spec = [dr.zeros(dr.llvm.ad.Array3f, record_size_delta) for _ in range(record_level)]
		record_sum = [dr.zeros(dr.llvm.ad.Array3f, record_size_delta) for _ in range(record_level)]
		record_val = [dr.zeros(dr.llvm.ad.Float, record_size_delta) for _ in range(record_level)]

		enable_prb_pro = not self.is_primal_phase and self.dtree_sample_method == 6
		β_df = mi.Spectrum(0)

		sdtree_valid_epsilon = 1e-18
		
		
		
		
		
		
		for i in range(depth_lim):
			
			apply_to_this_bounce = i == self.target_bounce or not apply_to_single_bounce
			should_guide = True
			if self.guide_primal_phase_only and not self.is_primal_phase:
				should_guide = False
			record_idx = 0 if apply_to_single_bounce else i 

			with dr.resume_grad(when=not primal):
				si = scene.ray_intersect(ray)
				Le = β * si.emitter(scene).eval(si)
			active_next = si.is_valid() 
			bsdf = si.bsdf(ray)

			

			bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
												   sampler.next_1d(),
												   sampler.next_2d(),
												   active_next)
			bsdf_sample.wo = si.to_world(bsdf_sample.wo)
			bsdf_dir = bsdf_sample.wo
			bsdf_pdf = bsdf_sample.pdf
			bsdf_pdf = dr.select(dr.isfinite(bsdf_pdf), bsdf_pdf, 0)
			

			
			
			
			
			

			is_valid_guide = mi.Bool(True)
			guide_dir = mi.Vector3f()
			dtree_mix = None
			if self.sdtree_type == 4 and self.sdtree_mix is not None and apply_to_this_bounce and should_guide:
				dtree = self.sdtree_mix.query_stree_wrapped(si.p, scene.bbox())
				dtree_mix = mi.Float(dtree)
				

				guide_dir = self.sdtree_mix.sample_dtree_wrapped(sampler, mi.Float(dtree), 1 if self.is_primal_phase else self.dtree_sample_method, bsdf, si, bsdf_ctx, bsdf_sample.alpha, dr.mean(β), dr.mean(dr.abs(β_df)), outputname + str(i))	
				
				
				

			
			
			
			
			
			
			
			
			
			
			

			
			
			
			
			
			
			
			
			
			
			

			
			

			
			

			
			
			if self.sdtree_mix is not None and apply_to_this_bounce and should_guide:
				active_surface = active_next & (bsdf_sample.alpha > self.bsdf_roughness_threshold)

				frac_guide = dr.select(is_valid_guide & active_surface, self.guiding_frac, 0.0)

				mix_dir = dr.select(sampler.next_1d() < frac_guide, guide_dir, bsdf_dir)
				mix_dir_local = si.to_local(mix_dir)

				
				if self.is_primal_phase:
					guide_pdf = self.sdtree_mix.query_dtree_wrapped(mix_dir, dtree_mix, 1, bsdf, si, bsdf_ctx, bsdf_sample.alpha, dr.mean(β), dr.mean(dr.abs(β_df)))[0]
				else:
					query_record = self.sdtree_mix.query_dtree_wrapped(mix_dir, dtree_mix * 1, self.dtree_sample_method, bsdf, si, bsdf_ctx, bsdf_sample.alpha, dr.mean(β), dr.mean(dr.abs(β_df)))
					if self.dtree_sample_method != 2:
						guide_pdf = query_record[3]
						
						
						
						
						
						
						
						
					else:
						guide_pdf = query_record[3]
				

				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				

				mix_value = dr.select(active_surface, bsdf.eval(bsdf_ctx, si, mix_dir_local, active_surface), bsdf_weight * bsdf_pdf)
				bsdf_pdf = dr.select(active_surface, bsdf.pdf(bsdf_ctx, si, mix_dir_local, active_surface), bsdf_pdf)

				mix_pdf = bsdf_pdf * (1.0 - frac_guide) + dr.select(dr.isfinite(guide_pdf), guide_pdf, 0) * frac_guide
				
				bsdf_dir = mix_dir
				bsdf_pdf = mix_pdf
				bsdf_sample.pdf = mix_pdf
				bsdf_sample.wo = mix_dir
				bsdf_weight = mix_value * dr.select(dr.neq(mix_pdf, 0), dr.rcp(mix_pdf), 0)

			L = L + Le if primal else L - Le
			ray = si.spawn_ray(bsdf_sample.wo)
			β_before = β * 1
			β *= bsdf_weight
			active_next &= dr.any(dr.neq(β, 0)) 

			
			
			
			
			

			
			active_record = si.is_valid() & (bsdf_sample.alpha > self.bsdf_roughness_threshold)
			if apply_to_this_bounce:
				if primal:
					record_spec[record_idx] = dr.select(si.is_valid(), L if apply_to_single_bounce else Le, record_spec[record_idx])
					
				record_pos[record_idx] = dr.select(active_record, si.p, record_pos[record_idx])
				record_dir[record_idx] = dr.select(active_record, ray.d, record_dir[record_idx])

				record_beta[record_idx] = dr.select(si.is_valid(), β_before, record_beta[record_idx])

			if not self.is_primal_phase:
				if (not primal and apply_to_this_bounce) or enable_prb_pro:
					with dr.resume_grad():
						wo = si.to_local(ray.d)
						bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next)
						if not primal:
							bsdf_val_detach = bsdf_weight * bsdf_sample.pdf
							inv_bsdf_val_detach = dr.select(dr.neq(bsdf_val_detach, 0),
															dr.rcp(bsdf_val_detach), 0)
							Lr = L * dr.replace_grad(1, inv_bsdf_val_detach * bsdf_val)
							Lo = Le + Lr

						if mode == dr.ADMode.Forward:
							tmp = dr.forward_to(Lo, flags=dr.ADFlag.Default if not enable_prb_pro else dr.ADFlag.ClearEdges)
						elif mode == dr.ADMode.Backward:
							param_tensor: mi.TensorXf = params.get(self.param_key)	
							dr.forward(param_tensor, flags=dr.ADFlag.Default if not enable_prb_pro else dr.ADFlag.ClearEdges)
							tmp = dr.grad(Lo)
						else:
							
							dr.forward_to(bsdf_val, flags=dr.ADFlag.ClearEdges) 

						if enable_prb_pro:
							dbsdf = dr.grad(bsdf_val)
							β_df = β_before * dbsdf * dr.select(dr.neq(bsdf_sample.pdf, 0), dr.rcp(bsdf_sample.pdf), 0) + β_df * bsdf_weight

						
						

				
				if not primal and apply_to_this_bounce:
					record_spec[record_idx] = dr.select(active_record, tmp, record_spec[record_idx])

					if mode == dr.ADMode.Backward:
						
						
				
						if 'normal' not in self.param_key:
							val = tmp
						else:
							dL_df = L*inv_bsdf_val_detach
							val = bsdf.compute_normalmap_deriv(bsdf_ctx, si, si.to_local(ray.d),  dL_df, si.is_valid())   

						if self.enable_positivization:
							assert self.positivization_iter == 0 or self.positivization_iter == 1
							if self.positivization_iter == 0:
								val = dr.maximum(val, 0)
							else:
								val = dr.minimum(val, 0)
						
						val = dr.select(si.is_valid(), δL * val, 0)
						if self.enable_ssguiding:
							if apply_to_single_bounce:
								self.val_backward = val 
							else:
								if i == 0:
									self.val_backward = val 
								else:
									self.val_backward += val 
						if self.target_channel_one:
							val = dr.sum(val)
						index = bsdf.get_texel_index(si, wo, self.texture_id, si.is_valid())
						dr.scatter_reduce(dr.ReduceOp.Add, self.grad_out.array, val, index, si.is_valid())
					else:
						val = tmp
						if self.enable_positivization:
							assert self.positivization_iter == 0 or self.positivization_iter == 1
							if self.positivization_iter == 0:
								val = dr.maximum(val, 0)
							else:
								val = dr.minimum(val, 0)
						
						δL += val 
					
						
			depth[si.is_valid()] += 1
			active = active_next


		if apply_to_single_bounce:
			if self.target_func_full and not primal:
				record_val[0] = dr.sum(record_spec[0] / record_beta[0]) * mi.Float(kwargs['inv_spp']) / 3 
			else:
				record_val[0] = dr.sum((L - record_spec[0]) / record_beta[0]) / 3 * mi.Float(kwargs['inv_spp']) 
		else:
			
			record_sum[0] = record_spec[0]
			for j in range(1, record_level):
				record_sum[j] = record_spec[j] + record_sum[j - 1]
			record_sum_all = record_sum[record_level - 1] 
			for j in range(record_level):
				beta = record_beta[j]

				
				if self.target_func_full and not primal:
					tmp = dr.sum((record_sum_all - record_sum[j] + record_spec[j]) / beta) / 3		
					record_val[j] = tmp * mi.Float(kwargs['inv_spp'])
				else:
					tmp = dr.sum((record_sum_all - record_sum[j]) / beta) / 3 
					record_val[j] = tmp * mi.Float(kwargs['inv_spp'])


		kernel_output = dr.zeros(dr.llvm.ad.TensorXf, shape=(7 * record_level + 3, record_size_delta))
		if self.require_record:
			for i in range(record_level):
				kernel_output[7 * i + 0] = record_pos[i][0]
				kernel_output[7 * i + 1] = record_pos[i][1]
				kernel_output[7 * i + 2] = record_pos[i][2]
				kernel_output[7 * i + 3] = record_dir[i][0]
				kernel_output[7 * i + 4] = record_dir[i][1]
				kernel_output[7 * i + 5] = record_dir[i][2]
				kernel_output[7 * i + 6] = record_val[i]		

		if primal:
			L = dr.clamp(L, 0, 1e9)
		else:
			δL = dr.clamp(δL, -1e9, 1e9)
		kernel_output[7 * record_level + 0] = L[0] if primal else δL[0]
		kernel_output[7 * record_level + 1] = L[1] if primal else δL[1]
		kernel_output[7 * record_level + 2] = L[2] if primal else δL[2]

		
		record_arr = np.array(kernel_output[:7*record_level], dtype=np.float64).T

		if primal:
			L = dr.llvm.ad.Array3f(dr.llvm.ad.Float(dr.ravel(kernel_output[-3])),dr.llvm.ad.Float(dr.ravel(kernel_output[-2])),dr.llvm.ad.Float(dr.ravel(kernel_output[-1])))
		else:
			δL = dr.llvm.ad.Array3f(dr.llvm.ad.Float(dr.ravel(kernel_output[-3])), dr.llvm.ad.Float(dr.ravel(kernel_output[-2])), dr.llvm.ad.Float(dr.ravel(kernel_output[-1])))


		assert np.shape(record_arr) == (record_size_delta, 7*record_level)
		
		record_arr = np.row_stack([record_arr[:, i*7:(i+1)*7] for i in range(record_level)])
		assert np.shape(record_arr) == (record_size, 7)

		record_arr = record_arr[np.linalg.norm(record_arr[:,3:6],axis=-1)>0]
		
		record_arr = record_arr[np.abs(record_arr[:,-1])<1e19]

		global rendering_time
		rendering_time += time.time() - wall_clock
		print("    rendering uses %.3f sec, total %.3f sec" % (time.time() - wall_clock, rendering_time))
		
		np.set_printoptions(threshold=np.inf) 
		time1 = time.time()


		
		
		
		
		
		
		
		
		

		if primal:
			with open('guiding_samples.txt','wb') as f:  
				record_arr.tofile(f)  
		else:
			with open('guiding_samples_adjoint.txt','wb') as f:
				record_arr.tofile(f)
		
		global data_swap_time
		time1 = time.time() - time1
		data_swap_time += time1
		

		return (
			L if primal else δL, 
			dr.neq(depth, 0),    
			L                    
		)
	
	def reparam(self,
				scene: mi.Scene,
				rng: mi.PCG32,
				params: Any,
				ray: mi.Ray3f,
				depth: mi.UInt32,
				active: mi.Bool):
		"""
		Helper function to reparameterize rays internally and within ADIntegrator
		"""

		
		if self.reparam_max_depth == 0:
			return dr.detach(ray.d, True), mi.Float(1)

		active = active & (depth < self.reparam_max_depth)

		return mi.ad.reparameterize_ray(scene, rng, params, ray,
										num_rays=self.reparam_rays,
										kappa=self.reparam_kappa,
										exponent=self.reparam_exp,
										antithetic=self.reparam_antithetic,
										unroll=self.reparam_unroll,
										active=active)

	
	def sample_reparam(self,
				mode: dr.ADMode,
				scene: mi.Scene,
				sampler: mi.Sampler,
				ray: mi.Ray3f,
				δL: Optional[mi.Spectrum],
				state_in: Optional[mi.Spectrum],
				reparam: Optional[
					Callable[[mi.Ray3f, mi.Bool],
							Tuple[mi.Ray3f, mi.Float]]],
				active: mi.Bool,
				params: Optional[mi.SceneParameters] = None,
				**kwargs 
	) -> Tuple[mi.Spectrum, mi.Bool, mi.Spectrum]:
	
		
		primal = mode == dr.ADMode.Primal
		bsdf_ctx = mi.BSDFContext()
		
		depth = mi.UInt32(0)                          
		L = mi.Spectrum(0 if primal else state_in)    
		δL = mi.Spectrum(δL if δL is not None else 0) 
		β = mi.Spectrum(1)                            
		active = mi.Bool(active)                      
		δrec = mi.Float(0)

		ray_prev = dr.zeros(mi.Ray3f)
		ray_cur  = mi.Ray3f(dr.detach(ray))
		pi_prev  = dr.zeros(mi.PreliminaryIntersection3f)
		pi_cur   = scene.ray_intersect_preliminary(ray_cur, coherent=True, active=active)

		
		assert(self.sdtree_type == 4)
		
		wall_clock = time.time()
		
		apply_to_single_bounce = self.target_bounce >= 0  
		

		depth_lim = self.max_depth  
		assert depth_lim >= 1
		record_size_delta = dr.shape(ray.o)[1]  
		record_level = (1 if apply_to_single_bounce else depth_lim)
		record_size = record_size_delta * record_level
		if self.sdtree_mix is not None:
			self.sdtree_mix.guiding_mis_weight_ad = self.guiding_mis_weight_ad 
			self.sdtree_mix.enable_prb_pro = not self.is_primal_phase and self.dtree_sample_method == 6
			self.sdtree_mix.spp = sampler.sample_count()
		
		outputname = self.outputfile_prefix + 'ourprb_' + str(mode).split('.')[1] + '_depth'

		record_pos = [dr.zeros(dr.llvm.ad.Array3f, record_size_delta) for _ in range(record_level)] 
		record_dir = [dr.zeros(dr.llvm.ad.Array3f, record_size_delta) for _ in range(record_level)]
		record_beta = [dr.zeros(dr.llvm.ad.Array3f, record_size_delta) for _ in range(record_level)]
		record_spec = [dr.zeros(dr.llvm.ad.Array3f, record_size_delta) for _ in range(record_level)]
		record_sum = [dr.zeros(dr.llvm.ad.Array3f, record_size_delta) for _ in range(record_level)]
		record_val = [dr.zeros(dr.llvm.ad.Float, record_size_delta) for _ in range(record_level)]

		enable_prb_pro = False 
		β_df = mi.Spectrum(0)

		sdtree_valid_epsilon = 1e-18
		for i in range(depth_lim):
			first_vertex = dr.eq(depth, 0)
			ray_reparam = mi.Ray3f(ray_cur)
			ray_reparam_det = 1

			
			if not primal:
				with dr.resume_grad():
					
					
					si_prev = pi_prev.compute_surface_interaction(
						ray_prev, mi.RayFlags.All | mi.RayFlags.FollowShape)

					
					
					ray_reparam.d, ray_reparam_det = reparam(
						dr.select(first_vertex, ray_cur,
									si_prev.spawn_ray(ray_cur.d)), depth)
					ray_reparam_det[first_vertex] = 1
					
					dr.disable_grad(si_prev)

			
			apply_to_this_bounce = i == self.target_bounce or not apply_to_single_bounce
			should_guide = True
			if self.guide_primal_phase_only and not self.is_primal_phase:
				should_guide = False
			record_idx = 0 if apply_to_single_bounce else i 

			with dr.resume_grad(when=not primal):
				si_cur = pi_cur.compute_surface_interaction(ray_reparam)
				Le = β * si_cur.emitter(scene).eval(si_cur)
			active_next = si_cur.is_valid() 
			bsdf = si_cur.bsdf(ray_cur)

			

			bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si_cur,
												   sampler.next_1d(),
												   sampler.next_2d(),
												   active_next)
			bsdf_sample.wo = si_cur.to_world(bsdf_sample.wo)
			bsdf_dir = bsdf_sample.wo
			bsdf_pdf = bsdf_sample.pdf
			bsdf_pdf = dr.select(dr.isfinite(bsdf_pdf), bsdf_pdf, 0)
		
			is_valid_guide = mi.Bool(True)
			guide_dir = mi.Vector3f()
			dtree_mix = None
			if self.sdtree_mix is not None and apply_to_this_bounce and should_guide:
				dtree = self.sdtree_mix.query_stree_wrapped(si_cur.p, scene.bbox())
				dtree_mix = mi.Float(dtree)
				
				guide_dir = self.sdtree_mix.sample_dtree_wrapped(sampler, mi.Float(dtree), 1 if self.is_primal_phase else self.dtree_sample_method, bsdf, si_cur, bsdf_ctx, bsdf_sample.alpha, dr.mean(β), dr.mean(dr.abs(β_df)), outputname + str(i))	
				

				active_surface = active_next & (bsdf_sample.alpha > self.bsdf_roughness_threshold)

				frac_guide = dr.select(is_valid_guide & active_surface, self.guiding_frac, 0.0)

				mix_dir = dr.select(sampler.next_1d() < frac_guide, guide_dir, bsdf_dir)
				mix_dir_local = si_cur.to_local(mix_dir)

				
				if self.is_primal_phase:
					guide_pdf = self.sdtree_mix.query_dtree_wrapped(mix_dir, dtree_mix, 1, bsdf, si_cur, bsdf_ctx, bsdf_sample.alpha, dr.mean(β), dr.mean(dr.abs(β_df)))[0]
				else:
					query_record = self.sdtree_mix.query_dtree_wrapped(mix_dir, dtree_mix * 1, self.dtree_sample_method, bsdf, si_cur, bsdf_ctx, bsdf_sample.alpha, dr.mean(β), dr.mean(dr.abs(β_df)))
					guide_pdf = query_record[3]

				mix_value = dr.select(active_surface, bsdf.eval(bsdf_ctx, si_cur, mix_dir_local, active_surface), bsdf_weight * bsdf_pdf)
				bsdf_pdf = dr.select(active_surface, bsdf.pdf(bsdf_ctx, si_cur, mix_dir_local, active_surface), bsdf_pdf)

				mix_pdf = bsdf_pdf * (1.0 - frac_guide) + dr.select(dr.isfinite(guide_pdf), guide_pdf, 0) * frac_guide
				
				bsdf_sample.pdf = mix_pdf
				bsdf_sample.wo = mix_dir
				bsdf_weight = mix_value * dr.select(dr.neq(mix_pdf, 0), dr.rcp(mix_pdf), 0)

			L_prev = L * 1
			L = (L + Le) if primal else (L - Le)

			β_before = β * 1
			β *= bsdf_weight
			active_next &= dr.any(dr.neq(dr.max(β), 0)) 

			ray_next = si_cur.spawn_ray(bsdf_sample.wo)
			pi_next = scene.ray_intersect_preliminary(ray_next, active=active_next)
			si_next = pi_next.compute_surface_interaction(ray_next)

			
			active_record = si_cur.is_valid() & (bsdf_sample.alpha > self.bsdf_roughness_threshold)
			if apply_to_this_bounce:
				if primal:
					record_spec[record_idx] = dr.select(si_cur.is_valid(), L if apply_to_single_bounce else Le, record_spec[record_idx])
					
				record_pos[record_idx] = dr.select(active_record, si_cur.p, record_pos[record_idx])
				record_dir[record_idx] = dr.select(active_record, ray_next.d, record_dir[record_idx])

				record_beta[record_idx] = dr.select(si_cur.is_valid(), β_before, record_beta[record_idx])

			if not self.is_primal_phase:
				if (not primal and apply_to_this_bounce) or enable_prb_pro:
					
					
					sampler_clone = sampler.clone()

					
					active_next_next = active_next & si_next.is_valid() & (depth + 2 < depth_lim)

					
					bsdf_next = si_next.bsdf(ray_next)
					bsdf_prev = si_prev.bsdf(ray_prev)

					
					bsdf_sample_next, bsdf_weight_next = bsdf_next.sample(
						bsdf_ctx, si_next, sampler_clone.next_1d(),
						sampler_clone.next_2d(), active_next_next
					)
					is_valid_guide = mi.Bool(True)
					if self.sdtree_mix is not None and apply_to_this_bounce and should_guide:
						dtree = self.sdtree_mix.query_stree_wrapped(si_next.p, scene.bbox())
						dtree_mix = mi.Float(dtree)
						
						guide_dir = self.sdtree_mix.sample_dtree_wrapped(sampler_clone, mi.Float(dtree), 1 if self.is_primal_phase else self.dtree_sample_method, bsdf_next, si_next, bsdf_ctx, bsdf_sample_next.alpha, dr.mean(β), dr.mean(dr.abs(β_df)), outputname + str(i))	
						
						guide_dir = si_next.to_local(guide_dir)
						active_surface = active_next_next & (bsdf_sample_next.alpha > self.bsdf_roughness_threshold)
						frac_guide = dr.select(is_valid_guide & active_surface, self.guiding_frac, 0.0)
						mix_dir = dr.select(sampler_clone.next_1d() < frac_guide, guide_dir, bsdf_sample_next.wo)
						bsdf_sample_next.wo = mix_dir
	

					
					
					with dr.resume_grad(ray_reparam):
						
						
						si_cur_reparam_only = pi_cur.compute_surface_interaction(
							ray_reparam, mi.RayFlags.All | mi.RayFlags.DetachShape)

						
						
						wo_prev = dr.normalize(si_cur_reparam_only.p - si_prev.p)
						wi_next = dr.normalize(si_cur_reparam_only.p - si_next.p)

						
						si_next.wi = si_next.to_local(wi_next)
						Le_next = β * si_next.emitter(scene).eval(si_next, active_next)

						
						L_next = L - dr.detach(Le_next)

						
						bsdf_val_prev = bsdf_prev.eval(bsdf_ctx, si_prev,
													si_prev.to_local(wo_prev))
						bsdf_val_next = bsdf_next.eval(bsdf_ctx, si_next,
													bsdf_sample_next.wo)

						extra = mi.Spectrum(Le_next)
						extra[~first_vertex]      += L_prev * bsdf_val_prev / dr.maximum(1e-8, dr.detach(bsdf_val_prev))
						extra[si_next.is_valid()] += L_next * bsdf_val_next / dr.maximum(1e-8, dr.detach(bsdf_val_next))

					with dr.resume_grad():
						wo = si_cur.to_local(ray_next.d)
						bsdf_val = bsdf.eval(bsdf_ctx, si_cur, wo, active_next)
						if not primal:
							bsdf_val_detach = bsdf_weight * bsdf_sample.pdf
							inv_bsdf_val_detach = dr.select(dr.neq(bsdf_val_detach, 0),
															dr.rcp(bsdf_val_detach), 0)
							Lr = L * dr.replace_grad(1, inv_bsdf_val_detach * bsdf_val)
							Lo = (Le + Lr) * ray_reparam_det + extra

						if mode == dr.ADMode.Forward:
							tmp = dr.forward_to(Lo, flags=dr.ADFlag.Default if not enable_prb_pro else dr.ADFlag.ClearEdges)
						elif mode == dr.ADMode.Backward:
							param_tensor: mi.TensorXf = params.get(self.param_key)	
							dr.forward(param_tensor, flags=dr.ADFlag.Default if not enable_prb_pro else dr.ADFlag.ClearEdges)
							tmp = dr.grad(Lo)
						else:
							
							dr.forward_to(bsdf_val, flags=dr.ADFlag.ClearEdges) 

						if enable_prb_pro:
							dbsdf = dr.grad(bsdf_val)
							β_df = β_before * dbsdf * dr.select(dr.neq(bsdf_sample.pdf, 0), dr.rcp(bsdf_sample.pdf), 0) + β_df * bsdf_weight

				
				if not primal and apply_to_this_bounce:
					record_spec[record_idx] = dr.select(active_record, tmp, record_spec[record_idx])

					if mode == dr.ADMode.Backward:
						if 'normal' not in self.param_key:
							val = tmp
						else:
							dL_df = L*inv_bsdf_val_detach
							val = bsdf.compute_normalmap_deriv(bsdf_ctx, si_cur, si_cur.to_local(ray_cur.d),  dL_df, si_cur.is_valid())   

						if self.enable_positivization:
							assert self.positivization_iter == 0 or self.positivization_iter == 1
							if self.positivization_iter == 0:
								val = dr.maximum(val, 0)
							else:
								val = dr.minimum(val, 0)
						
						val = dr.select(si_cur.is_valid(), δL * val, 0)
						if self.enable_ssguiding:
							if apply_to_single_bounce:
								self.val_backward = val 
							else:
								if i == 0:
									self.val_backward = val 
								else:
									self.val_backward += val 
						if self.target_channel_one:
							val = dr.sum(val)
						index = bsdf.get_texel_index(si_cur, wo, self.texture_id, si_cur.is_valid())
						dr.scatter_reduce(dr.ReduceOp.Add, self.grad_out.array, val, index, si_cur.is_valid())
					else:
						val = tmp
						if self.enable_positivization:
							assert self.positivization_iter == 0 or self.positivization_iter == 1
							if self.positivization_iter == 0:
								val = dr.maximum(val, 0)
							else:
								val = dr.minimum(val, 0)
						δL += val
					

			
			if not primal:
				pi_prev  = pi_cur
				ray_prev = ray_cur

			
			pi_cur   = pi_next
			ray_cur  = ray_next

			depth[si_cur.is_valid()] += 1
			active = active_next


		if apply_to_single_bounce:
			if self.target_func_full and not primal:
				record_val[0] = dr.sum(record_spec[0] / record_beta[0]) * mi.Float(kwargs['inv_spp']) / 3 
			else:
				record_val[0] = dr.sum((L - record_spec[0]) / record_beta[0]) / 3 * mi.Float(kwargs['inv_spp']) 
		else:
			
			record_sum[0] = record_spec[0]
			for j in range(1, record_level):
				record_sum[j] = record_spec[j] + record_sum[j - 1]
			record_sum_all = record_sum[record_level - 1] 
			for j in range(record_level):
				beta = record_beta[j]

				
				if self.target_func_full and not primal:
					tmp = dr.sum((record_sum_all - record_sum[j] + record_spec[j]) / beta) / 3		
					record_val[j] = tmp * mi.Float(kwargs['inv_spp'])
				else:
					tmp = dr.sum((record_sum_all - record_sum[j]) / beta) / 3 
					record_val[j] = tmp * mi.Float(kwargs['inv_spp'])


		kernel_output = dr.zeros(dr.llvm.ad.TensorXf, shape=(7 * record_level + 3, record_size_delta))
		if self.require_record:
			for i in range(record_level):
				kernel_output[7 * i + 0] = record_pos[i][0]
				kernel_output[7 * i + 1] = record_pos[i][1]
				kernel_output[7 * i + 2] = record_pos[i][2]
				kernel_output[7 * i + 3] = record_dir[i][0]
				kernel_output[7 * i + 4] = record_dir[i][1]
				kernel_output[7 * i + 5] = record_dir[i][2]
				kernel_output[7 * i + 6] = record_val[i]		

		if primal:
			L = dr.clamp(L, 0, 1e9)
		else:
			δL = dr.clamp(δL, -1e9, 1e9)
		kernel_output[7 * record_level + 0] = L[0] if primal else δL[0]
		kernel_output[7 * record_level + 1] = L[1] if primal else δL[1]
		kernel_output[7 * record_level + 2] = L[2] if primal else δL[2]

		
		record_arr = np.array(kernel_output[:7*record_level], dtype=np.float64).T

		if primal:
			L = dr.llvm.ad.Array3f(dr.llvm.ad.Float(dr.ravel(kernel_output[-3])),dr.llvm.ad.Float(dr.ravel(kernel_output[-2])),dr.llvm.ad.Float(dr.ravel(kernel_output[-1])))
		else:
			δL = dr.llvm.ad.Array3f(dr.llvm.ad.Float(dr.ravel(kernel_output[-3])), dr.llvm.ad.Float(dr.ravel(kernel_output[-2])), dr.llvm.ad.Float(dr.ravel(kernel_output[-1])))

		assert np.shape(record_arr) == (record_size_delta, 7*record_level)
		
		record_arr = np.row_stack([record_arr[:, i*7:(i+1)*7] for i in range(record_level)])
		assert np.shape(record_arr) == (record_size, 7)

		record_arr = record_arr[np.linalg.norm(record_arr[:,3:6],axis=-1)>0]
		
		record_arr = record_arr[np.abs(record_arr[:,-1])<1e19]

		global rendering_time
		rendering_time += time.time() - wall_clock
		print("    rendering uses %.3f sec, total %.3f sec" % (time.time() - wall_clock, rendering_time))
		
		np.set_printoptions(threshold=np.inf) 
		time1 = time.time()

		if primal:
			with open('guiding_samples.txt','wb') as f:  
				record_arr.tofile(f)  
		else:
			with open('guiding_samples_adjoint.txt','wb') as f:
				record_arr.tofile(f)
		
		global data_swap_time
		time1 = time.time() - time1
		data_swap_time += time1
		

		return (
			L if primal else δL, 
			dr.neq(depth, 0),    
			L                    
		)

	def get_real_spp_inv(self, film, spp, pos):
		t_begin = time.time()
		
		
		pos = np.floor(pos).astype(int)
		_, inv_idx, cnt = np.unique(pos, axis=0, return_counts=True, return_inverse=True)
		total_num = dr.prod(film.crop_size()) * spp
		assert inv_idx.shape[0] == total_num
		inv_spp = 1 / cnt[inv_idx]
		inv_spp[cnt[inv_idx] == 0] = 0
		
		t_end = time.time()
		print("    realsppin uses %.3f sec" % (t_end - t_begin))
		return inv_spp


	
	def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        
		
		
		
		
		
		

		if self.sdtree_mix:
			self.sdtree_mix.record_ratio = self.record_ratio 
			self.sdtree_mix.resolution = sensor.film().crop_size()[0]

		def render_forward_impl(self: mi.SamplingIntegrator,
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

				
				
				
				if self.need_reparam:
					sample_func = self.sample_reparam
					reparam = _ReparamWrapper(
						scene=scene,
						params=params,
						reparam=self.reparam,
						wavefront_size=sampler.wavefront_size(),
						seed=seed
					)
				else:
					sample_func = self.sample
					reparam = None

				
				
				ray, weight, pos, det = self.sample_rays(scene, sensor,
															sampler, reparam)
				if self.enable_ssguiding:
					inv_real_spp = self.get_real_spp_inv(film, spp, pos)

				
				L, valid, state_out = sample_func(
					mode=dr.ADMode.Primal,
					scene=scene,
					sampler=sampler.clone(),
					ray=ray,
					depth=mi.UInt32(0),
					δL=None,
					state_in=None,
					reparam=None,
					active=mi.Bool(True),
					inv_spp=inv_real_spp if self.enable_ssguiding else 1/spp
				)

				
				δL, valid_2, state_out_2 = sample_func(
					mode=dr.ADMode.Forward,
					scene=scene,
					sampler=sampler,
					ray=ray,
					depth=mi.UInt32(0),
					δL=None,
					state_in=state_out,
					reparam=reparam,
					active=mi.Bool(True),
					inv_spp=inv_real_spp if self.enable_ssguiding else 1/spp
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
								value=δL*weight,
								weight=1.0,
								alpha=dr.select(valid_2, mi.Float(1), mi.Float(0)),
								wavelengths=ray.wavelengths)

				
				film.put_block(block)
				result_grad = film.develop()

				
				del sampler, ray, weight, pos, L, valid, δL, valid_2, params, \
					state_out, state_out_2, block

				gc.collect()

				
				if sample_pos_deriv is not None:
					with dr.resume_grad():
						film.clear()
						
						film.put_block(sample_pos_deriv)
						reparam_result = film.develop()
						dr.forward_to(reparam_result)
						result_grad += dr.grad(reparam_result)

			return result_grad

		self.positivization_iter = -1
		if self.sdtree_mix:
			self.sdtree_mix.enable_positivization = self.enable_positivization
			self.sdtree_mix.positivization_iter = -1
		
		w, h = sensor.film().crop_size()
		if not self.enable_positivization:
			self.guiding_film_idx = 0	
			result = render_forward_impl(self, scene, params, sensor, seed, spp)
			self.update_film_distr(self.guiding_film_idx, w, h, spp, result)
		else:
			assert spp % 2 == 0
			half_spp = max(1, spp // 2)

			
			if self.sdtree_mix:
				self.sdtree_mix.positivization_iter = 0
			self.positivization_iter = 0
			self.guiding_film_idx = 0
			result = render_forward_impl(self, scene, params, sensor, seed, half_spp)
			self.update_film_distr(self.guiding_film_idx, w, h, half_spp, result)
			save_tmp_samples()
			
			
			if self.sdtree_mix:
				self.sdtree_mix.positivization_iter = 1
			self.positivization_iter = 1
			self.guiding_film_idx = 1
			if self.change_seed:
				
				
				result2 = render_forward_impl(self, scene, params, sensor, mi.sample_tea_32(seed, 1)[0], half_spp)
			else:
				result2 = render_forward_impl(self, scene, params, sensor, seed, half_spp)
			self.update_film_distr(self.guiding_film_idx, w, h, half_spp, result2)
			result += result2
			merge_tmp_samples()

		return result 

	def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: mi.SceneParameters,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:
		
		
		
		
		
		
		
		if self.param_key == None:
			raise Exception('need to set self.param_key first!')
		param_tensor: mi.TensorXf = params.get(self.param_key)
		self.grad_out = mi.TensorXf(0, param_tensor.shape)
		if self.param_key not in params.properties:
			raise Exception(f'Unknown texture {self.param_name}!')
		_, _, node, _ = params.properties[self.param_key]
		self.texture_id = node.id()  
		if param_tensor.shape[2] == 1:
			self.target_channel_one = True

		if self.sdtree_mix:
			self.sdtree_mix.record_ratio = self.record_ratio 
			self.sdtree_mix.resolution = sensor.film().crop_size()[0]

		pos_backup = None 

		def render_backward_impl(self: mi.SamplingIntegrator,
					scene: mi.Scene,
					params: mi.SceneParameters,
					grad_in: mi.TensorXf,
					sensor: Union[int, mi.Sensor] = 0,
					seed: int = 0,
					spp: int = 0) -> mi.Vector2i:

			dr.set_grad(param_tensor, 1)  

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
															sampler)
				if self.enable_ssguiding:
					inv_real_spp = self.get_real_spp_inv(film, spp, pos)

				
				
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
					active=mi.Bool(True),
					params=params,
					inv_spp=inv_real_spp if self.enable_ssguiding else 1/spp
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
					active=mi.Bool(True),
					params=params,
					inv_spp=inv_real_spp if self.enable_ssguiding else 1/spp
				)

				with dr.resume_grad():
					dr.set_grad(param_tensor, self.grad_out)

				
				del L_2, valid_2, state_out, state_out_2, δL, \
					ray, weight, sampler

				gc.collect()

				
				dr.eval()

				return mi.Vector2i(pos)  


		self.positivization_iter = -1
		if self.sdtree_mix:
			self.sdtree_mix.enable_positivization = self.enable_positivization
			self.sdtree_mix.positivization_iter = -1


		self.guiding_film_idx = 0
		if not self.enable_positivization:
			pos_backup = render_backward_impl(self, scene, params, grad_in, sensor, seed, spp)
		else:
			assert spp % 2 == 0
			half_spp = max(1, spp // 2)
			
			if self.sdtree_mix:
				self.sdtree_mix.positivization_iter = 0
			self.positivization_iter = 0
			render_backward_impl(self, scene, params, grad_in, sensor, seed, half_spp)
			save_tmp_samples()

			
			if self.sdtree_mix:
				self.sdtree_mix.positivization_iter = 1
			self.positivization_iter = 1
			if self.change_seed:
				render_backward_impl(self, scene, params, grad_in, sensor, mi.sample_tea_32(seed, 1)[0], half_spp)
			else:
				pos_backup = render_backward_impl(self, scene, params, grad_in, sensor, seed, half_spp)
			merge_tmp_samples()

		if self.enable_ssguiding:
			w, h = sensor.film().crop_size()
			val_result = mi.TensorXf(0, shape=(w, h, 3))
			pos1d = pos_backup[1] * w + pos_backup[0]
			dr.scatter_reduce(dr.ReduceOp.Add, val_result.array, self.val_backward, pos1d)
			
			
			
			
			self.update_film_distr(self.guiding_film_idx, w, h, spp if not self.enable_positivization else half_spp, val_result)

	


	def render(self: mi.SamplingIntegrator,
				scene: mi.Scene,
				sensor: Union[int, mi.Sensor] = 0,
				seed: int = 0,
				spp: int = 0,
				develop: bool = True,
				evaluate: bool = True) -> mi.TensorXf:

		if not develop:
			raise Exception("develop=True must be specified when "
							"invoking AD integrators")

		if isinstance(sensor, int):
			sensor = scene.sensors()[sensor]
		film = sensor.film()
		
		
		self.require_load_sdtree = self.require_record
		if self.require_load_sdtree:
			if os.path.exists("guiding_log_mix.txt"):
				
				self.sdtree_mix = MySDTree(type_id=4, mixture_count=4)
				self.sdtree_mix.record_ratio = self.record_ratio 
				self.sdtree_mix.resolution = film.crop_size()[0]
			else:
				
				self.sdtree_mix = None
		
		assert not self.enable_positivization
		self.positivization_iter = -1
		if self.sdtree_mix:
			self.sdtree_mix.enable_positivization = False
			self.sdtree_mix.positivization_iter = -1

		
		with dr.suspend_grad():
			
			sampler, spp = self.prepare(
				sensor=sensor,
				seed=seed,
				spp=spp,
				aovs=self.aovs()
			)

			
			ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)

			
			if self.need_reparam:
				sample_func = self.sample_reparam
			else:
				sample_func = self.sample

			L, valid, state = sample_func(
				mode=dr.ADMode.Primal,
				scene=scene,
				sampler=sampler,
				ray=ray,
				depth=mi.UInt32(0),
				δL=None,
				state_in=None,
				reparam=None,
				active=mi.Bool(True),
				params=None,  
				inv_spp=1/spp
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

mi.register_integrator("prb_basic_mod", lambda props: BasicPRBModIntegrator(props))
