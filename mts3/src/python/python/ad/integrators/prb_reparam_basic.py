# prb_reparam without MIS


from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from .common import RBIntegrator, mis_weight

class PRBReparamBasicIntegrator(RBIntegrator):

    def __init__(self, props):
        super().__init__(props)

        # The reparameterization is computed stochastically and removes
        # gradient bias at the cost of additional variance. Use this parameter
        # to disable the reparameterization after a certain path depth to
        # control this tradeoff. A value of zero disables it completely.
        self.reparam_max_depth = props.get('reparam_max_depth', self.max_depth)

        # Specifies the number of auxiliary rays used to evaluate the
        # reparameterization
        self.reparam_rays = props.get('reparam_rays', 16)

        # Specifies the von Mises Fisher distribution parameter for sampling
        # auxiliary rays in Bangaru et al.'s [2020] parameterization
        self.reparam_kappa = props.get('reparam_kappa', 1e5)

        # Harmonic weight exponent in Bangaru et al.'s [2020] parameterization
        self.reparam_exp = props.get('reparam_exp', 3.0)

        # Enable antithetic sampling in the reparameterization?
        self.reparam_antithetic = props.get('reparam_antithetic', False)

        # Unroll the loop tracing auxiliary rays in the reparameterization?
        self.reparam_unroll = props.get('reparam_unroll', False)

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

        # Potentially disable the reparameterization completely
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

    def sample(self,
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
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, mi.Spectrum]:
        """
        See ``ADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        β = mi.Spectrum(1)                            # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes

        # Initialize loop state variables caching the rays and preliminary
        # intersections of the previous (zero-initialized) and current vertex
        ray_prev = dr.zeros(mi.Ray3f)
        ray_cur  = mi.Ray3f(dr.detach(ray))
        pi_prev  = dr.zeros(mi.PreliminaryIntersection3f)
        pi_cur   = scene.ray_intersect_preliminary(ray_cur, coherent=True,
                                                   active=active)

        # dr.set_flag(dr.JitFlag.LoopRecord, False)

        # Record the following loop in its entirety
        loop = mi.Loop(name="Path Replay Backpropagation (%s)" % mode.name,
                       state=lambda: (sampler, depth, L, δL, β, η, active,
                                      ray_prev, ray_cur, pi_prev, pi_cur))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            # The first path vertex requires some special handling (see below)
            first_vertex = dr.eq(depth, 0)

            # Reparameterized ray (a copy of 'ray_cur' in primal mode)
            ray_reparam = mi.Ray3f(ray_cur)

            # Jacobian determinant of the parameterization (1 in primal mode)
            ray_reparam_det = 1

            # ----------- Reparameterize (differential phase only) -----------

            if not primal:
                with dr.resume_grad():
                    # Compute a surface interaction of the previous vertex with
                    # derivative tracking (no-op if there is no prev. vertex)
                    si_prev = pi_prev.compute_surface_interaction(
                        ray_prev, mi.RayFlags.All | mi.RayFlags.FollowShape)

                    # Adjust the ray origin of 'ray_cur' so that it follows the
                    # previous shape, then pass this information to 'reparam'
                    ray_reparam.d, ray_reparam_det = reparam(
                        dr.select(first_vertex, ray_cur,
                                  si_prev.spawn_ray(ray_cur.d)), depth)
                    ray_reparam_det[first_vertex] = 1

                    # Finally, disable all derivatives in 'si_prev', as we are
                    # only interested in tracking derivatives related to the
                    # current interaction in the remainder of this function
                    dr.disable_grad(si_prev)

            # ------ Compute detailed record of the current interaction ------

            # Compute a surface interaction that potentially tracks derivatives
            # due to differentiable shape parameters (position, normals, etc.)

            with dr.resume_grad(when=not primal):
                si_cur = pi_cur.compute_surface_interaction(ray_reparam)

            # ---------------------- Direct emission ----------------------

            # Evaluate the emitter (with derivative tracking if requested)
            with dr.resume_grad(when=not primal):
                emitter = si_cur.emitter(scene)
                Le = β * emitter.eval(si_cur)

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si_cur.is_valid()

            # Get the BSDF, potentially computes texture-space differentials.
            bsdf_cur = si_cur.bsdf(ray_cur)

            # ------------------ Detached BSDF sampling -------------------

            # Perform detached BSDF sampling.
            bsdf_sample, bsdf_weight = bsdf_cur.sample(bsdf_ctx, si_cur,
                                                       sampler.next_1d(),
                                                       sampler.next_2d(),
                                                       active_next)

            # ---- Update loop variables based on current interaction -----

            η     *= bsdf_sample.eta
            β     *= bsdf_weight
            L_prev = L  # Value of 'L' at previous vertex
            L      = (L + Le) if primal else (L - Le)

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            # ------------------ Intersect next surface -------------------

            ray_next = si_cur.spawn_ray(si_cur.to_world(bsdf_sample.wo))
            pi_next = scene.ray_intersect_preliminary(ray_next,
                                                      active=active_next)

            # Compute a detached intersection record for the next vertex
            si_next = pi_next.compute_surface_interaction(ray_next)

            # ------------------ Differential phase only ------------------

            if not primal:
                # Clone the sampler to run ahead in the random number sequence
                # without affecting the PRB random walk
                sampler_clone = sampler.clone()

                # 'active_next' value at the next vertex
                active_next_next = active_next & si_next.is_valid() & \
                    (depth + 2 < self.max_depth)

                # Retrieve the BSDFs of the two adjacent vertices
                bsdf_next = si_next.bsdf(ray_next)
                bsdf_prev = si_prev.bsdf(ray_prev)

                # Generate a detached BSDF sample at the next vertex
                bsdf_sample_next, bsdf_weight_next = bsdf_next.sample(
                    bsdf_ctx, si_next, sampler_clone.next_1d(),
                    sampler_clone.next_2d(), active_next_next
                )

                # Account for adjacent vertices, but only consider derivatives
                # that arise from the reparameterization at 'si_cur.p'
                with dr.resume_grad(ray_reparam):
                    # Compute a surface interaction that only tracks derivatives
                    # that arise from the reparameterization.
                    si_cur_reparam_only = pi_cur.compute_surface_interaction(
                        ray_reparam, mi.RayFlags.All | mi.RayFlags.DetachShape)

                    # Differentiably recompute the outgoing direction at 'prev'
                    # and the incident direction at 'next'
                    wo_prev = dr.normalize(si_cur_reparam_only.p - si_prev.p)
                    wi_next = dr.normalize(si_cur_reparam_only.p - si_next.p)

                    # Compute the emission at the next vertex
                    si_next.wi = si_next.to_local(wi_next)
                    Le_next = β * si_next.emitter(scene).eval(si_next, active_next)

                    # Value of 'L' at the next vertex
                    L_next = L - dr.detach(Le_next)

                    # Account for the BSDF of the previous and next vertices
                    bsdf_val_prev = bsdf_prev.eval(bsdf_ctx, si_prev,
                                                   si_prev.to_local(wo_prev))
                    bsdf_val_next = bsdf_next.eval(bsdf_ctx, si_next,
                                                   bsdf_sample_next.wo)

                    extra = mi.Spectrum(Le_next)
                    extra[~first_vertex]      += L_prev * bsdf_val_prev / dr.maximum(1e-8, dr.detach(bsdf_val_prev))
                    extra[si_next.is_valid()] += L_next * bsdf_val_next / dr.maximum(1e-8, dr.detach(bsdf_val_next))

                with dr.resume_grad():
                    # 'L' stores the indirectly reflected radiance at the
                    # current vertex but does not track parameter derivatives.
                    # The following addresses this by canceling the detached
                    # BSDF value and replacing it with an equivalent term that
                    # has derivative tracking enabled. (nit picking: the
                    # direct/indirect terminology isn't 100% accurate here,
                    # since there may be a direct component that is weighted
                    # via multiple importance sampling)

                    # Recompute 'wo' to propagate derivatives to cosine term
                    wo = si_cur.to_local(ray_next.d)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf_cur.eval(bsdf_ctx, si_cur, wo, active_next)

                    # Detached version of the above term and inverse
                    bsdf_val_detach = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_detach = dr.select(dr.neq(bsdf_val_detach, 0),
                                                    dr.rcp(bsdf_val_detach), 0)

                    # Differentiable version of the reflected indirect
                    # radiance. Minor optional tweak: indicate that the primal
                    # value of the second term is always 1.
                    Lr_ind = L * dr.replace_grad(1, inv_bsdf_val_detach * bsdf_val)

                with dr.resume_grad():
                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = (Le + Lr_ind) * ray_reparam_det + extra

                    if dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            # Differential phases need access to the previous interaction, too
            if not primal:
                pi_prev  = pi_cur
                ray_prev = ray_cur

            # Provide ray/interaction to the next iteration
            pi_cur   = pi_next
            ray_cur  = ray_next

            depth[si_cur.is_valid()] += 1
            active = active_next
        # dr.set_flag(dr.JitFlag.LoopRecord, True)

        return (
            L if primal else δL, # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            L                    # State for the differential phase
        )

mi.register_integrator("prb_reparam_basic", lambda props: PRBReparamBasicIntegrator(props))
