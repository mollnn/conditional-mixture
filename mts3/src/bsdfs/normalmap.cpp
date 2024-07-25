#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _bsdf-normalmap:

Normal map BSDF (:monosp:`normalmap`)
-------------------------------------

.. pluginparameters::

 * - normalmap
   - |texture|
   - The color values of this texture specify the perturbed normals relative in the local surface coordinate system
   - |exposed|, |differentiable|, |discontinuous|

 * - (Nested plugin)
   - |bsdf|
   - A BSDF model that should be affected by the normal map
   - |exposed|, |differentiable|

Normal mapping is a simple technique for cheaply adding surface detail to a rendering. This is done
by perturbing the shading coordinate frame based on a normal map provided as a texture. This method
can lend objects a highly realistic and detailed appearance (e.g. wrinkled or covered by scratches
and other imperfections) without requiring any changes to the input geometry.

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/bsdf_normalmap_without.jpg
   :caption: Roughplastic BSDF
.. subfigure:: ../../resources/data/docs/images/render/bsdf_normalmap_with.jpg
   :caption: Roughplastic BSDF with normal mapping
.. subfigend::
   :label: fig-normalmap

A normal map is a RGB texture, whose color channels encode the XYZ coordinates of the desired
surface normals. These are specified **relative** to the local shading frame, which means that a
normal map with a value of :math:`(0,0,1)` everywhere causes no changes to the surface. To turn the
3D normal directions into (nonnegative) color values suitable for this plugin, the mapping
:math:`x \mapsto (x+1)/2` must be applied to each component.

The following XML snippet describes a smooth mirror material affected by a normal map. Note the we set the
``raw`` properties of the normal map ``bitmap`` object to ``true`` in order to disable the
transformation from sRGB to linear encoding:

.. tabs::
    .. code-tab:: xml
        :name: normalmap

        <bsdf type="normalmap">
            <texture name="normalmap" type="bitmap">
                <boolean name="raw" value="true"/>
                <string name="filename" value="textures/normalmap.jpg"/>
            </texture>
            <bsdf type="roughplastic"/>
        </bsdf>

    .. code-tab:: python

        'type': 'normalmap',
        'normalmap': {
            'type': 'bitmap',
            'raw': True,
            'filename': 'textures/normalmap.jpg'
        },
        'bsdf': {
            'type': 'roughplastic'
        }
*/

//#define ORIGINAL_SHADING_FRAME 

template <typename Float, typename Spectrum>
class NormalMap final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture)

    NormalMap(const Properties &props) : Base(props) {
        for (auto &[name, obj] : props.objects(false)) {
            auto bsdf = dynamic_cast<Base *>(obj.get());

            if (bsdf) {
                if (m_nested_bsdf)
                    Throw("Only a single BSDF child object can be specified.");
                m_nested_bsdf = bsdf;
                props.mark_queried(name);
            }
        }
        if (!m_nested_bsdf)
            Throw("Exactly one BSDF child object must be specified.");

        // TODO: How to assert this is actually a RGBDataTexture?
        m_normalmap = props.texture<Texture>("normalmap");

        // Add all nested components
        m_flags = (uint32_t) 0;
        for (size_t i = 0; i < m_nested_bsdf->component_count(); ++i) {
            m_components.push_back((m_nested_bsdf->flags(i)));
            m_flags |= m_components.back();
        }
        if (props.get<bool>("differentiable", false)) {
            m_flags |= +BSDFFlags::NeedsDifferentials;
        }

        dr::set_attr(this, "flags", m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_object("nested_bsdf", m_nested_bsdf.get(), +ParamFlags::Differentiable);
        callback->put_object("normalmap",   m_normalmap.get(),   ParamFlags::Differentiable | ParamFlags::Discontinuous);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
        // Sample nested BSDF with perturbed shading frame
#ifdef ORIGINAL_SHADING_FRAME
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.wi);
#else
        auto [perturbed_frame_wrt_si, perturbed_frame_wrt_world] =
            frame(si, active);
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = perturbed_frame_wrt_world;
        perturbed_si.wi       = perturbed_frame_wrt_si.to_local(si.wi);
#endif
        auto [bs, weight] = m_nested_bsdf->sample(ctx, perturbed_si,
                                                  sample1, sample2, active);
        active &= dr::any(dr::neq(unpolarized_spectrum(weight), 0.f));
        if (dr::none_or<false>(active))
            return { bs, 0.f };

        // Transform sampled 'wo' back to original frame and check orientation
#ifdef ORIGINAL_SHADING_FRAME
        Vector3f perturbed_wo = perturbed_si.to_world(bs.wo);
#else
        Vector3f perturbed_wo = perturbed_frame_wrt_si.to_world(bs.wo);
#endif
        active &= Frame3f::cos_theta(bs.wo) *
                  Frame3f::cos_theta(perturbed_wo) > 0.f;
        bs.wo = perturbed_wo;
        bs.pdf = dr::select(active, bs.pdf, 0.f);

        return { bs, weight & active };
    }


    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        // Evaluate nested BSDF with perturbed shading frame
#ifdef ORIGINAL_SHADING_FRAME
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_si.to_local(wo);
#else
        auto [perturbed_frame_wrt_si, perturbed_frame_wrt_world] =
            frame(si, active);
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = perturbed_frame_wrt_world;
        perturbed_si.wi       = perturbed_frame_wrt_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_frame_wrt_si.to_local(wo);
#endif
        active &= Frame3f::cos_theta(wo) *
                  Frame3f::cos_theta(perturbed_wo) > 0.f;

        return m_nested_bsdf->eval(ctx, perturbed_si, perturbed_wo, active) & active;
    }


    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        // Evaluate nested BSDF with perturbed shading frame
#ifdef ORIGINAL_SHADING_FRAME
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_si.to_local(wo);
#else
        auto [perturbed_frame_wrt_si, perturbed_frame_wrt_world] =
            frame(si, active);
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = perturbed_frame_wrt_world;
        perturbed_si.wi       = perturbed_frame_wrt_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_frame_wrt_si.to_local(wo);
#endif
        active &= Frame3f::cos_theta(wo) *
                  Frame3f::cos_theta(perturbed_wo) > 0.f;

        return dr::select(active, m_nested_bsdf->pdf(ctx, perturbed_si, perturbed_wo, active), 0.f);
    }

    std::pair<Spectrum, Float> eval_pdf(const BSDFContext &ctx,
                                        const SurfaceInteraction3f &si,
                                        const Vector3f &wo,
                                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        // Evaluate nested BSDF with perturbed shading frame
#ifdef ORIGINAL_SHADING_FRAME
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_si.to_local(wo);
#else
        auto [perturbed_frame_wrt_si, perturbed_frame_wrt_world] =
            frame(si, active);
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = perturbed_frame_wrt_world;
        perturbed_si.wi       = perturbed_frame_wrt_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_frame_wrt_si.to_local(wo);
#endif
        active &= Frame3f::cos_theta(wo) *
                  Frame3f::cos_theta(perturbed_wo) > 0.f;

        auto [value, pdf] = m_nested_bsdf->eval_pdf(ctx, perturbed_si, perturbed_wo, active);
        return { value & active, dr::select(active, pdf, 0.f) };
    }

#ifdef ORIGINAL_SHADING_FRAME
    Frame3f frame(const SurfaceInteraction3f &si, Mask active) const {
        Normal3f n = dr::fmadd(m_normalmap->eval_3(si, active), 2, -1.f);

        Frame3f result;
        result.n = dr::normalize(n);
        result.s = dr::normalize(dr::fnmadd(result.n, dr::dot(result.n, si.dp_du), si.dp_du));
        result.t = dr::cross(result.n, result.s);
        return result;
    }
#else
    // https://github.com/mitsuba-renderer/mitsuba3/pull/905
    std::pair<Frame3f, Frame3f> frame(const SurfaceInteraction3f &si,
                                      Mask active) const {
        Normal3f n = dr::fmadd(m_normalmap->eval_3(si, active), 2, -1.f);

        Frame3f frame_wrt_si;
        frame_wrt_si.n = dr::normalize(n);
        frame_wrt_si.s = dr::normalize(dr::fnmadd(
            frame_wrt_si.n, frame_wrt_si.n.x(), ScalarVector3f(1, 0, 0)));
        frame_wrt_si.t = dr::cross(frame_wrt_si.n, frame_wrt_si.s);

        Frame3f frame_wrt_world;
        frame_wrt_world.n = si.to_world(frame_wrt_si.n);
        frame_wrt_world.s = si.to_world(frame_wrt_si.s);
        frame_wrt_world.t = si.to_world(frame_wrt_si.t);

        return { frame_wrt_si, frame_wrt_world };
    }
#endif // ORIGINAL_SHADING_FRAME

   /*
   // this version now only consider diffuse contribution
   // need glossy part i.e. gradients from FDG
   // note this won't consider any transmission
    Color3f compute_normalmap_deriv(const BSDFContext &ctx,
                                    const SurfaceInteraction3f &si,
                                    const Vector3f &wo,
                                    const Vector3f &d_output,
                                Mask active) const override {
        // this 'wo' in world space
       
	    auto d_length_squared = [](const Vector3f &v0, Float d_l_sq) {
            return 2 * d_l_sq * v0;
        };

        auto d_length = [d_length_squared](const Vector3f &v0, Float d_l) {
            auto l    = dr::norm(v0);
            Mask mask = l > 0.f;
            return dr::select(mask, d_length_squared(v0, 0.5f * d_l / l), 0.f);
        };

        auto d_normalize = [d_length](const Vector3f &v0, const Vector3f &d_n) {
            auto l  = dr::norm(v0);
            Mask mask = l > 0.f;
            auto n    = dr::select(mask, v0 / l, 0.f);
            auto d_v0 = dr::select(mask, d_n / l, 0.f);
            auto d_l  = dr::select(mask, -dr::dot(d_n, n) / l, 0.f);
            return dr::select(mask, d_v0 + d_length(v0, d_l), 0.f);
        };

        auto shdFrame = frame(si, active);
        auto w        = wo;
#ifdef ORIGINAL_SHADING_FRAME
        Vector3f d_out =
            dr::mulsign(dr::sum(d_output), dr::dot(shdFrame.n, w))
            * m_nested_bsdf->eval_diffuse_reflectance(si, active) / dr::Pi<Float>;
#else
        Vector3f d_out =
            dr::mulsign(dr::sum(d_output), dr::dot(shdFrame.second.n, w)) 
            * m_nested_bsdf->eval_diffuse_reflectance(si, active) / dr::Pi<Float>;

#endif
        Vector3f d_n = w * d_out; // TODO now only diffuse, need glossy part

        Normal3f n_local = dr::fmadd(m_normalmap->eval_3(si, active), 2, -1.f);
        Normal3f n_world = si.to_world(n_local);
        Color3f d_n_world = d_normalize(n_world, d_n);

        // d_to_world
        Color3f d_v = dr::zeros<Color3f>(); 
        d_v[0] = dr::sum(d_n_world * si.sh_frame.s);
        d_v[1] = dr::sum(d_n_world * si.sh_frame.t);
        d_v[2] = dr::sum(d_n_world * si.sh_frame.n);

        return dr::select(active, 2 * d_v, 0.f);
    }
    */
    
    
    //finite difference
    Color3f compute_normalmap_deriv(const BSDFContext &ctx,
                                    const SurfaceInteraction3f &si,
                                    const Vector3f &wo,
                                    const Vector3f &d_output,
                                    Mask active) const override {
          // this 'wo' in local space
        Color3f val0, val1, val2, val3;
        Float eps = 0.01;

        Color3f raw_normal = m_normalmap->eval_3(si, active);

        {
            SurfaceInteraction3f perturbed_si(si);
            Normal3f n = dr::fmadd(raw_normal, 2, -1.f);

#ifdef ORIGINAL_SHADING_FRAME
            Frame3f result;
            result.n = dr::normalize(n);
            result.s = dr::normalize(
                dr::fnmadd(result.n, dr::dot(result.n, si.dp_du), si.dp_du));
            result.t              = dr::cross(result.n, result.s);
            perturbed_si.sh_frame = result;
            perturbed_si.wi       = perturbed_si.to_local(si.wi);
            Vector3f perturbed_wo = perturbed_si.to_local(wo);
#else
            Frame3f frame_wrt_si;
            frame_wrt_si.n = dr::normalize(n);
            frame_wrt_si.s = dr::normalize(dr::fnmadd(
                frame_wrt_si.n, frame_wrt_si.n.x(), ScalarVector3f(1, 0, 0)));
            frame_wrt_si.t = dr::cross(frame_wrt_si.n, frame_wrt_si.s);

            Frame3f frame_wrt_world;
            frame_wrt_world.n = si.to_world(frame_wrt_si.n);
            frame_wrt_world.s = si.to_world(frame_wrt_si.s);
            frame_wrt_world.t = si.to_world(frame_wrt_si.t);

            perturbed_si.sh_frame = frame_wrt_world;
            perturbed_si.wi       = frame_wrt_si.to_local(si.wi);
            Vector3f perturbed_wo = frame_wrt_si.to_local(wo);
#endif // ORIGINAL_SHADING_FRAME

            auto new_active = active & (Frame3f::cos_theta(wo) * Frame3f::cos_theta(perturbed_wo) > 0.f);

            val0 = m_nested_bsdf->eval(ctx, perturbed_si, perturbed_wo, new_active) & new_active;
        }
        {
            SurfaceInteraction3f perturbed_si(si);
            auto new_normal = Color3f(raw_normal);
            new_normal[0] += eps;
            Normal3f n = dr::fmadd(new_normal, 2, -1.f);

#ifdef ORIGINAL_SHADING_FRAME
            Frame3f result;
            result.n = dr::normalize(n);
            result.s = dr::normalize(
                dr::fnmadd(result.n, dr::dot(result.n, si.dp_du), si.dp_du));
            result.t              = dr::cross(result.n, result.s);
            perturbed_si.sh_frame = result;
            perturbed_si.wi       = perturbed_si.to_local(si.wi);
            Vector3f perturbed_wo = perturbed_si.to_local(wo);
#else
            Frame3f frame_wrt_si;
            frame_wrt_si.n = dr::normalize(n);
            frame_wrt_si.s = dr::normalize(dr::fnmadd(
                frame_wrt_si.n, frame_wrt_si.n.x(), ScalarVector3f(1, 0, 0)));
            frame_wrt_si.t = dr::cross(frame_wrt_si.n, frame_wrt_si.s);

            Frame3f frame_wrt_world;
            frame_wrt_world.n = si.to_world(frame_wrt_si.n);
            frame_wrt_world.s = si.to_world(frame_wrt_si.s);
            frame_wrt_world.t = si.to_world(frame_wrt_si.t);

            perturbed_si.sh_frame = frame_wrt_world;
            perturbed_si.wi       = frame_wrt_si.to_local(si.wi);
            Vector3f perturbed_wo = frame_wrt_si.to_local(wo);
#endif // ORIGINAL_SHADING_FRAME

            auto new_active = active & (Frame3f::cos_theta(wo) * Frame3f::cos_theta(perturbed_wo) > 0.f);

            val1 = m_nested_bsdf->eval(ctx, perturbed_si, perturbed_wo, new_active) & new_active;
        }
        {
            SurfaceInteraction3f perturbed_si(si);
            auto new_normal = raw_normal;
            new_normal[1] += eps;

            Normal3f n = dr::fmadd(new_normal, 2, -1.f);

#ifdef ORIGINAL_SHADING_FRAME
            Frame3f result;
            result.n = dr::normalize(n);
            result.s = dr::normalize(
                dr::fnmadd(result.n, dr::dot(result.n, si.dp_du), si.dp_du));
            result.t              = dr::cross(result.n, result.s);
            perturbed_si.sh_frame = result;
            perturbed_si.wi       = perturbed_si.to_local(si.wi);
            Vector3f perturbed_wo = perturbed_si.to_local(wo);
#else
            Frame3f frame_wrt_si;
            frame_wrt_si.n = dr::normalize(n);
            frame_wrt_si.s = dr::normalize(dr::fnmadd(
                frame_wrt_si.n, frame_wrt_si.n.x(), ScalarVector3f(1, 0, 0)));
            frame_wrt_si.t = dr::cross(frame_wrt_si.n, frame_wrt_si.s);

            Frame3f frame_wrt_world;
            frame_wrt_world.n = si.to_world(frame_wrt_si.n);
            frame_wrt_world.s = si.to_world(frame_wrt_si.s);
            frame_wrt_world.t = si.to_world(frame_wrt_si.t);

            perturbed_si.sh_frame = frame_wrt_world;
            perturbed_si.wi       = frame_wrt_si.to_local(si.wi);
            Vector3f perturbed_wo = frame_wrt_si.to_local(wo);
#endif // ORIGINAL_SHADING_FRAME

            auto new_active = active & (Frame3f::cos_theta(wo) * Frame3f::cos_theta(perturbed_wo) > 0.f);

            val2 = m_nested_bsdf->eval(ctx, perturbed_si, perturbed_wo, new_active) & new_active;

        }
        {
            SurfaceInteraction3f perturbed_si(si);
            auto new_normal = raw_normal;
            new_normal[2] += eps;

            Normal3f n = dr::fmadd(new_normal, 2, -1.f);

#ifdef ORIGINAL_SHADING_FRAME
            Frame3f result;
            result.n = dr::normalize(n);
            result.s = dr::normalize(
                dr::fnmadd(result.n, dr::dot(result.n, si.dp_du), si.dp_du));
            result.t              = dr::cross(result.n, result.s);
            perturbed_si.sh_frame = result;
            perturbed_si.wi       = perturbed_si.to_local(si.wi);
            Vector3f perturbed_wo = perturbed_si.to_local(wo);
#else
            Frame3f frame_wrt_si;
            frame_wrt_si.n = dr::normalize(n);
            frame_wrt_si.s = dr::normalize(dr::fnmadd(
                frame_wrt_si.n, frame_wrt_si.n.x(), ScalarVector3f(1, 0, 0)));
            frame_wrt_si.t = dr::cross(frame_wrt_si.n, frame_wrt_si.s);

            Frame3f frame_wrt_world;
            frame_wrt_world.n = si.to_world(frame_wrt_si.n);
            frame_wrt_world.s = si.to_world(frame_wrt_si.s);
            frame_wrt_world.t = si.to_world(frame_wrt_si.t);

            perturbed_si.sh_frame = frame_wrt_world;
            perturbed_si.wi       = frame_wrt_si.to_local(si.wi);
            Vector3f perturbed_wo = frame_wrt_si.to_local(wo);
#endif // ORIGINAL_SHADING_FRAME

            auto new_active = active & (Frame3f::cos_theta(wo) * Frame3f::cos_theta(perturbed_wo) > 0.f);

            val3 = m_nested_bsdf->eval(ctx, perturbed_si, perturbed_wo, new_active) & new_active;
        }
        auto dfdnormalmmap1 = (val1 - val0) / eps;
        auto dfdnormalmmap2 = (val2 - val0) / eps;
        auto dfdnormalmmap3 = (val3 - val0) / eps;

        return dr::select(active,
            d_output[0] * Vector3f(dfdnormalmmap1[0], dfdnormalmmap2[0], dfdnormalmmap3[0]) 
          + d_output[1] * Vector3f(dfdnormalmmap1[1], dfdnormalmmap2[1], dfdnormalmmap3[1])
          + d_output[2] * Vector3f(dfdnormalmmap1[2], dfdnormalmmap2[2], dfdnormalmmap3[2]) 
            , 0.f);
    }
    
    


    UInt32 get_texel_index(const SurfaceInteraction3f &si, 
                           const Vector3f &wo,
                           const std::string &texture_id,
                           Mask active) const override {

#ifdef ORIGINAL_SHADING_FRAME
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_si.to_local(wo);
#else
        auto [perturbed_frame_wrt_si, perturbed_frame_wrt_world] =
            frame(si, active);
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = perturbed_frame_wrt_world;
        perturbed_si.wi       = perturbed_frame_wrt_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_frame_wrt_si.to_local(wo);
#endif
        active &= Frame3f::cos_theta(wo) * Frame3f::cos_theta(perturbed_wo) > 0.f;
        if (texture_id == m_normalmap->id()) {
            return m_normalmap->get_texel_index(perturbed_si, active);
        }
        if (m_nested_bsdf) {
            return m_nested_bsdf->get_texel_index(perturbed_si, perturbed_wo,
                                                  texture_id,
                                                  active);
        }
        return 0u;
    }

    Spectrum eval_with_roughness(const BSDFContext &ctx,
                                const SurfaceInteraction3f &si,
                                const Vector3f &wo, Float roughness,
                                Mask active) const override {
#ifdef ORIGINAL_SHADING_FRAME
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = frame(si, active);
        perturbed_si.wi       = perturbed_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_si.to_local(wo);
#else
        auto [perturbed_frame_wrt_si, perturbed_frame_wrt_world] =
            frame(si, active);
        SurfaceInteraction3f perturbed_si(si);
        perturbed_si.sh_frame = perturbed_frame_wrt_world;
        perturbed_si.wi       = perturbed_frame_wrt_si.to_local(si.wi);
        Vector3f perturbed_wo = perturbed_frame_wrt_si.to_local(wo);
#endif
        active &=
            Frame3f::cos_theta(wo) * Frame3f::cos_theta(perturbed_wo) > 0.f;

        return m_nested_bsdf->eval_with_roughness(ctx, perturbed_si, perturbed_wo, roughness, active) & active;
    }

    
    Spectrum eval_diffuse_reflectance(const SurfaceInteraction3f &si,
                                      Mask active) const override {
        return m_nested_bsdf->eval_diffuse_reflectance(si, active);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "NormalMap[" << std::endl
            << "  nested_bsdf = " << string::indent(m_nested_bsdf) << ","
            << std::endl
            << "  normalmap = " << string::indent(m_normalmap) << ","
            << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    ref<Base> m_nested_bsdf;
    ref<Texture> m_normalmap;
};

MI_IMPLEMENT_CLASS_VARIANT(NormalMap, BSDF)
MI_EXPORT_PLUGIN(NormalMap, "Normal map material adapter");
NAMESPACE_END(mitsuba)
