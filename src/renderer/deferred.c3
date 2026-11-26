// The deferred lighting pass's shader (shader_deferred.glsl), in a module of its own so that
// its generated uniform structs don't collide with drie::shaders'.
module drie;
import sokol, drie_shaders::deferred;

// what a transmission material refracts: a view slot and a sampler slot the engine's shaders
// leave free (the lighting pass reads the depth in the same view slot)
const int VIEW_TEX_TRANSMISSION @private = deferred::VIEW_TEX_TRANSMISSION;
const int SMP_TRANSMISSION_SMP @private = 5;

struct DeferredLightParams @align(16) @packed @private {
    Mat4 inv_view_proj;
    Vec4 view_origin;
    Vec4 clear_color;
    Vec4 screen_info;
    Mat4 view_proj;
}

fn SgShaderDesc split_deferred_shader_desc(bool mrt, bool vsm, SgBackend backend) @private {
    if (vsm) return mrt ? deferred::deferred_mrt_vsm_shader_desc(backend) : deferred::deferred_vsm_shader_desc(backend);
    return mrt ? deferred::deferred_mrt_shader_desc(backend) : deferred::deferred_shader_desc(backend);
}
