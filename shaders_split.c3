// Wrappers for the split shader modules so their generated uniform structs
// do not collide with drie::shaders in renderer.c3 and clusters.c3.
module drie;
import sokol, drie_shaders::physical, drie_shaders::unlit, drie_shaders::gbuffer, drie_shaders::deferred;

const int VIEW_TEX_TRANSMISSION @private = physical::VIEW_TEX_TRANSMISSION;
const int SMP_TRANSMISSION_SMP @private = physical::SMP_TRANSMISSION_SMP;

struct DeferredLightParams @align(16) @packed @private {
    Mat4 inv_view_proj;
    Vec4 view_origin;
    Vec4 clear_color;
    Vec4 screen_info;
}

fn SgShaderDesc split_unlit_shader_desc(NodeVariant v, SgBackend backend) @private {
    return (v == MRT || v == OPAQUE_MRT || v == MRT_VSM)
        ? unlit::unlit_mrt_shader_desc(backend)
        : unlit::unlit_shader_desc(backend);
}

fn SgShaderDesc split_physical_shader_desc(NodeVariant v, SgBackend backend) @private {
    switch (v) {
        case MAIN: return physical::physical_shader_desc(backend);
        case OPAQUE: return physical::physical_opaque_shader_desc(backend);
        case MRT: return physical::physical_mrt_shader_desc(backend);
        case OPAQUE_MRT: return physical::physical_opaque_mrt_shader_desc(backend);
        case VSM: return physical::physical_vsm_shader_desc(backend);
        case MRT_VSM: return physical::physical_mrt_vsm_shader_desc(backend);
        case DEPTH:
        case SHADOW: unreachable("split_physical_shader_desc");
    }
}

fn SgShaderDesc split_gbuffer_shader_desc(NodeVariant v, SgBackend backend) @private {
    return (v == OPAQUE || v == OPAQUE_MRT)
        ? gbuffer::gbuffer_opaque_shader_desc(backend)
        : gbuffer::gbuffer_shader_desc(backend);
}

fn SgShaderDesc split_deferred_shader_desc(bool mrt, bool vsm, SgBackend backend) @private {
    if (vsm) return mrt ? deferred::deferred_mrt_vsm_shader_desc(backend) : deferred::deferred_vsm_shader_desc(backend);
    return mrt ? deferred::deferred_mrt_shader_desc(backend) : deferred::deferred_shader_desc(backend);
}
