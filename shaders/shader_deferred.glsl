// Deferred fullscreen lighting pass shaders (Renderer): evaluates clustered lights, shadows (PCF
// and VSM), GI, reflection probes, SSR, and fog over the G-Buffer using shaders/shader_core.glsl
// with DEFERRED_LIGHT. Regenerate generated/shader_deferred.c3 with `just shaders`.
//
// Kept in `drie_shaders::deferred` outside `drie::*` because C3 imports `drie::*` recursively;
// placing it under `drie::shaders::*` would collide generated uniform/SSBO types with drie::shaders.
@header module drie_shaders::deferred;
@header import drie;

@ctype mat4 Mat4
@ctype vec4 Vec4

// (the lighting pass reads the depth through tex_transmission's slot: a depth image, which
// WebGPU binds only where the texture isn't filtered)
@image_sample_type tex_transmission unfilterable_float
@sampler_type transmission_smp nonfiltering

@include tonemapping.glsl
@include shader_core.glsl

@vs deferred_vs
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@fs fs_deferred
#define DEFERRED_LIGHT
#define NO_DISCARD
@include_block fs_main
@end

@fs fs_deferred_mrt
#define DEFERRED_LIGHT
#define NO_DISCARD
#define MRT
@include_block fs_main
@end

@fs fs_deferred_vsm
#define DEFERRED_LIGHT
#define NO_DISCARD
#define VSM
@include_block fs_main
@end

@fs fs_deferred_mrt_vsm
#define DEFERRED_LIGHT
#define NO_DISCARD
#define MRT
#define VSM
@include_block fs_main
@end

@program deferred deferred_vs fs_deferred
@program deferred_mrt deferred_vs fs_deferred_mrt
@program deferred_vsm deferred_vs fs_deferred_vsm
@program deferred_mrt_vsm deferred_vs fs_deferred_mrt_vsm
