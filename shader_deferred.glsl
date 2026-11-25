// Deferred fullscreen lighting pass shaders (clustered lights, shadows, GI, probes, SSR, fog).
// Regenerate generated/shader_deferred.c3 with: just shaders
@header module drie_shaders::deferred;
@header import drie;

@ctype mat4 Mat4
@ctype vec4 Vec4

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
