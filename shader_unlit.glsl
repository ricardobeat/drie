// Unlit material shaders (BASIC, NORMAL, DEPTH, MATCAP, lines, background).
// Regenerate generated/shader_unlit.c3 with: just shaders
@header module drie_shaders::unlit;
@header import drie;

@ctype mat4 Mat4
@ctype vec4 Vec4

@block vertex_hooks
void vertex_hook(inout vec3 world, vec3 local, mat4 model) {}
@end

@block surface_hooks
void surface_hook(vec3 geo_n, inout vec4 base, inout float rough, inout float metal) {}
void normal_hook(vec3 geo_n, inout vec3 n) {}
void reflection_hook(vec3 dir, inout vec3 sky) {}
@end

@include tonemapping.glsl
@include shader_core.glsl

@fs fs_unlit
#define UNLIT
@include_block fs_main
@end

@fs fs_unlit_mrt
#define UNLIT
#define MRT
@include_block fs_main
@end

@program unlit vs fs_unlit
@program unlit_mrt vs fs_unlit_mrt
