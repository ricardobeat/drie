// Deferred G-Buffer geometry pass shaders (plain and TSL opaque materials).
// Regenerate generated/shader_gbuffer.c3 with: just shaders
@header module drie_shaders::gbuffer;
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

@fs fs_gbuffer
#define GBUFFER
@include_block fs_main
@end

@fs fs_gbuffer_opaque
#define GBUFFER
#define NO_DISCARD
@include_block fs_main
@end

@program gbuffer vs fs_gbuffer
@program gbuffer_opaque vs fs_gbuffer_opaque
