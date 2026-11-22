// Physical material shaders (clearcoat, sheen, iridescence, anisotropy, transmission).
// Regenerate generated/shader_physical.c3 with: just shaders
@header module drie_shaders::physical;
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

@fs fs_physical
#define PHYSICAL
@include_block fs_main
@end

@fs fs_physical_opaque
#define PHYSICAL
#define NO_DISCARD
@include_block fs_main
@end

@fs fs_physical_mrt
#define PHYSICAL
#define MRT
@include_block fs_main
@end

@fs fs_physical_opaque_mrt
#define PHYSICAL
#define NO_DISCARD
#define MRT
@include_block fs_main
@end

@fs fs_physical_vsm
#define PHYSICAL
#define VSM
@include_block fs_main
@end

@fs fs_physical_mrt_vsm
#define PHYSICAL
#define MRT
#define VSM
@include_block fs_main
@end

@program physical vs fs_physical
@program physical_opaque vs fs_physical_opaque
@program physical_mrt vs fs_physical_mrt
@program physical_opaque_mrt vs fs_physical_opaque_mrt
@program physical_vsm vs fs_physical_vsm
@program physical_mrt_vsm vs fs_physical_mrt_vsm
