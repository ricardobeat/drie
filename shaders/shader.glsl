// Main shader entry around shaders/shader_core.glsl. Material draws use runtime node programs
// (drie::nodes); this build supplies shared uniform/SSBO structs and binding slots, plus the
// depth, shadow, and clear_rect programs. Regenerate generated/shader.c3 with `just shaders`.
@header module drie::shaders;
@header import drie;

@ctype mat4 Mat4
@ctype vec4 Vec4

@include tonemapping.glsl
@include shader_core.glsl

@fs fs
@include_block fs_main
@end

@fs fs_vsm
#define VSM
@include_block fs_main
@end

@fs depth_fs
@include_block depth_main
@end

@vs shadow_vs
@include_block shadow_vs_main
@end

@fs shadow_fs
@include_block shadow_fs_main
@end

@vs clear_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 1.0, 1.0);
}
@end

@fs clear_fs
layout(binding=0) uniform clear_params {
    vec4 clear_rgba;
};
out vec4 frag_color;
void main() {
    frag_color = clear_rgba;
}
@end

@program three vs fs
@program three_vsm vs fs_vsm
@program depth vs depth_fs
@program shadow shadow_vs shadow_fs
@program clear_rect clear_vs clear_fs
