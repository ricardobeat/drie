// Fullscreen scissor-region clear shader used by Renderer.clear_rect() to fill a viewport or
// scissor box with a flat RGBA color and far depth (z = 1). Regenerate generated/clear_rect.c3
// with `just shaders`.
@header module drie::shaders;
@header import drie;

@ctype vec4 Vec4

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

@program clear_rect clear_vs clear_fs
