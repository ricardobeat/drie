// OutputPass (postprocessing.c3): the last step of a chain that works in linear light. Tone
// maps the picture with the renderer's tone mapping and exposure and encodes it as sRGB, as
// three.js's OutputPass does. Alpha is kept. Regenerate output.c3 with: ./build.sh shaders
@header module three::postprocessing;
@header import three;

@ctype vec4 Vec4

@include tonemapping.glsl

@vs output_vs
out vec2 uv;

void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    uv = vec2(p.x, 1.0 - p.y);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@fs output_fs
@include_block tonemapping
layout(binding=0) uniform texture2D output_src;
layout(binding=1) uniform texture2D output_exposure;
layout(binding=0) uniform sampler output_smp;
layout(binding=0) uniform output_params {
    vec4 params;   // x = tone mapping (TONE_*), y = exposure, z = 1: multiply it by output_exposure's r,
                   //   w = 1: no sRGB encoding (for a target that is sRGB itself, or more passes in linear)
};

in vec2 uv;
out vec4 frag_color;

void main() {
    vec4 c = texture(sampler2D(output_src, output_smp), uv);
    float exposure = params.y;
    if (params.z > 0.5) exposure *= texelFetch(sampler2D(output_exposure, output_smp), ivec2(0), 0).r;
    vec3 col = tone_map(c.rgb, int(params.x), exposure);
    frag_color = vec4(params.w > 0.5 ? col : linear_to_srgb(col), c.a);
}
@end

@program output output_vs output_fs
