// OutputPass (postprocessing.c3): the last step of a chain that works in linear light. Tone
// maps the picture with the renderer's tone mapping and exposure and encodes it as sRGB, as
// three.js's OutputPass does, or into Display P3's colors, or for an HDR screen. Alpha is kept. Regenerate generated/output.c3 with: just shaders
@header module drie::postprocessing;
@header import drie;

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
@include_block grade_shaper
layout(binding=0) uniform texture2D output_src;
layout(binding=1) uniform texture2D output_exposure;
layout(binding=2) uniform texture3D output_lut;
layout(binding=0) uniform sampler output_smp;
layout(binding=0) uniform output_params {
    vec4 params;   // x = tone mapping (TONE_*), y = exposure, z = 1: multiply it by output_exposure's r,
                   //   w = 1: no sRGB encoding (for a target that is sRGB itself, or more passes in linear)
    vec4 params2;  // x = 1: into Display P3's colors, y = HDR: the screen's peak (times SDR white;
                   //   0: SDR), the result left linear, z = 1: through the grade's LUT (see
                   //   grade.glsl) instead of the tone mapping, w = the LUT's size
};

in vec2 uv;
out vec4 frag_color;

void main() {
    vec4 c = texture(sampler2D(output_src, output_smp), uv);
    float exposure = params.y;
    if (params.z > 0.5) exposure *= texelFetch(sampler2D(output_exposure, output_smp), ivec2(0), 0).r;
    if (params2.z > 0.5 && params2.y <= 0.0) {
        float n = params2.w;
        vec3 at = grade_shape(c.rgb * exposure) * ((n - 1.0) / n) + 0.5 / n;
        vec3 s = textureLod(sampler3D(output_lut, output_smp), at, 0.0).rgb;
        frag_color = vec4(params.w > 0.5 ? srgb_to_linear(s) : s, c.a);
        return;
    }
    vec3 col = tone_map_gamut(c.rgb, int(params.x), exposure, params2.x > 0.5);
    if (params2.y > 0.0) {
        frag_color = vec4(col * hdr_boost(c.rgb, exposure, params2.y, col), c.a);
        return;
    }
    frag_color = vec4(params.w > 0.5 ? col : linear_to_srgb(col), c.a);
}
@end

@program output output_vs output_fs
