// OutputPass (drie::postprocessing): final pass for a linear-light post-processing chain. Applies
// renderer tone mapping and exposure (or a ColorGrade 3D LUT) and encodes for sRGB, Display P3,
// or extended-linear HDR while preserving alpha. Regenerate generated/output.c3 with `just shaders`.
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
layout(binding=3) uniform texture2D output_noise;   // 64x64 blue noise (src/postprocessing/blue_noise.c3)
layout(binding=0) uniform sampler output_smp;
layout(binding=0) uniform output_params {
    vec4 params;   // x = tone mapping (TONE_*), y = exposure, z = 1: multiply it by output_exposure's r,
                   //   w = 1: no sRGB encoding (for a target that is sRGB itself, or more passes in linear)
    vec4 params2;  // x = 1: into Display P3's colors, y = HDR: the screen's peak (times SDR white;
                   //   0: SDR), the result left linear, z = 1: through the grade's LUT (see
                   //   shaders/grade.glsl) instead of the tone mapping, w = the LUT's size
    vec4 params3;  // x = dither, in steps of the 8-bit target (0: none), y = this frame's shift of the noise
};

in vec2 uv;
out vec4 frag_color;

// Blue-noise dither of an encoded (sRGB or P3) color before the 8-bit target rounds it: +-half a
// step times params3.x, the noise moved on every frame (by the golden ratio) for TAA and the eye
// to average out.
vec3 dither(vec3 c) {
    if (params3.x <= 0.0) return c;
    float n = texelFetch(sampler2D(output_noise, output_smp), ivec2(gl_FragCoord.xy) & 63, 0).r;
    vec3 d = fract(n + params3.y + vec3(0.0, 0.381966, 0.763932)) - 0.5;
    return c + d * (params3.x / 255.0);
}

// Dithered in the encoding the target ends up in; linear_target: the target encodes it (an sRGB
// image) or more passes follow in linear.
vec3 dither_out(vec3 encoded, bool linear_target) {
    vec3 c = dither(encoded);
    return linear_target ? srgb_to_linear(max(c, 0.0)) : c;
}

void main() {
    vec4 c = texture(sampler2D(output_src, output_smp), uv);
    float exposure = params.y;
    if (params.z > 0.5) exposure *= texelFetch(sampler2D(output_exposure, output_smp), ivec2(0), 0).r;
    if (params2.z > 0.5 && params2.y <= 0.0) {
        float n = params2.w;
        vec3 at = grade_shape(c.rgb * exposure) * ((n - 1.0) / n) + 0.5 / n;
        vec3 s = textureLod(sampler3D(output_lut, output_smp), at, 0.0).rgb;
        frag_color = vec4(dither_out(s, params.w > 0.5), c.a);
        return;
    }
    vec3 col = tone_map_gamut(c.rgb, int(params.x), exposure, params2.x > 0.5);
    if (params2.y > 0.0) {
        frag_color = vec4(col * hdr_boost(c.rgb, exposure, params2.y, col), c.a);
        return;
    }
    frag_color = vec4(dither_out(linear_to_srgb(col), params.w > 0.5), c.a);
}
@end

@program output output_vs output_fs
