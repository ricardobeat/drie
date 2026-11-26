// ColorGrade (grading.c3): bakes the grade, the tone mapping and the output encoding into a
// 3D LUT that OutputPass reads through a log shaper.
// A slice of the LUT a pass: each texel is a scene color (linear light, after the exposure)
// encoded by grade_shape (log2 stops round middle grey), graded:
//   - white balance, in LMS (the cones' responses)
//   - contrast round middle grey, in log; gain (a multiply), lift (an offset of the log
//     encoding) and power (a power of it), as ASC CDL's slope, offset and power
//   - saturation, a turn of the hue, and the shadows and highlights tinted (split toning), in
//     Oklab, where these keep the lightness and the hue even
// then tone mapped (tonemapping.glsl) and sRGB encoded; a look (a .cube LUT on the sRGB
// picture) after, if there is one.
// Regenerate generated/grade.c3 with: just shaders
@header module drie::postprocessing;
@header import drie;

@ctype vec4 Vec4

@include tonemapping.glsl


@vs grade_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@fs grade_fs
@include_block tonemapping
@include_block grade_shaper
layout(binding=0) uniform texture3D grade_look;
layout(binding=0) uniform sampler grade_smp;   // linear, clamped
layout(binding=0) uniform grade_params {
    vec4 lut;          // x = the slice, y = the LUT's size, z = tone mapping (TONE_*), w = 1: Display P3
    vec4 balance;      // xyz = white balance's LMS gains
    vec4 lift;         // xyz = lift, w = contrast
    vec4 power;        // xyz = power, w = saturation
    vec4 gain;         // xyz = gain, w = the hue's turn (radians)
    vec4 shadows;      // xy = the shadows' tint (Oklab a, b), z = the look's strength, w = its size
    vec4 highlights;   // xy = the highlights' tint (Oklab a, b)
};

out vec4 frag_color;

vec3 lin_to_lms(vec3 c) {
    return vec3(dot(c, vec3(3.90405e-1, 5.49941e-1, 8.92632e-3)),
                dot(c, vec3(7.08416e-2, 9.63172e-1, 1.35775e-3)),
                dot(c, vec3(2.31082e-2, 1.28021e-1, 9.36245e-1)));
}

vec3 lms_to_lin(vec3 c) {
    return vec3(dot(c, vec3(2.85847e+0, -1.62879e+0, -2.48910e-2)),
                dot(c, vec3(-2.10182e-1, 1.15820e+0, 3.24281e-4)),
                dot(c, vec3(-4.18120e-2, -1.18169e-1, 1.06867e+0)));
}

// Björn Ottosson's Oklab, from and to linear sRGB
vec3 to_oklab(vec3 c) {
    vec3 l = vec3(dot(c, vec3(0.4122214708, 0.5363325363, 0.0514459929)),
                  dot(c, vec3(0.2119034982, 0.6806995451, 0.1073969566)),
                  dot(c, vec3(0.0883024619, 0.2817188376, 0.6299787005)));
    l = sign(l) * pow(abs(l), vec3(1.0 / 3.0));
    return vec3(dot(l, vec3(0.2104542553, 0.7936177850, -0.0040720468)),
                dot(l, vec3(1.9779984951, -2.4285922050, 0.4505937099)),
                dot(l, vec3(0.0259040371, 0.7827717662, -0.8086757660)));
}

vec3 from_oklab(vec3 c) {
    vec3 l = vec3(c.x + 0.3963377774 * c.y + 0.2158037573 * c.z,
                  c.x - 0.1055613458 * c.y - 0.0638541728 * c.z,
                  c.x - 0.0894841775 * c.y - 1.2914855480 * c.z);
    l = l * l * l;
    return vec3(dot(l, vec3(4.0767416621, -3.3077115913, 0.2309699292)),
                dot(l, vec3(-1.2684380046, 2.6097574011, -0.3413193965)),
                dot(l, vec3(-0.0041960863, -0.7034186147, 1.7076147010)));
}

void main() {
    float n = lut.y;
    vec3 s = vec3(floor(gl_FragCoord.xy), lut.x) / (n - 1.0);
    vec3 c = grade_unshape(s);
    c = max(lms_to_lin(lin_to_lms(c) * balance.xyz), vec3(0.0));
    c = GRADE_GREY * pow(c / GRADE_GREY, vec3(lift.w));
    c *= gain.xyz;
    vec3 e = grade_shape(c);
    e = pow(max(e + lift.xyz * 0.1, vec3(0.0)), power.xyz);
    c = grade_unshape(e);
    vec3 lab = to_oklab(c);
    float cs = cos(gain.w);
    float sn = sin(gain.w);
    lab.yz = vec2(lab.y * cs - lab.z * sn, lab.y * sn + lab.z * cs) * power.w;
    float l = clamp(lab.x, 0.0, 1.0);
    lab.yz += shadows.xy * (1.0 - l) * (1.0 - l) + highlights.xy * l * l;
    c = max(from_oklab(lab), vec3(0.0));
    vec3 col = linear_to_srgb(clamp(tone_map_gamut(c, int(lut.z), 1.0, lut.w > 0.5), 0.0, 1.0));
    if (shadows.z > 0.0) {
        float m = shadows.w;
        vec3 at = col * ((m - 1.0) / m) + 0.5 / m;
        col = mix(col, textureLod(sampler3D(grade_look, grade_smp), at, 0.0).rgb, shadows.z);
    }
    frag_color = vec4(col, 1.0);
}
@end

@program grade grade_vs grade_fs
