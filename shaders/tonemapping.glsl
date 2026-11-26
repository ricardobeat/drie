// Shared tone mapping, gamut conversion, HDR boost, sRGB encoding, and color-grade log shaper
// blocks (`tonemapping` and `grade_shaper`), matching three.js's tone mapping curves. Included by
// shaders/shader.glsl, shaders/shader_deferred.glsl, shaders/output.glsl, and shaders/grade.glsl
// (regenerate their generated/*.c3 bindings with `just shaders`). `mode` is drei::ToneMapping's
// ordinal and `exposure` is Renderer.tone_mapping_exposure.
@block tonemapping
#define TONE_NONE 0
#define TONE_LINEAR 1
#define TONE_REINHARD 2
#define TONE_CINEON 3
#define TONE_ACES_FILMIC 4
#define TONE_AGX 5
#define TONE_NEUTRAL 6

vec3 linear_to_srgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

// ACES filmic, the same fit as three.js's ACESFilmicToneMapping (Stephen Hill's).
vec3 rrt_odt_fit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

// Unclamped: colors more saturated than sRGB come out below 0 in a channel (see tone_map_gamut).
vec3 aces_filmic_wide(vec3 c) {
    const mat3 aces_in = mat3(
        vec3(0.59719, 0.07600, 0.02840),
        vec3(0.35458, 0.90834, 0.13383),
        vec3(0.04823, 0.01566, 0.83777));
    const mat3 aces_out = mat3(
        vec3(1.60475, -0.10208, -0.00327),
        vec3(-0.53108, 1.10813, -0.07276),
        vec3(-0.07367, -0.00605, 1.07602));
    c *= 1.0 / 0.6;
    c = aces_in * c;
    c = rrt_odt_fit(c);
    return aces_out * c;
}

vec3 aces_filmic(vec3 c) {
    return clamp(aces_filmic_wide(c), 0.0, 1.0);
}

// AgX (Troy Sobotka), as three.js's AgXToneMapping: bright light desaturates towards white
// instead of skewing to yellow and cyan.
vec3 agx_contrast(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

// In Rec. 2020 (see tone_map_gamut).
vec3 agx_rec2020(vec3 c) {
    const mat3 srgb_to_rec2020 = mat3(
        vec3(0.6274, 0.0691, 0.0164),
        vec3(0.3293, 0.9195, 0.0880),
        vec3(0.0433, 0.0113, 0.8956));
    const mat3 inset = mat3(
        vec3(0.856627153315983, 0.137318972929847, 0.11189821299995),
        vec3(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
        vec3(0.0482516061458583, 0.101439036467562, 0.811302368396859));
    const mat3 outset = mat3(
        vec3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
        vec3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
        vec3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
    const float min_ev = -12.47393;
    const float max_ev = 4.026069;
    c = inset * (srgb_to_rec2020 * c);
    c = clamp((log2(max(c, 1e-10)) - min_ev) / (max_ev - min_ev), 0.0, 1.0);
    c = outset * agx_contrast(c);
    return pow(max(c, vec3(0.0)), vec3(2.2));
}

const mat3 REC2020_TO_SRGB = mat3(
    vec3(1.6605, -0.1246, -0.0182),
    vec3(-0.5876, 1.1329, -0.1006),
    vec3(-0.0728, -0.0083, 1.1187));

vec3 agx(vec3 c) {
    return clamp(REC2020_TO_SRGB * agx_rec2020(c), 0.0, 1.0);
}

// Khronos PBR Neutral, as three.js's NeutralToneMapping: base colors stay as they are up to
// a bright highlight.
vec3 neutral(vec3 c) {
    const float start = 0.8 - 0.04;
    const float desaturation = 0.15;
    float x = min(c.r, min(c.g, c.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    c -= offset;
    float peak = max(c.r, max(c.g, c.b));
    if (peak < start) return c;
    float d = 1.0 - start;
    float new_peak = 1.0 - d * d / (peak + d - start);
    c *= new_peak / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(c, vec3(new_peak), g);
}

vec3 cineon(vec3 c) {
    c = max(vec3(0.0), c - 0.004);
    return pow((c * (6.2 * c + 0.5)) / (c * (6.2 * c + 1.7) + 0.06), vec3(2.2));
}

// Linear light (scaled by `exposure`) into 0..1, still linear.
vec3 tone_map(vec3 c, int mode, float exposure) {
    if (mode == TONE_NONE) return c;
    c = max(c, vec3(0.0)) * exposure;
    if (mode == TONE_LINEAR) return clamp(c, 0.0, 1.0);
    if (mode == TONE_REINHARD) return clamp(c / (vec3(1.0) + c), 0.0, 1.0);
    if (mode == TONE_CINEON) return cineon(c);
    if (mode == TONE_ACES_FILMIC) return aces_filmic(c);
    if (mode == TONE_AGX) return agx(c);
    if (mode == TONE_NEUTRAL) return neutral(c);
    return c;
}

// Linear sRGB (Rec. 709) to linear Display P3, and back.
const mat3 SRGB_TO_P3 = mat3(
    vec3(0.8225, 0.0332, 0.0171),
    vec3(0.1774, 0.9669, 0.0724),
    vec3(0.0000, 0.0000, 0.9108));
const mat3 P3_TO_SRGB = mat3(
    vec3(1.2249, -0.0420, -0.0196),
    vec3(-0.2247, 1.0419, -0.0786),
    vec3(0.0000, 0.0000, 1.0979));

// tone_map into Display P3 (p3 = true; linear) instead of sRGB: the ACES and AgX curves clip
// to P3's colors at the end instead of sRGB's, so a saturated light keeps more of its color.
vec3 tone_map_gamut(vec3 c, int mode, float exposure, bool p3) {
    if (!p3) return tone_map(c, mode, exposure);
    if (mode == TONE_NONE) return SRGB_TO_P3 * c;
    vec3 e = max(c, vec3(0.0)) * exposure;
    if (mode == TONE_ACES_FILMIC) return clamp(SRGB_TO_P3 * aces_filmic_wide(e), 0.0, 1.0);
    if (mode == TONE_AGX) return clamp(SRGB_TO_P3 * (REC2020_TO_SRGB * agx_rec2020(e)), 0.0, 1.0);
    return SRGB_TO_P3 * tone_map(c, mode, exposure);
}

// HDR output: how much brighter than `shown` (its SDR tone mapping, linear, 0..1) a pixel of
// linear light `c` (times `exposure`) shows on a screen that goes up to `peak` times SDR white.
// The picture stays as in SDR up to its upper midtones (so a grade's shadows and contrast are
// kept); from there its brightest channel goes over to following the light, rolling off into
// the peak (peak * (1 - e^(-1.5 x / peak))), with the hue of the SDR picture.
float hdr_boost(vec3 c, float exposure, float peak, vec3 shown) {
    if (peak <= 1.0) return 1.0;
    vec3 e = max(c, vec3(0.0)) * exposure;
    float x = max(e.r, max(e.g, e.b));
    float target = peak * (1.0 - exp(-1.5 * x / peak));
    float now = max(shown.r, max(shown.g, shown.b));
    float boost = max(1.0, target / max(now, 1e-4));
    return mix(1.0, boost, smoothstep(0.45, 0.9, now));
}

#define DISPLAY_SRGB 0
#define DISPLAY_P3 1
#define DISPLAY_HDR 2

// A picture ready to show on an sRGB screen (sRGB-encoded, 0..1) as `display` takes it:
// DISPLAY_SRGB as it is, DISPLAY_P3 encoded in Display P3 (the same colors), DISPLAY_HDR in
// extended linear Display P3 (1 = SDR white).
vec3 encode_for_display(vec3 srgb, int display) {
    if (display == DISPLAY_SRGB) return srgb;
    vec3 p3 = SRGB_TO_P3 * srgb_to_linear(srgb);
    return display == DISPLAY_HDR ? p3 : linear_to_srgb(max(p3, vec3(0.0)));
}
@end

// The log shaper of the color grading LUT (shaders/grade.glsl, shaders/output.glsl)
@block grade_shaper
// the log shaper: GRADE_LOW stops under middle grey to GRADE_LOW + GRADE_RANGE over it, 0..1
const float GRADE_GREY = 0.18;
const float GRADE_LOW = 10.0;
const float GRADE_RANGE = 17.0;

vec3 grade_shape(vec3 c) {
    return clamp((log2(max(c, vec3(1e-10)) / GRADE_GREY) + GRADE_LOW) / GRADE_RANGE, 0.0, 1.0);
}

vec3 grade_unshape(vec3 s) {
    return GRADE_GREY * exp2(s * GRADE_RANGE - GRADE_LOW);
}
@end
