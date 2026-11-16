// Tone mapping and output encoding, as three.js's tonemapping_pars_fragment and
// colorspace_pars_fragment: the material shader uses it when it draws into the window or an
// 8-bit target, OutputPass after post-processing in linear light. An application's own passes
// can @include it too (`@include_block tonemapping` inside a shader block).
//
// `mode` is the ToneMapping enum's ordinal (renderer.c3); `exposure` is
// renderer.tone_mapping_exposure.
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

vec3 aces_filmic(vec3 c) {
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
    c = aces_out * c;
    return clamp(c, 0.0, 1.0);
}

// AgX (Troy Sobotka), as three.js's AgXToneMapping: bright light desaturates towards white
// instead of skewing to yellow and cyan.
vec3 agx_contrast(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 agx(vec3 c) {
    const mat3 srgb_to_rec2020 = mat3(
        vec3(0.6274, 0.0691, 0.0164),
        vec3(0.3293, 0.9195, 0.0880),
        vec3(0.0433, 0.0113, 0.8956));
    const mat3 rec2020_to_srgb = mat3(
        vec3(1.6605, -0.1246, -0.0182),
        vec3(-0.5876, 1.1329, -0.1006),
        vec3(-0.0728, -0.0083, 1.1187));
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
    c = pow(max(c, vec3(0.0)), vec3(2.2));
    return clamp(rec2020_to_srgb * c, 0.0, 1.0);
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
@end
