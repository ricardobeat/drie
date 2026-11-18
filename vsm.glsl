// Renderer.shadow_type = VSM (renderer.c3, vsm_blur): a shadow map's tile, from its depths to
// their moments, blurred as three's VSMPass: a box of `samples` taps spread over +-`radius`
// texels, first along y (from the depths), then along x (from the first pass's moments).
// Each pass writes r = the mean depth, g = its standard deviation. Regenerate
// generated/vsm.c3 with: just shaders
@header module drie::shaders;
@header import drie;

@ctype vec4 Vec4

@vs vsm_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@fs vsm_fs
@image_sample_type vsm_src unfilterable_float
@sampler_type vsm_src_smp nonfiltering
// the depth map (first pass), or the first pass's moments
layout(binding=0) uniform texture2D vsm_src;
layout(binding=0) uniform sampler vsm_src_smp;
layout(binding=0) uniform vsm_params {
    vec4 vsm_rect;   // the tile, in texels: xy = offset (as gl_FragCoord counts), zw = size
    vec4 vsm_spread; // x = radius (texels), y = samples, z = 1: along x, from moments (0: along
                     //   y, from depths)
};

out vec4 frag_color;

// The source at `p` (texels, along the pass's axis between two texels' centres), kept inside
// the tile: its neighbours belong to other lights (or cascades).
vec2 vsm_at(vec2 p, bool along_x) {
    vec2 lo = vsm_rect.xy;
    vec2 hi = vsm_rect.xy + vsm_rect.zw - 1.0;
    float t = along_x ? p.x : p.y;
    float f = fract(t);
    vec2 a = clamp(along_x ? vec2(floor(t), p.y) : vec2(p.x, floor(t)), lo, hi);
    vec2 b = clamp(along_x ? vec2(floor(t) + 1.0, p.y) : vec2(p.x, floor(t) + 1.0), lo, hi);
    vec4 va = texelFetch(sampler2D(vsm_src, vsm_src_smp), ivec2(a), 0);
    vec4 vb = texelFetch(sampler2D(vsm_src, vsm_src_smp), ivec2(b), 0);
    vec2 v = mix(va.rg, vb.rg, f);
    // (a depth map's other channels are nothing)
    return along_x ? v : vec2(v.x, 0.0);
}

void main() {
    bool along_x = vsm_spread.z > 0.5;
    float samples = max(vsm_spread.y, 1.0);
    float stride = samples <= 1.0 ? 0.0 : 2.0 / (samples - 1.0);
    float start = samples <= 1.0 ? 0.0 : -1.0;
    vec2 here = floor(gl_FragCoord.xy);
    float mean = 0.0;
    float sq_mean = 0.0;
    for (int i = 0; i < 64; i++) {
        if (float(i) >= samples) break;
        float o = (start + float(i) * stride) * vsm_spread.x;
        vec2 m = vsm_at(here + (along_x ? vec2(o, 0.0) : vec2(0.0, o)), along_x);
        mean += m.x;
        // the depths' squares: the first pass's have no spread yet
        sq_mean += m.y * m.y + m.x * m.x;
    }
    mean /= samples;
    sq_mean /= samples;
    frag_color = vec4(mean, sqrt(max(0.0, sq_mean - mean * mean)), 0.0, 1.0);
}
@end

@program vsm vsm_vs vsm_fs
