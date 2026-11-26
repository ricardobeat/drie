// BokehPass (postprocessing.c3): depth of field, what is nearer or further than the focus going
// soft as through a lens's wide aperture. three's BokehPass gathers a ring of taps round each
// pixel; this is made as games make it (Jimenez 2014, "Next generation post processing in Call
// of Duty: Advanced Warfare"; Unreal's diaphragm depth of field: a gather, and a scatter of
// what is small and bright):
//
//   focus:     the distance in focus: the one set, or the autofocus's (the middle of the
//              picture, followed with a delay), in a 1x1 texture
//   tiles:     the largest blur in each 16x16 block of the picture
//   reach:     ...and the largest that reaches each block from the blocks round it: how far
//              the gather's taps need to look from there (nowhere, where all is sharp)
//   prefilter: the picture at half its size, of what is soft only, premultiplied (alpha: how
//              much of it is): a sharp edge in front of something soft doesn't bleed into it
//   sparks:    what is much brighter than what is round it (a mote of dust in a beam, a lamp
//              far off), and soft enough to be drawn as a disc of its own: its light, and in
//              alpha its distance
//   rest:      the half picture without the sparks; shrink: the same at a quarter of the size
//   clean:     the whole picture without them, each in the colour of what is round it (where
//              little round a tap is soft, the gather reads the picture itself)
//   gather:    each pixel takes the light of what spreads over it, from taps in a disc as
//              wide as the reach, each averaged over its share of the disc (from the half and
//              quarter pictures), so what spreads makes an even disc and not a few dots
//   scatter:   each spark drawn as a disc as wide as its blur with its light spread over it,
//              added to the gather's picture: no gather of a few taps can make an even disc
//              out of a speck
//
// Regenerate generated/bokeh.c3 with: just shaders
@header module drie::postprocessing;
@header import drie;

@ctype vec4 Vec4

@vs bokeh_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@block bokeh_common
layout(binding=0) uniform bokeh_params {
    vec4 size;    // xy = the picture's size, z = a seed for the noise, w = 1: row 0 is the top
    vec4 tune;    // x = the gather's taps, y = aperture (pixels * m), z = maxblur (pixels),
                  //   w = 1: scatter the sparks
    vec4 focus;   // x = the distance in focus (m), y = the share of the way the autofocus goes
                  //   in a frame, z = 1: autofocus, w = 1: it jumps there (no frame before)
    vec4 range;   // xy = the nearest and furthest the autofocus focuses (m)
};

const float PI = 3.14159265;
// the side of the blocks of the tiles and the reach
const int TILE = 16;

// The blur of something at distance z (m) with the lens focused at f, in pixels: a thin
// lens's circle of confusion.
float blur_of(float z, float f) {
    return min(tune.y / f * abs(1.0 - f / z), tune.z);
}

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float noise(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}
@end

@fs bokeh_focus_fs
@include_block bokeh_common
layout(binding=0) uniform texture2D bokeh_focus_depth;
layout(binding=1) uniform texture2D bokeh_focus_prev;
layout(binding=0) uniform sampler bokeh_focus_smp;

out vec4 frag_color;

void main() {
    if (focus.z < 0.5) {
        frag_color = vec4(focus.x, 0.0, 0.0, 1.0);
        return;
    }
    // the middle of the picture, as an autofocus sensor sees it: a 7 x 7 grid of samples over
    // its middle fifth, weighted towards the centre, and the weighted median of their distances
    // (as logarithms). A blade of grass or a twig that crosses a few samples near the lens can't
    // take the focus from the ground or the wall that fills the rest, as an average lets it.
    float lz[49];
    float w[49];
    float total = 0.0;
    for (int y = 0; y < 7; y++) {
        for (int x = 0; x < 7; x++) {
            vec2 d = vec2(float(x), float(y)) / 3.0 - 1.0;
            vec2 uv = vec2(0.5) + d * 0.1;
            float z = texelFetch(sampler2D(bokeh_focus_depth, bokeh_focus_smp), ivec2(uv * size.xy), 0).g;
            int i = y * 7 + x;
            lz[i] = log(clamp(z, range.x, range.y));
            w[i] = exp(-dot(d, d) * 1.2);
            total += w[i];
        }
    }
    float aim = lz[24];
    float best = 1e9;
    for (int i = 0; i < 49; i++) {
        // the share of the weight nearer than this sample, and as near: the median is where
        // half lies on either side
        float below = 0.0;
        float same = 0.0;
        for (int j = 0; j < 49; j++) {
            below += lz[j] < lz[i] ? w[j] : 0.0;
            same += lz[j] == lz[i] ? w[j] : 0.0;
        }
        float off = abs(below + 0.5 * same - 0.5 * total);
        if (off < best) {
            best = off;
            aim = lz[i];
        }
    }
    float prev = log(clamp(texelFetch(sampler2D(bokeh_focus_prev, bokeh_focus_smp), ivec2(0), 0).r, range.x, range.y));
    float v = focus.w > 0.5 ? aim : mix(prev, aim, focus.y);
    frag_color = vec4(exp(v), 0.0, 0.0, 1.0);
}
@end

@fs bokeh_tiles_fs
@include_block bokeh_common
layout(binding=0) uniform texture2D bokeh_tiles_depth;
layout(binding=1) uniform texture2D bokeh_tiles_focus;
layout(binding=0) uniform sampler bokeh_tiles_smp;

out vec4 frag_color;

void main() {
    float f = texelFetch(sampler2D(bokeh_tiles_focus, bokeh_tiles_smp), ivec2(0), 0).r;
    ivec2 last = ivec2(size.xy) - 1;
    ivec2 base = ivec2(gl_FragCoord.xy) * TILE;
    float m = 0.0;
    for (int y = 0; y < TILE; y++) {
        for (int x = 0; x < TILE; x++) {
            float z = texelFetch(sampler2D(bokeh_tiles_depth, bokeh_tiles_smp), min(base + ivec2(x, y), last), 0).g;
            m = max(m, blur_of(z, f));
        }
    }
    frag_color = vec4(m, 0.0, 0.0, 1.0);
}
@end

@fs bokeh_reach_fs
@include_block bokeh_common
layout(binding=0) uniform texture2D bokeh_reach_tiles;
layout(binding=0) uniform sampler bokeh_reach_smp;

out vec4 frag_color;

// a block's own largest blur, and those of the blocks round it whose blur reaches this one
void main() {
    ivec2 t = ivec2(gl_FragCoord.xy);
    ivec2 last = textureSize(sampler2D(bokeh_reach_tiles, bokeh_reach_smp), 0) - 1;
    int k = min(int(ceil(tune.z / float(TILE))), 3);
    float m = 0.0;
    for (int y = -3; y <= 3; y++) {
        for (int x = -3; x <= 3; x++) {
            int ring = max(abs(x), abs(y));
            if (ring > k) continue;
            float b = texelFetch(sampler2D(bokeh_reach_tiles, bokeh_reach_smp), clamp(t + ivec2(x, y), ivec2(0), last), 0).r;
            if (b > float((ring - 1) * TILE)) m = max(m, b);
        }
    }
    frag_color = vec4(m, 0.0, 0.0, 1.0);
}
@end

@fs bokeh_prefilter_fs
@include_block bokeh_common
layout(binding=0) uniform texture2D bokeh_prefilter_color;
layout(binding=1) uniform texture2D bokeh_prefilter_depth;
layout(binding=2) uniform texture2D bokeh_prefilter_focus;
layout(binding=0) uniform sampler bokeh_prefilter_smp;

out vec4 frag_color;

// Half the picture's size: a 4x4 tent of its pixels, of those only that are soft by more than
// a pixel or two, premultiplied (alpha: how much of the tent they are).
void main() {
    float f = texelFetch(sampler2D(bokeh_prefilter_focus, bokeh_prefilter_smp), ivec2(0), 0).r;
    ivec2 last = textureSize(sampler2D(bokeh_prefilter_color, bokeh_prefilter_smp), 0) - 1;
    ivec2 base = ivec2(gl_FragCoord.xy) * 2 - 1;
    vec4 sum = vec4(0.0);
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            ivec2 q = clamp(base + ivec2(x, y), ivec2(0), last);
            float t = (x == 0 || x == 3 ? 1.0 : 3.0) * (y == 0 || y == 3 ? 1.0 : 3.0);
            float z = texelFetch(sampler2D(bokeh_prefilter_depth, bokeh_prefilter_smp), q, 0).g;
            float soft = smoothstep(1.0, 3.0, blur_of(z, f));
            sum += vec4(texelFetch(sampler2D(bokeh_prefilter_color, bokeh_prefilter_smp), q, 0).rgb, 1.0) * (soft * t);
        }
    }
    frag_color = sum / 64.0;
}
@end

@fs bokeh_sparks_fs
@include_block bokeh_common
layout(binding=0) uniform texture2D bokeh_sparks_soft;    // the prefilter's
layout(binding=1) uniform texture2D bokeh_sparks_color;   // the picture
layout(binding=2) uniform texture2D bokeh_sparks_depth;
layout(binding=3) uniform texture2D bokeh_sparks_focus;
layout(binding=0) uniform sampler bokeh_sparks_smp;

out vec4 frag_color;

vec3 g_around;
float g_brightest;

void ring_add(ivec2 t, ivec2 last) {
    vec3 c = texelFetch(sampler2D(bokeh_sparks_soft, bokeh_sparks_smp), clamp(t, ivec2(0), last), 0).rgb;
    g_around += c;
    g_brightest = max(g_brightest, luminance(c));
}

void main() {
    frag_color = vec4(0.0);
    if (tune.w < 0.5) return;
    ivec2 t = ivec2(gl_FragCoord.xy);
    ivec2 last = textureSize(sampler2D(bokeh_sparks_soft, bokeh_sparks_smp), 0) - 1;
    vec3 s = texelFetch(sampler2D(bokeh_sparks_soft, bokeh_sparks_smp), t, 0).rgb;
    // what is round it: the ring two texels out (a speck of a pixel or two is in 2x2 of these,
    // none of them on the ring of any of the others)
    g_around = vec3(0.0);
    g_brightest = 0.0;
    for (int i = -2; i <= 2; i++) {
        ring_add(t + ivec2(i, -2), last);
        ring_add(t + ivec2(i, 2), last);
    }
    for (int i = -1; i <= 1; i++) {
        ring_add(t + ivec2(-2, i), last);
        ring_add(t + ivec2(2, i), last);
    }
    vec3 around = g_around / 16.0;
    // a speck on its own: much brighter than anything on the ring (a blade of grass or an edge
    // runs on through it, and is gathered like the rest), by enough to see once spread out
    float l = luminance(s);
    float k = smoothstep(1.5, 3.0, l / max(g_brightest, 1e-5)) * smoothstep(0.002, 0.01, l - luminance(around));
    if (k <= 0.0) return;
    // how far it is: the brightest of the soft pixels under the texel (its light is theirs: a
    // lamp in focus next to a mote out of focus is not the spark)
    float f = texelFetch(sampler2D(bokeh_sparks_focus, bokeh_sparks_smp), ivec2(0), 0).r;
    ivec2 lastp = textureSize(sampler2D(bokeh_sparks_color, bokeh_sparks_smp), 0) - 1;
    ivec2 base = t * 2 - 1;
    float best = -1.0;
    float z = 1.0;
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            ivec2 q = clamp(base + ivec2(x, y), ivec2(0), lastp);
            float zq = texelFetch(sampler2D(bokeh_sparks_depth, bokeh_sparks_smp), q, 0).g;
            float lq = luminance(texelFetch(sampler2D(bokeh_sparks_color, bokeh_sparks_smp), q, 0).rgb)
                     * smoothstep(1.0, 3.0, blur_of(zq, f));
            if (lq > best) {
                best = lq;
                z = zq;
            }
        }
    }
    // soft by a few pixels at least: a smaller disc, the gather makes well enough
    k *= smoothstep(2.0, 4.0, blur_of(z, f));
    frag_color = vec4(max(s - around, vec3(0.0)) * k, z);
}
@end

@fs bokeh_rest_fs
layout(binding=0) uniform texture2D bokeh_rest_soft;
layout(binding=1) uniform texture2D bokeh_rest_sparks;
layout(binding=0) uniform sampler bokeh_rest_smp;

out vec4 frag_color;

void main() {
    ivec2 t = ivec2(gl_FragCoord.xy);
    vec4 s = texelFetch(sampler2D(bokeh_rest_soft, bokeh_rest_smp), t, 0);
    vec3 spark = texelFetch(sampler2D(bokeh_rest_sparks, bokeh_rest_smp), t, 0).rgb;
    frag_color = vec4(max(s.rgb - spark, vec3(0.0)), s.a);
}
@end

@fs bokeh_clean_fs
@include_block bokeh_common
layout(binding=0) uniform texture2D bokeh_clean_color;
layout(binding=1) uniform texture2D bokeh_clean_sparks;
layout(binding=2) uniform texture2D bokeh_clean_depth;
layout(binding=3) uniform texture2D bokeh_clean_focus;
layout(binding=0) uniform sampler bokeh_clean_smp;

out vec4 frag_color;

vec3 color_at(ivec2 q, ivec2 last) {
    return texelFetch(sampler2D(bokeh_clean_color, bokeh_clean_smp), clamp(q, ivec2(0), last), 0).rgb;
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec4 c = texelFetch(sampler2D(bokeh_clean_color, bokeh_clean_smp), px, 0);
    frag_color = c;
    // the 2x2 texels of the sparks this pixel is under (a texel's tent covers 4x4 pixels)
    ivec2 lastt = textureSize(sampler2D(bokeh_clean_sparks, bokeh_clean_smp), 0) - 1;
    ivec2 t = (px + 1) / 2;
    float e = 0.0;
    for (int y = -1; y <= 0; y++) {
        for (int x = -1; x <= 0; x++) {
            e = max(e, luminance(texelFetch(sampler2D(bokeh_clean_sparks, bokeh_clean_smp), clamp(t + ivec2(x, y), ivec2(0), lastt), 0).rgb));
        }
    }
    if (e <= 0.0) return;
    // what is round it: the ring of pixels three out
    ivec2 last = ivec2(size.xy) - 1;
    vec3 ring = (color_at(px + ivec2(-3, -3), last) + color_at(px + ivec2(0, -3), last) + color_at(px + ivec2(3, -3), last)
               + color_at(px + ivec2(-3, 0), last) + color_at(px + ivec2(3, 0), last)
               + color_at(px + ivec2(-3, 3), last) + color_at(px + ivec2(0, 3), last) + color_at(px + ivec2(3, 3), last)) / 8.0;
    // a soft pixel much brighter than that is (part of) the spark
    float f = texelFetch(sampler2D(bokeh_clean_focus, bokeh_clean_smp), ivec2(0), 0).r;
    float soft = smoothstep(1.0, 3.0, blur_of(texelFetch(sampler2D(bokeh_clean_depth, bokeh_clean_smp), px, 0).g, f));
    float k = soft * smoothstep(1.5, 3.0, luminance(c.rgb) / max(luminance(ring), 1e-5));
    frag_color = vec4(mix(c.rgb, ring, k), c.a);
}
@end

@fs bokeh_shrink_fs
layout(binding=0) uniform texture2D bokeh_shrink_src;
layout(binding=0) uniform sampler bokeh_shrink_smp;   // linear, clamped

out vec4 frag_color;

// half the source's size (premultiplied as it is): four bilinear taps round each pixel, a 4x4
// tent of the source's
void main() {
    vec2 t = 1.0 / vec2(textureSize(sampler2D(bokeh_shrink_src, bokeh_shrink_smp), 0));
    vec2 uv = gl_FragCoord.xy * 2.0 * t;
    frag_color = (textureLod(sampler2D(bokeh_shrink_src, bokeh_shrink_smp), uv + vec2(-t.x, -t.y), 0.0)
                + textureLod(sampler2D(bokeh_shrink_src, bokeh_shrink_smp), uv + vec2(t.x, -t.y), 0.0)
                + textureLod(sampler2D(bokeh_shrink_src, bokeh_shrink_smp), uv + vec2(-t.x, t.y), 0.0)
                + textureLod(sampler2D(bokeh_shrink_src, bokeh_shrink_smp), uv + vec2(t.x, t.y), 0.0)) * 0.25;
}
@end

@fs bokeh_gather_fs
@include_block bokeh_common
layout(binding=0) uniform texture2D bokeh_gather_color;     // the picture without the sparks (clean)
layout(binding=1) uniform texture2D bokeh_gather_depth;
layout(binding=2) uniform texture2D bokeh_gather_focus;
layout(binding=3) uniform texture2D bokeh_gather_half;      // what is soft, at half the size (rest)
layout(binding=4) uniform texture2D bokeh_gather_quarter;   // ...and at a quarter (shrink)
layout(binding=5) uniform texture2D bokeh_gather_reach;
layout(binding=0) uniform sampler bokeh_gather_smp;         // linear, clamped

out vec4 frag_color;

// (the most taps the loop goes to)
const int MAX_TAPS = 64;

float g_focus;

float depth_at(ivec2 px) {
    return texelFetch(sampler2D(bokeh_gather_depth, bokeh_gather_smp), clamp(px, ivec2(0), ivec2(size.xy) - 1), 0).g;
}

// What is soft of the picture at uv, averaged over about `foot` pixels (where nothing is soft,
// the picture there).
vec3 color_over(vec2 uv, float foot) {
    if (foot < 1.5) return textureLod(sampler2D(bokeh_gather_color, bokeh_gather_smp), uv, 0.0).rgb;
    vec4 h = textureLod(sampler2D(bokeh_gather_half, bokeh_gather_smp), uv, 0.0);
    vec4 avg = foot < 3.0
        ? mix(vec4(textureLod(sampler2D(bokeh_gather_color, bokeh_gather_smp), uv, 0.0).rgb, 1.0) * h.a, h, (foot - 1.5) / 1.5)
        : mix(h, textureLod(sampler2D(bokeh_gather_quarter, bokeh_gather_smp), uv, 0.0), clamp((foot - 3.0) / 3.0, 0.0, 1.0));
    if (avg.a >= 0.05) return avg.rgb / avg.a;
    vec3 c = textureLod(sampler2D(bokeh_gather_color, bokeh_gather_smp), uv, 0.0).rgb;
    return avg.a > 1e-4 ? mix(c, avg.rgb / avg.a, smoothstep(0.0, 0.05, avg.a)) : c;
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec4 c0 = texelFetch(sampler2D(bokeh_gather_color, bokeh_gather_smp), px, 0);
    // the taps go as far as the largest blur that reaches here, and no further: as close
    // together as they can be
    float reach = min(texelFetch(sampler2D(bokeh_gather_reach, bokeh_gather_smp), px / TILE, 0).r, tune.z);
    if (tune.y <= 0.0 || reach < 0.5) {
        frag_color = c0;
        return;
    }
    g_focus = texelFetch(sampler2D(bokeh_gather_focus, bokeh_gather_smp), ivec2(0), 0).r;
    float z0 = depth_at(px);
    float b0 = blur_of(z0, g_focus);
    int taps = clamp(int(tune.x), 4, MAX_TAPS);
    // the gap between taps over their disc: each reads a little more than that round it, so
    // that the ones next to each other overlap, and at least as much as its own blur (what is
    // that soft is no sharper than that anyway, and a spark is in what is read there no more)
    float gap = sqrt(PI * reach * reach / float(taps));
    // the pixel itself stands for the middle of the disc, as a tap does for its share
    vec3 sum = color_over(gl_FragCoord.xy / size.xy, b0);
    float wsum = 1.0;
    float turn = noise(gl_FragCoord.xy + size.z * vec2(13.0, 29.0)) * 2.0 * PI;
    for (int i = 0; i < MAX_TAPS; i++) {
        if (i >= taps) break;
        float t = (float(i) + 0.5) / float(taps);
        float a = turn + float(i) * 2.39996;
        float d = reach * sqrt(t);
        vec2 o = vec2(cos(a), sin(a)) * d;
        float z = depth_at(ivec2(gl_FragCoord.xy + o));
        // a neighbour spreads over this pixel if its own blur reaches this far; what lies
        // behind does not spread over something sharp in front of it
        float b = blur_of(z, g_focus);
        if (z > z0) b = min(b, b0);
        float w = smoothstep(d - 1.0, d + 1.0, b);
        if (w <= 0.0) continue;
        sum += color_over((gl_FragCoord.xy + o) / size.xy, max(min(gap * 1.5, d), b)) * w;
        wsum += w;
    }
    // (alpha, which marks the lights, the pixel's own)
    frag_color = vec4(sum / wsum, c0.a);
}
@end

@vs bokeh_scatter_vs
// (bokeh_params's size and tune: a uniform block is the vertex stage's or the fragment's)
layout(binding=0) uniform bokeh_scatter_params {
    vec4 size;
    vec4 tune;
};
layout(binding=0) uniform texture2D bokeh_scatter_sparks;
layout(binding=1) uniform texture2D bokeh_scatter_focus;
layout(binding=0) uniform sampler bokeh_scatter_vsmp;

out vec2 v_off;      // from the disc's middle (pixels)
out vec3 v_light;    // the light of each pixel of the disc
out vec4 v_disc;     // x = its radius (pixels), y = the spark's distance (m), z = the focus (m),
                     //   w = aperture (pixels * m)
out float v_maxblur;

// A quad round each texel of the sparks (6 vertices, one instance a texel): a disc as wide as
// its blur, its light spread over it; nothing where there is no spark.
void main() {
    gl_Position = vec4(2.0, 2.0, 0.5, 1.0);
    v_off = vec2(0.0);
    v_light = vec3(0.0);
    v_disc = vec4(0.0);
    v_maxblur = tune.z;
    ivec2 ts = textureSize(sampler2D(bokeh_scatter_sparks, bokeh_scatter_vsmp), 0);
    ivec2 t = ivec2(gl_InstanceIndex % ts.x, gl_InstanceIndex / ts.x);
    vec4 s = texelFetch(sampler2D(bokeh_scatter_sparks, bokeh_scatter_vsmp), t, 0);
    if (s.r + s.g + s.b <= 0.0) return;
    float f = texelFetch(sampler2D(bokeh_scatter_focus, bokeh_scatter_vsmp), ivec2(0), 0).r;
    float r = max(min(tune.y / f * abs(1.0 - f / s.a), tune.z), 1.0);
    int k = gl_VertexIndex;
    vec2 corner = vec2(k == 1 || k == 2 || k == 4 ? 1.0 : -1.0, k == 2 || k == 4 || k == 5 ? 1.0 : -1.0);
    // the texel's middle, in the picture's pixels (gl_FragCoord's)
    vec2 c = (vec2(t) + 0.5) * 2.0;
    v_off = corner * (r + 1.0);
    vec2 ndc = (c + v_off) / size.xy * 2.0 - 1.0;
    if (size.w > 0.5) ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.5, 1.0);
    // the texel stands for 2x2 of the picture's pixels
    v_light = s.rgb * 4.0 / (3.14159265 * r * r);
    v_disc = vec4(r, s.a, f, tune.y);
}
@end

@fs bokeh_scatter_fs
layout(binding=2) uniform texture2D bokeh_scatter_depth;
layout(binding=1) uniform sampler bokeh_scatter_fsmp;

in vec2 v_off;
in vec3 v_light;
in vec4 v_disc;
in float v_maxblur;

out vec4 frag_color;

void main() {
    float edge = clamp(v_disc.x + 0.5 - length(v_off), 0.0, 1.0);
    if (edge <= 0.0) discard;
    // something sharp in front of it hides it
    float z = texelFetch(sampler2D(bokeh_scatter_depth, bokeh_scatter_fsmp), ivec2(gl_FragCoord.xy), 0).g;
    if (z < v_disc.y * 0.95) {
        float b = min(v_disc.w / v_disc.z * abs(1.0 - v_disc.z / z), v_maxblur);
        edge *= smoothstep(1.0, 3.0, b);
    }
    frag_color = vec4(v_light * edge, 0.0);
}
@end

@program bokeh_focus bokeh_vs bokeh_focus_fs
@program bokeh_tiles bokeh_vs bokeh_tiles_fs
@program bokeh_reach bokeh_vs bokeh_reach_fs
@program bokeh_prefilter bokeh_vs bokeh_prefilter_fs
@program bokeh_sparks bokeh_vs bokeh_sparks_fs
@program bokeh_rest bokeh_vs bokeh_rest_fs
@program bokeh_clean bokeh_vs bokeh_clean_fs
@program bokeh_shrink bokeh_vs bokeh_shrink_fs
@program bokeh_gather bokeh_vs bokeh_gather_fs
@program bokeh_scatter bokeh_scatter_vs bokeh_scatter_fs
