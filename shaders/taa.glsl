// TAAPass (drie::postprocessing): temporal antialiasing and optional upscaling. Each frame
// jitters the camera (Camera.jitter, Halton sequence) and blends into MRT-reprojected history
// using Catmull-Rom filtering and YCoCg variance clipping (weighting nearby jittered samples
// when upscaling). Regenerate generated/taa.c3 with `just shaders`.
@header module drie::postprocessing;
@header import drie;

@ctype vec4 Vec4

@vs taa_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@fs taa_fs
@image_sample_type taa_velocity unfilterable_float
@sampler_type taa_point_smp nonfiltering
layout(binding=0) uniform texture2D taa_current;
layout(binding=1) uniform texture2D taa_history;
layout(binding=2) uniform texture2D taa_velocity;
layout(binding=0) uniform sampler taa_smp;          // linear, clamped
layout(binding=1) uniform sampler taa_point_smp;    // nearest
layout(binding=0) uniform taa_params {
    vec4 size;   // xy = picture size, z = 1: no history (start over), w = share of the new frame
    vec4 tune;   // x = variance clip width (in standard deviations), y = 1: upscaling
    vec4 source; // xy = the rendered frame's size, zw = the camera's shift in its pixels (along its
                 //   rows and columns: rendered pixel i shows what lies at i + 0.5 - zw)
};

out vec4 frag_color;

vec3 to_ycocg(vec3 c) {
    return vec3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

vec3 from_ycocg(vec3 c) {
    return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// in linear light a few very bright pixels would outweigh everything around them and flicker:
// each counts less the brighter it is (Karis), and back again after
vec3 squash(vec3 c) { return c / (1.0 + max(c.r, max(c.g, c.b))); }
vec3 unsquash(vec3 c) { return c / max(1.0 - max(c.r, max(c.g, c.b)), 1e-4); }

// The history at uv through a Catmull-Rom filter, in 5 bilinear taps (Jimenez's).
vec4 history_at(vec2 uv) {
    vec2 pos = uv * size.xy;
    vec2 center = floor(pos - 0.5) + 0.5;
    vec2 f = pos - center;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 tc12 = (center + w2 / w12) / size.xy;
    vec2 tc0 = (center - 1.0) / size.xy;
    vec2 tc3 = (center + 2.0) / size.xy;
    vec4 sum = textureLod(sampler2D(taa_history, taa_smp), vec2(tc12.x, tc0.y), 0.0) * (w12.x * w0.y)
             + textureLod(sampler2D(taa_history, taa_smp), vec2(tc0.x, tc12.y), 0.0) * (w0.x * w12.y)
             + textureLod(sampler2D(taa_history, taa_smp), tc12, 0.0) * (w12.x * w12.y)
             + textureLod(sampler2D(taa_history, taa_smp), vec2(tc3.x, tc12.y), 0.0) * (w3.x * w12.y)
             + textureLod(sampler2D(taa_history, taa_smp), vec2(tc12.x, tc3.y), 0.0) * (w12.x * w3.y);
    float wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(sum / wsum, vec4(0.0));
}

void main() {
    vec2 uv = gl_FragCoord.xy / size.xy;
    ivec2 top = ivec2(source.xy) - 1;
    bool up = tune.y > 0.5;
    // this pixel's middle in the rendered frame's pixels, and the rendered pixel nearest it
    vec2 q = uv * source.xy;
    ivec2 base = up ? ivec2(floor(q + source.zw)) : ivec2(gl_FragCoord.xy);
    // the neighbourhood's colors (mean and spread), and the longest motion in it: an edge
    // that moves takes its motion along to the pixels just outside it; upscaling, also the
    // pixel's color from the samples, weighted by how near they lie (in the picture's pixels)
    vec3 m1 = vec3(0.0);
    vec3 m2 = vec3(0.0);
    vec2 vel = vec2(0.0);
    vec4 rec = vec4(0.0);
    float wsum = 0.0;
    vec2 scale = size.xy / source.xy;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            ivec2 at = base + ivec2(x, y);
            ivec2 qq = clamp(at, ivec2(0), top);
            vec4 s = texelFetch(sampler2D(taa_current, taa_point_smp), qq, 0);
            vec3 c = to_ycocg(squash(s.rgb));
            m1 += c;
            m2 += c * c;
            vec2 v = texelFetch(sampler2D(taa_velocity, taa_point_smp), qq, 0).xy;
            if (dot(v, v) > dot(vel, vel)) vel = v;
            if (up) {
                vec2 d = (vec2(at) + 0.5 - source.zw - q) * scale;
                float w = exp(-1.5 * dot(d, d));
                rec += vec4(c, s.a) * w;
                wsum += w;
            }
        }
    }
    vec4 cur;
    if (up) {
        rec /= max(wsum, 1e-6);
        cur = vec4(unsquash(clamp(from_ycocg(rec.rgb), 0.0, 0.999)), rec.a);
    } else {
        cur = texelFetch(sampler2D(taa_current, taa_point_smp), base, 0);
    }
    if (size.z > 0.5) {
        frag_color = cur;
        return;
    }
    vec2 was = uv - vel;
    if (any(lessThan(was, vec2(0.0))) || any(greaterThan(was, vec2(1.0)))) {
        frag_color = cur;
        return;
    }
    vec3 mean = m1 / 9.0;
    vec3 sigma = sqrt(max(m2 / 9.0 - mean * mean, 0.0)) * tune.x;
    vec3 hist = to_ycocg(squash(history_at(was).rgb));
    // clipped towards the mean, onto the box the neighbourhood spans
    vec3 d = hist - mean;
    vec3 ratio = abs(d) / max(sigma, vec3(1e-5));
    float k = max(ratio.x, max(ratio.y, ratio.z));
    if (k > 1.0) hist = mean + d / k;
    vec3 c = to_ycocg(squash(cur.rgb));
    // upscaling: a frame whose samples all lie far from this pixel's middle counts for less
    // (but never for nothing: what the history lost has to come back)
    float share = up ? size.w * mix(0.2, 1.0, clamp(wsum, 0.0, 1.0)) : size.w;
    vec3 blended = mix(hist, c, share);
    frag_color = vec4(unsquash(clamp(from_ycocg(blended), 0.0, 0.999)), cur.a);
}
@end

@program taa taa_vs taa_fs
