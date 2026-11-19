// UnrealBloomPass (postprocessing.c3): the glow round what is bright, as a lens spreads a
// little of all the light it lets in. As in Call of Duty: Advanced Warfare (Jimenez 2014) and
// Unreal's bloom: the picture (linear light) is halved again and again, each level with a
// 13-tap filter (the first one also weighs each tap down by its brightness, Karis's average, so
// one very bright pixel doesn't flicker as it moves), then the levels are added back up from
// the smallest, each through a 3x3 tent filter, into the one above it. The largest level, all
// the others added in, is added to the picture.
// Regenerate generated/bloom.c3 with: just shaders
@header module drie::postprocessing;
@header import drie;

@ctype vec4 Vec4

@vs bloom_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@fs bloom_down_fs
layout(binding=0) uniform texture2D bloom_down_src;
layout(binding=0) uniform sampler bloom_down_smp;   // linear, clamped
layout(binding=0) uniform bloom_down_params {
    vec4 size;   // xy = the target's size, zw = 1 / the source's size
    vec4 tune;   // x = 1: the first level (from the picture), y = threshold, z = its soft knee
};

out vec4 frag_color;

vec3 tap(vec2 uv, vec2 o) {
    // (no infinities or NaNs from the picture: they would spread over the whole glow)
    vec3 c = textureLod(sampler2D(bloom_down_src, bloom_down_smp), uv + o * size.zw, 0.0).rgb;
    return clamp(c, vec3(0.0), vec3(60000.0));
}

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// four taps' mean; on the first level, each weighed by 1 / (1 + its brightness)
vec3 box(vec3 a, vec3 b, vec3 c, vec3 d) {
    if (tune.x < 0.5) return (a + b + c + d) * 0.25;
    float wa = 1.0 / (1.0 + luma(a));
    float wb = 1.0 / (1.0 + luma(b));
    float wc = 1.0 / (1.0 + luma(c));
    float wd = 1.0 / (1.0 + luma(d));
    return (a * wa + b * wb + c * wc + d * wd) / (wa + wb + wc + wd);
}

void main() {
    vec2 uv = gl_FragCoord.xy / size.xy;
    vec3 a = tap(uv, vec2(-2.0, -2.0));
    vec3 b = tap(uv, vec2(0.0, -2.0));
    vec3 c = tap(uv, vec2(2.0, -2.0));
    vec3 d = tap(uv, vec2(-1.0, -1.0));
    vec3 e = tap(uv, vec2(1.0, -1.0));
    vec3 f = tap(uv, vec2(-2.0, 0.0));
    vec3 g = tap(uv, vec2(0.0, 0.0));
    vec3 h = tap(uv, vec2(2.0, 0.0));
    vec3 i = tap(uv, vec2(-1.0, 1.0));
    vec3 j = tap(uv, vec2(1.0, 1.0));
    vec3 k = tap(uv, vec2(-2.0, 2.0));
    vec3 l = tap(uv, vec2(0.0, 2.0));
    vec3 m = tap(uv, vec2(2.0, 2.0));
    // the middle box counts half, the four overlapping ones round it an eighth each
    vec3 col = box(d, e, i, j) * 0.5 + (box(a, b, f, g) + box(b, c, g, h) + box(f, g, k, l) + box(g, h, l, m)) * 0.125;
    if (tune.x > 0.5 && tune.y > 0.0) {
        // only what is over the threshold, eased in over the knee below it
        float br = max(col.r, max(col.g, col.b));
        float knee = max(tune.z, 1e-5);
        float soft = clamp(br - tune.y + knee, 0.0, 2.0 * knee);
        soft = soft * soft / (4.0 * knee);
        col *= max(soft, br - tune.y) / max(br, 1e-5);
    }
    frag_color = vec4(col, 1.0);
}
@end

@fs bloom_up_fs
layout(binding=0) uniform texture2D bloom_up_src;
layout(binding=0) uniform sampler bloom_up_smp;   // linear, clamped
layout(binding=0) uniform bloom_up_params {
    vec4 size;   // xy = the target's size, zw = 1 / the source's size
    vec4 tune;   // x = times what is added
};

out vec4 frag_color;

vec3 tap(vec2 uv, vec2 o) {
    return textureLod(sampler2D(bloom_up_src, bloom_up_smp), uv + o * size.zw, 0.0).rgb;
}

void main() {
    vec2 uv = gl_FragCoord.xy / size.xy;
    // a 3x3 tent (1 2 1, 2 4 2, 1 2 1)
    vec3 c = tap(uv, vec2(0.0)) * 4.0;
    c += (tap(uv, vec2(-1.0, 0.0)) + tap(uv, vec2(1.0, 0.0)) + tap(uv, vec2(0.0, -1.0)) + tap(uv, vec2(0.0, 1.0))) * 2.0;
    c += tap(uv, vec2(-1.0, -1.0)) + tap(uv, vec2(1.0, -1.0)) + tap(uv, vec2(-1.0, 1.0)) + tap(uv, vec2(1.0, 1.0));
    // (added with ADDITIVE_BLEND, which keeps the target's alpha)
    frag_color = vec4(c / 16.0 * tune.x, 0.0);
}
@end

@program bloom_down bloom_vs bloom_down_fs
@program bloom_up bloom_vs bloom_up_fs
