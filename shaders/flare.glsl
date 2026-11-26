// LensFlarePass (postprocessing.c3): the ghosts and halo a lens makes of what is bright. Light
// reflected between a lens's elements lands mirrored through the picture's middle, so each
// pixel of the flare looks along the line through the picture's middle at `ghosts` places of the bright picture (the bloom's glow, over
// `threshold`), each fainter the further from the middle it is, its red, green and blue a
// little apart (the glass's dispersion), and at a ring round the middle (the halo, fainter).
// Regenerate generated/flare.c3 with: just shaders
@header module drie::postprocessing;
@header import drie;

@ctype vec4 Vec4

@vs flare_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@fs flare_fs
layout(binding=0) uniform texture2D flare_src;
layout(binding=0) uniform sampler flare_smp;   // linear, clamped
layout(binding=0) uniform flare_params {
    vec4 size;   // xy = the target's size, z = threshold, w = the source's aspect (width / height)
    vec4 tune;   // x = ghosts, y = their spacing, z = the halo's radius, w = dispersion
};

out vec4 frag_color;

vec3 bright(vec2 uv) {
    vec3 c = textureLod(sampler2D(flare_src, flare_smp), uv, 0.0).rgb;
    c = clamp(c, vec3(0.0), vec3(60000.0));
    c = max(c - vec3(size.z), vec3(0.0));
    // (squeezed: a large bright area, a window filling the view, makes a flare not much
    // brighter than a small one)
    return c / (1.0 + max(c.r, max(c.g, c.b)));
}

vec3 split(vec2 uv, vec2 dir) {
    vec2 d = dir * tune.w;
    return vec3(bright(uv + d).r, bright(uv).g, bright(uv - d).b);
}

// cooler towards the edge of the lens, warmer at its middle
vec3 tint(float r) {
    return mix(vec3(1.0, 0.75, 0.45), vec3(0.45, 0.7, 1.0), clamp(r * 1.6, 0.0, 1.0));
}

void main() {
    vec2 uv = gl_FragCoord.xy / size.xy;
    vec2 tc = vec2(1.0) - uv;
    vec2 step_ = (vec2(0.5) - tc) * tune.y;
    vec2 dir = normalize(vec2(0.5) - tc + vec2(1e-5));
    vec3 col = vec3(0.0);
    int n = int(tune.x);
    for (int i = 0; i < 8; i++) {
        if (i >= n) break;
        vec2 at = fract(tc + step_ * float(i));
        float r = length(vec2(0.5) - at) / 0.7071;
        float w = pow(max(1.0 - r, 0.0), 10.0);
        col += split(at, dir) * w * tint(r);
    }
    vec2 hv = dir * tune.z;
    hv.x /= size.w;
    vec2 at = fract(tc + hv);
    float r = length(vec2(0.5) - at) / 0.7071;
    float w = pow(max(1.0 - r, 0.0), 5.0);
    col += split(at, dir) * w * vec3(0.25, 0.27, 0.32);
    frag_color = vec4(col, 1.0);
}
@end

@program flare flare_vs flare_fs
