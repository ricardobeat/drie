// ProgressiveLightMap (lightmap.c3): the "blur plane" drawn under the objects of each update,
// as three's. It covers the whole light map with the average of the last one's 8 neighbours
// half a texel away (filtered: each tap blends four texels), so the texels just outside of
// each object's uv islands take its light and its seams don't show. Regenerate
// generated/lightmap.c3 with: just shaders
@header module drie::shaders;
@header import drie;

@ctype vec4 Vec4

@vs lm_seam_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@fs lm_seam_fs
// the last update's light map (gl_FragCoord / its size is the same texel's uv on every backend)
layout(binding=0) uniform texture2D lm_prev;
layout(binding=0) uniform sampler lm_smp;
layout(binding=0) uniform lm_params {
    vec4 lm_size; // xy = 1 / the map's size, z = the taps' offset (uv)
};

out vec4 frag_color;

void main() {
    vec2 uv = gl_FragCoord.xy * lm_size.xy;
    float o = lm_size.z;
    vec3 sum = texture(sampler2D(lm_prev, lm_smp), uv + vec2(o, 0.0)).rgb
        + texture(sampler2D(lm_prev, lm_smp), uv + vec2(0.0, o)).rgb
        + texture(sampler2D(lm_prev, lm_smp), uv + vec2(0.0, -o)).rgb
        + texture(sampler2D(lm_prev, lm_smp), uv + vec2(-o, 0.0)).rgb
        + texture(sampler2D(lm_prev, lm_smp), uv + vec2(o, o)).rgb
        + texture(sampler2D(lm_prev, lm_smp), uv + vec2(-o, o)).rgb
        + texture(sampler2D(lm_prev, lm_smp), uv + vec2(o, -o)).rgb
        + texture(sampler2D(lm_prev, lm_smp), uv + vec2(-o, -o)).rgb;
    frag_color = vec4(sum / 8.0, 1.0);
}
@end

@program lm_seam lm_seam_vs lm_seam_fs
