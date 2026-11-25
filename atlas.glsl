// The shadow atlas (renderer.c3): copies a part of one depth texture into another, texel for
// texel, as depth: the atlas into a larger one, and a light's maps of its still casters into
// the atlas (see AtlasSlot). Regenerate generated/atlas.c3 with: just shaders
@header module drie::shaders;
@header import drie;

@ctype vec4 Vec4

@vs atlas_copy_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@fs atlas_copy_fs
@image_sample_type atlas_src unfilterable_float
@sampler_type atlas_src_smp nonfiltering
layout(binding=0) uniform texture2D atlas_src;
layout(binding=0) uniform sampler atlas_src_smp;
layout(binding=0) uniform atlas_copy_params {
    vec4 atlas_shift;   // xy: the source's texel minus the target's (as gl_FragCoord counts)
};

void main() {
    ivec2 at = ivec2(gl_FragCoord.xy + atlas_shift.xy);
    gl_FragDepth = texelFetch(sampler2D(atlas_src, atlas_src_smp), at, 0).r;
}
@end

@program atlas_copy atlas_copy_vs atlas_copy_fs
