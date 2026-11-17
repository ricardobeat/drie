// GTAOPass (postprocessing.c3): ground-truth ambient occlusion (Jimenez et al. 2016, after
// Intel's XeGTAO), from the depth and normals of a scene rendered with MRT, taken off the
// indirect light only. Regenerate generated/gtao.c3 with: just shaders
//
//   gtao_fs / gtao_ms_fs  per pixel, the visibility over a few slices of the hemisphere above
//                         it: r = visibility, g = distance from the camera's plane (m), b = n·v
//   gtao_apply_fs         a 4x4 blur over pixels of about the same distance, then the factor
//                         that darkens the pixel's indirect diffuse light by the visibility and
//                         its indirect specular light by the specular occlusion (Lagarde's);
//                         multiplied into the picture by the pipeline's blending
@header module drie::postprocessing;
@header import drie;

@ctype vec4 Vec4

@vs gtao_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@block gtao_main
layout(binding=1) uniform texture2D gtao_normal;
layout(binding=0) uniform sampler gtao_smp;
layout(binding=0) uniform gtao_params {
    vec4 clip;       // x = near, y = far, z = 1 when depth runs 0..1 (else -1..1), w = 1 when the
                     //   picture's rows run down (all but GL)
    vec4 view;       // xy = tan of half the field of view (x, y), zw = picture size
    vec4 tune;       // x = radius (m), y = intensity (a power), zw = fades out between these
                     //   distances (m; w = 0: never)
    vec4 cam_right;  // the camera's axes in the world (for the normals); cam_right.w = frame
    vec4 cam_up;
    vec4 cam_fwd;
};

out vec4 frag_color;

// the direction a point of the octahedral square stands for (RenderTarget.normal_texture)
vec3 unpack_octahedral_normal(vec2 e) {
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.xy += vec2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}

const int SLICES = 2;
const int STEPS = 6;
const float PI = 3.14159265;
const float HALF_PI = 1.5707963;

float distance_at(ivec2 px) {
    px = clamp(px, ivec2(0), ivec2(view.zw) - 1);
    float d = DEPTH_AT(px);
    if (clip.z < 0.5) d = d * 2.0 - 1.0;
    float n = clip.x;
    float f = clip.y;
    return clip.z > 0.5 ? n * f / (f - d * (f - n)) : 2.0 * n * f / (f + n - d * (f - n));
}

// in front of the camera: x right, y along the picture's rows, z into the picture
vec3 position_at(ivec2 px) {
    float z = distance_at(px);
    vec2 ndc = (vec2(px) + 0.5) / view.zw * 2.0 - 1.0;
    return vec3(ndc * view.xy * z, z);
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec3 p = position_at(px);
    vec3 wn = unpack_octahedral_normal(texelFetch(sampler2D(gtao_normal, gtao_smp), px, 0).xy);
    bool faded = tune.w > 0.0 && p.z > tune.w;
    if (faded || p.z > clip.y * 0.98) {
        frag_color = vec4(1.0, p.z, 1.0, 1.0);
        return;
    }
    vec3 v = normalize(-p);
    vec3 n = vec3(dot(wn, cam_right.xyz), dot(wn, cam_up.xyz) * (clip.w > 0.5 ? -1.0 : 1.0), dot(wn, cam_fwd.xyz));
    // (leaves lit from behind face away)
    if (dot(n, v) < 0.0) n = -n;

    float radius = tune.x;
    // the radius in pixels
    float reach = clamp(radius / p.z * view.w * 0.5 / view.y, float(STEPS), 64.0);
    // past 0.385 of the radius, what is in the way counts for less and less
    float falloff_mul = -1.0 / (0.615 * radius);
    float falloff_add = (1.0 - 0.615) / 0.615 + 1.0;

    // a different turn of the slices for each pixel of a 4x4 block (the blur evens it out),
    // and from frame to frame (for whatever averages over time)
    float frame = cam_right.w;
    float turn = fract((float((px.x & 3) + ((px.y & 3) << 2)) + 0.5) / 16.0 + frame * 0.618034);
    float jitter = fract(52.9829189 * fract(dot(vec2(px) + frame * 5.588238, vec2(0.06711056, 0.00583715))));

    float visibility = 0.0;
    for (int s = 0; s < SLICES; s++) {
        float phi = (float(s) + turn) * PI / float(SLICES);
        vec2 omega = vec2(cos(phi), sin(phi));
        vec3 dir = vec3(omega, 0.0);
        vec3 ortho = dir - dot(dir, v) * v;
        vec3 axis = normalize(cross(ortho, v));
        vec3 proj_n = n - axis * dot(n, axis);
        float proj_len = length(proj_n);
        float cos_n = clamp(dot(proj_n, v) / max(proj_len, 1e-4), 0.0, 1.0);
        float an = sign(dot(ortho, proj_n)) * acos(cos_n);
        // the surface's own plane is as low as the horizon goes on each side
        float low0 = cos(an + HALF_PI);
        float low1 = cos(an - HALF_PI);
        float h0c = low0;
        float h1c = low1;
        for (int j = 0; j < STEPS; j++) {
            float t = (float(j) + jitter) / float(STEPS);
            ivec2 o = ivec2(round(omega * (1.0 + t * t * (reach - 1.0))));
            vec3 d0 = position_at(px + o) - p;
            vec3 d1 = position_at(px - o) - p;
            float l0 = length(d0);
            float l1 = length(d1);
            float c0 = dot(d0, v) / max(l0, 1e-5);
            float c1 = dot(d1, v) / max(l1, 1e-5);
            h0c = max(h0c, mix(low0, c0, clamp(l0 * falloff_mul + falloff_add, 0.0, 1.0)));
            h1c = max(h1c, mix(low1, c1, clamp(l1 * falloff_mul + falloff_add, 0.0, 1.0)));
        }
        float h0 = -acos(clamp(h1c, -1.0, 1.0));
        float h1 = acos(clamp(h0c, -1.0, 1.0));
        h0 = an + clamp(h0 - an, -HALF_PI, HALF_PI);
        h1 = an + clamp(h1 - an, -HALF_PI, HALF_PI);
        float sin_n = sin(an);
        float arc0 = (cos_n + 2.0 * h0 * sin_n - cos(2.0 * h0 - an)) * 0.25;
        float arc1 = (cos_n + 2.0 * h1 * sin_n - cos(2.0 * h1 - an)) * 0.25;
        visibility += proj_len * (arc0 + arc1);
    }
    visibility = clamp(visibility / float(SLICES), 0.03, 1.0);
    visibility = pow(visibility, tune.y);
    if (tune.w > 0.0) visibility = mix(visibility, 1.0, smoothstep(tune.z, tune.w, p.z));
    frag_color = vec4(visibility, p.z, clamp(dot(n, v), 0.0, 1.0), 1.0);
}
@end

@fs gtao_fs
@image_sample_type gtao_depth unfilterable_float
@image_sample_type gtao_normal unfilterable_float
@sampler_type gtao_smp nonfiltering
layout(binding=0) uniform texture2D gtao_depth;
#define DEPTH_AT(px) texelFetch(sampler2D(gtao_depth, gtao_smp), px, 0).r
@include_block gtao_main
@end

// the same for a multisampled scene (its first sample)
@fs gtao_ms_fs
@image_sample_type gtao_depth_ms unfilterable_float
layout(binding=0) uniform texture2DMS gtao_depth_ms;
#define DEPTH_AT(px) texelFetch(sampler2DMS(gtao_depth_ms, gtao_smp), px, 0).r
@include_block gtao_main
@end

@fs gtao_apply_fs
@image_sample_type gtao_ao unfilterable_float
@image_sample_type gtao_indirect unfilterable_float
@sampler_type gtao_apply_smp nonfiltering
layout(binding=0) uniform texture2D gtao_ao;
layout(binding=1) uniform texture2D gtao_indirect;
layout(binding=0) uniform sampler gtao_apply_smp;
layout(binding=0) uniform gtao_apply_params {
    vec4 size;     // xy = picture size, z = the share of the occlusion that darkens direct light
                   //   too (0: none, as it should be), w = 1: no specular occlusion
};

out vec4 frag_color;

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    ivec2 top = ivec2(size.xy) - 1;
    vec4 c = texelFetch(sampler2D(gtao_ao, gtao_apply_smp), px, 0);
    float z = c.g;
    float sum = 0.0;
    float wsum = 0.0;
    for (int y = -2; y < 2; y++) {
        for (int x = -2; x < 2; x++) {
            vec2 s = texelFetch(sampler2D(gtao_ao, gtao_apply_smp), clamp(px + ivec2(x, y), ivec2(0), top), 0).rg;
            float w = max(1.0 - abs(s.g - z) / (0.04 * z + 0.02), 0.0) + 0.001;
            sum += s.r * w;
            wsum += w;
        }
    }
    float ao = sum / wsum;
    vec4 surface = texelFetch(sampler2D(gtao_indirect, gtao_apply_smp), px, 0);
    vec2 share = surface.rg;
    float rough = surface.b;
    // specular occlusion (Lagarde and de Rousiers 2014): a rough surface's reflection is
    // occluded about as its diffuse light is, a smooth one's only at a grazing angle
    float nov = c.b;
    float so = size.w > 0.5 ? 1.0 : clamp(pow(nov + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    float direct = max(1.0 - share.x - share.y, 0.0);
    float m = 1.0 - share.x * (1.0 - ao) - share.y * (1.0 - so) - size.z * direct * (1.0 - ao);
    frag_color = vec4(m, m, m, 1.0);
}
@end

@program gtao gtao_vs gtao_fs
@program gtao_ms gtao_vs gtao_ms_fs
@program gtao_apply gtao_vs gtao_apply_fs
