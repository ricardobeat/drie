// Shared by the main and the shadow vertex shaders, so both move vertices identically.
@block vertex_common
in vec3 position;
in vec3 normal;
in vec2 uv;
in vec4 color0;
// Rows of the per-instance 3x4 matrix (identity for plain meshes).
in vec4 inst0;
in vec4 inst1;
in vec4 inst2;

// Instances culled on the GPU (renderer.c3, kernels.glsl cull_cs) are not read from the
// attributes above but from the storage buffers below: instance i of the draw is
// rows[3 * visible[first + i]], if i is under counts[job] (the cull's survivors; the draw may
// ask for more, its count comes from an earlier frame).
struct inst_row {
    vec4 row;
};
struct inst_index {
    uint index;
};
layout(binding=16) readonly buffer inst_rows { inst_row rows[]; };
layout(binding=17) readonly buffer inst_visible { inst_index visible[]; };
layout(binding=18) readonly buffer inst_counts { inst_index counts[]; };

vec4 m0;
vec4 m1;
vec4 m2;

// Picks this vertex's instance matrix. `cull`: x = 1 if culled on the GPU, y = its first entry
// in visible, z = its job (in counts). False: an instance past the survivors, not drawn.
bool fetch_instance(vec4 cull) {
    if (cull.x < 0.5) {
        m0 = inst0;
        m1 = inst1;
        m2 = inst2;
        return true;
    }
    uint i = uint(gl_InstanceIndex);
    if (i >= counts[uint(cull.z)].index) return false;
    uint s = visible[uint(cull.y) + i].index * 3;
    m0 = rows[s].row;
    m1 = rows[s + 1].row;
    m2 = rows[s + 2].row;
    return true;
}

vec3 instance_point(vec3 p) {
    vec4 l = vec4(p, 1.0);
    return vec3(dot(m0, l), dot(m1, l), dot(m2, l));
}

vec3 instance_dir(vec3 d) {
    return vec3(dot(m0.xyz, d), dot(m1.xyz, d), dot(m2.xyz, d));
}

// Wind: bends vertices sideways the higher they sit above the model's origin.
// sway = (time, amplitude per meter of height, speed, unused)
vec3 apply_sway(vec3 wp, float height, vec4 sway) {
    if (sway.y <= 0.0) return wp;
    float h = max(height, 0.0);
    float phase = wp.x * 0.37 + wp.z * 0.29;
    float t = sway.x * sway.z;
    float gust = 0.6 + 0.4 * sin(t * 0.31 + phase * 0.21);
    vec2 d = vec2(sin(t + phase) + 0.4 * sin(t * 2.7 + phase * 3.1),
                  cos(t * 0.83 + phase * 1.3) + 0.4 * sin(t * 2.1 + phase * 2.3));
    wp.xz += d * gust * sway.y * h * sqrt(h);
    return wp;
}

// What an application's shader hooks can read (see shader.glsl): Material.custom and
// Renderer.custom, set at the top of main() from the stage's uniforms.
vec4 material_custom[2];
vec4 frame_custom[2];

@include_block vertex_hooks
@end

@vs vs
layout(binding=0) uniform vs_params {
    // world -> clip space, measured from view_origin: gl_Position = view_proj * (world - origin).
    // Far from the world's origin (a farm 6 km out) a plain view_proj * world loses the depth:
    // clip z and w come out as the small difference of two large sums, rounded apart, and
    // surfaces centimetres apart swap places from frame to frame
    mat4 view_proj;
    vec4 view_origin;   // xyz: the camera's position
    mat4 model;
    mat4 normal_mat;
    vec4 uv_transform;  // xy = repeat, zw = offset
    vec4 sway;          // see apply_sway
    vec4 inst_cull;     // see fetch_instance
    vec4 vs_material_custom[2];
    vec4 vs_frame_custom[2];
    // motion vectors, for a render with MRT: where the vertex was a frame ago
    mat4 prev_view_proj;
    mat4 prev_model;
    vec4 motion;        // x = 0: as if nothing moved, 1: only the camera moved, 2: the mesh may have
                        //   moved or sways too; y = the last frame's sway time; zw = the camera's
                        //   jitter (in clip space, per unit of w), left out of the motion
};

@include_block vertex_common

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_color;
out float v_alpha;
out vec4 v_clip;
out vec4 v_prev_clip;
// lines: the distance along the line, for dashes (a line's vertex normal x holds it; lines
// have no normals)
out float v_line_distance;

void main() {
    v_line_distance = normal.x;
    if (!fetch_instance(inst_cull)) {
        // outside the clip volume: nothing drawn
        v_world_pos = vec3(0.0);
        v_normal = vec3(0.0, 1.0, 0.0);
        v_uv = vec2(0.0);
        v_color = vec3(0.0);
        v_alpha = 0.0;
        v_clip = vec4(2.0, 2.0, 2.0, 1.0);
        v_prev_clip = v_clip;
        gl_Position = v_clip;
        return;
    }
    vec3 ip = instance_point(position);
    vec4 wp = model * vec4(ip, 1.0);
    wp.xyz = apply_sway(wp.xyz, ip.y, sway);
    material_custom[0] = vs_material_custom[0];
    material_custom[1] = vs_material_custom[1];
    frame_custom[0] = vs_frame_custom[0];
    frame_custom[1] = vs_frame_custom[1];
    vec3 swayed = wp.xyz;
    vertex_hook(wp.xyz, ip, model);
    v_world_pos = wp.xyz;
    v_normal = (normal_mat * vec4(instance_dir(normal), 0.0)).xyz;
    v_uv = uv * uv_transform.xy + uv_transform.zw;
    v_color = color0.rgb;
    v_alpha = color0.a;
    gl_Position = view_proj * vec4(wp.xyz - view_origin.xyz, 1.0);
    v_clip = gl_Position;
    v_clip.xy -= motion.zw * v_clip.w;
    v_prev_clip = v_clip;
    if (motion.x > 1.5) {
        // the same vertex, placed and swayed as it was a frame ago (instances are taken to
        // stand still, and the hook to have moved it as far as it does now)
        vec4 pp = prev_model * vec4(ip, 1.0);
        pp.xyz = apply_sway(pp.xyz, ip.y, vec4(motion.y, sway.yzw)) + (wp.xyz - swayed);
        v_prev_clip = prev_view_proj * pp;
    } else if (motion.x > 0.5) {
        v_prev_clip = prev_view_proj * wp;
    }
}
@end

@vs shadow_vs
layout(binding=0) uniform shadow_vs_params {
    mat4 s_view_proj;   // world - s_origin -> the light's clip space (see vs_params)
    vec4 s_origin;      // xyz: the light's position (0 for the sun's orthographic maps)
    mat4 s_model;
    vec4 s_uv_transform;
    vec4 s_sway;
    vec4 s_inst_cull;
    vec4 s_material_custom[2];
    vec4 s_frame_custom[2];
};

@include_block vertex_common

out vec2 v_uv;

void main() {
    if (!fetch_instance(s_inst_cull)) {
        v_uv = vec2(0.0);
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    vec3 ip = instance_point(position);
    vec4 wp = s_model * vec4(ip, 1.0);
    wp.xyz = apply_sway(wp.xyz, ip.y, s_sway);
    material_custom[0] = s_material_custom[0];
    material_custom[1] = s_material_custom[1];
    frame_custom[0] = s_frame_custom[0];
    frame_custom[1] = s_frame_custom[1];
    vertex_hook(wp.xyz, ip, s_model);
    v_uv = uv * s_uv_transform.xy + s_uv_transform.zw;
    // keep the unused attributes alive so both programs share one vertex layout
    v_uv += (normal.xy + color0.xy) * 0.0;
    gl_Position = s_view_proj * vec4(wp.xyz - s_origin.xyz, 1.0);
}
@end

@fs shadow_fs
layout(binding=0) uniform texture2D s_map;
layout(binding=0) uniform sampler s_smp;
layout(binding=1) uniform shadow_fs_params {
    vec4 s_params;  // x = alpha test threshold (0 = off), y = material opacity
};

in vec2 v_uv;

void main() {
    if (s_params.x > 0.0) {
        float a = texture(sampler2D(s_map, s_smp), v_uv).a * s_params.y;
        if (a < s_params.x) discard;
    }
}
@end

@block fs_main
layout(binding=0) uniform texture2D tex_map;
layout(binding=1) uniform texture2D tex_normal;
layout(binding=2) uniform texture2D tex_arm;
layout(binding=3) uniform texture2D shadow_map;
layout(binding=4) uniform texture2D tex_env;
layout(binding=5) uniform texture2D shadow_atlas;
layout(binding=6) uniform texture3D tex_gi;
layout(binding=7) uniform texture2D ssr_color;   // the last frame: tone mapped, sRGB, a = 0 at light sources
layout(binding=8) uniform texture2D ssr_depth;   // g = its distance from the camera's plane (m)
layout(binding=9) uniform texture2D sun_shadow;  // the directional light's map: 2x2 cascades, or one box
layout(binding=21) uniform texture2D reflector_map; // a Reflector's mirrored render, linear
layout(binding=20) uniform texture2D ssr_back;    // r = the nearest back face's distance from the camera's plane (m), 0: none
layout(binding=23) uniform texture2DArray tex_probes; // the scene's reflection probes (reflection_probes.c3)
layout(binding=24) uniform texture2D tex_parallax; // Material.parallax_map: the surface's height, 1 = top
layout(binding=19) uniform texture2D fog_volume; // volumetric fog: the froxel grid (fog.c3, fog_cs in kernels.glsl)
layout(binding=0) uniform sampler tex_smp;
layout(binding=1) uniform sampler shadow_smp;
layout(binding=2) uniform sampler env_smp;
layout(binding=3) uniform sampler gi_smp;
layout(binding=4) uniform sampler ssr_smp;

layout(binding=1) uniform fs_params {
    vec4 color;        // linear rgb, a = opacity
    vec4 emissive;     // linear rgb * intensity
    vec4 specular;     // phong specular color, w = flags (1: vertex colors, 2: foliage, 4: receive shadow, 8: glass,
                       //   16: vertex colors are baked light (emissive.w = strength), 32: screen-space map)
    vec4 mat_params;   // x = mode, y = roughness, z = metalness, w = shininess
    vec4 tex_params;   // x = has map, y = normal map scale (0 = none), z = has arm map, w = alpha test
    vec4 env_params;   // x = has environment map, y = intensity, z = 1: the scene's environment
                       // lights the diffuse part too (env_sh), w = meters that alpha 1 of the
                       // map stands for (0: it holds no distances)
    vec4 env_pos;      // where the map was captured, w = 1 for box projection
    vec4 env_box_min;  // the room the map shows, for box projection
    vec4 env_box_max;
    vec4 env_sh[9];    // the scene's environment's irradiance, as Texture.env_sh (rgb)
    vec4 screen_params; // xy = 1 / render target size (screen-space map), z = LOD cross-fade (0..1),
                        //   w = 1 fading in, -1 fading out, 0 none
    vec4 cull_min;      // w = 1: fragments inside cull_min..cull_max are not drawn
    vec4 cull_max;
    vec4 camera_pos;   // w = tone mapping exposure
    vec4 output_params; // x = 1: linear light out (a half-float target: no tone mapping, no sRGB),
                        //   y = 1: the reflection source (ssr_color) is linear light too,
                        //   z = tone mapping (TONE_*), w = the window's colors (DISPLAY_*) when
                        //   drawn straight into it
    vec4 ambient;      // summed ambient light
    vec4 hemi_sky;
    vec4 hemi_ground;
    vec4 gi_min;       // irradiance volume: its box, w = 1 when there is one, 2 when it replaces the ambient light
    vec4 gi_inv_size;  // w = intensity
    vec4 gi_dims;      // probes along x, y, z, w = lookup offset along the normal (m)
    vec4 fog_color;    // w = how strongly lights glow in the fog
    vec4 fog_params;   // x = near, y = far, z = density, w = mode (0 none, 1 linear, 2 exp2, 3 exp)
    vec4 fog_box_min;  // fog-free box, w = 1 when enabled
    vec4 fog_box_max;  // w = the part of the density left in the box
    vec4 fog_height;   // x = base height, y = falloff per meter (0 = none), w = density away from the sun's side
    vec4 fog_sun_dir;  // towards the sun, w = sharpness (0 = no sun)
    vec4 fog_sun_color; // w = light shafts (0..1)
    vec4 fog_grid;     // volumetric fog: x, y = cells across and up, z = slices, w = slices per row of
                       //   tiles in fog_volume (0: no volumetric fog, the fog is worked out here)
    vec4 fog_slices;   // x = where its slices start (m of view depth), y = slices per unit of
                       //   log(depth / x)
    vec4 counts;       // x = directional lights, y = point lights, z = spot lights (all of them, in
                       //   dir_lights, point_lights and spot_lights)
    mat4 cluster_view_proj; // world -> the camera's clip space, for the fragment's cluster
    vec4 cluster_grid;      // xyz = clusters across, up and in depth
    vec4 cluster_depth;     // x = view depth where the first slice starts, y = slices per unit of
                            //   log(depth), z = where the last ends
    mat4 dir_shadow_mat[4]; // world -> uv in its tile + depth, per cascade of the directional light's shadow
    vec4 dir_shadow_params; // x = which directional light + 1 (0: none), y = normal bias (m),
                            //   z = texel size of sun_shadow, w = blur radius in texels
    vec4 dir_cascade_rect[4];  // each cascade's tile in sun_shadow: xy = offset, zw = size
    vec4 dir_cascade_info[4];  // per cascade: x = view depth (m) where it ends, y = size of a texel (m)
    vec4 dir_cascade_params;   // x = cascades, y = blend band (part of a cascade), z = 1: tint them,
                               //   w = 1: one box (no cascades), faded at its edge
    vec4 dir_cascade_fwd;      // the camera's forward direction, w = shadow intensity (0..1)
    mat4 ssr_view_proj;    // world -> the last frame's clip space
    vec4 ssr_params;       // x = this material's screen-space reflections (0: none), y = reach (m),
                           //   z = noise seed, w = 1: the last frame's row 0 is its top
    vec4 ssr_thickness;    // x = 1: ssr_back has the last frame's back faces, y = this material's reach (m)
    mat4 reflector_view_proj;  // world -> the Reflector's picture's clip space (reflector.c3)
    vec4 reflector_params;     // x = strength (0: no reflector), y = distortion (m), z = blur
    vec4 probe_info;           // x = the scene's reflection probes (0: none; env_params.x = 2: this material uses them)
    vec4 probe_pos[4];         // where each was captured, w = meters its map's alpha 1 stands for
    vec4 probe_min[4];         // its box (min == max: everywhere else), w = blend width (m)
    vec4 probe_max[4];         // w = intensity
    vec4 rc_min;               // radiance cascades (radiance.c3): the box's corner, w = probe spacing (m)
    vec4 rc_size;              // xyz = probes, w = intensity (0: none)
    vec4 material_custom[2];   // for the application's hooks: Material.custom
    vec4 frame_custom[2];      //   and Renderer.custom
    vec4 parallax_params;      // Material.parallax_map: x = depth (m; 0: none), y = distance (m) where
                               //   it has faded out, z = self-shadowing toward the sun (0..1)
    vec4 line_dash;            // dashed lines: x = dash length, y = dash + gap, z = scale, w = 1: dashed
};

// Lights live in storage buffers, as many as there are. Point and spot lights are sorted into
// a grid of clusters (froxels: tiles of the screen, cut into slices of depth that grow
// exponentially), and a fragment only looks at the lights of its own cluster.
struct gpu_dir_light {
    vec4 dir;    // world-space direction towards the light
    vec4 color;
};
struct gpu_point_light {
    vec4 pos;    // w = distance cutoff
    vec4 color;  // w = decay
    vec4 params; // x = source radius
};
struct gpu_spot_light {
    vec4 pos;    // w = distance cutoff
    vec4 dir;    // direction the light points to, w = cos(angle)
    vec4 color;  // w = decay
    vec4 params; // x = cos(inner angle) (from penumbra), y = source radius, z = specular radius,
                 //   w = how much the light glows in the fog
    vec4 shadow; // x = its shadow in spot_shadows (-1: none), y = shadow intensity (0..1)
};
struct gpu_spot_shadow {
    mat4 mat;    // world - the light's position -> shadow map uv + depth
    vec4 params; // x = 1: in shadow_map, 2: in shadow_atlas, y = normal bias (m), z = texel size,
                 //   w = blur radius in texels
    vec4 rect;   // the light's tile in shadow_atlas: xy = offset, zw = size
    vec4 soft;   // x = size of the source (m; 0: edges equally sharp everywhere), y = near,
                 //   z = far, w = width of the map 1 m from the light (negative: depth is -1..1)
};
struct gpu_light_cluster {
    uvec4 range; // x = its first entry in light_indices, y = point lights, z = spot lights after them
};
struct gpu_light_index {
    uint index;  // into point_lights or spot_lights
};
layout(binding=10) readonly buffer dir_lights { gpu_dir_light dirs[]; };
layout(binding=11) readonly buffer point_lights { gpu_point_light points[]; };
layout(binding=12) readonly buffer spot_lights { gpu_spot_light spots[]; };
layout(binding=13) readonly buffer spot_shadows { gpu_spot_shadow spot_shadow_data[]; };
layout(binding=14) readonly buffer light_clusters { gpu_light_cluster clusters[]; };
layout(binding=15) readonly buffer light_indices { gpu_light_index cluster_lights[]; };
// RadianceCascades (radiance.c3): per probe of cascade 0, the light arriving on surfaces
// facing +x -x +y -y +z -z (w = 0: the probe is inside something)
struct rc_probe {
    vec4 e;
};
layout(binding=22) readonly buffer rc_probes { rc_probe rc_probe_data[]; };

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in vec3 v_color;
in float v_alpha;
in vec4 v_clip;
in vec4 v_prev_clip;
in float v_line_distance;

layout(location=0) out vec4 frag_color;
#ifdef MRT
// Renderer.render into a target with MRT (RenderTarget.mrt): the surface's normal in world
// space (octahedral, in rg), and how far it moved across the picture since the last frame.
layout(location=1) out vec4 frag_normal;
layout(location=2) out vec4 frag_velocity;
// the shares of the light that are indirect: r = diffuse, g = specular (what AO darkens);
// b = roughness, a = metalness
layout(location=3) out vec4 frag_indirect;

// A direction as a point of the square -1..1 (the octahedron around it unfolded; Cigolle et
// al. 2014, three's packNormalToOctahedral): two numbers that keep their precision all around.
vec2 pack_normal_octahedral(vec3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return n.xy;
}
#endif

const float PI = 3.14159265359;
const float RECIPROCAL_PI = 0.31830988618;

@include_block tonemapping

float distance_attenuation(float d, float cutoff, float decay) {
    float a = 1.0 / max(pow(d, decay), 0.01);
    if (cutoff > 0.0) {
        float r = d / cutoff;
        float f = clamp(1.0 - r * r * r * r, 0.0, 1.0);
        a *= f * f;
    }
    return a;
}

vec3 f_schlick(vec3 f0, float voh) {
    float f = pow(1.0 - voh, 5.0);
    return f0 + (vec3(1.0) - f0) * f;
}

float d_ggx(float noh, float a) {
    float a2 = a * a;
    float d = noh * noh * (a2 - 1.0) + 1.0;
    return RECIPROCAL_PI * a2 / (d * d);
}

float v_smith(float nov, float nol, float a) {
    float a2 = a * a;
    float gv = nol * sqrt(a2 + (1.0 - a2) * nov * nov);
    float gl = nov * sqrt(a2 + (1.0 - a2) * nol * nol);
    return 0.5 / max(gv + gl, 1e-6);
}

int flags() { return int(specular.w + 0.5); }

vec4 g_base;      // rgb = albedo, a = alpha
float g_rough;
float g_metal;
float g_radius;   // source radius of the light being evaluated
vec3 g_dpdx;      // how the world position changes from one pixel to the next (taken where
vec3 g_dpdy;      // the flow is uniform, for the shadows' tap counts)
// Parallax occlusion mapping (Material.parallax_map). The material's maps are read at g_uv,
// with the derivatives of v_uv (the offset jumps where the height map steps: its own
// derivatives would pick a blurry mip level there).
vec2 g_uv;
vec2 g_duvdx;
vec2 g_duvdy;
// how the uv changes per meter, along the surface (the cotangent frame, as in perturb_normal)
vec3 g_parallax_gu;
vec3 g_parallax_gv;
float g_parallax_h;     // the height map where g_uv lands (0..1)
float g_parallax_depth; // the depth after the distance fade (m)

// Direct light contribution, following three.js's (r155+) physically based light units.
// Returns diffuse in rgb; adds specular to `spec`.
vec3 direct(int mode, vec3 n, vec3 v, vec3 l, float dist, vec3 radiance, vec3 diffuse_color, inout vec3 spec) {
    bool foliage = (flags() & 2) != 0;
    float ndl = dot(n, l);
    // thin leaves and blades: light wraps around and shines through
    float nol = foliage ? 0.35 + 0.65 * abs(ndl) : clamp(ndl, 0.0, 1.0);
    vec3 irradiance = nol * radiance;
    vec3 result = irradiance * diffuse_color * RECIPROCAL_PI;
    if (mode == 1) return result;

    vec3 h = normalize(l + v);
    float noh = clamp(dot(n, h), 0.0, 1.0);
    float voh = clamp(dot(v, h), 0.0, 1.0);
    if (mode == 2) {
        float shininess = mat_params.w;
        float d = RECIPROCAL_PI * (shininess * 0.5 + 1.0) * pow(noh, shininess);
        spec += irradiance * f_schlick(specular.rgb, voh) * 0.25 * d;
    } else {
        float rough = max(g_rough, 0.0525);
        float a = rough * rough;
        // a light with a size gives a wider, dimmer highlight (Karis' sphere light normalization)
        float a_wide = clamp(a + g_radius / (2.0 * max(dist, 1e-3)), 0.0, 1.0);
        float norm = a / a_wide;
        vec3 f0 = mix(vec3(0.04), g_base.rgb, g_metal);
        float nov = clamp(abs(dot(n, v)), 1e-4, 1.0);
        spec += irradiance * f_schlick(f0, voh) * v_smith(nov, nol, a_wide) * d_ggx(noh, a_wide) * norm * norm;
    }
    return result;
}

// Length of the part of the segment a-b that lies in the fog (outside the fog-free box).
float fog_distance(vec3 a, vec3 b) {
    vec3 d = b - a;
    float len = length(d);
    if (fog_box_min.w < 0.5) return len;
    vec3 inv = 1.0 / (d + vec3(equal(d, vec3(0.0))) * 1e-6);
    vec3 t0 = (fog_box_min.xyz - a) * inv;
    vec3 t1 = (fog_box_max.xyz - a) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float enter = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    float exit = min(min(tmax.x, tmax.y), min(tmax.z, 1.0));
    return len * (1.0 - (1.0 - fog_box_max.w) * max(exit - enter, 0.0));
}

// Fraction of light that makes it through `d` meters of fog.
float fog_transmittance(float d) {
    int fog_mode = int(fog_params.w);
    if (fog_mode == 0) return 1.0;
    if (fog_mode == 1) return 1.0 - smoothstep(fog_params.x, fog_params.y, d);
    float dd = fog_params.z * d;
    return fog_mode == 2 ? exp(-dd * dd) : exp(-dd);
}

// Light scattered towards the camera by the fog along the view ray, for a light at `lp`:
// the integral of cone * transmittance / distance^2 along the ray. Substituting s = b + h tan(a)
// turns the 1 / distance^2 kernel into a constant, so a few samples in `a` are enough.
// `cone` is (spot direction, cos outer), `cos_inner` < -1 for a point light.
float fog_glow(vec3 lp, float radius, vec3 ray, float depth, vec4 cone, float cos_inner) {
    vec3 to_light = lp - camera_pos.xyz;
    float b = dot(to_light, ray);
    float h = max(sqrt(max(dot(to_light, to_light) - b * b, 0.0)), max(radius, 0.05));
    float a0 = atan(-b / h);
    float a1 = atan((depth - b) / h);
    float sum = 0.0;
    for (int i = 0; i < 6; i++) {
        float a = mix(a0, a1, (float(i) + 0.5) / 6.0);
        float s = b + h * tan(a);
        vec3 from_light = ray * s - to_light;
        float d = max(length(from_light), 1e-4);
        float w = fog_transmittance(s + d);
        if (cos_inner >= -1.0) w *= smoothstep(cone.w, cos_inner, dot(from_light / d, cone.xyz));
        sum += w;
    }
    return sum / 6.0 * (a1 - a0) / h;
}

// Volumetric fog in front of world position `p`: rgb = the light it scatters towards the
// camera, a = how much of p gets through. Read from the froxel grid between the far ends of
// the two slices around p (the first's near end is the camera: nothing yet).
vec4 fog_volume_at(vec3 p) {
    vec4 clip = cluster_view_proj * vec4(p, 1.0);
    vec2 uv = clamp(clip.xy / max(abs(clip.w), 1e-6) * 0.5 + 0.5, vec2(0.0), vec2(1.0));
    float depth = dot(p - camera_pos.xyz, dir_cascade_fwd.xyz);
    float slices = fog_grid.z;
    float s = depth <= fog_slices.x ? -1.0 : clamp(log(depth / fog_slices.x) * fog_slices.y - 1.0, -1.0, slices - 1.0);
    float k = floor(s);
    float f = s - k;
    vec2 size = vec2(textureSize(sampler2D(fog_volume, ssr_smp), 0));
    vec2 cell = clamp(uv * fog_grid.xy, vec2(0.5), fog_grid.xy - 0.5);
    float per_row = fog_grid.w;
    vec4 a = vec4(0.0, 0.0, 0.0, 1.0);
    if (k >= 0.0) {
        vec2 at = vec2(mod(k, per_row), floor(k / per_row)) * fog_grid.xy + cell;
        a = textureLod(sampler2D(fog_volume, ssr_smp), at / size, 0.0);
    }
    float k1 = min(k + 1.0, slices - 1.0);
    vec2 at1 = vec2(mod(k1, per_row), floor(k1 / per_row)) * fog_grid.xy + cell;
    vec4 b = textureLod(sampler2D(fog_volume, ssr_smp), at1 / size, 0.0);
    return mix(a, b, f);
}

// Distance from the light's plane of a depth in its shadow map, and back.
float shadow_distance(float z, vec4 soft) {
    if (soft.w < 0.0) z = 2.0 * z - 1.0;
    return soft.w < 0.0 ? 2.0 * soft.y * soft.z / (soft.z + soft.y - z * (soft.z - soft.y))
                        : soft.y * soft.z / (soft.z - z * (soft.z - soft.y));
}

float shadow_depth(float d, vec4 soft) {
    return soft.w < 0.0 ? 0.5 + 0.5 * ((soft.z + soft.y) * d - 2.0 * soft.y * soft.z) / (d * (soft.z - soft.y))
                        : soft.z * (d - soft.y) / (d * (soft.z - soft.y));
}

// The cluster that holds world position `p`: its tile of the screen, and its slice of view
// depth (slices grow exponentially from cluster_depth.x to .z). Matches cluster_lights_cpu in
// renderer.c3.
int cluster_of(vec3 p) {
    ivec3 grid = ivec3(cluster_grid.xyz);
    vec4 clip = cluster_view_proj * vec4(p, 1.0);
    vec2 ndc = clip.xy / max(abs(clip.w), 1e-6);
    ivec2 tile = clamp(ivec2(floor((ndc * 0.5 + 0.5) * vec2(grid.xy))), ivec2(0), grid.xy - 1);
    float depth = dot(p - camera_pos.xyz, dir_cascade_fwd.xyz);
    int slice = depth <= cluster_depth.x ? 0 : int(floor(log(depth / cluster_depth.x) * cluster_depth.y));
    slice = clamp(slice, 0, grid.z - 1);
    return tile.x + grid.x * (tile.y + grid.y * slice);
}

// 1 where the light's map has nothing nearer than depth `z` at `uv`.
float shadow_tap(bool tiled, vec4 rect, float texel, vec2 uv, float z) {
    if (tiled) {
        // stay half a texel inside the tile: the neighbours belong to other lights
        uv = rect.xy + clamp(uv, vec2(texel), vec2(1.0 - texel)) * rect.zw;
        return texture(sampler2DShadow(shadow_atlas, shadow_smp), vec3(uv, z));
    }
    return texture(sampler2DShadow(shadow_map, shadow_smp), vec3(uv, z));
}

// Shadow of spot light `s`, whose map is `si` in spot_shadows. Lights whose map is redrawn
// every frame have shadow_map to themselves; the ones drawn once share shadow_atlas, a tile each.
float spot_shadow_of(int s, int si, vec3 n) {
    vec4 params = spot_shadow_data[si].params;
    vec3 to_light = spots[s].pos.xyz - v_world_pos;
    float dist = length(to_light);
    // push the lookup off the surface, more at grazing angles and far away where texels are big
    float slope = 1.0 - clamp(dot(n, to_light / dist), 0.0, 1.0);
    vec3 p = v_world_pos + n * params.y * dist * (0.4 + slope);
    // (the matrix takes the point from the light, see vs_params)
    mat4 m = spot_shadow_data[si].mat;
    p -= spots[s].pos.xyz;
    vec4 sp = m * vec4(p, 1.0);
    if (sp.w <= 0.0) return 1.0;
    vec3 c = sp.xyz / sp.w;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0 || c.z > 1.0) return 1.0;
    // the map's footprint of a pixel of the screen: blurs much smaller than that need few taps,
    // and the search for what casts the shadow none
    vec4 spx = m * vec4(p + g_dpdx, 1.0);
    vec4 spy = m * vec4(p + g_dpdy, 1.0);
    vec3 dx = spx.w > 0.0 ? spx.xyz / spx.w - c : vec3(0.0);
    vec3 dy = spy.w > 0.0 ? spy.xyz / spy.w - c : vec3(0.0);
    float pixel = max(max(length(dx.xy), length(dy.xy)), 1e-6);
    // rotated disk of taps for a soft edge
    float angle = fract(sin(dot(gl_FragCoord.xy + ssr_params.z * vec2(5.0, 3.0), vec2(12.9898, 78.233))) * 43758.5453) * 6.2831;
    float ca = cos(angle);
    float sa = sin(angle);
    float r = params.z * params.w;
    bool tiled = params.x > 1.5;
    vec4 rect = spot_shadow_data[si].rect;
    vec4 soft = spot_shadow_data[si].soft;
    // how the depth changes over the map along this surface, so that a wide blur does not
    // run the surface into its own shadow
    vec2 grad = vec2(0.0);
    int most = 8;
    if (soft.x > 0.0) {
        // A source with a size: the edge of a shadow is as wide as the source looks from there
        // past the thing that casts it. Sharp where the two touch, wide far behind. How far the
        // caster is comes from asking the map at a few depths, nearer each time.
        float dr = shadow_distance(c.z, soft);
        float per_m = 1.0 / (abs(soft.w) * dr);
        float search = clamp(soft.x * 1.6 * per_m, r, 0.04);
        most = 14;
        // the widest edge the source could make here is under a pixel: no need to look
        if (search > pixel) {
            float found = 0.0;
            float nearest = 0.0;
            for (int k = 0; k < 7; k++) {
                float a = float(k) * 1.0472;
                vec2 o = k == 0 ? vec2(0.0) : vec2(cos(a) * ca - sin(a) * sa, cos(a) * sa + sin(a) * ca) * search;
                if (shadow_tap(tiled, rect, params.z, c.xy + o, c.z) > 0.5) continue;
                float lo = soft.y;
                float hi = dr;
                for (int j = 0; j < 5; j++) {
                    float mid = 0.5 * (lo + hi);
                    if (shadow_tap(tiled, rect, params.z, c.xy + o, shadow_depth(mid, soft)) > 0.5) lo = mid; else hi = mid;
                }
                nearest += 0.5 * (lo + hi);
                found += 1.0;
            }
            if (found < 0.5) return 1.0;
            float db = nearest / found;
            r = clamp(soft.x * (dr - db) / db * per_m, r, 0.04);
        }
        float det = dx.x * dy.y - dx.y * dy.x;
        if (abs(det) > 1e-12) grad = vec2(dy.y * dx.z - dx.y * dy.z, dx.x * dy.z - dy.x * dx.z) / det;
    }
    // about three taps per pixel across the blur, at least four (each tap is a bilinear 2x2)
    int taps = clamp(int(ceil(r / pixel * 3.0)), 4, most);
    float sum = 0.0;
    for (int k = 0; k < 14; k++) {
        if (k >= taps) break;
        float a = float(k) * 2.39996;
        float rr = sqrt((float(k) + 0.5) / float(taps)) * r;
        vec2 o = vec2(cos(a), sin(a)) * rr;
        o = vec2(o.x * ca - o.y * sa, o.x * sa + o.y * ca);
        sum += shadow_tap(tiled, rect, params.z, c.xy + o, c.z + clamp(dot(grad, o), -0.004, 0.004));
    }
    return sum / float(taps);
}

float shadow(int s, vec3 n) {
    int si = int(spots[s].shadow.x);
    if (si < 0 || (flags() & 4) == 0) return 1.0;
    return mix(1.0, spot_shadow_of(s, si, n), spots[s].shadow.y);
}

// 1 where cascade `i` of the sun's map has nothing nearer than depth `z` at `uv` (0..1 in its tile).
float sun_tap(int i, vec2 uv, float z) {
    vec4 rect = dir_cascade_rect[i];
    // stay half a texel inside the tile: the neighbours belong to other cascades
    vec2 t = dir_shadow_params.z / rect.zw;
    uv = rect.xy + clamp(uv, t * 0.5, 1.0 - t * 0.5) * rect.zw;
    return texture(sampler2DShadow(sun_shadow, shadow_smp), vec3(uv, z));
}

// The cascade that covers view depth `depth`, -1 past the last.
int sun_cascade(float depth) {
    if (dir_cascade_params.w > 0.5) return 0;
    int count = int(dir_cascade_params.x);
    for (int i = 0; i < 4; i++) {
        if (i >= count) break;
        if (depth < dir_cascade_info[i].x) return i;
    }
    return -1;
}

// Shadow of the directional light in cascade `i`: a box of parallel rays towards `l`.
float dir_shadow_in(int i, vec3 n, vec3 l) {
    vec4 params = dir_shadow_params;
    // texels of the far cascades are bigger: push further off the surface, blur fewer texels
    float scale = dir_cascade_info[i].y / max(dir_cascade_info[0].y, 1e-6);
    float slope = 1.0 - clamp(dot(n, l), 0.0, 1.0);
    vec3 c = (dir_shadow_mat[i] * vec4(v_world_pos + n * params.y * scale * (0.4 + slope), 1.0)).xyz;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0 || c.z > 1.0 || c.z < 0.0) return 1.0;
    float angle = fract(sin(dot(gl_FragCoord.xy + ssr_params.z * vec2(5.0, 3.0), vec2(12.9898, 78.233))) * 43758.5453) * 6.2831;
    float ca = cos(angle);
    float sa = sin(angle);
    float texel = params.z / dir_cascade_rect[i].z;
    float r = texel * max(params.w / scale, min(params.w, 1.0));
    float sum = 0.0;
    for (int k = 0; k < 10; k++) {
        float a = float(k) * 2.39996;
        vec2 o = vec2(cos(a), sin(a)) * (sqrt((float(k) + 0.5) / 10.0) * r);
        sum += sun_tap(i, c.xy + vec2(o.x * ca - o.y * sa, o.x * sa + o.y * ca), c.z);
    }
    if (dir_cascade_params.w < 0.5) return sum / 10.0;
    // one box: nothing is shadowed outside of it, fade towards its edge
    vec2 edge = min(c.xy, 1.0 - c.xy);
    return mix(1.0, sum / 10.0, clamp(min(edge.x, edge.y) * 10.0, 0.0, 1.0));
}

// Shadow of the directional light that casts one, picking the cascade by view depth and
// fading into the next one near its end.
float dir_shadow_full(vec3 n, vec3 l) {
    float depth = dot(v_world_pos - camera_pos.xyz, dir_cascade_fwd.xyz);
    int i = sun_cascade(depth);
    if (i < 0) return 1.0;
    float lit = dir_shadow_in(i, n, l);
    if (dir_cascade_params.w > 0.5) return lit;
    float start = i == 0 ? 0.0 : dir_cascade_info[i - 1].x;
    float band = (dir_cascade_info[i].x - start) * dir_cascade_params.y;
    if (band <= 0.0) return lit;
    float t = clamp((depth - (dir_cascade_info[i].x - band)) / band, 0.0, 1.0);
    if (t <= 0.0) return lit;
    // the last one fades out to no shadow
    float next = i + 1 < int(dir_cascade_params.x) ? dir_shadow_in(i + 1, n, l) : 1.0;
    return mix(lit, next, t);
}

float dir_shadow(vec3 n, vec3 l) {
    if ((flags() & 4) == 0) return 1.0;
    return mix(1.0, dir_shadow_full(n, l), dir_cascade_fwd.w);
}

// How much of the fog between the camera and here the directional light reaches, 0..1:
// marched through its shadow map, each step weighted by how much of the fog's light it gives
// (`sigma`: the fog's density along this ray). Beyond the map's reach the fog counts as lit.
float sun_in_fog(vec3 from, vec3 to, float sigma) {
    vec3 ray = to - from;
    float len = length(ray);
    const float REACH = 70.0;
    const int STEPS = 14;
    float span = min(len, REACH);
    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))) + ssr_params.z * 0.618);
    float lit = 0.0;
    float last = 1.0;
    for (int i = 0; i < STEPS; i++) {
        float s1 = span * (float(i) + 1.0) / float(STEPS);
        float s = span * (float(i) + jitter) / float(STEPS);
        float next = exp(-sigma * s1);
        vec3 at = from + ray * (s / max(len, 1e-4));
        int k = sun_cascade(dot(at - camera_pos.xyz, dir_cascade_fwd.xyz));
        vec3 c = (dir_shadow_mat[max(k, 0)] * vec4(at, 1.0)).xyz;
        float seen = 1.0;
        if (k >= 0 && c.x > 0.0 && c.x < 1.0 && c.y > 0.0 && c.y < 1.0 && c.z < 1.0 && c.z > 0.0) {
            seen = sun_tap(k, c.xy, c.z);
            vec2 edge = min(c.xy, 1.0 - c.xy);
            seen = mix(1.0, seen, clamp(min(edge.x, edge.y) * 10.0, 0.0, 1.0));
        }
        lit += (last - next) * seen;
        last = next;
    }
    float all_fog = 1.0 - exp(-sigma * len);
    lit += last - (1.0 - all_fog);
    return all_fog > 1e-5 ? mix(1.0, clamp(lit / all_fog, 0.0, 1.0), dir_cascade_fwd.w) : 1.0;
}

// Light from the scene's irradiance volume, for a surface facing `n`; `fade` is 1 inside the
// volume, 0 outside it, between the two over its outermost half cell. Each probe holds the
// light arriving from the six axis directions, side by side along x in the texture.
vec3 gi_irradiance(vec3 n, out float fade) {
    fade = 0.0;
    if (gi_min.w < 0.5) return vec3(0.0);
    vec3 p = (v_world_pos + n * gi_dims.w - gi_min.xyz) * gi_inv_size.xyz;
    vec3 edge = min(p, 1.0 - p) * gi_dims.xyz;
    fade = clamp(min(edge.x, min(edge.y, edge.z)) * 2.0, 0.0, 1.0);
    if (fade <= 0.0) return vec3(0.0);
    // stay inside a direction's slab
    float x = clamp(p.x, 0.5 / gi_dims.x, 1.0 - 0.5 / gi_dims.x) / 6.0;
    vec3 slab = vec3(n.x >= 0.0 ? 0.0 : 1.0, n.y >= 0.0 ? 2.0 : 3.0, n.z >= 0.0 ? 4.0 : 5.0) / 6.0;
    vec3 w = n * n;
    vec3 e = textureLod(sampler3D(tex_gi, gi_smp), vec3(x + slab.x, p.y, p.z), 0.0).rgb * w.x
           + textureLod(sampler3D(tex_gi, gi_smp), vec3(x + slab.y, p.y, p.z), 0.0).rgb * w.y
           + textureLod(sampler3D(tex_gi, gi_smp), vec3(x + slab.z, p.y, p.z), 0.0).rgb * w.z;
    return e * gi_inv_size.w;
}

// Light from the scene's radiance cascades for a surface facing `n`; `fade` as gi_irradiance's.
// Blended between the 8 probes around, leaving out those inside things and those behind the
// surface's own plane: a wall takes none of the light of the room behind it.
vec3 rc_irradiance(vec3 n, out float fade) {
    fade = 0.0;
    if (rc_size.w <= 0.0) return vec3(0.0);
    float spacing = rc_min.w;
    vec3 g = (v_world_pos + n * (0.25 * spacing) - rc_min.xyz) / spacing;
    vec3 edge = min(g, rc_size.xyz - g);
    fade = clamp(min(edge.x, min(edge.y, edge.z)) * 2.0, 0.0, 1.0);
    if (fade <= 0.0) return vec3(0.0);
    g -= 0.5;
    ivec3 b = ivec3(floor(g));
    vec3 f = g - vec3(b);
    ivec3 dims = ivec3(rc_size.xyz);
    ivec3 side = ivec3(n.x >= 0.0 ? 0 : 1, n.y >= 0.0 ? 2 : 3, n.z >= 0.0 ? 4 : 5);
    vec3 nn = n * n;
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    vec3 sum_any = vec3(0.0);
    float wsum_any = 0.0;
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        ivec3 c = clamp(b + o, ivec3(0), dims - 1);
        int at = ((c.z * dims.y + c.y) * dims.x + c.x) * 6;
        vec4 ex = rc_probe_data[at + side.x].e;
        vec3 e = ex.rgb * nn.x + rc_probe_data[at + side.y].e.rgb * nn.y + rc_probe_data[at + side.z].e.rgb * nn.z;
        vec3 w3 = mix(1.0 - f, f, vec3(o));
        float w = (w3.x * w3.y * w3.z + 0.002) * ex.w;
        vec3 probe_pos = rc_min.xyz + (vec3(c) + 0.5) * spacing;
        float front = clamp((dot(probe_pos - v_world_pos, n) + 0.04) / 0.06, 0.0, 1.0);
        sum += e * (w * front);
        wsum += w * front;
        sum_any += e * w;
        wsum_any += w;
    }
    if (wsum > 1e-4) return sum / wsum * rc_size.w;
    return wsum_any > 1e-5 ? sum_any / wsum_any * rc_size.w : vec3(0.0);
}

float ssr_noise(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// Where a world-space point was in the last frame: xy = uv, z = distance from the camera's plane.
vec3 ssr_project(vec3 p) {
    vec4 clip = ssr_view_proj * vec4(p, 1.0);
    vec2 ndc = clip.xy / max(clip.w, 1e-4);
    if (ssr_params.w > 0.5) ndc.y = -ndc.y;
    return vec3(ndc * 0.5 + 0.5, clip.w);
}

float ssr_back_at(vec2 uv) {
    ivec2 size = textureSize(sampler2D(ssr_back, ssr_smp), 0);
    return texelFetch(sampler2D(ssr_back, ssr_smp), clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1), 0).r;
}

float ssr_depth_at(vec2 uv) {
    ivec2 size = textureSize(sampler2D(ssr_depth, ssr_smp), 0);
    return texelFetch(sampler2D(ssr_depth, ssr_smp), clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1), 0).g;
}

// Screen-space reflection: follows the mirrored view ray through the last frame's depth and
// returns what it runs into there (linear, as the camera saw it); `found` is 0 when it leaves
// the picture or hits nothing. A rough surface sends its one ray somewhere in a wider cone,
// which shows as noise until something downstream (a temporal filter, a tape) evens it out.
// A Reflector's picture (reflector.c3) where the ray from here along the normal's reflection
// would be seen: exact for a flat surface, and for a bumpy one as if what it mirrors were
// reflector_params.y away. Blurred by roughness; `found` fades out at the picture's edges.
vec3 planar_reflection(vec3 n, vec3 v, float rough, out float found) {
    vec3 r = reflect(-v, n);
    vec4 clip = reflector_view_proj * vec4(v_world_pos + r * reflector_params.y, 1.0);
    vec2 ndc = clip.xy / max(clip.w, 1e-4);
    if (ssr_params.w > 0.5) ndc.y = -ndc.y;
    vec2 uv = ndc * 0.5 + 0.5;
    vec2 edge = min(uv, 1.0 - uv);
    found = clip.w > 0.0 ? smoothstep(-0.02, 0.03, min(edge.x, edge.y)) : 0.0;
    vec2 size = vec2(textureSize(sampler2D(reflector_map, ssr_smp), 0));
    float spread = rough * rough * reflector_params.z;
    vec2 radius = vec2(spread * size.y / size.x, spread);
    float turn = ssr_noise(gl_FragCoord.xy + ssr_params.z * vec2(47.0, 17.0)) * 2.0 * PI;
    vec3 sum = vec3(0.0);
    for (int k = 0; k < 8; k++) {
        float a = turn + float(k) * 2.39996;
        vec2 at = uv + vec2(cos(a), sin(a)) * radius * sqrt((float(k) + 0.5) / 8.0);
        sum += min(textureLod(sampler2D(reflector_map, ssr_smp), at, 0.0).rgb, vec3(64.0));
    }
    return sum / 8.0;
}

// how many steps a reflected ray takes at most, and how far one goes on the screen at most (in
// pixels of the reflection source)
const int SSR_STEPS = 40;
const float SSR_MAX_STEP_PX = 6.0;

// how much of a rough surface's lobe the ray itself is sent through (the inner part, by
// chance: the frames add up to it); the rest is blurred from what the ray finds. All of it by
// chance makes a small bright thing into a patch of sparks.
const float SSR_LOBE_SHARE = 0.45;
const float SSR_BLUR = 0.28;

// A microfacet normal of a GGX surface as the eye sees them (the distribution of visible
// normals, Heitz 2018), in the surface's own frame (z = its normal); ve = towards the eye.
vec3 ggx_visible_normal(vec3 ve, float alpha, vec2 u) {
    vec3 vh = normalize(vec3(alpha * ve.xy, ve.z));
    float lensq = dot(vh.xy, vh.xy);
    vec3 t1 = lensq > 1e-8 ? vec3(-vh.y, vh.x, 0.0) * inversesqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 t2 = cross(vh, t1);
    float rad = sqrt(u.x);
    float phi = 2.0 * PI * u.y;
    float a = rad * cos(phi);
    float b = rad * sin(phi);
    float half_z = 0.5 * (1.0 + vh.z);
    b = (1.0 - half_z) * sqrt(max(1.0 - a * a, 0.0)) + half_z * b;
    vec3 nh = a * t1 + b * t2 + sqrt(max(1.0 - a * a - b * b, 0.0)) * vh;
    return normalize(vec3(alpha * nh.xy, max(nh.z, 1e-4)));
}

vec3 screen_reflection(vec3 n, vec3 geo_n, vec3 v, float rough, out float found) {
    found = 0.0;
    vec2 seed = gl_FragCoord.xy + ssr_params.z * vec2(47.0, 17.0);
    float n1 = ssr_noise(seed);
    float n2 = ssr_noise(seed + vec2(31.7, 11.3));
    float turn = n1 * 2.0 * PI;
    // the ray mirrors off one of the surface's microfacets, picked as the eye meets them: a
    // rough floor draws a light out into a streak towards the eye, as a real one does
    float alpha = rough * rough;
    vec3 tx = normalize(cross(abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0), n));
    vec3 ty = cross(n, tx);
    vec3 ve = normalize(vec3(dot(v, tx), dot(v, ty), max(dot(v, n), 0.05)));
    vec3 h = ggx_visible_normal(ve, alpha, vec2(n2 * SSR_LOBE_SHARE, n1));
    vec3 r = reflect(-v, normalize(tx * h.x + ty * h.y + n * h.z));
    if (dot(r, geo_n) < 0.02) return vec3(0.0);

    vec3 origin = v_world_pos + geo_n * 0.02;
    vec2 screen_px = vec2(textureSize(sampler2D(ssr_depth, ssr_smp), 0));
    float step_len = 0.06;
    float t = step_len * (0.4 + n2);
    float t_prev = 0.0;
    for (int i = 0; i < SSR_STEPS; i++) {
        vec3 s = ssr_project(origin + r * t);
        if (s.z < 0.05 || s.x < 0.0 || s.y < 0.0 || s.x > 1.0 || s.y > 1.0) break;
        // behind what the camera saw there: where the ray went in lies between the step before
        // and this one (steps get longer than thin things are thick, so this is found first)
        if (s.z - ssr_depth_at(s.xy) > 0.02) {
            float lo = t_prev;
            float hi = t;
            for (int k = 0; k < 6; k++) {
                float mid = 0.5 * (lo + hi);
                vec3 m = ssr_project(origin + r * mid);
                if (m.z - ssr_depth_at(m.xy) > 0.02) hi = mid; else lo = mid;
            }
            // and there it has to be inside the thing, not past it: as thick as its back faces
            // say, or else about as thick as a thing is, fading out past that
            vec3 p_in = ssr_project(origin + r * hi);
            float front = ssr_depth_at(p_in.xy);
            float back = ssr_thickness.x > 0.5 ? ssr_back_at(p_in.xy) : 0.0;
            float hit;
            if (back > front + 0.005 && back < front + 2.0) {
                hit = 1.0 - smoothstep(0.0, 0.03, p_in.z - back);
            } else {
                hit = 1.0 - smoothstep(0.15, 0.3, p_in.z - front);
            }
            if (hit > 0.0) {
            s = ssr_project(origin + r * hi);
            vec2 edge = min(s.xy, 1.0 - s.xy);
            found = hit * smoothstep(0.0, 0.07, min(edge.x, edge.y)) * (1.0 - smoothstep(0.6, 1.0, hi / ssr_thickness.y));
            // what a rough surface mirrors is spread over a disc that grows with the way the
            // ray went: a few taps over it, turned differently at every pixel
            vec2 size = vec2(textureSize(sampler2D(ssr_color, ssr_smp), 0));
            float spread = alpha * SSR_BLUR * hi / max(s.z, 0.1);
            vec2 radius = vec2(spread * size.y / size.x, spread);
            vec3 sum = vec3(0.0);
            for (int k = 0; k < 8; k++) {
                float a = turn + float(k) * 2.39996;
                vec2 at = s.xy + vec2(cos(a), sin(a)) * radius * sqrt((float(k) + 0.5) / 8.0);
                vec4 c = textureLod(sampler2D(ssr_color, ssr_smp), at, 0.0);
                if (output_params.y > 0.5) {
                    // as it was lit; kept within bounds so a few bright pixels don't feed
                    // themselves frame after frame
                    sum += min(c.rgb, vec3(64.0));
                    continue;
                }
                // back from what the tone mapping made of it, roughly; light sources were far over white
                vec3 lin = srgb_to_linear(c.rgb);
                lin /= max(1.0 - 0.7 * max(lin.r, max(lin.g, lin.b)), 0.3);
                sum += lin * (1.0 + 5.0 * (1.0 - c.a)) / max(camera_pos.w, 0.05);
            }
            return sum / 8.0;
            }
        }
        // longer steps further out, but never more than a few pixels on the screen: a step
        // longer than something is thick would go through it without seeing it
        t_prev = t;
        step_len *= 1.2;
        vec3 next = ssr_project(origin + r * (t + step_len));
        float px = length((next.xy - s.xy) * screen_px);
        if (next.z > 0.05 && px > SSR_MAX_STEP_PX) step_len *= SSR_MAX_STEP_PX / px;
        t += step_len;
        if (t > ssr_thickness.y) break;
    }
    return vec3(0.0);
}

// Tangent-space normal mapping without vertex tangents (cotangent frame from screen derivatives).
vec3 perturb_normal(vec3 n, float scale) {
    vec3 t = textureGrad(sampler2D(tex_normal, tex_smp), g_uv, g_duvdx, g_duvdy).xyz * 2.0 - 1.0;  // (parallax: g_uv)
    t.xy *= scale;
    vec3 dp1 = dFdx(v_world_pos);
    vec3 dp2 = dFdy(v_world_pos);
    vec2 duv1 = g_duvdx;
    vec2 duv2 = g_duvdy;
    vec3 dp2perp = cross(dp2, n);
    vec3 dp1perp = cross(n, dp1);
    vec3 tangent = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 bitangent = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1e-12));
    return normalize(tangent * invmax * t.x + bitangent * invmax * t.y + n * t.z);
}

// The parallax map at uv.
vec4 parallax_height(vec2 uv) { return textureGrad(sampler2D(tex_parallax, tex_smp), uv, g_duvdx, g_duvdy); }

// Marches the view ray down into the height field (1 = the surface, 0 = depth meters under it)
// and moves g_uv to where it meets it.
void parallax_offset(vec3 n, vec3 v, float dist) {
    g_parallax_depth = parallax_params.x * (1.0 - smoothstep(parallax_params.y * 0.6, parallax_params.y, dist));
    g_parallax_h = 1.0;
    if (g_parallax_depth <= 0.0) return;
    vec3 dp1 = dFdx(v_world_pos);
    vec3 dp2 = dFdy(v_world_pos);
    vec3 dp2perp = cross(dp2, n);
    vec3 dp1perp = cross(n, dp1);
    float det = dot(dp1, dp2perp);
    if (abs(det) < 1e-14) return;
    g_parallax_gu = (dp2perp * g_duvdx.x + dp1perp * g_duvdy.x) / det;
    g_parallax_gv = (dp2perp * g_duvdx.y + dp1perp * g_duvdy.y) / det;
    // the uv a step of 1 m down the view ray moves by (flattened at a glancing angle, where
    // the march would smear the texture out)
    float nv = max(dot(n, v), 0.2);
    vec2 per_m = -vec2(dot(g_parallax_gu, v), dot(g_parallax_gv, v)) / nv;
    vec2 total = per_m * g_parallax_depth;
    float steps = mix(24.0, 8.0, nv);
    float dh = 1.0 / steps;
    float ray = 1.0;
    float h = parallax_height(g_uv).r;
    float prev_h = h;
    vec2 uv = g_uv;
    for (int i = 0; i < 24; i++) {
        if (h >= ray || float(i) >= steps) break;
        prev_h = h;
        ray -= dh;
        uv += total * dh;
        h = parallax_height(uv).r;
    }
    // between the last step above the surface and the first below it
    float after = h - ray;
    float before = prev_h - (ray + dh);
    float t = after / min(after - before, -1e-5);
    t = clamp(t, 0.0, 1.0);
    g_uv = uv - total * dh * t;
    g_parallax_h = mix(h, prev_h, t);
}

// How much of the light from l reaches g_uv past the bumps of the height map around it (soft:
// the more a bump stands above the ray to the light, the darker).
float parallax_shadow(vec3 n, vec3 l) {
    if (g_parallax_depth <= 0.0 || parallax_params.z <= 0.0) return 1.0;
    float nl = dot(n, l);
    if (nl <= 0.0) return 1.0;
    vec2 per_m = vec2(dot(g_parallax_gu, l), dot(g_parallax_gv, l)) / max(nl, 0.1);
    float rise = 1.0 - g_parallax_h;
    if (rise <= 0.01) return 1.0;
    vec2 total = per_m * g_parallax_depth * rise;
    float block = 0.0;
    for (int i = 1; i <= 8; i++) {
        float f = float(i) / 8.0;
        float above = parallax_height(g_uv + total * f).r - (g_parallax_h + rise * f);
        block = max(block, above * (1.0 - f) * 8.0);
    }
    return 1.0 - parallax_params.z * clamp(block, 0.0, 1.0);
}

// Equirectangular environment map lookup, blurred with roughness through its mip levels.
// With a box set, the reflected ray from the surface point is traced to the room's walls and
// the map is read towards that hit, so nearby walls keep their size and place.
vec2 env_uv(vec3 d) {
    return vec2(atan(d.z, d.x) * (0.5 * RECIPROCAL_PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) * RECIPROCAL_PI);
}

// The light the scene's environment puts on a surface facing n: env_sh's 9 spherical
// harmonics, in the order of texture.c3's sh_basis.
vec3 env_irradiance(vec3 n) {
    vec3 e = env_sh[0].rgb + env_sh[1].rgb * n.y + env_sh[2].rgb * n.z + env_sh[3].rgb * n.x
           + env_sh[4].rgb * (n.x * n.y) + env_sh[5].rgb * (n.y * n.z) + env_sh[6].rgb * (3.0 * n.z * n.z - 1.0)
           + env_sh[7].rgb * (n.x * n.z) + env_sh[8].rgb * (n.x * n.x - n.y * n.y);
    return max(e, vec3(0.0));
}

// The map being read: the material's own (g_env_layer < 0), or a layer of the scene's probes.
float g_env_layer;
vec4 g_env_pos;      // as env_pos
vec3 g_env_min;
vec3 g_env_max;
float g_env_range;   // as env_params.w

vec4 env_texel(vec2 uv, float lod) {
    if (g_env_layer < 0.0) return textureLod(sampler2D(tex_env, env_smp), uv, lod);
    return textureLod(sampler2DArray(tex_probes, env_smp), vec3(uv, g_env_layer), lod);
}

// How far the point `q` (relative to where the map was captured) lies behind what the map saw
// in its direction.
float env_behind(vec3 q) {
    float len = length(q);
    return len - env_texel(env_uv(q / len), 0.0).a * g_env_range;
}

vec3 env_sample(vec3 d, float rough) {
    if (g_env_pos.w > 0.5) {
        vec3 inv = 1.0 / d;
        vec3 t_far = max((g_env_max - v_world_pos) * inv, (g_env_min - v_world_pos) * inv);
        float t = max(min(min(t_far.x, t_far.y), t_far.z), 0.0);
        vec3 o = v_world_pos - g_env_pos.xyz;
        if (g_env_range > 0.0 && rough < 0.3) {
            // A map with distances: walk the ray through the room as the map saw it, and stop
            // where it goes behind something (not far behind: then it only passes at the back
            // of it, out of the map's sight).
            const int STEPS = 72;
            float dt = t / float(STEPS);
            // the distances come in steps of 1/255 of their range: nearer than that to a
            // surface, a ray is not yet behind it (or a wall seen at a glancing angle breaks
            // up into rings, one per step)
            float slack = g_env_range / 255.0 * 1.5;
            float prev = 0.0;
            float at = dt * (0.3 + 0.7 * ssr_noise(gl_FragCoord.xy + ssr_params.z * vec2(47.0, 17.0)));
            for (int i = 0; i < STEPS; i++) {
                float behind = env_behind(o + d * at);
                if (behind > slack && behind < dt + 0.12 + slack) {
                    float lo = prev;
                    float hi = at;
                    for (int k = 0; k < 4; k++) {
                        float mid = 0.5 * (lo + hi);
                        if (env_behind(o + d * mid) > slack) hi = mid; else lo = mid;
                    }
                    t = hi;
                    break;
                }
                prev = at;
                at += dt;
            }
        }
        d = normalize(o + d * t);
    }
    return env_texel(env_uv(d), rough * 7.0).rgb * env_params.y;
}

@include_block surface_hooks

// What a surface facing `facing` mirrors of its surroundings in direction d: from the material's own
// map, or from the scene's reflection probes: the one whose box the point is in (or just
// outside of, facing in: a window pane in the room's wall), cross-faded with its neighbour
// over the blend width outside the box, and with the probe for everywhere else beyond that.
vec3 env_reflection(vec3 d, float rough, vec3 facing) {
    g_env_layer = -1.0;
    g_env_pos = env_pos;
    g_env_min = env_box_min.xyz;
    g_env_max = env_box_max.xyz;
    g_env_range = env_params.w;
    if (env_params.x < 1.5) return env_sample(d, rough);
    int n = int(probe_info.x);
    int first = -1;
    int second = -1;
    int outer = -1;
    float w1 = 0.0;
    float w2 = 0.0;
    for (int i = 0; i < n; i++) {
        if (probe_min[i].xyz == probe_max[i].xyz) {
            outer = i;
            continue;
        }
        vec3 lo = v_world_pos - probe_min[i].xyz;
        vec3 hi = probe_max[i].xyz - v_world_pos;
        float inside = min(min(lo.x, min(lo.y, lo.z)), min(hi.x, min(hi.y, hi.z)));
        float blend = max(probe_min[i].w, 1e-3);
        float w = clamp(1.0 + inside / blend, 0.0, 1.0);
        // the outer side of the room's own wall shows what is outside
        if (dot(facing, probe_pos[i].xyz - v_world_pos) <= 0.0) w = 0.0;
        if (w > w1) {
            second = first;
            w2 = w1;
            first = i;
            w1 = w;
        } else if (w > w2) {
            second = i;
            w2 = w;
        }
    }
    float sum = w1 + w2;
    float rest = outer >= 0 ? max(1.0 - sum, 0.0) : 0.0;
    // with a probe for everywhere else the boxes' shares stay as they are (it takes the
    // rest); without one they are made to add up to all of it
    float norm = 1.0 / max(outer >= 0 ? max(sum, 1.0) : sum, 1e-4);
    vec3 result = vec3(0.0);
    for (int k = 0; k < 3; k++) {
        int i = k == 0 ? first : (k == 1 ? second : outer);
        float w = k == 0 ? w1 * norm : (k == 1 ? w2 * norm : rest);
        if (i < 0 || w <= 0.0) continue;
        g_env_layer = float(i);
        g_env_pos = vec4(probe_pos[i].xyz, k == 2 ? 0.0 : 1.0);
        g_env_min = probe_min[i].xyz;
        g_env_max = probe_max[i].xyz;
        g_env_range = probe_pos[i].w;
        result += env_sample(d, rough) * (w * probe_max[i].w);
    }
    return result;
}

void main() {
    int mode = int(mat_params.x);
    int fl = flags();
    bool foliage = (fl & 2) != 0;
#ifdef MRT
    {
        // in the picture's texture coordinates, row 0 where the target's first row is
        vec2 now = v_clip.xy / v_clip.w;
        vec2 was = v_prev_clip.xy / max(v_prev_clip.w, 1e-5);
        vec2 d = (now - was) * 0.5;
        if (ssr_params.w > 0.5) d.y = -d.y;
        frag_velocity = vec4(d, 0.0, 1.0);
        frag_indirect = vec4(0.0, 0.0, mat_params.y, mat_params.z);
        frag_normal = vec4(pack_normal_octahedral(normalize(v_normal)), 0.0, 1.0);
    }
#endif

#ifndef NO_DISCARD
    if (cull_min.w > 0.5 && all(greaterThan(v_world_pos, cull_min.xyz)) && all(lessThan(v_world_pos, cull_max.xyz))) discard;
    // a dashed line's gaps (as three's LineDashedMaterial)
    if (line_dash.w > 0.5 && mod(v_line_distance * line_dash.z, line_dash.y) > line_dash.x) discard;
    // LOD cross-fade: a screen-door dither, the incoming level where it is below the fade and
    // the outgoing one where it is not (interleaved gradient noise: no visible pattern)
    if (screen_params.w != 0.0) {
        float dither = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        if (screen_params.w > 0.0 ? dither >= screen_params.z : dither < screen_params.z) discard;
    }
#endif
    if ((fl & 32) != 0) {
        // a window onto another render: its pixels as they are, already tone mapped
        frag_color = vec4(texture(sampler2D(tex_map, tex_smp), gl_FragCoord.xy * screen_params.xy).rgb, 1.0);
        return;
    }

    g_base = vec4(color.rgb, color.a);
    bool baked = (fl & 16) != 0;
    if ((fl & 1) != 0 && !baked) g_base.rgb *= v_color;
    vec3 map_rgb = vec3(1.0);
    // parallax_map: where the view ray meets the height map, for all of the material's maps
    g_uv = v_uv;
    g_duvdx = dFdx(v_uv);
    g_duvdy = dFdy(v_uv);
    g_parallax_depth = 0.0;
    if (parallax_params.x > 0.0 && mode != 0) {
        vec3 pn = normalize(v_normal);
        if (!gl_FrontFacing && !foliage) pn = -pn;
        vec3 pv = camera_pos.xyz - v_world_pos;
        float pd = length(pv);
        parallax_offset(pn, pv / max(pd, 1e-5), pd);
    }
    if (tex_params.x > 0.5) {
        vec4 texel = textureGrad(sampler2D(tex_map, tex_smp), g_uv, g_duvdx, g_duvdy);  // (parallax: g_uv)
        g_base *= texel;
        map_rgb = texel.rgb;
    }
    float alpha = g_base.a;
    if ((fl & 1) != 0) alpha *= v_alpha;
#ifdef NO_DISCARD
    // cut out by the depth prepass already (depth_fs): whatever is drawn here is covered
    if (tex_params.w < 0.0) alpha = 1.0;
#else
    if (tex_params.w > 0.0) {
        // far away the texture's smaller levels average leaves and gaps into an alpha under
        // the cutoff, and the whole plant would go: make up for it, so as much stays covered
        if (tex_params.x > 0.5) {
            vec2 texel = v_uv * vec2(textureSize(sampler2D(tex_map, tex_smp), 0));
            vec2 tx = dFdx(texel);
            vec2 ty = dFdy(texel);
            float level = 0.5 * log2(max(max(dot(tx, tx), dot(ty, ty)), 1.0));
            alpha *= 1.0 + 0.3 * min(level, 6.0);
        }
        // sharpen the cutout edge to about a pixel so alpha-to-coverage can antialias it
        alpha = (alpha - tex_params.w) / max(fwidth(alpha), 1e-4) + 0.5;
        if (alpha <= 0.0) discard;
        alpha = min(alpha, 1.0);
    }
#endif

    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing && !foliage) n = -n;
    vec3 geo_n = n;
    g_dpdx = dFdx(v_world_pos);
    g_dpdy = dFdy(v_world_pos);
    if (tex_params.y != 0.0 && mode != 0) n = perturb_normal(n, tex_params.y);
    if (mode != 0) normal_hook(geo_n, n);
#ifdef MRT
    frag_normal.xy = pack_normal_octahedral(n);
#endif

    g_rough = mat_params.y;
    g_metal = mat_params.z;
    float ao = 1.0;
    if (tex_params.z > 0.5) {
        vec3 arm = textureGrad(sampler2D(tex_arm, tex_smp), g_uv, g_duvdx, g_duvdy).rgb;  // (parallax: g_uv)
        ao = arm.r;
        g_rough *= arm.g;
        g_metal *= arm.b;
    }

    if (mode != 0) surface_hook(geo_n, g_base, g_rough, g_metal);
#ifdef MRT
    frag_indirect.ba = vec2(g_rough, g_metal);
#endif

    vec3 outc;
    vec3 spec = vec3(0.0);
    // the indirect light in outc, diffuse and specular (for frag_indirect)
    vec3 ind_d = vec3(0.0);
    vec3 ind_s = vec3(0.0);
    vec3 to_cam = camera_pos.xyz - v_world_pos;
    float depth = length(to_cam);
    vec3 v = to_cam / max(depth, 1e-5);

    if (mode == 0) {
        outc = g_base.rgb;
    } else if (mode == 4) {
        // MeshNormalMaterial: output normal as color (bypasses color management, like three)
        frag_color = vec4(n * 0.5 + 0.5, color.a);
        return;
    } else {
        vec3 diffuse_color = g_base.rgb;
        if (mode == 3) diffuse_color *= (1.0 - g_metal);

        vec3 sky_light = ambient.rgb + mix(hemi_ground.rgb, hemi_sky.rgb, 0.5 * n.y + 0.5);
        // the scene's environment (Scene.environment): the light from all around n
        if (env_params.z > 0.0) sky_light += env_params.y * env_irradiance(n);
        float gi_fade;
        vec3 gi = gi_irradiance(n, gi_fade);
        vec3 irradiance = gi_min.w > 1.5 ? mix(sky_light, gi, gi_fade) : sky_light + gi * gi_fade;
        if (rc_size.w > 0.0) {
            // radiance cascades: all the indirect light inside their box
            float rc_fade;
            vec3 rc = rc_irradiance(n, rc_fade);
            irradiance = mix(irradiance, rc, rc_fade);
        }
        outc = irradiance * diffuse_color * RECIPROCAL_PI * ao;
        ind_d = outc;

        int ndir = int(counts.x);
        for (int i = 0; i < ndir; i++) {
            g_radius = 0.0;
            vec3 l = normalize(dirs[i].dir.xyz);
            float lit = int(dir_shadow_params.x) == i + 1 ? dir_shadow(n, l) : 1.0;
            // parallax_map: the bumps' own shadows, toward the first directional light
            if (i == 0) lit *= parallax_shadow(geo_n, l);
            if (lit > 0.0) outc += direct(mode, n, v, l, 1e6, dirs[i].color.rgb * lit, diffuse_color, spec);
        }
        if (dir_cascade_params.z > 0.5 && dir_shadow_params.x > 0.5) {
            // renderer.debug_cascades: red, green, blue, yellow
            int k = sun_cascade(dot(v_world_pos - camera_pos.xyz, dir_cascade_fwd.xyz));
            if (k >= 0) outc = mix(outc, k == 0 ? vec3(1.0, 0.0, 0.0) : (k == 1 ? vec3(0.0, 1.0, 0.0) : (k == 2 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 1.0, 0.0))), 0.5);
        }
        // the point and spot lights that reach this fragment's cluster
        uvec4 range = clusters[cluster_of(v_world_pos)].range;
        int first = int(range.x);
        int npoint = int(range.y);
        int nspot = int(range.z);
        for (int k = 0; k < npoint; k++) {
            gpu_point_light pl = points[cluster_lights[first + k].index];
            vec3 lv = pl.pos.xyz - v_world_pos;
            float d = length(lv);
            // a light source with a size: falloff softens up close instead of blowing out
            g_radius = pl.params.x;
            float d_soft = sqrt(d * d + g_radius * g_radius);
            vec3 radiance = pl.color.rgb * distance_attenuation(d_soft, pl.pos.w, pl.color.w);
            radiance *= fog_transmittance(fog_distance(pl.pos.xyz, v_world_pos));
            outc += direct(mode, n, v, lv / d, d, radiance, diffuse_color, spec);
        }
        for (int k = 0; k < nspot; k++) {
            int i = int(cluster_lights[first + npoint + k].index);
            vec3 lv = spots[i].pos.xyz - v_world_pos;
            float d = length(lv);
            // out of the light's reach
            if (spots[i].pos.w > 0.0 && d > spots[i].pos.w) continue;
            vec3 l = lv / d;
            float cone = smoothstep(spots[i].dir.w, spots[i].params.x, dot(-l, spots[i].dir.xyz));
            if (cone <= 0.0) continue;
            g_radius = spots[i].params.y;
            float d_soft = sqrt(d * d + g_radius * g_radius);
            g_radius = spots[i].params.z;
            vec3 radiance = spots[i].color.rgb * cone * distance_attenuation(d_soft, spots[i].pos.w, spots[i].color.w);
            radiance *= fog_transmittance(fog_distance(spots[i].pos.xyz, v_world_pos));
            radiance *= shadow(i, geo_n);
            outc += direct(mode, n, v, l, d, radiance, diffuse_color, spec);
        }
        spec *= ao;
        // Material.map_emissive: the color map glows too
        outc += (fl & 64) != 0 ? emissive.rgb * map_rgb : emissive.rgb;
        // light baked into the vertex colors, as if from fixed lamps
        if (baked) {
            vec3 b = diffuse_color * v_color * emissive.w * ao;
            outc += b;
            ind_d += b;
        }
    }

    if ((fl & 8) != 0) {
        // glass: reflections stay visible however clear the pane is
        float fresnel = 0.04 + 0.96 * pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 5.0);
        vec3 sky = mix(hemi_ground.rgb, hemi_sky.rgb, 0.5 * reflect(-v, n).y + 0.5) + ambient.rgb;
        if (env_params.x > 0.5) sky = env_reflection(reflect(-v, n), g_rough, n);
        reflection_hook(reflect(-v, n), sky);
        spec += sky * fresnel;
        float spec_a = max(spec.r, max(spec.g, spec.b));
        float a = clamp(alpha + fresnel + spec_a, 0.0, 1.0);
        // blended with src-alpha: pre-divide so that outc * alpha + spec comes out
        outc = (outc * alpha + spec) / max(a, 1e-4);
        alpha = a;
    } else {
        vec3 env_term = vec3(0.0);
        if (env_params.x > 0.5 && mode != 1) {
            // reflections of the surroundings, fading as the surface gets rougher
            float nov = clamp(dot(n, v), 0.0, 1.0);
            vec3 f0 = mix(vec3(0.04), g_base.rgb, g_metal);
            vec3 fr = f0 + (max(vec3(1.0 - g_rough), f0) - f0) * pow(1.0 - nov, 5.0);
            vec3 r = reflect(-v, n);
            vec3 env = env_reflection(r, g_rough, n);
            reflection_hook(r, env);
            env_term = env * fr;
        }
        if (ssr_params.x > 0.0 || reflector_params.x > 0.0) {
            // the surface's sheen: a mirror image, strongest at a glancing angle
            // (far off, where a pixel covers many bumps of the normal map, each would throw
            // the image somewhere else: blotches. There the surface mirrors as if flat)
            float bumps = 0.5 * (1.0 - smoothstep(1.5, 5.0, length(camera_pos.xyz - v_world_pos)));
            vec3 rn = normalize(mix(geo_n, n, bumps));
            float fresnel = 0.04 + 0.96 * pow(1.0 - clamp(dot(rn, v), 0.0, 1.0), 5.0);
            vec3 mirrored = vec3(0.0);
            float found = 0.0;
            float strength = 0.0;
            if (reflector_params.x > 0.0) {
                // the planar reflection: the picture of the mirrored scene where the ray points
                mirrored = planar_reflection(rn, v, g_rough, found);
                found *= reflector_params.x;
                strength = 1.0;
            }
            if (ssr_params.x > 0.0) {
                // what is on screen near it, where a ray finds that: sharper, at full size
                float near_found;
                vec3 near_mirrored = screen_reflection(rn, geo_n, v, g_rough, near_found);
                mirrored = found > 0.0 ? mix(mirrored, near_mirrored, near_found) : near_mirrored;
                found = max(found, near_found);
                strength = max(strength, ssr_params.x);
            }
            // where something was found, it takes the place of the surroundings' map
            vec3 mirror_term = mirrored * (found * fresnel * strength * ao);
            env_term *= 1.0 - clamp(found * strength, 0.0, 1.0);
            outc += mirror_term;
            ind_s += mirror_term;
        }
        outc += env_term;
        ind_s += env_term;
        outc += spec;
    }

    int fog_mode = int(fog_params.w);
    if (fog_mode != 0 && fog_grid.w > 0.0) {
        vec4 fv = fog_volume_at(v_world_pos);
        outc = outc * fv.a + fv.rgb;
        ind_d *= fv.a;
        ind_s *= fv.a;
    } else if (fog_mode != 0) {
        float fd = fog_distance(camera_pos.xyz, v_world_pos);
        vec3 fog_own = fog_color.rgb;
        if (fog_height.y > 0.0) {
            // fog lying on the ground: the mean of exp(-k * height) along the ray
            float ya = max(camera_pos.y - fog_height.x, 0.0);
            float yb = max(v_world_pos.y - fog_height.x, 0.0);
            float k = fog_height.y;
            float dy = yb - ya;
            fd *= abs(dy) > 0.01 ? (exp(-k * ya) - exp(-k * yb)) / (k * dy) : exp(-k * ya);
        }
        if (fog_sun_dir.w > 0.0) {
            vec3 look = normalize(v_world_pos - camera_pos.xyz);
            float side = dot(normalize(look.xz + vec2(1e-5)), normalize(fog_sun_dir.xz + vec2(1e-5)));
            fd *= mix(fog_height.w, 1.0, smoothstep(-0.3, 0.7, side));
            float sunlit = pow(max(dot(look, fog_sun_dir.xyz), 0.0), fog_sun_dir.w);
            if (fog_sun_color.w > 0.0 && dir_shadow_params.x > 0.5 && fog_mode == 3) {
                // light shafts: the sun's part of the fog only where the sun gets to it
                float len = length(v_world_pos - camera_pos.xyz);
                float lit = sun_in_fog(camera_pos.xyz, v_world_pos, fog_params.z * fd / max(len, 1e-4));
                sunlit *= mix(1.0, lit, fog_sun_color.w);
            }
            fog_own = mix(fog_own, fog_sun_color.rgb, sunlit);
        }
        float t = fog_transmittance(fd);
        vec3 fog_lit = fog_own * (1.0 - t);
        if (fog_color.w > 0.0 && fd > 0.0) {
            vec3 ray = -v;
            float in_fog = fd / max(depth, 1e-4);
            vec3 glow = vec3(0.0);
            // (all of them: the glow is along the whole view ray, not at the fragment's cluster)
            int npoint = int(counts.y);
            for (int i = 0; i < npoint; i++) {
                glow += points[i].color.rgb * fog_glow(points[i].pos.xyz, points[i].params.x, ray, depth, vec4(0.0), -2.0);
            }
            int nspot = int(counts.z);
            for (int i = 0; i < nspot; i++) {
                if (spots[i].params.w <= 0.0) continue;
                glow += spots[i].color.rgb * spots[i].params.w
                    * fog_glow(spots[i].pos.xyz, spots[i].params.y, ray, depth, spots[i].dir, spots[i].params.x);
            }
            fog_lit += glow * fog_color.w * in_fog;
        }
        outc = outc * t + fog_lit;
        ind_d *= t;
        ind_s *= t;
    }

    outc = max(outc, vec3(0.0));
#ifdef MRT
    {
        const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
        float total = max(dot(outc, lw), 1e-6);
        frag_indirect.rg = clamp(vec2(dot(max(ind_d, 0.0), lw), dot(max(ind_s, 0.0), lw)) / total, 0.0, 1.0);
    }
#endif
    // a light source (emissive far over white) leaves the target's alpha at 0, for
    // post-processing that has to tell a burnt-out lamp from a brightly lit wall
    alpha *= 1.0 - smoothstep(4.0, 16.0, max(emissive.r, max(emissive.g, emissive.b)));
    // into a half-float target as it is (OutputPass tone maps it later), otherwise ready to show
    if (output_params.x > 0.5) frag_color = vec4(min(outc, vec3(60000.0)), alpha);
    else {
        // the window's colors: sRGB, Display P3, or extended linear Display P3 (HDR: in SDR's range here)
        int display = int(output_params.w);
        vec3 toned = tone_map_gamut(outc, int(output_params.z), camera_pos.w, display != DISPLAY_SRGB);
        frag_color = vec4(display == DISPLAY_HDR ? toned : linear_to_srgb(toned), alpha);
    }
}
@end

@fs fs
@include_block fs_main
@end

// The same without discard: for what the depth prepass has cut out already, and what has
// no holes
@fs fs_opaque
#define NO_DISCARD
@include_block fs_main
@end

// The same two, also writing the normal and the motion (RenderTarget.mrt)
@fs fs_mrt
#define MRT
@include_block fs_main
@end

@fs fs_opaque_mrt
#define NO_DISCARD
#define MRT
@include_block fs_main
@end

// The depth prepass (renderer.c3, Renderer.depth_prepass): depth only, for what has holes in
// it (alpha tested, LOD cross-fades, the cull box). The main pass then draws those with
// fs_opaque and an EQUAL depth test, so that no shader of the main pass discards (a shader
// that can discard costs the GPU its early depth test, whatever it draws). Cuts out exactly
// what fs does.
@fs depth_fs
layout(binding=0) uniform texture2D tex_map;
layout(binding=0) uniform sampler tex_smp;
layout(binding=1) uniform depth_fs_params {
    vec4 d_color;          // a = the material's opacity
    vec4 d_params;         // x = has a map, y = alpha test threshold, z = 1: vertex colors' alpha
    vec4 d_screen_params;  // as screen_params in fs
    vec4 d_cull_min;       // as cull_min in fs
    vec4 d_cull_max;
};

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in vec3 v_color;
in float v_alpha;
in vec4 v_clip;
in vec4 v_prev_clip;
in float v_line_distance;

out vec4 frag_color;

void main() {
    if (d_cull_min.w > 0.5 && all(greaterThan(v_world_pos, d_cull_min.xyz)) && all(lessThan(v_world_pos, d_cull_max.xyz))) discard;
    if (d_screen_params.w != 0.0) {
        float dither = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        if (d_screen_params.w > 0.0 ? dither >= d_screen_params.z : dither < d_screen_params.z) discard;
    }
    float alpha = 1.0;
    if (d_params.y > 0.0) {
        alpha = d_color.a;
        if (d_params.x > 0.5) alpha *= texture(sampler2D(tex_map, tex_smp), v_uv).a;
        if (d_params.z > 0.5) alpha *= v_alpha;
        if (d_params.x > 0.5) {
            vec2 texel = v_uv * vec2(textureSize(sampler2D(tex_map, tex_smp), 0));
            vec2 tx = dFdx(texel);
            vec2 ty = dFdy(texel);
            float level = 0.5 * log2(max(max(dot(tx, tx), dot(ty, ty)), 1.0));
            alpha *= 1.0 + 0.3 * min(level, 6.0);
        }
        alpha = (alpha - d_params.y) / max(fwidth(alpha), 1e-4) + 0.5;
        if (alpha <= 0.0) discard;
        alpha = min(alpha, 1.0);
    }
    // for alpha to coverage (the depth prepass writes no color); r = the distance from the
    // camera's plane, for Renderer.render_back_faces
    frag_color = vec4(v_clip.w + (v_normal + v_color + v_world_pos + v_clip.xyz + v_prev_clip.xyz).x * 0.0
        + v_line_distance * 0.0, 0.0, 0.0, alpha);
}
@end

@program three vs fs
@program three_opaque vs fs_opaque
@program three_mrt vs fs_mrt
@program three_opaque_mrt vs fs_opaque_mrt
@program depth vs depth_fs
@program shadow shadow_vs shadow_fs
