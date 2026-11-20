// Node materials (nodes.c3): the engine's shaders with the bodies of node_vertex, node_color,
// node_normal, node_surface and node_output swapped, at runtime, for code printed from the
// material's node graph in the backend's own language (Metal, HLSL, GLSL), as three prints
// TSL into WGSL or GLSL. So that each one's signature in sokol-shdc's output stays put whatever the graph
// does, they read nothing but their parameters (a NodeIn, and in the fragment stage the six
// tex_ext textures with tex_smp), and write all their outputs. The stubs here do just that;
// they run only when node_flags.x (the vertex shader's bake.z) is 1, which only a node
// material's own build sets.
@block node_common
#define NODE_PARAMS 32
// The graph's uniforms into a NodeIn, one by one: from a copy made in a loop, Apple's Metal
// compiler reads zeros where a value decides a discard_fragment (macOS 27, Apple GPUs)
#define NODE_COPY_PARAMS(dst, src) \
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3]; \
    dst[4] = src[4]; dst[5] = src[5]; dst[6] = src[6]; dst[7] = src[7]; \
    dst[8] = src[8]; dst[9] = src[9]; dst[10] = src[10]; dst[11] = src[11]; \
    dst[12] = src[12]; dst[13] = src[13]; dst[14] = src[14]; dst[15] = src[15]; \
    dst[16] = src[16]; dst[17] = src[17]; dst[18] = src[18]; dst[19] = src[19]; \
    dst[20] = src[20]; dst[21] = src[21]; dst[22] = src[22]; dst[23] = src[23]; \
    dst[24] = src[24]; dst[25] = src[25]; dst[26] = src[26]; dst[27] = src[27]; \
    dst[28] = src[28]; dst[29] = src[29]; dst[30] = src[30]; dst[31] = src[31];
struct NodeIn {
    vec3 local;          // the position in the model (with an instance's matrix)
    vec3 local_normal;   //   and its normal (vertex stage)
    vec3 world;          // the position in the world (fragment: before the cull box...)
    vec3 normal;         // the normal in the world: in the fragment stage the shaded one
    vec3 geo_normal;     //   and the surface's own, facing the camera (fragment stage)
    vec2 uv;
    vec2 uv1;
    vec4 color;          // the vertex color, a = its alpha
    vec4 frag;           // fragment stage: xy = gl_FragCoord.xy, z = 1 on a front face, 0 on a back one
    vec4 vary0;          // what node_vertex passed on (fragment stage)
    vec4 vary1;
    vec4 vary2;
    vec4 vary3;
    mat4 model;          // model -> world
    mat4 view;           // world -> view
    mat4 proj;           // view -> clip
    vec4 info;           // xyz = the camera's position, w = time (s)
    vec4 params[NODE_PARAMS]; // the graph's uniforms
};
@end

// Shared by the main and the shadow vertex shaders, so both move vertices identically.
@block vertex_common
@include_block node_common
in vec3 position;
in vec3 normal;
in vec2 uv;
in vec4 color0;
in vec2 uv1;      // the second uv set (glTF's TEXCOORD_1, three's uv1), zeros without one
// Rows of the per-instance 3x4 matrix (identity for plain meshes).
in vec4 inst0;
in vec4 inst1;
in vec4 inst2;
// Material.displacement_map at the vertex (0 without one): it moves out along its normal by
// this times the displacement's scale, plus its bias (see displaced)
in float height;

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

// Material.displacement_map: `p` out along the vertex's normal by its height times d.x plus
// d.y (d.z = 1: there is a map).
vec3 displaced(vec3 p, vec4 d) {
    return d.z > 0.5 ? p + normalize(normal) * (height * d.x + d.y) : p;
}

// A node material's: see node_common
layout(binding=2) uniform node_vs_params {
    mat4 node_view;
    mat4 node_proj;
    mat4 node_model;
    vec4 node_info;
    vec4 node_params[NODE_PARAMS];
};

void node_vertex(NodeIn ni, inout vec3 local, inout vec4 vary0, inout vec4 vary1, inout vec4 vary2, inout vec4 vary3) {
    local += ni.params[0].xyz;
    vary0 = ni.params[1];
    vary1 = ni.params[2];
    vary2 = ni.params[3];
    vary3 = ni.params[4];
}

// Runs a node material's node_vertex on `local` (the vertex with its instance's matrix),
// whose varyings come out in vary.
void run_node_vertex(inout vec3 local, vec3 local_normal, mat4 model, out vec4 vary[4]) {
    NodeIn ni;
    ni.local = local;
    ni.local_normal = local_normal;
    ni.world = (model * vec4(local, 1.0)).xyz;
    ni.normal = normalize((model * vec4(local_normal, 0.0)).xyz);
    ni.geo_normal = ni.normal;
    ni.uv = uv;
    ni.uv1 = uv1;
    ni.color = color0;
    ni.frag = vec4(0.0);
    ni.vary0 = vec4(0.0);
    ni.vary1 = vec4(0.0);
    ni.vary2 = vec4(0.0);
    ni.vary3 = vec4(0.0);
    ni.model = model;
    ni.view = node_view;
    ni.proj = node_proj;
    ni.info = node_info;
    NODE_COPY_PARAMS(ni.params, node_params);
    vec4 v0 = vec4(0.0);
    vec4 v1 = vec4(0.0);
    vec4 v2 = vec4(0.0);
    vec4 v3 = vec4(0.0);
    node_vertex(ni, local, v0, v1, v2, v3);
    vary[0] = v0;
    vary[1] = v1;
    vary[2] = v2;
    vary[3] = v3;
}
@end

// The main vertex shader, as vs, and as vs_tangent (TANGENT) for geometries with tangents
// (three's USE_TANGENT): the attribute and what it hands the fragment shader cost every vertex
// drawn, so the others go without.
@block vs_main
layout(binding=0) uniform vs_params {
    // world -> clip space, measured from view_origin: gl_Position = view_proj * (world - origin).
    // Far from the world's origin (a farm 6 km out) a plain view_proj * world loses the depth:
    // clip z and w come out as the small difference of two large sums, rounded apart, and
    // surfaces centimetres apart swap places from frame to frame
    mat4 view_proj;
    vec4 view_origin;   // xyz: the camera's position
    mat4 model;
    mat4 normal_mat;
    vec4 uv_transform[2]; // the color map's uv transform (three's texture.matrix): u' = dot(row 0,
                          //   (u, v, 1)), v' = dot(row 1, (u, v, 1))
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
    vec4 bake;          // x = 1: baking a light map (lightmap.c3): drawn where v_uv1 is in it,
                        //   y = 1: the target's row 0 is its top (texel row r at v = (r + 0.5) / size);
                        //   z = 1: a node material's own build (nodes.c3), node_vs_params set
    vec4 surface_vs;    // x = Material.outline: how far the vertices are pushed out along their
                        //   normals, in clip space per unit of w (0: not); yzw = the displacement
                        //   (see displaced). (The block is 512 bytes: sokol rounds a draw's
                        //   uniforms up to 256, and a bake draws thousands of times a frame)
};

@include_block vertex_common
// the instance's color (InstancedMesh.set_color_at, three's instanceColor), linear; white
// for meshes without (before the tangent: the same location in every program)
in vec3 instance_color;
#ifdef TANGENT
in vec4 tangent;  // the tangents (BufferGeometry.tangents: glTF's TANGENT)
#endif

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec2 v_uv1;
#ifdef TANGENT
out vec4 v_tangent;   // the tangent in the world, w = the bitangent's sign
#endif
out vec3 v_color;
out float v_alpha;
out vec4 v_clip;
out vec4 v_prev_clip;
// lines: the distance along the line, for dashes (a line's vertex normal x holds it; lines
// have no normals)
out float v_line_distance;
out vec3 v_instance_color;

out vec4 v_node0;     // a node material's varyings (node_vertex)
out vec4 v_node1;
out vec4 v_node2;
out vec4 v_node3;

void main() {
    v_line_distance = normal.x;
    v_instance_color = instance_color;
    v_node0 = vec4(0.0);
    v_node1 = vec4(0.0);
    v_node2 = vec4(0.0);
    v_node3 = vec4(0.0);
    if (!fetch_instance(inst_cull)) {
        // outside the clip volume: nothing drawn
        v_world_pos = vec3(0.0);
        v_normal = vec3(0.0, 1.0, 0.0);
        v_uv = vec2(0.0);
        v_uv1 = vec2(0.0);
#ifdef TANGENT
        v_tangent = vec4(0.0);
#endif
        v_color = vec3(0.0);
        v_alpha = 0.0;
        v_clip = vec4(2.0, 2.0, 2.0, 1.0);
        v_prev_clip = v_clip;
        gl_Position = v_clip;
        return;
    }
    vec3 ip = instance_point(displaced(position, vec4(surface_vs.yzw, 0.0)));
    if (bake.z > 0.5) {
        vec4 vary[4];
        run_node_vertex(ip, instance_dir(normal), model, vary);
        v_node0 = vary[0];
        v_node1 = vary[1];
        v_node2 = vary[2];
        v_node3 = vary[3];
    }
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
    v_uv = vec2(dot(uv_transform[0].xyz, vec3(uv, 1.0)), dot(uv_transform[1].xyz, vec3(uv, 1.0)));
    v_uv1 = uv1;
#ifdef TANGENT
    // (a direction along the surface: the model matrix's, as three's)
    v_tangent = vec4((model * vec4(instance_dir(tangent.xyz), 0.0)).xyz, tangent.w);
#endif
    v_color = color0.rgb;
    v_alpha = color0.a;
    gl_Position = view_proj * vec4(wp.xyz - view_origin.xyz, 1.0);
    if (surface_vs.x > 0.0) {
        // Material.outline: out along the normal as the screen sees it, the same width at any
        // distance (three's OutlineEffect)
        vec3 inward = wp.xyz - (model * vec4(instance_dir(normal), 0.0)).xyz;
        vec4 p2 = view_proj * vec4(inward - view_origin.xyz, 1.0);
        gl_Position += normalize(gl_Position - p2) * surface_vs.x * gl_Position.w;
    }
    v_clip = gl_Position;
    v_clip.xy -= motion.zw * v_clip.w;
    v_prev_clip = v_clip;
    if (bake.x > 0.5) {
        gl_Position = vec4(2.0 * uv1.x - 1.0, bake.y > 0.5 ? 1.0 - 2.0 * uv1.y : 2.0 * uv1.y - 1.0, 0.0, 1.0);
        return;
    }
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

@vs vs
@include_block vs_main
@end

@vs vs_tangent
#define TANGENT
@include_block vs_main
@end

@vs shadow_vs
layout(binding=0) uniform shadow_vs_params {
    mat4 s_view_proj;   // world - s_origin -> the light's clip space (see vs_params)
    vec4 s_origin;      // xyz: the light's position (0 for the sun's orthographic maps)
    mat4 s_model;
    vec4 s_uv_transform[2]; // as uv_transform
    vec4 s_sway;
    vec4 s_inst_cull;
    vec4 s_material_custom[2];
    vec4 s_frame_custom[2];
    vec4 s_displacement;
    vec4 s_node_flags;  // as node_flags in vs_params
};

@include_block vertex_common

out vec2 v_uv;

void main() {
    if (!fetch_instance(s_inst_cull)) {
        v_uv = vec2(0.0);
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    vec3 ip = instance_point(displaced(position, s_displacement));
    if (s_node_flags.x > 0.5) {
        vec4 vary[4];
        run_node_vertex(ip, instance_dir(normal), s_model, vary);
    }
    vec4 wp = s_model * vec4(ip, 1.0);
    wp.xyz = apply_sway(wp.xyz, ip.y, s_sway);
    material_custom[0] = s_material_custom[0];
    material_custom[1] = s_material_custom[1];
    frame_custom[0] = s_frame_custom[0];
    frame_custom[1] = s_frame_custom[1];
    vertex_hook(wp.xyz, ip, s_model);
    v_uv = vec2(dot(s_uv_transform[0].xyz, vec3(uv, 1.0)), dot(s_uv_transform[1].xyz, vec3(uv, 1.0)));
    // keep the unused attributes alive so both programs share one vertex layout
    v_uv += (normal.xy + color0.xy + uv1 + height) * 0.0;
    gl_Position = s_view_proj * vec4(wp.xyz - s_origin.xyz, 1.0);
}
@end

@fs shadow_fs
layout(binding=0) uniform texture2D s_map;
layout(binding=1) uniform texture2D s_alpha;
layout(binding=0) uniform sampler s_smp;
layout(binding=1) uniform sampler s_alpha_smp;
layout(binding=1) uniform shadow_fs_params {
    vec4 s_params;      // x = alpha test threshold (0 = off), y = material opacity, z = 1: has a
                        //   map, w = 1: has an alpha map (s_alpha, three's alphaMap: its green)
    vec4 s_alpha_uv[2]; // the alpha map's uv from v_uv, as uv_rel
};

in vec2 v_uv;

void main() {
    if (s_params.x > 0.0) {
        float a = texture(sampler2D(s_map, s_smp), v_uv).a * s_params.y;
        if (s_params.w > 0.5) {
            vec3 p = vec3(v_uv, 1.0);
            a *= texture(sampler2D(s_alpha, s_alpha_smp), vec2(dot(s_alpha_uv[0].xyz, p), dot(s_alpha_uv[1].xyz, p))).g;
        }
        if (a < s_params.x) discard;
    }
}
@end

// The shadow maps' program for a node material whose color_node or opacity_node can cut it
// (a discard, or an alpha under the material's alpha_test): shadow_vs and shadow_fs, with
// what node_color reads carried to the fragments. Its node_color is spliced as fs's is.
@vs shadow_nodes_vs
layout(binding=0) uniform shadow_vs_params {
    mat4 s_view_proj;
    vec4 s_origin;
    mat4 s_model;
    vec4 s_uv_transform[2];
    vec4 s_sway;
    vec4 s_inst_cull;
    vec4 s_material_custom[2];
    vec4 s_frame_custom[2];
    vec4 s_displacement;
    vec4 s_node_flags;
};

@include_block vertex_common

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec2 v_uv1;
out vec4 v_color;
out vec4 v_node0;
out vec4 v_node1;
out vec4 v_node2;
out vec4 v_node3;

void main() {
    v_world_pos = vec3(0.0);
    v_normal = vec3(0.0, 1.0, 0.0);
    v_uv = vec2(0.0);
    v_uv1 = vec2(0.0);
    v_color = vec4(0.0);
    v_node0 = vec4(0.0);
    v_node1 = vec4(0.0);
    v_node2 = vec4(0.0);
    v_node3 = vec4(0.0);
    if (!fetch_instance(s_inst_cull)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    vec3 ip = instance_point(displaced(position, s_displacement));
    vec4 vary[4];
    run_node_vertex(ip, instance_dir(normal), s_model, vary);
    v_node0 = vary[0];
    v_node1 = vary[1];
    v_node2 = vary[2];
    v_node3 = vary[3];
    vec4 wp = s_model * vec4(ip, 1.0);
    wp.xyz = apply_sway(wp.xyz, ip.y, s_sway);
    material_custom[0] = s_material_custom[0];
    material_custom[1] = s_material_custom[1];
    frame_custom[0] = s_frame_custom[0];
    frame_custom[1] = s_frame_custom[1];
    vertex_hook(wp.xyz, ip, s_model);
    v_world_pos = wp.xyz;
    v_normal = (s_model * vec4(instance_dir(normal), 0.0)).xyz;
    v_uv = vec2(dot(s_uv_transform[0].xyz, vec3(uv, 1.0)), dot(s_uv_transform[1].xyz, vec3(uv, 1.0)));
    v_uv1 = uv1;
    v_color = color0;
    gl_Position = s_view_proj * vec4(wp.xyz - s_origin.xyz, 1.0);
}
@end

@fs shadow_nodes_fs
@include_block node_common
layout(binding=0) uniform texture2D tex_map;
layout(binding=26) uniform texture2D tex_ext0;
layout(binding=27) uniform texture2D tex_ext1;
layout(binding=28) uniform texture2D tex_ext2;
layout(binding=29) uniform texture2D tex_ext3;
layout(binding=30) uniform texture2D tex_ext4;
layout(binding=31) uniform texture2D tex_ext5;
layout(binding=0) uniform sampler tex_smp;
layout(binding=1) uniform shadow_fs_params {
    vec4 s_params;      // as shadow_fs's
    vec4 s_alpha_uv[2];
};
layout(binding=3) uniform node_fs_params {
    mat4 node_view;
    mat4 node_proj;
    mat4 node_model;
    vec4 node_info;
    vec4 node_params[NODE_PARAMS];
};

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in vec2 v_uv1;
in vec4 v_color;
in vec4 v_node0;
in vec4 v_node1;
in vec4 v_node2;
in vec4 v_node3;

// fs_main's stub
void node_color(NodeIn ni, inout vec4 base) {
    vec2 uv = ni.uv;
    base *= texture(sampler2D(tex_ext0, tex_smp), uv) * texture(sampler2D(tex_ext1, tex_smp), uv)
        * texture(sampler2D(tex_ext2, tex_smp), uv) * texture(sampler2D(tex_ext3, tex_smp), uv)
        * texture(sampler2D(tex_ext4, tex_smp), uv) * texture(sampler2D(tex_ext5, tex_smp), uv) * ni.params[0];
}

void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;
    NodeIn ni;
    ni.local = vec3(0.0);
    ni.local_normal = vec3(0.0);
    ni.world = v_world_pos;
    ni.normal = n;
    ni.geo_normal = n;
    ni.uv = v_uv;
    ni.uv1 = v_uv1;
    ni.color = v_color;
    ni.frag = vec4(gl_FragCoord.xy, gl_FrontFacing ? 1.0 : 0.0, 0.0);
    ni.vary0 = v_node0;
    ni.vary1 = v_node1;
    ni.vary2 = v_node2;
    ni.vary3 = v_node3;
    ni.model = node_model;
    ni.view = node_view;
    ni.proj = node_proj;
    ni.info = node_info;
    NODE_COPY_PARAMS(ni.params, node_params);
    vec4 base = vec4(1.0, 1.0, 1.0, s_params.y);
    if (s_params.z > 0.5) base *= texture(sampler2D(tex_map, tex_smp), v_uv);
    node_color(ni, base);
    if (s_params.x > 0.0 && base.a < s_params.x) discard;
}
@end

@block fs_main
@include_block node_common
layout(binding=0) uniform texture2D tex_map;
layout(binding=1) uniform texture2D tex_normal;
layout(binding=2) uniform texture2D tex_arm;
layout(binding=4) uniform texture2D tex_env;
layout(binding=6) uniform texture3D tex_gi;
layout(binding=7) uniform texture2D ssr_color;   // the last frame: tone mapped, sRGB, a = 0 at light sources
layout(binding=8) uniform texture2D ssr_depth;   // g = its distance from the camera's plane (m)
#ifdef VSM
// Renderer.shadow_type = VSM: the maps' moments (mean depth, its standard deviation; vsm.glsl)
// in place of their depths, on the same slots (the fog's kernel reads the depths)
layout(binding=3) uniform texture2D vsm_map;
layout(binding=5) uniform texture2D vsm_atlas;
layout(binding=9) uniform texture2D vsm_sun;
#else
layout(binding=3) uniform texture2D shadow_map;
layout(binding=5) uniform texture2D shadow_atlas;
layout(binding=9) uniform texture2D sun_shadow;  // the directional light's map: 2x2 cascades, or one box
#endif
layout(binding=21) uniform texture2D reflector_map; // a Reflector's mirrored render, linear
layout(binding=20) uniform texture2D ssr_back;    // r = the nearest back face's distance from the camera's plane (m), 0: none
layout(binding=23) uniform texture2DArray tex_probes; // the scene's reflection probes (reflection_probes.c3)
layout(binding=24) uniform texture2D tex_parallax; // Material.parallax_map: the surface's height, 1 = top
layout(binding=19) uniform texture2D fog_volume; // volumetric fog: the froxel grid (fog.c3, fog_cs in kernels.glsl)
layout(binding=25) uniform texture2D tex_transmission; // the scene behind the transmission materials, with mip levels
// The material's other maps (ao_map, specular_color_map, clearcoat_map...): six slots they
// share, which holds which in map_slots (see material_map)
layout(binding=26) uniform texture2D tex_ext0;
layout(binding=27) uniform texture2D tex_ext1;
layout(binding=28) uniform texture2D tex_ext2;
layout(binding=29) uniform texture2D tex_ext3;
layout(binding=30) uniform texture2D tex_ext4;
layout(binding=31) uniform texture2D tex_ext5;
layout(binding=0) uniform sampler tex_smp;
#ifdef VSM
layout(binding=1) uniform sampler vsm_smp;
#else
layout(binding=1) uniform sampler shadow_smp;
#endif
layout(binding=2) uniform sampler env_smp;
layout(binding=3) uniform sampler gi_smp;
layout(binding=4) uniform sampler ssr_smp;
layout(binding=5) uniform sampler transmission_smp;

layout(binding=1) uniform fs_params {
    vec4 color;        // linear rgb, a = opacity
    vec4 emissive;     // linear rgb * intensity
    vec4 specular;     // phong specular color, w = flags (1: vertex colors, 2: foliage, 4: receive shadow, 8: glass,
                       //   16: vertex colors are baked light (emissive.w = strength), 32: screen-space map,
                       //   64: the map glows, 128: vertex occlusion, 256: premultiplied alpha,
                       //   512: not tone mapped)
    vec4 mat_params;   // x = mode, y = roughness, z = metalness, w = shininess
    vec4 tex_params;   // x = has map, y = has a normal map (tangent_params: its scale), z = what tex_arm holds (1: ambient
                       //   occlusion in r, 2: roughness in g, 4: metalness in b), w = alpha test
    vec4 env_params;   // x = has environment map, y = intensity, z = 1: the scene's environment
                       // lights the diffuse part too (env_sh), w = meters that alpha 1 of the
                       // map stands for (0: it holds no distances)
    vec4 env_pos;      // where the map was captured, w = 1 for box projection
    vec4 env_box_min;  // the room the map shows, for box projection
    vec4 env_box_max;
    vec4 env_sh[9];    // the scene's environment's irradiance, as Texture.env_light: its
                       //   harmonics (rgb), its sun's direction (w of 0..2) and light (w of 3..5)
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
    // irradiance volumes (Scene.irradiance and irradiance_volumes, MAX_IRRADIANCE_VOLUMES):
    vec4 gi_min[4];       // each one's box, w = 1 (the first's 2 when they replace the ambient light), 0 past the last
    vec4 gi_inv_size[4];  // w = intensity
    vec4 gi_dims[4];      // probes along x, y, z, w = lookup offset along the normal (m)
    vec4 gi_atlas[4];     // where it is in tex_gi: x = its first column, y = its width, z = height, w = depth (0..1)
    vec4 fog_color;    // w = how strongly lights glow in the fog
    vec4 fog_params;   // x = near, y = far, z = density, w = mode (0 none, 1 linear, 2 exp2, 3 exp)
    vec4 fog_box_min;  // fog-free box, w = 1 when enabled
    vec4 fog_box_max;  // w = the part of the density left in the box
    vec4 fog_height;   // x = base height, y = falloff per meter (0 = none), z = 1: mixed in after tone mapping
                       //   (Renderer.display_fog), w = density away from the sun's side
    vec4 fog_sun_dir;  // towards the sun, w = sharpness (0 = no sun)
    vec4 fog_sun_color; // w = light shafts (0..1)
    vec4 fog_grid;     // volumetric fog: x, y = cells across and up, z = slices, w = slices per row of
                       //   tiles in fog_volume (0: no volumetric fog, the fog is worked out here)
    vec4 fog_slices;   // x = where its slices start (m of view depth), y = slices per unit of
                       //   log(depth / x)
    vec4 counts;       // x = directional lights, y = point lights, z = spot lights, w = rect area lights (all of them, in
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
    vec4 quality_params;   // x = Renderer.shadow_sample_scale, y = ssr_steps, z = ssr_max_step (px)
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
                               //   it has faded out, z = self-shadowing toward the sun (0..1);
                               //   w = Material.hex_tiling (0: none)
    vec4 line_dash;            // dashed lines: x = dash length, y = dash + gap, z = scale, w = 1: dashed
    vec4 transmission_params;  // Material.transmission: x = IOR, y = thickness (m), z = dispersion,
                               //   w = attenuation distance (m; 0: the volume does not tint)
    vec4 transmission_color;   // linear rgb = the color the volume tints light toward, a = how much
                               //   of what is behind shows through (0: off)
    vec4 specular_params;      // Material.specular_color (linear rgb) and specular_intensity (a)
    vec4 ext_params;           // x = 32: it is the scene's background, y = ao_intensity, z = the
                               //   object's scale (for the thickness), w = the width of
                               //   tex_transmission (pixels)
    vec4 map_slots[6];         // which of tex_ext0..5 holds each of the maps (-1: none), by MAP_*
    vec4 clearcoat_params;     // Material.clearcoat: x = strength (0: none), y = roughness,
                               //   zw = clearcoat_normal_scale
    vec4 sheen_params;         // Material.sheen_color * sheen (linear rgb; 0: none), a = sheen_roughness
    vec4 iridescence_params;   // Material.iridescence: x = strength (0: none), y = the film's IOR,
                               //   z, w = its thinnest and thickest (nm)
    vec4 tangent_params;       // what is read in the tangent frame: xy = Material.normal_scale, zw =
                               //   Material.anisotropy's direction in the frame (the cos and sin
                               //   of anisotropy_rotation) times its strength (0: none)
    vec4 light_map_params;     // x = Material.light_map_intensity
    vec4 camera_right;         // the camera's x and y axes in the world: with camera_pos, its view
    vec4 camera_up;            //   space (MeshNormalMaterial's and the matcap's normals)
    vec4 surface_params;       // x = Material.bump_scale (three's bumpScale), y = 1: the toon
                               //   material's gradient map is read texel by texel (NearestFilter),
                               //   z = MeshDepthMaterial's packing (Material.depth_packing)
    vec4 env_combine;          // a basic, lambert or phong material's env_map (three's): x =
                               //   reflectivity, y = combine (0 multiply, 1 mix, 2 add), z = the
                               //   refraction ratio (0: it mirrors), w = 1: on
    vec4 focus_params;         // Material.defocus: x = Renderer.focus_aperture (0: not), y = focus_maxblur
    vec4 shadow_ext;           // x = the shadow casting directional light's shadow.bias (VSM)
    vec4 bake_params;          // baking a light map (lightmap.c3): x = 1, y = the new sample's
                               //   share (1 / blend window); ssr_color holds the last map
    vec4 uv_rel[16];           // per map slot (0: tex_normal, 1: tex_arm, 2..7: tex_ext0..5), its uv
                               //   from the color map's (g_uv; each texture has its own
                               //   transform, three's texture.matrix): two rows, as uv_transform;
                               //   the first's w = 1: from the second uv set instead (v_uv1)
    vec4 node_flags;           // x = 1: a node material's own build (nodes.c3), node_fs_params set
};

// The maps in map_slots, and the channels each is read from (as three's)
#define MAP_AO 0                    // r
#define MAP_SPECULAR_COLOR 1        // rgb
#define MAP_SPECULAR_INTENSITY 2    // a
#define MAP_TRANSMISSION 3          // r
#define MAP_THICKNESS 4             // g
#define MAP_IRIDESCENCE 5           // r
#define MAP_IRIDESCENCE_THICKNESS 6 // g
#define MAP_CLEARCOAT 7             // r
#define MAP_CLEARCOAT_ROUGHNESS 8   // g
#define MAP_CLEARCOAT_NORMAL 9      // rgb
#define MAP_SHEEN_COLOR 10          // rgb
#define MAP_SHEEN_ROUGHNESS 11      // a
#define MAP_ALPHA 12                // g
#define MAP_EMISSIVE 13             // rgb
#define MAP_LIGHT 14                // rgb
#define MAP_ANISOTROPY 15           // rg = direction, b = strength
#define MAP_CUSTOM 16               // Material.custom_map, for the application's hooks
#define MAP_FOCUS 17                // Renderer.focus_map (Material.defocus): r = the distance in focus
#define MAP_SPECULAR 18             // r: three's specularMap (basic, lambert and phong)
#define MAP_MATCAP 19               // rgb: MeshMatcapMaterial's matcap, read by the view-space normal
#define MAP_GRADIENT 20             // r: MeshToonMaterial's gradientMap, read by the light's angle
#define MAP_BUMP 21                 // r: three's bumpMap, a height

// Lights live in storage buffers, as many as there are. Point and spot lights are sorted into
// a grid of clusters (froxels: tiles of the screen, cut into slices of depth that grow
// exponentially), and a fragment only looks at the lights of its own cluster.
struct gpu_dir_light {
    vec4 dir;    // world-space direction towards the light
    vec4 color;
    vec4 shadow; // x = its shadow in spot_shadows (-1: none; the sun's is dir_shadow_mat's),
                 //   y = shadow intensity (0..1)
};
struct gpu_point_light {
    vec4 pos;    // w = distance cutoff
    vec4 color;  // w = decay
    vec4 shadow; // x = the first of its six maps in spot_shadows (+x -x +y -y +z -z; -1: none),
                 //   y = shadow intensity (0..1)
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
    mat4 mat;    // world - the light's position -> shadow map uv + depth (directional: world ->)
    vec4 params; // x = 1: in shadow_map, 2: in shadow_atlas, 3: a directional light's box in
                 //   shadow_atlas, y = normal bias (m), z = texel size, w = blur radius in texels
                 //   (Renderer.shadow_type = VSM: shadow.bias, added to the depth)
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
in vec2 v_uv1;
#ifdef TANGENT
in vec4 v_tangent;
#endif
in vec3 v_color;
in float v_alpha;
in vec4 v_clip;
in vec4 v_prev_clip;
in float v_line_distance;
in vec3 v_instance_color;
in vec4 v_node0;
in vec4 v_node1;
in vec4 v_node2;
in vec4 v_node3;

// A node material's (see node_common): its uniforms, as node_vs_params
layout(binding=3) uniform node_fs_params {
    mat4 node_view;
    mat4 node_proj;
    mat4 node_model;
    vec4 node_info;
    vec4 node_params[NODE_PARAMS];
};

// The stubs: the color (three's colorNode and opacityNode; before the alpha test), the normal
// (normalNode; after the normal map), roughness, metalness and the glow (emissive.w = 1: its
// rgb in place of the material's emissive), and the lit color with its alpha (outputNode;
// before the fog)
void node_color(NodeIn ni, inout vec4 base) {
    vec2 uv = ni.uv;
    base *= texture(sampler2D(tex_ext0, tex_smp), uv) * texture(sampler2D(tex_ext1, tex_smp), uv)
        * texture(sampler2D(tex_ext2, tex_smp), uv) * texture(sampler2D(tex_ext3, tex_smp), uv)
        * texture(sampler2D(tex_ext4, tex_smp), uv) * texture(sampler2D(tex_ext5, tex_smp), uv) * ni.params[0];
}

void node_normal(NodeIn ni, inout vec3 n) {
    vec2 uv = ni.uv;
    n = normalize(n + (texture(sampler2D(tex_ext0, tex_smp), uv) + texture(sampler2D(tex_ext1, tex_smp), uv)
        + texture(sampler2D(tex_ext2, tex_smp), uv) + texture(sampler2D(tex_ext3, tex_smp), uv)
        + texture(sampler2D(tex_ext4, tex_smp), uv) + texture(sampler2D(tex_ext5, tex_smp), uv)).xyz * ni.params[0].x);
}

void node_surface(NodeIn ni, inout float rough, inout float metal, inout vec4 emissive) {
    vec2 uv = ni.uv;
    vec4 t = texture(sampler2D(tex_ext0, tex_smp), uv) * texture(sampler2D(tex_ext1, tex_smp), uv)
        * texture(sampler2D(tex_ext2, tex_smp), uv) * texture(sampler2D(tex_ext3, tex_smp), uv)
        * texture(sampler2D(tex_ext4, tex_smp), uv) * texture(sampler2D(tex_ext5, tex_smp), uv) * ni.params[0];
    rough *= t.g;
    metal *= t.b;
    emissive = t;
}

void node_output(NodeIn ni, inout vec4 result) {
    vec2 uv = ni.uv;
    result *= texture(sampler2D(tex_ext0, tex_smp), uv) * texture(sampler2D(tex_ext1, tex_smp), uv)
        * texture(sampler2D(tex_ext2, tex_smp), uv) * texture(sampler2D(tex_ext3, tex_smp), uv)
        * texture(sampler2D(tex_ext4, tex_smp), uv) * texture(sampler2D(tex_ext5, tex_smp), uv) * ni.params[0];
}

// What a node material's fragment hooks read: `n` the normal they shade with, `geo_n` the
// surface's own (both facing the camera)
NodeIn node_in(vec3 n, vec3 geo_n) {
    NodeIn ni;
    ni.local = vec3(0.0);
    ni.local_normal = vec3(0.0);
    ni.world = v_world_pos;
    ni.normal = n;
    ni.geo_normal = geo_n;
    ni.uv = v_uv;
    ni.uv1 = v_uv1;
    ni.color = vec4(v_color, v_alpha);
    ni.frag = vec4(gl_FragCoord.xy, gl_FrontFacing ? 1.0 : 0.0, 0.0);
    ni.vary0 = v_node0;
    ni.vary1 = v_node1;
    ni.vary2 = v_node2;
    ni.vary3 = v_node3;
    ni.model = node_model;
    ni.view = node_view;
    ni.proj = node_proj;
    ni.info = node_info;
    NODE_COPY_PARAMS(ni.params, node_params);
    return ni;
}

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

// three's window: the light fades out to nothing at `cutoff` (0: never)
float distance_window(float d, float cutoff) {
    if (cutoff <= 0.0) return 1.0;
    float r = d / cutoff;
    float f = clamp(1.0 - r * r * r * r, 0.0, 1.0);
    return f * f;
}

float distance_attenuation(float d, float cutoff, float decay) {
    return distance_window(d, cutoff) / max(pow(d, decay), 0.01);
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

// Anisotropic GGX (Filament's, as three's): alpha t along the direction t, alpha b across it
// (b); the cosines are with t and b.
float v_smith_aniso(float at, float ab, float tov, float bov, float tol, float bol, float nov, float nol) {
    float gv = nol * length(vec3(at * tov, ab * bov, nov));
    float gl = nov * length(vec3(at * tol, ab * bol, nol));
    return 0.5 / max(gv + gl, 1e-6);
}

float d_ggx_aniso(float at, float ab, float noh, float toh, float boh) {
    float a2 = at * ab;
    vec3 d = vec3(ab * toh, at * boh, a2 * noh);
    float w2 = a2 / dot(d, d);
    return RECIPROCAL_PI * a2 * w2 * w2;
}

int flags() { return int(specular.w + 0.5); }

// Sheen (three's BRDF_Sheen): the Charlie distribution of fibers, and Neubelt's visibility.
float d_charlie(float rough, float noh) {
    float inv = 1.0 / (rough * rough);
    float sin2h = max(1.0 - noh * noh, 0.0078125);
    return (2.0 + inv) * pow(sin2h, inv * 0.5) / (2.0 * PI);
}

float v_neubelt(float nov, float nol) {
    return clamp(1.0 / (4.0 * (nol + nov - nol * nov)), 0.0, 1.0);
}

// How much a sheen reflects of the light from all around (three's IBLSheenBRDF fit).
float sheen_albedo(vec3 n, vec3 v, float rough) {
    float nov = clamp(dot(n, v), 0.0, 1.0);
    float r2 = rough * rough;
    float inv = 1.0 / (rough + 0.1);
    float a = -1.9362 + 1.0678 * rough + 0.4573 * r2 - 0.8469 * inv;
    float b = -0.6014 + 0.5538 * rough - 0.4670 * r2 - 0.1255 * inv;
    return clamp(exp(a * nov + b), 0.0, 1.0);
}

// Iridescence (KHR_materials_iridescence, three's evalIridescence: Belcour and Barla's thin
// film): light reflected off the film's top and its bottom interferes, by wavelength.
const mat3 XYZ_TO_REC709 = mat3(
     3.2404542, -0.9692660,  0.0556434,
    -1.5371385,  1.8760108, -0.2040259,
    -0.4985314,  0.0415560,  1.0572252
);

vec3 fresnel0_to_ior(vec3 f0) {
    vec3 s = sqrt(f0);
    return (vec3(1.0) + s) / (vec3(1.0) - s);
}

vec3 ior_to_fresnel0(vec3 transmitted, float incident) {
    vec3 r = (transmitted - vec3(incident)) / (transmitted + vec3(incident));
    return r * r;
}

float ior_to_fresnel0(float transmitted, float incident) {
    float r = (transmitted - incident) / (transmitted + incident);
    return r * r;
}

// The XYZ sensitivity curves in Fourier space, at an optical path difference (nm)
vec3 eval_sensitivity(float opd, vec3 shift) {
    float phase = 2.0 * PI * opd * 1.0e-9;
    vec3 val = vec3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    vec3 pos = vec3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    vec3 var = vec3(4.3278e+09, 9.3046e+09, 6.6121e+09);
    vec3 xyz = val * sqrt(2.0 * PI * var) * cos(pos * phase + shift) * exp(-(phase * phase) * var);
    xyz.x += 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift[0]) * exp(-4.5282e+09 * (phase * phase));
    xyz /= 1.0685e-7;
    return XYZ_TO_REC709 * xyz;
}

float schlick(float f0, float f90, float c) {
    return f0 + (f90 - f0) * pow(1.0 - c, 5.0);
}

vec3 schlick(vec3 f0, float f90, float c) {
    return f0 + (vec3(f90) - f0) * pow(1.0 - c, 5.0);
}

vec3 eval_iridescence(float outside_ior, float eta2, float cos1, float thickness, vec3 base_f0) {
    // (the film fades into the air as it thins out)
    float film_ior = mix(outside_ior, eta2, smoothstep(0.0, 0.03, thickness));
    float sin2_sq = (outside_ior / film_ior) * (outside_ior / film_ior) * (1.0 - cos1 * cos1);
    float cos2_sq = 1.0 - sin2_sq;
    // total internal reflection
    if (cos2_sq < 0.0) return vec3(1.0);
    float cos2 = sqrt(cos2_sq);
    // the first interface
    float r0 = ior_to_fresnel0(film_ior, outside_ior);
    float r12 = schlick(r0, 1.0, cos1);
    float t121 = 1.0 - r12;
    float phi12 = film_ior < outside_ior ? PI : 0.0;
    float phi21 = PI - phi12;
    // the second
    vec3 base_ior = fresnel0_to_ior(clamp(base_f0, 0.0, 0.9999));
    vec3 r1 = ior_to_fresnel0(base_ior, film_ior);
    vec3 r23 = schlick(r1, 1.0, cos2);
    vec3 phi23 = vec3(base_ior.x < film_ior ? PI : 0.0, base_ior.y < film_ior ? PI : 0.0, base_ior.z < film_ior ? PI : 0.0);
    // the phase shift
    float opd = 2.0 * film_ior * thickness * cos2;
    vec3 phi = vec3(phi21) + phi23;
    vec3 r123 = clamp(r12 * r23, 1e-5, 0.9999);
    vec3 sr123 = sqrt(r123);
    vec3 rs = t121 * t121 * r23 / (vec3(1.0) - r123);
    vec3 result = r12 + rs;
    vec3 cm = rs - t121;
    for (int m = 1; m <= 2; ++m) {
        cm *= sr123;
        result += cm * 2.0 * eval_sensitivity(float(m) * opd, float(m) * phi);
    }
    // (out of gamut colors come out negative)
    return max(result, vec3(0.0));
}

// The reflectance head on that Schlick's approximation would take to reach f at this angle.
vec3 schlick_to_f0(vec3 f, float f90, float c) {
    float x = clamp(1.0 - c, 0.0, 1.0);
    float x5 = clamp(x * x * x * x * x, 0.0, 0.9999);
    return (f - vec3(f90) * x5) / (1.0 - x5);
}

vec4 g_base;      // rgb = albedo, a = alpha
float g_rough;
float g_metal;
vec3 g_f0;        // a standard material's reflectance head on, and at a grazing angle
float g_f90;      //   (from ior, specular_color and specular_intensity; set after the maps)
// A light with a size (light_toward), set by the light loops before direct(): g_sized, the
// direction of its highlight, and how much its shape widens one (x along a tube, y round a
// ball: light_norm)
bool g_sized = false;
vec3 g_spec_l;
vec2 g_spec_k;
float g_nol = -1.0; // the light's cosine over the squared distance, when it is not a point's
                  //   (a tube's, integrated along it): the radiance is then without its falloff
// three's MeshPhysicalMaterial layers, set after the maps (0: off): a clear coat over the
// surface (its own normal and roughness), a sheen of fibers on it, a thin film's rainbow
float g_clearcoat;
float g_clearcoat_rough;
vec3 g_clearcoat_n;
vec3 g_clearcoat_direct;  // what the lights' highlights on the coat add up to
vec3 g_sheen_color;
float g_sheen_rough;
vec3 g_sheen_light;       // the sheen's light, from the lights and all around
float g_iridescence;
vec3 g_iridescence_fresnel; // the film's reflectance toward the viewer
vec3 g_iridescence_f0;      //   and head on
float g_anisotropy;         // how much the highlights stretch along g_anisotropy_t (0..1)
vec3 g_anisotropy_t;        // the direction they stretch along, in the tangent frame's scale
vec3 g_anisotropy_b;        //   and across it
vec3 g_dpdx;      // how the world position changes from one pixel to the next (taken where
vec3 g_dpdy;      // the flow is uniform, for the shadows' tap counts)
// Parallax occlusion mapping (Material.parallax_map). The material's maps are read at g_uv,
// with the derivatives of v_uv (the offset jumps where the height map steps: its own
// derivatives would pick a blurry mip level there).
vec2 g_uv;
vec2 g_duvdx;
vec2 g_duvdy;
vec2 g_duv1dx;    // and v_uv1's
vec2 g_duv1dy;
// how the uv changes per meter, along the surface (the cotangent frame, as in perturb_normal)
vec3 g_parallax_gu;
vec3 g_parallax_gv;
float g_parallax_h;     // the height map where g_uv lands (0..1)
float g_parallax_depth; // the depth after the distance fade (m)
// Hex tiling (Material.hex_tiling): the three patches nearest the pixel, as offsets of g_uv
// where each reads the texture, and how much of each it takes (see hex_setup)
bool g_hex = false;
vec2 g_hex_off[3];
vec3 g_hex_w;

float toon_gradient(float ndl);

// three's packDepthToRGBA, packDepthToRGB and packDepthToRG, or 1 - z as a gray (packing 0).
vec4 pack_depth(float v, int packing, float alpha) {
    if (packing == 0) return vec4(vec3(1.0 - v), alpha);
    if (v <= 0.0) return packing == 1 ? vec4(0.0) : vec4(0.0, 0.0, 0.0, 1.0);
    if (v >= 1.0) return packing == 3 ? vec4(1.0, 1.0, 0.0, 1.0) : vec4(1.0);
    float vuf;
    if (packing == 3) {
        float gf = modf(v * 256.0, vuf);
        return vec4(vuf / 255.0, gf, 0.0, 1.0);
    }
    if (packing == 2) {
        float bf = modf(v * 65536.0, vuf);
        float gf = modf(vuf / 256.0, vuf);
        return vec4(vuf / 255.0, gf * 256.0 / 255.0, bf, 1.0);
    }
    float af = modf(v * 16777216.0, vuf);
    float bf = modf(vuf / 256.0, vuf);
    float gf = modf(vuf / 256.0, vuf);
    return vec4(vuf / 255.0, gf * 256.0 / 255.0, bf * 256.0 / 255.0, af);
}

// A light with a size, seen in reflections (Karis, "Real Shading in Unreal Engine 4", 2013,
// the representative point): sets g_spec_l, the direction toward the point of the light nearest
// the ray `r` reflected off the surface, for the highlight, and g_spec_k for light_norm. `lv`
// is from the surface to the light's centre, `seg` from its centre to one end (a tube), `radius`
// its thickness. A tube first takes the point of its segment nearest the ray (his eq. 19), then
// the ball of `radius` round that point the point of it nearest the ray (eq. 11).
void light_toward(vec3 r, vec3 lv, vec3 seg, float radius) {
    g_sized = true;
    g_spec_k = vec2(0.0);
    if (dot(seg, seg) > 0.0) {
        vec3 l0 = lv - seg;
        vec3 ld = 2.0 * seg;
        float rld = dot(r, ld);
        float t = (dot(r, l0) * rld - dot(l0, ld)) / max(dot(ld, ld) - rld * rld, 1e-8);
        lv = l0 + clamp(t, 0.0, 1.0) * ld;
        g_spec_k.x = length(seg) / (3.0 * max(length(lv), 1e-4));
    }
    if (radius > 0.0) {
        g_spec_k.y = radius / (3.0 * max(length(lv), 1e-4));
        // (the light's centre to the ray: Karis' eq. 11 with its sign put right)
        vec3 to_ray = dot(lv, r) * r - lv;
        lv += to_ray * clamp(radius / max(length(to_ray), 1e-6), 0.0, 1.0);
    }
    g_spec_l = normalize(lv);
}

// The light the highlight of light_toward's light keeps at roughness alpha `a`: the light stays
// as bright as the point light at its centre, and this takes away what the highlight gains by
// moving its source around on the light: a / a' along a tube (eq. 20, a widening in one
// direction only) times (a / a')^2 for its ball (eq. 14), a' = saturate(a + half the length,
// or the radius, / (3 distance)) (eq. 10).
float light_norm(float a) {
    float k = a / clamp(a + g_spec_k.y, 0.0, 1.0);
    return a / clamp(a + g_spec_k.x, 0.0, 1.0) * k * k;
}

// Direct light contribution, following three.js's (r155+) physically based light units.
// Returns diffuse in rgb; adds specular to `spec`. `l` is toward the light's centre (the
// diffuse light's direction); point and spot lights with a size (g_sized) take their
// highlights from light_toward.
vec3 direct(int mode, vec3 n, vec3 v, vec3 l, float dist, vec3 radiance, vec3 diffuse_color, inout vec3 spec) {
    bool foliage = (flags() & 2) != 0;
    float ndl = dot(n, l);
    // thin leaves and blades, lit from either side: on the side the light falls on, as any
    // surface, with a little of it wrapping round; from the other, what shines through them,
    // about half of it (a lamp by the eye lights only the faces it sees, and a blade turned
    // edge-on to it stays dark)
    float through = dot(n, v) * ndl < 0.0 ? 0.5 : 1.0;
    float nol = foliage ? through * (0.1 + 0.9 * abs(ndl)) : clamp(ndl, 0.0, 1.0);
    vec3 irradiance = (g_nol >= 0.0 ? g_nol : nol) * radiance;
    // the toon material: the gradient's step in place of n.l, and no highlights
    if (mode == 6) return toon_gradient(ndl) * radiance * diffuse_color * RECIPROCAL_PI;
    vec3 h = normalize(l + v);
    if (mode == 3) {
        if (g_clearcoat > 0.0) {
            // the coat's own highlight, off its own normal (f0 0.04: an IOR of 1.5)
            vec3 cn = g_clearcoat_n;
            float a = g_clearcoat_rough * g_clearcoat_rough;
            float cnl = clamp(dot(cn, l), 0.0, 1.0);
            float cnv = clamp(dot(cn, v), 1e-4, 1.0);
            // (a light with a size: the point of it the surface under the coat mirrors, as
            // bright as the coat's roughness keeps it)
            float cnorm = g_sized ? light_norm(a) : 1.0;
            vec3 cl = g_sized ? g_spec_l : l;
            vec3 ch = normalize(cl + v);
            float cnh = clamp(dot(cn, ch), 0.0, 1.0);
            float f = schlick(0.04, 1.0, clamp(dot(v, ch), 0.0, 1.0));
            float ccl = clamp(dot(cn, cl), 0.0, 1.0);
            g_clearcoat_direct += cnl * radiance * f * v_smith(cnv, ccl, a) * d_ggx(cnh, a) * cnorm;
        }
        if (g_sheen_rough > 0.0) {
            float snv = clamp(dot(n, v), 0.0, 1.0);
            float snl = clamp(dot(n, l), 0.0, 1.0);
            g_sheen_light += irradiance * g_sheen_color * d_charlie(g_sheen_rough, clamp(dot(n, h), 0.0, 1.0)) * v_neubelt(snv, snl);
            // what the fibers take is not left for the surface under them
            float albedo = max(sheen_albedo(n, v, g_sheen_rough), sheen_albedo(n, l, g_sheen_rough));
            irradiance *= 1.0 - max(g_sheen_color.r, max(g_sheen_color.g, g_sheen_color.b)) * albedo;
        }
    }
    vec3 result = irradiance * diffuse_color * RECIPROCAL_PI;
    if (mode == 1) return result;

    if (mode == 2) {
        float noh = clamp(dot(n, h), 0.0, 1.0);
        float voh = clamp(dot(v, h), 0.0, 1.0);
        float shininess = mat_params.w;
        float d = RECIPROCAL_PI * (shininess * 0.5 + 1.0) * pow(noh, shininess);
        spec += irradiance * f_schlick(specular.rgb, voh) * 0.25 * d;
    } else {
        float rough = max(g_rough, 0.0525);
        float a = rough * rough;
        // a light with a size: the highlight is the light's own shape, seen in the surface
        // (light_toward), as bright in all as the point light's
        float norm = 1.0;
        vec3 sl = l;
        float snl = nol;
        if (g_sized) {
            sl = g_spec_l;
            norm = light_norm(a);
            h = normalize(sl + v);
            snl = clamp(dot(n, sl), 0.0, 1.0);
        }
        float noh = clamp(dot(n, h), 0.0, 1.0);
        float voh = clamp(dot(v, h), 0.0, 1.0);
        float nov = clamp(abs(dot(n, v)), 1e-4, 1.0);
        vec3 fresnel = g_f0 + (vec3(g_f90) - g_f0) * pow(1.0 - voh, 5.0);
        if (g_iridescence > 0.0) fresnel = mix(fresnel, g_iridescence_fresnel, g_iridescence);
        float vis = v_smith(nov, snl, a);
        float d = d_ggx(noh, a);
        if (g_anisotropy > 0.0) {
            // rougher along g_anisotropy_t, by as much as it is anisotropic (three's alphaT)
            float at = max(mix(a, 1.0, g_anisotropy * g_anisotropy), a);
            vec3 t = g_anisotropy_t;
            vec3 b = g_anisotropy_b;
            vis = v_smith_aniso(at, a, dot(t, v), dot(b, v), dot(t, sl), dot(b, sl), nov, snl);
            d = d_ggx_aniso(at, a, noh, dot(t, h), dot(b, h));
        }
        spec += irradiance * fresnel * vis * d * norm;
    }
    return result;
}

// RectAreaLight, as three's (lights_physical_pars_fragment: LTC_Uv, LTC_Evaluate and
// RE_Direct_RectArea_Physical), with linearly transformed cosines (Heitz, Dupuy, Hill and
// Neubelt 2016). The lights are in point_lights after the point lights, three vec4s each:
// pos = centre + half width's x, color = color * intensity + half width's y, params = half
// width's z + half height; and after them the two LTC tables (clusters.c3's ltc_entries),
// a texel an entry (pos = LTC_1's, color = LTC_2's), read here with the bilinear filter
// three's textures have.

void ltc_tables(int base, vec3 n, vec3 v, float rough, out vec4 t1, out vec4 t2) {
    // three's LTC_Uv: roughness across, sqrt(1 - n.v) down, at the texels' centres
    vec2 uv = vec2(rough, sqrt(1.0 - clamp(dot(n, v), 0.0, 1.0))) * (63.0 / 64.0) + 0.5 / 64.0;
    vec2 p = uv * 64.0 - 0.5;
    vec2 f = fract(p);
    ivec2 i0 = clamp(ivec2(floor(p)), ivec2(0), ivec2(63));
    ivec2 i1 = min(i0 + 1, ivec2(63));
    int a = base + i0.y * 64 + i0.x;
    int b = base + i0.y * 64 + i1.x;
    int c = base + i1.y * 64 + i0.x;
    int d = base + i1.y * 64 + i1.x;
    t1 = mix(mix(points[a].pos, points[b].pos, f.x), mix(points[c].pos, points[d].pos, f.x), f.y);
    t2 = mix(mix(points[a].color, points[b].color, f.x), mix(points[c].color, points[d].color, f.x), f.y);
}

vec3 ltc_edge(vec3 v1, vec3 v2) {
    float x = dot(v1, v2);
    float y = abs(x);
    float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float b = 3.4175940 + (4.1616724 + y) * y;
    float k = a / b;
    float theta_sintheta = x > 0.0 ? k : 0.5 * inversesqrt(max(1.0 - x * x, 1e-7)) - k;
    return cross(v1, v2) * theta_sintheta;
}

// How much of the cosine lobe, transformed by m_inv, the rectangle (corners q, from the
// surface) covers: three's LTC_Evaluate.
float ltc_evaluate(vec3 n, vec3 v, mat3 m_inv, vec3 q[4]) {
    vec3 t1 = normalize(v - n * dot(v, n));
    vec3 t2 = -cross(n, t1);
    mat3 m = m_inv * transpose(mat3(t1, t2, n));
    vec3 c0 = normalize(m * q[0]);
    vec3 c1 = normalize(m * q[1]);
    vec3 c2 = normalize(m * q[2]);
    vec3 c3 = normalize(m * q[3]);
    vec3 f = ltc_edge(c0, c1) + ltc_edge(c1, c2) + ltc_edge(c2, c3) + ltc_edge(c3, c0);
    // three's LTC_ClippedSphereFormFactor
    float l = length(f);
    return max((l * l + f.z) / (l + 1.0), 0.0);
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

#ifdef VSM
// three's VSMShadowMap: how lit depth `z` is where the map's depths have mean `m.x` and
// standard deviation `m.y` (Chebyshev's upper bound, its low end cut off against light
// bleeding, as three's).
float vsm_lookup(vec2 m, float z) {
    float hard = step(z, m.x);
    if (hard == 1.0) return 1.0;
    float variance = max(m.y * m.y, 1e-7);
    float d = z - m.x;
    float p = variance / (variance + d * d);
    p = clamp((p - 0.3) / 0.65, 0.0, 1.0);
    return max(hard, p);
}

// The moments of the spot light's map (or its tile of the atlas) at `uv`.
vec2 vsm_moments(bool tiled, vec4 rect, float texel, vec2 uv) {
    if (tiled) {
        uv = rect.xy + clamp(uv, vec2(texel), vec2(1.0 - texel)) * rect.zw;
        return textureLod(sampler2D(vsm_atlas, vsm_smp), uv, 0.0).rg;
    }
    return textureLod(sampler2D(vsm_map, vsm_smp), uv, 0.0).rg;
}
#endif

// 1 where the light's map has nothing nearer than depth `z` at `uv`.
float shadow_tap(bool tiled, vec4 rect, float texel, vec2 uv, float z) {
#ifdef VSM
    // (with VSM: nearer than the mean depth)
    return step(z, vsm_moments(tiled, rect, texel, uv).x);
#else
    if (tiled) {
        // stay half a texel inside the tile: the neighbours belong to other lights
        uv = rect.xy + clamp(uv, vec2(texel), vec2(1.0 - texel)) * rect.zw;
        return texture(sampler2DShadow(shadow_atlas, shadow_smp), vec3(uv, z));
    }
    return texture(sampler2DShadow(shadow_map, shadow_smp), vec3(uv, z));
#endif
}

// The soft shadows' taps: `base` of them (as many as Renderer.shadow_sample_scale 1 takes),
// times that scale.
const int SHADOW_MAX_TAPS = 40;
float shadow_sample_scale() { return quality_params.x > 0.0 ? quality_params.x : 1.0; }
int shadow_taps(float base) { return clamp(int(base * shadow_sample_scale() + 0.5), 4, SHADOW_MAX_TAPS); }

// Shadow of the light at `light_pos` (a spot light, or one face of a point light's) whose
// map is `si` in spot_shadows. A spot light whose map is redrawn every frame has shadow_map to
// itself; the other maps share shadow_atlas, a tile each.
float light_shadow_of(vec3 light_pos, int si, vec3 n) {
    vec4 params = spot_shadow_data[si].params;
    vec3 to_light = light_pos - v_world_pos;
    float dist = length(to_light);
    // push the lookup off the surface, more at grazing angles and far away where texels are big
    float slope = 1.0 - clamp(dot(n, to_light / dist), 0.0, 1.0);
    vec3 p = v_world_pos + n * params.y * dist * (0.4 + slope);
    // (the matrix takes the point from the light, see vs_params)
    mat4 m = spot_shadow_data[si].mat;
    p -= light_pos;
    vec4 sp = m * vec4(p, 1.0);
    if (sp.w <= 0.0) return 1.0;
    vec3 c = sp.xyz / sp.w;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0 || c.z > 1.0) return 1.0;
#ifdef VSM
    // one lookup: the map is blurred already (params.w is the bias)
    return vsm_lookup(vsm_moments(params.x > 1.5, spot_shadow_data[si].rect, params.z, c.xy), c.z + params.w);
#endif
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
    int most = shadow_taps(8.0);
    if (soft.x > 0.0) {
        // A source with a size: the edge of a shadow is as wide as the source looks from there
        // past the thing that casts it. Sharp where the two touch, wide far behind. How far the
        // caster is comes from asking the map at a few depths, nearer each time.
        float dr = shadow_distance(c.z, soft);
        float per_m = 1.0 / (abs(soft.w) * dr);
        float search = clamp(soft.x * 1.6 * per_m, r, 0.04);
        most = shadow_taps(14.0);
        // the widest edge the source could make here is under a pixel: no need to look
        if (search > pixel) {
            float found = 0.0;
            float nearest = 0.0;
            // the middle and a ring round it
            int looks = clamp(int(7.0 * shadow_sample_scale() + 0.5), 4, SHADOW_MAX_TAPS);
            for (int k = 0; k < SHADOW_MAX_TAPS; k++) {
                if (k >= looks) break;
                float a = float(k) * 6.2831853 / float(looks - 1);
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
    for (int k = 0; k < SHADOW_MAX_TAPS; k++) {
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
    return mix(1.0, light_shadow_of(spots[s].pos.xyz, si, n), spots[s].shadow.y);
}

// Shadow of a point light: the face of its cube of maps that the point is in.
float point_shadow(gpu_point_light pl, vec3 n) {
    int si = int(pl.shadow.x);
    if (si < 0 || (flags() & 4) == 0) return 1.0;
    vec3 d = v_world_pos - pl.pos.xyz;
    vec3 a = abs(d);
    int face = a.x >= a.y && a.x >= a.z ? (d.x >= 0.0 ? 0 : 1) : a.y >= a.z ? (d.y >= 0.0 ? 2 : 3) : (d.z >= 0.0 ? 4 : 5);
    return mix(1.0, light_shadow_of(pl.pos.xyz, si + face, n), pl.shadow.y);
}

// 1 where cascade `i` of the sun's map has nothing nearer than depth `z` at `uv` (0..1 in its tile).
float sun_tap(int i, vec2 uv, float z) {
    vec4 rect = dir_cascade_rect[i];
    // stay half a texel inside the tile: the neighbours belong to other cascades
    vec2 t = dir_shadow_params.z / rect.zw;
    uv = rect.xy + clamp(uv, t * 0.5, 1.0 - t * 0.5) * rect.zw;
#ifdef VSM
    return vsm_lookup(textureLod(sampler2D(vsm_sun, vsm_smp), uv, 0.0).rg, z + shadow_ext.x);
#else
    return texture(sampler2DShadow(sun_shadow, shadow_smp), vec3(uv, z));
#endif
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
#ifdef VSM
    // one lookup: each cascade's tile is blurred already
    float sum = sun_tap(i, c.xy, c.z) * 10.0;
#else
    float angle = fract(sin(dot(gl_FragCoord.xy + ssr_params.z * vec2(5.0, 3.0), vec2(12.9898, 78.233))) * 43758.5453) * 6.2831;
    float ca = cos(angle);
    float sa = sin(angle);
    float texel = params.z / dir_cascade_rect[i].z;
    float r = texel * max(params.w / scale, min(params.w, 1.0));
    int taps = shadow_taps(10.0);
    float sum = 0.0;
    for (int k = 0; k < SHADOW_MAX_TAPS; k++) {
        if (k >= taps) break;
        float a = float(k) * 2.39996;
        vec2 o = vec2(cos(a), sin(a)) * (sqrt((float(k) + 0.5) / float(taps)) * r);
        sum += sun_tap(i, c.xy + vec2(o.x * ca - o.y * sa, o.x * sa + o.y * ca), c.z);
    }
    sum *= 10.0 / float(taps);
#endif
    if (dir_cascade_params.w < 0.5) return sum / 10.0;
    // one box: nothing is shadowed outside of it, fade towards its edge
    vec2 edge = min(c.xy, 1.0 - c.xy);
    return mix(1.0, sum / 10.0, clamp(min(edge.x, edge.y) * 10.0, 0.0, 1.0));
}

// Shadow of a directional light besides the sun, whose box is tile `si` of shadow_atlas
// (spot_shadows): as dir_shadow_in, without cascades; nothing is shadowed outside of it.
float dir_tile_shadow(int si, vec3 n, vec3 l) {
    vec4 params = spot_shadow_data[si].params;
    vec4 rect = spot_shadow_data[si].rect;
    float slope = 1.0 - clamp(dot(n, l), 0.0, 1.0);
    vec3 c = (spot_shadow_data[si].mat * vec4(v_world_pos + n * params.y * (0.4 + slope), 1.0)).xyz;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0 || c.z > 1.0 || c.z < 0.0) return 1.0;
#ifdef VSM
    // one lookup: the tile is blurred already (params.w is the bias)
    float sum = vsm_lookup(vsm_moments(true, rect, params.z, c.xy), c.z + params.w) * 10.0;
#else
    float angle = fract(sin(dot(gl_FragCoord.xy + ssr_params.z * vec2(5.0, 3.0), vec2(12.9898, 78.233))) * 43758.5453) * 6.2831;
    float ca = cos(angle);
    float sa = sin(angle);
    float r = params.z * params.w;
    int taps = shadow_taps(10.0);
    float sum = 0.0;
    for (int k = 0; k < SHADOW_MAX_TAPS; k++) {
        if (k >= taps) break;
        float a = float(k) * 2.39996;
        vec2 o = vec2(cos(a), sin(a)) * (sqrt((float(k) + 0.5) / float(taps)) * r);
        sum += shadow_tap(true, rect, params.z, c.xy + vec2(o.x * ca - o.y * sa, o.x * sa + o.y * ca), c.z);
    }
    sum *= 10.0 / float(taps);
#endif
    // fade out towards the box's edge
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
#ifdef VSM
            // (a hard test against the mean, as the PCF map's one tap)
            seen = seen < 1.0 ? 0.0 : 1.0;
#endif
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

// Light from the scene's irradiance volumes for a surface facing `n`, already times `cover`:
// how much of the point they cover, 1 inside a volume, 0 outside them all. Each volume covers
// its box and fades out over half a cell outside it, in front of the ones before it. Each
// probe holds the light arriving from the six axis directions, side by side along x in the
// volume's part of the texture.
vec3 gi_irradiance(vec3 n, out float cover) {
    cover = 0.0;
    vec3 e = vec3(0.0);
    vec3 slab = vec3(n.x >= 0.0 ? 0.0 : 1.0, n.y >= 0.0 ? 2.0 : 3.0, n.z >= 0.0 ? 4.0 : 5.0) / 6.0;
    vec3 w = n * n;
    for (int i = 0; i < 4; i++) {
        if (gi_min[i].w < 0.5) break;
        vec3 res = gi_dims[i].xyz;
        vec3 p = (v_world_pos + n * gi_dims[i].w - gi_min[i].xyz) * gi_inv_size[i].xyz;
        vec3 edge = min(p, 1.0 - p) * res;
        float fade = clamp(1.0 + min(edge.x, min(edge.y, edge.z)) * 2.0, 0.0, 1.0);
        if (fade <= 0.0) continue;
        // stay inside the volume and a direction's slab
        p = clamp(p, 0.5 / res, 1.0 - 0.5 / res);
        vec3 u = gi_atlas[i].x + (p.x / 6.0 + slab) * gi_atlas[i].y;
        vec2 yz = p.yz * gi_atlas[i].zw;
        vec3 l = textureLod(sampler3D(tex_gi, gi_smp), vec3(u.x, yz), 0.0).rgb * w.x
               + textureLod(sampler3D(tex_gi, gi_smp), vec3(u.y, yz), 0.0).rgb * w.y
               + textureLod(sampler3D(tex_gi, gi_smp), vec3(u.z, yz), 0.0).rgb * w.z;
        e = mix(e, l * gi_inv_size[i].w, fade);
        cover = mix(cover, 1.0, fade);
    }
    return e;
}

// Light from the scene's radiance cascades for a surface facing `n`; `fade` is 1 inside their
// box, 0 outside it, between the two over its outermost half cell.
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

// how many steps a reflected ray takes at most (Renderer.ssr_steps), and how far one goes on
// the screen at most (Renderer.ssr_max_step, in pixels of the reflection source)
const int SSR_MAX_STEPS = 128;
int ssr_steps() { return quality_params.y > 0.0 ? min(int(quality_params.y), SSR_MAX_STEPS) : 40; }
float ssr_max_step_px() { return quality_params.z > 0.0 ? quality_params.z : 6.0; }

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
    int steps = ssr_steps();
    float max_step = ssr_max_step_px();
    for (int i = 0; i < SSR_MAX_STEPS; i++) {
        if (i >= steps) break;
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
        if (next.z > 0.05 && px > max_step) step_len *= max_step / px;
        t += step_len;
        if (t > ssr_thickness.y) break;
    }
    return vec3(0.0);
}

// A map slot's uv (see uv_rel), `shift` away from g_uv, and how it changes from one pixel to
// the next.
vec2 slot_uv(int slot, vec2 shift) {
    vec3 p = vec3(uv_rel[slot * 2].w > 0.5 ? v_uv1 : g_uv + shift, 1.0);
    return vec2(dot(uv_rel[slot * 2].xyz, p), dot(uv_rel[slot * 2 + 1].xyz, p));
}

vec2 slot_d(int slot, vec2 d) { return vec2(dot(uv_rel[slot * 2].xy, d), dot(uv_rel[slot * 2 + 1].xy, d)); }
vec2 slot_dx(int slot) { return slot_d(slot, uv_rel[slot * 2].w > 0.5 ? g_duv1dx : g_duvdx); }
vec2 slot_dy(int slot) { return slot_d(slot, uv_rel[slot * 2].w > 0.5 ? g_duv1dy : g_duvdy); }

#define slot_texel_at(tex, slot, shift) textureGrad(sampler2D(tex, tex_smp), slot_uv(slot, shift), slot_dx(slot), slot_dy(slot))
// (hex tiled: the three patches, with the color map's weights; not a map in the second uv set,
// a light map's, say, which is laid out once over the surface)
#define slot_texel(tex, slot) ((g_hex && uv_rel[(slot) * 2].w < 0.5) \
    ? g_hex_w.x * slot_texel_at(tex, slot, g_hex_off[0]) + g_hex_w.y * slot_texel_at(tex, slot, g_hex_off[1]) \
        + g_hex_w.z * slot_texel_at(tex, slot, g_hex_off[2]) \
    : slot_texel_at(tex, slot, vec2(0.0)))

// Hex tiling (Material.hex_tiling; Mikkelsen, "Practical Real-Time Hex-Tiling", JCGT 11(2),
// 2022): the uv plane cut into hexagons `size` across, each reading the texture from its own
// random place, so that a texture repeated over a wall doesn't show its repeats. The hexagons'
// middles make a grid of triangles; a pixel takes the three corners of the one it is in,
// weighted by how near it is to each, sharpened (the 7th power) so that a patch is mostly
// itself and only blends into the next near their seam. Moved, not turned: what runs down the
// texture still runs down, and the normal map's directions stay as they are.
// (Dave Hoskins' hash22: no sin, whose precision goes far from 0, as a uv in world meters does)
vec2 hex_hash(vec2 v) {
    vec3 p = fract(v.xyx * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yzx + 33.33);
    return fract((p.xx + p.yz) * p.zy);
}

void hex_setup(vec2 uv, float size) {
    vec2 st = uv / size;
    // the grid skewed so that its triangles are halves of unit squares
    vec2 skewed = vec2(st.x - 0.57735027 * st.y, 1.15470054 * st.y);
    vec2 cell = floor(skewed);
    vec3 f = vec3(fract(skewed), 0.0);
    f.z = 1.0 - f.x - f.y;
    // the square's upper triangle (1) or lower (0), and the barycentric weights in it
    float s = step(0.0, -f.z);
    float s2 = 2.0 * s - 1.0;
    vec3 w = vec3(-f.z * s2, s - f.y * s2, s - f.x * s2);
    g_hex_off[0] = hex_hash(cell + vec2(s, s));
    g_hex_off[1] = hex_hash(cell + vec2(s, 1.0 - s));
    g_hex_off[2] = hex_hash(cell + vec2(1.0 - s, s));
    vec3 w3 = w * w * w;
    w = w3 * w3 * w;
    g_hex_w = w / (w.x + w.y + w.z);
    g_hex = true;
}

// The color map, hex tiled: where two patches meet, the brighter of them shows more (the
// paper's luminance weighting, 0.6 of it), so the seam follows the texture's own dark marks
// instead of fading one into the other; the other maps take the same weights.
vec4 hex_color(vec2 uv, vec2 dx, vec2 dy) {
    vec4 c0 = textureGrad(sampler2D(tex_map, tex_smp), uv + g_hex_off[0], dx, dy);
    vec4 c1 = textureGrad(sampler2D(tex_map, tex_smp), uv + g_hex_off[1], dx, dy);
    vec4 c2 = textureGrad(sampler2D(tex_map, tex_smp), uv + g_hex_off[2], dx, dy);
    const vec3 luma = vec3(0.2126, 0.7152, 0.0722);
    vec3 w = g_hex_w * mix(vec3(1.0), vec3(dot(c0.rgb, luma), dot(c1.rgb, luma), dot(c2.rgb, luma)), 0.6);
    g_hex_w = w / max(w.x + w.y + w.z, 1e-6);
    return g_hex_w.x * c0 + g_hex_w.y * c1 + g_hex_w.z * c2;
}

// The surface's tangent frame (three's tbn) around the normal n (facing the viewer): the
// tangent t, along which the uv's u grows, and the bitangent b. From the vertex tangents where
// the geometry has them (b = cross(normal, tangent) * w, glTF's), else from how the position
// and a map's uv change from one pixel to the next (duv1, duv2: that map's; three's
// getTangentFrame, the cotangent frame: t and b scaled together, not to a length of 1 each).
void tangent_frame(vec3 n, vec2 duv1, vec2 duv2, out vec3 t, out vec3 b) {
    // a back face (its normal turned to the camera): the frame turned over too, as three's
    // tbn *= faceDirection
    bool back = dot(n, v_normal) < 0.0;
#ifdef TANGENT
    // (a vertex without one, all zeros: the frame from the uv instead)
    if (dot(v_tangent.xyz, v_tangent.xyz) > 0.0) {
        vec3 vn = normalize(v_normal);
        t = normalize(v_tangent.xyz);
        b = normalize(cross(vn, t) * v_tangent.w);
        if (back) {
            t = -t;
            b = -b;
        }
        return;
    }
#endif
    vec3 dp2perp = cross(g_dpdy, n);
    vec3 dp1perp = cross(n, g_dpdx);
    t = dp2perp * duv1.x + dp1perp * duv2.x;
    b = dp2perp * duv1.y + dp1perp * duv2.y;
    // (three's formula takes the screen's y up, as GL's dFdy; Metal's and D3D's go down, which
    // turns the frame around: its sign, the determinant's, keeps it the same on every backend)
    float invmax = inversesqrt(max(max(dot(t, t), dot(b, b)), 1e-12));
    if (dot(g_dpdx, dp2perp) < 0.0) invmax = -invmax;
    if (back) invmax = -invmax;
    t *= invmax;
    b *= invmax;
}

// Tangent-space normal mapping. m: the map's texel, in -1..1; duv1, duv2: how the map's uv
// changes from one pixel to the next.
vec3 perturb_with(vec3 n, vec3 m, vec2 duv1, vec2 duv2) {
    vec3 t;
    vec3 b;
    tangent_frame(n, duv1, duv2, t, b);
    return normalize(t * m.x + b * m.y + n * m.z);
}

vec3 perturb_normal(vec3 n) {
    vec3 m = slot_texel(tex_normal, 0).xyz * 2.0 - 1.0;  // (parallax: g_uv)
    m.xy *= tangent_params.xy;
    return perturb_with(n, m, slot_dx(0), slot_dy(0));
}

// The material's map `role` (MAP_*) at g_uv, from the slot it is in; white without one.
// (macros: a constant index into map_slots, which can't be indexed by a variable here)
#define has_map(role) (map_slots[(role) / 4][(role) % 4] >= 0.0)
#define material_map(role) ext_texel(int(map_slots[(role) / 4][(role) % 4]))
// (its slot in uv_rel)
#define map_uv_slot(role) (int(map_slots[(role) / 4][(role) % 4]) + 2)

vec4 ext_texel(int slot) {
    if (slot == 0) return slot_texel(tex_ext0, 2);
    if (slot == 1) return slot_texel(tex_ext1, 3);
    if (slot == 2) return slot_texel(tex_ext2, 4);
    if (slot == 3) return slot_texel(tex_ext3, 5);
    if (slot == 4) return slot_texel(tex_ext4, 6);
    if (slot == 5) return slot_texel(tex_ext5, 7);
    return vec4(1.0);
}

// Material.custom_map at uv, for the application's hooks: a uv of the hook's own, and how it
// changes from one pixel to the next (which picks the mip level); white without one.
// One of the six shared map slots (tex_ext0..5), at its top level (white for none).
vec4 ext_slot_texel(int slot, vec2 uv) {
    if (slot == 0) return textureLod(sampler2D(tex_ext0, tex_smp), uv, 0.0);
    if (slot == 1) return textureLod(sampler2D(tex_ext1, tex_smp), uv, 0.0);
    if (slot == 2) return textureLod(sampler2D(tex_ext2, tex_smp), uv, 0.0);
    if (slot == 3) return textureLod(sampler2D(tex_ext3, tex_smp), uv, 0.0);
    if (slot == 4) return textureLod(sampler2D(tex_ext4, tex_smp), uv, 0.0);
    if (slot == 5) return textureLod(sampler2D(tex_ext5, tex_smp), uv, 0.0);
    return vec4(1.0);
}

// One of the six slots at uv, at the mip level that dx and dy (uv per pixel) pick.
vec4 ext_slot_grad(int slot, vec2 uv, vec2 dx, vec2 dy) {
    if (slot == 0) return textureGrad(sampler2D(tex_ext0, tex_smp), uv, dx, dy);
    if (slot == 1) return textureGrad(sampler2D(tex_ext1, tex_smp), uv, dx, dy);
    if (slot == 2) return textureGrad(sampler2D(tex_ext2, tex_smp), uv, dx, dy);
    if (slot == 3) return textureGrad(sampler2D(tex_ext3, tex_smp), uv, dx, dy);
    if (slot == 4) return textureGrad(sampler2D(tex_ext4, tex_smp), uv, dx, dy);
    if (slot == 5) return textureGrad(sampler2D(tex_ext5, tex_smp), uv, dx, dy);
    return vec4(1.0);
}

// One of the six slots' texel nearest uv (three's NearestFilter), at the top level.
vec4 ext_slot_nearest(int slot, vec2 uv) {
    if (slot == 0) return texelFetch(sampler2D(tex_ext0, tex_smp), ivec2(fract(uv) * vec2(textureSize(sampler2D(tex_ext0, tex_smp), 0))), 0);
    if (slot == 1) return texelFetch(sampler2D(tex_ext1, tex_smp), ivec2(fract(uv) * vec2(textureSize(sampler2D(tex_ext1, tex_smp), 0))), 0);
    if (slot == 2) return texelFetch(sampler2D(tex_ext2, tex_smp), ivec2(fract(uv) * vec2(textureSize(sampler2D(tex_ext2, tex_smp), 0))), 0);
    if (slot == 3) return texelFetch(sampler2D(tex_ext3, tex_smp), ivec2(fract(uv) * vec2(textureSize(sampler2D(tex_ext3, tex_smp), 0))), 0);
    if (slot == 4) return texelFetch(sampler2D(tex_ext4, tex_smp), ivec2(fract(uv) * vec2(textureSize(sampler2D(tex_ext4, tex_smp), 0))), 0);
    if (slot == 5) return texelFetch(sampler2D(tex_ext5, tex_smp), ivec2(fract(uv) * vec2(textureSize(sampler2D(tex_ext5, tex_smp), 0))), 0);
    return vec4(1.0);
}

// A world direction in the camera's view space (x right, y up, z toward the viewer).
vec3 to_view(vec3 d) {
    vec3 back = cross(camera_right.xyz, camera_up.xyz);
    return vec3(dot(d, camera_right.xyz), dot(d, camera_up.xyz), dot(d, back));
}

// MeshToonMaterial's light: the gradient map (or two steps without one) by where the light
// falls, in place of n.l (three's getGradientIrradiance)
float toon_gradient(float ndl) {
    float x = ndl * 0.5 + 0.5;
    int role_slot = int(map_slots[MAP_GRADIENT / 4][MAP_GRADIENT % 4]);
    if (role_slot >= 0) {
        vec2 uv = vec2(clamp(x, 0.0, 0.999), 0.0);
        return surface_params.y > 0.5 ? ext_slot_nearest(role_slot, uv).r : ext_slot_texel(role_slot, uv).r;
    }
    float fw = fwidth(x) * 0.5;
    return mix(0.7, 1.0, smoothstep(0.7 - fw, 0.7 + fw, x));
}

vec4 custom_map_texel(vec2 uv, vec2 dx, vec2 dy) {
    int slot = int(map_slots[MAP_CUSTOM / 4][MAP_CUSTOM % 4]);
    if (slot == 0) return textureGrad(sampler2D(tex_ext0, tex_smp), uv, dx, dy);
    if (slot == 1) return textureGrad(sampler2D(tex_ext1, tex_smp), uv, dx, dy);
    if (slot == 2) return textureGrad(sampler2D(tex_ext2, tex_smp), uv, dx, dy);
    if (slot == 3) return textureGrad(sampler2D(tex_ext3, tex_smp), uv, dx, dy);
    if (slot == 4) return textureGrad(sampler2D(tex_ext4, tex_smp), uv, dx, dy);
    if (slot == 5) return textureGrad(sampler2D(tex_ext5, tex_smp), uv, dx, dy);
    return vec4(1.0);
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

// The light the scene's environment puts on a surface facing n, but for its sun (env_sun):
// env_sh's 9 spherical harmonics, in the order of texture.c3's sh_basis.
vec3 env_irradiance(vec3 n) {
    vec3 e = env_sh[0].rgb + env_sh[1].rgb * n.y + env_sh[2].rgb * n.z + env_sh[3].rgb * n.x
           + env_sh[4].rgb * (n.x * n.y) + env_sh[5].rgb * (n.y * n.z) + env_sh[6].rgb * (3.0 * n.z * n.z - 1.0)
           + env_sh[7].rgb * (n.x * n.z) + env_sh[8].rgb * (n.x * n.x - n.y * n.y);
    return max(e, vec3(0.0));
}

vec3 env_sun_dir() { return vec3(env_sh[0].w, env_sh[1].w, env_sh[2].w); }

// The environment's sun on a surface facing n (EnvLight.irradiance is this and env_irradiance).
vec3 env_sun(vec3 n) { return vec3(env_sh[3].w, env_sh[4].w, env_sh[5].w) * max(dot(n, env_sun_dir()), 0.0); }

// The map being read: the material's own (g_env_layer < 0), or a layer of the scene's probes.
float g_env_layer;
vec4 g_env_pos;      // as env_pos
vec3 g_env_min;
vec3 g_env_max;
float g_env_range;   // as env_params.w

// The material's map in direction d, sharp but read at the mip level its footprint on the
// screen wants (as a cube map with mip levels is read): three's envMap on a basic, lambert or
// phong material. (The u derivative wraps at the seam behind.)
vec3 env_classic(vec3 d) {
    vec2 uv = env_uv(d);
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);
    dx.x -= floor(dx.x + 0.5);
    dy.x -= floor(dy.x + 0.5);
    return textureGrad(sampler2D(tex_env, env_smp), uv, dx, dy).rgb;
}

vec4 env_texel(vec2 uv, float lod) {
    if (g_env_layer < 0.0) return textureLod(sampler2D(tex_env, env_smp), uv, lod);
    return textureLod(sampler2DArray(tex_probes, env_smp), vec3(uv, g_env_layer), lod);
}

// The material's own map at its sharpest, as three shows an equirectangular background: drawn
// into a cube map of twice its texels across, which is read bilinearly. Here, four bilinear
// reads at the corners of a grid twice as fine, mixed bilinearly; one bilinear read of the map
// shows its texels as steps where it is magnified.
vec4 env_texel_smooth(vec2 uv) {
    vec2 size = vec2(textureSize(sampler2D(tex_env, env_smp), 0)) * 2.0;
    vec2 p = uv * size - 0.5;
    vec2 i = floor(p);
    vec2 f = p - i;
    vec2 a = (i + 0.5) / size;
    vec2 b = (i + 1.5) / size;
    // (around the horizon, the map wraps)
    vec4 c00 = textureLod(sampler2D(tex_env, env_smp), vec2(fract(a.x), a.y), 0.0);
    vec4 c10 = textureLod(sampler2D(tex_env, env_smp), vec2(fract(b.x), a.y), 0.0);
    vec4 c01 = textureLod(sampler2D(tex_env, env_smp), vec2(fract(a.x), b.y), 0.0);
    vec4 c11 = textureLod(sampler2D(tex_env, env_smp), vec2(fract(b.x), b.y), 0.0);
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// How far the point `q` (relative to where the map was captured) lies behind what the map saw
// in its direction.
float env_behind(vec3 q) {
    float len = length(q);
    return len - env_texel(env_uv(q / len), 0.0).a * g_env_range;
}

// How widely a rough surface's reflection is blurred (radians, a gaussian's sigma), as three's
// PMREM blurs it (its roughnessToMip and EXTRA_LOD_SIGMA): at roughness 0.21 and under, as
// its cube map's mip levels of (1.16 * rough)^-2 texels across; above, its extra blurred
// levels.
float env_blur(float rough) {
    if (rough >= 0.8) return mix(0.526, 0.582, (rough - 0.8) / 0.2);
    if (rough >= 0.4) return mix(0.215, 0.526, (rough - 0.4) / 0.4);
    if (rough >= 0.305) return mix(0.125, 0.215, (rough - 0.305) / 0.095);
    if (rough >= 0.21) return mix(0.06, 0.125, (rough - 0.21) / 0.095);
    return 1.37 * rough * rough;
}

// The mip level of an equirectangular map `width` texels around whose texels are about
// `sigma` radians wide (a box of 2^lod texels, filtered, about the gaussian's width).
float env_lod(float sigma, float width) { return max(log2(max(sigma * width / PI, 1e-6)), 0.0); }

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
    float width = g_env_layer < 0.0 ? float(textureSize(sampler2D(tex_env, env_smp), 0).x)
        : float(textureSize(sampler2DArray(tex_probes, env_smp), 0).x);
    return env_texel(env_uv(d), env_lod(env_blur(rough), width)).rgb * env_params.y;
}

@include_block surface_hooks

// What a surface facing `facing` mirrors of its surroundings in direction d: from the material's own
// map, or from the scene's reflection probes: the one whose box the point is in (or just
// outside of, facing in: a window pane in the room's wall), cross-faded with its neighbour
// over the blend width outside the box, and with the probe for everywhere else beyond that.
// Where a rough surface's reflection mostly comes from: the mirror direction r, leaning towards
// the normal as the surface gets rougher (the GGX lobe's dominant direction; three's
// getIBLRadiance).
vec3 dominant_reflection(vec3 r, vec3 n, float rough) {
    float r2 = rough * rough;
    return normalize(mix(r, n, r2 * r2));
}

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

// three's bicubic filter for the transmission picture (transmission_pars_fragment): four
// bilinear taps, weighted, for a smooth picture at the blurrier mip levels.
float bicubic_w0(float a) { return (1.0 / 6.0) * (a * (a * (-a + 3.0) - 3.0) + 1.0); }
float bicubic_w1(float a) { return (1.0 / 6.0) * (a * a * (3.0 * a - 6.0) + 4.0); }
float bicubic_w2(float a) { return (1.0 / 6.0) * (a * (a * (-3.0 * a + 3.0) + 3.0) + 1.0); }
float bicubic_w3(float a) { return (1.0 / 6.0) * (a * a * a); }

vec4 transmission_bicubic(vec2 uv, float lod) {
    vec2 size = vec2(textureSize(sampler2D(tex_transmission, transmission_smp), int(lod)));
    vec2 texel = 1.0 / size;
    uv = uv * size + 0.5;
    vec2 iuv = floor(uv);
    vec2 f = fract(uv);
    float g0x = bicubic_w0(f.x) + bicubic_w1(f.x);
    float g1x = bicubic_w2(f.x) + bicubic_w3(f.x);
    float h0x = -1.0 + bicubic_w1(f.x) / g0x;
    float h1x = 1.0 + bicubic_w3(f.x) / g1x;
    float g0y = bicubic_w0(f.y) + bicubic_w1(f.y);
    float g1y = bicubic_w2(f.y) + bicubic_w3(f.y);
    float h0y = -1.0 + bicubic_w1(f.y) / g0y;
    float h1y = 1.0 + bicubic_w3(f.y) / g1y;
    vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - 0.5) * texel;
    vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - 0.5) * texel;
    vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - 0.5) * texel;
    vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - 0.5) * texel;
    return g0y * (g0x * textureLod(sampler2D(tex_transmission, transmission_smp), p0, lod)
                + g1x * textureLod(sampler2D(tex_transmission, transmission_smp), p1, lod))
         + g1y * (g0x * textureLod(sampler2D(tex_transmission, transmission_smp), p2, lod)
                + g1x * textureLod(sampler2D(tex_transmission, transmission_smp), p3, lod));
}

// What is behind the surface where a ray leaves it at `exit`: the transmission picture there,
// blurred by the roughness (three's getTransmissionSample: a rough surface scatters the rays,
// the more the more it bends them).
vec3 transmission_sample(vec3 exit, float rough, float ior) {
    vec4 clip = cluster_view_proj * vec4(exit, 1.0);
    vec2 uv = (clip.xy / max(abs(clip.w), 1e-5)) * 0.5 + 0.5;
    // (the picture's row 0 is its top where the target's is, as for the reflection source)
    if (ssr_params.w > 0.5) uv.y = 1.0 - uv.y;
    float top = float(textureQueryLevels(sampler2D(tex_transmission, transmission_smp)) - 1);
    float lod = clamp(log2(max(ext_params.w, 1.0)) * rough * clamp(ior * 2.0 - 2.0, 0.0, 1.0), 0.0, top);
    vec3 fine = transmission_bicubic(uv, floor(lod)).rgb;
    if (fract(lod) == 0.0) return fine;
    return mix(fine, transmission_bicubic(uv, min(ceil(lod), top)).rgb, fract(lod));
}

// The ray bent by `ior` as it enters, through `thickness` meters of the volume (times the
// object's scale): three's getVolumeTransmissionRay.
// With RefractionModel.PLANE (ext_params.x bit 64) the volume is a wall `thickness` across:
// a slanted ray crosses thickness / cos of it (up to 12 times as much, at the silhouette of
// a curved wall), so it is shifted and tinted more toward the edges.
vec3 transmission_ray(vec3 n, vec3 v, float thickness, float ior) {
    vec3 ray = refract(-v, n, 1.0 / ior);
    // total internal reflection: no way through
    if (dot(ray, ray) < 1e-8) return vec3(0.0);
    ray = normalize(ray);
    float across = thickness * ext_params.z;
    if ((int(ext_params.x + 0.5) & 64) != 0) across /= max(dot(ray, -n), 1.0 / 12.0);
    return ray * across;
}

// Beer's law over `distance` meters (transmission_params.w: attenuation distance; 0: none).
vec3 volume_attenuation(float distance) {
    if (transmission_params.w <= 0.0) return vec3(1.0);
    vec3 coefficient = -log(max(transmission_color.rgb, vec3(1e-4))) / transmission_params.w;
    return exp(-coefficient * distance);
}

// three's EnvironmentBRDF: how much of the light a surface reflects, all directions taken
// together (Karis' fit of the split-sum integral).
vec3 environment_brdf(vec3 n, vec3 v, vec3 f0, float f90, float rough) {
    float nov = clamp(dot(n, v), 0.0, 1.0);
    vec4 r = rough * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
    float a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
    vec2 fab = vec2(-1.04, 1.04) * a004 + r.zw;
    return f0 * fab.x + f90 * fab.y;
}

// The light that comes through a transmission material from behind (three's
// getIBLVolumeRefraction): bent through the volume, tinted by the surface's color and the
// volume's, less what the surface reflects. With dispersion each channel is bent by its own
// IOR, so edges fringe with rainbow colors.
vec3 refracted_light(vec3 n, vec3 v, vec3 world_pos, float rough, vec3 diffuse_color, float thickness) {
    float ior = transmission_params.x > 0.0 ? transmission_params.x : 1.5;
    float spread = (ior - 1.0) * 0.025 * transmission_params.z;
    vec3 light;
    vec3 transmittance;
    if (spread > 0.0) {
        vec3 iors = vec3(ior - spread, ior, ior + spread);
        for (int i = 0; i < 3; i++) {
            vec3 ray = transmission_ray(n, v, thickness, iors[i]);
            light[i] = transmission_sample(world_pos + ray, rough, iors[i])[i];
            transmittance[i] = diffuse_color[i] * volume_attenuation(length(ray))[i];
        }
    } else {
        vec3 ray = transmission_ray(n, v, thickness, ior);
        light = transmission_sample(world_pos + ray, rough, ior);
        transmittance = diffuse_color * volume_attenuation(length(ray));
    }
    return (1.0 - environment_brdf(n, v, g_f0, g_f90, rough)) * transmittance * light;
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
    g_base.rgb *= v_instance_color;
    vec3 map_rgb = vec3(1.0);
    // parallax_map: where the view ray meets the height map, for all of the material's maps
    g_uv = v_uv;
    g_duvdx = dFdx(v_uv);
    g_duvdy = dFdy(v_uv);
    g_duv1dx = dFdx(v_uv1);
    g_duv1dy = dFdy(v_uv1);
    // Material.defocus: the maps read over the lens's blur at this distance, as wide as it is
    // (the depth of field pass sees through what is transparent)
    if (focus_params.x > 0.0) {
        float f = max(ext_slot_texel(int(map_slots[MAP_FOCUS / 4][MAP_FOCUS % 4]), vec2(0.5)).r, 0.01);
        float z = max(length(v_world_pos - camera_pos.xyz), 0.01);
        float blur = min(focus_params.x / f * abs(1.0 - f / z), focus_params.y);
        float widen = max(2.0 * blur, 1.0);
        g_duvdx *= widen;
        g_duvdy *= widen;
        g_duv1dx *= widen;
        g_duv1dy *= widen;
    }
    g_parallax_depth = 0.0;
    if (parallax_params.x > 0.0 && mode != 0) {
        vec3 pn = normalize(v_normal);
        if (!gl_FrontFacing && !foliage) pn = -pn;
        vec3 pv = camera_pos.xyz - v_world_pos;
        float pd = length(pv);
        parallax_offset(pn, pv / max(pd, 1e-5), pd);
    }
    // Material.hex_tiling
    if (parallax_params.w > 0.0) hex_setup(g_uv, parallax_params.w);
    if (tex_params.x > 0.5) {
        vec4 texel = g_hex ? hex_color(g_uv, g_duvdx, g_duvdy)
                           : textureGrad(sampler2D(tex_map, tex_smp), g_uv, g_duvdx, g_duvdy);  // (parallax: g_uv)
        g_base *= texel;
        map_rgb = texel.rgb;
    }
    // three's alphaMap: its green channel
    if (has_map(MAP_ALPHA)) g_base.a *= material_map(MAP_ALPHA).g;
    bool nodes = node_flags.x > 0.5;
    NodeIn ni;
    if (nodes) {
        vec3 fn = normalize(v_normal);
        if (!gl_FrontFacing && !foliage) fn = -fn;
        ni = node_in(fn, fn);
        node_color(ni, g_base);
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

    // (baking: the faces as the uv unwrap turned them, all front)
    bool baking = bake_params.x > 0.5;
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing && !foliage && !baking) n = -n;
    vec3 geo_n = n;
    g_dpdx = dFdx(v_world_pos);
    g_dpdy = dFdy(v_world_pos);
    if (tex_params.y != 0.0 && mode != 0) n = perturb_normal(n);
    if (has_map(MAP_BUMP) && mode != 0) {
        // three's bumpMap: the normal tilted by how the height changes from pixel to pixel
        // (its perturbNormalArb, Mikkelsen's)
        int bs = int(map_slots[MAP_BUMP / 4][MAP_BUMP % 4]);
        int bu = bs + 2;
        vec2 buv = slot_uv(bu, vec2(0.0));
        vec2 dx = slot_dx(bu);
        vec2 dy = slot_dy(bu);
        float hll = ext_slot_grad(bs, buv, dx, dy).r;
        vec2 dh = surface_params.x * vec2(ext_slot_grad(bs, buv + dx, dx, dy).r - hll, ext_slot_grad(bs, buv + dy, dx, dy).r - hll);
        vec3 sx = normalize(g_dpdx);
        vec3 sy = normalize(g_dpdy);
        vec3 r1 = cross(sy, n);
        vec3 r2 = cross(n, sx);
        float det = dot(sx, r1) * (gl_FrontFacing ? 1.0 : -1.0);
        vec3 grad = sign(det) * (dh.x * r1 + dh.y * r2);
        n = normalize(abs(det) * n - grad);
    }
    if (mode != 0) normal_hook(geo_n, n);
    if (nodes && mode != 0) {
        ni.normal = n;
        ni.geo_normal = geo_n;
        node_normal(ni, n);
    }
#ifdef MRT
    frag_normal.xy = pack_normal_octahedral(n);
#endif

    g_rough = mat_params.y;
    g_metal = mat_params.z;
    float ao = 1.0;
    int arm_bits = int(tex_params.z + 0.5);
    if (arm_bits != 0) {
        vec3 arm = slot_texel(tex_arm, 1).rgb;  // (parallax: g_uv)
        if ((arm_bits & 1) != 0) ao = arm.r;
        if ((arm_bits & 2) != 0) g_rough *= arm.g;
        if ((arm_bits & 4) != 0) g_metal *= arm.b;
    }
    int ext_bits = int(ext_params.x + 0.5);
    if (has_map(MAP_AO)) {
        // three's aoMap and aoMapIntensity
        float occlusion = material_map(MAP_AO).r;
        ao *= 1.0 + (occlusion - 1.0) * ext_params.y;
    }

    g_iridescence = 0.0;
    g_clearcoat = 0.0;
    // (a coat that the surface hook lays on: smooth, along the surface's own normal)
    g_clearcoat_rough = 0.045;
    g_clearcoat_n = geo_n;
    g_clearcoat_direct = vec3(0.0);
    g_sheen_rough = 0.0;
    g_sheen_light = vec3(0.0);
    // the clear coat and the sheen before the surface hook, which can change them (a wet
    // surface: a coat of water over it)
    if (mode == 3) {
        // the clear coat: its strength and roughness, and its normal (the surface's own, under
        // the normal map, or from its own map)
        g_clearcoat = clearcoat_params.x;
        if (g_clearcoat > 0.0) {
            if (has_map(MAP_CLEARCOAT)) g_clearcoat *= material_map(MAP_CLEARCOAT).r;
            g_clearcoat = clamp(g_clearcoat, 0.0, 1.0);
            g_clearcoat_rough = clearcoat_params.y;
            if (has_map(MAP_CLEARCOAT_ROUGHNESS)) g_clearcoat_rough *= material_map(MAP_CLEARCOAT_ROUGHNESS).g;
            g_clearcoat_rough = clamp(g_clearcoat_rough, 0.045, 1.0);
            g_clearcoat_n = geo_n;
            if (has_map(MAP_CLEARCOAT_NORMAL)) {
                vec3 t = material_map(MAP_CLEARCOAT_NORMAL).xyz * 2.0 - 1.0;
                t.xy *= clearcoat_params.zw;
                int cs = map_uv_slot(MAP_CLEARCOAT_NORMAL);
                g_clearcoat_n = perturb_with(geo_n, t, slot_dx(cs), slot_dy(cs));
            }
        }
        // the sheen (0 roughness: none)
        if (any(greaterThan(sheen_params.rgb, vec3(0.0)))) {
            g_sheen_color = sheen_params.rgb;
            g_sheen_rough = clamp(sheen_params.a, 1e-4, 1.0);
            if (has_map(MAP_SHEEN_COLOR)) g_sheen_color *= material_map(MAP_SHEEN_COLOR).rgb;
            if (has_map(MAP_SHEEN_ROUGHNESS)) g_sheen_rough *= material_map(MAP_SHEEN_ROUGHNESS).a;
        }
    }
    vec4 node_glow = vec4(0.0);
    if (nodes) {
        ni.normal = n;
        node_surface(ni, g_rough, g_metal, node_glow);
        g_rough = clamp(g_rough, 0.0, 1.0);
        g_metal = clamp(g_metal, 0.0, 1.0);
    }
    if (mode != 0) {
        // (the hook's opacity counts too, as its contract says: a water's edge fading out by
        // the depth behind it, say)
        float before = g_base.a;
        surface_hook(geo_n, g_base, g_rough, g_metal);
        if (before > 0.0) alpha *= clamp(g_base.a / before, 0.0, 1.0);
    }
    // what it reflects head on: 4% for glass's IOR of 1.5, tinted and scaled by the specular
    // color and intensity (KHR_materials_specular); metals reflect their own color
    g_f0 = mix(vec3(0.04), g_base.rgb, g_metal);
    g_f90 = 1.0;
    g_anisotropy = 0.0;
    if (mode == 3) {
        float ior = transmission_params.x > 0.0 ? transmission_params.x : 1.5;
        float r0 = (ior - 1.0) / (ior + 1.0);
        vec3 tint = specular_params.rgb;
        float strength = specular_params.a;
        if (has_map(MAP_SPECULAR_COLOR)) tint *= material_map(MAP_SPECULAR_COLOR).rgb;
        if (has_map(MAP_SPECULAR_INTENSITY)) strength *= material_map(MAP_SPECULAR_INTENSITY).a;
        vec3 dielectric = min(vec3(r0 * r0) * tint, vec3(1.0)) * strength;
        g_f0 = mix(dielectric, g_base.rgb, g_metal);
        g_f90 = mix(strength, 1.0, g_metal);
        vec3 pv = normalize(camera_pos.xyz - v_world_pos);
        float nov = clamp(dot(n, pv), 0.0, 1.0);
        // iridescence (three's): the film's reflectance over a dielectric and over a metal
        g_iridescence = iridescence_params.x;
        if (g_iridescence > 0.0) {
            if (has_map(MAP_IRIDESCENCE)) g_iridescence *= material_map(MAP_IRIDESCENCE).r;
            float film = iridescence_params.w;
            if (has_map(MAP_IRIDESCENCE_THICKNESS)) film = mix(iridescence_params.z, iridescence_params.w, material_map(MAP_IRIDESCENCE_THICKNESS).g);
            g_iridescence = film == 0.0 ? 0.0 : clamp(g_iridescence, 0.0, 1.0);
            if (g_iridescence > 0.0) {
                vec3 over_dielectric = eval_iridescence(1.0, iridescence_params.y, nov, film, dielectric);
                vec3 over_metal = eval_iridescence(1.0, iridescence_params.y, nov, film, g_base.rgb);
                g_iridescence_fresnel = mix(over_dielectric, over_metal, g_metal);
                g_iridescence_f0 = mix(schlick_to_f0(over_dielectric, 1.0, nov), schlick_to_f0(over_metal, 1.0, nov), g_metal);
            }
        }
        // anisotropy (three's): highlights drawn out along a direction of the surface, as on
        // brushed metal. The direction in the tangent frame, turned by the map's (rg, around
        // its middle) and scaled by its b
        if (any(notEqual(tangent_params.zw, vec2(0.0)))) {
            vec2 dir = tangent_params.zw;
            if (has_map(MAP_ANISOTROPY)) {
                vec3 polar = material_map(MAP_ANISOTROPY).rgb;
                vec2 turn = 2.0 * polar.rg - 1.0;
                turn *= polar.b / max(length(turn), 1e-6);
                dir = vec2(dir.x * turn.x - dir.y * turn.y, dir.y * turn.x + dir.x * turn.y);
            }
            g_anisotropy = length(dir);
            dir = g_anisotropy > 0.0 ? dir / g_anisotropy : vec2(1.0, 0.0);
            g_anisotropy = min(g_anisotropy, 1.0);
            // the frame: the normal map's uv, or the clear coat's normal map's, or the color map's
            vec3 t;
            vec3 b;
            if (tex_params.y != 0.0) {
                tangent_frame(geo_n, slot_dx(0), slot_dy(0), t, b);
            } else if (has_map(MAP_CLEARCOAT_NORMAL)) {
                int cs = map_uv_slot(MAP_CLEARCOAT_NORMAL);
                tangent_frame(geo_n, slot_dx(cs), slot_dy(cs), t, b);
            } else {
                tangent_frame(geo_n, g_duvdx, g_duvdy, t, b);
            }
            g_anisotropy_t = t * dir.x + b * dir.y;
            g_anisotropy_b = b * dir.x - t * dir.y;
        }
    }
#ifdef MRT
    frag_indirect.ba = vec2(g_rough, g_metal);
#endif

    vec3 outc;
    vec3 spec = vec3(0.0);
    // the light the surface gives off itself (emissive, baked), which transmission leaves alone
    vec3 own_light = vec3(0.0);
    // the indirect light in outc, diffuse and specular (for frag_indirect)
    vec3 ind_d = vec3(0.0);
    vec3 ind_s = vec3(0.0);
    vec3 to_cam = camera_pos.xyz - v_world_pos;
    float depth = length(to_cam);
    vec3 v = to_cam / max(depth, 1e-5);

    if (mode == 0) {
        outc = g_base.rgb;
        // three's lightMap: the indirect light is the map's instead of 1
        if (has_map(MAP_LIGHT)) outc *= material_map(MAP_LIGHT).rgb * light_map_params.x * RECIPROCAL_PI;
        if ((ext_bits & 32) != 0) {
            // the scene's background (Scene.background_texture): its map in the direction of the
            // view, blurred about as three's PMREM blurs it for backgroundBlurriness (a gaussian
            // about `blurriness` radians wide)
            vec3 d = normalize(v_world_pos - camera_pos.xyz);
            float width = float(textureSize(sampler2D(tex_env, env_smp), 0).x);
            float lod = g_rough > 0.0 ? env_lod(g_rough, width) : 0.0;
            g_env_layer = -1.0;
            outc = (lod > 0.0 ? env_texel(env_uv(d), lod) : env_texel_smooth(env_uv(d))).rgb * env_params.y;
        }
    } else if (mode == 4) {
        // MeshNormalMaterial: the normal in view space as the color (bypasses color
        // management, like three)
        frag_color = vec4(normalize(to_view(n)) * 0.5 + 0.5, color.a);
        return;
    } else if (mode == 7) {
        // MeshDepthMaterial: the window depth, packed as surface_params.z says (three's
        // packing), not color managed
        frag_color = pack_depth(gl_FragCoord.z, int(surface_params.z), color.a);
        return;
    } else if (mode == 5) {
        // MeshMatcapMaterial: the matcap image at the view-space normal, from a frame turned
        // toward the viewer (three's), times the color
        vec3 vd = normalize(to_view(v));
        vec3 vn = normalize(to_view(n));
        vec3 x = normalize(vec3(vd.z, 0.0, -vd.x));
        vec3 y = cross(vd, x);
        vec2 muv = vec2(dot(x, vn), dot(y, vn)) * 0.495 + 0.5;
        int ms = int(map_slots[MAP_MATCAP / 4][MAP_MATCAP % 4]);
        vec3 matcap = ms >= 0 ? ext_slot_grad(ms, muv, dFdx(muv), dFdy(muv)).rgb : vec3(mix(0.2, 0.8, muv.y));
        outc = g_base.rgb * matcap;
    } else {
        vec3 diffuse_color = g_base.rgb;
        if (mode == 3) diffuse_color *= (1.0 - g_metal);

        vec3 sky_light = ambient.rgb + mix(hemi_ground.rgb, hemi_sky.rgb, 0.5 * n.y + 0.5);
        // the scene's environment (Scene.environment): the light from all around n, and its
        // sun, unless a directional light comes from where the sun is: that light is the sun
        // (with its shadow), the picture's is the same light again
        if (env_params.z > 0.0) {
            sky_light += env_params.y * env_irradiance(n);
            vec3 sun = env_sun(n);
            for (int i = 0; i < int(counts.x); i++) {
                if (dot(normalize(dirs[i].dir.xyz), env_sun_dir()) > 0.996) sun = vec3(0.0);
            }
            sky_light += env_params.y * sun;
        }
        float gi_cover;
        vec3 gi = gi_irradiance(n, gi_cover);
        vec3 irradiance = gi_min[0].w > 1.5 ? sky_light * (1.0 - gi_cover) + gi : sky_light + gi;
        if (rc_size.w > 0.0) {
            // radiance cascades: all the indirect light inside their box
            float rc_fade;
            vec3 rc = rc_irradiance(n, rc_fade);
            irradiance = mix(irradiance, rc, rc_fade);
        }
        // three's lightMap: baked light, added to the indirect light
        if (has_map(MAP_LIGHT)) irradiance += material_map(MAP_LIGHT).rgb * light_map_params.x;
        outc = irradiance * diffuse_color * RECIPROCAL_PI * ao;
        if (g_sheen_rough > 0.0) {
            float albedo = sheen_albedo(n, v, g_sheen_rough);
            g_sheen_light += irradiance * g_sheen_color * albedo * RECIPROCAL_PI;
            outc *= 1.0 - max(g_sheen_color.r, max(g_sheen_color.g, g_sheen_color.b)) * albedo;
        }
        ind_d = outc;

        int ndir = int(counts.x);
        g_sized = false;
        g_nol = -1.0;
        // where a light with a size is mirrored from (light_toward), with the material's
        // surface (not a standard material's: not at all)
        vec3 mirror_r = reflect(-v, n);
        bool sized_ok = mode == 3;
        for (int i = 0; i < ndir; i++) {
            vec3 l = normalize(dirs[i].dir.xyz);
            float lit = int(dir_shadow_params.x) == i + 1 ? dir_shadow(n, l) : 1.0;
            // the other shadowed ones: a tile of shadow_atlas each
            int dsi = int(dirs[i].shadow.x);
            if (dsi >= 0 && (flags() & 4) != 0) lit = mix(1.0, dir_tile_shadow(dsi, n, l), dirs[i].shadow.y);
            // parallax_map: the bumps' own shadows, toward the first directional light
            if (i == 0) lit *= parallax_shadow(geo_n, l);
            if (lit > 0.0) outc += direct(mode, n, v, l, 1e6, dirs[i].color.rgb * lit, diffuse_color, spec);
        }
        if (dir_cascade_params.z > 0.5 && dir_shadow_params.x > 0.5) {
            // renderer.debug_cascades: red, green, blue, yellow
            int k = sun_cascade(dot(v_world_pos - camera_pos.xyz, dir_cascade_fwd.xyz));
            if (k >= 0) outc = mix(outc, k == 0 ? vec3(1.0, 0.0, 0.0) : (k == 1 ? vec3(0.0, 1.0, 0.0) : (k == 2 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 1.0, 0.0))), 0.5);
        }
        // the point and spot lights that reach this fragment's cluster (baking: all of them,
        // the clusters are the camera's)
        int first = 0;
        int npoint = int(counts.y);
        int nspot = int(counts.z);
        if (!baking) {
            uvec4 range = clusters[cluster_of(v_world_pos)].range;
            first = int(range.x);
            npoint = int(range.y);
            nspot = int(range.z);
        }
        for (int k = 0; k < npoint; k++) {
            gpu_point_light pl = points[baking ? uint(k) : cluster_lights[first + k].index];
            vec3 lv = pl.pos.xyz - v_world_pos;
            float d = length(lv);
            // a light source with a size: falloff softens up close instead of blowing out
            float radius = pl.params.x;
            vec3 seg = pl.params.yzw;
            vec3 l = lv / d;
            vec3 radiance;
            g_sized = false;
            if (sized_ok && radius + abs(seg.x) + abs(seg.y) + abs(seg.z) > 0.0) light_toward(mirror_r, lv, seg, radius);
            if (dot(seg, seg) > 0.0) {
                // a tube (PointLight.length): the light of the whole segment, its intensity
                // spread along it (Karis' eq. 16, the radius softening it as a ball's
                // falloff: d^2 + radius^2 at length 0), faded out by the distance to its
                // nearest point
                vec3 l0 = lv - seg;
                vec3 l1 = lv + seg;
                float d0 = length(l0);
                float d1 = length(l1);
                vec3 ld = 2.0 * seg;
                vec3 near = l0 + clamp(-dot(l0, ld) / dot(ld, ld), 0.0, 1.0) * ld;
                float dn = length(near);
                g_nol = 2.0 * clamp(0.5 * dot(n, l0) / d0 + 0.5 * dot(n, l1) / d1, 0.0, 1.0)
                    / max(d0 * d1 + dot(l0, l1) + 2.0 * radius * radius, 1e-4);
                l = near / max(dn, 1e-6);
                radiance = pl.color.rgb * distance_window(dn, pl.pos.w);
            } else {
                g_nol = -1.0;
                float d_soft = sqrt(d * d + radius * radius);
                radiance = pl.color.rgb * distance_attenuation(d_soft, pl.pos.w, pl.color.w);
            }
            radiance *= fog_transmittance(fog_distance(pl.pos.xyz, v_world_pos));
            if (pl.shadow.x >= 0.0) radiance *= point_shadow(pl, geo_n);
            outc += direct(mode, n, v, l, d, radiance, diffuse_color, spec);
        }
        g_sized = false;
        g_nol = -1.0;
        for (int k = 0; k < nspot; k++) {
            int i = baking ? k : int(cluster_lights[first + npoint + k].index);
            vec3 lv = spots[i].pos.xyz - v_world_pos;
            float d = length(lv);
            // out of the light's reach
            if (spots[i].pos.w > 0.0 && d > spots[i].pos.w) continue;
            vec3 l = lv / d;
            float cone = smoothstep(spots[i].dir.w, spots[i].params.x, dot(-l, spots[i].dir.xyz));
            if (cone <= 0.0) continue;
            float d_soft = sqrt(d * d + spots[i].params.y * spots[i].params.y);
            g_sized = false;
            if (sized_ok && spots[i].params.z > 0.0) light_toward(mirror_r, lv, vec3(0.0), spots[i].params.z);
            vec3 radiance = spots[i].color.rgb * cone * distance_attenuation(d_soft, spots[i].pos.w, spots[i].color.w);
            radiance *= fog_transmittance(fog_distance(spots[i].pos.xyz, v_world_pos));
            radiance *= shadow(i, geo_n);
            outc += direct(mode, n, v, l, d, radiance, diffuse_color, spec);
        }
        // three's specularMap: the highlights' strength
        if (mode == 2 && has_map(MAP_SPECULAR)) spec *= material_map(MAP_SPECULAR).r;
        int nrect = int(counts.w);
        // three's RectAreaLight lights standard and physical materials only
        if (mode == 3 && nrect > 0) {
            int ltc_base = int(counts.y) + nrect;
            float rough = max(g_rough, 0.0525);
            vec4 t1;
            vec4 t2;
            ltc_tables(ltc_base, n, v, rough, t1, t2);
            mat3 m_inv = mat3(vec3(t1.x, 0.0, t1.y), vec3(0.0, 1.0, 0.0), vec3(t1.z, 0.0, t1.w));
            vec3 fresnel = g_f0 * t2.x + (vec3(g_f90) - g_f0) * t2.y;
            mat3 cc_inv = mat3(1.0);
            float cc_fresnel = 0.0;
            if (g_clearcoat > 0.0) {
                vec4 c1;
                vec4 c2;
                ltc_tables(ltc_base, g_clearcoat_n, v, max(g_clearcoat_rough, 0.0525), c1, c2);
                cc_inv = mat3(vec3(c1.x, 0.0, c1.y), vec3(0.0, 1.0, 0.0), vec3(c1.z, 0.0, c1.w));
                cc_fresnel = 0.04 * c2.x + 0.96 * c2.y;
            }
            for (int k = 0; k < nrect; k++) {
                gpu_point_light ra = points[int(counts.y) + k];
                vec3 hw = vec3(ra.pos.w, ra.color.w, ra.params.x);
                vec3 hh = ra.params.yzw;
                vec3 to = ra.pos.xyz - v_world_pos;
                // behind the light (it shines from its front only)
                if (dot(cross(hw, hh), to) < 0.0) continue;
                vec3 q[4];
                q[0] = to + hw - hh;
                q[1] = to - hw - hh;
                q[2] = to - hw + hh;
                q[3] = to + hw + hh;
                vec3 c = ra.color.rgb * fog_transmittance(fog_distance(ra.pos.xyz, v_world_pos));
                spec += c * fresnel * ltc_evaluate(n, v, m_inv, q);
                outc += c * diffuse_color * ltc_evaluate(n, v, mat3(1.0), q);
                if (g_clearcoat > 0.0) g_clearcoat_direct += c * cc_fresnel * ltc_evaluate(g_clearcoat_n, v, cc_inv, q);
            }
        }
        spec *= ao;
        // Material.map_emissive: the color map glows too
        vec3 glow = (fl & 64) != 0 ? emissive.rgb * map_rgb : emissive.rgb;
        // three's emissiveMap
        if (has_map(MAP_EMISSIVE)) glow *= material_map(MAP_EMISSIVE).rgb;
        // a node material's emissiveNode
        if (node_glow.w > 0.5) glow = node_glow.rgb;
        outc += glow;
        own_light = glow;
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
        vec3 mirror_term = vec3(0.0);
        // (not the background: it is the surroundings)
        if (env_params.x > 0.5 && mode != 1 && (ext_bits & 32) == 0 && env_combine.w < 0.5) {
            // reflections of the surroundings, fading as the surface gets rougher
            float nov = clamp(dot(n, v), 0.0, 1.0);
            vec3 f0 = g_iridescence > 0.0 ? mix(g_f0, g_iridescence_f0, g_iridescence) : g_f0;
            vec3 fr = f0 + (max(vec3((1.0 - g_rough) * g_f90), f0) - f0) * pow(1.0 - nov, 5.0);
            vec3 rn = n;
            if (g_anisotropy > 0.0) {
                // the normal bent toward the stretched highlight (three's
                // getIBLAnisotropyRadiance, Filament's)
                vec3 bent = normalize(cross(cross(g_anisotropy_b, v), g_anisotropy_b));
                float k = 1.0 - g_anisotropy * (1.0 - g_rough);
                k *= k;
                rn = normalize(mix(bent, n, k * k));
            }
            vec3 r = dominant_reflection(reflect(-v, rn), rn, g_rough);
            vec3 env = env_reflection(r, g_rough, n);
            reflection_hook(r, env);
            env_term = env * fr;
            if (g_sheen_rough > 0.0) env_term *= 1.0 - max(g_sheen_color.r, max(g_sheen_color.g, g_sheen_color.b)) * sheen_albedo(n, v, g_sheen_rough);
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
            mirror_term = mirrored * (found * fresnel * strength * ao);
            env_term *= 1.0 - clamp(found * strength, 0.0, 1.0);
            outc += mirror_term;
            ind_s += mirror_term;
        }
        outc += env_term;
        ind_s += env_term;
        outc += spec;
        // transmission (Material.transmission): what is behind the surface, refracted through
        // the volume, takes the place of its diffuse light; highlights and reflections stay
        if (mode == 3 && transmission_color.a > 0.0) {
            float amount = transmission_color.a;
            if (has_map(MAP_TRANSMISSION)) amount *= material_map(MAP_TRANSMISSION).r;
            float thickness = max(transmission_params.y, 0.0);
            if (has_map(MAP_THICKNESS)) thickness *= material_map(MAP_THICKNESS).g;
            vec3 refracted = refracted_light(n, v, v_world_pos, g_rough, g_base.rgb * (1.0 - g_metal), thickness);
            // (the light it gives off stays too: three's totalEmissiveRadiance)
            vec3 highlights = spec + env_term + mirror_term + own_light;
            outc = mix(max(outc - highlights, vec3(0.0)), refracted, amount) + highlights;
        }
        // the physical material's layers over it all: the sheen, then the clear coat, which
        // lets through what it does not reflect (three's)
        if (g_sheen_rough > 0.0) {
            outc += g_sheen_light;
            ind_s += g_sheen_light;
        }
        if (g_clearcoat > 0.0) {
            vec3 cn = g_clearcoat_n;
            vec3 coat = g_clearcoat_direct;
            if (env_params.x > 0.5) {
                vec3 r = dominant_reflection(reflect(-v, cn), cn, g_clearcoat_rough);
                vec3 env = env_reflection(r, g_clearcoat_rough, cn);
                reflection_hook(r, env);
                vec3 reflected = env * environment_brdf(cn, v, vec3(0.04), 1.0, g_clearcoat_rough);
                coat += reflected;
                ind_s += reflected * g_clearcoat;
            }
            float f = schlick(0.04, 1.0, clamp(dot(cn, v), 0.0, 1.0));
            outc = outc * (1.0 - g_clearcoat * f) + coat * g_clearcoat;
        }
    }

    // three's envMap on a basic, lambert or phong material: what it mirrors (or what is seen
    // through it) over all of its light, by its reflectivity times the specular map's red
    if (env_combine.w > 0.5) {
        vec3 to_frag = -v;
        vec3 d = env_combine.z > 0.0 ? refract(to_frag, n, env_combine.z) : reflect(to_frag, n);
        vec3 seen = env_classic(d) * env_params.y;
        float k = env_combine.x;
        if (has_map(MAP_SPECULAR)) k *= material_map(MAP_SPECULAR).r;
        int combine = int(env_combine.y + 0.5);
        if (combine == 0) outc = mix(outc, outc * seen, k);
        else if (combine == 1) outc = mix(outc, seen, k);
        else outc += seen * k;
    }

    // Material.vertex_occlusion: the vertex colors' brightness dims all of it
    if ((fl & 128) != 0) {
        float k = dot(v_color, vec3(0.2126, 0.7152, 0.0722));
        outc *= k;
        ind_d *= k;
        ind_s *= k;
    }

    // a node material's outputNode: what the light made of it, and its alpha
    if (nodes) {
        ni.normal = n;
        vec4 lit = vec4(outc, alpha);
        node_output(ni, lit);
        outc = lit.rgb;
        alpha = lit.a;
    }

    int fog_mode = int(fog_params.w);
    float display_fog_t = 1.0;
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
        // (Renderer.display_fog: mixed in after tone mapping, below)
        if (fog_height.z > 0.5) display_fog_t = t;
        vec3 fog_lit = fog_height.z > 0.5 ? vec3(0.0) : fog_own * (1.0 - t);
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
        if (fog_height.z <= 0.5) outc = outc * t + fog_lit;
        ind_d *= t;
        ind_s *= t;
    }

    outc = max(outc, vec3(0.0));
    if (baking) {
        // the light map: the new sample blended into the last map's texel, in linear light
        vec3 prev = texture(sampler2D(ssr_color, ssr_smp), v_uv1).rgb;
        frag_color = vec4(mix(prev, min(outc, vec3(60000.0)), bake_params.y), 1.0);
        return;
    }
#ifdef MRT
    {
        const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
        float total = max(dot(outc, lw), 1e-6);
        frag_indirect.rg = clamp(vec2(dot(max(ind_d, 0.0), lw), dot(max(ind_s, 0.0), lw)) / total, 0.0, 1.0);
    }
#endif
    float shown_alpha = alpha;
    // a light source (emissive far over white) leaves the target's alpha at 0, for
    // post-processing that has to tell a burnt-out lamp from a brightly lit wall
    alpha *= 1.0 - smoothstep(4.0, 16.0, max(emissive.r, max(emissive.g, emissive.b)));
    // into a half-float target as it is (OutputPass tone maps it later), otherwise ready to show
    if (output_params.x > 0.5) frag_color = vec4(mix(fog_color.rgb, min(outc, vec3(60000.0)), display_fog_t), alpha);
    else {
        // the window's colors: sRGB, Display P3, or extended linear Display P3 (HDR: in SDR's range here)
        int display = int(output_params.w);
        // (Material.tone_mapped off: none)
        int tone = (fl & 512) != 0 ? 0 : int(output_params.z);
        vec3 toned = tone_map_gamut(outc, tone, camera_pos.w, display != DISPLAY_SRGB);
        // (three mixes the fog in the colors as they are shown)
        frag_color = display == DISPLAY_HDR ? vec4(mix(fog_color.rgb, toned, display_fog_t), alpha)
            : vec4(mix(linear_to_srgb(fog_color.rgb), linear_to_srgb(toned), display_fog_t), alpha);
    }
    // Material.premultiplied_alpha: the color as it is shown, times its opacity (three's)
    if ((fl & 256) != 0) frag_color.rgb *= shown_alpha;
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

// Renderer.shadow_type = VSM: the shadows from the maps' moments. Only these two: what has no
// holes is drawn by them too (each variant is a few MB of generated code).
@fs fs_vsm
#define VSM
@include_block fs_main
@end

@fs fs_mrt_vsm
#define MRT
#define VSM
@include_block fs_main
@end

@fs fs_opaque_mrt
#define NO_DISCARD
#define MRT
@include_block fs_main
@end

// The geometries with tangents (vs_tangent): the four that can cut holes, the opaque ones'
// draws too (each variant is a few MB of generated code)
@fs fs_tangent
#define TANGENT
@include_block fs_main
@end

@fs fs_mrt_tangent
#define TANGENT
#define MRT
@include_block fs_main
@end

@fs fs_vsm_tangent
#define TANGENT
#define VSM
@include_block fs_main
@end

@fs fs_mrt_vsm_tangent
#define TANGENT
#define MRT
#define VSM
@include_block fs_main
@end

// The depth prepass (renderer.c3, Renderer.depth_prepass): depth only, for what has holes in
// it (alpha tested, LOD cross-fades, the cull box). The main pass then draws those with
// fs_opaque and an EQUAL depth test, so that no shader of the main pass discards (a shader
// that can discard costs the GPU its early depth test, whatever it draws). Cuts out exactly
// what fs does.
@block depth_main
layout(binding=0) uniform texture2D tex_map;
layout(binding=1) uniform texture2D tex_alpha;
layout(binding=0) uniform sampler tex_smp;
layout(binding=1) uniform sampler tex_alpha_smp;
layout(binding=1) uniform depth_fs_params {
    vec4 d_color;          // a = the material's opacity
    vec4 d_params;         // x = has a map, y = alpha test threshold, z = 1: vertex colors' alpha,
                           //   w = 1: has an alpha map (tex_alpha, three's alphaMap: its green)
    vec4 d_screen_params;  // as screen_params in fs
    vec4 d_cull_min;       // as cull_min in fs
    vec4 d_cull_max;
    vec4 d_alpha_uv[2];    // the alpha map's uv from v_uv, as uv_rel
};

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in vec2 v_uv1;
#ifdef TANGENT
in vec4 v_tangent;
#endif
in vec3 v_color;
in float v_alpha;
in vec4 v_clip;
in vec4 v_prev_clip;
in float v_line_distance;
in vec3 v_instance_color;
in vec4 v_node0;
in vec4 v_node1;
in vec4 v_node2;
in vec4 v_node3;

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
        if (d_params.w > 0.5) {
            vec3 p = vec3(v_uv, 1.0);
            alpha *= texture(sampler2D(tex_alpha, tex_alpha_smp), vec2(dot(d_alpha_uv[0].xyz, p), dot(d_alpha_uv[1].xyz, p))).g;
        }
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
    float unread = 0.0;
#ifdef TANGENT
    unread = v_tangent.x * 0.0;
#endif
    frag_color = vec4(v_clip.w + unread + (v_normal + v_color + v_world_pos + v_clip.xyz + v_prev_clip.xyz).x * 0.0
        + v_line_distance * 0.0 + v_instance_color.x * 0.0 + (v_node0 + v_node1 + v_node2 + v_node3).x * 0.0,
        0.0, 0.0, alpha);
}
@end

@fs depth_fs
@include_block depth_main
@end

// The same for geometries with tangents: the main pass tests against the depth of this pass
// with EQUAL, which needs the same vertex shader (one compiled apart can place them a hair off)
@fs depth_tangent_fs
#define TANGENT
@include_block depth_main
@end

@program three vs fs
@program three_opaque vs fs_opaque
@program three_mrt vs fs_mrt
@program three_opaque_mrt vs fs_opaque_mrt
@program three_vsm vs fs_vsm
@program three_mrt_vsm vs fs_mrt_vsm
@program three_tangent vs_tangent fs_tangent
@program three_mrt_tangent vs_tangent fs_mrt_tangent
@program three_vsm_tangent vs_tangent fs_vsm_tangent
@program three_mrt_vsm_tangent vs_tangent fs_mrt_vsm_tangent
@program depth vs depth_fs
@program depth_tangent vs_tangent depth_tangent_fs
@program shadow shadow_vs shadow_fs

// Clears the scissor box (Renderer.clear and render() with scissor_test): a triangle over the
// whole target at the far depth, in one color.
@vs clear_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 1.0, 1.0);
}
@end

@fs clear_fs
layout(binding=0) uniform clear_params {
    vec4 clear_rgba;
};
out vec4 frag_color;
void main() {
    frag_color = clear_rgba;
}
@end

@program clear_rect clear_vs clear_fs
@program shadow_nodes shadow_nodes_vs shadow_nodes_fs
