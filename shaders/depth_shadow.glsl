// Utility depth (Renderer.render_back_faces) and shadow-atlas tile clear (clear_atlas_tile)
// shaders, plus the shared vertex/shadow uniform and GPU-cull storage buffer layouts.
// Regenerate generated/depth_shadow.c3 with `just shaders`.
@header module drie::shaders;
@header import drie;

@ctype mat4 Mat4
@ctype vec4 Vec4

@block vertex_common
in vec3 position;
in vec3 normal;
in vec2 uv;
in vec4 color0;
in vec2 uv1;
in vec4 inst0;
in vec4 inst1;
in vec4 inst2;
in float height;

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
uint inst_id;

bool fetch_instance(vec4 cull) {
    if (cull.x < 0.5) {
        m0 = inst0;
        m1 = inst1;
        m2 = inst2;
        inst_id = uint(gl_InstanceIndex);
        return true;
    }
    uint i = uint(gl_InstanceIndex);
    if (i >= counts[uint(cull.z)].index) return false;
    inst_id = visible[uint(cull.y) + i].index;
    uint s = inst_id * 3;
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

vec3 displaced(vec3 p, vec4 d) {
    return d.z > 0.5 ? p + normalize(normal) * (height * d.x + d.y) : p;
}
@end

@vs vs
layout(binding=0) uniform vs_params {
    mat4 view_proj;
    vec4 view_origin;
    mat4 model;
    mat4 normal_mat;
    vec4 uv_transform[2];
    vec4 sway;
    vec4 inst_cull;
    mat4 prev_view_proj;
    mat4 prev_model;
    vec4 motion;
    vec4 bake;
    vec4 surface_vs;
};

@include_block vertex_common
in vec3 instance_color;
in vec4 tangent;
in vec3 prev_position;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec2 v_uv1;
out vec4 v_tangent;
out vec3 v_color;
out float v_alpha;
out vec4 v_clip;
out vec4 v_prev_clip;
out float v_line_distance;
out vec3 v_instance_color;

void main() {
    v_line_distance = normal.x;
    v_instance_color = instance_color;
    if (!fetch_instance(inst_cull)) {
        v_world_pos = vec3(0.0);
        v_normal = vec3(0.0, 1.0, 0.0);
        v_uv = vec2(0.0);
        v_uv1 = vec2(0.0);
        v_tangent = vec4(0.0);
        v_color = vec3(0.0);
        v_alpha = 0.0;
        v_clip = vec4(2.0, 2.0, 2.0, 1.0);
        v_prev_clip = v_clip;
        gl_Position = v_clip;
        return;
    }
    vec3 ip = instance_point(displaced(position, vec4(surface_vs.yzw, 0.0)));
    vec4 wp = model * vec4(ip, 1.0);
    wp.xyz = apply_sway(wp.xyz, ip.y, sway);
    v_world_pos = wp.xyz;
    v_normal = (normal_mat * vec4(instance_dir(normal), 0.0)).xyz;
    v_uv = vec2(dot(uv_transform[0].xyz, vec3(uv, 1.0)), dot(uv_transform[1].xyz, vec3(uv, 1.0)));
    v_uv1 = uv1;
    v_tangent = vec4((model * vec4(instance_dir(tangent.xyz), 0.0)).xyz, tangent.w);
    v_color = color0.rgb;
    v_alpha = color0.a;
    gl_Position = view_proj * vec4(wp.xyz - view_origin.xyz, 1.0);
    if (surface_vs.x > 0.0) {
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
        vec3 was = instance_point(displaced(prev_position, vec4(surface_vs.yzw, 0.0)));
        vec4 pp = prev_model * vec4(was, 1.0);
        pp.xyz = apply_sway(pp.xyz, was.y, vec4(motion.y, sway.yzw));
        v_prev_clip = prev_view_proj * pp;
    } else if (motion.x > 0.5) {
        v_prev_clip = prev_view_proj * wp;
    }
}
@end

@fs depth_fs
layout(binding=0) uniform texture2D tex_map;
layout(binding=1) uniform texture2D tex_alpha;
layout(binding=0) uniform sampler tex_smp;
layout(binding=1) uniform sampler tex_alpha_smp;
layout(binding=1) uniform depth_fs_params {
    vec4 d_color;
    vec4 d_params;
    vec4 d_screen_params;
    vec4 d_cull_min;
    vec4 d_cull_max;
    vec4 d_alpha_uv[2];
};

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in vec2 v_uv1;
in vec4 v_tangent;
in vec3 v_color;
in float v_alpha;
in vec4 v_clip;
in vec4 v_prev_clip;
in float v_line_distance;
in vec3 v_instance_color;

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
    // R = view-space distance from the camera plane (Renderer.render_back_faces).
    float unread = v_tangent.x * 0.0;
    frag_color = vec4(v_clip.w + unread + (v_normal + v_color + v_world_pos + v_clip.xyz + v_prev_clip.xyz).x * 0.0
        + v_line_distance * 0.0 + v_instance_color.x * 0.0,
        0.0, 0.0, alpha);
}
@end

@vs shadow_vs
layout(binding=0) uniform shadow_vs_params {
    mat4 s_view_proj;
    vec4 s_origin;
    mat4 s_model;
    vec4 s_uv_transform[2];
    vec4 s_sway;
    vec4 s_inst_cull;
    vec4 s_displacement;
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
    vec4 wp = s_model * vec4(ip, 1.0);
    wp.xyz = apply_sway(wp.xyz, ip.y, s_sway);
    v_uv = vec2(dot(s_uv_transform[0].xyz, vec3(uv, 1.0)), dot(s_uv_transform[1].xyz, vec3(uv, 1.0)));
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
    vec4 s_params;
    vec4 s_alpha_uv[2];
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

@program depth vs depth_fs
@program shadow shadow_vs shadow_fs
