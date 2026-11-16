// Uber-shader for three's built-in materials, plus the depth-only shadow pass.
// Regenerate shader.c3 with: ./build.sh shaders
@header module three;

@ctype mat4 Mat4
@ctype vec4 Vec4

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

vec3 instance_point(vec3 p) {
    vec4 l = vec4(p, 1.0);
    return vec3(dot(inst0, l), dot(inst1, l), dot(inst2, l));
}

vec3 instance_dir(vec3 d) {
    return vec3(dot(inst0.xyz, d), dot(inst1.xyz, d), dot(inst2.xyz, d));
}

// Wind: bends vertices sideways the higher they sit above the model's origin.
// sway = (time, amplitude per meter of height, speed, 0)
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

// Someone pushing through: what is near `pusher` (xyz = their feet, w = reach in m) gives
// way sideways and sags a little, the more the nearer. `pin` is the point the vertex
// follows: itself for wood, which bends, and the middle of the leaf for a leaf's vertices,
// so that the leaf moves in one piece. What is close to the plant's stem (at `stem` on the
// ground plan) stays put. `strength` is the material's push.
vec3 apply_push(vec3 wp, vec3 pin, vec2 stem, vec4 pusher, float strength) {
    if (strength <= 0.0 || pusher.w <= 0.0) return wp;
    vec2 away = pin.xz - pusher.xz;
    float d = length(away);
    if (d >= pusher.w) return wp;
    float dy = pin.y - pusher.y;
    float k = 1.0 - d / pusher.w;
    k = k * k * (3.0 - 2.0 * k) * smoothstep(-0.3, 0.2, dy) * (1.0 - smoothstep(1.7, 2.3, dy));
    k *= smoothstep(0.12, 0.8, length(pin.xz - stem));
    float amount = k * strength * pusher.w * 0.22;
    // no direction right at the pusher's own spot
    wp.xz += away / max(d, 0.15) * amount;
    wp.y -= amount * 0.3;
    return wp;
}
@end

@vs vs
layout(binding=0) uniform vs_params {
    mat4 view_proj;
    mat4 model;
    mat4 normal_mat;
    vec4 uv_transform;  // xy = repeat, zw = offset
    vec4 sway;          // w = how much the material gives way to the pusher
    vec4 pusher;
};

@include_block vertex_common

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_color;
out float v_alpha;

void main() {
    vec3 ip = instance_point(position);
    vec4 wp = model * vec4(ip, 1.0);
    wp.xyz = apply_sway(wp.xyz, ip.y, sway);
    if (sway.w > 0.0) {
        wp.xyz = apply_push(wp.xyz, (model * vec4(instance_point(color0.xyz), 1.0)).xyz,
                            (model * vec4(inst0.w, inst1.w, inst2.w, 1.0)).xz, pusher, sway.w);
    }
    v_world_pos = wp.xyz;
    v_normal = (normal_mat * vec4(instance_dir(normal), 0.0)).xyz;
    v_uv = uv * uv_transform.xy + uv_transform.zw;
    v_color = color0.rgb;
    v_alpha = color0.a;
    gl_Position = view_proj * wp;
}
@end

@vs shadow_vs
layout(binding=0) uniform shadow_vs_params {
    mat4 s_view_proj;
    mat4 s_model;
    vec4 s_uv_transform;
    vec4 s_sway;
    vec4 s_pusher;
};

@include_block vertex_common

out vec2 v_uv;

void main() {
    vec3 ip = instance_point(position);
    vec4 wp = s_model * vec4(ip, 1.0);
    wp.xyz = apply_sway(wp.xyz, ip.y, s_sway);
    if (s_sway.w > 0.0) {
        wp.xyz = apply_push(wp.xyz, (s_model * vec4(instance_point(color0.xyz), 1.0)).xyz,
                            (s_model * vec4(inst0.w, inst1.w, inst2.w, 1.0)).xz, s_pusher, s_sway.w);
    }
    v_uv = uv * s_uv_transform.xy + s_uv_transform.zw;
    // keep the unused attributes alive so both programs share one vertex layout
    v_uv += (normal.xy + color0.xy) * 0.0;
    gl_Position = s_view_proj * wp;
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

@fs fs
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
    vec4 env_params;   // x = has environment map, y = intensity, z = darkness of the viewer in reflections,
                       // w = meters that alpha 1 of the map stands for (0: it holds no distances)
    vec4 env_pos;      // where the map was captured, w = 1 for box projection
    vec4 env_box_min;  // the room the map shows, for box projection
    vec4 env_box_max;
    vec4 screen_params; // xy = 1 / render target size (screen-space map), z = LOD cross-fade (0..1),
                        //   w = 1 fading in, -1 fading out, 0 none
    vec4 cull_min;      // w = 1: fragments inside cull_min..cull_max are not drawn
    vec4 cull_max;
    vec4 camera_pos;   // w = tone mapping exposure (0 = no tone mapping)
    vec4 ambient;      // summed ambient light
    vec4 hemi_sky;
    vec4 hemi_ground;
    vec4 gi_min;       // irradiance volume: its box, w = 1 when there is one, 2 when it replaces the ambient light
    vec4 gi_inv_size;  // w = intensity
    vec4 gi_dims;      // probes along x, y, z, w = lookup offset along the normal (m)
    vec4 fog_color;    // w = how strongly lights glow in the fog
    vec4 fog_params;   // x = near, y = far, z = density, w = mode (0 none, 1 linear, 2 exp2, 3 exp)
    vec4 fog_box_min;  // fog-free box, w = 1 when enabled
    vec4 fog_box_max;
    vec4 fog_height;   // x = base height, y = falloff per meter (0 = none), w = density away from the sun's side
    vec4 fog_sun_dir;  // towards the sun, w = sharpness (0 = no sun)
    vec4 fog_sun_color; // w = light shafts (0..1)
    vec4 counts;       // x = directional lights, y = point lights, z = spot lights
    vec4 dir_dir[4];   // world-space direction towards the light
    vec4 dir_color[4];
    vec4 point_pos[6];   // w = distance cutoff
    vec4 point_color[6]; // w = decay
    vec4 point_params[6]; // x = source radius
    vec4 spot_pos[8];    // w = distance cutoff
    vec4 spot_dir[8];    // direction the light points to, w = cos(angle)
    vec4 spot_color[8];  // w = decay
    vec4 spot_params[8]; // x = cos(inner angle) (from penumbra), y = source radius, z = specular radius,
                         //   w = how much the light glows in the fog
    mat4 shadow_mat[8];    // world -> shadow map uv + depth, per spot light
    vec4 shadow_params[8]; // x = 0: none, 1: in shadow_map, 2: in shadow_atlas, y = normal bias (m),
                           //   z = texel size, w = blur radius in texels
    vec4 shadow_rect[8];   // the light's tile in shadow_atlas: xy = offset, zw = size
    vec4 shadow_soft[8];   // x = size of the source (m; 0: edges equally sharp everywhere), y = near,
                           //   z = far, w = width of the map 1 m from the light (negative: depth is -1..1)
    mat4 dir_shadow_mat[4]; // world -> uv in its tile + depth, per cascade of the directional light's shadow
    vec4 dir_shadow_params; // x = which directional light + 1 (0: none), y = normal bias (m),
                            //   z = texel size of sun_shadow, w = blur radius in texels
    vec4 dir_cascade_rect[4];  // each cascade's tile in sun_shadow: xy = offset, zw = size
    vec4 dir_cascade_info[4];  // per cascade: x = view depth (m) where it ends, y = size of a texel (m)
    vec4 dir_cascade_params;   // x = cascades, y = blend band (part of a cascade), z = 1: tint them,
                               //   w = 1: one box (no cascades), faded at its edge
    vec4 dir_cascade_fwd;      // the camera's forward direction
    vec4 grime_params;     // x = dirt on the surface (0: none), y = the room's floor, z = its height (m),
                           //   w = seed
    mat4 ssr_view_proj;    // world -> the last frame's clip space
    vec4 ssr_params;       // x = this material's screen-space reflections (0: none), y = reach (m),
                           //   z = noise seed, w = 1: the last frame's row 0 is its top
};

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in vec3 v_color;
in float v_alpha;

out vec4 frag_color;

const float PI = 3.14159265359;
const float RECIPROCAL_PI = 0.31830988618;

vec3 linear_to_srgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

// ACES filmic tone mapping, same fit as three.js's ACESFilmicToneMapping.
vec3 rrt_odt_fit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

vec3 aces_filmic(vec3 c) {
    const mat3 aces_in = mat3(
        vec3(0.59719, 0.07600, 0.02840),
        vec3(0.35458, 0.90834, 0.13383),
        vec3(0.04823, 0.01566, 0.83777));
    const mat3 aces_out = mat3(
        vec3(1.60475, -0.10208, -0.00327),
        vec3(-0.53108, 1.10813, -0.07276),
        vec3(-0.07367, -0.00605, 1.07602));
    c *= 1.0 / 0.6;
    c = aces_in * c;
    c = rrt_odt_fit(c);
    c = aces_out * c;
    return clamp(c, 0.0, 1.0);
}

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
    return len * (1.0 - max(exit - enter, 0.0));
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

// 1 where the light's map has nothing nearer than depth `z` at `uv`.
float shadow_tap(bool tiled, vec4 rect, float texel, vec2 uv, float z) {
    if (tiled) {
        // stay half a texel inside the tile: the neighbours belong to other lights
        uv = rect.xy + clamp(uv, vec2(texel), vec2(1.0 - texel)) * rect.zw;
        return texture(sampler2DShadow(shadow_atlas, shadow_smp), vec3(uv, z));
    }
    return texture(sampler2DShadow(shadow_map, shadow_smp), vec3(uv, z));
}

// Shadow of spot light `i`. Lights whose map is redrawn every frame have shadow_map to
// themselves; the ones drawn once share shadow_atlas, a tile each.
float shadow(int i, vec3 n) {
    vec4 params = shadow_params[i];
    if (params.x < 0.5 || (flags() & 4) == 0) return 1.0;
    vec3 to_light = spot_pos[i].xyz - v_world_pos;
    float dist = length(to_light);
    // push the lookup off the surface, more at grazing angles and far away where texels are big
    float slope = 1.0 - clamp(dot(n, to_light / dist), 0.0, 1.0);
    vec3 p = v_world_pos + n * params.y * dist * (0.4 + slope);
    vec4 sp = shadow_mat[i] * vec4(p, 1.0);
    if (sp.w <= 0.0) return 1.0;
    vec3 c = sp.xyz / sp.w;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0 || c.z > 1.0) return 1.0;
    // rotated disk of taps for a soft edge
    float angle = fract(sin(dot(gl_FragCoord.xy + ssr_params.z * vec2(5.0, 3.0), vec2(12.9898, 78.233))) * 43758.5453) * 6.2831;
    float ca = cos(angle);
    float sa = sin(angle);
    float r = params.z * params.w;
    bool tiled = params.x > 1.5;
    vec4 rect = shadow_rect[i];
    vec4 soft = shadow_soft[i];
    int taps = 8;
    // how the depth changes over the map along this surface, so that a wide blur does not
    // run the surface into its own shadow
    vec2 grad = vec2(0.0);
    if (soft.x > 0.0) {
        // A source with a size: the edge of a shadow is as wide as the source looks from there
        // past the thing that casts it. Sharp where the two touch, wide far behind. How far the
        // caster is comes from asking the map at a few depths, nearer each time.
        float dr = shadow_distance(c.z, soft);
        float per_m = 1.0 / (abs(soft.w) * dr);
        float search = clamp(soft.x * 1.6 * per_m, r, 0.04);
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
        taps = 14;
        vec3 dx = dFdx(c);
        vec3 dy = dFdy(c);
        float det = dx.x * dy.y - dx.y * dy.x;
        if (abs(det) > 1e-12) grad = vec2(dy.y * dx.z - dx.y * dy.z, dx.x * dy.z - dy.x * dx.z) / det;
    }
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
float dir_shadow(vec3 n, vec3 l) {
    if ((flags() & 4) == 0) return 1.0;
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
    return all_fog > 1e-5 ? clamp(lit / all_fog, 0.0, 1.0) : 1.0;
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

vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
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

float ssr_depth_at(vec2 uv) {
    ivec2 size = textureSize(sampler2D(ssr_depth, ssr_smp), 0);
    return texelFetch(sampler2D(ssr_depth, ssr_smp), clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1), 0).g;
}

// Screen-space reflection: follows the mirrored view ray through the last frame's depth and
// returns what it runs into there (linear, as the camera saw it); `found` is 0 when it leaves
// the picture or hits nothing. A rough surface sends its one ray somewhere in a wider cone,
// which shows as noise until something downstream (a temporal filter, a tape) evens it out.
vec3 screen_reflection(vec3 n, vec3 geo_n, vec3 v, float rough, out float found) {
    found = 0.0;
    vec2 seed = gl_FragCoord.xy + ssr_params.z * vec2(47.0, 17.0);
    float n1 = ssr_noise(seed);
    float n2 = ssr_noise(seed + vec2(31.7, 11.3));
    vec3 r = reflect(-v, n);
    vec3 t1 = normalize(cross(r, abs(r.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
    vec3 t2 = cross(r, t1);
    float turn = n1 * 2.0 * PI;
    // (a quarter of the cone by sending the ray astray, the rest by blurring what it finds:
    // all of it by chance makes a small bright thing into a patch of sparks)
    r = normalize(r + (t1 * cos(turn) + t2 * sin(turn)) * sqrt(n2) * rough * rough * 0.125);
    if (dot(r, geo_n) < 0.02) return vec3(0.0);

    vec3 origin = v_world_pos + geo_n * 0.02;
    float step_len = 0.06;
    float t = step_len * (0.4 + n2);
    float t_prev = 0.0;
    for (int i = 0; i < 28; i++) {
        vec3 s = ssr_project(origin + r * t);
        if (s.z < 0.05 || s.x < 0.0 || s.y < 0.0 || s.x > 1.0 || s.y > 1.0) break;
        float behind = s.z - ssr_depth_at(s.xy);
        // behind what the camera saw there, and not by more than something of that size is thick
        if (behind > 0.02 && behind < step_len * 1.5 + 0.15) {
            float lo = t_prev;
            float hi = t;
            for (int k = 0; k < 4; k++) {
                float mid = 0.5 * (lo + hi);
                vec3 m = ssr_project(origin + r * mid);
                if (m.z - ssr_depth_at(m.xy) > 0.02) hi = mid; else lo = mid;
            }
            s = ssr_project(origin + r * hi);
            vec2 edge = min(s.xy, 1.0 - s.xy);
            found = smoothstep(0.0, 0.07, min(edge.x, edge.y)) * (1.0 - smoothstep(0.6, 1.0, hi / ssr_params.y));
            // what a rough surface mirrors is spread over a disc that grows with the way the
            // ray went: a few taps over it, turned differently at every pixel
            vec2 size = vec2(textureSize(sampler2D(ssr_color, ssr_smp), 0));
            float spread = rough * rough * 0.4 * hi / max(s.z, 0.1);
            vec2 radius = vec2(spread * size.y / size.x, spread);
            vec3 sum = vec3(0.0);
            for (int k = 0; k < 8; k++) {
                float a = turn + float(k) * 2.39996;
                vec2 at = s.xy + vec2(cos(a), sin(a)) * radius * sqrt((float(k) + 0.5) / 8.0);
                vec4 c = textureLod(sampler2D(ssr_color, ssr_smp), at, 0.0);
                // back from what the tone mapping made of it, roughly; light sources were far over white
                vec3 lin = srgb_to_linear(c.rgb);
                lin /= max(1.0 - 0.7 * max(lin.r, max(lin.g, lin.b)), 0.3);
                sum += lin * (1.0 + 5.0 * (1.0 - c.a));
            }
            return sum / 8.0 / max(camera_pos.w, 0.05);
        }
        t_prev = t;
        step_len *= 1.16;
        t += step_len;
        if (t > ssr_params.y) break;
    }
    return vec3(0.0);
}

// Tangent-space normal mapping without vertex tangents (cotangent frame from screen derivatives).
vec3 perturb_normal(vec3 n, float scale) {
    vec3 t = texture(sampler2D(tex_normal, tex_smp), v_uv).xyz * 2.0 - 1.0;
    t.xy *= scale;
    vec3 dp1 = dFdx(v_world_pos);
    vec3 dp2 = dFdy(v_world_pos);
    vec2 duv1 = dFdx(v_uv);
    vec2 duv2 = dFdy(v_uv);
    vec3 dp2perp = cross(dp2, n);
    vec3 dp1perp = cross(n, dp1);
    vec3 tangent = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 bitangent = dp2perp * duv1.y + dp1perp * duv2.y;
    float invmax = inversesqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1e-12));
    return normalize(tangent * invmax * t.x + bitangent * invmax * t.y + n * t.z);
}

// Equirectangular environment map lookup, blurred with roughness through its mip levels.
// With a box set, the reflected ray from the surface point is traced to the room's walls and
// the map is read towards that hit, so nearby walls keep their size and place.
vec2 env_uv(vec3 d) {
    return vec2(atan(d.z, d.x) * (0.5 * RECIPROCAL_PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) * RECIPROCAL_PI);
}

// How far the point `q` (relative to where the map was captured) lies behind what the map saw
// in its direction.
float env_behind(vec3 q) {
    float len = length(q);
    return len - textureLod(sampler2D(tex_env, env_smp), env_uv(q / len), 0.0).a * env_params.w;
}

vec3 env_sample(vec3 d, float rough) {
    if (env_pos.w > 0.5) {
        vec3 inv = 1.0 / d;
        vec3 t_far = max((env_box_max.xyz - v_world_pos) * inv, (env_box_min.xyz - v_world_pos) * inv);
        float t = max(min(min(t_far.x, t_far.y), t_far.z), 0.0);
        vec3 o = v_world_pos - env_pos.xyz;
        if (env_params.w > 0.0 && rough < 0.3) {
            // A map with distances: walk the ray through the room as the map saw it, and stop
            // where it goes behind something (not far behind: then it only passes at the back
            // of it, out of the map's sight).
            const int STEPS = 72;
            float dt = t / float(STEPS);
            // the distances come in steps of 1/255 of their range: nearer than that to a
            // surface, a ray is not yet behind it (or a wall seen at a glancing angle breaks
            // up into rings, one per step)
            float slack = env_params.w / 255.0 * 1.5;
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
    return textureLod(sampler2D(tex_env, env_smp), env_uv(d), rough * 7.0).rgb * env_params.y;
}

float cloud_hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Smooth noise, 0..1.
float cloud_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(cloud_hash(i), cloud_hash(i + vec2(1.0, 0.0)), f.x),
               mix(cloud_hash(i + vec2(0.0, 1.0)), cloud_hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

float cloud_fbm(vec2 p) {
    return cloud_noise(p) * 0.55 + cloud_noise(p * 2.3 + 5.1) * 0.3 + cloud_noise(p * 5.9 + 11.7) * 0.15;
}

// Where water has come through above a room: 0 dry .. 1 soaked, over the plan of the house.
// The ceiling, the walls under it and the floor all read the same places from it.
float grime_leak(vec2 xz) {
    return smoothstep(0.5, 0.66, cloud_fbm(xz * 0.42 + 5.5));
}

// Years of neglect on the surfaces of a room (material.grime), worked out from where the
// surface is. Water has come through the ceiling in places and left yellowed patches there;
// under those it has run down the walls, as far as there was water, the paint has lifted in
// the damp, and the floor below is marked. Apart from that, walls are dirty along the floor.
void apply_grime(vec3 n, inout vec3 albedo, inout float rough) {
    vec3 p = v_world_pos;
    float h = (p.y - grime_params.y) / grime_params.z;
    float seed = grime_params.w;
    vec3 tint = vec3(0.4, 0.29, 0.16);
    float wet = grime_leak(p.xz);
    float dirt;
    if (abs(n.y) < 0.6) {
        bool along_z = abs(n.x) > abs(n.z);
        float u = (along_z ? p.z + 17.3 : p.x) + seed;
        vec2 q = vec2(u, p.y);
        float down = 1.0 - h;
        // how far down the water got: further where there was more of it, in uneven tongues
        float reach = wet * (0.25 + 0.95 * cloud_noise(vec2(u * 0.8, 4.0)))
                    + wet * (cloud_noise(vec2(u * 6.0, 2.0)) - 0.5) * 0.35;
        float damp = reach > 0.02 ? 1.0 - smoothstep(reach * 0.35, reach, down) : 0.0;
        // it dried from the edge inwards: the lower edge is the darkest
        float edge = reach > 0.02 ? smoothstep(reach * 0.7, reach * 0.97, down) * (1.0 - smoothstep(reach * 0.97, reach * 1.06, down)) : 0.0;
        float trickle = 0.65 + 0.35 * cloud_noise(vec2(u * 22.0, p.y * 1.2));
        float low = (1.0 - smoothstep(0.0, 0.06 + 0.22 * cloud_noise(vec2(u * 0.9, 3.0)), h)) * (0.5 + 0.5 * cloud_noise(q * 5.0));
        float mottle = cloud_fbm(q * 0.45 + 9.0);
        dirt = damp * trickle * 0.9 + edge * 0.5 * wet + low * 0.7 + smoothstep(0.5, 0.9, mottle) * 0.15;
        // in the damp the paint has lifted and come away, in a few places
        float flake = cloud_fbm(q * 2.6 + 31.0);
        float bare = smoothstep(0.6, 0.68, flake) * smoothstep(0.35, 0.9, damp) * grime_params.x;
        float lum = dot(albedo, vec3(0.33));
        albedo = mix(albedo, vec3(0.6, 0.57, 0.5) * (0.45 + 0.55 * lum), bare * 0.7);
    } else if (n.y > 0.0) {
        vec2 q = p.xz + seed;
        dirt = wet * (0.45 + 0.25 * cloud_noise(q * 3.0)) + smoothstep(0.55, 0.9, cloud_fbm(q * 0.5)) * 0.25
             + smoothstep(0.75, 0.92, cloud_noise(q * 8.0)) * 0.15;
    } else {
        float raw = cloud_fbm(p.xz * 0.42 + 5.5);
        // yellowed where it was wet, with the darker line it dried to
        dirt = wet * (0.5 + 0.25 * cloud_noise(p.xz * 2.5)) + smoothstep(0.48, 0.51, raw) * (1.0 - smoothstep(0.51, 0.56, raw)) * 0.4;
        tint = vec3(0.5, 0.36, 0.16);
    }
    dirt = clamp(dirt * grime_params.x, 0.0, 0.85);
    albedo *= mix(vec3(1.0), tint, dirt);
    rough = mix(rough, 1.0, dirt * 0.6);
}

// How much a reflected ray from `o` along `r` hits the viewer: no figure, only a dark cloud
// hanging where the camera stands, thick in the middle, thinning out in wisps and grain.
float viewer_mask(vec3 o, vec3 r) {
    vec3 cam = camera_pos.xyz;
    float lh = length(r.xz);
    if (lh < 1e-3) return 0.0;
    vec2 rh = r.xz / lh;
    // where the ray passes the vertical plane through the camera, facing it
    float t = dot(cam.xz - o.xz, rh) / lh;
    if (t <= 0.0) return 0.0;
    vec3 p = o + r * t;
    vec2 q = vec2(dot(p.xz - cam.xz, vec2(-rh.y, rh.x)), p.y - cam.y + 0.55);
    // an upright blob about a person's size, without an edge
    float d = length(q / vec2(0.5, 0.95));
    float wisps = cloud_noise(q * 3.1) * 0.6 + cloud_noise(q * 7.3 + 4.7) * 0.3 + cloud_noise(q * 17.0 + 9.1) * 0.1;
    float grain = ssr_noise(gl_FragCoord.xy + ssr_params.z * vec2(47.0, 17.0)) - 0.5;
    float density = (1.0 - d) * 1.5 + (wisps - 0.5) * 1.1 + grain * 0.25;
    return smoothstep(0.0, 1.0, density);
}

void main() {
    int mode = int(mat_params.x);
    int fl = flags();
    bool foliage = (fl & 2) != 0;

    if (cull_min.w > 0.5 && all(greaterThan(v_world_pos, cull_min.xyz)) && all(lessThan(v_world_pos, cull_max.xyz))) discard;
    // LOD cross-fade: a screen-door dither, the incoming level where it is below the fade and
    // the outgoing one where it is not (interleaved gradient noise: no visible pattern)
    if (screen_params.w != 0.0) {
        float dither = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        if (screen_params.w > 0.0 ? dither >= screen_params.z : dither < screen_params.z) discard;
    }
    if ((fl & 32) != 0) {
        // a window onto another render: its pixels as they are, already tone mapped
        frag_color = vec4(texture(sampler2D(tex_map, tex_smp), gl_FragCoord.xy * screen_params.xy).rgb, 1.0);
        return;
    }

    g_base = vec4(color.rgb, color.a);
    bool baked = (fl & 16) != 0;
    if ((fl & 1) != 0 && !baked) g_base.rgb *= v_color;
    if (tex_params.x > 0.5) g_base *= texture(sampler2D(tex_map, tex_smp), v_uv);
    float alpha = g_base.a;
    if ((fl & 1) != 0) alpha *= v_alpha;
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

    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing && !foliage) n = -n;
    vec3 geo_n = n;
    if (tex_params.y != 0.0 && mode != 0) n = perturb_normal(n, tex_params.y);

    g_rough = mat_params.y;
    g_metal = mat_params.z;
    float ao = 1.0;
    if (tex_params.z > 0.5) {
        vec3 arm = texture(sampler2D(tex_arm, tex_smp), v_uv).rgb;
        ao = arm.r;
        g_rough *= arm.g;
        g_metal *= arm.b;
    }

    if (grime_params.x > 0.0 && mode != 0) apply_grime(geo_n, g_base.rgb, g_rough);

    vec3 outc;
    vec3 spec = vec3(0.0);
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
        float gi_fade;
        vec3 gi = gi_irradiance(n, gi_fade);
        vec3 irradiance = gi_min.w > 1.5 ? mix(sky_light, gi, gi_fade) : sky_light + gi * gi_fade;
        outc = irradiance * diffuse_color * RECIPROCAL_PI * ao;

        int ndir = int(counts.x);
        for (int i = 0; i < 4; i++) {
            if (i >= ndir) break;
            g_radius = 0.0;
            vec3 l = normalize(dir_dir[i].xyz);
            float lit = int(dir_shadow_params.x) == i + 1 ? dir_shadow(n, l) : 1.0;
            if (lit > 0.0) outc += direct(mode, n, v, l, 1e6, dir_color[i].rgb * lit, diffuse_color, spec);
        }
        if (dir_cascade_params.z > 0.5 && dir_shadow_params.x > 0.5) {
            // renderer.debug_cascades: red, green, blue, yellow
            int k = sun_cascade(dot(v_world_pos - camera_pos.xyz, dir_cascade_fwd.xyz));
            if (k >= 0) outc = mix(outc, k == 0 ? vec3(1.0, 0.0, 0.0) : (k == 1 ? vec3(0.0, 1.0, 0.0) : (k == 2 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 1.0, 0.0))), 0.5);
        }
        int npoint = int(counts.y);
        for (int i = 0; i < 6; i++) {
            if (i >= npoint) break;
            vec3 lv = point_pos[i].xyz - v_world_pos;
            float d = length(lv);
            // a light source with a size: falloff softens up close instead of blowing out
            g_radius = point_params[i].x;
            float d_soft = sqrt(d * d + g_radius * g_radius);
            vec3 radiance = point_color[i].rgb * distance_attenuation(d_soft, point_pos[i].w, point_color[i].w);
            radiance *= fog_transmittance(fog_distance(point_pos[i].xyz, v_world_pos));
            outc += direct(mode, n, v, lv / d, d, radiance, diffuse_color, spec);
        }
        int nspot = int(counts.z);
        for (int i = 0; i < 8; i++) {
            if (i >= nspot) break;
            vec3 lv = spot_pos[i].xyz - v_world_pos;
            float d = length(lv);
            // out of the light's reach
            if (spot_pos[i].w > 0.0 && d > spot_pos[i].w) continue;
            vec3 l = lv / d;
            float cone = smoothstep(spot_dir[i].w, spot_params[i].x, dot(-l, spot_dir[i].xyz));
            if (cone <= 0.0) continue;
            g_radius = spot_params[i].y;
            float d_soft = sqrt(d * d + g_radius * g_radius);
            g_radius = spot_params[i].z;
            vec3 radiance = spot_color[i].rgb * cone * distance_attenuation(d_soft, spot_pos[i].w, spot_color[i].w);
            radiance *= fog_transmittance(fog_distance(spot_pos[i].xyz, v_world_pos));
            radiance *= shadow(i, geo_n);
            outc += direct(mode, n, v, l, d, radiance, diffuse_color, spec);
        }
        spec *= ao;
        outc += emissive.rgb;
        // light baked into the vertex colors, as if from fixed lamps
        if (baked) outc += diffuse_color * v_color * emissive.w * ao;
    }

    if ((fl & 8) != 0) {
        // glass: reflections stay visible however clear the pane is
        float fresnel = 0.04 + 0.96 * pow(1.0 - clamp(dot(n, v), 0.0, 1.0), 5.0);
        vec3 sky = mix(hemi_ground.rgb, hemi_sky.rgb, 0.5 * reflect(-v, n).y + 0.5) + ambient.rgb;
        if (env_params.x > 0.5) sky = env_sample(reflect(-v, n), g_rough);
        if (env_params.z > 0.0) sky *= 1.0 - env_params.z * viewer_mask(v_world_pos, reflect(-v, n));
        spec += sky * fresnel;
        float spec_a = max(spec.r, max(spec.g, spec.b));
        float a = clamp(alpha + fresnel + spec_a, 0.0, 1.0);
        // blended with src-alpha: pre-divide so that outc * alpha + spec comes out
        outc = (outc * alpha + spec) / max(a, 1e-4);
        alpha = a;
    } else {
        if (env_params.x > 0.5 && mode != 1) {
            // reflections of the surroundings, fading as the surface gets rougher
            float nov = clamp(dot(n, v), 0.0, 1.0);
            vec3 f0 = mix(vec3(0.04), g_base.rgb, g_metal);
            vec3 fr = f0 + (max(vec3(1.0 - g_rough), f0) - f0) * pow(1.0 - nov, 5.0);
            outc += env_sample(reflect(-v, n), g_rough) * fr;
        }
        if (ssr_params.x > 0.0) {
            // the surface's sheen: a mirror image, strongest at a glancing angle
            float found;
            // (far off, where a pixel covers many bumps of the normal map, each would throw
            // the image somewhere else: blotches. There the surface mirrors as if flat)
            float bumps = 0.5 * (1.0 - smoothstep(1.5, 5.0, length(camera_pos.xyz - v_world_pos)));
            vec3 rn = normalize(mix(geo_n, n, bumps));
            vec3 mirrored = screen_reflection(rn, geo_n, v, g_rough, found);
            float fresnel = 0.04 + 0.96 * pow(1.0 - clamp(dot(rn, v), 0.0, 1.0), 5.0);
            outc += mirrored * (found * fresnel * ssr_params.x * ao);
        }
        outc += spec;
    }

    int fog_mode = int(fog_params.w);
    if (fog_mode != 0) {
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
            int npoint = int(counts.y);
            for (int i = 0; i < 6; i++) {
                if (i >= npoint) break;
                glow += point_color[i].rgb * fog_glow(point_pos[i].xyz, point_params[i].x, ray, depth, vec4(0.0), -2.0);
            }
            int nspot = int(counts.z);
            for (int i = 0; i < 8; i++) {
                if (i >= nspot) break;
                if (spot_params[i].w <= 0.0) continue;
                glow += spot_color[i].rgb * spot_params[i].w
                    * fog_glow(spot_pos[i].xyz, spot_params[i].y, ray, depth, spot_dir[i], spot_params[i].x);
            }
            fog_lit += glow * fog_color.w * in_fog;
        }
        outc = outc * t + fog_lit;
    }

    outc = max(outc, vec3(0.0));
    if (camera_pos.w > 0.0) outc = aces_filmic(outc * camera_pos.w);
    // a light source (emissive far over white) leaves the target's alpha at 0, for
    // post-processing that has to tell a burnt-out lamp from a brightly lit wall
    alpha *= 1.0 - smoothstep(4.0, 16.0, max(emissive.r, max(emissive.g, emissive.b)));
    frag_color = vec4(linear_to_srgb(outc), alpha);
}
@end

@program three vs fs
@program shadow shadow_vs shadow_fs
