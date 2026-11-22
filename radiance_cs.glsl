// The kernels of RadianceCascades (radiance.c3): world-space radiance cascades (Sannikov's)
// on a grid of probes, traced through a voxel copy of the scene.
// just shaders turns this into generated/radiance_cs.c3.
//
//   rc_trace_cs       one thread per probe and direction of one cascade: marches its ray over
//                     the cascade's stretch of distance through the voxels; what it hits gives
//                     off its own light and what the lights and the last frame's bounce put on
//                     it; a ray that hits nothing goes on in the cascade above (merged from
//                     its 8 probes around and 4 directions), or ends in the sky. Run from the
//                     top cascade down.
//   rc_gather_cs      one thread per probe of cascade 0: the light arriving on surfaces facing
//                     each of the six axes (what the material shader and the next frame's
//                     bounce read).
@header module drie::shaders;
@header import drie;

@ctype vec4 Vec4

@block rc_common
layout(binding=0) uniform rc_params {
    vec4 rc_min;      // xyz = the box's corner, w = cascade 0's probe spacing (m)
    ivec4 rc_dims;    // xyz = cascade 0's probes, w = the cascade being traced
    vec4 rc_tune;     // x = cascade 0's reach (m), y = the new frame's share, z = frame, w = voxel size (m)
    ivec4 rc_info;    // x = directions across cascade 0's square, y = cascades, z = lights, w = rays of this cascade
    vec4 rc_sky;      // rgb = the sky's radiance overhead, w = the bounce's strength
    vec4 rc_ground;   // rgb = the sky's radiance from below
    ivec4 rc_vox;     // xyz = voxels, w = one ray in this many is traced each frame
    ivec4 rc_off0;    // where cascades 0..3 start in rc_rad
    ivec4 rc_off1;    // 4..7
};
struct rc_voxel {
    uint albedo;      // rgb8, bit 24 = solid
    uint emissive;    // rgb8 mantissas and a shared exponent (bits 24..31, 0 = none)
};
struct rc_texel {
    vec4 v;
};
struct rc_rad_texel {
    uvec2 p;          // rgb packed as half2 + half2
};
vec3 rc_unpack_rad(uvec2 p) {
    vec2 rg = unpackHalf2x16(p.x);
    vec2 ba = unpackHalf2x16(p.y);
    return vec3(rg, ba.x);
}
uvec2 rc_pack_rad(vec3 rgb) {
    return uvec2(packHalf2x16(rgb.rg), packHalf2x16(vec2(rgb.b, 0.0)));
}
struct rc_brick {
    uint solid;       // the solid voxels in a block of 4 x 4 x 4 (a cell of cascade 0)
};
layout(binding=0) readonly buffer rc_voxels { rc_voxel rc_vox_data[]; };
layout(binding=4) readonly buffer rc_bricks { rc_brick rc_brick_data[]; };

const float RC_PI = 3.14159265;

bool rc_in_grid(ivec3 c) {
    return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, rc_vox.xyz));
}

int rc_voxel_index(ivec3 c) {
    return (c.z * rc_vox.y + c.y) * rc_vox.x + c.x;
}

bool rc_solid(ivec3 c) {
    return (rc_vox_data[rc_voxel_index(c)].albedo & 0x1000000u) != 0u;
}

// the direction a point of the octahedral square (-1..1) stands for
vec3 rc_direction(vec2 e) {
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.xy += vec2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}

ivec3 rc_level_dims(int level) {
    return max((rc_dims.xyz + ((1 << level) - 1)) >> level, ivec3(1));
}

int rc_level_offset(int level) {
    return level < 4 ? rc_off0[level] : rc_off1[level - 4];
}
@end

@cs rc_trace_cs
@include_block rc_common
struct rc_light {
    vec4 pos;     // w = distance cutoff; a directional light: xyz = towards it
    vec4 color;   // w = decay
    vec4 dir;     // where a spot light points, w = cos(angle); w = -2: a point light, -3: directional
    vec4 params;  // x = cos(inner angle)
};
layout(binding=1) buffer rc_cascades { rc_rad_texel rc_rad[]; };
layout(binding=2) readonly buffer rc_irradiance { rc_texel rc_irr[]; };
layout(binding=3) readonly buffer rc_lights { rc_light rc_light_data[]; };

layout(local_size_x=64) in;

// Follows the ray through the voxels for up to `reach` meters. 0: nothing in the way, 1: hit
// (t, the voxel and the face it came in through), 2: left the box, 3: started inside something.
// Blocks of 4 x 4 x 4 voxels with nothing in them are crossed in one step.
int rc_march(vec3 ro, vec3 rd, float reach, out float t_hit, out ivec3 cell, out vec3 face) {
    float vs = rc_tune.w;
    vec3 g = (ro - rc_min.xyz) / vs;
    float reach_v = reach / vs;
    ivec3 c = ivec3(floor(g));
    t_hit = 0.0;
    cell = c;
    face = vec3(0.0);
    if (!rc_in_grid(c)) return 2;
    if (rc_solid(c)) return 3;
    vec3 inv = 1.0 / max(abs(rd), vec3(1e-6));
    ivec3 stp = ivec3(rd.x >= 0.0 ? 1 : -1, rd.y >= 0.0 ? 1 : -1, rd.z >= 0.0 ? 1 : -1);
    vec3 ahead = vec3(greaterThan(stp, ivec3(0)));
    // how far along the ray (in voxels) the next voxel starts, on each axis
    vec3 next = abs(vec3(c) + ahead - g) * inv;
    for (int i = 0; i < 400; i++) {
        ivec3 bc = c >> 2;
        int axis;
        float t;
        if (rc_brick_data[(bc.z * rc_dims.y + bc.y) * rc_dims.x + bc.x].solid == 0u) {
            vec3 leave = abs(vec3(bc * 4) + ahead * 4.0 - g) * inv;
            axis = leave.x < leave.y ? (leave.x < leave.z ? 0 : 2) : (leave.y < leave.z ? 1 : 2);
            t = leave[axis];
            if (t > reach_v) return 0;
            c = clamp(ivec3(floor(g + rd * t)), bc * 4, bc * 4 + 3);
            c[axis] = bc[axis] * 4 + (stp[axis] > 0 ? 4 : -1);
            if (!rc_in_grid(c)) return 2;
            next = abs(vec3(c) + ahead - g) * inv;
        } else {
            axis = next.x < next.y ? (next.x < next.z ? 0 : 2) : (next.y < next.z ? 1 : 2);
            t = next[axis];
            if (t > reach_v) return 0;
            c[axis] += stp[axis];
            if (!rc_in_grid(c)) return 2;
            next[axis] += inv[axis];
        }
        if (rc_solid(c)) {
            t_hit = t * vs;
            cell = c;
            face[axis] = -float(stp[axis]);
            return 1;
        }
    }
    return 0;
}

float rc_attenuation(float d, float cutoff, float decay) {
    float a = 1.0 / max(pow(d, decay), 0.01);
    if (cutoff > 0.0) {
        float r = d / cutoff;
        float f = clamp(1.0 - r * r * r * r, 0.0, 1.0);
        a *= f * f;
    }
    return a;
}

// The light the last frame's probes put on a surface at p facing the axis `face`.
vec3 rc_bounce(vec3 p, vec3 face) {
    int axis = face.x != 0.0 ? (face.x > 0.0 ? 0 : 1) : (face.y != 0.0 ? (face.y > 0.0 ? 2 : 3) : (face.z > 0.0 ? 4 : 5));
    vec3 g = (p - rc_min.xyz) / rc_min.w - 0.5;
    ivec3 b = ivec3(floor(g));
    vec3 f = g - vec3(b);
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        ivec3 c = clamp(b + o, ivec3(0), rc_dims.xyz - 1);
        vec4 e = rc_irr[((c.z * rc_dims.y + c.y) * rc_dims.x + c.x) * 6 + axis].v;
        vec3 w3 = mix(1.0 - f, f, vec3(o));
        float w = w3.x * w3.y * w3.z * e.w;
        sum += e.rgb * w;
        wsum += w;
    }
    return wsum > 1e-4 ? sum / wsum : vec3(0.0);
}

vec3 rc_sky_at(vec3 rd) {
    return mix(rc_ground.rgb, rc_sky.rgb, 0.5 * rd.y + 0.5);
}

uint rc_hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

void main() {
    // a different ray of each run of rc_vox.w every frame
    int turn = int(gl_GlobalInvocationID.x);
    int i = turn * rc_vox.w + (turn + int(rc_tune.z)) % rc_vox.w;
    if (i >= rc_info.w) return;
    int level = rc_dims.w;
    int d = rc_info.x << level;
    int dd = d * d;
    int probe = i / dd;
    int k = i - probe * dd;
    ivec3 dims = rc_level_dims(level);
    ivec3 pc = ivec3(probe % dims.x, (probe / dims.x) % dims.y, probe / (dims.x * dims.y));
    float spacing = rc_min.w * float(1 << level);
    vec3 pos = rc_min.xyz + (vec3(pc) + 0.5) * spacing;

    // Low-discrepancy R2 quasi-random angular jitter per ray update (eliminates white-noise shimmer)
    uint cell_h = rc_hash(uint(i) * 2654435761u + uint(level) * 7919u);
    vec2 cell_seed = vec2(float(cell_h & 0xffffu), float(cell_h >> 16)) / 65536.0;
    float r2_step = float(int(rc_tune.z) / max(rc_vox.w, 1));
    vec2 jitter = fract(cell_seed + r2_step * vec2(0.7548776662, 0.5698402909));
    uint h = rc_hash(cell_h ^ (uint(rc_tune.z) * 362437u));
    ivec2 uv = ivec2(k % d, k / d);
    vec3 rd = rc_direction((vec2(uv) + jitter) / float(d) * 2.0 - 1.0);

    // the cascade's stretch of the ray: each reaches 4 times as far as the one below
    float base = rc_tune.x;
    float t0 = base * (pow(4.0, float(level)) - 1.0) / 3.0;
    float t1 = base * (pow(4.0, float(level + 1)) - 1.0) / 3.0;

    float t_hit;
    ivec3 cell;
    vec3 face;
    float t_start = t0;
    int what = rc_march(pos + rd * t_start, rd, t1 - t_start, t_hit, cell, face);
    // If an upper-cascade ray starts inside a wall while the probe itself is in open air,
    // back up slightly so it hits the wall's lit front face instead of returning black.
    if (what == 3 && level > 0) {
        ivec3 pvc = ivec3(floor((pos - rc_min.xyz) / rc_tune.w));
        if (rc_in_grid(pvc) && !rc_solid(pvc)) {
            t_start = max(t0 - spacing * 0.75, 0.0);
            what = rc_march(pos + rd * t_start, rd, t1 - t_start, t_hit, cell, face);
        }
    }
    vec4 result = vec4(0.0);
    if (what == 1) {
        rc_voxel vx = rc_vox_data[rc_voxel_index(cell)];
        vec3 albedo = vec3(float(vx.albedo & 0xffu), float((vx.albedo >> 8) & 0xffu), float((vx.albedo >> 16) & 0xffu)) / 255.0;
        vec3 light = vec3(0.0);
        uint e = vx.emissive >> 24;
        if (e != 0u) {
            light = vec3(float(vx.emissive & 0xffu), float((vx.emissive >> 8) & 0xffu), float((vx.emissive >> 16) & 0xffu))
                  / 255.0 * exp2(float(e) - 128.0);
        }
        // just outside the face the ray came in through
        vec3 hp = pos + rd * (t_start + t_hit) + face * (0.5 * rc_tune.w);
        vec3 arriving = rc_bounce(hp, face) * rc_sky.w;
        // Deterministically shadow-test the first 4 contributing lights (zero light-selection flicker);
        // fall back to RIS only for any extra lights beyond 4.
        int nl = rc_info.z;
        int direct_tested = 0;
        float total = 0.0;
        vec3 picked = vec3(0.0);
        float picked_w = 0.0;
        vec3 picked_dir = vec3(0.0);
        float picked_dist = 0.0;
        uint seed = h;
        for (int l = 0; l < nl; l++) {
            rc_light lt = rc_light_data[l];
            bool sun = lt.dir.w < -2.5;
            vec3 lv = lt.pos.xyz - hp;
            float dist = length(lv);
            if (!sun && lt.pos.w > 0.0 && dist > lt.pos.w) continue;
            vec3 ld = lv / max(dist, 1e-4);
            // (a directional light: its shadow ray goes on out of the box)
            if (sun) {
                ld = lt.pos.xyz;
                dist = 1e4;
            }
            float ndl = dot(face, ld);
            if (ndl <= 0.0) continue;
            float cone = 1.0;
            if (lt.dir.w > -1.5) {
                cone = smoothstep(lt.dir.w, lt.params.x, dot(-ld, lt.dir.xyz));
                if (cone <= 0.0) continue;
            }
            vec3 gives = lt.color.rgb * (ndl * cone * (sun ? 1.0 : rc_attenuation(dist, lt.pos.w, lt.color.w)));
            float w = max(gives.r, max(gives.g, gives.b));
            if (w <= 0.0) continue;
            if (direct_tested < 4) {
                direct_tested++;
                float st;
                ivec3 sc;
                vec3 sf;
                if (rc_march(hp, ld, dist - rc_tune.w, st, sc, sf) != 1) arriving += gives;
            } else {
                total += w;
                seed = rc_hash(seed + uint(l) * 747796405u + 1u);
                if (float(seed >> 8) / 16777216.0 * total < w) {
                    picked = gives;
                    picked_w = w;
                    picked_dir = ld;
                    picked_dist = dist;
                }
            }
        }
        if (picked_w > 0.0) {
            float st;
            ivec3 sc;
            vec3 sf;
            if (rc_march(hp, picked_dir, picked_dist - rc_tune.w, st, sc, sf) != 1) arriving += picked * (total / picked_w);
        }
        light += albedo * arriving / RC_PI;
        result = vec4(light, 0.0);
    } else if (what == 2) {
        result = vec4(rc_sky_at(rd), 0.0);
    } else if (what == 0) {
        if (level + 1 >= rc_info.y) {
            result = vec4(rc_sky_at(rd), 0.0);
        } else {
            // Merge with the 8 probes of the cascade above, rejecting solid probes and probes
            // separated by a solid wall so coarse blocks and light leaks cannot contaminate `pos`.
            int up = level + 1;
            ivec3 udims = rc_level_dims(up);
            int ud = d * 2;
            int uoff = rc_level_offset(up);
            float usp = spacing * 2.0;
            vec3 g = (pos - rc_min.xyz) / usp - 0.5;
            ivec3 b = ivec3(floor(g));
            vec3 f = g - vec3(b);
            vec3 sum = vec3(0.0);
            float wsum = 0.0;
            vec3 sum_any = vec3(0.0);
            float wsum_any = 0.0;
            for (int n = 0; n < 8; n++) {
                ivec3 o = ivec3(n & 1, (n >> 1) & 1, (n >> 2) & 1);
                ivec3 c = clamp(b + o, ivec3(0), udims - 1);
                vec3 up_pos = rc_min.xyz + (vec3(c) + 0.5) * usp;
                ivec3 uvc = ivec3(floor((up_pos - rc_min.xyz) / rc_tune.w));
                if (rc_in_grid(uvc) && rc_solid(uvc)) continue;
                int at = uoff + ((c.z * udims.y + c.y) * udims.x + c.x) * ud * ud;
                vec3 four = rc_unpack_rad(rc_rad[at + (uv.y * 2) * ud + uv.x * 2].p)
                          + rc_unpack_rad(rc_rad[at + (uv.y * 2) * ud + uv.x * 2 + 1].p)
                          + rc_unpack_rad(rc_rad[at + (uv.y * 2 + 1) * ud + uv.x * 2].p)
                          + rc_unpack_rad(rc_rad[at + (uv.y * 2 + 1) * ud + uv.x * 2 + 1].p);
                vec3 w3 = mix(1.0 - f, f, vec3(o));
                float w = w3.x * w3.y * w3.z;
                sum_any += four * (0.25 * w);
                wsum_any += w;
                // Check two midpoints for wall occlusion between child `pos` and parent `up_pos`
                ivec3 m1 = ivec3(floor((mix(pos, up_pos, 0.35) - rc_min.xyz) / rc_tune.w));
                ivec3 m2 = ivec3(floor((mix(pos, up_pos, 0.65) - rc_min.xyz) / rc_tune.w));
                if ((rc_in_grid(m1) && rc_solid(m1)) || (rc_in_grid(m2) && rc_solid(m2))) continue;
                sum += four * (0.25 * w);
                wsum += w;
            }
            vec3 merged = wsum > 1e-4 ? sum / wsum : (wsum_any > 1e-4 ? sum_any / wsum_any : rc_sky_at(rd));
            result = vec4(merged, 0.0);
        }
    }
    result.rgb = min(result.rgb, vec3(256.0));
    int at = rc_level_offset(level) + i;
    vec3 prev = rc_unpack_rad(rc_rad[at].p);
    rc_rad[at].p = rc_pack_rad(mix(prev, result.rgb, rc_tune.y));
}
@end

@cs rc_gather_cs
@include_block rc_common
layout(binding=1) readonly buffer rc_gather_cascades { rc_rad_texel rc_rad[]; };
layout(binding=2) buffer rc_gather_irradiance { rc_texel rc_irr[]; };

layout(local_size_x=64) in;

void main() {
    int probe = int(gl_GlobalInvocationID.x);
    ivec3 dims = rc_dims.xyz;
    if (probe >= dims.x * dims.y * dims.z) return;
    ivec3 pc = ivec3(probe % dims.x, (probe / dims.x) % dims.y, probe / (dims.x * dims.y));
    vec3 pos = rc_min.xyz + (vec3(pc) + 0.5) * rc_min.w;
    ivec3 vc = ivec3(floor((pos - rc_min.xyz) / rc_tune.w));
    // a probe inside something holds no light: whoever reads the probes leaves it out
    if (!rc_in_grid(vc) || rc_solid(vc)) {
        for (int a = 0; a < 6; a++) rc_irr[probe * 6 + a].v = vec4(0.0);
        return;
    }
    int d = rc_info.x;
    vec3 sum[6];
    float wsum[6];
    for (int a = 0; a < 6; a++) {
        sum[a] = vec3(0.0);
        wsum[a] = 0.0;
    }
    for (int k = 0; k < d * d; k++) {
        vec2 e = (vec2(k % d, k / d) + 0.5) / float(d) * 2.0 - 1.0;
        vec3 flat_dir = vec3(e, 1.0 - abs(e.x) - abs(e.y));
        // the part of the sphere its cell of the square covers
        float len = length(flat_dir);
        float solid_angle = 1.0 / (len * len * len);
        vec3 dir = rc_direction(e);
        vec3 radiance = rc_unpack_rad(rc_rad[rc_off0.x + probe * d * d + k].p);
        for (int a = 0; a < 6; a++) {
            float c = dir[a >> 1] * ((a & 1) == 0 ? 1.0 : -1.0);
            // (a little of what comes in from just below the horizon: with so few directions
            // a hard cut makes the six sides differ in steps)
            float w = max(c + 0.15, 0.0) * solid_angle;
            sum[a] += radiance * w;
            wsum[a] += w;
        }
    }
    for (int a = 0; a < 6; a++) {
        rc_irr[probe * 6 + a].v = vec4(RC_PI * sum[a] / max(wsum[a], 1e-4), 1.0);
    }
}
@end

@program rc_trace rc_trace_cs
@program rc_gather rc_gather_cs
