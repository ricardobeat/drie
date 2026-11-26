// Shared Radiance Cascades building blocks included by shader_core.glsl, radiance_cs.glsl,
// radiance_rays_cs.glsl and the material shaders.
//
//   rc_key_common    32-bit spatial hash and probe key packing (shared with material shaders)
//   rc_pack_common   voxel, brick, light and fp16 radiance structs and packing helpers
//   rc_trace_common  64-bit occupancy-mask brick DDA (rc_march) and RIS direct light

@block rc_key_common
const float RC_PI = 3.14159265;

uint rc_hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// A probe's key: the cell it sits in, its cascade level, and bit 31 set so that 0 is always
// an empty slot.
uint rc_pack_key32(ivec3 c, int level) {
    return (uint(c.x) & 0x1ffu)
         | ((uint(c.y) & 0x1ffu) << 9)
         | ((uint(c.z) & 0x1ffu) << 18)
         | ((uint(level) & 0x7u) << 27)
         | 0x80000000u;
}

// The secondary cache's probes share the hash table, the meta and the free lists with the
// primary ones. Bit 30 tells them apart: the primary levels only use bits 27..29.
uint rc_pack_key32_cache(ivec3 c, int level) {
    return rc_pack_key32(c, level) | 0x40000000u;
}


uint rc_pack_xy(ivec3 c) {
    return (uint(c.x + 32768) & 0xffffu) | ((uint(c.y + 32768) & 0xffffu) << 16);
}
@end

@block rc_pack_common
struct rc_voxel {
    uint albedo;      // rgb8, bit 24 = solid
    uint emissive;    // rgb8 mantissas and a shared exponent (bits 24..31, 0 = none)
};

struct rc_brick {
    uvec2 mask;       // 64-bit occupancy mask of the 4 x 4 x 4 voxel block
};

struct rc_rad_texel {
    uvec2 p;          // rgb packed as half2 + half2
};

struct rc_light {
    vec4 pos;         // w = distance cutoff; directional: xyz = towards the light
    vec4 color;       // w = decay
    vec4 dir;         // spot direction, w = cos(outer); w = -2: point, -3: directional
    vec4 params;      // x = cos(inner)
};

vec3 rc_unpack_rad(uvec2 p) {
    vec2 rg = unpackHalf2x16(p.x);
    vec2 ba = unpackHalf2x16(p.y);
    return vec3(rg, ba.x);
}

uvec2 rc_pack_rad(vec3 rgb) {
    return uvec2(packHalf2x16(rgb.rg), packHalf2x16(vec2(rgb.b, 0.0)));
}

vec3 rc_direction(vec2 e) {
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.xy += vec2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}

vec3 rc_unpack_oct_normal(vec2 e) {
    vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
    float t = max(-n.z, 0.0);
    n.xy += vec2(n.x >= 0.0 ? -t : t, n.y >= 0.0 ? -t : t);
    return normalize(n);
}

vec3 rc_voxel_albedo(rc_voxel vx) {
    return vec3(float(vx.albedo & 0xffu), float((vx.albedo >> 8) & 0xffu), float((vx.albedo >> 16) & 0xffu)) / 255.0;
}

vec3 rc_voxel_emissive(rc_voxel vx) {
    uint e = vx.emissive >> 24;
    if (e == 0u) return vec3(0.0);
    return vec3(float(vx.emissive & 0xffu), float((vx.emissive >> 8) & 0xffu), float((vx.emissive >> 16) & 0xffu))
         / 255.0 * exp2(float(e) - 128.0);
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
@end

// The occupancy masks of the voxelized scene, for the kernels that bind rc_brick_data.
@block rc_cell_common
// Requires rc_min, rc_dims, rc_vox, rc_tune and rc_brick_data.

bool rc_in_grid(ivec3 c) {
    return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, rc_vox.xyz));
}

int rc_voxel_index(ivec3 c) {
    return (c.z * rc_vox.y + c.y) * rc_vox.x + c.x;
}

uvec2 rc_brick_mask(ivec3 bc) {
    return rc_brick_data[(bc.z * rc_dims.y + bc.y) * rc_dims.x + bc.x].mask;
}

bool rc_mask_bit(uvec2 mask, ivec3 c) {
    uint w = (c.z & 2) != 0 ? mask.y : mask.x;
    int bit = ((c.z & 1) << 4) | ((c.y & 3) << 2) | (c.x & 3);
    return (w & (1u << bit)) != 0u;
}

bool rc_solid(ivec3 c) {
    return rc_mask_bit(rc_brick_mask(c >> 2), c);
}

// Whether solid geometry fills the voxel at the center of probe cell `c` on a grid whose
// probes sit `spacing` apart: a probe there would sit inside a wall.
bool rc_cell_solid(ivec3 c, float spacing) {
    vec3 pos = (vec3(c) + 0.5) * spacing;
    ivec3 vc = ivec3(floor((pos - rc_min.xyz) / rc_tune.w));
    return rc_in_grid(vc) && rc_solid(vc);
}
@end

// Requires uniform block with rc_min, rc_dims, rc_tune, rc_vox and rc_brick_data[].
@block rc_march_common
@include_block rc_cell_common

// Follows the ray through the voxels for up to `reach` meters using 64-bit brick occupancy
// masks. 0: nothing in the way, 1: hit (t, the voxel and the face it came in through),
// 2: left the box, 3: started inside something. Empty 4x4x4 bricks are crossed in one step,
// and non-empty bricks test occupancy bits in registers without touching rc_vox_data.
int rc_march(vec3 ro, vec3 rd, float reach, out float t_hit, out ivec3 cell, out vec3 face) {
    float vs = rc_tune.w;
    vec3 g = (ro - rc_min.xyz) / vs;
    float reach_v = reach / vs;
    ivec3 c = ivec3(floor(g));
    t_hit = 0.0;
    cell = c;
    face = vec3(0.0);
    if (!rc_in_grid(c)) return 2;
    ivec3 bc = c >> 2;
    uvec2 mask = rc_brick_mask(bc);
    if (rc_mask_bit(mask, c)) return 3;
    vec3 inv = 1.0 / max(abs(rd), vec3(1e-6));
    ivec3 stp = ivec3(rd.x >= 0.0 ? 1 : -1, rd.y >= 0.0 ? 1 : -1, rd.z >= 0.0 ? 1 : -1);
    vec3 ahead = vec3(greaterThan(stp, ivec3(0)));
    vec3 next = abs(vec3(c) + ahead - g) * inv;
    for (int i = 0; i < 400; i++) {
        int axis;
        float t;
        if ((mask.x | mask.y) == 0u) {
            vec3 leave = abs(vec3(bc * 4) + ahead * 4.0 - g) * inv;
            axis = leave.x < leave.y ? (leave.x < leave.z ? 0 : 2) : (leave.y < leave.z ? 1 : 2);
            t = leave[axis];
            if (t > reach_v) return 0;
            c = clamp(ivec3(floor(g + rd * t)), bc * 4, bc * 4 + 3);
            c[axis] = bc[axis] * 4 + (stp[axis] > 0 ? 4 : -1);
            if (!rc_in_grid(c)) return 2;
            next = abs(vec3(c) + ahead - g) * inv;
            bc = c >> 2;
            mask = rc_brick_mask(bc);
        } else {
            axis = next.x < next.y ? (next.x < next.z ? 0 : 2) : (next.y < next.z ? 1 : 2);
            t = next[axis];
            if (t > reach_v) return 0;
            c[axis] += stp[axis];
            if (!rc_in_grid(c)) return 2;
            next[axis] += inv[axis];
            int nbc = c[axis] >> 2;
            if (nbc != bc[axis]) {
                bc[axis] = nbc;
                mask = rc_brick_mask(bc);
            }
        }
        if (rc_mask_bit(mask, c)) {
            t_hit = t * vs;
            cell = c;
            face[axis] = -float(stp[axis]);
            return 1;
        }
    }
    return 0;
}
@end

// Requires uniform block with rc_min, rc_dims, rc_tune, rc_info, rc_sky, rc_ground, rc_vox
// and buffers rc_vox_data[], rc_brick_data[], rc_light_data[].
@block rc_trace_common
@include_block rc_march_common

int rc_every() {
    return max(rc_vox.w & 0xffff, 1);
}

int rc_shadow_rays() {
    return max(rc_vox.w >> 16, 1);
}

vec3 rc_sky_at(vec3 rd) {
    return mix(rc_ground.rgb, rc_sky.rgb, 0.5 * rd.y + 0.5);
}

// Direct light arriving at hit point `hp` with normal `face`, evaluated via Reservoir
// Importance Sampling (RIS) with `rc_shadow_rays()` shadow marches (1: pure single-ray RIS;
// >1: first shadow_rays-1 contributing lights tested deterministically, remainder via RIS).
vec3 rc_direct_light(vec3 hp, vec3 face, uint seed) {
    vec3 arriving = vec3(0.0);
    int nl = rc_info.z;
    int shadow_rays = rc_shadow_rays();
    int direct_cap = max(shadow_rays - 1, 0);
    int direct_tested = 0;
    float total = 0.0;
    vec3 picked = vec3(0.0);
    float picked_w = 0.0;
    vec3 picked_dir = vec3(0.0);
    float picked_dist = 0.0;
    for (int l = 0; l < nl; l++) {
        rc_light lt = rc_light_data[l];
        bool sun = lt.dir.w < -2.5;
        vec3 lv = lt.pos.xyz - hp;
        float dist = length(lv);
        if (!sun && lt.pos.w > 0.0 && dist > lt.pos.w) continue;
        vec3 ld = lv / max(dist, 1e-4);
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
        if (direct_tested < direct_cap) {
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
    if (picked_w > 0.0 && shadow_rays > 0) {
        float st;
        ivec3 sc;
        vec3 sf;
        if (rc_march(hp, picked_dir, picked_dist - rc_tune.w, st, sc, sf) != 1) arriving += picked * (total / picked_w);
    }
    return arriving;
}
@end
