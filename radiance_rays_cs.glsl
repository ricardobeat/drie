// Split Radiance Cascades (Freeman, Sannikov & Pitt-Francis, arXiv:2607.20384):
//   0. rc_rays_clear_cs        zero hash table, probe meta, free stacks, history and accumulators
//   1. rc_rays_rebuild_cs      rebuild hash table every frame from live meta slots; evict stale/OOB
//   2. rc_rays_alloc_cs        per traced G-Buffer pixel: touch nearest L0 probe, count rays
//   2b. rc_rays_cache_alloc_cs seed the secondary cache's L0 probes from this frame's trace hits
//   3. rc_rays_propagate_cs    bottom-up: each active probe touches its single parent (c >> 1)
//   4. rc_rays_offsets_cs      top-down hierarchical R2 prefix segments (Algorithm 3)
//   5. rc_rays_trace_cs        surface-driven R2 ray march + multi-cascade interval deposit
//   6. rc_rays_merge_cs        top-down resolve, per-bin temporal blend, sparse trilinear merge,
//                              and L0 6-face irradiance integration
//
// The primary and the secondary cache both run 2-6 through the same kernels; a bit in
// rc_dims.w says which one a dispatch is for.
@header module drie_shaders::rc_rays;
@header import drie;

@ctype mat4 Mat4
@ctype vec4 Vec4

@include radiance_common.glsl

@block rc_rays_common
@include_block rc_key_common
@include_block rc_pack_common

layout(binding=0) uniform rc_rays_params {
    vec4 rc_min;       // xyz = voxel box min, w = L0 probe spacing (m)
    ivec4 rc_dims;     // xyz = voxel brick dims (nx, ny, nz), w = pass descriptor
    vec4 rc_tune;      // x = t0 (m), y = blend, z = frame, w = voxel size (m)
    ivec4 rc_info;     // x = theta0 (4 -> 32 dirs), y = levels, z = lights, w = hash_cap
    vec4 rc_sky;       // rgb = sky radiance, w = bounce strength
    vec4 rc_ground;    // rgb = ground radiance, w = normal_offset
    ivec4 rc_vox;      // xyz = voxel dims (vx, vy, vz), w = stride | (shadow_rays << 16)
    ivec4 rc_lcap0;    // probe pool capacity for levels 0..3
    ivec4 rc_lcap1;    // probe pool capacity for levels 4..7
    ivec4 rc_loff0;    // bin buffer start offset for levels 0..3
    ivec4 rc_loff1;    // bin buffer start offset for levels 4..7
    ivec4 rc_pbase0;   // probe_meta start index for levels 0..3
    ivec4 rc_pbase1;   // probe_meta start index for levels 4..7
    mat4 rc_inv_vp;    // inverse relative view-projection for G-Buffer allocation
    vec4 rc_cam_pos;   // xyz = camera world pos, w = depth_zero_to_one (1 or 0)
    vec4 rc_screen;    // xy = 1/w, 1/h, z = flip_y (1 on Metal/D3D), w = rays_per_probe
};

// rc_dims.w, the descriptor of what a dispatch is for:
//   bits 0..3 level, bit 4 cache, bit 5 phase (propagate's two halves).
bool rc_is_cache() {
    return (rc_dims.w & 16) != 0;
}

int rc_pass_level() {
    return rc_dims.w & 15;
}

int rc_pass_phase() {
    return (rc_dims.w >> 5) & 1;
}

// The primary trace reads the cache's irradiance for its bounce light. False leaves it out
// (RadianceCascades.multibounce off), so the previous frames' cache values can stay put.
bool rc_reads_cache() {
    return (rc_dims.w & 64) != 0;
}

// 32 bytes per probe slot:
//   a: x = pack_xy(c), y = key32 (0 = free), z = parent_slot, w = last_frame | birth << 16 |
//      vismask << 24
//   b: x = rays, y = ray_off, z = cursor, w = coord_z (signed int as uint)
struct rc_ray_meta {
    uvec4 a;
    uvec4 b;
};

struct rc_uvec4 {
    uvec4 v;
};

struct rc_uvec2 {
    uvec2 p;
};

struct rc_uint {
    uint v;
};

// A surface hit the primary trace found, for the secondary cache to seed probes from:
// xyz = the point, w as a float = (frame << 3) | the face's axis.
struct rc_hit {
    vec4 p;
};

const uint RC_RAYS_STALE = 24u;
// meta_buf's b.x: rays in the low 16 bits, then the anchor: which way the surface a probe was
// made for faces, as rc_face_axis's 0..5 (+1, 0 = none yet). A surface behind that plane is on
// the probe's far side, across the wall the probe sits on, and takes none of its light.
const uint RC_ANCHOR_SHIFT = 16u;
const uint RC_ANCHOR_MASK = 7u;
// meta_buf's a.w: last_frame in the low 12 bits, birth in the next 8, then the visibility mask:
// one bit per trilinear corner of the coarser grid around the probe, set when nothing solid
// stands between them, so the merge never blends light across a wall.
const uint RC_BIRTH_SHIFT = 16u;
const uint RC_VISMASK_SHIFT = 24u;
// How much of its light a corner gets through when something stands in the way: a wall
// leaks a little, a nook with no other corner at all still has some light.
const float RC_VIS_HIDDEN = 0.02;
const float RC_FP_SCALE = 16384.0;
const uint RC_MAX_TRACE_RAYS = 1048576u;
// free_buf: everything below RC_FREE_BASE is counters; from there on, one slot id per probe
// slot, the primary levels' then the cache's.
//   0..7   primary free-stack tops     19..26 cache free-stack tops
//   8..15  primary bump counts        27..34 cache bump counts
//   16     primary R2 prefix          35     cache R2 prefix
//   17     primary traced rays        36     cache traced rays
const int RC_FREE_BASE = 40;

int rc_level_cap(int l) { return l < 4 ? rc_lcap0[l] : rc_lcap1[l - 4]; }
int rc_level_off(int l) { return l < 4 ? rc_loff0[l] : rc_loff1[l - 4]; }
int rc_probe_base(int l) { return l < 4 ? rc_pbase0[l] : rc_pbase1[l - 4]; }

int rc_dirs(int level) {
    int th = rc_info.x << level;
    return 2 * th * th;
}

int rc_total_meta() {
    int top = rc_info.y - 1;
    return rc_probe_base(top) + rc_level_cap(top);
}

int rc_total_bins() {
    int top = rc_info.y - 1;
    return rc_level_off(top) + rc_level_cap(top) * rc_dirs(top);
}

// The secondary cache: 4x the primary's spacing and reach, its level 0 with 1/16 of the
// primary's probes, the same number of cascades. Its pools start where the primary's end, so
// where they start is all the layout it needs.
int rc_cache_levels() {
    return rc_info.y;
}

int rc_cache_cap(int l) {
    return max(rc_level_cap(l) / 16, 16);
}

int rc_cache_meta_base() {
    return rc_total_meta();
}

int rc_cache_bin_base() {
    return rc_total_bins();
}

// irradiance_buf: the hash table, then the L0 6-face irradiance, then the L1 one (the fallback
// where no L0 probe is near), then the cache's L0 one.
int rc_l0_irr_base() {
    return rc_info.w;
}

int rc_l1_irr_base() {
    return rc_l0_irr_base() + rc_level_cap(0) * 6;
}

int rc_cache_irr_base() {
    return rc_l1_irr_base() + (rc_info.y > 1 ? rc_level_cap(1) * 6 : 0);
}

int rc_cache_probe_base(int l) {
    int pb = rc_cache_meta_base();
    for (int k = 0; k < l; k++) pb += rc_cache_cap(k);
    return pb;
}

int rc_cache_bin_off(int l) {
    int bo = rc_cache_bin_base();
    for (int k = 0; k < l; k++) bo += rc_cache_cap(k) * rc_dirs(k);
    return bo;
}

float rc_cache_spacing(int l) {
    return rc_min.w * 4.0 * float(1 << l);
}

float rc_cache_t0() {
    return rc_tune.x * 4.0;
}

int rc_total_meta_all() {
    int top = rc_info.y - 1;
    return rc_cache_probe_base(top) + rc_cache_cap(top);
}

int rc_total_bins_all() {
    int top = rc_info.y - 1;
    return rc_cache_bin_off(top) + rc_cache_cap(top) * rc_dirs(top);
}

int rc_cap_of(bool cache, int l) {
    return cache ? rc_cache_cap(l) : rc_level_cap(l);
}

int rc_probe_base_of(bool cache, int l) {
    return cache ? rc_cache_probe_base(l) : rc_probe_base(l);
}

int rc_bin_off_of(bool cache, int l) {
    return cache ? rc_cache_bin_off(l) : rc_level_off(l);
}

float rc_spacing_of(bool cache, int l) {
    return cache ? rc_cache_spacing(l) : rc_min.w * float(1 << l);
}

float rc_t0_of(bool cache) {
    return cache ? rc_cache_t0() : rc_tune.x;
}

int rc_free_top(bool cache, int l) { return (cache ? 19 : 0) + l; }
int rc_free_bump(bool cache, int l) { return (cache ? 27 : 8) + l; }
int rc_ray_prefix(bool cache) { return cache ? 35 : 16; }
int rc_ray_count(bool cache) { return cache ? 36 : 17; }

// hcap records of (xyz, tag): 1 in 4 traced hits is stored for the cache to seed from.
int rc_hits_cap() {
    return max(rc_cache_cap(0) * 4, 1024);
}

ivec3 rc_unpack_coord(uvec4 a, uvec4 b) {
    int ix = int(a.x & 0xffffu) - 32768;
    int iy = int((a.x >> 16) & 0xffffu) - 32768;
    int iz = int(b.w);
    return ivec3(ix, iy, iz);
}

int rc_face_axis(vec3 face) {
    return face.x != 0.0 ? (face.x > 0.0 ? 0 : 1) : (face.y != 0.0 ? (face.y > 0.0 ? 2 : 3) : (face.z > 0.0 ? 4 : 5));
}

vec3 rc_axis_normal(int axis) {
    return axis == 0 ? vec3(1.0, 0.0, 0.0) : axis == 1 ? vec3(-1.0, 0.0, 0.0)
        : axis == 2 ? vec3(0.0, 1.0, 0.0) : axis == 3 ? vec3(0.0, -1.0, 0.0)
        : axis == 4 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 0.0, -1.0);
}

// The anchor a surface with normal `n` gives its probe: rc_face_axis's index, +1 so that 0
// means "none yet".
uint rc_anchor_code(vec3 n) {
    return uint(rc_face_axis(n) + 1);
}

// A point is on a probe's side of the wall it was made on: how much of its light the point
// may take. 1 in front of the anchor's plane, 0 a cell and a half behind it. A surface the
// probe was made for sits half a cell behind its anchor, so the test leaves that alone and
// only cuts what is through the wall.
float rc_anchor_trust(ivec3 c, float spacing, uint points, vec3 p) {
    uint code = points & RC_ANCHOR_MASK;
    if (code == 0u) return 1.0;
    vec3 n = rc_axis_normal(int(code) - 1);
    vec3 probe_pos = (vec3(c) + 0.5) * spacing;
    return smoothstep(-1.5 * spacing, -0.9 * spacing, dot(p - probe_pos, n));
}

// Morton 2D encode/decode for direction grids (up to 64x64)
uint rc_part1by1(uint x) {
    x &= 0x0000ffffu;
    x = (x ^ (x << 8)) & 0x00ff00ffu;
    x = (x ^ (x << 4)) & 0x0f0f0f0fu;
    x = (x ^ (x << 2)) & 0x33333333u;
    x = (x ^ (x << 1)) & 0x55555555u;
    return x;
}

uint rc_compact1by1(uint x) {
    x &= 0x55555555u;
    x = (x ^ (x >> 1)) & 0x33333333u;
    x = (x ^ (x >> 2)) & 0x0f0f0f0fu;
    x = (x ^ (x >> 4)) & 0x00ff00ffu;
    x = (x ^ (x >> 8)) & 0x0000ffffu;
    return x;
}

uint rc_morton2d(uint x, uint y) {
    return rc_part1by1(x) | (rc_part1by1(y) << 1);
}

uvec2 rc_unmorton2d(uint m) {
    return uvec2(rc_compact1by1(m), rc_compact1by1(m >> 1));
}

// Equal-area Lambert cylindrical mapping:
//   u in [0, 1) -> azimuth phi in [0, 2*pi)
//   v in [0, 1) -> y = 2*v - 1, r = sqrt(1 - y^2)
vec3 rc_cyl_to_dir(vec2 uv) {
    float phi = uv.x * (2.0 * RC_PI);
    float y = 2.0 * uv.y - 1.0;
    float r = sqrt(max(1.0 - y * y, 0.0));
    return vec3(r * cos(phi), y, r * sin(phi));
}

vec2 rc_dir_to_cyl(vec3 d) {
    float u = fract(atan(d.z, d.x) / (2.0 * RC_PI));
    float v = clamp(0.5 * d.y + 0.5, 0.0, 0.999999);
    return vec2(u, v);
}

int rc_dir_bin(vec2 uv, int theta) {
    int iu = clamp(int(floor(uv.x * float(2 * theta))), 0, 2 * theta - 1);
    int iv = clamp(int(floor(uv.y * float(theta))), 0, theta - 1);
    int half_off = iu >= theta ? (theta * theta) : 0;
    int mx = iu >= theta ? (iu - theta) : iu;
    return half_off + int(rc_morton2d(uint(mx), uint(iv)));
}

vec3 rc_bin_center_dir(int k, int theta) {
    int tt = theta * theta;
    int h = k >= tt ? 1 : 0;
    uvec2 xy = rc_unmorton2d(uint(k - h * tt));
    int iu = h * theta + int(xy.x);
    int iv = int(xy.y);
    vec2 uv = (vec2(float(iu), float(iv)) + 0.5) / vec2(float(2 * theta), float(theta));
    return rc_cyl_to_dir(uv);
}

uvec2 rc_pack_hist(vec3 J, float beta, uint n) {
    uvec2 p = rc_pack_rad(J);
    uint b8 = uint(clamp(beta * 255.0 + 0.5, 0.0, 255.0));
    uint n8 = min(n, 255u);
    p.y = (p.y & 0xffffu) | (b8 << 16) | (n8 << 24);
    return p;
}

void rc_unpack_hist(uvec2 p, out vec3 J, out float beta, out uint n) {
    J = rc_unpack_rad(uvec2(p.x, p.y & 0xffffu));
    beta = float((p.y >> 16) & 0xffu) / 255.0;
    n = p.y >> 24;
}

// Interval boundaries: level 0 is [0, t0], level l >= 1 is [t0 * 4^(l-1), t0 * 4^l]
float rc_interval_end(int l, bool cache) {
    return rc_t0_of(cache) * pow(4.0, float(l));
}
@end

// -1: the key isn't in the table
@block rc_rays_lookup_common
// Requires rc_info, rc_ht.
int rc_lookup(ivec3 c, int level, bool cache) {
    uint key = cache ? rc_pack_key32_cache(c, level) : rc_pack_key32(c, level);
    uint mask = uint(rc_info.w - 1);
    uint idx = rc_hash(key) & mask;
    for (int step = 0; step < 32; step++) {
        uint cur = rc_ht[idx].v.y;
        if (cur == 0u) return -1;
        if (cur == key) {
            uint s = rc_ht[idx].v.z;
            return s != 0xffffffffu ? int(s) : -1;
        }
        idx = (idx + 1u) & mask;
    }
    return -1;
}
@end

// Shared lock-free probe touch/insert used by alloc, cache alloc and propagate
@block rc_rays_touch_common
int rc_rays_free_pop(int level, bool cache) {
    int top = rc_free_top(cache, level);
    if (rc_free[top].v == 0u) return -1;
    int cap = rc_cap_of(cache, level);
    int pbase = rc_probe_base_of(cache, level);
    int old = int(atomicAdd(rc_free[top].v, uint(-1)));
    if (old > 0 && old <= cap) return int(rc_free[RC_FREE_BASE + pbase + (old - 1)].v);
    atomicAdd(rc_free[top].v, 1u);
    return -1;
}

int rc_rays_touch_probe(ivec3 c, int level, bool cache, uint frame, uint anchor) {
    uint key = cache ? rc_pack_key32_cache(c, level) : rc_pack_key32(c, level);
    int cap = rc_cap_of(cache, level);
    int bump = rc_free_bump(cache, level);
    uint mask = uint(rc_info.w - 1);
    uint idx = rc_hash(key) & mask;
    for (int step = 0; step < 32; step++) {
        uint cur = rc_ht[idx].v.y;
        if (cur == key) {
            uint s = rc_ht[idx].v.z;
            if (s != 0xffffffffu) {
                int mi = rc_probe_base_of(cache, level) + int(s);
                rc_meta[mi].a.w = (rc_meta[mi].a.w & 0xffff0000u) | (frame & 0xfffu);
                rc_ht[idx].v.w = frame | ((rc_meta[mi].b.x >> RC_ANCHOR_SHIFT) << 16);
                return int(s);
            }
            rc_ht[idx].v.w = frame;
            return -1;
        }
        if (cur == 0u) {
            if (rc_free[rc_free_top(cache, level)].v == 0u && rc_free[bump].v >= uint(cap)) return -1;
            uint prev = atomicCompSwap(rc_ht[idx].v.y, 0u, key);
            if (prev == 0u) {
                int slot = rc_rays_free_pop(level, cache);
                if (slot < 0 && rc_free[bump].v < uint(cap)) {
                    uint s = atomicAdd(rc_free[bump].v, 1u);
                    slot = int(s) < cap ? int(s) : -1;
                }
                if (slot < 0) {
                    rc_ht[idx].v.y = 0u;
                    return -1;
                }
                uint pxy = rc_pack_xy(c);
                int mi = rc_probe_base_of(cache, level) + slot;
                rc_meta[mi].a = uvec4(pxy, key, 0xffffffffu, (frame & 0xfffu) | ((frame & 0xffu) << RC_BIRTH_SHIFT));
                rc_meta[mi].b = uvec4(1u | ((anchor & RC_ANCHOR_MASK) << RC_ANCHOR_SHIFT), 0u, 0u, uint(c.z));
                rc_ht[idx].v.x = pxy;
                rc_ht[idx].v.z = uint(slot);
                rc_ht[idx].v.w = frame;
                return slot;
            }
            if (prev == key) {
                rc_ht[idx].v.w = frame;
                uint s = rc_ht[idx].v.z;
                if (s != 0xffffffffu) {
                    int mi = rc_probe_base_of(cache, level) + int(s);
                    rc_meta[mi].a.w = (rc_meta[mi].a.w & 0xffff0000u) | (frame & 0xfffu);
                    rc_ht[idx].v.w = frame | ((rc_meta[mi].b.x >> RC_ANCHOR_SHIFT) << 16);
                    return int(s);
                }
                return -1;
            }
        }
        idx = (idx + 1u) & mask;
    }
    return -1;
}
@end

// Kernel 0: Full clear on reset()
@cs rc_rays_clear_cs
@include_block rc_rays_common
layout(binding=0) buffer rc_ht_clear { rc_uvec4 rc_ht[]; };
layout(binding=1) buffer rc_meta_clear { rc_ray_meta rc_meta[]; };
layout(binding=2) buffer rc_free_clear { rc_uint rc_free[]; };
layout(binding=3) buffer rc_hist_clear { rc_uvec2 rc_hist[]; };
layout(binding=4) buffer rc_acc_clear { rc_uvec4 rc_acc[]; };
layout(binding=5) buffer rc_hits_clear { rc_hit rc_hits[]; };

layout(local_size_x=64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    int tmeta = rc_total_meta_all();
    int tbins = rc_total_bins_all();
    if (i < uint(rc_info.w)) rc_ht[i].v = uvec4(0u, 0u, 0xffffffffu, 0u);
    if (i < uint(tmeta)) {
        rc_meta[i].a = uvec4(0u);
        rc_meta[i].b = uvec4(0u);
    }
    if (i < uint(RC_FREE_BASE + tmeta)) rc_free[i].v = 0u;
    if (i < uint(tbins)) {
        rc_hist[i].p = uvec2(0u);
        rc_acc[i].v = uvec4(0u);
    }
    if (i < uint(rc_hits_cap())) rc_hits[i].p = vec4(0.0);
}
@end

// Kernel 1: Per-frame hash table rebuild from meta_buf (2 phases: rc_dims.w == 0 clears table
// and free stacks; rc_dims.w == 1 re-inserts live probes and pushes stale/OOB ones to free_buf)
@cs rc_rays_rebuild_cs
@include_block rc_rays_common
layout(binding=0) buffer rc_ht_reb { rc_uvec4 rc_ht[]; };
layout(binding=1) buffer rc_meta_reb { rc_ray_meta rc_meta[]; };
layout(binding=2) buffer rc_free_reb { rc_uint rc_free[]; };
layout(binding=3) buffer rc_hist_reb { rc_uvec2 rc_hist[]; };

layout(local_size_x=64) in;

void rc_push_free(bool cache, int level, uint slot) {
    int cap = rc_cap_of(cache, level);
    int pbase = rc_probe_base_of(cache, level);
    int top = rc_free_top(cache, level);
    uint idx = atomicAdd(rc_free[top].v, 1u);
    if (int(idx) < cap) rc_free[RC_FREE_BASE + pbase + int(idx)].v = slot;
}

// Which pool a meta index belongs to: the primary levels first, then the cache's.
void rc_slot_of(int mi, out int level, out bool cache, out int slot) {
    if (mi >= rc_cache_probe_base(0)) {
        cache = true;
        for (int l = rc_cache_levels() - 1; l >= 0; l--) {
            int b = rc_cache_probe_base(l);
            if (mi >= b) {
                level = l;
                slot = mi - b;
                return;
            }
        }
        level = 0;
        slot = mi - rc_cache_probe_base(0);
        return;
    }
    cache = false;
    for (int l = rc_info.y - 1; l >= 0; l--) {
        int b = rc_probe_base(l);
        if (mi >= b) {
            level = l;
            slot = mi - b;
            return;
        }
    }
    level = 0;
    slot = mi;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    int tmeta = rc_total_meta_all();
    if (rc_pass_phase() == 0) {
        if (i < uint(rc_info.w)) rc_ht[i].v = uvec4(0u, 0u, 0xffffffffu, 0u);
        // The free-stack tops go back to empty every frame; the bump counts (and so the slot
        // numbers of live probes) are left alone.
        if (i < uint(rc_info.y)) {
            rc_free[rc_free_top(false, int(i))].v = 0u;
            rc_free[rc_free_top(true, int(i))].v = 0u;
        }
        if (i == 0u) {
            rc_free[rc_ray_prefix(false)].v = 0u;
            rc_free[rc_ray_count(false)].v = 0u;
            rc_free[rc_ray_prefix(true)].v = 0u;
            rc_free[rc_ray_count(true)].v = 0u;
        }
        return;
    }
    if (int(i) >= tmeta) return;
    int level;
    bool cache;
    int slot;
    rc_slot_of(int(i), level, cache, slot);
    if (slot >= int(rc_free[rc_free_bump(cache, level)].v)) return;

    uvec4 a = rc_meta[i].a;
    uvec4 b = rc_meta[i].b;
    uint key = a.y;
    if (key == 0u) {
        rc_push_free(cache, level, uint(slot));
        return;
    }
    uint frame = uint(rc_tune.z) + 1u;
    uint last_f = a.w & 0xffffu;
    uint age = frame >= last_f ? (frame - last_f) : (frame + 4096u - last_f);

    float sp = rc_spacing_of(cache, level);
    ivec3 c = rc_unpack_coord(a, b);
    vec3 pos = (vec3(c) + 0.5) * sp;
    vec3 bmax = rc_min.xyz + vec3(rc_dims.xyz) * rc_min.w;
    bool oob = any(lessThan(pos, rc_min.xyz - sp)) || any(greaterThan(pos, bmax + sp));

    if (age > RC_RAYS_STALE || oob) {
        rc_meta[i].a = uvec4(0u);
        rc_meta[i].b = uvec4(0u);
        // Zero the history of a freed slot so its next occupant starts clean
        int dirs = rc_dirs(level);
        int boff = rc_bin_off_of(cache, level) + slot * dirs;
        for (int k = 0; k < dirs; k++) rc_hist[boff + k].p = uvec2(0u);
        rc_push_free(cache, level, uint(slot));
        return;
    }

    // Reset this frame's ray counters and parent link
    rc_meta[i].a.z = 0xffffffffu;
    rc_meta[i].b.x = 0u;
    rc_meta[i].b.y = 0u;
    rc_meta[i].b.z = 0u;

    uint mask = uint(rc_info.w - 1);
    uint h = rc_hash(key) & mask;
    for (int step = 0; step < 32; step++) {
        uint prev = atomicCompSwap(rc_ht[h].v.y, 0u, key);
        if (prev == 0u) {
            rc_ht[h].v.x = a.x;
            rc_ht[h].v.z = uint(slot);
            rc_ht[h].v.w = last_f | ((b.x >> RC_ANCHOR_SHIFT) << 16);
            return;
        }
        if (prev == key) break;
        h = (h + 1u) & mask;
    }
    rc_meta[i].a = uvec4(0u);
    rc_push_free(cache, level, uint(slot));
}
@end

// Kernel 2: Allocate/touch nearest L0 probe per traced G-Buffer pixel and tally rays
@block rc_rays_alloc_body
@include_block rc_rays_common
layout(binding=0) uniform sampler rc_gb_smp;
#ifdef RC_DEPTH_MS
// a multisampled depth buffer (a render target with MSAA): sample 0 of each pixel
layout(binding=0) uniform texture2DMS rc_depth_ms_tex;
ivec2 rc_depth_size() { return textureSize(sampler2DMS(rc_depth_ms_tex, rc_gb_smp)); }
float rc_depth_at(ivec2 p) { return texelFetch(sampler2DMS(rc_depth_ms_tex, rc_gb_smp), p, 0).r; }
#else
layout(binding=0) uniform texture2D rc_depth_tex;
ivec2 rc_depth_size() { return textureSize(sampler2D(rc_depth_tex, rc_gb_smp), 0); }
float rc_depth_at(ivec2 p) { return texelFetch(sampler2D(rc_depth_tex, rc_gb_smp), p, 0).r; }
#endif
layout(binding=1) uniform texture2D rc_normal_tex;
layout(binding=2) buffer rc_ht_alloc { rc_uvec4 rc_ht[]; };
layout(binding=3) buffer rc_meta_alloc { rc_ray_meta rc_meta[]; };
layout(binding=4) buffer rc_free_alloc { rc_uint rc_free[]; };
layout(binding=5) readonly buffer rc_bricks_alloc { rc_brick rc_brick_data[]; };

layout(local_size_x=8, local_size_y=8) in;

@include_block rc_cell_common
@include_block rc_rays_touch_common

void main() {
    int stride = max(rc_vox.w & 0xffff, 1);
    ivec2 sz = rc_depth_size();
    int tw = (sz.x + stride - 1) / stride;
    int th = (sz.y + stride - 1) / stride;
    ivec2 tid = ivec2(gl_GlobalInvocationID.xy);
    if (tid.x >= tw || tid.y >= th) return;

    uint frame = uint(rc_tune.z) + 1u;
    // Rotate sub-pixel offset inside the stride x stride tile across frames
    int ox = int(frame) % stride;
    int oy = (int(frame) / stride) % stride;
    ivec2 px = min(tid * stride + ivec2(ox, oy), sz - 1);

    float raw_d = rc_depth_at(px);
    if (raw_d >= 0.99999) return;

    vec2 su = (vec2(px) + 0.5) * rc_screen.xy;
    vec2 ndc_xy = su * 2.0 - 1.0;
    if (rc_screen.z > 0.5) ndc_xy.y = -ndc_xy.y;
    float ndc_z = rc_cam_pos.w > 0.5 ? raw_d : raw_d * 2.0 - 1.0;
    vec4 rel = rc_inv_vp * vec4(ndc_xy, ndc_z, 1.0);
    vec3 wp = rel.xyz / rel.w + rc_cam_pos.xyz;
    vec3 n = rc_unpack_oct_normal(texelFetch(sampler2D(rc_normal_tex, rc_gb_smp), px, 0).xy);
    vec3 p = wp + n * rc_ground.w;

    float sp0 = rc_min.w;
    vec3 bmax = rc_min.xyz + vec3(rc_dims.xyz) * sp0;
    if (any(lessThan(p, rc_min.xyz)) || any(greaterThan(p, bmax))) return;

    // Nearest c0 probe (paper §3.2); if solid, pick nearest non-solid trilinear corner
    ivec3 c_near = ivec3(floor(p / sp0));
    if (rc_cell_solid(c_near, sp0)) {
        ivec3 b0 = ivec3(floor(p / sp0 - 0.5));
        float best_d2 = 1e30;
        for (int k = 0; k < 8; k++) {
            ivec3 o = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
            ivec3 ck = b0 + o;
            if (rc_cell_solid(ck, sp0)) continue;
            vec3 d = (vec3(ck) + 0.5) * sp0 - p;
            float d2 = dot(d, d);
            if (d2 < best_d2) {
                best_d2 = d2;
                c_near = ck;
            }
        }
    }

    // (the surface this probe is made for goes in with it: which side its light comes from)
    int slot = rc_rays_touch_probe(c_near, 0, false, frame, rc_anchor_code(n));
    if (slot >= 0) atomicAdd(rc_meta[rc_probe_base(0) + slot].b.x, 1u);
    // Also touch the 3 other bilinear corners in the same normal layer (ck[an] == c_near[an])
    // so grazing walls and strided G-Buffer tiles never leave missing trilinear corners, while
    // keeping the probe sheet strictly 1 cell thick along the surface normal (~4x drop/level).
    int an = abs(n.x) >= abs(n.y) ? (abs(n.x) >= abs(n.z) ? 0 : 2) : (abs(n.y) >= abs(n.z) ? 1 : 2);
    ivec3 b0 = ivec3(floor(p / sp0 - 0.5));
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        ivec3 ck = b0 + o;
        if (ck[an] != c_near[an] || ck == c_near) continue;
        if (rc_cell_solid(ck, sp0)) continue;
        int s = rc_rays_touch_probe(ck, 0, false, frame, rc_anchor_code(n));
        if (s >= 0) atomicMax(rc_meta[rc_probe_base(0) + s].b.x, 1u);
    }
}
@end

@cs rc_rays_alloc_cs
@include_block rc_rays_alloc_body
@end

@cs rc_rays_alloc_ms_cs
#define RC_DEPTH_MS
@include_block rc_rays_alloc_body
@end

// Kernel 2b: Seed the secondary cache's L0 probes from this frame's trace hits. The primary
// trace stores one hit in four at rc_hit[ray_id / 4 % cap], tagged with the frame it was taken
// in, so the records written this frame are the ones to use.
@cs rc_rays_cache_alloc_cs
@include_block rc_rays_common
layout(binding=0) readonly buffer rc_hits_alloc { rc_hit rc_hits[]; };
layout(binding=1) buffer rc_ht_cache_alloc { rc_uvec4 rc_ht[]; };
layout(binding=2) buffer rc_meta_cache_alloc { rc_ray_meta rc_meta[]; };
layout(binding=3) buffer rc_free_cache_alloc { rc_uint rc_free[]; };
layout(binding=4) readonly buffer rc_bricks_cache_alloc { rc_brick rc_brick_data[]; };

layout(local_size_x=64) in;

@include_block rc_cell_common
@include_block rc_rays_touch_common

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(rc_hits_cap())) return;
    vec4 h = rc_hits[i].p;
    uint tag = uint(int(h.w));
    uint frame = uint(rc_tune.z) + 1u;
    if ((tag >> 3) != frame) return;

    float sp = rc_cache_spacing(0);
    vec3 p = h.xyz;
    ivec3 c = ivec3(floor(p / sp));
    // A probe is as big as a cache cell: if its center is inside the surface the ray hit,
    // walk a cell at a time along the face the ray came in through until it is outside.
    vec3 nrm = rc_axis_normal(int(tag & 7u));
    for (int k = 0; k < 3 && rc_cell_solid(c, sp); k++) {
        p += nrm * sp;
        c = ivec3(floor(p / sp));
    }
    if (rc_cell_solid(c, sp)) return;
    vec3 bmax = rc_min.xyz + vec3(rc_dims.xyz) * rc_min.w;
    if (any(lessThan(p, rc_min.xyz)) || any(greaterThan(p, bmax))) return;
    // (the face the ray came in through: the side this probe's light comes from)
    int slot = rc_rays_touch_probe(c, 0, true, frame, uint(int(tag & 7u) + 1));
    if (slot >= 0) atomicMax(rc_meta[rc_cache_probe_base(0) + slot].b.x, 1u);
}
@end

// Kernel 3: Bottom-up propagation (l = 0 .. levels - 2) in two phases per level:
//   phase 0: touch/insert the single nearest parent (c >> 1) and set the level-0 ray budget
//   phase 1: look up the finalized parent slot, set parent_slot, add the child's rays
@cs rc_rays_propagate_cs
@include_block rc_rays_common
layout(binding=2) buffer rc_ht_prop { rc_uvec4 rc_ht[]; };
layout(binding=3) buffer rc_meta_prop { rc_ray_meta rc_meta[]; };
layout(binding=4) buffer rc_free_prop { rc_uint rc_free[]; };
layout(binding=5) readonly buffer rc_bricks_prop { rc_brick rc_brick_data[]; };

layout(local_size_x=64) in;

@include_block rc_rays_touch_common
@include_block rc_rays_lookup_common
@include_block rc_march_common

// Which of the 8 corners of the coarser grid around the probe the merge may blend in: the
// ones light can reach from here, and the ones inside geometry, which sum up the surface this
// probe sits on and are what its own light arrives through. A corner across a wall in free
// space carries another room's light and is left out. Made once, for a probe new this frame.
uint rc_vis_mask(vec3 pos, float spacing, float coarse) {
    uint m = 0u;
    ivec3 cbase = ivec3(floor(pos / coarse - 0.5));
    for (int k = 0; k < 8; k++) {
        vec3 o = vec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        if (rc_cell_solid(cbase + ivec3(o), coarse)) {
            m |= 1u << uint(k);
            continue;
        }
        vec3 d = (o * 2.0 - 1.0) * spacing;
        float t_hit;
        ivec3 cell;
        vec3 face;
        if (rc_march(pos, normalize(d), length(d), t_hit, cell, face) != 1) m |= 1u << uint(k);
    }
    return m;
}

void main() {
    bool cache = rc_is_cache();
    int level = rc_pass_level();
    int phase = rc_pass_phase();
    int up = level + 1;
    int nlevels = cache ? rc_cache_levels() : rc_info.y;
    int slot = int(gl_GlobalInvocationID.x);
    int count = min(int(rc_free[rc_free_bump(cache, level)].v), rc_cap_of(cache, level));
    if (slot >= count) return;

    int mi = rc_probe_base_of(cache, level) + slot;
    uvec4 a = rc_meta[mi].a;
    uvec4 b = rc_meta[mi].b;
    if (a.y == 0u) return;

    uint rays = b.x & 0xffffu;
    if (rays == 0u) return;
    uint frame = uint(rc_tune.z) + 1u;
    uint birth = (a.w >> RC_BIRTH_SHIFT) & 0xffu;
    uint age = (frame - birth) & 0xffu;
    if (phase == 0) {
        if (age == 0u && level + 1 < nlevels) {
            ivec3 cc = rc_unpack_coord(a, b);
            float sp = rc_spacing_of(cache, level);
            uint m = rc_vis_mask((vec3(cc) + 0.5) * sp, sp, sp * 2.0);
            rc_meta[mi].a.w = (a.w & 0x00ffffffu) | (m << RC_VISMASK_SHIFT);
        }
        if (level == 0) {
            if (cache) {
                // the cache's share of the primary's rays, spread evenly over its probes
                uint np = max(rc_free[rc_free_bump(true, 0)].v, 1u);
                uint per = max(uint(float(rc_free[rc_ray_count(false)].v) * 0.25) / np, 4u);
                rays = age < 4u ? min(per * 4u, 1024u) : per;
            } else {
                uint cap_r = uint(max(int(rc_screen.w), 1));
                rays = age < 4u ? min(cap_r * 4u, 256u) : clamp(rays, 4u, cap_r);
            }
            rc_meta[mi].b.x = (b.x & 0xffff0000u) | rays;
        }
        if (up >= nlevels) return;
        ivec3 c = rc_unpack_coord(a, b);
        rc_rays_touch_probe(c >> 1, up, cache, frame, 0u);
        return;
    }
    if (up >= nlevels) return;
    ivec3 uc = rc_unpack_coord(a, b) >> 1;
    int uslot = rc_lookup(uc, up, cache);
    if (uslot < 0) return;
    rc_meta[mi].a.z = uint(uslot);
    atomicAdd(rc_meta[rc_probe_base_of(cache, up) + uslot].b.x, rays & 0xffffu);
}
@end

// Kernel 4: Top-down R2 segment offsets (Algorithm 3), run for l = levels - 1 down to 0.
// At level == 0, also populates rc_rays[0..total_rays-1] with (slot | (local_r << 16)).
@cs rc_rays_offsets_cs
@include_block rc_rays_common
layout(binding=3) buffer rc_meta_off { rc_ray_meta rc_meta[]; };
layout(binding=4) buffer rc_free_off { rc_uint rc_free[]; };
layout(binding=5) buffer rc_rays_off { rc_uint rc_rays[]; };

layout(local_size_x=64) in;

void main() {
    bool cache = rc_is_cache();
    int level = rc_pass_level();
    int nlevels = cache ? rc_cache_levels() : rc_info.y;
    int slot = int(gl_GlobalInvocationID.x);
    int count = min(int(rc_free[rc_free_bump(cache, level)].v), rc_cap_of(cache, level));
    if (slot >= count) return;

    int mi = rc_probe_base_of(cache, level) + slot;
    uvec4 a = rc_meta[mi].a;
    uint rays = rc_meta[mi].b.x & 0xffffu;
    if (a.y == 0u || rays == 0u) return;

    uint my_off = 0u;
    if (level == nlevels - 1) {
        my_off = atomicAdd(rc_free[rc_ray_prefix(cache)].v, rays);
    } else {
        uint uslot = a.z;
        if (uslot != 0xffffffffu) {
            int umi = rc_probe_base_of(cache, level + 1) + int(uslot);
            uint sub = atomicAdd(rc_meta[umi].b.z, rays);
            my_off = rc_meta[umi].b.y + sub;
        }
    }
    rc_meta[mi].b.y = my_off;
    rc_meta[mi].b.z = 0u;

    if (level == 0) {
        uint t_off = atomicAdd(rc_free[rc_ray_count(cache)].v, rays);
        uint lim = min(t_off + rays, RC_MAX_TRACE_RAYS - 1u);
        for (uint r = t_off; r < lim; r++) {
            rc_rays[1u + r].v = uint(slot) | ((r - t_off) << 16);
        }
    }
}
@end

// Kernel 4b: Write indirect dispatch group counts from live probe/ray counters.
// rc_dims.w says whose ray count to publish: 0 primary, 1 cache, anything else neither.
//   rc_args[0]     : trace groups
//   rc_args[1..8]  : merge groups for primary levels 0..7
//   rc_args[9]     : gather groups for primary level 0
//   rc_args[10]    : trace groups for the cache
//   rc_args[11..18]: merge groups for cache levels 0..7
//   rc_args[19]    : gather groups for cache level 0
//   rc_args[20]    : gather groups for the primary's level 1
@cs rc_rays_args_cs
@include_block rc_rays_common
layout(binding=4) readonly buffer rc_free_args { rc_uint rc_free[]; };
layout(binding=5) buffer rc_args_out { rc_uvec4 rc_args[]; };
layout(binding=6) buffer rc_rays_args { rc_uint rc_rays[]; };

layout(local_size_x=32) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i == 0u) {
        uint pr = min(rc_free[17].v, RC_MAX_TRACE_RAYS - 1u);
        uint cr = min(rc_free[36].v, RC_MAX_TRACE_RAYS - 1u);
        if (rc_dims.w == 0) rc_rays[0].v = pr;
        if (rc_dims.w == 1) rc_rays[0].v = cr;
        rc_args[0].v = uvec4(max((pr + 63u) / 64u, 1u), 1u, 1u, 0u);
        rc_args[10].v = uvec4(max((cr + 63u) / 64u, 1u), 1u, 1u, 0u);
        uint n0 = min(rc_free[8].v, uint(rc_level_cap(0)));
        rc_args[9].v = uvec4(max((n0 + 63u) / 64u, 1u), 1u, 1u, 0u);
        uint c0 = min(rc_free[27].v, uint(rc_cache_cap(0)));
        rc_args[19].v = uvec4(max((c0 + 63u) / 64u, 1u), 1u, 1u, 0u);
        uint n1 = rc_info.y > 1 ? min(rc_free[9].v, uint(rc_level_cap(1))) : 0u;
        rc_args[20].v = uvec4(max((n1 + 63u) / 64u, 1u), 1u, 1u, 0u);
    }
    if (i < uint(rc_info.y)) {
        int l = int(i);
        uint dirs = uint(rc_dirs(l));
        uint nl = min(rc_free[8 + l].v, uint(rc_level_cap(l)));
        rc_args[1 + l].v = uvec4(max((nl * dirs + 63u) / 64u, 1u), 1u, 1u, 0u);
        uint cl = min(rc_free[27 + l].v, uint(rc_cache_cap(l)));
        rc_args[11 + l].v = uvec4(max((cl * dirs + 63u) / 64u, 1u), 1u, 1u, 0u);
    }
}
@end

// Kernel 5: Surface-driven R2 ray trace and multi-cascade interval deposit. The primary pass
// also stores every fourth hit for the cache to seed from.
@cs rc_rays_trace_cs
@include_block rc_rays_common
layout(binding=0) readonly buffer rc_trace_voxels { rc_voxel rc_vox_data[]; };
layout(binding=1) readonly buffer rc_trace_bricks { rc_brick rc_brick_data[]; };
layout(binding=2) readonly buffer rc_trace_lights { rc_light rc_light_data[]; };
layout(binding=3) readonly buffer rc_ht_trace { rc_uvec4 rc_ht[]; };
layout(binding=4) readonly buffer rc_meta_trace { rc_ray_meta rc_meta[]; };
layout(binding=5) readonly buffer rc_rays_trace { rc_uint rc_rays[]; };
layout(binding=6) buffer rc_acc_trace { rc_uvec4 rc_acc[]; };
layout(binding=7) buffer rc_trace_hits { rc_hit rc_hits[]; };

@include_block rc_trace_common
@include_block rc_rays_lookup_common

layout(local_size_x=64) in;

// Irradiance at `p` (on a surface with `face`) from the 6-face probes of the primary set or
// of the cache, trilinearly over the 8 around it. `weight` is how much of it was there.
vec3 rc_face_irr(bool cache, vec3 p, vec3 face, out float weight) {
    int axis = rc_face_axis(face);
    int base = cache ? rc_cache_irr_base() : rc_l0_irr_base();
    int pbase = rc_probe_base_of(cache, 0);
    float sp = rc_spacing_of(cache, 0);
    vec3 g = (p + face * (0.5 * sp)) / sp - 0.5;
    ivec3 b = ivec3(floor(g));
    vec3 f = g - vec3(b);
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        ivec3 c = b + o;
        int slot = rc_lookup(c, 0, cache);
        if (slot < 0) continue;
        vec4 e = uintBitsToFloat(rc_ht[base + slot * 6 + axis].v);
        vec3 probe_pos = (vec3(c) + 0.5) * sp;
        // the probe has to be on the surface's own side of it: a coarse one this close to
        // the plane is as good as there, one across the wall is not there at all
        float fs = rc_min.w;
        float front = smoothstep(-0.35 * fs, 0.25 * fs, dot(probe_pos - p, face));
        float trust = rc_anchor_trust(c, sp, rc_meta[pbase + slot].b.x, p);
        // The cache is 4x coarser, so its probes around a point are often a wall away: its
        // own bounce light is kept only from the ones light can actually travel from.
        if (cache && rc_is_cache()) {
            vec3 d = probe_pos - p;
            float dist = length(d);
            float t_hit;
            ivec3 hc;
            vec3 hf;
            if (rc_march(p, d / max(dist, 1e-5), dist - rc_tune.w, t_hit, hc, hf) == 1) continue;
        }
        vec3 w3 = mix(1.0 - f, f, vec3(o));
        float w = w3.x * w3.y * w3.z * e.w * front * trust;
        sum += e.rgb * w;
        wsum += w;
    }
    // Nothing level 0 near: the level 1 probe over the same place has seen the same light
    // from further away, which beats going dark.
    if (wsum <= 1e-4 && !cache && rc_info.y > 1 && false) {
        float sp1 = sp * 2.0;
        ivec3 c1 = ivec3(floor(p / sp1));
        int slot = rc_lookup(c1, 1, false);
        if (slot >= 0) {
            vec4 e = uintBitsToFloat(rc_ht[rc_l1_irr_base() + slot * 6 + axis].v);
            weight = e.w;
            return e.rgb;
        }
    }
    weight = wsum;
    return wsum > 1e-4 ? sum / wsum : vec3(0.0);
}

// The light arriving at a surface from its surroundings: the secondary cache's, so it carries
// more than one bounce and light from around corners; the primary probes' own while it fills.
vec3 rc_bounce(vec3 p, vec3 face) {
    float w;
    if (rc_reads_cache()) {
        vec3 e = rc_face_irr(true, p, face, w);
        if (w > 1e-4) return e;
    }
    return rc_face_irr(false, p, face, w);
}

void main() {
    bool cache = rc_is_cache();
    uint ray_id = gl_GlobalInvocationID.x;
    uint total_rays = rc_rays[0].v;
    if (ray_id >= total_rays) return;

    uint packed = rc_rays[1u + ray_id].v;
    uint slot = packed & 0xffffu;
    uint local_r = packed >> 16;

    int levels = cache ? rc_cache_levels() : rc_info.y;
    int mi0 = rc_probe_base_of(cache, 0) + int(slot);
    uint seq_idx = rc_meta[mi0].b.y + local_r;
    uvec4 a0 = rc_meta[mi0].a;
    uvec4 b0 = rc_meta[mi0].b;
    ivec3 c0 = rc_unpack_coord(a0, b0);
    vec3 pos = (vec3(c0) + 0.5) * rc_spacing_of(cache, 0);

    // R2 quasi-random sequence in 32-bit integers + per-frame golden jitter
    uint f_u = uint(rc_tune.z);
    vec2 f_jit = fract(float(f_u) * vec2(0.7548776662, 0.5698402909));
    float u = fract(float(seq_idx * 0xC13FA9A9u) * 2.3283064365386963e-10 + f_jit.x);
    float v = fract(float(seq_idx * 0x91E10DA6u) * 2.3283064365386963e-10 + f_jit.y);
    vec2 cyl_uv = vec2(u, v);
    vec3 rd = rc_cyl_to_dir(cyl_uv);

    float t_max = rc_interval_end(levels - 1, cache);
    float t_hit;
    ivec3 cell;
    vec3 face;
    int what = rc_march(pos, rd, t_max, t_hit, cell, face);
    if (what == 3) {
        float nudge = rc_tune.w * 1.5;
        what = rc_march(pos + rd * nudge, rd, t_max - nudge, t_hit, cell, face);
        if (what == 1) t_hit += nudge;
    }

    int hit_level = levels;
    vec3 L = vec3(0.0);
    if (what == 1) {
        for (int l = 0; l < levels; l++) {
            if (t_hit <= rc_interval_end(l, cache)) {
                hit_level = l;
                break;
            }
        }
        rc_voxel vx = rc_vox_data[rc_voxel_index(cell)];
        vec3 albedo = rc_voxel_albedo(vx);
        L = rc_voxel_emissive(vx);
        vec3 hp = pos + rd * t_hit + face * (0.5 * rc_tune.w);
        uint seed = rc_hash(seq_idx ^ (f_u * 362437u));
        float w;
        vec3 bounce = cache ? rc_face_irr(true, hp, face, w) : rc_bounce(hp, face);
        L += albedo * (bounce * rc_sky.w + rc_direct_light(hp, face, seed)) / RC_PI;
        L = min(L, vec3(256.0));

        // Every fourth hit is a seed for the cache's probes, tagged with this frame
        if (!cache && (ray_id & 3u) == 0u) {
            uint at = (ray_id >> 2) % uint(rc_hits_cap());
            rc_hits[at].p = vec4(hp, float((f_u + 1u) * 8u + uint(rc_face_axis(face))));
        }
    } else if (what == 3) {
        return;
    }

    // Walk parent_slot chain from level 0 up to min(hit_level, levels - 1) and deposit:
    //   - for l < hit_level (including all levels on a miss where hit_level == levels):
    //     deposit (J = 0, beta = 1); merge at top level adds beta * sky_at(rd)
    //   - for l == hit_level (when hit_level < levels):
    //     deposit (J = L, beta = 0)
    int cur_slot = int(slot);
    for (int l = 0; l <= min(hit_level, levels - 1); l++) {
        if (cur_slot < 0) break;
        int theta = rc_info.x << l;
        int dirs = 2 * theta * theta;
        int d_idx = rc_dir_bin(cyl_uv, theta);
        int bin = rc_bin_off_of(cache, l) + cur_slot * dirs + d_idx;
        if (l < hit_level) {
            atomicAdd(rc_acc[bin].v.w, 0x00010001u);
        } else {
            uvec3 fp = uvec3(clamp(L * RC_FP_SCALE + 0.5, vec3(0.0), vec3(4.0e9)));
            if (fp.x > 0u) atomicAdd(rc_acc[bin].v.x, fp.x);
            if (fp.y > 0u) atomicAdd(rc_acc[bin].v.y, fp.y);
            if (fp.z > 0u) atomicAdd(rc_acc[bin].v.z, fp.z);
            atomicAdd(rc_acc[bin].v.w, 0x00000001u);
        }
        int cmi = rc_probe_base_of(cache, l) + cur_slot;
        uint pslot = rc_meta[cmi].a.z;
        cur_slot = pslot != 0xffffffffu ? int(pslot) : -1;
    }
}
@end

// Kernel 6: Top-down resolve, temporal blend, sparse trilinear merge, and L0 6-face gather
@cs rc_rays_merge_cs
@include_block rc_rays_common
layout(binding=1) buffer rc_merge_cascades { rc_rad_texel rc_rad[]; };
layout(binding=2) buffer rc_ht_merge { rc_uvec4 rc_ht[]; };
layout(binding=3) readonly buffer rc_meta_merge { rc_ray_meta rc_meta[]; };
layout(binding=4) readonly buffer rc_free_merge { rc_uint rc_free[]; };
layout(binding=5) buffer rc_hist_merge { rc_uvec2 rc_hist[]; };
layout(binding=6) buffer rc_acc_merge { rc_uvec4 rc_acc[]; };

layout(local_size_x=64) in;

@include_block rc_rays_lookup_common

vec3 rc_sky_at(vec3 rd) {
    return mix(rc_ground.rgb, rc_sky.rgb, 0.5 * rd.y + 0.5);
}

void main() {
    bool cache = rc_is_cache();
    int level = rc_pass_level();
    int nlevels = cache ? rc_cache_levels() : rc_info.y;
    int count = min(int(rc_free[rc_free_bump(cache, level)].v), rc_cap_of(cache, level));
    int theta = rc_info.x << level;
    int dirs = 2 * theta * theta;
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= count * dirs) return;

    int slot = idx / dirs;
    int k = idx - slot * dirs;
    int mi = rc_probe_base_of(cache, level) + slot;
    uvec4 a = rc_meta[mi].a;
    uvec4 b = rc_meta[mi].b;
    if (a.y == 0u) return;

    int bin = rc_bin_off_of(cache, level) + idx;
    uvec4 acc = rc_acc[bin].v;
    rc_acc[bin].v = uvec4(0u);

    vec3 J_prev;
    float beta_prev;
    uint n_prev;
    rc_unpack_hist(rc_hist[bin].p, J_prev, beta_prev, n_prev);

    uint cnt = acc.w & 0xffffu;
    uint beta_cnt = acc.w >> 16;
    vec3 J = J_prev;
    float beta = beta_prev;
    uint n_new = n_prev;
    if (cnt > 0u) {
        vec3 J_obs = vec3(acc.xyz) / (RC_FP_SCALE * float(cnt));
        float beta_obs = float(beta_cnt) / float(cnt);
        float alpha = (n_prev == 0u) ? 1.0 : max(rc_tune.y, float(cnt) / float(n_prev + cnt));
        J = mix(J_prev, J_obs, alpha);
        beta = mix(beta_prev, beta_obs, alpha);
        n_new = min(n_prev + cnt, 255u);
        rc_hist[bin].p = rc_pack_hist(J, beta, n_new);
    }

    vec3 rd = rc_bin_center_dir(k, theta);
    vec3 I = J;
    if (beta > 1e-4 && n_new > 0u) {
        if (level + 1 >= nlevels) {
            I += beta * rc_sky_at(rd);
        } else {
            int up = level + 1;
            int udirs = rc_dirs(up);
            int uoff = rc_bin_off_of(cache, up);
            float sp = rc_spacing_of(cache, level);
            float usp = sp * 2.0;
            ivec3 pc = rc_unpack_coord(a, b);
            vec3 pos = (vec3(pc) + 0.5) * sp;
            vec3 g = pos / usp - 0.5;
            ivec3 base_c = ivec3(floor(g));
            vec3 f = g - vec3(base_c);
            vec3 sum = vec3(0.0);
            float wsum = 0.0;
            // the corners nothing solid stands between this probe and (rc_vis_mask): light
            // from across a wall comes through the corner on the wall's far side barely at
            // all, and through nothing at all where that is the only corner there is
            uint vis = a.w >> RC_VISMASK_SHIFT;
            for (int n = 0; n < 8; n++) {
                ivec3 o = ivec3(n & 1, (n >> 1) & 1, (n >> 2) & 1);
                ivec3 uc = base_c + o;
                int uslot = rc_lookup(uc, up, cache);
                if (uslot < 0) continue;
                int ubase = uoff + uslot * udirs + k * 4;
                vec3 four = vec3(0.0);
                int valid = 0;
                for (int q = 0; q < 4; q++) {
                    if ((rc_hist[ubase + q].p.y >> 24) == 0u) continue;
                    four += rc_unpack_rad(rc_rad[ubase + q].p);
                    valid++;
                }
                if (valid == 0) continue;
                four /= float(valid);
                vec3 w3 = mix(1.0 - f, f, vec3(o));
                float vf = (vis == 0u || (vis & (1u << uint(n))) != 0u) ? 1.0 : RC_VIS_HIDDEN;
                float w = w3.x * w3.y * w3.z * vf;
                sum += four * w;
                wsum += w;
            }
            vec3 beyond = wsum > 1e-5 ? (sum / wsum) : rc_sky_at(rd);
            I += beta * beyond;
        }
    }
    I = min(I, vec3(256.0));
    rc_rad[bin].p = rc_pack_rad(I);
}
@end

// Kernel 6b: Gather L0 merged bins into 6-face irradiance (equal-area cylindrical directions)
@cs rc_rays_gather_cs
@include_block rc_rays_common
layout(binding=0) readonly buffer rc_g_bricks { rc_brick rc_brick_data[]; };
layout(binding=1) readonly buffer rc_g_cascades { rc_rad_texel rc_rad[]; };
layout(binding=2) buffer rc_ht_g { rc_uvec4 rc_ht[]; };
layout(binding=3) readonly buffer rc_meta_g { rc_ray_meta rc_meta[]; };
layout(binding=4) readonly buffer rc_free_g { rc_uint rc_free[]; };
layout(binding=5) readonly buffer rc_hist_g { rc_uvec2 rc_hist[]; };

layout(local_size_x=64) in;

@include_block rc_cell_common

// Where a probe's 6 faces go: level 0 into the block the materials read, level 1 into the
// coarser one they fall back to where no level 0 probe is near (only the primary has one).
int rc_gather_irr_base(bool cache, int level, int slot) {
    if (level == 0) return (cache ? rc_cache_irr_base() : rc_l0_irr_base()) + slot * 6;
    return rc_l1_irr_base() + slot * 6;
}

void main() {
    bool cache = rc_is_cache();
    int level = rc_pass_level();
    if (level > 0 && cache) return;
    int slot = int(gl_GlobalInvocationID.x);
    int count = min(int(rc_free[rc_free_bump(cache, level)].v), rc_cap_of(cache, level));
    if (slot >= count) return;

    int mi = rc_probe_base_of(cache, level) + slot;
    uvec4 a = rc_meta[mi].a;
    uvec4 b = rc_meta[mi].b;
    int irr_base = rc_gather_irr_base(cache, level, slot);
    if (a.y == 0u) {
        for (int f = 0; f < 6; f++) rc_ht[irr_base + f].v = uvec4(0u);
        return;
    }
    ivec3 pc = rc_unpack_coord(a, b);
    float sp = rc_spacing_of(cache, level);
    vec3 pos = (vec3(pc) + 0.5) * sp;
    ivec3 vc = ivec3(floor((pos - rc_min.xyz) / rc_tune.w));
    if (rc_in_grid(vc) && rc_solid(vc)) {
        for (int f = 0; f < 6; f++) rc_ht[irr_base + f].v = uvec4(0u);
        return;
    }
    if (level > 0) {
        // the materials find it from the probe's hash entry, without knowing the layout
        uint mask = uint(rc_info.w - 1);
        uint h = rc_hash(a.y) & mask;
        for (int step = 0; step < 32; step++) {
            uint cur = rc_ht[h].v.y;
            if (cur == a.y) {
                rc_ht[h].v.x = uint(irr_base);
                break;
            }
            if (cur == 0u) break;
            h = (h + 1u) & mask;
        }
    }

    int theta = rc_info.x << level;
    int dirs = 2 * theta * theta;
    int boff = rc_bin_off_of(cache, level) + slot * dirs;
    vec3 sum[6];
    float wsum[6];
    for (int f = 0; f < 6; f++) { sum[f] = vec3(0.0); wsum[f] = 0.0; }
    int valid = 0;
    vec3 fallback_avg = vec3(0.0);
    for (int k = 0; k < dirs; k++) {
        uint n = rc_hist[boff + k].p.y >> 24;
        if (n == 0u) continue;
        valid++;
        vec3 dir = rc_bin_center_dir(k, theta);
        vec3 rad = rc_unpack_rad(rc_rad[boff + k].p);
        fallback_avg += rad;
        // Equal-area bins all have identical solid angle 4*pi / dirs!
        for (int f = 0; f < 6; f++) {
            float c = dir[f >> 1] * ((f & 1) == 0 ? 1.0 : -1.0);
            float w = max(c + 0.15, 0.0);
            sum[f] += rad * w;
            wsum[f] += w;
        }
    }
    if (valid == 0) {
        for (int f = 0; f < 6; f++) rc_ht[irr_base + f].v = uvec4(0u);
        return;
    }
    fallback_avg *= RC_PI / float(valid);
    float trust = clamp(float(valid) / float(max(dirs / 2, 1)), 0.1, 1.0);
    for (int f = 0; f < 6; f++) {
        vec3 irr = wsum[f] > 1e-4 ? (RC_PI * sum[f] / wsum[f]) : fallback_avg;
        rc_ht[irr_base + f].v = floatBitsToUint(vec4(irr, trust));
    }
}
@end

@program rc_rays_clear rc_rays_clear_cs
@program rc_rays_rebuild rc_rays_rebuild_cs
@program rc_rays_alloc rc_rays_alloc_cs
@program rc_rays_alloc_ms rc_rays_alloc_ms_cs
@program rc_rays_cache_alloc rc_rays_cache_alloc_cs
@program rc_rays_propagate rc_rays_propagate_cs
@program rc_rays_offsets rc_rays_offsets_cs
@program rc_rays_args rc_rays_args_cs
@program rc_rays_trace rc_rays_trace_cs
@program rc_rays_merge rc_rays_merge_cs
@program rc_rays_gather rc_rays_gather_cs
