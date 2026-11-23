// Split Radiance Cascades (Freeman, Sannikov & Pitt-Francis, arXiv:2607.20384):
//   0. rc_rays_clear_cs      zero hash table, probe meta, free stacks, history and accumulators
//   1. rc_rays_rebuild_cs    rebuild hash table every frame from live meta slots; evict stale/OOB
//   2. rc_rays_alloc_cs      per traced G-Buffer pixel: touch nearest L0 probe, count rays
//   3. rc_rays_propagate_cs  bottom-up: each active probe touches its single parent (c >> 1)
//   4. rc_rays_offsets_cs    top-down hierarchical R2 prefix segments (Algorithm 3)
//   5. rc_rays_trace_cs      surface-driven R2 ray march + multi-cascade interval deposit
//   6. rc_rays_merge_cs      top-down resolve, per-bin temporal blend, sparse trilinear merge,
//                            and L0 6-face irradiance integration
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
    ivec4 rc_dims;     // xyz = voxel brick dims (nx, ny, nz), w = active level / phase
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

// 32 bytes per probe slot:
//   a: x = pack_xy(c), y = key32 (0 = free), z = parent_slot, w = last_frame | (birth << 16)
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

const uint RC_RAYS_STALE = 24u;
const float RC_FP_SCALE = 16384.0;
const uint RC_MAX_TRACE_RAYS = 1048576u;

int rc_level_cap(int l) { return l < 4 ? rc_lcap0[l] : rc_lcap1[l - 4]; }
int rc_level_off(int l) { return l < 4 ? rc_loff0[l] : rc_loff1[l - 4]; }
int rc_probe_base(int l) { return l < 4 ? rc_pbase0[l] : rc_pbase1[l - 4]; }

int rc_total_meta() {
    int top = rc_info.y - 1;
    return rc_probe_base(top) + rc_level_cap(top);
}

int rc_total_bins() {
    int top = rc_info.y - 1;
    int th = rc_info.x << top;
    return rc_level_off(top) + rc_level_cap(top) * 2 * th * th;
}

int rc_level_of_slot(int mi, out int s) {
    for (int l = rc_info.y - 1; l >= 0; l--) {
        int pb = rc_probe_base(l);
        if (mi >= pb) {
            s = mi - pb;
            return l;
        }
    }
    s = mi;
    return 0;
}

ivec3 rc_unpack_coord(uvec4 a, uvec4 b) {
    int ix = int(a.x & 0xffffu) - 32768;
    int iy = int((a.x >> 16) & 0xffffu) - 32768;
    int iz = int(b.w);
    return ivec3(ix, iy, iz);
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
float rc_interval_end(int l) {
    return rc_tune.x * pow(4.0, float(l));
}
@end

// Shared lock-free probe touch/insert used by alloc and propagate
// free_buf layout:
//   [0..7]                 : free stack top per level
//   [8..15]                : bump allocated count per level
//   [16]                   : global ray prefix counter (for offsets kernel)
//   [24 .. 24+total_meta-1]: recycled slot indices per level
@block rc_rays_touch_common
int rc_rays_free_pop(int level) {
    if (rc_free[level].v == 0u) return -1;
    int cap = rc_level_cap(level);
    int pbase = rc_probe_base(level);
    int old = int(atomicAdd(rc_free[level].v, uint(-1)));
    if (old > 0 && old <= cap) return int(rc_free[24 + pbase + (old - 1)].v);
    atomicAdd(rc_free[level].v, 1u);
    return -1;
}

int rc_rays_touch_probe(ivec3 c, int level, uint frame) {
    uint key = rc_pack_key32(c, level);
    uint mask = uint(rc_info.w - 1);
    uint idx = rc_hash(key) & mask;
    int cap = rc_level_cap(level);
    for (int step = 0; step < 32; step++) {
        uint cur = rc_ht[idx].v.y;
        if (cur == key) {
            rc_ht[idx].v.w = frame;
            uint s = rc_ht[idx].v.z;
            if (s != 0xffffffffu) {
                int mi = rc_probe_base(level) + int(s);
                rc_meta[mi].a.w = (rc_meta[mi].a.w & 0xffff0000u) | frame;
                return int(s);
            }
            return -1;
        }
        if (cur == 0u) {
            if (rc_free[level].v == 0u && rc_free[8 + level].v >= uint(cap)) return -1;
            uint prev = atomicCompSwap(rc_ht[idx].v.y, 0u, key);
            if (prev == 0u) {
                int slot = rc_rays_free_pop(level);
                if (slot < 0 && rc_free[8 + level].v < uint(cap)) {
                    uint s = atomicAdd(rc_free[8 + level].v, 1u);
                    slot = int(s) < cap ? int(s) : -1;
                }
                if (slot < 0) {
                    rc_ht[idx].v.y = 0u;
                    return -1;
                }
                uint pxy = rc_pack_xy(c);
                int mi = rc_probe_base(level) + slot;
                rc_meta[mi].a = uvec4(pxy, key, 0xffffffffu, frame | (frame << 16));
                rc_meta[mi].b = uvec4(1u, 0u, 0u, uint(c.z));
                rc_ht[idx].v.x = pxy;
                rc_ht[idx].v.z = uint(slot);
                rc_ht[idx].v.w = frame;
                return slot;
            }
            if (prev == key) {
                rc_ht[idx].v.w = frame;
                uint s = rc_ht[idx].v.z;
                if (s != 0xffffffffu) {
                    int mi = rc_probe_base(level) + int(s);
                    rc_meta[mi].a.w = (rc_meta[mi].a.w & 0xffff0000u) | frame;
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

layout(local_size_x=64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    int tmeta = rc_total_meta();
    int tbins = rc_total_bins();
    if (i < uint(rc_info.w)) rc_ht[i].v = uvec4(0u, 0u, 0xffffffffu, 0u);
    if (i < uint(tmeta)) {
        rc_meta[i].a = uvec4(0u);
        rc_meta[i].b = uvec4(0u);
    }
    if (i < uint(24 + tmeta)) rc_free[i].v = 0u;
    if (i < uint(tbins)) {
        rc_hist[i].p = uvec2(0u);
        rc_acc[i].v = uvec4(0u);
    }
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

void rc_push_free(int level, uint slot) {
    int cap = rc_level_cap(level);
    int pbase = rc_probe_base(level);
    uint idx = atomicAdd(rc_free[level].v, 1u);
    if (int(idx) < cap) rc_free[24 + pbase + int(idx)].v = slot;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (rc_dims.w == 0) {
        if (i < uint(rc_info.w)) rc_ht[i].v = uvec4(0u, 0u, 0xffffffffu, 0u);
        if (i < 8u) rc_free[i].v = 0u;
        if (i == 16u) {
            rc_free[16].v = 0u;
            rc_free[17].v = 0u;
        }
        return;
    }
    int tmeta = rc_total_meta();
    if (int(i) >= tmeta) return;
    int slot;
    int level = rc_level_of_slot(int(i), slot);
    if (slot >= int(rc_free[8 + level].v)) return;

    uvec4 a = rc_meta[i].a;
    uvec4 b = rc_meta[i].b;
    uint key = a.y;
    if (key == 0u) {
        rc_push_free(level, uint(slot));
        return;
    }
    uint frame = uint(rc_tune.z) + 1u;
    uint last_f = a.w & 0xffffu;
    uint age = frame >= last_f ? (frame - last_f) : (frame + 4096u - last_f);

    // Check if probe is outside the current box (+ 1 cell margin)
    ivec3 c = rc_unpack_coord(a, b);
    float sp = rc_min.w * float(1 << level);
    vec3 pos = (vec3(c) + 0.5) * sp;
    vec3 bmax = rc_min.xyz + vec3(rc_dims.xyz) * rc_min.w;
    bool oob = any(lessThan(pos, rc_min.xyz - sp)) || any(greaterThan(pos, bmax + sp));

    if (age > RC_RAYS_STALE || oob) {
        rc_meta[i].a = uvec4(0u);
        rc_meta[i].b = uvec4(0u);
        // Zero history bins of freed slot so next occupant starts clean
        int th = rc_info.x << level;
        int dirs = 2 * th * th;
        int boff = rc_level_off(level) + slot * dirs;
        for (int k = 0; k < dirs; k++) rc_hist[boff + k].p = uvec2(0u);
        rc_push_free(level, uint(slot));
        return;
    }

    // Reset per-frame ray counters & parent link
    rc_meta[i].a.z = 0xffffffffu;
    rc_meta[i].b.x = 0u;
    rc_meta[i].b.y = 0u;
    rc_meta[i].b.z = 0u;

    // Re-insert into hash table
    uint mask = uint(rc_info.w - 1);
    uint h = rc_hash(key) & mask;
    for (int step = 0; step < 32; step++) {
        uint prev = atomicCompSwap(rc_ht[h].v.y, 0u, key);
        if (prev == 0u) {
            rc_ht[h].v.x = a.x;
            rc_ht[h].v.z = uint(slot);
            rc_ht[h].v.w = last_f;
            return;
        }
        if (prev == key) break;
        h = (h + 1u) & mask;
    }
    // Overflow/duplicate: free slot
    rc_meta[i].a = uvec4(0u);
    rc_push_free(level, uint(slot));
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

bool rc_cell_solid(ivec3 c, float sp) {
    vec3 pos = (vec3(c) + 0.5) * sp;
    ivec3 vc = ivec3(floor((pos - rc_min.xyz) / rc_tune.w));
    if (any(lessThan(vc, ivec3(0))) || any(greaterThanEqual(vc, rc_vox.xyz))) return false;
    ivec3 bc = vc >> 2;
    uvec2 mask = rc_brick_data[(bc.z * rc_dims.y + bc.y) * rc_dims.x + bc.x].mask;
    uint w = (vc.z & 2) != 0 ? mask.y : mask.x;
    int bit = ((vc.z & 1) << 4) | ((vc.y & 3) << 2) | (vc.x & 3);
    return (w & (1u << bit)) != 0u;
}

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

    int slot = rc_rays_touch_probe(c_near, 0, frame);
    if (slot >= 0) {
        atomicAdd(rc_meta[rc_probe_base(0) + slot].b.x, 1u);
    }
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
        int s = rc_rays_touch_probe(ck, 0, frame);
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

// Kernel 3: Bottom-up propagation (l = 0 .. levels - 2) in two phases per level:
//   phase 0 (rc_dims.w = l * 2):     touch/insert single nearest parent (c >> 1) in rc_ht
//   phase 1 (rc_dims.w = l * 2 + 1): look up finalized parent slot, set parent_slot, add rays
@cs rc_rays_propagate_cs
@include_block rc_rays_common
layout(binding=2) buffer rc_ht_prop { rc_uvec4 rc_ht[]; };
layout(binding=3) buffer rc_meta_prop { rc_ray_meta rc_meta[]; };
layout(binding=4) buffer rc_free_prop { rc_uint rc_free[]; };

layout(local_size_x=64) in;

@include_block rc_rays_touch_common

int rc_prop_lookup(ivec3 c, int level) {
    uint key = rc_pack_key32(c, level);
    uint mask = uint(rc_info.w - 1);
    uint idx = rc_hash(key) & mask;
    for (int step = 0; step < 32; step++) {
        uint cur = rc_ht[idx].v.y;
        if (cur == 0u) return -1;
        if (cur == key) return int(rc_ht[idx].v.z);
        idx = (idx + 1u) & mask;
    }
    return -1;
}

void main() {
    int level = rc_dims.w >> 1;
    int phase = rc_dims.w & 1;
    int up = level + 1;
    int slot = int(gl_GlobalInvocationID.x);
    int count = min(int(rc_free[8 + level].v), rc_level_cap(level));
    if (slot >= count) return;

    int mi = rc_probe_base(level) + slot;
    uvec4 a = rc_meta[mi].a;
    uvec4 b = rc_meta[mi].b;
    if (a.y == 0u) return;

    uint rays = b.x;
    if (rays == 0u) return;
    uint frame = uint(rc_tune.z) + 1u;
    if (phase == 0) {
        if (level == 0) {
            uint cap_r = uint(max(int(rc_screen.w), 1));
            uint birth = a.w >> 16;
            uint age = frame >= birth ? (frame - birth) : (frame + 4096u - birth);
            if (age < 4u) {
                rays = min(cap_r * 4u, 256u);
            } else {
                rays = clamp(rays, 4u, cap_r);
            }
            rc_meta[mi].b.x = rays;
        }
        if (up >= rc_info.y) return;
        ivec3 c = rc_unpack_coord(a, b);
        ivec3 uc = c >> 1;
        rc_rays_touch_probe(uc, up, frame);
        return;
    }
    if (up >= rc_info.y) return;
    ivec3 c = rc_unpack_coord(a, b);
    ivec3 uc = c >> 1;
    int uslot = rc_prop_lookup(uc, up);
    if (uslot < 0) return;
    rc_meta[mi].a.z = uint(uslot);
    atomicAdd(rc_meta[rc_probe_base(up) + uslot].b.x, rays);
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
    int level = rc_dims.w;
    int slot = int(gl_GlobalInvocationID.x);
    int count = min(int(rc_free[8 + level].v), rc_level_cap(level));
    if (slot >= count) return;

    int mi = rc_probe_base(level) + slot;
    uvec4 a = rc_meta[mi].a;
    uint rays = rc_meta[mi].b.x;
    if (a.y == 0u || rays == 0u) return;

    uint my_off = 0u;
    if (level == rc_info.y - 1) {
        my_off = atomicAdd(rc_free[16].v, rays);
    } else {
        uint uslot = a.z;
        if (uslot != 0xffffffffu) {
            int umi = rc_probe_base(level + 1) + int(uslot);
            uint sub = atomicAdd(rc_meta[umi].b.z, rays);
            my_off = rc_meta[umi].b.y + sub;
        }
    }
    rc_meta[mi].b.y = my_off;
    rc_meta[mi].b.z = 0u;

    if (level == 0) {
        uint t_off = atomicAdd(rc_free[17].v, rays);
        uint lim = min(t_off + rays, RC_MAX_TRACE_RAYS - 1u);
        for (uint r = t_off; r < lim; r++) {
            rc_rays[1u + r].v = uint(slot) | ((r - t_off) << 16);
        }
    }
}
@end

// Kernel 4b: Write indirect dispatch group counts from live probe/ray counters
//   rc_args[0]    : trace groups
//   rc_args[1..8] : merge groups for levels 0..7
//   rc_args[9]    : gather groups for level 0
@cs rc_rays_args_cs
@include_block rc_rays_common
layout(binding=4) readonly buffer rc_free_args { rc_uint rc_free[]; };
layout(binding=5) buffer rc_args_out { rc_uvec4 rc_args[]; };
layout(binding=6) buffer rc_rays_args { rc_uint rc_rays[]; };

layout(local_size_x=32) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i == 0u) {
        uint nr = min(rc_free[17].v, RC_MAX_TRACE_RAYS - 1u);
        rc_rays[0].v = nr;
        rc_args[0].v = uvec4(max((nr + 63u) / 64u, 1u), 1u, 1u, 0u);
        uint n0 = min(rc_free[8].v, uint(rc_level_cap(0)));
        rc_args[9].v = uvec4(max((n0 + 63u) / 64u, 1u), 1u, 1u, 0u);
    }
    if (i < uint(rc_info.y)) {
        int l = int(i);
        uint nl = min(rc_free[8 + l].v, uint(rc_level_cap(l)));
        int th = rc_info.x << l;
        uint dirs = uint(2 * th * th);
        rc_args[1 + l].v = uvec4(max((nl * dirs + 63u) / 64u, 1u), 1u, 1u, 0u);
    }
}
@end

// Kernel 5: Surface-driven R2 ray trace and multi-cascade interval deposit
@cs rc_rays_trace_cs
@include_block rc_rays_common
layout(binding=0) readonly buffer rc_trace_voxels { rc_voxel rc_vox_data[]; };
layout(binding=1) readonly buffer rc_trace_bricks { rc_brick rc_brick_data[]; };
layout(binding=2) readonly buffer rc_trace_lights { rc_light rc_light_data[]; };
layout(binding=3) readonly buffer rc_ht_trace { rc_uvec4 rc_ht[]; };
layout(binding=4) readonly buffer rc_meta_trace { rc_ray_meta rc_meta[]; };
layout(binding=5) readonly buffer rc_rays_trace { rc_uint rc_rays[]; };
layout(binding=6) buffer rc_acc_trace { rc_uvec4 rc_acc[]; };

@include_block rc_trace_common

layout(local_size_x=64) in;

int rc_lookup_l0(ivec3 c) {
    uint key = rc_pack_key32(c, 0);
    uint mask = uint(rc_info.w - 1);
    uint idx = rc_hash(key) & mask;
    for (int step = 0; step < 32; step++) {
        uint cur = rc_ht[idx].v.y;
        if (cur == 0u) return -1;
        if (cur == key) return int(rc_ht[idx].v.z);
        idx = (idx + 1u) & mask;
    }
    return -1;
}

vec3 rc_l0_bounce(vec3 p, vec3 face) {
    int axis = face.x != 0.0 ? (face.x > 0.0 ? 0 : 1) : (face.y != 0.0 ? (face.y > 0.0 ? 2 : 3) : (face.z > 0.0 ? 4 : 5));
    vec3 g = p / rc_min.w - 0.5;
    ivec3 b = ivec3(floor(g));
    vec3 f = g - vec3(b);
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    int irr_base = rc_info.w;
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        int slot = rc_lookup_l0(b + o);
        if (slot < 0) continue;
        vec4 e = uintBitsToFloat(rc_ht[irr_base + slot * 6 + axis].v);
        vec3 w3 = mix(1.0 - f, f, vec3(o));
        float w = w3.x * w3.y * w3.z * e.w;
        sum += e.rgb * w;
        wsum += w;
    }
    return wsum > 1e-4 ? sum / wsum : vec3(0.0);
}

void main() {
    uint ray_id = gl_GlobalInvocationID.x;
    uint total_rays = rc_rays[0].v;
    if (ray_id >= total_rays) return;

    uint packed = rc_rays[1u + ray_id].v;
    uint slot = packed & 0xffffu;
    uint local_r = packed >> 16;

    int mi0 = rc_probe_base(0) + int(slot);
    uint seq_idx = rc_meta[mi0].b.y + local_r;
    uvec4 a0 = rc_meta[mi0].a;
    uvec4 b0 = rc_meta[mi0].b;
    ivec3 c0 = rc_unpack_coord(a0, b0);
    vec3 pos = (vec3(c0) + 0.5) * rc_min.w;

    // R2 quasi-random sequence in 32-bit integers + per-frame golden jitter
    uint f_u = uint(rc_tune.z);
    vec2 f_jit = fract(float(f_u) * vec2(0.7548776662, 0.5698402909));
    float u = fract(float(seq_idx * 0xC13FA9A9u) * 2.3283064365386963e-10 + f_jit.x);
    float v = fract(float(seq_idx * 0x91E10DA6u) * 2.3283064365386963e-10 + f_jit.y);
    vec2 cyl_uv = vec2(u, v);
    vec3 rd = rc_cyl_to_dir(cyl_uv);

    int levels = rc_info.y;
    float t_max = rc_interval_end(levels - 1);
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
            if (t_hit <= rc_interval_end(l)) {
                hit_level = l;
                break;
            }
        }
        rc_voxel vx = rc_vox_data[rc_voxel_index(cell)];
        vec3 albedo = rc_voxel_albedo(vx);
        L = rc_voxel_emissive(vx);
        vec3 hp = pos + rd * t_hit + face * (0.5 * rc_tune.w);
        uint seed = rc_hash(seq_idx ^ (f_u * 362437u));
        vec3 arriving = rc_l0_bounce(hp, face) * rc_sky.w + rc_direct_light(hp, face, seed);
        L += albedo * arriving / RC_PI;
        L = min(L, vec3(256.0));
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
        int bin = rc_level_off(l) + cur_slot * dirs + d_idx;
        if (l < hit_level) {
            atomicAdd(rc_acc[bin].v.w, 0x00010001u);
        } else {
            uvec3 fp = uvec3(clamp(L * RC_FP_SCALE + 0.5, vec3(0.0), vec3(4.0e9)));
            if (fp.x > 0u) atomicAdd(rc_acc[bin].v.x, fp.x);
            if (fp.y > 0u) atomicAdd(rc_acc[bin].v.y, fp.y);
            if (fp.z > 0u) atomicAdd(rc_acc[bin].v.z, fp.z);
            atomicAdd(rc_acc[bin].v.w, 0x00000001u);
        }
        int cmi = rc_probe_base(l) + cur_slot;
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

int rc_lookup_probe(ivec3 c, int level) {
    uint key = rc_pack_key32(c, level);
    uint mask = uint(rc_info.w - 1);
    uint idx = rc_hash(key) & mask;
    for (int step = 0; step < 32; step++) {
        uint cur = rc_ht[idx].v.y;
        if (cur == 0u) return -1;
        if (cur == key) return int(rc_ht[idx].v.z);
        idx = (idx + 1u) & mask;
    }
    return -1;
}

vec3 rc_sky_at(vec3 rd) {
    return mix(rc_ground.rgb, rc_sky.rgb, 0.5 * rd.y + 0.5);
}

void main() {
    int level = rc_dims.w;
    int count = min(int(rc_free[8 + level].v), rc_level_cap(level));
    int theta = rc_info.x << level;
    int dirs = 2 * theta * theta;
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= count * dirs) return;

    int slot = idx / dirs;
    int k = idx - slot * dirs;
    int mi = rc_probe_base(level) + slot;
    uvec4 a = rc_meta[mi].a;
    uvec4 b = rc_meta[mi].b;
    if (a.y == 0u) return;

    int bin = rc_level_off(level) + idx;
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
        if (level + 1 >= rc_info.y) {
            I += beta * rc_sky_at(rd);
        } else {
            int up = level + 1;
            int utheta = theta * 2;
            int udirs = 2 * utheta * utheta;
            int uoff = rc_level_off(up);
            float sp = rc_min.w * float(1 << level);
            float usp = sp * 2.0;
            ivec3 pc = rc_unpack_coord(a, b);
            vec3 pos = (vec3(pc) + 0.5) * sp;
            vec3 g = pos / usp - 0.5;
            ivec3 base_c = ivec3(floor(g));
            vec3 f = g - vec3(base_c);
            vec3 sum = vec3(0.0);
            float wsum = 0.0;
            for (int n = 0; n < 8; n++) {
                ivec3 o = ivec3(n & 1, (n >> 1) & 1, (n >> 2) & 1);
                ivec3 uc = base_c + o;
                int uslot = rc_lookup_probe(uc, up);
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
                float w = w3.x * w3.y * w3.z;
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

bool rc_g_solid(ivec3 vc) {
    if (any(lessThan(vc, ivec3(0))) || any(greaterThanEqual(vc, rc_vox.xyz))) return false;
    ivec3 bc = vc >> 2;
    uvec2 mask = rc_brick_data[(bc.z * rc_dims.y + bc.y) * rc_dims.x + bc.x].mask;
    uint w = (vc.z & 2) != 0 ? mask.y : mask.x;
    int bit = ((vc.z & 1) << 4) | ((vc.y & 3) << 2) | (vc.x & 3);
    return (w & (1u << bit)) != 0u;
}

void main() {
    int slot = int(gl_GlobalInvocationID.x);
    int count = min(int(rc_free[8].v), rc_level_cap(0));
    if (slot >= count) return;

    int mi = rc_probe_base(0) + slot;
    uvec4 a = rc_meta[mi].a;
    uvec4 b = rc_meta[mi].b;
    int irr_base = rc_info.w + slot * 6;
    if (a.y == 0u) {
        for (int f = 0; f < 6; f++) rc_ht[irr_base + f].v = uvec4(0u);
        return;
    }
    ivec3 pc = rc_unpack_coord(a, b);
    vec3 pos = (vec3(pc) + 0.5) * rc_min.w;
    ivec3 vc = ivec3(floor((pos - rc_min.xyz) / rc_tune.w));
    if (rc_g_solid(vc)) {
        for (int f = 0; f < 6; f++) rc_ht[irr_base + f].v = uvec4(0u);
        return;
    }

    int theta = rc_info.x;
    int dirs = 2 * theta * theta;
    int boff = rc_level_off(0) + slot * dirs;
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
@program rc_rays_propagate rc_rays_propagate_cs
@program rc_rays_offsets rc_rays_offsets_cs
@program rc_rays_args rc_rays_args_cs
@program rc_rays_trace rc_rays_trace_cs
@program rc_rays_merge rc_rays_merge_cs
@program rc_rays_gather rc_rays_gather_cs
