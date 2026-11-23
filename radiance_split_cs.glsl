// Sparse probe-based radiance cascades: world-space probes live only near surfaces the camera
// sees, in a GPU hash table, each cascade level's probes twice as far apart as the level
// below with 4x the directions. (Not the paper's split / binned rays: each probe marches its
// own rays over its level's interval and merges with the level above's probes.)
//   rc_alloc_cs        reads G-Buffer depth + normal and allocates world-space probes near
//                      visible surfaces across all cascade levels in a GPU hash table.
//   rc_propagate_cs    gives every probe the 8 probes of the level above around it.
//   rc_evict_cs        frees probes not seen lately and rebuilds the hash table.
//   rc_split_trace_cs  traces rays ONLY for allocated probes, each over its level's distance
//                      interval, and merges a ray that hits nothing with the 2x2 rays of the
//                      8 probes of the level above around it (hash-table lookups of L+1).
//   rc_split_gather_cs gathers L0 allocated probes' angular radiance into 6-face irradiance
//                      stored right after the hash table in `irradiance_buf` (so material
//                      shaders need no extra buffer binding).
@header module drie_shaders::rc_split;
@header import drie;

@ctype mat4 Mat4
@ctype vec4 Vec4

@include radiance_common.glsl

@block rc_split_common
@include_block rc_key_common
@include_block rc_pack_common
layout(binding=0) uniform rc_split_params {
    vec4 rc_min;       // xyz = voxel box min, w = L0 probe spacing (m)
    ivec4 rc_dims;     // xyz = voxel brick dims (vx/4, vy/4, vz/4), w = active cascade level
    vec4 rc_tune;      // x = L0 reach (m), y = blend, z = frame, w = voxel size (m)
    ivec4 rc_info;     // x = base_directions, y = levels, z = lights, w = hash_cap (power of 2)
    vec4 rc_sky;       // rgb = sky radiance, w = bounce strength
    vec4 rc_ground;    // rgb = ground radiance, w = normal_offset
    ivec4 rc_vox;      // xyz = voxel dims (vx, vy, vz), w = update_every
    ivec4 rc_lcap0;    // probe pool capacity for levels 0..3
    ivec4 rc_lcap1;    // probe pool capacity for levels 4..7
    ivec4 rc_loff0;    // ray buffer start offset (in rays) for levels 0..3
    ivec4 rc_loff1;    // ray buffer start offset (in rays) for levels 4..7
    ivec4 rc_pbase0;   // probe_meta start index for levels 0..3
    ivec4 rc_pbase1;   // probe_meta start index for levels 4..7
    mat4 rc_inv_vp;    // inverse relative view-projection for G-Buffer allocation
    vec4 rc_cam_pos;   // xyz = camera world pos, w = depth_zero_to_one (1 or 0)
    vec4 rc_screen;    // xy = 1/w, 1/h, z = flip_y (1 on Metal/D3D), w = screen width (px)
};

struct rc_probe_meta {
    ivec4 coord;       // xyz = probe grid coord (world integer), w = see rc_meta_w, -1 while the slot is free
};
// Stored inside irradiance_buf:
//   [0 .. hash_cap - 1]                 : uvec4(key_lo, key_hi, slot, last_frame)
//   [hash_cap .. hash_cap + lcap0*6 - 1]: vec4(irradiance.rgb, valid) via floatBitsToUint
struct rc_uvec4 {
    uvec4 v;
};
struct rc_uint {
    uint v;
};

// Probes untouched for this many frames are evicted (their hash slot freed and their data
// slot returned to the level's free list for reuse by newly seen surfaces).
const uint RC_STALE_FRAMES = 24u;
// rc_ht[].v.y states: 0u = never used (a safe linear-probe chain terminator for lookups that
// never walked through it), RC_TOMBSTONE = vacated (overflowed or evicted; still has to be
// skipped, not treated as a chain end, since other keys may have probed past it), anything
// else = an occupied key. Doesn't collide with rc_pack_key32's output (always has bit 31 set).
const uint RC_TOMBSTONE = 0x7fffffffu;

int rc_level_cap(int l) { return l < 4 ? rc_lcap0[l] : rc_lcap1[l - 4]; }

// rc_meta[].coord.w of a live probe: its level in bits 0-3 and, while it's new, the frame it
// was born in above them (rc_tune.z + 1: 1..4096; 0 once rc_evict_cs sees every one of its
// rays has been traced). A new probe, or one in a recycled slot, has nothing of its own in its
// rays yet (zeros, or the light of the probe that had the slot before): its first trace of
// each ray replaces what's there instead of blending into it, and until every ray has had
// that first trace, what reads its rays leaves out those that haven't (rc_ray_traced).
int rc_meta_w(int level, uint frame) { return level | (int(frame) << 4); }
uint rc_frame_age(uint now, uint then) { return now >= then ? now - then : now + 4096u - then; }
uint rc_probe_age(int w) { return rc_frame_age(uint(rc_tune.z) + 1u, uint(w >> 4)); }
// Whether ray `i` (its index among its level's rays) of a probe whose rc_meta w is `w` has been
// traced since the probe was born, this frame's trace included. rc_split_trace_cs traces ray
// t * every + (t + rc_tune.z) % every in each group t of `every` rays: one ray in `every` a
// frame, each in turn.
int rc_split_every() { return max(rc_vox.w & 0xffff, 1); }

bool rc_ray_traced(int w, int i) {
    int birth = w >> 4;
    if (birth == 0) return true;
    int every = rc_split_every();
    uint age = rc_probe_age(w);
    if (age + 1u >= uint(every)) return true;
    int t = i / every;
    int o = i - t * every;
    // frames from the probe's birth to this ray's first trace
    int wait = (o - (t + birth - 1) % every + every) % every;
    return uint(wait) <= age;
}
int rc_level_off(int l) { return l < 4 ? rc_loff0[l] : rc_loff1[l - 4]; }
int rc_probe_base(int l) { return l < 4 ? rc_pbase0[l] : rc_pbase1[l - 4]; }
@end

// Probe allocation, shared by rc_alloc_cs and rc_propagate_cs (both declare rc_ht, rc_meta,
// rc_cnt and rc_free under these names).
@block rc_touch_common
// Pops a recycled data slot for `level` from its free list (pushed by rc_evict_cs), or -1 if
// the list is empty. Lock-free stack: atomicAdd(-1) claims an entry by decrementing the top,
// then reverts if it went to/past zero (someone else already took the last one). Pushes only
// happen in rc_evict_cs, never alongside a pop.
int rc_free_pop(int level) {
    if (rc_free[level].v == 0u) return -1;
    int cap = rc_level_cap(level);
    int pbase = rc_probe_base(level);
    int old = int(atomicAdd(rc_free[level].v, uint(-1)));
    if (old > 0 && old <= cap) return int(rc_free[8 + pbase + (old - 1)].v);
    atomicAdd(rc_free[level].v, 1u);
    return -1;
}

// Marks the probe at `c` of `level` as seen this frame, allocating it if it's new and its
// level has a data slot left.
void rc_touch_probe(ivec3 c, int level, uint frame) {
    uint key = rc_pack_key32(c, level);
    uint mask = uint(rc_info.w - 1);
    uint idx = rc_hash(key) & mask;
    // Walk the chain to its end (a never-used slot) looking for the key, remembering the
    // first vacated slot on the way: inserting there before looking further would give a key
    // already further along a second entry.
    int open = -1;
    uint open_was = 0u;
    for (int step = 0; step < 32; step++) {
        uint cur = rc_ht[idx].v.y;
        if (cur == key) {
            rc_ht[idx].v.w = frame;
            return;
        }
        if (cur == 0u || cur == RC_TOMBSTONE) {
            if (open < 0) {
                open = int(idx);
                open_was = cur;
            }
            if (cur == 0u) break;
        }
        idx = (idx + 1u) & mask;
    }
    if (open < 0) return;
    // A full level (most of a big view's probes, every frame) gives up here, before any
    // atomic: claiming the slot only to vacate it again would leave a tombstone behind each
    // time, and a table full of them makes every lookup that misses walk all 32 steps.
    int cap = rc_level_cap(level);
    if (rc_free[level].v == 0u && rc_cnt[level].v >= uint(cap)) return;
    // Walk linear-probing chain, retrying on CAS collision with another key so concurrent
    // inserts never drop probes and leave blocky holes in the probe grid.
    uint cur_idx = (open >= 0) ? uint(open) : idx;
    for (int retry = 0; retry < 32; retry++) {
        uint cur = rc_ht[cur_idx].v.y;
        if (cur == key) {
            rc_ht[cur_idx].v.w = frame;
            return;
        }
        if (cur == 0u || cur == RC_TOMBSTONE) {
            uint prev = atomicCompSwap(rc_ht[cur_idx].v.y, cur, key);
            if (prev == cur) {
                int slot = rc_free_pop(level);
                if (slot < 0 && rc_cnt[level].v < uint(cap)) {
                    uint s = atomicAdd(rc_cnt[level].v, 1u);
                    slot = int(s) < cap ? int(s) : -1;
                }
                if (slot < 0) {
                    rc_ht[cur_idx].v.y = cur;
                    return;
                }
                rc_ht[cur_idx].v.x = rc_pack_xy(c);
                rc_ht[cur_idx].v.z = uint(slot);
                rc_ht[cur_idx].v.w = frame;
                rc_meta[rc_probe_base(level) + slot].coord = ivec4(c, rc_meta_w(level, frame));
                return;
            }
            if (prev == key) {
                rc_ht[cur_idx].v.w = frame;
                return;
            }
        }
        cur_idx = (cur_idx + 1u) & mask;
    }
}
@end

@cs rc_clear_cs
@include_block rc_split_common
layout(binding=2) buffer rc_ht_clear { rc_uvec4 rc_ht[]; };
layout(binding=4) buffer rc_cnt_clear { rc_uint rc_cnt[]; };
layout(binding=5) buffer rc_free_clear { rc_uint rc_free[]; };

layout(local_size_x=64) in;

int rc_free_buf_len() {
    int top = rc_info.y - 1;
    int cap = top < 4 ? rc_lcap0[top] : rc_lcap1[top - 4];
    int pbase = top < 4 ? rc_pbase0[top] : rc_pbase1[top - 4];
    return 8 + pbase + cap;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < uint(rc_info.w)) rc_ht[i].v = uvec4(0u);
    if (i < 8u) rc_cnt[i].v = 0u;
    if (int(i) < rc_free_buf_len()) rc_free[i].v = 0u;
}
@end

@cs rc_alloc_cs
@include_block rc_split_common
layout(binding=0) uniform texture2D rc_depth_tex;
layout(binding=1) uniform texture2D rc_normal_tex;
layout(binding=0) uniform sampler rc_gb_smp;
layout(binding=2) buffer rc_ht_alloc { rc_uvec4 rc_ht[]; };
layout(binding=3) buffer rc_meta_alloc { rc_probe_meta rc_meta[]; };
layout(binding=4) buffer rc_cnt_alloc { rc_uint rc_cnt[]; };
layout(binding=5) readonly buffer rc_vox_alloc { rc_voxel rc_vox_data[]; };
layout(binding=6) buffer rc_free_alloc { rc_uint rc_free[]; };

layout(local_size_x=8, local_size_y=8) in;

bool rc_probe_is_solid(ivec3 c, float sp) {
    vec3 pos = (vec3(c) + 0.5) * sp;
    ivec3 vc = ivec3(floor((pos - rc_min.xyz) / rc_tune.w));
    if (any(lessThan(vc, ivec3(0))) || any(greaterThanEqual(vc, rc_vox.xyz))) return false;
    return (rc_vox_data[(vc.z * rc_vox.y + vc.y) * rc_vox.x + vc.x].albedo & 0x1000000u) != 0u;
}

@include_block rc_touch_common

void main() {
    uint frame = uint(rc_tune.z) + 1u;

    // Seed coarse background levels (L >= 2) across the voxel box whenever they fit in the
    // level pool, so off-screen walls still carry multi-bounce light with zero L0 probes behind camera.
    uint row_w = uint(max((int(rc_screen.w) + 3) / 4, 1));
    uint gid = gl_GlobalInvocationID.y * row_w + gl_GlobalInvocationID.x;
    vec3 box_max = rc_min.xyz + vec3(rc_dims.xyz) * rc_min.w;
    for (int l = 2; l < rc_info.y; l++) {
        float sp = rc_min.w * float(1 << l);
        ivec3 cmin = ivec3(floor(rc_min.xyz / sp));
        ivec3 cmax = ivec3(ceil(box_max / sp));
        ivec3 cdim = max(cmax - cmin, ivec3(1));
        int total = cdim.x * cdim.y * cdim.z;
        if (total <= rc_level_cap(l) && int(gid) < total) {
            int idx = int(gid);
            int cx = idx % cdim.x;
            int cy = (idx / cdim.x) % cdim.y;
            int cz = idx / (cdim.x * cdim.y);
            ivec3 c = cmin + ivec3(cx, cy, cz);
            if (!rc_probe_is_solid(c, sp)) rc_touch_probe(c, l, frame);
        }
    }

    // Subsample G-Buffer by 4x4 pixels and allocate non-solid L0 probes around visible surfaces
    ivec2 px = ivec2(gl_GlobalInvocationID.xy) * 4 + ivec2(2);
    ivec2 sz = textureSize(sampler2D(rc_depth_tex, rc_gb_smp), 0);
    if (px.x >= sz.x || px.y >= sz.y) return;

    float raw_d = texelFetch(sampler2D(rc_depth_tex, rc_gb_smp), px, 0).r;
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
    ivec3 b0 = ivec3(floor(p / sp0 - 0.5));
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        ivec3 c = b0 + o;
        if (!rc_probe_is_solid(c, sp0)) rc_touch_probe(c, 0, frame);
    }
}
@end

@cs rc_propagate_cs
@include_block rc_split_common
layout(binding=2) buffer rc_ht_prop { rc_uvec4 rc_ht[]; };
layout(binding=3) buffer rc_meta_prop { rc_probe_meta rc_meta[]; };
layout(binding=4) buffer rc_cnt_prop { rc_uint rc_cnt[]; };
layout(binding=5) readonly buffer rc_vox_prop { rc_voxel rc_vox_data[]; };
layout(binding=6) buffer rc_free_prop { rc_uint rc_free[]; };

layout(local_size_x=64) in;

bool rc_prop_is_solid(ivec3 c, float sp) {
    vec3 pos = (vec3(c) + 0.5) * sp;
    ivec3 vc = ivec3(floor((pos - rc_min.xyz) / rc_tune.w));
    if (any(lessThan(vc, ivec3(0))) || any(greaterThanEqual(vc, rc_vox.xyz))) return false;
    return (rc_vox_data[(vc.z * rc_vox.y + vc.y) * rc_vox.x + vc.x].albedo & 0x1000000u) != 0u;
}

@include_block rc_touch_common

void main() {
    int level = rc_dims.w;
    int up = level + 1;
    if (up >= rc_info.y) return;
    int slot = int(gl_GlobalInvocationID.x);
    int count = min(int(rc_cnt[level].v), rc_level_cap(level));
    if (slot >= count) return;

    ivec4 pm = rc_meta[rc_probe_base(level) + slot].coord;
    if (pm.w < 0) return; // a freed slot (rc_free_push)
    ivec3 pc = pm.xyz;
    float sp = rc_min.w * float(1 << level);
    float usp = sp * 2.0;
    vec3 pos = (vec3(pc) + 0.5) * sp;
    ivec3 b = ivec3(floor(pos / usp - 0.5));
    uint frame = uint(rc_tune.z) + 1u;
    for (int n = 0; n < 8; n++) {
        ivec3 o = ivec3(n & 1, (n >> 1) & 1, (n >> 2) & 1);
        ivec3 c = b + o;
        if (!rc_prop_is_solid(c, usp)) rc_touch_probe(c, up, frame);
    }
}
@end

// Runs every RC_EVICT_PERIOD frames (driven from the c3 side) over the whole hash table, in
// two dispatches that rebuild it:
//   rc_dims.w = 0: moves each entry out to rc_rehash and clears its slot. Probes not touched
//                  in RC_STALE_FRAMES frames (a surface the camera hasn't seen lately) stay
//                  out, and their data slot goes back to the level's free list, so
//                  rc_touch_probe can reuse it for a newly seen surface instead of a level's
//                  pool staying full once its level_cap is first reached.
//   rc_dims.w = 1: inserts the kept entries again.
// Rebuilding instead of leaving tombstones where the stale probes were keeps the chains short:
// lookups only stop at a never-used slot, so a table whose holes are all tombstones makes every
// lookup that misses (most of them, at the edges of the probed surfaces) walk all 32 steps.
@cs rc_evict_cs
@include_block rc_split_common
layout(binding=0) buffer rc_ht_evict { rc_uvec4 rc_ht[]; };
layout(binding=1) buffer rc_free_evict { rc_uint rc_free[]; };
layout(binding=2) buffer rc_rehash_evict { rc_uvec4 rc_rehash[]; };
layout(binding=3) buffer rc_meta_evict { rc_probe_meta rc_meta[]; };

layout(local_size_x=64) in;

// Hands a data slot back to its level's free list and marks it free in rc_meta, so the
// passes that walk a level's slots (propagate, trace, gather) skip it until it's reused.
void rc_free_push(int level, uint slot) {
    int cap = rc_level_cap(level);
    int pbase = rc_probe_base(level);
    rc_meta[pbase + int(slot)].coord.w = -1;
    uint idx = atomicAdd(rc_free[level].v, 1u);
    if (int(idx) < cap) rc_free[8 + pbase + int(idx)].v = slot;
    else atomicAdd(rc_free[level].v, uint(-1));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(rc_info.w)) return;
    if (rc_dims.w == 0) {
        uvec4 e = rc_ht[idx].v;
        rc_ht[idx].v = uvec4(0u);
        rc_rehash[idx].v = uvec4(0u);
        if (e.y == 0u || e.y == RC_TOMBSTONE) return;
        int level = int((e.y >> 27) & 0x7u);
        uint frame = uint(rc_tune.z) + 1u;
        uint age = frame >= e.w ? (frame - e.w) : (frame + 4096u - e.w);
        if (age > RC_STALE_FRAMES) {
            rc_free_push(level, e.z);
            return;
        }
        int mi = rc_probe_base(level) + int(e.z);
        int w = rc_meta[mi].coord.w;
        // every ray traced at least once: no longer new (and the 4096-frame wrap of the
        // frame number can't make it look new again)
        if ((w >> 4) != 0 && rc_probe_age(w) >= uint(rc_split_every())) rc_meta[mi].coord.w = level;
        rc_rehash[idx].v = e;
        return;
    }
    uvec4 e = rc_rehash[idx].v;
    if (e.y == 0u) return;
    uint mask = uint(rc_info.w - 1);
    uint h = rc_hash(e.y) & mask;
    for (int step = 0; step < 32; step++) {
        uint prev = atomicCompSwap(rc_ht[h].v.y, 0u, e.y);
        if (prev == 0u) {
            rc_ht[h].v.x = e.x;
            rc_ht[h].v.z = e.z;
            rc_ht[h].v.w = e.w;
            return;
        }
        // the same probe twice (an insert race): keep one entry, free the other's data slot
        if (prev == e.y) break;
        h = (h + 1u) & mask;
    }
    rc_free_push(int((e.y >> 27) & 0x7u), e.z);
}
@end

@cs rc_split_trace_cs
@include_block rc_split_common
layout(binding=0) readonly buffer rc_split_voxels { rc_voxel rc_vox_data[]; };
layout(binding=1) buffer rc_split_cascades { rc_rad_texel rc_rad[]; };
layout(binding=2) readonly buffer rc_ht_trace { rc_uvec4 rc_ht[]; };
layout(binding=3) readonly buffer rc_split_lights { rc_light rc_light_data[]; };
layout(binding=4) readonly buffer rc_split_bricks { rc_brick rc_brick_data[]; };
layout(binding=5) readonly buffer rc_meta_trace { rc_probe_meta rc_meta[]; };
layout(binding=6) readonly buffer rc_cnt_trace { rc_uint rc_cnt[]; };
@include_block rc_trace_common

layout(local_size_x=64) in;

int rc_lookup_probe(ivec3 c, int level) {
    uint key = rc_pack_key32(c, level);
    uint mask = uint(rc_info.w - 1);
    uint idx = rc_hash(key) & mask;
    for (int step = 0; step < 32; step++) {
        uint cur = rc_ht[idx].v.y;
        if (cur == 0u) return -1;
        if (cur == key) return int(rc_ht[idx].v.z);
        // RC_TOMBSTONE: an overflowed or evicted slot — skip it, a later key in this chain
        // may still have probed past it to an occupied slot further along.
        idx = (idx + 1u) & mask;
    }
    return -1;
}

// Whether the segment from `a` to the probe at `probe` crosses a wall, by the same two solid
// voxel tests the cascade merge makes (at 35% and 65% of the way), plus the probe's own voxel.
bool rc_probe_walled(vec3 a, vec3 probe) {
    ivec3 pv = ivec3(floor((probe - rc_min.xyz) / rc_tune.w));
    if (rc_in_grid(pv) && rc_solid(pv)) return true;
    ivec3 m1 = ivec3(floor((mix(a, probe, 0.35) - rc_min.xyz) / rc_tune.w));
    ivec3 m2 = ivec3(floor((mix(a, probe, 0.65) - rc_min.xyz) / rc_tune.w));
    return (rc_in_grid(m1) && rc_solid(m1)) || (rc_in_grid(m2) && rc_solid(m2));
}

// The light arriving at `p` (just off a surface facing `face`) from the hemisphere around
// `face`, from cascade `level`'s probes: each of the up to 8 probes around that exists, isn't
// behind a wall from `p`, and has rays traced (rc_ray_traced) gives a cosine-weighted sum of
// 16 of its rays, and those are blended by their trilinear weights. -1 when none of them does.
vec3 rc_level_hemisphere(vec3 p, vec3 face, int level) {
    float sp = rc_min.w * float(1 << level);
    int d = rc_info.x << level;
    int dd = d * d;
    int off = rc_level_off(level);
    int pbase = rc_probe_base(level);
    int step = max(d / 4, 1);
    vec3 g = p / sp - 0.5;
    ivec3 b = ivec3(floor(g));
    vec3 f = g - vec3(b);
    vec3 total = vec3(0.0);
    float total_w = 0.0;
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        int slot = rc_lookup_probe(b + o, level);
        if (slot < 0) continue;
        if (rc_probe_walled(p, (vec3(b + o) + 0.5) * sp)) continue;
        int mw = rc_meta[pbase + slot].coord.w;
        int base = slot * dd;
        vec3 sum = vec3(0.0);
        float wsum = 0.0;
        for (int ky = 0; ky < d; ky += step) {
            for (int kx = 0; kx < d; kx += step) {
                if (!rc_ray_traced(mw, base + ky * d + kx)) continue;
                vec2 e = (vec2(kx, ky) + 0.5) / float(d) * 2.0 - 1.0;
                vec3 rd = rc_direction(e);
                float c = dot(rd, face);
                if (c <= 0.0) continue;
                vec3 flat_d = vec3(e, 1.0 - abs(e.x) - abs(e.y));
                float len = length(flat_d);
                float w = c / (len * len * len);
                sum += rc_unpack_rad(rc_rad[off + base + ky * d + kx].p) * w;
                wsum += w;
            }
        }
        if (wsum <= 1e-4) continue;
        vec3 w3 = mix(1.0 - f, f, vec3(o));
        float tw = w3.x * w3.y * w3.z;
        total += (sum / wsum) * tw;
        total_w += tw;
    }
    return total_w > 1e-5 ? RC_PI * total / total_w : vec3(-1.0);
}

vec3 rc_sparse_bounce(vec3 p, vec3 face) {
    int axis = face.x != 0.0 ? (face.x > 0.0 ? 0 : 1) : (face.y != 0.0 ? (face.y > 0.0 ? 2 : 3) : (face.z > 0.0 ? 4 : 5));
    vec3 g = p / rc_min.w - 0.5;
    ivec3 b = ivec3(floor(g));
    vec3 f = g - vec3(b);
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    int irr_base = rc_info.w;
    for (int k = 0; k < 8; k++) {
        ivec3 o = ivec3(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        int slot = rc_lookup_probe(b + o, 0);
        if (slot < 0) continue;
        vec4 e = uintBitsToFloat(rc_ht[irr_base + slot * 6 + axis].v);
        vec3 w3 = mix(1.0 - f, f, vec3(o));
        float w = w3.x * w3.y * w3.z * e.w;
        sum += e.rgb * w;
        wsum += w;
    }
    if (wsum > 1e-4) return sum / wsum;
    // An off-screen surface (no L0 probes there): the coarse levels' rays, from the finest
    // level that has probes around it. Levels 2 and up are seeded over the whole box when they
    // fit in their pool (rc_alloc_cs), which a big box's level 2 may not: then level 3 is.
    for (int l = 1; l < rc_info.y; l++) {
        vec3 fb = rc_level_hemisphere(p, face, l);
        if (fb.x >= 0.0) return fb;
    }
    return vec3(0.0);
}

void main() {
    int level = rc_dims.w;
    int count = min(int(rc_cnt[level].v), rc_level_cap(level));
    int d = rc_info.x << level;
    int dd = d * d;
    int total_rays = count * dd;
    int every = rc_every();
    int turn = int(gl_GlobalInvocationID.x);
    int i = turn * every + (turn + int(rc_tune.z)) % every;
    if (i >= total_rays) return;

    int slot = i / dd;
    int k = i - slot * dd;
    ivec4 pm = rc_meta[rc_probe_base(level) + slot].coord;
    if (pm.w < 0) return; // a freed slot (rc_free_push)
    ivec3 pc = pm.xyz;
    float spacing = rc_min.w * float(1 << level);
    vec3 pos = (vec3(pc) + 0.5) * spacing;

    // Low-discrepancy R2 quasi-random angular jitter per ray update (eliminates white-noise shimmer)
    uint cell_h = rc_hash(uint(i) * 2654435761u + uint(level) * 7919u);
    vec2 cell_seed = vec2(float(cell_h & 0xffffu), float(cell_h >> 16)) / 65536.0;
    float r2_step = float(int(rc_tune.z) / every);
    vec2 jitter = fract(cell_seed + r2_step * vec2(0.7548776662, 0.5698402909));
    uint h = rc_hash(cell_h ^ (uint(rc_tune.z) * 362437u));
    ivec2 uv = ivec2(k % d, k / d);
    vec3 rd = rc_direction((vec2(uv) + jitter) / float(d) * 2.0 - 1.0);

    float base = rc_tune.x;
    float t0 = base * (pow(4.0, float(level)) - 1.0) / 3.0;
    float t1 = base * (pow(4.0, float(level + 1)) - 1.0) / 3.0;

    float t_hit;
    ivec3 cell;
    vec3 face;
    float t_start = t0;
    int what = rc_march(pos + rd * t_start, rd, t1 - t_start, t_hit, cell, face);
    // Recover lit front face when an upper-cascade ray starts just inside a wall
    if (what == 3 && level > 0) {
        ivec3 pvc = ivec3(floor((pos - rc_min.xyz) / rc_tune.w));
        if (rc_in_grid(pvc) && !rc_solid(pvc)) {
            t_start = max(t0 - spacing * 0.75, 0.0);
            what = rc_march(pos + rd * t_start, rd, t1 - t_start, t_hit, cell, face);
        }
    }
    vec3 result = vec3(0.0);
    if (what == 1) {
        rc_voxel vx = rc_vox_data[rc_voxel_index(cell)];
        vec3 albedo = rc_voxel_albedo(vx);
        vec3 light = rc_voxel_emissive(vx);
        vec3 hp = pos + rd * (t_start + t_hit) + face * (0.5 * rc_tune.w);
        vec3 arriving = rc_sparse_bounce(hp, face) * rc_sky.w + rc_direct_light(hp, face, h);
        light += albedo * arriving / RC_PI;
        result = light;
    } else if (what == 2) {
        result = rc_sky_at(rd);
    } else if (what == 0) {
        if (level + 1 >= rc_info.y) {
            result = rc_sky_at(rd);
        } else {
            int up = level + 1;
            int ud = d * 2;
            int uoff = rc_level_off(up);
            float usp = spacing * 2.0;
            vec3 g = pos / usp - 0.5;
            ivec3 b = ivec3(floor(g));
            vec3 f = g - vec3(b);
            vec3 sum = vec3(0.0);
            float wsum = 0.0;
            vec3 sum_any = vec3(0.0);
            float wsum_any = 0.0;
            for (int n = 0; n < 8; n++) {
                ivec3 o = ivec3(n & 1, (n >> 1) & 1, (n >> 2) & 1);
                ivec3 uc = b + o;
                int uslot = rc_lookup_probe(uc, up);
                if (uslot < 0) continue;
                vec3 up_pos = (vec3(uc) + 0.5) * usp;
                ivec3 uvc = ivec3(floor((up_pos - rc_min.xyz) / rc_tune.w));
                if (rc_in_grid(uvc) && rc_solid(uvc)) continue;
                // the 2x2 rays of the probe above that this ray's cone covers, those of them
                // traced (see rc_ray_traced) scaled up to stand for all four
                int uw = rc_meta[rc_probe_base(up) + uslot].coord.w;
                int ur = uslot * ud * ud + (uv.y * 2) * ud + uv.x * 2;
                vec3 four = vec3(0.0);
                int traced = 0;
                for (int q = 0; q < 4; q++) {
                    int r = ur + (q >> 1) * ud + (q & 1);
                    if (!rc_ray_traced(uw, r)) continue;
                    four += rc_unpack_rad(rc_rad[uoff + r].p);
                    traced++;
                }
                if (traced == 0) continue;
                four *= 4.0 / float(traced);
                vec3 w3 = mix(1.0 - f, f, vec3(o));
                float w = w3.x * w3.y * w3.z;
                sum_any += four * (0.25 * w);
                wsum_any += w;
                ivec3 m1 = ivec3(floor((mix(pos, up_pos, 0.35) - rc_min.xyz) / rc_tune.w));
                ivec3 m2 = ivec3(floor((mix(pos, up_pos, 0.65) - rc_min.xyz) / rc_tune.w));
                if ((rc_in_grid(m1) && rc_solid(m1)) || (rc_in_grid(m2) && rc_solid(m2))) continue;
                sum += four * (0.25 * w);
                wsum += w;
            }
            vec3 fb = wsum_any > 1e-4 ? (sum_any / wsum_any) : rc_sky_at(rd);
            vec3 pri = wsum > 1e-4 ? (sum / wsum) : fb;
            float tf = clamp(wsum / max(wsum_any * 0.25, 1e-4), 0.0, 1.0);
            result = mix(fb, pri, tf);
        }
    }
    result = min(result, vec3(256.0));
    int at = rc_level_off(level) + i;
    vec3 prev = rc_unpack_rad(rc_rad[at].p);
    // a new probe's first trace of this ray replaces what the slot held (see rc_meta_w)
    bool first = (pm.w >> 4) != 0 && rc_probe_age(pm.w) < uint(every);
    rc_rad[at].p = rc_pack_rad(first ? result : mix(prev, result, rc_tune.y));
}
@end

@cs rc_split_gather_cs
@include_block rc_split_common
layout(binding=0) readonly buffer rc_gather_voxels { rc_voxel rc_vox_data[]; };
layout(binding=1) readonly buffer rc_split_gather_cascades { rc_rad_texel rc_rad[]; };
layout(binding=2) buffer rc_ht_gather { rc_uvec4 rc_ht[]; };
layout(binding=5) readonly buffer rc_meta_gather { rc_probe_meta rc_meta[]; };
layout(binding=6) readonly buffer rc_cnt_gather { rc_uint rc_cnt[]; };

layout(local_size_x=64) in;

bool rc_in_grid(ivec3 c) {
    return all(greaterThanEqual(c, ivec3(0))) && all(lessThan(c, rc_vox.xyz));
}

bool rc_solid(ivec3 c) {
    return (rc_vox_data[(c.z * rc_vox.y + c.y) * rc_vox.x + c.x].albedo & 0x1000000u) != 0u;
}

void main() {
    int slot = int(gl_GlobalInvocationID.x);
    int count = min(int(rc_cnt[0].v), rc_level_cap(0));
    if (slot >= count) return;
    ivec4 pm = rc_meta[rc_probe_base(0) + slot].coord;
    if (pm.w < 0) return; // a freed slot (rc_free_push): nothing looks it up
    ivec3 pc = pm.xyz;
    vec3 pos = (vec3(pc) + 0.5) * rc_min.w;
    ivec3 vc = ivec3(floor((pos - rc_min.xyz) / rc_tune.w));
    int irr_base = rc_info.w + slot * 6;
    if (rc_in_grid(vc) && rc_solid(vc)) {
        for (int a = 0; a < 6; a++) rc_ht[irr_base + a].v = uvec4(0u);
        return;
    }
    int d = rc_info.x;
    vec3 sum[6];
    float wsum[6];
    for (int a = 0; a < 6; a++) { sum[a] = vec3(0.0); wsum[a] = 0.0; }
    // a new probe's rays not traced yet are left out (rc_ray_traced), and its irradiance
    // counts for as much as the share of them that are (w), so probes around that have all
    // theirs outweigh it
    int traced = 0;
    vec3 avg_rad = vec3(0.0);
    for (int k = 0; k < d * d; k++) {
        if (!rc_ray_traced(pm.w, slot * d * d + k)) continue;
        traced++;
        vec2 e = (vec2(k % d, k / d) + 0.5) / float(d) * 2.0 - 1.0;
        vec3 flat_dir = vec3(e, 1.0 - abs(e.x) - abs(e.y));
        float len = length(flat_dir);
        float solid_angle = 1.0 / (len * len * len);
        vec3 dir = rc_direction(e);
        vec3 radiance = rc_unpack_rad(rc_rad[rc_loff0.x + slot * d * d + k].p);
        avg_rad += radiance;
        for (int a = 0; a < 6; a++) {
            float c = dir[a >> 1] * ((a & 1) == 0 ? 1.0 : -1.0);
            float w = max(c + 0.15, 0.0) * solid_angle;
            sum[a] += radiance * w;
            wsum[a] += w;
        }
    }
    if (traced == 0) {
        for (int a = 0; a < 6; a++) rc_ht[irr_base + a].v = uvec4(0u);
        return;
    }
    avg_rad *= RC_PI / float(traced);
    for (int a = 0; a < 6; a++) {
        vec3 face_irr = wsum[a] > 1e-3 ? (RC_PI * sum[a] / wsum[a]) : avg_rad;
        vec4 outv = vec4(face_irr, float(traced) / float(d * d));
        rc_ht[irr_base + a].v = floatBitsToUint(outv);
    }
}
@end

@program rc_clear rc_clear_cs
@program rc_alloc rc_alloc_cs
@program rc_propagate rc_propagate_cs
@program rc_evict rc_evict_cs
@program rc_split_trace rc_split_trace_cs
@program rc_split_gather rc_split_gather_cs
