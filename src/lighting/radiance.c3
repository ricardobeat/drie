// World-space Radiance Cascades global illumination (Sannikov 2023) with dense
// (`shaders/radiance_cs.glsl`) and sparse Split Radiance Cascades (`shaders/radiance_rays_cs.glsl`,
// arXiv:2607.20384) GPU compute paths. Voxelize solid meshes into a linear albedo/emissive grid
// with `voxelize()`, assign to `Scene.radiance`, then call `update()` before `Renderer.render()`
// for dense volumes (sparse volumes update automatically from the G-Buffer during rendering).
// Distances are in meters and lighting values are linear radiance; lit materials inside
// `[min, max]` take indirect light here instead of ambient, hemisphere, or irradiance-volume
// lighting. GPU resources allocate lazily in `upload()`, and `dispose()` releases them.

module drie;
import std::math, std::collections::list;
// `drie_shaders::rc_rays` lives outside `drie::*` because its generated shader structs share
// names with `drie::shaders` that collide under recursive imports.
import sokol, drie_shaders::rc_rays;

// ---- Constants & configuration

// Maximum cascade levels supported by uniform offsets and per-level arrays.
const int RC_MAX_LEVELS     = 8;
// Maximum scene lights uploaded per frame to the cascade trace shader.
const int RC_MAX_LIGHTS     = 64;
// Maximum surface rays stored in `rc_shared_rays` for one split-cascade trace dispatch.
const int RC_MAX_TRACE_RAYS = 1048576;

StorageBuffer* rc_shared_acc @local;
StorageBuffer* rc_shared_rays @local;
// Default frame interval for periodic hash-table eviction; the split path rebuilds the hash
// table every frame instead.
const int RC_EVICT_PERIOD = 8;
// Start offset of recycled slot lists in `free_buf` (`RC_FREE_BASE` in
// `shaders/radiance_rays_cs.glsl`): indices below hold free-stack tops, bump counters, and ray
// counters for the primary volume and secondary cache.
const int RC_FREE_BASE    = 40;

// Kernel bitmask flag for `RadianceCascades.skip_kernels` (used by
// `tests/radiance_split_bench.c3` to isolate per-kernel GPU timings): skips the clear pass.
const int RC_KERNEL_CLEAR     = 1;
// `skip_kernels` flag: skips the hash-table rebuild and eviction pass (sparse path).
const int RC_KERNEL_EVICT     = 2;
// `skip_kernels` flag: skips G-Buffer and secondary-cache probe allocation passes (sparse path).
const int RC_KERNEL_ALLOC     = 4;
// `skip_kernels` flag: skips hierarchy propagation, R2 offset, and indirect-arg passes (sparse).
const int RC_KERNEL_PROPAGATE = 8;
// `skip_kernels` flag: skips ray marching passes (dense and sparse paths).
const int RC_KERNEL_TRACE     = 16;
// `skip_kernels` flag: skips cascade merge and 6-face irradiance gather passes (dense and sparse).
const int RC_KERNEL_GATHER    = 32;

// World-space Radiance Cascades volume for real-time bounced indirect lighting and emissive glow.
//
//   RadianceCascades* rc = drie::radiance_cascades(room_min, room_max);
//   rc.voxelize(scene);
//   scene.radiance = rc;
//   // Each frame before render() (dense mode):
//   rc.update(scene);
//
// Cascade 0 places probes at `spacing` meters with `base_directions`^2 rays over `interval`
// meters; each coarser level doubles spacing, quadruples angular resolution, and quadruples ray
// length. Rays march through a 4x-resolution voxel copy of solid meshes and blend temporally
// across frames (`blend`).
struct RadianceCascades {
    Vector3 min; // World-space volume minimum corner (m).
    Vector3 max; // World-space volume maximum corner snapped to whole probe cells (m).
    float spacing; // Distance between cascade 0 probes (m).
    int nx; // Cascade 0 probe count along X (also brick count along X).
    int ny; // Cascade 0 probe count along Y (also brick count along Y).
    int nz; // Cascade 0 probe count along Z (also brick count along Z).
    int levels; // Configured cascade count (`1..RC_MAX_LEVELS`).
    int base_directions; // Directional grid width at cascade 0 (`4` gives 16 directions in dense mode).
    float interval; // Cascade 0 ray march distance (m).
    float voxel_size; // Edge length of one voxel (`spacing / 4`, in meters).
    int vx; // Voxel grid resolution along X (`nx * 4`).
    int vy; // Voxel grid resolution along Y (`ny * 4`).
    int vz; // Voxel grid resolution along Z (`nz * 4`).
    float blend; // Temporal blend weight of each new frame in probe history (`0..1`).
    // Ray-count threshold per direction bin at which a frame reaches full `blend` weight in split
    // mode; bins receiving fewer rays scale down proportionally (`1` applies full `blend` to all).
    float blend_rays;
    float intensity; // Scalar multiplier on indirect light sampled by lit materials.
    float bounce; // Fraction of previous-frame probe irradiance bounced on voxel hits (`0..1`).
    Vector3 sky_color; // Overhead linear sky radiance for rays exiting the box.
    Vector3 ground_color; // Lower-hemisphere linear ground radiance for rays exiting the box.
    bool sky_from_scene; // When true, derives `sky_color` and `ground_color` from scene ambient/hemisphere lights.
    // Maximum linear RGB component allowed in voxel emissive glow (`0` = unlimited), applied
    // during `voxelize()`. Clamping tiny bright bulbs avoids wandering fireflies while scene
    // point/spot lights handle their direct illumination.
    float max_glow;
    float normal_offset; // World-space offset along surface normal (m) applied when sampling probes.
    // CPU voxel array: 2 `uint`s per voxel (`rgb8` albedo + bit 24 solid flag; RGBE8 linear glow).
    uint[] voxels;
    bool voxels_changed; // True when `voxels` needs re-uploading and brick-mask rebuilding on GPU.
    int update_every; // Trace stride (`1` traces every ray/pixel; larger values amortize cost across frames).
    int shadow_rays; // Shadow rays per voxel hit (`1` = pure RIS; `> 1` = `shadow_rays - 1` deterministic + 1 RIS).
    StorageBuffer* voxel_buf; // GPU voxel albedo and glow storage buffer.
    StorageBuffer* brick_buf; // GPU 64-bit 4x4x4 brick occupancy bitmasks.
    StorageBuffer* cascade_buf; // GPU cascade radiance intervals (dense) or merged bins (split).
    StorageBuffer* irradiance_buf; // GPU 6-face probe irradiance (preceded by hash table in sparse mode).
    StorageBuffer* light_buf; // GPU per-frame scene light array (`RC_MAX_LIGHTS` entries).
    ComputePass* trace; // Dense cascade trace compute pass.
    ComputePass* gather; // Dense cascade 6-face irradiance gather pass.
    int[RC_MAX_LEVELS] offsets; // Starting ray/bin index per cascade level in dense mode.
    int[RC_MAX_LEVELS] counts; // Ray/bin count per cascade level in dense mode.
    int frame; // Frame counter driving jitter rotation and initial blend ramp.
    int skip_kernels; // Bitmask of `RC_KERNEL_*` stages to skip for benchmarking (`0` runs all).

    // Enables sparse Split Radiance Cascades (arXiv:2607.20384), allocating probes near visible
    // G-Buffer surfaces in a GPU hash table at the front of `irradiance_buf`.
    bool sparse;
    int rays_per_probe; // Maximum surface rays traced per L0 probe per frame in split mode (default 64).
    bool multibounce; // Enables the secondary 4x-spacing cache for multi-bounce light in split mode.
    float lod_distance; // Chebyshev base distance (m) for LOD rings (`0` = automatic).
    int max_lod; // Maximum LOD ring index (`0` disables LOD rings).
    bool freeze_alloc; // Pauses `update_sparse()` so debug cameras do not reallocate or advance probes.
    bool sparse_dirty; // Triggers full clear of hash table, metadata, and history on next split update.
    int hash_cap; // Power-of-two probe hash-table capacity in slots.
    int[RC_MAX_LEVELS] level_cap; // Per-level probe capacities for sparse layout sizing.
    int[RC_MAX_LEVELS] probe_base; // Per-level base probe index for sparse layout sizing.
    int split_levels; // Effective cascade count in split mode (drops levels starting past box diagonal).
    int[RC_MAX_LEVELS] split_cap; // Maximum primary probes per level in split mode: `max(P0 >> 2n, 64)`.
    int[RC_MAX_LEVELS] split_pbase; // Primary probe metadata start index per level in `meta_buf`.
    int[RC_MAX_LEVELS] split_offsets; // Primary bin start offset per level in `cascade_buf`/`hist_buf`.
    int[RC_MAX_LEVELS] split_counts; // Primary bin count per level in split mode.
    // Secondary multi-bounce cache (4x spacing, 1/16 primary L0 probe cap) seeded from primary
    // trace hits; its pools follow the primary pools in `meta_buf`, bin buffers, and `irradiance_buf`.
    int cache_levels;
    int[RC_MAX_LEVELS] cache_cap; // Maximum secondary cache probes per level.
    int[RC_MAX_LEVELS] cache_pbase; // Secondary cache metadata start index per level in `meta_buf`.
    int[RC_MAX_LEVELS] cache_offsets; // Secondary cache bin start offset per level.
    int[RC_MAX_LEVELS] cache_counts; // Secondary cache bin count per level.
    int cache_meta_base; // Start index of secondary cache metadata in `meta_buf` (total primary probes).
    int cache_bin_base; // Start index of secondary cache bins (total primary bins).
    int cache_irr_base; // Start offset of secondary cache 6-face irradiance in `irradiance_buf`.
    int hits_cap; // Maximum primary trace hit records stored in `hits_buf` to seed the cache.
    StorageBuffer* meta_buf; // Per-probe metadata (`32 B`/probe) for primary and secondary cache pools.
    // Counters (`0..RC_FREE_BASE-1`) followed by per-level free-slot stacks sized to match
    // `meta_buf` for recycling evicted probe slots.
    StorageBuffer* free_buf;
    int free_buf_len; // Element count (`uint`s) allocated in `free_buf`.
    StorageBuffer* hist_buf; // Per-bin temporal `(J, beta, count)` history in split mode (`8 B`/bin).
    StorageBuffer* args_buf; // Indirect compute dispatch arguments written on the GPU in split mode.
    StorageBuffer* hits_buf; // Surface hit positions written by primary trace for cache probe seeding.
    ComputePass* rays_clear_pass; // Split-mode buffer clear pass.
    ComputePass* rays_rebuild_pass; // Split-mode per-frame hash-table rebuild and eviction pass.
    ComputePass* rays_alloc_pass; // Split-mode single-sample G-Buffer probe allocation pass.
    ComputePass* rays_alloc_ms_pass; // Split-mode MSAA G-Buffer probe allocation pass.
    ComputePass* rays_cache_alloc_pass; // Split-mode secondary cache probe seeding pass.
    ComputePass* rays_propagate_pass; // Split-mode bottom-up parent probe propagation pass.
    ComputePass* rays_offsets_pass; // Split-mode top-down R2 ray-segment prefix-offset pass.
    ComputePass* rays_args_pass; // Split-mode indirect dispatch argument builder pass.
    ComputePass* rays_trace_pass; // Split-mode R2 surface ray marching pass.
    ComputePass* rays_merge_pass; // Split-mode top-down cascade resolve and merge pass.
    ComputePass* rays_gather_pass; // Split-mode 6-face irradiance gather pass.
    // Scene root last passed to `voxelize()` or `recenter()`; a different root forces a full
    // re-voxelize even when the snapped box origin did not move.
    Object3D* voxelized_root;
    // Active while `voxelize_region()` runs so `set_voxel()` and `add_triangle()` only modify
    // voxels inside `[clip_min, clip_max]`.
    bool clip_region;
    Vector3 clip_min; // World-space minimum corner of the active partial-voxelization box (m).
    Vector3 clip_max; // World-space maximum corner of the active partial-voxelization box (m).

    // Camera whose render drives sparse probe allocation and tracing. Leave null to defer the
    // update to `flush_pending_sparse()` at `drie::commit()` using the frame's last render.
    Camera* camera;
    bool pending_sparse; // True when a deferred `update_sparse()` awaits `flush_pending_sparse()`.
    Scene* pending_scene; // Scene recorded for the deferred sparse update.
    SgView pending_depth_tex; // G-Buffer depth view recorded for the deferred sparse update.
    SgView pending_normal_tex; // G-Buffer normal view recorded for the deferred sparse update.
    SgSampler pending_gb_sampler; // G-Buffer sampler recorded for the deferred sparse update.
    Mat4 pending_inv_view_proj; // Inverse view-projection recorded for the deferred sparse update.
    Vector3 pending_cam_pos; // Camera world position recorded for the deferred sparse update.
    int pending_width; // G-Buffer width recorded for the deferred sparse update.
    int pending_height; // G-Buffer height recorded for the deferred sparse update.
}

// ---- Construction

// Maximum probe cells per axis supported by the 32-bit probe key (9 bits per axis, keeping one
// boundary cell at the box's far edge to avoid key collisions).
const int RC_KEY_CELLS = 480;

// Doubles `spacing` up to 8 times until `size / spacing` fits within `RC_KEY_CELLS` on every
// axis so 32-bit probe keys stay collision-free across the box.
fn float rc_fit_spacing(Vector3 size, float spacing) {
    for (int i = 0; i < 8; i++) {
        Vector3 cells = size / spacing;
        if (cells.x <= RC_KEY_CELLS && cells.y <= RC_KEY_CELLS && cells.z <= RC_KEY_CELLS) break;
        spacing *= 2;
    }
    return spacing;
}

// Allocates a dense Radiance Cascades volume covering `min..max` with L0 probe `spacing` (m)
// and `levels` cascades (`1..RC_MAX_LEVELS`). No GPU resources are allocated until `upload()`.
fn RadianceCascades* radiance_cascades(Vector3 min, Vector3 max, float spacing = 0.25, int levels = 4) {
    RadianceCascades* r = mem::new(RadianceCascades);
    Vector3 size        = max - min;
    spacing = rc_fit_spacing(size, spacing);
    *r = {
        .min = min,
        .spacing = spacing,
        .nx = math::max((int)math::ceil(size.x / spacing), 1),
        .ny = math::max((int)math::ceil(size.y / spacing), 1),
        .nz = math::max((int)math::ceil(size.z / spacing), 1),
        .levels = math::clamp(levels, 1, RC_MAX_LEVELS),
        .base_directions = 4,
        .interval = spacing,
        .voxel_size = spacing / 4,
        .blend = 0.1,
        .blend_rays = 1,
        .update_every = 1,
        .shadow_rays = 1,
        .rays_per_probe = 64,
        .multibounce = true,
        .intensity = 1,
        .bounce = 1,
        .sky_from_scene = true,
        .normal_offset = spacing * 0.25f,
    };
    r.max = min + (Vector3) { r.nx, r.ny, r.nz } * spacing;
    r.vx = r.nx * 4;
    r.vy = r.ny * 4;
    r.vz = r.nz * 4;
    r.voxels = mem::new_array(uint, (sz)r.vx * r.vy * r.vz * 2);
    r.voxels_changed = true;
    int at = 0;
    for (int l = 0; l < r.levels; l++) {
        int d      = r.base_directions << l;
        int[<3>] n = r.level_dims(l);
        r.offsets[l] = at;
        r.counts[l] = n.x * n.y * n.z * d * d;
        at += r.counts[l];
    }
    return r;
}

// Allocates a sparse Split Radiance Cascades volume over `min..max`. Probes are allocated on the
// GPU near visible G-Buffer surfaces in a hash table (`hash_cap`, rounded up to a power of two)
// with bounded per-level probe pools starting at `max_probes` for L0.
fn RadianceCascades* radiance_cascades_sparse(
    Vector3 min,
    Vector3 max,
    float spacing = 0.25,
    int levels = 4,
    int max_probes = 16384,
    int hash_cap = 65536
) {
    RadianceCascades* r = radiance_cascades(min, max, spacing, levels);
    r.sparse = true;
    // Round `hash_cap` up to a power of two in `[1024, 1 << 20]`.
    int hc = 1024;
    while (hc < hash_cap && hc < 1 << 20) hc <<= 1;
    r.hash_cap = hc;
    int pbase  = 0;
    int ray_at = 0;
    for (int l = 0; l < r.levels; l++) {
        // Surface probe density drops roughly 4x per level as spacing doubles.
        int cap;
        switch (l) {
            case 0: cap = max_probes;
            case 1: cap = math::max(max_probes / 4, 512);
            case 2: cap = math::max(max_probes / 8, 512);
            case 3: cap = math::max(max_probes / 32, 256);
            default: cap = math::max(max_probes >> (l * 2), 128);
        }
        r.level_cap[l] = cap;
        r.probe_base[l] = pbase;
        pbase += cap;
        int d = r.base_directions << l;
        r.offsets[l] = ray_at;
        r.counts[l] = cap * d * d;
        ray_at += r.counts[l];
    }
    // Split RC layout (`shaders/radiance_rays_cs.glsl`): `P_n = max(P0 >> 2n, 64)`,
    // `|Omega_n| = 2 * (Theta0 * 2^n)^2`. Drop levels whose start distance reaches the box diagonal.
    float diag     = (r.max - r.min).length();
    float t0       = r.spacing * 1.6f;
    int eff_levels = 1;
    for (int l = 1; l < r.levels; l++) {
        float t_start = t0 * math::pow(4.0f, (float)(l - 1));
        if (l >= 2 && t_start >= diag * 0.98f) break;
        eff_levels = l + 1;
    }
    r.split_levels = eff_levels;
    int sp_base = 0;
    int sbin_at = 0;
    for (int l = 0; l < r.levels; l++) {
        int scap = math::max(max_probes >> (l * 2), 64);
        r.split_cap[l] = scap;
        r.split_pbase[l] = sp_base;
        sp_base += scap;
        int th   = r.base_directions << l;
        int dirs = 2 * th * th;
        r.split_offsets[l] = sbin_at;
        r.split_counts[l] = scap * dirs;
        sbin_at += r.split_counts[l];
    }
    // Secondary multi-bounce cache: 4x spacing, 1/16 primary L0 probe cap, packed immediately
    // after the primary pools in `meta_buf` and the bin buffers.
    r.cache_levels = eff_levels;
    r.cache_meta_base = sp_base;
    r.cache_bin_base = sbin_at;
    int cmeta = sp_base;
    int cbin  = sbin_at;
    for (int l = 0; l < r.levels; l++) {
        int ccap = math::max(r.split_cap[l] / 16, 16);
        int th   = r.base_directions << l;
        r.cache_cap[l] = ccap;
        r.cache_pbase[l] = cmeta;
        r.cache_offsets[l] = cbin;
        r.cache_counts[l] = ccap * 2 * th * th;
        cmeta += ccap;
        cbin += r.cache_counts[l];
    }
    // `irradiance_buf` layout: hash table, L0 6-face, L1 6-face fallback, then cache 6-face.
    r.cache_irr_base = r.hash_cap + r.split_cap[0] * 6 + (eff_levels > 1 ? r.split_cap[1] * 6 : 0);
    r.hits_cap = math::max(r.cache_cap[0] * 4, 1024);
    return r;
}

// ---- GPU resource upload

// Allocates GPU storage buffers and compute passes for sparse Split Radiance Cascades if not yet
// initialized, and marks the hash table dirty for initial clearing.
fn void RadianceCascades.upload_split(&self) {
    if (self.rays_alloc_pass) return;
    int eff        = math::max(self.split_levels, 1);
    int total_bins = self.cache_offsets[eff - 1] + self.cache_counts[eff - 1];
    int total_meta = self.cache_pbase[eff - 1] + self.cache_cap[eff - 1];
    if (!self.irradiance_buf) {
        self.irradiance_buf = storage_buffer(
            (sz)(self.cache_irr_base + self.cache_cap[0] * 6) * 16,
            label: "drie-rc-split-irr"
        );
    }
    if (!self.light_buf) {
        self.light_buf = storage_buffer(
            RC_MAX_LIGHTS * shaders::RcLight::size,
            cpu_write: true,
            label: "drie-rc-lights"
        );
    }
    if (!self.voxel_buf) self.voxel_buf = storage_buffer(16, label: "drie-rc-voxels");
    if (!self.brick_buf) self.brick_buf = storage_buffer(16, label: "drie-rc-bricks");
    if (!self.cascade_buf || self.cascade_buf.size < (usz)total_bins * 8) {
        if (self.cascade_buf) self.cascade_buf.dispose();
        self.cascade_buf = storage_buffer((sz)total_bins * 8, label: "drie-rc-rays-merged");
    }
    if (!self.meta_buf || self.meta_buf.size < (usz)total_meta * 32) {
        if (self.meta_buf) self.meta_buf.dispose();
        self.meta_buf = storage_buffer((sz)total_meta * 32, label: "drie-rc-rays-meta");
    }
    int want_free = RC_FREE_BASE + total_meta;
    if (!self.free_buf || self.free_buf.size < (usz)want_free * 4) {
        if (self.free_buf) self.free_buf.dispose();
        self.free_buf_len = want_free;
        self.free_buf = storage_buffer((sz)want_free * 4, label: "drie-rc-rays-free");
    }
    self.hist_buf = storage_buffer((sz)total_bins * 8, label: "drie-rc-rays-hist");
    self.hits_buf = storage_buffer((sz)self.hits_cap * 16, label: "drie-rc-rays-hits");
    self.args_buf = storage_buffer(24 * 16, label: "drie-rc-rays-args", indirect: true);
    if (!rc_shared_acc || rc_shared_acc.size < (usz)total_bins * 16) {
        if (rc_shared_acc) rc_shared_acc.dispose();
        rc_shared_acc = storage_buffer((sz)total_bins * 16, label: "drie-rc-shared-acc");
    }
    if (!rc_shared_rays) {
        rc_shared_rays = storage_buffer((sz)RC_MAX_TRACE_RAYS * 4, label: "drie-rc-shared-rays");
    }
    SgBackend backend = sg::query_backend();
    self.rays_clear_pass = compute_pass(rc_rays::rc_rays_clear_shader_desc(backend), "drie-rc-clear");
    self.rays_rebuild_pass = compute_pass(rc_rays::rc_rays_rebuild_shader_desc(backend), "drie-rc-evict");
    self.rays_alloc_pass = compute_pass(rc_rays::rc_rays_alloc_shader_desc(backend), "drie-rc-alloc");
    self.rays_alloc_ms_pass = compute_pass(rc_rays::rc_rays_alloc_ms_shader_desc(backend), "drie-rc-alloc");
    self.rays_cache_alloc_pass = compute_pass(rc_rays::rc_rays_cache_alloc_shader_desc(backend), "drie-rc-cache-alloc");
    self.rays_propagate_pass = compute_pass(rc_rays::rc_rays_propagate_shader_desc(backend), "drie-rc-propagate");
    self.rays_offsets_pass = compute_pass(rc_rays::rc_rays_offsets_shader_desc(backend), "drie-rc-propagate");
    self.rays_args_pass = compute_pass(rc_rays::rc_rays_args_shader_desc(backend), "drie-rc-propagate");
    self.rays_trace_pass = compute_pass(rc_rays::rc_rays_trace_shader_desc(backend), "drie-rc-split-trace");
    self.rays_merge_pass = compute_pass(rc_rays::rc_rays_merge_shader_desc(backend), "drie-rc-split-gather");
    self.rays_gather_pass = compute_pass(rc_rays::rc_rays_gather_shader_desc(backend), "drie-rc-split-gather");
    self.sparse_dirty = true;
}

// Lazily creates GPU buffers and compute passes for dense or sparse mode. Deferred until the
// first `update()` or render so construction and `voxelize()` can run off the render thread.
fn void RadianceCascades.upload(&self) {
    if (self.sparse) {
        self.upload_split();
        return;
    }
    if (self.trace) return;
    int at = self.offsets[self.levels - 1] + self.counts[self.levels - 1];
    self.cascade_buf = storage_buffer((sz)at * 8, label: "drie-rc-cascades");
    self.irradiance_buf = storage_buffer((sz)self.nx * self.ny * self.nz * 6 * 16, label: "drie-rc-irradiance");
    self.light_buf = storage_buffer(RC_MAX_LIGHTS * shaders::RcLight::size, cpu_write: true, label: "drie-rc-lights");
    self.voxel_buf = storage_buffer(16, label: "drie-rc-voxels");
    self.brick_buf = storage_buffer(16, label: "drie-rc-bricks");
    SgBackend backend = sg::query_backend();
    self.trace = compute_pass(shaders::rc_trace_shader_desc(backend), "drie-rc-trace");
    self.gather = compute_pass(shaders::rc_gather_shader_desc(backend), "drie-rc-gather");
}

// Returns the probe grid dimensions `(nx, ny, nz)` at cascade `level`.
fn int[<3>] RadianceCascades.level_dims(&self, int level) {
    int add = (1 << level) - 1;
    return {
        math::max((self.nx + add) >> level, 1),
        math::max((self.ny + add) >> level, 1),
        math::max((self.nz + add) >> level, 1)
    };
}

// ---- Voxelization

// Clears all CPU voxels to empty and marks the GPU voxel buffer dirty.
fn void RadianceCascades.clear_voxels(&self) {
    self.voxels[..] = 0;
    self.voxels_changed = true;
}

fn uint rc_pack_glow(Vector3 c) @local {
    float m = math::max(c.x, math::max(c.y, c.z));
    if (m < 1e-5f) return 0;
    int e = (int)math::ceil(math::log2(m));
    e = math::clamp(e, -126, 126);
    float scale = 255.0f / math::pow(2.0f, (float)e);
    uint r      = (uint)math::clamp(c.x * scale + 0.5f, 0.0f, 255.0f);
    uint g      = (uint)math::clamp(c.y * scale + 0.5f, 0.0f, 255.0f);
    uint b      = (uint)math::clamp(c.z * scale + 0.5f, 0.0f, 255.0f);
    return r | (g << 8) | (b << 16) | ((uint)(e + 128) << 24);
}

// Marks the voxel at world position `p` solid with linear `albedo` (`0..1`) and linear emissive
// `glow`. When two surfaces occupy the same voxel, the higher glow exponent wins. Respects
// `[clip_min, clip_max]` while `clip_region` is active during `voxelize_region()`.
fn void RadianceCascades.set_voxel(&self, Vector3 p, Vector3 albedo, Vector3 glow) {
    if (self.clip_region) {
        for (int k = 0; k < 3; k++) if (p[k] < self.clip_min[k] || p[k] > self.clip_max[k]) return;
    }
    Vector3 g = (p - self.min) / self.voxel_size;
    int x     = (int)math::floor(g.x);
    int y     = (int)math::floor(g.y);
    int z     = (int)math::floor(g.z);
    if (x < 0 || y < 0 || z < 0 || x >= self.vx || y >= self.vy || z >= self.vz) return;
    sz at       = (((sz)z * self.vy + y) * self.vx + x) * 2;
    uint packed = rc_pack_glow(glow);
    if (self.voxels[at] && (self.voxels[at + 1] >> 24) > (packed >> 24)) return;
    uint r  = (uint)math::clamp(albedo.x * 255 + 0.5f, 0.0f, 255.0f);
    uint gg = (uint)math::clamp(albedo.y * 255 + 0.5f, 0.0f, 255.0f);
    uint b  = (uint)math::clamp(albedo.z * 255 + 0.5f, 0.0f, 255.0f);
    self.voxels[at] = r | (gg << 8) | (b << 16) | 0x1000000;
    self.voxels[at + 1] = packed;
    self.voxels_changed = true;
}

// Rasterizes a world-space triangle `(a, b, c)` into the voxel grid with linear `albedo` and
// `glow`, sampling points spaced at most half a voxel apart.
fn void RadianceCascades.add_triangle(&self, Vector3 a, Vector3 b, Vector3 c, Vector3 albedo, Vector3 glow) {
    Vector3 bmin = self.clip_region ? self.clip_min : self.min;
    Vector3 bmax = self.clip_region ? self.clip_max : self.max;
    for (int k = 0; k < 3; k++) {
        float lo = math::min(a[k], math::min(b[k], c[k]));
        float hi = math::max(a[k], math::max(b[k], c[k]));
        if (hi < bmin[k] || lo > bmax[k]) return;
    }
    // Sample barycentric grid at half-voxel step so thin triangles do not leave gaps.
    float h    = self.voxel_size * 0.5f;
    Vector3 ab = b - a;
    Vector3 ac = c - a;
    int n1     = math::min((int)math::ceil(ab.length() / h) + 1, 4096);
    int n2     = math::min((int)math::ceil(ac.length() / h) + 1, 4096);
    for (int i = 0; i <= n1; i++) {
        float u = (float)i / n1;
        for (int j = 0; j <= n2; j++) {
            float v = (float)j / n2;
            if (u + v > 1.0001f) break;
            self.set_voxel(a + ab * u + ac * v, albedo, glow);
        }
    }
}

// Returns true if `g`'s bounding sphere transformed by `m` overlaps `[min, max]`.
fn bool RadianceCascades.touches(&self, BufferGeometry* g, Mat4 m) {
    Vector3 c = m.transform_point(g.bounds_center);
    float scale = math::max(
        ((Vector3) { m[0][0], m[0][1], m[0][2] }).length(),
        math::max(
            ((Vector3) { m[1][0], m[1][1], m[1][2] }).length(),
            ((Vector3) { m[2][0], m[2][1], m[2][2] }).length()
        )
    );
    return self.reaches(c, g.bounds_radius * scale + 1e-3f);
}

// Voxelizes a single opaque mesh using its current `matrix_world` (and any instance transforms).
// Skips hidden, transparent, alpha-tested, glass, shadow-only, and line meshes; `BASIC`
// materials emit their albedo as glow. `albedo_scale` multiplies the linear surface color.
fn void RadianceCascades.add_mesh(&self, Mesh* mesh, Vector3 albedo_scale = { 1, 1, 1 }) {
    Material* mat     = mesh.material;
    BufferGeometry* g = mesh.geometry;
    if (!mat || !g || mesh.shadow_only || !mat.visible || mat.transparent || mat.opacity < 1 || mat.alpha_test > 0
        || mat.glass) return;
    // Line geometry has no surface area for indirect bounce or emission.
    if (mesh.is_line()) return;
    if (g.bounds_radius <= 0) g.compute_bounding_sphere();
    if (mesh.instance_count <= 0 && !self.touches(g, mesh.matrix_world)) return;
    Vector3 albedo = color_to_linear(mat.color) * albedo_scale;
    if (mat.map) albedo = albedo * mat.map.average_color();
    Vector3 glow    = color_to_linear(mat.emissive) * mat.emissive_intensity;
    float brightest = math::max(glow.x, math::max(glow.y, glow.z));
    if (self.max_glow > 0 && brightest > self.max_glow) glow = glow * (self.max_glow / brightest);
    if (mat.type == BASIC) {
        glow = albedo;
        albedo = { };
    }
    sz count      = g.indices.len() ? g.indices.len() : (sz)g.vertex_count();
    int instances = math::max(mesh.instance_count, 1);
    for (int n = 0; n < instances; n++) {
        Mat4 m = mesh.matrix_world;
        if (mesh.instance_count > 0) {
            // 12 floats per instance stored as row-major 3x4 affine matrices.
            float[] row = mesh.instance_data.entries[(sz)n * 12:12];
            Mat4 im     = mat4_identity();
            for (int r = 0; r < 3; r++) {
                for (int col = 0; col < 4; col++) im[col][r] = row[r * 4 + col];
            }
            m = m.mul(im);
            if (!self.touches(g, m)) continue;
        }
        for (sz i = 0; i + 2 < count; i += 3) {
            Vector3[3] p;
            for (int k = 0; k < 3; k++) {
                sz vi     = g.indices.len() ? (sz)g.indices[i + k] : i + k;
                float[] v = g.vertices.entries[vi * FLOATS_PER_VERTEX:3];
                p[k] = m.transform_point({ v[0], v[1], v[2] });
            }
            self.add_triangle(p[0], p[1], p[2], albedo, glow);
        }
    }
}

fn void rc_add_tree(RadianceCascades* self, Object3D* o) @local {
    if (!o.visible) return;
    if (o.kind == MESH) self.add_mesh((Mesh*)o);
    foreach (c : o.children) rc_add_tree(self, c);
}

// Clears the voxel grid, updates world matrices under `root`, and voxelizes all visible opaque
// meshes beneath it. Call initially and whenever static solid geometry inside the box changes.
fn void RadianceCascades.voxelize(&self, Object3D* root) {
    root.update_matrix_world();
    self.clear_voxels();
    rc_add_tree(self, root);
    self.voxelized_root = root;
}

// Voxelizes meshes into `[region_min, region_max]` (world space) without clearing voxels outside
// that box. Used by `recenter()` to populate newly exposed slabs; caller must keep `root`'s
// world matrices up to date.
fn void RadianceCascades.voxelize_region(&self, Object3D* root, Vector3 region_min, Vector3 region_max) {
    self.clip_region = true;
    self.clip_min = region_min;
    self.clip_max = region_max;
    rc_add_tree(self, root);
    self.clip_region = false;
    self.voxels_changed = true;
}

// ---- Light collection & voxel upload

struct RcLightList @local {
    shaders::RcLight[RC_MAX_LIGHTS] lights;
    int count;
    Vector3 sky;
    Vector3 ground;
}

fn void rc_collect_lights(RadianceCascades* self, Object3D* o, RcLightList* out) @local {
    if (!o.visible) return;
    if (o.kind == LIGHT) {
        Light* l  = (Light*)o;
        Vector3 c = color_to_linear(l.color) * l.intensity;
        switch (l.light_type) {
            case AMBIENT:
                out.sky += c;
                out.ground += c;
            case HEMISPHERE:
                out.sky += c;
                out.ground += color_to_linear(((HemisphereLight*)l).ground_color) * l.intensity;
            case DIRECTIONAL:
                // (towards the light; w = -3 marks it)
                DirectionalLight* dl = (DirectionalLight*)l;
                Vector3 from = l.matrix_world.get_position();
                Vector3 to = dl.target ? dl.target.matrix_world.get_position() : (Vector3) { };
                Vector3 d = from - to;
                if (out.count < RC_MAX_LIGHTS && l.intensity > 0 && d.length() > 1e-6f) {
                    out.lights[out.count++] = {
                        .pos = { ...d.normalize(), 0 },
                        .color = { ...c, 0 },
                        .dir = { 0, 0, 0, -3 }
                    };
                }
            case POINT:
                PointLight* pl = (PointLight*)l;
                Vector3 p = l.gi_fixed ? l.gi_position : l.matrix_world.get_position();
                if (out.count < RC_MAX_LIGHTS && l.intensity > 0 && self.reaches(p, pl.distance)) {
                    out.lights[out.count++] = {
                        .pos = { ...p, pl.distance },
                        .color = { ...c, pl.decay },
                        .dir = { 0, 0, 0, -2 }
                    };
                }
            case SPOT:
                SpotLight* sl = (SpotLight*)l;
                Vector3 p = l.gi_fixed ? l.gi_position : l.matrix_world.get_position();
                Vector3 aim = l.gi_fixed ? l.gi_direction.normalize() : spot_direction(sl);
                if (out.count < RC_MAX_LIGHTS && l.intensity > 0 && self.reaches(p, sl.distance)) {
                    float cos_outer = math::cos(clamp(sl.angle, 0, PI / 2));
                    float cos_inner = math::cos(clamp(sl.angle, 0, PI / 2) * (1 - clamp(sl.penumbra, 0, 1)));
                    out.lights[out.count++] = {
                        .pos = { ...p, sl.distance },
                        .color = { ...c, sl.decay },
                        .dir = { ...aim, cos_outer },
                        .params = { math::max(cos_inner, cos_outer + 1e-4f), 0, 0, 0 },
                    };
                }
            default:
        }
    }
    foreach (c : o.children) rc_collect_lights(self, c, out);
}

// Returns true if a sphere centered at `p` with radius `distance` (`<= 0` = infinite) overlaps
// the volume's `[min, max]` box.
fn bool RadianceCascades.reaches(&self, Vector3 p, float distance) {
    if (distance <= 0) return true;
    float d2 = 0;
    for (int k = 0; k < 3; k++) {
        float d = math::max(math::max(self.min[k] - p[k], p[k] - self.max[k]), 0.0f);
        d2 += d * d;
    }
    return d2 < distance * distance;
}

// Uploads the CPU voxel array and rebuilds the 64-bit 4x4x4 brick occupancy masks on the GPU
// when `voxels_changed` is set.
fn void rc_upload_voxels(RadianceCascades* self) @local {
    if (!self.voxels_changed) return;
    self.voxel_buf.write(self.voxels.ptr, (sz)self.voxels.len * uint::size);
    uint[] bricks = mem::new_array(uint, (sz)self.nx * self.ny * self.nz * 2);
    defer free(bricks);
    for (int z = 0; z < self.vz; z++) {
        int bz     = z >> 2;
        int lz     = z & 3;
        int word   = lz >> 1;
        int zshift = (lz & 1) << 4;
        for (int y = 0; y < self.vy; y++) {
            int by      = y >> 2;
            int yzshift = zshift | ((y & 3) << 2);
            sz vrow     = ((sz)z * self.vy + y) * self.vx;
            sz brow     = (((sz)bz * self.ny + by) * self.nx) * 2 + word;
            for (int x = 0; x < self.vx; x++) {
                if (self.voxels[(vrow + x) * 2]) {
                    bricks[brow + (sz)(x >> 2) * 2] |= 1u << (yzshift | (x & 3));
                }
            }
        }
    }
    self.brick_buf.write(bricks.ptr, (sz)bricks.len * uint::size);
    self.voxels_changed = false;
}

// ---- Dense cascade update

// Dispatches dense cascade tracing and 6-face irradiance gathering (`shaders/radiance_cs.glsl`)
// for the current `scene` lights. Call once per frame before `Renderer.render()`, outside any
// active render pass.
fn void RadianceCascades.update(&self, Scene* scene) {
    self.upload();
    rc_upload_voxels(self);
    RcLightList* list = mem::new(RcLightList);
    defer free(list);
    rc_collect_lights(self, scene, list);
    if (self.sky_from_scene) {
        // Convert scene irradiance (hemisphere/ambient lights) to isotropic sky radiance (/ PI).
        self.sky_color = list.sky / (float)PI;
        self.ground_color = list.ground / (float)PI;
    }
    self.light_buf.write(&list.lights, (sz)math::max(list.count, 1) * shaders::RcLight::size);

    int every = math::clamp(self.update_every, 1, 0xffff);
    int sr    = math::clamp(self.shadow_rays, 1, 0xffff);
    int turn  = self.frame / every;
    RcParams p = {
        .rc_min = { ...self.min, self.spacing },
        .rc_dims = {
            self.nx,
            self.ny,
            self.nz,
            0
        }, // Ramp blend from 1.0 on early turns so probes converge immediately instead of from black.
        .rc_tune = {
            self.interval,
            math::max(self.blend, 1.0f / (float)(turn + 1)),
            (float)(self.frame % 4096),
            self.voxel_size
        },
        .rc_info = { self.base_directions, self.levels, list.count, 0 },
        .rc_sky = { ...self.sky_color, self.bounce },
        .rc_ground = { ...self.ground_color, 0 },
        .rc_vox = { self.vx, self.vy, self.vz, every | (sr << 16) },
    };
    for (int l = 0; l < self.levels; l++) {
        if (l < 4) {
            p.rc_off0[l] = self.offsets[l];
        } else {
            p.rc_off1[l - 4] = self.offsets[l];
        }
    }
    ComputePass* t = self.trace;
    t.set_buffer(shaders::VIEW_RC_VOXELS, self.voxel_buf);
    t.set_buffer(shaders::VIEW_RC_CASCADES, self.cascade_buf);
    t.set_buffer(shaders::VIEW_RC_IRRADIANCE, self.irradiance_buf);
    t.set_buffer(shaders::VIEW_RC_LIGHTS, self.light_buf);
    t.set_buffer(shaders::VIEW_RC_BRICKS, self.brick_buf);
    ComputePass* g = self.gather;
    g.set_buffer(shaders::VIEW_RC_VOXELS, self.voxel_buf);
    g.set_buffer(shaders::VIEW_RC_BRICKS, self.brick_buf);
    g.set_buffer(shaders::VIEW_RC_GATHER_CASCADES, self.cascade_buf);
    g.set_buffer(shaders::VIEW_RC_GATHER_IRRADIANCE, self.irradiance_buf);
    // Dispatch each level in its own compute pass so level `l` reads freshly written level `l + 1`.
    for (int l = self.levels - 1; l >= 0; l--) {
        p.rc_dims[3] = l;
        p.rc_info[3] = self.counts[l];
        t.set_uniforms(shaders::UB_RC_PARAMS, &p, RcParams::size);
        if (self.skip_kernels & RC_KERNEL_TRACE) continue;
        begin_compute("drie-rc-trace");
        t.dispatch(((self.counts[l] + every - 1) / every + 63) / 64);
        end_compute();
    }
    g.set_uniforms(shaders::UB_RC_PARAMS, &p, RcParams::size);
    if (!(self.skip_kernels & RC_KERNEL_GATHER)) {
        begin_compute("drie-rc-gather");
        g.dispatch((self.nx * self.ny * self.nz + 63) / 64);
        end_compute();
    }
    self.frame++;
}

// ---- Sparse split cascade update

// Runs the full sparse Split Radiance Cascades pipeline (`shaders/radiance_rays_cs.glsl`:
// rebuild, alloc, propagate, trace, merge, gather, plus optional secondary cache passes) from
// `depth_tex` and `normal_tex`. Invoked once per engine frame by `update_sparse()` or
// `flush_pending_sparse()`.
fn void RadianceCascades.run_split_update(
    &self,
    Scene* scene,
    SgView depth_tex,
    SgView normal_tex,
    SgSampler gb_sampler,
    Mat4 inv_view_proj,
    Vector3 cam_pos,
    int width,
    int height
) {
    self.upload_split();
    rc_upload_voxels(self);
    RcLightList* list = mem::new(RcLightList);
    defer free(list);
    rc_collect_lights(self, scene, list);
    if (self.sky_from_scene) {
        self.sky_color = list.sky / (float)PI;
        self.ground_color = list.ground / (float)PI;
    }
    self.light_buf.write(&list.lights, (sz)math::max(list.count, 1) * shaders::RcLight::size);

    int eff        = math::max(self.split_levels, 1);
    int total_bins = self.cache_offsets[eff - 1] + self.cache_counts[eff - 1];
    int total_meta = self.cache_pbase[eff - 1] + self.cache_cap[eff - 1];
    if (rc_shared_acc.size < (usz)total_bins * 16) rc_shared_acc.ensure((sz)total_bins * 16);
    bool cache_on = self.multibounce && self.cache_levels > 0;

    int stride = math::clamp(self.update_every, 1, 0xffff);
    int sr     = math::clamp(self.shadow_rays, 1, 0xffff);
    bool zo    = depth_zero_to_one();
    // Default split-mode interval is `1.6 * spacing` unless caller set a custom `interval`.
    float t0   = (math::abs(self.interval - self.spacing) < 1e-5f) ? (self.spacing * 1.6f) : self.interval;
    rc_rays::RcRaysParams p = {
        .rc_min = { ...self.min, self.spacing },
        .rc_dims = { self.nx, self.ny, self.nz, 0 },
        .rc_tune = {
            t0,
            self.blend,
            (float)(self.frame % 4096),
            self.voxel_size,
        },
        .rc_info = { self.base_directions, eff, list.count, self.hash_cap },
        .rc_sky = { ...self.sky_color, self.bounce },
        .rc_ground = { ...self.ground_color, self.normal_offset },
        .rc_vox = { self.vx, self.vy, self.vz, stride | (sr << 16) },
        .rc_inv_vp = inv_view_proj,
        .rc_cam_pos = { ...cam_pos, (zo ? 1 : -1) * math::max(self.blend_rays, 1.0f) },
        .rc_screen = {
            1.0f / (float)width,
            1.0f / (float)height,
            zo ? 1 : 0,
            (float)math::max(self.rays_per_probe, 1)
        },
    };
    for (int l = 0; l < eff; l++) {
        if (l < 4) {
            p.rc_lcap0[l] = self.split_cap[l];
            p.rc_loff0[l] = self.split_offsets[l];
            p.rc_pbase0[l] = self.split_pbase[l];
        } else {
            p.rc_lcap1[l - 4] = self.split_cap[l];
            p.rc_loff1[l - 4] = self.split_offsets[l];
            p.rc_pbase1[l - 4] = self.split_pbase[l];
        }
    }

    // 0. Clear on `reset()` or first frame.
    if (self.sparse_dirty && !(self.skip_kernels & RC_KERNEL_CLEAR)) {
        ComputePass* cp = self.rays_clear_pass;
        cp.set_buffer(rc_rays::VIEW_RC_HT_CLEAR, self.irradiance_buf);
        cp.set_buffer(rc_rays::VIEW_RC_META_CLEAR, self.meta_buf);
        cp.set_buffer(rc_rays::VIEW_RC_FREE_CLEAR, self.free_buf);
        cp.set_buffer(rc_rays::VIEW_RC_HIST_CLEAR, self.hist_buf);
        cp.set_buffer(rc_rays::VIEW_RC_ACC_CLEAR, rc_shared_acc);
        cp.set_buffer(rc_rays::VIEW_RC_HITS_CLEAR, self.hits_buf);
        cp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
        int clear_n = math::max(self.hash_cap, math::max(RC_FREE_BASE + total_meta, total_bins));
        clear_n = math::max(clear_n, self.hits_cap);
        begin_compute("drie-rc-clear");
        cp.dispatch((clear_n + 63) / 64);
        end_compute();
        self.sparse_dirty = false;
    }

    // 1. Rebuild hash table from `meta_buf` every frame (2 dispatches in 1 pass).
    if (!(self.skip_kernels & RC_KERNEL_EVICT)) {
        ComputePass* rp = self.rays_rebuild_pass;
        rp.set_buffer(rc_rays::VIEW_RC_HT_REB, self.irradiance_buf);
        rp.set_buffer(rc_rays::VIEW_RC_META_REB, self.meta_buf);
        rp.set_buffer(rc_rays::VIEW_RC_FREE_REB, self.free_buf);
        rp.set_buffer(rc_rays::VIEW_RC_HIST_REB, self.hist_buf);
        begin_compute("drie-rc-evict");
        p.rc_dims[3] = 0;
        rp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
        rp.dispatch((self.hash_cap + 63) / 64);
        p.rc_dims[3] = 32; // phase 1
        rp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
        rp.dispatch((total_meta + 63) / 64);
        end_compute();
        p.rc_dims[3] = 0;
    }

    // 2. Allocate or touch nearest L0 probe per traced G-Buffer pixel.
    if (!(self.skip_kernels & RC_KERNEL_ALLOC)) {
        bool ms         = sg::query_image_sample_count(sg::query_view_image(depth_tex)) > 1;
        ComputePass* ap = ms ? self.rays_alloc_ms_pass : self.rays_alloc_pass;
        ap.bindings.views[rc_rays::VIEW_RC_DEPTH_TEX] = depth_tex;
        ap.bindings.views[rc_rays::VIEW_RC_NORMAL_TEX] = normal_tex;
        ap.bindings.samplers[rc_rays::SMP_RC_GB_SMP] = gb_sampler;
        ap.set_buffer(rc_rays::VIEW_RC_HT_ALLOC, self.irradiance_buf);
        ap.set_buffer(rc_rays::VIEW_RC_META_ALLOC, self.meta_buf);
        ap.set_buffer(rc_rays::VIEW_RC_FREE_ALLOC, self.free_buf);
        ap.set_buffer(rc_rays::VIEW_RC_BRICKS_ALLOC, self.brick_buf);
        ap.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
        int tw = (width + stride - 1) / stride;
        int th = (height + stride - 1) / stride;
        begin_compute("drie-rc-alloc");
        ap.dispatch((tw + 7) / 8, (th + 7) / 8);
        end_compute();
    }

    // 3 & 4. Propagate bottom-up, compute R2 segments top-down, and write indirect args.
    if (!(self.skip_kernels & RC_KERNEL_PROPAGATE)) {
        ComputePass* pp = self.rays_propagate_pass;
        pp.set_buffer(rc_rays::VIEW_RC_HT_PROP, self.irradiance_buf);
        pp.set_buffer(rc_rays::VIEW_RC_META_PROP, self.meta_buf);
        pp.set_buffer(rc_rays::VIEW_RC_FREE_PROP, self.free_buf);
        pp.set_buffer(rc_rays::VIEW_RC_BRICKS_PROP, self.brick_buf);

        ComputePass* op = self.rays_offsets_pass;
        op.set_buffer(rc_rays::VIEW_RC_META_OFF, self.meta_buf);
        op.set_buffer(rc_rays::VIEW_RC_FREE_OFF, self.free_buf);
        op.set_buffer(rc_rays::VIEW_RC_RAYS_OFF, rc_shared_rays);

        ComputePass* argp = self.rays_args_pass;
        argp.set_buffer(rc_rays::VIEW_RC_FREE_ARGS, self.free_buf);
        argp.set_buffer(rc_rays::VIEW_RC_ARGS_OUT, self.args_buf);
        argp.set_buffer(rc_rays::VIEW_RC_RAYS_ARGS, rc_shared_rays);

        begin_compute("drie-rc-propagate");
        for (int l = 0; l < eff; l++) {
            int max_phase = (l + 1 < eff) ? 2 : 1;
            for (int phase = 0; phase < max_phase; phase++) {
                p.rc_dims[3] = l | (phase << 5);
                pp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
                pp.dispatch((self.split_cap[l] + 63) / 64);
            }
        }
        for (int l = eff - 1; l >= 0; l--) {
            p.rc_dims[3] = l;
            op.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
            op.dispatch((self.split_cap[l] + 63) / 64);
        }
        p.rc_dims[3] = 0;
        argp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
        argp.dispatch(1);
        end_compute();
    }

    // 5. Trace R2 surface rays and deposit into cascade intervals (indirect dispatch).
    if (!(self.skip_kernels & RC_KERNEL_TRACE)) {
        ComputePass* tp = self.rays_trace_pass;
        tp.set_buffer(rc_rays::VIEW_RC_TRACE_VOXELS, self.voxel_buf);
        tp.set_buffer(rc_rays::VIEW_RC_TRACE_BRICKS, self.brick_buf);
        tp.set_buffer(rc_rays::VIEW_RC_TRACE_LIGHTS, self.light_buf);
        tp.set_buffer(rc_rays::VIEW_RC_HT_TRACE, self.irradiance_buf);
        tp.set_buffer(rc_rays::VIEW_RC_META_TRACE, self.meta_buf);
        tp.set_buffer(rc_rays::VIEW_RC_RAYS_TRACE, rc_shared_rays);
        tp.set_buffer(rc_rays::VIEW_RC_ACC_TRACE, rc_shared_acc);
        tp.set_buffer(rc_rays::VIEW_RC_TRACE_HITS, self.hits_buf);
        p.rc_dims[3] = cache_on ? 64 : 0; // Bit 64 enables reading secondary-cache bounce light.
        tp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
        begin_compute("drie-rc-split-trace");
        tp.dispatch_indirect(self.args_buf, 0);
        end_compute();
    }

    // 6. Top-down merge across levels and L0/L1 6-face irradiance gather (indirect dispatch).
    if (!(self.skip_kernels & RC_KERNEL_GATHER)) {
        ComputePass* mp = self.rays_merge_pass;
        mp.set_buffer(rc_rays::VIEW_RC_MERGE_CASCADES, self.cascade_buf);
        mp.set_buffer(rc_rays::VIEW_RC_HT_MERGE, self.irradiance_buf);
        mp.set_buffer(rc_rays::VIEW_RC_META_MERGE, self.meta_buf);
        mp.set_buffer(rc_rays::VIEW_RC_FREE_MERGE, self.free_buf);
        mp.set_buffer(rc_rays::VIEW_RC_HIST_MERGE, self.hist_buf);
        mp.set_buffer(rc_rays::VIEW_RC_ACC_MERGE, rc_shared_acc);

        ComputePass* gp = self.rays_gather_pass;
        gp.set_buffer(rc_rays::VIEW_RC_G_BRICKS, self.brick_buf);
        gp.set_buffer(rc_rays::VIEW_RC_G_CASCADES, self.cascade_buf);
        gp.set_buffer(rc_rays::VIEW_RC_HT_G, self.irradiance_buf);
        gp.set_buffer(rc_rays::VIEW_RC_META_G, self.meta_buf);
        gp.set_buffer(rc_rays::VIEW_RC_FREE_G, self.free_buf);
        gp.set_buffer(rc_rays::VIEW_RC_HIST_G, self.hist_buf);

        begin_compute("drie-rc-split-gather");
        for (int l = eff - 1; l >= 0; l--) {
            p.rc_dims[3] = l;
            mp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
            mp.dispatch_indirect(self.args_buf, (1 + l) * 16);
        }
        p.rc_dims[3] = 0;
        gp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
        gp.dispatch_indirect(self.args_buf, 9 * 16);
        // Gather L1 6-face irradiance as a fallback where no L0 probe is allocated nearby.
        if (eff > 1) {
            p.rc_dims[3] = 1;
            gp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
            gp.dispatch_indirect(self.args_buf, 20 * 16);
        }
        end_compute();
    }

    // 7. Seed secondary cache L0 probes from primary trace hits before cache propagation.
    if (cache_on && !(self.skip_kernels & RC_KERNEL_ALLOC)) {
        ComputePass* cap = self.rays_cache_alloc_pass;
        cap.set_buffer(rc_rays::VIEW_RC_HITS_ALLOC, self.hits_buf);
        cap.set_buffer(rc_rays::VIEW_RC_HT_CACHE_ALLOC, self.irradiance_buf);
        cap.set_buffer(rc_rays::VIEW_RC_META_CACHE_ALLOC, self.meta_buf);
        cap.set_buffer(rc_rays::VIEW_RC_FREE_CACHE_ALLOC, self.free_buf);
        cap.set_buffer(rc_rays::VIEW_RC_BRICKS_CACHE_ALLOC, self.brick_buf);
        cap.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
        begin_compute("drie-rc-cache-alloc");
        cap.dispatch((self.hits_cap + 63) / 64);
        end_compute();
    }

    // 8. Secondary cache propagation, R2 offsets, and indirect dispatch counts.
    if (cache_on && !(self.skip_kernels & RC_KERNEL_PROPAGATE)) {
        ComputePass* pp   = self.rays_propagate_pass;
        ComputePass* op   = self.rays_offsets_pass;
        ComputePass* argp = self.rays_args_pass;
        begin_compute("drie-rc-cache-propagate");
        for (int l = 0; l < eff; l++) {
            int max_phase = (l + 1 < eff) ? 2 : 1;
            for (int phase = 0; phase < max_phase; phase++) {
                p.rc_dims[3] = l | 16 | (phase << 5);
                pp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
                pp.dispatch((self.cache_cap[l] + 63) / 64);
            }
        }
        for (int l = eff - 1; l >= 0; l--) {
            p.rc_dims[3] = l | 16;
            op.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
            op.dispatch((self.cache_cap[l] + 63) / 64);
        }
        p.rc_dims[3] = 1; // Publish secondary cache ray count into `args_buf`.
        argp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
        argp.dispatch(1);
        end_compute();
    }

    // 9. Secondary cache trace, reading previous-frame cache irradiance to accumulate bounces.
    if (cache_on && !(self.skip_kernels & RC_KERNEL_TRACE)) {
        p.rc_dims[3] = 16;
        ComputePass* tp = self.rays_trace_pass;
        tp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
        begin_compute("drie-rc-cache-trace");
        tp.dispatch_indirect(self.args_buf, 10 * 16);
        end_compute();
    }

    // 10. Secondary cache merge and 6-face irradiance gather.
    if (cache_on && !(self.skip_kernels & RC_KERNEL_GATHER)) {
        ComputePass* mp = self.rays_merge_pass;
        ComputePass* gp = self.rays_gather_pass;
        begin_compute("drie-rc-cache-gather");
        for (int l = eff - 1; l >= 0; l--) {
            p.rc_dims[3] = l | 16;
            mp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
            mp.dispatch_indirect(self.args_buf, (11 + l) * 16);
        }
        p.rc_dims[3] = 16;
        gp.set_uniforms(rc_rays::UB_RC_RAYS_PARAMS, &p, rc_rays::RcRaysParams::size);
        gp.dispatch_indirect(self.args_buf, 19 * 16);
        end_compute();
    }

    self.frame++;
}

// Dispatches `run_split_update()` for the sparse volume.
fn void RadianceCascades.run_sparse_update(
    &self,
    Scene* scene,
    SgView depth_tex,
    SgView normal_tex,
    SgSampler gb_sampler,
    Mat4 inv_view_proj,
    Vector3 cam_pos,
    int width,
    int height
) {
    self.run_split_update(scene, depth_tex, normal_tex, gb_sampler, inv_view_proj, cam_pos, width, height);
}

// Called by `Renderer.render()` between the G-Buffer pass and deferred lighting for every
// non-reflecting render of a scene with a sparse volume. When `self.camera` is set, only that
// camera's render executes `run_sparse_update()` immediately and all other views reuse its
// irradiance. When `self.camera` is null, parameters are recorded and deferred to
// `flush_pending_sparse()` at `drie::commit()` so multi-camera frames still update once using
// the frame's final view.
fn void RadianceCascades.update_sparse(
    &self,
    Scene* scene,
    Camera* camera,
    SgView depth_tex,
    SgView normal_tex,
    SgSampler gb_sampler,
    Mat4 inv_view_proj,
    Vector3 cam_pos,
    int width,
    int height
) {
    if (!self.sparse || self.freeze_alloc) return;
    if (self.camera) {
        if (camera != self.camera) return;
        self.run_sparse_update(scene, depth_tex, normal_tex, gb_sampler, inv_view_proj, cam_pos, width, height);
        return;
    }
    self.pending_scene = scene;
    self.pending_depth_tex = depth_tex;
    self.pending_normal_tex = normal_tex;
    self.pending_gb_sampler = gb_sampler;
    self.pending_inv_view_proj = inv_view_proj;
    self.pending_cam_pos = cam_pos;
    self.pending_width = width;
    self.pending_height = height;
    if (!self.pending_sparse) {
        self.pending_sparse = true;
        queue_sparse_flush(self);
    }
}

// Executes the sparse update deferred by `update_sparse()` using the last render's G-Buffer
// views. Called once per frame by `drie::commit()` before the next frame overwrites the shared
// G-Buffer; skips silently if the recorded views are no longer valid.
fn void RadianceCascades.flush_pending_sparse(&self) {
    if (!self.pending_sparse) return;
    self.pending_sparse = false;
    // Skip if the target from the recorded render was destroyed or resized before `commit()`.
    if (sg::query_view_state(
        self.pending_depth_tex
    ) != SgResourceState.VALID || sg::query_view_state(self.pending_normal_tex) != SgResourceState.VALID) return;
    self.run_sparse_update(
        self.pending_scene,
        self.pending_depth_tex,
        self.pending_normal_tex,
        self.pending_gb_sampler,
        self.pending_inv_view_proj,
        self.pending_cam_pos,
        self.pending_width,
        self.pending_height
    );
}

// ---- Recentering & lifecycle

// Re-centers the volume horizontally around `center` (snapped to `spacing`) for sliding GI
// volumes. Scrolls existing voxels in whole-voxel steps and re-voxelizes only the newly exposed
// slabs while preserving sparse world-space probes; performs a full `voxelize()` and probe clear
// if `root` changed or the shift exceeds the box size.
fn void RadianceCascades.recenter(&self, Object3D* root, Vector3 center) {
    Vector3 size = self.max - self.min;
    Vector3 new_min = {
        math::floor((center.x - size.x * 0.5f) / self.spacing) * self.spacing,
        self.min.y,
        math::floor((center.z - size.z * 0.5f) / self.spacing) * self.spacing,
    };
    bool root_changed = root != self.voxelized_root;
    if (new_min == self.min && !root_changed) return;

    Vector3 old_min = self.min;
    self.min = new_min;
    self.max = new_min + size;

    int shift_vx = (int)math::round((new_min.x - old_min.x) / self.voxel_size);
    int shift_vz = (int)math::round((new_min.z - old_min.z) / self.voxel_size);
    if (root_changed || math::abs(shift_vx) >= self.vx || math::abs(shift_vz) >= self.vz) {
        self.voxelize(root);
        if (self.sparse) self.sparse_dirty = true;
        return;
    }
    // Keep live sparse probes across incremental scrolls: probes are keyed in world space, so
    // probes leaving the box age out naturally while newly visible slabs allocate on demand.
    if (shift_vx == 0 && shift_vz == 0) return;

    uint[] scrolled = mem::new_array(uint, (sz)self.voxels.len);
    for (int zn = 0; zn < self.vz; zn++) {
        int zo = zn + shift_vz;
        if (zo < 0 || zo >= self.vz) continue;
        for (int yn = 0; yn < self.vy; yn++) {
            for (int xn = 0; xn < self.vx; xn++) {
                int xo = xn + shift_vx;
                if (xo < 0 || xo >= self.vx) continue;
                sz dst = (((sz)zn * self.vy + yn) * self.vx + xn) * 2;
                sz src = (((sz)zo * self.vy + yn) * self.vx + xo) * 2;
                scrolled[dst] = self.voxels[src];
                scrolled[dst + 1] = self.voxels[src + 1];
            }
        }
    }
    free(self.voxels);
    self.voxels = scrolled;
    self.voxels_changed = true;

    // Re-voxelize each newly exposed world-space slab with a 2-voxel margin along seams.
    float margin = self.voxel_size * 2;
    if (shift_vx != 0) {
        float shift_x = new_min.x - old_min.x;
        float w       = math::abs(shift_x) + margin;
        Vector3 smin, smax;
        if (shift_x > 0) {
            smin = { self.max.x - w, self.min.y - margin, self.min.z - margin };
            smax = { self.max.x + margin, self.max.y + margin, self.max.z + margin };
        } else {
            smin = { self.min.x - margin, self.min.y - margin, self.min.z - margin };
            smax = { self.min.x + w, self.max.y + margin, self.max.z + margin };
        }
        self.voxelize_region(root, smin, smax);
    }
    if (shift_vz != 0) {
        float shift_z = new_min.z - old_min.z;
        float w       = math::abs(shift_z) + margin;
        Vector3 smin, smax;
        if (shift_z > 0) {
            smin = { self.min.x - margin, self.min.y - margin, self.max.z - w };
            smax = { self.max.x + margin, self.max.y + margin, self.max.z + margin };
        } else {
            smin = { self.min.x - margin, self.min.y - margin, self.min.z - margin };
            smax = { self.max.x + margin, self.max.y + margin, self.min.z + w };
        }
        self.voxelize_region(root, smin, smax);
    }
    self.voxelized_root = root;
}

// Resets temporal accumulation (and schedules a full sparse hash-table clear) so the next frame
// converges fresh after a camera cut or major lighting change.
fn void RadianceCascades.reset(&self) {
    self.frame = 0;
    if (self.sparse) self.sparse_dirty = true;
}

// Unregisters any pending sparse flush and frees the CPU voxel array and owned GPU buffers.
fn void RadianceCascades.dispose(&self) {
    if (self.pending_sparse) unqueue_sparse_flush(self);
    self.pending_sparse = false;
    free(self.voxels);
    if (self.voxel_buf) self.voxel_buf.dispose();
    if (self.brick_buf) self.brick_buf.dispose();
    if (self.cascade_buf) self.cascade_buf.dispose();
    if (self.irradiance_buf) self.irradiance_buf.dispose();
    if (self.light_buf) self.light_buf.dispose();
    if (self.meta_buf) self.meta_buf.dispose();
    if (self.free_buf) self.free_buf.dispose();
    if (self.hist_buf) self.hist_buf.dispose();
    if (self.hits_buf) self.hits_buf.dispose();
    if (self.args_buf) self.args_buf.dispose();
}