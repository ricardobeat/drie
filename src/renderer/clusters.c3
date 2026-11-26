// Clustered lighting froxel grid and per-render GPU light storage buffers.
// Partitions the camera frustum into CLUSTER_X x CLUSTER_Y screen tiles and CLUSTER_Z exponential
// depth slices matching cluster_of() in shaders/shader.glsl, assigning point and spot lights via
// cluster_lights_gpu (cluster_cs in shaders/kernels.glsl) or cluster_lights_cpu. LightData rotates
// transient LightBufferSets per render within a frame and packs RectAreaLight entries plus the
// LTC tables (generated/ltc.c3, read by ltc_tables in shaders/shader_core.glsl) after point lights
// in the point-light buffer. Connects to build_light_params(), draw_item(), and draw_deferred_lighting().
module drie;
import std::collections::list, std::math, std::io;
import sokol;

// ---- Cluster grid and bounds

const int CLUSTER_X      = 16;                               // Screen-space cluster tiles horizontally.
const int CLUSTER_Y      = 9;                                // Screen-space cluster tiles vertically.
const int CLUSTER_Z      = 24;                               // Exponential view-depth cluster slices.
const int CLUSTER_COUNT  = CLUSTER_X * CLUSTER_Y * CLUSTER_Z; // Total clusters in the froxel grid.
                                                             // Minimum depth (meters) where exponential slicing begins; closer fragments map to slice 0.
const float CLUSTER_NEAR = 0.1f;

// Camera frustum parameters for light clustering; matches cluster_of() in shaders/shader.glsl.
struct ClusterGrid {
    Mat4 view; // World space to camera view space (looking down -Z).
    Mat4 projection; // Camera view space to clip space.
    float near; // Depth (m) where slice 0 starts (>= CLUSTER_NEAR).
    float far; // Depth (m) where the last slice ends.
    float log_scale; // CLUSTER_Z / ln(far / near).
}

// World-space bounding sphere of a light's reach; `radius < 0` means unbounded, `0` skips it.
struct LightBounds {
    Vector3 center; // World-space sphere center.
    float radius; // Reach radius in meters (< 0 unbounded, 0 inactive).
}

// Destination buffers for ClusterAssignFn: each cluster lists its point lights, then spot lights.
struct ClusterOutput {
    List{GpuLightCluster}* clusters; // CPU output: per-cluster offset and light counts.
    List{GpuLightIndex}* indices; // CPU output: flat list of light buffer indices.
    StorageBuffer* gpu_clusters; // GPU compute output: per-cluster offset and counts.
    StorageBuffer* gpu_indices; // GPU compute output: flat light index buffer.
    StorageBuffer* gpu_ranges; // Transient CPU-to-GPU input: per-light cluster AABBs.
}

// Callback that assigns point and spot lights (indexed by slice position) into clusters.
// Returns true when results were written into `out.gpu_*`, or false when written to CPU lists.
alias ClusterAssignFn = fn bool(ClusterGrid* grid, LightBounds[] points, LightBounds[] spots, ClusterOutput* out);

fn ClusterGrid cluster_grid(Camera* camera) @private {
    float near = math::max(CLUSTER_NEAR, camera.near);
    float far  = math::max(camera.far, near * 2);
    return {
        .view = camera.matrix_world_inverse,
        .projection = camera.projection_matrix,
        .near = near,
        .far = far,
        .log_scale = (float)CLUSTER_Z / math::ln(far / near),
    };
}

fn int cluster_slice(ClusterGrid* grid, float depth) @private {
    if (depth <= grid.near) return 0;
    int s = (int)math::floor(math::ln(depth / grid.near) * grid.log_scale);
    return s < 0 ? 0 : (s >= CLUSTER_Z ? CLUSTER_Z - 1 : s);
}

fn int cluster_tile(float ndc, int n) @private {
    int t = (int)math::floor((ndc * 0.5f + 0.5f) * (float)n);
    return t < 0 ? 0 : (t >= n ? n - 1 : t);
}

// Projects light sphere `b` into inclusive froxel bounds `[lo, hi]`; returns false if outside view.
fn bool cluster_range(ClusterGrid* grid, LightBounds b, int[<3>]* lo, int[<3>]* hi) @private {
    // Inactive light (intensity <= 0).
    if (b.radius == 0) return false;
    if (b.radius < 0) {
        *lo = { 0, 0, 0 };
        *hi = { CLUSTER_X - 1, CLUSTER_Y - 1, CLUSTER_Z - 1 };
        return true;
    }
    Vector3 c  =  grid.view.transform_point(b.center);
    float r    =  b.radius;
    // Camera view space looks down -Z.
    float zmin = -c.z - r;
    float zmax = -c.z + r;
    if (zmax <= 0) return false;
    lo.z = cluster_slice(grid, zmin);
    hi.z = cluster_slice(grid, zmax);
    Mat4 p = grid.projection;
    if (zmin <= 1e-3f) {
        // Sphere straddles the near plane: cover all screen tiles.
        lo.x = 0;
        lo.y = 0;
        hi.x = CLUSTER_X - 1;
        hi.y = CLUSTER_Y - 1;
        return true;
    }
    // Project the 8 view-space AABB corners (all in front of the camera) to NDC.
    float x0 =  1e30f;
    float y0 =  1e30f;
    float x1 = -1e30f;
    float y1 = -1e30f;
    for (int k = 0; k < 8; k++) {
        float x  =  c.x + ((k & 1) ? r : -r);
        float y  =  c.y + ((k & 2) ? r : -r);
        float z  = -((k & 4) ? zmax : zmin);
        float cx =  p[0][0] * x + p[1][0] * y + p[2][0] * z + p[3][0];
        float cy =  p[0][1] * x + p[1][1] * y + p[2][1] * z + p[3][1];
        float cw =  p[0][3] * x + p[1][3] * y + p[2][3] * z + p[3][3];
        cw = math::max(math::abs(cw), 1e-6f);
        x0 = math::min(x0, cx / cw);
        x1 = math::max(x1, cx / cw);
        y0 = math::min(y0, cy / cw);
        y1 = math::max(y1, cy / cw);
    }
    if (x1 < -1 || x0 > 1 || y1 < -1 || y0 > 1) return false;
    lo.x = cluster_tile(x0, CLUSTER_X);
    hi.x = cluster_tile(x1, CLUSTER_X);
    lo.y = cluster_tile(y0, CLUSTER_Y);
    hi.y = cluster_tile(y1, CLUSTER_Y);
    return true;
}

// ---- Light assignment

// CPU implementation of ClusterAssignFn; populates `out.clusters` and `out.indices`.
fn bool cluster_lights_cpu(ClusterGrid* grid, LightBounds[] points, LightBounds[] spots, ClusterOutput* out) {
    fill_clusters(grid, points, spots, out.clusters, out.indices);
    return false;
}

ComputePass* cluster_kernel @local;
List{LightRange} cluster_ranges @local;

// Compute-shader implementation of ClusterAssignFn (falls back to cluster_lights_cpu when
// compute is unavailable). Computes per-light froxel ranges on the CPU via cluster_range(), then
// dispatches `cluster_cs` (shaders/kernels.glsl) to populate `out.gpu_clusters` and `out.gpu_indices`.
fn bool cluster_lights_gpu(ClusterGrid* grid, LightBounds[] points, LightBounds[] spots, ClusterOutput* out) {
    if (!compute_supported() || !out.gpu_clusters) return cluster_lights_cpu(grid, points, spots, out);
    if (!cluster_kernel) cluster_kernel = compute_pass(
        shaders::cluster_shader_desc(sg::query_backend()),
        "drie-clusters"
    );
    cluster_ranges.allocator = mem;
    cluster_ranges.clear();
    sz total;
    for (int kind = 0; kind < 2; kind++) {
        foreach (b : kind == 0 ? points : spots) {
            int[<3>] lo;
            int[<3>] hi;
            LightRange r;
            if (cluster_range(grid, b, &lo, &hi)) {
                r = { { lo.x, lo.y, lo.z, 1 }, { hi.x, hi.y, hi.z, 0 } };
                total += (sz)(hi.x - lo.x + 1) * (sz)(hi.y - lo.y + 1) * (sz)(hi.z - lo.z + 1);
            }
            cluster_ranges.push(r);
        }
    }
    // Keep at least one dummy entry so the storage buffer is never zero-sized.
    if (!cluster_ranges.len()) cluster_ranges.push({ });
    out.gpu_ranges.write(cluster_ranges.entries, cluster_ranges.len() * LightRange::size);
    out.gpu_clusters.ensure((sz)CLUSTER_COUNT * (sz)GpuLightCluster::size);
    out.gpu_indices.ensure(math::max(total, (sz)1) * (sz)GpuLightIndex::size);
    ClusterParams p = {
        .cl_lights = { (int)points.len, (int)spots.len, 0, 0 },
        .cl_grid = { CLUSTER_X, CLUSTER_Y, CLUSTER_Z, CLUSTER_COUNT },
    };
    ComputePass* k = cluster_kernel;
    k.set_buffer(shaders::VIEW_CLUSTER_RANGES, out.gpu_ranges);
    k.set_buffer(shaders::VIEW_CLUSTER_LIST, out.gpu_clusters);
    k.set_buffer(shaders::VIEW_CLUSTER_LIGHT_LIST, out.gpu_indices);
    k.set_uniforms(shaders::UB_CLUSTER_PARAMS, &p, ClusterParams::size);
    begin_compute("drie-clusters");
    k.dispatch(1);
    end_compute();
    return true;
}

// Two-pass CPU light assignment: counts lights per cluster, computes prefix offsets, then writes
// point-light indices followed by spot-light indices into `indices`.
fn void fill_clusters(
    ClusterGrid* grid,
    LightBounds[] points,
    LightBounds[] spots,
    List{GpuLightCluster}* clusters,
    List{GpuLightIndex}* indices
) @private {
    clusters.clear();
    for (int i = 0; i < CLUSTER_COUNT; i++) clusters.push({ });
    GpuLightCluster* cl = clusters.entries;
    // Pass 1: count touching lights per cluster (range[1] = points, range[2] = spots).
    for (int kind = 0; kind < 2; kind++) {
        LightBounds[] list = kind == 0 ? points : spots;
        foreach (b : list) {
            int[<3>] lo;
            int[<3>] hi;
            if (!cluster_range(grid, b, &lo, &hi)) continue;
            for (int z = lo.z; z <= hi.z; z++) {
                for (int y = lo.y; y <= hi.y; y++) {
                    for (int x = lo.x; x <= hi.x; x++) cl[x + CLUSTER_X * (y + CLUSTER_Y * z)].range[1 + kind]++;
                }
            }
        }
    }
    uint total;
    for (int i = 0; i < CLUSTER_COUNT; i++) {
        cl[i].range[0] = total;
        total += cl[i].range[1] + cl[i].range[2];
        // range[3] serves as a scratch write cursor during pass 2.
        cl[i].range[3] = 0;
    }
    indices.clear();
    // Reserve at least one entry so the GPU storage buffer is never zero-sized.
    for (uint i = 0; i < math::max(total, 1u); i++) indices.push({ });
    GpuLightIndex* out = indices.entries;
    // Pass 2: write point indices first, then spot indices after each cluster's points.
    for (int kind = 0; kind < 2; kind++) {
        LightBounds[] list = kind == 0 ? points : spots;
        foreach (uint li, b : list) {
            int[<3>] lo;
            int[<3>] hi;
            if (!cluster_range(grid, b, &lo, &hi)) continue;
            for (int z = lo.z; z <= hi.z; z++) {
                for (int y = lo.y; y <= hi.y; y++) {
                    for (int x = lo.x; x <= hi.x; x++) {
                        GpuLightCluster* c = &cl[x + CLUSTER_X * (y + CLUSTER_Y * z)];
                        uint at            = c.range[0] + (kind == 0 ? c.range[3] : c.range[1] + c.range[3]);
                        out[at].index = li;
                        c.range[3]++;
                    }
                }
            }
        }
        if (kind == 0) foreach (&c : clusters) c.range[3] = 0;
    }
}

// World-space bounding sphere for a point (or tube) light; radius -1 when distance == 0.
fn LightBounds point_light_bounds(PointLight* pl) @private {
    // Tube lights extend half their length from center along their X axis.
    return { pl.matrix_world.get_position(), pl.distance > 0 ? pl.distance + math::max(pl.length, 0.0f) * 0.5f : -1 };
}

// Minimal bounding sphere enclosing a spot light's cone (plus source radius); -1 when unbounded.
fn LightBounds spot_light_bounds(SpotLight* sl) @private {
    Vector3 pos = sl.matrix_world.get_position();
    if (sl.distance <= 0) return { pos, -1 };
    float a     = clamp(sl.angle, 0, PI / 2);
    float range = sl.distance;
    // Account for area-source radius extending illumination slightly past the cone.
    float extra = math::max(sl.radius, 0.0f);
    Vector3 dir = spot_direction(sl);
    if (a > PI / 4) return { pos + dir * (range * math::cos(a)), range * math::sin(a) + extra };
    float r = range / (2 * math::cos(a));
    return { pos + dir * r, r + extra };
}

// ---- GPU light buffers and LTC packing

// Growable transient storage buffer written at most once per frame (each render in a frame uses
// its own LightBufferSet).
struct LightBuffer @private {
    SgBuffer buffer;
    SgView view;
    usz capacity;
}

fn void LightBuffer.write(&self, void* data, sz bytes) {
    usz size = (usz)bytes;
    // Storage buffers cannot be zero-sized.
    char[16] zero;
    if (!size) {
        data = &zero;
        size = zero.len;
    }
    if (size > self.capacity) {
        if (self.view.id) sg::destroy_view(self.view);
        if (self.buffer.id) sg::destroy_buffer(self.buffer);
        usz cap = math::max(self.capacity * 2, (usz)256);
        while (cap < size) cap *= 2;
        self.capacity = cap;
        self.buffer = sg::make_buffer(
            &&{
                .usage = { .storage_buffer = true, .write_transient = true },
                .size = cap,
                .label = "drie-lights",
            }
        );
        self.view = sg::make_view(&&{ .storage_buffer = { .buffer = self.buffer } });
    }
    sg::write_buffer_transient(&&{ .src.data = { .ptr = data, .size = size }, .dst.buffer = self.buffer });
}

struct LightBufferSet @private {
    LightBuffer dirs;
    LightBuffer points;
    LightBuffer spots;
    LightBuffer shadows;
    LightBuffer clusters;
    LightBuffer indices;
    // Compute-populated cluster buffers (bound when `on_gpu` is true).
    StorageBuffer gpu_clusters;
    StorageBuffer gpu_indices;
    StorageBuffer gpu_ranges;
    bool on_gpu;
    // Renderer.check_compute: CPU reference results and readback slots from the prior frame.
    List{GpuLightCluster} check_clusters;
    List{GpuLightIndex} check_indices;
    int check_slot;
    int check_slot2;
    bool check_pending;
}

// Staged lights, shadows, and cluster sets for the current render().
struct LightData @private {
    List{GpuDirLight} dirs;
    List{GpuPointLight} points;
    List{GpuSpotLight} spots;
    // RectAreaLights appended to `points` on upload, followed by packed LTC tables when non-empty.
    List{GpuPointLight} rects;
    List{GpuSpotShadow} shadows;
    List{LightBounds} point_bounds;
    List{LightBounds} spot_bounds;
    List{GpuLightCluster} clusters;
    List{GpuLightIndex} indices;
    bool sized; // True when any point/spot light has non-zero radius or tube length.
    bool tubes; // True when any point light is a tube light (length > 0).
    bool point_shadows; // True when any point or rect area light has atlas cube shadows.
    bool spot_shadows; // True when any spot light casts shadows.
    bool dir_tiles; // True when any secondary directional light uses an atlas tile.
    bool soft_shadows; // True when any shadowed light uses PCSS (softness > 0).
    Texture* cookie; // Active SpotLight.map projection texture for this frame, or null.
    List{LightBufferSet*} sets; // Per-render buffer sets rotated within a frame.
    int used;
    uint frame;
    bool warned;
    LightBufferSet* current;
    int checks; // Renderer.check_compute: cluster grids verified.
    int check_failures; // Renderer.check_compute: cluster grids with mismatches.
}

LightData lights_gpu @private;

fn void LightData.init_lists(&self) {
    self.dirs.allocator = mem;
    self.points.allocator = mem;
    self.spots.allocator = mem;
    self.rects.allocator = mem;
    self.shadows.allocator = mem;
    self.point_bounds.allocator = mem;
    self.spot_bounds.allocator = mem;
    self.clusters.allocator = mem;
    self.indices.allocator = mem;
    self.sets.allocator = mem;
}

fn void LightData.clear(&self) {
    self.dirs.clear();
    self.points.clear();
    self.spots.clear();
    self.rects.clear();
    self.shadows.clear();
    self.point_bounds.clear();
    self.spot_bounds.clear();
    self.sized = false;
    self.tubes = false;
    self.point_shadows = false;
    self.spot_shadows = false;
    self.dir_tiles = false;
    self.soft_shadows = false;
    self.cookie = null;
}

// Assigns lights to clusters and uploads all light/shadow/cluster buffers into a fresh
// LightBufferSet for this render. With `check`, also runs CPU clustering and tracks GPU readback.
fn void LightData.upload(&self, ClusterGrid* grid, ClusterAssignFn assign, bool check = false) {
    if (!assign) assign = &cluster_lights_gpu;
    uint frame = sg::query_stats().prev_frame.frame_index;
    if (frame != self.frame || !self.sets.len()) {
        self.frame = frame;
        self.used = 0;
    }
    if (self.used >= self.sets.len()) {
        LightBufferSet* n = mem::new(LightBufferSet);
        n.gpu_clusters = { .label = "drie-clusters" };
        n.gpu_indices = { .label = "drie-cluster-lights" };
        n.gpu_ranges = { .cpu_write = true, .label = "drie-cluster-ranges" };
        n.check_clusters.allocator = mem;
        n.check_indices.allocator = mem;
        n.check_slot = -1;
        n.check_slot2 = -1;
        self.sets.push(n);
    }
    LightBufferSet* s = self.sets[self.used++];
    if (s.check_pending) s.compare_with_gpu();
    ClusterOutput out = { &self.clusters, &self.indices, &s.gpu_clusters, &s.gpu_indices, &s.gpu_ranges };
    s.on_gpu = assign(grid, self.point_bounds.array_view(), self.spot_bounds.array_view(), &out);
    if (self.rects.len()) {
        // Append rect lights and LTC tables after point lights so cluster point indices stay intact.
        self.points.add_all(&self.rects);
        foreach (e : ltc_entries()) self.points.push(e);
    }
    s.dirs.write(self.dirs.entries, self.dirs.len() * GpuDirLight::size);
    s.points.write(self.points.entries, self.points.len() * GpuPointLight::size);
    s.spots.write(self.spots.entries, self.spots.len() * GpuSpotLight::size);
    s.shadows.write(self.shadows.entries, self.shadows.len() * GpuSpotShadow::size);
    if (!s.on_gpu) {
        s.clusters.write(self.clusters.entries, self.clusters.len() * GpuLightCluster::size);
        s.indices.write(self.indices.entries, self.indices.len() * GpuLightIndex::size);
    } else if (check) {
        fill_clusters(
            grid,
            self.point_bounds.array_view(),
            self.spot_bounds.array_view(),
            &s.check_clusters,
            &s.check_indices
        );
        if (s.check_slot < 0) s.check_slot = readback_slot();
        if (s.check_slot2 < 0) s.check_slot2 = readback_slot();
        // Empty compute pass used solely to attach readback tracking to the cluster buffers.
        begin_compute("drie-clusters-check");
        s.check_pending = s.gpu_clusters.track(s.check_slot) && s.gpu_indices.track(s.check_slot2);
        end_compute();
    }
    self.current = s;
}

// Compares the GPU cluster grid read back from the previous frame against the CPU reference.
fn void LightBufferSet.compare_with_gpu(&self) {
    self.check_pending = false;
    sz nc               = self.check_clusters.len();
    sz ni               = self.check_indices.len();
    GpuLightCluster[] c = mem::new_array(GpuLightCluster, nc);
    GpuLightIndex[] ix  = mem::new_array(GpuLightIndex, ni);
    defer {
        mem::free(c);
        mem::free(ix);
    }
    if (!read_back(
        self.check_slot,
        c.ptr,
        nc * (sz)GpuLightCluster::size,
        true
    ) || !read_back(self.check_slot2, ix.ptr, ni * (sz)GpuLightIndex::size, true)) {
        io::eprintn("drie: check_compute: could not read the clusters back");
        return;
    }
    int bad_clusters;
    int bad_entries;
    for (sz i = 0; i < nc; i++) {
        if (c[i].range != self.check_clusters[i].range) bad_clusters++;
    }
    // Compare only the populated CPU prefix; the GPU buffer may hold excess capacity.
    for (sz i = 0; i < ni; i++) {
        if (ix[i].index != self.check_indices[i].index) bad_entries++;
    }
    lights_gpu.checks++;
    if (bad_clusters || bad_entries) {
        lights_gpu.check_failures++;
        io::eprintfn(
            "drie: check_compute: GPU clusters differ from the CPU's: %d of %d clusters, %d of %d entries",
            bad_clusters,
            nc,
            bad_entries,
            ni
        );
    }
}

fn void LightData.bind(&self, SgBindings* bind) {
    LightBufferSet* s = self.current;
    bind.views[shaders::VIEW_DIR_LIGHTS] = s.dirs.view;
    bind.views[shaders::VIEW_POINT_LIGHTS] = s.points.view;
    bind.views[shaders::VIEW_SPOT_LIGHTS] = s.spots.view;
    bind.views[shaders::VIEW_SPOT_SHADOWS] = s.shadows.view;
    bind.views[shaders::VIEW_LIGHT_CLUSTERS] = s.on_gpu ? s.gpu_clusters.view : s.clusters.view;
    bind.views[shaders::VIEW_LIGHT_INDICES] = s.on_gpu ? s.gpu_indices.view : s.indices.view;
}

// Checks whether another light fits under `cap` (Renderer.max_lights), warning once if exceeded.
fn bool LightData.room(&self, sz count, int cap) {
    if (cap <= 0 || count < cap) return true;
    if (!self.warned) io::eprintfn(
        "drie: more than %d lights of one kind (Renderer.max_lights), the rest are left out",
        cap
    );
    self.warned = true;
    return false;
}

GpuPointLight[] ltc_packed @local;

// Packs the LTC lookup tables (generated/ltc.c3) as GpuPointLight entries appended to the point
// light buffer (read by `ltc_tables` in shaders/shader_core.glsl; avoids consuming a texture slot).
// Each entry stores one LTC_1 texel in `pos` and one LTC_2 texel in `color`.
fn GpuPointLight[] ltc_entries() @private {
    if (ltc_packed.len) return ltc_packed;
    usz texels = LTC_1.len / 4;
    ltc_packed = mem::new_array(GpuPointLight, (sz)texels);
    foreach (i, &e : ltc_packed) {
        e.pos = { LTC_1[i * 4], LTC_1[i * 4 + 1], LTC_1[i * 4 + 2], LTC_1[i * 4 + 3] };
        e.color = { LTC_2[i * 4], LTC_2[i * 4 + 1], LTC_2[i * 4 + 2], LTC_2[i * 4 + 3] };
    }
    return ltc_packed;
}