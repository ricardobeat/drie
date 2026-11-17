// Clustered forward lighting: the lights of a render go to the GPU in storage buffers, and the
// view is cut into a grid of clusters (froxels: tiles of the screen, each cut into slices of
// view depth that grow exponentially). Every cluster gets the list of the point and spot lights
// that can reach it, and a fragment only looks at its own cluster's lights, so there is no
// limit on the number of lights in a scene, only on how many light one spot.
//
// The assignment runs behind ClusterAssignFn: it takes the grid and the lights' bounding
// spheres and fills the clusters and the index list, the two buffers the fragment shader
// reads. cluster_lights_gpu (the default where there are compute shaders) does it in a compute
// shader (cluster_cs in kernels.glsl), cluster_lights_cpu on the CPU, with the same result.
module drie;
import std::collections::list, std::math, std::io;
import sokol;

// The cluster grid: tiles across and up the screen, and slices of depth.
const int CLUSTER_X      = 16;
const int CLUSTER_Y      = 9;
const int CLUSTER_Z      = 24;
const int CLUSTER_COUNT  = CLUSTER_X * CLUSTER_Y * CLUSTER_Z;
// The slices of depth start here (m) whatever the camera's near plane: nearer than this,
// everything is in the first slice.
const float CLUSTER_NEAR = 0.1f;

// The view to cluster for. Matches cluster_of() in shader.glsl.
struct ClusterGrid {
    // world -> the camera's view space, and view space -> clip space
    Mat4 view;
    Mat4 projection;
    // where the first slice of depth starts and the last ends (m)
    float near;
    float far;
    // slices per unit of log(depth / near)
    float log_scale;
}

// What can be lit by a light: a sphere in world space. radius < 0: everywhere.
struct LightBounds {
    Vector3 center;
    float radius;
}

// Where an assignment puts the clusters (CLUSTER_COUNT of them) and the index list: per
// cluster its point lights, then its spot lights. On the CPU in the lists, on the GPU in the
// buffers (`gpu_ranges` is there for the assignment's own input).
struct ClusterOutput {
    List{GpuLightCluster}* clusters;
    List{GpuLightIndex}* indices;
    StorageBuffer* gpu_clusters;
    StorageBuffer* gpu_indices;
    StorageBuffer* gpu_ranges;
}

// Sorts lights into clusters. `points` and `spots` are indices into the lights' storage
// buffers by position. Returns true if it filled out.gpu_*, false if it filled the lists.
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

// The clusters a light's sphere touches, as ranges [lo, hi] along x, y and slices.
fn bool cluster_range(ClusterGrid* grid, LightBounds b, int[<3>]* lo, int[<3>]* hi) @private {
    // (lights that add nothing)
    if (b.radius == 0) return false;
    if (b.radius < 0) {
        *lo = { 0, 0, 0 };
        *hi = { CLUSTER_X - 1, CLUSTER_Y - 1, CLUSTER_Z - 1 };
        return true;
    }
    Vector3 c  =  grid.view.transform_point(b.center);
    float r    =  b.radius;
    // view space looks down -z
    float zmin = -c.z - r;
    float zmax = -c.z + r;
    if (zmax <= 0) return false;
    lo.z = cluster_slice(grid, zmin);
    hi.z = cluster_slice(grid, zmax);
    Mat4 p = grid.projection;
    if (zmin <= 1e-3f) {
        // around the camera: anywhere on the screen
        lo.x = 0;
        lo.y = 0;
        hi.x = CLUSTER_X - 1;
        hi.y = CLUSTER_Y - 1;
        return true;
    }
    // the corners of the sphere's box, all in front of the camera, through the projection
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

// The CPU version of ClusterAssignFn.
fn bool cluster_lights_cpu(ClusterGrid* grid, LightBounds[] points, LightBounds[] spots, ClusterOutput* out) {
    fill_clusters(grid, points, spots, out.clusters, out.indices);
    return false;
}

ComputePass* cluster_kernel @local;
List{LightRange} cluster_ranges @local;

// The compute version of ClusterAssignFn (falls back on the CPU without compute shaders). The
// clusters each light reaches are found here, as for the CPU (cluster_range); the lists are
// filled on the GPU, one thread per run of clusters.
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
    // the kernel's input can't be empty
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

// Sorts on the CPU: two passes over the lights, counting and then filling.
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
    // pass 1: how many lights of each kind per cluster (range.y: points, range.z: spots)
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
        // range.w: the next free entry while filling
        cl[i].range[3] = 0;
    }
    indices.clear();
    // at least one entry: a storage buffer can't be empty
    for (uint i = 0; i < math::max(total, 1u); i++) indices.push({ });
    GpuLightIndex* out = indices.entries;
    // pass 2: points first in each cluster, then spots after all of its points
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

// The sphere a point light can light: all of space without a distance cutoff.
fn LightBounds point_light_bounds(PointLight* pl) @private {
    return { pl.matrix_world.get_position(), pl.distance > 0 ? pl.distance : -1 };
}

// The sphere around a spot light's cone.
fn LightBounds spot_light_bounds(SpotLight* sl) @private {
    Vector3 pos = sl.matrix_world.get_position();
    if (sl.distance <= 0) return { pos, -1 };
    float a     = clamp(sl.angle, 0, PI / 2);
    float range = sl.distance;
    // (a source with a size lights a little beyond its cone)
    float extra = math::max(sl.radius, 0.0f);
    Vector3 dir = spot_direction(sl);
    if (a > PI / 4) return { pos + dir * (range * math::cos(a)), range * math::sin(a) + extra };
    float r = range / (2 * math::cos(a));
    return { pos + dir * r, r + extra };
}

// One storage buffer the shader reads, grown as needed. Written once per frame at most
// (sokol's transient writes), so each render() in a frame takes a set of its own.
struct LightBuffer @private {
    SgBuffer buffer;
    SgView view;
    usz capacity;
}

fn void LightBuffer.write(&self, void* data, sz bytes) {
    usz size = (usz)bytes;
    // a storage buffer can't be empty
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
    // filled by a compute shader instead (when `on_gpu`)
    StorageBuffer gpu_clusters;
    StorageBuffer gpu_indices;
    StorageBuffer gpu_ranges;
    bool on_gpu;
    // Renderer.check_compute: the CPU's result of the last frame, to compare with the GPU's
    List{GpuLightCluster} check_clusters;
    List{GpuLightIndex} check_indices;
    int check_slot;
    int check_slot2;
    bool check_pending;
}

// The lights of one render(), as the shader gets them.
struct LightData @private {
    List{GpuDirLight} dirs;
    List{GpuPointLight} points;
    List{GpuSpotLight} spots;
    List{GpuSpotShadow} shadows;
    List{LightBounds} point_bounds;
    List{LightBounds} spot_bounds;
    List{GpuLightCluster} clusters;
    List{GpuLightIndex} indices;
    // the sets of buffers, one per render() in a frame; `used` of them this frame
    List{LightBufferSet*} sets;
    int used;
    uint frame;
    bool warned;
    LightBufferSet* current;
    // Renderer.check_compute: comparisons made, and how many differed
    int checks;
    int check_failures;
}

LightData lights_gpu @private;

fn void LightData.init_lists(&self) {
    self.dirs.allocator = mem;
    self.points.allocator = mem;
    self.spots.allocator = mem;
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
    self.shadows.clear();
    self.point_bounds.clear();
    self.spot_bounds.clear();
}

// Sorts the lights into clusters and writes them all to a set of buffers of their own.
// `check`: also sorts them on the CPU and compares (a frame later) with the GPU's result.
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
        // (an empty compute pass, just to track the buffers in)
        begin_compute("drie-clusters-check");
        s.check_pending = s.gpu_clusters.track(s.check_slot) && s.gpu_indices.track(s.check_slot2);
        end_compute();
    }
    self.current = s;
}

// Renderer.check_compute: compares the GPU's clusters of the last frame with the CPU's.
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
    // (only where the CPU's index list has entries: the GPU's may be longer)
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

// Room for one more light of a kind, up to Renderer.max_lights (a one-time warning past it).
fn bool LightData.room(&self, sz count, int cap) {
    if (cap <= 0 || count < cap) return true;
    if (!self.warned) io::eprintfn(
        "drie: more than %d lights of one kind (Renderer.max_lights), the rest are left out",
        cap
    );
    self.warned = true;
    return false;
}