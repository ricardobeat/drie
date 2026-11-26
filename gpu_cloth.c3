// GpuCloth: Cloth (cloth.c3) solved on the GPU with compute kernels (gpu_cloth.glsl), for
// sheets of tens of thousands of points. The same XPBD in small steps: each fixed step of
// 1/60 s cut into `substeps`, each substep moving the points by their velocity, projecting the
// stretch, shear and bend links and the tethers to the pins once, and pushing the points out of
// the colliders (spheres, capsules, boxes, planes, swept over the substeps) and above the floor.
//
//   - the links are colored on the CPU so that no two of a color share a point, and each color
//     is one dispatch: Gauss-Seidel in batches, in the same order every run, without atomics
//   - self-collision: a spatial hash of the points (counted into cells, summed, filled) and a
//     push apart of the points nearer than `self_distance` (no nearer than at rest), summed in
//     fixed point: the order a cell lists its points in, which changes from run to run, does
//     not change the result. The same calls give the same points every run.
//   - the points stay on the GPU: gc_write_cs writes the mesh's positions and normals into its
//     vertex buffer (BufferGeometry.gpu_vertices); nothing is copied either way per frame.
//
// The air pushes on each triangle as in Cloth (wind, gust, drag); a point's force is gathered
// from the triangles around it. Not here: from_fine's detail grid, moved()/mark() (the points
// are not on the CPU), the cloth's own bounds (make_mesh gives the mesh a sphere the cloth is
// expected to stay in; set_bounds for another).
//
//     GpuCloth* c = gpu_cloth::grid(200, 200, { -1.5, 2, -1.5 }, { 3, 0, 0 }, { 0, 0, 3 }, 1.4);
//     c.floor_y = 0;
//     c.self_distance = 0.015;
//     c.add_box({ 0.4, 0.2, 0.4 }, drie::mat4_compose({ 0, 0.2, 0 }, { }, { 1, 1, 1 }));
//     scene.add(c.make_mesh(drie::mesh_standard_material(color: 0xd8d0c0, side: DOUBLE_SIDE)));
//     // each frame, before rendering:
//     c.update(delta);
module drie::gpu_cloth;
import std::math, std::collections::list, drie, drie::shaders, sokol;

const float STEP    = 1.0f / 60;
const int MAX_STEPS = 4;
// A collider that moves further than this in one update has been put somewhere else, not
// swept there through the cloth.
const float JUMP = 1.0f;
const uint NONE  = 0xffffffff;
const int GROUP  = 128;

enum ColliderKind : char {
    SPHERE,
    CAPSULE,
    BOX,
    PLANE,
}

// As cloth::Collider: a sphere (a), a capsule (a..b) of `radius`; a box centered at a, `half`
// its half size along `axes`; a plane through a facing axes[0].
struct Collider {
    ColliderKind kind;
    float radius;
    Vector3 from_a, from_b;
    Vector3 a, b;
    bool active;
    Vector3 half;
    Vector3[3] from_axes;
    Vector3[3] axes;
}

// A run of links of one kind and color, solved in one dispatch.
struct Batch {
    uint first, count;
    // 0 stretch, 1 shear, 2 bend: which compliance
    int kind;
}

// The compute kernels, made once for every cloth.
struct Kernels {
    ComputePass* begin;
    ComputePass* forces;
    ComputePass* integrate;
    ComputePass* links;
    ComputePass* tethers;
    ComputePass* clear;
    ComputePass* count;
    ComputePass* scan_block;
    ComputePass* scan_sums;
    ComputePass* scan_add;
    ComputePass* fill;
    ComputePass* repel;
    ComputePass* collide;
    ComputePass* write;
}

Kernels* kernels @local;

struct GpuCloth {
    int cols, rows;
    // where the points are kept from (the rest shape's first point), as Cloth.origin
    Vector3 origin;
    // the rest shape, from origin
    Vector3[] rest;
    // per point: the inverse mass (0 pinned) and a pinned point's place
    float[] inv_mass;
    Vector3[] pin_to;
    List{shaders::GcLink} links;
    List{Batch} batches;
    shaders::GcTether[] tether_data;
    List{Collider} colliders;
    bool tethers_dirty;
    bool pins_dirty;
    // the points' state must be written to the GPU (at the start and by reset)
    bool state_dirty;
    bool has_tethers;

    int substeps;
    // link passes per substep
    int iterations;
    float stretch_compliance;
    float shear_compliance;
    float bend_compliance;
    float tether_slack;
    float damping;
    Vector3 gravity;
    Vector3 wind;
    float gust;
    float drag;
    // points stay this far from the colliders' surfaces
    float thickness;
    float friction;
    // a floor (y up) the points stay above; -1e30: none
    float floor_y;
    float mass;
    // Self-collision: points are kept this far apart (0: none), or as far as they are at rest
    // if that is less. About the spacing of the points, so that no point slips between four.
    float self_distance;
    // the share of the overlap pushed out per substep (1: all of it)
    float self_stiffness;
    // 0 slides over the cloth's other layers, 1 moves with them
    float self_friction;
    // self-collision every this many substeps (1: each)
    int self_every;

    double time;
    float accumulator;
    ulong steps;
    Mesh* mesh;

    // GPU state: per point (vec4s), the links, the hash
    StorageBuffer* pos;
    StorageBuffer* prev;
    StorageBuffer* vel;
    StorageBuffer* start;
    StorageBuffer* acc;
    StorageBuffer* rest_buf;
    StorageBuffer* pins;
    StorageBuffer* link_buf;
    StorageBuffer* tether_buf;
    StorageBuffer* collider_buf;
    StorageBuffer* cells;
    StorageBuffer* sums;
    StorageBuffer* ids;
    StorageBuffer* push;
    // hash cells (a power of two, at least twice the points)
    int table;
    // the readback slot of read_points (-1: none yet)
    int read_slot;
}

fn uint GpuCloth.index(&self, int col, int row) @inline => (uint)(row * self.cols + col);
fn uint GpuCloth.count(&self) @inline => (uint)(self.cols * self.rows);
fn sz GpuCloth.link_count(&self) => self.links.len();
fn sz GpuCloth.batch_count(&self) => self.batches.len();

// A flat grid of `cols` x `rows` points from `origin`, `across` to the last column and `down`
// to the last row. Its front (the mesh's front faces) is cross(across, -down). `mass` is the
// whole cloth's, in kg.
fn GpuCloth* grid(int cols, int rows, Vector3 origin, Vector3 across, Vector3 down, float mass = 0.4f) {
    Vector3[] rest = mem::new_array(Vector3, (sz)cols * rows);
    defer free(rest);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            rest[r * cols + c] = origin + across * ((float)c / (float)(cols - 1)) + down * ((float)r / (float)(rows - 1));
        }
    }
    return from_positions(cols, rows, rest, mass);
}

// A grid of points at rest where `rest` has them (row by row, `cols` to a row).
fn GpuCloth* from_positions(int cols, int rows, Vector3[] rest, float mass = 0.4f) {
    GpuCloth* c = mem::new(GpuCloth);
    c.cols = cols;
    c.rows = rows;
    sz n = (sz)cols * rows;
    c.rest = mem::new_array(Vector3, n);
    c.inv_mass = mem::new_array(float, n);
    c.pin_to = mem::new_array(Vector3, n);
    c.links.allocator = mem;
    c.batches.allocator = mem;
    c.colliders.allocator = mem;
    c.substeps = 8;
    c.iterations = 1;
    c.stretch_compliance = 0;
    c.shear_compliance = 1e-3f;
    c.bend_compliance = 1e-2f;
    c.tether_slack = 0.02f;
    c.damping = 0.6f;
    c.gravity = { 0, -9.81f, 0 };
    c.drag = 0.6f;
    c.thickness = 0.02f;
    c.friction = 0.3f;
    c.floor_y = -1e30f;
    c.mass = mass;
    c.self_stiffness = 0.5f;
    c.self_friction = 0.2f;
    c.self_every = 1;
    c.read_slot = -1;
    c.origin = rest[0];
    float w = (float)n / math::max(mass, 1e-6f);
    for (sz i = 0; i < n; i++) {
        c.rest[i] = rest[i] - c.origin;
        c.pin_to[i] = c.rest[i];
        c.inv_mass[i] = w;
    }
    @pool() {
        List{shaders::GcLink}[3] kinds;
        for (int k = 0; k < 3; k++) kinds[k].allocator = tmem;
        for (int r = 0; r < rows; r++) {
            for (int k = 0; k < cols; k++) {
                uint i = c.index(k, r);
                if (k + 1 < cols) c.link(&kinds[0], i, c.index(k + 1, r));
                if (r + 1 < rows) c.link(&kinds[0], i, c.index(k, r + 1));
                if (k + 1 < cols && r + 1 < rows) {
                    c.link(&kinds[1], i, c.index(k + 1, r + 1));
                    c.link(&kinds[1], c.index(k + 1, r), c.index(k, r + 1));
                }
                if (k + 2 < cols) c.link(&kinds[2], i, c.index(k + 2, r));
                if (r + 2 < rows) c.link(&kinds[2], i, c.index(k, r + 2));
            }
        }
        for (int k = 0; k < 3; k++) c.add_colored(&kinds[k], k, n);
    };
    c.tethers_dirty = true;
    c.pins_dirty = true;
    c.state_dirty = true;
    return c;
}

fn void GpuCloth.link(&self, List{shaders::GcLink}* list, uint a, uint b) {
    list.push({ a, b, (self.rest[a] - self.rest[b]).length(), 0 });
}

// Appends `list` to the links color by color (each link the first color its points' other
// links don't have), a batch per color.
fn void GpuCloth.add_colored(&self, List{shaders::GcLink}* list, int kind, sz points) {
    uint[] used = mem::new_array(uint, points);
    char[] color = mem::new_array(char, list.len());
    defer {
        free(used);
        free(color);
    }
    int colors;
    foreach (i, l : *list) {
        uint taken = used[l.a] | used[l.b];
        int k;
        while (k < 31 && (taken & (1u << k))) k++;
        color[i] = (char)k;
        used[l.a] |= 1u << k;
        used[l.b] |= 1u << k;
        colors = math::max(colors, k + 1);
    }
    for (int k = 0; k < colors; k++) {
        uint first = (uint)self.links.len();
        foreach (i, l : *list) {
            if (color[i] == k) self.links.push(l);
        }
        self.batches.push({ first, (uint)self.links.len() - first, kind });
    }
}

fn void GpuCloth.free(&self) {
    free(self.rest);
    free(self.inv_mass);
    free(self.pin_to);
    free(self.tether_data);
    self.links.free();
    self.batches.free();
    self.colliders.free();
    foreach (b : (StorageBuffer*[14]) { self.pos, self.prev, self.vel, self.start, self.acc, self.rest_buf, self.pins,
        self.link_buf, self.tether_buf, self.collider_buf, self.cells, self.sums, self.ids, self.push }) {
        if (!b) continue;
        b.dispose();
        free(b);
    }
    free(self);
}

// Holds point `i` where it is at rest, or where move_pin puts it.
fn void GpuCloth.pin(&self, uint i) {
    self.inv_mass[i] = 0;
    self.pin_to[i] = self.rest[i];
    self.tethers_dirty = true;
    self.pins_dirty = true;
}

fn void GpuCloth.pin_row(&self, int row) {
    for (int k = 0; k < self.cols; k++) self.pin(self.index(k, row));
}

fn void GpuCloth.unpin(&self, uint i) {
    self.inv_mass[i] = (float)self.count() / math::max(self.mass, 1e-6f);
    self.tethers_dirty = true;
    self.pins_dirty = true;
}

// Moves pinned point `i`: it gets there at the next fixed step.
fn void GpuCloth.move_pin(&self, uint i, Vector3 to) {
    self.pin_to[i] = to - self.origin;
    self.pins_dirty = true;
}

// Puts every point back where it is at rest, still.
fn void GpuCloth.reset(&self) {
    self.state_dirty = true;
    self.accumulator = 0;
    self.time = 0;
    self.steps = 0;
}

fn int GpuCloth.add_sphere(&self, Vector3 center, float radius) {
    center -= self.origin;
    self.colliders.push({ .kind = SPHERE, .radius = radius, .from_a = center, .from_b = center, .a = center, .b = center, .active = true });
    return (int)self.colliders.len() - 1;
}

fn int GpuCloth.add_capsule(&self, Vector3 a, Vector3 b, float radius) {
    a -= self.origin;
    b -= self.origin;
    self.colliders.push({ .kind = CAPSULE, .radius = radius, .from_a = a, .from_b = b, .a = a, .b = b, .active = true });
    return (int)self.colliders.len() - 1;
}

fn Vector3 column(drie::Mat4 m, int i) @local => { m[i][0], m[i][1], m[i][2] };

// A box of half size `half`, placed by `placement` (centered on its origin, its axes along the
// matrix's columns; their length is left out).
fn int GpuCloth.add_box(&self, Vector3 half, drie::Mat4 placement) {
    Collider k = { .kind = BOX, .half = half, .active = true };
    k.a = column(placement, 3) - self.origin;
    for (int i = 0; i < 3; i++) k.axes[i] = column(placement, i).normalize();
    k.from_a = k.a;
    k.from_axes = k.axes;
    self.colliders.push(k);
    return (int)self.colliders.len() - 1;
}

// A plane through `point`: the points stay on the side `normal` faces (thickness away).
fn int GpuCloth.add_plane(&self, Vector3 point, Vector3 normal) {
    point -= self.origin;
    Collider k = { .kind = PLANE, .a = point, .from_a = point, .active = true };
    k.axes[0] = normal.normalize();
    k.from_axes = k.axes;
    self.colliders.push(k);
    return (int)self.colliders.len() - 1;
}

// Moves a sphere or capsule (b is ignored for a sphere), seen to travel there over the next
// update's steps; from inactive, or further than JUMP, it appears there.
fn void GpuCloth.move_collider(&self, int id, Vector3 a, Vector3 b = { 0, 0, 0 }) {
    Collider* k = &self.colliders[id];
    a -= self.origin;
    b -= self.origin;
    if (k.kind == SPHERE) b = a;
    if (!k.active || (a - k.from_a).length() > JUMP || (b - k.from_b).length() > JUMP) {
        k.from_a = a;
        k.from_b = b;
    }
    k.a = a;
    k.b = b;
}

// Moves (and turns) a box, as move_collider; turned more than 60 degrees, it appears there.
fn void GpuCloth.move_box(&self, int id, drie::Mat4 placement) {
    Collider* k = &self.colliders[id];
    Vector3 a = column(placement, 3) - self.origin;
    Vector3[3] axes;
    for (int i = 0; i < 3; i++) axes[i] = column(placement, i).normalize();
    bool turned = axes[0].dot(k.from_axes[0]) < 0.5f || axes[1].dot(k.from_axes[1]) < 0.5f;
    if (!k.active || (a - k.from_a).length() > JUMP || turned) {
        k.from_a = a;
        k.from_axes = axes;
    }
    k.a = a;
    k.axes = axes;
}

fn void GpuCloth.set_collider_active(&self, int id, bool active) => self.colliders[id].active = active;

// Each free point's nearest pin along the cloth at rest (Dijkstra over the grid's neighbours,
// diagonals too, with a binary heap), and how far it is.
fn void GpuCloth.build_tethers(&self) {
    self.tethers_dirty = false;
    uint n = self.count();
    if (!self.tether_data.len) self.tether_data = mem::new_array(shaders::GcTether, n);
    float[] dist = mem::new_array(float, n);
    uint[] from = mem::new_array(uint, n);
    defer {
        free(dist);
        free(from);
    }
    List{HeapEntry} heap;
    heap.allocator = mem;
    defer heap.free();
    for (uint i = 0; i < n; i++) {
        dist[i] = 1e30f;
        from[i] = NONE;
        if (self.inv_mass[i] == 0) {
            dist[i] = 0;
            from[i] = i;
            heap_push(&heap, { 0, i });
        }
    }
    while (heap.len()) {
        HeapEntry e = heap_pop(&heap);
        if (e.dist > dist[e.at]) continue;
        int col = (int)(e.at % (uint)self.cols);
        int row = (int)(e.at / (uint)self.cols);
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                int c2 = col + dc;
                int r2 = row + dr;
                if ((dr == 0 && dc == 0) || c2 < 0 || r2 < 0 || c2 >= self.cols || r2 >= self.rows) continue;
                uint j = self.index(c2, r2);
                float d = e.dist + (self.rest[e.at] - self.rest[j]).length();
                if (d < dist[j]) {
                    dist[j] = d;
                    from[j] = from[e.at];
                    heap_push(&heap, { d, j });
                }
            }
        }
    }
    self.has_tethers = false;
    for (uint i = 0; i < n; i++) {
        bool tied = self.inv_mass[i] != 0 && from[i] != NONE;
        self.tether_data[i] = { tied ? from[i] : NONE, tied ? dist[i] : 0 };
        self.has_tethers = self.has_tethers || tied;
    }
}

struct HeapEntry @local {
    float dist;
    uint at;
}

fn void heap_push(List{HeapEntry}* h, HeapEntry e) @local {
    h.push(e);
    sz i = h.len() - 1;
    while (i > 0) {
        sz up = (i - 1) / 2;
        if ((*h)[up].dist <= (*h)[i].dist) break;
        HeapEntry t = (*h)[up];
        (*h)[up] = (*h)[i];
        (*h)[i] = t;
        i = up;
    }
}

fn HeapEntry heap_pop(List{HeapEntry}* h) @local {
    HeapEntry top = (*h)[0];
    HeapEntry last = h.pop()!!;
    sz n = h.len();
    if (!n) return top;
    (*h)[0] = last;
    sz i = 0;
    while (true) {
        sz l = i * 2 + 1;
        sz r = l + 1;
        sz m = i;
        if (l < n && (*h)[l].dist < (*h)[m].dist) m = l;
        if (r < n && (*h)[r].dist < (*h)[m].dist) m = r;
        if (m == i) break;
        HeapEntry t = (*h)[m];
        (*h)[m] = (*h)[i];
        (*h)[i] = t;
        i = m;
    }
    return top;
}

// A mesh of the cloth, drawn where the points are (its matrix is the identity: add it anywhere
// in the scene), its vertices written on the GPU. Use a DOUBLE_SIDE material to see both faces.
// Its uvs are in meters along the cloth at rest, as Cloth's. Its bounding sphere (for culling
// and shadows) is centered on the cloth at rest, as wide as the cloth's diagonal.
fn Mesh* GpuCloth.make_mesh(&self, Material* material = null) {
    int cols = self.cols;
    int rows = self.rows;
    BufferGeometry* g = drie::buffer_geometry();
    g.dynamic = true;
    g.gpu_vertices = true;
    float[] down = mem::new_array(float, cols);
    defer free(down);
    Vector3 lo = { 1e30f, 1e30f, 1e30f };
    Vector3 hi = -lo;
    for (int r = 0; r < rows; r++) {
        float across = 0;
        for (int k = 0; k < cols; k++) {
            int i = r * cols + k;
            if (k > 0) across += (self.rest[i] - self.rest[i - 1]).length();
            if (r > 0) down[k] += (self.rest[i] - self.rest[i - cols]).length();
            Vector3 p = self.origin + self.rest[i];
            lo = math::min(lo, p);
            hi = math::max(hi, p);
            g.add_vertex(p, { 0, 0, 1 }, { across, -down[k] });
        }
    }
    for (int r = 0; r + 1 < rows; r++) {
        for (int k = 0; k + 1 < cols; k++) {
            uint a = (uint)(r * cols + k);
            uint b = a + (uint)cols;
            uint c = b + 1;
            uint d = a + 1;
            // (the diagonal alternates, so the folds have no grain)
            if ((k + r) % 2 == 0) {
                g.add_triangle(a, b, c);
                g.add_triangle(a, c, d);
            } else {
                g.add_triangle(a, b, d);
                g.add_triangle(b, c, d);
            }
        }
    }
    g.bounds_center = (lo + hi) * 0.5f;
    g.bounds_radius = math::max((hi - lo).length(), 1e-3f);
    Mesh* m = drie::mesh(g, material);
    m.matrix_auto_update = false;
    m.matrix_world_auto_update = false;
    m.matrix = drie::mat4_identity();
    m.matrix_world = drie::mat4_identity();
    self.mesh = m;
    return m;
}

// The sphere the mesh is culled and shadowed with, in the world.
fn void GpuCloth.set_bounds(&self, Vector3 center, float radius) {
    if (!self.mesh) return;
    self.mesh.geometry.bounds_center = center;
    self.mesh.geometry.bounds_radius = radius;
}

fn void make_kernels() @local {
    if (kernels) return;
    SgBackend be = sg::query_backend();
    Kernels* k = mem::new(Kernels);
    k.begin = drie::compute_pass(shaders::gc_begin_shader_desc(be), "gpu-cloth-begin");
    k.forces = drie::compute_pass(shaders::gc_forces_shader_desc(be), "gpu-cloth-forces");
    k.integrate = drie::compute_pass(shaders::gc_integrate_shader_desc(be), "gpu-cloth-integrate");
    k.links = drie::compute_pass(shaders::gc_links_shader_desc(be), "gpu-cloth-links");
    k.tethers = drie::compute_pass(shaders::gc_tethers_shader_desc(be), "gpu-cloth-tethers");
    k.clear = drie::compute_pass(shaders::gc_clear_shader_desc(be), "gpu-cloth-clear");
    k.count = drie::compute_pass(shaders::gc_count_shader_desc(be), "gpu-cloth-count");
    k.scan_block = drie::compute_pass(shaders::gc_scan_block_shader_desc(be), "gpu-cloth-scan-block");
    k.scan_sums = drie::compute_pass(shaders::gc_scan_sums_shader_desc(be), "gpu-cloth-scan-sums");
    k.scan_add = drie::compute_pass(shaders::gc_scan_add_shader_desc(be), "gpu-cloth-scan-add");
    k.fill = drie::compute_pass(shaders::gc_fill_shader_desc(be), "gpu-cloth-fill");
    k.repel = drie::compute_pass(shaders::gc_self_shader_desc(be), "gpu-cloth-self");
    k.collide = drie::compute_pass(shaders::gc_collide_shader_desc(be), "gpu-cloth-collide");
    k.write = drie::compute_pass(shaders::gc_write_shader_desc(be), "gpu-cloth-write");
    kernels = k;
}

// The GPU's buffers, made the first time; the points' state written at the start and after
// reset, the pins when they change, the colliders every update.
fn void GpuCloth.sync(&self) {
    uint n = self.count();
    sz vec4s = (sz)n * 16;
    if (!self.pos) {
        self.pos = drie::storage_buffer(vec4s, label: "gpu-cloth-pos");
        self.prev = drie::storage_buffer(vec4s, label: "gpu-cloth-prev");
        self.vel = drie::storage_buffer(vec4s, label: "gpu-cloth-vel");
        self.start = drie::storage_buffer(vec4s, label: "gpu-cloth-start");
        self.acc = drie::storage_buffer(vec4s, label: "gpu-cloth-acc");
        self.rest_buf = drie::storage_buffer(vec4s, label: "gpu-cloth-rest");
        self.pins = drie::storage_buffer(vec4s, cpu_write: true, label: "gpu-cloth-pins");
        self.link_buf = drie::storage_buffer(16, label: "gpu-cloth-links");
        self.link_buf.write(self.links.entries, (sz)self.links.len() * shaders::GcLink::size);
        self.tether_buf = drie::storage_buffer((sz)n * shaders::GcTether::size, label: "gpu-cloth-tethers");
        self.collider_buf = drie::storage_buffer(shaders::GcCollider::size, cpu_write: true, label: "gpu-cloth-colliders");
        self.table = 1024;
        while (self.table < (int)n * 2) self.table *= 2;
        // (a cell past the table, and the scan's last block whole)
        self.cells = drie::storage_buffer((sz)(self.table + 1024) * 4, label: "gpu-cloth-cells");
        self.sums = drie::storage_buffer(1024 * 4, label: "gpu-cloth-sums");
        self.ids = drie::storage_buffer((sz)n * 4, label: "gpu-cloth-ids");
        self.push = drie::storage_buffer(vec4s, label: "gpu-cloth-push");
    }
    if (self.state_dirty) {
        self.state_dirty = false;
        drie::Vec4[] data = mem::new_array(drie::Vec4, n);
        defer free(data);
        foreach (i, p : self.rest) data[i] = { p.x, p.y, p.z, 0 };
        self.rest_buf.write(data.ptr, vec4s);
        foreach (i, &d : data) {
            Vector3 p = self.inv_mass[i] == 0 ? self.pin_to[i] : self.rest[i];
            *d = { p.x, p.y, p.z, 0 };
        }
        self.pos.write(data.ptr, vec4s);
        self.prev.write(data.ptr, vec4s);
        self.start.write(data.ptr, vec4s);
        foreach (&d : data) *d = { 0, 0, 0, 0 };
        self.vel.write(data.ptr, vec4s);
        self.acc.write(data.ptr, vec4s);
    }
    if (self.tethers_dirty) {
        self.build_tethers();
        self.tether_buf.write(self.tether_data.ptr, (sz)n * shaders::GcTether::size);
    }
    if (self.pins_dirty) {
        self.pins_dirty = false;
        drie::Vec4[] data = mem::new_array(drie::Vec4, n);
        defer free(data);
        foreach (i, &d : data) *d = { self.pin_to[i].x, self.pin_to[i].y, self.pin_to[i].z, self.inv_mass[i] };
        self.pins.write(data.ptr, vec4s);
    }
}

fn void GpuCloth.write_colliders(&self) {
    sz count = self.colliders.len();
    if (!count) return;
    shaders::GcCollider[] data = mem::new_array(shaders::GcCollider, count);
    defer free(data);
    foreach (i, k : self.colliders) {
        data[i] = {
            .info = { (float)(int)k.kind, k.radius, k.active ? 1 : 0, 0 },
            .from_a = v4(k.from_a),
            .a = v4(k.a),
            .from_b = v4(k.from_b),
            .b = v4(k.b),
            .from_x = v4(k.from_axes[0]),
            .from_y = v4(k.from_axes[1]),
            .x = v4(k.axes[0]),
            .y = v4(k.axes[1]),
            .half_size = v4(k.half),
        };
    }
    self.collider_buf.write(data.ptr, (sz)count * shaders::GcCollider::size);
}

fn drie::Vec4 v4(Vector3 v) @local => { v.x, v.y, v.z, 0 };

fn void GpuCloth.bind(&self) {
    Kernels* k = kernels;
    k.begin.set_buffer(shaders::VIEW_GC_BEGIN_POS, self.pos);
    k.begin.set_buffer(shaders::VIEW_GC_BEGIN_START, self.start);
    k.begin.set_buffer(shaders::VIEW_GC_BEGIN_VEL, self.vel);
    k.begin.set_buffer(shaders::VIEW_GC_BEGIN_PINS, self.pins);
    k.forces.set_buffer(shaders::VIEW_GC_FORCES_POS, self.pos);
    k.forces.set_buffer(shaders::VIEW_GC_FORCES_VEL, self.vel);
    k.forces.set_buffer(shaders::VIEW_GC_FORCES_ACC, self.acc);
    k.integrate.set_buffer(shaders::VIEW_GC_INT_POS, self.pos);
    k.integrate.set_buffer(shaders::VIEW_GC_INT_PREV, self.prev);
    k.integrate.set_buffer(shaders::VIEW_GC_INT_VEL, self.vel);
    k.integrate.set_buffer(shaders::VIEW_GC_INT_ACC, self.acc);
    k.integrate.set_buffer(shaders::VIEW_GC_INT_PINS, self.pins);
    k.links.set_buffer(shaders::VIEW_GC_LINKS_POS, self.pos);
    k.links.set_buffer(shaders::VIEW_GC_LINKS_LINKS, self.link_buf);
    k.links.set_buffer(shaders::VIEW_GC_LINKS_PINS, self.pins);
    k.tethers.set_buffer(shaders::VIEW_GC_TETH_POS, self.pos);
    k.tethers.set_buffer(shaders::VIEW_GC_TETH_TETHERS, self.tether_buf);
    k.clear.set_buffer(shaders::VIEW_GC_CLEAR_CELLS, self.cells);
    k.count.set_buffer(shaders::VIEW_GC_COUNT_POS, self.pos);
    k.count.set_buffer(shaders::VIEW_GC_COUNT_CELLS, self.cells);
    k.scan_block.set_buffer(shaders::VIEW_GC_SCANB_CELLS, self.cells);
    k.scan_block.set_buffer(shaders::VIEW_GC_SCANB_SUMS, self.sums);
    k.scan_sums.set_buffer(shaders::VIEW_GC_SCANS_SUMS, self.sums);
    k.scan_add.set_buffer(shaders::VIEW_GC_SCANA_CELLS, self.cells);
    k.scan_add.set_buffer(shaders::VIEW_GC_SCANA_SUMS, self.sums);
    k.fill.set_buffer(shaders::VIEW_GC_FILL_POS, self.pos);
    k.fill.set_buffer(shaders::VIEW_GC_FILL_CELLS, self.cells);
    k.fill.set_buffer(shaders::VIEW_GC_FILL_IDS, self.ids);
    k.repel.set_buffer(shaders::VIEW_GC_SELF_POS, self.pos);
    k.repel.set_buffer(shaders::VIEW_GC_SELF_PREV, self.prev);
    k.repel.set_buffer(shaders::VIEW_GC_SELF_REST, self.rest_buf);
    k.repel.set_buffer(shaders::VIEW_GC_SELF_CELLS, self.cells);
    k.repel.set_buffer(shaders::VIEW_GC_SELF_IDS, self.ids);
    k.repel.set_buffer(shaders::VIEW_GC_SELF_PINS, self.pins);
    k.repel.set_buffer(shaders::VIEW_GC_SELF_PUSH, self.push);
    k.collide.set_buffer(shaders::VIEW_GC_COLL_POS, self.pos);
    k.collide.set_buffer(shaders::VIEW_GC_COLL_PREV, self.prev);
    k.collide.set_buffer(shaders::VIEW_GC_COLL_VEL, self.vel);
    k.collide.set_buffer(shaders::VIEW_GC_COLL_PINS, self.pins);
    k.collide.set_buffer(shaders::VIEW_GC_COLL_PUSH, self.push);
    k.collide.set_buffer(shaders::VIEW_GC_COLL_COLLIDERS, self.collider_buf);
    k.write.set_buffer(shaders::VIEW_GC_WRITE_POS, self.pos);
    k.write.set_buffer(shaders::VIEW_GC_WRITE_START, self.start);
}

// The uniforms every kernel shares, for a substep at `t` (and `t_prev` a substep ago) of the
// colliders' way.
fn shaders::GcParams GpuCloth.params(&self, float t, float t_prev) {
    float h = STEP / self.substeps;
    uint n = self.count();
    float span = (self.rest[self.index(self.cols - 1, 0)] - self.rest[0]).length();
    float drop = (self.rest[self.index(0, self.rows - 1)] - self.rest[0]).length();
    float cell = math::max(self.self_distance, 1e-4f);
    return {
        .gc_counts = { (int)n, self.cols, self.rows, self.table },
        .gc_range = { 0, 0, (int)self.colliders.len(), self.floor_y > -1e29f ? 1 : 0 },
        .gc_step = { h, math::exp(-self.damping * h) / h, 0, self.friction },
        .gc_gravity = { self.gravity.x, self.gravity.y, self.gravity.z, self.thickness },
        // a third of a triangle's force on each of its points, as an acceleration
        .gc_wind = { self.wind.x, self.wind.y, self.wind.z, self.drag * 0.5f / 3 * (float)n / math::max(self.mass, 1e-6f) },
        .gc_sweep = { t, t_prev, self.floor_y - self.origin.y, self.self_distance },
        .gc_origin = { self.origin.x, self.origin.y, self.origin.z, 1 },
        .gc_gust = { self.gust, (float)self.time, span, drop },
        .gc_self = { 1 / cell, self.self_stiffness, self.tether_slack, self.self_friction },
    };
}

fn void dispatch(ComputePass* k, shaders::GcParams* u, int threads, int group = GROUP) @local {
    k.set_uniforms(shaders::UB_GC_PARAMS, u, shaders::GcParams::size);
    k.dispatch((threads + group - 1) / group);
}

// One fixed step, its colliders going from `from` to `until` (0..1) of their way.
fn void GpuCloth.step_once(&self, float from, float until) {
    Kernels* k = kernels;
    int n = (int)self.count();
    shaders::GcParams u = self.params(from, from);
    dispatch(k.begin, &u, n);
    dispatch(k.forces, &u, n);
    float h = STEP / self.substeps;
    float[3] alpha = { self.stretch_compliance / (h * h), self.shear_compliance / (h * h), self.bend_compliance / (h * h) };
    int blocks = (self.table + 1 + 1023) / 1024;
    for (int s = 0; s < self.substeps; s++) {
        float t_prev = from + (until - from) * (float)s / self.substeps;
        float t = from + (until - from) * (float)(s + 1) / self.substeps;
        u = self.params(t, t_prev);
        dispatch(k.integrate, &u, n);
        for (int it = 0; it < math::max(self.iterations, 1); it++) {
            foreach (b : self.batches) {
                u.gc_range[0] = (int)b.first;
                u.gc_range[1] = (int)b.count;
                u.gc_step[2] = alpha[b.kind];
                dispatch(k.links, &u, (int)b.count);
            }
        }
        u = self.params(t, t_prev);
        if (self.has_tethers) dispatch(k.tethers, &u, n);
        bool self_now = self.self_distance > 0 && s % math::max(self.self_every, 1) == 0;
        if (self_now) {
            dispatch(k.clear, &u, self.table + 1);
            dispatch(k.count, &u, n);
            dispatch(k.scan_block, &u, blocks * 256, 256);
            dispatch(k.scan_sums, &u, 256, 256);
            dispatch(k.scan_add, &u, blocks * 256, 256);
            dispatch(k.fill, &u, n);
            dispatch(k.repel, &u, n);
        } else {
            u.gc_sweep[3] = 0;
        }
        dispatch(k.collide, &u, n);
    }
    self.time += STEP;
    self.steps++;
}

// Advances by `dt` seconds in fixed steps (at most MAX_STEPS) and writes the mesh, if it has
// one, all in one compute pass. Call it outside render(), before it. Returns the steps taken.
fn int GpuCloth.update(&self, float dt) {
    if (!drie::compute_supported()) return 0;
    make_kernels();
    self.accumulator += math::max(dt, 0.0f);
    int n = (int)(self.accumulator / STEP);
    if (n > MAX_STEPS) {
        n = MAX_STEPS;
        self.accumulator = 0;
    } else {
        self.accumulator -= n * STEP;
    }
    self.sync();
    if (n > 0) self.write_colliders();
    SgView target;
    if (self.mesh) target = self.mesh.geometry.gpu_vertices_target();
    self.bind();
    drie::begin_compute("drie-gpu-cloth");
    for (int s = 0; s < n; s++) self.step_once((float)s / n, (float)(s + 1) / n);
    if (target.id) {
        kernels.write.set_view(shaders::VIEW_GC_WRITE_VERTS, target);
        shaders::GcParams u = self.params(1, 1);
        u.gc_origin[3] = self.accumulator / STEP;
        dispatch(kernels.write, &u, (int)self.count());
    }
    drie::end_compute();
    // (without a step, the colliders' move waits for the next one)
    if (n > 0) {
        foreach (&c : self.colliders) {
            c.from_a = c.a;
            c.from_b = c.b;
            c.from_axes = c.axes;
        }
    }
    return n;
}

// Copies the points (in the world) into `out` (count() of them), waiting for the GPU: for tests
// and tools, not every frame. False if they could not be read.
fn bool GpuCloth.read_points(&self, Vector3[] out) {
    if (!self.pos || out.len < self.count()) return false;
    if (self.read_slot < 0) self.read_slot = drie::readback_slot();
    int slot = self.read_slot;
    if (slot < 0) return false;
    sz n = self.count();
    drie::Vec4[] data = mem::new_array(drie::Vec4, n);
    defer free(data);
    drie::begin_compute("drie-gpu-cloth-read");
    self.pos.track(slot);
    drie::end_compute();
    drie::commit();
    if (!drie::read_back(slot, data.ptr, (sz)n * 16, wait: true)) return false;
    foreach (i, p : data) out[i] = self.origin + (Vector3) { p.x, p.y, p.z };
    return true;
}
