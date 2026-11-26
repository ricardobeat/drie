// GPU compute cloth simulation (`shaders/gpu_cloth.glsl`), mirroring `drie::cloth::Cloth` for
// grids of tens of thousands of vertices. Runs deterministic graph-colored XPBD substeps,
// fixed-point spatial-hash self-collision, multi-panel sheets (`panels()`), two-way dynamic body
// coupling (up to `BODIES` spheres/capsules with fixed-latency readback), and direct GPU vertex
// buffer writes (`BufferGeometry.gpu_vertices`) with zero per-frame vertex readback.
//
//   GpuCloth* c = gpu_cloth::grid(200, 200, { -1.5, 2, -1.5 }, { 3, 0, 0 }, { 0, 0, 3 }, 1.4);
//   c.floor_y = 0;
//   scene.add(c.make_mesh(drie::mesh_standard_material(color: 0xd8d0c0, side: DOUBLE_SIDE)));
//   c.update(delta);  // call outside render(), before drawing
module drie::gpu_cloth;
import std::math, std::collections::list, drie, drie::shaders, sokol;

// Fixed simulation timestep in seconds (60 Hz).
const float STEP    = 1.0f / 60;
// Maximum fixed steps taken in a single `update()` call before dropping excess accumulator time.
const int MAX_STEPS = 4;
// Displacement threshold (m) above which a collider teleports rather than sweeping through cloth.
const float JUMP = 1.0f;
// Sentinel vertex index marking an untethered particle.
const uint NONE  = 0xffffffff;
// Compute workgroup size shared with `shaders/gpu_cloth.glsl`.
const int GROUP  = 128;
// Maximum number of dynamic (mass-bearing) sphere/capsule colliders (`GC_BODIES` in shader).
const int BODIES = 8;
// Number of ring-buffered async readback slots kept in flight for dynamic bodies.
const int RING = 4;
// Fixed update latency between GPU body pushes and CPU `collider_shift()`/`collider_impulse()`.
const int LATENCY = 2;
// History ring length for applied CPU body corrections (larger than max readback latency).
const int LOG = 16;

// Geometric primitive type for a `Collider`.
enum ColliderKind : char {
    SPHERE, // Sphere centered at `a` with `radius`.
    CAPSULE, // Capsule segment `a..b` with `radius`.
    BOX, // Oriented box centered at `a` with half-extents `half` along `axes`.
    PLANE, // Half-space through `a` keeping points on the side facing `axes[0]`.
}

// Swept collision primitive uploaded to the GPU each update, matching `drie::cloth::Collider`.
struct Collider {
    ColliderKind kind; // Primitive shape kind.
    float radius; // Sphere or capsule radius in meters.
    Vector3 from_a, from_b; // Sweep-start endpoints in cloth-local space.
    Vector3 a, b; // Sweep-end endpoints in cloth-local space.
    bool active; // Skipped by collision kernels when false.
    Vector3 half; // Box half-extents along `axes` in meters.
    Vector3[3] from_axes; // Box orthonormal basis at sweep start.
    Vector3[3] axes; // Box orthonormal basis at sweep end (or plane normal in `axes[0]`).
    float inv_mass; // Inverse mass in 1/kg (0 = kinematic).
    int body; // Dynamic body slot in `0..BODIES-1`, or -1 for kinematic colliders.
    bool restart; // Resets GPU body offset and velocity on the next update (`place_collider()`).
    float friction; // Per-collider friction override, or -1 to inherit `GpuCloth.friction`.
    Vector3 shift; // Latest CPU-side position correction read back from the GPU.
    Vector3 dv; // Latest CPU-side velocity delta read back from the GPU.
}

// GPU-driven render mesh tracking a collider's live GPU position (`make_collider_mesh()`).
struct Shape {
    int collider; // Index into `GpuCloth.colliders`.
    Mesh* mesh; // Render mesh whose vertex buffer is rewritten on the GPU each update.
    StorageBuffer* template; // Unshifted local template vertices uploaded on first update.
    int count; // Vertex count in `template` and `mesh`.
    float radius; // Bounding radius of the template geometry around its origin.
}

// Record of dynamic body corrections delivered to the CPU at the end of a past update.
struct Applied {
    long update; // Update index when these corrections were reported (-1 = unused).
    ulong steps; // Cumulative fixed steps completed by that update.
    Vector3[BODIES] off; // Position shifts applied per dynamic body slot.
    Vector3[BODIES] dv; // Velocity changes applied per dynamic body slot.
}

// Contiguous range of same-color links or bend stencils dispatched in a single compute pass.
struct Batch {
    uint first, count; // Start index and element count in `links` or `bends`.
    int kind; // Constraint kind: 0 = stretch, 1 = shear, 2 = distance bend, 3 = signed bend.
}

// Render mesh bound to one panel of a `GpuCloth`, written directly on the GPU by `gc_write_cs`.
struct PanelMesh {
    int panel; // Panel index in `0..panel_count()-1`.
    Mesh* mesh; // Render mesh with `BufferGeometry.gpu_vertices = true`.
    int detail; // Subdivision factor per simulation cell edge (1 = native grid).
    int quiet; // Consecutive updates written while every vertex in the panel was pinned.
}

// Shared compute pipeline instances compiled once for all `GpuCloth` instances.
struct Kernels {
    ComputePass* begin;
    ComputePass* forces;
    ComputePass* integrate;
    ComputePass* links;
    ComputePass* bends;
    ComputePass* tethers;
    ComputePass* clear;
    ComputePass* count;
    ComputePass* scan_block;
    ComputePass* scan_sums;
    ComputePass* scan_add;
    ComputePass* fill;
    ComputePass* repel;
    ComputePass* collide;
    ComputePass* collide_bodies;
    ComputePass* bodies;
    ComputePass* shape;
    ComputePass* damp;
    ComputePass* write;
}

Kernels* kernels @local;

// GPU-simulated XPBD cloth grid (or multi-panel sheet) driven by `shaders/gpu_cloth.glsl`.
struct GpuCloth {
    int cols, rows; // Total simulation grid dimensions across all panels.
    int panel_cols; // Columns per independent panel (`cols` when single-panel).
    Vector3 origin; // World-space anchor matching the first rest vertex.
    Vector3[] rest; // Rest positions relative to `origin`.
    float[] inv_mass; // Per-vertex inverse mass in 1/kg (0 = pinned).
    Vector3[] pin_to; // Per-vertex pin target positions relative to `origin`.
    int[] panel_free; // Free (unpinned) vertex count per panel.
    bool[] panel_changed; // True when a pin change or reset dirtied the panel since last write.
    List{shaders::GcLink} links; // Graph-colored stretch, shear, and distance-bend links.
    List{shaders::GcBend} bends; // Nine-phase signed curvature bend stencils.
    List{Batch} batches; // Dispatch ranges across `links` and `bends`.
    shaders::GcTether[] tether_data; // Nearest-pin tether entries uploaded to `tether_buf`.
    List{Collider} colliders; // Registered collision primitives.
    bool tethers_dirty; // Rebuilds and uploads `tether_buf` on the next `sync()`.
    bool pins_dirty; // Re-uploads `pins` storage buffer on the next `sync()`.
    bool state_dirty; // Resets GPU particle positions, velocities, and bodies on next `sync()`.
    Vector3[] start_from; // Custom reset pose from `reset_to()` (empty resets to `rest`).
    bool has_tethers; // True when at least one free vertex is tethered to a pin.

    int substeps; // XPBD substeps per 1/60 s fixed step (default 8).
    int iterations; // Constraint projection passes per substep (default 1).
    float stretch_compliance; // Stretch link compliance in m/N (0 = rigid).
    float shear_compliance; // Shear link compliance in m/N.
    // Bend compliance on the active model's scale: `DISTANCE` in m/N (default 1e-2, 0.05 in
    // `examples/cloth_furniture.c3`); `SIGNED` per meter of curvature offset (~0.1..20).
    float bend_compliance;
    // Active bend solver (`DISTANCE` 2-hop links or `SIGNED` rest curvature); switchable anytime.
    BendModel bend_model;
    float tether_slack; // Allowed stretch ratio past rest tether distance (0.02 = 2%).
    float damping; // Exponential velocity damping rate per second.
    // Share of neighbor velocity difference damped per fixed step (0..1) to suppress jitter.
    float internal_damping;
    Vector3 gravity; // Constant world acceleration in m/s^2.
    Vector3 wind; // Base air velocity in m/s.
    float gust; // Deterministic spatial-temporal wind gust amplitude.
    float drag; // Aerodynamic normal drag coefficient.
    // Cross-flow lift relative to `drag` (0 disables; ~0.5 makes falling sheets glide).
    float lift;
    float thickness; // Surface separation in meters from colliders and floor.
    // Static/kinetic collider friction in [0, 1] (~0.8 upholstery, ~0.4 wood); also floor slide.
    float friction;
    float floor_y; // World-space Y height of an infinite floor plane (-1e30 disables).
    float mass; // Total mass in kg across all simulated vertices.
    // Minimum self-collision separation in meters (0 disables), clamped to rest spacing.
    float self_distance;
    float self_stiffness; // Fraction of self-collision overlap pushed out per substep (0..1).
    float self_friction; // Tangential friction between colliding cloth layers (0..1).
    int self_every; // Runs spatial-hash self-collision every `self_every` substeps.

    double time; // Total simulated time in seconds.
    float accumulator; // Unstepped frame time remainder in seconds.
    ulong steps; // Total fixed steps simulated so far.
    List{PanelMesh} meshes; // Panel render meshes created by `make_mesh()`.

    StorageBuffer* pos; // Current cloth-local positions (vec4 per vertex).
    StorageBuffer* prev; // Previous-substep positions (vec4 per vertex).
    StorageBuffer* vel; // Per-vertex velocities (vec4 per vertex).
    StorageBuffer* start; // Last fixed-step start positions for frame interpolation.
    StorageBuffer* acc; // Per-vertex gravity and wind accelerations.
    StorageBuffer* rest_buf; // Rest-pose positions uploaded once on sync.
    StorageBuffer* pins; // Per-vertex pin target `.xyz` and inverse mass `.w`.
    StorageBuffer* link_buf; // Graph-colored stretch, shear, and distance-bend links.
    StorageBuffer* bend_buf; // Nine-phase signed curvature bend stencils.
    StorageBuffer* tether_buf; // Nearest-pin tether targets and rest distances.
    StorageBuffer* collider_buf; // Packed active colliders uploaded each update.
    StorageBuffer* cells; // Spatial-hash cell counts and prefix-summed offsets.
    StorageBuffer* sums; // Per-block prefix-sum scratch buffer for spatial hashing.
    StorageBuffer* ids; // Vertex indices sorted into spatial-hash buckets.
    StorageBuffer* push; // Accumulated self-collision displacement per vertex.
    int table; // Spatial-hash bucket count (power of two, >= 2 * vertex count).
    int read_slot; // Readback slot used by synchronous `read_points()` (-1 until allocated).
    bool watching; // Queues non-blocking position readbacks every update when true.
    int[3] watch_slot; // Triple-buffered async readback slots for `seen_points()`.
    ulong[3] watch_update; // Update sequence number associated with each `watch_slot`.
    drie::Vec4[] watched; // CPU staging buffer receiving `seen_points()` vec4 data.

    int body_count; // Number of dynamic mass-bearing colliders allocated (`0..BODIES`).
    StorageBuffer* bodies; // Live GPU offset and velocity delta per dynamic body.
    StorageBuffer* partials; // Per-workgroup dynamic body impulse reduction buffer.
    int[RING] ring_slot; // Ring of async readback slots tracking `bodies`.
    Applied[LOG] applied; // History of body corrections already reported to the CPU.
    int applied_at; // Next write cursor in `applied`.
    long applied_through; // Latest update index whose GPU body state was consumed.
    long[BODIES] body_reset; // Update index at which each dynamic body last restarted.
    long updates; // Total `update()` calls completed.
    List{Shape} shapes; // Collider-tracking meshes created by `make_collider_mesh()`.
}

// Returns the linear vertex index for `(col, row)`.
fn uint GpuCloth.index(&self, int col, int row) @inline => (uint)(row * self.cols + col);
// Returns the total number of simulated vertices across all panels.
fn uint GpuCloth.count(&self) @inline => (uint)(self.cols * self.rows);
// Returns the total number of stretch, shear, and distance-bend links.
fn sz GpuCloth.link_count(&self) => self.links.len();
// Returns the number of signed curvature bend stencils.
fn sz GpuCloth.bend_count(&self) => self.bends.len();
// Returns the number of independent panels in this cloth.
fn int GpuCloth.panel_count(&self) => self.cols / self.panel_cols;
// Whether columns `a` and `b` are both in the cloth and in one panel.
fn bool GpuCloth.joined(&self, int a, int b) => a >= 0 && b < self.cols && a / self.panel_cols == b / self.panel_cols;
// The batches (dispatches per link pass) of the bend model in use.
fn sz GpuCloth.batch_count(&self) {
    sz count;
    foreach (b : self.batches) if (self.solves(b.kind)) count++;
    return count;
}

// Whether batches of `kind` are solved: the bends of the model in use only.
fn bool GpuCloth.solves(&self, int kind) {
    if (kind == 2) return self.bend_model == DISTANCE;
    if (kind == 3) return self.bend_model == SIGNED;
    return true;
}

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

// `count` panels side by side, each a grid as grid() makes, the i-th from origin + apart * i:
// solved as one cloth, so that they collide with each other and share the colliders (a ball
// with a mass pushed by all of them). `mass` is a panel's. Each panel gets its mesh from
// make_mesh(panel: i).
fn GpuCloth* panels(int count, int cols, int rows, Vector3 origin, Vector3 across, Vector3 down, Vector3 apart, float mass = 0.4f) {
    int width = cols * count;
    Vector3[] rest = mem::new_array(Vector3, (sz)width * rows);
    defer free(rest);
    for (int i = 0; i < count; i++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                rest[r * width + i * cols + c] = origin + apart * (float)i + across * ((float)c / (float)(cols - 1)) + down * ((float)r / (float)(rows - 1));
            }
        }
    }
    return from_positions(width, rows, rest, mass * count, panel_cols: cols);
}

// A grid of points at rest where `rest` has them (row by row, `cols` to a row), in panels of
// `panel_cols` columns (0: one panel).
fn GpuCloth* from_positions(int cols, int rows, Vector3[] rest, float mass = 0.4f, int panel_cols = 0) {
    GpuCloth* c = mem::new(GpuCloth);
    c.bend_model = DISTANCE;
    c.cols = cols;
    c.rows = rows;
    c.panel_cols = panel_cols > 0 ? panel_cols : cols;
    sz n = (sz)cols * rows;
    c.rest = mem::new_array(Vector3, n);
    c.inv_mass = mem::new_array(float, n);
    c.pin_to = mem::new_array(Vector3, n);
    c.links.allocator = mem;
    c.bends.allocator = mem;
    c.batches.allocator = mem;
    c.meshes.allocator = mem;
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
    c.watch_slot = { -1, -1, -1 };
    c.shapes.allocator = mem;
    c.applied_through = -1;
    for (int k = 0; k < RING; k++) c.ring_slot[k] = -1;
    for (int k = 0; k < LOG; k++) c.applied[k].update = -1;
    c.origin = rest[0];
    float w = (float)n / math::max(mass, 1e-6f);
    for (sz i = 0; i < n; i++) {
        c.rest[i] = rest[i] - c.origin;
        c.pin_to[i] = c.rest[i];
        c.inv_mass[i] = w;
    }
    c.panel_free = mem::new_array(int, c.cols / c.panel_cols);
    c.panel_changed = mem::new_array(bool, c.cols / c.panel_cols);
    foreach (&f : c.panel_free) *f = c.panel_cols * rows;
    @pool() {
        List{shaders::GcLink}[3] kinds;
        for (int k = 0; k < 3; k++) kinds[k].allocator = tmem;
        for (int r = 0; r < rows; r++) {
            for (int k = 0; k < cols; k++) {
                uint i = c.index(k, r);
                bool across = c.joined(k, k + 1);
                if (across) c.link(&kinds[0], i, c.index(k + 1, r));
                if (r + 1 < rows) c.link(&kinds[0], i, c.index(k, r + 1));
                if (across && r + 1 < rows) {
                    c.link(&kinds[1], i, c.index(k + 1, r + 1));
                    c.link(&kinds[1], c.index(k + 1, r), c.index(k, r + 1));
                }
                if (c.joined(k, k + 2)) c.link(&kinds[2], i, c.index(k + 2, r));
                if (r + 2 < rows) c.link(&kinds[2], i, c.index(k, r + 2));
            }
        }
        for (int k = 0; k < 3; k++) c.add_colored(&kinds[k], k, n);
    };
    // in runs, rows then columns, at every third middle point along the bends and every third
    // point across them, so that bends of a run share no point (as Cloth's)
    for (int phase = 0; phase < 9; phase++) {
        uint first = (uint)c.bends.len();
        for (int r = phase / 3; r < rows; r += 3) {
            for (int k = 1 + phase % 3; k + 1 < cols; k += 3) {
                if (!c.joined(k - 1, k + 1)) continue;
                c.add_bend(c.index(k - 1, r), c.index(k, r), c.index(k + 1, r), c.index(k, math::max(r - 1, 0)), c.index(k, math::min(r + 1, rows - 1)), 1);
            }
        }
        c.add_bend_batch(first);
    }
    for (int phase = 0; phase < 9; phase++) {
        uint first = (uint)c.bends.len();
        for (int r = 1 + phase % 3; r + 1 < rows; r += 3) {
            for (int k = phase / 3; k < cols; k += 3) {
                int left = c.joined(k - 1, k) ? k - 1 : k;
                int right = c.joined(k, k + 1) ? k + 1 : k;
                c.add_bend(c.index(k, r - 1), c.index(k, r), c.index(k, r + 1), c.index(left, r), c.index(right, r), -1);
            }
        }
        c.add_bend_batch(first);
    }
    c.tethers_dirty = true;
    c.pins_dirty = true;
    c.state_dirty = true;
    return c;
}

// Appends a distance link between vertices `a` and `b` using their rest separation.
fn void GpuCloth.link(&self, List{shaders::GcLink}* list, uint a, uint b) {
    list.push({ a, b, (self.rest[a] - self.rest[b]).length(), 0 });
}

// m between a and b, c and d its neighbours the other way: (a + b - 2m) along the normal
// s (b - a) x (d - c) held at the rest shape's.
fn void GpuCloth.add_bend(&self, uint a, uint m, uint b, uint c, uint d, float s) {
    Vector3* p = self.rest.ptr;
    Vector3 n = (p[b] - p[a]).cross(p[d] - p[c]) * s;
    float n2 = n.dot(n);
    float rest = n2 > 1e-24f ? (p[a] + p[b] - p[m] * 2).dot(n) / math::sqrt(n2) : 0;
    self.bends.push({ a, m, b, c, d, s, rest, 0 });
}

// The bends from `first` on, a batch.
fn void GpuCloth.add_bend_batch(&self, uint first) {
    uint count = (uint)self.bends.len() - first;
    if (count) self.batches.push({ first, count, 3 });
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

// Releases all CPU arrays, GPU storage buffers, shape templates, and the `GpuCloth` itself.
fn void GpuCloth.free(&self) {
    free(self.rest);
    free(self.inv_mass);
    free(self.panel_free);
    free(self.panel_changed);
    free(self.pin_to);
    free(self.watched);
    free(self.start_from);
    free(self.tether_data);
    self.links.free();
    self.bends.free();
    self.batches.free();
    self.meshes.free();
    self.colliders.free();
    foreach (sh : self.shapes) {
        if (!sh.template) continue;
        sh.template.dispose();
        free(sh.template);
    }
    self.shapes.free();
    foreach (b : (StorageBuffer*[17]) { self.pos, self.prev, self.vel, self.start, self.acc, self.rest_buf, self.pins,
        self.link_buf, self.bend_buf, self.tether_buf, self.collider_buf, self.cells, self.sums, self.ids, self.push, self.bodies,
        self.partials }) {
        if (!b) continue;
        b.dispose();
        free(b);
    }
    free(self);
}

// Returns the panel index containing vertex `i`.
fn int GpuCloth.panel_of(&self, uint i) @inline => (int)(i % (uint)self.cols) / self.panel_cols;

// Returns true when every vertex of `panel` is pinned (so its mesh stops updating once settled).
fn bool GpuCloth.panel_asleep(&self, int panel) => self.panel_free[panel] == 0;

// Pins vertex `i` at its rest position (reposition later with `move_pin()`).
fn void GpuCloth.pin(&self, uint i) {
    if (self.inv_mass[i] != 0) self.panel_free[self.panel_of(i)]--;
    self.panel_changed[self.panel_of(i)] = true;
    self.inv_mass[i] = 0;
    self.pin_to[i] = self.rest[i];
    self.tethers_dirty = true;
    self.pins_dirty = true;
}

// Pins every vertex along `row` across all panels at its rest position.
fn void GpuCloth.pin_row(&self, int row) {
    for (int k = 0; k < self.cols; k++) self.pin(self.index(k, row));
}

// Restores vertex `i` to a free mass and marks pins and tethers for re-upload.
fn void GpuCloth.unpin(&self, uint i) {
    if (self.inv_mass[i] == 0) self.panel_free[self.panel_of(i)]++;
    self.panel_changed[self.panel_of(i)] = true;
    self.inv_mass[i] = (float)self.count() / math::max(self.mass, 1e-6f);
    self.tethers_dirty = true;
    self.pins_dirty = true;
}

// Moves pinned point `i`: it gets there at the next fixed step.
fn void GpuCloth.move_pin(&self, uint i, Vector3 to) {
    self.pin_to[i] = to - self.origin;
    self.panel_changed[self.panel_of(i)] = true;
    self.pins_dirty = true;
}

// Puts every point back where it is at rest, still, and the bodies where they are moved to.
fn void GpuCloth.reset(&self) {
    self.state_dirty = true;
    foreach (&c : self.panel_changed) *c = true;
    self.accumulator = 0;
    self.time = 0;
    foreach (&r : self.body_reset) *r = self.updates;
    foreach (&k : self.colliders) k.restart = k.body >= 0;
}

// As reset, with the points (in the world, count() of them) where `points` puts them, still;
// their rest shape stays as it was.
fn void GpuCloth.reset_to(&self, Vector3[] points) {
    sz n = self.count();
    if (self.start_from.len != n) {
        free(self.start_from);
        self.start_from = mem::new_array(Vector3, n);
    }
    foreach (i, &p : self.start_from) *p = points[i] - self.origin;
    self.reset();
}

fn float hash01(uint x) @local {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return (float)(x & 0xffffff) / (float)0x1000000;
}

// As reset, with each panel a little off its rest, by `seed` (each seed its own): smooth waves
// of about `ripple` m across it, and the panel moved up to `shift` m sideways, turned up to
// `turn` radians about the vertical and tilted up to `tilt` radians, about its middle. A cloth
// dropped flat then folds differently each time. Pinned points stay where they are held (pass
// shift, turn and tilt 0 for a panel that hangs from them).
fn void GpuCloth.reset_scattered(&self, uint seed, float ripple = 0.03f, float shift = 0.15f, float turn = 0.2f, float tilt = 0.1f) {
    sz n = self.count();
    Vector3[] points = mem::new_array(Vector3, n);
    defer free(points);
    int pc = self.panel_cols > 0 ? self.panel_cols : self.cols;
    for (int panel = 0; panel < self.cols / pc; panel++) {
        uint s = (seed * 7919 + (uint)panel * 104729) * 31;
        float[10] r;
        foreach (k, &v : r) *v = hash01(s + (uint)k) * 2 - 1;
        int c0 = panel * pc;
        Vector3 mid;
        for (int row = 0; row < self.rows; row++) {
            for (int c = 0; c < pc; c++) mid += self.rest[self.index(c0 + c, row)];
        }
        mid = mid * (1.0f / (float)(pc * self.rows));
        Vector3 across = self.rest[self.index(c0 + pc - 1, 0)] - self.rest[self.index(c0, 0)];
        Vector3 down = self.rest[self.index(c0, self.rows - 1)] - self.rest[self.index(c0, 0)];
        Vector3 normal = across.cross(down).normalize();
        float yaw = r[0] * turn;
        float cy = math::cos(yaw);
        float sy = math::sin(yaw);
        float pitch = r[1] * tilt;
        Vector3 move = { r[2] * shift, 0, r[3] * shift };
        for (int row = 0; row < self.rows; row++) {
            float v = (float)row / (float)(self.rows - 1);
            for (int c = 0; c < pc; c++) {
                float u = (float)c / (float)(pc - 1);
                uint i = self.index(c0 + c, row);
                float wave = math::sin(u * 6.5f + r[4] * 3.2f) * math::sin(v * 5.3f + r[5] * 3.2f)
                           + 0.6f * math::sin(u * 11.0f + v * 9.0f + r[6] * 3.2f)
                           + 0.5f * math::sin(v * 13.0f - u * 4.0f + r[7] * 3.2f);
                Vector3 q = self.rest[i] - mid + normal * (wave * ripple * 0.5f);
                q.y += q.x * pitch;
                Vector3 t = { q.x * cy + q.z * sy, q.y, -q.x * sy + q.z * cy };
                points[i] = self.origin + mid + move + t;
            }
        }
    }
    self.reset_to(points);
}

// Registers a sphere collider at world `center` with `radius` and returns its collider ID.
fn int GpuCloth.add_sphere(&self, Vector3 center, float radius) {
    center -= self.origin;
    self.colliders.push({ .kind = SPHERE, .radius = radius, .from_a = center, .from_b = center, .a = center, .b = center, .active = true, .body = -1, .friction = -1 });
    return (int)self.colliders.len() - 1;
}

// Registers a capsule collider spanning world segment `a..b` with `radius` and returns its ID.
fn int GpuCloth.add_capsule(&self, Vector3 a, Vector3 b, float radius) {
    a -= self.origin;
    b -= self.origin;
    self.colliders.push({ .kind = CAPSULE, .radius = radius, .from_a = a, .from_b = b, .a = a, .b = b, .active = true, .body = -1, .friction = -1 });
    return (int)self.colliders.len() - 1;
}

fn Vector3 column(drie::Mat4 m, int i) @local => { m[i][0], m[i][1], m[i][2] };

// A box of half size `half`, placed by `placement` (centered on its origin, its axes along the
// matrix's columns; their length is left out).
fn int GpuCloth.add_box(&self, Vector3 half, drie::Mat4 placement) {
    Collider k = { .kind = BOX, .half = half, .active = true, .body = -1, .friction = -1 };
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
    Collider k = { .kind = PLANE, .a = point, .from_a = point, .active = true, .body = -1, .friction = -1 };
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

// Enables or disables collider `id` without removing it.
fn void GpuCloth.set_collider_active(&self, int id, bool active) => self.colliders[id].active = active;

// Collider `id`'s own friction coefficient (see GpuCloth.friction); -1: the cloth's.
fn void GpuCloth.set_collider_friction(&self, int id, float friction) => self.colliders[id].friction = friction;

// Gives a sphere or capsule a mass (kg; 0: kinematic, it goes where it is moved whatever the
// cloth does): each push then moves the point and the collider apart by their inverse masses,
// at every substep, and the collider goes on at the velocity the pushes left it. After each
// update(), move the body it stands for by collider_shift() and change its velocity by
// -collider_impulse() / its mass. Up to BODIES colliders have a mass; false past them.
fn bool GpuCloth.set_collider_mass(&self, int id, float kg) {
    Collider* k = &self.colliders[id];
    if (k.kind == BOX || k.kind == PLANE) return false;
    if (kg <= 0) {
        k.inv_mass = 0;
        return true;
    }
    if (k.body < 0) {
        if (self.body_count >= BODIES) return false;
        k.body = self.body_count++;
        self.body_reset[k.body] = self.updates;
        k.restart = true;
    }
    k.inv_mass = 1 / kg;
    return true;
}

// How far the cloth has pushed collider `id` (with a mass) off the path it was moved along, as
// far as the CPU has heard in the last update(): add it to the body's place.
fn Vector3 GpuCloth.collider_shift(&self, int id) => self.colliders[id].shift;

// The momentum (kg m/s) the cloth has taken from collider `id` (with a mass), as far as the CPU
// has heard in the last update(): take it off the body (velocity -= impulse / mass).
fn Vector3 GpuCloth.collider_impulse(&self, int id) {
    Collider* k = &self.colliders[id];
    return k.inv_mass > 0 ? -k.dv / k.inv_mass : (Vector3) { 0, 0, 0 };
}

// Puts a sphere or capsule there (b is ignored for a sphere), not swept from where it was; a
// body starts over, still, what the cloth did to it before forgotten.
fn void GpuCloth.place_collider(&self, int id, Vector3 a, Vector3 b = { 0, 0, 0 }) {
    Collider* k = &self.colliders[id];
    a -= self.origin;
    b -= self.origin;
    if (k.kind == SPHERE) b = a;
    k.from_a = a;
    k.from_b = b;
    k.a = a;
    k.b = b;
    if (k.body >= 0) {
        k.restart = true;
        self.body_reset[k.body] = self.updates;
    }
}

// A mesh drawn where collider `id` is: `shape` (its vertices around the collider's center, as
// a sphere_geometry is) moved there on the GPU at every update. A collider with a mass is drawn
// where the cloth has pushed it, before the CPU hears of it, so it meets the cloth on screen.
// Its matrix is the identity.
fn Mesh* GpuCloth.make_collider_mesh(&self, int id, BufferGeometry* shape, Material* material = null) {
    BufferGeometry* g = drie::buffer_geometry();
    g.dynamic = true;
    g.gpu_vertices = true;
    foreach (f : shape.vertices) g.vertices.push(f);
    foreach (i : shape.indices) g.indices.push(i);
    int count = (int)shape.vertex_count();
    float radius = 0;
    for (int i = 0; i < count; i++) radius = math::max(radius, shape.get_position((uint)i).length());
    Mesh* m = drie::mesh(g, material);
    m.matrix_auto_update = false;
    m.matrix_world_auto_update = false;
    m.matrix = drie::mat4_identity();
    m.matrix_world = drie::mat4_identity();
    self.shapes.push({ id, m, null, count, radius });
    return m;
}

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
                if ((dr == 0 && dc == 0) || !self.joined(math::min(col, c2), math::max(col, c2)) || r2 < 0 || r2 >= self.rows) continue;
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

// A mesh of the cloth (of panel `panel` of a cloth made by panels()), drawn where the points
// are (its matrix is the identity: add it anywhere in the scene), its vertices written on the
// GPU. Use a DOUBLE_SIDE material to see both faces. Its uvs are in meters along the cloth at
// rest, as Cloth's. Its bounding sphere (for culling and shadows) is centered on the panel at
// rest, as wide as its diagonal. With `detail` above 1 it is finer than the points, `detail`
// vertices to a cell's side, on a smooth surface through them (as Cloth's from_fine, without
// simulating more): round folds and edges from a coarse grid.
fn Mesh* GpuCloth.make_mesh(&self, Material* material = null, int panel = 0, int detail = 1) {
    int cols = self.panel_cols;
    int rows = self.rows;
    int first = panel * cols;
    detail = math::max(detail, 1);
    BufferGeometry* g = drie::buffer_geometry();
    g.dynamic = true;
    g.gpu_vertices = true;
    Vector2[] uv = mem::new_array(Vector2, (sz)cols * rows);
    defer free(uv);
    float[] down = mem::new_array(float, cols);
    defer free(down);
    Vector3 lo = { 1e30f, 1e30f, 1e30f };
    Vector3 hi = -lo;
    for (int r = 0; r < rows; r++) {
        float across = 0;
        for (int k = 0; k < cols; k++) {
            uint i = self.index(first + k, r);
            if (k > 0) across += (self.rest[i] - self.rest[i - 1]).length();
            if (r > 0) down[k] += (self.rest[i] - self.rest[i - (uint)self.cols]).length();
            Vector3 p = self.origin + self.rest[i];
            lo = math::min(lo, p);
            hi = math::max(hi, p);
            uv[r * cols + k] = { across, -down[k] };
        }
    }
    int fw = (cols - 1) * detail + 1;
    int fh = (rows - 1) * detail + 1;
    for (int fr = 0; fr < fh; fr++) {
        int r0 = math::min(fr / detail, rows - 2);
        float tv = (float)(fr - r0 * detail) / (float)detail;
        for (int fk = 0; fk < fw; fk++) {
            int k0 = math::min(fk / detail, cols - 2);
            float tu = (float)(fk - k0 * detail) / (float)detail;
            int a = r0 * cols + k0;
            Vector2 top = uv[a] + (uv[a + 1] - uv[a]) * tu;
            Vector2 bottom = uv[a + cols] + (uv[a + cols + 1] - uv[a + cols]) * tu;
            Vector3 pa = self.rest[self.index(first + k0, r0)];
            Vector3 pb = self.rest[self.index(first + k0 + 1, r0)];
            Vector3 pc = self.rest[self.index(first + k0, r0 + 1)];
            Vector3 pd = self.rest[self.index(first + k0 + 1, r0 + 1)];
            Vector3 p = self.origin + (pa + (pb - pa) * tu) * (1 - tv) + (pc + (pd - pc) * tu) * tv;
            g.add_vertex(p, { 0, 0, 1 }, top + (bottom - top) * tv);
        }
    }
    for (int r = 0; r + 1 < fh; r++) {
        for (int k = 0; k + 1 < fw; k++) {
            uint a = (uint)(r * fw + k);
            uint b = a + (uint)fw;
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
    self.meshes.push({ panel, m, detail, 0 });
    return m;
}

// The sphere the meshes (of every panel) are culled and shadowed with, in the world.
fn void GpuCloth.set_bounds(&self, Vector3 center, float radius) {
    foreach (pm : self.meshes) {
        pm.mesh.geometry.bounds_center = center;
        pm.mesh.geometry.bounds_radius = radius;
    }
}

fn void make_kernels() @local {
    if (kernels) return;
    SgBackend be = sg::query_backend();
    Kernels* k = mem::new(Kernels);
    k.begin = drie::compute_pass(shaders::gc_begin_shader_desc(be), "gpu-cloth-begin");
    k.forces = drie::compute_pass(shaders::gc_forces_shader_desc(be), "gpu-cloth-forces");
    k.integrate = drie::compute_pass(shaders::gc_integrate_shader_desc(be), "gpu-cloth-integrate");
    k.links = drie::compute_pass(shaders::gc_links_shader_desc(be), "gpu-cloth-links");
    k.bends = drie::compute_pass(shaders::gc_bends_shader_desc(be), "gpu-cloth-bends");
    k.tethers = drie::compute_pass(shaders::gc_tethers_shader_desc(be), "gpu-cloth-tethers");
    k.clear = drie::compute_pass(shaders::gc_clear_shader_desc(be), "gpu-cloth-clear");
    k.count = drie::compute_pass(shaders::gc_count_shader_desc(be), "gpu-cloth-count");
    k.scan_block = drie::compute_pass(shaders::gc_scan_block_shader_desc(be), "gpu-cloth-scan-block");
    k.scan_sums = drie::compute_pass(shaders::gc_scan_sums_shader_desc(be), "gpu-cloth-scan-sums");
    k.scan_add = drie::compute_pass(shaders::gc_scan_add_shader_desc(be), "gpu-cloth-scan-add");
    k.fill = drie::compute_pass(shaders::gc_fill_shader_desc(be), "gpu-cloth-fill");
    k.repel = drie::compute_pass(shaders::gc_self_shader_desc(be), "gpu-cloth-self");
    k.collide = drie::compute_pass(shaders::gc_collide_shader_desc(be), "gpu-cloth-collide");
    k.collide_bodies = drie::compute_pass(shaders::gc_collide_bodies_shader_desc(be), "gpu-cloth-collide-bodies");
    k.bodies = drie::compute_pass(shaders::gc_bodies_shader_desc(be), "gpu-cloth-bodies");
    k.shape = drie::compute_pass(shaders::gc_shape_shader_desc(be), "gpu-cloth-shape");
    k.damp = drie::compute_pass(shaders::gc_damp_shader_desc(be), "gpu-cloth-damp");
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
        self.bend_buf = drie::storage_buffer(32, label: "gpu-cloth-bends");
        if (self.bends.len()) self.bend_buf.write(self.bends.entries, (sz)self.bends.len() * shaders::GcBend::size);
        self.tether_buf = drie::storage_buffer((sz)n * shaders::GcTether::size, label: "gpu-cloth-tethers");
        self.collider_buf = drie::storage_buffer(shaders::GcCollider::size, cpu_write: true, label: "gpu-cloth-colliders");
        self.table = 1024;
        while (self.table < (int)n * 2) self.table *= 2;
        // (a cell past the table, and the scan's last block whole)
        self.cells = drie::storage_buffer((sz)(self.table + 1024) * 4, label: "gpu-cloth-cells");
        self.sums = drie::storage_buffer(1024 * 4, label: "gpu-cloth-sums");
        self.ids = drie::storage_buffer((sz)n * 4, label: "gpu-cloth-ids");
        self.push = drie::storage_buffer(vec4s, label: "gpu-cloth-push");
        self.bodies = drie::storage_buffer(BODIES * shaders::GcBody::size, label: "gpu-cloth-bodies");
        self.partials = drie::storage_buffer((sz)((n + GROUP - 1) / GROUP) * BODIES * 16, label: "gpu-cloth-partials");
    }
    if (self.state_dirty) {
        self.state_dirty = false;
        drie::Vec4[] data = mem::new_array(drie::Vec4, n);
        defer free(data);
        foreach (i, p : self.rest) data[i] = { p.x, p.y, p.z, 0 };
        self.rest_buf.write(data.ptr, vec4s);
        foreach (i, &d : data) {
            Vector3 p = self.inv_mass[i] == 0 ? self.pin_to[i] : self.start_from.len == n ? self.start_from[i] : self.rest[i];
            *d = { p.x, p.y, p.z, 0 };
        }
        self.pos.write(data.ptr, vec4s);
        self.prev.write(data.ptr, vec4s);
        self.start.write(data.ptr, vec4s);
        foreach (&d : data) *d = { 0, 0, 0, 0 };
        self.vel.write(data.ptr, vec4s);
        self.acc.write(data.ptr, vec4s);
        shaders::GcBody[BODIES] still;
        self.bodies.write(&still, BODIES * shaders::GcBody::size);
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

// Packs and uploads the current `colliders` array to the GPU `collider_buf` storage buffer.
fn void GpuCloth.write_colliders(&self) {
    sz count = self.colliders.len();
    if (!count) return;
    shaders::GcCollider[] data = mem::new_array(shaders::GcCollider, count);
    defer free(data);
    foreach (i, k : self.colliders) {
        data[i] = {
            .info = { (float)(int)k.kind, k.radius, k.active ? 1 : 0, (float)k.body },
            .mass = { k.inv_mass, k.restart ? 1 : 0, k.friction, 0 },
            .corr_off = v4(k.shift),
            .corr_dv = v4(k.dv),
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

// Binds this cloth's storage buffers to every shared compute kernel before dispatching.
fn void GpuCloth.bind(&self) {
    Kernels* k = kernels;
    k.begin.set_buffer(shaders::VIEW_GC_BEGIN_POS, self.pos);
    k.begin.set_buffer(shaders::VIEW_GC_BEGIN_START, self.start);
    k.begin.set_buffer(shaders::VIEW_GC_BEGIN_VEL, self.vel);
    k.begin.set_buffer(shaders::VIEW_GC_BEGIN_PINS, self.pins);
    k.forces.set_buffer(shaders::VIEW_GC_FORCES_POS, self.pos);
    k.forces.set_buffer(shaders::VIEW_GC_FORCES_VEL, self.vel);
    k.forces.set_buffer(shaders::VIEW_GC_FORCES_ACC, self.acc);
    k.forces.set_buffer(shaders::VIEW_GC_FORCES_PINS, self.pins);
    k.integrate.set_buffer(shaders::VIEW_GC_INT_POS, self.pos);
    k.integrate.set_buffer(shaders::VIEW_GC_INT_PREV, self.prev);
    k.integrate.set_buffer(shaders::VIEW_GC_INT_VEL, self.vel);
    k.integrate.set_buffer(shaders::VIEW_GC_INT_ACC, self.acc);
    k.integrate.set_buffer(shaders::VIEW_GC_INT_PINS, self.pins);
    k.links.set_buffer(shaders::VIEW_GC_LINKS_POS, self.pos);
    k.links.set_buffer(shaders::VIEW_GC_LINKS_LINKS, self.link_buf);
    k.links.set_buffer(shaders::VIEW_GC_LINKS_PINS, self.pins);
    k.bends.set_buffer(shaders::VIEW_GC_BENDS_POS, self.pos);
    k.bends.set_buffer(shaders::VIEW_GC_BENDS_BENDS, self.bend_buf);
    k.bends.set_buffer(shaders::VIEW_GC_BENDS_PINS, self.pins);
    k.tethers.set_buffer(shaders::VIEW_GC_TETH_POS, self.pos);
    k.tethers.set_buffer(shaders::VIEW_GC_TETH_TETHERS, self.tether_buf);
    k.tethers.set_buffer(shaders::VIEW_GC_TETH_PINS, self.pins);
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
    k.collide_bodies.set_buffer(shaders::VIEW_GC_COLLB_POS, self.pos);
    k.collide_bodies.set_buffer(shaders::VIEW_GC_COLLB_PREV, self.prev);
    k.collide_bodies.set_buffer(shaders::VIEW_GC_COLLB_VEL, self.vel);
    k.collide_bodies.set_buffer(shaders::VIEW_GC_COLLB_PINS, self.pins);
    k.collide_bodies.set_buffer(shaders::VIEW_GC_COLLB_PUSH, self.push);
    k.collide_bodies.set_buffer(shaders::VIEW_GC_COLLB_COLLIDERS, self.collider_buf);
    k.collide_bodies.set_buffer(shaders::VIEW_GC_COLLB_BODIES, self.bodies);
    k.collide_bodies.set_buffer(shaders::VIEW_GC_COLLB_PARTIALS, self.partials);
    k.bodies.set_buffer(shaders::VIEW_GC_BOD_COLLIDERS, self.collider_buf);
    k.bodies.set_buffer(shaders::VIEW_GC_BOD_BODIES, self.bodies);
    k.bodies.set_buffer(shaders::VIEW_GC_BOD_PARTIALS, self.partials);
    k.shape.set_buffer(shaders::VIEW_GC_SHAPE_COLLIDERS, self.collider_buf);
    k.shape.set_buffer(shaders::VIEW_GC_SHAPE_BODIES, self.bodies);
    k.damp.set_buffer(shaders::VIEW_GC_DAMP_VEL, self.vel);
    k.damp.set_buffer(shaders::VIEW_GC_DAMP_LINKS, self.link_buf);
    k.damp.set_buffer(shaders::VIEW_GC_DAMP_PINS, self.pins);
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
        .gc_grid = { self.panel_cols, 0, 1, 0 },
        .gc_mode = { 0, self.lift, 0, 0 },
    };
}

fn void dispatch(ComputePass* k, shaders::GcParams* u, int threads, int group = GROUP) @local {
    k.set_uniforms(shaders::UB_GC_PARAMS, u, shaders::GcParams::size);
    k.dispatch((threads + group - 1) / group);
}

// gc_bodies_cs in `mode`, a thread per collider.
fn void GpuCloth.dispatch_bodies(&self, int mode, shaders::GcParams* params) {
    shaders::GcParams u = *params;
    u.gc_mode[3] = (float)mode;
    dispatch(kernels.bodies, &u, (int)self.colliders.len(), 32);
}

// What the CPU learns of the bodies at the start of an update: the state at the end of the
// update LATENCY before (read back, waiting if the GPU has not got there: it rarely has not, a
// frame in flight being waited for anyway), carried on to the end of this one (`until` steps)
// and less what the CPU was told to apply since, each carried on the same way. The colliders'
// shift and dv are set to it; the GPU takes it off at this update's end. A fixed latency, not
// whatever has arrived: the same calls give the same result whatever the GPU's pace.
fn void GpuCloth.hear_bodies(&self, ulong until) {
    foreach (&k : self.colliders) {
        k.shift = { 0, 0, 0 };
        k.dv = { 0, 0, 0 };
    }
    if (!self.body_count) return;
    long heard = self.updates - LATENCY;
    if (heard < 0 || heard <= self.applied_through) return;
    int slot = self.ring_slot[(int)(heard % RING)];
    shaders::GcBody[BODIES] newest;
    if (slot < 0 || !drie::read_back(slot, &newest, (sz)shaders::GcBody[BODIES]::size, wait: true)) return;
    // (a readback copies the buffer when the GPU gets to it: the mark says which update's end)
    if ((long)newest[0].mark.x != heard) return;
    self.applied_through = heard;
    ulong heard_steps = (ulong)newest[0].mark.y;
    foreach (&k : self.colliders) {
        int b = k.body;
        if (b < 0 || heard < self.body_reset[b]) continue;
        Vector3 off = { newest[b].off.x, newest[b].off.y, newest[b].off.z };
        Vector3 dv = { newest[b].dv.x, newest[b].dv.y, newest[b].dv.z };
        off += dv * ((float)(until - heard_steps) * STEP);
        foreach (&a : self.applied) {
            if (a.update <= heard || a.update < self.body_reset[b]) continue;
            off -= a.off[b] + a.dv[b] * ((float)(until - a.steps) * STEP);
            dv -= a.dv[b];
        }
        k.shift = off;
        k.dv = dv;
    }
}

// One fixed step, its colliders going from `from` to `until` (0..1) of their way.
fn void GpuCloth.step_once(&self, float from, float until) {
    Kernels* k = kernels;
    int n = (int)self.count();
    shaders::GcParams u = self.params(from, from);
    dispatch(k.begin, &u, n);
    dispatch(k.forces, &u, n);
    if (self.body_count) self.dispatch_bodies(1, &u);
    float h = STEP / self.substeps;
    float[4] alpha = { self.stretch_compliance / (h * h), self.shear_compliance / (h * h), self.bend_compliance / (h * h), self.bend_compliance / (h * h) };
    int blocks = (self.table + 1 + 1023) / 1024;
    for (int s = 0; s < self.substeps; s++) {
        float t_prev = from + (until - from) * (float)s / self.substeps;
        float t = from + (until - from) * (float)(s + 1) / self.substeps;
        u = self.params(t, t_prev);
        dispatch(k.integrate, &u, n);
        for (int it = 0; it < math::max(self.iterations, 1); it++) {
            foreach (b : self.batches) {
                if (!self.solves(b.kind)) continue;
                u.gc_range[0] = (int)b.first;
                u.gc_range[1] = (int)b.count;
                u.gc_step[2] = alpha[b.kind];
                dispatch(b.kind == 3 ? k.bends : k.links, &u, (int)b.count);
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
        if (self.body_count) {
            dispatch(k.collide_bodies, &u, n);
            self.dispatch_bodies(0, &u);
        } else {
            dispatch(k.collide, &u, n);
        }
    }
    if (self.internal_damping > 0) {
        u = self.params(until, until);
        u.gc_mode[0] = self.internal_damping;
        foreach (b : self.batches) {
            if (b.kind != 0) continue;
            u.gc_range[0] = (int)b.first;
            u.gc_range[1] = (int)b.count;
            dispatch(k.damp, &u, (int)b.count);
        }
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
    ulong until = self.steps + (ulong)n;
    self.sync();
    // (made here, not in make_collider_mesh: before a window opens there is no GPU to make them on)
    foreach (&sh : self.shapes) {
        if (sh.template) continue;
        BufferGeometry* g = sh.mesh.geometry;
        sh.template = drie::storage_buffer(16, label: "gpu-cloth-shape");
        sh.template.write(g.vertices.entries, (sz)g.vertices.len() * float::size);
    }
    self.hear_bodies(until);
    if (n > 0 || self.body_count) self.write_colliders();
    self.bind();
    drie::begin_compute("drie-gpu-cloth");
    shaders::GcParams u = self.params(1, 1);
    if (self.body_count) self.dispatch_bodies(3, &u);
    for (int s = 0; s < n; s++) self.step_once((float)s / n, (float)(s + 1) / n);
    u = self.params(1, 1);
    u.gc_origin[3] = self.accumulator / STEP;
    foreach (&pm : self.meshes) {
        // (an asleep panel's points reach their pins at the next step, and the mesh, drawn
        // between the last two, there the update after: written twice more after a change)
        if (self.panel_changed[pm.panel] || !self.panel_asleep(pm.panel)) pm.quiet = 0;
        if (pm.quiet >= 2) continue;
        if (n > 0) pm.quiet++;
        SgView target = pm.mesh.geometry.gpu_vertices_target();
        if (!target.id) continue;
        kernels.write.set_view(shaders::VIEW_GC_WRITE_VERTS, target);
        shaders::GcParams wu = u;
        wu.gc_grid[1] = pm.panel * self.panel_cols;
        wu.gc_grid[2] = pm.detail;
        dispatch(kernels.write, &wu, ((self.panel_cols - 1) * pm.detail + 1) * ((self.rows - 1) * pm.detail + 1));
    }
    foreach (&c : self.panel_changed) *c = false;
    foreach (sh : self.shapes) {
        Collider* c = &self.colliders[sh.collider];
        BufferGeometry* g = sh.mesh.geometry;
        g.bounds_center = self.origin + c.a;
        g.bounds_radius = sh.radius + JUMP;
        SgView v = g.gpu_vertices_target();
        if (!v.id) continue;
        kernels.shape.set_buffer(shaders::VIEW_GC_SHAPE_TEMPLATE, sh.template);
        kernels.shape.set_view(shaders::VIEW_GC_SHAPE_VERTS, v);
        shaders::GcParams su = u;
        su.gc_range[0] = sh.collider;
        su.gc_range[1] = sh.count;
        dispatch(kernels.shape, &su, sh.count);
    }
    if (self.body_count) {
        shaders::GcParams mark = u;
        mark.gc_mode = { (float)self.updates, (float)until, 0, 0 };
        self.dispatch_bodies(2, &mark);
        int k = (int)(self.updates % RING);
        if (self.ring_slot[k] < 0) self.ring_slot[k] = drie::readback_slot();
        if (self.ring_slot[k] >= 0) self.bodies.track(self.ring_slot[k]);
        Applied* a = &self.applied[self.applied_at];
        self.applied_at = (self.applied_at + 1) % LOG;
        a.update = self.updates;
        a.steps = until;
        foreach (&c : self.colliders) {
            if (c.body < 0) continue;
            a.off[c.body] = c.shift;
            a.dv[c.body] = c.dv;
        }
    }
    if (self.watching) {
        int k = (int)(self.updates % 3);
        if (self.watch_slot[k] < 0) self.watch_slot[k] = drie::readback_slot();
        if (self.watch_slot[k] >= 0 && self.pos.track(self.watch_slot[k])) self.watch_update[k] = (ulong)self.updates + 1;
    }
    drie::end_compute();
    // (without a step, the colliders' move waits for the next one; a body's from where the CPU
    // will put it)
    foreach (&c : self.colliders) {
        if (n > 0) {
            c.from_a = c.a;
            c.from_b = c.b;
            c.from_axes = c.axes;
        }
        c.from_a += c.shift;
        c.from_b += c.shift;
        c.restart = false;
    }
    self.updates++;
    return n;
}

// Reads the points back at every update from now on, without waiting for the GPU: seen_points
// gives them a frame or two late. (A copy of the points each frame: for a few of them on the
// CPU, as a handle at a corner.)
fn void GpuCloth.watch_points(&self) => self.watching = true;

// The points (in the world, count() of them) as the last readback that has arrived left them;
// false while none has.
fn bool GpuCloth.seen_points(&self, Vector3[] out) {
    if (out.len < self.count()) return false;
    sz n = self.count();
    if (self.watched.len != n) {
        free(self.watched);
        self.watched = mem::new_array(drie::Vec4, n);
    }
    int[3] order = { 0, 1, 2 };
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 3; j++) {
            if (self.watch_update[order[j]] > self.watch_update[order[i]]) {
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
        }
    }
    foreach (k : order) {
        if (self.watch_slot[k] < 0 || !self.watch_update[k]) continue;
        if (!drie::read_back(self.watch_slot[k], self.watched.ptr, (sz)n * 16)) continue;
        foreach (i, p : self.watched) out[i] = self.origin + (Vector3) { p.x, p.y, p.z };
        return true;
    }
    return false;
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
