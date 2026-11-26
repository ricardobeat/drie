// CPU cloth simulation solved with small-step XPBD (Macklin et al., 2019). Each 1/60 s fixed step
// splits into `substeps` enforcing stretch, shear, bend (`SIGNED` curvature or `DISTANCE`),
// pin tethers, spatial-hash self-collision, and swept sphere/capsule/box/plane colliders with
// optional two-way impulse coupling. Points are stored relative to `origin` for precision far from
// world zero, and every run is deterministic.
//
//   Cloth* c = cloth::grid(24, 36, { -0.6, 2.4, 0 }, { 1.2, 0, 0 }, { 0, -2.4, 0 });
//   c.pin_row(0);
//   scene.add(c.make_mesh(drie::mesh_standard_material(color: 0x8a1010, side: DOUBLE_SIDE)));
//   c.update(delta);
module drie::cloth;
import std::math, std::collections::list, std::thread, std::atomic, std::os, std::time, drie;

// Fixed simulation timestep in seconds (60 Hz).
const float STEP       = 1.0f / 60;
// Maximum fixed steps taken in a single `update()` call before dropping excess accumulator time.
const int MAX_STEPS    = 4;
const float FAR @local = 1e30f;
// Displacement threshold (m) above which a collider teleports rather than sweeping through cloth.
const float JUMP       = 1.0f;
// Minimum point displacement (0.1 mm) required before `write_mesh()` re-uploads vertex data.
const float STILL      = 1e-4f;

// Geometric primitive type for a `Collider`.
enum ColliderKind : char {
    SPHERE,  // Sphere centered at `a` with `radius`.
    CAPSULE, // Capsule segment `a..b` with `radius`.
    BOX,     // Oriented box centered at `a` with half-extents `half` along `axes`.
    PLANE,   // Half-space through `a` keeping points on the side facing `axes[0]`.
}

// Swept collision primitive (sphere, capsule, oriented box, or plane) with optional dynamic mass.
struct Collider {
    ColliderKind kind; // Primitive shape kind.
    float radius; // Sphere or capsule radius in meters.
    Vector3 from_a, from_b; // Sweep-start endpoints in cloth-local space.
    Vector3 a, b; // Sweep-end endpoints in cloth-local space.
    bool active; // Skipped during collision when false.
    Vector3 half; // Box half-extents along `axes` in meters.
    Vector3[3] from_axes; // Box orthonormal basis at the start of the sweep.
    Vector3[3] axes; // Box orthonormal basis at the end of the sweep (or plane normal in `axes[0]`).
    // Momentum (kg m/s) transferred from this collider to the cloth in the last `update()`.
    Vector3 impulse;
    // Inverse mass in 1/kg; 0 marks a kinematic collider unaffected by cloth pushback.
    float inv_mass;
    Vector3 shift; // Cumulative displacement applied to a dynamic sphere/capsule by the cloth.
    Vector3 dv; // Cumulative velocity delta applied to a dynamic sphere/capsule by the cloth.
}

// Distance constraint between two cloth vertices with precomputed inverse-mass weights.
struct Link {
    uint a, b; // Endpoint vertex indices.
    float rest; // Rest distance in meters.
    // Precomputed per-endpoint correction shares: `inv_mass / (w_a + w_b + alpha)`.
    float ka, kb;
}

// Signed curvature bend constraint over point `m` between `a` and `b` along a row or column.
// Holds `(a + b - 2m)` projected onto `s * (b - a) x (d - c)` at `rest`, where `c` and `d` are
// `m`'s orthogonal neighbors (`m` itself on borders) and `s` is +1 along rows or -1 down columns.
struct Bend {
    uint a, m, b, c, d; // Stencil vertex indices.
    float s; // Normal orientation sign (+1 for rows, -1 for columns).
    float rest; // Signed rest curvature offset in meters.
    float alpha; // Compliance scaled by `1 / h^2`.
}

// Long-range tether keeping a free vertex within geodesic rest distance of its nearest pin.
struct Tether {
    uint point, pin; // Free vertex index and its nearest pinned vertex index.
    float rest; // Geodesic rest distance along the cloth grid in meters.
}

// Point-mass cloth grid simulated on the CPU with XPBD constraints, wind, and swept colliders.
struct Cloth {
    int cols, rows; // Simulation grid dimensions.
    // World-space anchor equal to the first rest point; all particle arrays are stored relative to
    // `origin` to retain float precision far from world zero. Use `point()` for world coordinates.
    Vector3 origin;
    Vector3[] pos; // Current cloth-local vertex positions.
    Vector3[] prev; // Cloth-local positions at the start of the current substep.
    Vector3[] vel; // Per-vertex velocities in m/s.
    Vector3[] start; // Cloth-local positions at the start of the last fixed step.
    Vector3[] accel; // Per-vertex gravity and aerodynamic accelerations in m/s^2.
    Vector3[] rest; // Rest-pose positions relative to `origin`.
    float[] inv_mass; // Per-vertex inverse mass in 1/kg (0 = pinned).
    Vector3[] pin_to; // Target cloth-local positions for pinned vertices.
    float[] gust_cols; // Precomputed column trig terms combined by `gust_at()`.
    float[] gust_rows; // Precomputed row trig terms combined by `gust_at()`.
    List{Link} stretch; // Graph-colored horizontal and vertical structural links.
    List{Link} shear; // Graph-colored diagonal shear links.
    List{Bend} bend; // Nine-phase signed curvature bend constraints.
    List{Link} bend_links; // Graph-colored two-step distance links used by `BendModel.DISTANCE`.
    List{Tether} tethers; // Geodesic tethers from free vertices to nearest pins.
    bool tethers_dirty; // Rebuilds `tethers` on the next fixed step when true.
    // Scaled compliances (`alpha`) cached in link weights; -1 forces recomputation.
    float[3] links_alpha;
    List{Collider} colliders; // Registered collision primitives.

    int substeps; // XPBD substeps per 1/60 s fixed step (default 8).
    float stretch_compliance; // Stretch link compliance in m/N (0 = rigid).
    float shear_compliance; // Diagonal shear link compliance in m/N.
    // Bend compliance on the active model's scale: `SIGNED` ~0.1..20, `DISTANCE` ~1e-3..1e-1.
    float bend_compliance;
    // Bend solver model: `SIGNED` preserves pleated rest curvature; `DISTANCE` uses 2-hop links.
    BendModel bend_model;
    float tether_slack; // Allowed stretch ratio past rest tether length (0.02 = 2%).
    float damping; // Exponential velocity damping rate per second (0..).
    // Fraction of neighbor velocity difference damped per fixed step (0..1) to quiet jitter.
    float internal_damping;
    Vector3 gravity; // Constant world acceleration in m/s^2.
    Vector3 wind; // Base air velocity in m/s.
    float gust; // Wind gust variation amplitude (0 = steady, 1 = 0..2x wind).
    float drag; // Aerodynamic drag coefficient per m^2 per m/s of normal airflow.
    float thickness; // Surface offset in meters maintained from colliders and the floor.
    float friction; // Collider tangential friction in [0, 1] (0 = slide, 1 = stick).
    float floor_y; // World-space Y height of an infinite floor plane (-1e30 disables).
    float floor_friction; // Tangential friction on the floor plane in [0, 1].
    // Minimum self-collision separation in meters (0 disables), clamped to rest distance.
    float self_distance;
    int self_every; // Runs `collide_self()` every `self_every` substeps.
    // Spatial-hash scratch buffers for `collide_self()`.
    uint[] self_bucket;
    uint[] self_order;
    uint[] self_start;
    // Frame-start world-space snapshot and overlapping cloths for `update_all()` cloth-cloth push.
    Vector3[] snap;
    uint[] snap_bucket;
    uint[] snap_order;
    uint[] snap_start;
    uint snap_mask;
    List{Cloth*} touching;
    float sleep_speed; // Speed threshold in m/s below which the cloth can sleep (0 disables).
    float sleep_time; // Consecutive seconds below `sleep_speed` required before sleeping.
    float still_for; // Consecutive seconds spent below `sleep_speed`.
    bool asleep; // True while simulation is paused until woken by a collider or moved pin.
    // Cloth-local AABB padded by `thickness`, updated each substep for coarse collider culling.
    Vector3 lo, hi;
    float mass; // Total cloth mass in kg distributed uniformly across vertices.

    double time; // Total simulated time in seconds.
    float accumulator; // Leftover unstepped frame delta time in seconds.
    ulong steps; // Total fixed steps simulated so far.
    Mesh* mesh; // Render mesh created by `make_mesh()`, or null.
    Vector3[] shown; // Interpolated cloth-local positions written to `mesh` this frame.
    // Optional high-resolution skinning grid from `from_fine()`; null renders the coarse grid.
    Detail* detail;
    Vector3[] marked; // Cloth-local positions captured by the last `mark()` call.
    Vector3[] drawn; // Positions last uploaded to `mesh`, used for `STILL` skip checks.
}

// High-resolution display grid skinned onto a coarser simulated `Cloth` via `from_fine()`.
// Stores each fine vertex's coarse cell coordinates and rest-frame offset so fine folds ride on
// the coarse bilinear patches.
struct Detail {
    int cols, rows; // Fine mesh dimensions.
    Vector3[] rest; // Fine rest positions in world space.
    int[] col_cell; // Coarse column cell index per fine column.
    float[] col_t; // Interpolation parameter in [0, 1] across each coarse column cell.
    int[] row_cell; // Coarse row cell index per fine row.
    float[] row_s; // Interpolation parameter in [0, 1] down each coarse row cell.
    Vector3[] offset; // Rest position offset expressed in the coarse patch's `(u, v, n)` frame.
    Vector3[] normal; // Rest surface normal expressed in the coarse patch's `(u, v, n)` frame.
}

// Returns the linear vertex index for `(col, row)`.
fn uint Cloth.index(&self, int col, int row) @inline => (uint)(row * self.cols + col);
// Returns the total number of simulated vertices (`cols * rows`).
fn uint Cloth.count(&self) @inline => (uint)(self.cols * self.rows);

// Returns vertex `i`'s current world-space position.
fn Vector3 Cloth.point(&self, uint i) => self.origin + self.pos[i];

// A flat grid of `cols` x `rows` points from `origin`, `across` to the last column and `down`
// to the last row. Its front (the mesh's front faces) is cross(across, -down).
fn Cloth* grid(int cols, int rows, Vector3 origin, Vector3 across, Vector3 down, float mass = 0.4f) {
    Vector3[] rest = mem::new_array(Vector3, (sz)cols * rows);
    defer free(rest);
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            Vector3 at = origin + across * ((float)c / (float)(cols - 1)) + down * ((float)r / (float)(rows - 1));
            rest[r * cols + c] = at;
        }
    }
    return from_positions(cols, rows, rest, mass);
}

// A grid of points at rest where `rest` has them (row by row, `cols` to a row): the rest
// lengths, and so the folds, are the rest shape's. `mass` is the whole cloth's, in kg.
fn Cloth* from_positions(int cols, int rows, Vector3[] rest, float mass = 0.4f) {
    Cloth* c = mem::new(Cloth);
    c.cols = cols;
    c.rows = rows;
    sz n = (sz)cols * rows;
    c.pos = mem::new_array(Vector3, n);
    c.prev = mem::new_array(Vector3, n);
    c.vel = mem::new_array(Vector3, n);
    c.start = mem::new_array(Vector3, n);
    c.accel = mem::new_array(Vector3, n);
    c.rest = mem::new_array(Vector3, n);
    c.inv_mass = mem::new_array(float, n);
    c.pin_to = mem::new_array(Vector3, n);
    c.stretch.allocator = mem;
    c.shear.allocator = mem;
    c.bend.allocator = mem;
    c.bend_links.allocator = mem;
    c.tethers.allocator = mem;
    c.colliders.allocator = mem;
    c.touching.allocator = mem;
    c.substeps = 8;
    c.stretch_compliance = 0;
    c.shear_compliance = 1e-3f;
    c.bend_compliance = 1e-1f;
    c.tether_slack = 0.02f;
    c.damping = 0.6f;
    c.gravity = { 0, -9.81f, 0 };
    c.drag = 0.6f;
    c.thickness = 0.02f;
    c.friction = 0.3f;
    c.floor_y = -FAR;
    c.floor_friction = 0.3f;
    c.self_every = 2;
    c.sleep_time = 0.5f;
    c.lo = { -FAR, -FAR, -FAR };
    c.hi = { FAR, FAR, FAR };
    c.mass = mass;
    c.links_alpha = { -1, -1, -1 };
    c.origin = rest[0];
    float w = (float)n / math::max(mass, 1e-6f);
    for (sz i = 0; i < n; i++) {
        Vector3 at = rest[i] - c.origin;
        c.pos[i] = at;
        c.rest[i] = at;
        c.prev[i] = at;
        c.start[i] = at;
        c.pin_to[i] = at;
        c.inv_mass[i] = w;
    }
    for (int r = 0; r < rows; r++) {
        for (int k = 0; k < cols; k++) {
            uint i = c.index(k, r);
            if (k + 1 < cols) c.link(&c.stretch, i, c.index(k + 1, r));
            if (r + 1 < rows) c.link(&c.stretch, i, c.index(k, r + 1));
            if (k + 1 < cols && r + 1 < rows) {
                c.link(&c.shear, i, c.index(k + 1, r + 1));
                c.link(&c.shear, c.index(k + 1, r), c.index(k, r + 1));
            }
            if (k + 2 < cols) c.link(&c.bend_links, i, c.index(k + 2, r));
            if (r + 2 < rows) c.link(&c.bend_links, i, c.index(k, r + 2));
        }
    }
    // in runs, rows then columns, at every third middle point along the bends and every third
    // point across them, so that bends next to each other share no point (see color_links)
    for (int phase = 0; phase < 9; phase++) {
        for (int r = phase / 3; r < rows; r += 3) {
            for (int k = 1 + phase % 3; k + 1 < cols; k += 3) {
                c.add_bend(
                    c.index(k - 1, r),
                    c.index(k, r),
                    c.index(k + 1, r),
                    c.index(k, math::max(r - 1, 0)),
                    c.index(k, math::min(r + 1, rows - 1)),
                    1
                );
            }
        }
    }
    for (int phase = 0; phase < 9; phase++) {
        for (int r = 1 + phase % 3; r + 1 < rows; r += 3) {
            for (int k = phase / 3; k < cols; k += 3) {
                c.add_bend(
                    c.index(k, r - 1),
                    c.index(k, r),
                    c.index(k, r + 1),
                    c.index(math::max(k - 1, 0), r),
                    c.index(math::min(k + 1, cols - 1), r),
                    -1
                );
            }
        }
    }
    color_links(&c.stretch, n);
    color_links(&c.shear, n);
    color_links(&c.bend_links, n);
    return c;
}

// A cloth simulated on every `step_cols`-th column and `step_rows`-th row (and the last ones)
// of a `cols` x `rows` grid at rest at `rest`, and drawn with the whole grid (make_mesh): the
// fine folds of the rest shape, at the cost of the coarse grid.
fn Cloth* from_fine(int cols, int rows, Vector3[] rest, int step_cols, int step_rows, float mass = 0.4f) {
    int[] cc = coarse_lines(cols, step_cols);
    int[] rr = coarse_lines(rows, step_rows);
    defer {
        free(cc);
        free(rr);
    }
    Vector3[] coarse = mem::new_array(Vector3, (sz)cc.len * rr.len);
    defer free(coarse);
    foreach (r, fr : rr) {
        foreach (c, fc : cc) coarse[r * cc.len + c] = rest[fr * cols + fc];
    }
    Cloth* cl = from_positions((int)cc.len, (int)rr.len, coarse, mass);
    Detail* d = mem::new(Detail);
    d.cols = cols;
    d.rows = rows;
    d.rest = mem::new_array(Vector3, (sz)cols * rows);
    d.rest[..] = rest[..];
    d.col_cell = mem::new_array(int, cols);
    d.col_t = mem::new_array(float, cols);
    d.row_cell = mem::new_array(int, rows);
    d.row_s = mem::new_array(float, rows);
    place_lines(cc, d.col_cell, d.col_t);
    place_lines(rr, d.row_cell, d.row_s);
    d.offset = mem::new_array(Vector3, (sz)cols * rows);
    d.normal = mem::new_array(Vector3, (sz)cols * rows);
    cl.detail = d;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            Frame f   = cl.frame_at(cl.pos, d.col_cell[c], d.col_t[c], d.row_cell[r], d.row_s[r]);
            Vector3 o = rest[r * cols + c] - cl.origin - f.at;
            Vector3 n = grid_normal(rest, cols, rows, c, r);
            d.offset[r * cols + c] = { o.dot(f.u), o.dot(f.v), o.dot(f.n) };
            d.normal[r * cols + c] = { n.dot(f.u), n.dot(f.v), n.dot(f.n) };
        }
    }
    return cl;
}

// The normal of a grid of points at (c, r), from its neighbours across and down.
fn Vector3 grid_normal(Vector3[] p, int cols, int rows, int c, int r) @local {
    Vector3 du = p[r * cols + math::min(c + 1, cols - 1)] - p[r * cols + math::max(c - 1, 0)];
    Vector3 dv = p[math::max(r - 1, 0) * cols + c] - p[math::min(r + 1, rows - 1) * cols + c];
    Vector3 n  = du.cross(dv);
    float len  = n.length();
    return len > 1e-12f ? n / len : (Vector3) { 0, 0, 1 };
}

// The fine lines (columns or rows) the coarse grid keeps: every `step`-th, and the last.
fn int[] coarse_lines(int count, int step) @local {
    step = math::max(step, 1);
    int n       = (count - 1 + step - 1) / step + 1;
    int[] lines = mem::new_array(int, n);
    for (int k = 0; k < n; k++) lines[k] = math::min(k * step, count - 1);
    return lines;
}

// For each fine line: the coarse cell it is in and how far across it.
fn void place_lines(int[] lines, int[] cell, float[] t) @local {
    int k = 0;
    foreach (i, &c : cell) {
        while ((usz)(k + 2) < lines.len && lines[k + 1] <= (int)i) k++;
        *c = k;
        t[i] = (float)((int)i - lines[k]) / (float)(lines[k + 1] - lines[k]);
    }
}

// Orthonormal surface frame evaluated on a coarse cloth cell's bilinear patch.
struct Frame {
    Vector3 at, u, v, n; // Patch point, across tangent `u`, orthogonal tangent `v`, and outward normal `n`.
}

// The bilinear patch of cell (c, r) of the points `p` at (t, s): the place and an orthonormal
// frame, u across, n out of the cloth (the side make_mesh's normals face) and v = n x u.
fn Frame Cloth.frame_at(&self, Vector3[] p, int c, float t, int r, float s) {
    Vector3 p00    = p[self.index(c, r)];
    Vector3 p10    = p[self.index(c + 1, r)];
    Vector3 p01    = p[self.index(c, r + 1)];
    Vector3 p11    = p[self.index(c + 1, r + 1)];
    Vector3 top    = p00 + (p10 - p00) * t;
    Vector3 bottom = p01 + (p11 - p01) * t;
    Vector3 du     = (p10 - p00) * (1 - s) + (p11 - p01) * s;
    Vector3 dv     = top - bottom;
    Vector3 n      = du.cross(dv);
    float nl       = n.length();
    float ul       = du.length();
    Frame f;
    f.at = top + (bottom - top) * s;
    f.u = ul > 1e-9f ? du / ul : (Vector3) { 1, 0, 0 };
    f.n = nl > 1e-12f ? n / nl : (Vector3) { 0, 0, 1 };
    f.v = f.n.cross(f.u);
    return f;
}

// Orders the links so that those next to each other share no point: each gets the first
// color its two points' other links don't have, and they go color by color. (A link waits
// for the one before it only when they share a point, so the CPU can work on several at once.)
fn void color_links(List{Link}* links, sz points) @local {
    uint[] used   = mem::new_array(uint, points);
    char[] color  = mem::new_array(char, links.len());
    Link[] sorted = mem::new_array(Link, links.len());
    defer {
        free(used);
        free(color);
        free(sorted);
    }
    int colors;
    foreach (i, l : *links) {
        uint taken = used[l.a] | used[l.b];
        int k;
        while (k < 31 && (taken & (1u << k))) k++;
        color[i] = (char)k;
        used[l.a] |= 1u << k;
        used[l.b] |= 1u << k;
        colors = math::max(colors, k + 1);
    }
    sz at;
    for (int k = 0; k < colors; k++) {
        foreach (i, l : *links) {
            if (color[i] == k) sorted[at++] = l;
        }
    }
    foreach (i, l : sorted) (*links)[i] = l;
}

// Appends a distance link between vertices `a` and `b` using their current rest separation.
fn void Cloth.link(&self, List{Link}* list, uint a, uint b) {
    list.push({ a, b, (self.pos[a] - self.pos[b]).length(), 0, 0 });
}

// Appends a signed curvature bend stencil over `(a, m, b)` with cross neighbors `(c, d)` and sign `s`.
fn void Cloth.add_bend(&self, uint a, uint m, uint b, uint c, uint d, float s) {
    Vector3* p = self.pos.ptr;
    Vector3 n  = (p[b] - p[a]).cross(p[d] - p[c]) * s;
    float n2   = n.dot(n);
    float rest = n2 > 1e-24f ? (p[a] + p[b] - p[m] * 2).dot(n) / math::sqrt(n2) : 0;
    self.bend.push({ a, m, b, c, d, s, rest, 0 });
}

// Releases all particle arrays, constraint lists, detail skinning data, and the `Cloth` itself.
fn void Cloth.free(&self) {
    free(self.pos);
    free(self.prev);
    free(self.vel);
    free(self.start);
    free(self.accel);
    free(self.rest);
    free(self.inv_mass);
    free(self.pin_to);
    free(self.gust_cols);
    free(self.gust_rows);
    self.stretch.free();
    self.shear.free();
    self.bend.free();
    self.bend_links.free();
    self.tethers.free();
    self.colliders.free();
    free(self.shown);
    free(self.marked);
    free(self.drawn);
    free(self.self_bucket);
    free(self.self_order);
    free(self.self_start);
    free(self.snap);
    free(self.snap_bucket);
    free(self.snap_order);
    free(self.snap_start);
    self.touching.free();
    if (self.detail) {
        Detail* d = self.detail;
        free(d.rest);
        free(d.col_cell);
        free(d.col_t);
        free(d.row_cell);
        free(d.row_s);
        free(d.offset);
        free(d.normal);
        free(d);
    }
    free(self);
}

// Holds point `i` where it is (move it with move_pin).
fn void Cloth.pin(&self, uint i) {
    self.inv_mass[i] = 0;
    self.pin_to[i] = self.pos[i];
    self.vel[i] = { 0, 0, 0 };
    self.tethers_dirty = true;
    self.links_alpha = { -1, -1, -1 };
}

// Pins every vertex along `row` at its current position.
fn void Cloth.pin_row(&self, int row) {
    for (int k = 0; k < self.cols; k++) self.pin(self.index(k, row));
}

// Restores vertex `i` to a free mass and wakes the cloth.
fn void Cloth.unpin(&self, uint i) {
    self.inv_mass[i] = (float)self.count() / math::max(self.mass, 1e-6f);
    self.tethers_dirty = true;
    self.links_alpha = { -1, -1, -1 };
    self.wake();
}

// Moves pinned vertex `i` to world position `to` on the next fixed step and wakes the cloth.
fn void Cloth.move_pin(&self, uint i, Vector3 to) {
    Vector3 at = to - self.origin;
    if (at != self.pin_to[i]) self.wake();
    self.pin_to[i] = at;
}

// Clears sleep state and resets the stillness timer so `update()` resumes stepping.
fn void Cloth.wake(&self) {
    self.asleep = false;
    self.still_for = 0;
}

// Registers a sphere collider at world `center` with `radius` and returns its collider ID.
fn int Cloth.add_sphere(&self, Vector3 center, float radius) {
    center -= self.origin;
    self.colliders.push(
        {
            .kind = SPHERE,
            .radius = radius,
            .from_a = center,
            .from_b = center,
            .a = center,
            .b = center,
            .active = true
        }
    );
    return (int)self.colliders.len() - 1;
}

// Registers a capsule collider spanning world segment `a..b` with `radius` and returns its ID.
fn int Cloth.add_capsule(&self, Vector3 a, Vector3 b, float radius) {
    a -= self.origin;
    b -= self.origin;
    self.colliders.push(
        { .kind = CAPSULE, .radius = radius, .from_a = a, .from_b = b, .a = a, .b = b, .active = true }
    );
    return (int)self.colliders.len() - 1;
}

// Moves a collider (b is ignored for a sphere). The cloth sees it travel there over the steps
// of the next update; from inactive, or further than JUMP, it appears there.
fn void Cloth.move_collider(&self, int id, Vector3 a, Vector3 b = { 0, 0, 0 }) {
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

fn Vector3 column(drie::Mat4 m, int i) @local => { m[i][0], m[i][1], m[i][2] };

// A box of half size `half`, placed by `placement` (centered on its origin, its axes along the
// matrix's columns; their length is left out). Move it with move_box.
fn int Cloth.add_box(&self, Vector3 half, drie::Mat4 placement) {
    Collider k = { .kind = BOX, .half = half, .active = true };
    k.a = column(placement, 3) - self.origin;
    for (int i = 0; i < 3; i++) k.axes[i] = column(placement, i).normalize();
    k.from_a = k.a;
    k.from_axes = k.axes;
    self.colliders.push(k);
    return (int)self.colliders.len() - 1;
}

// A plane through `point`: the points stay on the side `normal` faces (thickness away).
fn int Cloth.add_plane(&self, Vector3 point, Vector3 normal) {
    point -= self.origin;
    Collider k = { .kind = PLANE, .a = point, .from_a = point, .active = true };
    k.axes[0] = normal.normalize();
    k.from_axes = k.axes;
    self.colliders.push(k);
    return (int)self.colliders.len() - 1;
}

// Moves (and turns) a box. As with move_collider, the cloth sees it travel there over the next
// update's steps; from inactive, further than JUMP or turned more than 60 degrees, it appears
// there.
fn void Cloth.move_box(&self, int id, drie::Mat4 placement) {
    Collider* k = &self.colliders[id];
    Vector3 a   = column(placement, 3) - self.origin;
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

// The momentum (kg m/s, in the world) the cloth took from collider `id` in the last update():
// the sum over its pushes of a point's mass times how far it moved the point over the
// substep's length. Two-way coupling: take it off the body the collider stands for (its
// velocity -= impulse / its mass), and the cloth slows a ball, catches it or throws it back.
fn Vector3 Cloth.collider_impulse(&self, int id) => self.colliders[id].impulse;

// Gives a sphere or capsule a mass (kg; 0, the default: kinematic, it goes where it is moved
// whatever the cloth does). Each push then moves the point and the collider apart by their
// inverse masses, at every substep, and the collider goes on at the velocity the pushes left
// it, so a fast ball meeting a taut sheet is stopped there, not a frame later when it is
// through. After update(), the body moves by collider_shift() and its velocity changes by
// -collider_impulse() / mass.
fn void Cloth.set_collider_mass(&self, int id, float kg) => self.colliders[id].inv_mass = kg > 0 ? 1 / kg : 0;

// How far the cloth pushed collider `id` (with a mass) off the path move_collider() gave it in
// the last update(); where the cloth left it is where it was moved plus this.
fn Vector3 Cloth.collider_shift(&self, int id) => self.colliders[id].shift;

// Enables or disables collider `id` without removing it from the list.
fn void Cloth.set_collider_active(&self, int id, bool active) => self.colliders[id].active = active;

// Advances by `dt` seconds in fixed steps (at most MAX_STEPS) and writes the mesh, if it has
// one. Returns the steps taken.
fn int Cloth.update(&self, float dt) {
    foreach (&k : self.colliders) {
        k.impulse = { };
        k.shift = { };
        k.dv = { };
    }
    if (self.asleep) {
        foreach (&k : self.colliders) {
            if (k.active && self.near(k)) self.wake();
        }
    }
    if (self.asleep) {
        // (no sweep from where the colliders were when it wakes)
        foreach (&k : self.colliders) {
            k.from_a = k.a;
            k.from_b = k.b;
            k.from_axes = k.axes;
        }
        self.accumulator = 0;
        return 0;
    }
    self.accumulator += math::max(dt, 0.0f);
    int n = (int)(self.accumulator / STEP);
    if (n > MAX_STEPS) {
        n = MAX_STEPS;
        self.accumulator = 0;
    } else {
        self.accumulator -= n * STEP;
    }
    for (int s = 0; s < n; s++) self.step_once((float)s / n, (float)(s + 1) / n);
    // (without a step, the colliders' move waits for the next one)
    if (n > 0) {
        foreach (&k : self.colliders) {
            k.from_a = k.a + k.shift;
            k.from_b = k.b + k.shift;
            k.from_axes = k.axes;
        }
    }
    if (self.mesh) self.write_mesh(self.accumulator / STEP);
    if (self.sleep_speed > 0 && n > 0) {
        // (how far each point went over the last step, not its velocity: a point the links and
        // the self collisions push back and forth within a step goes nowhere)
        float top = 0;
        foreach (i, at : self.pos) {
            Vector3 d = at - self.start[i];
            top = math::max(top, d.x * d.x + d.y * d.y + d.z * d.z);
        }
        float reach = self.sleep_speed * STEP;
        self.still_for = top < reach * reach ? self.still_for + n * STEP : 0;
        if (self.still_for >= self.sleep_time) self.asleep = true;
    }
    return n;
}

// Whether collider `k` (where it was and where it is going) is in the cloth's bounds.
fn bool Cloth.near(&self, Collider* k) {
    if (k.kind == BOX || k.kind == PLANE) return true;
    Vector3 r  = { k.radius, k.radius, k.radius };
    Vector3 lo = math::min(math::min(k.a, k.b), math::min(k.from_a, k.from_b)) - r;
    Vector3 hi = math::max(math::max(k.a, k.b), math::max(k.from_a, k.from_b)) + r;
    return lo.x <= self.hi.x && lo.y <= self.hi.y && lo.z <= self.hi.z && hi.x >= self.lo.x && hi.y >= self.lo.y
        && hi.z >= self.lo.z;
}

struct UpdateJob @local {
    Cloth* cloth;
    float dt;
    Atomic{int}* left;
}

ThreadPool update_pool @local;
bool update_pool_ready @local;

fn void update_job(void* arg) @local {
    UpdateJob* job = arg;
    job.cloth.update(job.dt);
    job.left.sub(1);
}

// update() on each of `cloths`, the cloths shared out between worker threads (the caller's
// among them) and all done when it returns. They must not share points, colliders or a mesh.
fn void update_all(Cloth*[] cloths, float dt) {
    if (cloths.len == 0) return;
    // the cloths with self_distance set keep apart from each other too: each sees where the
    // others' points were as the frame began, in a hash built here
    foreach (c : cloths) {
        c.touching.clear();
        if (c.self_distance <= 0) continue;
        if (c.snap.len != (sz)c.count()) {
            free(c.snap);
            c.snap = mem::new_array(Vector3, (sz)c.count());
        }
        foreach (i, at : c.pos) c.snap[i] = c.origin + at;
        c.snap_mask = build_hash(c.snap.ptr, c.count(), c.self_distance, &c.snap_bucket, &c.snap_order, &c.snap_start);
    }
    foreach (i, a : cloths) {
        if (a.self_distance <= 0) continue;
        foreach (j, b : cloths) {
            if (i == j || b.self_distance <= 0) continue;
            float d     = math::max(a.self_distance, b.self_distance);
            Vector3 alo = a.origin + a.lo;
            Vector3 ahi = a.origin + a.hi;
            Vector3 blo = b.origin + b.lo;
            Vector3 bhi = b.origin + b.hi;
            if (alo.x > bhi.x + d || alo.y > bhi.y + d || alo.z > bhi.z + d) continue;
            if (blo.x > ahi.x + d || blo.y > ahi.y + d || blo.z > ahi.z + d) continue;
            a.touching.push(b);
            // (a cloth that comes down on a sleeping one wakes it)
            if (a.asleep && !b.asleep) a.wake();
        }
    }
    if (cloths.len == 1) {
        cloths[0].update(dt);
        return;
    }
    if (!update_pool_ready) {
        int workers = math::clamp(os::num_cpu() - 1, 1, 15);
        if (catch update_pool.init(mem, workers, keep_alive: (Duration)0)) {
            foreach (c : cloths) c.update(dt);
            return;
        }
        update_pool_ready = true;
    }
    Atomic{int} left;
    left.store((int)cloths.len - 1);
    UpdateJob[] jobs = mem::new_array(UpdateJob, cloths.len - 1);
    defer free(jobs);
    for (sz i = 0; i + 1 < cloths.len; i++) {
        jobs[i] = { cloths[i], dt, &left };
        if (catch update_pool.add_task(&update_job, &jobs[i])) update_job(&jobs[i]);
    }
    cloths[cloths.len - 1].update(dt);
    while (left.load() > 0) thread::yield();
}

// One fixed step. The colliders go from `from` to `until` (0..1) of their way from where they
// were to where they are going.
fn void Cloth.step_once(&self, float from = 0, float until = 1) {
    if (self.tethers_dirty) self.build_tethers();
    uint n       = self.count();
    Vector3* p   = self.pos.ptr;
    Vector3* q   = self.prev.ptr;
    Vector3* v   = self.vel.ptr;
    Vector3* acc = self.accel.ptr;
    float* w     = self.inv_mass.ptr;
    for (uint i = 0; i < n; i++) {
        self.start[i] = p[i];
        if (w[i] == 0) p[i] = self.pin_to[i];
    }
    self.compute_forces();
    float h         = STEP / self.substeps;
    float keep      = math::exp(-self.damping * h) / h;
    float step_frac = 1.0f / self.substeps;
    float a         = 1 / (h * h);
    float[3] alpha  = { self.stretch_compliance * a, self.shear_compliance * a, self.bend_compliance * a };
    if (alpha != self.links_alpha) {
        self.links_alpha = alpha;
        prepare_links(&self.stretch, w, alpha[0]);
        prepare_links(&self.shear, w, alpha[1]);
        prepare_bends(&self.bend, alpha[2]);
        prepare_links(&self.bend_links, w, alpha[2]);
    }
    for (int s = 0; s < self.substeps; s++) {
        for (uint i = 0; i < n; i++) {
            q[i] = p[i];
            if (w[i] == 0) continue;
            v[i] += acc[i] * h;
            p[i] += v[i] * h;
        }
        solve_links(p, &self.stretch);
        solve_links(p, &self.shear);
        if (self.bend_model == DISTANCE) {
            solve_links(p, &self.bend_links);
        } else {
            solve_bends(p, w, &self.bend);
        }
        self.solve_tethers();
        if (self.self_distance > 0 && (s + 1) % math::max(self.self_every, 1) == 0) self.collide_self();
        self.collide(from + (until - from) * (float)(s + 1) * step_frac, from + (until - from) * (float)s * step_frac);
        for (uint i = 0; i < n; i++) {
            if (w[i] != 0) v[i] = (p[i] - q[i]) * keep;
        }
    }
    if (self.internal_damping > 0) damp_links(v, &self.stretch, self.internal_damping);
    self.time += STEP;
    self.steps++;
}

fn uint self_hash(int x, int y, int z, uint mask) @local @inline {
    return (((uint)x * 73856093u) ^ ((uint)y * 19349663u) ^ ((uint)z * 83492791u)) & mask;
}

// Keeps the cloth's points `self_distance` apart, or as far as they are along the cloth at rest
// when that is less (neighbours along it are left to the links): the points are sorted into a
// spatial hash of cells that size, and each is pushed away from the points in the 27 cells
// round it, by their inverse masses. Only points: a fold thinner than self_distance that
// crosses a cell of the cloth between them in one substep goes through.
fn void Cloth.collide_self(&self) {
    uint n      = self.count();
    Vector3* p  = self.pos.ptr;
    Vector3* r  = self.rest.ptr;
    float* w    = self.inv_mass.ptr;
    float d     = self.self_distance;
    float inv_d = 1 / d;
    uint mask   = build_hash(p, n, d, &self.self_bucket, &self.self_order, &self.self_start);
    uint* order = self.self_order.ptr;
    uint* start = self.self_start.ptr;
    float d2    = d * d;
    // each pair once: a point's own bucket (the points after it there) and the 13 cells on
    // one side of it
    for (uint i = 0; i < n; i++) {
        int cx = (int)math::floor(p[i].x * inv_d);
        int cy = (int)math::floor(p[i].y * inv_d);
        int cz = (int)math::floor(p[i].z * inv_d);
        for (int m = 13; m < 27; m++) {
            uint b = self_hash(cx + m % 3 - 1, cy + (m / 3) % 3 - 1, cz + m / 9 - 1, mask);
            for (uint k = start[b]; k < start[b + 1]; k++) {
                uint j = order[k];
                if (m == 13 && j <= i) continue;
                float wsum = w[i] + w[j];
                if (wsum == 0) continue;
                Vector3 e = p[i] - p[j];
                float e2  = e.x * e.x + e.y * e.y + e.z * e.z;
                if (e2 >= d2 || e2 < 1e-12f) continue;
                Vector3 f   = r[i] - r[j];
                float f2    = f.x * f.x + f.y * f.y + f.z * f.z;
                float min_d = f2 < d2 ? math::sqrt(f2) : d;
                if (e2 >= min_d * min_d) continue;
                float len = math::sqrt(e2);
                Vector3 c = e * ((min_d - len) / (len * wsum));
                p[i] += c * w[i];
                p[j] -= c * w[j];
            }
        }
    }
    foreach (other : self.touching) self.collide_other(other);
}

// Sorts the n points p into a spatial hash of cells `d` across: each point's bucket, the points
// in bucket order, and where each bucket starts in it (the arrays grown as needed). Returns
// the mask of its buckets.
fn uint build_hash(Vector3* p, uint n, float d, uint[]* bucket_arr, uint[]* order_arr, uint[]* start_arr) @local {
    uint size = 1;
    while (size < n * 2) size <<= 1;
    uint mask = size - 1;
    if (bucket_arr.len != (sz)n) {
        free(*bucket_arr);
        free(*order_arr);
        *bucket_arr = mem::new_array(uint, (sz)n);
        *order_arr = mem::new_array(uint, (sz)n);
    }
    if (start_arr.len != (sz)size + 1) {
        free(*start_arr);
        *start_arr = mem::new_array(uint, (sz)size + 1);
    }
    uint* bucket = bucket_arr.ptr;
    uint* order  = order_arr.ptr;
    uint* start  = start_arr.ptr;
    float inv_d  = 1 / d;
    for (uint k = 0; k <= size; k++) start[k] = 0;
    for (uint i = 0; i < n; i++) {
        uint b = self_hash(
            (int)math::floor(p[i].x * inv_d),
            (int)math::floor(p[i].y * inv_d),
            (int)math::floor(p[i].z * inv_d),
            mask
        );
        bucket[i] = b;
        start[b + 1]++;
    }
    for (uint k = 0; k < size; k++) start[k + 1] += start[k];
    for (uint i = 0; i < n; i++) order[start[bucket[i]]++] = i;
    for (uint k = size; k > 0; k--) start[k] = start[k - 1];
    start[0] = 0;
    return mask;
}

// Pushes the points away from `other`'s points as they were when the frame began (its
// snapshot), to self_distance: each cloth moves its own points half the way, and the other
// does the same from its side in its own update.
fn void Cloth.collide_other(&self, Cloth* other) {
    uint n       = self.count();
    Vector3* p   = self.pos.ptr;
    float* w     = self.inv_mass.ptr;
    float d      = self.self_distance;
    float d2     = d * d;
    float inv_d  = 1 / d;
    Vector3* o   = other.snap.ptr;
    uint* order  = other.snap_order.ptr;
    uint* start  = other.snap_start.ptr;
    uint mask    = other.snap_mask;
    // (the snapshot is hashed in cells of the other's self_distance: look as far as this one's)
    float od     = other.self_distance;
    int reach    = (int)math::ceil(d / od);
    float inv_od = 1 / od;
    for (uint i = 0; i < n; i++) {
        if (w[i] == 0) continue;
        Vector3 at = self.origin + p[i];
        if (at.x < other.lo.x + other.origin.x - d || at.y < other.lo.y + other.origin.y - d
            || at.z < other.lo.z + other.origin.z - d) continue;
        if (at.x > other.hi.x + other.origin.x + d || at.y > other.hi.y + other.origin.y + d
            || at.z > other.hi.z + other.origin.z + d) continue;
        int cx = (int)math::floor(at.x * inv_od);
        int cy = (int)math::floor(at.y * inv_od);
        int cz = (int)math::floor(at.z * inv_od);
        for (int dz = -reach; dz <= reach; dz++) {
            for (int dy = -reach; dy <= reach; dy++) {
                for (int dx = -reach; dx <= reach; dx++) {
                    uint b = self_hash(cx + dx, cy + dy, cz + dz, mask);
                    for (uint k = start[b]; k < start[b + 1]; k++) {
                        Vector3 e = at - o[order[k]];
                        float e2  = e.x * e.x + e.y * e.y + e.z * e.z;
                        if (e2 >= d2 || e2 < 1e-12f) continue;
                        float len = math::sqrt(e2);
                        Vector3 c = e * ((d - len) * 0.5f / len);
                        p[i] += c;
                        at += c;
                    }
                }
            }
        }
    }
}

// Takes `share` of the difference in velocity off each pair of linked points, split by their
// inverse masses (a stretch link's ka and kb: they add up to 1, its compliance being 0).
fn void damp_links(Vector3* v, List{Link}* links, float share) @local {
    foreach (&l : *links) {
        Vector3 dv = (v[l.b] - v[l.a]) * share;
        v[l.a] += dv * l.ka;
        v[l.b] -= dv * l.kb;
    }
}

fn void prepare_links(List{Link}* links, float* w, float alpha) @local {
    foreach (&l : *links) {
        float sum = w[l.a] + w[l.b] + alpha;
        l.ka = sum > 0 ? w[l.a] / sum : 0;
        l.kb = sum > 0 ? w[l.b] / sum : 0;
    }
}

// One pass of XPBD over the links, the Lagrange multipliers starting at 0 (one iteration a
// substep): each end moves by its share of how much the link is off its rest length.
fn void solve_links(Vector3* p, List{Link}* links) @local {
    Link* l  = links.entries;
    sz count = links.len();
    for (sz i = 0; i < count; i++, l++) {
        Vector3 d = p[l.a] - p[l.b];
        float d2  = d.dot(d);
        if (d2 < 1e-18f) continue;
        Vector3 c = d * (1 - l.rest / math::sqrt(d2));
        p[l.a] -= c * l.ka;
        p[l.b] += c * l.kb;
    }
}

fn void prepare_bends(List{Bend}* bends, float alpha) @local {
    foreach (&b : *bends) b.alpha = alpha;
}

macro float dot(Vector3 x, Vector3 y) @local => x.x * y.x + x.y * y.y + x.z * y.z;

// One pass of XPBD over the bends. The gradient includes how the normal rotates as a, b, c, and d
// move so the correction stays conservative (fixing the normal does spurious work on swings).
// Every vertex gradient lies along the normal: scalars are `1 - ge` (a), `1 + ge` (b), `-2` (m),
// `-gf` (c), and `+gf` (d).
fn void solve_bends(Vector3* p, float* w, List{Bend}* bends) @local {
    Bend* b  = bends.entries;
    sz count = bends.len();
    for (sz i = 0; i < count; i++, b++) {
        Vector3 e  = p[b.b] - p[b.a];
        Vector3 f  = p[b.d] - p[b.c];
        Vector3 nn = (e.yzx * f.zxy - e.zxy * f.yzx) * b.s;
        float n2   = dot(nn, nn);
        if (n2 < 1e-24f) continue;
        float inv2 =  1 / n2;
        Vector3 n  =  nn * math::sqrt(inv2);
        Vector3 dv =  p[b.a] + p[b.b] - p[b.m] * 2;
        float c    =  dot(dv, n) - b.rest;
        float ef   =  dot(e, f);
        float de   =  dot(dv, e);
        float df   =  dot(dv, f);
        float ge   =  (df * ef - de * dot(f, f)) * inv2;
        float gf   =  (de * ef - df * dot(e, e)) * inv2;
        float ga   =  1 - ge;
        float gb   =  1 + ge;
        float gm   = -2;
        float gc   = -gf;
        // (at an edge c or d is m)
        if (b.c == b.m) gm += gc;
        if (b.d == b.m) gm += gf;
        float sum = w[b.a] * ga * ga + w[b.b] * gb * gb + w[b.m] * gm * gm + b.alpha;
        if (b.c != b.m) sum += w[b.c] * gc * gc;
        if (b.d != b.m) sum += w[b.d] * gf * gf;
        if (sum <= 0) continue;
        Vector3 l = n * (-c / sum);
        p[b.a] += l * (ga * w[b.a]);
        p[b.b] += l * (gb * w[b.b]);
        p[b.m] += l * (gm * w[b.m]);
        if (b.c != b.m) p[b.c] += l * (gc * w[b.c]);
        if (b.d != b.m) p[b.d] += l * (gf * w[b.d]);
    }
}

// Pulls each tethered vertex back within `rest * (1 + tether_slack)` of its nearest pin.
fn void Cloth.solve_tethers(&self) {
    Vector3* p  = self.pos.ptr;
    Tether* t   = self.tethers.entries;
    sz count    = self.tethers.len();
    float slack = 1 + self.tether_slack;
    for (sz i = 0; i < count; i++, t++) {
        Vector3 d  = p[t.point] - p[t.pin];
        float max  = t.rest * slack;
        float len2 = d.dot(d);
        if (len2 <= max * max) continue;
        float len = math::sqrt(len2);
        p[t.point] -= d * ((len - max) / len);
    }
}

// Each free point's nearest pin along the cloth at rest (Dijkstra over the grid's neighbours,
// diagonals too), and how far it is.
fn void Cloth.build_tethers(&self) {
    self.tethers_dirty = false;
    self.tethers.clear();
    uint n       = self.count();
    float[] dist = mem::new_array(float, (sz)n);
    uint[] from  = mem::new_array(uint, (sz)n);
    bool[] done  = mem::new_array(bool, (sz)n);
    defer {
        free(dist);
        free(from);
        free(done);
    }
    bool pinned;
    for (uint i = 0; i < n; i++) {
        dist[i] = self.inv_mass[i] == 0 ? 0 : FAR;
        from[i] = i;
        pinned = pinned || self.inv_mass[i] == 0;
    }
    if (!pinned) return;
    for (uint iter = 0; iter < n; iter++) {
        uint best = n;
        float bd  = FAR;
        for (uint i = 0; i < n; i++) {
            if (!done[i] && dist[i] < bd) {
                bd = dist[i];
                best = i;
            }
        }
        if (best == n) break;
        done[best] = true;
        int col = (int)(best % (uint)self.cols);
        int row = (int)(best / (uint)self.cols);
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                int c2 = col + dc;
                int r2 = row + dr;
                if ((dr == 0 && dc == 0) || c2 < 0 || r2 < 0 || c2 >= self.cols || r2 >= self.rows) continue;
                uint j = self.index(c2, r2);
                if (done[j]) continue;
                float d = bd + (self.rest[best] - self.rest[j]).length();
                if (d < dist[j]) {
                    dist[j] = d;
                    from[j] = from[best];
                }
            }
        }
    }
    for (uint i = 0; i < n; i++) {
        if (self.inv_mass[i] == 0 || dist[i] == FAR) continue;
        self.tethers.push({ i, from[i], dist[i] });
    }
}

// Gravity and the wind on each triangle, as accelerations for this step.
fn void Cloth.compute_forces(&self) {
    uint n       = self.count();
    Vector3* acc = self.accel.ptr;
    for (uint i = 0; i < n; i++) acc[i] = self.gravity;
    if (self.drag <= 0) return;
    self.gusts();
    Vector3* p = self.pos.ptr;
    Vector3* v = self.vel.ptr;
    // a third of the force on a point, as its acceleration
    float k    = self.drag * 0.5f / 3 * (float)n / math::max(self.mass, 1e-6f);
    for (int r = 0; r + 1 < self.rows; r++) {
        for (int c = 0; c + 1 < self.cols; c++) {
            uint i00    = self.index(c, r);
            uint i01    = i00 + (uint)self.cols;
            uint i11    = i01 + 1;
            uint i10    = i00 + 1;
            Vector3 air = self.wind * (1 + self.gust * gust_at(self, c, r));
            for (int tri = 0; tri < 2; tri++) {
                uint a     = i00;
                uint b     = tri == 0 ? i01 : i11;
                uint d     = tri == 0 ? i11 : i10;
                Vector3 cr = (p[b] - p[a]).cross(p[d] - p[a]);
                float len2 = cr.dot(cr);
                if (len2 < 1e-12f) continue;
                Vector3 rel = air - (v[a] + v[b] + v[d]) * (1.0f / 3);
                // drag * area * (n . rel) * n, with |cr| = 2 * area and n = cr / |cr|
                Vector3 f   = cr * (k * cr.dot(rel) / math::sqrt(len2));
                acc[a] += f;
                acc[b] += f;
                acc[d] += f;
            }
        }
    }
}

// The gusts this step, for gust_at: sines of time and of the distance along the rest shape's
// columns and rows, kept apart so that a cell's is a few products (sin(x + y) = sin x cos y +
// cos x sin y).
fn void Cloth.gusts(&self) {
    if (!self.gust_cols.len) {
        self.gust_cols = mem::new_array(float, (sz)self.cols * 4);
        self.gust_rows = mem::new_array(float, (sz)self.rows * 4);
    }
    float t    = (float)self.time;
    float span = (self.rest[self.index(self.cols - 1, 0)] - self.rest[0]).length();
    float drop = (self.rest[self.index(0, self.rows - 1)] - self.rest[0]).length();
    for (int c = 0; c < self.cols; c++) {
        float x = span * (float)c / (float)(self.cols - 1);
        float a = t * 0.73f + x * 1.9f;
        float b = t * 1.37f + x * 0.6f;
        self.gust_cols[c * 4 + 0] = math::sin(a);
        self.gust_cols[c * 4 + 1] = math::cos(a);
        self.gust_cols[c * 4 + 2] = math::sin(b);
        self.gust_cols[c * 4 + 3] = math::cos(b);
    }
    for (int r = 0; r < self.rows; r++) {
        float y = drop * (float)r / (float)(self.rows - 1);
        float a = y * 0.8f;
        float b = y * 2.3f;
        self.gust_rows[r * 4 + 0] = math::sin(a);
        self.gust_rows[r * 4 + 1] = math::cos(a);
        self.gust_rows[r * 4 + 2] = math::sin(b);
        self.gust_rows[r * 4 + 3] = math::cos(b);
    }
}

// -1..1, smooth across the cloth and in time, the same every run.
fn float gust_at(Cloth* c, int col, int row) @local @inline {
    float* x = &c.gust_cols[col * 4];
    float* y = &c.gust_rows[row * 4];
    return 0.6f * (x[0] * y[1] + x[1] * y[0]) + 0.4f * (x[2] * y[3] + x[3] * y[2]);
}

// Pushes the points out of the colliders (placed at `t` of their way; at `t_prev` the substep
// before) and above the floor.
fn void Cloth.collide(&self, float t, float t_prev = 0) {
    uint n        = self.count();
    Vector3* p    = self.pos.ptr;
    Vector3* q    = self.prev.ptr;
    float* w      = self.inv_mass.ptr;
    // over the substep's length: a push's velocity
    float h       = STEP / self.substeps;
    float inv_h   = 1 / h;
    Vector3 min_p = { FAR, FAR, FAR };
    Vector3 max_p = { -FAR, -FAR, -FAR };
    for (uint i = 0; i < n; i++) {
        min_p = math::min(min_p, p[i]);
        max_p = math::max(max_p, p[i]);
    }
    Vector3 pad = { self.thickness, self.thickness, self.thickness };
    self.lo = min_p - pad;
    self.hi = max_p + pad;
    foreach (&k : self.colliders) {
        if (!k.active) continue;
        if (!self.near(k)) continue;
        if (k.kind == BOX) {
            self.collide_box(k, t, t_prev);
            continue;
        }
        if (k.kind == PLANE) {
            self.collide_plane(k, t);
            continue;
        }
        // carried on by the velocity the pushes gave it
        k.shift += k.dv * h;
        Vector3 a      = k.from_a + (k.a - k.from_a) * t + k.shift;
        Vector3 b      = k.from_b + (k.b - k.from_b) * t + k.shift;
        float r        = k.radius + self.thickness;
        // the box around it, for a cheap test first
        Vector3 lo     = math::min(a, b) - (Vector3) { r, r, r };
        Vector3 hi     = math::max(a, b) + (Vector3) { r, r, r };
        Vector3 ab     = b - a;
        float ab2      = ab.dot(ab);
        Vector3 was_a  = k.from_a + (k.a - k.from_a) * t_prev + k.shift - k.dv * h;
        Vector3 was_ab = (k.from_b - k.from_a) + (k.b - k.from_b - k.a + k.from_a) * t_prev;
        float was_ab2  = was_ab.dot(was_ab);
        for (uint i = 0; i < n; i++) {
            Vector3 pi = p[i];
            if (pi.x < lo.x || pi.y < lo.y || pi.z < lo.z || pi.x > hi.x || pi.y > hi.y || pi.z > hi.z) continue;
            if (w[i] == 0) continue;
            Vector3 c = a;
            if (ab2 > 1e-12f) c = a + ab * clamp((pi - a).dot(ab) / ab2, 0, 1);
            Vector3 d = pi - c;
            float d2  = d.dot(d);
            if (d2 >= r * r) continue;
            float len   = math::sqrt(d2);
            Vector3 nrm = len > 1e-6f ? d / len : (Vector3) { 0, 0, 1 };
            // a point the collider has overtaken within the substep (it is past the axis, on
            // the far side from where it was a substep ago) goes back out the side it was on
            Vector3 qa  = was_a;
            if (was_ab2 > 1e-12f) qa = was_a + was_ab * clamp((q[i] - was_a).dot(was_ab) / was_ab2, 0, 1);
            Vector3 e = q[i] - qa;
            if (e.dot(e) >= r * r * 0.25f && e.dot(nrm) < 0) nrm = e.normalize();
            Vector3 out   = c + nrm * r;
            // friction: of the substep's move along the surface, keep (1 - friction)
            Vector3 moved = out - q[i];
            Vector3 along = moved - nrm * moved.dot(nrm);
            // the push, shared by the point and the collider by their inverse masses
            Vector3 push  = (out - along * self.friction - pi) * (1 / (1 + k.inv_mass / w[i]));
            Vector3 back  = push * (k.inv_mass / w[i]);
            k.impulse += push * (inv_h / w[i]);
            k.shift -= back;
            k.dv -= back * inv_h;
            a -= back;
            b -= back;
            p[i] = pi + push;
        }
    }
    if (self.floor_y > -FAR) {
        float y = self.floor_y - self.origin.y + self.thickness;
        for (uint i = 0; i < n; i++) {
            if (p[i].y >= y || w[i] == 0) continue;
            p[i].y = y;
            Vector3 moved = p[i] - q[i];
            p[i].x -= moved.x * self.floor_friction;
            p[i].z -= moved.z * self.floor_friction;
        }
    }
}

// The box's center and axes at `t` of its way.
fn Vector3 box_at(Collider* k, float t, Vector3[3]* axes) @local {
    Vector3 x = k.from_axes[0] + (k.axes[0] - k.from_axes[0]) * t;
    Vector3 y = k.from_axes[1] + (k.axes[1] - k.from_axes[1]) * t;
    x = x.normalize();
    Vector3 z = x.cross(y).normalize();
    (*axes)[0] = x;
    (*axes)[1] = z.cross(x);
    (*axes)[2] = z;
    return k.from_a + (k.a - k.from_a) * t;
}

// Pushes the points out of a box, through the face they came in by: the face their place a
// substep ago was outside of, in the box as it was then (the nearest face for a point that was
// already in). So a thin box (a door) moving or met edge on sends no point out the far side.
fn void Cloth.collide_box(&self, Collider* k, float t, float t_prev) {
    uint n        = self.count();
    Vector3* p    = self.pos.ptr;
    Vector3* q    = self.prev.ptr;
    float* w      = self.inv_mass.ptr;
    Vector3[3] ax;
    Vector3[3] was_ax;
    Vector3 c     = box_at(k, t, &ax);
    Vector3 was_c = box_at(k, t_prev, &was_ax);
    Vector3 h     = k.half + (Vector3) { self.thickness, self.thickness, self.thickness };
    // the box around it, for a cheap test first
    Vector3 ext;
    for (int i = 0; i < 3; i++) ext += (Vector3) { math::abs(ax[i].x), math::abs(ax[i].y), math::abs(ax[i].z) } * h[i];
    Vector3 lo = c - ext;
    Vector3 hi = c + ext;
    for (uint i = 0; i < n; i++) {
        Vector3 pi = p[i];
        if (pi.x < lo.x || pi.y < lo.y || pi.z < lo.z || pi.x > hi.x || pi.y > hi.y || pi.z > hi.z) continue;
        if (w[i] == 0) continue;
        Vector3 d = pi - c;
        Vector3 l = { d.dot(ax[0]), d.dot(ax[1]), d.dot(ax[2]) };
        if (math::abs(l.x) >= h.x || math::abs(l.y) >= h.y || math::abs(l.z) >= h.z) continue;
        Vector3 e   =  q[i] - was_c;
        Vector3 was =  { e.dot(was_ax[0]), e.dot(was_ax[1]), e.dot(was_ax[2]) };
        int face    = -1;
        float side;
        float least =  FAR;
        for (int j = 0; j < 3; j++) {
            if (math::abs(was[j]) < h[j]) continue;
            float depth = h[j] - math::abs(l[j]);
            if (depth < least) {
                least = depth;
                face = j;
                side = was[j] < 0 ? -1 : 1;
            }
        }
        if (face < 0) {
            for (int j = 0; j < 3; j++) {
                float depth = h[j] - math::abs(l[j]);
                if (depth < least) {
                    least = depth;
                    face = j;
                    side = l[j] < 0 ? -1 : 1;
                }
            }
        }
        l[face] = side * h[face];
        Vector3 out   = c + ax[0] * l.x + ax[1] * l.y + ax[2] * l.z;
        Vector3 nrm   = ax[face] * side;
        // friction: of the substep's move along the surface, keep (1 - friction)
        Vector3 moved = out - q[i];
        Vector3 along = moved - nrm * moved.dot(nrm);
        Vector3 to    = out - along * self.friction;
        k.impulse += (to - pi) * (self.substeps / STEP / w[i]);
        p[i] = to;
    }
}

// Pushes free vertices out of plane collider `k` at sweep fraction `t` and applies friction.
fn void Cloth.collide_plane(&self, Collider* k, float t) {
    uint n      = self.count();
    Vector3* p  = self.pos.ptr;
    Vector3* q  = self.prev.ptr;
    float* w    = self.inv_mass.ptr;
    Vector3 nrm = k.axes[0];
    Vector3 o   = k.from_a + (k.a - k.from_a) * t;
    for (uint i = 0; i < n; i++) {
        float d = (p[i] - o).dot(nrm) - self.thickness;
        if (d >= 0 || w[i] == 0) continue;
        Vector3 out   = p[i] - nrm * d;
        Vector3 moved = out - q[i];
        Vector3 along = moved - nrm * moved.dot(nrm);
        Vector3 to    = out - along * self.friction;
        k.impulse += (to - p[i]) * (self.substeps / STEP / w[i]);
        p[i] = to;
    }
}

fn float clamp(float x, float lo, float hi) @local => x < lo ? lo : (x > hi ? hi : x);

// The largest speed of a point (m/s).
fn float Cloth.max_speed(&self) {
    float m = 0;
    foreach (v : self.vel) m = math::max(m, v.length());
    return m;
}

// Whether any point is more than `distance` from where it was at the last mark() (always, before
// the first): to redraw a static light's shadow of the cloth when it has moved enough to show.
fn bool Cloth.moved(&self, float distance) {
    if (!self.marked.len) return true;
    float d2 = distance * distance;
    foreach (i, p : self.pos) {
        Vector3 d = p - self.marked[i];
        if (d.dot(d) > d2) return true;
    }
    return false;
}

// Snapshots current vertex positions as the reference baseline for `moved()`.
fn void Cloth.mark(&self) {
    if (!self.marked.len) self.marked = mem::new_array(Vector3, self.pos.len);
    self.marked[..] = self.pos[..];
}

// A mesh of the cloth, drawn where the points are (its matrix is the identity: add it anywhere
// in the scene). Use a DOUBLE_SIDE material to see both faces. Its uvs are in meters along the
// cloth at rest: u across from the first column, v up from the top row (negative below it).
fn Mesh* Cloth.make_mesh(&self, Material* material = null) {
    int cols          = self.detail ? self.detail.cols : self.cols;
    int rows          = self.detail ? self.detail.rows : self.rows;
    Vector3[] rest    = self.detail ? self.detail.rest : self.rest;
    BufferGeometry* g = drie::buffer_geometry();
    g.dynamic = true;
    float[] down = mem::new_array(float, cols);
    defer free(down);
    for (int r = 0; r < rows; r++) {
        float across = 0;
        for (int k = 0; k < cols; k++) {
            int i = r * cols + k;
            if (k > 0) across += (rest[i] - rest[i - 1]).length();
            if (r > 0) down[k] += (rest[i] - rest[i - cols]).length();
            g.add_vertex(self.detail ? rest[i] : self.origin + rest[i], { 0, 0, 1 }, { across, -down[k] });
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
    Mesh* m = drie::mesh(g, material);
    m.matrix_auto_update = false;
    m.matrix_world_auto_update = false;
    m.matrix = drie::mat4_identity();
    m.matrix_world = drie::mat4_identity();
    self.mesh = m;
    self.write_mesh(1);
    return m;
}

// Writes the points, `alpha` (0..1) of the way from the last step's start to now (or the
// finer grid over them), and their normals into the mesh's geometry, and marks it for the GPU
// (BufferGeometry.needs_update): unless no point is more than STILL from where the mesh last
// had it, when the mesh is left as it is.
fn void Cloth.write_mesh(&self, float alpha) {
    if (!self.mesh) return;
    if (!self.shown.len) self.shown = mem::new_array(Vector3, self.pos.len);
    foreach (i, &p : self.shown) *p = self.start[i] + (self.pos[i] - self.start[i]) * alpha;
    if (!self.drawn.len) {
        self.drawn = mem::new_array(Vector3, self.pos.len);
    } else {
        float far2 = STILL * STILL;
        bool moved;
        foreach (i, p : self.shown) {
            Vector3 d = p - self.drawn[i];
            if (d.dot(d) > far2) {
                moved = true;
                break;
            }
        }
        if (!moved) return;
    }
    self.drawn[..] = self.shown[..];
    BufferGeometry* g = self.mesh.geometry;
    g.vertices_changed();
    float* v   =  g.vertices.entries;
    Vector3 lo =  { FAR, FAR, FAR };
    Vector3 hi = -lo;
    int cols   =  self.cols;
    int rows   =  self.rows;
    if (self.detail) {
        // (frame_at's frame, worked out a row of a cell at a time: along a row the cell's u
        // does not change, and its v changes linearly)
        Detail* d = self.detail;
        cols = d.cols;
        rows = d.rows;
        Vector3[] p = self.shown;
        int cc      = self.cols;
        for (int r = 0; r < rows; r++) {
            int rc   =  d.row_cell[r];
            float s  =  d.row_s[r];
            int cell = -1;
            Vector3 left;
            Vector3 du;
            Vector3 u;
            Vector3 dv_left;
            Vector3 dv_right;
            for (int c = 0; c < cols; c++) {
                if (d.col_cell[c] != cell) {
                    cell = d.col_cell[c];
                    Vector3 p00 = p[rc * cc + cell];
                    Vector3 p10 = p[rc * cc + cell + 1];
                    Vector3 p01 = p[(rc + 1) * cc + cell];
                    Vector3 p11 = p[(rc + 1) * cc + cell + 1];
                    left = p00 + (p01 - p00) * s;
                    du = p10 + (p11 - p10) * s - left;
                    float ul = du.length();
                    u = ul > 1e-9f ? du / ul : (Vector3) { 1, 0, 0 };
                    dv_left = p00 - p01;
                    dv_right = p10 - p11;
                }
                float t   = d.col_t[c];
                Vector3 n = du.cross(dv_left + (dv_right - dv_left) * t);
                float nl  = n.length();
                n = nl > 1e-12f ? n / nl : (Vector3) { 0, 0, 1 };
                Vector3 vv = n.cross(u);
                int i      = r * cols + c;
                Vector3 o  = d.offset[i];
                Vector3 q  = self.origin + left + du * t + u * o.x + vv * o.y + n * o.z;
                Vector3 m  = d.normal[i];
                Vector3 nn = u * m.x + vv * m.y + n * m.z;
                float* w   = v + (usz)i * drie::FLOATS_PER_VERTEX;
                w[0] = q.x;
                w[1] = q.y;
                w[2] = q.z;
                w[3] = nn.x;
                w[4] = nn.y;
                w[5] = nn.z;
                lo = math::min(lo, q);
                hi = math::max(hi, q);
            }
        }
    } else {
        foreach (i, at : self.shown) {
            Vector3 p = self.origin + at;
            float* w  = v + (usz)i * drie::FLOATS_PER_VERTEX;
            w[0] = p.x;
            w[1] = p.y;
            w[2] = p.z;
            lo = math::min(lo, p);
            hi = math::max(hi, p);
        }
    }
    if (!self.detail) grid_normals(v, cols, rows);
    g.bounds_center = (lo + hi) * 0.5f;
    g.bounds_radius = (hi - lo).length() * 0.5f + 1e-3f;
}

// Each vertex's normal from its neighbours across and down, in a grid of `cols` x `rows`.
fn void grid_normals(float* v, int cols, int rows) @local {
    for (int r = 0; r < rows; r++) {
        for (int k = 0; k < cols; k++) {
            float* o    = v + (usz)(r * cols + k) * drie::FLOATS_PER_VERTEX;
            float* xl   = v + (usz)(r * cols + math::max(k - 1, 0)) * drie::FLOATS_PER_VERTEX;
            float* xr   = v + (usz)(r * cols + math::min(k + 1, cols - 1)) * drie::FLOATS_PER_VERTEX;
            float* yu   = v + (usz)(math::max(r - 1, 0) * cols + k) * drie::FLOATS_PER_VERTEX;
            float* yd   = v + (usz)(math::min(r + 1, rows - 1) * cols + k) * drie::FLOATS_PER_VERTEX;
            Vector3 du  = { xr[0] - xl[0], xr[1] - xl[1], xr[2] - xl[2] };
            Vector3 dv  = { yu[0] - yd[0], yu[1] - yd[1], yu[2] - yd[2] };
            Vector3 nrm = du.cross(dv);
            float len   = nrm.length();
            nrm = len > 1e-12f ? nrm / len : (Vector3) { 0, 0, 1 };
            o[3] = nrm.x;
            o[4] = nrm.y;
            o[5] = nrm.z;
        }
    }
}