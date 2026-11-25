// Cloth: a grid of point masses held together by distance constraints, solved with XPBD in
// small steps (Macklin, Müller et al., "Small Steps in Physics Simulation", 2019): each fixed
// step of 1/60 s is cut into `substeps`, and each substep moves the points by their velocity,
// projects every constraint once and pushes the points out of the colliders.
//
//   - stretch: each point to its neighbours across and down
//   - shear: the diagonals of each cell
//   - bend: each point to the one two away, across and down (the rest shape's folds stay in)
//   - tethers: no point gets further from its nearest pin than it is along the cloth at rest
//
// Pinned points (inverse mass 0) are held where pin() or move_pin() puts them. Colliders are
// spheres and capsules, moved with move_collider(); the points go where the collider was at
// each substep, so a fast capsule does not pass through. The wind pushes on each triangle by
// how much the air crosses it, and gusts with a deterministic noise in time and space.
// Everything is in world space; the same calls give the same result every run.
//
//     Cloth* c = cloth::grid(24, 36, { -0.6, 2.4, 0 }, { 1.2, 0, 0 }, { 0, -2.4, 0 });
//     c.pin_row(0);
//     scene.add(c.make_mesh(drie::mesh_standard_material(color: 0x8a1010, side: DOUBLE_SIDE)));
//     int body = c.add_capsule(a, b, 0.25);
//     // each frame:
//     c.move_collider(body, a, b);
//     c.update(delta);   // steps, and writes the mesh
module drie::cloth;
import std::math, std::collections::list, drie;

const float STEP       = 1.0f / 60;
const int MAX_STEPS    = 4;
const float FAR @local = 1e30f;

enum ColliderKind : char {
    SPHERE,
    CAPSULE,
}

// A sphere (a) or a capsule (the segment a..b) of `radius`.
struct Collider {
    ColliderKind kind;
    float radius;
    // where it was at the end of the last step and where it is going
    Vector3 from_a, from_b;
    Vector3 a, b;
    bool active;
}

struct Link {
    uint a, b;
    float rest;
    // each end's share of the correction: its inverse mass over (both + the compliance)
    float ka, kb;
}

struct Tether {
    uint point, pin;
    float rest;
}

struct Cloth {
    int cols, rows;
    // per point: where it is, where it was a substep ago, its velocity, where it was at the
    // start of the last fixed step (the mesh is drawn between that and now)
    Vector3[] pos;
    Vector3[] prev;
    Vector3[] vel;
    Vector3[] start;
    Vector3[] accel;
    Vector3[] rest;
    float[] inv_mass;
    // a pinned point's place
    Vector3[] pin_to;
    // per column and row: the sines gust_at puts together
    float[] gust_cols;
    float[] gust_rows;
    List{Link} stretch;
    List{Link} shear;
    List{Link} bend;
    List{Tether} tethers;
    bool tethers_dirty;
    // the compliances (over h²) the links' ka and kb were worked out for; -1: not yet
    float[3] links_alpha;
    List{Collider} colliders;

    int substeps;
    // compliance (m/N, 0 rigid) of the three kinds of links
    float stretch_compliance;
    float shear_compliance;
    float bend_compliance;
    // a tether lets its point go this much further than the rest distance (0.02 = 2%)
    float tether_slack;
    // velocity lost per second (0..)
    float damping;
    Vector3 gravity;
    // the air's velocity (m/s); `gust` scales its variation (0 steady, 1 between 0 and twice)
    Vector3 wind;
    float gust;
    // how hard the air pushes on the cloth per m² per m/s of air across it
    float drag;
    // points stay this far from the colliders' surfaces
    float thickness;
    // 0 slides along a collider, 1 sticks to it
    float friction;
    // a floor (y up) the points stay above; -1e30: none
    float floor_y;
    float mass;

    double time;
    float accumulator;
    ulong steps;
    Mesh* mesh;
}

fn uint Cloth.index(&self, int col, int row) @inline => (uint)(row * self.cols + col);
fn uint Cloth.count(&self) @inline => (uint)(self.cols * self.rows);

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
    c.tethers.allocator = mem;
    c.colliders.allocator = mem;
    c.substeps = 8;
    c.stretch_compliance = 0;
    c.shear_compliance = 1e-3f;
    c.bend_compliance = 1e-2f;
    c.tether_slack = 0.02f;
    c.damping = 0.6f;
    c.gravity = { 0, -9.81f, 0 };
    c.drag = 0.6f;
    c.thickness = 0.02f;
    c.friction = 0.3f;
    c.floor_y = -FAR;
    c.mass = mass;
    c.links_alpha = { -1, -1, -1 };
    float w = (float)n / math::max(mass, 1e-6f);
    for (sz i = 0; i < n; i++) {
        c.pos[i] = rest[i];
        c.rest[i] = rest[i];
        c.prev[i] = rest[i];
        c.start[i] = rest[i];
        c.pin_to[i] = rest[i];
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
            if (k + 2 < cols) c.link(&c.bend, i, c.index(k + 2, r));
            if (r + 2 < rows) c.link(&c.bend, i, c.index(k, r + 2));
        }
    }
    color_links(&c.stretch, n);
    color_links(&c.shear, n);
    color_links(&c.bend, n);
    return c;
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

fn void Cloth.link(&self, List{Link}* list, uint a, uint b) {
    list.push({ a, b, (self.pos[a] - self.pos[b]).length(), 0, 0 });
}

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
    self.tethers.free();
    self.colliders.free();
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

fn void Cloth.pin_row(&self, int row) {
    for (int k = 0; k < self.cols; k++) self.pin(self.index(k, row));
}

// Lets point `i` go again.
fn void Cloth.unpin(&self, uint i) {
    self.inv_mass[i] = (float)self.count() / math::max(self.mass, 1e-6f);
    self.tethers_dirty = true;
    self.links_alpha = { -1, -1, -1 };
}

// Moves pinned point `i`: it gets there at the next fixed step.
fn void Cloth.move_pin(&self, uint i, Vector3 to) => self.pin_to[i] = to;

fn int Cloth.add_sphere(&self, Vector3 center, float radius) {
    self.colliders.push({ SPHERE, radius, center, center, center, center, true });
    return (int)self.colliders.len() - 1;
}

fn int Cloth.add_capsule(&self, Vector3 a, Vector3 b, float radius) {
    self.colliders.push({ CAPSULE, radius, a, b, a, b, true });
    return (int)self.colliders.len() - 1;
}

// Moves a collider (b is ignored for a sphere). The cloth sees it travel there over the steps
// of the next update; from inactive, it appears there.
fn void Cloth.move_collider(&self, int id, Vector3 a, Vector3 b = { 0, 0, 0 }) {
    Collider* k = &self.colliders[id];
    if (k.kind == SPHERE) b = a;
    if (!k.active) {
        k.from_a = a;
        k.from_b = b;
    }
    k.a = a;
    k.b = b;
}

fn void Cloth.set_collider_active(&self, int id, bool active) => self.colliders[id].active = active;

// Advances by `dt` seconds in fixed steps (at most MAX_STEPS) and writes the mesh, if it has
// one. Returns the steps taken.
fn int Cloth.update(&self, float dt) {
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
            k.from_a = k.a;
            k.from_b = k.b;
        }
    }
    if (self.mesh) self.write_mesh(self.accumulator / STEP);
    return n;
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
        prepare_links(&self.bend, w, alpha[2]);
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
        solve_links(p, &self.bend);
        self.solve_tethers();
        self.collide(from + (until - from) * (float)(s + 1) * step_frac);
        for (uint i = 0; i < n; i++) {
            if (w[i] != 0) v[i] = (p[i] - q[i]) * keep;
        }
    }
    self.time += STEP;
    self.steps++;
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
    float[] dist = mem::new_array(float, n);
    uint[] from  = mem::new_array(uint, n);
    bool[] done  = mem::new_array(bool, n);
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

// Pushes the points out of the colliders (placed at `t` of their way) and above the floor.
fn void Cloth.collide(&self, float t) {
    uint n     = self.count();
    Vector3* p = self.pos.ptr;
    Vector3* q = self.prev.ptr;
    float* w   = self.inv_mass.ptr;
    foreach (&k : self.colliders) {
        if (!k.active) continue;
        Vector3 a  = k.from_a + (k.a - k.from_a) * t;
        Vector3 b  = k.from_b + (k.b - k.from_b) * t;
        float r    = k.radius + self.thickness;
        // the box around it, for a cheap test first
        Vector3 lo = math::min(a, b) - (Vector3) { r, r, r };
        Vector3 hi = math::max(a, b) + (Vector3) { r, r, r };
        Vector3 ab = b - a;
        float ab2  = ab.dot(ab);
        for (uint i = 0; i < n; i++) {
            Vector3 pi = p[i];
            if (pi.x < lo.x || pi.y < lo.y || pi.z < lo.z || pi.x > hi.x || pi.y > hi.y || pi.z > hi.z) continue;
            if (w[i] == 0) continue;
            Vector3 c = a;
            if (ab2 > 1e-12f) c = a + ab * clamp((pi - a).dot(ab) / ab2, 0, 1);
            Vector3 d = pi - c;
            float d2  = d.dot(d);
            if (d2 >= r * r) continue;
            float len     = math::sqrt(d2);
            Vector3 nrm   = len > 1e-6f ? d / len : (Vector3) { 0, 0, 1 };
            Vector3 out   = c + nrm * r;
            // friction: of the substep's move along the surface, keep (1 - friction)
            Vector3 moved = out - q[i];
            Vector3 along = moved - nrm * moved.dot(nrm);
            p[i] = out - along * self.friction;
        }
    }
    if (self.floor_y > -FAR) {
        float y = self.floor_y + self.thickness;
        for (uint i = 0; i < n; i++) {
            if (p[i].y >= y || w[i] == 0) continue;
            p[i].y = y;
            Vector3 moved = p[i] - q[i];
            p[i].x -= moved.x * self.friction;
            p[i].z -= moved.z * self.friction;
        }
    }
}

fn float clamp(float x, float lo, float hi) @local => x < lo ? lo : (x > hi ? hi : x);

// The largest speed of a point (m/s).
fn float Cloth.max_speed(&self) {
    float m = 0;
    foreach (v : self.vel) m = math::max(m, v.length());
    return m;
}

// A mesh of the cloth, drawn where the points are (its matrix is the identity: add it anywhere
// in the scene). Use a DOUBLE_SIDE material to see both faces. Its uvs are in meters along the
// cloth at rest: u across from the first column, v up from the top row (negative below it).
fn Mesh* Cloth.make_mesh(&self, Material* material = null) {
    BufferGeometry* g = drie::buffer_geometry();
    g.dynamic = true;
    float[] down = mem::new_array(float, self.cols);
    defer free(down);
    for (int r = 0; r < self.rows; r++) {
        float across = 0;
        for (int k = 0; k < self.cols; k++) {
            uint i = self.index(k, r);
            if (k > 0) across += (self.rest[i] - self.rest[i - 1]).length();
            if (r > 0) down[k] += (self.rest[i] - self.rest[i - (uint)self.cols]).length();
            g.add_vertex(self.pos[i], { 0, 0, 1 }, { across, -down[k] });
        }
    }
    for (int r = 0; r + 1 < self.rows; r++) {
        for (int k = 0; k + 1 < self.cols; k++) {
            uint a = self.index(k, r);
            uint b = self.index(k, r + 1);
            uint c = self.index(k + 1, r + 1);
            uint d = self.index(k + 1, r);
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

// Writes the points, `alpha` (0..1) of the way from the last step's start to now, and their
// normals into the mesh's geometry.
fn void Cloth.write_mesh(&self, float alpha) {
    if (!self.mesh) return;
    BufferGeometry* g =  self.mesh.geometry;
    float* v          =  g.vertices.entries;
    Vector3 lo        =  { FAR, FAR, FAR };
    Vector3 hi        = -lo;
    uint n            =  self.count();
    for (uint i = 0; i < n; i++) {
        Vector3 p = self.start[i] + (self.pos[i] - self.start[i]) * alpha;
        float* o  = v + (usz)i * drie::FLOATS_PER_VERTEX;
        o[0] = p.x;
        o[1] = p.y;
        o[2] = p.z;
        lo = math::min(lo, p);
        hi = math::max(hi, p);
    }
    int cols = self.cols;
    int rows = self.rows;
    for (int r = 0; r < rows; r++) {
        for (int k = 0; k < cols; k++) {
            float* o    = v + (usz)self.index(k, r) * drie::FLOATS_PER_VERTEX;
            float* xl   = v + (usz)self.index(math::max(k - 1, 0), r) * drie::FLOATS_PER_VERTEX;
            float* xr   = v + (usz)self.index(math::min(k + 1, cols - 1), r) * drie::FLOATS_PER_VERTEX;
            float* yu   = v + (usz)self.index(k, math::max(r - 1, 0)) * drie::FLOATS_PER_VERTEX;
            float* yd   = v + (usz)self.index(k, math::min(r + 1, rows - 1)) * drie::FLOATS_PER_VERTEX;
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
    g.bounds_center = (lo + hi) * 0.5f;
    g.bounds_radius = (hi - lo).length() * 0.5f + 1e-3f;
}