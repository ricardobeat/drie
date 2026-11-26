// World-space raycasting against `Object3D` hierarchies: meshes, instanced meshes, posed
// `SkinnedMesh` geometry, and the active distance level of `Lod` nodes. Rejects meshes by
// bounding sphere then runs Möller-Trumbore linearly over triangles, honoring `Material.side`
// and skipping `shadow_only` meshes and lines. Reads `matrix_world` as stored (call
// `update_matrix_world()` or `Renderer.render()` first); hit lists belong to the caller.
module drie;
import std::collections::list, std::math, std::sort;

// Half-line defined by a world-space `origin` and unit `direction`.
struct Ray {
    // Ray starting point in world space.
    Vector3 origin;
    // Unit direction vector in world space.
    Vector3 direction;
}

// Evaluates the point at distance `t` along the ray (`origin + direction * t`).
fn Vector3 Ray.at(&self, float t) => self.origin + self.direction * t;

// Triangle hit by a ray: vertex indices into `BufferGeometry` and object-space face normal.
struct Face {
    // First triangle vertex index.
    uint a;
    // Second triangle vertex index.
    uint b;
    // Third triangle vertex index.
    uint c;
    // Unit geometric normal of the triangle in object space.
    Vector3 normal;
}

// Single ray-triangle hit returned by `Raycaster`.
struct Intersection {
    // Distance from `Ray.origin` to `point` in world units.
    float distance;
    // Hit position in world space.
    Vector3 point;
    // Scene object (`Mesh*`) that was hit.
    Object3D* object;
    // Hit triangle indices and object-space face normal.
    Face face;
    // Zero-based triangle index within `BufferGeometry.indices` (`indices[face_index * 3]`).
    int face_index;
    // Barycentrically interpolated texture coordinates at the hit point.
    Vector2 uv;
    // Interpolated vertex normal transformed into world space.
    Vector3 normal;
    // Instance index for an `InstancedMesh`, or `-1` for a non-instanced mesh.
    int instance_id;
}

// Ray intersection query over scene objects within `[near, far]`.
// 
//   Raycaster ray = drie::raycaster();
//   ray.set_from_camera({ x, y }, camera);            // x, y in -1..1 NDC
//   List{Intersection} hits;
//   ray.intersect_objects(scene.children.array_view(), &hits);
//   if (hits.len()) hits[0].object.visible = false;   // sorted nearest first
//   hits.clear();
// 
//   Intersection hit;
//   if (ray.intersect_first(scene, &hit)) aim_at(hit.point);
struct Raycaster {
    // World-space ray tested against scene geometry.
    Ray ray;
    // Minimum hit distance accepted along the ray.
    float near;
    // Maximum hit distance accepted along the ray.
    float far;
    // When true, skips objects with `visible == false` and their subtrees.
    bool visible_only;
}

// Creates a `Raycaster` with normalized `direction` and distance bounds `[near, far]`.
fn Raycaster raycaster(
    Vector3 origin = { 0, 0, 0 },
    Vector3 direction = { 0, 0, -1 },
    float near = 0,
    float far = float::inf
) {
    Raycaster r = { .near = near, .far = far };
    r.set(origin, direction);
    return r;
}

// Sets the ray origin and normalizes `direction`.
fn void Raycaster.set(&self, Vector3 origin, Vector3 direction) {
    self.ray = { origin, direction.normalize() };
}

// Aim the ray from `camera` through normalized device coordinates `coords` (`-1..1` left to
// right, bottom to top; from window pixels: `x = 2 * px / w - 1`, `y = 1 - 2 * py / h`).
fn void Raycaster.set_from_camera(&self, Vector2 coords, Camera* camera) {
    Mat4 w        = camera.matrix_world;
    Vector3 right = { w[0][0], w[0][1], w[0][2] };
    Vector3 up    = { w[1][0], w[1][1], w[1][2] };
    Vector3 back  = { w[2][0], w[2][1], w[2][2] };
    switch (camera.camera_type) {
        case PERSPECTIVE:
            PerspectiveCamera* p = (PerspectiveCamera*)camera;
            float tan_y = math::tan(p.fov * DEG2RAD * 0.5f) / camera.zoom;
            float tan_x = tan_y * p.aspect;
            self.set(w.get_position(), right * (coords.x * tan_x) + up * (coords.y * tan_y) - back);
        case ORTHOGRAPHIC:
            OrthographicCamera* o = (OrthographicCamera*)camera;
            float cx = (o.left + o.right) / 2;
            float cy = (o.top + o.bottom) / 2;
            float dx = (o.right - o.left) / (2 * camera.zoom);
            float dy = (o.top - o.bottom) / (2 * camera.zoom);
            Vector3 local = { cx + coords.x * dx, cy + coords.y * dy, -camera.near };
            self.set(w.transform_point(local), -back);
    }
}

// Destination for ray hits: appends all hits to `list`, or tracks only the closest in `nearest`.
struct HitSink @local {
    List{Intersection}* list;
    Intersection* nearest;
    bool found;
}

// Records `hit` into the list or replaces `nearest` if closer.
fn void HitSink.add(&self, Intersection hit) {
    if (self.list) {
        self.list.push(hit);
        return;
    }
    if (!self.found || hit.distance < self.nearest.distance) {
        *self.nearest = hit;
        self.found = true;
    }
}

// Returns true if a hit at `distance` could still be kept (for early bounding-sphere rejection).
fn bool HitSink.wants(&self, float distance) => self.list || !self.found || distance < self.nearest.distance;

fn int cmp_hits(
    Intersection a,
    Intersection b
) @local => a.distance < b.distance ? -1 : (a.distance > b.distance ? 1 : 0);

// Appends all hits on `object` (and descendants when `recursive`) to `intersects`, sorts the
// list nearest-first, and returns its length. Caller owns `intersects` (`clear`/`free`).
fn usz Raycaster.intersect_object(&self, Object3D* object, List{Intersection}* intersects, bool recursive = true) {
    HitSink sink = { .list = intersects };
    self.visit(object, &sink, recursive);
    sort::quicksort(intersects, &cmp_hits);
    return (usz)intersects.len();
}

// Appends hits across every root in `objects`, sorts `intersects` nearest-first, and returns
// its total length. Caller owns `intersects`.
fn usz Raycaster.intersect_objects(&self, Object3D*[] objects, List{Intersection}* intersects, bool recursive = true) {
    HitSink sink = { .list = intersects };
    foreach (o : objects) self.visit(o, &sink, recursive);
    sort::quicksort(intersects, &cmp_hits);
    return (usz)intersects.len();
}

// Writes the nearest hit on `object` (and descendants when `recursive`) to `hit` without
// allocating a list. Returns false when nothing is hit.
fn bool Raycaster.intersect_first(&self, Object3D* object, Intersection* hit, bool recursive = true) {
    HitSink sink = { .nearest = hit };
    self.visit(object, &sink, recursive);
    return sink.found;
}

// Recursively traverses `object`, testing meshes and selecting the active `Lod` level.
fn void Raycaster.visit(&self, Object3D* object, HitSink* sink, bool recursive) {
    if (self.visible_only && !object.visible) return;
    switch (object.kind) {
        case MESH:
            if (!((Mesh*)object).shadow_only) self.intersect_mesh((Mesh*)object, sink);
        case LOD:
            // Pick the level visible from the ray origin using the renderer's LOD distance scale.
            Lod* l = (Lod*)object;
            float scale = lod_distance_scale();
            float d = (l.matrix_world.get_position() - self.ray.origin).length() / scale;
            sz pick = -1;
            for (sz i = 0; i < l.distances.len(); i++) if (d >= l.distances[i]) pick = i;
            if (recursive && pick >= 0 && pick < l.children.len()) self.visit(l.children[pick], sink, true);
            return;
        default:
            break;
    }
    if (!recursive) return;
    foreach (c : object.children) self.visit(c, sink, true);
}

fn float scale_of(Mat4 m) @local {
    float sx = ((Vector3) { m[0][0], m[0][1], m[0][2] }).length();
    float sy = ((Vector3) { m[1][0], m[1][1], m[1][2] }).length();
    float sc = ((Vector3) { m[2][0], m[2][1], m[2][2] }).length();
    return math::max(sx, math::max(sy, sc));
}

// Returns true if the ray intersects the sphere `(center, radius)` within `[near, far]` and
// closer than any hit `sink` already requires.
fn bool Raycaster.near_sphere(&self, Vector3 center, float radius, HitSink* sink) {
    Vector3 to  = center - self.ray.origin;
    float along = to.dot(self.ray.direction);
    float off2  = to.dot(to) - along * along;
    if (off2 > radius * radius) return false;
    float half = math::sqrt(radius * radius - off2);
    // Ray segment inside the sphere spans `[along - half, along + half]`.
    if (along + half < self.near || along - half > self.far) return false;
    return sink.wants(math::max(along - half, self.near));
}

// Intersects a single `Mesh` (or every instance of an `InstancedMesh`), skipping lines.
fn void Raycaster.intersect_mesh(&self, Mesh* m, HitSink* sink) {
    BufferGeometry* g = m.geometry;
    // Lines have segment/strip indices rather than triangles.
    if (!g || g.indices.len() < 3 || m.is_line()) return;
    if (g.bounds_radius == 0) g.compute_bounding_sphere();
    Side side = m.material ? m.material.side : FRONT_SIDE;
    if (!m.instance_data.len()) {
        self.intersect_geometry(m, m.matrix_world, -1, side, sink);
        return;
    }
    for (int i = 0; i < m.instance_count; i++) {
        float* d = &m.instance_data[(sz)i * 12];
        // Unpack row-major 3x4 instance matrix from `InstancedMesh.set_matrix_at`.
        Mat4 inst = {
            { d[0], d[4], d[8], 0 },
            { d[1], d[5], d[9], 0 },
            { d[2], d[6], d[10], 0 },
            { d[3], d[7], d[11], 1 }
        };
        self.intersect_geometry(m, m.matrix_world.mul(inst), i, side, sink);
    }
}

// Runs Möller-Trumbore ray-triangle tests in `m`'s local space for transform `world`.
fn void Raycaster.intersect_geometry(&self, Mesh* m, Mat4 world, int instance, Side side, HitSink* sink) {
    BufferGeometry* g = m.geometry;
    if (!self.near_sphere(world.transform_point(g.bounds_center), g.bounds_radius * scale_of(world), sink)) return;
    Mat4 inv    = world.invert();
    Vector3 o   = inv.transform_point(self.ray.origin);
    Vector3 dir = inv.transform_point(self.ray.origin + self.ray.direction) - o;
    sz tris     = g.indices.len() / 3;
    for (sz t = 0; t < tris; t++) {
        uint ia    = g.indices[t * 3];
        uint ib    = g.indices[t * 3 + 1];
        uint ic    = g.indices[t * 3 + 2];
        Vector3 a  = g.get_position(ia);
        Vector3 e1 = g.get_position(ib) - a;
        Vector3 e2 = g.get_position(ic) - a;
        Vector3 p  = dir.cross(e2);
        float det  = e1.dot(p);
        // Counter-clockwise winding defines the front face.
        if (det > -1e-12f && det < 1e-12f) continue;
        if (side == FRONT_SIDE && det < 0) continue;
        if (side == BACK_SIDE && det > 0) continue;
        float inv_det = 1 / det;
        Vector3 s     = o - a;
        float u       = s.dot(p) * inv_det;
        if (u < 0 || u > 1) continue;
        Vector3 q = s.cross(e1);
        float v   = dir.dot(q) * inv_det;
        if (v < 0 || u + v > 1) continue;
        float tl = e2.dot(q) * inv_det;
        if (tl < 0) continue;
        Vector3 point  = world.transform_point(o + dir * tl);
        float distance = (point - self.ray.origin).length();
        if (distance < self.near || distance > self.far || !sink.wants(distance)) continue;
        float w         = 1 - u - v;
        Vector3 local_n = g.get_normal(ia) * w + g.get_normal(ib) * u + g.get_normal(ic) * v;
        // Transform normal by the inverse-transpose of `world`.
        Vector3 n = {
            inv[0][0] * local_n.x + inv[0][1] * local_n.y + inv[0][2] * local_n.z,
            inv[1][0] * local_n.x + inv[1][1] * local_n.y + inv[1][2] * local_n.z,
            inv[2][0] * local_n.x + inv[2][1] * local_n.y + inv[2][2] * local_n.z,
        };
        sink.add(
            {
                .distance = distance,
                .point = point,
                .object = &m.object,
                .face = { ia, ib, ic, e1.cross(e2)
            .normalize() }, .face_index = (int)t, .uv = g.get_uv(
                ia
            ) * w + g.get_uv(
                ib
            ) * u + g.get_uv(ic) * v, .normal = n.length() > 0 ? n.normalize() : n, .instance_id = instance, });
    }
}