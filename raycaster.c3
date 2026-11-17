// Raycasting, like three's Raycaster: what a ray from a point, or from the camera through a
// point of the picture, runs into.
//
//   Raycaster ray = drie::raycaster();
//   ray.set_from_camera({ x, y }, camera);           // x, y: -1..1 across and up the view
//   List{Intersection} hits;
//   ray.intersect_objects(scene.children.array_view(), &hits);
//   if (hits.len()) hits[0].object.visible = false; // the nearest first
//   hits.clear();
//
//   Intersection hit;
//   if (ray.intersect_first(scene, &hit)) aim_at(hit.point);   // only the nearest, no list
//
// It tests the triangles of meshes (each instance of an instanced mesh, the posed geometry of
// a skinned one) against the ray, after their bounding spheres, honouring the material's
// `side`. Of a Lod, only the level the ray's origin would see is tested. Like three, it looks
// at every triangle of the meshes the ray comes near: for big meshes that is linear in their
// size. World matrices are used as they are: update_matrix_world() (or a render) first.
module drie;
import std::collections::list, std::math, std::sort;

// A ray: `origin` and a `direction` of length 1, like three's Ray.
struct Ray {
    Vector3 origin;
    Vector3 direction;
}

fn Vector3 Ray.at(&self, float t) => self.origin + self.direction * t;

// The triangle that was hit: its vertices (indices into the geometry) and its normal in the
// object's own space, like three's `face`.
struct Face {
    uint a;
    uint b;
    uint c;
    Vector3 normal;
}

// One hit, like three's intersection objects.
struct Intersection {
    // from the ray's origin, in world units
    float distance;
    // where, in world space
    Vector3 point;
    Object3D* object;
    Face face;
    int face_index;
    // the texture coordinates there
    Vector2 uv;
    // the geometry's normal there (between the vertices' normals), in world space. three's
    // `normal` is in the object's space.
    Vector3 normal;
    // which instance of an instanced mesh (-1: not instanced)
    int instance_id;
}

struct Raycaster {
    Ray ray;
    // only hits between these distances count
    float near;
    float far;
    // skip what is not `visible` (and everything under it). three tests every object.
    bool visible_only;
}

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

// `direction` need not be of length 1.
fn void Raycaster.set(&self, Vector3 origin, Vector3 direction) {
    self.ray = { origin, direction.normalize() };
}

// The ray from the camera through a point of its picture: `coords` from -1 to 1, left to
// right and bottom to top (three's normalized device coordinates). For a point in the
// window: x = 2 * px / width - 1, y = 1 - 2 * py / height.
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

// Where the hits go: all of them into a list, or only the nearest.
struct HitSink @local {
    List{Intersection}* list;
    Intersection* nearest;
    bool found;
}

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

// Whether the sink can still take a hit at this distance (for skipping a far-off object).
fn bool HitSink.wants(&self, float distance) => self.list || !self.found || distance < self.nearest.distance;

fn int cmp_hits(
    Intersection a,
    Intersection b
) @local => a.distance < b.distance ? -1 : (a.distance > b.distance ? 1 : 0);

// Adds what the ray hits of `object` (and of its descendants, with `recursive`) to `intersects`,
// keeping the list sorted nearest first, and returns how many hits it has. The list is the
// caller's: clear() it between rays, free() it at the end.
fn usz Raycaster.intersect_object(&self, Object3D* object, List{Intersection}* intersects, bool recursive = true) {
    HitSink sink = { .list = intersects };
    self.visit(object, &sink, recursive);
    sort::quicksort(intersects, &cmp_hits);
    return (usz)intersects.len();
}

// The same for several objects (scene.children.array_view(), say).
fn usz Raycaster.intersect_objects(&self, Object3D*[] objects, List{Intersection}* intersects, bool recursive = true) {
    HitSink sink = { .list = intersects };
    foreach (o : objects) self.visit(o, &sink, recursive);
    sort::quicksort(intersects, &cmp_hits);
    return (usz)intersects.len();
}

// Only the nearest hit, into `hit`, without a list. False if the ray hits nothing.
fn bool Raycaster.intersect_first(&self, Object3D* object, Intersection* hit, bool recursive = true) {
    HitSink sink = { .nearest = hit };
    self.visit(object, &sink, recursive);
    return sink.found;
}

fn void Raycaster.visit(&self, Object3D* object, HitSink* sink, bool recursive) {
    if (self.visible_only && !object.visible) return;
    switch (object.kind) {
        case MESH:
            self.intersect_mesh((Mesh*)object, sink);
        case LOD:
            // the level the ray's origin sees, as the renderer would pick it for a camera there
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

// Whether the ray passes through the sphere within near..far (and the sink still wants it).
fn bool Raycaster.near_sphere(&self, Vector3 center, float radius, HitSink* sink) {
    Vector3 to  = center - self.ray.origin;
    float along = to.dot(self.ray.direction);
    float off2  = to.dot(to) - along * along;
    if (off2 > radius * radius) return false;
    float half = math::sqrt(radius * radius - off2);
    // the stretch of the ray inside the sphere: along - half .. along + half
    if (along + half < self.near || along - half > self.far) return false;
    return sink.wants(math::max(along - half, self.near));
}

fn void Raycaster.intersect_mesh(&self, Mesh* m, HitSink* sink) {
    BufferGeometry* g = m.geometry;
    if (!g || g.indices.len() < 3) return;
    if (g.bounds_radius == 0) g.compute_bounding_sphere();
    Side side = m.material ? m.material.side : FRONT_SIDE;
    if (!m.instance_data.len()) {
        self.intersect_geometry(m, m.matrix_world, -1, side, sink);
        return;
    }
    for (int i = 0; i < m.instance_count; i++) {
        float* d  = &m.instance_data[(sz)i * 12];
        // the instance's rows of a 3x4 matrix (InstancedMesh.set_matrix_at)
        Mat4 inst = {
            { d[0], d[4], d[8], 0 },
            { d[1], d[5], d[9], 0 },
            { d[2], d[6], d[10], 0 },
            { d[3], d[7], d[11], 1 }
        };
        self.intersect_geometry(m, m.matrix_world.mul(inst), i, side, sink);
    }
}

// The triangles of `m`'s geometry placed by `world` (Möller-Trumbore, in the geometry's own
// space).
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
        // counter-clockwise is the front, as in three
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
        // normals go through the inverse transpose
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