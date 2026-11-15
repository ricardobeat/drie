// Geometries. Vertices are interleaved: position (3), normal (3), uv (2), color (3).
module three;
import std::collections::list, std::math;
import sokol;

const int FLOATS_PER_VERTEX = 11;

struct BufferGeometry
{
    List{float} vertices;
    List{uint} indices;
    // Skinning (see skeleton.c3): per vertex, four bones of a Skeleton and their weights, like
    // three's skinIndex and skinWeight attributes. Empty unless set with set_skin / skin_range.
    List{ushort} skin_indices;
    List{float} skin_weights;
    // GPU state, created lazily on first render.
    SgBuffer vbuf;
    SgBuffer ibuf;
    SgBuffer wire_ibuf;
    uint wire_count;
    uint index_count;
    bool uploaded;
    // Bounding sphere in local space, computed on upload (or by compute_bounding_sphere).
    Vector3 bounds_center;
    float bounds_radius;
}

// Makes the vertex and index lists heap allocated (a plain List would use the temp allocator).
fn void BufferGeometry.init_lists(&self)
{
    self.vertices.allocator = mem;
    self.indices.allocator = mem;
    self.skin_indices.allocator = mem;
    self.skin_weights.allocator = mem;
}

fn void BufferGeometry.compute_bounding_sphere(&self)
{
    Vector3 lo = { 1e30f, 1e30f, 1e30f };
    Vector3 hi = -lo;
    uint n = self.vertex_count();
    if (!n) return;
    for (uint i = 0; i < n; i++)
    {
        Vector3 p = self.get_position(i);
        lo = math::min(lo, p);
        hi = math::max(hi, p);
    }
    self.bounds_center = (lo + hi) * 0.5f;
    float r2 = 0;
    for (uint i = 0; i < n; i++)
    {
        Vector3 d = self.get_position(i) - self.bounds_center;
        r2 = math::max(r2, d.dot(d));
    }
    self.bounds_radius = math::sqrt(r2);
}

// Replaces the UVs with a box projection of the positions: each face is mapped along its
// dominant normal axis, `tiles_per_unit` texture repeats per unit. Handy for walls and floors.
fn BufferGeometry* BufferGeometry.box_project_uv(&self, float tiles_per_unit = 1)
{
    for (sz o = 0; o < self.vertices.len(); o += FLOATS_PER_VERTEX)
    {
        float[] v = self.vertices.entries[o:FLOATS_PER_VERTEX];
        float ax = math::abs(v[3]);
        float ay = math::abs(v[4]);
        float az = math::abs(v[5]);
        Vector2 uv;
        if (ay >= ax && ay >= az)
        {
            uv = { v[0], v[2] };
        }
        else if (ax >= az)
        {
            uv = { v[2], -v[1] };
        }
        else
        {
            uv = { v[0], -v[1] };
        }
        v[6] = uv.x * tiles_per_unit;
        v[7] = uv.y * tiles_per_unit;
    }
    self.uploaded = false;
    return self;
}

fn BufferGeometry* BufferGeometry.add_vertex(&self, Vector3 p, Vector3 n, Vector2 uv, Vector3 color = { 1, 1, 1 })
{
    self.vertices.push(p.x);
    self.vertices.push(p.y);
    self.vertices.push(p.z);
    self.vertices.push(n.x);
    self.vertices.push(n.y);
    self.vertices.push(n.z);
    self.vertices.push(uv.x);
    self.vertices.push(uv.y);
    self.vertices.push(color.x);
    self.vertices.push(color.y);
    self.vertices.push(color.z);
    return self;
}

fn uint BufferGeometry.vertex_count(&self) => (uint)(self.vertices.len() / FLOATS_PER_VERTEX);

fn Vector3 BufferGeometry.get_position(&self, uint i)
{
    sz o = (sz)i * FLOATS_PER_VERTEX;
    return { self.vertices[o], self.vertices[o + 1], self.vertices[o + 2] };
}

fn void BufferGeometry.set_normal(&self, uint i, Vector3 n)
{
    sz o = (sz)i * FLOATS_PER_VERTEX + 3;
    self.vertices[o] = n.x;
    self.vertices[o + 1] = n.y;
    self.vertices[o + 2] = n.z;
}

// Recomputes smooth vertex normals from the triangles, like three's computeVertexNormals.
fn void BufferGeometry.compute_vertex_normals(&self)
{
    uint n = self.vertex_count();
    Vector3[] acc = mem::new_array(Vector3, n);
    defer free(acc);
    for (sz i = 0; i + 2 < self.indices.len(); i += 3)
    {
        uint a = self.indices[i];
        uint b = self.indices[i + 1];
        uint c = self.indices[i + 2];
        Vector3 pa = self.get_position(a);
        Vector3 pb = self.get_position(b);
        Vector3 pc = self.get_position(c);
        Vector3 fn_ = (pc - pb).cross(pa - pb);
        acc[a] += fn_;
        acc[b] += fn_;
        acc[c] += fn_;
    }
    for (uint i = 0; i < n; i++)
    {
        self.set_normal(i, acc[i].length() > 0 ? acc[i].normalize() : (Vector3){ 0, 1, 0 });
    }
    self.uploaded = false;
}

// Moves every vertex, like three's geometry.translate(x, y, z). Call before first render.
fn BufferGeometry* BufferGeometry.translate(&self, float x, float y, float z)
{
    for (sz o = 0; o < self.vertices.len(); o += FLOATS_PER_VERTEX)
    {
        self.vertices[o] += x;
        self.vertices[o + 1] += y;
        self.vertices[o + 2] += z;
    }
    self.uploaded = false;
    return self;
}

// Rotates every vertex (and normal) by an XYZ Euler, like rotateX/Y/Z combined.
fn BufferGeometry* BufferGeometry.rotate(&self, float x, float y, float z)
{
    Mat4 r = mat4_compose({ 0, 0, 0 }, { x, y, z }, { 1, 1, 1 });
    for (sz o = 0; o < self.vertices.len(); o += FLOATS_PER_VERTEX)
    {
        Vector3 p = r.transform_point({ self.vertices[o], self.vertices[o + 1], self.vertices[o + 2] });
        Vector3 n = r.transform_point({ self.vertices[o + 3], self.vertices[o + 4], self.vertices[o + 5] });
        self.vertices[o] = p.x;
        self.vertices[o + 1] = p.y;
        self.vertices[o + 2] = p.z;
        self.vertices[o + 3] = n.x;
        self.vertices[o + 4] = n.y;
        self.vertices[o + 5] = n.z;
    }
    self.uploaded = false;
    return self;
}
fn BufferGeometry* BufferGeometry.rotate_x(&self, float a) => self.rotate(a, 0, 0);
fn BufferGeometry* BufferGeometry.rotate_y(&self, float a) => self.rotate(0, a, 0);
fn BufferGeometry* BufferGeometry.rotate_z(&self, float a) => self.rotate(0, 0, a);

// Sets the vertex color of every vertex (used by materials with vertex_colors: true).
fn BufferGeometry* BufferGeometry.set_color(&self, Vector3 rgb)
{
    for (sz o = 0; o < self.vertices.len(); o += FLOATS_PER_VERTEX)
    {
        self.vertices[o + 8] = rgb.x;
        self.vertices[o + 9] = rgb.y;
        self.vertices[o + 10] = rgb.z;
    }
    self.uploaded = false;
    return self;
}

// Appends another geometry, transformed by `matrix` (like BufferGeometryUtils.mergeGeometries).
// Vertex colors of the appended vertices are multiplied by `tint`.
fn BufferGeometry* BufferGeometry.merge(&self, BufferGeometry* other, Mat4 matrix = mat4_identity(),
                                        Vector3 tint = { 1, 1, 1 })
{
    uint base = self.vertex_count();
    Mat4 nm = matrix.invert().transpose();
    nm[3] = { 0, 0, 0, 1 };
    for (sz o = 0; o < other.vertices.len(); o += FLOATS_PER_VERTEX)
    {
        Vector3 p = matrix.transform_point({ other.vertices[o], other.vertices[o + 1], other.vertices[o + 2] });
        Vector3 n = nm.transform_point({ other.vertices[o + 3], other.vertices[o + 4], other.vertices[o + 5] });
        if (n.length() > 0) n = n.normalize();
        Vector3 c = { other.vertices[o + 8], other.vertices[o + 9], other.vertices[o + 10] };
        self.add_vertex(p, n, { other.vertices[o + 6], other.vertices[o + 7] }, c * tint);
    }
    foreach (i : other.indices) self.indices.push(base + i);
    self.uploaded = false;
    return self;
}

// Frees CPU-side data (GPU buffers stay valid). Call after upload if the geometry is static.
fn void BufferGeometry.free_cpu(&self)
{
    self.vertices.free();
    self.indices.free();
}

// Uploads vertex/index data to the GPU (called by the renderer).
fn void BufferGeometry.upload(&self)
{
    if (self.uploaded) return;
    self.dispose();
    if (self.bounds_radius == 0) self.compute_bounding_sphere();
    self.index_count = (uint)self.indices.len();
    self.vbuf = sg::make_buffer(&&{
        .data = { .ptr = self.vertices.entries, .size = (usz)self.vertices.len() * float::size },
    });
    self.ibuf = sg::make_buffer(&&{
        .usage = { .index_buffer = true },
        .data = { .ptr = self.indices.entries, .size = (usz)self.indices.len() * uint::size },
    });
    self.uploaded = true;
}

// Builds the edge index buffer used by wireframe materials (called by the renderer).
fn void BufferGeometry.upload_wire(&self)
{
    if (self.wire_ibuf.id) return;
    // edge list for wireframe rendering
    List{uint} wire;
    defer wire.free();
    for (sz i = 0; i + 2 < self.indices.len(); i += 3)
    {
        uint a = self.indices[i];
        uint b = self.indices[i + 1];
        uint c = self.indices[i + 2];
        wire.push(a); wire.push(b);
        wire.push(b); wire.push(c);
        wire.push(c); wire.push(a);
    }
    self.wire_count = (uint)wire.len();
    self.wire_ibuf = sg::make_buffer(&&{
        .usage = { .index_buffer = true },
        .data = { .ptr = wire.entries, .size = (usz)wire.len() * uint::size },
    });
}

// Frees GPU buffers. Geometry is re-uploaded if rendered again.
fn void BufferGeometry.dispose(&self)
{
    if (self.vbuf.id) sg::destroy_buffer(self.vbuf);
    if (self.ibuf.id) sg::destroy_buffer(self.ibuf);
    if (self.wire_ibuf.id) sg::destroy_buffer(self.wire_ibuf);
    self.vbuf = {};
    self.ibuf = {};
    self.wire_ibuf = {};
    self.uploaded = false;
}

// Custom geometry from flat xyz positions (and optional triangle indices).
// Without indices, every three positions form a triangle. Normals are computed.
fn BufferGeometry* buffer_geometry(float[] positions = {}, uint[] indices = {})
{
    BufferGeometry* g = mem::new(BufferGeometry);
    g.init_lists();
    uint count = (uint)(positions.len / 3);
    for (uint i = 0; i < count; i++)
    {
        g.add_vertex({ positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2] }, { 0, 1, 0 }, { 0, 0 });
    }
    if (indices.len)
    {
        foreach (i : indices) g.indices.push(i);
    }
    else
    {
        for (uint i = 0; i < count; i++) g.indices.push(i);
    }
    if (count) g.compute_vertex_normals();
    return g;
}

// ----------------------------------------------------------------------------
// Built-in shapes (parameters and defaults match three.js)
// ----------------------------------------------------------------------------

struct BoxGeometry
{
    inline BufferGeometry geometry;
    float width, height, depth;
}

struct PlaneGeometry
{
    inline BufferGeometry geometry;
    float width, height;
}

struct SphereGeometry
{
    inline BufferGeometry geometry;
    float radius;
}

struct CylinderGeometry
{
    inline BufferGeometry geometry;
    float radius_top, radius_bottom, height;
}

alias ConeGeometry = CylinderGeometry;

struct TorusGeometry
{
    inline BufferGeometry geometry;
    float radius, tube;
}

// One face of a box/plane: a grid spanning u and v axes, offset along the normal.
fn void build_plane(BufferGeometry* g, Vector3 u_axis, Vector3 v_axis, Vector3 normal, Vector3 center,
                    float width, float height, int gx, int gy) @local
{
    uint base = g.vertex_count();
    for (int iy = 0; iy <= gy; iy++)
    {
        float v = (float)iy / gy;
        for (int ix = 0; ix <= gx; ix++)
        {
            float u = (float)ix / gx;
            Vector3 p = center + u_axis * ((u - 0.5f) * width) + v_axis * ((0.5f - v) * height);
            g.add_vertex(p, normal, { u, 1 - v });
        }
    }
    for (int iy = 0; iy < gy; iy++)
    {
        for (int ix = 0; ix < gx; ix++)
        {
            uint a = base + (uint)(ix + (gx + 1) * iy);
            uint b = base + (uint)(ix + (gx + 1) * (iy + 1));
            uint c = base + (uint)((ix + 1) + (gx + 1) * (iy + 1));
            uint d = base + (uint)((ix + 1) + (gx + 1) * iy);
            g.indices.push(a); g.indices.push(b); g.indices.push(d);
            g.indices.push(b); g.indices.push(c); g.indices.push(d);
        }
    }
}

fn BoxGeometry* box_geometry(float width = 1, float height = 1, float depth = 1,
                             int width_segments = 1, int height_segments = 1, int depth_segments = 1)
{
    BoxGeometry* g = mem::new(BoxGeometry);
    g.init_lists();
    g.width = width;
    g.height = height;
    g.depth = depth;
    float hw = width / 2;
    float hh = height / 2;
    float hd = depth / 2;
    // +x, -x, +y, -y, +z, -z
    build_plane(g, { 0, 0, -1 }, { 0, 1, 0 }, { 1, 0, 0 },  { hw, 0, 0 },  depth, height, depth_segments, height_segments);
    build_plane(g, { 0, 0, 1 },  { 0, 1, 0 }, { -1, 0, 0 }, { -hw, 0, 0 }, depth, height, depth_segments, height_segments);
    build_plane(g, { 1, 0, 0 },  { 0, 0, -1 }, { 0, 1, 0 }, { 0, hh, 0 },  width, depth, width_segments, depth_segments);
    build_plane(g, { 1, 0, 0 },  { 0, 0, 1 },  { 0, -1, 0 }, { 0, -hh, 0 }, width, depth, width_segments, depth_segments);
    build_plane(g, { 1, 0, 0 },  { 0, 1, 0 }, { 0, 0, 1 },  { 0, 0, hd },  width, height, width_segments, height_segments);
    build_plane(g, { -1, 0, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, -hd }, width, height, width_segments, height_segments);
    return g;
}

// A plane in the XY plane facing +Z (rotate it with mesh.rotation.x = -PI/2 for a floor).
fn PlaneGeometry* plane_geometry(float width = 1, float height = 1, int width_segments = 1, int height_segments = 1)
{
    PlaneGeometry* g = mem::new(PlaneGeometry);
    g.init_lists();
    g.width = width;
    g.height = height;
    build_plane(g, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0, 0 }, width, height, width_segments, height_segments);
    return g;
}

fn SphereGeometry* sphere_geometry(float radius = 1, int width_segments = 32, int height_segments = 16)
{
    SphereGeometry* g = mem::new(SphereGeometry);
    g.init_lists();
    g.radius = radius;
    width_segments = math::max(3, width_segments);
    height_segments = math::max(2, height_segments);
    for (int iy = 0; iy <= height_segments; iy++)
    {
        float v = (float)iy / height_segments;
        for (int ix = 0; ix <= width_segments; ix++)
        {
            float u = (float)ix / width_segments;
            Vector3 n = {
                -math::cos(u * 2 * PI) * math::sin(v * PI),
                math::cos(v * PI),
                math::sin(u * 2 * PI) * math::sin(v * PI),
            };
            g.add_vertex(n * radius, n, { u, 1 - v });
        }
    }
    int row = width_segments + 1;
    for (int iy = 0; iy < height_segments; iy++)
    {
        for (int ix = 0; ix < width_segments; ix++)
        {
            uint a = (uint)(iy * row + ix + 1);
            uint b = (uint)(iy * row + ix);
            uint c = (uint)((iy + 1) * row + ix);
            uint d = (uint)((iy + 1) * row + ix + 1);
            if (iy != 0) { g.indices.push(a); g.indices.push(b); g.indices.push(d); }
            if (iy != height_segments - 1) { g.indices.push(b); g.indices.push(c); g.indices.push(d); }
        }
    }
    return g;
}

fn CylinderGeometry* cylinder_geometry(float radius_top = 1, float radius_bottom = 1, float height = 1,
                                       int radial_segments = 32, int height_segments = 1, bool open_ended = false)
{
    CylinderGeometry* g = mem::new(CylinderGeometry);
    g.init_lists();
    g.radius_top = radius_top;
    g.radius_bottom = radius_bottom;
    g.height = height;
    radial_segments = math::max(3, radial_segments);
    height_segments = math::max(1, height_segments);
    float half = height / 2;
    float slope = (radius_bottom - radius_top) / height;

    // side
    int row = radial_segments + 1;
    for (int iy = 0; iy <= height_segments; iy++)
    {
        float v = (float)iy / height_segments;
        float r = v * (radius_bottom - radius_top) + radius_top;
        for (int ix = 0; ix <= radial_segments; ix++)
        {
            float u = (float)ix / radial_segments;
            float theta = u * 2 * PI;
            float s = math::sin(theta);
            float c = math::cos(theta);
            Vector3 n = ((Vector3){ s, slope, c }).normalize();
            g.add_vertex({ r * s, -v * height + half, r * c }, n, { u, 1 - v });
        }
    }
    for (int iy = 0; iy < height_segments; iy++)
    {
        for (int ix = 0; ix < radial_segments; ix++)
        {
            uint a = (uint)(iy * row + ix);
            uint b = (uint)((iy + 1) * row + ix);
            uint c = (uint)((iy + 1) * row + ix + 1);
            uint d = (uint)(iy * row + ix + 1);
            g.indices.push(a); g.indices.push(b); g.indices.push(d);
            g.indices.push(b); g.indices.push(c); g.indices.push(d);
        }
    }

    if (!open_ended)
    {
        for (int top = 1; top >= 0; top--)
        {
            float r = top ? radius_top : radius_bottom;
            if (r <= 0) continue;
            float y = top ? half : -half;
            float sign = top ? 1 : -1;
            uint center = g.vertex_count();
            g.add_vertex({ 0, y, 0 }, { 0, sign, 0 }, { 0.5f, 0.5f });
            for (int ix = 0; ix <= radial_segments; ix++)
            {
                float theta = (float)ix / radial_segments * 2 * PI;
                float s = math::sin(theta);
                float c = math::cos(theta);
                g.add_vertex({ r * s, y, r * c }, { 0, sign, 0 }, { s * 0.5f + 0.5f, c * 0.5f * sign + 0.5f });
            }
            for (int ix = 0; ix < radial_segments; ix++)
            {
                uint a = center + (uint)(1 + ix);
                uint b = center + (uint)(2 + ix);
                if (top) { g.indices.push(a); g.indices.push(b); g.indices.push(center); }
                else { g.indices.push(b); g.indices.push(a); g.indices.push(center); }
            }
        }
    }
    return g;
}

fn ConeGeometry* cone_geometry(float radius = 1, float height = 1, int radial_segments = 32,
                               int height_segments = 1, bool open_ended = false)
{
    return cylinder_geometry(0, radius, height, radial_segments, height_segments, open_ended);
}

fn TorusGeometry* torus_geometry(float radius = 1, float tube = 0.4, int radial_segments = 12, int tubular_segments = 48)
{
    TorusGeometry* g = mem::new(TorusGeometry);
    g.init_lists();
    g.radius = radius;
    g.tube = tube;
    for (int j = 0; j <= radial_segments; j++)
    {
        for (int i = 0; i <= tubular_segments; i++)
        {
            float u = (float)i / tubular_segments * 2 * PI;
            float v = (float)j / radial_segments * 2 * PI;
            Vector3 p = {
                (radius + tube * math::cos(v)) * math::cos(u),
                (radius + tube * math::cos(v)) * math::sin(u),
                tube * math::sin(v),
            };
            Vector3 center = { radius * math::cos(u), radius * math::sin(u), 0 };
            g.add_vertex(p, (p - center).normalize(), { (float)i / tubular_segments, (float)j / radial_segments });
        }
    }
    for (int j = 1; j <= radial_segments; j++)
    {
        for (int i = 1; i <= tubular_segments; i++)
        {
            uint a = (uint)((tubular_segments + 1) * j + i - 1);
            uint b = (uint)((tubular_segments + 1) * (j - 1) + i - 1);
            uint c = (uint)((tubular_segments + 1) * (j - 1) + i);
            uint d = (uint)((tubular_segments + 1) * j + i);
            g.indices.push(a); g.indices.push(b); g.indices.push(d);
            g.indices.push(b); g.indices.push(c); g.indices.push(d);
        }
    }
    return g;
}
