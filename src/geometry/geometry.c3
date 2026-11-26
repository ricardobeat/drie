// Indexed mesh geometry (`BufferGeometry`), lazy GPU buffer uploads and dynamic vertex
// streaming, and procedural shape builders (`box_geometry`, `plane_geometry`, `sphere_geometry`,
// `lathe_geometry`, `cylinder_geometry`, `cone_geometry`, `torus_geometry`). Vertices are
// interleaved as `FLOATS_PER_VERTEX` (12) floats: position (3), normal (3), UV (2), RGBA
// color (4), using right-handed Y-up coordinates and counter-clockwise front faces. CPU lists
// use the heap allocator (`mem`); GPU buffers are created by `Renderer.render` and freed by
// `BufferGeometry.dispose()`.
module drie;
import std::collections::list, std::math;
import sokol;

// Interleaved vertex stride in floats: position (3), normal (3), UV (2), RGBA color (4).
// Alpha is consumed by materials with `vertex_colors` enabled.
const int FLOATS_PER_VERTEX = 12;

// CPU vertex/index arrays, optional auxiliary attributes (skinning, UV1, tangents, morph
// targets), and lazily created GPU buffers for a `Mesh`.
struct BufferGeometry {
    // Interleaved vertex attributes (`FLOATS_PER_VERTEX` floats per vertex).
    List{float} vertices;
    // Vertex indices defining triangles or line topologies.
    List{uint} indices;
    // Per-vertex bone indices (`BONES_PER_VERTEX` per vertex) for `Skeleton`/`SkinnedMesh`;
    // empty until populated with `set_skin` or `skin_range`.
    List{ushort} skin_indices;
    // Per-vertex bone weights (`BONES_PER_VERTEX` per vertex) matching `skin_indices`.
    List{float} skin_weights;
    // Secondary UV set (`TEXCOORD_1`, 2 floats per vertex) read by textures with `channel == 1`
    // (e.g. baked light or AO maps); empty falls back to `(0, 0)`.
    List{float} uv1;
    // Per-vertex tangents (`x, y, z` along +U, `w` bitangent sign where `B = cross(N, T) * w`)
    // for normal and anisotropy maps; when empty, shaders derive the frame from screen gradients.
    List{float} tangents;
    // Morph target position deltas: `morph_targets * morph_vertex_count` vec4s of `(dx, dy, dz, 0)`.
    List{float} morph_positions;
    // Number of morph targets stored in `morph_positions`.
    int morph_targets;
    // Vertex count per morph target in `morph_positions`.
    int morph_vertex_count;
    // Active GPU vertex buffer (created lazily on first render).
    SgBuffer vbuf;
    // GPU buffer for secondary `uv1` coordinates.
    SgBuffer uv1buf;
    // GPU buffer for per-vertex `tangents`.
    SgBuffer tangent_buf;
    // GPU storage buffer holding `morph_positions`.
    SgBuffer morph_buf;
    // Storage view for `morph_buf`.
    SgView morph_view;
    // True once `morph_buf` matches `morph_positions`.
    bool morph_uploaded;
    // GPU triangle/segment index buffer.
    SgBuffer ibuf;
    // GPU edge-list index buffer built on demand by `upload_wire` for wireframe materials.
    SgBuffer wire_ibuf;
    // Number of indices in `wire_ibuf`.
    uint wire_count;
    // GPU segment-pair index buffer built by `upload_lines` for `Line` and `LineLoop`.
    SgBuffer line_ibuf;
    // Number of indices in `line_ibuf`.
    uint line_count;
    // Whether `line_ibuf` closes the loop (`LINE_LOOP` vs `LINE_STRIP`).
    bool line_loop;
    // Number of indices uploaded into `ibuf`.
    uint index_count;
    // True while GPU buffers match CPU data; clear to force full re-upload on next render.
    bool uploaded;
    // Marks frequently updated vertices (e.g. `drie::cloth`). Set before first upload and keep
    // vertex count fixed; after mutating `vertices` (and `tangents`), set `needs_update` or call
    // `vertices_changed` to stream them to the GPU once per frame before render passes.
    bool dynamic;
    // Dirty flag for a `dynamic` geometry; updates flagged after a frame's copy apply next frame.
    bool needs_update;
    // Double-buffered vertex buffer holding the previous frame's vertices for motion vectors.
    SgBuffer prev_vbuf;
    // Transient staging buffer used to copy `dynamic` CPU vertices (and tangents) to the GPU.
    SgBuffer staging;
    // Vertices are written directly on the GPU by compute kernels (`drie::gpu_cloth`) rather than
    // streamed from the CPU. Set before first upload alongside `dynamic`; both vertex buffers are
    // allocated as storage buffers and acquired via `gpu_vertices_target()`.
    bool gpu_vertices;
    // Storage view of `vbuf` when `gpu_vertices` is true.
    SgView vbuf_view;
    // Storage view of `prev_vbuf` when `gpu_vertices` is true.
    SgView prev_view;
    // `state.frame + 1` of the most recent dynamic or GPU vertex update.
    uint updated_frame;
    // Monotonic counter incremented on every GPU upload or dynamic vertex update.
    uint version;
    // Monotonic counter incremented only on full buffer rebuilds (`upload`).
    uint upload_version;
    // Local-space bounding sphere center computed on upload or via `compute_bounding_sphere`.
    Vector3 bounds_center;
    // Local-space bounding sphere radius.
    float bounds_radius;
}

// Configures all CPU attribute lists to use the heap allocator (`mem`) rather than temp memory.
fn void BufferGeometry.init_lists(&self) {
    self.vertices.allocator = mem;
    self.indices.allocator = mem;
    self.skin_indices.allocator = mem;
    self.skin_weights.allocator = mem;
    self.uv1.allocator = mem;
    self.tangents.allocator = mem;
    self.morph_positions.allocator = mem;
}

// Computes the local-space bounding sphere (`bounds_center`, `bounds_radius`) from vertex
// positions using the AABB midpoint as center.
fn void BufferGeometry.compute_bounding_sphere(&self) {
    Vector3 lo =  { 1e30f, 1e30f, 1e30f };
    Vector3 hi = -lo;
    uint n     =  self.vertex_count();
    if (!n) return;
    for (uint i = 0; i < n; i++) {
        Vector3 p = self.get_position(i);
        lo = math::min(lo, p);
        hi = math::max(hi, p);
    }
    self.bounds_center = (lo + hi) * 0.5f;
    float r2 = 0;
    for (uint i = 0; i < n; i++) {
        Vector3 d = self.get_position(i) - self.bounds_center;
        r2 = math::max(r2, d.dot(d));
    }
    self.bounds_radius = math::sqrt(r2);
}

// Replaces primary UVs with a tri-planar box projection chosen by each vertex's dominant normal
// axis, scaled by `tiles_per_unit`. Marks geometry for re-upload and returns `self`.
fn BufferGeometry* BufferGeometry.box_project_uv(&self, float tiles_per_unit = 1) {
    for (sz o = 0; o < self.vertices.len(); o += FLOATS_PER_VERTEX) {
        float[] v = self.vertices.entries[o:FLOATS_PER_VERTEX];
        float ax  = math::abs(v[3]);
        float ay  = math::abs(v[4]);
        float az  = math::abs(v[5]);
        Vector2 uv;
        if (ay >= ax && ay >= az) {
            uv = { v[0], v[2] };
        } else if (ax >= az) {
            uv = { v[2], -v[1] };
        } else {
            uv = { v[0], -v[1] };
        }
        v[6] = uv.x * tiles_per_unit;
        v[7] = uv.y * tiles_per_unit;
    }
    self.uploaded = false;
    return self;
}

// Appends one interleaved vertex `(p, n, uv, color, alpha)` and returns `self`.
fn BufferGeometry* BufferGeometry.add_vertex(
    &self,
    Vector3 p,
    Vector3 n,
    Vector2 uv,
    Vector3 color = { 1, 1, 1 },
    float alpha = 1
) {
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
    self.vertices.push(alpha);
    return self;
}

// Appends a counter-clockwise triangle `(a, b, c)` to `indices`, marks dirty, and returns `self`.
fn BufferGeometry* BufferGeometry.add_triangle(&self, uint a, uint b, uint c) {
    self.indices.push(a);
    self.indices.push(b);
    self.indices.push(c);
    self.uploaded = false;
    return self;
}

// Returns the number of vertices stored in `vertices`.
fn uint BufferGeometry.vertex_count(&self) => (uint)(self.vertices.len() / FLOATS_PER_VERTEX);

// Returns primary texture coordinates for vertex `i`.
fn Vector2 BufferGeometry.get_uv(&self, uint i) {
    sz at = (sz)i * FLOATS_PER_VERTEX + 6;
    return { self.vertices[at], self.vertices[at + 1] };
}

// Returns the local-space position of vertex `i`.
fn Vector3 BufferGeometry.get_position(&self, uint i) {
    sz o = (sz)i * FLOATS_PER_VERTEX;
    return { self.vertices[o], self.vertices[o + 1], self.vertices[o + 2] };
}

// Returns the local-space normal of vertex `i`.
fn Vector3 BufferGeometry.get_normal(&self, uint i) {
    sz o = (sz)i * FLOATS_PER_VERTEX + 3;
    return { self.vertices[o], self.vertices[o + 1], self.vertices[o + 2] };
}

// Sets vertex `i`'s RGB color channels (leaving alpha unchanged) and marks for re-upload.
fn void BufferGeometry.set_vertex_color(&self, uint i, Vector3 rgb) {
    sz o = (sz)i * FLOATS_PER_VERTEX;
    self.vertices[o + 8] = rgb.x;
    self.vertices[o + 9] = rgb.y;
    self.vertices[o + 10] = rgb.z;
    self.uploaded = false;
}

// Writes the local-space normal of vertex `i` without altering `uploaded` (call
// `vertices_changed()` when done).
fn void BufferGeometry.set_normal(&self, uint i, Vector3 n) {
    sz o = (sz)i * FLOATS_PER_VERTEX + 3;
    self.vertices[o] = n.x;
    self.vertices[o + 1] = n.y;
    self.vertices[o + 2] = n.z;
}

// Recomputes area-weighted smooth vertex normals from `indices` (falling back to `+Y` for
// degenerate vertices) and marks for re-upload.
fn void BufferGeometry.compute_vertex_normals(&self) {
    uint n        = self.vertex_count();
    Vector3[] acc = mem::new_array(Vector3, n);
    defer free(acc);
    for (sz i = 0; i + 2 < self.indices.len(); i += 3) {
        uint a      = self.indices[i];
        uint b      = self.indices[i + 1];
        uint c      = self.indices[i + 2];
        Vector3 pa  = self.get_position(a);
        Vector3 pb  = self.get_position(b);
        Vector3 pc  = self.get_position(c);
        Vector3 fn_ = (pc - pb).cross(pa - pb);
        acc[a] += fn_;
        acc[b] += fn_;
        acc[c] += fn_;
    }
    for (uint i = 0; i < n; i++) {
        self.set_normal(i, acc[i].length() > 0 ? acc[i].normalize() : (Vector3) { 0, 1, 0 });
    }
    self.uploaded = false;
}

// Translates every vertex position in place by `(x, y, z)`, marks for re-upload, and returns `self`.
fn BufferGeometry* BufferGeometry.translate(&self, float x, float y, float z) {
    for (sz o = 0; o < self.vertices.len(); o += FLOATS_PER_VERTEX) {
        self.vertices[o] += x;
        self.vertices[o + 1] += y;
        self.vertices[o + 2] += z;
    }
    self.uploaded = false;
    return self;
}

// Rotates every vertex position and normal in place by XYZ Euler angles `(x, y, z)` in radians,
// marks for re-upload, and returns `self`.
fn BufferGeometry* BufferGeometry.rotate(&self, float x, float y, float z) {
    Mat4 r = mat4_compose({ 0, 0, 0 }, { x, y, z }, { 1, 1, 1 });
    for (sz o = 0; o < self.vertices.len(); o += FLOATS_PER_VERTEX) {
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
// Rotates every vertex and normal by `a` radians around X and returns `self`.
fn BufferGeometry* BufferGeometry.rotate_x(&self, float a) => self.rotate(a, 0, 0);
// Rotates every vertex and normal by `a` radians around Y and returns `self`.
fn BufferGeometry* BufferGeometry.rotate_y(&self, float a) => self.rotate(0, a, 0);
// Rotates every vertex and normal by `a` radians around Z and returns `self`.
fn BufferGeometry* BufferGeometry.rotate_z(&self, float a) => self.rotate(0, 0, a);

// Sets RGB color channels on every vertex (leaving alpha unchanged) for materials with
// `vertex_colors = true`, marks for re-upload, and returns `self`.
fn BufferGeometry* BufferGeometry.set_color(&self, Vector3 rgb) {
    for (sz o = 0; o < self.vertices.len(); o += FLOATS_PER_VERTEX) {
        self.vertices[o + 8] = rgb.x;
        self.vertices[o + 9] = rgb.y;
        self.vertices[o + 10] = rgb.z;
    }
    self.uploaded = false;
    return self;
}

// Appends `other`'s vertices and indices transformed by `matrix` (normals via inverse-transpose),
// multiplying appended RGB colors by `tint`. Auxiliary attributes (`uv1`, `tangents`, skinning)
// are not copied. Returns `self`.
fn BufferGeometry* BufferGeometry.merge(
    &self,
    BufferGeometry* other,
    Mat4 matrix = mat4_identity(),
    Vector3 tint = { 1, 1, 1 }
) {
    uint base = self.vertex_count();
    Mat4 nm   = matrix.invert().transpose();
    nm[3] = { 0, 0, 0, 1 };
    for (sz o = 0; o < other.vertices.len(); o += FLOATS_PER_VERTEX) {
        Vector3 p = matrix.transform_point({ other.vertices[o], other.vertices[o + 1], other.vertices[o + 2] });
        Vector3 n = nm.transform_point({ other.vertices[o + 3], other.vertices[o + 4], other.vertices[o + 5] });
        if (n.length() > 0) n = n.normalize();
        Vector3 c = { other.vertices[o + 8], other.vertices[o + 9], other.vertices[o + 10] };
        self.add_vertex(p, n, { other.vertices[o + 6], other.vertices[o + 7] }, c * tint, other.vertices[o + 11]);
    }
    foreach (i : other.indices) self.indices.push(base + i);
    self.uploaded = false;
    return self;
}

// Frees CPU `vertices` and `indices` lists while leaving uploaded GPU buffers intact (safe for
// static geometry after `upload()`).
fn void BufferGeometry.free_cpu(&self) {
    self.vertices.free();
    self.indices.free();
}

// Creates GPU vertex, index, UV1, tangent, morph, and dynamic staging buffers from CPU data.
// Called lazily by `Renderer.render`; no-op when already `uploaded`.
fn void BufferGeometry.upload(&self) {
    if (self.uploaded) return;
    self.version++;
    self.upload_version++;
    self.dispose();
    // Skip empty buffers: Metal rejects zero-byte buffer creation.
    if (!self.indices.len() || !self.vertices.len()) return;
    if (self.bounds_radius == 0) self.compute_bounding_sphere();
    self.index_count = (uint)self.indices.len();
    SgBufferUsage dyn = {
        .vertex_buffer = true,
        .copy_dst = self.dynamic && !self.gpu_vertices,
        .storage_buffer = self.gpu_vertices
    };
    usz vbytes = (usz)self.vertices.len() * float::size;
    self.vbuf = sg::make_buffer(&&{ .usage = dyn, .data = { .ptr = self.vertices.entries, .size = vbytes } });
    self.ibuf = sg::make_buffer(
        &&{
            .usage = { .index_buffer = true },
            .data = { .ptr = self.indices.entries, .size = (usz)self.indices.len() * uint::size },
        }
    );
    if ((usz)self.uv1.len() >= (usz)self.vertex_count() * 2) {
        self.uv1buf = sg::make_buffer(
            &&{ .data = { .ptr = self.uv1.entries, .size = (usz)self.vertex_count() * 2 * float::size } }
        );
    }
    usz tbytes    = (usz)self.vertex_count() * 4 * float::size;
    bool tangents = (usz)self.tangents.len() >= (usz)self.vertex_count() * 4;
    if (tangents) {
        self.tangent_buf = sg::make_buffer(
            &&{ .usage = dyn, .data = { .ptr = self.tangents.entries, .size = tbytes } }
        );
    }
    if (self.dynamic && self.gpu_vertices) {
        self.prev_vbuf = sg::make_buffer(&&{ .usage = dyn, .data = { .ptr = self.vertices.entries, .size = vbytes } });
        self.vbuf_view = sg::make_view(&&{ .storage_buffer = { .buffer = self.vbuf } });
        self.prev_view = sg::make_view(&&{ .storage_buffer = { .buffer = self.prev_vbuf } });
        self.updated_frame = 0;
    } else if (self.dynamic) {
        self.prev_vbuf = sg::make_buffer(&&{ .usage = dyn, .data = { .ptr = self.vertices.entries, .size = vbytes } });
        self.staging = sg::make_buffer(
            &&{
                .usage = { .staging_buffer = true, .write_transient = true, .copy_src = true },
                .size = vbytes + (tangents ? tbytes : 0),
                .label = "drie-dynamic-staging",
            }
        );
        self.needs_update = false;
        self.updated_frame = 0;
        add_dynamic(self);
    }
    self.upload_morph();
    self.uploaded = true;
}

// Marks vertex data dirty: flags `needs_update` for an uploaded `dynamic` geometry, or clears
// `uploaded` on static geometries so buffers rebuild on next draw.
fn void BufferGeometry.vertices_changed(&self) {
    if (self.dynamic && self.uploaded) {
        self.needs_update = true;
    } else {
        self.uploaded = false;
    }
}

// Returns the vertex buffer holding the previous frame's positions for MRT motion vectors
// (`prev_vbuf` if updated this frame, otherwise `vbuf`).
fn SgBuffer BufferGeometry.prev_vertices(&self) {
    return self.dynamic && self.updated_frame == state.frame + 1 && self.prev_vbuf.id ? self.prev_vbuf : self.vbuf;
}

// Returns true if this `dynamic` geometry received new vertices during the current frame.
fn bool BufferGeometry.updated_now(&self) => self.dynamic && self.updated_frame == state.frame + 1;

// Returns the storage view a compute kernel writes this frame's vertices into (`gpu_vertices`
// and `dynamic` required), uploading first if needed. On the first call per frame, rotates the
// double buffer so `prev_vbuf` retains last frame's vertices and increments `version`; kernels
// must overwrite every vertex (the target holds vertices from two frames ago).
fn SgView BufferGeometry.gpu_vertices_target(&self) {
    if (!self.gpu_vertices || !self.dynamic) return { };
    self.upload();
    if (!self.vbuf_view.id) return { };
    if (self.updated_frame != state.frame + 1) {
        self.updated_frame = state.frame + 1;
        self.version++;
        SgBuffer older = self.prev_vbuf;
        self.prev_vbuf = self.vbuf;
        self.vbuf = older;
        SgView older_view = self.prev_view;
        self.prev_view = self.vbuf_view;
        self.vbuf_view = older_view;
    }
    return self.vbuf_view;
}

// Streams a dirty `dynamic` geometry's CPU `vertices` (and `tangents`) through `staging` into
// the rotated vertex buffer outside any render pass, at most once per frame. Re-flags full
// upload if byte size or tangent presence changed; returns bytes copied.
fn usz BufferGeometry.update_vertices(&self) {
    if (!self.needs_update || !self.uploaded || !self.vbuf.id || self.gpu_vertices) return 0;
    if (self.updated_frame == state.frame + 1) return 0;
    usz vbytes    = (usz)self.vertices.len() * float::size;
    usz tbytes    = self.tangent_buf.id ? (usz)self.vertex_count() * 4 * float::size : 0;
    bool tangents = (usz)self.tangents.len() >= (usz)self.vertex_count() * 4;
    if (vbytes != sg::query_buffer_size(self.vbuf) || tangents != (tbytes != 0)) {
        // Vertex count or tangent layout changed; rebuild buffers on the next draw.
        self.uploaded = false;
        self.needs_update = false;
        return 0;
    }
    self.needs_update = false;
    self.updated_frame = state.frame + 1;
    self.version++;
    SgBuffer older = self.prev_vbuf;
    self.prev_vbuf = self.vbuf;
    self.vbuf = older;
    sg::write_buffer_transient(
        &&{ .src.data = { .ptr = self.vertices.entries, .size = vbytes }, .dst.buffer = self.staging }
    );
    if (tbytes) {
        sg::write_buffer_transient(
            &&{
                .src.data = { .ptr = self.tangents.entries, .size = tbytes },
                .dst = { .buffer = self.staging, .offset = vbytes }
            }
        );
    }
    sg::copy_buffer_to_buffer(&&{ .src.buffer = self.staging, .dst.buffer = self.vbuf, .size = vbytes });
    if (tbytes) {
        sg::copy_buffer_to_buffer(
            &&{ .src = { .buffer = self.staging, .offset = vbytes }, .dst.buffer = self.tangent_buf, .size = tbytes }
        );
    }
    return vbytes + tbytes;
}

// Uploaded dynamic geometries tracked for per-frame `update_dynamic` transfers.
List{BufferGeometry*} dynamic_geometries @private;

fn void add_dynamic(BufferGeometry* g) @private {
    if (!dynamic_geometries.allocator) dynamic_geometries.allocator = mem;
    foreach (d : dynamic_geometries) if (d == g) return;
    dynamic_geometries.push(g);
}

fn void remove_dynamic(BufferGeometry* g) @private {
    foreach (i, d : dynamic_geometries) {
        if (d == g) {
            dynamic_geometries.remove_at(i);
            return;
        }
    }
}

// Copies all dirty `dynamic` geometries to the GPU before render passes begin; returns count
// updated and accumulates copied bytes into `*bytes`.
fn int update_dynamic(usz* bytes) @private {
    int n;
    foreach (g : dynamic_geometries) {
        usz b = g.update_vertices();
        if (!b) continue;
        n++;
        *bytes += b;
    }
    return n;
}

// Stores position morph target deltas (`target_count * vertex_count` vec4s of `(dx, dy, dz, 0)`)
// and marks the GPU morph buffer dirty.
fn void BufferGeometry.set_morph_positions(&self, float[] deltas, int target_count, int vertex_count) {
    self.morph_positions.clear();
    if (target_count <= 0 || vertex_count <= 0 || deltas.len < (usz)target_count * (usz)vertex_count * 4) {
        self.morph_targets = 0;
        self.morph_vertex_count = 0;
        self.morph_uploaded = false;
        return;
    }
    sz n = (sz)target_count * (sz)vertex_count * 4;
    self.morph_positions.reserve(n);
    for (sz i = 0; i < n; i++) self.morph_positions.push(deltas[i]);
    self.morph_targets = target_count;
    self.morph_vertex_count = vertex_count;
    self.morph_uploaded = false;
}

// Uploads `morph_positions` to a GPU storage buffer and view (called by `upload` / renderer).
fn void BufferGeometry.upload_morph(&self) {
    if (self.morph_uploaded) return;
    if (self.morph_view.id) sg::destroy_view(self.morph_view);
    if (self.morph_buf.id) sg::destroy_buffer(self.morph_buf);
    self.morph_view = { };
    self.morph_buf = { };
    if (self.morph_targets > 0 && self.morph_positions.len() > 0) {
        self.morph_buf = sg::make_buffer(
            &&{
                .usage = { .storage_buffer = true },
                .data = { .ptr = self.morph_positions.entries, .size = (usz)self.morph_positions.len() * float::size },
            }
        );
        self.morph_view = sg::make_view(&&{ .storage_buffer = { .buffer = self.morph_buf } });
    }
    self.morph_uploaded = true;
}

// Builds and uploads the triangle-edge index buffer (`wire_ibuf`) used by wireframe materials.
fn void BufferGeometry.upload_wire(&self) {
    if (self.wire_ibuf.id) return;
    List{uint} wire;
    defer wire.free();
    for (sz i = 0; i + 2 < self.indices.len(); i += 3) {
        uint a = self.indices[i];
        uint b = self.indices[i + 1];
        uint c = self.indices[i + 2];
        wire.push(a);
        wire.push(b);
        wire.push(b);
        wire.push(c);
        wire.push(c);
        wire.push(a);
    }
    self.wire_count = (uint)wire.len();
    self.wire_ibuf = sg::make_buffer(
        &&{
            .usage = { .index_buffer = true },
            .data = { .ptr = wire.entries, .size = (usz)wire.len() * uint::size },
        }
    );
}

// Builds the segment-pair index buffer (`line_ibuf`) for `Line` (`loop == false`) or `LineLoop`
// (`loop == true`). `LineSegments` renders `ibuf` directly without calling this.
fn void BufferGeometry.upload_lines(&self, bool loop) {
    if (self.line_ibuf.id && self.line_loop == loop) return;
    if (self.line_ibuf.id) sg::destroy_buffer(self.line_ibuf);
    self.line_ibuf = { };
    self.line_loop = loop;
    List{uint} pairs;
    defer pairs.free();
    sz n = self.indices.len();
    for (sz i = 0; i + 1 < n; i++) {
        pairs.push(self.indices[i]);
        pairs.push(self.indices[i + 1]);
    }
    if (loop && n > 2) {
        pairs.push(self.indices[n - 1]);
        pairs.push(self.indices[0]);
    }
    self.line_count = (uint)pairs.len();
    if (!self.line_count) return;
    self.line_ibuf = sg::make_buffer(
        &&{
            .usage = { .index_buffer = true },
            .data = { .ptr = pairs.entries, .size = (usz)pairs.len() * uint::size },
        }
    );
}

// Destroys all GPU buffers and views and clears `uploaded`; geometry re-uploads if drawn again.
fn void BufferGeometry.dispose(&self) {
    if (self.vbuf_view.id) sg::destroy_view(self.vbuf_view);
    if (self.prev_view.id) sg::destroy_view(self.prev_view);
    self.vbuf_view = { };
    self.prev_view = { };
    if (self.vbuf.id) sg::destroy_buffer(self.vbuf);
    if (self.prev_vbuf.id) sg::destroy_buffer(self.prev_vbuf);
    if (self.staging.id) sg::destroy_buffer(self.staging);
    self.prev_vbuf = { };
    self.staging = { };
    remove_dynamic(self);
    if (self.ibuf.id) sg::destroy_buffer(self.ibuf);
    if (self.wire_ibuf.id) sg::destroy_buffer(self.wire_ibuf);
    if (self.line_ibuf.id) sg::destroy_buffer(self.line_ibuf);
    if (self.uv1buf.id) sg::destroy_buffer(self.uv1buf);
    self.uv1buf = { };
    if (self.tangent_buf.id) sg::destroy_buffer(self.tangent_buf);
    self.tangent_buf = { };
    if (self.morph_view.id) sg::destroy_view(self.morph_view);
    if (self.morph_buf.id) sg::destroy_buffer(self.morph_buf);
    self.morph_view = { };
    self.morph_buf = { };
    self.morph_uploaded = false;
    self.vbuf = { };
    self.ibuf = { };
    self.wire_ibuf = { };
    self.line_ibuf = { };
    self.line_count = 0;
    self.uploaded = false;
}

// Allocates a `BufferGeometry` from flat XYZ `positions` and optional triangle `indices`
// (without `indices`, every three vertices form a triangle) and computes smooth normals.
fn BufferGeometry* buffer_geometry(float[] positions = { }, uint[] indices = { }) {
    BufferGeometry* g = mem::new(BufferGeometry);
    g.init_lists();
    uint count = (uint)(positions.len / 3);
    for (uint i = 0; i < count; i++) {
        g.add_vertex({ positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2] }, { 0, 1, 0 }, { 0, 0 });
    }
    if (indices.len) {
        foreach (i : indices) g.indices.push(i);
    } else {
        for (uint i = 0; i < count; i++) g.indices.push(i);
    }
    if (count) g.compute_vertex_normals();
    return g;
}

// ---- Built-in shapes

// Axis-aligned box geometry centered at the origin.
struct BoxGeometry {
    inline BufferGeometry geometry;
    // Box dimensions along X, Y, and Z in meters.
    float width, height, depth;
}

// Subdivided rectangular quad in the XY plane facing +Z.
struct PlaneGeometry {
    inline BufferGeometry geometry;
    // Plane extent along X and Y in meters.
    float width, height;
}

// UV-mapped sphere centered at the origin.
struct SphereGeometry {
    inline BufferGeometry geometry;
    // Sphere radius in meters.
    float radius;
}

// Tapered cylinder centered at the origin along Y, with optional top and bottom caps.
struct CylinderGeometry {
    inline BufferGeometry geometry;
    // Top cap radius, bottom cap radius, and total height along Y in meters.
    float radius_top, radius_bottom, height;
}

// Cone geometry represented as a `CylinderGeometry` with `radius_top == 0`.
alias ConeGeometry = CylinderGeometry;

// Torus centered at the origin in the XY plane.
struct TorusGeometry {
    inline BufferGeometry geometry;
    // Major ring radius and minor tube cross-section radius in meters.
    float radius, tube;
}

// Builds one subdivided face spanning `u_axis` and `v_axis` centered at `center`.
fn void build_plane(
    BufferGeometry* g,
    Vector3 u_axis,
    Vector3 v_axis,
    Vector3 normal,
    Vector3 center,
    float width,
    float height,
    int gx,
    int gy
) @local {
    uint base = g.vertex_count();
    for (int iy = 0; iy <= gy; iy++) {
        float v = (float)iy / gy;
        for (int ix = 0; ix <= gx; ix++) {
            float u   = (float)ix / gx;
            Vector3 p = center + u_axis * ((u - 0.5f) * width) + v_axis * ((0.5f - v) * height);
            g.add_vertex(p, normal, { u, 1 - v });
        }
    }
    for (int iy = 0; iy < gy; iy++) {
        for (int ix = 0; ix < gx; ix++) {
            uint a = base + (uint)(ix + (gx + 1) * iy);
            uint b = base + (uint)(ix + (gx + 1) * (iy + 1));
            uint c = base + (uint)((ix + 1) + (gx + 1) * (iy + 1));
            uint d = base + (uint)((ix + 1) + (gx + 1) * iy);
            g.indices.push(a);
            g.indices.push(b);
            g.indices.push(d);
            g.indices.push(b);
            g.indices.push(c);
            g.indices.push(d);
        }
    }
}

// Allocates a centered `BoxGeometry` with per-axis dimensions and face subdivision counts.
fn BoxGeometry* box_geometry(
    float width = 1,
    float height = 1,
    float depth = 1,
    int width_segments = 1,
    int height_segments = 1,
    int depth_segments = 1
) {
    BoxGeometry* g = mem::new(BoxGeometry);
    g.init_lists();
    g.width = width;
    g.height = height;
    g.depth = depth;
    float hw = width / 2;
    float hh = height / 2;
    float hd = depth / 2;
    // +x, -x, +y, -y, +z, -z
    build_plane(
        g,
        { 0, 0, -1 },
        { 0, 1, 0 },
        { 1, 0, 0 },
        { hw, 0, 0 },
        depth,
        height,
        depth_segments,
        height_segments
    );
    build_plane(
        g,
        { 0, 0, 1 },
        { 0, 1, 0 },
        { -1, 0, 0 },
        { -hw, 0, 0 },
        depth,
        height,
        depth_segments,
        height_segments
    );
    build_plane(g, { 1, 0, 0 }, { 0, 0, -1 }, { 0, 1, 0 }, { 0, hh, 0 }, width, depth, width_segments, depth_segments);
    build_plane(g, { 1, 0, 0 }, { 0, 0, 1 }, { 0, -1, 0 }, { 0, -hh, 0 }, width, depth, width_segments, depth_segments);
    build_plane(g, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0, hd }, width, height, width_segments, height_segments);
    build_plane(
        g,
        { -1, 0, 0 },
        { 0, 1, 0 },
        { 0, 0, -1 },
        { 0, 0, -hd },
        width,
        height,
        width_segments,
        height_segments
    );
    return g;
}

// Allocates a `PlaneGeometry` in the XY plane facing +Z (rotate by `-PI / 2` around X for a floor).
fn PlaneGeometry* plane_geometry(float width = 1, float height = 1, int width_segments = 1, int height_segments = 1) {
    PlaneGeometry* g = mem::new(PlaneGeometry);
    g.init_lists();
    g.width = width;
    g.height = height;
    build_plane(g, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0, 0 }, width, height, width_segments, height_segments);
    return g;
}

// Allocates a UV `SphereGeometry` centered at the origin (`width_segments >= 3`,
// `height_segments >= 2`).
fn SphereGeometry* sphere_geometry(float radius = 1, int width_segments = 32, int height_segments = 16) {
    SphereGeometry* g = mem::new(SphereGeometry);
    g.init_lists();
    g.radius = radius;
    width_segments = math::max(3, width_segments);
    height_segments = math::max(2, height_segments);
    for (int iy = 0; iy <= height_segments; iy++) {
        float v = (float)iy / height_segments;
        for (int ix = 0; ix <= width_segments; ix++) {
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
    for (int iy = 0; iy < height_segments; iy++) {
        for (int ix = 0; ix < width_segments; ix++) {
            uint a = (uint)(iy * row + ix + 1);
            uint b = (uint)(iy * row + ix);
            uint c = (uint)((iy + 1) * row + ix);
            uint d = (uint)((iy + 1) * row + ix + 1);
            if (iy != 0) {
                g.indices.push(a);
                g.indices.push(b);
                g.indices.push(d);
            }
            if (iy != height_segments - 1) {
                g.indices.push(b);
                g.indices.push(c);
                g.indices.push(d);
            }
        }
    }
    return g;
}

// Allocates a surface of revolution around +Y from `(radius, height)` profile `points` (ordered
// bottom to top for outward-facing normals, `segments >= 3`).
fn BufferGeometry* lathe_geometry(Vector2[] points, int segments = 12) {
    BufferGeometry* g = buffer_geometry();
    segments = math::max(3, segments);
    int rows = (int)points.len;
    for (int i = 0; i <= segments; i++) {
        float u = (float)i / segments;
        float s = math::sin(u * 2 * PI);
        float c = math::cos(u * 2 * PI);
        for (int j = 0; j < rows; j++) {
            // Compute outward profile normal perpendicular to the local tangent.
            Vector2 along = points[math::min(j + 1, rows - 1)] - points[math::max(j - 1, 0)];
            Vector3 n     = ((Vector3) { along.y * s, -along.x, along.y * c }).normalize();
            g.add_vertex(
                { points[j].x * s, points[j].y, points[j].x * c },
                n,
                { u, (float)j / math::max(rows - 1, 1) }
            );
        }
    }
    for (int i = 0; i < segments; i++) {
        for (int j = 0; j + 1 < rows; j++) {
            uint a = (uint)(i * rows + j);
            uint b = a + (uint)rows;
            g.indices.push(a);
            g.indices.push(b);
            g.indices.push(a + 1);
            g.indices.push(b);
            g.indices.push(b + 1);
            g.indices.push(a + 1);
        }
    }
    return g;
}

// Allocates a `CylinderGeometry` centered along Y (`open_ended` omits top and bottom caps).
fn CylinderGeometry* cylinder_geometry(
    float radius_top = 1,
    float radius_bottom = 1,
    float height = 1,
    int radial_segments = 32,
    int height_segments = 1,
    bool open_ended = false
) {
    CylinderGeometry* g = mem::new(CylinderGeometry);
    g.init_lists();
    g.radius_top = radius_top;
    g.radius_bottom = radius_bottom;
    g.height = height;
    radial_segments = math::max(3, radial_segments);
    height_segments = math::max(1, height_segments);
    float half  = height / 2;
    float slope = (radius_bottom - radius_top) / height;

    // Side surface.
    int row = radial_segments + 1;
    for (int iy = 0; iy <= height_segments; iy++) {
        float v = (float)iy / height_segments;
        float r = v * (radius_bottom - radius_top) + radius_top;
        for (int ix = 0; ix <= radial_segments; ix++) {
            float u     = (float)ix / radial_segments;
            float theta = u * 2 * PI;
            float s     = math::sin(theta);
            float c     = math::cos(theta);
            Vector3 n   = ((Vector3) { s, slope, c }).normalize();
            g.add_vertex({ r * s, -v * height + half, r * c }, n, { u, 1 - v });
        }
    }
    for (int iy = 0; iy < height_segments; iy++) {
        for (int ix = 0; ix < radial_segments; ix++) {
            uint a = (uint)(iy * row + ix);
            uint b = (uint)((iy + 1) * row + ix);
            uint c = (uint)((iy + 1) * row + ix + 1);
            uint d = (uint)(iy * row + ix + 1);
            g.indices.push(a);
            g.indices.push(b);
            g.indices.push(d);
            g.indices.push(b);
            g.indices.push(c);
            g.indices.push(d);
        }
    }

    if (!open_ended) {
        for (int top = 1; top >= 0; top--) {
            float r = top ? radius_top : radius_bottom;
            if (r <= 0) continue;
            float y     = top ? half : -half;
            float sign  = top ? 1 : -1;
            uint center = g.vertex_count();
            g.add_vertex({ 0, y, 0 }, { 0, sign, 0 }, { 0.5f, 0.5f });
            for (int ix = 0; ix <= radial_segments; ix++) {
                float theta = (float)ix / radial_segments * 2 * PI;
                float s     = math::sin(theta);
                float c     = math::cos(theta);
                g.add_vertex({ r * s, y, r * c }, { 0, sign, 0 }, { s * 0.5f + 0.5f, c * 0.5f * sign + 0.5f });
            }
            for (int ix = 0; ix < radial_segments; ix++) {
                uint a = center + (uint)(1 + ix);
                uint b = center + (uint)(2 + ix);
                if (top) {
                    g.indices.push(a);
                    g.indices.push(b);
                    g.indices.push(center);
                } else {
                    g.indices.push(b);
                    g.indices.push(a);
                    g.indices.push(center);
                }
            }
        }
    }
    return g;
}

// Allocates a `ConeGeometry` (`cylinder_geometry` with `radius_top = 0`).
fn ConeGeometry* cone_geometry(
    float radius = 1,
    float height = 1,
    int radial_segments = 32,
    int height_segments = 1,
    bool open_ended = false
) {
    return cylinder_geometry(0, radius, height, radial_segments, height_segments, open_ended);
}

// Allocates a `TorusGeometry` in the XY plane with major `radius` and cross-section `tube`.
fn TorusGeometry* torus_geometry(
    float radius = 1,
    float tube = 0.4,
    int radial_segments = 12,
    int tubular_segments = 48
) {
    TorusGeometry* g = mem::new(TorusGeometry);
    g.init_lists();
    g.radius = radius;
    g.tube = tube;
    for (int j = 0; j <= radial_segments; j++) {
        for (int i = 0; i <= tubular_segments; i++) {
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
    for (int j = 1; j <= radial_segments; j++) {
        for (int i = 1; i <= tubular_segments; i++) {
            uint a = (uint)((tubular_segments + 1) * j + i - 1);
            uint b = (uint)((tubular_segments + 1) * (j - 1) + i - 1);
            uint c = (uint)((tubular_segments + 1) * (j - 1) + i);
            uint d = (uint)((tubular_segments + 1) * j + i);
            g.indices.push(a);
            g.indices.push(b);
            g.indices.push(d);
            g.indices.push(b);
            g.indices.push(c);
            g.indices.push(d);
        }
    }
    return g;
}