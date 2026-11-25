// BufferGeometry.dynamic: a quad whose vertices change, drawn into a target with MRT, its
// picture and its motion vectors read back after each frame.
//   - marked (needs_update), the next render copies the vertices once and shows the new shape;
//     version goes up by one
//   - changed but not marked, the GPU keeps the old shape and nothing is copied
//   - marked again after the frame has copied it, the change waits for the next frame and is
//     shown then
//   - the frame it moves, the motion vectors carry the move; the next frame, still, they are 0
//   - with tangents, they are copied with the vertices
//   - a cloth pushed by a capsule writes motion vectors where it moves, and a still cloth copies
//     nothing
//
//   just test dynamic
module dynamic_test;
import drie, drie::cloth, sokol, drie::readback;
import std::io, std::math;

const int W = 160;
const int H = 120;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn float half_to_float(ushort h) {
    uint sign = (uint)(h >> 15) & 1;
    uint e    = (uint)(h >> 10) & 0x1f;
    uint m    = (uint)h & 0x3ff;
    float v;
    if (e == 0) {
        v = (float)m / 1024.0f / 16384.0f;
    } else if (e == 31) {
        v = 65504;
    } else {
        v = (1.0f + (float)m / 1024.0f) * (float)math::pow(2.0, (double)e - 15);
    }
    return sign ? -v : v;
}

RenderTarget* target;
Renderer* renderer;
char[] color_px;
char[] velocity_px;

// Renders, commits (unless `commit` is false) and reads back the picture and the motion vectors.
fn void frame(Scene* scene, Camera* camera, bool commit = true) {
    renderer.render(scene, camera);
    if (!commit) return;
    drie::commit();
    readback::read_image(target.texture_image, W, H, color_px.ptr, 8);
    readback::read_image(target.mrt_color[1], W, H, velocity_px.ptr, 4);
}

// The red of the picture at (x, y) (0..1, y down).
fn float red_at(float x, float y) {
    usz i = ((usz)(y * H) * W + (usz)(x * W)) * 8;
    return half_to_float((ushort)color_px[i] | (ushort)color_px[i + 1] << 8);
}

fn Vector2 velocity_at(float x, float y) {
    usz i = ((usz)(y * H) * W + (usz)(x * W)) * 4;
    return {
        half_to_float((ushort)velocity_px[i] | (ushort)velocity_px[i + 1] << 8),
        half_to_float((ushort)velocity_px[i + 2] | (ushort)velocity_px[i + 3] << 8)
    };
}

// The largest motion vector over the picture, in pixels.
fn float largest_motion() {
    float most;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            Vector2 v = velocity_at((x + 0.5f) / W, (y + 0.5f) / H);
            most = math::max(most, math::sqrt(v.x * W * v.x * W + v.y * H * v.y * H));
        }
    }
    return most;
}

// Puts the quad's four vertices `dx` to the right of where they started.
fn void place(BufferGeometry* g, float dx) {
    float[2][4] corners = { { -0.9, -0.5 }, { -0.1, -0.5 }, { -0.1, 0.5 }, { -0.9, 0.5 } };
    foreach (i, c : corners) g.vertices[(usz)i * drie::FLOATS_PER_VERTEX] = c[0] + dx;
}

fn int main(String[] args) {
    renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    target = drie::render_target(W, H, 1, RGBA16F, mrt: true);
    renderer.set_render_target(target);
    color_px = mem::new_array(char, (sz)W * H * 8);
    velocity_px = mem::new_array(char, (sz)W * H * 4);
    Scene* scene = drie::scene();
    scene.background = 0x000000;
    OrthographicCamera* camera = drie::orthographic_camera(-1, 1, 0.75f, -0.75f, 0.1f, 10);
    camera.position = { 0, 0, 5 };

    BufferGeometry* g = drie::buffer_geometry();
    g.add_vertex({ -0.9, -0.5, 0 }, { 0, 0, 1 }, { 0, 0 });
    g.add_vertex({ -0.1, -0.5, 0 }, { 0, 0, 1 }, { 1, 0 });
    g.add_vertex({ -0.1, 0.5, 0 }, { 0, 0, 1 }, { 1, 1 });
    g.add_vertex({ -0.9, 0.5, 0 }, { 0, 0, 1 }, { 0, 1 });
    g.add_triangle(0, 1, 2);
    g.add_triangle(0, 2, 3);
    g.dynamic = true;
    // a large bounding sphere: the quad moves without it being worked out again
    g.bounds_radius = 10;
    Mesh* quad = drie::mesh(g, drie::mesh_basic_material(color: 0xff0000, side: DOUBLE_SIDE));
    scene.add(quad);

    frame(scene, camera);
    frame(scene, camera);
    expect(red_at(0.25, 0.5) > 0.5 && red_at(0.75, 0.5) < 0.1, "drawn where it starts, on the left");
    uint v0 = g.version;

    // marked: copied, shown on the right, moving
    place(g, 1);
    g.needs_update = true;
    frame(scene, camera);
    usz bytes = renderer.info.update_bytes;
    expect(renderer.info.updates == 1 && bytes == 4 * drie::FLOATS_PER_VERTEX * float::size,
           string::tformat("marked: one geometry copied (%d), %d bytes", renderer.info.updates, bytes));
    expect(red_at(0.75, 0.5) > 0.5 && red_at(0.25, 0.5) < 0.1, "marked: the new shape is drawn");
    expect(g.version == v0 + 1, string::tformat("marked: version went up by one (%d to %d)", v0, g.version));
    Vector2 v = velocity_at(0.75, 0.5);
    expect(math::abs(v.x - 0.5f) < 0.01f && math::abs(v.y) < 0.01f,
           string::tformat("marked: its motion vectors carry the move (%.3f, %.3f; 0.5, 0 expected)", v.x, v.y));

    // still: no copy, no motion
    frame(scene, camera);
    expect(renderer.info.updates == 0 && largest_motion() < 0.01f,
           string::tformat("still: nothing copied, no motion (%.4f px)", largest_motion()));

    // changed without the mark: the GPU keeps the old shape
    place(g, 0);
    frame(scene, camera);
    expect(renderer.info.updates == 0 && red_at(0.75, 0.5) > 0.5 && red_at(0.25, 0.5) < 0.1,
           "unmarked: nothing copied, the old shape drawn");
    expect(g.version == v0 + 1, "unmarked: version stays");

    // marked, copied by the frame's first render; marked again in the same frame after that:
    // the second render of the frame keeps what the first copied, the next frame copies it
    g.needs_update = true;
    frame(scene, camera, false);
    expect(renderer.info.updates == 1, "a frame's first render copies the marked vertices");
    place(g, 1);
    g.needs_update = true;
    frame(scene, camera);
    expect(renderer.info.updates == 0 && red_at(0.25, 0.5) > 0.5 && red_at(0.75, 0.5) < 0.1,
           "marked again after the frame copied it: that frame draws the first change");
    frame(scene, camera);
    expect(renderer.info.updates == 1 && red_at(0.75, 0.5) > 0.5 && red_at(0.25, 0.5) < 0.1,
           "...and the next frame copies and draws the second");
    expect(g.version == v0 + 3, string::tformat("version counts every copy (%d)", g.version - v0));

    // tangents: copied with the vertices (the tangent buffer is a copy target too)
    BufferGeometry* t = drie::buffer_geometry();
    t.add_vertex({ -0.2, -0.2, 1 }, { 0, 0, 1 }, { 0, 0 });
    t.add_vertex({ 0.2, -0.2, 1 }, { 0, 0, 1 }, { 1, 0 });
    t.add_vertex({ 0, 0.2, 1 }, { 0, 0, 1 }, { 0.5, 1 });
    t.add_triangle(0, 1, 2);
    for (int i = 0; i < 3; i++) {
        t.tangents.push(1);
        t.tangents.push(0);
        t.tangents.push(0);
        t.tangents.push(1);
    }
    t.dynamic = true;
    t.bounds_radius = 10;
    MeshStandardMaterial* aniso = drie::mesh_standard_material(color: 0x00ff00);
    aniso.anisotropy = 0.8f;
    Mesh* tri = drie::mesh(t, aniso);
    scene.add(tri);
    frame(scene, camera);
    for (int i = 0; i < 3; i++) t.tangents[(usz)i * 4] = -1;
    t.needs_update = true;
    frame(scene, camera);
    usz copied = renderer.info.update_bytes;
    float[12] on_gpu;
    readback::track(t.tangent_buf, 0);
    readback::fetch(0, &on_gpu, 12 * float::size, true);
    expect(copied == 3 * drie::FLOATS_PER_VERTEX * float::size + 3 * 4 * float::size && on_gpu[0] == -1 && on_gpu[8] == -1,
           string::tformat("with tangents: the vertices and the tangents copied (%d bytes; x of the first on the GPU %.0f)", copied, on_gpu[0]));
    scene.remove(tri);
    scene.remove(quad);

    // a cloth: still, then pushed
    Cloth* c = cloth::grid(12, 12, { -0.6, 0.6, 0 }, { 1.2, 0, 0 }, { 0, -1.2, 0 });
    c.pin_row(0);
    c.drag = 0;
    c.damping = 4;
    scene.add(c.make_mesh(drie::mesh_basic_material(color: 0xff0000, side: DOUBLE_SIDE)));
    for (int k = 0; k < 600; k++) c.update(1.0f / 60);
    int body = c.add_capsule({ 0, -0.3, 2 }, { 0, 0.3, 2 }, 0.15f);
    frame(scene, camera);
    int copies;
    for (int k = 0; k < 10; k++) {
        c.update(1.0f / 60);
        frame(scene, camera);
        copies += renderer.info.updates;
    }
    expect(copies == 0, string::tformat("a cloth at rest copies nothing (%d copies in 10 frames)", copies));
    float most;
    copies = 0;
    for (int k = 0; k < 20; k++) {
        float x = -0.5f + 0.05f * k;
        c.move_collider(body, { x, -0.3, 0 }, { x, 0.3, 0 });
        c.update(1.0f / 60);
        frame(scene, camera);
        copies += renderer.info.updates;
        most = math::max(most, largest_motion());
    }
    expect(copies >= 15 && most > 1, string::tformat("a pushed cloth: copied in %d of 20 frames, motion vectors up to %.2f px", copies, most));

    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
