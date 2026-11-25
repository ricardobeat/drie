// While a node material's program compiles, its meshes are drawn with the plain material.
//
// - The scene has a floor, two static spot lights with shadows, and three boxes.
// - Box A is red. Its color_node makes it blue.
// - A gray box stands in front of box A and hides part of it.
// - Box B has a position_node that lifts it 0.4 m.
// - DRIE_WARM_DELAY holds the compiles back. The test sets it to 2 s if it is not set.
// - Frame 1 must count waiting draws.
// - Frame 1 must show box A in red, behind the gray box, with its shadow.
// - The last frame must show box A in blue, with the same pixels and the same shadow.
// - The last frame must equal a frame drawn without background compiles.
// - Box B checks the shadow tiles. A static light draws its tile once.
// - In frame 1 the tile shows box B unlifted, because the plain material has no position_node.
// - So the tile must be drawn again once the program is ready.
// - The last frame must equal a frame where every tile was drawn again.
//
// Run it with `just test program_fallback`.
// It writes tests/build/program_fallback.png, with frame 1 on the left and the last frame on the right.
module program_fallback_test;
import drie, drie::nodes, drie::readback, drie::png, sokol;
import std::io, std::math, std::os::env, std::thread;

const int W = 256;
const int H = 192;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn bool red(char[] p, usz i) => p[i] > 40 && p[i] > 2 * (int)p[i + 1] && p[i] > 2 * (int)p[i + 2];
fn bool blue(char[] p, usz i) => p[i + 2] > 40 && p[i + 2] > 2 * (int)p[i] && p[i + 2] > 2 * (int)p[i + 1];
fn int lum(char[] p, usz i) => ((int)p[i] + p[i + 1] + p[i + 2]) / 3;
// gray: the floor (and the box in front), lit or in shadow
fn bool gray(char[] p, usz i) => math::abs((int)p[i] - p[i + 2]) < 12 && math::abs((int)p[i] - p[i + 1]) < 12;

fn void grab(RenderTarget* rt, char[] px) => readback::read_image(rt.texture_image, W, H, px.ptr);

fn int main(String[] args) {
    if (!@ok(env::tget_var("DRIE_WARM_DELAY"))) (void)env::set_var("DRIE_WARM_DELAY", "2");
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.compile_in_background = true;
    Scene* scene = drie::scene();
    scene.add(drie::hemisphere_light(0xffffff, 0x404040, 0.4));
    SpotLight*[2] spots;
    float[2] xs = { -1.5f, 1.5f };
    for (int i = 0; i < 2; i++) {
        SpotLight* l = drie::spot_light(0xffffff, 40, 0, 0.6f, 0.2f, 2);
        l.position = { xs[i], 5, 2 };
        l.target.position = { xs[i], 0, -0.5f };
        l.cast_shadow = true;
        l.shadow.auto_update = false;
        l.shadow.map_size = 512;
        spots[i] = l;
        scene.add(l);
        scene.add(l.target);
    }
    Mesh* floor = drie::mesh(drie::plane_geometry(20, 20), drie::mesh_standard_material(color: 0x808080, roughness: 1));
    floor.rotation.x = -(float)math::PI / 2;
    floor.receive_shadow = true;
    scene.add(floor);
    Material* ma = drie::mesh_standard_material(color: 0xff2020);
    ma.color_node = nodes::color(0x2030ff);
    Mesh* a = drie::mesh(drie::box_geometry(1, 1, 1), ma);
    a.position = { -1.5f, 0.5f, 0 };
    a.cast_shadow = true;
    scene.add(a);
    Mesh* front = drie::mesh(drie::box_geometry(0.6f, 0.6f, 0.2f), drie::mesh_standard_material(color: 0x606060));
    front.position = { -1.5f, 0.6f, 1.2f };
    scene.add(front);
    Material* mb = drie::mesh_standard_material(color: 0xd0d0d0);
    mb.position_node = nodes::position_local() + nodes::vec3(0, 0.4f, 0);
    Mesh* b = drie::mesh(drie::box_geometry(1, 1, 1), mb);
    b.position = { 1.5f, 0.5f, 0 };
    b.cast_shadow = true;
    scene.add(b);

    PerspectiveCamera* camera = drie::perspective_camera(50, (float)W / H, 0.1f, 50);
    camera.position = { 0, 4, 6 };
    camera.look_at_vec({ 0, 0, 0 });
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    sz n = (sz)W * H * 4;
    char[] first = mem::new_array(char, n);
    char[] ready = mem::new_array(char, n);
    char[] fresh = mem::new_array(char, n);
    char[] sync = mem::new_array(char, n);
    char[] empty = mem::new_array(char, n);

    renderer.render(scene, camera);
    drie::commit();
    grab(rt, first);
    int waiting_first = renderer.info.waiting;
    io::printfn("frame 1: %d draws waiting, %d calls, %d shadow calls", waiting_first, renderer.info.calls, renderer.info.shadow_calls);
    expect(waiting_first > 0, "frame 1: the node materials' draws wait for their programs");
    int frames = 1;
    for (; frames < 2000 && renderer.info.waiting > 0; frames++) {
        thread::sleep_ms(10);
        renderer.render(scene, camera);
        drie::commit();
    }
    io::printfn("ready after %d frames", frames);
    expect(renderer.info.waiting == 0, "the programs get ready");
    // (a frame more: a static tile redrawn in the last frame that waited shows now)
    renderer.render(scene, camera);
    drie::commit();
    grab(rt, ready);
    foreach (l : spots) l.shadow.needs_update = true;
    renderer.render(scene, camera);
    drie::commit();
    grab(rt, fresh);
    renderer.compile_in_background = false;
    foreach (l : spots) l.shadow.needs_update = true;
    renderer.render(scene, camera);
    drie::commit();
    grab(rt, sync);
    a.visible = false;
    b.visible = false;
    front.visible = false;
    foreach (l : spots) l.shadow.needs_update = true;
    renderer.render(scene, camera);
    drie::commit();
    grab(rt, empty);

    // box A: red in frame 1, blue when ready, the same pixels (the gray box in front cut out
    // of both); its shadow on the floor in the left half: the same pixels too
    int a1, a2, both, s1, s2, sboth, front_px;
    for (usz y = 0; y < H; y++) {
        for (usz x = 0; x < W; x++) {
            usz i = (y * W + x) * 4;
            bool r1 = red(first, i);
            bool b2 = blue(ready, i);
            if (r1) a1++;
            if (b2) a2++;
            if (r1 && b2) both++;
            if (x >= W / 2) continue;
            bool sh1 = gray(first, i) && gray(empty, i) && lum(first, i) + 25 < lum(empty, i);
            bool sh2 = gray(ready, i) && gray(empty, i) && lum(ready, i) + 25 < lum(empty, i);
            if (sh1) s1++;
            if (sh2) s2++;
            if (sh1 && sh2) sboth++;
        }
    }
    io::printfn("box A: %d px in frame 1, %d ready, %d in both; its shadow: %d, %d, %d in both", a1, a2, both, s1, s2, sboth);
    expect(a1 > 500 && both >= a1 * 98 / 100 && both >= a2 * 98 / 100, "frame 1: box A drawn where it stands, behind the box in front");
    expect(s1 > 200 && sboth >= s1 * 95 / 100 && sboth >= s2 * 95 / 100, "frame 1: box A's shadow");
    int diff_fresh, diff_sync;
    for (sz i = 0; i < n; i++) {
        if (math::abs((int)ready[i] - fresh[i]) > 2) diff_fresh++;
        if (math::abs((int)ready[i] - sync[i]) > 2) diff_sync++;
    }
    io::printfn("ready vs every tile drawn again: %d bytes differ; vs no background compiles: %d", diff_fresh, diff_sync);
    expect(diff_fresh == 0, "a static light's tile drawn with the plain material is drawn again with the program");
    expect(diff_sync == 0, "the ready picture is the one made without background compiles");

    char[] sheet = mem::new_array(char, (sz)W * 2 * H * 3);
    for (usz y = 0; y < H; y++) {
        for (usz x = 0; x < W; x++) {
            for (usz c = 0; c < 3; c++) {
                sheet[(y * W * 2 + x) * 3 + c] = first[(y * W + x) * 4 + c];
                sheet[(y * W * 2 + W + x) * 3 + c] = ready[(y * W + x) * 4 + c];
            }
        }
    }
    (void)png::write("tests/build/program_fallback.png", sheet, W * 2, H, channels: 3);
    if (failures) {
        io::printfn("%d failed", failures);
        return 1;
    }
    io::printn("all passed");
    return 0;
}
