// Renderer.render_probe: a scene drawn in every direction into an environment map on the GPU.
// The scene: a 10 m room whose six walls are unlit colors, +x red, -x cyan, +y green, -y
// magenta, +z blue, -z yellow, probed from its middle.
//   - the map's light on surfaces facing each axis: as capture_environment's (the six views
//     read back and resampled on the CPU) within 3% (0.02 where it is near nothing)
//   - a mirror ball lit by the map, seen from +z: blue in its middle (it mirrors +z back),
//     green at its top, magenta at its bottom, red at its right, cyan at its left
//   - drawn again into the same map with the +x wall turned white: the same texture, the
//     ball's right side white
// The picture: the ball, then the ball after the second drawing.
//
//   just test render_probe [out.png]   (tests/build/render_probe.png)
module render_probe_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int S = 128;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn bool near(float got, float want) => math::abs(got - want) <= math::max(want * 0.03f, 0.02f);

Vector3[6] axes = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
Color[6] colors = { 0xff0000, 0x00ffff, 0x00ff00, 0xff00ff, 0x0000ff, 0xffff00 };

// The ball's pixel at (x, y), RGBA.
fn char[4] at(char[] px, int x, int y) {
    usz o = ((usz)y * S + (usz)x) * 4;
    return { px[o], px[o + 1], px[o + 2], px[o + 3] };
}

// Whether a pixel is about `c` (each channel within 60 of 0 or 255 as c has it).
fn bool is(char[4] p, Color c) {
    int[3] want = { (int)((c >> 16) & 0xff), (int)((c >> 8) & 0xff), (int)(c & 0xff) };
    for (int i = 0; i < 3; i++) {
        if (math::abs((int)p[i] - want[i]) > 60) return false;
    }
    return true;
}

fn void check_ball(char[] px, Color right, String when) {
    int r = S * 30 / 100;
    char[4] mid = at(px, S / 2, S / 2);
    char[4] top = at(px, S / 2, S / 2 - r);
    char[4] bottom = at(px, S / 2, S / 2 + r);
    char[4] right_px = at(px, S / 2 + r, S / 2);
    char[4] left = at(px, S / 2 - r, S / 2);
    expect(is(mid, 0x0000ff) && is(top, 0x00ff00) && is(bottom, 0xff00ff) && is(right_px, right) && is(left, 0x00ffff),
           string::tformat("%s: the ball's middle %s, top %s, bottom %s, right %s, left %s", when,
                           mid[:3], top[:3], bottom[:3], right_px[:3], left[:3]));
}

fn int main(String[] args) {
    String out_path = "tests/build/render_probe.png";
    if (args.len > 1) out_path = args[1];
    Renderer* renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();

    Scene* room = drie::scene();
    room.background = 0x000000;
    MeshBasicMaterial*[6] walls;
    foreach (i, a : axes) {
        walls[i] = drie::mesh_basic_material(color: colors[i], fog: false);
        Mesh* wall = drie::mesh(drie::plane_geometry(10, 10), walls[i]);
        wall.position = a * 5;
        if (math::abs(a.y) > 0.5f) wall.up = { 0, 0, 1 };
        wall.look_at_vec({ 0, 0, 0 });
        room.add(wall);
    }

    Texture* probe = renderer.render_probe(room, { 0, 0, 0 }, face_size: 128);
    Texture* captured = renderer.capture_environment(room, { 0, 0, 0 }, face_size: 128, width: 512, hdr: true);
    captured.upload();
    String[6] names = { "+x", "-x", "+y", "-y", "+z", "-z" };
    foreach (k, n : axes) {
        Vector3 got = probe.env_light.irradiance(n);
        Vector3 want = captured.env_light.irradiance(n);
        expect(near(got.x, want.x) && near(got.y, want.y) && near(got.z, want.z),
               string::tformat("facing %s: %.3f %.3f %.3f, captured %.3f %.3f %.3f", names[k], got.x, got.y, got.z,
                               want.x, want.y, want.z));
    }

    Scene* scene = drie::scene();
    scene.background = 0x000000;
    scene.environment = probe;
    Mesh* ball = drie::mesh(drie::sphere_geometry(1, 96, 48), drie::mesh_standard_material(color: 0xffffff, roughness: 0, metalness: 1));
    scene.add(ball);
    PerspectiveCamera* camera = drie::perspective_camera(30, 1, 0.1, 20);
    camera.position = { 0, 0, 4.2f };
    camera.look_at_vec({ 0, 0, 0 });
    RenderTarget* rt = drie::render_target(S, S, 1, RGBA8);
    char[] px_a = mem::new_array(char, S * S * 4);
    char[] px_b = mem::new_array(char, S * S * 4);
    renderer.tone_mapping = NO_TONE_MAPPING;
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, S, S, px_a.ptr);
    check_ball(px_a, 0xff0000, "first drawing");

    walls[0].color = 0xffffff;
    Texture* again = renderer.render_probe(room, { 0, 0, 0 }, probe, face_size: 128);
    expect(again == probe, "drawn again into the same map");
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, S, S, px_b.ptr);
    check_ball(px_b, 0xffffff, "the +x wall turned white");

    char[] sheet = mem::new_array(char, S * 2 * S * 4);
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            usz src = ((usz)y * S + (usz)x) * 4;
            for (int k = 0; k < 2; k++) {
                char[] from = k == 0 ? px_a : px_b;
                usz dst = ((usz)y * S * 2 + (usz)(k * S + x)) * 4;
                sheet[dst:3] = from[src:3];
                sheet[dst + 3] = 255;
            }
        }
    }
    (void)png::write(out_path, sheet, S * 2, S);
    io::printfn("%s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
