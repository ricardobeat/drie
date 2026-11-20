// Renderer.render_environment: a node material drawn into an environment map. Its graph is the
// sky max(d.y, 0) in red and max(d.x, 0) in green, d the direction (normal_geometry).
//   - its light on surfaces (EnvLight, from the drawing read back), facing up, down, +x and -x:
//     as env_light makes it of the same sky worked out on the CPU texel by texel (within 3%,
//     and 0.02 where it is near nothing)
//   - a mirror ball lit by the map, seen from +z: red at its top, not at its bottom; green at
//     its right (+x), not at its left
//   - drawn again into the same map with a uniform changed (the red halved): the same texture,
//     its light halved
//   - the ball with an env_node in place of the map's reflection, red as the reflected
//     direction's x: red at its right, none at its left, no green anywhere
// The picture: the ball, then the ball after the second drawing.
//
//   just test render_environment [out.png]   (tests/build/render_environment.png)
module render_environment_test;
import drie, sokol, drie::readback, drie::png, drie::nodes;
import std::io, std::math;

const int S = 128;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn bool near(float got, float want) => math::abs(got - want) <= math::max(want * 0.03f, 0.02f);

fn bool near3(Vector3 got, Vector3 want) => near(got.x, want.x) && near(got.y, want.y) && near(got.z, want.z);

// The sky on the CPU, as drie lays out an environment map (64 x 32, the shader's env_uv), with
// the red times `red`.
fn EnvLight cpu_sky(float red) {
    float[] px = mem::new_array(float, 64 * 32 * 4);
    defer free(px);
    for (int y = 0; y < 32; y++) {
        for (int x = 0; x < 64; x++) {
            float phi = ((float)y + 0.5f) / 32 * (float)math::PI;
            float theta = (((float)x + 0.5f) / 64 - 0.5f) * 2 * (float)math::PI;
            Vector3 d = { math::cos(theta) * math::sin(phi), math::cos(phi), math::sin(theta) * math::sin(phi) };
            usz i = ((usz)y * 64 + (usz)x) * 4;
            px[i] = math::max(d.y, 0.0f) * red;
            px[i + 1] = math::max(d.x, 0.0f);
            px[i + 3] = 1;
        }
    }
    return drie::env_light(px, { }, 64, 32, false);
}

fn int main(String[] args) {
    String out_path = "tests/build/render_environment.png";
    if (args.len > 1) out_path = args[1];
    Renderer* renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();

    Node d = nodes::normal_geometry();
    Node red = nodes::uniform(1);
    Node zero = nodes::num(0);
    Node sky = drie::node_join({ drie::node_call(MAX, d.y(), zero) * red, drie::node_call(MAX, d.x(), zero), zero }, VEC3);
    MeshBasicMaterial* graph = drie::mesh_basic_material(fog: false);
    graph.color_node = sky;

    Texture* env = renderer.render_environment(graph, width: 256);
    EnvLight want = cpu_sky(1);
    Vector3[4] normals = { { 0, 1, 0 }, { 0, -1, 0 }, { 1, 0, 0 }, { -1, 0, 0 } };
    String[4] names = { "up", "down", "+x", "-x" };
    foreach (k, n : normals) {
        Vector3 got = env.env_light.irradiance(n);
        Vector3 w = want.irradiance(n);
        expect(near3(got, w), string::tformat("facing %s: %.3f %.3f, on the CPU %.3f %.3f", names[k], got.x, got.y, w.x, w.y));
    }

    Scene* scene = drie::scene();
    scene.background = 0x000000;
    scene.environment = env;
    Mesh* ball = drie::mesh(drie::sphere_geometry(1, 96, 48), drie::mesh_standard_material(color: 0xffffff, roughness: 0, metalness: 1));
    scene.add(ball);
    PerspectiveCamera* camera = drie::perspective_camera(30, 1, 0.1, 20);
    camera.position = { 0, 0, 4.2f };
    camera.look_at_vec({ 0, 0, 0 });
    RenderTarget* rt = drie::render_target(S, S, 1, RGBA8);
    char[] px_a = mem::new_array(char, S * S * 4);
    char[] px_b = mem::new_array(char, S * S * 4);
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, S, S, px_a.ptr);
    // (RGBA; the ball's top, bottom, right and left, a little inside its edge)
    int r = S * 33 / 100;
    int[2][4] at = { { S / 2, S / 2 - r }, { S / 2, S / 2 + r }, { S / 2 + r, S / 2 }, { S / 2 - r, S / 2 } };
    int[4] reds;
    int[4] greens;
    for (int i = 0; i < 4; i++) {
        usz o = ((usz)at[i][1] * S + (usz)at[i][0]) * 4;
        reds[i] = px_a[o];
        greens[i] = px_a[o + 1];
    }
    expect(reds[0] > 100 && reds[1] < 20, string::tformat("the ball's red: top %d, bottom %d", reds[0], reds[1]));
    expect(greens[2] > 100 && greens[3] < 20, string::tformat("the ball's green: right %d, left %d", greens[2], greens[3]));

    red.set(0.5f);
    Texture* again = renderer.render_environment(graph, env, width: 256);
    Vector3 up2 = again.env_light.irradiance({ 0, 1, 0 });
    Vector3 want2 = cpu_sky(0.5f).irradiance({ 0, 1, 0 });
    expect(again == env && near3(up2, want2), string::tformat("drawn again with the red halved: %.3f facing up, on the CPU %.3f", up2.x, want2.x));
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, S, S, px_b.ptr);

    MeshStandardMaterial* mirror = drie::mesh_standard_material(color: 0xffffff, roughness: 0, metalness: 1);
    Node dir = nodes::env_direction();
    mirror.env_node = drie::node_join({ drie::node_call(MAX, dir.x(), zero), zero, zero }, VEC3);
    ball.material = mirror;
    char[] px_c = mem::new_array(char, S * S * 4);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, S, S, px_c.ptr);
    int[4] env_reds;
    int max_green;
    for (int i = 0; i < 4; i++) env_reds[i] = px_c[((usz)at[i][1] * S + (usz)at[i][0]) * 4];
    for (usz o = 0; o < (usz)S * S * 4; o += 4) max_green = math::max(max_green, (int)px_c[o + 1]);
    expect(env_reds[2] > 100 && env_reds[3] < 20 && max_green < 20,
           string::tformat("env_node: red at the right %d, the left %d; the most green %d", env_reds[2], env_reds[3], max_green));

    char[] sheet = mem::new_array(char, S * 2 * S * 4);
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            usz src = ((usz)y * S + (usz)x) * 4;
            for (int k = 0; k < 2; k++) {
                char[] from = k == 0 ? px_a : px_b;
                usz dst = ((usz)y * S * 2 + (usz)(k * S + x)) * 4;
                sheet[dst] = from[src];
                sheet[dst + 1] = from[src + 1];
                sheet[dst + 2] = from[src + 2];
                sheet[dst + 3] = 255;
            }
        }
    }
    (void)png::write(out_path, sheet, S * 2, S);
    io::printfn("%s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
