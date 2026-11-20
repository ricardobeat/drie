// Light.channels and Material.light_channels (three's lightsNode), and a phong material's
// specular_node. A white sphere between a red point light on the left (channel 1) and a blue
// one on the right (channel 2):
//   - light_channels 0 takes both: red on its left, blue on its right
//   - light_channels 1 takes the red one only, 2 the blue one only
//   - a black phong sphere with specular_node green under white lights: its highlight green,
//     no red or blue anywhere
// The picture: the four, left to right.
//
//   just test light_channels [out.png]   (tests/build/light_channels.png)
module light_channels_test;
import drie, sokol, drie::readback, drie::png, drie::nodes;
import std::io, std::math;

const int S = 96;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn char[4] at(char[] px, int x, int y) {
    usz o = ((usz)y * S + (usz)x) * 4;
    return { px[o], px[o + 1], px[o + 2], px[o + 3] };
}

fn int main(String[] args) {
    String out_path = "tests/build/light_channels.png";
    if (args.len > 1) out_path = args[1];
    Renderer* renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    renderer.tone_mapping = NO_TONE_MAPPING;

    Scene* scene = drie::scene();
    scene.background = 0x000000;
    MeshStandardMaterial* white = drie::mesh_standard_material(color: 0xffffff, roughness: 1, metalness: 0);
    Mesh* ball = drie::mesh(drie::sphere_geometry(1, 64, 32), white);
    scene.add(ball);
    PointLight* red = drie::point_light(0xff0000, 40, 0, 2);
    red.position = { -3, 0, 2 };
    red.channels = 1;
    scene.add(red);
    PointLight* blue = drie::point_light(0x0000ff, 40, 0, 2);
    blue.position = { 3, 0, 2 };
    blue.channels = 2;
    scene.add(blue);
    PerspectiveCamera* camera = drie::perspective_camera(40, 1, 0.1, 20);
    camera.position = { 0, 0, 4 };
    camera.look_at_vec({ 0, 0, 0 });
    RenderTarget* rt = drie::render_target(S, S, 1, RGBA8);
    renderer.set_render_target(rt);

    char[][4] px;
    uint[3] takes = { 0, 1, 2 };
    for (int i = 0; i < 3; i++) {
        px[i] = mem::new_array(char, S * S * 4);
        white.light_channels = takes[i];
        renderer.render(scene, camera);
        drie::commit();
        readback::read_image(rt.texture_image, S, S, px[i].ptr);
        char[4] left = at(px[i], S * 32 / 100, S / 2);
        char[4] right = at(px[i], S * 68 / 100, S / 2);
        bool lit_l = left[0] > 60;
        bool lit_r = right[2] > 60;
        bool want_l = takes[i] != 2;
        bool want_r = takes[i] != 1;
        expect(lit_l == want_l && lit_r == want_r && left[2] < 20 && right[0] < 20,
               string::tformat("light_channels %d: left %s, right %s", takes[i], left[:3], right[:3]));
    }

    MeshPhongMaterial* phong = drie::mesh_phong_material(color: 0x000000, specular: 0xffffff, shininess: 30);
    phong.specular_node = nodes::vec3(0, 1, 0);
    ball.material = phong;
    red.color = 0xffffff;
    blue.color = 0xffffff;
    px[3] = mem::new_array(char, S * S * 4);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, S, S, px[3].ptr);
    int max_g;
    int max_rb;
    for (usz o = 0; o < (usz)S * S * 4; o += 4) {
        max_g = math::max(max_g, (int)px[3][o + 1]);
        max_rb = math::max(max_rb, math::max((int)px[3][o], (int)px[3][o + 2]));
    }
    expect(max_g > 100 && max_rb < 20, string::tformat("specular_node green: the most green %d, red or blue %d", max_g, max_rb));

    char[] sheet = mem::new_array(char, S * 4 * S * 4);
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            usz src = ((usz)y * S + (usz)x) * 4;
            for (int k = 0; k < 4; k++) {
                usz dst = ((usz)y * S * 4 + (usz)(k * S + x)) * 4;
                sheet[dst:3] = px[k][src:3];
                sheet[dst + 3] = 255;
            }
        }
    }
    (void)png::write(out_path, sheet, S * 4, S);
    io::printfn("%s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
