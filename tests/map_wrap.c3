// The maps of one material that wrap differently, and Mesh.draw_start/draw_count.
//   - left: a plane with a clamped color map (white, 1x1) and an emissive checker repeated 4
//     times; the maps share one sampler, the color map's, so the checker's repeating is done
//     on its uv in the shader. Its row across the middle shows 4 black and white pairs, not
//     one pair and then a smear of the last column
//   - right: two meshes on one plane geometry, each drawing one of its two triangles (red
//     and green), as three draws a geometry's groups with an array of materials
// The picture: the two, left to right.
//
//   just test map_wrap [out.png]   (tests/build/map_wrap.png)
module map_wrap_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int S = 128;

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
    String out_path = "tests/build/map_wrap.png";
    if (args.len > 1) out_path = args[1];
    Renderer* renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    renderer.tone_mapping = NO_TONE_MAPPING;
    OrthographicCamera* camera = drie::orthographic_camera(-1, 1, 1, -1, 0.1, 10);
    camera.position = { 0, 0, 2 };
    RenderTarget* rt = drie::render_target(S, S, 1, RGBA8);
    renderer.set_render_target(rt);
    char[][2] px;

    // a checker of two 8 x 8 squares on a side, repeated 4 times; a white color map that clamps
    char[] checker = mem::new_array(char, 16 * 16 * 4);
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            char v = ((x / 8) ^ (y / 8)) & 1 ? 255 : 0;
            checker[(y * 16 + x) * 4:4] = (char[4]){ v, v, v, 255 }[..];
        }
    }
    Texture* glow = drie::data_texture(checker, 16, 16);
    glow.nearest = true;
    glow.repeat = { 4, 4 };
    Texture* white = drie::data_texture({ 255, 255, 255, 255 }, 1, 1);
    white.wrap_s = CLAMP_TO_EDGE_WRAPPING;
    white.wrap_t = CLAMP_TO_EDGE_WRAPPING;
    MeshLambertMaterial* lit = drie::mesh_lambert_material(color: 0x000000, emissive: 0xffffff);
    lit.map = white;
    lit.emissive_map = glow;
    Scene* scene = drie::scene();
    scene.background = 0x000000;
    scene.add(drie::mesh(drie::plane_geometry(2, 2), lit));
    px[0] = mem::new_array(char, S * S * 4);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, S, S, px[0].ptr);
    int changes;
    bool was_white = at(px[0], 0, S / 2 + 4)[0] > 128;
    for (int x = 1; x < S; x++) {
        bool is_white = at(px[0], x, S / 2 + 4)[0] > 128;
        if (is_white != was_white) changes++;
        was_white = is_white;
    }
    expect(changes == 7, string::tformat("the checker repeated 4 times: %d changes across the middle (7)", changes));

    // one geometry, two meshes, a triangle each
    PlaneGeometry* plane = drie::plane_geometry(2, 2);
    Mesh* red = drie::mesh(plane, drie::mesh_basic_material(color: 0xff0000));
    red.draw_start = 0;
    red.draw_count = 3;
    Mesh* green = drie::mesh(plane, drie::mesh_basic_material(color: 0x00ff00));
    green.draw_start = 3;
    green.draw_count = 3;
    Scene* halves = drie::scene();
    halves.background = 0x000000;
    halves.add(red);
    halves.add(green);
    px[1] = mem::new_array(char, S * S * 4);
    renderer.render(halves, camera);
    drie::commit();
    readback::read_image(rt.texture_image, S, S, px[1].ptr);
    char[4] tl = at(px[1], 8, 8);
    char[4] br = at(px[1], S - 8, S - 8);
    bool split = (tl[0] > 200 && tl[1] < 20 && br[1] > 200 && br[0] < 20)
        || (tl[1] > 200 && tl[0] < 20 && br[0] > 200 && br[1] < 20);
    expect(split, string::tformat("a triangle red and one green: top left %s, bottom right %s", tl[:3], br[:3]));

    char[] sheet = mem::new_array(char, S * 2 * S * 4);
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            usz src = ((usz)y * S + (usz)x) * 4;
            for (int k = 0; k < 2; k++) {
                usz dst = ((usz)y * S * 2 + (usz)(k * S + x)) * 4;
                sheet[dst:3] = px[k][src:3];
                sheet[dst + 3] = 255;
            }
        }
    }
    (void)png::write(out_path, sheet, S * 2, S);
    io::printfn("%s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
