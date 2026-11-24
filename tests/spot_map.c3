// SpotLight.map (a cookie): a spot light pointing straight down on a white floor projects a
// 2 by 2 picture, red green over blue white. Pointing down, the picture's up is world +x and
// its right world +z (see SpotLight.map), so from above:
//   - x > 0, z < 0 red; x > 0, z > 0 green; x < 0, z < 0 blue; x < 0, z > 0 white
//   - with the light's shadow on (soft, too), the same
//   - map_rotation PI / 2 turns it a quarter: red where blue was
//   - map_rect on the picture's top right quarter alone: green all over
//   - a second light with another texture draws without its map (white)
// The picture: the five, left to right.
//
//   just test spot_map [out.png]   (tests/build/spot_map.png)
module spot_map_test;
import drie, sokol, drie::readback, drie::png;
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

// the color at world (x, 0, z), from the camera straight above (+x right, -z up the
// picture; it sees 4 m across)
fn char[4] floor_at(char[] px, float x, float z) {
    return at(px, (int)((x / 4 + 0.5f) * S), (int)((z / 4 + 0.5f) * S));
}

fn bool is(char[4] c, int r, int g, int b) {
    return math::abs((int)c[0] - r) < 60 && math::abs((int)c[1] - g) < 60 && math::abs((int)c[2] - b) < 60;
}

fn void check_quarters(char[] px, String what, bool turned) {
    char[4] a = floor_at(px, 0.7, -0.7);
    char[4] b = floor_at(px, 0.7, 0.7);
    char[4] c = floor_at(px, -0.7, -0.7);
    char[4] d = floor_at(px, -0.7, 0.7);
    // turned a quarter counterclockwise as seen from the light (from below): the picture's
    // corners move on one
    if (turned) {
        expect(is(c, 220, 0, 0) && is(a, 0, 220, 0) && is(d, 0, 0, 220) && is(b, 220, 220, 220)
               || is(b, 220, 0, 0) && is(d, 0, 220, 0) && is(a, 0, 0, 220) && is(c, 220, 220, 220),
               string::tformat("%s: %s %s / %s %s", what, a[:3], b[:3], c[:3], d[:3]));
        return;
    }
    expect(is(a, 220, 0, 0) && is(b, 0, 220, 0) && is(c, 0, 0, 220) && is(d, 220, 220, 220),
           string::tformat("%s: %s %s / %s %s", what, a[:3], b[:3], c[:3], d[:3]));
}

fn int main(String[] args) {
    String out_path = "tests/build/spot_map.png";
    if (args.len > 1) out_path = args[1];
    Renderer* renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    renderer.tone_mapping = NO_TONE_MAPPING;

    Scene* scene = drie::scene();
    scene.background = 0x000000;
    MeshStandardMaterial* white = drie::mesh_standard_material(color: 0xffffff, roughness: 1, metalness: 0);
    Mesh* ground = drie::mesh(drie::plane_geometry(8, 8), white);
    ground.rotation = { -(float)math::PI / 2, 0, 0 };
    ground.receive_shadow = true;
    scene.add(ground);

    // 16 by 16: red green over blue white, a quarter each
    char[16 * 16 * 4] pic;
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            char[4] c = y < 8 ? (x < 8 ? (char[4]) { 255, 0, 0, 255 } : (char[4]) { 0, 255, 0, 255 })
                : (x < 8 ? (char[4]) { 0, 0, 255, 255 } : (char[4]) { 255, 255, 255, 255 });
            pic[(y * 16 + x) * 4 : 4] = c[..];
        }
    }
    Texture* map = drie::data_texture(&pic, 16, 16, NO_COLOR_SPACE);
    map.generate_mipmaps = false;
    SpotLight* spot = drie::spot_light(0xffffff, 12, 0, 0.6f, 0, 2);
    spot.position = { 0, 2, 0 };
    spot.target.position = { 0, 0, 0 };
    spot.map = map;
    scene.add(spot);
    scene.add(spot.target);

    OrthographicCamera* camera = drie::orthographic_camera(-2, 2, 2, -2, 0.1f, 20);
    camera.position = { 0, 5, 0 };
    camera.rotation.x = -(float)math::PI / 2;
    RenderTarget* rt = drie::render_target(S, S, 1, RGBA8);
    renderer.set_render_target(rt);

    char[][5] px;
    for (int i = 0; i < 5; i++) {
        spot.cast_shadow = i == 1;
        spot.shadow.softness = i == 1 ? 0.05f : 0;
        spot.map_rotation = i == 2 ? (float)math::PI / 2 : 0;
        spot.map_rect = i == 3 ? (float[4]) { 0.5f, 0, 0.5f, 0.5f } : (float[4]) { };
        if (i == 4) {
            // another light, far off, with a texture of its own: the first one's map is kept
            SpotLight* other = drie::spot_light(0xffffff, 0.001f, 0, 0.3f, 0, 2);
            other.position = { 30, 2, 0 };
            other.target.position = { 30, 0, 0 };
            other.map = drie::data_texture(&pic, 16, 16, NO_COLOR_SPACE);
            scene.add(other);
            scene.add(other.target);
            spot.position = { 30, 2, 0.001f };
            spot.target.position = { 30, 0, 0 };
            other.position = { 0, 2, 0 };
            other.target.position = { 0, 0, 0 };
            other.intensity = 12;
            spot.intensity = 0.001f;
            spot.map = drie::data_texture(&pic, 16, 16, NO_COLOR_SPACE);
        }
        px[i] = mem::new_array(char, S * S * 4);
        renderer.render(scene, camera);
        drie::commit();
        readback::read_image(rt.texture_image, S, S, px[i].ptr);
    }
    check_quarters(px[0], "the map", false);
    check_quarters(px[1], "the map with a soft shadow", false);
    check_quarters(px[2], "turned a quarter", true);
    bool all_green = true;
    float[2][4] spots = { { 0.7f, -0.7f }, { 0.7f, 0.7f }, { -0.7f, -0.7f }, { -0.7f, 0.7f } };
    foreach (p : spots) all_green = all_green && is(floor_at(px[3], p[0], p[1]), 0, 220, 0);
    expect(all_green, "map_rect: the top right quarter, green all over");
    char[4] w = floor_at(px[4], 0.3f, -0.3f);
    // (whichever of the two maps came first, the light at the origin has the same picture,
    // or none: a light whose texture differs from the frame's map shines white)
    expect(is(w, 220, 0, 0) || is(w, 220, 220, 220), string::tformat("two textures: %s", w[:3]));

    char[] sheet = mem::new_array(char, S * 5 * S * 4);
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            usz src = ((usz)y * S + (usz)x) * 4;
            for (int k = 0; k < 5; k++) {
                usz dst = ((usz)y * S * 5 + (usz)(k * S + x)) * 4;
                sheet[dst:3] = px[k][src:3];
                sheet[dst + 3] = 255;
            }
        }
    }
    (void)png::write(out_path, sheet, S * 5, S);
    io::printfn("%s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
