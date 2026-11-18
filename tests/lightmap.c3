// Light maps (three's lightMap), in a scene without lights. Three white planes, each with a
// 2x1 light map: left texel 0.5 * PI (grey), right texel (0.2, 0.4, 0.6) * PI.
//   0  BASIC: the indirect light 1 becomes map / PI: 0.5 on the left, (0.2, 0.4, 0.6) on the right
//   1  Phong: irradiance = map, times diffuse / PI: the same values
//   2  Phong, the map on channel 1, uv1 = (0.75, 0.5) everywhere: (0.2, 0.4, 0.6) all over
// No tone mapping; the values are read back through sRGB.
//
//   just test lightmap [out.png]   (tests/build/lightmap.png)
module lightmap_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int W = 384;
const int H = 128;

int failures;

fn Texture* light_map_texture(char channel) {
    float pi = (float)math::PI;
    float[8] px = { 0.5f * pi, 0.5f * pi, 0.5f * pi, 1, 0.2f * pi, 0.4f * pi, 0.6f * pi, 1 };
    Texture* t = drie::float_texture(px[..], 2, 1);
    t.nearest = true;
    t.generate_mipmaps = false;
    t.channel = channel;
    return t;
}

fn float to_linear(char c) {
    float v = (float)c / 255;
    return v <= 0.04045f ? v / 12.92f : math::pow((v + 0.055f) / 1.055f, 2.4f);
}

fn void expect(char[] px, String what, int x, int y, float r, float g, float b) {
    usz i = ((usz)y * W + (usz)x) * 4;
    float[3] got = { to_linear(px[i]), to_linear(px[i + 1]), to_linear(px[i + 2]) };
    float[3] want = { r, g, b };
    bool ok = true;
    for (int k = 0; k < 3; k++) {
        if (math::abs(got[k] - want[k]) > 0.02f + want[k] * 0.03f) ok = false;
    }
    io::printfn("%s %s: (%.3f, %.3f, %.3f), expected (%.3f, %.3f, %.3f)", ok ? "ok  " : "FAIL", what, got[0], got[1], got[2], r, g, b);
    if (!ok) failures++;
}

fn int main(String[] args) {
    String out_path = args.len > 1 ? args[1] : "tests/build/lightmap.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.tone_mapping = NO_TONE_MAPPING;
    Scene* scene = drie::scene();

    Material*[3] mats = {
        drie::mesh_basic_material(color: 0xffffff),
        drie::mesh_phong_material(color: 0xffffff),
        drie::mesh_phong_material(color: 0xffffff),
    };
    mats[0].light_map = light_map_texture(0);
    mats[1].light_map = light_map_texture(0);
    mats[2].light_map = light_map_texture(1);
    for (int i = 0; i < 3; i++) {
        PlaneGeometry* g = drie::plane_geometry(0.9f, 0.9f);
        if (i == 2) {
            for (uint v = 0; v < g.vertex_count(); v++) {
                g.uv1.push(0.75f);
                g.uv1.push(0.5f);
            }
        }
        Mesh* plane = drie::mesh(g, mats[i]);
        plane.position.set((float)(i - 1), 0, 0);
        scene.add(plane);
    }
    OrthographicCamera* camera = drie::orthographic_camera(-1.5f, 1.5f, 0.5f, -0.5f, 0.1f, 10);
    camera.position.set(0, 0, 2);
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();

    char[] px = mem::new_array(char, (usz)W * (usz)H * 4);
    readback::read_image(rt.texture_image, W, H, px.ptr);
    // each plane is 0.9 wide around x = -1, 0, 1 (128 pixels per unit); a quarter in from each side
    int y = H / 2;
    expect(px, "basic, left texel ", 64 - 29, y, 0.5f, 0.5f, 0.5f);
    expect(px, "basic, right texel", 64 + 29, y, 0.2f, 0.4f, 0.6f);
    expect(px, "phong, left texel ", 192 - 29, y, 0.5f, 0.5f, 0.5f);
    expect(px, "phong, right texel", 192 + 29, y, 0.2f, 0.4f, 0.6f);
    expect(px, "uv1, left         ", 320 - 29, y, 0.2f, 0.4f, 0.6f);
    expect(px, "uv1, right        ", 320 + 29, y, 0.2f, 0.4f, 0.6f);

    char[] rgb = mem::new_array(char, (usz)W * (usz)H * 3);
    for (usz i = 0; i < (usz)W * H; i++) rgb[i * 3:3] = px[i * 4:3];
    if (!png::write(out_path, rgb, W, H, channels: 3)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s; %d failed", out_path, failures);
    return failures ? 1 : 0;
}
