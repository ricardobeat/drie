// Several shadowed directional lights. A box on a floor, two shadowed directional lights from
// the front left and the front right: the first is the cascaded sun, the second gets a box in
// a tile of the shadow atlas. Each shadow is darker than the open floor, where they overlap
// (straight behind the box) darker still. A second render with debug_cascades shows that the
// first light still uses its cascades.
//
//   just test shadows [out.png]   (tests/build/shadows.png)
module shadows_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int W = 384;
const int H = 384;
const float FOV = 50;
const float HEIGHT = 8;

int failures;

// The pixel that shows floor point (x, 0, z) from the camera straight above.
fn int[2] pixel_of(float x, float z) {
    float k = (float)H / 2 / (HEIGHT * math::tan(FOV * 0.5f * (float)math::PI / 180));
    return { (int)(W / 2 + x * k), (int)(H / 2 + z * k) };
}

fn float lum(char[] px, float x, float z) {
    int[2] p = pixel_of(x, z);
    float sum;
    // a 5x5 average: the shadows' edges are dithered
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            usz i = ((usz)(p[1] + dy) * W + (usz)(p[0] + dx)) * 4;
            sum += ((float)px[i] + (float)px[i + 1] + (float)px[i + 2]) / 3;
        }
    }
    return sum / 25;
}

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn int main(String[] args) {
    String out_path = args.len > 1 ? args[1] : "tests/build/shadows.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    Scene* scene = drie::scene();
    scene.add(drie::hemisphere_light(0xffffff, 0x404040, 0.6));
    DirectionalLight*[2] lights = { drie::directional_light(0xffffff, 1.5), drie::directional_light(0xffffff, 1.5) };
    lights[0].position = { 4, 5, 3 };
    lights[1].position = { -4, 5, 3 };
    foreach (l : lights) {
        l.cast_shadow = true;
        l.shadow.map_size = 1024;
        l.shadow_extent = 4;
        scene.add(l);
    }
    Mesh* floor = drie::mesh(drie::plane_geometry(20, 20), drie::mesh_standard_material(color: 0x808080));
    floor.rotation.x = -(float)math::PI / 2;
    floor.receive_shadow = true;
    scene.add(floor);
    Mesh* box = drie::mesh(drie::box_geometry(1, 1, 1), drie::mesh_standard_material(color: 0xc08040));
    box.position = { 0, 0.5f, 0 };
    box.cast_shadow = true;
    scene.add(box);

    PerspectiveCamera* camera = drie::perspective_camera(FOV, (float)W / H, 0.1f, 50);
    camera.position = { 0, HEIGHT, 0 };
    camera.rotation.x = -(float)math::PI / 2;
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    char[] px = mem::new_array(char, (usz)W * (usz)H * 4);
    char[] tinted = mem::new_array(char, (usz)W * (usz)H * 4);
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, W, H, px.ptr);
    renderer.debug_cascades = true;
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, W, H, tinted.ptr);

    float open  = lum(px, 0, 2.5f);
    float a     = lum(px, -1.2f, -0.9f);   // in the first light's shadow only
    float b     = lum(px, 1.2f, -0.9f);    // in the second's only
    float both  = lum(px, 0, -0.7f);
    float open2 = lum(px, 2.5f, 1.5f);
    io::printfn("open %.1f / %.1f, first's shadow %.1f, second's %.1f, both %.1f", open, open2, a, b, both);
    expect(a < open - 20, "the first light's (the sun's) shadow");
    expect(b < open - 20, "the second light's shadow (an atlas tile)");
    expect(math::abs(a - b) < 6, "both shadows equally dark");
    expect(both < math::min(a, b) - 10, "darkest where they overlap");
    expect(math::abs(open - open2) < 6, "open floor lit");
    // debug_cascades tints the sun's cascades red, green, blue, yellow: the floor 8 m away
    // is in the second
    int[2] p = pixel_of(0, 2.5f);
    usz i = ((usz)p[1] * W + (usz)p[0]) * 4;
    expect(tinted[i + 1] > tinted[i] + 40 && tinted[i + 1] > tinted[i + 2] + 40, "the sun uses its cascades");

    char[] rgb = mem::new_array(char, (usz)W * (usz)H * 3);
    for (usz k = 0; k < (usz)W * H; k++) rgb[k * 3:3] = px[k * 4:3];
    if (!png::write(out_path, rgb, W, H, channels: 3)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s; %d failed", out_path, failures);
    return failures ? 1 : 0;
}
