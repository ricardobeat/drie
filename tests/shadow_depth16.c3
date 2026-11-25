// The atlas's 16-bit maps (the distance from the light's plane over its reach): a spot light
// that reaches 60 m over a floor, a box standing on it.
//   - the open floor is lit evenly (no acne)
//   - the shadow starts at the box's foot (no gap between them)
//   - the atlas takes 2 bytes a texel
//
//   just test shadow_depth16 [out.png]   (tests/build/shadow_depth16.png)
module shadow_depth16_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int S = 160;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

char[] px;

fn float bright(int x, int y) {
    usz o = ((usz)y * S + (usz)x) * 4;
    return ((float)px[o] + px[o + 1] + px[o + 2]) / 3;
}

// The mean brightness of the pixels from (x0, y0) to (x1, y1) of px, and how much a pixel
// differs from the one beside it on average (speckles of acne: much; light falling off: little).
fn float[2] stats(int x0, int x1, int y0, int y1) {
    float sum;
    float step;
    int n;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            sum += bright(x, y);
            step += math::abs(bright(x + 1, y) - bright(x, y));
            n++;
        }
    }
    return { sum / n, step / n };
}

fn int main(String[] args) {
    String out_path = "tests/build/shadow_depth16.png";
    if (args.len > 1) out_path = args[1];
    Renderer* renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    RenderTarget* rt = drie::render_target(S, S, 1, RGBA8);
    px = mem::new_array(char, (usz)S * S * 4);
    Scene* scene = drie::scene();
    Material* white = drie::mesh_standard_material(color: 0xffffff, roughness: 1);
    Mesh* floor = drie::mesh(drie::box_geometry(20, 0.2f, 20), white);
    floor.position = { 0, -0.1f, 0 };
    floor.receive_shadow = true;
    floor.cast_shadow = true;
    scene.add(floor);
    Mesh* box = drie::mesh(drie::box_geometry(1, 1, 1), white);
    box.position = { 0, 0.5f, 0 };
    box.cast_shadow = true;
    box.receive_shadow = true;
    scene.add(box);
    SpotLight* spot = drie::spot_light(0xffffff, 60, 60, 0.7f, 0.1f, 2);
    spot.position = { -6, 5, 0 };
    spot.target.position = { 2, 0, 0 };
    spot.cast_shadow = true;
    spot.shadow.map_size = 1024;
    // (drawn once: its map is in the atlas; one drawn every frame has a map to itself)
    spot.shadow.auto_update = false;
    spot.shadow.needs_update = true;
    scene.add(spot);
    scene.add(spot.target);
    PerspectiveCamera* camera = drie::perspective_camera(60, 1, 0.05f, 60);
    camera.position = { 0, 8, 0.01f };
    camera.look_at_vec({ 0, 0, 0 });
    for (int n = 0; n < 3000; n++) {
        renderer.set_render_target(rt);
        renderer.render(scene, camera);
        drie::commit();
        if (n > 4 && renderer.info.waiting == 0) break;
    }
    readback::read_image(rt.texture_image, S, S, px.ptr);
    // seen from above: the light comes from -x, the box in the middle, its shadow to +x
    float[2] open = stats(S * 20 / 100, S * 35 / 100, S * 20 / 100, S * 35 / 100);
    float[2] foot = stats(S * 61 / 100, S * 64 / 100, S * 47 / 100, S * 53 / 100);
    float[2] beside = stats(S * 61 / 100, S * 64 / 100, S * 20 / 100, S * 26 / 100);
    expect(open[0] > 40 && open[0] < 250 && open[1] < 2, string::tformat("the open floor is even (brightness %.0f, %.2f from one pixel to the next)", open[0], open[1]));
    expect(foot[0] < beside[0] * 0.5f, string::tformat("the shadow starts at the box's foot (%.0f there, %.0f beside it)", foot[0], beside[0]));
    int[4] r = renderer.shadow_map_rect(spot);
    usz texels = (usz)renderer.info.atlas_width * (usz)renderer.info.atlas_height;
    expect(r[2] > 0 && renderer.info.atlas_bytes < texels * 4, string::tformat("the atlas: %d x %d, %d bytes", renderer.info.atlas_width, renderer.info.atlas_height, renderer.info.atlas_bytes));

    if (!png::write(out_path, px, S, S)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
