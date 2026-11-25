// Ray-traced rect area light shadows (RectAreaLight.shadow.ray_traced, ray_shadows.c3), seen
// from above: a light facing down over a floor, a post standing to its side.
//   - the post's shadow is sharp at its foot and wider far from it, and wider under a wider
//     light
//   - from a still camera the picture settles: one frame much as the next
//   - a box that moves leaves no shadow where it was
//   - with Renderer.ray_traced_shadows off, the shadow maps still shadow the floor
// Skipped (passing) where the GPU has no ray tracing.
//
//   just test rt_shadows [out.png]   (tests/build/rt_shadows.png)
module rt_shadows_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int S = 256;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

Renderer* renderer;
RenderTarget* rt;
Scene* scene;
Camera* camera;
char[] px;
char[] last;

fn void frames(int n) {
    for (int k = 0; k < n; k++) {
        renderer.set_render_target(rt);
        renderer.render(scene, camera);
        drie::commit();
    }
    readback::read_image(rt.texture_image, S, S, px.ptr);
}

fn void settle() {
    frames(1);
    for (int n = 0; n < 3000 && renderer.info.waiting > 0; n++) frames(1);
    frames(24);
}

fn float at(int x, int y) {
    usz o = ((usz)y * S + (usz)x) * 4;
    return ((float)px[o] + px[o + 1] + px[o + 2]) / 3;
}

fn float mean(int x0, int x1, int y0, int y1) {
    float sum;
    int n;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            sum += at(x, y);
            n++;
        }
    }
    return sum / n;
}

// The width (pixels) of the shadow's edge across column `x`, from row y0 to y1: the rows
// between a fifth and four fifths of the way from the darkest to the brightest (averaged over
// three columns).
fn int edge(int x, int y0, int y1) {
    float lo = 1e9f;
    float hi = 0;
    for (int y = y0; y < y1; y++) {
        float v = (at(x - 1, y) + at(x, y) + at(x + 1, y)) / 3;
        lo = math::min(lo, v);
        hi = math::max(hi, v);
    }
    int n;
    for (int y = y0; y < y1; y++) {
        float v = (at(x - 1, y) + at(x, y) + at(x + 1, y)) / 3;
        if (v > lo + (hi - lo) * 0.2f && v < lo + (hi - lo) * 0.8f) n++;
    }
    return n;
}

fn int main(String[] args) {
    String out_path = "tests/build/rt_shadows.png";
    if (args.len > 1) out_path = args[1];
    renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    rt = drie::render_target(S, S, 1, RGBA8);
    px = mem::new_array(char, (usz)S * S * 4);
    last = mem::new_array(char, (usz)S * S * 4);
    scene = drie::scene();
    Material* white = drie::mesh_standard_material(color: 0xffffff, roughness: 1);
    Mesh* floor = drie::mesh(drie::box_geometry(12, 0.2f, 12), white);
    floor.position = { 0, -0.1f, 0 };
    floor.receive_shadow = true;
    scene.add(floor);
    // a post 0.3 m square and 1.5 m tall, 1 m to the light's +x
    Mesh* post = drie::mesh(drie::box_geometry(0.3f, 1.5f, 0.3f), white);
    post.position = { 1, 0.75f, 0 };
    post.cast_shadow = true;
    scene.add(post);
    RectAreaLight* panel = drie::rect_area_light(0xffffff, 40, 0.5f, 0.5f);
    panel.position = { 0, 2.5f, 0 };
    panel.look_at_vec({ 0, 0, 0 });
    panel.cast_shadow = true;
    scene.add(panel);
    PerspectiveCamera* cam = drie::perspective_camera(50, 1, 0.1f, 30);
    cam.position = { 2, 7, 0.001f };
    cam.look_at_vec({ 2, 0, 0 });
    camera = cam;
    settle();
    if (!renderer.traces_rays()) {
        io::printfn("ok   no ray tracing on this GPU: skipped");
        return 0;
    }
    // the shadow runs along +x from the post (screen x = 39 + 39 * world x at the floor; the
    // post's far side at x = 1.15, px 84; its shadow's tip at 2.9, px 152)
    int near_x = 92;
    int far_x = 135;
    int small_near = edge(near_x, 80, 176);
    int small_far = edge(far_x, 80, 176);
    if (!png::write(out_path, px, S, S)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    // stable from a still camera
    last[..] = px[..];
    frames(1);
    double moved;
    for (usz i = 0; i < px.len; i++) moved += math::abs((double)px[i] - (double)last[i]);
    moved /= (double)px.len;
    // a wider light
    panel.width = 2;
    panel.height = 2;
    panel.intensity = 2.5f;
    frames(24);
    int wide_far = edge(far_x, 80, 176);
    expect(small_far > small_near, string::tformat("the edge widens away from the post: %d px near it, %d far", small_near, small_far));
    expect(wide_far > small_far, string::tformat("...and under a wider light: %d px far from the post (%d under the small one)", wide_far, small_far));
    expect(moved < 1.0, string::tformat("from a still camera, one frame much as the last (mean difference %.2f)", moved));

    // a box that moves: no trail
    panel.width = 0.5f;
    panel.height = 0.5f;
    panel.intensity = 40;
    post.visible = false;
    Mesh* box = drie::mesh(drie::box_geometry(0.3f, 0.3f, 0.3f), white);
    box.cast_shadow = true;
    scene.add(box);
    for (int f = 0; f <= 20; f++) {
        box.position = { 0.5f + (float)f * 0.05f, 0.6f, 0 };
        frames(1);
    }
    // its shadow at x ~ 2 now (px 116); at the start it was at ~0.66 (px 65), lit again
    float was = mean(60, 70, 124, 132);
    float open = mean(60, 70, 88, 96);
    float now = mean(110, 122, 124, 132);
    expect(was > open * 0.8f && now < open * 0.6f,
           string::tformat("a box that moved: %.0f where it was (open %.0f), %.0f where it is", was, open, now));

    // the maps instead
    renderer.ray_traced_shadows = false;
    settle();
    float maps = mean(110, 122, 124, 132);
    expect(maps < open * 0.6f, string::tformat("without ray tracing the maps shadow it (%.0f, open %.0f)", maps, open));

    io::printfn("saved %s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
