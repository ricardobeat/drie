// The casters that keep still, cached apart from those that move (AtlasMap's cache): a room with
// four boxes, a point light drawn every frame (shadow.auto_update) and a spot light drawn once,
// and a fifth box that moves.
//   - while the box moves, only it is drawn into the maps, each frame, and the picture is the
//     one the maps drawn whole give
//   - once it stops, nothing is drawn
//   - a box that kept still moves to another place and stops there: its maps' caches are drawn
//     again, then nothing; its shadow is where the maps drawn whole put it
//
//   just test shadow_cache [out.png]   (tests/build/shadow_cache.png)
module shadow_cache_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int S = 160;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

Renderer* renderer;
Scene* scene;
Camera* camera;
RenderTarget* rt;
PointLight* bulb;
SpotLight* spot;
char[] px;
char[] ref;

fn void frame() {
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
}

// The picture with every map drawn whole into `into`.
fn void whole(char[] into) {
    bulb.shadow.needs_update = true;
    spot.shadow.needs_update = true;
    frame();
    readback::read_image(rt.texture_image, S, S, into.ptr);
}

// The mean difference of px from ref, per channel.
fn double difference() {
    double d;
    for (usz i = 0; i < px.len; i++) d += math::abs((double)px[i] - (double)ref[i]);
    return d / (double)px.len;
}

fn int main(String[] args) {
    String out_path = "tests/build/shadow_cache.png";
    if (args.len > 1) out_path = args[1];
    renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    rt = drie::render_target(S, S, 1, RGBA8);
    px = mem::new_array(char, (usz)S * S * 4);
    ref = mem::new_array(char, (usz)S * S * 4);
    scene = drie::scene();
    Material* white = drie::mesh_standard_material(color: 0xffffff, roughness: 1);
    Mesh* floor = drie::mesh(drie::box_geometry(12, 0.2f, 12), white);
    floor.position = { 0, -0.1f, 0 };
    floor.receive_shadow = true;
    floor.cast_shadow = true;
    scene.add(floor);
    Mesh*[4] boxes;
    for (int i = 0; i < 4; i++) {
        Mesh* b = drie::mesh(drie::box_geometry(0.5f, 1, 0.5f), white);
        b.position = { (float)(i % 2) * 3 - 1.5f, 0.5f, (float)(i / 2) * 3 - 1.5f };
        b.cast_shadow = true;
        b.receive_shadow = true;
        scene.add(b);
        boxes[i] = b;
    }
    Mesh* mover = drie::mesh(drie::box_geometry(0.4f, 0.8f, 0.4f), drie::mesh_standard_material(color: 0x4080ff));
    mover.position = { 0, 0.4f, 1 };
    mover.cast_shadow = true;
    mover.receive_shadow = true;
    scene.add(mover);
    bulb = drie::point_light(0xffffff, 20, 0, 2);
    bulb.position = { 0, 2, 0 };
    bulb.cast_shadow = true;
    bulb.shadow.map_size = 256;
    scene.add(bulb);
    spot = drie::spot_light(0xffd0a0, 60, 0, 0.8f, 0.3f, 2);
    spot.position = { 2, 4, 2 };
    spot.cast_shadow = true;
    spot.shadow.auto_update = false;
    spot.shadow.needs_update = true;
    scene.add(spot);
    // (the spot light the renderer draws every frame on its own: away from the room)
    PerspectiveCamera* cam = drie::perspective_camera(70, 1, 0.05f, 40);
    cam.position = { 0, 5, 6 };
    cam.look_at_vec({ 0, 0, 0 });
    camera = cam;

    frame();
    for (int n = 0; n < 3000 && renderer.info.waiting > 0; n++) frame();
    for (int n = 0; n < 12; n++) frame();

    // the box moves round the light
    int most;
    int least = 1 << 30;
    for (int f = 0; f < 30; f++) {
        float a = (float)f * 0.1f;
        mover.position = { math::cos(a) * 1.2f, 0.4f, math::sin(a) * 1.2f };
        frame();
        if (f < 2) continue;
        most = math::max(most, renderer.info.atlas_casters);
        least = math::min(least, renderer.info.atlas_casters);
    }
    readback::read_image(rt.texture_image, S, S, px.ptr);
    whole(ref);
    double moving_diff = difference();
    expect(least > 0 && most <= 7, string::tformat("a box moving: %d to %d casters drawn a frame (only it, into the maps it is in)", least, most));
    expect(moving_diff < 0.5, string::tformat("...and the picture is the one the maps drawn whole give (mean difference %.3f)", moving_diff));

    // it stops
    for (int f = 0; f < 14; f++) frame();
    int after_stop = renderer.info.atlas_casters;
    expect(after_stop == 0, string::tformat("once it stopped: %d casters drawn", after_stop));

    // a box that kept still goes elsewhere and stays there
    boxes[0].position = { -1.5f, 0.5f, 0.2f };
    int changed;
    for (int f = 0; f < 14; f++) {
        frame();
        changed += renderer.info.atlas_casters;
    }
    frame();
    int settled = renderer.info.atlas_casters;
    readback::read_image(rt.texture_image, S, S, px.ptr);
    whole(ref);
    double moved_diff = difference();
    expect(changed > 0 && settled == 0, string::tformat("a still box moved: %d casters drawn while it settled, then %d", changed, settled));
    expect(moved_diff < 0.5, string::tformat("...and its shadow is where the maps drawn whole put it (mean difference %.3f)", moved_diff));

    if (!png::write(out_path, px, S, S)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
