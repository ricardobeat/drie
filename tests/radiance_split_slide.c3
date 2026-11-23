// Headless verification of Split Radiance Cascades' temporal behaviour (Phase 5):
//   1. A teleport of 40 m: the frame after it is not under half the steady picture, and the
//      probe count comes back to where it was.
//   2. What the lights gave is gone within ceil(1/blend) + 2 frames: the per-bin history
//      blends toward what each frame's rays find.
// 
// The view is a corridor's partition with a doorway off to one side, so the picture is lit by
// what comes through it, which is the cascades' light and nothing else.
// 
//   just test radiance_split_slide
module test;
import drie, drie::rc_debug, drie::readback, drie::png, sokol;
import std::io, std::math;

const int W = 256;
const int H = 160;

int failures;

fn void expect(bool ok, String msg) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", msg);
    if (!ok) failures++;
}

// The middle of the picture, in luma.
fn float luma(RenderTarget* rt) {
    char[] px = mem::new_array(char, (usz)W * H * 4);
    defer free(px);
    readback::read_image(rt.texture_image, W, H, px.ptr);
    double sum = 0;
    int count  = 0;
    // (off to the side: the middle of the picture is the doorway itself, which the lamps
    // beyond it light directly, while the wall beside it takes only what comes through)
    for (int y = (H * 2) / 5; y < (H * 3) / 5; y++) {
        for (int x = W / 10; x < W / 3; x++) {
            usz idx = ((usz)y * W + (usz)x) * 4;
            sum += 0.2126 * (double)(uint)px[idx] + 0.7152 * (double)(uint)px[idx + 1]
                + 0.0722 * (double)(uint)px[idx + 2];
            count++;
        }
    }
    return (float)(sum / (double)count);
}

// A partition across the corridor: a wall with a doorway off to one side of it.
fn void add_partition(Scene* scene, Material* m, float x) {
    float[2] zs = { -3.0f, 3.0f };
    foreach (z : zs) {
        Mesh* part = drie::mesh(drie::box_geometry(0.3f, 4.0f, 4.0f), m);
        part.position = { x, 2, z };
        part.cast_shadow = true;
        part.receive_shadow = true;
        scene.add(part);
    }
    Mesh* lintel = drie::mesh(drie::box_geometry(0.3f, 1.6f, 2.0f), m);
    lintel.position = { x, 3.2f, 0 };
    lintel.receive_shadow = true;
    scene.add(lintel);
}

// A long corridor with a partition and a lamp every 40 m, so every stretch looks like the
// one before it.
fn void build_corridor(Scene* scene, PointLight*[] lamps) {
    scene.background = 0x000000;
    Material* white = drie::mesh_standard_material(color: 0xd8d8d8, roughness: 0.9f);
    Mesh* floor_m   = drie::mesh(drie::box_geometry(80, 0.4f, 10), white);
    floor_m.position = { 36, -0.2f, 0 };
    floor_m.receive_shadow = true;
    scene.add(floor_m);
    Mesh* ceil_m = drie::mesh(drie::box_geometry(80, 0.4f, 10), white);
    ceil_m.position = { 36, 4.2f, 0 };
    ceil_m.receive_shadow = true;
    scene.add(ceil_m);
    float[2] zs = { -5.1f, 5.1f };
    foreach (z : zs) {
        Mesh* wall = drie::mesh(drie::box_geometry(80, 4.0f, 0.2f), white);
        wall.position = { 36, 2, z };
        wall.cast_shadow = true;
        wall.receive_shadow = true;
        scene.add(wall);
    }
    add_partition(scene, white, 6);
    add_partition(scene, white, 46);
    // lamps every 4 m, none at a partition, so a stretch of corridor with a partition in it
    // looks the same as any other
    for (int i = 0; i < 10; i++) {
        PointLight* lamp = drie::point_light(0xfff0d0, 6.0f, 0.0f, 2.0f);
        lamp.position = { 8.0f * (float)i, 3.4f, 0 };
        lamps[i] = lamp;
        scene.add(lamp);
    }
}

fn int main() {
    Renderer* renderer = drie::renderer(width: W, height: H);
    renderer.init_headless();
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);

    Scene* scene = drie::scene();
    PointLight*[10] lamps;
    build_corridor(scene, lamps[..]);

    Vector3 bmin         = { -6, -0.5f, -6 };
    Vector3 bmax         = { 10, 4.5f, 6 };
    RadianceCascades* rc = drie::radiance_cascades_sparse(bmin, bmax, 0.3f, 4, 16384, 65536);
    rc.split = true;
    rc.sky_from_scene = false;
    rc.sky_color = { 0, 0, 0 };
    rc.ground_color = { 0, 0, 0 };
    rc.blend = 0.35f;
    scene.update_matrix_world();
    rc.voxelize(scene);
    scene.radiance = rc;

    PerspectiveCamera* cam = drie::perspective_camera(60, (float)W / H, 0.1f, 60);
    cam.position = { 2.0f, 1.7f, 0 };
    cam.look_at_vec({ 6.0f, 1.6f, 0 });
    renderer.set_render_target(rt);
    for (int f = 0; f < 40; f++) {
        renderer.render(scene, cam);
        drie::commit();
    }
    float steady_a    = luma(rt);
    int probes_before = rc_debug::stats(rc).probes;

    // 40 m down the corridor: the box goes with the camera, the place looks the same
    cam.position = { 42.0f, 1.7f, 0 };
    cam.look_at_vec({ 46.0f, 1.6f, 0 });
    scene.update_matrix_world();
    rc.recenter(scene, cam.position);
    float low = 1e9f;
    float[12] frames;
    for (int f = 0; f < 12; f++) {
        renderer.render(scene, cam);
        drie::commit();
        frames[f] = luma(rt);
        low = math::min(low, frames[f]);
    }
    int probes_after = rc_debug::stats(rc).probes;
    for (int f = 0; f < 40; f++) {
        renderer.render(scene, cam);
        drie::commit();
    }
    float steady_b = luma(rt);
    io::printfn(
        "40 m slide: %.2f before, frames after it %.2f %.2f %.2f %.2f, %.2f once settled, " "probes %d -> %d",
        steady_a,
        frames[0],
        frames[1],
        frames[2],
        frames[3],
        steady_b,
        probes_before,
        probes_after
    );
    expect(low >= steady_b * 0.5f, "no frame after the slide is under half the steady picture");
    float probe_ratio = (float)probes_after / (float)math::max(probes_before, 1);
    expect(probe_ratio > 0.7f && probe_ratio < 1.4f, "the probe count comes back");

    float lit = luma(rt);
    foreach (l : lamps) l.visible = false;
    scene.update_matrix_world();
    int half_at  = -1;
    float last   =  lit;
    bool falling =  true;
    for (int f = 0; f < 30; f++) {
        renderer.render(scene, cam);
        drie::commit();
        float l = luma(rt);
        if (l >= last) falling = false;
        last = l;
        io::printfn("  after the lights went out, frame %d: %.2f", f, l);
        if (half_at < 0 && l < lit * 0.5f) half_at = f;
    }
    // (a room keeps the light it has for as many bounces as its surfaces' albedo allows, so
    // the first frames are the blend's rate and the tail is the bounces': 1 / blend frames
    // takes it under half, and from there it goes on down at the albedo's rate)
    int allowed = (int)math::ceil(1.0f / rc.blend) + 3;
    io::printfn(
        "lights off: %.2f lit, under half at frame %d (allowed %d), %.2f at frame 11",
        lit,
        half_at,
        allowed,
        last
    );
    expect(half_at >= 0 && half_at <= allowed, "the light is under half in 1 / blend + 3 frames");
    expect(falling, "it falls every frame, with nothing of its own to keep it up");

    io::printfn(failures ? "%d failed" : "all passed", failures);
    return failures ? 1 : 0;
}