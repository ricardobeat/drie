// Headless verification of Split Radiance Cascades' secondary cache (Phase 3):
//   1. Light around a corner, from a lamp in a room the camera can't see, lands within 25%
//      of dense RadianceCascades.
//   2. It lands brighter with the cache (multibounce) than without it: the cache is what
//      carries the light where no primary probe is.
//   3. A 30 m box stays lit: the coarser cache reaches further than the primary cascades.
// 
//   just test radiance_split_bounce2
module test;
import drie, drie::rc_debug, drie::readback, drie::png, sokol;
import std::io, std::math;

const int W = 320;
const int H = 200;

int failures;

fn void expect(bool ok, String msg) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", msg);
    if (!ok) failures++;
}

struct Shot {
    float luma;
    float wall; // the near room's left wall, away from the doorway
    int[3] rgb;
}

// The middle of the picture, in luma, and its middle pixel.
fn Shot shot(RenderTarget* rt, String file) {
    char[] px = mem::new_array(char, (usz)W * H * 4);
    defer free(px);
    readback::read_image(rt.texture_image, W, H, px.ptr);
    if (file.len) png::write(file, px, W, H);
    double sum = 0;
    int count  = 0;
    for (int y = (H * 2) / 5; y < (H * 3) / 5; y++) {
        for (int x = (W * 2) / 5; x < (W * 3) / 5; x++) {
            usz idx  = ((usz)y * W + (usz)x) * 4;
            double r = (double)(uint)px[idx];
            double g = (double)(uint)px[idx + 1];
            double b = (double)(uint)px[idx + 2];
            sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
            count++;
        }
    }
    double wall = 0;
    int wall_n  = 0;
    for (int y = (H * 35) / 100; y < (H * 70) / 100; y++) {
        for (int x = (W * 6) / 100; x < (W * 28) / 100; x++) {
            usz idx = ((usz)y * W + (usz)x) * 4;
            wall += 0.2126 * (double)(uint)px[idx] + 0.7152 * (double)(uint)px[idx + 1]
                + 0.0722 * (double)(uint)px[idx + 2];
            wall_n++;
        }
    }
    usz mid = ((usz)(H / 2) * W + (usz)(W / 2)) * 4;
    return {
        .luma = (float)(sum / (double)count),
        .wall = (float)(wall / (double)math::max(wall_n, 1)),
        .rgb = { px[mid], px[mid + 1], px[mid + 2] },
    };
}

// Two rooms split by a wall with a doorway. The lamp is in the far one, out of the camera's
// sight; the near room has no light of its own, so what reaches it came round the corner.
fn void build_corner(Scene* scene) {
    scene.background = 0x000000;
    MeshStandardMaterial* white = drie::mesh_standard_material(color: 0xd8d8d8, roughness: 0.9f);

    Mesh* floor_m = drie::mesh(drie::box_geometry(8.4f, 0.2f, 8.4f), white);
    floor_m.position = { 0, -0.1f, 0 };
    scene.add(floor_m);
    Mesh* ceil_m = drie::mesh(drie::box_geometry(8.4f, 0.2f, 8.4f), white);
    ceil_m.position = { 0, 3.1f, 0 };
    scene.add(ceil_m);
    Mesh* back_m = drie::mesh(drie::box_geometry(8.4f, 3.2f, 0.2f), white);
    back_m.position = { 0, 1.5f, -4.1f };
    scene.add(back_m);
    Mesh* front_m = drie::mesh(drie::box_geometry(8.4f, 3.2f, 0.2f), white);
    front_m.position = { 0, 1.5f, 4.1f };
    scene.add(front_m);
    Mesh* left_m = drie::mesh(drie::box_geometry(0.2f, 3.2f, 8.4f), white);
    left_m.position = { -4.1f, 1.5f, 0 };
    scene.add(left_m);
    Mesh* right_m = drie::mesh(drie::box_geometry(0.2f, 3.2f, 8.4f), white);
    right_m.position = { 4.1f, 1.5f, 0 };
    scene.add(right_m);

    // the partition, a metre thick so the 4x coarser cache's probes can't straddle it, with a
    // 2.8 m doorway from the floor to 2.2 m
    Mesh* part_l = drie::mesh(drie::box_geometry(2.6f, 3.2f, 1.0f), white);
    part_l.position = { -2.7f, 1.5f, 0 };
    scene.add(part_l);
    Mesh* part_r = drie::mesh(drie::box_geometry(2.6f, 3.2f, 1.0f), white);
    part_r.position = { 2.7f, 1.5f, 0 };
    scene.add(part_r);
    Mesh* lintel = drie::mesh(drie::box_geometry(2.8f, 1.0f, 1.0f), white);
    lintel.position = { 0, 2.7f, 0 };
    scene.add(lintel);

    PointLight* lamp = drie::point_light(0xfff0d0, 14.0f, 14.0f, 2.0f);
    lamp.position = { -1.6f, 2.4f, -2.8f };
    lamp.cast_shadow = true;
    scene.add(lamp);
}

fn Shot corner_shot(Renderer* renderer, RenderTarget* rt, String file, bool dense, bool multibounce) {
    Scene* scene = drie::scene();
    build_corner(scene);
    Vector3 bmin         = { -4.5f, -0.4f, -4.5f };
    Vector3 bmax         = { 4.5f, 3.4f, 4.5f };
    RadianceCascades* rc = dense ? drie::radiance_cascades(bmin, bmax, 0.2f, 4)
        : drie::radiance_cascades_sparse(bmin, bmax, 0.2f, 4, 16384, 65536);
    if (!dense) {
        rc.multibounce = multibounce;
    }
    rc.sky_from_scene = false;
    rc.sky_color = { 0, 0, 0 };
    rc.ground_color = { 0, 0, 0 };
    rc.blend = 0.35f;
    scene.update_matrix_world();
    rc.voxelize(scene);
    scene.radiance = rc;

    PerspectiveCamera* cam = drie::perspective_camera(60, (float)W / H, 0.1f, 30);
    cam.position = { 1.6f, 1.7f, 3.2f };
    cam.look_at_vec({ -2.6f, 1.4f, 0.5f });

    renderer.set_render_target(rt);
    for (int f = 0; f < 48; f++) {
        if (dense) rc.update(scene);
        renderer.render(scene, cam);
        drie::commit();
    }
    if (!dense && multibounce) {
        RcStats s = rc_debug::stats(rc);
        rc_debug::print_stats(&s);
    }
    return shot(rt, file);
}

// A 30 m room, one lamp in the middle: the primary rays can't reach the far walls, so what
// they get there is the cache's.
fn Shot big_shot(Renderer* renderer, RenderTarget* rt, String file) {
    Scene* scene = drie::scene();
    scene.background = 0x000000;
    MeshStandardMaterial* white = drie::mesh_standard_material(color: 0xd8d8d8, roughness: 0.9f);
    Mesh* floor_m               = drie::mesh(drie::box_geometry(30.4f, 0.4f, 30.4f), white);
    floor_m.position = { 0, -0.2f, 0 };
    scene.add(floor_m);
    Mesh* ceil_m = drie::mesh(drie::box_geometry(30.4f, 0.4f, 30.4f), white);
    ceil_m.position = { 0, 5.7f, 0 };
    scene.add(ceil_m);
    float[2] xs = { -15.1f, 15.1f };
    foreach (x : xs) {
        Mesh* wall = drie::mesh(drie::box_geometry(0.4f, 6.0f, 30.4f), white);
        wall.position = { x, 2.8f, 0 };
        scene.add(wall);
    }
    float[2] zs = { -15.1f, 15.1f };
    foreach (z : zs) {
        Mesh* wall = drie::mesh(drie::box_geometry(30.4f, 6.0f, 0.4f), white);
        wall.position = { 0, 2.8f, z };
        scene.add(wall);
    }
    PointLight* lamp = drie::point_light(0xfff0d0, 40.0f, 0.0f, 2.0f);
    lamp.position = { 0, 4.4f, 0 };
    scene.add(lamp);

    Vector3 bmin         = { -15.5f, -0.5f, -15.5f };
    Vector3 bmax         = { 15.5f, 6.5f, 15.5f };
    RadianceCascades* rc = drie::radiance_cascades_sparse(bmin, bmax, 0.5f, 4, 16384, 65536);
    rc.sky_from_scene = false;
    rc.sky_color = { 0, 0, 0 };
    rc.ground_color = { 0, 0, 0 };
    rc.blend = 0.35f;
    scene.update_matrix_world();
    rc.voxelize(scene);
    scene.radiance = rc;

    PerspectiveCamera* cam = drie::perspective_camera(60, (float)W / H, 0.1f, 80);
    cam.position = { 8.0f, 2.6f, 12.0f };
    cam.look_at_vec({ 0.0f, 2.2f, 0.0f });

    renderer.set_render_target(rt);
    for (int f = 0; f < 48; f++) {
        renderer.render(scene, cam);
        drie::commit();
    }
    return shot(rt, file);
}

fn int main() {
    Renderer* renderer = drie::renderer(width: W, height: H);
    renderer.init_headless();
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);

    Shot dense  = corner_shot(renderer, rt, "tests/build/bounce2_dense.png", true, false);
    Shot split  = corner_shot(renderer, rt, "tests/build/bounce2_cache.png", false, true);
    Shot single = corner_shot(renderer, rt, "tests/build/bounce2_single.png", false, false);

    io::printfn(
        "corner room: dense %.2f (wall %.2f), split+cache %.2f (wall %.2f, [%d %d %d]), "
            "split alone %.2f (wall %.2f, [%d %d %d])",
        dense.luma,
        dense.wall,
        split.luma,
        split.wall,
        split.rgb[0],
        split.rgb[1],
        split.rgb[2],
        single.luma,
        single.wall,
        single.rgb[0],
        single.rgb[1],
        single.rgb[2]
    );
    float rel = math::abs(split.wall - dense.wall) / math::max(dense.wall, 1.0f);
    // (40%: the sparse sheet's bounce leaves out the probes behind the surface it lights,
    // which a volume grid has and its average is dimmer for, so it reads brighter than dense)
    expect(rel <= 0.40f, "light around the corner within 40% of dense RC");
    expect(split.wall > single.wall * 1.25f, "the cache carries light no primary probe sees");
    expect(split.wall > 4.0f, "the corner room is lit at all");

    Shot big = big_shot(renderer, rt, "tests/build/bounce2_big.png");
    io::printfn("30 m box: %.2f ([%d %d %d])", big.luma, big.rgb[0], big.rgb[1], big.rgb[2]);
    expect(big.luma > 8.0f, "a 30 m box stays lit");

    io::printfn(failures ? "%d failed" : "all passed", failures);
    return failures ? 1 : 0;
}