// Headless verification of Split Radiance Cascades' leak control (Phase 4):
//   1. Two sealed rooms side by side, one lamp in the left one: the right room, which no ray
//      of the lamp's light can reach, stays under 3% of the left room's brightness.
//   2. A node material takes the cascades' light: the same view is brighter with them on
//      than with their intensity at zero.
// 
//   just test radiance_split_leak
module test;
import drie, drie::nodes, drie::rc_debug, drie::readback, drie::png, sokol;
import std::io, std::math;

const int W = 320;
const int H = 200;

int failures;

fn void expect(bool ok, String msg) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", msg);
    if (!ok) failures++;
}

// The middle of the picture, in luma, and its middle pixel.
struct Shot {
    float luma;
    int[3] rgb;
}

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
    usz mid = ((usz)(H / 2) * W + (usz)(W / 2)) * 4;
    return {
        .luma = (float)(sum / (double)count),
        .rgb = { px[mid], px[mid + 1], px[mid + 2] },
    };
}

const float ROOM  = 4.2f; // each room is ROOM wide, on either side of the partition
const float DEPTH = 8.4f;
const float TALL  = 3.2f;

// Two sealed rooms sharing a 0.3 m wall. The lamp is in the left one; the right one gets
// nothing but what leaks.
fn void add_wall(Scene* scene, Mesh* m) {
    m.cast_shadow = true;
    m.receive_shadow = true;
    scene.add(m);
}

fn void build_rooms(Scene* scene, Material* white) {
    scene.background = 0x000000;
    Mesh* floor_m = drie::mesh(drie::box_geometry(2 * ROOM + 0.3f, 0.2f, DEPTH), white);
    floor_m.position = { 0, -0.1f, 0 };
    add_wall(scene, floor_m);
    Mesh* ceil_m = drie::mesh(drie::box_geometry(2 * ROOM + 0.3f, 0.2f, DEPTH), white);
    ceil_m.position = { 0, TALL + 0.1f, 0 };
    add_wall(scene, ceil_m);
    float[2] zs = { -(DEPTH / 2 + 0.1f), DEPTH / 2 + 0.1f };
    foreach (z : zs) {
        Mesh* wall = drie::mesh(drie::box_geometry(2 * ROOM + 0.3f, TALL, 0.2f), white);
        wall.position = { 0, TALL / 2, z };
        add_wall(scene, wall);
    }
    float[2] xs = { -(ROOM + 0.15f), ROOM + 0.15f };
    foreach (x : xs) {
        Mesh* wall = drie::mesh(drie::box_geometry(0.2f, TALL, DEPTH), white);
        wall.position = { x, TALL / 2, 0 };
        add_wall(scene, wall);
    }
    Mesh* part = drie::mesh(drie::box_geometry(0.3f, TALL, DEPTH), white);
    part.position = { 0, TALL / 2, 0 };
    add_wall(scene, part);

    PointLight* lamp = drie::point_light(0xfff0d0, 6.0f, 0.0f, 2.0f);
    lamp.position = { -(ROOM / 2), 2.4f, -1.5f };
    lamp.cast_shadow = true;
    scene.add(lamp);
}

// The same view from inside the left room and from inside the right one, mirrored.
fn Shot room_shot(
    Renderer* renderer,
    RenderTarget* rt,
    String file,
    bool dense,
    bool right,
    float intensity,
    bool node_material,
    bool multibounce = true,
    bool none = false
) {
    Scene* scene    = drie::scene();
    Material* white = drie::mesh_standard_material(color: 0xd8d8d8, roughness: 0.9f);
    if (node_material) {
        // a node material: its own lighting graph (materials_nodes.c3) instead of the fixed one
        white.color_node = nodes::color(0xd8d8d8);
        white.roughness_node = nodes::num(0.9f);
    }
    build_rooms(scene, white);
    Vector3 bmin         = { -(ROOM + 0.4f), -0.4f, -(DEPTH / 2 + 0.4f) };
    Vector3 bmax         = { ROOM + 0.4f, TALL + 0.4f, DEPTH / 2 + 0.4f };
    RadianceCascades* rc = dense ? drie::radiance_cascades(bmin, bmax, 0.2f, 4)
        : drie::radiance_cascades_sparse(bmin, bmax, 0.2f, 4, 16384, 65536);
    if (!dense) {
        rc.multibounce = multibounce;
    }
    rc.intensity = intensity;
    rc.sky_from_scene = false;
    rc.sky_color = { 0, 0, 0 };
    rc.ground_color = { 0, 0, 0 };
    rc.blend = 0.35f;
    scene.update_matrix_world();
    if (!none) {
        rc.voxelize(scene);
        scene.radiance = rc;
    }

    float side             = right ? 1.0f : -1.0f;
    PerspectiveCamera* cam = drie::perspective_camera(60, (float)W / H, 0.1f, 30);
    cam.position = { side * 2.2f, 1.7f, 2.6f };
    cam.look_at_vec({ side * 2.6f, 1.2f, -1.0f });

    renderer.set_render_target(rt);
    for (int f = 0; f < 48; f++) {
        if (dense && rc) rc.update(scene);
        renderer.render(scene, cam);
        drie::commit();
    }
    if (!none && !dense && !right) {
        RcStats s = rc_debug::stats(rc);
        rc_debug::print_stats(&s);
    }
    return shot(rt, file);
}

fn int main() {
    Renderer* renderer = drie::renderer(width: W, height: H);
    renderer.init_headless();
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);

    Shot lit  = room_shot(renderer, rt, "tests/build/leak_lit.png", false, false, 1.0f, false);
    Shot dark = room_shot(renderer, rt, "tests/build/leak_dark.png", false, true, 1.0f, false);
    io::printfn(
        "split: lit %.2f, dark %.2f (%.2f%% of lit)",
        lit.luma,
        dark.luma,
        100.0f * dark.luma / math::max(lit.luma, 1.0f)
    );
    expect(dark.luma <= lit.luma * 0.03f, "the sealed room stays under 3% of the lit one");

    Shot node_on  = room_shot(renderer, rt, "tests/build/leak_node_on.png", false, false, 1.0f, true);
    Shot node_off = room_shot(renderer, rt, "tests/build/leak_node_off.png", false, false, 0.0f, true);
    io::printfn("node material: %.2f with the cascades, %.2f without", node_on.luma, node_off.luma);
    expect(node_on.luma > node_off.luma * 1.15f + 0.5f, "a node material receives the cascades' light");

    Shot dense_lit  = room_shot(renderer, rt, "tests/build/leak_dense.png", true, false, 1.0f, false);
    Shot dense_dark = room_shot(renderer, rt, "tests/build/leak_dense_dark.png", true, true, 1.0f, false);
    io::printfn(
        "dense: lit %.2f, dark %.2f (%.2f%% of lit)",
        dense_lit.luma,
        dense_dark.luma,
        100.0f * dense_dark.luma / math::max(dense_lit.luma, 1.0f)
    );
    Shot plain_lit = room_shot(renderer, rt, "tests/build/leak_plain_lit.png", false, false, 1.0f, false, true, true);
    io::printfn("direct light alone: lit %.2f", plain_lit.luma);
    float rel = math::abs(lit.luma - dense_lit.luma) / math::max(dense_lit.luma, 1.0f);
    expect(rel <= 0.25f, "the lit room within 25% of dense RC");

    io::printfn(failures ? "%d failed" : "all passed", failures);
    return failures ? 1 : 0;
}