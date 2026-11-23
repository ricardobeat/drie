// Headless verification of Split Radiance Cascades (Phase 2: rc.split = true):
//   1. A plane fills >= 95% of L0 bins in a single frame.
//   2. Probe count drops by ~4x per cascade level on a 2D surface (nearest parent nesting).
//   3. All filled bins satisfy 0 <= beta <= 1.
//   4. A closed Cornell-style box converges within 10% of dense RadianceCascades.
// 
//   just test radiance_split_rays
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

fn void test_plane_one_frame(Renderer* renderer, RenderTarget* rt) {
    Scene* scene = drie::scene();
    scene.background = 0x101520;
    HemisphereLight* hemi = drie::hemisphere_light(0x90b0ff, 0x302820, 1.0f);
    scene.add(hemi);

    MeshStandardMaterial* white = drie::mesh_standard_material(color: 0xe0e0e0, roughness: 0.9f);
    Mesh* floor_m               = drie::mesh(drie::box_geometry(8.0f, 0.2f, 8.0f), white);
    floor_m.position = { 0, -0.1f, 0 };
    scene.add(floor_m);

    RadianceCascades* rc = drie::radiance_cascades_sparse(
        { -4.5f, -0.5f, -4.5f },
        { 4.5f, 3.5f, 4.5f },
        0.25f,
        4,
        8192,
        32768
    );
    scene.update_matrix_world();
    rc.voxelize(scene);
    scene.radiance = rc;

    PerspectiveCamera* cam = drie::perspective_camera(60, (float)W / H, 0.1f, 40);
    cam.position = { 0.0f, 3.2f, 3.2f };
    cam.look_at_vec({ 0.0f, 0.0f, -0.5f });

    // Render 1 single frame
    renderer.set_render_target(rt);
    renderer.render(scene, cam);
    drie::commit();

    RcStats s = rc_debug::stats(rc);
    rc_debug::print_stats(&s);

    float l0_fill = s.level[0].bins_total > 0 ? (float)s.level[0].bins_filled / (float)s.level[0].bins_total : 0.0f;
    io::printfn(
        "plane 1-frame L0 bin fill: %.1f%% (%d / %d)",
        l0_fill * 100.0f,
        s.level[0].bins_filled,
        s.level[0].bins_total
    );
    expect(l0_fill >= 0.95f, "a plane fills >= 95% of L0 bins in one frame");

    float ratio01 = s.level[1].live > 0 ? (float)s.level[0].live / (float)s.level[1].live : 0.0f;
    float ratio12 = s.level[2].live > 0 ? (float)s.level[1].live / (float)s.level[2].live : 0.0f;
    io::printfn(
        "plane level probe counts: L0=%d, L1=%d (%.2fx), L2=%d (%.2fx), L3=%d",
        s.level[0].live,
        s.level[1].live,
        ratio01,
        s.level[2].live,
        ratio12,
        s.level[3].live
    );
    expect(ratio01 >= 2.5f && ratio01 <= 5.5f && ratio12 >= 2.2f && ratio12 <= 5.5f, "about 4x fewer probes per level");

    expect(s.min_beta >= 0.0f && s.max_beta <= 1.0f && s.max_beta > 0.5f, "0 <= beta <= 1 across all filled bins");
}

fn float closed_box_mean_luma(Renderer* renderer, RenderTarget* rt, bool dense) {
    Scene* scene = drie::scene();
    scene.background = 0x000000;

    MeshStandardMaterial* white = drie::mesh_standard_material(color: 0xe0e0e0, roughness: 0.9f);
    MeshStandardMaterial* red_glow = drie::mesh_standard_material(
        color: 0xff3020,
        roughness: 0.9f,
        emissive: 0xff3020,
        emissive_intensity: 5.0f
    );

    // Closed 6-wall box: [-3, 3] x [0, 3] x [-3, 3]
    Mesh* floor_m = drie::mesh(drie::box_geometry(6.4f, 0.2f, 6.4f), white);
    floor_m.position = { 0, -0.1f, 0 };
    scene.add(floor_m);

    Mesh* ceil_m = drie::mesh(drie::box_geometry(6.4f, 0.2f, 6.4f), white);
    ceil_m.position = { 0, 3.1f, 0 };
    scene.add(ceil_m);

    Mesh* back_m = drie::mesh(drie::box_geometry(6.4f, 3.2f, 0.2f), white);
    back_m.position = { 0, 1.5f, -3.1f };
    scene.add(back_m);

    Mesh* front_m = drie::mesh(drie::box_geometry(6.4f, 3.2f, 0.2f), white);
    front_m.position = { 0, 1.5f, 3.1f };
    scene.add(front_m);

    Mesh* left_m = drie::mesh(drie::box_geometry(0.2f, 3.2f, 6.4f), red_glow);
    left_m.position = { -3.1f, 1.5f, 0 };
    scene.add(left_m);

    Mesh* right_m = drie::mesh(drie::box_geometry(0.2f, 3.2f, 6.4f), white);
    right_m.position = { 3.1f, 1.5f, 0 };
    scene.add(right_m);

    PointLight* lamp = drie::point_light(0xfff0d0, 6.0f, 12.0f, 2.0f);
    lamp.position = { 0.5f, 2.5f, 0.0f };
    scene.add(lamp);

    Vector3 bmin         = { -3.5f, -0.4f, -3.5f };
    Vector3 bmax         = { 3.5f, 3.4f, 3.5f };
    RadianceCascades* rc = dense ? drie::radiance_cascades(bmin, bmax, 0.25f, 4)
        : drie::radiance_cascades_sparse(bmin, bmax, 0.25f, 4, 8192, 32768);
    rc.sky_from_scene = false;
    rc.sky_color = { 0, 0, 0 };
    rc.ground_color = { 0, 0, 0 };
    rc.blend = 0.25f;
    scene.update_matrix_world();
    rc.voxelize(scene);
    scene.radiance = rc;

    PerspectiveCamera* cam = drie::perspective_camera(55, (float)W / H, 0.1f, 30);
    cam.position = { 0.8f, 1.5f, 2.2f };
    cam.look_at_vec({ -0.5f, 1.4f, -3.0f });

    renderer.set_render_target(rt);
    for (int f = 0; f < 16; f++) {
        if (dense) rc.update(scene);
        renderer.render(scene, cam);
        drie::commit();
    }

    char[] px = mem::new_array(char, (usz)W * H * 4);
    defer free(px);
    readback::read_image(rt.texture_image, W, H, px.ptr);
    if (!dense) png::write("tests/build/radiance_split_rays.png", px, W, H);

    double sum = 0;
    int count  = 0;
    for (int y = H / 4; y < (H * 3) / 4; y++) {
        for (int x = W / 4; x < (W * 3) / 4; x++) {
            usz idx  = ((usz)y * W + (usz)x) * 4;
            double r = (double)(uint)px[idx];
            double g = (double)(uint)px[idx + 1];
            double b = (double)(uint)px[idx + 2];
            sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
            count++;
        }
    }
    return (float)(sum / (double)count);
}

fn int main() {
    Renderer* renderer = drie::renderer(width: W, height: H);
    renderer.init_headless();
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);

    test_plane_one_frame(renderer, rt);

    float luma_dense = closed_box_mean_luma(renderer, rt, dense: true);
    float luma_split = closed_box_mean_luma(renderer, rt, dense: false);
    float rel_diff   = math::abs(luma_split - luma_dense) / math::max(luma_dense, 1.0f);
    io::printfn(
        "closed box center luma: dense=%.2f, split=%.2f (rel diff %.1f%%)",
        luma_dense,
        luma_split,
        rel_diff * 100.0f
    );
    expect(rel_diff <= 0.10f, "a closed box within 10% of dense RC");

    io::printfn(failures ? "%d failed" : "all passed", failures);
    return failures ? 1 : 0;
}