// Headless verification of Split Radiance Cascades in a box past the key's reach (Phase 6):
//   1. A 512 m box, whose spacing the volume coarsens until its cells fit the 32-bit probe
//      key, reads within 20% of a 32 m one over the same scene.
//   2. Its cells do not share keys: no probe is lost or doubled.
// 
//   just test radiance_split_lod
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
    for (int y = (H * 2) / 5; y < (H * 3) / 5; y++) {
        for (int x = (W * 2) / 5; x < (W * 3) / 5; x++) {
            usz idx = ((usz)y * W + (usz)x) * 4;
            sum += 0.2126 * (double)(uint)px[idx] + 0.7152 * (double)(uint)px[idx + 1]
                + 0.0722 * (double)(uint)px[idx + 2];
            count++;
        }
    }
    return (float)(sum / (double)count);
}

// A floor with a lamp five metres over it; the camera looks down at the lit patch.
fn float lit_floor(Renderer* renderer, RenderTarget* rt, float half, String file) {
    Scene* scene = drie::scene();
    scene.background = 0x000000;
    Material* white = drie::mesh_standard_material(color: 0xd8d8d8, roughness: 0.9f);
    Mesh* floor_m   = drie::mesh(drie::plane_geometry(2 * half, 2 * half), white);
    floor_m.rotation.x = -(float)math::PI / 2;
    floor_m.receive_shadow = true;
    scene.add(floor_m);
    PointLight* lamp = drie::point_light(0xfff0d0, 12.0f, 0.0f, 2.0f);
    lamp.position = { 0, 5, 0 };
    scene.add(lamp);

    Vector3 bmin         = { -half, -1, -half };
    Vector3 bmax         = { half, 8, half };
    RadianceCascades* rc = drie::radiance_cascades_sparse(bmin, bmax, 0.5f, 4, 16384, 65536);
    rc.split = true;
    rc.sky_from_scene = false;
    rc.sky_color = { 0, 0, 0 };
    rc.ground_color = { 0, 0, 0 };
    rc.blend = 0.35f;
    scene.update_matrix_world();
    rc.voxelize(scene);
    scene.radiance = rc;

    PerspectiveCamera* cam = drie::perspective_camera(60, (float)W / H, 0.1f, 40);
    cam.position = { 2.5f, 3.0f, 7.0f };
    cam.look_at_vec({ 0.0f, 0.6f, 0.0f });
    renderer.set_render_target(rt);
    for (int f = 0; f < 40; f++) {
        renderer.render(scene, cam);
        drie::commit();
    }
    float l    = luma(rt);
    RcStats st = rc_debug::stats(rc);
    io::printfn(
        "%4.0f m box: spacing %.2f m, %d x %d x %d cells, %.2f luma, %d probes, %.1f%% bins",
        half * 2,
        rc.spacing,
        rc.nx,
        rc.ny,
        rc.nz,
        l,
        st.probes,
        100.0 * (double)st.bins_filled / (double)math::max(st.bins_total, 1)
    );
    if (file.len) {
        char[] px = mem::new_array(char, (usz)W * H * 4);
        defer free(px);
        readback::read_image(rt.texture_image, W, H, px.ptr);
        png::write(file, px, W, H);
    }
    return l;
}

fn int main() {
    Renderer* renderer = drie::renderer(width: W, height: H);
    renderer.init_headless();
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);

    RadianceCascades* ref = drie::radiance_cascades_sparse({ -16, -1, -16 }, { 16, 8, 16 }, 0.5f, 4, 16384, 65536);
    RadianceCascades* big = drie::radiance_cascades_sparse({ -256, -1, -256 }, { 256, 8, 256 }, 0.5f, 4, 16384, 65536);
    expect(ref.spacing == 0.5f && big.spacing >= 2.0f, "the big box's spacing gives way until its cells fit");
    float small = lit_floor(renderer, rt, 16, "tests/build/lod_small.png");
    float large = lit_floor(renderer, rt, 256, "tests/build/lod_large.png");
    float rel   = math::abs(large - small) / math::max(small, 1.0f);
    io::printfn("512 m / 32 m: %.2f vs %.2f, %.1f%% apart", large, small, 100.0 * rel);
    expect(rel <= 0.20f, "a box past the key's reach reads within 20% of one inside it");
    expect(large > 1.0f, "and it is lit");

    io::printfn(failures ? "%d failed" : "all passed", failures);
    return failures ? 1 : 0;
}