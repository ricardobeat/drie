// Transmission and dispersion (Material.transmission, KHR_materials_transmission and
// KHR_materials_dispersion): a glass sphere in front of a wall of colored stripes, which is
// what the sphere refracts. Columns: dispersion off, and on: each color channel is bent by its
// own index of refraction, so the stripes' edges fringe with rainbow colors.
// 
//   just test dispersion [out.png] [--p3]   (tests/build/dispersion.png)
module dispersion_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int W     = 480;
const int H     = 270;
const int MODES = 2;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
RenderTarget* rt;
MeshStandardMaterial* glass;

fn void build_scene() {
    scene = drie::scene();
    scene.background = 0x080a10;
    scene.add(drie::hemisphere_light(0x8090b0, 0x202028, 0.5));
    DirectionalLight* sun = drie::directional_light(0xffffff, 2);
    sun.position.set(1.5f, 2, 1);
    scene.add(sun);

    // the wall behind: vertical stripes of emissive colors, high contrast so that a channel
    // bent into its neighbor shows
    Color[7] colors        = { 0xff2010, 0xffa010, 0xf0e020, 0x20d040, 0x20c0e0, 0x3040ff, 0xffffff };
    BufferGeometry* stripe = drie::box_geometry(0.14f, 6.0f, 0.02f);
    for (int i = 0; i < 31; i++) {
        float x = -2.2f + (float)i * 0.145f;
        Mesh* bar = drie::mesh(
            stripe,
            drie::mesh_lambert_material(color: 0x000000, emissive: colors[i % 7], emissive_intensity: 2)
        );
        bar.position.set(x, 0, -1.8f);
        scene.add(bar);
    }
    // a dull floor, so that the sphere has something to stand over
    MeshStandardMaterial* floor_mat = drie::mesh_standard_material(color: 0x30343c, roughness: 0.8);
    Mesh* floor                     = drie::mesh(drie::plane_geometry(8, 8), floor_mat);
    floor.rotation.x = -drie::PI / 2;
    floor.position.y = -0.7f;
    scene.add(floor);

    // the glass: nothing of the surface itself, all of it the picture behind, bent
    glass = drie::mesh_standard_material(
        color: 0xffffff,
        roughness: 0.02f,
        transmission: 1,
        thickness: 0.4f,
        ior: 1.5f,
        attenuation_color: 0xdff0ff,
        attenuation_distance: 1.5f
    );
    Mesh* ball = drie::mesh(drie::sphere_geometry(0.45f, 64, 32), glass);
    ball.position.set(0, 0, 0);
    scene.add(ball);
}

fn void render_column(int mode) {
    glass.dispersion = mode == 0 ? 0 : 6;
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
}

fn int main(String[] args) {
    String out_path = "tests/build/dispersion.png";
    bool p3;
    foreach (arg : args[1..]) {
        if (arg == "--p3") {
            p3 = true;
        } else {
            out_path = arg;
        }
    }
    renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.tone_mapping = NEUTRAL_TONE_MAPPING;
    renderer.output_color_space = p3 ? DISPLAY_P3_COLOR_SPACE : SRGB_COLOR_SPACE;
    build_scene();
    camera = drie::perspective_camera(45, (float)W / H, 0.01f, 30);
    camera.position.set(0, 0.2f, 1.7f);
    camera.look_at_vec({ 0, 0, 0 });
    rt = drie::render_target(W, H, 1, RGBA8);

    int sheet_w  = W * MODES;
    char[] sheet = mem::new_array(char, (usz)sheet_w * (usz)H * 3);
    char[] px    = mem::new_array(char, (usz)W * (usz)H * 4);
    for (int m = 0; m < MODES; m++) {
        render_column(m);
        readback::read_image(rt.texture_image, W, H, px.ptr);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                usz s = ((usz)y * W + (usz)x) * 4;
                usz d = ((usz)y * (usz)sheet_w + (usz)m * W + (usz)x) * 3;
                sheet[d] = px[s];
                sheet[d + 1] = px[s + 1];
                sheet[d + 2] = px[s + 2];
            }
        }
    }
    if (!png::write(out_path, sheet, sheet_w, H, p3: p3, channels: 3)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s: dispersion off, on", out_path);
    return 0;
}