// An application's material shader (Renderer.shaders), as house hands the renderer its own
// build of shader_core.glsl: here drie's own build (drie::shaders), whose hooks do nothing.
// The scene is drawn first with the generated programs, then with Renderer.shaders set: an
// opaque box casting a shadow onto the floor, a transparent sphere and a node material (its
// graph spliced into the shader's hooks). The two pictures are close (the material shader is
// not the generated programs to the pixel), and the shadow shows in both.
//
//   just test classic_shaders [out.png]   (tests/build/classic_shaders.png: the second picture)
module classic_shaders_test;
import drie, drie::nodes, drie::shaders, sokol, drie::readback, drie::png;
import std::io, std::math;

const int W = 256;
const int H = 256;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

// The average brightness of the 7x7 pixels around (x, y).
fn float lum(char[] px, int x, int y) {
    float sum;
    for (int dy = -3; dy <= 3; dy++) {
        for (int dx = -3; dx <= 3; dx++) {
            usz i = ((usz)(y + dy) * W + (usz)(x + dx)) * 4;
            sum += ((float)px[i] + (float)px[i + 1] + (float)px[i + 2]) / 3;
        }
    }
    return sum / 49;
}

fn int main(String[] args) {
    String out_path = args.len > 1 ? args[1] : "tests/build/classic_shaders.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    Scene* scene = drie::scene();
    scene.add(drie::hemisphere_light(0xffffff, 0x404040, 0.5));
    DirectionalLight* sun = drie::directional_light(0xffffff, 2);
    sun.position = { 3, 6, 2 };
    sun.cast_shadow = true;
    sun.shadow_extent = 4;
    scene.add(sun);
    Mesh* floor = drie::mesh(drie::plane_geometry(10, 10), drie::mesh_standard_material(color: 0x909090));
    floor.rotation.x = -(float)math::PI / 2;
    floor.receive_shadow = true;
    scene.add(floor);
    Mesh* box = drie::mesh(drie::box_geometry(1, 1, 1), drie::mesh_standard_material(color: 0xc06030, roughness: 0.5));
    box.position = { -0.8f, 0.5f, 0 };
    box.cast_shadow = true;
    scene.add(box);
    Material* glass = drie::mesh_standard_material(color: 0x3080ff);
    glass.transparent = true;
    glass.opacity = 0.5f;
    Mesh* ball = drie::mesh(drie::sphere_geometry(0.5f, 32, 16), glass);
    ball.position = { 1, 0.5f, 0.8f };
    scene.add(ball);
    Material* green = drie::mesh_standard_material();
    green.color_node = nodes::vec3(0.2f, 0.8f, 0.3f);
    Mesh* cone = drie::mesh(drie::box_geometry(0.6f, 0.6f, 0.6f), green);
    cone.position = { 1, 0.3f, -1 };
    cone.cast_shadow = true;
    scene.add(cone);

    PerspectiveCamera* camera = drie::perspective_camera(45, (float)W / H, 0.1f, 50);
    camera.position = { 0, 5, 5 };
    camera.look_at(0, 0, 0);
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    char[] generated = mem::new_array(char, (usz)W * (usz)H * 4);
    char[] classic = mem::new_array(char, (usz)W * (usz)H * 4);

    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, W, H, generated.ptr);

    renderer.shaders = {
        .main = &shaders::three_shader_desc,
        .opaque = &shaders::three_opaque_shader_desc,
        .depth = &shaders::depth_shader_desc,
        .shadow = &shaders::shadow_shader_desc,
        .main_mrt = &shaders::three_mrt_shader_desc,
        .opaque_mrt = &shaders::three_opaque_mrt_shader_desc,
        .main_vsm = &shaders::three_vsm_shader_desc,
        .main_mrt_vsm = &shaders::three_mrt_vsm_shader_desc,
    };
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, W, H, classic.ptr);
    png::write(out_path, classic, W, H);

    // how far apart the two pictures are
    double total;
    int far;
    for (usz i = 0; i < generated.len; i += 4) {
        int d;
        for (usz c = 0; c < 3; c++) d = math::max(d, math::abs((int)generated[i + c] - (int)classic[i + c]));
        total += d;
        if (d > 24) far++;
    }
    double mean = total / (W * H);
    io::printfn("mean difference %.2f, %d pixels more than 24 apart", mean, far);
    expect(mean < 3, "the material shader's picture is close to the generated programs'");
    expect(far < W * H / 50, "few pixels far apart");

    // the box's shadow falls away from the sun (-x, -z of it); the floor in the open
    int[2] shade = { 92, 128 };
    int[2] open = { 128, 200 };
    foreach (j, px : (char[][2]){ generated, classic }) {
        float s = lum(px, shade[0], shade[1]);
        float o = lum(px, open[0], open[1]);
        io::printfn("%s: shadow %.1f, open floor %.1f", j == 0 ? "generated" : "classic", s, o);
        expect(s < o - 15, j == 0 ? "the shadow, generated programs" : "the shadow, material shader");
    }
    // the node material's green
    usz i = ((usz)100 * W + 160) * 4;
    io::printfn("node material %d %d %d", classic[i], classic[i + 1], classic[i + 2]);
    return failures ? 1 : 0;
}
