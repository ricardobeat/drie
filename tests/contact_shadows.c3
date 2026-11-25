// Contact shadows (LightShadow.contact_length), seen from above, a sun low in the -x sky
// without a shadow map:
//   - a small cube on a floor: dark at its foot on the side away from the light with them
//     on, lit there with them off
//   - a thin pole standing a little way off the floor, the light grazing: no shadow smeared
//     across the floor far behind it
//
//   just test contact_shadows [out.png]   (tests/build/contact_shadows.png)
module contact_shadows_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int S = 200;

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

fn float mean(int x0, int x1, int y0, int y1) {
    float sum;
    int n;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            usz o = ((usz)y * S + (usz)x) * 4;
            sum += ((float)px[o] + px[o + 1] + px[o + 2]) / 3;
            n++;
        }
    }
    return sum / n;
}

fn void picture() {
    for (int n = 0; n < 3000; n++) {
        renderer.set_render_target(rt);
        renderer.render(scene, camera);
        drie::commit();
        if (n > 4 && renderer.info.waiting == 0) break;
    }
    readback::read_image(rt.texture_image, S, S, px.ptr);
}

fn int main(String[] args) {
    String out_path = "tests/build/contact_shadows.png";
    if (args.len > 1) out_path = args[1];
    renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    rt = drie::render_target(S, S, 1, RGBA8);
    px = mem::new_array(char, (usz)S * S * 4);
    scene = drie::scene();
    Material* white = drie::mesh_standard_material(color: 0xffffff, roughness: 1);
    Mesh* floor = drie::mesh(drie::box_geometry(10, 0.2f, 10), white);
    floor.position = { 0, -0.1f, 0 };
    floor.receive_shadow = true;
    scene.add(floor);
    // a 10 cm cube at x = -1
    Mesh* cube = drie::mesh(drie::box_geometry(0.1f, 0.1f, 0.1f), white);
    cube.position = { -1, 0.05f, 0 };
    cube.receive_shadow = true;
    scene.add(cube);
    // a 3 cm pole from 15 cm over the floor, at x = 1
    Mesh* pole = drie::mesh(drie::box_geometry(0.03f, 1, 0.03f), white);
    pole.position = { 1, 0.65f, 0 };
    scene.add(pole);
    DirectionalLight* sun = drie::directional_light(0xffffff, 2);
    sun.position = { -3, 1, 0 };
    sun.shadow.contact_length = 0.3f;
    scene.add(sun);
    OrthographicCamera* cam = drie::orthographic_camera(-2.5f, 2.5f, 2.5f, -2.5f, 0.1f, 20);
    cam.position = { 0, 8, 0 };
    cam.up = { 0, 0, -1 };
    cam.look_at_vec({ 0, 0, 0 });
    camera = cam;

    // (1 px = 2.5 cm, x = 0 at px 100: the cube's +x side at px 62, the pole at px 140)
    picture();
    float foot_on = mean(63, 66, 98, 102);
    float open_on = mean(40, 60, 40, 60);
    float behind_pole = mean(141, 150, 98, 102);
    if (!png::write(out_path, px, S, S)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    sun.shadow.contact_length = 0;
    picture();
    float foot_off = mean(63, 66, 98, 102);
    expect(foot_on < open_on * 0.5f && foot_off > open_on * 0.9f,
           string::tformat("the cube's foot: %.0f with contact shadows, %.0f without (open floor %.0f)", foot_on, foot_off, open_on));
    expect(behind_pole > open_on * 0.9f, string::tformat("behind the thin pole: %.0f (open floor %.0f)", behind_pole, open_on));

    io::printfn("saved %s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
