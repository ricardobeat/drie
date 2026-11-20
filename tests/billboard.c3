// Mesh.billboard: a quad in the xy plane, seen edge-on from +x, drawn turned toward the camera.
//   - plain: edge-on, nothing in the middle of the picture
//   - FACING: the quad faces the view, a square in the middle; its center (0.5, 0) puts the
//     quad above the middle, its bottom edge on it
//   - UPRIGHT: turned about the world's up toward the camera, seen from 45 degrees above: the
//     quad stands upright, shorter on screen than it is wide, its top (nearer the camera)
//     further from the middle than its bottom
// The picture: the four, left to right.
//
//   just test billboard [out.png]   (tests/build/billboard.png)
module billboard_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int S = 64;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn bool lit(char[] px, int x, int y) => px[((usz)y * S + (usz)x) * 4] > 128;

// how many rows of column x are lit
fn int height(char[] px, int x) {
    int n;
    for (int y = 0; y < S; y++) {
        if (lit(px, x, y)) n++;
    }
    return n;
}

fn int width(char[] px, int y) {
    int n;
    for (int x = 0; x < S; x++) {
        if (lit(px, x, y)) n++;
    }
    return n;
}

fn int main(String[] args) {
    String out_path = "tests/build/billboard.png";
    if (args.len > 1) out_path = args[1];
    Renderer* renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    renderer.tone_mapping = NO_TONE_MAPPING;
    RenderTarget* rt = drie::render_target(S, S, 1, RGBA8);
    renderer.set_render_target(rt);
    PerspectiveCamera* camera = drie::perspective_camera(60, 1, 0.1, 20);
    Scene* scene = drie::scene();
    scene.background = 0x000000;
    Mesh* quad = drie::mesh(drie::plane_geometry(1, 1), drie::mesh_basic_material(color: 0xffffff, side: DOUBLE_SIDE));
    scene.add(quad);
    char[][4] px;
    for (int i = 0; i < 4; i++) {
        quad.billboard = i == 0 ? NO_BILLBOARD : i == 3 ? UPRIGHT : FACING;
        quad.billboard_center = { 0.5, i == 2 ? 0 : 0.5 };
        if (i == 3) {
            camera.position = { 2, 2, 0 };
        } else {
            camera.position = { 3, 0, 0 };
        }
        camera.look_at(0, 0, 0);
        px[i] = mem::new_array(char, S * S * 4);
        renderer.render(scene, camera);
        drie::commit();
        readback::read_image(rt.texture_image, S, S, px[i].ptr);
    }
    expect(!lit(px[0], S / 2 - 4, S / 2 - 4), "edge-on without a billboard: nothing beside the middle");
    int w = width(px[1], S / 2);
    int h = height(px[1], S / 2);
    expect(w > 10 && math::abs(w - h) <= 1, string::tformat("FACING: a square in the middle (%d x %d)", w, h));
    expect(lit(px[2], S / 2, S / 2 - 6) && !lit(px[2], S / 2, S / 2 + 6),
           "FACING, center (0.5, 0): above the middle, not below");
    // (from 45 degrees above, an upright quad's top is nearer the camera: further from the middle)
    int above;
    int below;
    for (int y = 0; y < S; y++) {
        if (lit(px[3], S / 2, y)) {
            if (y < S / 2) {
                above++;
            } else {
                below++;
            }
        }
    }
    int hu = height(px[3], S / 2);
    int wu = width(px[3], S / 2);
    expect(wu > 10 && hu > 5 && hu < wu && above > below,
           string::tformat("UPRIGHT: standing, seen from above (%d wide, %d tall, %d above the middle, %d below)", wu, hu,
                           above, below));

    char[] sheet = mem::new_array(char, S * 4 * S * 4);
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            usz src = ((usz)y * S + (usz)x) * 4;
            for (int k = 0; k < 4; k++) {
                usz dst = ((usz)y * S * 4 + (usz)(k * S + x)) * 4;
                sheet[dst:3] = px[k][src:3];
                sheet[dst + 3] = 255;
            }
        }
    }
    (void)png::write(out_path, sheet, S * 4, S);
    io::printfn("%s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
