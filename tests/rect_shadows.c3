// RectAreaLight.shadow: a lit doorway, a rect area light 1 m wide and 2 m tall standing on a
// glossy floor, seen from 4 m away; its highlight is a bright streak on the floor towards
// the camera. A curtain hangs 0.3 m in front of it, over the whole doorway.
//   - without the curtain: the streak is bright, the floor beside it dark
//   - the curtain with the light's cast_shadow: the floor in front of the curtain, where the
//     streak was, is dark (the curtain hides the doorway from it)
//   - the same with shadow.auto_update off (the maps drawn once) and with VSM
//   - cast_shadow off: the streak is back, through the curtain
//   - a curtain half as wide, over the doorway's left half: the streak's left part is dark,
//     its right part as bright as without the curtain
//   - the floor drawn by a generated program (materials_nodes.c3: it has morph targets): the
//     streak as bright without the curtain, as dark with it
// The picture: the seven views side by side.
//
//   just test rect_shadows [out.png]   (tests/build/rect_shadows.png)
module rect_shadows_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math, std::thread;

const int W = 160;
const int H = 120;
const int VIEWS = 7;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
RenderTarget* rt;
char[] px;
char[] sheet;

// Renders the view into column `slot` of the sheet (again while a draw waits for its program).
fn void view(int slot) {
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
    for (int frames = 0; frames < 2000 && renderer.info.waiting > 0; frames++) {
        thread::sleep_ms(10);
        renderer.render(scene, camera);
        drie::commit();
    }
    readback::read_image(rt.texture_image, W, H, px.ptr);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            usz s = ((usz)y * W + (usz)x) * 4;
            usz d = ((usz)y * (usz)(W * VIEWS) + (usz)(slot * W + x)) * 3;
            sheet[d:3] = px[s:3];
        }
    }
}

// The mean brightness (0..255) of the box of pixels from (x0, y0) to (x1, y1).
fn float mean(int x0, int y0, int x1, int y1) {
    float sum;
    int n;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            usz o = ((usz)y * W + (usz)x) * 4;
            sum += ((float)px[o] + px[o + 1] + px[o + 2]) / 3;
            n++;
        }
    }
    return sum / n;
}

fn int main(String[] args) {
    String out_path = "tests/build/rect_shadows.png";
    if (args.len > 1) out_path = args[1];
    renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    scene = drie::scene();
    scene.background = 0x000000;

    // the floor: dark and glossy
    MeshStandardMaterial* gloss = drie::mesh_standard_material(color: 0x202020, roughness: 0.15f);
    Mesh* floor = drie::mesh(drie::plane_geometry(10, 10), gloss);
    floor.rotation = { -(float)math::PI / 2, 0, 0 };
    floor.receive_shadow = true;
    scene.add(floor);
    // the doorway: shining towards +z from its middle, 1 m up
    RectAreaLight* door = drie::rect_area_light(0xffffff, 6, 1, 2);
    door.position = { 0, 1, 0 };
    door.look_at_vec({ 0, 1, 1 });
    door.shadow.map_size = 512;
    scene.add(door);
    // the curtain: 1.4 m wide, to the floor, 0.3 m in front
    MeshStandardMaterial* cloth = drie::mesh_standard_material(color: 0x401010, roughness: 0.9f);
    Mesh* curtain = drie::mesh(drie::box_geometry(1.4f, 2.2f, 0.04f), cloth);
    curtain.position = { 0, 1.1f, 0.3f };
    curtain.cast_shadow = true;
    scene.add(curtain);
    // the left half of the doorway only
    Mesh* half = drie::mesh(drie::box_geometry(0.7f, 2.2f, 0.04f), cloth);
    half.position = { -0.35f, 1.1f, 0.3f };
    half.cast_shadow = true;
    scene.add(half);

    camera = drie::perspective_camera(50, (float)W / H, 0.05f, 30);
    camera.position = { 0, 1.6f, 4 };
    camera.look_at_vec({ 0, 0.3f, 0.5f });
    rt = drie::render_target(W, H, 1, RGBA8);
    px = mem::new_array(char, (usz)W * H * 4);
    sheet = mem::new_array(char, (usz)W * VIEWS * H * 3);

    // where the streak lies on the floor, between the camera and the doorway (found in the
    // view without the curtain), and the floor to the side of it
    int sx0 = W / 2 - 6;
    int sx1 = W / 2 + 6;
    int sy0 = H * 70 / 100;
    int sy1 = H * 80 / 100;

    curtain.visible = false;
    half.visible = false;
    door.cast_shadow = true;
    view(0);
    float open = mean(sx0, sy0, sx1, sy1);
    float aside = mean(W / 10, sy0, W / 10 + 12, sy1);
    expect(open > 60 && aside < open * 0.25f, string::tformat("no curtain: the streak on the floor is bright (%.0f), the floor beside it dark (%.0f)", open, aside));
    // the left and right parts of the streak
    float open_left = mean(W / 2 - 10, sy0, W / 2 - 4, sy1);
    float open_right = mean(W / 2 + 4, sy0, W / 2 + 10, sy1);

    curtain.visible = true;
    view(1);
    float shut = mean(sx0, sy0, sx1, sy1);
    expect(shut < open * 0.2f, string::tformat("the curtain in front: the floor where the streak was is dark (%.0f, open %.0f)", shut, open));

    door.shadow.auto_update = false;
    door.shadow.needs_update = true;
    view(2);
    float once = mean(sx0, sy0, sx1, sy1);
    expect(once < open * 0.2f, string::tformat("the maps drawn once: dark (%.0f, open %.0f)", once, open));
    door.shadow.auto_update = true;

    renderer.shadow_type = VSM;
    view(3);
    float vsm = mean(sx0, sy0, sx1, sy1);
    expect(vsm < open * 0.2f, string::tformat("VSM: dark (%.0f, open %.0f)", vsm, open));
    renderer.shadow_type = PCF;

    door.cast_shadow = false;
    view(4);
    float through = mean(sx0, sy0, sx1, sy1);
    expect(through > open * 0.9f, string::tformat("cast_shadow off: the streak (%.0f) as bright as without the curtain (%.0f)", through, open));

    door.cast_shadow = true;
    curtain.visible = false;
    half.visible = true;
    view(5);
    float left = mean(W / 2 - 10, sy0, W / 2 - 4, sy1);
    float right = mean(W / 2 + 4, sy0, W / 2 + 10, sy1);
    expect(left < open_left * 0.3f && right > open_right * 0.7f,
           string::tformat("half a curtain: the streak's left part dark (%.0f, open %.0f), its right part lit (%.0f, open %.0f)", left, open_left, right, open_right));

    // one morph target that moves nothing: the floor's draws take a generated program
    floor.visible = false;
    BufferGeometry* flat = drie::plane_geometry(10, 10);
    int verts = (int)flat.vertex_count();
    float[] deltas = mem::new_array(float, (usz)verts * 4);
    flat.set_morph_positions(deltas, 1, verts);
    Mesh* morphed = drie::mesh(flat, gloss);
    morphed.rotation = floor.rotation;
    morphed.receive_shadow = true;
    morphed.set_morph_weights({ 0 }, 1, 0, verts);
    scene.add(morphed);
    half.visible = false;
    view(6);
    float generated_open = mean(sx0, sy0, sx1, sy1);
    curtain.visible = true;
    view(6);
    float generated = mean(sx0, sy0, sx1, sy1);
    expect(renderer.info.waiting == 0 && generated_open > open * 0.8f && generated < generated_open * 0.2f,
           string::tformat("a generated program's floor: the streak bright (%.0f), dark with the curtain (%.0f; %d draws waiting)", generated_open, generated, renderer.info.waiting));

    if (!png::write(out_path, sheet, W * VIEWS, H, channels: 3)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
