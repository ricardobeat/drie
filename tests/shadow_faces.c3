// The shadow maps a light draws: only those it can use and someone sees.
//   - a rect area light has five maps, none behind it: its front, and its four sides' front
//     halves, half as tall, two to a block of the atlas
//   - a point light whose maps are drawn every frame, behind the camera: after the first
//     frame, the face that looks away from everything the camera sees isn't drawn, and the
//     picture is the one its maps drawn all at once give
//   - a point light over a floor that casts no shadow, with one box beside it: only the face
//     that sees the box is drawn; the others read as lit
//
//   just test shadow_faces [out.png]   (tests/build/shadow_faces.png)
module shadow_faces_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int S = 128;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

Renderer* renderer;
RenderTarget* rt;
char[] px;

fn void frame(Scene* scene, Camera* camera) {
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
}

// The mean brightness of the pixels from (x0, y0) to (x1, y1) of px.
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

// Renders until no draw waits for its program; then once more.
fn void settle(Scene* scene, Camera* camera) {
    frame(scene, camera);
    for (int n = 0; n < 3000 && renderer.info.waiting > 0; n++) frame(scene, camera);
}

fn void add_room(Scene* scene, Material* m) {
    Mesh* floor = drie::mesh(drie::box_geometry(10, 0.2f, 10), m);
    floor.position = { 0, -0.1f, 0 };
    floor.receive_shadow = true;
    floor.cast_shadow = true;
    scene.add(floor);
    for (int i = 0; i < 4; i++) {
        Mesh* b = drie::mesh(drie::box_geometry(0.4f, 1, 0.4f), m);
        b.position = { (float)(i % 2) * 3 - 1.5f, 0.5f, (float)(i / 2) * 3 - 4 };
        b.cast_shadow = true;
        b.receive_shadow = true;
        scene.add(b);
    }
}

fn int main(String[] args) {
    String out_path = "tests/build/shadow_faces.png";
    if (args.len > 1) out_path = args[1];
    renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    rt = drie::render_target(S, S, 1, RGBA8);
    px = mem::new_array(char, (usz)S * S * 4);
    Material* white = drie::mesh_standard_material(color: 0xffffff, roughness: 1);
    PerspectiveCamera* camera = drie::perspective_camera(60, 1, 0.05f, 40);

    // a rect area light
    Scene* rects = drie::scene();
    add_room(rects, white);
    RectAreaLight* door = drie::rect_area_light(0xffffff, 6, 1, 2);
    door.position = { 0, 1, 0 };
    door.look_at_vec({ 0, 1, -1 });
    door.cast_shadow = true;
    door.shadow.map_size = 512;
    rects.add(door);
    camera.position = { 0, 2, 2 };
    camera.look_at_vec({ 0, 0, -3 });
    settle(rects, camera);
    int[4][6] r;
    for (int f = 0; f < 6; f++) r[f] = renderer.shadow_map_rect(door, f);
    bool sides = true;
    for (int f = 1; f < 5; f++) sides = sides && r[f][2] == r[0][2] && r[f][3] == r[0][2] / 2;
    expect(r[0][2] > 0 && r[0][2] == r[0][3] && sides && r[5][2] == 0,
           string::tformat("a rect area light: a %dx%d front map and four %dx%d sides, no sixth", r[0][2], r[0][3], r[1][2], r[1][3]));
    // (the +x and -x sides share a block, as do +y and -y)
    expect(r[1][0] == r[2][0] && math::abs(r[1][1] - r[2][1]) == r[0][2] / 2 && r[3][0] == r[4][0],
           "its sides two to a block");

    // a point light drawn every frame, behind the camera that looks away from it
    Scene* points = drie::scene();
    add_room(points, white);
    PointLight* bulb = drie::point_light(0xffffff, 30, 0, 2);
    bulb.position = { 0, 1.5f, 1 };
    bulb.cast_shadow = true;
    bulb.shadow.map_size = 256;
    points.add(bulb);
    camera.position = { 0, 1.5f, 0 };
    camera.look_at_vec({ 0, 0.5f, -4 });
    settle(points, camera);
    frame(points, camera);
    int first = renderer.info.atlas_tiles;
    frame(points, camera);
    int later = renderer.info.atlas_tiles;
    readback::read_image(rt.texture_image, S, S, px.ptr);
    char[] every = mem::new_array(char, px.len);
    every[..] = px[..];
    // the same light drawn once: all six faces
    bulb.shadow.auto_update = false;
    bulb.shadow.needs_update = true;
    frame(points, camera);
    frame(points, camera);
    readback::read_image(rt.texture_image, S, S, px.ptr);
    double diff;
    for (usz i = 0; i < px.len; i++) diff += math::abs((double)px[i] - (double)every[i]);
    diff /= (double)px.len;
    expect(later < 6 && later > 0, string::tformat("a point light drawn every frame, behind the camera: %d of its 6 maps drawn (%d in the frame before)", later, first));
    expect(diff < 0.5, string::tformat("...and the same picture as with all six drawn (mean difference %.3f)", diff));

    // a point light whose only caster is one box to its +x
    Scene* lone = drie::scene();
    Mesh* floor = drie::mesh(drie::box_geometry(10, 0.2f, 10), white);
    floor.position = { 0, -0.1f, 0 };
    floor.receive_shadow = true;
    lone.add(floor);
    Mesh* box = drie::mesh(drie::box_geometry(0.5f, 0.5f, 0.5f), white);
    box.position = { 1.5f, 0.25f, 0 };
    box.cast_shadow = true;
    lone.add(box);
    PointLight* lamp = drie::point_light(0xffffff, 30, 0, 2);
    lamp.position = { 0, 1, 0 };
    lamp.cast_shadow = true;
    lamp.shadow.map_size = 256;
    lone.add(lamp);
    camera.position = { 0, 4, 4 };
    camera.look_at_vec({ 0, 0, 0 });
    settle(lone, camera);
    frame(lone, camera);
    int drawn = renderer.info.atlas_tiles;
    readback::read_image(rt.texture_image, S, S, px.ptr);
    // the floor behind the box (+x, beyond it) is in shadow; as far on the other side, lit
    float behind = mean(S * 84 / 100, S * 90 / 100, S * 47 / 100, S * 53 / 100);
    float open = mean(S * 10 / 100, S * 16 / 100, S * 47 / 100, S * 53 / 100);
    expect(drawn == 1, string::tformat("one caster beside a point light: %d map drawn", drawn));
    expect(behind < open * 0.5f, string::tformat("...and its shadow is there (%.0f behind the box, %.0f as far on the other side)", behind, open));

    if (!png::write(out_path, px, S, S)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
