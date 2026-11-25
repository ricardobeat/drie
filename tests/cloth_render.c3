// drie::cloth drawn: two pleated panels of a curtain hang from a rod in a slight draft, and a
// person (a capsule, drawn as a cylinder) walks through the gap between them, from the camera
// toward the wall. The cloth has a node material (stripes of the uv) and casts shadows from a
// spot light onto the floor and the wall. A still camera, a frame every 0.1 s:
// tests/build/cloth_frame_00.png .. cloth_frame_29.png and the sheet tests/build/cloth.png (every
// third frame, half size).
//   - the draft moves the cloth while nothing touches it, and the picture follows
//   - the walk parts the curtain, and the picture shows it
//   - the cloth is drawn with its node material (the stripes make its columns differ)
//   - it casts a shadow (the picture without it is lighter in places)
//
//   just test cloth_render [out.png]
module cloth_render_test;
import drie, drie::cloth, drie::nodes, sokol, drie::readback, drie::png;
import std::io, std::math;

const int W      = 480;
const int H      = 360;
const int FRAMES = 30;
const int COLS   = 18;
const int ROWS   = 36;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

// A panel `width` across at the rod from x0, hanging `drop` from y top at z = 0, in `pleats`
// soft folds `depth` deep.
fn Cloth* panel(float x0, float width, float top, float drop, int pleats, float depth) {
    Vector3[] rest = mem::new_array(Vector3, (sz)COLS * ROWS);
    defer free(rest);
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            float u = (float)c / (COLS - 1);
            float v = (float)r / (ROWS - 1);
            rest[r * COLS + c] = {
                x0 + u * width,
                top - v * drop,
                depth * math::sin(u * pleats * 2 * (float)math::PI)
            };
        }
    }
    Cloth* c = cloth::from_positions(COLS, ROWS, rest, mass: 0.5);
    c.pin_row(0);
    c.wind = { 0.05, 0, -0.25 };
    c.gust = 0.7;
    c.floor_y = 0;
    return c;
}

fn float lum(char[] px, int x, int y) {
    usz i = ((usz)y * W + (usz)x) * 4;
    return ((float)px[i] + (float)px[i + 1] + (float)px[i + 2]) / 3;
}

// how much a box of the picture changed between two frames (mean absolute difference)
fn float changed(char[] a, char[] b, int x0, int y0, int x1, int y1) {
    float sum;
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) sum += math::abs(lum(a, x, y) - lum(b, x, y));
    }
    return sum / (float)((x1 - x0) * (y1 - y0));
}

fn int main(String[] args) {
    String out_path    = args.len > 1 ? args[1] : "tests/build/cloth.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    Scene* scene = drie::scene();
    scene.background = 0x101014;
    scene.add(drie::hemisphere_light(0xffffff, 0x202020, 0.25));
    SpotLight* spot = drie::spot_light(0xfff0e0, 40, 0, 0.7f, 0.4f);
    spot.position = { 1.4, 3.2, 2.2 };
    spot.target.position = { 0, 0.8, -0.4 };
    spot.cast_shadow = true;
    scene.add(spot);
    scene.add(spot.target);

    MeshStandardMaterial* plaster = drie::mesh_standard_material(color: 0x9a948a);
    Mesh* floor                   = drie::mesh(drie::plane_geometry(8, 8), plaster);
    floor.rotation.x = -(float)math::PI / 2;
    floor.receive_shadow = true;
    scene.add(floor);
    Mesh* wall = drie::mesh(drie::plane_geometry(8, 4), plaster);
    wall.position = { 0, 2, -1.3 };
    wall.receive_shadow = true;
    scene.add(wall);
    Mesh* rod = drie::mesh(
        drie::cylinder_geometry(0.015, 0.015, 1.5, 12),
        drie::mesh_standard_material(color: 0x302010, roughness: 0.4f, metalness: 1)
    );
    rod.position = { 0, 2.42, 0 };
    rod.rotation.z = (float)math::PI / 2;
    scene.add(rod);

    // the cloth: dark red velvet with paler stripes down its length
    MeshStandardMaterial* velvet = drie::mesh_standard_material(color: 0x8a1018, roughness: 0.9f, side: DOUBLE_SIDE);
    Node stripe                  = nodes::smoothstep(0.6f, 0.9f, nodes::sin(nodes::uv().x() * 55.0f));
    velvet.color_node = nodes::mix(nodes::color(0x6a0c12), nodes::color(0xc04040), stripe);
    Cloth*[2] panels = { panel(-0.7, 0.68, 2.4, 2.38, 4, 0.04), panel(0.02, 0.68, 2.4, 2.38, 4, 0.04) };
    foreach (c : panels) {
        Mesh* m = c.make_mesh(velvet);
        m.cast_shadow = true;
        m.receive_shadow = true;
        scene.add(m);
    }
    // the person
    float radius = 0.2;
    Mesh* body   = drie::mesh(
        drie::cylinder_geometry(radius, radius, 1.7, 16),
        drie::mesh_standard_material(color: 0x404858)
    );
    body.cast_shadow = true;
    scene.add(body);
    int[2] ids;
    foreach (i, c : panels) ids[i] = c.add_capsule({ 0, 0.3, 1.5 }, { 0, 1.6, 1.5 }, radius);

    PerspectiveCamera* camera = drie::perspective_camera(50, (float)W / H, 0.1f, 30);
    camera.position = { 2.3, 1.5, 2.9 };
    camera.look_at(0, 1.1, -0.1);
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);

    // hang still first
    for (int i = 0; i < 180; i++) {
        foreach (c : panels) c.update(1.0f / 60);
    }
    char[][FRAMES] px;
    float parted;
    float swayed;
    Vector3[] before = mem::new_array(Vector3, panels[0].pos.len);
    foreach (i, &p : before) *p = panels[0].point((uint)i);
    for (int f = 0; f < FRAMES; f++) {
        // six simulation frames per picture; the walk starts at picture 5, at 1.1 m/s
        for (int k = 0; k < 6; k++) {
            float t = ((float)(f * 6 + k) - 30) / 60;
            float z = t < 0 ? 1.5f : 1.5f - t * 1.1f;
            body.position = { 0, 0.85, z };
            foreach (i, c : panels) {
                c.move_collider(ids[i], { 0, 0.3, z }, { 0, 1.6, z });
                c.update(1.0f / 60);
            }
        }
        if (f == 4) {
            foreach (i, p : before) swayed = math::max(swayed, (panels[0].point((uint)i) - p).length());
        }
        parted = math::max(
            parted,
            panels[1].point(panels[1].index(0, 18)).x - panels[0].point(panels[0].index(COLS - 1, 18)).x
        );
        renderer.render(scene, camera);
        drie::commit();
        px[f] = mem::new_array(char, (sz)W * H * 4);
        readback::read_image(rt.texture_image, W, H, px[f].ptr);
        (void)png::write(string::tformat("tests/build/cloth_frame_%02d.png", f), px[f], W, H);
    }

    // the curtain on screen: around the middle of the picture
    float moved = changed(px[0], px[4], W / 2 - 100, H / 2 - 100, W / 2 + 60, H / 2 + 60);
    expect(
        swayed > 0.005f && moved > 0.02f,
        string::tformat("the draft moves the cloth before the walk (%.3f m; %.3f of 255 per pixel)", swayed, moved)
    );
    float opened = changed(px[0], px[FRAMES - 6], W / 2 - 100, H / 2 - 100, W / 2 + 60, H / 2 + 60);
    expect(
        parted > 2 * radius && opened > 5,
        string::tformat("the walk parts the curtain (gap %.2f m; %.1f of 255 per pixel)", parted, opened)
    );
    // stripes: along a row across the left panel, the red channel goes up and down
    int turns;
    int y        = H / 2 - 30;
    float prev_r = px[0][((usz)y * W + 140) * 4];
    int dir;
    for (int x = 141; x < W / 2; x++) {
        float rr = px[0][((usz)y * W + (usz)x) * 4];
        int d    = rr > prev_r + 6 ? 1 : rr < prev_r - 6 ? -1 : 0;
        if (d != 0) {
            if (dir != 0 && d != dir) turns++;
            dir = d;
            prev_r = rr;
        }
    }
    expect(turns >= 4, string::tformat("the node material's stripes are on the cloth (%d turns along a row)", turns));

    // without the cloth's shadow
    foreach (c : panels) c.mesh.cast_shadow = false;
    renderer.render(scene, camera);
    drie::commit();
    char[] lit = mem::new_array(char, (sz)W * H * 4);
    readback::read_image(rt.texture_image, W, H, lit.ptr);
    int shaded;
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            if (lum(lit, xx, yy) > lum(px[FRAMES - 1], xx, yy) + 20) shaded++;
        }
    }
    expect(shaded > 1000, string::tformat("the cloth casts a shadow (%d pixels lighter without it)", shaded));

    // half-size sheet of every third frame, five across
    int sw       = W / 2;
    int sh       = H / 2;
    char[] sheet = mem::new_array(char, (sz)sw * 5 * sh * 2 * 4);
    for (int k = 0; k < 10; k++) {
        char[] src = px[k * 3];
        int ox     = (k % 5) * sw;
        int oy     = (k / 5) * sh;
        for (int yy = 0; yy < sh; yy++) {
            for (int xx = 0; xx < sw; xx++) {
                usz d = (usz)(((oy + yy) * sw * 5 + ox + xx) * 4);
                for (int ch = 0; ch < 3; ch++) {
                    uint sum;
                    for (int j = 0; j < 4; j++) {
                        sum += src[(usz)(((yy * 2 + j / 2) * W + xx * 2 + j % 2) * 4 + ch)];
                    }
                    sheet[d + (usz)ch] = (char)(sum / 4);
                }
                sheet[d + 3] = 255;
            }
        }
    }
    (void)png::write(out_path, sheet, sw * 5, sh * 2);
    io::printfn("%s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}