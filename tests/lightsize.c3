// Lights with a size: a bulb over a dark glossy floor, between a mirror-like metal ball (left)
// and a rough one (right). A small white ball (a capsule for a tube) marks the bulb's real size.
//   top row:    point lights of radius 0, 0.05 and 0.3 m (PointLight.radius): on the floor and
//               the metal ball the highlight should be a sharp-edged image of the bulb, as big
//               as the bulb, and the rough ball should look the same in all three
//   bottom row: tubes of radius 0.02 m, 0 m long (the same as a ball), 0.5 m and 1.5 m long
//               (PointLight.length), lying along the light's x axis
// Prints how bright each picture is on average, and on the rough ball: a source's size spreads
// its light, it should not add or take much of it.
//
//   just test lightsize [out.png]   (tests/build/lightsize.png)
module lightsize_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int W = 400;
const int H = 260;
const int COLS = 3;
const int ROWS = 2;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
RenderTarget* rt;
PointLight* bulb;
Mesh* marker;
Mesh* tube_marker;

fn void build_scene() {
    scene = drie::scene();
    scene.background = 0x000000;
    scene.add(drie::hemisphere_light(0x404858, 0x101010, 0.6f));

    Mesh* floor = drie::mesh(drie::plane_geometry(8, 8), drie::mesh_standard_material(color: 0x181818, roughness: 0.04f));
    floor.rotation.x = -(float)math::PI / 2;
    scene.add(floor);

    Mesh* mirror = drie::mesh(drie::sphere_geometry(0.32f, 96, 48), drie::mesh_standard_material(color: 0xd8d8d8, roughness: 0.02f, metalness: 1));
    mirror.position.set(-0.75f, 0.32f, 0);
    scene.add(mirror);

    Mesh* rough = drie::mesh(drie::sphere_geometry(0.32f, 96, 48), drie::mesh_standard_material(color: 0xb0b0b0, roughness: 0.7f));
    rough.position.set(0.75f, 0.32f, 0);
    scene.add(rough);

    bulb = drie::point_light(0xfff2e0, 10, 0, 2);
    bulb.position.set(0, 0.6f, -1.0f);
    scene.add(bulb);

    marker = drie::mesh(drie::sphere_geometry(1, 32, 16), drie::mesh_basic_material(color: 0xffffff));
    marker.position = bulb.position;
    scene.add(marker);
    // a capsule's middle, lying along x (the cylinder stands along y)
    tube_marker = drie::mesh(drie::cylinder_geometry(1, 1, 1, 24), drie::mesh_basic_material(color: 0xffffff));
    tube_marker.position = bulb.position;
    tube_marker.rotation.z = (float)math::PI / 2;
    scene.add(tube_marker);
}

// The bulb as `radius` and `len`, and its marker.
fn void set_bulb(float radius, float len) {
    bulb.radius = radius;
    bulb.length = len;
    float r = math::max(radius, 0.008f);
    marker.scale = { r, r, r };
    tube_marker.visible = len > 0;
    tube_marker.scale = { r, len, r };
}

fn int main(String[] args) {
    String out_path = "tests/build/lightsize.png";
    if (args.len > 1) out_path = args[1];
    renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.tone_mapping = ACES_FILMIC_TONE_MAPPING;
    build_scene();
    camera = drie::perspective_camera(40, (float)W / H, 0.05f, 30);
    camera.position.set(0, 1.0f, 3.2f);
    camera.look_at_vec({ 0, 0.15f, -0.3f });
    rt = drie::render_target(W, H, 1, RGBA8);

    float[2][COLS * ROWS] settings = { { 0, 0 }, { 0.05f, 0 }, { 0.3f, 0 }, { 0.02f, 0 }, { 0.02f, 0.5f }, { 0.02f, 1.5f } };
    int sheet_w = W * COLS;
    int sheet_h = H * ROWS;
    char[] sheet = mem::new_array(char, (usz)sheet_w * (usz)sheet_h * 3);
    char[] px = mem::new_array(char, (usz)W * (usz)H * 4);
    for (int k = 0; k < COLS * ROWS; k++) {
        set_bulb(settings[k][0], settings[k][1]);
        renderer.set_render_target(rt);
        renderer.render(scene, camera);
        drie::commit();
        readback::read_image(rt.texture_image, W, H, px.ptr);
        int cx = (k % COLS) * W;
        int cy = (k / COLS) * H;
        double all;
        double ball;
        int ball_n;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                usz s = ((usz)y * W + (usz)x) * 4;
                usz d = ((usz)(cy + y) * (usz)sheet_w + (usz)(cx + x)) * 3;
                sheet[d] = px[s];
                sheet[d + 1] = px[s + 1];
                sheet[d + 2] = px[s + 2];
                double v = ((double)px[s] + px[s + 1] + px[s + 2]) / 3;
                all += v;
                // the rough ball's middle (it sits right of the centre)
                if (x > W * 63 / 100 && x < W * 77 / 100 && y > H * 30 / 100 && y < H * 46 / 100) {
                    ball += v;
                    ball_n++;
                }
            }
        }
        io::printfn("radius %.2f length %.2f: mean %.2f, rough ball %.2f", settings[k][0], settings[k][1], all / (W * H), ball / math::max(ball_n, 1));
    }
    if (!png::write(out_path, sheet, sheet_w, sheet_h, channels: 3)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s", out_path);
    return 0;
}
