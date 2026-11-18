// three's MeshPhysicalMaterial layers on a sphere in front of colored stripes, under a sky map
// (the scene's background and environment). Columns:
//   0  iridescence: a thin film from 100 nm at the bottom to 800 nm at the top (a thickness map)
//   1  a clear coat (roughness 0.03) over rough red metal (roughness 0.6)
//   2  sheen: dark blue velvet with an orange sheen
//   3  double-sided glass: the sphere's far side shows through its near side
//
//   just test physical [out.png] [--p3]   (tests/build/physical.png)
module physical_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int W = 480;
const int H = 270;
const int MODES = 4;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
RenderTarget* rt;
Mesh* ball;
MeshStandardMaterial*[MODES] materials;

// A sky to put behind it all: blue above, warm at the horizon, a checkered band along it and a
// bright sun, in linear light.
fn Texture* sky_map() {
    int w = 512;
    int h = 256;
    float[] px = mem::new_array(float, (usz)(w * h * 4));
    for (int y = 0; y < h; y++) {
        // latitude: +90° at the top row
        float lat = (0.5f - ((float)y + 0.5f) / h) * (float)math::PI;
        for (int x = 0; x < w; x++) {
            float lon = ((float)x + 0.5f) / w * 2 * (float)math::PI;
            float up = math::sin(lat);
            Vector3 c = up > 0 ? (Vector3) { 0.25f, 0.45f, 0.9f } * up + (Vector3) { 1.0f, 0.6f, 0.35f } * (1 - up)
                : (Vector3) { 0.12f, 0.1f, 0.08f };
            if (math::abs(lat) < 0.2f && (((int)(lon * 8 / (float)math::PI) + (lat > 0 ? 1 : 0)) % 2 == 0)) {
                c = { 0.05f, 0.05f, 0.06f };
            }
            float sun = math::abs(lat - 0.5f) + math::abs(lon - 2.0f) * 0.5f;
            if (sun < 0.06f) c = { 20, 18, 14 };
            usz i = (usz)((y * w + x) * 4);
            px[i] = c.x;
            px[i + 1] = c.y;
            px[i + 2] = c.z;
            px[i + 3] = 1;
        }
    }
    return drie::float_texture(px, w, h);
}

// Green from 0 at the bottom of the uv square to 1 at the top.
fn Texture* wedge_map() {
    int w = 4;
    int h = 64;
    char[] px = mem::new_array(char, (usz)(w * h * 4));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            usz i = (usz)((y * w + x) * 4);
            // (data textures go up top row first, without a flip: v = 1 is the last row)
            px[i + 1] = (char)(255 * y / (h - 1));
            px[i + 3] = 255;
        }
    }
    return drie::data_texture(px, w, h, NO_COLOR_SPACE);
}

fn void build_scene() {
    scene = drie::scene();
    Texture* sky = sky_map();
    scene.background_texture = sky;
    scene.environment = sky;
    scene.add(drie::hemisphere_light(0x8090b0, 0x202028, 0.3));
    DirectionalLight* sun = drie::directional_light(0xffffff, 2);
    sun.position.set(1.5f, 2, 1);
    scene.add(sun);
    Color[7] colors = { 0xff2010, 0xffa010, 0xf0e020, 0x20d040, 0x20c0e0, 0x3040ff, 0xffffff };
    BufferGeometry* stripe = drie::box_geometry(0.14f, 1.6f, 0.02f);
    for (int i = 0; i < 17; i++) {
        float x = -1.2f + (float)i * 0.145f;
        Mesh* bar = drie::mesh(stripe, drie::mesh_lambert_material(color: colors[i % 7], emissive: colors[i % 7], emissive_intensity: 0.6));
        bar.position.set(x, 0.1f, -1.2f);
        scene.add(bar);
    }

    MeshStandardMaterial* film = drie::mesh_standard_material(color: 0x202020, roughness: 0.15f);
    film.iridescence = 1;
    film.iridescence_thickness_min = 100;
    film.iridescence_thickness_max = 800;
    film.iridescence_thickness_map = wedge_map();
    materials[0] = film;

    MeshStandardMaterial* coated = drie::mesh_standard_material(color: 0xc01010, roughness: 0.6f, metalness: 1);
    coated.clearcoat = 1;
    coated.clearcoat_roughness = 0.03f;
    materials[1] = coated;

    MeshStandardMaterial* velvet = drie::mesh_standard_material(color: 0x0c1430, roughness: 0.9f);
    velvet.sheen = 1;
    velvet.sheen_color = { 1, 0.45f, 0.15f };
    velvet.sheen_roughness = 0.4f;
    materials[2] = velvet;

    materials[3] = drie::mesh_standard_material(
        color: 0xffffff,
        roughness: 0.05f,
        side: DOUBLE_SIDE,
        transmission: 1,
        thickness: 0.05f,
        ior: 1.5f
    );

    ball = drie::mesh(drie::sphere_geometry(0.45f, 64, 32), materials[0]);
    scene.add(ball);
}

fn void render_column(int mode) {
    ball.material = materials[mode];
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
}

fn int main(String[] args) {
    String out_path = "tests/build/physical.png";
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
    renderer.tone_mapping = ACES_FILMIC_TONE_MAPPING;
    renderer.output_color_space = p3 ? DISPLAY_P3_COLOR_SPACE : SRGB_COLOR_SPACE;
    build_scene();
    camera = drie::perspective_camera(45, (float)W / H, 0.01f, 30);
    camera.position.set(0.3f, 0.25f, 1.7f);
    camera.look_at_vec({ 0, 0, 0 });
    rt = drie::render_target(W, H, 1, RGBA8);

    int sheet_w = W * MODES;
    char[] sheet = mem::new_array(char, (usz)sheet_w * (usz)H * 3);
    char[] px = mem::new_array(char, (usz)W * (usz)H * 4);
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
    io::printfn("saved %s: iridescence, a clear coat, sheen, double-sided glass", out_path);
    return 0;
}
