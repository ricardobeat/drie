// Transmission with the rest of three's MeshPhysicalMaterial around it: a glass sphere in front
// of colored stripes, under a background map (Scene.background_texture, also the scene's
// environment). Columns:
//   0  smooth glass, the background sharp
//   1  rough glass (roughness 0.35): what is behind it blurs (the transmission picture's mip
//      levels), the background blurred as three's backgroundBlurriness 0.35
//   2  smooth glass with a specular color map: rainbow reflections, 3 times as strong
//   3  a thickness map: thick at the top, thin at the bottom, so only the top bends the stripes
//
//   just test transmission [out.png] [--p3]   (tests/build/transmission.png)
module transmission_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int W = 480;
const int H = 270;
const int MODES = 4;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
RenderTarget* rt;
MeshStandardMaterial* glass;
Texture* rainbow;
Texture* wedge;

// A sky to put behind it all: blue above, warm at the horizon, a checkered band along it (to
// see the blur) and a bright sun, in linear light.
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

// Bands of hues, for the specular color map (sRGB).
fn Texture* rainbow_map() {
    int w = 64;
    int h = 64;
    char[] px = mem::new_array(char, (usz)(w * h * 4));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float t = (float)(x + y) / (float)(w + h) * 6;
            usz i = (usz)((y * w + x) * 4);
            px[i] = (char)(127 + 127 * math::cos(t));
            px[i + 1] = (char)(127 + 127 * math::cos(t - 2.1f));
            px[i + 2] = (char)(127 + 127 * math::cos(t + 2.1f));
            px[i + 3] = 255;
        }
    }
    return drie::data_texture(px, w, h);
}

// Green from 1 at the top of the uv square to 0 at the bottom, for the thickness map.
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
        Mesh* bar = drie::mesh(stripe, drie::mesh_lambert_material(color: 0x000000, emissive: colors[i % 7], emissive_intensity: 2));
        bar.position.set(x, 0.1f, -1.2f);
        scene.add(bar);
    }
    glass = drie::mesh_standard_material(color: 0xffffff, roughness: 0.02f, transmission: 1, thickness: 0.6f, ior: 1.5f);
    Mesh* ball = drie::mesh(drie::sphere_geometry(0.45f, 64, 32), glass);
    scene.add(ball);
    rainbow = rainbow_map();
    wedge = wedge_map();
}

fn void render_column(int mode) {
    glass.roughness = mode == 1 ? 0.35f : 0.02f;
    scene.background_blurriness = mode == 1 ? 0.35f : 0;
    glass.specular_color_map = mode == 2 ? rainbow : null;
    glass.specular_color = mode == 2 ? (Vector3) { 3, 3, 3 } : (Vector3) { 1, 1, 1 };
    glass.thickness_map = mode == 3 ? wedge : null;
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
}

fn int main(String[] args) {
    String out_path = "tests/build/transmission.png";
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
    io::printfn("saved %s: smooth, rough (and the background blurred), a specular color map, a thickness map", out_path);
    return 0;
}
