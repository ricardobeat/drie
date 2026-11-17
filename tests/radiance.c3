// Radiance cascades (radiance.c3) in the reflection test's scene: a floor, a wall, a thin
// panel, a lamp and glowing cubes. Columns: the scene's hemisphere light as it is; the
// cascades in its place (sky, lamp and cubes bouncing); the cascades with the lamp and the sky
// off, so that only the cubes' glow lights the room.
//
//   tests/run.sh radiance <libdir> [out.ppm]
module radiance_test;
import drie, drie::postprocessing, sokol, readback;
import std::io, std::math, std::io::file;

const int W = 480;
const int H = 270;
// frames rendered from each viewpoint before the picture is taken (TAA and the reflections
// of the frame before settle)
const int SETTLE = 64;

struct View
{
    String name;
    Vector3 eye;
    Vector3 at;
}

View[4] views = {
    { "low, along the cubes", { 0, 0.55f, 3.2f }, { 0, 0.5f, 0 } },
    { "grazing", { -1.4f, 0.25f, 3.0f }, { 0.2f, 0.4f, 0 } },
    { "from above, steep", { 1.2f, 1.9f, 2.2f }, { 0, 0.2f, 0 } },
    { "past the thin panel", { 2.2f, 0.6f, 2.0f }, { -0.5f, 0.4f, -0.4f } },
};

const int OFF = 0;
const int CASCADES = 1;
const int CUBES_ONLY = 2;
const int MODES = 3;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
RenderTarget* scene_rt;
RenderTarget* toned;
GTAOPass* ao;
TAAPass* taa;
OutputPass* output;
MeshStandardMaterial* wood;
RadianceCascades* rc;
PointLight* lamp;
HemisphereLight* hemi;

fn void build_scene()
{
    scene = drie::scene();
    scene.background = 0x10131a;
    hemi = drie::hemisphere_light(0x8090b0, 0x302820, 0.6);
    scene.add(hemi);
    lamp = drie::point_light(0xffe0b0, 6, 0, 2);
    lamp.position.set(-1.0f, 2.2f, 1.0f);
    scene.add(lamp);

    // the glossy floor, 6 x 6 m, one-sided: no back face anywhere near it
    wood = drie::mesh_standard_material(color: 0x6a3a1c, roughness: 0.18);
    Mesh* floor = drie::mesh(drie::plane_geometry(6, 6), wood);
    floor.rotation.x = -drie::PI / 2;
    scene.add(floor);

    // a thick wall behind (a box), and a thin one-sided panel standing on the floor
    MeshStandardMaterial* plaster = drie::mesh_standard_material(color: 0xb0a890, roughness: 0.9);
    Mesh* wall = drie::mesh(drie::box_geometry(6, 2.5f, 0.2f), plaster);
    wall.position.set(0, 1.25f, -1.6f);
    scene.add(wall);
    Mesh* panel = drie::mesh(drie::plane_geometry(0.8f, 1.2f), plaster);
    panel.position.set(-1.3f, 0.6f, -0.6f);
    panel.rotation.y = 0.5f;
    scene.add(panel);

    // emissive cubes floating at two heights (as the house's color test), a white one far over
    // white, and a dull cube standing on the floor
    Color[6] colors = { 0xff3010, 0xff8a10, 0xf0e030, 0x40d040, 0x30d0c0, 0x2030ff };
    BufferGeometry* cube = drie::box_geometry(0.16f, 0.16f, 0.16f);
    foreach (i, c : colors)
    {
        float x = -0.6f + (float)i * 0.24f;
        Mesh* top = drie::mesh(cube, drie::mesh_lambert_material(color: 0x000000, emissive: c, emissive_intensity: 2));
        top.position.set(x, 0.75f, 0);
        scene.add(top);
        Mesh* low = drie::mesh(cube, drie::mesh_lambert_material(color: 0x000000, emissive: c, emissive_intensity: 0.5f));
        low.position.set(x, 0.45f, 0);
        scene.add(low);
    }
    Mesh* white = drie::mesh(cube, drie::mesh_lambert_material(color: 0x000000, emissive: 0xffffff, emissive_intensity: 10));
    white.position.set(0.1f, 1.1f, -0.2f);
    scene.add(white);
    Mesh* block = drie::mesh(drie::box_geometry(0.4f, 0.4f, 0.4f), drie::mesh_standard_material(color: 0x707a88, roughness: 0.6));
    block.position.set(1.1f, 0.2f, -0.5f);
    scene.add(block);
}

fn void frame(View* v, int mode, bool reset)
{
    camera.position = v.eye;
    camera.look_at_vec(v.at);
    if (reset)
    {
        taa.reset = true;
        rc.reset();
    }
    taa.jitter(camera);
    lamp.intensity = mode == CUBES_ONLY ? 0 : 6;
    hemi.intensity = mode == CUBES_ONLY ? 0 : 0.6f;
    scene.radiance = mode == OFF ? null : rc;
    if (mode != OFF) rc.update(scene);
    renderer.set_render_target(scene_rt);
    renderer.render(scene, camera);
    ao.render(scene_rt);
    taa.blend = 0.1f;
    taa.render(scene_rt);
    output.set_source(taa.texture());
    output.render(toned);
    drie::commit();
}

fn int main(String[] args)
{
    String out_path = args.len > 1 ? args[1] : "radiance.ppm";
    renderer = drie::webgl_renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.tone_mapping = ACES_FILMIC_TONE_MAPPING;
    build_scene();
    scene.update_matrix_world();
    rc = drie::radiance_cascades({ -3, -0.25f, -2 }, { 3, 2.75f, 3 });
    rc.voxelize(scene);
    camera = drie::perspective_camera(55, (float)W / H, 0.05f, 40);
    scene_rt = drie::render_target(W, H, 1, RGBA16F, mrt: true);
    toned = drie::render_target(W, H, 1, RGBA8, false);
    ao = postprocessing::gtao_pass(camera);
    taa = postprocessing::taa_pass();
    output = postprocessing::output_pass(renderer);

    int cols = MODES;
    int sheet_w = W * cols;
    int sheet_h = H * (int)views.len;
    char[] sheet = mem::new_array(char, (usz)sheet_w * (usz)sheet_h * 3);
    char[] px = mem::new_array(char, (usz)W * (usz)H * 4);
    foreach (vi, &v : views)
    {
        for (int m = 0; m < cols; m++)
        {
            for (int f = 0; f < SETTLE; f++) frame(v, m, f == 0);
            readback::read_image(toned.texture_image, W, H, px.ptr);
            for (int y = 0; y < H; y++)
            {
                for (int x = 0; x < W; x++)
                {
                    usz s = ((usz)y * W + (usz)x) * 4;
                    usz d = (((usz)vi * H + (usz)y) * (usz)sheet_w + (usz)m * W + (usz)x) * 3;
                    sheet[d] = px[s];
                    sheet[d + 1] = px[s + 1];
                    sheet[d + 2] = px[s + 2];
                }
            }
        }
        io::printfn("%d: %s", vi, v.name);
    }
    File? f = file::open(out_path, "wb");
    if (catch err = f)
    {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    defer (void)f.close();
    (void)f.write(string::tformat("P6\n%d %d\n255\n", sheet_w, sheet_h));
    (void)f.write(sheet);
    io::printfn("saved %s: columns hemisphere light, cascades, cascades lit by the cubes only", out_path);
    return 0;
}
