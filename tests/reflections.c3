// Screen-space reflections, in a scene made for them: a glossy floor, cubes floating over it
// and standing on it, a thin one-sided panel, a thick wall. Renders it from a few low angles
// the way a game would (linear MRT target, GTAO for the distances, TAA for the last frame,
// OutputPass), with reflections off, on with the old thickness guess, on with
// render_back_faces, and with a Reflector for the floor (screen-space reflections only for what
// stands on it), side by side into one picture.
//
//   just test reflections [out.png] [--p3]  (--p3: in Display P3's colors)
//
// <libdir> holds drie.c3l and what it needs (the game's lib/).
module reflections_test;
import drie, drie::postprocessing, drie::png, sokol, drie::readback;
import std::io, std::math;

const int W = 480;
const int H = 270;
// frames rendered from each viewpoint before the picture is taken (TAA and the reflections
// of the frame before settle)
const int SETTLE = 24;

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

// the columns: reflections off, on with the old thickness guess, on with the back faces, and
const int OFF = 0;
const int GUESS = 1;
const int BACK_FACES = 2;
// a Reflector for the floor, with screen-space reflections only for what stands on it
const int PLANAR = 3;
const int MODES = 4;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
RenderTarget* scene_rt;
RenderTarget* back_rt;
RenderTarget* toned;
GTAOPass* ao;
TAAPass* taa;
OutputPass* output;
drie::Mat4 prev_view_proj;
MeshStandardMaterial* wood;
Reflector* mirror;

fn void build_scene()
{
    scene = drie::scene();
    scene.background = 0x10131a;
    scene.add(drie::hemisphere_light(0x8090b0, 0x302820, 0.6));
    PointLight* lamp = drie::point_light(0xffe0b0, 6, 0, 2);
    lamp.position.set(-1.0f, 2.2f, 1.0f);
    scene.add(lamp);

    // the glossy floor, 6 x 6 m, one-sided: no back face anywhere near it
    wood = drie::mesh_standard_material(color: 0x6a3a1c, roughness: 0.18);
    wood.ssr = 1;
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

    // emissive cubes floating at two heights, a white one far over
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
    if (reset) taa.reset = true;
    taa.jitter(camera);
    if (mode == OFF || reset)
    {
        renderer.clear_reflection_source();
    }
    else
    {
        renderer.set_reflection_source(taa.texture(), ao.ao_target.texture, prev_view_proj, linear: true,
                                       back: mode >= BACK_FACES ? back_rt.texture : (SgView){});
    }
    renderer.set_render_target(scene_rt);
    wood.reflector = mode == PLANAR ? mirror : null;
    wood.ssr_reach = mode == PLANAR ? 0.8f : 0;
    if (mode == PLANAR) renderer.render_reflector(mirror, scene, camera);
    renderer.render(scene, camera);
    if (mode >= BACK_FACES) renderer.render_back_faces(back_rt);
    ao.render(scene_rt);
    taa.blend = 0.1f;
    taa.render(scene_rt);
    output.set_source(taa.texture());
    output.render(toned);
    drie::commit();
    prev_view_proj = camera.projection_matrix.mul(camera.matrix_world_inverse);
}

fn int main(String[] args)
{
    String out_path = "tests/build/reflections.png";
    bool p3;
    foreach (arg : args[1..])
    {
        if (arg == "--p3")
        {
            p3 = true;
        }
        else
        {
            out_path = arg;
        }
    }
    renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.tone_mapping = ACES_FILMIC_TONE_MAPPING;
    build_scene();
    mirror = drie::reflector({ 0, 1, 0 }, 0);
    camera = drie::perspective_camera(55, (float)W / H, 0.05f, 40);
    scene_rt = drie::render_target(W, H, 1, RGBA16F, mrt: true);
    back_rt = drie::render_target(W, H, 1, R16F);
    toned = drie::render_target(W, H, 1, RGBA8, false);
    ao = postprocessing::gtao_pass(camera);
    taa = postprocessing::taa_pass();
    output = postprocessing::output_pass(renderer);
    if (p3) output.color_space = DISPLAY_P3_COLOR_SPACE;

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
    if (!png::write(out_path, sheet, sheet_w, sheet_h, p3: p3, channels: 3))
    {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s: columns off, thickness guessed, back faces, reflector", out_path);
    return 0;
}
