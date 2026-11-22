// Showcase of Radiance Cascades (dense) and Split Radiance Cascades (sparse) in a
// Cornell-inspired gallery room designed to highlight dynamic global illumination:
//   - Color bleeding from a sunlit red wall and blue wall onto white sculptures
//   - Emissive area-light panels (cyan strip + warm amber panel) casting soft shadows
//   - Indirect bounce light spilling around a corner baffle into an unlit alcove
//
// Columns: Direct only (no RC) | Dense RC (0.20 m) | Split RC (0.20 m, sparse)
//
//   just test radiance_showcase [out.png] [--p3]
module radiance_showcase;
import drie, drie::postprocessing, drie::png, sokol, drie::readback;
import std::io, std::math;

const int W = 640;
const int H = 360;
const int SETTLE = 36;

struct View
{
    String name;
    Vector3 eye;
    Vector3 at;
}

View[4] views = {
    { "gallery overview (color bleed + emissives)", { 0.0f, 1.55f, 3.6f }, { 0.0f, 1.25f, -1.5f } },
    { "cornell sculptures (red/blue bounce)", { -0.35f, 1.15f, 2.35f }, { 0.0f, 0.95f, -0.8f } },
    { "emissive archway (area lights + soft shadows)", { 1.45f, 1.20f, 1.85f }, { -1.6f, 1.10f, -1.6f } },
    { "bounced corner (indirect light around baffle)", { -2.45f, 1.20f, 2.85f }, { 1.6f, 1.15f, -2.5f } },
};

const int DIRECT_ONLY = 0;
const int DENSE_RC = 1;
const int SPLIT_RC = 2;
const int MODES = 3;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
RenderTarget* scene_rt;
RenderTarget* toned;
GTAOPass* ao;
TAAPass* taa;
OutputPass* output;
RadianceCascades* dense_rc;
RadianceCascades* split_rc;

fn void add_box(Vector3 pos, Vector3 size, Material* mat, bool cast_shadow = true, bool recv_shadow = true)
{
    Mesh* m = drie::mesh(drie::box_geometry(size.x, size.y, size.z), mat);
    m.position = pos;
    m.cast_shadow = cast_shadow;
    m.receive_shadow = recv_shadow;
    scene.add(m);
}

fn void build_scene()
{
    scene = drie::scene();
    scene.background = 0x040508;

    MeshStandardMaterial* white = drie::mesh_standard_material(color: 0xe6e4de, roughness: 0.88f);
    MeshStandardMaterial* floor_mat = drie::mesh_standard_material(color: 0xd8d4cc, roughness: 0.55f);
    MeshStandardMaterial* red_wall = drie::mesh_standard_material(color: 0xd82820, roughness: 0.90f);
    MeshStandardMaterial* blue_wall = drie::mesh_standard_material(color: 0x2258e8, roughness: 0.90f);
    MeshStandardMaterial* gold_wall = drie::mesh_standard_material(color: 0xf0a820, roughness: 0.88f);
    MeshStandardMaterial* pillar_mat = drie::mesh_standard_material(color: 0xeae8e4, roughness: 0.72f);

    MeshStandardMaterial* cyan_glow = drie::mesh_standard_material(
        color: 0x000000, roughness: 0.9f, emissive: 0x18f0e0, emissive_intensity: 7.5f);
    MeshStandardMaterial* amber_glow = drie::mesh_standard_material(
        color: 0x000000, roughness: 0.9f, emissive: 0xff8818, emissive_intensity: 7.0f);
    MeshStandardMaterial* magenta_glow = drie::mesh_standard_material(
        color: 0x000000, roughness: 0.9f, emissive: 0xff2898, emissive_intensity: 6.5f);

    // Room shell: X in [-3.8, 3.8], Y in [0, 3.2], Z in [-3.8, 4.0]
    add_box({ 0, -0.1f, 0.1f }, { 8.0f, 0.2f, 8.2f }, floor_mat, false, true);
    add_box({ 0, 3.3f, 0.1f }, { 8.0f, 0.2f, 8.2f }, white, false, true);
    add_box({ 0, 1.6f, -3.9f }, { 8.0f, 3.4f, 0.2f }, white, true, true);
    add_box({ 0, 1.6f, 4.1f }, { 8.0f, 3.4f, 0.2f }, white, false, true);

    // Vivid red left wall & cobalt blue right wall for strong Cornell color bleeding
    add_box({ -3.9f, 1.6f, 0.1f }, { 0.2f, 3.4f, 8.2f }, red_wall, true, true);
    add_box({ 3.9f, 1.6f, 0.1f }, { 0.2f, 3.4f, 8.2f }, blue_wall, true, true);

    // Central white sculptures (tall rotated block + short block + matte sphere)
    Mesh* tall = drie::mesh(drie::box_geometry(0.85f, 1.85f, 0.85f), pillar_mat);
    tall.position = { -0.85f, 0.925f, -0.65f };
    tall.rotation.y = 0.32f;
    tall.cast_shadow = true;
    tall.receive_shadow = true;
    scene.add(tall);

    Mesh* short_b = drie::mesh(drie::box_geometry(0.95f, 0.85f, 0.95f), pillar_mat);
    short_b.position = { 0.90f, 0.425f, -0.15f };
    short_b.rotation.y = -0.38f;
    short_b.cast_shadow = true;
    short_b.receive_shadow = true;
    scene.add(short_b);

    Mesh* orb = drie::mesh(drie::sphere_geometry(0.42f, 32, 24), pillar_mat);
    orb.position = { 0.90f, 1.27f, -0.15f };
    orb.cast_shadow = true;
    orb.receive_shadow = true;
    scene.add(orb);

    // Partition archway at Z = -2.0 separating the front gallery from the rear alcove
    add_box({ -2.75f, 1.6f, -2.0f }, { 2.1f, 3.2f, 0.24f }, white);
    add_box({ 2.75f, 1.6f, -2.0f }, { 2.1f, 3.2f, 0.24f }, white);
    add_box({ 0.0f, 2.75f, -2.0f }, { 3.4f, 0.9f, 0.24f }, white);

    // Rear golden reflector wall (hit by a hidden spotlight to bounce warm light through the archway)
    add_box({ 0.0f, 1.55f, -3.72f }, { 3.6f, 2.9f, 0.12f }, gold_wall);

    // Pure emissive area-light panels (no analytical PointLight/SpotLight attached!)
    // 1. Tall cyan vertical light bar tucked on the left side behind the archway
    add_box({ -3.65f, 1.45f, -2.95f }, { 0.14f, 2.3f, 0.65f }, cyan_glow, false, false);
    // 2. Warm amber horizontal bar on the right rear alcove
    add_box({ 2.55f, 2.35f, -3.65f }, { 1.9f, 0.32f, 0.14f }, amber_glow, false, false);
    // 3. Small magenta emissive cube resting on the floor in the front-left foreground
    add_box({ -2.35f, 0.22f, 1.25f }, { 0.36f, 0.44f, 0.36f }, magenta_glow, false, false);

    // Key overhead spotlight angled across the red wall & sculptures (casts sharp direct shadows)
    SpotLight* key = drie::spot_light(0xfff4e4, 28.0f, 14.0f, 0.85f, 0.35f, 2.0f);
    key.position = { 0.6f, 3.05f, 1.3f };
    key.target.position = { -1.2f, 0.4f, -0.4f };
    key.cast_shadow = true;
    key.shadow.map_size = 1024;
    key.shadow.near = 0.2f;
    key.shadow.far = 12.0f;
    key.shadow.bias = 0.0005f;
    scene.add(key);
    scene.add(key.target);

    // Second tight spotlight hitting the blue wall so both sides bleed vivid color
    SpotLight* fill = drie::spot_light(0xeef4ff, 22.0f, 12.0f, 0.72f, 0.4f, 2.0f);
    fill.position = { 0.4f, 2.95f, 1.1f };
    fill.target.position = { 3.7f, 1.2f, -0.3f };
    fill.cast_shadow = true;
    fill.shadow.map_size = 1024;
    fill.shadow.near = 0.2f;
    fill.shadow.far = 10.0f;
    fill.shadow.bias = 0.0005f;
    scene.add(fill);
    scene.add(fill.target);

    // Hidden alcove spotlight hitting the rear golden wall (no direct line of sight to front room)
    SpotLight* bounce = drie::spot_light(0xffe8b8, 34.0f, 9.0f, 0.68f, 0.35f, 2.0f);
    bounce.position = { 0.0f, 2.95f, -2.35f };
    bounce.target.position = { 0.0f, 1.3f, -3.7f };
    scene.add(bounce);
    scene.add(bounce.target);
}

fn void frame(View* v, int mode, bool reset)
{
    camera.position = v.eye;
    camera.update_matrix_world();
    camera.look_at_vec(v.at);
    if (reset)
    {
        taa.reset = true;
        if (mode == DENSE_RC) dense_rc.reset();
        if (mode == SPLIT_RC) split_rc.reset();
    }
    taa.jitter(camera);
    if (mode == DIRECT_ONLY)
    {
        scene.radiance = null;
    }
    else if (mode == DENSE_RC)
    {
        scene.radiance = dense_rc;
        dense_rc.update(scene);
    }
    else
    {
        scene.radiance = split_rc;
    }
    renderer.set_render_target(scene_rt);
    renderer.render(scene, camera);
    ao.render(scene_rt);
    taa.blend = 0.12f;
    taa.render(scene_rt);
    output.set_source(taa.texture());
    output.render(toned);
    drie::commit();
}

fn int main(String[] args)
{
    String out_path = "tests/build/radiance_showcase.png";
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
    renderer.tone_mapping_exposure = 1.15f;

    build_scene();
    scene.update_matrix_world();

    Vector3 bmin = { -4.0f, -0.2f, -4.0f };
    Vector3 bmax = { 4.0f, 3.4f, 4.2f };

    dense_rc = drie::radiance_cascades(bmin, bmax, spacing: 0.20f, levels: 4);
    dense_rc.sky_from_scene = false;
    dense_rc.sky_color = { 0.004f, 0.005f, 0.008f };
    dense_rc.ground_color = { 0.004f, 0.004f, 0.004f };
    dense_rc.update_every = 1;
    dense_rc.blend = 0.25f;
    dense_rc.voxelize(scene);

    split_rc = drie::radiance_cascades_sparse(bmin, bmax, spacing: 0.20f, levels: 4, max_probes: 16384, hash_cap: 65536);
    split_rc.sky_from_scene = false;
    split_rc.sky_color = { 0.004f, 0.005f, 0.008f };
    split_rc.ground_color = { 0.004f, 0.004f, 0.004f };
    split_rc.update_every = 1;
    split_rc.blend = 0.25f;
    split_rc.voxelize(scene);

    io::printfn("dense RC: %dx%dx%d (%d L0 probes), %dx%dx%d vox",
                dense_rc.nx, dense_rc.ny, dense_rc.nz, dense_rc.nx * dense_rc.ny * dense_rc.nz,
                dense_rc.vx, dense_rc.vy, dense_rc.vz);
    io::printfn("split RC: %d max L0 probes (hash %d), %dx%dx%d vox",
                split_rc.level_cap[0], split_rc.hash_cap,
                split_rc.vx, split_rc.vy, split_rc.vz);

    camera = drie::perspective_camera(58, (float)W / H, 0.05f, 40);
    scene_rt = drie::render_target(W, H, 1, RGBA16F, mrt: true);
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
    io::printfn("saved %s: columns = Direct Only | Dense RC | Split RC", out_path);
    return 0;
}
