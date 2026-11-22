// Interactive Radiance Cascades & Split Radiance Cascades demo.
//
// A Cornell-inspired gallery room with a red left wall, cobalt blue right wall, white
// sculptures, pure emissive area-light panels (cyan, amber, magenta), and a hidden rear
// golden alcove. The key spotlight sweeps across the room so you can watch live color
// bleeding and indirect bounce light respond in real time.
//
//   just example radiance                  open window (drag to orbit, scroll to zoom)
//   just example radiance --png out.png    render one headless frame (--mode split|dense|off)
//
// Controls:
//   1 / 2 / 3 or Tab   switch between Split RC, Dense RC, and Direct Only (Off)
//   Space              pause / resume spotlight sweep
//   L                  toggle direct spotlights (on / emissive panels only)
//   F3                 toggle drie::rc_debug probe visualizer (M: 2D/3D, C: age/rad, P: alloc)
module radiance_example;
import drie, drie::postprocessing, drie::ui, drie::controls, drie::rc_debug, drie::readback, drie::png, sokol;
import std::io, std::math;

const int W = 1280;
const int H = 720;

enum RcMode : int
{
    SPLIT,
    DENSE,
    OFF,
}

const String[3] MODE_NAMES = { "SPLIT RC (sparse)", "DENSE RC (uniform)", "OFF (direct only)" };

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
OrbitControls* controls;
RenderTarget* scene_rt;
OutputPass* output;
RadianceCascades* dense_rc;
RadianceCascades* split_rc;
SpotLight* key_light;
SpotLight* fill_light;
SpotLight* bounce_light;

RcMode mode = SPLIT;
bool sweep = true;
bool lights_on = true;
double sim_time = 0;
double last_frame_t = -1;
double smooth_ms = 16.0;

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
        color: 0x000000, roughness: 0.9f, emissive: 0x18f0e0, emissive_intensity: 10.0f);
    MeshStandardMaterial* amber_glow = drie::mesh_standard_material(
        color: 0x000000, roughness: 0.9f, emissive: 0xff8818, emissive_intensity: 10.0f);
    MeshStandardMaterial* magenta_glow = drie::mesh_standard_material(
        color: 0x000000, roughness: 0.9f, emissive: 0xff2898, emissive_intensity: 9.0f);

    // Room shell: X in [-3.8, 3.8], Y in [0, 3.2], Z in [-3.8, 4.0]
    add_box({ 0, -0.1f, 0.1f }, { 8.0f, 0.2f, 8.2f }, floor_mat, false, true);
    add_box({ 0, 3.3f, 0.1f }, { 8.0f, 0.2f, 8.2f }, white, false, true);
    add_box({ 0, 1.6f, -3.9f }, { 8.0f, 3.4f, 0.2f }, white, true, true);
    add_box({ 0, 1.6f, 4.1f }, { 8.0f, 3.4f, 0.2f }, white, false, true);
    add_box({ -3.9f, 1.6f, 0.1f }, { 0.2f, 3.4f, 8.2f }, red_wall, true, true);
    add_box({ 3.9f, 1.6f, 0.1f }, { 0.2f, 3.4f, 8.2f }, blue_wall, true, true);

    // Central matte white sculptures
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

    // Partition archway & rear golden alcove
    add_box({ -2.75f, 1.6f, -2.0f }, { 2.1f, 3.2f, 0.24f }, white);
    add_box({ 2.75f, 1.6f, -2.0f }, { 2.1f, 3.2f, 0.24f }, white);
    add_box({ 0.0f, 2.75f, -2.0f }, { 3.4f, 0.9f, 0.24f }, white);
    add_box({ 0.0f, 1.55f, -3.72f }, { 3.6f, 2.9f, 0.12f }, gold_wall);

    // Emissive area-light panels (no analytical lights attached)
    add_box({ -3.65f, 1.45f, -2.95f }, { 0.14f, 2.3f, 0.65f }, cyan_glow, false, false);
    add_box({ 2.55f, 2.35f, -3.65f }, { 1.9f, 0.32f, 0.14f }, amber_glow, false, false);
    add_box({ -2.35f, 0.22f, 1.25f }, { 0.36f, 0.44f, 0.36f }, magenta_glow, false, false);

    // Sweeping key spotlight
    key_light = drie::spot_light(0xfff4e4, 28.0f, 14.0f, 0.85f, 0.35f, 2.0f);
    key_light.position = { 0.6f, 3.05f, 1.3f };
    key_light.target.position = { -1.2f, 0.4f, -0.4f };
    key_light.cast_shadow = true;
    key_light.shadow.map_size = 1024;
    key_light.shadow.near = 0.2f;
    key_light.shadow.far = 12.0f;
    key_light.shadow.bias = 0.0005f;
    scene.add(key_light);
    scene.add(key_light.target);

    // Fill spotlight hitting the blue wall
    fill_light = drie::spot_light(0xeef4ff, 22.0f, 12.0f, 0.72f, 0.4f, 2.0f);
    fill_light.position = { 0.4f, 2.95f, 1.1f };
    fill_light.target.position = { 3.7f, 1.2f, -0.3f };
    fill_light.cast_shadow = true;
    fill_light.shadow.map_size = 1024;
    fill_light.shadow.near = 0.2f;
    fill_light.shadow.far = 10.0f;
    fill_light.shadow.bias = 0.0005f;
    scene.add(fill_light);
    scene.add(fill_light.target);

    // Hidden alcove spotlight hitting the rear golden wall
    bounce_light = drie::spot_light(0xffe8b8, 34.0f, 9.0f, 0.68f, 0.35f, 2.0f);
    bounce_light.position = { 0.0f, 2.95f, -2.35f };
    bounce_light.target.position = { 0.0f, 1.3f, -3.7f };
    scene.add(bounce_light);
    scene.add(bounce_light.target);

    scene.update_matrix_world();

    Vector3 bmin = { -4.0f, -0.2f, -4.0f };
    Vector3 bmax = { 4.0f, 3.4f, 4.2f };

    dense_rc = drie::radiance_cascades(bmin, bmax, spacing: 0.20f, levels: 4);
    dense_rc.sky_from_scene = false;
    dense_rc.sky_color = { 0.004f, 0.005f, 0.008f };
    dense_rc.ground_color = { 0.004f, 0.004f, 0.004f };
    dense_rc.update_every = 1;
    dense_rc.blend = 0.12f;
    dense_rc.voxelize(scene);

    split_rc = drie::radiance_cascades_sparse(bmin, bmax, spacing: 0.20f, levels: 4, max_probes: 16384, hash_cap: 65536);
    split_rc.sky_from_scene = false;
    split_rc.sky_color = { 0.004f, 0.005f, 0.008f };
    split_rc.ground_color = { 0.004f, 0.004f, 0.004f };
    split_rc.update_every = 1;
    split_rc.blend = 0.12f;
    split_rc.voxelize(scene);

    camera = drie::perspective_camera(58, (float)W / H, 0.05f, 40);
    camera.position = { 0.0f, 1.55f, 3.4f };
    camera.update_matrix_world();
    camera.look_at_vec({ 0.0f, 1.20f, -1.0f });
}

fn void set_mode(RcMode m)
{
    mode = m;
    if (mode == DENSE) dense_rc.reset();
    if (mode == SPLIT) split_rc.reset();
}

fn void update_scene(double dt)
{
    if (sweep) sim_time += dt;
    float a = (float)math::sin(sim_time * 0.75);
    key_light.target.position = { a * 3.1f, 0.85f + 0.35f * (float)math::cos(sim_time * 0.5), -0.4f };
    key_light.intensity = lights_on ? 28.0f : 0.0f;
    fill_light.intensity = lights_on ? 22.0f : 0.0f;
    bounce_light.intensity = lights_on ? 34.0f : 0.0f;

    switch (mode)
    {
        case SPLIT:
            scene.radiance = split_rc;
        case DENSE:
            scene.radiance = dense_rc;
            dense_rc.update(scene);
        case OFF:
            scene.radiance = null;
    }
}

fn void animate(double time)
{
    double dt = last_frame_t < 0 ? 1.0 / 60.0 : math::clamp((time - last_frame_t) * 0.001, 0.0, 0.1);
    last_frame_t = time;

    int win_w = math::max(sapp::width(), 1);
    int win_h = math::max(sapp::height(), 1);
    if (!scene_rt)
    {
        scene_rt = drie::render_target(win_w, win_h, 1, RGBA16F);
        output = postprocessing::output_pass(renderer);
        camera.aspect = (float)win_w / (float)win_h;
        camera.update_projection_matrix();
    }
    else if (scene_rt.width != win_w || scene_rt.height != win_h)
    {
        scene_rt.set_size(win_w, win_h);
        camera.aspect = (float)win_w / (float)win_h;
        camera.update_projection_matrix();
    }

    ulong t0 = stm::now();
    update_scene(dt);
    if (controls)
    {
        controls.enabled = !rc_debug::active;
        if (!rc_debug::active) controls.update();
    }

    renderer.set_render_target(scene_rt);
    Vector3 dbg_focus = controls ? controls.target : (Vector3){ 0, 1.2f, -0.5f };
    Camera* rc_cam = rc_debug::update((float)dt, win_w, win_h, dbg_focus, 2.95f);
    if (rc_debug::active && rc_cam)
    {
        if (scene.radiance && scene.radiance.sparse && rc_debug::freeze_alloc)
        {
            scene.radiance.freeze_alloc = false;
            renderer.render(scene, camera);
            scene.radiance.freeze_alloc = true;
            renderer.render(scene, rc_cam);
            scene.radiance.freeze_alloc = false;
        }
        else
        {
            renderer.render(scene, rc_cam);
        }
    }
    else
    {
        renderer.render(scene, camera);
    }
    output.set_source(scene_rt.texture);
    output.render(null);
    drie::commit();
    rc_debug::track(scene.radiance);

    double ms = stm::ms(stm::since(t0));
    smooth_ms = smooth_ms * 0.9 + ms * 0.1;
}

fn void on_event(Event* e)
{
    Vector3 dbg_focus = controls ? controls.target : (Vector3){ 0, 1.2f, -0.5f };
    if (rc_debug::handle_event(e, dbg_focus, 2.95f)) return;
    if (e.type != KEY_DOWN || e.key_repeat) return;
    switch (e.key_code)
    {
        case NUM_1: set_mode(SPLIT);
        case NUM_2: set_mode(DENSE);
        case NUM_3: set_mode(OFF);
        case TAB:   set_mode(mode == SPLIT ? DENSE : (mode == DENSE ? OFF : SPLIT));
        case SPACE: sweep = !sweep;
        case L:     lights_on = !lights_on;
        default: {}
    }
}

fn void hud()
{
    if (rc_debug::active)
    {
        Vector3 look = controls ? (controls.target - camera.position).normalize() : (Vector3){ 0, 0, -1 };
        int win_w = math::max(sapp::width(), 1);
        int win_h = math::max(sapp::height(), 1);
        int ui_h = H;
        int ui_w = math::max((int)math::round((float)ui_h * (float)win_w / (float)win_h), 640);
        rc_debug::draw_window(scene.radiance, MODE_NAMES[mode.ordinal], ui_w, ui_h, camera.position, look);
        return;
    }
    ui::setup();
    float scale = math::max(1.0f, math::round(sapp::heightf() / 640));
    sdtx::set_context(sdtx::default_context());
    sdtx::canvas(sapp::widthf() / scale, sapp::heightf() / scale);
    sdtx::origin(1.5f, 1.5f);
    sdtx::font(ui::FONT_ORIC);
    @pool() {
        sdtx::color3f(0.35f, 0.92f, 1.0f);
        sdtx::puts(string::tformat("RADIANCE CASCADES  [%s]   %.1f ms (%.0f fps)\n\n",
            MODE_NAMES[mode.ordinal], smooth_ms, 1000.0 / math::max(smooth_ms, 0.1)).zstr_tcopy());
        sdtx::color3f(0.90f, 0.90f, 0.92f);
        sdtx::puts("1: Split RC   2: Dense RC   3: Off   Tab: cycle\n");
        sdtx::puts(string::tformat("Space: sweep [%s]   L: spotlights [%s]   F3: probe debugger\n",
            sweep ? "ON" : "PAUSED", lights_on ? "ON" : "EMISSIVES ONLY").zstr_tcopy());
        sdtx::color3f(0.65f, 0.70f, 0.76f);
        sdtx::puts("Drag: orbit   Scroll: zoom   Right-drag: pan");
    };
    sdtx::draw();
}

fn int main(String[] args)
{
    String png_path;
    for (usz i = 1; i < args.len; i++)
    {
        if (args[i] == "--png" && i + 1 < args.len)
        {
            png_path = args[++i];
        }
        else if (args[i] == "--mode" && i + 1 < args.len)
        {
            String m = args[++i];
            if (m == "dense") { mode = DENSE; }
            else if (m == "off") { mode = OFF; }
            else { mode = SPLIT; }
        }
        else if (args[i] == "--no-lights")
        {
            lights_on = false;
        }
        else if (args[i] == "--rc-debug")
        {
            rc_debug::active = true;
        }
        else if (args[i] == "--rc-debug-3d")
        {
            rc_debug::active = true;
            rc_debug::set_3d(true);
        }
        else
        {
            io::eprintfn("unknown option %s", args[i]);
        }
    }

    if (!png_path)
    {
        renderer = drie::renderer(antialias: false, width: W, height: H, title: "drie: radiance cascades");
        renderer.tone_mapping = ACES_FILMIC_TONE_MAPPING;
        renderer.tone_mapping_exposure = 1.15f;
        build_scene();
        controls = controls::orbit_controls(camera);
        controls.target = { 0.0f, 1.20f, -1.0f };
        drie::add_event_listener(&on_event);
        drie::add_overlay(&hud);
        renderer.set_animation_loop(&animate);
        return 0;
    }

    renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.tone_mapping = ACES_FILMIC_TONE_MAPPING;
    renderer.tone_mapping_exposure = 1.15f;
    build_scene();
    scene_rt = drie::render_target(W, H, 1, RGBA16F);
    RenderTarget* toned = drie::render_target(W, H, 1, RGBA8, false);
    output = postprocessing::output_pass(renderer);
    renderer.set_render_target(scene_rt);
    for (int f = 0; f < 24; f++)
    {
        update_scene(0);
        Camera* rc_cam = rc_debug::active ? rc_debug::update(0, W, H, camera.position, 2.95f) : null;
        if (rc_debug::active && rc_cam)
        {
            if (scene.radiance && scene.radiance.sparse && rc_debug::freeze_alloc)
            {
                scene.radiance.freeze_alloc = false;
                renderer.render(scene, camera);
                scene.radiance.freeze_alloc = true;
                renderer.render(scene, rc_cam);
                scene.radiance.freeze_alloc = false;
            }
            else
            {
                renderer.render(scene, rc_cam);
            }
        }
        else
        {
            renderer.render(scene, camera);
        }
        rc_debug::track(scene.radiance, wait_now: f == 23);
        output.set_source(scene_rt.texture);
        output.render(toned);
        if (rc_debug::active)
        {
            rc_debug::draw_target(toned, scene.radiance, MODE_NAMES[mode.ordinal], W, H, camera.position, { 0, 0, -1 });
        }
        drie::commit();
    }
    char[] px = mem::new_array(char, (usz)W * H * 4);
    readback::read_image(toned.texture_image, W, H, px.ptr);
    if (!png::write(png_path, px, W, H))
    {
        io::eprintfn("could not write %s", png_path);
        return 1;
    }
    io::printfn("saved %s (%s)", png_path, MODE_NAMES[mode.ordinal]);
    return 0;
}
