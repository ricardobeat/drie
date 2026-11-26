// examples/cloth on the GPU (drie::gpu_cloth): handkerchiefs fall from the sky without end onto
// a checkered pavement and pile up. All are panels of one GpuCloth. One that has
// lain still for DORMANT_AFTER seconds is pinned where it lies: dormant, no longer solved, but
// still something the others land on. When every handkerchief is in use, the one dormant
// longest goes back up to fall again.
//
//   just example gpu_hankies                  in a window (drag to orbit, scroll to zoom)
//   just example gpu_hankies --png out.png    one frame, headless, at --time t (default 20 s)
module gpu_hankies_example;
import drie, drie::gpu_cloth, drie::objects, drie::postprocessing, drie::ui, drie::controls, drie::readback, drie::png, drie::gpu_timer, sokol;
import std::io, std::math;

const int W = 1280;
const int H = 720;
const int HANKIES = 64;
const int GRID = 21;
const float SIZE = 0.4f;
const float MASS = 0.03f;
const float SPAWN_EVERY = 0.3f;
const float AREA = 2.5f;
const float SPAWN_HEIGHT = 6;
const float CHECK_EVERY = 0.25f;
const float STILL = 0.01f;
const float DORMANT_AFTER = 3;
const float KNIT_REPEAT = 0.5f;
const Color[6] COLORS = { 0xe8e4dc, 0xc8303c, 0x2c4c9c, 0xe0b030, 0x3c8c5c, 0xd87ca0 };

enum State : char {
    PARKED,
    HELD,
    FALLING,
    DORMANT,
}

struct Hanky {
    State state;
    Mesh* mesh;
    int wait;
    float still;
    long dormant_order;
}

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
OrbitControls* controls;
Sky* sky;
RenderTarget* scene_rt;
GTAOPass* ao;
OutputPass* output;
GpuCloth* cloth;
Hanky[HANKIES] hankies;
Vector3[] seen;
Vector3[] last_check;
float spawn_clock;
float check_clock;
uint seed = 1;
long dormant_count;
int dropped;

fn float rand01(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return (float)(x & 0xffffff) / (float)0x1000000;
}

fn float rand() => rand01(seed++ * 7919);

fn uint point(int hanky, int c, int r) => cloth.index(hanky * GRID + c, r);

fn Texture* tiled(String path, float repeat, ColorSpace space = SRGB_COLOR_SPACE) {
    Texture* t = drie::load_texture(path, space, flip_y: false);
    t.wrap_s = REPEAT_WRAPPING;
    t.wrap_t = REPEAT_WRAPPING;
    t.repeat = { repeat, repeat };
    return t;
}

fn void make_cloth() {
    cloth = gpu_cloth::panels(HANKIES, GRID, GRID, { 0, -50, 0 }, { SIZE, 0, 0 }, { 0, 0, SIZE }, { SIZE * 1.5f, 0, 0 }, mass: MASS);
    for (uint i = 0; i < cloth.count(); i++) cloth.pin(i);
    cloth.substeps = 10;
    cloth.drag = 0.2f;
    cloth.lift = 0.5f;
    cloth.wind = { 0.8f, 0, 0.4f };
    cloth.gust = 1;
    cloth.damping = 1.0f;
    cloth.bend_model = DISTANCE;
    cloth.bend_compliance = 3;
    cloth.internal_damping = 0.6f;
    cloth.stretch_compliance = 2e-5f;
    cloth.thickness = 0.006f;
    cloth.floor_y = 0.005f;
    cloth.friction = 0.6f;
    cloth.self_distance = SIZE / (GRID - 1);
    cloth.self_every = 2;
    cloth.watch_points();
    seen = mem::new_array(Vector3, cloth.count());
    last_check = mem::new_array(Vector3, cloth.count());
    Texture* jersey = tiled("examples/textures/cotton_jersey_grey_diff.jpg", KNIT_REPEAT);
    Texture* jersey_normals = tiled("examples/textures/cotton_jersey_nor_gl.jpg", KNIT_REPEAT, NO_COLOR_SPACE);
    Texture* jersey_rough = tiled("examples/textures/cotton_jersey_rough.jpg", KNIT_REPEAT, NO_COLOR_SPACE);
    foreach (i, &h : hankies) {
        Color col = COLORS[i % COLORS.len];
        MeshStandardMaterial* m = drie::mesh_standard_material(color: col, roughness: 1, side: DOUBLE_SIDE);
        m.map = jersey;
        m.roughness_map = jersey_rough;
        m.sheen = 0.6f;
        m.normal_map = jersey_normals;
        m.normal_scale = { 2, -2 };
        m.sheen_color = drie::color_to_linear(col) * 0.5f + (Vector3) { 0.15f, 0.15f, 0.15f };
        h.mesh = cloth.make_mesh(m, panel: (int)i, detail: 3);
        h.mesh.cast_shadow = true;
        h.mesh.receive_shadow = true;
        h.mesh.visible = false;
        scene.add(h.mesh);
    }
    cloth.set_bounds({ 0, SPAWN_HEIGHT / 2, 0 }, AREA * 2 + SPAWN_HEIGHT);
}

fn int pick_hanky() {
    foreach (i, h : hankies) {
        if (h.state == PARKED) return (int)i;
    }
    int oldest = -1;
    foreach (i, h : hankies) {
        if (h.state == DORMANT && (oldest < 0 || h.dormant_order < hankies[oldest].dormant_order)) oldest = (int)i;
    }
    return oldest;
}

fn Vector3 turn(Vector3 p, float ax, float ay, float az) {
    Vector3 a = { p.x, p.y * math::cos(ax) - p.z * math::sin(ax), p.y * math::sin(ax) + p.z * math::cos(ax) };
    Vector3 b = { a.x * math::cos(az) - a.y * math::sin(az), a.x * math::sin(az) + a.y * math::cos(az), a.z };
    return { b.x * math::cos(ay) + b.z * math::sin(ay), b.y, -b.x * math::sin(ay) + b.z * math::cos(ay) };
}

fn void spawn() {
    int k = pick_hanky();
    if (k < 0) return;
    Hanky* h = &hankies[k];
    Vector3 center = { (rand() * 2 - 1) * AREA, SPAWN_HEIGHT + rand() * 2, (rand() * 2 - 1) * AREA };
    float ax = (rand() * 2 - 1) * 1.2f;
    float ay = rand() * 6.3f;
    float az = (rand() * 2 - 1) * 1.2f;
    float[4] ph;
    foreach (&p : ph) *p = rand() * 6.3f;
    for (int r = 0; r < GRID; r++) {
        float v = (float)r / (GRID - 1);
        for (int c = 0; c < GRID; c++) {
            float u = (float)c / (GRID - 1);
            float wave = 0.02f * math::sin(u * 7 + ph[0]) * math::sin(v * 6 + ph[1]) + 0.012f * math::sin(u * 11 + v * 9 + ph[2]);
            Vector3 local = { (u - 0.5f) * SIZE, wave, (v - 0.5f) * SIZE };
            uint i = point(k, c, r);
            if (cloth.inv_mass[i] != 0) cloth.pin(i);
            cloth.move_pin(i, center + turn(local, ax, ay, az));
        }
    }
    h.state = HELD;
    h.wait = 2;
    h.still = 0;
    h.mesh.visible = true;
    dropped++;
}

fn void let_go(int k) {
    for (int r = 0; r < GRID; r++) {
        for (int c = 0; c < GRID; c++) cloth.unpin(point(k, c, r));
    }
    hankies[k].state = FALLING;
}

fn void put_to_sleep(int k) {
    for (int r = 0; r < GRID; r++) {
        for (int c = 0; c < GRID; c++) {
            uint i = point(k, c, r);
            cloth.pin(i);
            cloth.move_pin(i, seen[i]);
        }
    }
    hankies[k].state = DORMANT;
    hankies[k].dormant_order = dormant_count++;
}

fn void check_still() {
    if (!cloth.seen_points(seen)) return;
    foreach (k, &h : hankies) {
        if (h.state != FALLING) continue;
        float most = 0;
        for (int r = 0; r < GRID; r++) {
            for (int c = 0; c < GRID; c++) {
                uint i = point((int)k, c, r);
                most = math::max(most, (seen[i] - last_check[i]).length());
                last_check[i] = seen[i];
            }
        }
        h.still = most < STILL ? h.still + CHECK_EVERY : 0;
        if (h.still >= DORMANT_AFTER) put_to_sleep((int)k);
    }
}

fn void step(float dt) {
    spawn_clock += dt;
    while (spawn_clock >= SPAWN_EVERY) {
        spawn_clock -= SPAWN_EVERY;
        spawn();
    }
    int steps = cloth.update(dt);
    foreach (k, &h : hankies) {
        if (h.state != HELD) continue;
        h.wait -= steps;
        if (h.wait <= 0) let_go((int)k);
    }
    check_clock += dt;
    if (check_clock >= CHECK_EVERY) {
        check_clock -= CHECK_EVERY;
        check_still();
    }
}

fn void build_scene() {
    scene = drie::scene();
    scene.background = 0xbcd4ec;
    scene.fog = drie::fog(0xc4d8ee, 60, 400);

    sky = objects::sky();
    sky.sun_position = { -0.5f, 0.3f, 0.8f };
    sky.cloud_coverage = 0.25f;
    sky.turbidity = 1.5f;
    sky.rayleigh = 1.2f;
    sky.mie_coefficient = 0.002f;
    sky.mie_directional_g = 0.9f;
    renderer.tone_mapping = AGX_TONE_MAPPING;
    renderer.tone_mapping_exposure = 0.75f;
    scene.add(sky);
    scene.add(drie::hemisphere_light(0xcfe0ff, 0x505860, 0.8));
    DirectionalLight* sun = drie::directional_light(0xfff2e0, 4.0);
    sun.position = sky.sun_position * 12;
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.cascades = 1;
    sun.shadow.normal_bias = 0.002f;
    sun.shadow_extent = 4;
    scene.add(sun);

    MeshStandardMaterial* floor_mat = drie::mesh_standard_material(color: 0xffffff, roughness: 1);
    floor_mat.map = tiled("examples/textures/checkered_pavement_tiles_diff.jpg", 500);
    floor_mat.normal_map = tiled("examples/textures/checkered_pavement_tiles_nor_gl.jpg", 500, NO_COLOR_SPACE);
    floor_mat.normal_scale = { 1, -1 };
    floor_mat.roughness_map = tiled("examples/textures/checkered_pavement_tiles_rough.jpg", 500, NO_COLOR_SPACE);
    floor_mat.ao_map = tiled("examples/textures/checkered_pavement_tiles_ao.jpg", 500, NO_COLOR_SPACE);
    floor_mat.ao_intensity = 1;
    floor_mat.parallax_map = tiled("examples/textures/checkered_pavement_tiles_disp.jpg", 500, NO_COLOR_SPACE);
    floor_mat.parallax_scale = 0.015f;
    floor_mat.parallax_shadow = 1;
    Mesh* floor = drie::mesh(drie::plane_geometry(1000, 1000), floor_mat);
    floor.rotation.x = -(float)math::PI / 2;
    floor.receive_shadow = true;
    scene.add(floor);

    make_cloth();

    camera = drie::perspective_camera(50, (float)W / H, 0.1f, 2000);
    camera.position = { 4.3f, 1.1f, 5.4f };
    camera.look_at_vec({ 0, 0.6f, 0 });
}

fn void render_frame(RenderTarget* target) {
    int w = target ? target.width : math::max(sapp::width(), 1);
    int h = target ? target.height : math::max(sapp::height(), 1);
    if (!scene_rt) {
        scene_rt = drie::render_target(w, h, 1, RGBA16F, mrt: true);
        ao = postprocessing::gtao_pass(camera);
        ao.radius = 0.3f;
        ao.fade_start = 15;
        ao.fade_end = 30;
        output = postprocessing::output_pass(renderer);
    } else if (scene_rt.width != w || scene_rt.height != h) {
        scene_rt.set_size(w, h);
    }
    if (camera.aspect != (float)w / h) {
        camera.aspect = (float)w / h;
        camera.update_projection_matrix();
    }
    sky.follow(camera.get_world_position(), camera.far);
    renderer.set_render_target(scene_rt);
    renderer.render(scene, camera);
    ao.render(scene_rt);
    output.set_source(scene_rt.texture);
    output.render(target);
    if (!target) drie::commit();
}

float last_time = -1;

fn void animate(double time) {
    float now = renderer.time;
    float dt = last_time < 0 ? 0 : math::min(now - last_time, 0.1f);
    last_time = now;
    controls.update();
    step(dt);
    render_frame(null);
}

fn void hud() {
    ui::setup();
    float scale = math::max(1.0f, math::round(sapp::heightf() / 540));
    sdtx::set_context(sdtx::default_context());
    sdtx::canvas(sapp::widthf() / scale, sapp::heightf() / scale);
    sdtx::origin(1, 1);
    sdtx::font(ui::FONT_KC853);
    sdtx::color3f(0.1f, 0.12f, 0.16f);
    int falling;
    int dormant;
    foreach (h : hankies) {
        if (h.state == FALLING || h.state == HELD) falling++;
        if (h.state == DORMANT) dormant++;
    }
    @pool() {
        sdtx::puts(string::tformat("dropped %d   moving %d   dormant %d   (%d handkerchiefs)\n", dropped, falling, dormant, HANKIES).zstr_tcopy());
        if (gpu_timer::enabled) {
            sdtx::puts(string::tformat("cloth on the GPU: %.2f ms\n", gpu_timer::last_ms("drie-gpu-cloth")).zstr_tcopy());
        }
    };
    sdtx::draw();
}

fn int main(String[] args) {
    String png_path;
    float at = 20;
    bool gpu_time;
    for (usz i = 1; i < args.len; i++) {
        switch (args[i]) {
            case "--png": if (i + 1 < args.len) png_path = args[++i];
            case "--time": if (i + 1 < args.len) at = (float)args[++i].to_double()!!;
            case "--gpu-time": gpu_time = true;
            default: io::eprintfn("unknown option %s", args[i]);
        }
    }
    if (!png_path) {
        renderer = drie::renderer(antialias: true, width: W, height: H, title: "drie: handkerchiefs");
        build_scene();
        controls = controls::orbit_controls(camera);
        controls.target = { 0, 0.6f, 0 };
        drie::add_overlay(&hud);
        renderer.set_animation_loop(&animate);
        return 0;
    }
    renderer = drie::renderer(antialias: true, width: W, height: H);
    renderer.init_headless();
    renderer.auto_time = false;
    renderer.time = at;
    build_scene();
    if (gpu_time) gpu_timer::enabled = true;
    float dt = 1.0f / 60;
    int frames = (int)math::round(at / dt);
    ulong began = stm::now();
    for (int f = 0; f < frames; f++) {
        step(dt);
        drie::commit();
        if (gpu_time && f == frames - 60) gpu_timer::reset();
    }
    int falling;
    int dormant;
    foreach (h : hankies) {
        if (h.state == FALLING || h.state == HELD) falling++;
        if (h.state == DORMANT) dormant++;
    }
    io::printfn("%.1f s in %.0f ms: dropped %d, moving %d, dormant %d", at, stm::ms(stm::since(began)), dropped, falling, dormant);
    if (gpu_time) {
        gpu_timer::poll();
        gpu_timer::report();
    }
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    render_frame(rt);
    drie::commit();
    char[] px = mem::new_array(char, (usz)W * H * 4);
    readback::read_image(rt.texture_image, W, H, px.ptr);
    if (!png::write(png_path, px, W, H)) {
        io::eprintfn("could not write %s", png_path);
        return 1;
    }
    io::printfn("saved %s", png_path);
    return 0;
}
