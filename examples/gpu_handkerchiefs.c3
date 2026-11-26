// examples/cloth on the GPU (drie::gpu_cloth): handkerchiefs fall from the sky without end onto
// the pebble floor of a stone hall lit through its windows, and pile up. All are panels of one GpuCloth. One that has
// lain still for DORMANT_AFTER seconds is pinned where it lies: dormant, no longer solved, but
// still something the others land on. When every handkerchief is in use, the one dormant
// longest goes back up to fall again.
//
//   just example gpu_handkerchiefs                  in a window (the mouse looks round, WASD to
//                                                   walk, E and Q up and down, Esc frees the mouse)
//   just example gpu_handkerchiefs --png out.png    one frame, headless, at --time t (default 20 s)
module gpu_handkerchiefs_example;
import drie, drie::gpu_cloth, drie::objects, drie::postprocessing, drie::ui, drie::controls, drie::readback, drie::png, drie::gpu_timer, drie::profile, sokol;
import std::io, std::math, std::collections::list;

const int W = 1280;
const int H = 720;
const float RENDER_SCALE = 0.7f;
const int HANDKERCHIEFS = 64;
const int GRID = 21;
const float SIZE = 0.4f;
const float MASS = 0.03f;
const float SPAWN_EVERY = 0.15f;
const float AREA = 2.1f;
const float SPAWN_HEIGHT = 4.6f;
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

struct Handkerchief {
    State state;
    Mesh* mesh;
    int wait;
    float still;
    long dormant_order;
}

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
WalkControls* controls;
Sky* sky;
RenderTarget* scene_rt;
GTAOPass* ao;
UnrealBloomPass* bloom;
LensFlarePass* flare;
TAAPass* taa;
OutputPass* output;
GpuCloth* cloth;
Handkerchief[HANDKERCHIEFS] handkerchiefs;
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

fn uint point(int handkerchief, int c, int r) => cloth.index(handkerchief * GRID + c, r);

fn Texture* tiled(String path, float repeat, ColorSpace space = SRGB_COLOR_SPACE) {
    Texture* t = drie::load_texture(path, space, flip_y: false);
    t.wrap_s = REPEAT_WRAPPING;
    t.wrap_t = REPEAT_WRAPPING;
    t.repeat = { repeat, repeat };
    return t;
}

fn void make_cloth() {
    cloth = gpu_cloth::panels(HANDKERCHIEFS, GRID, GRID, { 0, -50, 0 }, { SIZE, 0, 0 }, { 0, 0, SIZE }, { SIZE * 1.5f, 0, 0 }, mass: MASS);
    for (uint i = 0; i < cloth.count(); i++) cloth.pin(i);
    cloth.substeps = 8;
    cloth.drag = 0.3f;
    cloth.lift = 0.5f;
    cloth.wind = { 0.8f, 0, 0.4f };
    cloth.gust = 1;
    cloth.damping = 1.0f;
    cloth.bend_model = DISTANCE;
    cloth.bend_compliance = 3;
    cloth.internal_damping = 0.6f;
    cloth.stretch_compliance = 2e-5f;
    cloth.thickness = 0.006f;
    cloth.floor_y = 0.018f;
    cloth.friction = 0.6f;
    cloth.self_distance = SIZE / (GRID - 1);
    cloth.self_every = 3;
    cloth.watch_points();
    seen = mem::new_array(Vector3, (sz)cloth.count());
    last_check = mem::new_array(Vector3, (sz)cloth.count());
    Texture* jersey = tiled("examples/textures/cotton_jersey_grey_diff.jpg", KNIT_REPEAT);
    Texture* jersey_normals = tiled("examples/textures/cotton_jersey_nor_gl.jpg", KNIT_REPEAT, NO_COLOR_SPACE);
    Texture* jersey_rough = tiled("examples/textures/cotton_jersey_rough.jpg", KNIT_REPEAT, NO_COLOR_SPACE);
    foreach (i, &h : handkerchiefs) {
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

fn int pick_handkerchief() {
    foreach (i, h : handkerchiefs) {
        if (h.state == PARKED) return (int)i;
    }
    int oldest = -1;
    foreach (i, h : handkerchiefs) {
        if (h.state == DORMANT && (oldest < 0 || h.dormant_order < handkerchiefs[oldest].dormant_order)) oldest = (int)i;
    }
    return oldest;
}

fn Vector3 turn(Vector3 p, float ax, float ay, float az) {
    Vector3 a = { p.x, p.y * math::cos(ax) - p.z * math::sin(ax), p.y * math::sin(ax) + p.z * math::cos(ax) };
    Vector3 b = { a.x * math::cos(az) - a.y * math::sin(az), a.x * math::sin(az) + a.y * math::cos(az), a.z };
    return { b.x * math::cos(ay) + b.z * math::sin(ay), b.y, -b.x * math::sin(ay) + b.z * math::cos(ay) };
}

fn void spawn() {
    int k = pick_handkerchief();
    if (k < 0) return;
    Handkerchief* h = &handkerchiefs[k];
    Vector3 center = { (rand() * 2 - 1) * AREA, SPAWN_HEIGHT + rand() * 1.2f, (rand() * 2 - 1) * AREA };
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
    handkerchiefs[k].state = FALLING;
}

fn void put_to_sleep(int k) {
    for (int r = 0; r < GRID; r++) {
        for (int c = 0; c < GRID; c++) {
            uint i = point(k, c, r);
            cloth.pin(i);
            cloth.move_pin(i, seen[i]);
        }
    }
    handkerchiefs[k].state = DORMANT;
    handkerchiefs[k].dormant_order = dormant_count++;
}

fn void check_still() {
    if (!cloth.seen_points(seen)) return;
    foreach (k, &h : handkerchiefs) {
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
    foreach (k, &h : handkerchiefs) {
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

const Vector3 START_EYE = { -2.06f, 0.74f, -4.31f };
const Vector3 START_AT = { -1.0f, 0.61f, -2.7f };
const float ROOM = 5;
const float CEILING = 6.5f;
const float THICK = 0.8f;

struct Opening {
    float u0, u1, v0, v1;
}

fn MeshStandardMaterial* surface(String name, float tile, float depth) {
    MeshStandardMaterial* m = drie::mesh_standard_material(color: 0xffffff, roughness: 1, side: DOUBLE_SIDE);
    float r = 1 / tile;
    m.map = tiled(string::tformat("examples/textures/%s_diff.jpg", name), r);
    m.normal_map = tiled(string::tformat("examples/textures/%s_nor_gl.jpg", name), r, NO_COLOR_SPACE);
    m.normal_scale = { 1, -1 };
    m.roughness_map = tiled(string::tformat("examples/textures/%s_rough.jpg", name), r, NO_COLOR_SPACE);
    m.ao_map = tiled(string::tformat("examples/textures/%s_ao.jpg", name), r, NO_COLOR_SPACE);
    m.ao_intensity = 1;
    if (depth > 0) {
        Texture* heights = tiled(string::tformat("examples/textures/%s_disp.jpg", name), r, NO_COLOR_SPACE);
        heights.keep_pixels = true;
        m.displacement_map = heights;
        m.displacement_scale = depth;
        m.displacement_bias = 0;
    }
    return m;
}

float cell = 1e9;

fn void quad(BufferGeometry* g, Vector3 o, Vector3 u, Vector3 v) {
    Vector3 n = u.cross(v).normalize();
    Vector3 du = u.normalize();
    Vector3 dv = v.normalize();
    Vector3 mid = o + (u + v) * 0.5f;
    bool seen = math::abs(mid.x) <= ROOM + 0.01f && math::abs(mid.z) <= ROOM + 0.01f && mid.y >= -0.01f
                && mid.y <= CEILING + 0.01f && n.dot(mid - (Vector3) { 0, CEILING * 0.5f, 0 }) < 0;
    float size = seen ? (mid.y >= CEILING - 0.01f ? cell * 4 : cell) : 1e9;
    int nu = math::max((int)math::ceil(u.length() / size), 1);
    int nv = math::max((int)math::ceil(v.length() / size), 1);
    uint first = (uint)(g.vertices.len() / drie::FLOATS_PER_VERTEX);
    for (int j = 0; j <= nv; j++) {
        for (int i = 0; i <= nu; i++) {
            Vector3 p = o + u * ((float)i / nu) + v * ((float)j / nv);
            g.add_vertex(p, n, { p.dot(du), p.dot(dv) });
        }
    }
    for (int j = 0; j < nv; j++) {
        for (int i = 0; i < nu; i++) {
            uint a = first + (uint)(j * (nu + 1) + i);
            uint b = a + 1;
            uint c = b + (uint)(nu + 1);
            uint d = a + (uint)(nu + 1);
            g.add_triangle(a, b, c);
            g.add_triangle(a, c, d);
        }
    }
}

fn void block(BufferGeometry* g, Vector3 lo, Vector3 hi) {
    Vector3 d = hi - lo;
    if (d.x <= 0 || d.y <= 0 || d.z <= 0) return;
    quad(g, { hi.x, lo.y, hi.z }, { 0, 0, -d.z }, { 0, d.y, 0 });
    quad(g, lo, { 0, 0, d.z }, { 0, d.y, 0 });
    quad(g, { lo.x, lo.y, hi.z }, { d.x, 0, 0 }, { 0, d.y, 0 });
    quad(g, { hi.x, lo.y, lo.z }, { -d.x, 0, 0 }, { 0, d.y, 0 });
    quad(g, { lo.x, hi.y, lo.z }, { 0, 0, d.z }, { d.x, 0, 0 });
    quad(g, lo, { d.x, 0, 0 }, { 0, 0, d.z });
}

fn void wall_piece(BufferGeometry* g, int axis, float side, float u0, float u1, float v0, float v1) {
    float a = side > 0 ? ROOM : -ROOM - THICK;
    float b = side > 0 ? ROOM + THICK : -ROOM;
    if (axis == 0) {
        block(g, { a, v0, u0 }, { b, v1, u1 });
    } else {
        block(g, { u0, v0, a }, { u1, v1, b });
    }
}

fn void wall(BufferGeometry* g, int axis, float side, float from, float to, Opening[] holes) {
    List{float} cuts;
    defer cuts.free();
    cuts.push(from);
    foreach (o : holes) {
        cuts.push(o.u0);
        cuts.push(o.u1);
    }
    cuts.push(to);
    for (sz i = 0; i + 1 < cuts.len(); i++) {
        float u0 = cuts[i];
        float u1 = cuts[i + 1];
        bool open;
        foreach (o : holes) {
            if (o.u0 <= u0 && u1 <= o.u1) {
                wall_piece(g, axis, side, u0, u1, 0, o.v0);
                wall_piece(g, axis, side, u0, u1, o.v1, CEILING);
                open = true;
            }
        }
        if (!open) wall_piece(g, axis, side, u0, u1, 0, CEILING);
    }
}

fn void lattice(BufferGeometry* g, int axis, float side, Opening o) {
    float mid = side * (ROOM + THICK * 0.45f);
    const float BAR = 0.05f;
    const float THIN = 0.022f;
    const float DEEP = 0.04f;
    List{Vector3[2]} list;
    defer list.free();
    list.push({ { o.u0, o.v0, -DEEP }, { o.u0 + BAR, o.v1, DEEP } });
    list.push({ { o.u1 - BAR, o.v0, -DEEP }, { o.u1, o.v1, DEEP } });
    list.push({ { o.u0, o.v0, -DEEP }, { o.u1, o.v0 + BAR, DEEP } });
    list.push({ { o.u0, o.v1 - BAR, -DEEP }, { o.u1, o.v1, DEEP } });
    for (int i = 1; i < 3; i++) {
        float u = o.u0 + (o.u1 - o.u0) * i / 3;
        list.push({ { u - THIN / 2, o.v0, -DEEP / 2 }, { u + THIN / 2, o.v1, DEEP / 2 } });
    }
    for (int i = 1; i < 7; i++) {
        float v = o.v0 + (o.v1 - o.v0) * i / 7;
        list.push({ { o.u0, v - THIN / 2, -DEEP / 2 }, { o.u1, v + THIN / 2, DEEP / 2 } });
    }
    foreach (bar : list) {
        Vector3 lo = bar[0];
        Vector3 hi = bar[1];
        if (axis == 0) {
            block(g, { mid + lo.z, lo.y, lo.x }, { mid + hi.z, hi.y, hi.x });
        } else {
            block(g, { lo.x, lo.y, mid + lo.z }, { hi.x, hi.y, mid + hi.z });
        }
    }
}

fn Mesh* solid(BufferGeometry* g, Material* m) {
    g.bounds_center = { 0, CEILING / 2, 0 };
    g.bounds_radius = 12;
    Mesh* mesh = drie::mesh(g, m);
    mesh.cast_shadow = true;
    mesh.receive_shadow = true;
    scene.add(mesh);
    return mesh;
}

fn BufferGeometry* hall(Opening[] side_windows, Opening[] back_holes) {
    BufferGeometry* g = drie::buffer_geometry();
    wall(g, 0, 1, -ROOM, ROOM, side_windows);
    wall(g, 1, -1, -ROOM - THICK, ROOM + THICK, back_holes);
    wall(g, 0, -1, -ROOM, ROOM, { });
    wall(g, 1, 1, -ROOM - THICK, ROOM + THICK, { });
    block(g, { -3.4f, 0, -ROOM - THICK - 0.4f }, { -2.0f, 2.6f, -ROOM - THICK });
    block(g, { -ROOM - THICK, CEILING, -ROOM - THICK }, { ROOM + THICK, CEILING + 0.4f, ROOM + THICK });
    return g;
}

fn BufferGeometry* ground() {
    BufferGeometry* g = drie::buffer_geometry();
    block(g, { -ROOM - THICK - 0.4f, -0.3f, -ROOM - THICK - 0.4f }, { ROOM + THICK, 0, ROOM + THICK });
    return g;
}

fn void build_scene() {
    scene = drie::scene();
    scene.background = 0xbcd4ec;
    Fog* haze = drie::volumetric_fog(0x2a241e, 0.06f, 0);
    haze.sun_color = 0xffd9a8;
    haze.sun_sharpness = 6;
    haze.sun_shafts = 1;
    haze.shade_density = 0.35f;
    haze.noise_amount = 0.55f;
    haze.noise_scale = 0.5f;
    haze.noise_wind = { 0.06f, 0.015f, 0.03f };
    haze.resolution = 0.25f;
    haze.slices = 128;
    scene.fog = haze;

    sky = objects::sky();
    sky.sun_position = { 0.85f, 0.3f, 0.25f };
    sky.cloud_coverage = 0.25f;
    sky.turbidity = 2;
    sky.rayleigh = 1.2f;
    sky.mie_coefficient = 0.003f;
    sky.mie_directional_g = 0.9f;
    renderer.tone_mapping = AGX_TONE_MAPPING;
    renderer.tone_mapping_exposure = 2.4f;
    scene.add(sky);
    scene.add(drie::hemisphere_light(0xbfcbe0, 0x5a4a38, 0.12));
    sun = drie::directional_light(0xffe2b8, 8.0);
    sun.target.position = { 0, CEILING / 2, 0 };
    sun.position = sun.target.position + sky.sun_position.normalize() * 20;
    sun.shadow.far = 45;
    sun.shadow.radius = 1.5f;
    sun.shadow.softness = 0.03f;
    sun.shadow.contact_length = 0.25f;
    sun.cast_shadow = true;
    sun.shadow.map_size = 4096;
    sun.shadow.cascades = 1;
    sun.shadow.normal_bias = 0.002f;
    sun.shadow_extent = 10;
    scene.add(sun);
    scene.add(sun.target);

    Opening[2] side_windows = { { -2.6f, -1.6f, 1.3f, 3.9f }, { 0.9f, 1.9f, 1.3f, 3.9f } };
    Opening[2] back_holes = { { -3.4f, -2.0f, 0, 2.6f }, { 2.0f, 3.0f, 1.5f, 3.9f } };
    MeshStandardMaterial* stone_look = surface("old_stone_wall", 2, 0.06f);
    cell = 0.025f;
    Mesh* stone = solid(hall(&side_windows, &back_holes), stone_look);
    stone.cast_shadow = false;
    cell = 0.06f;
    Mesh* stone_shadow = solid(hall(&side_windows, &back_holes), stone_look);
    stone_shadow.shadow_only = true;

    MeshStandardMaterial* pebbles = surface("multicolored_pebble_floor", 2, 0.025f);
    cell = 0.025f;
    Mesh* floor = solid(ground(), pebbles);
    floor.cast_shadow = false;
    cell = 0.06f;
    Mesh* floor_shadow = solid(ground(), pebbles);
    floor_shadow.shadow_only = true;
    io::printfn("room: %d vertices, its shadow %d", (stone.geometry.vertices.len() + floor.geometry.vertices.len()) / drie::FLOATS_PER_VERTEX,
                (stone_shadow.geometry.vertices.len() + floor_shadow.geometry.vertices.len()) / drie::FLOATS_PER_VERTEX);

    cell = 1e9;
    BufferGeometry* metal = drie::buffer_geometry();
    foreach (o : side_windows) lattice(metal, 0, 1, o);
    lattice(metal, 1, -1, back_holes[1]);
    MeshStandardMaterial* rust = surface("rusty_metal_04", 0.5f, 0);
    rust.metalness = 1;
    solid(metal, rust);

    make_cloth();
    cloth.add_plane({ ROOM, 0, 0 }, { -1, 0, 0 });
    cloth.add_plane({ -ROOM, 0, 0 }, { 1, 0, 0 });
    cloth.add_plane({ 0, 0, ROOM }, { 0, 0, -1 });
    cloth.add_plane({ 0, 0, -ROOM }, { 0, 0, 1 });

    scene.update_matrix_world();
    if (!env::EMSCRIPTEN) {
        RadianceCascades* gi = drie::radiance_cascades({ -ROOM - THICK, -0.3f, -ROOM - THICK }, { ROOM + THICK, CEILING + 0.4f, ROOM + THICK }, spacing: 0.25f, levels: 4);
        gi.update_every = 1;
        gi.blend = 0.12f;
        gi.voxelize(scene);
        gi.update_every = 4;
        scene.radiance = gi;
    }

    camera = drie::perspective_camera(55, (float)W / H, 0.05f, 2000);
    camera.position = START_EYE;
    camera.look_at_vec(START_AT);
}

String off = "";
DirectionalLight* sun;

fn void render_frame(RenderTarget* target) {
    int w = math::max((int)((target ? target.width : sapp::width()) * RENDER_SCALE), 1);
    int h = math::max((int)((target ? target.height : sapp::height()) * RENDER_SCALE), 1);
    if (!scene_rt) {
        scene_rt = drie::render_target(w, h, 1, RGBA16F, mrt: true);
        taa = postprocessing::taa_pass();
        bloom = postprocessing::unreal_bloom_pass(strength: 0.14f, radius: 0.7f, threshold: 0);
        flare = postprocessing::lens_flare_pass(strength: 0.05f, threshold: 2);
        flare.dispersion = 0.003f;
        ao = postprocessing::gtao_pass(camera);
        ao.radius = 0.3f;
        ao.fade_start = 20;
        ao.fade_end = 40;
        output = postprocessing::output_pass(renderer);
        ColorGrade* grade = postprocessing::color_grade();
        grade.temperature = 18;
        grade.tint = 3;
        grade.contrast = 1.12f;
        grade.saturation = 1.08f;
        grade.lift = { -0.05f, -0.03f, 0.02f };
        grade.shadows = { -0.012f, -0.018f };
        grade.highlights = { 0.012f, 0.03f };
        output.grade = grade;
    } else if (scene_rt.width != w || scene_rt.height != h) {
        scene_rt.set_size(w, h);
    }
    if (camera.aspect != (float)w / h) {
        camera.aspect = (float)w / h;
        camera.update_projection_matrix();
    }
    sky.follow(camera.get_world_position(), camera.far);
    if (!off.contains("rc") && !env::EMSCRIPTEN) scene.radiance.update(scene);
    taa.jitter(camera);
    renderer.set_render_target(scene_rt);
    renderer.render(scene, camera);
    if (!off.contains("ao")) ao.render(scene_rt);
    if (!off.contains("bloom")) bloom.render(scene_rt);
    if (!off.contains("flare")) flare.render(scene_rt, bloom.mips[3]);
    taa.render(scene_rt);
    output.set_source(taa.texture());
    output.render(target);
    if (!target) drie::commit();
}

float last_time = -1;

fn void on_event(Event* e) {
    if (e.type == KEY_DOWN && e.key_code == P) {
        Vector3 at = camera.position + controls.direction();
        io::printfn("camera { %.2f, %.2f, %.2f } looking at { %.2f, %.2f, %.2f }", camera.position.x, camera.position.y,
                    camera.position.z, at.x, at.y, at.z);
    }
}

fn void animate(double time) {
    float now = renderer.time;
    float dt = last_time < 0 ? 0 : math::min(now - last_time, 0.1f);
    last_time = now;
    controls.update(dt);
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
    sdtx::color3f(0.9f, 0.88f, 0.82f);
    int falling;
    int dormant;
    foreach (h : handkerchiefs) {
        if (h.state == FALLING || h.state == HELD) falling++;
        if (h.state == DORMANT) dormant++;
    }
    @pool() {
        sdtx::puts(string::tformat("dropped %d   moving %d   dormant %d   (%d handkerchiefs)\n", dropped, falling, dormant, HANDKERCHIEFS).zstr_tcopy());
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
            case "--off": if (i + 1 < args.len) off = args[++i];
            case "--time": if (i + 1 < args.len) at = (float)args[++i].to_double()!!;
            case "--gpu-time": gpu_time = true;
            default: io::eprintfn("unknown option %s", args[i]);
        }
    }
    if (!png_path) {
        renderer = drie::renderer(antialias: true, width: W, height: H, title: "drie: handkerchiefs");
        build_scene();
        controls = controls::walk_controls(camera, START_AT);
        controls.bounded = true;
        controls.min = { -ROOM + 0.25f, 0.25f, -ROOM + 0.25f };
        controls.max = { ROOM - 0.25f, CEILING - 0.25f, ROOM - 0.25f };
        drie::add_overlay(&hud);
        drie::add_event_listener(&on_event);
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
    foreach (h : handkerchiefs) {
        if (h.state == FALLING || h.state == HELD) falling++;
        if (h.state == DORMANT) dormant++;
    }
    io::printfn("%.1f s in %.0f ms: dropped %d, moving %d, dormant %d", at, stm::ms(stm::since(began)), dropped, falling, dormant);
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    for (int f = 0; f < 60; f++) {
        render_frame(rt);
        drie::commit();
    }
    if (off.contains("fog")) scene.fog = null;
    if (off.contains("shadow")) sun.cast_shadow = false;
    if (gpu_time) {
        gpu_timer::reset();
        profile::enabled = false;
        gpu_timer::enabled = false;
        ulong t0 = stm::now();
        for (int f = 0; f < 120; f++) {
            ulong t = profile::start();
            if (!off.contains("cloth")) step(dt);
            profile::stop("step", t);
            render_frame(rt);
            drie::commit();
        }
        io::printfn("cpu %.2f ms a frame", stm::ms(stm::since(t0)) / 120);
        profile::report(120);
        gpu_timer::poll();
        gpu_timer::report();
    }
    char[] px = mem::new_array(char, (usz)W * H * 4);
    readback::read_image(rt.texture_image, W, H, px.ptr);
    if (!png::write(png_path, px, W, H)) {
        io::eprintfn("could not write %s", png_path);
        return 1;
    }
    io::printfn("saved %s", png_path);
    return 0;
}
