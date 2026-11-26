// examples/cloth on the GPU (drie::gpu_cloth): balls thrown at three velvet panels side by side,
// dark blue, dark red and white, laid flat in the air and held along one edge, that swing down
// and hang from it over a mirror-like checkerboard platform that runs to the horizon under a
// sunny sky. The panels are one GpuCloth (gpu_cloth::panels): solved together, they keep apart
// from each other as each keeps its own folds apart (self-collision). Each ball flies from the
// camera (through the point clicked, or along the view), falls, bounces on the floor and is a
// sphere collider with a mass: the cloth pushes it back as it pushes the cloth, at every
// substep on the GPU, and the CPU hears a frame or two later how far and how much slower
// (collider_shift, collider_impulse). The balls are drawn where the GPU has them. At most
// MAX_BALLS are in play: a new one takes the place of the oldest. The velvet's textures are in
// examples/textures.
//
//   just example gpu_cloth_balls                  in a window (drag to orbit, scroll to zoom)
//   just example gpu_cloth_balls --png out.png    one frame, headless (--time t): two balls
//                                                 thrown, at 2.6 s and 3.0 s, the scene
//                                                 stepped up to t in frames of 1/60 s (--fps n:
//                                                 of 1/n s; --gpu-time: the cloth's GPU time
//                                                 per frame over the last 60 frames)
//
// Controls:
//   click (without dragging)            throw a ball through the point clicked
//   space                               throw a ball along the view
//   R                                   take the balls away and lay the cloth flat again
//   C                                   let go of the panels: they drop to the ground
module gpu_cloth_balls_example;
import drie, drie::gpu_cloth, drie::objects, drie::ui, drie::controls, drie::readback, drie::png, drie::gpu_timer, sokol;
import std::io, std::math;

const int W = 1280;
const int H = 720;
const int MAX_BALLS = 8;
const float BALL_RADIUS = 0.15f;
const float THROW_SPEED = 15.5f;
// kg: a tennis ball is 0.06, a softball 0.19
const float BALL_MASS = 0.25f;
const float GRAVITY = 9.81f;
const float MASS = 1.1f;
const float DRAG = 0.15f;
const float DAMPING = 0.8f;
// the distance bends' compliance (m/N)
const float BEND = 0.05f;
const float INTERNAL = 0.5f;
const int SUBSTEPS = 16;
// of its speed into the floor, what a ball keeps going up again
const float BOUNCE = 0.6f;
const int PANELS = 3;
// the panels' colours: dark blue, dark red, white
const Color[PANELS] PANEL_COLORS = { 0x1c2c5a, 0x5a1420, 0xe8e4dc };
// a ball's place while it is not in play, away from the cloth
const Vector3 PARKED = { 0, -100, 0 };

struct Ball {
    Mesh* mesh;
    int collider;
    Vector3 pos;
    Vector3 vel;
    bool live;
}

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
OrbitControls* controls;
Sky* sky;
Reflector* mirror;
GpuCloth* sheet;
Ball[MAX_BALLS] balls;
int next_ball;
int thrown;
// the mouse travel since the button went down: a drag orbits, a click throws
float drag_travel;

fn Texture* checker_texture() {
    const int N = 64;
    char[] px = mem::new_array(char, N * N * 4);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            char v = ((x / 32) + (y / 32)) % 2 == 0 ? 150 : 10;
            usz i = (usz)(y * N + x) * 4;
            px[i] = v;
            px[i + 1] = v;
            px[i + 2] = v;
            px[i + 3] = 255;
        }
    }
    Texture* t = drie::data_texture(px, N, N);
    t.wrap_s = REPEAT_WRAPPING;
    t.wrap_t = REPEAT_WRAPPING;
    t.generate_mipmaps = true;
    // a 1 m square of each color
    t.repeat = { 500, 500 };
    return t;
}

// The left edge of panel `i`; the panels are 0.9 m wide, 15 cm apart.
fn float panel_x0(int i) => -0.45f - (float)(PANELS - 1) * 0.525f + (float)i * 1.05f;

fn void make_cloth() {
    // 0.9 m wide and 3 m long (a point every 3 cm), side by side 15 cm apart, laid flat 3.4 m
    // up and held along the near edge: each swings down and hangs from it. 1.1 kg each: 0.4
    // kg/m², velvet
    sheet = gpu_cloth::panels(PANELS, 31, 100, { panel_x0(0), 3.4f, 0 }, { 0.9f, 0, 0 }, { 0, 0, -3.0f }, { 1.05f, 0, 0 }, mass: MASS);
    sheet.pin_row(0);
    sheet.substeps = SUBSTEPS;
    sheet.drag = DRAG;
    sheet.damping = DAMPING;
    sheet.bend_model = DISTANCE;
    sheet.bend_compliance = BEND;
    sheet.internal_damping = INTERNAL;
    sheet.thickness = 0.01f;
    sheet.floor_y = 0.01f;
    sheet.self_distance = 0.035f;
    sheet.self_every = 2;
    for (int i = 0; i < PANELS; i++) {
        // velvet, as the house's light trap: its pile in the normal map, a sheen along it
        MeshStandardMaterial* m = drie::mesh_standard_material(color: PANEL_COLORS[i], roughness: 1, side: DOUBLE_SIDE);
        m.map = drie::load_texture("examples/textures/velvet_grey_diff.jpg", flip_y: false);
        m.normal_map = drie::load_texture("examples/textures/velvet_nor.jpg", NO_COLOR_SPACE, flip_y: false);
        // (flip_y off: the green flipped back)
        m.normal_scale = { 0.6f, -0.6f };
        m.specular_intensity = 0.12f;
        m.sheen = 1;
        Vector3 col = drie::color_to_linear(PANEL_COLORS[i]);
        m.sheen_color = col * 0.5f + (Vector3) { 0.1f, 0.1f, 0.1f };
        m.sheen_roughness = 0.45f;
        Mesh* mesh = sheet.make_mesh(m, panel: i);
        mesh.cast_shadow = true;
        mesh.receive_shadow = true;
        scene.add(mesh);
    }

    MeshStandardMaterial*[3] ball_mats = {
        drie::mesh_standard_material(color: 0xe04030, roughness: 0.65f),
        drie::mesh_standard_material(color: 0xf0c020, roughness: 0.65f),
        drie::mesh_standard_material(color: 0x30a050, roughness: 0.65f),
    };
    SphereGeometry* ball_geo = drie::sphere_geometry(BALL_RADIUS, 32, 16);
    foreach (i, &b : balls) {
        b.collider = sheet.add_sphere(PARKED, BALL_RADIUS);
        sheet.set_collider_mass(b.collider, BALL_MASS);
        b.mesh = sheet.make_collider_mesh(b.collider, ball_geo, ball_mats[i % 3]);
        b.mesh.cast_shadow = true;
        b.mesh.receive_shadow = true;
        scene.add(b.mesh);
    }
    park_balls();
}

fn void park_balls() {
    foreach (&b : balls) {
        b.live = false;
        b.pos = PARKED;
        b.vel = { 0, 0, 0 };
        b.mesh.visible = false;
        sheet.place_collider(b.collider, PARKED);
    }
    next_ball = 0;
}

// The cloth laid flat again and held along its near edge, the balls taken away.
fn void hang_cloth() {
    sheet.pin_row(0);
    sheet.reset();
    park_balls();
}

// Lets go of every panel's pins.
fn void drop_cloth() {
    for (uint i = 0; i < sheet.count(); i++) {
        if (sheet.inv_mass[i] == 0) sheet.unpin(i);
    }
}

fn void build_scene() {
    scene = drie::scene();
    scene.background = 0xbcd4ec;
    scene.fog = drie::fog(0xc4d8ee, 60, 400);

    sky = objects::sky();
    sky.sun_position = { -0.5f, 0.3f, 0.8f };
    sky.cloud_coverage = 0.25f;
    // a clear sky, its sun's glow kept tight round the disc
    sky.turbidity = 1.5f;
    sky.rayleigh = 1.2f;
    sky.mie_coefficient = 0.002f;
    sky.mie_directional_g = 0.9f;
    // the sky is in physical units: tone mapped, the sun and the bright horizon don't clip
    renderer.tone_mapping = AGX_TONE_MAPPING;
    renderer.tone_mapping_exposure = 0.75f;
    scene.add(sky);
    scene.add(drie::hemisphere_light(0xcfe0ff, 0x505860, 0.8));
    DirectionalLight* sun = drie::directional_light(0xfff2e0, 4.0);
    sun.position = sky.sun_position * 12;
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.normal_bias = 0.03f;
    sun.shadow_extent = 8;
    scene.add(sun);

    // the platform: far bigger than the fog lets you see, so its edge never shows
    mirror = drie::reflector({ 0, 1, 0 }, 0);
    MeshStandardMaterial* floor_mat = drie::mesh_standard_material(color: 0xffffff, roughness: 0.03f, map: checker_texture());
    floor_mat.reflector = mirror;
    Mesh* floor = drie::mesh(drie::plane_geometry(1000, 1000), floor_mat);
    floor.rotation.x = -(float)math::PI / 2;
    floor.receive_shadow = true;
    scene.add(floor);

    make_cloth();

    camera = drie::perspective_camera(50, (float)W / H, 0.1f, 2000);
    camera.position = { 2.0f, 1.7f, 6.0f };
    camera.look_at_vec({ 0, 1.5f, 0 });
}

// A ball from `from` along `dir` (normalized), in place of the oldest when all are in play.
fn void throw_ball(Vector3 from, Vector3 dir) {
    Ball* b = &balls[next_ball];
    next_ball = (next_ball + 1) % MAX_BALLS;
    b.live = true;
    b.pos = from + dir * 0.5f;
    b.vel = dir * THROW_SPEED;
    b.mesh.visible = true;
    // put there, not swept from where it was, what the cloth did to it before forgotten
    sheet.place_collider(b.collider, b.pos);
    thrown++;
}

// A ball's flight `dt` seconds on: gravity, and a bounce on the floor.
fn void fly(Vector3* pos, Vector3* vel, float dt) {
    vel.y -= GRAVITY * dt;
    *pos += *vel * dt;
    if (pos.y < BALL_RADIUS) {
        pos.y = BALL_RADIUS;
        if (vel.y < 0) vel.y = -vel.y * BOUNCE;
        // rolling on: the floor takes a little of its speed along it
        vel.x *= 0.98f;
        vel.z *= 0.98f;
    }
}

// Through the point (x, y) of the window, in pixels.
fn void throw_at(float x, float y) {
    Raycaster r = drie::raycaster();
    r.set_from_camera({ 2 * x / sapp::widthf() - 1, 1 - 2 * y / sapp::heightf() }, camera);
    throw_ball(r.ray.origin, r.ray.direction);
}

fn void throw_from_camera() {
    Vector3 dir = (controls ? controls.target : (Vector3) { 0, 1.5f, 0 }) - camera.position;
    throw_ball(camera.position, dir.normalize());
}

// Moves the balls `dt` seconds on, and the cloth with them.
fn void step(float dt) {
    foreach (&b : balls) {
        if (!b.live) continue;
        fly(&b.pos, &b.vel, dt);
        sheet.move_collider(b.collider, b.pos);
    }
    sheet.update(dt);
    // the cloth pushes back: a ball goes where the cloth left it, and loses the momentum the
    // cloth took from it (as the GPU has told so far)
    foreach (&b : balls) {
        if (!b.live) continue;
        b.pos += sheet.collider_shift(b.collider);
        b.vel -= sheet.collider_impulse(b.collider) / BALL_MASS;
    }
}

fn void render_frame() {
    sky.follow(camera.get_world_position(), camera.far);
    renderer.render_reflector(mirror, scene, camera);
    renderer.render(scene, camera);
}

float last_time = -1;

fn void animate(double time) {
    float now = renderer.time;
    float dt = last_time < 0 ? 0 : math::min(now - last_time, 0.1f);
    last_time = now;
    controls.update();
    step(dt);
    render_frame();
}

fn void on_event(Event* e) {
    switch (e.type) {
        case KEY_DOWN:
            if (e.key_repeat) return;
            if (e.key_code == SPACE) throw_from_camera();
            if (e.key_code == R) hang_cloth();
            if (e.key_code == C) drop_cloth();
        case MOUSE_DOWN:
            if (e.mouse_button == LEFT) drag_travel = 0;
        case MOUSE_MOVE:
            drag_travel += math::abs(e.mouse_dx) + math::abs(e.mouse_dy);
        case MOUSE_UP:
            if (e.mouse_button == LEFT && drag_travel < 4) throw_at(e.mouse_x, e.mouse_y);
        default:
    }
}

fn void hud() {
    ui::setup();
    float scale = math::max(1.0f, math::round(sapp::heightf() / 540));
    sdtx::set_context(sdtx::default_context());
    sdtx::canvas(sapp::widthf() / scale, sapp::heightf() / scale);
    sdtx::origin(1, 1);
    sdtx::font(ui::FONT_KC853);
    sdtx::color3f(0.1f, 0.12f, 0.16f);
    @pool() {
        sdtx::puts("click or space: throw a ball   R: hang the cloth again   C: drop it   drag: orbit\n");
        sdtx::puts(string::tformat("balls thrown: %d (at most %d in play)\n", thrown, MAX_BALLS).zstr_tcopy());
        if (gpu_timer::enabled) {
            sdtx::puts(string::tformat("cloth on the GPU: %.2f ms\n", gpu_timer::last_ms("drie-gpu-cloth")).zstr_tcopy());
        }
    };
    sdtx::draw();
}

fn int main(String[] args) {
    String png_path;
    float at = 3.3f;
    bool gpu_time;
    float fps = 60;
    for (usz i = 1; i < args.len; i++) {
        switch (args[i]) {
            case "--png": if (i + 1 < args.len) png_path = args[++i];
            case "--time": if (i + 1 < args.len) at = (float)args[++i].to_double()!!;
            case "--gpu-time": gpu_time = true;
            case "--fps": if (i + 1 < args.len) fps = (float)args[++i].to_double()!!;
            default: io::eprintfn("unknown option %s", args[i]);
        }
    }
    if (!png_path) {
        renderer = drie::renderer(antialias: true, width: W, height: H, title: "drie: GPU cloth");
        build_scene();
        controls = controls::orbit_controls(camera);
        controls.target = { 0, 1.5f, 0 };
        drie::add_event_listener(&on_event);
        drie::add_overlay(&hud);
        renderer.set_animation_loop(&animate);
        return 0;
    }
    renderer = drie::renderer(antialias: true, width: W, height: H);
    renderer.init_headless();
    renderer.auto_time = false;
    renderer.time = at;
    build_scene();
    io::printfn("%d points in %d panels, %d links in %d batches, %d substeps", sheet.count(), sheet.panel_count(), sheet.link_count(),
                sheet.batch_count(), SUBSTEPS);
    if (gpu_time) gpu_timer::enabled = true;
    // two balls from beside the camera into the cloth, the scene stepped at `fps` up to `at`
    float dt = 1.0f / fps;
    bool first, second;
    int frames = (int)math::round(at / dt);
    for (int f = 0; f < frames; f++) {
        float t = f * dt;
        if (!first && t >= 2.6f) {
            first = true;
            throw_ball({ -1.5f, 1.8f, 5 }, ((Vector3) { -0.3f, 1.7f, 0 } - (Vector3) { -1.5f, 1.8f, 5 }).normalize());
        }
        if (!second && t >= 3.0f) {
            second = true;
            throw_ball({ 2.5f, 2.0f, 5 }, ((Vector3) { 0.5f, 1.2f, 0 } - (Vector3) { 2.5f, 2.0f, 5 }).normalize());
        }
        step(dt);
        drie::commit();
        if (gpu_time && f == frames - 60) gpu_timer::reset();
    }
    if (gpu_time) {
        gpu_timer::poll();
        gpu_timer::report();
    }
    foreach (i, b : balls) {
        if (b.live) io::printfn("ball %d at (%.3f, %.3f, %.3f), going (%.2f, %.2f, %.2f) m/s", i, b.pos.x, b.pos.y, b.pos.z, b.vel.x, b.vel.y, b.vel.z);
    }
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    render_frame();
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
