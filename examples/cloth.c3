// Cloth (drie::cloth) to throw balls at: three velvet panels side by side, dark blue, dark
// red and white, laid flat in the air and held along one edge, swing down and hang from it over
// a mirror-like checkerboard platform that runs to the horizon under a sunny sky. Each ball
// flies from the camera (through the point clicked, or along the view), falls, bounces on the
// floor and is a sphere collider with a mass on each panel, which wraps round it and catches it
// or throws it back. At most MAX_BALLS are in play: a new one takes the place of the oldest.
// The floor mirrors the cloth, the balls and the sky with a planar Reflector; the sun casts
// their shadows on it. The velvet's textures are in examples/textures.
//
//   just example cloth                  in a window (drag to orbit, scroll to zoom)
//   just example cloth --png out.png    one frame, headless (--time t): two balls thrown, at
//                                       2.6 s and 3.0 s, the scene stepped up to t
//
// Controls:
//   click (without dragging)            throw a ball through the point clicked
//   space                               throw a ball along the view
//   R                                   take the balls away and lay the cloth flat again
//   C                                   let go of the panels: they drop to the ground
module cloth_example;
import drie, drie::cloth, drie::objects, drie::ui, drie::controls, drie::readback, drie::png, sokol;
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
const float BEND = 20.0f;
const float INTERNAL = 0.5f;
const float SUBSTEPS = 16;
// of its speed into the floor, what a ball keeps going up again
const float BOUNCE = 0.6f;
// a ball's place while it is not in play, away from the cloth
const int PANELS = 3;
// the panels' colours: dark blue, dark red, white
const Color[PANELS] PANEL_COLORS = { 0x1c2c5a, 0x5a1420, 0xe8e4dc };
// how far ahead a ball's flight is followed (s)
const float TRACE_SECONDS = 4;
const Vector3 PARKED = { 0, -100, 0 };

struct Ball {
    Mesh* mesh;
    // its sphere on each panel
    int[PANELS] collider;
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
Cloth*[PANELS] sheets;
MeshStandardMaterial*[PANELS] velvets;
Ball[MAX_BALLS] balls;
int next_ball;
int thrown;
// how many times the panels have been let go (to vary their sway)
int drops;
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

// Lets go of every panel's pins, each panel given a slow uneven sway as it goes (a sheet let go
// perfectly still folds into even pleats; real cloth never hangs that still).
fn void drop_cloth() {
    foreach (k, sheet : sheets) {
        for (uint i = 0; i < sheet.count(); i++) {
            if (sheet.inv_mass[i] == 0) sheet.unpin(i);
        }
        // a few sines across the panel, their phases and directions from its index
        float s = (float)k * 1.7f + (float)drops * 2.3f + 0.4f;
        foreach (i, &v : sheet.vel) {
            Vector3 r = sheet.rest[i];
            float a = math::sin(r.x * 2.1f + r.z * 1.3f + s);
            float b = math::sin(r.x * 0.9f - r.z * 2.6f + s * 1.9f);
            *v += (Vector3) { a * 0.4f, 0, b * 0.4f };
        }
    }
    drops++;
}

fn void hang_cloth() {
    foreach (i, &sheet : sheets) {
        if (*sheet) {
            scene.remove((*sheet).mesh);
            (*sheet).free();
        }
        // 0.9 m wide and 3 m long (a point every 3 cm), side by side 15 cm apart, laid flat
        // 3.4 m up and held along the near edge: each swings down and hangs from it. 1.1 kg:
        // 0.4 kg/m², velvet
        float x0 = panel_x0((int)i);
        Cloth* c = cloth::grid(31, 100, { x0, 3.4f, 0 }, { 0.9f, 0, 0 }, { 0, 0, -3.0f }, mass: MASS);
        c.pin_row(0);
        c.substeps = (int)SUBSTEPS;
        c.drag = DRAG;
        c.damping = DAMPING;
        c.bend_compliance = BEND;
        c.internal_damping = INTERNAL;
        c.thickness = 0.01f;
        c.floor_y = 0.01f;
        c.self_distance = 0.035f;
        c.self_every = 2;
        c.sleep_speed = 0.02f;
        if (!velvets[i]) {
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
            velvets[i] = m;
        }
        Mesh* mesh = c.make_mesh(velvets[i]);
        mesh.cast_shadow = true;
        mesh.receive_shadow = true;
        scene.add(mesh);
        *sheet = c;
    }
    foreach (&b : balls) {
        b.live = false;
        b.pos = PARKED;
        b.mesh.visible = false;
        foreach (i, sheet : sheets) {
            b.collider[i] = sheet.add_sphere(PARKED, BALL_RADIUS);
            sheet.set_collider_mass(b.collider[i], BALL_MASS);
        }
    }
    next_ball = 0;
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

    MeshStandardMaterial*[3] ball_mats = {
        drie::mesh_standard_material(color: 0xe04030, roughness: 0.65f),
        drie::mesh_standard_material(color: 0xf0c020, roughness: 0.65f),
        drie::mesh_standard_material(color: 0x30a050, roughness: 0.65f),
    };
    SphereGeometry* ball_geo = drie::sphere_geometry(BALL_RADIUS, 32, 16);
    foreach (i, &b : balls) {
        b.mesh = drie::mesh(ball_geo, ball_mats[i % 3]);
        b.mesh.cast_shadow = true;
        b.mesh.receive_shadow = true;
        b.mesh.visible = false;
        scene.add(b.mesh);
    }
    hang_cloth();

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
    b.mesh.position = b.pos;
    // put there, not swept from where it was
    foreach (i, sheet : sheets) sheet.move_collider(b.collider[i], b.pos);
    aim(b);
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

// Follows the ball's flight ahead (TRACE_SECONDS, nothing in its way) and turns its sphere on
// only on the panels it comes near: anywhere a panel can swing to from its rail. Done again
// when a panel changes its flight.
fn void aim(Ball* b) {
    bool[PANELS] near;
    Vector3 pos = b.pos;
    Vector3 vel = b.vel;
    for (int s = 0; s < (int)(TRACE_SECONDS * 60); s++) {
        foreach (i, &hit : near) {
            float x0 = panel_x0((int)i) - 0.3f;
            float x1 = panel_x0((int)i) + 1.2f;
            if (pos.x > x0 - BALL_RADIUS && pos.x < x1 + BALL_RADIUS && pos.y < 3.8f + BALL_RADIUS && pos.z > -3.4f - BALL_RADIUS && pos.z < 3.4f + BALL_RADIUS) *hit = true;
        }
        fly(&pos, &vel, 1.0f / 60);
    }
    foreach (i, sheet : sheets) sheet.set_collider_active(b.collider[i], near[i]);
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
        b.mesh.position = b.pos;
        foreach (i, sheet : sheets) sheet.move_collider(b.collider[i], b.pos);
    }
    cloth::update_all(&sheets, dt);
    // the cloth pushes back: a ball goes where the cloth left it, and loses the momentum the
    // cloth took from it
    foreach (&b : balls) {
        if (!b.live) continue;
        bool pushed;
        foreach (i, sheet : sheets) {
            Vector3 impulse = sheet.collider_impulse(b.collider[i]);
            b.pos += sheet.collider_shift(b.collider[i]);
            b.vel -= impulse / BALL_MASS;
            if (impulse != (Vector3) {}) pushed = true;
        }
        b.mesh.position = b.pos;
        if (pushed) aim(b);
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
    };
    sdtx::draw();
}

fn int main(String[] args) {
    String png_path;
    float at = 3.3f;
    for (usz i = 1; i + 1 < args.len; i += 2) {
        switch (args[i]) {
            case "--png": png_path = args[i + 1];
            case "--time": at = (float)args[i + 1].to_double()!!;
            default: io::eprintfn("unknown option %s", args[i]);
        }
    }
    if (!png_path) {
        renderer = drie::renderer(antialias: true, width: W, height: H, title: "drie: cloth");
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
    // two balls from beside the camera into the cloth, the scene stepped at 60 Hz up to `at`
    const float DT = 1.0f / 60;
    bool first, second;
    for (float t = 0; t < at; t += DT) {
        if (!first && t >= 2.6f) {
            first = true;
            throw_ball({ -1.5f, 1.8f, 5 }, ((Vector3) { -0.3f, 1.7f, 0 } - (Vector3) { -1.5f, 1.8f, 5 }).normalize());
        }
        if (!second && t >= 3.0f) {
            second = true;
            throw_ball({ 2.5f, 2.0f, 5 }, ((Vector3) { 0.5f, 1.2f, 0 } - (Vector3) { 2.5f, 2.0f, 5 }).normalize());
        }
        step(DT);
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
