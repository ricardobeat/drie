// A dust sheet thrown over the furniture: drie::gpu_cloth, 200 x 200 points solved on the GPU,
// dropped over a sofa and a small table (boxes, capsules and a sphere as colliders) onto the
// floor, where it settles in folds that lie on each other (self-collision) instead of passing
// through. The points never leave the GPU: the kernels write the sheet's vertex buffer.
//
//   just example cloth_furniture                   in a window (drag to orbit, R drops it again,
//                                                  B switches the bend between DISTANCE and SIGNED)
//   just example cloth_furniture --png out.png     one frame, headless, at --time t (default 6 s),
//                                                  --bend signed|distance, --bend-compliance c
//                                                  (after --bend)
//
// Headless, --hash prints a hash of the points at that time (two runs give the same one) and
// --gpu-time the GPU time of the cloth's pass per frame.
module cloth_furniture_example;
import drie, drie::gpu_cloth, drie::ui, drie::controls, drie::readback, drie::png, drie::gpu_timer, sokol;
import std::io, std::math;

const float SIGNED_BEND = 20;
const int W = 1280;
const int H = 720;
const int COLS = 200;
const int ROWS = 200;
const float SIZE = 3.2f;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
OrbitControls* controls;
GpuCloth* sheet;
double last_time = -1;

Material* fabric;
Material* wood;

// ---- the furniture: each piece a collider and a mesh where it is

fn float friction_of(Material* m) => m == fabric ? 0.8f : m == wood ? 0.35f : 0.2f;

fn void box(Vector3 center, Vector3 half, Material* m) {
    sheet.set_collider_friction(sheet.add_box(half, drie::mat4_compose(center, { 0, 0, 0 }, { 1, 1, 1 })), friction_of(m));
    Mesh* mesh = drie::mesh(drie::box_geometry(half.x * 2, half.y * 2, half.z * 2), m);
    mesh.position = center;
    mesh.cast_shadow = true;
    mesh.receive_shadow = true;
    scene.add(mesh);
}

// A capsule from a to b, along one of the axes.
fn void capsule(Vector3 a, Vector3 b, float radius, Material* m) {
    sheet.set_collider_friction(sheet.add_capsule(a, b, radius), friction_of(m));
    Vector3 d = b - a;
    Mesh* body = drie::mesh(drie::cylinder_geometry(radius, radius, d.length(), 24), m);
    body.position = (a + b) * 0.5f;
    if (math::abs(d.x) > math::abs(d.y) && math::abs(d.x) > math::abs(d.z)) body.rotation.z = (float)math::PI / 2;
    if (math::abs(d.z) > math::abs(d.y) && math::abs(d.z) > math::abs(d.x)) body.rotation.x = (float)math::PI / 2;
    foreach (end : (Vector3[2]) { a, b }) {
        Mesh* cap = drie::mesh(drie::sphere_geometry(radius, 24, 12), m);
        cap.position = end;
        cap.cast_shadow = true;
        cap.receive_shadow = true;
        scene.add(cap);
    }
    body.cast_shadow = true;
    body.receive_shadow = true;
    scene.add(body);
}

fn void ball(Vector3 center, float radius, Material* m) {
    sheet.set_collider_friction(sheet.add_sphere(center, radius), friction_of(m));
    Mesh* mesh = drie::mesh(drie::sphere_geometry(radius, 32, 16), m);
    mesh.position = center;
    mesh.cast_shadow = true;
    mesh.receive_shadow = true;
    scene.add(mesh);
}

fn void furniture() {
    // a two-seat sofa against the back of the room
    float sx = -0.6f;
    float sofa_z = -0.65f;
    box({ sx, 0.22f, sofa_z }, { 0.65f, 0.22f, 0.45f }, fabric);
    box({ sx, 0.51f, sofa_z + 0.05f }, { 0.6f, 0.07f, 0.38f }, fabric);
    box({ sx, 0.68f, sofa_z - 0.34f }, { 0.65f, 0.34f, 0.11f }, fabric);
    capsule({ sx - 0.6f, 1.0f, sofa_z - 0.34f }, { sx + 0.6f, 1.0f, sofa_z - 0.34f }, 0.12f, fabric);
    foreach (side : (float[2]) { -1, 1 }) {
        float ax = sx + side * 0.75f;
        box({ ax, 0.3f, sofa_z }, { 0.1f, 0.3f, 0.45f }, fabric);
        capsule({ ax, 0.66f, sofa_z - 0.38f }, { ax, 0.66f, sofa_z + 0.36f }, 0.13f, fabric);
    }
    // a small table with a vase on it
    float tx = 0.85f;
    float tz = 0.55f;
    box({ tx, 0.56f, tz }, { 0.36f, 0.025f, 0.36f }, wood);
    foreach (corner : (float[2][4]) { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } }) {
        capsule({ tx + corner[0] * 0.3f, 0.03f, tz + corner[1] * 0.3f }, { tx + corner[0] * 0.3f, 0.52f, tz + corner[1] * 0.3f }, 0.025f, wood);
    }
    ball({ tx - 0.08f, 0.71f, tz + 0.05f }, 0.12f, drie::mesh_standard_material(color: 0x4a6c8c, roughness: 0.3));
}

// The sheet, flat and a little turned and tilted, above everything.
BendModel bend = DISTANCE;
float distance_bend = 0.05f;
float signed_bend = SIGNED_BEND;

fn void apply_bend() {
    sheet.bend_model = bend;
    sheet.bend_compliance = bend == SIGNED ? signed_bend : distance_bend;
}

uint drop_seed = 1;

fn void drop() => sheet.reset_scattered(drop_seed);

fn void make_sheet() {
    float turn = 0.1f;
    Vector3 across = { math::cos(turn) * SIZE, 0, -math::sin(turn) * SIZE };
    Vector3 down = { math::sin(turn) * SIZE, -0.25f, math::cos(turn) * SIZE };
    Vector3 center = { 0.15f, 1.75f, 0.1f };
    sheet = gpu_cloth::grid(COLS, ROWS, center - across * 0.5f - down * 0.5f, across, down, 1.5f);
    float spacing = SIZE / (COLS - 1);
    sheet.substeps = 12;
    sheet.floor_y = 0;
    sheet.thickness = 0.012f;
    sheet.friction = 0.5f;
    sheet.self_distance = spacing;
    sheet.self_every = 2;
    sheet.stretch_compliance = 2e-5f;
    sheet.shear_compliance = 2e-4f;
    apply_bend();
    sheet.damping = 0.4f;
    sheet.drag = 0.4f;
}

fn void build_scene() {
    scene = drie::scene();
    scene.background = 0x20242a;
    fabric = drie::mesh_standard_material(color: 0x3d5a46, roughness: 0.9);
    wood = drie::mesh_standard_material(color: 0x7a5232, roughness: 0.55);

    scene.add(drie::hemisphere_light(0xdfe8ff, 0x4a3c30, 0.45));
    DirectionalLight* sun = drie::directional_light(0xfff1dc, 2.8);
    sun.position = { 5, 4, 1.2f };
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow_extent = 4;
    scene.add(sun);
    PointLight* lamp = drie::point_light(0xffc890, 5, 9);
    lamp.position = { -1.2f, 2.7f, 2.2f };
    scene.add(lamp);

    Mesh* floor = drie::mesh(drie::plane_geometry(10, 10), drie::mesh_standard_material(color: 0x8c7660, roughness: 0.75));
    floor.rotation.x = -(float)math::PI / 2;
    floor.receive_shadow = true;
    scene.add(floor);
    Material* plaster = drie::mesh_standard_material(color: 0xc8c2b8, roughness: 0.95);
    Mesh* back = drie::mesh(drie::plane_geometry(10, 4), plaster);
    back.position = { 0, 2, -2.3f };
    back.receive_shadow = true;
    scene.add(back);
    Mesh* left = drie::mesh(drie::plane_geometry(10, 4), plaster);
    left.position = { -3.2f, 2, 0 };
    left.rotation.y = (float)math::PI / 2;
    left.receive_shadow = true;
    scene.add(left);

    make_sheet();
    drop();
    furniture();
    Material* linen = drie::mesh_standard_material(color: 0xe6dfd0, roughness: 0.85, side: DOUBLE_SIDE);
    Mesh* m = sheet.make_mesh(linen);
    m.cast_shadow = true;
    m.receive_shadow = true;
    scene.add(m);

    sheet.watch_points();
    seen = mem::new_array(Vector3, sheet.count());
    Material* brass = drie::mesh_standard_material(color: 0xc8a050, roughness: 0.35, metalness: 1);
    foreach (i, &h : handles) {
        h.point = sheet.index(i % 2 == 0 ? 0 : COLS - 1, i < 2 ? 0 : ROWS - 1);
        h.mesh = drie::mesh(drie::sphere_geometry(HANDLE_RADIUS, 20, 10), brass);
        h.mesh.cast_shadow = true;
        h.mesh.position = sheet.origin + sheet.rest[h.point];
        scene.add(h.mesh);
    }

    camera = drie::perspective_camera(45, (float)W / H, 0.1f, 100);
    camera.position = { 2.9f, 2.4f, 3.6f };
    camera.look_at_vec({ 0.05f, 0.45f, -0.2f });
}

const float HANDLE_RADIUS = 0.04f;

struct Handle {
    uint point;
    Mesh* mesh;
}

Handle[4] handles;
Vector3[] seen;
int grabbed = -1;
Vector3 grab_normal;
Vector3 grab_at;
Vector3 grab_target;
Vector3 grab_pos;
Vector3 grab_vel;
float slack;

fn Ray mouse_ray(float x, float y) {
    Raycaster r = drie::raycaster();
    r.set_from_camera({ 2 * x / sapp::widthf() - 1, 1 - 2 * y / sapp::heightf() }, camera);
    return r.ray;
}

fn int pick_handle(float x, float y) {
    Ray ray = mouse_ray(x, y);
    int best = -1;
    float best_t = 1e30f;
    foreach (i, h : handles) {
        Vector3 c = h.mesh.position;
        float t = (c - ray.origin).dot(ray.direction);
        if (t < 0) continue;
        if ((ray.at(t) - c).length() < HANDLE_RADIUS * 2.5f && t < best_t) {
            best_t = t;
            best = (int)i;
        }
    }
    return best;
}

fn void drag_to(float x, float y) {
    Ray ray = mouse_ray(x, y);
    float den = ray.direction.dot(grab_normal);
    if (math::abs(den) < 1e-5f) return;
    float t = (grab_at - ray.origin).dot(grab_normal) / den;
    if (t < 0) return;
    Vector3 to = ray.at(t);
    to.y = math::max(to.y, 0.02f);
    grab_target = to;
}

const float GRAB_STIFFNESS = 120;
const float GRAB_DAMPING = 18;
const float GRAB_MAX_SPEED = 3;

fn void follow_mouse(float dt) {
    if (grabbed < 0) return;
    Vector3 accel = (grab_target - grab_pos) * GRAB_STIFFNESS - grab_vel * GRAB_DAMPING;
    grab_vel += accel * dt;
    float speed = grab_vel.length();
    if (speed > GRAB_MAX_SPEED) grab_vel = grab_vel * (GRAB_MAX_SPEED / speed);
    grab_pos += grab_vel * dt;
    grab_pos.y = math::max(grab_pos.y, 0.02f);
    Handle* h = &handles[grabbed];
    sheet.move_pin(h.point, grab_pos);
    h.mesh.position = grab_pos;
}

fn void release() {
    if (grabbed < 0) return;
    sheet.unpin(handles[grabbed].point);
    sheet.tether_slack = slack;
    grabbed = -1;
    controls.enabled = true;
}

fn void on_mouse(Event* e) {
    switch (e.type) {
        case MOUSE_DOWN:
            if (e.mouse_button != LEFT) return;
            int i = pick_handle(e.mouse_x, e.mouse_y);
            if (i < 0) return;
            grabbed = i;
            controls.enabled = false;
            grab_at = handles[i].mesh.position;
            grab_normal = (camera.position - grab_at).normalize();
            grab_target = grab_at;
            grab_pos = grab_at;
            grab_vel = { 0, 0, 0 };
            slack = sheet.tether_slack;
            sheet.tether_slack = slack + 0.05f;
            sheet.pin(handles[i].point);
            sheet.move_pin(handles[i].point, grab_at);
        case MOUSE_MOVE:
            if (grabbed >= 0) drag_to(e.mouse_x, e.mouse_y);
        case MOUSE_UP:
            if (e.mouse_button == LEFT) release();
        default:
    }
}

fn void animate(double time) {
    double t = (double)renderer.time;
    float dt = last_time < 0 ? 1.0f / 60 : (float)(t - last_time);
    last_time = t;
    follow_mouse(math::min(dt, 0.1f));
    sheet.update(math::min(dt, 0.1f));
    if (sheet.seen_points(seen)) {
        foreach (i, h : handles) {
            if ((int)i != grabbed) h.mesh.position = seen[h.point];
        }
    }
    controls.update();
    renderer.render(scene, camera);
}

fn void on_key(Event* e) {
    on_mouse(e);
    if (e.type != KEY_DOWN || e.key_repeat) return;
    if (e.key_code == R) {
        release();
        drop_seed++;
        drop();
    }
    if (e.key_code == B) {
        bend = bend == SIGNED ? DISTANCE : SIGNED;
        apply_bend();
        release();
        drop_seed++;
        drop();
    }
}

fn void hud() {
    ui::setup();
    float scale = math::max(1.0f, math::round(sapp::heightf() / 540));
    sdtx::set_context(sdtx::default_context());
    sdtx::canvas(sapp::widthf() / scale, sapp::heightf() / scale);
    sdtx::origin(1, 1);
    sdtx::font(ui::FONT_KC853);
    sdtx::color3f(0.9f, 0.9f, 0.9f);
    @pool() {
        sdtx::puts(string::tformat("gpu cloth: %d points, %d links in %d batches, %d substeps, %.1f s\n",
            sheet.count(), sheet.link_count(), sheet.batch_count(), sheet.substeps, sheet.time).zstr_tcopy());
        if (gpu_timer::enabled) {
            sdtx::puts(string::tformat("cloth pass %.2f ms on the GPU\n", gpu_timer::last_ms("drie-gpu-cloth")).zstr_tcopy());
        }
        sdtx::puts(string::tformat("bend: %s (compliance %g)\n", bend == SIGNED ? "SIGNED" : "DISTANCE", sheet.bend_compliance).zstr_tcopy());
        sdtx::puts("drag a corner's handle to pull the sheet   R: drop it again   B: switch the bend and drop it again\n");
    };
    sdtx::draw();
}

fn int main(String[] args) {
    String png_path;
    float at = 6;
    bool hash;
    bool gpu_time;
    for (usz i = 1; i < args.len; i++) {
        switch (args[i]) {
            case "--png": if (i + 1 < args.len) png_path = args[++i];
            case "--time": if (i + 1 < args.len) at = (float)args[++i].to_double()!!;
            case "--bend": if (i + 1 < args.len) bend = args[++i] == "signed" ? SIGNED : DISTANCE;
            case "--bend-compliance":
                if (i + 1 < args.len) {
                    float v = (float)args[++i].to_double()!!;
                    if (bend == SIGNED) {
                        signed_bend = v;
                    } else {
                        distance_bend = v;
                    }
                }
            case "--hash": hash = true;
            case "--gpu-time": gpu_time = true;
            default: io::eprintfn("unknown option %s", args[i]);
        }
    }
    if (!png_path && !hash) {
        renderer = drie::renderer(antialias: true, width: W, height: H, title: "drie: a GPU cloth over the furniture");
        build_scene();
        controls = controls::orbit_controls(camera);
        controls.target = { 0.05f, 0.45f, -0.2f };
        drie::add_event_listener(&on_key);
        drie::add_overlay(&hud);
        renderer.set_animation_loop(&animate);
        return 0;
    }
    renderer = drie::renderer(antialias: true, width: W, height: H);
    renderer.init_headless();
    renderer.auto_time = false;
    build_scene();
    io::printfn("%d points, %d links in %d batches", sheet.count(), sheet.link_count(), sheet.batch_count());
    if (gpu_time) gpu_timer::enabled = true;
    // the sheet's frames up to `at`, each committed as a frame of its own
    int frames = (int)math::round(at * 60);
    ulong began = stm::now();
    for (int f = 0; f < frames; f++) {
        sheet.update(1.0f / 60);
        drie::commit();
        if (gpu_time && f == frames - 120) gpu_timer::reset();
    }
    io::printfn("simulated %.2f s (%d frames) in %.0f ms of wall time", at, frames, stm::ms(stm::since(began)));
    if (sheet.read_points(seen)) {
        foreach (i, h : handles) {
            if ((int)i != grabbed) h.mesh.position = seen[h.point];
        }
    }
    if (gpu_time) {
        gpu_timer::poll();
        gpu_timer::report();
    }
    if (hash) {
        Vector3[] points = mem::new_array(Vector3, sheet.count());
        if (!sheet.read_points(points)) {
            io::eprintfn("could not read the points back");
            return 1;
        }
        // FNV-1a over the floats' bits
        ulong h = 0xcbf29ce484222325;
        foreach (p : points) {
            for (int k = 0; k < 3; k++) {
                h ^= (ulong)bitcast(p[k], uint);
                h *= 0x100000001b3;
            }
        }
        float lowest = 1e30f;
        float highest = -1e30f;
        foreach (p : points) {
            lowest = math::min(lowest, p.y);
            highest = math::max(highest, p.y);
        }
        io::printfn("points hash %016x (y from %.3f to %.3f)", h, lowest, highest);
    }
    if (!png_path) return 0;
    renderer.time = at;
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
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
