// OrbitControls: left drag: rotate, right drag (or shift + left): pan, scroll: zoom.
//   Call controls.update() every frame.
// WalkControls (below): look with the mouse, walk with WASD.
//
//   import drie::controls;
//   OrbitControls* c = controls::orbit_controls(camera);
module drie::controls;
import std::math, std::collections::list;
import sokol, drie;
// Every OrbitControls made, fed the window's events by one listener.
List{OrbitControls*} all @local;

struct OrbitControls {
    Camera* camera;
    Vector3 target;
    bool enabled;
    bool enable_damping;
    float damping_factor;
    bool enable_zoom;
    float zoom_speed;
    bool enable_rotate;
    float rotate_speed;
    bool enable_pan;
    float pan_speed;
    float min_distance;
    float max_distance;
    float min_polar_angle;
    float max_polar_angle;
    bool auto_rotate;
    float auto_rotate_speed;

    // internal
    float delta_theta;
    float delta_phi;
    float scale;
    Vector3 pan_offset;
    bool rotating;
    bool panning;
}

fn OrbitControls* orbit_controls(Camera* camera) {
    OrbitControls* c = mem::new(OrbitControls);
    *c = {
        .camera = camera,
        .enabled = true,
        .damping_factor = 0.05,
        .enable_zoom = true,
        .zoom_speed = 1,
        .enable_rotate = true,
        .rotate_speed = 1,
        .enable_pan = true,
        .pan_speed = 1,
        .min_distance = 0,
        .max_distance = 1e30,
        .min_polar_angle = 0,
        .max_polar_angle = drie::PI,
        .auto_rotate_speed = 2,
        .scale = 1,
    };
    if (!all.len()) drie::add_event_listener(&on_event);
    all.push(c);
    return c;
}

fn void OrbitControls.pan(&self, float dx, float dy) {
    drie::Mat4 m  = self.camera.matrix_world;
    Vector3 right = { m[0][0], m[0][1], m[0][2] };
    Vector3 up    = { m[1][0], m[1][1], m[1][2] };
    float h       = drie::window_height();
    float world_per_pixel;
    if (self.camera.camera_type == PERSPECTIVE) {
        PerspectiveCamera* pc = (PerspectiveCamera*)self.camera;
        float dist            = (self.camera.position - self.target).length();
        world_per_pixel = 2 * dist * math::tan(pc.fov * drie::DEG2RAD / 2) / h;
    } else {
        OrthographicCamera* oc = (OrthographicCamera*)self.camera;
        world_per_pixel = (oc.top - oc.bottom) / self.camera.zoom / h;
    }
    self.pan_offset += (right * -dx + up * dy) * world_per_pixel * self.pan_speed;
}

fn void OrbitControls.handle_event(&self, SappEvent* e) {
    if (!self.enabled) return;
    switch (e.type) {
        case MOUSE_DOWN:
            bool shift = (e.modifiers & sapp::MODIFIER_SHIFT) != 0;
            if (e.mouse_button == LEFT && !shift) self.rotating = self.enable_rotate;
            if (e.mouse_button == RIGHT || e.mouse_button == MIDDLE || (e.mouse_button == LEFT && shift)) {
                self.panning = self.enable_pan;
            }
        case MOUSE_UP:
        case MOUSE_LEAVE:
            self.rotating = false;
            self.panning = false;
        case MOUSE_MOVE:
            float dpi = sapp::dpi_scale();
            float dx = e.mouse_dx / dpi;
            float dy = e.mouse_dy / dpi;
            float h = drie::window_height();
            if (self.rotating) {
                self.delta_theta -= 2 * drie::PI * dx / h * self.rotate_speed;
                self.delta_phi -= 2 * drie::PI * dy / h * self.rotate_speed;
            } else if (self.panning) {
                self.pan(dx, dy);
            }
        case MOUSE_SCROLL:
            if (!self.enable_zoom || e.scroll_y == 0) break;
            float factor = math::pow(0.95f, self.zoom_speed * math::abs(e.scroll_y));
            if (e.scroll_y > 0) {
                self.scale *= factor;
            } else {
                self.scale /= factor;
            }
        default:
            break;
    }
}

// Applies pending input to the camera. Call once per frame.
fn void OrbitControls.update(&self) {
    Camera* cam = self.camera;
    if (self.auto_rotate && !self.rotating) {
        self.delta_theta -= 2 * drie::PI / 60 / 60 * self.auto_rotate_speed;
    }

    Vector3 offset = cam.position - self.target;
    float radius   = offset.length();
    float theta    = math::atan2(offset.x, offset.z);
    float phi      = radius > 0 ? math::acos(drie::clamp(offset.y / radius, -1, 1)) : 0;

    float damp = self.enable_damping ? self.damping_factor : 1;
    theta += self.delta_theta * damp;
    phi += self.delta_phi * damp;
    phi = drie::clamp(phi, math::max(self.min_polar_angle, 1e-6f), math::min(self.max_polar_angle, drie::PI - 1e-6f));

    if (cam.camera_type == ORTHOGRAPHIC) {
        cam.zoom = math::max(cam.zoom / self.scale, 0.01f);
    } else {
        radius = drie::clamp(radius * self.scale, self.min_distance, self.max_distance);
    }

    self.target += self.pan_offset * damp;

    cam.position = self.target + (Vector3) {
        radius * math::sin(phi) * math::sin(theta),
        radius * math::cos(phi),
        radius * math::sin(phi) * math::cos(theta),
    };
    cam.look_at_vec(self.target);

    if (self.enable_damping) {
        self.delta_theta *= 1 - self.damping_factor;
        self.delta_phi *= 1 - self.damping_factor;
        self.pan_offset *= 1 - self.damping_factor;
    } else {
        self.delta_theta = 0;
        self.delta_phi = 0;
        self.pan_offset = { 0, 0, 0 };
    }
    self.scale = 1;
}

fn void on_event(SappEvent* e) @local {
    foreach (c : all) c.handle_event(e);
}
// WalkControls: look round with the mouse (the pointer locked to the window from the first
// update, no button held; Esc frees it, a click locks it again) and walk with W, A, S and D
// along the ground, E and Q up and down. Where the view points and how fast the eye moves
// each follow what the mouse and keys ask for on a damped spring, without overshoot: `spring`
// is how fast (per second; the gap left after t seconds is e^(-spring t)). With `bounded`, the
// eye stays inside min..max. Call update(dt) every frame.
//
//   WalkControls* c = controls::walk_controls(camera, { 1, 2, -3 });
//   c.bounded = true;
//   c.min = { -5, 0.2, -5 };
//   c.max = { 5, 3, 5 };
struct WalkControls {
    Camera* camera;
    bool enabled;
    // radians a pixel of mouse movement turns the view, and its spring
    float look_speed;
    float look_spring;
    // metres a second, and its spring
    float walk_speed;
    float walk_spring;
    bool bounded;
    Vector3 min;
    Vector3 max;

    // internal
    float yaw;
    float pitch;
    float want_yaw;
    float want_pitch;
    Vector3 velocity;
    bool[6] held;
    bool locked_once;
}

List{WalkControls*} walkers @local;

// Controls for `camera` (where it is now), looking at `at`.
fn WalkControls* walk_controls(Camera* camera, Vector3 at) {
    WalkControls* c = mem::new(WalkControls);
    *c = {
        .camera = camera,
        .enabled = true,
        .look_speed = 0.002f,
        .look_spring = 18,
        .walk_speed = 2.5f,
        .walk_spring = 6,
    };
    c.look_at(at);
    if (!walkers.len()) drie::add_event_listener(&on_walk_event);
    walkers.push(c);
    return c;
}

// Turns the view (at once, no spring) to look at `at`.
fn void WalkControls.look_at(&self, Vector3 at) {
    Vector3 d = (at - self.camera.position).normalize();
    self.yaw = self.want_yaw = math::atan2(d.x, -d.z);
    self.pitch = self.want_pitch = math::asin(math::clamp(d.y, -1.0f, 1.0f));
}

// Where the view points, a unit vector.
fn Vector3 WalkControls.direction(&self) {
    return { math::sin(self.yaw) * math::cos(self.pitch), math::sin(self.pitch), -math::cos(self.yaw) * math::cos(self.pitch) };
}

fn void WalkControls.handle_event(&self, SappEvent* e) {
    if (!self.enabled) return;
    switch (e.type) {
        case MOUSE_MOVE:
            if (!sapp::mouse_locked()) return;
            self.want_yaw += e.mouse_dx * self.look_speed;
            self.want_pitch = math::clamp(self.want_pitch - e.mouse_dy * self.look_speed, -1.5f, 1.5f);
        case MOUSE_DOWN:
            sapp::lock_mouse(true);
        case KEY_DOWN:
        case KEY_UP:
            bool down = e.type == KEY_DOWN;
            switch (e.key_code) {
                case W: self.held[0] = down;
                case S: self.held[1] = down;
                case A: self.held[2] = down;
                case D: self.held[3] = down;
                case E: self.held[4] = down;
                case Q: self.held[5] = down;
                case ESCAPE: if (down) sapp::lock_mouse(false);
                default:
            }
        case UNFOCUSED:
            self.held = {};
        default:
    }
}

fn void WalkControls.update(&self, float dt) {
    if (!self.enabled) return;
    if (!self.locked_once) {
        sapp::lock_mouse(true);
        self.locked_once = true;
    }
    float k = 1 - math::exp(-self.look_spring * dt);
    self.yaw += (self.want_yaw - self.yaw) * k;
    self.pitch += (self.want_pitch - self.pitch) * k;
    Vector3 ahead = { math::sin(self.yaw), 0, -math::cos(self.yaw) };
    Vector3 right = { -ahead.z, 0, ahead.x };
    Vector3 move;
    if (self.held[0]) move += ahead;
    if (self.held[1]) move -= ahead;
    if (self.held[2]) move -= right;
    if (self.held[3]) move += right;
    if (self.held[4]) move.y += 1;
    if (self.held[5]) move.y -= 1;
    Vector3 want = move.length() > 1e-4f ? move.normalize() * self.walk_speed : (Vector3) { 0, 0, 0 };
    self.velocity += (want - self.velocity) * (1 - math::exp(-self.walk_spring * dt));
    Vector3 p = self.camera.position + self.velocity * dt;
    if (self.bounded) {
        p = { math::clamp(p.x, self.min.x, self.max.x), math::clamp(p.y, self.min.y, self.max.y), math::clamp(p.z, self.min.z, self.max.z) };
    }
    self.camera.position = p;
    self.camera.look_at_vec(p + self.direction());
}

fn void on_walk_event(SappEvent* e) @local {
    foreach (c : walkers) c.handle_event(e);
}
