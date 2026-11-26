module drie::controls;
import std::math, std::collections::list;
import sokol, drie;

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
