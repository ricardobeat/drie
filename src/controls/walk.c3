// First-person walk controller: mouse look (pointer locked on first update; Esc releases, click
// re-locks) with WASD horizontal movement and E/Q vertical movement. Look angles and velocity
// follow input on exponential springs without overshoot. Call `update(dt)` once per frame.
// 
//   import drie::controls;
//   WalkControls* c = controls::walk_controls(camera, { 1, 2, -3 });
//   c.bounded = true;
//   c.min = { -5, 0.2, -5 };
//   c.max = { 5, 3, 5 };
module drie::controls;
import std::math, std::collections::list;
import sokol, drie;

// First-person mouse-look and keyboard movement controller with damped spring smoothing.
struct WalkControls {
    Camera* camera; // Controlled camera.
    bool enabled; // Ignores input and skips camera updates when false.
    float look_speed; // Radians of yaw/pitch turned per pixel of mouse movement.
    float look_spring; // Look spring rate per second (gap decays as e^(-look_spring * dt)).
    float walk_speed; // Target movement speed in meters per second.
    float walk_spring; // Velocity spring rate per second (gap decays as e^(-walk_spring * dt)).
    bool bounded; // Clamps camera position inside `min`..`max` when true.
    Vector3 min; // Lower world-space position bound when `bounded` is true.
    Vector3 max; // Upper world-space position bound when `bounded` is true.

    float yaw; // Current smoothed yaw in radians.
    float pitch; // Current smoothed pitch in radians, clamped to [-1.5, 1.5].
    float want_yaw; // Target yaw accumulated from mouse motion.
    float want_pitch; // Target pitch accumulated from mouse motion.
    Vector3 velocity; // Current smoothed world-space velocity in meters per second.
    bool[6] held; // Key state for W, S, A, D, E, Q in that order.
    bool locked_once; // True after the initial pointer lock in `update()`.
}

List{WalkControls*} walkers @local;

// Creates a WalkControls for `camera` initially oriented toward `at` and registers its listener.
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

// Snaps yaw and pitch immediately (without spring lag) so the camera looks at `at`.
fn void WalkControls.look_at(&self, Vector3 at) {
    Vector3 d = (at - self.camera.position).normalize();
    self.yaw = self.want_yaw = math::atan2(d.x, -d.z);
    self.pitch = self.want_pitch = math::asin(math::clamp(d.y, -1.0f, 1.0f));
}

// Returns the current unit forward direction derived from `yaw` and `pitch`.
fn Vector3 WalkControls.direction(&self) {
    return {
        math::sin(self.yaw) * math::cos(self.pitch),
        math::sin(self.pitch),
        -math::cos(self.yaw) * math::cos(self.pitch)
    };
}

// Processes keyboard, mouse, and focus events into target look angles and held movement keys.
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
            self.held = { };
        default:
    }
}

// Advances look and movement springs by `dt` seconds and updates the camera transform.
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
        p = {
            math::clamp(p.x, self.min.x, self.max.x),
            math::clamp(p.y, self.min.y, self.max.y),
            math::clamp(p.z, self.min.z, self.max.z)
        };
    }
    self.camera.position = p;
    self.camera.look_at_vec(p + self.direction());
}

fn void on_walk_event(SappEvent* e) @local {
    foreach (c : walkers) c.handle_event(e);
}