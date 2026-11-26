// Orbit camera controller: left-drag rotates around a target, right-drag or shift+left-drag
// pans, and scroll zooms. Events are wired automatically through a shared window listener; call
// `update()` once per frame to apply accumulated input.
//
//   import drie::controls;
//   OrbitControls* c = controls::orbit_controls(camera);
//   // each frame:
//   c.update();
module drie::controls;
import std::math, std::collections::list;
import sokol, drie;
// Shared registry of live OrbitControls fed by a single window event listener.
List{OrbitControls*} all @local;

// Spherical orbit controller around `target` for a perspective or orthographic camera.
struct OrbitControls {
    Camera* camera; // Controlled camera.
    Vector3 target; // World-space point the camera orbits and looks at.
    bool enabled; // Ignores input events when false.
    bool enable_damping; // Smooths rotation and panning toward rest each frame.
    float damping_factor; // Fraction of pending motion applied per frame when damping is enabled.
    bool enable_zoom; // Enables scroll-wheel zooming.
    float zoom_speed; // Multiplier for scroll-wheel zoom steps.
    bool enable_rotate; // Enables left-drag orbiting.
    float rotate_speed; // Multiplier for horizontal and vertical orbit drag.
    bool enable_pan; // Enables right/middle-drag and shift+left-drag panning.
    float pan_speed; // Multiplier for screen-space panning distance.
    float min_distance; // Minimum orbit radius for perspective cameras.
    float max_distance; // Maximum orbit radius for perspective cameras.
    float min_polar_angle; // Minimum polar angle from +Y in radians (clamped away from 0).
    float max_polar_angle; // Maximum polar angle from +Y in radians (clamped away from PI).
    bool auto_rotate; // Slowly orbits around `target` while the user is not dragging.
    float auto_rotate_speed; // Auto-rotate rate (at 60 fps, 2.0 completes a full turn in 30 s).

    float delta_theta; // Pending azimuth delta in radians.
    float delta_phi; // Pending polar angle delta in radians.
    float scale; // Pending zoom multiplier accumulated since the last update.
    Vector3 pan_offset; // Pending world-space offset to add to `target`.
    bool rotating; // True while a rotation drag is active.
    bool panning; // True while a pan drag is active.
}

// Creates an OrbitControls bound to `camera` and registers it with the window event listener.
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

// Accumulates a screen-space pan offset scaled to world units at `target` depth.
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

// Processes a sokol mouse or scroll event into pending orbit, pan, or zoom deltas.
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
