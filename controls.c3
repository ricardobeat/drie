// OrbitControls, like three's (three/addons/controls/OrbitControls.js):
//   left drag: rotate, right drag (or shift + left): pan, scroll: zoom.
//   Call controls.update() every frame, like in three.
//
//   import drie::controls;
//   OrbitControls* c = controls::orbit_controls(camera);
module drie::controls;
import std::math, std::collections::list;
import sokol, drie;
// Every OrbitControls made, fed the window's events by one listener.
List{OrbitControls*} all @local;

struct OrbitControls
{
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

fn OrbitControls* orbit_controls(Camera* camera)
{
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

fn void OrbitControls.pan(&self, float dx, float dy)
{
    drie::Mat4 m = self.camera.matrix_world;
    Vector3 right = { m[0][0], m[0][1], m[0][2] };
    Vector3 up = { m[1][0], m[1][1], m[1][2] };
    float h = drie::window_height();
    float world_per_pixel;
    if (self.camera.camera_type == PERSPECTIVE)
    {
        PerspectiveCamera* pc = (PerspectiveCamera*)self.camera;
        float dist = (self.camera.position - self.target).length();
        world_per_pixel = 2 * dist * math::tan(pc.fov * drie::DEG2RAD / 2) / h;
    }
    else
    {
        OrthographicCamera* oc = (OrthographicCamera*)self.camera;
        world_per_pixel = (oc.top - oc.bottom) / self.camera.zoom / h;
    }
    self.pan_offset += (right * -dx + up * dy) * world_per_pixel * self.pan_speed;
}

fn void OrbitControls.handle_event(&self, SappEvent* e)
{
    if (!self.enabled) return;
    switch (e.type)
    {
        case MOUSE_DOWN:
            bool shift = (e.modifiers & sapp::MODIFIER_SHIFT) != 0;
            if (e.mouse_button == LEFT && !shift) self.rotating = self.enable_rotate;
            if (e.mouse_button == RIGHT || e.mouse_button == MIDDLE || (e.mouse_button == LEFT && shift))
            {
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
            if (self.rotating)
            {
                self.delta_theta -= 2 * drie::PI * dx / h * self.rotate_speed;
                self.delta_phi -= 2 * drie::PI * dy / h * self.rotate_speed;
            }
            else if (self.panning)
            {
                self.pan(dx, dy);
            }
        case MOUSE_SCROLL:
            if (!self.enable_zoom || e.scroll_y == 0) break;
            float factor = math::pow(0.95f, self.zoom_speed * math::abs(e.scroll_y));
            if (e.scroll_y > 0) { self.scale *= factor; } else { self.scale /= factor; }
        default:
            break;
    }
}

// Applies pending input to the camera. Call once per frame.
fn void OrbitControls.update(&self)
{
    Camera* cam = self.camera;
    if (self.auto_rotate && !self.rotating)
    {
        self.delta_theta -= 2 * drie::PI / 60 / 60 * self.auto_rotate_speed;
    }

    Vector3 offset = cam.position - self.target;
    float radius = offset.length();
    float theta = math::atan2(offset.x, offset.z);
    float phi = radius > 0 ? math::acos(drie::clamp(offset.y / radius, -1, 1)) : 0;

    float damp = self.enable_damping ? self.damping_factor : 1;
    theta += self.delta_theta * damp;
    phi += self.delta_phi * damp;
    phi = drie::clamp(phi, math::max(self.min_polar_angle, 1e-6f), math::min(self.max_polar_angle, drie::PI - 1e-6f));

    if (cam.camera_type == ORTHOGRAPHIC)
    {
        cam.zoom = math::max(cam.zoom / self.scale, 0.01f);
    }
    else
    {
        radius = drie::clamp(radius * self.scale, self.min_distance, self.max_distance);
    }

    self.target += self.pan_offset * damp;

    cam.position = self.target + (Vector3){
        radius * math::sin(phi) * math::sin(theta),
        radius * math::cos(phi),
        radius * math::sin(phi) * math::cos(theta),
    };
    cam.look_at_vec(self.target);

    if (self.enable_damping)
    {
        self.delta_theta *= 1 - self.damping_factor;
        self.delta_phi *= 1 - self.damping_factor;
        self.pan_offset *= 1 - self.damping_factor;
    }
    else
    {
        self.delta_theta = 0;
        self.delta_phi = 0;
        self.pan_offset = { 0, 0, 0 };
    }
    self.scale = 1;
}

fn void on_event(SappEvent* e) @local
{
    foreach (c : all) c.handle_event(e);
}
