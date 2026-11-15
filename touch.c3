// Touch controls for phones and tablets: a virtual stick under the left thumb, dragging
// anywhere else to look around, and round on-screen buttons that press gamepad buttons.
//
// The buttons go into three::gamepad()'s `held` bits, so they work wherever a controller's
// buttons do, and with `keys` on they also send the same key events (menus work with them).
// The stick and the look drag are read from here:
//
//   TouchControls* touch = three::touch_controls();
//   touch.add_button(three::PAD_NORTH, "LAMP", 70, 150);             // from the bottom right
//   touch.add_button(three::PAD_MENU, "MENU", 40, 40, 22, TOP_RIGHT);
//   ...
//   walk(touch.stick_x, touch.stick_y);
//   float dx, dy;
//   touch.take_look(&dx, &dy);                                       // points dragged since last time
//
// The stick appears where the left thumb comes down (in the left half of the screen) and
// follows it when it slides past the edge; pushed past `run_beyond` it also holds PAD_L3. While
// a menu is open, flicking the stick presses the d-pad instead, to move through the items.
//
// It is shown and read once `enabled`: from the start on iOS and Android, elsewhere from the
// first touch. Positions are in points (pixels / sapp::dpi_scale()), like window_width().
// `mouse_as_touch` lets a mouse stand in for one finger.
module three;
import std::collections::list, std::math;
import sokol;

enum TouchAnchor
{
    BOTTOM_RIGHT,
    TOP_RIGHT,
    BOTTOM_LEFT,
    TOP_LEFT,
}

struct TouchButton
{
    uint button;          // PAD_* bits it holds down
    String label;
    TouchAnchor anchor;
    float x;              // its middle, in points from the anchor's corner
    float y;
    float radius;
    bool down;
    uptr finger;
}

struct TouchControls
{
    bool enabled;
    // -1..1, up is +y
    float stick_x;
    float stick_y;
    // the stick's size and where it rests when no thumb is on it (from the bottom left)
    float stick_radius;
    float stick_home_x;
    float stick_home_y;
    // past this many radii the stick holds PAD_L3 (run)
    float run_beyond;
    bool running;
    // the mouse is a finger (to try the controls on a desktop): it no longer reaches the app
    bool mouse_as_touch;
    List{TouchButton} buttons;

    float look_dx;
    float look_dy;
    bool stick_down;
    uptr stick_finger;
    float stick_x0;
    float stick_y0;
    float thumb_x;
    float thumb_y;
    bool look_down;
    uptr look_finger;
    float look_x;
    float look_y;
}

TouchControls touch_state @local = {
    .enabled = env::OS_TYPE == IOS || env::ANDROID,
    .stick_radius = 52,
    .stick_home_x = 110,
    .stick_home_y = 110,
    .run_beyond = 1.25,
};

fn TouchControls* touch_controls() => &touch_state;

fn void TouchControls.add_button(&self, uint button, String label, float x, float y, float radius = 32,
                                 TouchAnchor anchor = BOTTOM_RIGHT)
{
    self.buttons.allocator = mem;
    self.buttons.push({ .button = button, .label = label, .anchor = anchor, .x = x, .y = y, .radius = radius });
}

// The look drag since the last call, in points (right and down are positive), and starts over.
fn void TouchControls.take_look(&self, float* dx, float* dy)
{
    *dx = self.look_dx;
    *dy = self.look_dy;
    self.look_dx = 0;
    self.look_dy = 0;
}

// The PAD_* bits the touch buttons (and the stick) hold down now.
fn uint TouchControls.held(&self)
{
    if (!self.enabled) return 0;
    uint bits;
    foreach (&b : self.buttons) if (b.down) bits |= b.button;
    if (self.stick_down && any_menu_open())
    {
        // a flick of the stick moves through a menu
        if (self.stick_y > 0.6f) bits |= PAD_UP;
        if (self.stick_y < -0.6f) bits |= PAD_DOWN;
        if (self.stick_x < -0.6f) bits |= PAD_LEFT;
        if (self.stick_x > 0.6f) bits |= PAD_RIGHT;
    }
    else if (self.running)
    {
        bits |= PAD_L3;
    }
    return bits;
}

fn void TouchButton.center(&self, float* x, float* y)
{
    float w = window_width();
    float h = window_height();
    switch (self.anchor)
    {
        case BOTTOM_RIGHT: *x = w - self.x; *y = h - self.y;
        case TOP_RIGHT: *x = w - self.x; *y = self.y;
        case BOTTOM_LEFT: *x = self.x; *y = h - self.y;
        case TOP_LEFT: *x = self.x; *y = self.y;
    }
}

fn void TouchControls.move_stick(&self)
{
    float r = self.stick_radius;
    float dx = self.thumb_x - self.stick_x0;
    float dy = self.thumb_y - self.stick_y0;
    float len = math::sqrt(dx * dx + dy * dy);
    // a thumb that slides far past the ring drags the stick along
    float follow = r * (self.run_beyond + 0.35f);
    if (len > follow)
    {
        self.stick_x0 += dx * (len - follow) / len;
        self.stick_y0 += dy * (len - follow) / len;
    }
    self.running = len > r * self.run_beyond;
    float k = len > r ? 1 / len : 1 / r;
    self.stick_x = dx * k;
    self.stick_y = -dy * k;
}

fn void TouchControls.release_stick(&self)
{
    self.stick_down = false;
    self.running = false;
    self.stick_x = 0;
    self.stick_y = 0;
}

// Takes the touch events (the renderer passes them in before anything else sees them).
fn void touch_event(SappEvent* e)
{
    TouchControls* t = &touch_state;
    t.enabled = true;
    float scale = sapp::isvalid() ? sapp::dpi_scale() : 1;
    for (int i = 0; i < e.num_touches; i++)
    {
        SappTouchpoint* p = &e.touches[i];
        if (!p.changed) continue;
        float x = p.pos_x / scale;
        float y = p.pos_y / scale;
        uptr id = p.identifier;
        switch (e.type)
        {
            case TOUCHES_BEGAN:
                bool taken;
                foreach (&b : t.buttons)
                {
                    float bx, by;
                    b.center(&bx, &by);
                    float reach = b.radius * 1.25f;
                    if (!b.down && (x - bx) * (x - bx) + (y - by) * (y - by) < reach * reach)
                    {
                        b.down = true;
                        b.finger = id;
                        taken = true;
                        break;
                    }
                }
                if (taken) break;
                if (!t.stick_down && x < window_width() * 0.5f)
                {
                    t.stick_down = true;
                    t.stick_finger = id;
                    // centered where the thumb lands, so it starts at rest
                    t.stick_x0 = x;
                    t.stick_y0 = y;
                    t.thumb_x = x;
                    t.thumb_y = y;
                    t.move_stick();
                }
                else if (!t.look_down)
                {
                    t.look_down = true;
                    t.look_finger = id;
                    t.look_x = x;
                    t.look_y = y;
                }
            case TOUCHES_MOVED:
                if (t.stick_down && id == t.stick_finger)
                {
                    t.thumb_x = x;
                    t.thumb_y = y;
                    t.move_stick();
                }
                else if (t.look_down && id == t.look_finger)
                {
                    t.look_dx += x - t.look_x;
                    t.look_dy += y - t.look_y;
                    t.look_x = x;
                    t.look_y = y;
                }
            case TOUCHES_ENDED:
            case TOUCHES_CANCELLED:
                if (t.stick_down && id == t.stick_finger) t.release_stick();
                if (t.look_down && id == t.look_finger) t.look_down = false;
                foreach (&b : t.buttons) if (b.down && b.finger == id) b.down = false;
            default:
                break;
        }
    }
}

// With mouse_as_touch, turns the left mouse button's presses and drags into a finger's
// touches. Returns whether the event was taken.
fn bool touch_from_mouse(SappEvent* e)
{
    if (!touch_state.mouse_as_touch) return false;
    SappEventType type;
    switch (e.type)
    {
        case MOUSE_DOWN: if (e.mouse_button != LEFT) return true; type = TOUCHES_BEGAN;
        case MOUSE_UP: if (e.mouse_button != LEFT) return true; type = TOUCHES_ENDED;
        case MOUSE_MOVE: type = TOUCHES_MOVED;
        case MOUSE_SCROLL: case MOUSE_ENTER: case MOUSE_LEAVE: return true;
        default: return false;
    }
    SappEvent t = { .type = type, .num_touches = 1 };
    t.touches[0] = { .identifier = 1, .pos_x = e.mouse_x, .pos_y = e.mouse_y, .changed = true };
    touch_event(&t);
    return true;
}

// Lets go of everything (the app went to the background, say).
fn void touch_reset()
{
    TouchControls* t = &touch_state;
    t.release_stick();
    t.look_down = false;
    t.look_dx = t.look_dy = 0;
    foreach (&b : t.buttons) b.down = false;
}

fn void ui_circle(float x, float y, float r, float width, float cr, float cg, float cb, float a) @local
{
    // a ring `width` thick, or a disc when width is 0
    const int SEGMENTS = 40;
    float inner = width > 0 ? math::max(r - width, 0.0f) : 0;
    sgl::begin_triangle_strip();
    sgl::c4f(cr, cg, cb, a);
    for (int i = 0; i <= SEGMENTS; i++)
    {
        float ang = (float)i / SEGMENTS * 2 * PI;
        float c = math::cos(ang);
        float s = math::sin(ang);
        sgl::v2f(x + c * r, y + s * r);
        sgl::v2f(x + c * inner, y + s * inner);
    }
    sgl::end();
}

// Draws the stick and the buttons into the current sgl layer, and the labels into sdtx
// (whose canvas has `canvas_scale` points to a unit; see draw_ui).
fn void draw_touch_controls(float canvas_scale) @private
{
    TouchControls* t = &touch_state;
    sgl::matrix_mode_projection();
    sgl::push_matrix();
    sgl::load_identity();
    sgl::ortho(0, window_width(), window_height(), 0, -1, 1);
    float r = t.stick_radius;
    float x0 = t.stick_down ? t.stick_x0 : t.stick_home_x;
    float y0 = t.stick_down ? t.stick_y0 : window_height() - t.stick_home_y;
    float a = t.stick_down ? 0.5f : 0.22f;
    ui_circle(x0, y0, r, 3, 1, 1, 1, a);
    float kx = x0 + t.stick_x * r;
    float ky = y0 - t.stick_y * r;
    ui_circle(kx, ky, r * 0.42f, 0, 1, 1, 1, t.running ? 0.65f : a);
    foreach (&b : t.buttons)
    {
        float bx, by;
        b.center(&bx, &by);
        ui_circle(bx, by, b.radius, b.down ? 0 : 2.5f, 1, 1, 1, b.down ? 0.45f : 0.3f);
        // the label, centered (sdtx characters are 8 canvas units)
        float col = (bx / canvas_scale - (float)b.label.len * 4) / 8;
        float row = (by / canvas_scale - 4) / 8;
        ui_text(col, row, b.label, 1, 1, 1, b.down ? 0.9f : 0.6f);
    }
    sgl::pop_matrix();
}
