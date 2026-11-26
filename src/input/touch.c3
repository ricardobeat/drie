// On-screen touch controls: a floating left-thumb stick, right-side look dragging, and circular
// buttons whose pressed bits merge into `input::gamepad().held`. Enabled from startup on iOS and
// Android, or upon the first touch event elsewhere; positions are in points (`pixels / dpi_scale`),
// and `ui::draw()` renders the overlay.
//
//   TouchControls* touch = input::touch_controls();
//   touch.add_button(input::PAD_NORTH, "LAMP", 70, 150);
//   walk(touch.stick_x, touch.stick_y);
//   float dx, dy;
//   touch.take_look(&dx, &dy);
module drie::input;
import std::collections::list, std::math;
import sokol, drie, drie::ui;

// Screen corner from which a `TouchButton` offset `(x, y)` is measured.
enum TouchAnchor {
    BOTTOM_RIGHT, // Offset inward from the bottom-right corner.
    TOP_RIGHT, // Offset inward from the top-right corner.
    BOTTOM_LEFT, // Offset inward from the bottom-left corner.
    TOP_LEFT, // Offset inward from the top-left corner.
}

// Circular on-screen button that drives one or more `PAD_*` bits while held.
struct TouchButton {
    uint button; // `PAD_*` bitmask held while this button is pressed.
    String label; // Centered text label drawn inside the button ring.
    TouchAnchor anchor; // Corner of the window that `(x, y)` is measured from.
    float x; // Horizontal distance in points from the anchor corner to the center.
    float y; // Vertical distance in points from the anchor corner to the center.
    float radius; // Drawn radius in points (hit testing uses 1.25x this radius).
    bool down; // True while a finger is holding this button.
    uptr finger; // Identifier of the touchpoint holding this button.
}

// Virtual stick, look-drag accumulator, and button list for touch input.
struct TouchControls {
    bool enabled; // Drawn and included in `held()` when true; auto-enabled on first touch.
    float stick_x; // Virtual stick X in [-1, 1], right positive.
    float stick_y; // Virtual stick Y in [-1, 1], up positive.
    float stick_radius; // Ring radius of the virtual stick in points.
    float stick_home_x; // Resting center X in points from the left edge.
    float stick_home_y; // Resting center Y in points from the bottom edge.
    float run_beyond; // Stick radii beyond which `running` (and `PAD_L3`) activates.
    bool running; // True when the thumb is pushed past `stick_radius * run_beyond`.
    bool mouse_as_touch; // Translates left-mouse drags into touch input and consumes mouse events.
    bool stick_as_dpad; // Maps stick deflections past 0.6 into `PAD_UP/DOWN/LEFT/RIGHT` bits.
    List{TouchButton} buttons; // Registered on-screen buttons.

    float look_dx; // Accumulated horizontal look drag in points since `take_look()`.
    float look_dy; // Accumulated vertical look drag in points since `take_look()`.
    bool stick_down; // True while a finger owns the virtual stick.
    uptr stick_finger; // Identifier of the touchpoint owning the stick.
    float stick_x0; // Active stick ring center X in points.
    float stick_y0; // Active stick ring center Y in points.
    float thumb_x; // Current stick touchpoint X in points.
    float thumb_y; // Current stick touchpoint Y in points.
    bool look_down; // True while a finger is dragging to look.
    uptr look_finger; // Identifier of the touchpoint dragging to look.
    float look_x; // Previous look touchpoint X in points.
    float look_y; // Previous look touchpoint Y in points.
}

TouchControls touch_state @local = {
    .enabled = env::OS_TYPE == IOS || env::ANDROID,
    .stick_radius = 52,
    .stick_home_x = 110,
    .stick_home_y = 110,
    .run_beyond = 1.25,
};

// Returns the singleton `TouchControls` state.
fn TouchControls* touch_controls() => &touch_state;

// Registers an on-screen button of `radius` points at `(x, y)` from `anchor` that holds `button`.
fn void TouchControls.add_button(
    &self,
    uint button,
    String label,
    float x,
    float y,
    float radius = 32,
    TouchAnchor anchor = BOTTOM_RIGHT
) {
    self.buttons.allocator = mem;
    self.buttons.push({ .button = button, .label = label, .anchor = anchor, .x = x, .y = y, .radius = radius });
}

// Writes the look drag in points (right and down positive) since the last call and resets it to 0.
fn void TouchControls.take_look(&self, float* dx, float* dy) {
    *dx = self.look_dx;
    *dy = self.look_dy;
    self.look_dx = 0;
    self.look_dy = 0;
}

// Returns the `PAD_*` bitmask currently held by touch buttons and stick states.
fn uint TouchControls.held(&self) {
    if (!self.enabled) return 0;
    uint bits;
    foreach (&b : self.buttons) if (b.down) bits |= b.button;
    if (self.stick_down && self.stick_as_dpad) {
        // a flick of the stick moves through a menu
        if (self.stick_y > 0.6f) bits |= PAD_UP;
        if (self.stick_y < -0.6f) bits |= PAD_DOWN;
        if (self.stick_x < -0.6f) bits |= PAD_LEFT;
        if (self.stick_x > 0.6f) bits |= PAD_RIGHT;
    } else if (self.running) {
        bits |= PAD_L3;
    }
    return bits;
}

// Computes the button's screen-space center `(x, y)` in points from the current window size.
fn void TouchButton.center(&self, float* x, float* y) {
    float w = drie::window_width();
    float h = drie::window_height();
    switch (self.anchor) {
        case BOTTOM_RIGHT: *x = w - self.x; *y = h - self.y;
        case TOP_RIGHT: *x = w - self.x; *y = self.y;
        case BOTTOM_LEFT: *x = self.x; *y = h - self.y;
        case TOP_LEFT: *x = self.x; *y = self.y;
    }
}

// Recomputes `stick_x`, `stick_y`, `running`, and floating center from the current thumb position.
fn void TouchControls.move_stick(&self) {
    float r      = self.stick_radius;
    float dx     = self.thumb_x - self.stick_x0;
    float dy     = self.thumb_y - self.stick_y0;
    float len    = math::sqrt(dx * dx + dy * dy);
    // a thumb that slides far past the ring drags the stick along
    float follow = r * (self.run_beyond + 0.35f);
    if (len > follow) {
        self.stick_x0 += dx * (len - follow) / len;
        self.stick_y0 += dy * (len - follow) / len;
    }
    self.running = len > r * self.run_beyond;
    float k = len > r ? 1 / len : 1 / r;
    self.stick_x = dx * k;
    self.stick_y = -dy * k;
}

// Releases the virtual stick and zeroes its axis and run state.
fn void TouchControls.release_stick(&self) {
    self.stick_down = false;
    self.running = false;
    self.stick_x = 0;
    self.stick_y = 0;
}

// Takes the touch events (the renderer passes them in before anything else sees them).
fn void touch_event(SappEvent* e) @local {
    TouchControls* t = &touch_state;
    t.enabled = true;
    float scale = sapp::isvalid() ? sapp::dpi_scale() : 1;
    for (int i = 0; i < e.num_touches; i++) {
        SappTouchpoint* p = &e.touches[i];
        if (!p.changed) continue;
        float x = p.pos_x / scale;
        float y = p.pos_y / scale;
        uptr id = p.identifier;
        switch (e.type) {
            case TOUCHES_BEGAN:
                bool taken;
                foreach (&b : t.buttons) {
                    float bx, by;
                    b.center(&bx, &by);
                    float reach = b.radius * 1.25f;
                    if (!b.down && (x - bx) * (x - bx) + (y - by) * (y - by) < reach * reach) {
                        b.down = true;
                        b.finger = id;
                        taken = true;
                        break;
                    }
                }
                if (taken) break;
                if (!t.stick_down && x < drie::window_width() * 0.5f) {
                    t.stick_down = true;
                    t.stick_finger = id;
                    // centered where the thumb lands, so it starts at rest
                    t.stick_x0 = x;
                    t.stick_y0 = y;
                    t.thumb_x = x;
                    t.thumb_y = y;
                    t.move_stick();
                } else if (!t.look_down) {
                    t.look_down = true;
                    t.look_finger = id;
                    t.look_x = x;
                    t.look_y = y;
                }
            case TOUCHES_MOVED:
                if (t.stick_down && id == t.stick_finger) {
                    t.thumb_x = x;
                    t.thumb_y = y;
                    t.move_stick();
                } else if (t.look_down && id == t.look_finger) {
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
fn bool touch_from_mouse(SappEvent* e) @local {
    if (!touch_state.mouse_as_touch) return false;
    SappEventType type;
    switch (e.type) {
        case MOUSE_DOWN: if (e.mouse_button != LEFT) return true; type = TOUCHES_BEGAN;
        case MOUSE_UP: if (e.mouse_button != LEFT) return true; type = TOUCHES_ENDED;
        case MOUSE_MOVE: type = TOUCHES_MOVED;
        case MOUSE_SCROLL:
        case MOUSE_ENTER:
        case MOUSE_LEAVE: return true;
        default: return false;
    }
    SappEvent t = { .type = type, .num_touches = 1 };
    t.touches[0] = { .identifier = 1, .pos_x = e.mouse_x, .pos_y = e.mouse_y, .changed = true };
    touch_event(&t);
    return true;
}

// Lets go of everything (the app went to the background, say).
fn void touch_reset() @local {
    TouchControls* t = &touch_state;
    t.release_stick();
    t.look_down = false;
    t.look_dx = t.look_dy = 0;
    foreach (&b : t.buttons) b.down = false;
}

fn void ui_circle(float x, float y, float r, float width, float cr, float cg, float cb, float a) @local {
    // a ring `width` thick, or a disc when width is 0
    const int SEGMENTS = 40;
    float inner        = width > 0 ? math::max(r - width, 0.0f) : 0;
    sgl::begin_triangle_strip();
    sgl::c4f(cr, cg, cb, a);
    for (int i = 0; i <= SEGMENTS; i++) {
        float ang = (float)i / SEGMENTS * 2 * drie::PI;
        float c   = math::cos(ang);
        float s   = math::sin(ang);
        sgl::v2f(x + c * r, y + s * r);
        sgl::v2f(x + c * inner, y + s * inner);
    }
    sgl::end();
}

// Draws the stick and the buttons into the current sgl layer, and the labels into sdtx
// (whose canvas has `canvas_scale` points to a unit; see ui::draw).
fn void TouchControls.draw(&self, float canvas_scale) {
    TouchControls* t = self;
    sgl::matrix_mode_projection();
    sgl::push_matrix();
    sgl::load_identity();
    sgl::ortho(0, drie::window_width(), drie::window_height(), 0, -1, 1);
    float r  = t.stick_radius;
    float x0 = t.stick_down ? t.stick_x0 : t.stick_home_x;
    float y0 = t.stick_down ? t.stick_y0 : drie::window_height() - t.stick_home_y;
    float a  = t.stick_down ? 0.5f : 0.22f;
    ui_circle(x0, y0, r, 3, 1, 1, 1, a);
    float kx = x0 + t.stick_x * r;
    float ky = y0 - t.stick_y * r;
    ui_circle(kx, ky, r * 0.42f, 0, 1, 1, 1, t.running ? 0.65f : a);
    foreach (&b : t.buttons) {
        float bx, by;
        b.center(&bx, &by);
        ui_circle(bx, by, b.radius, b.down ? 0 : 2.5f, 1, 1, 1, b.down ? 0.45f : 0.3f);
        // the label, centered (sdtx characters are 8 canvas units)
        float col = (bx / canvas_scale - (float)b.label.len * 4) / 8;
        float row = (by / canvas_scale - 4) / 8;
        ui::text(col, row, b.label, 1, 1, 1, b.down ? 0.9f : 0.6f);
    }
    sgl::pop_matrix();
}

// The window's events go through here first (see drie::add_event_filter): the touch controls
// see touches, and the mouse when it stands in for a finger.
fn bool filter_touch_event(Event* e) @local {
    switch (e.type) {
        case TOUCHES_BEGAN:
        case TOUCHES_MOVED:
        case TOUCHES_ENDED:
        case TOUCHES_CANCELLED:
            touch_event(e);
        case MOUSE_DOWN:
        case MOUSE_UP:
        case MOUSE_MOVE:
        case MOUSE_SCROLL:
        case MOUSE_ENTER:
        case MOUSE_LEAVE:
            return touch_from_mouse(e);
        case UNFOCUSED:
        case SUSPENDED:
            touch_reset();
        default:
            break;
    }
    return false;
}

// Registered first, so the touch controls see the events before a game's own filters (menus).
fn void register_touch() @init @local {
    drie::add_event_filter(&filter_touch_event, first: true);
}
