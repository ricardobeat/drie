// Game controller: DualSense, DualShock, Xbox and other controllers the system knows
// (macOS GameController framework, through lib/gamectl.c3l). Needs `--lib gamectl`.
//
// The renderer polls it before every frame. Read it with input::gamepad():
//
//   Gamepad* pad = input::gamepad();
//   if (pad.connected) walk(pad.left_x, pad.left_y);    // sticks: -1..1, up is +y, dead zone removed
//   if (pad.pressed(input::PAD_SOUTH)) jump();          // went down this frame
//
// With `keys` on (the default) the pad also sends key events, so menus and keyboard
// code work with it unchanged: the d-pad sends the arrow keys, cross / A sends enter,
// circle / B backspace, and options / menu escape.
module drie::input;
import drie::gamectl, sokol, drie;
import std::math;

const uint PAD_SOUTH = 1 << 0; // cross / A
const uint PAD_EAST  = 1 << 1; // circle / B
const uint PAD_WEST  = 1 << 2; // square / X
const uint PAD_NORTH = 1 << 3; // triangle / Y
const uint PAD_L1    = 1 << 4;
const uint PAD_R1    = 1 << 5;
const uint PAD_L2    = 1 << 6;
const uint PAD_R2    = 1 << 7;
const uint PAD_L3    = 1 << 8; // left stick pressed
const uint PAD_R3    = 1 << 9;
const uint PAD_UP    = 1 << 10; // d-pad
const uint PAD_DOWN  = 1 << 11;
const uint PAD_LEFT  = 1 << 12;
const uint PAD_RIGHT = 1 << 13;
const uint PAD_MENU  = 1 << 14; // options / menu
const uint PAD_VIEW  = 1 << 15; // create / view
const uint PAD_HOME  = 1 << 16; // PS / Xbox button

// Sticks closer to the middle than this read as 0.
const float PAD_DEAD_ZONE = 0.12;

struct Gamepad {
    bool connected;
    String name;
    float left_x;
    float left_y;
    float right_x;
    float right_y;
    // analog triggers, 0..1
    float l2;
    float r2;
    uint held;
    uint last;
    // send key events for the d-pad and the face buttons
    bool keys;
    char[64] name_buf;
}

Gamepad pad_state @local = { .keys = true };

fn Gamepad* gamepad() => &pad_state;

fn bool Gamepad.down(&self, uint buttons) => (self.held & buttons) != 0;
// Went down since the last frame.
fn bool Gamepad.pressed(&self, uint buttons) => (self.held & ~self.last & buttons) != 0;
fn bool Gamepad.released(&self, uint buttons) => (~self.held & self.last & buttons) != 0;

// The stick without its dead zone, rescaled so it still reaches 1.
fn void dead_zone(float* x, float* y) @local {
    float len = math::sqrt(*x * *x + *y * *y);
    if (len <= PAD_DEAD_ZONE) {
        *x = 0;
        *y = 0;
        return;
    }
    float k = math::min((len - PAD_DEAD_ZONE) / (1 - PAD_DEAD_ZONE), 1.0f) / len;
    *x *= k;
    *y *= k;
}

fn void pad_key(KeyCode key, bool down) @local {
    SappEvent e = { .type = down ? KEY_DOWN : KEY_UP, .key_code = key };
    drie::send_event(&e);
}

// Reads the controller before every frame (see drie::add_frame_hook).
fn void register_input() @init @local => drie::add_frame_hook(&poll_gamepad);

fn void poll_gamepad() @local {
    Gamepad* p = &pad_state;
    GcState s;
    gamectl::poll(&s);
    p.last = p.held;
    p.connected = s.connected;
    // and the touch screen's buttons (touch_controls())
    p.held = s.buttons | touch_controls().held();
    p.left_x = s.left_x;
    p.left_y = s.left_y;
    p.right_x = s.right_x;
    p.right_y = s.right_y;
    dead_zone(&p.left_x, &p.left_y);
    dead_zone(&p.right_x, &p.right_y);
    p.l2 = s.l2;
    p.r2 = s.r2;
    p.name_buf = s.name;
    p.name = ((ZString)&p.name_buf).str_view();
    if (!p.keys) return;
    uint[*] buttons = { PAD_UP, PAD_DOWN, PAD_LEFT, PAD_RIGHT, PAD_SOUTH, PAD_EAST, PAD_MENU };
    KeyCode[*] keys = { UP, DOWN, LEFT, RIGHT, ENTER, BACKSPACE, ESCAPE };
    foreach (i, b : buttons) {
        if (p.pressed(b)) pad_key(keys[i], true);
        if (p.released(b)) pad_key(keys[i], false);
    }
}