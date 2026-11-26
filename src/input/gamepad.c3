// Game controller input over `drie::gamectl` (Apple GameController on macOS, XInput on Windows),
// merged with on-screen buttons from `input::touch_controls()`. Polled automatically before each
// frame via `drie::add_frame_hook()`.
// 
//   Gamepad* pad = input::gamepad();
//   if (pad.connected) walk(pad.left_x, pad.left_y);  // sticks in [-1, 1], +Y up, dead zone removed
//   if (pad.pressed(input::PAD_SOUTH)) jump();        // pressed this frame
// 
// When `keys` is true (default), D-pad, south, east, and menu buttons also emit sokol key events
// (arrow keys, Enter, Backspace, Escape) so keyboard-driven menus work unchanged.
module drie::input;
import drie::gamectl, sokol, drie;
import std::math;

// Button bitmask: south face button (Cross / A).
const uint PAD_SOUTH = 1 << 0;
// Button bitmask: east face button (Circle / B).
const uint PAD_EAST  = 1 << 1;
// Button bitmask: west face button (Square / X).
const uint PAD_WEST  = 1 << 2;
// Button bitmask: north face button (Triangle / Y).
const uint PAD_NORTH = 1 << 3;
// Button bitmask: left shoulder button (L1 / LB).
const uint PAD_L1    = 1 << 4;
// Button bitmask: right shoulder button (R1 / RB).
const uint PAD_R1    = 1 << 5;
// Button bitmask: left trigger digital bit (L2 / LT).
const uint PAD_L2    = 1 << 6;
// Button bitmask: right trigger digital bit (R2 / RT).
const uint PAD_R2    = 1 << 7;
// Button bitmask: left stick click (L3 / LS); also set by touch stick past `run_beyond`.
const uint PAD_L3    = 1 << 8;
// Button bitmask: right stick click (R3 / RS).
const uint PAD_R3    = 1 << 9;
// Button bitmask: D-pad up.
const uint PAD_UP    = 1 << 10;
// Button bitmask: D-pad down.
const uint PAD_DOWN  = 1 << 11;
// Button bitmask: D-pad left.
const uint PAD_LEFT  = 1 << 12;
// Button bitmask: D-pad right.
const uint PAD_RIGHT = 1 << 13;
// Button bitmask: menu / options / start button.
const uint PAD_MENU  = 1 << 14;
// Button bitmask: view / create / back button.
const uint PAD_VIEW  = 1 << 15;
// Button bitmask: system home button (PS / Xbox).
const uint PAD_HOME  = 1 << 16;

// Radial threshold below which analog stick inputs read as 0.
const float PAD_DEAD_ZONE = 0.12;

// Current controller state, updated once per frame before user frame callbacks run.
struct Gamepad {
    bool connected; // True when a physical controller is connected.
    String name; // Controller name reported by the OS (view into `name_buf`).
    float left_x; // Left stick X in [-1, 1] after radial dead-zone rescaling.
    float left_y; // Left stick Y in [-1, 1], +Y up, after radial dead-zone rescaling.
    float right_x; // Right stick X in [-1, 1] after radial dead-zone rescaling.
    float right_y; // Right stick Y in [-1, 1], +Y up, after radial dead-zone rescaling.
    float l2; // Left analog trigger in [0, 1].
    float r2; // Right analog trigger in [0, 1].
    uint held; // Bitmask of `PAD_*` buttons held this frame (controller plus touch buttons).
    uint last; // Bitmask of `PAD_*` buttons held on the previous frame.
    bool keys; // Emits keyboard events for D-pad, south, east, and menu buttons when true.
    char[64] name_buf; // Backing storage for `name`.
}

Gamepad pad_state @local = { .keys = true };

// Returns the singleton `Gamepad` state polled before every frame.
fn Gamepad* gamepad() => &pad_state;

// Returns true if any button in the `buttons` bitmask is currently held.
fn bool Gamepad.down(&self, uint buttons) => (self.held & buttons) != 0;
// Returns true if any button in `buttons` went down since the previous frame.
fn bool Gamepad.pressed(&self, uint buttons) => (self.held & ~self.last & buttons) != 0;
// Returns true if any button in `buttons` was released since the previous frame.
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