// Low-level native game controller polling backed by `platform/gamectl.c` (Apple GameController
// on macOS) and `platform/gamectl_win.c` (XInput on Windows). Higher-level dead-zone handling,
// edge detection, touch merging, and key synthesis live in `drie::input`.
module drie::gamectl @link(env::DARWIN, "GameController.framework", "Foundation.framework");

// Raw snapshot of the active game controller returned by `gamectl::poll()`.
struct GcState {
    bool connected; // True when a controller is currently connected.
    float left_x; // Raw left stick X axis in [-1, 1], right positive.
    float left_y; // Raw left stick Y axis in [-1, 1], up positive.
    float right_x; // Raw right stick X axis in [-1, 1], right positive.
    float right_y; // Raw right stick Y axis in [-1, 1], up positive.
    float l2; // Left analog trigger in [0, 1].
    float r2; // Right analog trigger in [0, 1].
    CUInt buttons; // Bitmask of held buttons matching `input::PAD_*` bit positions.
    char[64] name; // Null-terminated controller name from the OS.
}

// Polls the active controller (most recently used, or first connected) into `s`.
// Returns false when no controller is connected.
extern fn bool poll(GcState* s) @cname("gc_poll");