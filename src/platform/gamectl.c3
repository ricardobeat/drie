// The current game controller, through Apple's GameController framework (see gamepad.c3
// for the friendlier side).
module drie::gamectl @link(env::DARWIN, "GameController.framework", "Foundation.framework");

struct GcState {
    bool connected;
    float left_x;
    float left_y;
    float right_x;
    float right_y;
    float l2;
    float r2;
    CUInt buttons;
    char[64] name;
}

// Fills `s` from the current controller (the last one used, else the first connected).
// False when there is none.
extern fn bool poll(GcState* s) @cname("gc_poll");