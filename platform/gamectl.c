// Polls the active game controller (DualSense, DualShock, Xbox, MFi) via Apple's GameController
// framework for drie::gamectl and drie::input. Compiled as Objective-C on Apple platforms and
// empty elsewhere so every target builds the same C source list.
#if defined(__APPLE__)
#import <GameController/GameController.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
    bool connected;
    float left_x, left_y;     // -1..1, up is +y
    float right_x, right_y;
    float l2, r2;             // 0..1
    unsigned int buttons;     // GC_* bits
    char name[64];
} GcState;

enum {
    GC_SOUTH = 1 << 0,  // cross / A
    GC_EAST = 1 << 1,   // circle / B
    GC_WEST = 1 << 2,   // square / X
    GC_NORTH = 1 << 3,  // triangle / Y
    GC_L1 = 1 << 4,
    GC_R1 = 1 << 5,
    GC_L2 = 1 << 6,
    GC_R2 = 1 << 7,
    GC_L3 = 1 << 8,
    GC_R3 = 1 << 9,
    GC_UP = 1 << 10,
    GC_DOWN = 1 << 11,
    GC_LEFT = 1 << 12,
    GC_RIGHT = 1 << 13,
    GC_MENU = 1 << 14,  // options / menu
    GC_VIEW = 1 << 15,  // create / view
    GC_HOME = 1 << 16,  // PS / Xbox button
};

static void bit(unsigned int* b, GCControllerButtonInput* in, unsigned int mask) {
    if (in && in.isPressed) *b |= mask;
}

bool gc_poll(GcState* s) {
    memset(s, 0, sizeof *s);
    GCController* c = GCController.current;
    if (!c || !c.extendedGamepad) {
        for (GCController* other in GCController.controllers) {
            if (other.extendedGamepad) { c = other; break; }
        }
    }
    if (!c || !c.extendedGamepad) return false;
    GCExtendedGamepad* g = c.extendedGamepad;
    s->connected = true;
    s->left_x = g.leftThumbstick.xAxis.value;
    s->left_y = g.leftThumbstick.yAxis.value;
    s->right_x = g.rightThumbstick.xAxis.value;
    s->right_y = g.rightThumbstick.yAxis.value;
    s->l2 = g.leftTrigger.value;
    s->r2 = g.rightTrigger.value;
    unsigned int b = 0;
    bit(&b, g.buttonA, GC_SOUTH);
    bit(&b, g.buttonB, GC_EAST);
    bit(&b, g.buttonX, GC_WEST);
    bit(&b, g.buttonY, GC_NORTH);
    bit(&b, g.leftShoulder, GC_L1);
    bit(&b, g.rightShoulder, GC_R1);
    bit(&b, g.leftTrigger, GC_L2);
    bit(&b, g.rightTrigger, GC_R2);
    bit(&b, g.leftThumbstickButton, GC_L3);
    bit(&b, g.rightThumbstickButton, GC_R3);
    bit(&b, g.dpad.up, GC_UP);
    bit(&b, g.dpad.down, GC_DOWN);
    bit(&b, g.dpad.left, GC_LEFT);
    bit(&b, g.dpad.right, GC_RIGHT);
    bit(&b, g.buttonMenu, GC_MENU);
    bit(&b, g.buttonOptions, GC_VIEW);
    bit(&b, g.buttonHome, GC_HOME);
    s->buttons = b;
    NSString* name = c.vendorName ?: c.productCategory;
    if (name) strlcpy(s->name, name.UTF8String, sizeof s->name);
    return true;
}
#endif // __APPLE__
