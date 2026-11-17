// Reads the first connected XInput controller (Xbox and compatible pads).
#include <windows.h>
#include <xinput.h>
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
    GC_SOUTH = 1 << 0, GC_EAST = 1 << 1, GC_WEST = 1 << 2, GC_NORTH = 1 << 3,
    GC_L1 = 1 << 4, GC_R1 = 1 << 5, GC_L2 = 1 << 6, GC_R2 = 1 << 7,
    GC_L3 = 1 << 8, GC_R3 = 1 << 9,
    GC_UP = 1 << 10, GC_DOWN = 1 << 11, GC_LEFT = 1 << 12, GC_RIGHT = 1 << 13,
    GC_MENU = 1 << 14, GC_VIEW = 1 << 15, GC_HOME = 1 << 16,
};

typedef DWORD (WINAPI *GetStateFn)(DWORD, XINPUT_STATE*);
static GetStateFn get_state;
static bool loaded;
static int last_pad = -1;
static ULONGLONG next_scan;

static float axis(SHORT v) {
    return v < 0 ? (float)v / 32768.0f : (float)v / 32767.0f;
}

bool gc_poll(GcState* s) {
    memset(s, 0, sizeof *s);
    if (!loaded) {
        loaded = true;
        HMODULE m = LoadLibraryA("xinput1_4.dll");
        if (!m) m = LoadLibraryA("xinput9_1_0.dll");
        if (m) get_state = (GetStateFn)GetProcAddress(m, "XInputGetState");
    }
    if (!get_state) return false;
    XINPUT_STATE st;
    int pad = -1;
    if (last_pad >= 0 && get_state((DWORD)last_pad, &st) == ERROR_SUCCESS) {
        pad = last_pad;
    } else if (GetTickCount64() >= next_scan) {
        // probing empty slots is slow, so look for a new pad at most twice a second
        next_scan = GetTickCount64() + 500;
        for (DWORD i = 0; i < XUSER_MAX_COUNT; i++) {
            if (get_state(i, &st) == ERROR_SUCCESS) { pad = (int)i; break; }
        }
    }
    last_pad = pad;
    if (pad < 0) return false;
    XINPUT_GAMEPAD* g = &st.Gamepad;
    s->connected = true;
    s->left_x = axis(g->sThumbLX);
    s->left_y = axis(g->sThumbLY);
    s->right_x = axis(g->sThumbRX);
    s->right_y = axis(g->sThumbRY);
    s->l2 = g->bLeftTrigger / 255.0f;
    s->r2 = g->bRightTrigger / 255.0f;
    WORD w = g->wButtons;
    unsigned int b = 0;
    if (w & XINPUT_GAMEPAD_A) b |= GC_SOUTH;
    if (w & XINPUT_GAMEPAD_B) b |= GC_EAST;
    if (w & XINPUT_GAMEPAD_X) b |= GC_WEST;
    if (w & XINPUT_GAMEPAD_Y) b |= GC_NORTH;
    if (w & XINPUT_GAMEPAD_LEFT_SHOULDER) b |= GC_L1;
    if (w & XINPUT_GAMEPAD_RIGHT_SHOULDER) b |= GC_R1;
    if (g->bLeftTrigger > 30) b |= GC_L2;
    if (g->bRightTrigger > 30) b |= GC_R2;
    if (w & XINPUT_GAMEPAD_LEFT_THUMB) b |= GC_L3;
    if (w & XINPUT_GAMEPAD_RIGHT_THUMB) b |= GC_R3;
    if (w & XINPUT_GAMEPAD_DPAD_UP) b |= GC_UP;
    if (w & XINPUT_GAMEPAD_DPAD_DOWN) b |= GC_DOWN;
    if (w & XINPUT_GAMEPAD_DPAD_LEFT) b |= GC_LEFT;
    if (w & XINPUT_GAMEPAD_DPAD_RIGHT) b |= GC_RIGHT;
    if (w & XINPUT_GAMEPAD_START) b |= GC_MENU;
    if (w & XINPUT_GAMEPAD_BACK) b |= GC_VIEW;
    s->buttons = b;
    strcpy(s->name, "XInput controller");
    return true;
}
