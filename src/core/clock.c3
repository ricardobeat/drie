// Monotonic frame clock backed by sokol_time (`stm`), reporting frame delta and total elapsed
// time in seconds. `Clock` is a plain value struct with no heap allocations; the first `clock()`
// call initializes `stm::setup()` once. Used by animation loops and `Renderer` timing.
module drie;
import std::math;
import sokol;

// Monotonic timer tracking per-frame deltas and accumulated runtime in seconds.
struct Clock {
    // Raw `stm` tick recorded when the clock was created.
    ulong start_time;
    // Raw `stm` tick recorded on the most recent `get_delta()` call.
    ulong old_time;
    // Total seconds accumulated across `get_delta()` calls while running.
    double elapsed_time;
    // When false, `get_delta()` returns 0 without advancing `elapsed_time`.
    bool running;
}

bool stm_ready @local;

// Creates a running `Clock` starting now, initializing `stm` on the first call.
fn Clock clock() {
    if (!stm_ready) {
        stm::setup();
        stm_ready = true;
    }
    ulong now = stm::now();
    return { .start_time = now, .old_time = now, .running = true };
}

// Returns seconds since the last `get_delta()` (or since creation) and adds it to
// `elapsed_time`. Returns 0 when `running` is false.
fn double Clock.get_delta(&self) {
    if (!self.running) return 0;
    ulong now   = stm::now();
    double diff = stm::sec(stm::diff(now, self.old_time));
    self.old_time = now;
    self.elapsed_time += diff;
    return diff;
}

// Returns total seconds accumulated while running. Calls `get_delta()`, advancing `old_time`.
fn double Clock.get_elapsed_time(&self) {
    self.get_delta();
    return self.elapsed_time;
}