// Clock, like three's: seconds between frames and since the start.
module drie;
import std::math;
import sokol;

struct Clock {
    ulong start_time;
    ulong old_time;
    double elapsed_time;
    bool running;
}

bool stm_ready @local;

fn Clock clock() {
    if (!stm_ready) {
        stm::setup();
        stm_ready = true;
    }
    ulong now = stm::now();
    return { .start_time = now, .old_time = now, .running = true };
}

// Seconds since the last call to get_delta (or since the clock was created).
fn double Clock.get_delta(&self) {
    if (!self.running) return 0;
    ulong now   = stm::now();
    double diff = stm::sec(stm::diff(now, self.old_time));
    self.old_time = now;
    self.elapsed_time += diff;
    return diff;
}

// Total seconds the clock has been running.
fn double Clock.get_elapsed_time(&self) {
    self.get_delta();
    return self.elapsed_time;
}