// Per-stage CPU frame profiler and `sokol_gfx` pass wrapper. Inactive unless `profile::enabled`
// is true; when enabled, `start()` and `stop()` accumulate wall ticks per named stage (up to 48
// distinct names) and `begin_pass()` measures time spent waiting on the GPU in the first pass of
// each frame. Also hooks `drie::gpu_timer` into every pass and commit.
//
//   profile::enabled = true;
//   ulong t = profile::start();
//   ...
//   profile::stop("collect", t);
//   profile::report(frames);
module drie::profile;
import sokol, std::io, drie::gpu_timer;

// Enables CPU stage timing and first-pass GPU wait measurement when true.
bool enabled;

struct ProfSlot @local {
    String name;
    ulong ticks;
}

ProfSlot[48] prof_slots @local;
int prof_count @local;
// Ticks spent inside the first `sg::begin_pass()` of each frame while acquiring the command
// buffer or swapchain drawable.
ulong prof_wait @local;
bool prof_frame_started @local;

// Returns the current `sokol_time` tick counter when profiling is enabled, or 0 otherwise.
fn ulong start() => enabled ? stm::now() : 0;

// Adds elapsed ticks since `started` to the named stage bucket `name`.
fn void stop(String name, ulong started) {
    if (!enabled) return;
    ulong dt = stm::since(started);
    for (int i = 0; i < prof_count; i++) {
        if (prof_slots[i].name == name) {
            prof_slots[i].ticks += dt;
            return;
        }
    }
    if (prof_count == prof_slots.len) return;
    prof_slots[prof_count++] = { name, dt };
}

// Wraps `sg::begin_pass()`, forwarding pass labels to `drie::gpu_timer` and timing the first pass
// of each frame while `profile::enabled` is true.
fn void begin_pass(SgPass* pass) @inline {
    gpu_timer::before_pass(pass);
    if (!enabled || prof_frame_started) {
        sg::begin_pass(pass);
        return;
    }
    prof_frame_started = true;
    ulong t = stm::now();
    sg::begin_pass(pass);
    prof_wait += stm::since(t);
}

// Wraps `sg::commit()`, closing GPU timer capture for the frame and resetting per-frame state.
fn void commit() @inline {
    gpu_timer::before_commit();
    sg::commit();
    prof_frame_started = false;
}

// Returns total milliseconds spent waiting on the GPU in first passes since `reset()`.
fn double gpu_wait_ms() => stm::ms(prof_wait);

// Zeroes accumulated stage ticks and GPU wait time while keeping registered stage names.
fn void reset() {
    for (int i = 0; i < prof_count; i++) prof_slots[i].ticks = 0;
    prof_wait = 0;
}

// Prints mean milliseconds per frame (dividing totals by `frames`) for each stage in first-seen
// order, followed by the GPU wait line.
fn void report(int frames) {
    for (int i = 0; i < prof_count; i++) {
        io::printfn("    %-22s %6.3f ms", prof_slots[i].name, stm::ms(prof_slots[i].ticks) / frames);
    }
    io::printfn("    %-22s %6.3f ms", "(waiting for the GPU)", gpu_wait_ms() / frames);
}