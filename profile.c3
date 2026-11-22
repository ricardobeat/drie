module drie::profile;
import sokol, std::io;

// CPU time per stage of a frame, for finding where it goes (`build/scene --bench --profile`).
// Off unless `profile::enabled` is set: then start/stop cost a clock read each.
//
//     ulong t = profile::start();
//     ...
//     profile::stop("collect", t);
//
// report prints the totals divided by a frame count, in the order the stages first ran.

bool enabled;

struct ProfSlot @local {
    String name;
    ulong ticks;
}

ProfSlot[48] prof_slots @local;
int prof_count @local;
// time spent in the first begin_pass of each frame: sokol acquires the frame's command buffer
// there, waiting for a frame in flight to finish (and, for the window pass, for a drawable)
ulong prof_wait @local;
bool prof_frame_started @local;

fn ulong start() => enabled ? stm::now() : 0;

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

// sg::write_buffer_transient, timing the frame's first GPU call while profiling (on Metal,
// sokol acquires the in-flight frame semaphore on the first buffer write or pass).
fn void write_buffer_transient(SgWriteBufferDesc* desc) @inline {
    if (!enabled || prof_frame_started) {
        sg::write_buffer_transient(desc);
        return;
    }
    prof_frame_started = true;
    ulong t = stm::now();
    sg::write_buffer_transient(desc);
    prof_wait += stm::since(t);
}

// sg::begin_pass, timing the frame's first pass while profiling. drie calls this and commit()
// in place of sokol's.
fn void begin_pass(SgPass* pass) @inline {
    if (!enabled || prof_frame_started) {
        sg::begin_pass(pass);
        return;
    }
    prof_frame_started = true;
    ulong t = stm::now();
    sg::begin_pass(pass);
    prof_wait += stm::since(t);
}

fn void commit() @inline {
    sg::commit();
    prof_frame_started = false;
}

// Milliseconds the CPU has waited for the GPU since the last reset.
fn double gpu_wait_ms() => stm::ms(prof_wait);

fn void reset() {
    for (int i = 0; i < prof_count; i++) prof_slots[i].ticks = 0;
    prof_wait = 0;
}

fn void report(int frames) {
    for (int i = 0; i < prof_count; i++) {
        io::printfn("    %-22s %6.3f ms", prof_slots[i].name, stm::ms(prof_slots[i].ticks) / frames);
    }
    io::printfn("    %-22s %6.3f ms", "(waiting for the GPU)", gpu_wait_ms() / frames);
}