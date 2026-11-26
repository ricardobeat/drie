// Non-stalling per-pass GPU timestamp profiler backed by `platform/gputimer*.c`. Times each pass
// started through `profile::begin_pass()`, resolves completed frames after `SLOTS` frames in
// flight, and aggregates milliseconds by pass label. Metal samples pass stage boundaries via
// `lib/sokol.c3l/LOCAL_PATCHES`; D3D11 and GL timestamp pass starts and frame end. Set
// `DRIE_GPU_TIMER=<n>` in the environment to auto-enable and print every `n` frames (1 => 120).
//
//   gpu_timer::enabled = true;
//   gpu_timer::report();
module drie::gpu_timer;
import sokol, std::io, std::os::env, std::collections::list;

// Number of in-flight frame query ring slots (must match `platform/gputimer*.c`).
const int SLOTS = 4;
// Maximum number of GPU passes timed in a single frame.
const int MAX_PASSES = 128;
// Maximum number of distinct pass labels tracked simultaneously.
const int MAX_NAMES = 96;

// Enables GPU pass timestamping when true; also activated automatically by `DRIE_GPU_TIMER`.
bool enabled;
// Prints and resets stats every `report_every` resolved frames when positive (0 disables).
int report_every;

extern fn CInt atexit(void* f) @cname("atexit") @local;
extern fn bool gt_init(void* a, void* b) @cname("gt_init") @local;
extern fn void gt_pass_begin(CInt slot, CInt index) @cname("gt_pass_begin") @local;
extern fn bool gt_frame_end(CInt slot) @cname("gt_frame_end") @local;
extern fn bool gt_resolve(CInt slot, CInt count, double* ms) @cname("gt_resolve") @local;

struct Slot @local
{
    bool busy;              // timed, waiting to be resolved
    int count;              // passes timed
    char[MAX_PASSES] names; // name index of each pass
}

struct Totals @local
{
    String name;
    double ms;       // since reset
    double last_ms;  // in the last resolved frame
    double max_ms;
}

bool gt_checked @local;
bool gt_ok @local;
Slot[SLOTS] gt_slots @local;
int gt_cur @local = -1;    // the slot of the frame being encoded (-1: not timed)
int gt_next @local;
Totals[MAX_NAMES] gt_names @local;
int gt_name_count @local;
int gt_frames @local;     // resolved frames since reset
int gt_since_report @local;
double gt_frame_ms @local; // sum of the passes of the last resolved frame

fn bool init() @local {
    if (gt_checked) return gt_ok;
    gt_checked = true;
    SgBackend b = sg::query_backend();
    switch (b) {
        case METAL_MACOS:
        case METAL_IOS:
        case METAL_SIMULATOR:
            gt_ok = gt_init(sg::mtl_device(), null);
        case D3D11:
            gt_ok = gt_init(sg::d3d11_device(), sg::d3d11_device_context());
        case GLCORE:
            gt_ok = gt_init(null, null);
        default:
            gt_ok = false;
    }
    if (!gt_ok) io::eprintfn("drie: gpu timer: this GPU or backend has no pass timestamps");
    return gt_ok;
}

fn void check_env() @local {
    static bool done;
    if (done) return;
    done = true;
    if (try v = env::tget_var("DRIE_GPU_TIMER")) {
        if (v == "" || v == "0") return;
        enabled = true;
        int n = v.to_int() ?? 1;
        report_every = n <= 1 ? 120 : n;
        atexit(&report_at_exit);
    }
}

fn char name_index(ZString label) @local {
    String s = label ? label.str_view() : "(unlabeled)";
    for (int i = 0; i < gt_name_count; i++) if (gt_names[i].name == s) return (char)i;
    if (gt_name_count == MAX_NAMES) return (char)(MAX_NAMES - 1);
    gt_names[gt_name_count] = { .name = s.copy(mem) };
    return (char)gt_name_count++;
}

// Records a pass start timestamp right before `sg::begin_pass()` (called by `profile::begin_pass()`).
fn void before_pass(SgPass* pass) {
    check_env();
    if (!enabled) return;
    if (gt_cur < 0) {
        // the frame's first pass: take a free slot, or don't time this frame
        if (!init()) return;
        poll();
        if (gt_slots[gt_next].busy) return;
        gt_cur = gt_next;
        gt_next = (gt_next + 1) % SLOTS;
        gt_slots[gt_cur].count = 0;
    }
    Slot* s = &gt_slots[gt_cur];
    if (s.count >= MAX_PASSES) return;
    s.names[s.count] = name_index(pass.label);
    gt_pass_begin(gt_cur, s.count);
    s.count++;
}

// Closes the active frame query slot immediately before `sg::commit()` (called by `profile::commit()`).
fn void before_commit() {
    if (gt_cur < 0) return;
    gt_slots[gt_cur].busy = gt_frame_end(gt_cur);
    gt_cur = -1;
}

// Reads back the frames the GPU has finished (before_pass does it each frame).
fn void poll() {
    if (!gt_ok) return;
    double[MAX_PASSES] ms;
    for (int k = 0; k < SLOTS; k++) {
        int i = (gt_next + k) % SLOTS; // oldest first
        Slot* s = &gt_slots[i];
        if (!s.busy || i == gt_cur) continue;
        if (!gt_resolve(i, s.count, &ms)) continue;
        s.busy = false;
        for (int n = 0; n < gt_name_count; n++) gt_names[n].last_ms = 0;
        double total = 0;
        for (int p = 0; p < s.count; p++) {
            if (ms[p] < 0) continue;
            Totals* t = &gt_names[s.names[p]];
            t.last_ms += ms[p];
            t.ms += ms[p];
            total += ms[p];
        }
        for (int n = 0; n < gt_name_count; n++) if (gt_names[n].last_ms > gt_names[n].max_ms) gt_names[n].max_ms = gt_names[n].last_ms;
        gt_frame_ms = total;
        gt_frames++;
        if (report_every > 0 && ++gt_since_report >= report_every) {
            report();
            reset();
        }
    }
}

fn void report_at_exit() @local {
    if (gt_frames) report();
}

// GPU ms of the passes labelled `name` in the last frame read back (0 when none).
fn double last_ms(String name) {
    for (int i = 0; i < gt_name_count; i++) if (gt_names[i].name == name) return gt_names[i].last_ms;
    return 0;
}

// GPU ms of all the timed passes of the last frame read back.
fn double last_frame_ms() => gt_frame_ms;

// Frames read back since the last reset.
fn int frames() => gt_frames;

// Clears accumulated totals, peak timings, and frame counters while retaining pass labels.
fn void reset() {
    for (int i = 0; i < gt_name_count; i++) {
        gt_names[i].ms = 0;
        gt_names[i].max_ms = 0;
    }
    gt_frames = 0;
    gt_since_report = 0;
}

// Prints the GPU ms per frame of each pass label (mean and worst) since the last reset.
fn void report() {
    if (!gt_frames) {
        io::printfn("drie: gpu timer: no frames read back%s", enabled ? "" : " (gpu_timer::enabled is off)");
        return;
    }
    io::printfn("drie: gpu ms per pass, mean of %d frames (worst):", gt_frames);
    double total = 0;
    for (int i = 0; i < gt_name_count; i++) {
        Totals* t = &gt_names[i];
        if (t.ms <= 0) continue;
        io::printfn("    %-28s %7.3f  (%.3f)", t.name, t.ms / gt_frames, t.max_ms);
        total += t.ms;
    }
    io::printfn("    %-28s %7.3f", "total", total / gt_frames);
}
