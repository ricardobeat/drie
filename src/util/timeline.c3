// Deterministic input recording and rewind/fast-forward timeline parameterized by `Input` and
// `State`. Records each frame's input and `dt` alongside periodic state snapshots every
// `key_interval` seconds; seeking loads the nearest prior snapshot and replays forward with
// `replay = true` so the step function can suppress one-shot side effects.
//
//   alias GameTimeline = drie::timeline::Timeline{MyInput, MyState};
//   GameTimeline tl;
//   tl.init(&step, &save, &load, length: 30);
//   tl.advance(input, dt);
//   tl.seek(tl.time - 2 * dt);
module drie::timeline <Input, State>;
import std::collections::list;

// Deterministic simulation step callback; `replay` is true when called inside `seek_frame()`.
alias StepFn = fn void(Input* input, float dt, bool replay);
// Copies the current simulation state into `to` for a keyframe snapshot.
alias SaveFn = fn void(State* to);
// Restores the simulation state from keyframe snapshot `from`.
alias LoadFn = fn void(State* from);

// Recorded input and timestep for a single simulated frame.
struct Frame {
    Input input; // Input sampled for this frame.
    float dt; // Frame duration in seconds.
    double time; // Timeline timestamp when this frame started.
}

// Periodic simulation snapshot taken immediately before frame `frame` ran.
struct Keyframe {
    State state; // Saved state prior to `frame`.
    long frame; // Monotonic frame number corresponding to `state`.
    double time; // Timeline timestamp of `frame`.
}

// Rolling history of recorded frames and state keyframes supporting arbitrary time seeking.
struct Timeline {
    StepFn step; // Deterministic step callback.
    SaveFn save; // Snapshot capture callback.
    LoadFn load; // Snapshot restore callback.
    double length; // Maximum history duration retained in seconds.
    double key_interval; // Minimum seconds between saved keyframes (caps seek replay work).
    List{Frame} frames; // Recorded frames; `frames[i]` is frame number `first + i`.
    List{Keyframe} keys; // Ordered snapshots covering `[first, end()]`.
    long first; // Frame number of `frames[0]`.
    long cursor; // Next frame number to run (`end()` while live).
    double time; // Current timestamp at `cursor`.
    bool replaying; // True while `seek_frame()` is replaying recorded frames.
}

// Initializes the timeline with callbacks, history `length` (seconds), and `key_interval`.
fn void Timeline.init(&self, StepFn step, SaveFn save, LoadFn load, double length = 30, double key_interval = 0.25) {
    *self = { .step = step, .save = save, .load = load, .length = length, .key_interval = key_interval };
    self.frames.allocator = mem;
    self.keys.allocator = mem;
}

// Frees the internal frame and keyframe lists.
fn void Timeline.free(&self) {
    self.frames.free();
    self.keys.free();
}

// Clears all recorded history and restarts the timeline clock at `time` from the current state.
fn void Timeline.reset(&self, double time = 0) {
    self.frames.clear();
    self.keys.clear();
    self.first = 0;
    self.cursor = 0;
    self.time = time;
}

// Returns the frame number one past the newest recorded frame.
fn long Timeline.end(&self) => self.first + (long)self.frames.len();

// Returns true when `cursor` is at `end()` (recording live without truncating).
fn bool Timeline.is_live(&self) => self.cursor == self.end();

// Returns the timestamp of the oldest retained frame, or `time` if history is empty.
fn double Timeline.start_time(&self) => self.frames.len() ? self.frames[0].time : self.time;

// Returns the timestamp at the end of the newest retained frame, or `time` if history is empty.
fn double Timeline.end_time(&self) {
    if (!self.frames.len()) return self.time;
    Frame* last = &self.frames[self.frames.len() - 1];
    return last.time + last.dt;
}

// Returns the start timestamp of `frame` (or `end_time()` for `frame >= end()`).
fn double Timeline.frame_time(&self, long frame) {
    if (frame >= self.end()) return self.end_time();
    return self.frames[(sz)(frame - self.first)].time;
}

// Runs one frame of new input. If the cursor was moved back, what came after it is dropped
// first, like recording over a tape.
fn void Timeline.advance(&self, Input input, float dt) {
    if (!self.is_live()) self.truncate();
    if (!self.keys.len() || self.time - self.keys[self.keys.len() - 1].time >= self.key_interval) {
        Keyframe k = { .frame = self.cursor, .time = self.time };
        self.save(&k.state);
        self.keys.push(k);
    }
    self.frames.push({ .input = input, .dt = dt, .time = self.time });
    Frame* f = &self.frames[self.frames.len() - 1];
    self.step(&f.input, dt, false);
    self.cursor++;
    self.time += dt;
    self.trim();
}

// Drops history older than `length`, keeping a snapshot at or before the oldest frame.
fn void Timeline.trim(&self) {
    sz drop_keys = 0;
    while (drop_keys + 1 < self.keys.len() && self.keys[drop_keys + 1].time <= self.time - self.length) drop_keys++;
    if (!drop_keys) return;
    for (sz i = 0; i < drop_keys; i++) self.keys.remove_at(0);
    long keep_from = self.keys[0].frame;
    sz drop_frames = (sz)(keep_from - self.first);
    if (!drop_frames) return;
    sz n = self.frames.len();
    for (sz i = drop_frames; i < n; i++) self.frames[i - drop_frames] = self.frames[i];
    for (sz i = 0; i < drop_frames; i++) self.frames.pop()!!;
    self.first = keep_from;
}

// Moves to frame number `frame` (clamped to the history kept), replaying as needed.
fn void Timeline.seek_frame(&self, long frame) {
    if (frame < self.first) frame = self.first;
    if (frame > self.end()) frame = self.end();
    if (frame == self.cursor) return;
    if (frame < self.cursor) {
        // the last snapshot at or before the target
        sz k = 0;
        while (k + 1 < self.keys.len() && self.keys[k + 1].frame <= frame) k++;
        Keyframe* key = &self.keys[k];
        self.load(&key.state);
        self.cursor = key.frame;
        self.time = key.time;
    }
    self.replaying = true;
    while (self.cursor < frame) {
        Frame* f = &self.frames[(sz)(self.cursor - self.first)];
        self.step(&f.input, f.dt, true);
        self.time = f.time + f.dt;
        self.cursor++;
    }
    self.replaying = false;
}

// Moves to the last frame that starts at or before `time` (clamped to the history kept).
fn void Timeline.seek(&self, double time) {
    long lo = self.first;
    long hi = self.end();
    if (time >= self.end_time()) {
        self.seek_frame(hi);
        return;
    }
    // largest frame whose start time is <= time
    while (lo < hi) {
        long mid = (lo + hi + 1) / 2;
        if (self.frame_time(mid) <= time) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    self.seek_frame(lo);
}

// Forgets the frames after the cursor: the next advance() records from here.
fn void Timeline.truncate(&self) {
    long n = self.cursor - self.first;
    while ((long)self.frames.len() > n) self.frames.pop()!!;
    while (self.keys.len() && self.keys[self.keys.len() - 1].frame > self.cursor) self.keys.pop()!!;
}