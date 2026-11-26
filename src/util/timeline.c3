// Timeline: rewind and fast-forward by replaying recorded input.
//
// A game whose step is deterministic (same state + same input + same dt = same next state) can
// go back in time without storing every frame: the timeline keeps each frame's input and dt,
// plus a snapshot of the game state every `key_interval` seconds. Seeking back loads the
// snapshot before the target and replays the frames after it; seeking forward just replays.
//
//   alias GameTimeline = drie::timeline::Timeline{MyInput, MyState};
//   GameTimeline tl;
//   tl.init(&step, &save, &load, length: 30);
//   ...every frame: tl.advance(input, dt);       // records and steps
//   ...in a rewind screen: tl.seek(tl.time - 2 * dt);
//   ...to play on from there: tl.truncate();      // forgets what came after
//
// `step` is told when it is replaying, so it can leave out side effects that should only
// happen once (sounds, logging).
module drie::timeline <Input, State>;
import std::collections::list;

alias StepFn = fn void(Input* input, float dt, bool replay);
alias SaveFn = fn void(State* to);
alias LoadFn = fn void(State* from);

struct Frame {
    Input input;
    float dt;
    double time; // when the frame started
}

struct Keyframe {
    State state; // the state before frame `frame` ran
    long frame;
    double time;
}

struct Timeline {
    StepFn step;
    SaveFn save;
    LoadFn load;
    // seconds of history kept
    double length;
    // seconds between snapshots: a seek replays up to this much
    double key_interval;
    // frames[i] is frame number first + i
    List{Frame} frames;
    List{Keyframe} keys;
    long first;
    // the next frame to run; end() while live
    long cursor;
    // time at the cursor
    double time;
    // set while seek() replays frames
    bool replaying;
}

fn void Timeline.init(&self, StepFn step, SaveFn save, LoadFn load, double length = 30, double key_interval = 0.25) {
    *self = { .step = step, .save = save, .load = load, .length = length, .key_interval = key_interval };
    self.frames.allocator = mem;
    self.keys.allocator = mem;
}

fn void Timeline.free(&self) {
    self.frames.free();
    self.keys.free();
}

// Forgets everything and starts again at `time`, from the current state.
fn void Timeline.reset(&self, double time = 0) {
    self.frames.clear();
    self.keys.clear();
    self.first = 0;
    self.cursor = 0;
    self.time = time;
}

fn long Timeline.end(&self) => self.first + (long)self.frames.len();

// Whether the cursor is at the newest frame (new input can be recorded).
fn bool Timeline.is_live(&self) => self.cursor == self.end();

fn double Timeline.start_time(&self) => self.frames.len() ? self.frames[0].time : self.time;

fn double Timeline.end_time(&self) {
    if (!self.frames.len()) return self.time;
    Frame* last = &self.frames[self.frames.len() - 1];
    return last.time + last.dt;
}

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