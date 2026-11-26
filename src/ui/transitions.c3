// Full-screen fade transitions drawn on top of the frame by `ui::draw()`. Advance `screen_fade`
// with `update(dt)` every frame; set `custom = true` to apply `level` in your own shader instead.
// 
//   transitions::screen_fade.out(0.6, &restart);  // fade to black, then call restart()
//   transitions::screen_fade.in(0.4);             // fade back to the scene
//   transitions::screen_fade.update(dt);
module drie::transitions;
import std::math;
import drie::ui;

// Completion callback invoked when a `Fade` reaches its target level.
alias FadeCallback = fn void();

// Linear fade state between 0 (clear) and 1 (black).
struct Fade {
    float level; // Current opacity in [0, 1] (0 = scene, 1 = black).
    float target; // Destination opacity (0 or 1).
    float speed; // Opacity change per second (`1 / seconds`).
    FadeCallback on_done; // One-shot callback invoked once `level == target`.
    bool custom; // Skips automatic drawing in `ui::draw()` when true so the caller draws `level`.
}

// Global screen fade drawn by `ui::draw()` unless `screen_fade.custom` is set.
Fade screen_fade;

// Fades toward black over `seconds` (<= 0 snaps immediately), then invokes `done` if non-null.
fn void Fade.out(&self, float seconds, FadeCallback done = null) {
    self.target = 1;
    self.speed = seconds > 0 ? 1 / seconds : 1e9f;
    self.on_done = done;
}

// Fades back to the scene over `seconds` (<= 0 snaps immediately), then invokes `done` if non-null.
fn void Fade.in(&self, float seconds = 0, FadeCallback done = null) {
    self.target = 0;
    self.speed = seconds > 0 ? 1 / seconds : 1e9f;
    self.on_done = done;
    if (seconds <= 0) self.level = 0;
}

// Returns true while the fade is still moving or has a pending completion callback.
fn bool Fade.active(&self) => self.level != self.target || self.on_done != null;

// Advances `level` toward `target` by `dt` seconds and fires `on_done` upon arrival.
fn void Fade.update(&self, float dt) {
    float step = self.speed * dt;
    if (self.level < self.target) self.level = math::min(self.level + step, self.target);
    if (self.level > self.target) self.level = math::max(self.level - step, self.target);
    if (self.level == self.target && self.on_done) {
        // cleared first: the callback may start another fade
        FadeCallback cb = self.on_done;
        self.on_done = null;
        cb();
    }
}

// Returns true when `ui::draw()` should draw an active transition overlay.
fn bool visible() => !screen_fade.custom && screen_fade.level > 0;

// Draws active transitions over a `cw` x `ch` `sokol_gl` canvas (called from `ui::draw()`).
fn void draw(float cw, float ch) {
    if (!screen_fade.custom && screen_fade.level > 0) ui::rect(0, 0, cw, ch, 0, 0, 0, screen_fade.level);
}