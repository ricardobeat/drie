// Screen transitions drawn over the whole picture by the overlay (ui::draw).
//
// screen_fade fades to and from black:
//
//   transitions::screen_fade.out(0.6, &restart);   // to black, then restart() is called
//   transitions::screen_fade.in(0.4);               // back to the picture
//   transitions::screen_fade.update(dt);            // every frame
module drie::transitions;
import std::math;
import drie::ui;

alias FadeCallback = fn void();

// A fade to and from black. `level` 0 is the picture, 1 is black. The overlay draws it over
// everything unless `custom` is set, for a game that applies `level` itself (in its own
// post-processing, say). Advance it with update() every frame.
struct Fade {
    float level;
    float target;
    float speed;
    FadeCallback on_done;
    bool custom;
}

Fade screen_fade;

// Fades to black over `seconds`, then calls `done`.
fn void Fade.out(&self, float seconds, FadeCallback done = null) {
    self.target = 1;
    self.speed = seconds > 0 ? 1 / seconds : 1e9f;
    self.on_done = done;
}

// Fades back to the picture over `seconds` (0: at once), then calls `done`.
fn void Fade.in(&self, float seconds = 0, FadeCallback done = null) {
    self.target = 0;
    self.speed = seconds > 0 ? 1 / seconds : 1e9f;
    self.on_done = done;
    if (seconds <= 0) self.level = 0;
}

fn bool Fade.active(&self) => self.level != self.target || self.on_done != null;

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

// Whether the overlay has a transition to draw.
fn bool visible() => !screen_fade.custom && screen_fade.level > 0;

// Draws the transitions into the current sgl layer, over a canvas `cw` x `ch` (ui::draw).
fn void draw(float cw, float ch) {
    if (!screen_fade.custom && screen_fade.level > 0) ui::rect(0, 0, cw, ch, 0, 0, 0, screen_fade.level);
}
