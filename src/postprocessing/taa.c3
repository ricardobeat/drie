module drie::postprocessing;
import std::collections::list;
import sokol, drie;

// Temporal antialiasing, like three's TAARenderPass but over time instead of in one frame:
// the camera is shifted by a fraction of a pixel each frame (jitter()), and render() blends
// each new frame of a target rendered with MRT into the frames before, found again with the
// target's motion vectors. The result (linear light, as the target's) is in texture().
//
//   TAAPass* taa = postprocessing::taa_pass();
//   // each frame:
//   taa.jitter(camera);                 // before rendering
//   renderer.set_render_target(scene_rt); renderer.render(scene, camera);
//   taa.render(scene_rt);
//   output.set_source(taa.texture());
//
// With `width` and `height` larger than the target's, the picture is that large: temporal
// upscaling (as UE's TAAU and FSR 2 do), which gets most of a picture rendered at that size out
// of frames rendered at the target's. More shifts (`jitter_samples`) then put a sample near every
// pixel of it: about 8 times the ratio of the two sizes' pixels.
struct TAAPass {
    inline Pass pass;
    // the picture's size (0: the target's)
    int width;
    int height;
    // the share of each new frame in the average: less is smoother and slower to follow
    float blend;
    // how far the history may stray from the new frame's neighbourhood, in standard
    // deviations of its colors: more keeps more of it (smoother, more ghosting)
    float clip_width;
    // the Halton sequence's length: the number of shifts the camera goes round
    int jitter_samples;
    // start over with the next frame (after a cut, or when the camera jumps)
    bool reset;
    // the camera's shift for the frame being rendered (jitter()), in its pixels
    Vector2 shift;
    PingPong history;
    ShaderPass resolve;
    int frame;
}

fn TAAPass* taa_pass() {
    TAAPass* p = mem::new(TAAPass);
    *p = {
        .render_fn = &taa_pass_render,
        .enabled = true,
        .needs_swap = false,
        .blend = 0.1,
        .clip_width = 1.25,
        .jitter_samples = 8,
        .reset = true
    };
    p.resolve.init(taa_shader_desc(sg::query_backend()), "drie-taa");
    p.resolve.set_sampler(SMP_TAA_SMP, linear_sampler());
    p.resolve.set_sampler(SMP_TAA_POINT_SMP, nearest_sampler());
    return p;
}

// The i-th number of the Halton sequence of `base`, in 0..1.
fn float halton(int i, int base) {
    float f = 1;
    float r = 0;
    while (i > 0) {
        f /= base;
        r += f * (i % base);
        i /= base;
    }
    return r;
}

// Shifts the camera for the next frame (Halton 2, 3: within half a pixel of the middle); `on`
// false leaves it where it is (and the picture is made as from frames that were not shifted).
fn void TAAPass.jitter(&self, Camera* camera, bool on = true) {
    int i = self.frame % (self.jitter_samples > 0 ? self.jitter_samples : 1) + 1;
    self.shift = on ? (Vector2) { halton(i, 2) - 0.5f, halton(i, 3) - 0.5f } : (Vector2) { 0, 0 };
    camera.jitter = self.shift;
}

// Blends what `target` (rendered with MRT, with the jittered camera) shows now into the history.
// With `color`, that picture instead, on the target's motion vectors: what came of the target's
// after a pass of its own (a stochastic depth of field, whose noise the average evens out).
fn void TAAPass.render(&self, RenderTarget* target, SgView color = {}) {
    if (!target.mrt) abort("drie: TAAPass needs a target with MRT (for its motion vectors)");
    int w = self.width > 0 ? self.width : target.width;
    int h = self.height > 0 ? self.height : target.height;
    if (!self.history.targets[0]) self.history = ping_pong(w, h, RGBA16F);
    RenderTarget* last = self.history.read();
    if (last.width != w || last.height != h) {
        self.history.set_size(w, h);
        self.reset = true;
    }
    bool up = w != target.width || h != target.height;
    // the shift along the target's rows and columns: the camera's is in clip space, whose y runs
    // up, and row 0 is the top but in GL
    Vector2 shift = { self.shift.x, drie::depth_zero_to_one() ? -self.shift.y : self.shift.y };
    ShaderPass* r = &self.resolve;
    r.set_texture(VIEW_TAA_CURRENT, color.id ? color : target.texture);
    r.set_texture(VIEW_TAA_HISTORY, self.history.read().texture);
    r.set_texture(VIEW_TAA_VELOCITY, target.velocity_texture);
    TaaParams p = {
        .size = { w, h, self.reset ? 1 : 0, self.blend },
        .tune = { self.clip_width, up ? 1 : 0, 0, 0 },
        .source = { target.width, target.height, shift.x, shift.y },
    };
    r.set_uniforms(UB_TAA_PARAMS, &p);
    r.render(self.history.write());
    self.history.swap();
    self.reset = false;
    self.frame++;
}

// The antialiased picture, after render().
fn SgView TAAPass.texture(&self) => self.history.read().texture;

fn void TAAPass.dispose(&self) {
    self.resolve.dispose();
    foreach (t : self.history.targets) if (t) t.dispose();
}

// In an EffectComposer, after a RenderPass with `mrt` set (call jitter() before the composer
// renders): the next pass reads the antialiased picture.
fn void taa_pass_render(
    Pass* pass,
    EffectComposer* composer,
    RenderTarget* write,
    RenderTarget* read,
    float delta
) @local {
    TAAPass* self = (TAAPass*)pass;
    self.render(read);
    composer.read_buffer = self.history.read();
}