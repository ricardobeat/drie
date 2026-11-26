// Temporal antialiasing and upscaling pass (`TAAPass`, `shaders/taa.glsl`) for linear HDR
// render targets with `RenderTarget.mrt` (velocity buffer). Call `jitter()` before rendering
// each frame to apply a Halton (2, 3) subpixel offset to the camera, then call `render()` to
// reproject and variance-clip the `RGBA16F` ping-pong history buffer; read the resolved linear
// texture via `texture()`. Setting `width` and `height` larger than the input target enables
// temporal upscaling. Call `dispose()` to free history targets and shaders.

module drie::postprocessing;
import std::collections::list;
import sokol, drie;

// ---- TAAPass

// Motion-reprojected temporal antialiasing and upscaling pass operating in linear light.
//
//   TAAPass* taa = postprocessing::taa_pass();
//   // Each frame:
//   taa.jitter(camera);
//   renderer.set_render_target(scene_rt);
//   renderer.render(scene, camera);
//   taa.render(scene_rt);
//   output.set_source(taa.texture());
//
// When `width` and `height` exceed the input target's dimensions, increase `jitter_samples` to
// roughly `8 * (out_pixels / in_pixels)` for even subpixel coverage.
struct TAAPass {
    inline Pass pass;
    int width; // Output width in pixels (`0` matches the input render target).
    int height; // Output height in pixels (`0` matches the input render target).
    float blend; // Weight of the current frame in the exponential average (`0..1`, default `0.1`).
    float clip_width; // Neighborhood variance clip half-width in standard deviations (higher = smoother, more ghosting).
    int jitter_samples; // Length of the Halton (2, 3) camera jitter cycle (default 8).
    bool reset; // Discards history on the next `render()` (set after cuts or camera teleports), then clears.
    Vector2 shift; // Subpixel camera offset applied by the latest `jitter()` call (clip-space Y-up pixels).
    PingPong history; // `RGBA16F` ping-pong temporal accumulation buffers.
    ShaderPass resolve; // Temporal reprojection and variance-clipping pass (`shaders/taa.glsl`).
    int frame; // Monotonic frame counter indexing the Halton jitter sequence.
}

// Allocates a `TAAPass` with default `blend = 0.1`, `clip_width = 1.25`, and `8` jitter samples.
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

// Returns the `i`-th element of the radical-inverse Halton sequence in `base`, within `[0, 1)`.
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

// Applies the next Halton (2, 3) subpixel offset (`[-0.5, 0.5)` px) to `camera.jitter` before
// rendering. Pass `on = false` to zero the jitter.
fn void TAAPass.jitter(&self, Camera* camera, bool on = true) {
    int i = self.frame % (self.jitter_samples > 0 ? self.jitter_samples : 1) + 1;
    self.shift = on ? (Vector2) { halton(i, 2) - 0.5f, halton(i, 3) - 0.5f } : (Vector2) { 0, 0 };
    camera.jitter = self.shift;
}

// Reprojects `history` with `target.velocity_texture` and blends in `target.texture` (or
// optional post-processed `color`, such as `BokehPass` output sharing `target`'s motion vectors).
// Aborts if `target` was created without `mrt = true`.
fn void TAAPass.render(&self, RenderTarget* target, SgView color = { }) {
    if (!target.mrt) abort("drie: TAAPass needs a target with MRT (for its motion vectors)");
    int w = self.width > 0 ? self.width : target.width;
    int h = self.height > 0 ? self.height : target.height;
    if (!self.history.targets[0]) self.history = ping_pong(w, h, RGBA16F);
    RenderTarget* last = self.history.read();
    if (last.width != w || last.height != h) {
        self.history.set_size(w, h);
        self.reset = true;
    }
    bool up       = w != target.width || h != target.height;
    // Convert clip-space Y-up jitter to texture-space row direction (flipped on top-down backends).
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

// Returns the latest resolved linear HDR texture view after `render()`.
fn SgView TAAPass.texture(&self) => self.history.read().texture;

// Disposes the resolve shader pass and ping-pong history targets.
fn void TAAPass.dispose(&self) {
    self.resolve.dispose();
    foreach (t : self.history.targets) if (t) t.dispose();
}

// `EffectComposer` entry point: resolves `read` (from an MRT `RenderPass`, with `jitter()`
// called beforehand) and sets `composer.read_buffer` to the latest history target.
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