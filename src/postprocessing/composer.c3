// Post-processing pass primitives (`ShaderPass`, `RenderPass`, `PingPong`, `EffectComposer`)
// and render-pass helpers modeled after three.js's `EffectComposer`. Fullscreen passes draw a
// single triangle from `gl_VertexIndex` without a vertex buffer and lazily cache GPU pipelines
// per target format, depth format, and sample count. `EffectComposer` runs in linear `RGBA16F`
// by default and ends with `OutputPass` for tone mapping and display encoding; `end_screen()`
// draws registered overlays (`drie::add_overlay`). Call `dispose()` on passes and composers to
// release GPU shaders, pipelines, and render targets.

module drie::postprocessing;
import std::collections::list;
import sokol, drie;

// ---- Blend states & shared samplers

// Opaque replacement blend state (default for `ShaderPass`).
const SgBlendState NO_BLEND       = { };
// Multiplicative blend state (`dst.rgb * src.rgb`), preserving destination alpha.
const SgBlendState MULTIPLY_BLEND = {
    .enabled = true,
    .src_factor_rgb = DST_COLOR,
    .dst_factor_rgb = ZERO,
    .src_factor_alpha = ZERO,
    .dst_factor_alpha = ONE,
};
// Additive blend state (`src.rgb + dst.rgb`), preserving destination alpha.
const SgBlendState ADDITIVE_BLEND = {
    .enabled = true,
    .src_factor_rgb = ONE,
    .dst_factor_rgb = ONE,
    .src_factor_alpha = ZERO,
    .dst_factor_alpha = ONE,
};
// Standard straight-alpha blend state for RGB and alpha channels.
const SgBlendState ALPHA_BLEND    = {
    .enabled = true,
    .src_factor_rgb = SRC_ALPHA,
    .dst_factor_rgb = ONE_MINUS_SRC_ALPHA,
    .src_factor_alpha = ONE,
    .dst_factor_alpha = ONE_MINUS_SRC_ALPHA,
};

// Number of uniform block slots retained per `ShaderPass` (matches Sokol's 8-slot limit).
const int PASS_UNIFORM_SLOTS = 8;
// Maximum byte size cached per uniform block slot in `ShaderPass`.
const int PASS_UNIFORM_BYTES = 512;

// Attachment layout key caching `ShaderPass` pipelines per target configuration.
struct PassFormat @local {
    SgPixelFormat color;
    SgPixelFormat depth;
    int samples;
    bool screen;
}

struct PassPipeline @local {
    PassFormat format;
    SgPipeline pipeline;
}

struct PostState @local {
    SgSampler linear;
    SgSampler nearest;
    // Active pass format set by `begin_screen()`, `RenderTarget.begin()`, or `begin_over()`.
    PassFormat current;
}

PostState post_state @private;

// Returns a shared clamp-to-edge bilinear sampler created lazily on first call.
fn SgSampler linear_sampler() {
    if (!post_state.linear.id) {
        post_state.linear = sg::make_sampler(
            &&{
                .min_filter = LINEAR,
                .mag_filter = LINEAR,
                .wrap_u = CLAMP_TO_EDGE,
                .wrap_v = CLAMP_TO_EDGE,
            }
        );
    }
    return post_state.linear;
}

// Returns a shared clamp-to-edge nearest-neighbor sampler created lazily on first call.
fn SgSampler nearest_sampler() {
    if (!post_state.nearest.id) {
        post_state.nearest = sg::make_sampler(
            &&{
                .min_filter = NEAREST,
                .mag_filter = NEAREST,
                .wrap_u = CLAMP_TO_EDGE,
                .wrap_v = CLAMP_TO_EDGE,
            }
        );
    }
    return post_state.nearest;
}

fn SgColor clear_color_of(Color c, float alpha) @local {
    return { ((c >> 16) & 0xff) / 255.0f, ((c >> 8) & 0xff) / 255.0f, (c & 0xff) / 255.0f, alpha };
}

// ---- Render target & screen passes

// Begins a render pass into `self` for `ShaderPass.draw()` or custom Sokol drawing. Uses
// `clear` and `clear_alpha` when `load == CLEAR`, clears depth when attached, and pairs with
// `RenderTarget.end()`.
fn void RenderTarget.begin(
    &self,
    SgLoadAction load = CLEAR,
    Color clear = 0x000000,
    float clear_alpha = 1,
    ZString label = "drie-target"
) {
    SgPass pass = {
        .action.colors[0] = { .load_action = load, .clear_value = clear_color_of(clear, clear_alpha) },
        .label = label
    };
    pass.attachments = self.attachments();
    if (self.depth_buffer) pass.action.depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 };
    profile::begin_pass(&pass);
    post_state.current = { self.pixel_format(), self.depth_buffer ? DEPTH : NONE, self.samples, false };
}

// Begins a single-sample LOAD pass over the target's resolved color texture (`overlay_view`),
// preserving existing pixels for blended overlays (such as `MULTIPLY_BLEND` or `ADDITIVE_BLEND`).
// Pair with `RenderTarget.end()`.
fn void RenderTarget.begin_over(&self, ZString label = "drie-over") {
    SgPass pass = { .action.colors[0] = { .load_action = LOAD }, .label = label };
    pass.attachments.colors[0] = self.overlay_view;
    profile::begin_pass(&pass);
    post_state.current = { self.pixel_format(), NONE, 1, false };
}

// Ends a render pass begun with `RenderTarget.begin()` or `RenderTarget.begin_over()`.
fn void RenderTarget.end(&self) => sg::end_pass();

// Begins a swapchain pass cleared to `clear`. When `aspect > 0` (`width / height`), letterboxes
// the viewport to preserve that aspect ratio. Pair with `end_screen()`.
fn void begin_screen(Color clear = 0x000000, float aspect = 0, ZString label = "drie-screen") {
    profile::begin_pass(
        &&{
            .action.colors[0] = { .load_action = CLEAR, .clear_value = clear_color_of(clear, 1) },
            .swapchain = sglue::swapchain(),
            .label = label,
        }
    );
    post_state.current = { .screen = true };
    if (aspect > 0) {
        SlbxViewport vp = slbx::letterbox(sapp::width(), sapp::height(), &&{ .content_aspect_ratio = aspect });
        sg::apply_viewport(vp.x, vp.y, vp.width, vp.height, true);
    }
}

// Resets the viewport to the full window, draws overlays registered via `drie::add_overlay()`,
// and ends the swapchain pass begun with `begin_screen()`.
fn void end_screen() {
    sg::apply_viewport(0, 0, sapp::width(), sapp::height(), true);
    drie::draw_overlays();
    sg::end_pass();
}

// ---- ShaderPass

// Callback executed by `EffectComposer.render()` for each enabled `Pass`.
alias PassRenderFn = fn void(
    Pass* pass,
    EffectComposer* composer,
    RenderTarget* write,
    RenderTarget* read,
    float delta
);

// Base header embedded first (`inline Pass pass;`) by every pass used in `EffectComposer`.
struct Pass {
    PassRenderFn render_fn; // Per-frame execution callback invoked by `EffectComposer`.
    bool enabled; // Skipped by `EffectComposer.render()` when false.
    bool needs_swap; // When true, `EffectComposer` swaps `read_buffer` and `write_buffer` after this pass.
    bool render_to_screen; // Set by `EffectComposer` on the final enabled pass to draw to the swapchain.
}

// Fullscreen or procedural-draw shader pass caching bindings, uniforms, and per-target pipelines.
//
//   ShaderPass* blur = postprocessing::shader_pass(blur_shader_desc(sg::query_backend()), "blur");
//   blur.set_texture(VIEW_SRC, scene_target.texture);
//   blur.set_sampler(SMP_SMP, postprocessing::linear_sampler());
//   blur.set_uniforms(UB_BLUR_PARAMS, &params);
//   blur.render(blurred); // RenderTarget*, or null for the window
//
// Vertex shaders generate the fullscreen triangle from `gl_VertexIndex` without a vertex buffer:
//   vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
//   gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
struct ShaderPass {
    inline Pass pass;
    SgShader shader; // Compiled shader program owned by this pass.
    ZString label; // Debug label used for passes and lazily created pipelines.
    SgBlendState blend; // Color/alpha blend state applied to pipelines created by this pass.
    SgBindings bindings; // Texture views and samplers at sokol-shdc slot indices (`VIEW_*`, `SMP_*`).
    int input_view; // Binding slot populated with `read.texture` inside `EffectComposer` (`-1` = none).
    int input_sampler; // Sampler slot populated with `linear_sampler()` inside `EffectComposer` (`-1` = none).
    char[PASS_UNIFORM_BYTES][PASS_UNIFORM_SLOTS] uniforms; // Cached uniform block bytes per slot.
    usz[PASS_UNIFORM_SLOTS] uniform_size; // Active byte length per uniform slot (`0` = unbound).
    List{PassPipeline} pipelines; // Cached `SgPipeline` per `(color, depth, samples, screen)` format.
    int vertex_count; // Vertices per instance (`0` defaults to `3` for a fullscreen triangle).
    int instance_count; // Instance count (`0` defaults to `1`; used e.g. for instanced sprite quads).
}

// Allocates and initializes a `ShaderPass` from a sokol-shdc `SgShaderDesc`.
fn ShaderPass* shader_pass(
    SgShaderDesc desc,
    ZString label = "drie-pass",
    SgBlendState blend = NO_BLEND,
    int input_view = -1,
    int input_sampler = -1
) {
    ShaderPass* p = mem::new(ShaderPass);
    p.init(desc, label, blend, input_view, input_sampler);
    return p;
}

// Initializes an embedded `ShaderPass` in place with `desc`, `label`, `blend`, and optional
// `EffectComposer` input slots.
fn void ShaderPass.init(
    &self,
    SgShaderDesc desc,
    ZString label = "drie-pass",
    SgBlendState blend = NO_BLEND,
    int input_view = -1,
    int input_sampler = -1
) {
    self.shader = drie::make_program(desc);
    self.label = label;
    self.blend = blend;
    self.input_view = input_view;
    self.input_sampler = input_sampler;
    self.render_fn = &shader_pass_render;
    self.enabled = true;
    self.needs_swap = true;
}

// Binds texture view `view` to shader binding `slot`.
fn void ShaderPass.set_texture(&self, int slot, SgView view) => self.bindings.views[slot] = view;
// Binds sampler `sampler` to shader binding `slot`.
fn void ShaderPass.set_sampler(&self, int slot, SgSampler sampler) => self.bindings.samplers[slot] = sampler;

// Copies the uniform struct at `data` into `slot`, reapplying it on each draw until overwritten.
macro void ShaderPass.set_uniforms(&self, int slot, data) {
    self.set_uniform_bytes(slot, (char*)data, $Typeof(*data)::size);
}

// Copies `size` raw uniform bytes from `data` into `slot` (`slot < PASS_UNIFORM_SLOTS`,
// `size <= PASS_UNIFORM_BYTES`).
fn void ShaderPass.set_uniform_bytes(&self, int slot, char* data, usz size) {
    assert(slot >= 0 && slot < PASS_UNIFORM_SLOTS
        && size <= PASS_UNIFORM_BYTES, "uniform block slot %d of %d bytes", slot, size);
    mem::copy(&self.uniforms[slot], data, (sz)size);
    self.uniform_size[slot] = size;
}

// Returns the cached `SgPipeline` matching attachment format `f`, creating one on first use.
fn SgPipeline ShaderPass.pipeline_for(&self, PassFormat f) {
    foreach (p : self.pipelines) {
        PassFormat g = p.format;
        if (g.color == f.color && g.depth == f.depth && g.samples == f.samples
            && g.screen == f.screen) return p.pipeline;
    }
    SgPipelineDesc desc = { .shader = self.shader, .label = self.label };
    desc.colors[0].blend = self.blend;
    if (!f.screen) {
        desc.colors[0].pixel_format = f.color;
        desc.depth.pixel_format = f.depth;
        desc.sample_count = f.samples;
    }
    SgPipeline pip = sg::make_pipeline(&desc);
    self.pipelines.push({ f, pip });
    return pip;
}

// Issues the pass draw call inside an active pass (`begin_screen()`, `RenderTarget.begin()`, or
// `RenderTarget.begin_over()`).
fn void ShaderPass.draw(&self) {
    sg::apply_pipeline(self.pipeline_for(post_state.current));
    sg::apply_bindings(&self.bindings);
    for (int i = 0; i < PASS_UNIFORM_SLOTS; i++) {
        if (self.uniform_size[i]) sg::apply_uniforms(i, &&{ .ptr = &self.uniforms[i], .size = self.uniform_size[i] });
    }
    sg::draw(0, self.vertex_count ?: 3, self.instance_count ?: 1);
}

// Runs `self` in its own render pass into `target` (or the swapchain with overlays when null).
// Defaults to `DONTCARE` load action because fullscreen shaders cover the whole attachment.
fn void ShaderPass.render(
    &self,
    RenderTarget* target,
    SgLoadAction load = DONTCARE,
    Color clear = 0x000000,
    float clear_alpha = 1
) {
    if (!target) {
        begin_screen(clear, label: self.label);
        self.draw();
        end_screen();
        return;
    }
    target.begin(load, clear, clear_alpha, self.label);
    self.draw();
    target.end();
}

// Draws `self` over `target`'s resolved color texture while preserving its contents
// (`RenderTarget.begin_over()`).
fn void ShaderPass.render_over(&self, RenderTarget* target) {
    target.begin_over(self.label);
    self.draw();
    target.end();
}

// Destroys all cached GPU pipelines and the shader owned by this pass.
fn void ShaderPass.dispose(&self) {
    foreach (p : self.pipelines) sg::destroy_pipeline(p.pipeline);
    self.pipelines.free();
    sg::destroy_shader(self.shader);
}

fn void shader_pass_render(
    Pass* pass,
    EffectComposer* composer,
    RenderTarget* write,
    RenderTarget* read,
    float delta
) @local {
    ShaderPass* self = (ShaderPass*)pass;
    if (self.input_view >= 0) self.bindings.views[self.input_view] = read.texture;
    if (self.input_sampler >= 0) self.bindings.samplers[self.input_sampler] = linear_sampler();
    self.render(pass.render_to_screen ? null : write);
}

// ---- RenderPass

// `EffectComposer` pass that renders `scene` from `camera` into its own multisampled depth
// render target (exposed as `composer.read_buffer`). Set `mrt = true` on an `RGBA16F` composer
// when downstream passes need normals, velocity, or indirect light (e.g., `GTAOPass`, `TAAPass`).
struct RenderPass {
    inline Pass pass;
    Scene* scene; // Scene rendered on each composer frame.
    Camera* camera; // Camera used to render `scene`.
    int samples; // MSAA sample count for `target` (default 4).
    bool mrt; // Allocates MRT attachments (normals, velocity, indirect) when true.
    RenderTarget* target; // Offscreen scene target created lazily and resized to match the composer.
}

// Allocates a `RenderPass` rendering `scene` from `camera` at `samples` MSAA.
fn RenderPass* render_pass(Scene* scene, Camera* camera, int samples = 4) {
    RenderPass* p = mem::new(RenderPass);
    *p = { .render_fn = &render_pass_render, .enabled = true, .scene = scene, .camera = camera, .samples = samples };
    return p;
}

fn void render_pass_render(
    Pass* pass,
    EffectComposer* composer,
    RenderTarget* write,
    RenderTarget* read,
    float delta
) @local {
    RenderPass* self   = (RenderPass*)pass;
    Renderer* r        = composer.renderer;
    RenderTarget* prev = r.render_target;
    if (pass.render_to_screen) {
        r.set_render_target(null);
        r.render(self.scene, self.camera);
        r.set_render_target(prev);
        return;
    }
    if (!self.target) self.target = drie::render_target(
        composer.width,
        composer.height,
        self.samples,
        composer.format,
        mrt: self.mrt
    );
    self.target.set_size(composer.width, composer.height, self.samples);
    r.set_render_target(self.target);
    r.render(self.scene, self.camera);
    r.set_render_target(prev);
    // Point the next pass directly at the rendered target without an extra copy.
    composer.read_buffer = self.target;
}

// ---- PingPong

// Pair of single-sample render targets alternating between read and write across frames (for
// temporal history, autofocus, or auto-exposure meters). Call `swap()` after writing `write()`.
struct PingPong {
    RenderTarget*[2] targets; // The two alternating render targets.
    int cur; // Index (`0` or `1`) of the most recently written target (`read()`).
}

// Allocates a `PingPong` pair of `width` x `height` targets in `format` without depth buffers.
fn PingPong ping_pong(int width, int height, SgPixelFormat format = RGBA8) {
    return {
        .targets = {
            drie::render_target(width, height, 1, format, false),
            drie::render_target(width, height, 1, format, false)
        }
    };
}

// Returns the target written most recently.
fn RenderTarget* PingPong.read(&self) => self.targets[self.cur];
// Returns the inactive target to render the next frame into.
fn RenderTarget* PingPong.write(&self) => self.targets[1 - self.cur];
// Swaps the read and write targets.
fn void PingPong.swap(&self) => self.cur = 1 - self.cur;

// Resizes both targets to `width` x `height` and resets `cur` to `0`.
fn void PingPong.set_size(&self, int width, int height) {
    foreach (t : self.targets) t.set_size(width, height, 1);
    self.cur = 0;
}

// ---- EffectComposer

// Sequential post-processing chain ping-ponging two single-sample buffers (`RGBA16F` linear by
// default; `RGBA8` triggers inline tone mapping during scene rendering).
//
//   EffectComposer* composer = postprocessing::effect_composer(renderer);
//   composer.add_pass(postprocessing::render_pass(scene, camera));
//   composer.add_pass(my_effect);
//   composer.add_pass(postprocessing::output_pass(renderer));
//
// Pipelines with non-linear data flow or persistent history can drive `ShaderPass`, `RenderTarget`,
// and `PingPong` directly instead.
struct EffectComposer {
    Renderer* renderer; // Renderer used by `RenderPass` and `OutputPass`.
    int width; // Fixed buffer width in pixels (`0` tracks `sapp::width()`).
    int height; // Fixed buffer height in pixels (`0` tracks `sapp::height()`).
    SgPixelFormat format; // Pixel format of `buffers` (`RGBA16F` for linear HDR).
    RenderTarget*[2] buffers; // Ping-pong offscreen buffers allocated lazily on first `render()`.
    RenderTarget* read_buffer; // Buffer holding the input image for the current pass.
    RenderTarget* write_buffer; // Buffer receiving output from the current pass.
    List{Pass*} passes; // Ordered list of passes executed by `render()`.
    bool render_to_screen; // When true, routes the last enabled pass to the swapchain.
}

// Allocates an `EffectComposer` using `format` (`width`/`height` of `0` track the window size).
fn EffectComposer* effect_composer(Renderer* renderer, int width = 0, int height = 0, SgPixelFormat format = RGBA16F) {
    EffectComposer* c = mem::new(EffectComposer);
    *c = { .renderer = renderer, .width = width, .height = height, .format = format, .render_to_screen = true };
    return c;
}

// Appends `pass` to the end of the composer chain.
fn void EffectComposer.add_pass(&self, Pass* pass) => self.passes.push(pass);

// Removes the first occurrence of `pass` from the composer chain without disposing it.
fn void EffectComposer.remove_pass(&self, Pass* pass) {
    foreach (i, p : self.passes) {
        if (p == pass) {
            self.passes.remove_at(i);
            return;
        }
    }
}

// Swaps `read_buffer` and `write_buffer` between the composer's two internal ping-pong targets.
fn void EffectComposer.swap_buffers(&self) {
    self.read_buffer = self.write_buffer;
    self.write_buffer = self.read_buffer == self.buffers[0] ? self.buffers[1] : self.buffers[0];
}

// Sets an explicit buffer size and resizes any already-allocated internal targets.
fn void EffectComposer.set_size(&self, int width, int height) {
    self.width = width;
    self.height = height;
    foreach (b : self.buffers) if (b) b.set_size(width, height, 1);
}

// Executes all enabled passes in order and calls `profile::commit()` at the end. Call once per
// frame in place of `Renderer.render()`.
fn void EffectComposer.render(&self, float delta = 0) {
    int w = self.width ?: sapp::width();
    int h = self.height ?: sapp::height();
    if (!self.buffers[0]) {
        for (int i = 0; i < 2; i++) self.buffers[i] = drie::render_target(w, h, 1, self.format, false);
    }
    foreach (b : self.buffers) b.set_size(w, h, 1);
    int tw = self.width;
    int th = self.height;
    self.width = w;
    self.height = h;
    self.read_buffer = self.buffers[0];
    self.write_buffer = self.buffers[1];
    sz last = -1;
    foreach (i, p : self.passes) if (p.enabled) last = (sz)i;
    foreach (i, p : self.passes) {
        if (!p.enabled) continue;
        p.render_to_screen = self.render_to_screen && (sz)i == last;
        p.render_fn(p, self, self.write_buffer, self.read_buffer, delta);
        if (p.needs_swap) self.swap_buffers();
    }
    self.width = tw;
    self.height = th;
    profile::commit();
}

// Disposes the composer's internal ping-pong buffers and frees the pass list (callers own and
// dispose individual passes).
fn void EffectComposer.dispose(&self) {
    foreach (b : self.buffers) if (b) b.dispose();
    self.passes.free();
}
