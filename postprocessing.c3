// Post-processing, after three.js's EffectComposer, RenderPass and ShaderPass.
//
// A ShaderPass is a fullscreen shader (a GLSL @vs/@fs pair compiled with sokol-shdc, like
// the engine's own) drawn into a RenderTarget, over one, or into the window. It keeps its
// textures, samplers and uniforms between frames, and makes a pipeline for each kind of
// target it is drawn into, the first time, so one pass can draw into a half-float target and
// the window alike. Each draw is a single triangle in a pass of its own: nothing is copied.
//
//   ShaderPass* blur = drie::shader_pass(blur_shader_desc(sg::query_backend()), "blur");
//   blur.set_texture(VIEW_SRC, scene_target.texture);
//   blur.set_sampler(SMP_SMP, drie::linear_sampler());
//   blur.set_uniforms(UB_BLUR_PARAMS, &params);
//   blur.render(blurred);              // into a RenderTarget (null: the window)
//
// The vertex shader makes the triangle from gl_VertexIndex (no vertex buffer):
//   vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
//   gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
//
// EffectComposer chains passes the way three's does, handing each the picture of the one
// before (ShaderPass.input_view, three's 'tDiffuse'); the last one draws into the window.
// Like three's, it works in linear light (RGBA16F) by default, and the chain ends with an
// OutputPass, which tone maps the picture and encodes it as sRGB:
//
//   EffectComposer* composer = postprocessing::effect_composer(renderer);
//   composer.add_pass(postprocessing::render_pass(scene, camera));
//   composer.add_pass(my_effect);
//   composer.add_pass(postprocessing::output_pass(renderer));
// A chain whose passes read several pictures, or keep some from frame to frame, can call the
// passes itself in its own order instead, with render targets (any pixel format) and
// PingPong pairs for what builds up over frames.
module drie::postprocessing;
import std::collections::list;
import sokol, drie;

// Blend states for ShaderPass (three's MultiplyBlending, AdditiveBlending, NormalBlending).
// Multiply keeps the target's alpha.
const SgBlendState NO_BLEND       = { };
const SgBlendState MULTIPLY_BLEND = {
    .enabled = true,
    .src_factor_rgb = DST_COLOR,
    .dst_factor_rgb = ZERO,
    .src_factor_alpha = ZERO,
    .dst_factor_alpha = ONE,
};
const SgBlendState ADDITIVE_BLEND = {
    .enabled = true,
    .src_factor_rgb = ONE,
    .dst_factor_rgb = ONE,
    .src_factor_alpha = ZERO,
    .dst_factor_alpha = ONE,
};
const SgBlendState ALPHA_BLEND    = {
    .enabled = true,
    .src_factor_rgb = SRC_ALPHA,
    .dst_factor_rgb = ONE_MINUS_SRC_ALPHA,
    .src_factor_alpha = ONE,
    .dst_factor_alpha = ONE_MINUS_SRC_ALPHA,
};

// sokol has 8 uniform block slots; a pass keeps a copy of each block it was given.
const int PASS_UNIFORM_SLOTS = 8;
const int PASS_UNIFORM_BYTES = 512;

// What the pass being drawn into is made of: a pipeline is made for each.
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
    // the pass begun by begin_screen, RenderTarget.begin or ShaderPass.render
    PassFormat current;
}

PostState post_state @local;

// Shared samplers for passes: bilinear and nearest, both clamped to the edge.
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

// Starts a pass into the target, for ShaderPass.draw and drawing of your own (sokol_gl,
// sokol_debugtext). `clear` and `clear_alpha` are used with load: CLEAR. A target with a depth
// buffer has it cleared. End with end().
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

// Starts a pass over the target's resolved picture (its `texture`), as it is: for drawing on
// top of a render, such as darkening it with MULTIPLY_BLEND. End with end().
fn void RenderTarget.begin_over(&self, ZString label = "drie-over") {
    SgPass pass = { .action.colors[0] = { .load_action = LOAD }, .label = label };
    pass.attachments.colors[0] = self.overlay_view;
    profile::begin_pass(&pass);
    post_state.current = { self.pixel_format(), NONE, 1, false };
}

fn void RenderTarget.end(&self) => sg::end_pass();

// Starts a pass into the window, cleared to `clear`. With `aspect` (width / height) the
// picture keeps that shape, with black bars in a window of any other. End with end_screen().
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

// Draws the overlays (drie::add_overlay: menus, fades, touch controls) over the whole window
// and ends the pass begun with begin_screen.
fn void end_screen() {
    sg::apply_viewport(0, 0, sapp::width(), sapp::height(), true);
    drie::draw_overlays();
    sg::end_pass();
}

// The base of a pass in an EffectComposer, like three's Pass: a pass of your own is a struct
// with `inline Pass pass;` first and its own `render_fn`.
alias PassRenderFn = fn void(
    Pass* pass,
    EffectComposer* composer,
    RenderTarget* write,
    RenderTarget* read,
    float delta
);

struct Pass {
    PassRenderFn render_fn;
    bool enabled;
    // the composer swaps its read and write buffers after this pass
    bool needs_swap;
    // set by the composer on its last enabled pass: draw into the window instead of `write`
    bool render_to_screen;
}

struct ShaderPass {
    inline Pass pass;
    SgShader shader;
    ZString label;
    SgBlendState blend;
    // What the shader reads, at the slots sokol-shdc generated (VIEW_X, SMP_X).
    SgBindings bindings;
    // In an EffectComposer: where the picture of the pass before goes (three's 'tDiffuse'),
    // -1 for nowhere.
    int input_view;
    int input_sampler;
    char[PASS_UNIFORM_BYTES][PASS_UNIFORM_SLOTS] uniforms;
    usz[PASS_UNIFORM_SLOTS] uniform_size;
    List{PassPipeline} pipelines;
}

// A fullscreen pass with this shader (`my_shader_desc(sg::query_backend())`). `blend` blends
// it into what is in the target (NO_BLEND, MULTIPLY_BLEND, ADDITIVE_BLEND, ALPHA_BLEND).
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

// Sets up a ShaderPass in place (for a pass of your own that inlines one).
fn void ShaderPass.init(
    &self,
    SgShaderDesc desc,
    ZString label = "drie-pass",
    SgBlendState blend = NO_BLEND,
    int input_view = -1,
    int input_sampler = -1
) {
    self.shader = sg::make_shader(&desc);
    self.label = label;
    self.blend = blend;
    self.input_view = input_view;
    self.input_sampler = input_sampler;
    self.render_fn = &shader_pass_render;
    self.enabled = true;
    self.needs_swap = true;
}

fn void ShaderPass.set_texture(&self, int slot, SgView view) => self.bindings.views[slot] = view;
fn void ShaderPass.set_sampler(&self, int slot, SgSampler sampler) => self.bindings.samplers[slot] = sampler;

// Keeps a copy of a uniform block (the struct sokol-shdc generated), applied on every draw
// until it is set again: pass.set_uniforms(UB_MY_PARAMS, &params).
macro void ShaderPass.set_uniforms(&self, int slot, data) {
    self.set_uniform_bytes(slot, (char*)data, $Typeof(*data)::size);
}

fn void ShaderPass.set_uniform_bytes(&self, int slot, char* data, usz size) {
    assert(slot >= 0 && slot < PASS_UNIFORM_SLOTS
        && size <= PASS_UNIFORM_BYTES, "uniform block slot %d of %d bytes", slot, size);
    mem::copy(&self.uniforms[slot], data, (sz)size);
    self.uniform_size[slot] = size;
}

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

// Draws into the pass that is going on (begin_screen, RenderTarget.begin or begin_over).
fn void ShaderPass.draw(&self) {
    sg::apply_pipeline(self.pipeline_for(post_state.current));
    sg::apply_bindings(&self.bindings);
    for (int i = 0; i < PASS_UNIFORM_SLOTS; i++) {
        if (self.uniform_size[i]) sg::apply_uniforms(i, &&{ .ptr = &self.uniforms[i], .size = self.uniform_size[i] });
    }
    sg::draw(0, 3, 1);
}

// Draws into `target` in a pass of its own; null draws into the whole window (with the
// overlay, see end_screen). By default what was in the target is not kept (load: DONTCARE):
// the pass covers all of it.
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

// Draws over the target's resolved picture, keeping it (see RenderTarget.begin_over).
fn void ShaderPass.render_over(&self, RenderTarget* target) {
    target.begin_over(self.label);
    self.draw();
    target.end();
}

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

// Renders a scene, like three's RenderPass: into a target of its own (multisampled, with a
// depth buffer, as big as the composer's), which the next pass then reads. Set `mrt` (on an
// RGBA16F composer) for passes that read its normals, motion or indirect light (GTAOPass).
struct RenderPass {
    inline Pass pass;
    Scene* scene;
    Camera* camera;
    int samples;
    bool mrt;
    RenderTarget* target;
}

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
    // the next pass reads the render where it is
    composer.read_buffer = self.target;
}

// Two render targets that take turns, for what builds up from frame to frame (a running
// average, a meter): read() is the one written last, write() the other. swap() after writing.
struct PingPong {
    RenderTarget*[2] targets;
    int cur;
}

fn PingPong ping_pong(int width, int height, SgPixelFormat format = RGBA8) {
    return {
        .targets = {
            drie::render_target(width, height, 1, format, false),
            drie::render_target(width, height, 1, format, false)
        }
    };
}

fn RenderTarget* PingPong.read(&self) => self.targets[self.cur];
fn RenderTarget* PingPong.write(&self) => self.targets[1 - self.cur];
fn void PingPong.swap(&self) => self.cur = 1 - self.cur;

fn void PingPong.set_size(&self, int width, int height) {
    foreach (t : self.targets) t.set_size(width, height, 1);
    self.cur = 0;
}

// Runs passes one after the other, like three's EffectComposer: each ShaderPass reads the
// picture the pass before left in `read_buffer` and draws into `write_buffer`, then the two
// are swapped. The last enabled pass draws into the window. The buffers are `format`, one
// sample, no depth: RGBA16F (three's HalfFloatType) keeps the light as it is until an
// OutputPass at the end; with RGBA8 the scene is tone mapped as it is rendered.
struct EffectComposer {
    Renderer* renderer;
    int width;
    int height;
    SgPixelFormat format;
    RenderTarget*[2] buffers;
    RenderTarget* read_buffer;
    RenderTarget* write_buffer;
    List{Pass*} passes;
    // the last enabled pass draws into the window (off: everything stays in the buffers)
    bool render_to_screen;
}

// `width` and `height` 0: the window's size, followed as it changes.
fn EffectComposer* effect_composer(Renderer* renderer, int width = 0, int height = 0, SgPixelFormat format = RGBA16F) {
    EffectComposer* c = mem::new(EffectComposer);
    *c = { .renderer = renderer, .width = width, .height = height, .format = format, .render_to_screen = true };
    return c;
}

fn void EffectComposer.add_pass(&self, Pass* pass) => self.passes.push(pass);

fn void EffectComposer.remove_pass(&self, Pass* pass) {
    foreach (i, p : self.passes) {
        if (p == pass) {
            self.passes.remove_at(i);
            return;
        }
    }
}

fn void EffectComposer.swap_buffers(&self) {
    self.read_buffer = self.write_buffer;
    self.write_buffer = self.read_buffer == self.buffers[0] ? self.buffers[1] : self.buffers[0];
}

fn void EffectComposer.set_size(&self, int width, int height) {
    self.width = width;
    self.height = height;
    foreach (b : self.buffers) if (b) b.set_size(width, height, 1);
}

// Runs the passes. Like render(), call once per frame; the frame is committed after it.
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

fn void EffectComposer.dispose(&self) {
    foreach (b : self.buffers) if (b) b.dispose();
    self.passes.free();
}

// The end of a chain in linear light, like three's OutputPass: tone maps the picture with the
// renderer's tone_mapping and tone_mapping_exposure (read whenever it draws) and encodes it as
// sRGB. Alpha is kept.
//
//   OutputPass* out = postprocessing::output_pass(renderer);
//   out.set_source(hdr_target.texture);
//   out.render(ldr_target);           // or null: the window
struct OutputPass {
    inline ShaderPass shader_pass;
    Renderer* renderer;
    // A 1x1 texture whose red channel multiplies the exposure (an auto exposure's meter, say),
    // read on the GPU so the meter never has to come back to the CPU. None: {}.
    SgView exposure_map;
    // Leave the result linear (0..1) instead of encoding it as sRGB, for more passes after it
    // that work in linear light.
    bool keep_linear;
    // The colors of the result: SRGB_COLOR_SPACE, or DISPLAY_P3_COLOR_SPACE (tone mapping clips
    // to P3's colors instead of sRGB's). Set from renderer.output_color_space when it draws into
    // the window.
    ColorSpace color_space;
    // For an HDR screen (renderer.hdr_output): the result is linear, in Display P3, with the
    // highlights going up to the screen's headroom instead of to white. Set from the renderer
    // when it draws into the window.
    bool hdr;
}

fn OutputPass* output_pass(Renderer* renderer) {
    OutputPass* p = mem::new(OutputPass);
    p.init(
        output_shader_desc(sg::query_backend()),
        "drie-output",
        input_view: VIEW_OUTPUT_SRC,
        input_sampler: SMP_OUTPUT_SMP
    );
    p.renderer = renderer;
    p.render_fn = &output_pass_render;
    p.set_sampler(SMP_OUTPUT_SMP, linear_sampler());
    return p;
}

// The picture to tone map (linear light, any size: it is stretched over the target).
fn void OutputPass.set_source(&self, SgView picture) => self.bindings.views[VIEW_OUTPUT_SRC] = picture;

// Takes the renderer's tone mapping as it is now. render() and draw() call it.
fn void OutputPass.update(&self) {
    Renderer* r = self.renderer;
    OutputParams p = {
        .params = {
            (float)r.tone_mapping.ordinal,
            r.tone_mapping_exposure,
            self.exposure_map.id ? 1 : 0,
            self.keep_linear ? 1 : 0
        },
        .params2 = {
            self.hdr || self.color_space == DISPLAY_P3_COLOR_SPACE ? 1 : 0,
            self.hdr ? r.output_headroom() : 0,
            0,
            0
        },
    };
    self.set_uniforms(UB_OUTPUT_PARAMS, &p);
    // (an unused slot still needs a texture)
    self.bindings.views[VIEW_OUTPUT_EXPOSURE] = self.exposure_map.id ? self.exposure_map
        : self.bindings.views[VIEW_OUTPUT_SRC];
}

fn void OutputPass.render(
    &self,
    RenderTarget* target,
    SgLoadAction load = DONTCARE,
    Color clear = 0x000000,
    float clear_alpha = 1
) {
    if (!target) self.match_window();
    self.update();
    self.shader_pass.render(target, load, clear, clear_alpha);
}

// Takes the window's colors and range from the renderer (output_color_space, hdr_output).
fn void OutputPass.match_window(&self) {
    self.color_space = self.renderer.output_color_space;
    self.hdr = self.renderer.hdr_on();
}

fn void OutputPass.draw(&self) {
    if (post_state.current.screen) self.match_window();
    self.update();
    self.shader_pass.draw();
}

fn void output_pass_render(
    Pass* pass,
    EffectComposer* composer,
    RenderTarget* write,
    RenderTarget* read,
    float delta
) @local {
    OutputPass* self = (OutputPass*)pass;
    self.set_source(read.texture);
    self.render(pass.render_to_screen ? null : write);
}

// Ambient occlusion, like three's GTAOPass (ground-truth AO, after Intel's XeGTAO), for a
// scene rendered into a target with MRT (RenderTarget.mrt): from its depth and normals, it
// darkens only the light the scene got indirectly (RenderTarget.indirect_texture: the ambient
// and environment light, reflections), as the material shaders of three's WebGPU renderer do
// with an `ao` context. Direct light is the shadows' business. Smooth surfaces keep their
// reflections but at grazing angles (specular occlusion).
//
//   GTAOPass* ao = postprocessing::gtao_pass(camera);
//   ao.radius = 0.5;
//   renderer.set_render_target(scene_rt); renderer.render(scene, camera);
//   ao.render(scene_rt);               // darkens scene_rt's picture where it is occluded
//
// In an EffectComposer it goes right after a RenderPass with `mrt` set. `ao_target` keeps the
// occlusion (r), each pixel's distance from the camera's plane in meters (g) and n·v (b) until
// the next render, for other passes that want the distance.
struct GTAOPass {
    inline Pass pass;
    Camera* camera;
    // how far around a point things darken it (m)
    float radius;
    // a power on the visibility: over 1 darker, under lighter, 0 none at all
    float intensity;
    // fades out between these distances (m); fade_end 0: never
    float fade_start;
    float fade_end;
    // the share of the occlusion that darkens the direct light too: 0, as it should be; more
    // for a look where corners darken however they are lit
    float direct_occlusion;
    bool specular_occlusion;
    // slices of the hemisphere above each pixel (up to 8), and steps along each side of one (up
    // to 24): more is less noise for the blur and the frames to even out, at the cost of
    // slices * steps * 2 depth reads a pixel. Default 2 and 6.
    int slices;
    int steps;
    RenderTarget* ao_target;
    ShaderPass gtao;
    ShaderPass gtao_ms;
    ShaderPass apply;
    int frame;
}

fn GTAOPass* gtao_pass(Camera* camera) {
    GTAOPass* p = mem::new(GTAOPass);
    *p = {
        .render_fn = &gtao_pass_render,
        .enabled = true,
        .camera = camera,
        .radius = 0.5,
        .intensity = 1,
        .specular_occlusion = true,
        .slices = 2,
        .steps = 6
    };
    SgBackend backend = sg::query_backend();
    p.gtao.init(gtao_shader_desc(backend), "drie-gtao");
    p.gtao_ms.init(gtao_ms_shader_desc(backend), "drie-gtao");
    p.apply.init(gtao_apply_shader_desc(backend), "drie-gtao-apply", MULTIPLY_BLEND);
    p.gtao.set_sampler(SMP_GTAO_SMP, nearest_sampler());
    p.gtao_ms.set_sampler(SMP_GTAO_SMP, nearest_sampler());
    p.apply.set_sampler(SMP_GTAO_APPLY_SMP, nearest_sampler());
    return p;
}

// Works out the occlusion of what `target` (rendered from the pass's camera, with MRT) shows
// into ao_target, and darkens the target's picture with it.
fn void GTAOPass.render(&self, RenderTarget* target) {
    self.render_occlusion(target);
    self.apply.set_texture(VIEW_GTAO_AO, self.ao_target.texture);
    self.apply.set_texture(VIEW_GTAO_INDIRECT, target.indirect_texture);
    GtaoApplyParams ap = {
        .size = { target.width, target.height, self.direct_occlusion, self.specular_occlusion ? 0 : 1 }
    };
    self.apply.set_uniforms(UB_GTAO_APPLY_PARAMS, &ap);
    self.apply.render_over(target);
}

// Only the occlusion, into ao_target, leaving the picture as it is.
fn void GTAOPass.render_occlusion(&self, RenderTarget* target) {
    if (!target.mrt || !target.depth_buffer) abort("drie: GTAOPass needs a target with MRT and a depth buffer");
    if (!self.ao_target) self.ao_target = drie::render_target(target.width, target.height, 1, RGBA16F, false);
    self.ao_target.set_size(target.width, target.height, 1);
    Camera* cam = self.camera;
    Mat4 m      = cam.matrix_world;
    GtaoParams p = {
        .clip = { cam.near, cam.far, drie::depth_zero_to_one() ? 1 : 0, drie::depth_zero_to_one() ? 1 : 0 },
        .view = { 1 / cam.projection_matrix[0][0], 1 / cam.projection_matrix[1][1], target.width, target.height },
        .tune = { self.radius, self.intensity, self.fade_start, self.fade_end },
        .cam_right = { m[0][0], m[0][1], m[0][2], (float)(self.frame % 64) },
        .cam_up = { m[1][0], m[1][1], m[1][2], self.slices },
        .cam_fwd = { -m[2][0], -m[2][1], -m[2][2], self.steps },
    };
    self.frame++;
    ShaderPass* g = target.samples > 1 ? &self.gtao_ms : &self.gtao;
    g.set_texture(target.samples > 1 ? VIEW_GTAO_DEPTH_MS : VIEW_GTAO_DEPTH, target.depth_texture);
    g.set_texture(VIEW_GTAO_NORMAL, target.normal_texture);
    g.set_uniforms(UB_GTAO_PARAMS, &p);
    g.render(self.ao_target);
}

fn void GTAOPass.dispose(&self) {
    self.gtao.dispose();
    self.gtao_ms.dispose();
    self.apply.dispose();
    if (self.ao_target) self.ao_target.dispose();
}

fn void gtao_pass_render(
    Pass* pass,
    EffectComposer* composer,
    RenderTarget* write,
    RenderTarget* read,
    float delta
) @local {
    GTAOPass* self = (GTAOPass*)pass;
    self.render(read);
}

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