// Final tone-mapping and display-encoding pass (`OutputPass`, `shaders/output.glsl`) for linear
// HDR post-processing chains. Reads `Renderer.tone_mapping` and `Renderer.tone_mapping_exposure`
// on each draw, optionally multiplies exposure by a GPU `exposure_map` or substitutes a baked
// 3D `ColorGrade` LUT on SDR output, and encodes to sRGB, Display P3, or linear extended-range
// HDR while preserving alpha. Drawing to `null` (the window) syncs `color_space` and `hdr` from
// the renderer via `match_window()`.

module drie::postprocessing;
import std::collections::list;
import sokol, drie;

// ---- OutputPass

// Tone-mapping and display-encoding pass placed at the end of a linear HDR chain.
//
//   OutputPass* out = postprocessing::output_pass(renderer);
//   out.set_source(hdr_target.texture);
//   out.render(ldr_target); // or null for the window
struct OutputPass {
    inline ShaderPass shader_pass;
    Renderer* renderer; // Renderer providing tone mapping, exposure, color space, and HDR headroom.
    // Optional 1x1 texture whose red channel multiplies exposure on the GPU (e.g., auto-exposure
    // meter). Leave zero-initialized (`{}`) to disable.
    SgView exposure_map;
    // Optional `ColorGrade` (`drie::postprocessing`) whose baked 3D LUT replaces inline tone
    // mapping on SDR output (ignored when `hdr` is true).
    ColorGrade* grade;
    bool keep_linear; // When true, outputs linear `0..1` instead of sRGB encoding for downstream linear passes.
    // Output gamut (`SRGB_COLOR_SPACE` or `DISPLAY_P3_COLOR_SPACE`). Synced from
    // `renderer.output_color_space` automatically when rendering to the window.
    ColorSpace color_space;
    // Extended-range linear Display P3 output up to `renderer.output_headroom()`. Synced from
    // `renderer.hdr_on()` automatically when rendering to the window.
    bool hdr;
}

// Allocates an `OutputPass` reading tone-mapping settings from `renderer`.
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

// Sets the linear HDR source texture view to tone map (stretched to fit the output viewport).
fn void OutputPass.set_source(&self, SgView picture) => self.bindings.views[VIEW_OUTPUT_SRC] = picture;

// Refreshes uniforms and auxiliary texture bindings from `renderer`, `exposure_map`, and
// `grade`. Called automatically by `render()` and `draw()`.
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
            self.graded() ? 1 : 0,
            GRADE_LUT
        },
    };
    self.set_uniforms(UB_OUTPUT_PARAMS, &p);
    // Bind a valid fallback view when `exposure_map` is unset.
    self.bindings.views[VIEW_OUTPUT_EXPOSURE] = self.exposure_map.id ? self.exposure_map
        : self.bindings.views[VIEW_OUTPUT_SRC];
    self.bindings.views[VIEW_OUTPUT_LUT] = self.grade && self.grade.lut.id ? self.grade.lut_texture : empty_lut();
}

// Whether the grade's LUT takes the tone mapping's place. (Not on an HDR screen, and not on
// WebGPU, whose render passes can't draw into one slice of a 3D image as the bake does.)
fn bool OutputPass.graded(&self) => self.grade && !self.hdr && sg::query_backend() != WGPU;

// Prepares any attached `ColorGrade` LUT (outside a pass), refreshes uniforms, and renders into
// `target` (or the swapchain when `target == null`).
fn void OutputPass.render(
    &self,
    RenderTarget* target,
    SgLoadAction load = DONTCARE,
    Color clear = 0x000000,
    float clear_alpha = 1
) {
    if (!target) self.match_window();
    if (self.graded()) self.grade.prepare(self.renderer.tone_mapping, self.color_space == DISPLAY_P3_COLOR_SPACE);
    self.update();
    self.shader_pass.render(target, load, clear, clear_alpha);
}

// Copies `color_space` and `hdr` flags from `renderer`'s active window configuration.
fn void OutputPass.match_window(&self) {
    self.color_space = self.renderer.output_color_space;
    self.hdr = self.renderer.hdr_on();
}

// Draws inside an already-active render pass; callers using `grade` must call
// `ColorGrade.prepare()` before starting the pass.
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
