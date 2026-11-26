module drie::postprocessing;
import std::collections::list;
import sokol, drie;

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
    // The color grade (grading.c3): its LUT, baked with the renderer's tone mapping, takes the
    // place of the tone mapping (not on an HDR screen). Null: none.
    ColorGrade* grade;
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
            self.grade && !self.hdr ? 1 : 0,
            GRADE_LUT
        },
    };
    self.set_uniforms(UB_OUTPUT_PARAMS, &p);
    // (an unused slot still needs a texture)
    self.bindings.views[VIEW_OUTPUT_EXPOSURE] = self.exposure_map.id ? self.exposure_map
        : self.bindings.views[VIEW_OUTPUT_SRC];
    self.bindings.views[VIEW_OUTPUT_LUT] = self.grade && self.grade.lut.id ? self.grade.lut_texture : empty_lut();
}

fn void OutputPass.render(
    &self,
    RenderTarget* target,
    SgLoadAction load = DONTCARE,
    Color clear = 0x000000,
    float clear_alpha = 1
) {
    if (!target) self.match_window();
    if (self.grade && !self.hdr) self.grade.prepare(self.renderer.tone_mapping, self.color_space == DISPLAY_P3_COLOR_SPACE);
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
