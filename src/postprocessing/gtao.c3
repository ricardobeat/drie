// Ground-truth ambient occlusion pass (`GTAOPass`, `shaders/gtao.glsl`, based on Intel XeGTAO)
// for linear HDR render targets with `RenderTarget.mrt` and a depth buffer. Evaluates hemisphere
// visibility from depth and world normals into `ao_target` (R: occlusion, G: linear camera-plane
// distance in meters, B: `n.v`) and multiplies it over indirect lighting (`indirect_texture`)
// while leaving direct lighting intact unless `direct_occlusion > 0`. In `EffectComposer`, place
// right after an MRT `RenderPass`; call `dispose()` to free GPU targets and shaders.

module drie::postprocessing;
import std::collections::list;
import sokol, drie;

// ---- GTAOPass

// Screen-space ground-truth ambient occlusion pass over an MRT `RenderTarget`.
//
//   GTAOPass* ao = postprocessing::gtao_pass(camera);
//   ao.radius = 0.5;
//   renderer.set_render_target(scene_rt);
//   renderer.render(scene, camera);
//   ao.render(scene_rt); // darkens scene_rt where occluded
//
// `ao_target` keeps occlusion (R), linear camera-plane distance in meters (G), and `n.v` (B)
// until the next call, for downstream passes such as `BokehPass`.
struct GTAOPass {
    inline Pass pass;
    Camera* camera; // Camera whose projection and world basis match the rendered target.
    float radius; // World-space occlusion sampling radius (m).
    float intensity; // Exponent applied to visibility (`> 1` darker, `< 1` lighter, `0` disables).
    float fade_start; // Distance from camera where occlusion begins fading out (m).
    float fade_end; // Distance where occlusion reaches zero (m; `0` disables distance fade).
    float direct_occlusion; // Fraction of occlusion also applied to direct lighting (`0` = indirect only).
    bool specular_occlusion; // When true, preserves specular reflections except at grazing angles.
    int slices; // Hemisphere azimuth slices per pixel (`1..8`, default 2).
    int steps; // Depth samples per half-slice (`1..24`, default 6; total `slices * steps * 2` reads).
    RenderTarget* ao_target; // `RGBA16F` output storing occlusion (R), linear depth in meters (G), and `n.v` (B).
    ShaderPass gtao; // Single-sample depth/normal horizon integration pass.
    ShaderPass gtao_ms; // Multisampled depth horizon integration pass.
    ShaderPass apply; // Multiplicative composite pass over the scene target.
    int frame; // Frame counter rotating spatial sampling noise (mod 64).
}

// Allocates a `GTAOPass` bound to `camera` with default radius `0.5` m, `2` slices, and `6` steps.
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

// Computes ambient occlusion from `target` (MRT + depth) into `ao_target` and composites it
// multiplicatively over `target`'s resolved color texture.
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

// Evaluates occlusion and linear depth from `target` into `ao_target` without modifying `target`.
// Aborts if `target` lacks MRT attachments or a depth buffer.
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

// Disposes the internal shader passes and `ao_target`.
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