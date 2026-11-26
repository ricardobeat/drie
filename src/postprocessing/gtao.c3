module drie::postprocessing;
import std::collections::list;
import sokol, drie;

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
