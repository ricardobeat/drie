// Depth-of-field pass (`BokehPass`, `shaders/bokeh.glsl`) combining tiled gather blur with
// additive bokeh disc scattering for small out-of-focus highlights. Reads linear HDR color and
// a depth view whose green channel stores linear camera-plane distance in meters (such as
// `GTAOPass.ao_target.texture`). Optional `autofocus` tracks center depth in the 1x1 `R16F`
// `focus_dist` ping-pong target exposed via `focus_map()` for `Renderer.focus_map`. Run before
// `TAAPass` so temporal accumulation smooths the per-frame rotated gather taps.

module drie::postprocessing;
import std::collections::list;
import sokol, drie;

// ---- BokehPass

// Thin-lens depth-of-field pass with optional autofocus and highlight bokeh scattering.
//
//   BokehPass* dof = postprocessing::bokeh_pass(focus: 3, aperture: 6, maxblur: 16);
//   dof.render(dof_rt, scene_rt.texture, ao.ao_target.texture);
//
// Circle of confusion at distance `z` (m) is `aperture / focus * |1 - focus / z|` pixels,
// clamped to `maxblur`.
struct BokehPass {
    inline Pass pass;
    float focus; // Manual focus distance from camera plane (m), ignored when `autofocus` is true.
    float aperture; // Blur scale in pixels at infinity when focused at 1 m.
    float maxblur; // Maximum circle-of-confusion radius in pixels.
    int taps; // Gather taps per pixel (`4..64`, default 16).
    bool scatter; // When true (default), extracts bright out-of-focus sparks and scatters them as discs.
    bool autofocus; // When true, focuses on center screen depth instead of `focus`.
    float focus_speed; // Per-frame autofocus interpolation factor (`1` snaps immediately).
    float focus_min; // Nearest autofocus clamp distance (m).
    float focus_max; // Farthest autofocus clamp distance (m).
    bool reset; // Snaps autofocus to the current center depth on the next `render()`, then clears.
    SgView depth; // Depth view (`G` = linear camera-plane distance in meters) used inside `EffectComposer`.
    PingPong focus_dist; // 1x1 `R16F` ping-pong target storing the active focus distance (m) in R.
    int frame; // Frame counter rotating gather tap angles (mod 64).
    ShaderPass focus_pass; // Updates `focus_dist` from center depth or manual `focus`.
    ShaderPass tiles; // Computes per-16x16-tile maximum circle of confusion.
    ShaderPass reach; // Dilates tile CoC bounds to neighborhood maximum reach.
    ShaderPass prefilter; // Half-res CoC-weighted prefilter pass.
    ShaderPass sparks; // Extracts bright high-contrast sparks for disc scattering.
    ShaderPass rest; // Subtracts scattered sparks from the half-res prefiltered buffer.
    ShaderPass clean; // Removes scattered sparks from full-res color before gather.
    ShaderPass shrink; // Downsamples `half_rt` to `quarter_rt` for large-radius gather taps.
    ShaderPass gather; // Full-res multi-resolution depth-aware gather pass.
    ShaderPass scatter_pass; // Instanced additive quad pass drawing bokeh discs for sparks.
    RenderTarget* tiles_rt; // 1/16-res `R16F` per-tile max CoC target.
    RenderTarget* reach_rt; // 1/16-res `R16F` dilated neighborhood CoC target.
    RenderTarget* soft_rt; // Half-res `RGBA16F` prefiltered color target.
    RenderTarget* sparks_rt; // Half-res `RGBA16F` extracted spark color/CoC target.
    RenderTarget* half_rt; // Half-res `RGBA16F` spark-subtracted gather source.
    RenderTarget* quarter_rt; // Quarter-res `RGBA16F` gather source for wide blur radii.
    RenderTarget* clean_rt; // Full-res `RGBA16F` spark-subtracted color target (when `scatter` is true).
}

// Allocates a `BokehPass` with initial `focus` (m), `aperture`, and `maxblur` (pixels).
fn BokehPass* bokeh_pass(float focus = 1, float aperture = 4, float maxblur = 12) {
    BokehPass* p = mem::new(BokehPass);
    *p = {
        .render_fn = &bokeh_pass_render,
        .enabled = true,
        .needs_swap = true,
        .focus = focus,
        .aperture = aperture,
        .maxblur = maxblur,
        .taps = 16,
        .scatter = true,
        .focus_speed = 1,
        .focus_min = 0.1f,
        .focus_max = 1000,
    };
    SgBackend b = sg::query_backend();
    p.focus_pass.init(bokeh_focus_shader_desc(b), "drie-bokeh-focus");
    p.tiles.init(bokeh_tiles_shader_desc(b), "drie-bokeh-tiles");
    p.reach.init(bokeh_reach_shader_desc(b), "drie-bokeh-reach");
    p.prefilter.init(bokeh_prefilter_shader_desc(b), "drie-bokeh-prefilter");
    p.sparks.init(bokeh_sparks_shader_desc(b), "drie-bokeh-sparks");
    p.rest.init(bokeh_rest_shader_desc(b), "drie-bokeh-rest");
    p.clean.init(bokeh_clean_shader_desc(b), "drie-bokeh-clean");
    p.shrink.init(bokeh_shrink_shader_desc(b), "drie-bokeh-shrink");
    p.gather.init(bokeh_gather_shader_desc(b), "drie-bokeh-gather");
    p.scatter_pass.init(bokeh_scatter_shader_desc(b), "drie-bokeh-scatter", ADDITIVE_BLEND);
    p.scatter_pass.vertex_count = 6;
    p.focus_pass.set_sampler(SMP_BOKEH_FOCUS_SMP, nearest_sampler());
    p.tiles.set_sampler(SMP_BOKEH_TILES_SMP, nearest_sampler());
    p.reach.set_sampler(SMP_BOKEH_REACH_SMP, nearest_sampler());
    p.prefilter.set_sampler(SMP_BOKEH_PREFILTER_SMP, nearest_sampler());
    p.sparks.set_sampler(SMP_BOKEH_SPARKS_SMP, nearest_sampler());
    p.rest.set_sampler(SMP_BOKEH_REST_SMP, nearest_sampler());
    p.clean.set_sampler(SMP_BOKEH_CLEAN_SMP, nearest_sampler());
    p.shrink.set_sampler(SMP_BOKEH_SHRINK_SMP, linear_sampler());
    p.gather.set_sampler(SMP_BOKEH_GATHER_SMP, linear_sampler());
    p.scatter_pass.set_sampler(SMP_BOKEH_SCATTER_VSMP, nearest_sampler());
    p.scatter_pass.set_sampler(SMP_BOKEH_SCATTER_FSMP, nearest_sampler());
    p.focus_dist = ping_pong(1, 1, R16F);
    return p;
}

fn RenderTarget* bokeh_target(RenderTarget* t, int w, int h, SgPixelFormat format) @local {
    w = w > 1 ? w : 1;
    h = h > 1 ? h : 1;
    if (!t) return drie::render_target(w, h, 1, format, false);
    t.set_size(w, h, 1);
    return t;
}

// Returns the latest 1x1 `R16F` focus-distance texture (R = meters) for `Renderer.focus_map`
// (`Material.defocus`).
fn Texture* BokehPass.focus_map(&self) => self.focus_dist.read().as_map();

// Renders `color` (linear HDR) with depth of field into `target`, reading linear camera-plane
// distance in meters from `depth`'s green channel.
fn void BokehPass.render(&self, RenderTarget* target, SgView color, SgView depth) {
    int w = target.width;
    int h = target.height;
    self.tiles_rt = bokeh_target(self.tiles_rt, (w + 15) / 16, (h + 15) / 16, R16F);
    self.reach_rt = bokeh_target(self.reach_rt, (w + 15) / 16, (h + 15) / 16, R16F);
    self.soft_rt = bokeh_target(self.soft_rt, w / 2, h / 2, RGBA16F);
    self.sparks_rt = bokeh_target(self.sparks_rt, w / 2, h / 2, RGBA16F);
    self.half_rt = bokeh_target(self.half_rt, w / 2, h / 2, RGBA16F);
    self.quarter_rt = bokeh_target(self.quarter_rt, w / 4, h / 4, RGBA16F);
    if (self.scatter) self.clean_rt = bokeh_target(self.clean_rt, w, h, RGBA16F);
    BokehParams p = {
        .size = { w, h, (float)(self.frame % 64), drie::depth_zero_to_one() ? 1 : 0 },
        .tune = { self.taps, self.aperture, self.maxblur, self.scatter ? 1 : 0 },
        .focus = {
            self.focus > 1e-3f ? self.focus : 1e-3f,
            self.focus_speed,
            self.autofocus ? 1 : 0,
            self.reset ? 1 : 0
        },
        .range = { self.focus_min, self.focus_max, 0, 0 },
    };
    self.frame++;
    self.reset = false;

    ShaderPass* fp = &self.focus_pass;
    fp.set_texture(VIEW_BOKEH_FOCUS_DEPTH, depth);
    fp.set_texture(VIEW_BOKEH_FOCUS_PREV, self.focus_dist.read().texture);
    fp.set_uniforms(UB_BOKEH_PARAMS, &p);
    fp.render(self.focus_dist.write());
    self.focus_dist.swap();
    SgView focus = self.focus_dist.read().texture;

    self.tiles.set_texture(VIEW_BOKEH_TILES_DEPTH, depth);
    self.tiles.set_texture(VIEW_BOKEH_TILES_FOCUS, focus);
    self.tiles.set_uniforms(UB_BOKEH_PARAMS, &p);
    self.tiles.render(self.tiles_rt);
    self.reach.set_texture(VIEW_BOKEH_REACH_TILES, self.tiles_rt.texture);
    self.reach.set_uniforms(UB_BOKEH_PARAMS, &p);
    self.reach.render(self.reach_rt);

    self.prefilter.set_texture(VIEW_BOKEH_PREFILTER_COLOR, color);
    self.prefilter.set_texture(VIEW_BOKEH_PREFILTER_DEPTH, depth);
    self.prefilter.set_texture(VIEW_BOKEH_PREFILTER_FOCUS, focus);
    self.prefilter.set_uniforms(UB_BOKEH_PARAMS, &p);
    self.prefilter.render(self.soft_rt);
    self.sparks.set_texture(VIEW_BOKEH_SPARKS_SOFT, self.soft_rt.texture);
    self.sparks.set_texture(VIEW_BOKEH_SPARKS_COLOR, color);
    self.sparks.set_texture(VIEW_BOKEH_SPARKS_DEPTH, depth);
    self.sparks.set_texture(VIEW_BOKEH_SPARKS_FOCUS, focus);
    self.sparks.set_uniforms(UB_BOKEH_PARAMS, &p);
    self.sparks.render(self.sparks_rt);
    self.rest.set_texture(VIEW_BOKEH_REST_SOFT, self.soft_rt.texture);
    self.rest.set_texture(VIEW_BOKEH_REST_SPARKS, self.sparks_rt.texture);
    self.rest.render(self.half_rt);
    self.shrink.set_texture(VIEW_BOKEH_SHRINK_SRC, self.half_rt.texture);
    self.shrink.render(self.quarter_rt);
    // Strip scattered sparks from the gather source so they are not double-counted.
    SgView gathered = color;
    if (self.scatter) {
        self.clean.set_texture(VIEW_BOKEH_CLEAN_COLOR, color);
        self.clean.set_texture(VIEW_BOKEH_CLEAN_SPARKS, self.sparks_rt.texture);
        self.clean.set_texture(VIEW_BOKEH_CLEAN_DEPTH, depth);
        self.clean.set_texture(VIEW_BOKEH_CLEAN_FOCUS, focus);
        self.clean.set_uniforms(UB_BOKEH_PARAMS, &p);
        self.clean.render(self.clean_rt);
        gathered = self.clean_rt.texture;
    }

    ShaderPass* g = &self.gather;
    g.set_texture(VIEW_BOKEH_GATHER_COLOR, gathered);
    g.set_texture(VIEW_BOKEH_GATHER_DEPTH, depth);
    g.set_texture(VIEW_BOKEH_GATHER_FOCUS, focus);
    g.set_texture(VIEW_BOKEH_GATHER_HALF, self.half_rt.texture);
    g.set_texture(VIEW_BOKEH_GATHER_QUARTER, self.quarter_rt.texture);
    g.set_texture(VIEW_BOKEH_GATHER_REACH, self.reach_rt.texture);
    g.set_uniforms(UB_BOKEH_PARAMS, &p);
    g.render(target);

    if (self.scatter && self.aperture > 0) {
        ShaderPass* sc = &self.scatter_pass;
        sc.instance_count = self.sparks_rt.width * self.sparks_rt.height;
        sc.set_texture(VIEW_BOKEH_SCATTER_SPARKS, self.sparks_rt.texture);
        sc.set_texture(VIEW_BOKEH_SCATTER_FOCUS, focus);
        sc.set_texture(VIEW_BOKEH_SCATTER_DEPTH, depth);
        BokehScatterParams sp = { .size = p.size, .tune = p.tune };
        sc.set_uniforms(UB_BOKEH_SCATTER_PARAMS, &sp);
        sc.render(target, LOAD);
    }
}

fn void bokeh_pass_render(
    Pass* pass,
    EffectComposer* composer,
    RenderTarget* write,
    RenderTarget* read,
    float delta
) @local {
    BokehPass* self = (BokehPass*)pass;
    if (!self.depth.id) abort("drie: BokehPass in an EffectComposer needs its depth (GTAOPass.ao_target's texture)");
    self.render(write, read.texture, self.depth);
}