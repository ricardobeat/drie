// Linear HDR bloom (`UnrealBloomPass`, `shaders/bloom.glsl`) and screen-space lens flare
// (`LensFlarePass`, `shaders/flare.glsl`). `UnrealBloomPass` builds a progressive mip chain of
// up to 8 half-resolution `RGBA16F` targets and upsamples back through a tent filter with
// additive blending into the source target; `LensFlarePass` derives quarter-resolution chromatic
// ghosts and a halo from a blurred bright input (such as `bloom.mips[3]`) and adds them over the
// target. Both run in linear light before `OutputPass`; call `dispose()` to free GPU resources.

module drie::postprocessing;
import std::collections::list;
import sokol, drie;

// ---- UnrealBloomPass

// Progressive downsample/upsample bloom pass (`shaders/bloom.glsl`, after Call of Duty/Unreal
// bloom) operating in place on a linear HDR `RenderTarget`.
//
//   UnrealBloomPass* bloom = postprocessing::unreal_bloom_pass(strength: 0.05, threshold: 0);
//   bloom.render(hdr_target);
//
// With `threshold == 0`, all brightness levels contribute proportionally (use low `strength`,
// around `0.02..0.1`); a positive threshold restricts bloom to brighter highlights.
struct UnrealBloomPass {
    inline Pass pass;
    float strength; // Normalized multiplier on the final upsampled glow added to the target.
    float radius; // Per-mip upsample weight (`0..1`): each coarser level contributes `radius^i`.
    float threshold; // Linear luminance threshold above which pixels enter the bloom chain.
    float knee; // Soft-threshold transition width around `threshold` (default `0.5`).
    int levels; // Number of half-resolution mip levels (`1..8`); more levels widen the halo.
    RenderTarget*[8] mips; // Lazily allocated `RGBA16F` mip chain (also reusable as `LensFlarePass` input).
    ShaderPass down; // 13-tap downsample and threshold pass.
    ShaderPass up; // 9-tap tent upsample pass using `ADDITIVE_BLEND`.
}

// Allocates an `UnrealBloomPass` with the given `strength`, `radius`, `threshold`, and `levels`.
fn UnrealBloomPass* unreal_bloom_pass(float strength = 1, float radius = 0.4, float threshold = 0.85, int levels = 6) {
    UnrealBloomPass* p = mem::new(UnrealBloomPass);
    *p = {
        .render_fn = &unreal_bloom_pass_render,
        .enabled = true,
        .strength = strength,
        .radius = radius,
        .threshold = threshold,
        .knee = 0.5f,
        .levels = levels,
    };
    p.down.init(bloom_down_shader_desc(sg::query_backend()), "drie-bloom-down");
    p.up.init(bloom_up_shader_desc(sg::query_backend()), "drie-bloom-up", ADDITIVE_BLEND);
    p.down.set_sampler(SMP_BLOOM_DOWN_SMP, linear_sampler());
    p.up.set_sampler(SMP_BLOOM_UP_SMP, linear_sampler());
    return p;
}

// Downsamples `target` (linear light) through `levels` mips and additively composites the
// upsampled bloom back over `target`.
fn void UnrealBloomPass.render(&self, RenderTarget* target) {
    int n = self.levels < 1 ? 1 : (self.levels > 8 ? 8 : self.levels);
    int w = target.width;
    int h = target.height;
    for (int i = 0; i < n; i++) {
        w = w / 2 > 1 ? w / 2 : 1;
        h = h / 2 > 1 ? h / 2 : 1;
        if (!self.mips[i]) self.mips[i] = drie::render_target(w, h, 1, RGBA16F, false);
        self.mips[i].set_size(w, h, 1);
    }
    // Downsample chain, applying the brightness threshold on level 0.
    SgView src = target.texture;
    int sw = target.width;
    int sh = target.height;
    for (int i = 0; i < n; i++) {
        RenderTarget* m = self.mips[i];
        BloomDownParams p = {
            .size = { m.width, m.height, 1.0f / sw, 1.0f / sh },
            .tune = { i == 0 ? 1 : 0, self.threshold, self.knee, 0 },
        };
        self.down.set_texture(VIEW_BLOOM_DOWN_SRC, src);
        self.down.set_uniforms(UB_BLOOM_DOWN_PARAMS, &p);
        self.down.render(m);
        src = m.texture;
        sw = m.width;
        sh = m.height;
    }
    // Upsample chain, accumulating level `i` into `i - 1` with geometric weights `1, r, r^2, ...`.
    float r = self.radius < 0 ? 0 : (self.radius > 1 ? 1 : self.radius);
    float total = 1;
    float k = 1;
    for (int i = 1; i < n; i++) {
        k *= r;
        total += k;
    }
    for (int i = n - 1; i >= 1; i--) {
        RenderTarget* from = self.mips[i];
        RenderTarget* into = self.mips[i - 1];
        BloomUpParams p = {
            .size = { into.width, into.height, 1.0f / from.width, 1.0f / from.height },
            .tune = { r, 0, 0, 0 },
        };
        self.up.set_texture(VIEW_BLOOM_UP_SRC, from.texture);
        self.up.set_uniforms(UB_BLOOM_UP_PARAMS, &p);
        self.up.render_over(into);
    }
    RenderTarget* top = self.mips[0];
    BloomUpParams p = {
        .size = { target.width, target.height, 1.0f / top.width, 1.0f / top.height },
        .tune = { self.strength / total, 0, 0, 0 },
    };
    self.up.set_texture(VIEW_BLOOM_UP_SRC, top.texture);
    self.up.set_uniforms(UB_BLOOM_UP_PARAMS, &p);
    self.up.render_over(target);
}

// Disposes the downsample/upsample shaders and all allocated mip targets.
fn void UnrealBloomPass.dispose(&self) {
    self.down.dispose();
    self.up.dispose();
    foreach (m : self.mips) if (m) m.dispose();
}

// ---- LensFlarePass

// Screen-space lens ghosts and halo (`shaders/flare.glsl`) rendered at quarter resolution from a
// blurred bright image and additively composited onto the HDR target.
//
//   LensFlarePass* flare = postprocessing::lens_flare_pass();
//   bloom.render(hdr_target);
//   flare.render(hdr_target, bloom.mips[3]);
struct LensFlarePass {
    float strength; // Additive multiplier applied when compositing `flare` over `target`.
    float threshold; // Linear brightness threshold above which highlights generate flare features.
    int ghosts; // Number of mirrored lens ghosts (`0..8`, default 5).
    float spacing; // Ghost step size along the vector through screen center (UV fraction).
    float halo; // Halo ring radius as a fraction of screen height.
    float dispersion; // Chromatic RGB channel separation as a UV fraction.
    RenderTarget* flare; // Quarter-resolution `RGBA16F` scratch target holding generated flare.
    ShaderPass make; // Flare generation pass (`shaders/flare.glsl`).
    ShaderPass add; // Additive tent upsample pass reusing `shaders/bloom.glsl`.
}

// Allocates a `LensFlarePass` with the given `strength` and linear brightness `threshold`.
fn LensFlarePass* lens_flare_pass(float strength = 0.4, float threshold = 2) {
    LensFlarePass* p = mem::new(LensFlarePass);
    *p = { .strength = strength, .threshold = threshold, .ghosts = 5, .spacing = 0.32f, .halo = 0.42f, .dispersion = 0.006f };
    p.make.init(flare_shader_desc(sg::query_backend()), "drie-flare");
    p.make.set_sampler(SMP_FLARE_SMP, linear_sampler());
    p.add.init(bloom_up_shader_desc(sg::query_backend()), "drie-flare-add", ADDITIVE_BLEND);
    p.add.set_sampler(SMP_BLOOM_UP_SMP, linear_sampler());
    return p;
}

// Generates ghosts and halo from `bright` (typically a coarser `UnrealBloomPass.mips[i]`) and
// additively composites them over `target`.
fn void LensFlarePass.render(&self, RenderTarget* target, RenderTarget* bright) {
    int w = target.width / 4 > 1 ? target.width / 4 : 1;
    int h = target.height / 4 > 1 ? target.height / 4 : 1;
    if (!self.flare) self.flare = drie::render_target(w, h, 1, RGBA16F, false);
    self.flare.set_size(w, h, 1);
    FlareParams p = {
        .size = { w, h, self.threshold, (float)target.width / target.height },
        .tune = { self.ghosts < 0 ? 0 : (self.ghosts > 8 ? 8 : self.ghosts), self.spacing, self.halo, self.dispersion },
    };
    self.make.set_texture(VIEW_FLARE_SRC, bright.texture);
    self.make.set_uniforms(UB_FLARE_PARAMS, &p);
    self.make.render(self.flare);
    BloomUpParams u = {
        .size = { target.width, target.height, 1.0f / w, 1.0f / h },
        .tune = { self.strength, 0, 0, 0 },
    };
    self.add.set_texture(VIEW_BLOOM_UP_SRC, self.flare.texture);
    self.add.set_uniforms(UB_BLOOM_UP_PARAMS, &u);
    self.add.render_over(target);
}

// Disposes the flare shader passes and quarter-resolution scratch target.
fn void LensFlarePass.dispose(&self) {
    self.make.dispose();
    self.add.dispose();
    if (self.flare) self.flare.dispose();
}

// `EffectComposer` entry point: adds bloom in place to `read` for downstream passes.
fn void unreal_bloom_pass_render(
    Pass* pass,
    EffectComposer* composer,
    RenderTarget* write,
    RenderTarget* read,
    float delta
) @local {
    UnrealBloomPass* self = (UnrealBloomPass*)pass;
    self.render(read);
}
