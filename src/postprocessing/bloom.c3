module drie::postprocessing;
import std::collections::list;
import sokol, drie;

// The glow round what is bright, like three's UnrealBloomPass (strength, radius, threshold),
// made as Call of Duty's and Unreal's bloom is (see bloom.glsl): the picture (linear light)
// halved `levels` times, then added back up, each level through a tent filter into the one
// above, and the result added to the picture. With threshold 0 everything glows a little, by
// how bright it is, as through a real lens (keep strength low then: 0.02 to 0.1); a threshold
// makes only what is brighter glow.
//
//   UnrealBloomPass* bloom = postprocessing::unreal_bloom_pass(strength: 0.05, threshold: 0);
//   bloom.render(hdr_target);           // adds the glow to its picture
struct UnrealBloomPass {
    inline Pass pass;
    // how much of the glow is added
    float strength;
    // how far it spreads, 0..1: each level, twice as wide as the one above, counts this much
    // of it (1: all as much)
    float radius;
    // the brightness (linear light) above which things glow, and how softly they start to
    float threshold;
    float knee;
    // levels of halving (1..8): more spreads the glow further
    int levels;
    RenderTarget*[8] mips;
    ShaderPass down;
    ShaderPass up;
}

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

// Adds the glow of `target`'s picture (linear light) to it.
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
    // down, each level from the one above
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
    // up, each level added into the one above it; the levels' weights 1, r, r^2...
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

fn void UnrealBloomPass.dispose(&self) {
    self.down.dispose();
    self.up.dispose();
    foreach (m : self.mips) if (m) m.dispose();
}

// A lens's ghosts and halo of what is bright (see flare.glsl): render() makes them from a
// blurred bright picture, one of the bloom's smaller levels, say (the blurrier, the softer the
// ghosts), and adds them to the target's picture.
//
//   LensFlarePass* flare = postprocessing::lens_flare_pass();
//   bloom.render(hdr_target);
//   flare.render(hdr_target, bloom.mips[3]);
struct LensFlarePass {
    // how much of the flare is added
    float strength;
    // the brightness (linear light) the flare is made from above
    float threshold;
    // ghosts (up to 8), their spacing along the line through the middle (a part of the
    // picture), the halo's radius (a part of the picture's height), and how far apart red,
    // green and blue are (a part of the picture)
    int ghosts;
    float spacing;
    float halo;
    float dispersion;
    RenderTarget* flare;
    ShaderPass make;
    ShaderPass add;
}

fn LensFlarePass* lens_flare_pass(float strength = 0.4, float threshold = 2) {
    LensFlarePass* p = mem::new(LensFlarePass);
    *p = { .strength = strength, .threshold = threshold, .ghosts = 5, .spacing = 0.32f, .halo = 0.42f, .dispersion = 0.006f };
    p.make.init(flare_shader_desc(sg::query_backend()), "drie-flare");
    p.make.set_sampler(SMP_FLARE_SMP, linear_sampler());
    p.add.init(bloom_up_shader_desc(sg::query_backend()), "drie-flare-add", ADDITIVE_BLEND);
    p.add.set_sampler(SMP_BLOOM_UP_SMP, linear_sampler());
    return p;
}

// Adds the flare of `bright` (a blurred picture of the target's, any size) to `target`.
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

fn void LensFlarePass.dispose(&self) {
    self.make.dispose();
    self.add.dispose();
    if (self.flare) self.flare.dispose();
}

// In an EffectComposer (RGBA16F): adds the glow to the picture, which the next pass reads.
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
