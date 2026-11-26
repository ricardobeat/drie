// HDR color grading (`ColorGrade`, `shaders/grade.glsl`) baked together with tone mapping and
// output encoding into a 32x32x32 `RGBA16F` 3D LUT sampled by `OutputPass` on SDR output.
// Linear post-exposure scene colors enter the LUT through a log shaper (`grade_shape` in
// `shaders/tonemapping.glsl`, spanning 10 stops below to 7 stops above middle grey) so precision
// stays even across shadows and highlights. An optional `.cube` 3D look LUT (sRGB `0..1`) runs
// after tone mapping; `prepare()` rebakes slices whenever settings, tone mapping, or P3 change.

module drie::postprocessing;
import std::io, std::math, std::collections::list;
import sokol, drie;

// ---- Grade settings & state

// Edge length of the baked 3D grading LUT (`32 x 32 x 32` `RGBA16F`).
const int GRADE_LUT = 32;

// Color-grading parameters applied in field order before tone mapping (except `look_strength`,
// applied after). `color_grade()` initializes all fields to neutral defaults.
struct GradeSettings {
    float temperature; // White-balance warmth in LMS cone space (`-100..100`; positive = warmer).
    float tint; // White-balance green/magenta axis in LMS space (`-100..100`; positive = magenta).
    float contrast; // Contrast around middle grey in log stops (`1` = neutral; `> 1` = higher).
    Vector3 gain; // Per-channel linear slope (ASC CDL gain; `{ 1, 1, 1 }` = neutral).
    Vector3 lift; // Per-channel log-space offset (`0.1` of log range per `1.0`; `{ 0, 0, 0 }` = neutral).
    Vector3 power; // Per-channel log-space power curve (`> 1` darkens midtones; `{ 1, 1, 1 }` = neutral).
    float saturation; // Oklab chroma scale (`1` = neutral; `0` = monochrome).
    float hue; // Oklab hue rotation in radians.
    Vector2 shadows; // Oklab `(a, b)` tint added to shadows (~`0.02` produces a visible cast).
    Vector2 highlights; // Oklab `(a, b)` tint added to highlights (~`0.02` produces a visible cast).
    float look_strength; // Blend factor (`0..1`) for the optional `.cube` look LUT after tone mapping.
}

// Stateful HDR color grader and cached 3D LUT consumed by `OutputPass.grade`.
//
//   ColorGrade* grade = postprocessing::color_grade();
//   grade.temperature = 15;
//   grade.contrast = 1.1;
//   grade.load_look("look.cube")!!;
//   output.grade = grade;
struct ColorGrade {
    inline GradeSettings settings;
    SgImage lut; // Baked `GRADE_LUT`^3 `RGBA16F` 3D LUT image.
    SgView lut_texture; // 3D texture view sampled by `OutputPass`.
    SgView[GRADE_LUT] slices; // Per-slice color attachment views used while baking `lut`.
    SgImage look; // Optional `.cube` `RGBA8` 3D look texture.
    SgView look_texture; // Texture view for `look`.
    int look_size; // Edge length of `look` (`0` when no look is loaded).
    SgShader shader; // Baking shader (`shaders/grade.glsl`).
    SgPipeline pipeline; // Baking pipeline targeting `RGBA16F` slices.
    SgSampler sampler; // Trilinear clamp-to-edge sampler for `look`.
    GradeSettings baked; // Settings snapshot stored in `lut`.
    int baked_tone; // `ToneMapping.ordinal` baked into `lut`.
    bool baked_p3; // Display P3 flag baked into `lut`.
    bool baked_once; // True once `lut` contains a valid bake.
}

// Allocates a `ColorGrade` initialized to neutral grading defaults.
fn ColorGrade* color_grade() {
    ColorGrade* g = mem::new(ColorGrade);
    g.settings = {
        .contrast = 1,
        .gain = { 1, 1, 1 },
        .power = { 1, 1, 1 },
        .saturation = 1,
        .look_strength = 1,
    };
    return g;
}

// ---- LUT baking

SgSampler grade_clamp @local;
SgView grade_empty @local;

fn SgSampler grade_sampler() @local {
    if (!grade_clamp.id) {
        grade_clamp = sg::make_sampler(
            &&{
                .min_filter = LINEAR,
                .mag_filter = LINEAR,
                .wrap_u = CLAMP_TO_EDGE,
                .wrap_v = CLAMP_TO_EDGE,
                .wrap_w = CLAMP_TO_EDGE,
            }
        );
    }
    return grade_clamp;
}

// Returns a shared 1x1x1 white `RGBA8` 3D texture view for binding unused 3D LUT slots.
fn SgView empty_lut() {
    if (!grade_empty.id) {
        char[4] px = { 255, 255, 255, 255 };
        SgImage img = sg::make_image(
            &&{
                .type = TYPE_3D,
                .sample_count = 1,
                .width = 1,
                .height = 1,
                .num_slices = 1,
                .pixel_format = RGBA8,
                .data = { .ptr = &px, .size = 4 },
                .label = "drie-empty-lut",
            }
        );
        grade_empty = sg::make_view(&&{ .texture = { .image = img } });
    }
    return grade_empty;
}

// Computes LMS cone gains mapping the `(temperature, tint)` white point to D65.
fn Vector3 white_balance(float temperature, float tint) @local {
    float t1 = temperature / 65;
    float t2 = tint / 65;
    float x  = 0.31271f - t1 * (t1 < 0 ? 0.1f : 0.05f);
    float y  = 2.87f * x - 3 * x * x - 0.27509507f + t2 * 0.05f;
    float bx = x / y;
    float bz = (1 - x - y) / y;
    Vector3 lms = {
        0.7328f * bx + 0.4296f - 0.1624f * bz,
        -0.7036f * bx + 1.6975f + 0.0061f * bz,
        0.0030f * bx + 0.0136f + 0.9834f * bz,
    };
    return (Vector3) { 0.949237f, 1.03542f, 1.08728f } / lms;
}

// Rebakes the 32x32x32 3D LUT on the GPU if `settings`, `tone`, or `p3` changed since the last
// bake. Must be called outside any active render pass.
fn void ColorGrade.prepare(&self, ToneMapping tone, bool p3) {
    if (self.baked_once && mem::equals(
        &self.baked,
        &self.settings,
        GradeSettings::size
    ) && self.baked_tone == tone.ordinal && self.baked_p3 == p3) return;
    if (!self.lut.id) {
        self.lut = sg::make_image(
            &&{
                .type = TYPE_3D,
                .sample_count = 1,
                .usage = { .color_attachment = true },
                .width = GRADE_LUT,
                .height = GRADE_LUT,
                .num_slices = GRADE_LUT,
                .pixel_format = RGBA16F,
                .label = "drie-grade-lut",
            }
        );
        self.lut_texture = sg::make_view(&&{ .texture = { .image = self.lut } });
        for (int z = 0; z < GRADE_LUT; z++) {
            self.slices[z] = sg::make_view(&&{ .color_attachment = { .image = self.lut, .slice = z } });
        }
        self.shader = drie::make_program(grade_shader_desc(sg::query_backend()));
        self.pipeline = sg::make_pipeline(
            &&{
                .shader = self.shader,
                .colors[0].pixel_format = RGBA16F,
                .depth.pixel_format = NONE,
                .sample_count = 1,
                .label = "drie-grade",
            }
        );
        self.sampler = grade_sampler();
    }
    Vector3 wb = white_balance(self.temperature, self.tint);
    bool look  = self.look.id && self.look_strength > 0;
    SgBindings bind;
    bind.views[VIEW_GRADE_LOOK] = look ? self.look_texture : empty_lut();
    bind.samplers[SMP_GRADE_SMP] = self.sampler;
    for (int z = 0; z < GRADE_LUT; z++) {
        sg::begin_pass(
            &&{
                .action.colors[0] = { .load_action = DONTCARE, .store_action = STORE },
                .attachments.colors[0] = self.slices[z],
                .label = "drie-grade",
            }
        );
        sg::apply_pipeline(self.pipeline);
        sg::apply_bindings(&bind);
        GradeParams p = {
            .lut = { z, GRADE_LUT, tone.ordinal, p3 ? 1 : 0 },
            .balance = { wb.x, wb.y, wb.z, 0 },
            .lift = { self.lift.x, self.lift.y, self.lift.z, self.contrast },
            .power = { self.power.x, self.power.y, self.power.z, self.saturation },
            .gain = { self.gain.x, self.gain.y, self.gain.z, self.hue },
            .shadows = { self.shadows.x, self.shadows.y, look ? self.look_strength : 0, self.look_size },
            .highlights = { self.highlights.x, self.highlights.y, 0, 0 },
        };
        sg::apply_uniforms(UB_GRADE_PARAMS, &&{ .ptr = &p, .size = GradeParams::size });
        sg::draw(0, 3, 1);
        sg::end_pass();
    }
    self.baked = self.settings;
    self.baked_tone = tone.ordinal;
    self.baked_p3 = p3;
    self.baked_once = true;
}

// ---- Look loading

// Returned by `ColorGrade.load_look()` when a `.cube` file has an invalid size or too few entries.
faultdef BAD_CUBE;

// Loads a 3D `.cube` look LUT (`LUT_3D_SIZE n` followed by `n^3` RGB triples, red fastest,
// expecting sRGB `0..1` input after tone mapping), uploads it to the GPU, and marks `lut` dirty.
fn void? ColorGrade.load_look(&self, String path) {
    char[] text = file::load_temp(path)!;
    int n;
    List{float} values;
    values.tinit();
    foreach (raw : ((String)text).tsplit("\n")) {
        String line = raw.trim();
        if (!line.len || line[0] == '#') continue;
        if (line.starts_with("LUT_3D_SIZE")) {
            n = line[11..].trim().to_int()!;
            continue;
        }
        if (line[0] < '0' && line[0] != '-' && line[0] != '.'
            || line[0] > '9' && line[0] != '-' && line[0] != '.') continue;
        foreach (f : line.tsplit(" ")) {
            if (f.len) values.push((float)f.to_double()!);
        }
    }
    sz cells = (sz)n * n * n;
    if (n < 2 || values.len() < cells * 3) return BAD_CUBE~;
    char[] px = mem::new_array(char, cells * 4);
    defer free(px);
    for (sz i = 0; i < cells; i++) {
        for (sz k = 0; k < 3; k++) px[i * 4 + k] = (char)(int)(math::clamp(values[i * 3 + k], 0.0f, 1.0f) * 255 + 0.5f);
        px[i * 4 + 3] = 255;
    }
    if (self.look.id) sg::destroy_image(self.look);
    self.look = sg::make_image(
        &&{
            .type = TYPE_3D,
            .sample_count = 1,
            .width = n,
            .height = n,
            .num_slices = n,
            .pixel_format = RGBA8,
            .data = { .ptr = px.ptr, .size = px.len },
            .label = "drie-grade-look",
        }
    );
    self.look_texture = sg::make_view(&&{ .texture = { .image = self.look } });
    self.look_size = n;
    self.baked_once = false;
}