// Color grading in HDR, before the tone curve: the grade, the tone mapping and the output
// encoding baked into one 3D LUT (grade.glsl), which OutputPass reads in place of its tone
// mapping. The scene's color (linear light, after the exposure) is looked up through a log
// shaper (tonemapping.glsl's grade_shape): every stop, 10 under middle grey to 7 over, gets
// the same room in the LUT, so the grade treats a highlight five stops up as finely as a
// shadow, and the LUT's interpolation is even across exposure. The LUT is baked again on the GPU when a setting, the tone mapping or the output's
// colors change. A look, a .cube LUT on the sRGB picture (as a grade from Resolve is exported),
// can go on top.
//
//   ColorGrade* grade = postprocessing::color_grade();
//   grade.temperature = 15;                       // warmer
//   grade.contrast = 1.1;
//   grade.load_look("look.cube")!!;
//   output.grade = grade;                         // OutputPass
module drie::postprocessing;
import std::io, std::math, std::collections::list;
import sokol, drie;

const int GRADE_LUT = 32;

// What is graded, in this order. Neutral: the defaults (color_grade()).
struct GradeSettings {
    // White balance: temperature warms (+) or cools (-) the picture, tint pushes it to magenta
    // (+) or green (-), -100..100. The gains are applied in LMS (the eye's cone responses),
    // as the eye adapts to a light's color.
    float temperature;
    float tint;
    // Contrast round middle grey, in stops (1: none; over 1 more).
    float contrast;
    // ASC CDL's slope, offset and power, per channel: gain multiplies the light, lift raises
    // or lowers the log encoding (0.1 of it per 1), power bends it (over 1 darker mid-tones).
    Vector3 gain;
    Vector3 lift;
    Vector3 power;
    // In Oklab: saturation (1: as it is), a turn of the hue (radians), and the tints added to
    // the shadows and to the highlights (Oklab a: green to red, b: blue to yellow; 0.02 is a
    // clear tint).
    float saturation;
    float hue;
    Vector2 shadows;
    Vector2 highlights;
    float look_strength;
}

struct ColorGrade {
    inline GradeSettings settings;
    SgImage lut;
    SgView lut_texture;
    SgView[GRADE_LUT] slices;
    SgImage look;
    SgView look_texture;
    int look_size;
    SgShader shader;
    SgPipeline pipeline;
    SgSampler sampler;
    // what the LUT was baked with
    GradeSettings baked;
    int baked_tone;
    bool baked_p3;
    bool baked_once;
}

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

SgSampler grade_clamp @local;
SgView grade_empty @local;

fn SgSampler grade_sampler() @local {
    if (!grade_clamp.id) {
        grade_clamp = sg::make_sampler(&&{
            .min_filter = LINEAR,
            .mag_filter = LINEAR,
            .wrap_u = CLAMP_TO_EDGE,
            .wrap_v = CLAMP_TO_EDGE,
            .wrap_w = CLAMP_TO_EDGE,
        });
    }
    return grade_clamp;
}

// A 1x1x1 3D texture, for a 3D slot with nothing in it.
fn SgView empty_lut() {
    if (!grade_empty.id) {
        char[4] px = { 255, 255, 255, 255 };
        SgImage img = sg::make_image(&&{
            .type = TYPE_3D,
        .sample_count = 1,
            .width = 1,
            .height = 1,
            .num_slices = 1,
            .pixel_format = RGBA8,
            .data = { .ptr = &px, .size = 4 },
            .label = "drie-empty-lut",
        });
        grade_empty = sg::make_view(&&{ .texture = { .image = img } });
    }
    return grade_empty;
}

// The gains in LMS that take the white of the temperature and tint (a point near the daylight
// locus, the tint across it) to D65's.
fn Vector3 white_balance(float temperature, float tint) @local {
    float t1 = temperature / 65;
    float t2 = tint / 65;
    float x = 0.31271f - t1 * (t1 < 0 ? 0.1f : 0.05f);
    float y = 2.87f * x - 3 * x * x - 0.27509507f + t2 * 0.05f;
    float bx = x / y;
    float bz = (1 - x - y) / y;
    Vector3 lms = {
        0.7328f * bx + 0.4296f - 0.1624f * bz,
        -0.7036f * bx + 1.6975f + 0.0061f * bz,
        0.0030f * bx + 0.0136f + 0.9834f * bz,
    };
    return (Vector3) { 0.949237f, 1.03542f, 1.08728f } / lms;
}

// Bakes the LUT if a setting, the tone mapping or the output's colors have changed since.
// Outside of any pass.
fn void ColorGrade.prepare(&self, ToneMapping tone, bool p3) {
    if (self.baked_once && mem::equals(&self.baked, &self.settings, GradeSettings::size) && self.baked_tone == tone.ordinal && self.baked_p3 == p3) return;
    if (!self.lut.id) {
        self.lut = sg::make_image(&&{
            .type = TYPE_3D,
        .sample_count = 1,
            .usage = { .color_attachment = true },
            .width = GRADE_LUT,
            .height = GRADE_LUT,
            .num_slices = GRADE_LUT,
            .pixel_format = RGBA16F,
            .label = "drie-grade-lut",
        });
        self.lut_texture = sg::make_view(&&{ .texture = { .image = self.lut } });
        for (int z = 0; z < GRADE_LUT; z++) {
            self.slices[z] = sg::make_view(&&{ .color_attachment = { .image = self.lut, .slice = z } });
        }
        self.shader = drie::make_program(grade_shader_desc(sg::query_backend()));
        self.pipeline = sg::make_pipeline(&&{
            .shader = self.shader,
            .colors[0].pixel_format = RGBA16F,
            .depth.pixel_format = NONE,
            .sample_count = 1,
            .label = "drie-grade",
        });
        self.sampler = grade_sampler();
    }
    Vector3 wb = white_balance(self.temperature, self.tint);
    bool look = self.look.id && self.look_strength > 0;
    SgBindings bind;
    bind.views[VIEW_GRADE_LOOK] = look ? self.look_texture : empty_lut();
    bind.samplers[SMP_GRADE_SMP] = self.sampler;
    for (int z = 0; z < GRADE_LUT; z++) {
        sg::begin_pass(&&{
            .action.colors[0] = { .load_action = DONTCARE, .store_action = STORE },
            .attachments.colors[0] = self.slices[z],
            .label = "drie-grade",
        });
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

faultdef BAD_CUBE;

// A look from a .cube file (LUT_3D_SIZE n, then n³ lines of r g b, red fastest; its input the
// sRGB picture, 0..1), applied after the tone mapping, look_strength of it.
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
        if (line[0] < '0' && line[0] != '-' && line[0] != '.' || line[0] > '9' && line[0] != '-' && line[0] != '.') continue;
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
    self.look = sg::make_image(&&{
        .type = TYPE_3D,
        .sample_count = 1,
        .width = n,
        .height = n,
        .num_slices = n,
        .pixel_format = RGBA8,
        .data = { .ptr = px.ptr, .size = px.len },
        .label = "drie-grade-look",
    });
    self.look_texture = sg::make_view(&&{ .texture = { .image = self.look } });
    self.look_size = n;
    self.baked_once = false;
}
