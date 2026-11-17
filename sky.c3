// Like THREE's Sky (examples/jsm/objects/Sky.js, r187): Preetham's daylight sky with its sun
// disc and a layer of cloud, on a dome around the camera. The parameters are three's uniforms.
//
//     Sky* sky = objects::sky();
//     sky.rayleigh = 3;
//     sky.sun_position = { -0.8, 0.19, 0.56 };
//     scene.add(sky);
//     // before each render, and before capturing the scene into an environment map:
//     sky.follow(camera.world_position(), camera.far);
//
// three's shader runs per pixel; here the same equations fill an equirectangular HDR texture
// (update(), whenever a parameter changed) that the dome shows with a MeshBasicMaterial. three
// draws its dome on the far plane whatever its size; follow() puts this one just inside it.
module drie::objects;
import std::math, drie;

struct Sky {
    inline Mesh mesh;
    float turbidity;
    float rayleigh;
    float mie_coefficient;
    float mie_directional_g;
    // the sun's direction (three's sunPosition, which need not be normalized)
    Vector3 sun_position;
    float cloud_scale;
    float cloud_speed;
    float cloud_coverage;
    float cloud_density;
    float cloud_elevation;
    bool show_sun_disc;
    float time;
    // the texture's size: width by width / 2
    int resolution;
    SkyParams drawn;
    bool has_drawn;
}

struct SkyParams @local {
    float turbidity, rayleigh, mie_coefficient, mie_directional_g;
    Vector3 sun_position;
    float cloud_scale, cloud_speed, cloud_coverage, cloud_density, cloud_elevation;
    bool show_sun_disc;
    float time;
    int resolution;
}

fn Sky* sky(int resolution = 1024) {
    Sky* s = mem::new(Sky);
    s.init(MESH);
    s.instance_data.allocator = mem;
    s.geometry = drie::sphere_geometry(1, 64, 32);
    MeshBasicMaterial* m = drie::mesh_basic_material(side: BACK_SIDE, fog: false);
    m.depth_write = false;
    s.material = m;
    s.frustum_culled = false;
    s.turbidity = 2;
    s.rayleigh = 1;
    s.mie_coefficient = 0.005f;
    s.mie_directional_g = 0.8f;
    s.cloud_scale = 0.0002f;
    s.cloud_speed = 0.00002f;
    s.cloud_coverage = 0.4f;
    s.cloud_density = 0.4f;
    s.cloud_elevation = 0.5f;
    s.show_sun_disc = true;
    s.resolution = resolution;
    return s;
}

fn SkyParams sky_params(Sky* s) @local {
    return {
        s.turbidity,
        s.rayleigh,
        s.mie_coefficient,
        s.mie_directional_g,
        s.sun_position,
        s.cloud_scale,
        s.cloud_speed,
        s.cloud_coverage,
        s.cloud_density,
        s.cloud_elevation,
        s.show_sun_disc,
        s.time,
        s.resolution,
    };
}

fn bool sky_params_equal(SkyParams* a, SkyParams* b) @local {
    return a.turbidity == b.turbidity && a.rayleigh == b.rayleigh && a.mie_coefficient == b.mie_coefficient
        && a.mie_directional_g == b.mie_directional_g
        && a.sun_position == b.sun_position
        && a.cloud_scale == b.cloud_scale
        && a.cloud_speed == b.cloud_speed
        && a.cloud_coverage == b.cloud_coverage
        && a.cloud_density == b.cloud_density
        && a.cloud_elevation == b.cloud_elevation
        && a.show_sun_disc == b.show_sun_disc
        && a.time == b.time
        && a.resolution == b.resolution;
}

// Centers the dome on `eye`, just inside a camera's far plane, and fills its texture again if a
// parameter changed.
fn void Sky.follow(&self, Vector3 eye, float far) {
    self.position = eye;
    float r = far * 0.9f;
    self.scale = { r, r, r };
    // also for a scene whose matrices are set from outside (Scene.no_matrix_world_update)
    self.update_matrix();
    self.matrix_world = self.parent ? self.parent.matrix_world.mul(self.matrix) : self.matrix;
    self.update();
}

// Fills the texture again if a parameter changed since it was last filled.
fn void Sky.update(&self) {
    SkyParams p = sky_params(self);
    if (self.has_drawn && sky_params_equal(&p, &self.drawn)) return;
    self.drawn = p;
    self.has_drawn = true;
    int w          = math::max(self.resolution, 8);
    int h          = w / 2;
    float[] pixels = mem::new_array(float, (sz)w * h * 4);
    defer free(pixels);
    SkyLight light = sky_light(&p);
    for (int y = 0; y < h; y++) {
        // the sphere's uv.y is 1 at the top (y up), which samples the last row
        float theta = (1 - ((float)y + 0.5f) / h) * (float)drie::PI;
        for (int x = 0; x < w; x++) {
            float phi = ((float)x + 0.5f) / w * 2 * (float)drie::PI;
            Vector3 d = { -math::cos(phi) * math::sin(theta), math::cos(theta), math::sin(phi) * math::sin(theta) };
            Vector3 c = sky_color(&p, &light, d);
            sz o      = ((sz)y * w + x) * 4;
            pixels[o] = c.x;
            pixels[o + 1] = c.y;
            pixels[o + 2] = c.z;
            pixels[o + 3] = 1;
        }
    }
    Texture* t = drie::float_texture(pixels, w, h);
    t.wrap_t = CLAMP_TO_EDGE_WRAPPING;
    Material* m = self.material;
    if (m.map) m.map.dispose();
    m.map = t;
}

// ----------------------------------------------------------------------------
// three's SkyShader, its vertex shader's part (the same for every direction) and then its
// fragment shader's.

struct SkyLight @local {
    Vector3 sun;
    float sun_e;
    Vector3 beta_r;
    Vector3 beta_m;
}

const Vector3 SKY_TOTAL_RAYLEIGH @local = { 5.804542996261093E-6f, 1.3562911419845635E-5f, 3.0265902468824876E-5f };
const Vector3 SKY_MIE_CONST @local      = { 1.8399918514433978E14f, 2.7798023919660528E14f, 4.0790479543861094E14f };

fn SkyLight sky_light(SkyParams* p) @local {
    SkyLight l;
    float len = math::sqrt(p.sun_position.dot(p.sun_position));
    l.sun = len > 0 ? p.sun_position / len : (Vector3) { 0, 1, 0 };
    // sunIntensity, with the earth's shadow cutting off below the horizon
    const float CUTOFF = 1.6110731556870734f;
    float zenith       = math::clamp(l.sun.y, -1.0f, 1.0f);
    l.sun_e = 1000 * math::max(0.0f, 1 - math::exp(-((CUTOFF - math::acos(zenith)) / 1.5f)));
    float sunfade  = 1 - math::clamp(1 - math::exp(p.sun_position.y / 450000), 0.0f, 1.0f);
    float rayleigh = p.rayleigh - (1 - sunfade);
    l.beta_r = SKY_TOTAL_RAYLEIGH * rayleigh;
    float c = (0.2f * p.turbidity) * 10E-18f;
    l.beta_m = SKY_MIE_CONST * (0.434f * c) * p.mie_coefficient;
    return l;
}

fn Vector3 sky_pow(Vector3 v, float e) @local => (Vector3) { math::pow(v.x, e), math::pow(v.y, e), math::pow(v.z, e) };
fn Vector3 sky_exp(Vector3 v) @local => (Vector3) { math::exp(v.x), math::exp(v.y), math::exp(v.z) };
fn float sky_fract(float x) @local => x - math::floor(x);
fn float sky_mix(float a, float b, float t) @local => a + (b - a) * t;
fn Vector3 sky_mix3(Vector3 a, Vector3 b, float t) @local => a + (b - a) * t;
fn Vector3 sky_mixv(Vector3 a, Vector3 b, Vector3 t) @local => a + (b - a) * t;

fn float sky_smoothstep(float e0, float e1, float x) @local {
    float t = math::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3 - 2 * t);
}

// gradient at a lattice corner (three's sinless hash)
fn float[<2>] sky_gradient(float[<2>] i) @local {
    float[<3>] p = { sky_fract(i.x * 0.1031f), sky_fract(i.y * 0.1030f), sky_fract(i.x * 0.0973f) };
    float k      = p.x * (p.y + 33.33f) + p.y * (p.z + 33.33f) + p.z * (p.x + 33.33f);
    p += (float[<3>]) { k, k, k };
    return (float[<2>]) { sky_fract((p.x + p.y) * p.z) * 2 - 1, sky_fract((p.x + p.z) * p.y) * 2 - 1 };
}

fn float sky_dot2(float[<2>] a, float[<2>] b) @local => a.x * b.x + a.y * b.y;

fn float sky_noise(float[<2>] p) @local {
    float[<2>] i = { math::floor(p.x), math::floor(p.y) };
    float[<2>] f = p - i;
    float[<2>] u = f * f * f * (f * (f * 6 - 15) + 10);
    float a      = sky_dot2(sky_gradient(i), f);
    float b      = sky_dot2(sky_gradient(i + (float[<2>]) { 1, 0 }), f - (float[<2>]) { 1, 0 });
    float c      = sky_dot2(sky_gradient(i + (float[<2>]) { 0, 1 }), f - (float[<2>]) { 0, 1 });
    float d      = sky_dot2(sky_gradient(i + (float[<2>]) { 1, 1 }), f - (float[<2>]) { 1, 1 });
    return sky_mix(sky_mix(a, b, u.x), sky_mix(c, d, u.x), u.y) * 1.6f;
}

fn float sky_fbm(float[<2>] p, float drift) @local {
    float result    = 0;
    float amplitude = 1;
    for (int i = 0; i < 4; i++) {
        result += amplitude * sky_noise(p);
        amplitude *= 0.5f;
        p = p * 2 + (float[<2>]) { drift, drift };
    }
    return result;
}

fn Vector3 sky_color(SkyParams* p, SkyLight* l, Vector3 dir) @local {
    const float PI_F   = (float)drie::PI;
    // optical length, cut off at 90 degrees to avoid the formula's singularity
    float zenith_angle = math::acos(math::max(0.0f, dir.y));
    float inverse      = 1 / (math::cos(
        zenith_angle
    ) + 0.15f * math::pow(93.885f - zenith_angle * 180 / PI_F, -1.253f));
    float s_r          = 8.4E3f * inverse;
    float s_m          = 1.25E3f * inverse;
    Vector3 fex        = sky_exp(-(l.beta_r * s_r + l.beta_m * s_m));

    // in-scattering
    float cos_theta      = dir.dot(l.sun);
    float rc             = cos_theta * 0.5f + 0.5f;
    float r_phase        = 0.05968310365946075f * (1 + rc * rc);
    Vector3 beta_r_theta = l.beta_r * r_phase;
    float g              = p.mie_directional_g;
    float g2             = g * g;
    float m_phase        = 0.07957747154594767f * ((1 - g2) / math::pow(1 - 2 * g * cos_theta + g2, 1.5f));
    Vector3 beta_m_theta = l.beta_m * m_phase;
    Vector3 ratio        = (beta_r_theta + beta_m_theta) / (l.beta_r + l.beta_m);
    Vector3 lin          = sky_pow(ratio * l.sun_e * ((Vector3) { 1, 1, 1 } - fex), 1.5f);
    Vector3 lin_sun      = sky_pow(ratio * l.sun_e * fex, 0.5f);
    lin *= sky_mix3({ 1, 1, 1 }, lin_sun, math::clamp(math::pow(1 - l.sun.y, 5.0f), 0.0f, 1.0f));

    Vector3 l0            = fex * 0.1f;
    // the sun's disc
    float sundisc         = math::clamp(
        (cos_theta - 0.999956676946448443553574619906976478926848692873900859324f) * 50000,
        0.0f,
        1.0f
    ) * (p.show_sun_disc ? 1 : 0);
    Vector3 sun_e_fex     = l.sun_e * fex;
    Vector3 sundisc_color = (Vector3) {
        math::min(sun_e_fex.x, 80.0f),
        math::min(sun_e_fex.y, 80.0f),
        math::min(sun_e_fex.z, 80.0f)
    } * (760 * sundisc);
    Vector3 tex           = (lin + l0) * 0.04f + sundisc_color + (Vector3) { 0, 0.0003f, 0.00075f };

    if (dir.y > 0 && p.cloud_coverage > 0) {
        // onto the cloud plane (higher elevation brings the clouds lower and closer)
        float elevation     = sky_mix(1, 0.1f, p.cloud_elevation);
        float[<2>] cloud_uv = (float[<2>]) { dir.x, dir.z } / (dir.y * elevation);
        cloud_uv *= p.cloud_scale;
        cloud_uv += (float[<2>]) { p.time * p.cloud_speed, p.time * p.cloud_speed };
        float evolve       = p.time * p.cloud_speed * 300;
        float cloud_noise  = math::clamp(sky_fbm(cloud_uv * 1000, evolve) * 0.7f + 0.5f, 0.0f, 1.0f);
        float region       = sky_noise(cloud_uv * 300) * 0.37f + 0.5f;
        float cov          = math::clamp(p.cloud_coverage + (region - 0.5f) * 0.6f, 0.0f, 1.0f);
        float threshold    = 1 - cov;
        float cloud_mask   = sky_smoothstep(threshold, threshold + 0.3f, cloud_noise);
        float horizon_fade = sky_smoothstep(0, 0.03f + 0.06f * p.cloud_elevation, dir.y);
        cloud_mask *= horizon_fade;

        float day           = sky_smoothstep(-0.08f, 0.3f, l.sun.y);
        Vector3 sun_color   = sun_e_fex * (0.22f * 0.04f);
        Vector3 sky_ambient = lin * 0.04f + (Vector3) { 0, 0.0003f, 0.00075f };
        float depth         = math::max(0.0f, cloud_noise - threshold);
        float beer          = math::exp(depth * -4);
        float powder        = 1 - beer * beer;
        float shade         = sky_mix(0.45f, 1, math::clamp(beer * powder * 2.6f, 0.0f, 1.0f));
        float silver        = math::clamp(0.51f / math::pow(1.49f - cos_theta * 1.4f, 1.5f), 0.0f, 3.0f);
        float edge          = cloud_mask * (1 - cloud_mask) * 4;
        Vector3 cloud_color = sky_ambient + sun_color * shade;
        cloud_color += sun_color * (silver * edge * 0.6f);
        cloud_color *= math::max(day, 0.03f);
        float alpha = (1 - math::exp(depth * p.cloud_density * -12)) * horizon_fade;
        tex -= l0 * (0.04f * alpha);
        Vector3 aerial = sky_mixv(tex, cloud_color, fex);
        tex = sky_mix3(tex, aerial, alpha);
    }
    return tex;
}