// Preetham daylight sky dome with a sun disc and procedural cloud layer, modeled on three.js Sky.
// Evaluates per pixel on the GPU via a `drie::nodes` graph on a back-side MeshBasicMaterial;
// `update()` only uploads uniforms when parameters change, and `follow()` keeps the dome centered
// on the camera just inside its far plane.
//
//   Sky* sky = objects::sky();
//   sky.rayleigh = 3;
//   sky.sun_position = { -0.8, 0.19, 0.56 };
//   scene.add(sky);
//   sky.follow(camera.world_position(), camera.far);
module drie::objects;
import std::math, drie, drie::nodes;

// Sky dome mesh driven by Preetham atmospheric scattering and procedural fbm clouds.
struct Sky {
    inline Mesh mesh; // Base dome mesh (back-side unit sphere scaled by `follow()`).
    float turbidity; // Atmospheric haziness (default 2).
    float rayleigh; // Rayleigh scattering scale (default 1).
    float mie_coefficient; // Mie scattering coefficient (default 0.005).
    float mie_directional_g; // Henyey-Greenstein phase asymmetry in [-1, 1] (default 0.8).
    Vector3 sun_position; // Sun direction vector; need not be normalized.
    float cloud_scale; // Spatial frequency of the cloud noise layer (default 0.0002).
    float cloud_speed; // Cloud drift speed scaled by `time` (default 0.00002).
    float cloud_coverage; // Cloud coverage in [0, 1]; 0 disables clouds (default 0.4).
    float cloud_density; // Optical thickness of cloud volume (default 0.4).
    float cloud_elevation; // Normalized cloud layer height affecting horizon fade (default 0.5).
    bool show_sun_disc; // Draws the bright solar disc when true.
    float time; // Elapsed time driving cloud drift and evolution.
    int resolution; // Stored nominal resolution; ignored by the per-pixel sky shader.
    SkyParams drawn; // Last parameter snapshot uploaded to uniforms.
    bool has_drawn; // True after the first uniform upload.
    Node u_sun; // Uniform vec4: normalized sun direction and solar intensity.
    Node u_beta_r; // Uniform vec4: Rayleigh beta RGB and `mie_directional_g`.
    Node u_beta_m; // Uniform vec4: Mie beta RGB and `show_sun_disc` flag.
    Node u_cloud0; // Uniform vec4: cloud scale, speed, coverage, density.
    Node u_cloud1; // Uniform vec4: cloud elevation and `time`.
}

struct SkyParams @local {
    float turbidity, rayleigh, mie_coefficient, mie_directional_g;
    Vector3 sun_position;
    float cloud_scale, cloud_speed, cloud_coverage, cloud_density, cloud_elevation;
    bool show_sun_disc;
    float time;
    int resolution;
}

fn Node sky_gradient_node(Node i) @local {
    Node p = nodes::variable(nodes::fract(nodes::vec3(i.x(), i.y(), i.x()) * nodes::vec3(0.1031f, 0.1030f, 0.0973f)));
    Node pyzx = nodes::vec3(p.y(), p.z(), p.x()) + 33.33f;
    Node body = nodes::block(p, p.assign(p + nodes::dot(p, pyzx)));
    Node res = nodes::fract(
        (nodes::vec2(p.x(), p.x()) + nodes::vec2(p.y(), p.z())) * nodes::vec2(p.z(), p.y())
    ) * 2.0f - 1.0f;
    return nodes::seq(body, res);
}

fn Node sky_noise_node(Node p_in) @local {
    Node p   = nodes::variable(p_in);
    Node i   = nodes::variable(nodes::floor(p));
    Node f   = nodes::variable(p - i);
    Node u   = nodes::variable(f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f));
    Node a   = nodes::dot(sky_gradient_node(i), f);
    Node b   = nodes::dot(sky_gradient_node(i + nodes::vec2(1.0f, 0.0f)), f - nodes::vec2(1.0f, 0.0f));
    Node c   = nodes::dot(sky_gradient_node(i + nodes::vec2(0.0f, 1.0f)), f - nodes::vec2(0.0f, 1.0f));
    Node d   = nodes::dot(sky_gradient_node(i + nodes::vec2(1.0f, 1.0f)), f - nodes::vec2(1.0f, 1.0f));
    Node res = nodes::mix(nodes::mix(a, b, u.x()), nodes::mix(c, d, u.x()), u.y()) * 1.6f;
    return nodes::seq(nodes::block(p, i, f, u), res);
}

fn Node sky_fbm_node(Node pos_in, Node drift_in) @local {
    Node p      = nodes::variable(pos_in);
    Node drift  = nodes::variable(drift_in);
    Node result = nodes::variable(nodes::num(0.0f));
    Node amp    = nodes::variable(nodes::num(1.0f));
    Node idx    = nodes::loop_index();
    Node body = nodes::block(
        p,
        drift,
        result,
        amp,
        nodes::loop(
            idx,
            nodes::int_(0),
            nodes::int_(4),
            nodes::block(
                result.assign(result + amp * sky_noise_node(p)),
                amp.assign(amp * 0.5f),
                p.assign(p * 2.0f + drift)
            )
        )
    );
    return nodes::seq(body, result);
}

fn Node sky_graph(Sky* s) @local {
    Node sun             = s.u_sun.xyz();
    Node sun_e           = s.u_sun.w();
    Node beta_r          = s.u_beta_r.xyz();
    Node g               = s.u_beta_r.w();
    Node beta_m          = s.u_beta_m.xyz();
    Node show_disc       = s.u_beta_m.w();
    Node cloud_scale     = s.u_cloud0.x();
    Node cloud_speed     = s.u_cloud0.y();
    Node cloud_coverage  = s.u_cloud0.z();
    Node cloud_density   = s.u_cloud0.w();
    Node cloud_elevation = s.u_cloud1.x();
    Node t               = s.u_cloud1.y();

    Node dir    = nodes::variable(nodes::normalize(nodes::position_world() - nodes::camera_position()));
    Node zenith = nodes::variable(nodes::acos(nodes::max(nodes::num(0.0f), dir.y())));
    Node inv = nodes::variable(
        1.0f / (nodes::cos(zenith) + 0.15f * nodes::pow(93.885f - zenith * (180.0f / (float)drie::PI), -1.253f))
    );
    Node fex = nodes::variable(nodes::exp((beta_r * (8.4E3f * inv) + beta_m * (1.25E3f * inv)).negate()));

    Node cos_theta    = nodes::variable(nodes::dot(dir, sun));
    Node rc           = cos_theta * 0.5f + 0.5f;
    Node beta_r_theta = beta_r * (0.05968310365946075f * (1.0f + rc * rc));
    Node g2           = g * g;
    Node m_phase      = 0.07957747154594767f * ((1.0f - g2) / nodes::pow(1.0f - 2.0f * g * cos_theta + g2, 1.5f));
    Node beta_m_theta = beta_m * m_phase;
    Node ratio        = nodes::variable((beta_r_theta + beta_m_theta) / (beta_r + beta_m));

    Node lin_base  = nodes::pow(nodes::max(ratio * sun_e * (1.0f - fex), nodes::vec3(0.0f)), nodes::vec3(1.5f));
    Node lin_sun   = nodes::pow(nodes::max(ratio * sun_e * fex, nodes::vec3(0.0f)), nodes::vec3(0.5f));
    Node sun_blend = nodes::clamp(nodes::pow(nodes::max(1.0f - sun.y(), 0.0f), 5.0f), 0.0f, 1.0f);
    Node lin       = nodes::variable(lin_base * nodes::mix(nodes::vec3(1.0f), lin_sun, sun_blend));

    Node l0            = nodes::variable(fex * 0.1f);
    Node sundisc       = nodes::clamp((cos_theta - 0.9999566769464484f) * 50000.0f, 0.0f, 1.0f) * show_disc;
    Node sun_e_fex     = nodes::variable(sun_e * fex);
    Node sundisc_color = nodes::min(sun_e_fex, nodes::vec3(80.0f)) * (760.0f * sundisc);
    Node tex           = nodes::variable((lin + l0) * 0.04f + sundisc_color + nodes::vec3(0.0f, 0.0003f, 0.00075f));

    // Clouds (only when looking above the horizon and coverage > 0)
    Node elev = nodes::mix(nodes::num(1.0f), nodes::num(0.1f), cloud_elevation);
    Node cloud_uv = nodes::variable(
        nodes::vec2(dir.x(), dir.z()) / (dir.y() * elev) * cloud_scale + nodes::vec2(t * cloud_speed, t * cloud_speed)
    );
    Node evolve = t * cloud_speed * 300.0f;
    Node cloud_noise = nodes::variable(
        nodes::clamp(sky_fbm_node(cloud_uv * 1000.0f, nodes::vec2(evolve, evolve)) * 0.7f + 0.5f, 0.0f, 1.0f)
    );
    Node region       = sky_noise_node(cloud_uv * 300.0f) * 0.37f + 0.5f;
    Node cov          = nodes::clamp(cloud_coverage + (region - 0.5f) * 0.6f, 0.0f, 1.0f);
    Node threshold    = nodes::variable(1.0f - cov);
    Node cloud_mask   = nodes::variable(nodes::smoothstep(threshold, threshold + 0.3f, cloud_noise));
    Node horizon_fade = nodes::variable(nodes::smoothstep(0.0f, 0.03f + 0.06f * cloud_elevation, dir.y()));
    Node day          = nodes::smoothstep(-0.08f, 0.3f, sun.y());
    Node sun_color    = nodes::variable(sun_e_fex * (0.22f * 0.04f));
    Node sky_ambient  = lin * 0.04f + nodes::vec3(0.0f, 0.0003f, 0.00075f);
    Node depth        = nodes::variable(nodes::max(0.0f, cloud_noise - threshold));
    Node beer         = nodes::variable(nodes::exp(depth * -4.0f));
    Node powder       = 1.0f - beer * beer;
    Node shade        = nodes::mix(nodes::num(0.45f), nodes::num(1.0f), nodes::clamp(beer * powder * 2.6f, 0.0f, 1.0f));
    Node silver       = nodes::clamp(0.51f / nodes::pow(1.49f - cos_theta * 1.4f, 1.5f), 0.0f, 3.0f);
    Node mask_faded   = nodes::variable(cloud_mask * horizon_fade);
    Node edge         = mask_faded * (1.0f - mask_faded) * 4.0f;
    Node cloud_color  = (sky_ambient + sun_color * shade + sun_color * (silver * edge * 0.6f)) * nodes::max(day, 0.03f);
    Node alpha        = nodes::variable((1.0f - nodes::exp(depth * cloud_density * -12.0f)) * horizon_fade);
    Node cloud_block = nodes::block(
        cloud_uv,
        cloud_noise,
        threshold,
        cloud_mask,
        horizon_fade,
        sun_color,
        depth,
        beer,
        mask_faded,
        alpha,
        tex.assign(tex - l0 * (0.04f * alpha)),
        tex.assign(nodes::mix(tex, nodes::mix(tex, cloud_color, fex), alpha))
    );

    Node body = nodes::block(
        dir,
        zenith,
        inv,
        fex,
        cos_theta,
        ratio,
        lin,
        l0,
        sun_e_fex,
        tex,
        nodes::if_then(nodes::logical_and(dir.y()
        .greater_than(nodes::num(0.0f)), cloud_coverage.greater_than(nodes::num(0.0f))), cloud_block));
    return nodes::seq(body, tex);
}

// Creates a Sky dome mesh with default daylight and cloud parameters.
fn Sky* sky(int resolution = 1024) {
    Sky* s = mem::new(Sky);
    s.init(MESH);
    s.instance_data.allocator = mem;
    s.geometry = drie::sphere_geometry(1, 64, 32);
    MeshBasicMaterial* m = drie::mesh_basic_material(side: BACK_SIDE, fog: false);
    m.depth_write = false;
    s.u_sun = nodes::uniform_vec4(0, 1, 0, 1000);
    s.u_beta_r = nodes::uniform_vec4(0, 0, 0, 0.8f);
    s.u_beta_m = nodes::uniform_vec4(0, 0, 0, 1);
    s.u_cloud0 = nodes::uniform_vec4(0.0002f, 0.00002f, 0.4f, 0.4f);
    s.u_cloud1 = nodes::uniform_vec4(0.5f, 0, 0, 0);
    m.color_node = sky_graph(s);
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

// Centers the dome on `eye`, just inside a camera's far plane, and updates its uniforms if a
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

// Updates the sky shader's uniforms if a parameter changed since last time.
fn void Sky.update(&self) {
    SkyParams p = sky_params(self);
    if (self.has_drawn && sky_params_equal(&p, &self.drawn)) return;
    self.drawn = p;
    self.has_drawn = true;
    SkyLight light = sky_light(&p);
    self.u_sun.set(light.sun.x, light.sun.y, light.sun.z, light.sun_e);
    self.u_beta_r.set(light.beta_r.x, light.beta_r.y, light.beta_r.z, p.mie_directional_g);
    self.u_beta_m.set(light.beta_m.x, light.beta_m.y, light.beta_m.z, p.show_sun_disc ? 1.0f : 0.0f);
    self.u_cloud0.set(p.cloud_scale, p.cloud_speed, p.cloud_coverage, p.cloud_density);
    self.u_cloud1.set(p.cloud_elevation, p.time, 0, 0);
}

// ---- Preetham scattering

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