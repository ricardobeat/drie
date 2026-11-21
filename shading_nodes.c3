// The shading library of the built-in materials' node graphs (materials_nodes.c3): the
// functions of the engine's material shader (shader_core.glsl, tonemapping.glsl) as node
// builders, with the same math.
module drie::shading;
import drie, drie::nodes;

// Wind (shader_core.glsl's apply_sway): `wp` bent sideways the higher `height` is.
fn Node apply_sway(Node wp, Node height, Node sway) {
    Node h = nodes::max(height, 0);
    Node phase = wp.x() * 0.37f + wp.z() * 0.29f;
    Node t = sway.x() * sway.z();
    Node gust = nodes::sin(t * 0.31f + phase * 0.21f) * 0.4f + 0.6f;
    Node d = nodes::vec2(
        nodes::sin(t + phase) + nodes::sin(t * 2.7f + phase * 3.1f) * 0.4f,
        nodes::cos(t * 0.83f + phase * 1.3f) + nodes::sin(t * 2.1f + phase * 2.3f) * 0.4f
    );
    Node xz = wp.xz() + d * (gust * sway.y() * h * nodes::sqrt(h));
    return nodes::vec3(xz.x(), wp.y(), xz.y());
}

const float RECIPROCAL_PI = 0.31830988618f;

// The material's uv and how it changes from one pixel to the next (g_uv and its derivatives).
struct FragUv {
    Node uv;
    Node uv1;
    Node dx;
    Node dy;
    Node dx1;
    Node dy1;
}

// The uv of slot `u` of uv_rel (0: the normal map, 1: the arm map, 2..7: the shared slots) at
// the material's uv, and how it changes from one pixel to the next (shader_core.glsl's
// slot_uv, slot_dx, slot_dy).
fn Node slot_uv(FragUv* f, int u) {
    Node r0 = drie::fs_in("uv_rel", k: u * 2);
    Node r1 = drie::fs_in("uv_rel", k: u * 2 + 1);
    Node p = nodes::vec3(nodes::select(r0.w().greater_than(nodes::num(0.5f)), f.uv1, f.uv), 1);
    return nodes::vec2(nodes::dot(r0.xyz(), p), nodes::dot(r1.xyz(), p));
}

fn Node slot_d(FragUv* f, int u, bool y) {
    Node r0 = drie::fs_in("uv_rel", k: u * 2);
    Node r1 = drie::fs_in("uv_rel", k: u * 2 + 1);
    Node second = r0.w().greater_than(nodes::num(0.5f));
    Node d = y ? nodes::select(second, f.dy1, f.dy) : nodes::select(second, f.dx1, f.dx);
    return nodes::vec2(nodes::dot(r0.xy(), d), nodes::dot(r1.xy(), d));
}

// Slot `u` of uv_rel's texture (`view`) at the material's uv (slot_texel, without hex tiling):
// its own uv, wrapped as its texture wraps.
fn Node slot_texel(FragUv* f, int u, int view) => slot_grad_at(u, view, slot_uv(f, u), slot_d(f, u, false), slot_d(f, u, true));

// Slot `u` of uv_rel's texture (`view`) at its uv `suv` with derivatives dx and dy, wrapped as
// its texture wraps (slot_grad_at).
fn Node slot_grad_at(int u, int view, Node suv, Node dx, Node dy) {
    Node r1 = drie::fs_in("uv_rel", k: u * 2 + 1);
    Node size = nodes::convert(drie::node_sample(SIZE, view, shaders::SMP_TEX_SMP, TEX_2D, { }), VEC2);
    // slot_wrap: 1 + s + 3 t for the wrapping across (s) and down (t), 0: as the sampler
    Node code = nodes::convert(r1.w() + 0.5f, INT);
    Node c = code - nodes::int_(1);
    Node ex = dx * size;
    Node ey = dy * size;
    Node lod = nodes::clamp(nodes::log2(nodes::max(nodes::max(nodes::dot(ex, ex), nodes::dot(ey, ey)), 1e-8f)) * 0.5f, 0, 16);
    Node half = nodes::min(nodes::exp2(lod) * 0.5f / size, nodes::vec2(0.5f));
    Node wrapped = nodes::vec2(
        wrap_coord(suv.x(), nodes::rem(c, nodes::int_(3)), half.x()),
        wrap_coord(suv.y(), c / nodes::int_(3), half.y())
    );
    Node at = nodes::select(code.equal(nodes::int_(0)), suv, wrapped);
    return drie::node_sample(GRAD, view, shaders::SMP_TEX_SMP, TEX_2D, at, dx, dy);
}

// One of the six shared map slots (tex_ext0..5) at the material's uv.
fn Node ext_texel(FragUv* f, int slot) => slot_texel(f, slot + 2, shaders::VIEW_TEX_EXT0 + slot);

fn Node wrap_coord(Node x, Node mode, Node half) {
    Node mirrored = nodes::abs(nodes::mod(x, 2) - 1).one_minus();
    return nodes::select(mode.equal(nodes::int_(1)), nodes::clamp(x, half, half.one_minus()),
                         nodes::select(mode.equal(nodes::int_(2)), mirrored, nodes::fract(x)));
}

// Equirectangular uv of direction d (env_uv).
fn Node env_uv(Node d) {
    return nodes::vec2(
        nodes::atan2(d.z(), d.x()) * (0.5f * RECIPROCAL_PI) + 0.5f,
        nodes::acos(nodes::clamp(d.y(), -1.0f, 1.0f)) * RECIPROCAL_PI
    );
}

fn Node env_size(int lod = 0) {
    return drie::node_sample(SIZE, shaders::VIEW_TEX_ENV, shaders::SMP_ENV_SMP, TEX_2D, lod ? nodes::int_(lod) : (Node) { });
}

fn Node env_level(Node uv, Node lod) => drie::node_sample(LEVEL, shaders::VIEW_TEX_ENV, shaders::SMP_ENV_SMP, TEX_2D, uv, lod);

// The material's map at its sharpest (env_texel_smooth): four bilinear reads at the corners
// of a grid twice as fine, mixed bilinearly.
fn Node env_texel_smooth(Node uv) {
    Node size = nodes::convert(env_size(), VEC2) * 2.0f;
    Node p = uv * size - 0.5f;
    Node i = nodes::floor(p);
    Node f = p - i;
    Node a = (i + 0.5f) / size;
    Node b = (i + 1.5f) / size;
    Node c00 = env_level(nodes::vec2(nodes::fract(a.x()), a.y()), nodes::num(0));
    Node c10 = env_level(nodes::vec2(nodes::fract(b.x()), a.y()), nodes::num(0));
    Node c01 = env_level(nodes::vec2(nodes::fract(a.x()), b.y()), nodes::num(0));
    Node c11 = env_level(nodes::vec2(nodes::fract(b.x()), b.y()), nodes::num(0));
    return nodes::mix(nodes::mix(c00, c10, f.x()), nodes::mix(c01, c11, f.x()), f.y());
}

// The mip level of an equirectangular map `width` texels around whose texels are about `sigma`
// radians wide (env_lod).
fn Node env_lod(Node sigma, Node width) => nodes::max(nodes::log2(nodes::max(sigma * width / 3.14159265359f, 1e-6f)), 0);

// The material's map in direction d, sharp but at the mip level its footprint wants
// (env_classic; the u derivative wraps at the seam behind).
fn Node env_classic(Node d) {
    Node uv = env_uv(d);
    Node dx = nodes::dfdx(uv);
    Node dy = nodes::dfdy(uv);
    dx = nodes::vec2(dx.x() - nodes::floor(dx.x() + 0.5f), dx.y());
    dy = nodes::vec2(dy.x() - nodes::floor(dy.x() + 0.5f), dy.y());
    return drie::node_sample(GRAD, shaders::VIEW_TEX_ENV, shaders::SMP_ENV_SMP, TEX_2D, uv, dx, dy).xyz();
}

fn Node pack_normal_octahedral(Node n) {
    Node m = n / (nodes::abs(n.x()) + nodes::abs(n.y()) + nodes::abs(n.z()));
    Node signs = nodes::vec2(
        nodes::select(m.x().greater_than_equal(nodes::num(0)), 1.0f, -1.0f),
        nodes::select(m.y().greater_than_equal(nodes::num(0)), 1.0f, -1.0f)
    );
    Node folded = nodes::abs(m.swizzle("yx")).one_minus() * signs;
    return nodes::select(m.z().less_than(nodes::num(0)), folded, m.xy());
}

// Length of the part of the segment a-b that lies in the fog (fog_distance).
fn Node fog_distance(MaterialKey key, Node a, Node b) {
    Node d = b - a;
    Node len = nodes::length(d);
    if (!key.fog_box) return len;
    Node lo = drie::fs_in("fog_box_min");
    Node hi = drie::fs_in("fog_box_max");
    Node nudge = nodes::vec3(
        nodes::select(d.x().equal(nodes::num(0)), 1e-6f, 0.0f),
        nodes::select(d.y().equal(nodes::num(0)), 1e-6f, 0.0f),
        nodes::select(d.z().equal(nodes::num(0)), 1e-6f, 0.0f)
    );
    Node inv = 1.0f / (d + nudge);
    Node t0 = (lo.xyz() - a) * inv;
    Node t1 = (hi.xyz() - a) * inv;
    Node tmin = nodes::min(t0, t1);
    Node tmax = nodes::max(t0, t1);
    Node enter = nodes::max(nodes::max(tmin.x(), tmin.y()), nodes::max(tmin.z(), 0));
    Node exit = nodes::min(nodes::min(tmax.x(), tmax.y()), nodes::min(tmax.z(), 1));
    Node outside = nodes::max(exit - enter, 0) * hi.w().one_minus();
    Node box_on = lo.w().greater_than_equal(nodes::num(0.5f));
    return nodes::select(box_on, len * outside.one_minus(), len);
}

// Fraction of light that makes it through `d` meters of fog (fog_transmittance).
fn Node fog_transmittance(MaterialKey key, Node d) {
    Node p = drie::fs_in("fog_params");
    switch (key.fog) {
        case 1: return nodes::smoothstep(p.x(), p.y(), d).one_minus();
        case 2:
            Node dd = p.z() * d;
            return nodes::exp((dd * dd).negate());
        default: return nodes::exp((p.z() * d).negate());
    }
}

// ---- tone mapping (tonemapping.glsl)

const int TONE_NONE = 0;
const int TONE_LINEAR = 1;
const int TONE_REINHARD = 2;
const int TONE_CINEON = 3;
const int TONE_ACES_FILMIC = 4;
const int TONE_AGX = 5;
const int TONE_NEUTRAL = 6;
const int DISPLAY_SRGB = 0;
const int DISPLAY_HDR = 2;

fn Node mat3_of(float[9] m) {
    return drie::node_matrix({ nodes::vec3(m[0], m[1], m[2]), nodes::vec3(m[3], m[4], m[5]), nodes::vec3(m[6], m[7], m[8]) }, MAT3);
}

fn Node linear_to_srgb(Node c) {
    Node lo = c * 12.92f;
    Node hi = nodes::pow(c, nodes::vec3(1.0f / 2.4f)) * 1.055f - 0.055f;
    return nodes::mix(lo, hi, nodes::step(nodes::vec3(0.0031308f), c));
}

fn Node rrt_odt_fit(Node v) {
    Node a = v * (v + 0.0245786f) - 0.000090537f;
    Node b = v * (v * 0.983729f + 0.4329510f) + 0.238081f;
    return a / b;
}

fn Node aces_filmic_wide(Node c) {
    Node aces_in = mat3_of({ 0.59719f, 0.07600f, 0.02840f, 0.35458f, 0.90834f, 0.13383f, 0.04823f, 0.01566f, 0.83777f });
    Node aces_out = mat3_of({ 1.60475f, -0.10208f, -0.00327f, -0.53108f, 1.10813f, -0.07276f, -0.07367f, -0.00605f, 1.07602f });
    Node x = aces_in * (c * (1.0f / 0.6f));
    return aces_out * rrt_odt_fit(x);
}

fn Node agx_contrast(Node x) {
    Node x2 = x * x;
    Node x4 = x2 * x2;
    return x4 * x2 * 15.5f - x4 * x * 40.14f + x4 * 31.96f - x2 * x * 6.868f + x2 * 0.4298f + x * 0.1191f - 0.00232f;
}

fn Node agx_rec2020(Node c) {
    Node to_2020 = mat3_of({ 0.6274f, 0.0691f, 0.0164f, 0.3293f, 0.9195f, 0.0880f, 0.0433f, 0.0113f, 0.8956f });
    Node inset = mat3_of({ 0.856627153315983f, 0.137318972929847f, 0.11189821299995f, 0.0951212405381588f, 0.761241990602591f,
                           0.0767994186031903f, 0.0482516061458583f, 0.101439036467562f, 0.811302368396859f });
    Node outset = mat3_of({ 1.1271005818144368f, -0.1413297634984383f, -0.14132976349843826f, -0.11060664309660323f,
                            1.157823702216272f, -0.11060664309660294f, -0.016493938717834573f, -0.016493938717834257f,
                            1.2519364065950405f });
    float min_ev = -12.47393f;
    float max_ev = 4.026069f;
    Node x = inset * (to_2020 * c);
    x = nodes::clamp((nodes::log2(nodes::max(x, nodes::vec3(1e-10f))) - min_ev) / (max_ev - min_ev), 0, 1);
    x = outset * agx_contrast(x);
    return nodes::pow(nodes::max(x, nodes::vec3(0)), nodes::vec3(2.2f));
}

fn Node rec2020_to_srgb() => mat3_of({ 1.6605f, -0.1246f, -0.0182f, -0.5876f, 1.1329f, -0.1006f, -0.0728f, -0.0083f, 1.1187f });

fn Node neutral(Node c) {
    float start = 0.8f - 0.04f;
    float desaturation = 0.15f;
    Node x = nodes::min(c.x(), nodes::min(c.y(), c.z()));
    Node offset = nodes::select(x.less_than(nodes::num(0.08f)), x - x * x * 6.25f, 0.04f);
    Node d0 = c - offset;
    Node peak = nodes::max(d0.x(), nodes::max(d0.y(), d0.z()));
    float d = 1.0f - start;
    Node new_peak = (d * d / (peak + d - start)).one_minus();
    Node scaled = d0 * (new_peak / peak);
    Node g = (1.0f / ((peak - new_peak) * desaturation + 1.0f)).one_minus();
    Node rolled = nodes::mix(scaled, nodes::vec3(new_peak), g);
    return nodes::select(peak.less_than(nodes::num(start)), d0, rolled);
}

fn Node cineon(Node c) {
    Node x = nodes::max(nodes::vec3(0), c - 0.004f);
    return nodes::pow((x * (x * 6.2f + 0.5f)) / (x * (x * 6.2f + 1.7f) + 0.06f), nodes::vec3(2.2f));
}

// Linear light (times `exposure`) into 0..1, still linear (tone_map).
fn Node tone_map(Node c, int mode, Node exposure) {
    if (mode == TONE_NONE) return c;
    Node e = nodes::max(c, nodes::vec3(0)) * exposure;
    switch (mode) {
        case TONE_LINEAR: return nodes::clamp(e, 0, 1);
        case TONE_REINHARD: return nodes::clamp(e / (e + 1.0f), 0, 1);
        case TONE_CINEON: return cineon(e);
        case TONE_ACES_FILMIC: return nodes::clamp(aces_filmic_wide(e), 0, 1);
        case TONE_AGX: return nodes::clamp(rec2020_to_srgb() * agx_rec2020(e), 0, 1);
        case TONE_NEUTRAL: return neutral(e);
        default: return e;
    }
}

fn Node tone_map_gamut(Node c, int mode, Node exposure, bool p3) {
    if (!p3) return tone_map(c, mode, exposure);
    Node srgb_to_p3 = mat3_of({ 0.8225f, 0.0332f, 0.0171f, 0.1774f, 0.9669f, 0.0724f, 0.0000f, 0.0000f, 0.9108f });
    if (mode == TONE_NONE) return srgb_to_p3 * c;
    Node e = nodes::max(c, nodes::vec3(0)) * exposure;
    if (mode == TONE_ACES_FILMIC) return nodes::clamp(srgb_to_p3 * aces_filmic_wide(e), 0, 1);
    if (mode == TONE_AGX) return nodes::clamp(srgb_to_p3 * (rec2020_to_srgb() * agx_rec2020(e)), 0, 1);
    return srgb_to_p3 * tone_map(c, mode, exposure);
}
