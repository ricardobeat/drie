// `drie::shading` node-graph shading library mirroring shaders/shader_core.glsl and
// shaders/tonemapping.glsl: wind sway, UV slot transforms, hex tiling, equirectangular env
// sampling, octahedral normal packing, analytic fog, tone mapping, and depth packing.
//
// `uv_rel` stores 2 Vec4 rows per map slot (0 = normal, 1 = ARM, 2..7 = `tex_ext0..5`), with
// `r0.w > 0.5` selecting `uv1` and `r1.w` encoding software wrap modes (`0` = sampler default,
// `1 + s + 3*t` for repeat/clamp/mirror). Hex tiling skips patches with weight `<= HEX_SKIP`
// (`0.004`). Used by the material graph builders in src/material/.
module drie::shading;
import drie, drie::nodes;

// Wind (shaders/shader_core.glsl's apply_sway): `wp` bent sideways the higher `height` is.
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

// `1 / pi`, used for Lambertian diffuse normalization and equirectangular UV projection.
const float RECIPROCAL_PI = 0.31830988618f;

// The material's uv and how it changes from one pixel to the next (g_uv and its derivatives).
struct FragUv {
    Node uv;
    Node uv1;
    Node dx;
    Node dy;
    Node dx1;
    Node dy1;
    // hex tiling (Material.hex_tiling, hex_setup): the three patches nearest the pixel, as
    // offsets of uv where each reads the textures, and how much of each it takes (hex_w: none
    // without hex tiling)
    Node[3] hex_off;
    Node hex_w;
}

// Transformed UV for `uv_rel` slot `u` (0 = normal, 1 = ARM, 2..7 = `tex_ext0..5`), matching
// shaders/shader_core.glsl's `slot_uv`.
fn Node slot_uv(FragUv* f, int u, Node shift = { }) {
    Node r0 = drie::fs_in("uv_rel", k: u * 2);
    Node r1 = drie::fs_in("uv_rel", k: u * 2 + 1);
    Node own = shift.id ? f.uv + shift : f.uv;
    Node p = nodes::vec3(nodes::select(r0.w().greater_than(nodes::num(0.5f)), f.uv1, own), 1);
    return nodes::vec2(nodes::dot(r0.xyz(), p), nodes::dot(r1.xyz(), p));
}

// Screen-space UV partial derivative (`y == false`: dFdx, `y == true`: dFdy) for `uv_rel` slot `u`.
fn Node slot_d(FragUv* f, int u, bool y) {
    Node r0 = drie::fs_in("uv_rel", k: u * 2);
    Node r1 = drie::fs_in("uv_rel", k: u * 2 + 1);
    Node second = r0.w().greater_than(nodes::num(0.5f));
    Node d = y ? nodes::select(second, f.dy1, f.dy) : nodes::select(second, f.dx1, f.dx);
    return nodes::vec2(nodes::dot(r0.xy(), d), nodes::dot(r1.xy(), d));
}

// Slot `u` of uv_rel's texture (`view`) at the material's uv (slot_texel); hex tiled, the
// three patches' with the color map's weights (a map in the second uv set, a light map's, say,
// which is laid out once over the surface, reads the same place in all three):
// its own uv, wrapped as its texture wraps.
fn Node slot_texel(FragUv* f, int u, int view) {
    Node dx = slot_d(f, u, false);
    Node dy = slot_d(f, u, true);
    if (!f.hex_w.id) return slot_grad_at(u, view, slot_uv(f, u), dx, dy);
    Node[3] taps;
    for (int i = 0; i < 3; i++) taps[i] = slot_grad_at(u, view, slot_uv(f, u, f.hex_off[i]), dx, dy);
    return hex_sum(taps, f.hex_w);
}

// Hex tiling's patches below this weight are not read: most pixels are inside one patch (the
// weights are sharpened to the 7th power), where the other two would be read for nothing.
const float HEX_SKIP = 0.004f;

// taps[0] * w.x + taps[1] * w.y + taps[2] * w.z, each tap read only where its weight shows
// (an if in the program; the taps are vec4 samples).
fn Node hex_sum(Node[3] taps, Node w) @local {
    Node acc = nodes::variable(nodes::vec4(0.0f, 0.0f, 0.0f, 0.0f));
    Node[3] ws = { w.x(), w.y(), w.z() };
    Node[4] steps;
    steps[0] = acc;
    for (int i = 0; i < 3; i++) {
        steps[i + 1] = nodes::if_then(ws[i].greater_than(nodes::num(HEX_SKIP)), nodes::block(acc.assign(acc + taps[i] * ws[i])));
    }
    return nodes::seq(drie::node_block(steps[:4]), acc);
}

// Hex tiling (Material.hex_tiling; Mikkelsen, "Practical Real-Time Hex-Tiling", JCGT 11(2),
// 2022): the uv plane cut into hexagons `size` across, each reading the texture from its own
// random place, so that a texture repeated over a wall doesn't show its repeats. The hexagons'
// middles make a grid of triangles; a pixel takes the three corners of the one it is in,
// weighted by how near it is to each, sharpened (the 7th power) so that a patch is mostly
// itself and only blends into the next near their seam. Moved, not turned: what runs down the
// texture still runs down, and the normal map's directions stay as they are.
fn void FragUv.hex_setup(&self, Node size) {
    Node st = self.uv / size;
    // the grid skewed so that its triangles are halves of unit squares
    Node skewed = nodes::vec2(st.x() - st.y() * 0.57735027f, st.y() * 1.15470054f);
    Node cell = nodes::floor(skewed);
    Node fr = nodes::fract(skewed);
    Node fz = (fr.x() + fr.y()).one_minus();
    // the square's upper triangle (1) or lower (0), and the barycentric weights in it
    Node s = nodes::step(0.0f, fz.negate());
    Node s2 = s * 2.0f - 1.0f;
    Node w = nodes::vec3(fz.negate() * s2, s - fr.y() * s2, s - fr.x() * s2);
    self.hex_off[0] = hex_hash(cell + nodes::vec2(s, s));
    self.hex_off[1] = hex_hash(cell + nodes::vec2(s, s.one_minus()));
    self.hex_off[2] = hex_hash(cell + nodes::vec2(s.one_minus(), s));
    Node w3 = w * w * w;
    w = w3 * w3 * w;
    self.hex_w = w / (w.x() + w.y() + w.z());
}

// (Dave Hoskins' hash22: no sin, whose precision goes far from 0, as a uv in world meters does)
fn Node hex_hash(Node v) @local {
    Node p = nodes::fract(v.swizzle("xyx") * nodes::vec3(0.1031f, 0.1030f, 0.0973f));
    p = p + nodes::dot(p, p.swizzle("yzx") + 33.33f);
    return nodes::fract((p.swizzle("xx") + p.swizzle("yz")) * p.swizzle("zy"));
}

// The color map, hex tiled: where two patches meet, the brighter of them shows more (the
// paper's luminance weighting, 0.6 of it), so the seam follows the texture's own dark marks
// instead of fading one into the other; the other maps take the same weights.
fn Node FragUv.hex_color(&self) {
    // (a patch read only where it shows: see HEX_SKIP; one not read is black, and its weight
    // too small to matter)
    Node[3] c;
    Node[3] ws = { self.hex_w.x(), self.hex_w.y(), self.hex_w.z() };
    Node[7] steps;
    for (int i = 0; i < 3; i++) {
        c[i] = nodes::variable(nodes::vec4(0.0f, 0.0f, 0.0f, 0.0f));
        Node t = drie::node_sample(GRAD, shaders::VIEW_TEX_MAP, shaders::SMP_TEX_SMP, TEX_2D, self.uv + self.hex_off[i], self.dx, self.dy);
        steps[i * 2] = c[i];
        steps[i * 2 + 1] = nodes::if_then(ws[i].greater_than(nodes::num(HEX_SKIP)), nodes::block(c[i].assign(t)));
    }
    // (the reads once, then each variable after them)
    Node c0 = nodes::seq(drie::node_block(steps[:6]), c[0]);
    c[1] = nodes::seq(nodes::block(c0), c[1]);
    c[2] = nodes::seq(nodes::block(c0), c[2]);
    c[0] = c0;
    Node luma = nodes::vec3(0.2126f, 0.7152f, 0.0722f);
    Node bright = nodes::vec3(nodes::dot(c[0].xyz(), luma), nodes::dot(c[1].xyz(), luma), nodes::dot(c[2].xyz(), luma));
    Node w = self.hex_w * nodes::mix(nodes::vec3(1), bright, 0.6f);
    self.hex_w = w / nodes::max(w.x() + w.y() + w.z(), 1e-6f);
    return c[0] * self.hex_w.x() + c[1] * self.hex_w.y() + c[2] * self.hex_w.z();
}


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

// Software UV wrap (`mode`: `0` = repeat, `1` = clamp inset by `half`, `2` = mirrored repeat).
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

// Texel dimensions (`ivec2`) of `VIEW_TEX_ENV` at mip level `lod`.
fn Node env_size(int lod = 0) {
    return drie::node_sample(SIZE, shaders::VIEW_TEX_ENV, shaders::SMP_ENV_SMP, TEX_2D, lod ? nodes::int_(lod) : (Node) { });
}

// Samples `VIEW_TEX_ENV` at equirectangular `uv` and explicit mip level `lod`.
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

// Encodes unit normal `n` into a 2-component octahedral vector in `[-1, 1]^2`.
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

// ---- Tone mapping and color space

// Tone mapping disabled (matches shaders/tonemapping.glsl `TONE_NONE`).
const int TONE_NONE = 0;
// Linear exposure scale clamped to `[0, 1]`.
const int TONE_LINEAR = 1;
// Reinhard tone mapping `c / (c + 1)`.
const int TONE_REINHARD = 2;
// Filmic Cineon tone curve.
const int TONE_CINEON = 3;
// Stephen Hill ACES Filmic RRT+ODT fit.
const int TONE_ACES_FILMIC = 4;
// AgX wide-gamut tone mapper via Rec.2020 inset/outset.
const int TONE_AGX = 5;
// Khronos PBR Neutral tone mapper.
const int TONE_NEUTRAL = 6;
// Standard sRGB display output (`MaterialKey.display`).
const int DISPLAY_SRGB = 0;
// Linear HDR display output without sRGB transfer (`MaterialKey.display`).
const int DISPLAY_HDR = 2;

// Constructs a constant `MAT3` node from 9 column-major floats.
fn Node mat3_of(float[9] m) {
    return drie::node_matrix({ nodes::vec3(m[0], m[1], m[2]), nodes::vec3(m[3], m[4], m[5]), nodes::vec3(m[6], m[7], m[8]) }, MAT3);
}

// Converts linear RGB `c` to sRGB transfer encoding.
fn Node linear_to_srgb(Node c) {
    Node lo = c * 12.92f;
    Node hi = nodes::pow(c, nodes::vec3(1.0f / 2.4f)) * 1.055f - 0.055f;
    return nodes::mix(lo, hi, nodes::step(nodes::vec3(0.0031308f), c));
}

// Converts sRGB-encoded color `c` to linear RGB.
fn Node srgb_to_linear(Node c) {
    Node lo = c / 12.92f;
    Node hi = nodes::pow((c + 0.055f) / 1.055f, nodes::vec3(2.4f));
    return nodes::mix(lo, hi, nodes::step(nodes::vec3(0.04045f), c));
}

// Rational polynomial fit to the ACES RRT+ODT curve (Stephen Hill).
fn Node rrt_odt_fit(Node v) {
    Node a = v * (v + 0.0245786f) - 0.000090537f;
    Node b = v * (v * 0.983729f + 0.4329510f) + 0.238081f;
    return a / b;
}

// ACES Filmic tone curve applied in AP1 space, returning linear sRGB before clamping.
fn Node aces_filmic_wide(Node c) {
    Node aces_in = mat3_of({ 0.59719f, 0.07600f, 0.02840f, 0.35458f, 0.90834f, 0.13383f, 0.04823f, 0.01566f, 0.83777f });
    Node aces_out = mat3_of({ 1.60475f, -0.10208f, -0.00327f, -0.53108f, 1.10813f, -0.07276f, -0.07367f, -0.00605f, 1.07602f });
    Node x = aces_in * (c * (1.0f / 0.6f));
    return aces_out * rrt_odt_fit(x);
}

// 6th-order polynomial contrast sigmoid for AgX log-encoded values in `[0, 1]`.
fn Node agx_contrast(Node x) {
    Node x2 = x * x;
    Node x4 = x2 * x2;
    return x4 * x2 * 15.5f - x4 * x * 40.14f + x4 * 31.96f - x2 * x * 6.868f + x2 * 0.4298f + x * 0.1191f - 0.00232f;
}

// AgX tone mapper operating in Rec.2020 with inset/outset matrices, returning linear Rec.2020.
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

// Constant `MAT3` converting linear Rec.2020 primaries to linear sRGB.
fn Node rec2020_to_srgb() => mat3_of({ 1.6605f, -0.1246f, -0.0182f, -0.5876f, 1.1329f, -0.1006f, -0.0728f, -0.0083f, 1.1187f });

// Khronos PBR Neutral tone mapper preserving hue and base color fidelity.
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

// Optimized Cineon filmic tone curve (linear output).
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

// Applies tone mapping `mode` with `exposure`, converting to Display P3 primaries when `p3` holds.
fn Node tone_map_gamut(Node c, int mode, Node exposure, bool p3) {
    if (!p3) return tone_map(c, mode, exposure);
    Node srgb_to_p3 = mat3_of({ 0.8225f, 0.0332f, 0.0171f, 0.1774f, 0.9669f, 0.0724f, 0.0000f, 0.0000f, 0.9108f });
    if (mode == TONE_NONE) return srgb_to_p3 * c;
    Node e = nodes::max(c, nodes::vec3(0)) * exposure;
    if (mode == TONE_ACES_FILMIC) return nodes::clamp(srgb_to_p3 * aces_filmic_wide(e), 0, 1);
    if (mode == TONE_AGX) return nodes::clamp(srgb_to_p3 * (rec2020_to_srgb() * agx_rec2020(e)), 0, 1);
    return srgb_to_p3 * tone_map(c, mode, exposure);
}

// A world direction in the camera's view space (x right, y up, z towards the viewer).
fn Node to_view(Node d) {
    Node right = drie::fs_in("camera_right", VEC3);
    Node up = drie::fs_in("camera_up", VEC3);
    Node back = nodes::cross(right, up);
    return nodes::vec3(nodes::dot(d, right), nodes::dot(d, up), nodes::dot(d, back));
}

// Window depth `v` packed as three packs it (Material.depth_packing, `packing`: 0 basic, as
// 1 - v in gray with the material's opacity; 1 RGBA, 2 RGB and 3 RG, 8 bits a channel).
fn Node pack_depth(Node v, int packing, Node alpha) {
    if (packing == 0) return nodes::vec4(nodes::vec3(v.one_minus()), alpha);
    Node low = packing == 1 ? nodes::vec4(0) : nodes::vec4(0, 0, 0, 1);
    Node high = packing == 3 ? nodes::vec4(1, 1, 0, 1) : nodes::vec4(1);
    Node packed;
    switch (packing) {
        case 3:
            Node x = v * 256.0f;
            packed = nodes::vec4(nodes::floor(x) / 255.0f, nodes::fract(x), 0, 1);
        case 2:
            Node x = v * 65536.0f;
            Node y = nodes::floor(x) / 256.0f;
            packed = nodes::vec4(nodes::floor(y) / 255.0f, nodes::fract(y) * (256.0f / 255.0f), nodes::fract(x), 1);
        default:
            Node x = v * 16777216.0f;
            Node y = nodes::floor(x) / 256.0f;
            Node z = nodes::floor(y) / 256.0f;
            packed = nodes::vec4(nodes::floor(z) / 255.0f, nodes::fract(z) * (256.0f / 255.0f), nodes::fract(y) * (256.0f / 255.0f), nodes::fract(x));
    }
    return nodes::select(v.less_than_equal(nodes::num(0)), low, nodes::select(v.greater_than_equal(nodes::num(1)), high, packed));
}
