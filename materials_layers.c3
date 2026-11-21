// The physical material's layers as nodes (shader_core.glsl's sheen, iridescence, anisotropy and
// clear coat math), for the lit materials of materials_nodes.c3.
module drie;
import drie::nodes, drie::shading, std::math;

fn Node max3(Node c) @private => nodes::max(c.x(), nodes::max(c.y(), c.z()));

// Schlick's Fresnel: f0 head on, f90 at a glancing angle, `c` the cosine.
fn Node schlick(Node f0, Node f90, Node c) @private => f0 + (f90 - f0) * nodes::pow(c.one_minus(), 5);

// Sheen (three's BRDF_Sheen): the Charlie distribution of fibers, and Neubelt's visibility.
fn Node d_charlie(Node rough, Node noh) @private {
    Node inv = 1.0f / (rough * rough);
    Node sin2h = nodes::max((noh * noh).one_minus(), 0.0078125f);
    return (inv + 2.0f) * nodes::pow(sin2h, inv * 0.5f) / (2.0f * (float)math::PI);
}

fn Node v_neubelt(Node nov, Node nol) @private {
    return nodes::clamp(1.0f / ((nol + nov - nol * nov) * 4.0f), 0, 1);
}

// How much a sheen reflects of the light from all around (three's IBLSheenBRDF fit).
fn Node sheen_albedo(Node n, Node v, Node rough) @private {
    Node nov = nodes::clamp(nodes::dot(n, v), 0, 1);
    Node r2 = rough * rough;
    Node inv = 1.0f / (rough + 0.1f);
    Node a = rough * 1.0678f - 1.9362f + r2 * 0.4573f - inv * 0.8469f;
    Node b = rough * 0.5538f - 0.6014f - r2 * 0.4670f - inv * 0.1255f;
    return nodes::clamp(nodes::exp(a * nov + b), 0, 1);
}

// Anisotropic GGX (Filament's, as three's): alpha t along the direction t, alpha b across it
// (b); the cosines are with t and b.
fn Node v_smith_aniso(Node at, Node ab, Node tov, Node bov, Node tol, Node bol, Node nov, Node nol) @private {
    Node gv = nol * nodes::length(nodes::vec3(at * tov, ab * bov, nov));
    Node gl = nov * nodes::length(nodes::vec3(at * tol, ab * bol, nol));
    return 0.5f / nodes::max(gv + gl, 1e-6f);
}

fn Node d_ggx_aniso(Node at, Node ab, Node noh, Node toh, Node boh) @private {
    Node a2 = at * ab;
    Node d = nodes::vec3(ab * toh, at * boh, a2 * noh);
    Node w2 = a2 / nodes::dot(d, d);
    return shading::RECIPROCAL_PI * a2 * w2 * w2;
}

// three's EnvironmentBRDF: how much of the light a surface reflects, all directions taken
// together (Karis' fit of the split-sum integral).
fn Node environment_brdf(Node n, Node v, Node f0, Node f90, Node rough) @private {
    Node nov = nodes::clamp(nodes::dot(n, v), 0, 1);
    Node r = nodes::vec4(rough) * nodes::vec4(-1.0f, -0.0275f, -0.572f, 0.022f) + nodes::vec4(1.0f, 0.0425f, 1.04f, -0.04f);
    Node a004 = nodes::min(r.x() * r.x(), nodes::exp2(nov * -9.28f)) * r.x() + r.y();
    Node fab = nodes::vec2(-1.04f, 1.04f) * a004 + r.swizzle("zw");
    return f0 * fab.x() + f90 * fab.y();
}

// The reflectance head on that Schlick's approximation would take to reach f at this angle.
fn Node schlick_to_f0(Node f, Node f90, Node c) @private {
    Node x = nodes::clamp(c.one_minus(), 0, 1);
    Node x5 = nodes::clamp(x * x * x * x * x, 0, 0.9999f);
    return (f - nodes::vec3(f90) * x5) / x5.one_minus();
}

// Iridescence (KHR_materials_iridescence, three's evalIridescence: Belcour and Barla's thin
// film): light reflected off the film's top and its bottom interferes, by wavelength. The
// function eval_iridescence(outside_ior, eta2, cos1, thickness, base_f0) of the program (made
// once: `fn_id` -1 makes it).
fn Node eval_iridescence(int* fn_id, Node outside_ior, Node eta2, Node cos1, Node thickness, Node base_f0) @private {
    if (*fn_id < 0) {
        NodeType[5] types = { FLOAT, FLOAT, FLOAT, FLOAT, VEC3 };
        int f = node_func("eval_iridescence", VEC3, &types);
        Node outside = node_param(f, 0);
        Node film_eta = node_param(f, 1);
        Node c1 = node_param(f, 2);
        Node thick = node_param(f, 3);
        Node base = node_param(f, 4);
        Node pi = nodes::num((float)math::PI);
        // (the film fades into the air as it thins out)
        Node film_ior = nodes::mix(outside, film_eta, nodes::smoothstep(0, 0.03f, thick));
        Node ratio = outside / film_ior;
        Node sin2_sq = ratio * ratio * (c1 * c1).one_minus();
        Node cos2_sq = sin2_sq.one_minus();
        Node cos2 = nodes::sqrt(cos2_sq);
        // the first interface
        Node rr = (film_ior - outside) / (film_ior + outside);
        Node r12 = schlick(rr * rr, nodes::num(1), c1);
        Node t121 = r12.one_minus();
        Node phi12 = nodes::select(film_ior.less_than(outside), pi, 0.0f);
        Node phi21 = pi - phi12;
        // the second
        Node s = nodes::sqrt(nodes::clamp(base, 0, 0.9999f));
        Node base_ior = (nodes::vec3(1) + s) / (nodes::vec3(1) - s);
        Node rb = (base_ior - nodes::vec3(film_ior)) / (base_ior + nodes::vec3(film_ior));
        Node r23 = schlick(rb * rb, nodes::vec3(1), cos2);
        Node phi23 = nodes::vec3(nodes::select(base_ior.x().less_than(film_ior), pi, 0.0f),
                                 nodes::select(base_ior.y().less_than(film_ior), pi, 0.0f),
                                 nodes::select(base_ior.z().less_than(film_ior), pi, 0.0f));
        // the phase shift
        Node opd = film_ior * 2.0f * thick * cos2;
        Node phi = nodes::vec3(phi21) + phi23;
        Node r123 = nodes::clamp(r23 * r12, 1e-5f, 0.9999f);
        Node sr123 = nodes::sqrt(r123);
        Node rs = r23 * (t121 * t121) / (nodes::vec3(1) - r123);
        Node result = rs + r12;
        Node cm = rs - t121;
        for (int m = 1; m <= 2; m++) {
            cm = cm * sr123;
            result = result + cm * 2.0f * eval_sensitivity(opd * (float)m, phi * (float)m);
        }
        // total internal reflection; and out of gamut colors come out negative
        node_func_body(f, nodes::select(cos2_sq.less_than(nodes::num(0)), nodes::vec3(1), nodes::max(result, nodes::vec3(0))));
        *fn_id = f;
    }
    Node[5] args = { outside_ior, eta2, cos1, thickness, base_f0 };
    return node_call_fn(*fn_id, &args);
}

// The XYZ sensitivity curves in Fourier space, at an optical path difference (nm), in Rec.709.
fn Node eval_sensitivity(Node opd, Node shift) @local {
    Node phase = opd * (2.0f * (float)math::PI * 1.0e-9f);
    Node val = nodes::vec3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
    Node pos = nodes::vec3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
    Node spread = nodes::vec3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);
    Node xyz = val * nodes::sqrt(spread * (2.0f * (float)math::PI)) * nodes::cos(pos * phase + shift) * nodes::exp(spread * (phase * phase).negate());
    Node x = xyz.x() + 9.7470e-14f * math::sqrt(2.0f * (float)math::PI * 4.5282e+09f) * nodes::cos(phase * 2.2399e+06f + shift.x())
        * nodes::exp((phase * phase) * -4.5282e+09f);
    xyz = nodes::vec3(x, xyz.y(), xyz.z()) / 1.0685e-7f;
    Node[3] cols = {
        nodes::vec3(3.2404542f, -0.9692660f, 0.0556434f),
        nodes::vec3(-1.5371385f, 1.8760108f, -0.2040259f),
        nodes::vec3(-0.4985314f, 0.0415560f, 1.0572252f),
    };
    return node_matrix(&cols, MAT3) * xyz;
}

// ---- transmission (three's getIBLVolumeRefraction)

// The picture behind the transmission materials (Renderer's transmission_mips), read at uv on
// level lod with three's bicubic filter (transmission_pars_fragment): four bilinear taps.
fn Node transmission_bicubic(int* fn_id, Node uv, Node lod) @private {
    if (*fn_id < 0) {
        NodeType[2] types = { VEC2, FLOAT };
        int f = node_func("transmission_bicubic", VEC4, &types);
        Node p = node_param(f, 0);
        Node l = node_param(f, 1);
        Node size = nodes::convert(transmission_texture(SIZE, nodes::convert(l, INT)), VEC2);
        Node texel = nodes::vec2(1) / size;
        Node q = p * size + 0.5f;
        Node iuv = nodes::floor(q);
        Node fr = nodes::fract(q);
        Node fx = fr.x();
        Node fy = fr.y();
        Node g0x = bicubic_w0(fx) + bicubic_w1(fx);
        Node g1x = bicubic_w2(fx) + bicubic_w3(fx);
        Node h0x = bicubic_w1(fx) / g0x - 1.0f;
        Node h1x = bicubic_w3(fx) / g1x + 1.0f;
        Node g0y = bicubic_w0(fy) + bicubic_w1(fy);
        Node g1y = bicubic_w2(fy) + bicubic_w3(fy);
        Node h0y = bicubic_w1(fy) / g0y - 1.0f;
        Node h1y = bicubic_w3(fy) / g1y + 1.0f;
        Node t0 = transmission_texture(LEVEL, (nodes::vec2(iuv.x() + h0x, iuv.y() + h0y) - 0.5f) * texel, l);
        Node t1 = transmission_texture(LEVEL, (nodes::vec2(iuv.x() + h1x, iuv.y() + h0y) - 0.5f) * texel, l);
        Node t2 = transmission_texture(LEVEL, (nodes::vec2(iuv.x() + h0x, iuv.y() + h1y) - 0.5f) * texel, l);
        Node t3 = transmission_texture(LEVEL, (nodes::vec2(iuv.x() + h1x, iuv.y() + h1y) - 0.5f) * texel, l);
        node_func_body(f, (t0 * g0x + t1 * g1x) * g0y + (t2 * g0x + t3 * g1x) * g1y);
        *fn_id = f;
    }
    Node[2] args = { uv, lod };
    return node_call_fn(*fn_id, &args);
}

fn Node transmission_texture(NodeSampling how, Node coords = { }, Node a = { }) @local
    => node_sample(how, shaders::VIEW_TEX_TRANSMISSION, shaders::SMP_TRANSMISSION_SMP, TEX_2D, coords, a);

fn Node bicubic_w0(Node a) @local => (a * (a * (3.0f - a) - 3.0f) + 1.0f) * (1.0f / 6.0f);
fn Node bicubic_w1(Node a) @local => (a * a * (a * 3.0f - 6.0f) + 4.0f) * (1.0f / 6.0f);
fn Node bicubic_w2(Node a) @local => (a * (a * (a * -3.0f + 3.0f) + 3.0f) + 1.0f) * (1.0f / 6.0f);
fn Node bicubic_w3(Node a) @local => a * a * a * (1.0f / 6.0f);

// What is behind the surface where a ray leaves it at `exit`: the transmission picture there,
// blurrier the rougher the surface (and the stronger it bends light: none at an IOR of 1).
fn Node transmission_sample(int* fn_id, int* bicubic_fn, Node exit, Node rough, Node ior) @private {
    if (*fn_id < 0) {
        NodeType[3] types = { VEC3, FLOAT, FLOAT };
        int f = node_func("transmission_sample", VEC3, &types);
        Node e = node_param(f, 0);
        Node r = node_param(f, 1);
        Node i = node_param(f, 2);
        Node clip = fs_in("cluster_view_proj", MAT4) * nodes::vec4(e, 1);
        Node uv = clip.xy() / nodes::max(nodes::abs(clip.w()), 1e-5f) * 0.5f + 0.5f;
        // (the picture's row 0 is its top where the target's is, as for the reflection source)
        uv = nodes::select(fs_in("ssr_params", FLOAT, 3).greater_than(nodes::num(0.5f)), nodes::vec2(uv.x(), uv.y().one_minus()), uv);
        Node top = nodes::convert(transmission_texture(LEVELS), FLOAT) - 1.0f;
        Node lod = nodes::clamp(nodes::log2(nodes::max(fs_in("ext_params", FLOAT, 3), 1.0f)) * r * nodes::clamp(i * 2.0f - 2.0f, 0, 1),
                                nodes::num(0), top);
        Node light = nodes::variable(transmission_bicubic(bicubic_fn, uv, nodes::floor(lod)).xyz());
        Node coarse = transmission_bicubic(bicubic_fn, uv, nodes::min(nodes::ceil(lod), top)).xyz();
        Node between = nodes::if_then(nodes::fract(lod).greater_than(nodes::num(0)),
                                      nodes::block(light.assign(nodes::mix(light, coarse, nodes::fract(lod)))));
        node_func_body(f, nodes::seq(nodes::block(light, between), light));
        *fn_id = f;
    }
    Node[3] args = { exit, rough, ior };
    return node_call_fn(*fn_id, &args);
}

// The way through the volume from the surface: refracted at `ior`, `thickness` long in the
// model's meters (ext_params.z: the model's scale), longer at a slant through a wall
// (RefractionModel.PLANE, ext_params.x's bit 64); none on total internal reflection.
fn Node transmission_ray(int* fn_id, Node n, Node v, Node thickness, Node ior) @private {
    if (*fn_id < 0) {
        NodeType[4] types = { VEC3, VEC3, FLOAT, FLOAT };
        int f = node_func("transmission_ray", VEC3, &types);
        Node pn = node_param(f, 0);
        Node pv = node_param(f, 1);
        Node thick = node_param(f, 2);
        Node i = node_param(f, 3);
        Node ray = nodes::refract(pv.negate(), pn, 1.0f / i);
        Node dir = nodes::normalize(ray);
        Node across = thick * fs_in("ext_params", FLOAT, 2);
        Node bits = nodes::convert(fs_in("ext_params", FLOAT) + 0.5f, INT);
        Node plane = nodes::bit_and(bits, nodes::int_(64)).not_equal(nodes::int_(0));
        across = nodes::select(plane, across / nodes::max(nodes::dot(dir, pn.negate()), 1.0f / 12.0f), across);
        node_func_body(f, nodes::select(nodes::dot(ray, ray).less_than(nodes::num(1e-8f)), nodes::vec3(0), dir * across));
        *fn_id = f;
    }
    Node[4] args = { n, v, thickness, ior };
    return node_call_fn(*fn_id, &args);
}

// Beer's law over `distance` meters (transmission_params.w: the attenuation distance; 0: none).
fn Node volume_attenuation(Node distance) @private {
    Node tp = fs_in("transmission_params");
    Node coefficient = nodes::log(nodes::max(fs_in("transmission_color", VEC3), nodes::vec3(1e-4f))).negate() / tp.w();
    return nodes::select(tp.w().less_than_equal(nodes::num(0)), nodes::vec3(1), nodes::exp(coefficient.negate() * distance));
}

// The functions of the program transmission takes (made once each: -1 makes them).
struct Refraction @private {
    int ray_fn;
    int sample_fn;
    int bicubic_fn;
}

// The light that comes through a transmission material from behind: through `n` seen from `v`,
// tinted by `diffuse` and by the volume over its way through, less what the surface reflects
// (f0, f90). With dispersion (transmission_params.z) each channel takes its own IOR.
fn Node Refraction.light(&self, Node n, Node v, Node world, Node rough, Node diffuse, Node thickness, Node f0, Node f90) {
    Node tp = fs_in("transmission_params");
    Node ior = nodes::select(tp.x().greater_than(nodes::num(0)), tp.x(), 1.5f);
    Node spread = (ior - 1.0f) * 0.025f * tp.z();
    Node light = nodes::variable(nodes::vec3(0));
    Node through = nodes::variable(nodes::vec3(0));
    Node[3] s;
    Node[3] a;
    for (int i = 0; i < 3; i++) {
        Node ior_i = i == 1 ? ior : ior + spread * (float)(i - 1);
        Node ray = transmission_ray(&self.ray_fn, n, v, thickness, ior_i);
        s[i] = transmission_sample(&self.sample_fn, &self.bicubic_fn, world + ray, rough, ior_i);
        a[i] = volume_attenuation(nodes::length(ray));
    }
    Node ray = transmission_ray(&self.ray_fn, n, v, thickness, ior);
    Node spread_out = nodes::block(light.assign(nodes::vec3(s[0].x(), s[1].y(), s[2].z())),
                                   through.assign(diffuse * nodes::vec3(a[0].x(), a[1].y(), a[2].z())));
    Node one = nodes::block(light.assign(transmission_sample(&self.sample_fn, &self.bicubic_fn, world + ray, rough, ior)),
                            through.assign(diffuse * volume_attenuation(nodes::length(ray))));
    Node pick = nodes::if_then(spread.greater_than(nodes::num(0)), spread_out, one);
    return nodes::seq(nodes::block(light, through, pick), (nodes::vec3(1) - environment_brdf(n, v, f0, f90, rough)) * through * light);
}

// ---- the scene's baked and dynamic indirect light

// Light from the scene's irradiance volumes for a surface at `world` facing `n`, already times
// how much of the point they cover (1 inside a volume, 0 outside them all), and that cover
// (shader_core.glsl's gi_irradiance). Each volume covers its box and fades out over half a
// cell outside it, in front of the ones before it. Each probe holds the light arriving from
// the six axis directions, side by side along x in the volume's part of tex_gi.
fn Node[2] gi_irradiance(Node n, Node world) @private {
    Node cover = nodes::variable(nodes::num(0));
    Node e = nodes::variable(nodes::vec3(0));
    Node slab = nodes::vec3(nodes::select(n.x().greater_than_equal(nodes::num(0)), 0.0f, 1.0f),
                            nodes::select(n.y().greater_than_equal(nodes::num(0)), 2.0f, 3.0f),
                            nodes::select(n.z().greater_than_equal(nodes::num(0)), 4.0f, 5.0f)) / 6.0f;
    Node w = n * n;
    Node i = nodes::loop_index(INT);
    Node lo = fs_at("gi_min", VEC4, i);
    Node dims = fs_at("gi_dims", VEC4, i);
    Node inv = fs_at("gi_inv_size", VEC4, i);
    Node atlas = fs_at("gi_atlas", VEC4, i);
    Node res = dims.xyz();
    Node p = (world + n * dims.w() - lo.xyz()) * inv.xyz();
    Node edge = nodes::min(p, nodes::vec3(1) - p) * res;
    Node fade = nodes::clamp(nodes::min(edge.x(), nodes::min(edge.y(), edge.z())) * 2.0f + 1.0f, 0, 1);
    // stay inside the volume and a direction's slab
    Node margin = nodes::vec3(0.5f) / res;
    p = nodes::clamp(p, margin, nodes::vec3(1) - margin);
    Node u = (slab + p.x() / 6.0f) * atlas.y() + atlas.x();
    Node yz = p.swizzle("yz") * atlas.swizzle("zw");
    Node l = gi_texel(nodes::vec3(u.x(), yz)) * w.x() + gi_texel(nodes::vec3(u.y(), yz)) * w.y() + gi_texel(nodes::vec3(u.z(), yz)) * w.z();
    Node body = nodes::block(nodes::if_then(lo.w().less_than(nodes::num(0.5f)), nodes::block(nodes::break_())),
                             nodes::if_then(fade.less_than_equal(nodes::num(0)), nodes::block(nodes::continue_())),
                             e.assign(nodes::mix(e, l * inv.w(), fade)),
                             cover.assign(nodes::mix(cover, nodes::num(1), fade)));
    Node volumes = nodes::block(cover, e, nodes::loop(i, nodes::int_(0), nodes::int_(4), body));
    return { nodes::seq(volumes, e), cover };
}

fn Node gi_texel(Node uvw) @local => node_sample(LEVEL, shaders::VIEW_TEX_GI, shaders::SMP_GI_SMP, TEX_3D, uvw, nodes::num(0)).xyz();

// Light from the scene's radiance cascades for a surface at `world` facing `n`, and how much
// it counts: 1 inside their box, 0 outside it, between the two over its outermost half cell
// (shader_core.glsl's rc_irradiance). Blended between the 8 probes around, leaving out those
// inside things and those behind the surface's own plane: a wall takes none of the light of
// the room behind it.
fn Node[2] rc_irradiance(Node n, Node world) @private {
    Node corner = fs_in("rc_min");
    Node size = fs_in("rc_size");
    Node spacing = corner.w();
    Node g = (world + n * (spacing * 0.25f) - corner.xyz()) / spacing;
    Node edge = nodes::min(g, size.xyz() - g);
    Node fade = nodes::clamp(nodes::min(edge.x(), nodes::min(edge.y(), edge.z())) * 2.0f, 0, 1);
    Node gc = g - 0.5f;
    Node base = nodes::floor(gc);
    Node f = gc - base;
    Node b = nodes::convert(base, IVEC3);
    Node dims = nodes::convert(size.xyz(), IVEC3);
    Node side = nodes::vec3(nodes::select(n.x().greater_than_equal(nodes::num(0)), 0.0f, 1.0f),
                            nodes::select(n.y().greater_than_equal(nodes::num(0)), 2.0f, 3.0f),
                            nodes::select(n.z().greater_than_equal(nodes::num(0)), 4.0f, 5.0f));
    side = nodes::convert(side, IVEC3);
    Node nn = n * n;
    Node sum = nodes::variable(nodes::vec3(0));
    Node wsum = nodes::variable(nodes::num(0));
    Node sum_any = nodes::variable(nodes::vec3(0));
    Node wsum_any = nodes::variable(nodes::num(0));
    Node k = nodes::loop_index(INT);
    Node o = nodes::vec3(nodes::convert(nodes::bit_and(k, nodes::int_(1)), FLOAT),
                         nodes::convert(nodes::bit_and(nodes::shift_right(k, nodes::int_(1)), nodes::int_(1)), FLOAT),
                         nodes::convert(nodes::bit_and(nodes::shift_right(k, nodes::int_(2)), nodes::int_(1)), FLOAT));
    Node c = nodes::clamp(b + nodes::convert(o, IVEC3), nodes::convert(nodes::int_(0), IVEC3), dims - nodes::int_(1));
    Node at = ((c.z() * dims.y() + c.y()) * dims.x() + c.x()) * nodes::int_(6);
    Node ex = node_storage(shaders::VIEW_RC_PROBES, VEC4, at + side.x());
    Node e = ex.xyz() * nn.x() + node_storage(shaders::VIEW_RC_PROBES, VEC4, at + side.y()).xyz() * nn.y()
        + node_storage(shaders::VIEW_RC_PROBES, VEC4, at + side.z()).xyz() * nn.z();
    Node w3 = nodes::mix(nodes::vec3(1) - f, f, o);
    Node w = (w3.x() * w3.y() * w3.z() + 0.002f) * ex.w();
    Node probe_pos = corner.xyz() + (nodes::convert(c, VEC3) + 0.5f) * spacing;
    Node front = nodes::clamp((nodes::dot(probe_pos - world, n) + 0.04f) / 0.06f, 0, 1);
    Node body = nodes::block(sum.assign(sum + e * (w * front)), wsum.assign(wsum + w * front),
                             sum_any.assign(sum_any + e * w), wsum_any.assign(wsum_any + w));
    Node probes = nodes::if_then(fade.greater_than(nodes::num(0)), nodes::block(nodes::loop(k, nodes::int_(0), nodes::int_(8), body)));
    Node fallback = nodes::select(wsum_any.greater_than(nodes::num(1e-5f)), sum_any / wsum_any * size.w(), nodes::vec3(0));
    Node light = nodes::select(wsum.greater_than(nodes::num(1e-4f)), sum / wsum * size.w(), fallback);
    return { nodes::seq(nodes::block(sum, wsum, sum_any, wsum_any, probes), light), fade };
}
