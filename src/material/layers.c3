// Physical material layer BRDFs, volume transmission, indirect lighting (irradiance volumes and
// dense/sparse radiance cascades), planar/SSR reflections, and volumetric fog/shafts as nodes.
// 
// Helper shader functions (`eval_iridescence`, `transmission_*`, `screen_reflection`, `fog_glow`)
// are emitted once per program through `int* fn_id` caches initialized to -1. Screen-space UVs
// flip Y when `ssr_params.w > 0.5` (Metal/HLSL top-left origin vs GLSL bottom-left). Sparse
// radiance probe lookups match the 32-bit cell key layout in shaders/radiance_common.glsl and
// linear probing in shaders/radiance_rays_cs.glsl. Used by src/material/material_lit.c3 and
// src/material/material_graph.c3; exports SSR depth/projection helpers to drie::nodes.
module drie;
import drie::nodes, drie::shading, std::math;

fn Node max3(Node c) @private => nodes::max(c.x(), nodes::max(c.y(), c.z()));

// Schlick's Fresnel: f0 head on, f90 at a glancing angle, `c` the cosine.
fn Node schlick(Node f0, Node f90, Node c) @private => f0 + (f90 - f0) * nodes::pow(c.one_minus(), 5);

// Sheen (three's BRDF_Sheen): the Charlie distribution of fibers, and Neubelt's visibility.
fn Node d_charlie(Node rough, Node noh) @private {
    Node inv   = 1.0f / (rough * rough);
    Node sin2h = nodes::max((noh * noh).one_minus(), 0.0078125f);
    return (inv + 2.0f) * nodes::pow(sin2h, inv * 0.5f) / (2.0f * (float)math::PI);
}

fn Node v_neubelt(Node nov, Node nol) @private {
    return nodes::clamp(1.0f / ((nol + nov - nol * nov) * 4.0f), 0, 1);
}

// How much a sheen reflects of the light from all around (three's IBLSheenBRDF fit).
fn Node sheen_albedo(Node n, Node v, Node rough) @private {
    Node nov = nodes::clamp(nodes::dot(n, v), 0, 1);
    Node r2  = rough * rough;
    Node inv = 1.0f / (rough + 0.1f);
    Node a   = rough * 1.0678f - 1.9362f + r2 * 0.4573f - inv * 0.8469f;
    Node b   = rough * 0.5538f - 0.6014f - r2 * 0.4670f - inv * 0.1255f;
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
    Node d  = nodes::vec3(ab * toh, at * boh, a2 * noh);
    Node w2 = a2 / nodes::dot(d, d);
    return shading::RECIPROCAL_PI * a2 * w2 * w2;
}

// three's EnvironmentBRDF: how much of the light a surface reflects, all directions taken
// together (Karis' fit of the split-sum integral).
fn Node environment_brdf(Node n, Node v, Node f0, Node f90, Node rough) @private {
    Node nov  = nodes::clamp(nodes::dot(n, v), 0, 1);
    Node r    = nodes::vec4(
        rough
    ) * nodes::vec4(-1.0f, -0.0275f, -0.572f, 0.022f) + nodes::vec4(1.0f, 0.0425f, 1.04f, -0.04f);
    Node a004 = nodes::min(r.x() * r.x(), nodes::exp2(nov * -9.28f)) * r.x() + r.y();
    Node fab  = nodes::vec2(-1.04f, 1.04f) * a004 + r.swizzle("zw");
    return f0 * fab.x() + f90 * fab.y();
}

// The reflectance head on that Schlick's approximation would take to reach f at this angle.
fn Node schlick_to_f0(Node f, Node f90, Node c) @private {
    Node x  = nodes::clamp(c.one_minus(), 0, 1);
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
        int f             = node_func("eval_iridescence", VEC3, &types);
        Node outside      = node_param(f, 0);
        Node film_eta     = node_param(f, 1);
        Node c1           = node_param(f, 2);
        Node thick        = node_param(f, 3);
        Node base         = node_param(f, 4);
        Node pi           = nodes::num((float)math::PI);
        // (the film fades into the air as it thins out)
        Node film_ior     = nodes::mix(outside, film_eta, nodes::smoothstep(0, 0.03f, thick));
        Node ratio        = outside / film_ior;
        Node sin2_sq      = ratio * ratio * (c1 * c1).one_minus();
        Node cos2_sq      = sin2_sq.one_minus();
        Node cos2         = nodes::sqrt(cos2_sq);
        // the first interface
        Node rr           = (film_ior - outside) / (film_ior + outside);
        Node r12          = schlick(rr * rr, nodes::num(1), c1);
        Node t121         = r12.one_minus();
        Node phi12        = nodes::select(film_ior.less_than(outside), pi, 0.0f);
        Node phi21        = pi - phi12;
        // the second
        Node s            = nodes::sqrt(nodes::clamp(base, 0, 0.9999f));
        Node base_ior     = (nodes::vec3(1) + s) / (nodes::vec3(1) - s);
        Node rb           = (base_ior - nodes::vec3(film_ior)) / (base_ior + nodes::vec3(film_ior));
        Node r23          = schlick(rb * rb, nodes::vec3(1), cos2);
        Node phi23 = nodes::vec3(nodes::select(base_ior.x()
            .less_than(film_ior), pi, 0.0f), nodes::select(base_ior.y()
                .less_than(film_ior), pi, 0.0f), nodes::select(base_ior.z().less_than(film_ior), pi, 0.0f));
        // the phase shift
        Node opd    = film_ior * 2.0f * thick * cos2;
        Node phi    = nodes::vec3(phi21) + phi23;
        Node r123   = nodes::clamp(r23 * r12, 1e-5f, 0.9999f);
        Node sr123  = nodes::sqrt(r123);
        Node rs     = r23 * (t121 * t121) / (nodes::vec3(1) - r123);
        Node result = rs + r12;
        Node cm     = rs - t121;
        for (int m = 1; m <= 2; m++) {
            cm = cm * sr123;
            result = result + cm * 2.0f * eval_sensitivity(opd * (float)m, phi * (float)m);
        }
        // total internal reflection; and out of gamut colors come out negative
        node_func_body(
            f,
            nodes::select(cos2_sq.less_than(nodes::num(0)), nodes::vec3(1), nodes::max(result, nodes::vec3(0)))
        );
        *fn_id = f;
    }
    Node[5] args = { outside_ior, eta2, cos1, thickness, base_f0 };
    return node_call_fn(*fn_id, &args);
}

// The XYZ sensitivity curves in Fourier space, at an optical path difference (nm), in Rec.709.
fn Node eval_sensitivity(Node opd, Node shift) @local {
    Node phase  = opd * (2.0f * (float)math::PI * 1.0e-9f);
    Node val    = nodes::vec3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
    Node pos    = nodes::vec3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
    Node spread = nodes::vec3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);
    Node xyz    = val * nodes::sqrt(
        spread * (2.0f * (float)math::PI)
    ) * nodes::cos(pos * phase + shift) * nodes::exp(spread * (phase * phase).negate());
    Node x      = xyz.x() + 9.7470e-14f * math::sqrt(
        2.0f * (float)math::PI * 4.5282e+09f
    ) * nodes::cos(phase * 2.2399e+06f + shift.x()) * nodes::exp((phase * phase) * -4.5282e+09f);
    xyz = nodes::vec3(x, xyz.y(), xyz.z()) / 1.0685e-7f;
    Node[3] cols = {
        nodes::vec3(3.2404542f, -0.9692660f, 0.0556434f),
        nodes::vec3(-1.5371385f, 1.8760108f, -0.2040259f),
        nodes::vec3(-0.4985314f, 0.0415560f, 1.0572252f),
    };
    return node_matrix(&cols, MAT3) * xyz;
}

// ---- Transmission

// The picture behind the transmission materials (Renderer's transmission_mips), read at uv on
// level lod with three's bicubic filter (transmission_pars_fragment): four bilinear taps.
fn Node transmission_bicubic(int* fn_id, Node uv, Node lod) @private {
    if (*fn_id < 0) {
        NodeType[2] types = { VEC2, FLOAT };
        int f             = node_func("transmission_bicubic", VEC4, &types);
        Node p            = node_param(f, 0);
        Node l            = node_param(f, 1);
        Node size         = nodes::convert(transmission_texture(SIZE, nodes::convert(l, INT)), VEC2);
        Node texel        = nodes::vec2(1) / size;
        Node q            = p * size + 0.5f;
        Node iuv          = nodes::floor(q);
        Node fr           = nodes::fract(q);
        Node fx           = fr.x();
        Node fy           = fr.y();
        Node g0x          = bicubic_w0(fx) + bicubic_w1(fx);
        Node g1x          = bicubic_w2(fx) + bicubic_w3(fx);
        Node h0x          = bicubic_w1(fx) / g0x - 1.0f;
        Node h1x          = bicubic_w3(fx) / g1x + 1.0f;
        Node g0y          = bicubic_w0(fy) + bicubic_w1(fy);
        Node g1y          = bicubic_w2(fy) + bicubic_w3(fy);
        Node h0y          = bicubic_w1(fy) / g0y - 1.0f;
        Node h1y          = bicubic_w3(fy) / g1y + 1.0f;
        Node t0           = transmission_texture(LEVEL, (nodes::vec2(iuv.x() + h0x, iuv.y() + h0y) - 0.5f) * texel, l);
        Node t1           = transmission_texture(LEVEL, (nodes::vec2(iuv.x() + h1x, iuv.y() + h0y) - 0.5f) * texel, l);
        Node t2           = transmission_texture(LEVEL, (nodes::vec2(iuv.x() + h0x, iuv.y() + h1y) - 0.5f) * texel, l);
        Node t3           = transmission_texture(LEVEL, (nodes::vec2(iuv.x() + h1x, iuv.y() + h1y) - 0.5f) * texel, l);
        node_func_body(f, (t0 * g0x + t1 * g1x) * g0y + (t2 * g0x + t3 * g1x) * g1y);
        *fn_id = f;
    }
    Node[2] args = { uv, lod };
    return node_call_fn(*fn_id, &args);
}

fn Node transmission_texture(
    NodeSampling how,
    Node coords = { },
    Node a = { }
) @local => node_sample(how, VIEW_TEX_TRANSMISSION, SMP_TRANSMISSION_SMP, TEX_2D, coords, a);

fn Node bicubic_w0(Node a) @local => (a * (a * (3.0f - a) - 3.0f) + 1.0f) * (1.0f / 6.0f);
fn Node bicubic_w1(Node a) @local => (a * a * (a * 3.0f - 6.0f) + 4.0f) * (1.0f / 6.0f);
fn Node bicubic_w2(Node a) @local => (a * (a * (a * -3.0f + 3.0f) + 3.0f) + 1.0f) * (1.0f / 6.0f);
fn Node bicubic_w3(Node a) @local => a * a * a * (1.0f / 6.0f);

// What is behind the surface where a ray leaves it at `exit`: the transmission picture there,
// blurrier the rougher the surface (and the stronger it bends light: none at an IOR of 1).
fn Node transmission_sample(int* fn_id, int* bicubic_fn, Node exit, Node rough, Node ior) @private {
    if (*fn_id < 0) {
        NodeType[3] types = { VEC3, FLOAT, FLOAT };
        int f             = node_func("transmission_sample", VEC3, &types);
        Node e            = node_param(f, 0);
        Node r            = node_param(f, 1);
        Node i            = node_param(f, 2);
        Node clip         = fs_in("cluster_view_proj", MAT4) * nodes::vec4(e, 1);
        Node uv           = clip.xy() / nodes::max(nodes::abs(clip.w()), 1e-5f) * 0.5f + 0.5f;
        // (the picture's row 0 is its top where the target's is, as for the reflection source)
        uv = nodes::select(fs_in("ssr_params", FLOAT, 3)
            .greater_than(nodes::num(0.5f)), nodes::vec2(uv.x(), uv.y().one_minus()), uv);
        Node top = nodes::convert(transmission_texture(LEVELS), FLOAT) - 1.0f;
        Node lod = nodes::clamp(
            nodes::log2(nodes::max(fs_in("ext_params", FLOAT, 3), 1.0f)) * r * nodes::clamp(i * 2.0f - 2.0f, 0, 1),
            nodes::num(0),
            top
        );
        Node light  = nodes::variable(transmission_bicubic(bicubic_fn, uv, nodes::floor(lod)).xyz());
        Node coarse = transmission_bicubic(bicubic_fn, uv, nodes::min(nodes::ceil(lod), top)).xyz();
        Node between = nodes::if_then(nodes::fract(lod)
            .greater_than(nodes::num(0)), nodes::block(light.assign(nodes::mix(light, coarse, nodes::fract(lod)))));
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
        int f             = node_func("transmission_ray", VEC3, &types);
        Node pn           = node_param(f, 0);
        Node pv           = node_param(f, 1);
        Node thick        = node_param(f, 2);
        Node i            = node_param(f, 3);
        Node ray          = nodes::refract(pv.negate(), pn, 1.0f / i);
        Node dir          = nodes::normalize(ray);
        Node across       = thick * fs_in("ext_params", FLOAT, 2);
        Node bits         = nodes::convert(fs_in("ext_params", FLOAT) + 0.5f, INT);
        Node plane        = nodes::bit_and(bits, nodes::int_(64)).not_equal(nodes::int_(0));
        across = nodes::select(plane, across / nodes::max(nodes::dot(dir, pn.negate()), 1.0f / 12.0f), across);
        node_func_body(f, nodes::select(nodes::dot(ray, ray)
            .less_than(nodes::num(1e-8f)), nodes::vec3(0), dir * across));
        *fn_id = f;
    }
    Node[4] args = { n, v, thickness, ior };
    return node_call_fn(*fn_id, &args);
}

// Beer's law over `distance` meters (transmission_params.w: the attenuation distance; 0: none).
fn Node volume_attenuation(Node distance) @private {
    Node tp          = fs_in("transmission_params");
    Node coefficient = nodes::log(nodes::max(fs_in("transmission_color", VEC3), nodes::vec3(1e-4f))).negate() / tp.w();
    return nodes::select(tp.w()
        .less_than_equal(nodes::num(0)), nodes::vec3(1), nodes::exp(coefficient.negate() * distance));
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
fn Node Refraction.light(
    &self,
    Node n,
    Node v,
    Node world,
    Node rough,
    Node diffuse,
    Node thickness,
    Node f0,
    Node f90
) {
    Node tp      = fs_in("transmission_params");
    Node ior     = nodes::select(tp.x().greater_than(nodes::num(0)), tp.x(), 1.5f);
    Node spread  = (ior - 1.0f) * 0.025f * tp.z();
    Node light   = nodes::variable(nodes::vec3(0));
    Node through = nodes::variable(nodes::vec3(0));
    Node[3] s;
    Node[3] a;
    for (int i = 0; i < 3; i++) {
        Node ior_i = i == 1 ? ior : ior + spread * (float)(i - 1);
        Node ray   = transmission_ray(&self.ray_fn, n, v, thickness, ior_i);
        s[i] = transmission_sample(&self.sample_fn, &self.bicubic_fn, world + ray, rough, ior_i);
        a[i] = volume_attenuation(nodes::length(ray));
    }
    Node ray = transmission_ray(&self.ray_fn, n, v, thickness, ior);
    Node spread_out = nodes::block(
        light.assign(nodes::vec3(s[0].x(), s[1].y(), s[2].z())),
        through.assign(diffuse * nodes::vec3(a[0].x(), a[1].y(), a[2].z()))
    );
    Node one = nodes::block(
        light.assign(transmission_sample(&self.sample_fn, &self.bicubic_fn, world + ray, rough, ior)),
        through.assign(diffuse * volume_attenuation(nodes::length(ray)))
    );
    Node pick = nodes::if_then(spread.greater_than(nodes::num(0)), spread_out, one);
    return nodes::seq(
        nodes::block(light, through, pick),
        (nodes::vec3(1) - environment_brdf(n, v, f0, f90, rough)) * through * light
    );
}

// ---- Indirect lighting

// Light from the scene's irradiance volumes for a surface at `world` facing `n`, already times
// how much of the point they cover (1 inside a volume, 0 outside them all), and that cover
// (matching shaders/shader_core.glsl's gi_irradiance). Each volume covers its box and fades out
// over half a cell outside it, in front of earlier ones. Each probe holds irradiance from the
// six axis directions laid out side by side along x in the volume's region of VIEW_TEX_GI.
fn Node[2] gi_irradiance(Node n, Node world) @private {
    Node cover  = nodes::variable(nodes::num(0));
    Node e      = nodes::variable(nodes::vec3(0));
    Node slab   = nodes::vec3(nodes::select(n.x()
        .greater_than_equal(nodes::num(0)), 0.0f, 1.0f), nodes::select(n.y()
            .greater_than_equal(nodes::num(0)), 2.0f, 3.0f), nodes::select(n.z()
                .greater_than_equal(nodes::num(0)), 4.0f, 5.0f)) / 6.0f;
    Node w      = n * n;
    Node i      = nodes::loop_index(INT);
    Node lo     = fs_at("gi_min", VEC4, i);
    Node dims   = fs_at("gi_dims", VEC4, i);
    Node inv    = fs_at("gi_inv_size", VEC4, i);
    Node atlas  = fs_at("gi_atlas", VEC4, i);
    Node res    = dims.xyz();
    Node p      = (world + n * dims.w() - lo.xyz()) * inv.xyz();
    Node edge   = nodes::min(p, nodes::vec3(1) - p) * res;
    Node fade   = nodes::clamp(nodes::min(edge.x(), nodes::min(edge.y(), edge.z())) * 2.0f + 1.0f, 0, 1);
    // stay inside the volume and a direction's slab
    Node margin = nodes::vec3(0.5f) / res;
    p = nodes::clamp(p, margin, nodes::vec3(1) - margin);
    Node u  = (slab + p.x() / 6.0f) * atlas.y() + atlas.x();
    Node yz = p.swizzle("yz") * atlas.swizzle("zw");
    Node l  = gi_texel(
        nodes::vec3(u.x(), yz)
    ) * w.x() + gi_texel(nodes::vec3(u.y(), yz)) * w.y() + gi_texel(nodes::vec3(u.z(), yz)) * w.z();
    Node body = nodes::block(nodes::if_then(lo.w()
        .less_than(
            nodes::num(0.5f)
        ), nodes::block(
            nodes::break_()
        )), nodes::if_then(
            fade.less_than_equal(nodes::num(0)),
            nodes::block(nodes::continue_())
        ), e.assign(nodes::mix(e, l * inv.w(), fade)), cover.assign(nodes::mix(cover, nodes::num(1), fade)));
    Node volumes = nodes::block(cover, e, nodes::loop(i, nodes::int_(0), nodes::int_(4), body));
    return { nodes::seq(volumes, e), cover };
}

fn Node gi_texel(
    Node uvw
) @local => node_sample(LEVEL, shaders::VIEW_TEX_GI, shaders::SMP_GI_SMP, TEX_3D, uvw, nodes::num(0))
    .xyz();

// Radiance cascades irradiance at `world` facing `n`, plus box blend weight (1 inside, 0 outside,
// fading over the outermost half cell; matching shaders/shader_core.glsl's rc_irradiance). Blends
// the 8 surrounding probes while rejecting occluded probes and probes behind the surface plane.
fn Node[2] rc_irradiance(Node n, Node world) @private {
    Node corner    = fs_in("rc_min");
    Node size      = fs_in("rc_size");
    Node light     = nodes::variable(nodes::vec3(0));
    Node fade      = nodes::variable(nodes::num(0));
    Node[2] dense  = rc_irradiance_dense(n, world, corner, size);
    Node[2] sparse = rc_irradiance_sparse(n, world, corner, size);
    Node run = nodes::block(light, fade, nodes::if_then(size.x()
        .less_than(
            nodes::num(-0.5f)
        ), nodes::block(
            light.assign(sparse[0]),
            fade.assign(sparse[1])
        ), nodes::block(light.assign(dense[0]), fade.assign(dense[1]))));
    return { nodes::seq(run, light), fade };
}

// Sparse radiance volume lookup: probes the hash table for each of the 8 cells around `world`
// (`rc_sparse_lookup`), blends their 6 faces (matching shaders/shader_core.glsl's rc_irradiance),
// and falls back to the cascade level 1 probe when no level 0 probes are populated.
fn Node[2] rc_irradiance_sparse(Node n, Node world, Node corner, Node size) @private {
    Node spacing = corner.w();
    Node nn      = n * n;
    Node side = nodes::convert(nodes::vec3(nodes::select(n.x()
        .greater_than_equal(nodes::num(0)), 0.0f, 1.0f), nodes::select(n.y()
            .greater_than_equal(nodes::num(0)), 2.0f, 3.0f), nodes::select(n.z()
                .greater_than_equal(nodes::num(0)), 4.0f, 5.0f)), IVEC3);
    Node hash_cap = nodes::convert(nodes::negate(size.x()) + 0.5f, INT);
    // Box cell counts packed into FsParams.rc_size y and z (src/renderer/draw.c3).
    Node dims = nodes::vec3(
        nodes::convert(nodes::bit_and(nodes::convert(size.y(), INT), nodes::int_(511)), FLOAT),
        nodes::convert(nodes::shift_right(nodes::convert(size.y(), INT), nodes::int_(9)), FLOAT),
        size.z()
    );
    Node cell     = (world - corner.xyz()) / spacing;
    Node bedge    = nodes::min(cell, dims - cell);
    Node bfade    = nodes::clamp(nodes::min(bedge.x(), nodes::min(bedge.y(), bedge.z())) * 2.0f, 0, 1);
    // (half a cell out along the normal: the sheet of probes the kernels place sits in the
    // first cell off the surface, which a wall as thick as a cell would hide)
    Node out      = world + n * (spacing * 0.5f);
    Node g        = out / spacing - 0.5f;
    Node base     = nodes::convert(nodes::floor(g), IVEC3);
    Node f        = g - nodes::convert(base, VEC3);
    Node sum      = nodes::variable(nodes::vec3(0));
    Node wsum     = nodes::variable(nodes::num(0));
    Node sum_any  = nodes::variable(nodes::vec3(0));
    Node wsum_any = nodes::variable(nodes::num(0));
    Node light    = nodes::variable(nodes::vec3(0));
    Node fade     = nodes::variable(nodes::num(0));
    Node entry    = nodes::variable(nodes::vec4(0));
    Node slot     = nodes::variable(nodes::int_(-1));
    Node at       = nodes::variable(nodes::int_(0));
    Node ex       = nodes::variable(nodes::vec4(0));
    Node e3       = nodes::variable(nodes::vec3(0));
    Node k        = nodes::loop_index(INT);
    Node o = nodes::vec3(
        nodes::convert(nodes::bit_and(k, nodes::int_(1)), FLOAT),
        nodes::convert(nodes::bit_and(nodes::shift_right(k, nodes::int_(1)), nodes::int_(1)), FLOAT),
        nodes::convert(nodes::bit_and(nodes::shift_right(k, nodes::int_(2)), nodes::int_(1)), FLOAT)
    );
    Node c         = base + nodes::convert(o, IVEC3);
    // one of the 8 probes: its 6 faces, and how much of them this surface takes
    Node probe_pos = (nodes::convert(c, VEC3) + 0.5f) * spacing;
    Node front     = nodes::smoothstep(spacing * -0.35f, spacing * 0.25f, nodes::dot(probe_pos - world, n));
    // the anchor the kernels keep in the entry's w (rc_sparse_anchor): which way the surface
    // the probe was made for faces, so a wall takes none of the light of the room behind it
    Node code = nodes::convert(
        nodes::bit_and(nodes::shift_right(nodes::as_uint(entry.w()), nodes::uint_(16)), nodes::uint_(7u)),
        INT
    );
    Node a    = code - nodes::int_(1);
    Node axis = nodes::shift_right(a, nodes::int_(1));
    Node pick = nodes::vec3(
        nodes::select(axis.equal(nodes::int_(0)), 1.0f, 0.0f),
        nodes::select(axis.equal(nodes::int_(1)), 1.0f, 0.0f),
        nodes::select(axis.equal(nodes::int_(2)), 1.0f, 0.0f)
    );
    Node against = nodes::dot(world - probe_pos, pick) * nodes::select(nodes::bit_and(a, nodes::int_(1))
        .equal(nodes::int_(0)), 1.0f, -1.0f);
    Node trust = nodes::select(
        code.equal(nodes::int_(0)),
        nodes::num(1),
        nodes::smoothstep(spacing * -1.5f, spacing * -0.9f, against)
    );
    Node w3    = nodes::mix(nodes::vec3(1) - f, f, o);
    Node w     = w3.x() * w3.y() * w3.z() * ex.w();
    Node faces = ex.xyz() * nn.x() + node_storage(shaders::VIEW_RC_PROBES, VEC4, at + side.y())
        .xyz() * nn.y() + node_storage(shaders::VIEW_RC_PROBES, VEC4, at + side.z()).xyz() * nn.z();
    Node[11] body_stmts;
    body_stmts[:10] = rc_sparse_lookup(c, 0, hash_cap, entry, slot)[..];
    body_stmts[10] = nodes::if_then(
        slot.greater_than_equal(nodes::int_(0)),
        nodes::block(
            at.assign(hash_cap + slot * nodes::int_(6)),
            ex.assign(node_storage(shaders::VIEW_RC_PROBES, VEC4, at + side.x())),
            e3.assign(faces),
            sum.assign(sum + e3 * (w * front * trust)),
            wsum.assign(wsum + w * front * trust),
            sum_any.assign(sum_any + e3 * w),
            wsum_any.assign(wsum_any + w)
        )
    );
    Node body        = drie::node_block(&body_stmts);
    Node fallback    = sum_any / wsum_any;
    Node primary     = nodes::select(wsum.greater_than(nodes::num(1e-5f)), sum / wsum, fallback);
    Node trust_front = nodes::clamp(wsum / nodes::max(wsum_any * 0.25f, 1e-5f), 0, 1);
    // the level 1 probe: where its 6 faces are is in its entry's x
    Node c1          = nodes::convert(nodes::floor(out / (spacing * 2.0f)), IVEC3);
    Node[12] level1_stmts;
    level1_stmts[:10] = rc_sparse_lookup(c1, 1, hash_cap, entry, slot)[..];
    level1_stmts[10] = at.assign(nodes::convert(nodes::as_uint(entry.x()), INT));
    level1_stmts[11] = nodes::if_then(
        nodes::logical_and(slot.greater_than_equal(nodes::int_(0)), at.greater_than(nodes::int_(0))),
        nodes::block(
            ex.assign(node_storage(shaders::VIEW_RC_PROBES, VEC4, at + side.x())),
            fade.assign(nodes::num(1)),
            light.assign(faces * size.w())
        )
    );
    Node level1 = drie::node_block(&level1_stmts);
    Node run = nodes::block(
        sum,
        wsum,
        sum_any,
        wsum_any,
        light,
        fade,
        entry,
        slot,
        at,
        ex,
        e3,
        nodes::if_then(
            bfade.greater_than(nodes::num(0)),
            nodes::block(
                nodes::loop(k, nodes::int_(0), nodes::int_(8), body),
                nodes::if_then(
                    wsum_any.greater_than(nodes::num(1e-5f)),
                    nodes::block(
                        fade.assign(nodes::clamp(wsum_any * 4.0f, 0, 1) * bfade),
                        light.assign(nodes::mix(fallback, primary, trust_front) * size.w())
                    ),
                    level1
                )
            )
        )
    );
    return { nodes::seq(run, light), fade };
}

// Hash-table probe lookup for cell `c` at cascade `level` (shaders/radiance_rays_cs.glsl's
// rc_lookup: rc_hash plus up to 32 linear steps), writing `entry` and `slot` (-1 if absent).
fn Node[10] rc_sparse_lookup(Node c, uint level, Node hash_cap, Node entry, Node slot) @private {
    Node idx   = nodes::variable(nodes::uint_(0));
    Node stepn = nodes::variable(nodes::uint_(0));
    Node h     = nodes::variable(nodes::uint_(0));
    Node e     = nodes::variable(nodes::vec4(0));
    Node hmask = nodes::convert(hash_cap, UINT) - nodes::uint_(1);
    // rc_pack_key32 (shaders/radiance_common.glsl)
    Node key = nodes::bit_or(
        nodes::bit_or(
            nodes::bit_and(nodes::convert(c.x(), UINT), nodes::uint_(0x1ffu)),
            nodes::shift_left(nodes::bit_and(nodes::convert(c.y(), UINT), nodes::uint_(0x1ffu)), nodes::uint_(9))
        ),
        nodes::bit_or(
            nodes::shift_left(nodes::bit_and(nodes::convert(c.z(), UINT), nodes::uint_(0x1ffu)), nodes::uint_(18)),
            nodes::uint_(0x80000000u | level << 27)
        )
    );
    return {
        entry.assign(nodes::vec4(0)),
        slot.assign(nodes::int_(-1)),
        stepn.assign(nodes::uint_(0)),
        h.assign(key),
        h.assign(nodes::bit_xor(h, nodes::shift_right(h, nodes::uint_(16)))),
        h.assign(h * nodes::uint_(0x7feb352du)),
        h.assign(nodes::bit_xor(h, nodes::shift_right(h, nodes::uint_(15)))),
        h.assign(h * nodes::uint_(0x846ca68bu)),
        idx.assign(nodes::bit_and(nodes::bit_xor(h, nodes::shift_right(h, nodes::uint_(16))), hmask)),
        nodes::loop_while(
            stepn.less_than(nodes::uint_(32)),
            nodes::block(
                e.assign(node_storage(shaders::VIEW_RC_PROBES, VEC4, idx)),
                nodes::if_then(nodes::as_uint(e.y())
        .equal(nodes::uint_(0u)), nodes::block(nodes::break_())), nodes::if_then(nodes::as_uint(e.y())
            .equal(
                key
            ), nodes::block(
                entry.assign(e),
                slot.assign(nodes::convert(nodes::as_uint(e.z()), INT)),
                nodes::break_()
            )), idx.assign(nodes::bit_and(idx + nodes::uint_(1), hmask)), stepn.assign(stepn + nodes::uint_(1)))) };
}

fn Node[2] rc_irradiance_dense(Node n, Node world, Node corner, Node size) @private {
    Node spacing = corner.w();
    Node g       = (world + n * (spacing * 0.25f) - corner.xyz()) / spacing;
    Node edge    = nodes::min(g, size.xyz() - g);
    Node fade    = nodes::clamp(nodes::min(edge.x(), nodes::min(edge.y(), edge.z())) * 2.0f, 0, 1);
    Node gc      = g - 0.5f;
    Node base    = nodes::floor(gc);
    Node f       = gc - base;
    Node b       = nodes::convert(base, IVEC3);
    Node dims    = nodes::convert(size.xyz(), IVEC3);
    Node side = nodes::vec3(nodes::select(n.x()
        .greater_than_equal(nodes::num(0)), 0.0f, 1.0f), nodes::select(n.y()
            .greater_than_equal(nodes::num(0)), 2.0f, 3.0f), nodes::select(n.z()
                .greater_than_equal(nodes::num(0)), 4.0f, 5.0f));
    side = nodes::convert(side, IVEC3);
    Node nn       = n * n;
    Node sum      = nodes::variable(nodes::vec3(0));
    Node wsum     = nodes::variable(nodes::num(0));
    Node sum_any  = nodes::variable(nodes::vec3(0));
    Node wsum_any = nodes::variable(nodes::num(0));
    Node k        = nodes::loop_index(INT);
    Node o = nodes::vec3(
        nodes::convert(nodes::bit_and(k, nodes::int_(1)), FLOAT),
        nodes::convert(nodes::bit_and(nodes::shift_right(k, nodes::int_(1)), nodes::int_(1)), FLOAT),
        nodes::convert(nodes::bit_and(nodes::shift_right(k, nodes::int_(2)), nodes::int_(1)), FLOAT)
    );
    Node c = nodes::clamp(b + nodes::convert(o, IVEC3), nodes::convert(nodes::int_(0), IVEC3), dims - nodes::int_(1));
    Node at = ((c.z() * dims.y() + c.y()) * dims.x() + c.x()) * nodes::int_(6);
    Node ex = node_storage(shaders::VIEW_RC_PROBES, VEC4, at + side.x());
    Node e = ex.xyz() * nn.x() + node_storage(shaders::VIEW_RC_PROBES, VEC4, at + side.y())
        .xyz() * nn.y() + node_storage(shaders::VIEW_RC_PROBES, VEC4, at + side.z()).xyz() * nn.z();
    Node w3 = nodes::mix(nodes::vec3(1) - f, f, o);
    Node w = w3.x() * w3.y() * w3.z() * ex.w();
    Node probe_pos = corner.xyz() + (nodes::convert(c, VEC3) + 0.5f) * spacing;
    Node front = nodes::smoothstep(spacing * -0.35f, spacing * 0.25f, nodes::dot(probe_pos - world, n));
    Node body = nodes::block(
        sum.assign(sum + e * (w * front)),
        wsum.assign(wsum + w * front),
        sum_any.assign(sum_any + e * w),
        wsum_any.assign(wsum_any + w)
    );
    Node probes = nodes::if_then(
        fade.greater_than(nodes::num(0)),
        nodes::block(nodes::loop(k, nodes::int_(0), nodes::int_(8), body))
    );
    Node fallback    = nodes::select(wsum_any.greater_than(nodes::num(1e-5f)), sum_any / wsum_any, nodes::vec3(0));
    Node primary     = nodes::select(wsum.greater_than(nodes::num(1e-5f)), sum / wsum, fallback);
    Node trust_front = nodes::clamp(wsum / nodes::max(wsum_any * 0.25f, 1e-5f), 0, 1);
    Node light       = nodes::mix(fallback, primary, trust_front) * size.w();
    return { nodes::seq(nodes::block(sum, wsum, sum_any, wsum_any, probes), light), fade };
}

// ---- Planar and screen-space reflections

// Interleaved gradient noise (Jimenez, matching shaders/shader_core.glsl's ssr_noise) at pixel
// `frag`, rotated each frame by `ssr_params.z`.
fn Node frame_noise(Node frag, Node shift = { }) @private {
    Node p = frag + nodes::vec2(47.0f, 17.0f) * fs_in("ssr_params", FLOAT, 2);
    if (shift.id) p = p + shift;
    return nodes::fract(nodes::fract(nodes::dot(p, nodes::vec2(0.06711056f, 0.00583715f))) * 52.9829189f);
}

// Samples a Reflector image (src/lighting/reflector.c3) along the reflected ray from `world`:
// exact on flat planes and offset by `reflector_params.y` on perturbed normals, blurred by
// roughness with 8 rotated disc taps from shared map slot `slot` (`tex_ext0..5`); w fades at edges.
fn Node planar_reflection(Node n, Node v, Node rough, Node world, Node frag, int slot) @private {
    Node params = fs_in("reflector_params");
    Node r      = nodes::reflect(v.negate(), n);
    Node clip   = fs_in("reflector_view_proj", MAT4) * nodes::vec4(world + r * params.y(), 1);
    Node ndc    = clip.xy() / nodes::max(clip.w(), 1e-4f);
    ndc = nodes::select(fs_in("ssr_params", FLOAT, 3)
        .greater_than(nodes::num(0.5f)), nodes::vec2(ndc.x(), ndc.y().negate()), ndc);
    Node uv   = ndc * 0.5f + 0.5f;
    Node edge = nodes::min(uv, nodes::vec2(1) - uv);
    Node found = nodes::select(clip.w()
        .greater_than(nodes::num(0)), nodes::smoothstep(-0.02f, 0.03f, nodes::min(edge.x(), edge.y())), 0.0f);
    Node size = nodes::convert(ssr_texture(shaders::VIEW_TEX_EXT0 + slot, SIZE, nodes::int_(0)), VEC2);
    Node spread = rough * rough * params.z();
    Node radius = nodes::vec2(spread * size.y() / size.x(), spread);
    Node turn = frame_noise(frag) * (2.0f * (float)math::PI);
    Node sum = nodes::variable(nodes::vec3(0));
    Node k = nodes::loop_index(INT);
    Node a = turn + nodes::convert(k, FLOAT) * 2.39996f;
    Node at = uv + nodes::vec2(
        nodes::cos(a),
        nodes::sin(a)
    ) * radius * nodes::sqrt((nodes::convert(k, FLOAT) + 0.5f) / 8.0f);
    Node tap = nodes::min(ssr_texture(shaders::VIEW_TEX_EXT0 + slot, LEVEL, at, nodes::num(0)).xyz(), nodes::vec3(64));
    Node taps = nodes::loop(k, nodes::int_(0), nodes::int_(8), nodes::block(sum.assign(sum + tap)));
    return nodes::seq(nodes::block(sum, taps), nodes::vec4(sum / 8.0f, found));
}

fn Node ssr_texture(
    int view,
    NodeSampling how,
    Node coords = { },
    Node a = { }
) @local => node_sample(how, view, shaders::SMP_SSR_SMP, TEX_2D, coords, a);

// Where a world-space point was in the last frame: xy = uv, z = distance from the camera's plane.
fn Node ssr_project(Node p) @local {
    Node clip = fs_in("ssr_view_proj", MAT4) * nodes::vec4(p, 1);
    Node ndc  = clip.xy() / nodes::max(clip.w(), 1e-4f);
    ndc = nodes::select(fs_in("ssr_params", FLOAT, 3)
        .greater_than(nodes::num(0.5f)), nodes::vec2(ndc.x(), ndc.y().negate()), ndc);
    return nodes::vec3(ndc * 0.5f + 0.5f, clip.w());
}

// The last frame's distance from the camera's plane at uv: the depth source's (green), or its
// back faces' (red of ssr_back).
fn Node ssr_depth_at(int view, Node uv, bool back = false) @local {
    Node size = ssr_texture(view, SIZE, nodes::int_(0));
    Node texel = nodes::clamp(
        nodes::convert(uv * nodes::convert(size, VEC2), IVEC2),
        nodes::convert(nodes::int_(0), IVEC2),
        size - nodes::int_(1)
    );
    Node c = ssr_texture(view, FETCH, texel, nodes::int_(0));
    return back ? c.x() : c.y();
}

// Returns a bool node testing whether SSR depth is active on this material (`nodes::ssr_on`).
fn Node node_ssr_on() => fs_in("ssr_params", FLOAT).greater_than(nodes::num(0));
// Projects world point `p` to last-frame screen UV (xy) and linear depth (z).
fn Node node_ssr_project(Node p) => ssr_project(nodes::convert(p, VEC3));
// Fetches linear view depth (m) from VIEW_SSR_DEPTH at screen UV `uv` (`nodes::viewport_depth`).
fn Node node_viewport_depth(Node uv) => ssr_depth_at(shaders::VIEW_SSR_DEPTH, nodes::convert(uv, VEC2));

// A microfacet normal of a GGX surface as the eye sees them (the distribution of visible
// normals, Heitz 2018), in the surface's own frame (z = its normal); ve = towards the eye.
fn Node ggx_visible_normal(Node ve, Node alpha, Node u) @local {
    Node vh    = nodes::normalize(nodes::vec3(ve.xy() * alpha, ve.z()));
    Node lensq = nodes::dot(vh.xy(), vh.xy());
    Node t1 = nodes::select(lensq.greater_than(nodes::num(1e-8f)), nodes::vec3(vh.y()
        .negate(), vh.x(), 0) * nodes::inverse_sqrt(lensq), nodes::vec3(1, 0, 0));
    Node t2     = nodes::cross(vh, t1);
    Node rad    = nodes::sqrt(u.x());
    Node phi    = u.y() * (2.0f * (float)math::PI);
    Node a      = rad * nodes::cos(phi);
    Node b      = rad * nodes::sin(phi);
    Node half_z = (vh.z() + 1.0f) * 0.5f;
    b = half_z.one_minus() * nodes::sqrt(nodes::max((a * a).one_minus(), 0)) + half_z * b;
    Node nh = t1 * a + t2 * b + vh * nodes::sqrt(nodes::max((a * a + b * b).one_minus(), 0));
    return nodes::normalize(nodes::vec3(nh.xy() * alpha, nodes::max(nh.z(), 1e-4f)));
}

// Screen-space reflection (matching shaders/shader_core.glsl's screen_reflection; emitted once
// per program when `*fn_id < 0`): marches the reflected ray through last-frame depth and returns
// linear reflected radiance plus hit weight in w (0 on miss or off-screen). Rough surfaces jitter
// the ray within a GGX visible-normal cone and rely on temporal filtering to converge.
fn Node screen_reflection(int* fn_id, Node n, Node geo_n, Node v, Node rough, Node world, Node frag) @private {
    if (*fn_id < 0) {
        NodeType[6] types = { VEC3, VEC3, VEC3, FLOAT, VEC3, VEC2 };
        int f             = node_func("screen_reflection", VEC4, &types);
        Node pn           = node_param(f, 0);
        Node pg           = node_param(f, 1);
        Node pv           = node_param(f, 2);
        Node pr           = node_param(f, 3);
        Node pw           = node_param(f, 4);
        Node pf           = node_param(f, 5);
        node_func_body(f, ssr_march(pn, pg, pv, pr, pw, pf));
        *fn_id = f;
    }
    Node[6] args = { n, geo_n, v, rough, world, frag };
    return node_call_fn(*fn_id, &args);
}

// (how many steps a reflected ray takes at most, Renderer.ssr_steps; how much of a rough
// surface's lobe the ray itself is sent through, the inner part, by chance: the frames add up
// to it, the rest is blurred from what the ray finds; all of it by chance makes a small bright
// thing into a patch of sparks)
const int SSR_MAX_STEPS @local    = 128;
const float SSR_LOBE_SHARE @local = 0.45;
const float SSR_BLUR @local       = 0.28;

fn Node ssr_march(Node n, Node geo_n, Node v, Node rough, Node world, Node frag) @local {
    Node n1 = frame_noise(frag);
    Node n2 = frame_noise(frag, nodes::vec2(31.7f, 11.3f));
    Node turn = n1 * (2.0f * (float)math::PI);
    // the ray mirrors off one of the surface's microfacets, picked as the eye meets them: a
    // rough floor draws a light out into a streak towards the eye, as a real one does
    Node alpha = rough * rough;
    Node up = nodes::select(nodes::abs(n.y()).less_than(nodes::num(0.9f)), nodes::vec3(0, 1, 0), nodes::vec3(1, 0, 0));
    Node tx = nodes::normalize(nodes::cross(up, n));
    Node ty = nodes::cross(n, tx);
    Node ve = nodes::normalize(nodes::vec3(nodes::dot(v, tx), nodes::dot(v, ty), nodes::max(nodes::dot(v, n), 0.05f)));
    Node h = ggx_visible_normal(ve, alpha, nodes::vec2(n2 * SSR_LOBE_SHARE, n1));
    Node r = nodes::reflect(v.negate(), nodes::normalize(tx * h.x() + ty * h.y() + n * h.z()));
    Node away = nodes::if_then(nodes::dot(r, geo_n)
        .less_than(nodes::num(0.02f)), nodes::block(node_return(nodes::vec4(0))));

    Node origin    = world + geo_n * 0.02f;
    Node screen_px = nodes::convert(ssr_texture(shaders::VIEW_SSR_DEPTH, SIZE, nodes::int_(0)), VEC2);
    Node step_len  = nodes::variable(nodes::num(0.06f));
    Node t         = nodes::variable((n2 + 0.4f) * 0.06f);
    Node t_prev    = nodes::variable(nodes::num(0));
    Node quality   = fs_in("quality_params");
    Node steps = nodes::select(quality.y()
        .greater_than(
            nodes::num(0)
        ), nodes::min(nodes::convert(quality.y(), INT), nodes::int_(SSR_MAX_STEPS)), nodes::int_(40));
    Node max_step  = nodes::select(quality.z().greater_than(nodes::num(0)), quality.z(), 6.0f);
    Node thickness = fs_in("ssr_thickness");
    Node i         = nodes::loop_index(INT);
    Node s         = ssr_project(origin + r * t);
    Node outside = nodes::logical_or(nodes::logical_or(s.z()
        .less_than(nodes::num(0.05f)), s.x()
            .less_than(nodes::num(0))), nodes::logical_or(nodes::logical_or(s.y()
                .less_than(nodes::num(0)), s.x().greater_than(nodes::num(1))), s.y().greater_than(nodes::num(1))));
    // behind what the camera saw there: where the ray went in lies between the step before
    // and this one (steps get longer than thin things are thick, so this is found first)
    Node behind = (s.z() - ssr_depth_at(shaders::VIEW_SSR_DEPTH, s.xy())).greater_than(nodes::num(0.02f));
    Node lo     = nodes::variable(t_prev);
    Node hi     = nodes::variable(t);
    Node mid    = (lo + hi) * 0.5f;
    Node m      = ssr_project(origin + r * mid);
    Node k      = nodes::loop_index(INT);
    Node halve = nodes::loop(
        k,
        nodes::int_(0),
        nodes::int_(6),
        nodes::block(
            nodes::if_then(
                (m.z() - ssr_depth_at(shaders::VIEW_SSR_DEPTH, m.xy())).greater_than(nodes::num(0.02f)),
                nodes::block(hi.assign(mid)),
                nodes::block(lo.assign(mid))
            )
        )
    );
    // and there it has to be inside the thing, not past it: as thick as its back faces say, or
    // else about as thick as a thing is, fading out past that
    Node p_in  = ssr_project(origin + r * hi);
    Node front = ssr_depth_at(shaders::VIEW_SSR_DEPTH, p_in.xy());
    Node back  = nodes::variable(nodes::num(0));
    Node read_back = nodes::if_then(thickness.x()
        .greater_than(
            nodes::num(0.5f)
        ), nodes::block(back.assign(ssr_depth_at(shaders::VIEW_SSR_BACK, p_in.xy(), true))));
    Node backed = nodes::logical_and(back.greater_than(front + 0.005f), back.less_than(front + 2.0f));
    Node hit = nodes::select(backed, nodes::smoothstep(0.0f, 0.03f, p_in.z() - back)
        .one_minus(), nodes::smoothstep(0.15f, 0.3f, p_in.z() - front).one_minus());
    Node edge   = nodes::min(p_in.xy(), nodes::vec2(1) - p_in.xy());
    Node found  = hit * nodes::smoothstep(
        0.0f,
        0.07f,
        nodes::min(edge.x(), edge.y())
    ) * nodes::smoothstep(0.6f, 1.0f, hi / thickness.y())
        .one_minus();
    // what a rough surface mirrors is spread over a disc that grows with the way the ray went:
    // a few taps over it, turned differently at every pixel
    Node size   = nodes::convert(ssr_texture(shaders::VIEW_SSR_COLOR, SIZE, nodes::int_(0)), VEC2);
    Node spread = alpha * SSR_BLUR * hi / nodes::max(p_in.z(), 0.1f);
    Node radius = nodes::vec2(spread * size.y() / size.x(), spread);
    Node sum    = nodes::variable(nodes::vec3(0));
    Node j      = nodes::loop_index(INT);
    Node a      = turn + nodes::convert(j, FLOAT) * 2.39996f;
    Node at     = p_in.xy() + nodes::vec2(
        nodes::cos(a),
        nodes::sin(a)
    ) * radius * nodes::sqrt((nodes::convert(j, FLOAT) + 0.5f) / 8.0f);
    Node c      = ssr_texture(shaders::VIEW_SSR_COLOR, LEVEL, at, nodes::num(0));
    // as it was lit, kept within bounds so a few bright pixels don't feed themselves frame
    // after frame; or back from what the tone mapping made of it, roughly (light sources were
    // far over white)
    Node lin    = shading::srgb_to_linear(c.xyz());
    lin = lin / nodes::max(max3(lin) * -0.7f + 1.0f, 0.3f);
    Node toned = lin * (c.w().one_minus() * 5.0f + 1.0f) / nodes::max(fs_in("camera_pos", FLOAT, 3), 0.05f);
    Node tap = nodes::select(fs_in("output_params", FLOAT, 1)
        .greater_than(nodes::num(0.5f)), nodes::min(c.xyz(), nodes::vec3(64)), toned);
    Node gather = nodes::loop(j, nodes::int_(0), nodes::int_(8), nodes::block(sum.assign(sum + tap)));
    Node seen = nodes::if_then(
        hit.greater_than(nodes::num(0)),
        nodes::block(sum, gather, node_return(nodes::vec4(sum / 8.0f, found)))
    );
    Node met    = nodes::if_then(behind, nodes::block(lo, hi, halve, back, read_back, seen));
    // longer steps further out, but never more than a few pixels on the screen: a step longer
    // than something is thick would go through it without seeing it
    Node longer = step_len * 1.2f;
    Node next   = ssr_project(origin + r * (t + step_len));
    Node px     = nodes::length((next.xy() - s.xy()) * screen_px);
    Node shorten = nodes::if_then(nodes::logical_and(next.z()
        .greater_than(
            nodes::num(0.05f)
        ), px.greater_than(max_step)), nodes::block(step_len.assign(step_len * (max_step / px))));
    Node body = nodes::block(
        nodes::if_then(i.greater_than_equal(steps), nodes::block(nodes::break_())),
        nodes::if_then(outside, nodes::block(nodes::break_())),
        met,
        t_prev.assign(t),
        step_len.assign(longer),
        shorten,
        t.assign(t + step_len),
        nodes::if_then(t.greater_than(thickness.y()), nodes::block(nodes::break_()))
    );
    Node march = nodes::loop(i, nodes::int_(0), nodes::int_(SSR_MAX_STEPS), body);
    return nodes::seq(nodes::block(away, step_len, t, t_prev, march), nodes::vec4(0));
}

// ---- Volumetric fog and light shafts

// Volumetric fog in front of world position `p`: rgb = in-scattered light toward the camera,
// a = transmittance (matching shaders/shader_core.glsl's fog_volume_at). Interpolates between
// the two froxel depth slices straddling `p` (the camera plane starts at transmittance 1).
fn Node fog_volume_at(Node p) @private {
    Node clip = fs_in("cluster_view_proj", MAT4) * nodes::vec4(p, 1);
    Node uv = nodes::clamp(
        clip.xy() / nodes::max(nodes::abs(clip.w()), 1e-6f) * 0.5f + 0.5f,
        nodes::vec2(0),
        nodes::vec2(1)
    );
    Node depth  = sun_depth(p);
    Node grid   = fs_in("fog_grid");
    Node sl     = fs_in("fog_slices");
    Node slices = grid.z();
    Node s = nodes::select(
        depth.less_than_equal(sl.x()),
        -1.0f,
        nodes::clamp(nodes::log(depth / sl.x()) * sl.y() - 1.0f, nodes::num(-1), slices - 1.0f)
    );
    Node k = nodes::floor(s);
    Node f = s - k;
    Node size = nodes::convert(
        node_sample(SIZE, shaders::VIEW_FOG_VOLUME, shaders::SMP_SSR_SMP, TEX_2D, nodes::int_(0)),
        VEC2
    );
    Node cell = nodes::clamp(uv * grid.xy(), nodes::vec2(0.5f), grid.xy() - 0.5f);
    Node a = nodes::select(
        k.greater_than_equal(nodes::num(0)),
        fog_slice(k, grid, cell, size),
        nodes::vec4(0, 0, 0, 1)
    );
    Node b = fog_slice(nodes::min(k + 1.0f, slices - 1.0f), grid, cell, size);
    return nodes::mix(a, b, f);
}

// Slice `k` of the froxel grid at `cell`: the slices are laid out grid.w to a row.
fn Node fog_slice(Node k, Node grid, Node cell, Node size) @local {
    Node per_row = grid.w();
    Node at      = nodes::vec2(nodes::mod(k, per_row), nodes::floor(k / per_row)) * grid.xy() + cell;
    return node_sample(LEVEL, shaders::VIEW_FOG_VOLUME, shaders::SMP_SSR_SMP, TEX_2D, at / size, nodes::num(0));
}

// Light scattered towards the camera by the fog along the view ray `ray` (`depth` long), for a
// light at `lp`: the integral of cone * transmittance / distance^2 along the ray. Substituting
// s = b + h tan(a) turns the 1 / distance^2 kernel into a constant, so a few samples in `a`
// are enough. `cone` is (spot direction, cos outer), `cos_inner` < -1 for a point light. (The
// function fog_glow of the program, made once: `fn_id` -1 makes it.)
fn Node fog_glow(
    int* fn_id,
    MaterialKey key,
    Node lp,
    Node radius,
    Node ray,
    Node depth,
    Node cone,
    Node cos_inner
) @private {
    if (*fn_id < 0) {
        NodeType[6] types = { VEC3, FLOAT, VEC3, FLOAT, VEC4, FLOAT };
        int fid = node_func("fog_glow", FLOAT, &types);
        Node pl = node_param(fid, 0);
        Node pr = node_param(fid, 1);
        Node pray = node_param(fid, 2);
        Node pd = node_param(fid, 3);
        Node pc = node_param(fid, 4);
        Node pi = node_param(fid, 5);
        Node to_light = pl - fs_in("camera_pos", VEC3);
        Node b = nodes::dot(to_light, pray);
        Node h = nodes::max(nodes::sqrt(nodes::max(nodes::dot(to_light, to_light) - b * b, 0)), nodes::max(pr, 0.05f));
        Node a0 = nodes::atan(b.negate() / h);
        Node a1 = nodes::atan((pd - b) / h);
        Node sum = nodes::variable(nodes::num(0));
        Node i = nodes::loop_index(INT);
        Node a = nodes::mix(a0, a1, (nodes::convert(i, FLOAT) + 0.5f) / 6.0f);
        Node s = b + h * nodes::tan(a);
        Node from_light = pray * s - to_light;
        Node d = nodes::max(nodes::length(from_light), 1e-4f);
        Node w = shading::fog_transmittance(key, s + d);
        w = nodes::select(
            pi.greater_than_equal(nodes::num(-1)),
            w * nodes::smoothstep(pc.w(), pi, nodes::dot(from_light / d, pc.xyz())),
            w
        );
        Node samples = nodes::loop(i, nodes::int_(0), nodes::int_(6), nodes::block(sum.assign(sum + w)));
        node_func_body(fid, nodes::seq(nodes::block(sum, samples), sum / 6.0f * (a1 - a0) / h));
        *fn_id = fid;
    }
    Node[6] args = { lp, radius, ray, depth, cone, cos_inner };
    return node_call_fn(*fn_id, &args);
}

// The glow in the fog of all of the render's point and spot lights along the view ray `ray`
// (`depth` long): all of them, not the fragment's cluster's, the glow being along the whole
// ray.
fn Node fog_lights_glow(MaterialKey key, Node ray, Node depth) @private {
    int fn_id   = -1;
    Node counts =  fs_in("counts");
    Node glow   =  nodes::variable(nodes::vec3(0));
    Node i      =  nodes::loop_index(UINT);
    Node at     =  i * nodes::uint_(4);
    Node pos    =  node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, at);
    Node color  =  node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, at + nodes::uint_(1));
    Node params =  node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, at + nodes::uint_(3));
    Node point  =  color.xyz() * fog_glow(
        &fn_id,
        key,
        pos.xyz(),
        params.x(),
        ray,
        depth,
        nodes::vec4(0),
        nodes::num(-2)
    );
    Node points = nodes::loop(
        i,
        nodes::uint_(0),
        nodes::convert(counts.y(), UINT),
        nodes::block(glow.assign(glow + point))
    );
    Node j       = nodes::loop_index(UINT);
    Node sat     = j * nodes::uint_(5);
    Node spos    = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, sat);
    Node dir     = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, sat + nodes::uint_(1));
    Node scolor  = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, sat + nodes::uint_(2));
    Node sparams = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, sat + nodes::uint_(3));
    Node spot    = scolor.xyz() * sparams.w() * fog_glow(
        &fn_id,
        key,
        spos.xyz(),
        sparams.y(),
        ray,
        depth,
        dir,
        sparams.x()
    );
    Node spots = nodes::loop(
        j,
        nodes::uint_(0),
        nodes::convert(counts.z(), UINT),
        nodes::block(nodes::if_then(sparams.w()
        .less_than_equal(nodes::num(0)), nodes::block(nodes::continue_())), glow.assign(glow + spot)));
    return nodes::seq(nodes::block(glow, points, spots), glow);
}

// Sunlit fraction (0..1) of the fog from `from` to `to` (shaders/shader_core.glsl's sun_in_fog):
// marches 14 jittered steps through the sun shadow map, weighting each step by its fog scattering
// contribution (`sigma`: fog density along the ray). Past the map counts as lit.
fn Node sun_in_fog(ShadowFns* shadow, Node from, Node to, Node sigma, Node frag) @private {
    const int STEPS = 14;
    Node ray        = to - from;
    Node len        = nodes::length(ray);
    Node span       = nodes::min(len, 70.0f);
    Node noise      = nodes::fract(nodes::fract(nodes::dot(frag, nodes::vec2(0.06711056f, 0.00583715f))) * 52.9829189f);
    Node jitter     = nodes::fract(noise + fs_in("ssr_params", FLOAT, 2) * 0.618f);
    Node lit        = nodes::variable(nodes::num(0));
    Node last       = nodes::variable(nodes::num(1));
    Node i          = nodes::loop_index(INT);
    Node fi         = nodes::convert(i, FLOAT);
    Node s1         = span * (fi + 1.0f) / (float)STEPS;
    Node s          = span * (fi + jitter) / (float)STEPS;
    Node next       = nodes::exp((sigma * s1).negate());
    Node at         = from + ray * (s / nodes::max(len, 1e-4f));
    Node k          = nodes::variable(sun_cascade(sun_depth(at)));
    Node c          = (fs_at("dir_shadow_mat", MAT4, nodes::max(k, nodes::int_(0))) * nodes::vec4(at, 1)).xyz();
    Node seen       = nodes::variable(nodes::num(1));
    Node tap        = shadow.sun_tap(k, c.xy(), c.z());
    // (a hard test against the mean, as the PCF map's one tap)
    if (shadow.key.vsm) tap = nodes::select(tap.less_than(nodes::num(1)), 0.0f, 1.0f);
    Node edge = nodes::min(c.xy(), nodes::vec2(1) - c.xy());
    Node inside = nodes::logical_and(nodes::logical_and(k.greater_than_equal(nodes::int_(0)), nodes::logical_and(c.x()
        .greater_than(nodes::num(0)), c.x()
            .less_than(nodes::num(1)))), nodes::logical_and(nodes::logical_and(c.y()
                .greater_than(nodes::num(0)), c.y()
                    .less_than(nodes::num(1))), nodes::logical_and(c.z()
                        .less_than(nodes::num(1)), c.z().greater_than(nodes::num(0)))));
    Node look = nodes::if_then(
        inside,
        nodes::block(
            seen.assign(nodes::mix(nodes::num(1), tap, nodes::clamp(nodes::min(edge.x(), edge.y()) * 10.0f, 0, 1)))
        )
    );
    Node body    = nodes::block(k, seen, look, lit.assign(lit + (last - next) * seen), last.assign(next));
    Node march   = nodes::loop(i, nodes::int_(0), nodes::int_(STEPS), body);
    Node all_fog = nodes::exp((sigma * len).negate()).one_minus();
    Node through = lit + last - all_fog.one_minus();
    Node shafts = nodes::select(
        all_fog.greater_than(nodes::num(1e-5f)),
        nodes::mix(nodes::num(1), nodes::clamp(through / all_fog, 0, 1), fs_in("dir_cascade_fwd", FLOAT, 3)),
        1.0f
    );
    return nodes::seq(nodes::block(lit, last, march), shafts);
}