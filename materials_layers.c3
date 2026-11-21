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
