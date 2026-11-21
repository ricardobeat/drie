// Native MaterialX noise builders on top of drie::nodes (matching three.tsl.js's mx_* noise
// functions: Bob Jenkins hash, 2D/3D Perlin, Cell, Worley, and Fractal noise).
// Pure Node -> Node builders with no JS dependency, reusable from any C3 caller or walker.
module main;
import boomkat, drie, drie::nodes;

fn Node mx_uvec3(Node a, Node b, Node c) @local
{
    Node[3] parts = { a, b, c };
    return drie::node_join(&parts, UVEC3);
}

fn Node mx_rotl32(Node x, int k) @local
{
    return nodes::bit_or(
        nodes::shift_left(x, nodes::int_(k)),
        nodes::shift_right(x, nodes::int_(32 - k))
    );
}

fn Node mx_bjmix(Node a_in, Node b_in, Node c_in) @local
{
    Node a = nodes::variable(nodes::convert(a_in, UINT));
    Node b = nodes::variable(nodes::convert(b_in, UINT));
    Node c = nodes::variable(nodes::convert(c_in, UINT));
    Node body = nodes::block(
        a, b, c,
        a.assign(a - c), a.assign(nodes::bit_xor(a, mx_rotl32(c, 4))), c.assign(c + b),
        b.assign(b - a), b.assign(nodes::bit_xor(b, mx_rotl32(a, 6))), a.assign(a + c),
        c.assign(c - b), c.assign(nodes::bit_xor(c, mx_rotl32(b, 8))), b.assign(b + a),
        a.assign(a - c), a.assign(nodes::bit_xor(a, mx_rotl32(c, 16))), c.assign(c + b),
        b.assign(b - a), b.assign(nodes::bit_xor(b, mx_rotl32(a, 19))), a.assign(a + c),
        c.assign(c - b), c.assign(nodes::bit_xor(c, mx_rotl32(b, 4))), b.assign(b + a)
    );
    return nodes::seq(body, mx_uvec3(a, b, c));
}

fn Node mx_bjfinal(Node a_in, Node b_in, Node c_in) @local
{
    Node a = nodes::variable(nodes::convert(a_in, UINT));
    Node b = nodes::variable(nodes::convert(b_in, UINT));
    Node c = nodes::variable(nodes::convert(c_in, UINT));
    Node body = nodes::block(
        a, b, c,
        c.assign(nodes::bit_xor(c, b)), c.assign(c - mx_rotl32(b, 14)),
        a.assign(nodes::bit_xor(a, c)), a.assign(a - mx_rotl32(c, 11)),
        b.assign(nodes::bit_xor(b, a)), b.assign(b - mx_rotl32(a, 25)),
        c.assign(nodes::bit_xor(c, b)), c.assign(c - mx_rotl32(b, 16)),
        a.assign(nodes::bit_xor(a, c)), a.assign(a - mx_rotl32(c, 4)),
        b.assign(nodes::bit_xor(b, a)), b.assign(b - mx_rotl32(a, 14)),
        c.assign(nodes::bit_xor(c, b)), c.assign(c - mx_rotl32(b, 24))
    );
    return nodes::seq(body, c);
}

fn Node mx_hash_int1(Node x) @local
{
    uint seed = 0xdeadbeefu + (1u << 2) + 13u;
    Node s = nodes::uint_(seed);
    return mx_bjfinal(s + nodes::convert(x, UINT), s, s);
}

fn Node mx_hash_int2(Node x, Node y) @local
{
    uint seed = 0xdeadbeefu + (2u << 2) + 13u;
    Node s = nodes::uint_(seed);
    return mx_bjfinal(s + nodes::convert(x, UINT), s + nodes::convert(y, UINT), s);
}

fn Node mx_hash_int3(Node x, Node y, Node z) @local
{
    uint seed = 0xdeadbeefu + (3u << 2) + 13u;
    Node s = nodes::uint_(seed);
    return mx_bjfinal(s + nodes::convert(x, UINT), s + nodes::convert(y, UINT), s + nodes::convert(z, UINT));
}

fn Node mx_hash_int4(Node x, Node y, Node z, Node w) @local
{
    uint seed = 0xdeadbeefu + (4u << 2) + 13u;
    Node s = nodes::uint_(seed);
    Node m = nodes::variable(mx_bjmix(s + nodes::convert(x, UINT), s + nodes::convert(y, UINT), s + nodes::convert(z, UINT)));
    return nodes::seq(nodes::block(m), mx_bjfinal(m.x() + nodes::convert(w, UINT), m.y(), m.z()));
}

fn Node mx_hash_vec3_2d(Node x, Node y) @local
{
    Node h = nodes::variable(mx_hash_int2(x, y));
    Node mask = nodes::uint_(0xffu);
    Node res = mx_uvec3(
        nodes::bit_and(h, mask),
        nodes::bit_and(nodes::shift_right(h, nodes::int_(8)), mask),
        nodes::bit_and(nodes::shift_right(h, nodes::int_(16)), mask)
    );
    return nodes::seq(nodes::block(h), res);
}

fn Node mx_hash_vec3_3d(Node x, Node y, Node z) @local
{
    Node h = nodes::variable(mx_hash_int3(x, y, z));
    Node mask = nodes::uint_(0xffu);
    Node res = mx_uvec3(
        nodes::bit_and(h, mask),
        nodes::bit_and(nodes::shift_right(h, nodes::int_(8)), mask),
        nodes::bit_and(nodes::shift_right(h, nodes::int_(16)), mask)
    );
    return nodes::seq(nodes::block(h), res);
}

fn Node mx_bits_to_01(Node bits) @local
{
    return nodes::convert(bits, FLOAT) / nodes::num((float)0xffffffffu);
}

fn Node mx_fade(Node t) @local
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

fn Node mx_negate_if(Node val, Node cond) @local
{
    return nodes::select(cond, val.negate(), val);
}

fn Node mx_grad_f2(Node hash, Node x, Node y) @local
{
    Node h = nodes::variable(nodes::bit_and(hash, nodes::uint_(7)));
    Node lt4 = h.less_than(nodes::uint_(4));
    Node u = nodes::select(lt4, x, y);
    Node v = nodes::select(lt4, y, x) * 2.0f;
    Node b1 = nodes::convert(nodes::bit_and(h, nodes::uint_(1)), BOOL);
    Node b2 = nodes::convert(nodes::bit_and(h, nodes::uint_(2)), BOOL);
    return nodes::seq(nodes::block(h), mx_negate_if(u, b1) + mx_negate_if(v, b2));
}

fn Node mx_grad_f3(Node hash, Node x, Node y, Node z) @local
{
    Node h = nodes::variable(nodes::bit_and(hash, nodes::uint_(15)));
    Node u = nodes::select(h.less_than(nodes::uint_(8)), x, y);
    Node is12or14 = nodes::logical_or(h.equal(nodes::uint_(12)), h.equal(nodes::uint_(14)));
    Node v = nodes::select(h.less_than(nodes::uint_(4)), y, nodes::select(is12or14, x, z));
    Node b1 = nodes::convert(nodes::bit_and(h, nodes::uint_(1)), BOOL);
    Node b2 = nodes::convert(nodes::bit_and(h, nodes::uint_(2)), BOOL);
    return nodes::seq(nodes::block(h), mx_negate_if(u, b1) + mx_negate_if(v, b2));
}

fn Node mx_grad_v2(Node hash, Node x, Node y) @local
{
    Node h = nodes::variable(hash);
    return nodes::seq(nodes::block(h), nodes::vec3(mx_grad_f2(h.x(), x, y), mx_grad_f2(h.y(), x, y), mx_grad_f2(h.z(), x, y)));
}

fn Node mx_grad_v3(Node hash, Node x, Node y, Node z) @local
{
    Node h = nodes::variable(hash);
    return nodes::seq(nodes::block(h), nodes::vec3(mx_grad_f3(h.x(), x, y, z), mx_grad_f3(h.y(), x, y, z), mx_grad_f3(h.z(), x, y, z)));
}

fn Node mx_bilerp(Node v0, Node v1, Node v2, Node v3, Node s, Node t) @local
{
    Node s1 = 1.0f - s;
    Node t1 = 1.0f - t;
    return t1 * (v0 * s1 + v1 * s) + t * (v2 * s1 + v3 * s);
}

fn Node mx_trilerp(Node v0, Node v1, Node v2, Node v3, Node v4, Node v5, Node v6, Node v7, Node s, Node t, Node r) @local
{
    Node s1 = 1.0f - s;
    Node t1 = 1.0f - t;
    Node r1 = 1.0f - r;
    Node lo = t1 * (v0 * s1 + v1 * s) + t * (v2 * s1 + v3 * s);
    Node hi = t1 * (v4 * s1 + v5 * s) + t * (v6 * s1 + v7 * s);
    return r1 * lo + r * hi;
}

// ---- Public MaterialX noise builders

fn Node mx_perlin_noise_float(Node p_in)
{
    if (p_in.type() == VEC2)
    {
        Node p = nodes::variable(p_in);
        Node ix = nodes::variable(nodes::convert(nodes::floor(p.x()), INT));
        Node iy = nodes::variable(nodes::convert(nodes::floor(p.y()), INT));
        Node fx = nodes::variable(p.x() - nodes::convert(ix, FLOAT));
        Node fy = nodes::variable(p.y() - nodes::convert(iy, FLOAT));
        Node u = nodes::variable(mx_fade(fx));
        Node v = nodes::variable(mx_fade(fy));
        Node ix1 = ix + nodes::int_(1);
        Node iy1 = iy + nodes::int_(1);
        Node fx1 = fx - 1.0f;
        Node fy1 = fy - 1.0f;
        Node res = mx_bilerp(
            mx_grad_f2(mx_hash_int2(ix, iy), fx, fy),
            mx_grad_f2(mx_hash_int2(ix1, iy), fx1, fy),
            mx_grad_f2(mx_hash_int2(ix, iy1), fx, fy1),
            mx_grad_f2(mx_hash_int2(ix1, iy1), fx1, fy1),
            u, v
        ) * 0.6616f;
        return nodes::seq(nodes::block(p, ix, iy, fx, fy, u, v), res);
    }
    Node p = nodes::variable(nodes::convert(p_in, VEC3));
    Node ix = nodes::variable(nodes::convert(nodes::floor(p.x()), INT));
    Node iy = nodes::variable(nodes::convert(nodes::floor(p.y()), INT));
    Node iz = nodes::variable(nodes::convert(nodes::floor(p.z()), INT));
    Node fx = nodes::variable(p.x() - nodes::convert(ix, FLOAT));
    Node fy = nodes::variable(p.y() - nodes::convert(iy, FLOAT));
    Node fz = nodes::variable(p.z() - nodes::convert(iz, FLOAT));
    Node u = nodes::variable(mx_fade(fx));
    Node v = nodes::variable(mx_fade(fy));
    Node w = nodes::variable(mx_fade(fz));
    Node ix1 = ix + nodes::int_(1);
    Node iy1 = iy + nodes::int_(1);
    Node iz1 = iz + nodes::int_(1);
    Node fx1 = fx - 1.0f;
    Node fy1 = fy - 1.0f;
    Node fz1 = fz - 1.0f;
    Node res = mx_trilerp(
        mx_grad_f3(mx_hash_int3(ix, iy, iz), fx, fy, fz),
        mx_grad_f3(mx_hash_int3(ix1, iy, iz), fx1, fy, fz),
        mx_grad_f3(mx_hash_int3(ix, iy1, iz), fx, fy1, fz),
        mx_grad_f3(mx_hash_int3(ix1, iy1, iz), fx1, fy1, fz),
        mx_grad_f3(mx_hash_int3(ix, iy, iz1), fx, fy, fz1),
        mx_grad_f3(mx_hash_int3(ix1, iy, iz1), fx1, fy, fz1),
        mx_grad_f3(mx_hash_int3(ix, iy1, iz1), fx, fy1, fz1),
        mx_grad_f3(mx_hash_int3(ix1, iy1, iz1), fx1, fy1, fz1),
        u, v, w
    ) * 0.9820f;
    return nodes::seq(nodes::block(p, ix, iy, iz, fx, fy, fz, u, v, w), res);
}

fn Node mx_perlin_noise_vec3(Node p_in)
{
    if (p_in.type() == VEC2)
    {
        Node p = nodes::variable(p_in);
        Node ix = nodes::variable(nodes::convert(nodes::floor(p.x()), INT));
        Node iy = nodes::variable(nodes::convert(nodes::floor(p.y()), INT));
        Node fx = nodes::variable(p.x() - nodes::convert(ix, FLOAT));
        Node fy = nodes::variable(p.y() - nodes::convert(iy, FLOAT));
        Node u = nodes::variable(mx_fade(fx));
        Node v = nodes::variable(mx_fade(fy));
        Node ix1 = ix + nodes::int_(1);
        Node iy1 = iy + nodes::int_(1);
        Node fx1 = fx - 1.0f;
        Node fy1 = fy - 1.0f;
        Node res = mx_bilerp(
            mx_grad_v2(mx_hash_vec3_2d(ix, iy), fx, fy),
            mx_grad_v2(mx_hash_vec3_2d(ix1, iy), fx1, fy),
            mx_grad_v2(mx_hash_vec3_2d(ix, iy1), fx, fy1),
            mx_grad_v2(mx_hash_vec3_2d(ix1, iy1), fx1, fy1),
            u, v
        ) * 0.6616f;
        return nodes::seq(nodes::block(p, ix, iy, fx, fy, u, v), res);
    }
    Node p = nodes::variable(nodes::convert(p_in, VEC3));
    Node ix = nodes::variable(nodes::convert(nodes::floor(p.x()), INT));
    Node iy = nodes::variable(nodes::convert(nodes::floor(p.y()), INT));
    Node iz = nodes::variable(nodes::convert(nodes::floor(p.z()), INT));
    Node fx = nodes::variable(p.x() - nodes::convert(ix, FLOAT));
    Node fy = nodes::variable(p.y() - nodes::convert(iy, FLOAT));
    Node fz = nodes::variable(p.z() - nodes::convert(iz, FLOAT));
    Node u = nodes::variable(mx_fade(fx));
    Node v = nodes::variable(mx_fade(fy));
    Node w = nodes::variable(mx_fade(fz));
    Node ix1 = ix + nodes::int_(1);
    Node iy1 = iy + nodes::int_(1);
    Node iz1 = iz + nodes::int_(1);
    Node fx1 = fx - 1.0f;
    Node fy1 = fy - 1.0f;
    Node fz1 = fz - 1.0f;
    Node res = mx_trilerp(
        mx_grad_v3(mx_hash_vec3_3d(ix, iy, iz), fx, fy, fz),
        mx_grad_v3(mx_hash_vec3_3d(ix1, iy, iz), fx1, fy, fz),
        mx_grad_v3(mx_hash_vec3_3d(ix, iy1, iz), fx, fy1, fz),
        mx_grad_v3(mx_hash_vec3_3d(ix1, iy1, iz), fx1, fy1, fz),
        mx_grad_v3(mx_hash_vec3_3d(ix, iy, iz1), fx, fy, fz1),
        mx_grad_v3(mx_hash_vec3_3d(ix1, iy, iz1), fx1, fy, fz1),
        mx_grad_v3(mx_hash_vec3_3d(ix, iy1, iz1), fx, fy1, fz1),
        mx_grad_v3(mx_hash_vec3_3d(ix1, iy1, iz1), fx1, fy1, fz1),
        u, v, w
    ) * 0.9820f;
    return nodes::seq(nodes::block(p, ix, iy, iz, fx, fy, fz, u, v, w), res);
}

fn Node mx_cell_noise_float(Node p_in)
{
    Node p = nodes::variable(p_in);
    switch (p_in.type())
    {
        case FLOAT:
            return nodes::seq(nodes::block(p), mx_bits_to_01(mx_hash_int1(nodes::convert(nodes::floor(p), INT))));
        case VEC2:
            return nodes::seq(nodes::block(p), mx_bits_to_01(mx_hash_int2(
                nodes::convert(nodes::floor(p.x()), INT),
                nodes::convert(nodes::floor(p.y()), INT)
            )));
        case VEC4:
            return nodes::seq(nodes::block(p), mx_bits_to_01(mx_hash_int4(
                nodes::convert(nodes::floor(p.x()), INT),
                nodes::convert(nodes::floor(p.y()), INT),
                nodes::convert(nodes::floor(p.z()), INT),
                nodes::convert(nodes::floor(p.w()), INT)
            )));
        default:
            Node p3 = nodes::variable(nodes::convert(p_in, VEC3));
            return nodes::seq(nodes::block(p3), mx_bits_to_01(mx_hash_int3(
                nodes::convert(nodes::floor(p3.x()), INT),
                nodes::convert(nodes::floor(p3.y()), INT),
                nodes::convert(nodes::floor(p3.z()), INT)
            )));
    }
}

fn Node mx_cell_noise_vec3_3d_int(Node ix, Node iy, Node iz) @local
{
    uint seed = 0xdeadbeefu + (4u << 2) + 13u;
    Node s = nodes::uint_(seed);
    Node m = nodes::variable(mx_bjmix(s + nodes::convert(ix, UINT), s + nodes::convert(iy, UINT), s + nodes::convert(iz, UINT)));
    Node res = nodes::vec3(
        mx_bits_to_01(mx_bjfinal(m.x(), m.y(), m.z())),
        mx_bits_to_01(mx_bjfinal(m.x() + nodes::uint_(1), m.y(), m.z())),
        mx_bits_to_01(mx_bjfinal(m.x() + nodes::uint_(2), m.y(), m.z()))
    );
    return nodes::seq(nodes::block(m), res);
}

fn Node mx_worley_noise_float_2d(Node p_in, Node jitter_in, Node style_in)
{
    Node p = nodes::variable(nodes::convert(p_in, VEC2));
    Node jitter = nodes::variable(nodes::convert(jitter_in, FLOAT));
    Node style = nodes::variable(nodes::convert(style_in, INT));
    Node fx = nodes::variable(nodes::convert(nodes::floor(p.x()), INT));
    Node fy = nodes::variable(nodes::convert(nodes::floor(p.y()), INT));
    Node localpos = nodes::variable(nodes::vec2(nodes::fract(p.x()), nodes::fract(p.y())));
    Node sqdist = nodes::variable(nodes::num(1e6f));
    Node minpos = nodes::variable(nodes::vec2(0.0f, 0.0f));

    Node x = nodes::loop_index();
    Node y = nodes::loop_index();
    Node sx = fx + x;
    Node sy = fy + y;
    Node off = nodes::variable(nodes::vec2(
        mx_bits_to_01(mx_hash_int3(sx, sy, nodes::int_(0))),
        mx_bits_to_01(mx_hash_int3(sx, sy, nodes::int_(1)))
    ));
    Node cellpos = nodes::variable(
        nodes::vec2(nodes::convert(x, FLOAT), nodes::convert(y, FLOAT)) + (off - 0.5f) * jitter + 0.5f - localpos
    );
    Node dist = nodes::variable(nodes::dot(cellpos, cellpos));
    Node inner = nodes::block(
        off, cellpos, dist,
        nodes::if_then(dist.less_than(sqdist), nodes::block(
            sqdist.assign(dist),
            minpos.assign(cellpos)
        ))
    );
    Node body = nodes::block(
        p, jitter, style, fx, fy, localpos, sqdist, minpos,
        nodes::loop(x, nodes::int_(-1), nodes::int_(1), nodes::block(
            nodes::loop(y, nodes::int_(-1), nodes::int_(1), inner, cond: LE)
        ), cond: LE),
        nodes::if_then(
            style.equal(nodes::int_(1)),
            nodes::block(sqdist.assign(mx_cell_noise_float(minpos + p))),
            nodes::block(sqdist.assign(nodes::sqrt(sqdist)))
        )
    );
    return nodes::seq(body, sqdist);
}

fn Node mx_worley_noise_float_3d(Node p_in, Node jitter_in, Node style_in)
{
    Node p = nodes::variable(nodes::convert(p_in, VEC3));
    Node jitter = nodes::variable(nodes::convert(jitter_in, FLOAT));
    Node style = nodes::variable(nodes::convert(style_in, INT));
    Node ix = nodes::variable(nodes::convert(nodes::floor(p.x()), INT));
    Node iy = nodes::variable(nodes::convert(nodes::floor(p.y()), INT));
    Node iz = nodes::variable(nodes::convert(nodes::floor(p.z()), INT));
    Node localpos = nodes::variable(nodes::vec3(
        p.x() - nodes::convert(ix, FLOAT),
        p.y() - nodes::convert(iy, FLOAT),
        p.z() - nodes::convert(iz, FLOAT)
    ));
    Node sqdist = nodes::variable(nodes::num(1e6f));
    Node minpos = nodes::variable(nodes::vec3(0.0f, 0.0f, 0.0f));

    Node x = nodes::loop_index();
    Node y = nodes::loop_index();
    Node z = nodes::loop_index();
    Node off = nodes::variable(mx_cell_noise_vec3_3d_int(ix + x, iy + y, iz + z));
    Node cellpos = nodes::variable(
        nodes::vec3(nodes::convert(x, FLOAT), nodes::convert(y, FLOAT), nodes::convert(z, FLOAT))
        + (off - 0.5f) * jitter + 0.5f - localpos
    );
    Node dist = nodes::variable(nodes::dot(cellpos, cellpos));
    Node inner = nodes::block(
        off, cellpos, dist,
        nodes::if_then(dist.less_than(sqdist), nodes::block(
            sqdist.assign(dist),
            minpos.assign(cellpos)
        ))
    );
    Node body = nodes::block(
        p, jitter, style, ix, iy, iz, localpos, sqdist, minpos,
        nodes::loop(x, nodes::int_(-1), nodes::int_(1), nodes::block(
            nodes::loop(y, nodes::int_(-1), nodes::int_(1), nodes::block(
                nodes::loop(z, nodes::int_(-1), nodes::int_(1), inner, cond: LE)
            ), cond: LE)
        ), cond: LE),
        nodes::if_then(
            style.equal(nodes::int_(1)),
            nodes::block(sqdist.assign(mx_cell_noise_float(minpos + p))),
            nodes::block(sqdist.assign(nodes::sqrt(sqdist)))
        )
    );
    return nodes::seq(body, sqdist);
}

fn Node mx_fractal_noise_float(Node p_in, Node octaves_in, Node lacunarity_in, Node diminish_in)
{
    NodeType pt = p_in.type() == VEC2 ? VEC2 : VEC3;
    Node p = nodes::variable(nodes::convert(p_in, pt));
    Node octaves = nodes::variable(nodes::convert(octaves_in, INT));
    Node lacunarity = nodes::variable(nodes::convert(lacunarity_in, FLOAT));
    Node diminish = nodes::variable(nodes::convert(diminish_in, FLOAT));
    Node result = nodes::variable(nodes::num(0.0f));
    Node amp = nodes::variable(nodes::num(1.0f));
    Node i = nodes::loop_index();
    Node body = nodes::block(
        p, octaves, lacunarity, diminish, result, amp,
        nodes::loop(i, nodes::int_(0), octaves, nodes::block(
            result.assign(result + amp * mx_perlin_noise_float(p)),
            amp.assign(amp * diminish),
            p.assign(p * lacunarity)
        ))
    );
    return nodes::seq(body, result);
}

fn Node mx_fractal_noise_vec3(Node p_in, Node octaves_in, Node lacunarity_in, Node diminish_in)
{
    Node p = nodes::variable(nodes::convert(p_in, VEC3));
    Node octaves = nodes::variable(nodes::convert(octaves_in, INT));
    Node lacunarity = nodes::variable(nodes::convert(lacunarity_in, FLOAT));
    Node diminish = nodes::variable(nodes::convert(diminish_in, FLOAT));
    Node result = nodes::variable(nodes::vec3(0.0f, 0.0f, 0.0f));
    Node amp = nodes::variable(nodes::num(1.0f));
    Node i = nodes::loop_index();
    Node body = nodes::block(
        p, octaves, lacunarity, diminish, result, amp,
        nodes::loop(i, nodes::int_(0), octaves, nodes::block(
            result.assign(result + amp * mx_perlin_noise_vec3(p)),
            amp.assign(amp * diminish),
            p.assign(p * lacunarity)
        ))
    );
    return nodes::seq(body, result);
}

// ---- JS host bindings (thin wrappers for js/drie_nodes.js FUNCTIONS)

fn void h_mx_perlin_float(JsCtx ctx, void* udata) => ctx.ret_number((double)mx_perlin_noise_float(node_arg(ctx, 0)).id);
fn void h_mx_perlin_vec3(JsCtx ctx, void* udata) => ctx.ret_number((double)mx_perlin_noise_vec3(node_arg(ctx, 0)).id);
fn void h_mx_cell_float(JsCtx ctx, void* udata) => ctx.ret_number((double)mx_cell_noise_float(node_arg(ctx, 0)).id);
fn void h_mx_worley_2d(JsCtx ctx, void* udata) => ctx.ret_number((double)mx_worley_noise_float_2d(node_arg(ctx, 0), node_arg(ctx, 1), node_arg(ctx, 2)).id);
fn void h_mx_worley_3d(JsCtx ctx, void* udata) => ctx.ret_number((double)mx_worley_noise_float_3d(node_arg(ctx, 0), node_arg(ctx, 1), node_arg(ctx, 2)).id);
fn void h_mx_fractal_float(JsCtx ctx, void* udata) => ctx.ret_number((double)mx_fractal_noise_float(node_arg(ctx, 0), node_arg(ctx, 1), node_arg(ctx, 2), node_arg(ctx, 3)).id);
fn void h_mx_fractal_vec3(JsCtx ctx, void* udata) => ctx.ret_number((double)mx_fractal_noise_vec3(node_arg(ctx, 0), node_arg(ctx, 1), node_arg(ctx, 2), node_arg(ctx, 3)).id);
