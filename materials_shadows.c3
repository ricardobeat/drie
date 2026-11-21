// The lit materials' shadows as node functions (shader_core.glsl's light_shadow_of,
// dir_shadow_in, dir_tile_shadow and the taps they read), built once per program and called
// from the light loops of materials_nodes.c3.
module drie;
import drie::nodes, drie::shading;

const int SHADOW_MAX_TAPS @local = 40;

// The shadow functions of the program being built (-1: not made yet).
struct ShadowFns @private {
    MaterialKey key;
    int tap_fn;         // shadow_tap(tiled, rect, texel, uv, z): a spot light's map, or a tile of the atlas
    int light_fn;       // light_shadow_of(light_pos, si, n, world, dpdx, dpdy, frag)
    int sun_in_fn;      // dir_shadow_in(i, n, l, world, frag): the sun in cascade i
    int dir_tile_fn;    // dir_tile_shadow(si, n, l, world, frag)
}

fn ShadowFns shadow_fns(MaterialKey key) @private => { .key = key, .tap_fn = -1, .light_fn = -1, .sun_in_fn = -1, .dir_tile_fn = -1 };

// Element `field` of shadow `si` in spot_shadows (a GpuSpotShadow: its matrix's four columns,
// params, rect, soft).
fn Node spot_shadow(Node si, int field) @local {
    return node_storage(shaders::VIEW_SPOT_SHADOWS, VEC4, si * nodes::int_(7) + nodes::int_(field));
}

fn Node spot_shadow_mat(Node si) @local {
    Node[4] cols = { spot_shadow(si, 0), spot_shadow(si, 1), spot_shadow(si, 2), spot_shadow(si, 3) };
    return node_matrix(&cols, MAT4);
}

fn Node shadow_sample_scale() @local {
    Node q = fs_in("quality_params", FLOAT);
    return nodes::select(q.greater_than(nodes::num(0)), q, 1.0f);
}

// The soft shadows' taps: `base` of them, times Renderer.shadow_sample_scale.
fn Node shadow_taps(float base) @local {
    return nodes::clamp(nodes::convert(shadow_sample_scale() * base + 0.5f, INT), nodes::int_(4), nodes::int_(SHADOW_MAX_TAPS));
}

// The rotation of the tap disk at the pixel (a random angle per pixel and frame).
fn Node[2] tap_rotation(Node frag) @local {
    Node seed = frag + fs_in("ssr_params", FLOAT, 2) * nodes::vec2(5, 3);
    Node angle = nodes::fract(nodes::sin(nodes::dot(seed, nodes::vec2(12.9898f, 78.233f))) * 43758.5453f) * 6.2831f;
    return { nodes::cos(angle), nodes::sin(angle) };
}

fn Node rotate(Node o, Node ca, Node sa) @local => nodes::vec2(o.x() * ca - o.y() * sa, o.x() * sa + o.y() * ca);

// three's VSMShadowMap: how lit depth `z` is where the map's depths have mean `m.x` and
// standard deviation `m.y` (Chebyshev's upper bound, its low end cut off against light bleeding).
fn Node vsm_lookup(Node m, Node z) @local {
    Node hard = nodes::step(z, m.x());
    Node variance = nodes::max(m.y() * m.y(), 1e-7f);
    Node d = z - m.x();
    Node p = variance / (variance + d * d);
    p = nodes::clamp((p - 0.3f) / 0.65f, 0, 1);
    return nodes::select(hard.equal(nodes::num(1)), 1.0f, nodes::max(hard, p));
}

// 1 where the light's map (or its tile of the atlas) has nothing nearer than depth `z` at `uv`.
fn Node ShadowFns.tap(&self, Node tiled, Node rect, Node texel, Node uv, Node z) {
    if (self.tap_fn < 0) {
        NodeType[5] params = { BOOL, VEC4, FLOAT, VEC2, FLOAT };
        int f = node_func("shadow_tap", FLOAT, &params);
        Node t = node_param(f, 0);
        Node r = node_param(f, 1);
        Node tx = node_param(f, 2);
        Node at = node_param(f, 3);
        Node d = node_param(f, 4);
        // stay half a texel inside the tile: the neighbours belong to other lights
        Node in_tile = r.xy() + nodes::clamp(at, nodes::vec2(tx), nodes::vec2(tx.one_minus())) * r.swizzle("zw");
        Node atlas;
        Node own;
        if (self.key.vsm) {
            // (with VSM: nearer than the mean depth)
            atlas = nodes::step(d, node_sample(LEVEL, shaders::VIEW_VSM_ATLAS, shaders::SMP_VSM_SMP, TEX_2D, in_tile, nodes::num(0)).x());
            own = nodes::step(d, node_sample(LEVEL, shaders::VIEW_VSM_MAP, shaders::SMP_VSM_SMP, TEX_2D, at, nodes::num(0)).x());
        } else {
            atlas = node_sample(COMPARE, shaders::VIEW_SHADOW_ATLAS, shaders::SMP_SHADOW_SMP, TEX_DEPTH, in_tile, d);
            own = node_sample(COMPARE, shaders::VIEW_SHADOW_MAP, shaders::SMP_SHADOW_SMP, TEX_DEPTH, at, d);
        }
        node_func_body(f, nodes::seq(nodes::block(nodes::if_then(t, nodes::block(node_return(atlas)))), own));
        self.tap_fn = f;
    }
    Node[5] args = { tiled, rect, texel, uv, z };
    return node_call_fn(self.tap_fn, &args);
}

// The moments of a spot light's VSM map (or its tile of the atlas) at `uv`.
fn Node vsm_moments(Node tiled, Node rect, Node texel, Node uv) @local {
    Node in_tile = rect.xy() + nodes::clamp(uv, nodes::vec2(texel), nodes::vec2(texel.one_minus())) * rect.swizzle("zw");
    Node atlas = node_sample(LEVEL, shaders::VIEW_VSM_ATLAS, shaders::SMP_VSM_SMP, TEX_2D, in_tile, nodes::num(0)).xy();
    Node own = node_sample(LEVEL, shaders::VIEW_VSM_MAP, shaders::SMP_VSM_SMP, TEX_2D, uv, nodes::num(0)).xy();
    return nodes::select(tiled, atlas, own);
}

// Distance from the light's plane of a depth in its shadow map, and back.
fn Node shadow_distance(Node z, Node soft) @local {
    Node gl = soft.w().less_than(nodes::num(0));
    Node zz = nodes::select(gl, z * 2.0f - 1.0f, z);
    Node y = soft.y();
    Node s = soft.z();
    return nodes::select(gl, 2.0f * y * s / (s + y - zz * (s - y)), y * s / (s - zz * (s - y)));
}

fn Node shadow_depth(Node d, Node soft) @local {
    Node y = soft.y();
    Node s = soft.z();
    return nodes::select(soft.w().less_than(nodes::num(0)), ((s + y) * d - 2.0f * y * s) / (d * (s - y)) * 0.5f + 0.5f,
                         s * (d - y) / (d * (s - y)));
}

// Shadow of the light at `light_pos` (a spot light, or one face of a point light's) whose map is
// `si` in spot_shadows: a spot light whose map is redrawn every frame has shadow_map to itself,
// the other maps share shadow_atlas, a tile each.
fn Node ShadowFns.light(&self, Node light_pos, Node si, Node n, Node world, Node dpdx, Node dpdy, Node frag) {
    if (self.light_fn < 0) self.light_fn = self.make_light();
    Node[7] args = { light_pos, si, n, world, dpdx, dpdy, frag };
    return node_call_fn(self.light_fn, &args);
}

fn int ShadowFns.make_light(&self) {
    NodeType[7] types = { VEC3, INT, VEC3, VEC3, VEC3, VEC3, VEC2 };
    int f = node_func("light_shadow_of", FLOAT, &types);
    Node light_pos = node_param(f, 0);
    Node si = node_param(f, 1);
    Node n = node_param(f, 2);
    Node world = node_param(f, 3);
    Node dpdx = node_param(f, 4);
    Node dpdy = node_param(f, 5);
    Node frag = node_param(f, 6);
    Node[24] st;
    int ns;

    Node params = spot_shadow(si, 4);
    Node rect = spot_shadow(si, 5);
    Node soft = spot_shadow(si, 6);
    Node to_light = light_pos - world;
    Node dist = nodes::length(to_light);
    // push the lookup off the surface, more at grazing angles and far away where texels are big
    Node slope = nodes::clamp(nodes::dot(n, to_light / dist), 0, 1).one_minus();
    Node p = world + n * params.y() * dist * (slope + 0.4f) - light_pos;
    // (the matrix takes the point from the light, see vs_params)
    Node m = spot_shadow_mat(si);
    Node sp = m * nodes::vec4(p, 1);
    st[ns++] = nodes::if_then(sp.w().less_than_equal(nodes::num(0)), nodes::block(node_return(nodes::num(1))));
    Node c = sp.xyz() / sp.w();
    Node outside = nodes::logical_or(nodes::logical_or(c.x().less_than(nodes::num(0)), c.x().greater_than(nodes::num(1))),
                                     nodes::logical_or(nodes::logical_or(c.y().less_than(nodes::num(0)), c.y().greater_than(nodes::num(1))),
                                                       c.z().greater_than(nodes::num(1))));
    st[ns++] = nodes::if_then(outside, nodes::block(node_return(nodes::num(1))));
    Node tiled = params.x().greater_than(nodes::num(1.5f));
    if (self.key.vsm) {
        // one lookup: the map is blurred already (params.w is the bias)
        node_func_body(f, nodes::seq(drie::node_block(st[:ns]), vsm_lookup(vsm_moments(tiled, rect, params.z(), c.xy()), c.z() + params.w())));
        return f;
    }
    // the map's footprint of a pixel of the screen: blurs much smaller than that need few taps,
    // and the search for what casts the shadow none
    Node spx = m * nodes::vec4(p + dpdx, 1);
    Node spy = m * nodes::vec4(p + dpdy, 1);
    Node dx = nodes::select(spx.w().greater_than(nodes::num(0)), spx.xyz() / spx.w() - c, nodes::vec3(0));
    Node dy = nodes::select(spy.w().greater_than(nodes::num(0)), spy.xyz() / spy.w() - c, nodes::vec3(0));
    Node pixel = nodes::max(nodes::max(nodes::length(dx.xy()), nodes::length(dy.xy())), 1e-6f);
    // rotated disk of taps for a soft edge
    Node[2] rot = tap_rotation(frag);
    Node ca = rot[0];
    Node sa = rot[1];
    Node r = nodes::variable(params.z() * params.w());
    // how the depth changes over the map along this surface, so that a wide blur does not run
    // the surface into its own shadow
    Node grad = nodes::variable(nodes::vec2(0));
    Node most = nodes::variable(shadow_taps(8));
    st[ns++] = drie::node_eval(c);
    st[ns++] = drie::node_eval(pixel);
    st[ns++] = r;
    st[ns++] = grad;
    st[ns++] = most;
    if (self.key.soft_shadows) {
        // A source with a size: the edge of a shadow is as wide as the source looks from there
        // past the thing that casts it. Sharp where the two touch, wide far behind. How far the
        // caster is comes from asking the map at a few depths, nearer each time.
        Node dr = shadow_distance(c.z(), soft);
        Node per_m = 1.0f / (nodes::abs(soft.w()) * dr);
        Node search = nodes::clamp(soft.x() * 1.6f * per_m, r, 0.04f);
        Node found = nodes::variable(nodes::num(0));
        Node nearest = nodes::variable(nodes::num(0));
        // the middle and a ring round it
        Node looks = nodes::clamp(nodes::convert(shadow_sample_scale() * 7.0f + 0.5f, INT), nodes::int_(4), nodes::int_(SHADOW_MAX_TAPS));
        Node k = nodes::loop_index();
        Node a = nodes::convert(k, FLOAT) * 6.2831853f / nodes::convert(looks - nodes::int_(1), FLOAT);
        Node ring = nodes::vec2(nodes::cos(a) * ca - nodes::sin(a) * sa, nodes::cos(a) * sa + nodes::sin(a) * ca) * search;
        Node o = nodes::select(k.equal(nodes::int_(0)), nodes::vec2(0), ring);
        Node at = c.xy() + o;
        Node lo = nodes::variable(soft.y());
        Node hi = nodes::variable(dr);
        Node j = nodes::loop_index();
        Node mid = (lo + hi) * 0.5f;
        Node closer = self.tap(tiled, rect, params.z(), at, shadow_depth(mid, soft)).greater_than(nodes::num(0.5f));
        Node bisect = nodes::loop(j, 0, 5, nodes::block(nodes::if_then(closer, nodes::block(lo.assign(mid)), nodes::block(hi.assign(mid)))));
        Node look = nodes::block(
            nodes::if_then(self.tap(tiled, rect, params.z(), at, c.z()).greater_than(nodes::num(0.5f)), nodes::block(nodes::continue_())),
            lo, hi, bisect,
            nearest.assign(nearest + (lo + hi) * 0.5f),
            found.assign(found + 1.0f));
        Node db = nearest / found;
        Node widen = nodes::block(
            found, nearest,
            nodes::loop(k, 0, looks, look),
            nodes::if_then(found.less_than(nodes::num(0.5f)), nodes::block(node_return(nodes::num(1)))),
            r.assign(nodes::clamp(soft.x() * (dr - db) / db * per_m, r, 0.04f)));
        Node det = dx.x() * dy.y() - dx.y() * dy.x();
        Node slope_grad = nodes::vec2(dy.y() * dx.z() - dx.y() * dy.z(), dx.x() * dy.z() - dy.x() * dx.z()) / det;
        Node soft_block = nodes::block(
            drie::node_eval(dr),
            drie::node_eval(per_m),
            drie::node_eval(search),
            most.assign(shadow_taps(14)),
            // the widest edge the source could make here is under a pixel: no need to look
            nodes::if_then(search.greater_than(pixel), widen),
            nodes::if_then(nodes::abs(det).greater_than(nodes::num(1e-12f)), nodes::block(grad.assign(slope_grad))));
        st[ns++] = nodes::if_then(soft.x().greater_than(nodes::num(0)), soft_block);
    }
    // about three taps per pixel across the blur, at least four (each tap is a bilinear 2x2)
    Node taps = nodes::clamp(nodes::convert(nodes::ceil(r / pixel * 3.0f), INT), nodes::int_(4), most);
    Node sum = nodes::variable(nodes::num(0));
    Node k = nodes::loop_index();
    Node a = nodes::convert(k, FLOAT) * 2.39996f;
    Node rr = nodes::sqrt((nodes::convert(k, FLOAT) + 0.5f) / nodes::convert(taps, FLOAT)) * r;
    Node o = rotate(nodes::vec2(nodes::cos(a), nodes::sin(a)) * rr, ca, sa);
    Node z = c.z() + nodes::clamp(nodes::dot(grad, o), -0.004f, 0.004f);
    st[ns++] = drie::node_eval(taps);
    st[ns++] = sum;
    st[ns++] = nodes::loop(k, 0, taps, nodes::block(sum.assign(sum + self.tap(tiled, rect, params.z(), c.xy() + o, z))));
    node_func_body(f, nodes::seq(drie::node_block(st[:ns]), sum / nodes::convert(taps, FLOAT)));
    return f;
}

// Element `i` of one of FsParams' arrays of four (a `t` from component `comp` of it).
macro Node fs_at($name, NodeType t, Node i, int comp = 0) => node_input(FS, fs_row($name), t, comp, i, t == MAT4 ? 4 : 1, 4);

// 1 where cascade `i` of the sun's map has nothing nearer than depth `z` at `uv` (0..1 in its tile).
fn Node ShadowFns.sun_tap(&self, Node i, Node uv, Node z) {
    Node rect = fs_at("dir_cascade_rect", VEC4, i);
    // stay half a texel inside the tile: the neighbours belong to other cascades
    Node t = fs_in("dir_shadow_params", FLOAT, 2) / rect.swizzle("zw");
    Node at = rect.xy() + nodes::clamp(uv, t * 0.5f, (t * 0.5f).one_minus()) * rect.swizzle("zw");
    if (self.key.vsm) {
        return vsm_lookup(node_sample(LEVEL, shaders::VIEW_VSM_SUN, shaders::SMP_VSM_SMP, TEX_2D, at, nodes::num(0)).xy(),
                          z + fs_in("shadow_ext", FLOAT));
    }
    return node_sample(COMPARE, shaders::VIEW_SUN_SHADOW, shaders::SMP_SHADOW_SMP, TEX_DEPTH, at, z);
}

// Shadow of the directional light in cascade `i`: a box of parallel rays towards `l`.
fn Node ShadowFns.sun_in(&self, Node i, Node n, Node l, Node world, Node frag) {
    if (self.sun_in_fn < 0) {
        NodeType[5] types = { INT, VEC3, VEC3, VEC3, VEC2 };
        int f = node_func("dir_shadow_in", FLOAT, &types);
        Node ci = node_param(f, 0);
        Node pn = node_param(f, 1);
        Node pl = node_param(f, 2);
        Node pw = node_param(f, 3);
        Node pf = node_param(f, 4);
        Node params = fs_in("dir_shadow_params");
        // texels of the far cascades are bigger: push further off the surface, blur fewer texels
        Node scale = fs_at("dir_cascade_info", FLOAT, ci, 1) / nodes::max(fs_in("dir_cascade_info", FLOAT, 1), 1e-6f);
        Node slope = nodes::clamp(nodes::dot(pn, pl), 0, 1).one_minus();
        Node c = (fs_at("dir_shadow_mat", MAT4, ci) * nodes::vec4(pw + pn * params.y() * scale * (slope + 0.4f), 1)).xyz();
        Node[4] st;
        int ns;
        st[ns++] = nodes::if_then(outside_box(c, true), nodes::block(node_return(nodes::num(1))));
        Node sum;
        if (self.key.vsm) {
            // one lookup: each cascade's tile is blurred already
            sum = self.sun_tap(ci, c.xy(), c.z()) * 10.0f;
        } else {
            Node[2] rot = tap_rotation(pf);
            Node texel = params.z() / fs_at("dir_cascade_rect", FLOAT, ci, 2);
            Node r = texel * nodes::max(params.w() / scale, nodes::min(params.w(), 1));
            Node taps = shadow_taps(10);
            Node acc = nodes::variable(nodes::num(0));
            Node k = nodes::loop_index();
            Node a = nodes::convert(k, FLOAT) * 2.39996f;
            Node o = nodes::vec2(nodes::cos(a), nodes::sin(a)) * (nodes::sqrt((nodes::convert(k, FLOAT) + 0.5f) / nodes::convert(taps, FLOAT)) * r);
            st[ns++] = acc;
            st[ns++] = nodes::loop(k, 0, taps, nodes::block(acc.assign(acc + self.sun_tap(ci, c.xy() + rotate(o, rot[0], rot[1]), c.z()))));
            sum = acc * (10.0f / nodes::convert(taps, FLOAT));
        }
        // one box: nothing is shadowed outside of it, fade towards its edge
        Node edge = nodes::min(c.xy(), c.xy().one_minus());
        Node boxed = nodes::mix(1, sum / 10.0f, nodes::clamp(nodes::min(edge.x(), edge.y()) * 10.0f, 0, 1));
        Node shade = nodes::select(fs_in("dir_cascade_params", FLOAT, 3).less_than(nodes::num(0.5f)), sum / 10.0f, boxed);
        node_func_body(f, nodes::seq(drie::node_block(st[:ns]), shade));
        self.sun_in_fn = f;
    }
    Node[5] args = { i, n, l, world, frag };
    return node_call_fn(self.sun_in_fn, &args);
}

// Whether `c` is outside the 0..1 box of a shadow map (`near`: in front of it too).
fn Node outside_box(Node c, bool near) @local {
    Node out = nodes::logical_or(nodes::logical_or(c.x().less_than(nodes::num(0)), c.x().greater_than(nodes::num(1))),
                                 nodes::logical_or(nodes::logical_or(c.y().less_than(nodes::num(0)), c.y().greater_than(nodes::num(1))),
                                                   c.z().greater_than(nodes::num(1))));
    return near ? nodes::logical_or(out, c.z().less_than(nodes::num(0))) : out;
}

// The view depth of world position `p` along the sun's cascades.
fn Node sun_depth(Node p) @private => nodes::dot(p - fs_in("camera_pos", VEC3), fs_in("dir_cascade_fwd", VEC3));

// The cascade of the sun's map that covers view depth `depth`, -1 past the last (an INT).
fn Node sun_cascade(Node depth) @private {
    Node cp = fs_in("dir_cascade_params");
    Node i = nodes::variable(nodes::int_(-1));
    Node k = nodes::loop_index();
    Node pick = nodes::loop(k, 0, nodes::convert(cp.x(), INT), nodes::block(
        nodes::if_then(depth.less_than(fs_at("dir_cascade_info", FLOAT, k)), nodes::block(i.assign(k), nodes::break_()))));
    Node choose = nodes::if_then(cp.w().greater_than(nodes::num(0.5f)), nodes::block(i.assign(nodes::int_(0))), nodes::block(pick));
    return nodes::seq(nodes::block(i, choose), i);
}

// Shadow of the directional light that casts one, picking the cascade by view depth and fading
// into the next one near its end (dir_shadow_full and dir_shadow, its intensity).
fn Node ShadowFns.sun(&self, Node n, Node l, Node world, Node frag) {
    Node depth = sun_depth(world);
    Node cp = fs_in("dir_cascade_params");
    Node count = nodes::convert(cp.x(), INT);
    Node i = sun_cascade(depth);
    Node lit = nodes::variable(nodes::num(1));
    Node first = self.sun_in(i, n, l, world, frag);
    Node start = nodes::select(i.equal(nodes::int_(0)), 0.0f, fs_at("dir_cascade_info", FLOAT, nodes::max(i - nodes::int_(1), nodes::int_(0))));
    Node end = fs_at("dir_cascade_info", FLOAT, i);
    Node band = (end - start) * cp.y();
    Node t = nodes::clamp((depth - (end - band)) / band, 0, 1);
    // the last one fades out to no shadow
    Node next_i = i + nodes::int_(1);
    Node next = nodes::variable(nodes::num(1));
    Node blend = nodes::block(
        next,
        nodes::if_then(next_i.less_than(count), nodes::block(next.assign(self.sun_in(next_i, n, l, world, frag)))),
        lit.assign(nodes::mix(lit, next, t)));
    Node fade = nodes::logical_and(nodes::logical_and(cp.w().less_than_equal(nodes::num(0.5f)), band.greater_than(nodes::num(0))),
                                   t.greater_than(nodes::num(0)));
    Node full = nodes::seq(nodes::block(
        lit,
        nodes::if_then(i.greater_than_equal(nodes::int_(0)), nodes::block(
            lit.assign(first),
            nodes::if_then(fade, blend)))), lit);
    return nodes::mix(1, full, fs_in("dir_cascade_fwd", FLOAT, 3));
}

// Shadow of a directional light besides the sun, whose box is tile `si` of shadow_atlas: as the
// sun's, without cascades; nothing is shadowed outside of it.
fn Node ShadowFns.dir_tile(&self, Node si, Node n, Node l, Node world, Node frag) {
    if (self.dir_tile_fn < 0) {
        NodeType[5] types = { INT, VEC3, VEC3, VEC3, VEC2 };
        int f = node_func("dir_tile_shadow", FLOAT, &types);
        Node ps = node_param(f, 0);
        Node pn = node_param(f, 1);
        Node pl = node_param(f, 2);
        Node pw = node_param(f, 3);
        Node pf = node_param(f, 4);
        Node params = spot_shadow(ps, 4);
        Node rect = spot_shadow(ps, 5);
        Node slope = nodes::clamp(nodes::dot(pn, pl), 0, 1).one_minus();
        Node c = (spot_shadow_mat(ps) * nodes::vec4(pw + pn * params.y() * (slope + 0.4f), 1)).xyz();
        Node[4] st;
        int ns;
        st[ns++] = nodes::if_then(outside_box(c, true), nodes::block(node_return(nodes::num(1))));
        Node sum;
        if (self.key.vsm) {
            // one lookup: the tile is blurred already (params.w is the bias)
            sum = vsm_lookup(vsm_moments(nodes::bool_(true), rect, params.z(), c.xy()), c.z() + params.w()) * 10.0f;
        } else {
            Node[2] rot = tap_rotation(pf);
            Node r = params.z() * params.w();
            Node taps = shadow_taps(10);
            Node acc = nodes::variable(nodes::num(0));
            Node k = nodes::loop_index();
            Node a = nodes::convert(k, FLOAT) * 2.39996f;
            Node o = nodes::vec2(nodes::cos(a), nodes::sin(a)) * (nodes::sqrt((nodes::convert(k, FLOAT) + 0.5f) / nodes::convert(taps, FLOAT)) * r);
            st[ns++] = acc;
            st[ns++] = nodes::loop(k, 0, taps, nodes::block(
                acc.assign(acc + self.tap(nodes::bool_(true), rect, params.z(), c.xy() + rotate(o, rot[0], rot[1]), c.z()))));
            sum = acc * (10.0f / nodes::convert(taps, FLOAT));
        }
        // fade out towards the box's edge
        Node edge = nodes::min(c.xy(), c.xy().one_minus());
        Node shade = nodes::mix(1, sum / 10.0f, nodes::clamp(nodes::min(edge.x(), edge.y()) * 10.0f, 0, 1));
        node_func_body(f, nodes::seq(drie::node_block(st[:ns]), shade));
        self.dir_tile_fn = f;
    }
    Node[5] args = { si, n, l, world, frag };
    return node_call_fn(self.dir_tile_fn, &args);
}

// Shadow of a point light: the face of its cube of maps that the point is in.
fn Node ShadowFns.point(&self, Node pos, Node si, Node n, Node world, Node dpdx, Node dpdy, Node frag) {
    Node d = world - pos;
    Node a = nodes::abs(d);
    Node fx = nodes::select(d.x().greater_than_equal(nodes::num(0)), nodes::int_(0), nodes::int_(1));
    Node fy = nodes::select(d.y().greater_than_equal(nodes::num(0)), nodes::int_(2), nodes::int_(3));
    Node fz = nodes::select(d.z().greater_than_equal(nodes::num(0)), nodes::int_(4), nodes::int_(5));
    Node face = nodes::select(nodes::logical_and(a.x().greater_than_equal(a.y()), a.x().greater_than_equal(a.z())), fx,
                              nodes::select(a.y().greater_than_equal(a.z()), fy, fz));
    return self.light(pos, si + face, n, world, dpdx, dpdy, frag);
}
