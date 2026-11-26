// Vertex and fragment node-graph builders for forward materials, depth prepasses, shadow maps,
// lightmap baking, and the deferred G-buffer.
//
// `vertex_stage` evaluates morph targets, vertex displacement, instance transforms, position_node,
// model transforms, and sway in order. When `key.prepassed` is set, the depth prepass fragment
// stage is built first so its varyings lead the main pass's varying list. `atlas_depth` linearizes
// window depth using backend clip Z (`[-1, 1]` on GLSL, `[0, 1]` on Metal/HLSL). Invoked by
// `material_program` (src/material/material_key.c3); delegates lit shading to `Lit`
// (src/material/material_lit.c3), fog/reflection helpers to src/material/layers.c3, shadows to
// src/material/shadow_lookup.c3, and shared shading math to `drie::shading` (src/nodes/shading.c3).
module drie;
import sokol, drie::nodes, drie::shading, std::collections::map, std::io, std::os::env, std::thread;

// ---- Vertex stage

// What the vertex stage works out (its nodes): the fragment stage reads them as varyings.
struct VertexValues @private {
    Node position; // the clip position (a SEQ: the statements before it)
    Node world;
    Node normal;
    Node uv;
    Node uv1;
    Node color; // the vertex color, rgba
    Node instance_color;
    Node tangent; // xyz in the world, w its handedness
    Node line_distance;
    Node clip;
    Node prev_clip;
    // a node material's: the position in the model (with the instance's matrix) as
    // position_node leaves it, the normal there, the instance's index (a float)
    Node local;
    Node local_normal;
    Node instance_index;
}

// The instance's matrix rows, as fetch_instance picks them.
struct Instance @local {
    Node m0;
    Node m1;
    Node m2;
}

// Transforms local-space point `p` by the 3x4 affine instance matrix.
fn Node Instance.point(&self, Node p) {
    Node l = nodes::vec4(p, 1);
    return nodes::vec3(nodes::dot(self.m0, l), nodes::dot(self.m1, l), nodes::dot(self.m2, l));
}

// Transforms local-space direction `d` by the upper 3x3 of the instance matrix.
fn Node Instance.dir(&self, Node d) {
    return nodes::vec3(nodes::dot(self.m0.xyz(), d), nodes::dot(self.m1.xyz(), d), nodes::dot(self.m2.xyz(), d));
}

// `build`: a node material's graphs (position_node), whose builtins it binds in `binds`.
fn VertexValues vertex_stage(MaterialKey key, NodeBuild* build = null, UserBinds* binds = null) @local {
    VertexValues o;
    Node position = node_attribute(shaders::ATTR_THREE_POSITION, VEC3);
    Node normal   = node_attribute(shaders::ATTR_THREE_NORMAL, VEC3);
    Node uv       = node_attribute(shaders::ATTR_THREE_UV, VEC2);
    Node color0   = node_attribute(shaders::ATTR_THREE_COLOR0, VEC4);
    Node[8] statements;
    int ns;
    Node instance_id;
    Instance inst;
    if (key.gpu_cull) {
        // (fetch_instance: instance i of the draw is rows[3 * visible[first + i]], if i is
        // under the cull's survivors)
        Node cull  = vs_in("inst_cull");
        Node i     = node_system(INSTANCE_ID);
        Node count = node_storage(shaders::VIEW_INST_COUNTS, UINT, nodes::convert(cull.z(), UINT));
        statements[ns++] = nodes::if_then(
            i.greater_than_equal(count),
            nodes::block(node_return(nodes::vec4(2, 2, 2, 1)))
        );
        Node id = node_storage(shaders::VIEW_INST_VISIBLE, UINT, nodes::convert(cull.y(), UINT) + i);
        instance_id = id;
        Node s = id * nodes::uint_(3);
        inst = {
            node_storage(shaders::VIEW_INST_ROWS, VEC4, s),
            node_storage(shaders::VIEW_INST_ROWS, VEC4, s + nodes::uint_(1)),
            node_storage(shaders::VIEW_INST_ROWS, VEC4, s + nodes::uint_(2)),
        };
    } else {
        inst = {
            node_attribute(shaders::ATTR_THREE_INST0, VEC4),
            node_attribute(shaders::ATTR_THREE_INST1, VEC4),
            node_attribute(shaders::ATTR_THREE_INST2, VEC4),
        };
        instance_id = node_system(INSTANCE_ID);
    }
    Node local = position;
    if (key.morph) {
        Node hdr      = node_storage(VIEW_MORPH_WEIGHTS, VEC4, nodes::uint_(0));
        Node m_count  = nodes::convert(hdr.x(), UINT);
        Node v_count  = nodes::convert(hdr.y(), UINT);
        Node w_stride = nodes::convert(hdr.z(), UINT);
        Node vid      = node_system(VERTEX_ID);
        Node w_base   = nodes::uint_(1) + instance_id * w_stride;
        Node mp       = nodes::variable(local);
        Node ti       = node_var(UINT);
        Node w4       = node_storage(VIEW_MORPH_WEIGHTS, VEC4, w_base + node_binary(SHR, ti, nodes::uint_(2)));
        Node lane     = node_binary(BIT_AND, ti, nodes::uint_(3));
        Node w = nodes::select(
            lane.equal(nodes::uint_(0)),
            w4.x(),
            nodes::select(
                lane.equal(nodes::uint_(1)),
                w4.y(),
                nodes::select(lane.equal(nodes::uint_(2)), w4.z(), w4.w())
            )
        );
        Node delta = node_storage(VIEW_MORPH_DELTAS, VEC4, ti * v_count + vid).xyz();
        Node body  = nodes::block(mp.assign(mp + delta * w));
        statements[ns++] = mp;
        statements[ns++] = node_loop(LT, ti, nodes::uint_(0), m_count, body);
        local = mp;
    }
    if (key.displacement) {
        Node d      = vs_in("surface_vs");
        Node height = node_attribute(shaders::ATTR_THREE_HEIGHT, FLOAT);
        local = local + nodes::normalize(normal) * (height * d.y() + d.z());
    }
    Node ip     = inst.point(local);
    Node placed = ip;
    Node model  = vs_in("model", MAT4);
    Node dir    = inst.dir(normal);
    if (build) {
        // Bind vertex-stage builtins for node graphs and evaluate position_node in `ip`'s place
        // (after displacement and instance transform, before model transform and sway). Rebind
        // position and view builtins afterward so downstream varyings see the moved vertex.
        Node[NodeBuiltin::len]* b = &binds.v;
        user_header(b);
        Node world0  = (model * nodes::vec4(ip, 1)).xyz();
        Node normal0 = nodes::normalize((model * nodes::vec4(dir, 0)).xyz());
        o.instance_index = nodes::convert(instance_id, FLOAT);
        (*b)[NodeBuiltin.POSITION_LOCAL.ordinal] = ip;
        (*b)[NodeBuiltin.NORMAL_LOCAL.ordinal] = dir;
        (*b)[NodeBuiltin.POSITION_WORLD.ordinal] = world0;
        (*b)[NodeBuiltin.NORMAL_WORLD.ordinal] = normal0;
        (*b)[NodeBuiltin.NORMAL_GEOMETRY.ordinal] = normal0;
        (*b)[NodeBuiltin.VIEW_DIRECTION.ordinal] = nodes::normalize(user_camera() - world0);
        (*b)[NodeBuiltin.UV.ordinal] = uv;
        (*b)[NodeBuiltin.UV1.ordinal] = node_attribute(shaders::ATTR_THREE_UV1, VEC2);
        (*b)[NodeBuiltin.VERTEX_COLOR.ordinal] = color0;
        (*b)[NodeBuiltin.INSTANCE_INDEX.ordinal] = o.instance_index;
        // (the rows' matrix, by its columns: instance_matrix() * vec4(p, 1) is inst.point(p))
        Node[4] cols;
        for (int j = 0; j < 4; j++) {
            String c = "xyzw"[j:1];
            cols[j] = nodes::vec4(inst.m0.swizzle(c), inst.m1.swizzle(c), inst.m2.swizzle(c), j == 3 ? 1.0f : 0.0f);
        }
        (*b)[NodeBuiltin.INSTANCE_MATRIX.ordinal] = node_matrix(&cols, MAT4);
        Node moved = build.slots[6];
        if (moved.id) {
            Node at     = nodes::variable(nodes::convert(moved, VEC3));
            Node world1 = (model * nodes::vec4(at, 1)).xyz();
            statements[ns++] = at;
            statements[ns++] = rebind(POSITION_LOCAL, at);
            statements[ns++] = rebind(POSITION_WORLD, world1);
            statements[ns++] = rebind(VIEW_DIRECTION, nodes::normalize(user_camera() - world1));
            ip = at;
        }
        o.local = ip;
        o.local_normal = dir;
    }
    Node wp    = model * nodes::vec4(ip, 1);
    Node world = wp.xyz();
    Node sway  = vs_in("sway");
    if (key.sway) world = shading::apply_sway(world, ip.y(), sway);
    Node view_proj = vs_in("view_proj", MAT4);
    Node origin    = vs_in("view_origin", VEC3);
    Node pos       = view_proj * nodes::vec4(world - origin, 1);
    if (key.outline) {
        // (Material.outline: out along the normal as the screen sees it)
        Node inward = world - (model * nodes::vec4(dir, 0)).xyz();
        Node p2     = view_proj * nodes::vec4(inward - origin, 1);
        pos = pos + nodes::normalize(pos - p2) * (vs_in("surface_vs", FLOAT) * pos.w());
    }
    if (key.bake) {
        // (the light map's texel at uv1; the map's first row on top but in GL)
        Node uv1 = node_attribute(shaders::ATTR_THREE_UV1, VEC2);
        Node y = nodes::select(vs_in("bake", FLOAT, 1)
            .greater_than(nodes::num(0.5f)), uv1.y() * -2.0f + 1.0f, uv1.y() * 2.0f - 1.0f);
        pos = nodes::vec4(uv1.x() * 2.0f - 1.0f, y, 0, 1);
    }
    o.position = ns ? nodes::seq(drie::node_block(statements[:ns]), pos) : pos;
    o.world = world;
    o.normal = (vs_in("normal_mat", MAT4) * nodes::vec4(dir, 0)).xyz();
    Node t   = vs_in("uv_transform");
    Node t1  = vs_in("uv_transform", k: 1);
    Node uvh = nodes::vec3(uv, 1);
    o.uv = nodes::vec2(nodes::dot(t.xyz(), uvh), nodes::dot(t1.xyz(), uvh));
    o.uv1 = node_attribute(shaders::ATTR_THREE_UV1, VEC2);
    o.color = color0;
    o.instance_color = node_attribute(shaders::ATTR_THREE_INSTANCE_COLOR, VEC3);
    if (key.tangents) {
        // (a direction along the surface: the model matrix's, as three's)
        Node tangent = node_attribute(shaders::ATTR_THREE_TANGENT, VEC4);
        o.tangent = nodes::vec4((model * nodes::vec4(inst.dir(tangent.xyz()), 0)).xyz(), tangent.w());
    }
    o.line_distance = normal.x();
    if (key.motion) {
        Node motion = vs_in("motion");
        o.clip = nodes::vec4(pos.xy() - motion.swizzle("zw") * pos.w(), pos.z(), pos.w());
        Node prev_view_proj = vs_in("prev_view_proj", MAT4);
        // the same vertex where it was a frame ago (a dynamic geometry's vertices of then),
        // placed as it was, moved as far as position_node moves it now
        Node was            = node_attribute(shaders::ATTR_THREE_PREV_POSITION, VEC3);
        if (key.displacement) {
            Node d = vs_in("surface_vs");
            was = was + nodes::normalize(normal) * (node_attribute(shaders::ATTR_THREE_HEIGHT, FLOAT) * d.y() + d.z());
        }
        was = inst.point(was);
        if (build && build.slots[6].id) was = was + (ip - placed);
        Node pp = vs_in("prev_model", MAT4) * nodes::vec4(was, 1);
        if (key.sway) pp = nodes::vec4(
            shading::apply_sway(pp.xyz(), was.y(), nodes::vec4(motion.y(), sway.y(), sway.z(), sway.w())),
            pp.w()
        );
        Node swayed = nodes::vec4(world, wp.w());
        o.prev_clip = nodes::select(motion.x()
            .greater_than(nodes::num(1.5f)), prev_view_proj * pp, nodes::select(motion.x()
                .greater_than(nodes::num(0.5f)), prev_view_proj * swayed, o.clip));
    }
    return o;
}

// ---- Fragment stage and G-buffer

// Parallax occlusion mapping (Material.parallax_map): the uv where the view ray meets the
// height map, which the material's maps are read at (with the derivatives of v_uv: the offset
// jumps where the height map steps, and its own derivatives would pick a blurry mip level
// there); how the uv changes per meter along the surface (the cotangent frame, as in
// perturb_normal); the height map there (0..1) and the depth after the distance fade (m).
struct Parallax @private {
    Node uv;
    Node gu;
    Node gv;
    Node h;
    Node depth;
}

// The parallax map at uv; hex tiled (Material.hex_tiling), as the other maps are: the patches'
// heights blended with their weights, a patch read only where it shows (most pixels are
// inside one patch, and the march would read three times as much for nothing).
fn Node parallax_height(FragUv* f, Node uv) @local {
    if (!f.hex_w.id) return node_sample(GRAD, shaders::VIEW_TEX_PARALLAX, shaders::SMP_TEX_SMP, TEX_2D, uv, f.dx, f.dy)
        .x();
    Node acc  = nodes::variable(nodes::num(0));
    Node[3] w = { f.hex_w.x(), f.hex_w.y(), f.hex_w.z() };
    Node[4] steps;
    steps[0] = acc;
    for (int i = 0; i < 3; i++) {
        Node h = node_sample(
            GRAD,
            shaders::VIEW_TEX_PARALLAX,
            shaders::SMP_TEX_SMP,
            TEX_2D,
            uv + f.hex_off[i],
            f.dx,
            f.dy
        )
            .x();
        steps[i + 1] = nodes::if_then(w[i].greater_than(nodes::num(0.004f)), nodes::block(acc.assign(acc + h * w[i])));
    }
    return nodes::seq(drie::node_block(steps[:4]), acc);
}

// Marches the view ray (`v`, toward the camera `dist` away) down into the height field (1 = the
// surface, 0 = depth meters under it) to where it meets it.
fn Parallax parallax_offset(FragUv* f, Node n, Node v, Node dist, Node world) @local {
    Node params  = fs_in("parallax_params");
    Node depth   = params.x() * nodes::smoothstep(params.y() * 0.6f, params.y(), dist).one_minus();
    Node pu      = nodes::variable(f.uv);
    Node ph      = nodes::variable(nodes::num(1));
    Node gu      = nodes::variable(nodes::vec3(0));
    Node gv      = nodes::variable(nodes::vec3(0));
    Node dp1     = nodes::dfdx(world);
    Node dp2     = nodes::dfdy(world);
    Node dp2perp = nodes::cross(dp2, n);
    Node dp1perp = nodes::cross(n, dp1);
    Node det     = nodes::dot(dp1, dp2perp);
    // the uv a step of 1 m down the view ray moves by (flattened at a glancing angle, where the
    // march would smear the texture out)
    Node nv      = nodes::max(nodes::dot(n, v), 0.2f);
    Node per_m   = nodes::vec2(nodes::dot(gu, v), nodes::dot(gv, v)).negate() / nv;
    Node total   = per_m * depth;
    Node steps   = nodes::mix(24, 8, nv);
    Node dh      = 1.0f / steps;
    Node ray     = nodes::variable(nodes::num(1));
    Node h       = nodes::variable(parallax_height(f, f.uv));
    Node prev_h  = nodes::variable(h);
    Node uv      = nodes::variable(f.uv);
    Node i       = nodes::loop_index();
    Node march = nodes::loop(
        i,
        0,
        24,
        nodes::block(nodes::if_then(nodes::logical_or(h.greater_than_equal(ray), nodes::convert(i, FLOAT)
        .greater_than_equal(
            steps
        )), nodes::block(
            nodes::break_()
        )), prev_h.assign(h), ray.assign(ray - dh), uv.assign(uv + total * dh), h.assign(parallax_height(f, uv))));
    // between the last step above the surface and the first below it
    Node after  = h - ray;
    Node before = prev_h - (ray + dh);
    Node t      = nodes::clamp(after / nodes::min(after - before, -1e-5f), 0, 1);
    Node meet = nodes::block(
        gu.assign((dp2perp * f.dx.x() + dp1perp * f.dy.x()) / det),
        gv.assign((dp2perp * f.dx.y() + dp1perp * f.dy.y()) / det),
        drie::node_eval(total),
        drie::node_eval(dh),
        ray,
        h,
        prev_h,
        uv,
        march,
        drie::node_eval(t),
        pu.assign(uv - total * dh * t),
        ph.assign(nodes::mix(h, prev_h, t))
    );
    Node deep = nodes::logical_and(depth.greater_than(nodes::num(0)), nodes::abs(det)
        .greater_than_equal(nodes::num(1e-14f)));
    Node at = nodes::seq(
        nodes::block(pu, ph, gu, gv, drie::node_eval(dp2perp), drie::node_eval(dp1perp), nodes::if_then(deep, meet)),
        pu
    );
    return { .uv = at, .gu = gu, .gv = gv, .h = ph, .depth = depth };
}

// How much of the light from l reaches the parallax's uv past the bumps of the height map around
// it (soft: the more a bump stands above the ray to the light, the darker).
fn Node parallax_shadow(Parallax* px, FragUv* f, Node n, Node l) @private {
    Node strength = fs_in("parallax_params", FLOAT, 2);
    Node nl       = nodes::dot(n, l);
    Node per_m    = nodes::vec2(nodes::dot(px.gu, l), nodes::dot(px.gv, l)) / nodes::max(nl, 0.1f);
    Node rise     = px.h.one_minus();
    Node total    = per_m * px.depth * rise;
    Node block    = nodes::variable(nodes::num(0));
    Node i        = nodes::loop_index();
    Node fi       = nodes::convert(i, FLOAT) / 8.0f;
    Node above    = parallax_height(f, px.uv + total * fi) - (px.h + rise * fi);
    Node off = nodes::logical_or(
        nodes::logical_or(px.depth.less_than_equal(nodes::num(0)), strength.less_than_equal(nodes::num(0))),
        nodes::logical_or(nl.less_than_equal(nodes::num(0)), rise.less_than_equal(nodes::num(0.01f)))
    );
    Node shade = nodes::variable(nodes::num(1));
    return nodes::seq(
        nodes::block(
            shade,
            nodes::if_then(
                nodes::logical_not(off),
                nodes::block(
                    block,
                    nodes::loop(i, 1, 9, nodes::block(block.assign(nodes::max(block, above * fi.one_minus() * 8.0f)))),
                    shade.assign(strength.negate() * nodes::clamp(block, 0, 1) + 1.0f)
                )
            )
        ),
        shade
    );
}

// Shadow map program: runs the material's vertex stage from the light (VsParams filled by
// draw_shadow_item) and the alpha-test cutout (color map alpha, opacity, and alpha map green via
// ShadowFsParams). A node material (`build`) applies position_node and cuts the same holes as its
// color step (mask_node, color_node, opacity_node).
fn void build_shadow_graph(MaterialKey key, ProgramGraph* graph, NodeBuild* build, UserBinds* binds) @private {
    VertexValues vtx = vertex_stage(key, build, binds);
    graph.position = vtx.position;
    ShadowFsParams* sp = null;
    int params         = (int)((usz)&sp.s_params / 16);
    int alpha_uv       = (int)((usz)&sp.s_alpha_uv / 16);
    Node p             = node_input(SHADOW_FS, params, VEC4);
    if (key.atlas_depth) {
        // a map of the atlas: the distance from the light's plane, from its near (s_params.z) to
        // its far (w), as 0..1, the same everywhere over its reach (the window depth when w is
        // 0: an orthographic map's is that already)
        Node z   = node_system(FRAG_COORD).z();
        Node n   = p.z();
        Node f   = p.w();
        Node d   = backend_lang() == GLSL ? n * f * 2.0f / (f + n - (z * 2.0f - 1.0f) * (f - n))
            : n * f / (f - z * (f - n));
        Node lin = nodes::clamp((d - n) / nodes::max(f - n, 1e-6f), 0, 1);
        graph.depth = nodes::select(f.greater_than(nodes::num(0)), lin, z);
    }
    bool node_cuts = build && (build.cuts || build.slots[8].id);
    if (!key.alpha_test && !node_cuts) return;
    Node v_uv = nodes::varying(vtx.uv);
    Node a    = p.y();
    if (key.map) a = a * node_sample(SAMPLE, shaders::VIEW_S_MAP, shaders::SMP_S_SMP, TEX_2D, v_uv).w();
    if (key.slot(MAP_ALPHA)) {
        Node uvh = nodes::vec3(v_uv, 1);
        Node at = nodes::vec2(
            nodes::dot(node_input(SHADOW_FS, alpha_uv, VEC3), uvh),
            nodes::dot(node_input(SHADOW_FS, alpha_uv + 1, VEC3), uvh)
        );
        a = a * node_sample(SAMPLE, shaders::VIEW_S_ALPHA, shaders::SMP_S_ALPHA_SMP, TEX_2D, at).y();
    }
    Node[4] cuts;
    int ncuts;
    if (node_cuts) {
        Node world = nodes::varying(vtx.world);
        Node n     = nodes::normalize(nodes::varying(vtx.normal));
        if (!key.foliage) n = nodes::select(node_system(FRONT_FACING), n, n.negate());
        fragment_binds(&vtx, &binds.f, v_uv, world, n);
        Node rgb = nodes::vec3(1);
        user_color(build, n, &rgb, &a, &cuts, &ncuts);
    }
    if (key.alpha_test) cuts[ncuts++] = nodes::if_then(a.less_than(p.x()), nodes::block(nodes::discard()));
    graph.fragment = drie::node_block(cuts[:ncuts]);
}

// The graphs of `key`'s program, and with MaterialKey.prepassed of its depth prepass's
// (`prepass`: the holes it cuts, and the alpha it leaves for alpha-to-coverage). A node
// material's graphs (`build`, MaterialKey.nodes) go in at their steps, reading their builtins
// as `binds` and the steps below bind them.
fn void build_material_graph(
    MaterialKey key,
    ProgramGraph* graph,
    ProgramGraph* prepass,
    NodeBuild* build,
    UserBinds* binds
) @private {
    VertexValues vtx = vertex_stage(key, build, binds);
    graph.position = vtx.position;

    Node world    = nodes::varying(vtx.world);
    Node v_normal = nodes::varying(vtx.normal);
    if (key.flat_shading) {
        Node fn_      = nodes::cross(nodes::dfdx(world), nodes::dfdy(world));
        Node view_dir = fs_in("camera_pos", VEC3) - world;
        Node oriented = nodes::select(nodes::dot(fn_, view_dir).greater_than_equal(nodes::num(0)), fn_, fn_.negate());
        v_normal = nodes::select(nodes::dot(fn_, fn_).greater_than(nodes::num(1e-12f)), oriented, v_normal);
    }
    Node v_uv  = nodes::varying(vtx.uv);
    Node fc    = node_system(FRAG_COORD);
    Node front = node_system(FRONT_FACING);
    Node[12] cuts;
    int ncuts;
    Node[8] holes;
    int nholes;
    // a node material's slots (NodeBuild.slots: none without one), and the facing vertex normal
    // its color step reads
    Node[NODE_SLOTS] user;
    Node user_n;
    if (build) {
        user = build.slots;
        user_n = nodes::normalize(v_normal);
        if (!key.foliage && !key.flat_shading) user_n = nodes::select(front, user_n, user_n.negate());
        fragment_binds(&vtx, &binds.f, v_uv, world, user_n);
    }

    // the MRT targets' normal, motion and indirect light (from the start: the screen-space
    // map's early out keeps them)
    Node mrt_normal;
    Node velocity;
    Node rough = fs_in("mat_params", FLOAT, 1);
    Node metal = fs_in("mat_params", FLOAT, 2);
    if (key.mrt) {
        mrt_normal = nodes::vec4(shading::pack_normal_octahedral(nodes::normalize(v_normal)), 0, 1);
        if (key.motion) {
            Node clip = nodes::varying(vtx.clip);
            Node prev = nodes::varying(vtx.prev_clip);
            Node now  = clip.xy() / clip.w();
            Node was  = prev.xy() / nodes::max(prev.w(), 1e-5f);
            Node d    = (now - was) * 0.5f;
            d = nodes::select(fs_in("ssr_params", FLOAT, 3)
                .greater_than(nodes::num(0.5f)), nodes::vec2(d.x(), d.y().negate()), d);
            velocity = nodes::vec4(d, 0, 1);
        } else {
            velocity = nodes::vec4(0, 0, 0, 1);
        }
    }

    // the holes cut whatever the material (cull box, dashes, LOD cross-fade)
    if (key.cull_box) {
        Node lo     = fs_in("cull_min");
        Node hi     = fs_in("cull_max");
        Node inside = lo.w().greater_than(nodes::num(0.5f));
        for (int i = 0; i < 3; i++) {
            String c = "xyz"[i:1];
            inside = nodes::logical_and(inside, world.swizzle(c).greater_than(lo.swizzle(c)));
            inside = nodes::logical_and(inside, world.swizzle(c).less_than(hi.swizzle(c)));
        }
        cuts[ncuts++] = nodes::if_then(inside, nodes::block(nodes::discard()));
    }
    if (key.dash) {
        Node dash  = fs_in("line_dash");
        Node along = nodes::varying(vtx.line_distance);
        Node gap = nodes::logical_and(dash.w()
            .greater_than(nodes::num(0.5f)), nodes::mod(along * dash.z(), dash.y()).greater_than(dash.x()));
        cuts[ncuts++] = nodes::if_then(gap, nodes::block(nodes::discard()));
    }
    if (key.dither) {
        Node sp = fs_in("screen_params");
        Node dither = nodes::fract(
            nodes::fract(nodes::dot(fc.xy(), nodes::vec2(0.06711056f, 0.00583715f))) * 52.9829189f
        );
        Node out = nodes::select(sp.w()
            .greater_than(nodes::num(0)), dither.greater_than_equal(sp.z()), dither.less_than(sp.z()));
        Node cut = nodes::if_then(nodes::logical_and(sp.w()
            .not_equal(nodes::num(0)), out), nodes::block(nodes::discard()));
        if (key.prepassed) {
            holes[nholes++] = cut;
        } else {
            cuts[ncuts++] = cut;
        }
    }

    if (key.screen) {
        // a window onto another render: its pixels as they are, already tone mapped
        Node at    = fc.xy() * fs_in("screen_params", VEC2);
        Node shown = node_sample(SAMPLE, shaders::VIEW_TEX_MAP, shaders::SMP_TEX_SMP, TEX_2D, at);
        graph.colors[0] = nodes::vec4(shown.xyz(), 1);
        if (key.mrt) {
            graph.colors[1] = mrt_normal;
            graph.colors[2] = velocity;
            graph.colors[3] = nodes::vec4(0, 0, rough, metal);
        }
        if (ncuts) graph.fragment = drie::node_block(cuts[:ncuts]);
        return;
    }

    Node v_color = nodes::varying(vtx.color);
    Node base    = fs_in("color");
    Node rgb     = base.xyz();
    if (key.tint) rgb = rgb * v_color.xyz();
    rgb = rgb * nodes::varying(vtx.instance_color);
    Node a   = base.w();
    FragUv f = { .uv = v_uv, .dx = nodes::dfdx(v_uv), .dy = nodes::dfdy(v_uv) };
    Node widen;
    if (key.defocus) {
        // Material.defocus: the maps read over the lens's blur at this distance, as wide as it
        // is (the depth of field pass sees through what is transparent)
        Node focus = fs_in("focus_params");
        Node at = node_sample(
            LEVEL,
            shaders::VIEW_TEX_EXT0 + key.slot(MAP_FOCUS) - 1,
            shaders::SMP_TEX_SMP,
            TEX_2D,
            nodes::vec2(0.5f),
            nodes::num(0)
        );
        Node fd   = nodes::max(at.x(), 0.01f);
        Node z    = nodes::max(nodes::length(world - fs_in("camera_pos", VEC3)), 0.01f);
        Node blur = nodes::min(focus.x() / fd * nodes::abs((fd / z).one_minus()), focus.y());
        widen = nodes::max(blur * 2.0f, 1.0f);
        f.dx = f.dx * widen;
        f.dy = f.dy * widen;
    }
    // Material.hex_tiling: the patches where the pixel is (before the parallax: they stay put as
    // the march moves the uv, and its height map is read through them as the other maps are)
    if (key.hex) {
        f.hex_setup(fs_in("parallax_params", FLOAT, 3));
        if (key.parallax) {
            // (evaluated once, out here: the march reads them in its loop)
            foreach (o : f.hex_off) cuts[ncuts++] = drie::node_eval(o);
            cuts[ncuts++] = drie::node_eval(f.hex_w);
        }
    }
    Parallax px;
    if (key.parallax) {
        // parallax_map: where the view ray meets the height map, for all of the material's maps
        // (the statements first: the alpha test reads the maps)
        Node pn = nodes::normalize(v_normal);
        if (!key.foliage) pn = nodes::select(front, pn, pn.negate());
        Node pv = fs_in("camera_pos", VEC3) - world;
        Node pd = nodes::length(pv);
        px = parallax_offset(&f, pn, pv / nodes::max(pd, 1e-5f), pd, world);
        f.uv = px.uv;
        cuts[ncuts++] = drie::node_eval(px.uv);
    }
    // (the maps of uv_rel's slots: tex_normal, tex_arm, tex_ext0..5)
    if (key.has_maps() || key.normal_map || key.arm_ao || key.arm_rough || key.arm_metal) {
        f.uv1 = nodes::varying(vtx.uv1);
        f.dx1 = nodes::dfdx(f.uv1);
        f.dy1 = nodes::dfdy(f.uv1);
        if (widen.id) {
            f.dx1 = f.dx1 * widen;
            f.dy1 = f.dy1 * widen;
        }
    }
    Node map_rgb;
    if (key.map) {
        Node texel = key.hex ? f.hex_color()
            : node_sample(GRAD, shaders::VIEW_TEX_MAP, shaders::SMP_TEX_SMP, TEX_2D, f.uv, f.dx, f.dy);
        rgb = rgb * texel.xyz();
        a = a * texel.w();
        map_rgb = texel.xyz();
    }
    // three's alphaMap: its green channel
    if (key.slot(MAP_ALPHA)) a = a * role_texel(key, &f, MAP_ALPHA).y();
    if (build) {
        // Node material color step (mask_node, color_node, opacity_node): cut holes in the depth
        // prepass when present, since the main pass only draws where the prepass wrote depth.
        if (key.prepassed) {
            user_color(build, user_n, &rgb, &a, &holes, &nholes);
        } else {
            user_color(build, user_n, &rgb, &a, &cuts, &ncuts);
        }
    }
    Node alpha = a;
    if (key.vertex_alpha) alpha = alpha * v_color.w();
    if (key.alpha_test) {
        Node threshold = fs_in("tex_params", FLOAT, 3);
        if (key.map) {
            // far away the texture's smaller levels average leaves and gaps into an alpha under
            // the cutoff: made up for, so as much stays covered
            Node texel = v_uv * nodes::convert(
                node_sample(SIZE, shaders::VIEW_TEX_MAP, shaders::SMP_TEX_SMP, TEX_2D, { }),
                VEC2
            );
            Node tx    = nodes::dfdx(texel);
            Node ty    = nodes::dfdy(texel);
            Node level = nodes::log2(nodes::max(nodes::max(nodes::dot(tx, tx), nodes::dot(ty, ty)), 1)) * 0.5f;
            alpha = alpha * (nodes::min(level, 6) * 0.3f + 1.0f);
        }
        // the cutout edge sharpened to about a pixel, for alpha-to-coverage
        alpha = (alpha - threshold) / nodes::max(nodes::fwidth(alpha), 1e-4f) + 0.5f;
        Node cut = nodes::if_then(alpha.less_than_equal(nodes::num(0)), nodes::block(nodes::discard()));
        if (key.prepassed) {
            holes[nholes++] = cut;
        } else {
            cuts[ncuts++] = cut;
        }
        alpha = nodes::min(alpha, 1);
    }
    if (key.prepassed) {
        // the depth prepass: its holes, and the alpha the cut-out's edge leaves for
        // alpha-to-coverage (it writes no color); the main pass draws only where it has drawn,
        // whatever covered (an alpha test's cut-out whole)
        prepass.fragment = drie::node_block(holes[:nholes]);
        prepass.colors[0] = nodes::vec4(0, 0, 0, key.alpha_test ? alpha : nodes::num(1));
        if (key.alpha_test) alpha = nodes::num(1);
    }

    Node n = nodes::normalize(v_normal);
    // (baking: the faces as the uv unwrap turned them, all front)
    if (!key.foliage && !key.bake && !key.flat_shading) n = nodes::select(front, n, n.negate());
    // (the color step's NORMAL_GEOMETRY node itself, so that the later steps bind the same
    // node and what reads it isn't printed again)
    if (build && !key.bake) n = user_n;
    Node geo_n = n;
    if (key.normal_map) {
        // tangent-space normal mapping (perturb_normal), in the frame of the map's uv or of the
        // vertex tangents
        Node m     = shading::slot_texel(&f, 0, shaders::VIEW_TEX_NORMAL).xyz() * 2.0f - 1.0f;
        Node scale = fs_in("tangent_params", VEC2);
        m = nodes::vec3(m.xy() * scale, m.z());
        Node[2] tb = tangent_frame(
            key,
            n,
            v_normal,
            world,
            vtx.tangent,
            shading::slot_d(&f, 0, false),
            shading::slot_d(&f, 0, true)
        );
        n = nodes::normalize(tb[0] * m.x() + tb[1] * m.y() + n * m.z());
    }
    int bump = key.slot(MAP_BUMP);
    if (bump && key.type != (char)MaterialType.BASIC) {
        // three's bumpMap: the normal tilted by how the height changes from pixel to pixel (its
        // perturbNormalArb, Mikkelsen's)
        int bu    = bump + 1;
        int view  = shaders::VIEW_TEX_EXT0 + bump - 1;
        Node buv  = shading::slot_uv(&f, bu);
        Node dx   = shading::slot_d(&f, bu, false);
        Node dy   = shading::slot_d(&f, bu, true);
        Node hll  = shading::slot_grad_at(bu, view, buv, dx, dy).x();
        Node dh   = nodes::vec2(shading::slot_grad_at(bu, view, buv + dx, dx, dy)
            .x() - hll, shading::slot_grad_at(bu, view, buv + dy, dx, dy).x() - hll) * fs_in("surface_params", FLOAT);
        Node sx   = nodes::normalize(nodes::dfdx(world));
        Node sy   = nodes::normalize(nodes::dfdy(world));
        Node r1   = nodes::cross(sy, n);
        Node r2   = nodes::cross(n, sx);
        Node det  = nodes::dot(sx, r1) * nodes::select(front, 1.0f, -1.0f);
        Node grad = (r1 * dh.x() + r2 * dh.y()) * nodes::sign(det);
        n = nodes::normalize(n * nodes::abs(det) - grad);
    }
    if (user[2].id && key.type != (char)MaterialType.BASIC) {
        // Apply normal_node after tangent-space normal mapping and bump mapping.
        n = nodes::normalize(
            bound(nodes::convert(user[2], VEC3), { rebind(NORMAL_WORLD, n), rebind(NORMAL_GEOMETRY, geo_n) })
        );
    }
    if (key.mrt) mrt_normal = nodes::vec4(shading::pack_normal_octahedral(n), 0, 1);
    if (key.type == (char)MaterialType.NORMAL || key.type == (char)MaterialType.DEPTH) {
        // MeshNormalMaterial: the normal in view space as the color; MeshDepthMaterial: the
        // window depth, packed as Material.depth_packing says. Neither is lit, fogged or color
        // managed (three's), and both take the material's opacity alone.
        if (key.type == (char)MaterialType.NORMAL) {
            graph.colors[0] = nodes::vec4(nodes::normalize(shading::to_view(n)) * 0.5f + 0.5f, base.w());
        } else {
            graph.colors[0] = shading::pack_depth(fc.z(), key.depth_packing, base.w());
        }
        if (key.mrt) {
            graph.colors[1] = mrt_normal;
            graph.colors[2] = velocity;
            graph.colors[3] = nodes::vec4(0, 0, rough, metal);
        }
        if (ncuts) graph.fragment = drie::node_block(cuts[:ncuts]);
        return;
    }

    Node camera     = fs_in("camera_pos");
    Node to_cam     = camera.xyz() - world;
    Node depth      = nodes::length(to_cam);
    Node v          = to_cam / nodes::max(depth, 1e-5f);
    Node env_params = fs_in("env_params");

    Node outc;
    // the indirect light in outc, diffuse and specular (for the MRT target's indirect light)
    Node ind_d;
    Node ind_s;
    Node ao = nodes::num(1);
    if (key.arm_ao || key.arm_rough || key.arm_metal) {
        // three's aoMap, roughnessMap and metalnessMap in one (glTF's), or one of the two
        Node arm = shading::slot_texel(&f, 1, shaders::VIEW_TEX_ARM).xyz();
        if (key.arm_ao) ao = arm.x();
        if (key.arm_rough) rough = rough * arm.y();
        if (key.arm_metal) metal = metal * arm.z();
    }
    // three's aoMap and aoMapIntensity
    if (key.slot(MAP_AO)) ao = ao * ((role_texel(key, &f, MAP_AO).x() - 1.0f) * fs_in("ext_params", FLOAT, 1) + 1.0f);
    // Node material surface step: roughness_node, metalness_node, ao_node, emissive_node (in place
    // of material emissive), and Phong specular_node/shininess_node, with NORMAL_WORLD shaded.
    Node user_glow;
    Node phong_specular = fs_in("specular", VEC3);
    Node shininess      = fs_in("mat_params", FLOAT, 3);
    if (build) {
        // (MATERIAL_ROUGHNESS and MATERIAL_METALNESS: the material's own, after its maps)
        Node own_rough = rough;
        Node own_metal = metal;
        Node[4] sb = {
            rebind(NORMAL_WORLD, n),
            rebind(NORMAL_GEOMETRY, geo_n),
            rebind(MATERIAL_ROUGHNESS, own_rough),
            rebind(MATERIAL_METALNESS, own_metal)
        };
        if (user[3].id) rough = nodes::clamp(bound(nodes::convert(user[3], FLOAT), &sb), 0, 1);
        if (user[4].id) metal = nodes::clamp(bound(nodes::convert(user[4], FLOAT), &sb), 0, 1);
        if (user[14].id) ao = ao * nodes::clamp(bound(nodes::convert(user[14], FLOAT), &sb), 0, 1);
        if (user[5].id) user_glow = bound(nodes::convert(user[5], VEC3), &sb);
        if (key.type == (char)MaterialType.PHONG) {
            if (user[10].id) phong_specular = bound(nodes::convert(user[10], VEC3), &sb);
            if (user[11].id) shininess = bound(nodes::convert(user[11], FLOAT), &sb);
        }
    }
    if (key.gbuffer) {
        gbuffer_out(key, graph, &f, vtx, rgb, map_rgb, v_color.xyz(), n, geo_n, rough, metal, ao, user_glow);
        if (ncuts) graph.fragment = drie::node_block(cuts[:ncuts]);
        return;
    }
    Lit lit = { .key = key, .n = n, .geo_n = geo_n, .v = v, .world = world, .rough = rough, .metal = metal, .alpha = alpha,
        .ao = ao, .parallax = px, .f = &f, .v_normal = v_normal, .tangent = vtx.tangent, .irid_fn = -1, .frag = fc.xy(),
        .shadow = shadow_fns(key), .ltc_fn = -1, .refraction = { -1, -1, -1 }, .probe_fn = -1, .ssr_fn = -1,
        .mirror_slot = reflector_slot(0),
        .user_glow = user_glow, .user_env = user[9], .user_cc = user[12], .user_cc_rough = user[13], .phong_specular = phong_specular, .shininess = shininess };
    if (key.type == (char)MaterialType.MATCAP) {
        // MeshMatcapMaterial: the matcap image at the view-space normal, from a frame turned
        // toward the viewer (three's), times the color
        Node vd  = nodes::normalize(shading::to_view(v));
        Node vn  = nodes::normalize(shading::to_view(n));
        Node x   = nodes::normalize(nodes::vec3(vd.z(), 0, vd.x().negate()));
        Node y   = nodes::cross(vd, x);
        Node muv = nodes::vec2(nodes::dot(x, vn), nodes::dot(y, vn)) * 0.495f + 0.5f;
        Node matcap;
        int ms   = key.slot(MAP_MATCAP) - 1;
        if (ms >= 0) {
            matcap = shading::slot_grad_at(ms + 2, shaders::VIEW_TEX_EXT0 + ms, muv, nodes::dfdx(muv), nodes::dfdy(muv))
                .xyz();
        } else {
            matcap = nodes::vec3(nodes::mix(nodes::num(0.2f), nodes::num(0.8f), muv.y()));
        }
        outc = rgb * matcap;
    }
    if (key.type == (char)MaterialType.BASIC || key.type == (char)MaterialType.MATCAP) {
        if (key.type == (char)MaterialType.BASIC) outc = rgb;
        // three's lightMap: the indirect light is the map's instead of 1
        if (key.type == (char)MaterialType.BASIC && key.slot(MAP_LIGHT)) outc = outc * (role_texel(key, &f, MAP_LIGHT)
            .xyz() * (fs_in("light_map_params", FLOAT) * shading::RECIPROCAL_PI));
        if (key.background && key.type == (char)MaterialType.BASIC) {
            // the scene's background: its map in the direction of the view, blurred about as
            // three's PMREM blurs it for backgroundBlurriness
            Node d     = nodes::normalize(world - camera.xyz());
            Node width = nodes::convert(shading::env_size().x(), FLOAT);
            Node lod   = nodes::select(rough.greater_than(nodes::num(0)), shading::env_lod(rough, width), 0.0f);
            Node uv    = shading::env_uv(d);
            Node seen = nodes::select(
                lod.greater_than(nodes::num(0)),
                shading::env_level(uv, lod),
                shading::env_texel_smooth(uv)
            );
            outc = seen.xyz() * env_params.y();
        }
        if (key.glass || key.env_reflect || key.ssr || key.reflector) {
            // glass, the reflection probes' and the mirror images on an unlit material too
            lit.f0 = nodes::mix(nodes::vec3(0.04f), rgb, metal);
            lit.f90 = nodes::num(1);
            lit.lights = outc;
            lit.ind_d = nodes::vec3(0);
            lit.finish();
            outc = lit.outc;
            ind_d = lit.ind_d;
            ind_s = lit.ind_s;
            alpha = lit.alpha;
        }
    } else {
        lit.shade(&f, rgb, map_rgb, v_color.xyz());
        outc = lit.outc;
        ind_d = lit.ind_d;
        ind_s = lit.ind_s;
        alpha = lit.alpha;
    }

    if (key.env_combine) {
        // three's envMap on a basic, lambert or phong material: what it mirrors (or what is
        // seen through it) over all of its light, by its reflectivity times the specular map's red
        Node ec      = fs_in("env_combine");
        Node to_frag = v.negate();
        Node d = nodes::select(ec.z()
            .greater_than(nodes::num(0)), nodes::refract(to_frag, n, ec.z()), nodes::reflect(to_frag, n));
        Node seen = shading::env_classic(d) * env_params.y();
        Node k    = ec.x();
        if (key.slot(MAP_SPECULAR)) k = k * role_texel(key, &f, MAP_SPECULAR).x();
        Node combine = nodes::convert(ec.y() + 0.5f, INT);
        outc = nodes::select(
            combine.equal(nodes::int_(0)),
            nodes::mix(outc, outc * seen, k),
            nodes::select(combine.equal(nodes::int_(1)), nodes::mix(outc, seen, k), outc + seen * k)
        );
    }

    // Material.vertex_occlusion: the vertex colors' brightness dims all of it
    if (key.vertex_occlusion) {
        Node k = nodes::dot(v_color.xyz(), nodes::vec3(0.2126f, 0.7152f, 0.0722f));
        outc = outc * k;
        if (ind_d.id) {
            ind_d = ind_d * k;
            ind_s = ind_s * k;
        }
    }

    // Node material output_node: overrides the lit color and alpha before fog and tone mapping.
    if (user[7].id) {
        Node o = bound(
            user[7],
            { rebind(OUTPUT, nodes::vec4(outc, alpha)), rebind(NORMAL_WORLD, n), rebind(NORMAL_GEOMETRY, geo_n) }
        );
        if (user[7].type() == VEC4) {
            outc = o.xyz();
            alpha = o.w();
        } else {
            outc = nodes::convert(o, VEC3);
        }
    }

    Node fog_color = fs_in("fog_color");
    Node[4] fogged = apply_fog(key, camera, world, v, depth, fc.xy(), outc, ind_d, ind_s);
    outc = fogged[0];
    ind_d = fogged[1];
    ind_s = fogged[2];
    Node display_fog_t = fogged[3];

    outc = nodes::max(outc, nodes::vec3(0));
    if (key.bake) {
        // the light map: the new sample blended into the last map's texel, in linear light
        Node uv1  = nodes::varying(vtx.uv1);
        Node prev = node_sample(SAMPLE, shaders::VIEW_SSR_COLOR, shaders::SMP_SSR_SMP, TEX_2D, uv1).xyz();
        graph.colors[0] = nodes::vec4(
            nodes::mix(prev, nodes::min(outc, nodes::vec3(60000)), fs_in("bake_params", FLOAT, 1)),
            1
        );
        if (ncuts) graph.fragment = drie::node_block(cuts[:ncuts]);
        return;
    }
    Node indirect = nodes::vec4(0, 0, rough, metal);
    if (key.mrt && ind_d.id) {
        Node lw    = nodes::vec3(0.2126f, 0.7152f, 0.0722f);
        Node total = nodes::max(nodes::dot(outc, lw), 1e-6f);
        Node rg = nodes::clamp(
            nodes::vec2(nodes::dot(nodes::max(ind_d, 0), lw), nodes::dot(nodes::max(ind_s, 0), lw)) / total,
            0,
            1
        );
        indirect = nodes::vec4(rg, rough, metal);
    }
    Node shown_alpha = alpha;
    // a light source (emissive far over white) leaves the target's alpha at 0
    Node glow        = fs_in("emissive", VEC3);
    alpha = alpha * nodes::smoothstep(4, 16, nodes::max(glow.x(), nodes::max(glow.y(), glow.z()))).one_minus();
    Node shown = encode_output(key, outc, camera.w(), fog_color.xyz(), display_fog_t);
    Node color = nodes::vec4(shown, alpha);
    // Material.premultiplied_alpha: the color as it is shown, times its opacity
    if (key.premultiplied) color = nodes::vec4(color.xyz() * shown_alpha, color.w());
    graph.colors[0] = color;
    if (key.mrt) {
        graph.colors[1] = mrt_normal;
        graph.colors[2] = velocity;
        graph.colors[3] = indirect;
    }
    if (ncuts) graph.fragment = drie::node_block(cuts[:ncuts]);
}

// Applies volumetric or analytic fog (with optional height falloff, sun scattering, shafts, and
// light glow) to `outc`, `ind_d`, and `ind_s`. Returns `{ outc, ind_d, ind_s, display_fog_t }`.
fn Node[4] apply_fog(
    MaterialKey key,
    Node camera,
    Node world,
    Node v,
    Node depth,
    Node frag,
    Node outc,
    Node ind_d,
    Node ind_s
) @private {
    Node fog_color = fs_in("fog_color");
    Node display_fog_t;
    if (key.fog_volume) {
        // volumetric fog: what it scatters towards the camera in front of the point, and how
        // much of the point gets through
        Node fv = fog_volume_at(world);
        outc = outc * fv.w() + fv.xyz();
        if (ind_d.id) {
            ind_d = ind_d * fv.w();
            ind_s = ind_s * fv.w();
        }
    } else if (key.fog) {
        Node fd      = shading::fog_distance(key, camera.xyz(), world);
        Node fog_own = fog_color.xyz();
        Node fh      = fs_in("fog_height");
        if (key.fog_height) {
            // fog lying on the ground: the mean of exp(-k * height) along the ray
            Node ya = nodes::max(camera.y() - fh.x(), 0);
            Node yb = nodes::max(world.y() - fh.x(), 0);
            Node k  = fh.y();
            Node dy = yb - ya;
            Node ea = nodes::exp((k * ya).negate());
            fd = fd * nodes::select(nodes::abs(dy)
                .greater_than(nodes::num(0.01f)), (ea - nodes::exp((k * yb).negate())) / (k * dy), ea);
        }
        if (key.fog_sun) {
            Node sun  = fs_in("fog_sun_dir");
            Node look = nodes::normalize(world - camera.xyz());
            Node side = nodes::dot(
                nodes::normalize(look.xz() + nodes::vec2(1e-5f)),
                nodes::normalize(sun.xz() + nodes::vec2(1e-5f))
            );
            fd = fd * nodes::mix(fh.w(), 1, nodes::smoothstep(-0.3f, 0.7f, side));
            Node sunlit = nodes::pow(nodes::max(nodes::dot(look, sun.xyz()), 0), sun.w());
            if (key.fog_shafts) {
                // light shafts: the sun's part of the fog only where the sun gets to it
                ShadowFns shadow = shadow_fns(key);
                Node len         = nodes::length(world - camera.xyz());
                Node in_sun = sun_in_fog(
                    &shadow,
                    camera.xyz(),
                    world,
                    fs_in("fog_params", FLOAT, 2) * fd / nodes::max(len, 1e-4f),
                    frag
                );
                sunlit = sunlit * nodes::mix(nodes::num(1), in_sun, fs_in("fog_sun_color", FLOAT, 3));
            }
            fog_own = nodes::mix(fog_own, fs_in("fog_sun_color", VEC3), sunlit);
        }
        Node t = shading::fog_transmittance(key, fd);
        // (Renderer.display_fog: mixed in after tone mapping, below)
        if (key.display_fog) {
            display_fog_t = t;
        } else {
            Node fog_lit = fog_own * t.one_minus();
            if (key.fog_glow) {
                // the lights' glow in the fog along the view ray
                Node glow = fog_lights_glow(key, v.negate(), depth) * fog_color.w() * (fd / nodes::max(depth, 1e-4f));
                fog_lit = fog_lit + nodes::select(fd.greater_than(nodes::num(0)), glow, nodes::vec3(0));
            }
            outc = outc * t + fog_lit;
        }
        if (ind_d.id) {
            ind_d = ind_d * t;
            ind_s = ind_s * t;
        }
    }
    return { outc, ind_d, ind_s, display_fog_t };
}

// Clamps linear HDR targets or tone-maps and encodes SDR/HDR display output, mixing display-space
// fog when `display_fog_t` is present.
fn Node encode_output(MaterialKey key, Node outc, Node exposure, Node fog_color, Node display_fog_t) @private {
    if (key.linear) {
        Node shown = nodes::min(outc, nodes::vec3(60000));
        if (display_fog_t.id) shown = nodes::mix(fog_color, shown, display_fog_t);
        return shown;
    }
    // (three mixes the fog in the colors as they are shown)
    Node shown = shading::tone_map_gamut(outc, key.tone, exposure, key.display != shading::DISPLAY_SRGB);
    bool hdr   = key.display == shading::DISPLAY_HDR;
    if (!hdr) shown = shading::linear_to_srgb(shown);
    if (display_fog_t.id) shown = nodes::mix(
        hdr ? fog_color : shading::linear_to_srgb(fog_color),
        shown,
        display_fog_t
    );
    return shown;
}

// Deferred G-buffer's four targets (MaterialKey.gbuffer, unpacked by `build_deferred_graph`):
// albedo and AO; octahedral shaded normal, roughness, and metalness; self-emitted light and
// lighting-pass flags; motion vectors and octahedral geometry normal.
fn void gbuffer_out(
    MaterialKey key,
    ProgramGraph* graph,
    FragUv* f,
    VertexValues vtx,
    Node rgb,
    Node map_rgb,
    Node v_color,
    Node n,
    Node geo_n,
    Node rough,
    Node metal,
    Node ao,
    Node user_glow
) @local {
    int mode      = key.type;
    Node diffuse  = mode == (int)MaterialType.STANDARD ? rgb * metal.one_minus() : rgb;
    Node emissive = fs_in("emissive");
    Node glow     = emissive.xyz();
    if (key.map_emissive && map_rgb.id) glow = glow * map_rgb;
    if (key.slot(MAP_EMISSIVE)) glow = glow * role_texel(key, f, MAP_EMISSIVE).xyz();
    if (user_glow.id) glow = user_glow;
    Node own = glow;
    if (key.baked) own = own + diffuse * v_color * emissive.w() * ao;
    if (key.slot(MAP_LIGHT)) {
        Node lm = role_texel(key, f, MAP_LIGHT).xyz() * (fs_in("light_map_params", FLOAT) * shading::RECIPROCAL_PI);
        if (mode == (int)MaterialType.BASIC) {
            rgb = rgb * lm;
        } else {
            own = own + lm * diffuse * ao;
        }
    }
    if (key.vertex_occlusion) {
        Node k = nodes::dot(v_color, nodes::vec3(0.2126f, 0.7152f, 0.0722f));
        rgb = rgb * k;
        own = own * k;
    }
    Node vel = nodes::vec2(0);
    if (key.motion) {
        Node clip = nodes::varying(vtx.clip);
        Node prev = nodes::varying(vtx.prev_clip);
        vel = (clip.xy() / clip.w() - prev.xy() / nodes::max(prev.w(), 1e-5f)) * 0.5f;
        vel = nodes::select(fs_in("ssr_params", FLOAT, 3)
            .greater_than(nodes::num(0.5f)), nodes::vec2(vel.x(), vel.y().negate()), vel);
    }
    // (the flags: the material type, receives shadows 8, foliage 16, SSR 32, the environment's
    // light 64, fog 128, a light source 256)
    Node flags = nodes::convert(fs_in("specular", FLOAT, 3) + 0.5f, INT);
    Node pf    = nodes::num((float)(mode + (key.foliage ? 16 : 0)));
    pf = pf + nodes::select(node_binary(BIT_AND, flags, nodes::int_(4)).not_equal(nodes::int_(0)), 8.0f, 0.0f);
    pf = pf + nodes::select(fs_in("ssr_params", FLOAT).greater_than(nodes::num(0)), 32.0f, 0.0f);
    if (mode != (int)MaterialType.LAMBERT) pf = pf + nodes::select(fs_in("env_params", FLOAT)
        .greater_than(nodes::num(0.5f)), 64.0f, 0.0f);
    pf = pf + nodes::select(fs_in("fog_params", FLOAT, 3).not_equal(nodes::num(0)), 128.0f, 0.0f);
    Node bright = nodes::max(emissive.x(), nodes::max(emissive.y(), emissive.z())).greater_than(nodes::num(4));
    pf = pf + nodes::select(bright, 256.0f, 0.0f);
    graph.colors[0] = nodes::vec4(rgb, ao);
    graph.colors[1] = nodes::vec4(shading::pack_normal_octahedral(n), rough, metal);
    graph.colors[2] = nodes::vec4(own, pf);
    graph.colors[3] = nodes::vec4(vel, shading::pack_normal_octahedral(geo_n));
}

// Whether the key holds a slot for any map role.
fn bool MaterialKey.has_maps(self) => (bitcast(self, KeyWords).hi >> 64) != 0;

// Samples map role `role` at the material's UV.
fn Node role_texel(MaterialKey key, FragUv* f, int role) @private => shading::ext_texel(f, key.slot(role) - 1);