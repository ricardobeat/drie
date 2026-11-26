// Deferred fullscreen lighting pass node-graph builder (`build_deferred_graph`).
//
// Generates a fullscreen triangle from `VERTEX_ID` and unpacks the four RGBA16F G-Buffer targets
// plus the scene depth texture (`VIEW_TEX_TRANSMISSION`, unfilterable float). Reconstructs
// camera-relative world position and ray-plane `dpdx`/`dpdy` derivatives, then shares `Lit`,
// `ShadowFns`, `apply_fog`, and `encode_output` with forward materials so forward and deferred
// shading use a single implementation. Invoked by `material_program()` (src/material/material_key.c3)
// when `MaterialKey.deferred_light` is set.
module drie;
import drie::nodes, drie::shading;

// Builds the vertex and fragment graphs for the deferred fullscreen lighting pass (`key.deferred_light`).
fn void build_deferred_graph(MaterialKey key, ProgramGraph* graph) @private {
    // Fullscreen triangle covering clip space from vertex index 0..2 without vertex buffers.
    Node vid = node_system(VERTEX_ID);
    Node px  = nodes::convert(node_binary(BIT_AND, node_binary(SHL, vid, nodes::uint_(1)), nodes::uint_(2)), FLOAT);
    Node py  = nodes::convert(node_binary(BIT_AND, vid, nodes::uint_(2)), FLOAT);
    graph.position = nodes::vec4(px * 2.0f - 1.0f, py * 2.0f - 1.0f, 0, 1);

    Node fc = node_system(FRAG_COORD);
    Node tc = nodes::convert(fc.xy(), IVEC2);
    Node raw_d = node_sample(
        FETCH,
        shaders::VIEW_TEX_TRANSMISSION,
        shaders::SMP_TRANSMISSION_SMP,
        TEX_UNFILTERABLE,
        tc,
        nodes::int_(0)
    )
        .x();

    Node out_c0 = nodes::variable(deferred_in("clear_color"));
    Node out_c1;
    Node out_c2;
    Node out_c3;
    Node[6] root;
    int nr;
    root[nr++] = out_c0;
    if (key.mrt) {
        out_c1 = nodes::variable(nodes::vec4(0));
        out_c2 = nodes::variable(nodes::vec4(0));
        out_c3 = nodes::variable(nodes::vec4(0));
        root[nr++] = out_c1;
        root[nr++] = out_c2;
        root[nr++] = out_c3;
    }

    // Reconstruct world position from depth and camera-relative inverse view-projection.
    Node sinfo   = deferred_in("screen_info");
    Node origin  = deferred_in("view_origin");
    Node ivp     = deferred_in("inv_view_proj", MAT4);
    Node su      = fc.xy() * sinfo.xy();
    Node ndc_raw = su * 2.0f - 1.0f;
    Node ndc_xy = nodes::select(sinfo.z()
        .greater_than(nodes::num(0.5f)), nodes::vec2(ndc_raw.x(), ndc_raw.y().negate()), ndc_raw);
    Node ndc_z = nodes::select(origin.w().greater_than(nodes::num(0.5f)), raw_d, raw_d * 2.0f - 1.0f);
    Node rel   = ivp * nodes::vec4(ndc_xy, ndc_z, 1);
    Node world = rel.xyz() / rel.w() + origin.xyz();

    // Unpack G-Buffer attachments (gbuffer_out in src/material/material_graph.c3).
    Node g0 = node_sample(FETCH, shaders::VIEW_TEX_MAP, shaders::SMP_TEX_SMP, TEX_2D, tc, nodes::int_(0));
    Node g1 = node_sample(FETCH, shaders::VIEW_TEX_NORMAL, shaders::SMP_TEX_SMP, TEX_2D, tc, nodes::int_(0));
    Node g2 = node_sample(FETCH, shaders::VIEW_TEX_ARM, shaders::SMP_TEX_SMP, TEX_2D, tc, nodes::int_(0));
    Node g3 = node_sample(FETCH, shaders::VIEW_TEX_PARALLAX, shaders::SMP_TEX_SMP, TEX_2D, tc, nodes::int_(0));

    Node pf         = nodes::convert(g2.w() + 0.5f, INT);
    Node mode       = node_binary(BIT_AND, pf, nodes::int_(7));
    Node df_shadow  = node_binary(BIT_AND, pf, nodes::int_(8)).not_equal(nodes::int_(0));
    Node df_foliage = node_binary(BIT_AND, pf, nodes::int_(16)).not_equal(nodes::int_(0));
    Node df_ssr     = node_binary(BIT_AND, pf, nodes::int_(32)).not_equal(nodes::int_(0));
    Node df_env     = node_binary(BIT_AND, pf, nodes::int_(64)).not_equal(nodes::int_(0));
    Node df_fog     = node_binary(BIT_AND, pf, nodes::int_(128)).not_equal(nodes::int_(0));
    Node df_bright  = node_binary(BIT_AND, pf, nodes::int_(256)).not_equal(nodes::int_(0));

    Node rgb   = g0.xyz();
    Node ao    = g0.w();
    Node n     = shading::unpack_normal_octahedral(g1.xy());
    Node geo_n = shading::unpack_normal_octahedral(g3.swizzle("zw"));
    Node rough = g1.z();
    Node metal = g1.w();

    // Intersect adjacent pixel rays with the geometry-normal plane so shadow footprints match
    // forward `dFdx(world)` without blowing up across silhouette edges.
    Node eye     = origin.xyz();
    Node rd      = world - eye;
    Node step_y  = nodes::select(sinfo.z().greater_than(nodes::num(0.5f)), -1.0f, 1.0f);
    Node step_px = sinfo.xy() * nodes::vec2(2.0f, 2.0f * step_y);
    Node qx      = ivp * nodes::vec4(ndc_xy + nodes::vec2(step_px.x(), 0), ndc_z, 1);
    Node qy      = ivp * nodes::vec4(ndc_xy + nodes::vec2(0, step_px.y()), ndc_z, 1);
    Node rx      = qx.xyz() / qx.w();
    Node ry      = qy.xyz() / qy.w();
    Node face    = nodes::dot(geo_n, rd);
    Node fx      = nodes::dot(geo_n, rx);
    Node fy      = nodes::dot(geo_n, ry);
    Node dpdx    = nodes::select(nodes::abs(fx).greater_than(nodes::abs(face) * 0.05f), rx * (face / fx) - rd, rx - rd);
    Node dpdy    = nodes::select(nodes::abs(fy).greater_than(nodes::abs(face) * 0.05f), ry * (face / fy) - rd, ry - rd);

    Node camera = fs_in("camera_pos");
    Node to_cam = camera.xyz() - world;
    Node depth  = nodes::length(to_cam);
    Node v      = to_cam / nodes::max(depth, 1e-5f);

    Node outc_v  = nodes::variable(rgb);
    Node ind_d_v = nodes::variable(nodes::vec3(0));
    Node ind_s_v = nodes::variable(nodes::vec3(0));
    Node display_fog_t;

    Node[16] body;
    int nb;
    body[nb++] = outc_v;
    body[nb++] = ind_d_v;
    body[nb++] = ind_s_v;

    Lit lit = {
        .key = key,
        .n = n,
        .geo_n = geo_n,
        .frag = fc.xy(),
        .dpdx = dpdx,
        .dpdy = dpdy,
        .shadow = shadow_fns(key),
        .v = v,
        .world = world,
        .rough = rough,
        .metal = metal,
        .ao = ao,
        .alpha = nodes::num(1),
        .ltc_fn = -1,
        .irid_fn = -1,
        .probe_fn = -1,
        .ssr_fn = -1,
        .refraction = { -1, -1, -1 },
        .df_mode = mode,
        .df_shadow = df_shadow,
        .df_foliage = df_foliage,
        .df_ssr = df_ssr,
        .df_env = df_env,
        .df_own = g2.xyz(),
    };
    lit.shade(null, rgb, { }, { });
    Node lit_block = nodes::block(outc_v.assign(lit.outc), ind_d_v.assign(lit.ind_d), ind_s_v.assign(lit.ind_s));
    if (key.env_reflect || key.ssr) {
        // Basic materials (mode 0) skip direct lighting but still participate in probes and SSR.
        Lit basic_lit = {
            .key = key,
            .n = n,
            .geo_n = geo_n,
            .frag = fc.xy(),
            .v = v,
            .world = world,
            .rough = rough,
            .metal = metal,
            .ao = ao,
            .f0 = nodes::mix(nodes::vec3(0.04f), rgb, metal),
            .f90 = nodes::num(1),
            .lights = rgb,
            .alpha = nodes::num(1),
            .ltc_fn = lit.ltc_fn,
            .irid_fn = lit.irid_fn,
            .probe_fn = lit.probe_fn,
            .ssr_fn = lit.ssr_fn,
            .refraction = { -1, -1, -1 },
            .df_mode = mode,
            .df_shadow = df_shadow,
            .df_foliage = df_foliage,
            .df_ssr = df_ssr,
            .df_env = df_env,
        };
        basic_lit.finish();
        Node basic_block = nodes::block(outc_v.assign(basic_lit.outc), ind_s_v.assign(basic_lit.ind_s));
        body[nb++] = nodes::if_then(mode.not_equal(nodes::int_(0)), lit_block, basic_block);
    } else {
        body[nb++] = nodes::if_then(mode.not_equal(nodes::int_(0)), lit_block);
    }

    if (key.fog_volume || key.fog) {
        if (key.display_fog) {
            display_fog_t = nodes::variable(nodes::num(1));
            body[nb++] = display_fog_t;
        }
        Node[4] fogged = apply_fog(key, camera, world, v, depth, fc.xy(), outc_v, ind_d_v, ind_s_v);
        Node[4] fst;
        int nfst;
        fst[nfst++] = outc_v.assign(fogged[0]);
        fst[nfst++] = ind_d_v.assign(fogged[1]);
        fst[nfst++] = ind_s_v.assign(fogged[2]);
        if (fogged[3].id) fst[nfst++] = display_fog_t.assign(fogged[3]);
        body[nb++] = nodes::if_then(df_fog, drie::node_block(fst[:nfst]));
    }

    Node clamped = nodes::max(outc_v, nodes::vec3(0));
    Node alpha   = nodes::select(df_bright, 0.0f, 1.0f);
    Node shown   = encode_output(key, clamped, camera.w(), fs_in("fog_color").xyz(), display_fog_t);
    body[nb++] = out_c0.assign(nodes::vec4(shown, alpha));
    if (key.mrt) {
        Node lw    = nodes::vec3(0.2126f, 0.7152f, 0.0722f);
        Node total = nodes::max(nodes::dot(clamped, lw), 1e-6f);
        Node rg = nodes::clamp(
            nodes::vec2(nodes::dot(nodes::max(ind_d_v, 0), lw), nodes::dot(nodes::max(ind_s_v, 0), lw)) / total,
            0,
            1
        );
        body[nb++] = out_c1.assign(nodes::vec4(g1.xy(), 0, 1));
        body[nb++] = out_c2.assign(nodes::vec4(g3.xy(), 0, 1));
        body[nb++] = out_c3.assign(nodes::vec4(rg, rough, metal));
    }

    root[nr++] = nodes::if_then(raw_d.less_than(nodes::num(1)), drie::node_block(body[:nb]));
    graph.fragment = drie::node_block(root[:nr]);
    graph.colors[0] = out_c0;
    if (key.mrt) {
        graph.colors[1] = out_c1;
        graph.colors[2] = out_c2;
        graph.colors[3] = out_c3;
    }
}