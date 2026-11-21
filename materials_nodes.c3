// The built-in materials as node graphs (three's NodeMaterial.setup), printed into programs
// that hold only what a material and the render it is drawn in use (nodes_program.c3).
//
// A program is chosen by a MaterialKey: the features that change its code, worked out from
// the same conditions the engine's material shader (shader_core.glsl) branches on, so that a
// generated program draws what that shader draws. What only changes numbers stays in the
// uniforms: the programs read the fields of FsParams and VsParams that they need (fs_in,
// vs_in), and a draw gathers them from the structs the renderer fills for every draw.
module drie;
import sokol, drie::nodes, drie::shading, std::collections::map, std::io, std::os::env;

// ---- the engine's uniforms, as the programs read them

// The row (16 bytes) of FsParams or VsParams where field `$name` starts.
macro int fs_row($name) {
    FsParams* p = null;
    return (int)((usz)&p.$eval($name) / 16);
}

macro int vs_row($name) {
    VsParams* p = null;
    return (int)((usz)&p.$eval($name) / 16);
}

// Field `$name` of FsParams (VsParams) as a `t`, from component `comp` of its row; `k` rows on
// for an element of an array.
macro Node fs_in($name, NodeType t = VEC4, int comp = 0, int k = 0) => node_input(FS, fs_row($name) + k, t, comp);
macro Node vs_in($name, NodeType t = VEC4, int comp = 0, int k = 0) => node_input(VS, vs_row($name) + k, t, comp);

// ---- the key

// What a material's program holds. Slots are the shared map slot (tex_ext0..5) a role's map
// is in, + 1 (0: the material has none).
bitstruct MaterialKey : ulong {
    char type : 0..3;            // MaterialType's ordinal
    bool mrt : 4;                // a target with MRT: normal, motion and indirect light too
    bool cut : 5;                // DrawKind.CUT: the program may cut holes
    bool cull_box : 6;           // Renderer.cull_box
    bool dash : 7;               // a dashed line
    bool dither : 8;             // an LOD's cross-fade
    bool screen : 9;             // Material.screen_target
    bool tint : 10;              // the vertex colors tint the color (not baked light)
    bool vertex_alpha : 11;      // the vertex colors' alpha
    bool foliage : 12;
    bool map : 13;
    bool alpha_test : 14;
    bool background : 15;        // the scene's background
    bool env_combine : 16;       // three's envMap on a basic, lambert or phong material
    bool vertex_occlusion : 17;
    bool premultiplied : 18;
    char fog : 19..20;           // the fog's mode (0: none)
    bool fog_box : 21;
    bool fog_height : 22;
    bool fog_sun : 23;
    bool display_fog : 24;       // Renderer.display_fog
    bool linear : 25;            // linear light out (a half-float target)
    char tone : 26..28;          // ToneMapping's ordinal (0 when linear or not tone mapped)
    char display : 29..30;       // the window's colors (DISPLAY_*)
    bool gpu_cull : 31;          // instances culled on the GPU
    bool displacement : 32;
    bool sway : 33;
    bool outline : 34;
    bool motion : 35;            // motion vectors (with mrt)
    char alpha_slot : 36..38;
    char light_slot : 39..41;
    char specular_slot : 42..44;
}

// A generated program and its pipelines (by get_pipeline's key).
struct MaterialProgram {
    ShaderProgram* program;
    HashMap{ulong, SgPipeline} pipelines;
}

HashMap{ulong, MaterialProgram*} material_programs @local;

// Set: every material draws with the material shader (the DRIE_CLASSIC_MATERIALS environment
// variable, for comparing the two).
bool classic_materials @private;

// The program for `key`, built the first time it is asked for.
fn MaterialProgram* material_program(MaterialKey key) @private {
    if (!material_programs.is_initialized()) material_programs.init(mem);
    if (try found = material_programs.get((ulong)key)) return found;
    uint mark = node_count();
    ProgramGraph graph;
    build_material_graph(key, &graph);
    MaterialProgram* mp = mem::new(MaterialProgram);
    // (DRIE_PROGRAM_SALT: a term the output depends on, at a pixel no target reaches, so the
    // compiled code differs from any earlier run's and the driver has none of it cached; for
    // timing a cold start)
    if (try salt = env::tget_var("DRIE_PROGRAM_SALT")) {
        float far = 1e7f + (float)(salt.hash() & 0xffff);
        Node off = nodes::select(node_system(FRAG_COORD).x().greater_than(nodes::num(far)), nodes::num(1e-20f), nodes::num(0));
        graph.colors[0] = graph.colors[0] + nodes::vec4(off, 0, 0, 0);
    }
    mp.program = build_program(backend_lang(), &graph);
    mp.pipelines.init(mem);
    node_release(mark);
    material_programs.set((ulong)key, mp);
    // (DRIE_TRACE_PROGRAMS: each program printed as it is built)
    if (try env::tget_var("DRIE_TRACE_PROGRAMS")) io::eprintfn("drie: program %x: vs %d B, fs %d B\n%s\n%s", (ulong)key, mp.program.vs.len, mp.program.fs.len, mp.program.vs, mp.program.fs);
    return mp;
}

// The bindings of the classic slots (shaders::VIEW_*, SMP_*) as `p` takes them.
fn SgBindings program_bindings(ShaderProgram* p, SgBindings* classic) @private {
    SgBindings b = {
        .vertex_buffers = classic.vertex_buffers,
        .vertex_buffer_offsets = classic.vertex_buffer_offsets,
        .index_buffer = classic.index_buffer,
        .index_buffer_offset = classic.index_buffer_offset,
    };
    foreach (i, v : p.views) b.views[i] = classic.views[v.source];
    foreach (i, s : p.samplers) b.samplers[i] = classic.samplers[s.source];
    return b;
}

// Applies `p`'s uniform block for stage `st`, gathered from the engine's.
fn void apply_program_uniforms(ShaderProgram* p, ProgramStage st, VsParams* vs, FsParams* fs) @private {
    int rows = p.block_rows[st.ordinal];
    if (!rows) return;
    void*[NodeSource::len] src;
    sz[NodeSource::len] sizes;
    src[NodeSource.VS.ordinal] = vs;
    sizes[NodeSource.VS.ordinal] = VsParams::size;
    src[NodeSource.FS.ordinal] = fs;
    sizes[NodeSource.FS.ordinal] = FsParams::size;
    @pool() {
        Vec4[] block = mem::temp_array(Vec4, rows);
        p.gather(st, src, sizes, block);
        sg::apply_uniforms(st.ordinal, &&{ .ptr = block.ptr, .size = (usz)rows * 16 });
    };
}

// ---- the vertex stage

// What the vertex stage works out (its nodes): the fragment stage reads them as varyings.
struct VertexValues @local {
    Node position;       // the clip position (a SEQ: the statements before it)
    Node world;
    Node normal;
    Node uv;
    Node uv1;
    Node color;          // the vertex color, rgba
    Node instance_color;
    Node line_distance;
    Node clip;
    Node prev_clip;
}

// The instance's matrix rows, as fetch_instance picks them.
struct Instance @local {
    Node m0;
    Node m1;
    Node m2;
}

fn Node Instance.point(&self, Node p) {
    Node l = nodes::vec4(p, 1);
    return nodes::vec3(nodes::dot(self.m0, l), nodes::dot(self.m1, l), nodes::dot(self.m2, l));
}

fn Node Instance.dir(&self, Node d) {
    return nodes::vec3(nodes::dot(self.m0.xyz(), d), nodes::dot(self.m1.xyz(), d), nodes::dot(self.m2.xyz(), d));
}

fn VertexValues vertex_stage(MaterialKey key) @local {
    VertexValues o;
    Node position = node_attribute(shaders::ATTR_THREE_POSITION, VEC3);
    Node normal = node_attribute(shaders::ATTR_THREE_NORMAL, VEC3);
    Node uv = node_attribute(shaders::ATTR_THREE_UV, VEC2);
    Node color0 = node_attribute(shaders::ATTR_THREE_COLOR0, VEC4);
    Node statements;
    Instance inst;
    if (key.gpu_cull) {
        // (fetch_instance: instance i of the draw is rows[3 * visible[first + i]], if i is
        // under the cull's survivors)
        Node cull = vs_in("inst_cull");
        Node i = node_system(INSTANCE_ID);
        Node count = node_storage(shaders::VIEW_INST_COUNTS, UINT, nodes::convert(cull.z(), UINT));
        statements = nodes::block(nodes::if_then(i.greater_than_equal(count), nodes::block(node_return(nodes::vec4(2, 2, 2, 1)))));
        Node id = node_storage(shaders::VIEW_INST_VISIBLE, UINT, nodes::convert(cull.y(), UINT) + i);
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
    }
    Node local = position;
    if (key.displacement) {
        Node d = vs_in("surface_vs");
        Node height = node_attribute(shaders::ATTR_THREE_HEIGHT, FLOAT);
        local = local + nodes::normalize(normal) * (height * d.y() + d.z());
    }
    Node ip = inst.point(local);
    Node model = vs_in("model", MAT4);
    Node wp = model * nodes::vec4(ip, 1);
    Node world = wp.xyz();
    Node sway = vs_in("sway");
    if (key.sway) world = shading::apply_sway(world, ip.y(), sway);
    Node view_proj = vs_in("view_proj", MAT4);
    Node origin = vs_in("view_origin", VEC3);
    Node pos = view_proj * nodes::vec4(world - origin, 1);
    Node dir = inst.dir(normal);
    if (key.outline) {
        // (Material.outline: out along the normal as the screen sees it)
        Node inward = world - (model * nodes::vec4(dir, 0)).xyz();
        Node p2 = view_proj * nodes::vec4(inward - origin, 1);
        pos = pos + nodes::normalize(pos - p2) * (vs_in("surface_vs", FLOAT) * pos.w());
    }
    o.position = statements.id ? nodes::seq(statements, pos) : pos;
    o.world = world;
    o.normal = (vs_in("normal_mat", MAT4) * nodes::vec4(dir, 0)).xyz();
    Node t = vs_in("uv_transform");
    Node t1 = vs_in("uv_transform", k: 1);
    Node uvh = nodes::vec3(uv, 1);
    o.uv = nodes::vec2(nodes::dot(t.xyz(), uvh), nodes::dot(t1.xyz(), uvh));
    o.uv1 = node_attribute(shaders::ATTR_THREE_UV1, VEC2);
    o.color = color0;
    o.instance_color = node_attribute(shaders::ATTR_THREE_INSTANCE_COLOR, VEC3);
    o.line_distance = normal.x();
    if (key.motion) {
        Node motion = vs_in("motion");
        o.clip = nodes::vec4(pos.xy() - motion.swizzle("zw") * pos.w(), pos.z(), pos.w());
        Node prev_view_proj = vs_in("prev_view_proj", MAT4);
        Node pp = vs_in("prev_model", MAT4) * nodes::vec4(ip, 1);
        if (key.sway) pp = nodes::vec4(shading::apply_sway(pp.xyz(), ip.y(), nodes::vec4(motion.y(), sway.y(), sway.z(), sway.w())), pp.w());
        Node swayed = nodes::vec4(world, wp.w());
        o.prev_clip = nodes::select(motion.x().greater_than(nodes::num(1.5f)), prev_view_proj * pp,
                                    nodes::select(motion.x().greater_than(nodes::num(0.5f)), prev_view_proj * swayed, o.clip));
    }
    return o;
}

// ---- the material

fn void build_material_graph(MaterialKey key, ProgramGraph* graph) @private {
    VertexValues vtx = vertex_stage(key);
    graph.position = vtx.position;

    Node world = nodes::varying(vtx.world);
    Node v_normal = nodes::varying(vtx.normal);
    Node v_uv = nodes::varying(vtx.uv);
    Node fc = node_system(FRAG_COORD);
    Node front = node_system(FRONT_FACING);
    Node[8] cuts;
    int ncuts;

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
            Node now = clip.xy() / clip.w();
            Node was = prev.xy() / nodes::max(prev.w(), 1e-5f);
            Node d = (now - was) * 0.5f;
            d = nodes::select(fs_in("ssr_params", FLOAT, 3).greater_than(nodes::num(0.5f)), nodes::vec2(d.x(), d.y().negate()), d);
            velocity = nodes::vec4(d, 0, 1);
        } else {
            velocity = nodes::vec4(0, 0, 0, 1);
        }
    }

    // the holes cut whatever the material (cull box, dashes, LOD cross-fade)
    if (key.cull_box) {
        Node lo = fs_in("cull_min");
        Node hi = fs_in("cull_max");
        Node inside = lo.w().greater_than(nodes::num(0.5f));
        for (int i = 0; i < 3; i++) {
            String c = "xyz"[i:1];
            inside = nodes::logical_and(inside, world.swizzle(c).greater_than(lo.swizzle(c)));
            inside = nodes::logical_and(inside, world.swizzle(c).less_than(hi.swizzle(c)));
        }
        cuts[ncuts++] = nodes::if_then(inside, nodes::block(nodes::discard()));
    }
    if (key.dash) {
        Node dash = fs_in("line_dash");
        Node along = nodes::varying(vtx.line_distance);
        Node gap = nodes::logical_and(dash.w().greater_than(nodes::num(0.5f)),
                                      nodes::mod(along * dash.z(), dash.y()).greater_than(dash.x()));
        cuts[ncuts++] = nodes::if_then(gap, nodes::block(nodes::discard()));
    }
    if (key.dither) {
        Node sp = fs_in("screen_params");
        Node dither = nodes::fract(nodes::fract(nodes::dot(fc.xy(), nodes::vec2(0.06711056f, 0.00583715f))) * 52.9829189f);
        Node out = nodes::select(sp.w().greater_than(nodes::num(0)), dither.greater_than_equal(sp.z()), dither.less_than(sp.z()));
        cuts[ncuts++] = nodes::if_then(nodes::logical_and(sp.w().not_equal(nodes::num(0)), out), nodes::block(nodes::discard()));
    }

    if (key.screen) {
        // a window onto another render: its pixels as they are, already tone mapped
        Node at = fc.xy() * fs_in("screen_params", VEC2);
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
    Node base = fs_in("color");
    Node rgb = base.xyz();
    if (key.tint) rgb = rgb * v_color.xyz();
    rgb = rgb * nodes::varying(vtx.instance_color);
    Node a = base.w();
    FragUv f = { .uv = v_uv, .dx = nodes::dfdx(v_uv), .dy = nodes::dfdy(v_uv) };
    if (key.alpha_slot || key.light_slot || key.specular_slot) {
        f.uv1 = nodes::varying(vtx.uv1);
        f.dx1 = nodes::dfdx(f.uv1);
        f.dy1 = nodes::dfdy(f.uv1);
    }
    if (key.map) {
        Node texel = node_sample(GRAD, shaders::VIEW_TEX_MAP, shaders::SMP_TEX_SMP, TEX_2D, f.uv, f.dx, f.dy);
        rgb = rgb * texel.xyz();
        a = a * texel.w();
    }
    // three's alphaMap: its green channel
    if (key.alpha_slot) a = a * shading::ext_texel(&f, key.alpha_slot - 1).y();
    Node alpha = a;
    if (key.vertex_alpha) alpha = alpha * v_color.w();
    if (key.alpha_test) {
        Node threshold = fs_in("tex_params", FLOAT, 3);
        if (key.map) {
            // far away the texture's smaller levels average leaves and gaps into an alpha under
            // the cutoff: made up for, so as much stays covered
            Node texel = v_uv * nodes::convert(node_sample(SIZE, shaders::VIEW_TEX_MAP, shaders::SMP_TEX_SMP, TEX_2D, { }), VEC2);
            Node tx = nodes::dfdx(texel);
            Node ty = nodes::dfdy(texel);
            Node level = nodes::log2(nodes::max(nodes::max(nodes::dot(tx, tx), nodes::dot(ty, ty)), 1)) * 0.5f;
            alpha = alpha * (nodes::min(level, 6) * 0.3f + 1.0f);
        }
        // the cutout edge sharpened to about a pixel, for alpha-to-coverage
        alpha = (alpha - threshold) / nodes::max(nodes::fwidth(alpha), 1e-4f) + 0.5f;
        cuts[ncuts++] = nodes::if_then(alpha.less_than_equal(nodes::num(0)), nodes::block(nodes::discard()));
        alpha = nodes::min(alpha, 1);
    }

    Node n = nodes::normalize(v_normal);
    if (!key.foliage) n = nodes::select(front, n, n.negate());
    if (key.mrt) mrt_normal = nodes::vec4(shading::pack_normal_octahedral(n), 0, 1);

    Node camera = fs_in("camera_pos");
    Node to_cam = camera.xyz() - world;
    Node depth = nodes::length(to_cam);
    Node v = to_cam / nodes::max(depth, 1e-5f);
    Node env_params = fs_in("env_params");

    Node outc = rgb;
    // three's lightMap: the indirect light is the map's instead of 1
    if (key.light_slot) outc = outc * (shading::ext_texel(&f, key.light_slot - 1).xyz() * (fs_in("light_map_params", FLOAT) * shading::RECIPROCAL_PI));
    if (key.background) {
        // the scene's background: its map in the direction of the view, blurred about as
        // three's PMREM blurs it for backgroundBlurriness
        Node d = nodes::normalize(world - camera.xyz());
        Node width = nodes::convert(shading::env_size().x(), FLOAT);
        Node lod = nodes::select(rough.greater_than(nodes::num(0)), shading::env_lod(rough, width), 0.0f);
        Node uv = shading::env_uv(d);
        Node seen = nodes::select(lod.greater_than(nodes::num(0)), shading::env_level(uv, lod), shading::env_texel_smooth(uv));
        outc = seen.xyz() * env_params.y();
    }

    if (key.env_combine) {
        // three's envMap on a basic, lambert or phong material: what it mirrors (or what is
        // seen through it) over all of its light, by its reflectivity times the specular map's red
        Node ec = fs_in("env_combine");
        Node to_frag = v.negate();
        Node d = nodes::select(ec.z().greater_than(nodes::num(0)), nodes::refract(to_frag, n, ec.z()), nodes::reflect(to_frag, n));
        Node seen = shading::env_classic(d) * env_params.y();
        Node k = ec.x();
        if (key.specular_slot) k = k * shading::ext_texel(&f, key.specular_slot - 1).x();
        Node combine = nodes::convert(ec.y() + 0.5f, INT);
        outc = nodes::select(combine.equal(nodes::int_(0)), nodes::mix(outc, outc * seen, k),
                             nodes::select(combine.equal(nodes::int_(1)), nodes::mix(outc, seen, k), outc + seen * k));
    }

    // Material.vertex_occlusion: the vertex colors' brightness dims all of it
    if (key.vertex_occlusion) outc = outc * nodes::dot(v_color.xyz(), nodes::vec3(0.2126f, 0.7152f, 0.0722f));

    Node fog_color = fs_in("fog_color");
    Node display_fog_t;
    if (key.fog) {
        Node fd = shading::fog_distance(key, camera.xyz(), world);
        Node fog_own = fog_color.xyz();
        Node fh = fs_in("fog_height");
        if (key.fog_height) {
            // fog lying on the ground: the mean of exp(-k * height) along the ray
            Node ya = nodes::max(camera.y() - fh.x(), 0);
            Node yb = nodes::max(world.y() - fh.x(), 0);
            Node k = fh.y();
            Node dy = yb - ya;
            Node ea = nodes::exp((k * ya).negate());
            fd = fd * nodes::select(nodes::abs(dy).greater_than(nodes::num(0.01f)), (ea - nodes::exp((k * yb).negate())) / (k * dy), ea);
        }
        if (key.fog_sun) {
            Node sun = fs_in("fog_sun_dir");
            Node look = nodes::normalize(world - camera.xyz());
            Node side = nodes::dot(nodes::normalize(look.xz() + nodes::vec2(1e-5f)), nodes::normalize(sun.xz() + nodes::vec2(1e-5f)));
            fd = fd * nodes::mix(fh.w(), 1, nodes::smoothstep(-0.3f, 0.7f, side));
            Node sunlit = nodes::pow(nodes::max(nodes::dot(look, sun.xyz()), 0), sun.w());
            fog_own = nodes::mix(fog_own, fs_in("fog_sun_color", VEC3), sunlit);
        }
        Node t = shading::fog_transmittance(key, fd);
        // (Renderer.display_fog: mixed in after tone mapping, below)
        if (key.display_fog) {
            display_fog_t = t;
        } else {
            outc = outc * t + fog_own * t.one_minus();
        }
    }

    outc = nodes::max(outc, nodes::vec3(0));
    Node shown_alpha = alpha;
    // a light source (emissive far over white) leaves the target's alpha at 0
    Node glow = fs_in("emissive", VEC3);
    alpha = alpha * nodes::smoothstep(4, 16, nodes::max(glow.x(), nodes::max(glow.y(), glow.z()))).one_minus();
    Node shown;
    if (key.linear) {
        shown = nodes::min(outc, nodes::vec3(60000));
        if (key.display_fog) shown = nodes::mix(fog_color.xyz(), shown, display_fog_t);
    } else {
        // (three mixes the fog in the colors as they are shown)
        shown = shading::tone_map_gamut(outc, key.tone, camera.w(), key.display != shading::DISPLAY_SRGB);
        bool hdr = key.display == shading::DISPLAY_HDR;
        if (!hdr) shown = shading::linear_to_srgb(shown);
        if (key.display_fog) shown = nodes::mix(hdr ? fog_color.xyz() : shading::linear_to_srgb(fog_color.xyz()), shown, display_fog_t);
    }
    Node color = nodes::vec4(shown, alpha);
    // Material.premultiplied_alpha: the color as it is shown, times its opacity
    if (key.premultiplied) color = nodes::vec4(color.xyz() * shown_alpha, color.w());
    graph.colors[0] = color;
    if (key.mrt) {
        graph.colors[1] = mrt_normal;
        graph.colors[2] = velocity;
        graph.colors[3] = nodes::vec4(0, 0, rough, metal);
    }
    if (ncuts) graph.fragment = drie::node_block(cuts[:ncuts]);
}
