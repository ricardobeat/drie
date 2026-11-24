// The built-in materials as node graphs (three's NodeMaterial.setup), printed into programs
// that hold only what a material and the render it is drawn in use (nodes_program.c3).
//
// A program is chosen by a MaterialKey: the features that change its code, worked out from
// the same conditions the engine's material shader (shader_core.glsl) branches on, so that a
// generated program draws what that shader draws. What only changes numbers stays in the
// uniforms: the programs read the fields of FsParams and VsParams that they need (fs_in,
// vs_in), and a draw gathers them from the structs the renderer fills for every draw.
module drie;
import sokol, drie::nodes, drie::shading, std::collections::map, std::io, std::os::env, std::thread;

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

// What a material's program holds; for each map role (shader_core.glsl's MAP_*) the shared
// map slot (tex_ext0..5) its map is in, + 1 (0: the material has none).
bitstruct MaterialKey : char[32] {
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
    bool dir_lights : 36;        // the render's light kinds (lit materials)
    bool point_lights : 37;
    bool spot_lights : 38;
    bool channels : 39;          // Material.light_channels: lights of some channels only
    bool map_emissive : 40;      // Material.map_emissive
    bool baked : 41;             // the vertex colors are baked light (Material.vertex_light)
    bool normal_map : 42;
    bool tangents : 43;          // the tangent frame from the vertex tangents
    bool env_light : 44;         // a standard material's environment: its light from all around;
    bool env_reflect : 45;       // the reflection of its environment (its map or the probes),
    bool env_box : 46;           //   box projected
    bool toon_nearest : 47;      // the toon material's gradient map read nearest
    bool shadows : 48;           // it receives the render's shadows:
    bool vsm : 49;               //   VSM's,
    bool soft_shadows : 50;      //   from sources with a size (PCSS),
    bool sun_shadow : 51;        //   the sun's (the directional light's with cascades),
    bool dir_tiles : 52;         //   other directional lights' (a tile of the atlas),
    bool point_shadows : 53;     //   point lights',
    bool spot_shadows : 54;      //   spot lights'
    bool debug_cascades : 55;    // renderer.debug_cascades: the sun's cascades tinted
    bool arm_ao : 56;            // tex_arm holds the ambient occlusion (red),
    bool arm_rough : 57;         //   the roughness (green),
    bool arm_metal : 58;         //   the metalness (blue)
    bool sized : 59;             // point or spot lights with a size, seen in a standard material's highlights
    bool tubes : 60;             // point lights with a length (tubes)
    bool rect_lights : 61;       // RectAreaLights (on a standard material)
    bool parallax : 62;          // Material.parallax_map (a standard material's)
    bool clearcoat : 63;         // the physical material's layers: a clear coat,
    bool sheen : 64;             //   a sheen,
    bool iridescence : 65;       //   a thin film's rainbow,
    bool anisotropy : 66;        //   highlights drawn out along a direction,
    bool transmission : 67;      //   the picture behind it refracted through its volume
    bool probes : 68;            // it mirrors the scene's reflection probes (Material.env_probes)
    bool gi : 69;                // the scene's irradiance volumes (lit materials),
    bool rc : 70;                //   its radiance cascades
    bool ssr : 71;               // screen-space reflections (Material.ssr)
    bool reflector : 72;         // a planar Reflector's picture (Material.reflector)
    bool glass : 73;             // Material.glass: its reflections stay however clear it is
    bool hex : 74;               // Material.hex_tiling
    bool defocus : 75;           // Material.defocus: the maps blurred by the lens's
    bool fog_volume : 76;        // the fog's froxel volume (in place of its analytic fog),
    bool fog_glow : 77;          //   its lights' glow,
    bool fog_shafts : 78;        //   its light shafts (the sun's shadow in it)
    char depth_packing : 79..80; // a depth material's Material.depth_packing
    // DrawKind.PRETESTED: drawn where the depth prepass has drawn it, which cuts its holes
    // (the alpha test's, the LOD cross-fade's: those bits say which); its program comes with
    // the prepass's (MaterialProgram.depth)
    bool prepassed : 81;
    // the shadow maps' program (build_shadow_graph), which reads only gpu_cull, displacement,
    // sway, and for the alpha test's cut-out alpha_test, map and MAP_ALPHA's slot (1: one);
    // a node material's nodes and foliage too
    bool shadow : 82;
    bool bake : 83;              // a light map's sample (lightmap.c3): drawn at uv1, its light blended in
    // a node material's (its *_node graphs plugged in where the material shader's hooks run
    // them): found through its NodeBuild (NodeBuild.programs), which the key alone doesn't name
    bool nodes : 84;
    bool morph : 85;             // GPU morph targets (BufferGeometry.morph_view + Mesh.morph_weights_view)
    bool flat_shading : 86;      // flat face normals via screen-space derivatives (Material.flat_shading)
}

// Logical view IDs for GPU morph target storage buffers (between engine slots < 32 and NODE_USER_VIEW = 64).
const int VIEW_MORPH_DELTAS = 32;
const int VIEW_MORPH_WEIGHTS = 33;

// The map roles (shader_core.glsl's MAP_*) a key holds a slot for.
const int MAP_AO = 0;
const int MAP_SPECULAR_COLOR = 1;
const int MAP_SPECULAR_INTENSITY = 2;
const int MAP_TRANSMISSION = 3;
const int MAP_THICKNESS = 4;
const int MAP_IRIDESCENCE = 5;
const int MAP_IRIDESCENCE_THICKNESS = 6;
const int MAP_CLEARCOAT = 7;
const int MAP_CLEARCOAT_ROUGHNESS = 8;
const int MAP_CLEARCOAT_NORMAL = 9;
const int MAP_SHEEN_COLOR = 10;
const int MAP_SHEEN_ROUGHNESS = 11;
const int MAP_ALPHA = 12;
const int MAP_EMISSIVE = 13;
const int MAP_LIGHT = 14;
const int MAP_ANISOTROPY = 15;
const int MAP_CUSTOM = 16;
const int MAP_FOCUS = 17;
const int MAP_SPECULAR = 18;
const int MAP_MATCAP = 19;
const int MAP_GRADIENT = 20;
const int MAP_BUMP = 21;
const int MAP_ROLES = 22;

// The key as two words: bits 0..127, and 128..255.
struct KeyWords @private {
    uint128 lo;
    uint128 hi;
}

fn uint KeyWords.hash(self) => (uint)(self.lo.hash() ^ self.hi.hash() * 31);
fn bool KeyWords.equals(self, KeyWords other) @operator(==) => self.lo == other.lo && self.hi == other.hi;

// Where role `role`'s slot is in the key's high word: three bits each from its bit 64 (the
// key's bit 192; MAP_CUSTOM, 16, the application's hooks' map, is the material shader's only).
fn int map_role_bit(int role) @local => 64 + 3 * (role < 16 ? role : role - 1);

// The shared slot role `role`'s map is in, + 1 (0: none).
fn int MaterialKey.slot(self, int role) {
    if (role == MAP_CUSTOM) return 0;
    return (int)((bitcast(self, KeyWords).hi >> map_role_bit(role)) & 7);
}

fn void MaterialKey.set_slot(&self, int role, int slot) {
    if (role == MAP_CUSTOM) return;
    int at = map_role_bit(role);
    KeyWords w = bitcast(*self, KeyWords);
    w.hi = (w.hi & ~((uint128)7 << at)) | ((uint128)(slot & 7) << at);
    *self = bitcast(w, MaterialKey);
}

// A generated program and its pipelines (by get_pipeline's key); with MaterialKey.prepassed,
// the depth prepass's too, which shares its vertex stage. A node material's (MaterialKey.nodes)
// is shared by the materials whose graphs print the same programs, reading the same rows of
// their uniforms and the same of their textures: what it draws with comes from the material
// drawn (user_block, user_views).
struct MaterialProgram {
    // what it was built from (for the pipeline cache)
    MaterialKey key;
    ShaderProgram* program;
    ShaderProgram* depth;
    HashMap{ulong, SgPipeline} pipelines;
    // a node material's: its place in node_material_programs, and how many builds have it
    ulong shared_as;
    int users;
}

HashMap{KeyWords, MaterialProgram*} material_programs @local;
// The node materials' programs, by a hash of what they print and read (program_signature).
HashMap{ulong, MaterialProgram*} node_material_programs @local;

// What makes two printings of a program the same program: its sources, and the rows of the
// blocks, the views and the samplers they read (two node materials can print the same text
// reading a uniform at different rows of their USER blocks, or a texture at a different k).
fn void program_signature(DString* out, ShaderProgram* p) @local {
    out.appendf("%s\n--\n%s\n--\n", p.vs, p.fs);
    for (int st = 0; st < 2; st++) {
        foreach (r : p.rows[st]) out.appendf("r%d %d %d %d %d;", st, r.source.ordinal, r.row, r.count, r.at);
    }
    foreach (v : p.views) out.appendf("v%d %d %d %d %d;", v.stage.ordinal, v.source, v.storage, v.kind.ordinal, v.elem.ordinal);
    foreach (sm : p.samplers) out.appendf("s%d %d %d;", sm.stage.ordinal, sm.source, sm.compare);
}

// The signature of `mp`'s programs, in the temporary memory.
fn String material_signature(MaterialProgram* mp) @local {
    DString sig = dstring::temp();
    program_signature(&sig, mp.program);
    if (mp.depth) program_signature(&sig, mp.depth);
    return sig.str_view();
}

// FNV-1a, 64 bits.
fn ulong hash64(String s) @local {
    ulong h = 0xcbf29ce484222325;
    foreach (c : s) h = (h ^ c) * 0x100000001b3;
    return h;
}

// (Materials draw with the precompiled split shaders and NodeProgram hooks; only GPU morph
// meshes route through materials_nodes.c3.)

// The program for `key`, built the first time it is asked for; with MaterialKey.nodes, the one
// node material `build`'s graphs print, found through the build (NodeBuild.programs) and shared
// with the builds that print the same.
fn MaterialProgram* material_program(MaterialKey key, NodeBuild* build = null) @private {
    // (a node material's: printed for each build, made once for the builds that print the same)
    KeyWords words = bitcast(key, KeyWords);
    HashMap{KeyWords, MaterialProgram*}* cache = &material_programs;
    if (key.nodes) {
        if (!build) abort("drie: a node material's program without its build");
        cache = &build.programs;
    } else {
        build = null;
    }
    if (!cache.is_initialized()) cache.init(mem);
    if (try found = cache.get(words)) return found;
    uint mark = node_count();
    ProgramGraph graph;
    ProgramGraph prepass;
    UserBinds binds;
    if (key.shadow) {
        build_shadow_graph(key, &graph, build, &binds);
    } else {
        build_material_graph(key, &graph, &prepass, build, &binds);
    }
    MaterialProgram* mp = mem::new(MaterialProgram);
    mp.key = key;
    // (DRIE_PROGRAM_SALT: a term the output depends on, at a pixel no target reaches, so the
    // compiled code differs from any earlier run's and the driver has none of it cached; for
    // timing a cold start)
    if (try salt = env::tget_var("DRIE_PROGRAM_SALT")) {
        float far = 1e7f + (float)(salt.hash() & 0xffff);
        Node off = nodes::select(node_system(FRAG_COORD).x().greater_than(nodes::num(far)), nodes::num(1e-20f), nodes::num(0));
        if (graph.colors[0].id) graph.colors[0] = graph.colors[0] + nodes::vec4(off, 0, 0, 0);
        if (key.prepassed) prepass.colors[0] = prepass.colors[0] + nodes::vec4(off, 0, 0, 0);
    }
    // (the prepass's fragment stage first: its varyings are the first of the main program's,
    // whose vertex stage it then takes)
    Node[NodeBuiltin::len]* vbind = build ? &binds.v : null;
    Node[NodeBuiltin::len]* fbind = build ? &binds.f : null;
    if (key.prepassed) {
        prepass.fragment_only = true;
        mp.depth = build_program(backend_lang(), &prepass, build, vbind, fbind);
        graph.varyings_from = mp.depth;
    }
    mp.program = build_program(backend_lang(), &graph, build, vbind, fbind);
    if (key.prepassed) mp.depth.take_vertex(mp.program);
    node_release(mark);
    if (build) {
        @pool() {
            // (not by its key: keys that differ in what the program doesn't print share it)
            String sig = material_signature(mp);
            ulong h = hash64(sig);
            if (!node_material_programs.is_initialized()) node_material_programs.init(mem);
            MaterialProgram*? shared = node_material_programs[h];
            if (try shared && material_signature(shared) == sig) {
                free_program(mp.program);
                if (mp.depth) free_program(mp.depth);
                mem::free(mp);
                shared.users++;
                cache.set(words, shared);
                return shared;
            }
            // (one whose hash another has, which is no likelier than a collision of 64-bit hashes:
            // kept out of the table, shared with none)
            mp.users = 1;
            if (catch shared) {
                mp.shared_as = h;
                node_material_programs[h] = mp;
            }
        };
    }
    mp.pipelines.init(mem);
    cache.set(words, mp);
    // (DRIE_TRACE_PROGRAMS: each program printed as it is built)
    if (try env::tget_var("DRIE_TRACE_PROGRAMS")) {
        io::eprintfn("drie: program %x%032x: vs %d B, fs %d B\n%s\n%s", words.hi, words.lo, mp.program.vs.len, mp.program.fs.len, mp.program.vs, mp.program.fs);
        if (mp.depth) io::eprintfn("drie: its depth prepass: fs %d B\n%s", mp.depth.fs.len, mp.depth.fs);
    }
    return mp;
}

// The bindings of the classic slots (shaders::VIEW_*, SMP_*) as `p` takes them; a node
// material's textures (NODE_USER_VIEW + k) from `user` (its NodeBuild.textures' views).
fn SgBindings program_bindings(ShaderProgram* p, SgBindings* classic, SgView[] user = { },
                               SgView morph_deltas = { }, SgView morph_weights = { }) @private {
    SgBindings b = {
        .vertex_buffers = classic.vertex_buffers,
        .vertex_buffer_offsets = classic.vertex_buffer_offsets,
        .index_buffer = classic.index_buffer,
        .index_buffer_offset = classic.index_buffer_offset,
    };
    foreach (i, v : p.views) {
        if (v.source >= NODE_USER_VIEW) {
            int k = v.source - NODE_USER_VIEW;
            if (k >= user.len) abort("drie: a node material's texture %d without its view", k);
            b.views[i] = user[k];
        } else if (v.source == VIEW_MORPH_DELTAS) {
            b.views[i] = morph_deltas;
        } else if (v.source == VIEW_MORPH_WEIGHTS) {
            b.views[i] = morph_weights;
        } else {
            b.views[i] = classic.views[v.source];
        }
    }
    foreach (i, s : p.samplers) b.samplers[i] = classic.samplers[s.source];
    return b;
}

// Applies `p`'s uniform block for stage `st`, gathered from the engine's (a shadow program's
// fragment stage: from `sfs`) and a node material's USER block (`user`, see user_block).
fn void apply_program_uniforms(ShaderProgram* p, ProgramStage st, VsParams* vs, FsParams* fs, ShadowFsParams* sfs = null,
                               Vec4[] user = { }) @private {
    int rows = p.block_rows[st.ordinal];
    if (!rows) return;
    void*[NodeSource::len] src;
    sz[NodeSource::len] sizes;
    src[NodeSource.VS.ordinal] = vs;
    sizes[NodeSource.VS.ordinal] = VsParams::size;
    src[NodeSource.FS.ordinal] = fs;
    sizes[NodeSource.FS.ordinal] = FsParams::size;
    src[NodeSource.SHADOW_FS.ordinal] = sfs;
    sizes[NodeSource.SHADOW_FS.ordinal] = sfs ? ShadowFsParams::size : 0;
    src[NodeSource.USER.ordinal] = user.len ? user.ptr : null;
    sizes[NodeSource.USER.ordinal] = user.len * Vec4::size;
    @pool() {
        Vec4[] block = mem::temp_array(Vec4, rows);
        p.gather(st, src, sizes, block);
        sg::apply_uniforms(st.ordinal, &&{ .ptr = block.ptr, .size = (usz)rows * 16 });
    };
}

// Lets go of a node material's generated programs (when node_build_of replaces its build): those
// no other build has.
fn void free_build_programs(NodeBuild* b) @private {
    if (!b.programs.is_initialized()) return;
    b.programs.@each(; KeyWords words, MaterialProgram* mp) {
        if (--mp.users == 0) free_material_program(mp);
    };
    b.programs.free();
}

fn void free_material_program(MaterialProgram* mp) @local {
    if (try in = node_material_programs[mp.shared_as] && in == mp) node_material_programs.remove(mp.shared_as);
    mp.pipelines.@each(; ulong key, SgPipeline pip) {
        sg::destroy_pipeline(pip);
    };
    mp.pipelines.free();
    free_program(mp.program);
    if (mp.depth) free_program(mp.depth);
    mem::free(mp);
}

fn void free_program(ShaderProgram* p) @local {
    // (its compile in the background run to its end first)
    if (p.job) {
        while (sg_warm_state(p.job - 1) == 0) thread::sleep_ms(1);
        sg_warm_end(p.job - 1);
    }
    if (p.shader.id) sg::destroy_shader(p.shader);
    p.free();
    mem::free(p);
}

// ---- a node material's graphs

// What a node material's graphs read their builtins from in each stage (build_program's
// vertex_bind and fragment_bind).
struct UserBinds @local {
    Node[NodeBuiltin::len] v;
    Node[NodeBuiltin::len] f;
}

// The builtins the USER block's header holds (NODE_USER_HEADER), for either stage.
fn void user_header(Node[NodeBuiltin::len]* b) @local {
    (*b)[NodeBuiltin.VIEW_MATRIX.ordinal] = node_input(USER, 0, MAT4);
    (*b)[NodeBuiltin.PROJECTION_MATRIX.ordinal] = node_input(USER, 4, MAT4);
    (*b)[NodeBuiltin.MODEL_MATRIX.ordinal] = node_input(USER, 8, MAT4);
    (*b)[NodeBuiltin.CAMERA_POSITION.ordinal] = node_input(USER, 12, VEC3);
    (*b)[NodeBuiltin.TIME.ordinal] = node_input(USER, 12, FLOAT, 3);
}

fn Node user_camera() @local => node_input(USER, 12, VEC3);

// A builtin bound to `value` from here on (a statement).
fn Node rebind(NodeBuiltin b, Node value) @local => drie::node_bind(b, value);

// `value` (a node material's graph), read with the builtins `binds` bound first: wherever it
// is printed, it reads them as they are at this step of the material.
fn Node bound(Node value, Node[] binds) @local => nodes::seq(drie::node_block(binds), value);

// What the fragment stage's builtins read before the material's steps bind them again: `n` the
// vertex normal facing the camera.
fn void fragment_binds(VertexValues* vtx, Node[NodeBuiltin::len]* b, Node v_uv, Node world, Node n) @local {
    user_header(b);
    (*b)[NodeBuiltin.UV.ordinal] = v_uv;
    (*b)[NodeBuiltin.UV1.ordinal] = nodes::varying(vtx.uv1);
    (*b)[NodeBuiltin.POSITION_WORLD.ordinal] = world;
    (*b)[NodeBuiltin.POSITION_LOCAL.ordinal] = nodes::varying(vtx.local);
    (*b)[NodeBuiltin.NORMAL_LOCAL.ordinal] = nodes::varying(vtx.local_normal);
    (*b)[NodeBuiltin.INSTANCE_INDEX.ordinal] = nodes::varying(vtx.instance_index);
    (*b)[NodeBuiltin.NORMAL_WORLD.ordinal] = n;
    (*b)[NodeBuiltin.NORMAL_GEOMETRY.ordinal] = n;
    (*b)[NodeBuiltin.VIEW_DIRECTION.ordinal] = nodes::normalize(user_camera() - world);
    (*b)[NodeBuiltin.VERTEX_COLOR.ordinal] = nodes::varying(vtx.color);
    (*b)[NodeBuiltin.FRAG_COORD.ordinal] = node_system(FRAG_COORD).xy();
    (*b)[NodeBuiltin.FRONT_FACING.ordinal] = node_system(FRONT_FACING);
}

// A node material's color step (where the material shader runs node_color): mask_node's
// discard, then color_node and opacity_node over the color `rgb` and its alpha `a`, read with
// NORMAL_WORLD the facing vertex normal `n`. `holes`: the statements that cut its holes (the
// mask's discard, and the slots' own when the graphs can cut: NodeProgram.cuts).
fn void user_color(NodeBuild* build, Node n, Node* rgb, Node* a, Node[] holes, int* count) @local {
    Node mask = build.slots[8];
    Node color = build.slots[0];
    Node opacity = build.slots[1];
    // (MATERIAL_COLOR: the color and opacity as they come to this step)
    Node own = nodes::vec4(*rgb, *a);
    if (mask.id) {
        if (mask.type() != BOOL) abort("drie: mask_node is a bool");
        holes[(*count)++] = nodes::if_then(nodes::logical_not(bound(mask, { rebind(NORMAL_WORLD, n), rebind(NORMAL_GEOMETRY, n) })),
                                           nodes::block(nodes::discard()));
    }
    if (color.id) {
        Node c = bound(color, { rebind(NORMAL_WORLD, n), rebind(NORMAL_GEOMETRY, n), rebind(MATERIAL_COLOR, own) });
        if (build.program.cuts) holes[(*count)++] = drie::node_eval(c);
        if (color.type() == VEC4) {
            *rgb = c.xyz();
            *a = c.w();
        } else {
            *rgb = nodes::convert(c, VEC3);
        }
    }
    if (opacity.id) {
        Node o = bound(nodes::convert(opacity, FLOAT), { rebind(NORMAL_WORLD, n), rebind(NORMAL_GEOMETRY, n), rebind(MATERIAL_COLOR, own) });
        if (build.program.cuts) holes[(*count)++] = drie::node_eval(o);
        *a = o;
    }
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
    Node tangent;        // xyz in the world, w its handedness
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

fn Node Instance.point(&self, Node p) {
    Node l = nodes::vec4(p, 1);
    return nodes::vec3(nodes::dot(self.m0, l), nodes::dot(self.m1, l), nodes::dot(self.m2, l));
}

fn Node Instance.dir(&self, Node d) {
    return nodes::vec3(nodes::dot(self.m0.xyz(), d), nodes::dot(self.m1.xyz(), d), nodes::dot(self.m2.xyz(), d));
}

// `build`: a node material's graphs (position_node), whose builtins it binds in `binds`.
fn VertexValues vertex_stage(MaterialKey key, NodeBuild* build = null, UserBinds* binds = null) @local {
    VertexValues o;
    Node position = node_attribute(shaders::ATTR_THREE_POSITION, VEC3);
    Node normal = node_attribute(shaders::ATTR_THREE_NORMAL, VEC3);
    Node uv = node_attribute(shaders::ATTR_THREE_UV, VEC2);
    Node color0 = node_attribute(shaders::ATTR_THREE_COLOR0, VEC4);
    Node[8] statements;
    int ns;
    Node instance_id;
    Instance inst;
    if (key.gpu_cull) {
        // (fetch_instance: instance i of the draw is rows[3 * visible[first + i]], if i is
        // under the cull's survivors)
        Node cull = vs_in("inst_cull");
        Node i = node_system(INSTANCE_ID);
        Node count = node_storage(shaders::VIEW_INST_COUNTS, UINT, nodes::convert(cull.z(), UINT));
        statements[ns++] = nodes::if_then(i.greater_than_equal(count), nodes::block(node_return(nodes::vec4(2, 2, 2, 1))));
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
        Node hdr = node_storage(VIEW_MORPH_WEIGHTS, VEC4, nodes::uint_(0));
        Node m_count = nodes::convert(hdr.x(), UINT);
        Node v_count = nodes::convert(hdr.y(), UINT);
        Node w_stride = nodes::convert(hdr.z(), UINT);
        Node vid = node_system(VERTEX_ID);
        Node w_base = nodes::uint_(1) + instance_id * w_stride;
        Node mp = nodes::variable(local);
        Node ti = node_var(UINT);
        Node w4 = node_storage(VIEW_MORPH_WEIGHTS, VEC4, w_base + node_binary(SHR, ti, nodes::uint_(2)));
        Node lane = node_binary(BIT_AND, ti, nodes::uint_(3));
        Node w = nodes::select(lane.equal(nodes::uint_(0)), w4.x(),
            nodes::select(lane.equal(nodes::uint_(1)), w4.y(),
                nodes::select(lane.equal(nodes::uint_(2)), w4.z(), w4.w())));
        Node delta = node_storage(VIEW_MORPH_DELTAS, VEC4, ti * v_count + vid).xyz();
        Node body = nodes::block(mp.assign(mp + delta * w));
        statements[ns++] = mp;
        statements[ns++] = node_loop(LT, ti, nodes::uint_(0), m_count, body);
        local = mp;
    }
    if (key.displacement) {
        Node d = vs_in("surface_vs");
        Node height = node_attribute(shaders::ATTR_THREE_HEIGHT, FLOAT);
        local = local + nodes::normalize(normal) * (height * d.y() + d.z());
    }
    Node ip = inst.point(local);
    Node model = vs_in("model", MAT4);
    Node dir = inst.dir(normal);
    if (build) {
        // a node material's builtins (run_node_vertex's), and position_node in ip's place:
        // after the displacement and the instance's matrix, before the model matrix and sway.
        // From there on its builtins read the moved vertex (the varyings it carries)
        Node[NodeBuiltin::len]* b = &binds.v;
        user_header(b);
        Node world0 = (model * nodes::vec4(ip, 1)).xyz();
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
            Node at = nodes::variable(nodes::convert(moved, VEC3));
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
    Node wp = model * nodes::vec4(ip, 1);
    Node world = wp.xyz();
    Node sway = vs_in("sway");
    if (key.sway) world = shading::apply_sway(world, ip.y(), sway);
    Node view_proj = vs_in("view_proj", MAT4);
    Node origin = vs_in("view_origin", VEC3);
    Node pos = view_proj * nodes::vec4(world - origin, 1);
    if (key.outline) {
        // (Material.outline: out along the normal as the screen sees it)
        Node inward = world - (model * nodes::vec4(dir, 0)).xyz();
        Node p2 = view_proj * nodes::vec4(inward - origin, 1);
        pos = pos + nodes::normalize(pos - p2) * (vs_in("surface_vs", FLOAT) * pos.w());
    }
    if (key.bake) {
        // (the light map's texel at uv1; the map's first row on top but in GL)
        Node uv1 = node_attribute(shaders::ATTR_THREE_UV1, VEC2);
        Node y = nodes::select(vs_in("bake", FLOAT, 1).greater_than(nodes::num(0.5f)), uv1.y() * -2.0f + 1.0f, uv1.y() * 2.0f - 1.0f);
        pos = nodes::vec4(uv1.x() * 2.0f - 1.0f, y, 0, 1);
    }
    o.position = ns ? nodes::seq(drie::node_block(statements[:ns]), pos) : pos;
    o.world = world;
    o.normal = (vs_in("normal_mat", MAT4) * nodes::vec4(dir, 0)).xyz();
    Node t = vs_in("uv_transform");
    Node t1 = vs_in("uv_transform", k: 1);
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
        Node pp = vs_in("prev_model", MAT4) * nodes::vec4(ip, 1);
        if (key.sway) pp = nodes::vec4(shading::apply_sway(pp.xyz(), ip.y(), nodes::vec4(motion.y(), sway.y(), sway.z(), sway.w())), pp.w());
        Node swayed = nodes::vec4(world, wp.w());
        o.prev_clip = nodes::select(motion.x().greater_than(nodes::num(1.5f)), prev_view_proj * pp,
                                    nodes::select(motion.x().greater_than(nodes::num(0.5f)), prev_view_proj * swayed, o.clip));
    }
    return o;
}

// ---- the material

// Parallax occlusion mapping (Material.parallax_map): the uv where the view ray meets the
// height map, which the material's maps are read at (with the derivatives of v_uv: the offset
// jumps where the height map steps, and its own derivatives would pick a blurry mip level
// there); how the uv changes per meter along the surface (the cotangent frame, as in
// perturb_normal); the height map there (0..1) and the depth after the distance fade (m).
struct Parallax @local {
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
    if (!f.hex_w.id) return node_sample(GRAD, shaders::VIEW_TEX_PARALLAX, shaders::SMP_TEX_SMP, TEX_2D, uv, f.dx, f.dy).x();
    Node acc = nodes::variable(nodes::num(0));
    Node[3] w = { f.hex_w.x(), f.hex_w.y(), f.hex_w.z() };
    Node[4] steps;
    steps[0] = acc;
    for (int i = 0; i < 3; i++) {
        Node h = node_sample(GRAD, shaders::VIEW_TEX_PARALLAX, shaders::SMP_TEX_SMP, TEX_2D, uv + f.hex_off[i], f.dx, f.dy).x();
        steps[i + 1] = nodes::if_then(w[i].greater_than(nodes::num(0.004f)), nodes::block(acc.assign(acc + h * w[i])));
    }
    return nodes::seq(drie::node_block(steps[:4]), acc);
}

// Marches the view ray (`v`, toward the camera `dist` away) down into the height field (1 = the
// surface, 0 = depth meters under it) to where it meets it.
fn Parallax parallax_offset(FragUv* f, Node n, Node v, Node dist, Node world) @local {
    Node params = fs_in("parallax_params");
    Node depth = params.x() * nodes::smoothstep(params.y() * 0.6f, params.y(), dist).one_minus();
    Node pu = nodes::variable(f.uv);
    Node ph = nodes::variable(nodes::num(1));
    Node gu = nodes::variable(nodes::vec3(0));
    Node gv = nodes::variable(nodes::vec3(0));
    Node dp1 = nodes::dfdx(world);
    Node dp2 = nodes::dfdy(world);
    Node dp2perp = nodes::cross(dp2, n);
    Node dp1perp = nodes::cross(n, dp1);
    Node det = nodes::dot(dp1, dp2perp);
    // the uv a step of 1 m down the view ray moves by (flattened at a glancing angle, where the
    // march would smear the texture out)
    Node nv = nodes::max(nodes::dot(n, v), 0.2f);
    Node per_m = nodes::vec2(nodes::dot(gu, v), nodes::dot(gv, v)).negate() / nv;
    Node total = per_m * depth;
    Node steps = nodes::mix(24, 8, nv);
    Node dh = 1.0f / steps;
    Node ray = nodes::variable(nodes::num(1));
    Node h = nodes::variable(parallax_height(f, f.uv));
    Node prev_h = nodes::variable(h);
    Node uv = nodes::variable(f.uv);
    Node i = nodes::loop_index();
    Node march = nodes::loop(i, 0, 24, nodes::block(
        nodes::if_then(nodes::logical_or(h.greater_than_equal(ray), nodes::convert(i, FLOAT).greater_than_equal(steps)), nodes::block(nodes::break_())),
        prev_h.assign(h),
        ray.assign(ray - dh),
        uv.assign(uv + total * dh),
        h.assign(parallax_height(f, uv))));
    // between the last step above the surface and the first below it
    Node after = h - ray;
    Node before = prev_h - (ray + dh);
    Node t = nodes::clamp(after / nodes::min(after - before, -1e-5f), 0, 1);
    Node meet = nodes::block(
        gu.assign((dp2perp * f.dx.x() + dp1perp * f.dy.x()) / det),
        gv.assign((dp2perp * f.dx.y() + dp1perp * f.dy.y()) / det),
        drie::node_eval(total),
        drie::node_eval(dh),
        ray, h, prev_h, uv,
        march,
        drie::node_eval(t),
        pu.assign(uv - total * dh * t),
        ph.assign(nodes::mix(h, prev_h, t)));
    Node deep = nodes::logical_and(depth.greater_than(nodes::num(0)), nodes::abs(det).greater_than_equal(nodes::num(1e-14f)));
    Node at = nodes::seq(nodes::block(
        pu, ph, gu, gv,
        drie::node_eval(dp2perp), drie::node_eval(dp1perp),
        nodes::if_then(deep, meet)), pu);
    return { .uv = at, .gu = gu, .gv = gv, .h = ph, .depth = depth };
}

// How much of the light from l reaches the parallax's uv past the bumps of the height map around
// it (soft: the more a bump stands above the ray to the light, the darker).
fn Node parallax_shadow(Parallax* px, FragUv* f, Node n, Node l) @local {
    Node strength = fs_in("parallax_params", FLOAT, 2);
    Node nl = nodes::dot(n, l);
    Node per_m = nodes::vec2(nodes::dot(px.gu, l), nodes::dot(px.gv, l)) / nodes::max(nl, 0.1f);
    Node rise = px.h.one_minus();
    Node total = per_m * px.depth * rise;
    Node block = nodes::variable(nodes::num(0));
    Node i = nodes::loop_index();
    Node fi = nodes::convert(i, FLOAT) / 8.0f;
    Node above = parallax_height(f, px.uv + total * fi) - (px.h + rise * fi);
    Node off = nodes::logical_or(nodes::logical_or(px.depth.less_than_equal(nodes::num(0)), strength.less_than_equal(nodes::num(0))),
                                 nodes::logical_or(nl.less_than_equal(nodes::num(0)), rise.less_than_equal(nodes::num(0.01f))));
    Node shade = nodes::variable(nodes::num(1));
    return nodes::seq(nodes::block(shade, nodes::if_then(nodes::logical_not(off), nodes::block(
        block,
        nodes::loop(i, 1, 9, nodes::block(block.assign(nodes::max(block, above * fi.one_minus() * 8.0f)))),
        shade.assign(strength.negate() * nodes::clamp(block, 0, 1) + 1.0f)))), shade);
}

// The shadow maps' program: the material's vertex stage placed by the light (VsParams filled
// for it: see draw_shadow_item), and the alpha test's cut-out as shader_core.glsl's shadow_fs
// (the color map's alpha, the opacity and the alpha map's green, ShadowFsParams). A node
// material's (`build`) moves its vertices as its main program does, and cuts the holes its
// color step cuts (mask_node's, color_node's and opacity_node's) as shadow_nodes_fs does.
fn void build_shadow_graph(MaterialKey key, ProgramGraph* graph, NodeBuild* build, UserBinds* binds) @local {
    VertexValues vtx = vertex_stage(key, build, binds);
    graph.position = vtx.position;
    bool node_cuts = build && (build.program.cuts || build.slots[8].id);
    if (!key.alpha_test && !node_cuts) return;
    ShadowFsParams* sp = null;
    int params = (int)((usz)&sp.s_params / 16);
    int alpha_uv = (int)((usz)&sp.s_alpha_uv / 16);
    Node p = node_input(SHADOW_FS, params, VEC4);
    Node v_uv = nodes::varying(vtx.uv);
    Node a = p.y();
    if (key.map) a = a * node_sample(SAMPLE, shaders::VIEW_S_MAP, shaders::SMP_S_SMP, TEX_2D, v_uv).w();
    if (key.slot(MAP_ALPHA)) {
        Node uvh = nodes::vec3(v_uv, 1);
        Node at = nodes::vec2(nodes::dot(node_input(SHADOW_FS, alpha_uv, VEC3), uvh), nodes::dot(node_input(SHADOW_FS, alpha_uv + 1, VEC3), uvh));
        a = a * node_sample(SAMPLE, shaders::VIEW_S_ALPHA, shaders::SMP_S_ALPHA_SMP, TEX_2D, at).y();
    }
    Node[4] cuts;
    int ncuts;
    if (node_cuts) {
        Node world = nodes::varying(vtx.world);
        Node n = nodes::normalize(nodes::varying(vtx.normal));
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
// material's graphs (`build`, MaterialKey.nodes) go in where the material shader runs its
// hooks, reading their builtins as `binds` and the steps below bind them.
fn void build_material_graph(MaterialKey key, ProgramGraph* graph, ProgramGraph* prepass, NodeBuild* build, UserBinds* binds) @private {
    VertexValues vtx = vertex_stage(key, build, binds);
    graph.position = vtx.position;

    Node world = nodes::varying(vtx.world);
    Node v_normal = nodes::varying(vtx.normal);
    if (key.flat_shading) {
        Node fn_ = nodes::cross(nodes::dfdx(world), nodes::dfdy(world));
        Node view_dir = fs_in("camera_pos", VEC3) - world;
        Node oriented = nodes::select(nodes::dot(fn_, view_dir).greater_than_equal(nodes::num(0)), fn_, fn_.negate());
        v_normal = nodes::select(nodes::dot(fn_, fn_).greater_than(nodes::num(1e-12f)), oriented, v_normal);
    }
    Node v_uv = nodes::varying(vtx.uv);
    Node fc = node_system(FRAG_COORD);
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
        Node cut = nodes::if_then(nodes::logical_and(sp.w().not_equal(nodes::num(0)), out), nodes::block(nodes::discard()));
        if (key.prepassed) {
            holes[nholes++] = cut;
        } else {
            cuts[ncuts++] = cut;
        }
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
    Node widen;
    if (key.defocus) {
        // Material.defocus: the maps read over the lens's blur at this distance, as wide as it
        // is (the depth of field pass sees through what is transparent)
        Node focus = fs_in("focus_params");
        Node at = node_sample(LEVEL, shaders::VIEW_TEX_EXT0 + key.slot(MAP_FOCUS) - 1, shaders::SMP_TEX_SMP, TEX_2D, nodes::vec2(0.5f), nodes::num(0));
        Node fd = nodes::max(at.x(), 0.01f);
        Node z = nodes::max(nodes::length(world - fs_in("camera_pos", VEC3)), 0.01f);
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
        Node texel = key.hex ? f.hex_color() : node_sample(GRAD, shaders::VIEW_TEX_MAP, shaders::SMP_TEX_SMP, TEX_2D, f.uv, f.dx, f.dy);
        rgb = rgb * texel.xyz();
        a = a * texel.w();
        map_rgb = texel.xyz();
    }
    // three's alphaMap: its green channel
    if (key.slot(MAP_ALPHA)) a = a * role_texel(key, &f, MAP_ALPHA).y();
    if (build) {
        // the node material's color step (node_color): its holes cut in the depth prepass when
        // it has one (the main pass draws only where that drew)
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
            Node texel = v_uv * nodes::convert(node_sample(SIZE, shaders::VIEW_TEX_MAP, shaders::SMP_TEX_SMP, TEX_2D, { }), VEC2);
            Node tx = nodes::dfdx(texel);
            Node ty = nodes::dfdy(texel);
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
        Node m = shading::slot_texel(&f, 0, shaders::VIEW_TEX_NORMAL).xyz() * 2.0f - 1.0f;
        Node scale = fs_in("tangent_params", VEC2);
        m = nodes::vec3(m.xy() * scale, m.z());
        Node[2] tb = tangent_frame(key, n, v_normal, world, vtx.tangent, shading::slot_d(&f, 0, false), shading::slot_d(&f, 0, true));
        n = nodes::normalize(tb[0] * m.x() + tb[1] * m.y() + n * m.z());
    }
    int bump = key.slot(MAP_BUMP);
    if (bump && key.type != (char)MaterialType.BASIC) {
        // three's bumpMap: the normal tilted by how the height changes from pixel to pixel (its
        // perturbNormalArb, Mikkelsen's)
        int bu = bump + 1;
        int view = shaders::VIEW_TEX_EXT0 + bump - 1;
        Node buv = shading::slot_uv(&f, bu);
        Node dx = shading::slot_d(&f, bu, false);
        Node dy = shading::slot_d(&f, bu, true);
        Node hll = shading::slot_grad_at(bu, view, buv, dx, dy).x();
        Node dh = nodes::vec2(shading::slot_grad_at(bu, view, buv + dx, dx, dy).x() - hll,
                              shading::slot_grad_at(bu, view, buv + dy, dx, dy).x() - hll) * fs_in("surface_params", FLOAT);
        Node sx = nodes::normalize(nodes::dfdx(world));
        Node sy = nodes::normalize(nodes::dfdy(world));
        Node r1 = nodes::cross(sy, n);
        Node r2 = nodes::cross(n, sx);
        Node det = nodes::dot(sx, r1) * nodes::select(front, 1.0f, -1.0f);
        Node grad = (r1 * dh.x() + r2 * dh.y()) * nodes::sign(det);
        n = nodes::normalize(n * nodes::abs(det) - grad);
    }
    if (user[2].id && key.type != (char)MaterialType.BASIC) {
        // normal_node (node_normal), after the normal map and the bump map
        n = nodes::normalize(bound(nodes::convert(user[2], VEC3), { rebind(NORMAL_WORLD, n), rebind(NORMAL_GEOMETRY, geo_n) }));
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

    Node camera = fs_in("camera_pos");
    Node to_cam = camera.xyz() - world;
    Node depth = nodes::length(to_cam);
    Node v = to_cam / nodes::max(depth, 1e-5f);
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
    // a node material's surface step (node_surface, node_specular): roughness_node,
    // metalness_node, emissive_node (in place of all the material's glow) and phong's
    // specular_node and shininess_node, read with NORMAL_WORLD the shaded normal
    Node user_glow;
    Node phong_specular = fs_in("specular", VEC3);
    Node shininess = fs_in("mat_params", FLOAT, 3);
    if (build) {
        // (MATERIAL_ROUGHNESS and MATERIAL_METALNESS: the material's own, after its maps)
        Node own_rough = rough;
        Node own_metal = metal;
        Node[4] sb = { rebind(NORMAL_WORLD, n), rebind(NORMAL_GEOMETRY, geo_n), rebind(MATERIAL_ROUGHNESS, own_rough),
                       rebind(MATERIAL_METALNESS, own_metal) };
        if (user[3].id) rough = nodes::clamp(bound(nodes::convert(user[3], FLOAT), &sb), 0, 1);
        if (user[4].id) metal = nodes::clamp(bound(nodes::convert(user[4], FLOAT), &sb), 0, 1);
        if (user[14].id) ao = ao * nodes::clamp(bound(nodes::convert(user[14], FLOAT), &sb), 0, 1);
        if (user[5].id) user_glow = bound(nodes::convert(user[5], VEC3), &sb);
        if (key.type == (char)MaterialType.PHONG) {
            if (user[10].id) phong_specular = bound(nodes::convert(user[10], VEC3), &sb);
            if (user[11].id) shininess = bound(nodes::convert(user[11], FLOAT), &sb);
        }
    }
    Lit lit = { .key = key, .n = n, .geo_n = geo_n, .v = v, .world = world, .rough = rough, .metal = metal, .alpha = alpha,
                .ao = ao, .parallax = px, .f = &f, .v_normal = v_normal, .tangent = vtx.tangent, .irid_fn = -1, .frag = fc.xy(),
                .shadow = shadow_fns(key), .ltc_fn = -1, .refraction = { -1, -1, -1 }, .probe_fn = -1, .ssr_fn = -1,
                .user_glow = user_glow, .user_env = user[9], .user_cc = user[12], .user_cc_rough = user[13], .phong_specular = phong_specular, .shininess = shininess };
    if (key.type == (char)MaterialType.MATCAP) {
        // MeshMatcapMaterial: the matcap image at the view-space normal, from a frame turned
        // toward the viewer (three's), times the color
        Node vd = nodes::normalize(shading::to_view(v));
        Node vn = nodes::normalize(shading::to_view(n));
        Node x = nodes::normalize(nodes::vec3(vd.z(), 0, vd.x().negate()));
        Node y = nodes::cross(vd, x);
        Node muv = nodes::vec2(nodes::dot(x, vn), nodes::dot(y, vn)) * 0.495f + 0.5f;
        Node matcap;
        int ms = key.slot(MAP_MATCAP) - 1;
        if (ms >= 0) {
            matcap = shading::slot_grad_at(ms + 2, shaders::VIEW_TEX_EXT0 + ms, muv, nodes::dfdx(muv), nodes::dfdy(muv)).xyz();
        } else {
            matcap = nodes::vec3(nodes::mix(nodes::num(0.2f), nodes::num(0.8f), muv.y()));
        }
        outc = rgb * matcap;
    }
    if (key.type == (char)MaterialType.BASIC || key.type == (char)MaterialType.MATCAP) {
        if (key.type == (char)MaterialType.BASIC) outc = rgb;
        // three's lightMap: the indirect light is the map's instead of 1
        if (key.type == (char)MaterialType.BASIC && key.slot(MAP_LIGHT)) outc = outc * (role_texel(key, &f, MAP_LIGHT).xyz() * (fs_in("light_map_params", FLOAT) * shading::RECIPROCAL_PI));
        if (key.background && key.type == (char)MaterialType.BASIC) {
            // the scene's background: its map in the direction of the view, blurred about as
            // three's PMREM blurs it for backgroundBlurriness
            Node d = nodes::normalize(world - camera.xyz());
            Node width = nodes::convert(shading::env_size().x(), FLOAT);
            Node lod = nodes::select(rough.greater_than(nodes::num(0)), shading::env_lod(rough, width), 0.0f);
            Node uv = shading::env_uv(d);
            Node seen = nodes::select(lod.greater_than(nodes::num(0)), shading::env_level(uv, lod), shading::env_texel_smooth(uv));
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
        Node ec = fs_in("env_combine");
        Node to_frag = v.negate();
        Node d = nodes::select(ec.z().greater_than(nodes::num(0)), nodes::refract(to_frag, n, ec.z()), nodes::reflect(to_frag, n));
        Node seen = shading::env_classic(d) * env_params.y();
        Node k = ec.x();
        if (key.slot(MAP_SPECULAR)) k = k * role_texel(key, &f, MAP_SPECULAR).x();
        Node combine = nodes::convert(ec.y() + 0.5f, INT);
        outc = nodes::select(combine.equal(nodes::int_(0)), nodes::mix(outc, outc * seen, k),
                             nodes::select(combine.equal(nodes::int_(1)), nodes::mix(outc, seen, k), outc + seen * k));
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

    // a node material's output_node (node_output): what the light made of it, and its alpha
    if (user[7].id) {
        Node o = bound(user[7], { rebind(OUTPUT, nodes::vec4(outc, alpha)), rebind(NORMAL_WORLD, n), rebind(NORMAL_GEOMETRY, geo_n) });
        if (user[7].type() == VEC4) {
            outc = o.xyz();
            alpha = o.w();
        } else {
            outc = nodes::convert(o, VEC3);
        }
    }

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
            if (key.fog_shafts) {
                // light shafts: the sun's part of the fog only where the sun gets to it
                ShadowFns shadow = shadow_fns(key);
                Node len = nodes::length(world - camera.xyz());
                Node in_sun = sun_in_fog(&shadow, camera.xyz(), world, fs_in("fog_params", FLOAT, 2) * fd / nodes::max(len, 1e-4f), fc.xy());
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

    outc = nodes::max(outc, nodes::vec3(0));
    if (key.bake) {
        // the light map: the new sample blended into the last map's texel, in linear light
        Node uv1 = nodes::varying(vtx.uv1);
        Node prev = node_sample(SAMPLE, shaders::VIEW_SSR_COLOR, shaders::SMP_SSR_SMP, TEX_2D, uv1).xyz();
        graph.colors[0] = nodes::vec4(nodes::mix(prev, nodes::min(outc, nodes::vec3(60000)), fs_in("bake_params", FLOAT, 1)), 1);
        if (ncuts) graph.fragment = drie::node_block(cuts[:ncuts]);
        return;
    }
    Node indirect = nodes::vec4(0, 0, rough, metal);
    if (key.mrt && ind_d.id) {
        Node lw = nodes::vec3(0.2126f, 0.7152f, 0.0722f);
        Node total = nodes::max(nodes::dot(outc, lw), 1e-6f);
        Node rg = nodes::clamp(nodes::vec2(nodes::dot(nodes::max(ind_d, 0), lw), nodes::dot(nodes::max(ind_s, 0), lw)) / total, 0, 1);
        indirect = nodes::vec4(rg, rough, metal);
    }
    Node shown_alpha = alpha;
    // a light source (emissive far over white) leaves the target's alpha at 0
    Node glow = fs_in("emissive", VEC3);
    alpha = alpha * nodes::smoothstep(4, 16, nodes::max(glow.x(), nodes::max(glow.y(), glow.z()))).one_minus();
    Node shown;
    if (key.linear) {
        shown = nodes::min(outc, nodes::vec3(60000));
        if (display_fog_t.id) shown = nodes::mix(fog_color.xyz(), shown, display_fog_t);
    } else {
        // (three mixes the fog in the colors as they are shown)
        shown = shading::tone_map_gamut(outc, key.tone, camera.w(), key.display != shading::DISPLAY_SRGB);
        bool hdr = key.display == shading::DISPLAY_HDR;
        if (!hdr) shown = shading::linear_to_srgb(shown);
        if (display_fog_t.id) shown = nodes::mix(hdr ? fog_color.xyz() : shading::linear_to_srgb(fog_color.xyz()), shown, display_fog_t);
    }
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

// Whether the key holds a slot for any map role.
fn bool MaterialKey.has_maps(self) => (bitcast(self, KeyWords).hi >> 64) != 0;

// Role `role`'s map at the material's uv (shader_core.glsl's material_map).
fn Node role_texel(MaterialKey key, FragUv* f, int role) @local => shading::ext_texel(f, key.slot(role) - 1);

// ---- the lit materials (lambert, phong, standard)

// A lit material's surface and what its light comes to (the material shader's lighting).
struct Lit @local {
    MaterialKey key;
    Node n;
    Node geo_n;            // the surface's own normal (without the normal map)
    Node frag;             // the pixel's position in the target
    int ltc_fn;            // ltc_evaluate's function (-1: not made yet)
    Parallax parallax;     // where the view ray meets the parallax map (key.parallax)
    FragUv* f;
    Node dpdx;             // how the world position changes from one pixel to the next
    Node dpdy;
    ShadowFns shadow;
    Node v;
    Node world;
    Node rough;
    Node metal;
    Node ao;               // the ambient occlusion of the ARM map (1: none)
    Node diffuse;          // the diffuse color
    Node f0;               // the standard material's reflectance head on, and at a glancing angle
    Node f90;
    Node v_normal;         // the vertex normal, and the vertex stage's tangent (tangent_frame)
    Node tangent;
    // the physical material's layers (where the key has them): a clear coat over the surface
    // (its strength, roughness and normal; what the lights' highlights on it add up to), a
    // sheen of fibers (its color and roughness; its light from the lights and all around), a
    // thin film's rainbow (its strength, and its Fresnel and reflectance head on), highlights
    // drawn out along aniso_t (its strength; aniso_t and across it aniso_b, in the tangent
    // frame's scale)
    Node cc;
    Node cc_rough;
    Node cc_n;
    Node cc_light;
    Node sheen_color;
    Node sheen_rough;
    Node sheen_light;
    Node irid;
    Node irid_fresnel;
    Node irid_f0;
    int irid_fn;
    int probe_fn;          // probe_sample's function (-1: not made yet)
    int ssr_fn;            // screen_reflection's
    Refraction refraction;
    Node aniso;
    Node aniso_t;
    Node aniso_b;
    // direct()'s light on the coat and on the sheen, for light_body
    Node coat_d;
    Node sheen_d;
    // the light on the surface: from the lights and all around and given off (all but its
    // highlights), its highlights (none: lambert, toon, basic) and what it gives off itself
    Node lights;
    Node highlights;
    Node glow;
    // its opacity, for glass
    Node alpha;
    // a node material's: emissive_node's glow (none: the material's), env_node (none: the
    // environment as it is); phong's specular color and shininess (specular_node,
    // shininess_node, or the material's)
    Node user_glow;
    Node user_env;
    Node user_cc;          // clearcoat_node, clearcoat_roughness_node (none: the material's)
    Node user_cc_rough;
    Node phong_specular;
    Node shininess;
    // what it comes to: all of the light, and the indirect diffuse part of it
    Node outc;
    Node ind_d;
    Node ind_s;
}

// The light of all the render's lights on the surface: `rgb` the material's color (with its
// map, `map_rgb`), `v_color` the vertex colors.
fn void Lit.shade(&self, FragUv* f, Node rgb, Node map_rgb, Node v_color) {
    MaterialKey key = self.key;
    Node n = self.n;
    self.dpdx = nodes::dfdx(self.world);
    self.dpdy = nodes::dfdy(self.world);
    Node ao = self.ao;
    self.diffuse = rgb;
    if (key.type == (char)MaterialType.STANDARD) {
        // what it reflects head on: 4% for glass's IOR of 1.5, tinted and scaled by the specular
        // color and intensity (KHR_materials_specular); metals reflect their own color
        Node ior_in = fs_in("transmission_params", FLOAT);
        Node ior = nodes::select(ior_in.greater_than(nodes::num(0)), ior_in, 1.5f);
        Node r0 = (ior - 1.0f) / (ior + 1.0f);
        Node sp = fs_in("specular_params");
        Node tint = sp.xyz();
        Node strength = sp.w();
        if (key.slot(MAP_SPECULAR_COLOR)) tint = tint * role_texel(key, f, MAP_SPECULAR_COLOR).xyz();
        if (key.slot(MAP_SPECULAR_INTENSITY)) strength = strength * role_texel(key, f, MAP_SPECULAR_INTENSITY).w();
        Node dielectric = nodes::min(nodes::vec3(r0 * r0) * tint, nodes::vec3(1)) * strength;
        self.f0 = nodes::mix(dielectric, rgb, self.metal);
        self.f90 = nodes::mix(strength, 1, self.metal);
        self.diffuse = rgb * self.metal.one_minus();
        self.layers(f, rgb, dielectric);
    } else if (key.env_reflect) {
        // (the reflection probes on a phong or toon material)
        self.f0 = nodes::mix(nodes::vec3(0.04f), rgb, self.metal);
        self.f90 = nodes::num(1);
    }
    Node diffuse = self.diffuse;

    // the light from all around (ambient and hemisphere lights) and baked into the light map
    Node irradiance = fs_in("ambient", VEC3) + nodes::mix(fs_in("hemi_ground", VEC3), fs_in("hemi_sky", VEC3), n.y() * 0.5f + 0.5f);
    Node env_params = fs_in("env_params");
    if (key.env_light) {
        // the scene's environment: the light from all around n, and its sun, unless a
        // directional light comes from where the sun is: that light is the sun (with its
        // shadow), the picture's is the same light again
        irradiance = irradiance + env_params.y() * env_irradiance(n);
        Node sun = nodes::variable(env_sun(n));
        if (key.dir_lights) {
            Node i = nodes::loop_index(UINT);
            Node dir = node_storage(shaders::VIEW_DIR_LIGHTS, VEC4, i * nodes::uint_(3));
            Node same = nodes::dot(nodes::normalize(dir.xyz()), env_sun_dir()).greater_than(nodes::num(0.996f));
            Node body = nodes::block(nodes::if_then(same, nodes::block(sun.assign(nodes::vec3(0)))));
            Node off = nodes::loop(i, nodes::uint_(0), nodes::convert(fs_in("counts", FLOAT), UINT), body);
            irradiance = irradiance + env_params.y() * nodes::seq(nodes::block(sun, off), sun);
        } else {
            irradiance = irradiance + env_params.y() * sun;
        }
    }
    if (key.gi) {
        // the scene's irradiance volumes: added to the light from all around, or in its place
        // where they cover the point (the first's gi_min.w 2)
        Node[2] gi = gi_irradiance(n, self.world);
        irradiance = nodes::select(fs_in("gi_min", FLOAT, 3).greater_than(nodes::num(1.5f)),
                                   irradiance * gi[1].one_minus() + gi[0], irradiance + gi[0]);
    }
    if (key.rc) {
        // radiance cascades: all the indirect light inside their box
        Node[2] rc = rc_irradiance(n, self.world);
        irradiance = nodes::mix(irradiance, rc[0], rc[1]);
    }
    if (key.slot(MAP_LIGHT)) irradiance = irradiance + role_texel(key, f, MAP_LIGHT).xyz() * fs_in("light_map_params", FLOAT);
    Node outc0 = irradiance * diffuse * shading::RECIPROCAL_PI * ao;
    Node sheen_ind;
    if (key.sheen) {
        // the sheen's light from all around, and what the fibers take of it
        Node albedo = sheen_albedo(n, self.v, self.sheen_rough);
        sheen_ind = irradiance * self.sheen_color * albedo * shading::RECIPROCAL_PI;
        outc0 = outc0 * (max3(self.sheen_color) * albedo).one_minus();
    }
    self.ind_d = outc0;

    // the lights, each added to outc and to spec, the highlights
    Node outc = nodes::variable(outc0);
    Node spec = nodes::variable(nodes::vec3(0));
    Node[20] stmts;
    int ns;
    stmts[ns++] = drie::node_eval(n);
    stmts[ns++] = drie::node_eval(self.v);
    stmts[ns++] = drie::node_eval(diffuse);
    stmts[ns++] = outc;
    stmts[ns++] = spec;
    if (key.clearcoat) {
        self.cc_light = nodes::variable(nodes::vec3(0));
        stmts[ns++] = self.cc_light;
    }
    if (key.sheen) {
        self.sheen_light = nodes::variable(sheen_ind);
        stmts[ns++] = self.sheen_light;
    }
    if (key.dir_lights) {
        Node i = nodes::loop_index(UINT);
        Node at = i * nodes::uint_(3);
        Node dir = node_storage(shaders::VIEW_DIR_LIGHTS, VEC4, at);
        Node color = node_storage(shaders::VIEW_DIR_LIGHTS, VEC4, at + nodes::uint_(1));
        Node shadow = node_storage(shaders::VIEW_DIR_LIGHTS, VEC4, at + nodes::uint_(2));
        Node l = nodes::normalize(dir.xyz());
        Node radiance = color.xyz();
        Node[6] pre;
        int np;
        Node skip;
        if (key.sun_shadow || key.dir_tiles || key.parallax) {
            Node world = self.world;
            Node lit = nodes::variable(nodes::num(1));
            pre[np++] = drie::node_eval(l);
            pre[np++] = lit;
            if (key.sun_shadow) {
                Node sun = nodes::convert(fs_in("dir_shadow_params", FLOAT), INT).equal(nodes::convert(i, INT) + nodes::int_(1));
                pre[np++] = nodes::if_then(sun, nodes::block(lit.assign(self.shadow.sun(n, l, world, self.frag))));
            }
            if (key.dir_tiles) {
                // the other shadowed ones: a tile of shadow_atlas each
                Node dsi = nodes::convert(shadow.x(), INT);
                Node tile = nodes::mix(1, self.shadow.dir_tile(dsi, n, l, world, self.frag), shadow.y());
                pre[np++] = nodes::if_then(dsi.greater_than_equal(nodes::int_(0)), nodes::block(lit.assign(tile)));
            }
            if (key.parallax) {
                // parallax_map: the bumps' own shadows, toward the first directional light
                Node bumps = parallax_shadow(&self.parallax, self.f, self.geo_n, l);
                pre[np++] = nodes::if_then(i.equal(nodes::uint_(0)), nodes::block(lit.assign(lit * bumps)));
            }
            skip = lit.less_than_equal(nodes::num(0));
            radiance = radiance * lit;
        }
        Node body = self.light_body(shadow, self.direct(l, radiance), outc, spec, skip, pre[:np]);
        stmts[ns++] = nodes::loop(i, nodes::uint_(0), nodes::convert(fs_in("counts", FLOAT), UINT), body);
    }
    if (key.debug_cascades) {
        // renderer.debug_cascades: red, green, blue, yellow
        Node k = sun_cascade(sun_depth(self.world));
        Node tint = nodes::select(k.equal(nodes::int_(0)), nodes::vec3(1, 0, 0),
                    nodes::select(k.equal(nodes::int_(1)), nodes::vec3(0, 1, 0),
                    nodes::select(k.equal(nodes::int_(2)), nodes::vec3(0, 0, 1), nodes::vec3(1, 1, 0))));
        stmts[ns++] = nodes::if_then(k.greater_than_equal(nodes::int_(0)), nodes::block(outc.assign(nodes::mix(outc, tint, 0.5f))));
    }
    // where a light with a size is mirrored from (light_toward)
    Node mirror_r = nodes::reflect(self.v.negate(), n);
    if (key.point_lights || key.spot_lights) {
        // the point and spot lights that reach the fragment's cluster
        // (baking: all of them, a light map's texel having no cluster)
        Node range;
        Node first;
        Node npoint;
        Node nspot;
        if (key.bake) {
            npoint = nodes::convert(fs_in("counts", FLOAT, 1), UINT);
            nspot = nodes::convert(fs_in("counts", FLOAT, 2), UINT);
        } else {
            range = node_storage(shaders::VIEW_LIGHT_CLUSTERS, UVEC4, nodes::convert(cluster_of(self.world), UINT));
            first = range.x();
            npoint = range.y();
            nspot = range.z();
            stmts[ns++] = drie::node_eval(range);
        }
        if (key.point_lights) {
            Node k = nodes::loop_index(UINT);
            Node at = (key.bake ? k : node_storage(shaders::VIEW_LIGHT_INDICES, UINT, first + k)) * nodes::uint_(4);
            Node pos = node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, at);
            Node color = node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, at + nodes::uint_(1));
            Node shadow = node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, at + nodes::uint_(2));
            Node params = node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, at + nodes::uint_(3));
            Node lv = pos.xyz() - self.world;
            Node d = nodes::length(lv);
            // a light source with a size: falloff softens up close instead of blowing out
            Node radius = params.x();
            Node seg = params.swizzle("yzw");
            Node l = lv / d;
            Node d_soft = nodes::sqrt(d * d + radius * radius);
            Node radiance = color.xyz() * distance_attenuation(d_soft, pos.w(), color.w());
            Sized size;
            if (key.sized) {
                size = light_toward(mirror_r, lv, seg, radius, key.tubes);
                size.on = (radius + nodes::abs(seg.x()) + nodes::abs(seg.y()) + nodes::abs(seg.z())).greater_than(nodes::num(0));
            }
            if (key.tubes) {
                // a tube (PointLight.length): the light of the whole segment, its intensity
                // spread along it (Karis' eq. 16, the radius softening it as a ball's falloff:
                // d^2 + radius^2 at length 0), faded out by the distance to its nearest point
                Node tube = nodes::dot(seg, seg).greater_than(nodes::num(0));
                Node l0 = lv - seg;
                Node l1 = lv + seg;
                Node d0 = nodes::length(l0);
                Node d1 = nodes::length(l1);
                Node ld = seg * 2.0f;
                Node near = l0 + ld * nodes::clamp(nodes::dot(l0, ld).negate() / nodes::dot(ld, ld), 0, 1);
                Node dn = nodes::length(near);
                Node across = nodes::clamp(nodes::dot(n, l0) * 0.5f / d0 + nodes::dot(n, l1) * 0.5f / d1, 0, 1) * 2.0f;
                Node nol = across / nodes::max(d0 * d1 + nodes::dot(l0, l1) + radius * radius * 2.0f, 1e-4f);
                size.nol = nodes::select(tube, nol, -1.0f);
                l = nodes::select(tube, near / nodes::max(dn, 1e-6f), l);
                radiance = nodes::select(tube, color.xyz() * distance_window(dn, pos.w()), radiance);
            }
            if (key.fog) radiance = radiance * shading::fog_transmittance(key, shading::fog_distance(key, pos.xyz(), self.world));
            Node[2] pre;
            int np;
            if (key.point_shadows) {
                Node shade = nodes::variable(nodes::num(1));
                Node si = nodes::convert(shadow.x(), INT);
                Node lit = nodes::mix(1, self.shadow.point(pos.xyz(), si, self.geo_n, self.world, self.dpdx, self.dpdy, self.frag), shadow.y());
                pre[np++] = shade;
                pre[np++] = nodes::if_then(si.greater_than_equal(nodes::int_(0)), nodes::block(shade.assign(lit)));
                radiance = radiance * shade;
            }
            Node body = self.light_body(shadow, self.direct(l, radiance, &size), outc, spec, { }, pre[:np]);
            stmts[ns++] = nodes::loop(k, nodes::uint_(0), npoint, body);
        }
        if (key.spot_lights) {
            Node k = nodes::loop_index(UINT);
            Node at = (key.bake ? k : node_storage(shaders::VIEW_LIGHT_INDICES, UINT, first + npoint + k)) * nodes::uint_(5);
            Node pos = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at);
            Node dir = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at + nodes::uint_(1));
            Node color = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at + nodes::uint_(2));
            Node params = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at + nodes::uint_(3));
            Node shadow = node_storage(shaders::VIEW_SPOT_LIGHTS, VEC4, at + nodes::uint_(4));
            Node lv = pos.xyz() - self.world;
            Node d = nodes::length(lv);
            Node l = lv / d;
            Node cone = nodes::smoothstep(dir.w(), params.x(), nodes::dot(l.negate(), dir.xyz()));
            Node d_soft = nodes::sqrt(d * d + params.y() * params.y());
            Node radiance = color.xyz() * cone * distance_attenuation(d_soft, pos.w(), color.w());
            Sized size;
            if (key.sized) {
                size = light_toward(mirror_r, lv, { }, params.z(), false);
                size.on = params.z().greater_than(nodes::num(0));
            }
            if (key.fog) radiance = radiance * shading::fog_transmittance(key, shading::fog_distance(key, pos.xyz(), self.world));
            // out of the light's reach, or of its cone
            Node away = nodes::logical_or(
                nodes::logical_and(pos.w().greater_than(nodes::num(0)), d.greater_than(pos.w())),
                cone.less_than_equal(nodes::num(0)));
            Node[2] pre;
            int np;
            if (key.spot_shadows) {
                Node shade = nodes::variable(nodes::num(1));
                Node si = nodes::convert(shadow.x(), INT);
                Node lit = nodes::mix(1, self.shadow.light(pos.xyz(), si, self.geo_n, self.world, self.dpdx, self.dpdy, self.frag), shadow.y());
                pre[np++] = shade;
                pre[np++] = nodes::if_then(si.greater_than_equal(nodes::int_(0)), nodes::block(shade.assign(lit)));
                radiance = radiance * shade;
            }
            Node body = self.light_body(shadow, self.direct(l, radiance, &size), outc, spec, away, pre[:np]);
            stmts[ns++] = nodes::loop(k, nodes::uint_(0), nspot, body);
        }
    }
    if (key.rect_lights) stmts[ns++] = self.rect_lights(outc, spec);
    Node lit = nodes::seq(drie::node_block(stmts[:ns]), outc);
    Node highlights = spec;
    // three's specularMap: the highlights' strength
    if (key.type == (char)MaterialType.PHONG && key.slot(MAP_SPECULAR)) highlights = highlights * role_texel(key, f, MAP_SPECULAR).x();
    bool shiny = key.type != (char)MaterialType.LAMBERT && key.type != (char)MaterialType.TOON;
    if (shiny) self.highlights = highlights * ao;
    // what it gives off itself: Material.map_emissive (the color map glows too), three's emissiveMap
    Node glow = fs_in("emissive", VEC3);
    if (key.map_emissive && map_rgb.id) glow = glow * map_rgb;
    if (key.slot(MAP_EMISSIVE)) glow = glow * role_texel(key, f, MAP_EMISSIVE).xyz();
    if (self.user_glow.id) glow = self.user_glow;
    self.glow = glow;
    Node result = lit + glow;
    // light baked into the vertex colors, as if from fixed lamps
    if (key.baked) {
        Node b = diffuse * v_color * fs_in("emissive", FLOAT, 3) * ao;
        result = result + b;
        self.ind_d = self.ind_d + b;
    }
    self.lights = result;
    self.finish();
}

// What comes over the light on the surface (self.lights, its highlights and the light it
// gives off), for every material type: glass's reflections; or the environment's reflection,
// the mirror images over it, the light through a transmission material and the physical
// material's sheen and clear coat.
fn void Lit.finish(&self) {
    MaterialKey key = self.key;
    Node n = self.n;
    Node ao = self.ao;
    FragUv* f = self.f;
    if (key.glass) {
        // glass: reflections stay visible however clear the pane is
        Node v = self.v;
        Node fresnel = nodes::pow(nodes::clamp(nodes::dot(n, v), 0, 1).one_minus(), 5) * 0.96f + 0.04f;
        Node r = nodes::reflect(v.negate(), n);
        Node sky;
        if (key.env_reflect) {
            sky = env_sample(key, &self.probe_fn, r, self.rough, self.world, self.frag, n);
        } else {
            sky = nodes::mix(fs_in("hemi_ground", VEC3), fs_in("hemi_sky", VEC3), r.y() * 0.5f + 0.5f) + fs_in("ambient", VEC3);
        }
        sky = self.env_node(sky, r, self.rough);
        Node spec = sky * fresnel;
        if (self.highlights.id) spec = self.highlights + spec;
        Node a = nodes::clamp(self.alpha + fresnel + max3(spec), 0, 1);
        // blended with src-alpha: pre-divided so that outc * alpha + spec comes out
        self.outc = (self.lights * self.alpha + spec) / nodes::max(a, 1e-4f);
        self.alpha = a;
        self.ind_s = nodes::vec3(0);
        return;
    }
    self.ind_s = nodes::vec3(0);
    Node env_term;
    if (key.env_reflect) {
        // reflections of the surroundings, fading as the surface gets rougher
        Node v = self.v;
        Node rough = self.rough;
        Node nov = nodes::clamp(nodes::dot(n, v), 0, 1);
        Node f0 = self.f0;
        if (key.iridescence) f0 = nodes::select(self.irid.greater_than(nodes::num(0)), nodes::mix(f0, self.irid_f0, self.irid), f0);
        Node fr = f0 + (nodes::max(nodes::vec3(rough.one_minus() * self.f90), f0) - f0) * nodes::pow(nov.one_minus(), 5);
        Node rn = n;
        if (key.anisotropy) {
            // the normal bent toward the stretched highlight (three's getIBLAnisotropyRadiance,
            // Filament's)
            Node bent = nodes::normalize(nodes::cross(nodes::cross(self.aniso_b, v), self.aniso_b));
            Node k = (self.aniso * rough.one_minus()).one_minus();
            k = k * k;
            rn = nodes::select(self.aniso.greater_than(nodes::num(0)), nodes::normalize(nodes::mix(bent, n, k * k)), n);
        }
        Node r = dominant_reflection(nodes::reflect(v.negate(), rn), rn, rough);
        env_term = self.env_node(env_sample(key, &self.probe_fn, r, rough, self.world, self.frag, n), r, rough) * fr;
        if (key.sheen) env_term = env_term * (max3(self.sheen_color) * sheen_albedo(n, v, self.sheen_rough)).one_minus();
        self.ind_s = env_term;
    }
    Node mirror_term;
    if (key.ssr || key.reflector) {
        // the surface's sheen: a mirror image, strongest at a glancing angle (far off, where a
        // pixel covers many bumps of the normal map, each would throw the image somewhere
        // else: blotches. There the surface mirrors as if flat)
        Node v = self.v;
        Node bumps = nodes::smoothstep(1.5f, 5.0f, nodes::length(fs_in("camera_pos", VEC3) - self.world)).one_minus() * 0.5f;
        Node rn = nodes::normalize(nodes::mix(self.geo_n, n, bumps));
        Node fresnel = nodes::pow(nodes::clamp(nodes::dot(rn, v), 0, 1).one_minus(), 5) * 0.96f + 0.04f;
        Node mirrored;
        Node found;
        Node strength;
        if (key.reflector) {
            // the planar reflection: the picture of the mirrored scene where the ray points
            Node seen = planar_reflection(rn, v, self.rough, self.world, self.frag);
            mirrored = seen.xyz();
            found = seen.w() * fs_in("reflector_params", FLOAT);
            strength = nodes::num(1);
        }
        if (key.ssr) {
            // what is on screen near it, where a ray finds that: sharper, at full size
            Node near = screen_reflection(&self.ssr_fn, rn, self.geo_n, v, self.rough, self.world, self.frag);
            Node ssr = fs_in("ssr_params", FLOAT);
            if (key.reflector) {
                mirrored = nodes::select(found.greater_than(nodes::num(0)), nodes::mix(mirrored, near.xyz(), near.w()), near.xyz());
                found = nodes::max(found, near.w());
                strength = nodes::max(strength, ssr);
            } else {
                mirrored = near.xyz();
                found = near.w();
                strength = ssr;
            }
        }
        // where something was found, it takes the place of the surroundings' map
        mirror_term = mirrored * (found * fresnel * strength * ao);
        if (env_term.id) env_term = env_term * nodes::clamp(found * strength, 0, 1).one_minus();
        self.ind_s = env_term.id ? env_term + mirror_term : mirror_term;
    }
    Node result = self.lights;
    if (env_term.id) result = result + env_term;
    if (mirror_term.id) result = result + mirror_term;
    self.outc = self.highlights.id ? result + self.highlights : result;
    if (key.transmission) {
        // what is behind the surface, refracted through the volume, takes the place of its
        // diffuse light; highlights, reflections and the light it gives off stay
        Node amount = fs_in("transmission_color", FLOAT, 3);
        if (key.slot(MAP_TRANSMISSION)) amount = amount * role_texel(key, f, MAP_TRANSMISSION).x();
        Node thickness = nodes::max(fs_in("transmission_params", FLOAT, 1), 0);
        if (key.slot(MAP_THICKNESS)) thickness = thickness * role_texel(key, f, MAP_THICKNESS).y();
        Node refracted = self.refraction.light(n, self.v, self.world, self.rough, self.diffuse, thickness, self.f0, self.f90);
        Node kept = self.highlights + self.glow;
        if (env_term.id) kept = kept + env_term;
        if (mirror_term.id) kept = kept + mirror_term;
        self.outc = nodes::mix(nodes::max(self.outc - kept, nodes::vec3(0)), refracted, amount) + kept;
    }
    // the physical material's layers over it all: the sheen, then the clear coat, which lets
    // through what it does not reflect (three's)
    if (key.sheen) {
        self.outc = self.outc + self.sheen_light;
        self.ind_s = self.ind_s + self.sheen_light;
    }
    if (key.clearcoat) {
        Node cn = self.cc_n;
        Node v = self.v;
        Node coat = self.cc_light;
        if (key.env_reflect) {
            Node r = dominant_reflection(nodes::reflect(v.negate(), cn), cn, self.cc_rough);
            Node env = self.env_node(env_sample(key, &self.probe_fn, r, self.cc_rough, self.world, self.frag, cn), r, self.cc_rough);
            Node reflected = env * environment_brdf(cn, v, nodes::vec3(0.04f), nodes::num(1), self.cc_rough);
            coat = coat + reflected;
            self.ind_s = self.ind_s + reflected * self.cc;
        }
        Node fc = schlick(nodes::num(0.04f), nodes::num(1), nodes::clamp(nodes::dot(cn, v), 0, 1));
        self.outc = self.outc * (self.cc * fc).one_minus() + coat * self.cc;
    }
}

// The environment's reflection `env` in direction `dir` at roughness `rough`, or a node
// material's env_node in its place (node_env), which reads them as ENV_DIRECTION and
// ENV_ROUGHNESS.
fn Node Lit.env_node(&self, Node env, Node dir, Node rough) {
    if (!self.user_env.id) return env;
    return bound(nodes::convert(self.user_env, VEC3), { rebind(ENV_DIRECTION, dir), rebind(ENV_ROUGHNESS, rough),
                                                        rebind(NORMAL_WORLD, self.n), rebind(NORMAL_GEOMETRY, self.geo_n),
                                                        rebind(ENV_COLOR, env) });
}

// The physical material's layers (key.clearcoat, sheen, iridescence, anisotropy) from its
// parameters and maps; `base` its color, `dielectric` its reflectance head on as a dielectric.
fn void Lit.layers(&self, FragUv* f, Node base, Node dielectric) {
    MaterialKey key = self.key;
    Node geo_n = self.geo_n;
    if (key.clearcoat) {
        // the clear coat: its strength and roughness, and its normal (the surface's own, under
        // the normal map, or from its own map)
        Node p = fs_in("clearcoat_params");
        Node cc = p.x();
        if (key.slot(MAP_CLEARCOAT)) cc = cc * role_texel(key, f, MAP_CLEARCOAT).x();
        // (a node material's clearcoat_node and clearcoat_roughness_node in their place)
        if (self.user_cc.id) cc = bound(nodes::convert(self.user_cc, FLOAT), { rebind(NORMAL_WORLD, self.n), rebind(NORMAL_GEOMETRY, geo_n) });
        self.cc = nodes::clamp(cc, 0, 1);
        Node rough = p.y();
        if (key.slot(MAP_CLEARCOAT_ROUGHNESS)) rough = rough * role_texel(key, f, MAP_CLEARCOAT_ROUGHNESS).y();
        if (self.user_cc_rough.id) {
            rough = bound(nodes::convert(self.user_cc_rough, FLOAT), { rebind(NORMAL_WORLD, self.n), rebind(NORMAL_GEOMETRY, geo_n) });
        }
        self.cc_rough = nodes::clamp(rough, 0.045f, 1);
        self.cc_n = geo_n;
        if (key.slot(MAP_CLEARCOAT_NORMAL)) {
            Node t = role_texel(key, f, MAP_CLEARCOAT_NORMAL).xyz() * 2.0f - 1.0f;
            t = nodes::vec3(t.xy() * p.swizzle("zw"), t.z());
            int u = key.slot(MAP_CLEARCOAT_NORMAL) + 1;
            Node[2] tb = tangent_frame(key, geo_n, self.v_normal, self.world, self.tangent, shading::slot_d(f, u, false), shading::slot_d(f, u, true));
            self.cc_n = nodes::normalize(tb[0] * t.x() + tb[1] * t.y() + geo_n * t.z());
        }
    }
    if (key.sheen) {
        // the sheen (0 roughness: none)
        Node p = fs_in("sheen_params");
        self.sheen_color = p.xyz();
        self.sheen_rough = nodes::clamp(p.w(), 1e-4f, 1);
        if (key.slot(MAP_SHEEN_COLOR)) self.sheen_color = self.sheen_color * role_texel(key, f, MAP_SHEEN_COLOR).xyz();
        if (key.slot(MAP_SHEEN_ROUGHNESS)) self.sheen_rough = self.sheen_rough * role_texel(key, f, MAP_SHEEN_ROUGHNESS).w();
    }
    if (key.iridescence) {
        // iridescence (three's): the film's reflectance over a dielectric and over a metal
        Node p = fs_in("iridescence_params");
        Node irid = p.x();
        if (key.slot(MAP_IRIDESCENCE)) irid = irid * role_texel(key, f, MAP_IRIDESCENCE).x();
        Node film = p.w();
        if (key.slot(MAP_IRIDESCENCE_THICKNESS)) film = nodes::mix(p.z(), p.w(), role_texel(key, f, MAP_IRIDESCENCE_THICKNESS).y());
        self.irid = nodes::select(film.equal(nodes::num(0)), 0.0f, nodes::clamp(irid, 0, 1));
        Node nov = nodes::clamp(nodes::dot(self.n, self.v), 0, 1);
        Node over_dielectric = eval_iridescence(&self.irid_fn, nodes::num(1), p.y(), nov, film, dielectric);
        Node over_metal = eval_iridescence(&self.irid_fn, nodes::num(1), p.y(), nov, film, base);
        self.irid_fresnel = nodes::mix(over_dielectric, over_metal, self.metal);
        self.irid_f0 = nodes::mix(schlick_to_f0(over_dielectric, nodes::num(1), nov), schlick_to_f0(over_metal, nodes::num(1), nov), self.metal);
    }
    if (key.anisotropy) {
        // anisotropy (three's): highlights drawn out along a direction of the surface, as on
        // brushed metal. The direction in the tangent frame, turned by the map's (rg, around its
        // middle) and scaled by its b
        Node dir = fs_in("tangent_params", VEC2, 2);
        if (key.slot(MAP_ANISOTROPY)) {
            Node polar = role_texel(key, f, MAP_ANISOTROPY).xyz();
            Node turn = polar.xy() * 2.0f - 1.0f;
            turn = turn * (polar.z() / nodes::max(nodes::length(turn), 1e-6f));
            dir = nodes::vec2(dir.x() * turn.x() - dir.y() * turn.y(), dir.y() * turn.x() + dir.x() * turn.y());
        }
        Node amount = nodes::length(dir);
        dir = nodes::select(amount.greater_than(nodes::num(0)), dir / amount, nodes::vec2(1, 0));
        self.aniso = nodes::min(amount, 1);
        // the frame: the normal map's uv, or the clear coat's normal map's, or the color map's
        Node duv1 = f.dx;
        Node duv2 = f.dy;
        if (key.normal_map) {
            duv1 = shading::slot_d(f, 0, false);
            duv2 = shading::slot_d(f, 0, true);
        } else if (key.slot(MAP_CLEARCOAT_NORMAL)) {
            int u = key.slot(MAP_CLEARCOAT_NORMAL) + 1;
            duv1 = shading::slot_d(f, u, false);
            duv2 = shading::slot_d(f, u, true);
        }
        Node[2] tb = tangent_frame(key, geo_n, self.v_normal, self.world, self.tangent, duv1, duv2);
        self.aniso_t = tb[0] * dir.x() + tb[1] * dir.y();
        self.aniso_b = tb[1] * dir.x() - tb[0] * dir.y();
    }
}

// A light's loop body: skipped where the light is in none of the material's channels; then the
// statements `pre` (its shadow), and skipped where `skip` holds, else its diffuse light `d[0]`
// added to `outc` and its highlight `d[1]` to `spec`.
fn Node Lit.light_body(&self, Node shadow, Node[2] d, Node outc, Node spec, Node skip = { }, Node[] pre = { }) {
    Node[12] st;
    int n;
    if (self.key.channels) {
        // (lights_material: a light of none of the material's channels)
        Node taken = nodes::convert(fs_in("surface_params", FLOAT, 3), UINT);
        Node none = nodes::bit_and(taken, nodes::convert(shadow.z(), UINT)).equal(nodes::uint_(0));
        st[n++] = nodes::if_then(none, nodes::block(nodes::continue_()));
    }
    foreach (p : pre) st[n++] = p;
    if (skip.id) st[n++] = nodes::if_then(skip, nodes::block(nodes::continue_()));
    st[n++] = outc.assign(outc + d[0]);
    if (d[1].id) st[n++] = spec.assign(spec + d[1]);
    // (the physical material's layers: what direct() left for the coat and the sheen)
    if (self.coat_d.id) st[n++] = self.cc_light.assign(self.cc_light + self.coat_d);
    if (self.sheen_d.id) st[n++] = self.sheen_light.assign(self.sheen_light + self.sheen_d);
    self.coat_d = { };
    self.sheen_d = { };
    return drie::node_block(st[:n]);
}

// RectAreaLight, as three's (lights_physical_pars_fragment: LTC_Uv, LTC_Evaluate and
// RE_Direct_RectArea_Physical), with linearly transformed cosines (Heitz, Dupuy, Hill and
// Neubelt 2016). The lights are in point_lights after the point lights: pos = centre + half
// width's x, color = color * intensity + half width's y, shadow.z = its channels, params = half
// width's z + half height; and after them the two LTC tables (clusters.c3's ltc_entries), a
// texel an entry (pos = LTC_1's, color = LTC_2's), read here with the bilinear filter three's
// textures have. The loop adds their diffuse light to `outc` and their highlights to `spec`.
fn Node Lit.rect_lights(&self, Node outc, Node spec) {
    MaterialKey key = self.key;
    Node n = self.n;
    Node v = self.v;
    Node counts = fs_in("counts");
    Node npoint = nodes::convert(counts.y(), INT);
    Node nrect = nodes::convert(counts.w(), INT);
    Node[2] t = ltc_tables(npoint + nrect, n, v, nodes::max(self.rough, 0.0525f));
    Node t1 = t[0];
    Node t2 = t[1];
    Node[3] inv_cols = { nodes::vec3(t1.x(), 0, t1.y()), nodes::vec3(0, 1, 0), nodes::vec3(t1.z(), 0, t1.w()) };
    Node m_inv = node_matrix(&inv_cols, MAT3);
    Node[3] id_cols = { nodes::vec3(1, 0, 0), nodes::vec3(0, 1, 0), nodes::vec3(0, 0, 1) };
    Node ident = node_matrix(&id_cols, MAT3);
    Node fresnel = self.f0 * t2.x() + (nodes::vec3(self.f90) - self.f0) * t2.y();
    Node cc_inv;
    Node cc_fresnel;
    if (key.clearcoat) {
        Node[2] c = ltc_tables(npoint + nrect, self.cc_n, v, nodes::max(self.cc_rough, 0.0525f));
        Node[3] cols = { nodes::vec3(c[0].x(), 0, c[0].y()), nodes::vec3(0, 1, 0), nodes::vec3(c[0].z(), 0, c[0].w()) };
        cc_inv = node_matrix(&cols, MAT3);
        cc_fresnel = c[1].x() * 0.04f + c[1].y() * 0.96f;
    }

    Node k = nodes::loop_index();
    Node at = (npoint + k) * nodes::int_(4);
    Node pos = point_light_row(at);
    Node color = point_light_row(at + nodes::int_(1));
    Node shadow = point_light_row(at + nodes::int_(2));
    Node params = point_light_row(at + nodes::int_(3));
    Node hw = nodes::vec3(pos.w(), color.w(), params.x());
    Node hh = params.swizzle("yzw");
    Node to = pos.xyz() - self.world;
    // behind the light (it shines from its front only)
    Node behind = nodes::dot(nodes::cross(hw, hh), to).less_than(nodes::num(0));
    Node[4] q = { to + hw - hh, to - hw - hh, to - hw + hh, to + hw + hh };
    Node c = color.xyz();
    if (key.fog) c = c * shading::fog_transmittance(key, shading::fog_distance(key, pos.xyz(), self.world));
    Node[2] d = { c * self.diffuse * self.ltc_evaluate(n, v, ident, &q), c * fresnel * self.ltc_evaluate(n, v, m_inv, &q) };
    if (key.clearcoat) self.coat_d = c * cc_fresnel * self.ltc_evaluate(self.cc_n, v, cc_inv, &q);
    Node body = self.light_body(shadow, d, outc, spec, behind);
    return nodes::seq(nodes::block(drie::node_eval(t1), drie::node_eval(t2), nodes::loop(k, 0, nrect, body)), outc);
}

// The two LTC tables at roughness `rough` and the angle of v to n (three's LTC_Uv: roughness
// across, sqrt(1 - n.v) down, at the texels' centres), read bilinearly from point_lights at
// `base`.
fn Node[2] ltc_tables(Node base, Node n, Node v, Node rough) @local {
    Node uv = nodes::vec2(rough, nodes::sqrt(nodes::clamp(nodes::dot(n, v), 0, 1).one_minus())) * (63.0f / 64.0f) + 0.5f / 64.0f;
    Node p = uv * 64.0f - 0.5f;
    Node fr = nodes::fract(p);
    Node fl = nodes::floor(p);
    Node i0x = nodes::clamp(nodes::convert(fl.x(), INT), nodes::int_(0), nodes::int_(63));
    Node i0y = nodes::clamp(nodes::convert(fl.y(), INT), nodes::int_(0), nodes::int_(63));
    Node i1x = nodes::min(i0x + nodes::int_(1), nodes::int_(63));
    Node i1y = nodes::min(i0y + nodes::int_(1), nodes::int_(63));
    Node ta = (base + i0y * nodes::int_(64) + i0x) * nodes::int_(4);
    Node tb = (base + i0y * nodes::int_(64) + i1x) * nodes::int_(4);
    Node tc = (base + i1y * nodes::int_(64) + i0x) * nodes::int_(4);
    Node td = (base + i1y * nodes::int_(64) + i1x) * nodes::int_(4);
    Node[2] t;
    for (int e = 0; e < 2; e++) {
        Node one = nodes::int_(e);
        Node top = nodes::mix(point_light_row(ta + one), point_light_row(tb + one), fr.x());
        Node bottom = nodes::mix(point_light_row(tc + one), point_light_row(td + one), fr.x());
        t[e] = nodes::mix(top, bottom, fr.y());
    }
    return t;
}

fn Node point_light_row(Node index) @local => node_storage(shaders::VIEW_POINT_LIGHTS, VEC4, index);

// How much of the cosine lobe, transformed by m_inv, the rectangle (corners q, from the
// surface) covers: three's LTC_Evaluate.
fn Node Lit.ltc_evaluate(&self, Node n, Node v, Node m_inv, Node[4]* q) {
    if (self.ltc_fn < 0) {
        NodeType[7] types = { VEC3, VEC3, MAT3, VEC3, VEC3, VEC3, VEC3 };
        int f = node_func("ltc_evaluate", FLOAT, &types);
        Node pn = node_param(f, 0);
        Node pv = node_param(f, 1);
        Node pm = node_param(f, 2);
        Node t1 = nodes::normalize(pv - pn * nodes::dot(pv, pn));
        Node t2 = nodes::cross(pn, t1).negate();
        // (transpose(mat3(t1, t2, n)))
        Node[3] cols = { nodes::vec3(t1.x(), t2.x(), pn.x()), nodes::vec3(t1.y(), t2.y(), pn.y()), nodes::vec3(t1.z(), t2.z(), pn.z()) };
        Node m = pm * node_matrix(&cols, MAT3);
        Node[4] c;
        for (int i = 0; i < 4; i++) c[i] = nodes::normalize(m * node_param(f, 3 + i));
        Node e = ltc_edge(c[0], c[1]) + ltc_edge(c[1], c[2]) + ltc_edge(c[2], c[3]) + ltc_edge(c[3], c[0]);
        // three's LTC_ClippedSphereFormFactor
        Node l = nodes::length(e);
        node_func_body(f, nodes::max((l * l + e.z()) / (l + 1.0f), 0));
        self.ltc_fn = f;
    }
    Node[7] args = { n, v, m_inv, (*q)[0], (*q)[1], (*q)[2], (*q)[3] };
    return node_call_fn(self.ltc_fn, &args);
}

fn Node ltc_edge(Node v1, Node v2) @local {
    Node x = nodes::dot(v1, v2);
    Node y = nodes::abs(x);
    Node a = (y * 0.0145206f + 0.4965155f) * y + 0.8543985f;
    Node b = (y + 4.1616724f) * y + 3.4175940f;
    Node k = a / b;
    Node far = nodes::inverse_sqrt(nodes::max((x * x).one_minus(), 1e-7f)) * 0.5f - k;
    return nodes::cross(v1, v2) * nodes::select(x.greater_than(nodes::num(0)), k, far);
}

// The light of `radiance` from direction `l` (toward it) on the surface: its diffuse light and
// its highlight (shader_core.glsl's direct(), three's physically based light units).
fn Node[2] Lit.direct(&self, Node l, Node radiance, Sized* size = null) {
    MaterialKey key = self.key;
    Node n = self.n;
    Node v = self.v;
    Node ndl = nodes::dot(n, l);
    Node nol;
    if (key.foliage) {
        // thin leaves and blades, lit from either side: on the side the light falls on, as any
        // surface, with a little of it wrapping round; from the other, about half of it
        Node through = nodes::select((nodes::dot(n, v) * ndl).less_than(nodes::num(0)), 0.5f, 1.0f);
        nol = through * (nodes::abs(ndl) * 0.9f + 0.1f);
    } else {
        nol = nodes::clamp(ndl, 0, 1);
    }
    // the toon material: the gradient's step in place of n.l, and no highlights
    if (key.type == (char)MaterialType.TOON) return { toon_gradient(key, ndl) * radiance * self.diffuse * shading::RECIPROCAL_PI, { } };
    // (a tube's cosine over the squared distance, integrated along it)
    Node irradiance = size && size.nol.id ? nodes::select(size.nol.greater_than_equal(nodes::num(0)), size.nol, nol) * radiance : nol * radiance;
    Node h = nodes::normalize(l + v);
    Node sized = size && size.on.id ? size.on : (Node){ };
    if (key.clearcoat) {
        // the coat's own highlight, off its own normal (f0 0.04: an IOR of 1.5); a light with a
        // size: the point of it the surface under the coat mirrors, as bright as the coat's
        // roughness keeps it
        Node cn = self.cc_n;
        Node a = self.cc_rough * self.cc_rough;
        Node cnl = nodes::clamp(nodes::dot(cn, l), 0, 1);
        Node cnv = nodes::clamp(nodes::dot(cn, v), 1e-4f, 1);
        Node cnorm = sized.id ? nodes::select(sized, light_norm(size.k, a), 1.0f) : nodes::num(1);
        Node cl = sized.id ? nodes::select(sized, size.l, l) : l;
        Node ch = nodes::normalize(cl + v);
        Node cnh = nodes::clamp(nodes::dot(cn, ch), 0, 1);
        Node fc = schlick(nodes::num(0.04f), nodes::num(1), nodes::clamp(nodes::dot(v, ch), 0, 1));
        Node ccl = nodes::clamp(nodes::dot(cn, cl), 0, 1);
        self.coat_d = cnl * radiance * fc * v_smith(cnv, ccl, a) * d_ggx(cnh, a) * cnorm;
    }
    if (key.sheen) {
        Node snv = nodes::clamp(nodes::dot(n, v), 0, 1);
        Node snl = nodes::clamp(nodes::dot(n, l), 0, 1);
        self.sheen_d = irradiance * self.sheen_color * d_charlie(self.sheen_rough, nodes::clamp(nodes::dot(n, h), 0, 1)) * v_neubelt(snv, snl);
        // what the fibers take is not left for the surface under them
        Node albedo = nodes::max(sheen_albedo(n, v, self.sheen_rough), sheen_albedo(n, l, self.sheen_rough));
        irradiance = irradiance * (max3(self.sheen_color) * albedo).one_minus();
    }
    Node result = irradiance * self.diffuse * shading::RECIPROCAL_PI;
    if (key.type == (char)MaterialType.LAMBERT) return { result, { } };
    Node noh = nodes::clamp(nodes::dot(n, h), 0, 1);
    Node voh = nodes::clamp(nodes::dot(v, h), 0, 1);
    if (key.type == (char)MaterialType.PHONG) {
        // (at least 1e-4 as three's: pow(0, 0) is undefined)
        Node shininess = nodes::max(self.shininess, 1e-4f);
        Node d = shading::RECIPROCAL_PI * (shininess * 0.5f + 1.0f) * nodes::pow(noh, shininess);
        Node f0 = self.phong_specular;
        Node fresnel = f0 + (nodes::vec3(1) - f0) * nodes::pow(voh.one_minus(), 5);
        return { result, irradiance * fresnel * 0.25f * d };
    }
    Node rough = nodes::max(self.rough, 0.0525f);
    Node a = rough * rough;
    Node snl = nol;
    Node sl = l;
    Node norm;
    if (sized.id) {
        // a light with a size: the highlight is the light's own shape, seen in the surface
        // (light_toward), as bright in all as the point light's
        Node sh = nodes::normalize(size.l + v);
        h = nodes::select(sized, sh, h);
        sl = nodes::select(sized, size.l, l);
        noh = nodes::select(sized, nodes::clamp(nodes::dot(n, sh), 0, 1), noh);
        voh = nodes::select(sized, nodes::clamp(nodes::dot(v, sh), 0, 1), voh);
        snl = nodes::select(sized, nodes::clamp(nodes::dot(n, size.l), 0, 1), nol);
        norm = nodes::select(sized, light_norm(size.k, a), 1.0f);
    }
    Node nov = nodes::clamp(nodes::abs(nodes::dot(n, v)), 1e-4f, 1);
    Node fresnel = self.f0 + (nodes::vec3(self.f90) - self.f0) * nodes::pow(voh.one_minus(), 5);
    if (key.iridescence) fresnel = nodes::select(self.irid.greater_than(nodes::num(0)), nodes::mix(fresnel, self.irid_fresnel, self.irid), fresnel);
    Node vis = v_smith(nov, snl, a);
    Node d = d_ggx(noh, a);
    if (key.anisotropy) {
        // rougher along aniso_t, by as much as it is anisotropic (three's alphaT)
        Node at = nodes::max(nodes::mix(a, 1, self.aniso * self.aniso), a);
        Node t = self.aniso_t;
        Node b = self.aniso_b;
        Node on = self.aniso.greater_than(nodes::num(0));
        vis = nodes::select(on, v_smith_aniso(at, a, nodes::dot(t, v), nodes::dot(b, v), nodes::dot(t, sl), nodes::dot(b, sl), nov, snl), vis);
        d = nodes::select(on, d_ggx_aniso(at, a, noh, nodes::dot(t, h), nodes::dot(b, h)), d);
    }
    Node highlight = irradiance * fresnel * vis * d;
    return { result, norm.id ? highlight * norm : highlight };
}

// The light the highlight of a light with a size keeps at roughness alpha `a`: a / a' along a
// tube (eq. 20) times (a / a')^2 for its ball (eq. 14), a' = saturate(a + half the length, or
// the radius, / (3 distance)) (eq. 10); `k` from light_toward.
fn Node light_norm(Node k, Node a) @local {
    Node kb = a / nodes::clamp(a + k.y(), 0, 1);
    return a / nodes::clamp(a + k.x(), 0, 1) * kb * kb;
}

// The light the scene's environment puts on a surface facing n, but for its sun (env_sun):
// env_sh's 9 spherical harmonics, in the order of texture.c3's sh_basis.
fn Node env_irradiance(Node n) @local {
    Node e = fs_in("env_sh", VEC3) + fs_in("env_sh", VEC3, k: 1) * n.y() + fs_in("env_sh", VEC3, k: 2) * n.z()
        + fs_in("env_sh", VEC3, k: 3) * n.x() + fs_in("env_sh", VEC3, k: 4) * (n.x() * n.y())
        + fs_in("env_sh", VEC3, k: 5) * (n.y() * n.z()) + fs_in("env_sh", VEC3, k: 6) * (3.0f * n.z() * n.z() - 1.0f)
        + fs_in("env_sh", VEC3, k: 7) * (n.x() * n.z()) + fs_in("env_sh", VEC3, k: 8) * (n.x() * n.x() - n.y() * n.y());
    return nodes::max(e, nodes::vec3(0));
}

fn Node env_sun_dir() @local => nodes::vec3(fs_in("env_sh", FLOAT, 3), fs_in("env_sh", FLOAT, 3, k: 1), fs_in("env_sh", FLOAT, 3, k: 2));

// The environment's sun on a surface facing n.
fn Node env_sun(Node n) @local {
    Node light = nodes::vec3(fs_in("env_sh", FLOAT, 3, k: 3), fs_in("env_sh", FLOAT, 3, k: 4), fs_in("env_sh", FLOAT, 3, k: 5));
    return light * nodes::max(nodes::dot(n, env_sun_dir()), 0);
}

// Where a rough surface's reflection mostly comes from: the mirror direction r, leaning towards
// the normal as the surface gets rougher (three's getIBLRadiance).
fn Node dominant_reflection(Node r, Node n, Node rough) @local {
    Node r2 = rough * rough;
    return nodes::normalize(nodes::mix(r, n, r2 * r2));
}

// How widely a rough surface's reflection is blurred (radians), as three's PMREM blurs it.
fn Node env_blur(Node rough) @local {
    Node low = 1.37f * rough * rough;
    Node b = nodes::select(rough.greater_than_equal(nodes::num(0.21f)), nodes::mix(0.06f, 0.125f, (rough - 0.21f) / 0.095f), low);
    b = nodes::select(rough.greater_than_equal(nodes::num(0.305f)), nodes::mix(0.125f, 0.215f, (rough - 0.305f) / 0.095f), b);
    b = nodes::select(rough.greater_than_equal(nodes::num(0.4f)), nodes::mix(0.215f, 0.526f, (rough - 0.4f) / 0.4f), b);
    return nodes::select(rough.greater_than_equal(nodes::num(0.8f)), nodes::mix(0.526f, 0.582f, (rough - 0.8f) / 0.2f), b);
}

// What the surface mirrors in direction d, blurred for `rough` (shader_core.glsl's
// env_reflection): the material's own map, box projected (the ray met with the room's walls
// first, or with what a map with distances saw in the room); or the scene's reflection probes
// (key.probes), mixed by the boxes `world` is in. `facing` is the surface's normal, `frag` the
// pixel's position in the target; `probe_fn` the probes' function (-1: not made yet).
fn Node env_sample(MaterialKey key, int* probe_fn, Node d, Node rough, Node world, Node frag, Node facing) @local {
    if (key.probes) return probe_reflection(probe_fn, d, rough, world, frag, facing);
    if (key.env_box) {
        EnvSource own = { .pos = fs_in("env_pos", VEC3), .lo = fs_in("env_box_min", VEC3), .hi = fs_in("env_box_max", VEC3),
                          .range = fs_in("env_params", FLOAT, 3) };
        d = own.project(d, rough, world, frag);
    }
    return env_read({ }, d, rough);
}

// A map a reflection is read from: where it was captured, its box, the range of its
// distances (0: none) and, for a probe, its layer of the probes' texture (none: the
// material's own map).
struct EnvSource @local {
    Node layer;
    Node pos;
    Node lo;
    Node hi;
    Node range;
}

// The direction from where the map was captured to where the ray from `world` in direction
// d meets the room (box projection).
fn Node EnvSource.project(&self, Node d, Node rough, Node world, Node frag) {
    Node inv = 1.0f / d;
    Node t_far = nodes::max((self.hi - world) * inv, (self.lo - world) * inv);
    Node t = nodes::max(nodes::min(nodes::min(t_far.x(), t_far.y()), t_far.z()), 0);
    Node o = world - self.pos;
    return nodes::normalize(o + d * self.march(o, d, t, rough, frag));
}

// The map in direction d at the level `rough` blurs it to.
fn Node env_read(Node layer, Node d, Node rough) @local {
    Node size = layer.id ? node_sample(SIZE, shaders::VIEW_TEX_PROBES, shaders::SMP_ENV_SMP, TEX_ARRAY, nodes::int_(0)) : shading::env_size();
    Node width = nodes::convert(size.x(), FLOAT);
    return env_texel(layer, shading::env_uv(d), shading::env_lod(env_blur(rough), width)).xyz() * fs_in("env_params", FLOAT, 1);
}

fn Node env_texel(Node layer, Node uv, Node lod) @local {
    if (!layer.id) return shading::env_level(uv, lod);
    return node_sample(LEVEL, shaders::VIEW_TEX_PROBES, shaders::SMP_ENV_SMP, TEX_ARRAY, nodes::vec3(uv, layer), lod);
}

// The scene's reflection probes in direction d: the probe whose box `world` is in (or just
// outside of, facing in: a window pane in the room's wall), cross-faded with its neighbour
// over the blend width outside the box, and with the probe for everywhere else (its box
// empty) beyond that.
fn Node probe_reflection(int* probe_fn, Node d, Node rough, Node world, Node frag, Node facing) @local {
    Node first = nodes::variable(nodes::int_(-1));
    Node second = nodes::variable(nodes::int_(-1));
    Node outer = nodes::variable(nodes::int_(-1));
    Node w1 = nodes::variable(nodes::num(0));
    Node w2 = nodes::variable(nodes::num(0));
    Node i = nodes::loop_index(INT);
    Node lo = fs_at("probe_min", VEC4, i);
    Node hi = fs_at("probe_max", VEC4, i);
    Node empty = nodes::dot(nodes::abs(hi.xyz() - lo.xyz()), nodes::vec3(1)).equal(nodes::num(0));
    Node below = world - lo.xyz();
    Node above = hi.xyz() - world;
    Node inside = nodes::min(nodes::min(below.x(), nodes::min(below.y(), below.z())), nodes::min(above.x(), nodes::min(above.y(), above.z())));
    Node w = nodes::clamp(inside / nodes::max(lo.w(), 1e-3f) + 1.0f, 0, 1);
    // the outer side of the room's own wall shows what is outside
    w = nodes::select(nodes::dot(facing, fs_at("probe_pos", VEC3, i) - world).less_than_equal(nodes::num(0)), 0.0f, w);
    Node rank = nodes::if_then(w.greater_than(w1),
                               nodes::block(second.assign(first), w2.assign(w1), first.assign(i), w1.assign(w)),
                               nodes::block(nodes::if_then(w.greater_than(w2), nodes::block(second.assign(i), w2.assign(w)))));
    Node body = nodes::block(nodes::if_then(empty, nodes::block(outer.assign(i), nodes::continue_()), nodes::block(rank)));
    Node pick = nodes::loop(i, nodes::int_(0), nodes::convert(fs_in("probe_info", FLOAT), INT), body);
    Node sum = w1 + w2;
    Node has_outer = outer.greater_than_equal(nodes::int_(0));
    // with a probe for everywhere else the boxes' shares stay as they are (it takes the rest);
    // without one they are made to add up to all of it
    Node rest = nodes::select(has_outer, nodes::max(sum.one_minus(), 0), 0.0f);
    Node norm = 1.0f / nodes::max(nodes::select(has_outer, nodes::max(sum, 1.0f), sum), 1e-4f);
    Node result = nodes::variable(nodes::vec3(0));
    Node[6] st = { first, second, outer, w1, w2, pick };
    Node[3] which = { first, second, outer };
    Node[3] share = { w1 * norm, w2 * norm, rest };
    Node[3] add;
    for (int k = 0; k < 3; k++) {
        Node seen = probe_sample(probe_fn, d, rough, world, frag, which[k], nodes::num(k == 2 ? 0.0f : 1.0f));
        Node on = nodes::logical_and(which[k].greater_than_equal(nodes::int_(0)), share[k].greater_than(nodes::num(0)));
        add[k] = nodes::if_then(on, nodes::block(result.assign(result + seen * (share[k] * fs_at("probe_max", FLOAT, which[k], 3)))));
    }
    return nodes::seq(nodes::block(st[0], st[1], st[2], st[3], st[4], st[5], result, add[0], add[1], add[2]), result);
}

// Probe `index`'s map in direction d, box projected where `boxed` is 1 (the function
// probe_sample of the program, made once: `fn_id` -1 makes it).
fn Node probe_sample(int* fn_id, Node d, Node rough, Node world, Node frag, Node index, Node boxed) @local {
    if (*fn_id < 0) {
        NodeType[6] types = { VEC3, FLOAT, VEC3, VEC2, INT, FLOAT };
        int f = node_func("probe_sample", VEC3, &types);
        Node pd = node_param(f, 0);
        Node pr = node_param(f, 1);
        Node pw = node_param(f, 2);
        Node pf = node_param(f, 3);
        Node pi = node_param(f, 4);
        Node pb = node_param(f, 5);
        Node pos = fs_at("probe_pos", VEC4, pi);
        EnvSource probe = { .layer = nodes::convert(pi, FLOAT), .pos = pos.xyz(), .lo = fs_at("probe_min", VEC3, pi),
                            .hi = fs_at("probe_max", VEC3, pi), .range = pos.w() };
        Node dir = nodes::variable(pd);
        Node box = nodes::if_then(pb.greater_than(nodes::num(0.5f)), nodes::block(dir.assign(probe.project(pd, pr, pw, pf))));
        node_func_body(f, nodes::seq(nodes::block(dir, box), env_read(probe.layer, dir, pr)));
        *fn_id = f;
    }
    Node[6] args = { d, rough, world, frag, index, boxed };
    return node_call_fn(*fn_id, &args);
}

// How far the ray from `o` (relative to where the map was captured) in direction d goes, `t`
// to the room's walls, before it goes behind something the map saw: a map with distances walks
// it through the room in 72 steps and stops there (not far behind: then it only passes at the
// back of it, out of the map's sight). Only for sharp reflections (rough < 0.3); t otherwise.
fn Node EnvSource.march(&self, Node o, Node d, Node t, Node rough, Node frag) {
    const int STEPS = 72;
    Node range = self.range;
    Node dt = t / (float)STEPS;
    // the distances come in steps of 1/255 of their range: nearer than that to a surface, a
    // ray is not yet behind it (or a wall seen at a glancing angle breaks up into rings, one
    // per step)
    Node slack = range * (1.5f / 255.0f);
    Node noise = shading_noise(frag + nodes::vec2(47.0f, 17.0f) * fs_in("ssr_params", FLOAT, 2));
    Node far = nodes::variable(t);
    Node prev = nodes::variable(nodes::num(0));
    Node at = nodes::variable(dt * (noise * 0.7f + 0.3f));
    Node lo = nodes::variable(prev);
    Node hi = nodes::variable(at);
    Node mid = (lo + hi) * 0.5f;
    Node k = nodes::loop_index(UINT);
    Node halve = nodes::loop(k, nodes::uint_(0), nodes::uint_(4),
                             nodes::block(nodes::if_then(self.behind(o + d * mid).greater_than(slack),
                                                         nodes::block(hi.assign(mid)), nodes::block(lo.assign(mid)))));
    Node behind = self.behind(o + d * at);
    Node hit = nodes::logical_and(behind.greater_than(slack), behind.less_than(dt + 0.12f + slack));
    Node i = nodes::loop_index(UINT);
    Node step = nodes::loop(i, nodes::uint_(0), nodes::uint_(STEPS),
                            nodes::block(nodes::if_then(hit, nodes::block(lo, hi, halve, far.assign(hi), nodes::break_())),
                                         prev.assign(at), at.assign(at + dt)));
    Node on = nodes::logical_and(range.greater_than(nodes::num(0)), rough.less_than(nodes::num(0.3f)));
    return nodes::seq(nodes::block(far, nodes::if_then(on, nodes::block(prev, at, step))), far);
}

// How far the point `q` (relative to where the map was captured) lies behind what the map saw
// in its direction (its alpha: the distance over the range).
fn Node EnvSource.behind(&self, Node q) {
    Node len = nodes::length(q);
    return len - env_texel(self.layer, shading::env_uv(q / len), nodes::num(0)).w() * self.range;
}

// Interleaved gradient noise (Jimenez's), as shader_core.glsl's ssr_noise.
fn Node shading_noise(Node p) @local
    => nodes::fract(nodes::fract(nodes::dot(p, nodes::vec2(0.06711056f, 0.00583715f))) * 52.9829189f);

// The frame of a normal map at `n`: the vertex tangents', or the cotangent frame of the map's
// uv, `duv1` and `duv2` its change from one pixel to the next (tangent_frame, three's
// getTangentFrame: t and b scaled together).
fn Node[2] tangent_frame(MaterialKey key, Node n, Node v_normal, Node world, Node tangent, Node duv1, Node duv2) @local {
    // a back face (its normal turned to the camera): the frame turned over too
    Node back = nodes::dot(n, v_normal).less_than(nodes::num(0));
    Node dpdx = nodes::dfdx(world);
    Node dpdy = nodes::dfdy(world);
    Node dp2perp = nodes::cross(dpdy, n);
    Node dp1perp = nodes::cross(n, dpdx);
    Node t = dp2perp * duv1.x() + dp1perp * duv2.x();
    Node b = dp2perp * duv1.y() + dp1perp * duv2.y();
    // (three's formula takes the screen's y up, as GL's dFdy; Metal's and D3D's go down, which
    // turns the frame around: its sign, the determinant's, keeps it the same on every backend)
    Node invmax = nodes::inverse_sqrt(nodes::max(nodes::max(nodes::dot(t, t), nodes::dot(b, b)), 1e-12f));
    invmax = nodes::select(nodes::dot(dpdx, dp2perp).less_than(nodes::num(0)), invmax.negate(), invmax);
    invmax = nodes::select(back, invmax.negate(), invmax);
    t = t * invmax;
    b = b * invmax;
    if (key.tangents) {
        // (a vertex without one, all zeros: the frame from the uv instead)
        Node vt = nodes::varying(tangent);
        Node vn = nodes::normalize(v_normal);
        Node tt = nodes::normalize(vt.xyz());
        Node tb = nodes::normalize(nodes::cross(vn, tt) * vt.w());
        tt = nodes::select(back, tt.negate(), tt);
        tb = nodes::select(back, tb.negate(), tb);
        Node has = nodes::dot(vt.xyz(), vt.xyz()).greater_than(nodes::num(0));
        t = nodes::select(has, tt, t);
        b = nodes::select(has, tb, b);
    }
    return { t, b };
}

// MeshToonMaterial's light: the gradient map (or two steps without one) by where the light
// falls, in place of n.l (three's getGradientIrradiance).
fn Node toon_gradient(MaterialKey key, Node ndl) @local {
    Node x = ndl * 0.5f + 0.5f;
    int slot = key.slot(MAP_GRADIENT) - 1;
    if (slot >= 0) {
        int view = shaders::VIEW_TEX_EXT0 + slot;
        Node uv = nodes::vec2(nodes::clamp(x, 0, 0.999f), 0);
        if (key.toon_nearest) {
            Node size = nodes::convert(node_sample(SIZE, view, shaders::SMP_TEX_SMP, TEX_2D, { }), VEC2);
            Node at = nodes::convert(nodes::fract(uv) * size, IVEC2);
            return node_sample(FETCH, view, shaders::SMP_TEX_SMP, TEX_2D, at, nodes::int_(0)).x();
        }
        return node_sample(LEVEL, view, shaders::SMP_TEX_SMP, TEX_2D, uv, nodes::num(0)).x();
    }
    Node fw = nodes::fwidth(x) * 0.5f;
    return nodes::mix(0.7f, 1.0f, nodes::smoothstep(nodes::num(0.7f) - fw, fw + 0.7f, x));
}

// three's window: the light fades out to nothing at `cutoff` (0: never), and the falloff by
// `decay` (shader_core.glsl's distance_window, distance_attenuation).
fn Node distance_window(Node d, Node cutoff) @local {
    Node r = d / cutoff;
    Node f = nodes::clamp((r * r * r * r).one_minus(), 0, 1);
    return nodes::select(cutoff.less_than_equal(nodes::num(0)), 1.0f, f * f);
}

fn Node distance_attenuation(Node d, Node cutoff, Node decay) @local {
    return distance_window(d, cutoff) / nodes::max(nodes::pow(d, decay), 0.01f);
}

// A light with a size, seen in reflections (light_toward): where it is (`on`), the direction
// toward the point of it its highlight comes from, and how much its shape widens one (x along a
// tube, y round a ball); and a tube's cosine over the squared distance (-1: not a tube).
struct Sized @local {
    Node on;
    Node l;
    Node k;
    Node nol;
}

// Karis, "Real Shading in Unreal Engine 4", 2013, the representative point: the point of the
// light nearest the ray `r` reflected off the surface. `lv` is from the surface to the light's
// centre, `seg` from its centre to one end (a tube), `radius` its thickness. A tube first takes
// the point of its segment nearest the ray (his eq. 19), then the ball of `radius` round that
// point the point of it nearest the ray (eq. 11).
fn Sized light_toward(Node r, Node lv, Node seg, Node radius, bool tube) @local {
    Node kx = nodes::num(0);
    if (tube) {
        Node l0 = lv - seg;
        Node ld = seg * 2.0f;
        Node rld = nodes::dot(r, ld);
        Node t = (nodes::dot(r, l0) * rld - nodes::dot(l0, ld)) / nodes::max(nodes::dot(ld, ld) - rld * rld, 1e-8f);
        Node has = nodes::dot(seg, seg).greater_than(nodes::num(0));
        lv = nodes::select(has, l0 + ld * nodes::clamp(t, 0, 1), lv);
        kx = nodes::select(has, nodes::length(seg) / (nodes::max(nodes::length(lv), 1e-4f) * 3.0f), 0.0f);
    }
    Node ball = radius.greater_than(nodes::num(0));
    Node ky = nodes::select(ball, radius / (nodes::max(nodes::length(lv), 1e-4f) * 3.0f), 0.0f);
    // (the light's centre to the ray: Karis' eq. 11 with its sign put right)
    Node to_ray = r * nodes::dot(lv, r) - lv;
    lv = nodes::select(ball, lv + to_ray * nodes::clamp(radius / nodes::max(nodes::length(to_ray), 1e-6f), 0, 1), lv);
    return { .l = nodes::normalize(lv), .k = nodes::vec2(kx, ky) };
}

fn Node d_ggx(Node noh, Node a) @local {
    Node a2 = a * a;
    Node d = noh * noh * (a2 - 1.0f) + 1.0f;
    return shading::RECIPROCAL_PI * a2 / (d * d);
}

fn Node v_smith(Node nov, Node nol, Node a) @local {
    Node a2 = a * a;
    Node gv = nol * nodes::sqrt(a2 + a2.one_minus() * nov * nov);
    Node gl = nov * nodes::sqrt(a2 + a2.one_minus() * nol * nol);
    return 0.5f / nodes::max(gv + gl, 1e-6f);
}

// The light cluster `p` is in (shader_core.glsl's cluster_of).
fn Node cluster_of(Node p) @local {
    Node grid = nodes::convert(fs_in("cluster_grid", VEC3), IVEC3);
    Node clip = fs_in("cluster_view_proj", MAT4) * nodes::vec4(p, 1);
    Node ndc = clip.xy() / nodes::max(nodes::abs(clip.w()), 1e-6f);
    Node gxy = grid.xy();
    Node cell = nodes::convert(nodes::floor((ndc * 0.5f + 0.5f) * nodes::convert(gxy, VEC2)), IVEC2);
    Node tile = nodes::clamp(cell, nodes::convert(nodes::int_(0), IVEC2), gxy - nodes::int_(1));
    Node depth = nodes::dot(p - fs_in("camera_pos", VEC3), fs_in("dir_cascade_fwd", VEC3));
    Node cd = fs_in("cluster_depth");
    Node slice = nodes::select(depth.less_than_equal(cd.x()), nodes::int_(0),
                               nodes::convert(nodes::floor(nodes::log(depth / cd.x()) * cd.y()), INT));
    slice = nodes::clamp(slice, nodes::int_(0), grid.z() - nodes::int_(1));
    return tile.x() + grid.x() * (tile.y() + grid.y() * slice);
}
