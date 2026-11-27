// Material specialization keys, generated program caching, and node-material builtin bindings.
//
// MaterialKey packs every feature flag that alters shader code into 32 bytes (two uint128 words):
// bits 0..91 hold boolean and enum feature fields for forward, shadow, G-Buffer, and deferred
// lighting programs, and bits 192..254 hold 3-bit shared texture slot indices (`slot + 1`, 0 =
// absent) for 21 MAP_* roles (MAP_CUSTOM uses a dedicated node texture view instead of a shared
// slot). Uniform row macros (`fs_row`, `vs_row`, `deferred_row`, `fs_in`, `vs_in`, `deferred_in`)
// index 16-byte Vec4 rows of FsParams, VsParams, and DeferredLightParams.
// VIEW_MORPH_DELTAS (32) and VIEW_MORPH_WEIGHTS (33) occupy logical view IDs between engine slots
// (< 32) and NODE_USER_VIEW (64). Delegates graph construction to src/material/material_graph.c3
// and src/material/material_deferred.c3, and compiles ShaderPrograms via src/nodes/program.c3.
module drie;
import sokol, drie::nodes, drie::shading, std::collections::map, std::io, std::os::env, std::thread;

// ---- Uniform row inputs

// 16-byte Vec4 row offset of field `$name` inside FsParams.
macro int fs_row($name) {
    FsParams* p = null;
    return (int)((usz)&p.$eval($name) / 16);
}

// 16-byte Vec4 row offset of field `$name` inside VsParams.
macro int vs_row($name) {
    VsParams* p = null;
    return (int)((usz)&p.$eval($name) / 16);
}

// 16-byte Vec4 row offset of field `$name` inside DeferredLightParams.
macro int deferred_row($name) {
    DeferredLightParams* p = null;
    return (int)((usz)&p.$eval($name) / 16);
}

// Reads FsParams field `$name` as type `t`, starting at component `comp` and array row offset `k`.
macro Node fs_in($name, NodeType t = VEC4, int comp = 0, int k = 0) => node_input(FS, fs_row($name) + k, t, comp);
// Reads VsParams field `$name` as type `t`, starting at component `comp` and array row offset `k`.
macro Node vs_in($name, NodeType t = VEC4, int comp = 0, int k = 0) => node_input(VS, vs_row($name) + k, t, comp);
// Reads DeferredLightParams field `$name` as type `t`, starting at component `comp` and row offset `k`.
macro Node deferred_in(
    $name,
    NodeType t = VEC4,
    int comp = 0,
    int k = 0
) => node_input(DEFERRED, deferred_row($name) + k, t, comp);

// ---- Material key and program cache

// Feature bits and shared map slots (`tex_ext0..5`, stored as `slot + 1`, 0 = absent) for each
// `MAP_*` role.
bitstruct MaterialKey : char[32] {
    char type : 0..3; // MaterialType's ordinal
    bool mrt : 4; // a target with MRT: normal, motion and indirect light too
    bool cut : 5; // DrawKind.CUT: the program may cut holes
    bool cull_box : 6; // Renderer.cull_box
    bool dash : 7; // a dashed line
    bool dither : 8; // an LOD's cross-fade
    bool screen : 9; // Material.screen_target
    bool tint : 10; // the vertex colors tint the color (not baked light)
    bool vertex_alpha : 11; // the vertex colors' alpha
    bool foliage : 12;
    bool map : 13;
    bool alpha_test : 14;
    bool background : 15; // the scene's background
    bool env_combine : 16; // three's envMap on a basic, lambert or phong material
    bool vertex_occlusion : 17;
    bool premultiplied : 18;
    char fog : 19..20; // the fog's mode (0: none)
    bool fog_box : 21;
    bool fog_height : 22;
    bool fog_sun : 23;
    bool display_fog : 24; // Renderer.display_fog
    bool linear : 25; // linear light out (a half-float target)
    char tone : 26..28; // ToneMapping's ordinal (0 when linear or not tone mapped)
    char display : 29..30; // the window's colors (DISPLAY_*)
    bool gpu_cull : 31; // instances culled on the GPU
    bool displacement : 32;
    bool sway : 33;
    bool outline : 34;
    bool motion : 35; // motion vectors (with mrt)
    bool dir_lights : 36; // the render's light kinds (lit materials)
    bool point_lights : 37;
    bool spot_lights : 38;
    bool channels : 39; // Material.light_channels: lights of some channels only
    bool map_emissive : 40; // Material.map_emissive
    bool baked : 41; // the vertex colors are baked light (Material.vertex_light)
    bool normal_map : 42;
    bool tangents : 43; // the tangent frame from the vertex tangents
    bool env_light : 44; // a standard material's environment: its light from all around;
    bool env_reflect : 45; // the reflection of its environment (its map or the probes),
    bool env_box : 46; // box projected
    bool toon_nearest : 47; // the toon material's gradient map read nearest
    bool shadows : 48; // it receives the render's shadows:
    bool vsm : 49; // VSM's,
    bool soft_shadows : 50; // from sources with a size (PCSS),
    bool sun_shadow : 51; // the sun's (the directional light's with cascades),
    bool dir_tiles : 52; // other directional lights' (a tile of the atlas),
    bool point_shadows : 53; // point lights',
    bool spot_shadows : 54; // spot lights'
    bool debug_cascades : 55; // renderer.debug_cascades: the sun's cascades tinted
    bool arm_ao : 56; // tex_arm holds the ambient occlusion (red),
    bool arm_rough : 57; // the roughness (green),
    bool arm_metal : 58; // the metalness (blue)
    bool sized : 59; // point or spot lights with a size, seen in a standard material's highlights
    bool tubes : 60; // point lights with a length (tubes)
    bool rect_lights : 61; // RectAreaLights (on a standard material)
    bool parallax : 62; // Material.parallax_map (a standard material's)
    bool clearcoat : 63; // the physical material's layers: a clear coat,
    bool sheen : 64; // a sheen,
    bool iridescence : 65; // a thin film's rainbow,
    bool anisotropy : 66; // highlights drawn out along a direction,
    bool transmission : 67; // the picture behind it refracted through its volume
    bool probes : 68; // it mirrors the scene's reflection probes (Material.env_probes)
    bool gi : 69; // the scene's irradiance volumes (lit materials),
    bool rc : 70; // its radiance cascades
    bool ssr : 71; // screen-space reflections (Material.ssr)
    bool reflector : 72; // a planar Reflector's picture (Material.reflector)
    bool glass : 73; // Material.glass: its reflections stay however clear it is
    bool hex : 74; // Material.hex_tiling
    bool defocus : 75; // Material.defocus: the maps blurred by the lens's
    bool fog_volume : 76; // the fog's froxel volume (in place of its analytic fog),
    bool fog_glow : 77; // its lights' glow,
    bool fog_shafts : 78; // its light shafts (the sun's shadow in it)
    char depth_packing : 79..80; // a depth material's Material.depth_packing
    // DrawKind.PRETESTED: drawn where the depth prepass has drawn it, which cuts its holes
    // (the alpha test's, the LOD cross-fade's: those bits say which); its program comes with
    // the prepass's (MaterialProgram.depth)
    bool prepassed : 81;
    // the shadow maps' program (build_shadow_graph), which reads only gpu_cull, displacement,
    // sway, and for the alpha test's cut-out alpha_test, map and MAP_ALPHA's slot (1: one);
    // a node material's nodes and foliage too
    bool shadow : 82;
    bool bake : 83; // light map sample (src/lighting/lightmap.c3): drawn at uv1, blended in
    // a node material's (its *_node graphs in the program): found through its NodeBuild
    // (NodeBuild.programs), which the key alone doesn't name
    bool nodes : 84;
    bool morph : 85; // GPU morph targets (BufferGeometry.morph_view + Mesh.morph_weights_view)
    bool flat_shading : 86; // flat face normals via screen-space derivatives (Material.flat_shading)
    // Deferred G-buffer program (gbuffer_key): writes surface data into the four targets read by
    // the deferred lighting pass (build_deferred_graph), without direct lights.
    bool gbuffer : 87;
    // Shadow atlas map (MaterialKey.shadow): rendered into ATLAS_FORMAT as linear distance from
    // the light's plane over its reach (see build_shadow_graph).
    bool atlas_depth : 88;
    // Deferred fullscreen lighting pass (build_deferred_graph): unpacks the G-Buffer and evaluates
    // scene lighting over a fullscreen triangle using the same shading graph as forward materials.
    bool deferred_light : 89;
    bool contact_shadows : 90; // Screen-space contact shadows in the deferred lighting pass.
    bool rt_shadows : 91; // Hardware ray-traced rect area light shadow mask (VIEW_TEX_EXT5).
    bool cloud_shadow : 92; // the clouds' shadow on the first directional light (Renderer.cloud_shadow)
}

// Logical view ID for the GPU morph delta buffer (between engine slots < 32 and NODE_USER_VIEW).
const int VIEW_MORPH_DELTAS  = 32;
// Logical view ID for the GPU morph weight buffer (between engine slots < 32 and NODE_USER_VIEW).
const int VIEW_MORPH_WEIGHTS = 33;

// Shared map role for ambient occlusion.
const int MAP_AO                    = 0;
// Shared map role for KHR_materials_specular color tint.
const int MAP_SPECULAR_COLOR        = 1;
// Shared map role for KHR_materials_specular intensity.
const int MAP_SPECULAR_INTENSITY    = 2;
// Shared map role for KHR_materials_transmission factor.
const int MAP_TRANSMISSION          = 3;
// Shared map role for transmission volume thickness.
const int MAP_THICKNESS             = 4;
// Shared map role for KHR_materials_iridescence strength.
const int MAP_IRIDESCENCE           = 5;
// Shared map role for iridescence thin-film thickness.
const int MAP_IRIDESCENCE_THICKNESS = 6;
// Shared map role for clearcoat strength.
const int MAP_CLEARCOAT             = 7;
// Shared map role for clearcoat roughness.
const int MAP_CLEARCOAT_ROUGHNESS   = 8;
// Shared map role for clearcoat tangent-space normals.
const int MAP_CLEARCOAT_NORMAL      = 9;
// Shared map role for sheen color tint.
const int MAP_SHEEN_COLOR           = 10;
// Shared map role for sheen roughness.
const int MAP_SHEEN_ROUGHNESS       = 11;
// Shared map role for alpha opacity (green channel).
const int MAP_ALPHA                 = 12;
// Shared map role for emissive color.
const int MAP_EMISSIVE              = 13;
// Shared map role for baked lightmap irradiance.
const int MAP_LIGHT                 = 14;
// Shared map role for anisotropy direction and strength.
const int MAP_ANISOTROPY            = 15;
// Reserved map role for Material.custom_map (read via a dedicated node view, not a shared slot).
const int MAP_CUSTOM                = 16;
// Shared map role for lens focus depth (Material.defocus).
const int MAP_FOCUS                 = 17;
// Shared map role for Phong specular and classic env_combine reflectivity.
const int MAP_SPECULAR              = 18;
// Shared map role for MeshMatcapMaterial sphere lookup.
const int MAP_MATCAP                = 19;
// Shared map role for MeshToonMaterial lighting gradient.
const int MAP_GRADIENT              = 20;
// Shared map role for height-based bump perturbation.
const int MAP_BUMP                  = 21;
// Total number of shared map roles.
const int MAP_ROLES                 = 22;

// The key as two words: bits 0..127, and 128..255.
struct KeyWords @private {
    uint128 lo;
    uint128 hi;
}

// Hash of a 256-bit MaterialKey word pair for program cache lookup.
fn uint KeyWords.hash(self) => (uint)(self.lo.hash() ^ self.hi.hash() * 31);
// Equality comparison for 256-bit MaterialKey word pairs.
fn bool KeyWords.equals(self, KeyWords other) @operator(==) => self.lo == other.lo && self.hi == other.hi;

// Where role `role`'s slot is in the key's high word: three bits each from its bit 64 (the
// key's bit 192; MAP_CUSTOM, 16, has none: a graph reads custom_map as a texture of its own).
fn int map_role_bit(int role) @local => 64 + 3 * (role < 16 ? role : role - 1);

// Returns the shared map slot (`1..6`) assigned to `role`, or `0` if unassigned.
fn int MaterialKey.slot(self, int role) {
    if (role == MAP_CUSTOM) return 0;
    return (int)((bitcast(self, KeyWords).hi >> map_role_bit(role)) & 7);
}

// Stores shared map slot (`1..6`, or `0` for none) for `role` in the key's high word.
fn void MaterialKey.set_slot(&self, int role, int slot) {
    if (role == MAP_CUSTOM) return;
    int at     = map_role_bit(role);
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
    foreach (v : p.views) out.appendf(
        "v%d %d %d %d %d;",
        v.stage.ordinal,
        v.source,
        v.storage,
        v.kind.ordinal,
        v.elem.ordinal
    );
    foreach (sm : p.samplers) out.appendf("s%d %d %d %d;", sm.stage.ordinal, sm.source, sm.compare, sm.nonfiltering);
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

// The program for `key`, built the first time it is asked for; with MaterialKey.nodes, the one
// node material `build`'s graphs print, found through the build (NodeBuild.programs) and shared
// with the builds that print the same.
fn MaterialProgram* material_program(MaterialKey key, NodeBuild* build = null) @private {
    // (a node material's: printed for each build, made once for the builds that print the same)
    KeyWords words                             = bitcast(key, KeyWords);
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
    if (key.deferred_light) {
        build_deferred_graph(key, &graph);
    } else if (key.shadow) {
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
        Node off = nodes::select(node_system(FRAG_COORD).x()
            .greater_than(nodes::num(far)), nodes::num(1e-20f), nodes::num(0));
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
    // (DRIE_DUMP_WGSL=dir: each program also printed in WGSL there, for checking it on a machine
    // without WebGPU)
    if (try dir = env::tget_var("DRIE_DUMP_WGSL")) dump_wgsl(dir, &graph, &prepass, key.prepassed, build, vbind, fbind);
    node_release(mark);
    mp.pipelines.init(mem);
    if (build) {
        mp = share_material_program(mp);
        mp.users++;
    }
    cache.set(words, mp);
    // (DRIE_TRACE_PROGRAMS: each program printed as it is built)
    if (try env::tget_var("DRIE_TRACE_PROGRAMS")) {
        io::eprintfn(
            "drie: program %x%032x: vs %d B, fs %d B\n%s\n%s",
            words.hi,
            words.lo,
            mp.program.vs.len,
            mp.program.fs.len,
            mp.program.vs,
            mp.program.fs
        );
        if (mp.depth) io::eprintfn("drie: its depth prepass: fs %d B\n%s", mp.depth.fs.len, mp.depth.fs);
    }
    return mp;
}

int wgsl_dumped @local;

fn void dump_wgsl(
    String dir,
    ProgramGraph* graph,
    ProgramGraph* prepass,
    bool prepassed,
    NodeBuild* build,
    Node[NodeBuiltin::len]* vbind,
    Node[NodeBuiltin::len]* fbind
) @local {
    ProgramGraph g = *graph;
    g.varyings_from = null;
    ShaderProgram*[2] ps;
    ps[0] = build_program(WGSL, &g, build, vbind, fbind);
    if (prepassed) {
        ProgramGraph d = *prepass;
        d.fragment_only = false;
        d.varyings_from = null;
        d.position = graph.position;
        ps[1] = build_program(WGSL, &d, build, vbind, fbind);
    }
    foreach (sp : ps) {
        if (!sp) continue;
        int k = wgsl_dumped++;
        (void)file::save(string::tformat("%s/p%03d.vert.wgsl", dir, k), sp.vs);
        (void)file::save(string::tformat("%s/p%03d.frag.wgsl", dir, k), sp.fs);
        sp.free();
        mem::free(sp);
    }
}

// A node material's program `mp` (MaterialKey.nodes), or the one made already that prints the
// same (`mp` then let go of): shared by the builds whose graphs print it, and by a loaded
// pipeline cache's list (see read_material_program).
fn MaterialProgram* share_material_program(MaterialProgram* mp) @private {
    @pool() {
        // (not by its key: keys that differ in what the program doesn't print share it)
        String sig = material_signature(mp);
        ulong h    = hash64(sig);
        if (!node_material_programs.is_initialized()) node_material_programs.init(mem);
        MaterialProgram*? shared = node_material_programs[h];
        if (try shared && material_signature(shared) == sig) {
            free_material_program(mp);
            return shared;
        }
        // (one whose hash another has, which is no likelier than a collision of 64-bit hashes:
        // kept out of the table, shared with none)
        if (catch shared) {
            mp.shared_as = h;
            node_material_programs[h] = mp;
        }
    };
    return mp;
}

// The bindings of the classic slots (shaders::VIEW_*, SMP_*) as `p` takes them; a node
// material's textures (NODE_USER_VIEW + k) from `user` (its NodeBuild.textures' views).
fn SgBindings program_bindings(
    ShaderProgram* p,
    SgBindings* classic,
    SgView[] user = { },
    SgView morph_deltas = { },
    SgView morph_weights = { }
) @private {
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
fn void apply_program_uniforms(
    ShaderProgram* p,
    ProgramStage st,
    VsParams* vs,
    FsParams* fs,
    ShadowFsParams* sfs = null,
    Vec4[] user = { },
    DeferredLightParams* dp = null
) @private {
    int rows = p.block_rows[st.ordinal];
    if (!rows) return;
    void*[NodeSource::len] src;
    sz[NodeSource::len] sizes;
    src[NodeSource.VS.ordinal] = vs;
    sizes[NodeSource.VS.ordinal] = vs ? VsParams::size : 0;
    src[NodeSource.FS.ordinal] = fs;
    sizes[NodeSource.FS.ordinal] = fs ? FsParams::size : 0;
    src[NodeSource.SHADOW_FS.ordinal] = sfs;
    sizes[NodeSource.SHADOW_FS.ordinal] = sfs ? ShadowFsParams::size : 0;
    src[NodeSource.USER.ordinal] = user.len ? user.ptr : null;
    sizes[NodeSource.USER.ordinal] = user.len * Vec4::size;
    src[NodeSource.DEFERRED.ordinal] = dp;
    sizes[NodeSource.DEFERRED.ordinal] = dp ? DeferredLightParams::size : 0;
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

fn void free_material_program(MaterialProgram* mp) @private {
    if (try in = node_material_programs[mp.shared_as] && in == mp) node_material_programs.remove(mp.shared_as);
    mp.pipelines.@each(; ulong key, SgPipeline pip) {
        sg::destroy_pipeline(pip);
    };
    mp.pipelines.free();
    free_program(mp.program);
    if (mp.depth) free_program(mp.depth);
    mem::free(mp);
}

fn void free_program(ShaderProgram* p) @private {
    // (its compile in the background run to its end first)
    if (p.job) {
        sg_warm_wait(p.job - 1);
        sg_warm_end(p.job - 1);
    }
    if (p.shader.id) sg::destroy_shader(p.shader);
    p.free();
    mem::free(p);
}

// ---- Node material builtin bindings

// What a node material's graphs read their builtins from in each stage (build_program's
// vertex_bind and fragment_bind).
struct UserBinds @private {
    Node[NodeBuiltin::len] v;
    Node[NodeBuiltin::len] f;
}

// The builtins the USER block's header holds (NODE_USER_HEADER), for either stage.
fn void user_header(Node[NodeBuiltin::len]* b) @private {
    (*b)[NodeBuiltin.VIEW_MATRIX.ordinal] = node_input(USER, 0, MAT4);
    (*b)[NodeBuiltin.PROJECTION_MATRIX.ordinal] = node_input(USER, 4, MAT4);
    (*b)[NodeBuiltin.MODEL_MATRIX.ordinal] = node_input(USER, 8, MAT4);
    (*b)[NodeBuiltin.CAMERA_POSITION.ordinal] = node_input(USER, 12, VEC3);
    (*b)[NodeBuiltin.TIME.ordinal] = node_input(USER, 12, FLOAT, 3);
}

fn Node user_camera() @private => node_input(USER, 12, VEC3);

// A builtin bound to `value` from here on (a statement).
fn Node rebind(NodeBuiltin b, Node value) @private => drie::node_bind(b, value);

// `value` (a node material's graph), read with the builtins `binds` bound first: wherever it
// is printed, it reads them as they are at this step of the material.
fn Node bound(Node value, Node[] binds) @private => nodes::seq(drie::node_block(binds), value);

// What the fragment stage's builtins read before the material's steps bind them again: `n` the
// vertex normal facing the camera.
fn void fragment_binds(VertexValues* vtx, Node[NodeBuiltin::len]* b, Node v_uv, Node world, Node n) @private {
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

// Applies a node material's mask_node discard, color_node, and opacity_node over `rgb` and `a`,
// with NORMAL_WORLD bound to facing vertex normal `n`. Cut-out statements (mask_node discard,
// plus color_node/opacity_node evaluation when NodeBuild.cuts is set) are appended to `holes`.
fn void user_color(NodeBuild* build, Node n, Node* rgb, Node* a, Node[] holes, int* count) @private {
    Node mask    = build.slots[8];
    Node color   = build.slots[0];
    Node opacity = build.slots[1];
    // (MATERIAL_COLOR: the color and opacity as they come to this step)
    Node own     = nodes::vec4(*rgb, *a);
    if (mask.id) {
        if (mask.type() != BOOL) abort("drie: mask_node is a bool");
        holes[(*count)++] = nodes::if_then(
            nodes::logical_not(bound(mask, { rebind(NORMAL_WORLD, n), rebind(NORMAL_GEOMETRY, n) })),
            nodes::block(nodes::discard())
        );
    }
    if (color.id) {
        Node c = bound(color, { rebind(NORMAL_WORLD, n), rebind(NORMAL_GEOMETRY, n), rebind(MATERIAL_COLOR, own) });
        if (build.cuts) holes[(*count)++] = drie::node_eval(c);
        if (color.type() == VEC4) {
            *rgb = c.xyz();
            *a = c.w();
        } else {
            *rgb = nodes::convert(c, VEC3);
        }
    }
    if (opacity.id) {
        Node o = bound(
            nodes::convert(opacity, FLOAT),
            { rebind(NORMAL_WORLD, n), rebind(NORMAL_GEOMETRY, n), rebind(MATERIAL_COLOR, own) }
        );
        if (build.cuts) holes[(*count)++] = drie::node_eval(o);
        *a = o;
    }
}