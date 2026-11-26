// Whole programs printed from node graphs, as three's WebGPU renderer prints TSL into WGSL: a
// vertex graph (the clip position, with the statements before it) and up to four fragment
// outputs become the program's Metal, HLSL or GLSL source, holding only what the graphs reach,
// and its sokol description (ShaderProgram.desc), made at runtime.
//
// The graphs read the engine through nodes of their own:
//   INPUT      rows of the uniform blocks the renderer fills (NodeSource: shaders::FsParams...);
//              a program's own block holds only the rows it reads, copied in by gather()
//   STORAGE    elements of storage buffers, by the engine's view slots (shaders::VIEW_*)
//   SAMPLE     engine textures by view and sampler slot: sampled, at a level, with gradients,
//              texels fetched, their size, or a depth comparison
//   ATTRIBUTE  vertex attributes by location (shaders::ATTR_*)
//   SYSTEM     the instance and vertex index, the fragment's position and facing
//   VARYING    a vertex value read per fragment (any number, of any type but matrices)
// and through functions (node_func): printed once per stage and called, as TSL's Fn with a
// layout, where inlining the graph (a C3 function making nodes) would print it again at each
// call. A user graph's builtins (nodes::normal_world()...) read what the program binds them to
// (BIND), which changes as the program goes, as three's positionLocal does after positionNode.
//
// Values are printed once where they can be read again: a variable's readers again after it is
// assigned (each variable has its own count of assignments), and inside a loop that assigns
// what they read. Two nodes that print the same expression are one value (the second reads the
// first's name), and what both blocks of an if read is printed before it (hoist_shared), where
// the blocks and what follows them all read it. Names number the nodes in the order they are printed, so the same graphs
// print the same text whatever else was made before, and the driver's cache knows them.
module drie;
import std::collections::list, std::collections::map, std::io, std::thread, sokol;

// The uniform blocks programs read rows of (INPUT), in the order the renderer hands them to
// ShaderProgram.gather.
enum NodeSource : char {
    VS,         // shaders::VsParams
    FS,         // shaders::FsParams
    SHADOW_VS,  // shaders::ShadowVsParams
    SHADOW_FS,  // shaders::ShadowFsParams
    DEPTH_FS,   // shaders::DepthFsParams
    USER,       // a node material's uniforms (nodes::uniform), as its NodeBuild lists them
}

// A node material's textures (nodes::texture) take the view sources from here on, in the order
// its NodeBuild lists them; the engine's are its slots (shaders::VIEW_*).
const int NODE_USER_VIEW = 64;

// The rows of a node material's USER block before its uniforms: 0-3 the view matrix, 4-7 the
// projection, 8-11 the mesh's matrix_world, 12 the camera's position and the time (w).
const int NODE_USER_HEADER = 13;

enum ProgramStage : char {
    VERTEX,
    FRAGMENT,
}

// ---- functions (TSL's Fn with a layout)

struct NodeFunc {
    String name;
    NodeType ret;            // VOID: returns nothing
    NodeType[12] params;
    int count;
    Node body;               // what it returns: a value (a SEQ with its statements); VOID: a block's SEQ
    uint mark;               // node_count() when it was made
    // nodes::function's: calls with the same arguments are one value, and can be hoisted out
    // of an if (hoist_shared); no discard
    bool pure;
}

List{NodeFunc} node_funcs @private;
List{uint} node_args @private;

// A new function `name` returning a `ret` of `params`: its parameters are node_param(f, i), its
// body is set with node_func_body. Functions are printed in a program once per stage that calls
// them; `name` is what they are called there (unique among a program's functions).
fn int node_func(String name, NodeType ret, NodeType[] params, bool pure = false) {
    if (params.len > 12) abort("drie: nodes: a function of more than 12 parameters");
    foreach (t : params) {
        if (t == VOID) abort("drie: nodes: %s has a parameter of no type (VOID)", name);
    }
    if (!node_funcs.is_initialized()) node_funcs.init(mem);
    NodeFunc f = { .name = name.copy(mem), .ret = ret, .count = (int)params.len, .mark = node_count(), .pure = pure };
    foreach (i, t : params) f.params[i] = t;
    node_funcs.push(f);
    return (int)node_funcs.len() - 1;
}

fn Node node_param(int f, int i) {
    NodeFunc* fd = node_funcs.get_ref(f);
    if (i < 0 || i >= fd.count) abort("drie: nodes: %s has no parameter %d", fd.name, i);
    return add_node({ .op = PARAM, .type = fd.params[i], .kind = (char)i, .res = { f, 0, 0 } });
}

fn void node_func_body(int f, Node body) {
    NodeFunc* fd = node_funcs.get_ref(f);
    if (fd.ret != VOID && body.type() == VOID) abort("drie: nodes: %s returns a %s", fd.name, fd.ret);
    fd.body = body;
}

fn Node node_call_fn(int f, Node[] args) {
    NodeFunc* fd = node_funcs.get_ref(f);
    if (args.len != fd.count) abort("drie: nodes: %s takes %d arguments, not %d", fd.name, fd.count, args.len);
    if (!fd.body.id) abort("drie: nodes: %s has no body yet", fd.name);
    if (!node_args.is_initialized()) node_args.init(mem);
    uint at = (uint)node_args.len();
    foreach (a : args) node_args.push(a.id);
    NodeData d = { .op = CALL_FN, .type = fd.ret, .count = (char)args.len };
    d.value[0] = bitcast((uint)f, float);
    d.value[1] = bitcast(at, float);
    return add_node(d);
}

fn void release_funcs(uint mark) {
    if (!node_funcs.is_initialized()) return;
    while (node_funcs.len() && node_funcs[node_funcs.len() - 1].mark >= mark) {
        node_funcs[node_funcs.len() - 1].name.free(mem);
        node_funcs.remove_last();
    }
}

fn uint call_arg(NodeData* d, int i) @private => node_args[(sz)bitcast(d.value[1], uint) + (sz)i];

fn int call_func(NodeData* d) @private => (int)bitcast(d.value[0], uint);

// ---- the engine's nodes

// Rows of uniform block `src` from `row`: a `t` (a vector from component `comp` of the row, a
// matrix from its columns, a row each). With `index`, element `index` of an array of `len` of
// them, `stride` rows apart.
fn Node node_input(NodeSource src, int row, NodeType t, int comp = 0, Node index = { }, int stride = 0, int len = 0) {
    if (t == VOID || t == BOOL) abort("drie: nodes: no input of a %s", t);
    if (!is_matrix(t) && comp + components(t) > 4) abort("drie: nodes: a %s from component %d is past its row", t, comp);
    NodeData d = { .op = INPUT, .type = t, .count = index.id ? 1 : 0, .args = { index.id, 0, 0, 0 }, .res = { src.ordinal, row, comp } };
    d.value[0] = bitcast((uint)stride, float);
    d.value[1] = bitcast((uint)len, float);
    return add_node(d);
}

// Element `index` of the storage buffer in view slot `view`: of vec4s, uvec4s or uints.
fn Node node_storage(int view, NodeType elem, Node index) {
    if (elem != VEC4 && elem != UVEC4 && elem != UINT) abort("drie: nodes: storage buffers of vec4s, uvec4s or uints, not %ss", elem);
    return add_node({ .op = STORAGE, .type = elem, .count = 1, .args = { index.id, 0, 0, 0 }, .res = { view, 0, 0 } });
}

fn Node node_sample(NodeSampling how, int view, int sampler, NodeTextureKind kind, Node coords, Node a = { }, Node b = { }) {
    NodeType t = VEC4;
    switch (how) {
        case SIZE: t = kind == TEX_3D || kind == TEX_ARRAY ? IVEC3 : IVEC2;
        case LEVELS: t = INT;
        case COMPARE:
            if (kind != TEX_DEPTH) abort("drie: nodes: compare() reads depth textures");
            t = FLOAT;
        default:
            if (kind == TEX_DEPTH) abort("drie: nodes: a depth texture is read with compare()");
    }
    return add_node({ .op = SAMPLE, .type = t, .kind = (char)how.ordinal, .count = 3, .args = { coords.id, a.id, b.id, 0 },
                      .res = { view, sampler, kind.ordinal } });
}

fn Node node_attribute(int location, NodeType t) => add_node({ .op = ATTRIBUTE, .type = t, .kind = (char)location });

fn Node node_system(NodeSystem s) {
    NodeType t;
    switch (s) {
        case INSTANCE_ID:
        case VERTEX_ID: t = UINT;
        case FRAG_COORD: t = VEC4;
        case FRONT_FACING: t = BOOL;
    }
    return add_node({ .op = SYSTEM, .type = t, .kind = (char)s.ordinal });
}

// A matrix of its columns: three vec3s (MAT3) or four vec4s (MAT4); or the 3x3 corner of a
// MAT4 (one part).
fn Node node_matrix(Node[] cols, NodeType t) {
    if (t == MAT3 && cols.len == 1 && cols[0].type() == MAT4) {
        return add_node({ .op = CONVERT, .type = MAT3, .count = 1, .args = { cols[0].id, 0, 0, 0 } });
    }
    int n = t == MAT3 ? 3 : 4;
    if (!is_matrix(t) || cols.len != n) abort("drie: nodes: a %s of %d columns", t, n);
    NodeData d = { .op = JOIN, .type = t, .count = (char)n };
    foreach (i, c : cols) {
        if (c.type() != vector_of(n)) abort("drie: nodes: a %s's columns are vec%ds", t, n);
        d.args[i] = c.id;
    }
    return add_node(d);
}

fn NodeType vector_of(int n) @local => type_of(FLOAT, n);

// A statement: what builtin `b` reads from here on.
fn Node node_bind(NodeBuiltin b, Node value) {
    return add_node({ .op = BIND, .type = VOID, .kind = (char)b.ordinal, .count = 1, .args = { value.id, 0, 0, 0 } });
}

// A statement: return `value` from a function (or none); in a vertex program's main, the clip
// position the vertex ends at (nothing more is worked out for it).
fn Node node_return(Node value = { }) {
    return add_node({ .op = JUMP, .type = VOID, .kind = 3, .count = value.id ? 1 : 0, .args = { value.id, 0, 0, 0 } });
}

// ---- programs

// What a program is printed from.
struct ProgramGraph {
    Node position;          // vertex: the clip position (a vec4; a SEQ for statements before it)
    Node[4] colors;         // fragment: its outputs (vec4s; 0 past the last); none: depth only
    Node depth;             // fragment: the depth it writes (a float; 0: the rasterizer's)
    Node fragment;          // fragment: a block worked out before the outputs (0: none)
    // Two programs that place vertices exactly alike (the depth prepass and the main pass
    // that tests against its depth with EQUAL) share one vertex stage: the one whose fragment
    // stage reads less is built first with `fragment_only` (no vertex stage yet), the other
    // with it as `varyings_from`, its varyings first and then its own; the first then takes
    // the second's vertex stage (ShaderProgram.take_vertex). Its fragment stage reads the
    // first of the varyings, which every backend allows.
    bool fragment_only;
    ShaderProgram* varyings_from;
}

struct ProgramView {
    ProgramStage stage;
    int source;               // the engine's view slot (shaders::VIEW_*) or NODE_USER_VIEW + k
    bool storage;
    NodeTextureKind kind;     // a texture's
    NodeType elem;            // a storage buffer's: VEC4, UVEC4 or UINT
    int msl;                  // Metal's texture(n), or buffer(n) for a storage buffer
    int hlsl;                 // HLSL's t register
    int glsl;                 // GLSL's binding of a storage buffer (its place in the list when made)
}

struct ProgramSampler {
    ProgramStage stage;
    int source;               // the engine's sampler slot (shaders::SMP_*)
    bool compare;
    int msl;                  // and HLSL's s register
}

struct ProgramPair {          // GLSL's combined samplers
    ProgramStage stage;
    int view;
    int sampler;
    ZString name;
}

struct ProgramRows {          // a stage block's rows
    NodeSource source;
    int row;
    int count;
    int at;
}

struct ShaderProgram {
    NodeLang lang;
    String vs;
    String fs;
    List{ProgramRows}[2] rows;    // by ProgramStage
    int[2] block_rows;
    List{ProgramView} views;
    List{ProgramSampler} samplers;
    List{ProgramPair} pairs;
    List{Node} varyings;          // (for programs sharing the vertex stage)
    NodeType[16] attrs;           // the vertex attributes it reads by location (VOID: not read)
    int colors;
    bool depth_out;
    bool discards;
    SgShader shader;
    // its compile in the background, plus 1 (0: none): sg_warm_shader_begin's, or the job of
    // a pipeline that compiles it with it (sg_warm_begin's, see claim)
    int job;
}

fn void ShaderProgram.free(&self) {
    self.vs.free(mem);
    self.fs.free(mem);
    self.rows[0].free();
    self.rows[1].free();
    self.views.free();
    self.samplers.free();
    foreach (pr : self.pairs) mem::free(pr.name);
    self.pairs.free();
    self.varyings.free();
}

// Its uniform block for stage `st`, from the engine's blocks (`sources`, by NodeSource; null:
// not filled for this draw, read as zeros), into `out` (block_rows[st] vec4s).
fn void ShaderProgram.gather(&self, ProgramStage st, void*[NodeSource::len] sources, sz[NodeSource::len] sizes, Vec4[] out) {
    foreach (r : self.rows[st.ordinal]) {
        char* src = sources[r.source.ordinal];
        sz from = (sz)r.row * 16;
        sz bytes = (sz)r.count * 16;
        if (!src || from + bytes > sizes[r.source.ordinal]) {
            for (int i = 0; i < r.count; i++) out[(sz)r.at + i] = { };
            if (src && from < sizes[r.source.ordinal]) mem::copy(&out[r.at], src + from, sizes[r.source.ordinal] - from);
            continue;
        }
        mem::copy(&out[r.at], src + from, bytes);
    }
}

// ---- printing

// One function's (or a stage's main's) text, and what printing it knows.
struct GenBody @private {
    DString out;
    uint[] memo;          // per node: the scope it was printed in (0: not printed)
    uint[] memo_stamp;    //   the assignment count then
    uint[] serial;        //   how many times (its name's suffix)
    uint[] local;         //   its number in this body
    String[] same_as;     //   the name it reads instead, printed for another node of the same value
                          //   (see ProgramGen.emit; null: its own)
    uint locals;
    // what has been printed, by its type and expression: the node whose name it is, and that
    // name's suffix then
    HashMap{String, uint[2]} exprs;
    List{bool} scope_open;
    uint scope;
    int indent;
    int func;             // the function (in node_funcs), -1 for main
    // what it reads that Metal hands functions as arguments (its own and its calls')
    bool block;
    List{int} views;
    List{int} samplers;
}

struct FuncDone @private {
    bool printed;
    bool printing;
    bool block;
    List{int} views;
    List{int} samplers;
}

struct ProgramGen @private {
    NodeLang lang;
    ProgramStage stage;
    ShaderProgram* p;
    GenBody* body;
    uint nodes;              // node_count() when printing started
    uint stamp;
    uint[] var_stamp;        // per VAR: its last assignment's count; per builtin b, at nodes + 1 + b
    uint[][] var_sets;       // per node: the VARs (and builtins) its value reads, ascending
    bool[] var_set_done;
    Node[NodeBuiltin::len] binding;
    DString funcs;           // the stage's functions, each before its callers
    FuncDone[] func_done;    // by node_funcs index
    NodeBuild* build;
    bool frag_coord;
    bool front_facing;
    bool instance_id;
    bool vertex_id;
}

fn GenBody* ProgramGen.new_body(&self, int func) {
    GenBody* b = mem::tnew(GenBody);
    sz count = (sz)self.nodes + 1;
    b.out = dstring::temp();
    b.memo = mem::temp_array(uint, count);
    b.memo_stamp = mem::temp_array(uint, count);
    b.serial = mem::temp_array(uint, count);
    b.local = mem::temp_array(uint, count);
    b.same_as = mem::temp_array(String, count);
    b.exprs.tinit();
    b.scope_open.tinit();
    b.scope_open.push(true);
    b.scope = 1;
    b.func = func;
    b.views.tinit();
    b.samplers.tinit();
    return b;
}

fn void ProgramGen.line(&self, String format, args...) {
    GenBody* b = self.body;
    b.out.append("    ");
    for (int i = 0; i < b.indent; i++) b.out.append("    ");
    b.out.appendf(format, ...args);
    b.out.append("\n");
}

fn String ProgramGen.type_name(&self, NodeType t) {
    bool glsl = self.lang == GLSL;
    switch (t) {
        case FLOAT: return "float";
        case BOOL: return "bool";
        case INT: return "int";
        case UINT: return "uint";
        case VOID: return "void";
        case VEC2: return glsl ? "vec2" : "float2";
        case VEC3: return glsl ? "vec3" : "float3";
        case VEC4: return glsl ? "vec4" : "float4";
        case IVEC2: return glsl ? "ivec2" : "int2";
        case IVEC3: return glsl ? "ivec3" : "int3";
        case IVEC4: return glsl ? "ivec4" : "int4";
        case UVEC2: return glsl ? "uvec2" : "uint2";
        case UVEC3: return glsl ? "uvec3" : "uint3";
        case UVEC4: return glsl ? "uvec4" : "uint4";
        case MAT4: return glsl ? "mat4" : "float4x4";
        case MAT3: return glsl ? "mat3" : "float3x3";
    }
}

fn String scalar_zero(NodeScalar s) @local {
    switch (s) {
        case FLOAT: return "0.0";
        case INT: return "0";
        case UINT: return "0u";
        case BOOL: return "false";
    }
}

fn String scalar_one(NodeScalar s) @local {
    switch (s) {
        case FLOAT: return "1.0";
        case INT: return "1";
        case UINT: return "1u";
        case BOOL: return "true";
    }
}

fn String ProgramGen.zero(&self, NodeType t) {
    int n = components(t);
    String z = scalar_zero(scalar_of(t));
    if (n == 1) return z;
    if (self.lang == HLSL) return string::tformat("((%s)0)", self.type_name(t));
    return string::tformat("%s(%s)", self.type_name(t), z);
}

fn String float_literal(float f) @local => string::tformat("(%.8e)", f);

fn String const_component(NodeType t, float[16]* v, int i) @private {
    uint bits = bitcast((*v)[i], uint);
    switch (scalar_of(t)) {
        case INT: return string::tformat("(%d)", (int)bits);
        case UINT: return string::tformat("%du", bits);
        case BOOL: return (*v)[i] != 0 ? "true" : "false";
        case FLOAT: return float_literal((*v)[i]);
    }
}

fn String ProgramGen.name(&self, Node n) {
    GenBody* b = self.body;
    if (b.same_as[n.id].ptr) return b.same_as[n.id];
    uint id = b.local[n.id];
    uint k = b.serial[n.id];
    if (k <= 1) return string::tformat("n%d", id);
    return string::tformat("n%d_%d", id, k);
}

// `x` (of type `from`) as a `to`, as TSL converts: a number fills a vector, a narrower vector
// gets zeros and a w of 1, a wider one loses its last components.
fn String ProgramGen.convert(&self, String x, NodeType from, NodeType to) {
    if (from == to) return x;
    if (from == VOID || to == VOID) abort("drie: nodes: a statement (or a function that returns nothing) where a value goes");
    if (is_matrix(from) || is_matrix(to)) abort("drie: nodes: a %s doesn't convert to a %s (node_matrix)", from, to);
    NodeScalar sf = scalar_of(from);
    NodeScalar st = scalar_of(to);
    int f = components(from);
    int t = components(to);
    if (st == BOOL) return string::tformat("(%s != %s)", f == 1 ? x : string::tformat("%s.x", x), scalar_zero(sf));
    if (sf == BOOL) return self.convert(string::tformat("(%s ? %s : %s)", x, scalar_one(st), scalar_zero(st)), type_of(st, 1), to);
    if (f != t) {
        NodeType mid = type_of(sf, t);
        String z = scalar_zero(sf);
        if (f == 1) {
            x = self.lang == HLSL ? string::tformat("((%s)%s)", self.type_name(mid), x) : string::tformat("%s(%s)", self.type_name(mid), x);
        } else if (t < f) {
            x = string::tformat("%s.%s", x, "xyzw"[:(sz)t]);
        } else {
            switch (f * 10 + t) {
                case 23: x = string::tformat("%s(%s, %s)", self.type_name(mid), x, z);
                case 24: x = string::tformat("%s(%s, %s, %s)", self.type_name(mid), x, z, scalar_one(sf));
                default: x = string::tformat("%s(%s, %s)", self.type_name(mid), x, scalar_one(sf));
            }
        }
    }
    if (sf == st) return x;
    if (self.lang == HLSL) return string::tformat("((%s)%s)", self.type_name(to), x);
    return string::tformat("%s(%s)", self.type_name(to), x);
}

fn String ProgramGen.value_as(&self, Node n, NodeType t) => self.convert(self.emit(n), n.type(), t);

// A matrix product, a * b in GLSL's order (HLSL turns it around, as SPIRV-Cross does: its
// matrices are made of the columns as rows).
fn String ProgramGen.mat_mul(&self, String a, String b) {
    if (self.lang == HLSL) return string::tformat("mul(%s, %s)", b, a);
    return string::tformat("(%s * %s)", a, b);
}

fn String ProgramGen.fn_name(&self, NodeFn f) {
    switch (f) {
        case INVERSESQRT: return self.lang == GLSL ? "inversesqrt" : "rsqrt";
        case FRACT: return self.lang == HLSL ? "frac" : "fract";
        case DFDX: return self.lang == GLSL ? "dFdx" : (self.lang == MSL ? "dfdx" : "ddx");
        case DFDY: return self.lang == GLSL ? "dFdy" : (self.lang == MSL ? "dfdy" : "ddy");
        case ATAN2: return self.lang == GLSL ? "atan" : "atan2";
        case MIX: return self.lang == HLSL ? "lerp" : "mix";
        default:
            String s = string::tformat("%s", f);
            return s.to_lower_tcopy();
    }
}

// ---- what reads what

fn uint[] merge_sets(uint[] a, uint[] b) @local {
    if (!a.len) return b;
    if (!b.len) return a;
    uint[] r = mem::temp_array(uint, a.len + b.len);
    sz i;
    sz j;
    sz k;
    while (i < a.len || j < b.len) {
        uint x;
        if (j >= b.len || (i < a.len && a[i] < b[j])) {
            x = a[i++];
        } else if (i >= a.len || b[j] < a[i]) {
            x = b[j++];
        } else {
            x = a[i++];
            j++;
        }
        r[k++] = x;
    }
    return r[:k];
}

fn uint ProgramGen.builtin_var(&self, NodeBuiltin b) => self.nodes + 1 + b.ordinal;

// The variables (and builtins) `n`'s value reads. A SEQ's are the builtins its statements and
// value read and don't bind themselves; a builtin reads what it is bound to.
fn uint[] ProgramGen.vars_of(&self, Node n) {
    if (self.var_set_done[n.id]) return self.var_sets[n.id];
    NodeData* d = node_data(n);
    uint[] r;
    switch (d.op) {
        case VAR:
            r = mem::temp_array(uint, 1);
            r[0] = n.id;
        case SEQ:
            // (a builtin it reads can be bound to it)
            self.var_sets[n.id] = { };
            self.var_set_done[n.id] = true;
            sz count = (sz)self.nodes + 1;
            bool[] seen = mem::temp_array(bool, count);
            bool[] reads = mem::temp_array(bool, count + NodeBuiltin::len);
            bool[] assigns = mem::temp_array(bool, count + NodeBuiltin::len);
            self.seq_reads_in(d.args[0], seen, reads, assigns);
            self.seq_reads({ d.args[1] }, seen, reads, assigns);
            uint k;
            for (sz i = count; i < reads.len; i++) {
                if (reads[i] && !assigns[i]) k++;
            }
            r = mem::temp_array(uint, k);
            k = 0;
            for (sz i = count; i < reads.len; i++) {
                if (reads[i] && !assigns[i]) r[k++] = (uint)i;
            }
        case CONST:
        case UNIFORM:
        case PARAM:
        case ATTRIBUTE:
        case SYSTEM:
            break;
        case BUILTIN:
            uint[] own = mem::temp_array(uint, 1);
            own[0] = self.builtin_var((NodeBuiltin)d.kind);
            Node bound = self.binding[d.kind];
            r = bound.id ? merge_sets(own, self.vars_of(bound)) : own;
        case VARYING:
            if (self.stage == VERTEX) r = self.vars_of({ d.args[0] });
        case CALL_FN:
            for (int i = 0; i < d.count; i++) r = merge_sets(r, self.vars_of({ call_arg(d, i) }));
        default:
            for (int i = 0; i < 4; i++) {
                if (d.args[i]) r = merge_sets(r, self.vars_of({ d.args[i] }));
            }
    }
    self.var_sets[n.id] = r;
    self.var_set_done[n.id] = true;
    return r;
}

// What the statements from `first` read and assign, for a SEQ's vars_of.
fn void ProgramGen.seq_reads_in(&self, uint first, bool[] seen, bool[] reads, bool[] assigns) {
    for (uint s = first; s; s = node_data({ s }).next) self.seq_reads({ s }, seen, reads, assigns);
}

fn void ProgramGen.seq_reads(&self, Node n, bool[] seen, bool[] reads, bool[] assigns) {
    if (!n.id || n.id >= seen.len || seen[n.id]) return;
    seen[n.id] = true;
    NodeData* d = node_data(n);
    switch (d.op) {
        case VAR:
            reads[n.id] = true;
            self.seq_reads({ d.args[0] }, seen, reads, assigns);
        case SEQ:
            foreach (v : self.vars_of(n)) reads[v] = true;
            self.seq_assigns_in(d.args[0], assigns);
        case ASSIGN:
            NodeData* t = node_data({ d.args[0] });
            assigns[t.op == VAR ? d.args[0] : t.args[0]] = true;
            self.seq_reads({ d.args[0] }, seen, reads, assigns);
            self.seq_reads({ d.args[1] }, seen, reads, assigns);
        case IF:
            self.seq_reads({ d.args[0] }, seen, reads, assigns);
            self.seq_reads_in(d.args[1], seen, reads, assigns);
            self.seq_reads_in(d.args[2], seen, reads, assigns);
        case LOOP:
            if (d.args[0]) assigns[d.args[0]] = true;
            for (int i = 0; i < 3; i++) self.seq_reads({ d.args[i] }, seen, reads, assigns);
            self.seq_reads_in(d.args[3], seen, reads, assigns);
        case BIND:
            assigns[self.builtin_var((NodeBuiltin)d.kind)] = true;
            self.seq_reads({ d.args[0] }, seen, reads, assigns);
        case BUILTIN:
            reads[self.builtin_var((NodeBuiltin)d.kind)] = true;
            Node bound = self.binding[d.kind];
            if (bound.id) {
                foreach (v : self.vars_of(bound)) reads[v] = true;
            }
        case VARYING:
            if (self.stage == VERTEX) self.seq_reads({ d.args[0] }, seen, reads, assigns);
        case CALL_FN:
            for (int i = 0; i < d.count; i++) self.seq_reads({ call_arg(d, i) }, seen, reads, assigns);
        case CONST:
        case UNIFORM:
        case PARAM:
        case ATTRIBUTE:
        case SYSTEM:
            break;
        default:
            for (int i = 0; i < 4; i++) self.seq_reads({ d.args[i] }, seen, reads, assigns);
    }
}

// What the statements from `first` assign, in their blocks too (not in SEQs in them).
fn void ProgramGen.seq_assigns_in(&self, uint first, bool[] assigns) {
    for (uint s = first; s; s = node_data({ s }).next) {
        NodeData* d = node_data({ s });
        switch (d.op) {
            case ASSIGN:
                NodeData* t = node_data({ d.args[0] });
                assigns[t.op == VAR ? d.args[0] : t.args[0]] = true;
            case IF:
                self.seq_assigns_in(d.args[1], assigns);
                self.seq_assigns_in(d.args[2], assigns);
            case LOOP:
                if (d.args[0]) assigns[d.args[0]] = true;
                self.seq_assigns_in(d.args[3], assigns);
            case BIND:
                assigns[self.builtin_var((NodeBuiltin)d.kind)] = true;
            default:
                break;
        }
    }
}

fn bool ProgramGen.printed(&self, Node n) {
    GenBody* b = self.body;
    uint at = b.memo[n.id];
    if (!at || !b.scope_open[(sz)at - 1]) return false;
    if (node_data(n).op == VAR) return true;
    foreach (v : self.vars_of(n)) {
        if (self.var_stamp[v] > b.memo_stamp[n.id]) return false;
    }
    return true;
}

fn String ProgramGen.mark(&self, Node n) {
    GenBody* b = self.body;
    if (!b.local[n.id]) b.local[n.id] = ++b.locals;
    b.same_as[n.id] = { };
    b.memo[n.id] = b.scope;
    b.memo_stamp[n.id] = self.stamp;
    b.serial[n.id]++;
    return self.name(n);
}

fn void ProgramGen.assigned(&self, uint v) {
    self.stamp++;
    self.var_stamp[v] = self.stamp;
}

// Marks the variables the statements from `first` (and the blocks and values in them) assign
// as assigned now: what was printed before them and reads them is printed again inside a loop.
fn void ProgramGen.assigns_in(&self, uint first, bool[] seen) {
    for (uint s = first; s; s = node_data({ s }).next) self.assigns_of({ s }, seen);
}

fn void ProgramGen.assigns_of(&self, Node n, bool[] seen) {
    if (!n.id || seen[n.id]) return;
    seen[n.id] = true;
    NodeData* d = node_data(n);
    switch (d.op) {
        case ASSIGN:
            NodeData* t = node_data({ d.args[0] });
            self.assigned(t.op == VAR ? d.args[0] : t.args[0]);
            self.assigns_of({ d.args[1] }, seen);
        case IF:
            self.assigns_of({ d.args[0] }, seen);
            self.assigns_in(d.args[1], seen);
            self.assigns_in(d.args[2], seen);
        case LOOP:
            if (d.kind != NodeLoop.WHILE.ordinal) self.assigned(d.args[0]);
            self.assigns_of({ d.args[1] }, seen);
            self.assigns_of({ d.args[2] }, seen);
            self.assigns_in(d.args[3], seen);
        case SEQ:
            self.assigns_in(d.args[0], seen);
            self.assigns_of({ d.args[1] }, seen);
        case CALL_FN:
            for (int i = 0; i < d.count; i++) self.assigns_of({ call_arg(d, i) }, seen);
        case BIND:
            self.assigned(self.builtin_var((NodeBuiltin)d.kind));
            self.assigns_of({ d.args[0] }, seen);
        default:
            for (int i = 0; i < 4; i++) self.assigns_of({ d.args[i] }, seen);
    }
}

fn void ProgramGen.open_scope(&self) {
    GenBody* b = self.body;
    b.scope_open.push(true);
    b.scope = (uint)b.scope_open.len();
    b.indent++;
}

fn void ProgramGen.close_scope(&self, uint outer) {
    GenBody* b = self.body;
    b.scope_open[(sz)b.scope - 1] = false;
    b.scope = outer;
    b.indent--;
}

fn void ProgramGen.statements(&self, uint first) {
    for (uint s = first; s; s = node_data({ s }).next) self.statement({ s });
}

fn void ProgramGen.block(&self, uint first) {
    uint outer = self.body.scope;
    self.open_scope();
    self.statements(first);
    self.close_scope(outer);
}

fn String ProgramGen.lvalue(&self, Node n) {
    NodeData* d = node_data(n);
    if (d.op == VAR) return self.emit(n);
    String v = self.emit({ d.args[0] });
    return string::tformat("%s.%s", v, (String)d.swizzle[:d.count]);
}

fn String ProgramGen.discard(&self) => self.lang == MSL ? "discard_fragment();" : "discard;";

fn void ProgramGen.statement(&self, Node s) {
    NodeData d = *node_data(s);
    switch (d.op) {
        case EVAL:
            self.emit({ d.args[0] });
        case ASSIGN:
            Node target = { d.args[0] };
            String value = self.value_as({ d.args[1] }, target.type());
            self.line("%s = %s;", self.lvalue(target), value);
            NodeData* t = node_data(target);
            self.assigned(t.op == VAR ? target.id : t.args[0]);
        case IF:
            String cond = self.value_as({ d.args[0] }, BOOL);
            self.hoist_shared(d.args[1], d.args[2]);
            self.line("if (%s) {", cond);
            self.block(d.args[1]);
            if (d.args[2]) {
                self.line("} else {");
                self.block(d.args[2]);
            }
            self.line("}");
        case LOOP:
            NodeLoop how = (NodeLoop)d.kind;
            GenBody* b = self.body;
            uint outer = b.scope;
            // (what the loop assigns: read again inside it)
            bool[] seen = mem::temp_array(bool, (sz)self.nodes + 1);
            self.assigns_in(d.args[3], seen);
            if (how == WHILE) {
                self.line("for (;;) {");
                self.open_scope();
                self.line("if (!%s) break;", self.value_as({ d.args[2] }, BOOL));
            } else {
                Node index = { d.args[0] };
                NodeType t = index.type();
                String from = self.value_as({ d.args[1] }, t);
                String to = self.value_as({ d.args[2] }, t);
                String[*] ops = { "<", "<=", ">", ">=" };
                b.scope_open.push(true);
                b.scope = (uint)b.scope_open.len();
                self.assigned(index.id);
                String i = self.mark(index);
                self.line("for (%s %s = %s; %s %s %s; %s%s) {", self.type_name(t), i, from, i, ops[how.ordinal], to, i,
                          how.ordinal < 2 ? "++" : "--");
                b.indent++;
            }
            self.statements(d.args[3]);
            self.close_scope(outer);
            self.line("}");
        case JUMP:
            switch (d.kind) {
                case 0: self.line("break;");
                case 1: self.line("continue;");
                case 2:
                    if (self.stage != FRAGMENT) abort("drie: nodes: discard is for the fragment stage");
                    if (self.body.func >= 0 && node_funcs[self.body.func].pure) abort("drie: nodes: discard in a function (%s)", node_funcs[self.body.func].name);
                    self.p.discards = true;
                    self.line(self.discard());
                default:
                    self.ret(d.count ? (Node) { d.args[0] } : (Node) { });
            }
        case BIND:
            if (self.body.func >= 0) abort("drie: nodes: bind() in a function");
            // (bound again to what it is bound to: what reads it reads the same)
            if (self.binding[d.kind].id == d.args[0]) break;
            self.binding[d.kind] = { d.args[0] };
            // (what reads the builtin, or what it was bound to, reads it again)
            for (sz i = 0; i < self.var_set_done.len; i++) self.var_set_done[i] = false;
            self.assigned(self.builtin_var((NodeBuiltin)d.kind));
        default:
            self.emit(s);
    }
}

// What the blocks `a` and `b` (an if's two) both read and can be worked out before the if:
// printed here, once, rather than in each of them (and again in any other block after them).
// That is a value of the graph (no statement, variable or function call in it) reading no
// variable or builtin either block assigns.
fn void ProgramGen.hoist_shared(&self, uint a, uint b) {
    if (!a || !b) return;
    // (in the temporary memory the printing is in: emit grows the body's text and lists)
    sz count = (sz)self.nodes + 1;
    char[] in = mem::temp_array(char, count);
    bool[] assigned = mem::temp_array(bool, count + NodeBuiltin::len);
    self.reads_in(a, in, 1, assigned);
    self.reads_in(b, in, 2, assigned);
    char[] can = mem::temp_array(char, count);
    for (uint i = 1; i < count; i++) {
        if (in[i] == 3 && self.hoistable({ i }, can, assigned) && !self.printed({ i })) self.emit({ i });
    }
}

// Marks with `bit` in `in` the nodes the statements from `first` read that aren't printed (and
// in `assigned` the variables and builtins they assign).
fn void ProgramGen.reads_in(&self, uint first, char[] in, char bit, bool[] assigned) {
    for (uint s = first; s; s = node_data({ s }).next) self.reads_of({ s }, in, bit, assigned);
}

fn void ProgramGen.reads_of(&self, Node n, char[] in, char bit, bool[] assigned) {
    if (!n.id || n.id >= in.len || (in[n.id] & bit)) return;
    NodeData* d = node_data(n);
    bool statement = d.op == ASSIGN || d.op == IF || d.op == LOOP || d.op == JUMP || d.op == EVAL || d.op == BIND;
    if (!statement && self.printed(n)) return;
    in[n.id] |= bit;
    switch (d.op) {
        case ASSIGN:
            NodeData* t = node_data({ d.args[0] });
            assigned[t.op == VAR ? d.args[0] : t.args[0]] = true;
            self.reads_of({ d.args[0] }, in, bit, assigned);
            self.reads_of({ d.args[1] }, in, bit, assigned);
        case IF:
            self.reads_of({ d.args[0] }, in, bit, assigned);
            self.reads_in(d.args[1], in, bit, assigned);
            self.reads_in(d.args[2], in, bit, assigned);
        case LOOP:
            assigned[d.args[0]] = true;
            self.reads_of({ d.args[1] }, in, bit, assigned);
            self.reads_of({ d.args[2] }, in, bit, assigned);
            self.reads_in(d.args[3], in, bit, assigned);
        case SEQ:
            self.reads_in(d.args[0], in, bit, assigned);
            self.reads_of({ d.args[1] }, in, bit, assigned);
        case BIND:
            assigned[self.builtin_var((NodeBuiltin)d.kind)] = true;
            self.reads_of({ d.args[0] }, in, bit, assigned);
        case BUILTIN:
            self.reads_of(self.binding[d.kind], in, bit, assigned);
        case VARYING:
            if (self.stage == VERTEX) self.reads_of({ d.args[0] }, in, bit, assigned);
        case CALL_FN:
            for (int i = 0; i < d.count; i++) self.reads_of({ call_arg(d, i) }, in, bit, assigned);
        default:
            for (int i = 0; i < 4; i++) self.reads_of({ d.args[i] }, in, bit, assigned);
    }
}

// Whether `n` can be printed before the blocks that assign `assigned` (`can`: per node, 0 not
// known, 1 yes, 2 no).
fn bool ProgramGen.hoistable(&self, Node n, char[] can, bool[] assigned) {
    if (!n.id) return true;
    if (can[n.id]) return can[n.id] == 1;
    bool ok = true;
    NodeData* d = node_data(n);
    if (self.printed(n)) {
        ok = true;
    } else {
        switch (d.op) {
            case CONST:
            case UNIFORM:
            case INPUT:
            case ATTRIBUTE:
            case SYSTEM:
                ok = d.op != INPUT || self.hoistable({ d.args[0] }, can, assigned);
            case BUILTIN:
                Node bound = self.binding[d.kind];
                ok = bound.id && !assigned[self.builtin_var((NodeBuiltin)d.kind)] && self.hoistable(bound, can, assigned);
            case VARYING:
                ok = self.stage == FRAGMENT || self.hoistable({ d.args[0] }, can, assigned);
            case TEXTURE:
            case STORAGE:
            case SAMPLE:
            case BINARY:
            case CALL:
            case SELECT:
            case SWIZZLE:
            case JOIN:
            case CONVERT:
            case NOT:
                for (int i = 0; i < 4 && ok; i++) ok = self.hoistable({ d.args[i] }, can, assigned);
            case CALL_FN:
                ok = d.type != VOID && node_funcs[call_func(d)].pure;
                for (int i = 0; i < d.count && ok; i++) ok = self.hoistable({ call_arg(d, i) }, can, assigned);
            default:
                ok = false;
        }
    }
    if (ok) {
        foreach (v : self.vars_of(n)) {
            if (assigned[v]) ok = false;
        }
    }
    can[n.id] = ok ? 1 : 2;
    return ok;
}

fn void ProgramGen.ret(&self, Node value) {
    GenBody* b = self.body;
    if (b.func >= 0) {
        NodeType rt = node_funcs[b.func].ret;
        if (rt == VOID) {
            self.line("return;");
        } else {
            if (!value.id) abort("drie: nodes: %s returns a %s", node_funcs[b.func].name, rt);
            self.line("return %s;", self.value_as(value, rt));
        }
        return;
    }
    if (self.stage != VERTEX) abort("drie: nodes: return in a fragment program's main");
    String pos = self.value_as(value, VEC4);
    if (self.lang == GLSL) {
        self.line("gl_Position = %s;", pos);
        self.line("return;");
    } else {
        self.line("vout.pos = %s;", pos);
        self.line("return vout;");
    }
}


// ---- sokol's description

const ZString[16] ATTR_GLSL_NAMES @local = { "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12", "a13", "a14", "a15" };

fn SgShaderStage sg_stage(ProgramStage st) @local => st == VERTEX ? SgShaderStage.VERTEX : SgShaderStage.FRAGMENT;

// Its sokol description (pointing into the program's text).
fn SgShaderDesc ShaderProgram.desc(&self) {
    SgShaderDesc d;
    d.label = "drie-program";
    d.vertex_func.source = (ZString)self.vs.ptr;
    d.fragment_func.source = (ZString)self.fs.ptr;
    switch (self.lang) {
        case MSL:
            d.vertex_func.entry = "main0";
            d.fragment_func.entry = "main0";
        case HLSL:
            d.vertex_func.entry = "main";
            d.fragment_func.entry = "main";
            d.vertex_func.d3d11_target = "vs_5_0";
            d.fragment_func.d3d11_target = "ps_5_0";
        case GLSL:
            d.vertex_func.entry = "main";
            d.fragment_func.entry = "main";
    }
    // (every location the engine's vertex layouts have: D3D11's input layout names them all)
    for (int i = 0; i < 16; i++) {
        d.attrs[i].base_type = SgShaderAttrBaseType.FLOAT;
        d.attrs[i].glsl_name = ATTR_GLSL_NAMES[i];
        d.attrs[i].hlsl_sem_name = "TEXCOORD";
        d.attrs[i].hlsl_sem_index = (char)i;
    }
    for (int st = 0; st < 2; st++) {
        if (!self.block_rows[st]) continue;
        SgShaderUniformBlock* b = &d.uniform_blocks[st];
        b.stage = sg_stage((ProgramStage)st);
        b.size = (uint)self.block_rows[st] * 16;
        b.layout = SgUniformLayout.STD140;
        b.msl_buffer_n = (char)st;
        b.hlsl_register_b_n = 0;
        b.glsl_uniforms[0].type = SgUniformType.FLOAT4;
        b.glsl_uniforms[0].array_count = (ushort)self.block_rows[st];
        b.glsl_uniforms[0].glsl_name = st == 0 ? "vu" : "fu";
    }
    if (self.views.len() > 32) abort("drie: a program reads more than 32 textures and buffers");
    foreach (i, v : self.views) {
        if (v.storage) {
            SgShaderStorageBufferView* b = &d.views[i].storage_buffer;
            b.stage = sg_stage(v.stage);
            b.readonly = true;
            b.msl_buffer_n = (char)v.msl;
            b.hlsl_register_t_n = (char)v.hlsl;
            b.glsl_binding_n = (char)v.glsl;
        } else {
            SgShaderTextureView* t = &d.views[i].texture;
            t.stage = sg_stage(v.stage);
            switch (v.kind) {
                case TEX_2D:
                case TEX_DEPTH: t.image_type = SgImageType.TYPE_2D;
                case TEX_3D: t.image_type = SgImageType.TYPE_3D;
                case TEX_ARRAY: t.image_type = SgImageType.ARRAY;
            }
            t.sample_type = v.kind == TEX_DEPTH ? SgImageSampleType.DEPTH : SgImageSampleType.FLOAT;
            t.msl_texture_n = (char)v.msl;
            t.hlsl_register_t_n = (char)v.hlsl;
        }
    }
    if (self.samplers.len() > 14) abort("drie: a program uses more than 14 samplers");
    foreach (i, s : self.samplers) {
        d.samplers[i].stage = sg_stage(s.stage);
        d.samplers[i].sampler_type = s.compare ? SgSamplerType.COMPARISON : SgSamplerType.FILTERING;
        d.samplers[i].msl_sampler_n = (char)s.msl;
        d.samplers[i].hlsl_register_s_n = (char)s.msl;
    }
    // (GLSL's combined samplers; the other backends take any view with any sampler)
    foreach (i, pr : self.pairs) {
        d.texture_sampler_pairs[i].stage = sg_stage(pr.stage);
        d.texture_sampler_pairs[i].view_slot = (char)pr.view;
        d.texture_sampler_pairs[i].sampler_slot = (char)pr.sampler;
        d.texture_sampler_pairs[i].glsl_name = pr.name;
    }
    with_bytecode(&d);
    return d;
}

// The program made (once): its sources are printed if the backend's compiler takes them not.
fn SgShader ShaderProgram.make(&self) {
    if (self.shader.id) return self.shader;
    // (the driver's cache has it once its compile in the background is done: made at once
    // then, rather than a second time beside it)
    if (self.job) {
        sg_warm_wait(self.job - 1);
        sg_warm_end(self.job - 1);
        self.job = 0;
    }
    SgShaderDesc d = self.desc();
    self.shader = sg::make_shader(&d);
    sg_warm_archivable(self.shader.id);
    if (sg::query_shader_state(self.shader) != VALID) {
        io::eprintfn("drie: a generated program failed to build.\n--- vertex:\n%s\n--- fragment:\n%s", self.vs, self.fs);
        abort("drie: a generated program failed to build");
    }
    return self.shader;
}

// Starts compiling the program on a worker thread (Metal; elsewhere it is compiled when
// make first asks for it), if it isn't made or compiling yet.
fn void ShaderProgram.start(&self) {
    if (self.shader.id || self.job) return;
    SgShaderDesc d = self.desc();
    int job = sg_warm_shader_begin(&d);
    if (job >= 0) self.job = job + 1;
}

// Makes `job` (sg_warm_begin's, a pipeline's that compiles the program) the one compiling the
// program, if it isn't made or compiling yet: false if it is (the job is then the caller's to
// end).
fn bool ShaderProgram.claim(&self, int job) {
    if (self.shader.id || self.job) return false;
    self.job = job + 1;
    return true;
}

// Whether its compile in the background is still running.
fn bool ShaderProgram.compiling(&self) => self.job && sg_warm_state(self.job - 1) == 0;

// Takes `from`'s vertex stage (see ProgramGraph.fragment_only): its source, block, attributes,
// views and samplers.
fn void ShaderProgram.take_vertex(&self, ShaderProgram* from) {
    self.vs = from.vs.copy(mem);
    self.rows[0].clear();
    foreach (r : from.rows[0]) self.rows[0].push(r);
    self.block_rows[0] = from.block_rows[0];
    self.attrs = from.attrs;
    // (each at its own place in this program's lists: Metal and HLSL number them per stage,
    // GLSL's source names them as printed, which the pairs and storage bindings keep)
    int[32] view_at;
    foreach (i, v : from.views) {
        if (v.stage != VERTEX) continue;
        view_at[i] = (int)self.views.len();
        self.views.push(v);
    }
    int[16] sampler_at;
    foreach (i, sm : from.samplers) {
        if (sm.stage != VERTEX) continue;
        sampler_at[i] = (int)self.samplers.len();
        self.samplers.push(sm);
    }
    foreach (pr : from.pairs) {
        if (pr.stage != VERTEX) continue;
        foreach (mine : self.pairs) {
            if (((ZString)mine.name).str_view() == ((ZString)pr.name).str_view()) abort("drie: nodes: two programs sharing a vertex stage name a sampler alike");
        }
        String name = ((ZString)pr.name).str_view().copy(mem);
        self.pairs.push({ .stage = VERTEX, .view = view_at[pr.view], .sampler = sampler_at[pr.sampler], .name = (ZString)name.ptr });
    }
}

// The program as text, for the pipeline cache (read_program reads it back): a line of counts,
// the two sources, then a line for each of its rows, views, samplers and pairs.
fn void ShaderProgram.write(&self, DString* out) {
    out.appendf("%d %d %d %d %d %d %d %d %d %d\n", self.lang.ordinal, self.vs.len, self.fs.len, self.block_rows[0],
                self.block_rows[1], self.rows[0].len(), self.rows[1].len(), self.views.len(), self.samplers.len(),
                self.pairs.len());
    out.append(self.vs);
    out.append(self.fs);
    for (int st = 0; st < 2; st++) {
        foreach (r : self.rows[st]) out.appendf("%d %d %d %d\n", r.source.ordinal, r.row, r.count, r.at);
    }
    foreach (v : self.views) {
        out.appendf("%d %d %d %d %d %d %d %d\n", v.stage.ordinal, v.source, v.storage ? 1 : 0, v.kind.ordinal, v.elem.ordinal,
                    v.msl, v.hlsl, v.glsl);
    }
    foreach (sm : self.samplers) out.appendf("%d %d %d %d\n", sm.stage.ordinal, sm.source, sm.compare ? 1 : 0, sm.msl);
    foreach (pr : self.pairs) out.appendf("%d %d %d %s\n", pr.stage.ordinal, pr.view, pr.sampler, (ZString)pr.name);
}

// The numbers of the line at `*at` in `text` (up to `n.len`, the count in the result's last
// place: -1 if there is no whole line there or a field isn't a number), moving `*at` past it.
fn int read_numbers(String text, usz* at, int[] n) @local {
    sz? end = text[*at..].index_of_char('\n');
    if (catch end) return -1;
    String line = text[*at:(usz)end];
    *at += (usz)end + 1;
    int count;
    @pool() {
        foreach (f : line.tsplit(" ")) {
            if (count == n.len) return -1;
            int? x = f.to_int();
            if (catch x) return -1;
            n[count++] = x;
        }
    };
    return count;
}

// A program `write` wrote, at `*at` in `text` (moved past it); none if it isn't one.
fn ShaderProgram*? read_program(String text, usz* at) {
    int[10] h;
    // (any language: a list can be read before the backend is up, see Renderer.pump_pipeline_cache)
    if (read_numbers(text, at, &h) != 10 || h[0] >= NodeLang::len) return NOT_FOUND~;
    foreach (x : h) if (x < 0) return NOT_FOUND~;
    if (*at + (usz)h[1] + (usz)h[2] > text.len || h[7] > 32 || h[8] > 16 || h[9] > 32) return NOT_FOUND~;
    ShaderProgram* p = mem::new(ShaderProgram);
    p.lang = (NodeLang)h[0];
    p.rows[0].init(mem);
    p.rows[1].init(mem);
    p.views.init(mem);
    p.samplers.init(mem);
    p.pairs.init(mem);
    p.varyings.init(mem);
    p.vs = text[*at:(usz)h[1]].copy(mem);
    *at += (usz)h[1];
    p.fs = text[*at:(usz)h[2]].copy(mem);
    *at += (usz)h[2];
    p.block_rows = { h[3], h[4] };
    bool ok = true;
    for (int st = 0; st < 2 && ok; st++) {
        for (int i = 0; i < h[5 + st] && ok; i++) {
            int[4] r;
            ok = read_numbers(text, at, &r) == 4 && r[0] >= 0 && r[0] < NodeSource::len;
            if (ok) p.rows[st].push({ (NodeSource)r[0], r[1], r[2], r[3] });
        }
    }
    for (int i = 0; i < h[7] && ok; i++) {
        int[8] v;
        ok = read_numbers(text, at, &v) == 8 && v[0] >= 0 && v[0] < 2 && v[3] >= 0 && v[3] < NodeTextureKind::len
            && v[4] >= 0 && v[4] < NodeType::len;
        if (ok) p.views.push({ (ProgramStage)v[0], v[1], v[2] == 1, (NodeTextureKind)v[3], (NodeType)v[4], v[5], v[6], v[7] });
    }
    for (int i = 0; i < h[8] && ok; i++) {
        int[4] sm;
        ok = read_numbers(text, at, &sm) == 4 && sm[0] >= 0 && sm[0] < 2;
        if (ok) p.samplers.push({ (ProgramStage)sm[0], sm[1], sm[2] == 1, sm[3] });
    }
    for (int i = 0; i < h[9] && ok; i++) {
        sz? end = text[*at..].index_of_char('\n');
        if (catch end) {
            ok = false;
            break;
        }
        String[] f = text[*at:(usz)end].tsplit(" ");
        *at += (usz)end + 1;
        ok = f.len == 4 && (f[0] == "0" || f[0] == "1");
        if (!ok) break;
        int? view = f[1].to_int();
        int? sampler = f[2].to_int();
        ok = @ok(view) && @ok(sampler);
        if (ok) p.pairs.push({ f[0] == "1" ? FRAGMENT : VERTEX, view!!, sampler!!, (ZString)f[3].copy(mem).ptr });
    }
    if (!ok) {
        p.free();
        mem::free(p);
        return NOT_FOUND~;
    }
    return p;
}

// The language the backend's programs are printed in.
fn NodeLang backend_lang() {
    switch (sg::query_backend()) {
        case GLCORE:
        case GLES3: return GLSL;
        case METAL_IOS:
        case METAL_MACOS:
        case METAL_SIMULATOR: return MSL;
        case D3D11: return HLSL;
        default: abort("drie: generated programs need the GL, Metal or D3D11 backend");
    }
}
