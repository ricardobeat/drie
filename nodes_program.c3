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
// what they read. Names number the nodes in the order they are printed, so the same graphs
// print the same text whatever else was made before, and the driver's cache knows them.
module drie;
import std::collections::list, std::collections::map, std::io, sokol;

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
}

List{NodeFunc} node_funcs @local;
List{uint} node_args @local;

// A new function `name` returning a `ret` of `params`: its parameters are node_param(f, i), its
// body is set with node_func_body. Functions are printed in a program once per stage that calls
// them; `name` is what they are called there (unique among a program's functions).
fn int node_func(String name, NodeType ret, NodeType[] params) {
    if (params.len > 12) abort("drie: nodes: a function of more than 12 parameters");
    if (!node_funcs.is_initialized()) node_funcs.init(mem);
    NodeFunc f = { .name = name.copy(mem), .ret = ret, .count = (int)params.len, .mark = node_count() };
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

fn uint call_arg(NodeData* d, int i) @local => node_args[(sz)bitcast(d.value[1], uint) + (sz)i];

fn int call_func(NodeData* d) @local => (int)bitcast(d.value[0], uint);

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
    Node fragment;          // fragment: a statement worked out before the outputs (0: none)
    // a program sharing another's vertex stage (the depth prepass the main pass's, so both
    // place vertices exactly alike): its vertex source and varyings, the fragment's from them
    ShaderProgram* vertex_of;
}

struct ProgramView {
    ProgramStage stage;
    int source;               // the engine's view slot (shaders::VIEW_*) or NODE_USER_VIEW + k
    bool storage;
    NodeTextureKind kind;     // a texture's
    NodeType elem;            // a storage buffer's: VEC4, UVEC4 or UINT
    int msl;                  // Metal's texture(n), or buffer(n) for a storage buffer
    int hlsl;                 // HLSL's t register
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
    bool discards;
    SgShader shader;
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
struct GenBody @local {
    DString out;
    uint[] memo;          // per node: the scope it was printed in (0: not printed)
    uint[] memo_stamp;    //   the assignment count then
    uint[] serial;        //   how many times (its name's suffix)
    uint[] local;         //   its number in this body
    uint locals;
    List{bool} scope_open;
    uint scope;
    int indent;
    int func;             // the function (in node_funcs), -1 for main
    // what it reads that Metal hands functions as arguments (its own and its calls')
    bool block;
    List{int} views;
    List{int} samplers;
}

struct FuncDone @local {
    bool printed;
    bool block;
    List{int} views;
    List{int} samplers;
}

struct ProgramGen @local {
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
    bool fixed_varyings;     // varyings are the shared vertex stage's: none added
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

fn String const_component(NodeType t, float[16]* v, int i) @local {
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

// The variables (and builtins) `n`'s value reads. A SEQ's is worked out once, where it is
// printed, and reads none after; a builtin reads what it is bound to.
fn uint[] ProgramGen.vars_of(&self, Node n) {
    if (self.var_set_done[n.id]) return self.var_sets[n.id];
    NodeData* d = node_data(n);
    uint[] r;
    switch (d.op) {
        case VAR:
            r = mem::temp_array(uint, 1);
            r[0] = n.id;
        case SEQ:
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
                    self.p.discards = true;
                    self.line(self.discard());
                default:
                    self.ret(d.count ? (Node) { d.args[0] } : (Node) { });
            }
        case BIND:
            if (self.body.func >= 0) abort("drie: nodes: bind() in a function");
            self.binding[d.kind] = { d.args[0] };
            // (what reads the builtin, or what it was bound to, reads it again)
            for (sz i = 0; i < self.var_set_done.len; i++) self.var_set_done[i] = false;
            self.assigned(self.builtin_var((NodeBuiltin)d.kind));
        default:
            self.emit(s);
    }
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

// ---- resources

fn int ProgramGen.view(&self, int source, bool storage, NodeTextureKind kind, NodeType elem) {
    ShaderProgram* p = self.p;
    int msl;
    int hlsl;
    foreach (i, v : p.views) {
        if (v.stage != self.stage) continue;
        if (v.source == source && v.storage == storage) {
            if (!storage && v.kind != kind) abort("drie: nodes: view %d read as a %s and as a %s", source, v.kind, kind);
            if (storage && v.elem != elem) abort("drie: nodes: storage buffer %d read as %ss and as %ss", source, v.elem, elem);
            self.uses_view((int)i);
            return (int)i;
        }
        if (v.storage == storage) msl++;
        hlsl++;
    }
    p.views.push({ .stage = self.stage, .source = source, .storage = storage, .kind = kind, .elem = elem,
                   .msl = storage ? 8 + msl : msl, .hlsl = hlsl });
    int at = (int)p.views.len() - 1;
    self.uses_view(at);
    return at;
}

fn void ProgramGen.uses_view(&self, int v) {
    if (!self.body.views.contains(v)) self.body.views.push(v);
}

fn int ProgramGen.sampler(&self, int source, bool compare) {
    ShaderProgram* p = self.p;
    int n;
    foreach (i, s : p.samplers) {
        if (s.stage != self.stage) continue;
        if (s.source == source) {
            if (s.compare != compare) abort("drie: nodes: sampler %d used to compare and to filter", source);
            if (!self.body.samplers.contains((int)i)) self.body.samplers.push((int)i);
            return (int)i;
        }
        n++;
    }
    p.samplers.push({ .stage = self.stage, .source = source, .compare = compare, .msl = n });
    int at = (int)p.samplers.len() - 1;
    self.body.samplers.push(at);
    return at;
}

// GLSL's combined sampler of view `v` with sampler `s`.
fn String ProgramGen.pair(&self, int v, int s) {
    ShaderProgram* p = self.p;
    foreach (pr : p.pairs) {
        if (pr.view == v && pr.sampler == s) return ((ZString)pr.name).str_view();
    }
    String name = string::format(mem, "t%ds%d", v, s);
    p.pairs.push({ .stage = self.stage, .view = v, .sampler = s, .name = (ZString)name.ptr });
    return name;
}

// The row of the stage's block where rows `row`..+`count` of `source` are.
fn int ProgramGen.rows(&self, NodeSource source, int row, int count) {
    ShaderProgram* p = self.p;
    int st = self.stage.ordinal;
    foreach (r : p.rows[st]) {
        if (r.source == source && r.row == row && r.count == count) return r.at;
    }
    int at = p.block_rows[st];
    p.rows[st].push({ .source = source, .row = row, .count = count, .at = at });
    p.block_rows[st] += count;
    return at;
}

fn String ProgramGen.block_name(&self) => self.stage == VERTEX ? "vu" : "fu";

fn String ProgramGen.input(&self, NodeData* d) {
    self.body.block = true;
    NodeType t = d.type;
    int rows = t == MAT4 ? 4 : (t == MAT3 ? 3 : 1);
    int stride = (int)bitcast(d.value[0], uint);
    int len = (int)bitcast(d.value[1], uint);
    bool indexed = d.args[0] != 0;
    int at = self.rows((NodeSource)d.res[0], d.res[1], indexed ? stride * (len - 1) + rows : rows);
    String base = self.block_name();
    String index = indexed ? string::tformat(" + %s * %d", self.value_as({ d.args[0] }, INT), stride) : "";
    String[4] r;
    for (int i = 0; i < rows; i++) r[i] = string::tformat("%s[%d%s]", base, at + i, index);
    switch (t) {
        case MAT4:
            // (its columns; HLSL's constructor takes rows, which mul(v, m) reads as columns)
            return string::tformat("%s(%s, %s, %s, %s)", self.type_name(MAT4), r[0], r[1], r[2], r[3]);
        case MAT3:
            return string::tformat("%s(%s.xyz, %s.xyz, %s.xyz)", self.type_name(MAT3), r[0], r[1], r[2]);
        default:
            int c = components(t);
            String x = c == 4 ? r[0] : string::tformat("%s.%s", r[0], "xyzw"[(sz)d.res[2]:(sz)c]);
            return self.convert(x, type_of(FLOAT, c), t);
    }
}

// A node material's uniform (nodes::uniform): a row of its USER block, in its NodeBuild's order.
fn String ProgramGen.user_uniform(&self, Node n) {
    NodeBuild* b = self.build;
    if (!b) abort("drie: nodes: a uniform in a program without a material's build");
    int at = -1;
    int row;
    foreach (u : b.uniforms) {
        if (u.id == n.id) at = row;
        row += u.type() == MAT4 ? 4 : 1;
    }
    if (at < 0) {
        at = row;
        b.uniforms.push(n);
    }
    NodeType t = n.type();
    NodeData d = { .op = INPUT, .type = t == MAT4 ? MAT4 : type_of(FLOAT, components(t)), .res = { NodeSource.USER.ordinal, at, 0 } };
    String x = self.input(&d);
    // (integers and bools as floats: Node.set)
    return t == MAT4 ? x : self.convert(x, d.type, t);
}

fn String ProgramGen.storage(&self, NodeData* d) {
    int v = self.view(d.res[0], true, TEX_2D, d.type);
    String i = self.value_as({ d.args[0] }, d.type == UINT ? UINT : INT);
    if (self.lang != HLSL) return string::tformat("b%d[%s]", v, i);
    switch (d.type) {
        case VEC4: return string::tformat("asfloat(b%d.Load4(uint(%s) * 16u))", v, i);
        case UVEC4: return string::tformat("b%d.Load4(uint(%s) * 16u)", v, i);
        default: return string::tformat("b%d.Load(uint(%s) * 4u)", v, i);
    }
}

fn String ProgramGen.sample(&self, Node n, NodeData* d) {
    NodeSampling how = (NodeSampling)d.kind;
    NodeTextureKind kind = (NodeTextureKind)d.res[2];
    int v = self.view(d.res[0], false, kind, VOID);
    int s = self.sampler(d.res[1], kind == TEX_DEPTH);
    String t = string::tformat("t%d", v);
    String smp = string::tformat("s%d", s);
    String ts = self.lang == GLSL ? self.pair(v, s) : "";
    bool vol = kind == TEX_3D || kind == TEX_ARRAY;
    NodeType ct = vol ? VEC3 : VEC2;
    switch (how) {
        case SAMPLE:
        case LEVEL:
        case GRAD:
            String c = self.value_as({ d.args[0] }, ct);
            String extra;
            switch (how) {
                case LEVEL:
                    String l = self.value_as({ d.args[1] }, FLOAT);
                    if (self.lang == MSL) extra = string::tformat(", level(%s)", l);
                    if (self.lang == GLSL) return string::tformat("textureLod(%s, %s, %s)", ts, c, l);
                    if (self.lang == HLSL) return string::tformat("%s.SampleLevel(%s, %s, %s)", t, smp, c, l);
                case GRAD:
                    NodeType gt = kind == TEX_3D ? VEC3 : VEC2;
                    String dx = self.value_as({ d.args[1] }, gt);
                    String dy = self.value_as({ d.args[2] }, gt);
                    if (self.lang == MSL) extra = string::tformat(", gradient%s(%s, %s)", kind == TEX_3D ? "3d" : "2d", dx, dy);
                    if (self.lang == GLSL) return string::tformat("textureGrad(%s, %s, %s, %s)", ts, c, dx, dy);
                    if (self.lang == HLSL) return string::tformat("%s.SampleGrad(%s, %s, %s, %s)", t, smp, c, dx, dy);
                default:
                    if (self.lang == GLSL) return string::tformat("texture(%s, %s)", ts, c);
                    if (self.lang == HLSL) return string::tformat("%s.Sample(%s, %s)", t, smp, c);
            }
            if (kind == TEX_ARRAY) return string::tformat("%s.sample(%s, %s.xy, uint(rint(%s.z))%s)", t, smp, c, c, extra);
            return string::tformat("%s.sample(%s, %s%s)", t, smp, c, extra);
        case FETCH:
            String c = self.value_as({ d.args[0] }, vol ? IVEC3 : IVEC2);
            String l = d.args[1] ? self.value_as({ d.args[1] }, INT) : "0";
            switch (self.lang) {
                case GLSL: return string::tformat("texelFetch(%s, %s, %s)", ts, c, l);
                case HLSL: return string::tformat("%s.Load(%s(%s, %s))", t, vol ? "int4" : "int3", c, l);
                case MSL:
                    if (kind == TEX_ARRAY) return string::tformat("%s.read(uint2(%s.xy), uint(%s.z), uint(%s))", t, c, c, l);
                    return string::tformat("%s.read(%s(%s), uint(%s))", t, vol ? "uint3" : "uint2", c, l);
            }
        case SIZE:
            String l = d.args[0] ? self.value_as({ d.args[0] }, INT) : "0";
            switch (self.lang) {
                case GLSL: return string::tformat("textureSize(%s, %s)", ts, l);
                case MSL:
                    String w = string::tformat("int(%s.get_width(uint(%s)))", t, l);
                    String h = string::tformat("int(%s.get_height(uint(%s)))", t, l);
                    if (kind == TEX_3D) return string::tformat("int3(%s, %s, int(%s.get_depth(uint(%s))))", w, h, t, l);
                    if (kind == TEX_ARRAY) return string::tformat("int3(%s, %s, int(%s.get_array_size()))", w, h, t);
                    return string::tformat("int2(%s, %s)", w, h);
                case HLSL:
                    String z = string::tformat("ts%d", self.body.locals + 1);
                    self.line("uint %s_w, %s_h, %s_d, %s_l;", z, z, z, z);
                    if (vol) {
                        self.line("%s.GetDimensions(uint(%s), %s_w, %s_h, %s_d, %s_l);", t, l, z, z, z, z);
                        return string::tformat("int3(%s_w, %s_h, %s_d)", z, z, z);
                    }
                    self.line("%s.GetDimensions(uint(%s), %s_w, %s_h, %s_l);", t, l, z, z, z);
                    return string::tformat("int2(%s_w, %s_h)", z, z);
            }
        case COMPARE:
            String c = self.value_as({ d.args[0] }, VEC2);
            String z = self.value_as({ d.args[1] }, FLOAT);
            switch (self.lang) {
                case GLSL: return string::tformat("texture(%s, vec3(%s, %s))", ts, c, z);
                case HLSL: return string::tformat("%s.SampleCmpLevelZero(%s, %s, %s)", t, smp, c, z);
                case MSL: return string::tformat("%s.sample_compare(%s, %s, %s)", t, smp, c, z);
            }
    }
}

// A node material's texture (nodes::texture): its view NODE_USER_VIEW + k, with the material's
// sampler (shaders::SMP_tex_smp, 0).
fn String ProgramGen.user_texture(&self, Node n, NodeData* d) {
    NodeBuild* b = self.build;
    if (!b) abort("drie: nodes: a texture in a program without a material's build");
    int k = -1;
    for (int i = 0; i < b.texture_count; i++) {
        if (b.textures[i] == d.texture) k = i;
    }
    if (k < 0) {
        if (b.texture_count >= NODE_TEXTURES) abort("drie: nodes: more than %d textures", NODE_TEXTURES);
        k = b.texture_count++;
        b.textures[k] = d.texture;
    }
    NodeData s = { .op = SAMPLE, .type = VEC4, .kind = (char)NodeSampling.SAMPLE.ordinal, .args = { d.args[0], d.args[1], 0, 0 },
                   .res = { NODE_USER_VIEW + k, 0, NodeTextureKind.TEX_2D.ordinal } };
    if (d.kind == 1) {
        s.kind = (char)NodeSampling.SIZE.ordinal;
        s.type = IVEC2;
        s.args = { };
        return string::tformat("%s(%s)", self.type_name(VEC2), self.sample(n, &s));
    }
    if (d.args[1]) s.kind = (char)NodeSampling.LEVEL.ordinal;
    return self.sample(n, &s);
}

fn String ProgramGen.attribute(&self, NodeData* d) {
    if (self.stage != VERTEX) abort("drie: nodes: attributes are read in the vertex stage");
    if (self.body.func >= 0) abort("drie: nodes: attributes are read in main (a function takes them as arguments)");
    NodeType have = self.p.attrs[d.kind];
    if (have != VOID && have != d.type) abort("drie: nodes: attribute %d read as a %s and as a %s", d.kind, have, d.type);
    self.p.attrs[d.kind] = d.type;
    return self.lang == GLSL ? string::tformat("a%d", d.kind) : string::tformat("vin.a%d", d.kind);
}

fn String ProgramGen.system(&self, NodeData* d) {
    if (self.body.func >= 0) abort("drie: nodes: %s is read in main (a function takes it as an argument)", (NodeSystem)d.kind);
    switch ((NodeSystem)d.kind) {
        case INSTANCE_ID:
            if (self.stage != VERTEX) abort("drie: nodes: the instance index is the vertex stage's");
            self.instance_id = true;
            return self.lang == GLSL ? "uint(gl_InstanceID)" : (self.lang == MSL ? "iid" : "vin.iid");
        case VERTEX_ID:
            if (self.stage != VERTEX) abort("drie: nodes: the vertex index is the vertex stage's");
            self.vertex_id = true;
            return self.lang == GLSL ? "uint(gl_VertexID)" : (self.lang == MSL ? "vid" : "vin.vid");
        case FRAG_COORD:
            if (self.stage != FRAGMENT) abort("drie: nodes: the fragment's position is the fragment stage's");
            self.frag_coord = true;
            return self.lang == GLSL ? "gl_FragCoord" : "fin.fc";
        case FRONT_FACING:
            if (self.stage != FRAGMENT) abort("drie: nodes: facing is the fragment stage's");
            self.front_facing = true;
            return self.lang == GLSL ? "gl_FrontFacing" : (self.lang == MSL ? "ff" : "fin.ff");
    }
}

fn int ProgramGen.varying(&self, Node n) {
    ShaderProgram* p = self.p;
    foreach (i, v : p.varyings) {
        if (v.id == n.id) return (int)i;
    }
    if (self.fixed_varyings) abort("drie: nodes: a fragment stage reads a varying its shared vertex stage has not");
    p.varyings.push(n);
    return (int)p.varyings.len() - 1;
}

// A varying's type as carried: bools as uints.
fn NodeType carried(NodeType t) @local => t == BOOL ? UINT : t;

fn String ProgramGen.read_varying(&self, Node n) {
    if (self.body.func >= 0) abort("drie: nodes: varyings are read in main (a function takes them as arguments)");
    int k = self.varying(n);
    NodeType t = n.type();
    if (is_matrix(t) || t == VOID) abort("drie: nodes: a %s can't be a varying", t);
    String x = self.lang == GLSL ? string::tformat("v%d", k) : string::tformat("fin.v%d", k);
    return self.convert(x, carried(t), t);
}

fn String ProgramGen.builtin(&self, Node n, NodeBuiltin b) {
    if (self.body.func >= 0) abort("drie: nodes: %s is read in main, not in a function", b);
    Node bound = self.binding[b.ordinal];
    if (!bound.id) abort("drie: nodes: nothing is %s in this program's %s stage", b, self.stage);
    return self.emit(bound);
}

// The function `f` printed (once in the stage), and what Metal hands it.
fn FuncDone* ProgramGen.func(&self, int f) {
    FuncDone* done = &self.func_done[f];
    if (done.printed) return done;
    NodeFunc* fd = node_funcs.get_ref(f);
    GenBody* outer = self.body;
    GenBody* b = self.new_body(f);
    self.body = b;
    if (fd.ret == VOID) {
        self.emit(fd.body);
    } else {
        self.line("return %s;", self.value_as(fd.body, fd.ret));
    }
    self.body = outer;
    done.printed = true;
    done.block = b.block;
    done.views.init(tmem);
    done.samplers.init(tmem);
    foreach (v : b.views) done.views.push(v);
    foreach (s : b.samplers) done.samplers.push(s);
    DString sig = dstring::temp();
    if (self.lang == MSL) sig.append("static inline ");
    sig.appendf("%s %s(", self.type_name(fd.ret), fd.name);
    for (int i = 0; i < fd.count; i++) sig.appendf("%s%s p%d", i ? ", " : "", self.type_name(fd.params[i]), i);
    if (self.lang == MSL) {
        bool first = fd.count == 0;
        foreach (a : self.resource_args(done, true)) {
            sig.appendf("%s%s", first ? "" : ", ", a);
            first = false;
        }
    }
    sig.append(") {\n");
    self.funcs.append(sig.str_view());
    self.funcs.append(b.out.str_view());
    self.funcs.append("}\n\n");
    return done;
}

// Metal's arguments for what a function reads: declarations, or the names passed.
fn String[] ProgramGen.resource_args(&self, FuncDone* done, bool declare) {
    List{String} out;
    out.tinit();
    if (done.block) out.push(declare ? string::tformat("constant float4* %s", self.block_name()) : self.block_name());
    List{int} views;
    views.tinit();
    foreach (v : done.views) views.push(v);
    sort_ints(views.array_view());
    foreach (v : views) {
        ProgramView* pv = self.p.views.get_ref(v);
        String name = string::tformat(pv.storage ? "b%d" : "t%d", v);
        out.push(declare ? string::tformat("%s %s", msl_view_type(pv), name) : name);
    }
    List{int} smps;
    smps.tinit();
    foreach (s : done.samplers) smps.push(s);
    sort_ints(smps.array_view());
    foreach (s : smps) out.push(declare ? string::tformat("sampler s%d", s) : string::tformat("s%d", s));
    return out.to_tarray();
}

fn void sort_ints(int[] a) @local {
    for (sz i = 1; i < a.len; i++) {
        for (sz j = i; j > 0 && a[j - 1] > a[j]; j--) {
            int x = a[j];
            a[j] = a[j - 1];
            a[j - 1] = x;
        }
    }
}

fn String msl_view_type(ProgramView* v) @local {
    if (v.storage) {
        switch (v.elem) {
            case UVEC4: return "const device uint4*";
            case UINT: return "const device uint*";
            default: return "const device float4*";
        }
    }
    switch (v.kind) {
        case TEX_2D: return "texture2d<float>";
        case TEX_3D: return "texture3d<float>";
        case TEX_ARRAY: return "texture2d_array<float>";
        case TEX_DEPTH: return "depth2d<float>";
    }
}

fn String ProgramGen.call_fn(&self, Node n, NodeData* d) {
    int f = call_func(d);
    FuncDone* done = self.func(f);
    NodeFunc* fd = node_funcs.get_ref(f);
    DString c = dstring::temp();
    c.appendf("%s(", fd.name);
    for (int i = 0; i < d.count; i++) c.appendf("%s%s", i ? ", " : "", self.value_as({ call_arg(d, i) }, fd.params[i]));
    if (self.lang == MSL) {
        bool first = d.count == 0;
        foreach (a : self.resource_args(done, false)) {
            c.appendf("%s%s", first ? "" : ", ", a);
            first = false;
        }
    }
    c.append(")");
    // (what it reads, the caller hands on)
    GenBody* b = self.body;
    if (done.block) b.block = true;
    foreach (v : done.views) self.uses_view(v);
    foreach (s : done.samplers) {
        if (!b.samplers.contains(s)) b.samplers.push(s);
    }
    return c.str_view();
}

// Prints node `n` (and what it reads) into the body, once where it can be read again; its
// value is in name(n).
fn String ProgramGen.emit(&self, Node n) {
    if (self.printed(n)) return self.name(n);
    NodeData d = *node_data(n);
    String e;
    switch (d.op) {
        case CONST:
            // (written out where it is read)
            if (d.type == BOOL) return d.value[0] != 0 ? "true" : "false";
            if (components(d.type) == 1) return const_component(d.type, &d.value, 0);
            if (is_matrix(d.type)) abort("drie: nodes: no matrix constants (node_matrix of vectors)");
            e = string::tformat("%s(", self.type_name(d.type));
            for (int i = 0; i < components(d.type); i++) e = string::tformat("%s%s%s", e, i ? ", " : "", const_component(d.type, &d.value, i));
            return string::tformat("%s)", e);
        case VAR:
            String init = d.args[0] ? self.value_as({ d.args[0] }, d.type) : self.zero(d.type);
            String v = self.mark(n);
            self.line("%s %s = %s;", self.type_name(d.type), v, init);
            return v;
        case SEQ:
            self.statements(d.args[0]);
            if (!d.args[1]) {
                self.mark(n);
                return "";
            }
            e = self.emit({ d.args[1] });
        case ASSIGN:
        case IF:
        case LOOP:
        case JUMP:
        case EVAL:
        case BIND:
            abort("drie: nodes: a statement (%s) where a value goes", d.op);
        case UNIFORM:
            e = self.user_uniform(n);
        case BUILTIN:
            // (what it is bound to, printed once there)
            return self.builtin(n, (NodeBuiltin)d.kind);
        case TEXTURE:
            e = self.user_texture(n, &d);
        case VARYING:
            if (self.stage == FRAGMENT) {
                e = self.read_varying(n);
            } else {
                return self.emit({ d.args[0] });
            }
        case INPUT:
            e = self.input(&d);
        case STORAGE:
            e = self.storage(&d);
        case SAMPLE:
            e = self.sample(n, &d);
        case ATTRIBUTE:
            e = self.attribute(&d);
        case SYSTEM:
            e = self.system(&d);
        case PARAM:
            if (self.body.func != d.res[0]) abort("drie: nodes: a parameter of %s outside it", node_funcs[d.res[0]].name);
            return string::tformat("p%d", d.kind);
        case CALL_FN:
            e = self.call_fn(n, &d);
            if (d.type == VOID) {
                self.line("%s;", e);
                self.mark(n);
                return "";
            }
        case BINARY:
            e = self.binary(&d);
        case CALL:
            e = self.call(&d);
        case SELECT:
            e = string::tformat("(%s ? %s : %s)", self.value_as({ d.args[0] }, BOOL), self.value_as({ d.args[1] }, d.type),
                                self.value_as({ d.args[2] }, d.type));
        case SWIZZLE:
            Node a = { d.args[0] };
            String s = (String)d.swizzle[:d.count];
            if (components(a.type()) == 1) {
                e = self.value_as(a, d.type);
            } else {
                e = string::tformat("%s.%s", self.emit(a), s);
            }
        case JOIN:
            e = string::tformat("%s(", self.type_name(d.type));
            for (int i = 0; i < d.count; i++) {
                Node p = { d.args[i] };
                NodeType pt = is_matrix(d.type) ? p.type() : type_of(scalar_of(d.type), components(p.type()));
                e = string::tformat("%s%s%s", e, i ? ", " : "", self.value_as(p, pt));
            }
            e = string::tformat("%s)", e);
        case CONVERT:
            Node a = { d.args[0] };
            if (d.type == MAT3 && a.type() == MAT4) {
                String m = self.emit(a);
                switch (self.lang) {
                    case GLSL: e = string::tformat("mat3(%s)", m);
                    case HLSL: e = string::tformat("((float3x3)%s)", m);
                    case MSL: e = string::tformat("float3x3(%s[0].xyz, %s[1].xyz, %s[2].xyz)", m, m, m);
                }
            } else {
                e = self.value_as(a, d.type);
            }
        case NOT:
            if (d.kind == 1) {
                e = string::tformat("(~%s)", self.emit({ d.args[0] }));
            } else {
                e = string::tformat("(!%s)", self.value_as({ d.args[0] }, BOOL));
            }
    }
    String name = self.mark(n);
    self.line("%s %s = %s;", self.type_name(d.type), name, e);
    return name;
}

fn String ProgramGen.binary(&self, NodeData* d) {
    Node a = { d.args[0] };
    Node b = { d.args[1] };
    NodeBinary op = (NodeBinary)d.kind;
    String[*] ops = { "+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=", "&&", "||", "%", "&", "|", "^", "<<", ">>", "!=" };
    NodeType ta = a.type();
    NodeType tb = b.type();
    switch (op) {
        case MUL:
            if (is_matrix(ta) || is_matrix(tb)) {
                if (ta == MAT3 || tb == MAT3) {
                    NodeType o = ta == MAT3 ? tb : ta;
                    if (components(o) == 1) return string::tformat("(%s * %s)", self.value_as(a, ta == MAT3 ? MAT3 : FLOAT), self.value_as(b, tb == MAT3 ? MAT3 : FLOAT));
                    return self.mat_mul(self.emit(a), self.emit(b));
                }
                if (ta == MAT4 && tb == MAT4) return self.mat_mul(self.emit(a), self.emit(b));
                NodeType v = ta == MAT4 ? tb : ta;
                String x = ta == MAT4 ? self.emit(a) : self.value_as(a, components(v) == 1 ? FLOAT : VEC4);
                String y = tb == MAT4 ? self.emit(b) : self.value_as(b, components(v) == 1 ? FLOAT : VEC4);
                return components(v) == 1 ? string::tformat("(%s * %s)", x, y) : self.mat_mul(x, y);
            }
            nextcase ADD;
        case ADD:
        case SUB:
        case DIV:
        case REM:
        case BIT_AND:
        case BIT_OR:
        case BIT_XOR:
        case SHL:
        case SHR:
            NodeScalar s = scalar_of(d.type);
            String x = self.value_as(a, type_of(s, components(ta) == 1 ? 1 : components(d.type)));
            String y = self.value_as(b, type_of(s, components(tb) == 1 ? 1 : components(d.type)));
            if (op == REM && s == FLOAT) return string::tformat("(%s - %s * floor(%s / %s))", x, y, x, y);
            return string::tformat("(%s %s %s)", x, ops[op.ordinal], y);
        case LT:
        case LE:
        case GT:
        case GE:
        case EQ:
        case NE:
            NodeType t = wider(ta, tb);
            return string::tformat("(%s %s %s)", self.value_as(a, t), ops[op.ordinal], self.value_as(b, t));
        case AND:
        case OR:
        case XOR:
            return string::tformat("(%s %s %s)", self.value_as(a, BOOL), ops[op.ordinal], self.value_as(b, BOOL));
    }
}

fn String ProgramGen.call(&self, NodeData* d) {
    NodeFn f = (NodeFn)d.kind;
    NodeType t = d.type;
    Node a = { d.args[0] };
    Node b = { d.args[1] };
    Node c = { d.args[2] };
    switch (f) {
        case TRANSPOSE:
        case DETERMINANT:
            return string::tformat("%s(%s)", self.fn_name(f), self.emit(a));
        case LENGTH:
        case NORMALIZE:
            return string::tformat("%s(%s)", self.fn_name(f), self.value_as(a, type_of(FLOAT, components(a.type()))));
        case SATURATE:
            if (self.lang == GLSL) return string::tformat("clamp(%s, 0.0, 1.0)", self.value_as(a, t));
            return string::tformat("saturate(%s)", self.value_as(a, t));
        case MOD:
            // (floored, as GLSL's mod and TSL's: HLSL's and Metal's fmod truncate; of
            // integers, the remainder)
            String x = self.value_as(a, t);
            String y = self.value_as(b, t);
            if (scalar_of(t) != FLOAT) return string::tformat("(%s %% %s)", x, y);
            return string::tformat("(%s - %s * floor(%s / %s))", x, y, x, y);
        case DOT:
        case DISTANCE:
            NodeType v = type_of(FLOAT, components(wider(a.type(), b.type())));
            return string::tformat("%s(%s, %s)", self.fn_name(f), self.value_as(a, v), self.value_as(b, v));
        case CROSS:
            return string::tformat("cross(%s, %s)", self.value_as(a, VEC3), self.value_as(b, VEC3));
        case REFLECT:
            return string::tformat("reflect(%s, %s)", self.value_as(a, t), self.value_as(b, t));
        case REFRACT:
            return string::tformat("refract(%s, %s, %s)", self.value_as(a, t), self.value_as(b, t), self.value_as(c, FLOAT));
        case FACEFORWARD:
            return string::tformat("faceforward(%s, %s, %s)", self.value_as(a, t), self.value_as(b, t), self.value_as(c, t));
        case FWIDTH:
            if (self.lang == HLSL) return string::tformat("fwidth(%s)", self.value_as(a, t));
            nextcase default;
        default:
            if (d.count == 1) return string::tformat("%s(%s)", self.fn_name(f), self.value_as(a, t));
            if (d.count == 2) return string::tformat("%s(%s, %s)", self.fn_name(f), self.value_as(a, t), self.value_as(b, t));
            return string::tformat("%s(%s, %s, %s)", self.fn_name(f), self.value_as(a, t), self.value_as(b, t), self.value_as(c, t));
    }
}

// ---- the stages

fn ProgramGen* new_gen(ShaderProgram* p, ProgramStage st, NodeBuild* build, Node[NodeBuiltin::len]* bindings) @local {
    ProgramGen* g = mem::tnew(ProgramGen);
    g.lang = p.lang;
    g.stage = st;
    g.p = p;
    g.build = build;
    g.nodes = node_count();
    sz vars = (sz)g.nodes + 1 + NodeBuiltin::len;
    g.var_stamp = mem::temp_array(uint, vars);
    g.var_sets = mem::temp_array(uint[], (sz)g.nodes + 1);
    g.var_set_done = mem::temp_array(bool, (sz)g.nodes + 1);
    g.funcs = dstring::temp();
    g.func_done = mem::temp_array(FuncDone, node_funcs.is_initialized() ? node_funcs.len() + 1 : 1);
    if (bindings) g.binding = *bindings;
    g.body = g.new_body(-1);
    return g;
}

// The program of `graph` in `lang`, its sources and resources (not made yet: ShaderProgram.make).
// `build`: the node material whose uniforms and textures its graphs read. `vertex_bind` and
// `fragment_bind`: what the user graphs' builtins read in each stage at first.
fn ShaderProgram* build_program(NodeLang lang, ProgramGraph* graph, NodeBuild* build = null, Node[NodeBuiltin::len]* vertex_bind = null,
                                Node[NodeBuiltin::len]* fragment_bind = null) {
    ShaderProgram* p = mem::new(ShaderProgram);
    p.lang = lang;
    p.rows[0].init(mem);
    p.rows[1].init(mem);
    p.views.init(mem);
    p.samplers.init(mem);
    p.pairs.init(mem);
    p.varyings.init(mem);
    foreach (&t : p.attrs) *t = VOID;
    @pool() {
        // the fragment stage first: it finds the varyings
        ProgramGen* f = new_gen(p, FRAGMENT, build, fragment_bind);
        if (graph.vertex_of) {
            foreach (v : graph.vertex_of.varyings) p.varyings.push(v);
            f.fixed_varyings = true;
        }
        if (graph.fragment.id) f.statement(graph.fragment);
        String[4] colors;
        foreach (i, c : graph.colors) {
            if (!c.id) break;
            colors[i] = f.value_as(c, VEC4);
            p.colors = (int)i + 1;
        }
        for (int i = 0; i < p.colors; i++) {
            f.line(lang == GLSL ? "c%d = %s;" : "fout.c%d = %s;", i, colors[i]);
        }
        p.fs = f.fragment_source().copy(mem);
        if (graph.vertex_of) {
            p.vs = graph.vertex_of.vs.copy(mem);
            p.rows[0].free();
            p.rows[0].init(mem);
            foreach (r : graph.vertex_of.rows[0]) p.rows[0].push(r);
            p.block_rows[0] = graph.vertex_of.block_rows[0];
            p.attrs = graph.vertex_of.attrs;
            foreach (v : graph.vertex_of.views) {
                if (v.stage == VERTEX) p.views.push(v);
            }
            return p;
        }
        ProgramGen* v = new_gen(p, VERTEX, build, vertex_bind);
        String pos = v.value_as(graph.position, VEC4);
        foreach (i, n : p.varyings) {
            NodeData* d = node_data(n);
            Node src = d.op == VARYING ? (Node) { d.args[0] } : n;
            NodeType t = carried(n.type());
            v.line(lang == GLSL ? "v%d = %s;" : "vout.v%d = %s;", i, v.value_as(src, t));
        }
        v.line(lang == GLSL ? "gl_Position = %s;" : "vout.pos = %s;", pos);
        p.vs = v.vertex_source().copy(mem);
    };
    return p;
}

fn String varying_qualifier(NodeLang lang, NodeType t) @local {
    if (scalar_of(t) == FLOAT) return "";
    switch (lang) {
        case GLSL: return "flat ";
        case HLSL: return "nointerpolation ";
        case MSL: return "";
    }
}

fn String ProgramGen.msl_header(&self) {
    return "#include <metal_stdlib>\n#include <simd/simd.h>\n\nusing namespace metal;\n\n";
}

// The declarations of the stage's views and samplers (GLSL, HLSL), or Metal's entry arguments.
fn String ProgramGen.resource_decls(&self) {
    ShaderProgram* p = self.p;
    DString s = dstring::temp();
    foreach (i, v : p.views) {
        if (v.stage != self.stage) continue;
        switch (self.lang) {
            case MSL:
                s.appendf(", %s %s%d [[%s(%d)]]", msl_view_type(&v), v.storage ? "b" : "t", i, v.storage ? "buffer" : "texture", v.msl);
            case HLSL:
                if (v.storage) {
                    s.appendf("ByteAddressBuffer b%d : register(t%d);\n", i, v.hlsl);
                } else {
                    String[*] types = { "Texture2D<float4>", "Texture3D<float4>", "Texture2DArray<float4>", "Texture2D<float>" };
                    s.appendf("%s t%d : register(t%d);\n", types[v.kind.ordinal], i, v.hlsl);
                }
            case GLSL:
                if (v.storage) {
                    String[*] elems = { "vec4", "uvec4", "uint" };
                    int e = v.elem == VEC4 ? 0 : (v.elem == UVEC4 ? 1 : 2);
                    s.appendf("layout(binding = %d, std430) readonly buffer B%d { %s b%d[]; };\n", i, i, elems[e], i);
                }
        }
    }
    foreach (i, sm : p.samplers) {
        if (sm.stage != self.stage) continue;
        switch (self.lang) {
            case MSL: s.appendf(", sampler s%d [[sampler(%d)]]", i, sm.msl);
            case HLSL: s.appendf("%s s%d : register(s%d);\n", sm.compare ? "SamplerComparisonState" : "SamplerState", i, sm.msl);
            case GLSL: break;
        }
    }
    if (self.lang == GLSL) {
        foreach (i, pr : p.pairs) {
            if (pr.stage != self.stage) continue;
            ProgramView* v = p.views.get_ref(pr.view);
            String[*] types = { "sampler2D", "sampler3D", "sampler2DArray", "sampler2DShadow" };
            s.appendf("layout(binding = %d) uniform %s %s;\n", i, types[v.kind.ordinal], (ZString)pr.name);
        }
    }
    return s.str_view();
}

fn String ProgramGen.block_decl(&self) {
    int rows = self.p.block_rows[self.stage.ordinal];
    if (!rows) return "";
    String name = self.block_name();
    switch (self.lang) {
        case MSL: return string::tformat(", constant float4* %s [[buffer(%d)]]", name, self.stage.ordinal);
        case HLSL: return string::tformat("cbuffer %s_block : register(b0) {\n    float4 %s[%d];\n};\n", name, name, rows);
        case GLSL: return string::tformat("uniform vec4 %s[%d];\n", name, rows);
    }
}

fn String ProgramGen.vertex_source(&self) {
    ShaderProgram* p = self.p;
    DString s = dstring::temp();
    switch (self.lang) {
        case MSL:
            s.append(self.msl_header());
            bool attrs;
            s.append("struct VIn {\n");
            foreach (i, t : p.attrs) {
                if (t == VOID) continue;
                s.appendf("    %s a%d [[attribute(%d)]];\n", self.type_name(t), i, i);
                attrs = true;
            }
            s.append("};\n\nstruct VOut {\n");
            foreach (i, n : p.varyings) {
                NodeType t = carried(n.type());
                s.appendf("    %s v%d [[user(locn%d)%s]];\n", self.type_name(t), i, i, scalar_of(t) == FLOAT ? "" : ", flat");
            }
            s.append("    float4 pos [[position]];\n};\n\n");
            s.append(self.funcs.str_view());
            s.appendf("vertex VOut main0(%s", attrs ? "VIn vin [[stage_in]]" : "");
            String rest = string::tformat("%s%s%s%s", self.block_decl(), self.resource_decls(),
                                          self.instance_id ? ", uint iid [[instance_id]]" : "", self.vertex_id ? ", uint vid [[vertex_id]]" : "");
            s.append(attrs ? rest : (rest.len ? rest[2..] : ""));
            s.append(") {\n    VOut vout = {};\n");
            s.append(self.body.out.str_view());
            s.append("    return vout;\n}\n");
        case HLSL:
            s.append(self.block_decl());
            s.append(self.resource_decls());
            s.append("\nstruct VIn {\n");
            foreach (i, t : p.attrs) {
                if (t != VOID) s.appendf("    %s a%d : TEXCOORD%d;\n", self.type_name(t), i, i);
            }
            if (self.instance_id) s.append("    uint iid : SV_InstanceID;\n");
            if (self.vertex_id) s.append("    uint vid : SV_VertexID;\n");
            s.append("};\n\nstruct VOut {\n");
            foreach (i, n : p.varyings) {
                NodeType t = carried(n.type());
                s.appendf("    %s%s v%d : TEXCOORD%d;\n", varying_qualifier(HLSL, t), self.type_name(t), i, i);
            }
            s.append("    float4 pos : SV_Position;\n};\n\n");
            s.append(self.funcs.str_view());
            s.append("VOut main(VIn vin) {\n    VOut vout = (VOut)0;\n");
            s.append(self.body.out.str_view());
            s.append("    return vout;\n}\n");
        case GLSL:
            s.append("#version 430\n\n");
            foreach (i, t : p.attrs) {
                if (t != VOID) s.appendf("layout(location = %d) in %s a%d;\n", i, self.type_name(t), i);
            }
            foreach (i, n : p.varyings) {
                NodeType t = carried(n.type());
                s.appendf("layout(location = %d) %sout %s v%d;\n", i, varying_qualifier(GLSL, t), self.type_name(t), i);
            }
            s.append(self.block_decl());
            s.append(self.resource_decls());
            s.append("\n");
            s.append(self.funcs.str_view());
            s.append("void main() {\n");
            s.append(self.body.out.str_view());
            s.append("}\n");
    }
    return s.str_view();
}

fn String ProgramGen.fragment_source(&self) {
    ShaderProgram* p = self.p;
    DString s = dstring::temp();
    bool input = p.varyings.len() > 0 || self.frag_coord;
    switch (self.lang) {
        case MSL:
            s.append(self.msl_header());
            if (input) {
                s.append("struct FIn {\n");
                foreach (i, n : p.varyings) {
                    NodeType t = carried(n.type());
                    s.appendf("    %s v%d [[user(locn%d)%s]];\n", self.type_name(t), i, i, scalar_of(t) == FLOAT ? "" : ", flat");
                }
                if (self.frag_coord) s.append("    float4 fc [[position]];\n");
                s.append("};\n\n");
            }
            if (p.colors) {
                s.append("struct FOut {\n");
                for (int i = 0; i < p.colors; i++) s.appendf("    float4 c%d [[color(%d)]];\n", i, i);
                s.append("};\n\n");
            }
            s.append(self.funcs.str_view());
            s.appendf("fragment %s main0(%s", p.colors ? "FOut" : "void", input ? "FIn fin [[stage_in]]" : "");
            String rest = string::tformat("%s%s%s", self.block_decl(), self.resource_decls(), self.front_facing ? ", bool ff [[front_facing]]" : "");
            s.append(input ? rest : (rest.len ? rest[2..] : ""));
            s.append(") {\n");
            if (p.colors) s.append("    FOut fout = {};\n");
            s.append(self.body.out.str_view());
            if (p.colors) s.append("    return fout;\n");
            s.append("}\n");
        case HLSL:
            s.append(self.block_decl());
            s.append(self.resource_decls());
            if (input || self.front_facing) s.append("\nstruct FIn {\n");
            foreach (i, n : p.varyings) {
                NodeType t = carried(n.type());
                s.appendf("    %s%s v%d : TEXCOORD%d;\n", varying_qualifier(HLSL, t), self.type_name(t), i, i);
            }
            if (self.frag_coord) s.append("    float4 fc : SV_Position;\n");
            if (self.front_facing) s.append("    bool ff : SV_IsFrontFace;\n");
            if (input || self.front_facing) s.append("};\n\n");
            if (p.colors) {
                s.append("struct FOut {\n");
                for (int i = 0; i < p.colors; i++) s.appendf("    float4 c%d : SV_Target%d;\n", i, i);
                s.append("};\n\n");
            }
            s.append(self.funcs.str_view());
            s.appendf("%s main(%s) {\n", p.colors ? "FOut" : "void", input || self.front_facing ? "FIn fin" : "");
            if (p.colors) s.append("    FOut fout = (FOut)0;\n");
            s.append(self.body.out.str_view());
            if (p.colors) s.append("    return fout;\n");
            s.append("}\n");
        case GLSL:
            s.append("#version 430\n\n");
            foreach (i, n : p.varyings) {
                NodeType t = carried(n.type());
                s.appendf("layout(location = %d) %sin %s v%d;\n", i, varying_qualifier(GLSL, t), self.type_name(t), i);
            }
            for (int i = 0; i < p.colors; i++) s.appendf("layout(location = %d) out vec4 c%d;\n", i, i);
            s.append(self.block_decl());
            s.append(self.resource_decls());
            s.append("\n");
            s.append(self.funcs.str_view());
            s.append("void main() {\n");
            s.append(self.body.out.str_view());
            s.append("}\n");
    }
    return s.str_view();
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
            b.glsl_binding_n = (char)i;
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
    return d;
}

// The program made (once): its sources are printed if the backend's compiler takes them not.
fn SgShader ShaderProgram.make(&self) {
    if (self.shader.id) return self.shader;
    SgShaderDesc d = self.desc();
    self.shader = sg::make_shader(&d);
    if (sg::query_shader_state(self.shader) != VALID) {
        io::eprintfn("drie: a generated program failed to build.\n--- vertex:\n%s\n--- fragment:\n%s", self.vs, self.fs);
        abort("drie: a generated program failed to build");
    }
    return self.shader;
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
