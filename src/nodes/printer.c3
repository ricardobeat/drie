// Shader source emission for node programs: resource slot allocation, expression printing, and
// Metal/HLSL/GLSL stage source assembly (`build_program`, `vertex_source`, `fragment_source`).
//
// `build_program` emits the fragment stage before the vertex stage so `p.varyings` captures
// every interpolated value needed. Metal assigns textures `texture(0..)` and storage buffers
// `buffer(8..)` and passes resources explicitly into functions; HLSL assigns per-stage `t`/`s`
// registers, reads storage via `ByteAddressBuffer.Load*`, and constructs matrices by row
// (`mul(b, a)`); GLSL emits `#version 430` combined samplers and `std430` buffers. All three mark
// clip position invariant for depth-prepass `EQUAL` tests. Extends `ProgramGen` (src/nodes/program.c3).
module drie;
import std::collections::list, std::collections::map, std::io, std::thread, sokol;

// ---- Resource allocation and expression printing

// Allocates or reuses a texture or readonly storage buffer view slot for the active stage.
fn int ProgramGen.view(&self, int source, bool storage, NodeTextureKind kind, NodeType elem) {
    ShaderProgram* p = self.p;
    int msl;
    int hlsl;
    foreach (i, v : p.views) {
        if (v.stage != self.stage) continue;
        if (v.source == source && v.storage == storage) {
            if (!storage && v.kind != kind) abort(
                "drie: nodes: view %d read as a %s and as a %s",
                source,
                v.kind,
                kind
            );
            if (storage && v.elem != elem) abort(
                "drie: nodes: storage buffer %d read as %ss and as %ss",
                source,
                v.elem,
                elem
            );
            self.uses_view((int)i);
            return (int)i;
        }
        if (v.storage == storage) msl++;
        hlsl++;
    }
    p.views.push(
        {
            .stage = self.stage,
            .source = source,
            .storage = storage,
            .kind = kind,
            .elem = elem,
            .msl = storage ? 8 + msl : msl,
            .hlsl = hlsl,
            .glsl = (int)p.views.len()
        }
    );
    int at = (int)p.views.len() - 1;
    self.uses_view(at);
    return at;
}

// Records that the active body references view index `v` (for Metal helper parameter passing).
fn void ProgramGen.uses_view(&self, int v) {
    if (!self.body.views.contains(v)) self.body.views.push(v);
}

// Allocates or reuses a filtering or comparison sampler slot for the active stage.
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
    int st           = self.stage.ordinal;
    foreach (r : p.rows[st]) {
        if (r.source == source && r.row == row && r.count == count) return r.at;
    }
    int at = p.block_rows[st];
    p.rows[st].push({ .source = source, .row = row, .count = count, .at = at });
    p.block_rows[st] += count;
    return at;
}

// Returns the stage uniform-array identifier (`"vu"` for vertex, `"fu"` for fragment).
fn String ProgramGen.block_name(&self) => self.stage == VERTEX ? "vu" : "fu";

// Formats a `NodeOp.INPUT` read from the active stage's packed uniform block (`vu` or `fu`).
fn String ProgramGen.input(&self, NodeData* d) {
    self.body.block = true;
    NodeType t   = d.type;
    int rows     = t == MAT4 ? 4 : (t == MAT3 ? 3 : 1);
    int stride   = (int)bitcast(d.value[0], uint);
    int len      = (int)bitcast(d.value[1], uint);
    bool indexed = d.args[0] != 0;
    int at       = self.rows((NodeSource)d.res[0], d.res[1], indexed ? stride * (len - 1) + rows : rows);
    String base  = self.block_name();
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

// A node material's uniform (nodes::uniform): a row of its USER block, in its NodeBuild's order
// after the block's header (NODE_USER_HEADER).
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
    NodeData d = {
        .op = INPUT,
        .type = t == MAT4 ? MAT4 : type_of(FLOAT, components(t)),
        .res = { NodeSource.USER.ordinal, NODE_USER_HEADER + at, 0 }
    };
    String x = self.input(&d);
    // (integers and bools as floats: Node.set)
    return t == MAT4 ? x : self.convert(x, d.type, t);
}

// Formats a readonly storage buffer element load (indexing in GLSL/MSL, `Load`/`Load4` in HLSL).
fn String ProgramGen.storage(&self, NodeData* d) {
    int v    = self.view(d.res[0], true, TEX_2D, d.type);
    String i = self.value_as({ d.args[0] }, d.type == UINT ? UINT : INT);
    if (self.lang != HLSL) return string::tformat("b%d[%s]", v, i);
    switch (d.type) {
        case VEC4: return string::tformat("asfloat(b%d.Load4(uint(%s) * 16u))", v, i);
        case UVEC4: return string::tformat("b%d.Load4(uint(%s) * 16u)", v, i);
        default: return string::tformat("b%d.Load(uint(%s) * 4u)", v, i);
    }
}

// Formats a texture sample, level, grad, texel fetch, size, mip-count, or depth-compare expression.
fn String ProgramGen.sample(&self, Node n, NodeData* d) {
    NodeSampling how     = (NodeSampling)d.kind;
    NodeTextureKind kind = (NodeTextureKind)d.res[2];
    int v                = self.view(d.res[0], false, kind, VOID);
    int s                = self.sampler(d.res[1], kind == TEX_DEPTH);
    String t             = string::tformat("t%d", v);
    String smp           = string::tformat("s%d", s);
    String ts            = self.lang == GLSL ? self.pair(v, s) : "";
    bool vol             = kind == TEX_3D || kind == TEX_ARRAY;
    NodeType ct          = vol ? VEC3 : VEC2;
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
                    if (self.lang == MSL) extra = string::tformat(
                        ", gradient%s(%s, %s)",
                        kind == TEX_3D ? "3d" : "2d",
                        dx,
                        dy
                    );
                    if (self.lang == GLSL) return string::tformat("textureGrad(%s, %s, %s, %s)", ts, c, dx, dy);
                    if (self.lang == HLSL) return string::tformat("%s.SampleGrad(%s, %s, %s, %s)", t, smp, c, dx, dy);
                default:
                    if (self.lang == GLSL) return string::tformat("texture(%s, %s)", ts, c);
                    if (self.lang == HLSL) return string::tformat("%s.Sample(%s, %s)", t, smp, c);
            }
            if (kind == TEX_ARRAY) return string::tformat(
                "%s.sample(%s, %s.xy, uint(rint(%s.z))%s)",
                t,
                smp,
                c,
                c,
                extra
            );
            return string::tformat("%s.sample(%s, %s%s)", t, smp, c, extra);
        case FETCH:
            String c = self.value_as({ d.args[0] }, vol ? IVEC3 : IVEC2);
            String l = d.args[1] ? self.value_as({ d.args[1] }, INT) : "0";
            switch (self.lang) {
                case GLSL: return string::tformat("texelFetch(%s, %s, %s)", ts, c, l);
                case HLSL: return string::tformat("%s.Load(%s(%s, %s))", t, vol ? "int4" : "int3", c, l);
                case MSL:
                    if (kind == TEX_ARRAY) return string::tformat(
                        "%s.read(uint2(%s.xy), uint(%s.z), uint(%s))",
                        t,
                        c,
                        c,
                        l
                    );
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
        case LEVELS:
            switch (self.lang) {
                case GLSL: return string::tformat("textureQueryLevels(%s)", ts);
                case MSL: return string::tformat("int(%s.get_num_mip_levels())", t);
                case HLSL:
                    String z = string::tformat("tl%d", self.body.locals + 1);
                    self.line("uint %s_w, %s_h, %s_l;", z, z, z);
                    self.line("%s.GetDimensions(0u, %s_w, %s_h, %s_l);", t, z, z, z);
                    return string::tformat("int(%s_l)", z);
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
    NodeData s = {
        .op = SAMPLE,
        .type = VEC4,
        .kind = (char)NodeSampling.SAMPLE.ordinal,
        .args = { d.args[0], d.args[1], 0, 0 },
        .res = { NODE_USER_VIEW + k, 0, NodeTextureKind.TEX_2D.ordinal }
    };
    if (d.kind == 1) {
        s.kind = (char)NodeSampling.SIZE.ordinal;
        s.type = IVEC2;
        s.args = { };
        return string::tformat("%s(%s)", self.type_name(VEC2), self.sample(n, &s));
    }
    if (d.kind == 2) {
        s.kind = (char)NodeSampling.GRAD.ordinal;
        s.args[2] = d.args[2];
    } else if (d.args[1]) {
        s.kind = (char)NodeSampling.LEVEL.ordinal;
    }
    return self.sample(n, &s);
}

// Records vertex attribute read `d.kind` and returns its input expression (`aN` or `vin.aN`).
fn String ProgramGen.attribute(&self, NodeData* d) {
    if (self.stage != VERTEX) abort("drie: nodes: attributes are read in the vertex stage");
    if (self.body.func >= 0) abort("drie: nodes: attributes are read in main (a function takes them as arguments)");
    NodeType have = self.p.attrs[d.kind];
    if (have != VOID && have != d.type) abort(
        "drie: nodes: attribute %d read as a %s and as a %s",
        d.kind,
        have,
        d.type
    );
    self.p.attrs[d.kind] = d.type;
    return self.lang == GLSL ? string::tformat("a%d", d.kind) : string::tformat("vin.a%d", d.kind);
}

// Records a stage system value and returns its backend expression (`gl_FragCoord`, `fin.fc`, etc.).
fn String ProgramGen.system(&self, NodeData* d) {
    if (self.body.func >= 0) abort(
        "drie: nodes: %s is read in main (a function takes it as an argument)",
        (NodeSystem)d.kind
    );
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

// Registers or looks up node `n` in `ShaderProgram.varyings`, returning its varying index.
fn int ProgramGen.varying(&self, Node n) {
    ShaderProgram* p = self.p;
    foreach (i, v : p.varyings) {
        if (v.id == n.id) return (int)i;
    }
    p.varyings.push(n);
    return (int)p.varyings.len() - 1;
}

// A varying's type as carried: bools as uints.
fn NodeType carried(NodeType t) @local => t == BOOL ? UINT : t;

// Registers varying `n` and formats its fragment-stage input expression (`vN` or `fin.vN`).
fn String ProgramGen.read_varying(&self, Node n) {
    if (self.body.func >= 0) abort("drie: nodes: varyings are read in main (a function takes them as arguments)");
    int k      = self.varying(n);
    NodeType t = n.type();
    if (is_matrix(t) || t == VOID) abort("drie: nodes: a %s can't be a varying", t);
    String x = self.lang == GLSL ? string::tformat("v%d", k) : string::tformat("fin.v%d", k);
    return self.convert(x, carried(t), t);
}

// Resolves stage builtin `b` to its currently bound node and emits that node.
fn String ProgramGen.builtin(&self, Node n, NodeBuiltin b) {
    if (self.body.func >= 0) abort("drie: nodes: %s is read in main, not in a function", b);
    Node bound = self.binding[b.ordinal];
    if (!bound.id) {
        switch (b) {
            case MATERIAL_COLOR: abort("drie: nodes: material_color() is for color_node and opacity_node");
            case MATERIAL_ROUGHNESS:
            case MATERIAL_METALNESS: abort("drie: nodes: material_roughness() and material_metalness() are for roughness_node, metalness_node and the surface step's nodes");
            case ENV_COLOR: abort("drie: nodes: env_color() is for env_node");
            case INSTANCE_MATRIX: abort("drie: nodes: instance_matrix() is for the vertex stage (position_node)");
            case OUTPUT: abort("drie: nodes: output() is for output_node");
            case ENV_DIRECTION:
            case ENV_ROUGHNESS: abort("drie: nodes: env_direction() and env_roughness() are for env_node");
            default: abort("drie: nodes: nothing is %s in this program's %s stage", b, self.stage);
        }
    }
    return self.emit(bound);
}

// The function `f` printed (once in the stage), and what Metal hands it.
fn FuncDone* ProgramGen.func(&self, int f) {
    FuncDone* done = &self.func_done[f];
    if (done.printed) return done;
    NodeFunc* fd = node_funcs.get_ref(f);
    if (done.printing) abort("drie: nodes: %s calls itself", fd.name);
    foreach (i, &other : self.func_done) {
        if (other.printed && i < node_funcs.len() && node_funcs[i].name == fd.name) abort(
            "drie: nodes: two functions named %s in one program",
            fd.name
        );
    }
    done.printing = true;
    GenBody* outer = self.body;
    GenBody* b     = self.new_body(f);
    self.body = b;
    if (fd.ret == VOID) {
        self.emit(fd.body);
    } else {
        self.line("return %s;", self.value_as(fd.body, fd.ret));
    }
    self.body = outer;
    done.printing = false;
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
        String name     = string::tformat(pv.storage ? "b%d" : "t%d", v);
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

// Emits the callee definition if needed and formats the call (with Metal resource args).
fn String ProgramGen.call_fn(&self, Node n, NodeData* d) {
    int f          = call_func(d);
    FuncDone* done = self.func(f);
    NodeFunc* fd   = node_funcs.get_ref(f);
    DString c      = dstring::temp();
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
            for (int i = 0; i < components(
                d.type
            ); i++) e = string::tformat("%s%s%s", e, i ? ", " : "", const_component(d.type, &d.value, i));
            return string::tformat("%s)", e);
        case VAR:
            String init = d.args[0] ? self.value_as({ d.args[0] }, d.type) : self.zero(d.type);
            String v = self.mark(n);
            self.line("%s %s = %s;", self.type_name(d.type), v, init);
            return v;
        case SEQ:
            // (printed again where its variables are still declared: they start again from
            // their first values)
            bool again = self.body.serial[n.id] > 0;
            for (uint st = d.args[0]; st; st = node_data({ st }).next) {
                NodeData* sd = node_data({ st });
                Node v       = sd.op == EVAL ? (Node) { sd.args[0] } : (Node) { st };
                NodeData* vd = node_data(v);
                if (again && vd.op == VAR && self.printed(v)) {
                    self.line(
                        "%s = %s;",
                        self.name(v),
                        vd.args[0] ? self.value_as({ vd.args[0] }, vd.type) : self.zero(vd.type)
                    );
                    self.assigned(v.id);
                    continue;
                }
                self.statement({ st });
            }
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
            if (self.body.func != d.res[0]) abort(
                "drie: nodes: a parameter of %s outside it",
                node_funcs[d.res[0]].name
            );
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
            e = string::tformat(
                "(%s ? %s : %s)",
                self.value_as({ d.args[0] }, BOOL),
                self.value_as({ d.args[1] }, d.type),
                self.value_as({ d.args[2] }, d.type)
            );
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
                Node p      = { d.args[i] };
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
    // (a value printed already for another node, where it can still be read: its name; what
    // reads a variable has the same variables as that node, and the same assignments since)
    String key   = string::tformat("%s %s", self.type_name(d.type), e);
    GenBody* b   = self.body;
    uint[2]? had = d.op == CALL_FN && !node_funcs[call_func(&d)].pure ? NOT_FOUND~ : b.exprs[key];
    if (try had) {
        Node m = { had[0] };
        if (b.serial[m.id] == had[1] && self.printed(m)) {
            String same = self.name(m);
            b.memo[n.id] = b.memo[m.id];
            b.memo_stamp[n.id] = self.stamp;
            b.same_as[n.id] = same;
            return same;
        }
    }
    String name = self.mark(n);
    self.line("%s %s = %s;", self.type_name(d.type), name, e);
    b.exprs[key] = { n.id, b.serial[n.id] };
    return name;
}

// Formats a `NodeOp.BINARY` expression (handling matrix multiplies and float floored remainder).
fn String ProgramGen.binary(&self, NodeData* d) {
    Node a        = { d.args[0] };
    Node b        = { d.args[1] };
    NodeBinary op = (NodeBinary)d.kind;
    String[*] ops = {
        "+",
        "-",
        "*",
        "/",
        "<",
        "<=",
        ">",
        ">=",
        "==",
        "!=",
        "&&",
        "||",
        "%",
        "&",
        "|",
        "^",
        "<<",
        ">>",
        "!="
    };
    NodeType ta = a.type();
    NodeType tb = b.type();
    switch (op) {
        case MUL:
            if (is_matrix(ta) || is_matrix(tb)) {
                if (ta == MAT3 || tb == MAT3) {
                    NodeType o = ta == MAT3 ? tb : ta;
                    if (components(
                        o
                    ) == 1) return string::tformat(
                        "(%s * %s)",
                        self.value_as(a, ta == MAT3 ? MAT3 : FLOAT),
                        self.value_as(b, tb == MAT3 ? MAT3 : FLOAT)
                    );
                    return self.mat_mul(self.emit(a), self.emit(b));
                }
                if (ta == MAT4 && tb == MAT4) return self.mat_mul(self.emit(a), self.emit(b));
                NodeType v = ta == MAT4 ? tb : ta;
                String x   = ta == MAT4 ? self.emit(a) : self.value_as(a, components(v) == 1 ? FLOAT : VEC4);
                String y   = tb == MAT4 ? self.emit(b) : self.value_as(b, components(v) == 1 ? FLOAT : VEC4);
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

// Formats a `NodeOp.CALL` built-in intrinsic expression for GLSL, MSL, or HLSL.
fn String ProgramGen.call(&self, NodeData* d) {
    NodeFn f   = (NodeFn)d.kind;
    NodeType t = d.type;
    Node a     = { d.args[0] };
    Node b     = { d.args[1] };
    Node c     = { d.args[2] };
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
            return string::tformat(
                "refract(%s, %s, %s)",
                self.value_as(a, t),
                self.value_as(b, t),
                self.value_as(c, FLOAT)
            );
        case FACEFORWARD:
            return string::tformat(
                "faceforward(%s, %s, %s)",
                self.value_as(a, t),
                self.value_as(b, t),
                self.value_as(c, t)
            );
        case BITS_AS_UINT:
            if (self.lang == GLSL) return string::tformat(
                "floatBitsToUint(%s)",
                self.value_as(a, type_of(FLOAT, components(t)))
            );
            if (self.lang == MSL) return string::tformat(
                "as_type<%s>(%s)",
                self.type_name(t),
                self.value_as(a, type_of(FLOAT, components(t)))
            );
            return string::tformat("asuint(%s)", self.value_as(a, type_of(FLOAT, components(t))));
        case BITS_AS_FLOAT:
            if (self.lang == GLSL) return string::tformat(
                "uintBitsToFloat(%s)",
                self.value_as(a, type_of(UINT, components(t)))
            );
            if (self.lang == MSL) return string::tformat(
                "as_type<%s>(%s)",
                self.type_name(t),
                self.value_as(a, type_of(UINT, components(t)))
            );
            return string::tformat("asfloat(%s)", self.value_as(a, type_of(UINT, components(t))));
        case FWIDTH:
            if (self.lang == HLSL) return string::tformat("fwidth(%s)", self.value_as(a, t));
            nextcase default;
        default:
            if (d.count == 1) return string::tformat("%s(%s)", self.fn_name(f), self.value_as(a, t));
            if (d.count == 2) return string::tformat(
                "%s(%s, %s)",
                self.fn_name(f),
                self.value_as(a, t),
                self.value_as(b, t)
            );
            return string::tformat(
                "%s(%s, %s, %s)",
                self.fn_name(f),
                self.value_as(a, t),
                self.value_as(b, t),
                self.value_as(c, t)
            );
    }
}

// ---- Stage source assembly

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
fn ShaderProgram* build_program(
    NodeLang lang,
    ProgramGraph* graph,
    NodeBuild* build = null,
    Node[NodeBuiltin::len]* vertex_bind = null,
    Node[NodeBuiltin::len]* fragment_bind = null
) {
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
        if (graph.varyings_from) {
            foreach (v : graph.varyings_from.varyings) p.varyings.push(v);
        }
        if (graph.fragment.id) f.statements(graph.fragment.id);
        String[4] colors;
        foreach (i, c : graph.colors) {
            if (!c.id) break;
            colors[i] = f.value_as(c, VEC4);
            p.colors = (int)i + 1;
        }
        for (int i = 0; i < p.colors; i++) {
            f.line(lang == GLSL ? "c%d = %s;" : "fout.c%d = %s;", i, colors[i]);
        }
        if (graph.depth.id) {
            String depth = f.value_as(graph.depth, FLOAT);
            p.depth_out = true;
            f.line(lang == GLSL ? "gl_FragDepth = %s;" : "fout.depth = %s;", depth);
        }
        p.fs = f.fragment_source().copy(mem);
        if (graph.fragment_only) return p;
        ProgramGen* v = new_gen(p, VERTEX, build, vertex_bind);
        String pos    = v.value_as(graph.position, VEC4);
        foreach (i, n : p.varyings) {
            NodeData* d = node_data(n);
            Node src    = d.op == VARYING ? (Node) { d.args[0] } : n;
            NodeType t  = carried(n.type());
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

// Returns the Metal shader preamble (`<metal_stdlib>`, `<simd/simd.h>`, `using namespace metal;`).
fn String ProgramGen.msl_header(&self) {
    return "#include <metal_stdlib>\n#include <simd/simd.h>\n\nusing namespace metal;\n\n";
}

// The declarations of the stage's views and samplers (GLSL, HLSL), or Metal's entry arguments.
fn String ProgramGen.resource_decls(&self) {
    ShaderProgram* p = self.p;
    DString s        = dstring::temp();
    foreach (i, v : p.views) {
        if (v.stage != self.stage) continue;
        switch (self.lang) {
            case MSL:
                s.appendf(
                    ", %s %s%d [[%s(%d)]]",
                    msl_view_type(&v),
                    v.storage ? "b" : "t",
                    i,
                    v.storage ? "buffer" : "texture",
                    v.msl
                );
            case HLSL:
                if (v.storage) {
                    s.appendf("ByteAddressBuffer b%d : register(t%d);\n", i, v.hlsl);
                } else {
                    String[*] types = {
                        "Texture2D<float4>",
                        "Texture3D<float4>",
                        "Texture2DArray<float4>",
                        "Texture2D<float>"
                    };
                    s.appendf("%s t%d : register(t%d);\n", types[v.kind.ordinal], i, v.hlsl);
                }
            case GLSL:
                if (v.storage) {
                    String[*] elems = { "vec4", "uvec4", "uint" };
                    int e           = v.elem == VEC4 ? 0 : (v.elem == UVEC4 ? 1 : 2);
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
            ProgramView* v  = p.views.get_ref(pr.view);
            String[*] types = { "sampler2D", "sampler3D", "sampler2DArray", "sampler2DShadow" };
            s.appendf("layout(binding = %d) uniform %s %s;\n", i, types[v.kind.ordinal], (ZString)pr.name);
        }
    }
    return s.str_view();
}

// Formats the active stage's packed uniform block declaration (empty if `block_rows == 0`).
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

// Assembles the complete vertex shader source in MSL, HLSL, or GLSL from the emitted body.
fn String ProgramGen.vertex_source(&self) {
    ShaderProgram* p = self.p;
    DString s        = dstring::temp();
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
                s.appendf(
                    "    %s v%d [[user(locn%d)%s]];\n",
                    self.type_name(t),
                    i,
                    i,
                    scalar_of(t) == FLOAT ? "" : ", flat"
                );
            }
            // (invariant: the depth prepass's pipeline and the EQUAL-tested one after it compute
            // the same depth from the same vertex function)
            s.append("    float4 pos [[position, invariant]];\n};\n\n");
            s.append(self.funcs.str_view());
            s.appendf("vertex VOut main0(%s", attrs ? "VIn vin [[stage_in]]" : "");
            String rest = string::tformat(
                "%s%s%s%s",
                self.block_decl(),
                self.resource_decls(),
                self.instance_id ? ", uint iid [[instance_id]]" : "",
                self.vertex_id ? ", uint vid [[vertex_id]]" : ""
            );
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
            // (precise: as Metal's invariant position)
            s.append("    precise float4 pos : SV_Position;\n};\n\n");
            s.append(self.funcs.str_view());
            s.append("VOut main(VIn vin) {\n    VOut vout = (VOut)0;\n");
            s.append(self.body.out.str_view());
            s.append("    return vout;\n}\n");
        case GLSL:
            s.append("#version 430\n\ninvariant gl_Position;\n");
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

// Assembles the complete fragment shader source in MSL, HLSL, or GLSL from the emitted body.
fn String ProgramGen.fragment_source(&self) {
    ShaderProgram* p = self.p;
    DString s        = dstring::temp();
    bool input       = p.varyings.len() > 0 || self.frag_coord;
    switch (self.lang) {
        case MSL:
            s.append(self.msl_header());
            if (input) {
                s.append("struct FIn {\n");
                foreach (i, n : p.varyings) {
                    NodeType t = carried(n.type());
                    s.appendf(
                        "    %s v%d [[user(locn%d)%s]];\n",
                        self.type_name(t),
                        i,
                        i,
                        scalar_of(t) == FLOAT ? "" : ", flat"
                    );
                }
                if (self.frag_coord) s.append("    float4 fc [[position]];\n");
                s.append("};\n\n");
            }
            bool fout = p.colors || p.depth_out;
            if (fout) {
                s.append("struct FOut {\n");
                for (int i = 0; i < p.colors; i++) s.appendf("    float4 c%d [[color(%d)]];\n", i, i);
                if (p.depth_out) s.append("    float depth [[depth(any)]];\n");
                s.append("};\n\n");
            }
            s.append(self.funcs.str_view());
            s.appendf("fragment %s main0(%s", fout ? "FOut" : "void", input ? "FIn fin [[stage_in]]" : "");
            String rest = string::tformat(
                "%s%s%s",
                self.block_decl(),
                self.resource_decls(),
                self.front_facing ? ", bool ff [[front_facing]]" : ""
            );
            s.append(input ? rest : (rest.len ? rest[2..] : ""));
            s.append(") {\n");
            if (fout) s.append("    FOut fout = {};\n");
            s.append(self.body.out.str_view());
            if (fout) s.append("    return fout;\n");
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
            bool fout = p.colors || p.depth_out;
            if (fout) {
                s.append("struct FOut {\n");
                for (int i = 0; i < p.colors; i++) s.appendf("    float4 c%d : SV_Target%d;\n", i, i);
                if (p.depth_out) s.append("    float depth : SV_Depth;\n");
                s.append("};\n\n");
            }
            s.append(self.funcs.str_view());
            s.appendf("%s main(%s) {\n", fout ? "FOut" : "void", input || self.front_facing ? "FIn fin" : "");
            if (fout) s.append("    FOut fout = (FOut)0;\n");
            s.append(self.body.out.str_view());
            if (fout) s.append("    return fout;\n");
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