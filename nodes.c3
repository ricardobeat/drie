// Node materials: a material's color, opacity, normal, roughness, metalness, glow, vertex
// positions and final color worked out by graphs of nodes made at runtime, like three's TSL
// and NodeMaterial (colorNode, positionNode, outputNode...):
//
//   Node wave = nodes::sin(nodes::position_local().y() * 8 + nodes::time() * 2);
//   mat.color_node = nodes::mix(nodes::color(0x2266ff), nodes::color(0xffffff), wave * 0.5f + 0.5f);
//   mat.position_node = nodes::position_local() + nodes::normal_local() * (wave * 0.05f);
//
// The renderer prints a material's graphs in the backend's own shading language (Metal, HLSL
// or GLSL), the way three hands TSL to the browser as WGSL or GLSL: drie carries no shader
// compiler. They go into the material's generated programs (materials_nodes.c3), where the
// material's own steps would work out what they replace; with an application's material
// shader (Renderer.shaders) they are swapped into its hooks instead
// (node_vertex, node_color, node_normal, node_surface and node_output, see node_common in
// shader_core.glsl). A material's programs are built the first time it is drawn, and again
// when one of its *_node fields changes. A uniform (nodes::uniform) changes with Node.set,
// without a rebuild.
//
// The material's own settings still apply where no node replaces them (its maps, clear coat,
// transmission...). Node materials work in every pass: the main one, the depth prepass and
// the shadow maps, which see position_node, and the holes color_node and opacity_node cut (a
// discard, or alpha under the material's alpha_test). With the material shader, textures are
// read in the fragment stage, at most six with the material's other maps (see
// Material.ao_map...), and the graphs can use 32 vec4s of uniforms and four vec4s of
// varyings.
//
// Graphs hold floats, ints, uints, bools and their vectors, and 4x4 matrices. They can also
// hold statements, for TSL's Fn bodies: variables, assignments, ifs, loops, break, continue
// and discard, in blocks that seq() turns into a value (see drie::nodes), and call functions
// (nodes::function, three's Fn with a layout), which a program prints once and calls. A value
// is printed once where it is first needed, and printed again after an assignment when it
// reads a variable, or outside the block it was printed in.
module drie;
import std::collections::list, std::collections::map, std::io, std::os::env, sokol, drie::nodes;

// A node of a shader graph: a handle (0: none). Nodes don't change once made, but for a
// uniform's value (Node.set). See drie::nodes for the constructors.
struct Node {
    uint id;
}

// A graph function (nodes::function), printed once in each stage that calls it.
struct NodeFunction {
    int id;
}

fn Node NodeFunction.param(self, int i) => node_param(self.id, i);

// Sets its body: the value it returns (a seq() for statements before it).
fn NodeFunction NodeFunction.returns(self, Node value) {
    node_func_body(self.id, value);
    return self;
}

// A call, of nodes or numbers.
macro Node NodeFunction.call(self, ...) {
    $if $vaarg.len == 0:
        return node_call_fn(self.id, { });
    $else
        Node[$vaarg.len] args;
        $for var $i = 0; $i < $vaarg.len; $i++:
            args[$i] = nodes::to_node($vaarg[$i]);
        $endfor
        return node_call_fn(self.id, &args);
    $endif
}

enum NodeType : char {
    FLOAT,
    VEC2,
    VEC3,
    VEC4,
    BOOL,
    MAT4,
    INT,
    IVEC2,
    IVEC3,
    IVEC4,
    UINT,
    UVEC2,
    UVEC3,
    UVEC4,
    VOID,     // a statement's
    MAT3,
}

// What a type's components are.
enum NodeScalar : char {
    FLOAT,
    INT,
    UINT,
    BOOL,
}

enum NodeOp : char {
    CONST,    // value
    UNIFORM,  // value (Node.set)
    BUILTIN,  // fn: a NodeBuiltin
    TEXTURE,  // texture; args[0] = uv, args[1] = the level (0: from the derivatives); kind 1:
              //   its size instead (a vec2); kind 2: with the gradients args[1] and args[2]
    VARYING,  // args[0], worked out per vertex
    BINARY,   // fn: a NodeBinary
    CALL,     // fn: a NodeFn, args
    SELECT,   // args[0] ? args[1] : args[2]
    SWIZZLE,  // args[0]'s components swizzle[0..count)
    JOIN,     // a vector of args[0..count)
    CONVERT,  // args[0], as type
    NOT,      // kind 0: !args[0]; 1: ~args[0]
    // statements, of type VOID, in blocks (the first statement; each one's `next`):
    VAR,      // a variable, of args[0] at first (0: zeros); declared where it is first read
    ASSIGN,   // args[0] (a VAR, or a swizzle of one) = args[1]
    IF,       // if (args[0]) the block args[1] else the block args[2]
    LOOP,     // kind: a NodeLoop; args[0] the index (a VAR), args[1] from, args[2] to, args[3] the block
    JUMP,     // kind 0: break, 1: continue, 2: discard (the fragment stage)
    EVAL,     // args[0] worked out here (a VAR's declaration, a SEQ that returns nothing)
    // and the one that gives a value
    SEQ,      // the block args[0], then args[1] (0: nothing, a VOID)
    // what a whole program reads (nodes_program.c3):
    INPUT,    // rows of an engine uniform block: res[0] = its NodeSource, res[1] = the first row,
              //   res[2] = the component of the row a vector starts at; args[0] an index into an
              //   array (0: none) of value[0] rows a step, value[1] steps long (as bits)
    STORAGE,  // element args[0] of a storage buffer (res[0]: the view slot) of vec4s, uvec4s or uints
    SAMPLE,   // an engine texture: kind = a NodeSampling; res[0] = the view slot, res[1] = the
              //   sampler slot, res[2] = a NodeTextureKind; args = coordinates, then the level, the
              //   gradients or the depth compared against
    ATTRIBUTE, // vertex attribute kind (its location)
    SYSTEM,   // kind = a NodeSystem
    PARAM,    // a function's parameter kind
    CALL_FN,  // the function value[0] (as bits, in node_funcs) of the count arguments in node_args
              //   from value[1] (as bits)
    BIND,     // a statement: the builtin kind is args[0] from here on
}

// What SAMPLE does with its texture.
enum NodeSampling : char {
    SAMPLE,   // filtered, the level from the derivatives
    LEVEL,    // at level args[1]
    GRAD,     // with the derivatives args[1] and args[2]
    FETCH,    // texel args[0] (integers) of level args[1] (an int)
    SIZE,     // the size of level args[0] (an int): an ivec2, or an ivec3 for 3D textures and arrays
    COMPARE,  // a depth texture's comparison sampler: how much of the 2x2 footprint at args[0]
              //   is nearer than args[1]
    LEVELS,   // how many mip levels the texture has (an int)
}

enum NodeTextureKind : char {
    TEX_2D,
    TEX_3D,
    TEX_ARRAY,  // 2D layers: coordinates (u, v, layer)
    TEX_DEPTH,  // 2D depths, read with COMPARE
}

// The stage's own inputs.
enum NodeSystem : char {
    INSTANCE_ID,   // uint (vertex)
    VERTEX_ID,     // uint (vertex)
    FRAG_COORD,    // vec4 (fragment): the backend's own, as sokol-shdc hands it on
    FRONT_FACING,  // bool (fragment)
}

// How a LOOP goes: its index from args[1] while below (LT, LE) or above (GT, GE) args[2], by 1;
// WHILE: while args[2] holds (a bool), with no index.
enum NodeLoop : char {
    LT,
    LE,
    GT,
    GE,
    WHILE,
}

enum NodeBuiltin : char {
    UV,
    UV1,
    POSITION_LOCAL,   // in the model (with the instance's matrix), as three's positionLocal
    POSITION_WORLD,
    NORMAL_LOCAL,
    NORMAL_WORLD,     // the shaded normal (fragment: after the normal map and normal_node)
    NORMAL_GEOMETRY,  // the surface's own, facing the camera
    CAMERA_POSITION,
    VIEW_DIRECTION,   // from the point toward the camera, normalized
    TIME,             // Renderer.time (s)
    VERTEX_COLOR,     // rgba
    FRAG_COORD,       // the pixel's position in the target (fragment)
    FRONT_FACING,     // bool (fragment)
    MODEL_MATRIX,     // the mesh's matrix_world (a billboard's as it is, before it is turned)
    VIEW_MATRIX,
    PROJECTION_MATRIX,
    OUTPUT,           // the lit color, rgba (in output_node)
    ENV_DIRECTION,    // the direction the surface reflects, in the world (in env_node)
    ENV_ROUGHNESS,    // the roughness it reflects with (in env_node)
    INSTANCE_INDEX,   // the instance's index in its mesh (0 without instances), a float
    // the material's own value at the step a graph replaces, for a graph that changes it:
    MATERIAL_COLOR,     // rgba (in color_node, opacity_node): the color after its maps, vertex
                        //   and instance colors, and its opacity
    MATERIAL_ROUGHNESS, // (in roughness_node, metalness_node and the surface step's others):
    MATERIAL_METALNESS, //   after the material's maps
    ENV_COLOR,          // the environment's reflection along ENV_DIRECTION (in env_node)
    INSTANCE_MATRIX,    // the instance's matrix (identity without instances; vertex stage)
}

enum NodeBinary : char {
    ADD,
    SUB,
    MUL,
    DIV,
    LT,
    LE,
    GT,
    GE,
    EQ,
    NE,
    AND,
    OR,
    REM,      // %: of floats as mod (floored), of integers as their remainder
    BIT_AND,
    BIT_OR,
    BIT_XOR,
    SHL,
    SHR,
    XOR,      // of bools
}

enum NodeFn : char {
    // one argument
    SIN,
    COS,
    TAN,
    ASIN,
    ACOS,
    ATAN,
    EXP,
    EXP2,
    LOG,
    LOG2,
    SQRT,
    INVERSESQRT,
    ABS,
    SIGN,
    FLOOR,
    CEIL,
    FRACT,
    ROUND,
    TRUNC,
    NORMALIZE,
    LENGTH,
    SATURATE,
    DFDX,
    DFDY,
    FWIDTH,
    // two
    POW,
    MIN,
    MAX,
    STEP,
    DOT,
    CROSS,
    DISTANCE,
    ATAN2,
    REFLECT,
    MOD,
    // three
    MIX,
    CLAMP,
    SMOOTHSTEP,
    REFRACT,
    FACEFORWARD,
    // matrices
    TRANSPOSE,
    DETERMINANT,
    // the bits of a number read as the other kind (nodes::as_uint, as_float)
    BITS_AS_UINT,
    BITS_AS_FLOAT,
}

struct NodeData {
    NodeOp op;
    NodeType type;
    char kind;
    char count;
    uint[4] args;
    char[4] swizzle;
    float[16] value;    // (an int's or a uint's: its bits)
    Texture* texture;
    int[3] res;         // INPUT, STORAGE, SAMPLE: where they read
    uint next;          // a statement's: the one after it in its block (0: the last)
}

List{NodeData} node_graph @local;

fn NodeData* node_data(Node n) {
    if (n.id == 0 || n.id > node_graph.len()) abort("drie: nodes: not a node (%d)", n.id);
    return &node_graph[(sz)n.id - 1];
}

fn Node add_node(NodeData d) {
    if (!node_graph.is_initialized()) node_graph.init(mem, 256);
    node_graph.push(d);
    return { (uint)node_graph.len() };
}

fn NodeType Node.type(self) => node_data(self).type;

// How many nodes there are: a mark node_release goes back to.
fn uint node_count() => (uint)node_graph.len();

// Drops the nodes made after `mark` (node_count), with the functions (node_func) made since:
// what building a program needed and nothing else holds. The nodes before it never read the
// ones after.
fn void node_release(uint mark) {
    while (node_graph.len() > mark) node_graph.remove_last();
    release_funcs(mark);
}

fn int components(NodeType t) {
    switch (t) {
        case FLOAT:
        case INT:
        case UINT:
        case BOOL: return 1;
        case VEC2:
        case IVEC2:
        case UVEC2: return 2;
        case VEC3:
        case IVEC3:
        case UVEC3: return 3;
        case VEC4:
        case IVEC4:
        case UVEC4: return 4;
        case MAT4: return 16;
        case MAT3: return 9;
        case VOID: return 0;
    }
}

fn NodeScalar scalar_of(NodeType t) {
    switch (t) {
        case INT:
        case IVEC2:
        case IVEC3:
        case IVEC4: return INT;
        case UINT:
        case UVEC2:
        case UVEC3:
        case UVEC4: return UINT;
        case BOOL: return BOOL;
        default: return FLOAT;
    }
}

// The type of `n` components of `s` (no vectors of bools: 1 only).
fn NodeType type_of(NodeScalar s, int n) {
    if (n < 1 || n > 4) abort("drie: nodes: no vector of %d components", n);
    switch (s) {
        case FLOAT: return (NodeType)(NodeType.FLOAT.ordinal + n - 1);
        case INT: return (NodeType)(NodeType.INT.ordinal + n - 1);
        case UINT: return (NodeType)(NodeType.UINT.ordinal + n - 1);
        case BOOL:
            if (n != 1) abort("drie: nodes: no vectors of bools");
            return BOOL;
    }
}

fn NodeType vector_type(int n) @local => type_of(FLOAT, n);

fn bool is_matrix(NodeType t) => t == MAT4 || t == MAT3;

// The type two types combine into component by component (a number goes with any vector; a
// float with an integer gives floats; bools count as floats).
fn NodeType wider(NodeType a, NodeType b) {
    if (a == VOID || b == VOID) abort("drie: nodes: a statement has no value");
    if (is_matrix(a) || is_matrix(b)) {
        if (a == b) return a;
        abort("drie: nodes: a %s and a %s don't combine", a, b);
    }
    NodeScalar sa = scalar_of(a);
    NodeScalar sb = scalar_of(b);
    if (sa == BOOL) sa = sb == BOOL ? FLOAT : sb;
    if (sb == BOOL) sb = sa;
    NodeScalar s = sa == FLOAT || sb == FLOAT ? FLOAT : sa;
    int na = components(a);
    int nb = components(b);
    if (na != nb && na != 1 && nb != 1) {
        abort("drie: nodes: a %s and a %s don't combine (convert one: nodes::vec3(...), .xyz()...)", a, b);
    }
    return type_of(s, na > nb ? na : nb);
}

// ---- making nodes (the constructors are in drie::nodes)

fn Node node_const(NodeType type, float[16] value) => add_node({ .op = CONST, .type = type, .value = value });

fn Node node_binary(NodeBinary op, Node a, Node b) {
    NodeType ta = a.type();
    NodeType tb = b.type();
    NodeType t;
    switch (op) {
        case ADD:
        case SUB:
        case DIV:
        case REM:
            if (is_matrix(ta) || is_matrix(tb)) abort("drie: nodes: matrices only multiply");
            t = wider(ta, tb);
        case BIT_AND:
        case BIT_OR:
        case BIT_XOR:
            t = wider(ta, tb);
            if (scalar_of(t) == FLOAT) abort("drie: nodes: bitwise operators take integers");
        case SHL:
        case SHR:
            if (scalar_of(ta) == FLOAT || scalar_of(ta) == BOOL || scalar_of(tb) == FLOAT) abort("drie: nodes: shifts take integers");
            t = components(ta) >= components(tb) ? ta : type_of(scalar_of(ta), components(tb));
        case XOR:
            t = BOOL;
        case MUL:
            // (as TSL: a matrix times a number is a matrix; times a vector, a vec4, the vector
            // made one first, a vec3 with w = 1)
            if (ta == MAT3 || tb == MAT3) {
                // (a 3x3 matrix times a 3x3 one, a number or a vec3)
                NodeType o = ta == MAT3 ? tb : ta;
                if (o == MAT3 || components(o) == 1) {
                    t = MAT3;
                } else if (o == VEC3) {
                    t = VEC3;
                } else {
                    abort("drie: nodes: a 3x3 matrix multiplies numbers, vec3s and 3x3 matrices, not a %s", o);
                }
            } else if (ta == MAT4 && tb == MAT4) {
                t = MAT4;
            } else if (ta == MAT4 || tb == MAT4) {
                NodeType v = ta == MAT4 ? tb : ta;
                if (scalar_of(v) == BOOL) abort("drie: nodes: a matrix multiplies numbers, vectors and matrices");
                t = components(v) == 1 ? MAT4 : VEC4;
            } else {
                t = wider(ta, tb);
            }
        case LT:
        case LE:
        case GT:
        case GE:
        case EQ:
        case NE:
            if (components(ta) != 1 || components(tb) != 1) abort("drie: nodes: comparisons take numbers, not vectors");
            t = BOOL;
        case AND:
        case OR:
            t = BOOL;
    }
    return add_node({ .op = BINARY, .type = t, .kind = (char)op.ordinal, .count = 2, .args = { a.id, b.id, 0, 0 } });
}

fn Node node_call(NodeFn f, Node a, Node b = { }, Node c = { }) {
    NodeType ta = a.type();
    if (f == TRANSPOSE || f == DETERMINANT) {
        if (!is_matrix(ta)) abort("drie: nodes: %s takes a matrix", f);
        return add_node({ .op = CALL, .type = f == TRANSPOSE ? ta : FLOAT, .kind = (char)f.ordinal, .count = 1, .args = { a.id, 0, 0, 0 } });
    }
    NodeType t;
    int count = 1;
    switch (f) {
        case BITS_AS_UINT:
            t = type_of(UINT, components(ta));
            count = 1;
        case BITS_AS_FLOAT:
            t = type_of(FLOAT, components(ta));
            count = 1;
        case LENGTH:
            t = FLOAT;
        case MIN:
        case MAX:
        case MOD:
            t = wider(ta, b.type());
            count = 2;
        case POW:
        case STEP:
        case ATAN2:
            t = wider(ta, b.type());
            count = 2;
        case DOT:
        case DISTANCE:
            t = FLOAT;
            count = 2;
        case CROSS:
            t = VEC3;
            count = 2;
        case REFLECT:
            t = ta;
            count = 2;
        case MIX:
            t = wider(ta, b.type());
            count = 3;
        case CLAMP:
            t = wider(ta, wider(b.type(), c.type()));
            count = 3;
            if (t == BOOL) t = FLOAT;
        case SMOOTHSTEP:
            t = wider(c.type(), wider(ta, b.type()));
            count = 3;
        case REFRACT:
        case FACEFORWARD:
            t = ta;
            count = 3;
        case ABS:
            t = ta == BOOL ? FLOAT : ta;
        default:
            t = ta;
    }
    if (is_matrix(t) || t == VOID) abort("drie: nodes: %s takes numbers or vectors", f);
    // (but for abs, min, max, mod and clamp, functions of floats: integers converted)
    switch (f) {
        case ABS:
        case MIN:
        case MAX:
        case MOD:
        case CLAMP:
            break;
        default:
            t = type_of(FLOAT, components(t));
    }
    return add_node({ .op = CALL, .type = t, .kind = (char)f.ordinal, .count = (char)count, .args = { a.id, b.id, c.id, 0 } });
}

fn Node node_convert(Node a, NodeType t) {
    if (a.type() == t) return a;
    return add_node({ .op = CONVERT, .type = t, .count = 1, .args = { a.id, 0, 0, 0 } });
}

fn Node node_swizzle(Node a, String s) @local {
    if (s.len < 1 || s.len > 4) abort("drie: nodes: a swizzle of 1 to 4 components, not \"%s\"", s);
    NodeScalar sa = scalar_of(a.type());
    NodeData d = { .op = SWIZZLE, .type = type_of(sa, (int)s.len), .count = (char)s.len, .args = { a.id, 0, 0, 0 } };
    int have = components(a.type());
    foreach (i, ch : s) {
        int c;
        switch (ch) {
            case 'x':
            case 'r':
            case 's':
                c = 0;
            case 'y':
            case 'g':
            case 't':
                c = 1;
            case 'z':
            case 'b':
            case 'p':
                c = 2;
            case 'w':
            case 'a':
            case 'q':
                c = 3;
            default:
                abort("drie: nodes: no component '%c' (swizzle \"%s\")", ch, s);
        }
        if (c >= have || have > 4) abort("drie: nodes: a %s has no component '%c'", a.type(), ch);
        d.swizzle[i] = "xyzw"[c];
    }
    return add_node(d);
}

fn Node node_join(Node[] parts, NodeType t) {
    if (parts.len < 1 || parts.len > 4) abort("drie: nodes: a vector of 1 to 4 parts");
    int n;
    NodeData d = { .op = JOIN, .type = t, .count = (char)parts.len };
    foreach (i, p : parts) {
        NodeType pt = p.type();
        if (is_matrix(pt) || pt == VOID) abort("drie: nodes: a %s can't be part of a vector", pt);
        n += components(pt);
        d.args[i] = p.id;
    }
    if (parts.len == 1) return node_convert(parts[0], t);
    if (n != components(t)) abort("drie: nodes: %d components don't make a %s", n, t);
    return add_node(d);
}

// An int or uint constant (its components' bits).
fn Node node_const_bits(NodeType type, uint[4] bits) {
    NodeData d = { .op = CONST, .type = type };
    for (int i = 0; i < 4; i++) d.value[i] = bitcast(bits[i], float);
    return add_node(d);
}

// ---- statements

// A variable (three's toVar()), of `init` at first ({}: zeros of type `t`).
fn Node node_var(NodeType t, Node init = { }) {
    if (init.id && init.type() != t) init = node_convert(init, t);
    return add_node({ .op = VAR, .type = t, .count = 1, .args = { init.id, 0, 0, 0 } });
}

fn Node node_assign(Node target, Node value) {
    NodeData* t = node_data(target);
    // (a varying of a variable: the variable, in the vertex stage)
    if (t.op == VARYING && node_data({ t.args[0] }).op == VAR) target = { t.args[0] };
    t = node_data(target);
    if (t.op != VAR && !(t.op == SWIZZLE && node_data({ t.args[0] }).op == VAR)) abort("drie: nodes: assign() to a variable (or its components)");
    return add_node({ .op = ASSIGN, .type = VOID, .count = 2, .args = { target.id, value.id, 0, 0 } });
}

fn Node node_if(Node cond, Node then, Node otherwise = { }) {
    return add_node({ .op = IF, .type = VOID, .count = 3, .args = { cond.id, then.id, otherwise.id, 0 } });
}

// A loop of the block `body` with `index` (a VAR of a number) from `from`, while `cond` holds
// against `to`; with WHILE, while `to` (a bool) holds.
fn Node node_loop(NodeLoop cond, Node index, Node from, Node to, Node body) {
    if (cond != WHILE && node_data(index).op != VAR) abort("drie: nodes: a loop's index is a variable");
    return add_node({ .op = LOOP, .type = VOID, .kind = (char)cond.ordinal, .count = 4, .args = { index.id, from.id, to.id, body.id } });
}

fn Node node_jump(int kind) => add_node({ .op = JUMP, .type = VOID, .kind = (char)kind });

// A statement of an expression: worked out at this point.
fn Node node_eval(Node x) => add_node({ .op = EVAL, .type = VOID, .count = 1, .args = { x.id, 0, 0, 0 } });

// Statements in a block: chained in order; the block is its first one ({}: empty). Each
// statement belongs to one block.
fn Node node_block(Node[] stmts) {
    Node first;
    Node last;
    foreach (st : stmts) {
        if (!st.id) continue;
        if (node_data(st).type != VOID) st = node_eval(st);
        if (last.id) {
            node_data(last).next = st.id;
        } else {
            first = st;
        }
        last = st;
    }
    return first;
}

// Appends `stmt` after `last` (a block's last statement) and returns it (the new last).
fn Node node_append(Node last, Node stmt) {
    if (node_data(stmt).type != VOID) stmt = node_eval(stmt);
    if (last.id) node_data(last).next = stmt.id;
    return stmt;
}

// The block `block` worked out, then `value` ({}: nothing): an expression with statements in
// it (a TSL Fn's body and what it returns).
fn Node node_seq(Node block, Node value = { }) {
    return add_node({ .op = SEQ, .type = value.id ? value.type() : VOID, .count = 2, .args = { block.id, value.id, 0, 0 } });
}

// ---- Node's operators and methods (TSL's chaining: a.mul(b) as a * b)

fn Node Node.add(self, Node b) @operator(+) => node_binary(ADD, self, b);
fn Node Node.sub(self, Node b) @operator(-) => node_binary(SUB, self, b);
fn Node Node.mul(self, Node b) @operator(*) => node_binary(MUL, self, b);
fn Node Node.div(self, Node b) @operator(/) => node_binary(DIV, self, b);
fn Node Node.add_f(self, float b) @operator_s(+) => node_binary(ADD, self, nodes::num(b));
fn Node Node.mul_f(self, float b) @operator_s(*) => node_binary(MUL, self, nodes::num(b));
fn Node Node.sub_f(self, float b) @operator(-) => node_binary(SUB, self, nodes::num(b));
fn Node Node.rsub_f(self, float a) @operator_r(-) => node_binary(SUB, nodes::num(a), self);
fn Node Node.div_f(self, float b) @operator(/) => node_binary(DIV, self, nodes::num(b));
fn Node Node.rdiv_f(self, float a) @operator_r(/) => node_binary(DIV, nodes::num(a), self);
fn Node Node.negate(self) => node_binary(MUL, self, nodes::num(-1));
fn Node Node.one_minus(self) => node_binary(SUB, nodes::num(1), self);

fn Node Node.less_than(self, Node b) => node_binary(LT, self, b);
fn Node Node.less_than_equal(self, Node b) => node_binary(LE, self, b);
fn Node Node.greater_than(self, Node b) => node_binary(GT, self, b);
fn Node Node.greater_than_equal(self, Node b) => node_binary(GE, self, b);
fn Node Node.equal(self, Node b) => node_binary(EQ, self, b);
fn Node Node.not_equal(self, Node b) => node_binary(NE, self, b);

// Its components, as a swizzle: n.x(), n.xyz(), n.swizzle("zyx")
fn Node Node.swizzle(self, String s) => node_swizzle(self, s);
fn Node Node.x(self) => node_swizzle(self, "x");
fn Node Node.y(self) => node_swizzle(self, "y");
fn Node Node.z(self) => node_swizzle(self, "z");
fn Node Node.w(self) => node_swizzle(self, "w");
fn Node Node.xy(self) => node_swizzle(self, "xy");
fn Node Node.xz(self) => node_swizzle(self, "xz");
fn Node Node.yz(self) => node_swizzle(self, "yz");
fn Node Node.xyz(self) => node_swizzle(self, "xyz");
fn Node Node.rgb(self) => node_swizzle(self, "xyz");

// A uniform's new value (nodes::uniform...): read by the next draw, without a rebuild.
fn void Node.set(self, float x, float y = 0, float z = 0, float w = 0) {
    NodeData* d = node_data(self);
    if (d.op != UNIFORM) abort("drie: nodes: set() is for uniforms");
    d.value[0] = x;
    d.value[1] = y;
    d.value[2] = z;
    d.value[3] = w;
}

fn void Node.set_vec3(self, Vector3 v) => self.set(v.x, v.y, v.z);

// (sRGB hex, as three's Color: linear values)
fn void Node.set_color(self, Color c) => self.set_vec3(color_to_linear(c));

fn void Node.set_matrix(self, Mat4 m) {
    NodeData* d = node_data(self);
    if (d.op != UNIFORM || d.type != MAT4) abort("drie: nodes: set_matrix() is for matrix uniforms");
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) d.value[c * 4 + r] = m[c][r];
    }
}

// A statement: `value` into this variable (or these components of one).
fn Node Node.assign(self, Node value) => node_assign(self, value);

// The value a uniform has now (x for a float).
fn float[4] Node.value(self) {
    float[4] v = node_data(self).value[0:4];
    return v;
}

// ---- building: a material's graphs printed into its shaders

// The shading languages the graphs are printed in, one per sokol backend.
enum NodeLang : char {
    GLSL,
    MSL,
    HLSL,
}

// Which of a hook's bodies is being printed (they share the uniforms, textures and varyings)
enum NodeStage : char @local {
    VERTEX,       // position_node: positions as they come in
    VERTEX_OUT,   // the varyings: positions as position_node moved them
    FRAGMENT,
}

// The engine's programs that a node material has its own builds of (see shader_core.glsl).
enum NodeVariant : char {
    MAIN,
    OPAQUE,
    MRT,
    OPAQUE_MRT,
    VSM,
    MRT_VSM,
    DEPTH,
    SHADOW,
}

const int NODE_PARAMS = 32;
const int NODE_VARYINGS = 4;
const int NODE_TEXTURES = 6;

// The shaders a graph prints into (shared by the materials whose graphs print the same), and
// their pipelines.
// The hooks a node program fills: node_vertex, node_color, node_normal, node_surface,
// node_output, node_env and node_specular (NodeProgram's bodies, in this order).
const int NODE_HOOKS = 7;

enum ShaderFamily : char {
    PBR,
    PHYSICAL,
    UNLIT,
    GBUFFER,
}

struct NodeProgram {
    String vertex_funcs;
    String frag_funcs;
    String vertex_body;
    String color_body;
    String normal_body;
    String surface_body;
    String output_body;
    String env_body;
    String specular_body;
    bool cuts;               // color_node or opacity_node can cut holes: a discard, or alpha
                             // (with the material's alpha_test), which the shadow maps cut too
    bool has_frag;           // at least one fragment hook body is non-empty
    bool has_vert;           // vertex hook body is non-empty
    String unsupported;      // what the graphs use that the hooks can't print ("": nothing)
    SgShader[8][4] shaders;  // by ShaderFamily, NodeVariant
    int[8][4] shader_jobs;   // background sg_warm_shader_begin jobs (+1; 0: none)
    HashMap{ulong, SgPipeline} pipelines;
    SgPipeline shadow_pipeline;
    // its pipelines being compiled in the background (sg_warm_begin), by their pipelines' keys:
    // the job, or -1 once done; the program draws nothing while one runs (busy)
    HashMap{ulong, int} warming;
    uint busy_epoch;
    bool busy_now;
}

// Its pipelines compiled in the background, and how many are still running: { running, all }.
// (Those done are let go of: their pipelines are then made at once, from the driver's cache.)
fn int[2] NodeProgram.compiling(&self) {
    int[2] n;
    if (!self.warming.is_initialized()) return n;
    List{ulong} done;
    done.tinit();
    self.warming.@each(; ulong key, int job) {
        n[1]++;
        if (job >= 0) {
            if (sg_warm_state(job) == 0) {
                n[0]++;
            } else {
                sg_warm_end(job);
                done.push(key);
            }
        }
    };
    foreach (key : done) self.warming[key] = -1;
    return n;
}

// The key of a node program's shadow pipeline among its pipelines' (see get_pipeline).
const ulong NODE_SHADOW_KEY = ~(ulong)0;

extern fn int sg_warm_begin(SgShaderDesc* shader, SgPipelineDesc* pipeline) @cname("sg_warm_begin");
extern fn int sg_warm_state(int job) @cname("sg_warm_state");
extern fn void sg_warm_end(int job) @cname("sg_warm_end");
extern fn void sg_warm_wait(int job) @cname("sg_warm_wait");
extern fn int sg_warm_shader_begin(SgShaderDesc* shader) @cname("sg_warm_shader_begin");

// A new one each render (Renderer.render): what NodeProgram.busy says holds for all of a
// render's passes, so that a program the depth prepass left out is left out of the main pass
// too.
uint node_epoch;

// Whether one of the program's pipelines is still being compiled in the background: then
// nothing of it is drawn, in any pass.
fn bool NodeProgram.busy(&self) {
    if (self.busy_epoch == node_epoch) return self.busy_now;
    self.busy_epoch = node_epoch;
    @pool() {
        self.busy_now = self.compiling()[0] > 0;
    };
    return self.busy_now;
}

// Whether the pipeline `key` of the program (variant `v`, `desc` but for its shader) can be made
// now without the driver compiling for long: it was compiled in the background, or can't be
// (not Metal, or not `background`: Renderer.compile_in_background). Otherwise starts that and
// returns false.
fn bool NodeProgram.pipeline_ready(&self, ulong key, NodeVariant v, SgShaderDesc base, SgPipelineDesc* desc, bool background) {
    if (!self.warming.is_initialized()) self.warming.init(mem);
    if (!background) return true;
    int? job = self.warming[key];
    if (try job) {
        if (job < 0) return true;
        if (sg_warm_state(job) == 0) return false;
        sg_warm_end(job);
        self.warming[key] = -1;
        return true;
    }
    @pool() {
        SgShaderDesc spliced = self.spliced(v, base);
        int j = sg_warm_begin(&spliced, desc);
        self.warming[key] = j;
        // (busy from now on, in this render too)
        if (j >= 0 && self.busy_epoch == node_epoch) self.busy_now = true;
        return j < 0;
    };
}

// A material's build: its graphs' program, and where its uniforms and textures go.
struct NodeBuild {
    Node[NODE_SLOTS] slots;  // the material's *_node fields it was built from
    Material* material;      // whose they are (its maps for nodes::material_map)
    NodeProgram* program;
    List{Node} uniforms;     // in their order in node_params (a matrix takes four)
    Texture*[NODE_TEXTURES] textures; // node texture k: tex_ext(5 - k), or a generated program's
                                      // view NODE_USER_VIEW + k
    int texture_count;
    // its generated programs (materials_nodes.c3), by their MaterialKey (with `nodes`): only
    // its own graphs print them
    HashMap{KeyWords, MaterialProgram*} programs;
    // whether one of them is still compiling in the background (in render busy_epoch, see
    // node_epoch): then none of it is drawn, in any pass (see material_busy)
    uint busy_epoch;
    bool busy_now;
}

// Node texture k: the texture a graph names, or for nodes::material_map the material's map.
fn Texture* NodeBuild.texture(&self, int k) {
    Texture* t = self.textures[k];
    if (try m = map_of_mark(t)) {
        Texture* own = self.material.own_map(m);
        if (!own) abort("drie: a node material reads nodes::material_map(%s), and the material has no %s", m, m);
        return own;
    }
    return t;
}

HashMap{String, NodeProgram*} node_programs @local;

struct NodeGen @local {
    NodeLang lang;
    NodeStage stage;
    DString out;
    // per node, once printed: the scope it was printed in (0: not printed), the epoch, and how
    // many times (its name's suffix)
    uint[] memo;
    uint[] memo_epoch;
    uint[] serial;
    // per node, its number in the body being printed (0: none yet), which its name carries:
    // the same graph prints the same text wherever its nodes sit in node_graph, so the driver's
    // cache of compiled shaders knows it (and node_programs shares it) after other graphs were
    // made
    uint[] local;
    uint locals;
    char[] reads_var;   // 0: not known, 1: no, 2: it reads a variable
    List{bool} scope_open; // by scope (from 1)
    uint scope;
    uint epoch;         // a new one after every assignment and block: what reads a variable is worked out again
    int indent;
    String[] kept;      // VERTEX_OUT: the variables carried by varyings, by their names in VERTEX
    NodeBuild* build;
    List{Node} varyings;
    int params;   // node_params used
    bool output;  // printing node_output's body (nodes::output() is there)
    bool env;     // printing node_env's (nodes::env_direction() and env_roughness() are there)
    bool color;   // printing node_color's (nodes::material_color() is there)
    bool surface; // printing node_surface's (material_roughness() and material_metalness())
    bool defer_color; // color_node emitted inside node_surface to share subgraphs with surface nodes
    bool discards; // a discard printed
    int cur_func;  // NodeFunc index being printed (-1: inside a hook body)
    List{int} frag_funcs;
    List{int} vert_funcs;
    DString frag_fn_defs;
    DString vert_fn_defs;
    // what the graphs use that the hooks can't print (the program aborts when it is made)
    String unsupported;
}

// How many *_node fields a material has (NodeBuild.slots).
const int NODE_SLOTS = 15;

fn Node[NODE_SLOTS] node_slots(Material* mat) @local {
    Node[NODE_SLOTS] s = {
        mat.color_node, mat.opacity_node, mat.normal_node, mat.roughness_node, mat.metalness_node,
        mat.emissive_node, mat.position_node, mat.output_node, mat.mask_node, mat.env_node,
        mat.specular_node, mat.shininess_node, mat.clearcoat_node, mat.clearcoat_roughness_node, mat.ao_node
    };
    return s;
}

fn bool same_nodes(Node[NODE_SLOTS] a, Node[NODE_SLOTS] b) @local {
    foreach (i, n : a) {
        if (n.id != b[i].id) return false;
    }
    return true;
}

fn bool has_nodes(Material* mat) {
    foreach (n : node_slots(mat)) {
        if (n.id) return true;
    }
    return false;
}

// The material's build, made or remade if its nodes changed; null without nodes.
fn NodeBuild* node_build_of(Material* mat) {
    if (!has_nodes(mat)) return null;
    Node[NODE_SLOTS] slots = node_slots(mat);
    NodeBuild* b = mat.node_build;
    if (b && same_nodes(b.slots, slots)) return b;
    // (its program stays: other materials may share it, and the same graphs may come back)
    if (b) {
        b.uniforms.free();
        free_build_programs(b);
        mem::free(b);
    }
    b = mem::new(NodeBuild);
    b.slots = slots;
    b.material = mat;
    b.uniforms.init(mem);
    mat.node_build = b;
    NodeLang lang;
    switch (sg::query_backend()) {
        case GLCORE:
        case GLES3: lang = GLSL;
        case METAL_IOS:
        case METAL_MACOS:
        case METAL_SIMULATOR: lang = MSL;
        case D3D11: lang = HLSL;
        default: abort("drie: node materials need the GL, Metal or D3D11 backend");
    }
    @pool() {
        NodeGen g = { .lang = lang, .build = b, .cur_func = -1 };
        g.varyings.tinit();
        g.scope_open.tinit();
        g.frag_funcs.tinit();
        g.vert_funcs.tinit();
        g.frag_fn_defs.tinit();
        g.vert_fn_defs.tinit();
        sz count = node_graph.len() + 1;
        g.memo = mem::temp_array(uint, count);
        g.memo_epoch = mem::temp_array(uint, count);
        g.serial = mem::temp_array(uint, count);
        g.local = mem::temp_array(uint, count);
        g.reads_var = mem::temp_array(char, count);
        g.kept = mem::temp_array(String, count);
        // the fragment stage first: it finds the varyings the vertex stage writes
        // When color_node is opaque (VEC3, no mask/opacity/alpha_test) and surface nodes are
        // also present, run color_node inside node_surface so shared subgraphs (like grime or
        // ground noise) are evaluated once per pixel instead of twice.
        bool has_surface_nodes = mat.roughness_node.id || mat.metalness_node.id || mat.emissive_node.id
            || mat.clearcoat_node.id || mat.clearcoat_roughness_node.id;
        g.defer_color = mat.color_node.id && mat.color_node.type() != VEC4
            && !mat.mask_node.id && !mat.opacity_node.id && mat.alpha_test <= 0 && has_surface_nodes;
        g.stage = FRAGMENT;
        g.color = true;
        String color = g.body(fn void(NodeGen* g, Material* mat) {
            if (mat.mask_node.id) {
                if (mat.mask_node.type() != BOOL) abort("drie: mask_node is a bool");
                g.line("if (!(%s)) %s", g.value_as(mat.mask_node, BOOL), g.lang == MSL ? "discard_fragment();" : "discard;");
                g.discards = true;
            }
            if (mat.color_node.id && !g.defer_color) {
                Node c = mat.color_node;
                if (c.type() == VEC4) {
                    g.line("base = %s;", g.value_as(c, VEC4));
                } else {
                    g.line("base.xyz = %s;", g.value_as(c, VEC3));
                }
            }
            if (mat.opacity_node.id) g.line("base.w = %s;", g.value_as(mat.opacity_node, FLOAT));
        }, mat);
        g.color = false;
        bool cuts = g.discards || mat.opacity_node.id || (mat.color_node.id && mat.color_node.type() == VEC4);
        String normal = g.body(fn void(NodeGen* g, Material* mat) {
            if (mat.normal_node.id) g.line("n = normalize(%s);", g.value_as(mat.normal_node, VEC3));
        }, mat);
        g.surface = true;
        g.color = g.defer_color;
        String surface = g.body(fn void(NodeGen* g, Material* mat) {
            if (!g.defer_color && !mat.roughness_node.id && !mat.metalness_node.id && !mat.emissive_node.id
                && !mat.clearcoat_node.id && !mat.clearcoat_roughness_node.id) return;
            if (g.defer_color) g.line("base.xyz = %s;", g.value_as(mat.color_node, VEC3));
            if (mat.roughness_node.id) g.line("rough = %s;", g.value_as(mat.roughness_node, FLOAT));
            if (mat.metalness_node.id) g.line("metal = %s;", g.value_as(mat.metalness_node, FLOAT));
            if (mat.emissive_node.id) {
                g.line("emissive = %s(%s, 1.0);", g.type_name(VEC4), g.value_as(mat.emissive_node, VEC3));
            } else {
                g.line("emissive = %s;", g.zero(VEC4));
            }
            if (mat.clearcoat_node.id) g.line("clearcoat = %s;", g.value_as(mat.clearcoat_node, FLOAT));
            if (mat.clearcoat_roughness_node.id) g.line("clearcoat_rough = %s;", g.value_as(mat.clearcoat_roughness_node, FLOAT));
        }, mat);
        g.color = false;
        g.surface = false;
        g.output = true;
        String output = g.body(fn void(NodeGen* g, Material* mat) {
            if (mat.output_node.id) {
                Node o = mat.output_node;
                if (o.type() == VEC4) {
                    g.line("result = %s;", g.value_as(o, VEC4));
                } else {
                    g.line("result.xyz = %s;", g.value_as(o, VEC3));
                }
            }
        }, mat);
        g.output = false;
        g.env = true;
        String env = g.body(fn void(NodeGen* g, Material* mat) {
            if (mat.env_node.id) g.line("env = %s;", g.value_as(mat.env_node, VEC3));
        }, mat);
        g.env = false;
        String specular = g.body(fn void(NodeGen* g, Material* mat) {
            if (mat.specular_node.id) g.line("specular = %s;", g.value_as(mat.specular_node, VEC3));
            if (mat.shininess_node.id) g.line("shininess = %s;", g.value_as(mat.shininess_node, FLOAT));
        }, mat);
        String vertex = g.body(fn void(NodeGen* g, Material* mat) {
            g.stage = VERTEX;
            // the variables that varyings carry (TSL's varying(...).assign() in position_node):
            // declared first, where every statement of the stage sees them
            foreach (v : g.varyings) {
                NodeData* d = node_data(v);
                if (d.op == VARYING && node_data({ d.args[0] }).op == VAR) g.emit({ d.args[0] });
            }
            if (mat.position_node.id) {
                String p = g.value_as(mat.position_node, VEC3);
                g.line("local = %s;", p);
            }
            // (the varyings from the moved position; the variables as the position left them)
            foreach (v : g.varyings) {
                NodeData* d = node_data(v);
                if (d.op == VARYING && node_data({ d.args[0] }).op == VAR) g.kept[d.args[0]] = g.name({ d.args[0] });
            }
            g.stage = VERTEX_OUT;
            g.reset();
            foreach (i, v : g.varyings) {
                // a varying node's value, or a builtin read in the fragment stage
                NodeData* d = node_data(v);
                Node src = d.op == VARYING ? (Node) { d.args[0] } : v;
                g.line("vary%d = %s;", i, g.padded(src));
            }
            for (sz i = g.varyings.len(); i < NODE_VARYINGS; i++) g.line("vary%d = %s;", i, g.zero(VEC4));
        }, mat);
        // (the material's maps its graphs read: there when it is built)
        for (int k = 0; k < b.texture_count; k++) b.texture(k);
        b.program = node_program(
            { vertex, color, normal, surface, output, env, specular },
            cuts,
            g.unsupported,
            g.vert_fn_defs.str_view(),
            g.frag_fn_defs.str_view()
        );
    };
    return b;
}

// The program of these hook bodies (vertex, color, normal, surface, output), made the first
// time: shared by the materials whose graphs print them.
fn NodeProgram* node_program(
    String[NODE_HOOKS] bodies,
    bool cuts,
    String unsupported = "",
    String vertex_funcs = "",
    String frag_funcs = ""
) @private {
    @pool() {
        String key = string::tformat("%s\n--\n%s\n--\n%s\n--\n%s\n--\n%s\n--\n%s\n--\n%s\n--\n%s\n--\n%s\n--\n%s",
                                     bodies[0], bodies[1], bodies[2], bodies[3], bodies[4], bodies[5], bodies[6],
                                     cuts, vertex_funcs, frag_funcs);
        if (unsupported.len) key = string::tformat("%s\n--\n%s", key, unsupported);
        if (!node_programs.is_initialized()) node_programs.init(mem);
        NodeProgram*? found = node_programs[key];
        if (try found) return found;
        NodeProgram* p = mem::new(NodeProgram);
        p.vertex_funcs = vertex_funcs.copy(mem);
        p.frag_funcs = frag_funcs.copy(mem);
        p.vertex_body = bodies[0].copy(mem);
        p.color_body = bodies[1].copy(mem);
        p.normal_body = bodies[2].copy(mem);
        p.surface_body = bodies[3].copy(mem);
        p.output_body = bodies[4].copy(mem);
        p.env_body = bodies[5].copy(mem);
        p.specular_body = bodies[6].copy(mem);
        p.cuts = cuts;
        p.has_vert = bodies[0].len > 0;
        p.has_frag = bodies[1].len > 0 || bodies[2].len > 0 || bodies[3].len > 0
            || bodies[4].len > 0 || bodies[5].len > 0 || bodies[6].len > 0;
        p.unsupported = unsupported.copy(mem);
        p.pipelines.init(mem);
        node_programs[key.copy(mem)] = p;
        return p;
    };
}

// Prints a NodeFunction once per stage before any hook that calls it.
fn void NodeGen.emit_func(&self, int fi) {
    bool frag = self.stage == FRAGMENT;
    List{int}* seen = frag ? &self.frag_funcs : &self.vert_funcs;
    foreach (id : *seen) if (id == fi) return;
    seen.push(fi);

    NodeFunc* fd = node_funcs.get_ref(fi);
    DString saved_out = self.out;
    uint[] saved_memo = self.memo;
    uint[] saved_memo_epoch = self.memo_epoch;
    uint[] saved_serial = self.serial;
    uint[] saved_local = self.local;
    uint saved_locals = self.locals;
    uint saved_scope = self.scope;
    uint saved_epoch = self.epoch;
    int saved_indent = self.indent;
    int saved_cur_func = self.cur_func;
    List{bool} saved_scope_open = self.scope_open;

    sz count = node_graph.len() + 1;
    self.out = dstring::temp();
    self.memo = mem::temp_array(uint, count);
    self.memo_epoch = mem::temp_array(uint, count);
    self.serial = mem::temp_array(uint, count);
    self.local = mem::temp_array(uint, count);
    self.locals = 0;
    self.scope_open = {};
    self.scope_open.tinit();
    self.scope_open.push(true);
    self.scope = 1;
    self.epoch = 1;
    self.indent = 1;
    self.cur_func = fi;

    if (fd.ret == VOID) {
        self.emit(fd.body);
    } else {
        String r = self.value_as(fd.body, fd.ret);
        self.line("return %s;", r);
    }
    String body_text = self.out.str_view();

    self.out = saved_out;
    self.memo = saved_memo;
    self.memo_epoch = saved_memo_epoch;
    self.serial = saved_serial;
    self.local = saved_local;
    self.locals = saved_locals;
    self.scope = saved_scope;
    self.epoch = saved_epoch;
    self.indent = saved_indent;
    self.cur_func = saved_cur_func;
    self.scope_open = saved_scope_open;

    DString* defs = frag ? &self.frag_fn_defs : &self.vert_fn_defs;
    String ret_t = fd.ret == VOID ? "void" : self.type_name(fd.ret);
    if (self.lang == MSL) {
        if (frag) {
            defs.appendf("%s %s(thread const NodeIn& ni, constant node_fs_params& nu, texture2d<float> tex_ext0, sampler tex_smp, texture2d<float> tex_ext1, texture2d<float> tex_ext2, texture2d<float> tex_ext3, texture2d<float> tex_ext4, texture2d<float> tex_ext5", ret_t, fd.name);
        } else {
            defs.appendf("%s %s(thread const NodeIn& ni, constant node_vs_params& nu", ret_t, fd.name);
        }
    } else {
        defs.appendf("%s %s(NodeIn ni", ret_t, fd.name);
    }
    for (int i = 0; i < fd.count; i++) {
        defs.appendf(", %s p%d", self.type_name(fd.params[i]), i);
    }
    defs.appendf(")\n{\n%s}\n\n", body_text);
}


// Prints one hook's body: `fill` writes its lines.
fn String NodeGen.body(&self, NodeBodyFn fill, Material* mat) {
    self.out = dstring::temp();
    for (sz i = 0; i < self.local.len; i++) self.local[i] = 0;
    self.locals = 0;
    self.reset();
    fill(self, mat);
    return self.out.str_view();
}

// Nothing printed yet (in a new body, or stage).
fn void NodeGen.reset(&self) {
    for (sz i = 0; i < self.memo.len; i++) {
        self.memo[i] = 0;
        self.serial[i] = 0;
    }
    self.scope_open.clear();
    self.scope_open.push(true);
    self.scope = 1;
    self.epoch = 1;
    self.indent = 0;
}

alias NodeBodyFn @local = fn void(NodeGen* g, Material* mat);

fn void NodeGen.line(&self, String format, args...) {
    self.out.append("    ");
    for (int i = 0; i < self.indent; i++) self.out.append("    ");
    self.out.appendf(format, ...args);
    self.out.append("\n");
}

fn String NodeGen.type_name(&self, NodeType t) {
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

fn String zero_literal(NodeScalar s) @local {
    switch (s) {
        case FLOAT: return "0.0";
        case INT: return "0";
        case UINT: return "0u";
        case BOOL: return "false";
    }
}

fn String one_literal(NodeScalar s) @local {
    switch (s) {
        case FLOAT: return "1.0";
        case INT: return "1";
        case UINT: return "1u";
        case BOOL: return "true";
    }
}

fn String NodeGen.zero(&self, NodeType t) {
    int n = components(t);
    String z = zero_literal(scalar_of(t));
    if (n == 1) return z;
    if (t == MAT4) return string::tformat("%s(%s)", self.type_name(t), z);
    if (self.lang != HLSL) return string::tformat("%s(%s)", self.type_name(t), z);
    String all = z;
    for (int i = 1; i < n; i++) all = string::tformat("%s, %s", all, z);
    return string::tformat("%s(%s)", self.type_name(t), all);
}

fn String literal(float f) @local => string::tformat("(%.8e)", f);

// A constant's component `i`, in its type's spelling.
fn String component_literal(NodeType t, float[16]* v, int i) @local {
    uint bits = bitcast((*v)[i], uint);
    switch (scalar_of(t)) {
        case INT: return string::tformat("(%d)", (int)bits);
        case UINT: return string::tformat("%du", bits);
        case BOOL: return (*v)[i] != 0 ? "true" : "false";
        case FLOAT: return literal((*v)[i]);
    }
}

// (VERTEX_OUT has its own names: the position moved, what reads it differs; a node printed
// again, after its scope closed or a variable it reads changed, takes a suffix)
fn String NodeGen.name(&self, Node n) {
    String prefix = self.stage == VERTEX_OUT ? "no" : "nd";
    uint id = self.local[n.id];
    uint k = self.serial[n.id];
    if (k <= 1) return string::tformat("%s%d", prefix, id);
    return string::tformat("%s%d_%d", prefix, id, k);
}

// `x` (of type `from`) as a `to`, as TSL converts: a number fills a vector, a narrower vector
// gets zeros and a w of 1, a wider one loses its last components.
fn String NodeGen.convert(&self, String x, NodeType from, NodeType to) {
    if (from == to) return x;
    if (from == VOID || to == VOID) abort("drie: nodes: a statement (or a function that returns nothing) where a value goes");
    if (from == MAT4 || to == MAT4) abort("drie: nodes: a matrix converts to nothing else");
    NodeScalar sf = scalar_of(from);
    NodeScalar st = scalar_of(to);
    int f = components(from);
    int t = components(to);
    if (st == BOOL) {
        return string::tformat("(%s != %s)", f == 1 ? x : string::tformat("%s.x", x), zero_literal(sf));
    }
    if (sf == BOOL) {
        return self.convert(string::tformat("(%s ? %s : %s)", x, one_literal(st), zero_literal(st)), type_of(st, 1), to);
    }
    // the components first, then what they are
    if (f != t) {
        NodeType mid = type_of(sf, t);
        String z = zero_literal(sf);
        if (f == 1) {
            x = self.lang == HLSL ? string::tformat("((%s)%s)", self.type_name(mid), x) : string::tformat("%s(%s)", self.type_name(mid), x);
        } else if (t < f) {
            x = string::tformat("%s.%s", x, "xyzw"[:(sz)t]);
        } else {
            switch (f * 10 + t) {
                case 23: x = string::tformat("%s(%s, %s)", self.type_name(mid), x, z);
                case 24: x = string::tformat("%s(%s, %s, %s)", self.type_name(mid), x, z, one_literal(sf));
                default: x = string::tformat("%s(%s, %s)", self.type_name(mid), x, one_literal(sf));
            }
        }
    }
    if (sf == st) return x;
    if (self.lang == HLSL) return string::tformat("((%s)%s)", self.type_name(to), x);
    return string::tformat("%s(%s)", self.type_name(to), x);
}

// Node `n`'s value as a `t`.
fn String NodeGen.value_as(&self, Node n, NodeType t) => self.convert(self.emit(n), n.type(), t);

// `n` padded into a vec4 (a varying)
fn String NodeGen.padded(&self, Node n) {
    NodeType t = n.type();
    if (t == MAT4 || t == VOID) abort("drie: nodes: a %s can't be a varying", t);
    String x = self.emit(n);
    if (scalar_of(t) != FLOAT) {
        NodeType f = type_of(FLOAT, components(t));
        x = self.convert(x, t, f);
        t = f;
    }
    switch (t) {
        case FLOAT: return string::tformat("%s(%s, 0.0, 0.0, 0.0)", self.type_name(VEC4), x);
        case VEC2: return string::tformat("%s(%s, 0.0, 0.0)", self.type_name(VEC4), x);
        case VEC3: return string::tformat("%s(%s, 0.0)", self.type_name(VEC4), x);
        default: return x;
    }
}

// A matrix product, a * b in GLSL's order (HLSL turns it around, as SPIRV-Cross does).
fn String NodeGen.mat_mul(&self, String a, String b) {
    if (self.lang == HLSL) return string::tformat("mul(%s, %s)", b, a);
    return string::tformat("(%s * %s)", a, b);
}

// The varying that carries `n` (a VARYING node or a builtin) to the fragment stage.
fn int NodeGen.varying(&self, Node n) {
    foreach (i, v : self.varyings) {
        if (v.id == n.id) return (int)i;
    }
    if (self.varyings.len() >= NODE_VARYINGS) abort("drie: nodes: more than %d varyings", NODE_VARYINGS);
    self.varyings.push(n);
    return (int)self.varyings.len() - 1;
}

fn String NodeGen.read_varying(&self, Node n) {
    int k = self.varying(n);
    NodeType t = n.type();
    int c = components(t);
    String x = c == 4 ? string::tformat("ni.vary%d", k) : string::tformat("ni.vary%d.%s", k, "xyzw"[:(sz)c]);
    return self.convert(x, type_of(FLOAT, c), t);
}

fn String NodeGen.builtin(&self, Node n, NodeBuiltin b) {
    bool fragment = self.stage == FRAGMENT;
    switch (b) {
        case UV: return "ni.uv";
        case UV1: return "ni.uv1";
        case POSITION_LOCAL:
            if (fragment) return self.read_varying(n);
            return self.stage == VERTEX ? "ni.local" : "local";
        case NORMAL_LOCAL:
            if (fragment) return self.read_varying(n);
            return "ni.local_normal";
        case POSITION_WORLD:
            if (self.stage == VERTEX_OUT) {
                String mm = self.lang == MSL ? "nu.node_model" : "node_model";
                return string::tformat("%s.xyz", self.mat_mul(mm, string::tformat("%s(local, 1.0)", self.type_name(VEC4))));
            }
            return "ni.world";
        case NORMAL_WORLD: return "ni.normal";
        case NORMAL_GEOMETRY: return "ni.geo_normal";
        case CAMERA_POSITION: return self.lang == MSL ? "nu.node_info.xyz" : "node_info.xyz";
        case VIEW_DIRECTION:
            return string::tformat("normalize(%s - %s)", self.lang == MSL ? "nu.node_info.xyz" : "node_info.xyz", self.builtin(n, POSITION_WORLD));
        case TIME: return self.lang == MSL ? "nu.node_info.w" : "node_info.w";
        case VERTEX_COLOR: return "ni.color";
        case FRAG_COORD:
            if (!fragment) abort("drie: nodes: frag_coord is for the fragment stage");
            return "ni.frag.xy";
        case FRONT_FACING:
            if (!fragment) abort("drie: nodes: front_facing is for the fragment stage");
            return "(ni.frag.z > 0.5)";
        case MODEL_MATRIX: return self.lang == MSL ? "nu.node_model" : "node_model";
        case VIEW_MATRIX: return self.lang == MSL ? "nu.node_view" : "node_view";
        case PROJECTION_MATRIX: return self.lang == MSL ? "nu.node_proj" : "node_proj";
        case OUTPUT:
            if (!self.output) abort("drie: nodes: output() is for output_node");
            return "result";
        case ENV_DIRECTION:
            if (!self.env) abort("drie: nodes: env_direction() is for env_node");
            return "dir";
        case ENV_ROUGHNESS:
            if (!self.env) abort("drie: nodes: env_roughness() is for env_node");
            return "rough";
        case INSTANCE_INDEX:
            if (fragment) return self.read_varying(n);
            return "ni.frag.w";
        case MATERIAL_COLOR:
            if (!self.color) abort("drie: nodes: material_color() is for color_node and opacity_node");
            return "base";
        case MATERIAL_ROUGHNESS:
            if (!self.surface) abort("drie: nodes: material_roughness() is for roughness_node and metalness_node");
            return "rough";
        case MATERIAL_METALNESS:
            if (!self.surface) abort("drie: nodes: material_metalness() is for roughness_node and metalness_node");
            return "metal";
        case ENV_COLOR:
            if (!self.env) abort("drie: nodes: env_color() is for env_node");
            return "env";
        case INSTANCE_MATRIX:
            if (fragment) abort("drie: nodes: instance_matrix() is for the vertex stage (position_node)");
            return "ni.instance_mat";
    }
}

fn String NodeGen.texture(&self, Node n, NodeData* d) {
    if (self.stage != FRAGMENT) abort("drie: nodes: textures are read in the fragment stage (not in position_node)");
    NodeBuild* b = self.build;
    int k = -1;
    for (int i = 0; i < b.texture_count; i++) {
        if (b.textures[i] == d.texture) k = i;
    }
    if (k < 0) {
        if (b.texture_count >= NODE_TEXTURES) abort("drie: nodes: more than %d textures", NODE_TEXTURES);
        k = b.texture_count++;
        b.textures[k] = d.texture;
    }
    int slot = NODE_TEXTURES - 1 - k;
    if (d.kind == 1) {
        switch (self.lang) {
            case GLSL: return string::tformat("vec2(textureSize(tex_ext%d_tex_smp, 0))", slot);
            case MSL: return string::tformat("float2(tex_ext%d.get_width(), tex_ext%d.get_height())", slot, slot);
            case HLSL:
                String size = string::tformat("ts%d", n.id);
                self.line("uint2 %s;", size);
                self.line("tex_ext%d.GetDimensions(%s.x, %s.y);", slot, size, size);
                return string::tformat("float2(%s)", size);
        }
    }
    String uv = self.value_as({ d.args[0] }, VEC2);
    if (d.kind == 2) {
        String dx = self.value_as({ d.args[1] }, VEC2);
        String dy = self.value_as({ d.args[2] }, VEC2);
        switch (self.lang) {
            case GLSL: return string::tformat("textureGrad(tex_ext%d_tex_smp, %s, %s, %s)", slot, uv, dx, dy);
            case MSL: return string::tformat("tex_ext%d.sample(tex_smp, %s, gradient2d(%s, %s))", slot, uv, dx, dy);
            case HLSL: return string::tformat("tex_ext%d.SampleGrad(tex_smp, %s, %s, %s)", slot, uv, dx, dy);
        }
    }
    bool lod = d.args[1] != 0;
    String level = lod ? self.value_as({ d.args[1] }, FLOAT) : "";
    switch (self.lang) {
        case GLSL:
            if (lod) return string::tformat("textureLod(tex_ext%d_tex_smp, %s, %s)", slot, uv, level);
            return string::tformat("texture(tex_ext%d_tex_smp, %s)", slot, uv);
        case MSL:
            if (lod) return string::tformat("tex_ext%d.sample(tex_smp, %s, level(%s))", slot, uv, level);
            return string::tformat("tex_ext%d.sample(tex_smp, %s)", slot, uv);
        case HLSL:
            if (lod) return string::tformat("tex_ext%d.SampleLevel(tex_smp, %s, %s)", slot, uv, level);
            return string::tformat("tex_ext%d.Sample(tex_smp, %s)", slot, uv);
    }
}

fn String NodeGen.uniform(&self, Node n) {
    NodeBuild* b = self.build;
    int at = -1;
    int p;
    foreach (u : b.uniforms) {
        if (u.id == n.id) at = p;
        p += u.type() == MAT4 ? 4 : 1;
    }
    if (at < 0) {
        at = p;
        p += n.type() == MAT4 ? 4 : 1;
        if (p > NODE_PARAMS) abort("drie: nodes: more than %d vec4s of uniforms", NODE_PARAMS);
        b.uniforms.push(n);
    }
    String np = self.lang == MSL ? "nu.node_params" : "node_params";
    NodeType t = n.type();
    switch (t) {
        case MAT4:
            // (its columns; HLSL's constructor takes rows, which SPIRV-Cross's mul(v, m) reads as columns)
            return string::tformat(
                "%s(%s[%d], %s[%d], %s[%d], %s[%d])", self.type_name(MAT4), np, at, np, at + 1, np, at + 2, np, at + 3
            );
        default:
            // (integers and bools as floats: Node.set)
            int c = components(t);
            String x = c == 4 ? string::tformat("%s[%d]", np, at) : string::tformat("%s[%d].%s", np, at, "xyzw"[:(sz)c]);
            return self.convert(x, type_of(FLOAT, c), t);
    }
}

fn String NodeGen.fn_name(&self, NodeFn f) {
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

// Whether `n`'s value reads a variable (then it is worked out again after an assignment).
fn bool NodeGen.reads_variable(&self, Node n) {
    char k = self.reads_var[n.id];
    if (k) return k == 2;
    NodeData* d = node_data(n);
    bool reads;
    switch (d.op) {
        case VAR:
            reads = true;
        case SEQ:
        case CONST:
        case UNIFORM:
        case BUILTIN:
            reads = false;
        default:
            for (int i = 0; i < 4 && !reads; i++) {
                if (d.args[i] && self.reads_variable({ d.args[i] })) reads = true;
            }
    }
    self.reads_var[n.id] = reads ? 2 : 1;
    return reads;
}

// Whether `n` was printed where it still can be read, with the value it has now.
fn bool NodeGen.printed(&self, Node n) {
    uint at = self.memo[n.id];
    if (!at || !self.scope_open[(sz)at - 1]) return false;
    if (node_data(n).op == VAR) return true;
    return !self.reads_variable(n) || self.memo_epoch[n.id] == self.epoch;
}

// `n` printed, here: its (new) name.
fn String NodeGen.mark(&self, Node n) {
    if (!self.local[n.id]) self.local[n.id] = ++self.locals;
    self.memo[n.id] = self.scope;
    self.memo_epoch[n.id] = self.epoch;
    self.serial[n.id]++;
    return self.name(n);
}

fn void NodeGen.open_scope(&self) {
    self.scope_open.push(true);
    self.scope = (uint)self.scope_open.len();
    self.epoch++;
    self.indent++;
}

fn void NodeGen.close_scope(&self, uint outer) {
    self.scope_open[(sz)self.scope - 1] = false;
    self.scope = outer;
    self.epoch++;
    self.indent--;
}

// The statements of a block, each in turn.
fn void NodeGen.statements(&self, uint first) {
    for (uint s = first; s; s = node_data({ s }).next) self.statement({ s });
}

// A block of its own: `{ ... }`'s inside.
fn void NodeGen.block(&self, uint first) {
    uint outer = self.scope;
    self.open_scope();
    self.statements(first);
    self.close_scope(outer);
}

// Where an assignment writes: a variable or some of its components.
fn String NodeGen.lvalue(&self, Node n) {
    NodeData* d = node_data(n);
    if (d.op == VAR) return self.emit(n);
    String v = self.emit({ d.args[0] });
    return string::tformat("%s.%s", v, (String)d.swizzle[:d.count]);
}

fn void NodeGen.statement(&self, Node s) {
    NodeData d = *node_data(s);
    switch (d.op) {
        case EVAL:
            self.emit({ d.args[0] });
        case ASSIGN:
            Node target = { d.args[0] };
            String value = self.value_as({ d.args[1] }, target.type());
            self.line("%s = %s;", self.lvalue(target), value);
            self.epoch++;
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
            uint outer = self.scope;
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
                // (the index belongs to the loop's scope)
                self.epoch++;
                self.scope_open.push(true);
                self.scope = (uint)self.scope_open.len();
                String i = self.mark(index);
                self.line("for (%s %s = %s; %s %s %s; %s%s) {", self.type_name(t), i, from, i, ops[how.ordinal], to, i, how.ordinal < 2 ? "++" : "--");
                self.indent++;
            }
            self.statements(d.args[3]);
            self.close_scope(outer);
            self.line("}");
        case JUMP:
            switch (d.kind) {
                case 0: self.line("break;");
                case 1: self.line("continue;");
                default:
                    if (self.stage != FRAGMENT) abort("drie: nodes: discard is for the fragment stage");
                    self.discards = true;
                    self.line(self.lang == MSL ? "discard_fragment();" : "discard;");
            }
        default:
            // an expression as a statement: worked out here
            self.emit(s);
    }
}

// Prints node `n` (and what it reads) into the body, once where it can be read again; its
// value is in name(n).
fn String NodeGen.emit(&self, Node n) {
    if (self.stage == VERTEX_OUT && self.kept[n.id].len) return self.kept[n.id];
    if (self.printed(n)) return self.name(n);
    NodeData d = *node_data(n);
    String e;
    switch (d.op) {
        case CONST:
            if (d.type == BOOL) {
                e = d.value[0] != 0 ? "true" : "false";
            } else if (components(d.type) == 1) {
                e = component_literal(d.type, &d.value, 0);
            } else {
                e = string::tformat("%s(", self.type_name(d.type));
                for (int i = 0; i < components(d.type); i++) e = string::tformat("%s%s%s", e, i ? ", " : "", component_literal(d.type, &d.value, i));
                e = string::tformat("%s)", e);
            }
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
            abort("drie: nodes: a statement (%s) where a value goes", d.op);
        case UNIFORM:
            e = self.uniform(n);
        case BUILTIN:
            e = self.builtin(n, (NodeBuiltin)d.kind);
        case TEXTURE:
            e = self.texture(n, &d);
        case VARYING:
            e = self.stage == FRAGMENT ? self.read_varying(n) : self.emit({ d.args[0] });
        case BINARY:
            Node a = { d.args[0] };
            Node b = { d.args[1] };
            NodeBinary op = (NodeBinary)d.kind;
            String[*] ops = { "+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!=", "&&", "||", "%", "&", "|", "^", "<<", ">>", "!=" };
            NodeType ta = a.type();
            NodeType tb = b.type();
            switch (op) {
                case MUL:
                    if (ta == MAT4 && tb == MAT4) {
                        e = self.mat_mul(self.emit(a), self.emit(b));
                        break;
                    }
                    if (ta == MAT4 || tb == MAT4) {
                        NodeType v = ta == MAT4 ? tb : ta;
                        String x = ta == MAT4 ? self.emit(a) : self.value_as(a, components(v) == 1 ? FLOAT : VEC4);
                        String y = tb == MAT4 ? self.emit(b) : self.value_as(b, components(v) == 1 ? FLOAT : VEC4);
                        e = components(v) == 1 ? string::tformat("(%s * %s)", x, y) : self.mat_mul(x, y);
                        break;
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
                    // (a number with a vector as it is; both of the result's kind)
                    NodeScalar s = scalar_of(d.type);
                    String x = self.value_as(a, type_of(s, components(ta) == 1 ? 1 : components(d.type)));
                    String y = self.value_as(b, type_of(s, components(tb) == 1 ? 1 : components(d.type)));
                    if (op == REM && s == FLOAT) {
                        e = string::tformat("(%s - %s * floor(%s / %s))", x, y, x, y);
                    } else {
                        e = string::tformat("(%s %s %s)", x, ops[op.ordinal], y);
                    }
                case LT:
                case LE:
                case GT:
                case GE:
                case EQ:
                case NE:
                    NodeType t = wider(ta, tb);
                    e = string::tformat("(%s %s %s)", self.value_as(a, t), ops[op.ordinal], self.value_as(b, t));
                case AND:
                case OR:
                case XOR:
                    e = string::tformat("(%s %s %s)", self.value_as(a, BOOL), ops[op.ordinal], self.value_as(b, BOOL));
            }
        case CALL:
            e = self.call(&d);
        case SELECT:
            e = string::tformat(
                "(%s ? %s : %s)", self.value_as({ d.args[0] }, BOOL), self.value_as({ d.args[1] }, d.type), self.value_as({ d.args[2] }, d.type)
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
                Node p = { d.args[i] };
                NodeType pt = p.type();
                String x = self.value_as(p, type_of(scalar_of(d.type), components(pt)));
                e = string::tformat("%s%s%s", e, i ? ", " : "", x);
            }
            e = string::tformat("%s)", e);
        case CONVERT:
            e = self.value_as({ d.args[0] }, d.type);
        case NOT:
            if (d.kind == 1) {
                e = string::tformat("(~%s)", self.emit({ d.args[0] }));
            } else {
                e = string::tformat("(!%s)", self.value_as({ d.args[0] }, BOOL));
            }
        case INPUT:
            if (d.res[0] == NodeSource.FS.ordinal && d.res[1] == fs_row("ssr_params")) {
                int c = components(d.type);
                int off = d.res[2];
                String sp = self.lang == MSL ? "fu.ssr_params" : "ssr_params";
                String x = c == 4 && off == 0 ? sp : string::tformat("%s.%s", sp, "xyzw"[off:c]);
                e = self.convert(x, type_of(FLOAT, c), d.type);
            } else if (d.res[0] == NodeSource.FS.ordinal && d.res[1] == fs_row("ssr_view_proj")) {
                e = self.lang == MSL ? "fu.ssr_view_proj" : "ssr_view_proj";
            } else {
                self.unsupported = "engine uniform INPUT";
                e = self.zero(d.type);
            }
        case SAMPLE:
            if (d.res[0] == shaders::VIEW_SSR_DEPTH) {
                NodeSampling sk = (NodeSampling)d.kind;
                if (sk == SIZE) {
                    switch (self.lang) {
                        case GLSL: e = "textureSize(ssr_depth_tex_smp, 0)";
                        case MSL: e = "int2(ssr_depth.get_width(), ssr_depth.get_height())";
                        case HLSL:
                            String sz_name = string::tformat("ssz%d", n.id);
                            self.line("uint2 %s;", sz_name);
                            self.line("ssr_depth.GetDimensions(%s.x, %s.y);", sz_name, sz_name);
                            e = string::tformat("int2(%s)", sz_name);
                    }
                } else if (sk == FETCH) {
                    String tc = self.value_as({ d.args[0] }, IVEC2);
                    switch (self.lang) {
                        case GLSL: e = string::tformat("texelFetch(ssr_depth_tex_smp, %s, 0)", tc);
                        case MSL: e = string::tformat("ssr_depth.read(uint2(%s), 0)", tc);
                        case HLSL: e = string::tformat("ssr_depth.Load(int3(%s, 0))", tc);
                    }
                } else {
                    self.unsupported = "unsupported ssr_depth sample";
                    e = self.zero(d.type);
                }
            } else {
                self.unsupported = "engine texture SAMPLE";
                e = self.zero(d.type);
            }
        case PARAM:
            if (self.cur_func != d.res[0]) abort("drie: nodes: a parameter of %s outside it", node_funcs[d.res[0]].name);
            return string::tformat("p%d", d.kind);
        case CALL_FN:
            int fi = (int)bitcast(d.value[0], uint);
            self.emit_func(fi);
            NodeFunc* fd = node_funcs.get_ref(fi);
            DString call;
            call.tinit();
            if (self.lang == MSL) {
                call.appendf("%s(ni, nu", fd.name);
                if (self.stage == FRAGMENT) {
                    call.append(", tex_ext0, tex_smp, tex_ext1, tex_ext2, tex_ext3, tex_ext4, tex_ext5");
                }
            } else {
                call.appendf("%s(ni", fd.name);
            }
            uint arg0 = bitcast(d.value[1], uint);
            for (int i = 0; i < fd.count; i++) {
                call.appendf(", %s", self.value_as({ node_args[(usz)arg0 + (usz)i] }, fd.params[i]));
            }
            call.append_char(')');
            e = call.str_view();
            if (d.type == VOID) {
                self.line("%s;", e);
                self.mark(n);
                return "";
            }
        case STORAGE:
        case ATTRIBUTE:
        case SYSTEM:
        case BIND:
            abort("drie: nodes: a %s node is for whole programs (nodes_program.c3), not a material's hooks", d.op);
    }
    String name = self.mark(n);
    self.line("%s %s = %s;", self.type_name(d.type), name, e);
    return name;
}

fn String NodeGen.call(&self, NodeData* d) {
    NodeFn f = (NodeFn)d.kind;
    NodeType t = d.type;
    Node a = { d.args[0] };
    Node b = { d.args[1] };
    Node c = { d.args[2] };
    switch (f) {
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
        case BITS_AS_UINT:
            if (self.lang == GLSL) return string::tformat("floatBitsToUint(%s)", self.value_as(a, type_of(FLOAT, components(t))));
            if (self.lang == MSL) return string::tformat("as_type<%s>(%s)", self.type_name(t), self.value_as(a, type_of(FLOAT, components(t))));
            return string::tformat("asuint(%s)", self.value_as(a, type_of(FLOAT, components(t))));
        case BITS_AS_FLOAT:
            if (self.lang == GLSL) return string::tformat("uintBitsToFloat(%s)", self.value_as(a, type_of(UINT, components(t))));
            if (self.lang == MSL) return string::tformat("as_type<%s>(%s)", self.type_name(t), self.value_as(a, type_of(UINT, components(t))));
            return string::tformat("asfloat(%s)", self.value_as(a, type_of(UINT, components(t))));
        default:
            if (d.count == 1) return string::tformat("%s(%s)", self.fn_name(f), self.value_as(a, t));
            if (d.count == 2) return string::tformat("%s(%s, %s)", self.fn_name(f), self.value_as(a, t), self.value_as(b, t));
            return string::tformat("%s(%s, %s, %s)", self.fn_name(f), self.value_as(a, t), self.value_as(b, t), self.value_as(c, t));
    }
}

// ---- the program's shaders

fn String msl_param_after(String sig, String prefix) @local {
    sz? p = sig.index_of(prefix);
    if (catch p) return "";
    sz start = p + prefix.len;
    sz end = start;
    while (end < sig.len && sig[end] != ',' && sig[end] != ')') end++;
    return sig[start..end - 1];
}

// `src` with the body of function `name` replaced by `body`; sokol-shdc (SPIRV-Cross) prints
// a function as its signature, then "{" and "}" on lines of their own.
fn String splice_body(String src, String name, String body) @local {
    String head = string::tformat("\nvoid %s(", name);
    sz? at = src.index_of(head);
    if (catch at) {
        abort("drie: this build of the material shader has no %s (rebuild it with drie's shader_core.glsl)", name);
    }
    sz? open = src[at..].index_of("\n{\n");
    sz? close = src[at..].index_of("\n}\n");
    if (catch open) abort("drie: %s's body not found", name);
    if (catch close) abort("drie: %s's body not found", name);
    String sig = src[at..at + open];
    DString preamble;
    preamble.tinit();
    if (body.len) {
        String nfp = msl_param_after(sig, "constant node_fs_params& ");
        if (nfp.len) preamble.appendf("    constant node_fs_params& nu = %s;\n", nfp);
        String nvp = msl_param_after(sig, "constant node_vs_params& ");
        if (nvp.len) preamble.appendf("    constant node_vs_params& nu = %s;\n", nvp);
        String fp = msl_param_after(sig, "constant fs_params& ");
        if (fp.len) preamble.appendf("    constant fs_params& fu = %s;\n", fp);
    }
    sz from = at + open + 3;
    sz to = at + close + 1;
    return string::tformat("%s%s%s%s", src[:from], preamble.str_view(), body, src[to..]);
}

// Inserts `extra` immediately before `\nvoid <name>(`.
fn String insert_before(String src, String name, String extra) @local {
    if (!extra.len) return src;
    String head = string::tformat("\nvoid %s(", name);
    sz? at = src.index_of(head);
    if (catch at) return src;
    return string::tformat("%s\n%s%s", src[:at], extra, src[at..]);
}

// `desc` (the engine's or the application's program) with the hooks' bodies swapped, its
// sources in temporary memory.
fn SgShaderDesc NodeProgram.spliced(&self, NodeVariant v, SgShaderDesc desc) {
    if (self.unsupported.len) {
        abort("drie: a node material uses %s, which the application's material shader (Renderer.shaders) has no hook for: only generated programs draw it", self.unsupported);
    }
    // (DRIE_PROGRAM_SALT: a statement that changes the compiled code and no picture, so the
    // driver has none of it cached; for timing a cold start)
    String vertex_salt;
    String color_salt;
    if (try salt = env::tget_var("DRIE_PROGRAM_SALT")) {
        float far = 1e7f + (float)(salt.hash() & 0xffff);
        vertex_salt = string::tformat("    if (local.x == %.1f) local.x = 0.0;\n", far);
        color_salt = string::tformat("    if (base.x == %.1f) base.x = 0.0;\n", far);
    }
    String vs = ((ZString)desc.vertex_func.source).str_view();
    vs = insert_before(vs, "node_vertex", self.vertex_funcs);
    vs = splice_body(vs, "node_vertex", vertex_salt.len ? vertex_salt.tconcat(self.vertex_body) : self.vertex_body);
    desc.vertex_func.source = vs.zstr_tcopy();
    String color_body = color_salt.len ? color_salt.tconcat(self.color_body) : self.color_body;
    if (v == SHADOW && self.cuts) {
        // (shadow_nodes: node_color only)
        String fs = ((ZString)desc.fragment_func.source).str_view();
        fs = insert_before(fs, "node_color", self.frag_funcs);
        fs = splice_body(fs, "node_color", color_body);
        desc.fragment_func.source = fs.zstr_tcopy();
    } else if (v != DEPTH && v != SHADOW) {
        String fs = ((ZString)desc.fragment_func.source).str_view();
        fs = insert_before(fs, "node_color", self.frag_funcs);
        if (fs.contains("\nvoid node_color(")) fs = splice_body(fs, "node_color", color_body);
        if (fs.contains("\nvoid node_normal(")) fs = splice_body(fs, "node_normal", self.normal_body);
        if (fs.contains("\nvoid node_surface(")) fs = splice_body(fs, "node_surface", self.surface_body);
        if (fs.contains("\nvoid node_output(")) fs = splice_body(fs, "node_output", self.output_body);
        if (fs.contains("\nvoid node_env(")) fs = splice_body(fs, "node_env", self.env_body);
        if (fs.contains("\nvoid node_specular(")) fs = splice_body(fs, "node_specular", self.specular_body);
        desc.fragment_func.source = fs.zstr_tcopy();
    }
    desc.label = "drie-node-material";
    return desc;
}

// Variant `v` of the program, from the engine's (or the application's, Renderer.shaders)
// description of that program: its sources with the hooks' bodies swapped.
fn SgShader NodeProgram.make_shader(&self, NodeVariant v, SgShaderDesc desc) {
    @pool() {
        desc = self.spliced(v, desc);
        SgShader shader = sg::make_shader(&desc);
        if (sg::query_shader_state(shader) != VALID) {
            io::eprintfn("drie: a node material's shader failed to build. Its hooks:");
            io::eprintfn("node_vertex:\n%snode_color:\n%snode_normal:\n%snode_surface:\n%snode_output:\n%snode_env:\n%snode_specular:\n%s",
                self.vertex_body, self.color_body, self.normal_body, self.surface_body, self.output_body, self.env_body,
                self.specular_body);
            abort("drie: a node material's shader failed to build");
        }
        return shader;
    };
}
