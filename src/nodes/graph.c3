// Shader node graph IR (`Node`, `NodeData`, `NodeFunction`), operator/swizzle methods, and
// per-material graph builds (`NodeBuild`, `node_build_of`).
//
// Nodes are immutable uint handles (0 = none) stored in a monotonic arena; only `UNIFORM` nodes
// mutate in place via `Node.set*`, updating parameters without rebuilding shaders. A material
// binds at most 6 custom node textures (`NODE_TEXTURES`) across its 15 `*_node` slots
// (`NODE_SLOTS`), and `node_release` reclaims transient nodes made during program builds.
// Constructors live in `drie::nodes` (src/nodes/tsl.c3); program analysis and backend codegen
// live in src/nodes/program.c3 and src/nodes/printer.c3; material integration lives in
// src/material/material_key.c3 and src/material/material_graph.c3.
module drie;
import std::collections::list, std::collections::map, std::io, std::os::env, sokol, drie::nodes;

// Handle to an immutable shader graph node (0 = none); only uniform values mutate via `Node.set*`.
// Constructed through `drie::nodes`, e.g.:
//
//   Node wave = nodes::sin(nodes::position_local().y() * 8 + nodes::time() * 2);
//   mat.color_node = nodes::mix(nodes::color(0x2266ff), nodes::color(0xffffff), wave * 0.5f + 0.5f);
//   mat.position_node = nodes::position_local() + nodes::normal_local() * (wave * 0.05f);
struct Node {
    uint id;
}

// A graph function (`nodes::function`), printed once in each shader stage that calls it.
struct NodeFunction {
    int id;
}

// Returns the node representing parameter `i` of this function.
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

// Value type carried by a node (scalars, vectors, 3x3/4x4 matrices, or `VOID` for statements).
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
    VOID, // a statement's
    MAT3,
}

// What a type's components are.
enum NodeScalar : char {
    FLOAT,
    INT,
    UINT,
    BOOL,
}

// Operation tag identifying the expression, statement, or engine input encoded in `NodeData`.
enum NodeOp : char {
    CONST,     // value
    UNIFORM,   // value (Node.set)
    BUILTIN,   // fn: a NodeBuiltin
    TEXTURE,   // texture; args[0] = uv, args[1] = the level (0: from the derivatives); kind 1:
               // its size instead (a vec2); kind 2: with the gradients args[1] and args[2]
    VARYING,   // args[0], worked out per vertex
    BINARY,    // fn: a NodeBinary
    CALL,      // fn: a NodeFn, args
    SELECT,    // args[0] ? args[1] : args[2]
    SWIZZLE,   // args[0]'s components swizzle[0..count)
    JOIN,      // a vector of args[0..count)
    CONVERT,   // args[0], as type
    NOT,       // kind 0: !args[0]; 1: ~args[0]
               // statements, of type VOID, in blocks (the first statement; each one's `next`):
    VAR,       // a variable, of args[0] at first (0: zeros); declared where it is first read
    ASSIGN,    // args[0] (a VAR, or a swizzle of one) = args[1]
    IF,        // if (args[0]) the block args[1] else the block args[2]
    LOOP,      // kind: a NodeLoop; args[0] the index (a VAR), args[1] from, args[2] to, args[3] the block
    JUMP,      // kind 0: break, 1: continue, 2: discard (the fragment stage)
    EVAL,      // args[0] worked out here (a VAR's declaration, a SEQ that returns nothing)
               // and the one that gives a value
    SEQ,       // the block args[0], then args[1] (0: nothing, a VOID)
               // Engine program inputs (src/nodes/program.c3):
    INPUT,     // rows of an engine uniform block: res[0] = its NodeSource, res[1] = the first row,
               // res[2] = the component of the row a vector starts at; args[0] an index into an
               // array (0: none) of value[0] rows a step, value[1] steps long (as bits)
    STORAGE,   // element args[0] of a storage buffer (res[0]: the view slot) of vec4s, uvec4s or uints
    SAMPLE,    // an engine texture: kind = a NodeSampling; res[0] = the view slot, res[1] = the
               // sampler slot, res[2] = a NodeTextureKind; args = coordinates, then the level, the
               // gradients or the depth compared against
    ATTRIBUTE, // vertex attribute kind (its location)
    SYSTEM,    // kind = a NodeSystem
    PARAM,     // a function's parameter kind
    CALL_FN,   // the function value[0] (as bits, in node_funcs) of the count arguments in node_args
               // from value[1] (as bits)
    BIND,      // a statement: the builtin kind is args[0] from here on
}

// What SAMPLE does with its texture.
enum NodeSampling : char {
    SAMPLE,  // filtered, the level from the derivatives
    LEVEL,   // at level args[1]
    GRAD,    // with the derivatives args[1] and args[2]
    FETCH,   // texel args[0] (integers) of level args[1] (an int)
    SIZE,    // the size of level args[0] (an int): an ivec2, or an ivec3 for 3D textures and arrays
    COMPARE, // a depth texture's comparison sampler: how much of the 2x2 footprint at args[0]
             // is nearer than args[1]
    LEVELS,  // how many mip levels the texture has (an int)
}

// Texture dimensionality and sample mode for `NodeOp.SAMPLE`.
enum NodeTextureKind : char {
    TEX_2D,
    TEX_3D,
    TEX_ARRAY,        // 2D layers: coordinates (u, v, layer)
    TEX_DEPTH,        // 2D depths, read with COMPARE
    TEX_UNFILTERABLE, // 2D unfilterable float (e.g. depth texture read with FETCH)
}

// The stage's own inputs.
enum NodeSystem : char {
    INSTANCE_ID,  // uint (vertex)
    VERTEX_ID,    // uint (vertex)
    FRAG_COORD,   // vec4 (fragment): the backend's own, as sokol-shdc hands it on
    FRONT_FACING, // bool (fragment)
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

// Shader stage builtins bound dynamically by material programs for user node graphs.
enum NodeBuiltin : char {
    UV,
    UV1,
    POSITION_LOCAL,     // in the model (with the instance's matrix), as three's positionLocal
    POSITION_WORLD,
    NORMAL_LOCAL,
    NORMAL_WORLD,       // the shaded normal (fragment: after the normal map and normal_node)
    NORMAL_GEOMETRY,    // the surface's own, facing the camera
    CAMERA_POSITION,
    VIEW_DIRECTION,     // from the point toward the camera, normalized
    TIME,               // Renderer.time (s)
    VERTEX_COLOR,       // rgba
    FRAG_COORD,         // the pixel's position in the target (fragment)
    FRONT_FACING,       // bool (fragment)
    MODEL_MATRIX,       // the mesh's matrix_world (a billboard's as it is, before it is turned)
    VIEW_MATRIX,
    PROJECTION_MATRIX,
    OUTPUT,             // the lit color, rgba (in output_node)
    ENV_DIRECTION,      // the direction the surface reflects, in the world (in env_node)
    ENV_ROUGHNESS,      // the roughness it reflects with (in env_node)
    INSTANCE_INDEX,     // the instance's index in its mesh (0 without instances), a float
                        // the material's own value at the step a graph replaces, for a graph that changes it:
    MATERIAL_COLOR,     // rgba (in color_node, opacity_node): the color after its maps, vertex
                        // and instance colors, and its opacity
    MATERIAL_ROUGHNESS, // (in roughness_node, metalness_node and the surface step's others):
    MATERIAL_METALNESS, // after the material's maps
    ENV_COLOR,          // the environment's reflection along ENV_DIRECTION (in env_node)
    INSTANCE_MATRIX,    // the instance's matrix (identity without instances; vertex stage)
}

// Binary arithmetic, comparison, logical, and bitwise operators for `NodeOp.BINARY`.
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
    REM, // %: of floats as mod (floored), of integers as their remainder
    BIT_AND,
    BIT_OR,
    BIT_XOR,
    SHL,
    SHR,
    XOR, // of bools
}

// Built-in shader math, derivative, matrix, and bit-reinterpretation intrinsics for `NodeOp.CALL`.
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

// Internal arena payload for a single shader graph node.
struct NodeData {
    NodeOp op;
    NodeType type;
    char kind;
    char count;
    uint[4] args;
    char[4] swizzle;
    float[16] value; // (an int's or a uint's: its bits)
    Texture* texture;
    int[3] res; // INPUT, STORAGE, SAMPLE: where they read
    uint next; // a statement's: the one after it in its block (0: the last)
}

List{NodeData} node_graph @local;

// Returns the arena record for node `n`, aborting if `n.id` is 0 or out of bounds.
fn NodeData* node_data(Node n) {
    if (n.id == 0 || (sz)n.id > node_graph.len()) abort("drie: nodes: not a node (%d)", n.id);
    return &node_graph[(sz)n.id - 1];
}

// Appends `d` to the global node arena and returns its 1-based `Node` handle.
fn Node add_node(NodeData d) {
    if (!node_graph.is_initialized()) node_graph.init(mem, 256);
    node_graph.push(d);
    return { (uint)node_graph.len() };
}

// Returns the `NodeType` produced by this node.
fn NodeType Node.type(self) => node_data(self).type;

// How many nodes there are: a mark node_release goes back to.
fn uint node_count() => (uint)node_graph.len();

// Drops the nodes made after `mark` (node_count), with the functions (node_func) made since:
// what building a program needed and nothing else holds. The nodes before it never read the
// ones after.
fn void node_release(uint mark) {
    while (node_graph.len() > (sz)mark) node_graph.remove_last();
    release_funcs(mark);
}

// Returns the component count of `t` (1..4 scalars/vectors, 9 MAT3, 16 MAT4, 0 VOID).
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

// Returns the element scalar kind (`FLOAT`, `INT`, `UINT`, or `BOOL`) of `t`.
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

// Returns true if `t` is `MAT3` or `MAT4`.
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
    int na       = components(a);
    int nb       = components(b);
    if (na != nb && na != 1 && nb != 1) {
        abort("drie: nodes: a %s and a %s don't combine (convert one: nodes::vec3(...), .xyz()...)", a, b);
    }
    return type_of(s, na > nb ? na : nb);
}

// ---- Node constructors

// Creates a constant node of `type` carrying `value`.
fn Node node_const(NodeType type, float[16] value) => add_node({ .op = CONST, .type = type, .value = value });

// Creates a binary operator node (`op`) over `a` and `b`, inferring the result type.
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
            if (scalar_of(
                ta
            ) == FLOAT || scalar_of(ta) == BOOL || scalar_of(tb) == FLOAT) abort("drie: nodes: shifts take integers");
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

// Creates a built-in intrinsic call node (`f`) with 1 to 3 arguments.
fn Node node_call(NodeFn f, Node a, Node b = { }, Node c = { }) {
    NodeType ta = a.type();
    if (f == TRANSPOSE || f == DETERMINANT) {
        if (!is_matrix(ta)) abort("drie: nodes: %s takes a matrix", f);
        return add_node(
            {
                .op = CALL,
                .type = f == TRANSPOSE ? ta : FLOAT,
                .kind = (char)f.ordinal,
                .count = 1,
                .args = { a.id, 0, 0, 0 }
            }
        );
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
    // (but for abs, min, max, mod, clamp and the bit casts, functions of floats: integers
    // converted)
    switch (f) {
        case ABS:
        case MIN:
        case MAX:
        case MOD:
        case CLAMP:
        case BITS_AS_UINT:
        case BITS_AS_FLOAT:
            break;
        default:
            t = type_of(FLOAT, components(t));
    }
    return add_node(
        { .op = CALL, .type = t, .kind = (char)f.ordinal, .count = (char)count, .args = { a.id, b.id, c.id, 0 } }
    );
}

// Converts node `a` to type `t`, returning `a` unchanged if its type already matches.
fn Node node_convert(Node a, NodeType t) {
    if (a.type() == t) return a;
    return add_node({ .op = CONVERT, .type = t, .count = 1, .args = { a.id, 0, 0, 0 } });
}

fn Node node_swizzle(Node a, String s) @local {
    if (s.len < 1 || s.len > 4) abort("drie: nodes: a swizzle of 1 to 4 components, not \"%s\"", s);
    NodeScalar sa = scalar_of(a.type());
    NodeData d    = { .op = SWIZZLE, .type = type_of(sa, (int)s.len), .count = (char)s.len, .args = { a.id, 0, 0, 0 } };
    int have      = components(a.type());
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

// Concatenates 1 to 4 scalar/vector nodes into a vector of type `t`.
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

// ---- Statements

// A variable (three's toVar()), of `init` at first ({}: zeros of type `t`).
fn Node node_var(NodeType t, Node init = { }) {
    if (init.id && init.type() != t) init = node_convert(init, t);
    return add_node({ .op = VAR, .type = t, .count = 1, .args = { init.id, 0, 0, 0 } });
}

// Creates an assignment statement writing `value` into variable (or variable swizzle) `target`.
fn Node node_assign(Node target, Node value) {
    NodeData* t = node_data(target);
    // (a varying of a variable: the variable, in the vertex stage)
    if (t.op == VARYING && node_data({ t.args[0] }).op == VAR) target = { t.args[0] };
    t = node_data(target);
    if (t.op != VAR && !(t.op == SWIZZLE && node_data(
        { t.args[0] }
    ).op == VAR)) abort("drie: nodes: assign() to a variable (or its components)");
    return add_node({ .op = ASSIGN, .type = VOID, .count = 2, .args = { target.id, value.id, 0, 0 } });
}

// Creates a conditional statement running `then` when `cond` holds, and optional `otherwise`.
fn Node node_if(Node cond, Node then, Node otherwise = { }) {
    return add_node({ .op = IF, .type = VOID, .count = 3, .args = { cond.id, then.id, otherwise.id, 0 } });
}

// A loop of the block `body` with `index` (a VAR of a number) from `from`, while `cond` holds
// against `to`; with WHILE, while `to` (a bool) holds.
fn Node node_loop(NodeLoop cond, Node index, Node from, Node to, Node body) {
    if (cond != WHILE && node_data(index).op != VAR) abort("drie: nodes: a loop's index is a variable");
    return add_node(
        {
            .op = LOOP,
            .type = VOID,
            .kind = (char)cond.ordinal,
            .count = 4,
            .args = { index.id, from.id, to.id, body.id }
        }
    );
}

// Creates a control-flow jump statement (`0` = break, `1` = continue, `2` = discard).
fn Node node_jump(int kind) => add_node({ .op = JUMP, .type = VOID, .kind = (char)kind });

// A statement of an expression: worked out at this point.
fn Node node_eval(Node x) => add_node({ .op = EVAL, .type = VOID, .count = 1, .args = { x.id, 0, 0, 0 } });

// Statements in a block: chained in order; the block is its first one ({}: empty). A block
// among them (a statement already chained to another one) is worked out in its place, its
// statements still chained as they were; the same statements in the same order are chained
// again as they are.
fn Node node_block(Node[] stmts) {
    Node first;
    Node last;
    foreach (i, st : stmts) {
        if (!st.id) continue;
        if (node_data(st).type != VOID) {
            st = node_eval(st);
        } else if (node_data(st).next) {
            uint after;
            for (sz j = i + 1; j < stmts.len && !after; j++) after = stmts[j].id;
            if (node_data(st).next != after) st = node_eval(node_seq(st));
        }
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
// (a block: worked out there, as in node_block)
fn Node node_append(Node last, Node stmt) {
    if (node_data(
        stmt
    ).type != VOID || node_data(stmt).next) stmt = node_eval(stmt.type() == VOID ? node_seq(stmt) : stmt);
    if (last.id) node_data(last).next = stmt.id;
    return stmt;
}

// The block `block` worked out, then `value` ({}: nothing): an expression with statements in
// it (a TSL Fn's body and what it returns).
fn Node node_seq(Node block, Node value = { }) {
    return add_node(
        { .op = SEQ, .type = value.id ? value.type() : VOID, .count = 2, .args = { block.id, value.id, 0, 0 } }
    );
}

// ---- Operators and methods

// Component-wise addition (`self + b`).
fn Node Node.add(self, Node b) @operator(+) => node_binary(ADD, self, b);
// Component-wise subtraction (`self - b`).
fn Node Node.sub(self, Node b) @operator(-) => node_binary(SUB, self, b);
// Component-wise or matrix multiplication (`self * b`).
fn Node Node.mul(self, Node b) @operator(*) => node_binary(MUL, self, b);
// Component-wise division (`self / b`).
fn Node Node.div(self, Node b) @operator(/) => node_binary(DIV, self, b);
// Adds float scalar `b` to `self` (commutative).
fn Node Node.add_f(self, float b) @operator_s(+) => node_binary(ADD, self, nodes::num(b));
// Multiplies `self` by float scalar `b` (commutative).
fn Node Node.mul_f(self, float b) @operator_s(*) => node_binary(MUL, self, nodes::num(b));
// Subtracts float scalar `b` from `self`.
fn Node Node.sub_f(self, float b) @operator(-) => node_binary(SUB, self, nodes::num(b));
// Subtracts `self` from float scalar `a`.
fn Node Node.rsub_f(self, float a) @operator_r(-) => node_binary(SUB, nodes::num(a), self);
// Divides `self` by float scalar `b`.
fn Node Node.div_f(self, float b) @operator(/) => node_binary(DIV, self, nodes::num(b));
// Divides float scalar `a` by `self`.
fn Node Node.rdiv_f(self, float a) @operator_r(/) => node_binary(DIV, nodes::num(a), self);
// Negates `self` (`self * -1`).
fn Node Node.negate(self) => node_binary(MUL, self, nodes::num(-1));
// Complement `1 - self`.
fn Node Node.one_minus(self) => node_binary(SUB, nodes::num(1), self);

// Scalar comparison `self < b`.
fn Node Node.less_than(self, Node b) => node_binary(LT, self, b);
// Scalar comparison `self <= b`.
fn Node Node.less_than_equal(self, Node b) => node_binary(LE, self, b);
// Scalar comparison `self > b`.
fn Node Node.greater_than(self, Node b) => node_binary(GT, self, b);
// Scalar comparison `self >= b`.
fn Node Node.greater_than_equal(self, Node b) => node_binary(GE, self, b);
// Scalar equality `self == b`.
fn Node Node.equal(self, Node b) => node_binary(EQ, self, b);
// Scalar inequality `self != b`.
fn Node Node.not_equal(self, Node b) => node_binary(NE, self, b);

// Extracts 1 to 4 components by swizzle string (`"x"`, `"xyz"`, `"zyx"`, etc.).
fn Node Node.swizzle(self, String s) => node_swizzle(self, s);
// Extracts the `.x` component.
fn Node Node.x(self) => node_swizzle(self, "x");
// Extracts the `.y` component.
fn Node Node.y(self) => node_swizzle(self, "y");
// Extracts the `.z` component.
fn Node Node.z(self) => node_swizzle(self, "z");
// Extracts the `.w` component.
fn Node Node.w(self) => node_swizzle(self, "w");
// Extracts the `.xy` components.
fn Node Node.xy(self) => node_swizzle(self, "xy");
// Extracts the `.xz` components.
fn Node Node.xz(self) => node_swizzle(self, "xz");
// Extracts the `.yz` components.
fn Node Node.yz(self) => node_swizzle(self, "yz");
// Extracts the `.xyz` components.
fn Node Node.xyz(self) => node_swizzle(self, "xyz");
// Extracts the `.xyz` color channels.
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

// Updates a vec3 uniform node's value from `v` without rebuilding shaders.
fn void Node.set_vec3(self, Vector3 v) => self.set(v.x, v.y, v.z);

// Updates a vec3 color uniform from sRGB hex `c`, converting to linear RGB.
fn void Node.set_color(self, Color c) => self.set_vec3(color_to_linear(c));

// Updates a mat4 uniform node's 16 column-major elements from `m` without rebuilding shaders.
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

// ---- Material builds

// The shading languages the graphs are printed in, one per sokol backend.
enum NodeLang : char {
    GLSL,
    MSL,
    HLSL,
    WGSL,
}

// Maximum number of custom textures a single node material may sample (`NODE_USER_VIEW + 0..5`).
const int NODE_TEXTURES = 6;

// Starts background shader+pipeline compilation; returns a job handle (< 0 if unsupported).
extern fn int sg_warm_begin(SgShaderDesc* shader, SgPipelineDesc* pipeline) @cname("sg_warm_begin");
// Polls background compile job status (0 = still running, non-zero = finished).
extern fn int sg_warm_state(int job) @cname("sg_warm_state");
// Releases resources for a finished background compile job.
extern fn void sg_warm_end(int job) @cname("sg_warm_end");
// Blocks until background compile `job` finishes.
extern fn void sg_warm_wait(int job) @cname("sg_warm_wait");
// Starts background shader-only compilation; returns a job handle (< 0 if unsupported).
extern fn int sg_warm_shader_begin(SgShaderDesc* shader) @cname("sg_warm_shader_begin");

// A new one each render (Renderer.render): what material_busy says holds for all of a
// render's passes, so that a mesh the depth prepass drew with the plain material is drawn
// with it in the main pass too.
uint node_epoch;

// A material's build: its graphs' program, and where its uniforms and textures go.
struct NodeBuild {
    Node[NODE_SLOTS] slots; // the material's *_node fields it was built from
    Material* material; // whose they are (its maps for nodes::material_map)
    // color_node or opacity_node can cut holes: a discard, or alpha (with the material's
    // alpha_test), which the shadow maps cut too
    bool cuts;
    List{Node} uniforms; // in their order in the USER block (a matrix takes four)
    Texture*[NODE_TEXTURES] textures; // node texture k: a generated program's view NODE_USER_VIEW + k
    int texture_count;
    // Generated programs (src/material/material_key.c3) keyed by MaterialKey (with `nodes` set).
    HashMap{KeyWords, MaterialProgram*} programs;
    // whether one of them is still compiling in the background (in render busy_epoch, see
    // node_epoch): then its draws take the plain material's, in every pass (see material_busy)
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

// Returns true if any of `mat`'s 15 `*_node` graph slots is set.
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
    NodeBuild* b           = mat.node_build;
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
    b.cuts = mat.mask_node.id || node_discards(
        mat.color_node
    ) || mat.opacity_node.id || (mat.color_node.id && mat.color_node.type() == VEC4);
    return b;
}

// Whether the graph of `root` (0: none) has a discard in it.
fn bool node_discards(Node root) @local {
    if (!root.id) return false;
    bool found;
    @pool() {
        bool[] seen = mem::temp_array(bool, node_graph.len() + 1);
        List{uint} todo;
        todo.tinit();
        todo.push(root.id);
        while (todo.len() && !found) {
            uint id = todo.pop()!!;
            if (!id || seen[id]) continue;
            seen[id] = true;
            NodeData* d = node_data({ id });
            if (d.op == JUMP && d.kind == 2) found = true;
            if (d.op == CALL_FN) {
                uint at = bitcast(d.value[1], uint);
                for (uint i = 0; i < d.count; i++) todo.push(node_args[(sz)at + (sz)i]);
                continue;
            }
            foreach (a : d.args) todo.push(a);
            todo.push(d.next);
        }
    };
    return found;
}