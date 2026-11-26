// `drie::nodes` shader graph constructors mirroring three.js TSL in snake_case.
//
// Expression macros accept either `Node` handles or numeric scalars via `to_node`, e.g.
// `nodes::mix(a, b, 0.5f)` or `nodes::vec3(x, 0, 1)`. Color helpers (`color`, `uniform_color`)
// convert sRGB hex to linear RGB; `function()` bodies cannot read stage builtins or varyings
// directly and receive them as parameters. Wraps the IR in src/nodes/graph.c3, program nodes in
// src/nodes/program.c3, and SSR depth helpers in src/material/layers.c3; paired with
// src/nodes/snippet.c3 for text expressions.
module drie::nodes;
import drie;

// A node of a number or a node as it is (for the macros below).
macro Node to_node(x) {
    $if $Typeof(x) == Node:
        return x;
    $else
        return num((float)x);
    $endif
}

// ---- Values and uniforms

// Float scalar constant node.
fn Node num(float x) {
    float[16] v;
    v[0] = x;
    return drie::node_const(FLOAT, v);
}

// Boolean constant node.
fn Node bool_(bool b) {
    float[16] v;
    v[0] = b ? 1 : 0;
    return drie::node_const(BOOL, v);
}

// A linear color from sRGB hex, as three's color(0xff8800).
fn Node color(Color c) {
    Vector3 l   = drie::color_to_linear(c);
    float[16] v = { [0] = l.x, [1] = l.y, [2] = l.z };
    return drie::node_const(VEC3, v);
}

// Constructs a `vec2` from scalars or nodes (a single argument splats to both lanes).
macro Node vec2(...) => vector(VEC2, ...$vaarg);
// Constructs a `vec3` from scalars or nodes (a single argument splats to all three lanes).
macro Node vec3(...) => vector(VEC3, ...$vaarg);
// Constructs a `vec4` from scalars or nodes (a single argument splats to all four lanes).
macro Node vec4(...) => vector(VEC4, ...$vaarg);

macro Node vector(NodeType t, ...) @local {
    $if $vaarg.len == 1:
        return drie::node_convert(to_node($vaarg[0]), t);
    $else
        Node[$vaarg.len] parts;
        $for var $i = 0; $i < $vaarg.len; $i++:
            parts[$i] = to_node($vaarg[$i]);
        $endfor
        return drie::node_join(&parts, t);
    $endif
}

// Mutable float uniform updated via `Node.set` without rebuilding shaders.
fn Node uniform(float x) {
    float[16] v = { [0] = x };
    return drie::add_node({ .op = UNIFORM, .type = FLOAT, .value = v });
}

// Mutable vec2 uniform updated via `Node.set` without rebuilding shaders.
fn Node uniform_vec2(float x, float y) {
    float[16] v = { [0] = x, [1] = y };
    return drie::add_node({ .op = UNIFORM, .type = VEC2, .value = v });
}

// Mutable vec3 uniform updated via `Node.set` or `Node.set_vec3` without rebuilding shaders.
fn Node uniform_vec3(Vector3 x) {
    float[16] v = { [0] = x.x, [1] = x.y, [2] = x.z };
    return drie::add_node({ .op = UNIFORM, .type = VEC3, .value = v });
}

// Mutable vec4 uniform updated via `Node.set` without rebuilding shaders.
fn Node uniform_vec4(float x, float y, float z, float w) {
    float[16] v = { [0] = x, [1] = y, [2] = z, [3] = w };
    return drie::add_node({ .op = UNIFORM, .type = VEC4, .value = v });
}

// Mutable vec3 color uniform initialized from sRGB hex `c` (stored as linear RGB).
fn Node uniform_color(Color c) => uniform_vec3(drie::color_to_linear(c));

// Mutable mat4 uniform updated via `Node.set_matrix` without rebuilding shaders.
fn Node uniform_matrix(Mat4 m) {
    Node n = drie::add_node({ .op = UNIFORM, .type = MAT4 });
    n.set_matrix(m);
    return n;
}

// ---- Builtins and textures

fn Node builtin(
    NodeBuiltin b,
    NodeType t
) @local => drie::add_node({ .op = BUILTIN, .type = t, .kind = (char)b.ordinal });

// Primary UV coordinate (`vec2`).
fn Node uv() => builtin(UV, VEC2);
// Secondary UV coordinate (`vec2`, e.g. lightmaps).
fn Node uv1() => builtin(UV1, VEC2);
// Vertex position in model space including instance transform (`vec3`, three's positionLocal).
fn Node position_local() => builtin(POSITION_LOCAL, VEC3);
// Surface position in world space (`vec3`).
fn Node position_world() => builtin(POSITION_WORLD, VEC3);
// Vertex normal in local/instance space (`vec3`).
fn Node normal_local() => builtin(NORMAL_LOCAL, VEC3);
// Shaded world-space normal (`vec3`; after normal/bump maps and `normal_node` in fragment stage).
fn Node normal_world() => builtin(NORMAL_WORLD, VEC3);
// Camera-facing geometric surface normal in world space (`vec3`).
fn Node normal_geometry() => builtin(NORMAL_GEOMETRY, VEC3);
// Active camera world-space position (`vec3`).
fn Node camera_position() => builtin(CAMERA_POSITION, VEC3);
// Normalized world-space direction from the surface point toward the camera (`vec3`).
fn Node view_direction() => builtin(VIEW_DIRECTION, VEC3);
// Renderer elapsed time in seconds (`float`).
fn Node time() => builtin(TIME, FLOAT);
// Interpolated vertex color (`vec4`, rgba).
fn Node vertex_color() => builtin(VERTEX_COLOR, VEC4);
// Fragment pixel coordinates (`vec2`, fragment stage only).
fn Node frag_coord() => builtin(FRAG_COORD, VEC2);
// True for front-facing fragments (`bool`, fragment stage only).
fn Node front_facing() => builtin(FRONT_FACING, BOOL);
// Mesh world transform matrix (`mat4`, pre-billboard orientation).
fn Node model_matrix() => builtin(MODEL_MATRIX, MAT4);
// Camera view matrix (`mat4`).
fn Node view_matrix() => builtin(VIEW_MATRIX, MAT4);
// Camera projection matrix (`mat4`).
fn Node projection_matrix() => builtin(PROJECTION_MATRIX, MAT4);
// Lit material color and alpha (`vec4`, three's output), available in `Material.output_node`.
fn Node output() => builtin(OUTPUT, VEC4);
// World-space reflection direction (`vec3`), available in `Material.env_node`.
fn Node env_direction() => builtin(ENV_DIRECTION, VEC3);
// Reflection roughness (`float`), available in `Material.env_node`.
fn Node env_roughness() => builtin(ENV_ROUGHNESS, FLOAT);
// The instance's index in its InstancedMesh (0 for a plain mesh), a float (three's instanceIndex).
fn Node instance_index() => builtin(INSTANCE_INDEX, FLOAT);

// Material color/alpha after maps, vertex colors, and opacity (`vec4`; `color_node`/`opacity_node`).
fn Node material_color() => builtin(MATERIAL_COLOR, VEC4);
// Material roughness after roughness/ARM maps (`float`; `roughness_node` and surface-step nodes).
fn Node material_roughness() => builtin(MATERIAL_ROUGHNESS, FLOAT);
// Material metalness after metalness/ARM maps (`float`; `metalness_node` and surface-step nodes).
fn Node material_metalness() => builtin(MATERIAL_METALNESS, FLOAT);
// In Material.env_node: what the environment itself reflects along env_direction() at
// env_roughness(), rgb (what env_node replaces).
fn Node env_color() => builtin(ENV_COLOR, VEC3);
// The instance's matrix in the vertex stage (position_node): instance_matrix() * vec4(p, 1)
// is p moved as the instance moves it; the identity for a mesh without instances.
fn Node instance_matrix() => builtin(INSTANCE_MATRIX, MAT4);

// Whether the screen's depth (viewport_depth) is there this frame: on a material with
// Material.ssr, when the renderer has screen-space reflections' depth.
fn Node ssr_on() => drie::node_ssr_on();
// Where world-space point `p` is on the screen: xy its uv there (as viewport_depth takes it),
// z its distance from the camera's plane (m), as viewport_depth measures.
fn Node screen_position(Node p) => drie::node_ssr_project(p);
// The distance from the camera's plane (m) of what the screen shows at uv `at` (three's
// viewportLinearDepth), the frame's opaque surfaces: behind a transparent surface at
// screen_position(p), viewport_depth(s.xy()) - s.z() is how deep the water is there. Only
// where ssr_on() holds.
fn Node viewport_depth(Node at) => drie::node_viewport_depth(at);

// Material map `which` of the material drawn, where a graph reads a texture (texture,
// texture_grad, texture_size): one graph in several materials reads each one's own. The
// material must have it when its programs are built.
fn Texture* material_map(MaterialMap which) => drie::material_map_mark(which);

// `t` read at `at` (the mesh's uv by default): its rgba, as the material's maps (sRGB images
// in linear values). `level`: a mip level (none: from the pixel's footprint).
fn Node texture(Texture* t, Node at = { }, Node level = { }) {
    if (!t) abort("drie: nodes::texture of no texture");
    if (!at.id) at = uv();
    return drie::add_node({ .op = TEXTURE, .type = VEC4, .texture = t, .count = 2, .args = { at.id, level.id, 0, 0 } });
}

// `t` read at `uv` with the gradients `dx` and `dy` of the uv from one pixel to the next
// (three's texture().grad()).
fn Node texture_grad(Texture* t, Node uv, Node dx, Node dy) {
    if (!t) abort("drie: nodes::texture_grad of no texture");
    return drie::add_node(
        { .op = TEXTURE, .type = VEC4, .texture = t, .kind = 2, .count = 3, .args = { uv.id, dx.id, dy.id, 0 } }
    );
}

// The size of `t`'s largest level in texels, a vec2 (three's textureSize).
fn Node texture_size(Texture* t) {
    if (!t) abort("drie: nodes::texture_size of no texture");
    return drie::add_node({ .op = TEXTURE, .type = VEC2, .texture = t, .kind = 1 });
}

// `x` worked out per vertex and blended across the triangle (three's varying()).
fn Node varying(Node x) => drie::add_node({ .op = VARYING, .type = x.type(), .count = 1, .args = { x.id, 0, 0, 0 } });

// ---- Arithmetic and logic

// Component-wise sum `a + b`.
macro Node add(a, b) => drie::node_binary(ADD, to_node(a), to_node(b));
// Component-wise difference `a - b`.
macro Node sub(a, b) => drie::node_binary(SUB, to_node(a), to_node(b));
// Component-wise or matrix product `a * b`.
macro Node mul(a, b) => drie::node_binary(MUL, to_node(a), to_node(b));
// Component-wise quotient `a / b`.
macro Node div(a, b) => drie::node_binary(DIV, to_node(a), to_node(b));
// Scalar comparison `a < b`.
macro Node less_than(a, b) => drie::node_binary(LT, to_node(a), to_node(b));
// Scalar comparison `a <= b`.
macro Node less_than_equal(a, b) => drie::node_binary(LE, to_node(a), to_node(b));
// Scalar comparison `a > b`.
macro Node greater_than(a, b) => drie::node_binary(GT, to_node(a), to_node(b));
// Scalar comparison `a >= b`.
macro Node greater_than_equal(a, b) => drie::node_binary(GE, to_node(a), to_node(b));
// Scalar equality `a == b`.
macro Node equal(a, b) => drie::node_binary(EQ, to_node(a), to_node(b));
// Scalar inequality `a != b`.
macro Node not_equal(a, b) => drie::node_binary(NE, to_node(a), to_node(b));
// Boolean conjunction `a && b`.
fn Node logical_and(Node a, Node b) => drie::node_binary(AND, a, b);
// Boolean disjunction `a || b`.
fn Node logical_or(Node a, Node b) => drie::node_binary(OR, a, b);
// Boolean negation `!a`.
fn Node logical_not(Node a) => drie::add_node({ .op = NOT, .type = BOOL, .count = 1, .args = { a.id, 0, 0, 0 } });

// `cond` ? a : b (TSL's select)
macro Node select(Node cond, a, b) {
    Node x = to_node(a);
    Node y = to_node(b);
    return drie::add_node(
        { .op = SELECT, .type = drie::wider(x.type(), y.type()), .count = 3, .args = { cond.id, x.id, y.id, 0 } }
    );
}

// `x` as another type (TSL's convert: vec3 -> vec4 adds a w of 1, a float fills a vector)
fn Node convert(Node x, NodeType t) => drie::node_convert(x, t);

// ---- Math

// Sine of `x` (radians).
macro Node sin(x) => drie::node_call(SIN, to_node(x));
// Cosine of `x` (radians).
macro Node cos(x) => drie::node_call(COS, to_node(x));
// Tangent of `x` (radians).
macro Node tan(x) => drie::node_call(TAN, to_node(x));
// Arcsine of `x`.
macro Node asin(x) => drie::node_call(ASIN, to_node(x));
// Arccosine of `x`.
macro Node acos(x) => drie::node_call(ACOS, to_node(x));
// Arctangent of `x`.
macro Node atan(x) => drie::node_call(ATAN, to_node(x));
// Natural exponential `e^x`.
macro Node exp(x) => drie::node_call(EXP, to_node(x));
// Base-2 exponential `2^x`.
macro Node exp2(x) => drie::node_call(EXP2, to_node(x));
// Natural logarithm `ln(x)`.
macro Node log(x) => drie::node_call(LOG, to_node(x));
// Base-2 logarithm `log2(x)`.
macro Node log2(x) => drie::node_call(LOG2, to_node(x));
// Square root of `x`.
macro Node sqrt(x) => drie::node_call(SQRT, to_node(x));
// Reciprocal square root `1 / sqrt(x)`.
macro Node inverse_sqrt(x) => drie::node_call(INVERSESQRT, to_node(x));
// Component-wise absolute value of `x`.
macro Node abs(x) => drie::node_call(ABS, to_node(x));
// Component-wise sign (`-1`, `0`, or `1`) of `x`.
macro Node sign(x) => drie::node_call(SIGN, to_node(x));
// Nearest integer `<= x`.
macro Node floor(x) => drie::node_call(FLOOR, to_node(x));
// Nearest integer `>= x`.
macro Node ceil(x) => drie::node_call(CEIL, to_node(x));
// Fractional part `x - floor(x)`.
macro Node fract(x) => drie::node_call(FRACT, to_node(x));

// Reinterprets float bits as unsigned integer (GLSL `floatBitsToUint`, e.g. hash keys).
macro Node as_uint(x) => drie::node_call(BITS_AS_UINT, to_node(x));
// Reinterprets unsigned integer bits as float (GLSL `uintBitsToFloat`).
macro Node as_float(x) => drie::node_call(BITS_AS_FLOAT, to_node(x));
// Rounds `x` to the nearest integer.
macro Node round(x) => drie::node_call(ROUND, to_node(x));
// Truncates `x` toward zero.
macro Node trunc(x) => drie::node_call(TRUNC, to_node(x));
// Unit-length vector in the direction of `x`.
macro Node normalize(x) => drie::node_call(NORMALIZE, to_node(x));
// Euclidean length of vector `x`.
macro Node length(x) => drie::node_call(LENGTH, to_node(x));
// Clamps `x` to `[0, 1]`.
macro Node saturate(x) => drie::node_call(SATURATE, to_node(x));
// Screen-space partial derivative of `x` along X (fragment stage only).
macro Node dfdx(x) => drie::node_call(DFDX, to_node(x));
// Screen-space partial derivative of `x` along Y (fragment stage only).
macro Node dfdy(x) => drie::node_call(DFDY, to_node(x));
// Sum of absolute screen-space derivatives `|dfdx(x)| + |dfdy(x)|` (fragment stage only).
macro Node fwidth(x) => drie::node_call(FWIDTH, to_node(x));
// Complement `1 - x`.
macro Node one_minus(x) => drie::node_binary(SUB, num(1), to_node(x));
// Negation `-x`.
macro Node negate(x) => drie::node_binary(MUL, to_node(x), num(-1));
// Power `x^y`.
macro Node pow(x, y) => drie::node_call(POW, to_node(x), to_node(y));
// Component-wise minimum of `x` and `y`.
macro Node min(x, y) => drie::node_call(MIN, to_node(x), to_node(y));
// Component-wise maximum of `x` and `y`.
macro Node max(x, y) => drie::node_call(MAX, to_node(x), to_node(y));
// Step function: `0` when `x < edge`, `1` otherwise.
macro Node step(edge, x) => drie::node_call(STEP, to_node(edge), to_node(x));
// Dot product of vectors `x` and `y`.
macro Node dot(x, y) => drie::node_call(DOT, to_node(x), to_node(y));
// Cross product of 3D vectors `x` and `y`.
macro Node cross(x, y) => drie::node_call(CROSS, to_node(x), to_node(y));
// Euclidean distance between points `x` and `y`.
macro Node distance(x, y) => drie::node_call(DISTANCE, to_node(x), to_node(y));
// Two-argument arctangent of `y / x` in `[-pi, pi]`.
macro Node atan2(y, x) => drie::node_call(ATAN2, to_node(y), to_node(x));
// Reflection of incident vector `i` across normal `n`.
macro Node reflect(i, n) => drie::node_call(REFLECT, to_node(i), to_node(n));
// Floored modulo `x - y * floor(x / y)` for floats, or integer remainder.
macro Node mod(x, y) => drie::node_call(MOD, to_node(x), to_node(y));
// Linear interpolation `a * (1 - t) + b * t`.
macro Node mix(a, b, t) => drie::node_call(MIX, to_node(a), to_node(b), to_node(t));
// Clamps `x` to `[lo, hi]` (defaults to `[0, 1]`).
macro Node clamp(x, lo = 0.0f, hi = 1.0f) => drie::node_call(CLAMP, to_node(x), to_node(lo), to_node(hi));
// Hermite interpolation between `0` and `1` across `[e0, e1]`.
macro Node smoothstep(e0, e1, x) => drie::node_call(SMOOTHSTEP, to_node(e0), to_node(e1), to_node(x));
// Refraction of incident vector `i` through normal `n` with IOR ratio `eta`.
macro Node refract(i, n, eta) => drie::node_call(REFRACT, to_node(i), to_node(n), to_node(eta));
// Orients normal `n` so `dot(nref, i) < 0`.
macro Node faceforward(n, i, nref) => drie::node_call(FACEFORWARD, to_node(n), to_node(i), to_node(nref));

// ---- Integers and bitwise

// Signed 32-bit integer constant node.
fn Node int_(int x) {
    uint[4] b = { (uint)x, 0, 0, 0 };
    return drie::node_const_bits(INT, b);
}

// Unsigned 32-bit integer constant node.
fn Node uint_(uint x) {
    uint[4] b = { x, 0, 0, 0 };
    return drie::node_const_bits(UINT, b);
}

// Remainder operator (`%` for integers, floored mod for floats).
macro Node rem(a, b) => drie::node_binary(REM, to_node(a), to_node(b));
// Bitwise AND `a & b` on integer nodes.
macro Node bit_and(a, b) => drie::node_binary(BIT_AND, to_node(a), to_node(b));
// Bitwise OR `a | b` on integer nodes.
macro Node bit_or(a, b) => drie::node_binary(BIT_OR, to_node(a), to_node(b));
// Bitwise XOR `a ^ b` on integer nodes.
macro Node bit_xor(a, b) => drie::node_binary(BIT_XOR, to_node(a), to_node(b));
// Bitwise left shift `a << b` on integer nodes.
macro Node shift_left(a, b) => drie::node_binary(SHL, to_node(a), to_node(b));
// Bitwise right shift `a >> b` on integer nodes.
macro Node shift_right(a, b) => drie::node_binary(SHR, to_node(a), to_node(b));
// Bitwise complement `~a` on integer nodes.
fn Node bit_not(
    Node a
) => drie::add_node({ .op = NOT, .type = a.type(), .kind = 1, .count = 1, .args = { a.id, 0, 0, 0 } });
// Boolean exclusive-OR `a != b`.
fn Node logical_xor(Node a, Node b) => drie::node_binary(XOR, a, b);

// ---- Statements

// Mutable local variable initialized to `init` (three's `toVar()`).
fn Node variable(Node init) => drie::node_var(init.type(), init);
// Zero-initialized variable (defaults to `INT`), typically used as a `loop` counter.
fn Node loop_index(NodeType t = INT) => drie::node_var(t);

// Chains statements (or expression evaluations, such as declaring variables) in execution order.
macro Node block(...) {
    $if $vaarg.len == 0:
        return (Node) { };
    $else
        Node[$vaarg.len] stmts;
        $for var $i = 0; $i < $vaarg.len; $i++:
            stmts[$i] = $vaarg[$i];
        $endfor
        return drie::node_block(&stmts);
    $endif
}

// Conditional statement executing block `then` when `cond` holds, and optional `otherwise` else.
fn Node if_then(Node cond, Node then, Node otherwise = { }) => drie::node_if(cond, then, otherwise);

// `index` from `from` while below `to` (or as `cond` says), by 1 (-1 for GT and GE).
macro Node loop(Node index, from, to, Node body, NodeLoop cond = LT) {
    return drie::node_loop(
        cond,
        index,
        drie::node_convert(to_node(from), index.type()),
        drie::node_convert(to_node(to), index.type()),
        body
    );
}

// The block, while `cond` holds.
fn Node loop_while(Node cond, Node body) => drie::node_loop(WHILE, { }, { }, cond, body);

// Exits the innermost enclosing loop.
fn Node break_() => drie::node_jump(0);
// Advances to the next iteration of the innermost enclosing loop.
fn Node continue_() => drie::node_jump(1);
// Discards the current fragment (fragment stage only).
fn Node discard() => drie::node_jump(2);

// Evaluates statement `block` and yields `value` (or `VOID` if omitted), e.g.:
//
//   Node sum = nodes::variable(nodes::num(0));
//   Node i = nodes::loop_index();
//   Node total = nodes::seq(nodes::block(
//       sum,
//       nodes::loop(i, 0, 8, nodes::block(
//           sum.assign(sum + 0.125f),
//           nodes::if_then(i.equal(nodes::int_(3)), nodes::block(nodes::break_()))))),
//       sum);
fn Node seq(Node block, Node value = { }) => drie::node_seq(block, value);

// ---- Functions

// Declares a reusable shader function `fn_<name>` returning `ret` (three's `Fn` with a layout),
// emitted once per stage that calls it. Bodies may read uniforms, textures (`material_map`
// included), statements without `discard`, and other functions, but must take builtins (`uv()`,
// `position_world()`, `time()`, ...) and varyings as parameters, e.g.:
//
//   NodeFunction hash = nodes::function("hash", FLOAT, VEC2);
//   Node p = hash.param(0);
//   hash.returns(nodes::fract(nodes::sin(nodes::dot(p, nodes::vec2(127.1f, 311.7f))) * 43758.5453f));
//   Node n = hash.call(nodes::uv() * 8) + hash.call(nodes::uv() * 8 + 1);
fn NodeFunction function(String name, NodeType ret, NodeType ...params) {
    if (!name.len) abort("drie: nodes: a function needs a name");
    foreach (c : name) {
        if (!c.is_alnum() && c != '_') abort(
            "drie: nodes: '%s' is not a function's name (letters, digits and _)",
            name
        );
    }
    return { drie::node_func(string::tformat("fn_%s", name), ret, params, true) };
}