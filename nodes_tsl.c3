// drie::nodes: the constructors of shader graph nodes (see nodes.c3), named as three's TSL
// (in snake case). Numbers go where a node does: nodes::mix(a, b, 0.5f), nodes::vec3(x, 0, 1).
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

// ---- values

fn Node num(float x) {
    float[16] v;
    v[0] = x;
    return drie::node_const(FLOAT, v);
}

fn Node bool_(bool b) {
    float[16] v;
    v[0] = b ? 1 : 0;
    return drie::node_const(BOOL, v);
}

// A linear color from sRGB hex, as three's color(0xff8800).
fn Node color(Color c) {
    Vector3 l = drie::color_to_linear(c);
    float[16] v = { [0] = l.x, [1] = l.y, [2] = l.z };
    return drie::node_const(VEC3, v);
}

// Vectors, of numbers or nodes (a vector's components count for as many): vec3(1, 0, 0),
// vec4(n.xyz(), 1), vec3(x) for x in every component.
macro Node vec2(...) => vector(VEC2, ...$vaarg);
macro Node vec3(...) => vector(VEC3, ...$vaarg);
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

// A value changed from the application (Node.set), without rebuilding the shaders.
fn Node uniform(float x) {
    float[16] v = { [0] = x };
    return drie::add_node({ .op = UNIFORM, .type = FLOAT, .value = v });
}

fn Node uniform_vec2(float x, float y) {
    float[16] v = { [0] = x, [1] = y };
    return drie::add_node({ .op = UNIFORM, .type = VEC2, .value = v });
}

fn Node uniform_vec3(Vector3 x) {
    float[16] v = { [0] = x.x, [1] = x.y, [2] = x.z };
    return drie::add_node({ .op = UNIFORM, .type = VEC3, .value = v });
}

fn Node uniform_vec4(float x, float y, float z, float w) {
    float[16] v = { [0] = x, [1] = y, [2] = z, [3] = w };
    return drie::add_node({ .op = UNIFORM, .type = VEC4, .value = v });
}

// (sRGB hex: linear values)
fn Node uniform_color(Color c) => uniform_vec3(drie::color_to_linear(c));

fn Node uniform_matrix(Mat4 m) {
    Node n = drie::add_node({ .op = UNIFORM, .type = MAT4 });
    n.set_matrix(m);
    return n;
}

// ---- what the shader knows

fn Node builtin(NodeBuiltin b, NodeType t) @local => drie::add_node({ .op = BUILTIN, .type = t, .kind = (char)b.ordinal });

fn Node uv() => builtin(UV, VEC2);
fn Node uv1() => builtin(UV1, VEC2);
fn Node position_local() => builtin(POSITION_LOCAL, VEC3);
fn Node position_world() => builtin(POSITION_WORLD, VEC3);
fn Node normal_local() => builtin(NORMAL_LOCAL, VEC3);
fn Node normal_world() => builtin(NORMAL_WORLD, VEC3);
fn Node normal_geometry() => builtin(NORMAL_GEOMETRY, VEC3);
fn Node camera_position() => builtin(CAMERA_POSITION, VEC3);
fn Node view_direction() => builtin(VIEW_DIRECTION, VEC3);
fn Node time() => builtin(TIME, FLOAT);
fn Node vertex_color() => builtin(VERTEX_COLOR, VEC4);
fn Node frag_coord() => builtin(FRAG_COORD, VEC2);
fn Node front_facing() => builtin(FRONT_FACING, BOOL);
fn Node model_matrix() => builtin(MODEL_MATRIX, MAT4);
fn Node view_matrix() => builtin(VIEW_MATRIX, MAT4);
fn Node projection_matrix() => builtin(PROJECTION_MATRIX, MAT4);
// What the light made of the material, rgba (three's output): for Material.output_node.
fn Node output() => builtin(OUTPUT, VEC4);
// In Material.env_node: the direction the surface reflects (in the world), and its roughness.
fn Node env_direction() => builtin(ENV_DIRECTION, VEC3);
fn Node env_roughness() => builtin(ENV_ROUGHNESS, FLOAT);
// The instance's index in its InstancedMesh (0 for a plain mesh), a float (three's instanceIndex).
fn Node instance_index() => builtin(INSTANCE_INDEX, FLOAT);

// The material's own value at the step a graph replaces, to change it rather than replace it:
// in color_node and opacity_node its color (rgb: after its map, vertex and instance colors;
// a: its opacity); in roughness_node and metalness_node (and the surface step's emissive,
// specular and shininess nodes) its roughness and metalness after its maps.
fn Node material_color() => builtin(MATERIAL_COLOR, VEC4);
fn Node material_roughness() => builtin(MATERIAL_ROUGHNESS, FLOAT);
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
    return drie::add_node({ .op = TEXTURE, .type = VEC4, .texture = t, .kind = 2, .count = 3, .args = { uv.id, dx.id, dy.id, 0 } });
}

// The size of `t`'s largest level in texels, a vec2 (three's textureSize).
fn Node texture_size(Texture* t) {
    if (!t) abort("drie: nodes::texture_size of no texture");
    return drie::add_node({ .op = TEXTURE, .type = VEC2, .texture = t, .kind = 1 });
}

// `x` worked out per vertex and blended across the triangle (three's varying()).
fn Node varying(Node x) => drie::add_node({ .op = VARYING, .type = x.type(), .count = 1, .args = { x.id, 0, 0, 0 } });

// ---- arithmetic and logic (also Node's + - * /)

macro Node add(a, b) => drie::node_binary(ADD, to_node(a), to_node(b));
macro Node sub(a, b) => drie::node_binary(SUB, to_node(a), to_node(b));
macro Node mul(a, b) => drie::node_binary(MUL, to_node(a), to_node(b));
macro Node div(a, b) => drie::node_binary(DIV, to_node(a), to_node(b));
macro Node less_than(a, b) => drie::node_binary(LT, to_node(a), to_node(b));
macro Node less_than_equal(a, b) => drie::node_binary(LE, to_node(a), to_node(b));
macro Node greater_than(a, b) => drie::node_binary(GT, to_node(a), to_node(b));
macro Node greater_than_equal(a, b) => drie::node_binary(GE, to_node(a), to_node(b));
macro Node equal(a, b) => drie::node_binary(EQ, to_node(a), to_node(b));
macro Node not_equal(a, b) => drie::node_binary(NE, to_node(a), to_node(b));
fn Node logical_and(Node a, Node b) => drie::node_binary(AND, a, b);
fn Node logical_or(Node a, Node b) => drie::node_binary(OR, a, b);
fn Node logical_not(Node a) => drie::add_node({ .op = NOT, .type = BOOL, .count = 1, .args = { a.id, 0, 0, 0 } });

// `cond` ? a : b (TSL's select)
macro Node select(Node cond, a, b) {
    Node x = to_node(a);
    Node y = to_node(b);
    return drie::add_node({ .op = SELECT, .type = drie::wider(x.type(), y.type()), .count = 3, .args = { cond.id, x.id, y.id, 0 } });
}

// `x` as another type (TSL's convert: vec3 -> vec4 adds a w of 1, a float fills a vector)
fn Node convert(Node x, NodeType t) => drie::node_convert(x, t);

// ---- math

macro Node sin(x) => drie::node_call(SIN, to_node(x));
macro Node cos(x) => drie::node_call(COS, to_node(x));
macro Node tan(x) => drie::node_call(TAN, to_node(x));
macro Node asin(x) => drie::node_call(ASIN, to_node(x));
macro Node acos(x) => drie::node_call(ACOS, to_node(x));
macro Node atan(x) => drie::node_call(ATAN, to_node(x));
macro Node exp(x) => drie::node_call(EXP, to_node(x));
macro Node exp2(x) => drie::node_call(EXP2, to_node(x));
macro Node log(x) => drie::node_call(LOG, to_node(x));
macro Node log2(x) => drie::node_call(LOG2, to_node(x));
macro Node sqrt(x) => drie::node_call(SQRT, to_node(x));
macro Node inverse_sqrt(x) => drie::node_call(INVERSESQRT, to_node(x));
macro Node abs(x) => drie::node_call(ABS, to_node(x));
macro Node sign(x) => drie::node_call(SIGN, to_node(x));
macro Node floor(x) => drie::node_call(FLOOR, to_node(x));
macro Node ceil(x) => drie::node_call(CEIL, to_node(x));
macro Node fract(x) => drie::node_call(FRACT, to_node(x));

// The bits of a number read as the other kind, as GLSL's floatBitsToUint and uintBitsToFloat:
// nodes::as_uint(entry.y()) is a hash entry's key, which is stored as a number's bits.
macro Node as_uint(x) => drie::node_call(BITS_AS_UINT, to_node(x));
macro Node as_float(x) => drie::node_call(BITS_AS_FLOAT, to_node(x));
macro Node round(x) => drie::node_call(ROUND, to_node(x));
macro Node trunc(x) => drie::node_call(TRUNC, to_node(x));
macro Node normalize(x) => drie::node_call(NORMALIZE, to_node(x));
macro Node length(x) => drie::node_call(LENGTH, to_node(x));
macro Node saturate(x) => drie::node_call(SATURATE, to_node(x));
macro Node dfdx(x) => drie::node_call(DFDX, to_node(x));
macro Node dfdy(x) => drie::node_call(DFDY, to_node(x));
macro Node fwidth(x) => drie::node_call(FWIDTH, to_node(x));
macro Node one_minus(x) => drie::node_binary(SUB, num(1), to_node(x));
macro Node negate(x) => drie::node_binary(MUL, to_node(x), num(-1));
macro Node pow(x, y) => drie::node_call(POW, to_node(x), to_node(y));
macro Node min(x, y) => drie::node_call(MIN, to_node(x), to_node(y));
macro Node max(x, y) => drie::node_call(MAX, to_node(x), to_node(y));
macro Node step(edge, x) => drie::node_call(STEP, to_node(edge), to_node(x));
macro Node dot(x, y) => drie::node_call(DOT, to_node(x), to_node(y));
macro Node cross(x, y) => drie::node_call(CROSS, to_node(x), to_node(y));
macro Node distance(x, y) => drie::node_call(DISTANCE, to_node(x), to_node(y));
macro Node atan2(y, x) => drie::node_call(ATAN2, to_node(y), to_node(x));
macro Node reflect(i, n) => drie::node_call(REFLECT, to_node(i), to_node(n));
macro Node mod(x, y) => drie::node_call(MOD, to_node(x), to_node(y));
macro Node mix(a, b, t) => drie::node_call(MIX, to_node(a), to_node(b), to_node(t));
macro Node clamp(x, lo = 0.0f, hi = 1.0f) => drie::node_call(CLAMP, to_node(x), to_node(lo), to_node(hi));
macro Node smoothstep(e0, e1, x) => drie::node_call(SMOOTHSTEP, to_node(e0), to_node(e1), to_node(x));
macro Node refract(i, n, eta) => drie::node_call(REFRACT, to_node(i), to_node(n), to_node(eta));
macro Node faceforward(n, i, nref) => drie::node_call(FACEFORWARD, to_node(n), to_node(i), to_node(nref));

// ---- integers (TSL's int(), uint(); bitwise operators)

fn Node int_(int x) {
    uint[4] b = { (uint)x, 0, 0, 0 };
    return drie::node_const_bits(INT, b);
}

fn Node uint_(uint x) {
    uint[4] b = { x, 0, 0, 0 };
    return drie::node_const_bits(UINT, b);
}

macro Node rem(a, b) => drie::node_binary(REM, to_node(a), to_node(b));
macro Node bit_and(a, b) => drie::node_binary(BIT_AND, to_node(a), to_node(b));
macro Node bit_or(a, b) => drie::node_binary(BIT_OR, to_node(a), to_node(b));
macro Node bit_xor(a, b) => drie::node_binary(BIT_XOR, to_node(a), to_node(b));
macro Node shift_left(a, b) => drie::node_binary(SHL, to_node(a), to_node(b));
macro Node shift_right(a, b) => drie::node_binary(SHR, to_node(a), to_node(b));
fn Node bit_not(Node a) => drie::add_node({ .op = NOT, .type = a.type(), .kind = 1, .count = 1, .args = { a.id, 0, 0, 0 } });
fn Node logical_xor(Node a, Node b) => drie::node_binary(XOR, a, b);

// ---- statements (TSL's Fn bodies: toVar, assign, If, Loop, Break...)
//
//   Node sum = nodes::variable(nodes::num(0));
//   Node i = nodes::loop_index();
//   Node total = nodes::seq(nodes::block(
//       sum,                                       // (declared here: before the loop)
//       nodes::loop(i, 0, 8, nodes::block(
//           sum.assign(sum + 0.125f),
//           nodes::if_then(i.equal(nodes::int_(3)), nodes::block(nodes::break_()))))),
//       sum);
//
// A block is its statements in order (an expression among them is worked out there: a
// variable is declared where it first appears). seq() makes an expression of a block and the
// value it leaves.

// A variable, of `init` at first (three's toVar()).
fn Node variable(Node init) => drie::node_var(init.type(), init);
// A loop's index (or a variable of zeros).
fn Node loop_index(NodeType t = INT) => drie::node_var(t);

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

fn Node if_then(Node cond, Node then, Node otherwise = { }) => drie::node_if(cond, then, otherwise);

// `index` from `from` while below `to` (or as `cond` says), by 1 (-1 for GT and GE).
macro Node loop(Node index, from, to, Node body, NodeLoop cond = LT) {
    return drie::node_loop(cond, index, drie::node_convert(to_node(from), index.type()), drie::node_convert(to_node(to), index.type()), body);
}

// The block, while `cond` holds.
fn Node loop_while(Node cond, Node body) => drie::node_loop(WHILE, { }, { }, cond, body);

fn Node break_() => drie::node_jump(0);
fn Node continue_() => drie::node_jump(1);
// No fragment here (the fragment stage only).
fn Node discard() => drie::node_jump(2);

fn Node seq(Node block, Node value = { }) => drie::node_seq(block, value);

// ---- functions (three's Fn with a layout)
//
//   NodeFunction hash = nodes::function("hash", FLOAT, VEC2);
//   Node p = hash.param(0);
//   hash.returns(nodes::fract(nodes::sin(nodes::dot(p, nodes::vec2(127.1f, 311.7f))) * 43758.5453f));
//   Node n = hash.call(nodes::uv() * 8) + hash.call(nodes::uv() * 8 + 1);
//
// A function `name` (letters, digits, _) returning a `ret`. It is printed once in each stage
// of a program that calls it, as fn_`name`; a C3 function making nodes prints them again at
// every use. The body can read uniforms, textures (material_map too), statements without
// discard, and other functions, but not builtins (uv(), position_world(), time()...) or
// varyings: pass those as parameters. Make functions once, like the graphs that call them.
// Programs for an application's material shader (Renderer.shaders) can't call them.
fn NodeFunction function(String name, NodeType ret, NodeType... params) {
    if (!name.len) abort("drie: nodes: a function needs a name");
    foreach (c : name) {
        if (!c.is_alnum() && c != '_') abort("drie: nodes: '%s' is not a function's name (letters, digits and _)", name);
    }
    return { drie::node_func(string::tformat("fn_%s", name), ret, params, true) };
}
