// Text expression parser (`drie::nodes::snippet`) compiling GLSL-like expressions with `let`
// bindings into shader `Node` graphs, plus error reporting (`SNIPPET_ERROR`, `snippet_error`).
// 
// Validates operand types, swizzles, and arity before calling `drie::nodes` constructors so
// syntax or type mistakes return `SNIPPET_ERROR~` with `"line:col: message"` instead of
// aborting. Supports up to 32 `let` bindings and 4 arguments per call. Emits nodes through
// `drie::nodes` (src/nodes/tsl.c3) and `drie` (src/nodes/graph.c3).
module drie::nodes;
import drie;
import std::io, std::core::string;

// Fault returned by `snippet` on syntax or type errors; call `snippet_error()` for details.
faultdef SNIPPET_ERROR;

// A name a snippet can use, and the node it stands for.
struct SnippetName {
    String name;
    Node node;
}

char[256] snippet_message @local;
usz snippet_message_len @local;

// What the last snippet that failed got wrong, with its line and column.
fn String snippet_error() => (String)snippet_message[:snippet_message_len];

struct Snippet @local {
    String src;
    usz at;
    SnippetName[] names;
    SnippetName[32] lets;
    int let_count;
    bool failed;
}

// Parses GLSL-like expression text `src` (with optional leading `let name = expr;` bindings and
// external `names`) into a shader `Node`, returning `SNIPPET_ERROR~` on failure:
// 
//   Node dirt = nodes::snippet(`
//       let h = position_world.y - floor_y;
//       let d = smoothstep(0.6, 0.0, h) * amount;
//       mix(material_color.rgb, vec3(0.2, 0.17, 0.12), d)
//   `, { { "floor_y", floor_y }, { "amount", amount } })!!;
//   mat.color_node = nodes::vec4(dirt, 1);
// 
// Supports numbers, `+ - * / % < <= > >= == != && || ! ?: ()`, swizzles (`.x`, `.xy`, `.rgb`,
// ...), `vec2`/`vec3`/`vec4`, `drie::nodes` math functions (`sin`..`smoothstep`, `one_minus`),
// and shader builtins (`uv`, `uv1`, `position_local`, `position_world`, `normal_local`,
// `normal_world`, `normal_geometry`, `camera_position`, `view_direction`, `time`, `vertex_color`,
// `frag_coord`, `material_color`, `material_roughness`, `material_metalness`, `instance_index`).
// Pass textures, loops, or functions constructed with `drie::nodes` through `names`.
fn Node? snippet(String src, SnippetName[] names = { }) {
    Snippet s = { .src = src, .names = names };
    snippet_message_len = 0;
    while (s.word("let")) {
        usz start   = s.at;
        String name = s.ident();
        if (!name.len) return s.fail_at(start, "a name after let");
        if (!s.symbol("=")) return s.fail("'=' after let %s", name);
        Node v = s.expr()!;
        if (!s.symbol(";")) return s.fail("';' after let %s = ...", name);
        if (s.let_count == s.lets.len) return s.fail("too many lets");
        s.lets[s.let_count++] = { name, v };
    }
    Node n = s.expr()!;
    s.space();
    if (s.at < s.src.len) return s.fail("'%c' after the expression", s.src[s.at]);
    return n;
}

// ---- Lexing and diagnostics

// Records a formatted parse error at the current cursor position and returns `SNIPPET_ERROR~`.
fn Node? Snippet.fail(&self, String fmt, args...) => self.fail_at(self.at, fmt, ...args);

// Records a formatted `"line:col: msg"` error at byte offset `at` and returns `SNIPPET_ERROR~`.
fn Node? Snippet.fail_at(&self, usz at, String fmt, args...) {
    int line = 1;
    int col  = 1;
    for (usz i = 0; i < at && i < self.src.len; i++) {
        if (self.src[i] == '\n') {
            line++;
            col = 1;
        } else {
            col++;
        }
    }
    @pool() {
        String what = string::tformat(fmt, ...args);
        String msg  = string::tformat("%d:%d: %s", line, col, what);
        snippet_message_len = msg.len < snippet_message.len ? msg.len : snippet_message.len;
        snippet_message[:snippet_message_len] = msg[:snippet_message_len];
    };
    return SNIPPET_ERROR~;
}

// Skips whitespace and `//` line comments starting at the current cursor position.
fn void Snippet.space(&self) {
    while (self.at < self.src.len) {
        char c = self.src[self.at];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            self.at++;
        } else if (c == '/' && self.at + 1 < self.src.len && self.src[self.at + 1] == '/') {
            while (self.at < self.src.len && self.src[self.at] != '\n') self.at++;
        } else {
            return;
        }
    }
}

fn bool is_ident_char(char c) @local => c.is_alnum() || c == '_';

// The name here (none: "").
fn String Snippet.ident(&self) {
    self.space();
    usz start = self.at;
    if (start >= self.src.len || !(self.src[start].is_alpha() || self.src[start] == '_')) return "";
    while (self.at < self.src.len && is_ident_char(self.src[self.at])) self.at++;
    return self.src[start:self.at - start];
}

// Whether keyword `w` is here (and then past it).
fn bool Snippet.word(&self, String w) {
    self.space();
    usz end = self.at + w.len;
    if (end > self.src.len || self.src[self.at:w.len] != w) return false;
    if (end < self.src.len && is_ident_char(self.src[end])) return false;
    self.at = end;
    return true;
}

// Whether `sym` is here (and then past it). "<" is not "<=".
fn bool Snippet.symbol(&self, String sym) {
    self.space();
    usz end = self.at + sym.len;
    if (end > self.src.len || self.src[self.at:sym.len] != sym) return false;
    if (sym.len == 1 && end < self.src.len && self.src[end] == '='
        && (sym[0] == '<' || sym[0] == '>' || sym[0] == '=' || sym[0] == '!')) return false;
    self.at = end;
    return true;
}

// A type as GLSL names it (in messages).
fn String glsl(NodeType t) @local {
    switch (t) {
        case FLOAT: return "float";
        case VEC2: return "vec2";
        case VEC3: return "vec3";
        case VEC4: return "vec4";
        case BOOL: return "bool";
        case MAT3: return "mat3";
        case MAT4: return "mat4";
        case INT: return "int";
        case IVEC2: return "ivec2";
        case IVEC3: return "ivec3";
        case IVEC4: return "ivec4";
        case UINT: return "uint";
        case UVEC2: return "uvec2";
        case UVEC3: return "uvec3";
        case UVEC4: return "uvec4";
        case VOID: return "statement";
    }
}

// ---- Type checking

fn bool fits(Node a, Node b) @local {
    NodeType ta = a.type();
    NodeType tb = b.type();
    if (ta == VOID || tb == VOID) return false;
    if (drie::is_matrix(ta) || drie::is_matrix(tb)) return ta == tb;
    int na = drie::components(ta);
    int nb = drie::components(tb);
    return na == nb || na == 1 || nb == 1;
}

// Type-checks binary operator `op` on `x` and `y` and builds the corresponding `NodeBinary` node.
fn Node? Snippet.binary(&self, usz at, String op, NodeBinary b, Node x, Node y) {
    if (!fits(
        x,
        y
    )) return self.fail_at(
        at,
        "%s %s %s: the sizes don't go together (.xyz, vec3(...))",
        glsl(x.type()),
        op,
        glsl(y.type())
    );
    if (drie::is_matrix(x.type()) && b != MUL) return self.fail_at(at, "matrices only multiply");
    return drie::node_binary(b, x, y);
}

// ---- Expression parser

// Parses a ternary conditional expression (`c ? a : b`) or lower-precedence expression.
fn Node? Snippet.expr(&self) {
    Node c = self.or_expr()!;
    usz at = self.at;
    if (!self.symbol("?")) return c;
    Node a = self.expr()!;
    if (!self.symbol(":")) return self.fail("':' in c ? a : b");
    Node b = self.expr()!;
    if (!fits(
        a,
        b
    )) return self.fail_at(at, "c ? %s : %s: the sizes don't go together", glsl(a.type()), glsl(b.type()));
    return select(c, a, b);
}

// Parses logical OR (`||`) expressions.
fn Node? Snippet.or_expr(&self) {
    Node a = self.and_expr()!;
    while (self.symbol("||")) a = logical_or(a, self.and_expr()!);
    return a;
}

// Parses logical AND (`&&`) expressions.
fn Node? Snippet.and_expr(&self) {
    Node a = self.compare()!;
    while (self.symbol("&&")) a = logical_and(a, self.compare()!);
    return a;
}

// Parses relational and equality comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`).
fn Node? Snippet.compare(&self) {
    Node a = self.sum()!;
    self.space();
    usz at = self.at;
    if (self.symbol("<=")) return self.binary(at, "<=", LE, a, self.sum()!);
    if (self.symbol(">=")) return self.binary(at, ">=", GE, a, self.sum()!);
    if (self.symbol("==")) return self.binary(at, "==", EQ, a, self.sum()!);
    if (self.symbol("!=")) return self.binary(at, "!=", NE, a, self.sum()!);
    if (self.symbol("<")) return self.binary(at, "<", LT, a, self.sum()!);
    if (self.symbol(">")) return self.binary(at, ">", GT, a, self.sum()!);
    return a;
}

// Parses additive expressions (`+`, `-`).
fn Node? Snippet.sum(&self) {
    Node a = self.product()!;
    while (true) {
        self.space();
        usz at = self.at;
        if (self.symbol("+")) {
            a = self.binary(at, "+", ADD, a, self.product()!)!;
        } else if (self.symbol("-")) {
            a = self.binary(at, "-", SUB, a, self.product()!)!;
        } else {
            return a;
        }
    }
}

// Parses multiplicative and floored-modulo expressions (`*`, `/`, `%`).
fn Node? Snippet.product(&self) {
    Node a = self.unary()!;
    while (true) {
        self.space();
        usz at = self.at;
        if (self.symbol("*")) {
            a = self.binary(at, "*", MUL, a, self.unary()!)!;
        } else if (self.symbol("/")) {
            a = self.binary(at, "/", DIV, a, self.unary()!)!;
        } else if (self.symbol("%")) {
            Node b = self.unary()!;
            if (!fits(
                a,
                b
            )) return self.fail_at(at, "%s %% %s: the sizes don't go together", glsl(a.type()), glsl(b.type()));
            a = mod(a, b);
        } else {
            return a;
        }
    }
}

// Parses unary negation (`-`) and logical NOT (`!`).
fn Node? Snippet.unary(&self) {
    if (self.symbol("-")) return self.unary()!.negate();
    if (self.symbol("!")) return logical_not(self.unary()!);
    return self.postfix();
}

// Parses chained component swizzles (`.x`, `.xy`, `.rgb`, etc.) on a primary expression.
fn Node? Snippet.postfix(&self) {
    Node n = self.primary()!;
    while (self.symbol(".")) {
        usz at    = self.at;
        String sw = self.ident();
        if (!sw.len || sw.len > 4) return self.fail_at(at, "a swizzle after '.' (x, xy, rgb...)");
        if (drie::is_matrix(n.type())) return self.fail_at(at, "no swizzle of a matrix");
        int have = drie::components(n.type());
        foreach (ch : sw) {
            int c = -1;
            switch (ch) {
                case 'x':
                case 'r': c = 0;
                case 'y':
                case 'g': c = 1;
                case 'z':
                case 'b': c = 2;
                case 'w':
                case 'a': c = 3;
                default: c = -1;
            }
            if (c < 0) return self.fail_at(at, "'%s' is not a swizzle (x y z w, r g b a)", sw);
            if (c >= have) return self.fail_at(at, ".%s of a %s", sw, glsl(n.type()));
        }
        n = n.swizzle(sw);
    }
    return n;
}

// Parses parenthesized expressions, numbers, calls, `let`/`names` lookups, and shader builtins.
fn Node? Snippet.primary(&self) {
    self.space();
    usz at = self.at;
    if (at >= self.src.len) return self.fail("an expression, not the end");
    char c = self.src[at];
    if (c == '(') {
        self.at++;
        Node n = self.expr()!;
        if (!self.symbol(")")) return self.fail("')'");
        return n;
    }
    if (c.is_digit() || (c == '.' && at + 1 < self.src.len && self.src[at + 1].is_digit())) return self.number();
    String name = self.ident();
    if (!name.len) return self.fail("an expression, not '%c'", c);
    if (self.symbol("(")) return self.call(at, name);
    for (int i = self.let_count - 1; i >= 0; i--) {
        if (self.lets[i].name == name) return self.lets[i].node;
    }
    foreach (nm : self.names) {
        if (nm.name == name) return nm.node;
    }
    switch (name) {
        case "uv": return uv();
        case "uv1": return uv1();
        case "position_local": return position_local();
        case "position_world": return position_world();
        case "normal_local": return normal_local();
        case "normal_world": return normal_world();
        case "normal_geometry": return normal_geometry();
        case "camera_position": return camera_position();
        case "view_direction": return view_direction();
        case "time": return time();
        case "vertex_color": return vertex_color();
        case "frag_coord": return frag_coord();
        case "material_color": return material_color();
        case "material_roughness": return material_roughness();
        case "material_metalness": return material_metalness();
        case "instance_index": return instance_index();
        case "true": return bool_(true);
        case "false": return bool_(false);
    }
    return self.fail_at(at, "no '%s' here", name);
}

// Parses a floating-point literal (optional exponent and `f` suffix) into a `FLOAT` constant node.
fn Node? Snippet.number(&self) {
    usz start = self.at;
    while (self.at < self.src.len && (self.src[self.at].is_digit() || self.src[self.at] == '.')) self.at++;
    if (self.at < self.src.len && (self.src[self.at] == 'e' || self.src[self.at] == 'E')) {
        self.at++;
        if (self.at < self.src.len && (self.src[self.at] == '-' || self.src[self.at] == '+')) self.at++;
        while (self.at < self.src.len && self.src[self.at].is_digit()) self.at++;
    }
    String text = self.src[start:self.at - start];
    if (self.at < self.src.len && self.src[self.at] == 'f') self.at++;
    double? v = text.to_double();
    if (catch v) return self.fail_at(start, "'%s' is not a number", text);
    return num((float)v);
}

// Parses a vector constructor (`vec2`/`vec3`/`vec4`) or built-in math function call.
fn Node? Snippet.call(&self, usz at, String name) {
    Node[4] args;
    int n;
    if (!self.symbol(")")) {
        while (true) {
            if (n == args.len) return self.fail("more than %d arguments to %s", args.len, name);
            args[n++] = self.expr()!;
            if (self.symbol(")")) break;
            if (!self.symbol(",")) return self.fail("',' or ')' in %s(...)", name);
        }
    }
    // vectors: their parts' components add up
    int size;
    switch (name) {
        case "vec2": size = 2;
        case "vec3": size = 3;
        case "vec4": size = 4;
    }
    if (size) {
        if (n == 0) return self.fail_at(at, "%s() of nothing", name);
        int total;
        for (int i = 0; i < n; i++) {
            if (drie::is_matrix(args[i].type())) return self.fail_at(at, "a matrix in %s()", name);
            total += drie::components(args[i].type());
        }
        if (n == 1 && total == 1) return drie::node_convert(args[0], drie::type_of(FLOAT, size));
        if (total != size) return self.fail_at(at, "%s() of %d components", name, total);
        if (n == 1) return drie::node_convert(args[0], drie::type_of(FLOAT, size));
        return drie::node_join(args[:n], drie::type_of(FLOAT, size));
    }
    // the math: GLSL's way, a vector with numbers or with a vector as large
    for (int i = 1; i < n; i++) {
        if (!fits(
            args[0],
            args[i]
        ) && !fits(
            args[i],
            args[0]
        )) return self.fail_at(
            at,
            "%s(): %s and %s don't go together",
            name,
            glsl(args[0].type()),
            glsl(args[i].type())
        );
    }
    for (int i = 0; i < n; i++) {
        if (drie::is_matrix(args[i].type())) return self.fail_at(at, "%s() of a matrix", name);
    }
    int want = math_args(name);
    if (want == 0) return self.fail_at(at, "no function '%s' here", name);
    if (name == "clamp" && (n == 1 || n == 3)) {
        return n == 1 ? clamp(args[0]) : clamp(args[0], args[1], args[2]);
    }
    if (n != want) return self.fail_at(at, "%s() takes %d, not %d", name, want, n);
    Node a = args[0];
    Node b = args[1];
    Node c = args[2];
    switch (name) {
        case "sin": return sin(a);
        case "cos": return cos(a);
        case "tan": return tan(a);
        case "asin": return asin(a);
        case "acos": return acos(a);
        case "atan": return atan(a);
        case "exp": return exp(a);
        case "exp2": return exp2(a);
        case "log": return log(a);
        case "log2": return log2(a);
        case "sqrt": return sqrt(a);
        case "inversesqrt": return inverse_sqrt(a);
        case "abs": return abs(a);
        case "sign": return sign(a);
        case "floor": return floor(a);
        case "ceil": return ceil(a);
        case "fract": return fract(a);
        case "round": return round(a);
        case "trunc": return trunc(a);
        case "normalize": return normalize(a);
        case "length": return length(a);
        case "saturate": return saturate(a);
        case "dfdx": return dfdx(a);
        case "dfdy": return dfdy(a);
        case "fwidth": return fwidth(a);
        case "one_minus": return one_minus(a);
        case "pow": return pow(a, b);
        case "min": return min(a, b);
        case "max": return max(a, b);
        case "step": return step(a, b);
        case "dot": return dot(a, b);
        case "cross": return cross(a, b);
        case "distance": return distance(a, b);
        case "atan2": return atan2(a, b);
        case "reflect": return reflect(a, b);
        case "mod": return mod(a, b);
        case "mix": return mix(a, b, c);
        case "smoothstep": return smoothstep(a, b, c);
    }
    return self.fail_at(at, "no function '%s' here", name);
}

// How many arguments math function `name` takes (0: there is none).
fn int math_args(String name) @local {
    switch (name) {
        case "sin":
        case "cos":
        case "tan":
        case "asin":
        case "acos":
        case "atan":
        case "exp":
        case "exp2":
        case "log":
        case "log2":
        case "sqrt":
        case "inversesqrt":
        case "abs":
        case "sign":
        case "floor":
        case "ceil":
        case "fract":
        case "round":
        case "trunc":
        case "normalize":
        case "length":
        case "saturate":
        case "dfdx":
        case "dfdy":
        case "fwidth":
        case "one_minus":
            return 1;
        case "pow":
        case "min":
        case "max":
        case "step":
        case "dot":
        case "cross":
        case "distance":
        case "atan2":
        case "reflect":
        case "mod":
            return 2;
        case "mix":
        case "smoothstep":
        case "clamp":
            return 3;
    }
    return 0;
}