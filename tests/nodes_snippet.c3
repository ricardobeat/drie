// Snippets (nodes_snippet.c3): material code written as text makes the same graph as the
// drie::nodes calls it stands for, and a mistake in it is an error with its place, no crash.
// No GPU: it compares the graphs node by node.
//
//   just test nodes_snippet
module nodes_snippet_test;
import drie, drie::nodes;
import std::io;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

// Whether `a` and `b` are the same graph (each node's op, type and values, and its inputs).
fn bool same(Node a, Node b) {
    if (a.id == b.id) return true;
    if (!a.id || !b.id) return false;
    NodeData* x = drie::node_data(a);
    NodeData* y = drie::node_data(b);
    if (x.op != y.op || x.type != y.type || x.kind != y.kind || x.count != y.count) return false;
    if (x.swizzle != y.swizzle || x.value != y.value || x.texture != y.texture) return false;
    for (int i = 0; i < 4; i++) {
        if (!same({ x.args[i] }, { y.args[i] })) return false;
    }
    return true;
}

fn void check(String src, Node want, SnippetName[] names = {}) {
    Node? got = nodes::snippet(src, names);
    if (catch got) {
        expect(false, string::tformat("%s (%s)", src, nodes::snippet_error()));
        return;
    }
    expect(same(got, want), src);
}

fn void error(String src, String want) {
    Node? got = nodes::snippet(src);
    if (try got) {
        expect(false, string::tformat("%s fails", src));
        return;
    }
    expect(nodes::snippet_error() == want, string::tformat("%s: \"%s\"", src, nodes::snippet_error()));
}

fn int main(String[] args) {
    Node amount = nodes::uniform(0.5f);
    Node floor_y = nodes::uniform(0);
    SnippetName[] names = { { "amount", amount }, { "floor_y", floor_y } };

    check("1 + 2 * 3", nodes::add(1, nodes::mul(2, 3)));
    check("(1 + 2) * 3", nodes::mul(nodes::add(1, 2), 3));
    check("-uv.x", nodes::uv().x().negate());
    check("uv.yx * 2.5e1", nodes::uv().swizzle("yx") * 25.0f);
    check("vec3(uv, 1)", nodes::vec3(nodes::uv(), 1));
    check("vec3(0.5)", nodes::vec3(0.5f));
    check("mix(material_color.rgb, vec3(0.2, 0.17, 0.12), amount)",
        nodes::mix(nodes::material_color().rgb(), nodes::vec3(0.2f, 0.17f, 0.12f), amount), names);
    check("clamp(time) + clamp(time, 0.2, 0.8)",
        nodes::clamp(nodes::time()) + nodes::clamp(nodes::time(), 0.2f, 0.8f));
    check("uv.x < 0.5 ? 1.0 : 0.0", nodes::select(nodes::uv().x().less_than(nodes::num(0.5f)), 1.0f, 0.0f));
    check("uv.x <= 0.5 && !(uv.y > 1)",
        nodes::logical_and(nodes::uv().x().less_than_equal(nodes::num(0.5f)), nodes::logical_not(nodes::uv().y().greater_than(nodes::num(1)))));
    check("time % 2", nodes::mod(nodes::time(), 2));
    check("// dirt\nlet h = position_world.y - floor_y;\nlet d = smoothstep(0.6, 0.0, h) * amount; // falls off\nd * d",
        nodes::mul(nodes::mul(nodes::smoothstep(0.6f, 0.0f, nodes::position_world().y() - floor_y), amount),
                   nodes::mul(nodes::smoothstep(0.6f, 0.0f, nodes::position_world().y() - floor_y), amount)), names);

    error("uv + vec3(1)", "1:4: vec2 + vec3: the sizes don't go together (.xyz, vec3(...))");
    error("let a = 1;\n  foor_y * 2", "2:3: no 'foor_y' here");
    error("uv.z", "1:4: .z of a vec2");
    error("vec3(uv, uv)", "1:1: vec3() of 4 components");
    error("mix(1, 2)", "1:1: mix() takes 3, not 2");
    error("noise(uv)", "1:1: no function 'noise' here");
    error("1 +", "1:4: an expression, not the end");
    error("(1 + 2", "1:7: ')'");
    error("1 2", "1:3: '2' after the expression");

    io::printfn("%s", failures ? "FAILED" : "all good");
    return failures ? 1 : 0;
}
