// The pipeline cache (pipeline_cache.c3): what a render made is listed, and a list loaded is
// compiled in the background till Renderer.compiling says none are left. A node material's
// pipelines are listed with the material shader's hooks only (Renderer.shaders): its
// generated programs are not, its key alone can't build them again. `just test pipeline_cache`
module pipeline_cache_test;
import drie, drie::nodes, sokol;
import std::io, std::time, std::math, std::thread;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn int count_lines(String text, String start) {
    int n;
    foreach (line : text.tsplit("\n")) {
        if (line.starts_with(start)) n++;
    }
    return n;
}

// Whether a P line of `text` ends with family `fam`.
fn bool has_family(String text, ShaderFamily fam) {
    foreach (line : text.tsplit("\n")) {
        if (line.starts_with("P ") && line.ends_with(string::tformat(" %d", fam.ordinal))) return true;
    }
    return false;
}

// `text` with a comment at the start of its node program's vertex functions: another program.
// (The N line: cuts, then the lengths of unsupported, vertex_funcs, frag_funcs and the 7 hooks;
// their texts follow it in that order.)
fn String with_comment(String text) {
    // (new text each run: a program the driver's cache has from an earlier run compiles
    // before compiling() is asked)
    String comment = string::tformat("// another %d\n", (long)time::now());
    usz at = (usz)text.index_of("\nN ")!! + 1;
    usz end = at + (usz)text[at..].index_of_char('\n')!!;
    String[] f = text[at:end - at].tsplit(" ");
    usz unsupported = (usz)f[2].to_int()!!;
    int funcs = f[3].to_int()!!;
    DString line = dstring::temp();
    line.appendf("N %s %s %d", f[1], f[2], funcs + (int)comment.len);
    foreach (x : f[4..]) line.appendf(" %s", x);
    usz texts = end + 1;
    return string::tformat("%s%s\n%s%s%s", text[:at], line.str_view(), text[texts:unsupported], comment, text[texts + unsupported..]);
}

fn int main(String[] args) {
    Renderer* renderer = drie::renderer(antialias: false, width: 320, height: 180);
    renderer.init_headless();
    Scene* scene = drie::scene();
    scene.add(drie::hemisphere_light(0xffffff, 0x404040, 1));
    Mesh* plain = drie::mesh(drie::box_geometry(1, 1, 1), drie::mesh_standard_material(color: 0x808080));
    plain.position.x = -1;
    scene.add(plain);
    Material* m = drie::mesh_standard_material(color: 0xffffff);
    m.color_node = nodes::mix(nodes::color(0x2050ff), nodes::color(0xffc020), nodes::uv().x());
    Mesh* noded = drie::mesh(drie::box_geometry(1, 1, 1), m);
    noded.position.x = 1;
    scene.add(noded);
    // (a clearcoat: the material shader's PHYSICAL programs, which the list names as such)
    Material* coated = drie::mesh_standard_material(color: 0xffffff);
    coated.clearcoat = 1;
    coated.color_node = nodes::color(0x40c040);
    Mesh* coated_mesh = drie::mesh(drie::box_geometry(1, 1, 1), coated);
    coated_mesh.position = { 0, 1, 0 };
    scene.add(coated_mesh);
    PerspectiveCamera* camera = drie::perspective_camera(50, 320.0f / 180, 0.1, 100);
    camera.position = { 0, 1, 4 };
    camera.look_at_vec({ 0, 0, 0 });
    RenderTarget* rt = drie::render_target(320, 180, 1, RGBA8);
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();

    String text = renderer.pipeline_cache(tmem);
    // (the node material drawn with the material shader's hooks)
    bool classic = m.node_build.programs.len() == 0;
    expect(text.starts_with("drie-pipelines 5\n"), "the list has its header");
    expect(count_lines(text, classic ? "P -1 " : "G ") >= 1, "it has the engine material's pipeline (with its generated program)");
    if (classic) {
        expect(count_lines(text, "N ") == 2, "and the node materials' programs");
        expect(count_lines(text, "P 0 ") >= 1, "and the node material's pipeline");
        expect(has_family(text, ShaderFamily.PHYSICAL), "the clearcoat's pipeline is listed with the PHYSICAL programs");
    } else {
        expect(count_lines(text, "N ") == 0 && count_lines(text, "P ") == 0, "and not the node material's generated ones");
    }

    // what the render made: listed, nothing to compile
    int listed = renderer.load_pipeline_cache(text);
    expect(listed == count_lines(text, "P ") + count_lines(text, "S ") + count_lines(text, "G ") + count_lines(text, "H "), "loading it takes every pipeline");
    expect(renderer.load_pipeline_cache(text) == 0, "loading it again adds none");
    renderer.render(scene, camera);
    drie::commit();
    expect(renderer.compiling()[0] == 0, "the pipelines this run made need no compiling");

    // a program no graph here prints: compiled in the background
    if (classic) {
        String other = with_comment(text);
        int more = renderer.load_pipeline_cache(other);
        int[2] n = renderer.compiling();
        expect(more > 0 && n[0] > 0, "a program not made yet compiles in the background");
        for (int i = 0; i < 1000 && renderer.compiling()[0] > 0; i++) (void)thread::sleep_ms(10);
        n = renderer.compiling();
        expect(n[0] == 0 && n[1] == listed + more, "till none are left");
    }

    // a frame after: the meshes draw as before
    renderer.render(scene, camera);
    drie::commit();
    expect(renderer.info.calls >= 2, "the meshes still draw");

    if (failures) {
        io::printfn("%d failed", failures);
        return 1;
    }
    return 0;
}
