// The pipeline cache (pipeline_cache.c3): what a render made is listed, and a list loaded is
// compiled in the background till Renderer.compiling says none are left. `just test pipeline_cache`
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

// `text` with a comment at the start of its node program's vertex hook: another program.
fn String with_comment(String text) {
    // (new text each run: a program the driver's cache has from an earlier run compiles
    // before compiling() is asked)
    String comment = string::tformat("// another %d\n", (long)time::now());
    usz at = (usz)text.index_of("\nN ")!! + 1;
    usz end = at + (usz)text[at..].index_of_char('\n')!!;
    String[] f = text[at:end - at].tsplit(" ");
    int vertex = f[2].to_int()!!;
    String line = string::tformat("N %s %d %s %s %s %s %s %s", f[1], vertex + (int)comment.len, f[3], f[4], f[5], f[6],
                                  f[7], f[8]);
    return string::tformat("%s%s\n%s%s", text[:at], line, comment, text[end + 1..]);
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
    PerspectiveCamera* camera = drie::perspective_camera(50, 320.0f / 180, 0.1, 100);
    camera.position = { 0, 1, 4 };
    camera.look_at_vec({ 0, 0, 0 });
    RenderTarget* rt = drie::render_target(320, 180, 1, RGBA8);
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();

    String text = renderer.pipeline_cache(tmem);
    expect(text.starts_with("drie-pipelines 2\n"), "the list has its header");
    expect(count_lines(text, "N ") == 1, "it has the node material's program");
    expect(count_lines(text, "P -1 ") >= 1, "and the engine's pipeline");
    expect(count_lines(text, "P 0 ") >= 1, "and the node material's");

    // what the render made: listed, nothing to compile
    int listed = renderer.load_pipeline_cache(text);
    expect(listed == count_lines(text, "P ") + count_lines(text, "S "), "loading it takes every pipeline");
    expect(renderer.load_pipeline_cache(text) == 0, "loading it again adds none");
    renderer.render(scene, camera);
    drie::commit();
    expect(renderer.compiling()[0] == 0, "the pipelines this run made need no compiling");

    // a program no graph here prints: compiled in the background
    String other = with_comment(text);
    int more = renderer.load_pipeline_cache(other);
    int[2] n = renderer.compiling();
    expect(more > 0 && n[0] > 0, "a program not made yet compiles in the background");
    for (int i = 0; i < 1000 && renderer.compiling()[0] > 0; i++) (void)thread::sleep_ms(10);
    n = renderer.compiling();
    expect(n[0] == 0 && n[1] == listed + more, "till none are left");

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
