// The pipeline archive (pipeline_cache.c3, sokol_warm.h): pipelines compiled to the GPU's code
// in one run, made without compiling in the next. Metal only.
//
// - A run with no archive file renders a scene (a node material, a plain one, a shadow) with
//   compiles in the background: its pipelines are compiled and added (one compiled in the
//   background is then made by the frame from the archive), and the archive is saved.
// - A second process loads it and renders the same scene: every pipeline comes from the
//   archive (MTLPipelineOptionFailOnBinaryArchiveMiss finds it), none is compiled, and the
//   picture is the first's.
// - A third renders the scene with one more node material: only its pipelines are compiled,
//   and saved into the archive's second file (ARCHIVE.1); a fourth finds every pipeline in the
//   two files.
// - A file that isn't an archive is taken for none: everything is compiled, and the picture is
//   the same.
//
// Run it with `just test pipeline_archive`.
module pipeline_archive_test;
import drie, drie::nodes, drie::readback, sokol;
import std::io, std::io::file, std::io::path, std::math, std::os::process, std::thread;

const int W = 160;
const int H = 120;
const String DIR = "tests/build/pipeline_archive";
const String ARCHIVE = "tests/build/pipeline_archive/pipelines.archive";
const String PICTURE = "tests/build/pipeline_archive/first.rgba";
const String PICTURE_MORE = "tests/build/pipeline_archive/more.rgba";

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

// The scene (with one more box, if `more`) rendered till no draw waits for its program, with
// the archive `archive`; its pixels into `px`.
fn void render(String archive, char[] px, bool more) {
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.compile_in_background = true;
    renderer.use_pipeline_archive(archive);
    Scene* scene = drie::scene();
    scene.add(drie::hemisphere_light(0xffffff, 0x404040, 0.5));
    SpotLight* l = drie::spot_light(0xffffff, 40, 0, 0.6f, 0.2f, 2);
    l.position = { 1, 5, 2 };
    l.cast_shadow = true;
    scene.add(l);
    scene.add(l.target);
    Mesh* floor = drie::mesh(drie::plane_geometry(10, 10), drie::mesh_standard_material(color: 0x808080, roughness: 1));
    floor.rotation.x = -(float)math::PI / 2;
    floor.receive_shadow = true;
    scene.add(floor);
    Material* m = drie::mesh_standard_material(color: 0xffffff);
    m.color_node = nodes::mix(nodes::color(0x20a0ff), nodes::color(0xff6020), nodes::uv().y());
    Mesh* box = drie::mesh(drie::box_geometry(1, 1, 1), m);
    box.position = { -0.8f, 0.5f, 0 };
    box.cast_shadow = true;
    scene.add(box);
    Mesh* plain = drie::mesh(drie::box_geometry(1, 1, 1), drie::mesh_standard_material(color: 0x40c040));
    plain.position = { 0.8f, 0.5f, 0 };
    plain.cast_shadow = true;
    scene.add(plain);
    if (more) {
        Material* m2 = drie::mesh_standard_material(color: 0xffffff);
        m2.color_node = nodes::color(0xffe080) * nodes::uv().x();
        Mesh* third = drie::mesh(drie::box_geometry(0.6f, 0.6f, 0.6f), m2);
        third.position = { 0, 0.3f, 1.2f };
        scene.add(third);
    }
    PerspectiveCamera* camera = drie::perspective_camera(50, (float)W / H, 0.1f, 50);
    camera.position = { 0, 3, 5 };
    camera.look_at_vec({ 0, 0.3f, 0 });
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
    for (int i = 0; i < 3000 && renderer.info.waiting > 0; i++) {
        thread::sleep_ms(10);
        renderer.render(scene, camera);
        drie::commit();
    }
    // (a frame more: a shadow tile drawn while a program waited is drawn again)
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, W, H, px.ptr);
    expect(renderer.save_pipeline_archive(), "the archive is saved");
}

fn int run(String archive, String want) {
    char[] px = mem::new_array(char, (sz)W * H * 4);
    bool more = want.starts_with("more");
    render(archive, px, more);
    int[2] n = drie::pipeline_archive_counts();
    io::printfn("%s: %d pipelines from the archive, %d compiled and added", archive, n[0], n[1]);
    switch (want) {
        case "new":
            expect(n[1] > 0, "with no archive yet, the pipelines are compiled and added");
            expect(file::exists(archive), "and the archive is written");
            file::save(PICTURE, px)!!;
            return failures ? 1 : 0;
        case "more":
            expect(n[0] > 0 && n[1] > 0, "with one more material, only its pipelines are compiled and added");
            expect(file::exists(ARCHIVE.tconcat(".1")), "into the archive's second file");
            file::save(PICTURE_MORE, px)!!;
            return failures ? 1 : 0;
        case "more_loaded":
            expect(n[0] > 0 && n[1] == 0, "with its two files, every pipeline comes from the archive");
            expect(!file::exists(ARCHIVE.tconcat(".2")), "and no file is added");
        case "loaded":
            expect(n[0] > 0 && n[1] == 0, "with the archive, every pipeline comes from it");
        case "broken":
            expect(n[1] > 0, "a file that isn't an archive is none: the pipelines are compiled");
    }
    char[] first = file::load(mem, more ? PICTURE_MORE : PICTURE)!!;
    int differ;
    foreach (i, c : px) if (math::abs((int)c - first[i]) > 2) differ++;
    expect(differ == 0, string::tformat("the picture is the first run's (%d bytes differ)", differ));
    return failures ? 1 : 0;
}

fn int main(String[] args) {
    if (args.len == 4 && args[1] == "--run") return run(args[2], args[3]);
    (void)path::mkdir(DIR, true);
    (void)file::delete(ARCHIVE);
    (void)file::delete(ARCHIVE.tconcat(".1"));
    (void)file::delete(ARCHIVE.tconcat(".2"));
    (void)file::delete(PICTURE);
    String broken = DIR.tconcat("/broken.archive");
    file::save(broken, "not an archive")!!;
    (void)file::delete(broken.tconcat(".1"));
    String[2][5] runs = { { ARCHIVE, "new" }, { ARCHIVE, "loaded" }, { ARCHIVE, "more" }, { ARCHIVE, "more_loaded" }, { broken, "broken" } };
    foreach (r : runs) {
        // (each run its own process: the archive is loaded once, at the GPU's setup)
        int? code = process::run({ args[0], "--run", r[0], r[1] }, INHERIT_ENV | INHERIT_STDIO);
        expect(@ok(code) && (code ?? 1) == 0, string::tformat("the %s run", r[1]));
    }
    if (failures) {
        io::printfn("%d failed", failures);
        return 1;
    }
    io::printn("all passed");
    return 0;
}
