// The program pack (program_pack.c3): programs compiled ahead of time and made from their
// bytecode. Metal only (xcrun metal compiles the pack).
//
// - A run renders a scene (a node material, a plain one, shadows) and saves its programs'
//   sources (drie::dump_programs). Each source is compiled with xcrun metal, and the test
//   writes a pack of them.
// - A second process loads the pack and renders the same scene: every shader stage of the
//   frame comes from the pack (none compiled from source), and the picture is the first's.
// - A pack for another target, and one whose bytecode doesn't load, are refused: that run
//   compiles everything from source and the picture is the same.
//
// Run it with `just test program_pack`. It writes tests/build/program_pack.png.
module program_pack_test;
import drie, drie::nodes, drie::readback, drie::png, sokol;
import std::io, std::io::file, std::io::path, std::math, std::os::process, std::collections::list;

const int W = 192;
const int H = 128;
const String DIR = "tests/build/program_pack";
const String PICTURE = "tests/build/program_pack/first.rgba";

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

// The scene drawn into a W x H target, its pixels (RGBA8) into `px`.
fn void render(char[] px) {
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
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
    m.color_node = nodes::mix(nodes::color(0x2050ff), nodes::color(0xffc020), nodes::uv().x());
    Mesh* box = drie::mesh(drie::box_geometry(1, 1, 1), m);
    box.position = { -0.8f, 0.5f, 0 };
    box.cast_shadow = true;
    scene.add(box);
    Mesh* plain = drie::mesh(drie::box_geometry(1, 1, 1), drie::mesh_standard_material(color: 0xc04040));
    plain.position = { 0.8f, 0.5f, 0 };
    plain.cast_shadow = true;
    scene.add(plain);
    PerspectiveCamera* camera = drie::perspective_camera(50, (float)W / H, 0.1f, 50);
    camera.position = { 0, 3, 5 };
    camera.look_at_vec({ 0, 0.3f, 0 });
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, W, H, px.ptr);
}

fn void put_u32(List{char}* out, uint v) {
    for (int i = 0; i < 4; i++) out.push((char)(v >> (i * 8)));
}

// A pack of the metallibs in `libs` (by their sources' hashes), for `target`.
fn char[] pack(ulong[] hashes, char[][] libs, String target) {
    List{char} out;
    out.init(mem);
    foreach (c : "drie-programs 1\n") out.push(c);
    for (int i = 0; i < 16; i++) out.push(i < target.len ? target[i] : 0);
    put_u32(&out, (uint)hashes.len);
    put_u32(&out, 0);
    uint at = (uint)(16 + 16 + 8 + 16 * hashes.len);
    foreach (i, h : hashes) {
        put_u32(&out, (uint)h);
        put_u32(&out, (uint)(h >> 32));
        put_u32(&out, at);
        put_u32(&out, (uint)libs[i].len);
        at += (uint)libs[i].len;
    }
    foreach (lib : libs) foreach (c : lib) out.push(c);
    return out.array_view();
}

// The second run: the pack `file` loaded, the scene rendered and compared with the first's.
fn int second(String file, String want) {
    char[] data = file::load(mem, file)!!;
    int n = drie::load_program_pack(data);
    char[] px = mem::new_array(char, (sz)W * H * 4);
    render(px);
    int[2] counts = drie::program_pack_counts();
    io::printfn("%s: %d programs in the pack; %d stages from it, %d from source", file, n, counts[0], counts[1]);
    char[] first = file::load(mem, PICTURE)!!;
    int differ;
    foreach (i, c : px) if (math::abs((int)c - first[i]) > 2) differ++;
    switch (want) {
        case "all":
            expect(n > 0 && counts[0] > 0 && counts[1] == 0, "every shader stage of the frame comes from the pack");
            (void)png::write("tests/build/program_pack.png", px, W, H, channels: 4);
        case "none":
            expect(counts[0] == 0 && counts[1] == 0, "the pack is refused: every stage compiled from source");
    }
    expect(differ == 0, string::tformat("the picture is the first run's (%d bytes differ)", differ));
    return failures ? 1 : 0;
}

fn int main(String[] args) {
    if (args.len == 4 && args[1] == "--pack") return second(args[2], args[3]);

    // the first run: sources saved (none left from an earlier run), the picture kept
    (void)path::mkdir(DIR, true);
    foreach (name : path::ls(tmem, (Path)DIR)!!) (void)file::delete(string::tformat("%s/%s", DIR, (String)name));
    char[] px = mem::new_array(char, (sz)W * H * 4);
    drie::dump_programs(DIR);
    render(px);
    file::save(PICTURE, px)!!;

    // each source compiled
    List{ulong} hashes;
    hashes.init(mem);
    List{char[]} libs;
    libs.init(mem);
    foreach (name : path::ls(tmem, (Path)DIR)!!) {
        String s = (String)name;
        if (!s.ends_with(".metal")) continue;
        String src = string::tformat("%s/%s", DIR, s);
        String out = string::tformat("%s/%s.metallib", DIR, s[:s.len - 6]);
        int? code = process::run({ "xcrun", "-sdk", "macosx", "metal", "-o", out, src }, INHERIT_ENV);
        if (catch code) {
            io::printfn("FAIL can't run xcrun metal");
            return 1;
        }
        expect(code == 0, string::tformat("%s compiles", s));
        hashes.push(s[:s.len - 6].to_ulong(16)!!);
        libs.push(file::load(mem, out)!!);
    }
    expect(hashes.len() >= 6, string::tformat("the frame's programs are saved (%d stages)", hashes.len()));
    char[] good = pack(hashes.array_view(), libs.array_view(), "metal-macos");
    file::save(DIR.tconcat("/good.pack"), good)!!;
    file::save(DIR.tconcat("/ios.pack"), pack(hashes.array_view(), libs.array_view(), "metal-ios"))!!;
    // (bytes that aren't a metallib)
    foreach (&lib : libs) foreach (&c : *lib) *c = 0x5a;
    file::save(DIR.tconcat("/broken.pack"), pack(hashes.array_view(), libs.array_view(), "metal-macos"))!!;

    String[3] packs = { "good", "ios", "broken" };
    foreach (p : packs) {
        String file = string::tformat("%s/%s.pack", DIR, p);
        int? code = process::run({ args[0], "--pack", file, p == "good" ? "all" : "none" }, INHERIT_ENV | INHERIT_STDIO);
        expect(@ok(code) && (code ?? 1) == 0, string::tformat("the run with the %s pack", p));
    }
    if (failures) {
        io::printfn("%d failed", failures);
        return 1;
    }
    io::printn("all passed");
    return 0;
}
