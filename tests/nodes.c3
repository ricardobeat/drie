// Node materials (nodes.c3, drie::nodes; three's TSL): shaders built from graphs at runtime.
// Five spheres over a floor, lit by a shadow casting sun:
//   1. color_node: a checker texture, four times across, times a uniform tint (red, then green
//      in the second picture: Node.set without a rebuild)
//   2. a procedural stripe of two colors from the uv, and roughness_node going from smooth at
//      the bottom to rough at the top
//   3. position_node: rings pushed out along the normals, moving with time; the shadow on the
//      floor (the shadow map's program) and the depth prepass see them too
//   4. normal_node: vertical ridges in the shading of a smooth metal sphere, and an
//      emissive_node orange rim (fresnel)
//   5. color_node from a varying of the position in the model
// then statements and integers against a constant, a ball that a uniform discards along
// with its shadow, graph functions (nodes::function) against the same graphs inlined, and a
// seq() reading normal_world() in two steps that bind it to different normals.
//
//   just test nodes [out.png]   (tests/build/nodes.png; nodes_2.png: the second picture)
module nodes_test;
import drie, drie::nodes, drie::readback, drie::png, sokol;
import std::io, std::math;

const int W = 960;
const int H = 360;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn Texture* checker() {
    int n = 64;
    char[] px = mem::new_array(char, (usz)(n * n * 4));
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
            char c = ((x / 8 + y / 8) % 2) == 0 ? 255 : 40;
            usz i = (usz)((y * n + x) * 4);
            px[i:4] = (char[4]) { c, c, c, 255 }[..];
        }
    }
    Texture* t = drie::data_texture(px, n, n);
    t.wrap_s = REPEAT_WRAPPING;
    t.wrap_t = REPEAT_WRAPPING;
    return t;
}

// The pixel where `p` (world) lands.
fn int[2] pixel_of(PerspectiveCamera* camera, Vector3 p) {
    drie::Mat4 vp = camera.projection_matrix.mul(camera.matrix_world_inverse);
    float[4] c;
    for (int r = 0; r < 4; r++) c[r] = vp[0][r] * p.x + vp[1][r] * p.y + vp[2][r] * p.z + vp[3][r];
    return { (int)((c[0] / c[3] * 0.5f + 0.5f) * W), (int)((0.5f - c[1] / c[3] * 0.5f) * H) };
}

fn float[3] rgb_at(char[] px, int[2] p) {
    float[3] sum;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            usz i = ((usz)(p[1] + dy) * W + (usz)(p[0] + dx)) * 4;
            for (int k = 0; k < 3; k++) sum[k] += (float)px[i + (usz)k] / 25;
        }
    }
    return sum;
}

// The material's generated programs (materials_nodes.c3), and of what kind: the main pass's
// with the depth prepass's, the shadow maps', for a target with MRT. None with an application's
// material shader (Renderer.shaders), whose node programs NodeProgram holds.
struct Generated {
    int count;
    bool depth;
    bool shadow;
    bool mrt;
}

fn Generated generated(Material* mat) {
    Generated g;
    NodeBuild* b = mat.node_build;
    if (!b || !b.programs.is_initialized()) return g;
    foreach (mp : b.programs.tvalues()) {
        g.count++;
        if (mp.depth && mp.depth.shader.id) g.depth = true;
        if (mp.key.shadow && mp.program.shader.id) g.shadow = true;
        if (mp.key.mrt && mp.program.shader.id) g.mrt = true;
    }
    return g;
}

fn int count_of(String s, String what) {
    int n;
    for (usz i = 0; i + what.len <= s.len; i++) {
        if (s[i:what.len] == what) n++;
    }
    return n;
}

fn Node inline_hash(Node p) => nodes::fract(nodes::sin(nodes::dot(p, nodes::vec2(127.1f, 311.7f))) * 43758.5453f);

// tests' graph functions, as C3 functions making the nodes
fn Node inline_noise(Node p, Node scale) {
    Node np = p * scale;
    Node i = nodes::floor(np);
    Node f = nodes::fract(np);
    Node u = f * f * (3.0f - f * 2.0f);
    Node shade = nodes::variable(nodes::mix(nodes::mix(inline_hash(i), inline_hash(i + nodes::vec2(1, 0)), u.x()),
                                            nodes::mix(inline_hash(i + nodes::vec2(0, 1)), inline_hash(i + nodes::vec2(1, 1)), u.x()), u.y()));
    return nodes::seq(nodes::block(shade, nodes::if_then(shade.greater_than(nodes::num(0.9f)),
        nodes::block(shade.assign(shade * 0.5f)))), shade);
}

fn bool save(String path, char[] px) {
    if (png::write(path, px, W, H)) return true;
    io::printfn("could not write %s", path);
    return false;
}

fn int main(String[] args) {
    String out_path = args.len > 1 ? args[1] : "tests/build/nodes.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.auto_time = false;
    renderer.time = 0.5f;
    // (the depth prepass for every cut-out, however small on the screen)
    renderer.depth_prepass_pixels = 0.01f;
    Scene* scene = drie::scene();
    scene.background = 0x2e3440;
    scene.add(drie::hemisphere_light(0xffffff, 0x404040, 0.8));
    DirectionalLight* sun = drie::directional_light(0xffffff, 2.5);
    sun.position = { -3, 6, 4 };
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow_extent = 8;
    scene.add(sun);
    Mesh* floor = drie::mesh(drie::plane_geometry(30, 30), drie::mesh_standard_material(color: 0x909090));
    floor.rotation.x = -(float)math::PI / 2;
    floor.position.y = -1;
    floor.receive_shadow = true;
    scene.add(floor);

    Material*[5] mats;
    for (int i = 0; i < 5; i++) mats[i] = drie::mesh_standard_material(color: 0xffffff, roughness: 0.5);

    // 1. a texture times a uniform
    Node tint = nodes::uniform_color(0xff4040);
    mats[0].color_node = nodes::texture(checker(), nodes::uv() * nodes::vec2(4, 2)).xyz() * tint;

    // 2. stripes from the uv, rougher upwards
    Node stripe = nodes::step(0.5f, nodes::fract((nodes::uv().x() + nodes::uv().y()) * 6));
    mats[1].color_node = nodes::mix(nodes::color(0x2050ff), nodes::color(0xffc020), stripe);
    mats[1].roughness_node = nodes::mix(0.05f, 1.0f, nodes::uv().y());
    mats[1].metalness_node = nodes::num(0.3f);

    // 3. rings pushed out along the normals, moving with time
    Node p = nodes::position_local();
    Node wave = nodes::sin(p.y() * 14 + nodes::time() * 3);
    mats[2].position_node = p + nodes::normal_local() * (nodes::max(wave, 0) * 0.15f);
    mats[2].color_node = nodes::color(0x40c080);
    // (cut out where its map's alpha is under a half, nowhere: so it goes through the depth
    // prepass, whose program must move the vertices as the main pass's does, or the main
    // pass's EQUAL depth test would leave holes)
    mats[2].map = checker();
    mats[2].alpha_test = 0.5f;

    // 4. ridges in the normal, a fresnel rim that glows
    Node pw = nodes::position_world();
    Node ridge = nodes::sin(pw.x() * 40) * 0.35f;
    mats[3].normal_node = nodes::normalize(nodes::normal_world() + nodes::vec3(ridge, 0, 0));
    mats[3].metalness_node = nodes::num(1);
    mats[3].roughness_node = nodes::num(0.25f);
    Node facing = nodes::saturate(nodes::dot(nodes::normal_world(), nodes::view_direction()));
    mats[3].emissive_node = nodes::color(0xff6010) * nodes::pow(nodes::one_minus(facing), 3) * 2;

    // 5. the position in the model, carried to the fragments
    mats[4].color_node = nodes::varying(nodes::position_local()) * 0.5f + 0.5f;

    Mesh*[5] balls;
    foreach (i, m : mats) {
        balls[i] = drie::mesh(drie::sphere_geometry(0.8f, 64, 32), m);
        balls[i].position = { ((float)i - 2) * 2, 0, 0 };
        balls[i].cast_shadow = true;
        balls[i].receive_shadow = true;
        scene.add(balls[i]);
    }

    // every function and operation of drie::nodes at once (so that each prints into code the
    // backend compiles), on a small box in the corner
    Material* all = drie::mesh_standard_material(color: 0xffffff);
    {
        Node u = nodes::uv();
        Node x = u.x() + 0.25f;
        Node v3 = nodes::vec3(u, 0.5f);
        Node acc = nodes::sin(x) + nodes::cos(x) + nodes::tan(x) + nodes::asin(x * 0.5f) + nodes::acos(x * 0.5f)
            + nodes::atan(x) + nodes::exp(x) + nodes::exp2(x) + nodes::log(x) + nodes::log2(x) + nodes::sqrt(x)
            + nodes::inverse_sqrt(x) + nodes::abs(x) + nodes::sign(x) + nodes::floor(x) + nodes::ceil(x)
            + nodes::fract(x) + nodes::round(x) + nodes::trunc(x) + nodes::length(v3) + nodes::saturate(x)
            + nodes::dfdx(x) + nodes::dfdy(x) + nodes::fwidth(x) + nodes::pow(x, 2) + nodes::min(x, 0.5f)
            + nodes::max(x, 0.5f) + nodes::step(0.5f, x) + nodes::dot(v3, v3) + nodes::distance(v3, nodes::vec3(1))
            + nodes::atan2(x, 0.5f) + nodes::mod(x, 0.3f) + nodes::clamp(x) + nodes::smoothstep(0, 1, x)
            + nodes::one_minus(x) + nodes::negate(x) + (1 - x) + (1 / x) + x / 2 + x - 1;
        Node vec = nodes::normalize(v3) + nodes::cross(v3, nodes::vec3(0, 1, 0)) + nodes::reflect(v3, nodes::normal_world())
            + nodes::refract(nodes::normalize(v3), nodes::normal_world(), 0.7f) + nodes::faceforward(v3, v3, v3)
            + nodes::mix(v3, nodes::vec3(1, 0, 0), 0.5f) + nodes::fract(v3 * 3);
        Node cond = nodes::logical_and(x.greater_than(nodes::num(0.5f)), nodes::logical_not(x.equal(nodes::num(0.7f))));
        cond = nodes::logical_or(cond, nodes::less_than_equal(x, 0.1f));
        Node m = nodes::model_matrix() * nodes::view_matrix() * nodes::uniform_matrix(drie::mat4_identity());
        Node clip = nodes::projection_matrix() * (m * nodes::vec4(nodes::position_world(), 1));
        Node misc = nodes::vertex_color().x() + nodes::frag_coord().x() * 0.001f + nodes::time()
            + nodes::convert(nodes::front_facing(), FLOAT) + nodes::length(nodes::camera_position()) * 0.01f
            + nodes::uniform_vec2(1, 2).y() + nodes::uniform_vec4(1, 2, 3, 4).w() + nodes::uniform(1)
            + nodes::uv1().x() + clip.w() * 0.001f + nodes::length(nodes::normal_geometry())
            + nodes::convert(nodes::bool_(true), FLOAT) + nodes::varying(nodes::normal_local()).x()
            + nodes::select(cond, 1, 0) + nodes::texture(checker(), u, nodes::num(1)).x();
        all.color_node = nodes::vec4(nodes::fract(vec * 0.1f + acc * 0.01f + misc * 0.01f), 1);
        all.position_node = nodes::position_local() + nodes::normal_local() * (nodes::sin(nodes::position_world().x()) * 0.01f)
            + nodes::uniform_vec3({ 0, 0, 0 });
    }
    all.output_node = nodes::vec4(nodes::output().xyz() * 0.9f, nodes::output().w());
    Mesh* sink = drie::mesh(drie::box_geometry(0.4f, 0.4f, 0.4f), all);
    sink.position = { 4.6f, -0.8f, 1.5f };
    scene.add(sink);

    PerspectiveCamera* camera = drie::perspective_camera(40, (float)W / H, 0.1f, 100);
    camera.position = { 0, 2.2f, 9 };
    camera.look_at_vec({ 0, -0.2f, 0 });
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    char[] px = mem::new_array(char, (usz)W * H * 4);
    char[] px2 = mem::new_array(char, (usz)W * H * 4);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, W, H, px.ptr);
    if (!save(out_path, px)) return 1;

    NodeProgram* rings = mats[2].node_build.program;
    Generated gen = generated(mats[2]);
    expect(gen.count ? gen.depth : (renderer.deferred
               ? rings.shaders[ShaderFamily.GBUFFER.ordinal][NodeVariant.MAIN.ordinal].id != 0
               : rings.shaders[ShaderFamily.PBR.ordinal][NodeVariant.DEPTH.ordinal].id != 0),
           "the third ball through the gbuffer or depth prepass, with its own program");
    expect(gen.count ? gen.shadow : rings.shadow_pipeline.id != 0, "the third ball's shadow, with its own program");
    NodeBuild* build = mats[0].node_build;
    tint.set_color(0x40ff40);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, W, H, px2.ptr);
    String second = out_path.ends_with(".png") ? string::tformat("%s_2.png", out_path[:^4]) : string::tformat("%s_2", out_path);
    if (!save(second, px2)) return 1;

    // the first ball's lit side: red, then green
    int[2] at = pixel_of(camera, { -4.3f, 0.3f, 0.6f });
    float[3] a = rgb_at(px, at);
    float[3] b = rgb_at(px2, at);
    io::printfn("ball 1 at %s: %s, then %s", at, a, b);
    expect(a[0] > a[1] + 30, "the tint uniform: red");
    expect(b[1] > b[0] + 30, "Node.set: green, the next render");
    expect(mats[0].node_build == build, "Node.set rebuilds nothing");
    float[3] e = rgb_at(px, pixel_of(camera, { 4, 0.3f, 0.6f }));
    io::printfn("ball 5: %s", e);
    expect(e[0] > 150 && e[1] > 100, "the varying: the model's position as a color");

    // the other programs: without the depth prepass, with VSM shadows, into a target with
    // MRT; and the shadows cast toward the camera, by a low sun behind the balls (the third's
    // rings in its shadow's outline)
    tint.set_color(0xff4040);
    String[4] names = { "no depth prepass", "VSM shadows", "MRT", "low sun" };
    foreach (k, name : names) {
        renderer.depth_prepass = k != 0;
        renderer.shadow_type = k == 1 ? VSM : PCF;
        // (MRT: half floats, built and drawn, not compared)
        RenderTarget* target = k == 2 ? drie::render_target(W, H, 1, RGBA16F, mrt: true) : rt;
        if (k == 3) sun.position = { 0, 2.5f, -10 };
        renderer.set_render_target(target);
        renderer.render(scene, camera);
        drie::commit();
        if (k == 2) {
            // (solid: the opaque program; with deferred, the gbuffer opaque program is reused for MRT)
            ShaderFamily fam = renderer.deferred ? ShaderFamily.GBUFFER : ShaderFamily.PBR;
            NodeVariant v = renderer.deferred ? NodeVariant.OPAQUE : NodeVariant.OPAQUE_MRT;
            SgShader[8] built = mats[0].node_build.program.shaders[fam.ordinal];
            Generated g = generated(mats[0]);
            expect(g.count ? g.mrt : built[v.ordinal].id != 0, "MRT: the node materials' programs for it");
            continue;
        }
        readback::read_image(target.texture_image, W, H, px2.ptr);
        String path = string::tformat("%s_%d.png", out_path.ends_with(".png") ? out_path[:^4] : out_path, k + 3);
        if (!save(path, px2)) return 1;
        if (k == 3) continue;
        // the balls as in the first picture
        float worst;
        for (int i = 0; i < 5; i++) {
            float[3] x = rgb_at(px, pixel_of(camera, { ((float)i - 2) * 2 - 0.3f, 0.3f, 0.6f }));
            float[3] y = rgb_at(px2, pixel_of(camera, { ((float)i - 2) * 2 - 0.3f, 0.3f, 0.6f }));
            for (int c = 0; c < 3; c++) worst = math::max(worst, math::abs(x[c] - y[c]));
        }
        expect(worst < 12, string::tformat("%s: the balls as in the first picture (%.1f)", name, worst));
    }
    // statements and integers: a loop with a break, a variable, uints' bits and an int's
    // remainder, unlit, against the same value as a constant (0.5, 0.5, 0.5)
    {
        Scene* flat = drie::scene();
        Node sum = nodes::variable(nodes::num(0));
        Node i = nodes::loop_index();
        Node half = nodes::seq(nodes::block(
            sum,
            nodes::loop(i, 0, 8, nodes::block(
                sum.assign(sum + 0.125f),
                nodes::if_then(i.equal(nodes::int_(3)), nodes::block(nodes::break_()))))),
            sum);
        Node h = nodes::variable(nodes::uint_(0xdeadbeef));
        Node bits = nodes::seq(nodes::block(h, h.assign(nodes::bit_xor(nodes::shift_right(h, nodes::uint_(28)), nodes::uint_(0)))), h);
        Node top = nodes::convert(bits, FLOAT) / 26;
        Node r = nodes::convert(nodes::rem(nodes::int_(7), nodes::int_(4)), FLOAT) / 6;
        Material* built = drie::mesh_basic_material(color: 0xffffff);
        built.color_node = nodes::vec3(half, top, r);
        Material* plain = drie::mesh_basic_material(color: 0xffffff);
        plain.color_node = nodes::vec3(0.5f);
        Mesh* left = drie::mesh(drie::sphere_geometry(0.8f, 32, 16), built);
        left.position = { -1.5f, 0, 0 };
        Mesh* right = drie::mesh(drie::sphere_geometry(0.8f, 32, 16), plain);
        right.position = { 1.5f, 0, 0 };
        flat.add(left);
        flat.add(right);
        PerspectiveCamera* cam = drie::perspective_camera(40, (float)W / H, 0.1f, 100);
        cam.position = { 0, 0, 6 };
        cam.look_at_vec({ 0, 0, 0 });
        renderer.set_render_target(rt);
        renderer.render(flat, cam);
        drie::commit();
        readback::read_image(rt.texture_image, W, H, px2.ptr);
        float[3] x = rgb_at(px2, pixel_of(cam, { -1.5f, 0, 0.8f }));
        float[3] y = rgb_at(px2, pixel_of(cam, { 1.5f, 0, 0.8f }));
        io::printfn("statements: %s, constant: %s", x, y);
        expect(math::abs(x[0] - y[0]) < 2 && math::abs(x[1] - y[1]) < 2 && math::abs(x[2] - y[2]) < 2 && y[0] > 100,
            "statements and integers: as the constant");
    }
    // a discard that a uniform decides: the ball and its shadow go together (shadow_nodes)
    {
        Scene* lit = drie::scene();
        lit.add(drie::hemisphere_light(0xffffff, 0x404040, 0.5));
        DirectionalLight* key = drie::directional_light(0xffffff, 2.5);
        key.position = { -4, 6, 0 };
        key.cast_shadow = true;
        key.shadow_extent = 4;
        lit.add(key);
        Mesh* ground = drie::mesh(drie::plane_geometry(10, 10), drie::mesh_standard_material(color: 0x909090));
        ground.rotation.x = -(float)math::PI / 2;
        ground.receive_shadow = true;
        lit.add(ground);
        Node gone = nodes::uniform(0);
        Material* cut = drie::mesh_standard_material(color: 0xffffff);
        cut.color_node = nodes::seq(nodes::block(nodes::if_then(gone.greater_than(nodes::num(0.5f)), nodes::block(nodes::discard()))),
            nodes::color(0x2050ff));
        Mesh* ball = drie::mesh(drie::sphere_geometry(0.5f, 32, 16), cut);
        ball.position = { 0, 1, 0 };
        ball.cast_shadow = true;
        lit.add(ball);
        PerspectiveCamera* cam = drie::perspective_camera(40, (float)W / H, 0.1f, 100);
        cam.position = { 0, 5, 8 };
        cam.look_at_vec({ 0, 0, 0 });
        renderer.shadow_type = PCF;
        renderer.set_render_target(rt);
        float[3][2] shade;
        float[3][2] blue;
        for (int k = 0; k < 2; k++) {
            gone.set(k);
            renderer.render(lit, cam);
            drie::commit();
            readback::read_image(rt.texture_image, W, H, px2.ptr);
            shade[k] = rgb_at(px2, pixel_of(cam, { 0.67f, 0, 0 }));
            blue[k] = rgb_at(px2, pixel_of(cam, { 0, 1, 0.45f }));
        }
        io::printfn("uniform discard: the shadow %s, then %s; the ball %s, then %s", shade[0], shade[1], blue[0], blue[1]);
        expect(blue[0][2] > blue[0][0] + 40 && math::abs(blue[1][2] - blue[1][0]) < 20, "a discard a uniform decides: the ball, then none");
        expect(shade[1][0] > shade[0][0] + 30, "a discard a uniform decides: its shadow goes too");
    }
    // graph functions: a noise of a hash reading a uniform, called three times
    // in the fragment stage and once in the vertex stage; the same graphs inlined, made by C3
    // functions, draw the same picture; a function reads a texture
    {
        Scene* flat = drie::scene();
        Node scale = nodes::uniform(1.0f);
        Texture* tex = checker();
        NodeFunction hash = nodes::function("hash", FLOAT, VEC2);
        Node hp = hash.param(0);
        hash.returns(nodes::fract(nodes::sin(nodes::dot(hp, nodes::vec2(127.1f, 311.7f))) * 43758.5453f));
        NodeFunction noise = nodes::function("noise", FLOAT, VEC2);
        Node np = noise.param(0) * scale;
        Node ni = nodes::floor(np);
        Node nf = nodes::fract(np);
        Node nu = nf * nf * (3.0f - nf * 2.0f);
        Node shade = nodes::variable(nodes::mix(nodes::mix(hash.call(ni), hash.call(ni + nodes::vec2(1, 0)), nu.x()),
                                                nodes::mix(hash.call(ni + nodes::vec2(0, 1)), hash.call(ni + nodes::vec2(1, 1)), nu.x()), nu.y()));
        noise.returns(nodes::seq(nodes::block(shade, nodes::if_then(shade.greater_than(nodes::num(0.9f)),
            nodes::block(shade.assign(shade * 0.5f)))), shade));
        NodeFunction tinted = nodes::function("tinted", VEC3, VEC3, VEC2);
        tinted.returns(tinted.param(0) * nodes::texture(tex, tinted.param(1)).xyz());
        NodeFunction bulge = nodes::function("bulge", VEC3, VEC3, VEC3);
        bulge.returns(bulge.param(0) + bulge.param(1) * (noise.call(bulge.param(0).xy() * 4) * 0.1f));

        Material* called = drie::mesh_basic_material(color: 0xffffff);
        Node u = nodes::uv();
        called.color_node = tinted.call(nodes::vec3(noise.call(u * 8), noise.call(u * 16 + 3), noise.call(u * 32 + 7)), u * 2);
        called.position_node = bulge.call(nodes::position_local(), nodes::normal_local());
        Material* inlined = drie::mesh_basic_material(color: 0xffffff);
        inlined.color_node = nodes::vec3(inline_noise(u * 8, scale), inline_noise(u * 16 + 3, scale), inline_noise(u * 32 + 7, scale))
                           * nodes::texture(tex, u * 2).xyz();
        Node lp = nodes::position_local();
        inlined.position_node = lp + nodes::normal_local() * (inline_noise(lp.xy() * 4, scale) * 0.1f);
        Mesh* ball = drie::mesh(drie::sphere_geometry(1.2f, 64, 32), called);
        flat.add(ball);
        PerspectiveCamera* cam = drie::perspective_camera(40, (float)W / H, 0.1f, 100);
        cam.position = { 0, 0, 6 };
        cam.look_at_vec({ 0, 0, 0 });
        renderer.set_render_target(rt);
        renderer.render(flat, cam);
        drie::commit();
        readback::read_image(rt.texture_image, W, H, px.ptr);
        String vs;
        String fs;
        foreach (mp : called.node_build.programs.tvalues()) {
            if (mp.key.shadow) continue;
            vs = mp.program.vs;
            fs = mp.program.fs;
        }
        if (!fs.len && called.node_build.program) {
            NodeProgram* prog = called.node_build.program;
            vs = string::tformat("%s\n%s", prog.vertex_funcs, prog.vertex_body);
            fs = string::tformat("%s\n%s", prog.frag_funcs, prog.color_body);
        }
        ball.material = inlined;
        renderer.render(flat, cam);
        drie::commit();
        readback::read_image(rt.texture_image, W, H, px2.ptr);
        String fpath = string::tformat("%s_functions.png", out_path.ends_with(".png") ? out_path[:^4] : out_path);
        if (!save(fpath, px)) return 1;
        int worst;
        int differ;
        foreach (i, c : px) {
            int d = math::abs((int)c - (int)px2[i]);
            if (d) differ++;
            worst = math::max(worst, d);
        }
        int defined = count_of(fs, "float fn_noise(");
        io::printfn("functions: fs %d B (fn_noise defined %d times, called %d), vs %d B; against inlined: %d bytes differ, by up to %d",
                    fs.len, defined, count_of(fs, "fn_noise(") - defined, vs.len, differ, worst);
        expect(defined == 1 && count_of(fs, "fn_noise(") >= 4, "a function called three times prints once");
        expect(count_of(fs, "float fn_hash(") == 1,
               "a function a function calls prints once");
        expect(count_of(vs, "3 fn_bulge(") == 1 && count_of(vs, "float fn_noise(") == 1, "functions in the vertex stage");
        expect(differ == 0, "functions: the picture of the same graphs inlined");
    }
    // a seq() reading normal_world() in color_node (the surface's normal) and emissive_node
    // (normal_node's, straight up): its value in each step
    {
        Scene* dark = drie::scene();
        Node v = nodes::variable(nodes::normal_world());
        Node n = nodes::seq(nodes::block(v, v.assign(v * 1.0f)), v);
        Material* seqd = drie::mesh_standard_material(color: 0xffffff);
        seqd.normal_node = nodes::vec3(0, 1, 0);
        seqd.color_node = nodes::vec3(0) * n;
        seqd.emissive_node = n * 0.5f + 0.5f;
        Material* plain = drie::mesh_standard_material(color: 0x000000);
        plain.normal_node = nodes::vec3(0, 1, 0);
        plain.emissive_node = nodes::normal_world() * 0.5f + 0.5f;
        Mesh* left = drie::mesh(drie::sphere_geometry(0.8f, 32, 16), seqd);
        left.position = { -1.5f, 0, 0 };
        Mesh* right = drie::mesh(drie::sphere_geometry(0.8f, 32, 16), plain);
        right.position = { 1.5f, 0, 0 };
        dark.add(left);
        dark.add(right);
        PerspectiveCamera* cam = drie::perspective_camera(40, (float)W / H, 0.1f, 100);
        cam.position = { 0, 0, 6 };
        cam.look_at_vec({ 0, 0, 0 });
        renderer.set_render_target(rt);
        renderer.render(dark, cam);
        drie::commit();
        readback::read_image(rt.texture_image, W, H, px2.ptr);
        float worst;
        foreach (Vector3 q : (Vector3[]) { { 0, 0.5f, 0.6f }, { 0.5f, -0.4f, 0.4f }, { -0.5f, 0, 0.6f } }) {
            float[3] x = rgb_at(px2, pixel_of(cam, { q.x - 1.5f, q.y, q.z }));
            float[3] y = rgb_at(px2, pixel_of(cam, { q.x + 1.5f, q.y, q.z }));
            for (int c = 0; c < 3; c++) worst = math::max(worst, math::abs(x[c] - y[c]));
        }
        io::printfn("seq after a bind: up to %.1f from normal_world()", worst);
        expect(worst < 2, "a seq() reading normal_world() is printed again after normal_node binds it");
    }
    {
        // the hooks' text: a constant is written where it is read, and a value that two nodes
        // print alike is printed once
        Material* m = drie::mesh_standard_material(color: 0xffffff);
        Node first = nodes::sin(nodes::uv().x() * 7.5f);
        Node again = nodes::sin(nodes::uv().x() * 7.5f);
        m.color_node = nodes::vec3(first, again * 0.5f, 0.25f);
        NodeProgram* prog = drie::node_build_of(m).program;
        io::printfn("color hook:\n%s", prog.color_body);
        expect(count_of(prog.color_body, "sin(") == 1, "two nodes that print alike: one value");
        expect(count_of(prog.color_body, " = (7.5") == 0 && count_of(prog.color_body, "(7.50000000e+00)") == 1,
               "a constant written where it is read, not in a variable of its own");
    }
    io::printfn("saved %s, %s", out_path, second);
    return failures ? 1 : 0;
}
