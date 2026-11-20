// Node materials (drie::nodes): shaders built from graphs while the program runs.
//
//   left    a torus that burns away: fractal noise from integer hashes, summed in a loop, a
//           discard under a threshold, and a glowing rim along the edge. The threshold is a
//           uniform, set every frame without a rebuild; the shadow burns away with it.
//   middle  a column twisted and swelled by position_node; its shadow twists too.
//   right   a ball whose color_node is replaced by another graph: keys 1 to 4, or space for
//           the next (every five seconds by itself). A graph is printed and compiled the
//           first time it is drawn; the program is kept, so the next time it is free.
//
//   just example nodes                  in a window (drag to orbit)
//   just example nodes --png out.png    one frame, headless (--time t, --graph k)
module nodes_example;
import drie, drie::nodes, drie::ui, drie::controls, drie::readback, drie::png, sokol;
import std::io, std::math;

const int W = 1280;
const int H = 720;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
OrbitControls* controls;
Material* ball;
Mesh* torus;
Mesh* column;
Node threshold;

const String[4] GRAPHS = { "stripes", "checker", "marble", "fresnel" };
int graph = -1;
int pending = 0;       // the graph to switch to before the next frame (-1: none)
double switched_at;
double build_ms = -1;  // the render that drew the new graph first

// ---- noise from integer hashes

// A number in 0..1 for each integer cell (pcg3d's mixing, Jarzynski and Olano 2020).
fn Node hash(Node cell) {
    Node v = nodes::convert(nodes::convert(cell, IVEC3), UVEC3) * nodes::uint_(1664525) + nodes::uint_(1013904223);
    Node x = v.x() + v.y() * v.z();
    Node y = v.y() + v.z() * x;
    Node z = v.z() + x * y;
    Node h = nodes::bit_xor(x + y * z, nodes::shift_right(z, nodes::uint_(16)));
    h = h * nodes::uint_(0x27d4eb2d);
    h = nodes::bit_xor(h, nodes::shift_right(h, nodes::uint_(15)));
    return nodes::convert(h, FLOAT) * (1.0f / 4294967296.0f);
}

// Value noise: the hashes of the eight cells around `p`, blended smoothly.
fn Node value_noise(Node p) {
    Node i = nodes::floor(p);
    Node f = nodes::fract(p);
    Node u = f * f * (3 - f * 2);
    Node[8] c;
    for (int k = 0; k < 8; k++) c[k] = hash(i + nodes::vec3(k & 1, (k >> 1) & 1, k >> 2));
    Node x0 = nodes::mix(nodes::mix(c[0], c[1], u.x()), nodes::mix(c[2], c[3], u.x()), u.y());
    Node x1 = nodes::mix(nodes::mix(c[4], c[5], u.x()), nodes::mix(c[6], c[7], u.x()), u.y());
    return nodes::mix(x0, x1, u.z());
}

// Four octaves of value noise, summed in a loop of the shader's: about 0.1 to 0.9.
fn Node fbm(Node p) {
    Node sum = nodes::variable(nodes::num(0));
    Node amp = nodes::variable(nodes::num(0.5f));
    Node q = nodes::variable(p);
    Node i = nodes::loop_index();
    return nodes::seq(nodes::block(
        sum, amp, q,
        nodes::loop(i, 0, 4, nodes::block(
            sum.assign(sum + value_noise(q) * amp),
            q.assign(q * 2.03f + 1.7f),
            amp.assign(amp * 0.5f)))),
        sum);
}

// ---- the materials

fn Material* burning() {
    Material* m = drie::mesh_standard_material(color: 0xffffff, roughness: 0.35, metalness: 0.7);
    threshold = nodes::uniform(0.5f);
    Node n = fbm(nodes::position_local() * 2.2f);
    // the burnt part goes (in the shadow maps too), the rest keeps a copper color
    m.color_node = nodes::seq(
        nodes::block(nodes::if_then(n.less_than(threshold), nodes::block(nodes::discard()))),
        nodes::color(0xc8743c));
    // just above the threshold: a glowing rim, white hot at the edge
    Node edge = nodes::one_minus(nodes::smoothstep(0, 0.06f, n - threshold));
    m.emissive_node = nodes::mix(nodes::color(0xff4000), nodes::color(0xffe0a0), edge * edge) * (edge * 6);
    m.side = DOUBLE_SIDE;
    return m;
}

fn Material* twisting() {
    Material* m = drie::mesh_standard_material(color: 0x5a8fd8, roughness: 0.4);
    Node p = nodes::position_local();
    Node t = nodes::time();
    // turned about y, more the higher, back and forth; swelled by a wave going up
    Node angle = p.y() * nodes::sin(t * 0.9f) * 1.4f;
    Node c = nodes::cos(angle);
    Node s = nodes::sin(angle);
    Node swell = 1 + nodes::sin(p.y() * 4 - t * 3) * 0.12f;
    Node xz = nodes::vec2(p.x() * c - p.z() * s, p.x() * s + p.z() * c) * swell;
    m.position_node = nodes::vec3(xz.x(), p.y(), xz.y());
    // stripes up the column from its uv, which stay on the vertices as they move: the twist
    Node ridge = nodes::smoothstep(0.3f, 0.7f, nodes::fract(nodes::uv().x() * 12));
    m.color_node = nodes::mix(nodes::color(0x5a8fd8), nodes::color(0xe8eef8), ridge);
    return m;
}

fn Node color_graph(int k) {
    Node u = nodes::uv();
    Node p = nodes::position_local();
    switch (k) {
        case 0:
            Node stripe = nodes::step(0.5f, nodes::fract((u.x() * 2 + u.y()) * 8 + nodes::time() * 0.5f));
            return nodes::mix(nodes::color(0x2050ff), nodes::color(0xffc020), stripe);
        case 1:
            Node cell = nodes::floor(u * nodes::vec2(16, 8));
            Node odd = nodes::mod(cell.x() + cell.y(), 2);
            return nodes::mix(nodes::color(0x202428), nodes::color(0xf0f0f0), odd);
        case 2:
            Node vein = nodes::sin(p.x() * 5 + fbm(p * 3) * 9) * 0.5f + 0.5f;
            return nodes::mix(nodes::color(0x303846), nodes::color(0xf4f0e8), nodes::pow(vein, 0.4f));
        default:
            Node facing = nodes::saturate(nodes::dot(nodes::normal_world(), nodes::view_direction()));
            return nodes::mix(nodes::color(0xff2080), nodes::color(0x20e0ff), facing);
    }
}

fn void use_graph(int k) {
    graph = k;
    ball.color_node = color_graph(k);
}

// ---- the program

fn void build_scene() {
    scene = drie::scene();
    scene.background = 0x1c2028;
    scene.add(drie::hemisphere_light(0xc8d8ff, 0x403830, 0.7));
    DirectionalLight* sun = drie::directional_light(0xfff0e0, 2.6);
    sun.position = { -4, 7, 5 };
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow_extent = 7;
    scene.add(sun);

    Mesh* floor = drie::mesh(drie::plane_geometry(40, 40), drie::mesh_standard_material(color: 0x8a8a8a, roughness: 0.9));
    floor.rotation.x = -(float)math::PI / 2;
    floor.receive_shadow = true;
    scene.add(floor);

    torus = drie::mesh(drie::torus_geometry(1.1f, 0.42f, 48, 160), burning());
    torus.position = { -3.6f, 1.6f, 0 };
    column = drie::mesh(drie::cylinder_geometry(0.55f, 0.55f, 3, 64, 96), twisting());
    column.position = { 0, 1.5f, 0 };
    ball = drie::mesh_standard_material(color: 0xffffff, roughness: 0.3);
    Mesh* sphere = drie::mesh(drie::sphere_geometry(1.2f, 96, 48), ball);
    sphere.position = { 3.6f, 1.25f, 0 };
    foreach (m : (Mesh*[3]) { torus, column, sphere }) {
        m.cast_shadow = true;
        m.receive_shadow = true;
        scene.add(m);
    }

    camera = drie::perspective_camera(45, (float)W / H, 0.1f, 100);
    camera.position = { 0, 4.2f, 10.5f };
    camera.look_at_vec({ 0, 1.2f, 0 });
}

// Before a frame at `t` seconds: the burn's threshold, and the ball's graph.
fn void update(double t) {
    // down to the bottom of the noise and up past its top: whole, gone, whole...
    threshold.set((float)(0.5 - 0.4 * math::cos(t * 0.5)));
    torus.rotation.y = (float)t * 0.3f;
    torus.rotation.x = 0.5f;
    if (pending < 0 && t - switched_at > 5) pending = (graph + 1) % 4;
    if (pending >= 0) {
        use_graph(pending);
        pending = -1;
        switched_at = t;
        build_ms = -1;
    }
}

fn void animate(double time) {
    double t = (double)renderer.time;
    update(t);
    controls.update();
    bool first = build_ms < 0;
    ulong start = stm::now();
    renderer.render(scene, camera);
    if (first) build_ms = stm::ms(stm::since(start));
}

fn void on_key(Event* e) {
    if (e.type != KEY_DOWN || e.key_repeat) return;
    if (e.key_code >= NUM_1 && e.key_code <= NUM_4) pending = (int)e.key_code - (int)KeyCode.NUM_1;
    if (e.key_code == SPACE) pending = (graph + 1) % 4;
}

fn void hud() {
    ui::setup();
    float scale = math::max(1.0f, math::round(sapp::heightf() / 540));
    sdtx::set_context(sdtx::default_context());
    sdtx::canvas(sapp::widthf() / scale, sapp::heightf() / scale);
    sdtx::origin(1, 1);
    sdtx::font(ui::FONT_KC853);
    sdtx::color3f(0.9f, 0.9f, 0.9f);
    @pool() {
        sdtx::puts(string::tformat("burn threshold %.2f: a uniform, set every frame, nothing rebuilt\n\n",
            threshold.value()[0]).zstr_tcopy());
        sdtx::puts(string::tformat("ball: graph %d of 4, %s (keys 1-4, space)\n", graph + 1, GRAPHS[graph]).zstr_tcopy());
        if (build_ms >= 0) sdtx::puts(string::tformat("  the frame that drew it first: %.1f ms\n", build_ms).zstr_tcopy());
    };
    sdtx::draw();
}

fn int main(String[] args) {
    String png_path;
    float at = 6;
    for (usz i = 1; i + 1 < args.len; i += 2) {
        switch (args[i]) {
            case "--png": png_path = args[i + 1];
            case "--time": at = (float)args[i + 1].to_double()!!;
            case "--graph": pending = args[i + 1].to_int()!!;
            default: io::eprintfn("unknown option %s", args[i]);
        }
    }
    if (!png_path) {
        renderer = drie::renderer(antialias: true, width: W, height: H, title: "drie: node materials");
        build_scene();
        controls = controls::orbit_controls(camera);
        controls.target = { 0, 1.2f, 0 };
        drie::add_event_listener(&on_key);
        drie::add_overlay(&hud);
        renderer.set_animation_loop(&animate);
        return 0;
    }
    renderer = drie::renderer(antialias: true, width: W, height: H);
    renderer.init_headless();
    renderer.auto_time = false;
    renderer.time = at;
    build_scene();
    switched_at = at;
    update(at);
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
    char[] px = mem::new_array(char, (usz)W * H * 4);
    readback::read_image(rt.texture_image, W, H, px.ptr);
    if (!png::write(png_path, px, W, H)) {
        io::eprintfn("could not write %s", png_path);
        return 1;
    }
    io::printfn("saved %s", png_path);
    return 0;
}
