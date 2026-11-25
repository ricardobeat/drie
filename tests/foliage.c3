// Foliage (Material.foliage) cut out by its map's alpha, double-sided, lit by a lamp with a
// size at the camera and a sun above, in each way an opaque surface is drawn: into the
// deferred G-buffer, forward after the depth prepass (an EQUAL depth test where the prepass
// drew), and forward without it. Columns: the plain material, then a node material whose
// position_node is house's plants' push (a long vertex stage that moves nothing here); rows:
// front faces, back faces (turned around by their instance's matrix). The three pictures,
// side by side, light each quarter alike.
//
//   just test foliage [out.png]   (tests/build/foliage.png)
module foliage_test;
import drie, drie::readback, drie::png, drie::nodes, sokol;
import std::io, std::math;

const int W = 320;
const int PASSES = 3;
const int H = 320;
const int N = 64;

// Leaves: a checker of opaque squares with holes, so that alpha_test cuts.
fn Texture* leaf_map() {
    char[] px = mem::new_array(char, N * N * 4);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            usz i = (usz)((y * N + x) * 4);
            px[i] = 120;
            px[i + 1] = 200;
            px[i + 2] = 80;
            px[i + 3] = ((x / 8 + y / 8) % 2 == 0 || (x / 4) % 4 == 0) ? 255 : 0;
        }
    }
    return drie::data_texture(px, N, N);
}

// house's plants giving way to whoever walks into them (`pusher`: xyz = their feet, w = reach;
// 0: nobody, nothing moves): the move worked out in the world and taken back into the model.
fn Node push_position(Node pusher) {
    Node model = nodes::model_matrix();
    Node inst = nodes::instance_matrix();
    Node local = nodes::position_local();
    Node pin = (model * (inst * nodes::vec4(nodes::vertex_color().xyz(), 1))).xyz();
    Node stem = (model * (inst * nodes::vec4(0, 0, 0, 1))).xz();
    Node away = pin.xz() - pusher.xz();
    Node d = nodes::length(away);
    Node dy = pin.y() - pusher.y();
    Node k = 1.0f - d / pusher.w();
    k = k * k * (3.0f - k * 2.0f) * nodes::smoothstep(-0.3f, 0.2f, dy) * (1.0f - nodes::smoothstep(1.7f, 2.3f, dy));
    k = k * nodes::smoothstep(0.12f, 0.8f, nodes::length(pin.xz() - stem));
    Node amount = k * pusher.w() * 0.5f;
    Node side = away / nodes::max(d, 0.15f) * amount;
    Node move = nodes::vec3(side.x(), amount * -0.3f, side.y());
    Node near = nodes::logical_and(nodes::greater_than(pusher.w(), 0), d.less_than(pusher.w()));
    move = nodes::select(near, move, nodes::vec3(0));
    Node a = (model * nodes::vec4(1, 0, 0, 0)).xyz();
    Node b = (model * nodes::vec4(0, 1, 0, 0)).xyz();
    Node c = (model * nodes::vec4(0, 0, 1, 0)).xyz();
    Node bc = nodes::cross(b, c);
    Node back = nodes::vec3(nodes::dot(bc, move), nodes::dot(nodes::cross(c, a), move), nodes::dot(nodes::cross(a, b), move));
    return local + back / nodes::dot(a, bc);
}

// A normal map that leaves the normals as they are.
fn Texture* flat_normals() {
    char[] px = mem::new_array(char, 4 * 4 * 4);
    for (usz i = 0; i < 16; i++) px[i * 4:4] = (char[4]){ 128, 128, 255, 255 }[..];
    return drie::data_texture(px, 4, 4, NO_COLOR_SPACE);
}

fn float mean(char[] px, int x0, int y0, int w, int h) {
    double s;
    for (int y = y0; y < y0 + h; y++) {
        for (int x = x0; x < x0 + w; x++) {
            usz i = (usz)((y * W + x) * 4);
            s += (double)px[i] + px[i + 1] + px[i + 2];
        }
    }
    return (float)(s / (double)(w * h * 3));
}

fn int main(String[] args) {
    String out_path = args.len > 1 ? args[1] : "tests/build/foliage.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    Scene* scene = drie::scene();
    Texture* leaves = leaf_map();
    MeshStandardMaterial*[2] mats;
    foreach (i, &m : mats) {
        *m = drie::mesh_standard_material(color: 0xffffff, roughness: 0.9f, side: DOUBLE_SIDE, vertex_colors: true);
        m.map = leaves;
        m.normal_map = flat_normals();
        m.alpha_test = 0.5f;
        m.foliage = true;
        m.sway = 0.0001f;
    }
    mats[1].position_node = push_position(nodes::uniform_vec4(0, 0, 0, 0));
    for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 2; col++) {
            BufferGeometry* g = drie::plane_geometry(1.6f, 1.6f);
            for (uint i = 0; i < g.vertex_count(); i++) g.set_vertex_color(i, { 0.8f, 0.8f, 0.8f });
            InstancedMesh* m = drie::instanced_mesh(g, mats[col], 1);
            // the back row turned around by its instance's matrix
            m.set_matrix_at(0, drie::mat4_compose({ col == 0 ? -0.9f : 0.9f, row == 0 ? 0.9f : -0.9f, 0 },
                                                  { 0, row == 1 ? (float)math::PI : 0, 0 }, { 1, 1, 1 }));
            scene.add(m);
        }
    }
    // (a lamp with a size, as one carried by the camera: wider than its distance from it)
    SpotLight* lamp = drie::spot_light(intensity: 40, angle: 1.15f, penumbra: 0.9f);
    lamp.radius = 0.35f;
    lamp.specular_radius = 0.012f;
    lamp.position.set(0, 0, 5.4f);
    lamp.target.position.set(0, 0, 0);
    scene.add(lamp);
    scene.add(lamp.target);
    DirectionalLight* sun = drie::directional_light(intensity: 1);
    sun.position.set(0, 1, -0.3f);
    scene.add(sun);
    PerspectiveCamera* camera = drie::perspective_camera(40, (float)W / H, 0.1f, 100);
    camera.position.set(0, 0, 5.4f);
    camera.look_at_vec({ 0, 0, 0 });
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    char[] all = mem::new_array(char, (usz)W * PASSES * H * 4);
    float[4][PASSES] at;
    for (int pass = 0; pass < PASSES; pass++) {
        renderer.deferred = pass == 0;
        renderer.depth_prepass = pass != 2;
        // (a few frames: node programs compile in the background)
        for (int f = 0; f < 3; f++) {
            renderer.render(scene, camera);
            drie::commit();
        }
        char[] px = mem::new_array(char, (usz)W * H * 4);
        readback::read_image(rt.texture_image, W, H, px.ptr);
        for (int y = 0; y < H; y++) {
            mem::copy(&all[(usz)(y * W * PASSES + pass * W) * 4], &px[(usz)(y * W) * 4], (usz)W * 4);
        }
        for (int q = 0; q < 4; q++) at[pass][q] = mean(px, (q % 2) * W / 2 + W / 8, (q / 2) * H / 2 + H / 8, W / 4, H / 4);
    }
    if (!png::write(out_path, all, W * PASSES, H)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    int fails;
    for (int q = 0; q < 4; q++) {
        io::printfn("%s %s: deferred %.1f, forward %.1f, without the prepass %.1f", q / 2 ? "back " : "front",
                    q % 2 ? "nodes" : "plain", at[0][q], at[1][q], at[2][q]);
        if (math::abs(at[0][q] - at[2][q]) > 3 || math::abs(at[1][q] - at[2][q]) > 3) {
            io::printfn("FAIL: the ways of drawing it differ");
            fails++;
        }
    }
    io::printfn("saved %s", out_path);
    return fails ? 1 : 0;
}
