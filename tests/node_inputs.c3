// What a node material's graphs read besides their own nodes (drie::nodes): the material's own
// values at the step a graph changes (material_color, material_roughness, material_metalness,
// env_color), the instance's matrix, explicit-gradient sampling, the material's own maps
// (material_map), clearcoat_node, the screen's depth behind a surface (ssr_on, screen_position,
// viewport_depth), and one uniform shared by several materials. Most checks draw a sphere
// with a graph and then with the material settings it should equal, and compare them.
//
//   just test node_inputs [out.png]   (tests/build/node_inputs.png: the last picture)
module node_inputs_test;
import drie, drie::nodes, drie::readback, drie::png, sokol;
import std::io, std::math;

const int W = 640;
const int H = 320;

int failures;
Renderer* renderer;
RenderTarget* rt;
PerspectiveCamera* camera;
char[] px;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn Texture* checker(char r, char g, char b) {
    int n = 64;
    char[] data = mem::new_array(char, (usz)(n * n * 4));
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
            bool on = ((x / 8 + y / 8) % 2) == 0;
            usz i = (usz)((y * n + x) * 4);
            data[i:4] = (char[4]) { on ? r : 40, on ? g : 40, on ? b : 40, 255 }[..];
        }
    }
    Texture* t = drie::data_texture(data, n, n);
    t.wrap_s = REPEAT_WRAPPING;
    t.wrap_t = REPEAT_WRAPPING;
    return t;
}

fn Texture* flat(char r, char g, char b, char a = 255, ColorSpace space = SRGB_COLOR_SPACE) {
    char[] data = mem::new_array(char, 4 * 4 * 4);
    for (int i = 0; i < 16; i++) data[(usz)i * 4:4] = (char[4]) { r, g, b, a }[..];
    return drie::data_texture(data, 4, 4, space);
}

// The pixel where `p` (world) lands.
fn int[2] pixel_of(Vector3 p) {
    drie::Mat4 vp = camera.projection_matrix.mul(camera.matrix_world_inverse);
    float[4] c;
    for (int r = 0; r < 4; r++) c[r] = vp[0][r] * p.x + vp[1][r] * p.y + vp[2][r] * p.z + vp[3][r];
    return { (int)((c[0] / c[3] * 0.5f + 0.5f) * W), (int)((0.5f - c[1] / c[3] * 0.5f) * H) };
}

fn float[3] rgb_at(int[2] p) {
    float[3] sum;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            usz i = ((usz)(p[1] + dy) * W + (usz)(p[0] + dx)) * 4;
            for (int k = 0; k < 3; k++) sum[k] += (float)px[i + (usz)k] / 9;
        }
    }
    return sum;
}

fn void draw(Scene* scene) {
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, W, H, px.ptr);
}

// A scene lit by a sun and a dim sky, with an environment map for reflections.
fn Scene* lit_scene() {
    Scene* s = drie::scene();
    s.background = 0x202020;
    s.add(drie::hemisphere_light(0x8090b0, 0x202028, 0.4));
    DirectionalLight* sun = drie::directional_light(0xffffff, 2.5);
    sun.position = { 2, 3, 4 };
    s.add(sun);
    int w = 8;
    int h = 64;
    char[] data = mem::new_array(char, (usz)(w * h * 4));
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            usz i = (usz)((y * w + x) * 4);
            data[i] = (char)(60 + 150 * y / (h - 1));
            data[i + 1] = (char)(255 * y / (h - 1));
            data[i + 2] = (char)(255 - 200 * y / (h - 1));
            data[i + 3] = 255;
        }
    }
    s.environment = drie::data_texture(data, w, h);
    return s;
}

// The colors over a grid of points on the front of a sphere at (x, 0, 0).
fn float[3][25] sphere_grid(float at) {
    float[3][25] r;
    int k;
    for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
            float x = (float)i * 0.25f;
            float y = (float)j * 0.25f;
            r[k++] = rgb_at(pixel_of({ at + x, y, math::sqrt(0.9f * 0.9f - x * x - y * y) }));
        }
    }
    return r;
}

// Draws `a` and then `b` as a sphere in the same place in `scene`, and returns the largest
// difference in a channel between the two pictures over the sphere's front. Then draws them
// side by side (a at x = -1.2, b at 1.2), for the pixels a check reads after.
fn float compare(Scene* scene, Material* a, Material* b) {
    BufferGeometry* g = drie::sphere_geometry(0.9f, 64, 32);
    Mesh* m = drie::mesh(g, a);
    scene.add(m);
    draw(scene);
    float[3][25] ga = sphere_grid(0);
    m.material = b;
    draw(scene);
    float[3][25] gb = sphere_grid(0);
    scene.remove(m);
    float worst;
    for (int k = 0; k < 25; k++) {
        for (int c = 0; c < 3; c++) worst = math::max(worst, math::abs(ga[k][c] - gb[k][c]));
    }
    Mesh* left = drie::mesh(g, a);
    left.position = { -1.2f, 0, 0 };
    Mesh* right = drie::mesh(g, b);
    right.position = { 1.2f, 0, 0 };
    scene.add(left);
    scene.add(right);
    draw(scene);
    scene.remove(left);
    scene.remove(right);
    return worst;
}

fn void check_pair(Scene* scene, Material* a, Material* b, String what, float tolerance = 3) {
    float d = compare(scene, a, b);
    expect(d <= tolerance, string::tformat("%s (%.1f)", what, d));
}

fn float srgb_byte(float linear) {
    float c = linear <= 0.0031308f ? linear * 12.92f : 1.055f * math::pow(linear, 1.0f / 2.4f) - 0.055f;
    return c * 255;
}

fn int main(String[] args) {
    String out_path = args.len > 1 ? args[1] : "tests/build/node_inputs.png";
    renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.auto_time = false;
    camera = drie::perspective_camera(40, (float)W / H, 0.1f, 100);
    camera.position = { 0, 0, 7 };
    camera.look_at_vec({ 0, 0, 0 });
    rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    px = mem::new_array(char, (usz)W * H * 4);
    Scene* unlit = drie::scene();
    unlit.background = 0x202020;
    Scene* lit = lit_scene();

    // material_color: the color after the map and color, halved, against the same worked out
    // by hand; opacity_node from material_color's alpha (0.5 doubled: opaque)
    {
        Texture* map = checker(255, 200, 80);
        Material* a = drie::mesh_basic_material(color: 0x80c0ff, map: map);
        a.color_node = nodes::material_color().xyz() * 0.5f;
        a.opacity = 0.5f;
        a.transparent = true;
        a.opacity_node = nodes::material_color().w() * 2;
        Material* b = drie::mesh_basic_material(color: 0x80c0ff, map: map);
        b.color_node = nodes::texture(map).xyz() * nodes::color(0x80c0ff) * 0.5f;
        check_pair(unlit, a, b, "material_color: the material's color and opacity at color_node's step");
    }
    // material_roughness and material_metalness: the material's own scaled, against the
    // material with the scaled values; and as they are, against the material without nodes
    {
        Material* a = drie::mesh_standard_material(color: 0xc08040, roughness: 1, metalness: 1);
        a.roughness_node = nodes::material_roughness() * 0.3f;
        a.metalness_node = nodes::material_metalness() * 0.5f;
        Material* b = drie::mesh_standard_material(color: 0xc08040, roughness: 0.3f, metalness: 0.5f);
        check_pair(lit, a, b, "material_roughness, material_metalness: the material's values at the surface step");
        Texture* arm = flat(255, 100, 200, space: NO_COLOR_SPACE);
        Material* c = drie::mesh_standard_material(color: 0xc08040, roughness: 0.8f, metalness: 1);
        c.arm_map = arm;
        c.roughness_node = nodes::material_roughness();
        c.metalness_node = nodes::material_metalness();
        Material* d = drie::mesh_standard_material(color: 0xc08040, roughness: 0.8f, metalness: 1);
        d.arm_map = arm;
        check_pair(lit, c, d, "material_roughness, material_metalness: after the ARM map");
    }
    // env_color: what env_node replaces, as it is, against no env_node; and darkened
    {
        Material* a = drie::mesh_standard_material(color: 0xffffff, roughness: 0.1f, metalness: 1);
        a.env_node = nodes::env_color();
        Material* b = drie::mesh_standard_material(color: 0xffffff, roughness: 0.1f, metalness: 1);
        check_pair(lit, a, b, "env_color: the environment's own reflection");
        Material* c = drie::mesh_standard_material(color: 0xffffff, roughness: 0.1f, metalness: 1);
        c.env_node = nodes::env_color() * 0.25f;
        float dark = compare(lit, c, b);
        float[3] l = rgb_at(pixel_of({ -1.2f, 0.3f, 0.85f }));
        float[3] r = rgb_at(pixel_of({ 1.2f, 0.3f, 0.85f }));
        expect(dark > 20 && l[1] < r[1], string::tformat("env_color: a quarter of it is darker (%s against %s)", l, r));
    }
    // clearcoat_node, clearcoat_roughness_node: against clearcoat and clearcoat_roughness
    {
        Material* a = drie::mesh_standard_material(color: 0x802020, roughness: 0.7f);
        a.clearcoat_node = nodes::num(1);
        a.clearcoat_roughness_node = nodes::num(0.1f);
        Material* b = drie::mesh_standard_material(color: 0x802020, roughness: 0.7f);
        b.clearcoat = 1;
        b.clearcoat_roughness = 0.1f;
        check_pair(lit, a, b, "clearcoat_node, clearcoat_roughness_node: as clearcoat 1, roughness 0.1");
        Material* c = drie::mesh_standard_material(color: 0x802020, roughness: 0.7f);
        float none = compare(lit, a, c);
        expect(none > 10, string::tformat("clearcoat_node: the coat is on (%.1f from none)", none));
    }
    // texture_grad: with the uv's own gradients, as texture()
    {
        Texture* map = checker(255, 255, 255);
        Node at = nodes::uv() * nodes::vec2(3, 2);
        Material* a = drie::mesh_basic_material(color: 0xffffff);
        a.color_node = nodes::texture_grad(map, at, nodes::dfdx(at), nodes::dfdy(at)).xyz();
        Material* b = drie::mesh_basic_material(color: 0xffffff);
        b.color_node = nodes::texture(map, at).xyz();
        check_pair(unlit, a, b, "texture_grad: with the uv's gradients, as texture()");
        Material* c = drie::mesh_basic_material(color: 0xffffff);
        c.color_node = nodes::texture_grad(map, at, nodes::dfdx(at) * 64, nodes::dfdy(at) * 64).xyz();
        float blur = compare(unlit, c, b);
        expect(blur > 30, string::tformat("texture_grad: wide gradients read a small level (%.1f)", blur));
    }
    // material_map: one graph in two materials reads each one's custom_map (red, blue), with
    // texture, texture_grad and texture_size; and the normal map
    {
        Texture* custom = nodes::material_map(CUSTOM_MAP);
        Node at = nodes::uv() * (nodes::texture_size(custom).x() / 4);
        Node shared = nodes::texture(custom, at).xyz() * 0.5f + nodes::texture_grad(custom, at, nodes::dfdx(at), nodes::dfdy(at)).xyz() * 0.5f;
        Material* a = drie::mesh_basic_material(color: 0xffffff);
        a.custom_map = flat(255, 0, 0);
        a.color_node = shared;
        Material* b = drie::mesh_basic_material(color: 0xffffff);
        b.custom_map = flat(0, 0, 255);
        b.color_node = shared;
        compare(unlit, a, b);
        float[3] l = rgb_at(pixel_of({ -1.2f, 0, 0.9f }));
        float[3] r = rgb_at(pixel_of({ 1.2f, 0, 0.9f }));
        io::printfn("material_map: %s, %s", l, r);
        expect(l[0] > 200 && l[2] < 30 && r[2] > 200 && r[0] < 30, "material_map: each material's own custom_map");
        Material* c = drie::mesh_basic_material(color: 0xffffff);
        c.normal_map = flat(128, 128, 255, space: NO_COLOR_SPACE);
        c.color_node = nodes::texture(nodes::material_map(NORMAL_MAP)).xyz();
        Material* d = drie::mesh_basic_material(color: 0xffffff);
        d.color_node = nodes::vec3(128.0f / 255, 128.0f / 255, 1);
        check_pair(unlit, c, d, "material_map: the material's normal map");
    }
    // instance_matrix: position_node adds the instance's translation once more (an instance
    // at x = 0.6 drawn at 1.2); identity for a plain mesh; vertex_color() in position_node
    {
        Material* m = drie::mesh_basic_material(color: 0xffffff);
        m.position_node = nodes::position_local() + (nodes::instance_matrix() * nodes::vec4(0, 0, 0, 1)).xyz();
        m.color_node = nodes::color(0x40ff40);
        BufferGeometry* g = drie::sphere_geometry(0.5f, 32, 16);
        InstancedMesh* inst = drie::instanced_mesh(g, m, 1);
        drie::Mat4 t = drie::mat4_identity();
        t[3][0] = 0.6f;
        t[3][1] = 0.4f;
        inst.set_matrix_at(0, t);
        Material* p = drie::mesh_basic_material(color: 0xffffff);
        p.position_node = nodes::position_local() + (nodes::instance_matrix() * nodes::vec4(0, 0, 0, 1)).xyz()
            + nodes::vec3(0, nodes::vertex_color().x() * -1.2f, 0);
        p.color_node = nodes::color(0xff4040);
        Mesh* plain = drie::mesh(g, p);
        plain.position = { -1.5f, 0.4f, 0 };
        Scene* s = drie::scene();
        s.background = 0x202020;
        s.add(inst);
        s.add(plain);
        draw(s);
        float[3] moved = rgb_at(pixel_of({ 1.2f, 0.8f, 0.5f }));
        float[3] at_own = rgb_at(pixel_of({ 0.6f, 0.4f, 0.5f }));
        float[3] down = rgb_at(pixel_of({ -1.5f, -0.8f, 0.5f }));
        io::printfn("instance_matrix: %s at (1.2, 0.8), %s at (0.6, 0.4); plain mesh: %s", moved, at_own, down);
        expect(moved[1] > 150 && at_own[1] < 60, "instance_matrix: the instance's translation (rows as instance_point)");
        expect(down[0] > 150, "instance_matrix: the identity on a plain mesh; vertex_color() in position_node");
    }
    // the screen's depth: a source whose depth (green) is 200/255 m everywhere; output_node
    // shows viewport_depth at the fragment's screen position, screen_position's z / 10 and its
    // x, and ssr_on (green) or not (red)
    {
        Texture* depth = flat(0, 200, 0, space: NO_COLOR_SPACE);
        depth.upload();
        drie::Mat4 vp = camera.projection_matrix.mul(camera.matrix_world_inverse);
        Node s = nodes::screen_position(nodes::position_world());
        Material* a = drie::mesh_basic_material(color: 0xffffff);
        a.ssr = 1;
        a.output_node = nodes::vec4(nodes::viewport_depth(s.xy()), s.z() * 0.1f, s.x(), 1);
        Material* b = drie::mesh_basic_material(color: 0xffffff);
        b.ssr = 1;
        b.output_node = nodes::vec4(nodes::select(nodes::ssr_on(), nodes::vec3(0, 1, 0), nodes::vec3(1, 0, 0)), 1);
        renderer.set_reflection_source(depth.view, depth.view, vp);
        compare(unlit, a, b);
        int[2] front = pixel_of({ -1.2f, 0, 0.9f });
        float[3] l = rgb_at(front);
        float[3] r = rgb_at(pixel_of({ 1.2f, 0, 0.9f }));
        float want_z = srgb_byte((7 - 0.9f) * 0.1f);
        float want_x = srgb_byte((float)front[0] / W);
        io::printfn("screen depth: %s (want %.1f, %.1f, %.1f); ssr_on: %s", l, srgb_byte(200.0f / 255), want_z, want_x, r);
        expect(math::abs(l[0] - srgb_byte(200.0f / 255)) < 3, "viewport_depth: the source's depth (its green)");
        expect(math::abs(l[1] - want_z) < 4, "screen_position: z, the distance from the camera's plane");
        expect(math::abs(l[2] - want_x) < 4, "screen_position: xy, the uv on the screen");
        expect(r[1] > 200 && r[0] < 30, "ssr_on: on with a source and Material.ssr");
        renderer.clear_reflection_source();
        compare(unlit, a, b);
        r = rgb_at(pixel_of({ 1.2f, 0, 0.9f }));
        expect(r[0] > 200 && r[1] < 30, "ssr_on: off without a source");
    }
    // one uniform in three materials' different graphs: each draw reads its value as set
    {
        Node shared = nodes::uniform_vec4(1, 0, 0, 1);
        Material*[3] mats;
        for (int i = 0; i < 3; i++) mats[i] = drie::mesh_basic_material(color: 0xffffff);
        mats[0].color_node = shared.xyz();
        mats[1].color_node = shared.xyz() * shared.w();
        mats[2].color_node = nodes::vec3(shared.x(), shared.y(), shared.z()) + nodes::uniform(0);
        Scene* s = drie::scene();
        s.background = 0x202020;
        BufferGeometry* g = drie::sphere_geometry(0.6f, 32, 16);
        for (int i = 0; i < 3; i++) {
            Mesh* m = drie::mesh(g, mats[i]);
            m.position = { ((float)i - 1) * 1.6f, 0, 0 };
            s.add(m);
        }
        bool ok = true;
        Vector3[2] values = { { 1, 0, 0 }, { 0, 0, 1 } };
        foreach (k, v : values) {
            shared.set(v.x, v.y, v.z, 1);
            draw(s);
            for (int i = 0; i < 3; i++) {
                float[3] c = rgb_at(pixel_of({ ((float)i - 1) * 1.6f, 0, 0.6f }));
                bool red = c[0] > 200 && c[2] < 30;
                bool blue = c[2] > 200 && c[0] < 30;
                if (k == 0 ? !red : !blue) {
                    io::printfn("shared uniform, value %d, sphere %d: %s", k, i, c);
                    ok = false;
                }
            }
        }
        expect(ok, "a uniform shared by three materials: each draw reads its value");
    }
    if (!png::write(out_path, px, W, H)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s", out_path);
    return failures ? 1 : 0;
}
