// Progressive light maps (three's ProgressiveLightMap). A 4 x 4 floor and a unit box on it,
// packed into one 256 x 256 light map, lit by one shadowed directional light of intensity PI
// from (2, 4, 1.5) in the light map's scene; the main scene has no lights, and shows the
// floor and the box through BASIC materials with the light map (map / PI).
//   - the packing: the floor's uv1 is (uv + padding) / potpack's container, the box's square
//     beside it (its faces are then spread over a 3 x 2 grid of their own in it)
//   - blend window 1 (the last sample only): an open floor texel is Phong's white diffuse,
//     N.L * PI / PI (+ a faint highlight), the floor under the box's shadow black, the top of
//     the box as lit as the floor, its -x side (facing away) black
//   - 100 updates with a window of 50, the light jittered by up to 0.6 m: the shadow's edge
//     soft (between dark and lit), its middle still dark
//   - the seams' pass: just outside the floor's square the map has light; without it, none
//   - in the final picture (from straight above), a point in the shadow is dark and its
//     mirror images across x and z are lit: the map is not flipped
//   - a spot light with shadow.auto_update off in the main scene: its tile of the shadow atlas
//     (shared with the bake's light) is not drawn again after a bake; info is the main render's
// The same with Renderer.shadow_type = VSM. The picture: PCF, VSM, and show_debug's plane.
//
//   just test bake [out.png]   (tests/build/bake.png)
module bake_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int W        = 256;
const int H        = 256;
const int RES      = 256;
const float FOV    = 50;
const float HEIGHT = 7;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn float to_linear(char c) {
    float v = (float)c / 255;
    return v <= 0.04045f ? v / 12.92f : math::pow((v + 0.055f) / 1.055f, 2.4f);
}

struct Bake {
    ProgressiveLightMap* plm;
    Scene* scene;
    Mesh* floor;
    Mesh* box;
    DirectionalLight* light;
    float[] map;
}

// The map the last update drew, read back as floats (rgba per texel, row 0 first).
fn void read_map(Bake* b) {
    drie::commit();
    ProgressiveLightMap* plm = b.plm;
    RenderTarget* rt         = plm.targets[plm.first_active ? 1 : 0];
    bool full                = rt.pixel_format() == RGBA32F;
    usz n                    = (usz)RES * RES * 4;
    if (!b.map.len) b.map = mem::new_array(float, (sz)n);
    if (full) {
        readback::read_image(rt.texture_image, RES, RES, (char*)b.map.ptr, 16);
    } else {
        ushort[] h = mem::new_array(ushort, (sz)n);
        defer free(h);
        readback::read_image(rt.texture_image, RES, RES, (char*)h.ptr, 8);
        for (usz i = 0; i < n; i++) b.map[i] = drie::half_to_float(h[i]);
    }
}

fn float texel(Bake* b, Vector2 uv1) {
    int x = math::clamp((int)(uv1.x * RES), 0, RES - 1);
    int y = math::clamp((int)(uv1.y * RES), 0, RES - 1);
    return b.map[((usz)y * RES + (usz)x) * 4];
}

// The floor's uv1 at world point (x, 0, z): along the plane's own axes (local x = x, y = -z).
fn Vector2 floor_uv1(Bake* b, float x, float z) {
    BufferGeometry* g = b.floor.geometry;
    Vector3 p0        = g.get_position(0);
    Vector3 p3        = g.get_position(3);
    Vector2 a         = { g.uv1[0], g.uv1[1] };
    Vector2 c         = { g.uv1[6], g.uv1[7] };
    float tx          = (x - p0.x) / (p3.x - p0.x);
    float ty          = (-z - p0.y) / (p3.y - p0.y);
    return { a.x + (c.x - a.x) * tx, a.y + (c.y - a.y) * ty };
}

// The middle of the box's face `f` (+x -x +y -y +z -z) in the light map.
fn Vector2 face_uv1(Bake* b, int f) {
    BufferGeometry* g = b.box.geometry;
    Vector2 sum;
    for (int k = 0; k < 4; k++) {
        sz v = (sz)(f * 4 + k) * 2;
        sum += (Vector2) { g.uv1[v], g.uv1[v + 1] };
    }
    return sum / 4;
}

fn Bake make_bake(Renderer* renderer) {
    Bake b;
    b.scene = drie::scene();
    b.floor = drie::mesh(drie::plane_geometry(4, 4), drie::mesh_basic_material(color: 0xffffff));
    b.floor.rotation.x = -(float)math::PI / 2;
    b.scene.add(b.floor);
    b.box = drie::mesh(drie::box_geometry(1, 1, 1), drie::mesh_basic_material(color: 0xffffff));
    b.box.position = { 0, 0.5f, 0 };
    b.scene.add(b.box);
    b.light = drie::directional_light(0xffffff, (float)math::PI);
    b.light.position = { 2, 4, 1.5f };
    b.light.cast_shadow = true;
    b.light.shadow.map_size = 1024;
    b.light.shadow_extent = 4;
    b.plm = drie::progressive_lightmap(renderer, RES);
    Object3D*[3] objects = { b.light, b.floor, b.box };
    b.plm.add_objects(&objects);
    // the box's faces share its square: spread them over a 3 x 2 grid in it
    BufferGeometry* g = b.box.geometry;
    Vector2 lo        = { 1, 1 };
    Vector2 hi        = { 0, 0 };
    for (uint i = 0; i < g.vertex_count(); i++) {
        Vector2 u = { g.uv1[(sz)i * 2], g.uv1[(sz)i * 2 + 1] };
        lo = math::min(lo, u);
        hi = math::max(hi, u);
    }
    for (uint i = 0; i < g.vertex_count(); i++) {
        int f      = (int)i / 4;
        Vector2 uv = g.get_uv(i);
        g.uv1[(sz)i * 2] = lo.x + (hi.x - lo.x) * ((float)(f % 3) + 0.1f + 0.8f * uv.x) / 3;
        g.uv1[(sz)i * 2 + 1] = lo.y + (hi.y - lo.y) * ((float)(f / 3) + 0.1f + 0.8f * uv.y) / 2;
    }
    g.uploaded = false;
    b.scene.update_matrix_world();
    return b;
}

// Phong's light on the floor or the box's top: N.L * PI * white / PI, plus the highlight's
// (specular 0x111111, shininess 30) seen from `eye`.
fn float phong_up(Vector3 at, Vector3 eye) {
    Vector3 l = ((Vector3) { 2, 4, 1.5f }).normalize();
    Vector3 v = (eye - at).normalize();
    Vector3 h = (l + v).normalize();
    float nol = l.y;
    float f0  = math::pow(17.0f / 255, 2.2f);
    float fr  = f0 + (1 - f0) * math::pow(1 - math::clamp(v.dot(h), 0.0f, 1.0f), 5.0f);
    float d   = (1 / (float)math::PI) * 16 * math::pow(h.y, 30.0f);
    return nol + nol * (float)math::PI * fr * 0.25f * d;
}

// The pixel that shows floor point (x, 0, z) from the camera straight above.
fn int[2] pixel_of(float x, float z) {
    float k = (float)H / 2 / (HEIGHT * math::tan(FOV * 0.5f * (float)math::PI / 180));
    return { (int)(W / 2 + x * k), (int)(H / 2 + z * k) };
}

fn float shown(char[] px, float x, float z) {
    int[2] p = pixel_of(x, z);
    usz i    = ((usz)p[1] * W + (usz)p[0]) * 4;
    return to_linear(px[i]);
}

fn void check(Renderer* renderer, String name, char[] picture, char[] debug) {
    io::printfn("-- %s", name);
    Bake b                    = make_bake(renderer);
    BufferGeometry* fg        = b.floor.geometry;
    PerspectiveCamera* camera = drie::perspective_camera(FOV, (float)W / H, 0.1f, 50);
    camera.position = { 0, HEIGHT, 0 };
    camera.rotation.x = -(float)math::PI / 2;
    camera.update_matrix_world();
    Vector3 eye = camera.position;

    // the packing: two squares of 1 + 6 / 256, potpack puts the second over the first (its
    // container is 2 wide at first: the second doesn't fit beside the first)
    float pad   = 3.0f / RES;
    float side  = 1 + 2 * pad;
    bool packed = true;
    for (uint i = 0; i < fg.vertex_count(); i++) {
        Vector2 uv = fg.get_uv(i);
        if (math::abs(fg.uv1[(sz)i * 2] - (uv.x + pad) / side) > 1e-5f) packed = false;
        if (math::abs(fg.uv1[(sz)i * 2 + 1] - (uv.y + pad) / (2 * side)) > 1e-5f) packed = false;
    }
    expect(packed, "the floor's uv1 is potpack's: (uv + padding) / (1, 2) squares");
    expect(face_uv1(&b, 2).y > 0.5f, "the box's square is above the floor's");

    // one sample
    b.plm.update(camera, blend_window: 1);
    read_map(&b);
    float lit    = phong_up({ 1.5f, 0, 1.5f }, eye);
    float open   = texel(&b, floor_uv1(&b, 1.5f, 1.5f));
    float shade  = texel(&b, floor_uv1(&b, -0.85f, -0.6f));
    float top    = texel(&b, face_uv1(&b, 2));
    float away   = texel(&b, face_uv1(&b, 1));
    float facing = texel(&b, face_uv1(&b, 0));
    io::printfn(
        "   open %.4f (Phong %.4f), shadow %.4f, top %.4f, -x %.4f, +x %.4f",
        open,
        lit,
        shade,
        top,
        away,
        facing
    );
    expect(math::abs(open - lit) < 0.02f * lit, "open floor: N.L, as Phong");
    expect(shade < 0.05f * lit, "under the box's shadow: dark");
    expect(top > 0.9f * lit && top < 1.1f * phong_up({ 0, 1, 0 }, eye), "the box's top: lit as the floor");
    expect(away < 0.02f, "the box's -x side: dark");
    float side_lit = 2 / ((Vector3) { 2, 4, 1.5f }).length();
    expect(math::abs(facing - side_lit) < 0.05f * side_lit, "the box's +x side: its N.L");

    // many samples, the light jittered: a soft edge
    uint seed = 12345;
    for (int i = 0; i < 100; i++) {
        seed = seed * 1664525 + 1013904223;
        float jx = (float)((seed >> 8) & 0xffff) / 65535 - 0.5f;
        seed = seed * 1664525 + 1013904223;
        float jz = (float)((seed >> 8) & 0xffff) / 65535 - 0.5f;
        b.light.position = { 2 + jx * 1.2f, 4, 1.5f + jz * 1.2f };
        b.plm.update(camera, blend_window: 50);
    }
    read_map(&b);
    float middle = texel(&b, floor_uv1(&b, -0.75f, -0.3f));
    float edge   = texel(&b, floor_uv1(&b, -1.0f, -0.3f));
    float open2  = texel(&b, floor_uv1(&b, 1.5f, 1.5f));
    io::printfn("   soft: middle %.4f, edge %.4f, open %.4f", middle, edge, open2);
    expect(middle < 0.15f * lit, "soft: the middle of the shadow stays dark");
    expect(edge > 0.15f * lit && edge < 0.85f * lit, "soft: its edge is in between");
    expect(math::abs(open2 - lit) < 0.05f * lit, "soft: the open floor still N.L");
    // the seams' pass: a texel and a half past the floor's +u edge (open floor there) the
    // light has spread
    Vector2 outside_uv = { (1 + 1.5f * pad) / side, 0.4f / side };
    Vector2 inside_uv  = { (1 + 0.5f * pad) / side, 0.4f / side };
    float outside      = texel(&b, outside_uv);
    io::printfn("   seams: inside %.4f, just outside %.4f", texel(&b, inside_uv), outside);
    expect(outside > 0.5f * lit, "seams: the light spreads past the floor's edge");

    // the final picture: the objects show texture(1); shown = map / PI
    renderer.set_render_target(drie::render_target(W, H, 1, RGBA8));
    renderer.render(b.scene, camera);
    drie::commit();
    readback::read_image(renderer.render_target.texture_image, W, H, picture.ptr);
    // (texture(1) is from update 100 or 99: take the map it holds)
    float in_shadow = shown(picture, -0.9f, -0.7f);
    float mirror_z  = shown(picture, -0.9f, 0.7f);
    float mirror_x  = shown(picture, 0.9f, -0.7f);
    float far       = shown(picture, 1.5f, 1.5f);
    io::printfn(
        "   picture: shadow %.4f, mirrored in z %.4f, in x %.4f, open %.4f (map / PI %.4f)",
        in_shadow,
        mirror_z,
        mirror_x,
        far,
        lit / (float)math::PI
    );
    expect(in_shadow < 0.3f * far, "picture: the shadow where it is");
    expect(mirror_z > 0.8f * far && mirror_x > 0.8f * far, "picture: not mirrored (in x or z)");
    expect(math::abs(far - lit / (float)math::PI) < 0.03f, "picture: the open floor shows map / PI");

    // without the seams' pass, nothing past the floor's edge
    b.plm.update(camera, blend_window: 1, blur_edges: false);
    read_map(&b);
    float bare = texel(&b, outside_uv);
    io::printfn("   no seams: inside %.4f, just outside %.4f", texel(&b, inside_uv), bare);
    expect(bare == 0, "no seams' pass: nothing outside the floor's square");

    // the shadow atlas is shared: the main render's drawn-once tile (a spot light with
    // shadow.auto_update off) stays as it is through a bake, and a bake leaves info alone
    SpotLight* spot = drie::spot_light(0xffffff, 10);
    spot.position = { 0, 5, 0 };
    spot.cast_shadow = true;
    spot.shadow.auto_update = false;
    b.scene.add(spot);
    renderer.render(b.scene, camera);
    drie::commit();
    int drawn       = renderer.info.shadow_calls;
    RenderInfo info = renderer.info;
    b.plm.update(camera, blend_window: 1);
    expect(
        renderer.info.calls == info.calls && renderer.info.shadow_calls == info.shadow_calls
            && renderer.info.triangles == info.triangles,
        "a bake leaves info as it was"
    );
    renderer.render(b.scene, camera);
    drie::commit();
    io::printfn("   main render's shadow draws: %d, after a bake %d", drawn, renderer.info.shadow_calls);
    expect(drawn > 0 && renderer.info.shadow_calls == 0, "the main render's drawn-once tile is kept through a bake");
    spot.visible = false;

    // show_debug's plane, the map: in front of an orthographic camera
    b.plm.update(camera, blend_window: 1);
    b.plm.show_debug(true, { 0, 50, 0 });
    OrthographicCamera* ortho = drie::orthographic_camera(-50, 50, 50, -50, 1, 200);
    ortho.position = { 0, 50, 100 };
    renderer.render(b.scene, ortho);
    drie::commit();
    readback::read_image(renderer.render_target.texture_image, W, H, debug.ptr);
}

fn int main(String[] args) {
    String out_path    = args.len > 1 ? args[1] : "tests/build/bake.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.tone_mapping = NO_TONE_MAPPING;
    char[] pcf       = mem::new_array(char, (usz)W * H * 4);
    char[] vsm       = mem::new_array(char, (usz)W * H * 4);
    char[] debug     = mem::new_array(char, (usz)W * H * 4);
    char[] debug_vsm = mem::new_array(char, (usz)W * H * 4);
    check(renderer, "PCF", pcf, debug);
    renderer.shadow_type = VSM;
    check(renderer, "VSM", vsm, debug_vsm);

    int ow     = W * 3;
    char[] rgb = mem::new_array(char, (usz)ow * H * 3);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            usz s = ((usz)y * W + (usz)x) * 4;
            for (int k = 0; k < 3; k++) {
                usz d      = ((usz)y * (usz)ow + (usz)(x + k * W)) * 3;
                char[] src = k == 0 ? pcf : (k == 1 ? vsm : debug);
                rgb[d:3] = src[s:3];
            }
        }
    }
    if (!png::write(out_path, rgb, ow, H, channels: 3)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s; %d failed", out_path, failures);
    return failures ? 1 : 0;
}
