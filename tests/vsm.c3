// Variance shadow maps (Renderer.shadow_type = VSM), on three's webgl_shadowmap_vsm scene: a
// torus (three's has a torus knot, which drie lacks) over a floor among four pillars, a red
// spot light and a blue directional light, both shadowed with radius 4. Rendered with PCF
// (left) and VSM (right): lit places alike, shadowed ones dark in both, VSM's edges wider,
// what is outside of the spot light's map lit; and again with blur_samples = 1. The floor casts
// shadows in the VSM renders only (drie's PCF has no depth bias for it).
//
//   just test vsm [out.png]   (tests/build/vsm.png)
module vsm_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int W = 480;
const int H = 360;

int failures;
drie::Mat4 view_proj;

// The pixel that shows world point `p`.
fn int[2] pixel_of(Vector3 p) {
    drie::Mat4 m = view_proj;
    float x      = m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0];
    float y      = m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1];
    float w      = m[0][3] * p.x + m[1][3] * p.y + m[2][3] * p.z + m[3][3];
    return { (int)((x / w * 0.5f + 0.5f) * W), (int)((0.5f - y / w * 0.5f) * H) };
}

fn float lum_at(char[] px, int x, int y) {
    usz i = ((usz)y * W + (usz)x) * 4;
    return ((float)px[i] + (float)px[i + 1] + (float)px[i + 2]) / 3;
}

// A 3x3 average around the floor's point (x, 0, z).
fn float lum(char[] px, float x, float z) {
    int[2] p = pixel_of({ x, 0, z });
    float sum;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) sum += lum_at(px, p[0] + dx, p[1] + dy);
    }
    return sum / 9;
}

// How much of the floor from `a` to `b` (200 steps) is between its darkest and its lightest
// (10% off either): how wide the edges of the shadows on the way are.
fn int edge_width(char[] px, Vector3 a, Vector3 b) {
    float[201] l;
    float lo = 255;
    float hi = 0;
    foreach (i, &v : l) {
        float t = (float)i / 200;
        *v = lum(px, a.x + (b.x - a.x) * t, a.z + (b.z - a.z) * t);
        lo = math::min(lo, *v);
        hi = math::max(hi, *v);
    }
    int n;
    foreach (v : l) {
        if (v > lo + (hi - lo) * 0.1f && v < hi - (hi - lo) * 0.1f) n++;
    }
    return n;
}

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn void render_into(Renderer* renderer, Scene* scene, Camera* camera, RenderTarget* rt, char[] px) {
    // (twice: the second frame is what is kept, as a demo shows it)
    for (int i = 0; i < 2; i++) {
        renderer.render(scene, camera);
        drie::commit();
    }
    readback::read_image(rt.texture_image, W, H, px.ptr);
}

fn int main(String[] args) {
    String out_path    = args.len > 1 ? args[1] : "tests/build/vsm.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    Scene* scene = drie::scene();
    scene.background = 0x222244;
    scene.fog = drie::fog(0x222244, 50, 100);
    scene.add(drie::ambient_light(0x444444));

    SpotLight* spot = drie::spot_light(0xff8888, 400);
    spot.angle = (float)math::PI / 5;
    spot.penumbra = 0.3f;
    spot.position = { 8, 10, 5 };
    spot.cast_shadow = true;
    spot.shadow.near = 8;
    spot.shadow.far = 200;
    spot.shadow.map_size = 256;
    spot.shadow.bias = -0.002f;
    spot.shadow.radius = 4;
    scene.add(spot);

    DirectionalLight* sun = drie::directional_light(0x8888ff, 3);
    sun.position = { 3, 12, 17 };
    sun.cast_shadow = true;
    sun.shadow.cascades = 1;
    sun.shadow_extent = 17;
    sun.shadow.near = 0.1f;
    sun.shadow.far = 500;
    sun.shadow.map_size = 512;
    sun.shadow.radius = 4;
    sun.shadow.bias = -0.0005f;
    scene.add(sun);

    MeshPhongMaterial* phong = drie::mesh_phong_material(color: 0x999999, specular: 0x222222, shininess: 0);
    Mesh* knot               = drie::mesh(drie::torus_geometry(25.0f / 18, 8.0f / 18, 20, 75), phong);
    knot.position = { 0, 3, 0 };
    knot.cast_shadow = true;
    knot.receive_shadow = true;
    scene.add(knot);
    BufferGeometry* pillar = drie::cylinder_geometry(0.75f, 0.75f, 7, 32, 1);
    float[2] sides         = { -8, 8 };
    foreach (x : sides) {
        foreach (z : sides) {
            Mesh* m = drie::mesh(pillar, phong);
            m.position = { x, 3.5f, z };
            m.cast_shadow = true;
            m.receive_shadow = true;
            scene.add(m);
        }
    }
    Mesh* ground = drie::mesh(
        drie::plane_geometry(600, 600),
        drie::mesh_phong_material(color: 0x999999, specular: 0x111111, shininess: 0)
    );
    ground.rotation.x = -(float)math::PI / 2;
    ground.cast_shadow = true;
    ground.receive_shadow = true;
    scene.add(ground);

    PerspectiveCamera* camera = drie::perspective_camera(45, (float)W / H, 1, 1000);
    camera.position = { 0, 10, 30 };
    camera.look_at(0, 0, 0);
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    sz size    = (sz)W * H * 4;
    char[] pcf = mem::new_array(char, size);
    char[] vsm = mem::new_array(char, size);
    char[] one = mem::new_array(char, size);
    // (drie's PCF has normal_bias but no depth bias: a floor in texels of 7 cm, the sun's,
    // shadows itself in specks; three's demo has its floor cast, as the VSM renders do)
    ground.cast_shadow = false;
    render_into(renderer, scene, camera, rt, pcf);
    ground.cast_shadow = true;
    view_proj = camera.projection_matrix.mul(camera.matrix_world_inverse);
    renderer.shadow_type = VSM;
    render_into(renderer, scene, camera, rt, vsm);
    spot.shadow.blur_samples = 1;
    sun.shadow.blur_samples = 1;
    render_into(renderer, scene, camera, rt, one);

    // lit by both lights; in the spot light's shadow (of the torus), in the sun's (of the torus),
    // and far out, where the spot light's map does not reach (lit by the sun)
    float lit_p  = lum(pcf, 3, 3);
    float lit_v  = lum(vsm, 3, 3);
    float spot_p = lum(pcf, -2.6f, -2.6f);
    float spot_v = lum(vsm, -2.6f, -2.6f);
    float sun_p  = lum(pcf, -1.2f, -4.5f);
    float sun_v  = lum(vsm, -1.2f, -4.5f);
    float out_p  = lum(pcf, 14, 6);
    float out_v  = lum(vsm, 14, 6);
    io::printfn(
        "lit %.1f / %.1f, spot shadow %.1f / %.1f, sun shadow %.1f / %.1f, outside %.1f / %.1f (PCF / VSM)",
        lit_p,
        lit_v,
        spot_p,
        spot_v,
        sun_p,
        sun_v,
        out_p,
        out_v
    );
    expect(math::abs(lit_p - lit_v) < 3, "lit floor alike");
    expect(spot_p < lit_p - 30 && spot_v < lit_v - 30, "the spot light's shadow in both");
    expect(sun_p < lit_p - 10 && sun_v < lit_v - 10, "the sun's shadow in both");
    expect(math::abs(out_p - out_v) < 3, "outside of the spot light's map lit alike");
    // across the sun's shadow of the back right pillar, out of the spot light's reach
    Vector3 a  = { 5.5f, 0, -10.7f };
    Vector3 b  = { 9.5f, 0, -11.4f };
    int wide_p = edge_width(pcf, a, b);
    int wide_v = edge_width(vsm, a, b);
    io::printfn("edges: %d (PCF), %d (VSM) of 200 steps", wide_p, wide_v);
    expect(wide_v > wide_p, "VSM's edges are wider");
    float spot_1 = lum(one, -2.6f, -2.6f);
    float lit_1  = lum(one, 3, 3);
    io::printfn("blur_samples = 1: lit %.1f, spot shadow %.1f", lit_1, spot_1);
    expect(math::abs(lit_1 - lit_v) < 3 && spot_1 < lit_1 - 30, "blur_samples = 1 renders");

    // side by side: PCF, VSM
    char[] rgb = mem::new_array(char, (usz)W * 2 * (usz)H * 3);
    for (usz y = 0; y < H; y++) {
        for (usz x = 0; x < W; x++) {
            usz s = (y * W + x) * 4;
            rgb[(y * W * 2 + x) * 3:3] = pcf[s:3];
            rgb[(y * W * 2 + W + x) * 3:3] = (args.len > 2 ? one : vsm)[s:3];
        }
    }
    if (!png::write(out_path, rgb, W * 2, H, channels: 3)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s; %d failed", out_path, failures);
    return failures ? 1 : 0;
}