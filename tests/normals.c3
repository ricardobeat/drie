// Normal maps' tangent frame, through the glTF loader: Khronos's NormalTangentTest (left;
// CC0) and NormalTangentMirrorTest (right; CC-BY 4.0, © 2018 Analytical Graphics, Inc., by
// Ed Mackey), in tests/assets, under a sky with a level horizon. In each pair of cells the left
// one is real geometry, bumps and bowls, and the right one a flat square with a normal map of
// the same shapes: the two must reflect the sky alike, the horizon level. The first model has
// no tangents (the frame comes from the uv's derivatives, and the loader turns glTF's normal
// maps' y the right way for it); the second supplies them (glTF's TANGENT), with the columns on
// its right mirrored copies, which only the tangents' w gets right. Compare with
// https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/NormalTangentTest.
//
//   just test normals [out.png]   (tests/build/normals.png)
module normals_test;
import drie, drie::loaders, drie::readback, drie::png, sokol;
import std::io, std::math;

const int W = 960;
const int H = 480;

// A sky with a level horizon to reflect: blue above, warm at the horizon, a checkered band
// along it, dark ground below, in linear light.
fn Texture* sky_map() {
    int w = 512;
    int h = 256;
    float[] px = mem::new_array(float, (usz)(w * h * 4));
    for (int y = 0; y < h; y++) {
        // latitude: +90° at the top row
        float lat = (0.5f - ((float)y + 0.5f) / h) * (float)math::PI;
        for (int x = 0; x < w; x++) {
            float lon = ((float)x + 0.5f) / w * 2 * (float)math::PI;
            float up = math::sin(lat);
            Vector3 c = up > 0 ? (Vector3) { 0.25f, 0.45f, 0.9f } * up + (Vector3) { 1.0f, 0.6f, 0.35f } * (1 - up)
                : (Vector3) { 0.12f, 0.1f, 0.08f };
            if (math::abs(lat) < 0.2f && (((int)(lon * 8 / (float)math::PI) + (lat > 0 ? 1 : 0)) % 2 == 0)) {
                c = { 0.05f, 0.05f, 0.06f };
            }
            usz i = (usz)((y * w + x) * 4);
            px[i] = c.x;
            px[i + 1] = c.y;
            px[i + 2] = c.z;
            px[i + 3] = 1;
        }
    }
    return drie::float_texture(px, w, h);
}

fn int main(String[] args) {
    String out_path = args.len > 1 ? args[1] : "tests/build/normals.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    Scene* scene = drie::scene();
    Texture* sky = sky_map();
    scene.background_texture = sky;
    scene.environment = sky;
    String[2] paths = {
        "tests/assets/NormalTangentTest/NormalTangentTest.glb",
        "tests/assets/NormalTangentMirrorTest/NormalTangentMirrorTest.glb",
    };
    float[2] xs = { -1.5f, 1.55f };
    foreach (i, path : paths) {
        GltfModel*? model = loaders::load_gltf(path);
        if (catch err = model) {
            io::printfn("could not load %s: %s", path, err);
            return 1;
        }
        model.scene.position.set(xs[i], 0.12f, 0);
        scene.add(model.scene);
    }
    PerspectiveCamera* camera = drie::perspective_camera(40, (float)W / H, 0.1f, 100);
    camera.position.set(0, 0, 4.4f);
    camera.look_at_vec({ 0, 0, 0 });
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
    char[] px = mem::new_array(char, (usz)W * H * 4);
    readback::read_image(rt.texture_image, W, H, px.ptr);
    if (!png::write(out_path, px, W, H)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s: NormalTangentTest, NormalTangentMirrorTest", out_path);
    return 0;
}
