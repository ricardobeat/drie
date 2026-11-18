// glTF through cgltf: Khronos's TextureTransformTest (tests/assets, CC0), six quads whose
// textures are placed by KHR_texture_transform. Top row: offsets in u, v and both (no yellow
// may show); bottom row: rotation, scale, and all three, each marking its result green
// ("correct"), yellow (not applied) or red (applied the wrong way). Compare with
// https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/TextureTransformTest.
//
//   just test gltf [out.png]   (tests/build/gltf.png)
module gltf_test;
import drie, drie::loaders, drie::readback, drie::png, sokol;
import std::io;

const int W = 640;
const int H = 400;

fn int main(String[] args) {
    String out_path = args.len > 1 ? args[1] : "tests/build/gltf.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.tone_mapping = NO_TONE_MAPPING;
    Scene* scene = drie::scene();
    scene.background = 0x404040;
    // (an ambient light of π shows a matte surface's own color, as in three)
    scene.add(drie::ambient_light(0xffffff, 3.14159265f));
    GltfModel*? model = loaders::load_gltf("tests/assets/TextureTransformTest/TextureTransformTest.gltf");
    if (catch err = model) {
        io::printfn("could not load the model: %s", err);
        return 1;
    }
    scene.add(model.scene);
    PerspectiveCamera* camera = drie::perspective_camera(40, (float)W / H, 0.1f, 100);
    camera.position.set(0, 0, 3.3f);
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
    io::printfn("saved %s: %d meshes from TextureTransformTest", out_path, model.meshes.len());
    return 0;
}
