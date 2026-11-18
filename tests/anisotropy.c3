// Anisotropy and the tangent frame, through the glTF loader: Khronos's AnisotropyRotationTest
// (tests/assets; CC-BY 4.0, © 2023 Analytical Graphics, Inc., model and textures by Ed
// Mackey). Ten glossy bands, lit from above and in front: each highlight must be drawn out
// sideways, a horizontal stripe. The bands are KHR_materials_anisotropy along the mesh's
// tangents (glTF's TANGENT), and those marked "Tangents + Texture", "+ Rotation" and
// "+ Texture + Rotation" have their tangents slanted 30° and turned back by the anisotropy
// map, anisotropy_rotation or both (turned the wrong way, their stripes slant 60°). The band
// marked "Normal map of grooves" has no anisotropy, but a normal map of vertical grooves read
// in the same frame. Compare with
// https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/AnisotropyRotationTest.
//
//   just test anisotropy [out.png]   (tests/build/anisotropy.png)
module anisotropy_test;
import drie, drie::loaders, drie::readback, drie::png, sokol;
import std::io;

const int W = 800;
const int H = 440;

fn int main(String[] args) {
    String out_path = args.len > 1 ? args[1] : "tests/build/anisotropy.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    Scene* scene = drie::scene();
    scene.background = 0x2e3440;
    scene.add(drie::ambient_light(0xffffff, 0.6f));
    DirectionalLight* light = drie::directional_light(0xffffff, 4);
    light.position.set(0, 3, 4);
    scene.add(light);
    GltfModel*? model = loaders::load_gltf("tests/assets/AnisotropyRotationTest/AnisotropyRotationTest.glb");
    if (catch err = model) {
        io::printfn("could not load the model: %s", err);
        return 1;
    }
    scene.add(model.scene);
    PerspectiveCamera* camera = drie::perspective_camera(40, (float)W / H, 0.1f, 100);
    camera.position.set(0, 0, 10);
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
    io::printfn("saved %s: %d meshes from AnisotropyRotationTest", out_path, model.meshes.len());
    return 0;
}
