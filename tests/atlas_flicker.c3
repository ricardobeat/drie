// A static shadowed light that flickers down to 0 keeps its tile of the shadow atlas: three
// spot lights with shadow.auto_update off over a box; once their tiles are drawn, a frame
// draws no shadow maps, with the first light at intensity 0 too, and after it comes back.
// (A light at 0 used to drop out of the atlas's list, handing every tile after its own to
// another light: all of them drawn again, twice per flicker.) The second light's shadow is
// still where it was while the first is out.
//
//   just test atlas_flicker
module atlas_flicker_test;
import drie, sokol, drie::readback;
import std::io, std::math;

const int S = 64;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn int main(String[] args) {
    Renderer* renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    Scene* scene = drie::scene();
    Mesh* ground = drie::mesh(drie::plane_geometry(20, 20), drie::mesh_standard_material(color: 0xffffff, roughness: 1));
    ground.rotation = { -(float)math::PI / 2, 0, 0 };
    ground.receive_shadow = true;
    scene.add(ground);
    Mesh* box = drie::mesh(drie::box_geometry(1, 1, 1), drie::mesh_standard_material(color: 0x808080));
    box.position = { 0, 0.5f, 0 };
    box.cast_shadow = true;
    scene.add(box);
    SpotLight*[3] lights;
    float[3] xs = { -3, 0, 3 };
    for (int i = 0; i < 3; i++) {
        SpotLight* l = drie::spot_light(0xffffff, 20, 0, 0.7f, 0, 2);
        l.position = { xs[i], 4, 0.5f };
        l.target.position = { 0, 0, 0 };
        l.cast_shadow = true;
        l.shadow.auto_update = false;
        l.shadow.map_size = 256;
        lights[i] = l;
        scene.add(l);
        scene.add(l.target);
    }
    PerspectiveCamera* camera = drie::perspective_camera(60, 1, 0.1f, 50);
    camera.position = { 0, 8, 6 };
    camera.rotation.x = -0.93f;
    RenderTarget* rt = drie::render_target(S, S, 1, RGBA8);
    renderer.set_render_target(rt);

    int[5] calls;
    for (int f = 0; f < 5; f++) {
        // frames 0, 1: all on; 2, 3: the first out; 4: back
        lights[0].intensity = f == 2 || f == 3 ? 0 : 20;
        renderer.render(scene, camera);
        drie::commit();
        calls[f] = renderer.info.shadow_calls;
    }
    io::printfn("shadow draws per frame: %s", calls);
    expect(calls[0] > 0, "the tiles drawn at first");
    expect(calls[1] == 0, "nothing drawn again while nothing changes");
    expect(calls[2] == 0 && calls[3] == 0, "nothing drawn again when a light goes to 0");
    expect(calls[4] == 0, "nothing drawn again when it comes back");
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
