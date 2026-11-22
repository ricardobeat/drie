// Headless test for Sparse / Split Radiance Cascades (radiance_cascades_sparse).
// Renders a room with a bright red emissive wall and verifies that sparse probes allocated
// from the G-Buffer bounce red light onto an adjacent white wall.
module test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int W = 320;
const int H = 200;

fn int main() {
    Renderer* renderer = drie::renderer(width: W, height: H);
    renderer.init_headless();
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);

    Scene* scene = drie::scene();
    scene.background = 0x050508;

    // White floor and right wall (unlit directly: only lit by red bounce from the left wall)
    MeshStandardMaterial* white = drie::mesh_standard_material(color: 0xe8e8e8, roughness: 0.9f);
    Mesh* floor_m = drie::mesh(drie::box_geometry(8, 0.2f, 8), white);
    floor_m.position = { 0, -0.1f, 0 };
    scene.add(floor_m);

    Mesh* back_wall = drie::mesh(drie::box_geometry(8, 4, 0.2f), white);
    back_wall.position = { 0, 2, -3.5f };
    scene.add(back_wall);

    // Bright red emissive left wall
    MeshStandardMaterial* red_glow = drie::mesh_standard_material(color: 0xff2020, roughness: 0.9f, emissive: 0xff2020, emissive_intensity: 6.0f);
    Mesh* left_wall = drie::mesh(drie::box_geometry(0.2f, 4, 8), red_glow);
    left_wall.position = { -3.0f, 2, 0 };
    scene.add(left_wall);

    // Sparse / Split Radiance Cascades volume
    RadianceCascades* rc = drie::radiance_cascades_sparse({ -4, -0.5f, -4.5f }, { 4, 4.5f, 4.5f }, 0.25f, 4, 8192, 32768);
    rc.sky_from_scene = false;
    rc.sky_color = { 0.01f, 0.01f, 0.015f };
    rc.ground_color = { 0.01f, 0.01f, 0.01f };
    rc.blend = 0.5f;
    rc.voxelize(scene);
    scene.radiance = rc;

    PerspectiveCamera* cam = drie::perspective_camera(55, (float)W / H, 0.1f, 50);
    cam.position = { 1.5f, 1.8f, 2.5f };
    cam.look_at_vec({ -0.5f, 1.5f, -3.5f });

    // Render 8 frames so sparse probes allocate and converge
    for (int f = 0; f < 8; f++) {
        renderer.render(scene, cam);
        drie::commit();
    }

    char[] px = mem::new_array(char, (usz)W * H * 4);
    defer free(px);
    readback::read_image(rt.texture_image, W, H, px.ptr);
    png::write("tests/build/radiance_split.png", px, W, H);

    // Sample the white back wall near the red left wall (center-left of image)
    int sx = W / 2;
    int sy = H / 2;
    usz idx = ((usz)sy * (usz)W + (usz)sx) * 4;
    int r = (int)px[idx];
    int g = (int)px[idx + 1];
    int b = (int)px[idx + 2];
    io::printfn("back wall center pixel: [%d, %d, %d]", r, g, b);
    if (r < g + 20 || r < 40) {
        io::printn("FAIL expected strong red bounce light on white back wall from sparse RC");
        return 1;
    }
    io::printn("ok   sparse Split Radiance Cascades bounced red light onto white wall");
    return 0;
}
