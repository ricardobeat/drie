// bench/cubes.js in C3 on drie, timed like `driejs --frames N`:
//   build/cubes [frames] [--profile]    (--profile: drie's time per stage)
module main;
import drie, drie::profile, std::io, std::time, std::math;

const int WIDTH = 1280;
const int HEIGHT = 720;
const int COUNT = 5000;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
Mesh*[COUNT] cubes;
int side;
int frame;

fn void animate(double time)
{
    float t = (float)frame++ / 60;
    for (int i = 0; i < COUNT; i++) cubes[i].rotation.y = t + (float)i;
    camera.position = { math::cos(t * 0.2f) * (float)side * 3, (float)side * 2, math::sin(t * 0.2f) * (float)side * 3 };
    camera.look_at(0, 0, 0);
    renderer.render(scene, camera);
}

fn int main(String[] args)
{
    int frames = args.len > 1 ? args[1].to_int() ?? 600 : 600;
    profile::enabled = args.len > 2 && args[2] == "--profile";
    renderer = drie::renderer(antialias: true, width: WIDTH, height: HEIGHT, title: "cubes");
    scene = drie::scene();
    camera = drie::perspective_camera(60, (float)WIDTH / HEIGHT, 0.1, 500);

    scene.add(drie::ambient_light(0xffffff, 0.5));
    DirectionalLight* sun = drie::directional_light(0xffffff, 2);
    sun.position = { 5, 8, 3 };
    scene.add(sun);

    BufferGeometry* geometry = (BufferGeometry*)drie::box_geometry(1, 1, 1);
    Material* material = (Material*)drie::mesh_standard_material(color: 0x44aa88, roughness: 0.5);
    side = (int)math::ceil(math::sqrt((double)COUNT / 10));
    for (int g = 0; g * 10 < COUNT; g++)
    {
        Group* group = drie::group();
        group.position = { (float)(g % side) * 4 - (float)side * 2, 0, (float)(g / side) * 4 - (float)side * 2 };
        scene.add(group);
        for (int i = 0; i < 10 && g * 10 + i < COUNT; i++)
        {
            Mesh* cube = drie::mesh(geometry, material);
            cube.position = { (float)(i % 3) - 1, (float)(i / 3) * 0.5f, 0 };
            cube.scale = { 0.3, 0.3, 0.3 };
            group.add(cube);
            cubes[g * 10 + i] = cube;
        }
    }

    renderer.init_headless();
    renderer.set_render_target(drie::render_target(WIDTH, HEIGHT));
    for (int i = 0; i < 10; i++)
    {
        animate(0);
        drie::commit();
    }
    profile::reset();
    time::Clock clock = time::clock::now();
    for (int i = 0; i < frames; i++)
    {
        animate(0);
        drie::commit();
    }
    io::printfn("%d frames, %d draw calls, %d triangles", frames, renderer.info.calls, renderer.info.triangles);
    io::printfn("frame        %8.3f ms", (double)(long)clock.to_now() / 1e6 / frames);
    if (profile::enabled) profile::report(frames);
    return 0;
}
