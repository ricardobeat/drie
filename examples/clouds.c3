// Volumetric clouds (drie::postprocessing CloudPass) over hills: a layer between 1.5 and 4 km,
// their shadows drifting over the ground and through the ground fog (the froxel fog), which lies in
// the valleys. Fly up into them and above them.
//
//   just example clouds                       in a window (drag to look round, WASD to fly, E and
//                                             Q up and down, shift faster)
//   just example clouds --png out.png         one frame, headless (--view ground, valley, inside,
//                                             above, dusk; --size 2560x1440; --off clouds,fog,shadow)
module clouds_example;
import drie, drie::objects, drie::postprocessing, drie::controls, drie::readback, drie::png, drie::gpu_timer, sokol;
import std::io, std::math;

int width            = 1280;
int height           = 720;
const float LAND     = 120000; // the ground's side (m)
const int LAND_CELLS = 600;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
WalkControls* controls;
Sky* sky;
DirectionalLight* sun;
Fog* fog;
CloudPass* clouds;
RenderTarget* scene_rt;
OutputPass* output;
String off;
float time          = 600;
float coverage      = 0.48f; // --coverage
String tune;
float cloud_res     = 0; // --res: CloudPass.resolution (and half that near), 0: its own
float cloud_history = 0; // --history: CloudPass.history, 0: its own

// ---- The ground

fn float hash2(int x, int y) {
    uint h = (uint)x * 0x8da6b343 ^ (uint)y * 0xd8163841;
    h ^= h >> 15;
    h *= 0x2c1b3c6d;
    h ^= h >> 12;
    h *= 0x297a2d39;
    h ^= h >> 15;
    return (float)(h & 0xffffff) / 16777216.0f;
}

fn float value_noise(float x, float y) {
    int ix   = (int)math::floor(x);
    int iy   = (int)math::floor(y);
    float fx = x - ix;
    float fy = y - iy;
    float u  = fx * fx * (3 - 2 * fx);
    float v  = fy * fy * (3 - 2 * fy);
    float a  = hash2(ix, iy);
    float b  = hash2(ix + 1, iy);
    float c  = hash2(ix, iy + 1);
    float d  = hash2(ix + 1, iy + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

// Height of the ground (m): rolling hills, ridged mountains to the north, a valley along x = 0.
fn float ground_height(float x, float z) {
    float h   = 0;
    float amp = 1;
    float f   = 1.0f / 6000;
    for (int o = 0; o < 6; o++) {
        float n = value_noise(x * f + o * 17.3f, z * f - o * 9.1f);
        if (o < 3) n = 1 - math::abs(n * 2 - 1); // ridges
        h += n * amp;
        amp *= 0.45f;
        f *= 2.1f;
    }
    float north  = math::clamp((-z - 2000) / 9000, 0.0f, 1.0f);
    float valley = 1 - math::exp(-(x * x) / (2 * 1800.0f * 1800.0f)) * 0.7f;
    return (h * 380 + north * north * h * 1300) * valley - 40;
}

fn Mesh* make_ground() {
    BufferGeometry* g = drie::buffer_geometry();
    int n             = LAND_CELLS;
    float cell        = LAND / n;
    for (int j = 0; j <= n; j++) {
        for (int i = 0; i <= n; i++) {
            float x = -LAND / 2 + i * cell;
            float z = -LAND / 2 + j * cell;
            float y =  ground_height(x, z);
            float e =  4;
            Vector3 nrm = ((Vector3) {
                ground_height(x - e, z) - ground_height(x + e, z),
                2 * e,
                ground_height(x, z - e) - ground_height(x, z + e)
            }).normalize();
            // grass in the low ground, rock where it is steep, snow up high
            Vector3 grass = { 0.16f, 0.22f, 0.08f };
            Vector3 dry   = { 0.3f, 0.28f, 0.16f };
            Vector3 rock  = { 0.28f, 0.26f, 0.24f };
            Vector3 snow  = { 0.85f, 0.87f, 0.9f };
            float t       = value_noise(x / 700, z / 700);
            Vector3 col   = grass + (dry - grass) * t;
            float steep   = math::clamp((0.9f - nrm.y) * 6, 0.0f, 1.0f);
            col = col + (rock - col) * steep;
            float high = math::clamp((y - 1100) / 250, 0.0f, 1.0f) * math::clamp((nrm.y - 0.6f) * 4, 0.0f, 1.0f);
            col = col + (snow - col) * high;
            g.add_vertex({ x, y, z }, nrm, { (float)i / n, (float)j / n }, col);
        }
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            uint a = (uint)(j * (n + 1) + i);
            uint b = a + 1;
            uint c = a + (uint)(n + 1);
            uint d = c + 1;
            g.add_triangle(a, c, b);
            g.add_triangle(b, c, d);
        }
    }
    MeshStandardMaterial* m = drie::mesh_standard_material(0xffffff, roughness: 0.95f, vertex_colors: true);
    Mesh* ground            = drie::mesh(g, m);
    ground.cast_shadow = true;
    ground.receive_shadow = true;
    return ground;
}

// ---- The scene

fn void build_scene(Vector3 sun_dir) {
    scene = drie::scene();
    scene.background = 0x8cb0d8;
    fog = drie::volumetric_fog(0xb8c4d0, 0.00035f, 0);
    fog.height_base = 0;
    fog.height_falloff = 1.0f / 250;
    fog.sun_sharpness = 4;
    fog.sun_shafts = 1;
    fog.shade_density = 0.7f;
    fog.noise_amount = 0.4f;
    fog.noise_scale = 1.0f / 300;
    fog.noise_wind = { 3, 0, 1 };
    fog.resolution = 0.25f;
    fog.slices = 128;
    if (!off.contains("fog")) scene.fog = fog;

    sky = objects::sky();
    sky.sun_position = sun_dir;
    sky.cloud_coverage = 0; // the volumetric ones in its place
    sky.turbidity = 2.5f;
    sky.rayleigh = 1.2f;
    sky.mie_coefficient = 0.004f;
    sky.mie_directional_g = 0.85f;
    scene.add(sky);
    renderer.tone_mapping = AGX_TONE_MAPPING;
    renderer.tone_mapping_exposure = 1.6f;

    scene.add(drie::hemisphere_light(0x9fb6d8, 0x3a3428, 0.6f));
    sun = drie::directional_light(0xfff0dc, 4.0f);
    sun.cast_shadow = !off.contains("shadow");
    sun.shadow.map_size = 4096;
    sun.shadow.cascades = 4;
    sun.shadow.max_distance = 6000;
    sun.shadow.far = 20000;
    sun.shadow.normal_bias = 2;
    scene.add(sun);
    scene.add(sun.target);
    scene.add(make_ground());

    clouds = postprocessing::cloud_pass(renderer);
    clouds.coverage = coverage;
    clouds.time = time;
    if (cloud_res > 0) {
        clouds.resolution = cloud_res;
        clouds.near_resolution = cloud_res / 2;
    }
    if (cloud_history > 0) clouds.history = cloud_history;
    if (tune) {
        String[] t = tune.tsplit(",");
        if (t.len > 0) clouds.density = (float)t[0].to_double()!!;
        if (t.len > 1) clouds.detail_erosion = (float)t[1].to_double()!!;
        if (t.len > 2) clouds.detail_scale = (float)t[2].to_double()!!;
        if (t.len > 3) clouds.extinction = (float)t[3].to_double()!!;
    }
    set_sun(sun_dir);
}

// Points the sun, the sky's and the clouds', along `dir`, and colors their light by its height.
fn void set_sun(Vector3 dir) {
    dir = dir.normalize();
    sky.sun_position = dir;
    sun.target.position = camera ? camera.position : (Vector3) { };
    sun.position = sun.target.position + dir * 10000;
    float low     = math::clamp(1 - dir.y * 3, 0.0f, 1.0f);
    Vector3 white = { 1, 0.95f, 0.88f };
    Vector3 gold  = { 1, 0.62f, 0.35f };
    Vector3 c     = white + (gold - white) * low;
    sun.color = drie::rgb(c.x, c.y, c.z);
    clouds.sun_direction = dir;
    clouds.sun_color = c * 18;
    clouds.ambient_top = (Vector3) { 0.32f, 0.42f, 0.62f } * (1 - low * 0.6f);
    clouds.ambient_bottom = (Vector3) { 0.14f, 0.15f, 0.16f } * (1 - low * 0.6f);
    clouds.haze_color = (Vector3) { 0.5f, 0.62f, 0.8f } * (1 - low * 0.5f) + gold * (low * 0.2f);
}

fn void render_frame(RenderTarget* target) {
    int w = target ? target.width : sapp::width();
    int h = target ? target.height : sapp::height();
    if (!scene_rt) {
        scene_rt = drie::render_target(w, h, 1, RGBA16F);
        output = postprocessing::output_pass(renderer);
    } else if (scene_rt.width != w || scene_rt.height != h) {
        scene_rt.set_size(w, h);
    }
    if (camera.aspect != (float)w / h) {
        camera.aspect = (float)w / h;
        camera.update_projection_matrix();
    }
    camera.update_matrix_world();
    sky.follow(camera.get_world_position(), camera.far);
    sun.target.position = camera.position;
    sun.position = camera.position + sky.sun_position.normalize() * 10000;
    clouds.time = time;
    bool use_clouds = !off.contains("clouds");
    if (use_clouds) {
        clouds.prepare(camera);
    } else {
        renderer.cloud_shadow = { };
    }
    renderer.set_render_target(scene_rt);
    renderer.render(scene, camera);
    if (use_clouds) clouds.render(scene_rt, camera);
    output.set_source(scene_rt.texture);
    output.render(target);
    if (!target) drie::commit();
}

// ---- Views

struct View {
    String name;
    Vector2 at; // x, z
    float height; // over the ground there (m)
    Vector2 look; // x, z looked at
    float look_height; // over the ground there
    Vector3 sun;
}

const View[*] VIEWS = {
    { "ground", { 2000, 9000 }, 150, { -3000, -4000 }, 2200, { -0.5f, 0.45f, 0.75f } },
    { "backlit", { 2000, 9000 }, 150, { -3000, -4000 }, 2200, { 0.3f, 0.3f, -0.9f } },
    { "valley", { 0, 15000 }, 900, { 300, 5000 }, 0, { -0.45f, 0.55f, -0.7f } },
    { "inside", { -1500, 2000 }, 2350, { 2500, -4000 }, 2500, { 0.4f, 0.5f, -0.75f } },
    { "above", { 0, 6000 }, 5600, { 4000, -8000 }, 3800, { 0.62f, 0.2f, -0.76f } },
    {
        "dusk",
        { 1200, 7000 },
        150,
        { -1500, -6000 },
        1400,
        { -0.25f, 0.06f, -0.97f }
    }, // flying into a cloud: towards where the weather map has the most cloud near the middle
    // (place()), from `height` meters away
    { "approach", { 0, 0 }, 1500, { 0, 0 }, 0, { 0.4f, 0.5f, -0.75f } },
    { "cloud", { 0, 0 }, -150, { 0, 0 }, 0, { 0.4f, 0.5f, -0.75f } },
};

fn View view_named(String name) {
    foreach (v : VIEWS) {
        if (v.name == name) return v;
    }
    io::eprintfn("no view %s", name);
    return VIEWS[0];
}

// Whether the weather map has cloud at x, z (CloudPass's own test: coverage over 1 - coverage).
fn bool cloud_at(float x, float z) {
    int n      = postprocessing::CLOUD_WEATHER;
    float span = clouds.weather_scale;
    float u    = (x + clouds.wind.x * clouds.time) / span;
    float w    = (z + clouds.wind.y * clouds.time) / span;
    int i      = (int)math::floor((u - math::floor(u)) * n) % n;
    int j      = (int)math::floor((w - math::floor(w)) * n) % n;
    return clouds.weather_px[(j * n + i) * 4] / 255.0f > 1 - clouds.coverage + 0.05f;
}

// The camera at a view: its heights are over the ground (absolute when the ground is under the
// clouds' layer anyway).
fn void place(View v) {
    if (v.name == "cloud" || v.name == "approach") {
        // the weather map's cloudiest texel within 10 km, where its (moved) texels are now
        int n      =  postprocessing::CLOUD_WEATHER;
        float span =  clouds.weather_scale;
        float best = -1;
        Vector3 at;
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                float x = (i + 0.5f) / n * span - clouds.wind.x * clouds.time;
                float z = (j + 0.5f) / n * span - clouds.wind.y * clouds.time;
                x -= math::round(x / span) * span;
                z -= math::round(z / span) * span;
                if (x * x + z * z > 10000.0f * 10000.0f) continue;
                float c    = clouds.weather_px[(j * n + i) * 4] / 255.0f;
                float type = clouds.weather_px[(j * n + i) * 4 + 1] / 255.0f;
                c *= 0.6f + 0.4f * type;
                if (c > best) {
                    best = c;
                    at = { x, clouds.bottom + 450, z };
                }
            }
        }
        // back out of it to its edge (where the weather map has no cloud), then `height` further
        Vector3 dir  = ((Vector3) { 1, 0.06f, -0.3f }).normalize();
        Vector3 edge = at;
        for (int k = 0; k < 400 && cloud_at(edge.x, edge.z); k++) edge -= dir * 50;
        camera.position = edge - dir * v.height;
        camera.look_at_vec(edge + dir * 1000);
        camera.update_matrix_world();
        return;
    }
    camera.position = { v.at.x, ground_height(v.at.x, v.at.y) + v.height, v.at.y };
    camera.look_at_vec({ v.look.x, ground_height(v.look.x, v.look.y) + v.look_height, v.look.y });
    camera.update_matrix_world();
}

// ---- In a window

View start_view;

fn void animate(double t) {
    static double last = -1;
    float dt           =  last < 0 ? 1.0f / 60 : (float)(t - last);
    last = t;
    if (!controls) {
        // the GPU is there from the first frame on (not yet in main())
        build_scene(start_view.sun);
        place(start_view);
        controls = controls::walk_controls(camera, camera.position);
        controls.drag_look = true;
        controls.walk_speed = 150;
    }
    time += dt;
    controls.update(dt);
    render_frame(null);
}

fn int main(String[] args) {
    String png_path;
    String view_name = "ground";
    bool gpu_time;
    for (usz i = 1; i < args.len; i++) {
        switch (args[i]) {
            case "--png": if (i + 1 < args.len) png_path = args[++i];
            case "--view": if (i + 1 < args.len) view_name = args[++i];
            case "--off": if (i + 1 < args.len) off = args[++i];
            case "--time": if (i + 1 < args.len) time = (float)args[++i].to_double()!!;
            case "--gpu-time": gpu_time = true;
            case "--coverage": if (i + 1 < args.len) coverage = (float)args[++i].to_double()!!;
            case "--tune":
                // --tune density,erosion,detail_scale,extinction (testing)
                if (i + 1 < args.len) tune = args[++i];
            case "--res": if (i + 1 < args.len) cloud_res = (float)args[++i].to_double()!!;
            case "--history": if (i + 1 < args.len) cloud_history = (float)args[++i].to_double()!!;
            case "--size":
                if (i + 1 < args.len) {
                    String[] wh = args[++i].tsplit("x");
                    width = wh[0].to_int()!!;
                    height = wh[1].to_int()!!;
                }
            default: io::eprintfn("unknown option %s", args[i]);
        }
    }
    View v = view_named(view_name);
    if (!png_path) {
        renderer = drie::renderer(antialias: false, width: width, height: height, title: "drie: clouds");
        camera = drie::perspective_camera(60, (float)width / height, 1, 80000);
        start_view = v;
        renderer.set_animation_loop(&animate);
        return 0;
    }
    renderer = drie::renderer(antialias: false, width: width, height: height);
    renderer.init_headless();
    renderer.auto_time = false;
    camera = drie::perspective_camera(60, (float)width / height, 1, 80000);
    build_scene(v.sun);
    place(v);
    RenderTarget* rt = drie::render_target(width, height, 1, RGBA8);
    // the clouds blend over frames: enough of them for the blend to settle
    for (int f = 0; f < 48; f++) {
        render_frame(rt);
        drie::commit();
    }
    if (gpu_time) {
        gpu_timer::enabled = true;
        for (int f = 0; f < 60; f++) {
            render_frame(rt);
            drie::commit();
        }
        gpu_timer::reset();
        for (int f = 0; f < 120; f++) {
            render_frame(rt);
            drie::commit();
        }
        gpu_timer::poll();
        gpu_timer::report();
    }
    char[] px = mem::new_array(char, (sz)width * height * 4);
    readback::read_image(rt.texture_image, width, height, px.ptr);
    if (!png::write(png_path, px, width, height)) {
        io::eprintfn("could not write %s", png_path);
        return 1;
    }
    io::printfn("saved %s", png_path);
    return 0;
}