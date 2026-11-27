// Volumetric clouds (drie::postprocessing CloudPass) over hills: a layer between 1.5 and 4 km,
// their shadows drifting over the ground and through the ground fog (the froxel fog), which lies in
// the valleys. Fly up into them and above them.
//
//   just example clouds                       in a window (drag to look round, WASD to fly, E and
//                                             Q up and down, shift faster)
//   just example clouds --png out.png         one frame, headless (--view sunset, ground, valley, inside,
//                                             above, dusk; --size 2560x1440; --off clouds,fog,shadow)
module clouds_example;
import drie, drie::objects, drie::postprocessing, drie::controls, drie::readback, drie::png, drie::gpu_timer, sokol;
import std::io, std::math;

int width            = 1280;
int height           = 720;
const float LAND     = 120000; // the ground's side (m)
const int LAND_CELLS = 512;

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
float coverage      = 0.55f; // --coverage
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

// Value noise and its gradient (for the erosion-like fBm below).
fn Vector3 noise_d(float x, float y) {
    int ix   = (int)math::floor(x);
    int iy   = (int)math::floor(y);
    float fx = x - ix;
    float fy = y - iy;
    float u  = fx * fx * (3 - 2 * fx);
    float v  = fy * fy * (3 - 2 * fy);
    float du = 6 * fx * (1 - fx);
    float dv = 6 * fy * (1 - fy);
    float a  = hash2(ix, iy);
    float b  = hash2(ix + 1, iy);
    float c  = hash2(ix, iy + 1);
    float d  = hash2(ix + 1, iy + 1);
    float k  = a - b - c + d;
    return { a + (b - a) * u + (c - a) * v + k * u * v, du * ((b - a) + k * v), dv * ((c - a) + k * u) };
}

// Height of the ground (m): fBm whose octaves are damped where the slope so far is steep (so that
// the detail gathers in the valleys and the ridges stay sharp, as if eroded), its coordinates
// warped, mountains rising to the north and a broad valley along x = 0.
fn float ground_height(float x, float z) {
    float wx  = x + (value_noise(x / 9000, z / 9000) - 0.5f) * 5000;
    float wz  = z + (value_noise(x / 9000 + 31, z / 9000 - 7) - 0.5f) * 5000;
    float h   = 0;
    float amp = 1;
    float f   = 1.0f / 9000;
    float gx  = 0;
    float gz  = 0;
    for (int o = 0; o < 8; o++) {
        Vector3 n = noise_d(wx * f + o * 17.3f, wz * f - o * 9.1f);
        gx += n.y * amp;
        gz += n.z * amp;
        h += n.x * amp / (1 + (gx * gx + gz * gz) * 0.7f);
        amp *= 0.5f;
        f *= 2.0f;
    }
    float north  = math::clamp((-z + 4000) / 25000, 0.0f, 1.0f);
    float valley = 1 - math::exp(-(x * x) / (2 * 2500.0f * 2500.0f)) * 0.75f;
    float lift   = 420 + north * north * 1500;
    float k      = math::max(h, 0.0f);
    return k * math::sqrt(k) * lift * valley - 60;
}

// A tiling grey detail texture for the ground, around 1 so that it only breaks up the vertex colors
// close by: clumps and speckles over a few octaves of noise. It repeats every 60 m.
const int DETAIL = 256;

fn Texture* detail_texture() {
    char[] px = mem::new_array(char, DETAIL * DETAIL * 4);
    for (int j = 0; j < DETAIL; j++) {
        for (int i = 0; i < DETAIL; i++) {
            float v   = 0;
            float amp = 0.5f;
            int p     = 8;
            for (int o = 0; o < 5; o++) {
                // tileable: the lattice wraps every `p` cells
                float x  = (float)i / DETAIL * p;
                float y  = (float)j / DETAIL * p;
                int ix   = (int)x;
                int iy   = (int)y;
                float fx = x - ix;
                float fy = y - iy;
                float u  = fx * fx * (3 - 2 * fx);
                float w  = fy * fy * (3 - 2 * fy);
                float a  = hash2(ix % p + o * 101, iy % p);
                float b  = hash2((ix + 1) % p + o * 101, iy % p);
                float c  = hash2(ix % p + o * 101, (iy + 1) % p);
                float d  = hash2((ix + 1) % p + o * 101, (iy + 1) % p);
                v += (a + (b - a) * u + (c - a) * w + (a - b - c + d) * u * w) * amp;
                amp *= 0.5f;
                p *= 2;
            }
            float speck = hash2(i + 7777, j) > 0.95f ? 0.88f : 1;
            float g     = math::clamp((0.7f + v * 0.6f) * speck, 0.0f, 1.0f);
            char c      = (char)(g * 255);
            usz k       = (usz)(j * DETAIL + i) * 4;
            px[k] = c;
            px[k + 1] = c;
            px[k + 2] = c;
            px[k + 3] = 255;
        }
    }
    Texture* t = drie::data_texture(px, DETAIL, DETAIL);
    t.wrap_s = REPEAT_WRAPPING;
    t.wrap_t = REPEAT_WRAPPING;
    t.repeat = { LAND / 60, LAND / 60 };
    return t;
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
            // meadow and forest in the low ground, scree and rock where it is steep, snow up high
            Vector3 grass  = { 0.2f, 0.25f, 0.09f };
            Vector3 dry    = { 0.36f, 0.31f, 0.17f };
            Vector3 forest = { 0.06f, 0.1f, 0.05f };
            Vector3 rock   = { 0.3f, 0.27f, 0.24f };
            Vector3 scree  = { 0.42f, 0.39f, 0.34f };
            Vector3 snow   = { 0.88f, 0.9f, 0.94f };
            float t        = value_noise(x / 900, z / 900) * 0.7f + value_noise(x / 210, z / 210) * 0.3f;
            Vector3 col    = grass + (dry - grass) * math::clamp(t * 1.4f - 0.3f + y / 2500, 0.0f, 1.0f);
            float trees    = math::clamp(
                (value_noise(x / 400 + 5, z / 400) - 0.45f) * 5,
                0.0f,
                1.0f
            ) * math::clamp((nrm.y - 0.8f) * 8, 0.0f, 1.0f) * math::clamp((1500 - y) / 300, 0.0f, 1.0f);
            col = col + (forest - col) * trees;
            float steep   = math::clamp((0.86f - nrm.y) * 7, 0.0f, 1.0f);
            Vector3 stone = scree + (rock - scree) * value_noise(x / 150, z / 150);
            col = col + (stone - col) * steep;
            float line = 1400 + (value_noise(x / 600, z / 600) - 0.5f) * 500;
            float high = math::clamp((y - line) / 200, 0.0f, 1.0f) * math::clamp((nrm.y - 0.55f) * 5, 0.0f, 1.0f);
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
    m.map = detail_texture();
    Mesh* ground = drie::mesh(g, m);
    ground.cast_shadow = true;
    ground.receive_shadow = true;
    return ground;
}

// ---- The scene

fn void build_scene(Vector3 sun_dir) {
    scene = drie::scene();
    scene.background = 0x8cb0d8;
    fog = drie::volumetric_fog(0xb8c4d0, 0.0002f, 0);
    fog.height_base = 0;
    fog.height_falloff = 1.0f / 180;
    fog.sun_sharpness = 4;
    fog.sun_shafts = 0; // (its shafts would cost much over 80 km; the clouds darken it all the same)
    fog.shade_density = 0.7f;
    fog.noise_amount = 0.4f;
    fog.noise_scale = 1.0f / 300;
    fog.noise_wind = { 3, 0, 1 };
    fog.resolution = 0.125f;
    fog.slices = 64;
    if (!off.contains("fog")) scene.fog = fog;

    sky = objects::sky();
    sky.sun_position = sun_dir;
    sky.cloud_coverage = 0; // the volumetric ones in its place
    sky.turbidity = 4.0f;
    sky.rayleigh = 2.0f;
    sky.mie_coefficient = 0.004f;
    sky.mie_directional_g = 0.85f;
    scene.add(sky);
    renderer.tone_mapping = AGX_TONE_MAPPING;
    renderer.tone_mapping_exposure = 1.6f;

    scene.add(drie::hemisphere_light(0x9fb6d8, 0x3a3428, 0.6f));
    sun = drie::directional_light(0xfff0dc, 4.0f);
    sun.cast_shadow = !off.contains("shadow");
    sun.shadow.map_size = 2048;
    sun.shadow.cascades = 4;
    sun.shadow.max_distance = 6000;
    sun.shadow.far = 20000;
    sun.shadow.normal_bias = 2;
    scene.add(sun);
    scene.add(sun.target);
    scene.add(make_ground());

    clouds = postprocessing::cloud_pass(renderer);
    clouds.coverage = coverage;
    clouds.top = 6000; // towering
    clouds.weather_scale = 160000; // masses of several km
    clouds.shape_scale = 7000;
    clouds.detail_scale = 700;
    clouds.wind = { 30, 9 }; // a brisk wind, so that they are seen to drift
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
    float low = math::clamp(1 - dir.y * 3, 0.0f, 1.0f);
    low = low * low * (3 - 2 * low);
    Vector3 white = { 1, 0.95f, 0.88f };
    Vector3 gold  = { 1, 0.48f, 0.2f };
    Vector3 c     = white + (gold - white) * low;
    sun.color = drie::rgb(c.x, c.y, c.z);
    sun.intensity = 4.0f - low * 1.2f;
    clouds.sun_direction = dir;
    clouds.sun_color = c * (18 + low * 6);
    clouds.ambient_top = (Vector3) { 0.32f, 0.42f, 0.62f } * (1 - low * 0.55f) + (Vector3) { 0.2f, 0.08f, 0.1f } * low;
    clouds.ambient_bottom = (Vector3) {
        0.14f,
        0.15f,
        0.16f
    } * (1 - low * 0.6f) + (Vector3) { 0.06f, 0.03f, 0.03f } * low;
    clouds.haze_color = (Vector3) { 0.5f, 0.62f, 0.8f } * (1 - low * 0.6f) + gold * (low * 0.35f);
    Vector3 f = (Vector3) { 0.72f, 0.77f, 0.82f } * (1 - low) + (Vector3) { 0.62f, 0.56f, 0.6f } * low;
    fog.color = drie::rgb(f.x, f.y, f.z);
    // a hazier, redder sky as the sun sets (Preetham reddens only with much haze)
    sky.turbidity = 4 + low * 6;
    sky.rayleigh = 2 + low * 1.5f;
    sky.mie_directional_g = 0.85f - low * 0.1f;
}

// In a window, the scene is drawn at no more than this many pixels (about 1600 x 900), stretched
// to the window by the output pass: on a Retina screen the window has four times its points.
const float MAX_WINDOW_PIXELS = 1600.0f * 900.0f;

fn void render_frame(RenderTarget* target) {
    int w = target ? target.width : sapp::width();
    int h = target ? target.height : sapp::height();
    if (!target && (float)w * h > MAX_WINDOW_PIXELS) {
        float k = math::sqrt(MAX_WINDOW_PIXELS / ((float)w * h));
        w = (int)(w * k);
        h = (int)(h * k);
    }
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
    { "sunset", { 6000, 16000 }, 1250, { -3000, -2000 }, 1500, { -0.35f, 0.07f, -0.93f } },
    { "ground", { 2000, 9000 }, 150, { -3000, -4000 }, 2200, { -0.5f, 0.45f, 0.75f } },
    { "backlit", { 2000, 9000 }, 150, { -3000, -4000 }, 2200, { 0.3f, 0.3f, -0.9f } },
    { "valley", { 0, 15000 }, 900, { 300, 5000 }, 0, { -0.45f, 0.55f, -0.7f } },
    { "inside", { -1500, 2000 }, 2350, { 2500, -4000 }, 2500, { 0.4f, 0.5f, -0.75f } },
    { "above", { 0, 24000 }, 7000, { 4000, 8000 }, 4000, { 0.62f, 0.2f, -0.76f } },
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
    // seconds, from the renderer's clock (t is not in seconds)
    static float last = -1;
    float now         =  (float)stm::sec(stm::now());
    float dt          =  last < 0 ? 0 : math::min(now - last, 0.1f);
    last = now;
    if (!controls) {
        // the GPU is there from the first frame on (not yet in main())
        build_scene(start_view.sun);
        place(start_view);
        controls = controls::walk_controls(camera, camera.position);
        controls.drag_look = true;
        controls.walk_speed = 2500;
        controls.walk_spring = 1.2f; // glides up to speed and coasts to a stop
    }
    time += dt;
    controls.update(dt);
    render_frame(null);
}

fn int main(String[] args) {
    String png_path;
    String view_name = "sunset";
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
        ulong t0 = stm::now();
        for (int f = 0; f < 120; f++) {
            render_frame(rt);
            drie::commit();
        }
        // (the GPU can run at most a frame or two behind: about its time per frame)
        io::printfn("wall %.2f ms a frame", stm::ms(stm::since(t0)) / 120);
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