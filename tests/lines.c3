// Lines: a grid of LineSegments with vertex colors (as three's GridHelper makes it), a red
// Line (a sine wave), a see-through green LineLoop (a circle), a yellow dashed Line, and a box
// on a floor in the sun's shadow, seen through fog. Checks what the renderer counts, that
// lines cast no shadow, that a Raycaster passes through them and that dashes leave gaps, then
// renders the scene with and without its lines and saves both side by side.
//
//   just test lines [out.png]
module lines_test;
import drie, drie::png, sokol, drie::readback;
import std::io, std::math, std::collections::list;

const int W = 480;
const int H = 270;
const int GRID = 10;
const int WAVE = 64;
const int CIRCLE = 48;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
LineSegments* grid;
Line* wave;
LineLoop* circle;
Line* dashed;
int failures;

fn void expect(bool ok, String what)
{
    if (!ok) failures++;
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
}

// GridHelper(size, divisions, color_center_line, color_grid): GRID + 1 lines each way
fn LineSegments* grid_helper(float size, Vector3 center_color, Vector3 color)
{
    List{float} positions;
    positions.tinit();
    float half = size / 2;
    float step = size / GRID;
    for (int i = 0; i <= GRID; i++)
    {
        float k = -half + i * step;
        positions.push_all({ -half, 0, k, half, 0, k, k, 0, -half, k, 0, half });
    }
    BufferGeometry* g = drie::buffer_geometry(positions.array_view());
    for (uint v = 0; v < g.vertex_count(); v++) g.set_vertex_color(v, v / 4 == GRID / 2 ? center_color : color);
    return drie::line_segments(g, drie::line_basic_material(vertex_colors: true));
}

fn void build_scene()
{
    scene = drie::scene();
    scene.background = 0x202428;
    scene.fog = drie::fog(0x202428, 8, 30);
    scene.add(drie::hemisphere_light(0xffffff, 0x404040, 1));
    DirectionalLight* sun = drie::directional_light(0xffffff, 2);
    sun.position = { 3, 6, 2 };
    sun.cast_shadow = true;
    scene.add(sun);

    Mesh* floor = drie::mesh(drie::plane_geometry(20, 20), drie::mesh_standard_material(color: 0x606870));
    floor.rotation.x = -(float)math::PI / 2;
    floor.position.y = -0.01f;
    floor.receive_shadow = true;
    scene.add(floor);
    Mesh* box = drie::mesh(drie::box_geometry(1, 1, 1), drie::mesh_standard_material(color: 0xc08040));
    box.position = { -2, 0.5f, 0 };
    box.cast_shadow = true;
    scene.add(box);

    grid = grid_helper(10, { 1, 1, 1 }, { 0.45f, 0.45f, 0.45f });
    // (asks for a shadow it must not get: lines cast none)
    grid.cast_shadow = true;
    scene.add(grid);

    float[WAVE * 3] w;
    for (int i = 0; i < WAVE; i++)
    {
        float t = (float)i / (WAVE - 1);
        w[i * 3] = -4 + 8 * t;
        w[i * 3 + 1] = 1.5f + 0.6f * math::sin(t * 4 * (float)math::PI);
        w[i * 3 + 2] = -1;
    }
    wave = drie::line(drie::buffer_geometry(w[..]), drie::line_basic_material(color: 0xff3020));
    scene.add(wave);

    float[CIRCLE * 3] c;
    for (int i = 0; i < CIRCLE; i++)
    {
        float a = (float)i / CIRCLE * 2 * (float)math::PI;
        c[i * 3] = 1.5f * math::cos(a);
        c[i * 3 + 1] = 0.05f;
        c[i * 3 + 2] = 1.5f * math::sin(a);
    }
    circle = drie::line_loop(
        drie::buffer_geometry(c[..]),
        drie::line_basic_material(color: 0x30ff60, opacity: 0.6f, transparent: true)
    );
    circle.position = { 2, 0, 1 };
    scene.add(circle);

    // dashes and gaps of 0.25 along a line of 6, in front: about half of it drawn
    dashed = drie::line(
        drie::buffer_geometry({ -3, 0.3f, 2.5f, 0, 0.3f, 2.5f, 3, 0.3f, 2.5f }),
        drie::line_dashed_material(color: 0xffff00, dash_size: 0.25f, gap_size: 0.25f)
    );
    dashed.compute_line_distances();
    scene.add(dashed);
}

fn bool is_yellow(char[] px, usz i) => px[i] > 170 && px[i + 1] > 170 && px[i + 2] < 90;

fn int count_yellow(char[] px)
{
    int n;
    for (usz i = 0; i < (usz)W * H * 4; i += 4) if (is_yellow(px, i)) n++;
    return n;
}

fn void render_into(RenderTarget* target, char[] px)
{
    renderer.set_render_target(target);
    // a few frames: the shadow map and the lines' index buffers are made on the first
    for (int f = 0; f < 3; f++)
    {
        renderer.render(scene, camera);
        drie::commit();
    }
    readback::read_image(target.texture_image, W, H, px.ptr);
}

fn int main(String[] args)
{
    String out_path = args.len > 1 ? args[1] : "tests/build/lines.png";
    renderer = drie::renderer(antialias: true, width: W, height: H);
    renderer.init_headless();
    build_scene();
    camera = drie::perspective_camera(50, (float)W / H, 0.1f, 100);
    camera.position = { 0, 4, 8 };
    camera.look_at(0, 0.5f, 0);
    RenderTarget* target = drie::render_target(W, H, samples: 4);

    char[] with_lines = mem::new_array(char, (usz)W * H * 4);
    char[] without = mem::new_array(char, (usz)W * H * 4);
    render_into(target, with_lines);
    int segments = (GRID + 1) * 2 + (WAVE - 1) + CIRCLE + 2;
    expect(renderer.info.lines == segments, string::tformat("info.lines: %d segments (want %d)", renderer.info.lines, segments));
    expect(renderer.info.shadow_calls > 0, "the box casts a shadow");

    // a ray straight down through the grid's center line hits the floor, not the lines
    Raycaster ray = drie::raycaster({ 0, 3, 0.001f }, { 0, -1, 0 });
    List{Intersection} hits;
    hits.tinit();
    ray.intersect_object(scene, &hits);
    bool through = hits.len() > 0;
    foreach (h : hits) if (((Mesh*)h.object).is_line()) through = false;
    expect(through, "a ray passes through the lines to the floor");

    // the same line solid: twice the pixels, about
    char[] solid = mem::new_array(char, (usz)W * H * 4);
    dashed.material.gap_size = 0;
    render_into(target, solid);
    dashed.material.gap_size = 0.25f;
    int dashes = count_yellow(with_lines);
    int whole = count_yellow(solid);
    float part = (float)dashes / math::max(whole, 1);
    expect(whole > 150 && part > 0.35f && part < 0.65f,
        string::tformat("the dashed line covers %d of its %d pixels (%.2f)", dashes, whole, part));

    Object3D* copy = grid.clone();
    expect(((Mesh*)copy).draw_mode == LINE_SEGMENTS, "a clone of a LineSegments is one too");

    grid.visible = false;
    wave.visible = false;
    circle.visible = false;
    dashed.visible = false;
    render_into(target, without);
    expect(renderer.info.lines == 0, "hidden lines are not drawn");

    // the lines change pixels, and the wave's are red
    int changed;
    int red;
    for (usz i = 0; i < (usz)W * H * 4; i += 4)
    {
        int d = math::abs((int)with_lines[i] - (int)without[i]) + math::abs((int)with_lines[i + 1] - (int)without[i + 1])
            + math::abs((int)with_lines[i + 2] - (int)without[i + 2]);
        if (d > 30) changed++;
        if (with_lines[i] > 160 && with_lines[i + 1] < 90 && with_lines[i + 2] < 90) red++;
    }
    expect(changed > 1500, string::tformat("the lines change %d pixels", changed));
    expect(red > 100, string::tformat("%d pixels of the red line", red));

    char[] sheet = mem::new_array(char, (usz)W * 2 * H * 3);
    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            usz s = ((usz)y * W + (usz)x) * 4;
            usz d = ((usz)y * W * 2 + (usz)x) * 3;
            for (usz k = 0; k < 3; k++)
            {
                sheet[d + k] = with_lines[s + k];
                sheet[d + (usz)W * 3 + k] = without[s + k];
            }
        }
    }
    if (!png::write(out_path, sheet, W * 2, H, channels: 3))
    {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s: with lines, without", out_path);
    if (failures) io::printfn("%d FAILED", failures);
    return failures ? 1 : 0;
}
