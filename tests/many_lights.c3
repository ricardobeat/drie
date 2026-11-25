// Many shadowed lights: a 24 m square room with pillars, crates and a hanging cloth, lit by
// 50 spot lights, 40 point lights and 30 rect area lights, every one casting shadows. Half of
// them draw their maps once (shadow.auto_update off), half every frame. Four boxes circle the
// room and the cloth waves (a dynamic geometry), so some maps see things move.
// Prints, per frame, what the shadow atlas holds and draws (RenderInfo.atlas_*), and checks:
//   - the picture is lit, and every light that asks for a shadow gets one
//   - no two maps overlap in the atlas, and from a still camera none moves
//   - a small atlas: the lights near the camera keep their maps, far ones lose theirs
//
//   just test many_lights [out.png] [frames]   (tests/build/many_lights.png)
module many_lights_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math, std::collections::list;

const int W = 320;
const int H = 180;
const float ROOM = 24;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
RenderTarget* rt;
char[] px;
Mesh*[4] movers;
BufferGeometry* cloth;
List{float} cloth_rest;
int lights;
List{Light*} all_lights;

fn void box(Material* m, Vector3 size, Vector3 at, bool cast = true) {
    Mesh* b = drie::mesh(drie::box_geometry(size.x, size.y, size.z), m);
    b.position = at;
    b.cast_shadow = cast;
    b.receive_shadow = true;
    scene.add(b);
}

fn void build() {
    scene = drie::scene();
    scene.background = 0x000000;
    Material* floor_m = drie::mesh_standard_material(color: 0x808080, roughness: 0.8f);
    Material* wall_m = drie::mesh_standard_material(color: 0xa0a0a0, roughness: 0.9f);
    Material* crate_m = drie::mesh_standard_material(color: 0x906040, roughness: 0.7f);
    Material* mover_m = drie::mesh_standard_material(color: 0x3060a0, roughness: 0.5f);
    float h = ROOM / 2;
    box(floor_m, { ROOM, 0.2f, ROOM }, { 0, -0.1f, 0 });
    box(wall_m, { ROOM, 4, 0.2f }, { 0, 2, -h });
    box(wall_m, { ROOM, 4, 0.2f }, { 0, 2, h });
    box(wall_m, { 0.2f, 4, ROOM }, { -h, 2, 0 });
    box(wall_m, { 0.2f, 4, ROOM }, { h, 2, 0 });
    // pillars and crates, still
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            float x = -10 + (float)i * 4;
            float z = -10 + (float)j * 4;
            if ((i + j) % 2 == 0) {
                box(wall_m, { 0.4f, 4, 0.4f }, { x, 2, z });
            } else {
                box(crate_m, { 0.6f, 0.6f, 0.6f }, { x + 0.5f, 0.3f, z - 0.3f });
            }
        }
    }
    // the boxes that circle the room
    foreach (i, &m : movers) {
        *m = drie::mesh(drie::box_geometry(0.5f, 1.2f, 0.5f), mover_m);
        (*m).cast_shadow = true;
        (*m).receive_shadow = true;
        scene.add(*m);
    }
    // the cloth: a dynamic plane hanging 1.5 m off the floor
    cloth = drie::plane_geometry(2, 2, 12, 12);
    cloth.dynamic = true;
    cloth_rest.init(mem);
    all_lights.init(mem);
    foreach (f : cloth.vertices) cloth_rest.push(f);
    Mesh* c = drie::mesh(cloth, drie::mesh_standard_material(color: 0x802020, roughness: 0.9f, side: DOUBLE_SIDE));
    c.position = { 3, 2.5f, 2 };
    c.cast_shadow = true;
    c.receive_shadow = true;
    scene.add(c);

    // the lights: spots in a grid on the ceiling, points between them, rects on the walls
    for (int i = 0; i < 50; i++) {
        float x = -11 + (float)(i % 10) * 2.4f;
        float z = -9 + (float)(i / 10) * 4.5f;
        SpotLight* s = drie::spot_light(0xffe0c0, 30, 8, 0.7f, 0.5f, 2);
        s.position = { x, 3.9f, z };
        Object3D* t = drie::object3d();
        t.position = { x + 0.5f, 0, z + 0.3f };
        scene.add(t);
        s.target = t;
        s.cast_shadow = true;
        s.shadow.map_size = i % 3 == 0 ? 1024 : 512;
        s.shadow.auto_update = i % 2 == 0;
        s.shadow.needs_update = true;
        scene.add(s);
        all_lights.push(s);
        lights++;
    }
    for (int i = 0; i < 40; i++) {
        float x = -10 + (float)(i % 8) * 2.8f;
        float z = -8 + (float)(i / 8) * 4;
        PointLight* p = drie::point_light(0xc0d0ff, 4, 5, 2);
        p.position = { x, 2.2f + (float)(i % 3) * 0.4f, z };
        p.cast_shadow = true;
        p.shadow.map_size = i % 4 == 0 ? 512 : 256;
        p.shadow.auto_update = i % 2 == 1;
        p.shadow.needs_update = true;
        scene.add(p);
        all_lights.push(p);
        lights++;
    }
    for (int i = 0; i < 30; i++) {
        // on the four walls, facing the middle
        int wall = i % 4;
        float along = -9 + (float)(i / 4) * 2.5f;
        Vector3 at;
        Vector3 into;
        switch (wall) {
            case 0: at = { along, 2.5f, -11.8f }; into = { 0, 0, 1 };
            case 1: at = { along, 2.5f, 11.8f }; into = { 0, 0, -1 };
            case 2: at = { -11.8f, 2.5f, along }; into = { 1, 0, 0 };
            default: at = { 11.8f, 2.5f, along }; into = { -1, 0, 0 };
        }
        RectAreaLight* r = drie::rect_area_light(0xffd0a0, 8, 0.8f, 0.4f);
        r.position = at;
        r.look_at_vec(at + into);
        r.cast_shadow = true;
        r.shadow.map_size = 512;
        r.shadow.auto_update = i % 2 == 0;
        r.shadow.needs_update = true;
        scene.add(r);
        all_lights.push(r);
        lights++;
    }
    scene.add(drie::ambient_light(0xffffff, 0.02f));
}

// Moves the boxes and waves the cloth for frame `f`.
fn void animate(int f) {
    float t = (float)f * 0.05f;
    foreach (i, m : movers) {
        float a = t + (float)i * (float)math::PI / 2;
        m.position = { math::cos(a) * 6, 0.6f, math::sin(a) * 6 };
    }
    for (sz v = 0; v < cloth.vertex_count(); v++) {
        sz o = v * drie::FLOATS_PER_VERTEX;
        float y = cloth_rest[o + 1];
        cloth.vertices[o + 2] = cloth_rest[o + 2] + math::sin(t * 3 + y * 2) * 0.2f;
    }
    cloth.vertices_changed();
}

fn void frame(int f) {
    animate(f);
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
}

fn void print_info(int f) {
    RenderInfo* i = &renderer.info;
    io::printfn("frame %3d: atlas %dx%d %.1f MB, %d lights (%d dropped), %d maps drawn, %d caster draws, %d shadow calls",
                f, i.atlas_width, i.atlas_height, (double)i.atlas_bytes / (1024.0 * 1024.0), i.atlas_lights, i.atlas_dropped,
                i.atlas_tiles, i.atlas_casters, i.shadow_calls);
}

fn float mean_brightness() {
    readback::read_image(rt.texture_image, W, H, px.ptr);
    double sum;
    for (usz i = 0; i < (usz)W * H; i++) sum += ((double)px[i * 4] + px[i * 4 + 1] + px[i * 4 + 2]) / 3;
    return (float)(sum / ((double)W * H));
}

fn int main(String[] args) {
    String out_path = "tests/build/many_lights.png";
    int frames = 40;
    if (args.len > 1) out_path = args[1];
    if (args.len > 2) frames = args[2].to_int()!!;
    renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    renderer.max_rect_area_lights = 64;
    build();
    camera = drie::perspective_camera(60, (float)W / H, 0.1f, 60);
    camera.position = { -10, 3, 10 };
    camera.look_at_vec({ 2, 0.5f, -2 });
    rt = drie::render_target(W, H, 1, RGBA8);
    px = mem::new_array(char, (usz)W * H * 4);

    // let every program compile: the waiting draws keep a drawn-once map asking again
    frame(0);
    for (int n = 0; n < 3000 && renderer.info.waiting > 0; n++) frame(0);
    for (int f = 1; f <= frames; f++) {
        frame(f);
        print_info(f);
    }
    // the maps: none overlaps another, and from a still camera none moves
    List{int[4]} rects;
    rects.init(mem);
    foreach (l : all_lights) {
        for (int f = 0; f < 6; f++) rects.push(renderer.shadow_map_rect(l, f));
    }
    long used;
    bool overlap;
    foreach (i, a : rects) {
        if (!a[2]) continue;
        used += (long)a[2] * a[3];
        for (sz j = (sz)i + 1; j < rects.len(); j++) {
            int[4] b = rects[j];
            if (!b[2]) continue;
            if (a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]) overlap = true;
        }
    }
    io::printfn("maps: %.1f%% of the atlas", 100.0 * (double)used / ((double)renderer.info.atlas_width * renderer.info.atlas_height));
    expect(!overlap, "no two maps overlap");
    // the camera still: the boxes and the cloth go on moving, the maps stay where they are
    for (int f = frames + 1; f <= frames + 40; f++) frame(f);
    bool same = true;
    foreach (i, l : all_lights) {
        for (int f = 0; f < 6; f++) {
            if (renderer.shadow_map_rect(l, f) != rects[(usz)i * 6 + (usz)f]) same = false;
        }
    }
    expect(same, "from a still camera, every map stays where it is");

    float lit = mean_brightness();
    expect(lit > 8, string::tformat("the room is lit (mean %.1f)", lit));

    expect(renderer.info.atlas_dropped == 0, string::tformat("every shadowed light has a map (%d of %d without)", renderer.info.atlas_dropped, lights));

    // a small atlas: the lights nearest the camera keep their maps, the farthest lose theirs
    renderer.shadow_atlas_size = 2048;
    for (int f = 0; f < 30; f++) frame(frames + 40 + f);
    print_info(frames + 70);
    Vector3 eye = camera.position;
    int near_kept;
    int near_count;
    int far_lost;
    int far_count;
    foreach (l : all_lights) {
        float d = (l.matrix_world.get_position() - eye).length();
        bool has = renderer.shadow_map_rect(l, 0)[2] > 0;
        if (d < 5) {
            near_count++;
            if (has) near_kept++;
        }
        if (d > 25) {
            far_count++;
            if (!has) far_lost++;
        }
    }
    expect(renderer.info.atlas_dropped > 0 && near_kept == near_count && far_lost > 0,
           string::tformat("a 2048 atlas: %d lights without maps; the %d near the camera keep theirs (%d), %d of the %d far ones lose theirs",
                           renderer.info.atlas_dropped, near_count, near_kept, far_lost, far_count));
    renderer.shadow_atlas_size = 8192;

    if (!png::write(out_path, px, W, H)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
