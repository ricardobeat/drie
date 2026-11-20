// Point light shadows (three's PointLightShadow): a 6 x 6 x 6 m room of white walls lit by
// one point light in its middle, with a small cube 1.2 m from the light along each of the six
// axes. Each cube's shadow falls on the middle of the wall behind it, one shadow on each face
// of the light's cube of maps.
//   - six views, each from 0.5 m in front of a wall towards it: the middle of the view (the
//     shadow) is dark, a point near the view's edge (outside the shadow) is lit
//   - the same with shadow.auto_update off (the maps drawn once, kept in the atlas): the same
//   - cast_shadow off: the middle as lit as the edge
//   - with renderer.shadow_type VSM: as the first
//   - a shadow across two faces' maps: the light 1.5 m under the middle, a cube on its
//     (1, 1, 0) diagonal, whose shadow on the +x wall lies across the line (y = 1.5) where the
//     +x face's map ends and the +y face's begins: dark just above it and just below it
// The picture: the six views with shadows, then the six without, then the two faces'.
//
//   just test pointshadows [out.png]   (tests/build/pointshadows.png)
module pointshadows_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int S = 128;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

Vector3[6] axes = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };

fn void add_wall(Scene* scene, Material* m, Vector3 axis) {
    // a plane faces +z: turned to face the middle of the room from 3 m along `axis`
    Mesh* wall = drie::mesh(drie::plane_geometry(6, 6), m);
    wall.position = axis * 3;
    if (math::abs(axis.y) > 0.5f) wall.up = { 0, 0, 1 };
    wall.look_at_vec({ 0, 0, 0 });
    wall.receive_shadow = true;
    scene.add(wall);
}

fn int main(String[] args) {
    String out_path = "tests/build/pointshadows.png";
    if (args.len > 1) out_path = args[1];
    Renderer* renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    Scene* scene = drie::scene();
    scene.background = 0x000000;
    Material* white = drie::mesh_standard_material(color: 0xffffff, roughness: 1);
    foreach (a : axes) add_wall(scene, white, a);
    Mesh*[6] cubes;
    foreach (i, a : axes) {
        Mesh* cube = drie::mesh(drie::box_geometry(0.2f, 0.2f, 0.2f), white);
        cube.position = a * 1.2f;
        cube.cast_shadow = true;
        scene.add(cube);
        cubes[i] = cube;
    }
    PointLight* light = drie::point_light(0xffffff, 20, 0, 2);
    light.cast_shadow = true;
    light.shadow.map_size = 512;
    scene.add(light);
    scene.add(drie::ambient_light(0xffffff, 0.02f));

    PerspectiveCamera* camera = drie::perspective_camera(90, 1, 0.05f, 20);
    RenderTarget* rt = drie::render_target(S, S, 1, RGBA8);
    char[] px = mem::new_array(char, (usz)S * S * 4);
    char[] sheet = mem::new_array(char, (usz)S * 7 * S * 2 * 3);
    // middle, near the edge
    float[2][6] seen;
    for (int pass = 0; pass < 4; pass++) {
        light.cast_shadow = pass != 2;
        light.shadow.auto_update = pass != 1;
        renderer.shadow_type = pass == 3 ? VSM : PCF;
        light.shadow.needs_update = true;
        for (int i = 0; i < 6; i++) {
            Vector3 a = axes[i];
            camera.position = a * 2.5f;
            camera.up = math::abs(a.y) > 0.5f ? (Vector3) { 0, 0, 1 } : (Vector3) { 0, 1, 0 };
            camera.look_at_vec(a * 3);
            renderer.set_render_target(rt);
            renderer.render(scene, camera);
            drie::commit();
            readback::read_image(rt.texture_image, S, S, px.ptr);
            usz mid  = ((usz)(S / 2) * S + S / 2) * 4;
            usz edge = ((usz)(S / 2) * S + S * 19 / 20) * 4;
            float m  = ((float)px[mid] + px[mid + 1] + px[mid + 2]) / 3;
            float e  = ((float)px[edge] + px[edge + 1] + px[edge + 2]) / 3;
            String names = "+x-x+y-y+z-z";
            String name = names[i * 2:2];
            switch (pass) {
                case 0: expect(m < e * 0.4f, string::tformat("%s: the middle in the cube's shadow (%.0f), the edge lit (%.0f)", name, m, e));
                case 1: expect(m < e * 0.4f, string::tformat("%s, maps drawn once: the middle in shadow (%.0f), the edge lit (%.0f)", name, m, e));
                case 3: expect(m < e * 0.4f, string::tformat("%s, VSM: the middle in shadow (%.0f), the edge lit (%.0f)", name, m, e));
                default: expect(m > e * 0.9f, string::tformat("%s, no shadows: the middle (%.0f) as lit as the edge (%.0f)", name, m, e));
            }
            if (pass == 1 || pass == 3) continue;
            int row = pass == 0 ? 0 : 1;
            for (int y = 0; y < S; y++) {
                for (int x = 0; x < S; x++) {
                    usz s = ((usz)y * S + (usz)x) * 4;
                    usz d = ((usz)(row * S + y) * (usz)(S * 7) + (usz)(i * S + x)) * 3;
                    sheet[d:3] = px[s:3];
                }
            }
        }
    }

    renderer.shadow_type = PCF;
    foreach (c : cubes) c.visible = false;
    light.cast_shadow = true;
    light.shadow.auto_update = true;
    light.position = { 0, -1.5f, 0 };
    Vector3 diagonal = ((Vector3) { 1, 1, 0 }).normalize();
    Mesh* across = drie::mesh(drie::box_geometry(0.2f, 0.2f, 0.2f), white);
    across.position = light.position + diagonal * 1.2f;
    across.cast_shadow = true;
    scene.add(across);
    camera.position = { 2.5f, 1.5f, 0 };
    camera.up = { 0, 1, 0 };
    camera.look_at_vec({ 3, 1.5f, 0 });
    renderer.set_render_target(rt);
    renderer.render(scene, camera);
    drie::commit();
    readback::read_image(rt.texture_image, S, S, px.ptr);
    // 0.15 m above and below the line, in a shadow about 0.7 m across; the lit wall 0.45 m aside
    float[3] v;
    int[2][3] at = { { S / 2, S / 2 - S * 15 / 100 }, { S / 2, S / 2 + S * 15 / 100 }, { S * 19 / 20, S / 2 } };
    for (int k = 0; k < 3; k++) {
        usz o = ((usz)at[k][1] * S + (usz)at[k][0]) * 4;
        v[k] = ((float)px[o] + px[o + 1] + px[o + 2]) / 3;
    }
    expect(v[0] < v[2] * 0.4f && v[1] < v[2] * 0.4f,
           string::tformat("a shadow across the +x and +y faces' maps: dark above (%.0f) and below (%.0f) the line, the wall aside lit (%.0f)", v[0], v[1], v[2]));
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            usz s = ((usz)y * S + (usz)x) * 4;
            usz d = ((usz)y * (usz)(S * 7) + (usz)(6 * S + x)) * 3;
            sheet[d:3] = px[s:3];
        }
    }
    if (!png::write(out_path, sheet, S * 7, S * 2, channels: 3)) {
        io::printfn("could not write %s", out_path);
        return 1;
    }
    io::printfn("saved %s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
