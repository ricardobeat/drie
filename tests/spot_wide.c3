// A wide spot light (angle 1.4 rad, a 160 degree cone) on a wall, as the red room's
// safelights: the bulb 3.4 cm off the wall, aimed down and out, the wall
// lit at a grazing angle. The wall must be as bright with a soft shadow (shadow.softness > 0,
// PCSS) as with a hard one (softness 0): nothing stands between it and the light. It went
// dark with softness: the wide blur ran the wall into its own shadow (the receiver plane's
// depth bias was clamped too tight for a surface the light grazes, and the search for
// blockers took the wall for one).
//   - the wall 0.3, 0.6 and 0.9 m under the light: soft no more than 5% darker than hard (it may lose the hard one's acne)
// The picture: hard, soft, left to right (the wall, from in front).
//
//   just test spot_wide [out.png]   (tests/build/spot_wide.png)
module spot_wide_test;
import drie, sokol, drie::readback, drie::png;
import std::io, std::math;

const int S = 512;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

// the mean red over a 5x5 patch at world (0, y) on the wall; the camera sees 3 m across,
// its middle 1.5 m up
fn float wall_at(char[] px, float y) {
    int cx = S / 2;
    int cy = (int)((0.5f - (y - 1.5f) / 3) * S);
    float sum;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) sum += px[((usz)(cy + dy) * S + (usz)(cx + dx)) * 4];
    }
    return sum / 25;
}

fn int main(String[] args) {
    String out_path = "tests/build/spot_wide.png";
    if (args.len > 1) out_path = args[1];
    Renderer* renderer = drie::renderer(antialias: false, width: S, height: S);
    renderer.init_headless();
    renderer.tone_mapping = NO_TONE_MAPPING;

    Scene* scene = drie::scene();
    scene.background = 0x000000;
    MeshStandardMaterial* white = drie::mesh_standard_material(color: 0xffffff, roughness: 1, metalness: 0);
    Mesh* ground = drie::mesh(drie::plane_geometry(12, 12), white);
    ground.rotation = { -(float)math::PI / 2, 0, 0 };
    ground.receive_shadow = true;
    scene.add(ground);
    Mesh* wall = drie::mesh(drie::box_geometry(6, 3, 0.1f), white);
    wall.position = { 0, 1.5f, -0.034f - 0.05f };
    wall.cast_shadow = true;
    wall.receive_shadow = true;
    scene.add(wall);
    SpotLight* spot = drie::spot_light(0xffffff, 3, 6.5f, 1.4f, 1.0f, 2);
    spot.position = { 0, 1.5f, 0 };
    spot.target.position = { 0, 0.22f, 0.45f };
    spot.radius = 0.07f;
    spot.cast_shadow = true;
    // (a map drawn once, in the shadow atlas, as the red room's)
    spot.shadow.auto_update = false;
    spot.shadow.map_size = 1024;
    spot.shadow.radius = 2;
    scene.add(spot);
    scene.add(spot.target);

    // (3 m across at the wall)
    PerspectiveCamera* camera = drie::perspective_camera(2 * math::atan(1.5f / 4.0f) * drie::RAD2DEG, 1, 0.1f, 20);
    camera.position = { 0, 1.5f, 4 };
    RenderTarget* rt = drie::render_target(S, S, 1, RGBA8);
    renderer.set_render_target(rt);

    char[][2] px;
    for (int i = 0; i < 2; i++) {
        spot.shadow.softness = i == 0 ? 0 : 0.12f;
        spot.shadow.needs_update = true;
        px[i] = mem::new_array(char, S * S * 4);
        // (twice: the second with the map made)
        renderer.render(scene, camera);
        renderer.render(scene, camera);
        drie::commit();
        readback::read_image(rt.texture_image, S, S, px[i].ptr);
    }
    float[3] downs = { 0.3f, 0.6f, 0.9f };
    foreach (down : downs) {
        float hard = wall_at(px[0], 1.5f - down);
        float soft = wall_at(px[1], 1.5f - down);
        expect(hard > 10 && soft >= hard * 0.95f, string::tformat("%.1f m under: hard %.0f, soft %.0f", down, hard, soft));
    }

    char[] sheet = mem::new_array(char, S * 2 * S * 4);
    for (int y = 0; y < S; y++) {
        for (int x = 0; x < S; x++) {
            usz src = ((usz)y * S + (usz)x) * 4;
            for (int k = 0; k < 2; k++) {
                usz dst = ((usz)y * S * 2 + (usz)(k * S + x)) * 4;
                sheet[dst:3] = px[k][src:3];
                sheet[dst + 3] = 255;
            }
        }
    }
    (void)png::write(out_path, sheet, S * 2, S);
    io::printfn("%s", out_path);
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
