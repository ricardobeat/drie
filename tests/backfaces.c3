// Normal maps on the back faces of double-sided materials. Four mirror-like planes with a
// normal map of a bump, under a sky with a level horizon: on the top row their front faces,
// on the bottom row their back faces (turned around). In each row the left plane has
// tangents (glTF's TANGENT) and the right one none (its frame from the uv's derivatives):
// the two must look alike, front and back, as in three, where both frames are turned over
// on a back face (tbn *= faceDirection).
//
//   just test backfaces [out.png]   (tests/build/backfaces.png)
module backfaces_test;
import drie, drie::readback, drie::png, sokol;
import std::io, std::math;

const int W = 640;
const int H = 640;
const int N = 128;

// A sky to reflect: blue above a warm horizon, dark ground below, in linear light.
fn Texture* sky_map() {
    int w = 256;
    int h = 128;
    float[] px = mem::new_array(float, (usz)(w * h * 4));
    for (int y = 0; y < h; y++) {
        float up = 1 - 2 * ((float)y + 0.5f) / h;
        Vector3 c = up > 0 ? (Vector3) { 0.25f, 0.45f, 0.9f } * up + (Vector3) { 1.0f, 0.6f, 0.35f } * (1 - up)
            : (Vector3) { 0.12f, 0.1f, 0.08f };
        for (int x = 0; x < w; x++) {
            usz i = (usz)((y * w + x) * 4);
            px[i] = c.x;
            px[i + 1] = c.y;
            px[i + 2] = c.z;
            px[i + 3] = 1;
        }
    }
    return drie::float_texture(px, w, h);
}

// A normal map (OpenGL's, +y the way v grows) of a round bump off the middle, up and to the
// right, so that a frame turned the wrong way shows in both x and y.
fn Texture* bump_map() {
    char[] px = mem::new_array(char, N * N * 4);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            // u to the right; v up the plane: the first row is v = 0 (data textures aren't
            // flipped, and the plane's v grows up)
            float u = ((float)x + 0.5f) / N * 2 - 1.3f;
            float v = ((float)y + 0.5f) / N * 2 - 1.3f;
            float r2 = u * u + v * v;
            Vector3 n = { 0, 0, 1 };
            if (r2 < 0.36f) {
                float h = math::sqrt(0.36f - r2);
                n = (Vector3) { u, v, h }.normalize();
            }
            usz i = (usz)((y * N + x) * 4);
            px[i] = (char)((n.x * 0.5f + 0.5f) * 255);
            px[i + 1] = (char)((n.y * 0.5f + 0.5f) * 255);
            px[i + 2] = (char)((n.z * 0.5f + 0.5f) * 255);
            px[i + 3] = 255;
        }
    }
    return drie::data_texture(px, N, N, NO_COLOR_SPACE);
}

fn int main(String[] args) {
    String out_path = args.len > 1 ? args[1] : "tests/build/backfaces.png";
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    Scene* scene = drie::scene();
    Texture* sky = sky_map();
    scene.background_texture = sky;
    scene.environment = sky;
    MeshStandardMaterial* mat = drie::mesh_standard_material(color: 0xffffff, roughness: 0.15f, metalness: 1,
                                                             side: DOUBLE_SIDE);
    mat.normal_map = bump_map();
    for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 2; col++) {
            BufferGeometry* g = drie::plane_geometry(1.6f, 1.6f);
            // (u grows along +x and v along +y: the tangent is +x, the bitangent +y)
            if (col == 0) {
                for (uint i = 0; i < g.vertex_count(); i++) {
                    g.tangents.push(1);
                    g.tangents.push(0);
                    g.tangents.push(0);
                    g.tangents.push(1);
                }
            }
            Mesh* m = drie::mesh(g, mat);
            m.position.set(col == 0 ? -0.9f : 0.9f, row == 0 ? 0.9f : -0.9f, 0);
            // the bottom row turned around: its back faces toward the camera
            if (row == 1) m.rotation.y = (float)math::PI;
            scene.add(m);
        }
    }
    PerspectiveCamera* camera = drie::perspective_camera(40, (float)W / H, 0.1f, 100);
    camera.position.set(0, 0, 5.4f);
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
    io::printfn("saved %s: front faces (top), back faces (bottom); tangents left, none right", out_path);
    return 0;
}
