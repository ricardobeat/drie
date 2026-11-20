// The diffuse light of an environment map (drie's env_sh_irradiance and the shader's
// env_irradiance) against the exact cosine integral of the same map, for surfaces facing up,
// down, towards the sun, away from it and sideways. The map: a sky (blue above the horizon, a
// dark ground below it) and a sun 10 degrees above the horizon, a few texels that give a
// surface facing it three times the light the sky does.
//   - each direction within 10% of the integral (and 0.02 where that is near nothing)
// No GPU: the coefficients are worked out on the CPU, as at a texture's upload.
//
//   just test envlight
module envlight_test;
import drie;
import std::io, std::math;

const int W = 256;
const int H = 128;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

// drie's longitude/latitude layout (see env_sh_irradiance)
fn Vector3 dir_of(int x, int y) {
    float phi   = ((float)y + 0.5f) / H * drie::PI;
    float theta = (((float)x + 0.5f) / W - 0.5f) * 2 * drie::PI;
    return { math::cos(theta) * math::sin(phi), math::cos(phi), math::sin(theta) * math::sin(phi) };
}

fn float omega_of(int y) => (2 * drie::PI / W) * (drie::PI / H) * math::sin(((float)y + 0.5f) / H * drie::PI);

fn int main(String[] args) {
    float[] px = mem::new_array(float, (usz)W * H * 4);
    Vector3 sun = { math::cos(10 * drie::DEG2RAD), math::sin(10 * drie::DEG2RAD), 0 };
    // the sky first, then the sun's texels: those within 1.5 degrees of it
    float sky_up;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            Vector3 d = dir_of(x, y);
            Vector3 c = d.y > 0 ? (Vector3) { 0.25f, 0.4f, 0.8f } * (0.6f + 0.4f * d.y) : (Vector3) { 0.06f, 0.05f, 0.04f };
            sz i = ((sz)y * W + x) * 4;
            px[i] = c.x;
            px[i + 1] = c.y;
            px[i + 2] = c.z;
            px[i + 3] = 1;
            if (d.y > 0) sky_up += c.z * d.y * omega_of(y);
        }
    }
    float sun_omega;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (dir_of(x, y).dot(sun) > math::cos(1.5f * drie::DEG2RAD)) sun_omega += omega_of(y);
        }
    }
    // radiance that gives a surface facing the sun 3 times the sky's light on one facing up
    float sun_l = 3 * sky_up / sun_omega;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (dir_of(x, y).dot(sun) <= math::cos(1.5f * drie::DEG2RAD)) continue;
            sz i = ((sz)y * W + x) * 4;
            px[i] += sun_l * 1.0f;
            px[i + 1] += sun_l * 0.7f;
            px[i + 2] += sun_l * 0.4f;
        }
    }

    drie::EnvLight env = drie::env_light(px, { }, W, H, false);
    Vector3 away = { -sun.x, 0, 0 };
    Vector3[6] normals = { { 0, 1, 0 }, { 0, -1, 0 }, sun, away, { 0, 0, 1 }, { 1, 0, 0 } };
    String[6] names = { "up", "down", "towards the sun", "away from the sun", "sideways", "+x" };
    foreach (k, n : normals) {
        Vector3 exact;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float c = dir_of(x, y).dot(n);
                if (c <= 0) continue;
                sz i = ((sz)y * W + x) * 4;
                exact += (Vector3) { px[i], px[i + 1], px[i + 2] } * (c * omega_of(y));
            }
        }
        Vector3 got = env.irradiance(n);
        bool ok = true;
        for (int c = 0; c < 3; c++) {
            float tolerance = math::max(exact[c] * 0.1f, 0.02f);
            if (math::abs(got[c] - exact[c]) > tolerance) ok = false;
        }
        expect(ok, string::tformat("%s: %.3f %.3f %.3f, exact %.3f %.3f %.3f", names[k], got.x, got.y, got.z,
                                   exact.x, exact.y, exact.z));
    }
    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}
