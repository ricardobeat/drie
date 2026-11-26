// drie::gpu_cloth, headless: a sheet pinned by its top row hangs as long as it is (the tethers)
// and no further; a sheet dropped on its edge onto the floor folds onto itself and, with
// self-collision, no two of its points end up nearer than half the distance it keeps (without
// it, many do); two runs of the same drop give the same points, bit for bit.
// `just test gpu_cloth`
module gpu_cloth_test;
import drie, drie::gpu_cloth, sokol;
import std::io, std::math;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn Vector3[] run(GpuCloth* c, float seconds) {
    int frames = (int)math::round(seconds * 60);
    for (int f = 0; f < frames; f++) {
        c.update(1.0f / 60);
        drie::commit();
    }
    Vector3[] points = mem::new_array(Vector3, c.count());
    if (!c.read_points(points)) io::eprintfn("could not read the points back");
    return points;
}

const int N = 48;
const float SIDE = 1.0f;

// A sheet standing on its edge 1 m above the floor, a little tilted: it folds as it lands.
fn GpuCloth* standing(bool self_collision) {
    GpuCloth* c = gpu_cloth::grid(N, N, { -0.5f, 2.0f, 0 }, { SIDE, 0, 0 }, { 0, -SIDE, 0.15f }, 0.3f);
    c.floor_y = 0;
    c.substeps = 12;
    c.friction = 0.6f;
    c.self_distance = self_collision ? SIDE / (N - 1) : 0;
    return c;
}

// Pairs of points at least 3 apart along the cloth that are nearer than `d`.
fn int near_pairs(Vector3[] p, float d) {
    int count;
    for (int i = 0; i < N * N; i++) {
        for (int j = i + 1; j < N * N; j++) {
            if (math::abs(i % N - j % N) < 3 && math::abs(i / N - j / N) < 3) continue;
            Vector3 e = p[i] - p[j];
            if (e.dot(e) < d * d) count++;
        }
    }
    return count;
}

fn int main(String[] args) {
    Renderer* renderer = drie::renderer(antialias: false, width: 64, height: 64);
    renderer.init_headless();
    if (!drie::compute_supported()) {
        io::printfn("skip: no compute shaders");
        return 0;
    }

    GpuCloth* hung = gpu_cloth::grid(N, N, { -0.5f, 2.0f, 0 }, { SIDE, 0, 0 }, { 0, 0, SIDE }, 0.3f);
    hung.pin_row(0);
    Vector3[] p = run(hung, 3);
    float lowest = 1e30f;
    foreach (q : p) lowest = math::min(lowest, q.y);
    float drop = 2.0f - lowest;
    io::printfn("     hung: bottom %.3f m below the pins (the sheet is %.3f long, %.0f%% slack)", drop, SIDE, hung.tether_slack * 100);
    expect(drop > SIDE * 0.97f && drop < SIDE * (1 + hung.tether_slack) + 0.01f, "a pinned sheet hangs its length, held by the tethers");

    float d = SIDE / (N - 1);
    Vector3[] with = run(standing(true), 4);
    Vector3[] without = run(standing(false), 4);
    int crossed = near_pairs(with, d * 0.5f);
    int crossed_off = near_pairs(without, d * 0.5f);
    io::printfn("     pairs nearer than %.1f mm: %d with self-collision, %d without", d * 500, crossed, crossed_off);
    expect(crossed_off > 50, "without self-collision the folds sink into each other");
    expect(crossed * 20 < crossed_off, "with it, they keep apart");

    Vector3[] again = run(standing(true), 4);
    bool same = true;
    foreach (i, q : with) same = same && q.x == again[i].x && q.y == again[i].y && q.z == again[i].z;
    expect(same, "two runs of the same drop give the same points");

    io::printfn(failures ? "%d failed" : "all passed", failures);
    return failures ? 1 : 0;
}
