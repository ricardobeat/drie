// drie::cloth, without the GPU:
//   - a sheet pinned along its top settles: no point moves after ten seconds, none drifts,
//     no link stretches past a few percent
//   - the same run twice gives the same points, bit for bit
//   - two panels hung side by side (a parted curtain): a capsule (a person) walks through the
//     gap, pushes them aside without going into them, and they fall back to where they hung
//   - box and plane colliders: a sheet pressed on a wall stays in front of it, one falling on a
//     slope stays on it, and a door swinging through a sheet pushes it without any of it
//     going into or through the door
//   - the cost of one curtain's update
//
//   just test cloth
module cloth_test;
import drie, drie::cloth;
import std::io, std::math, std::time::clock;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn Cloth* curtain(float x0, float width, int cols) {
    Cloth* c = cloth::grid(cols, 36, { x0, 2.4, 0 }, { width, 0, 0 }, { 0, -2.3, 0 }, mass: 0.6);
    c.pin_row(0);
    return c;
}

fn void run(Cloth* c, float seconds) {
    int n = (int)(seconds * 60 + 0.5f);
    for (int i = 0; i < n; i++) c.update(1.0f / 60);
}

fn bool finite(Cloth* c) {
    foreach (p : c.pos) {
        if (!math::is_finite(p.x) || !math::is_finite(p.y) || !math::is_finite(p.z)) return false;
    }
    return true;
}

fn float max_strain(Cloth* c) {
    float m = 0;
    foreach (l : c.stretch) m = math::max(m, ((c.pos[l.a] - c.pos[l.b]).length() - l.rest) / l.rest);
    return m;
}

fn float max_distance(Vector3[] a, Vector3[] b) {
    float m = 0;
    foreach (i, p : a) m = math::max(m, (p - b[i]).length());
    return m;
}

fn Vector3[] copy(Vector3[] a) {
    Vector3[] b = mem::new_array(Vector3, a.len);
    b[..] = a[..];
    return b;
}

// how far the deepest point is inside the capsule a..b
fn float penetration(Cloth* c, Vector3 a, Vector3 b, float r) {
    float worst = 0;
    Vector3 ab  = b - a;
    foreach (p : c.pos) {
        float t = math::clamp((p - a).dot(ab) / ab.dot(ab), 0.0f, 1.0f);
        float d = (p - (a + ab * t)).length();
        worst = math::max(worst, r - d);
    }
    return worst;
}

fn int main(String[] args) {
    // settling
    Cloth* sheet = curtain(-0.6, 1.2, 24);
    run(sheet, 10);
    Vector3[] at10 = copy(sheet.pos);
    run(sheet, 5);
    expect(finite(sheet), "settled sheet: every point is a number");
    expect(
        sheet.max_speed() < 2e-3f,
        string::tformat("settled sheet: still after 10 s (fastest point %.5f m/s)", sheet.max_speed())
    );
    float drift = max_distance(at10, sheet.pos);
    expect(drift < 1e-3f, string::tformat("settled sheet: no drift from 10 s to 15 s (%.6f m)", drift));
    expect(
        max_strain(sheet) < 0.03f,
        string::tformat("settled sheet: links stretched %.2f%% at most", max_strain(sheet) * 100)
    );
    float bottom = sheet.pos[sheet.index(12, 35)].y;
    expect(
        bottom < 0.12f && bottom > 0.0f,
        string::tformat("settled sheet: hangs straight down (bottom at %.3f m)", bottom)
    );

    // determinism, with wind
    Cloth* w1 = curtain(-0.6, 1.2, 24);
    Cloth* w2 = curtain(-0.6, 1.2, 24);
    w1.wind = w2.wind = { 0.2, 0, 0.6 };
    w1.gust = w2.gust = 0.8;
    run(w1, 4);
    run(w2, 4);
    bool same = true;
    foreach (i, p : w1.pos) same = same && p == w2.pos[i];
    float swayed = max_distance(w1.pos, sheet.pos);
    expect(same, "two runs give the same points");
    expect(swayed > 0.02f && finite(w1), string::tformat("the draft moves the sheet (%.3f m)", swayed));

    // a person through a parted curtain
    Cloth* left  = curtain(-0.62, 0.6, 14);
    Cloth* right = curtain(0.02, 0.6, 14);
    run(left, 8);
    run(right, 8);
    Vector3[] left_rest  = copy(left.pos);
    Vector3[] right_rest = copy(right.pos);
    float r              = 0.22;
    Vector3 lo           = { 0, 0.3, -0.8 };
    Vector3 hi           = { 0, 1.7, -0.8 };
    int lid              = left.add_capsule(lo, hi, r);
    int rid              = right.add_capsule(lo, hi, r);
    float deepest        = 0;
    float parted         = 0;
    // walking at 1.2 m/s from z = -0.8 to 1.2
    for (int f = 0; f <= 100; f++) {
        float z   = -0.8f + 2.0f * f / 100;
        Vector3 a =  { 0, 0.3, z };
        Vector3 b =  { 0, 1.7, z };
        left.move_collider(lid, a, b);
        right.move_collider(rid, a, b);
        left.update(1.0f / 60);
        right.update(1.0f / 60);
        deepest = math::max(deepest, math::max(penetration(left, a, b, r), penetration(right, a, b, r)));
        // how far the inner edges at chest height were pushed apart
        float gap = right.pos[right.index(0, 18)].x - left.pos[left.index(13, 18)].x;
        parted = math::max(parted, gap);
    }
    expect(finite(left) && finite(right), "parted curtain: every point is a number");
    expect(
        deepest < 0.01f,
        string::tformat("parted curtain: nothing deeper than 1 cm in the capsule (%.4f m)", deepest)
    );
    expect(
        parted > 2 * r,
        string::tformat("parted curtain: the gap opened to %.3f m (the capsule is %.2f m wide)", parted, 2 * r)
    );
    left.set_collider_active(lid, false);
    right.set_collider_active(rid, false);
    run(left, 12);
    run(right, 12);
    float back = math::max(max_distance(left.pos, left_rest), max_distance(right.pos, right_rest));
    expect(back < 0.01f, string::tformat("parted curtain: back where it hung 12 s later (%.4f m off)", back));

    // a body put somewhere else (further than cloth::JUMP) is not swept through the cloth
    {
        Cloth* t = curtain(-0.6, 1.2, 24);
        run(t, 6);
        Vector3[] before = copy(t.pos);
        int id = t.add_capsule({ 0, 0.3, -1.5 }, { 0, 1.7, -1.5 }, 0.22);
        t.update(1.0f / 60);
        t.move_collider(id, { 0, 0.3, 1.5 }, { 0, 1.7, 1.5 });
        t.update(1.0f / 60);
        float moved = max_distance(before, t.pos);
        expect(moved < 0.005f, string::tformat("a body moved 3 m in a frame jumps there, not through the sheet (%.4f m)", moved));
    }

    // a gale blows a sheet against a wall (a box) behind it: the cloth stays in front of its face
    {
        Cloth* t = curtain(-0.6, 1.2, 24);
        t.floor_y = 0;
        run(t, 6);
        t.add_box({ 2, 1.5, 0.1 }, drie::mat4_compose({ 0, 1.5, -0.45 }, { 0, 0, 0 }, { 1, 1, 1 }));
        t.wind = { 0, 0, -8 };
        t.gust = 0.5;
        float worst;
        float reached = 1;
        for (int f = 0; f < 240; f++) {
            t.update(1.0f / 60);
            foreach (p : t.pos) {
                worst = math::max(worst, -0.35f - p.z);
                reached = math::min(reached, p.z);
            }
        }
        expect(finite(t) && reached < -0.3f && worst < 0.005f,
               string::tformat("box: a sheet blown onto a wall stays in front of it (down to z %.3f; %.4f m in at most)", reached, worst));
    }

    // a plane, tilted: the points stay on its side
    {
        Cloth* t = curtain(-0.6, 1.2, 24);
        Vector3 nrm = { 0, 0.8, 0.6 };
        t.add_plane({ 0, 0.8, 0 }, nrm);
        run(t, 4);
        float under;
        foreach (p : t.pos) under = math::max(under, -(p - (Vector3) { 0, 0.8, 0 }).dot(nrm.normalize()));
        expect(finite(t) && under < 0.001f, string::tformat("plane: a sheet falls onto a slope and stays on it (%.4f m under)", under));
    }

    // a door (a thin box on a hinge) swings through a hanging sheet at 2.4 m/s at its edge, and
    // back: no point goes into it, and none crosses from one face to the other within it
    {
        Cloth* t = curtain(-0.6, 1.2, 24);
        t.floor_y = 0;
        run(t, 6);
        Vector3 hinge = { 0, 0.9, 0.6 };
        Vector3 half = { 0.45, 0.9, 0.022 };
        int door;
        int inside;
        int through;
        float worst;
        float swept;
        // per point: its side of the leaf when last within its span (0: not)
        float[] side = mem::new_array(float, t.pos.len);
        defer free(side);
        Vector3[] start = copy(t.pos);
        defer free(start);
        for (int f = 0; f < 216; f++) {
            // open (0 to 180 degrees) in 1.2 s, held 0.6 s, shut in 1.2 s
            float k = f < 72 ? (float)f / 72 : f < 108 ? 1 : 1 - (float)(f - 108) / 72;
            float a = k * (float)math::PI;
            Vector3 d = { math::cos(a), 0, -math::sin(a) };
            drie::Mat4 place = drie::mat4_compose(hinge + d * half.x, { 0, a, 0 }, { 1, 1, 1 });
            if (f == 0) {
                door = t.add_box(half, place);
            } else {
                t.move_box(door, place);
            }
            t.update(1.0f / 60);
            // the leaf's frame: x along it, z out of its face
            Vector3 c = hinge + d * half.x;
            Vector3 az = { math::sin(a), 0, math::cos(a) };
            foreach (i, p : t.pos) {
                if (t.inv_mass[i] == 0) continue;
                swept = math::max(swept, (p - start[i]).length());
                Vector3 rel = p - c;
                float lx = rel.dot(d);
                float ly = rel.y;
                float lz = rel.dot(az);
                if (math::abs(lx) > half.x - 0.02f || math::abs(ly) > half.y - 0.02f) {
                    side[i] = 0;
                    continue;
                }
                float in = half.z + t.thickness - math::abs(lz);
                if (in > 0.005f) {
                    inside++;
                    worst = math::max(worst, in);
                }
                float now = lz < 0 ? -1 : 1;
                if (side[i] != 0 && side[i] != now) through++;
                side[i] = now;
            }
        }
        expect(finite(t) && swept > 0.3f && inside == 0 && through == 0,
               string::tformat("box: a swinging door pushes the sheet (%.2f m), none of it in the door (%d, %.4f m) or through it (%d)", swept, inside, worst, through));
    }

    // a fine pleated grid over a coarse cloth: drawn where its rest shape is, and carried along
    // when the cloth turns and moves as a whole
    {
        int fc = 49;
        int fr = 37;
        Vector3[] fine = mem::new_array(Vector3, (sz)fc * fr);
        for (int rr = 0; rr < fr; rr++) {
            for (int cc = 0; cc < fc; cc++) {
                float u = (float)cc / (float)(fc - 1);
                fine[rr * fc + cc] = { u, 2 - (float)rr / (float)(fr - 1) * 2, 0.03f * math::sin(u * 12 * (float)math::PI) };
            }
        }
        Cloth* d = cloth::from_fine(fc, fr, fine, 3, 2);
        expect(d.cols == 17 && d.rows == 19, string::tformat("from_fine: a %d x %d cloth under 49 x 37", d.cols, d.rows));
        Mesh* dm = d.make_mesh();
        float off = 0;
        for (int i = 0; i < fc * fr; i++) {
            float* v = &dm.geometry.vertices[(sz)i * drie::FLOATS_PER_VERTEX];
            off = math::max(off, ((Vector3) { v[0], v[1], v[2] } - fine[i]).length());
        }
        expect(off < 1e-5f, string::tformat("from_fine: drawn at its rest shape (%.7f m off)", off));
        Vector3[] normals = mem::new_array(Vector3, (sz)fc * fr);
        for (int i = 0; i < fc * fr; i++) {
            float* v = &dm.geometry.vertices[(sz)i * drie::FLOATS_PER_VERTEX];
            normals[i] = { v[3], v[4], v[5] };
        }
        // turned a quarter about y and moved
        Vector3 move = { 3, 1, -2 };
        foreach (i, &p : d.pos) {
            *p = (Vector3) { p.z, p.y, -p.x } + move;
            d.start[i] = *p;
        }
        d.write_mesh(1);
        off = 0;
        for (int i = 0; i < fc * fr; i++) {
            float* v = &dm.geometry.vertices[(sz)i * drie::FLOATS_PER_VERTEX];
            Vector3 want = (Vector3) { fine[i].z, fine[i].y, -fine[i].x } + move;
            off = math::max(off, ((Vector3) { v[0], v[1], v[2] } - want).length());
        }
        float worst = 1;
        for (int i = 0; i < fc * fr; i++) {
            float* v = &dm.geometry.vertices[(sz)i * drie::FLOATS_PER_VERTEX];
            Vector3 want = { normals[i].z, normals[i].y, -normals[i].x };
            worst = math::min(worst, want.dot({ v[3], v[4], v[5] }));
        }
        expect(off < 1e-4f && worst > 0.9999f,
               string::tformat("from_fine: turned and moved with the cloth (%.7f m off; normals within %.5f)", off, 1 - worst));
    }

    // cost: one curtain in a draft with a capsule near it
    Cloth* c = curtain(-0.6, 1.2, 24);
    c.wind = { 0.2, 0, 0.6 };
    c.gust = 0.8;
    c.make_mesh();
    int body = c.add_capsule({ 0, 0.3, -0.3 }, { 0, 1.7, -0.3 }, r);
    run(c, 1);
    std::time::Clock t0 = clock::now();
    int frames          = 600;
    for (int f = 0; f < frames; f++) {
        float z = -0.3f + 0.4f * math::sin(f * 0.05f);
        c.move_collider(body, { 0, 0.3, z }, { 0, 1.7, z });
        c.update(1.0f / 60);
    }
    double ms = (double)t0.to_now() / 1e6 / frames;
    io::printfn("cost: %.3f ms per frame for %d x %d points, %d substeps", ms, c.cols, c.rows, c.substeps);

    if (failures) io::printfn("%d failed", failures);
    return failures ? 1 : 0;
}