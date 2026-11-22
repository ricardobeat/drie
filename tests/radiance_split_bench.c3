// A bench of the radiance cascades, headless: GPU ms of each of their passes (drie::gpu_timer),
// mean and best over the timed frames after a warm-up, and what the sparse probes are doing
// (rc_debug::stats). By default it times every kernel, then the frame again with each one left
// out (RadianceCascades.skip_kernels), so each kernel's share shows as the time saved without it.
//
//   just test radiance_split_bench [--dense] [--frames N] [--warmup N] [--skip k,k]
//
// --dense       the dense volume (radiance_cascades) in place of the sparse one
// --frames N    frames timed in each run (default 120)
// --warmup N    frames before them, not timed (default 60)
// --skip k,k    one run, with these kernels left out (clear, evict, alloc, propagate, trace,
//               gather); DRIE_RC_SKIP=k,k in the environment does the same
// --no-sweep    only the run with every kernel
module radiance_split_bench;
import drie, drie::rc_debug, drie::gpu_timer, sokol;
import std::io, std::math, std::os::env;

const int W = 1280;
const int H = 720;

struct Kernel {
    String name;
    int bit;
    bool sparse_only;
}

Kernel[6] kernels = {
    { "clear", drie::RC_KERNEL_CLEAR, true },
    { "evict", drie::RC_KERNEL_EVICT, true },
    { "alloc", drie::RC_KERNEL_ALLOC, true },
    { "propagate", drie::RC_KERNEL_PROPAGATE, true },
    { "trace", drie::RC_KERNEL_TRACE, false },
    { "gather", drie::RC_KERNEL_GATHER, false },
};

// the passes the cascades begin, by label
String[8] labels = {
    "drie-rc-clear", "drie-rc-evict", "drie-rc-alloc", "drie-rc-propagate",
    "drie-rc-split-trace", "drie-rc-split-gather", "drie-rc-trace", "drie-rc-gather",
};

struct Sample {
    double sum;
    double min;
    int n;
}

fn void Sample.add(&self, double ms) {
    if (!self.n || ms < self.min) self.min = ms;
    self.sum += ms;
    self.n++;
}

fn double Sample.mean(&self) => self.n ? self.sum / self.n : 0;

struct Result {
    Sample[8] pass;
    Sample rc;     // the cascades' passes
    Sample frame;  // every timed pass of the frame
}

Renderer* renderer;
Scene* scene;
PerspectiveCamera* cam;
RadianceCascades* rc;

// a long room: a floor, walls with a doorway, pillars, glowing panels and a lamp, so that the
// probes cover surfaces at several distances and the rays have things to hit
fn void build_scene(bool dense) {
    scene = drie::scene();
    scene.background = 0x080a10;
    MeshStandardMaterial* white = drie::mesh_standard_material(color: 0xd8d4cc, roughness: 0.9f);
    MeshStandardMaterial* wood = drie::mesh_standard_material(color: 0x7a4a2a, roughness: 0.7f);
    Mesh* floor = drie::mesh(drie::box_geometry(12, 0.2f, 20), wood);
    floor.position = { 0, -0.1f, 0 };
    scene.add(floor);
    Mesh* ceiling = drie::mesh(drie::box_geometry(12, 0.2f, 20), white);
    ceiling.position = { 0, 3.6f, 0 };
    scene.add(ceiling);
    float[2] xs = { -6, 6 };
    foreach (x : xs) {
        Mesh* wall = drie::mesh(drie::box_geometry(0.2f, 3.6f, 20), white);
        wall.position = { x, 1.8f, 0 };
        scene.add(wall);
    }
    Mesh* back = drie::mesh(drie::box_geometry(12, 3.6f, 0.2f), white);
    back.position = { 0, 1.8f, -10 };
    scene.add(back);
    // a cross wall with a doorway
    Mesh* cross_l = drie::mesh(drie::box_geometry(5, 3.6f, 0.2f), white);
    cross_l.position = { -3.5f, 1.8f, -2 };
    scene.add(cross_l);
    Mesh* cross_r = drie::mesh(drie::box_geometry(5, 3.6f, 0.2f), white);
    cross_r.position = { 3.5f, 1.8f, -2 };
    scene.add(cross_r);
    BufferGeometry* pillar = drie::box_geometry(0.4f, 3.6f, 0.4f);
    for (int i = 0; i < 4; i++) {
        float z = 6.0f - (float)i * 2.5f;
        float[2] px = { -3, 3 };
        foreach (x : px) {
            Mesh* p = drie::mesh(pillar, white);
            p.position = { x, 1.8f, z };
            scene.add(p);
        }
    }
    Color[3] glow = { 0xff3020, 0x30a0ff, 0x40ff60 };
    foreach (i, c : glow) {
        MeshStandardMaterial* m = drie::mesh_standard_material(color: c, emissive: c, emissive_intensity: 5.0f);
        Mesh* panel = drie::mesh(drie::box_geometry(0.1f, 1.2f, 1.6f), m);
        panel.position = { i == 1 ? 5.85f : -5.85f, 1.4f, 5.0f - (float)i * 6.0f };
        scene.add(panel);
    }
    PointLight* lamp = drie::point_light(0xffe0b0, 8, 0, 2);
    lamp.position = { 0, 3.0f, -6 };
    scene.add(lamp);

    Vector3 lo = { -6.5f, -0.5f, -10.5f };
    Vector3 hi = { 6.5f, 4.0f, 10.5f };
    rc = dense ? drie::radiance_cascades(lo, hi, 0.25f, 4) : drie::radiance_cascades_sparse(lo, hi, 0.25f, 4, 16384, 65536);
    rc.sky_from_scene = false;
    rc.sky_color = { 0.02f, 0.02f, 0.03f };
    rc.ground_color = { 0.01f, 0.01f, 0.01f };
    rc.voxelize(scene);
    scene.radiance = rc;

    cam = drie::perspective_camera(60, (float)W / H, 0.1f, 60);
    cam.position = { 1.2f, 1.7f, 9.0f };
    cam.look_at_vec({ -0.5f, 1.3f, -6.0f });
}

fn void frame() {
    if (!rc.sparse) rc.update(scene);
    renderer.render(scene, cam);
    drie::commit();
}

fn Result run(int skip, int warmup, int frames) {
    rc.skip_kernels = skip;
    for (int i = 0; i < warmup; i++) frame();
    Result r;
    gpu_timer::reset();
    int seen = gpu_timer::frames();
    for (int i = 0; i < frames; i++) {
        frame();
        int now = gpu_timer::frames();
        if (now == seen) continue;
        seen = now;
        double total = 0;
        foreach (j, l : labels) {
            double ms = gpu_timer::last_ms(l);
            r.pass[j].add(ms);
            total += ms;
        }
        r.rc.add(total);
        r.frame.add(gpu_timer::last_frame_ms());
    }
    rc.skip_kernels = 0;
    return r;
}

fn int parse_skip(String list) {
    int bits;
    foreach (part : list.tsplit(",")) {
        String name = part.trim();
        if (!name.len) continue;
        bool found;
        foreach (k : kernels) {
            if (k.name == name) {
                bits |= k.bit;
                found = true;
            }
        }
        if (!found) io::printfn("unknown kernel '%s' (clear, evict, alloc, propagate, trace, gather)", name);
    }
    return bits;
}

fn String skip_names(int bits) {
    if (!bits) return "none";
    String s;
    foreach (k : kernels) {
        if (bits & k.bit) s = s.len ? string::tformat("%s,%s", s, k.name) : k.name;
    }
    return s;
}

fn void print_passes(Result* r) {
    io::printn("  pass                        mean ms   min ms");
    foreach (j, l : labels) {
        if (r.pass[j].sum <= 0) continue;
        io::printfn("  %-26s %8.3f %8.3f", l, r.pass[j].mean(), r.pass[j].min);
    }
    io::printfn("  %-26s %8.3f %8.3f", "cascades", r.rc.mean(), r.rc.min);
    io::printfn("  %-26s %8.3f %8.3f", "frame (all timed passes)", r.frame.mean(), r.frame.min);
}

fn int main(String[] args) {
    bool dense;
    bool sweep = true;
    int frames = 120;
    int warmup = 60;
    int skip = -1;
    if (try e = env::tget_var("DRIE_RC_SKIP")) skip = parse_skip(e);
    for (int i = 0; i < (int)args.len; i++) {
        String a = args[i];
        if (a == "--dense") {
            dense = true;
        } else if (a == "--no-sweep") {
            sweep = false;
        } else if (a == "--frames" && i + 1 < (int)args.len) {
            frames = math::max(args[++i].to_int() ?? 120, 8);
        } else if (a == "--warmup" && i + 1 < (int)args.len) {
            warmup = math::max(args[++i].to_int() ?? 60, 8);
        } else if (a == "--skip" && i + 1 < (int)args.len) {
            skip = parse_skip(args[++i]);
        } else if (i > 0) {
            io::printfn("unknown argument '%s'", a);
            return 1;
        }
    }

    renderer = drie::renderer(width: W, height: H);
    renderer.init_headless();
    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    gpu_timer::enabled = true;
    build_scene(dense);
    io::printfn("radiance bench: %s, %dx%d, %d warm-up frames, %d timed frames a run",
        dense ? "dense" : "sparse", W, H, warmup, frames);

    int[16] runs;
    int nruns;
    if (skip >= 0) {
        runs[nruns++] = skip;
    } else {
        runs[nruns++] = 0;
        if (sweep) {
            foreach (k : kernels) if (!dense || !k.sparse_only) runs[nruns++] = k.bit;
        }
    }

    double full_rc;
    double full_frame;
    for (int i = 0; i < nruns; i++) {
        Result r = run(runs[i], warmup, frames);
        if (!r.frame.n) {
            io::printn("FAIL no GPU times read back (gpu_timer unavailable on this backend?)");
            return 1;
        }
        io::printfn("\nskip %s (%d frames read back)", skip_names(runs[i]), r.frame.n);
        print_passes(&r);
        if (i == 0) {
            full_rc = r.rc.mean();
            full_frame = r.frame.mean();
            io::printn("  every pass of the frame (Metal times a pass from its encoder's start, so a pass\n  that waits on the one before it can carry part of that one's time):");
            gpu_timer::report();
            RcStats s = rc_debug::stats(rc);
            rc_debug::print_stats(&s);
        } else {
            double saved = full_rc - r.rc.mean();
            io::printfn("  saved without it: %.3f ms of the cascades' %.3f (%.1f%%); %.3f ms of the frame's %.3f",
                saved, full_rc, full_rc > 0 ? saved / full_rc * 100 : 0, full_frame - r.frame.mean(), full_frame);
        }
    }
    return 0;
}
