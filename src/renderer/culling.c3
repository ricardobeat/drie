// GPU per-instance frustum culling (Renderer.gpu_culling) and shared instance matrix arena.
// CPU collect() culls whole meshes by aggregate bounding sphere; cull_dispatch() then runs
// cull_cs (shaders/kernels.glsl) to test each instance sphere against camera and shadow frusta
// and compact visible indices for fetch_instance() in shaders/shader.glsl. Because sokol lacks
// indirect draws, draw counts come from a 3-frame async readback ring with padding, and the
// vertex shader discards excess instances. Static meshes live in one shared InstanceArena; meshes
// whose instances change after two uploads get dedicated buffers and skip per-instance culling.
module drie;
import std::collections::list, std::math, std::io;
import sokol;

// ---- Instance arena

// Minimum instance count required to submit a mesh for GPU per-instance culling.
const int CULL_MIN_INSTANCES @private = 8;
// Maximum renders per frame that can run GPU culling (each owns buffers and readback slots).
const int CULL_MAX_SETS @private      = 16;

// Maximum views culled in a single render (camera plus shadow maps drawn that render).
const int CULL_MAX_VIEWS @private = 8;

struct InstanceArena @private {
    List{float} data;
    StorageBuffer* buffer;
    bool dirty;
}

InstanceArena arena @private;

// Recomputes aggregate instance bounds for `m` and uploads its 3x4 instance matrices to the GPU.
// Places matrices in the shared arena on first upload (updating in place on a single same-size
// re-upload), or switches to a dedicated vertex buffer for frequently updated meshes.
fn void upload_instances_gpu(Mesh* m) @private {
    if (m.instances_uploaded) return;
    m.instance_uploads++;
    note_moved_mesh(m);
    BufferGeometry* g =  m.geometry;
    Vector3 lo        =  { 1e30f, 1e30f, 1e30f };
    Vector3 hi        = -lo;
    for (int i = 0; i < m.instance_count; i++) {
        float* d   = &m.instance_data[(sz)i * 12];
        Vector3 gc = g.bounds_center;
        Vector3 c = {
            d[0] * gc.x + d[1] * gc.y + d[2] * gc.z + d[3],
            d[4] * gc.x + d[5] * gc.y + d[6] * gc.z + d[7],
            d[8] * gc.x + d[9] * gc.y + d[10] * gc.z + d[11],
        };
        float s = math::max(
            ((Vector3) { d[0], d[4], d[8] }).length(),
            math::max(((Vector3) { d[1], d[5], d[9] }).length(), ((Vector3) { d[2], d[6], d[10] }).length())
        );
        Vector3 r = { s, s, s };
        r *= g.bounds_radius;
        lo = math::min(lo, c - r);
        hi = math::max(hi, c + r);
    }
    m.instance_center = (lo + hi) * 0.5f;
    m.instance_radius = (hi - lo).length() * 0.5f;
    m.instances_uploaded = true;
    if (m.instance_uploads == 1 && !m.in_arena) {
        // First upload: append to the shared arena.
        arena.data.allocator = mem;
        m.arena_base = (int)(arena.data.len() / 12);
        m.in_arena = true;
        m.arena_size = m.instance_count;
        arena.data.push_all(m.instance_data.array_view());
        arena.dirty = true;
        return;
    }
    if (m.in_arena && m.instance_uploads == 2 && m.arena_size == m.instance_count) {
        // One same-size update (e.g. instances finalized over two frames): overwrite in place.
        mem::copy(
            &arena.data[(sz)m.arena_base * 12],
            m.instance_data.entries,
            (sz)m.instance_count * 12 * (sz)float::size
        );
        arena.dirty = true;
        return;
    }
    // Frequently updated mesh: allocate a dedicated buffer (arena slice remains unused).
    m.in_arena = false;
    if (m.instance_buf.id) sg::destroy_buffer(m.instance_buf);
    m.instance_buf = sg::make_buffer(
        &&{
            .data = { .ptr = m.instance_data.entries, .size = (usz)m.instance_data.len() * float::size },
        }
    );
}

// Returns the shared arena StorageBuffer, re-uploading if dirty.
fn StorageBuffer* arena_buffer() @private {
    if (!arena.buffer) arena.buffer = storage_buffer(16, false, true, "drie-instances");
    if (arena.dirty) {
        arena.dirty = false;
        arena.buffer.write(arena.data.entries, arena.data.len() * float::size);
    }
    return arena.buffer;
}

// Returns the vertex buffer and byte offset for `m`'s instance matrices.
fn SgBuffer instance_vertex_buffer(Mesh* m, int* offset) @private {
    *offset = 0;
    if (!m.in_arena) return m.instance_buf;
    *offset = m.arena_base * 12 * (int)float::size;
    return arena_buffer().buffer;
}

// ---- GPU culling ring and dispatch

struct CullJobInfo @private {
    Mesh* mesh;
    int view;
    int draw_count; // Padded instance count submitted to sg::draw().
    int cpu_count; // CPU reference count when Renderer.check_compute is enabled (-1 otherwise).
}

// Per-render culling buffers. The 3-slot `counts` ring lets the GPU write the current frame
// while the CPU reads back a completed earlier frame without stalling.
struct CullSet @private {
    StorageBuffer* jobs;
    StorageBuffer* planes;
    StorageBuffer* visible;
    StorageBuffer*[3] counts;
    int[3] slots;
    uint[3] frames;
    List{CullJobInfo}[3] infos;
    bool[3] tracked;
    List{uint} fetched; // Most recent readback counts (from `fetched_ring`) for this frame.
    int fetched_ring;
    uint fetched_frame;
    bool[3] checked;
}

struct CullState @private {
    List{CullSet*} sets;
    uint frame;
    int used;
    CullSet* current;
    int ring;
    CullPlane[CULL_MAX_VIEWS * 6] planes;
    int views;
    List{CullJob} jobs;
    sz visible_total;
    ComputePass* kernel;
    bool active;
    bool check;
    int checks; // Renderer.check_compute: total jobs verified against CPU.
    int check_failures; // Renderer.check_compute: frames with a CPU/GPU mismatch.
    int short_draws; // Draws whose lagged draw_count was smaller than actual visible count.
    int short_instances; // Total instances clipped across short_draws.
}

CullState cull @private;

// Initializes culling state for the current render (`enabled` = Renderer.gpu_culling && compute).
fn void cull_begin(bool enabled, bool check) @private {
    cull.active = false;
    cull.views = 0;
    cull.jobs.allocator = mem;
    cull.jobs.clear();
    cull.visible_total = 0;
    cull.check = check;
    if (!enabled || !compute_supported()) return;
    uint frame = sg::query_stats().prev_frame.frame_index;
    if (frame != cull.frame || !cull.sets.len()) {
        cull.frame = frame;
        cull.used = 0;
    }
    if (cull.used >= CULL_MAX_SETS) return;
    if (cull.used >= cull.sets.len()) {
        CullSet* s = mem::new(CullSet);
        s.jobs = storage_buffer(16, true, false, "drie-cull-jobs");
        s.planes = storage_buffer(16, true, false, "drie-cull-planes");
        s.visible = storage_buffer(16, false, false, "drie-cull-visible");
        for (int i = 0; i < 3; i++) {
            s.counts[i] = storage_buffer(16, false, false, "drie-cull-counts");
            s.slots[i] = readback_slot();
            s.infos[i].allocator = mem;
        }
        s.fetched.allocator = mem;
        s.fetched_ring = -1;
        cull.sets.push(s);
    }
    cull.current = cull.sets[cull.used++];
    cull.ring = (int)(frame % 3);
    cull.current.infos[cull.ring].clear();
    cull.current.tracked[cull.ring] = false;
    cull.current.frames[cull.ring] = frame;
    cull.active = true;
}

// Registers a frustum view (up to CULL_MAX_VIEWS) and returns its view index, or -1 if full.
fn int cull_view(Frustum* f) @private {
    if (!cull.active || cull.views >= CULL_MAX_VIEWS) return -1;
    int v = cull.views++;
    for (int k = 0; k < 5; k++) {
        Vec4 p = f.planes[k];
        cull.planes[v * 6 + k].plane = { p.x, p.y, p.z, p.w };
    }
    // Sixth plane is a dummy always-inside plane to pad to 6 planes per view.
    cull.planes[v * 6 + 5].plane = { 0, 0, 0, 1 };
    return v;
}

// Queues GPU per-instance culling for `m` against `view`. Returns the job index, or -1 if `m`
// is drawn without per-instance culling.
fn int cull_job(Mesh* m, int view) @private {
    // Per-instance colors are not compacted alongside matrices, so colored instanced meshes draw whole.
    if (!cull.active || view < 0 || !m.in_arena || !m.frustum_culled || m.instance_count < CULL_MIN_INSTANCES
        || m.instance_colors.len()) return -1;
    int j             = (int)cull.jobs.len();
    Mat4 w            = m.matrix_world;
    BufferGeometry* g = m.geometry;
    cull.jobs.push(
        {
            .m0 = { w[0][0], w[1][0], w[2][0], w[3][0] },
            .m1 = { w[0][1], w[1][1], w[2][1], w[3][1] },
            .m2 = { w[0][2], w[1][2], w[2][2], w[3][2] },
            .sphere = { g.bounds_center.x, g.bounds_center.y, g.bounds_center.z, g.bounds_radius },
            .extra = { m.material.sway * 2, max_scale_of(w), 0, 0 },
            .range = { (uint)m.arena_base, (uint)m.instance_count, (uint)cull.visible_total, (uint)view },
        }
    );
    cull.visible_total += m.instance_count;
    cull.current.infos[cull.ring].push({ .mesh = m, .view = view, .draw_count = m.instance_count, .cpu_count = -1 });
    return j;
}

fn float max_scale_of(Mat4 m) @private {
    float sx = ((Vector3) { m[0][0], m[0][1], m[0][2] }).length();
    float sy = ((Vector3) { m[1][0], m[1][1], m[1][2] }).length();
    float sc = ((Vector3) { m[2][0], m[2][1], m[2][2] }).length();
    return math::max(sx, math::max(sy, sc));
}

// Dispatches `cull_cs` for all queued views/jobs, tracks count readback, and computes each job's
// padded `draw_count` from the newest completed readback frame.
fn void cull_dispatch() @private {
    if (!cull.active || !cull.jobs.len()) return;
    CullSet* s = cull.current;
    if (!cull.kernel) cull.kernel = compute_pass(shaders::cull_shader_desc(sg::query_backend()), "drie-cull");
    s.jobs.write(cull.jobs.entries, cull.jobs.len() * CullJob::size);
    s.planes.write(&cull.planes, (sz)cull.views * 6 * (sz)CullPlane::size);
    s.visible.ensure(cull.visible_total * CullIndex::size);
    StorageBuffer* counts = s.counts[cull.ring];
    counts.ensure(cull.jobs.len() * CullIndex::size);
    ComputePass* k = cull.kernel;
    k.set_buffer(shaders::VIEW_CULL_JOBS, s.jobs);
    k.set_buffer(shaders::VIEW_CULL_PLANES, s.planes);
    k.set_buffer(shaders::VIEW_CULL_INSTANCES, arena_buffer());
    k.set_buffer(shaders::VIEW_CULL_VISIBLE, s.visible);
    k.set_buffer(shaders::VIEW_CULL_COUNTS, counts);
    begin_compute("drie-cull");
    k.dispatch((int)cull.jobs.len());
    s.tracked[cull.ring] = counts.track(s.slots[cull.ring]);
    end_compute();
    // Fetch the newest completed readback from 1 or 2 frames back.
    if (s.fetched_frame != cull.frame || s.fetched_ring < 0) {
        s.fetched_frame = cull.frame;
        s.fetched_ring = -1;
        for (uint back = 1; back <= 2; back++) {
            int r = (int)((cull.frame + 3 - back) % 3);
            if (!s.tracked[r] || s.frames[r] != cull.frame - back) continue;
            sz n = s.infos[r].len();
            s.fetched.clear();
            for (sz i = 0; i < n; i++) s.fetched.push(0);
            // Under check_compute, block on frame-1 so every frame is compared.
            bool wait = cull.check && back == 1;
            if (n && read_back(s.slots[r], s.fetched.entries, n * (sz)uint::size, wait)) {
                s.fetched_ring = r;
                if (cull.check) cull_compare(s, r);
                break;
            }
        }
    }
    List{CullJobInfo}* infos  = &s.infos[cull.ring];
    List{CullJobInfo}* before = s.fetched_ring >= 0 ? &s.infos[s.fetched_ring] : null;
    foreach (j, &info : *infos) {
        if (cull.check) info.cpu_count = cpu_cull_count(&cull.jobs[j]);
        if (!before || j >= before.len()) continue;
        CullJobInfo* b = &before.entries[j];
        if (b.mesh != info.mesh || b.view != info.view) continue;
        int kept = (int)s.fetched[j];
        // Pad the lagged count so camera motion does not clip newly visible instances.
        info.draw_count = math::min(info.mesh.instance_count, kept + math::max(kept / 8, 4));
    }
}

// Returns the instance count to submit to sg::draw() for `job`.
fn int cull_draw_count(int job) @private => cull.current.infos[cull.ring][job].draw_count;

// Binds instance culling buffers to `bind` and returns the `inst_cull` vertex uniform (`job == -1`
// binds dummy buffers and returns zeros).
fn Vec4 cull_bind(SgBindings* bind, int job) @private {
    if (job < 0 || !cull.active) {
        StorageBuffer* dummy = arena_buffer();
        bind.views[shaders::VIEW_INST_ROWS] = dummy.view;
        bind.views[shaders::VIEW_INST_VISIBLE] = dummy.view;
        bind.views[shaders::VIEW_INST_COUNTS] = dummy.view;
        return { 0, 0, 0, 0 };
    }
    CullSet* s = cull.current;
    bind.views[shaders::VIEW_INST_ROWS] = arena_buffer().view;
    bind.views[shaders::VIEW_INST_VISIBLE] = s.visible.view;
    bind.views[shaders::VIEW_INST_COUNTS] = s.counts[cull.ring].view;
    return { 1, (float)cull.jobs[job].range[2], (float)job, 0 };
}

// ---- CPU verification

// Reference CPU sphere-vs-frustum count matching `cull_cs` (used by Renderer.check_compute).
fn int cpu_cull_count(CullJob* job) @private {
    int kept;
    CullPlane* planes = &cull.planes[job.range[3] * 6];
    for (uint i = 0; i < job.range[1]; i++) {
        float* d   = &arena.data[(sz)(job.range[0] + i) * 12];
        float[4] c = { job.sphere[0], job.sphere[1], job.sphere[2], 1 };
        float[4] lc = {
            d[0] * c[0] + d[1] * c[1] + d[2] * c[2] + d[3],
            d[4] * c[0] + d[5] * c[1] + d[6] * c[2] + d[7],
            d[8] * c[0] + d[9] * c[1] + d[10] * c[2] + d[11],
            1,
        };
        Vector3 wc = {
            job.m0[0] * lc[0] + job.m0[1] * lc[1] + job.m0[2] * lc[2] + job.m0[3],
            job.m1[0] * lc[0] + job.m1[1] * lc[1] + job.m1[2] * lc[2] + job.m1[3],
            job.m2[0] * lc[0] + job.m2[1] * lc[1] + job.m2[2] * lc[2] + job.m2[3],
        };
        float scale = math::max(
            ((Vector3) { d[0], d[4], d[8] }).length(),
            math::max(((Vector3) { d[1], d[5], d[9] }).length(), ((Vector3) { d[2], d[6], d[10] }).length())
        );
        float radius = job.sphere[3] * scale * job.extra[1] + job.extra[0];
        bool inside  = true;
        for (int k = 0; k < 6; k++) {
            float[4] p = planes[k].plane;
            if (p[0] * wc.x + p[1] * wc.y + p[2] * wc.z + p[3] < -radius) inside = false;
        }
        if (inside) kept++;
    }
    return kept;
}

// Compares GPU readback counts in ring `r` against CPU reference counts (Renderer.check_compute).
fn void cull_compare(CullSet* s, int r) @private {
    int bad;
    int worst;
    foreach (j, info : s.infos[r]) {
        if (info.cpu_count < 0) continue;
        int d = math::abs((int)s.fetched[j] - info.cpu_count);
        cull.checks++;
        // Record when the lagged padded draw_count fell below the actual visible instance count.
        if (info.draw_count < (int)s.fetched[j]) {
            cull.short_draws++;
            cull.short_instances += (int)s.fetched[j] - info.draw_count;
        }
        if (d) {
            bad++;
            worst = math::max(worst, d);
        }
    }
    if (bad) {
        cull.check_failures++;
        io::eprintfn(
            "drie: check_compute: GPU culling differs from the CPU's in %d of %d jobs (by up to %d instances)",
            bad,
            s.infos[r].len(),
            worst
        );
    }
}

// Prints accumulated Renderer.check_compute statistics for cluster grids and instance culling
// to stderr.
fn void print_compute_checks() {
    io::eprintfn(
        "drie: check_compute: %d cluster grids compared (%d differed), %d culled draws compared (%d frames differed); "
            "%d draws asked for fewer instances than their cull kept (%d instances left out for a frame)",
        lights_gpu.checks,
        lights_gpu.check_failures,
        cull.checks,
        cull.check_failures,
        cull.short_draws,
        cull.short_instances
    );
}