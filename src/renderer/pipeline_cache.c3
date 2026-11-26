// Pipeline warmup cache (text PSO list) and Metal binary pipeline archives (sokol_warm.h).
// Records pipelines created by get_pipeline() and gen_shadow_pipeline() so later runs can warm
// them up to PIPELINE_CACHE_JOBS at a time on background worker threads (Metal only; no-op on
// other backends). Standard materials serialize by MaterialKey; node materials include shader
// source text so matching graphs share compiled programs. Window pipelines record `target_samples`
// as 0 (including Renderer.screen_stand_in) and compile against the active window's formats.
module drie;
import std::io, std::collections::list, sokol::sg;

// ---- Text pipeline cache

const String PIPELINE_CACHE_HEADER @local = "drie-pipelines 6\n";
const int PIPELINE_CACHE_JOBS @local      = 6; // Maximum concurrent background warmup jobs.

// Recorded pipeline configuration emitted by get_pipeline() or gen_shadow_pipeline().
struct PipelineRecord @private {
    MaterialKey key; // Generated material program key (src/material/material_key.c3).
    MaterialProgram* shared; // Node-material program (refcounted via MaterialProgram.users).
    bool shadow; // True for shadow-map pipelines (remaining render fields unused).
    Side side;
    bool transparent;
    bool wireframe;
    bool alpha_test;
    ulong blend; // Packed blend state from blend_key().
    DrawKind kind;
    bool gbuffer; // True for deferred G-Buffer pipelines.
    bool depth_test;
    bool depth_write;
    bool vsm;
    int target_samples; // 0 denotes the window swapchain (or Renderer.screen_stand_in).
    SgPixelFormat target_format;
    bool target_mrt;
}

fn bool PipelineRecord.same(&self, PipelineRecord* o) {
    return bitcast(
        self.key,
        KeyWords
    ) == bitcast(
        o.key,
        KeyWords
    ) && self.shared
        == o.shared && self.shadow
            == o.shadow && self.side
                == o.side && self.transparent
                    == o.transparent && self.wireframe
                        == o.wireframe && self.alpha_test
                            == o.alpha_test && self.blend
                                == o.blend && self.kind
                                    == o.kind && self.gbuffer
                                        == o.gbuffer && self.depth_test
                                            == o.depth_test && self.depth_write
                                                == o.depth_write && self.vsm
                                                    == o.vsm && self.target_samples
                                                        == o.target_samples && self.target_format
                                                            == o.target_format && self.target_mrt == o.target_mrt;
}

// Unique pipelines built from scratch during this run, exported by Renderer.pipeline_cache().
List{PipelineRecord} pipeline_records @local;

// Background compilation queue for loaded pipeline caches.
struct PipelineWarmup @private {
    List{PipelineRecord} pending; // Records not yet dispatched to worker threads.
    List{PipelineRecord} loaded; // Deduplicated union of all loaded cache records.
    List{int} jobs; // Active sg_warm_begin job IDs on already-compiling shaders.
    List{ShaderProgram*} generated; // Programs claimed by warmup jobs (ShaderProgram.claim).
    int started; // Number of records popped from `pending`.
}

PipelineWarmup cache_warmup @local;
// Non-null while pump_pipeline_cache() invokes get_pipeline() / gen_shadow_pipeline() so they
// dispatch background compilation instead of blocking on synchronous pipeline creation.
PipelineWarmup* pipeline_warmup @private;

fn void record_pipeline(PipelineRecord r) @private {
    // Skip warmup dispatches so only pipelines actually drawn with are recorded.
    if (pipeline_warmup) return;
    if (!pipeline_records.is_initialized()) pipeline_records.init(mem);
    foreach (&o : pipeline_records) {
        if (o.same(&r)) return;
    }
    if (r.shared) r.shared.users++;
    pipeline_records.push(r);
}

// Enqueues `r` for background warmup if not already present in `loaded`.
fn bool PipelineWarmup.add(&self, PipelineRecord r) {
    foreach (&o : self.loaded) {
        if (o.same(&r)) return false;
    }
    self.loaded.push(r);
    self.pending.push(r);
    return true;
}

// Parses a 64-hex-digit MaterialKey written by Renderer.pipeline_cache().
fn MaterialKey? key_from_hex(String hex) @local {
    if (hex.len != 64) return NOT_FOUND~;
    ulong[4] q;
    foreach (i, c : hex) {
        ulong v;
        switch {
            case c >= '0' && c <= '9': v = (ulong)(c - '0');
            case c >= 'a' && c <= 'f': v = (ulong)(c - 'a' + 10);
            default: return NOT_FOUND~;
        }
        q[i / 16] = q[i / 16] << 4 | v;
    }
    KeyWords k = { (uint128)q[0] << 64 | q[1], (uint128)q[2] << 64 | q[3] };
    return bitcast(k, MaterialKey);
}

fn String key_hex(MaterialKey key) @local {
    KeyWords k = bitcast(key, KeyWords);
    return string::tformat("%016x%016x%016x%016x", (ulong)(k.lo >> 64), (ulong)k.lo, (ulong)(k.hi >> 64), (ulong)k.hi);
}

// Dispatches a background compile job for generated program `p` and pipeline `desc`.
fn void PipelineWarmup.start_generated(&self, ShaderProgram* p, SgPipelineDesc* desc) {
    SgShaderDesc shader = p.desc();
    int job             = sg_warm_begin(&shader, desc);
    if (job < 0) return;
    if (p.claim(job)) {
        self.generated.push(p);
    } else {
        self.jobs.push(job);
    }
}

// Serializes the recorded pipelines into a portable text cache allocated in `a`:
//   drie-pipelines 6
//   M <key> <depth>                       Node material program (0-indexed in emission order),
//                                         followed by ShaderProgram.write output (and prepass
//                                         program when `<depth>` is 1).
//   G <key> <program> <side> <transparent> <wireframe> <alpha_test> <blend> <kind>
//     <depth_test> <depth_write> <vsm> <target_samples> <target_format> <mrt> <gbuffer>
//                                         Render pipeline (`<program>` -1 for non-node materials,
//                                         `<target_samples>` 0 for the window).
//   H <key> <program>                     Shadow-map pipeline.
// 
// With `include_loaded` (default true), merges records loaded via load_pipeline_cache() so saving
// after a warmed run preserves entries that hit the cache without being built from scratch.
fn String Renderer.pipeline_cache(&self, Allocator a, bool include_loaded = true) {
    DString out = dstring::new(a);
    out.append(PIPELINE_CACHE_HEADER);
    @pool() {
        List{PipelineRecord} merged;
        merged.tinit();
        if (pipeline_records.is_initialized()) foreach (&r : pipeline_records) merged.push(*r);
        if (include_loaded && cache_warmup.loaded.is_initialized()) {
            foreach (&r : cache_warmup.loaded) {
                bool dup = false;
                foreach (&o : merged) {
                    if (o.same(r)) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) merged.push(*r);
            }
        }
        List{MaterialProgram*} programs;
        programs.tinit();
        foreach (&r : merged) {
            int program = -1;
            if (r.shared) {
                foreach (i, p : programs) {
                    if (p == r.shared) program = (int)i;
                }
                if (program < 0) {
                    MaterialProgram* mp = r.shared;
                    program = (int)programs.len();
                    programs.push(mp);
                    out.appendf("M %s %d\n", key_hex(mp.key), mp.depth ? 1 : 0);
                    mp.program.write(&out);
                    if (mp.depth) mp.depth.write(&out);
                }
            }
            if (r.shadow) {
                out.appendf("H %s %d\n", key_hex(r.key), program);
                continue;
            }
            out.appendf(
                "G %s %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                key_hex(r.key),
                program,
                r.side.ordinal,
                r.transparent ? 1 : 0,
                r.wireframe ? 1 : 0,
                r.alpha_test ? 1 : 0,
                r.blend,
                r.kind.ordinal,
                r.depth_test ? 1 : 0,
                r.depth_write ? 1 : 0,
                r.vsm ? 1 : 0,
                r.target_samples,
                (uint)r.target_format,
                r.target_mrt ? 1 : 0,
                r.gbuffer ? 1 : 0
            );
        }
    };
    return out.str_view();
}

// Writes Renderer.pipeline_cache() to `path`. Returns false if saving fails.
fn bool Renderer.save_pipeline_cache(&self, String path) {
    @pool() {
        if (catch file::save(path, self.pipeline_cache(tmem))) return false;
    };
    return true;
}

// Clears the pipelines recorded during this run (useful when saving per-scene caches in one process).
fn void Renderer.clear_pipeline_cache(&self) {
    if (!pipeline_records.is_initialized()) return;
    foreach (&r : pipeline_records) {
        if (r.shared && --r.shared.users == 0) free_material_program(r.shared);
    }
    pipeline_records.clear();
}

// Parses a serialized node material program at `*at` in `text`, advancing `*at`, and deduplicates
// it against already-registered MaterialPrograms.
fn MaterialProgram*? read_material_program(String text, usz* at, MaterialKey key, bool depth) @local {
    ShaderProgram* program  = read_program(text, at)!;
    ShaderProgram*? prepass = depth ? read_program(text, at) : (ShaderProgram*)null;
    if (catch prepass) {
        free_program(program);
        return NOT_FOUND~;
    }
    MaterialProgram* mp = mem::new(MaterialProgram);
    mp.key = key;
    mp.program = program;
    mp.depth = prepass;
    mp.pipelines.init(mem);
    mp = share_material_program(mp);
    // Reference held by cache_warmup.loaded.
    mp.users++;
    return mp;
}

// Parses a text pipeline cache and queues its unique entries for background compilation on Metal.
// Returns the number of newly added records; safe to call before the first frame or between frames.
fn int Renderer.load_pipeline_cache(&self, String text) {
    if (!text.starts_with(PIPELINE_CACHE_HEADER)) return 0;
    PipelineWarmup* w = &cache_warmup;
    if (!w.pending.is_initialized()) {
        w.pending.init(mem);
        w.loaded.init(mem);
        w.jobs.init(mem);
        w.generated.init(mem);
    }
    int added;
    @pool() {
        List{MaterialProgram*} programs;
        programs.tinit();
        usz at = PIPELINE_CACHE_HEADER.len;
        while (at < text.len) {
            sz? end = text[at..].index_of_char('\n');
            if (catch end) break;
            String[] f = text[at:(usz)end].tsplit(" ");
            at += (usz)end + 1;
            int[15] n;
            foreach (i, x : f[2..]) {
                if (i < n.len) n[i] = x.to_int() ?? -100;
            }
            if (f.len < 2) return added;
            MaterialKey? key = key_from_hex(f[1]);
            if (catch key) return added;
            switch (f[0]) {
                case "M":
                    if (f.len != 3) return added;
                    MaterialProgram*? mp = read_material_program(text, &at, key, n[0] == 1);
                    if (catch mp) return added;
                    programs.push(mp);
                case "H":
                case "G":
                    bool shadow = f[0] == "H";
                    if (f.len != (shadow ? 3 : 16) || n[0] < -1 || n[0] >= (int)programs.len()) return added;
                    MaterialProgram* shared = n[0] < 0 ? null : programs[n[0]];
                    if (shadow) {
                        if (w.add({ .key = key, .shared = shared, .shadow = true })) added++;
                        continue;
                    }
                    if (n[1] < 0 || n[1] > Side.DOUBLE_SIDE.ordinal || n[5] < 0 || n[6] < 0
                        || n[6] > DrawKind.DEPTH.ordinal
                        || n[10] < 0) {
                        continue;
                    }
                    PipelineRecord r = {
                        .key = key,
                        .shared = shared,
                        .side = (Side)n[1],
                        .transparent = n[2] == 1,
                        .wireframe = n[3] == 1,
                        .alpha_test = n[4] == 1,
                        .blend = (ulong)n[5],
                        .kind = (DrawKind)n[6],
                        .depth_test = n[7] == 1,
                        .depth_write = n[8] == 1,
                        .vsm = n[9] == 1,
                        .target_samples = n[10],
                        .target_format = (SgPixelFormat)(uint)n[11],
                        .target_mrt = n[12] == 1,
                        .gbuffer = n[13] == 1,
                    };
                    if (w.add(r)) added++;
                default:
                    return added;
            }
        }
    };
    self.pump_pipeline_cache();
    return added;
}

// Loads a text pipeline cache from `path`. Returns false if the file cannot be read.
fn bool Renderer.load_pipeline_cache_file(&self, String path) {
    @pool() {
        char[]? text = file::load_temp(path);
        if (catch text) return false;
        self.load_pipeline_cache((String)text);
    };
    return true;
}

// Returns `{ remaining, total }` for loaded pipeline caches, advancing pending warmup jobs.
// Warmup is complete when `remaining` reaches 0.
fn int[2] Renderer.compiling(&self) {
    PipelineWarmup* w = &cache_warmup;
    self.pump_pipeline_cache();
    int[2] n = { (int)w.pending.len(), w.started + (int)w.pending.len() };
    if (w.jobs.is_initialized()) {
        for (sz i = 0; i < w.jobs.len();) {
            if (sg_warm_state(w.jobs[i]) == 0) {
                n[0]++;
                i++;
            } else {
                sg_warm_end(w.jobs[i]);
                w.jobs.remove_at(i);
            }
        }
    }
    foreach (p : w.generated) {
        if (p.compiling()) n[0]++;
    }
    return n;
}

// Dispatches pending pipeline cache entries on Metal while fewer than PIPELINE_CACHE_JOBS are
// running; clears the queue on backends without background compilation.
fn void Renderer.pump_pipeline_cache(&self) {
    PipelineWarmup* w = &cache_warmup;
    // Wait until setup_gfx() finishes; Renderer.render() pumps on the first frame.
    if (!w.pending.len() || !state.gfx_ready) return;
    // Background worker compilation is supported on Metal only; other backends would stall.
    switch (sg::query_backend()) {
        case METAL_MACOS:
        case METAL_IOS:
        case METAL_SIMULATOR:
            break;
        default:
            w.pending.clear();
            return;
    }
    int running;
    foreach (job : w.jobs) {
        if (sg_warm_state(job) == 0) running++;
    }
    foreach (p : w.generated) {
        if (p.compiling()) running++;
    }
    ShadowType shadow_type = self.shadow_type;
    defer self.shadow_type = shadow_type;
    pipeline_warmup = w;
    defer pipeline_warmup = null;
    while (running < PIPELINE_CACHE_JOBS && w.pending.len()) {
        PipelineRecord r = w.pending.pop_first()!!;
        w.started++;
        // Skip node programs generated for another backend's shading language.
        if (r.shared && r.shared.program.lang != backend_lang()) continue;
        running++;
        MaterialProgram* gen = r.shared ?: material_program(r.key);
        if (r.shadow) {
            (void)gen_shadow_pipeline(gen);
            continue;
        }
        // get_pipeline() reads VSM mode off the renderer.
        self.shadow_type = r.vsm ? ShadowType.VSM : shadow_type == VSM ? ShadowType.PCF : shadow_type;
        RenderTarget target = { .samples = r.target_samples, .format = r.target_format, .mrt = r.target_mrt };
        (void)get_pipeline(
            r.side,
            r.transparent,
            r.wireframe,
            r.target_samples ? &target : null,
            r.alpha_test,
            r.blend,
            r.kind,
            r.depth_test,
            r.depth_write,
            r.gbuffer,
            gen
        );
    }
}

// ---- Binary pipeline archive

// Metal MTLBinaryArchive persistence (sokol_warm.h): caches compiled GPU pipelines on disk per
// GPU and OS version. Newly compiled pipelines are saved into sequential files (`path`, `path.1`,
// `path.2`, ...). Metal only; ignored on other backends.

extern fn CInt sg_warm_archive_load(ZString path) @cname("sg_warm_archive_load");
extern fn CInt sg_warm_archive_save(ZString path) @cname("sg_warm_archive_save");
extern fn void sg_warm_archive_counts(CInt* hits, CInt* misses) @cname("sg_warm_archive_counts");

String archive_path @local;
bool archive_loaded @local;

// Configures the Metal binary archive base path (`path`, `path.1`, ...) and loads existing files
// immediately if the GPU is ready, or during setup_gfx() otherwise.
fn void Renderer.use_pipeline_archive(&self, String path) {
    if (archive_path.len) free(archive_path.ptr);
    archive_path = path.copy(mem);
    archive_loaded = false;
    load_pipeline_archive();
}

// Loads any configured binary archive files once setup_gfx() has initialized the Metal device.
fn void load_pipeline_archive() @private {
    if (!archive_path.len || archive_loaded || !state.gfx_ready) return;
    archive_loaded = true;
    @pool() {
        CInt r = sg_warm_archive_load(archive_path.zstr_tcopy());
        if (r > 0) io::printfn(
            "drie: pipelines compiled ahead of time in %s (%d %s)",
            archive_path,
            r,
            r == 1 ? "file" : "files"
        );
    };
}

// Persists any pipelines newly compiled this run into the next archive file. Returns false on
// write failure.
fn bool Renderer.save_pipeline_archive(&self) {
    if (!archive_loaded) return true;
    @pool() {
        return sg_warm_archive_save(archive_path.zstr_tcopy()) >= 0;
    };
}

// Returns `{ archive_hits, archive_misses }` for Metal binary archive lookups this run.
fn int[2] pipeline_archive_counts() {
    CInt hits;
    CInt misses;
    sg_warm_archive_counts(&hits, &misses);
    return { hits, misses };
}