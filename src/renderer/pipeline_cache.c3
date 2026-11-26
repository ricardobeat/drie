// The pipeline cache: the list of the render pipelines a run made, saved to be compiled again,
// in the background, at the start of a later run. Like Unreal's PSO cache: play the game once
// (or each level), save what it drew with, ship that, and load it at startup; or load each
// level's while its loading screen is up.
//
// A pipeline is a program and the state it draws with (sides, blending, depth, the target's
// format). The driver compiles one the first time it is made, which can take from a few to
// hundreds of milliseconds: a frame that makes one stalls. The driver keeps what it compiles
// in a cache of its own on disk, so a pipeline made once before takes next to no time; the
// pipeline cache is what gets it into the driver's cache before the frame that wants it.
//
//   // a run that plays the game:
//   renderer.save_pipeline_cache("pipelines.txt");
//
//   // at startup:
//   renderer.load_pipeline_cache(file::load_temp("pipelines.txt"))!!;
//   // each frame, for a loading bar: what is left of it
//   int[2] n = renderer.compiling();      // { still compiling, all }
//
// Loading compiles a few at a time on worker threads (Metal only: elsewhere it does
// nothing), so the frames go on: a pipeline still compiling when something wants it is
// made then, after its compile in the background (or drawn with the plain material till its
// pipelines are ready, for a node material with Renderer.compile_in_background). The
// materials' generated programs are listed by their MaterialKey; a node material's, which its
// key alone can't print, as the program itself (its text): a graph that prints the same then
// gets the program already made. The list is text, one line a pipeline (see
// Renderer.pipeline_cache); it holds no compiled code, so it doesn't go stale when the
// driver or the GPU change, only less complete.
//
// What a render target stands for: a pipeline that draws into the window is listed as the
// window's, and is compiled for whatever window the run that loads it has (its samples, its
// formats). A run without a window (making pictures, tests) can name the render target that
// stands for one, Renderer.screen_stand_in.
module drie;
import std::io, std::collections::list, sokol::sg;

const String PIPELINE_CACHE_HEADER @local = "drie-pipelines 6\n";
// pipelines compiled at once while a cache loads
const int PIPELINE_CACHE_JOBS @local = 6;

// A pipeline as get_pipeline (or gen_shadow_pipeline) made it.
struct PipelineRecord @private {
    // its generated program's key (materials_nodes.c3)
    MaterialKey key;
    // a node material's program (MaterialKey.nodes), which the key alone doesn't print: listed
    // by its text (kept while it is listed: MaterialProgram.users)
    MaterialProgram* shared;
    // a shadow map's (the rest unused)
    bool shadow;
    Side side;
    bool transparent;
    bool wireframe;
    bool alpha_test;
    // the blending, as blend_key gives it (the material's blending and its factors)
    ulong blend;
    DrawKind kind;
    // into the deferred G-buffer
    bool gbuffer;
    bool depth_test;
    bool depth_write;
    bool vsm;
    // the render target: 0 samples for the window's
    int target_samples;
    SgPixelFormat target_format;
    bool target_mrt;
}

fn bool PipelineRecord.same(&self, PipelineRecord* o) {
    return bitcast(self.key, KeyWords) == bitcast(o.key, KeyWords) && self.shared == o.shared
        && self.shadow == o.shadow && self.side == o.side && self.transparent == o.transparent
        && self.wireframe == o.wireframe && self.alpha_test == o.alpha_test && self.blend == o.blend
        && self.kind == o.kind && self.gbuffer == o.gbuffer && self.depth_test == o.depth_test
        && self.depth_write == o.depth_write && self.vsm == o.vsm && self.target_samples == o.target_samples
        && self.target_format == o.target_format && self.target_mrt == o.target_mrt;
}

// the pipelines made in this run, for Renderer.pipeline_cache
List{PipelineRecord} pipeline_records @local;

// What a loaded cache has left to do.
struct PipelineWarmup @private {
    // the records not started yet
    List{PipelineRecord} pending;
    // all it was given (one of each, however many caches list it)
    List{PipelineRecord} loaded;
    // pipelines compiling (sg_warm_begin's jobs) whose programs were compiling already
    List{int} jobs;
    // generated programs compiled by the jobs of the pipelines it has started (ShaderProgram.claim)
    List{ShaderProgram*} generated;
    // the pipelines it has started
    int started;
}

PipelineWarmup cache_warmup @local;
// set while one of cache_warmup's pending records is started: get_pipeline then compiles
// in the background rather than making the pipeline (see start)
PipelineWarmup* pipeline_warmup @private;

fn void record_pipeline(PipelineRecord r) @private {
    // (not what a cache being loaded asks for: only what was drawn with)
    if (pipeline_warmup) return;
    if (!pipeline_records.is_initialized()) pipeline_records.init(mem);
    foreach (&o : pipeline_records) {
        if (o.same(&r)) return;
    }
    if (r.shared) r.shared.users++;
    pipeline_records.push(r);
}

// Adds `r` to what is left to start, if it hasn't been given it before.
fn bool PipelineWarmup.add(&self, PipelineRecord r) {
    foreach (&o : self.loaded) {
        if (o.same(&r)) return false;
    }
    self.loaded.push(r);
    self.pending.push(r);
    return true;
}

// A MaterialKey from the 64 hex digits pipeline_cache writes.
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

// Compiles generated program `p`'s pipeline `desc` in the background (for get_pipeline): the
// program with it, if it isn't made or compiling yet.
fn void PipelineWarmup.start_generated(&self, ShaderProgram* p, SgPipelineDesc* desc) {
    SgShaderDesc shader = p.desc();
    int job = sg_warm_begin(&shader, desc);
    if (job < 0) return;
    if (p.claim(job)) {
        self.generated.push(p);
    } else {
        self.jobs.push(job);
    }
}

// The pipelines (those made so far), as the text load_pipeline_cache takes, in `a`'s memory:
//
//   drie-pipelines 6
//   M <key> <depth>                       a node material's program, numbered from 0 in the
//                                         order listed: the key it was printed for (64 hex
//                                         digits), then the program as ShaderProgram.write
//                                         writes it, and with <depth> 1 its depth prepass's
//   G <key> <program> <side> <transparent> <wireframe> <alpha_test> <blend> <kind>
//     <depth_test> <depth_write> <vsm> <target samples, 0: the window> <target format> <mrt>
//     <gbuffer>                           a pipeline of the generated program for <key>, or
//                                         of node program <program> (-1: none)
//   H <key> <program>                     a shadow map's pipeline
//
// `include_loaded` (on by default) also writes what a loaded cache's records had
// (cache_warmup.loaded: load_pipeline_cache's own full list, not just what's left pending),
// so that saving after a run that mostly drew with an already-loaded cache doesn't lose
// them: record_pipeline skips a pipeline started by the cache's own warmup (so a run doesn't
// save back what it only compiled for someone else to use later), and get_pipeline finding
// one already made the same way (compiled by the warmup, or by an earlier draw this run)
// returns before ever reaching record_pipeline: a pipeline the loaded cache already knew
// about is never (re-)recorded by drawing with it, only by making it from scratch.
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
            out.appendf("G %s %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", key_hex(r.key), program, r.side.ordinal,
                r.transparent ? 1 : 0, r.wireframe ? 1 : 0, r.alpha_test ? 1 : 0, r.blend, r.kind.ordinal,
                r.depth_test ? 1 : 0, r.depth_write ? 1 : 0, r.vsm ? 1 : 0, r.target_samples, (uint)r.target_format,
                r.target_mrt ? 1 : 0, r.gbuffer ? 1 : 0);
        }
    };
    return out.str_view();
}

// pipeline_cache, into a file.
fn bool Renderer.save_pipeline_cache(&self, String path) {
    @pool() {
        if (catch file::save(path, self.pipeline_cache(tmem))) return false;
    };
    return true;
}

// Clears the recorded pipelines (for a process that renders several scenes in turn and saves
// each one's pipeline cache).
fn void Renderer.clear_pipeline_cache(&self) {
    if (!pipeline_records.is_initialized()) return;
    foreach (&r : pipeline_records) {
        if (r.shared && --r.shared.users == 0) free_material_program(r.shared);
    }
    pipeline_records.clear();
}

// A node material's program as pipeline_cache lists it, at `*at` in `text` (moved past it):
// the one made already that prints the same, or this one, which a graph that prints the same
// takes from then on (see material_program).
fn MaterialProgram*? read_material_program(String text, usz* at, MaterialKey key, bool depth) @local {
    ShaderProgram* program = read_program(text, at)!;
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
    // (held by the loaded list)
    mp.users++;
    return mp;
}

// Starts compiling the pipelines `text` lists (pipeline_cache, from another run) in the
// background, a few at a time from here on (each render starts more as those finish): see
// compiling. Returns how many it lists that no cache loaded before has (the text up to
// anything it doesn't read). Call it between frames, before the first one too.
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
                    if (n[1] < 0 || n[1] > Side.DOUBLE_SIDE.ordinal || n[5] < 0
                        || n[6] < 0 || n[6] > DrawKind.DEPTH.ordinal || n[10] < 0) {
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

// load_pipeline_cache, from a file: false if it can't be read.
fn bool Renderer.load_pipeline_cache_file(&self, String path) {
    @pool() {
        char[]? text = file::load_temp(path);
        if (catch text) return false;
        self.load_pipeline_cache((String)text);
    };
    return true;
}

// The loaded caches' pipelines, for a loading bar: { still compiling (or not started), all
// of them } (all: since the first cache was loaded, so the bar fills once from 0). Done when
// the first is 0.
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

// Starts what the loaded caches have left, while fewer than PIPELINE_CACHE_JOBS compile.
fn void Renderer.pump_pipeline_cache(&self) {
    PipelineWarmup* w = &cache_warmup;
    // (before the renderer's first frame: from its first)
    if (!w.pending.len() || !state.gfx_ready) return;
    // (compiled in the background on Metal only: elsewhere it would stall the frames)
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
        // (a node material's program printed for another backend's language: never drawn here)
        if (r.shared && r.shared.program.lang != backend_lang()) continue;
        running++;
        MaterialProgram* gen = r.shared ?: material_program(r.key);
        if (r.shadow) {
            (void)gen_shadow_pipeline(gen);
            continue;
        }
        // (get_pipeline reads VSM off the renderer)
        self.shadow_type = r.vsm ? ShadowType.VSM : shadow_type == VSM ? ShadowType.PCF : shadow_type;
        RenderTarget target = { .samples = r.target_samples, .format = r.target_format, .mrt = r.target_mrt };
        (void)get_pipeline(r.side, r.transparent, r.wireframe, r.target_samples ? &target : null, r.alpha_test,
            r.blend, r.kind, r.depth_test, r.depth_write, r.gbuffer, gen);
    }
}

// ---- the pipeline archive

// Metal's binary archives of compiled pipelines (sokol_warm.h): files a run saves and the next
// loads, so a pipeline they have is made without compiling. Every pipeline drie makes is looked
// up in them; those they haven't are compiled, and the run saves them into a new file: `path`
// for the first run, then `path.1`, `path.2`... (Metal can't always save a loaded archive with
// more pipelines added.) Deleting them all starts over.
//
//   renderer.use_pipeline_archive("pipelines.archive");   // before the first frame
//   renderer.save_pipeline_archive();                     // now and then, and at the end
//
// Made on one GPU and OS, it has nothing for another (its pipelines are then compiled as
// without it). Metal only.

extern fn CInt sg_warm_archive_load(ZString path) @cname("sg_warm_archive_load");
extern fn CInt sg_warm_archive_save(ZString path) @cname("sg_warm_archive_save");
extern fn void sg_warm_archive_counts(CInt* hits, CInt* misses) @cname("sg_warm_archive_counts");

// the archive's file ("": none), and whether it was loaded yet
String archive_path @local;
bool archive_loaded @local;

// Takes the pipeline archive at `path` (its files `path`, `path.1`...; none when there is no
// file there, or none this device can read), from the GPU's setup (now, if that was before),
// and saves this run's pipelines into its next file.
fn void Renderer.use_pipeline_archive(&self, String path) {
    if (archive_path.len) free(archive_path.ptr);
    archive_path = path.copy(mem);
    archive_loaded = false;
    load_pipeline_archive();
}

// (from setup_gfx too)
fn void load_pipeline_archive() @private {
    if (!archive_path.len || archive_loaded || !state.gfx_ready) return;
    archive_loaded = true;
    @pool() {
        CInt r = sg_warm_archive_load(archive_path.zstr_tcopy());
        if (r > 0) io::printfn("drie: pipelines compiled ahead of time in %s (%d %s)", archive_path, r, r == 1 ? "file" : "files");
    };
}

// Saves the pipeline archive, if pipelines were added to it: false if it couldn't.
fn bool Renderer.save_pipeline_archive(&self) {
    if (!archive_loaded) return true;
    @pool() {
        return sg_warm_archive_save(archive_path.zstr_tcopy()) >= 0;
    };
}

// { pipelines made from the archive, pipelines compiled and added to it }.
fn int[2] pipeline_archive_counts() {
    CInt hits;
    CInt misses;
    sg_warm_archive_counts(&hits, &misses);
    return { hits, misses };
}
