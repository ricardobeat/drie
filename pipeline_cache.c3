// The pipeline cache: the list of the render pipelines a run made, saved to be compiled again,
// in the background, at the start of a later run. Like Unreal's PSO cache: play the game once
// (or each level), save what it drew with, ship that, and load it at startup; or load each
// level's while its loading screen is up.
//
// A pipeline is a program and the state it draws with (sides, blending, depth, the target's
// format). The driver compiles one the first time it is made, which can take from a few to
// hundreds of milliseconds (a second or two for a node material, whose program is printed
// at runtime): a frame that makes one stalls. The driver keeps what it compiles in a cache of
// its own on disk, so a pipeline made once before takes next to no time; the pipeline cache
// is what gets it into the driver's cache before the frame that wants it.
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
// made then, at once if the driver has finished it (a node material's mesh is left out till
// its pipelines are ready, see Renderer.compile_in_background). Node materials' programs are
// in the list too, as their hooks' text: when a graph that prints the same text is drawn, it
// gets the program already made. The list is text, one line a pipeline (see
// Renderer.pipeline_cache); it holds no compiled code, so it doesn't go stale when the
// driver, the GPU or drie's own shaders change, only less complete.
//
// What a render target stands for: a pipeline that draws into the window is listed as the
// window's, and is compiled for whatever window the run that loads it has (its samples, its
// formats). A run without a window (making pictures, tests) can name the render target that
// stands for one, Renderer.screen_stand_in.
module drie;
import std::io, std::collections::list, sokol::sg;

const String PIPELINE_CACHE_HEADER @local = "drie-pipelines 2\n";
// pipelines compiled at once while a cache loads
const int PIPELINE_CACHE_JOBS @local = 6;

// A pipeline as get_pipeline (or get_shadow_pipeline) made it.
struct PipelineRecord @private {
    // a node material's program, or null: the engine's material shader
    NodeProgram* program;
    // a shadow map's (the rest unused)
    bool shadow;
    Side side;
    bool transparent;
    bool wireframe;
    bool alpha_test;
    // the blending, as blend_key gives it (the material's blending and its factors)
    ulong blend;
    DrawKind kind;
    bool depth_test;
    bool depth_write;
    bool tangents;
    bool vsm;
    // the render target: 0 samples for the window's
    int target_samples;
    SgPixelFormat target_format;
    bool target_mrt;
}

fn bool PipelineRecord.same(&self, PipelineRecord* o) {
    return self.program == o.program && self.shadow == o.shadow && self.side == o.side
        && self.transparent == o.transparent && self.wireframe == o.wireframe && self.alpha_test == o.alpha_test
        && self.blend == o.blend && self.kind == o.kind && self.depth_test == o.depth_test
        && self.depth_write == o.depth_write && self.tangents == o.tangents && self.vsm == o.vsm
        && self.target_samples == o.target_samples && self.target_format == o.target_format
        && self.target_mrt == o.target_mrt;
}

// the pipelines made in this run, for Renderer.pipeline_cache
List{PipelineRecord} pipeline_records @local;

// What a loaded cache has left to do.
struct PipelineWarmup @private {
    // the records not started yet
    List{PipelineRecord} pending;
    // all it was given (one of each, however many caches list it)
    List{PipelineRecord} loaded;
    // the engine's pipelines compiling (sg_warm_begin's jobs)
    List{int} jobs;
    // node programs whose pipelines it has started
    List{NodeProgram*} programs;
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

// Compiles the engine's program `shader` with `desc` in the background (for get_pipeline).
fn void PipelineWarmup.start(&self, SgShaderDesc shader, SgPipelineDesc* desc) {
    int job = sg_warm_begin(&shader, desc);
    if (job >= 0) self.jobs.push(job);
}

// The pipelines (those made so far), as the text load_pipeline_cache takes, in `a`'s memory:
//
//   drie-pipelines 2
//   N <cuts> <7 lengths>                  a node program, numbered from 0 in the order listed,
//   <its 7 hooks' bodies, one after the other (NODE_HOOKS)>
//   S <program>                           a shadow map's pipeline (-1: the engine's program)
//   P <program> <side> <transparent> <wireframe> <alpha_test> <blend> <kind> <depth_test>
//     <depth_write> <tangents> <vsm> <target samples, 0: the window> <target format> <mrt>
fn String Renderer.pipeline_cache(&self, Allocator a) {
    DString out = dstring::new(a);
    out.append(PIPELINE_CACHE_HEADER);
    if (!pipeline_records.is_initialized()) return out.str_view();
    @pool() {
        List{NodeProgram*} programs;
        programs.tinit();
        foreach (&r : pipeline_records) {
            int program = -1;
            if (r.program) {
                foreach (i, p : programs) {
                    if (p == r.program) program = (int)i;
                }
                if (program < 0) {
                    NodeProgram* p = r.program;
                    program = (int)programs.len();
                    programs.push(p);
                    out.appendf("N %d %d %d %d %d %d %d %d\n", p.cuts ? 1 : 0, p.vertex_body.len, p.color_body.len,
                        p.normal_body.len, p.surface_body.len, p.output_body.len, p.env_body.len, p.specular_body.len);
                    out.append(p.vertex_body);
                    out.append(p.color_body);
                    out.append(p.normal_body);
                    out.append(p.surface_body);
                    out.append(p.output_body);
                    out.append(p.env_body);
                    out.append(p.specular_body);
                }
            }
            if (r.shadow) {
                out.appendf("S %d\n", program);
                continue;
            }
            out.appendf("P %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", program, r.side.ordinal, r.transparent ? 1 : 0,
                r.wireframe ? 1 : 0, r.alpha_test ? 1 : 0, r.blend, r.kind.ordinal, r.depth_test ? 1 : 0,
                r.depth_write ? 1 : 0, r.tangents ? 1 : 0, r.vsm ? 1 : 0, r.target_samples, (uint)r.target_format,
                r.target_mrt ? 1 : 0);
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
        w.programs.init(mem);
    }
    int added;
    @pool() {
        List{NodeProgram*} programs;
        programs.tinit();
        usz at = PIPELINE_CACHE_HEADER.len;
        while (at < text.len) {
            sz? end = text[at..].index_of_char('\n');
            if (catch end) break;
            String[] f = text[at:(usz)end].tsplit(" ");
            at += (usz)end + 1;
            int[14] n;
            foreach (i, x : f[1..]) {
                if (i < n.len) n[i] = x.to_int() ?? -1;
            }
            switch (f[0]) {
                case "N":
                    if (f.len != NODE_HOOKS + 2) return added;
                    String[NODE_HOOKS] bodies;
                    foreach (i, &b : bodies) {
                        int len = n[1 + i];
                        if (len < 0 || at + (usz)len > text.len) return added;
                        *b = text[at:(usz)len];
                        at += (usz)len;
                    }
                    programs.push(node_program(bodies, n[0] == 1));
                case "S":
                    if (f.len != 2 || n[0] >= (int)programs.len()) return added;
                    if (w.add({ .program = n[0] < 0 ? null : programs[n[0]], .shadow = true })) added++;
                case "P":
                    if (f.len != 15 || n[0] >= (int)programs.len()) return added;
                    if (n[1] < 0 || n[1] > Side.DOUBLE_SIDE.ordinal || n[5] < 0
                        || n[6] < 0 || n[6] > DrawKind.DEPTH.ordinal || n[11] < 0) {
                        continue;
                    }
                    PipelineRecord r = {
                        .program = n[0] < 0 ? null : programs[n[0]],
                        .side = (Side)n[1],
                        .transparent = n[2] == 1,
                        .wireframe = n[3] == 1,
                        .alpha_test = n[4] == 1,
                        .blend = (ulong)n[5],
                        .kind = (DrawKind)n[6],
                        .depth_test = n[7] == 1,
                        .depth_write = n[8] == 1,
                        .tangents = n[9] == 1,
                        .vsm = n[10] == 1,
                        .target_samples = n[11],
                        .target_format = (SgPixelFormat)(uint)n[12],
                        .target_mrt = n[13] == 1,
                    };
                    // (a program for tangents only where the renderer has them)
                    if (r.tangents && !r.program && !state.tangent_shader.id) continue;
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
    if (!w.jobs.is_initialized()) return n;
    for (sz i = 0; i < w.jobs.len();) {
        if (sg_warm_state(w.jobs[i]) == 0) {
            n[0]++;
            i++;
        } else {
            sg_warm_end(w.jobs[i]);
            w.jobs.remove_at(i);
        }
    }
    @pool() {
        foreach (p : w.programs) n[0] += p.compiling()[0];
    };
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
    @pool() {
        foreach (p : w.programs) running += p.compiling()[0];
    };
    ShadowType shadow_type = self.shadow_type;
    defer self.shadow_type = shadow_type;
    pipeline_warmup = w;
    defer pipeline_warmup = null;
    while (running < PIPELINE_CACHE_JOBS && w.pending.len()) {
        PipelineRecord r = w.pending.pop_first()!!;
        w.started++;
        running++;
        if (r.program) {
            bool known;
            foreach (p : w.programs) known = known || p == r.program;
            if (!known) w.programs.push(r.program);
        }
        if (r.shadow) {
            (void)get_shadow_pipeline(r.program);
            continue;
        }
        // (get_pipeline reads VSM off the renderer)
        self.shadow_type = r.vsm ? ShadowType.VSM : shadow_type == VSM ? ShadowType.PCF : shadow_type;
        RenderTarget target = { .samples = r.target_samples, .format = r.target_format, .mrt = r.target_mrt };
        (void)get_pipeline(r.side, r.transparent, r.wireframe, r.target_samples ? &target : null, r.alpha_test,
            r.blend, r.kind, r.depth_test, r.depth_write, r.tangents, r.program);
    }
}
