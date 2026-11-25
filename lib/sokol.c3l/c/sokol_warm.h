// drie's addition to sokol_gfx (not upstream): compiles a shader and a render pipeline on a
// worker thread, the way sg_make_shader and sg_make_pipeline would, and throws the results
// away. What it leaves behind is the driver's cache of compiled shaders: sg_make_shader and
// sg_make_pipeline with the same descriptions then take next to no time on the main thread.
// For shaders whose source is made at runtime (drie's node materials), which the driver
// takes a second or two to compile.
//
//   int job = sg_warm_begin(&shader_desc, &pipeline_desc);  // -1: not supported (not Metal)
//   ... later frames: sg_warm_state(job): 0 running, 1 done, 2 failed
//   sg_warm_end(job);                                       // once done or failed
//
// sg_warm_shader_begin(&shader_desc) compiles the shader alone (its two functions), for a
// program whose pipelines are made later: several of them at once, on as many threads.
//
// A stage is compiled from its source, or loaded from its bytecode (a metallib) when it has
// that: then only the pipeline's compile is left. sg_warm_library_loads(ptr, size) says whether
// bytecode loads on this device; sg_warm_dry(1) makes every job finish at once, compiling
// nothing.
//
// Metal only. Included at the end of sokol.c, with sokol_gfx's implementation.
#if defined(SOKOL_IMPL)

#define SG_WARM_JOBS 256

#if defined(SOKOL_METAL)
#include <stdatomic.h>
#include <os/lock.h>
#include <unistd.h>

// per job: 0 free, 1 running, 2 done, 3 failed
static _Atomic int _sg_warm_jobs[SG_WARM_JOBS];
// when each job started (CFAbsoluteTime), and whether it was given up on (see sg_warm_state)
static double _sg_warm_started[SG_WARM_JOBS];
static bool _sg_warm_given_up[SG_WARM_JOBS];
// Whether a job is a pipeline job from sg_warm_begin. DRIE_WARM_DELAY holds these back.
static bool _sg_warm_pipeline[SG_WARM_JOBS];

// A compile in the background that has run this long is given up on: sg_warm_state says it
// failed, so that its caller makes the shader or pipeline itself, on its own thread, rather
// than leave out what draws with it for good. (Seen under a heavily loaded machine: a job whose
// block never finished, and a rock's material never drawn.) Its slot stays taken.
#define SG_WARM_GIVE_UP_SECONDS 60.0

// DRIE_WARM_TIMES=1 in the environment: each job prints how long its libraries (source to
// AIR) and its pipeline (AIR to the GPU's code) took.
static bool _sg_warm_times(void) {
    static int on = -1;
    if (on < 0) on = getenv("DRIE_WARM_TIMES") != NULL;
    return on == 1;
}

// sg_warm_dry(1): a job compiles nothing and is done at once (to go through what a list of
// pipelines makes, for its sources: drie's DRIE_DUMP_PROGRAMS).
static bool _sg_warm_dry;

void sg_warm_dry(int on) {
    _sg_warm_dry = on != 0;
}

// The pipeline archive (MTLBinaryArchives): render pipelines compiled to the GPU's code, saved
// by a run and loaded by the next. A pipeline it has is made without compiling, by the jobs here
// and by sg_make_pipeline (sokol_gfx.h calls _sg_warm_render_pipeline); one it hasn't is
// compiled and added.
//
//   sg_warm_archive_load(path);   // before the first pipeline: the number of files taken, 0 for
//                                 // none (no file, or none this device can read), -1 no GPU
//   sg_warm_archive_save(path);   // 1 written, 0 nothing new to write, -1 failed
//
// The archive is a file per run that added to it: `path`, then `path.1`, `path.2`... up to
// SG_WARM_ARCHIVE_FILES; a run looks pipelines up in all of them and saves what it added into
// the next. (A loaded archive with pipelines added to it can fail to save, "missing 'vertex'
// stage in pipeline no. N", and then nothing new is kept; a new one saves.)
//
// An archive made on another GPU or OS loads, or not, and has nothing that matches: then every
// pipeline is compiled as without one. Only pipelines of shaders marked with
// sg_warm_archivable(shader) are added (and every background job's): a library loaded from old
// bytecode (sokol's own shaders, built for macOS 10.12) can't be packed, and would keep the
// whole archive from being saved.
#define SG_WARM_ARCHIVE_FILES 16
// the loaded files' archives and this run's (last), looked up in that order
static NSMutableArray* _sg_warm_archives;
// this run's, which pipelines are added to, and the file it is saved to (-1: none, all taken)
static id<MTLBinaryArchive> _sg_warm_archive;
static int _sg_warm_archive_file = -1;
static bool _sg_warm_archive_dirty;
static int _sg_warm_archive_hits, _sg_warm_archive_misses;
static os_unfair_lock _sg_warm_archive_lock = OS_UNFAIR_LOCK_INIT;
// the shaders (ids) whose pipelines are added (main thread only)
static NSMutableSet* _sg_warm_archivable;

void sg_warm_archivable(uint32_t shader_id) {
    if (!_sg_warm_archivable) _sg_warm_archivable = [[NSMutableSet alloc] init];
    [_sg_warm_archivable addObject:[NSNumber numberWithUnsignedInt:shader_id]];
}

// The archive's file `i`: `path`, then `path.i`.
static NSString* _sg_warm_archive_path(const char* path, int i) {
    NSString* p = [NSString stringWithUTF8String:path];
    return i == 0 ? p : [NSString stringWithFormat:@"%@.%d", p, i];
}

int sg_warm_archive_load(const char* path) {
    if (!_sg.valid || !path) return -1;
    int taken = 0;
    @autoreleasepool {
        if (!_sg_warm_archives) _sg_warm_archives = [[NSMutableArray alloc] init];
        [_sg_warm_archives removeAllObjects];
        _sg_warm_archive_file = -1;
        MTLBinaryArchiveDescriptor* d = [[MTLBinaryArchiveDescriptor alloc] init];
        for (int i = 0; i < SG_WARM_ARCHIVE_FILES; i++) {
            NSString* p = _sg_warm_archive_path(path, i);
            if (access([p UTF8String], R_OK) != 0) {
                _sg_warm_archive_file = i;
                break;
            }
            NSError* err = nil;
            d.url = [NSURL fileURLWithPath:p];
            id<MTLBinaryArchive> a = [_sg.mtl.device newBinaryArchiveWithDescriptor:d error:&err];
            if (a) {
                [_sg_warm_archives addObject:a];
                _SG_OBJC_RELEASE(a);
                taken++;
            } else {
                fprintf(stderr, "drie: the pipeline archive %s doesn't load here (%s)\n", [p UTF8String],
                    err ? [err.localizedDescription UTF8String] : "?");
            }
        }
        d.url = nil;
        NSError* err = nil;
        _SG_OBJC_RELEASE(_sg_warm_archive);
        _sg_warm_archive = [_sg.mtl.device newBinaryArchiveWithDescriptor:d error:&err];
        if (_sg_warm_archive) [_sg_warm_archives addObject:_sg_warm_archive];
        _SG_OBJC_RELEASE(d);
    }
    _sg_warm_archive_dirty = false;
    return _sg_warm_archive ? taken : -1;
}

int sg_warm_archive_save(const char* path) {
    if (!_sg_warm_archive || !path) return -1;
    int r = 0;
    os_unfair_lock_lock(&_sg_warm_archive_lock);
    if (_sg_warm_archive_dirty && _sg_warm_archive_file < 0) {
        fprintf(stderr, "drie: the pipeline archive %s has its %d files: nothing new is saved\n", path, SG_WARM_ARCHIVE_FILES);
        _sg_warm_archive_dirty = false;
    }
    if (_sg_warm_archive_dirty) {
        @autoreleasepool {
            // (into a file of its own, then renamed: a file is whole or not there)
            NSString* p = _sg_warm_archive_path(path, _sg_warm_archive_file);
            NSString* tmp = [p stringByAppendingString:@".tmp"];
            NSError* err = nil;
            if ([_sg_warm_archive serializeToURL:[NSURL fileURLWithPath:tmp] error:&err]
                && rename([tmp UTF8String], [p UTF8String]) == 0) {
                _sg_warm_archive_dirty = false;
                r = 1;
            } else {
                fprintf(stderr, "drie: the pipeline archive can't be saved to %s (%s)\n", [p UTF8String],
                    err ? [err.localizedDescription UTF8String] : "rename failed");
                r = -1;
            }
        }
    }
    os_unfair_lock_unlock(&_sg_warm_archive_lock);
    return r;
}

// { pipelines found in the archive, pipelines compiled and added } since it was loaded.
void sg_warm_archive_counts(int* hits, int* misses) {
    os_unfair_lock_lock(&_sg_warm_archive_lock);
    if (hits) *hits = _sg_warm_archive_hits;
    if (misses) *misses = _sg_warm_archive_misses;
    os_unfair_lock_unlock(&_sg_warm_archive_lock);
}

// A render pipeline state for `rp_desc` (retained, as newRenderPipelineState's): the archive's,
// or compiled (and added to it if its shader `shader_id` is archivable; 0: a background job's,
// which is). Called from any thread; with a shader id, from the main thread.
id<MTLRenderPipelineState> _sg_warm_render_pipeline(MTLRenderPipelineDescriptor* rp_desc, NSError** err, uint32_t shader_id) {
    id<MTLDevice> device = _sg.mtl.device;
    id<MTLBinaryArchive> archive = _sg_warm_archive;
    if (!archive) return [device newRenderPipelineStateWithDescriptor:rp_desc error:err];
    rp_desc.binaryArchives = _sg_warm_archives;
    // (looked up with this run's archive held: it isn't added to meanwhile)
    os_unfair_lock_lock(&_sg_warm_archive_lock);
    id<MTLRenderPipelineState> rps = [device newRenderPipelineStateWithDescriptor:rp_desc
        options:MTLPipelineOptionFailOnBinaryArchiveMiss reflection:nil error:nil];
    if (rps) _sg_warm_archive_hits++;
    os_unfair_lock_unlock(&_sg_warm_archive_lock);
    if (rps) return rps;
    rp_desc.binaryArchives = nil;
    rps = [device newRenderPipelineStateWithDescriptor:rp_desc error:err];
    bool add = shader_id == 0 || [_sg_warm_archivable containsObject:[NSNumber numberWithUnsignedInt:shader_id]];
    if (rps && add) {
        os_unfair_lock_lock(&_sg_warm_archive_lock);
        NSError* add_err = nil;
        if ([archive addRenderPipelineFunctionsWithDescriptor:rp_desc error:&add_err]) {
            _sg_warm_archive_dirty = true;
            _sg_warm_archive_misses++;
        }
        os_unfair_lock_unlock(&_sg_warm_archive_lock);
    }
    return rps;
}

// What a job makes a stage's library from: its bytecode (a copy), or else its source.
typedef struct {
    NSString* src;
    dispatch_data_t data;
} _sg_warm_stage_t;

static bool _sg_warm_stage(const sg_shader_function* f, _sg_warm_stage_t* out) {
    out->src = nil;
    out->data = nil;
    if (f->bytecode.ptr && f->bytecode.size) {
        out->data = dispatch_data_create(f->bytecode.ptr, f->bytecode.size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        return true;
    }
    if (f->source) {
        out->src = [[NSString alloc] initWithUTF8String:f->source];
        return true;
    }
    return false;
}

static id<MTLLibrary> _sg_warm_library(id<MTLDevice> device, NSString* src, dispatch_data_t data, NSError** err) {
    if (data) return [device newLibraryWithData:data error:err];
    // (as _sg_mtl_compile_library: the invariant position kept; one set of options for all
    // the worker threads, never freed)
    static MTLCompileOptions* opts;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        opts = [[MTLCompileOptions alloc] init];
        opts.preserveInvariance = YES;
    });
    return [device newLibraryWithSource:src options:opts error:err];
}

static void _sg_warm_stage_release(_sg_warm_stage_t* st) {
    _SG_OBJC_RELEASE(st->src);
    _SG_OBJC_RELEASE(st->data);
}

// Whether `size` bytes at `ptr` load as a library on this device (a metallib built for it).
int sg_warm_library_loads(const void* ptr, size_t size) {
    if (!_sg.valid || !ptr || !size) return 0;
    int ok = 0;
    @autoreleasepool {
        dispatch_data_t data = dispatch_data_create(ptr, size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError* err = nil;
        id<MTLLibrary> lib = [_sg.mtl.device newLibraryWithData:data error:&err];
        ok = lib != nil;
        _SG_OBJC_RELEASE(lib);
        _SG_OBJC_RELEASE(data);
    }
    return ok;
}

static void _sg_warm_started_now(int job) {
    _sg_warm_started[job] = CFAbsoluteTimeGetCurrent();
    _sg_warm_given_up[job] = false;
    _sg_warm_pipeline[job] = false;
}

int sg_warm_begin(const sg_shader_desc* shader_desc, const sg_pipeline_desc* pipeline_desc) {
    if (!_sg.valid || !shader_desc || !pipeline_desc) return -1;
    sg_shader_desc sd = _sg_shader_desc_defaults(shader_desc);
    sg_pipeline_desc pd = _sg_pipeline_desc_defaults(pipeline_desc);
    _sg_warm_stage_t vst, fst;
    if (!_sg_warm_stage(&sd.vertex_func, &vst)) return -1;
    if (!_sg_warm_stage(&sd.fragment_func, &fst)) {
        _sg_warm_stage_release(&vst);
        return -1;
    }
    int job = -1;
    for (int i = 0; i < SG_WARM_JOBS; i++) {
        int free_slot = 0;
        if (atomic_compare_exchange_strong(&_sg_warm_jobs[i], &free_slot, 1)) {
            job = i;
            break;
        }
    }
    if (job < 0) {
        _sg_warm_stage_release(&vst);
        _sg_warm_stage_release(&fst);
        return -1;
    }
    if (_sg_warm_dry) {
        _sg_warm_stage_release(&vst);
        _sg_warm_stage_release(&fst);
        _sg_warm_started_now(job);
        atomic_store(&_sg_warm_jobs[job], 2);
        return job;
    }

    // the pipeline's description, as _sg_mtl_create_pipeline makes it (but for the functions)
    bool layout_active[SG_MAX_VERTEXBUFFER_BINDSLOTS] = { false };
    MTLVertexDescriptor* vtx_desc = [MTLVertexDescriptor vertexDescriptor];
    for (NSUInteger i = 0; i < SG_MAX_VERTEX_ATTRIBUTES; i++) {
        const sg_vertex_attr_state* a = &pd.layout.attrs[i];
        if (a->format == SG_VERTEXFORMAT_INVALID) break;
        layout_active[a->buffer_index] = true;
        vtx_desc.attributes[i].format = _sg_mtl_vertex_format(a->format);
        vtx_desc.attributes[i].offset = (NSUInteger)a->offset;
        vtx_desc.attributes[i].bufferIndex = _sg_mtl_vertexbuffer_bindslot((size_t)a->buffer_index);
    }
    for (NSUInteger i = 0; i < SG_MAX_VERTEXBUFFER_BINDSLOTS; i++) {
        if (!layout_active[i]) continue;
        const sg_vertex_buffer_layout_state* l = &pd.layout.buffers[i];
        const NSUInteger slot = _sg_mtl_vertexbuffer_bindslot(i);
        vtx_desc.layouts[slot].stride = (NSUInteger)l->stride;
        vtx_desc.layouts[slot].stepFunction = _sg_mtl_step_function(l->step_func);
        vtx_desc.layouts[slot].stepRate = (NSUInteger)l->step_rate;
    }
    MTLRenderPipelineDescriptor* rp_desc = [[MTLRenderPipelineDescriptor alloc] init];
    rp_desc.vertexDescriptor = vtx_desc;
    rp_desc.rasterSampleCount = (NSUInteger)pd.sample_count;
    rp_desc.alphaToCoverageEnabled = pd.alpha_to_coverage_enabled;
    rp_desc.alphaToOneEnabled = NO;
    rp_desc.rasterizationEnabled = YES;
    rp_desc.depthAttachmentPixelFormat = _sg_mtl_pixel_format(pd.depth.pixel_format);
    if (pd.depth.pixel_format == SG_PIXELFORMAT_DEPTH_STENCIL) {
        rp_desc.stencilAttachmentPixelFormat = _sg_mtl_pixel_format(pd.depth.pixel_format);
    }
    for (NSUInteger i = 0; i < (NSUInteger)pd.color_count; i++) {
        const sg_color_target_state* cs = &pd.colors[i];
        rp_desc.colorAttachments[i].pixelFormat = _sg_mtl_pixel_format(cs->pixel_format);
        rp_desc.colorAttachments[i].writeMask = _sg_mtl_color_write_mask(cs->write_mask);
        rp_desc.colorAttachments[i].blendingEnabled = cs->blend.enabled;
        rp_desc.colorAttachments[i].alphaBlendOperation = _sg_mtl_blend_op(cs->blend.op_alpha);
        rp_desc.colorAttachments[i].rgbBlendOperation = _sg_mtl_blend_op(cs->blend.op_rgb);
        rp_desc.colorAttachments[i].destinationAlphaBlendFactor = _sg_mtl_blend_factor(cs->blend.dst_factor_alpha);
        rp_desc.colorAttachments[i].destinationRGBBlendFactor = _sg_mtl_blend_factor(cs->blend.dst_factor_rgb);
        rp_desc.colorAttachments[i].sourceAlphaBlendFactor = _sg_mtl_blend_factor(cs->blend.src_factor_alpha);
        rp_desc.colorAttachments[i].sourceRGBBlendFactor = _sg_mtl_blend_factor(cs->blend.src_factor_rgb);
    }
    for (size_t i = 0; i < SG_MAX_VERTEXBUFFER_BINDSLOTS; i++) {
        if (layout_active[i]) rp_desc.vertexBuffers[_sg_mtl_vertexbuffer_bindslot(i)].mutability = MTLMutabilityImmutable;
    }
    for (size_t i = 0; i < SG_MAX_VIEW_BINDSLOTS; i++) {
        const sg_shader_view* view = &sd.views[i];
        if (view->storage_buffer.stage == SG_SHADERSTAGE_NONE) continue;
        const NSUInteger slot = view->storage_buffer.msl_buffer_n;
        if (view->storage_buffer.stage == SG_SHADERSTAGE_VERTEX) {
            rp_desc.vertexBuffers[slot].mutability = MTLMutabilityImmutable;
        } else if (view->storage_buffer.stage == SG_SHADERSTAGE_FRAGMENT && view->storage_buffer.readonly) {
            rp_desc.fragmentBuffers[slot].mutability = MTLMutabilityImmutable;
        }
    }

    // (copies: the descriptions' strings and bytes are the caller's)
    NSString* vs_src = vst.src;
    NSString* fs_src = fst.src;
    dispatch_data_t vs_data = vst.data;
    dispatch_data_t fs_data = fst.data;
    NSString* vs_entry = [[NSString alloc] initWithUTF8String:sd.vertex_func.entry ? sd.vertex_func.entry : "main0"];
    NSString* fs_entry = [[NSString alloc] initWithUTF8String:sd.fragment_func.entry ? sd.fragment_func.entry : "main0"];
    id<MTLDevice> device = _sg.mtl.device;
    _sg_warm_started_now(job);
    _sg_warm_pipeline[job] = true;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        @autoreleasepool {
            int state = 3;
            NSError* err = nil;
            double t0 = CFAbsoluteTimeGetCurrent();
            id<MTLLibrary> vs_lib = _sg_warm_library(device, vs_src, vs_data, &err);
            id<MTLLibrary> fs_lib = vs_lib ? _sg_warm_library(device, fs_src, fs_data, &err) : nil;
            id<MTLFunction> vs = vs_lib ? [vs_lib newFunctionWithName:vs_entry] : nil;
            id<MTLFunction> fs = fs_lib ? [fs_lib newFunctionWithName:fs_entry] : nil;
            double t1 = CFAbsoluteTimeGetCurrent();
            if (vs && fs) {
                rp_desc.vertexFunction = vs;
                rp_desc.fragmentFunction = fs;
                id<MTLRenderPipelineState> rps = _sg_warm_render_pipeline(rp_desc, &err, 0);
                if (rps) state = 2;
                _SG_OBJC_RELEASE(rps);
            }
            if (_sg_warm_times()) {
                fprintf(stderr, "drie warm: pipeline job %d: library %.1f ms, pipeline %.1f ms\n", job,
                    (t1 - t0) * 1000.0, (CFAbsoluteTimeGetCurrent() - t1) * 1000.0);
            }
            _SG_OBJC_RELEASE(vs);
            _SG_OBJC_RELEASE(fs);
            _SG_OBJC_RELEASE(vs_lib);
            _SG_OBJC_RELEASE(fs_lib);
            atomic_store(&_sg_warm_jobs[job], state);
        }
    });
    // (the block holds its own references to what it uses)
    _SG_OBJC_RELEASE(rp_desc);
    _sg_warm_stage_release(&vst);
    _sg_warm_stage_release(&fst);
    _SG_OBJC_RELEASE(vs_entry);
    _SG_OBJC_RELEASE(fs_entry);
    return job;
}

int sg_warm_shader_begin(const sg_shader_desc* shader_desc) {
    if (!_sg.valid || !shader_desc) return -1;
    sg_shader_desc sd = _sg_shader_desc_defaults(shader_desc);
    _sg_warm_stage_t vst, fst;
    if (!_sg_warm_stage(&sd.vertex_func, &vst)) return -1;
    if (!_sg_warm_stage(&sd.fragment_func, &fst)) {
        _sg_warm_stage_release(&vst);
        return -1;
    }
    int job = -1;
    for (int i = 0; i < SG_WARM_JOBS; i++) {
        int free_slot = 0;
        if (atomic_compare_exchange_strong(&_sg_warm_jobs[i], &free_slot, 1)) {
            job = i;
            break;
        }
    }
    if (job < 0) {
        _sg_warm_stage_release(&vst);
        _sg_warm_stage_release(&fst);
        return -1;
    }
    if (_sg_warm_dry) {
        _sg_warm_stage_release(&vst);
        _sg_warm_stage_release(&fst);
        _sg_warm_started_now(job);
        atomic_store(&_sg_warm_jobs[job], 2);
        return job;
    }
    NSString* vs_src = vst.src;
    NSString* fs_src = fst.src;
    dispatch_data_t vs_data = vst.data;
    dispatch_data_t fs_data = fst.data;
    id<MTLDevice> device = _sg.mtl.device;
    _sg_warm_started_now(job);
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            NSError* err = nil;
            double t0 = CFAbsoluteTimeGetCurrent();
            id<MTLLibrary> lib = _sg_warm_library(device, vs_src, vs_data, &err);
            bool ok = lib != nil;
            _SG_OBJC_RELEASE(lib);
            lib = ok ? _sg_warm_library(device, fs_src, fs_data, &err) : nil;
            ok = lib != nil;
            _SG_OBJC_RELEASE(lib);
            if (_sg_warm_times()) {
                fprintf(stderr, "drie warm: shader job %d: library %.1f ms\n", job, (CFAbsoluteTimeGetCurrent() - t0) * 1000.0);
            }
            atomic_store(&_sg_warm_jobs[job], ok ? 2 : 3);
        }
    });
    // (the block holds its own references to what it uses)
    _sg_warm_stage_release(&vst);
    _sg_warm_stage_release(&fst);
    return job;
}

int sg_warm_state(int job) {
    if (job < 0 || job >= SG_WARM_JOBS) return 2;
    if (_sg_warm_given_up[job]) return 2;
    int s = atomic_load(&_sg_warm_jobs[job]);
    // (DRIE_WARM_GIVE_UP=<seconds> in the environment sets the limit, to try this path)
    static double limit = 0;
    if (limit == 0) {
        const char* e = getenv("DRIE_WARM_GIVE_UP");
        limit = e ? atof(e) : SG_WARM_GIVE_UP_SECONDS;
        if (limit <= 0) limit = SG_WARM_GIVE_UP_SECONDS;
    }
    if (s == 1 && CFAbsoluteTimeGetCurrent() - _sg_warm_started[job] > limit) {
        _sg_warm_given_up[job] = true;
        fprintf(stderr, "drie: a shader compile in the background (job %d) has run %.0f s: given up on, made on the main thread instead\n",
            job, CFAbsoluteTimeGetCurrent() - _sg_warm_started[job]);
        return 2;
    }
    // DRIE_WARM_DELAY=<seconds> holds pipeline jobs back, to test slow compiles.
    // A pipeline job reports "running" until that many seconds after it started.
    // Shader-only jobs are not held back, because callers block on them.
    static double delay = -1;
    if (delay < 0) {
        const char* d = getenv("DRIE_WARM_DELAY");
        delay = d ? atof(d) : 0;
        if (delay < 0) delay = 0;
    }
    if (delay > 0 && _sg_warm_pipeline[job] && s != 0 && CFAbsoluteTimeGetCurrent() - _sg_warm_started[job] < delay) return 0;
    return s == 1 ? 0 : s == 2 ? 1 : 2;
}

// Blocks until `job` is done or failed. DRIE_WARM_DELAY does not hold this back:
// a caller that waits needs the result now.
void sg_warm_wait(int job) {
    if (job < 0 || job >= SG_WARM_JOBS) return;
    while (atomic_load(&_sg_warm_jobs[job]) == 1) {
        // (sg_warm_state says 2 once the job is given up on)
        if (sg_warm_state(job) == 2) return;
        usleep(1000);
    }
}

void sg_warm_end(int job) {
    if (job < 0 || job >= SG_WARM_JOBS) return;
    int s = atomic_load(&_sg_warm_jobs[job]);
    // (a running one is left to finish: its slot frees itself never, rather than twice; nor
    // one given up on, whose block may still store its end)
    if (_sg_warm_given_up[job]) return;
    if (s == 2 || s == 3) atomic_store(&_sg_warm_jobs[job], 0);
}

#else

int sg_warm_archive_load(const char* path) {
    (void)path;
    return -1;
}

int sg_warm_archive_save(const char* path) {
    (void)path;
    return -1;
}

void sg_warm_archive_counts(int* hits, int* misses) {
    if (hits) *hits = 0;
    if (misses) *misses = 0;
}

void sg_warm_archivable(uint32_t shader_id) {
    (void)shader_id;
}

void sg_warm_dry(int on) {
    (void)on;
}

int sg_warm_library_loads(const void* ptr, size_t size) {
    (void)ptr;
    (void)size;
    return 0;
}

int sg_warm_begin(const sg_shader_desc* shader_desc, const sg_pipeline_desc* pipeline_desc) {
    (void)shader_desc;
    (void)pipeline_desc;
    return -1;
}

int sg_warm_shader_begin(const sg_shader_desc* shader_desc) {
    (void)shader_desc;
    return -1;
}

int sg_warm_state(int job) {
    (void)job;
    return 2;
}

void sg_warm_wait(int job) {
    (void)job;
}

void sg_warm_end(int job) {
    (void)job;
}

#endif
#endif
