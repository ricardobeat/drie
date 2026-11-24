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
// Metal only. Included at the end of sokol.c, with sokol_gfx's implementation.
#if defined(SOKOL_IMPL)

#define SG_WARM_JOBS 256

#if defined(SOKOL_METAL)
#include <stdatomic.h>

// per job: 0 free, 1 running, 2 done, 3 failed
static _Atomic int _sg_warm_jobs[SG_WARM_JOBS];
// when each job started (CFAbsoluteTime), and whether it was given up on (see sg_warm_state)
static double _sg_warm_started[SG_WARM_JOBS];
static bool _sg_warm_given_up[SG_WARM_JOBS];

// A compile in the background that has run this long is given up on: sg_warm_state says it
// failed, so that its caller makes the shader or pipeline itself, on its own thread, rather
// than leave out what draws with it for good. (Seen under a heavily loaded machine: a job whose
// block never finished, and a rock's material never drawn.) Its slot stays taken.
#define SG_WARM_GIVE_UP_SECONDS 20.0

static void _sg_warm_started_now(int job) {
    _sg_warm_started[job] = CFAbsoluteTimeGetCurrent();
    _sg_warm_given_up[job] = false;
}

int sg_warm_begin(const sg_shader_desc* shader_desc, const sg_pipeline_desc* pipeline_desc) {
    if (!_sg.valid || !shader_desc || !pipeline_desc) return -1;
    sg_shader_desc sd = _sg_shader_desc_defaults(shader_desc);
    sg_pipeline_desc pd = _sg_pipeline_desc_defaults(pipeline_desc);
    if (!sd.vertex_func.source || !sd.fragment_func.source || sd.vertex_func.bytecode.ptr || sd.fragment_func.bytecode.ptr) {
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
    if (job < 0) return -1;

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

    // (copies: the descriptions' strings are the caller's)
    NSString* vs_src = [[NSString alloc] initWithUTF8String:sd.vertex_func.source];
    NSString* fs_src = [[NSString alloc] initWithUTF8String:sd.fragment_func.source];
    NSString* vs_entry = [[NSString alloc] initWithUTF8String:sd.vertex_func.entry ? sd.vertex_func.entry : "main0"];
    NSString* fs_entry = [[NSString alloc] initWithUTF8String:sd.fragment_func.entry ? sd.fragment_func.entry : "main0"];
    id<MTLDevice> device = _sg.mtl.device;
    _sg_warm_started_now(job);
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        @autoreleasepool {
            int state = 3;
            NSError* err = nil;
            id<MTLLibrary> vs_lib = [device newLibraryWithSource:vs_src options:nil error:&err];
            id<MTLLibrary> fs_lib = vs_lib ? [device newLibraryWithSource:fs_src options:nil error:&err] : nil;
            id<MTLFunction> vs = vs_lib ? [vs_lib newFunctionWithName:vs_entry] : nil;
            id<MTLFunction> fs = fs_lib ? [fs_lib newFunctionWithName:fs_entry] : nil;
            if (vs && fs) {
                rp_desc.vertexFunction = vs;
                rp_desc.fragmentFunction = fs;
                id<MTLRenderPipelineState> rps = [device newRenderPipelineStateWithDescriptor:rp_desc error:&err];
                if (rps) state = 2;
                _SG_OBJC_RELEASE(rps);
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
    _SG_OBJC_RELEASE(vs_src);
    _SG_OBJC_RELEASE(fs_src);
    _SG_OBJC_RELEASE(vs_entry);
    _SG_OBJC_RELEASE(fs_entry);
    return job;
}

int sg_warm_shader_begin(const sg_shader_desc* shader_desc) {
    if (!_sg.valid || !shader_desc) return -1;
    sg_shader_desc sd = _sg_shader_desc_defaults(shader_desc);
    if (!sd.vertex_func.source || !sd.fragment_func.source || sd.vertex_func.bytecode.ptr || sd.fragment_func.bytecode.ptr) {
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
    if (job < 0) return -1;
    NSString* vs_src = [[NSString alloc] initWithUTF8String:sd.vertex_func.source];
    NSString* fs_src = [[NSString alloc] initWithUTF8String:sd.fragment_func.source];
    id<MTLDevice> device = _sg.mtl.device;
    _sg_warm_started_now(job);
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        @autoreleasepool {
            NSError* err = nil;
            id<MTLLibrary> lib = [device newLibraryWithSource:vs_src options:nil error:&err];
            bool ok = lib != nil;
            _SG_OBJC_RELEASE(lib);
            lib = ok ? [device newLibraryWithSource:fs_src options:nil error:&err] : nil;
            ok = lib != nil;
            _SG_OBJC_RELEASE(lib);
            atomic_store(&_sg_warm_jobs[job], ok ? 2 : 3);
        }
    });
    // (the block holds its own references to what it uses)
    _SG_OBJC_RELEASE(vs_src);
    _SG_OBJC_RELEASE(fs_src);
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
    return s == 1 ? 0 : s == 2 ? 1 : 2;
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

void sg_warm_end(int job) {
    (void)job;
}

#endif
#endif
