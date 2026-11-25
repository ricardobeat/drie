// drie's addition to sokol_gfx (not upstream): ray-traced shadows of rectangle lights with
// Metal's ray tracing, which sokol doesn't expose.
//
//   if (sg_rt_supported()) ...                     // 0: not Metal, or a GPU without it
//   int g = sg_rt_geometry(0, vbuf, stride, vertex_count, ibuf, index_count);
//   sg_rt_geometry(g, ...);                        // its vertices changed: refitted
//   sg_rt_geometry_free(g);
//   sg_rt_trace(&desc);                            // between passes, after the depth is drawn
//
// A geometry is a primitive acceleration structure over a sokol vertex buffer (float3 positions
// at the start of each vertex) and a uint32 index buffer, built (or refitted) in the frame's
// command buffer by the next sg_rt_trace. sg_rt_trace builds an instance acceleration structure
// over desc.instances, then for each pixel of desc.depth and each of up to four lights casts
// rays to points on the light's rectangle (two a pixel a frame, stratified and turned every
// frame), and denoises what they saw:
//   - accumulated over frames (desc.history), the history reprojected with the last frame's
//     camera and clamped to what this frame's rays saw round the pixel (no trails behind what
//     moves), dropped where the depth doesn't match;
//   - blurred across the surface (depth and normal aware) by the width of the penumbra there:
//     the light's size times the distance to what blocks it over the distance from that to
//     the light.
// desc.mask gets each light's visible part, 0..1, in r, g, b and a.
//
// Metal only (macOS 11+, Apple silicon or a GPU with ray tracing). Included at the end of
// sokol.c, with sokol_gfx's implementation.
#if defined(SOKOL_IMPL)

#define SG_RT_MAX_LIGHTS 4
#define SG_RT_MAX_GEOMETRIES 8192

typedef struct sg_rt_light {
    float center[4];  // xyz: the rectangle's centre
    float hw[4];      // xyz: half its width, along it
    float hh[4];      // xyz: half its height, along it (it shines toward cross(hh, hw))
} sg_rt_light;

typedef struct sg_rt_instance {
    float columns[4][3];  // object to world: the matrix's four columns, xyz
    int geometry;         // sg_rt_geometry's
} sg_rt_instance;

typedef struct sg_rt_trace_desc {
    sg_image depth;       // the frame's depth (DEPTH)
    sg_image normal;      // the surfaces' normals: zw octahedral (drie's G-buffer target 3)
    sg_image raw;         // RGBA16F storage images the size of depth: this frame's rays,
    sg_image blocker;     //   the distance to what they hit,
    sg_image history[2];  //   accumulated (written to history[cur], read from the other),
    sg_image dist[2];     //   R32F: each pixel's distance from the eye with them,
    sg_image mask;        //   and the result
    int width, height;
    int cur;
    int frame;
    int reset;            // the history is not this view's
    float inv_view_proj[16];  // clip -> world - eye
    float prev_view_proj[16]; // last frame's world - prev_eye -> clip
    float eye[4];
    float prev_eye[4];
    float px_scale;       // pixels per m at 1 m from the eye
    int zero_to_one;      // depth 0..1 (else -1..1)
    int flip_y;           // uv's v down from the top
    const sg_rt_instance* instances;
    int instance_count;
    sg_rt_light lights[SG_RT_MAX_LIGHTS];
    int light_count;
} sg_rt_trace_desc;

#if defined(SOKOL_METAL)

typedef struct {
    id<MTLAccelerationStructure> as;
    MTLPrimitiveAccelerationStructureDescriptor* desc;
    int state;  // 0 free, 1 to build, 2 to refit, 3 built
} _sg_rt_geometry_t;

static struct {
    int checked, supported;
    _sg_rt_geometry_t geoms[SG_RT_MAX_GEOMETRIES];
    id<MTLComputePipelineState> trace, temporal, blur;
    // per frame in flight: the instances and their structure
    id<MTLBuffer> inst_buf[3];
    id<MTLAccelerationStructure> tlas[3];
    NSUInteger tlas_size[3];
    id<MTLBuffer> scratch;
    int ring;
} _sg_rt;

static const char* _sg_rt_msl =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"using namespace raytracing;\n"
"struct Light { float4 c; float4 hw; float4 hh; };\n"
"struct Params {\n"
"    float4x4 inv_vp; float4x4 prev_vp; float4 eye; float4 prev_eye;\n"
"    float4 info;   // width, height, frame, px_scale\n"
"    int4 flags;    // zero_to_one, flip_y, lights, reset\n"
"    Light lights[4];\n"
"};\n"
"static float3 oct_decode(float2 e) {\n"
"    float3 n = float3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));\n"
"    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);\n"
"    return normalize(n);\n"
"}\n"
"static float3 world_rel(uint2 p, float d, constant Params& P) {\n"
"    float2 ndc = (float2(p) + 0.5) / P.info.xy * 2.0 - 1.0;\n"
"    if (P.flags.y != 0) ndc.y = -ndc.y;\n"
"    float4 r = P.inv_vp * float4(ndc, P.flags.x != 0 ? d : d * 2.0 - 1.0, 1.0);\n"
"    return r.xyz / r.w;\n"
"}\n"
"static float hash(uint2 p, uint s) {\n"
"    uint h = p.x * 1973u + p.y * 9277u + s * 26699u;\n"
"    h = (h ^ (h >> 16)) * 0x7feb352du; h = (h ^ (h >> 15)) * 0x846ca68bu; h ^= h >> 16;\n"
"    return float(h & 0xffffffu) / 16777216.0;\n"
"}\n"
"kernel void rt_trace(depth2d<float, access::read> depth [[texture(0)]],\n"
"                     texture2d<float, access::read> normal [[texture(1)]],\n"
"                     texture2d<float, access::write> raw [[texture(2)]],\n"
"                     texture2d<float, access::write> blocker [[texture(3)]],\n"
"                     constant Params& P [[buffer(0)]],\n"
"                     instance_acceleration_structure as [[buffer(1)]],\n"
"                     uint2 gid [[thread_position_in_grid]]) {\n"
"    if (gid.x >= uint(P.info.x) || gid.y >= uint(P.info.y)) return;\n"
"    float d = depth.read(gid);\n"
"    if (d >= 1.0) { raw.write(float4(1.0), gid); blocker.write(float4(0.0), gid); return; }\n"
"    float3 rel = world_rel(gid, d, P);\n"
"    float3 p = rel + P.eye.xyz;\n"
"    float3 n = oct_decode(normal.read(gid).zw);\n"
"    float far = length(rel);\n"
"    float3 origin = p + n * (0.003 + 0.002 * far);\n"
"    intersector<triangle_data, instancing> isect;\n"
"    isect.accept_any_intersection(true);\n"
"    isect.assume_geometry_type(geometry_type::triangle);\n"
"    isect.force_opacity(forced_opacity::opaque);\n"
"    float4 seen = float4(1.0);\n"
"    float4 hit = float4(0.0);\n"
"    uint frame = uint(P.info.z);\n"
"    for (int i = 0; i < P.flags.z; i++) {\n"
"        Light L = P.lights[i];\n"
"        float3 front = cross(L.hh.xyz, L.hw.xyz);\n"
"        if (dot(p - L.c.xyz, front) <= 0.0) continue;\n"
"        float vis = 0.0;\n"
"        float far_hit = 0.0;\n"
"        float hits = 0.0;\n"
"        for (int k = 0; k < 2; k++) {\n"
"            // two strata of the rectangle, a point in each at random, turned every frame\n"
"            float u = (hash(gid, frame * 8u + uint(i * 2 + k)) + float(k)) * 0.5;\n"
"            float v = fract(hash(gid.yx, frame * 8u + uint(i * 2 + k) + 3u) + 0.6180339 * float(frame & 63u));\n"
"            float3 q = L.c.xyz + L.hw.xyz * (u * 2.0 - 1.0) + L.hh.xyz * (v * 2.0 - 1.0);\n"
"            float3 to = q - origin;\n"
"            float len = length(to);\n"
"            ray r;\n"
"            r.origin = origin;\n"
"            r.direction = to / len;\n"
"            r.min_distance = 0.0;\n"
"            r.max_distance = max(len - 0.03, 0.0);\n"
"            intersection_result<triangle_data, instancing> res = isect.intersect(r, as, 0xffu);\n"
"            if (res.type == intersection_type::none) vis += 0.5;\n"
"            else { far_hit += res.distance; hits += 1.0; }\n"
"        }\n"
"        seen[i] = vis;\n"
"        hit[i] = hits > 0.0 ? far_hit / hits : 0.0;\n"
"    }\n"
"    raw.write(seen, gid);\n"
"    blocker.write(hit, gid);\n"
"}\n"
"kernel void rt_temporal(depth2d<float, access::read> depth [[texture(0)]],\n"
"                        texture2d<float, access::read> raw [[texture(1)]],\n"
"                        texture2d<float, access::sample> history_in [[texture(2)]],\n"
"                        texture2d<float, access::read> dist_in [[texture(3)]],\n"
"                        texture2d<float, access::write> history_out [[texture(4)]],\n"
"                        texture2d<float, access::write> dist_out [[texture(5)]],\n"
"                        constant Params& P [[buffer(0)]],\n"
"                        uint2 gid [[thread_position_in_grid]]) {\n"
"    if (gid.x >= uint(P.info.x) || gid.y >= uint(P.info.y)) return;\n"
"    float d = depth.read(gid);\n"
"    float4 now = raw.read(gid);\n"
"    if (d >= 1.0) { history_out.write(now, gid); dist_out.write(float4(0.0), gid); return; }\n"
"    float3 rel = world_rel(gid, d, P);\n"
"    float dist = length(rel);\n"
"    dist_out.write(float4(dist), gid);\n"
"    // what the rays saw round the pixel: the history is held within it\n"
"    float4 lo = now;\n"
"    float4 hi = now;\n"
"    int2 size = int2(P.info.xy);\n"
"    for (int y = -1; y <= 1; y++) {\n"
"        for (int x = -1; x <= 1; x++) {\n"
"            float4 s = raw.read(uint2(clamp(int2(gid) + int2(x, y), int2(0), size - 1)));\n"
"            lo = min(lo, s);\n"
"            hi = max(hi, s);\n"
"        }\n"
"    }\n"
"    float4 out = now;\n"
"    if (P.flags.w == 0) {\n"
"        float3 p = rel + P.eye.xyz;\n"
"        float4 c = P.prev_vp * float4(p - P.prev_eye.xyz, 1.0);\n"
"        if (c.w > 0.0) {\n"
"            float2 uv = c.xy / c.w * 0.5 + 0.5;\n"
"            if (P.flags.y != 0) uv.y = 1.0 - uv.y;\n"
"            if (all(uv > 0.0) && all(uv < 1.0)) {\n"
"                uint2 at = uint2(uv * P.info.xy);\n"
"                float was = dist_in.read(at).x;\n"
"                float there = length(p - P.prev_eye.xyz);\n"
"                if (abs(was - there) < 0.03 * there) {\n"
"                    constexpr sampler lin(filter::linear, address::clamp_to_edge);\n"
"                    float4 h = clamp(history_in.sample(lin, uv), lo, hi);\n"
"                    out = mix(h, now, 0.15);\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"    history_out.write(out, gid);\n"
"}\n"
"kernel void rt_blur(depth2d<float, access::read> depth [[texture(0)]],\n"
"                    texture2d<float, access::read> normal [[texture(1)]],\n"
"                    texture2d<float, access::read> history [[texture(2)]],\n"
"                    texture2d<float, access::read> blocker [[texture(3)]],\n"
"                    texture2d<float, access::read> dist [[texture(4)]],\n"
"                    texture2d<float, access::write> mask [[texture(5)]],\n"
"                    constant Params& P [[buffer(0)]],\n"
"                    uint2 gid [[thread_position_in_grid]]) {\n"
"    if (gid.x >= uint(P.info.x) || gid.y >= uint(P.info.y)) return;\n"
"    float d = depth.read(gid);\n"
"    float4 h = history.read(gid);\n"
"    if (d >= 1.0) { mask.write(h, gid); return; }\n"
"    float3 rel = world_rel(gid, d, P);\n"
"    float3 p = rel + P.eye.xyz;\n"
"    float here = length(rel);\n"
"    float3 n = oct_decode(normal.read(gid).zw);\n"
"    // each light's penumbra here, in pixels: its size times the distance to the blocker over\n"
"    // the blocker's distance to the light\n"
"    float4 b = blocker.read(gid);\n"
"    float4 r = float4(1.0);\n"
"    float widest = 1.0;\n"
"    for (int i = 0; i < P.flags.z; i++) {\n"
"        Light L = P.lights[i];\n"
"        float size = length(L.hw.xyz) + length(L.hh.xyz);\n"
"        float to_light = length(L.c.xyz - p);\n"
"        float w = b[i] > 0.0 ? size * b[i] / max(to_light - b[i], 0.05) : size * 0.25;\n"
"        r[i] = clamp(w * P.info.w / here * 0.5, 1.0, 10.0);\n"
"        widest = max(widest, r[i]);\n"
"    }\n"
"    float4 sum = h;\n"
"    float4 wsum = float4(1.0);\n"
"    int2 size = int2(P.info.xy);\n"
"    float turn = fract(sin(dot(float2(gid), float2(12.9898, 78.233))) * 43758.5453) * 6.2831853;\n"
"    for (int k = 0; k < 16; k++) {\n"
"        float a = float(k) * 2.39996 + turn;\n"
"        float rr = sqrt((float(k) + 0.5) / 16.0) * widest;\n"
"        float2 o = float2(cos(a), sin(a)) * rr;\n"
"        int2 at = clamp(int2(gid) + int2(round(o)), int2(0), size - 1);\n"
"        float there = dist.read(uint2(at)).x;\n"
"        float3 m = oct_decode(normal.read(uint2(at)).zw);\n"
"        float same = exp(-abs(there - here) / (0.01 * here + 0.005)) * max(dot(n, m) * 4.0 - 3.0, 0.0);\n"
"        float4 w = same * exp(-(rr * rr) / (r * r));\n"
"        sum += history.read(uint2(at)) * w;\n"
"        wsum += w;\n"
"    }\n"
"    mask.write(sum / wsum, gid);\n"
"}\n";

int sg_rt_supported(void) {
    if (_sg_rt.checked) return _sg_rt.supported;
    _sg_rt.checked = 1;
    if (!_sg.valid || _sg.backend != SG_BACKEND_METAL_MACOS) return 0;
    if (getenv("DRIE_NO_RAY_TRACING")) return 0;
    if (@available(macOS 11.0, *)) {
        if (![_sg.mtl.device supportsRaytracing]) return 0;
        @autoreleasepool {
            NSError* err = nil;
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            id<MTLLibrary> lib = [_sg.mtl.device newLibraryWithSource:[NSString stringWithUTF8String:_sg_rt_msl] options:opts error:&err];
            _SG_OBJC_RELEASE(opts);
            if (!lib) {
                _SG_ERROR(METAL_SHADER_COMPILATION_FAILED);
                if (err) NSLog(@"sokol_rt: %@", err.localizedDescription);
                return 0;
            }
            const char* names[3] = { "rt_trace", "rt_temporal", "rt_blur" };
            id<MTLComputePipelineState>* out[3] = { &_sg_rt.trace, &_sg_rt.temporal, &_sg_rt.blur };
            for (int i = 0; i < 3; i++) {
                id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
                *out[i] = [_sg.mtl.device newComputePipelineStateWithFunction:f error:&err];
                _SG_OBJC_RELEASE(f);
                if (!*out[i]) {
                    if (err) NSLog(@"sokol_rt: %@", err.localizedDescription);
                    _SG_OBJC_RELEASE(lib);
                    return 0;
                }
            }
            _SG_OBJC_RELEASE(lib);
        }
        _sg_rt.supported = 1;
    }
    return _sg_rt.supported;
}

static id<MTLBuffer> _sg_rt_buffer(sg_buffer b) {
    const _sg_buffer_t* buf = _sg_lookup_buffer(b.id);
    if (!buf) return nil;
    return _sg_mtl_id(buf->mtl.buf[buf->cmn.active_slot]);
}

static id<MTLTexture> _sg_rt_texture(sg_image i) {
    const _sg_image_t* img = _sg_lookup_image(i.id);
    if (!img) return nil;
    return _sg_mtl_id(img->mtl.tex[img->cmn.active_slot]);
}

int sg_rt_geometry(int geom, sg_buffer vbuf, int stride, int vertex_count, sg_buffer ibuf, int index_count) {
    if (!sg_rt_supported()) return 0;
    id<MTLBuffer> vb = _sg_rt_buffer(vbuf);
    id<MTLBuffer> ib = _sg_rt_buffer(ibuf);
    if (!vb || !ib || index_count < 3) return geom;
    if (geom <= 0) {
        for (int i = 1; i < SG_RT_MAX_GEOMETRIES; i++) {
            if (_sg_rt.geoms[i].state == 0) { geom = i; break; }
        }
        if (geom <= 0) return 0;
    }
    _sg_rt_geometry_t* g = &_sg_rt.geoms[geom];
    bool same = g->desc && g->as;
    if (same) {
        MTLAccelerationStructureTriangleGeometryDescriptor* t = (MTLAccelerationStructureTriangleGeometryDescriptor*)g->desc.geometryDescriptors[0];
        same = t.triangleCount == (NSUInteger)(index_count / 3);
    }
    @autoreleasepool {
        MTLAccelerationStructureTriangleGeometryDescriptor* t = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        t.vertexBuffer = vb;
        t.vertexBufferOffset = 0;
        t.vertexStride = (NSUInteger)stride;
        t.indexBuffer = ib;
        t.indexType = MTLIndexTypeUInt32;
        t.triangleCount = (NSUInteger)(index_count / 3);
        t.opaque = YES;
        MTLPrimitiveAccelerationStructureDescriptor* d = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        d.geometryDescriptors = @[t];
        d.usage = MTLAccelerationStructureUsageRefit;
        _SG_OBJC_RELEASE(g->desc);
        g->desc = [d retain];
        if (!same) {
            MTLAccelerationStructureSizes s = [_sg.mtl.device accelerationStructureSizesWithDescriptor:d];
            _SG_OBJC_RELEASE(g->as);
            g->as = [_sg.mtl.device newAccelerationStructureWithSize:s.accelerationStructureSize];
            g->state = 1;
        } else if (g->state == 3) {
            g->state = 2;
        }
    }
    (void)vertex_count;
    return geom;
}

void sg_rt_geometry_free(int geom) {
    if (geom <= 0 || geom >= SG_RT_MAX_GEOMETRIES) return;
    _sg_rt_geometry_t* g = &_sg_rt.geoms[geom];
    _SG_OBJC_RELEASE(g->as);
    _SG_OBJC_RELEASE(g->desc);
    g->as = nil;
    g->desc = nil;
    g->state = 0;
}

// The scratch buffer, at least `size` bytes.
static id<MTLBuffer> _sg_rt_scratch(NSUInteger size) {
    if (!_sg_rt.scratch || _sg_rt.scratch.length < size) {
        _SG_OBJC_RELEASE(_sg_rt.scratch);
        _sg_rt.scratch = [_sg.mtl.device newBufferWithLength:size * 2 options:MTLResourceStorageModePrivate];
    }
    return _sg_rt.scratch;
}

int sg_rt_trace(const sg_rt_trace_desc* desc) {
    if (!sg_rt_supported() || !_sg.mtl.cmd_buffer) return 0;
    id<MTLCommandBuffer> cb = _sg.mtl.cmd_buffer;
    id<MTLDevice> dev = _sg.mtl.device;
    int ring = _sg_rt.ring = (_sg_rt.ring + 1) % 3;
    int built = 0;
    @autoreleasepool {
        // the geometries that changed, one encoder (each its own scratch: they may run at once)
        NSMutableArray* used = [NSMutableArray array];
        int* slot = (int*)calloc(SG_RT_MAX_GEOMETRIES, sizeof(int));
        id<MTLAccelerationStructureCommandEncoder> ae = nil;
        NSMutableArray* scratches = [NSMutableArray array];
        for (int i = 0; i < desc->instance_count; i++) {
            int gi = desc->instances[i].geometry;
            if (gi <= 0 || gi >= SG_RT_MAX_GEOMETRIES || slot[gi]) continue;
            _sg_rt_geometry_t* g = &_sg_rt.geoms[gi];
            if (!g->as || g->state == 0) continue;
            if (g->state == 1 || g->state == 2) {
                if (!ae) ae = [cb accelerationStructureCommandEncoder];
                MTLAccelerationStructureSizes s = [dev accelerationStructureSizesWithDescriptor:g->desc];
                NSUInteger need = g->state == 1 ? s.buildScratchBufferSize : s.refitScratchBufferSize;
                id<MTLBuffer> sc = [dev newBufferWithLength:need > 0 ? need : 16 options:MTLResourceStorageModePrivate];
                [scratches addObject:sc];
                [sc release];
                if (g->state == 1) {
                    [ae buildAccelerationStructure:g->as descriptor:g->desc scratchBuffer:sc scratchBufferOffset:0];
                } else {
                    [ae refitAccelerationStructure:g->as descriptor:g->desc destination:g->as scratchBuffer:sc scratchBufferOffset:0];
                }
                g->state = 3;
                built++;
            }
            [used addObject:g->as];
            slot[gi] = (int)used.count;
        }
        if (ae) [ae endEncoding];
        // the instances
        int n = 0;
        size_t isz = sizeof(MTLAccelerationStructureInstanceDescriptor);
        if (!_sg_rt.inst_buf[ring] || _sg_rt.inst_buf[ring].length < (NSUInteger)desc->instance_count * isz) {
            _SG_OBJC_RELEASE(_sg_rt.inst_buf[ring]);
            NSUInteger len = (NSUInteger)(desc->instance_count + 256) * isz;
            _sg_rt.inst_buf[ring] = [dev newBufferWithLength:len options:MTLResourceStorageModeShared];
        }
        MTLAccelerationStructureInstanceDescriptor* id_ = (MTLAccelerationStructureInstanceDescriptor*)_sg_rt.inst_buf[ring].contents;
        for (int i = 0; i < desc->instance_count; i++) {
            int gi = desc->instances[i].geometry;
            if (gi <= 0 || gi >= SG_RT_MAX_GEOMETRIES || !slot[gi]) continue;
            MTLAccelerationStructureInstanceDescriptor* o = &id_[n++];
            memset(o, 0, isz);
            for (int c = 0; c < 4; c++) {
                o->transformationMatrix.columns[c].x = desc->instances[i].columns[c][0];
                o->transformationMatrix.columns[c].y = desc->instances[i].columns[c][1];
                o->transformationMatrix.columns[c].z = desc->instances[i].columns[c][2];
            }
            o->options = MTLAccelerationStructureInstanceOptionOpaque | MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
            o->mask = 0xff;
            o->accelerationStructureIndex = (uint32_t)(slot[gi] - 1);
        }
        free(slot);
        id<MTLAccelerationStructure> tlas = nil;
        if (n > 0) {
            MTLInstanceAccelerationStructureDescriptor* td = [MTLInstanceAccelerationStructureDescriptor descriptor];
            td.instancedAccelerationStructures = used;
            td.instanceCount = (NSUInteger)n;
            td.instanceDescriptorBuffer = _sg_rt.inst_buf[ring];
            MTLAccelerationStructureSizes s = [dev accelerationStructureSizesWithDescriptor:td];
            if (!_sg_rt.tlas[ring] || _sg_rt.tlas_size[ring] < s.accelerationStructureSize) {
                _SG_OBJC_RELEASE(_sg_rt.tlas[ring]);
                _sg_rt.tlas_size[ring] = s.accelerationStructureSize * 3 / 2;
                _sg_rt.tlas[ring] = [dev newAccelerationStructureWithSize:_sg_rt.tlas_size[ring]];
            }
            tlas = _sg_rt.tlas[ring];
            id<MTLAccelerationStructureCommandEncoder> te = [cb accelerationStructureCommandEncoder];
            [te buildAccelerationStructure:tlas descriptor:td scratchBuffer:_sg_rt_scratch(s.buildScratchBufferSize) scratchBufferOffset:0];
            [te endEncoding];
        }
        // the parameters
        struct {
            float inv_vp[16], prev_vp[16], eye[4], prev_eye[4], info[4];
            int flags[4];
            sg_rt_light lights[SG_RT_MAX_LIGHTS];
        } P;
        memset(&P, 0, sizeof(P));
        memcpy(P.inv_vp, desc->inv_view_proj, sizeof(P.inv_vp));
        memcpy(P.prev_vp, desc->prev_view_proj, sizeof(P.prev_vp));
        memcpy(P.eye, desc->eye, sizeof(P.eye));
        memcpy(P.prev_eye, desc->prev_eye, sizeof(P.prev_eye));
        P.info[0] = (float)desc->width;
        P.info[1] = (float)desc->height;
        P.info[2] = (float)desc->frame;
        P.info[3] = desc->px_scale;
        P.flags[0] = desc->zero_to_one;
        P.flags[1] = desc->flip_y;
        P.flags[2] = n > 0 ? (desc->light_count < SG_RT_MAX_LIGHTS ? desc->light_count : SG_RT_MAX_LIGHTS) : 0;
        P.flags[3] = desc->reset;
        memcpy(P.lights, desc->lights, sizeof(P.lights));

        id<MTLTexture> depth = _sg_rt_texture(desc->depth);
        id<MTLTexture> normal = _sg_rt_texture(desc->normal);
        id<MTLTexture> raw = _sg_rt_texture(desc->raw);
        id<MTLTexture> blocker = _sg_rt_texture(desc->blocker);
        id<MTLTexture> hist_out = _sg_rt_texture(desc->history[desc->cur]);
        id<MTLTexture> hist_in = _sg_rt_texture(desc->history[1 - desc->cur]);
        id<MTLTexture> dist_out = _sg_rt_texture(desc->dist[desc->cur]);
        id<MTLTexture> dist_in = _sg_rt_texture(desc->dist[1 - desc->cur]);
        id<MTLTexture> mask = _sg_rt_texture(desc->mask);
        if (!depth || !normal || !raw || !blocker || !hist_out || !hist_in || !dist_out || !dist_in || !mask) return 0;
        MTLSize tg = MTLSizeMake(8, 8, 1);
        MTLSize grid = MTLSizeMake((NSUInteger)(desc->width + 7) / 8, (NSUInteger)(desc->height + 7) / 8, 1);
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:_sg_rt.trace];
        [ce setTexture:depth atIndex:0];
        [ce setTexture:normal atIndex:1];
        [ce setTexture:raw atIndex:2];
        [ce setTexture:blocker atIndex:3];
        [ce setBytes:&P length:sizeof(P) atIndex:0];
        if (tlas) {
            [ce setAccelerationStructure:tlas atBufferIndex:1];
            for (id<MTLAccelerationStructure> a in used) [ce useResource:a usage:MTLResourceUsageRead];
        }
        [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [ce memoryBarrierWithScope:MTLBarrierScopeTextures];
        [ce setComputePipelineState:_sg_rt.temporal];
        [ce setTexture:depth atIndex:0];
        [ce setTexture:raw atIndex:1];
        [ce setTexture:hist_in atIndex:2];
        [ce setTexture:dist_in atIndex:3];
        [ce setTexture:hist_out atIndex:4];
        [ce setTexture:dist_out atIndex:5];
        [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [ce memoryBarrierWithScope:MTLBarrierScopeTextures];
        [ce setComputePipelineState:_sg_rt.blur];
        [ce setTexture:depth atIndex:0];
        [ce setTexture:normal atIndex:1];
        [ce setTexture:hist_out atIndex:2];
        [ce setTexture:blocker atIndex:3];
        [ce setTexture:dist_out atIndex:4];
        [ce setTexture:mask atIndex:5];
        [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [ce endEncoding];
        // (the command buffer may not keep what it uses: keep this frame's scratch buffers
        // alive until it is done)
        if (scratches.count) {
            NSArray* keep = [scratches copy];
            [cb addCompletedHandler:^(id<MTLCommandBuffer> b) { (void)b; [keep release]; }];
        }
    }
    return 1 + built;
}

#else

int sg_rt_supported(void) { return 0; }
int sg_rt_geometry(int geom, sg_buffer vbuf, int stride, int vertex_count, sg_buffer ibuf, int index_count) {
    (void)geom; (void)vbuf; (void)stride; (void)vertex_count; (void)ibuf; (void)index_count;
    return 0;
}
void sg_rt_geometry_free(int geom) { (void)geom; }
int sg_rt_trace(const sg_rt_trace_desc* desc) { (void)desc; return 0; }

#endif
#endif
