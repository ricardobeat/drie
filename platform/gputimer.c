// Per-pass GPU timing on Metal for drie::gpu_timer: allocates one MTLCounterSampleBuffer per
// frame in flight, samples pass encoder stage boundaries via sokol's sg_mtl_set_pass_timestamps
// patch, and resolves elapsed ms once the command buffer completes. Compiled as Objective-C (ARC)
// on Apple; platform/gputimer_win.c and platform/gputimer_gl.c provide D3D11 and GL equivalents.
#if defined(__APPLE__)
#import <Metal/Metal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>

#define GT_SLOTS 4
#define GT_MAX_PASSES 128

extern const void* sg_mtl_command_buffer(void);
extern void sg_mtl_set_pass_timestamps(const void* sample_buffer, int start_index, int end_index);

static struct {
    bool ready;
    id<MTLDevice> device;
    id<MTLCounterSampleBuffer> buf[GT_SLOTS];
    atomic_int done[GT_SLOTS];
    int passes[GT_SLOTS];
    // GPU ticks to ns: from two (cpu, gpu) samples, the first at init
    MTLTimestamp cpu0, gpu0;
    double ns_per_tick;
    double cpu_ns_per_tick;
} gt;

bool gt_init(const void* device_ptr, const void* unused) {
    (void)unused;
    if (gt.ready) return true;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_ptr;
    if (!dev) return false;
    if (@available(macOS 11.0, iOS 14.0, *)) {
        if (![dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return false;
    } else {
        return false;
    }
    id<MTLCounterSet> ts = nil;
    for (id<MTLCounterSet> cs in dev.counterSets) {
        if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) { ts = cs; break; }
    }
    if (!ts) return false;
    for (int i = 0; i < GT_SLOTS; i++) {
        MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet = ts;
        d.storageMode = MTLStorageModeShared;
        d.sampleCount = GT_MAX_PASSES * 2;
        d.label = @"drie-gpu-timer";
        NSError* err = nil;
        gt.buf[i] = [dev newCounterSampleBufferWithDescriptor:d error:&err];
        if (!gt.buf[i]) return false;
        atomic_store(&gt.done[i], 0);
        gt.passes[i] = 0;
    }
    gt.device = dev;
    [dev sampleTimestamps:&gt.cpu0 gpuTimestamp:&gt.gpu0];
    // (sampleTimestamps' CPU side is in ns, CLOCK_UPTIME_RAW: not mach ticks)
    gt.cpu_ns_per_tick = 1.0;
    gt.ns_per_tick = 1.0;
    gt.ready = true;
    return true;
}

// Before sg_begin_pass: pass `index` of the frame in `slot` gets its two samples.
void gt_pass_begin(int slot, int index) {
    if (!gt.ready || index >= GT_MAX_PASSES) return;
    sg_mtl_set_pass_timestamps((__bridge const void*)gt.buf[slot], index * 2, index * 2 + 1);
    gt.passes[slot] = index + 1;
}

// After sg_end_pass (nothing on Metal: the encoder's end sample is taken already).
void gt_pass_end(int slot, int index) { (void)slot; (void)index; }

// Before sg_commit: the slot is ready to read once the frame's command buffer has completed.
// False when no pass was timed (or there is no command buffer).
bool gt_frame_end(int slot) {
    if (!gt.ready) return false;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)sg_mtl_command_buffer();
    if (!cb || gt.passes[slot] == 0) return false;
    atomic_store(&gt.done[slot], 0);
    int s = slot;
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
        (void)b;
        atomic_store(&gt.done[s], 1);
    }];
    return true;
}

// The ms of each pass of the frame in `slot` into ms[0..count-1] (a pass that wasn't sampled is
// -1). Returns false while the GPU hasn't finished it.
bool gt_resolve(int slot, int count, double* ms) {
    if (!gt.ready || !atomic_load(&gt.done[slot])) return false;
    if (count > gt.passes[slot]) count = gt.passes[slot];
    if (count <= 0) return true;
    // recalibrate GPU ticks against the CPU clock once enough time has passed
    MTLTimestamp c1, g1;
    [gt.device sampleTimestamps:&c1 gpuTimestamp:&g1];
    if (g1 > gt.gpu0 + 1000000 && c1 > gt.cpu0) {
        gt.ns_per_tick = (double)(c1 - gt.cpu0) * gt.cpu_ns_per_tick / (double)(g1 - gt.gpu0);
    }
    NSData* data = [gt.buf[slot] resolveCounterRange:NSMakeRange(0, (NSUInteger)count * 2)];
    if (!data) return false;
    const MTLCounterResultTimestamp* r = (const MTLCounterResultTimestamp*)data.bytes;
    NSUInteger n = data.length / sizeof(MTLCounterResultTimestamp);
    for (int i = 0; i < count; i++) {
        ms[i] = -1.0;
        if ((NSUInteger)(i * 2 + 1) >= n) continue;
        uint64_t a = r[i * 2].timestamp, b = r[i * 2 + 1].timestamp;
        if (a == MTLCounterErrorValue || b == MTLCounterErrorValue || a == 0 || b < a) continue;
        ms[i] = (double)(b - a) * gt.ns_per_tick * 1e-6;
    }
    atomic_store(&gt.done[slot], 0);
    gt.passes[slot] = 0;
    return true;
}
#endif
