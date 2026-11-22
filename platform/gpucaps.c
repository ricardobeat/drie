// What the GPU can do beyond sokol's portable feature set (gpu_caps.c3): on Metal, the Apple GPU
// family and the newest Metal Shading Language version the OS compiles. Compiled as Objective-C
// on Apple; elsewhere it reports nothing (the portable tier).
#include <stdbool.h>
#include <string.h>

typedef struct {
    int apple_family;      // highest MTLGPUFamilyAppleN supported (0: not an Apple GPU)
    int msl_major;         // newest MSL version this OS compiles
    int msl_minor;
    bool metal3;           // MTLGPUFamilyMetal3
    bool metal4;           // MTLGPUFamilyMetal4
    bool unified_memory;
    bool macos;            // a Mac (Apple8 has 64-bit atomic min/max there)
    unsigned int max_threadgroup_memory;
    unsigned int max_threads_per_threadgroup;
    char name[64];
} GpucapsMetal;

#if defined(__APPLE__)
#import <Metal/Metal.h>
#include <TargetConditionals.h>

bool gpucaps_query_metal(const void* device_ptr, GpucapsMetal* out) {
    memset(out, 0, sizeof(*out));
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_ptr;
    if (!dev) return false;
    // (families as numbers, so an older SDK still builds: MTLGPUFamilyApple1 = 1001 ...)
    for (int f = 12; f >= 1; f--) {
        if ([dev supportsFamily:(MTLGPUFamily)(1000 + f)]) { out->apple_family = f; break; }
    }
    out->metal3 = [dev supportsFamily:(MTLGPUFamily)5001];
    out->metal4 = [dev supportsFamily:(MTLGPUFamily)5002];
    out->unified_memory = dev.hasUnifiedMemory;
    out->macos = TARGET_OS_OSX ? true : false;
    out->max_threadgroup_memory = (unsigned int)dev.maxThreadgroupMemoryLength;
    out->max_threads_per_threadgroup = (unsigned int)dev.maxThreadsPerThreadgroup.width;
    const char* n = dev.name.UTF8String;
    if (n) strncpy(out->name, n, sizeof(out->name) - 1);
    // the language version follows the OS (MTLLanguageVersion's availability)
    out->msl_major = 2; out->msl_minor = 4;
    if (@available(macOS 13.0, iOS 16.0, *)) { out->msl_major = 3; out->msl_minor = 0; }
    if (@available(macOS 14.0, iOS 17.0, *)) { out->msl_minor = 1; }
    if (@available(macOS 15.0, iOS 18.0, *)) { out->msl_minor = 2; }
    if (@available(macOS 26.0, iOS 26.0, *)) { out->msl_major = 4; out->msl_minor = 0; }
    if (@available(macOS 27.0, iOS 27.0, *)) { out->msl_minor = 1; }
    return true;
}
#else
bool gpucaps_query_metal(const void* device_ptr, GpucapsMetal* out) {
    (void)device_ptr;
    memset(out, 0, sizeof(*out));
    return false;
}
#endif
