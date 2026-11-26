// The platform functions for the web (emscripten): images decoded by stb_image, storage buffer
// readback through WebGPU; no GPU timer, texture readback, game controllers or text rendering
// yet (each answers that it has nothing).
#if defined(__EMSCRIPTEN__)
#include <stdbool.h>
#include <stddef.h>
#include <string.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "../third_party/stb/stb_image.h"

unsigned char* imageio_decode(const void* data, size_t len, int* width, int* height) {
    int n;
    return stbi_load_from_memory((const stbi_uc*)data, (int)len, width, height, &n, 4);
}

void imageio_free(unsigned char* pixels) { stbi_image_free(pixels); }

void imageio_text_metrics(const char* text, const char* family, float size, int bold, int italic, float* metrics) {
    (void)text; (void)family; (void)bold; (void)italic;
    metrics[0] = 0;
    metrics[1] = size * 0.8f;
    metrics[2] = size * 0.2f;
}

void imageio_text(const char* text, const char* family, float size, int bold, int italic, unsigned char* mask, int w, int h,
                  float x, float y) {
    (void)text; (void)family; (void)size; (void)bold; (void)italic; (void)x; (void)y;
    memset(mask, 0, (size_t)w * (size_t)h);
}

bool gc_poll(void* s) { (void)s; return false; }

bool gt_init(void* a, void* b) { (void)a; (void)b; return false; }
void gt_pass_begin(int slot, int index) { (void)slot; (void)index; }
bool gt_frame_end(int slot) { (void)slot; return false; }
bool gt_resolve(int slot, int count, double* ms) { (void)slot; (void)count; (void)ms; return false; }

bool rb_read_texture(void* tex, void* queue, int width, int height, int bytes_per_pixel, unsigned char* out) {
    (void)tex; (void)queue; (void)width; (void)height; (void)bytes_per_pixel; (void)out;
    return false;
}
void* rb_create_device(void) { return NULL; }
void* rb_device_context(void) { return NULL; }

// Buffer readback without stalling, as readback.c's: rb_track() copies the buffer into the
// slot's staging buffer (a submission of its own, so it sees the work submitted before it: the
// last frame's, the frame being encoded is submitted at commit) and maps it; rb_fetch() copies
// it out once mapped. A slot still waiting for its map isn't tracked again.
#include <webgpu/webgpu.h>
const void* sg_wgpu_device(void);
const void* sg_wgpu_queue(void);

#define RB_SLOTS 64
enum { RB_FREE, RB_MAPPING, RB_MAPPED };
static struct {
    WGPUBuffer staging;
    uint64_t size;
    int state;
} rb_slots[RB_SLOTS];

static void rb_mapped(WGPUMapAsyncStatus status, WGPUStringView message, void* ud1, void* ud2) {
    (void)message; (void)ud2;
    int slot = (int)(intptr_t)ud1;
    rb_slots[slot].state = status == WGPUMapAsyncStatus_Success ? RB_MAPPED : RB_FREE;
}

bool rb_track(void* buf_ptr, int slot, void* queue_ptr) {
    (void)queue_ptr;
    if (slot < 0 || slot >= RB_SLOTS || !buf_ptr) return false;
    if (rb_slots[slot].state == RB_MAPPING) return false;
    if (rb_slots[slot].state == RB_MAPPED) {
        wgpuBufferUnmap(rb_slots[slot].staging);
        rb_slots[slot].state = RB_FREE;
    }
    WGPUBuffer buf = (WGPUBuffer)buf_ptr;
    WGPUDevice device = (WGPUDevice)sg_wgpu_device();
    WGPUQueue queue = (WGPUQueue)sg_wgpu_queue();
    uint64_t size = wgpuBufferGetSize(buf);
    if (!rb_slots[slot].staging || rb_slots[slot].size != size) {
        if (rb_slots[slot].staging) wgpuBufferRelease(rb_slots[slot].staging);
        WGPUBufferDescriptor d = { .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, .size = size };
        rb_slots[slot].staging = wgpuDeviceCreateBuffer(device, &d);
        rb_slots[slot].size = size;
    }
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(enc, buf, 0, rb_slots[slot].staging, 0, size);
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(enc);
    rb_slots[slot].state = RB_MAPPING;
    WGPUBufferMapCallbackInfo info = { .mode = WGPUCallbackMode_AllowSpontaneous, .callback = rb_mapped, .userdata1 = (void*)(intptr_t)slot };
    wgpuBufferMapAsync(rb_slots[slot].staging, WGPUMapMode_Read, 0, (size_t)size, info);
    return true;
}

// (`wait` can't be honoured: the browser maps buffers between frames)
bool rb_fetch(int slot, void* out, int bytes, bool wait) {
    (void)wait;
    if (slot < 0 || slot >= RB_SLOTS || rb_slots[slot].state != RB_MAPPED || !out) return false;
    if ((uint64_t)bytes > rb_slots[slot].size) bytes = (int)rb_slots[slot].size;
    const void* p = wgpuBufferGetConstMappedRange(rb_slots[slot].staging, 0, (size_t)rb_slots[slot].size);
    if (!p) return false;
    memcpy(out, p, (size_t)bytes);
    wgpuBufferUnmap(rb_slots[slot].staging);
    rb_slots[slot].state = RB_FREE;
    return true;
}

bool gpucaps_query_metal(void* device, void* out) { (void)device; (void)out; return false; }

#endif
