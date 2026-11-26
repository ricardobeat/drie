// Metal GPU readback and headless device helpers for drie::readback: copies textures to CPU
// memory, creates the default MTLDevice for headless rendering, and tracks non-blocking storage
// buffer readbacks across up to 64 staging slots. Compiled as Objective-C on Apple; empty elsewhere.
#if defined(__APPLE__)
#import <Metal/Metal.h>
#include <stdbool.h>
#include <string.h>

bool rb_read_texture(const void* tex_ptr, const void* queue_ptr, int width, int height, int bytes_per_pixel, unsigned char* out) {
    id<MTLTexture> tex = (__bridge id<MTLTexture>)tex_ptr;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
    if (!tex || !queue || !out) return false;
    NSUInteger row = (NSUInteger)width * (NSUInteger)bytes_per_pixel;
    id<MTLBuffer> buf = [queue.device newBufferWithLength:row * (NSUInteger)height options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromTexture:tex
              sourceSlice:0
              sourceLevel:0
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake((NSUInteger)width, (NSUInteger)height, 1)
                 toBuffer:buf
        destinationOffset:0
   destinationBytesPerRow:row
 destinationBytesPerImage:row * (NSUInteger)height];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(out, buf.contents, row * (NSUInteger)height);
    return true;
}

// Creates the default Metal device, for rendering without a window. The caller owns the reference.
void* rb_create_device(void) {
    return (__bridge_retained void*)MTLCreateSystemDefaultDevice();
}

// Buffer readback without stalling: rb_track() queues a copy of a buffer into a staging buffer
// of the slot's own, in a command buffer after the ones queued so far (the frame being
// encoded, which it waits for), and rb_fetch() copies it out once that has run. Slots are the
// caller's (0..RB_SLOTS-1).
#define RB_SLOTS 64
static struct {
    id<MTLBuffer> staging;
    id<MTLCommandBuffer> cmd;
} rb_slots[RB_SLOTS];

bool rb_track(const void* buf_ptr, int slot, const void* queue_ptr) {
    if (slot < 0 || slot >= RB_SLOTS || !buf_ptr || !queue_ptr) return false;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buf_ptr;
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
    if (!rb_slots[slot].staging || rb_slots[slot].staging.length != buf.length) {
        rb_slots[slot].staging = [queue.device newBufferWithLength:buf.length options:MTLResourceStorageModeShared];
    }
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:buf sourceOffset:0 toBuffer:rb_slots[slot].staging destinationOffset:0 size:buf.length];
    [blit endEncoding];
    [cb commit];
    rb_slots[slot].cmd = cb;
    return true;
}

bool rb_fetch(int slot, void* out, int bytes, bool wait) {
    if (slot < 0 || slot >= RB_SLOTS || !rb_slots[slot].staging || !rb_slots[slot].cmd || !out) return false;
    id<MTLCommandBuffer> cmd = rb_slots[slot].cmd;
    if (cmd.status < MTLCommandBufferStatusCompleted) {
        if (!wait) return false;
        [cmd waitUntilCompleted];
    }
    if (cmd.status != MTLCommandBufferStatusCompleted) return false;
    if ((NSUInteger)bytes > rb_slots[slot].staging.length) bytes = (int)rb_slots[slot].staging.length;
    memcpy(out, rb_slots[slot].staging.contents, (size_t)bytes);
    return true;
}
#endif // __APPLE__
