// The platform functions for the web (emscripten): images decoded by stb_image; no GPU timer,
// readback, game controllers or text rendering yet (each answers that it has nothing).
#if defined(__EMSCRIPTEN__)
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

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
bool rb_track(void* buf, int slot, void* queue) { (void)buf; (void)slot; (void)queue; return false; }
bool rb_fetch(int slot, void* out, int bytes, bool wait) { (void)slot; (void)out; (void)bytes; (void)wait; return false; }

bool gpucaps_query_metal(void* device, void* out) { (void)device; (void)out; return false; }

#endif
