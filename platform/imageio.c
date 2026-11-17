// Decodes images with ImageIO into straight (non-premultiplied) RGBA8.
#include <ImageIO/ImageIO.h>
#include <CoreGraphics/CoreGraphics.h>
#include <stdlib.h>
#include <string.h>

unsigned char* imageio_decode(const void* data, size_t len, int* width, int* height) {
    CFDataRef cfdata = CFDataCreateWithBytesNoCopy(NULL, data, (CFIndex)len, kCFAllocatorNull);
    if (!cfdata) return NULL;
    CGImageSourceRef src = CGImageSourceCreateWithData(cfdata, NULL);
    CFRelease(cfdata);
    if (!src) return NULL;
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    CFRelease(src);
    if (!img) return NULL;

    size_t w = CGImageGetWidth(img), h = CGImageGetHeight(img);
    unsigned char* out = malloc(w * h * 4);
    size_t bpp = CGImageGetBitsPerPixel(img), bpc = CGImageGetBitsPerComponent(img);
    size_t stride = CGImageGetBytesPerRow(img);
    CGImageAlphaInfo alpha = CGImageGetAlphaInfo(img);
    CGBitmapInfo order = CGImageGetBitmapInfo(img) & kCGBitmapByteOrderMask;
    CGColorSpaceModel model = CGColorSpaceGetModel(CGImageGetColorSpace(img));
    int done = 0;

    // Fast paths that read the decoded bytes directly, so alpha is never premultiplied.
    if (bpc == 8 && (order == kCGBitmapByteOrderDefault || order == kCGBitmapByteOrder32Big)) {
        CFDataRef raw = CGDataProviderCopyData(CGImageGetDataProvider(img));
        if (raw) {
            const unsigned char* p = CFDataGetBytePtr(raw);
            int has_alpha = alpha == kCGImageAlphaLast || alpha == kCGImageAlphaPremultipliedLast;
            if (model == kCGColorSpaceModelRGB && (bpp == 32 || bpp == 24) &&
                (bpp == 24 || alpha == kCGImageAlphaLast || alpha == kCGImageAlphaNoneSkipLast ||
                 alpha == kCGImageAlphaNone)) {
                size_t n = bpp / 8;
                for (size_t y = 0; y < h; y++) {
                    const unsigned char* s = p + y * stride;
                    unsigned char* d = out + y * w * 4;
                    for (size_t x = 0; x < w; x++, s += n, d += 4) {
                        d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
                        d[3] = (n == 4 && has_alpha) ? s[3] : 255;
                    }
                }
                done = 1;
            } else if (model == kCGColorSpaceModelMonochrome && bpp == 8) {
                for (size_t y = 0; y < h; y++) {
                    const unsigned char* s = p + y * stride;
                    unsigned char* d = out + y * w * 4;
                    for (size_t x = 0; x < w; x++, d += 4) { d[0] = d[1] = d[2] = s[x]; d[3] = 255; }
                }
                done = 1;
            }
            CFRelease(raw);
        }
    }
    if (!done) {
        // Generic path: draw into a premultiplied context, then undo the premultiplication.
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef ctx = CGBitmapContextCreate(out, w, h, 8, w * 4, cs,
            (CGBitmapInfo)kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
        CGColorSpaceRelease(cs);
        if (!ctx) { free(out); CGImageRelease(img); return NULL; }
        CGContextSetBlendMode(ctx, kCGBlendModeCopy);
        CGContextDrawImage(ctx, CGRectMake(0, 0, w, h), img);
        CGContextRelease(ctx);
        for (size_t i = 0; i < w * h; i++) {
            unsigned char* d = out + i * 4;
            if (d[3] != 0 && d[3] != 255) {
                for (int c = 0; c < 3; c++) { int v = d[c] * 255 / d[3]; d[c] = v > 255 ? 255 : v; }
            }
        }
    }
    CGImageRelease(img);
    *width = (int)w;
    *height = (int)h;
    return out;
}

void imageio_free(unsigned char* pixels) { free(pixels); }
