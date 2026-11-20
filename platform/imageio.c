// Decodes images with ImageIO into straight (non-premultiplied) RGBA8.
#include <ImageIO/ImageIO.h>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreText/CoreText.h>
#include <strings.h>
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

// A line of text (UTF-8) in `family` (a font's family or full name, or a CSS generic family),
// `size` pixels tall, set with CoreText. width, ascent and descent: its advance and how far it
// reaches above and below the baseline, in pixels.
static CTLineRef text_line(const char* text, const char* family, float size, int bold, int italic) {
    const char* name = family;
    if (!strcasecmp(family, "sans-serif") || !strcasecmp(family, "system-ui")) name = "Helvetica";
    else if (!strcasecmp(family, "serif")) name = "Times";
    else if (!strcasecmp(family, "monospace")) name = "Menlo";
    CFStringRef fname = CFStringCreateWithCString(NULL, name, kCFStringEncodingUTF8);
    CTFontRef font = CTFontCreateWithName(fname, size, NULL);
    CFRelease(fname);
    CTFontSymbolicTraits traits = (bold ? kCTFontBoldTrait : 0) | (italic ? kCTFontItalicTrait : 0);
    if (traits) {
        CTFontRef styled = CTFontCreateCopyWithSymbolicTraits(font, size, NULL, traits, traits);
        if (styled) {
            CFRelease(font);
            font = styled;
        }
    }
    CFStringRef str = CFStringCreateWithCString(NULL, text, kCFStringEncodingUTF8);
    if (!str) str = CFStringCreateWithCString(NULL, "", kCFStringEncodingUTF8);
    CFStringRef keys[] = { kCTFontAttributeName, kCTForegroundColorFromContextAttributeName };
    CFTypeRef values[] = { font, kCFBooleanTrue };
    CFDictionaryRef attrs = CFDictionaryCreate(NULL, (const void**)keys, (const void**)values, 2,
                                               &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFAttributedStringRef astr = CFAttributedStringCreate(NULL, str, attrs);
    CTLineRef line = CTLineCreateWithAttributedString(astr);
    CFRelease(astr);
    CFRelease(attrs);
    CFRelease(str);
    CFRelease(font);
    return line;
}

void imageio_text_metrics(const char* text, const char* family, float size, int bold, int italic, float* metrics) {
    CTLineRef line = text_line(text, family, size, bold, italic);
    CGFloat ascent, descent, leading;
    metrics[0] = (float)CTLineGetTypographicBounds(line, &ascent, &descent, &leading);
    metrics[1] = (float)ascent;
    metrics[2] = (float)descent;
    CFRelease(line);
}

// Draws the line into `mask` (w x h, 8 bits a pixel, top row first: how much of each pixel the
// glyphs cover, antialiased), its baseline starting at (x, y) from the top left.
void imageio_text(const char* text, const char* family, float size, int bold, int italic, unsigned char* mask, int w,
                  int h, float x, float y) {
    CTLineRef line = text_line(text, family, size, bold, italic);
    CGContextRef ctx = CGBitmapContextCreate(mask, (size_t)w, (size_t)h, 8, (size_t)w, NULL, (CGBitmapInfo)kCGImageAlphaOnly);
    if (ctx) {
        CGContextSetShouldAntialias(ctx, true);
        CGContextSetGrayFillColor(ctx, 1.0, 1.0);
        CGContextSetTextPosition(ctx, x, (CGFloat)h - y);
        CTLineDraw(line, ctx);
        CGContextRelease(ctx);
    }
    CFRelease(line);
}
