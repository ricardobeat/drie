// Windows image decoding for drie::imageio: decodes images via Windows Imaging Component (WIC)
// into straight (non-premultiplied) RGBA8, with stub text metrics and rasterization entry points.
// Compiles empty outside Windows so every target shares the same C source list.
#if defined(_WIN32)
#define COBJMACROS
#include <windows.h>
#include <wincodec.h>
#include <stdlib.h>

static IWICImagingFactory* factory;

unsigned char* imageio_decode(const void* data, size_t len, int* width, int* height) {
    if (!factory) {
        HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        (void)hr;  // already initialised (in any mode) is fine
        if (FAILED(CoCreateInstance(&CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER,
                                    &IID_IWICImagingFactory, (void**)&factory))) return NULL;
    }
    unsigned char* out = NULL;
    IWICStream* stream = NULL;
    IWICBitmapDecoder* decoder = NULL;
    IWICBitmapFrameDecode* frame = NULL;
    IWICFormatConverter* conv = NULL;
    if (FAILED(IWICImagingFactory_CreateStream(factory, &stream))) goto done;
    if (FAILED(IWICStream_InitializeFromMemory(stream, (BYTE*)data, (DWORD)len))) goto done;
    if (FAILED(IWICImagingFactory_CreateDecoderFromStream(factory, (IStream*)stream, NULL,
                                                         WICDecodeMetadataCacheOnDemand, &decoder))) goto done;
    if (FAILED(IWICBitmapDecoder_GetFrame(decoder, 0, &frame))) goto done;
    if (FAILED(IWICImagingFactory_CreateFormatConverter(factory, &conv))) goto done;
    if (FAILED(IWICFormatConverter_Initialize(conv, (IWICBitmapSource*)frame, &GUID_WICPixelFormat32bppRGBA,
                                             WICBitmapDitherTypeNone, NULL, 0.0, WICBitmapPaletteTypeCustom))) goto done;
    UINT w, h;
    if (FAILED(IWICFormatConverter_GetSize(conv, &w, &h))) goto done;
    out = malloc((size_t)w * h * 4);
    if (!out) goto done;
    if (FAILED(IWICFormatConverter_CopyPixels(conv, NULL, w * 4, w * h * 4, out))) { free(out); out = NULL; goto done; }
    *width = (int)w;
    *height = (int)h;
done:
    if (conv) IWICFormatConverter_Release(conv);
    if (frame) IWICBitmapFrameDecode_Release(frame);
    if (decoder) IWICBitmapDecoder_Release(decoder);
    if (stream) IWICStream_Release(stream);
    return out;
}

void imageio_free(unsigned char* pixels) {
    free(pixels);
}

// (text is not set on Windows yet: no width, and nothing drawn)
void imageio_text_metrics(const char* text, const char* family, float size, int bold, int italic, float* metrics) {
    metrics[0] = 0;
    metrics[1] = size * 0.8f;
    metrics[2] = size * 0.2f;
}

void imageio_text(const char* text, const char* family, float size, int bold, int italic, unsigned char* mask, int w,
                  int h, float x, float y) {
}
#endif // _WIN32
