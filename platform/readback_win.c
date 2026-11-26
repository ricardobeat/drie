// D3D11 GPU readback and headless device helpers for drie::readback: copies textures through a
// staging resource, creates a hardware or WARP D3D11 device for headless rendering, and tracks
// non-blocking buffer readbacks across 64 staging slots. Compiles empty outside Windows.
#if defined(_WIN32)
#define COBJMACROS
#include <d3d11.h>
#include <stdbool.h>
#include <string.h>

bool rb_read_texture(void* tex_ptr, void* ctx_ptr, int width, int height, int bytes_per_pixel, unsigned char* out) {
    ID3D11Texture2D* tex = (ID3D11Texture2D*)tex_ptr;
    ID3D11DeviceContext* ctx = (ID3D11DeviceContext*)ctx_ptr;
    if (!tex || !ctx || !out) return false;
    ID3D11Device* dev = NULL;
    ID3D11Texture2D_GetDevice(tex, &dev);
    D3D11_TEXTURE2D_DESC desc;
    ID3D11Texture2D_GetDesc(tex, &desc);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    ID3D11Texture2D* staging = NULL;
    HRESULT hr = ID3D11Device_CreateTexture2D(dev, &desc, NULL, &staging);
    ID3D11Device_Release(dev);
    if (FAILED(hr)) return false;
    ID3D11DeviceContext_CopySubresourceRegion(ctx, (ID3D11Resource*)staging, 0, 0, 0, 0, (ID3D11Resource*)tex, 0, NULL);
    D3D11_MAPPED_SUBRESOURCE map;
    hr = ID3D11DeviceContext_Map(ctx, (ID3D11Resource*)staging, 0, D3D11_MAP_READ, 0, &map);
    if (FAILED(hr)) { ID3D11Texture2D_Release(staging); return false; }
    size_t row = (size_t)width * (size_t)bytes_per_pixel;
    for (int y = 0; y < height; y++) {
        memcpy(out + row * (size_t)y, (const unsigned char*)map.pData + (size_t)map.RowPitch * (size_t)y, row);
    }
    ID3D11DeviceContext_Unmap(ctx, (ID3D11Resource*)staging, 0);
    ID3D11Texture2D_Release(staging);
    return true;
}

static ID3D11DeviceContext* rb_context;

// Creates a D3D11 device for rendering without a window (the GPU, else the WARP software
// rasterizer, so it also runs on machines without one). The caller owns the reference;
// the matching immediate context is kept for rb_device_context().
void* rb_create_device(void) {
    ID3D11Device* dev = NULL;
    D3D_FEATURE_LEVEL level;
    HRESULT hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, D3D11_CREATE_DEVICE_SINGLETHREADED,
                                   NULL, 0, D3D11_SDK_VERSION, &dev, &level, &rb_context);
    if (FAILED(hr)) {
        hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_WARP, NULL, D3D11_CREATE_DEVICE_SINGLETHREADED,
                               NULL, 0, D3D11_SDK_VERSION, &dev, &level, &rb_context);
    }
    if (FAILED(hr)) return NULL;
    return dev;
}

void* rb_device_context(void) {
    return rb_context;
}

// Buffer readback without stalling: rb_track() copies a buffer into a staging buffer of the
// slot's own (on the GPU, in order with what came before), rb_fetch() maps it once the copy
// is done. Slots are the caller's (0..RB_SLOTS-1).
#define RB_SLOTS 64
static ID3D11Buffer* rb_staging[RB_SLOTS];
static ID3D11DeviceContext* rb_staging_ctx[RB_SLOTS];

bool rb_track(void* buf_ptr, int slot, void* ctx_ptr) {
    if (slot < 0 || slot >= RB_SLOTS || !buf_ptr || !ctx_ptr) return false;
    ID3D11Buffer* buf = (ID3D11Buffer*)buf_ptr;
    ID3D11DeviceContext* ctx = (ID3D11DeviceContext*)ctx_ptr;
    D3D11_BUFFER_DESC desc;
    ID3D11Buffer_GetDesc(buf, &desc);
    ID3D11Buffer* st = rb_staging[slot];
    if (st) {
        D3D11_BUFFER_DESC have;
        ID3D11Buffer_GetDesc(st, &have);
        if (have.ByteWidth != desc.ByteWidth || have.MiscFlags != desc.MiscFlags) {
            ID3D11Buffer_Release(st);
            st = NULL;
        }
    }
    if (!st) {
        ID3D11Device* dev = NULL;
        ID3D11Buffer_GetDevice(buf, &dev);
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        HRESULT hr = ID3D11Device_CreateBuffer(dev, &desc, NULL, &st);
        ID3D11Device_Release(dev);
        if (FAILED(hr)) st = NULL;
    }
    rb_staging[slot] = st;
    rb_staging_ctx[slot] = ctx;
    if (!st) return false;
    ID3D11DeviceContext_CopyResource(ctx, (ID3D11Resource*)st, (ID3D11Resource*)buf);
    return true;
}

bool rb_fetch(int slot, void* out, int bytes, bool wait) {
    if (slot < 0 || slot >= RB_SLOTS || !rb_staging[slot] || !out) return false;
    D3D11_BUFFER_DESC desc;
    ID3D11Buffer_GetDesc(rb_staging[slot], &desc);
    if ((UINT)bytes > desc.ByteWidth) bytes = (int)desc.ByteWidth;
    D3D11_MAPPED_SUBRESOURCE map;
    HRESULT hr = ID3D11DeviceContext_Map(rb_staging_ctx[slot], (ID3D11Resource*)rb_staging[slot], 0, D3D11_MAP_READ,
                                         wait ? 0 : D3D11_MAP_FLAG_DO_NOT_WAIT, &map);
    if (FAILED(hr)) return false;
    memcpy(out, map.pData, (size_t)bytes);
    ID3D11DeviceContext_Unmap(rb_staging_ctx[slot], (ID3D11Resource*)rb_staging[slot], 0);
    return true;
}
#endif // _WIN32
