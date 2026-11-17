module drie::readback;
import sokol;

// Reads `img` into `out` (width * height * bytes_per_pixel bytes): 4 for a BGRA8/RGBA8 sokol
// image, 8 for RGBA16F. Waits for the GPU, so call it after sg::commit().
fn bool read_image(SgImage img, CInt width, CInt height, char* out, CInt bytes_per_pixel = 4)
{
    $if env::DARWIN:
        SgMtlImageInfo info = sg::mtl_query_image_info(img);
        return read_texture(info.tex[info.active_slot], sg::mtl_command_queue(), width, height, bytes_per_pixel, out);
    $else
        SgD3d11ImageInfo info = sg::d3d11_query_image_info(img);
        return read_texture(info.tex2d, sg::d3d11_device_context(), width, height, bytes_per_pixel, out);
    $endif
}

// The GPU setup for sokol_gfx without a window: the system's default MTLDevice* on macOS, a new
// D3D11 device and its immediate context on Windows.
fn SgEnvironment headless_environment()
{
    SgEnvironment env = { .defaults = { .color_format = BGRA8, .depth_format = DEPTH_STENCIL, .sample_count = 1 } };
    $if env::DARWIN:
        env.metal.device = create_device();
    $else
        env.d3d11.device = create_device();
        env.d3d11.device_context = device_context();
    $endif
    return env;
}

// Copies a native texture (MTLTexture* / ID3D11Texture2D*) through `queue` (an MTLCommandQueue* /
// ID3D11DeviceContext*) into `out`.
extern fn bool read_texture(void* tex, void* queue, CInt width, CInt height, CInt bytes_per_pixel, char* out) @cname("rb_read_texture");
extern fn void* create_device() @cname("rb_create_device");
extern fn void* device_context() @cname("rb_device_context"); // Windows only

// Readback of a storage buffer without waiting for the GPU (see compute.c3). Call track()
// after the work that writes `buf` is encoded: it queues a copy of the buffer after it. Then
// fetch() in a later frame: false while the GPU has not got that far, unless `wait` (which
// needs the frame committed). Up to 64 slots, the caller's to number.
fn bool track(SgBuffer buf, CInt slot)
{
    $if env::DARWIN:
        SgMtlBufferInfo info = sg::mtl_query_buffer_info(buf);
        return track_native(info.buf[info.active_slot], slot, sg::mtl_command_queue());
    $else
        return track_native(sg::d3d11_query_buffer_info(buf).buf, slot, sg::d3d11_device_context());
    $endif
}

extern fn bool track_native(void* buf, CInt slot, void* queue_or_context) @cname("rb_track");
extern fn bool fetch(CInt slot, void* out, CInt bytes, bool wait) @cname("rb_fetch");
