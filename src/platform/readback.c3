// Platform GPU readback and headless device creation backed by `platform/readback.c` (Metal on
// macOS) and `platform/readback_win.c` (D3D11 on Windows). Provides synchronous texture readback,
// headless sokol_gfx environments, and slotted async storage-buffer readbacks for `StorageBuffer`.
module drie::readback @link(env::DARWIN, "Metal.framework", "Foundation.framework");
import sokol;

// Copies `img` into `out` (`width * height * bytes_per_pixel` bytes: 4 for BGRA8/RGBA8, 8 for
// RGBA16F). Blocks until the GPU finishes, so call after `sg::commit()`.
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

// Builds a windowless `SgEnvironment` backed by the default MTLDevice on macOS or a new D3D11
// device and immediate context on Windows.
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

// Copies a native texture (`MTLTexture*` or `ID3D11Texture2D*`) through `queue`
// (`MTLCommandQueue*` or `ID3D11DeviceContext*`) into `out`.
extern fn bool read_texture(void* tex, void* queue, CInt width, CInt height, CInt bytes_per_pixel, char* out) @cname("rb_read_texture") @link(env::WIN32, "d3d11");

// Returns the platform graphics device pointer for headless rendering (`MTLDevice*` or
// `ID3D11Device*`).
extern fn void* create_device() @cname("rb_create_device") @link(env::WIN32, "d3d11");

// Returns the D3D11 immediate device context paired with `create_device()` (Windows only).
extern fn void* device_context() @cname("rb_device_context") @link(env::WIN32, "d3d11");

// Queues an async readback copy of storage buffer `buf` into `slot` (0..63). Call after encoding
// the compute work that writes `buf`, then retrieve bytes with `readback::fetch()` on a later
// frame (or use `drie::StorageBuffer.track()` and `drie::read_back()`).
fn bool track(SgBuffer buf, CInt slot)
{
    $if env::DARWIN:
        SgMtlBufferInfo info = sg::mtl_query_buffer_info(buf);
        return track_native(info.buf[info.active_slot], slot, sg::mtl_command_queue());
    $else
        $if env::EMSCRIPTEN:
            return track_native(sg::wgpu_query_buffer_info(buf).buf, slot, sg::wgpu_queue());
        $else
            return track_native(sg::d3d11_query_buffer_info(buf).buf, slot, sg::d3d11_device_context());
        $endif
    $endif
}

// Native entry point behind `readback::track()`, taking a raw `MTLBuffer*` or `ID3D11Buffer*`.
extern fn bool track_native(void* buf, CInt slot, void* queue_or_context) @cname("rb_track") @link(env::WIN32, "d3d11");

// Copies `bytes` from readback `slot` into `out`. Returns false while the GPU copy is still in
// flight unless `wait` is true (which requires the frame to have been committed).
extern fn bool fetch(CInt slot, void* out, CInt bytes, bool wait) @cname("rb_fetch") @link(env::WIN32, "d3d11");
