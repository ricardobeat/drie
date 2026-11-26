// Platform image decoding and single-line text rasterization backed by `platform/imageio.c`
// (ImageIO and CoreText on macOS) and `platform/imageio_win.c` (WIC on Windows).
module drie::imageio @link(env::DARWIN, "ImageIO.framework", "CoreGraphics.framework", "CoreText.framework", "CoreFoundation.framework");

// Decodes an encoded image (PNG, JPEG, WebP where supported) into newly allocated 8-bit straight
// RGBA pixels, top row first. Returns null on failure; release with `imageio::free_pixels()`.
extern fn char* decode(void* data, usz len, CInt* width, CInt* height) @cname("imageio_decode") @link(env::WIN32, "windowscodecs", "ole32");

// Frees a pixel buffer returned by `imageio::decode()`.
extern fn void free_pixels(char* pixels) @cname("imageio_free") @link(env::WIN32, "windowscodecs", "ole32");

// Measures a single UTF-8 line in `family` (font name or CSS generic: sans-serif, serif,
// monospace) at `size` pixels. Writes width, ascent, and descent into `metrics` (macOS CoreText).
extern fn void text_metrics(ZString text, ZString family, float size, CInt bold, CInt italic, float* metrics) @cname("imageio_text_metrics") @link(env::WIN32, "windowscodecs", "ole32");

// Rasterizes a single UTF-8 line into an 8-bit alpha `mask` (`w` x `h`, top row first) with the
// baseline starting at `(x, y)` from the top-left corner (macOS CoreText).
extern fn void text(ZString text, ZString family, float size, CInt bold, CInt italic, char* mask, CInt w, CInt h, float x, float y) @cname("imageio_text") @link(env::WIN32, "windowscodecs", "ole32");
