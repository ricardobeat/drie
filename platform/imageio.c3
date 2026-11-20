// Image decoding (PNG, JPEG, ...) through the platform's decoder (ImageIO on macOS), and lines of
// text set in the platform's fonts.
module drie::imageio;

// Decodes an encoded image into newly malloc'ed 8-bit RGBA (non-premultiplied, top row first).
// Returns null on failure. Free the result with imageio::free_pixels.
extern fn char* decode(void* data, usz len, CInt* width, CInt* height) @cname("imageio_decode");
extern fn void free_pixels(char* pixels) @cname("imageio_free");

// A line of text (UTF-8, zero-terminated) in a font family (or a CSS generic family:
// sans-serif, serif, monospace), `size` pixels tall: metrics = its width, and how far it reaches
// above and below the baseline (CoreText on macOS; not on Windows yet).
extern fn void text_metrics(ZString text, ZString family, float size, CInt bold, CInt italic, float* metrics) @cname("imageio_text_metrics");
// Draws it into `mask` (w x h, a byte a pixel, top row first: how much of each pixel it
// covers), its baseline starting at (x, y) from the top left.
extern fn void text(ZString text, ZString family, float size, CInt bold, CInt italic, char* mask, CInt w, CInt h, float x, float y) @cname("imageio_text");
