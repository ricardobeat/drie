// Image decoding (PNG, JPEG, ...) through the platform's decoder (ImageIO on macOS).
module drie::imageio;

// Decodes an encoded image into newly malloc'ed 8-bit RGBA (non-premultiplied, top row first).
// Returns null on failure. Free the result with imageio::free_pixels.
extern fn char* decode(void* data, usz len, CInt* width, CInt* height) @cname("imageio_decode");
extern fn void free_pixels(char* pixels) @cname("imageio_free");
