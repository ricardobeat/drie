// Display capabilities and color space control backed by platform/display.c.
// Queries macOS EDR headroom and Display P3 wide-gamut support for the window's screen, and
// switches the swapchain color space at runtime; non-macOS platforms report an SDR sRGB display.
// Connects to Renderer.output_color_space, Renderer.hdr_output, and apply_window_colors; call
// set_display_color_space after the window opens and poll display_headroom each frame.
module drie @link(env::OS_TYPE == MACOS, "AppKit.framework", "QuartzCore.framework");
import sokol;
import std::math;

// Color spaces the window swapchain can present on macOS (mapped to platform/display.c).
enum DisplayColorSpace {
    SRGB,                       // Standard sRGB gamut and transfer curve.
    DISPLAY_P3,                 // Wide-gamut Display P3 with SDR range.
    EXTENDED_LINEAR_DISPLAY_P3, // Linear Display P3 with EDR headroom (1.0 is SDR white).
}

extern fn float drie_display_headroom(void* window) @local;
extern fn float drie_display_potential_headroom(void* window) @local;
extern fn bool drie_display_wide_gamut(void* window) @local;
extern fn bool drie_display_set_color_space(void* window, CInt space) @local;

// Current EDR headroom as a multiple of SDR white (1.0 on SDR screens). Varies with display
// brightness and ambient light, so read it each frame.
fn float display_headroom() => math::max(drie_display_headroom(sapp::macos_get_window()), 1.0f);

// Maximum EDR headroom the display supports at minimum brightness (16.0 on an XDR panel).
fn float display_potential_headroom() => math::max(drie_display_potential_headroom(sapp::macos_get_window()), 1.0f);

// True when the window's screen supports the Display P3 color gamut.
fn bool display_wide_gamut() => drie_display_wide_gamut(sapp::macos_get_window());

// Sets the color space the window swapchain interprets pixels in. Call after the window opens.
fn bool set_display_color_space(
    DisplayColorSpace space
) => drie_display_set_color_space(sapp::macos_get_window(), (CInt)space.ordinal);

// Output headroom for the renderer: display_headroom() when HDR output is active, otherwise 1.0.
fn float Renderer.output_headroom(&self) => self.hdr_on() ? display_headroom() : 1.0f;