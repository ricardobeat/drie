// The screen the window is on (display.c): how much brighter than SDR white it can show
// (EDR headroom, macOS) and whether it shows Display P3's colors. Elsewhere it is taken to be
// an SDR sRGB screen.
module drie;
import sokol;
import std::math;

enum DisplayColorSpace
{
    SRGB,
    DISPLAY_P3,
    // linear, with EDR: 1 is SDR white
    EXTENDED_LINEAR_DISPLAY_P3,
}

extern fn float drie_display_headroom(void* window) @local;
extern fn float drie_display_potential_headroom(void* window) @local;
extern fn bool drie_display_wide_gamut(void* window) @local;
extern fn bool drie_display_set_color_space(void* window, CInt space) @local;

// How many times SDR white the screen shows now (1 on an SDR screen). It changes with the
// screen's brightness and the light around it: read it each frame.
fn float display_headroom() => math::max(drie_display_headroom(sapp::macos_get_window()), 1.0f);

// The most it could show (at the lowest brightness): 16 on a MacBook Pro's XDR screen.
fn float display_potential_headroom() => math::max(drie_display_potential_headroom(sapp::macos_get_window()), 1.0f);

// Whether the screen shows Display P3's colors.
fn bool display_wide_gamut() => drie_display_wide_gamut(sapp::macos_get_window());

// Sets the color space the window's pixels are taken in (after it opened).
fn bool set_display_color_space(DisplayColorSpace space) => drie_display_set_color_space(sapp::macos_get_window(), (CInt)space.ordinal);

// The headroom the renderer's output has: the screen's with HDR output on, else 1.
fn float Renderer.output_headroom(&self) => self.hdr_on() ? display_headroom() : 1.0f;
