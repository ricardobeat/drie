// Queries display capabilities and sets the swapchain color space on macOS (EDR headroom,
// Display P3 gamut, and CAMetalLayer color space / EDR flags). Non-macOS targets compile stub
// fallbacks that report SDR sRGB. Used by drie::display_headroom and drie::set_display_color_space.
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#if defined(__APPLE__) && TARGET_OS_OSX
#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

static NSScreen* drie_screen_of(void* window) {
    NSWindow* w = (__bridge NSWindow*)window;
    NSScreen* s = w ? w.screen : nil;
    return s ? s : NSScreen.mainScreen;
}

float drie_display_headroom(void* window) {
    NSScreen* s = drie_screen_of(window);
    return s ? (float)s.maximumExtendedDynamicRangeColorComponentValue : 1.0f;
}

float drie_display_potential_headroom(void* window) {
    NSScreen* s = drie_screen_of(window);
    return s ? (float)s.maximumPotentialExtendedDynamicRangeColorComponentValue : 1.0f;
}

bool drie_display_wide_gamut(void* window) {
    NSScreen* s = drie_screen_of(window);
    return s ? [s canRepresentDisplayGamut:NSDisplayGamutP3] : false;
}

// space: 0 sRGB, 1 Display P3, 2 extended linear Display P3 (with EDR)
bool drie_display_set_color_space(void* window, int space) {
    NSWindow* w = (__bridge NSWindow*)window;
    if (!w) return false;
    CALayer* layer = w.contentView.layer;
    if (![layer isKindOfClass:[CAMetalLayer class]]) return false;
    CAMetalLayer* ml = (CAMetalLayer*)layer;
    CFStringRef name = space == 2 ? kCGColorSpaceExtendedLinearDisplayP3 : (space == 1 ? kCGColorSpaceDisplayP3 : kCGColorSpaceSRGB);
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(name);
    ml.colorspace = cs;
    CGColorSpaceRelease(cs);
    ml.wantsExtendedDynamicRangeContent = space == 2;
    return true;
}

#else
#include <stdbool.h>

float drie_display_headroom(void* window) { (void)window; return 1.0f; }
float drie_display_potential_headroom(void* window) { (void)window; return 1.0f; }
bool drie_display_wide_gamut(void* window) { (void)window; return false; }
bool drie_display_set_color_space(void* window, int space) { (void)window; (void)space; return false; }
#endif
