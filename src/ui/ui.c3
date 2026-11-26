// Window overlay layer and shared `sokol_debugtext` / `sokol_gl` initialization. Registered via
// `drie::add_overlay()` at startup so `render()` and post-processing `end_screen()` automatically
// draw enabled touch controls (`input::touch_controls()`) and screen transitions
// (`transitions::screen_fade`) over the final window pass.
//
//   ui::setup();
//   sdtx::font(ui::FONT_KC853);
//   ui::text(2, 2, "HELLO", 1, 1, 1);
module drie::ui;
import std::math;
import sokol, drie, drie::input, drie::transitions;

struct UiState @local {
    bool ready;
    SglPipeline blend;
}

UiState ui @local;

// Built-in `sokol_debugtext` font slot: KC85/3.
const int FONT_KC853 = 0;
// Built-in `sokol_debugtext` font slot: KC85/4.
const int FONT_KC854 = 1;
// Built-in `sokol_debugtext` font slot: Z1013.
const int FONT_Z1013 = 2;
// Built-in `sokol_debugtext` font slot: Amstrad CPC.
const int FONT_CPC   = 3;
// Built-in `sokol_debugtext` font slot: Commodore 64.
const int FONT_C64   = 4;
// Built-in `sokol_debugtext` font slot: Oric.
const int FONT_ORIC  = 5;

// Initializes `sokol_debugtext` (with all `FONT_*` slots), `sokol_gl`, and the alpha-blend
// pipeline once. Requires an active GPU context (call from a frame or after `init_headless()`).
fn void setup() {
    if (ui.ready) return;
    sdtx::setup(
        &&{
            .fonts = {
                [FONT_KC853] = sdtx::font_kc853(),
                [FONT_KC854] = sdtx::font_kc854(),
                [FONT_Z1013] = sdtx::font_z1013(),
                [FONT_CPC] = sdtx::font_cpc(),
                [FONT_C64] = sdtx::font_c64(),
                [FONT_ORIC] = sdtx::font_oric(),
            },
            .logger.func = &slog::func,
        }
    );
    sgl::setup(&&{ .logger.func = &slog::func });
    ui.blend = sgl::make_pipeline(
        &&{
            .colors[0].blend = {
                .enabled = true,
                .src_factor_rgb = SRC_ALPHA,
                .dst_factor_rgb = ONE_MINUS_SRC_ALPHA,
                .src_factor_alpha = ONE,
                .dst_factor_alpha = ONE_MINUS_SRC_ALPHA,
            },
        }
    );
    ui.ready = true;
}

// The alpha-blending sgl pipeline the overlay draws with (after setup()), for overlays of the
// application's that draw translucent shapes.
fn SglPipeline blend_pipeline() => ui.blend;

// A filled rectangle into the current sgl layer.
fn void rect(float x, float y, float w, float h, float r, float g, float b, float a) {
    sgl::begin_quads();
    sgl::c4f(r, g, b, a);
    sgl::v2f(x, y);
    sgl::v2f(x + w, y);
    sgl::v2f(x + w, y + h);
    sgl::v2f(x, y + h);
    sgl::end();
}

// One line of `text` into sdtx at a column and row (it stops at a newline).
fn void text(float col, float row, String text, float r, float g, float b, float a = 1) {
    sdtx::color4f(r, g, b, a);
    sdtx::pos(col, row);
    foreach (c : text) {
        if (c == '\n') break;
        sdtx::putc(c);
    }
}

// Draws the overlay (the touch controls once they are enabled, then the screen transitions)
// into the current pass, which must be a pass to the window. It is one of the renderer's
// overlays: render() to the window and end_screen() draw it; a game whose last pass to the
// window is its own calls drie::draw_overlays() (or this) before ending that pass.
fn void draw() {
    bool touch = input::touch_controls().enabled;
    bool trans = transitions::visible();
    if (!touch && !trans) return;
    setup();
    // in pixels (without a window: the renderer's size)
    float w     = sapp::isvalid() ? sapp::widthf() : drie::window_width();
    float h     = sapp::isvalid() ? sapp::heightf() : drie::window_height();
    // about 360 canvas units high: big enough to read on any window
    float scale = math::max(1.0f, math::round(h / 360));
    float cw    = w / scale;
    float ch    = h / scale;

    sgl::set_context(sgl::default_context());
    sgl::defaults();
    sgl::load_pipeline(ui.blend);
    sgl::matrix_mode_projection();
    sgl::ortho(0, cw, ch, 0, -1, 1);
    sdtx::set_context(sdtx::default_context());
    sdtx::canvas(cw, ch);
    sdtx::origin(0, 0);
    sdtx::font(FONT_KC853);

    sgl::layer(0);
    if (touch) input::touch_controls().draw(scale / (sapp::isvalid() ? sapp::dpi_scale() : 1));
    sgl::layer(1);
    if (trans) transitions::draw(cw, ch);

    sgl::draw_layer(0);
    sdtx::draw();
    sgl::draw_layer(1);
}

// The overlay hooks into the renderer's picture in the window as soon as the program starts.
fn void register_ui() @init @local {
    drie::add_overlay(&draw);
}
