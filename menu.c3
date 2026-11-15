// Game menu, screen fade, and an overlay layer drawn on top of the window.
//
// A Menu holds the items and the input logic: arrow keys / WS move, enter or space picks,
// the toggle key (escape) opens and closes it, backspace goes back. While it is open it takes
// the keyboard and mouse, so the game's own listeners (and OrbitControls) don't see them;
// key releases still go through, so nothing stays held. The game reacts in its handler.
//
// How it looks is up to the game:
//   - `overlay = true` draws it with a plain built-in look on top of the finished frame
//     (render() to the window does it; a game with its own final pass calls draw_ui()).
//   - otherwise read the fields (items, selected, page, is_open...) and draw it yourself,
//     inside the picture if you like (into a texture that goes through post-processing).
//
//   Menu* menu = three::menu("PAUSED", &on_menu);
//   menu.add("RESUME");
//   menu.add("ABOUT", page: true, text: "Made with three.c3");
//   menu.add("QUIT");
//   menu.overlay = true;
//
//   fn void on_menu(Menu* m, MenuEvent ev, int item)
//   {
//       if (ev == SELECT && item == 0) m.close();
//       if (ev == SELECT && item == 2) sapp::request_quit();
//   }
//
// Items that are pages keep the menu open on a screen of their own (`page` is their index):
// the built-in look shows their `text`; a game can draw anything there and read the held
// arrow keys with axis_x() / axis_y(), to scrub or adjust something.
module three;
import std::collections::list, std::math, std::time::clock;
import sokol;

enum MenuEvent
{
    OPEN,       // the menu was opened
    CLOSE,      // the menu was closed (by the toggle key, or close())
    SELECT,     // enter on an item; for a page item the page is now showing
    CONFIRM,    // enter on a page
    BACK,       // escape or backspace on a page; the list shows again afterwards, unless the
                // handler moved somewhere else
}

alias MenuHandler = fn void(Menu* menu, MenuEvent event, int item);

struct MenuItem
{
    String label;
    // shown on the item's page by the built-in look, line by line
    String text;
    bool page;
    bool disabled;
}

struct Menu
{
    // While set, the toggle key does not open the menu: the game has another use for it.
    bool locked;
    String title;
    List{MenuItem} items;
    int selected;
    // the item whose page is showing, or -1 for the list
    int page;
    bool is_open;
    // opens and closes the menu
    KeyCode toggle_key;
    // draw with the built-in look in draw_ui()
    bool overlay;
    // releases the mouse while open, and locks it again on close if it was locked
    bool keep_mouse;
    MenuHandler handler;
    void* user_data;
    // held arrow keys (or WASD) while open
    bool left;
    bool right;
    bool up;
    bool down;
    // when the selection or page last changed (ns), for blinking cursors
    ulong changed_at;
    bool relock;
}

struct UiState @local
{
    bool ready;
    List{Menu*} menus;
    SglPipeline blend;
}

UiState ui @local;

const int FONT_KC853 = 0;
const int FONT_KC854 = 1;
const int FONT_Z1013 = 2;
const int FONT_CPC = 3;
const int FONT_C64 = 4;
const int FONT_ORIC = 5;

// Sets up sokol_debugtext (with the FONT_* fonts) and sokol_gl, once. Call it instead of
// sdtx::setup / sgl::setup when the game draws text or shapes of its own as well.
// Needs the GPU (call it from a frame, or after init_headless).
fn void ui_setup()
{
    if (ui.ready) return;
    sdtx::setup(&&{
        .fonts = {
            [FONT_KC853] = sdtx::font_kc853(),
            [FONT_KC854] = sdtx::font_kc854(),
            [FONT_Z1013] = sdtx::font_z1013(),
            [FONT_CPC] = sdtx::font_cpc(),
            [FONT_C64] = sdtx::font_c64(),
            [FONT_ORIC] = sdtx::font_oric(),
        },
        .logger.func = &slog::func,
    });
    sgl::setup(&&{ .logger.func = &slog::func });
    ui.blend = sgl::make_pipeline(&&{
        .colors[0].blend = {
            .enabled = true,
            .src_factor_rgb = SRC_ALPHA, .dst_factor_rgb = ONE_MINUS_SRC_ALPHA,
            .src_factor_alpha = ONE, .dst_factor_alpha = ONE_MINUS_SRC_ALPHA,
        },
    });
    ui.menus.allocator = mem;
    ui.ready = true;
}

// ----------------------------------------------------------------------------
// Menu
// ----------------------------------------------------------------------------

// A menu, registered for input: it opens with `toggle_key` from now on.
fn Menu* menu(String title = "MENU", MenuHandler handler = null, KeyCode toggle_key = ESCAPE)
{
    Menu* m = mem::new(Menu);
    *m = { .title = title, .handler = handler, .toggle_key = toggle_key, .page = -1 };
    m.items.allocator = mem;
    ui.menus.allocator = mem;
    ui.menus.push(m);
    return m;
}

// Adds an item and returns its index.
fn int Menu.add(&self, String label, bool page = false, String text = "")
{
    self.items.push({ .label = label, .page = page, .text = text });
    return (int)self.items.len() - 1;
}

fn void Menu.notify(&self, MenuEvent ev, int item)
{
    if (self.handler) self.handler(self, ev, item);
}

fn void Menu.touch(&self)
{
    self.changed_at = (ulong)clock::now();
}

// Seconds since the selection or page last changed.
fn float Menu.elapsed(&self) => (float)((double)((ulong)clock::now() - self.changed_at) / 1e9);

// Whether a blinking cursor is lit now (on at every change, then `rate` times a second).
fn bool Menu.blink(&self, float rate = 1.6) => ((int)(self.elapsed() * rate * 2)) % 2 == 0;

fn int Menu.axis_x(&self) => (self.right ? 1 : 0) - (self.left ? 1 : 0);
fn int Menu.axis_y(&self) => (self.down ? 1 : 0) - (self.up ? 1 : 0);

fn void Menu.open(&self)
{
    if (self.is_open) return;
    // only one menu at a time
    foreach (m : ui.menus) if (m != self && m.is_open) m.close();
    self.is_open = true;
    self.page = -1;
    self.selected = 0;
    if (self.items.len() && self.items[0].disabled) self.move(1);
    self.left = self.right = self.up = self.down = false;
    self.touch();
    if (!self.keep_mouse && sapp::isvalid())
    {
        self.relock = sapp::mouse_locked();
        sapp::lock_mouse(false);
    }
    self.notify(OPEN, -1);
}

fn void Menu.close(&self)
{
    if (!self.is_open) return;
    self.is_open = false;
    self.page = -1;
    self.left = self.right = self.up = self.down = false;
    if (self.relock && sapp::isvalid()) sapp::lock_mouse(true);
    self.relock = false;
    self.notify(CLOSE, -1);
}

// Shows item `item`'s page (-1: back to the list).
fn void Menu.show_page(&self, int item)
{
    self.page = item;
    self.touch();
}

// Moves the selection by `step`, over disabled items, wrapping around.
fn void Menu.move(&self, int step)
{
    int n = (int)self.items.len();
    if (!n) return;
    int s = self.selected;
    for (int i = 0; i < n; i++)
    {
        s = (s + step + n) % n;
        if (!self.items[s].disabled) break;
    }
    self.selected = s;
    self.touch();
}

fn void Menu.set_held(&self, KeyCode k, bool down)
{
    switch (k)
    {
        case LEFT: case A: self.left = down;
        case RIGHT: case D: self.right = down;
        case UP: case W: self.up = down;
        case DOWN: case S: self.down = down;
        default: break;
    }
}

// Input while the menu is open, or the toggle key while it is closed. Returns whether the
// event was taken (the game should not see it).
fn bool Menu.handle_event(&self, Event* e)
{
    if (!self.is_open)
    {
        if (self.locked) return false;
        if (e.type == KEY_DOWN && e.key_code == self.toggle_key && !e.key_repeat)
        {
            self.open();
            return true;
        }
        return false;
    }
    switch (e.type)
    {
        case KEY_DOWN:
            self.set_held(e.key_code, true);
            KeyCode k = e.key_code;
            bool enter = k == ENTER || k == KP_ENTER || k == SPACE;
            bool back = k == self.toggle_key || k == BACKSPACE;
            if (self.page >= 0)
            {
                if (e.key_repeat) break;
                int p = self.page;
                if (enter) self.notify(CONFIRM, p);
                if (back)
                {
                    self.notify(BACK, p);
                    if (self.is_open && self.page == p) self.show_page(-1);
                }
                break;
            }
            if (k == UP || k == W) self.move(-1);
            if (k == DOWN || k == S) self.move(1);
            if (e.key_repeat) break;
            if (enter && self.selected < self.items.len() && !self.items[self.selected].disabled)
            {
                int s = self.selected;
                if (self.items[s].page) self.show_page(s);
                self.notify(SELECT, s);
            }
            else if (back)
            {
                self.close();
            }
        case KEY_UP:
            self.set_held(e.key_code, false);
            return false;
        case UNFOCUSED:
            self.left = self.right = self.up = self.down = false;
            return false;
        case CHAR:
        case MOUSE_DOWN:
        case MOUSE_UP:
        case MOUSE_MOVE:
        case MOUSE_SCROLL:
            break;
        default:
            return false;
    }
    return true;
}

// Called by the renderer for every event before the listeners.
fn bool menu_filter_event(Event* e)
{
    // an open menu gets everything first
    foreach (m : ui.menus) if (m.is_open) return m.handle_event(e);
    foreach (m : ui.menus) if (m.handle_event(e)) return true;
    return false;
}

fn bool any_menu_open()
{
    foreach (m : ui.menus) if (m.is_open) return true;
    return false;
}

// ----------------------------------------------------------------------------
// Fade
// ----------------------------------------------------------------------------

alias FadeCallback = fn void();

// A fade to and from black. `level` 0 is the picture, 1 is black. draw_ui() draws it over
// everything (menus included) unless `custom` is set, for a game that applies `level`
// itself (in its own post-processing, say). Advance it with update() every frame.
struct Fade
{
    float level;
    float target;
    float speed;
    FadeCallback on_done;
    bool custom;
}

Fade screen_fade;

// Fades to black over `seconds`, then calls `done`.
fn void Fade.out(&self, float seconds, FadeCallback done = null)
{
    self.target = 1;
    self.speed = seconds > 0 ? 1 / seconds : 1e9f;
    self.on_done = done;
}

// Fades back to the picture over `seconds` (0: at once), then calls `done`.
fn void Fade.in(&self, float seconds = 0, FadeCallback done = null)
{
    self.target = 0;
    self.speed = seconds > 0 ? 1 / seconds : 1e9f;
    self.on_done = done;
    if (seconds <= 0) self.level = 0;
}

fn bool Fade.active(&self) => self.level != self.target || self.on_done != null;

fn void Fade.update(&self, float dt)
{
    float step = self.speed * dt;
    if (self.level < self.target) self.level = math::min(self.level + step, self.target);
    if (self.level > self.target) self.level = math::max(self.level - step, self.target);
    if (self.level == self.target && self.on_done)
    {
        // cleared first: the callback may start another fade
        FadeCallback cb = self.on_done;
        self.on_done = null;
        cb();
    }
}

// ----------------------------------------------------------------------------
// Overlay
// ----------------------------------------------------------------------------

fn void ui_rect(float x, float y, float w, float h, float r, float g, float b, float a) @local
{
    sgl::begin_quads();
    sgl::c4f(r, g, b, a);
    sgl::v2f(x, y);
    sgl::v2f(x + w, y);
    sgl::v2f(x + w, y + h);
    sgl::v2f(x, y + h);
    sgl::end();
}

fn void ui_text(float col, float row, String text, float r, float g, float b, float a = 1) @private
{
    sdtx::color4f(r, g, b, a);
    sdtx::pos(col, row);
    foreach (c : text)
    {
        if (c == '\n') break;
        sdtx::putc(c);
    }
}

// The built-in look: the picture dimmed, a panel in the middle with the title, the items
// and a bar on the selected one. Canvas units; a character is 8 x 8.
fn void Menu.draw_default(&self, float cw, float ch)
{
    float cols = cw / 8;
    float rows = ch / 8;
    ui_rect(0, 0, cw, ch, 0, 0, 0, 0.55f);

    // the page's lines, or the items
    List{String} lines;
    lines.tinit();
    String heading = self.title;
    if (self.page >= 0)
    {
        MenuItem* it = &self.items[self.page];
        heading = it.label;
        foreach (l : it.text.tsplit("\n")) lines.push(l);
    }
    else
    {
        foreach (it : self.items) lines.push(it.label);
    }
    usz width = heading.len;
    foreach (l : lines) width = math::max(width, (usz)l.len + 4);
    float pw = (float)math::max(width + 6, (usz)18);
    float ph = (float)lines.len() * (self.page >= 0 ? 1.0f : 1.5f) + 6;
    float px = math::floor((cols - pw) / 2);
    float py = math::floor((rows - ph) / 2);
    ui_rect(px * 8, py * 8, pw * 8, ph * 8, 0.06f, 0.07f, 0.09f, 0.92f);
    ui_rect(px * 8, py * 8, pw * 8, 1, 0.9f, 0.9f, 0.92f, 0.6f);
    ui_rect(px * 8, (py + ph) * 8 - 1, pw * 8, 1, 0.9f, 0.9f, 0.92f, 0.6f);
    ui_text(px + (pw - heading.len) / 2, py + 1.5f, heading, 1, 1, 1);

    if (self.page >= 0)
    {
        foreach (i, l : lines) ui_text(px + 3, py + 3.5f + i, l, 0.85f, 0.86f, 0.9f);
        ui_text(px + (pw - 8) / 2, py + ph - 1.5f, "ESC BACK", 0.55f, 0.56f, 0.6f);
        return;
    }
    foreach (i, it : self.items)
    {
        float y = py + 3.5f + i * 1.5f;
        if (i == self.selected)
        {
            ui_rect((px + 1) * 8, (y - 0.25f) * 8, (pw - 2) * 8, 12, 0.92f, 0.92f, 0.95f, 1);
            ui_text(px + 3, y, it.label, 0.05f, 0.05f, 0.07f);
        }
        else
        {
            float v = it.disabled ? 0.4f : 0.85f;
            ui_text(px + 3, y, it.label, v, v, v + 0.03f);
        }
    }
}

// Draws the overlay (open menus with `overlay`, the fade unless it is `custom`, the touch
// controls once they are enabled) into the current pass, which must be a pass to the window.
// render() to the window calls it; a game whose last pass to the window is its own calls it
// before ending that pass.
fn void draw_ui()
{
    bool menus;
    foreach (m : ui.menus) if (m.is_open && m.overlay) menus = true;
    bool fade = !screen_fade.custom && screen_fade.level > 0;
    bool touch = touch_controls().enabled;
    if (!menus && !fade && !touch) return;
    ui_setup();
    // in pixels (without a window: the renderer's size)
    float w = sapp::isvalid() ? sapp::widthf() : window_width();
    float h = sapp::isvalid() ? sapp::heightf() : window_height();
    // about 360 canvas units high: big enough to read on any window
    float scale = math::max(1.0f, math::round(h / 360));
    float cw = w / scale;
    float ch = h / scale;

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
    @pool()
    {
        foreach (m : ui.menus) if (m.is_open && m.overlay) m.draw_default(cw, ch);
        if (touch) draw_touch_controls(scale / (sapp::isvalid() ? sapp::dpi_scale() : 1));
    };
    sgl::layer(1);
    if (fade) ui_rect(0, 0, cw, ch, 0, 0, 0, screen_fade.level);

    sgl::draw_layer(0);
    sdtx::draw();
    sgl::draw_layer(1);
}
