// Renderer: owns the window and the GPU, draws a Scene from a Camera.
module three;
import std::collections::list, std::sort, std::io, std::math;
import sokol, readback;

alias AnimationLoop = fn void(double time_ms);
alias EventListener = fn void(SappEvent* event);
alias Event = SappEvent;
alias KeyCode = SappKeycode;

const int MAX_DIR_LIGHTS = 4;
const int MAX_POINT_LIGHTS = 6;
const int MAX_SPOT_LIGHTS = 8;
// spot lights whose shadow map is drawn once (shadow.auto_update off): an atlas of ATLAS_COLUMNS x 2 tiles
const int MAX_FIXED_SHADOWS = 7;
// their tiles in the atlas: this many in a row, in two rows
const int ATLAS_COLUMNS = 4;

// An offscreen image to render into, like three's WebGLRenderTarget.
// Use `texture` (a sokol texture view) to sample the result in your own passes.
struct RenderTarget
{
    int width;
    int height;
    int samples;
    SgImage color;
    SgImage resolve;
    SgImage depth;
    SgView color_view;
    SgView resolve_view;
    SgView depth_view;
    SgView texture;
    // The resolved RGBA8 image that `texture` reads from.
    SgImage texture_image;
    // For post-processing: the depth of the last render (multisampled if the target is), and
    // an attachment for drawing over the resolved picture.
    SgView depth_texture;
    SgView overlay_view;
    // see as_map
    Texture* map_texture;
}

// The picture as a Texture, for any material's map, like three's `renderTarget.texture`: a
// screen in the scene showing what another camera sees, say. It follows set_size. Call it
// once the target has a size.
//   RenderTarget* feed = three::render_target(256, 144);
//   screen_material.map = feed.as_map();
//   renderer.set_render_target(feed); renderer.render(scene, other_camera);
fn Texture* RenderTarget.as_map(&self)
{
    if (!self.map_texture)
    {
        Texture* t = mem::new(Texture);
        *t = { .repeat = { 1, 1 }, .wrap_s = CLAMP_TO_EDGE_WRAPPING, .wrap_t = CLAMP_TO_EDGE_WRAPPING, .uploaded = true };
        t.sampler = sg::make_sampler(&&{ .min_filter = LINEAR, .mag_filter = LINEAR, .wrap_u = CLAMP_TO_EDGE, .wrap_v = CLAMP_TO_EDGE });
        // a render target's first row is at the top, except in GL
        if (depth_zero_to_one())
        {
            t.repeat.y = -1;
            t.offset.y = 1;
        }
        self.map_texture = t;
    }
    self.map_texture.width = self.width;
    self.map_texture.height = self.height;
    self.map_texture.image = self.texture_image;
    self.map_texture.view = self.texture;
    return self.map_texture;
}

// samples > 1 renders with MSAA and resolves into `texture`. Must be called after the window opened.
fn RenderTarget* render_target(int width, int height, int samples = 1)
{
    RenderTarget* rt = mem::new(RenderTarget);
    rt.set_size(width, height, samples);
    return rt;
}

fn void RenderTarget.dispose(&self)
{
    if (self.texture.id) sg::destroy_view(self.texture);
    if (self.color_view.id) sg::destroy_view(self.color_view);
    if (self.resolve_view.id) sg::destroy_view(self.resolve_view);
    if (self.depth_view.id) sg::destroy_view(self.depth_view);
    if (self.depth_texture.id) sg::destroy_view(self.depth_texture);
    if (self.overlay_view.id) sg::destroy_view(self.overlay_view);
    if (self.color.id) sg::destroy_image(self.color);
    if (self.resolve.id) sg::destroy_image(self.resolve);
    if (self.depth.id) sg::destroy_image(self.depth);
    Texture* map = self.map_texture;
    *self = { .map_texture = map };
}

fn void RenderTarget.set_size(&self, int width, int height, int samples = 1)
{
    if (self.width == width && self.height == height && self.samples == samples) return;
    self.dispose();
    self.width = width;
    self.height = height;
    self.samples = samples;
    self.color = sg::make_image(&&{
        .usage = { .color_attachment = true },
        .width = width, .height = height, .sample_count = samples, .pixel_format = RGBA8,
    });
    self.depth = sg::make_image(&&{
        .usage = { .depth_stencil_attachment = true },
        .width = width, .height = height, .sample_count = samples, .pixel_format = DEPTH,
    });
    self.color_view = sg::make_view(&&{ .color_attachment = { .image = self.color } });
    self.depth_view = sg::make_view(&&{ .depth_stencil_attachment = { .image = self.depth } });
    self.texture_image = self.color;
    if (samples > 1)
    {
        self.resolve = sg::make_image(&&{
            .usage = { .resolve_attachment = true, .color_attachment = true },
            .width = width, .height = height, .sample_count = 1, .pixel_format = RGBA8,
        });
        self.resolve_view = sg::make_view(&&{ .resolve_attachment = { .image = self.resolve } });
        self.texture_image = self.resolve;
    }
    self.texture = sg::make_view(&&{ .texture = { .image = self.texture_image } });
    self.depth_texture = sg::make_view(&&{ .texture = { .image = self.depth } });
    self.overlay_view = sg::make_view(&&{ .color_attachment = { .image = self.texture_image } });
    if (self.map_texture) self.as_map();
}

fn SgAttachments RenderTarget.attachments(&self)
{
    SgAttachments a;
    a.colors[0] = self.color_view;
    a.depth_stencil = self.depth_view;
    if (self.samples > 1) a.resolves[0] = self.resolve_view;
    return a;
}

struct Renderer
{
    int width;
    int height;
    int sample_count;
    String title;
    // Open the window full screen (set before set_animation_loop). F11 or Alt+Enter toggles it.
    bool fullscreen;
    Color clear_color;
    float clear_alpha;
    // Keep perspective cameras' aspect in sync with the window (three apps usually do this in onresize).
    bool auto_aspect;
    AnimationLoop animation_loop;
    RenderTarget* render_target;
    ToneMapping tone_mapping;
    float tone_mapping_exposure;
    // Fragments inside cull_box_min..max are not drawn (to look through a portal past whatever
    // stands between the camera and it).
    bool cull_box;
    Vector3 cull_box_min;
    Vector3 cull_box_max;
    // LOD distances are multiplied by this (above 1 keeps detail and draws further).
    float lod_scale;
    // Someone walking through the plants: materials with `push` give way around this point
    // (their feet), up to pusher_reach meters from it. 0 = nobody.
    Vector3 pusher;
    float pusher_reach;
    // In shadow maps, LOD objects are picked as if they were this many times further away
    // (1 or less: as in the picture). Shadows of grass and leaves need less than the plants.
    float shadow_lod_bias;
    // Seconds, drives material sway. Follows the clock unless auto_time is switched off.
    float time;
    bool auto_time;
    // Statistics of the last render() call, like three's renderer.info.render.
    RenderInfo info;
    // What materials with `ssr` reflect (see set_reflection_source).
    SgView ssr_color;
    SgView ssr_depth;
    Mat4 ssr_view_proj;
    float ssr_reach;
    int ssr_frame;
}

// Screen-space reflections read the frame before: `color` is that picture as the render
// target held it (tone mapped; alpha 0 at light sources), `depth` has each pixel's distance from
// the camera's plane in its green channel, and `view_proj` is that frame's
// projection_matrix * matrix_world_inverse. Both the same size, row for row as rendered.
// `reach` is how far a reflected ray is followed (m). Call it every frame before render();
// clear_reflection_source() switches reflections off (for a render from another viewpoint).
fn void Renderer.set_reflection_source(&self, SgView color, SgView depth, Mat4 view_proj, float reach = 12)
{
    self.ssr_color = color;
    self.ssr_depth = depth;
    self.ssr_view_proj = view_proj;
    self.ssr_reach = reach;
    self.ssr_frame++;
}

fn void Renderer.clear_reflection_source(&self)
{
    self.ssr_color = {};
    self.ssr_depth = {};
}

struct RenderInfo
{
    int calls;
    int triangles;
    int shadow_calls;
    int shadow_triangles;
}

alias WebGLRenderer = Renderer;

enum ToneMapping
{
    NO_TONE_MAPPING,
    ACES_FILMIC_TONE_MAPPING,
}

struct DrawItem @local
{
    Mesh* mesh;
    float depth;
}

struct Frustum @local
{
    Vec4[5] planes;
}

struct RendererState @local
{
    Renderer* renderer;
    bool gfx_ready;
    SgShader shader;
    SgShader shadow_shader;
    // indexed [side][opaque, transparent, additive][wireframe][target: swapchain, offscreen 1x, offscreen 4x][alpha to coverage]
    // (C3 declares array dimensions innermost first)
    SgPipeline[2][3][2][3][3] pipelines;
    SgPipeline shadow_pipeline;
    List{DrawItem} opaque;
    List{DrawItem} transparent;
    List{DrawItem} casters;
    List{Light*} lights;
    List{EventListener} listeners;
    List{OrbitControls*} controls;
    ulong start_ticks;
    // Bound where a material has no texture.
    Texture* white;
    // Bound where a scene has no irradiance volume.
    IrradianceVolume* no_irradiance;
    IrradianceVolume* irradiance;
    // One identity instance, bound for meshes that aren't instanced.
    SgBuffer identity_instance;
    SgImage shadow_image;
    SgView shadow_attachment;
    SgView shadow_view;
    SgSampler shadow_sampler;
    SgSampler clamp_sampler;
    int shadow_size;
    // The shadow maps that are drawn once, a tile per light.
    SgImage atlas_image;
    SgView atlas_attachment;
    SgView atlas_view;
    int atlas_tile;
    SpotLight*[MAX_FIXED_SHADOWS] atlas_lights;
    // set while gathering the casters of a light with shadow.skip_instanced
    bool skip_instanced;
    // Sphere around the camera that carries the fog where there is no geometry.
    Mesh* backdrop;
}

RendererState state @local;

fn Renderer* webgl_renderer(bool antialias = true, int width = 800, int height = 600, String title = "three")
{
    Renderer* r = mem::new(Renderer);
    *r = {
        .width = width,
        .height = height,
        .sample_count = antialias ? 4 : 1,
        .title = title,
        .clear_color = 0x000000,
        .clear_alpha = 1,
        .auto_aspect = true,
        .tone_mapping_exposure = 1,
        .lod_scale = 1,
        .auto_time = true,
    };
    state.renderer = r;
    state.opaque.allocator = mem;
    state.transparent.allocator = mem;
    state.casters.allocator = mem;
    state.lights.allocator = mem;
    state.listeners.allocator = mem;
    state.controls.allocator = mem;
    return r;
}

fn void Renderer.set_size(&self, int width, int height)
{
    self.width = width;
    self.height = height;
}

// Render into `target` instead of the window (null renders to the window again).
fn void Renderer.set_render_target(&self, RenderTarget* target)
{
    self.render_target = target;
}

fn void Renderer.set_clear_color(&self, Color color, float alpha = 1)
{
    self.clear_color = color;
    self.clear_alpha = alpha;
}

// Opens the window and calls `cb` every frame with the elapsed time in milliseconds.
// Like three's renderer.setAnimationLoop. This call blocks until the window is closed.
fn void Renderer.set_animation_loop(&self, AnimationLoop cb)
{
    self.animation_loop = cb;
    state.renderer = self;
    sapp::run(&&{
        .init_cb = &on_init,
        .frame_cb = &on_frame,
        .cleanup_cb = &on_cleanup,
        .event_cb = &on_event,
        .width = self.width,
        .height = self.height,
        .sample_count = self.sample_count,
        .high_dpi = true,
        .fullscreen = self.fullscreen,
        .window_title = (ZString)self.title.zstr_tcopy(),
        .icon.sokol_default = true,
        .logger.func = &slog::func,
    });
}

// Register a listener for raw input / window events (keyboard, mouse, resize).
fn void add_event_listener(EventListener listener)
{
    state.listeners.push(listener);
}

fn float window_width() => sapp::isvalid() ? sapp::widthf() / sapp::dpi_scale() : (float)state.renderer.width;
fn float window_height() => sapp::isvalid() ? sapp::heightf() / sapp::dpi_scale() : (float)state.renderer.height;
// Equivalent of `window.innerWidth / window.innerHeight`.
fn float window_aspect() => window_width() / window_height();

fn bool depth_zero_to_one()
{
    if (!state.gfx_ready) return true;
    SgBackend b = sg::query_backend();
    return b != GLCORE && b != GLES3;
}

fn void setup_gfx(SgEnvironment environment) @local
{
    sg::setup(&&{
        .environment = environment,
        .logger.func = &slog::func,
        .buffer_pool_size = 16384,
        .image_pool_size = 1024,
        .view_pool_size = 1024,
        .sampler_pool_size = 1024,
    });
    stm::setup();
    state.start_ticks = stm::now();
    state.shader = sg::make_shader(&&three_shader_desc(sg::query_backend()));
    state.shadow_shader = sg::make_shader(&&shadow_shader_desc(sg::query_backend()));
    state.white = data_texture({ 255, 255, 255, 255 }, 1, 1, NO_COLOR_SPACE);
    float[12] identity = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
    state.identity_instance = sg::make_buffer(&&{ .data = { .ptr = &identity, .size = identity.len * float::size } });
    state.gfx_ready = true;
}

fn void on_init() @local
{
    setup_gfx(sglue::environment());
}

// Sets up the GPU without opening a window (Metal on macOS, D3D11 on Windows), for rendering into
// RenderTargets from scripts, tests or on a machine without a display. Use instead of
// set_animation_loop: call render() with a render target set, then three::commit().
fn void Renderer.init_headless(&self)
{
    state.renderer = self;
    setup_gfx(readback::headless_environment());
}

fn void on_frame() @local
{
    // the controller's state for this frame, and its key events (gamepad.c3)
    poll_gamepad();
    if (state.renderer.animation_loop) state.renderer.animation_loop(stm::ms(stm::since(state.start_ticks)));
}

fn void on_cleanup() @local
{
    sg::shutdown();
}

// Passes an event to the menus and listeners as if it came from the window (the gamepad's
// key events, say).
fn void send_event(SappEvent* e) => on_event(e);

fn void on_event(SappEvent* e) @local
{
    switch (e.type)
    {
        case TOUCHES_BEGAN:
        case TOUCHES_MOVED:
        case TOUCHES_ENDED:
        case TOUCHES_CANCELLED:
            touch_event(e);
        case MOUSE_DOWN:
        case MOUSE_UP:
        case MOUSE_MOVE:
        case MOUSE_SCROLL:
        case MOUSE_ENTER:
        case MOUSE_LEAVE:
            if (touch_from_mouse(e)) return;
        case UNFOCUSED:
        case SUSPENDED:
            touch_reset();
        default:
            break;
    }
    if (e.type == KEY_DOWN && !e.key_repeat
        && (e.key_code == F11 || (e.key_code == ENTER && (e.modifiers & sapp::MODIFIER_ALT))))
    {
        sapp::toggle_fullscreen();
        return;
    }
    // an open menu takes the input (see menu.c3)
    if (menu_filter_event(e)) return;
    foreach (c : state.controls) c.handle_event(e);
    foreach (l : state.listeners) l(e);
}

fn void register_controls(OrbitControls* c)
{
    state.controls.push(c);
}

fn SgVertexLayoutState vertex_layout(int[7] attrs) @local
{
    SgVertexLayoutState layout;
    layout.buffers[0].stride = FLOATS_PER_VERTEX * float::size;
    layout.buffers[1] = { .stride = 12 * float::size, .step_func = PER_INSTANCE };
    layout.attrs[attrs[0]] = { .format = FLOAT3, .offset = 0 };
    layout.attrs[attrs[1]] = { .format = FLOAT3, .offset = 12 };
    layout.attrs[attrs[2]] = { .format = FLOAT2, .offset = 24 };
    layout.attrs[attrs[3]] = { .format = FLOAT3, .offset = 32 };
    layout.attrs[attrs[4]] = { .format = FLOAT4, .offset = 0, .buffer_index = 1 };
    layout.attrs[attrs[5]] = { .format = FLOAT4, .offset = 16, .buffer_index = 1 };
    layout.attrs[attrs[6]] = { .format = FLOAT4, .offset = 32, .buffer_index = 1 };
    return layout;
}

fn SgPipeline get_pipeline(Side side, bool transparent, bool wireframe, RenderTarget* rt, bool alpha_test,
                           Blending blending = NORMAL_BLENDING) @local
{
    int target = rt ? (rt.samples > 1 ? 2 : 1) : 0;
    if (rt && rt.samples != 1 && rt.samples != 4) abort("three: render targets support 1 or 4 samples");
    int samples = rt ? rt.samples : sapp::sample_count();
    bool a2c = alpha_test && !transparent && samples > 1;
    bool additive = transparent && blending == ADDITIVE_BLENDING;
    SgPipeline* pip = &state.pipelines[side.ordinal][transparent ? (additive ? 2 : 1) : 0][wireframe ? 1 : 0][target][a2c ? 1 : 0];
    if (pip.id) return *pip;
    SgCullMode cull;
    switch (side)
    {
        case FRONT_SIDE: cull = BACK;
        case BACK_SIDE: cull = FRONT;
        case DOUBLE_SIDE: cull = NONE;
    }
    SgPipelineDesc desc = {
        .shader = state.shader,
        .index_type = UINT32,
        .primitive_type = wireframe ? LINES : TRIANGLES,
        .cull_mode = wireframe ? NONE : cull,
        .face_winding = CCW,
        .layout = vertex_layout({ ATTR_THREE_POSITION, ATTR_THREE_NORMAL, ATTR_THREE_UV, ATTR_THREE_COLOR0,
                                  ATTR_THREE_INST0, ATTR_THREE_INST1, ATTR_THREE_INST2 }),
        .depth = {
            .write_enabled = !transparent,
            .compare = LESS_EQUAL,
        },
        .alpha_to_coverage_enabled = a2c,
        .label = "three-pipeline",
    };
    if (rt)
    {
        desc.sample_count = rt.samples;
        desc.colors[0].pixel_format = RGBA8;
        desc.depth.pixel_format = DEPTH;
    }
    if (transparent)
    {
        desc.colors[0].blend = {
            .enabled = true,
            .src_factor_rgb = SRC_ALPHA,
            .dst_factor_rgb = additive ? SgBlendFactor.ONE : SgBlendFactor.ONE_MINUS_SRC_ALPHA,
            .src_factor_alpha = additive ? SgBlendFactor.ZERO : SgBlendFactor.ONE,
            .dst_factor_alpha = additive ? SgBlendFactor.ONE : SgBlendFactor.ONE_MINUS_SRC_ALPHA,
        };
    }
    *pip = sg::make_pipeline(&desc);
    return *pip;
}

fn SgPipeline get_shadow_pipeline() @local
{
    if (state.shadow_pipeline.id) return state.shadow_pipeline;
    state.shadow_pipeline = sg::make_pipeline(&&{
        .shader = state.shadow_shader,
        .index_type = UINT32,
        .cull_mode = NONE,
        .layout = vertex_layout({ ATTR_SHADOW_POSITION, ATTR_SHADOW_NORMAL, ATTR_SHADOW_UV, ATTR_SHADOW_COLOR0,
                                  ATTR_SHADOW_INST0, ATTR_SHADOW_INST1, ATTR_SHADOW_INST2 }),
        .depth = {
            .pixel_format = DEPTH,
            .write_enabled = true,
            .compare = LESS_EQUAL,
        },
        .color_count = 0,
        .colors[0].pixel_format = NONE,
        .sample_count = 1,
        .label = "three-shadow-pipeline",
    });
    return state.shadow_pipeline;
}

fn void ensure_shadow_map(int size) @local
{
    if (state.shadow_size == size) return;
    if (state.shadow_size)
    {
        sg::destroy_view(state.shadow_attachment);
        sg::destroy_view(state.shadow_view);
        sg::destroy_image(state.shadow_image);
    }
    else
    {
        state.shadow_sampler = sg::make_sampler(&&{
            .min_filter = LINEAR,
            .mag_filter = LINEAR,
            .wrap_u = CLAMP_TO_EDGE,
            .wrap_v = CLAMP_TO_EDGE,
            .compare = LESS_EQUAL,
        });
    }
    state.shadow_size = size;
    state.shadow_image = sg::make_image(&&{
        .usage = { .depth_stencil_attachment = true },
        .width = size, .height = size, .sample_count = 1, .pixel_format = DEPTH,
    });
    state.shadow_attachment = sg::make_view(&&{ .depth_stencil_attachment = { .image = state.shadow_image } });
    state.shadow_view = sg::make_view(&&{ .texture = { .image = state.shadow_image } });
}

fn void ensure_shadow_atlas(int tile) @local
{
    if (state.atlas_tile == tile) return;
    if (state.atlas_tile)
    {
        sg::destroy_view(state.atlas_attachment);
        sg::destroy_view(state.atlas_view);
        sg::destroy_image(state.atlas_image);
    }
    state.atlas_tile = tile;
    state.atlas_lights = {};
    state.atlas_image = sg::make_image(&&{
        .usage = { .depth_stencil_attachment = true },
        .width = tile * ATLAS_COLUMNS, .height = tile * 2, .sample_count = 1, .pixel_format = DEPTH,
    });
    state.atlas_attachment = sg::make_view(&&{ .depth_stencil_attachment = { .image = state.atlas_image } });
    state.atlas_view = sg::make_view(&&{ .texture = { .image = state.atlas_image } });
}

// The tile of a light in the atlas, -1 if it has none.
fn int atlas_tile_of(SpotLight* sl) @local
{
    foreach (i, l : state.atlas_lights)
    {
        if (l == sl) return (int)i;
    }
    return -1;
}

// Draws the maps of the lights with shadow.auto_update off, when one of them asks for it or
// the set of lights has changed.
fn void update_shadow_atlas(Scene* scene, Camera* camera) @local
{
    SpotLight*[MAX_FIXED_SHADOWS] lights;
    int count;
    int tile = 4;
    bool dirty;
    foreach (l : state.lights)
    {
        if (l.light_type != SPOT || !l.cast_shadow || !l.visible || l.intensity <= 0) continue;
        SpotLight* sl = (SpotLight*)l;
        if (sl.shadow.auto_update || count >= MAX_FIXED_SHADOWS) continue;
        lights[count++] = sl;
        tile = math::max(tile, sl.shadow.map_size);
        dirty = dirty || sl.shadow.needs_update;
    }
    // keep the atlas while nothing uses it: the lights may only be switched off for a moment
    if (!count && state.atlas_tile) return;
    ensure_shadow_atlas(tile);
    for (int i = 0; i < count; i++) dirty = dirty || atlas_tile_of(lights[i]) < 0;
    if (!dirty) return;
    state.atlas_lights = lights;
    Vector3 camera_pos = camera.matrix_world.get_position();
    bool top_left = depth_zero_to_one();
    sg::begin_pass(&&{
        .action = { .depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 } },
        .attachments = { .depth_stencil = state.atlas_attachment },
    });
    sg::apply_pipeline(get_shadow_pipeline());
    for (int i = 0; i < count; i++)
    {
        SpotLight* sl = lights[i];
        sl.shadow.needs_update = false;
        Mat4 light_vp = shadow_view_proj(sl);
        Frustum light_frustum = frustum_from(light_vp);
        state.casters.clear();
        state.skip_instanced = sl.shadow.skip_instanced;
        collect(scene, camera.matrix_world_inverse, &light_frustum, camera_pos, true);
        state.skip_instanced = false;
        sg::apply_viewport((i % ATLAS_COLUMNS) * tile, (i / ATLAS_COLUMNS) * tile, tile, tile, top_left);
        sg::apply_scissor_rect((i % ATLAS_COLUMNS) * tile, (i / ATLAS_COLUMNS) * tile, tile, tile, top_left);
        foreach (&item : state.casters) draw_shadow_item(item, light_vp);
    }
    sg::end_pass();
    state.casters.clear();
}

// Clip planes of a view-projection matrix (left, right, bottom, top, far), pointing inwards.
fn Frustum frustum_from(Mat4 vp) @local
{
    Vec4[4] rows;
    for (int r = 0; r < 4; r++) rows[r] = { vp[0][r], vp[1][r], vp[2][r], vp[3][r] };
    Frustum f;
    f.planes[0] = rows[3] + rows[0];
    f.planes[1] = rows[3] - rows[0];
    f.planes[2] = rows[3] + rows[1];
    f.planes[3] = rows[3] - rows[1];
    f.planes[4] = rows[3] - rows[2];
    foreach (&p : f.planes)
    {
        float len = ((Vector3){ (*p).x, (*p).y, (*p).z }).length();
        if (len > 0) *p /= len;
    }
    return f;
}

fn bool Frustum.contains_sphere(&self, Vector3 c, float radius)
{
    foreach (p : self.planes)
    {
        if (p.x * c.x + p.y * c.y + p.z * c.z + p.w < -radius) return false;
    }
    return true;
}

fn float max_scale(Mat4 m) @local
{
    float sx = ((Vector3){ m[0][0], m[0][1], m[0][2] }).length();
    float sy = ((Vector3){ m[1][0], m[1][1], m[1][2] }).length();
    float sc = ((Vector3){ m[2][0], m[2][1], m[2][2] }).length();
    return math::max(sx, math::max(sy, sc));
}

// Uploads the instance transforms and computes the sphere around all instances.
fn void upload_instances(Mesh* m) @local
{
    if (m.instances_uploaded) return;
    if (m.instance_buf.id) sg::destroy_buffer(m.instance_buf);
    m.instance_buf = sg::make_buffer(&&{
        .data = { .ptr = m.instance_data.entries, .size = (usz)m.instance_data.len() * float::size },
    });
    BufferGeometry* g = m.geometry;
    Vector3 lo = { 1e30f, 1e30f, 1e30f };
    Vector3 hi = -lo;
    for (int i = 0; i < m.instance_count; i++)
    {
        float* d = &m.instance_data[(sz)i * 12];
        Vector3 gc = g.bounds_center;
        Vector3 c = {
            d[0] * gc.x + d[1] * gc.y + d[2] * gc.z + d[3],
            d[4] * gc.x + d[5] * gc.y + d[6] * gc.z + d[7],
            d[8] * gc.x + d[9] * gc.y + d[10] * gc.z + d[11],
        };
        float s = math::max(((Vector3){ d[0], d[4], d[8] }).length(),
            math::max(((Vector3){ d[1], d[5], d[9] }).length(), ((Vector3){ d[2], d[6], d[10] }).length()));
        Vector3 r = { s, s, s };
        r *= g.bounds_radius;
        lo = math::min(lo, c - r);
        hi = math::max(hi, c + r);
    }
    m.instance_center = (lo + hi) * 0.5f;
    m.instance_radius = (hi - lo).length() * 0.5f;
    m.instances_uploaded = true;
}

// Gathers what to draw. `lod_from` is the position LOD levels are measured from.
fn void collect(Object3D* obj, Mat4 view, Frustum* frustum, Vector3 lod_from, bool shadow_pass) @local
{
    if (!obj.visible) return;
    switch (obj.kind)
    {
        case MESH:
            Mesh* m = (Mesh*)obj;
            if (!m.geometry || !m.material || !m.material.visible) break;
            if (shadow_pass && (!m.cast_shadow || m.material.transparent)) break;
            if (shadow_pass && state.skip_instanced && m.instance_data.len() > 0) break;
            BufferGeometry* g = m.geometry;
            if (!g.uploaded && g.indices.len() == 0) break;
            g.upload();
            Vector3 center = g.bounds_center;
            float radius = g.bounds_radius;
            if (m.instance_data.len() > 0)
            {
                upload_instances(m);
                center = m.instance_center;
                radius = m.instance_radius;
            }
            Vector3 wc = m.matrix_world.transform_point(center);
            if (m.frustum_culled)
            {
                float r = radius * max_scale(m.matrix_world) + m.material.sway * 2;
                if (!frustum.contains_sphere(wc, r)) break;
            }
            DrawItem item = { m, view.transform_point(wc).z };
            if (shadow_pass)
            {
                state.casters.push(item);
            }
            else if (m.material.transparent)
            {
                state.transparent.push(item);
            }
            else
            {
                state.opaque.push(item);
            }
        case LIGHT:
            if (!shadow_pass) state.lights.push((Light*)obj);
        case LOD:
            Lod* l = (Lod*)obj;
            float d = (l.matrix_world.get_position() - lod_from).length() / math::max(state.renderer.lod_scale, 0.01f);
            // a shadow shows less of a thing's shape than the thing does
            if (shadow_pass) d *= math::max(state.renderer.shadow_lod_bias, 1.0f);
            sz pick = -1;
            for (sz i = 0; i < l.distances.len(); i++)
            {
                if (d >= l.distances[i]) pick = i;
            }
            if (pick >= 0 && pick < l.children.len()) collect(l.children[pick], view, frustum, lod_from, shadow_pass);
            return;
        default:
            break;
    }
    foreach (c : obj.children) collect(c, view, frustum, lod_from, shadow_pass);
}

// The one light whose shadow map is drawn every frame.
fn SpotLight* shadow_light() @local
{
    foreach (l : state.lights)
    {
        if (l.light_type != SPOT || !l.cast_shadow || !l.visible || l.intensity <= 0) continue;
        if (((SpotLight*)l).shadow.auto_update) return (SpotLight*)l;
    }
    return null;
}

fn Vector3 spot_direction(SpotLight* sl) @local
{
    Vector3 target = sl.target ? sl.target.get_world_position() : (Vector3){ 0, 0, 0 };
    return (target - sl.matrix_world.get_position()).normalize();
}

// View-projection of the shadow casting spot light.
fn Mat4 shadow_view_proj(SpotLight* sl) @local
{
    Vector3 pos = sl.matrix_world.get_position();
    Vector3 dir = spot_direction(sl);
    Vector3 up = math::abs(dir.y) > 0.99f ? (Vector3){ 1, 0, 0 } : (Vector3){ 0, 1, 0 };
    Mat4 world = mat4_look_at(pos, pos + dir, up, true);
    world[3] = { pos.x, pos.y, pos.z, 1 };
    float fov = math::min(sl.angle * 2 * RAD2DEG * clamp(sl.shadow.focus, 0.05f, 1) + 4, 150.0f);
    float far = sl.shadow.far > 0 ? sl.shadow.far : (sl.distance > 0 ? sl.distance : 50.0f);
    Mat4 proj = mat4_perspective(fov, 1, math::max(sl.shadow.near, 0.01f), far, 1, depth_zero_to_one());
    return proj.mul(world.invert());
}

fn FsParams build_light_params(Scene* scene, Camera* camera, SpotLight* caster) @local
{
    FsParams p;
    Renderer* r = state.renderer;
    float exposure = r.tone_mapping == ACES_FILMIC_TONE_MAPPING ? math::max(r.tone_mapping_exposure, 1e-4f) : 0;
    p.camera_pos = { ...camera.matrix_world.get_position(), exposure };
    p.cull_min = { ...r.cull_box_min, r.cull_box ? 1 : 0 };
    p.cull_max = { ...r.cull_box_max, 0 };
    int ndir;
    int npoint;
    // the shadow caster always takes spot slot 0
    int nspot = caster ? 1 : 0;
    foreach (l : state.lights)
    {
        Vector3 c = color_to_linear(l.color) * l.intensity;
        switch (l.light_type)
        {
            case AMBIENT:
                p.ambient += { ...c, 0 };
            case HEMISPHERE:
                HemisphereLight* h = (HemisphereLight*)l;
                p.hemi_sky += { ...c, 0 };
                p.hemi_ground += { ...color_to_linear(h.ground_color) * l.intensity, 0 };
            case DIRECTIONAL:
                if (ndir >= MAX_DIR_LIGHTS) break;
                DirectionalLight* d = (DirectionalLight*)l;
                Vector3 target = d.target ? d.target.get_world_position() : (Vector3){ 0, 0, 0 };
                Vector3 dir = l.matrix_world.get_position() - target;
                p.dir_dir[ndir] = { ...dir, 0 };
                p.dir_color[ndir] = { ...c, 0 };
                ndir++;
            case POINT:
                if (npoint >= MAX_POINT_LIGHTS) break;
                PointLight* pl = (PointLight*)l;
                p.point_pos[npoint] = { ...l.matrix_world.get_position(), pl.distance };
                p.point_color[npoint] = { ...c, pl.decay };
                p.point_params[npoint] = { pl.radius, 0, 0, 0 };
                npoint++;
            case SPOT:
                SpotLight* sl = (SpotLight*)l;
                int slot = sl == caster ? 0 : nspot;
                if (slot >= MAX_SPOT_LIGHTS) break;
                float cos_outer = math::cos(clamp(sl.angle, 0, PI / 2));
                float cos_inner = math::cos(clamp(sl.angle, 0, PI / 2) * (1 - clamp(sl.penumbra, 0, 1)));
                p.spot_pos[slot] = { ...l.matrix_world.get_position(), sl.distance };
                p.spot_dir[slot] = { ...spot_direction(sl), cos_outer };
                p.spot_color[slot] = { ...c, sl.decay };
                p.spot_params[slot] = { math::max(cos_inner, cos_outer + 1e-4f), sl.radius,
                                        sl.specular_radius > 0 ? sl.specular_radius : sl.radius, sl.fog_glow };
                // clip space -> shadow map uv (v points down outside of GL) and 0..1 depth
                int tile = atlas_tile_of(sl);
                if (sl == caster || (tile >= 0 && sl.cast_shadow))
                {
                    bool zo = depth_zero_to_one();
                    Mat4 bias = mat4_identity();
                    bias[0][0] = 0.5f;
                    bias[3][0] = 0.5f;
                    bias[1][1] = zo ? -0.5f : 0.5f;
                    bias[3][1] = 0.5f;
                    bias[2][2] = zo ? 1.0f : 0.5f;
                    bias[3][2] = zo ? 0.0f : 0.5f;
                    p.shadow_mat[slot] = bias.mul(shadow_view_proj(sl));
                    float size = sl == caster ? (float)state.shadow_size : (float)state.atlas_tile;
                    p.shadow_params[slot] = { sl == caster ? 1 : 2, sl.shadow.normal_bias, 1.0f / size, sl.shadow.radius };
                    float fov = math::min(sl.angle * 2 * RAD2DEG * clamp(sl.shadow.focus, 0.05f, 1) + 4, 150.0f);
                    float width = 2 * math::tan(fov * DEG2RAD * 0.5f);
                    float far = sl.shadow.far > 0 ? sl.shadow.far : (sl.distance > 0 ? sl.distance : 50.0f);
                    p.shadow_soft[slot] = { sl.shadow.softness, math::max(sl.shadow.near, 0.01f), far, zo ? width : -width };
                    if (sl != caster) p.shadow_rect[slot] = { (float)(tile % ATLAS_COLUMNS) / ATLAS_COLUMNS, (float)(tile / ATLAS_COLUMNS) * 0.5f,
                                                             1.0f / ATLAS_COLUMNS, 0.5f };
                }
                if (sl != caster) nspot++;
        }
    }
    p.counts = { ndir, npoint, nspot, 0 };
    if (scene.irradiance)
    {
        IrradianceVolume* v = scene.irradiance;
        Vector3 size = v.max - v.min;
        p.gi_min = { ...v.min, 1 };
        p.gi_inv_size = { 1 / size.x, 1 / size.y, 1 / size.z, v.intensity };
        p.gi_dims = { v.nx, v.ny, v.nz, v.normal_offset };
    }
    if (scene.fog)
    {
        Fog* f = scene.fog;
        float mode;
        switch (f.type)
        {
            case LINEAR: mode = 1;
            case EXP2: mode = 2;
            case EXP: mode = 3;
        }
        p.fog_color = { ...color_to_linear(f.color), f.light_scatter };
        p.fog_params = { f.near, f.far, f.density, mode };
        p.fog_box_min = { ...f.clear_box_min, f.has_clear_box ? 1 : 0 };
        p.fog_box_max = { ...f.clear_box_max, 0 };
        p.fog_height = { f.height_base, f.height_falloff, 0, f.shade_density };
        p.fog_sun_dir = { ...f.sun_direction, f.sun_sharpness };
        p.fog_sun_color = { ...color_to_linear(f.sun_color), 0 };
    }
    return p;
}

fn Texture* ready(Texture* t) @local
{
    if (!t) t = state.white;
    t.upload();
    return t;
}

fn Vec4 uv_transform(Material* mat) @local
{
    Texture* t = mat.map ?: (mat.normal_map ?: mat.arm_map);
    return t ? (Vec4){ t.repeat.x, t.repeat.y, t.offset.x, t.offset.y } : (Vec4){ 1, 1, 0, 0 };
}

fn Vec4 sway_params(Material* mat) @local => { state.renderer.time, mat.sway, mat.sway_speed, mat.push };
fn Vec4 pusher_params() @local => { ...state.renderer.pusher, state.renderer.pusher_reach };

fn int draw_geometry(Mesh* m, bool wireframe) @local
{
    BufferGeometry* g = m.geometry;
    int instances = m.instance_data.len() > 0 ? m.instance_count : 1;
    int elements = wireframe ? (int)g.wire_count : (int)g.index_count;
    sg::draw(0, elements, instances);
    return elements / 3 * instances;
}

fn void draw_item(DrawItem* item, Mat4 view_proj, FsParams* fs) @local
{
    Mesh* m = item.mesh;
    Material* mat = m.material;
    BufferGeometry* g = m.geometry;
    if (mat.wireframe) g.upload_wire();
    Renderer* r = state.renderer;

    Texture* map = ready(mat.map);
    if (mat.screen_target) map = state.white;
    sg::apply_pipeline(get_pipeline(mat.side, mat.transparent, mat.wireframe, r.render_target, mat.alpha_test > 0, mat.blending));
    SgBindings bind = {
        .vertex_buffers[0] = g.vbuf,
        .vertex_buffers[1] = m.instance_data.len() > 0 ? m.instance_buf : state.identity_instance,
        .index_buffer = mat.wireframe ? g.wire_ibuf : g.ibuf,
    };
    bind.views[VIEW_TEX_MAP] = map.view;
    bind.views[VIEW_TEX_NORMAL] = ready(mat.normal_map).view;
    bind.views[VIEW_TEX_ARM] = ready(mat.arm_map).view;
    bind.views[VIEW_SHADOW_MAP] = state.shadow_view;
    bind.views[VIEW_SHADOW_ATLAS] = state.atlas_view;
    bool ssr = r.ssr_color.id != 0 && r.ssr_depth.id != 0 && mat.ssr > 0;
    bind.views[VIEW_SSR_COLOR] = ssr ? r.ssr_color : ready(state.white).view;
    bind.views[VIEW_SSR_DEPTH] = ssr ? r.ssr_depth : ready(state.white).view;
    if (!state.clamp_sampler.id)
    {
        state.clamp_sampler = sg::make_sampler(&&{
            .min_filter = LINEAR, .mag_filter = LINEAR, .wrap_u = CLAMP_TO_EDGE, .wrap_v = CLAMP_TO_EDGE,
        });
    }
    bind.samplers[SMP_SSR_SMP] = state.clamp_sampler;
    bind.views[VIEW_TEX_GI] = state.irradiance.view;
    bind.samplers[SMP_GI_SMP] = state.irradiance.sampler;
    Texture* env = ready(mat.env_map);
    bind.views[VIEW_TEX_ENV] = env.view;
    bind.samplers[SMP_ENV_SMP] = env.sampler;
    bind.samplers[SMP_TEX_SMP] = (mat.map ?: (mat.normal_map ?: (mat.arm_map ?: state.white))).sampler;
    if (mat.screen_target)
    {
        bind.views[VIEW_TEX_MAP] = mat.screen_target.texture;
        bind.samplers[SMP_TEX_SMP] = state.white.sampler;
    }
    bind.samplers[SMP_SHADOW_SMP] = state.shadow_sampler;
    sg::apply_bindings(&bind);

    VsParams vs = {
        .view_proj = view_proj,
        .model = m.matrix_world,
        .normal_mat = m.matrix_world.invert().transpose(),
        .uv_transform = uv_transform(mat),
        .sway = sway_params(mat),
        .pusher = pusher_params(),
    };
    sg::apply_uniforms(UB_VS_PARAMS, &&{ .ptr = &vs, .size = VsParams::size });

    int flags = (mat.vertex_colors ? 1 : 0) | (mat.foliage ? 2 : 0) | (m.receive_shadow ? 4 : 0) | (mat.glass ? 8 : 0)
              | (mat.vertex_light > 0 ? 16 : 0) | (mat.screen_target ? 32 : 0);
    fs.color = { ...color_to_linear(mat.color), mat.opacity };
    fs.emissive = { ...color_to_linear(mat.emissive) * mat.emissive_intensity, mat.vertex_light };
    if (r.render_target)
    {
        fs.screen_params = { 1.0f / r.render_target.width, 1.0f / r.render_target.height, 0, 0 };
    }
    else
    {
        fs.screen_params = { 1.0f / sapp::widthf(), 1.0f / sapp::heightf(), 0, 0 };
    }
    fs.specular = { ...color_to_linear(mat.specular), (float)flags };
    fs.mat_params = { (float)mat.type, mat.roughness, mat.metalness, mat.shininess };
    fs.tex_params = { mat.map ? 1 : 0, mat.normal_map ? mat.normal_scale : 0, mat.arm_map ? 1 : 0, mat.alpha_test };
    fs.env_params = { mat.env_map ? 1 : 0, mat.env_intensity > 0 ? mat.env_intensity : 1, mat.reflect_viewer,
                      mat.env_map ? mat.env_map.env_distance_range : 0 };
    bool boxed = mat.env_box_min != mat.env_box_max;
    fs.env_pos = { ...mat.env_position, boxed ? 1 : 0 };
    fs.env_box_min = { ...mat.env_box_min, 0 };
    fs.env_box_max = { ...mat.env_box_max, 0 };
    fs.grime_params = { mat.grime, mat.grime_floor, mat.grime_height > 0 ? mat.grime_height : 2.6f, mat.grime_seed };
    fs.ssr_view_proj = r.ssr_view_proj;
    fs.ssr_params = { ssr ? mat.ssr : 0, r.ssr_reach, (float)(r.ssr_frame % 64), depth_zero_to_one() ? 1 : 0 };
    float fog_mode = fs.fog_params.w;
    if (!mat.fog) fs.fog_params.w = 0;
    sg::apply_uniforms(UB_FS_PARAMS, &&{ .ptr = fs, .size = FsParams::size });
    fs.fog_params.w = fog_mode;

    r.info.calls++;
    r.info.triangles += draw_geometry(m, mat.wireframe);
}

fn void draw_shadow_item(DrawItem* item, Mat4 light_vp) @local
{
    Mesh* m = item.mesh;
    Material* mat = m.material;
    Texture* map = ready(mat.map);
    SgBindings bind = {
        .vertex_buffers[0] = m.geometry.vbuf,
        .vertex_buffers[1] = m.instance_data.len() > 0 ? m.instance_buf : state.identity_instance,
        .index_buffer = m.geometry.ibuf,
    };
    bind.views[VIEW_S_MAP] = map.view;
    bind.samplers[SMP_S_SMP] = map.sampler;
    sg::apply_bindings(&bind);
    ShadowVsParams vs = {
        .s_view_proj = light_vp,
        .s_model = m.matrix_world,
        .s_uv_transform = uv_transform(mat),
        .s_sway = sway_params(mat),
        .s_pusher = pusher_params(),
    };
    sg::apply_uniforms(UB_SHADOW_VS_PARAMS, &&{ .ptr = &vs, .size = ShadowVsParams::size });
    ShadowFsParams fs = { .s_params = { mat.map ? mat.alpha_test : 0, mat.opacity, 0, 0 } };
    sg::apply_uniforms(UB_SHADOW_FS_PARAMS, &&{ .ptr = &fs, .size = ShadowFsParams::size });
    state.renderer.info.shadow_calls++;
    state.renderer.info.shadow_triangles += draw_geometry(m, false);
}

fn int cmp_opaque(DrawItem a, DrawItem b) @local => a.depth < b.depth ? 1 : (a.depth > b.depth ? -1 : 0);
fn int cmp_transparent(DrawItem a, DrawItem b) @local
{
    if (a.mesh.render_order != b.mesh.render_order) return a.mesh.render_order < b.mesh.render_order ? -1 : 1;
    return a.depth < b.depth ? -1 : (a.depth > b.depth ? 1 : 0);
}

// The background color as it comes out of the lit shader, so that it matches fog of the same color.
fn Vector3 tone_mapped(Renderer* r, Color color) @local
{
    if (r.tone_mapping != ACES_FILMIC_TONE_MAPPING) return color_to_rgb(color);
    Vector3 c = color_to_linear(color) * (math::max(r.tone_mapping_exposure, 1e-4f) / 0.6f);
    Vector3 v = {
        0.59719f * c.x + 0.35458f * c.y + 0.04823f * c.z,
        0.07600f * c.x + 0.90834f * c.y + 0.01566f * c.z,
        0.02840f * c.x + 0.13383f * c.y + 0.83777f * c.z,
    };
    Vector3 a = v * (v + (Vector3){ 0.0245786f, 0.0245786f, 0.0245786f }) - (Vector3){ 0.000090537f, 0.000090537f, 0.000090537f };
    Vector3 b = v * (v * 0.983729f + (Vector3){ 0.4329510f, 0.4329510f, 0.4329510f }) + (Vector3){ 0.238081f, 0.238081f, 0.238081f };
    v = a / b;
    Vector3 o = {
        1.60475f * v.x - 0.53108f * v.y - 0.07367f * v.z,
        -0.10208f * v.x + 1.10813f * v.y - 0.00605f * v.z,
        -0.00327f * v.x - 0.07276f * v.y + 1.07602f * v.z,
    };
    Vector3 result;
    for (int i = 0; i < 3; i++)
    {
        float x = clamp(o[i], 0, 1);
        result[i] = x <= 0.0031308f ? x * 12.92f : 1.055f * math::pow(x, 1.0f / 2.4f) - 0.055f;
    }
    return result;
}

// Fog that is lit by lights isn't one color: where the view ends in empty space, a sphere
// just inside the far plane picks up the same fog as the geometry in front of it.
fn void draw_backdrop(Scene* scene, Camera* camera, Mat4 view_proj, FsParams* fs) @local
{
    if (!scene.fog || scene.fog.light_scatter <= 0 || camera.camera_type != PERSPECTIVE) return;
    if (!state.backdrop)
    {
        MeshBasicMaterial* mat = mesh_basic_material();
        mat.side = BACK_SIDE;
        state.backdrop = mesh(sphere_geometry(1, 24, 12), mat);
        state.backdrop.receive_shadow = false;
    }
    Mesh* m = state.backdrop;
    m.material.color = scene.background ? scene.background : scene.fog.color;
    float r = ((PerspectiveCamera*)camera).far * 0.9f;
    m.matrix_world = mat4_compose(camera.matrix_world.get_position(), { 0, 0, 0 }, { r, r, r });
    m.geometry.upload();
    DrawItem item = { m, 0 };
    draw_item(&item, view_proj, fs);
}

// Draws the scene from the camera's point of view. Call once per frame from the animation loop.
fn void Renderer.render(&self, Scene* scene, Camera* camera)
{
    if (!state.gfx_ready) return;
    if (self.auto_time) self.time = (float)stm::sec(stm::since(state.start_ticks));
    self.info = {};

    if (self.auto_aspect && camera.camera_type == PERSPECTIVE)
    {
        PerspectiveCamera* pc = (PerspectiveCamera*)camera;
        RenderTarget* rt = self.render_target;
        pc.aspect = rt ? (float)rt.width / rt.height : sapp::widthf() / sapp::heightf();
    }
    camera.update_projection_matrix();

    if (!scene.no_matrix_world_update) scene.update_matrix_world();
    // cameras don't have to be part of the scene
    if (!camera.parent) camera.update_matrix_world();
    camera.matrix_world_inverse = camera.matrix_world.invert();
    Mat4 view_proj = camera.projection_matrix.mul(camera.matrix_world_inverse);
    Vector3 camera_pos = camera.matrix_world.get_position();

    state.opaque.clear();
    state.transparent.clear();
    state.casters.clear();
    state.lights.clear();
    Frustum frustum = frustum_from(view_proj);
    collect(scene, camera.matrix_world_inverse, &frustum, camera_pos, false);
    // front-to-back for opaque, back-to-front for transparent (view space looks down -Z)
    sort::quicksort(&state.opaque, &cmp_opaque);
    sort::quicksort(&state.transparent, &cmp_transparent);

    // shadow map of the shadow casting spot light
    SpotLight* caster = shadow_light();
    // without a caster keep whatever map there is: renders that switch shadows off (a portal
    // view, say) must not reallocate it every frame
    ensure_shadow_map(caster ? caster.shadow.map_size : (state.shadow_size ?: 4));
    if (caster)
    {
        Mat4 light_vp = shadow_view_proj(caster);
        Frustum light_frustum = frustum_from(light_vp);
        collect(scene, camera.matrix_world_inverse, &light_frustum, camera_pos, true);
        sg::begin_pass(&&{
            // depth is not stored by default: ask for it, the main pass samples this map
            .action = { .depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 } },
            .attachments = { .depth_stencil = state.shadow_attachment },
        });
        sg::apply_pipeline(get_shadow_pipeline());
        foreach (&item : state.casters) draw_shadow_item(item, light_vp);
        sg::end_pass();
    }

    update_shadow_atlas(scene, camera);
    if (!state.no_irradiance) state.no_irradiance = irradiance_volume({ 0, 0, 0 }, { 1, 1, 1 }, 1);
    state.irradiance = scene.irradiance ?: state.no_irradiance;
    state.irradiance.upload();

    FsParams fs = build_light_params(scene, camera, caster);

    Vector3 clear = tone_mapped(self, scene.background ? scene.background : self.clear_color);
    SgPass pass = {
        .action = {
            .colors[0] = {
                .load_action = CLEAR,
                .clear_value = { clear.x, clear.y, clear.z, scene.background ? 1.0f : self.clear_alpha },
            },
        },
    };
    if (self.render_target)
    {
        // kept for RenderTarget.depth_texture
        pass.action.depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 };
        pass.attachments = self.render_target.attachments();
    }
    else
    {
        pass.swapchain = sglue::swapchain();
    }
    sg::begin_pass(&pass);
    foreach (&item : state.opaque) draw_item(item, view_proj, &fs);
    draw_backdrop(scene, camera, view_proj, &fs);
    foreach (&item : state.transparent) draw_item(item, view_proj, &fs);
    // menus and fades on top (see menu.c3)
    if (!self.render_target) draw_ui();
    sg::end_pass();
    // When rendering to the window, finish the frame. With a render target, the caller
    // usually runs more passes (post-processing) and must call three::commit() at the end.
    if (!self.render_target) sg::commit();
}

// Ends the frame. Only needed when the last render() went to a render target.
// Called once per cube face by capture_environment, after the camera is pointed (to aim a
// light with the view, for example).
alias EnvFaceFn = fn void(Camera* camera);
// How far it is from `origin` along `dir` to the nearest surface (for capture_environment).
// Called once for every texel of the map, row by row from the top, always in the same order.
alias EnvDistanceFn = fn float(Vector3 origin, Vector3 dir);

// Renders the scene in every direction from `position`, like a light probe, into an
// equirectangular texture for Material.env_map. Needs the GPU (call it from a frame, or after
// init_headless) and reads the result back, so do it once at load, not every frame.
// With `distance` (the application's own ray caster) the map also holds how far away each
// direction's surface is, up to `distance_range` meters. A box-projected material then follows
// its reflected rays through the room as the map saw it, instead of taking everything to be on
// the walls: furniture in the middle of the room is mirrored where it stands.
fn Texture* Renderer.capture_environment(&self, Scene* scene, Vector3 position, int face_size = 256,
                                         int width = 1024, float near = 0.05, float far = 100,
                                         EnvFaceFn on_face = null, EnvDistanceFn distance = null,
                                         float distance_range = 16)
{
    Vector3[6] fwd = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
    Vector3[6] ups = { { 0, 1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 0 } };
    Vector3[6] rights;
    RenderTarget* rt = render_target(face_size, face_size, 1);
    PerspectiveCamera* cam = perspective_camera(90, 1, near, far);
    RenderTarget* prev_target = self.render_target;
    bool prev_auto = self.auto_aspect;
    self.auto_aspect = false;
    usz face_bytes = (usz)face_size * (usz)face_size * 4;
    char[] faces = mem::new_array(char, (sz)face_bytes * 6);
    defer free(faces);
    for (int i = 0; i < 6; i++)
    {
        rights[i] = fwd[i].cross(ups[i]);
        cam.position = position;
        cam.up = ups[i];
        cam.update_matrix_world();
        cam.look_at_vec(position + fwd[i]);
        cam.update_matrix_world();
        if (on_face) on_face(cam);
        self.set_render_target(rt);
        self.render(scene, cam);
        commit();
        readback::read_image(rt.texture_image, face_size, face_size,
                               &faces[face_bytes * (usz)i]);
    }
    self.set_render_target(prev_target);
    self.auto_aspect = prev_auto;
    rt.dispose();
    free(rt);
    free(cam);

    // resample the six faces into longitude/latitude
    int height = width / 2;
    char[] px = mem::new_array(char, (sz)width * height * 4);
    defer free(px);
    for (int y = 0; y < height; y++)
    {
        float phi = ((float)y + 0.5f) / height * PI;
        for (int x = 0; x < width; x++)
        {
            float theta = (((float)x + 0.5f) / width - 0.5f) * 2 * PI;
            Vector3 d = { math::cos(theta) * math::sin(phi), math::cos(phi), math::sin(theta) * math::sin(phi) };
            int best;
            float best_dot = -2;
            for (int i = 0; i < 6; i++)
            {
                float dd = d.dot(fwd[i]);
                if (dd > best_dot)
                {
                    best_dot = dd;
                    best = i;
                }
            }
            float fx = d.dot(rights[best]) / best_dot;
            float fy = d.dot(ups[best]) / best_dot;
            int sx = math::clamp((int)((fx * 0.5f + 0.5f) * face_size), 0, face_size - 1);
            int sy = math::clamp((int)((0.5f - fy * 0.5f) * face_size), 0, face_size - 1);
            usz src = face_bytes * (usz)best + ((usz)sy * (usz)face_size + (usz)sx) * 4;
            usz dst = ((usz)y * (usz)width + (usz)x) * 4;
            px[dst:3] = faces[src:3];
            px[dst + 3] = 255;
            if (distance) px[dst + 3] = (char)(math::clamp(distance(position, d) / distance_range, 0.0f, 1.0f) * 255 + 0.5f);
        }
    }
    Texture* map = data_texture(px, width, height);
    if (distance) map.env_distance_range = distance_range;
    return map;
}

fn void commit()
{
    sg::commit();
}
