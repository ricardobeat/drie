// Renderer: owns the window and the GPU, draws a Scene from a Camera.
module three;
import std::collections::list, std::sort, std::io, std::math;
import sokol, readback;

alias AnimationLoop = fn void(double time_ms);
alias EventListener = fn void(SappEvent* event);
alias Event = SappEvent;
alias KeyCode = SappKeycode;

// Defaults of Renderer.max_lights and Renderer.shadow_atlas_capacity.
const int DEFAULT_MAX_LIGHTS = 1024;
const int DEFAULT_SHADOW_ATLAS_CAPACITY = 8;

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
    // For profiling (build/scene --bench --profile): what a frame costs without alpha testing
    // (cut-outs drawn solid), without light in the fog, or without the transparent objects.
    bool prof_no_alpha_test;
    bool prof_no_fog_scatter;
    bool prof_no_transparent;
    // Extension: how many lights of each kind (directional, point, spot) a render takes at
    // most; the buffers that hold them grow up to it. 0 = no limit.
    int max_lights;
    // Extension: how many spot lights with shadow.auto_update off get a tile in the shadow
    // atlas (the atlas has max(4, sqrt(n)) columns of tiles).
    int shadow_atlas_capacity;
    // Extension: sorts the lights into clusters (see clusters.c3). null: in a compute shader
    // (cluster_lights_gpu), or on the CPU without them.
    ClusterAssignFn assign_clusters;
    // Extension: culls the instances of instanced meshes one by one on the GPU, for the
    // camera and the shadow maps drawn every frame (see culling.c3). On by default.
    bool gpu_culling;
    // Extension: what has holes in it (alpha_test cut-outs, LOD cross-fades) is drawn twice:
    // first only its depth, with a cheap shader that cuts the holes, then lit where that depth
    // is, by a shader that cuts nothing. The lit shader then runs once per pixel instead of
    // for every layer of leaves and blades, and no shader of the main pass discards (one that
    // can costs the GPU its early depth test). On by default.
    bool depth_prepass;
    // ...for what covers at least this many pixels per triangle (about: its bounding sphere's
    // disk on the screen, over its triangles). Default 1.5.
    float depth_prepass_pixels;
    // Extension: also does on the CPU what compute shaders do (light clusters, instance
    // culling) and reports where the results differ, on stderr.
    bool check_compute;
    // Extension: tint what each cascade of the directional light's shadow covers (red, green,
    // blue, yellow, nearest first), to see where they fall.
    bool debug_cascades;
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
    // in a LOD's cross-fade: how far it has gone (0..1), and 1 for the level fading in, -1 for
    // the one fading out (0: drawn whole)
    float fade;
    float fade_dir;
    // its instances culled on the GPU: the job (culling.c3) + 1; 0: drawn whole
    int job;
    // how much of the view its bounding sphere takes, about (radius / distance)^2
    float cover;
}

struct Frustum @private
{
    Vec4[5] planes;
}

struct RendererState @local
{
    Renderer* renderer;
    bool gfx_ready;
    SgShader shader;
    SgShader opaque_shader;
    SgShader depth_shader;
    SgShader shadow_shader;
    // indexed [DrawKind][side][opaque, transparent, additive][wireframe][target: swapchain, offscreen 1x, offscreen 4x]
    // [alpha to coverage] (C3 declares array dimensions innermost first)
    SgPipeline[2][3][2][3][3][4] pipelines;
    SgPipeline shadow_pipeline;
    // writes the far depth wherever it draws: clears one tile of the shadow atlas
    SgPipeline tile_clear_pipeline;
    BufferGeometry* tile_quad;
    List{DrawItem} opaque;
    List{DrawItem} transparent;
    List{DrawItem} casters;
    // what the shadow maps drawn every frame show: the spot light's, and each cascade's
    List{DrawItem} spot_casters;
    List{DrawItem}[4] sun_casters;
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
    // the lights in it, in the order of their tiles, and its grid
    List{SpotLight*} atlas_lights;
    int atlas_columns;
    int atlas_rows;
    // the lights that want a tile, while updating it
    List{SpotLight*} atlas_pending;
    // set while gathering the casters of a light with shadow.skip_instanced
    bool skip_instanced;
    // the directional light with a shadow, if any, and its map: one box, or 2x2 cascades
    DirectionalLight* sun_caster;
    SgImage sun_image;
    SgView sun_attachment;
    SgView sun_view;
    int sun_size;
    // its cascades: count (0: one box), view-projection, tile, where each ends (view depth, m)
    // and the size of its texels (m)
    int sun_cascades;
    Mat4[4] sun_view_proj;
    Vec4[4] sun_rect;
    Vec4[4] sun_info;
    Vector3 sun_forward;
    // Sphere around the camera that carries the fog where there is no geometry.
    Mesh* backdrop;
    // the volumetric fog's grid for this render (fog.c3), or a white texture
    SgView fog_view;
    // the camera's focal length in pixels (0: not a perspective camera), for prepassed()
    float prepass_focal;
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
        .gpu_culling = true,
        .depth_prepass = true,
        .depth_prepass_pixels = 1.5f,
        .max_lights = DEFAULT_MAX_LIGHTS,
        .shadow_atlas_capacity = DEFAULT_SHADOW_ATLAS_CAPACITY,
    };
    state.renderer = r;
    state.opaque.allocator = mem;
    state.transparent.allocator = mem;
    state.casters.allocator = mem;
    state.spot_casters.allocator = mem;
    foreach (&l : state.sun_casters) l.allocator = mem;
    state.lights.allocator = mem;
    state.atlas_lights.allocator = mem;
    state.atlas_pending.allocator = mem;
    lights_gpu.init_lists();
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
        // a buffer per geometry (two, indexed), and per instanced mesh whose instances change
        // often: a farm of scattered plants has tens of thousands. sokol's maximum.
        .buffer_pool_size = 65535,
        // room for the uniforms of some thousands of draws a frame (sokol's default, 4 MB,
        // is too little for a scene of many scattered plants seen from afar)
        .uniform_buffer_size = 16 * 1024 * 1024,
        .image_pool_size = 1024,
        .view_pool_size = 4096,
        .sampler_pool_size = 1024,
    });
    stm::setup();
    state.start_ticks = stm::now();
    state.shader = sg::make_shader(&&three_shader_desc(sg::query_backend()));
    state.opaque_shader = sg::make_shader(&&three_opaque_shader_desc(sg::query_backend()));
    state.depth_shader = sg::make_shader(&&depth_shader_desc(sg::query_backend()));
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
    layout.attrs[attrs[3]] = { .format = FLOAT4, .offset = 32 };
    layout.attrs[attrs[4]] = { .format = FLOAT4, .offset = 0, .buffer_index = 1 };
    layout.attrs[attrs[5]] = { .format = FLOAT4, .offset = 16, .buffer_index = 1 };
    layout.attrs[attrs[6]] = { .format = FLOAT4, .offset = 32, .buffer_index = 1 };
    return layout;
}

// How a draw of the main pass is made (see Renderer.depth_prepass).
enum DrawKind : char @local
{
    // the shader that can cut holes (transparent things, or without a depth prepass)
    CUT,
    // the shader that cuts none, for what has no holes
    SOLID,
    // the same, where the depth prepass has drawn: an EQUAL depth test, no depth written
    PRETESTED,
    // the depth prepass: depth only, holes cut
    DEPTH,
}

fn SgPipeline get_pipeline(Side side, bool transparent, bool wireframe, RenderTarget* rt, bool alpha_test,
                           Blending blending = NORMAL_BLENDING, DrawKind kind = CUT) @local
{
    int target = rt ? (rt.samples > 1 ? 2 : 1) : 0;
    if (rt && rt.samples != 1 && rt.samples != 4) abort("three: render targets support 1 or 4 samples");
    int samples = rt ? rt.samples : sapp::sample_count();
    // after the prepass, the coverage of the cut-out's edge is in the depth buffer's samples
    bool a2c = alpha_test && !transparent && samples > 1 && kind != PRETESTED;
    bool additive = transparent && blending == ADDITIVE_BLENDING;
    SgPipeline* pip = &state.pipelines[kind.ordinal][side.ordinal][transparent ? (additive ? 2 : 1) : 0][wireframe ? 1 : 0][target][a2c ? 1 : 0];
    if (pip.id) return *pip;
    SgCullMode cull;
    switch (side)
    {
        case FRONT_SIDE: cull = BACK;
        case BACK_SIDE: cull = FRONT;
        case DOUBLE_SIDE: cull = NONE;
    }
    SgShader shader;
    switch (kind)
    {
        case CUT: shader = state.shader;
        case SOLID:
        case PRETESTED: shader = state.opaque_shader;
        case DEPTH: shader = state.depth_shader;
    }
    SgPipelineDesc desc = {
        .shader = shader,
        .index_type = UINT32,
        .primitive_type = wireframe ? LINES : TRIANGLES,
        .cull_mode = wireframe ? NONE : cull,
        .face_winding = CCW,
        .layout = vertex_layout({ ATTR_THREE_POSITION, ATTR_THREE_NORMAL, ATTR_THREE_UV, ATTR_THREE_COLOR0,
                                  ATTR_THREE_INST0, ATTR_THREE_INST1, ATTR_THREE_INST2 }),
        .depth = {
            .write_enabled = !transparent && kind != PRETESTED,
            .compare = kind == PRETESTED ? SgCompareFunc.EQUAL : SgCompareFunc.LESS_EQUAL,
        },
        .alpha_to_coverage_enabled = a2c,
        .label = kind == DEPTH ? "three-depth-prepass" : "three-pipeline",
    };
    if (kind == DEPTH) desc.colors[0].write_mask = NONE;
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

// Clears the atlas's tile at the current viewport to the far depth (a pass clears all of it).
fn void clear_atlas_tile() @local
{
    if (!state.tile_clear_pipeline.id)
    {
        state.tile_clear_pipeline = sg::make_pipeline(&&{
            .shader = state.shadow_shader,
            .index_type = UINT32,
            .cull_mode = NONE,
            .layout = vertex_layout({ ATTR_SHADOW_POSITION, ATTR_SHADOW_NORMAL, ATTR_SHADOW_UV, ATTR_SHADOW_COLOR0,
                                      ATTR_SHADOW_INST0, ATTR_SHADOW_INST1, ATTR_SHADOW_INST2 }),
            .depth = { .pixel_format = DEPTH, .write_enabled = true, .compare = ALWAYS },
            .color_count = 0,
            .colors[0].pixel_format = NONE,
            .sample_count = 1,
            .label = "three-atlas-tile-clear",
        });
        state.tile_quad = plane_geometry(2, 2);
    }
    BufferGeometry* g = state.tile_quad;
    g.upload();
    sg::apply_pipeline(state.tile_clear_pipeline);
    SgBindings bind = { .vertex_buffers[0] = g.vbuf, .vertex_buffers[1] = state.identity_instance, .index_buffer = g.ibuf };
    Vec4 inst_cull = cull_bind(&bind, -1);
    Texture* white = ready(state.white);
    bind.views[VIEW_S_MAP] = white.view;
    bind.samplers[SMP_S_SMP] = white.sampler;
    sg::apply_bindings(&bind);
    // the quad's x and y as they are, at depth 1 (clip z = w: the far plane in both of the
    // depth ranges)
    Mat4 to_far = mat4_identity();
    to_far[2][2] = 0;
    to_far[3][2] = 1;
    ShadowVsParams vs = { .s_view_proj = to_far, .s_model = mat4_identity(), .s_uv_transform = { 1, 1, 0, 0 },
                          .s_inst_cull = inst_cull };
    sg::apply_uniforms(UB_SHADOW_VS_PARAMS, &&{ .ptr = &vs, .size = ShadowVsParams::size });
    ShadowFsParams fs;
    sg::apply_uniforms(UB_SHADOW_FS_PARAMS, &&{ .ptr = &fs, .size = ShadowFsParams::size });
    sg::draw(0, (int)g.index_count, 1);
    sg::apply_pipeline(get_shadow_pipeline());
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

fn void ensure_sun_map(int size) @local
{
    if (state.sun_size == size) return;
    if (state.sun_size)
    {
        sg::destroy_view(state.sun_attachment);
        sg::destroy_view(state.sun_view);
        sg::destroy_image(state.sun_image);
    }
    state.sun_size = size;
    state.sun_image = sg::make_image(&&{
        .usage = { .depth_stencil_attachment = true },
        .width = size, .height = size, .sample_count = 1, .pixel_format = DEPTH,
    });
    state.sun_attachment = sg::make_view(&&{ .depth_stencil_attachment = { .image = state.sun_image } });
    state.sun_view = sg::make_view(&&{ .texture = { .image = state.sun_image } });
}

fn void ensure_shadow_atlas(int tile, int capacity) @local
{
    int columns = math::max(4, (int)math::ceil(math::sqrt((float)capacity)));
    int rows = math::max((capacity + columns - 1) / columns, 1);
    // within the GPU's largest texture
    int most = sg::query_limits().max_image_size_2d;
    if (most > 0) tile = math::max(math::min(tile, most / math::max(columns, rows)), 4);
    if (state.atlas_tile == tile && state.atlas_columns == columns && state.atlas_rows == rows) return;
    if (state.atlas_tile)
    {
        sg::destroy_view(state.atlas_attachment);
        sg::destroy_view(state.atlas_view);
        sg::destroy_image(state.atlas_image);
    }
    state.atlas_tile = tile;
    state.atlas_columns = columns;
    state.atlas_rows = rows;
    state.atlas_lights.clear();
    state.atlas_image = sg::make_image(&&{
        .usage = { .depth_stencil_attachment = true },
        .width = tile * columns, .height = tile * rows, .sample_count = 1, .pixel_format = DEPTH,
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

// Draws the maps of the lights with shadow.auto_update off: the tiles of those that ask for
// it, or whose tile has changed hands.
fn void update_shadow_atlas(Scene* scene, Camera* camera) @local
{
    int capacity = math::max(state.renderer.shadow_atlas_capacity, 1);
    state.atlas_pending.clear();
    int tile = 4;
    foreach (l : state.lights)
    {
        if (l.light_type != SPOT || !l.cast_shadow || !l.visible || l.intensity <= 0) continue;
        SpotLight* sl = (SpotLight*)l;
        if (sl.shadow.auto_update || state.atlas_pending.len() >= capacity) continue;
        state.atlas_pending.push(sl);
        tile = math::max(tile, sl.shadow.map_size);
    }
    int count = (int)state.atlas_pending.len();
    // keep the atlas while nothing uses it: the lights may only be switched off for a moment
    if (!count && state.atlas_tile) return;
    int old_tile = state.atlas_tile;
    ensure_shadow_atlas(tile, capacity);
    tile = state.atlas_tile;
    // a new atlas (ensure_shadow_atlas forgot its lights): every tile
    bool all = state.atlas_tile != old_tile || state.atlas_lights.len() == 0;
    // a tile is drawn again for a light that asks for it, or that it newly belongs to
    bool[64] redraw;
    int redraws;
    for (int i = 0; i < count; i++)
    {
        SpotLight* sl = state.atlas_pending[i];
        bool moved = i >= state.atlas_lights.len() || state.atlas_lights[i] != sl;
        if (all || moved || sl.shadow.needs_update || i >= redraw.len)
        {
            if (i < redraw.len) redraw[i] = true;
            redraws++;
        }
    }
    if (!redraws) return;
    // most of them: clear all of it, as a pass does
    if (redraws * 2 > count || count > redraw.len) all = true;
    state.atlas_lights.clear();
    foreach (l : state.atlas_pending) state.atlas_lights.push(l);
    Vector3 camera_pos = camera.matrix_world.get_position();
    bool top_left = depth_zero_to_one();
    sg::begin_pass(&&{
        .action = { .depth = { .load_action = all ? CLEAR : LOAD, .store_action = STORE, .clear_value = 1 } },
        .attachments = { .depth_stencil = state.atlas_attachment },
        .label = "three-spot-atlas",
    });
    sg::apply_pipeline(get_shadow_pipeline());
    for (int i = 0; i < count; i++)
    {
        if (!all && !redraw[i]) continue;
        SpotLight* sl = state.atlas_pending[i];
        sl.shadow.needs_update = false;
        Mat4 light_vp = shadow_view_proj(sl);
        Frustum light_frustum = frustum_from(light_vp);
        state.casters.clear();
        state.skip_instanced = sl.shadow.skip_instanced;
        collect(scene, camera.matrix_world_inverse, &light_frustum, camera_pos, true);
        state.skip_instanced = false;
        int columns = state.atlas_columns;
        sg::apply_viewport((i % columns) * tile, (i / columns) * tile, tile, tile, top_left);
        sg::apply_scissor_rect((i % columns) * tile, (i / columns) * tile, tile, tile, top_left);
        if (!all) clear_atlas_tile();
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

// Gathers what to draw. `lod_from` is the position LOD levels are measured from.
fn void collect(Object3D* obj, Mat4 view, Frustum* frustum, Vector3 lod_from, bool shadow_pass, float fade = 0, float fade_dir = 0) @local
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
                upload_instances_gpu(m);
                center = m.instance_center;
                radius = m.instance_radius;
            }
            Vector3 wc = m.matrix_world.transform_point(center);
            float r = radius * max_scale(m.matrix_world) + m.material.sway * 2;
            if (m.frustum_culled && !frustum.contains_sphere(wc, r)) break;
            Vector3 vc = view.transform_point(wc);
            float dist = math::max(vc.length(), r);
            DrawItem item = { m, vc.z, fade, fade_dir, 0, r * r / (dist * dist) };
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
            // near the next level's distance (or past this one's), both are drawn, dissolving
            if (l.fade > 0 && !shadow_pass)
            {
                float half = l.fade * 0.5f;
                for (sz i = 1; i < l.distances.len() && i < l.children.len(); i++)
                {
                    float t = (d - (l.distances[i] - half)) / l.fade;
                    if (t <= 0 || t >= 1) continue;
                    collect(l.children[i - 1], view, frustum, lod_from, shadow_pass, t, -1);
                    collect(l.children[i], view, frustum, lod_from, shadow_pass, t, 1);
                    return;
                }
            }
            if (pick >= 0 && pick < l.children.len()) collect(l.children[pick], view, frustum, lod_from, shadow_pass);
            return;
        default:
            break;
    }
    foreach (c : obj.children) collect(c, view, frustum, lod_from, shadow_pass, fade, fade_dir);
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

// The directional light to draw a shadow map for: the first that casts one.
fn DirectionalLight* directional_shadow_light() @local
{
    foreach (l : state.lights)
    {
        if (l.light_type == DIRECTIONAL && l.cast_shadow && l.visible && l.intensity > 0) return (DirectionalLight*)l;
    }
    return null;
}

// View-projection of a directional light's shadow box.
fn Mat4 directional_shadow_view_proj(DirectionalLight* d, Camera* camera) @local
{
    Vector3 target = d.target ? d.target.get_world_position() : (Vector3){ 0, 0, 0 };
    Vector3 dir = (d.matrix_world.get_position() - target).normalize();
    float extent = math::max(d.shadow_extent, 0.1f);
    Vector3 center = target;
    if (d.shadow_follow)
    {
        Vector3 at = camera.matrix_world.get_position();
        Mat4 cw = camera.matrix_world;
        Vector3 ahead = { -cw[2][0], 0, -cw[2][2] };
        if (ahead.length() > 1e-4f) center = at + ahead.normalize() * (extent * 0.55f);
    }
    return sun_box_view_proj(d, dir, center, extent, d.shadow.map_size);
}

// View-projection of a box of parallel rays towards `dir`, `radius` meters around `center`,
// moved a whole texel of a `size` map at a time (or the shadows' edges crawl as it travels).
fn Mat4 sun_box_view_proj(DirectionalLight* d, Vector3 dir, Vector3 center, float radius, int size) @local
{
    Vector3 up = math::abs(dir.y) > 0.99f ? (Vector3){ 1, 0, 0 } : (Vector3){ 0, 1, 0 };
    Vector3 right = up.cross(dir).normalize();
    up = dir.cross(right);
    float texel = 2 * radius / (float)math::max(size, 1);
    center -= right * (center.dot(right) % texel) + up * (center.dot(up) % texel);
    float far = d.shadow.far > 0 ? d.shadow.far : 300.0f;
    Vector3 eye = center + dir * (far * 0.5f);
    Mat4 world = mat4_look_at(eye, center, up, true);
    world[3] = { eye.x, eye.y, eye.z, 1 };
    Mat4 proj = mat4_orthographic(-radius, radius, radius, -radius, math::max(d.shadow.near, 0.01f), far, depth_zero_to_one());
    return proj.mul(world.invert());
}

// Splits the camera's view up to shadow.max_distance into the light's cascades and fits a box
// to each: around the sphere that holds the slice, so its size does not change as the camera
// turns. Fills state.sun_*; returns how many there are (0: the camera is not a perspective
// one, use one box).
fn int fit_sun_cascades(DirectionalLight* d, Camera* camera, Vector3 dir) @local
{
    int n = math::min(math::max(d.shadow.cascades, 1), 4);
    if (n < 2 || camera.camera_type != PERSPECTIVE) return 0;
    Mat4 p = camera.projection_matrix;
    // tangents of the half angles of the view
    float tx = p[0][0] != 0 ? 1 / math::abs(p[0][0]) : 1;
    float ty = p[1][1] != 0 ? 1 / math::abs(p[1][1]) : 1;
    float k = tx * tx + ty * ty;
    float near = math::max(camera.near, 0.01f);
    float far = math::max(math::min(d.shadow.max_distance > 0 ? d.shadow.max_distance : 150.0f, camera.far), near + 1);
    float lambda = clamp(d.shadow.split_lambda, 0, 1);
    Mat4 cw = camera.matrix_world;
    Vector3 at = cw.get_position();
    Vector3 fwd = ((Vector3){ -cw[2][0], -cw[2][1], -cw[2][2] }).normalize();
    state.sun_forward = fwd;
    int tile = math::max(d.shadow.map_size, 4);
    float from = near;
    for (int i = 0; i < n; i++)
    {
        float f = (float)(i + 1) / (float)n;
        float to = lambda * near * math::pow(far / near, f) + (1 - lambda) * (near + (far - near) * f);
        // the slice's sphere: centred on the view axis, as far from its near corners as from its far ones
        float z = math::min((from + to) * (1 + k) * 0.5f, to);
        float radius = math::sqrt((to - z) * (to - z) + to * to * k);
        // in steps, so that rounding does not change its size from one frame to the next
        radius = math::ceil(radius * 16) / 16;
        state.sun_view_proj[i] = sun_box_view_proj(d, dir, at + fwd * z, radius, tile);
        state.sun_rect[i] = { (float)(i % 2) * 0.5f, (float)(i / 2) * 0.5f, 0.5f, 0.5f };
        state.sun_info[i] = { to, 2 * radius / (float)tile, 0, 0 };
        from = to;
    }
    return n;
}

// Gathers what to draw into the directional light's shadow map: its cascades, each into a tile
// of a 2x2 atlas with the casters inside its own box (state.sun_casters), or one box.
fn void prepare_sun_shadow(Scene* scene, Camera* camera, DirectionalLight* sun) @local
{
    foreach (&l : state.sun_casters) l.clear();
    if (!sun)
    {
        ensure_sun_map(state.sun_size ?: 4);
        return;
    }
    Vector3 target = sun.target ? sun.target.get_world_position() : (Vector3){ 0, 0, 0 };
    Vector3 dir = (sun.matrix_world.get_position() - target).normalize();
    int tile = math::max(sun.shadow.map_size, 4);
    int n = fit_sun_cascades(sun, camera, dir);
    state.sun_cascades = n;
    if (!n)
    {
        state.sun_view_proj[0] = directional_shadow_view_proj(sun, camera);
        state.sun_rect[0] = { 0, 0, 1, 1 };
        state.sun_info[0] = { 1e9f, 2 * math::max(sun.shadow_extent, 0.1f) / (float)tile, 0, 0 };
    }
    ensure_sun_map(n ? tile * 2 : tile);
    for (int i = 0; i < math::max(n, 1); i++)
    {
        Frustum light_frustum = frustum_from(state.sun_view_proj[i]);
        gather_casters(scene, camera, &light_frustum, sun.shadow.skip_instanced, &state.sun_casters[i]);
    }
}

// Draws what prepare_sun_shadow gathered.
fn void draw_sun_shadow(DirectionalLight* sun) @local
{
    if (!sun) return;
    int n = state.sun_cascades;
    int tile = math::max(sun.shadow.map_size, 4);
    bool top_left = depth_zero_to_one();
    sg::begin_pass(&&{
        // depth is not stored by default: ask for it, the main pass samples this map
        .action = { .depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 } },
        .attachments = { .depth_stencil = state.sun_attachment },
        .label = "three-sun-cascades",
    });
    sg::apply_pipeline(get_shadow_pipeline());
    for (int i = 0; i < math::max(n, 1); i++)
    {
        if (n)
        {
            sg::apply_viewport((i % 2) * tile, (i / 2) * tile, tile, tile, top_left);
            sg::apply_scissor_rect((i % 2) * tile, (i / 2) * tile, tile, tile, top_left);
        }
        foreach (&item : state.sun_casters[i]) draw_shadow_item(item, state.sun_view_proj[i]);
    }
    sg::end_pass();
}

// Gathers the shadow casters inside `frustum` into `out`, their instances culled on the GPU
// for it.
fn void gather_casters(Scene* scene, Camera* camera, Frustum* frustum, bool skip_instanced, List{DrawItem}* out) @local
{
    state.casters.clear();
    state.skip_instanced = skip_instanced;
    collect(scene, camera.matrix_world_inverse, frustum, camera.matrix_world.get_position(), true);
    state.skip_instanced = false;
    out.clear();
    int view = cull_view(frustum);
    foreach (item : state.casters)
    {
        item.job = cull_job(item.mesh, view) + 1;
        out.push(item);
    }
    state.casters.clear();
}

// Clip space -> shadow map uv (v points down outside of GL) and 0..1 depth.
fn Mat4 shadow_bias() @local
{
    bool zo = depth_zero_to_one();
    Mat4 bias = mat4_identity();
    bias[0][0] = 0.5f;
    bias[3][0] = 0.5f;
    bias[1][1] = zo ? -0.5f : 0.5f;
    bias[3][1] = 0.5f;
    bias[2][2] = zo ? 1.0f : 0.5f;
    bias[3][2] = zo ? 0.0f : 0.5f;
    return bias;
}

fn Vector3 spot_direction(SpotLight* sl) @private
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
    LightData* g = &lights_gpu;
    g.clear();
    float exposure = r.tone_mapping == ACES_FILMIC_TONE_MAPPING ? math::max(r.tone_mapping_exposure, 1e-4f) : 0;
    p.camera_pos = { ...camera.matrix_world.get_position(), exposure };
    p.cull_min = { ...r.cull_box_min, r.cull_box ? 1 : 0 };
    p.cull_max = { ...r.cull_box_max, 0 };
    Mat4 cw = camera.matrix_world;
    p.dir_cascade_fwd = { ...((Vector3){ -cw[2][0], -cw[2][1], -cw[2][2] }).normalize(), 1 };
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
                if (!g.room(g.dirs.len(), r.max_lights)) break;
                DirectionalLight* d = (DirectionalLight*)l;
                Vector3 target = d.target ? d.target.get_world_position() : (Vector3){ 0, 0, 0 };
                Vector3 dir = l.matrix_world.get_position() - target;
                if (d == state.sun_caster)
                {
                    int n = state.sun_cascades;
                    for (int i = 0; i < math::max(n, 1); i++)
                    {
                        p.dir_shadow_mat[i] = shadow_bias().mul(state.sun_view_proj[i]);
                        p.dir_cascade_rect[i] = state.sun_rect[i];
                        p.dir_cascade_info[i] = state.sun_info[i];
                    }
                    p.dir_shadow_params = { (float)(g.dirs.len() + 1), d.shadow.normal_bias, 1.0f / (float)state.sun_size, d.shadow.radius };
                    p.dir_cascade_params = { (float)math::max(n, 1), clamp(d.shadow.cascade_blend, 0, 1),
                                             r.debug_cascades ? 1 : 0, n ? 0 : 1 };
                    p.dir_cascade_fwd.w = clamp(d.shadow.intensity, 0, 1);
                }
                g.dirs.push({ .dir = { ...dir, 0 }, .color = { ...c, 0 } });
            case POINT:
                if (!g.room(g.points.len(), r.max_lights)) break;
                PointLight* pl = (PointLight*)l;
                g.points.push({
                    .pos = { ...l.matrix_world.get_position(), pl.distance },
                    .color = { ...c, pl.decay },
                    .params = { pl.radius, 0, 0, 0 },
                });
                // a light that adds nothing needs no place in the clusters
                LightBounds b = point_light_bounds(pl);
                if (l.intensity <= 0) b = { {}, 0 };
                g.point_bounds.push(b);
            case SPOT:
                SpotLight* sl = (SpotLight*)l;
                if (!g.room(g.spots.len(), r.max_lights)) break;
                float cos_outer = math::cos(clamp(sl.angle, 0, PI / 2));
                float cos_inner = math::cos(clamp(sl.angle, 0, PI / 2) * (1 - clamp(sl.penumbra, 0, 1)));
                GpuSpotLight gs = {
                    .pos = { ...l.matrix_world.get_position(), sl.distance },
                    .dir = { ...spot_direction(sl), cos_outer },
                    .color = { ...c, sl.decay },
                    .params = { math::max(cos_inner, cos_outer + 1e-4f), sl.radius,
                                sl.specular_radius > 0 ? sl.specular_radius : sl.radius, sl.fog_glow },
                    .shadow = { -1, 1, 0, 0 },
                };
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
                    GpuSpotShadow sh;
                    sh.mat = bias.mul(shadow_view_proj(sl));
                    float size = sl == caster ? (float)state.shadow_size : (float)state.atlas_tile;
                    sh.params = { sl == caster ? 1 : 2, sl.shadow.normal_bias, 1.0f / size, sl.shadow.radius };
                    float fov = math::min(sl.angle * 2 * RAD2DEG * clamp(sl.shadow.focus, 0.05f, 1) + 4, 150.0f);
                    float width = 2 * math::tan(fov * DEG2RAD * 0.5f);
                    float far = sl.shadow.far > 0 ? sl.shadow.far : (sl.distance > 0 ? sl.distance : 50.0f);
                    sh.soft = { sl.shadow.softness, math::max(sl.shadow.near, 0.01f), far, zo ? width : -width };
                    if (sl != caster)
                    {
                        float cols = (float)state.atlas_columns;
                        float rows = (float)state.atlas_rows;
                        sh.rect = { (float)(tile % state.atlas_columns) / cols, (float)(tile / state.atlas_columns) / rows,
                                    1.0f / cols, 1.0f / rows };
                    }
                    gs.shadow = { (float)g.shadows.len(), clamp(sl.shadow.intensity, 0, 1), 0, 0 };
                    g.shadows.push(sh);
                }
                g.spots.push(gs);
                LightBounds b = spot_light_bounds(sl);
                if (l.intensity <= 0) b = { {}, 0 };
                g.spot_bounds.push(b);
        }
    }
    int ndir = (int)g.dirs.len();
    int npoint = (int)g.points.len();
    int nspot = (int)g.spots.len();
    ClusterGrid grid = cluster_grid(camera);
    p.cluster_view_proj = camera.projection_matrix.mul(camera.matrix_world_inverse);
    p.cluster_grid = { CLUSTER_X, CLUSTER_Y, CLUSTER_Z, 0 };
    p.cluster_depth = { grid.near, grid.log_scale, grid.far, 0 };
    g.upload(&grid, r.assign_clusters, r.check_compute);
    p.counts = { ndir, npoint, nspot, 0 };
    if (scene.irradiance)
    {
        IrradianceVolume* v = scene.irradiance;
        Vector3 size = v.max - v.min;
        p.gi_min = { ...v.min, v.replaces_ambient ? 2 : 1 };
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
        p.fog_color = { ...color_to_linear(f.color), state.renderer.prof_no_fog_scatter ? 0 : f.light_scatter };
        p.fog_params = { f.near, f.far, f.density, mode };
        p.fog_box_min = { ...f.clear_box_min, f.has_clear_box ? 1 : 0 };
        p.fog_box_max = { ...f.clear_box_max, clamp(f.clear_box_density, 0, 1) };
        p.fog_height = { f.height_base, f.height_falloff, 0, f.shade_density };
        Vector3 sun_dir = f.sun_direction;
        Color sun_color = f.sun_color;
        if (sun_dir == (Vector3){ 0, 0, 0 } || !sun_color)
        {
            // the scene's sun: the directional light that casts a shadow, else the first one
            Light* sun = state.sun_caster;
            if (!sun) foreach (l : state.lights) if (l.light_type == DIRECTIONAL && l.visible) { sun = l; break; }
            if (sun)
            {
                DirectionalLight* d = (DirectionalLight*)sun;
                Vector3 target = d.target ? d.target.get_world_position() : (Vector3){ 0, 0, 0 };
                if (sun_dir == (Vector3){ 0, 0, 0 }) sun_dir = (sun.matrix_world.get_position() - target).normalize();
                if (!sun_color) sun_color = sun.color;
            }
        }
        p.fog_sun_dir = { ...sun_dir, sun_dir == (Vector3){ 0, 0, 0 } ? 0 : f.sun_sharpness };
        p.fog_sun_color = { ...color_to_linear(sun_color), f.sun_shafts };
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

// Draws `m`'s geometry; with item.job, as many instances as its cull asks for.
fn int draw_geometry(DrawItem* item, bool wireframe) @local
{
    Mesh* m = item.mesh;
    BufferGeometry* g = m.geometry;
    int instances = m.instance_data.len() > 0 ? m.instance_count : 1;
    if (item.job) instances = cull_draw_count(item.job - 1);
    int elements = wireframe ? (int)g.wire_count : (int)g.index_count;
    sg::draw(0, elements, instances);
    return elements / 3 * instances;
}

fn float alpha_test_of(Material* mat) @local => state.renderer.prof_no_alpha_test ? 0 : mat.alpha_test;

// Whether the main pass's shader would cut holes into `item`.
fn bool has_holes(DrawItem* item) @local
{
    Material* mat = item.mesh.material;
    if (mat.screen_target) return false;
    return alpha_test_of(mat) > 0 || item.fade_dir != 0 || state.renderer.cull_box;
}

// Whether `item` goes through the depth prepass (see Renderer.depth_prepass): it has holes,
// and enough pixels per triangle that shading them once saves more than drawing its
// triangles twice costs (a far tree is mostly triangles, a meadow mostly pixels).
fn bool prepassed(DrawItem* item) @local
{
    if (item.mesh.material.wireframe || !has_holes(item)) return false;
    if (state.prepass_focal <= 0) return true;
    Mesh* m = item.mesh;
    int instances = m.instance_data.len() > 0 ? m.instance_count : 1;
    if (item.job) instances = cull_draw_count(item.job - 1);
    float triangles = (float)m.geometry.index_count / 3 * math::max(instances, 1);
    // the pixels of its bounding sphere's disk, about
    float pixels = PI * item.cover * state.prepass_focal * state.prepass_focal;
    return pixels > state.renderer.depth_prepass_pixels * triangles;
}

// `normals`: false when nothing reads them (the depth prepass), to save the inverse.
fn VsParams vs_params_of(DrawItem* item, Mat4 view_proj, bool normals = true) @local
{
    Mesh* m = item.mesh;
    Material* mat = m.material;
    return {
        .view_proj = view_proj,
        .model = m.matrix_world,
        .normal_mat = normals ? m.matrix_world.invert().transpose() : m.matrix_world,
        .uv_transform = uv_transform(mat),
        .sway = sway_params(mat),
        .pusher = pusher_params(),
    };
}

// The depth prepass's draw of an opaque item with holes (see Renderer.depth_prepass).
fn void draw_depth_item(DrawItem* item, Mat4 view_proj) @local
{
    Mesh* m = item.mesh;
    Material* mat = m.material;
    BufferGeometry* g = m.geometry;
    Renderer* r = state.renderer;
    float alpha_test = alpha_test_of(mat);
    sg::apply_pipeline(get_pipeline(mat.side, false, false, r.render_target, alpha_test > 0, mat.blending, DEPTH));
    SgBindings bind = { .vertex_buffers[0] = g.vbuf, .index_buffer = g.ibuf };
    bind_instances(&bind, item);
    Vec4 inst_cull = cull_bind(&bind, item.job - 1);
    Texture* map = ready(mat.map);
    bind.views[VIEW_TEX_MAP] = map.view;
    bind.samplers[SMP_TEX_SMP] = map.sampler;
    sg::apply_bindings(&bind);
    VsParams vs = vs_params_of(item, view_proj, false);
    vs.inst_cull = inst_cull;
    sg::apply_uniforms(UB_VS_PARAMS, &&{ .ptr = &vs, .size = VsParams::size });
    DepthFsParams fs = {
        .d_color = { 0, 0, 0, mat.opacity },
        .d_params = { mat.map ? 1 : 0, alpha_test, mat.vertex_colors ? 1 : 0, 0 },
        .d_screen_params = { 0, 0, item.fade, item.fade_dir },
        .d_cull_min = { ...r.cull_box_min, r.cull_box ? 1 : 0 },
        .d_cull_max = { ...r.cull_box_max, 0 },
    };
    sg::apply_uniforms(UB_DEPTH_FS_PARAMS, &&{ .ptr = &fs, .size = DepthFsParams::size });
    r.info.calls++;
    draw_geometry(item, false);
}

fn void draw_item(DrawItem* item, Mat4 view_proj, FsParams* fs, DrawKind kind = CUT) @local
{
    Mesh* m = item.mesh;
    Material* mat = m.material;
    BufferGeometry* g = m.geometry;
    if (mat.wireframe) g.upload_wire();
    Renderer* r = state.renderer;

    Texture* map = ready(mat.map);
    if (mat.screen_target) map = state.white;
    float alpha_test = alpha_test_of(mat);
    sg::apply_pipeline(get_pipeline(mat.side, mat.transparent, mat.wireframe, r.render_target, alpha_test > 0, mat.blending, kind));
    // the prepass has cut it out already (the shader takes a negative threshold as "covered")
    if (kind == PRETESTED && alpha_test > 0) alpha_test = -1;
    SgBindings bind = {
        .vertex_buffers[0] = g.vbuf,
        .index_buffer = mat.wireframe ? g.wire_ibuf : g.ibuf,
    };
    bind_instances(&bind, item);
    Vec4 inst_cull = cull_bind(&bind, item.job - 1);
    bind.views[VIEW_TEX_MAP] = map.view;
    bind.views[VIEW_TEX_NORMAL] = ready(mat.normal_map).view;
    bind.views[VIEW_TEX_ARM] = ready(mat.arm_map).view;
    bind.views[VIEW_SHADOW_MAP] = state.shadow_view;
    bind.views[VIEW_SHADOW_ATLAS] = state.atlas_view;
    bind.views[VIEW_SUN_SHADOW] = state.sun_view;
    bind.views[VIEW_FOG_VOLUME] = state.fog_view;
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
    lights_gpu.bind(&bind);
    sg::apply_bindings(&bind);

    VsParams vs = vs_params_of(item, view_proj);
    vs.inst_cull = inst_cull;
    sg::apply_uniforms(UB_VS_PARAMS, &&{ .ptr = &vs, .size = VsParams::size });

    int flags = (mat.vertex_colors ? 1 : 0) | (mat.foliage ? 2 : 0) | (m.receive_shadow ? 4 : 0) | (mat.glass ? 8 : 0)
              | (mat.vertex_light > 0 ? 16 : 0) | (mat.screen_target ? 32 : 0);
    fs.color = { ...color_to_linear(mat.color), mat.opacity };
    fs.emissive = { ...color_to_linear(mat.emissive) * mat.emissive_intensity, mat.vertex_light };
    if (r.render_target)
    {
        fs.screen_params = { 1.0f / r.render_target.width, 1.0f / r.render_target.height, item.fade, item.fade_dir };
    }
    else
    {
        fs.screen_params = { 1.0f / sapp::widthf(), 1.0f / sapp::heightf(), item.fade, item.fade_dir };
    }
    fs.specular = { ...color_to_linear(mat.specular), (float)flags };
    fs.mat_params = { (float)mat.type, mat.roughness, mat.metalness, mat.shininess };
    fs.tex_params = { mat.map ? 1 : 0, mat.normal_map ? mat.normal_scale : 0, mat.arm_map ? 1 : 0, alpha_test };
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
    r.info.triangles += draw_geometry(item, mat.wireframe);
}

fn void draw_shadow_item(DrawItem* item, Mat4 light_vp) @local
{
    Mesh* m = item.mesh;
    Material* mat = m.material;
    Texture* map = ready(mat.map);
    SgBindings bind = {
        .vertex_buffers[0] = m.geometry.vbuf,
        .index_buffer = m.geometry.ibuf,
    };
    bind_instances(&bind, item);
    Vec4 inst_cull = cull_bind(&bind, item.job - 1);
    bind.views[VIEW_S_MAP] = map.view;
    bind.samplers[SMP_S_SMP] = map.sampler;
    sg::apply_bindings(&bind);
    ShadowVsParams vs = {
        .s_view_proj = light_vp,
        .s_model = m.matrix_world,
        .s_uv_transform = uv_transform(mat),
        .s_sway = sway_params(mat),
        .s_pusher = pusher_params(),
        .s_inst_cull = inst_cull,
    };
    sg::apply_uniforms(UB_SHADOW_VS_PARAMS, &&{ .ptr = &vs, .size = ShadowVsParams::size });
    ShadowFsParams fs = { .s_params = { mat.map && !state.renderer.prof_no_alpha_test ? mat.alpha_test : 0, mat.opacity, 0, 0 } };
    sg::apply_uniforms(UB_SHADOW_FS_PARAMS, &&{ .ptr = &fs, .size = ShadowFsParams::size });
    state.renderer.info.shadow_calls++;
    state.renderer.info.shadow_triangles += draw_geometry(item, false);
}

// The instance attributes of a draw (one identity instance for a plain mesh).
fn void bind_instances(SgBindings* bind, DrawItem* item) @local
{
    Mesh* m = item.mesh;
    if (!m.instance_data.len())
    {
        bind.vertex_buffers[1] = state.identity_instance;
        return;
    }
    int offset;
    bind.vertex_buffers[1] = instance_vertex_buffer(m, &offset);
    bind.vertex_buffer_offsets[1] = offset;
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
    if (state.renderer.prof_no_fog_scatter) return;
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
    DrawItem item = { m, 0, 0, 0, 0, 1 };
    draw_item(&item, view_proj, fs, has_holes(&item) ? CUT : SOLID);
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
    ulong pt = prof_start();
    camera.update_projection_matrix();

    if (!scene.no_matrix_world_update) scene.update_matrix_world();
    // cameras don't have to be part of the scene
    if (!camera.parent) camera.update_matrix_world();
    camera.matrix_world_inverse = camera.matrix_world.invert();
    Mat4 view_proj = camera.projection_matrix.mul(camera.matrix_world_inverse);
    Vector3 camera_pos = camera.matrix_world.get_position();

    prof_stop("render: matrices", pt);
    pt = prof_start();
    state.opaque.clear();
    state.transparent.clear();
    state.casters.clear();
    state.lights.clear();
    cull_begin(self.gpu_culling, self.check_compute);
    Frustum frustum = frustum_from(view_proj);
    collect(scene, camera.matrix_world_inverse, &frustum, camera_pos, false);
    prof_stop("render: collect", pt);
    pt = prof_start();
    // front-to-back for opaque, back-to-front for transparent (view space looks down -Z)
    sort::quicksort(&state.opaque, &cmp_opaque);
    sort::quicksort(&state.transparent, &cmp_transparent);
    int view = cull_view(&frustum);
    foreach (&item : state.opaque) item.job = cull_job(item.mesh, view) + 1;
    foreach (&item : state.transparent) item.job = cull_job(item.mesh, view) + 1;
    prof_stop("render: sort + jobs", pt);
    pt = prof_start();

    // shadow map of the shadow casting spot light
    SpotLight* caster = shadow_light();
    // without a caster keep whatever map there is: renders that switch shadows off (a portal
    // view, say) must not reallocate it every frame
    ensure_shadow_map(caster ? caster.shadow.map_size : (state.shadow_size ?: 4));
    Mat4 caster_vp;
    state.spot_casters.clear();
    if (caster)
    {
        caster_vp = shadow_view_proj(caster);
        Frustum light_frustum = frustum_from(caster_vp);
        gather_casters(scene, camera, &light_frustum, caster.shadow.skip_instanced, &state.spot_casters);
    }
    // the directional light's map is its own, so a sun and a spot light can both have shadows
    prof_stop("render: spot casters", pt);
    pt = prof_start();
    DirectionalLight* sun = directional_shadow_light();
    state.sun_caster = sun;
    prepare_sun_shadow(scene, camera, sun);
    prof_stop("render: sun casters", pt);
    pt = prof_start();
    // every view's instances culled at once, before the passes that draw them
    cull_dispatch();
    prof_stop("render: cull dispatch", pt);
    pt = prof_start();

    if (caster)
    {
        sg::begin_pass(&&{
            // depth is not stored by default: ask for it, the main pass samples this map
            .action = { .depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 } },
            .attachments = { .depth_stencil = state.shadow_attachment },
            .label = "three-spot-shadow",
        });
        sg::apply_pipeline(get_shadow_pipeline());
        foreach (&item : state.spot_casters) draw_shadow_item(item, caster_vp);
        sg::end_pass();
    }
    prof_stop("render: spot shadow", pt);
    pt = prof_start();
    draw_sun_shadow(sun);
    prof_stop("render: sun shadow", pt);
    pt = prof_start();
    update_shadow_atlas(scene, camera);
    prof_stop("render: spot atlas", pt);
    pt = prof_start();
    if (!state.no_irradiance) state.no_irradiance = irradiance_volume({ 0, 0, 0 }, { 1, 1, 1 }, 1);
    state.irradiance = scene.irradiance ?: state.no_irradiance;
    state.irradiance.upload();

    FsParams fs = build_light_params(scene, camera, caster);
    prof_stop("render: lights/clusters", pt);
    pt = prof_start();
    state.fog_view = ready(state.white).view;
    if (fog_volume_wanted(scene.fog, camera, self.prof_no_fog_scatter))
    {
        RenderTarget* rt = self.render_target;
        FogShadowViews views = { state.shadow_view, state.atlas_view, state.sun_view, state.shadow_sampler };
        state.fog_view = update_fog_volume(scene, camera, &fs, rt ? rt.width : sapp::width(), rt ? rt.height : sapp::height(),
                                           views, self.time);
    }
    prof_stop("render: volumetric fog", pt);
    pt = prof_start();

    Vector3 clear = tone_mapped(self, scene.background ? scene.background : self.clear_color);
    SgPass pass = {
        .action = {
            .colors[0] = {
                .load_action = CLEAR,
                .clear_value = { clear.x, clear.y, clear.z, scene.background ? 1.0f : self.clear_alpha },
            },
        },
        .label = "three-main",
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
    // (not with a cull box: everything would have holes)
    bool prepass = self.depth_prepass && !self.cull_box;
    // pixels per unit of tan(angle) up the view, for prepassed()
    state.prepass_focal = 0;
    if (camera.camera_type == PERSPECTIVE)
    {
        RenderTarget* rt = self.render_target;
        state.prepass_focal = camera.projection_matrix[1][1] * 0.5f * (rt ? rt.height : sapp::heightf());
    }
    if (prepass)
    {
        foreach (&item : state.opaque)
        {
            if (prepassed(item)) draw_depth_item(item, view_proj);
        }
    }
    prof_stop("render: depth prepass", pt);
    pt = prof_start();
    foreach (&item : state.opaque)
    {
        DrawKind kind = SOLID;
        if (item.mesh.material.wireframe || has_holes(item)) kind = prepass && prepassed(item) ? PRETESTED : CUT;
        draw_item(item, view_proj, &fs, kind);
    }
    prof_stop("render: main opaque", pt);
    pt = prof_start();
    draw_backdrop(scene, camera, view_proj, &fs);
    if (!self.prof_no_transparent) foreach (&item : state.transparent) draw_item(item, view_proj, &fs);
    // menus and fades on top (see menu.c3)
    if (!self.render_target) draw_ui();
    sg::end_pass();
    prof_stop("render: main transparent", pt);
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
