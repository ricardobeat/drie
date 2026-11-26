// Core Renderer configuration, window/GPU lifecycle, tone mapping, and frame orchestration.
// Owns Renderer, RendererState, sokol setup (windowed via set_animation_loop or offscreen via
// init_headless), RenderInfo statistics, back-face thickness rendering, and Renderer.render().
// Frame order in render(): dynamic geometry upload -> frustum collect/sort -> shadow maps and
// GPU instance culling -> GI/clusters/volumetric fog -> worker shader warmup -> transmission
// (non-linear) -> optional deferred G-Buffer + RT shadows + sparse RC + deferred lighting ->
// forward depth prepass, background, opaque, backdrop, split transmission, transparent, overlays.
// Connects to draw_item, draw_deferred_lighting, prepare_and_draw_shadows, AtlasPacker,
// get_pipeline, and render_bake.
module drie;
import std::collections::list, std::collections::map, std::sort, std::io, std::math, std::thread, std::os::env, libc;
import sokol, drie::readback, drie::postprocessing, drie::nodes;

// ---- Renderer configuration and state

// Per-frame callback registered with Renderer.set_animation_loop(); receives elapsed milliseconds.
alias AnimationLoop = fn void(double time_ms);
// Raw window/input event callback registered with add_event_listener().
alias EventListener = fn void(SappEvent* event);
// Alias for sokol_app input/window event structs.
alias Event = SappEvent;
// Alias for sokol_app keyboard scan codes.
alias KeyCode = SappKeycode;

// Default cap per light kind (directional, point, spot) in Renderer.max_lights.
const int DEFAULT_MAX_LIGHTS            = 1024;
// Default cap for RectAreaLight count in Renderer.max_rect_area_lights.
const int DEFAULT_MAX_RECT_AREA_LIGHTS  = 16;
// Default maximum shadow atlas side length in pixels (Renderer.shadow_atlas_size).
const int DEFAULT_SHADOW_ATLAS_SIZE = 8192;
// Maximum IrradianceVolumes bound simultaneously (Scene.irradiance + Scene.irradiance_volumes).
const int MAX_IRRADIANCE_VOLUMES        = 4;

// Shadow map filtering mode (Renderer.shadow_type).
enum ShadowType {
    // Percentage-closer filtering over a disk of `shadow.radius` texels (with `shadow.softness`
    // PCSS and `normal_bias`) across sun cascades, primary spot map, and shadow atlas tiles
    // (spots, 6-face point cubes, 5-face rect area maps, and secondary directional lights).
    PCF,
    // Variance shadow maps: depth maps are blurred via two-pass separable box filter
    // (`shadow.radius`, `shadow.blur_samples`) into RG32F/RG16F moments and sampled once with
    // `shadow.bias`, across sun cascades, primary spot map, and all shadow atlas tiles
    // (including 6-face point light cubes and 5-face rect area maps).
    VSM,
}

// Main rendering context owning window settings, quality knobs, render target binding, and stats.
struct Renderer {
    int width;                      // Window width in logical pixels.
    int height;                     // Window height in logical pixels.
    int sample_count;               // Window MSAA sample count (1 or 4).
    String title;                   // Window title string.
    // Open the window fullscreen (set before set_animation_loop). F11 or Alt+Enter toggles it.
    bool fullscreen;
    Color clear_color;              // Default background clear color when Scene.background is 0.
    float clear_alpha;              // Default clear alpha (0..1).
    // Automatically sync perspective camera aspect ratios with the active viewport or target.
    bool auto_aspect;
    AnimationLoop animation_loop;   // Active per-frame callback set by set_animation_loop().
    RenderTarget* render_target;    // Active offscreen target, or null to render to the window.
    // Clear the target before drawing (restricted to `scissor` when `scissor_test` is enabled).
    // When false, preserves prior frame contents unless `Scene.background` is a solid color.
    bool auto_clear;
    // Draw viewport `{ x, y, width, height }` in target pixels from bottom-left; zeros = full target.
    int[4] viewport;
    // Scissor rectangle `{ x, y, width, height }` in target pixels from bottom-left.
    int[4] scissor;
    bool scissor_test;              // Restricts render() and clear() writes to `scissor`.
    ToneMapping tone_mapping;       // Tone mapping operator applied on SDR/window output.
    float tone_mapping_exposure;    // Exposure multiplier applied before tone mapping.
    // Mix fog after tone mapping in display space and clear background without tone mapping
    // (three.js style). When false, both fog and background are shaded and tone mapped in linear light.
    bool display_fog;
    // Window output color space: SRGB_COLOR_SPACE or DISPLAY_P3_COLOR_SPACE on wide-gamut displays.
    // Can be changed at runtime on macOS.
    ColorSpace output_color_space;
    // Enables macOS EDR HDR output (linear Display P3 where 1.0 is SDR white, up to
    // output_headroom()). Requires `hdr_output` or `hdr_window` set before set_animation_loop().
    bool hdr_output;
    // Requests a half-float swapchain at window creation so `hdr_output` can be toggled later.
    bool hdr_window;
    // Enables the deferred 4-target G-Buffer and fullscreen lighting pass on single-sampled targets.
    bool deferred;
    // Enables GPU ray-traced shadows for RectAreaLights with `shadow.ray_traced`
    // (src/lighting/ray_shadows.c3).
    bool ray_traced_shadows;
    // Discards fragments inside `cull_box_min..cull_box_max` (used for portal clipping).
    bool cull_box;
    Vector3 cull_box_min;
    Vector3 cull_box_max;
    // Multiplier for LOD switch distances (> 1 keeps higher detail further away).
    float lod_scale;
    // Optional 1x1 focus-distance map (meters in R, e.g. BokehPass.focus_map()) driving Material.defocus.
    Texture* focus_map;
    float focus_aperture;           // Lens aperture factor for Material.defocus blur radius.
    float focus_maxblur;            // Maximum defocus blur radius in pixels.
    // Distance multiplier for LOD selection during shadow passes (<= 1 matches main view).
    float shadow_lod_bias;
    // Elapsed time in seconds driving sway and node shaders; follows clock when `auto_time` is true.
    float time;
    bool auto_time;
    // Draw call, geometry, shadow atlas, and shader compilation counters from the last render().
    RenderInfo info;
    // Profiling toggles: disable alpha testing, fog light scattering, or transparent draws.
    bool prof_no_alpha_test;
    bool prof_no_fog_scatter;
    bool prof_no_transparent;
    // Maximum lights per kind (directional, point, spot) accepted per render; 0 = unbounded.
    int max_lights;
    // Maximum RectAreaLights accepted per render (evaluated unclustered on standard materials).
    int max_rect_area_lights;
    // Maximum shadow atlas dimension in pixels (capped by GPU 2D texture limit; see AtlasPacker).
    int shadow_atlas_size;
    // Custom light-clustering callback (src/renderer/clusters.c3); null uses cluster_lights_gpu().
    ClusterAssignFn assign_clusters;
    // Enables GPU per-instance frustum culling for camera and live shadow passes (cull_dispatch).
    bool gpu_culling;
    // When true on Metal, node materials render with a plain fallback while compiling on worker
    // threads; when false, draws wait for their shaders (started in parallel at render start).
    bool compile_in_background;
    // Sokol shader pool capacity read during setup_gfx(); 0 uses DEFAULT_PROGRAMS.
    int max_programs;
    // Offscreen RenderTarget recorded as the window target in save_pipeline_cache() during headless runs.
    RenderTarget* screen_stand_in;
    // Enables depth-only prepass for alpha-tested and LOD-fading meshes to preserve early-Z.
    bool depth_prepass;
    // Re-renders shadow maps inside render(); turn off for secondary views reusing current shadows.
    bool shadow_map_auto_update;
    // Shadow filtering mode (PCF or VSM); defaults to PCF.
    ShadowType shadow_type;
    // Minimum estimated screen pixels per triangle required for a mesh to enter the depth prepass.
    float depth_prepass_pixels;
    // Runs CPU reference checks against GPU clustering and instance culling, logging mismatches.
    bool check_compute;
    // Tints directional sun shadow cascades (red, green, blue, yellow, nearest first) for debugging.
    bool debug_cascades;
    // Global resolution multiplier applied to every light's `shadow.map_size`.
    float shadow_map_scale;
    // Global tap-count multiplier for soft shadow PCF sampling (capped at 40 taps per lookup).
    float shadow_sample_scale;
    // Maximum ray-march steps for screen-space reflections (capped at 128; default 40).
    int ssr_steps;
    // Maximum step length in screen pixels for SSR marching (default 6).
    float ssr_max_step;
    // Global resolution multiplier for volumetric fog grid dimensions (Fog.resolution, Fog.slices).
    float fog_resolution_scale;
    // Resolution multiplier for the separate transmission pass (non-linear targets; default 1).
    float transmission_resolution_scale;
    // Previous-frame inputs bound for screen-space reflections via set_reflection_source().
    SgView ssr_color;
    SgView ssr_depth;
    SgView ssr_back;
    // Active Reflector during render_reflector(); its own meshes are skipped during collection.
    Reflector* reflecting;
    Mat4 ssr_view_proj;
    float ssr_reach;
    bool ssr_linear;
    int ssr_frame;
}

// ---- Back-face thickness pass and render info

// Binds previous-frame `color`, `depth` (view-space distance in G), and `view_proj` for
// screen-space reflections up to `reach` meters. Pass `linear: true` for RGBA16F HDR color
// buffers, and optional `back` from render_back_faces() for object thickness estimation.
// Call before render() each frame; call clear_reflection_source() for alternate camera views.
fn void Renderer.set_reflection_source(
    &self,
    SgView color,
    SgView depth,
    Mat4 view_proj,
    float reach = 12,
    bool linear = false,
    SgView back = { }
) {
    self.ssr_linear = linear;
    self.ssr_color = color;
    self.ssr_depth = depth;
    self.ssr_back = back;
    self.ssr_view_proj = view_proj;
    self.ssr_reach = reach;
    self.ssr_frame++;
}

// Disables screen-space reflections by clearing the bound reflection source views.
fn void Renderer.clear_reflection_source(&self) {
    self.ssr_color = { };
    self.ssr_depth = { };
    self.ssr_back = { };
}

// Renders back-face view-space distances (meters in R, 0 where empty) of opaque, front-sided,
// solid meshes from the last render() into `target` (single-sampled R16F with depth). Pass
// `target.texture` as `back` to set_reflection_source() on the next frame for SSR thickness.
fn void Renderer.render_back_faces(&self, RenderTarget* target) {
    if (target.samples != 1 || target.pixel_format() != R16F || !target.depth_buffer) abort(
        "drie: back faces go into an R16F target with depth, one sample"
    );
    ulong pt = profile::start();
    if (!state.back_pipeline.id) {
        state.back_pipeline = sg::make_pipeline(
            &&{
                .shader = engine_program(false),
                .index_type = UINT32,
                .cull_mode = FRONT,
                .face_winding = CCW,
                .layout = vertex_layout(
                    {
                        shaders::ATTR_THREE_POSITION,
                        shaders::ATTR_THREE_NORMAL,
                        shaders::ATTR_THREE_UV,
                        shaders::ATTR_THREE_COLOR0,
                        shaders::ATTR_THREE_UV1,
                        shaders::ATTR_THREE_TANGENT,
                        shaders::ATTR_THREE_INST0,
                        shaders::ATTR_THREE_INST1,
                        shaders::ATTR_THREE_INST2,
                        shaders::ATTR_THREE_HEIGHT,
                        shaders::ATTR_THREE_INSTANCE_COLOR,
                        shaders::ATTR_THREE_PREV_POSITION
                    }
                ),
                .depth = { .pixel_format = DEPTH, .write_enabled = true, .compare = LESS_EQUAL },
                .colors[0].pixel_format = R16F,
                .label = "drie-back-faces",
            }
        );
    }
    profile::begin_pass(
        &&{
            .action = {
                .colors[0] = { .load_action = CLEAR, .store_action = STORE, .clear_value = { 0, 0, 0, 0 } },
                .depth = { .load_action = CLEAR, .clear_value = 1 },
            },
            .attachments = target.attachments(),
            .label = "drie-back-faces",
        }
    );
    Mat4 view_proj = state.last_view_proj_rel;
    foreach (&item : state.opaque) {
        Material* mat = item.mesh.material;
        // Skip double-sided, cutout, line, and screen-target meshes (no closed interior volume).
        if (mat.side != FRONT_SIDE || draws_lines(item) || has_holes(item) || mat.screen_target) continue;
        BufferGeometry* g = item.mesh.geometry;
        sg::apply_pipeline(state.back_pipeline);
        SgBindings bind = { .vertex_buffers[0] = g.vbuf, .index_buffer = g.ibuf };
        bind_instances(&bind, item);
        Vec4 inst_cull = cull_bind(&bind, item.job - 1);
        Texture* map   = ready(state.white);
        bind.views[shaders::VIEW_TEX_MAP] = map.view;
        bind.samplers[shaders::SMP_TEX_SMP] = map.sampler;
        bind.views[shaders::VIEW_TEX_ALPHA] = map.view;
        bind.samplers[shaders::SMP_TEX_ALPHA_SMP] = map.sampler;
        sg::apply_bindings(&bind);
        VsParams vs = vs_params_of(item, view_proj, false);
        vs.inst_cull = inst_cull;
        sg::apply_uniforms(shaders::UB_VS_PARAMS, &&{ .ptr = &vs, .size = VsParams::size });
        DepthFsParams fs = { .d_color = { 0, 0, 0, 1 } };
        sg::apply_uniforms(shaders::UB_DEPTH_FS_PARAMS, &&{ .ptr = &fs, .size = DepthFsParams::size });
        self.info.calls++;
        draw_geometry(item, false);
    }
    sg::end_pass();
    profile::stop("render: back faces", pt);
}

// Per-render statistics populated by Renderer.render().
struct RenderInfo {
    int calls;             // Main/deferred/prepass draw calls submitted.
    int triangles;         // Triangles submitted in main and deferred passes.
    int lines;             // Line segments submitted (Line, LineSegments, LineLoop).
    int shadow_calls;      // Draw calls submitted across all shadow passes.
    int shadow_triangles;  // Triangles submitted across all shadow passes.
    int atlas_width;       // Current shadow atlas texture width in pixels.
    int atlas_height;      // Current shadow atlas texture height in pixels.
    usz atlas_bytes;       // GPU bytes used by shadow atlas, static cache, and VSM atlas targets.
    int atlas_tiles;       // Shadow atlas and cache tiles rendered this frame.
    int atlas_casters;     // Caster draw calls submitted into atlas/cache tiles this frame.
    int atlas_lights;      // Active shadowed lights currently allocated in the atlas.
    int atlas_dropped;     // Shadowed lights dropped from the atlas due to budget pressure.
    int waiting;           // Draws using a fallback or skipped while background compilation runs.
    Mesh* waiting_mesh;    // Last mesh counted in `waiting`.
    int waiting_why;       // Reason for `waiting_mesh`: 1 node job, 2 shader compiling, 3 pipeline pending.
    int updates;           // Dynamic BufferGeometries re-uploaded to the GPU this frame.
    usz update_bytes;      // Total vertex bytes uploaded by dynamic geometry updates.
}

// Alias for Renderer matching three.js naming.
alias WebGLRenderer = Renderer;

// Tone mapping operators matching TONE_* in shaders/tonemapping.glsl. Applied directly on SDR
// targets and windows, or after post-processing by OutputPass in drie::postprocessing.
enum ToneMapping {
    NO_TONE_MAPPING,          // Pass-through linear values clamped to [0, 1].
    LINEAR_TONE_MAPPING,      // Exposure scaling only, clamped to [0, 1].
    REINHARD_TONE_MAPPING,    // Per-channel Reinhard curve `c / (1 + c)`.
    CINEON_TONE_MAPPING,      // Filmic Cineon curve.
    ACES_FILMIC_TONE_MAPPING, // RRT/ODT ACES Filmic approximation.
    AGX_TONE_MAPPING,         // AgX inset/outset filmic transform.
    NEUTRAL_TONE_MAPPING,     // Khronos PBR Neutral tone mapper.
}

// ---- Internal renderer state

struct DrawItem @private {
    Mesh* mesh;
    float depth;
    // LOD cross-fade progress in [0, 1] and direction (+1 fading in, -1 fading out, 0 solid).
    float fade;
    float fade_dir;
    // GPU instance culling job index + 1 (cull_job); 0 draws all instances unculled.
    int job;
    // Approximate screen coverage `(radius / distance)^2` for depth-prepass heuristics.
    float cover;
}

// VSM moment target and intermediate blur scratch image (vsm_blur).
struct VsmMaps @private {
    SgImage image;
    SgView view;
    SgView attachment;
    SgImage scratch;
    SgView scratch_view;
    SgView scratch_attachment;
    int width;
    int height;
}

// Active lightmap bake parameters set during Renderer.render_bake().
struct BakeState @local {
    bool on;
    SgPixelFormat format;
    SgView prev;
    int width;
    int height;
}

struct Frustum @private {
    Vec4[5] planes;
}

struct RendererState @private {
    Renderer* renderer;
    // Active camera matrices/position exposed to node material uniforms (drie::nodes).
    Mat4 node_view;
    Mat4 node_proj;
    Vector3 node_camera;
    List{FrameHook} frame_hooks;
    List{EventFilter} event_filters;
    List{FrameHook} overlays;
    // Camera-less sparse RadianceCascades queued to flush once commit() ends the engine frame.
    List{RadianceCascades*} pending_sparse_flush;
    bool gfx_ready;
    // Set when a shadow draw used a plain fallback shader, triggering static tile redraw next frame.
    bool shadow_plain;
    // Built-in depth (`render_back_faces`) and shadow (`clear_atlas_tile`) shaders.
    SgShader[2] engine_programs;
    // Deferred G-Buffer targets and fullscreen lighting shaders/pipelines.
    int gb_width;
    int gb_height;
    SgImage[4] gb_color;
    SgView[4] gb_att;
    SgView[4] gb_tex;
    SgShader[4] deferred_shaders;
    SgPipeline[8] deferred_pipelines;
    SgSampler gb_sampler;
    // Two-pass separable box blur for VSM shadow moments (shaders/vsm.glsl).
    SgShader vsm_blur_shader;
    SgPipeline vsm_blur_pipeline;
    SgSampler vsm_sampler;
    SgSampler vsm_point_sampler;
    SgPixelFormat vsm_format;
    // VSM moment maps sized to match the primary spot, sun cascade, and shadow atlas textures.
    VsmMaps vsm_spot;
    VsmMaps vsm_sun;
    VsmMaps vsm_atlas;
    // MRT motion-vector tracking across consecutive MRT renders.
    bool motion;
    Mat4 motion_view_proj;
    // True when swapchain uses half-float pixels; `window_colors` caches DisplayColorSpace ordinal.
    bool float_window;
    int window_colors;
    // Last render's camera-relative view-projection (for render_back_faces) and world origin.
    Mat4 last_view_proj_rel;
    Vector3 view_origin;
    // Camera world matrix used to orient billboard meshes (Mesh.billboard).
    Mat4 view_world;
    SgPipeline back_pipeline;
    // Sub-pixel clip-space jitter removed from MRT motion vectors.
    Vector2 motion_jitter;
    int motion_frame;
    float motion_time;
    float motion_prev_time;
    // Cached clear_rect() pipelines keyed by target format/sample/MRT configuration.
    HashMap{ulong, SgPipeline} pipelines;
    // Depth-only quad pipeline writing far depth to clear a single shadow atlas tile.
    SgPipeline tile_clear_pipeline;
    // Default white per-instance color buffer for instanced meshes without custom colors.
    SgBuffer whites;
    uint whites_count;
    // Fullscreen clear shader for scissored Renderer.clear() and render() clears.
    SgShader clear_shader;
    // Committed engine frame counter and last frame rendered into the window swapchain.
    uint frame;
    uint window_drawn_frame;
    // True inside on_frame(): defers commit() until animation_loop returns so multiple renders compose.
    bool in_frame;
    bool commit_pending;
    BufferGeometry* tile_quad;
    List{DrawItem} opaque;
    List{DrawItem} transparent;
    List{DrawItem} casters;
    // Gathered shadow casters for the primary spot map and up to 4 sun cascades.
    List{DrawItem} spot_casters;
    List{DrawItem}[4] sun_casters;
    List{Light*} lights;
    List{EventListener} listeners;
    ulong start_ticks;
    // Default 1x1 white fallback texture bound to unused material slots.
    Texture* white;
    // Scene currently being rendered.
    Scene* scene;
    // Empty fallback volume bound when a scene has no IrradianceVolume.
    IrradianceVolume* no_irradiance;
    // Active scene irradiance volumes and combined 3D texture atlas (`gi_view`).
    IrradianceVolume* irradiance;
    IrradianceVolume*[MAX_IRRADIANCE_VOLUMES] gi_volumes;
    int gi_count;
    SgView gi_view;
    SgImage gi_atlas;
    SgView gi_atlas_view;
    IrradianceVolume*[MAX_IRRADIANCE_VOLUMES] gi_atlas_of;
    // Temporary overrides used by bake_irradiance() to exclude/include the volume being baked.
    IrradianceVolume* gi_skip;
    IrradianceVolume* gi_add;
    // Fallback zeroed probe buffer bound when Scene.radiance is null.
    StorageBuffer* no_radiance;
    SgView radiance_view;
    // Reflection probe array texture view/sampler and 1x1 dummy array fallback.
    SgView probes_view;
    SgSampler probes_sampler;
    SgImage no_probes_image;
    SgView no_probes;
    // Single identity 3x4 instance matrix bound for non-instanced meshes.
    SgBuffer identity_instance;
    // Growable zero-filled attribute buffer bound for missing uv1/tangent/displacement streams.
    SgBuffer zeros;
    uint zeros_vertices;
    SgImage shadow_image;
    SgView shadow_attachment;
    SgView shadow_view;
    SgSampler shadow_sampler;
    SgSampler clamp_sampler;
    int shadow_size;
    // Shared DEPTH16 shadow atlas managed by AtlasPacker (src/renderer/shadow_atlas.c3).
    SgImage atlas_image;
    SgView atlas_attachment;
    SgView atlas_view;
    int atlas_width;
    int atlas_height;
    AtlasPacker atlas_packer;
    // Depth-blit pipeline and nearest sampler used by copy_depth() (shaders/atlas.glsl).
    SgPipeline atlas_copy_pipeline;
    SgSampler atlas_copy_sampler;
    // Active light allocations in the shadow atlas.
    List{AtlasSlot} atlas_slots;
    // Companion DEPTH16 static-caster cache backing dynamic atlas tile composition.
    SgImage cache_image;
    SgView cache_attachment;
    SgView cache_view;
    int cache_width;
    int cache_height;
    AtlasPacker cache_packer;
    // Per-scene static/moving caster trackers (walk_casters).
    CasterWalk[4] caster_walks;
    // Set while gathering casters for lights with `shadow.skip_instanced`.
    bool skip_instanced;
    // Primary directional sun light and its 32-bit DEPTH cascade map.
    DirectionalLight* sun_caster;
    SgImage sun_image;
    SgView sun_attachment;
    SgView sun_view;
    int sun_size;
    // Sun cascade state: count (0 = single box), projections, UV rects, and split/texel info.
    int sun_cascades;
    Mat4[4] sun_view_proj;
    Vec4[4] sun_rect;
    Vec4[4] sun_info;
    Vector3 sun_forward;
    // Far-plane sphere carrying scattered fog where no scene geometry renders.
    Mesh* backdrop;
    // 3D volumetric fog texture from update_fog_volume() (src/lighting/fog.c3), or white fallback.
    SgView fog_view;
    // Perspective focal length in pixels (0 for orthographic), used by prepassed().
    float prepass_focal;
    // Offscreen target for non-linear transmission passes (render_transmission).
    RenderTarget* transmission;
    // Mipmapped RGBA16F opaque scene copy sampled by rough transmissive materials.
    SgImage transmission_mips;
    SgView transmission_mips_view;
    SgView[16] transmission_level_views;
    SgView[16] transmission_level_attachments;
    int transmission_levels;
    int transmission_width;
    int transmission_height;
    SgPipeline mip_pipeline;
    SgSampler mip_sampler;
    // Offscreen window-sized target for multi-pass windowed frames (auto_clear = false,
    // scissor_test, sparse RC, or clear()): preserves MSAA LOAD/STORE across passes and blits
    // once to the swapchain in present_window_rt() at frame end.
    RenderTarget* window_rt;
    SgPipeline blit_pipeline;
    bool window_rt_used;
    // Reusable meshes for equirectangular, flat, and custom-material scene backgrounds.
    Mesh* background;
    Mesh* background_quad;
    Mesh* background_sphere;
    // Cached resources for Renderer.render_environment() (src/renderer/baking.c3).
    Scene* env_scene;
    Mesh* env_grid;
    OrthographicCamera* env_camera;
    RenderTarget* env_small;
    List{EnvTarget} env_targets;
    // Cached cubemap faces, camera, and resampling material for Renderer.render_probe().
    RenderTarget*[6] probe_faces;
    PerspectiveCamera* probe_camera;
    MeshBasicMaterial* probe_resample;
    // Active lightmap bake state and seam-dilation pipelines (src/renderer/baking.c3).
    BakeState bake;
    Object3D*[] caster_roots;
    SgShader seam_shader;
    SgPipeline[2] seam_pipelines;
}

RendererState state @private;

// ---- Lifecycle, window, and hooks

// Allocates a Renderer with default dimensions, MSAA sample count (4 when `antialias` is true),
// and feature defaults. Call set_animation_loop() or init_headless() to initialize the GPU.
fn Renderer* renderer(bool antialias = true, int width = 800, int height = 600, String title = "drie") {
    Renderer* r = mem::new(Renderer);
    *r = {
        .width = width,
        .height = height,
        .sample_count = antialias ? 4 : 1,
        .title = title,
        .clear_color = 0x000000,
        .clear_alpha = 1,
        .auto_aspect = true,
        .auto_clear = true,
        .tone_mapping_exposure = 1,
        .lod_scale = 1,
        .auto_time = true,
        .gpu_culling = true,
        .depth_prepass = true,
        .deferred = true,
        .ray_traced_shadows = true,
        .shadow_map_auto_update = true,
        .depth_prepass_pixels = 1.5f,
        .max_lights = DEFAULT_MAX_LIGHTS,
        .max_rect_area_lights = DEFAULT_MAX_RECT_AREA_LIGHTS,
        .shadow_atlas_size = DEFAULT_SHADOW_ATLAS_SIZE,
        .shadow_map_scale = 1,
        .shadow_sample_scale = 1,
        .ssr_steps = 40,
        .ssr_max_step = 6,
        .fog_resolution_scale = 1,
        .transmission_resolution_scale = 1,
    };
    state.renderer = r;
    state.opaque.allocator = mem;
    state.transparent.allocator = mem;
    if (!state.pipelines.is_initialized()) state.pipelines.init(mem);
    state.casters.allocator = mem;
    state.spot_casters.allocator = mem;
    foreach (&l : state.sun_casters) l.allocator = mem;
    state.lights.allocator = mem;
    state.atlas_slots.allocator = mem;
    lights_gpu.init_lists();
    state.listeners.allocator = mem;
    return r;
}

// Sets the initial window size in logical pixels (call before set_animation_loop()).
fn void Renderer.set_size(&self, int width, int height) {
    self.width = width;
    self.height = height;
}

// Directs subsequent render() and clear() calls into `target` (pass null to target the window).
fn void Renderer.set_render_target(&self, RenderTarget* target) {
    self.render_target = target;
}

// Sets the default clear color and alpha used when `Scene.background` is unset.
fn void Renderer.set_clear_color(&self, Color color, float alpha = 1) {
    self.clear_color = color;
    self.clear_alpha = alpha;
}

// Opens the application window, initializes sokol_gfx, and invokes `cb` every frame with elapsed
// milliseconds. Blocks until the window closes.
fn void Renderer.set_animation_loop(&self, AnimationLoop cb) {
    self.animation_loop = cb;
    state.renderer = self;
    sapp::run(
        &&{
            .init_cb = &on_init,
            .frame_cb = &on_frame,
            .cleanup_cb = &on_cleanup,
            .event_cb = &on_event,
            .width = self.width,
            .height = self.height,
            .sample_count = self.sample_count,
            .high_dpi = !env::EMSCRIPTEN, // (in a browser, one pixel per CSS pixel: a Retina canvas costs 4x the pixels)
            .hdr = self.hdr_window || self.hdr_output,
            .fullscreen = self.fullscreen,
            .window_title = (ZString)self.title.zstr_tcopy(),
            .icon.sokol_default = true,
            .logger.func = &slog::func,
        }
    );
}

// Registers a listener for raw window and input events (keyboard, mouse, resize).
fn void add_event_listener(EventListener listener) {
    state.listeners.push(listener);
}

// Logical window width in points (physical pixels divided by DPI scale).
fn float window_width() => sapp::isvalid() ? sapp::widthf() / sapp::dpi_scale() : (float)state.renderer.width;
// Logical window height in points (physical pixels divided by DPI scale).
fn float window_height() => sapp::isvalid() ? sapp::heightf() / sapp::dpi_scale() : (float)state.renderer.height;
// Current window aspect ratio (`window_width() / window_height()`).
fn float window_aspect() => window_width() / window_height();

// True on backends with [0, 1] clip-space depth and top-left render-target row origin (Metal,
// D3D11, Vulkan, WebGPU); false on OpenGL / GLES3 ([-1, 1] depth, bottom-left origin).
fn bool depth_zero_to_one() {
    if (!state.gfx_ready) return true;
    SgBackend b = sg::query_backend();
    return b != GLCORE && b != GLES3;
}

// Factory function returning a backend-specific sokol-shdc SgShaderDesc.
alias ShaderDescFn = fn SgShaderDesc(SgBackend backend);

// Default sokol shader pool capacity when Renderer.max_programs is 0.
const int DEFAULT_PROGRAMS = 1024;

fn void setup_gfx(SgEnvironment environment) @private {
    sg::setup(
        &&{
            .environment = environment,
            .logger.func = &slog::func, // Pool sized to sokol's maximum for dense instanced and indexed scenes.
            .buffer_pool_size = 65535, // 16 MB transient uniform buffer for thousands of draws per frame.
            .uniform_buffer_size = 16 * 1024 * 1024, // Generous pools for scenes with hundreds of materials and maps.
            .image_pool_size = 4096,
            .view_pool_size = 4096,
            .sampler_pool_size = 4096, // Covers material/pass/target/VSM/G-Buffer pipeline permutations.
            .pipeline_pool_size = 4096,
            .shader_pool_size = state.renderer && state.renderer.max_programs > 0 ? state.renderer.max_programs : DEFAULT_PROGRAMS,
        }
    );
    stm::setup();
    state.start_ticks = stm::now();
    query_gpu_caps();
    // Material shaders compile lazily on first use via material_program().
    SgBackend backend = sg::query_backend();
    state.vsm_blur_shader = make_program(shaders::vsm_shader_desc(backend));
    state.white = data_texture({ 255, 255, 255, 255 }, 1, 1, NO_COLOR_SPACE);
    float[12] identity = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
    state.identity_instance = sg::make_buffer(&&{ .data = { .ptr = &identity, .size = identity.len * float::size } });
    state.gfx_ready = true;
    load_pipeline_archive();
}

fn void on_init() @local {
    Renderer* r = state.renderer;
    state.float_window = r.hdr_window || r.hdr_output;
    state.window_colors = -1;
    setup_gfx(sglue::environment());
}

// Initializes sokol_gfx without opening a window (Metal on macOS, D3D11 on Windows) for offscreen
// RenderTarget scripts and tests. Call render() with a RenderTarget set, then drie::commit().
fn void Renderer.init_headless(&self) {
    state.renderer = self;
    setup_gfx(readback::headless_environment());
}

// Callback invoked before each frame (add_frame_hook) or during window overlay rendering (add_overlay).
alias FrameHook = fn void();
// Window event interceptor registered with add_event_filter(); return true to consume the event.
alias EventFilter = fn bool(Event* e);

// Registers `hook` to run at the start of every frame before `animation_loop`.
fn void add_frame_hook(FrameHook hook) => state.frame_hooks.push(hook);

// Registers an event interceptor run ahead of add_event_listener() callbacks; pass `first: true`
// to insert ahead of existing filters.
fn void add_event_filter(EventFilter filter, bool first = false) {
    if (first) {
        state.event_filters.insert_at(0, filter);
    } else {
        state.event_filters.push(filter);
    }
}

// Registers `overlay` to draw on top of windowed output (main pass, present_window_rt, or draw_overlays).
fn void add_overlay(FrameHook overlay) => state.overlays.push(overlay);

// Invokes all registered overlays inside the currently open swapchain pass.
fn void draw_overlays() {
    foreach (o : state.overlays) o();
}

// Effective LOD divisor derived from the active Renderer.lod_scale.
fn float lod_distance_scale() @private => state.renderer ? math::max(state.renderer.lod_scale, 0.01f) : 1;

// True when HDR EDR output is enabled and the window was created with a half-float swapchain.
fn bool Renderer.hdr_on(&self) => self.hdr_output && state.float_window;

// Syncs the swapchain's DisplayColorSpace with Renderer.hdr_output and output_color_space.
fn void apply_window_colors() @local {
    Renderer* r            = state.renderer;
    DisplayColorSpace want = r.hdr_on() ? EXTENDED_LINEAR_DISPLAY_P3
        : (r.output_color_space == DISPLAY_P3_COLOR_SPACE ? DISPLAY_P3 : SRGB);
    if (state.window_colors == want.ordinal) return;
    // Leave an unmodified 8-bit SDR window in the OS default color space.
    if (state.window_colors >= 0 || want != SRGB || state.float_window) set_display_color_space(want);
    state.window_colors = want.ordinal;
}

fn void on_frame() @local {
    apply_window_colors();
    foreach (h : state.frame_hooks) h();
    state.in_frame = true;
    state.window_rt_used = false;
    if (state.renderer.animation_loop) state.renderer.animation_loop(stm::ms(stm::since(state.start_ticks)));
    state.in_frame = false;
    if (state.window_rt_used) present_window_rt();
    if (state.commit_pending) commit();
}

fn void on_cleanup() @local {
    sg::shutdown();
}

// Dispatches a synthetic SappEvent through event filters and listeners as if sent by the window.
fn void send_event(SappEvent* e) => on_event(e);

fn void on_event(SappEvent* e) @local {
    if (e.type == KEY_DOWN && !e.key_repeat
        && (e.key_code == F11 || (e.key_code == ENTER && (e.modifiers & sapp::MODIFIER_ALT)))) {
        sapp::toggle_fullscreen();
        return;
    }
    // Give event filters (e.g. active drie::ui menus) first chance to consume input.
    foreach (f : state.event_filters) if (f(e)) return;
    foreach (l : state.listeners) l(e);
}

// ---- Clearing, viewport, and tone mapping

// Converts `color` into the target's clear space: linear for RGBA16F, raw sRGB for `display_fog`,
// or tone-mapped sRGB to match fog color in SDR renders.
fn Vector3 clear_value(Renderer* r, Color color) @private {
    if (r.render_target && r.render_target.is_linear()) return color_to_linear(color);
    if (r.display_fog) return { ((color >> 16) & 0xff) / 255.0f, ((color >> 8) & 0xff) / 255.0f, (color & 0xff) / 255.0f };
    return tone_mapped(r, color);
}

// Marks `rt` (or the window) drawn on the current frame and returns true if this is its first pass.
fn bool first_pass(RenderTarget* rt) @private {
    uint* drawn = rt ? &rt.drawn_frame : &state.window_drawn_frame;
    bool first = *drawn != state.frame + 1;
    *drawn = state.frame + 1;
    return first;
}

// Configures `pass` to preserve prior draws (clearing on the frame's first pass) and store results.
fn void keep_pass(SgPass* pass, bool first, RenderTarget* rt) @local {
    SgLoadAction load = first ? CLEAR : LOAD;
    pass.action.colors[0].load_action = load;
    pass.action.colors[0].store_action = STORE;
    pass.action.depth = { .load_action = load, .store_action = STORE, .clear_value = 1 };
    if (rt && rt.mrt) {
        for (int i = 1; i <= MRT_COUNT; i++) pass.action.colors[i] = { .load_action = load, .store_action = STORE };
    }
}

// Applies `r.viewport` (when non-zero) and `r.scissor` (when `scissor_test` is enabled).
fn void apply_view_rect(Renderer* r) @local {
    int[4] v = r.viewport;
    if (v[2] > 0 && v[3] > 0) sg::apply_viewport(v[0], v[1], v[2], v[3], false);
    int[4] s = r.scissor;
    if (r.scissor_test) sg::apply_scissor_rect(s[0], s[1], s[2], s[3], false);
}

// Draws a fullscreen triangle restricted to `r.scissor` that writes `rgba` and far depth (z = 1).
fn void clear_rect(Renderer* r, float[4] rgba) @local {
    RenderTarget* rt = r.render_target;
    int target = rt ? (rt.samples > 1 ? 2 : 1) + (rt.mrt ? 4 : (rt.is_linear() ? 2 : 0)) : 0;
    ulong key = (ulong)target;
    SgPipeline pip;
    if (try known = state.pipelines.get(key)) {
        pip = known;
    } else {
        if (!state.clear_shader.id) state.clear_shader = make_program(shaders::clear_rect_shader_desc(sg::query_backend()));
        SgPipelineDesc desc = {
            .shader = state.clear_shader,
            .depth = { .write_enabled = true, .compare = ALWAYS },
            .label = "drie-clear-rect",
        };
        if (rt) {
            desc.sample_count = rt.samples;
            desc.colors[0].pixel_format = rt.pixel_format();
            desc.depth.pixel_format = DEPTH;
            if (rt.mrt) {
                desc.color_count = 1 + MRT_COUNT;
                for (int i = 0; i < MRT_COUNT; i++) desc.colors[i + 1] = { .pixel_format = mrt_format(i), .write_mask = NONE };
            }
        }
        pip = sg::make_pipeline(&desc);
        state.pipelines.set(key, pip);
    }
    int w = rt ? rt.width : sapp::width();
    int h = rt ? rt.height : sapp::height();
    sg::apply_viewport(0, 0, w, h, false);
    int[4] s = r.scissor;
    sg::apply_scissor_rect(s[0], s[1], s[2], s[3], false);
    sg::apply_pipeline(pip);
    shaders::ClearParams p = { .clear_rgba = { rgba[0], rgba[1], rgba[2], rgba[3] } };
    sg::apply_uniforms(shaders::UB_CLEAR_PARAMS, &&{ .ptr = &p, .size = shaders::ClearParams::size });
    sg::draw(0, 3, 1);
}

// Marks the window swapchain dirty, deferring commit() until on_frame() finishes when inside it.
fn void window_drawn() @local {
    if (state.in_frame) {
        state.commit_pending = true;
    } else {
        commit();
    }
}

// Allocates or resizes the internal offscreen target used for multi-pass windowed frames.
fn RenderTarget* ensure_window_rt(Renderer* r) @local {
    int w = math::max(sapp::width(), 1);
    int h = math::max(sapp::height(), 1);
    int samples = r.sample_count > 1 ? 4 : 1;
    if (!state.window_rt) {
        state.window_rt = render_target(w, h, samples, RGBA8);
    } else {
        state.window_rt.set_size(w, h, samples);
    }
    return state.window_rt;
}

// Blits `state.window_rt` to the swapchain once at frame end and draws overlays on top, avoiding
// multiple swapchain passes per frame (which lose MSAA contents on Metal).
fn void present_window_rt() @local {
    if (!state.window_rt) return;
    mip_pipeline();
    if (!state.blit_pipeline.id) {
        state.blit_pipeline = sg::make_pipeline(
            &&{
                .shader = make_program(postprocessing::output_shader_desc(sg::query_backend())),
                .depth = { .write_enabled = false, .compare = ALWAYS },
                .label = "drie-window-blit",
            }
        );
    }
    SgPass pass = {
        .action = {
            .colors[0] = { .load_action = DONTCARE, .store_action = STORE },
            .depth = { .load_action = DONTCARE, .store_action = DONTCARE },
        },
        .swapchain = sglue::swapchain(),
        .label = "drie-present",
    };
    profile::begin_pass(&pass);
    sg::apply_pipeline(state.blit_pipeline);
    SgBindings bind;
    bind.views[postprocessing::VIEW_OUTPUT_SRC] = state.window_rt.texture;
    bind.views[postprocessing::VIEW_OUTPUT_EXPOSURE] = state.window_rt.texture;
    bind.views[postprocessing::VIEW_OUTPUT_LUT] = postprocessing::empty_lut();
    bind.samplers[postprocessing::SMP_OUTPUT_SMP] = state.mip_sampler;
    sg::apply_bindings(&bind);
    postprocessing::OutputParams p = { .params = { 0, 1, 0, 1 } };
    sg::apply_uniforms(postprocessing::UB_OUTPUT_PARAMS, &&{ .ptr = &p, .size = postprocessing::OutputParams::size });
    sg::draw(0, 3, 1);
    draw_overlays();
    sg::end_pass();
    state.commit_pending = true;
}

// Clears the active target (or `scissor` rectangle when `scissor_test` is enabled) to
// `clear_color`/`clear_alpha` and far depth.
fn void Renderer.clear(&self) {
    if (!state.gfx_ready) return;
    RenderTarget* saved_rt = self.render_target;
    if (!saved_rt && state.in_frame) {
        self.render_target = ensure_window_rt(self);
        state.window_rt_used = true;
    }
    defer self.render_target = saved_rt;
    bool first = first_pass(self.render_target);
    Vector3 c = clear_value(self, self.clear_color);
    SgPass pass = {
        .action = {
            .colors[0] = { .load_action = CLEAR, .store_action = STORE, .clear_value = { c.x, c.y, c.z, self.clear_alpha } },
            .depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 },
        },
        .label = "drie-clear",
    };
    if (self.scissor_test) keep_pass(&pass, first, self.render_target);
    if (self.render_target) {
        pass.attachments = self.render_target.attachments();
        if (self.render_target.mrt) {
            for (int i = 1; i <= MRT_COUNT; i++) {
                if (!self.scissor_test) pass.action.colors[i] = { .load_action = CLEAR, .store_action = STORE };
            }
        }
    } else {
        pass.swapchain = sglue::swapchain();
    }
    profile::begin_pass(&pass);
    if (self.scissor_test) clear_rect(self, { c.x, c.y, c.z, self.clear_alpha });
    sg::end_pass();
    if (!self.render_target) window_drawn();
}

// Sets the render viewport in pixels from the target's bottom-left (0 width/height = full target).
fn void Renderer.set_viewport(&self, int x, int y, int width, int height) => self.viewport = { x, y, width, height };

// Sets the scissor rectangle in pixels from the target's bottom-left (active when `scissor_test` is true).
fn void Renderer.set_scissor(&self, int x, int y, int width, int height) => self.scissor = { x, y, width, height };

// Tone-maps and sRGB-encodes `color` to match shader fog output in SDR targets.
fn Vector3 tone_mapped(Renderer* r, Color color) @private {
    Vector3 c = tone_map(color_to_linear(color), r.tone_mapping, r.tone_mapping_exposure);
    Vector3 result;
    for (int i = 0; i < 3; i++) {
        float x = clamp(c[i], 0, 1);
        result[i] = x <= 0.0031308f ? x * 12.92f : 1.055f * math::pow(x, 1.0f / 2.4f) - 0.055f;
    }
    return result;
}

fn Vector3 mat3_mul(float[9] m, Vector3 v) @local {
    return {
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z
    };
}

// Applies tone mapping `mode` and `exposure` to linear RGB `c`, matching shaders/tonemapping.glsl.
// Returns clamped linear RGB in [0, 1].
fn Vector3 tone_map(Vector3 c, ToneMapping mode, float exposure) {
    if (mode == NO_TONE_MAPPING) return c;
    c = (Vector3) { math::max(c.x, 0.0f), math::max(c.y, 0.0f), math::max(c.z, 0.0f) } * math::max(exposure, 1e-4f);
    Vector3 out;
    switch (mode) {
        case LINEAR_TONE_MAPPING:
            out = c;
        case REINHARD_TONE_MAPPING:
            out = { c.x / (1 + c.x), c.y / (1 + c.y), c.z / (1 + c.z) };
        case CINEON_TONE_MAPPING:
            for (int i = 0; i < 3; i++) {
                float x = math::max(0.0f, c[i] - 0.004f);
                out[i] = math::pow((x * (6.2f * x + 0.5f)) / (x * (6.2f * x + 1.7f) + 0.06f), 2.2f);
            }
        case ACES_FILMIC_TONE_MAPPING:
            Vector3 v = mat3_mul(
                { 0.59719f, 0.35458f, 0.04823f, 0.07600f, 0.90834f, 0.01566f, 0.02840f, 0.13383f, 0.83777f },
                c / 0.6f
            );
            for (int i = 0; i < 3; i++) {
                float x = v[i];
                v[i] = (x * (x + 0.0245786f) - 0.000090537f) / (x * (0.983729f * x + 0.4329510f) + 0.238081f);
            }
            out = mat3_mul(
                { 1.60475f, -0.53108f, -0.07367f, -0.10208f, 1.10813f, -0.00605f, -0.00327f, -0.07276f, 1.07602f },
                v
            );
        case AGX_TONE_MAPPING:
            // Row-major form of the column-major AgX inset/outset matrices in shaders/tonemapping.glsl.
            Vector3 v = mat3_mul(
                { 0.6274f, 0.3293f, 0.0433f, 0.0691f, 0.9195f, 0.0113f, 0.0164f, 0.0880f, 0.8956f },
                c
            );
            v = mat3_mul(
                {
                    0.856627153315983f,
                    0.0951212405381588f,
                    0.0482516061458583f,
                    0.137318972929847f,
                    0.761241990602591f,
                    0.101439036467562f,
                    0.11189821299995f,
                    0.0767994186031903f,
                    0.811302368396859f
                },
                v
            );
            for (int i = 0; i < 3; i++) {
                float x  = clamp((math::log2(math::max(v[i], 1e-10f)) + 12.47393f) / (4.026069f + 12.47393f), 0, 1);
                float x2 = x * x;
                float x4 = x2 * x2;
                v[i] = 15.5f * x4 * x2 - 40.14f * x4 * x + 31.96f * x4 - 6.868f * x2 * x + 0.4298f * x2 + 0.1191f * x
                    - 0.00232f;
            }
            v = mat3_mul(
                {
                    1.1271005818144368f,
                    -0.11060664309660323f,
                    -0.016493938717834573f,
                    -0.1413297634984383f,
                    1.157823702216272f,
                    -0.016493938717834257f,
                    -0.14132976349843826f,
                    -0.11060664309660294f,
                    1.2519364065950405f
                },
                v
            );
            for (int i = 0; i < 3; i++) v[i] = math::pow(math::max(v[i], 0.0f), 2.2f);
            out = mat3_mul(
                { 1.6605f, -0.5876f, -0.0728f, -0.1246f, 1.1329f, -0.0083f, -0.0182f, -0.1006f, 1.1187f },
                v
            );
        case NEUTRAL_TONE_MAPPING:
            float start = 0.8f - 0.04f;
            float x = math::min(c.x, math::min(c.y, c.z));
            float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
            out = c - (Vector3) { offset, offset, offset };
            float peak = math::max(out.x, math::max(out.y, out.z));
            if (peak >= start) {
                float d        = 1 - start;
                float new_peak = 1 - d * d / (peak + d - start);
                out *= new_peak / peak;
                float g = 1 - 1 / (0.15f * (peak - new_peak) + 1);
                out = out + ((Vector3) { new_peak, new_peak, new_peak } - out) * g;
            }
        default:
            out = c;
    }
    return { clamp(out.x, 0, 1), clamp(out.y, 0, 1), clamp(out.z, 0, 1) };
}

// ---- Frame rendering and commit

// Records a draw waiting on background shader/pipeline compilation into RenderInfo.waiting.
fn void note_waiting(Mesh* m, int why) @private {
    if (!state.renderer) return;
    state.renderer.info.waiting++;
    state.renderer.info.waiting_mesh = m;
    state.renderer.info.waiting_why = why;
}

// Logs diagnostic details to stderr for the last mesh waiting on background compilation
// (`RenderInfo.waiting_mesh`) and its NodeBuild program job states.
fn void Renderer.say_waiting(&self) {
    Mesh* m = self.info.waiting_mesh;
    if (!self.info.waiting || !m || !m.material) return;
    NodeBuild* b = node_build_of(m.material);
    io::eprintfn("drie: %d draws waiting; one is mesh '%s', %d triangles, why %d, node build %s",
        self.info.waiting, m.name, m.geometry ? m.geometry.vertex_count() : 0, self.info.waiting_why, b ? "yes" : "no");
    if (!b || !b.programs.is_initialized()) return;
    io::eprintfn("  node build busy_now %s epoch %d/%d", b.busy_now, b.busy_epoch, node_epoch);
    b.programs.@each(; KeyWords words, MaterialProgram* mp) {
        io::eprintfn("  program %x%032x: job %d compiling %s, depth %s, pipelines made %d", words.hi, words.lo, mp.program.job,
                     mp.program.compiling(), mp.depth ? (mp.depth.compiling() ? "compiling" : "made") : "none", mp.pipelines.len());
    };
}

// Renders `scene` from `camera` into `self.render_target` (or the window swapchain when null).
// Call once per view from the animation loop; when targeting a RenderTarget, call drie::commit()
// after post-processing completes.
fn void Renderer.render(&self, Scene* scene, Camera* camera) {
    if (!state.gfx_ready) return;
    if (self.auto_time) self.time = (float)stm::sec(stm::since(state.start_ticks));
    self.info = { };
    // Flush pending dynamic vertex buffer uploads outside any GPU pass.
    self.info.updates = update_dynamic(&self.info.update_bytes);

    // 1. Camera matrices, sub-pixel jitter, and MRT motion-vector history.
    if (self.auto_aspect && camera.camera_type == PERSPECTIVE && !camera.custom_projection) {
        PerspectiveCamera* pc = (PerspectiveCamera*)camera;
        RenderTarget* rt      = self.render_target;
        pc.aspect = self.viewport[2] > 0 && self.viewport[3] > 0 ? (float)self.viewport[2] / self.viewport[3]
            : rt ? (float)rt.width / rt.height : sapp::widthf() / sapp::heightf();
    }
    ulong pt = profile::start();
    if (!camera.custom_projection) camera.update_projection_matrix();

    if (!scene.no_matrix_world_update) scene.update_matrix_world();
    state.scene = scene;
    // Detached cameras still need their world matrix updated.
    if (!camera.parent) camera.update_matrix_world();
    camera.matrix_world_inverse = camera.matrix_world.invert();
    Mat4 view_proj        = camera.projection_matrix.mul(camera.matrix_world_inverse);
    Mat4 steady_view_proj = view_proj;
    Vector3 camera_pos    = camera.matrix_world.get_position();
    // Preserve outer camera uniforms across nested renders (such as Reflector passes).
    Mat4 outer_view = state.node_view;
    Mat4 outer_proj = state.node_proj;
    Vector3 outer_camera = state.node_camera;
    defer {
        state.node_view = outer_view;
        state.node_proj = outer_proj;
        state.node_camera = outer_camera;
    }
    node_epoch++;
    if (state.renderer) state.renderer.pump_pipeline_cache();
    state.node_view = camera.matrix_world_inverse;
    state.node_proj = camera.projection_matrix;
    state.node_camera = camera_pos;
    Mat4 view_proj_rel    = relative_view_proj(camera, camera_pos);
    state.motion_jitter = { };
    if (camera.jitter != { }) {
        // Apply post-projection clip-space jitter (`x' = x + s * w`) for TAA.
        RenderTarget* rt = self.render_target;
        Vector2 s = {
            2 * camera.jitter.x / (rt ? (float)rt.width : sapp::widthf()),
            2 * camera.jitter.y / (rt ? (float)rt.height : sapp::heightf())
        };
        for (int c = 0; c < 4; c++) {
            view_proj[c][0] += s.x * view_proj[c][3];
            view_proj[c][1] += s.y * view_proj[c][3];
            view_proj_rel[c][0] += s.x * view_proj_rel[c][3];
            view_proj_rel[c][1] += s.y * view_proj_rel[c][3];
        }
        state.motion_jitter = s;
    }
    state.motion = self.render_target && self.render_target.mrt;
    if (state.motion) {
        state.motion_frame++;
        state.motion_prev_time = state.motion_frame > 1 ? state.motion_time : self.time;
        state.motion_time = self.time;
        camera.motion_prev = camera.motion_frame == state.motion_frame - 1 ? camera.motion_cur : steady_view_proj;
        camera.motion_cur = steady_view_proj;
        camera.motion_frame = state.motion_frame;
        state.motion_view_proj = camera.motion_prev;
    }

    // 2. Frustum collection, sorting, and GPU instance-cull job registration.
    profile::stop("render: matrices", pt);
    pt = profile::start();
    state.last_view_proj_rel = view_proj_rel;
    state.view_origin = camera_pos;
    state.view_world = camera.matrix_world;
    state.opaque.clear();
    state.transparent.clear();
    state.casters.clear();
    state.lights.clear();
    cull_begin(self.gpu_culling, self.check_compute);
    Frustum frustum = frustum_from(view_proj);
    if (self.scissor_test && self.scissor[2] > 0 && self.scissor[3] > 0) {
        float tw = self.render_target ? (float)self.render_target.width : sapp::widthf();
        float th = self.render_target ? (float)self.render_target.height : sapp::heightf();
        if (tw > 0 && th > 0) {
            float nx0 = clamp(2.0f * (float)self.scissor[0] / tw - 1.0f, -1.0f, 1.0f);
            float nx1 = clamp(2.0f * (float)(self.scissor[0] + self.scissor[2]) / tw - 1.0f, -1.0f, 1.0f);
            float ny0 = clamp(2.0f * (float)self.scissor[1] / th - 1.0f, -1.0f, 1.0f);
            float ny1 = clamp(2.0f * (float)(self.scissor[1] + self.scissor[3]) / th - 1.0f, -1.0f, 1.0f);
            frustum = frustum_from_ndc(view_proj, nx0, nx1, ny0, ny1);
        }
    }
    collect(scene, camera.matrix_world_inverse, &frustum, camera_pos, false);
    profile::stop("render: collect", pt);
    pt = profile::start();
    // Sort opaque front-to-back and transparent back-to-front (camera looks down -Z).
    sort::quicksort(&state.opaque, &cmp_opaque);
    sort::quicksort(&state.transparent, &cmp_transparent);
    int view = cull_view(&frustum);
    foreach (&item : state.opaque) item.job = cull_job(item.mesh, view) + 1;
    foreach (&item : state.transparent) item.job = cull_job(item.mesh, view) + 1;
    profile::stop("render: sort + jobs", pt);

    // 3. Shadow passes (also dispatches GPU culling), GI/radiance upload, clusters, and fog.
    SpotLight* caster = prepare_and_draw_shadows(scene, camera);
    pt = profile::start();
    use_irradiance(scene);
    if (!state.no_radiance) state.no_radiance = storage_buffer(6 * 16, label: "drie-rc-none");
    if (scene.radiance) scene.radiance.upload();
    state.radiance_view = scene.radiance ? scene.radiance.irradiance_buf.view : state.no_radiance.view;

    FsParams fs = build_light_params(scene, camera, caster);
    profile::stop("render: lights/clusters", pt);
    pt = profile::start();
    state.fog_view = ready(state.white).view;
    if (fog_volume_wanted(scene.fog, camera, self.prof_no_fog_scatter)) {
        RenderTarget* rt     = self.render_target;
        FogShadowViews views = { state.shadow_view, state.atlas_view, state.sun_view, state.shadow_sampler,
                                 lights_gpu.cookie ? lights_gpu.cookie.view : ready(state.white).view, clamp_sampler() };
        state.fog_view = update_fog_volume(
            scene,
            camera,
            &fs,
            rt ? rt.width : sapp::width(),
            rt ? rt.height : sapp::height(),
            views,
            self.time,
            self.fog_resolution_scale
        );
    }
    profile::stop("render: volumetric fog", pt);
    pt = profile::start();

    // 4. Parallel shader warmup and non-linear transmission pre-pass.
    start_material_programs(self, scene, camera, &fs, self.depth_prepass && !self.cull_box);
    profile::stop("render: start programs", pt);
    pt = profile::start();
    bool refraction;
    foreach (&item : state.opaque) refraction = refraction || refracts(item);
    foreach (&item : state.transparent) refraction = refraction || refracts(item);
    // Linear targets split the main pass via copy_for_transmission(); SDR targets render a
    // separate linear pass first.
    bool split = refraction && self.render_target && self.render_target.is_linear();
    if (refraction && !split) {
        render_transmission(self, scene, camera, view_proj_rel, &fs);
        // Re-warm programs whose MaterialKey depends on `state.transmission_mips` now being allocated.
        start_material_programs(self, scene, camera, &fs, self.depth_prepass && !self.cull_box);
    }
    profile::stop("render: transmission", pt);
    pt = profile::start();

    // Multi-pass windowed frames route into `state.window_rt` so MSAA LOAD/STORE is preserved
    // and presented once in present_window_rt() at frame end.
    RenderTarget* saved_window_rt = self.render_target;
    if (!saved_window_rt && state.in_frame && (state.window_rt_used || !self.auto_clear || self.scissor_test
        || (scene.radiance && scene.radiance.sparse))) {
        self.render_target = ensure_window_rt(self);
        state.window_rt_used = true;
    }
    defer self.render_target = saved_window_rt;

    Color background = scene.background ? scene.background : self.clear_color;
    Vector3 clear    = clear_value(self, background);
    SgPass pass = {
        .action = {
            .colors[0] = {
                .load_action = CLEAR,
                .clear_value = { clear.x, clear.y, clear.z, scene.background ? 1.0f : self.clear_alpha },
            },
        },
        .label = "drie-main",
    };
    if (self.render_target) {
        // Store depth for RenderTarget.depth_texture and post-processing passes.
        pass.action.depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 };
        pass.attachments = self.render_target.attachments();
        if (self.render_target.mrt) {
            for (int i = 1; i <= MRT_COUNT; i++) pass.action.colors[i] = { .load_action = CLEAR };
        }
    } else {
        pass.swapchain = sglue::swapchain();
    }
    bool clears = self.auto_clear || (scene.background && !scene.background_texture);
    bool whole = clears && !self.scissor_test;
    bool first = first_pass(self.render_target);
    if (!whole) keep_pass(&pass, first, self.render_target);
    if (split) {
        // Preserve color/MRT attachments across copy_for_transmission() split passes.
        for (int i = 0; i <= (self.render_target.mrt ? MRT_COUNT : 0); i++) pass.action.colors[i].store_action = STORE;
    }
    // Disable depth prepass when cull_box is active (every mesh would require cutout).
    bool prepass = self.depth_prepass && !self.cull_box;
    state.prepass_focal = 0;
    if (camera.camera_type == PERSPECTIVE) {
        RenderTarget* rt = self.render_target;
        state.prepass_focal = camera.projection_matrix[1][1] * 0.5f * (rt ? rt.height : sapp::heightf());
    }

    // 5. Deferred path: G-Buffer pass, ray-traced shadows, sparse RC update, and fullscreen lighting.
    bool use_deferred = can_defer(self, clears, first);
    if (use_deferred) {
        bool has_deferred = false;
        foreach (&item : state.opaque) {
            if (is_deferred_item(item)) {
                has_deferred = true;
                break;
            }
        }
        use_deferred = has_deferred;
    }
    if (use_deferred) {
        RenderTarget* rt = self.render_target;
        ensure_gbuffer(rt.width, rt.height);
        SgPass gb_pass = {
            .action = {
                .colors[0] = { .load_action = CLEAR, .store_action = STORE },
                .colors[1] = { .load_action = CLEAR, .store_action = STORE },
                .colors[2] = { .load_action = CLEAR, .store_action = STORE },
                .colors[3] = { .load_action = CLEAR, .store_action = STORE },
                .depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 },
            },
            .attachments = {
                .colors[0] = state.gb_att[0],
                .colors[1] = state.gb_att[1],
                .colors[2] = state.gb_att[2],
                .colors[3] = state.gb_att[3],
                .depth_stencil = rt.depth_view,
            },
            .label = "drie-gbuffer",
        };
        profile::begin_pass(&gb_pass);
        apply_view_rect(self);
        foreach (&item : state.opaque) {
            if (!is_deferred_item(item)) continue;
            DrawKind kind = has_holes(item) ? CUT : SOLID;
            draw_item(item, view_proj_rel, &fs, kind, deferred_gb: true);
        }
        sg::end_pass();
        profile::stop("render: gbuffer", pt);
        pt = profile::start();

        rt_trace(scene, camera, view_proj_rel, rt.depth, state.gb_color[3], rt.width, rt.height);

        if (scene.radiance && scene.radiance.sparse && !self.reflecting) {
            scene.radiance.update_sparse(
                scene,
                camera,
                rt.depth_texture,
                state.gb_tex[1],
                state.gb_sampler,
                view_proj_rel.invert(),
                camera_pos,
                rt.width,
                rt.height
            );
        }

        SgLoadAction light_load = self.scissor_test ? SgLoadAction.CLEAR : SgLoadAction.DONTCARE;
        SgPass light_pass = {
            .action = {
                .colors[0] = { .load_action = light_load, .store_action = STORE, .clear_value = { clear.x, clear.y, clear.z, scene.background ? 1.0f : self.clear_alpha } },
            },
            .attachments = { .colors[0] = rt.color_view },
            .label = "drie-deferred-light",
        };
        if (rt.mrt) {
            for (int i = 0; i < MRT_COUNT; i++) {
                light_pass.action.colors[i + 1] = { .load_action = light_load, .store_action = STORE };
                light_pass.attachments.colors[i + 1] = rt.mrt_color_view[i];
            }
        }
        profile::begin_pass(&light_pass);
        apply_view_rect(self);
        draw_deferred_lighting(self, scene, camera, view_proj_rel, &fs, clear, scene.background ? 1.0f : self.clear_alpha);
        sg::end_pass();
        profile::stop("render: deferred light", pt);
        pt = profile::start();

        pass.action.colors[0].load_action = LOAD;
        pass.action.colors[0].store_action = STORE;
        pass.action.depth = { .load_action = LOAD, .store_action = STORE };
        if (rt.mrt) {
            for (int i = 1; i <= MRT_COUNT; i++) {
                pass.action.colors[i] = { .load_action = LOAD, .store_action = STORE };
            }
        }
    }

    // 6. Main forward pass: depth prepass, background, opaque, fog backdrop, glass, transparent, overlays.
    profile::begin_pass(&pass);
    if (!use_deferred && !whole && clears) clear_rect(self, { clear.x, clear.y, clear.z, scene.background ? 1.0f : self.clear_alpha });
    apply_view_rect(self);
    if (prepass) {
        foreach (&item : state.opaque) {
            if (use_deferred && is_deferred_item(item)) continue;
            if (prepassed(item)) draw_item(item, view_proj_rel, &fs, DEPTH);
        }
    }
    profile::stop("render: depth prepass", pt);
    pt = profile::start();
    draw_background(scene, camera, view_proj_rel, &fs);
    foreach (&item : state.opaque) {
        if (use_deferred && is_deferred_item(item)) continue;
        // Transmissive opaque meshes render after copy_for_transmission() when `split` is true.
        if (split && refracts(item)) continue;
        DrawKind kind = SOLID;
        if (draws_lines(item) || has_holes(item)) kind = prepass && prepassed(item) ? PRETESTED : CUT;
        draw_item(item, view_proj_rel, &fs, kind);
    }
    profile::stop("render: main opaque", pt);
    pt = profile::start();
    draw_backdrop(scene, camera, view_proj_rel, &fs);
    bool backs;
    if (split) {
        profile::stop("render: backdrop", pt);
        pt = profile::start();
        // Double-sided glass draws back faces first, copies again, then draws front faces.
        for (int list = 0; list < 2 && !backs; list++) {
            foreach (&item : list == 0 ? state.opaque : state.transparent) {
                if (refracts(item) && item.mesh.material.side == DOUBLE_SIDE) backs = true;
            }
        }
        copy_for_transmission(self, &pass);
        if (backs) {
            for (int list = 0; list < 2; list++) {
                foreach (&item : list == 0 ? state.opaque : state.transparent) {
                    Material* mat = item.mesh.material;
                    if (!refracts(item) || mat.side != DOUBLE_SIDE) continue;
                    mat.side = BACK_SIDE;
                    draw_item(item, view_proj_rel, &fs, list == 0 && !mat.wireframe && !has_holes(item) ? SOLID : CUT);
                    mat.side = DOUBLE_SIDE;
                }
            }
            copy_for_transmission(self, &pass);
        }
        foreach (&item : state.opaque) {
            if (!refracts(item)) continue;
            Material* mat = item.mesh.material;
            Side side = mat.side;
            if (backs && side == DOUBLE_SIDE) mat.side = FRONT_SIDE;
            draw_item(item, view_proj_rel, &fs, mat.wireframe || has_holes(item) ? CUT : SOLID);
            mat.side = side;
        }
        profile::stop("render: transmission", pt);
        pt = profile::start();
    }
    if (!self.prof_no_transparent) {
        foreach (&item : state.transparent) {
            Material* mat = item.mesh.material;
            Side side = mat.side;
            // Back faces of double-sided transmissive meshes were already drawn in the split block.
            if (split && backs && side == DOUBLE_SIDE && refracts(item)) mat.side = FRONT_SIDE;
            // Without split, render double-sided glass back-then-front so near reflections stay on top.
            if (!split && side == DOUBLE_SIDE && refracts(item)) {
                mat.side = BACK_SIDE;
                draw_item(item, view_proj_rel, &fs);
                mat.side = FRONT_SIDE;
            }
            draw_item(item, view_proj_rel, &fs);
            mat.side = side;
        }
    }
    if (!self.render_target) draw_overlays();
    sg::end_pass();
    profile::stop("render: main transparent", pt);

    // 7. Sparse RC fallback on forward/MSAA MRT targets and swapchain commit handoff.
    if (!use_deferred && scene.radiance && scene.radiance.sparse && !self.reflecting
        && self.render_target && self.render_target.depth_texture.id && self.render_target.normal_texture.id) {
        if (!state.gb_sampler.id) {
            state.gb_sampler = sg::make_sampler(
                &&{ .min_filter = NEAREST, .mag_filter = NEAREST, .wrap_u = CLAMP_TO_EDGE, .wrap_v = CLAMP_TO_EDGE }
            );
        }
        scene.radiance.update_sparse(
            scene,
            camera,
            self.render_target.depth_texture,
            self.render_target.normal_texture,
            state.gb_sampler,
            view_proj_rel.invert(),
            camera_pos,
            self.render_target.width,
            self.render_target.height
        );
    }
    if (!self.render_target) window_drawn();
}

// Registers a camera-less sparse RadianceCascades to flush once commit() ends the current frame.
fn void queue_sparse_flush(RadianceCascades* rc) {
    foreach (q : state.pending_sparse_flush) if (q == rc) return;
    state.pending_sparse_flush.push(rc);
}

// Removes `rc` from the end-of-frame sparse flush queue (called by RadianceCascades.dispose()).
fn void unqueue_sparse_flush(RadianceCascades* rc) {
    for (sz i = state.pending_sparse_flush.len(); i > 0; i--) {
        if (state.pending_sparse_flush[i - 1] == rc) state.pending_sparse_flush.remove_at(i - 1);
    }
}

// Flushes queued sparse radiance updates, increments the engine frame counter, and submits the
// sokol command buffer. Required after offscreen RenderTarget rendering or post-processing.
fn void commit() {
    foreach (rc : state.pending_sparse_flush) rc.flush_pending_sparse();
    state.pending_sparse_flush.clear();
    state.commit_pending = false;
    state.frame++;
    profile::commit();
}