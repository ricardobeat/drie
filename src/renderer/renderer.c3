// Renderer: owns the window and the GPU, draws a Scene from a Camera.
module drie;
import std::collections::list, std::collections::map, std::sort, std::io, std::math, std::thread, std::os::env, libc;
import sokol, drie::readback, drie::postprocessing, drie::nodes;

alias AnimationLoop = fn void(double time_ms);
alias EventListener = fn void(SappEvent* event);
alias Event = SappEvent;
alias KeyCode = SappKeycode;

// Defaults of Renderer.max_lights, max_rect_area_lights and shadow_atlas_size.
const int DEFAULT_MAX_LIGHTS            = 1024;
const int DEFAULT_MAX_RECT_AREA_LIGHTS  = 16;
const int DEFAULT_SHADOW_ATLAS_SIZE = 8192;
// Irradiance volumes a scene can have (Scene.irradiance and irradiance_volumes), as many as
// the shader's gi_* uniforms hold.
const int MAX_IRRADIANCE_VOLUMES        = 4;

// How shadow maps are filtered (Renderer.shadow_type), as three's renderer.shadowMap.type.
enum ShadowType {
    // percentage-closer: taps of the depth map over a disk of shadow.radius texels (drie's own
    // soft shadows: shadow.softness, normal_bias)
    PCF,
    // three's VSMShadowMap: each map's depths are blurred into their mean and spread
    // (shadow.radius, shadow.blur_samples) and looked up once (shadow.bias); soft, wide edges
    // cheaply. Point lights cast no shadows either way.
    VSM,
}

struct Renderer {
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
    // three's autoClear: render() clears the target before it draws (with scissor_test only
    // the scissor box, as WebGL's clear). Off, it draws over what the frame has there, unless
    // the scene's background is a color, which clears as in three.
    bool auto_clear;
    // Where render() draws, in pixels of the target from its bottom left: x, y, width, height
    // (three's setViewport). All zeros: the whole target.
    int[4] viewport;
    // three's setScissor and setScissorTest: render() and clear() touch only this box (pixels
    // from the bottom left).
    int[4] scissor;
    bool scissor_test;
    ToneMapping tone_mapping;
    float tone_mapping_exposure;
    // Fog and the background color as three has them: the fog mixed in after tone mapping, in
    // the colors that are shown, and the background color cleared to as it is, not tone mapped.
    // Otherwise both are light like any other, tone mapped with the rest of the picture.
    bool display_fog;
    // What the window takes, like three's outputColorSpace: SRGB_COLOR_SPACE, or
    // DISPLAY_P3_COLOR_SPACE on a wide-gamut screen, whose extra colors tone mapping then keeps
    // instead of clipping them to sRGB's. Can change at any time (macOS).
    ColorSpace output_color_space;
    // HDR output (macOS, EDR): the window takes linear light in Display P3 with 1 as SDR white,
    // and what is brighter shows brighter, up to output_headroom(). Tone mapping (OutputPass
    // with `hdr`) rolls highlights off there. Needs a window that can show it: set it, or
    // hdr_window, before set_animation_loop; then it can change at any time.
    bool hdr_output;
    // Open the window with half-float pixels, so that hdr_output can be turned on later.
    bool hdr_window;
    // Deferred G-Buffer + fullscreen lighting pass for single-sampled render targets.
    bool deferred;
    // Rect area lights with shadow.ray_traced get ray-traced shadows where the GPU can
    // (ray_shadows.c3). Default true; false: their shadow maps everywhere.
    bool ray_traced_shadows;
    // Fragments inside cull_box_min..max are not drawn (to look through a portal past whatever
    // stands between the camera and it).
    bool cull_box;
    Vector3 cull_box_min;
    Vector3 cull_box_max;
    // LOD distances are multiplied by this (above 1 keeps detail and draws further).
    float lod_scale;
    // The lens's focus, for Material.defocus: a 1x1 map whose r is the distance in focus (m;
    // BokehPass.focus_map()), and the blur of something at distance z, in pixels of the
    // render: min(focus_aperture / f * |1 - f / z|, focus_maxblur) (BokehPass's aperture and
    // maxblur). Unset, nothing is defocused.
    Texture* focus_map;
    float focus_aperture;
    float focus_maxblur;
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
    // Extension: how many RectAreaLights a render takes at most (16). Every one of them lights
    // every pixel of a standard material in view (three's have no reach to cut them off, so
    // they are not sorted into the clusters).
    int max_rect_area_lights;
    // Extension: the side of the shadow atlas at most, in pixels (8192; at most the GPU's
    // largest texture). Every shadowed light besides the primary spot light and the sun has
    // its maps there, each map as large as the light looks on screen (up to its
    // shadow.map_size, at least 128). The texture is only as large as the maps in it need. When
    // they don't fit, the least important lights' maps get smaller, then those lights lose their
    // shadows (see AtlasSlot).
    int shadow_atlas_size;
    // Extension: sorts the lights into clusters (see clusters.c3). null: in a compute shader
    // (cluster_lights_gpu), or on the CPU without them.
    ClusterAssignFn assign_clusters;
    // Extension: culls the instances of instanced meshes one by one on the GPU, for the
    // camera and the shadow maps drawn every frame (see culling.c3). On by default.
    bool gpu_culling;
    // Extension: a node material's mesh is left out of the picture, shadows included, while
    // its programs compile on worker threads (Metal), so that a new one doesn't hold up the
    // frames. Off by default: a frame waits for the programs its draws take, which start
    // compiling all at once when the render begins.
    bool compile_in_background;
    // Extension: how many programs (sokol shaders) can exist at once: the materials' generated
    // programs, one per material key and graphs drawn (shared by the materials that print the
    // same), the passes' and the application's. Read when the renderer starts; 0: DEFAULT_PROGRAMS.
    int max_programs;
    // Extension: a render target that stands for the window in what the pipeline cache records
    // (save_pipeline_cache): for a program that renders headless on behalf of one with a window,
    // as driejs makes its demos' thumbnails. Null: none.
    RenderTarget* screen_stand_in;
    // Extension: what has holes in it (alpha_test cut-outs, LOD cross-fades) is drawn twice:
    // first only its depth, with a cheap shader that cuts the holes, then lit where that depth
    // is, by a shader that cuts nothing. The lit shader then runs once per pixel instead of
    // for every layer of leaves and blades, and no shader of the main pass discards (one that
    // can costs the GPU its early depth test). On by default.
    bool depth_prepass;
    // Draws the shadow maps in render() (like three's shadowMap.autoUpdate). Off, a render uses
    // the maps as the last one left them: for a second view of the same moment (a Reflector's),
    // which would otherwise draw them all again from its own viewpoint. On by default.
    bool shadow_map_auto_update;
    // How the shadow maps are filtered, like three's shadowMap.type (PCFSoftShadowMap or
    // VSMShadowMap). Default PCF.
    ShadowType shadow_type;
    // ...for what covers at least this many pixels per triangle (about: its bounding sphere's
    // disk on the screen, over its triangles). Default 1.5.
    float depth_prepass_pixels;
    // Extension: also does on the CPU what compute shaders do (light clusters, instance
    // culling) and reports where the results differ, on stderr.
    bool check_compute;
    // Extension: tint what each cascade of the directional light's shadow covers (red, green,
    // blue, yellow, nearest first), to see where they fall.
    bool debug_cascades;
    // Extensions, for graphics settings (quality against frame time; 1 and the defaults are as
    // drie draws without them):
    // every light's shadow map is drawn at this many times its shadow.map_size (texels across);
    float shadow_map_scale;
    // the soft shadows take this many times their taps (smoother edges, less noise to average
    // out over frames), up to 40 per lookup;
    float shadow_sample_scale;
    // a screen-space reflection's ray takes at most ssr_steps steps (up to 128), none longer
    // than ssr_max_step pixels on the screen (shorter misses less of what is thin, reaches less
    // far in as many steps). Default 40 and 6;
    int ssr_steps;
    float ssr_max_step;
    // the volumetric fog's grid has this many times its cells across, up and in depth
    // (Fog.resolution, Fog.slices).
    float fog_resolution_scale;
    // The picture the transmission materials refract is this size times the target's (three's
    // transmissionResolutionScale; default 1). Not when it is the main pass's own picture,
    // copied half way through (into a linear target, see copy_for_transmission).
    float transmission_resolution_scale;
    // What materials with `ssr` reflect (see set_reflection_source).
    SgView ssr_color;
    SgView ssr_depth;
    SgView ssr_back;
    // the Reflector being rendered (render_reflector), whose own surfaces are left out
    Reflector* reflecting;
    Mat4 ssr_view_proj;
    float ssr_reach;
    bool ssr_linear;
    int ssr_frame;
}

// Screen-space reflections read the frame before: `color` is that picture as the render
// target held it (alpha 0 at light sources): tone mapped, or with `linear` in linear light, as
// an RGBA16F target holds it (better: what bright things reflect isn't cut off at white).
// `depth` has each pixel's distance from
// the camera's plane in its green channel, and `view_proj` is that frame's
// projection_matrix * matrix_world_inverse. Both the same size, row for row as rendered.
// `reach` is how far a reflected ray is followed (m). Call it every frame before render();
// clear_reflection_source() switches reflections off (for a render from another viewpoint).
// `back` (optional) is render_back_faces' picture of the same frame: how thick things are, so a
// reflected ray that passes behind something thin goes on to what is behind it.
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

fn void Renderer.clear_reflection_source(&self) {
    self.ssr_color = { };
    self.ssr_depth = { };
    self.ssr_back = { };
}

// After render(): draws the back faces of the opaque, one-sided, solid things it drew into
// `target` (an R16F target with a depth buffer, one sample, the size of the render): r = the
// distance of the nearest back face from the camera's plane (m), 0 where there is none. Give it
// to set_reflection_source next frame, so reflections know how thick things are.
//   RenderTarget* back = drie::render_target(w, h, 1, R16F);
//   renderer.render(scene, camera); renderer.render_back_faces(back);
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
        // (what has holes or two sides has no inside to speak of)
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

struct RenderInfo {
    int calls;
    int triangles;
    // line segments (Line, LineSegments, LineLoop)
    int lines;
    int shadow_calls;
    int shadow_triangles;
    // the shadow atlas: its size in pixels and the GPU memory of it and what goes with it
    // (VSM's moments, the cache of static casters), in bytes
    int atlas_width;
    int atlas_height;
    usz atlas_bytes;
    // maps of the atlas drawn this render (a spot light's, a face of a point light's) and the
    // draws of casters into them
    int atlas_tiles;
    int atlas_casters;
    // shadowed lights with maps in the atlas, and those that wanted one and got none
    int atlas_lights;
    int atlas_dropped;
    // draws left out because their program was still compiling in the background (0: the
    // picture is whole; a check that saves a picture waits for this)
    int waiting;
    // one of them (the last), and why: 1 a node program's pipeline or program compiling,
    // 2 a generated program compiling, 3 its pipeline not made yet
    Mesh* waiting_mesh;
    int waiting_why;
    // dynamic geometries whose vertices this render copied to the GPU (needs_update), and the
    // bytes
    int updates;
    usz update_bytes;
}

alias WebGLRenderer = Renderer;

// three's tone mappings (the order is TONE_* in tonemapping.glsl). The renderer applies it when
// it draws into the window or an RGBA8 target, OutputPass after post-processing in linear light.
enum ToneMapping {
    NO_TONE_MAPPING,
    LINEAR_TONE_MAPPING,
    REINHARD_TONE_MAPPING,
    CINEON_TONE_MAPPING,
    ACES_FILMIC_TONE_MAPPING,
    AGX_TONE_MAPPING,
    NEUTRAL_TONE_MAPPING,
}

struct DrawItem @private {
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

// A shadow map's moments for VSM (vsm_blur), and the image between the blur's two passes.
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

// A light map being baked (Renderer.render_bake): the objects are drawn into it at their uv1,
// with gen_bake_pipeline's pipelines, and read the last map from `prev`.
struct BakeState @local {
    bool on;
    // the map's format
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
    // the render's camera, for node materials (nodes.c3)
    Mat4 node_view;
    Mat4 node_proj;
    Vector3 node_camera;
    List{FrameHook} frame_hooks;
    List{EventFilter} event_filters;
    List{FrameHook} overlays;
    // RadianceCascades with no `camera` set that recorded an update_sparse() call this engine
    // frame, to flush (their last call's G-Buffer still wins) once commit() ends the frame.
    List{RadianceCascades*} pending_sparse_flush;
    bool gfx_ready;
    // Set when a shadow draw used the plain material because a program was still compiling.
    // A static light then draws its tile again in the next frame.
    bool shadow_plain;
    // the engine's depth and shadow programs (engine_program)
    SgShader[2] engine_programs;
    // Deferred G-Buffer targets and fullscreen lighting pipelines.
    int gb_width;
    int gb_height;
    SgImage[4] gb_color;
    SgView[4] gb_att;
    SgView[4] gb_tex;
    SgShader[4] deferred_shaders;
    SgPipeline[8] deferred_pipelines;
    SgSampler gb_sampler;
    // the blur that makes the moments of Renderer.shadow_type = VSM's maps (vsm.glsl)
    SgShader vsm_blur_shader;
    SgPipeline vsm_blur_pipeline;
    SgSampler vsm_sampler;
    SgSampler vsm_point_sampler;
    SgPixelFormat vsm_format;
    // the moments of the spot light's map, of the sun's and of the atlas (made when first
    // used, the size of their depth maps)
    VsmMaps vsm_spot;
    VsmMaps vsm_sun;
    VsmMaps vsm_atlas;
    // drawing with motion vectors (a target with MRT): the camera's view_proj a frame ago, and
    // which frame of them this is
    bool motion;
    Mat4 motion_view_proj;
    // the window has half-float pixels (Renderer.hdr_window), and the color space set for it
    // last (DisplayColorSpace's ordinal, -1: none yet)
    bool float_window;
    int window_colors;
    // the last render's view_proj (as drawn, jitter and all), for render_back_faces: of
    // world - view_origin (the camera's position), as the vertex shader takes it (see
    // relative_view_proj)
    Mat4 last_view_proj_rel;
    Vector3 view_origin;
    // the camera's matrix_world, for billboards (Mesh.billboard)
    Mat4 view_world;
    SgPipeline back_pipeline;
    // the camera's jitter in clip space (per unit of w), left out of the motion vectors
    Vector2 motion_jitter;
    int motion_frame;
    float motion_time;
    float motion_prev_time;
    // clear_rect's pipelines, by their target
    HashMap{ulong, SgPipeline} pipelines;
    // writes the far depth wherever it draws: clears one tile of the shadow atlas
    SgPipeline tile_clear_pipeline;
    // white instance colors (see whites)
    SgBuffer whites;
    uint whites_count;
    // Renderer.clear's and scissored render()'s (see clear_rect), and their program
    SgShader clear_shader;
    // frames committed (commit()), and the frame the window was last drawn into (first_pass)
    uint frame;
    uint window_drawn_frame;
    // inside on_frame's animation loop: a render() to the window leaves the commit to the end
    // of the frame (commit_pending), so that several can draw into it
    bool in_frame;
    bool commit_pending;
    BufferGeometry* tile_quad;
    List{DrawItem} opaque;
    List{DrawItem} transparent;
    List{DrawItem} casters;
    // what the shadow maps drawn every frame show: the spot light's, and each cascade's
    List{DrawItem} spot_casters;
    List{DrawItem}[4] sun_casters;
    List{Light*} lights;
    List{EventListener} listeners;
    ulong start_ticks;
    // Bound where a material has no texture.
    Texture* white;
    // The scene being rendered (for its environment).
    Scene* scene;
    // Bound where a scene has no irradiance volume.
    IrradianceVolume* no_irradiance;
    // the scene's first irradiance volume (no_irradiance without), all of them, and the
    // texture the shader reads them from: the first one's own, or with more than one the
    // gi_atlas, all of them side by side (gi_atlas_of)
    IrradianceVolume* irradiance;
    IrradianceVolume*[MAX_IRRADIANCE_VOLUMES] gi_volumes;
    int gi_count;
    SgView gi_view;
    SgImage gi_atlas;
    SgView gi_atlas_view;
    IrradianceVolume*[MAX_IRRADIANCE_VOLUMES] gi_atlas_of;
    // (bake_irradiance: the volume it bakes, left out of the scene's lights, or added to them)
    IrradianceVolume* gi_skip;
    IrradianceVolume* gi_add;
    // the scene's radiance cascades' probes, or a buffer of one dark probe
    StorageBuffer* no_radiance;
    SgView radiance_view;
    // the scene's reflection probes' array texture (none: id 0), and a 1x1 one to bind then
    SgView probes_view;
    SgSampler probes_sampler;
    SgImage no_probes_image;
    SgView no_probes;
    // One identity instance, bound for meshes that aren't instanced.
    SgBuffer identity_instance;
    // zeros for the second uv set of the geometries without one, for this many vertices
    SgBuffer zeros;
    uint zeros_vertices;
    SgImage shadow_image;
    SgView shadow_attachment;
    SgView shadow_view;
    SgSampler shadow_sampler;
    SgSampler clamp_sampler;
    int shadow_size;
    // The shadow maps of the lights besides the primary spot light and the sun (spot, point,
    // rect area and directional ones), each in its own part of one texture (see shadow_atlas.c3)
    SgImage atlas_image;
    SgView atlas_attachment;
    SgView atlas_view;
    int atlas_width;
    int atlas_height;
    AtlasPacker atlas_packer;
    // the program that copies depth from one texture into another (copy_depth)
    SgPipeline atlas_copy_pipeline;
    SgSampler atlas_copy_sampler;
    // the lights with a place in it, and their maps
    List{AtlasSlot} atlas_slots;
    // the maps' caches of the casters that keep still, for maps with casters that move (see
    // AtlasMap), in blocks of their own
    SgImage cache_image;
    SgView cache_attachment;
    SgView cache_view;
    int cache_width;
    int cache_height;
    AtlasPacker cache_packer;
    // the shadow casters of the last scenes rendered (walk_casters)
    CasterWalk[4] caster_walks;
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
    // What the transmission materials refract (Material.transmission): the opaque scene from
    // the same camera, drawn before the main pass (see render_transmission).
    RenderTarget* transmission;
    // ...and its mip levels, blurrier and blurrier, for rough surfaces (three's transmission
    // target has them): level 0 is a copy of the resolved picture, each next one half the size
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
    // scissor_test, or clear()): all passes render into window_rt with full MSAA LOAD/STORE,
    // then present_window_rt() blits it to the swapchain once at the end of on_frame().
    RenderTarget* window_rt;
    SgPipeline blit_pipeline;
    bool window_rt_used;
    // the scene's background_texture, drawn as a sphere around the camera
    Mesh* background;
    // the scene's background_texture when it is flat: a quad that fills the view
    Mesh* background_quad;
    // the sphere the scene's background_material is drawn on
    Mesh* background_sphere;
    // Renderer.render_environment's: the grid it draws a material on, its scene and camera, the
    // small drawing read back for the diffuse light, and the maps it drew into
    Scene* env_scene;
    Mesh* env_grid;
    OrthographicCamera* env_camera;
    RenderTarget* env_small;
    List{EnvTarget} env_targets;
    // Renderer.render_probe's: the six views, the camera they are drawn with and the graph
    // that reads them into a map
    RenderTarget*[6] probe_faces;
    PerspectiveCamera* probe_camera;
    MeshBasicMaterial* probe_resample;
    // baking a light map (render_bake): what it draws with, and the objects that cast the
    // shadows (not in the bake's scene of lights)
    BakeState bake;
    Object3D*[] caster_roots;
    SgShader seam_shader;
    SgPipeline[2] seam_pipelines;
}

RendererState state @private;

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

fn void Renderer.set_size(&self, int width, int height) {
    self.width = width;
    self.height = height;
}

// Render into `target` instead of the window (null renders to the window again).
fn void Renderer.set_render_target(&self, RenderTarget* target) {
    self.render_target = target;
}

fn void Renderer.set_clear_color(&self, Color color, float alpha = 1) {
    self.clear_color = color;
    self.clear_alpha = alpha;
}

// Opens the window and calls `cb` every frame with the elapsed time in milliseconds.
// Like three's renderer.setAnimationLoop. This call blocks until the window is closed.
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
            .high_dpi = true,
            .hdr = self.hdr_window || self.hdr_output,
            .fullscreen = self.fullscreen,
            .window_title = (ZString)self.title.zstr_tcopy(),
            .icon.sokol_default = true,
            .logger.func = &slog::func,
        }
    );
}

// Register a listener for raw input / window events (keyboard, mouse, resize).
fn void add_event_listener(EventListener listener) {
    state.listeners.push(listener);
}

fn float window_width() => sapp::isvalid() ? sapp::widthf() / sapp::dpi_scale() : (float)state.renderer.width;
fn float window_height() => sapp::isvalid() ? sapp::heightf() / sapp::dpi_scale() : (float)state.renderer.height;
// Equivalent of `window.innerWidth / window.innerHeight`.
fn float window_aspect() => window_width() / window_height();

fn bool depth_zero_to_one() {
    if (!state.gfx_ready) return true;
    SgBackend b = sg::query_backend();
    return b != GLCORE && b != GLES3;
}

alias ShaderDescFn = fn SgShaderDesc(SgBackend backend);

// The programs sokol has room for (its shader pool) with Renderer.max_programs unset.
const int DEFAULT_PROGRAMS = 1024;

fn void setup_gfx(SgEnvironment environment) @private {
    sg::setup(
        &&{
            .environment = environment,
            .logger.func = &slog::func, // a buffer per geometry (two, indexed), and per instanced mesh whose instances change
            // often: a farm of scattered plants has tens of thousands. sokol's maximum.
            .buffer_pool_size = 65535, // room for the uniforms of some thousands of draws a frame (sokol's default, 4 MB,
            // is too little for a scene of many scattered plants seen from afar)
            .uniform_buffer_size = 16 * 1024 * 1024, // an image, a view and a sampler per texture: a world of many
            // materials has hundreds (house's test views about 840 images and 790 samplers at once)
            .image_pool_size = 4096,
            .view_pool_size = 4096,
            .sampler_pool_size = 4096, // a pipeline per material program that is drawn and its state (get_pipeline: kind,
            // side, blending, wireframe, target, alpha to coverage, VSM; up to 2016), and the
            // passes'. sokol's default (64) runs out once a scene is drawn into a few kinds of target
            .pipeline_pool_size = 4096,
            .shader_pool_size = state.renderer && state.renderer.max_programs > 0 ? state.renderer.max_programs : DEFAULT_PROGRAMS,
        }
    );
    stm::setup();
    state.start_ticks = stm::now();
    query_gpu_caps();
    // (the materials' programs are made when they are first drawn with: material_program)
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

// Sets up the GPU without opening a window (Metal on macOS, D3D11 on Windows), for rendering into
// RenderTargets from scripts, tests or on a machine without a display. Use instead of
// set_animation_loop: call render() with a render target set, then drie::commit().
fn void Renderer.init_headless(&self) {
    state.renderer = self;
    setup_gfx(readback::headless_environment());
}

// What runs around the frame besides the application: before it (reading a game controller),
// on the window's events before the listeners (menus that take the input), and over the
// picture in the window (menus, fades). drie::input and drie::ui hook in this way, and an
// application can too.
alias FrameHook = fn void();
alias EventFilter = fn bool(Event* e);

// Runs `hook` before every frame, ahead of the animation loop.
fn void add_frame_hook(FrameHook hook) => state.frame_hooks.push(hook);

// Gives every window event to `filter` first: when it returns true, the event is used up.
// `first`: ahead of the filters added before it (the menus', say), not after them.
fn void add_event_filter(EventFilter filter, bool first = false) {
    if (first) {
        state.event_filters.insert_at(0, filter);
    } else {
        state.event_filters.push(filter);
    }
}

// Draws `overlay` over the picture whenever a frame goes into the window (render() to the
// window, postprocessing's end_screen(), or draw_overlays() in a pass of the application's).
fn void add_overlay(FrameHook overlay) => state.overlays.push(overlay);

// Draws the overlays into the current pass, which must be a pass to the window.
fn void draw_overlays() {
    foreach (o : state.overlays) o();
}

// What Lod distances are divided by (Renderer.lod_scale of the renderer in use).
fn float lod_distance_scale() @private => state.renderer ? math::max(state.renderer.lod_scale, 0.01f) : 1;

// HDR output is on, and the window can show it.
fn bool Renderer.hdr_on(&self) => self.hdr_output && state.float_window;

// The color space the window's pixels are taken in, as the renderer's settings ask.
fn void apply_window_colors() @local {
    Renderer* r            = state.renderer;
    DisplayColorSpace want = r.hdr_on() ? EXTENDED_LINEAR_DISPLAY_P3
        : (r.output_color_space == DISPLAY_P3_COLOR_SPACE ? DISPLAY_P3 : SRGB);
    if (state.window_colors == want.ordinal) return;
    // (an 8-bit window that never changed stays as the platform made it)
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

// Passes an event to the menus and listeners as if it came from the window (the gamepad's
// key events, say).
fn void send_event(SappEvent* e) => on_event(e);

fn void on_event(SappEvent* e) @local {
    if (e.type == KEY_DOWN && !e.key_repeat
        && (e.key_code == F11 || (e.key_code == ENTER && (e.modifiers & sapp::MODIFIER_ALT)))) {
        sapp::toggle_fullscreen();
        return;
    }
    // an open menu takes the input (drie::ui), for one
    foreach (f : state.event_filters) if (f(e)) return;
    foreach (l : state.listeners) l(e);
}

// What the target is cleared to for `color`: linear light in a linear target; otherwise as it
// is (display_fog, three's) or as it comes out of the lit shader, tone mapped, so that it
// matches fog of the same color.
fn Vector3 clear_value(Renderer* r, Color color) @private {
    if (r.render_target && r.render_target.is_linear()) return color_to_linear(color);
    if (r.display_fog) return { ((color >> 16) & 0xff) / 255.0f, ((color >> 8) & 0xff) / 255.0f, (color & 0xff) / 255.0f };
    return tone_mapped(r, color);
}

// Whether this is the frame's first pass into `rt` (or the window), marking it drawn: the
// first clears the whole target, the others can keep what is there.
fn bool first_pass(RenderTarget* rt) @private {
    uint* drawn = rt ? &rt.drawn_frame : &state.window_drawn_frame;
    bool first = *drawn != state.frame + 1;
    *drawn = state.frame + 1;
    return first;
}

// A pass that keeps what the frame drew into the target before it (all of it cleared when it
// is the frame's first), and stores what it draws for the next.
fn void keep_pass(SgPass* pass, bool first, RenderTarget* rt) @local {
    SgLoadAction load = first ? CLEAR : LOAD;
    pass.action.colors[0].load_action = load;
    pass.action.colors[0].store_action = STORE;
    pass.action.depth = { .load_action = load, .store_action = STORE, .clear_value = 1 };
    if (rt && rt.mrt) {
        for (int i = 1; i <= MRT_COUNT; i++) pass.action.colors[i] = { .load_action = load, .store_action = STORE };
    }
}

// Renderer.viewport and the scissor box, for the draws of the pass.
fn void apply_view_rect(Renderer* r) @local {
    int[4] v = r.viewport;
    if (v[2] > 0 && v[3] > 0) sg::apply_viewport(v[0], v[1], v[2], v[3], false);
    int[4] s = r.scissor;
    if (r.scissor_test) sg::apply_scissor_rect(s[0], s[1], s[2], s[3], false);
}

// Clears the scissor box of the current pass to `rgba` and the far depth (WebGL's clear with
// the scissor test on), whatever the viewport.
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

// A pass into the window has ended: the frame is committed now, or at the end of the
// animation loop's frame.
fn void window_drawn() @local {
    if (state.in_frame) {
        state.commit_pending = true;
    } else {
        commit();
    }
}

// Ensures the internal window-sized offscreen target exists and matches the window's size.
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

// Blits `state.window_rt` to the swapchain in a single pass at the end of on_frame(), then
// draws overlays on top. Avoids multiple swapchain passes per frame (which Metal discards
// MSAA on and presents repeatedly).
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

// Clears the target (the window, or render_target) to clear_color and the far depth now, as
// three's renderer.clear(): only the scissor box with scissor_test.
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

// Where render() draws, in pixels from the target's bottom left (three's setViewport);
// zero width or height: the whole target.
fn void Renderer.set_viewport(&self, int x, int y, int width, int height) => self.viewport = { x, y, width, height };

// The box render() and clear() touch while scissor_test is on (three's setScissor).
fn void Renderer.set_scissor(&self, int x, int y, int width, int height) => self.scissor = { x, y, width, height };

// The background color as it comes out of the lit shader, so that it matches fog of the same color.
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

// Linear light through a tone mapping, as the shaders do it (tonemapping.glsl): 0..1, still
// linear. For matching a color of your own to the picture.
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
            // (rows of the shader's column-major matrices)
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

// A draw left out for its program still compiling (RenderInfo.waiting).
fn void note_waiting(Mesh* m, int why) @private {
    if (!state.renderer) return;
    state.renderer.info.waiting++;
    state.renderer.info.waiting_mesh = m;
    state.renderer.info.waiting_why = why;
}

// Prints what the draw RenderInfo.waiting_mesh names waits for: its material's program's jobs
// compiling in the background and their states (sg_warm_state: 0 running, 1 done, 2 failed).
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

// Draws the scene from the camera's point of view. Call once per frame from the animation loop.
fn void Renderer.render(&self, Scene* scene, Camera* camera) {
    if (!state.gfx_ready) return;
    if (self.auto_time) self.time = (float)stm::sec(stm::since(state.start_ticks));
    self.info = { };
    // (outside any pass: no render is drawn inside another's pass)
    self.info.updates = update_dynamic(&self.info.update_bytes);

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
    // cameras don't have to be part of the scene
    if (!camera.parent) camera.update_matrix_world();
    camera.matrix_world_inverse = camera.matrix_world.invert();
    Mat4 view_proj        = camera.projection_matrix.mul(camera.matrix_world_inverse);
    Mat4 steady_view_proj = view_proj;
    Vector3 camera_pos    = camera.matrix_world.get_position();
    // (a render inside this one, a Reflector's, leaves this one's camera for its node materials)
    Mat4 outer_view = state.node_view;
    Mat4 outer_proj = state.node_proj;
    Vector3 outer_camera = state.node_camera;
    defer {
        state.node_view = outer_view;
        state.node_proj = outer_proj;
        state.node_camera = outer_camera;
    }
    node_epoch++;
    // (a loaded pipeline cache: more of it, as what it compiles finishes)
    if (state.renderer) state.renderer.pump_pipeline_cache();
    state.node_view = camera.matrix_world_inverse;
    state.node_proj = camera.projection_matrix;
    state.node_camera = camera_pos;
    Mat4 view_proj_rel    = relative_view_proj(camera, camera_pos);
    state.motion_jitter = { };
    if (camera.jitter != { }) {
        // shifted across the picture after the projection: x' = x + s * w
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
    // motion vectors: every render with MRT is the next frame of them
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
    // front-to-back for opaque, back-to-front for transparent (view space looks down -Z)
    sort::quicksort(&state.opaque, &cmp_opaque);
    sort::quicksort(&state.transparent, &cmp_transparent);
    int view = cull_view(&frustum);
    foreach (&item : state.opaque) item.job = cull_job(item.mesh, view) + 1;
    foreach (&item : state.transparent) item.job = cull_job(item.mesh, view) + 1;
    profile::stop("render: sort + jobs", pt);

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
                                 lights_gpu.cookie ? lights_gpu.cookie.view : ready(state.white).view, state.clamp_sampler };
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
    start_material_programs(self, scene, camera, &fs, self.depth_prepass && !self.cull_box);
    profile::stop("render: start programs", pt);
    pt = profile::start();
    // what the transmission materials refract: the opaque scene behind them, without them
    // (theirs, opaque or transparent)
    bool refraction;
    foreach (&item : state.opaque) refraction = refraction || refracts(item);
    foreach (&item : state.transparent) refraction = refraction || refracts(item);
    // Into a linear target (RGBA16F), the main pass's own picture of the opaque scene is what
    // the glass refracts: it is copied half way through the pass (copy_for_transmission), as
    // engines that draw the scene once do. Otherwise (a tone mapped picture can't be refracted
    // as light), the opaque scene is drawn again for it first, as three does.
    bool split = refraction && self.render_target && self.render_target.is_linear();
    if (refraction && !split) {
        render_transmission(self, scene, camera, view_proj_rel, &fs);
        // (the refracting materials' programs read the transmission's map, made by its first
        // pass: their keys change with it)
        start_material_programs(self, scene, camera, &fs, self.depth_prepass && !self.cull_box);
    }
    profile::stop("render: transmission", pt);
    pt = profile::start();

    // Multi-pass windowed frames (auto_clear = false, scissor_test, or after clear()) render
    // into an internal window-sized offscreen target so MSAA LOAD/STORE is preserved and the
    // swapchain is presented once in present_window_rt() at frame end.
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
        // kept for RenderTarget.depth_texture
        pass.action.depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 };
        pass.attachments = self.render_target.attachments();
        // no normal, no motion, no indirect light where nothing is drawn
        if (self.render_target.mrt) {
            for (int i = 1; i <= MRT_COUNT; i++) pass.action.colors[i] = { .load_action = CLEAR };
        }
    } else {
        pass.swapchain = sglue::swapchain();
    }
    // auto_clear, or a background color (which clears whatever auto_clear says, as three's
    // does), clears the target first: all of it, or the scissor box. Otherwise what the frame
    // has drawn there already is kept.
    bool clears = self.auto_clear || (scene.background && !scene.background_texture);
    bool whole = clears && !self.scissor_test;
    bool first = first_pass(self.render_target);
    if (!whole) keep_pass(&pass, first, self.render_target);
    if (split) {
        // (multisampled: kept for the second half of the pass)
        for (int i = 0; i <= (self.render_target.mrt ? MRT_COUNT : 0); i++) pass.action.colors[i].store_action = STORE;
    }
    // (not with a cull box: everything would have holes)
    bool prepass = self.depth_prepass && !self.cull_box;
    // pixels per unit of tan(angle) up the view, for prepassed()
    state.prepass_focal = 0;
    if (camera.camera_type == PERSPECTIVE) {
        RenderTarget* rt = self.render_target;
        state.prepass_focal = camera.projection_matrix[1][1] * 0.5f * (rt ? rt.height : sapp::heightf());
    }
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
        // (the glass after the copy, see below)
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
        // the far side of double-sided glass is behind its near side: its back faces first,
        // refracting the opaque scene, and the picture copied again with them in it
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
            // (its back faces are drawn already)
            if (split && backs && side == DOUBLE_SIDE && refracts(item)) mat.side = FRONT_SIDE;
            // Double-sided glass without the split: its back faces, then its front faces, as
            // three draws a transparent double-sided material (one draw without culling
            // would blend them in the order of the triangles, and where a back face comes
            // after the front one over it, it would cover the near side's own reflection)
            if (!split && side == DOUBLE_SIDE && refracts(item)) {
                mat.side = BACK_SIDE;
                draw_item(item, view_proj_rel, &fs);
                mat.side = FRONT_SIDE;
            }
            draw_item(item, view_proj_rel, &fs);
            mat.side = side;
        }
    }
    // menus and fades on top (drie::ui)
    if (!self.render_target) draw_overlays();
    sg::end_pass();
    profile::stop("render: main transparent", pt);
    // Fallback for sparse RC when a render uses MSAA or has no deferred opaque items:
    // uses the resolved depth_texture and normal_texture from MRT.
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
    // When rendering to the window, finish the frame (at the end of the animation loop's
    // frame, when inside it: another render() may draw into it). With a render target, the
    // caller usually runs more passes (post-processing) and must call drie::commit() at the end.
    if (!self.render_target) window_drawn();
}

// Queues `rc` (a sparse RadianceCascades with no `camera` set) for flush_pending_sparse() at
// the end of the engine frame, once; see RadianceCascades.update_sparse().
fn void queue_sparse_flush(RadianceCascades* rc) {
    foreach (q : state.pending_sparse_flush) if (q == rc) return;
    state.pending_sparse_flush.push(rc);
}

// Takes `rc` off the queue (RadianceCascades.dispose(): commit() mustn't flush it after).
fn void unqueue_sparse_flush(RadianceCascades* rc) {
    for (sz i = state.pending_sparse_flush.len(); i > 0; i--) {
        if (state.pending_sparse_flush[i - 1] == rc) state.pending_sparse_flush.remove_at(i - 1);
    }
}

fn void commit() {
    foreach (rc : state.pending_sparse_flush) rc.flush_pending_sparse();
    state.pending_sparse_flush.clear();
    state.commit_pending = false;
    state.frame++;
    profile::commit();
}