// Deferred G-Buffer allocation, item eligibility, and fullscreen lighting pass.
//
// Renderer.render() runs the 4-target RGBA16F G-Buffer pass, optional ray-traced shadows and
// sparse radiance updates, then draw_deferred_lighting() before forward-shading remaining opaque
// and transparent items. The fullscreen lighting shader is built through `material_program()`
// (src/material/material_deferred.c3) so forward and deferred shading share one node-graph
// implementation. Reconstructs camera-relative world position from depth using `depth_zero_to_one()`.
module drie;
import std::math, std::collections::list;
import sokol;

// Per-pass camera and screen uniforms for the deferred fullscreen lighting pass (NodeSource.DEFERRED).
struct DeferredLightParams @align(16) @packed @private {
    Mat4 inv_view_proj;
    Vec4 view_origin; // xyz = camera world position, w = 1 on [0,1] depth backends else 0.
    Vec4 clear_color;
    Vec4 screen_info; // xy = inv target size, z = 1 on [0,1] depth backends, w = frame % 64.
    Mat4 view_proj;
}

// A bilinear sampler, clamped to the edge (made at first use).
fn SgSampler clamp_sampler() @private {
    if (!state.clamp_sampler.id) {
        state.clamp_sampler = sg::make_sampler(
            &&{ .min_filter = LINEAR, .mag_filter = LINEAR, .wrap_u = CLAMP_TO_EDGE, .wrap_v = CLAMP_TO_EDGE }
        );
    }
    return state.clamp_sampler;
}

// Allocates or resizes the 4 single-sampled RGBA16F G-Buffer color attachments and nearest sampler.
fn void ensure_gbuffer(int w, int h) @private {
    if (state.gb_width == w && state.gb_height == h) return;
    if (state.gb_width) {
        for (int i = 0; i < 4; i++) {
            sg::destroy_view(state.gb_att[i]);
            sg::destroy_view(state.gb_tex[i]);
            sg::destroy_image(state.gb_color[i]);
        }
    }
    if (!state.gb_sampler.id) {
        state.gb_sampler = sg::make_sampler(
            &&{
                .min_filter = NEAREST,
                .mag_filter = NEAREST,
                .wrap_u = CLAMP_TO_EDGE,
                .wrap_v = CLAMP_TO_EDGE,
            }
        );
    }
    state.gb_width = w;
    state.gb_height = h;
    for (int i = 0; i < 4; i++) {
        state.gb_color[i] = sg::make_image(
            &&{
                .usage = { .color_attachment = true },
                .width = w,
                .height = h,
                .sample_count = 1,
                .pixel_format = RGBA16F,
                .label = "drie-gbuffer-color",
            }
        );
        state.gb_att[i] = sg::make_view(&&{ .color_attachment = { .image = state.gb_color[i] } });
        state.gb_tex[i] = sg::make_view(&&{ .texture = { .image = state.gb_color[i] } });
    }
}

// True when this render can use the deferred path: single-sampled offscreen target with depth,
// full-target clear, no active lightmap bake, and no sub-viewport.
fn bool can_defer(Renderer* self, bool clears, bool first) @private {
    RenderTarget* rt = self.render_target;
    return self.deferred && rt && rt.samples == 1 && rt.depth_buffer && clears && (!self.scissor_test || first)
        && !state.bake.on
        && self.viewport[2] == 0
        && self.viewport[3] == 0;
}

// True when `item` can be encoded into the G-Buffer without losing material features.
fn bool is_deferred_item(DrawItem* item) @private {
    Mesh* m       = item.mesh;
    Material* mat = m.material;
    if (mat.transparent || draws_lines(item) || refracts(item)) return false;
    if (m.geometry.morph_view.id != 0 && m.morph_weights_view.id != 0) return false;
    if (mat.type != STANDARD && mat.type != LAMBERT && mat.type != BASIC) return false;
    if (physical_layers(mat)) return false;
    if (mat.screen_target || mat.reflector || mat.env_map) return false;
    if (mat.env_intensity != 0 && mat.env_intensity != 1) return false;
    if (state.probes_view.id && !mat.env_probes && state.scene && state.scene.environment) return false;
    if (mat.light_channels != 0) return false;
    if (mat.ssr != 0 && mat.ssr != 1) return false;
    if (mat.ssr > 0 && mat.ssr_reach > 0 && state.renderer && mat.ssr_reach != state.renderer.ssr_reach) return false;
    if (mat.env_node.id || mat.output_node.id || mat.specular_node.id || mat.shininess_node.id) return false;
    if (!mat.depth_test || !mat.depth_write || !mat.tone_mapped || mat.premultiplied_alpha) return false;
    if (mat.parallax_map && mat.parallax_shadow > 0) return false;
    if (mat.type == STANDARD) {
        if (mat.ior != 0 && mat.ior != 1.5f) return false;
        if (mat.specular_intensity != 1.0f || mat.specular_color != (Vector3) { 1, 1, 1 }) return false;
        if (mat.specular_color_map || mat.specular_intensity_map) return false;
    }
    return true;
}

// Builds the `MaterialKey` for the deferred fullscreen lighting pass matching the active scene
// lights, shadows, GI, probes, SSR, fog, and output encoding.
fn MaterialKey deferred_key_of(Renderer* r, Scene* scene, FsParams* fs, RenderTarget* rt) @private {
    Texture* env_map = scene.environment;
    if (env_map) (void)ready(env_map);
    bool probes     = state.probes_view.id != 0;
    int fog         = (int)fs.fog_params.w;
    bool fog_volume = fog && fs.fog_grid.w > 0;
    bool shafts     = !fog_volume && fog == 3 && fs.fog_sun_dir.w > 0 && fs.fog_sun_color.w > 0
        && fs.dir_shadow_params.x > 0.5f;
    bool shadows    = lights_gpu.shadows.len() > 0 || fs.dir_shadow_params.x > 0.5f;
    bool linear     = fs.output_params.x > 0.5f;
    return {
        .type = (char)MaterialType.STANDARD,
        .mrt = rt.mrt,
        .deferred_light = true,
        .dir_lights = fs.counts.x > 0,
        .point_lights = fs.counts.y > 0,
        .spot_lights = fs.counts.z > 0,
        .rect_lights = fs.counts.w > 0,
        .sized = lights_gpu.sized,
        .tubes = lights_gpu.tubes,
        .shadows = shadows,
        .vsm = (shadows || shafts) && vsm_on(),
        .soft_shadows = shadows && lights_gpu.soft_shadows,
        .sun_shadow = shadows && fs.dir_shadow_params.x > 0.5f,
        .cloud_shadow = fs.cloud_shadow.y > 0,
        .dir_tiles = shadows && lights_gpu.dir_tiles,
        .point_shadows = shadows && lights_gpu.point_shadows,
        .spot_shadows = shadows && lights_gpu.spot_shadows,
        .debug_cascades = fs.dir_cascade_params.z > 0.5f && fs.dir_shadow_params.x > 0.5f,
        .contact_shadows = lights_gpu.contact_shadows,
        .rt_shadows = lights_gpu.rt_shadows,
        .env_light = !probes && env_map && env_map.has_env_sh,
        .env_reflect = probes || env_map != null,
        .probes = probes,
        .gi = state.gi_count > 0,
        .rc = fs.rc_size.w > 0,
        .ssr = r.ssr_color.id != 0 && r.ssr_depth.id != 0,
        .fog = (char)fog,
        .fog_box = fog && fs.fog_box_min.w >= 0.5f,
        .fog_height = fog && fs.fog_height.y > 0,
        .fog_sun = fog && fs.fog_sun_dir.w > 0,
        .display_fog = fog && !fog_volume && fs.fog_height.z > 0.5f,
        .fog_volume = fog_volume,
        .fog_glow = fog && !fog_volume && fs.fog_color.w > 0,
        .fog_shafts = shafts,
        .linear = linear,
        .tone = linear ? 0 : (char)fs.output_params.z,
        .display = linear ? 0 : (char)fs.output_params.w,
    };
}

// Looks up or creates the fullscreen deferred lighting pipeline on `gen` for `rt`.
fn SgPipeline deferred_pipeline(RenderTarget* rt, MaterialProgram* gen) @private {
    SgPixelFormat fmt = rt.pixel_format();
    bool mrt          = rt.mrt;
    ulong pkey        = (ulong)(uint)fmt | (mrt ? 1UL << 8 : 0UL);
    if (try found = gen.pipelines[pkey]) return found;
    record_pipeline(
        {
            .key = gen.key,
            .target_samples = 1,
            .target_format = fmt,
            .target_mrt = mrt,
        }
    );
    SgPipelineDesc desc = {
        .depth = { .pixel_format = NONE, .write_enabled = false, .compare = ALWAYS },
        .sample_count = 1,
        .color_count = mrt ? 1 + MRT_COUNT : 1,
        .label = "drie-deferred-light",
    };
    desc.colors[0].pixel_format = fmt;
    if (mrt) {
        for (int i = 0; i < MRT_COUNT; i++) desc.colors[i + 1].pixel_format = mrt_format(i);
    }
    if (pipeline_warmup) {
        pipeline_warmup.start_generated(gen.program, &desc);
        return { };
    }
    desc.shader = gen.program.make();
    SgPipeline pip = sg::make_pipeline(&desc);
    if (!pip.id) abort("drie: sokol's pipeline pool is full (pipeline_pool_size in setup_gfx)");
    gen.pipelines[pkey] = pip;
    return pip;
}

// Shades the G-Buffer in a single fullscreen triangle pass into `r.render_target` (plus MRT
// attachments when enabled), binding clustered lights, shadows, GI, SSR, fog, and RT shadow mask.
fn void draw_deferred_lighting(
    Renderer* r,
    Scene* scene,
    Camera* camera,
    Mat4 view_proj_rel,
    FsParams* fs,
    Vector3 clear,
    float clear_alpha
) @private {
    RenderTarget* rt     = r.render_target;
    bool vsm             = vsm_on();
    MaterialKey key      = deferred_key_of(r, scene, fs, rt);
    MaterialProgram* gen = material_program(key);
    sg::apply_pipeline(deferred_pipeline(rt, gen));

    Texture* none = ready(state.white);
    if (!state.no_probes.id) {
        char[4] white = { 255, 255, 255, 255 };
        state.no_probes_image = sg::make_image(
            &&{ .type = ARRAY, .width = 1, .height = 1, .num_slices = 1, .data = { .ptr = &white, .size = 4 } }
        );
        state.no_probes = sg::make_view(&&{ .texture = { .image = state.no_probes_image } });
    }

    SgBindings classic;
    classic.views[shaders::VIEW_TEX_MAP] = state.gb_tex[0];
    classic.views[shaders::VIEW_TEX_NORMAL] = state.gb_tex[1];
    classic.views[shaders::VIEW_TEX_ARM] = state.gb_tex[2];
    classic.views[shaders::VIEW_TEX_PARALLAX] = state.gb_tex[3];
    classic.views[shaders::VIEW_TEX_TRANSMISSION] = rt.depth_texture;
    classic.samplers[shaders::SMP_TEX_SMP] = state.gb_sampler;
    classic.samplers[shaders::SMP_TRANSMISSION_SMP] = state.gb_sampler;
    for (int k = 0; k < 6; k++) classic.views[shaders::VIEW_TEX_EXT0 + k] = none.view;
    // Ray-traced shadow visibility mask from rt_trace (rt_shadow in src/material/shadow_lookup.c3).
    classic.views[shaders::VIEW_TEX_EXT5] = rt_mask(none.view);
    if (vsm) {
        classic.views[shaders::VIEW_VSM_MAP] = state.vsm_spot.view;
        classic.views[shaders::VIEW_VSM_ATLAS] = state.vsm_atlas.view;
        classic.views[shaders::VIEW_VSM_SUN] = state.vsm_sun.view;
        classic.samplers[shaders::SMP_VSM_SMP] = state.vsm_sampler;
    } else {
        classic.views[shaders::VIEW_SHADOW_MAP] = state.shadow_view;
        classic.views[shaders::VIEW_SHADOW_ATLAS] = state.atlas_view;
        classic.views[shaders::VIEW_SUN_SHADOW] = state.sun_view;
    }
    classic.samplers[shaders::SMP_SHADOW_SMP] = state.shadow_sampler;
    classic.views[shaders::VIEW_FOG_VOLUME] = state.fog_view;
    classic.views[shaders::VIEW_CLOUD_SHADOW] = r.cloud_shadow.id ? r.cloud_shadow : none.view;
    bool ssr = r.ssr_color.id != 0 && r.ssr_depth.id != 0;
    classic.views[shaders::VIEW_SSR_COLOR] = ssr ? r.ssr_color : none.view;
    classic.views[shaders::VIEW_SSR_DEPTH] = ssr ? r.ssr_depth : none.view;
    classic.views[shaders::VIEW_SSR_BACK] = ssr && r.ssr_back.id ? r.ssr_back : none.view;
    // SpotLight.map projection texture (light_cookies).
    classic.views[shaders::VIEW_LIGHT_COOKIES] = lights_gpu.cookie ? lights_gpu.cookie.view : none.view;
    classic.samplers[shaders::SMP_SSR_SMP] = clamp_sampler();
    classic.views[shaders::VIEW_TEX_GI] = state.gi_view;
    classic.views[shaders::VIEW_RC_PROBES] = state.radiance_view;
    classic.samplers[shaders::SMP_GI_SMP] = state.irradiance.sampler;

    FsParams df      = *fs;
    Texture* env_map = scene.environment;
    Texture* env     = ready(env_map);
    bool env_diffuse = false;
    if (env_map && env_map.has_env_sh) {
        env_diffuse = true;
        EnvLight* el = &env_map.env_light;
        float[6] sun = { el.sun_dir.x, el.sun_dir.y, el.sun_dir.z, el.sun.x, el.sun.y, el.sun.z };
        foreach (i, c : el.sh) df.env_sh[i] = { ...c, i < 6 ? sun[i] : 0 };
    }
    bool probes = state.probes_view.id != 0;
    classic.views[shaders::VIEW_TEX_ENV] = env.view;
    classic.views[shaders::VIEW_TEX_PROBES] = probes ? state.probes_view : state.no_probes;
    classic.samplers[shaders::SMP_ENV_SMP] = probes ? state.probes_sampler : env.sampler;
    lights_gpu.bind(&classic);
    SgBindings bind = program_bindings(gen.program, &classic);
    sg::apply_bindings(&bind);

    bool zo = depth_zero_to_one();
    DeferredLightParams dp = {
        .inv_view_proj = view_proj_rel.invert(),
        .view_origin = { ...camera.matrix_world.get_position(), zo ? 1 : 0 },
        .clear_color = { clear.x, clear.y, clear.z, clear_alpha },
        .screen_info = { 1.0f / rt.width, 1.0f / rt.height, zo ? 1 : 0, (float)(state.frame % 64) },
        .view_proj = view_proj_rel,
    };

    df.screen_params = { 1.0f / rt.width, 1.0f / rt.height, 0, 0 };
    df.surface_params = { 0, 0, 0, 0 };
    df.env_params = probes ? (Vec4) { 2, 1, 0, 0 }
        : (Vec4) {
            env_map ? 1 : 0,
            env_map ? scene.environment_intensity : 1,
            env_diffuse ? 1 : 0,
            env_map ? env_map.env_distance_range : 0
        };
    df.env_combine = { };
    df.reflector_params = { };
    df.ssr_view_proj = r.ssr_view_proj;
    df.ssr_params = { ssr ? 1.0f : 0.0f, r.ssr_reach, (float)(r.ssr_frame % 64), zo ? 1 : 0 };
    df.ssr_thickness = { ssr && r.ssr_back.id ? 1 : 0, r.ssr_reach, 0, 0 };
    df.quality_params = { r.shadow_sample_scale, r.ssr_steps, r.ssr_max_step, 0 };

    apply_program_uniforms(gen.program, VERTEX, null, &df, dp: &dp);
    apply_program_uniforms(gen.program, FRAGMENT, null, &df, dp: &dp);

    sg::draw(0, 3, 1);
    r.info.calls++;
    r.info.triangles += 1;
}