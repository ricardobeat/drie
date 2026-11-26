// Deferred G-Buffer allocation, item eligibility, and fullscreen lighting pass.
// Uses generated/shader_deferred.c3 (`drie_shaders::deferred`), which lives outside `drie::*`
// because C3 imports `drie::*` recursively and shaders/shader_deferred.glsl defines structs
// (`FsParams`, `GpuPointLight`, etc.) that share names with `drie::shaders`. Renderer.render()
// runs the 4-target RGBA16F G-Buffer pass, optional ray-traced shadows and sparse radiance
// updates, then draw_deferred_lighting() before forward-shading remaining opaque and transparent
// items. Reconstructs camera-relative world position from depth using `depth_zero_to_one()`.
module drie;
import std::math, std::collections::list;
import sokol, drie_shaders::deferred;

// Shared view and sampler slots: forward transmission binds the opaque scene copy here, while
// the deferred lighting pass binds the target's depth texture in the same view slot.
const int VIEW_TEX_TRANSMISSION @private = deferred::VIEW_TEX_TRANSMISSION;
const int SMP_TRANSMISSION_SMP @private = 5;

// Per-pass camera and screen uniforms for shaders/shader_deferred.glsl (uniform block 0).
struct DeferredLightParams @align(16) @packed @private {
    Mat4 inv_view_proj;
    Vec4 view_origin;   // xyz = camera world position, w = 1 on [0,1] depth backends else 0.
    Vec4 clear_color;
    Vec4 screen_info;   // xy = inv target size, z = 1 on [0,1] depth backends, w = frame % 64.
    Mat4 view_proj;
}


// A bilinear sampler, clamped to the edge (made at first use).
fn SgSampler clamp_sampler() @private {
    if (!state.clamp_sampler.id) {
        state.clamp_sampler = sg::make_sampler(&&{ .min_filter = LINEAR, .mag_filter = LINEAR, .wrap_u = CLAMP_TO_EDGE, .wrap_v = CLAMP_TO_EDGE });
    }
    return state.clamp_sampler;
}

fn SgShaderDesc split_deferred_shader_desc(bool mrt, bool vsm, SgBackend backend) @private {
    if (vsm) return mrt ? deferred::deferred_mrt_vsm_shader_desc(backend) : deferred::deferred_vsm_shader_desc(backend);
    return mrt ? deferred::deferred_mrt_shader_desc(backend) : deferred::deferred_shader_desc(backend);
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
    return self.deferred && rt && rt.samples == 1 && rt.depth_buffer && clears
        && (!self.scissor_test || first) && !state.bake.on && self.viewport[2] == 0 && self.viewport[3] == 0;
}

// True when `item` can be encoded into the G-Buffer without losing material features.
fn bool is_deferred_item(DrawItem* item) @private {
    Mesh* m = item.mesh;
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

// Shades the G-Buffer in a single fullscreen triangle pass into `r.render_target` (plus MRT
// attachments when enabled), binding clustered lights, shadows, GI, SSR, fog, and RT shadow mask.
fn void draw_deferred_lighting(
    Renderer* r,
    Scene* scene,
    Camera* camera,
    Mat4 view_proj_rel,
    shaders::FsParams* fs,
    Vector3 clear,
    float clear_alpha
) @private {
    RenderTarget* rt = r.render_target;
    bool mrt = rt.mrt;
    bool vsm = vsm_on();
    int sidx = (mrt ? 1 : 0) | (vsm ? 2 : 0);
    if (!state.deferred_shaders[sidx].id) {
        state.deferred_shaders[sidx] = make_program(split_deferred_shader_desc(mrt, vsm, sg::query_backend()));
    }
    int pidx = sidx | (rt.is_linear() ? 4 : 0);
    if (!state.deferred_pipelines[pidx].id) {
        SgPipelineDesc desc = {
            .shader = state.deferred_shaders[sidx],
            .depth = { .pixel_format = NONE, .write_enabled = false, .compare = ALWAYS },
            .sample_count = 1,
            .color_count = mrt ? 1 + MRT_COUNT : 1,
            .label = "drie-deferred-light",
        };
        desc.colors[0].pixel_format = rt.pixel_format();
        if (mrt) {
            for (int i = 0; i < MRT_COUNT; i++) desc.colors[i + 1].pixel_format = mrt_format(i);
        }
        state.deferred_pipelines[pidx] = sg::make_pipeline(&desc);
    }
    sg::apply_pipeline(state.deferred_pipelines[pidx]);

    Texture* none = ready(state.white);
    if (!state.no_probes.id) {
        char[4] white = { 255, 255, 255, 255 };
        state.no_probes_image = sg::make_image(
            &&{ .type = ARRAY, .width = 1, .height = 1, .num_slices = 1, .data = { .ptr = &white, .size = 4 } }
        );
        state.no_probes = sg::make_view(&&{ .texture = { .image = state.no_probes_image } });
    }

    SgBindings bind;
    bind.views[shaders::VIEW_TEX_MAP] = state.gb_tex[0];
    bind.views[shaders::VIEW_TEX_NORMAL] = state.gb_tex[1];
    bind.views[shaders::VIEW_TEX_ARM] = state.gb_tex[2];
    bind.views[shaders::VIEW_TEX_PARALLAX] = state.gb_tex[3];
    bind.views[VIEW_TEX_TRANSMISSION] = rt.depth_texture;
    bind.samplers[shaders::SMP_TEX_SMP] = state.gb_sampler;
    bind.samplers[SMP_TRANSMISSION_SMP] = state.gb_sampler;
    for (int k = 0; k < 6; k++) bind.views[shaders::VIEW_TEX_EXT0 + k] = none.view;
    // Ray-traced shadow visibility mask from rt_trace (rt_shadow in shaders/shader_deferred.glsl).
    bind.views[shaders::VIEW_TEX_EXT5] = rt_mask(none.view);
    if (vsm) {
        bind.views[shaders::VIEW_VSM_MAP] = state.vsm_spot.view;
        bind.views[shaders::VIEW_VSM_ATLAS] = state.vsm_atlas.view;
        bind.views[shaders::VIEW_VSM_SUN] = state.vsm_sun.view;
        bind.samplers[shaders::SMP_VSM_SMP] = state.vsm_sampler;
    } else {
        bind.views[shaders::VIEW_SHADOW_MAP] = state.shadow_view;
        bind.views[shaders::VIEW_SHADOW_ATLAS] = state.atlas_view;
        bind.views[shaders::VIEW_SUN_SHADOW] = state.sun_view;
    }
    bind.samplers[shaders::SMP_SHADOW_SMP] = state.shadow_sampler;
    bind.views[shaders::VIEW_FOG_VOLUME] = state.fog_view;
    bool ssr = r.ssr_color.id != 0 && r.ssr_depth.id != 0;
    bind.views[shaders::VIEW_SSR_COLOR] = ssr ? r.ssr_color : none.view;
    bind.views[shaders::VIEW_SSR_DEPTH] = ssr ? r.ssr_depth : none.view;
    bind.views[shaders::VIEW_SSR_BACK] = ssr && r.ssr_back.id ? r.ssr_back : none.view;
    // SpotLight.map projection texture (light_cookies in shaders/shader_core.glsl).
    bind.views[shaders::VIEW_LIGHT_COOKIES] = lights_gpu.cookie ? lights_gpu.cookie.view : none.view;
    bind.samplers[shaders::SMP_SSR_SMP] = clamp_sampler();
    bind.views[shaders::VIEW_TEX_GI] = state.gi_view;
    bind.views[shaders::VIEW_RC_PROBES] = state.radiance_view;
    bind.samplers[shaders::SMP_GI_SMP] = state.irradiance.sampler;

    shaders::FsParams df = *fs;
    Texture* env_map = scene.environment;
    Texture* env = ready(env_map);
    bool env_diffuse = false;
    if (env_map && env_map.has_env_sh) {
        env_diffuse = true;
        EnvLight* el = &env_map.env_light;
        float[6] sun = { el.sun_dir.x, el.sun_dir.y, el.sun_dir.z, el.sun.x, el.sun.y, el.sun.z };
        foreach (i, c : el.sh) df.env_sh[i] = { ...c, i < 6 ? sun[i] : 0 };
    }
    bool probes = state.probes_view.id != 0;
    bind.views[shaders::VIEW_TEX_ENV] = env.view;
    bind.views[shaders::VIEW_TEX_PROBES] = probes ? state.probes_view : state.no_probes;
    bind.samplers[shaders::SMP_ENV_SMP] = probes ? state.probes_sampler : env.sampler;
    lights_gpu.bind(&bind);
    sg::apply_bindings(&bind);

    bool zo = depth_zero_to_one();
    DeferredLightParams dp = {
        .inv_view_proj = view_proj_rel.invert(),
        .view_origin = { ...camera.matrix_world.get_position(), zo ? 1 : 0 },
        .clear_color = { clear.x, clear.y, clear.z, clear_alpha },
        .screen_info = { 1.0f / rt.width, 1.0f / rt.height, zo ? 1 : 0, (float)(state.frame % 64) },
        .view_proj = view_proj_rel,
    };
    sg::apply_uniforms(0, &&{ .ptr = &dp, .size = DeferredLightParams::size });

    df.screen_params = { 1.0f / rt.width, 1.0f / rt.height, 0, 0 };
    df.surface_params = { 0, 0, 0, 0 };
    df.env_params = probes
        ? (Vec4) { 2, 1, 0, 0 }
        : (Vec4) { env_map ? 1 : 0, env_map ? scene.environment_intensity : 1, env_diffuse ? 1 : 0, env_map ? env_map.env_distance_range : 0 };
    df.env_combine = { };
    df.reflector_params = { };
    df.ssr_view_proj = r.ssr_view_proj;
    df.ssr_params = { ssr ? 1.0f : 0.0f, r.ssr_reach, (float)(r.ssr_frame % 64), zo ? 1 : 0 };
    df.ssr_thickness = { ssr && r.ssr_back.id ? 1 : 0, r.ssr_reach, 0, 0 };
    df.quality_params = { r.shadow_sample_scale, r.ssr_steps, r.ssr_max_step, 0 };
    sg::apply_uniforms(1, &&{ .ptr = &df, .size = shaders::FsParams::size });

    sg::draw(0, 3, 1);
    r.info.calls++;
    r.info.triangles += 1;
}
