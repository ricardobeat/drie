// Volumetric fog compute pass (`fog_cs` in `shaders/kernels.glsl`) for scenes with
// `VolumetricFog` enabled. Marches fog through a perspective froxel grid tiled into a 2D RGBA16F
// storage atlas before the main pass; each slice stores accumulated linear in-scattered light
// (RGB) and transmittance (A) to its far edge. `fog_volume_at` in `src/material/layers.c3`
// samples the atlas before tone mapping, covering opaque and transparent surfaces without a
// pre-pass depth buffer. Up to four volumes are cached by `(cells_x, cells_y, slices)` and
// reused across frames.

module drie;
import std::math;
import sokol;

// ---- Volume cache

// Tiled 2D storage atlas holding a froxel grid's depth slices (`per_row` slices per row).
struct FogVolume @local {
    int cells_x;
    int cells_y;
    int slices;
    int per_row;
    SgImage image;
    SgView storage;
    SgView texture;
}

struct FogPassState @local {
    SgShader shader;
    SgPipeline pipeline;
    // Ring cache keyed by `(cells_x, cells_y, slices)` so multi-target frames keep separate grids.
    FogVolume[4] volumes;
    int next;
    int frame;
}

FogPassState fog_pass @local;

// Returns true when volumetric fog should run through the froxel compute pass (perspective
// camera, compute shader support, and at least one scattering, shaft, or noise term active).
fn bool fog_volume_wanted(Fog* f, Camera* camera, bool no_scatter) @private {
    if (!f || !f.volumetric || camera.camera_type != PERSPECTIVE) return false;
    bool scatter = f.light_scatter > 0 && !no_scatter;
    bool shafts  = f.sun_shafts > 0 && f.sun_sharpness > 0;
    if (!scatter && !shafts && f.noise_amount <= 0) return false;
    return compute_supported();
}

fn FogVolume* fog_volume_for(int cells_x, int cells_y, int slices) @local {
    foreach (&v : fog_pass.volumes) {
        if (v.image.id && v.cells_x == cells_x && v.cells_y == cells_y && v.slices == slices) return v;
    }
    FogVolume* v = &fog_pass.volumes[fog_pass.next];
    fog_pass.next = (fog_pass.next + 1) % (int)fog_pass.volumes.len;
    if (v.image.id) {
        sg::destroy_view(v.storage);
        sg::destroy_view(v.texture);
        sg::destroy_image(v.image);
    }
    int per_row = (int)math::ceil(math::sqrt((float)slices));
    int rows    = (slices + per_row - 1) / per_row;
    *v = { .cells_x = cells_x, .cells_y = cells_y, .slices = slices, .per_row = per_row };
    v.image = sg::make_image(
        &&{
            .usage = { .storage_image = true },
            .width = cells_x * per_row,
            .height = cells_y * rows,
            .pixel_format = RGBA16F,
            .label = "drie-fog-volume",
        }
    );
    v.storage = sg::make_view(&&{ .storage_image = { .image = v.image } });
    v.texture = sg::make_view(&&{ .texture = { .image = v.image } });
    return v;
}

// ---- Compute dispatch

// Shadow maps and spotlight cookie textures bound as inputs to the volumetric fog compute pass.
struct FogShadowViews @private {
    SgView spot_map;
    SgView spot_atlas;
    SgView sun_map;
    SgSampler sampler;
    // Spotlight projection cookies (`SpotLight.map`) and their regular linear sampler.
    SgView cookies;
    SgSampler cookie_sampler;
}

// Marches `scene`'s volumetric fog for `camera` into a froxel atlas sized to `width` x `height`
// pixels scaled by `resolution_scale`. Reads light and shadow parameters from `p`, writes the
// resulting grid layout back into `p.fog_grid` and `p.fog_slices`, and returns the atlas view.
fn SgView update_fog_volume(
    Scene* scene,
    Camera* camera,
    FsParams* p,
    int width,
    int height,
    FogShadowViews views,
    float time,
    float resolution_scale = 1
) @private {
    Fog* f = scene.fog;
    if (!fog_pass.shader.id) {
        fog_pass.shader = make_program(shaders::fog_shader_desc(sg::query_backend()));
        fog_pass.pipeline = sg::make_pipeline(&&{ .compute = true, .shader = fog_pass.shader, .label = "drie-fog" });
    }
    // Scale both xy cell density and slice count (`Renderer.fog_resolution_scale`).
    float q     = resolution_scale > 0 ? resolution_scale : 1;
    float res   = math::min((f.resolution > 0 ? f.resolution : 0.25f) * q, 1.0f);
    int cx      = math::max((int)math::ceil((float)width * res), 1);
    int cy      = math::max((int)math::ceil((float)height * res), 1);
    int slices  = math::min(math::max((int)((f.slices > 0 ? f.slices : 96) * q + 0.5f), 2), 256);
    // Keep the tiled slice atlas within GPU limits and a bounded VRAM budget on HiDPI/4K targets.
    int per_row = (int)math::ceil(math::sqrt((float)slices));
    int rows    = (slices + per_row - 1) / per_row;
    int max_dim = sg::query_limits().max_image_size_2d;
    int cap_dim = max_dim > 0 ? math::min(max_dim, 4096) : 4096;
    if (cx * per_row > cap_dim || cy * rows > cap_dim) {
        float sx = (float)cap_dim / (float)(cx * per_row);
        float sy = (float)cap_dim / (float)(cy * rows);
        float s  = math::min(sx, sy);
        cx = math::max((int)math::floor((float)cx * s), 1);
        cy = math::max((int)math::floor((float)cy * s), 1);
    }
    FogVolume* v = fog_volume_for(cx, cy, slices);

    Mat4 cw        = camera.matrix_world;
    Mat4 pr        = camera.projection_matrix;
    Vector3 fwd    = ((Vector3) { -cw[2][0], -cw[2][1], -cw[2][2] }).normalize();
    Vector3 right  = ((Vector3) { cw[0][0], cw[0][1], cw[0][2] }).normalize() / pr[0][0];
    Vector3 up     = ((Vector3) { cw[1][0], cw[1][1], cw[1][2] }).normalize() / pr[1][1];
    // Account for off-center or jittered projections when computing the center view ray.
    Vector3 middle = fwd + right * pr[2][0] + up * pr[2][1];
    float near     = math::max(camera.near, CLUSTER_NEAR);
    float far      = math::max(camera.far, near * 2);
    float per_log  = (float)slices / math::ln(far / near);
    fog_pass.frame = (fog_pass.frame + 1) % 1024;

    FogCsParams c;
    c.vf_cam = { ...camera.matrix_world.get_position(), (float)fog_pass.frame };
    c.vf_fwd = { ...middle, clamp(f.noise_amount, 0, 1) };
    c.vf_right = { ...right, f.noise_scale > 0 ? f.noise_scale : 0.15f };
    c.vf_up = { ...up, time };
    c.vf_grid = { cx, cy, slices, v.per_row };
    c.vf_depth = { near, per_log, far, 0 };
    c.vf_wind = { ...f.noise_wind, 0 };
    c.vf_color = p.fog_color;
    c.vf_params = p.fog_params;
    c.vf_box_min = p.fog_box_min;
    c.vf_box_max = p.fog_box_max;
    c.vf_height = p.fog_height;
    c.vf_sun_dir = p.fog_sun_dir;
    c.vf_sun_color = p.fog_sun_color;
    c.vf_clusters = { p.cluster_grid.x, p.cluster_grid.y, p.cluster_grid.z, p.counts.y + p.counts.z > 0 ? 1 : 0 };
    c.vf_cluster_depth = p.cluster_depth;
    c.vf_shadow_mat = p.dir_shadow_mat;
    c.vf_shadow_params = {
        p.dir_shadow_params.x > 0.5f && p.dir_cascade_fwd.w > 0 ? 1 : 0,
        0,
        p.dir_shadow_params.z,
        0
    };
    c.vf_cascade_rect = p.dir_cascade_rect;
    c.vf_cascade_info = p.dir_cascade_info;
    c.vf_cascade_params = p.dir_cascade_params;

    SgBindings bind;
    bind.views[shaders::VIEW_VF_OUT] = v.storage;
    bind.views[shaders::VIEW_VF_SPOT_MAP] = views.spot_map;
    bind.views[shaders::VIEW_VF_SPOT_ATLAS] = views.spot_atlas;
    bind.views[shaders::VIEW_VF_SUN_MAP] = views.sun_map;
    bind.samplers[shaders::SMP_VF_SHADOW_SMP] = views.sampler;
    bind.views[shaders::VIEW_VF_COOKIES] = views.cookies;
    bind.samplers[shaders::SMP_VF_COOKIE_SMP] = views.cookie_sampler;
    LightBufferSet* s = lights_gpu.current;
    bind.views[shaders::VIEW_VF_POINTS] = s.points.view;
    bind.views[shaders::VIEW_VF_SPOTS] = s.spots.view;
    bind.views[shaders::VIEW_VF_SHADOWS] = s.shadows.view;
    bind.views[shaders::VIEW_VF_LIGHT_CLUSTERS] = s.on_gpu ? s.gpu_clusters.view : s.clusters.view;
    bind.views[shaders::VIEW_VF_LIGHT_INDICES] = s.on_gpu ? s.gpu_indices.view : s.indices.view;

    profile::begin_pass(&&{ .compute = true, .label = "drie-fog" });
    sg::apply_pipeline(fog_pass.pipeline);
    sg::apply_bindings(&bind);
    sg::apply_uniforms(shaders::UB_FOG_CS_PARAMS, &&{ .ptr = &c, .size = FogCsParams::size });
    sg::dispatch((cx + 7) / 8, (cy + 7) / 8, 1);
    sg::end_pass();

    p.fog_grid = { cx, cy, slices, v.per_row };
    p.fog_slices = { near, per_log, far, 0 };
    return v.texture;
}