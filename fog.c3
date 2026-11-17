// Volumetric fog (see VolumetricFog in core.c3): before the main pass, a compute shader
// (fog_cs in kernels.glsl) marches the fog through a froxel grid, a cell per few pixels of the
// picture cut into slices of view depth, and the lit shader reads what lies in front of each
// surface from it (fog_volume_at in shader.glsl). Like the clusters of lights, but finer, and
// holding light instead of lists: per slice, the light the fog has scattered towards the
// camera up to its far end and how much of what lies there still gets through.
//
// Unlike a fog pass over the finished picture this needs no depth buffer (the scene is drawn
// in one pass, with no depth before it), it lands before tone mapping, which the lit shader
// does itself, and see-through surfaces get the fog in front of them too.
module drie;
import std::math;
import sokol;

// A grid's image: its slices side by side, `per_row` to a row.
struct FogVolume @local
{
    int cells_x;
    int cells_y;
    int slices;
    int per_row;
    SgImage image;
    SgView storage;
    SgView texture;
}

struct FogPassState @local
{
    SgShader shader;
    SgPipeline pipeline;
    // one per size of grid (renders into targets of other sizes in a frame keep theirs)
    FogVolume[4] volumes;
    int next;
    int frame;
}

FogPassState fog_pass @local;

// Whether the fog of a render goes through the froxel grid: there is something for it to
// scatter, a perspective camera and compute shaders.
fn bool fog_volume_wanted(Fog* f, Camera* camera, bool no_scatter) @private
{
    if (!f || !f.volumetric || camera.camera_type != PERSPECTIVE) return false;
    bool scatter = f.light_scatter > 0 && !no_scatter;
    bool shafts = f.sun_shafts > 0 && f.sun_sharpness > 0;
    if (!scatter && !shafts && f.noise_amount <= 0) return false;
    return compute_supported();
}

fn FogVolume* fog_volume_for(int cells_x, int cells_y, int slices) @local
{
    foreach (&v : fog_pass.volumes)
    {
        if (v.image.id && v.cells_x == cells_x && v.cells_y == cells_y && v.slices == slices) return v;
    }
    FogVolume* v = &fog_pass.volumes[fog_pass.next];
    fog_pass.next = (fog_pass.next + 1) % (int)fog_pass.volumes.len;
    if (v.image.id)
    {
        sg::destroy_view(v.storage);
        sg::destroy_view(v.texture);
        sg::destroy_image(v.image);
    }
    int per_row = (int)math::ceil(math::sqrt((float)slices));
    int rows = (slices + per_row - 1) / per_row;
    *v = { .cells_x = cells_x, .cells_y = cells_y, .slices = slices, .per_row = per_row };
    v.image = sg::make_image(&&{
        .usage = { .storage_image = true },
        .width = cells_x * per_row, .height = cells_y * rows, .pixel_format = RGBA16F,
        .label = "drie-fog-volume",
    });
    v.storage = sg::make_view(&&{ .storage_image = { .image = v.image } });
    v.texture = sg::make_view(&&{ .texture = { .image = v.image } });
    return v;
}

// The views the fog pass reads: the shadow maps (the spot light's drawn every frame, the
// atlas, the sun's) and their comparison sampler.
struct FogShadowViews @private
{
    SgView spot_map;
    SgView spot_atlas;
    SgView sun_map;
    SgSampler sampler;
}

// Marches the fog of `scene` for `camera` into its grid, a picture of width x height pixels.
// `p` holds the render's lights and shadows (build_light_params); the grid's layout goes into
// it for the lit shader. Returns the grid's texture.
fn SgView update_fog_volume(Scene* scene, Camera* camera, FsParams* p, int width, int height, FogShadowViews views,
                            float time) @private
{
    Fog* f = scene.fog;
    if (!fog_pass.shader.id)
    {
        fog_pass.shader = sg::make_shader(&&shaders::fog_shader_desc(sg::query_backend()));
        fog_pass.pipeline = sg::make_pipeline(&&{ .compute = true, .shader = fog_pass.shader, .label = "drie-fog" });
    }
    float res = f.resolution > 0 ? math::min(f.resolution, 1.0f) : 0.125f;
    int cx = math::max((int)math::ceil((float)width * res), 1);
    int cy = math::max((int)math::ceil((float)height * res), 1);
    int slices = math::min(math::max(f.slices > 0 ? f.slices : 96, 2), 256);
    FogVolume* v = fog_volume_for(cx, cy, slices);

    Mat4 cw = camera.matrix_world;
    Mat4 pr = camera.projection_matrix;
    Vector3 fwd = ((Vector3){ -cw[2][0], -cw[2][1], -cw[2][2] }).normalize();
    Vector3 right = ((Vector3){ cw[0][0], cw[0][1], cw[0][2] }).normalize() / pr[0][0];
    Vector3 up = ((Vector3){ cw[1][0], cw[1][1], cw[1][2] }).normalize() / pr[1][1];
    // an off-centre projection (a jittered one, say) moves the middle of the view
    Vector3 middle = fwd + right * pr[2][0] + up * pr[2][1];
    float near = math::max(camera.near, CLUSTER_NEAR);
    float far = math::max(camera.far, near * 2);
    float per_log = (float)slices / math::ln(far / near);
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
    c.vf_shadow_params = { p.dir_shadow_params.x > 0.5f && p.dir_cascade_fwd.w > 0 ? 1 : 0, 0, p.dir_shadow_params.z, 0 };
    c.vf_cascade_rect = p.dir_cascade_rect;
    c.vf_cascade_info = p.dir_cascade_info;
    c.vf_cascade_params = p.dir_cascade_params;

    SgBindings bind;
    bind.views[shaders::VIEW_VF_OUT] = v.storage;
    bind.views[shaders::VIEW_VF_SPOT_MAP] = views.spot_map;
    bind.views[shaders::VIEW_VF_SPOT_ATLAS] = views.spot_atlas;
    bind.views[shaders::VIEW_VF_SUN_MAP] = views.sun_map;
    bind.samplers[shaders::SMP_VF_SHADOW_SMP] = views.sampler;
    LightBufferSet* s = lights_gpu.current;
    bind.views[shaders::VIEW_VF_POINTS] = s.points.view;
    bind.views[shaders::VIEW_VF_SPOTS] = s.spots.view;
    bind.views[shaders::VIEW_VF_SHADOWS] = s.shadows.view;
    bind.views[shaders::VIEW_VF_LIGHT_CLUSTERS] = s.on_gpu ? s.gpu_clusters.view : s.clusters.view;
    bind.views[shaders::VIEW_VF_LIGHT_INDICES] = s.on_gpu ? s.gpu_indices.view : s.indices.view;

    sg::begin_pass(&&{ .compute = true, .label = "drie-fog" });
    sg::apply_pipeline(fog_pass.pipeline);
    sg::apply_bindings(&bind);
    sg::apply_uniforms(shaders::UB_FOG_CS_PARAMS, &&{ .ptr = &c, .size = FogCsParams::size });
    sg::dispatch((cx + 7) / 8, (cy + 7) / 8, 1);
    sg::end_pass();

    p.fog_grid = { cx, cy, slices, v.per_row };
    p.fog_slices = { near, per_log, far, 0 };
    return v.texture;
}
