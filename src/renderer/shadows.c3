// Shadow map allocation, projection math, VSM filtering, and frame shadow orchestration.
// Owns the 32-bit DEPTH maps for the primary auto-updated spot light and cascaded directional sun
// (1 box or 2x2 cascades), the atlas depth-copy and tile-clear passes (shaders/atlas.glsl), and
// the separable 2-pass VSM moment blur (shaders/vsm.glsl, RG32F or RG16F). Computes view-projection
// matrices for spot cones, 6-face point cubes (96-deg FOV), 5-face rect area lights, and
// texel-snapped directional boxes, respecting depth_zero_to_one() ([0,1] vs [-1,1]). Connects to
// update_shadow_atlas(), cull_dispatch(), build_light_params(), and Renderer.render().
module drie;
import std::collections::list, std::math, std::io;
import sokol;

// ---- Shadow map targets and tile operations

// Active atlas draw state read by draw_shadow_item(); `atlas_far == 0` selects orthographic depth.
bool atlas_drawing @private;
float atlas_near @private;
float atlas_far @private;

// Clears the active viewport/scissor tile of the shadow atlas to far depth (z = 1) without
// clearing the entire atlas texture.
fn void clear_atlas_tile() @private {
    if (!state.tile_clear_pipeline.id) {
        state.tile_clear_pipeline = sg::make_pipeline(
            &&{
                .shader = engine_program(true),
                .index_type = UINT32,
                .cull_mode = NONE,
                .layout = vertex_layout({
            shaders::ATTR_SHADOW_POSITION,
            shaders::ATTR_SHADOW_NORMAL,
            shaders::ATTR_SHADOW_UV,
            shaders::ATTR_SHADOW_COLOR0,
            shaders::ATTR_SHADOW_UV1,
            -1,
            shaders::ATTR_SHADOW_INST0,
            shaders::ATTR_SHADOW_INST1,
            shaders::ATTR_SHADOW_INST2,
            shaders::ATTR_SHADOW_HEIGHT,
            -1,
            -1
        }), .depth = {
            .pixel_format = ATLAS_FORMAT,
            .write_enabled = true,
            .compare = ALWAYS
        }, .color_count = 0, .colors[0].pixel_format = NONE, .sample_count = 1, .label = "drie-atlas-tile-clear", });
        state.tile_quad = plane_geometry(2, 2);
    }
    BufferGeometry* g = state.tile_quad;
    g.upload();
    sg::apply_pipeline(state.tile_clear_pipeline);
    SgBindings bind = {
        .vertex_buffers[0] = g.vbuf,
        .vertex_buffers[1] = state.identity_instance,
        .vertex_buffers[2] = zeros(g.vertex_count()),
        .vertex_buffers[4] = zeros(g.vertex_count()),
        .index_buffer = g.ibuf
    };
    Vec4 inst_cull = cull_bind(&bind, -1);
    Texture* white = ready(state.white);
    bind.views[shaders::VIEW_S_MAP] = white.view;
    bind.samplers[shaders::SMP_S_SMP] = white.sampler;
    bind.views[shaders::VIEW_S_ALPHA] = white.view;
    bind.samplers[shaders::SMP_S_ALPHA_SMP] = white.sampler;
    sg::apply_bindings(&bind);
    // Keep quad XY in [-1, 1] and force clip z = w (far plane in both [0,1] and [-1,1] depth).
    Mat4 to_far = mat4_identity();
    to_far[2][2] = 0;
    to_far[3][2] = 1;
    ShadowVsParams vs = {
        .s_view_proj = to_far,
        .s_model = mat4_identity(),
        .s_uv_transform = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 } },
        .s_inst_cull = inst_cull
    };
    sg::apply_uniforms(shaders::UB_SHADOW_VS_PARAMS, &&{ .ptr = &vs, .size = ShadowVsParams::size });
    ShadowFsParams fs;
    sg::apply_uniforms(shaders::UB_SHADOW_FS_PARAMS, &&{ .ptr = &fs, .size = ShadowFsParams::size });
    sg::draw(0, (int)g.index_count, 1);
}

// Scales a light's requested `map_size` by Renderer.shadow_map_scale, clamped to [4, 16384].
fn int shadow_map_size(int map_size) @private {
    float scale = state.renderer.shadow_map_scale > 0 ? state.renderer.shadow_map_scale : 1;
    return math::clamp((int)(map_size * scale + 0.5f), 4, 16384);
}

fn void ensure_shadow_map(int size) @local {
    if (state.shadow_size == size) return;
    if (state.shadow_size) {
        sg::destroy_view(state.shadow_attachment);
        sg::destroy_view(state.shadow_view);
        sg::destroy_image(state.shadow_image);
    } else {
        state.shadow_sampler = sg::make_sampler(
            &&{
                .min_filter = LINEAR,
                .mag_filter = LINEAR,
                .wrap_u = CLAMP_TO_EDGE,
                .wrap_v = CLAMP_TO_EDGE,
                .compare = LESS_EQUAL,
            }
        );
    }
    state.shadow_size = size;
    state.shadow_image = sg::make_image(
        &&{
            .usage = { .depth_stencil_attachment = true },
            .width = size,
            .height = size,
            .sample_count = 1,
            .pixel_format = DEPTH,
        }
    );
    state.shadow_attachment = sg::make_view(&&{ .depth_stencil_attachment = { .image = state.shadow_image } });
    state.shadow_view = sg::make_view(&&{ .texture = { .image = state.shadow_image } });
}

fn void ensure_sun_map(int size) @local {
    if (state.sun_size == size) return;
    if (state.sun_size) {
        sg::destroy_view(state.sun_attachment);
        sg::destroy_view(state.sun_view);
        sg::destroy_image(state.sun_image);
    }
    state.sun_size = size;
    state.sun_image = sg::make_image(
        &&{
            .usage = { .depth_stencil_attachment = true },
            .width = size,
            .height = size,
            .sample_count = 1,
            .pixel_format = DEPTH,
        }
    );
    state.sun_attachment = sg::make_view(&&{ .depth_stencil_attachment = { .image = state.sun_image } });
    state.sun_view = sg::make_view(&&{ .texture = { .image = state.sun_image } });
}

// Ensures the shadow atlas texture is `width` x `height`, preserving existing texels at (0, 0).
fn void ensure_shadow_atlas(int width, int height) @private {
    grow_depth(
        &state.atlas_image,
        &state.atlas_attachment,
        &state.atlas_view,
        &state.atlas_width,
        &state.atlas_height,
        width,
        height
    );
}

// Ensures the static caster cache texture is `width` x `height`, preserving existing texels at (0, 0).
fn void ensure_atlas_cache(int width, int height) @private {
    grow_depth(
        &state.cache_image,
        &state.cache_attachment,
        &state.cache_view,
        &state.cache_width,
        &state.cache_height,
        width,
        height
    );
}

// Reallocates a DEPTH16 texture to `width` x `height` and copies overlapping texels from the
// existing texture into the top-left corner via copy_depth().
fn void grow_depth(
    SgImage* image,
    SgView* attachment,
    SgView* view,
    int* cur_width,
    int* cur_height,
    int width,
    int height
) @local {
    if (*cur_width == width && *cur_height == height) return;
    SgImage old_image     = *image;
    SgView old_attachment = *attachment;
    SgView old_view       = *view;
    int old_width         = *cur_width;
    int old_height        = *cur_height;
    *cur_width = width;
    *cur_height = height;
    *image = sg::make_image(
        &&{
            .usage = { .depth_stencil_attachment = true },
            .width = width,
            .height = height,
            .sample_count = 1,
            .pixel_format = ATLAS_FORMAT,
        }
    );
    *attachment = sg::make_view(&&{ .depth_stencil_attachment = { .image = *image } });
    *view = sg::make_view(&&{ .texture = { .image = *image } });
    if (!old_width) return;
    profile::begin_pass(
        &&{
            .action = { .depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 } },
            .attachments = { .depth_stencil = *attachment },
            .label = "drie-shadow-atlas-grow",
        }
    );
    copy_depth(old_view, 0, 0, 0, 0, math::min(old_width, width), math::min(old_height, height));
    sg::end_pass();
    sg::destroy_view(old_attachment);
    sg::destroy_view(old_view);
    sg::destroy_image(old_image);
}

// Copies a `w` x `h` depth rectangle from `(sx, sy)` of `src` to `(dx, dy)` of the active depth
// attachment using shaders/atlas.glsl.
fn void copy_depth(SgView src, int sx, int sy, int dx, int dy, int w, int h) @private {
    if (!state.atlas_copy_pipeline.id) {
        state.atlas_copy_sampler = sg::make_sampler(
            &&{ .min_filter = NEAREST, .mag_filter = NEAREST, .wrap_u = CLAMP_TO_EDGE, .wrap_v = CLAMP_TO_EDGE }
        );
        state.atlas_copy_pipeline = sg::make_pipeline(
            &&{
                .shader = make_program(shaders::atlas_copy_shader_desc(sg::query_backend())),
                .depth = { .pixel_format = ATLAS_FORMAT, .write_enabled = true, .compare = ALWAYS },
                .color_count = 0,
                .colors[0].pixel_format = NONE,
                .sample_count = 1,
                .label = "drie-atlas-copy",
            }
        );
    }
    bool top_left = depth_zero_to_one();
    sg::apply_pipeline(state.atlas_copy_pipeline);
    SgBindings bind;
    bind.views[shaders::VIEW_ATLAS_SRC] = src;
    bind.samplers[shaders::SMP_ATLAS_SRC_SMP] = state.atlas_copy_sampler;
    sg::apply_bindings(&bind);
    sg::apply_viewport(dx, dy, w, h, top_left);
    sg::apply_scissor_rect(dx, dy, w, h, top_left);
    shaders::AtlasCopyParams p = { .atlas_shift = { (float)(sx - dx), (float)(sy - dy), 0, 0 } };
    sg::apply_uniforms(shaders::UB_ATLAS_COPY_PARAMS, &&{ .ptr = &p, .size = shaders::AtlasCopyParams::size });
    sg::draw(0, 3, 1);
}

// ---- Variance shadow map blur

// True when Renderer.shadow_type is set to VSM.
fn bool vsm_on() @private => state.renderer && state.renderer.shadow_type == VSM;

// Ensures `v` holds moment and scratch targets sized `width` x `height` (RG32F when renderable
// and filterable, otherwise RG16F). Returns true when newly allocated or resized.
fn bool ensure_vsm(VsmMaps* v, int width, int height) @private {
    if (v.width == width && v.height == height) return false;
    if (!state.vsm_format) {
        SgPixelformatInfo f = sg::query_pixelformat(RG32F);
        state.vsm_format = f.render && f.filter ? SgPixelFormat.RG32F : SgPixelFormat.RG16F;
        state.vsm_sampler = sg::make_sampler(
            &&{ .min_filter = LINEAR, .mag_filter = LINEAR, .wrap_u = CLAMP_TO_EDGE, .wrap_v = CLAMP_TO_EDGE }
        );
        state.vsm_point_sampler = sg::make_sampler(
            &&{ .min_filter = NEAREST, .mag_filter = NEAREST, .wrap_u = CLAMP_TO_EDGE, .wrap_v = CLAMP_TO_EDGE }
        );
        state.vsm_blur_pipeline = sg::make_pipeline(
            &&{
                .shader = state.vsm_blur_shader,
                .colors[0].pixel_format = state.vsm_format,
                .depth.pixel_format = NONE,
                .sample_count = 1,
                .label = "drie-vsm-blur",
            }
        );
    }
    if (v.width) {
        sg::destroy_view(v.view);
        sg::destroy_view(v.attachment);
        sg::destroy_image(v.image);
        sg::destroy_view(v.scratch_view);
        sg::destroy_view(v.scratch_attachment);
        sg::destroy_image(v.scratch);
    }
    v.width = width;
    v.height = height;
    SgImageDesc desc = {
        .usage = { .color_attachment = true },
        .width = width,
        .height = height,
        .sample_count = 1,
        .pixel_format = state.vsm_format,
    };
    v.image = sg::make_image(&desc);
    v.view = sg::make_view(&&{ .texture = { .image = v.image } });
    v.attachment = sg::make_view(&&{ .color_attachment = { .image = v.image } });
    v.scratch = sg::make_image(&desc);
    v.scratch_view = sg::make_view(&&{ .texture = { .image = v.scratch } });
    v.scratch_attachment = sg::make_view(&&{ .color_attachment = { .image = v.scratch } });
    return true;
}

// Tile rectangle and blur settings passed to vsm_blur() (texel coordinates match shadow viewports).
struct VsmTile @private {
    int x;
    int y;
    int w;
    int h;
    float radius;
    int samples;
}

// Converts `tiles` of depth texture `depth` into VSM moments in `v` via shaders/vsm.glsl:
// vertical box blur from `depth` into `v.scratch`, then horizontal blur into `v.image`.
// When `load` is true, untouched tiles in `v.image` preserve their existing moments.
fn void vsm_blur(VsmMaps* v, SgView depth, VsmTile[] tiles, bool load) @private {
    if (!tiles.len) return;
    bool top_left = depth_zero_to_one();
    for (int pass = 0; pass < 2; pass++) {
        bool along_x = pass == 1;
        profile::begin_pass(
            &&{
                .action = {
                    .colors[0] = {
                        .load_action = along_x && load ? LOAD : CLEAR,
                        .store_action = STORE,
                        .clear_value = { 1, 0, 0, 1 }
                    }
                },
                .attachments = { .colors[0] = along_x ? v.attachment : v.scratch_attachment },
                .label = along_x ? "drie-vsm-blur-x" : "drie-vsm-blur-y",
            }
        );
        sg::apply_pipeline(state.vsm_blur_pipeline);
        SgBindings bind;
        bind.views[shaders::VIEW_VSM_SRC] = along_x ? v.scratch_view : depth;
        bind.samplers[shaders::SMP_VSM_SRC_SMP] = state.vsm_point_sampler;
        sg::apply_bindings(&bind);
        foreach (t : tiles) {
            sg::apply_viewport(t.x, t.y, t.w, t.h, top_left);
            sg::apply_scissor_rect(t.x, t.y, t.w, t.h, top_left);
            shaders::VsmParams p = {
                .vsm_rect = { (float)t.x, (float)t.y, (float)t.w, (float)t.h },
                .vsm_spread = { math::max(t.radius, 0.0f), (float)math::max(t.samples, 1), along_x ? 1 : 0, 0 },
            };
            sg::apply_uniforms(shaders::UB_VSM_PARAMS, &&{ .ptr = &p, .size = shaders::VsmParams::size });
            sg::draw(0, 3, 1);
        }
        sg::end_pass();
    }
}

// ---- Light shadow projections and cascades

// Returns the LightShadow block for a spot, point, rect area, or directional light.
fn LightShadow* light_shadow(Light* l) @private {
    switch (l.light_type) {
        case SPOT: return &((SpotLight*)l).shadow;
        case POINT: return &((PointLight*)l).shadow;
        case RECT_AREA: return &((RectAreaLight*)l).shadow;
        default: return &((DirectionalLight*)l).shadow;
    }
}

// True for lights that project multi-face cube/hemisphere shadow maps (point and rect area).
fn bool casts_cube(Light* l) @private => l.light_type == POINT || l.light_type == RECT_AREA;

// Basis directions (+X, -X, +Y, -Y, +Z, -Z) for the 6 faces of a point light shadow cube.
const Vector3[6] CUBE_FORWARD @local = {
    { 1, 0, 0 },
    { -1, 0, 0 },
    { 0, 1, 0 },
    { 0, -1, 0 },
    { 0, 0, 1 },
    { 0, 0, -1 }
};
const Vector3[6] CUBE_UP @local      = {
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 0, -1 },
    { 0, 0, 1 },
    { 0, 1, 0 },
    { 0, 1, 0 }
};
// Slightly wider than 90 degrees so PCF taps near a face edge remain inside the tile.
const float POINT_SHADOW_FOV @local  = 96;

fn float point_shadow_far(Light* l) @private {
    LightShadow* sh = light_shadow(l);
    if (sh.far > 0) return sh.far;
    if (l.light_type == POINT && ((PointLight*)l).distance > 0) return ((PointLight*)l).distance;
    return 50.0f;
}

// View-projection matrix for face `face` of a point light cube (`relative`: camera at origin).
fn Mat4 point_shadow_view_proj(Light* l, int face, bool relative = false) @local {
    Vector3 pos = l.matrix_world.get_position();
    Mat4 world  = mat4_look_at(pos, pos + CUBE_FORWARD[face], CUBE_UP[face], true);
    world[3] = { pos.x, pos.y, pos.z, 1 };
    if (relative) world[3] = { 0, 0, 0, 1 };
    Mat4 proj = mat4_perspective(
        POINT_SHADOW_FOV,
        1,
        math::max(light_shadow(l).near, 0.01f),
        point_shadow_far(l),
        1,
        depth_zero_to_one()
    );
    return proj.mul(world.invert());
}

// Tangent margin behind a rect area light's plane included in its 4 half-height side shadow maps
// (face 0 looks forward along -Z; faces 1..4 look along +X, -X, +Y, -Y with asymmetric frusta).
const float RECT_SIDE_BEHIND @local = 0.1f;

fn Mat4 rect_shadow_view_proj(Light* l, int face, bool relative = false) @local {
    Mat4 m        =  l.matrix_world;
    Vector3 x     =  ((Vector3) { m[0][0], m[0][1], m[0][2] }).normalize();
    Vector3 y     =  ((Vector3) { m[1][0], m[1][1], m[1][2] }).normalize();
    Vector3 front = -((Vector3) { m[2][0], m[2][1], m[2][2] }).normalize();
    Vector3 dir   =  front;
    Vector3 up    =  y;
    switch (face) {
        case 1: dir = x; up = front;
        case 2: dir = -x; up = front;
        case 3: dir = y; up = front;
        case 4: dir = -y; up = front;
    }
    Vector3 pos = m.get_position();
    Mat4 world  = mat4_look_at(pos, pos + dir, up, true);
    world[3] = relative ? (Vec4) { 0, 0, 0, 1 } : (Vec4) { pos.x, pos.y, pos.z, 1 };
    Mat4 proj = mat4_perspective(
        POINT_SHADOW_FOV,
        1,
        math::max(light_shadow(l).near, 0.01f),
        point_shadow_far(l),
        1,
        depth_zero_to_one()
    );
    if (face > 0) {
        // Asymmetric vertical frustum from -RECT_SIDE_BEHIND up to the front face corner.
        float top    =  math::tan(POINT_SHADOW_FOV * DEG2RAD * 0.5f);
        float bottom = -RECT_SIDE_BEHIND;
        proj[1][1] = 2 / (top - bottom);
        proj[2][1] = (top + bottom) / (top - bottom);
    }
    return proj.mul(world.invert());
}

// Dispatches to rect_shadow_view_proj() or point_shadow_view_proj() for face `face`.
fn Mat4 cube_view_proj(Light* l, int face, bool relative = false) @local {
    return l.light_type == RECT_AREA ? rect_shadow_view_proj(l, face, relative)
        : point_shadow_view_proj(l, face, relative);
}

// View-projection matrix for an atlas light (spot frustum, cube face, or directional box).
fn Mat4 light_view_proj(Light* l, Camera* camera, bool relative = false, int face = 0, int size = 0) @private {
    if (l.light_type == SPOT) return shadow_view_proj((SpotLight*)l, relative);
    if (casts_cube(l)) return cube_view_proj(l, face, relative);
    return directional_shadow_view_proj((DirectionalLight*)l, camera, size);
}

// Returns the primary spot light that receives its own dedicated 32-bit shadow map each frame.
fn SpotLight* shadow_light() @local {
    foreach (l : state.lights) {
        if (l.light_type != SPOT || !l.cast_shadow || !l.visible || l.intensity <= 0) continue;
        if (((SpotLight*)l).shadow.auto_update) return (SpotLight*)l;
    }
    return null;
}

// Returns the first visible shadow-casting directional light (receives the dedicated sun map).
fn DirectionalLight* directional_shadow_light() @local {
    foreach (l : state.lights) {
        if (l.light_type == DIRECTIONAL && l.cast_shadow && l.visible && l.intensity > 0) return (DirectionalLight*)l;
    }
    return null;
}

// Orthographic view-projection for a directional light's single shadow box, snapped to whole
// texels of a `size`-wide map (defaults to shadow.map_size).
fn Mat4 directional_shadow_view_proj(DirectionalLight* d, Camera* camera, int size = 0) @local {
    Vector3 target = d.target ? d.target.get_world_position() : (Vector3) { 0, 0, 0 };
    Vector3 dir    = (d.matrix_world.get_position() - target).normalize();
    float extent   = math::max(d.shadow_extent, 0.1f);
    Vector3 center = target;
    // Keep the box anchored on the target during lightmap baking regardless of camera position.
    if (d.shadow_follow && !state.bake.on) {
        Vector3 at    = camera.matrix_world.get_position();
        Mat4 cw       = camera.matrix_world;
        Vector3 ahead = { -cw[2][0], 0, -cw[2][2] };
        if (ahead.length() > 1e-4f) center = at + ahead.normalize() * (extent * 0.55f);
    }
    return sun_box_view_proj(d, dir, center, extent, size ?: shadow_map_size(d.shadow.map_size));
}

// Builds a texel-snapped orthographic view-projection looking along `-dir` around `center` with
// half-extent `radius`, preventing shadow edge shimmer as the camera translates.
fn Mat4 sun_box_view_proj(DirectionalLight* d, Vector3 dir, Vector3 center, float radius, int size) @local {
    Vector3 up    = math::abs(dir.y) > 0.99f ? (Vector3) { 1, 0, 0 } : (Vector3) { 0, 1, 0 };
    Vector3 right = up.cross(dir).normalize();
    up = dir.cross(right);
    float texel = 2 * radius / (float)math::max(size, 1);
    center -= right * (center.dot(right) % texel) + up * (center.dot(up) % texel);
    float far   = d.shadow.far > 0 ? d.shadow.far : 300.0f;
    Vector3 eye = center + dir * (far * 0.5f);
    Mat4 world  = mat4_look_at(eye, center, up, true);
    world[3] = { eye.x, eye.y, eye.z, 1 };
    Mat4 proj = mat4_orthographic(
        -radius,
        radius,
        radius,
        -radius,
        math::max(d.shadow.near, 0.01f),
        far,
        depth_zero_to_one()
    );
    return proj.mul(world.invert());
}

// Splits a perspective camera's view up to `d.shadow.max_distance` into 2..4 practical-split
// cascades, fitting a rotation-invariant bounding sphere to each slice and populating
// `state.sun_*`. Returns 0 for single-box mode or non-perspective cameras.
fn int fit_sun_cascades(DirectionalLight* d, Camera* camera, Vector3 dir) @local {
    int n = math::min(math::max(d.shadow.cascades, 1), 4);
    if (n < 2 || camera.camera_type != PERSPECTIVE) return 0;
    Mat4 p = camera.projection_matrix;
    // Half-FOV tangents along X and Y.
    float tx = p[0][0] != 0 ? 1 / math::abs(p[0][0]) : 1;
    float ty = p[1][1] != 0 ? 1 / math::abs(p[1][1]) : 1;
    float k = tx * tx + ty * ty;
    float near = math::max(camera.near, 0.01f);
    float far = math::max(math::min(d.shadow.max_distance > 0 ? d.shadow.max_distance : 150.0f, camera.far), near + 1);
    float lambda = clamp(d.shadow.split_lambda, 0, 1);
    Mat4 cw = camera.matrix_world;
    Vector3 at = cw.get_position();
    Vector3 fwd = ((Vector3) { -cw[2][0], -cw[2][1], -cw[2][2] }).normalize();
    state.sun_forward = fwd;
    int tile   = shadow_map_size(d.shadow.map_size);
    float from = near;
    for (int i = 0; i < n; i++) {
        float f      = (float)(i + 1) / (float)n;
        float to     = lambda * near * math::pow(far / near, f) + (1 - lambda) * (near + (far - near) * f);
        // Minimal bounding sphere of the frustum slice, centered on the camera view axis.
        float z      = math::min((from + to) * (1 + k) * 0.5f, to);
        float radius = math::sqrt((to - z) * (to - z) + to * to * k);
        // Quantize radius to 1/16 m steps so texel size stays constant across camera rotations.
        radius = math::ceil(radius * 16) / 16;
        state.sun_view_proj[i] = sun_box_view_proj(d, dir, at + fwd * z, radius, tile);
        state.sun_rect[i] = { (float)(i % 2) * 0.5f, (float)(i / 2) * 0.5f, 0.5f, 0.5f };
        state.sun_info[i] = { to, 2 * radius / (float)tile, 0, 0 };
        from = to;
    }
    return n;
}

// Fits sun cascade projections, allocates the 2x2 (or 1x1) sun depth texture, and gathers casters
// per cascade into `state.sun_casters`.
fn void prepare_sun_shadow(Scene* scene, Camera* camera, DirectionalLight* sun) @local {
    foreach (&l : state.sun_casters) l.clear();
    if (!sun) {
        ensure_sun_map(state.sun_size ?: 4);
        return;
    }
    Vector3 target = sun.target ? sun.target.get_world_position() : (Vector3) { 0, 0, 0 };
    Vector3 dir    = (sun.matrix_world.get_position() - target).normalize();
    int tile       = shadow_map_size(sun.shadow.map_size);
    int n          = fit_sun_cascades(sun, camera, dir);
    state.sun_cascades = n;
    if (!n) {
        state.sun_view_proj[0] = directional_shadow_view_proj(sun, camera);
        state.sun_rect[0] = { 0, 0, 1, 1 };
        state.sun_info[0] = { 1e9f, 2 * math::max(sun.shadow_extent, 0.1f) / (float)tile, 0, 0 };
    }
    ensure_sun_map(n ? tile * 2 : tile);
    for (int i = 0; i < math::max(n, 1); i++) {
        Frustum light_frustum = frustum_from(state.sun_view_proj[i]);
        gather_casters(scene, camera, &light_frustum, sun.shadow.skip_instanced, &state.sun_casters[i]);
    }
}

// Renders gathered casters into `state.sun_attachment` and runs VSM blur when VSM is active.
fn void draw_sun_shadow(DirectionalLight* sun) @local {
    if (!sun) return;
    int n         = state.sun_cascades;
    int tile      = shadow_map_size(sun.shadow.map_size);
    bool top_left = depth_zero_to_one();
    profile::begin_pass(
        &&{ // Store depth explicitly so the main and deferred passes can sample the shadow map.
            .action = { .depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 } },
            .attachments = { .depth_stencil = state.sun_attachment },
            .label = "drie-sun-cascades",
        }
    );
    for (int i = 0; i < math::max(n, 1); i++) {
        if (n) {
            sg::apply_viewport((i % 2) * tile, (i / 2) * tile, tile, tile, top_left);
            sg::apply_scissor_rect((i % 2) * tile, (i / 2) * tile, tile, tile, top_left);
        }
        foreach (&item : state.sun_casters[i]) draw_shadow_item(item, state.sun_view_proj[i]);
    }
    sg::end_pass();
    if (!vsm_on()) return;
    // Scale VSM blur radius down for coarser far cascades (down to 1 texel) to match world-space softness.
    ensure_vsm(&state.vsm_sun, state.sun_size, state.sun_size);
    VsmTile[4] tiles;
    for (int i = 0; i < math::max(n, 1); i++) {
        float scale  = n ? state.sun_info[i].y / math::max(state.sun_info[0].y, 1e-6f) : 1;
        float radius = math::max(sun.shadow.radius / scale, math::min(sun.shadow.radius, 1.0f));
        tiles[i] = {
            n ? (i % 2) * tile : 0,
            n ? (i / 2) * tile : 0,
            n ? tile : state.sun_size,
            n ? tile : state.sun_size,
            radius,
            sun.shadow.blur_samples
        };
    }
    vsm_blur(&state.vsm_sun, state.sun_view, tiles[:math::max(n, 1)], false);
}

// ---- Caster gathering and GPU packing

// Collects shadow casters overlapping `frustum` into `out` and registers GPU instance cull jobs.
fn void gather_casters(
    Scene* scene,
    Camera* camera,
    Frustum* frustum,
    bool skip_instanced,
    List{DrawItem}* out
) @local {
    state.casters.clear();
    state.skip_instanced = skip_instanced;
    collect_casters(scene, camera, frustum);
    state.skip_instanced = false;
    out.clear();
    int view = cull_view(frustum);
    foreach (item : state.casters) {
        item.job = cull_job(item.mesh, view) + 1;
        out.push(item);
    }
    state.casters.clear();
}

// Collects shadow casters inside `frustum` into `state.casters` from `scene` (or `state.caster_roots`
// during lightmap baking).
fn void collect_casters(Scene* scene, Camera* camera, Frustum* frustum) @private {
    Vector3 from = camera.matrix_world.get_position();
    if (!state.bake.on) {
        collect(scene, camera.matrix_world_inverse, frustum, from, true);
        return;
    }
    foreach (o : state.caster_roots) collect(o, camera.matrix_world_inverse, frustum, from, true);
}

// Maps clip space to shadow texture UV (flipping V on depth-[0,1] backends) and [0,1] depth.
fn Mat4 shadow_bias() @private {
    bool zo   = depth_zero_to_one();
    Mat4 bias = mat4_identity();
    bias[0][0] = 0.5f;
    bias[3][0] = 0.5f;
    bias[1][1] = zo ? -0.5f : 0.5f;
    bias[3][1] = 0.5f;
    bias[2][2] = zo ? 1.0f : 0.5f;
    bias[3][2] = zo ? 0.0f : 0.5f;
    return bias;
}

fn Vector3 spot_direction(SpotLight* sl) @private {
    Vector3 target = sl.target ? sl.target.get_world_position() : (Vector3) { 0, 0, 0 };
    return (target - sl.matrix_world.get_position()).normalize();
}

// Packs the atlas shadow faces of a point (6 faces) or rect area (5 faces) light into `g.shadows`
// and writes the base index and faded intensity into `shadow.xy`.
fn void cube_shadows(Light* l, Vec4* shadow, LightData* g, bool vsm) @private {
    if (!atlas_has_shadow(l)) return;
    AtlasSlot* slot = atlas_slot_of(l);
    LightShadow* ls = light_shadow(l);
    (*shadow).x = (float)g.shadows.len();
    (*shadow).y = clamp(ls.intensity, 0, 1) * slot.fade;
    g.point_shadows = true;
    if (ls.softness > 0) g.soft_shadows = true;
    float width = 2 * math::tan(POINT_SHADOW_FOV * DEG2RAD * 0.5f);
    bool zo     = depth_zero_to_one();
    for (int f = 0; f < slot.maps; f++) {
        AtlasMap* m = &slot.map[f];
        GpuSpotShadow sh;
        sh.mat = shadow_bias().mul(cube_view_proj(l, f, true));
        sh.params = { 2, ls.normal_bias, atlas_texel(m), vsm ? ls.bias : ls.radius };
        sh.soft = { ls.softness, math::max(ls.near, 0.01f), point_shadow_far(l), zo ? width : -width };
        sh.rect = atlas_rect(m);
        g.shadows.push(sh);
    }
}

bool cookie_told @local; // Guards the one-time warning when multiple SpotLight.map textures appear.

// Packs `sl.map` (spot cookie projection) and its rotated cone-scaled UV axes into `gs`. All
// cookies in a frame must share one texture (atlas sub-rects via `map_rect`).
fn void spot_map(SpotLight* sl, GpuSpotLight* gs, LightData* g) @private {
    if (!sl.map) return;
    Texture* t = ready(sl.map);
    // Texture still loading asynchronously: skip cookie projection until ready.
    if (t == state.white || !t.view.id) return;
    if (!g.cookie) g.cookie = t;
    if (g.cookie != t) {
        if (!cookie_told) io::eprintfn(
            "drie: spot lights with different maps in one frame: only the first map is used; put them on one texture and pick each light's part with map_rect"
        );
        cookie_told = true;
        return;
    }
    float[4] rect = sl.map_rect;
    if (rect[2] <= 0 || rect[3] <= 0) rect = { 0, 0, 1, 1 };
    Vector3 dir = spot_direction(sl);
    Vector3 up  = math::abs(dir.y) > 0.99f ? (Vector3) { 1, 0, 0 } : (Vector3) { 0, 1, 0 };
    Vector3 u   = dir.cross(up).normalize();
    Vector3 v   = u.cross(dir);
    float c     = math::cos(sl.map_rotation);
    float s     = math::sin(sl.map_rotation);
    Vector3 ru  = u * c + v * s;
    Vector3 rv  = v * c - u * s;
    float k     = 1.0f / math::tan(clamp(sl.angle, 0.01f, PI / 2 - 0.01f));
    ru = ru * k;
    rv = rv * k;
    float size = (float)math::max(t.width, 1) * rect[2];
    gs.map = { rect[0], rect[1], rect[2], rect[3] };
    gs.map_u = { ru.x, ru.y, ru.z, rv.x };
    gs.map_v = { rv.y, rv.z, math::log2(size), 0 };
}

// Far clip distance for a spot light's shadow frustum.
fn float spot_shadow_far(
    SpotLight* sl
) @private => sl.shadow.far > 0 ? sl.shadow.far : (sl.distance > 0 ? sl.distance : 50.0f);

// View-projection matrix for a spot light's shadow map (`relative`: translates light to origin).
fn Mat4 shadow_view_proj(SpotLight* sl, bool relative = false) @private {
    Vector3 pos = sl.matrix_world.get_position();
    Vector3 dir = spot_direction(sl);
    Vector3 up  = math::abs(dir.y) > 0.99f ? (Vector3) { 1, 0, 0 } : (Vector3) { 0, 1, 0 };
    Mat4 world  = mat4_look_at(pos, pos + dir, up, true);
    world[3] = { pos.x, pos.y, pos.z, 1 };
    if (relative) world[3] = { 0, 0, 0, 1 };
    float fov = math::min(sl.angle * 2 * RAD2DEG * clamp(sl.shadow.focus, 0.05f, 1) + 4, 150.0f);
    float far = spot_shadow_far(sl);
    Mat4 proj = mat4_perspective(fov, 1, math::max(sl.shadow.near, 0.01f), far, 1, depth_zero_to_one());
    return proj.mul(world.invert());
}

// ---- Frame shadow orchestration

// Prepares casters, runs GPU instance culling (`cull_dispatch`), and renders the primary spot
// shadow map, sun cascades, and shadow atlas tiles (plus VSM blurs when enabled). Respects
// `Renderer.shadow_map_auto_update` while ensuring placeholder maps always exist for shader
// bindings. In `bake` mode (render_bake), renders only bake-scoped atlas tiles so main maps stay
// intact. Returns the primary spot light, if any.
fn SpotLight* prepare_and_draw_shadows(Scene* scene, Camera* camera, bool bake = false) @private {
    ulong pt          = profile::start();
    SpotLight* caster = bake ? null : shadow_light();
    // Retain existing map size when no primary spot caster is active to avoid per-frame realloc.
    ensure_shadow_map(caster ? shadow_map_size(caster.shadow.map_size) : (state.shadow_size ?: 4));
    Mat4 caster_vp;
    bool shadows = state.renderer.shadow_map_auto_update || bake;
    if (shadows) state.spot_casters.clear();
    if (caster && !shadows) caster_vp = shadow_view_proj(caster);
    if (caster && shadows) {
        caster_vp = shadow_view_proj(caster);
        Frustum light_frustum = frustum_from(caster_vp);
        gather_casters(scene, camera, &light_frustum, caster.shadow.skip_instanced, &state.spot_casters);
    }
    profile::stop("render: spot casters", pt);
    pt = profile::start();
    // Dedicated sun shadow map allows a directional sun and primary spot to cast simultaneously.
    DirectionalLight* sun = shadows ? directional_shadow_light() : state.sun_caster;
    if (bake) sun = null;
    if (shadows && !bake) {
        state.sun_caster = sun;
        prepare_sun_shadow(scene, camera, sun);
    }
    profile::stop("render: sun casters", pt);
    pt = profile::start();
    // Dispatch GPU instance culling once across camera, spot, and sun cascade views.
    cull_dispatch();
    profile::stop("render: cull dispatch", pt);
    pt = profile::start();

    if (caster && shadows) {
        profile::begin_pass(
            &&{ // Store depth explicitly so the main and deferred passes can sample the spot map.
                .action = { .depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 } },
                .attachments = { .depth_stencil = state.shadow_attachment },
                .label = "drie-spot-shadow",
            }
        );
        Mat4 caster_rel    = shadow_view_proj(caster, true);
        Vector3 caster_pos = caster.matrix_world.get_position();
        foreach (&item : state.spot_casters) draw_shadow_item(item, caster_rel, caster_pos);
        sg::end_pass();
        if (vsm_on()) {
            int size = state.shadow_size;
            ensure_vsm(&state.vsm_spot, size, size);
            VsmTile[1] tiles = { { 0, 0, size, size, caster.shadow.radius, caster.shadow.blur_samples } };
            vsm_blur(&state.vsm_spot, state.shadow_view, &tiles, false);
        }
    }
    profile::stop("render: spot shadow", pt);
    pt = profile::start();
    if (shadows && !bake) draw_sun_shadow(sun);
    profile::stop("render: sun shadow", pt);
    pt = profile::start();
    if (shadows) update_shadow_atlas(scene, camera, bake, caster);
    // Ensure valid textures exist even when shadows are skipped (e.g. reflector views).
    if (!shadows || bake) {
        ensure_sun_map(state.sun_size ?: 4);
        if (!state.atlas_width) ensure_shadow_atlas(ATLAS_MIN_BLOCK, ATLAS_MIN_BLOCK);
    }
    // Keep VSM moment targets allocated and sized to match all three shadow textures.
    if (vsm_on()) {
        ensure_vsm(&state.vsm_spot, state.shadow_size, state.shadow_size);
        ensure_vsm(&state.vsm_sun, state.sun_size, state.sun_size);
        ensure_vsm(&state.vsm_atlas, state.atlas_width, state.atlas_height);
    }
    note_atlas_info();
    profile::stop("render: spot atlas", pt);
    return caster;
}

// Records shadow atlas dimensions, GPU memory usage, and active light count into RenderInfo.
fn void note_atlas_info() @local {
    RenderInfo* info = &state.renderer.info;
    info.atlas_width = state.atlas_width;
    info.atlas_height = state.atlas_height;
    usz pixels = (usz)info.atlas_width * (usz)info.atlas_height;
    info.atlas_bytes = (pixels + (usz)state.cache_width * (usz)state.cache_height) * 2;
    // Include VSM moment image plus scratch blur target when active.
    if (state.vsm_atlas.width) info.atlas_bytes += (usz)state.vsm_atlas.width * (usz)state.vsm_atlas.height
        * (usz)(state.vsm_format == RG32F ? 16
        : 8);
    info.atlas_lights = 0;
    foreach (&s : state.atlas_slots) {
        if (s.size && s.seen == state.frame) info.atlas_lights++;
    }
}