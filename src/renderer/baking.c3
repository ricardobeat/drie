// Offscreen baking and environment capture: lightmaps, irradiance probes, and env maps.
// Implements progressive UV1 lightmap baking (Renderer.render_bake, called by ProgressiveLightMap
// in src/lighting/lightmap.c3, with seam dilation via shaders/lightmap.glsl), 6-direction
// IrradianceVolume baking (bake_irradiance), CPU-readback equirectangular capture
// (capture_environment), procedural GPU env rendering (render_environment), and GPU cubemap
// probe resampling (render_probe via drie::nodes). Restores renderer target/viewport state after
// each pass and accounts for backend row-origin orientation (depth_zero_to_one()).
module drie;
import std::collections::list, std::math, std::io;
import sokol, drie::readback, drie::nodes;

// ---- Progressive lightmap baking

// Creates or retrieves the UV1 lightmap baking pipeline for `gen` (`MaterialKey.bake`): no depth
// buffer, no face culling (UV unwrap winding varies), and no blending.
fn SgPipeline gen_bake_pipeline(MaterialProgram* gen, SgPixelFormat format) @private {
    // Bit 63 keeps bake keys disjoint from get_pipeline() keys on the same MaterialProgram.
    ulong key         = 1ul << 63 | (ulong)(uint)format;
    SgPipeline? found = gen.pipelines[key];
    if (try found) return found;
    SgPipeline pip = sg::make_pipeline(
        &&{
            .shader = gen.program.make(),
            .index_type = UINT32,
            .cull_mode = NONE,
            .layout = vertex_layout({
        shaders::ATTR_THREE_POSITION,
        shaders::ATTR_THREE_NORMAL,
        shaders::ATTR_THREE_UV,
        shaders::ATTR_THREE_COLOR0,
        shaders::ATTR_THREE_UV1,
        -1,
        shaders::ATTR_THREE_INST0,
        shaders::ATTR_THREE_INST1,
        shaders::ATTR_THREE_INST2,
        shaders::ATTR_THREE_HEIGHT,
        shaders::ATTR_THREE_INSTANCE_COLOR,
        -1
    }), .depth.pixel_format = NONE, .colors[0].pixel_format = format, .sample_count = 1, .label = "drie-bake", });
    gen.pipelines[key] = pip;
    return pip;
}

// Runs the seam-dilation fullscreen pass (shaders/lightmap.glsl) from `prev` into the active bake target.
fn void draw_bake_seams(SgPixelFormat format, SgView prev, int width, int height) @local {
    if (!state.seam_shader.id) state.seam_shader = make_program(shaders::lm_seam_shader_desc(sg::query_backend()));
    SgPipeline* pip = &state.seam_pipelines[format == RGBA32F ? 1 : 0];
    if (!pip.id) {
        *pip = sg::make_pipeline(
            &&{
                .shader = state.seam_shader,
                .depth.pixel_format = NONE,
                .colors[0].pixel_format = format,
                .sample_count = 1,
                .label = "drie-bake-seams",
            }
        );
    }
    sg::apply_pipeline(*pip);
    SgBindings bind;
    bind.views[shaders::VIEW_LM_PREV] = prev;
    bind.samplers[shaders::SMP_LM_SMP] = clamp_sampler();
    sg::apply_bindings(&bind);
    shaders::LmParams p = { .lm_size = { 1.0f / width, 1.0f / height, 0.5f / width, 0 } };
    sg::apply_uniforms(shaders::UB_LM_PARAMS, &&{ .ptr = &p, .size = shaders::LmParams::size });
    sg::draw(0, 3, 1);
}

// Collects all visible lights in the subtree rooted at `obj` into `state.lights`.
fn void collect_lights(Object3D* obj) @local {
    if (!obj.visible) return;
    if (obj.kind == LIGHT) state.lights.push((Light*)obj);
    foreach (c : obj.children) collect_lights(c);
}

// Runs one progressive lightmap accumulation pass (used by ProgressiveLightMap). Draws each mesh
// in `objects` into `target` (RGBA32F or RGBA16F, no depth) at its UV1 coordinates using `material`,
// lit by `lights`, shadowed by `objects`, and blended into `prev` with weight `1 / blend_window`.
// When `seams` is true, pre-fills the target by dilating `prev` across UV island borders.
// Leaves `info`, `time`, motion vectors, and reflections untouched; call before render() without
// a separate commit().
fn void Renderer.render_bake(
    &self,
    Scene* lights,
    Camera* camera,
    Object3D*[] objects,
    Material* material,
    RenderTarget* target,
    SgView prev,
    float blend_window = 100,
    bool seams = true
) {
    if (!state.gfx_ready) return;
    RenderInfo info                                   = self.info;
    bool motion                                       = state.motion;
    Scene* scene                                      = state.scene;
    IrradianceVolume*[MAX_IRRADIANCE_VOLUMES] volumes = state.gi_volumes;
    int volume_count                                  = state.gi_count;
    IrradianceVolume* volume                          = state.irradiance;
    SgView gi_view                                    = state.gi_view;
    SgView radiance                                   = state.radiance_view;
    SgView fog                                        = state.fog_view;
    SgView probes                                     = state.probes_view;
    SgSampler probes_sampler                          = state.probes_sampler;
    DirectionalLight* sun                             = state.sun_caster;
    state.motion = false;

    if (!camera.custom_projection) camera.update_projection_matrix();
    if (!camera.parent) camera.update_matrix_world();
    camera.matrix_world_inverse = camera.matrix_world.invert();
    if (!lights.no_matrix_world_update) lights.update_matrix_world();
    state.lights.clear();
    collect_lights(lights);
    state.bake = { .on = true, .prev = prev, .width = target.width, .height = target.height };
    state.bake.format = target.pixel_format();
    state.caster_roots = objects;
    state.sun_caster = null;
    cull_begin(false, false);
    prepare_and_draw_shadows(lights, camera, true);

    use_irradiance(lights);
    if (!state.no_radiance) state.no_radiance = storage_buffer(6 * 16, label: "drie-rc-none");
    if (lights.radiance) lights.radiance.upload();
    state.radiance_view = lights.radiance ? lights.radiance.irradiance_buf.view : state.no_radiance.view;
    state.scene = lights;
    FsParams fs = build_light_params(lights, camera, null);
    // Output raw linear light with cull box disabled.
    fs.output_params = { 1, 0, 0, 0 };
    fs.cull_min.w = 0;
    fs.bake_params = { 1, 1 / math::max(blend_window, 1e-6f), 0, 0 };
    state.fog_view = ready(state.white).view;

    profile::begin_pass(
        &&{
            .action = { .colors[0] = { .load_action = CLEAR, .store_action = STORE, .clear_value = { 0, 0, 0, 1 } } },
            .attachments = target.attachments(),
            .label = "drie-bake",
        }
    );
    if (seams) draw_bake_seams(target.pixel_format(), prev, target.width, target.height);
    foreach (o : objects) {
        if (o.kind != MESH || !o.visible) continue;
        Mesh* m = (Mesh*)o;
        if (!m.geometry || m.is_line()) continue;
        BufferGeometry* g = m.geometry;
        if (!g.uploaded && g.indices.len() == 0) continue;
        g.upload();
        if (m.instance_data.len() > 0) upload_instances_gpu(m);
        Material* own = m.material;
        m.material = material;
        DrawItem item = { .mesh = m };
        draw_item(&item, mat4_identity(), &fs, SOLID);
        m.material = own;
    }
    sg::end_pass();

    state.bake = { };
    state.caster_roots = { };
    state.sun_caster = sun;
    state.motion = motion;
    state.scene = scene;
    state.irradiance = volume;
    state.gi_volumes = volumes;
    state.gi_count = volume_count;
    state.gi_view = gi_view;
    state.radiance_view = radiance;
    state.fog_view = fog;
    state.probes_view = probes;
    state.probes_sampler = probes_sampler;
    self.info = info;
}

// ---- Irradiance volume baking

// Callback invoked per cubemap face in capture_environment() after orienting the camera.
alias EnvFaceFn = fn void(Camera* camera);
// Ray-distance callback for capture_environment(), called per output texel in top-to-bottom row order.
alias EnvDistanceFn = fn float(Vector3 origin, Vector3 dir);

// Bakes directional irradiance for every probe in `v` by rendering 6 linear HDR cube faces
// (`face_size` pixels each, `near`..`far`) per probe and integrating cosine-weighted radiance
// along +/-X, +/-Y, +/-Z. With `bounces > 0`, runs extra passes with `v` illuminating the scene
// (the initial direct pass excludes `v` even if attached to `scene`).
fn void Renderer.bake_irradiance(
    &self,
    Scene* scene,
    IrradianceVolume* v,
    int face_size = 8,
    float near = 0.1,
    float far = 100,
    int bounces = 0
) {
    Vector3[6] fwd         = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
    Vector3[6] ups         = { { 0, 1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 0 } };
    int fs                 = math::max(face_size, 1);
    // Pack all six linear HDR faces side by side into a single strip for one readback per probe.
    RenderTarget* rt       = render_target(fs * 6, fs, 1, SgPixelFormat.RGBA16F);
    PerspectiveCamera* cam = perspective_camera(90, 1, near, far);
    ushort[] pixels        = mem::new_array(ushort, (sz)fs * 6 * fs * 4);
    float[] baked          = mem::new_array(float, v.data.len);
    // Precompute unit direction and solid angle per cubemap texel.
    Vector3[] dirs         = mem::new_array(Vector3, (sz)fs * fs * 6);
    float[] solid          = mem::new_array(float, (sz)fs * fs);
    defer {
        free(pixels);
        free(baked);
        free(dirs);
        free(solid);
        rt.dispose();
        free(rt);
        free(cam);
    }
    for (int i = 0; i < 6; i++) {
        Vector3 right = fwd[i].cross(ups[i]);
        for (int sy = 0; sy < fs; sy++) {
            for (int sx = 0; sx < fs; sx++) {
                float fx  = ((float)sx + 0.5f) / fs * 2 - 1;
                // Row 0 of read_image() is the top of the face.
                float fy  = 1 - ((float)sy + 0.5f) / fs * 2;
                Vector3 d = fwd[i] + right * fx + ups[i] * fy;
                dirs[((sz)i * fs + sy) * fs + sx] = d.normalize();
                float r2 = 1 + fx * fx + fy * fy;
                if (i == 0) solid[(sz)sy * fs + sx] = (4.0f / (float)(fs * fs)) / (r2 * math::sqrt(r2));
            }
        }
    }

    RenderTarget* prev_target = self.render_target;
    bool prev_auto_aspect     = self.auto_aspect;
    bool prev_auto_clear      = self.auto_clear;
    bool prev_scissor_test    = self.scissor_test;
    int[4] prev_viewport      = self.viewport;
    int[4] prev_scissor       = self.scissor;
    self.auto_aspect = false;
    self.auto_clear = true;
    self.scissor_test = true;
    for (int pass = 0; pass <= bounces; pass++) {
        state.gi_skip = pass == 0 ? v : null;
        state.gi_add = pass == 0 ? null : v;
        for (int z = 0; z < v.nz; z++) {
            for (int y = 0; y < v.ny; y++) {
                for (int x = 0; x < v.nx; x++) {
                    Vector3 position = v.probe_position(x, y, z);
                    self.set_render_target(rt);
                    for (int i = 0; i < 6; i++) {
                        self.viewport = { i * fs, 0, fs, fs };
                        self.scissor = self.viewport;
                        cam.position = position;
                        cam.up = ups[i];
                        cam.update_matrix_world();
                        cam.look_at_vec(position + fwd[i]);
                        cam.update_matrix_world();
                        self.render(scene, cam);
                    }
                    commit();
                    readback::read_image(rt.texture_image, fs * 6, fs, (char*)pixels.ptr, 8);
                    Vector3[6] sum;
                    for (int i = 0; i < 6; i++) {
                        for (int sy = 0; sy < fs; sy++) {
                            for (int sx = 0; sx < fs; sx++) {
                                sz p = ((sz)sy * fs * 6 + (sz)i * fs + sx) * 4;
                                Vector3 light = {
                                    half_to_float(pixels[p]),
                                    half_to_float(pixels[p + 1]),
                                    half_to_float(pixels[p + 2])
                                };
                                Vector3 d = dirs[((sz)i * fs + sy) * fs + sx];
                                float w   = solid[(sz)sy * fs + sx];
                                // Accumulate cosine-weighted irradiance along each positive/negative axis.
                                for (int a = 0; a < 3; a++) {
                                    float c = d[a];
                                    if (c > 0) sum[a * 2] += light * (c * w);
                                    if (c < 0) sum[a * 2 + 1] += light * (-c * w);
                                }
                            }
                        }
                    }
                    for (int dir = 0; dir < 6; dir++) {
                        sz o = v.offset(x, y, z, dir);
                        baked[o] = sum[dir].x;
                        baked[o + 1] = sum[dir].y;
                        baked[o + 2] = sum[dir].z;
                    }
                }
            }
        }
        v.data[..] = baked[..];
        v.uploaded = false;
    }
    state.gi_skip = null;
    state.gi_add = null;
    self.set_render_target(prev_target);
    self.auto_aspect = prev_auto_aspect;
    self.auto_clear = prev_auto_clear;
    self.scissor_test = prev_scissor_test;
    self.viewport = prev_viewport;
    self.scissor = prev_scissor;
}

// ---- Environment map and probe capture

// Renders `scene` into 6 cubemap faces from `position`, reads them back, and resamples into an
// equirectangular Texture (`width` x `width / 2`) for Scene.environment or Material.env_map.
// With `hdr`, captures linear RGBA16F radiance into a float texture. When `distance` is given,
// stores normalized ray hit distance (clamped to `distance_range`) in alpha for parallax-corrected
// reflections. Synchronous GPU readback; call at load time rather than every frame.
fn Texture* Renderer.capture_environment(
    &self,
    Scene* scene,
    Vector3 position,
    int face_size = 256,
    int width = 1024,
    float near = 0.05,
    float far = 100,
    EnvFaceFn on_face = null,
    EnvDistanceFn distance = null,
    float distance_range = 16,
    bool hdr = false
) {
    Vector3[6] fwd = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
    Vector3[6] ups = { { 0, 1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 0 } };
    Vector3[6] rights;
    RenderTarget* rt = render_target(face_size, face_size, 1, hdr ? SgPixelFormat.RGBA16F : SgPixelFormat.RGBA8);
    int pixel_bytes = hdr ? 8 : 4;
    PerspectiveCamera* cam = perspective_camera(90, 1, near, far);
    RenderTarget* prev_target = self.render_target;
    bool prev_auto = self.auto_aspect;
    self.auto_aspect = false;
    usz face_bytes = (usz)face_size * (usz)face_size * (usz)pixel_bytes;
    char[] faces   = mem::new_array(char, (sz)face_bytes * 6);
    defer free(faces);
    for (int i = 0; i < 6; i++) {
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
        readback::read_image(rt.texture_image, face_size, face_size, &faces[face_bytes * (usz)i], pixel_bytes);
    }
    self.set_render_target(prev_target);
    self.auto_aspect = prev_auto;
    rt.dispose();
    free(rt);
    free(cam);

    // Resample the six cube faces into equirectangular longitude/latitude layout.
    int height = width / 2;
    char[] px;
    float[] fpx;
    if (hdr) {
        fpx = mem::new_array(float, (sz)width * height * 4);
    } else {
        px = mem::new_array(char, (sz)width * height * 4);
    }
    defer {
        free(px);
        free(fpx);
    }
    for (int y = 0; y < height; y++) {
        float phi = ((float)y + 0.5f) / height * PI;
        for (int x = 0; x < width; x++) {
            float theta    =  (((float)x + 0.5f) / width - 0.5f) * 2 * PI;
            Vector3 d      =  { math::cos(theta) * math::sin(phi), math::cos(phi), math::sin(theta) * math::sin(phi) };
            int best;
            float best_dot = -2;
            for (int i = 0; i < 6; i++) {
                float dd = d.dot(fwd[i]);
                if (dd > best_dot) {
                    best_dot = dd;
                    best = i;
                }
            }
            float fx       = d.dot(rights[best]) / best_dot;
            float fy       = d.dot(ups[best]) / best_dot;
            int sx         = math::clamp((int)((fx * 0.5f + 0.5f) * face_size), 0, face_size - 1);
            int sy         = math::clamp((int)((0.5f - fy * 0.5f) * face_size), 0, face_size - 1);
            usz src        = face_bytes * (usz)best + ((usz)sy * (usz)face_size + (usz)sx) * (usz)pixel_bytes;
            usz dst        = ((usz)y * (usz)width + (usz)x) * 4;
            float far_part = distance ? math::clamp(distance(position, d) / distance_range, 0.0f, 1.0f) : 1;
            if (hdr) {
                ushort* h = (ushort*)&faces[src];
                for (usz c = 0; c < 3; c++) fpx[dst + c] = half_to_float(h[c]);
                fpx[dst + 3] = far_part;
            } else {
                px[dst:3] = faces[src:3];
                px[dst + 3] = (char)(far_part * 255 + 0.5f);
            }
        }
    }
    Texture* map = hdr ? float_texture(fpx, width, height) : data_texture(px, width, height);
    if (distance) map.env_distance_range = distance_range;
    return map;
}

// Persistent RGBA16F mipmapped render target backing a Renderer.render_environment() texture.
struct EnvTarget @private {
    Texture* texture;
    RenderTarget* rt;
    int levels;
    SgView[12] level_views;
    SgView[12] attachments;
}

// Renders `material` onto a directional grid directly into a mipmapped RGBA16F equirectangular
// environment Texture (`width` x `width / 2`) on the GPU; node graphs read the unit ray direction
// from `normal_geometry`. Pass an existing texture in `into` to update it in place. With `diffuse`,
// also renders a 64x32 pass and reads it back synchronously to compute `Texture.env_light` SH.
fn Texture* Renderer.render_environment(
    &self,
    Material* material,
    Texture* into = null,
    int width = 512,
    bool diffuse = true
) {
    width = math::max(width & ~1, 8);
    int height = width / 2;
    if (!state.env_scene) {
        const int COLS    = 128;
        const int ROWS    = 64;
        // Align row 0 (zenith, phi = 0) with top of texture memory on [0,1] depth backends.
        float up          = depth_zero_to_one() ? 1 : -1;
        BufferGeometry* g = mem::new(BufferGeometry);
        for (int j = 0; j <= ROWS; j++) {
            for (int i = 0; i <= COLS; i++) {
                float u     = (float)i / COLS;
                float v     = (float)j / ROWS;
                float theta = (u - 0.5f) * 2 * PI;
                float phi   = v * PI;
                Vector3 d   = { math::cos(theta) * math::sin(phi), math::cos(phi), math::sin(theta) * math::sin(phi) };
                g.add_vertex({ u * 2 - 1, up * (1 - 2 * v), 0 }, d, { u, 1 - v });
            }
        }
        for (uint j = 0; j < ROWS; j++) {
            for (uint i = 0; i < COLS; i++) {
                uint a = j * (COLS + 1) + i;
                uint c = a + COLS + 1;
                // Keep CCW winding facing the orthographic camera after Y orientation flip.
                if (up > 0) {
                    g.add_triangle(a, c, a + 1);
                    g.add_triangle(a + 1, c, c + 1);
                } else {
                    g.add_triangle(a, a + 1, c);
                    g.add_triangle(a + 1, c + 1, c);
                }
            }
        }
        state.env_scene = scene();
        state.env_grid = mesh(g, material);
        state.env_grid.frustum_culled = false;
        state.env_grid.receive_shadow = false;
        state.env_scene.add(state.env_grid);
        state.env_camera = orthographic_camera(-1, 1, 1, -1, 0.1, 10);
        state.env_camera.position = { 0, 0, 1 };
        state.env_camera.update_matrix_world();
        state.env_small = render_target(64, 32, 1, RGBA16F, depth_buffer: false);
    }
    state.env_grid.material = material;

    EnvTarget* et;
    if (into) {
        foreach (&t : state.env_targets) {
            if (t.texture == into) et = t;
        }
        if (!et) abort("drie: render_environment into a map it didn't draw");
    }
    if (et && (et.texture.width != width || et.texture.height != height)) {
        et.rt.dispose();
        free(et.rt);
        for (int i = 0; i < et.levels; i++) {
            sg::destroy_view(et.level_views[i]);
            sg::destroy_view(et.attachments[i]);
        }
        sg::destroy_view(et.texture.view);
        sg::destroy_image(et.texture.image);
        sg::destroy_sampler(et.texture.sampler);
        et.texture.uploaded = false;
    }
    if (!et) {
        Texture* t = mem::new(Texture);
        *t = {
            .color_space = NO_COLOR_SPACE,
            .repeat = { 1, 1 },
            .generate_mipmaps = true,
            .min_filter = LINEAR_MIPMAP_LINEAR
        };
        state.env_targets.push({ .texture = t });
        et = &state.env_targets[state.env_targets.len() - 1];
    }
    Texture* t = et.texture;
    if (!t.uploaded) {
        int levels = 1;
        while (levels < 12 && width >> levels > 0) levels++;
        t.width = width;
        t.height = height;
        t.image = sg::make_image(
            &&{
                .usage = { .color_attachment = true },
                .width = width,
                .height = height,
                .num_mipmaps = levels,
                .sample_count = 1,
                .pixel_format = RGBA16F,
                .label = "drie-environment",
            }
        );
        t.view = sg::make_view(&&{ .texture = { .image = t.image } });
        t.sampler = sg::make_sampler(
            &&{
                .min_filter = LINEAR,
                .mag_filter = LINEAR,
                .mipmap_filter = LINEAR,
                .wrap_u = REPEAT,
                .wrap_v = CLAMP_TO_EDGE,
            }
        );
        for (int i = 0; i < levels; i++) {
            et.level_views[i] = sg::make_view(
                &&{ .texture = { .image = t.image, .mip_levels = { .base = i, .count = 1 } } }
            );
            et.attachments[i] = sg::make_view(&&{ .color_attachment = { .image = t.image, .mip_level = i } });
        }
        et.levels = levels;
        et.rt = render_target(width, height, 1, RGBA16F, depth_buffer: false);
        t.uploaded = true;
    }

    RenderTarget* prev_target = self.render_target;
    bool prev_auto            = self.auto_aspect;
    bool prev_auto_clear      = self.auto_clear;
    bool prev_scissor_test    = self.scissor_test;
    int[4] prev_viewport      = self.viewport;
    self.auto_aspect = false;
    self.auto_clear = true;
    self.scissor_test = false;
    self.viewport = { };
    self.set_render_target(et.rt);
    self.render(state.env_scene, state.env_camera);
    fill_levels(et.rt.texture, et.level_views[:et.levels], et.attachments[:et.levels]);
    if (diffuse) {
        self.set_render_target(state.env_small);
        self.render(state.env_scene, state.env_camera);
        commit();
        ushort[] halves = mem::new_array(ushort, 64 * 32 * 4);
        defer free(halves);
        readback::read_image(state.env_small.texture_image, 64, 32, (char*)halves.ptr, 8);
        float[] floats = mem::new_array(float, 64 * 32 * 4);
        defer free(floats);
        foreach (i, h : halves) floats[i] = half_to_float(h);
        t.env_light = env_light(floats, { }, 64, 32, false);
        t.has_env_sh = true;
    }
    self.set_render_target(prev_target);
    self.auto_aspect = prev_auto;
    self.auto_clear = prev_auto_clear;
    self.scissor_test = prev_scissor_test;
    self.viewport = prev_viewport;
    return t;
}

// Forward directions (+X, -X, +Y, -Y, +Z, -Z) for render_probe() cubemap capture.
const Vector3[6] PROBE_AHEAD = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
// Up vectors matching PROBE_AHEAD for render_probe() cubemap capture.
const Vector3[6] PROBE_UP    = { { 0, 1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 }, { 0, 1, 0 }, { 0, 1, 0 } };

// Renders `scene` into 6 linear RGBA16F cubemap faces (`face_size` px, 4x MSAA) from `position`
// and resamples them on the GPU into an equirectangular environment map (`4 * face_size` wide).
// Pass an existing texture in `into` to update it in place; `diffuse` triggers SH readback.
fn Texture* Renderer.render_probe(
    &self,
    Scene* scene,
    Vector3 position,
    Texture* into = null,
    int face_size = 256,
    float near = 0.1,
    float far = 1000,
    bool diffuse = true
) {
    if (!state.probe_camera) {
        state.probe_camera = perspective_camera(90, 1, near, far);
        for (int i = 0; i < 6; i++) state.probe_faces[i] = render_target(face_size, face_size, 4, RGBA16F);
        state.probe_resample = probe_resample();
    }
    PerspectiveCamera* cam = state.probe_camera;
    cam.near = near;
    cam.far = far;
    cam.update_projection_matrix();
    RenderTarget* prev_target = self.render_target;
    bool prev_auto            = self.auto_aspect;
    bool prev_auto_clear      = self.auto_clear;
    bool prev_scissor_test    = self.scissor_test;
    int[4] prev_viewport      = self.viewport;
    self.auto_aspect = false;
    self.auto_clear = true;
    self.scissor_test = false;
    self.viewport = { };
    for (int i = 0; i < 6; i++) {
        RenderTarget* face = state.probe_faces[i];
        face.set_size(face_size, face_size, 4);
        face.as_map();
        cam.position = position;
        cam.up = PROBE_UP[i];
        cam.update_matrix_world();
        cam.look_at_vec(position + PROBE_AHEAD[i]);
        cam.update_matrix_world();
        self.set_render_target(face);
        self.render(scene, cam);
    }
    self.set_render_target(prev_target);
    self.auto_aspect = prev_auto;
    self.auto_clear = prev_auto_clear;
    self.scissor_test = prev_scissor_test;
    self.viewport = prev_viewport;
    return self.render_environment(state.probe_resample, into, 4 * face_size, diffuse);
}

fn Node vec3_node(Vector3 v) @local => node_const(VEC3, { v.x, v.y, v.z, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

// Builds the node material that selects the dominant cubemap face from `normal_geometry` and
// samples `state.probe_faces` at mip 0 for render_probe().
fn MeshBasicMaterial* probe_resample() @private {
    Node d     = nodes::normal_geometry();
    Node a     = node_call(ABS, d);
    Node zero  = nodes::num(0);
    float flip = depth_zero_to_one() ? 1 : -1;
    Node[6] seen;
    for (int i = 0; i < 6; i++) {
        Vector3 ahead = PROBE_AHEAD[i];
        Vector3 up    = PROBE_UP[i];
        Vector3 right = ahead.cross(up);
        Node along    = node_call(MAX, node_call(DOT, d, vec3_node(ahead)), nodes::num(1e-4f));
        Node x        = node_call(DOT, d, vec3_node(right)) / along;
        Node y        = node_call(DOT, d, vec3_node(up)) / along;
        Node uv       = node_join({ x * 0.5f + 0.5f, y * (-0.5f * flip) + 0.5f }, VEC2);
        seen[i] = nodes::texture(state.probe_faces[i].as_map(), uv, zero).xyz();
    }
    Node along_x         = node_binary(GE, a.x(), node_call(MAX, a.y(), a.z()));
    Node along_y         = node_binary(GE, a.y(), a.z());
    Node on_x            = nodes::select(node_binary(GT, d.x(), zero), seen[0], seen[1]);
    Node on_y            = nodes::select(node_binary(GT, d.y(), zero), seen[2], seen[3]);
    Node on_z            = nodes::select(node_binary(GT, d.z(), zero), seen[4], seen[5]);
    MeshBasicMaterial* m = mesh_basic_material(fog: false);
    m.color_node = nodes::select(along_x, on_x, nodes::select(along_y, on_y, on_z));
    return m;
}