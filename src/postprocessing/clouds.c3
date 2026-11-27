// Volumetric clouds (`CloudPass`, `shaders/clouds.glsl`), after Schneider's Nubis talks on Horizon
// Zero Dawn (2015, 2017) and Forbidden West (2022, 2023): a layer of clouds between two altitudes
// around a curved earth, which the camera can fly into and above. A weather map (coverage, cloud
// type, and its distance field) shapes them, two tiling 3D noises (Perlin-Worley and Worley, made
// on the CPU once) erode them, and each pixel's ray is marched from the camera, its first step
// pushed along by blue noise. Two marches: what is within `near_distance` with coarser pixels and
// finer steps, and the rest. The two are put together and blended into the last frames' along the
// camera's motion (clamped to the new frame's neighbours), then drawn over the picture by depth,
// through the froxel fog in front of them (`Renderer.fog_info`). Their shadow (straight under
// them, towards the sun) darkens the sun on lit surfaces and in the volumetric fog
// (`Renderer.cloud_shadow`).
//
//   CloudPass* clouds = postprocessing::cloud_pass(renderer);
//   // each frame:
//   clouds.sun_direction = sky.sun_position;
//   clouds.prepare(camera);          // their shadow, for the render
//   renderer.set_render_target(scene_rt);
//   renderer.render(scene, camera);
//   clouds.render(scene_rt, camera); // over it (a single-sampled target with a depth buffer)
module drie::postprocessing;
import std::math;
import sokol, drie;

const int CLOUD_NOISE   = 64;  // the shape noise's texels a side
const int CLOUD_DETAIL  = 32;  // the detail noise's
const int CLOUD_WEATHER = 512; // the weather map's

struct CloudPass {
    Renderer* renderer;
    // ---- The layer
    float bottom; // altitude of the clouds' bottom (m; default 1500)
    float top; // and top (m; default 4000)
    float earth_radius; // the earth's radius (m; its middle is (0, -earth_radius, 0))
    float coverage; // how much of the sky they cover, 0..1 (default 0.5)
    float density; // density inside them (default 3)
    float extinction; // light lost per meter at density 1 (1/m; default 0.02)
    float weather_scale; // meters the weather map spans before it repeats (default 30000)
    float shape_scale; // meters the shape noise spans (default 4500)
    float detail_scale; // meters the detail noise spans (default 600)
    float detail_erosion; // how much the detail noise eats into their edges, 0..1 (default 0.85)
    Vector2 wind; // drift in x and z (m/s), times `time`
    float time; // seconds, for the wind
    // ---- Their light
    Vector3 sun_direction; // towards the sun; need not be normalized
    Vector3 sun_color; // the sun's light on them (linear)
    Vector3 ambient_top; // the sky's light on their tops (linear)
    Vector3 ambient_bottom; // the light on their bottoms, from the ground (linear)
    float forward_scattering; // Henyey-Greenstein g of the main lobe (default 0.75)
    Vector3 haze_color; // the air's color far away (linear): they fade into it with distance
    float haze_density; // and how fast (1/m; default 1/60000)
    // ---- Marching
    float resolution; // the far march's share of the picture's width and height (default 0.5)
    float near_resolution; // the near march's (default 0.25)
    float near_distance; // where the near march ends and the far starts (m; default 250)
    float near_step; // the shortest step (m; default 4)
    float step_growth; // steps grow as sqrt(distance) times this (default 0.6)
    float max_distance; // marched no further than this (m; default 60000)
    int view_steps; // the most steps along a view ray (default 128)
    int light_steps; // steps towards the sun from each (default 6)
    float history; // the new frame's share of each pixel (default 0.1; 1: no blending over time)
    // ---- Their shadow
    bool cast_shadow; // on Renderer.cloud_shadow, for the sun on surfaces and in the fog (default on)
    float shadow_span; // meters the shadow map spans, around the camera (default 20000)
    int shadow_size; // its texels a side (default 512)

    // ---- GPU
    SgView noise;
    SgView detail;
    SgImage weather;
    SgView weather_view;
    char[] weather_px; // r = coverage, g = type, b = distance to cloud (filled for `coverage`)
    float weather_coverage; // the coverage the distance field is for
    SgSampler repeat;
    SgSampler point;
    SgShader shadow_shader;
    SgPipeline shadow_pipeline;
    SgShader march_shader;
    SgPipeline march_pipeline;
    SgShader resolve_shader;
    SgPipeline resolve_pipeline;
    ShaderPass composite;
    CloudImage shadow;
    CloudImage far_color;
    CloudImage far_depth;
    CloudImage near_color;
    CloudImage near_depth;
    CloudImage[2] resolved;
    CloudImage resolved_depth;
    int cur;
    int width; // the picture's size the images are made for
    int height;
    bool has_history;
    drie::Mat4 prev_view_proj;
    uint frame;
}

// A storage image with a texture view of it.
struct CloudImage {
    SgImage image;
    SgView storage;
    SgView texture;
    int width;
    int height;
}

// Clouds over a single-sampled target: CloudPass.prepare before each render() for their shadow,
// CloudPass.render after it.
fn CloudPass* cloud_pass(Renderer* renderer) {
    CloudPass* c = mem::new(CloudPass);
    *c = {
        .renderer = renderer,
        .bottom = 1500,
        .top = 4000,
        .earth_radius = 6360000,
        .coverage = 0.5f,
        .density = 3,
        .extinction = 0.02f,
        .weather_scale = 50000,
        .shape_scale = 4500,
        .detail_scale = 600,
        .detail_erosion = 0.85f,
        .wind = { 10, 3 },
        .sun_direction = { 0.4f, 0.5f, -0.6f },
        .sun_color = { 6, 5.6f, 5 },
        .ambient_top = { 0.55f, 0.7f, 0.95f },
        .ambient_bottom = { 0.25f, 0.27f, 0.3f },
        .forward_scattering = 0.75f,
        .haze_color = { 0.55f, 0.65f, 0.8f },
        .haze_density = 1.0f / 60000,
        .resolution = 0.5f,
        .near_resolution = 0.25f,
        .near_distance = 250,
        .near_step = 4,
        .step_growth = 0.6f,
        .max_distance = 60000,
        .view_steps = 128,
        .light_steps = 6,
        .history = 0.1f,
        .cast_shadow = true,
        .shadow_span = 20000,
        .shadow_size = 512,
        .weather_coverage = -1,
    };
    SgBackend backend = sg::query_backend();
    c.shadow_shader = drie::make_program(cloud_shadow_shader_desc(backend));
    c.shadow_pipeline = sg::make_pipeline(
        &&{ .compute = true, .shader = c.shadow_shader, .label = "drie-cloud-shadow" }
    );
    c.march_shader = drie::make_program(cloud_march_shader_desc(backend));
    c.march_pipeline = sg::make_pipeline(&&{ .compute = true, .shader = c.march_shader, .label = "drie-cloud-march" });
    c.resolve_shader = drie::make_program(cloud_resolve_shader_desc(backend));
    c.resolve_pipeline = sg::make_pipeline(
        &&{ .compute = true, .shader = c.resolve_shader, .label = "drie-cloud-resolve" }
    );
    // over the picture: what it lets through of it, and its own light
    SgBlendState over = {
        .enabled = true,
        .src_factor_rgb = ONE,
        .dst_factor_rgb = SRC_ALPHA,
        .src_factor_alpha = ZERO,
        .dst_factor_alpha = ONE,
    };
    c.composite.init(cloud_composite_shader_desc(backend), "drie-clouds", over);
    c.repeat = sg::make_sampler(
        &&{
            .min_filter = LINEAR,
            .mag_filter = LINEAR,
            .wrap_u = REPEAT,
            .wrap_v = REPEAT,
            .wrap_w = REPEAT,
            .label = "drie-clouds-repeat",
        }
    );
    c.point = nearest_sampler();
    c.make_noise();
    c.weather_px = mem::new_array(char, (sz)CLOUD_WEATHER * CLOUD_WEATHER * 4);
    make_weather(c.weather_px);
    return c;
}

// ---- Each frame

// The clouds' shadow, for the render that follows (Renderer.cloud_shadow); call before it.
fn void CloudPass.prepare(&self, Camera* camera) {
    self.update_weather();
    Renderer* r = self.renderer;
    if (!self.cast_shadow) {
        r.cloud_shadow = { };
        r.cloud_shadow_params = { };
        return;
    }
    int n = math::max(self.shadow_size, 16);
    self.shadow.ensure(n, n, "drie-cloud-shadow");
    // the middle follows the camera a texel at a time, so the shadow doesn't crawl
    Vector3 eye   = camera.matrix_world.get_position();
    float texel   = self.shadow_span / n;
    float mx      = math::floor(eye.x / texel) * texel;
    float mz      = math::floor(eye.z / texel) * texel;
    CloudParams p = self.params(camera, n, n, 0, 0, null);
    p.cl_shadow = { mx, 1 / self.shadow_span, mz, 0 };
    SgBindings b;
    b.views[VIEW_CL_NOISE] = self.noise;
    b.views[VIEW_CL_DETAIL] = self.detail;
    b.views[VIEW_CL_WEATHER_MAP] = self.weather_view;
    b.views[VIEW_CL_SHADOW_OUT] = self.shadow.storage;
    b.samplers[SMP_CL_SMP] = self.repeat;
    profile::begin_pass(&&{ .compute = true, .label = "drie-cloud-shadow" });
    sg::apply_pipeline(self.shadow_pipeline);
    sg::apply_bindings(&b);
    sg::apply_uniforms(UB_CLOUD_PARAMS, &&{ .ptr = &p, .size = CloudParams::size });
    sg::dispatch((n + 7) / 8, (n + 7) / 8, 1);
    sg::end_pass();
    r.cloud_shadow = self.shadow.texture;
    r.cloud_shadow_params = { mx, 1 / self.shadow_span, mz, self.bottom };
}

// Marches the clouds for `camera` and draws them over `target` (single-sampled, with a depth
// buffer: the scene `camera` has just rendered into it).
fn void CloudPass.render(&self, RenderTarget* target, Camera* camera) {
    if (!target.depth_buffer || target.samples > 1) {
        abort("drie: CloudPass needs a single-sampled target with a depth buffer");
    }
    self.update_weather();
    int w = target.width;
    int h = target.height;
    if (w != self.width || h != self.height) {
        self.width = w;
        self.height = h;
        self.has_history = false;
    }
    int fw = math::max((int)math::ceil(w * self.resolution), 1);
    int fh = math::max((int)math::ceil(h * self.resolution), 1);
    int nw = math::max((int)math::ceil(w * self.near_resolution), 1);
    int nh = math::max((int)math::ceil(h * self.near_resolution), 1);
    self.far_color.ensure(fw, fh, "drie-cloud-far");
    self.far_depth.ensure(fw, fh, "drie-cloud-far-depth");
    self.near_color.ensure(nw, nh, "drie-cloud-near");
    self.near_depth.ensure(nw, nh, "drie-cloud-near-depth");
    for (int i = 0; i < 2; i++) {
        if (self.resolved[i].width != fw || self.resolved[i].height != fh) self.has_history = false;
        self.resolved[i].ensure(fw, fh, "drie-cloud-resolved");
    }
    self.resolved_depth.ensure(fw, fh, "drie-cloud-resolved-depth");
    self.frame++;

    // the two marches: near, then far
    float near_end = math::max(self.near_distance, 0.0f);
    self.march(camera, target, &self.near_color, &self.near_depth, 0, near_end, self.near_step * 0.5f);
    self.march(camera, target, &self.far_color, &self.far_depth, near_end, 1e9, self.near_step);

    // put together and blended over time
    Rays rays = view_rays(camera);
    bool flip = drie::depth_zero_to_one();
    CloudResolveParams rp = {
        .cr_size = { fw, fh, self.has_history ? 0 : 1, math::clamp(self.history, 0.01f, 1.0f) },
        .cr_cam = { ...rays.eye, flip ? 1 : 0 },
        .cr_fwd = { ...rays.fwd, 0 },
        .cr_right = { ...rays.right, 0 },
        .cr_up = { ...rays.up, 0 },
        .cr_prev_view_proj = self.prev_view_proj,
    };
    int prev = self.cur;
    self.cur = 1 - self.cur;
    SgBindings b;
    b.views[VIEW_CR_FAR] = self.far_color.texture;
    b.views[VIEW_CR_FAR_DEPTH] = self.far_depth.texture;
    b.views[VIEW_CR_NEAR] = self.near_color.texture;
    b.views[VIEW_CR_NEAR_DEPTH] = self.near_depth.texture;
    b.views[VIEW_CR_HISTORY] = self.resolved[prev].texture;
    b.views[VIEW_CR_OUT] = self.resolved[self.cur].storage;
    b.views[VIEW_CR_DEPTH_OUT] = self.resolved_depth.storage;
    b.samplers[SMP_CR_LINEAR] = linear_sampler();
    b.samplers[SMP_CR_POINT] = self.point;
    profile::begin_pass(&&{ .compute = true, .label = "drie-cloud-resolve" });
    sg::apply_pipeline(self.resolve_pipeline);
    sg::apply_bindings(&b);
    sg::apply_uniforms(UB_CLOUD_RESOLVE_PARAMS, &&{ .ptr = &rp, .size = CloudResolveParams::size });
    sg::dispatch((fw + 7) / 8, (fh + 7) / 8, 1);
    sg::end_pass();
    self.has_history = true;
    self.prev_view_proj = camera.projection_matrix.mul(camera.matrix_world_inverse);

    // over the picture
    FogInfo fog = self.renderer.fog_info;
    CloudCompositeParams cp = {
        .cc_size = { w, h, fw, fh },
        .cc_clip = { camera.near, camera.far, flip ? 1 : 0, fog.view.id ? 1 : 0 },
        .cc_fwd = { ...rays.fwd, 0 },
        .cc_right = { ...rays.right, 0 },
        .cc_up = { ...rays.up, flip ? 1 : 0 },
        .cc_fog_grid = fog.grid,
        .cc_fog_slices = fog.slices,
    };
    self.composite.set_texture(VIEW_CC_DEPTH, target.depth_texture);
    self.composite.set_texture(VIEW_CC_CLOUD, self.resolved[self.cur].texture);
    self.composite.set_texture(VIEW_CC_CLOUD_DEPTH, self.resolved_depth.texture);
    self.composite.set_texture(VIEW_CC_FOG, fog.view.id ? fog.view : self.resolved_depth.texture);
    self.composite.set_sampler(SMP_CC_POINT, self.point);
    self.composite.set_sampler(SMP_CC_LINEAR, linear_sampler());
    self.composite.set_uniforms(UB_CLOUD_COMPOSITE_PARAMS, &cp);
    self.composite.render_over(target);
}

// Frees the GPU resources.
fn void CloudPass.dispose(&self) {
    self.composite.dispose();
    sg::destroy_pipeline(self.shadow_pipeline);
    sg::destroy_pipeline(self.march_pipeline);
    sg::destroy_pipeline(self.resolve_pipeline);
    sg::destroy_shader(self.shadow_shader);
    sg::destroy_shader(self.march_shader);
    sg::destroy_shader(self.resolve_shader);
    self.shadow.free();
    self.far_color.free();
    self.far_depth.free();
    self.near_color.free();
    self.near_depth.free();
    self.resolved[0].free();
    self.resolved[1].free();
    self.resolved_depth.free();
    sg::destroy_view(self.weather_view);
    sg::destroy_image(self.weather);
    sg::destroy_sampler(self.repeat);
    free(self.weather_px);
}

// ---- Inside

struct Rays @local {
    Vector3 eye;
    Vector3 fwd; // through the middle of the picture (with the projection's shift)
    Vector3 right; // right * tan(half the horizontal view)
    Vector3 up; // up * tan(half the vertical view)
}

// The camera's rays, as the volumetric fog has them (update_fog_volume in src/lighting/fog.c3).
fn Rays view_rays(Camera* camera) @local {
    drie::Mat4 cw = camera.matrix_world;
    drie::Mat4 pr = camera.projection_matrix;
    Vector3 fwd   = ((Vector3) { -cw[2][0], -cw[2][1], -cw[2][2] }).normalize();
    Vector3 right = ((Vector3) { cw[0][0], cw[0][1], cw[0][2] }).normalize() / pr[0][0];
    Vector3 up    = ((Vector3) { cw[1][0], cw[1][1], cw[1][2] }).normalize() / pr[1][1];
    return { cw.get_position(), fwd + right * pr[2][0] + up * pr[2][1], right, up };
}

fn CloudParams CloudPass.params(&self, Camera* camera, int w, int h, int full_w, int full_h, RenderTarget* target) {
    Rays rays   = view_rays(camera);
    Vector3 sun = self.sun_direction;
    float len   = math::sqrt(sun.dot(sun));
    sun = len > 0 ? sun / len : (Vector3) { 0, 1, 0 };
    float span = math::max(self.weather_scale, 1.0f);
    return {
        .cl_cam = { ...rays.eye, (float)(self.frame % 1024) },
        .cl_fwd = { ...rays.fwd, self.time },
        .cl_right = { ...rays.right, 0 },
        .cl_up = { ...rays.up, drie::depth_zero_to_one() ? 1 : 0 },
        .cl_layer = { self.bottom, math::max(self.top, self.bottom + 1), self.earth_radius, self.density },
        .cl_shape = { 1 / self.shape_scale, 1 / self.detail_scale, self.detail_erosion, self.coverage },
        .cl_weather = { self.wind.x * self.time, self.wind.y * self.time, 1 / span, 255 * span / CLOUD_WEATHER },
        .cl_sun_dir = { ...sun, self.forward_scattering },
        .cl_sun_color = { ...self.sun_color, self.extinction },
        .cl_amb_top = { ...self.ambient_top, math::clamp(self.light_steps, 1, 16) },
        .cl_amb_bottom = { ...self.ambient_bottom, math::clamp(self.view_steps, 8, 512) },
        .cl_haze = { ...self.haze_color, self.haze_density },
        .cl_size = { w, h, full_w, full_h },
        .cl_range = { 0, 1e9, camera.near, camera.far },
        .cl_shadow = { 0, 0, 0, drie::depth_zero_to_one() ? 1 : 0 },
        .cl_step = { self.near_step, self.step_growth, self.max_distance, 0 },
    };
}

// One march into `color` and `depth`, from `start` to `end` meters along each ray.
fn void CloudPass.march(
    &self,
    Camera* camera,
    RenderTarget* target,
    CloudImage* color,
    CloudImage* depth,
    float start,
    float end,
    float near_step
) {
    CloudParams p = self.params(camera, color.width, color.height, target.width, target.height, target);
    p.cl_range.x = start;
    p.cl_range.y = end;
    p.cl_step.x = math::max(near_step, 0.5f);
    SgBindings b;
    b.views[VIEW_CL_NOISE] = self.noise;
    b.views[VIEW_CL_DETAIL] = self.detail;
    b.views[VIEW_CL_WEATHER_MAP] = self.weather_view;
    b.views[VIEW_CL_DEPTH] = target.depth_texture;
    b.views[VIEW_CL_BLUE] = blue_noise();
    b.views[VIEW_CL_COLOR_OUT] = color.storage;
    b.views[VIEW_CL_DEPTH_OUT] = depth.storage;
    b.samplers[SMP_CL_SMP] = self.repeat;
    b.samplers[SMP_CL_POINT] = self.point;
    profile::begin_pass(&&{ .compute = true, .label = "drie-cloud-march" });
    sg::apply_pipeline(self.march_pipeline);
    sg::apply_bindings(&b);
    sg::apply_uniforms(UB_CLOUD_PARAMS, &&{ .ptr = &p, .size = CloudParams::size });
    sg::dispatch((color.width + 7) / 8, (color.height + 7) / 8, 1);
    sg::end_pass();
}

fn void CloudImage.ensure(&self, int w, int h, ZString label) {
    if (self.image.id && self.width == w && self.height == h) return;
    self.free();
    self.width = w;
    self.height = h;
    self.image = sg::make_image(
        &&{
            .usage = { .storage_image = true },
            .width = w,
            .height = h,
            .pixel_format = RGBA16F,
            .label = label,
        }
    );
    self.storage = sg::make_view(&&{ .storage_image = { .image = self.image } });
    self.texture = sg::make_view(&&{ .texture = { .image = self.image } });
}

fn void CloudImage.free(&self) {
    if (!self.image.id) return;
    sg::destroy_view(self.storage);
    sg::destroy_view(self.texture);
    sg::destroy_image(self.image);
    *self = { };
}

// The weather map's distance field, again when the coverage has changed (and on the first frame).
fn void CloudPass.update_weather(&self) {
    if (self.weather_coverage == self.coverage) return;
    self.weather_coverage = self.coverage;
    weather_distance(self.weather_px, 1 - self.coverage);
    if (self.weather.id) {
        sg::destroy_view(self.weather_view);
        sg::destroy_image(self.weather);
    }
    self.weather = sg::make_image(
        &&{
            .width = CLOUD_WEATHER,
            .height = CLOUD_WEATHER,
            .pixel_format = RGBA8,
            .data = { .ptr = self.weather_px.ptr, .size = self.weather_px.len },
            .label = "drie-cloud-weather",
        }
    );
    self.weather_view = sg::make_view(&&{ .texture = { .image = self.weather } });
}

// ---- Noise, made once

fn uint hash3(int x, int y, int z, uint seed) @local {
    uint h = (uint)x * 0x8da6b343 ^ (uint)y * 0xd8163841 ^ (uint)z * 0xcb1ab31f ^ seed * 0x9e3779b9;
    h ^= h >> 16;
    h *= 0x7feb352d;
    h ^= h >> 15;
    h *= 0x846ca68b;
    h ^= h >> 16;
    return h;
}

fn float unit(uint h) @local => (float)(h & 0xffffff) / 16777216.0f;

// Worley noise that repeats every `period` cells, inverted (1 at the feature points), at p in
// 0..1 of the period.
fn float worley(float px, float py, float pz, int period, uint seed) @local {
    float x    = px * period;
    float y    = py * period;
    float z    = pz * period;
    int cx     = (int)math::floor(x);
    int cy     = (int)math::floor(y);
    int cz     = (int)math::floor(z);
    float best = 1e9;
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int gx   = cx + i;
                int gy   = cy + j;
                int gz   = cz + k;
                uint h   = hash3(
                    (gx % period + period) % period,
                    (gy % period + period) % period,
                    (gz % period + period) % period,
                    seed
                );
                float fx = gx + unit(h);
                float fy = gy + unit(hash3((int)h, 1, 7, seed));
                float fz = gz + unit(hash3((int)h, 3, 11, seed));
                float d  = (fx - x) * (fx - x) + (fy - y) * (fy - y) + (fz - z) * (fz - z);
                if (d < best) best = d;
            }
        }
    }
    return math::clamp(1 - math::sqrt(best), 0.0f, 1.0f);
}

fn float worley_fbm(float x, float y, float z, int period, uint seed) @local {
    return worley(
        x,
        y,
        z,
        period,
        seed
    ) * 0.625f + worley(x, y, z, period * 2, seed + 1) * 0.25f + worley(x, y, z, period * 4, seed + 2) * 0.125f;
}

fn float fade(float t) @local => t * t * t * (t * (t * 6 - 15) + 10);

fn float grad(uint h, float x, float y, float z) @local {
    switch (h % 12) {
        case 0: return x + y;
        case 1: return -x + y;
        case 2: return x - y;
        case 3: return -x - y;
        case 4: return x + z;
        case 5: return -x + z;
        case 6: return x - z;
        case 7: return -x - z;
        case 8: return y + z;
        case 9: return -y + z;
        case 10: return y - z;
        default: return -y - z;
    }
}

// Perlin noise repeating every `period` cells (about -1..1), at p in 0..1 of the period.
fn float perlin(float px, float py, float pz, int period, uint seed) @local {
    float x  = px * period;
    float y  = py * period;
    float z  = pz * period;
    int ix   = (int)math::floor(x);
    int iy   = (int)math::floor(y);
    int iz   = (int)math::floor(z);
    float fx = x - ix;
    float fy = y - iy;
    float fz = z - iz;
    float u  = fade(fx);
    float v  = fade(fy);
    float w  = fade(fz);
    float[8] c;
    for (int k = 0; k < 8; k++) {
        int dx = k & 1;
        int dy = (k >> 1) & 1;
        int dz = (k >> 2) & 1;
        uint h = hash3((ix + dx) % period, (iy + dy) % period, (iz + dz) % period, seed);
        c[k] = grad(h, fx - dx, fy - dy, fz - dz);
    }
    float x00 = c[0] + (c[1] - c[0]) * u;
    float x10 = c[2] + (c[3] - c[2]) * u;
    float x01 = c[4] + (c[5] - c[4]) * u;
    float x11 = c[6] + (c[7] - c[6]) * u;
    float y0  = x00 + (x10 - x00) * v;
    float y1  = x01 + (x11 - x01) * v;
    return y0 + (y1 - y0) * w;
}

fn float perlin_fbm(float x, float y, float z, int period, int octaves, uint seed) @local {
    float sum  = 0;
    float amp  = 1;
    float norm = 0;
    for (int o = 0; o < octaves; o++) {
        sum += perlin(x, y, z, period << o, seed + (uint)o) * amp;
        norm += amp;
        amp *= 0.5f;
    }
    return sum / norm;
}

fn char byte_of(float v) @local => (char)math::clamp((int)(v * 255 + 0.5f), 0, 255);

// The two 3D noises: shape (r = Perlin-Worley, gba = Worley fbm at three frequencies) and detail
// (rgb = Worley fbm at three).
fn void CloudPass.make_noise(&self) {
    usz n     = CLOUD_NOISE;
    char[] px = mem::new_array(char, (sz)(n * n * n * 4));
    for (usz z = 0; z < n; z++) {
        for (usz y = 0; y < n; y++) {
            for (usz x = 0; x < n; x++) {
                float fx = (x + 0.5f) / n;
                float fy = (y + 0.5f) / n;
                float fz = (z + 0.5f) / n;
                float p  = perlin_fbm(fx, fy, fz, 4, 4, 11) * 0.5f + 0.5f;
                float wf = worley_fbm(fx, fy, fz, 4, 21);
                // Perlin-Worley: the Perlin, its lows cut into billows by the Worley
                float pw = math::clamp(wf + (p - 0) / (1 - 0) * (1 - wf), 0.0f, 1.0f);
                pw = math::clamp((pw - 0.25f) / 0.75f, 0.0f, 1.0f);
                usz i = ((z * n + y) * n + x) * 4;
                px[i] = byte_of(pw);
                px[i + 1] = byte_of(worley_fbm(fx, fy, fz, 8, 25));
                px[i + 2] = byte_of(worley_fbm(fx, fy, fz, 16, 31));
                px[i + 3] = byte_of(worley_fbm(fx, fy, fz, 32, 41));
            }
        }
    }
    self.noise = make_3d(px, CLOUD_NOISE, "drie-cloud-noise");
    free(px);
    usz d     = CLOUD_DETAIL;
    char[] dp = mem::new_array(char, (sz)(d * d * d * 4));
    for (usz z = 0; z < d; z++) {
        for (usz y = 0; y < d; y++) {
            for (usz x = 0; x < d; x++) {
                float fx = (x + 0.5f) / d;
                float fy = (y + 0.5f) / d;
                float fz = (z + 0.5f) / d;
                usz i    = ((z * d + y) * d + x) * 4;
                dp[i] = byte_of(worley_fbm(fx, fy, fz, 4, 51));
                dp[i + 1] = byte_of(worley_fbm(fx, fy, fz, 8, 61));
                dp[i + 2] = byte_of(worley_fbm(fx, fy, fz, 16, 71));
                dp[i + 3] = 255;
            }
        }
    }
    self.detail = make_3d(dp, CLOUD_DETAIL, "drie-cloud-detail");
    free(dp);
}

fn SgView make_3d(char[] px, int n, ZString label) @local {
    SgImage img = sg::make_image(
        &&{
            .type = TYPE_3D,
            .width = n,
            .height = n,
            .num_slices = n,
            .pixel_format = RGBA8,
            .data = { .ptr = px.ptr, .size = px.len },
            .label = label,
        }
    );
    return sg::make_view(&&{ .texture = { .image = img } });
}

// The weather map's coverage (r: clumps of cumulus in wider drifts) and cloud type (g), repeating.
fn void make_weather(char[] px) @local {
    usz n       = CLOUD_WEATHER;
    float[] cov = mem::new_array(float, (sz)(n * n));
    defer free(cov);
    float lo =  1e9;
    float hi = -1e9;
    for (usz y = 0; y < n; y++) {
        for (usz x = 0; x < n; x++) {
            float fx     = (x + 0.5f) / n;
            float fy     = (y + 0.5f) / n;
            float broad  = perlin_fbm(fx, fy, 0.3f, 3, 3, 81) * 0.5f + 0.5f;
            float clumps = worley(fx, fy, 0.5f, 24, 91) * 0.65f + worley(fx, fy, 0.5f, 48, 92) * 0.35f;
            float fine   = perlin_fbm(fx, fy, 0.7f, 16, 3, 93) * 0.5f + 0.5f;
            float c      = broad * 0.55f + clumps * 0.35f + fine * 0.2f;
            cov[y * n + x] = c;
            lo = math::min(lo, c);
            hi = math::max(hi, c);
            float type = perlin_fbm(fx, fy, 0.9f, 2, 2, 101) * 0.5f + 0.5f;
            px[(y * n + x) * 4 + 1] = byte_of(math::clamp((type - 0.3f) / 0.4f, 0.0f, 1.0f));
        }
    }
    for (usz i = 0; i < n * n; i++) px[i * 4] = byte_of((cov[i] - lo) / (hi - lo));
}

// b of the weather map: the distance (texels, at most 255) to the nearest texel of cloud (r over
// `threshold`), less a texel and a half (it is read filtered): a two-pass chamfer, round the
// repeat.
fn void weather_distance(char[] px, float threshold) @local {
    int n     = CLOUD_WEATHER;
    float[] d = mem::new_array(float, (sz)n * n);
    defer free(d);
    char t = byte_of(threshold);
    for (int i = 0; i < n * n; i++) d[i] = px[i * 4] > t ? 0 : 1e6f;
    const float A = 1;
    const float B = 1.41421356f;
    for (int round = 0; round < 2; round++) {
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < n; x++) {
                float v = d[y * n + x];
                int xm  = (x + n - 1) % n;
                int xp  = (x + 1) % n;
                int ym  = (y + n - 1) % n;
                v = math::min(v, d[y * n + xm] + A);
                v = math::min(v, d[ym * n + x] + A);
                v = math::min(v, d[ym * n + xm] + B);
                v = math::min(v, d[ym * n + xp] + B);
                d[y * n + x] = v;
            }
        }
        for (int y = n - 1; y >= 0; y--) {
            for (int x = n - 1; x >= 0; x--) {
                float v = d[y * n + x];
                int xm  = (x + n - 1) % n;
                int xp  = (x + 1) % n;
                int yp  = (y + 1) % n;
                v = math::min(v, d[y * n + xp] + A);
                v = math::min(v, d[yp * n + x] + A);
                v = math::min(v, d[yp * n + xp] + B);
                v = math::min(v, d[yp * n + xm] + B);
                d[y * n + x] = v;
            }
        }
    }
    for (int i = 0; i < n * n; i++) {
        px[i * 4 + 2] = (char)math::clamp((int)(d[i] - 1.5f), 0, 255);
        px[i * 4 + 3] = 255;
    }
}