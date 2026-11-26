// Hardware ray-traced shadows for `RectAreaLight` (`RectAreaLight.shadow.ray_traced`) backed by
// Metal ray tracing (`lib/sokol.c3l/c/sokol_rt.h`). Up to `RT_MAX_LIGHTS` (4) rect area lights
// per deferred render trace stochastic shadow rays against cached per-geometry acceleration
// structures and denoise them temporally and spatially into a visibility mask bound at
// `VIEW_TEX_EXT5`. Non-Metal GPUs, forward transparent/layered passes, and `Reflector` renders
// fall back to PCSS shadow maps. Unused acceleration structures expire after `RT_KEEP_FRAMES`.

module drie;
import sokol, std::collections::map, std::collections::list, std::math;

// ---- Native ray-tracing interface

// World-space rect area light descriptor passed to `sg_rt_trace()`.
struct RtLight {
    float[4] center; // World-space rectangle center (`xyz`).
    float[4] hw; // World-space half-width basis vector (`xyz`).
    float[4] hh; // World-space half-height basis vector (`xyz`).
}

// Instance transform and acceleration-structure handle passed to `sg_rt_trace()`.
struct RtInstance {
    float[12] columns; // Column-major 4x3 world transform (`[col * 3 + row]`).
    CInt geometry; // Handle returned by `sg_rt_geometry()`.
}

// Parameter block for `sg_rt_trace()` matching `lib/sokol.c3l/c/sokol_rt.h`.
struct RtTraceDesc {
    SgImage depth; // G-Buffer depth image.
    SgImage normal; // G-Buffer world-normal image.
    SgImage raw; // Scratch raw shadow-ray visibility target (`RGBA16F`).
    SgImage blocker; // Scratch blocker-distance target (`RGBA16F`).
    SgImage[2] history; // Ping-pong temporal visibility history (`RGBA16F`).
    SgImage[2] dist; // Ping-pong linear camera distance history (`R32F`).
    SgImage mask; // Final denoised 4-channel shadow visibility mask (`RGBA16F`).
    CInt width; // G-Buffer width in pixels.
    CInt height; // G-Buffer height in pixels.
    CInt cur; // Active history index (`0` or `1`).
    CInt frame; // Monotonic frame counter for stochastic ray jitter.
    CInt reset; // Non-zero to discard temporal history on camera switch or resize.
    float[16] inv_view_proj; // Inverse camera-relative view-projection matrix (column-major).
    float[16] prev_view_proj; // Previous frame's camera-relative view-projection matrix.
    float[4] eye; // Current camera world position (`xyz`).
    float[4] prev_eye; // Previous frame's camera world position (`xyz`).
    float px_scale; // Perspective screen-space pixel scale (`|P[1][1]| * height * 0.5`).
    CInt zero_to_one; // Non-zero when NDC depth spans `[0, 1]`.
    CInt flip_y; // Non-zero when image Y runs top-to-bottom.
    RtInstance* instances; // Array of world-space shadow-caster instances.
    CInt instance_count; // Number of entries in `instances`.
    RtLight[4] lights; // Active ray-traced rect area lights for this render.
    CInt light_count; // Number of entries in `lights` (`<= RT_MAX_LIGHTS`).
}

// Returns non-zero when the active graphics backend supports hardware ray tracing.
extern fn CInt sg_rt_supported() @cname("sg_rt_supported");
// Builds or refits (when `id != 0`) a bottom-level acceleration structure from mesh buffers.
extern fn CInt sg_rt_geometry(
    CInt id,
    SgBuffer vbuf,
    CInt stride,
    CInt vertex_count,
    SgBuffer ibuf,
    CInt index_count
) @cname("sg_rt_geometry");
// Destroys a bottom-level acceleration structure returned by `sg_rt_geometry()`.
extern fn void sg_rt_geometry_free(CInt id) @cname("sg_rt_geometry_free");
// Traces and denoises rect-area shadow visibility for the current G-Buffer; returns non-zero on success.
extern fn CInt sg_rt_trace(RtTraceDesc* desc) @cname("sg_rt_trace");

// ---- Configuration & state

// Maximum number of ray-traced rect area lights packed into one RGBA visibility mask.
const int RT_MAX_LIGHTS           = 4;
// Maximum shadow-map resolution for a ray-traced light (used only for forward fallback passes).
const int RT_MAP_SIZE             = 256;
// Number of frames an unused geometry retains its cached acceleration structure.
const uint RT_KEEP_FRAMES @local  = 240;
// Upper bound on shadow-caster instances submitted per frame (including instanced meshes).
const int RT_MAX_INSTANCES @local = 65536;

// Cached acceleration structure for a `BufferGeometry` and the buffer versions it matches.
struct RtGeometry @local {
    CInt id;
    uint version;
    uint upload_version;
    uint used;
}

struct RtState @private {
    bool checked;
    bool supported;
    // Active ray-traced lights for the current render (indexed by RGBA mask channel).
    RtLight[RT_MAX_LIGHTS] lights;
    int count;
    HashMap{ulong, RtGeometry} geometries;
    List{RtInstance} instances;
    int width;
    int height;
    SgImage[8] images;
    SgView mask_view;
    int cur;
    Camera* camera;
    Mat4 prev_view_proj;
    Vector3 prev_eye;
    uint traced_frame;
}

RtState rts @private;

// Returns true when hardware ray tracing is available (disabled by `DRIE_NO_RAY_TRACING=1`).
fn bool rt_supported() @private {
    if (!rts.checked) {
        rts.checked = true;
        rts.supported = sg_rt_supported() != 0;
    }
    return rts.supported;
}

// Returns true when the renderer will ray-trace rect area light shadows in the current render.
fn bool Renderer.traces_rays(
    &self
) => self.ray_traced_shadows && self.deferred && !state.bake.on && !self.reflecting && rt_supported();

// ---- Per-frame tracing

// Resets the per-render ray-traced light list before light collection begins.
fn void rt_begin() @private {
    rts.count = 0;
}

// Registers `ra` (with world half-width `hw` and half-height `hh`) for ray tracing and returns
// its 1-based mask channel (`1..RT_MAX_LIGHTS`), or `0` to fall back to shadow maps.
fn int rt_light(Renderer* r, RectAreaLight* ra, Vector3 hw, Vector3 hh) @private {
    if (!ra.cast_shadow || !ra.shadow.ray_traced || rts.count >= RT_MAX_LIGHTS || !r.traces_rays()) return 0;
    Vector3 c = ra.matrix_world.get_position();
    rts.lights[rts.count] = { { c.x, c.y, c.z, 0 }, { hw.x, hw.y, hw.z, 0 }, { hh.x, hh.y, hh.z, 0 } };
    rts.count++;
    return rts.count;
}

// Returns true when `l` is a rect area light ray-traced in the deferred pass (so its fallback
// shadow map can stay at `RT_MAP_SIZE`).
fn bool rt_traced(Light* l) @private {
    if (l.light_type != RECT_AREA || !l.cast_shadow) return false;
    RectAreaLight* ra = (RectAreaLight*)l;
    return ra.shadow.ray_traced && state.renderer && state.renderer.traces_rays();
}

// Returns the denoised shadow mask view (`VIEW_TEX_EXT5`), or `none` if no trace ran this frame.
fn SgView rt_mask(SgView none) @private => rts.traced_frame == state.frame + 1 ? rts.mask_view : none;

fn void rt_images(int w, int h) @local {
    if (rts.width == w && rts.height == h) return;
    if (rts.width) {
        sg::destroy_view(rts.mask_view);
        foreach (img : rts.images) sg::destroy_image(img);
    }
    rts.width = w;
    rts.height = h;
    for (int i = 0; i < 7; i++) {
        bool dist = i == 4 || i == 5;
        rts.images[i] = sg::make_image(
            &&{
                .usage = { .storage_image = true },
                .width = w,
                .height = h,
                .pixel_format = dist ? R32F : RGBA16F,
                .label = "drie-rt-shadow",
            }
        );
    }
    rts.mask_view = sg::make_view(&&{ .texture = { .image = rts.images[6] } });
    rts.camera = null;
}

fn void rt_add_instance(Mat4 m, CInt geometry) @local {
    if (rts.instances.len() >= RT_MAX_INSTANCES) return;
    RtInstance inst = { .geometry = geometry };
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 3; r++) inst.columns[c * 3 + r] = m[c][r];
    }
    rts.instances.push(inst);
}

// Returns the acceleration-structure handle for `g`, building or refitting when vertices update.
fn CInt rt_geometry(BufferGeometry* g) @local {
    if (!g.uploaded || !g.vbuf.id || !g.ibuf.id || g.index_count < 3) return 0;
    ulong key        = (ulong)(uptr)g;
    RtGeometry* have = rts.geometries.get_ref(key) ?? null;
    if (have && have.version == g.version && have.upload_version == g.upload_version) {
        have.used = state.frame;
        return have.id;
    }
    // Reallocated GPU buffers require a fresh structure; updated vertices on the same buffers refit.
    CInt id = have && have.upload_version == g.upload_version ? have.id : 0;
    if (have && !id) sg_rt_geometry_free(have.id);
    id = sg_rt_geometry(
        id,
        g.vbuf,
        (CInt)(FLOATS_PER_VERTEX * float::size),
        (CInt)g.vertex_count(),
        g.ibuf,
        (CInt)g.index_count
    );
    if (!id) return 0;
    rts.geometries[key] = { id, g.version, g.upload_version, state.frame };
    return id;
}

fn void rt_walk(Object3D* obj) @local {
    if (!obj.visible) return;
    if (obj.kind == MESH) {
        Mesh* m           = (Mesh*)obj;
        BufferGeometry* g = m.geometry;
        bool caster       = g && m.material && m.material.visible && m.cast_shadow && !m.material.transparent
            && !m.is_line()
            && m.billboard == NO_BILLBOARD;
        CInt id           = caster ? rt_geometry(g) : 0;
        if (id) {
            sz n = m.instance_data.len() > 0 ? (sz)m.instance_count : 0;
            if (!n) {
                rt_add_instance(m.matrix_world, id);
            } else {
                for (sz i = 0; i < n; i++) {
                    float* d  = &m.instance_data[i * 12];
                    Mat4 inst = mat4_identity();
                    for (int r = 0; r < 3; r++) {
                        for (int c = 0; c < 4; c++) inst[c][r] = d[r * 4 + c];
                    }
                    rt_add_instance(m.matrix_world.mul(inst), id);
                }
            }
        }
    }
    foreach (c : obj.children) rt_walk(c);
}

// Traces and denoises the visibility mask for this render's ray-traced lights between the
// G-Buffer pass (`depth`, `normal`, `width` x `height`) and the deferred lighting pass.
fn void rt_trace(
    Scene* scene,
    Camera* camera,
    Mat4 view_proj_rel,
    SgImage depth,
    SgImage normal,
    int width,
    int height
) @private {
    if (!rts.count) return;
    if (!rts.geometries.allocator) {
        rts.geometries.init(mem);
        rts.instances.init(mem);
    }
    rt_images(width, height);
    rts.instances.clear();
    rt_walk(scene);
    // Evict acceleration structures not touched for `RT_KEEP_FRAMES` frames.
    @pool() {
        List{ulong} gone;
        gone.tinit();
        rts.geometries.@each(; ulong key, RtGeometry g) {
            if (state.frame - g.used > RT_KEEP_FRAMES) gone.push(key);
        };
        foreach (key : gone) {
            sg_rt_geometry_free(rts.geometries[key]!!.id);
            rts.geometries.remove(key);
        }
    };
    Vector3 eye = camera.matrix_world.get_position();
    bool reset  = rts.camera != camera;
    Mat4 inv    = view_proj_rel.invert();
    bool zo     = depth_zero_to_one();
    float p11   = math::abs(camera.projection_matrix[1][1]);
    RtTraceDesc d = {
        .depth = depth,
        .normal = normal,
        .raw = rts.images[0],
        .blocker = rts.images[1],
        .history = { rts.images[2], rts.images[3] },
        .dist = { rts.images[4], rts.images[5] },
        .mask = rts.images[6],
        .width = (CInt)width,
        .height = (CInt)height,
        .cur = (CInt)rts.cur,
        .frame = (CInt)state.frame,
        .reset = reset ? 1 : 0,
        .eye = { eye.x, eye.y, eye.z, 0 },
        .prev_eye = { rts.prev_eye.x, rts.prev_eye.y, rts.prev_eye.z, 0 },
        .px_scale = camera.camera_type == PERSPECTIVE ? p11 * (float)height * 0.5f : 0,
        .zero_to_one = zo ? 1 : 0,
        .flip_y = zo ? 1 : 0,
        .instances = rts.instances.entries,
        .instance_count = (CInt)rts.instances.len(),
        .lights = rts.lights,
        .light_count = (CInt)rts.count,
    };
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            d.inv_view_proj[c * 4 + r] = inv[c][r];
            d.prev_view_proj[c * 4 + r] = rts.prev_view_proj[c][r];
        }
    }
    if (sg_rt_trace(&d) == 0) return;
    rts.traced_frame = state.frame + 1;
    rts.cur = 1 - rts.cur;
    rts.camera = camera;
    rts.prev_view_proj = view_proj_rel;
    rts.prev_eye = eye;
}