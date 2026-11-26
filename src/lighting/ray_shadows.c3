// Ray-traced shadows of rect area lights (RectAreaLight.shadow.ray_traced), with Metal's ray
// tracing (lib/sokol.c3l/c/sokol_rt.h). In the deferred lighting pass, up to four of a
// render's rect area lights read their shadow from a mask that rays to points on the lights'
// rectangles fill, denoised over frames and across the surface: soft where the light is wide
// and the caster far, sharp where they touch. Where there is no ray tracing (not Metal, an
// older GPU), outside the deferred pass (transparent and layered materials) and in a
// Reflector's render, the lights' shadow maps (PCSS with shadow.softness) stand in for it.
module drie;
import sokol, std::collections::map, std::collections::list, std::math;

struct RtLight {
    float[4] center;
    float[4] hw;
    float[4] hh;
}

struct RtInstance {
    float[12] columns;
    CInt geometry;
}

struct RtTraceDesc {
    SgImage depth;
    SgImage normal;
    SgImage raw;
    SgImage blocker;
    SgImage[2] history;
    SgImage[2] dist;
    SgImage mask;
    CInt width;
    CInt height;
    CInt cur;
    CInt frame;
    CInt reset;
    float[16] inv_view_proj;
    float[16] prev_view_proj;
    float[4] eye;
    float[4] prev_eye;
    float px_scale;
    CInt zero_to_one;
    CInt flip_y;
    RtInstance* instances;
    CInt instance_count;
    RtLight[4] lights;
    CInt light_count;
}

extern fn CInt sg_rt_supported() @cname("sg_rt_supported");
extern fn CInt sg_rt_geometry(CInt id, SgBuffer vbuf, CInt stride, CInt vertex_count, SgBuffer ibuf, CInt index_count) @cname("sg_rt_geometry");
extern fn void sg_rt_geometry_free(CInt id) @cname("sg_rt_geometry_free");
extern fn CInt sg_rt_trace(RtTraceDesc* desc) @cname("sg_rt_trace");

const int RT_MAX_LIGHTS = 4;
// The largest shadow map of a ray-traced light (for what the deferred pass doesn't draw).
const int RT_MAP_SIZE = 256;
// Frames a geometry no render has traced keeps its acceleration structure.
const uint RT_KEEP_FRAMES @local = 240;
// Instances traced at most (an instanced mesh's each count).
const int RT_MAX_INSTANCES @local = 65536;

// A geometry's acceleration structure (sg_rt_geometry), and what it was built from.
struct RtGeometry @local {
    CInt id;
    uint version;
    uint upload_version;
    uint used;
}

struct RtState @private {
    bool checked;
    bool supported;
    // this render's ray-traced lights (rt_light: their channel of the mask)
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

// Whether this GPU traces rays (Metal's ray tracing; DRIE_NO_RAY_TRACING=1: not).
fn bool rt_supported() @private {
    if (!rts.checked) {
        rts.checked = true;
        rts.supported = sg_rt_supported() != 0;
    }
    return rts.supported;
}

// Whether the renderer is ray tracing rect area lights' shadows in this render.
fn bool Renderer.traces_rays(&self) => self.ray_traced_shadows && self.deferred && !state.bake.on && !self.reflecting && rt_supported();

// A new render's lights: none ray traced yet.
fn void rt_begin() @private {
    rts.count = 0;
}

// The mask channel of rect area light `ra` (its half width `hw` and half height `hh` in the
// world) plus 1, when it is ray traced in this render; 0: its shadow maps instead.
fn int rt_light(Renderer* r, RectAreaLight* ra, Vector3 hw, Vector3 hh) @private {
    if (!ra.cast_shadow || !ra.shadow.ray_traced || rts.count >= RT_MAX_LIGHTS || !r.traces_rays()) return 0;
    Vector3 c = ra.matrix_world.get_position();
    rts.lights[rts.count] = { { c.x, c.y, c.z, 0 }, { hw.x, hw.y, hw.z, 0 }, { hh.x, hh.y, hh.z, 0 } };
    rts.count++;
    return rts.count;
}

// Whether `l` is a rect area light whose shadow is ray traced where the deferred pass draws
// (its maps are then only for the rest, and smaller).
fn bool rt_traced(Light* l) @private {
    if (l.light_type != RECT_AREA || !l.cast_shadow) return false;
    RectAreaLight* ra = (RectAreaLight*)l;
    return ra.shadow.ray_traced && state.renderer && state.renderer.traces_rays();
}

// The mask the lighting pass reads (bound in tex_ext5): white until rays fill it.
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

// The acceleration structure of a caster's geometry, built or refitted when its vertices
// changed.
fn CInt rt_geometry(BufferGeometry* g) @local {
    if (!g.uploaded || !g.vbuf.id || !g.ibuf.id || g.index_count < 3) return 0;
    ulong key = (ulong)(uptr)g;
    RtGeometry* have = rts.geometries.get_ref(key) ?? null;
    if (have && have.version == g.version && have.upload_version == g.upload_version) {
        have.used = state.frame;
        return have.id;
    }
    // (new buffers: a new structure; the same ones with other vertices: refitted)
    CInt id = have && have.upload_version == g.upload_version ? have.id : 0;
    if (have && !id) sg_rt_geometry_free(have.id);
    id = sg_rt_geometry(id, g.vbuf, (CInt)(FLOATS_PER_VERTEX * float::size), (CInt)g.vertex_count(), g.ibuf, (CInt)g.index_count);
    if (!id) return 0;
    rts.geometries[key] = { id, g.version, g.upload_version, state.frame };
    return id;
}

fn void rt_walk(Object3D* obj) @local {
    if (!obj.visible) return;
    if (obj.kind == MESH) {
        Mesh* m = (Mesh*)obj;
        BufferGeometry* g = m.geometry;
        bool caster = g && m.material && m.material.visible && m.cast_shadow && !m.material.transparent && !m.is_line()
            && m.billboard == NO_BILLBOARD;
        CInt id = caster ? rt_geometry(g) : 0;
        if (id) {
            sz n = m.instance_data.len() > 0 ? (sz)m.instance_count : 0;
            if (!n) {
                rt_add_instance(m.matrix_world, id);
            } else {
                for (sz i = 0; i < n; i++) {
                    float* d = &m.instance_data[i * 12];
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

// Fills the mask for this render's ray-traced lights, after the G-buffer is drawn (its depth
// `depth` and normals `normal`, `width` x `height`) and before the lighting pass reads it.
fn void rt_trace(Scene* scene, Camera* camera, Mat4 view_proj_rel, SgImage depth, SgImage normal, int width, int height) @private {
    if (!rts.count) return;
    if (!rts.geometries.allocator) {
        rts.geometries.init(mem);
        rts.instances.init(mem);
    }
    rt_images(width, height);
    rts.instances.clear();
    rt_walk(scene);
    // structures no render used for a while
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
    bool reset = rts.camera != camera;
    Mat4 inv = view_proj_rel.invert();
    bool zo = depth_zero_to_one();
    float p11 = math::abs(camera.projection_matrix[1][1]);
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
