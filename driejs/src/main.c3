// Runs three.js in boomkat with drie as its renderer.
//
// js/drie_renderer.js stands in for WebGLRenderer: three.js keeps its own scene graph and
// computes the world matrices, and the renderer mirrors what it finds into a flat drie
// scene through the handle API below (numbers in, numbers out). Once per frame, `sync`
// copies every object's world matrix and visibility out of one Float32Array; drie then
// renders with its own matrix update switched off.
//
//   out/driejs demos/webgl_geometries.js                 in a window
//   out/driejs --frames 600 demos/webgl_geometries.js    offscreen, timed
//   out/driejs --frames 3 --png /tmp/shot.png demos/webgl_geometries.js
//   out/driejs --frames 600 --profile ...   the same, with the time per part of a frame
//   out/driejs --no-jit ...     interpreted only (boomkat's JIT, both tiers, is on by default)
//
// Handle spaces are disjoint so one number identifies one thing:
//
//   objects     1      .. 1_000_000
//   geometries  1e6    ..
//   materials   2e6    ..
//   textures    3e6    ..
//   renderers   5e6    ..
module main;

import boomkat;
import boomkat::builtins, boomkat::hobject;
import drie, drie::ui;
import sokol;
import std::io, std::time, std::math, std::collections::list;
import drie::readback, libc, draco;

const double GEOMETRY_BASE = 1_000_000;
const double MATERIAL_BASE = 2_000_000;
const double TEXTURE_BASE = 3_000_000;
const double RENDERER_BASE = 5_000_000;

// one record of `sync`: handle, visible, then the 16 numbers of the world matrix
const int SYNC_STRIDE = 18;

List{Object3D*} objects;
List{BufferGeometry*} geometries;
List{Material*} materials;
List{Texture*} textures;
List{Renderer*} renderers;

JsContext js;
JsValue js_loop;
Renderer* loop_renderer;
String script_path;
String base_dir;
// the working directory, where js/ and vendor/ are
String root;
String png_path;
int frames;
bool jit = true;
bool profile;
// time spent in sg::commit after each frame
long commit_ns;
time::Clock start_clock;
// time spent in drie's render, to tell it apart from the JS side of a frame
long render_ns;

// ----------------------------------------------------------------------------
// Handles
// ----------------------------------------------------------------------------

fn double reg_object(Object3D* o)
{
    objects.push(o);
    return (double)objects.len();
}

fn Object3D* get_object(double id)
{
    long i = (long)id - 1;
    if (i < 0 || i >= objects.len()) return null;
    return objects[i];
}

macro double @reg(#list, ptr, double base)
{
    #list.push(ptr);
    return base + (double)(#list.len() - 1);
}

macro @get(#list, double id, double base)
{
    long i = (long)(id - base);
    if (id < base || i >= #list.len()) return null;
    return #list[i];
}

fn BufferGeometry* get_geometry(double id) => @get(geometries, id, GEOMETRY_BASE);
fn Material* get_material(double id) => @get(materials, id, MATERIAL_BASE);
fn Texture* get_texture(double id) => @get(textures, id, TEXTURE_BASE);
fn Renderer* get_renderer(double id) => @get(renderers, id, RENDERER_BASE);

// ----------------------------------------------------------------------------
// Arguments
// ----------------------------------------------------------------------------

fn double num_arg(JsCtx ctx, uint i, double fallback)
{
    double? v = ctx.arg(i).as_number();
    if (catch v) return fallback;
    return v;
}

fn bool bool_arg(JsCtx ctx, uint i, bool fallback)
{
    JsArg a = ctx.arg(i);
    if (a.is_undefined()) return fallback;
    bool? v = a.as_bool();
    if (catch v) return fallback;
    return v;
}

fn Color color_of(double hex) => (Color)(uint)hex;

fn Side side_of(double v)
{
    if (v > 1.5) return DOUBLE_SIDE;
    if (v > 0.5) return BACK_SIDE;
    return FRONT_SIDE;
}

fn Wrapping wrap_of(double v)
{
    switch ((int)v)
    {
        case 1001: return CLAMP_TO_EDGE_WRAPPING;
        case 1002: return MIRRORED_REPEAT_WRAPPING;
        default: return REPEAT_WRAPPING;
    }
}

// A typed array argument, read in place.
struct View
{
    void* data;
    usz len;
    ObjClass cls;
}

fn bool view_arg(JsCtx ctx, uint i, View* out)
{
    TVal tv = ctx.arg(i).tv;
    if (!tv.is_object() || tv.get_heapptr() == null) return false;
    HObject* view = (HObject*)tv.get_heapptr();
    if (!builtins::is_typed_array_class(view.flags.obj_class) || builtins::ta_is_detached(view)) return false;
    HObject* buf = (*view.extra_ptr()).view.buffer;
    out.data = (*buf.extra_ptr()).ab.data + (*view.extra_ptr()).view.byte_offset;
    out.len = builtins::ta_element_count(view);
    out.cls = view.flags.obj_class;
    return true;
}

fn float[] floats_arg(JsCtx ctx, uint i)
{
    View v;
    if (!view_arg(ctx, i, &v) || v.cls != ObjClass.FLOAT32ARRAY) return {};
    return ((float*)v.data)[:v.len];
}

// ----------------------------------------------------------------------------
// Scene graph
// ----------------------------------------------------------------------------

fn void h_renderer(JsCtx ctx, void* udata)
{
    String title = ctx.arg(3).as_string() ?? "three.js";
    Renderer* r = drie::renderer(antialias: bool_arg(ctx, 0, true), width: (int)num_arg(ctx, 1, 1280),
                                        height: (int)num_arg(ctx, 2, 720), title: title);
    ctx.ret_number(@reg(renderers, r, RENDERER_BASE));
}

fn void h_scene(JsCtx ctx, void* udata)
{
    Scene* s = drie::scene();
    // three.js has computed the world matrices: `sync` hands them over
    s.no_matrix_world_update = true;
    ctx.ret_number(reg_object((Object3D*)s));
}

fn void h_group(JsCtx ctx, void* udata) => ctx.ret_number(reg_object((Object3D*)drie::group()));

fn void h_camera(JsCtx ctx, void* udata)
{
    PerspectiveCamera* c = drie::perspective_camera((float)num_arg(ctx, 0, 50), (float)num_arg(ctx, 1, 1),
                                                    (float)num_arg(ctx, 2, 0.1), (float)num_arg(ctx, 3, 2000));
    ctx.ret_number(reg_object((Object3D*)c));
}

fn void h_camera_update(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != CAMERA) { ctx.throw_error(TYPE, "cameraUpdate: not a camera"); return; }
    Camera* c = (Camera*)o;
    c.near = (float)num_arg(ctx, 3, c.near);
    c.far = (float)num_arg(ctx, 4, c.far);
    c.zoom = (float)num_arg(ctx, 5, c.zoom);
    if (c.camera_type == PERSPECTIVE)
    {
        PerspectiveCamera* p = (PerspectiveCamera*)c;
        p.fov = (float)num_arg(ctx, 1, p.fov);
        p.aspect = (float)num_arg(ctx, 2, p.aspect);
    }
    c.update_projection_matrix();
}

// light(kind, color, intensity, distance, decay, angle, penumbra, groundColor)
// kind: 0 ambient, 1 hemisphere, 2 directional, 3 point, 4 spot
fn void h_light(JsCtx ctx, void* udata)
{
    Color color = color_of(num_arg(ctx, 1, 0xffffff));
    float intensity = (float)num_arg(ctx, 2, 1);
    float distance = (float)num_arg(ctx, 3, 0);
    float decay = (float)num_arg(ctx, 4, 2);
    Light* l;
    switch ((int)num_arg(ctx, 0, 0))
    {
        case 1: l = (Light*)drie::hemisphere_light(color, color_of(num_arg(ctx, 7, 0xffffff)), intensity);
        case 2: l = (Light*)drie::directional_light(color, intensity);
        case 3: l = (Light*)drie::point_light(color, intensity, distance, decay);
        case 4: l = (Light*)drie::spot_light(color, intensity, distance, (float)num_arg(ctx, 5, drie::PI / 3),
                                              (float)num_arg(ctx, 6, 0), decay);
        default: l = (Light*)drie::ambient_light(color, intensity);
    }
    ctx.ret_number(reg_object((Object3D*)l));
}

// The object a directional or spot light points at, or null.
fn void h_light_target(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != LIGHT) { ctx.ret_null(); return; }
    Light* l = (Light*)o;
    Object3D* target = null;
    if (l.light_type == DIRECTIONAL) target = ((DirectionalLight*)l).target;
    if (l.light_type == SPOT) target = ((SpotLight*)l).target;
    if (target == null) { ctx.ret_null(); return; }
    ctx.ret_number(reg_object(target));
}

// geometry(position, normal, uv, color, index): three's attribute arrays (Float32Array;
// index Uint16Array or Uint32Array), any but position may be null. Without normals they
// are computed, without an index every three vertices are a triangle.
fn void h_geometry(JsCtx ctx, void* udata)
{
    float[] pos = floats_arg(ctx, 0);
    float[] nrm = floats_arg(ctx, 1);
    float[] uv = floats_arg(ctx, 2);
    float[] col = floats_arg(ctx, 3);
    uint count = (uint)(pos.len / 3);
    uint col_size = count && col.len >= (usz)count * 4 ? 4 : 3;
    if (col.len < (usz)count * 3) col = {};
    if (nrm.len < (usz)count * 3) nrm = {};
    if (uv.len < (usz)count * 2) uv = {};

    BufferGeometry* g = mem::new(BufferGeometry);
    g.init_lists();
    g.vertices.reserve((usz)count * drie::FLOATS_PER_VERTEX);
    for (uint i = 0; i < count; i++)
    {
        Vector3 n = nrm.len ? { nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2] } : { 0, 1, 0 };
        Vector2 t = uv.len ? { uv[i * 2], uv[i * 2 + 1] } : { 0, 0 };
        Vector3 c = { 1, 1, 1 };
        float a = 1;
        if (col.len)
        {
            c = { col[i * col_size], col[i * col_size + 1], col[i * col_size + 2] };
            if (col_size == 4) a = col[i * 4 + 3];
        }
        g.add_vertex({ pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2] }, n, t, c, a);
    }
    View idx;
    if (view_arg(ctx, 4, &idx) && idx.len)
    {
        g.indices.reserve((sz)idx.len);
        switch (idx.cls)
        {
            case ObjClass.UINT16ARRAY:
                foreach (i : ((ushort*)idx.data)[:idx.len]) g.indices.push(i);
            case ObjClass.UINT32ARRAY:
                foreach (i : ((uint*)idx.data)[:idx.len]) g.indices.push(i);
            default:
                ctx.throw_error(TYPE, "geometry: index must be a Uint16Array or Uint32Array");
                return;
        }
    }
    else
    {
        for (uint i = 0; i < count; i++) g.indices.push(i);
    }
    if (!nrm.len && count) g.compute_vertex_normals();
    // skinIndex (Uint16Array) and skinWeight, 4 of each per vertex
    View joints;
    float[] weights = floats_arg(ctx, 6);
    if (view_arg(ctx, 5, &joints) && joints.cls == ObjClass.UINT16ARRAY && joints.len >= (usz)count * 4
        && weights.len >= (usz)count * 4)
    {
        ushort[] j = ((ushort*)joints.data)[:joints.len];
        g.ensure_skin();
        for (uint i = 0; i < count; i++)
        {
            g.set_skin(i, { j[i * 4], j[i * 4 + 1], j[i * 4 + 2], j[i * 4 + 3] },
                       { weights[i * 4], weights[i * 4 + 1], weights[i * 4 + 2], weights[i * 4 + 3] });
        }
    }
    ctx.ret_number(@reg(geometries, g, GEOMETRY_BASE));
}

// texture(source, srgb, wrapS, wrapT, anisotropy, repeatX, repeatY, offsetX, offsetY, flipY):
// `source` is a file's path or an encoded image (PNG, JPEG) in a Uint8Array
fn void h_texture(JsCtx ctx, void* udata)
{
    ColorSpace space = bool_arg(ctx, 1, true) ? SRGB_COLOR_SPACE : NO_COLOR_SPACE;
    Texture* t;
    View bytes;
    if (view_arg(ctx, 0, &bytes) && bytes.cls == ObjClass.UINT8ARRAY)
    {
        if (try decoded = drie::decode_texture(((char*)bytes.data)[:bytes.len], space))
        {
            t = decoded;
        }
        else
        {
            io::eprintn("driejs: cannot decode an image");
            t = drie::data_texture({ 255, 0, 255, 255 }, 1, 1, space);
        }
    }
    else
    {
        String? path = ctx.arg(0).as_string();
        if (catch path) { ctx.throw_error(TYPE, "texture: expected a path or a Uint8Array"); return; }
        t = drie::load_texture(path, space);
    }
    t.flip_y = bool_arg(ctx, 9, true);
    t.wrap_s = wrap_of(num_arg(ctx, 2, 1001));
    t.wrap_t = wrap_of(num_arg(ctx, 3, 1001));
    t.anisotropy = (int)num_arg(ctx, 4, 1);
    t.repeat = { (float)num_arg(ctx, 5, 1), (float)num_arg(ctx, 6, 1) };
    t.offset = { (float)num_arg(ctx, 7, 0), (float)num_arg(ctx, 8, 0) };
    ctx.ret_number(@reg(textures, t, TEXTURE_BASE));
}

// material(kind, color, emissive, specular, shininess, roughness, metalness, opacity,
//          transparent, side, wireframe, map, vertexColors)
// kind: 0 basic, 1 lambert, 2 phong, 3 standard, 4 normal
fn void h_material(JsCtx ctx, void* udata)
{
    Color color = color_of(num_arg(ctx, 1, 0xffffff));
    Color emissive = color_of(num_arg(ctx, 2, 0));
    float opacity = (float)num_arg(ctx, 7, 1);
    bool transparent = bool_arg(ctx, 8, false);
    Side side = side_of(num_arg(ctx, 9, 0));
    bool wireframe = bool_arg(ctx, 10, false);
    Material* m;
    switch ((int)num_arg(ctx, 0, 0))
    {
        case 1:
            m = (Material*)drie::mesh_lambert_material(color: color, emissive: emissive, opacity: opacity,
                                                        transparent: transparent, side: side, wireframe: wireframe);
        case 2:
            m = (Material*)drie::mesh_phong_material(color: color, specular: color_of(num_arg(ctx, 3, 0x111111)),
                                                      shininess: (float)num_arg(ctx, 4, 30), emissive: emissive,
                                                      opacity: opacity, transparent: transparent, side: side,
                                                      wireframe: wireframe);
        case 3:
            m = (Material*)drie::mesh_standard_material(color: color, roughness: (float)num_arg(ctx, 5, 1),
                                                         metalness: (float)num_arg(ctx, 6, 0), emissive: emissive,
                                                         opacity: opacity, transparent: transparent, side: side,
                                                         wireframe: wireframe);
        case 4:
            m = (Material*)drie::mesh_normal_material(opacity: opacity, transparent: transparent, side: side,
                                                       wireframe: wireframe);
        default:
            m = (Material*)drie::mesh_basic_material(color: color, opacity: opacity, transparent: transparent,
                                                      side: side, wireframe: wireframe);
    }
    m.map = get_texture(num_arg(ctx, 11, -1));
    m.vertex_colors = bool_arg(ctx, 12, false);
    m.alpha_test = (float)num_arg(ctx, 13, 0);
    m.color_scale = (float)num_arg(ctx, 14, 1);
    m.map_emissive = bool_arg(ctx, 15, false);
    m.emissive_intensity = (float)num_arg(ctx, 16, 1);
    ctx.ret_number(@reg(materials, m, MATERIAL_BASE));
}

// materialUpdate(material, color, emissive, specular, shininess, roughness, metalness,
// opacity): what three takes from a material every frame, when it changed.
fn void h_material_update(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m == null) { ctx.throw_error(TYPE, "materialUpdate: invalid material"); return; }
    m.color = color_of(num_arg(ctx, 1, 0xffffff));
    m.emissive = color_of(num_arg(ctx, 2, 0));
    m.specular = color_of(num_arg(ctx, 3, 0x111111));
    m.shininess = (float)num_arg(ctx, 4, 30);
    m.roughness = (float)num_arg(ctx, 5, 1);
    m.metalness = (float)num_arg(ctx, 6, 0);
    m.opacity = (float)num_arg(ctx, 7, 1);
    m.color_scale = (float)num_arg(ctx, 8, 1);
}

// shadow(object, cast, receive)
fn void h_shadow(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null) return;
    o.cast_shadow = bool_arg(ctx, 1, false);
    o.receive_shadow = bool_arg(ctx, 2, false);
}

// lightUpdate(light, color, intensity, groundColor)
fn void h_light_update(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != LIGHT) return;
    Light* l = (Light*)o;
    l.color = color_of(num_arg(ctx, 1, 0xffffff));
    l.intensity = (float)num_arg(ctx, 2, 1);
    if (l.light_type == HEMISPHERE) ((HemisphereLight*)l).ground_color = color_of(num_arg(ctx, 3, 0));
}

// lightShadow(light, cast, mapSize, radius, near, far, extent, normalBias, intensity): a
// directional light's shadow, as one box `extent` meters from its target to each side.
fn void h_light_shadow(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != LIGHT || ((Light*)o).light_type != DIRECTIONAL) return;
    DirectionalLight* l = (DirectionalLight*)o;
    l.cast_shadow = bool_arg(ctx, 1, false);
    l.shadow.map_size = (int)num_arg(ctx, 2, 512);
    l.shadow.radius = (float)num_arg(ctx, 3, 1);
    l.shadow.near = (float)num_arg(ctx, 4, 0.5);
    l.shadow.far = (float)num_arg(ctx, 5, 500);
    l.shadow.cascades = 1;
    l.shadow_extent = (float)num_arg(ctx, 6, 5);
    // three's normalBias is 0 unless set; drie's default keeps flat surfaces from shadowing
    // themselves
    float normal_bias = (float)num_arg(ctx, 7, 0);
    if (normal_bias != 0) l.shadow.normal_bias = normal_bias;
    l.shadow.intensity = (float)num_arg(ctx, 8, 1);
}

// fog(scene, kind, color, near, far, density): kind 0 none, 1 linear, 2 exponential
fn void h_fog(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != SCENE) return;
    Scene* s = (Scene*)o;
    Color color = color_of(num_arg(ctx, 2, 0xffffff));
    switch ((int)num_arg(ctx, 1, 0))
    {
        case 1: s.fog = drie::fog(color, (float)num_arg(ctx, 3, 1), (float)num_arg(ctx, 4, 1000));
        case 2: s.fog = drie::fog_exp(color, (float)num_arg(ctx, 5, 0.00025));
        default: s.fog = null;
    }
}

// toneMapping(renderer, three's constant, exposure)
fn void h_tone_mapping(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    if (r == null) return;
    switch ((int)num_arg(ctx, 1, 0))
    {
        case 1: r.tone_mapping = LINEAR_TONE_MAPPING;
        case 2: r.tone_mapping = REINHARD_TONE_MAPPING;
        case 3: r.tone_mapping = CINEON_TONE_MAPPING;
        case 4: r.tone_mapping = ACES_FILMIC_TONE_MAPPING;
        case 6: r.tone_mapping = AGX_TONE_MAPPING;
        case 7: r.tone_mapping = NEUTRAL_TONE_MAPPING;
        default: r.tone_mapping = NO_TONE_MAPPING;
    }
    r.tone_mapping_exposure = (float)num_arg(ctx, 2, 1);
}

// fileSize(path): bytes, or -1 if it cannot be read
fn void h_file_size(JsCtx ctx, void* udata)
{
    String? path = ctx.arg(0).as_string();
    if (catch path) { ctx.ret_number(-1); return; }
    long? size = file::get_size(path);
    if (catch size) { ctx.ret_number(-1); return; }
    ctx.ret_number((double)size);
}

// readFile(path, bytes): fills the Uint8Array (sized by fileSize) with the file
fn void h_read_file(JsCtx ctx, void* udata)
{
    String? path = ctx.arg(0).as_string();
    View out;
    if (catch path) { ctx.throw_error(TYPE, "readFile: expected a path"); return; }
    if (!view_arg(ctx, 1, &out) || out.cls != ObjClass.UINT8ARRAY) { ctx.throw_error(TYPE, "readFile: expected a Uint8Array"); return; }
    char[]? data = file::load_temp(path);
    if (catch data) { ctx.throw_error(TYPE, "readFile: cannot read the file"); return; }
    usz n = math::min((usz)data.len, out.len);
    libc::memcpy(out.data, data.ptr, n);
}

// ----------------------------------------------------------------------------
// Draco: DRACOLoader's worker decodes with libdraco (js/dom.js, DracoWorker)
// ----------------------------------------------------------------------------

List{draco::Decoded} dracos;

fn draco::Decoded draco_arg(JsCtx ctx)
{
    int i = (int)num_arg(ctx, 0, -1);
    return i >= 0 && i < dracos.len() ? dracos[i] : null;
}

// dracoDecode(bytes): a handle to the decoded mesh or point cloud
fn void h_draco_decode(JsCtx ctx, void* udata)
{
    View v;
    if (!view_arg(ctx, 0, &v) || (v.cls != ObjClass.INT8ARRAY && v.cls != ObjClass.UINT8ARRAY))
    {
        ctx.throw_error(TYPE, "dracoDecode: expected an Int8Array or Uint8Array");
        return;
    }
    draco::Decoded d = draco::decode(v.data, v.len);
    if (d == null)
    {
        ctx.throw_error(ERROR, string::tformat("THREE.DRACOLoader: Decoding failed: %s", draco::error()));
        return;
    }
    foreach (i, slot : dracos)
    {
        if (slot == null)
        {
            dracos[i] = d;
            ctx.ret_number((double)i);
            return;
        }
    }
    dracos.push(d);
    ctx.ret_number((double)(dracos.len() - 1));
}

// dracoInfo(handle, what): whether it is a mesh (0), its points (1) or its faces (2)
fn void h_draco_info(JsCtx ctx, void* udata)
{
    draco::Decoded d = draco_arg(ctx);
    if (d == null) { ctx.throw_error(TYPE, "dracoInfo: invalid handle"); return; }
    switch ((int)num_arg(ctx, 1, 0))
    {
        case 0: ctx.ret_number(draco::is_mesh(d) ? 1 : 0);
        case 1: ctx.ret_number(draco::num_points(d));
        default: ctx.ret_number(draco::num_faces(d));
    }
}

// dracoAttribute(handle, byUniqueId, id): the attribute with that unique id, or the first of that
// semantic type (draco::GeometryAttribute::Type), or -1
fn void h_draco_attribute(JsCtx ctx, void* udata)
{
    draco::Decoded d = draco_arg(ctx);
    if (d == null) { ctx.throw_error(TYPE, "dracoAttribute: invalid handle"); return; }
    int id = (int)num_arg(ctx, 2, -1);
    ctx.ret_number(bool_arg(ctx, 1, true) ? draco::attribute_by_unique_id(d, (uint)id) : draco::attribute_by_type(d, (draco::AttributeType)id));
}

fn void h_draco_components(JsCtx ctx, void* udata)
{
    draco::Decoded d = draco_arg(ctx);
    if (d == null) { ctx.throw_error(TYPE, "dracoComponents: invalid handle"); return; }
    ctx.ret_number(draco::attribute_components(d, (int)num_arg(ctx, 1, 0)));
}

// dracoAttributeData(handle, attribute, array): fills the typed array, points * components
// long, converting to its element type
fn void h_draco_attribute_data(JsCtx ctx, void* udata)
{
    draco::Decoded d = draco_arg(ctx);
    if (d == null) { ctx.throw_error(TYPE, "dracoAttributeData: invalid handle"); return; }
    int att = (int)num_arg(ctx, 1, 0);
    View v;
    if (!view_arg(ctx, 2, &v)) { ctx.throw_error(TYPE, "dracoAttributeData: expected a typed array"); return; }
    draco::DataType type;
    switch (v.cls)
    {
        case ObjClass.INT8ARRAY: type = INT8;
        case ObjClass.UINT8ARRAY: type = UINT8;
        case ObjClass.INT16ARRAY: type = INT16;
        case ObjClass.UINT16ARRAY: type = UINT16;
        case ObjClass.INT32ARRAY: type = INT32;
        case ObjClass.UINT32ARRAY: type = UINT32;
        case ObjClass.FLOAT32ARRAY: type = FLOAT32;
        default: ctx.throw_error(TYPE, "dracoAttributeData: unsupported array type"); return;
    }
    if (v.len < (usz)draco::num_points(d) * (usz)draco::attribute_components(d, att))
    {
        ctx.throw_error(RANGE, "dracoAttributeData: array too short");
        return;
    }
    if (!draco::attribute_data(d, att, type, v.data)) ctx.throw_error(ERROR, "dracoAttributeData: conversion failed");
}

// dracoIndices(handle, Uint32Array): fills faces * 3 indices
fn void h_draco_indices(JsCtx ctx, void* udata)
{
    draco::Decoded d = draco_arg(ctx);
    View v;
    if (d == null || !view_arg(ctx, 1, &v) || v.cls != ObjClass.UINT32ARRAY || v.len < (usz)draco::num_faces(d) * 3)
    {
        ctx.throw_error(TYPE, "dracoIndices: expected a handle and a Uint32Array of faces * 3");
        return;
    }
    draco::indices(d, v.data);
}

fn void h_draco_free(JsCtx ctx, void* udata)
{
    draco::Decoded d = draco_arg(ctx);
    if (d == null) return;
    draco::free(d);
    dracos[(int)num_arg(ctx, 0, -1)] = null;
}

fn void h_lock_mouse(JsCtx ctx, void* udata)
{
    if (sapp::isvalid()) sapp::lock_mouse(bool_arg(ctx, 0, false));
}

fn void h_mesh(JsCtx ctx, void* udata)
{
    BufferGeometry* g = get_geometry(num_arg(ctx, 0, -1));
    if (g == null) { ctx.throw_error(TYPE, "mesh: invalid geometry"); return; }
    ctx.ret_number(reg_object((Object3D*)drie::mesh(g, get_material(num_arg(ctx, 1, -1)))));
}

// skinnedMesh(geometry, material): a drie SkinnedMesh, posed by `skin`
fn void h_skinned_mesh(JsCtx ctx, void* udata)
{
    BufferGeometry* g = get_geometry(num_arg(ctx, 0, -1));
    if (g == null) { ctx.throw_error(TYPE, "skinnedMesh: invalid geometry"); return; }
    SkinnedMesh* m = drie::skinned_mesh(g, get_material(num_arg(ctx, 1, -1)));
    skinned.push(m);
    ctx.ret_number(reg_object((Object3D*)m));
}

// skin(mesh, matrices, count): one column-major matrix per bone, from the bind pose in mesh
// space to the posed mesh (three's bindMatrixInverse * boneMatrix * bindMatrix)
fn void h_skin(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    float[] mats = floats_arg(ctx, 1);
    usz count = (usz)num_arg(ctx, 2, 0);
    if (o == null || !skinned.contains((SkinnedMesh*)o)) { ctx.throw_error(TYPE, "skin: not a skinned mesh"); return; }
    if (count * 16 > mats.len) { ctx.throw_error(RANGE, "skin: count exceeds the matrices"); return; }
    ((SkinnedMesh*)o).skin_with(((drie::Mat4*)mats.ptr)[:count]);
}

// instancedMesh(geometry, material, count)
fn void h_instanced_mesh(JsCtx ctx, void* udata)
{
    BufferGeometry* g = get_geometry(num_arg(ctx, 0, -1));
    if (g == null) { ctx.throw_error(TYPE, "instancedMesh: invalid geometry"); return; }
    InstancedMesh* m = drie::instanced_mesh(g, get_material(num_arg(ctx, 1, -1)), (int)num_arg(ctx, 2, 0));
    instanced.push(m);
    ctx.ret_number(reg_object((Object3D*)m));
}

// instances(mesh, matrices, count): three's instanceMatrix.array
fn void h_instances(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    float[] mats = floats_arg(ctx, 1);
    int count = (int)num_arg(ctx, 2, 0);
    if (o == null || !instanced.contains((InstancedMesh*)o)) { ctx.throw_error(TYPE, "instances: not an instanced mesh"); return; }
    if ((usz)count * 16 > mats.len) { ctx.throw_error(RANGE, "instances: count exceeds the matrices"); return; }
    InstancedMesh* m = (InstancedMesh*)o;
    m.instance_data.clear();
    for (int i = 0; i < count * 12; i++) m.instance_data.push(0);
    drie::Mat4* src = (drie::Mat4*)mats.ptr;
    for (int i = 0; i < count; i++) m.set_matrix_at(i, src[i]);
}

// An environment that PMREMGenerator.fromScene asked for, captured at the next render (drie
// needs the GPU running for it).
struct Capture
{
    Texture* texture;
    Scene* scene;
    Vector3 position;
    float near;
    float far;
}
List{Capture} captures;
List{SkinnedMesh*} skinned;
List{InstancedMesh*} instanced;
List{Sky*} skies;

// environment(scene, x, y, z, near, far): a texture handle for the scene's environment map
fn void h_environment(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != SCENE) { ctx.throw_error(TYPE, "environment: not a scene"); return; }
    Texture* t = mem::new(Texture);
    captures.push({ t, (Scene*)o, { (float)num_arg(ctx, 1, 0), (float)num_arg(ctx, 2, 0), (float)num_arg(ctx, 3, 0) },
                    (float)num_arg(ctx, 4, 0.1), (float)num_arg(ctx, 5, 100) });
    ctx.ret_number(@reg(textures, t, TEXTURE_BASE));
}

// sky(): a drie Sky, for three's Sky
fn void h_sky(JsCtx ctx, void* udata)
{
    Sky* s = objects::sky();
    skies.push(s);
    ctx.ret_number(reg_object((Object3D*)s));
}

// skyParams(sky, turbidity, rayleigh, mieCoefficient, mieDirectionalG, sunX, sunY, sunZ,
// cloudScale, cloudSpeed, cloudCoverage, cloudDensity, cloudElevation, showSunDisc, time)
fn void h_sky_params(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || !skies.contains((Sky*)o)) { ctx.throw_error(TYPE, "skyParams: not a sky"); return; }
    Sky* s = (Sky*)o;
    s.turbidity = (float)num_arg(ctx, 1, 2);
    s.rayleigh = (float)num_arg(ctx, 2, 1);
    s.mie_coefficient = (float)num_arg(ctx, 3, 0.005);
    s.mie_directional_g = (float)num_arg(ctx, 4, 0.8);
    s.sun_position = { (float)num_arg(ctx, 5, 0), (float)num_arg(ctx, 6, 0), (float)num_arg(ctx, 7, 0) };
    s.cloud_scale = (float)num_arg(ctx, 8, 0.0002);
    s.cloud_speed = (float)num_arg(ctx, 9, 0.00002);
    s.cloud_coverage = (float)num_arg(ctx, 10, 0);
    s.cloud_density = (float)num_arg(ctx, 11, 0.4);
    s.cloud_elevation = (float)num_arg(ctx, 12, 0.5);
    s.show_sun_disc = num_arg(ctx, 13, 1) != 0;
    s.time = (float)num_arg(ctx, 14, 0);
}

// Puts every sky around the eye that is about to draw, inside its far plane (three draws its
// sky dome on the far plane).
fn void follow_skies(Vector3 eye, float far)
{
    foreach (s : skies) s.follow(eye, far);
}

fn void run_captures(Renderer* r)
{
    foreach (c : captures)
    {
        follow_skies(c.position, c.far);
        Texture* map = r.capture_environment(c.scene, c.position, near: c.near, far: c.far, hdr: true);
        *c.texture = *map;
        free(map);
    }
    captures.clear();
}

// sceneEnvironment(scene, texture or -1, intensity)
fn void h_scene_environment(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != SCENE) { ctx.throw_error(TYPE, "sceneEnvironment: not a scene"); return; }
    Scene* s = (Scene*)o;
    s.environment = get_texture(num_arg(ctx, 1, -1));
    s.environment_intensity = (float)num_arg(ctx, 2, 1);
}

fn void h_add(JsCtx ctx, void* udata)
{
    Object3D* parent = get_object(num_arg(ctx, 0, -1));
    Object3D* child = get_object(num_arg(ctx, 1, -1));
    if (parent == null || child == null) { ctx.throw_error(TYPE, "add: invalid object"); return; }
    parent.add(child);
}

// sync(records, count): `count` records of SYNC_STRIDE numbers, each a handle, a
// visibility and a column-major world matrix (three's matrixWorld.elements).
fn void h_sync(JsCtx ctx, void* udata)
{
    float[] rec = floats_arg(ctx, 0);
    usz count = (usz)num_arg(ctx, 1, 0);
    if (count * SYNC_STRIDE > rec.len) { ctx.throw_error(RANGE, "sync: count exceeds the records"); return; }
    for (usz i = 0; i < count; i++)
    {
        float[] r = rec[i * SYNC_STRIDE : SYNC_STRIDE];
        Object3D* o = get_object(r[0]);
        if (o == null) continue;
        o.visible = r[1] != 0;
        libc::memcpy(&o.matrix_world, &r[2], drie::Mat4::size);
    }
}

fn void h_background(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != SCENE) { ctx.throw_error(TYPE, "background: not a scene"); return; }
    ((Scene*)o).background = color_of(num_arg(ctx, 1, 0));
}

fn void h_render(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    Object3D* so = get_object(num_arg(ctx, 1, -1));
    Object3D* co = get_object(num_arg(ctx, 2, -1));
    if (r == null || so == null || so.kind != SCENE || co == null || co.kind != CAMERA)
    {
        ctx.throw_error(TYPE, "render: invalid argument");
        return;
    }
    time::Clock t = time::clock::now();
    if (captures.len()) run_captures(r);
    Camera* cam = (Camera*)co;
    follow_skies({ cam.matrix_world[3][0], cam.matrix_world[3][1], cam.matrix_world[3][2] }, cam.far);
    r.render((Scene*)so, cam);
    render_ns += (long)t.to_now();
}

fn void h_set_clear_color(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    if (r) r.set_clear_color(color_of(num_arg(ctx, 1, 0)));
}

fn void h_set_size(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    if (r) r.set_size((int)num_arg(ctx, 1, r.width), (int)num_arg(ctx, 2, r.height));
}

fn void h_info(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    if (r == null) { ctx.ret_number(0); return; }
    ctx.ret_number((int)num_arg(ctx, 1, 0) == 0 ? r.info.calls : r.info.triangles);
}

// Milliseconds from an arbitrary start (boomkat has no `performance`).
fn void h_now(JsCtx ctx, void* udata) => ctx.ret_number((double)(long)start_clock.to_now() / 1e6);

// stats(fps, fpsMin, fpsMax, ms, msMin, msMax), from js/stats.js each frame (fps -1 until
// the first second has passed). The first call puts the panel over the picture.
double[6] stats_values;
bool stats_shown;

fn void h_stats(JsCtx ctx, void* udata)
{
    for (uint i = 0; i < stats_values.len; i++) stats_values[i] = num_arg(ctx, i, -1);
    if (stats_shown) return;
    stats_shown = true;
    drie::add_overlay(&draw_stats);
}

// Like three's Stats panel: frames per second and milliseconds per frame in the top left
// corner, each with the lowest and highest so far.
fn void draw_stats()
{
    ui::setup();
    float h = sapp::heightf();
    // about 720 canvas units high: half the size of the menu text
    float scale = math::max(1.0f, math::round(h / 720));
    sdtx::set_context(sdtx::default_context());
    sdtx::canvas(sapp::widthf() / scale, h / scale);
    sdtx::font(ui::FONT_ORIC);
    // before the first second (FPS) or frame (MS) there is no value yet: 0 until then
    double[6] v = stats_values;
    foreach (&x : v) if (*x < 0) *x = 0;
    ZString fps = string::tformat_zstr("%d FPS (%d-%d)", (int)math::round(v[0]), (int)math::round(v[1]),
                                       (int)math::round(v[2]));
    ZString ms = string::tformat_zstr("%d MS (%d-%d)", (int)math::round(v[3]), (int)math::round(v[4]),
                                      (int)math::round(v[5]));
    // a 50% black box behind the text, with a glyph's half-width of padding
    float cw = sapp::widthf() / scale;
    float ch = h / scale;
    float bw = (float)(math::max(fps.len(), ms.len()) + 1) * 8;
    sgl::set_context(sgl::default_context());
    sgl::defaults();
    sgl::load_pipeline(ui::blend_pipeline());
    sgl::matrix_mode_projection();
    sgl::ortho(0, cw, ch, 0, -1, 1);
    sgl::begin_quads();
    sgl::c4f(0, 0, 0, 0.5f);
    sgl::v2f(0, 0);
    sgl::v2f(bw, 0);
    sgl::v2f(bw, 3.25f * 8);
    sgl::v2f(0, 3.25f * 8);
    sgl::end();
    sgl::draw();

    sdtx::origin(0.5f, 0.5f);
    sdtx::color3b(255, 216, 102);
    sdtx::puts(fps);
    // the second line a quarter of a glyph lower than the next row
    sdtx::pos(0, 1.25f);
    sdtx::color3b(120, 220, 232);
    sdtx::puts(ms);
    sdtx::draw();
}

// The loop starts once the script has finished, so promises made while it set up (texture
// loads) settle first.
fn void h_set_animation_loop(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    if (r == null) { ctx.throw_error(TYPE, "setAnimationLoop: invalid renderer"); return; }
    if (!js_loop.is_empty()) js.release(js_loop);
    js_loop = {};
    loop_renderer = null;
    if (ctx.arg(1).type_of() != FUNCTION) return;
    JsValue? f = ctx.persist(ctx.arg(1));
    if (catch f) { ctx.throw_error(TYPE, "setAnimationLoop: expected a function"); return; }
    js_loop = f;
    loop_renderer = r;
}

// ----------------------------------------------------------------------------
// Frames
// ----------------------------------------------------------------------------

fn bool js_frame(double time_ms)
{
    fire_timers();
    JsValue? t = js.number(time_ms);
    if (catch t) return false;
    defer js.release(t);
    JsValue? outcome = js.call(js_loop, { t });
    if (catch outcome)
    {
        io::eprintfn("uncaught error in the animation loop: %s", js.last_error());
        return false;
    }
    js.release(outcome);
    if (catch js.get_runtime().drain_microtasks())
    {
        io::eprintfn("uncaught error in a promise job: %s", js.last_error());
        return false;
    }
    return true;
}

// The window's mouse and keyboard, handed to js/dom.js (globalThis.__drieInput), which makes
// DOM events of them.
JsValue js_input;

fn bool input_event(Event* e)
{
    if (js_input.is_empty()) return false;
    double kind;
    double a = (double)(uint)e.mouse_button;
    double dx = e.mouse_dx;
    double dy = e.mouse_dy;
    switch (e.type)
    {
        case MOUSE_MOVE: kind = 0;
        case MOUSE_DOWN: kind = 1;
        case MOUSE_UP: kind = 2;
        case MOUSE_SCROLL:
            kind = 3;
            dx = e.scroll_x;
            dy = e.scroll_y;
        case KEY_DOWN:
        case KEY_UP:
            kind = e.type == KEY_DOWN ? 4 : 5;
            a = (double)(int)e.key_code;
            dx = e.key_repeat ? 1 : 0;
        default: return false;
    }
    // sokol gives framebuffer pixels, the DOM points
    double scale = sapp::dpi_scale();
    double[6] args = { kind, e.mouse_x / scale, e.mouse_y / scale, a, dx, dy };
    JsValue[6] values;
    for (int i = 0; i < 6; i++) values[i] = js.number(args[i]) ?? {};
    JsValue? outcome = js.call(js_input, values[..]);
    if (catch outcome)
    {
        io::eprintfn("uncaught error in an input event: %s", js.last_error());
    }
    else
    {
        js.release(outcome);
    }
    for (int i = 0; i < 6; i++) if (!values[i].is_empty()) js.release(values[i]);
    return false;
}

fn void window_frame(double time_ms)
{
    if (!js_frame(time_ms)) sapp::request_quit();
}

// Renders `frames` frames offscreen and prints the time per frame: all of it, and the
// part outside drie's render (three.js, the renderer's JS and the sync).
fn void run_offscreen(Renderer* r)
{
    r.init_headless();
    // antialias: true gives the picture the window's 4 samples, resolved into target.texture_image
    RenderTarget* target = drie::render_target(r.width, r.height, samples: r.sample_count);
    r.set_render_target(target);
    int warmup = frames > 20 ? 10 : 0;
    for (int i = 0; i < warmup; i++)
    {
        if (!js_frame((double)i * 1000.0 / 60)) return;
        drie::commit();
    }
    render_ns = 0;
    time::Clock t = time::clock::now();
    if (profile) (void)js.exec("globalThis.__drieReset?.()");
    for (int i = 0; i < frames; i++)
    {
        if (!js_frame((double)(warmup + i) * 1000.0 / 60)) return;
        time::Clock c = time::clock::now();
        drie::commit();
        commit_ns += (long)c.to_now();
    }
    double total = (double)(long)t.to_now() / 1e6 / frames;
    double render = (double)render_ns / 1e6 / frames;
    io::printfn("%d frames, %d draw calls, %d triangles", frames, r.info.calls, r.info.triangles);
    io::printfn("frame        %8.3f ms", total);
    io::printfn("  JS + sync  %8.3f ms", total - render);
    io::printfn("  drie       %8.3f ms (render, CPU side)", render);
    if (profile)
    {
        io::printn("of which:");
        (void)js.exec(string::tformat("globalThis.__drieReport?.(%d)", frames));
        io::printfn("  %-28s %8.3f ms", "sg commit (native)", (double)commit_ns / 1e6 / frames);
    }
    if (!png_path.len) return;
    char[] pixels = mem::new_array(char, (usz)r.width * (usz)r.height * 4);
    defer free(pixels);
    if (!readback::read_image(target.texture_image, r.width, r.height, pixels.ptr))
    {
        io::printn("readback failed");
        return;
    }
    io::printfn(write_png(png_path, pixels, r.width, r.height) ? "saved %s" : "could not save %s", png_path);
}

// ----------------------------------------------------------------------------
// Wiring
// ----------------------------------------------------------------------------

// Modules from files, with three's "three/addons/..." (an import map in the examples' pages)
// taken from vendor/jsm.
fn char[]? read_module(String name, void* udata)
{
    const String ADDONS = "three/addons/";
    if (name.starts_with(ADDONS))
    {
        return boomkat::read_module_file(string::tformat("%s/vendor/jsm/%s", root, name[ADDONS.len..]));
    }
    return boomkat::read_module_file(name);
}

fn void reg(JsValue obj, String name, JsHostFn handler, int arity)
{
    js.register_method(obj, name, handler, arity: arity)!!;
}

fn int usage()
{
    io::printn("usage: driejs [--no-jit] [--frames N] [--profile] [--png out.png] script.js");
    return 2;
}

fn int main(String[] args)
{
    for (sz i = 1; i < args.len; i++)
    {
        switch (args[i])
        {
            case "--frames":
                if (++i >= args.len) return usage();
                int? n = args[i].to_int();
                if (catch n) return usage();
                frames = n;
            case "--profile":
                profile = true;
            case "--no-jit":
                jit = false;
            case "--png":
                if (++i >= args.len) return usage();
                png_path = args[i];
            default:
                script_path = args[i];
        }
    }
    if (!script_path.len) return usage();
    if (try slash = script_path.rindex_of_char('/')) base_dir = script_path[:slash + 1];

    objects.init(mem);
    geometries.init(mem);
    materials.init(mem);
    textures.init(mem);
    renderers.init(mem);
    dracos.init(mem);

    start_clock = time::clock::now();
    js.create()!!;
    defer js.destroy();
    if (jit && (!js.get_runtime().set_jit() || !js.get_runtime().set_jit_opt()))
    {
        io::eprintn("could not enable the JIT");
        return 1;
    }

    // `import * as THREE from "three"`: three.js's core with drie's renderer as
    // WebGLRenderer, read from js/ and vendor/ under the working directory.
    Path? cwd = path::cwd(mem);
    if (catch cwd)
    {
        io::eprintn("cannot get the working directory");
        return 1;
    }
    root = (String)cwd;
    js.get_runtime().set_module_reader(&read_module);
    js.define_module("three", string::tformat(
        "import \"%s/js/dom.js\";\n"
        "import \"%s/js/wasm.js\";\n"
        "export * from \"%s/vendor/three.core.js\";\n"
        "export { DrieRenderer as WebGLRenderer } from \"%s/js/drie_renderer.js\";\n"
        "export { PMREMGenerator } from \"%s/js/pmrem.js\";\n", root, root, root, root, root))!!;
    // three's Stats draws into the DOM: this one is drawn by the host
    js.define_module("three/addons/libs/stats.module.js",
                     string::tformat("export { default } from \"%s/js/stats.js\";\n", root))!!;
    // and lil-gui's panel is not drawn at all
    js.define_module("three/addons/libs/lil-gui.module.min.js",
                     string::tformat("export * from \"%s/js/gui.js\";\nexport { default } from \"%s/js/gui.js\";\n", root, root))!!;

    JsValue natives = js.object()!!;
    defer js.release(natives);
    reg(natives, "renderer", &h_renderer, 4);
    reg(natives, "scene", &h_scene, 0);
    reg(natives, "group", &h_group, 0);
    reg(natives, "camera", &h_camera, 4);
    reg(natives, "cameraUpdate", &h_camera_update, 6);
    reg(natives, "light", &h_light, 8);
    reg(natives, "lightTarget", &h_light_target, 1);
    reg(natives, "geometry", &h_geometry, 7);
    reg(natives, "texture", &h_texture, 10);
    reg(natives, "material", &h_material, 17);
    reg(natives, "materialUpdate", &h_material_update, 9);
    reg(natives, "mesh", &h_mesh, 2);
    reg(natives, "add", &h_add, 2);
    reg(natives, "sync", &h_sync, 2);
    reg(natives, "background", &h_background, 2);
    reg(natives, "render", &h_render, 3);
    reg(natives, "setClearColor", &h_set_clear_color, 2);
    reg(natives, "setSize", &h_set_size, 3);
    reg(natives, "info", &h_info, 2);
    reg(natives, "now", &h_now, 0);
    reg(natives, "setAnimationLoop", &h_set_animation_loop, 2);
    reg(natives, "stats", &h_stats, 6);
    reg(natives, "shadow", &h_shadow, 3);
    reg(natives, "lightUpdate", &h_light_update, 4);
    reg(natives, "lightShadow", &h_light_shadow, 9);
    reg(natives, "fog", &h_fog, 6);
    reg(natives, "toneMapping", &h_tone_mapping, 3);
    reg(natives, "fileSize", &h_file_size, 1);
    reg(natives, "readFile", &h_read_file, 2);
    reg(natives, "dracoDecode", &h_draco_decode, 1);
    reg(natives, "dracoInfo", &h_draco_info, 2);
    reg(natives, "dracoAttribute", &h_draco_attribute, 3);
    reg(natives, "dracoComponents", &h_draco_components, 2);
    reg(natives, "dracoAttributeData", &h_draco_attribute_data, 3);
    reg(natives, "dracoIndices", &h_draco_indices, 2);
    reg(natives, "dracoFree", &h_draco_free, 1);
    register_wasm(natives);
    reg(natives, "lockMouse", &h_lock_mouse, 1);
    reg(natives, "skinnedMesh", &h_skinned_mesh, 2);
    reg(natives, "skin", &h_skin, 3);
    reg(natives, "instancedMesh", &h_instanced_mesh, 3);
    reg(natives, "instances", &h_instances, 3);
    reg(natives, "environment", &h_environment, 6);
    reg(natives, "sky", &h_sky, 0);
    reg(natives, "skyParams", &h_sky_params, 15);
    reg(natives, "sceneEnvironment", &h_scene_environment, 3);
    reg(natives, "xmlParse", &h_xml_parse, 1);

    JsValue flag = js.boolean(profile)!!;
    js.set(natives, "profile", flag)!!;
    js.release(flag);
    JsValue base = js.string(base_dir)!!;
    js.set(natives, "baseDir", base)!!;
    js.release(base);
    JsValue root_dir = js.string(string::tformat("%s/", root))!!;
    js.set(natives, "rootDir", root_dir)!!;
    js.release(root_dir);

    JsValue global = js.global()!!;
    defer js.release(global);
    js.set(global, "__drie", natives)!!;
    register_timers(global);

    char[]? source = file::load_temp(script_path);
    if (catch source)
    {
        io::eprintfn("cannot read %s", script_path);
        return 1;
    }
    JsValue? loaded = js.eval_module((String)source, script_path);
    if (catch loaded)
    {
        io::eprintfn("%s:%d: %s", script_path, js.last_error_line(), js.last_error());
        return 1;
    }
    js.release(loaded);
    if (catch js.get_runtime().drain_microtasks())
    {
        io::eprintfn("uncaught error in a promise job: %s", js.last_error());
        return 1;
    }
    wait_for_timers();
    if (js_loop.is_empty() || loop_renderer == null) return 0;
    if (frames > 0)
    {
        run_offscreen(loop_renderer);
        return 0;
    }
    JsValue? input = js.get(global, "__drieInput");
    if (try input) js_input = input;
    drie::add_event_filter(&input_event);
    loop_renderer.set_animation_loop(&window_frame);
    return 0;
}
