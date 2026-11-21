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
//   out/driejs --frames 3 --png /tmp/shot.png demos/webgl_geometries.js   (time: the frames' own clock)
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
//   light maps  6e6    ..
module main;

import boomkat;
import boomkat::builtins, boomkat::hobject;
import drie, drie::ui;
import sokol;
import std::io, std::time, std::math, std::collections::list, std::compression::deflate;
import drie::readback, drie::imageio, libc, draco;

const double GEOMETRY_BASE = 1_000_000;
const double MATERIAL_BASE = 2_000_000;
const double TEXTURE_BASE = 3_000_000;
const double RENDERER_BASE = 5_000_000;
const double LIGHTMAP_BASE = 6_000_000;
const double VOLUME_BASE = 7_000_000;

// one record of `sync`: handle, visible, then the 16 numbers of the world matrix
const int SYNC_STRIDE = 18;

List{Object3D*} objects;
List{BufferGeometry*} geometries;
List{Material*} materials;
List{Texture*} textures;
List{Renderer*} renderers;
List{ProgressiveLightMap*} lightmaps;
List{IrradianceVolume*} volumes;

JsContext js;
JsValue js_loop;
Renderer* loop_renderer;
String script_path;
String base_dir;
// the working directory, where js/ and vendor/ are
String root;
String png_path;
// --png-width: the picture shrunk to this many pixels across (deflated, with the pipeline cache:
// `just thumbs`)
int png_width;
int frames;
bool jit = true;
bool profile;
// time spent in sg::commit after each frame
long commit_ns;
time::Clock start_clock;
// --png without --profile: performance.now() is the frames' own clock (0 while the script
// starts, then 1/60 s a frame), as three's screenshots are taken, so a script that moves
// things by the time passed (THREE.Timer) shows the same picture however slow the machine
double fixed_clock = -1;
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
fn ProgressiveLightMap* get_lightmap(double id) => @get(lightmaps, id, LIGHTMAP_BASE);
fn IrradianceVolume* get_volume(double id) => @get(volumes, id, VOLUME_BASE);

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

Renderer* first_renderer;
double first_renderer_id;
bool reuse_renderer;

fn void h_renderer(JsCtx ctx, void* udata)
{
    String title = ctx.arg(3).as_string() ?? "three.js";
    // a script that took another's place (switch_script) draws with the renderer the window
    // was opened with, set back to how a new one is
    if (reuse_renderer && first_renderer)
    {
        reuse_renderer = false;
        Renderer* r = first_renderer;
        r.set_clear_color(0x000000);
        r.clear_alpha = 1;
        r.tone_mapping = NO_TONE_MAPPING;
        r.tone_mapping_exposure = 1;
        r.transmission_resolution_scale = 1;
        r.shadow_type = PCF;
        ctx.ret_number(first_renderer_id);
        return;
    }
    Renderer* r = drie::renderer(antialias: bool_arg(ctx, 0, true), width: (int)num_arg(ctx, 1, 1280),
                                        height: (int)num_arg(ctx, 2, 720), title: title);
    r.display_fog = true;
    // (three's apps keep their cameras' aspect themselves)
    r.auto_aspect = false;
    double id = @reg(renderers, r, RENDERER_BASE);
    if (!first_renderer)
    {
        first_renderer = r;
        first_renderer_id = id;
    }
    ctx.ret_number(id);
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

// orthoCamera(left, right, top, bottom, near, far)
fn void h_ortho_camera(JsCtx ctx, void* udata)
{
    OrthographicCamera* c = drie::orthographic_camera((float)num_arg(ctx, 0, -1), (float)num_arg(ctx, 1, 1),
                                                      (float)num_arg(ctx, 2, 1), (float)num_arg(ctx, 3, -1),
                                                      (float)num_arg(ctx, 4, 0.1), (float)num_arg(ctx, 5, 2000));
    ctx.ret_number(reg_object((Object3D*)c));
}

// cameraUpdate(camera, fov, aspect, near, far, zoom, left, right, top, bottom): fov and aspect
// for a perspective camera, left .. bottom for an orthographic one
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
    else if (c.camera_type == ORTHOGRAPHIC)
    {
        OrthographicCamera* p = (OrthographicCamera*)c;
        p.left = (float)num_arg(ctx, 6, p.left);
        p.right = (float)num_arg(ctx, 7, p.right);
        p.top = (float)num_arg(ctx, 8, p.top);
        p.bottom = (float)num_arg(ctx, 9, p.bottom);
    }
    c.update_projection_matrix();
}

// light(kind, color, intensity, distance, decay, angle, penumbra, groundColor)
// kind: 0 ambient, 1 hemisphere, 2 directional, 3 point, 4 spot, 5 rect area (distance and
// decay are its width and height)
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
        case 5: l = (Light*)drie::rect_area_light(color, intensity, (float)num_arg(ctx, 3, 10), (float)num_arg(ctx, 4, 10));
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

// geometry(position, normal, uv, color, index, skinIndex, skinWeight, uv1, tangent): three's attribute
// arrays (Float32Array; index Uint16Array or Uint32Array), any but position may be null. Without normals they
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
    float[] uv1 = floats_arg(ctx, 7);
    if (uv1.len >= (usz)count * 2) foreach (f : uv1[:(usz)count * 2]) g.uv1.push(f);
    float[] tangents = floats_arg(ctx, 8);
    if (tangents.len >= (usz)count * 4) foreach (f : tangents[:(usz)count * 4]) g.tangents.push(f);
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

// texture(source, srgb, wrapS, wrapT, anisotropy, repeatX, repeatY, offsetX, offsetY, flipY,
// rotation, centerX, centerY, width, height, nearest, channel, minFilter, generateMipmaps):
// `source` is a file's path or an encoded image (PNG, JPEG) in a Uint8Array, or with `width`
// and `height` its pixels (RGBA, a Uint8Array or Uint8ClampedArray: a canvas's). minFilter is
// three's (NearestFilter 1003 to LinearMipmapLinearFilter 1008).
fn void h_texture(JsCtx ctx, void* udata)
{
    ColorSpace space = bool_arg(ctx, 1, true) ? SRGB_COLOR_SPACE : NO_COLOR_SPACE;
    Texture* t;
    View bytes;
    int width = (int)num_arg(ctx, 13, 0);
    int height = (int)num_arg(ctx, 14, 0);
    bool raw = width > 0 && height > 0 && view_arg(ctx, 0, &bytes)
        && (bytes.cls == ObjClass.UINT8ARRAY || bytes.cls == ObjClass.UINT8CLAMPEDARRAY);
    if (raw)
    {
        if (bytes.len < (usz)width * (usz)height * 4)
        {
            ctx.throw_error(TYPE, "texture: expected width * height * 4 bytes");
            return;
        }
        t = drie::data_texture(((char*)bytes.data)[:(usz)width * (usz)height * 4], width, height, space);
    }
    else if (view_arg(ctx, 0, &bytes) && bytes.cls == ObjClass.UINT8ARRAY)
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
    t.rotation = (float)num_arg(ctx, 10, 0);
    t.center = { (float)num_arg(ctx, 11, 0), (float)num_arg(ctx, 12, 0) };
    t.nearest = bool_arg(ctx, 15, false);
    t.channel = (char)num_arg(ctx, 16, 0);
    switch ((int)num_arg(ctx, 17, 1008))
    {
        case 1003: t.min_filter = NEAREST;
        case 1004: t.min_filter = NEAREST_MIPMAP_NEAREST;
        case 1005: t.min_filter = NEAREST_MIPMAP_LINEAR;
        case 1006: t.min_filter = LINEAR;
        case 1007: t.min_filter = LINEAR_MIPMAP_NEAREST;
        default: t.min_filter = LINEAR_MIPMAP_LINEAR;
    }
    t.generate_mipmaps = bool_arg(ctx, 18, true);
    ctx.ret_number(@reg(textures, t, TEXTURE_BASE));
}

// textureMipmap(texture, pixels, width, height): the next mip level of a texture made from
// pixels (three's texture.mipmaps[1], [2]...), RGBA bytes half the size of the one before.
// Before the texture is first drawn.
fn void h_texture_mipmap(JsCtx ctx, void* udata)
{
    Texture* t = get_texture(num_arg(ctx, 0, -1));
    View bytes;
    if (t == null || !view_arg(ctx, 1, &bytes)
        || (bytes.cls != ObjClass.UINT8ARRAY && bytes.cls != ObjClass.UINT8CLAMPEDARRAY))
    {
        ctx.throw_error(TYPE, "textureMipmap: expected a texture and a Uint8Array");
        return;
    }
    usz size = (usz)num_arg(ctx, 2, 0) * (usz)num_arg(ctx, 3, 0) * 4;
    if (bytes.len < size) { ctx.throw_error(TYPE, "textureMipmap: expected width * height * 4 bytes"); return; }
    char[] level = mem::new_array(char, (sz)size);
    level[..] = ((char*)bytes.data)[:size];
    char[][] levels = mem::new_array(char[], t.mipmaps.len + 1);
    levels[:t.mipmaps.len] = t.mipmaps[..];
    levels[t.mipmaps.len] = level;
    if (t.mipmaps.len) mem::free(t.mipmaps.ptr);
    t.mipmaps = levels;
}

// textureClone(texture, repeatX, repeatY, offsetX, offsetY, rotation, centerX, centerY,
// channel): another texture of the same image (uploaded once, see drie's Texture.clone),
// placed its own way: three's texture.clone(), as GLTFLoader makes them for
// KHR_texture_transform and texCoord.
fn void h_texture_clone(JsCtx ctx, void* udata)
{
    Texture* source = get_texture(num_arg(ctx, 0, -1));
    if (source == null) { ctx.throw_error(TYPE, "textureClone: not a texture"); return; }
    Texture* t = source.clone();
    t.repeat = { (float)num_arg(ctx, 1, 1), (float)num_arg(ctx, 2, 1) };
    t.offset = { (float)num_arg(ctx, 3, 0), (float)num_arg(ctx, 4, 0) };
    t.rotation = (float)num_arg(ctx, 5, 0);
    t.center = { (float)num_arg(ctx, 6, 0), (float)num_arg(ctx, 7, 0) };
    t.channel = (char)num_arg(ctx, 8, 0);
    ctx.ret_number(@reg(textures, t, TEXTURE_BASE));
}

// cubeTexture(px, nx, py, ny, pz, nz, size): an equirectangular texture (linear light, 4 * size
// by 2 * size) of a cube map's six faces of half floats (Uint16Array) or floats, RGBA, as
// HDRCubeTextureLoader makes them. Looked up as three looks up a CubeTexture (x mirrored).
fn void h_cube_texture(JsCtx ctx, void* udata)
{
    int size = (int)num_arg(ctx, 6, 0);
    float[][6] faces;
    float[] all = mem::new_array(float, (sz)size * size * 4 * 6);
    defer mem::free(all.ptr);
    for (int f = 0; f < 6; f++)
    {
        View v;
        usz count = (usz)size * (usz)size * 4;
        if (size <= 0 || !view_arg(ctx, (uint)f, &v) || (v.cls != ObjClass.UINT16ARRAY && v.cls != ObjClass.FLOAT32ARRAY) || v.len < count)
        {
            ctx.throw_error(TYPE, "cubeTexture: expected six faces of size * size * 4 half floats or floats");
            return;
        }
        faces[f] = all[(sz)count * f:(sz)count];
        if (v.cls == ObjClass.FLOAT32ARRAY)
        {
            libc::memcpy(faces[f].ptr, v.data, count * 4);
        }
        else
        {
            ushort* halfs = (ushort*)v.data;
            for (usz i = 0; i < count; i++) faces[f][i] = drie::half_to_float(halfs[i]);
        }
    }
    char[][6] none;
    Texture* tex = drie::cube_texture(faces, none, size);
    ctx.ret_number(@reg(textures, tex, TEXTURE_BASE));
}

// cubeImages(px, nx, py, ny, pz, nz, srgb): the same of six PNG or JPEG faces (three's
// CubeTextureLoader), each a path or the file's bytes (a Uint8Array).
fn void h_cube_images(JsCtx ctx, void* udata)
{
    char[][6] files;
    bool[6] owned;
    defer for (int f = 0; f < 6; f++) if (owned[f]) mem::free(files[f].ptr);
    for (int f = 0; f < 6; f++)
    {
        View v;
        if (view_arg(ctx, (uint)f, &v) && v.cls == ObjClass.UINT8ARRAY)
        {
            files[f] = ((char*)v.data)[:v.len];
            continue;
        }
        String? path = ctx.arg((uint)f).as_string();
        if (catch path) { ctx.throw_error(TYPE, "cubeImages: expected six paths or Uint8Arrays"); return; }
        char[]? data = file::load(mem, path);
        if (catch data) { ctx.throw_error(TYPE, "cubeImages: cannot read a face"); return; }
        files[f] = data;
        owned[f] = true;
    }
    Texture*? tex = drie::decode_cube_texture(files, bool_arg(ctx, 6, true));
    if (catch tex) { ctx.throw_error(TYPE, "cubeImages: cannot decode the faces (six square images of one size)"); return; }
    ctx.ret_number(@reg(textures, tex, TEXTURE_BASE));
}

// material(kind, color, emissive, specular, shininess, roughness, metalness, opacity,
//          transparent, side, wireframe, map, vertexColors, alphaTest, colorScale,
//          mapEmissive, emissiveIntensity, transmission, thickness, ior, dispersion,
//          attenuationColor, attenuationDistance, specularR, specularG, specularB,
//          specularIntensity)
// kind: 0 basic, 1 lambert, 2 phong, 3 standard, 4 normal, 5 matcap, 6 toon
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
        case 5:
            m = (Material*)drie::mesh_matcap_material(color: color, opacity: opacity, transparent: transparent,
                                                       side: side);
        case 6:
            m = (Material*)drie::mesh_toon_material(color: color, emissive: emissive, opacity: opacity,
                                                     transparent: transparent, side: side);
        case 7:
            m = (Material*)drie::mesh_depth_material(opacity: opacity, transparent: transparent, side: side);
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
    m.transmission = (float)num_arg(ctx, 17, 0);
    m.thickness = (float)num_arg(ctx, 18, 0);
    m.ior = (float)num_arg(ctx, 19, 1.5);
    m.dispersion = (float)num_arg(ctx, 20, 0);
    m.attenuation_color = color_of(num_arg(ctx, 21, 0xffffff));
    m.attenuation_distance = (float)num_arg(ctx, 22, 0);
    // MeshPhysicalMaterial's specularColor (linear, may pass 1) and specularIntensity
    m.specular_color = { (float)num_arg(ctx, 23, 1), (float)num_arg(ctx, 24, 1), (float)num_arg(ctx, 25, 1) };
    m.specular_intensity = (float)num_arg(ctx, 26, 1);
    ctx.ret_number(@reg(materials, m, MATERIAL_BASE));
}

// materialUpdate(material, color, emissive, specular, shininess, roughness, metalness,
// opacity, colorScale, transmission, thickness, ior, dispersion, attenuationColor,
// attenuationDistance, specularR, specularG, specularB, specularIntensity): what three takes
// from a material every frame, when it changed.
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
    m.transmission = (float)num_arg(ctx, 9, 0);
    m.thickness = (float)num_arg(ctx, 10, 0);
    m.ior = (float)num_arg(ctx, 11, 1.5);
    m.dispersion = (float)num_arg(ctx, 12, 0);
    m.attenuation_color = color_of(num_arg(ctx, 13, 0xffffff));
    m.attenuation_distance = (float)num_arg(ctx, 14, 0);
    m.specular_color = { (float)num_arg(ctx, 15, 1), (float)num_arg(ctx, 16, 1), (float)num_arg(ctx, 17, 1) };
    m.specular_intensity = (float)num_arg(ctx, 18, 1);
}

// materialMaps(material, normalMap, normalScaleX, _Y, roughnessMap, metalnessMap, aoMap,
// aoMapIntensity, specularColorMap, specularIntensityMap, transmissionMap, thicknessMap,
// clearcoatMap, clearcoatRoughnessMap, clearcoatNormalMap, clearcoatNormalScaleX, _Y,
// sheenColorMap, sheenRoughnessMap, iridescenceMap, iridescenceThicknessMap, anisotropyMap,
// alphaMap, emissiveMap, envMap, envMapIntensity): a material's maps besides `map`, as three's
// (texture handles; -1: none)
fn void h_material_maps(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m == null) { ctx.throw_error(TYPE, "materialMaps: invalid material"); return; }
    m.normal_map = get_texture(num_arg(ctx, 1, -1));
    m.normal_scale = { (float)num_arg(ctx, 2, 1), (float)num_arg(ctx, 3, 1) };
    m.roughness_map = get_texture(num_arg(ctx, 4, -1));
    m.metalness_map = get_texture(num_arg(ctx, 5, -1));
    m.ao_map = get_texture(num_arg(ctx, 6, -1));
    m.ao_intensity = (float)num_arg(ctx, 7, 1);
    m.specular_color_map = get_texture(num_arg(ctx, 8, -1));
    m.specular_intensity_map = get_texture(num_arg(ctx, 9, -1));
    m.transmission_map = get_texture(num_arg(ctx, 10, -1));
    m.thickness_map = get_texture(num_arg(ctx, 11, -1));
    m.clearcoat_map = get_texture(num_arg(ctx, 12, -1));
    m.clearcoat_roughness_map = get_texture(num_arg(ctx, 13, -1));
    m.clearcoat_normal_map = get_texture(num_arg(ctx, 14, -1));
    m.clearcoat_normal_scale = { (float)num_arg(ctx, 15, 1), (float)num_arg(ctx, 16, 1) };
    m.sheen_color_map = get_texture(num_arg(ctx, 17, -1));
    m.sheen_roughness_map = get_texture(num_arg(ctx, 18, -1));
    m.iridescence_map = get_texture(num_arg(ctx, 19, -1));
    m.iridescence_thickness_map = get_texture(num_arg(ctx, 20, -1));
    m.anisotropy_map = get_texture(num_arg(ctx, 21, -1));
    m.alpha_map = get_texture(num_arg(ctx, 22, -1));
    m.emissive_map = get_texture(num_arg(ctx, 23, -1));
    m.env_map = get_texture(num_arg(ctx, 24, -1));
    // (drie takes 0 for 1)
    m.env_intensity = math::max((float)num_arg(ctx, 25, 1), 1e-6f);
}

// materialLayers(material, clearcoat, clearcoatRoughness, sheen, sheenR, sheenG, sheenB,
// sheenRoughness, iridescence, iridescenceIOR, iridescenceThicknessMin, _Max, anisotropy,
// anisotropyRotation): MeshPhysicalMaterial's clear coat, sheen (its color linear),
// iridescence and anisotropy
fn void h_material_layers(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m == null) { ctx.throw_error(TYPE, "materialLayers: invalid material"); return; }
    m.clearcoat = (float)num_arg(ctx, 1, 0);
    m.clearcoat_roughness = (float)num_arg(ctx, 2, 0);
    m.sheen = (float)num_arg(ctx, 3, 0);
    m.sheen_color = { (float)num_arg(ctx, 4, 0), (float)num_arg(ctx, 5, 0), (float)num_arg(ctx, 6, 0) };
    m.sheen_roughness = (float)num_arg(ctx, 7, 1);
    m.iridescence = (float)num_arg(ctx, 8, 0);
    m.iridescence_ior = (float)num_arg(ctx, 9, 1.3);
    m.iridescence_thickness_min = (float)num_arg(ctx, 10, 100);
    m.iridescence_thickness_max = (float)num_arg(ctx, 11, 400);
    m.anisotropy = (float)num_arg(ctx, 12, 0);
    m.anisotropy_rotation = (float)num_arg(ctx, 13, 0);
}

// floatTexture(data, width, height, flipY, wrapS, wrapT): a texture of RGBA in linear light,
// from a DataTexture of half floats (Uint16Array, HalfFloatType) or floats (Float32Array);
// three's wrapping constants (repeat by default)
fn void h_float_texture(JsCtx ctx, void* udata)
{
    View v;
    int width = (int)num_arg(ctx, 1, 0);
    int height = (int)num_arg(ctx, 2, 0);
    usz count = (usz)width * (usz)height * 4;
    if (!view_arg(ctx, 0, &v) || (v.cls != ObjClass.UINT16ARRAY && v.cls != ObjClass.FLOAT32ARRAY) || v.len < count)
    {
        ctx.throw_error(TYPE, "floatTexture: expected width * height * 4 half floats (Uint16Array) or floats");
        return;
    }
    float[] px = mem::new_array(float, (sz)count);
    defer mem::free(px.ptr);
    if (v.cls == ObjClass.FLOAT32ARRAY)
    {
        libc::memcpy(px.ptr, v.data, count * 4);
    }
    else
    {
        ushort* halfs = (ushort*)v.data;
        for (usz i = 0; i < count; i++) px[i] = drie::half_to_float(halfs[i]);
    }
    Texture* t = drie::float_texture(px, width, height);
    t.flip_y = bool_arg(ctx, 3, false);
    t.wrap_s = wrap_of(num_arg(ctx, 4, 1000));
    t.wrap_t = wrap_of(num_arg(ctx, 5, 1000));
    ctx.ret_number(@reg(textures, t, TEXTURE_BASE));
}

// ----------------------------------------------------------------------------
// Decoded images: createImageBitmap and a canvas's drawImage (js/dom.js)
// ----------------------------------------------------------------------------

struct Decoded
{
    char* pixels;
    int width;
    int height;
}
List{Decoded} decoded;

// imageDecode(bytes): a handle to the decoded PNG or JPEG (-1: it cannot be decoded)
fn void h_image_decode(JsCtx ctx, void* udata)
{
    View v;
    if (!view_arg(ctx, 0, &v) || v.cls != ObjClass.UINT8ARRAY) { ctx.throw_error(TYPE, "imageDecode: expected a Uint8Array"); return; }
    CInt w;
    CInt h;
    char* px = imageio::decode(v.data, v.len, &w, &h);
    if (!px) { ctx.ret_number(-1); return; }
    Decoded d = { px, w, h };
    foreach (i, slot : decoded)
    {
        if (slot.pixels == null)
        {
            decoded[i] = d;
            ctx.ret_number((double)i);
            return;
        }
    }
    decoded.push(d);
    ctx.ret_number((double)(decoded.len() - 1));
}

fn Decoded* decoded_arg(JsCtx ctx)
{
    int i = (int)num_arg(ctx, 0, -1);
    return i >= 0 && i < decoded.len() && decoded[i].pixels ? &decoded[i] : null;
}

// imageSize(handle, 0: width, 1: height)
fn void h_image_size(JsCtx ctx, void* udata)
{
    Decoded* d = decoded_arg(ctx);
    if (!d) { ctx.throw_error(TYPE, "imageSize: invalid image"); return; }
    ctx.ret_number(num_arg(ctx, 1, 0) > 0.5 ? d.height : d.width);
}

// textMetrics(text, family, size, bold, italic, out): out (a Float32Array of 3) = the text's
// width and how far it reaches above and below the baseline, in pixels
fn void h_text_metrics(JsCtx ctx, void* udata)
{
    View v;
    if (!view_arg(ctx, 5, &v) || v.cls != ObjClass.FLOAT32ARRAY || v.len < 3)
    {
        ctx.throw_error(TYPE, "textMetrics: expected a Float32Array of 3 for the metrics");
        return;
    }
    @pool()
    {
        ZString text = (ctx.arg(0).as_string() ?? "").zstr_tcopy();
        ZString family = (ctx.arg(1).as_string() ?? "sans-serif").zstr_tcopy();
        imageio::text_metrics(text, family, (float)num_arg(ctx, 2, 10), bool_arg(ctx, 3, false) ? 1 : 0,
                              bool_arg(ctx, 4, false) ? 1 : 0, (float*)v.data);
    };
}

// textMask(text, family, size, bold, italic, mask, width, height, x, y): the text drawn into
// mask (a Uint8Array of width * height: how much of each pixel it covers), its baseline
// starting at (x, y)
fn void h_text_mask(JsCtx ctx, void* udata)
{
    View v;
    int w = (int)num_arg(ctx, 6, 0);
    int h = (int)num_arg(ctx, 7, 0);
    if (!view_arg(ctx, 5, &v) || v.cls != ObjClass.UINT8ARRAY || w <= 0 || h <= 0 || v.len < (usz)w * (usz)h)
    {
        ctx.throw_error(TYPE, "textMask: expected a Uint8Array of width * height");
        return;
    }
    @pool()
    {
        ZString text = (ctx.arg(0).as_string() ?? "").zstr_tcopy();
        ZString family = (ctx.arg(1).as_string() ?? "sans-serif").zstr_tcopy();
        imageio::text(text, family, (float)num_arg(ctx, 2, 10), bool_arg(ctx, 3, false) ? 1 : 0,
                      bool_arg(ctx, 4, false) ? 1 : 0, (char*)v.data, w, h, (float)num_arg(ctx, 8, 0),
                      (float)num_arg(ctx, 9, 0));
    };
}


// imagePixels(handle, out, width, height): its RGBA, stretched to width x height (bilinear,
// as a canvas draws an image at another size), into the Uint8Array or Uint8ClampedArray
fn void h_image_pixels(JsCtx ctx, void* udata)
{
    Decoded* d = decoded_arg(ctx);
    View v;
    int w = (int)num_arg(ctx, 2, 0);
    int h = (int)num_arg(ctx, 3, 0);
    if (!d || !view_arg(ctx, 1, &v) || (v.cls != ObjClass.UINT8ARRAY && v.cls != ObjClass.UINT8CLAMPEDARRAY)
        || w <= 0 || h <= 0 || v.len < (usz)w * (usz)h * 4)
    {
        ctx.throw_error(TYPE, "imagePixels: expected an image, a byte array and its width and height");
        return;
    }
    char* out = (char*)v.data;
    if (w == d.width && h == d.height)
    {
        libc::memcpy(out, d.pixels, (usz)w * (usz)h * 4);
        return;
    }
    for (int y = 0; y < h; y++)
    {
        float sy = math::clamp(((float)y + 0.5f) * d.height / h - 0.5f, 0, (float)(d.height - 1));
        int y0 = (int)sy;
        int y1 = math::min(y0 + 1, d.height - 1);
        float fy = sy - y0;
        for (int x = 0; x < w; x++)
        {
            float sx = math::clamp(((float)x + 0.5f) * d.width / w - 0.5f, 0, (float)(d.width - 1));
            int x0 = (int)sx;
            int x1 = math::min(x0 + 1, d.width - 1);
            float fx = sx - x0;
            usz r0 = (usz)y0 * (usz)d.width;
            usz r1 = (usz)y1 * (usz)d.width;
            for (usz c = 0; c < 4; c++)
            {
                float a = d.pixels[(r0 + (usz)x0) * 4 + c] * (1 - fx) + d.pixels[(r0 + (usz)x1) * 4 + c] * fx;
                float b = d.pixels[(r1 + (usz)x0) * 4 + c] * (1 - fx) + d.pixels[(r1 + (usz)x1) * 4 + c] * fx;
                out[((usz)y * (usz)w + (usz)x) * 4 + c] = (char)(a * (1 - fy) + b * fy + 0.5f);
            }
        }
    }
}

fn void h_image_free(JsCtx ctx, void* udata)
{
    Decoded* d = decoded_arg(ctx);
    if (!d) return;
    imageio::free_pixels(d.pixels);
    d.pixels = null;
}

// gainmapApply(sdr, gain, out, params): UltraHDRLoader's HDR recovery over RGBA8 pixels,
// the gain map already stretched to the SDR image's size. out is a Uint16Array (half floats)
// or a Float32Array of the same length, its alpha left as it is. params (Float32Array):
// gainMapMin, gainMapMax, offsetSDR, offsetHDR, 1/gamma, weightFactor. The arithmetic is the
// loader's own, in doubles, its sRGB table included.
fn void h_gainmap_apply(JsCtx ctx, void* udata)
{
    View sdr;
    View gain;
    View out;
    float[] p = floats_arg(ctx, 3);
    if (!view_arg(ctx, 0, &sdr) || !view_arg(ctx, 1, &gain) || !view_arg(ctx, 2, &out) || p.len < 6
        || (sdr.cls != ObjClass.UINT8ARRAY && sdr.cls != ObjClass.UINT8CLAMPEDARRAY)
        || (gain.cls != ObjClass.UINT8ARRAY && gain.cls != ObjClass.UINT8CLAMPEDARRAY)
        || (out.cls != ObjClass.UINT16ARRAY && out.cls != ObjClass.FLOAT32ARRAY)
        || gain.len < sdr.len || out.len < sdr.len)
    {
        ctx.throw_error(TYPE, "gainmapApply: expected two byte arrays, an output array and its parameters");
        return;
    }
    double[1024] table;
    foreach (i, &t : table) *t = math::pow((double)i * 0.003717127 + 0.0521327014, 2.4);
    double lo = p[0];
    double range = (double)p[1] - p[0];
    double offset_sdr = p[2];
    double offset_hdr = p[3];
    double inv_gamma = p[4];
    double weight = p[5];
    bool gamma_one = inv_gamma == 1.0;
    // the boost for each of the 256 gain-map values
    double[256] boost;
    foreach (g, &b : boost)
    {
        double v = (double)g * 0.00392156862745098;
        double log_boost = lo + range * (gamma_one ? v : math::pow(v, inv_gamma));
        *b = log_boost * weight == 0.0 ? 1.0 : math::pow(2.0, log_boost * weight);
    }
    char* s = (char*)sdr.data;
    char* g = (char*)gain.data;
    bool half = out.cls == ObjClass.UINT16ARRAY;
    for (usz i = 0; i < sdr.len; i += 4)
    {
        for (usz c = i; c < i + 3; c++)
        {
            double hdr = ((double)s[c] + offset_sdr) * boost[g[c]] - offset_hdr;
            double lin;
            if (hdr < 10.31475) {
                lin = hdr * 0.000303527;
            } else if (hdr < 1024) {
                lin = table[(int)hdr];
            } else {
                lin = math::pow(hdr * 0.003717127 + 0.0521327014, 2.4);
            }
            float f = (float)math::clamp(lin, 0.0, 65504.0);
            if (half) {
                ((ushort*)out.data)[c] = drie::float_to_half(f);
            } else {
                ((float*)out.data)[c] = f;
            }
        }
    }
}

// shadow(object, cast, receive)
fn void h_shadow(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null) return;
    o.cast_shadow = bool_arg(ctx, 1, false);
    o.receive_shadow = bool_arg(ctx, 2, false);
}

// lightUpdate(light, color, intensity, groundColor, width, height) (the last two: a rect area light's)
fn void h_light_update(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != LIGHT) return;
    Light* l = (Light*)o;
    l.color = color_of(num_arg(ctx, 1, 0xffffff));
    l.intensity = (float)num_arg(ctx, 2, 1);
    if (l.light_type == HEMISPHERE) ((HemisphereLight*)l).ground_color = color_of(num_arg(ctx, 3, 0));
    if (l.light_type == RECT_AREA)
    {
        RectAreaLight* ra = (RectAreaLight*)l;
        ra.width = (float)num_arg(ctx, 4, ra.width);
        ra.height = (float)num_arg(ctx, 5, ra.height);
    }
}

// lightChannels(light, channels): drie's Light.channels
fn void h_light_channels(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != LIGHT) return;
    ((Light*)o).channels = (uint)num_arg(ctx, 1, 0);
}

// materialLightChannels(material, channels): drie's Material.light_channels (0: every light)
fn void h_material_light_channels(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m != null) m.light_channels = (uint)num_arg(ctx, 1, 0);
}

// lightShadow(light, cast, mapSize, radius, near, far, extent, normalBias, intensity, bias,
// blurSamples, focus, cascades, splitLambda, cascadeBlend): a directional or spot light's
// shadow. A directional light's is one box `extent` meters from its target to each side (the
// first such light has drie's sun map, the others a tile of the shadow atlas each), or with
// `cascades` above 1 drie's cascaded shadow maps up to `far` from the camera (SunLight's); a
// spot light's covers its cone (its angle, `focus` of it), a point light's all round it (six
// maps).
fn void h_light_shadow(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != LIGHT) return;
    Light* l = (Light*)o;
    LightShadow* shadow;
    switch (l.light_type)
    {
        case DIRECTIONAL:
            DirectionalLight* d = (DirectionalLight*)l;
            shadow = &d.shadow;
            shadow.cascades = math::clamp((int)num_arg(ctx, 12, 1), 1, 4);
            d.shadow_extent = (float)num_arg(ctx, 6, 5);
            if (shadow.cascades > 1)
            {
                shadow.max_distance = (float)num_arg(ctx, 5, 500);
                shadow.split_lambda = (float)num_arg(ctx, 13, 0.5);
                shadow.cascade_blend = (float)num_arg(ctx, 14, 0.1);
            }
        case SPOT:
            shadow = &((SpotLight*)l).shadow;
            shadow.focus = (float)num_arg(ctx, 11, 1);
        case POINT:
            shadow = &((PointLight*)l).shadow;
        default:
            return;
    }
    l.cast_shadow = bool_arg(ctx, 1, false);
    shadow.map_size = (int)num_arg(ctx, 2, 512);
    shadow.radius = (float)num_arg(ctx, 3, 1);
    shadow.near = (float)num_arg(ctx, 4, 0.5);
    shadow.far = (float)num_arg(ctx, 5, 500);
    // three's normalBias is 0 unless set (three draws the back faces into the map instead);
    // drie's default keeps flat surfaces from shadowing themselves in a scene in meters, and is
    // at least a texel of a directional light's box in a larger one. A negative one (three's
    // examples set it against shadows coming loose from what casts them) would push drie's
    // lookups into the surface (drie's is per meter from the light): drie's default then.
    float normal_bias = (float)num_arg(ctx, 7, 0);
    if (normal_bias > 0) {
        shadow.normal_bias = normal_bias;
    } else if (l.light_type == DIRECTIONAL) {
        float texel = 2 * ((DirectionalLight*)l).shadow_extent / math::max(shadow.map_size, 1);
        shadow.normal_bias = math::max(0.03f, texel);
    }
    shadow.intensity = (float)num_arg(ctx, 8, 1);
    shadow.bias = (float)num_arg(ctx, 9, 0);
    shadow.blur_samples = math::max((int)num_arg(ctx, 10, 8), 1);
}

// shadowType(renderer, three's renderer.shadowMap.type): VSMShadowMap (3) is drie's VSM, the
// others its PCF
fn void h_shadow_type(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    if (r == null) return;
    r.shadow_type = (int)num_arg(ctx, 1, 1) == 3 ? ShadowType.VSM : ShadowType.PCF;
}

// materialLightMap(material, lightMap, lightMapIntensity): a texture handle or -1 (the uv set
// is the texture's channel)
fn void h_material_light_map(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m == null) { ctx.throw_error(TYPE, "materialLightMap: invalid material"); return; }
    m.light_map = get_texture(num_arg(ctx, 1, -1));
    m.light_map_intensity = (float)num_arg(ctx, 2, 1);
}

// materialEnv(material, reflectivity, combine, refraction, refractionRatio, specularMap):
// how a basic, lambert or phong material's envMap shows (three's reflectivity, combine,
// CubeRefractionMapping and refractionRatio, specularMap)
fn void h_material_env(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m == null) { ctx.throw_error(TYPE, "materialEnv: invalid material"); return; }
    m.reflectivity = (float)num_arg(ctx, 1, 1);
    int combine = (int)num_arg(ctx, 2, 0);
    m.combine = combine == 1 ? MIX_OPERATION : combine == 2 ? ADD_OPERATION : MULTIPLY_OPERATION;
    m.env_refraction = bool_arg(ctx, 3, false);
    m.refraction_ratio = (float)num_arg(ctx, 4, 0.98);
    m.specular_map = get_texture(num_arg(ctx, 5, -1));
}

// materialSurface(material, bumpMap, bumpScale, matcap, gradientMap, displacementMap,
// displacementScale, displacementBias, depthPacking): three's bumpMap and bumpScale,
// MeshMatcapMaterial's matcap, MeshToonMaterial's gradientMap, displacementMap with its scale
// and bias, and MeshDepthMaterial's depthPacking
fn void h_material_surface(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m == null) { ctx.throw_error(TYPE, "materialSurface: invalid material"); return; }
    m.bump_map = get_texture(num_arg(ctx, 1, -1));
    m.bump_scale = (float)num_arg(ctx, 2, 1);
    m.matcap = get_texture(num_arg(ctx, 3, -1));
    m.gradient_map = get_texture(num_arg(ctx, 4, -1));
    m.displacement_map = get_texture(num_arg(ctx, 5, -1));
    m.displacement_scale = (float)num_arg(ctx, 6, 1);
    m.displacement_bias = (float)num_arg(ctx, 7, 0);
    // (three's BasicDepthPacking is 3200, then RGBA, RGB and RG)
    m.depth_packing = (DepthPacking)math::clamp((int)num_arg(ctx, 8, 3200) - 3200, 0, 3);
}

// materialOutline(material, thickness): drie's Material.outline
fn void h_material_outline(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m == null) { ctx.throw_error(TYPE, "materialOutline: invalid material"); return; }
    m.outline = (float)num_arg(ctx, 1, 0);
}

// materialBlend(material, blending, premultipliedAlpha, blendEquation, blendSrc, blendDst,
// blendEquationAlpha, blendSrcAlpha, blendDstAlpha): three's numbers (NoBlending 0 ..
// CustomBlending 5, AddEquation 100.., ZeroFactor 200..)
fn void h_material_blend(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m == null) { ctx.throw_error(TYPE, "materialBlend: invalid material"); return; }
    switch ((int)num_arg(ctx, 1, 1))
    {
        case 0: m.blending = NO_BLENDING;
        case 2: m.blending = ADDITIVE_BLENDING;
        case 3: m.blending = SUBTRACTIVE_BLENDING;
        case 4: m.blending = MULTIPLY_BLENDING;
        case 5: m.blending = CUSTOM_BLENDING;
        default: m.blending = NORMAL_BLENDING;
    }
    m.premultiplied_alpha = bool_arg(ctx, 2, false);
    m.blend_equation = equation_of(num_arg(ctx, 3, 100));
    m.blend_src = factor_of(num_arg(ctx, 4, 204));
    m.blend_dst = factor_of(num_arg(ctx, 5, 205));
    m.blend_equation_alpha = equation_of(num_arg(ctx, 6, 100));
    m.blend_src_alpha = factor_of(num_arg(ctx, 7, 204));
    m.blend_dst_alpha = factor_of(num_arg(ctx, 8, 205));
}

fn BlendEquation equation_of(double e) => (BlendEquation)math::clamp((int)e - 100, 0, 4);
// (three's constant color factors, 211 and up, have no counterpart: one)
fn BlendFactor factor_of(double f) => (int)f >= 200 && (int)f <= 210 ? (BlendFactor)((int)f - 200) : ONE_FACTOR;

// materialDepth(material, depthTest, depthWrite)
fn void h_material_depth(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m == null) { ctx.throw_error(TYPE, "materialDepth: invalid material"); return; }
    m.depth_test = bool_arg(ctx, 1, true);
    m.depth_write = bool_arg(ctx, 2, true);
    m.tone_mapped = bool_arg(ctx, 3, true);
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

// toneMapping(renderer, three's constant, exposure, transmissionResolutionScale)
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
    r.transmission_resolution_scale = (float)num_arg(ctx, 3, 1);
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

// meshDrawRange(mesh, start, count): drie's Mesh.draw_start and draw_count (0: all)
fn void h_mesh_draw_range(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != MESH) return;
    Mesh* m = (Mesh*)o;
    m.draw_start = (int)num_arg(ctx, 1, 0);
    m.draw_count = (int)num_arg(ctx, 2, 0);
}

// meshBillboard(mesh, kind, sizeAttenuation, rotation, centerX, centerY): Mesh.billboard (kind:
// a Billboard's ordinal, 0 none) and how it is turned
fn void h_mesh_billboard(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != MESH) return;
    Mesh* m = (Mesh*)o;
    int kind = (int)num_arg(ctx, 1, 0);
    if (kind < 0 || kind > Billboard.UPRIGHT.ordinal) kind = 0;
    m.billboard = (Billboard)kind;
    m.billboard_fixed_size = !bool_arg(ctx, 2, true);
    m.billboard_rotation = (float)num_arg(ctx, 3, 0);
    m.billboard_center = { (float)num_arg(ctx, 4, 0.5), (float)num_arg(ctx, 5, 0.5) };
}

// lineDash(material, dashSize, gapSize, scale): LineDashedMaterial's dashes (0, 0: solid)
fn void h_line_dash(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m == null) return;
    m.dash_size = (float)num_arg(ctx, 1, 0);
    m.gap_size = (float)num_arg(ctx, 2, 0);
    m.dash_scale = (float)num_arg(ctx, 3, 1);
}

// line(geometry, material, mode): a drie Line; mode 1 LineSegments, 2 Line, 3 LineLoop
fn void h_line(JsCtx ctx, void* udata)
{
    BufferGeometry* g = get_geometry(num_arg(ctx, 0, -1));
    if (g == null) { ctx.throw_error(TYPE, "line: invalid geometry"); return; }
    Material* m = get_material(num_arg(ctx, 1, -1));
    Object3D* o;
    switch ((int)num_arg(ctx, 2, 2))
    {
        case 1: o = (Object3D*)drie::line_segments(g, m);
        case 3: o = (Object3D*)drie::line_loop(g, m);
        default: o = (Object3D*)drie::line(g, m);
    }
    ctx.ret_number(reg_object(o));
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
    m.instance_count = count;
    drie::Mat4* src = (drie::Mat4*)mats.ptr;
    for (int i = 0; i < count; i++) m.set_matrix_at(i, src[i]);
}

// instanceColors(mesh, colors, count): three's instanceColor.array, 3 floats in linear light
// per instance (white past `count`)
fn void h_instance_colors(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    float[] colors = floats_arg(ctx, 1);
    int count = (int)num_arg(ctx, 2, 0);
    if (o == null || !instanced.contains((InstancedMesh*)o)) { ctx.throw_error(TYPE, "instanceColors: not an instanced mesh"); return; }
    if ((usz)count * 3 > colors.len) { ctx.throw_error(RANGE, "instanceColors: count exceeds the colors"); return; }
    InstancedMesh* m = (InstancedMesh*)o;
    m.instance_colors.clear();
    m.instance_colors.allocator = mem;
    m.instance_colors.push_all(colors[:(usz)count * 3]);
    for (int i = count; i < m.instance_count; i++) m.instance_colors.push_all({ 1, 1, 1 });
    m.instance_colors_uploaded = false;
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

// ----------------------------------------------------------------------------
// Progressive light maps (js/progressive_lightmap.js)
// ----------------------------------------------------------------------------

// lightmap(renderer, res, scene): drie's ProgressiveLightMap, baked with the lights of
// `scene` (the mirror of the three Scene that three's keeps its lights in). Needs the GPU:
// called from the animation loop, not while the script loads.
fn void h_lightmap(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    Object3D* s = get_object(num_arg(ctx, 2, -1));
    if (r == null || s == null || s.kind != SCENE) { ctx.throw_error(TYPE, "lightmap: invalid argument"); return; }
    ProgressiveLightMap* p = drie::progressive_lightmap(r, (int)num_arg(ctx, 1, 1024));
    // (the scene it made for its lights stays empty)
    p.scene = (Scene*)s;
    ctx.ret_number(@reg(lightmaps, p, LIGHTMAP_BASE));
}

// lightmapTexture(lightmap, i): a texture handle for its map i (0: the one showDebugLightmap
// shows, 1: the objects' light map, on their uv1)
fn void h_lightmap_texture(JsCtx ctx, void* udata)
{
    ProgressiveLightMap* p = get_lightmap(num_arg(ctx, 0, -1));
    if (p == null) { ctx.throw_error(TYPE, "lightmapTexture: invalid light map"); return; }
    ctx.ret_number(@reg(textures, p.texture((int)num_arg(ctx, 1, 0)), TEXTURE_BASE));
}

// lightmapObjects(lightmap, handles, count): the meshes baked into it, `count` object handles
// from a Float32Array, in place of the last ones. Their uv1 is laid out already (three's
// packing, done in JS).
fn void h_lightmap_objects(JsCtx ctx, void* udata)
{
    ProgressiveLightMap* p = get_lightmap(num_arg(ctx, 0, -1));
    float[] ids = floats_arg(ctx, 1);
    usz count = (usz)num_arg(ctx, 2, 0);
    if (p == null || count > ids.len) { ctx.throw_error(TYPE, "lightmapObjects: invalid argument"); return; }
    List{Object3D*} found;
    found.init(tmem);
    for (usz i = 0; i < count; i++)
    {
        Object3D* o = get_object(ids[i]);
        if (o != null) found.push(o);
    }
    p.objects.clear();
    p.add_objects(found.array_view(), pack: false);
}

// lightmapUpdate(lightmap, camera, blendWindow, blurEdges): one more sample baked in
fn void h_lightmap_update(JsCtx ctx, void* udata)
{
    ProgressiveLightMap* p = get_lightmap(num_arg(ctx, 0, -1));
    Object3D* c = get_object(num_arg(ctx, 1, -1));
    if (p == null || c == null || c.kind != CAMERA) { ctx.throw_error(TYPE, "lightmapUpdate: invalid argument"); return; }
    p.update((Camera*)c, (float)num_arg(ctx, 2, 100), bool_arg(ctx, 3, true));
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

// ----------------------------------------------------------------------------
// Light probe grids (js/light_probe_grid.js): drie's irradiance volumes
// ----------------------------------------------------------------------------

// irradianceGrid(minX, minY, minZ, maxX, maxY, maxZ, nx, ny, nz): a volume handle, dark
// until baked
fn void h_irradiance_grid(JsCtx ctx, void* udata)
{
    IrradianceVolume* v = drie::irradiance_grid(
        { (float)num_arg(ctx, 0, -0.5), (float)num_arg(ctx, 1, -0.5), (float)num_arg(ctx, 2, -0.5) },
        { (float)num_arg(ctx, 3, 0.5), (float)num_arg(ctx, 4, 0.5), (float)num_arg(ctx, 5, 0.5) },
        (int)num_arg(ctx, 6, 1), (int)num_arg(ctx, 7, 1), (int)num_arg(ctx, 8, 1));
    ctx.ret_number(@reg(volumes, v, VOLUME_BASE));
}

// A bake that LightProbeGridWebGL.bake asked for, done at the next render (drie needs the
// GPU running for it).
struct Bake
{
    IrradianceVolume* volume;
    Scene* scene;
    int face_size;
    float near;
    float far;
    int bounces;
}
List{Bake} bakes;

// bakeIrradiance(volume, scene, faceSize, near, far, bounces): Renderer.bake_irradiance at
// the next render
fn void h_bake_irradiance(JsCtx ctx, void* udata)
{
    IrradianceVolume* v = get_volume(num_arg(ctx, 0, -1));
    Object3D* o = get_object(num_arg(ctx, 1, -1));
    if (v == null || o == null || o.kind != SCENE) { ctx.throw_error(TYPE, "bakeIrradiance: invalid argument"); return; }
    bakes.push({ v, (Scene*)o, (int)num_arg(ctx, 2, 8), (float)num_arg(ctx, 3, 0.1), (float)num_arg(ctx, 4, 100),
                 (int)num_arg(ctx, 5, 0) });
}

fn void run_bakes(Renderer* r)
{
    foreach (b : bakes) r.bake_irradiance(b.scene, b.volume, b.face_size, b.near, b.far, b.bounces);
    bakes.clear();
}

// sceneIrradiance(scene, v0, v1, v2, v3): the volumes that light the scene (its probe grids,
// -1 past the last), each one in front of the ones before it
fn void h_scene_irradiance(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != SCENE) { ctx.throw_error(TYPE, "sceneIrradiance: not a scene"); return; }
    Scene* s = (Scene*)o;
    if (!s.irradiance_volumes.len) s.irradiance_volumes = mem::new_array(IrradianceVolume*, drie::MAX_IRRADIANCE_VOLUMES - 1);
    s.irradiance = get_volume(num_arg(ctx, 1, -1));
    for (uint i = 0; i < drie::MAX_IRRADIANCE_VOLUMES - 1; i++) s.irradiance_volumes[i] = get_volume(num_arg(ctx, 2 + i, -1));
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

// backgroundTexture(scene, texture or -1, blurriness, intensity): an equirectangular map
// behind everything (three's scene.background as a texture)
fn void h_background_texture(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != SCENE) { ctx.throw_error(TYPE, "backgroundTexture: not a scene"); return; }
    Scene* s = (Scene*)o;
    s.background_texture = get_texture(num_arg(ctx, 1, -1));
    s.background_blurriness = (float)num_arg(ctx, 2, 0);
    s.background_intensity = (float)num_arg(ctx, 3, 1);
    s.background_flat = bool_arg(ctx, 4, false);
}

// backgroundMaterial(scene, material or -1): a material drawn behind everything, on a sphere
// around the camera (three's scene.backgroundNode, as a node material)
fn void h_background_material(JsCtx ctx, void* udata)
{
    Object3D* o = get_object(num_arg(ctx, 0, -1));
    if (o == null || o.kind != SCENE) { ctx.throw_error(TYPE, "backgroundMaterial: not a scene"); return; }
    ((Scene*)o).background_material = get_material(num_arg(ctx, 1, -1));
}

// textureUv(texture, repeatX, repeatY, offsetX, offsetY, rotation, centerX, centerY): its uv
// placement, changed (three's repeat, offset, rotation and center need no needsUpdate)
fn void h_texture_uv(JsCtx ctx, void* udata)
{
    Texture* t = get_texture(num_arg(ctx, 0, -1));
    if (t == null) { ctx.throw_error(TYPE, "textureUv: not a texture"); return; }
    t.repeat = { (float)num_arg(ctx, 1, 1), (float)num_arg(ctx, 2, 1) };
    t.offset = { (float)num_arg(ctx, 3, 0), (float)num_arg(ctx, 4, 0) };
    t.rotation = (float)num_arg(ctx, 5, 0);
    t.center = { (float)num_arg(ctx, 6, 0), (float)num_arg(ctx, 7, 0) };
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
    // (before the first frame: nothing to draw with yet; the captures and bakes wait for it)
    if (!gpu_ready) return;
    time::Clock t = time::clock::now();
    if (captures.len()) run_captures(r);
    if (bakes.len()) run_bakes(r);
    Camera* cam = (Camera*)co;
    follow_skies({ cam.matrix_world[3][0], cam.matrix_world[3][1], cam.matrix_world[3][2] }, cam.far);
    r.render((Scene*)so, cam);
    render_ns += (long)t.to_now();
}

// renderEnvironment(renderer, material, into or -1, width, diffuse): `material` drawn into an
// equirectangular environment map (drie's Renderer.render_environment: three's
// scene.environmentNode), into the one given (a map this made before) or a new one; -1 before
// the first frame (nothing to draw with yet)
fn void h_render_environment(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    Material* m = get_material(num_arg(ctx, 1, -1));
    if (r == null || m == null) { ctx.throw_error(TYPE, "renderEnvironment: invalid argument"); return; }
    if (!gpu_ready) { ctx.ret_number(-1); return; }
    double into = num_arg(ctx, 2, -1);
    Texture* t = r.render_environment(m, into < 0 ? null : get_texture(into), (int)num_arg(ctx, 3, 512), bool_arg(ctx, 4, true));
    ctx.ret_number(into < 0 ? @reg(textures, t, TEXTURE_BASE) : into);
}

// renderProbe(renderer, scene, x, y, z, into or -1, faceSize, near, far, diffuse): the scene
// drawn in every direction from (x, y, z) into an environment map (drie's Renderer.render_probe:
// three's CubeCamera.update), into the one given (a map this made before) or a new one; -1
// before the first frame (nothing to draw with yet)
fn void h_render_probe(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    Object3D* o = get_object(num_arg(ctx, 1, -1));
    if (r == null || o == null || o.kind != SCENE) { ctx.throw_error(TYPE, "renderProbe: invalid argument"); return; }
    if (!gpu_ready) { ctx.ret_number(-1); return; }
    double into = num_arg(ctx, 5, -1);
    Vector3 p = { (float)num_arg(ctx, 2, 0), (float)num_arg(ctx, 3, 0), (float)num_arg(ctx, 4, 0) };
    float far = (float)num_arg(ctx, 8, 1000);
    // (the skies around the probe; render puts them back around the camera)
    follow_skies(p, far);
    Texture* t = r.render_probe((Scene*)o, p, into < 0 ? null : get_texture(into), (int)num_arg(ctx, 6, 256),
                                (float)num_arg(ctx, 7, 0.1), far, bool_arg(ctx, 9, true));
    ctx.ret_number(into < 0 ? @reg(textures, t, TEXTURE_BASE) : into);
}

// textureLightProbe(texture, out): out (a Float32Array of 27) = the environment map's light on
// surfaces as three's LightProbe has it (its sh, rgb by coefficient), the sun in it too; 1, or
// 0 when the map has none yet
fn void h_texture_light_probe(JsCtx ctx, void* udata)
{
    Texture* t = get_texture(num_arg(ctx, 0, -1));
    View v;
    if (t == null || !view_arg(ctx, 1, &v) || v.cls != ObjClass.FLOAT32ARRAY || v.len < 27)
    {
        ctx.throw_error(TYPE, "textureLightProbe: expected a texture and a Float32Array of 27");
        return;
    }
    if (!t.has_env_sh) { ctx.ret_number(0); return; }
    // three's irradiance of coefficients c (SphericalHarmonics3.getIrradianceAt) against drie's
    // of e (EnvLight.irradiance), term by term: c[k] = e[k] / these
    float[9] scale = { 0.886227f, 1.023328f, 1.023328f, 1.023328f, 0.858086f, 0.858086f, 0.247708f, 0.858086f, 0.429043f };
    Vector3[9] e = t.env_light.harmonics();
    float* out = (float*)v.data;
    for (int k = 0; k < 9; k++) {
        Vector3 c = e[k] / scale[k];
        out[k * 3] = c.x;
        out[k * 3 + 1] = c.y;
        out[k * 3 + 2] = c.z;
    }
    ctx.ret_number(1);
}

fn void h_set_clear_color(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    if (r) r.set_clear_color(color_of(num_arg(ctx, 1, 0)), (float)num_arg(ctx, 2, 1));
}

fn void h_set_size(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    if (r) r.set_size((int)num_arg(ctx, 1, r.width), (int)num_arg(ctx, 2, r.height));
}

// viewState(renderer, autoClear, viewport x, y, width, height, scissor x, y, width, height,
// scissorTest): three's, in CSS pixels from the bottom left (the target's pixels may be more)
fn void h_view_state(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    if (r == null) return;
    float scale = r.render_target ? (float)r.render_target.width / r.width : sapp::isvalid() ? sapp::widthf() / r.width : 1;
    r.auto_clear = bool_arg(ctx, 1, true);
    for (int i = 0; i < 4; i++)
    {
        r.viewport[i] = (int)math::round(num_arg(ctx, (uint)(2 + i), 0) * scale);
        r.scissor[i] = (int)math::round(num_arg(ctx, (uint)(6 + i), 0) * scale);
    }
    r.scissor_test = bool_arg(ctx, 10, false);
}

// clear(renderer): three's renderer.clear()
fn void h_clear(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    if (r) r.clear();
}

// info(renderer, what): what the last render drew, 0 draw calls, 1 triangles, 2 line segments
fn void h_info(JsCtx ctx, void* udata)
{
    Renderer* r = get_renderer(num_arg(ctx, 0, -1));
    if (r == null) { ctx.ret_number(0); return; }
    switch ((int)num_arg(ctx, 1, 0))
    {
        case 0: ctx.ret_number(r.info.calls);
        case 1: ctx.ret_number(r.info.triangles);
        default: ctx.ret_number(r.info.lines);
    }
}

// Milliseconds from an arbitrary start (boomkat has no `performance`).
fn void h_now(JsCtx ctx, void* udata)
{
    if (fixed_clock >= 0) { ctx.ret_number(fixed_clock); return; }
    ctx.ret_number((double)(long)start_clock.to_now() / 1e6);
}

// Whether now() is the frames' own clock (see fixed_clock).
fn void h_fixed_clock(JsCtx ctx, void* udata)
{
    ctx.ret_bool(fixed_clock >= 0);
}

// stats(fps, fpsMin, fpsMax, ms, msMin, msMax), from js/stats.js each frame (fps -1 until
// the first second has passed). The first call puts the panel over the picture.
double[6] stats_values;
bool stats_shown;
// (false once another script takes this one's place: switch_script)
bool stats_live;

fn void h_stats(JsCtx ctx, void* udata)
{
    for (uint i = 0; i < stats_values.len; i++) stats_values[i] = num_arg(ctx, i, -1);
    stats_live = true;
    if (stats_shown) return;
    stats_shown = true;
    drie::add_overlay(&draw_stats);
}

// Like three's Stats panel: frames per second and milliseconds per frame in the top left
// corner, each with the lowest and highest so far.
fn void draw_stats()
{
    if (!stats_live) return;
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

// ----------------------------------------------------------------------------
// For a script that picks others (demos/index.js): text over the picture, a folder's files,
// and another script run in this one's place
// ----------------------------------------------------------------------------

struct Label
{
    String text;
    float x;
    float y;
    char[3] color;
}

List{Label} labels;
bool labels_shown;

// label(text, x, y, r, g, b): a line of text over this frame's picture, its top left corner at
// x, y in window points, in the stats panel's size (a glyph 8 by 8 canvas units, the canvas
// about 720 units high); r, g, b from 0 to 255
fn void h_label(JsCtx ctx, void* udata)
{
    String? text = ctx.arg(0).as_string();
    // (offscreen there is no window to write over)
    if (catch text) return;
    if (!sapp::isvalid()) return;
    if (!labels.is_initialized()) labels.init(mem);
    labels.push({
        text.copy(mem), (float)num_arg(ctx, 1, 0), (float)num_arg(ctx, 2, 0),
        { (char)num_arg(ctx, 3, 255), (char)num_arg(ctx, 4, 255), (char)num_arg(ctx, 5, 255) }
    });
    if (labels_shown) return;
    labels_shown = true;
    drie::add_overlay(&draw_labels);
}

// labelScale(): how many window points a canvas unit of the labels is (a glyph is 8 units)
fn void h_label_scale(JsCtx ctx, void* udata)
{
    if (!sapp::isvalid()) { ctx.ret_number(1); return; }
    ctx.ret_number((double)(math::max(1.0f, math::round(sapp::heightf() / 720)) / sapp::dpi_scale()));
}

fn void draw_labels()
{
    ui::setup();
    float scale = math::max(1.0f, math::round(sapp::heightf() / 720));
    // (window points to canvas units)
    float k = sapp::dpi_scale() / scale;
    sdtx::set_context(sdtx::default_context());
    sdtx::canvas(sapp::widthf() / scale, sapp::heightf() / scale);
    // (the stats panel's Oric font has a pound sign where the underscore goes)
    sdtx::font(ui::FONT_CPC);
    sdtx::origin(0, 0);
    foreach (l : labels)
    {
        sdtx::pos(l.x * k / 8, l.y * k / 8);
        sdtx::color3b(l.color[0], l.color[1], l.color[2]);
        sdtx::puts(l.text.zstr_tcopy());
        l.text.free(mem);
    }
    labels.clear();
    sdtx::draw();
}

// files(folder): the names of its files, one a line
fn void h_files(JsCtx ctx, void* udata)
{
    String? folder = ctx.arg(0).as_string();
    if (catch folder) { ctx.throw_error(TYPE, "files: expected a folder"); return; }
    PathList? found = path::ls(tmem, path::tnew(folder) ?? path::tnew(".")!!);
    if (catch found) { ctx.ret_string(""); return; }
    DString names = dstring::temp();
    foreach (f : found)
    {
        names.append((String)f);
        names.append("\n");
    }
    ctx.ret_string(names.str_view());
}

// The script to load in this one's place at the next frame (run(), or Escape going back), and
// the one Escape goes back to ("": none).
String pending_script;
String back_path;

// The renderer the pipeline cache's natives work on: the window's.
fn Renderer* cache_renderer() => first_renderer ? first_renderer : loop_renderer;

// loadPipelineCache(path): starts compiling, in the background, the pipelines a pipeline cache
// lists (drie's Renderer.load_pipeline_cache): a text file (savePipelineCache), or a PNG with
// one deflated in a drPc chunk (a thumbnail from `just thumbs`). Returns how many it lists that aren't
// made yet; 0 for a file without one.
fn void h_load_pipeline_cache(JsCtx ctx, void* udata)
{
    String? path = ctx.arg(0).as_string();
    if (catch path) { ctx.throw_error(TYPE, "loadPipelineCache: expected a path"); return; }
    Renderer* r = cache_renderer();
    if (!r) { ctx.ret_number(0); return; }
    @pool()
    {
        char[]? bytes = file::load_temp(path);
        if (catch bytes) { ctx.ret_number(0); return; }
        char[] text = bytes;
        char[] chunk;
        if (drie::png::find_chunk(bytes, "drPc", &chunk)) text = deflate::decompress(tmem, chunk) ?? { };
        ctx.ret_number(r.load_pipeline_cache((String)text));
    };
}

// savePipelineCache(path): the pipelines made so far, as a pipeline cache (a text file).
fn void h_save_pipeline_cache(JsCtx ctx, void* udata)
{
    String? path = ctx.arg(0).as_string();
    if (catch path) { ctx.throw_error(TYPE, "savePipelineCache: expected a path"); return; }
    Renderer* r = cache_renderer();
    ctx.ret_bool(r && r.save_pipeline_cache(path));
}

// pipelineCacheProgress(): how far the loaded pipeline caches have compiled, 0 to 1 (1: all,
// or none loaded)
fn void h_pipeline_cache_progress(JsCtx ctx, void* udata)
{
    Renderer* r = cache_renderer();
    if (!r) { ctx.ret_number(1); return; }
    int[2] n = r.compiling();
    ctx.ret_number(n[1] ? (double)(n[1] - n[0]) / n[1] : 1);
}

// run(script): the script in this one's place, in the same window, from the next frame on;
// Escape there comes back to this one
fn void h_run(JsCtx ctx, void* udata)
{
    String? script = ctx.arg(0).as_string();
    if (catch script) { ctx.throw_error(TYPE, "run: expected a script"); return; }
    pending_script = script.copy(mem);
    back_path = script_path;
}

// Evaluates the module at `path` and lets what it started while setting up settle (promises,
// timers before there is an animation loop).
fn bool load_script(String path)
{
    char[]? source = file::load_temp(path);
    if (catch source)
    {
        io::eprintfn("cannot read %s", path);
        return false;
    }
    JsValue? loaded = js.eval_module((String)source, path);
    if (catch loaded)
    {
        // (a syntax error has a line; an error the module threw, a stack)
        if (js.last_error_line() > 0)
        {
            io::eprintfn("%s:%d: %s", path, js.last_error_line(), js.last_error());
        }
        else
        {
            report_uncaught(path);
        }
        return false;
    }
    js.release(loaded);
    if (!run_jobs()) return false;
    wait_for_timers();
    return true;
}

// Puts the script at `path` in the running one's place: the running one's animation loop,
// timers, event listeners and page (js/dom.js, __drieReset) go, its stats panel and the
// pointer lock too; what it made in drie stays in memory. False if the new one fails or
// starts no animation loop.
fn bool switch_script(String path)
{
    if (!js_loop.is_empty()) js.release(js_loop);
    js_loop = {};
    loop_renderer = null;
    clear_timers();
    stats_live = false;
    if (sapp::isvalid()) sapp::lock_mouse(false);
    JsValue? global = js.global();
    if (try global)
    {
        JsValue? reset = js.get(global, "__drieReset");
        if (try reset)
        {
            JsValue? outcome = js.call(reset, {});
            if (try outcome) js.release(outcome);
            js.release(reset);
        }
        js.release(global);
    }
    reuse_renderer = true;
    script_path = path;
    return load_script(path) && !js_loop.is_empty() && loop_renderer != null;
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

// Whether the GPU is up: from the first frame on. A script that renders while it loads (once,
// with no animation loop) is drawn by the frames, which draw its last picture again.
bool gpu_ready;

fn bool js_frame(double time_ms)
{
    gpu_ready = true;
    if (fixed_clock >= 0) fixed_clock = time_ms;
    if (pending_script.len)
    {
        String next = pending_script;
        pending_script = "";
        if (!switch_script(next))
        {
            // (back to the script that asked for it, if there is one)
            String back = back_path;
            back_path = "";
            if (!back.len || !switch_script(back)) return false;
        }
    }
    fire_timers();
    JsValue? t = js.number(time_ms);
    if (catch t) return false;
    defer js.release(t);
    JsValue? outcome = js.call(js_loop, { t });
    if (catch outcome)
    {
        report_uncaught("the animation loop");
        return false;
    }
    js.release(outcome);
    return run_jobs();
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
    // (--back: Escape goes back, unless it is letting go of the pointer)
    bool back = e.type == KEY_DOWN && e.key_code == ESCAPE && back_path.len && !sapp::mouse_locked();
    defer if (back)
    {
        pending_script = back_path;
        back_path = "";
    }
    double scale = sapp::dpi_scale();
    double[6] args = { kind, e.mouse_x / scale, e.mouse_y / scale, a, dx, dy };
    JsValue[6] values;
    for (int i = 0; i < 6; i++) values[i] = js.number(args[i]) ?? {};
    JsValue? outcome = js.call(js_input, values[..]);
    if (catch outcome)
    {
        report_uncaught("an input event");
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
    // (what it draws with is listed as the window's, for the pipeline cache in --png-width's PNG)
    r.screen_stand_in = target;
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
    if (png_width > 0 && png_width < r.width)
    {
        // a thumbnail: shrunk, deflated, the pipelines it drew with in a drPc chunk (for
        // loadPipelineCache)
        int h = (int)((long)r.height * png_width / r.width);
        char[] small = shrink(pixels, r.width, r.height, png_width, h);
        defer free(small);
        @pool()
        {
            // (deflated: a node program's text can run to hundreds of KB)
            char[] cache = deflate::compress(tmem, r.pipeline_cache(tmem)) ?? { };
            bool saved = drie::png::write(png_path, small, png_width, h, extra: { { "drPc", cache } }, compress: true);
            io::printfn(saved ? "saved %s" : "could not save %s", png_path);
        };
        return;
    }
    io::printfn(write_png(png_path, pixels, r.width, r.height) ? "saved %s" : "could not save %s", png_path);
}

// RGBA pixels, w × h, shrunk to sw × sh: each the mean of the pixels it covers.
fn char[] shrink(char[] px, int w, int h, int sw, int sh)
{
    char[] out = mem::new_array(char, (usz)sw * (usz)sh * 4);
    for (int y = 0; y < sh; y++)
    {
        int y0 = (int)((long)y * h / sh);
        int y1 = max(y0 + 1, (int)((long)(y + 1) * h / sh));
        for (int x = 0; x < sw; x++)
        {
            int x0 = (int)((long)x * w / sw);
            int x1 = max(x0 + 1, (int)((long)(x + 1) * w / sw));
            uint[4] sum;
            for (int sy = y0; sy < y1; sy++)
            {
                for (int sx = x0; sx < x1; sx++)
                {
                    usz o = ((usz)sy * (usz)w + (usz)sx) * 4;
                    for (usz c = 0; c < 4; c++) sum[c] += px[o + c];
                }
            }
            uint n = (uint)((y1 - y0) * (x1 - x0));
            usz o = ((usz)y * (usz)sw + (usz)x) * 4;
            for (usz c = 0; c < 4; c++) out[o + c] = (char)((sum[c] + n / 2) / n);
        }
    }
    return out;
}

// ----------------------------------------------------------------------------
// Node materials: js/drie_nodes.js rebuilds three's TSL graphs as drie's (nodes.c3), a node
// at a time; a node is its number (Node.id)
// ----------------------------------------------------------------------------

fn Node node_arg(JsCtx ctx, uint i) => { (uint)num_arg(ctx, i, 0) };

fn NodeType type_arg(JsCtx ctx, uint i) @local
{
    int t = (int)num_arg(ctx, i, 0);
    if (t < 0 || t > NodeType.VOID.ordinal) t = 0;
    return (NodeType)t;
}

// nodeConst(type, x, y, z, w) (an int's or a uint's: whole numbers)
fn void h_node_const(JsCtx ctx, void* udata)
{
    NodeType t = type_arg(ctx, 0);
    NodeScalar s = drie::scalar_of(t);
    if (s == INT || s == UINT)
    {
        uint[4] bits;
        for (uint i = 0; i < 4; i++) bits[i] = (uint)(long)num_arg(ctx, i + 1, 0);
        ctx.ret_number(drie::node_const_bits(t, bits).id);
        return;
    }
    float[16] v;
    for (uint i = 0; i < 4; i++) v[i] = (float)num_arg(ctx, i + 1, 0);
    ctx.ret_number(drie::node_const(t, v).id);
}

// nodeUniform(type): its value from nodeSet
fn void h_node_uniform(JsCtx ctx, void* udata)
{
    ctx.ret_number(drie::add_node({ .op = UNIFORM, .type = type_arg(ctx, 0) }).id);
}

// nodeSet(node, x, y, z, w), or nodeSet(node, Float32Array of 16) for a matrix
fn void h_node_set(JsCtx ctx, void* udata)
{
    Node n = node_arg(ctx, 0);
    float[] m = floats_arg(ctx, 1);
    if (m.len >= 16)
    {
        drie::Mat4 mat;
        for (int c = 0; c < 4; c++)
        {
            for (int r = 0; r < 4; r++) mat[c][r] = m[c * 4 + r];
        }
        n.set_matrix(mat);
        return;
    }
    n.set((float)num_arg(ctx, 1, 0), (float)num_arg(ctx, 2, 0), (float)num_arg(ctx, 3, 0), (float)num_arg(ctx, 4, 0));
}

// nodeBuiltin(kind, type): a NodeBuiltin
fn void h_node_builtin(JsCtx ctx, void* udata)
{
    int kind = (int)num_arg(ctx, 0, 0);
    if (kind < 0 || kind > NodeBuiltin.INSTANCE_INDEX.ordinal) { ctx.throw_error(TYPE, "nodeBuiltin: no such builtin"); return; }
    ctx.ret_number(drie::add_node({ .op = BUILTIN, .type = type_arg(ctx, 1), .kind = (char)kind }).id);
}

Texture* white_texture;

// nodeTexture(texture, uv, level): uv 0 for the mesh's, level 0 for the pixel's footprint; a
// texture not loaded yet (-1) reads white
fn void h_node_texture(JsCtx ctx, void* udata)
{
    Texture* t = get_texture(num_arg(ctx, 0, -1));
    if (t == null)
    {
        if (!white_texture) white_texture = drie::data_texture({ 255, 255, 255, 255 }, 1, 1);
        t = white_texture;
    }
    ctx.ret_number(nodes::texture(t, node_arg(ctx, 1), node_arg(ctx, 2)).id);
}

// nodeTextureSize(texture): its size in texels, a vec2 (a texture not loaded yet: 1 x 1)
fn void h_node_texture_size(JsCtx ctx, void* udata)
{
    Texture* t = get_texture(num_arg(ctx, 0, -1));
    if (t == null)
    {
        if (!white_texture) white_texture = drie::data_texture({ 255, 255, 255, 255 }, 1, 1);
        t = white_texture;
    }
    ctx.ret_number(nodes::texture_size(t).id);
}

// nodeVarying(x)
fn void h_node_varying(JsCtx ctx, void* udata) => ctx.ret_number(nodes::varying(node_arg(ctx, 0)).id);

// nodeBinary(op, a, b): a NodeBinary
fn void h_node_binary(JsCtx ctx, void* udata)
{
    int op = (int)num_arg(ctx, 0, 0);
    if (op < 0 || op > NodeBinary.XOR.ordinal) { ctx.throw_error(TYPE, "nodeBinary: no such operator"); return; }
    ctx.ret_number(drie::node_binary((NodeBinary)op, node_arg(ctx, 1), node_arg(ctx, 2)).id);
}

// nodeCall(fn, a, b, c): a NodeFn
fn void h_node_call(JsCtx ctx, void* udata)
{
    int f = (int)num_arg(ctx, 0, 0);
    if (f < 0 || f > NodeFn.FACEFORWARD.ordinal) { ctx.throw_error(TYPE, "nodeCall: no such function"); return; }
    ctx.ret_number(drie::node_call((NodeFn)f, node_arg(ctx, 1), node_arg(ctx, 2), node_arg(ctx, 3)).id);
}

// nodeSelect(cond, a, b)
fn void h_node_select(JsCtx ctx, void* udata)
{
    ctx.ret_number(nodes::select(node_arg(ctx, 0), node_arg(ctx, 1), node_arg(ctx, 2)).id);
}

// nodeSwizzle(a, "xyz")
fn void h_node_swizzle(JsCtx ctx, void* udata)
{
    String? s = ctx.arg(1).as_string();
    if (catch s) { ctx.throw_error(TYPE, "nodeSwizzle: no components"); return; }
    ctx.ret_number(node_arg(ctx, 0).swizzle(s).id);
}

// nodeJoin(type, a, b, c, d): a vector of the parts given (0: none)
fn void h_node_join(JsCtx ctx, void* udata)
{
    Node[4] parts;
    int n;
    for (uint i = 1; i <= 4; i++)
    {
        Node p = node_arg(ctx, i);
        if (p.id) parts[n++] = p;
    }
    ctx.ret_number(drie::node_join(parts[:n], type_arg(ctx, 0)).id);
}

// nodeConvert(a, type)
fn void h_node_convert(JsCtx ctx, void* udata) => ctx.ret_number(drie::node_convert(node_arg(ctx, 0), type_arg(ctx, 1)).id);

// nodeNot(a, bitwise)
fn void h_node_not(JsCtx ctx, void* udata)
{
    Node a = node_arg(ctx, 0);
    ctx.ret_number((bool_arg(ctx, 1, false) ? nodes::bit_not(a) : nodes::logical_not(a)).id);
}

// nodeVar(type, init): init 0 for zeros
fn void h_node_var(JsCtx ctx, void* udata) => ctx.ret_number(drie::node_var(type_arg(ctx, 0), node_arg(ctx, 1)).id);

// nodeAssign(target, value)
fn void h_node_assign(JsCtx ctx, void* udata) => ctx.ret_number(drie::node_assign(node_arg(ctx, 0), node_arg(ctx, 1)).id);

// nodeIf(cond, then, else): blocks (their first statements; 0: empty)
fn void h_node_if(JsCtx ctx, void* udata) => ctx.ret_number(drie::node_if(node_arg(ctx, 0), node_arg(ctx, 1), node_arg(ctx, 2)).id);

// nodeLoop(how, index, from, to, body): how a NodeLoop
fn void h_node_loop(JsCtx ctx, void* udata)
{
    int how = (int)num_arg(ctx, 0, 0);
    if (how < 0 || how > NodeLoop.WHILE.ordinal) { ctx.throw_error(TYPE, "nodeLoop: no such loop"); return; }
    ctx.ret_number(drie::node_loop((NodeLoop)how, node_arg(ctx, 1), node_arg(ctx, 2), node_arg(ctx, 3), node_arg(ctx, 4)).id);
}

// nodeJump(kind): 0 break, 1 continue, 2 discard
fn void h_node_jump(JsCtx ctx, void* udata) => ctx.ret_number(drie::node_jump((int)num_arg(ctx, 0, 0)).id);

// nodeAppend(last, statement): the statement after `last` (0: the first of a block); returns it
fn void h_node_append(JsCtx ctx, void* udata) => ctx.ret_number(drie::node_append(node_arg(ctx, 0), node_arg(ctx, 1)).id);

// nodeSeq(block, value): value 0 for none
fn void h_node_seq(JsCtx ctx, void* udata) => ctx.ret_number(drie::node_seq(node_arg(ctx, 0), node_arg(ctx, 1)).id);

// nodeType(a): its NodeType
fn void h_node_type(JsCtx ctx, void* udata) => ctx.ret_number(node_arg(ctx, 0).type().ordinal);

// materialNodes(material, color, opacity, normal, roughness, metalness, emissive, position,
// output, mask, env, specular, shininess): 0 for none
fn void h_material_nodes(JsCtx ctx, void* udata)
{
    Material* m = get_material(num_arg(ctx, 0, -1));
    if (m == null) { ctx.throw_error(TYPE, "materialNodes: invalid material"); return; }
    m.color_node = node_arg(ctx, 1);
    m.opacity_node = node_arg(ctx, 2);
    m.normal_node = node_arg(ctx, 3);
    m.roughness_node = node_arg(ctx, 4);
    m.metalness_node = node_arg(ctx, 5);
    m.emissive_node = node_arg(ctx, 6);
    m.position_node = node_arg(ctx, 7);
    m.output_node = node_arg(ctx, 8);
    m.mask_node = node_arg(ctx, 9);
    m.env_node = node_arg(ctx, 10);
    m.specular_node = node_arg(ctx, 11);
    m.shininess_node = node_arg(ctx, 12);
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
    // (three's examples sit beside its jsm folder, and a few import "./jsm/..." from there:
    // vendor/jsm's file, where there is no such file beside the script)
    sz? at = name.rindex_of("/jsm/");
    if (try at && !file::is_file(name))
    {
        return boomkat::read_module_file(string::tformat("%s/vendor/jsm/%s", root, name[at + 5..]));
    }
    return boomkat::read_module_file(name);
}

fn void reg(JsValue obj, String name, JsHostFn handler, int arity)
{
    js.register_method(obj, name, handler, arity: arity)!!;
}

fn int usage()
{
    io::printn("usage: driejs [--no-jit] [--frames N] [--profile] [--png out.png [--png-width N]] script.js");
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
            case "--png-width":
                if (++i >= args.len) return usage();
                int? n = args[i].to_int();
                if (catch n) return usage();
                png_width = n;
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
    lightmaps.init(mem);
    volumes.init(mem);
    dracos.init(mem);

    start_clock = time::clock::now();
    if (png_path.len && !profile) fixed_clock = 0;
    js.create()!!;
    defer js.destroy();
    track_rejections();
    if (jit && (!js.get_runtime().set_jit() || !js.get_runtime().set_jit_opt()))
    {
        io::eprintn("could not enable the JIT");
        return 1;
    }

    // `import * as THREE from "three"`: three.js's core and its WebGPU build with drie's
    // renderer as WebGLRenderer and WebGPURenderer, read from js/ and vendor/ under the
    // working directory. (three's webgpu examples map "three" to the WebGPU build; one
    // module serves both, as a script loaded after another (demos/index.js) shares its modules.)
    // The WebGPU build's node materials are drawn by drie: js/drie_nodes.js prints their
    // graphs into drie's node shaders.
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
        "import \"%s/js/drie_nodes.js\";\n"
        "export * from \"%s/vendor/three.webgpu.js\";\n"
        "export { DrieRenderer as WebGLRenderer, DrieRenderer as WebGPURenderer } from \"%s/js/drie_renderer.js\";\n"
        "export { PMREMGenerator } from \"%s/js/pmrem.js\";\n"
        "export { ShaderChunk, ShaderLib, UniformsLib, UniformsUtils, WebGLRenderTarget, WebGLCubeRenderTarget, WebGLUtils }"
        " from \"%s/vendor/three.module.js\";\n", root, root, root, root, root, root, root))!!;
    js.define_module("three/webgpu", "export * from \"three\";\n")!!;
    js.define_module("three/tsl", string::tformat(
        "import \"%s/js/dom.js\";\n"
        "export * from \"%s/vendor/three.tsl.js\";\n", root, root))!!;
    // three's Stats draws into the DOM: this one is drawn by the host
    js.define_module("three/addons/libs/stats.module.js",
                     string::tformat("export { default } from \"%s/js/stats.js\";\n", root))!!;
    // and lil-gui's panel is not drawn at all
    js.define_module("three/addons/libs/lil-gui.module.min.js",
                     string::tformat("export * from \"%s/js/gui.js\";\nexport { default } from \"%s/js/gui.js\";\n", root, root))!!;
    // three's OutlineEffect draws the scene again with shaders of its own: this one draws
    // outlines with drie's Material.outline
    js.define_module("three/addons/effects/OutlineEffect.js",
                     string::tformat("export * from \"%s/js/outline_effect.js\";\n", root))!!;
    // three's LightProbeGridWebGL bakes and draws with shaders of its own: this one is drie's
    // irradiance volume, baked by drie
    js.define_module("three/addons/lighting/LightProbeGridWebGL.js",
                     string::tformat("export { LightProbeGridWebGL } from \"%s/js/light_probe_grid.js\";\n", root))!!;
    js.define_module("three/addons/helpers/LightProbeGridHelperWebGL.js",
                     string::tformat("export { LightProbeGridHelperWebGL } from \"%s/js/light_probe_grid.js\";\n", root))!!;
    // and the WebGPU build's LightProbeGrid, the same with three's node shaders
    js.define_module("three/addons/lighting/LightProbeGrid.js", string::tformat(
        "export { LightProbeGridWebGL as LightProbeGrid } from \"%s/js/light_probe_grid.js\";\n", root))!!;
    js.define_module("three/addons/helpers/LightProbeGridHelper.js", string::tformat(
        "export { LightProbeGridHelperWebGL as LightProbeGridHelper } from \"%s/js/light_probe_grid.js\";\n", root))!!;
    // LightProbeGenerator.fromCubeRenderTarget from drie's map of a CubeCamera (DrieRenderer._lightProbe)
    js.define_module("three/addons/lights/LightProbeGenerator.js", string::tformat(
        "import { LightProbeGenerator } from \"%s/vendor/jsm/lights/LightProbeGenerator.js\";\n"
        "LightProbeGenerator.fromCubeRenderTarget = (renderer, target) => renderer._lightProbe(target);\n"
        "export { LightProbeGenerator };\n", root))!!;
    // drie draws what three's webgpu examples ask of WebGPU
    js.define_module("three/addons/capabilities/WebGPU.js",
        "export default class WebGPU {\n"
        "  static isAvailable() { return true; }\n"
        "  static getErrorMessage() { return null; }\n"
        "}\n")!!;
    // three/webgpu examples' Inspector: its parameters panel as lil-gui's (not drawn)
    js.define_module("three/addons/inspector/Inspector.js", string::tformat(
        "import { GUI } from \"%s/js/gui.js\";\n"
        "export class Inspector {\n"
        "  constructor() { this.domElement = document.createElement(\"div\"); }\n"
        "  createParameters(title) { return new GUI({ title }); }\n"
        "}\n", root))!!;
    // RectAreaLightUniformsLib fills WebGLRenderer's UniformsLib: drie has the tables itself
    // three's UltraHDRLoader applies the gain map in JS: this one natively
    js.define_module("three/addons/loaders/UltraHDRLoader.js",
                     string::tformat("export * from \"%s/js/ultrahdr.js\";\n", root))!!;
    js.define_module("three/addons/lights/RectAreaLightUniformsLib.js",
                     string::tformat("export * from \"%s/js/rect_area_light.js\";\n", root))!!;
    // three's ProgressiveLightMap (and its WebGPU one, with the same methods) draws with shaders
    // of its own: this one has drie bake
    js.define_module("three/addons/misc/ProgressiveLightMap.js",
                     string::tformat("export * from \"%s/js/progressive_lightmap.js\";\n", root))!!;
    js.define_module("three/addons/misc/ProgressiveLightMapGPU.js",
                     string::tformat("export * from \"%s/js/progressive_lightmap.js\";\n", root))!!;

    JsValue natives = js.object()!!;
    defer js.release(natives);
    reg(natives, "renderer", &h_renderer, 4);
    reg(natives, "scene", &h_scene, 0);
    reg(natives, "group", &h_group, 0);
    reg(natives, "camera", &h_camera, 4);
    reg(natives, "cameraUpdate", &h_camera_update, 10);
    reg(natives, "orthoCamera", &h_ortho_camera, 6);
    reg(natives, "light", &h_light, 8);
    reg(natives, "lightTarget", &h_light_target, 1);
    reg(natives, "geometry", &h_geometry, 9);
    reg(natives, "instanceColors", &h_instance_colors, 3);
    reg(natives, "irradianceGrid", &h_irradiance_grid, 9);
    reg(natives, "bakeIrradiance", &h_bake_irradiance, 6);
    reg(natives, "sceneIrradiance", &h_scene_irradiance, 5);
    reg(natives, "texture", &h_texture, 19);
    reg(natives, "textureMipmap", &h_texture_mipmap, 4);
    reg(natives, "textureClone", &h_texture_clone, 9);
    reg(natives, "cubeTexture", &h_cube_texture, 7);
    reg(natives, "cubeImages", &h_cube_images, 7);
    reg(natives, "material", &h_material, 27);
    reg(natives, "materialUpdate", &h_material_update, 19);
    reg(natives, "materialMaps", &h_material_maps, 26);
    reg(natives, "materialLayers", &h_material_layers, 14);
    reg(natives, "nodeConst", &h_node_const, 5);
    reg(natives, "nodeUniform", &h_node_uniform, 1);
    reg(natives, "nodeSet", &h_node_set, 5);
    reg(natives, "nodeBuiltin", &h_node_builtin, 2);
    reg(natives, "nodeTexture", &h_node_texture, 3);
    reg(natives, "nodeTextureSize", &h_node_texture_size, 1);
    reg(natives, "nodeVarying", &h_node_varying, 1);
    reg(natives, "nodeBinary", &h_node_binary, 3);
    reg(natives, "nodeCall", &h_node_call, 4);
    reg(natives, "nodeSelect", &h_node_select, 3);
    reg(natives, "nodeSwizzle", &h_node_swizzle, 2);
    reg(natives, "nodeJoin", &h_node_join, 5);
    reg(natives, "nodeConvert", &h_node_convert, 2);
    reg(natives, "nodeNot", &h_node_not, 2);
    reg(natives, "nodeVar", &h_node_var, 2);
    reg(natives, "nodeAssign", &h_node_assign, 2);
    reg(natives, "nodeIf", &h_node_if, 3);
    reg(natives, "nodeLoop", &h_node_loop, 5);
    reg(natives, "nodeJump", &h_node_jump, 1);
    reg(natives, "nodeAppend", &h_node_append, 2);
    reg(natives, "nodeSeq", &h_node_seq, 2);
    reg(natives, "nodeType", &h_node_type, 1);
    reg(natives, "materialNodes", &h_material_nodes, 13);
    reg(natives, "floatTexture", &h_float_texture, 6);
    reg(natives, "imageDecode", &h_image_decode, 1);
    reg(natives, "imageSize", &h_image_size, 2);
    reg(natives, "imagePixels", &h_image_pixels, 4);
    reg(natives, "textMetrics", &h_text_metrics, 6);
    reg(natives, "textMask", &h_text_mask, 10);
    reg(natives, "imageFree", &h_image_free, 1);
    reg(natives, "gainmapApply", &h_gainmap_apply, 4);
    reg(natives, "backgroundTexture", &h_background_texture, 5);
    reg(natives, "backgroundMaterial", &h_background_material, 2);
    reg(natives, "renderEnvironment", &h_render_environment, 5);
    reg(natives, "renderProbe", &h_render_probe, 10);
    reg(natives, "lightChannels", &h_light_channels, 2);
    reg(natives, "meshDrawRange", &h_mesh_draw_range, 3);
    reg(natives, "meshBillboard", &h_mesh_billboard, 6);
    reg(natives, "materialLightChannels", &h_material_light_channels, 2);
    reg(natives, "textureLightProbe", &h_texture_light_probe, 2);
    reg(natives, "textureUv", &h_texture_uv, 8);
    reg(natives, "mesh", &h_mesh, 2);
    reg(natives, "line", &h_line, 3);
    reg(natives, "lineDash", &h_line_dash, 4);
    reg(natives, "add", &h_add, 2);
    reg(natives, "sync", &h_sync, 2);
    reg(natives, "background", &h_background, 2);
    reg(natives, "render", &h_render, 3);
    reg(natives, "setClearColor", &h_set_clear_color, 3);
    reg(natives, "viewState", &h_view_state, 11);
    reg(natives, "clear", &h_clear, 1);
    reg(natives, "setSize", &h_set_size, 3);
    reg(natives, "info", &h_info, 2);
    reg(natives, "now", &h_now, 0);
    reg(natives, "fixedClock", &h_fixed_clock, 0);
    reg(natives, "setAnimationLoop", &h_set_animation_loop, 2);
    reg(natives, "stats", &h_stats, 6);
    reg(natives, "shadow", &h_shadow, 3);
    reg(natives, "lightUpdate", &h_light_update, 6);
    reg(natives, "lightShadow", &h_light_shadow, 15);
    reg(natives, "shadowType", &h_shadow_type, 2);
    reg(natives, "materialLightMap", &h_material_light_map, 3);
    reg(natives, "materialEnv", &h_material_env, 6);
    reg(natives, "materialSurface", &h_material_surface, 9);
    reg(natives, "materialOutline", &h_material_outline, 2);
    reg(natives, "materialBlend", &h_material_blend, 9);
    reg(natives, "materialDepth", &h_material_depth, 3);
    reg(natives, "lightmap", &h_lightmap, 3);
    reg(natives, "lightmapObjects", &h_lightmap_objects, 3);
    reg(natives, "lightmapUpdate", &h_lightmap_update, 4);
    reg(natives, "lightmapTexture", &h_lightmap_texture, 2);
    reg(natives, "fog", &h_fog, 6);
    reg(natives, "toneMapping", &h_tone_mapping, 4);
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
    reg(natives, "label", &h_label, 6);
    reg(natives, "labelScale", &h_label_scale, 0);
    reg(natives, "files", &h_files, 1);
    reg(natives, "run", &h_run, 1);
    reg(natives, "loadPipelineCache", &h_load_pipeline_cache, 1);
    reg(natives, "savePipelineCache", &h_save_pipeline_cache, 1);
    reg(natives, "pipelineCacheProgress", &h_pipeline_cache_progress, 0);

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

    if (!load_script(script_path)) return 1;
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
