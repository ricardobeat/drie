// glTF 2.0 loader (`.glb` and `.gltf` with external `.bin` and images) built on `third_party/cgltf`
// via `generated/cgltf.c3`. Supports node hierarchies, triangle meshes (positions, normals, tangents,
// two UV sets, vertex colors, sparse and quantized accessors), PBR metallic-roughness materials,
// `KHR_texture_transform`, `KHR_materials_unlit`, and volume/transmission/specular/clearcoat/sheen/
// iridescence/anisotropy/emissive_strength extensions. Skins, animations, morph targets, cameras,
// lights, Draco/meshopt compression, and `KHR_texture_basisu` are not implemented.
//
//   GltfModel* model = loaders::load_gltf("assets/models/tree.glb")!!;
//   scene.add(model.scene);
module drie::loaders;
import std::io, std::math, std::collections::list, libc;
import drie, drie::cgltf;

// Returned when a glTF file cannot be parsed, lacks buffer data, or references a broken image.
faultdef GLTF_INVALID;

// Loaded glTF scene graph together with flat lists of its materials and meshes.
struct GltfModel {
    // Root of the loaded node hierarchy. Each glTF node becomes a named Group containing one
    // child Mesh per primitive.
    Group* scene;
    // Loaded glTF materials in file order, followed by cloned normal-map variants generated for
    // primitives that omit vertex tangents.
    List{Material*} materials;
    // All Mesh instances created across the loaded scene graph.
    List{Mesh*} meshes;
}

// Returns the first node or mesh in `scene` with `name`, or null if none matches.
fn Object3D* GltfModel.get(&self, String name) => self.scene.get_object_by_name(name);

struct GltfReader @local {
    CgltfData* data;
    String dir;
    GltfModel* model;
    // per glTF image, its Texture in sRGB and in linear (data) colors: what each texture of
    // it shows, as is or placed another way by a clone (see GltfReader.texture)
    List{Texture*[2]} images;
    // per glTF mesh, its primitives' geometries (shared by the nodes that use the mesh)
    List{BufferGeometry*[]} geometries;
    // per glTF material, its copy for meshes without tangents (null: none made yet)
    List{Material*} untangented;
}

fn String text(CChar* s) @local => s ? ((ZString)s).str_view() : "";

// The image a glTF texture shows, decoded once per color space.
fn Texture*? GltfReader.image(&self, CgltfTexture* tex, ColorSpace space) {
    // (WebP where the platform's decoder reads it; KTX2/Basis is not decoded)
    CgltfImage* image = tex.image ?: (tex.has_webp ? tex.webp_image : null);
    if (!image) return GLTF_INVALID~;
    sz index = (sz)cgltf::image_index(self.data, image);
    while (self.images.len() <= index) self.images.push({ });
    int slot = space == SRGB_COLOR_SPACE ? 0 : 1;
    if (self.images[index][slot]) return self.images[index][slot];
    Texture* t;
    String uri = text(image.uri);
    if (image.buffer_view) {
        char* bytes = (char*)cgltf::buffer_view_data(image.buffer_view);
        if (!bytes) return GLTF_INVALID~;
        t = drie::decode_texture(bytes[:(sz)image.buffer_view.size], space, flip_y: false)!;
    } else if (uri.starts_with("data:")) {
        // base64 in the uri itself
        sz comma = uri.index_of(";base64,") ?? 0;
        if (!comma) return GLTF_INVALID~;
        String b64 = uri[comma + 8..];
        sz padding = b64.ends_with("==") ? 2 : b64.ends_with("=") ? 1 : 0;
        sz size    = b64.len / 4 * 3 - padding;
        void* bytes;
        CgltfOptions options;
        if (cgltf::load_buffer_base64(
            &options,
            (usz)size,
            (CChar*)b64.ptr,
            &bytes
        ) != cgltf::RESULT_SUCCESS) return GLTF_INVALID~;
        defer libc::free(bytes);
        t = drie::decode_texture(((char*)bytes)[:size], space, flip_y: false)!;
    } else {
        @pool() {
            // (percent-encoded, as uris are)
            ZString path = string::tformat("%s%s", self.dir, uri).zstr_tcopy();
            cgltf::decode_uri((CChar*)path);
            t = drie::load_texture(path.str_view(), space, flip_y: false);
        };
    }
    if (tex.sampler) {
        t.wrap_s = wrapping(tex.sampler.wrap_s);
        t.wrap_t = wrapping(tex.sampler.wrap_t);
        t.nearest = tex.sampler.mag_filter == cgltf::FILTER_TYPE_NEAREST;
        // (a minification filter without mipmaps: none made)
        CgltfFilterType min = tex.sampler.min_filter;
        if (min == cgltf::FILTER_TYPE_NEAREST || min == cgltf::FILTER_TYPE_LINEAR) t.generate_mipmaps = false;
    }
    self.images[index][slot] = t;
    return t;
}

fn Wrapping wrapping(CgltfWrapMode mode) @local {
    if (mode == cgltf::WRAP_MODE_CLAMP_TO_EDGE) return CLAMP_TO_EDGE_WRAPPING;
    if (mode == cgltf::WRAP_MODE_MIRRORED_REPEAT) return MIRRORED_REPEAT_WRAPPING;
    return REPEAT_WRAPPING;
}

// A material's texture: its image, read with the uv set it names and placed by its
// KHR_texture_transform (as three's GLTFLoader places it: offset, rotation and repeat). Null
// for none.
fn Texture*? GltfReader.texture(&self, CgltfTextureView* view, ColorSpace space) {
    if (!view.texture) return null;
    Texture* shown = self.image(view.texture, space)!;
    int channel    = view.has_transform && view.transform.has_texcoord ? view.transform.texcoord : view.texcoord;
    if (!view.has_transform && channel == 0) return shown;
    // (drie's geometries have two uv sets)
    if (channel > 1) return null;
    Texture* t = shown.clone();
    t.channel = (char)channel;
    if (view.has_transform) {
        CgltfTextureTransform* x = &view.transform;
        t.offset = { x.offset[0], x.offset[1] };
        t.rotation = x.rotation;
        t.repeat = { x.scale[0], x.scale[1] };
    }
    return t;
}

fn Color linear_color(float r, float g, float b) @local {
    float[3] c = { r, g, b };
    uint result;
    foreach (v : c) {
        v = drie::clamp(v, 0, 1);
        float s = v <= 0.0031308f ? v * 12.92f : 1.055f * math::pow(v, 1.0f / 2.4f) - 0.055f;
        result = result << 8 | (uint)(s * 255 + 0.5f);
    }
    return result;
}

// Converts a glTF material into a `MeshBasicMaterial` (if unlit) or `MeshStandardMaterial`.
fn Material*? GltfReader.material(&self, CgltfMaterial* m) {
    CgltfPbrMetallicRoughness* pbr = &m.pbr_metallic_roughness;
    float[4] base                  = m.has_pbr_metallic_roughness ? pbr.base_color_factor : { 1, 1, 1, 1 };
    if (m.unlit) {
        // KHR_materials_unlit: its color as it is, as three's MeshBasicMaterial
        MeshBasicMaterial* basic = drie::mesh_basic_material(
            color: linear_color(base[0], base[1], base[2]),
            opacity: base[3],
            transparent: m.alpha_mode == cgltf::ALPHA_MODE_BLEND,
            side: m.double_sided ? DOUBLE_SIDE : FRONT_SIDE
        );
        basic.map = self.texture(&pbr.base_color_texture, SRGB_COLOR_SPACE)!;
        if (m.alpha_mode == cgltf::ALPHA_MODE_MASK) basic.alpha_test = m.alpha_cutoff;
        return basic;
    }
    MeshStandardMaterial* mat = drie::mesh_standard_material();
    mat.color = linear_color(base[0], base[1], base[2]);
    mat.opacity = base[3];
    if (m.has_pbr_metallic_roughness) {
        mat.roughness = pbr.roughness_factor;
        mat.metalness = pbr.metallic_factor;
        mat.map = self.texture(&pbr.base_color_texture, SRGB_COLOR_SPACE)!;
        // roughness in green, metalness in blue, and (without an occlusionTexture) ambient
        // occlusion in red: an arm map, as house's models pack it
        mat.arm_map = self.texture(&pbr.metallic_roughness_texture, NO_COLOR_SPACE)!;
    }
    if (m.occlusion_texture.texture) {
        // its own ambient occlusion: the other map gives only roughness and metalness
        if (mat.arm_map) {
            mat.roughness_map = mat.arm_map;
            mat.metalness_map = mat.arm_map;
            mat.arm_map = null;
        }
        mat.ao_map = self.texture(&m.occlusion_texture, NO_COLOR_SPACE)!;
        mat.ao_intensity = m.occlusion_texture.scale;
    }
    if (m.normal_texture.texture) {
        mat.normal_map = self.texture(&m.normal_texture, NO_COLOR_SPACE)!;
        mat.normal_scale = { m.normal_texture.scale, m.normal_texture.scale };
    }
    mat.emissive = linear_color(m.emissive_factor[0], m.emissive_factor[1], m.emissive_factor[2]);
    mat.emissive_map = self.texture(&m.emissive_texture, SRGB_COLOR_SPACE)!;
    if (m.alpha_mode == cgltf::ALPHA_MODE_MASK) mat.alpha_test = m.alpha_cutoff;
    if (m.alpha_mode == cgltf::ALPHA_MODE_BLEND) mat.transparent = true;
    if (m.double_sided) mat.side = DOUBLE_SIDE;
    // transmission, and what the light does as it crosses the volume
    if (m.has_transmission) {
        mat.transmission = m.transmission.transmission_factor;
        mat.transmission_map = self.texture(&m.transmission.transmission_texture, NO_COLOR_SPACE)!;
    }
    if (m.has_volume) {
        CgltfVolume* vol = &m.volume;
        mat.thickness = vol.thickness_factor;
        mat.thickness_map = self.texture(&vol.thickness_texture, NO_COLOR_SPACE)!;
        // (cgltf's "infinitely far" is FLT_MAX; drie's is 0)
        mat.attenuation_distance = vol.attenuation_distance < 1e30f ? vol.attenuation_distance : 0;
        mat.attenuation_color = linear_color(
            vol.attenuation_color[0],
            vol.attenuation_color[1],
            vol.attenuation_color[2]
        );
    }
    if (m.has_ior) mat.ior = m.ior.ior;
    // the strength and color of its reflections where it isn't metal
    if (m.has_specular) {
        CgltfSpecular* sp = &m.specular;
        mat.specular_intensity = sp.specular_factor;
        mat.specular_color = { sp.specular_color_factor[0], sp.specular_color_factor[1], sp.specular_color_factor[2] };
        mat.specular_intensity_map = self.texture(&sp.specular_texture, NO_COLOR_SPACE)!;
        mat.specular_color_map = self.texture(&sp.specular_color_texture, SRGB_COLOR_SPACE)!;
    }
    if (m.has_dispersion) mat.dispersion = m.dispersion.dispersion;
    if (m.has_emissive_strength) mat.emissive_intensity = m.emissive_strength.emissive_strength;
    // a varnish over it
    if (m.has_clearcoat) {
        CgltfClearcoat* cc = &m.clearcoat;
        mat.clearcoat = cc.clearcoat_factor;
        mat.clearcoat_roughness = cc.clearcoat_roughness_factor;
        mat.clearcoat_map = self.texture(&cc.clearcoat_texture, NO_COLOR_SPACE)!;
        mat.clearcoat_roughness_map = self.texture(&cc.clearcoat_roughness_texture, NO_COLOR_SPACE)!;
        mat.clearcoat_normal_map = self.texture(&cc.clearcoat_normal_texture, NO_COLOR_SPACE)!;
        float scale = cc.clearcoat_normal_texture.scale;
        mat.clearcoat_normal_scale = { scale, scale };
    }
    // cloth's fibers
    if (m.has_sheen) {
        CgltfSheen* sh = &m.sheen;
        mat.sheen = 1;
        mat.sheen_color = { sh.sheen_color_factor[0], sh.sheen_color_factor[1], sh.sheen_color_factor[2] };
        mat.sheen_roughness = sh.sheen_roughness_factor;
        mat.sheen_color_map = self.texture(&sh.sheen_color_texture, SRGB_COLOR_SPACE)!;
        mat.sheen_roughness_map = self.texture(&sh.sheen_roughness_texture, NO_COLOR_SPACE)!;
    }
    // a thin film's rainbow
    if (m.has_iridescence) {
        CgltfIridescence* ir = &m.iridescence;
        mat.iridescence = ir.iridescence_factor;
        mat.iridescence_ior = ir.iridescence_ior;
        mat.iridescence_thickness_min = ir.iridescence_thickness_min;
        mat.iridescence_thickness_max = ir.iridescence_thickness_max;
        mat.iridescence_map = self.texture(&ir.iridescence_texture, NO_COLOR_SPACE)!;
        mat.iridescence_thickness_map = self.texture(&ir.iridescence_thickness_texture, NO_COLOR_SPACE)!;
    }
    // brushed metal's drawn out highlights
    if (m.has_anisotropy) {
        CgltfAnisotropy* an = &m.anisotropy;
        mat.anisotropy = an.anisotropy_strength;
        mat.anisotropy_rotation = an.anisotropy_rotation;
        mat.anisotropy_map = self.texture(&an.anisotropy_texture, NO_COLOR_SPACE)!;
    }
    return mat;
}

// The material for a mesh without tangents. Its tangent frame is then the one of the uv's
// derivatives (see Material.normal_scale), whose y is the way v grows, down a glTF image,
// where the normal maps' y is up: a copy of the material with their y negated, made once
// (three's GLTFLoader does the same).
fn Material* GltfReader.untangented_material(&self, sz index) {
    Material* mat = self.model.materials[index];
    if (!mat.normal_map && !mat.clearcoat_normal_map) return mat;
    if (self.untangented[index]) return self.untangented[index];
    Material* copy = mat.clone();
    copy.normal_scale.y = -copy.normal_scale.y;
    copy.clearcoat_normal_scale.y = -copy.clearcoat_normal_scale.y;
    self.untangented[index] = copy;
    self.model.materials.push(copy);
    return copy;
}

// An accessor's values as floats (normalized integers to -1..1 or 0..1, sparse values
// applied), `components` per element; empty for none. Free with mem::free.
fn float[] floats(CgltfAccessor* a, usz components) @local {
    if (!a) return { };
    usz n       = cgltf::num_components(a.type);
    float[] all = mem::new_array(float, (sz)(a.count * n));
    cgltf::accessor_unpack_floats(a, all.ptr, all.len);
    if (n == components) return all;
    // (colors as RGB from RGBA: the shape drie keeps)
    float[] out = mem::new_array(float, (sz)(a.count * components));
    for (usz i = 0; i < a.count; i++) {
        for (usz c = 0; c < components; c++) out[i * components + c] = c < n ? all[i * n + c] : 1;
    }
    mem::free(all.ptr);
    return out;
}

// A primitive's geometry; null for one drie does not draw (points, lines, compressed data).
fn BufferGeometry* primitive_geometry(CgltfPrimitive* p) @local {
    if (p.type != cgltf::PRIMITIVE_TYPE_TRIANGLES) return null;
    if (p.has_draco_mesh_compression) {
        io::eprintn("drie: glTF: a Draco compressed mesh is skipped (not supported)");
        return null;
    }
    CgltfAccessor* position = (CgltfAccessor*)cgltf::find_accessor(p, cgltf::ATTRIBUTE_TYPE_POSITION, 0);
    if (!position) return null;
    float[] pos      = floats(position, 3);
    float[] nrm      = floats((CgltfAccessor*)cgltf::find_accessor(p, cgltf::ATTRIBUTE_TYPE_NORMAL, 0), 3);
    float[] uv       = floats((CgltfAccessor*)cgltf::find_accessor(p, cgltf::ATTRIBUTE_TYPE_TEXCOORD, 0), 2);
    float[] uv1      = floats((CgltfAccessor*)cgltf::find_accessor(p, cgltf::ATTRIBUTE_TYPE_TEXCOORD, 1), 2);
    float[] color    = floats((CgltfAccessor*)cgltf::find_accessor(p, cgltf::ATTRIBUTE_TYPE_COLOR, 0), 3);
    float[] tangents = floats((CgltfAccessor*)cgltf::find_accessor(p, cgltf::ATTRIBUTE_TYPE_TANGENT, 0), 4);
    defer {
        mem::free(tangents.ptr);
        mem::free(pos.ptr);
        mem::free(nrm.ptr);
        mem::free(uv.ptr);
        mem::free(uv1.ptr);
        mem::free(color.ptr);
    }
    usz count = position.count;

    BufferGeometry* g = mem::new(BufferGeometry);
    g.init_lists();
    g.vertices.reserve((sz)count * drie::FLOATS_PER_VERTEX);
    for (usz i = 0; i < count; i++) {
        Vector3 n = nrm.len ? (Vector3) { nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2] } : (Vector3) { 0, 1, 0 };
        Vector2 t = uv.len ? (Vector2) { uv[i * 2], uv[i * 2 + 1] } : (Vector2) { };
        Vector3 c = color.len ? (Vector3) { color[i * 3], color[i * 3 + 1], color[i * 3 + 2] } : (Vector3) { 1, 1, 1 };
        g.add_vertex({ pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2] }, n, t, c);
    }
    if (uv1.len) foreach (f : uv1) g.uv1.push(f);
    if (tangents.len) foreach (f : tangents) g.tangents.push(f);
    if (p.indices) {
        usz n      = p.indices.count;
        uint[] idx = mem::new_array(uint, (sz)n);
        defer mem::free(idx.ptr);
        // (one at a time where cgltf can't unpack them all: sparse indices)
        if (cgltf::accessor_unpack_indices(p.indices, idx.ptr, 4, n) != n) {
            for (usz i = 0; i < n; i++) idx[i] = (uint)cgltf::accessor_read_index(p.indices, i);
        }
        g.indices.reserve((sz)n);
        foreach (i : idx) g.indices.push(i);
    } else {
        for (uint i = 0; i < (uint)count; i++) g.indices.push(i);
    }
    if (!nrm.len) g.compute_vertex_normals();
    return g;
}

// Recursively converts a glTF node, its decomposed local transform, meshes, and children.
fn Object3D*? GltfReader.node(&self, CgltfNode* n) {
    Group* obj = drie::group();
    if (n.name) obj.name = text(n.name).copy(mem);
    // its translation, rotation and scale (or matrix), taken apart
    float[16] local;
    cgltf::node_transform_local(n, &local);
    drie::Mat4 m;
    for (int c = 0; c < 4; c++) m[c] = { local[c * 4], local[c * 4 + 1], local[c * 4 + 2], local[c * 4 + 3] };
    Vector3 scale = {
        math::sqrt(m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2]),
        math::sqrt(m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2]),
        math::sqrt(m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]),
    };
    // (mirrored: one negative scale)
    float det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
        + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
    if (det < 0) scale.x = -scale.x;
    obj.position = { m[3][0], m[3][1], m[3][2] };
    obj.scale = scale;
    for (int c = 0; c < 3; c++) {
        float s = c == 0 ? scale.x : c == 1 ? scale.y : scale.z;
        if (s != 0) for (int r = 0; r < 3; r++) m[c][r] /= s;
    }
    obj.rotation = drie::euler_from_rotation(m);
    if (n.mesh) {
        sz mesh_index = (sz)cgltf::mesh_index(self.data, n.mesh);
        for (sz i = 0; i < (sz)n.mesh.primitives_count; i++) {
            CgltfPrimitive* prim = &n.mesh.primitives[i];
            // geometry is shared between nodes that use the same glTF mesh
            if (!self.geometries[mesh_index][i]) self.geometries[mesh_index][i] = primitive_geometry(prim);
            BufferGeometry* g = self.geometries[mesh_index][i];
            if (!g) continue;
            Material* mat;
            if (prim.material) {
                sz index = (sz)cgltf::material_index(self.data, prim.material);
                mat = g.tangents.len() ? self.model.materials[index] : self.untangented_material(index);
            } else {
                mat = drie::mesh_standard_material();
            }
            Mesh* mesh = drie::mesh(g, mat);
            mesh.name = obj.name;
            obj.add(mesh);
            self.model.meshes.push(mesh);
        }
    }
    for (usz i = 0; i < n.children_count; i++) obj.add(self.node(n.children[i])!);
    return obj;
}

// Loads a `.glb` or `.gltf` asset from `path` (resolving external `.bin` and image URIs relative
// to `path`'s directory) and returns the constructed `GltfModel`.
fn GltfModel*? load_gltf(String path) {
    CgltfOptions options;
    CgltfData* data;
    CgltfResult result;
    @pool() {
        CChar* zpath = (CChar*)path.zstr_tcopy();
        result = cgltf::parse_file(&options, zpath, &data);
        // (the .bin and data: uris; images are read as the materials use them)
        if (result == cgltf::RESULT_SUCCESS) result = cgltf::load_buffers(&options, data, zpath);
    };
    if (result != cgltf::RESULT_SUCCESS) {
        if (data) cgltf::free(data);
        io::eprintfn("drie: cannot load glTF '%s' (cgltf result %d)", path, result);
        return GLTF_INVALID~;
    }
    defer cgltf::free(data);
    GltfReader r = { .data = data };
    r.images.allocator = mem;
    r.geometries.allocator = mem;
    defer r.images.free();
    sz slash = -1;
    foreach (i, c : path) {
        if (c == '/') slash = i;
    }
    r.dir = path[:slash + 1];

    GltfModel* model = mem::new(GltfModel);
    model.materials.allocator = mem;
    model.meshes.allocator = mem;
    r.model = model;
    for (usz i = 0; i < data.materials_count; i++) model.materials.push(r.material(&data.materials[i])!);
    r.untangented.allocator = mem;
    defer r.untangented.free();
    for (usz i = 0; i < data.materials_count; i++) r.untangented.push(null);
    for (usz i = 0; i < data.meshes_count; i++) {
        r.geometries.push(mem::new_array(BufferGeometry*, (sz)data.meshes[i].primitives_count));
    }
    defer {
        foreach (g : r.geometries) mem::free(g.ptr);
        r.geometries.free();
    }

    model.scene = drie::group();
    CgltfScene* scene = data.scene ?: (data.scenes_count ? &data.scenes[0] : null);
    if (scene) {
        for (usz i = 0; i < scene.nodes_count; i++) model.scene.add(r.node(scene.nodes[i])!);
    } else {
        // (no scenes: the nodes at the top of the hierarchy)
        for (usz i = 0; i < data.nodes_count; i++) {
            if (!data.nodes[i].parent) model.scene.add(r.node(&data.nodes[i])!);
        }
    }
    return model;
}