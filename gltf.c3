// glTF 2.0 loader (.glb, or .gltf with one external .bin), like three's GLTFLoader.
//   GltfModel* model = loaders::load_gltf("assets/models/tree.glb")!!;
//   scene.add(model.scene);
// Supported: node hierarchy, triangle meshes (positions, normals, uvs, vertex colors),
// metallic-roughness materials with base color / normal / occlusion-roughness-metal textures,
// occlusion textures, alpha modes, double sided materials, and the transmission / volume / ior /
// dispersion / specular extensions (see Material.transmission and specular_color). Not supported: skins, animations, cameras, sparse
// accessors.
module drie::loaders;
import std::io, std::math, std::encoding::json, std::collections::object, std::collections::list;
import drie;

faultdef GLTF_INVALID;

struct GltfModel {
    // Root of the loaded node hierarchy. Each glTF node is an Object3D named like the node,
    // with one child Mesh per material it uses.
    Group* scene;
    List{Material*} materials;
    List{Mesh*} meshes;
}

// The node (or mesh) with this name, or null.
fn Object3D* GltfModel.get(&self, String name) => self.scene.get_object_by_name(name);

struct GltfReader @local {
    Object* json;
    char[] bin;
    String dir;
    GltfModel* model;
    // one per glTF image and color space: [image][srgb ? 0 : 1]
    List{Texture*[2]} textures;
    List{BufferGeometry*[]} geometries;
}

fn Object*? child(Object* o, String key) @local => o.has_key(key) ? o.get(key) : io::EOF~;

fn float num(Object* o, String key, float fallback) @local {
    if (!o.has_key(key)) return fallback;
    return (float)(o.get_float(key) ?? (double)fallback);
}

fn uint read_u32(
    char[] data,
    sz at
) @local => data[at] | (uint)data[at + 1] << 8 | (uint)data[at + 2] << 16 | (uint)data[at + 3] << 24;

struct Accessor @local {
    char* ptr;
    sz count;
    sz stride;
    int component_type;
    int components;
}

fn Accessor? GltfReader.accessor(&self, int index) {
    Object* a    = self.json.get("accessors")!.get_at(index);
    Object* view = self.json.get("bufferViews")!.get_at(a.get_int("bufferView")!);
    Accessor acc;
    String type  = a.get_string("type")!;
    switch (type) {
        case "SCALAR": acc.components = 1;
        case "VEC2": acc.components = 2;
        case "VEC3": acc.components = 3;
        case "VEC4": acc.components = 4;
        default: return GLTF_INVALID~;
    }
    acc.component_type = a.get_int("componentType")!;
    sz size;
    switch (acc.component_type) {
        case 5120:
        case 5121: size = 1;
        case 5122:
        case 5123: size = 2;
        case 5125:
        case 5126: size = 4;
        default: return GLTF_INVALID~;
    }
    acc.count = a.get_int("count")!;
    acc.stride = view.has_key("byteStride") ? view.get_int("byteStride")! : size * acc.components;
    sz offset = (sz)num(view, "byteOffset", 0) + (sz)num(a, "byteOffset", 0);
    if (offset + (acc.count - 1) * acc.stride + size * acc.components > self.bin.len) return GLTF_INVALID~;
    acc.ptr = &self.bin[offset];
    return acc;
}

fn float Accessor.value(&self, sz i, int c) {
    char* p = self.ptr + i * self.stride;
    switch (self.component_type) {
        case 5126: return ((float*)p)[c];  // may be unaligned, which arm64 and x86 allow
        // normalized integers (KHR_mesh_quantization)
        case 5120: return math::max((float)((ichar*)p)[c] / 127.0f, -1.0f);
        case 5121: return p[c] / 255.0f;
        case 5122: return math::max((float)((short*)p)[c] / 32767.0f, -1.0f);
        case 5123: return ((ushort*)p)[c] / 65535.0f;
        default: return 0;
    }
}

fn uint Accessor.index(&self, sz i) {
    char* p = self.ptr + i * self.stride;
    switch (self.component_type) {
        case 5121: return p[0];
        case 5123: return *(ushort*)p;
        default: return *(uint*)p;
    }
}

fn Texture*? GltfReader.texture(&self, Object* info, ColorSpace space) {
    Object* tex     = self.json.get("textures")!.get_at(info.get_int("index")!);
    int image_index = tex.get_int("source")!;
    while (self.textures.len() <= image_index) self.textures.push({ });
    int slot = space == SRGB_COLOR_SPACE ? 0 : 1;
    if (self.textures[image_index][slot]) return self.textures[image_index][slot];
    Object* image = self.json.get("images")!.get_at(image_index);
    Texture* t;
    if (image.has_key("uri")) {
        @pool() {
            t = drie::load_texture(string::tformat("%s%s", self.dir, image.get_string("uri")!), space, flip_y: false);
        };
    } else {
        Object* view = self.json.get("bufferViews")!.get_at(image.get_int("bufferView")!);
        sz offset    = (sz)num(view, "byteOffset", 0);
        sz len       = view.get_int("byteLength")!;
        if (offset + len > self.bin.len) return GLTF_INVALID~;
        t = drie::decode_texture(self.bin[offset:len], space, flip_y: false)!;
    }
    self.textures[image_index][slot] = t;
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

fn Material*? GltfReader.material(&self, Object* m) {
    MeshStandardMaterial* mat = drie::mesh_standard_material();
    if (try pbr = child(m, "pbrMetallicRoughness")) {
        if (try f = child(pbr, "baseColorFactor")) {
            mat.color = linear_color((float)f.get_float_at(0)!, (float)f.get_float_at(1)!, (float)f.get_float_at(2)!);
            mat.opacity = (float)f.get_float_at(3)!;
        }
        mat.roughness = num(pbr, "roughnessFactor", 1);
        mat.metalness = num(pbr, "metallicFactor", 1);
        if (try t = child(pbr, "baseColorTexture")) mat.map = self.texture(t, SRGB_COLOR_SPACE)!;
        // roughness in green, metalness in blue, and (without an occlusionTexture) ambient
        // occlusion in red: an arm map, as house's models pack it
        if (try t = child(pbr, "metallicRoughnessTexture")) mat.arm_map = self.texture(t, NO_COLOR_SPACE)!;
    }
    if (try t = child(m, "occlusionTexture")) {
        // its own ambient occlusion: the other map gives only roughness and metalness
        if (mat.arm_map) {
            mat.roughness_map = mat.arm_map;
            mat.metalness_map = mat.arm_map;
            mat.arm_map = null;
        }
        mat.ao_map = self.texture(t, NO_COLOR_SPACE)!;
        mat.ao_intensity = num(t, "strength", 1);
    }
    if (try t = child(m, "normalTexture")) {
        mat.normal_map = self.texture(t, NO_COLOR_SPACE)!;
        mat.normal_scale = num(t, "scale", 1);
    }
    if (try e = child(m, "emissiveFactor")) {
        mat.emissive = linear_color((float)e.get_float_at(0)!, (float)e.get_float_at(1)!, (float)e.get_float_at(2)!);
    }
    if (m.has_key("alphaMode")) {
        String mode = m.get_string("alphaMode")!;
        if (mode == "MASK") mat.alpha_test = num(m, "alphaCutoff", 0.5f);
        if (mode == "BLEND") mat.transparent = true;
    }
    if (m.has_key("doubleSided") && m.get_bool("doubleSided")!) mat.side = DOUBLE_SIDE;
    if (try ext = child(m, "extensions")) {
        // transmission, and what the light does as it crosses the volume
        if (try t = child(ext, "KHR_materials_transmission")) {
            mat.transmission = num(t, "transmissionFactor", 0);
            if (try tt = child(t, "transmissionTexture")) mat.transmission_map = self.texture(tt, NO_COLOR_SPACE)!;
        }
        if (try vol = child(ext, "KHR_materials_volume")) {
            mat.thickness = num(vol, "thicknessFactor", 0);
            if (try tt = child(vol, "thicknessTexture")) mat.thickness_map = self.texture(tt, NO_COLOR_SPACE)!;
            mat.attenuation_distance = num(vol, "attenuationDistance", 0);
            if (try c = child(vol, "attenuationColor")) {
                mat.attenuation_color = linear_color(
                    (float)c.get_float_at(0)!,
                    (float)c.get_float_at(1)!,
                    (float)c.get_float_at(2)!
                );
            }
        }
        if (try ior = child(ext, "KHR_materials_ior")) mat.ior = num(ior, "ior", 1.5f);
        // the strength and color of its reflections where it isn't metal
        if (try sp = child(ext, "KHR_materials_specular")) {
            mat.specular_intensity = num(sp, "specularFactor", 1);
            if (try c = child(sp, "specularColorFactor")) {
                mat.specular_color = { (float)c.get_float_at(0)!, (float)c.get_float_at(1)!, (float)c.get_float_at(2)! };
            }
            if (try tt = child(sp, "specularTexture")) mat.specular_intensity_map = self.texture(tt, NO_COLOR_SPACE)!;
            if (try tt = child(sp, "specularColorTexture")) mat.specular_color_map = self.texture(tt, SRGB_COLOR_SPACE)!;
        }
        if (try d = child(ext, "KHR_materials_dispersion")) mat.dispersion = num(d, "dispersion", 0);
    }
    return mat;
}

fn BufferGeometry*? GltfReader.geometry(&self, Object* primitive) {
    if (primitive.has_key("mode") && primitive.get_int("mode")! != 4) return GLTF_INVALID~;
    Object* attrs = primitive.get("attributes")!;
    Accessor pos  = self.accessor(attrs.get_int("POSITION")!)!;
    Accessor normal;
    Accessor uv;
    Accessor color;
    if (attrs.has_key("NORMAL")) normal = self.accessor(attrs.get_int("NORMAL")!)!;
    if (attrs.has_key("TEXCOORD_0")) uv = self.accessor(attrs.get_int("TEXCOORD_0")!)!;
    if (attrs.has_key("COLOR_0")) color = self.accessor(attrs.get_int("COLOR_0")!)!;

    BufferGeometry* g = mem::new(BufferGeometry);
    g.init_lists();
    g.vertices.reserve(pos.count * drie::FLOATS_PER_VERTEX);
    for (sz i = 0; i < pos.count; i++) {
        Vector3 p = { pos.value(i, 0), pos.value(i, 1), pos.value(i, 2) };
        Vector3 n = normal.ptr ? (Vector3) { normal.value(i, 0), normal.value(i, 1), normal.value(i, 2) }
            : (Vector3) { 0, 1, 0 };
        Vector2 t = uv.ptr ? (Vector2) { uv.value(i, 0), uv.value(i, 1) } : (Vector2) { };
        Vector3 c = color.ptr ? (Vector3) { color.value(i, 0), color.value(i, 1), color.value(i, 2) }
            : (Vector3) { 1, 1, 1 };
        g.add_vertex(p, n, t, c);
    }
    if (primitive.has_key("indices")) {
        Accessor idx = self.accessor(primitive.get_int("indices")!)!;
        g.indices.reserve(idx.count);
        for (sz i = 0; i < idx.count; i++) g.indices.push(idx.index(i));
    } else {
        for (uint i = 0; i < pos.count; i++) g.indices.push(i);
    }
    if (!normal.ptr) g.compute_vertex_normals();
    return g;
}

fn Object3D*? GltfReader.node(&self, int index) {
    Object* n  = self.json.get("nodes")!.get_at(index);
    Group* obj = drie::group();
    if (n.has_key("name")) obj.name = n.get_string("name")!.copy(mem);
    if (try t = child(
        n,
        "translation"
    )) obj.position = { (float)t.get_float_at(0)!, (float)t.get_float_at(1)!, (float)t.get_float_at(2)! };
    if (try s = child(
        n,
        "scale"
    )) obj.scale = { (float)s.get_float_at(0)!, (float)s.get_float_at(1)!, (float)s.get_float_at(2)! };
    if (try q = child(n, "rotation")) {
        // quaternion -> rotation matrix -> euler
        float x      = (float)q.get_float_at(0)!;
        float y      = (float)q.get_float_at(1)!;
        float z      = (float)q.get_float_at(2)!;
        float w      = (float)q.get_float_at(3)!;
        drie::Mat4 m = drie::mat4_identity();
        m[0] = { 1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0 };
        m[1] = { 2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0 };
        m[2] = { 2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0 };
        obj.rotation = drie::euler_from_rotation(m);
    }
    if (n.has_key("mesh")) {
        int mesh_index     = n.get_int("mesh")!;
        Object* primitives = self.json.get("meshes")!.get_at(mesh_index).get("primitives")!;
        for (sz i = 0; i < primitives.get_len(); i++) {
            Object* prim = primitives.get_at(i);
            // geometry is shared between nodes that use the same glTF mesh
            if (!self.geometries[mesh_index][i]) self.geometries[mesh_index][i] = self.geometry(prim)!;
            Material* mat = prim.has_key("material") ? self.model.materials[prim.get_int("material")!]
                : (Material*)drie::mesh_standard_material();
            Mesh* m       = drie::mesh(self.geometries[mesh_index][i], mat);
            m.name = obj.name;
            obj.add(m);
            self.model.meshes.push(m);
        }
    }
    if (try kids = child(n, "children")) {
        for (sz i = 0; i < kids.get_len(); i++) obj.add(self.node(kids.get_int_at(i)!)!);
    }
    return obj;
}

fn GltfModel*? load_gltf(String path) {
    char[] data = file::load(mem, path)!;
    defer mem::free(data.ptr);
    GltfReader r;
    r.textures.allocator = mem;
    r.geometries.allocator = mem;
    defer r.textures.free();
    sz slash = -1;
    foreach (i, c : path) {
        if (c == '/') slash = i;
    }
    r.dir = path[:slash + 1];

    String text;
    char[] external_bin;
    defer if (external_bin.ptr) mem::free(external_bin.ptr);
    if (data.len >= 20 && data[0] == 'g' && data[1] == 'l' && data[2] == 'T' && data[3] == 'F') {
        // GLB container: header, JSON chunk, optional BIN chunk
        sz json_len = read_u32(data, 12);
        if (20 + json_len > data.len) return GLTF_INVALID~;
        text = (String)data[20:json_len];
        sz at = 20 + json_len;
        if (at + 8 <= data.len) {
            sz bin_len = read_u32(data, at);
            if (at + 8 + bin_len > data.len) return GLTF_INVALID~;
            r.bin = data[at + 8:bin_len];
        }
    } else {
        text = (String)data;
    }
    r.json = json::parse(mem, text)!;
    defer r.json.free();
    if (!r.bin.len) {
        Object* buffer = r.json.get("buffers")!.get_at(0);
        @pool() {
            external_bin = file::load(mem, string::tformat("%s%s", r.dir, buffer.get_string("uri")!))!;
        };
        r.bin = external_bin;
    }

    GltfModel* model = mem::new(GltfModel);
    model.materials.allocator = mem;
    model.meshes.allocator = mem;
    r.model = model;
    if (try mats = child(r.json, "materials")) {
        for (sz i = 0; i < mats.get_len(); i++) model.materials.push(r.material(mats.get_at(i))!);
    }
    if (try meshes = child(r.json, "meshes")) {
        for (sz i = 0; i < meshes.get_len(); i++) {
            r.geometries.push(mem::new_array(BufferGeometry*, meshes.get_at(i).get("primitives")!.get_len()));
        }
    }
    defer {
        foreach (g : r.geometries) mem::free(g.ptr);
        r.geometries.free();
    }

    model.scene = drie::group();
    Object* scenes = r.json.get("scenes")!;
    Object* nodes  = scenes.get_at((sz)num(r.json, "scene", 0)).get("nodes")!;
    for (sz i = 0; i < nodes.get_len(); i++) model.scene.add(r.node(nodes.get_int_at(i)!)!);
    return model;
}