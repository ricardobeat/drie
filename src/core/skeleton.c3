// CPU skeletal skinning: `Bone`, `Skeleton`, `SkinnedMesh`, and `BufferGeometry` skin
// attributes. Bones are `Object3D` nodes (any child mesh or light moves with them), and each
// vertex blends up to `BONES_PER_VERTEX` (4) bone transforms. Skinning runs on the CPU in
// `SkinnedMesh.update()` so shaders, shadows, fog, and `Raycaster` consume posed vertices
// directly; call `update()` once per frame after posing (`Renderer.render` does not call it).
// Constructors allocate on the heap (`mem`); glTF skins and animation clips are not included.
module drie;
import std::collections::list, std::math;
import sokol;

// Maximum number of bones influencing a single vertex.
const int BONES_PER_VERTEX = 4;

// ---- Skin attributes

// Grows `skin_indices` and `skin_weights` to cover every vertex (new entries have zero weight).
fn void BufferGeometry.ensure_skin(&self) {
    sz want = (sz)self.vertex_count() * BONES_PER_VERTEX;
    while (self.skin_indices.len() < want) self.skin_indices.push(0);
    while (self.skin_weights.len() < want) self.skin_weights.push(0);
}

// Assigns up to four `Skeleton.bones` indices and weights to `vertex`. Weights are normalized
// during skinning and need not sum to 1.
fn void BufferGeometry.set_skin(&self, uint vertex, ushort[BONES_PER_VERTEX] bones, float[BONES_PER_VERTEX] weights) {
    self.ensure_skin();
    sz o = (sz)vertex * BONES_PER_VERTEX;
    for (int k = 0; k < BONES_PER_VERTEX; k++) {
        self.skin_indices[o + k] = bones[k];
        self.skin_weights[o + k] = weights[k];
    }
}

// Binds `vertex` between two bones: `blend = 0` follows `bone_a`, `1` follows `bone_b`.
fn void BufferGeometry.set_skin_blend(&self, uint vertex, int bone_a, int bone_b, float blend) {
    self.set_skin(vertex, { (ushort)bone_a, (ushort)bone_b, 0, 0 }, { 1 - blend, blend, 0, 0 });
}

// Binds vertices `[from, to)` rigidly to one bone (useful right after `BufferGeometry.merge`):
// 
//   uint first = g.vertex_count(); g.merge(part); g.skin_range(first, g.vertex_count(), bone);
fn void BufferGeometry.skin_range(&self, uint from, uint to, int bone) {
    for (uint i = from; i < to; i++) self.set_skin(i, { (ushort)bone, 0, 0, 0 }, { 1, 0, 0, 0 });
}

// ---- Bone

// Scene-graph joint node (`Object3D`) driven by a `Skeleton`; attached children move with it.
struct Bone {
    inline Object3D object;
}

// Allocates a `Bone` on the heap with optional `name`.
fn Bone* bone(String name = "") {
    Bone* b = mem::new(Bone);
    b.init(OBJECT);
    b.name = name;
    return b;
}

// ---- Skeleton

// Ordered bone array paired with bind-pose inverse transforms and current skinning matrices.
struct Skeleton {
    // Bones in index order matching `BufferGeometry.skin_indices`.
    Bone*[] bones;
    // World-to-bone inverse matrices captured in the bind pose.
    Mat4[] bone_inverses;
    // Per-bone `matrix_world * bone_inverses[i]` updated by `Skeleton.update()`.
    Mat4[] bone_matrices;
}

// Allocates a `Skeleton` referencing `bones`. When `bone_inverses` is empty (or mismatches
// `bones.len`), computes inverses from the bones' current world pose.
fn Skeleton* skeleton(Bone*[] bones, Mat4[] bone_inverses = { }) {
    Skeleton* s = mem::new(Skeleton);
    s.bones = mem::new_array(Bone*, bones.len);
    s.bone_inverses = mem::new_array(Mat4, bones.len);
    s.bone_matrices = mem::new_array(Mat4, bones.len);
    foreach (i, b : bones) s.bones[i] = b;
    if (bone_inverses.len == bones.len) {
        foreach (i, m : bone_inverses) s.bone_inverses[i] = m;
    } else {
        s.calculate_inverses();
    }
    s.update();
    return s;
}

// Records the bones' current world transforms as the bind pose (`bone_inverses`).
fn void Skeleton.calculate_inverses(&self) {
    foreach (i, b : self.bones) {
        b.update_world_matrix_from_parents();
        self.bone_inverses[i] = b.matrix_world.invert();
    }
}

// Restores every bone's local `position` and `rotation` to the bind pose (assumes unscaled
// bones ordered with parents before children).
fn void Skeleton.pose(&self) {
    // Parents come first so each child can invert its parent's restored world transform.
    foreach (i, b : self.bones) {
        Mat4 local = self.bone_inverses[i].invert();
        if (b.parent) {
            b.parent.update_world_matrix_from_parents();
            local = b.parent.matrix_world.invert().mul(local);
        }
        // Bones are unscaled, so translation and Euler angles recover the local transform.
        b.position = local.get_position();
        b.rotation = euler_from_rotation(local);
    }
}

// Refreshes every bone's world matrix and recomputes `bone_matrices`.
fn void Skeleton.update(&self) {
    foreach (i, b : self.bones) {
        b.update_world_matrix_from_parents();
        self.bone_matrices[i] = b.matrix_world.mul(self.bone_inverses[i]);
    }
}

// Returns the first bone whose `name` matches, or `null`.
fn Bone* Skeleton.get_bone_by_name(&self, String name) {
    foreach (b : self.bones) {
        if (b.name == name) return b;
    }
    return null;
}

// Returns the index of `bone` in `bones` for skin attributes, or `-1` if absent.
fn int Skeleton.index_of(&self, Bone* bone) {
    foreach (i, b : self.bones) {
        if (b == bone) return (int)i;
    }
    return -1;
}

// ---- SkinnedMesh

// Mesh deformed on the CPU from `bind_geometry` into `geometry` via a bound `Skeleton`.
// 
//   Bone* hip = drie::bone("hip");
//   Bone* knee = drie::bone("knee");
//   hip.add(knee);
//   knee.position.y = -0.4f;
// 
//   BufferGeometry* leg = ...;                              // modelled in bind pose
//   leg.skin_range(0, leg.vertex_count(), 0);
//   leg.set_skin(7, { 0, 1, 0, 0 }, { 0.5f, 0.5f, 0, 0 });
// 
//   SkinnedMesh* mesh = drie::skinned_mesh(leg, material);
//   mesh.add(hip);
//   mesh.bind(drie::skeleton({ hip, knee }));
//   scene.add(mesh);
// 
//   knee.rotation.x = 0.8f;                                 // each frame: pose, then skin
//   mesh.update();
struct SkinnedMesh {
    inline Mesh mesh;
    // Attached skeleton driving vertex deformation.
    Skeleton* skeleton;
    // Mesh world transform at bind time.
    Mat4 bind_matrix;
    // Inverse of `bind_matrix`.
    Mat4 bind_matrix_inverse;
    // Unposed source geometry holding rest positions, normals, and skin attributes;
    // `Mesh.geometry` is the posed copy submitted to the renderer.
    BufferGeometry* bind_geometry;
}

// Allocates a `SkinnedMesh` with a private posed copy of `geometry` (`bind_geometry` stays
// untouched). Uses a default `MeshBasicMaterial` when `material` is `null`.
fn SkinnedMesh* skinned_mesh(BufferGeometry* geometry, Material* material = null) {
    SkinnedMesh* m = mem::new(SkinnedMesh);
    m.init(MESH);
    m.instance_data.allocator = mem;
    m.material = material ? material : (Material*)mesh_basic_material();
    m.bind_geometry = geometry;
    BufferGeometry* posed = mem::new(BufferGeometry);
    posed.init_lists();
    posed.vertices.reserve(geometry.vertices.len());
    posed.indices.reserve(geometry.indices.len());
    foreach (v : geometry.vertices) posed.vertices.push(v);
    foreach (i : geometry.indices) posed.indices.push(i);
    m.geometry = posed;
    m.bind_matrix = mat4_identity();
    m.bind_matrix_inverse = mat4_identity();
    return m;
}

// Binds `skeleton`, capturing both the mesh's and bones' current transforms as the bind pose.
fn void SkinnedMesh.bind(&self, Skeleton* skeleton) {
    self.update_world_matrix_from_parents();
    skeleton.calculate_inverses();
    self.bind_with(skeleton, self.matrix_world);
}

// Binds `skeleton` keeping its existing `bone_inverses`, with `bind_matrix` as the mesh's
// bind-time world transform.
fn void SkinnedMesh.bind_with(&self, Skeleton* skeleton, Mat4 bind_matrix) {
    self.skeleton = skeleton;
    self.bind_matrix = bind_matrix;
    self.bind_matrix_inverse = bind_matrix.invert();
}

// Skins vertices to the bound skeleton's current pose and refreshes the GPU vertex buffer.
// Call once per frame after posing the bones.
fn void SkinnedMesh.update(&self) {
    Skeleton* sk = self.skeleton;
    if (!sk) return;
    sk.update();
    // Transform bind-pose mesh space -> bind world -> posed world -> current mesh local space.
    self.update_world_matrix_from_parents();
    Mat4 to_local = self.matrix_world.invert();
    Mat4[] skin   = mem::new_array(Mat4, sk.bones.len);
    defer free(skin);
    foreach (i, m : sk.bone_matrices) skin[i] = to_local.mul(m).mul(self.bind_matrix);
    self.skin_with(skin);
}

// Deforms `geometry` using caller-supplied mesh-space skinning matrices (`bind_matrix_inverse *
// bone_matrix * bind_matrix`), recomputes the bounding sphere, and re-uploads `vbuf` if active.
fn void SkinnedMesh.skin_with(&self, Mat4[] skin) {
    BufferGeometry* src = self.bind_geometry;
    BufferGeometry* dst = self.geometry;
    uint count          = src.vertex_count();
    sz skinned          = math::min((sz)count, src.skin_weights.len() / BONES_PER_VERTEX);
    for (sz v = 0; v < skinned; v++) {
        sz o        = v * FLOATS_PER_VERTEX;
        sz s        = v * BONES_PER_VERTEX;
        Vector3 p   = { src.vertices[o], src.vertices[o + 1], src.vertices[o + 2] };
        Vector3 n   = { src.vertices[o + 3], src.vertices[o + 4], src.vertices[o + 5] };
        Vector3 sp;
        Vector3 sn;
        float total = 0;
        for (int k = 0; k < BONES_PER_VERTEX; k++) {
            float w = src.skin_weights[s + k];
            if (w <= 0) continue;
            int b = src.skin_indices[s + k];
            if (b >= skin.len) continue;
            Mat4* m = &skin[b];
            sp += m.transform_point(p) * w;
            // Upper 3x3 only: bones are assumed to have uniform (or unit) scale.
            sn += (Vector3) {
                (*m)[0][0] * n.x + (*m)[1][0] * n.y + (*m)[2][0] * n.z,
                (*m)[0][1] * n.x + (*m)[1][1] * n.y + (*m)[2][1] * n.z,
                (*m)[0][2] * n.x + (*m)[1][2] * n.y + (*m)[2][2] * n.z,
            } * w;
            total += w;
        }
        // Leave vertices with zero total weight at their rest pose.
        if (total <= 0) continue;
        sp /= total;
        float len = sn.length();
        if (len > 1e-12f) sn /= len;
        dst.vertices[o] = sp.x;
        dst.vertices[o + 1] = sp.y;
        dst.vertices[o + 2] = sp.z;
        dst.vertices[o + 3] = sn.x;
        dst.vertices[o + 4] = sn.y;
        dst.vertices[o + 5] = sn.z;
    }
    dst.compute_bounding_sphere();
    // Only vertex data changed; replace the vertex buffer and keep the existing index buffer.
    if (dst.uploaded) {
        sg::destroy_buffer(dst.vbuf);
        dst.vbuf = sg::make_buffer(
            &&{
                .data = { .ptr = dst.vertices.entries, .size = (usz)dst.vertices.len() * float::size },
            }
        );
    }
}