// Skeletons: Bone, Skeleton and SkinnedMesh, like three's.
//
//   Bone* hip = drie::bone("hip");
//   Bone* knee = drie::bone("knee");
//   hip.add(knee);
//   knee.position.y = -0.4f;
//
//   BufferGeometry* leg = ...;                              // modelled in the bind pose
//   leg.skin_range(0, leg.vertex_count(), 0);               // everything follows bone 0 ...
//   leg.set_skin(7, { 0, 1, 0, 0 }, { 0.5f, 0.5f, 0, 0 });  // ... but vertex 7 bends with the knee
//
//   SkinnedMesh* mesh = drie::skinned_mesh(leg, material);
//   mesh.add(hip);                                          // bones are ordinary scene objects
//   mesh.bind(drie::skeleton({ hip, knee }));
//   scene.add(mesh);
//
//   knee.rotation.x = 0.8f;                                 // each frame: pose the bones, then
//   mesh.update();
//
// Bones are Object3Ds: anything added to one (a Mesh, a light) moves with it.
//
// Unlike three, the vertices are skinned on the CPU, in SkinnedMesh.update, and it is not
// called by the renderer: call it once per frame after posing. That keeps skinning out of the
// shaders (shadows, fog and every material work unchanged) and is plenty for a few characters
// of some thousand vertices; a crowd would want the bone matrices in the vertex shader.
// Not here either: glTF skins, and three's animation system (AnimationClip, AnimationMixer).
module drie;
import std::collections::list, std::math;
import sokol;

const int BONES_PER_VERTEX = 4;

// ----------------------------------------------------------------------------
// Skin attributes of a geometry
// ----------------------------------------------------------------------------

// Makes the skin lists cover every vertex (new ones are unskinned: all weights 0).
fn void BufferGeometry.ensure_skin(&self) {
    sz want = (sz)self.vertex_count() * BONES_PER_VERTEX;
    while (self.skin_indices.len() < want) self.skin_indices.push(0);
    while (self.skin_weights.len() < want) self.skin_weights.push(0);
}

// The bones (indices into Skeleton.bones) that move a vertex, and how much each one counts.
// Weights don't have to add up to 1: they are normalized when skinning.
fn void BufferGeometry.set_skin(&self, uint vertex, ushort[BONES_PER_VERTEX] bones, float[BONES_PER_VERTEX] weights) {
    self.ensure_skin();
    sz o = (sz)vertex * BONES_PER_VERTEX;
    for (int k = 0; k < BONES_PER_VERTEX; k++) {
        self.skin_indices[o + k] = bones[k];
        self.skin_weights[o + k] = weights[k];
    }
}

// A vertex shared by two bones: `blend` = 0 follows bone_a, 1 follows bone_b.
fn void BufferGeometry.set_skin_blend(&self, uint vertex, int bone_a, int bone_b, float blend) {
    self.set_skin(vertex, { (ushort)bone_a, (ushort)bone_b, 0, 0 }, { 1 - blend, blend, 0, 0 });
}

// Binds the vertices from..to (exclusive) rigidly to one bone. Handy right after a merge():
//   uint first = g.vertex_count(); g.merge(part); g.skin_range(first, g.vertex_count(), bone);
fn void BufferGeometry.skin_range(&self, uint from, uint to, int bone) {
    for (uint i = from; i < to; i++) self.set_skin(i, { (ushort)bone, 0, 0, 0 }, { 1, 0, 0, 0 });
}

// ----------------------------------------------------------------------------
// Bone
// ----------------------------------------------------------------------------

struct Bone {
    inline Object3D object;
}

fn Bone* bone(String name = "") {
    Bone* b = mem::new(Bone);
    b.init(OBJECT);
    b.name = name;
    return b;
}

// ----------------------------------------------------------------------------
// Skeleton
// ----------------------------------------------------------------------------

struct Skeleton {
    Bone*[] bones;
    // World-to-bone transforms of the bind pose.
    Mat4[] bone_inverses;
    // Where each bone has taken its part of the bind pose: world * inverse (see update).
    Mat4[] bone_matrices;
}

// Without `bone_inverses`, the bones' current pose is the bind pose.
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

// Takes the bones' current pose as the bind pose.
fn void Skeleton.calculate_inverses(&self) {
    foreach (i, b : self.bones) {
        b.update_world_matrix_from_parents();
        self.bone_inverses[i] = b.matrix_world.invert();
    }
}

// Puts the bones back into the bind pose.
fn void Skeleton.pose(&self) {
    // parents come back first, so that their children can be placed relative to them
    foreach (i, b : self.bones) {
        Mat4 local = self.bone_inverses[i].invert();
        if (b.parent) {
            b.parent.update_world_matrix_from_parents();
            local = b.parent.matrix_world.invert().mul(local);
        }
        // bones are not scaled
        b.position = local.get_position();
        b.rotation = euler_from_rotation(local);
    }
}

// Recomputes bone_matrices from where the bones are now.
fn void Skeleton.update(&self) {
    foreach (i, b : self.bones) {
        b.update_world_matrix_from_parents();
        self.bone_matrices[i] = b.matrix_world.mul(self.bone_inverses[i]);
    }
}

fn Bone* Skeleton.get_bone_by_name(&self, String name) {
    foreach (b : self.bones) {
        if (b.name == name) return b;
    }
    return null;
}

// Index of a bone, for the geometry's skin attributes (-1 if it is not part of the skeleton).
fn int Skeleton.index_of(&self, Bone* bone) {
    foreach (i, b : self.bones) {
        if (b == bone) return (int)i;
    }
    return -1;
}

// ----------------------------------------------------------------------------
// SkinnedMesh
// ----------------------------------------------------------------------------

struct SkinnedMesh {
    inline Mesh mesh;
    Skeleton* skeleton;
    // The mesh's world transform when it was bound, and its inverse.
    Mat4 bind_matrix;
    Mat4 bind_matrix_inverse;
    // The geometry as it was modelled, with the skin attributes. `geometry` is the posed copy
    // that gets drawn.
    BufferGeometry* bind_geometry;
}

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

// Attaches the skeleton, with the bones and the mesh where they are now as the bind pose.
fn void SkinnedMesh.bind(&self, Skeleton* skeleton) {
    self.update_world_matrix_from_parents();
    skeleton.calculate_inverses();
    self.bind_with(skeleton, self.matrix_world);
}

// Attaches the skeleton as it is, for bone inverses that come from elsewhere (a file).
// `bind_matrix` is the mesh's world transform in the bind pose.
fn void SkinnedMesh.bind_with(&self, Skeleton* skeleton, Mat4 bind_matrix) {
    self.skeleton = skeleton;
    self.bind_matrix = bind_matrix;
    self.bind_matrix_inverse = bind_matrix.invert();
}

// Moves the vertices to the bones' current pose. Call once per frame, after posing the bones.
fn void SkinnedMesh.update(&self) {
    Skeleton* sk = self.skeleton;
    if (!sk) return;
    sk.update();
    // bind pose (mesh space) -> world at bind time -> with the bone -> back into mesh space
    self.update_world_matrix_from_parents();
    Mat4 to_local = self.matrix_world.invert();
    Mat4[] skin   = mem::new_array(Mat4, sk.bones.len);
    defer free(skin);
    foreach (i, m : sk.bone_matrices) skin[i] = to_local.mul(m).mul(self.bind_matrix);
    self.skin_with(skin);
}

// Moves the vertices with one matrix per bone, from the bind pose in mesh space to the posed
// mesh in mesh space (three's bindMatrixInverse * boneMatrix * bindMatrix), for bones posed
// elsewhere (a skeleton that isn't drie's own).
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
            // rotation only: bones are not scaled unevenly
            sn += (Vector3) {
                (*m)[0][0] * n.x + (*m)[1][0] * n.y + (*m)[2][0] * n.z,
                (*m)[0][1] * n.x + (*m)[1][1] * n.y + (*m)[2][1] * n.z,
                (*m)[0][2] * n.x + (*m)[1][2] * n.y + (*m)[2][2] * n.z,
            } * w;
            total += w;
        }
        // a vertex without bones stays where it was modelled
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
    // only the vertices changed: swap the vertex buffer, keep the indices
    if (dst.uploaded) {
        sg::destroy_buffer(dst.vbuf);
        dst.vbuf = sg::make_buffer(
            &&{
                .data = { .ptr = dst.vertices.entries, .size = (usz)dst.vertices.len() * float::size },
            }
        );
    }
}