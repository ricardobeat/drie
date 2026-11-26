// Scene graph core: `Object3D`, `Group`, `Scene`, `Fog`/`VolumetricFog`, `Mesh`,
// `InstancedMesh`, `Lod`, `Line` variants, cameras, and lights. Coordinates are right-handed
// Y-up meters with intrinsic XYZ Euler rotations in radians; constructors allocate on the heap
// (`mem`) and `Object3D.children` uses `mem`. Consumed by `Renderer.render`, `Raycaster`,
// `Skeleton`, `walk_casters`, `IrradianceVolume`, `RadianceCascades`, and `ReflectionProbes`.
module drie;
import std::collections::list, libc;
import sokol;

// Discriminator stored in `Object3D.kind` for downcasting scene-graph nodes.
enum ObjectKind {
    // Plain transform node (`Object3D`).
    OBJECT,
    // Logical container (`Group`).
    GROUP,
    // Root scene container (`Scene`).
    SCENE,
    // Renderable triangle or line mesh (`Mesh`, `InstancedMesh`, `SkinnedMesh`, `Line`).
    MESH,
    // Distance-switched level-of-detail container (`Lod`).
    LOD,
    // Projection camera (`PerspectiveCamera`, `OrthographicCamera`).
    CAMERA,
    // Scene light (`Light` subtypes).
    LIGHT,
}

// Base scene-graph node owning local/world transforms, hierarchy links, and render flags.
struct Object3D {
    // Concrete node subtype for safe downcasting.
    ObjectKind kind;
    // Optional lookup label matched by `Object3D.get_object_by_name`.
    String name;
    // Local translation relative to `parent`, in meters.
    Vector3 position;
    // Local XYZ Euler rotation in radians.
    Euler rotation;
    // Local non-uniform scale factors (defaults to `{ 1, 1, 1 }`).
    Vector3 scale;
    // Reference up vector used by `look_at` (defaults to `{ 0, 1, 0 }`).
    Vector3 up;
    // When false, the node and its subtree are skipped during rendering.
    bool visible;
    // For meshes: skip drawing when the bounding sphere falls outside the camera frustum.
    bool frustum_culled;
    // For meshes and lights: render into shadow maps.
    bool cast_shadow;
    // For meshes: sample shadow maps during lighting.
    bool receive_shadow;
    // For meshes: render only into shadow maps (with `cast_shadow`) as a cheap proxy for a
    // detailed mesh whose own `cast_shadow` is false; excluded from main pass, picking, and GI.
    bool shadow_only;
    // Draw priority among transparent meshes: higher values render later regardless of depth.
    int render_order;
    // Parent node in the scene graph, or `null` at the root.
    Object3D* parent;
    // Direct child nodes, heap-allocated (`mem`).
    List{Object3D*} children;
    // Local transform matrix composed from `position`, `rotation`, and `scale`.
    Mat4 matrix;
    // Accumulated world transform matrix (`parent.matrix_world * matrix`).
    Mat4 matrix_world;
    // Cached `position` from the last `update_world` pass to skip unchanged matrices.
    Vector3 placed_position;
    // Cached `rotation` from the last `update_world` pass.
    Euler placed_rotation;
    // Cached `scale` from the last `update_world` pass.
    Vector3 placed_scale;
    // True once `placed_*` mirrors `matrix`.
    bool placed;
    // Set when `matrix_world` changed outside `update_world` and children still need propagation.
    bool world_dirty;
    // Bitmask of `visible` and mesh `cast_shadow` as `update_world` last observed them.
    char seen_switches;
    // When true (default), `update_world` rebuilds `matrix` when TRS changes. When false, the
    // caller writes `matrix` directly (or calls `update_matrix`).
    bool matrix_auto_update;
    // When true (default), `update_matrix_world` traverses this node and its descendants. Set
    // false on static subtrees to skip per-frame traversal; call `update_matrix_world` manually
    // after moving anything inside them.
    bool matrix_world_auto_update;
    // Opaque pointer reserved for application data.
    void* user_data;
}

// Initializes default transform, flags, identity matrices, and heap allocator on `children`.
fn void Object3D.init(&self, ObjectKind kind) {
    self.kind = kind;
    self.scale = { 1, 1, 1 };
    self.up = { 0, 1, 0 };
    self.visible = true;
    self.frustum_culled = true;
    self.matrix_auto_update = true;
    self.matrix_world_auto_update = true;
    self.matrix = mat4_identity();
    self.matrix_world = mat4_identity();
    // Keep `children` on the heap so it outlives any caller `@pool` frame.
    self.children.allocator = mem;
}

// World changes tracked since `walk_casters` last scanned the scene: meshes whose
// `matrix_world` changed (with `moved_meshes_dropped` counting overflows that force a full
// rescan), and `scene_structure_changes` counting graph edits or `visible`/`cast_shadow` flips.
List{Mesh*} moved_meshes @private;
uint moved_meshes_dropped @private;
uint scene_structure_changes @private;

// Records a `visible` or mesh `cast_shadow` change since `update_world` last checked `o`.
fn void note_switches(Object3D* o) @private {
    char now = (char)((o.visible ? 1 : 0) | (o.cast_shadow && o.kind == MESH ? 2 : 0) | 4);
    if (now == o.seen_switches) return;
    o.seen_switches = now;
    scene_structure_changes++;
}

fn void note_moved_mesh(Mesh* m) @private {
    if (!moved_meshes.allocator) moved_meshes.init(mem);
    if (moved_meshes.len() >= 1 << 16) {
        moved_meshes.clear();
        moved_meshes_dropped++;
    }
    moved_meshes.push(m);
}

// Reparents `child` under this node (detaching from any existing parent) and returns `self`.
fn Object3D* Object3D.add(&self, Object3D* child) {
    if (child == self) return self;
    scene_structure_changes++;
    if (child.parent) child.parent.remove(child);
    child.parent = self;
    child.placed = false;
    self.children.push(child);
    return self;
}

// Detaches `child` from `children` if present, clearing its `parent`, and returns `self`.
fn Object3D* Object3D.remove(&self, Object3D* child) {
    scene_structure_changes++;
    foreach (i, c : self.children) {
        if (c == child) {
            self.children.remove_at(i);
            child.parent = null;
            break;
        }
    }
    return self;
}

// Detaches all direct children from this node without freeing them.
fn void Object3D.clear(&self) {
    scene_structure_changes++;
    foreach (c : self.children) c.parent = null;
    self.children.clear();
}

// Depth-first search for the first node in this subtree with `node.name == name`, or `null`.
fn Object3D* Object3D.get_object_by_name(&self, String name) {
    if (self.name == name) return self;
    foreach (c : self.children) {
        Object3D* found = c.get_object_by_name(name);
        if (found) return found;
    }
    return null;
}

// Recursively clones this node and its descendants without a parent, sharing `BufferGeometry`
// and `Material` pointers (e.g. `scene.add(model.get_object_by_name("chair").clone())`). Plain
// meshes keep their `draw_mode`; lights, cameras, LODs, and instanced meshes clone as `Group`s.
fn Object3D* Object3D.clone(&self) {
    Object3D* copy;
    if (self.kind == MESH && ((Mesh*)self).instance_data.len() == 0) {
        Mesh* from = (Mesh*)self;
        copy = mesh(from.geometry, from.material);
        ((Mesh*)copy).draw_mode = from.draw_mode;
    } else {
        copy = group();
    }
    copy.name = self.name;
    copy.position = self.position;
    copy.rotation = self.rotation;
    copy.scale = self.scale;
    copy.up = self.up;
    copy.visible = self.visible;
    copy.frustum_culled = self.frustum_culled;
    copy.cast_shadow = self.cast_shadow;
    copy.receive_shadow = self.receive_shadow;
    copy.user_data = self.user_data;
    foreach (c : self.children) copy.add(c.clone());
    return copy;
}

// Callback invoked on each node visited by `Object3D.traverse`.
alias TraverseCallback = fn void(Object3D*);

// Visits this node and every descendant in pre-order depth-first traversal.
fn void Object3D.traverse(&self, TraverseCallback cb) {
    cb(self);
    foreach (c : self.children) c.traverse(cb);
}

// Recomputes local `matrix` from `position`, `rotation`, and `scale`.
fn void Object3D.update_matrix(&self) {
    self.matrix = mat4_compose(self.position, self.rotation, self.scale);
}

// Updates `matrix_world` for this node and its auto-updating descendants, recomputing only
// nodes whose TRS changed or whose ancestor moved.
fn void Object3D.update_matrix_world(&self) => self.update_world(true);

// Recursive world-matrix update pass tracking moved meshes and switches for `walk_casters`.
fn void Object3D.update_world(&self, bool parent_moved) {
    note_switches(self);
    // With `matrix_auto_update` off, `matrix` may have changed externally; recompute and diff.
    bool moved = !self.matrix_auto_update;
    if (self.matrix_auto_update
        && (!self.placed || self.position != self.placed_position || self.rotation != self.placed_rotation
            || self.scale != self.placed_scale)) {
        moved = true;
        self.update_matrix();
        self.placed_position = self.position;
        self.placed_rotation = self.rotation;
        self.placed_scale = self.scale;
        self.placed = true;
    }
    if (moved || parent_moved) {
        Mat4 world = self.parent ? self.parent.matrix_world.mul(self.matrix) : self.matrix;
        // Root of the walk checks `memcmp` so unchanged transforms do not force child updates.
        moved = libc::memcmp(&world, &self.matrix_world, Mat4::size) != 0;
        self.matrix_world = world;
    }
    // Propagate updates made earlier by `update_world_matrix_from_parents` (e.g. posed bones).
    moved = moved || self.world_dirty;
    if (moved && self.kind == MESH) note_moved_mesh((Mesh*)self);

    self.world_dirty = false;
    foreach (c : self.children) {
        // Still record visibility/shadow toggles on static subtrees for shadow-atlas tracking.
        if (c.matrix_world_auto_update) {
            c.update_world(moved);
        } else {
            note_switches(c);
        }
    }
}

// Recomputes `matrix_world` along the ancestor chain down to this node and flags `world_dirty`
// if the matrix changed.
fn void Object3D.update_world_matrix_from_parents(&self) {
    if (self.parent) self.parent.update_world_matrix_from_parents();
    if (self.matrix_auto_update) self.update_matrix();
    Mat4 world = self.parent ? self.parent.matrix_world.mul(self.matrix) : self.matrix;
    // Ensure the next `update_world` still propagates this change to descendants.
    if (libc::memcmp(&world, &self.matrix_world, Mat4::size) != 0) self.world_dirty = true;
    self.matrix_world = world;
}

// Updates ancestor transforms and returns this node's world-space position.
fn Vector3 Object3D.get_world_position(&self) {
    self.update_world_matrix_from_parents();
    return self.matrix_world.get_position();
}

// Rotates this node to face world point `(x, y, z)`. Cameras and lights orient `-Z` toward the
// target; other objects orient `+Z`.
fn void Object3D.look_at(&self, float x, float y, float z) => self.look_at_vec({ x, y, z });

// Rotates this node to face world-space `target` (cameras and lights face `-Z`, others `+Z`).
fn void Object3D.look_at_vec(&self, Vector3 target) {
    Vector3 eye = self.get_world_position();
    bool neg_z  = self.kind == CAMERA || self.kind == LIGHT;
    Mat4 rot    = mat4_look_at(eye, target, self.up, neg_z);
    if (self.parent) {
        // Strip parent world rotation (assumes uniform parent scale).
        Mat4 pm = self.parent.matrix_world;
        pm[3] = { 0, 0, 0, 1 };
        Vector3 s = { ((Vector3) { pm[0][0], pm[0][1], pm[0][2] }).length(), 0, 0 };
        if (s.x > 0) {
            for (int c = 0; c < 3; c++) for (int r = 0; r < 3; r++) pm[c][r] /= s.x;
        }
        rot = pm.transpose().mul(rot);
    }
    self.rotation = euler_from_rotation(rot);
}

// Adds `angle` radians to `rotation.x`.
fn void Object3D.rotate_x(&self, float angle) => self.rotation.x += angle;
// Adds `angle` radians to `rotation.y`.
fn void Object3D.rotate_y(&self, float angle) => self.rotation.y += angle;
// Adds `angle` radians to `rotation.z`.
fn void Object3D.rotate_z(&self, float angle) => self.rotation.z += angle;

// Translates `position` by `distance` along local `axis` rotated by `rotation`.
fn void Object3D.translate_on_axis(&self, Vector3 axis, float distance) {
    Mat4 r = mat4_compose({ 0, 0, 0 }, self.rotation, { 1, 1, 1 });
    self.position += r.transform_point(axis) * distance;
}
// Translates `d` meters along local +X.
fn void Object3D.translate_x(&self, float d) => self.translate_on_axis({ 1, 0, 0 }, d);
// Translates `d` meters along local +Y.
fn void Object3D.translate_y(&self, float d) => self.translate_on_axis({ 0, 1, 0 }, d);
// Translates `d` meters along local +Z.
fn void Object3D.translate_z(&self, float d) => self.translate_on_axis({ 0, 0, 1 }, d);

// Allocates an initialized `Object3D` on the heap (`mem`).
fn Object3D* object3d() {
    Object3D* o = mem::new(Object3D);
    o.init(OBJECT);
    return o;
}

// ---- Group, Scene, and Fog

// Transform container for grouping child `Object3D` nodes.
struct Group {
    inline Object3D object;
}

// Allocates an empty `Group` on the heap (`mem`).
fn Group* group() {
    Group* g = mem::new(Group);
    g.init(GROUP);
    return g;
}

// Distance attenuation curve used by `Fog`.
enum FogType {
    // Linear ramp from `near` to `far`.
    LINEAR,
    // Gaussian exponential falloff: `exp(-(density * d)^2)`.
    EXP2,
    // Beer-Lambert exponential falloff: `exp(-density * d)`.
    EXP,
}

// Lit and volumetric fog settings embedded in every `Fog` (`fog.volumetric`,
// `fog.light_scatter`, etc.). With `volumetric` enabled and at least one scattering term above
// zero (`light_scatter`, `sun_shafts`, or `noise_amount`), `Renderer.render` ray-marches a
// camera-aligned froxel grid lit by the sun's cascaded shadow, shadowed spot lights, and
// clustered point lights; surfaces then sample the grid. When `volumetric` is false, lighting
// in fog evaluates per pixel without lamp shadows.
struct VolumetricFog {
    // Enable froxel ray-marching when scattering is active; false falls back to per-pixel fog.
    bool volumetric;
    // Scattering strength for point and spot lights (0 = constant-color fog); fog also
    // attenuates light reaching surfaces.
    float light_scatter;
    // Enables the axis-aligned `clear_box_min..clear_box_max` interior volume.
    bool has_clear_box;
    // World-space minimum corner of the clear box (meters).
    Vector3 clear_box_min;
    // World-space maximum corner of the clear box (meters).
    Vector3 clear_box_max;
    // Density multiplier inside the clear box, in 0..1 (0 = clear air; small value = indoor haze).
    float clear_box_density;
    // Vertical decay rate above `height_base`: `exp(-height_falloff * (y - height_base))`
    // (0 = uniform with altitude).
    float height_falloff;
    // World Y altitude (meters) where ground fog has full `density`.
    float height_base;
    // Unit vector toward the sun for fog tinting; zero uses the primary directional light.
    Vector3 sun_direction;
    // Tint color when looking toward the sun (0 uses the directional light's color).
    Color sun_color;
    // Angular concentration exponent for the sun glow in fog (0 = disabled).
    float sun_sharpness;
    // Density multiplier when looking away from the sun, in 0..1 (1 = uniform all around).
    float shade_density;
    // Shadow-shaft strength in 0..1 from the primary `DirectionalLight` shadow (0 = off).
    float sun_shafts;
    // Procedural mist density variation in 0..1 (volumetric froxel pass only).
    float noise_amount;
    // Inverse noise blob size in `1 / meters` (default `0.15`).
    float noise_scale;
    // Drift velocity of procedural mist in meters per second.
    Vector3 noise_wind;
    // Froxel grid X/Y resolution as a fraction of viewport pixels (e.g. `0.25` = 1 cell per 4x4).
    float resolution;
    // Number of exponential depth slices from the camera to its far plane (default 96).
    int slices;
}

// Scene fog parameters combining distance falloff with `VolumetricFog` scattering settings.
struct Fog {
    // Volumetric scattering and froxel settings (`fog.volumetric`, `fog.light_scatter`, etc.).
    inline VolumetricFog volume;
    // Distance falloff curve (`LINEAR`, `EXP2`, or `EXP`).
    FogType type;
    // Base fog color in sRGB hex.
    Color color;
    // Start distance in meters (`LINEAR` fog only).
    float near;
    // Full-fog distance in meters (`LINEAR` fog only).
    float far;
    // Optical density coefficient (`EXP` and `EXP2` fog).
    float density;
}

// Default `VolumetricFog` settings applied by all fog constructors.
const VolumetricFog VOLUMETRIC_FOG_DEFAULTS = {
    .volumetric = true,
    .shade_density = 1,
    .noise_scale = 0.15f,
    .resolution = 0.25f,
    .slices = 96
};

// Allocates an `EXP` fog (`exp(-density * distance)`) on the heap (`mem`).
fn Fog* fog_exp(Color color, float density = 0.01) {
    Fog* f = mem::new(Fog);
    *f = { .type = EXP, .color = color, .density = density, .volume = VOLUMETRIC_FOG_DEFAULTS };
    return f;
}

// Allocates an `EXP` fog with `light_scatter` enabled for lamp beams and sun shafts.
fn Fog* volumetric_fog(Color color, float density = 0.01, float light_scatter = 1) {
    Fog* f = fog_exp(color, density);
    f.light_scatter = light_scatter;
    return f;
}

// Enables the fog clear box and sets its world-space `[min, max]` bounds.
fn void Fog.set_clear_box(&self, Vector3 min, Vector3 max) {
    self.has_clear_box = true;
    self.clear_box_min = min;
    self.clear_box_max = max;
}

// Allocates a `LINEAR` fog fading from `near` to `far` meters on the heap (`mem`).
fn Fog* fog(Color color, float near = 1, float far = 1000) {
    Fog* f = mem::new(Fog);
    *f = { .type = LINEAR, .color = color, .near = near, .far = far, .volume = VOLUMETRIC_FOG_DEFAULTS };
    return f;
}

// Allocates an `EXP2` fog (`exp(-(density * distance)^2)`) on the heap (`mem`).
fn Fog* fog_exp2(Color color, float density = 0.00025) {
    Fog* f = mem::new(Fog);
    *f = { .type = EXP2, .color = color, .density = density, .volume = VOLUMETRIC_FOG_DEFAULTS };
    return f;
}

// Root scene node holding background, fog, environment lighting, and global illumination volumes.
struct Scene {
    inline Object3D object;
    // Clear color in sRGB hex; 0 falls back to `Renderer.clear_color`.
    Color background;
    // Optional scene fog (`null` disables fog).
    Fog* fog;
    // Primary baked diffuse GI volume (`IrradianceVolume`) lighting materials inside its bounds.
    IrradianceVolume* irradiance;
    // Additional non-owning `IrradianceVolume` entries (total across `irradiance` and this slice
    // capped at `MAX_IRRADIANCE_VOLUMES`). Later volumes override earlier ones inside their box,
    // fading over half a cell outside; `replaces_ambient` is read from the first volume.
    IrradianceVolume*[] irradiance_volumes;
    // Real-time `RadianceCascades` GI volume; inside its bounds it replaces ambient/hemisphere
    // lights and `irradiance`.
    RadianceCascades* radiance;
    // Room-based `ReflectionProbes` set sampled by materials with `Material.env_probes`.
    ReflectionProbes* reflection_probes;
    // Equirectangular environment map (e.g. from `Renderer.capture_environment`) providing
    // diffuse IBL and reflections for standard materials without their own `env_map`.
    Texture* environment;
    // Scalar multiplier on `environment` lighting (defaults to 1).
    float environment_intensity;
    // Equirectangular (or flat when `background_flat`) backdrop drawn behind geometry instead
    // of `background`, also visible through transmission materials.
    Texture* background_texture;
    // Draws `background_texture` as a 2D full-viewport image with its UV `repeat`/`offset`,
    // skipping tone mapping when it is sRGB.
    bool background_flat;
    // Mip blur factor in 0..1 applied to equirectangular `background_texture`.
    float background_blurriness;
    // Scalar multiplier on `background_texture` brightness (defaults to 1).
    float background_intensity;
    // Custom material rendered on an enclosing sky sphere in place of `background` or
    // `background_texture` (typically a node material with `depth_write = false`, `fog = false`).
    Material* background_material;
    // When true, `Renderer.render` skips its automatic `update_matrix_world` pass (useful for a
    // second render of an unchanged scene within the same frame).
    bool no_matrix_world_update;
}

// Allocates an empty `Scene` on the heap (`mem`) with environment and background intensity 1.
fn Scene* scene() {
    Scene* s = mem::new(Scene);
    s.init(SCENE);
    s.environment_intensity = 1;
    s.background_intensity = 1;
    return s;
}

// ---- Mesh, InstancedMesh, and LOD

// Primitive topology used when drawing `Mesh.geometry`.
enum DrawMode {
    // Indexed triangles (default for meshes).
    TRIANGLES,
    // Disconnected line segments, one per pair of indices (`LineSegments`).
    LINE_SEGMENTS,
    // Connected polyline through consecutive indices (`Line`).
    LINE_STRIP,
    // Closed polyline connecting the last index back to the first (`LineLoop`).
    LINE_LOOP,
}

// Renderable scene node pairing a `BufferGeometry` with a `Material`, plus optional instancing,
// billboarding, GPU morph targets, and renderer-maintained tracking state.
struct Mesh {
    inline Object3D object;
    // Vertex and index data to draw.
    BufferGeometry* geometry;
    // Surface or line material applied when rendering.
    Material* material;
    // Topology mode (`TRIANGLES` for meshes, line modes for `Line` subtypes).
    DrawMode draw_mode;
    // First index drawn for triangle meshes (used for sub-mesh ranges sharing one geometry).
    int draw_start;
    // Number of indices drawn from `draw_start` (0 draws the full index buffer; wireframes draw all).
    int draw_count;
    // Camera-facing billboard mode (`NO_BILLBOARD` for regular meshes). Billboards align local
    // X/Y axes (scaled by `matrix_world` axis lengths) around `billboard_center` and cast no shadows.
    Billboard billboard;
    // For `FACING` billboards: keep constant screen size (1 scale unit = viewport height at z = 1).
    bool billboard_fixed_size;
    // Counterclockwise roll in radians around `billboard_center` for `FACING` billboards.
    float billboard_rotation;
    // Anchor point in the billboard quad's 0..1 UV space (`{ 0.5, 0.5 }` centers the quad).
    Vector2 billboard_center;
    // Per-instance transforms (12 floats per instance, row-major 3x4) for `InstancedMesh`.
    List{float} instance_data;
    // Active instance count in `instance_data`.
    int instance_count;
    // Dedicated GPU buffer for instance matrices when not packed in the GPU culling arena.
    SgBuffer instance_buf;
    // False when `instance_data` needs GPU upload.
    bool instances_uploaded;
    // Optional linear RGB per-instance tint (3 floats per instance; empty means white).
    List{float} instance_colors;
    // GPU buffer holding `instance_colors`.
    SgBuffer instance_color_buf;
    // False when `instance_colors` needs GPU upload.
    bool instance_colors_uploaded;
    // Local-space bounding sphere center covering all instances, for frustum culling.
    Vector3 instance_center;
    // Local-space bounding sphere radius covering all instances.
    float instance_radius;
    // True while instance matrices live in the shared `Renderer.gpu_culling` arena.
    bool in_arena;
    // First instance slot inside the shared GPU culling arena.
    int arena_base;
    // Allocated slot count inside the shared GPU culling arena.
    int arena_size;
    // Upload counter used to decide when static instances belong in the shared arena.
    int instance_uploads;
    // Previous frame's `matrix_world` used by MRT motion vectors (`RenderTarget.mrt`).
    Mat4 motion_prev;
    // Current frame's `matrix_world` stored for MRT motion vectors.
    Mat4 motion_cur;
    // Frame index when `motion_prev`/`motion_cur` were last rotated.
    int motion_frame;
    // Cached per-vertex displacement heights sampled from `Material.displacement_map`.
    SgBuffer displacement_buf;
    // Cache key identifying the geometry/material state that produced `displacement_buf`.
    ulong displacement_key;
    // Packed GPU morph target weights: header `vec4(targets, verts, stride, 0)` plus vec4 rows.
    List{float} morph_weights;
    // Number of active morph targets bound on this mesh.
    int morph_targets;
    // Transient GPU storage buffer holding `morph_weights`.
    SgBuffer morph_weights_buf;
    // Storage buffer view bound by vertex shaders for `morph_weights_buf`.
    SgView morph_weights_view;
    // Allocated float capacity of `morph_weights_buf`.
    int morph_weights_cap;
    // False when `morph_weights` needs GPU upload.
    bool morph_weights_uploaded;
    // Set by `walk_casters` when this caster moved or changed geometry/material recently so
    // shadow-atlas lights composite it over their static cache (`AtlasMap`).
    bool shadow_moving;
    // Set by `walk_casters` when vertices deform on the GPU every frame (sway, morphs, nodes).
    bool shadow_animated;
    // Internal pointer to the `CasterWalk` that last indexed this mesh.
    void* shadow_walk;
    // Frame counter of the last `CasterWalk` entry.
    uint shadow_walk_frame;
    // Slot index inside `shadow_walk`.
    int shadow_walk_at;
}

// Packs `target_count` morph target weights per instance (`instance_count == 0` for a plain
// mesh) into `morph_weights` as a header `vec4(targets, verts, stride, 0)` followed by
// vec4-aligned rows, marking the GPU buffer dirty.
fn void Mesh.set_morph_weights(&self, float[] weights, int target_count, int instance_count, int vertex_count) {
    if (!self.morph_weights.allocator) self.morph_weights.allocator = mem;
    self.morph_weights.clear();
    if (target_count <= 0 || vertex_count <= 0) {
        self.morph_targets = 0;
        self.morph_weights_uploaded = false;
        return;
    }
    int rows = instance_count > 0 ? instance_count : 1;
    int stride_v4 = (target_count + 3) / 4;
    int gpu_stride = instance_count > 0 ? stride_v4 : 0;
    sz total = (sz)(1 + rows * stride_v4) * 4;
    self.morph_weights.reserve(total);
    self.morph_weights.push((float)target_count);
    self.morph_weights.push((float)vertex_count);
    self.morph_weights.push((float)gpu_stride);
    self.morph_weights.push(0);
    for (int r = 0; r < rows; r++) {
        sz src = (sz)r * (sz)target_count;
        for (int t = 0; t < stride_v4 * 4; t++) {
            float w = (t < target_count && src + t < weights.len) ? weights[src + t] : 0;
            self.morph_weights.push(w);
        }
    }
    self.morph_targets = target_count;
    self.morph_weights_uploaded = false;
}

// Uploads dirty `morph_weights` to a transient GPU storage buffer (called by the renderer).
fn void Mesh.upload_morph_weights(&self) {
    if (self.morph_weights_uploaded || !self.morph_weights.len()) return;
    int n = (int)self.morph_weights.len();
    if (n > self.morph_weights_cap || !self.morph_weights_buf.id) {
        if (self.morph_weights_view.id) sg::destroy_view(self.morph_weights_view);
        if (self.morph_weights_buf.id) sg::destroy_buffer(self.morph_weights_buf);
        self.morph_weights_buf = sg::make_buffer(
            &&{
                .usage = { .storage_buffer = true, .write_transient = true },
                .size = (usz)n * float::size,
            }
        );
        self.morph_weights_view = sg::make_view(&&{ .storage_buffer = { .buffer = self.morph_weights_buf } });
        self.morph_weights_cap = n;
    }
    sg::write_buffer_transient(
        &&{
            .src.data = { .ptr = self.morph_weights.entries, .size = (usz)n * float::size },
            .dst.buffer = self.morph_weights_buf,
        }
    );
    self.morph_weights_uploaded = true;
}

// Mesh drawing `instance_count` copies of one `BufferGeometry` in a single draw call. Populate
// transforms with `set_matrix_at` or `add_instance`; GPU upload happens on first render.
struct InstancedMesh {
    inline Mesh mesh;
}

// Allocates an `InstancedMesh` on the heap (`mem`) pre-populated with `count` identity matrices.
fn InstancedMesh* instanced_mesh(BufferGeometry* geometry, Material* material, int count = 0) {
    InstancedMesh* m = mem::new(InstancedMesh);
    m.init(MESH);
    m.instance_data.allocator = mem;
    m.geometry = geometry;
    m.material = material;
    for (int i = 0; i < count; i++) m.add_instance(mat4_identity());
    return m;
}

// Writes the affine transform matrix for instance `index` (stored row-major 3x4) and marks GPU
// instance data dirty.
fn void InstancedMesh.set_matrix_at(&self, int index, Mat4 m) {
    float* d = &self.instance_data[(sz)index * 12];
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 4; col++) d[row * 4 + col] = m[col][row];
    }
    self.instances_uploaded = false;
}

// Sets per-instance sRGB hex tint at `index`, multiplied with the material color (unset
// instances default to white).
fn void InstancedMesh.set_color_at(&self, int index, Color color) {
    Vector3 c = color_to_linear(color);
    self.set_linear_color_at(index, c.x, c.y, c.z);
}

// Sets per-instance linear RGB tint at `index` (allocating the color list on first call).
fn void InstancedMesh.set_linear_color_at(&self, int index, float r, float g, float b) {
    if (index < 0 || index >= self.instance_count) return;
    if (!self.instance_colors.len()) {
        self.instance_colors.allocator = mem;
        for (int i = 0; i < self.instance_count * 3; i++) self.instance_colors.push(1);
    }
    float* c = &self.instance_colors[(sz)index * 3];
    c[0] = r;
    c[1] = g;
    c[2] = b;
    self.instance_colors_uploaded = false;
}

// Appends a new instance with transform `m` and returns its zero-based index.
fn int InstancedMesh.add_instance(&self, Mat4 m) {
    for (int i = 0; i < 12; i++) self.instance_data.push(0);
    if (self.instance_colors.len()) {
        for (int i = 0; i < 3; i++) self.instance_colors.push(1);
        self.instance_colors_uploaded = false;
    }
    self.instance_count++;
    self.set_matrix_at(self.instance_count - 1, m);
    return self.instance_count - 1;
}

// Level-of-detail node displaying the child matching the camera's distance band.
struct Lod {
    inline Object3D object;
    // Sorted threshold distances in meters corresponding one-to-one with `children`.
    List{float} distances;
    // Cross-fade width in meters over which adjacent levels dissolve using complementary
    // screen-door dithering (0 = hard swap).
    float fade;
}

// Allocates an empty `Lod` node on the heap (`mem`).
fn Lod* lod() {
    Lod* l = mem::new(Lod);
    l.init(LOD);
    l.distances.allocator = mem;
    return l;
}

// Adds a level shown from `distance` meters until the next level takes over; pass `null` to
// hide beyond `distance`. Add levels in increasing `distance` order; returns `self`.
fn Lod* Lod.add_level(&self, Object3D* object, float distance = 0) {
    if (!object) object = object3d();
    self.add(object);
    self.distances.push(distance);
    return self;
}

// Bending resistance model shared by `drie::cloth` and `drie::gpu_cloth`.
enum BendModel : char {
    // Holds each vertex's signed offset from its neighbors' midpoint along the normal to its
    // rest-shape value (`drie::cloth` default).
    SIGNED,
    // Connects each vertex to its two-step row and column neighbors by distance constraints
    // (`drie::gpu_cloth` default).
    DISTANCE,
}

// Camera-facing orientation mode for `Mesh.billboard`. Local X and Y (a quad in the XY plane,
// such as `PlaneGeometry`) are reoriented per variant.
enum Billboard : char {
    // Standard mesh orientation without billboarding.
    NO_BILLBOARD,
    // Aligns local X/Y with camera right and up (sprite parallel to the view plane).
    FACING,
    // Aligns local X with camera right while keeping the mesh's own Y axis.
    KEEP_Y,
    // Keeps the mesh's own X axis while aligning local Y with camera up.
    KEEP_X,
    // Uses world +Y as local Y and rotates around it to face the camera position.
    UPRIGHT,
}

// Allocates a triangle `Mesh` on the heap (`mem`), using a default `MeshBasicMaterial` if
// `material` is `null`.
fn Mesh* mesh(BufferGeometry* geometry, Material* material = null) {
    Mesh* m = mem::new(Mesh);
    m.init(MESH);
    m.instance_data.allocator = mem;
    m.geometry = geometry;
    m.material = material ? material : (Material*)mesh_basic_material();
    return m;
}

// Returns true if this mesh draws with a line topology (`draw_mode != TRIANGLES`).
fn bool Mesh.is_line(&self) => self.draw_mode != TRIANGLES;

// ---- Lines

// Unlit 1-pixel-wide polyline connecting `geometry` indices in order (`LINE_STRIP`). Lines
// cast no shadows, do not contribute to `RadianceCascades`, and are ignored by `Raycaster`.
struct Line {
    inline Mesh mesh;
}

// Disconnected 1-pixel-wide line segments, one per pair of indices (`LINE_SEGMENTS`).
struct LineSegments {
    inline Line line;
}

// Closed 1-pixel-wide polyline connecting the last index back to the first (`LINE_LOOP`).
struct LineLoop {
    inline Line line;
}

// Computes cumulative arc length along vertices in vertex order for `line_dashed_material`,
// storing distances in the vertex normal's X slot (lines have no normals) and flagging
// `geometry` for re-upload. For `LineSegments`, resets per segment pair.
fn Line* Line.compute_line_distances(&self) {
    BufferGeometry* g = self.geometry;
    uint n            = g.vertex_count();
    float d           = 0;
    for (uint i = 0; i < n; i++) {
        bool starts_segment = self.draw_mode == LINE_SEGMENTS && i % 2 == 0;
        if (i > 0 && !starts_segment) d += (g.get_position(i) - g.get_position(i - 1)).length();
        g.vertices[(sz)i * FLOATS_PER_VERTEX + 3] = d;
    }
    // Re-upload vertex buffer on next render.
    g.uploaded = false;
    return self;
}

fn Mesh* new_line(BufferGeometry* geometry, Material* material, DrawMode mode) @local {
    Mesh* m = mesh(geometry, material ?: (Material*)line_basic_material());
    m.draw_mode = mode;
    return m;
}

// Allocates a connected `Line` (`LINE_STRIP`) on the heap (`mem`); defaults to `line_basic_material`.
fn Line* line(BufferGeometry* geometry, Material* material = null) => (Line*)new_line(geometry, material, LINE_STRIP);

// Allocates `LineSegments` (`LINE_SEGMENTS`) on the heap (`mem`); defaults to `line_basic_material`.
fn LineSegments* line_segments(
    BufferGeometry* geometry,
    Material* material = null
) => (LineSegments*)new_line(geometry, material, LINE_SEGMENTS);

// Allocates a closed `LineLoop` (`LINE_LOOP`) on the heap (`mem`); defaults to `line_basic_material`.
fn LineLoop* line_loop(
    BufferGeometry* geometry,
    Material* material = null
) => (LineLoop*)new_line(geometry, material, LINE_LOOP);

// ---- Cameras

// Projection model used by `Camera`.
enum CameraType {
    // Perspective frustum (`PerspectiveCamera`).
    PERSPECTIVE,
    // Parallel orthographic box (`OrthographicCamera`).
    ORTHOGRAPHIC,
}

// Base scene camera holding clip planes, zoom, projection/view matrices, and TAA/motion state.
struct Camera {
    inline Object3D object;
    // Projection type (`PERSPECTIVE` or `ORTHOGRAPHIC`).
    CameraType camera_type;
    // Near clip plane distance in meters.
    float near;
    // Far clip plane distance in meters.
    float far;
    // Zoom multiplier (defaults to 1; higher values narrow the view).
    float zoom;
    // Current clip-space projection matrix built by `update_projection_matrix`.
    Mat4 projection_matrix;
    // World-to-view inverse of `matrix_world`, updated by `Renderer.render`.
    Mat4 matrix_world_inverse;
    // Previous frame's view-projection matrix for MRT motion vectors (`RenderTarget.mrt`).
    Mat4 motion_prev;
    // Current frame's view-projection matrix for MRT motion vectors.
    Mat4 motion_cur;
    // Frame index when `motion_prev`/`motion_cur` were last updated.
    int motion_frame;
    // When true, `Renderer.render` preserves `projection_matrix` verbatim instead of recomputing
    // it or matching target aspect (used by `Reflector`'s oblique clip plane).
    bool custom_projection;
    // Sub-pixel viewport shift in target pixels applied by `TAAPass`; omitted from motion vectors.
    Vector2 jitter;
}

// Perspective camera defined by vertical field of view and aspect ratio.
struct PerspectiveCamera {
    inline Camera camera;
    // Vertical field of view in degrees (default 50).
    float fov;
    // Viewport width divided by height (default 1; updated automatically with `Renderer.auto_aspect`).
    float aspect;
}

// Orthographic camera defined by view-space frustum bounds around the camera origin.
struct OrthographicCamera {
    inline Camera camera;
    // Left frustum boundary in view units.
    float left;
    // Right frustum boundary in view units.
    float right;
    // Top frustum boundary in view units.
    float top;
    // Bottom frustum boundary in view units.
    float bottom;
}

// Allocates a `PerspectiveCamera` on the heap (`mem`) and initializes `projection_matrix`.
fn PerspectiveCamera* perspective_camera(float fov = 50, float aspect = 1, float near = 0.1, float far = 2000) {
    PerspectiveCamera* c = mem::new(PerspectiveCamera);
    c.init(CAMERA);
    c.camera_type = PERSPECTIVE;
    c.fov = fov;
    c.aspect = aspect;
    c.near = near;
    c.far = far;
    c.zoom = 1;
    c.update_projection_matrix();
    return c;
}

// Allocates an `OrthographicCamera` on the heap (`mem`) and initializes `projection_matrix`.
fn OrthographicCamera* orthographic_camera(
    float left = -1,
    float right = 1,
    float top = 1,
    float bottom = -1,
    float near = 0.1,
    float far = 2000
) {
    OrthographicCamera* c = mem::new(OrthographicCamera);
    c.init(CAMERA);
    c.camera_type = ORTHOGRAPHIC;
    c.left = left;
    c.right = right;
    c.top = top;
    c.bottom = bottom;
    c.near = near;
    c.far = far;
    c.zoom = 1;
    c.update_projection_matrix();
    return c;
}

// Rebuilds `projection_matrix` from the camera's current parameters and backend depth range.
fn void Camera.update_projection_matrix(&self) {
    bool z01 = depth_zero_to_one();
    switch (self.camera_type) {
        case PERSPECTIVE:
            PerspectiveCamera* p = (PerspectiveCamera*)self;
            self.projection_matrix = mat4_perspective(p.fov, p.aspect, self.near, self.far, self.zoom, z01);
        case ORTHOGRAPHIC:
            OrthographicCamera* o = (OrthographicCamera*)self;
            float cx = (o.left + o.right) / 2;
            float cy = (o.top + o.bottom) / 2;
            float dx = (o.right - o.left) / (2 * self.zoom);
            float dy = (o.top - o.bottom) / (2 * self.zoom);
            self.projection_matrix = mat4_orthographic(cx - dx, cx + dx, cy + dy, cy - dy, self.near, self.far, z01);
    }
}

// ---- Lights

// Discriminator stored in `Light.light_type`.
enum LightType {
    // Uniform scene-wide ambient illumination (`AmbientLight`).
    AMBIENT,
    // Sky-to-ground gradient ambient illumination (`HemisphereLight`).
    HEMISPHERE,
    // Parallel rays from infinity (`DirectionalLight`).
    DIRECTIONAL,
    // Omnidirectional point, sphere, or tube emitter (`PointLight`).
    POINT,
    // Conical spotlight with optional cookie texture (`SpotLight`).
    SPOT,
    // Rectangular area emitter (`RectAreaLight`).
    RECT_AREA,
}

// Base scene light node shared by all concrete light subtypes.
struct Light {
    inline Object3D object;
    // Concrete light subtype.
    LightType light_type;
    // Emitted light color in sRGB hex.
    Color color;
    // Luminous intensity multiplier (units vary by subtype).
    float intensity;
    // 24-bit channel mask: lights materials whose `Material.light_channels` is 0 or shares a
    // bit. `AmbientLight` and `HemisphereLight` always illuminate every material.
    uint channels;
    // Locks the position/direction seen by `RadianceCascades` to `gi_position` and
    // `gi_direction` while direct lighting and shadows follow the live node (prevents probe
    // shimmer on swaying lamps).
    bool gi_fixed;
    // Fixed world position used by `RadianceCascades` when `gi_fixed` is true.
    Vector3 gi_position;
    // Fixed world direction used by `RadianceCascades` for spot lights when `gi_fixed` is true.
    Vector3 gi_direction;
}

// Uniform omnidirectional ambient light without shadows or directionality.
struct AmbientLight {
    inline Light light;
}

// Two-color ambient light interpolating from `color` (+Y sky) to `ground_color` (-Y ground).
struct HemisphereLight {
    inline Light light;
    // Lower-hemisphere ambient color in sRGB hex.
    Color ground_color;
}

// Parallel directional light shining from `position` toward `target`'s world position.
struct DirectionalLight {
    inline Light light;
    // Aim target whose world position defines the light direction (defaults to origin).
    Object3D* target;
    // Shadow settings (enabled by `cast_shadow`). The primary shadowed directional light gets a
    // dedicated map (or 2x2 cascaded tiles of `map_size` when `cascades > 1`, covering up to
    // `max_distance`); additional shadowed directional lights get one orthographic block each in
    // the shadow atlas (`Renderer.shadow_atlas_size`). `softness` and `focus` are unused.
    LightShadow shadow;
    // Half-width and half-height in meters of the orthographic shadow box when `cascades == 1`.
    float shadow_extent;
    // When `cascades == 1`, centers the shadow box ahead of the camera instead of `target`.
    bool shadow_follow;
}

// Omnidirectional light supporting zero-radius points, spherical bulbs (`radius`), and linear
// tube lights (`length` along local X).
struct PointLight {
    inline Light light;
    // Maximum reach in meters where light cuts off to zero (0 = infinite range).
    float distance;
    // Distance falloff exponent (default 2 for physically inverse-square decay).
    float decay;
    // Spherical emitter radius in meters (0 = point). Softens near falloff and produces
    // disk-shaped specular highlights (Karis representative-point area light).
    float radius;
    // Tube length in meters centered on `position` along local X (world rotation, ignoring
    // scale; 0 = sphere of `radius`). Uses inverse-square falloff and tube-shaped highlights;
    // shadows cast from the tube center.
    float length;
    // Cube-face shadow settings (enabled by `cast_shadow`): allocates six blocks in the shadow
    // atlas (`Renderer.shadow_atlas_size`), dynamically sized from 128 px up to `shadow.map_size`
    // by screen importance, covering `shadow.near` to `shadow.far` (0 defaults to `distance` or
    // 50 m). With `shadow.auto_update` off, renders once and on `shadow.needs_update`.
    LightShadow shadow;
}

// Shadow-map, PCF/VSM filter, cascade, and contact/ray-traced shadow settings for a light.
struct LightShadow {
    // Target shadow map resolution in pixels along one side.
    int map_size;
    // World-space offset along the surface normal per meter of distance to the light (PCF).
    float normal_bias;
    // Shadow edge blur radius in shadow-map texels.
    float radius;
    // Depth bias for `Renderer.shadow_type == VSM` (negative lights more readily; default 0).
    float bias;
    // Filter tap count across `2 * radius` texels for VSM blur (default 8).
    int blur_samples;
    // Shadow darkness in 0..1: 1 removes all direct light in shadow, 0.5 removes half (default 1).
    float intensity;
    // Emitter diameter in meters for blocker-distance penumbra widening (`radius` sets minimum
    // blur; 0 disables variable penumbra).
    float softness;
    // Shadow frustum near clip distance in meters.
    float near;
    // Screen-space contact shadow ray length in meters through the depth buffer in deferred mode
    // (`Renderer.deferred`), for directional, point, and spot lights (0 = off; 0.1..0.5 typical).
    // Combines with shadow maps and works regardless of `cast_shadow`.
    float contact_length;
    // For `RectAreaLight` only: enable hardware ray-traced area shadows when
    // `Renderer.ray_traced_shadows` and `Renderer.traces_rays` are active (default true); falls
    // back to shadow maps otherwise.
    bool ray_traced;
    // Shadow frustum far clip distance in meters (0 defaults to `distance` or 50 m).
    float far;
    // Fraction of a `SpotLight` cone covered by its shadow map in 0..1; narrower values spend
    // shadow texels near the cone center and leave the outer ring unshadowed.
    float focus;
    // Redraw shadow maps every frame while true; when false, draw once and on `needs_update`.
    bool auto_update;
    // Requests a shadow map redraw when `auto_update` is false.
    bool needs_update;
    // Excludes `InstancedMesh` nodes from this light's shadow map.
    bool skip_instanced;
    // For `DirectionalLight`: number of view-frustum shadow cascades in 1..4 (1 uses a single
    // box sized by `DirectionalLight.shadow_extent`).
    int cascades;
    // Cascade split distribution in 0..1: 0 = linear, 1 = logarithmic (tighter near camera).
    float split_lambda;
    // Maximum camera distance in meters covered by directional cascades (capped at `camera.far`).
    float max_distance;
    // Overlap blend width between adjacent cascades as a fraction of slice depth, in 0..1.
    float cascade_blend;
}

// Conical spotlight aimed from `position` toward `target`'s world position, with optional
// soft edge penumbra, cookie texture (`map`), and shadow map.
struct SpotLight {
    inline Light light;
    // Aim target whose world position defines the cone axis (defaults to origin).
    Object3D* target;
    // Maximum reach in meters (0 = infinite range).
    float distance;
    // Half-angle of the light cone in radians (up to `PI / 2`).
    float angle;
    // Outer cone fraction in 0..1 that smoothly fades to zero at the edge.
    float penumbra;
    // Distance falloff exponent (default 2).
    float decay;
    // Emitter radius in meters softening near-field falloff.
    float radius;
    // Specular highlight sphere radius in meters (0 uses `radius`), allowing a small bright bulb
    // inside a larger diffuser to keep crisp highlights.
    float specular_radius;
    // Multiplier on `VolumetricFog.light_scatter` glow (1 = normal, 0 = excluded from fog).
    float fog_glow;
    // Shadow map settings (enabled by `cast_shadow`).
    LightShadow shadow;
    // Projected gobo/cookie texture multiplied with light color across the cone's circumscribed
    // square (`+U` right, `+V` up, rotated by `map_rotation`). All spot lights in a frame share
    // the first encountered `map` texture; pack multiple cookies into one atlas using `map_rect`.
    // Supported on all lit materials except `Reflector`.
    Texture* map;
    // Sub-rectangle `{ x, y, width, height }` in 0..1 UV space within `map` (all zeros = full map).
    float[4] map_rect;
    // Counterclockwise roll of `map` around the cone axis in radians (seen from the light).
    float map_rotation;
}

// Rectangular area light (`width` by `height` meters) emitting from its local `-Z` face (aim
// with `look_at`). Intensity is in nits (`power() = intensity * width * height * PI` lumens).
// Uses linearly transformed cosines on `MeshStandardMaterial` (including clearcoat); world
// rotation orients the panel while scale is ignored. Capped at `Renderer.max_rect_area_lights`.
struct RectAreaLight {
    inline Light light;
    // Panel width in meters along local X.
    float width;
    // Panel height in meters along local Y.
    float height;
    // Shadow settings (enabled by `cast_shadow`): hardware ray-traced when `shadow.ray_traced`
    // and `Renderer.traces_rays` are active, otherwise six cube faces in the shadow atlas
    // (`Renderer.shadow_atlas_size`) from `shadow.near` to `shadow.far` (0 defaults to 50 m).
    LightShadow shadow;
}

// Initializes base `Object3D` and light type, color, and intensity fields.
fn void Light.init_light(&self, LightType t, Color color, float intensity) {
    self.init(LIGHT);
    self.light_type = t;
    self.color = color;
    self.intensity = intensity;
}

// Allocates an `AmbientLight` on the heap (`mem`).
fn AmbientLight* ambient_light(Color color = 0xffffff, float intensity = 1) {
    AmbientLight* l = mem::new(AmbientLight);
    l.init_light(AMBIENT, color, intensity);
    return l;
}

// Allocates a `HemisphereLight` positioned at `(0, 1, 0)` on the heap (`mem`).
fn HemisphereLight* hemisphere_light(Color sky_color = 0xffffff, Color ground_color = 0xffffff, float intensity = 1) {
    HemisphereLight* l = mem::new(HemisphereLight);
    l.init_light(HEMISPHERE, sky_color, intensity);
    l.ground_color = ground_color;
    l.position = { 0, 1, 0 };
    return l;
}

// Allocates a `DirectionalLight` at `(0, 1, 0)` aimed at a new origin `target`, with 4-cascade
// 2048 px shadow defaults.
fn DirectionalLight* directional_light(Color color = 0xffffff, float intensity = 1) {
    DirectionalLight* l = mem::new(DirectionalLight);
    l.init_light(DIRECTIONAL, color, intensity);
    l.position = { 0, 1, 0 };
    l.target = object3d();
    l.shadow = { .map_size = 2048, .normal_bias = 0.03f, .radius = 2, .blur_samples = 8, .intensity = 1, .near = 0.5f, .far = 300, .auto_update = true,
        .cascades = 4, .split_lambda = 0.7f, .max_distance = 150, .cascade_blend = 0.1f };
    l.shadow_extent = 30;
    return l;
}

// Allocates a `PointLight` on the heap (`mem`) with 512 px cube-face shadow defaults.
fn PointLight* point_light(Color color = 0xffffff, float intensity = 1, float distance = 0, float decay = 2) {
    PointLight* l = mem::new(PointLight);
    l.init_light(POINT, color, intensity);
    l.distance = distance;
    l.decay = decay;
    l.shadow = {
        .map_size = 512,
        .normal_bias = 0.006f,
        .radius = 2,
        .blur_samples = 8,
        .intensity = 1,
        .near = 0.1f,
        .auto_update = true,
        .needs_update = true
    };
    return l;
}

// Allocates a `RectAreaLight` on the heap (`mem`) with ray-traced shadows enabled by default.
fn RectAreaLight* rect_area_light(Color color = 0xffffff, float intensity = 1, float width = 10, float height = 10) {
    RectAreaLight* l = mem::new(RectAreaLight);
    l.init_light(RECT_AREA, color, intensity);
    l.width = width;
    l.height = height;
    l.shadow = {
        .map_size = 512,
        .normal_bias = 0.006f,
        .radius = 2,
        .blur_samples = 8,
        .intensity = 1,
        .near = 0.1f,
        .auto_update = true,
        .needs_update = true,
        .ray_traced = true,
    };
    return l;
}

// Returns total luminous flux in lumens (`intensity * width * height * PI`).
fn float RectAreaLight.power(&self) => self.intensity * self.width * self.height * (float)PI;

// Sets `intensity` from luminous flux `power` in lumens for the current `width` and `height`.
fn void RectAreaLight.set_power(&self, float power) {
    self.intensity = power / (self.width * self.height * (float)PI);
}

// Allocates a `SpotLight` at `(0, 1, 0)` aimed at a new origin `target` (1024 px shadow).
fn SpotLight* spot_light(
    Color color = 0xffffff,
    float intensity = 1,
    float distance = 0,
    float angle = PI / 3,
    float penumbra = 0,
    float decay = 2
) {
    SpotLight* l = mem::new(SpotLight);
    l.init_light(SPOT, color, intensity);
    l.position = { 0, 1, 0 };
    l.target = object3d();
    l.distance = distance;
    l.angle = angle;
    l.penumbra = penumbra;
    l.decay = decay;
    l.fog_glow = 1;
    l.shadow = {
        .map_size = 1024,
        .normal_bias = 0.006f,
        .radius = 2,
        .blur_samples = 8,
        .intensity = 1,
        .near = 0.05f,
        .focus = 1,
        .auto_update = true,
        .needs_update = true
    };
    return l;
}