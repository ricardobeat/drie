// Scene graph: Object3D, Group, Scene, Mesh, cameras, lights, fog.
module three;
import std::collections::list, libc;
import sokol;

enum ObjectKind
{
    OBJECT,
    GROUP,
    SCENE,
    MESH,
    LOD,
    CAMERA,
    LIGHT,
}

struct Object3D
{
    ObjectKind kind;
    String name;
    Vector3 position;
    Euler rotation;
    Vector3 scale;
    Vector3 up;
    bool visible;
    // Meshes: skip drawing when the bounding sphere is outside the camera's view.
    bool frustum_culled;
    // Meshes: render into shadow maps / get darkened by them.
    bool cast_shadow;
    bool receive_shadow;
    // Like three.js renderOrder: among see-through meshes, higher ones are drawn later
    // whatever their distance (a film of dirt over the glass it lies on).
    int render_order;
    Object3D* parent;
    List{Object3D*} children;
    Mat4 matrix;
    Mat4 matrix_world;
    // What update_matrix_world last built `matrix` from: an object that has not moved is
    // not computed again (most of a scene stands still).
    Vector3 placed_position;
    Euler placed_rotation;
    Vector3 placed_scale;
    bool placed;
    // matrix_world was changed outside update_world, which has yet to tell the children
    bool world_dirty;
    // Free slot for your own data, like three's object.userData.
    void* user_data;
}

fn void Object3D.init(&self, ObjectKind kind)
{
    self.kind = kind;
    self.scale = { 1, 1, 1 };
    self.up = { 0, 1, 0 };
    self.visible = true;
    self.frustum_culled = true;
    self.matrix = mat4_identity();
    self.matrix_world = mat4_identity();
    // lists live on the heap: by default they would use the temp allocator and be freed
    // with the caller's @pool
    self.children.allocator = mem;
}

// Adds a child, removing it from its previous parent first. Returns self for chaining.
fn Object3D* Object3D.add(&self, Object3D* child)
{
    if (child == self) return self;
    if (child.parent) child.parent.remove(child);
    child.parent = self;
    child.placed = false;
    self.children.push(child);
    return self;
}

fn Object3D* Object3D.remove(&self, Object3D* child)
{
    foreach (i, c : self.children)
    {
        if (c == child)
        {
            self.children.remove_at(i);
            child.parent = null;
            break;
        }
    }
    return self;
}

fn void Object3D.clear(&self)
{
    foreach (c : self.children) c.parent = null;
    self.children.clear();
}

fn Object3D* Object3D.get_object_by_name(&self, String name)
{
    if (self.name == name) return self;
    foreach (c : self.children)
    {
        Object3D* found = c.get_object_by_name(name);
        if (found) return found;
    }
    return null;
}

// Like three's object.clone(): a copy of this object and its descendants that shares their
// geometries and materials, without a parent. For groups and meshes (a loaded model's
// nodes, say: scene.add(model.get("chair").clone())); lights, cameras, LODs and instanced
// meshes come out as empty groups.
fn Object3D* Object3D.clone(&self)
{
    Object3D* copy;
    if (self.kind == MESH && ((Mesh*)self).instance_data.len() == 0)
    {
        Mesh* from = (Mesh*)self;
        copy = mesh(from.geometry, from.material);
    }
    else
    {
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

alias TraverseCallback = fn void(Object3D*);

fn void Object3D.traverse(&self, TraverseCallback cb)
{
    cb(self);
    foreach (c : self.children) c.traverse(cb);
}

fn void Object3D.update_matrix(&self)
{
    self.matrix = mat4_compose(self.position, self.rotation, self.scale);
}

// Brings matrix_world up to date for this object and all its descendants. Only what moved
// since the last call (or hangs under something that did) is computed. Set position,
// rotation and scale, not `matrix`: it is rebuilt from them.
fn void Object3D.update_matrix_world(&self) => self.update_world(true);

fn void Object3D.update_world(&self, bool parent_moved)
{
    bool moved = !self.placed
        || self.position != self.placed_position
        || self.rotation != self.placed_rotation
        || self.scale != self.placed_scale;
    if (moved)
    {
        self.update_matrix();
        self.placed_position = self.position;
        self.placed_rotation = self.rotation;
        self.placed_scale = self.scale;
        self.placed = true;
    }
    if (moved || parent_moved)
    {
        Mat4 world = self.parent ? self.parent.matrix_world.mul(self.matrix) : self.matrix;
        // (the first object of the walk can't tell whether its parent moved, so it is
        // always computed: its descendants only if that changed anything)
        moved = libc::memcmp(&world, &self.matrix_world, Mat4::size) != 0;
        self.matrix_world = world;
    }
    // moved already, by update_world_matrix_from_parents (a bone its skeleton placed)
    moved = moved || self.world_dirty;
    self.world_dirty = false;
    foreach (c : self.children) c.update_world(moved);
}

// Recomputes matrix_world for this object using its ancestors' current transforms.
fn void Object3D.update_world_matrix_from_parents(&self)
{
    if (self.parent) self.parent.update_world_matrix_from_parents();
    self.update_matrix();
    Mat4 world = self.parent ? self.parent.matrix_world.mul(self.matrix) : self.matrix;
    // update_world must still carry the move down to the children (a mesh hung on a bone)
    if (libc::memcmp(&world, &self.matrix_world, Mat4::size) != 0) self.world_dirty = true;
    self.matrix_world = world;
}

fn Vector3 Object3D.get_world_position(&self)
{
    self.update_world_matrix_from_parents();
    return self.matrix_world.get_position();
}

// Rotates the object to face a world-space point. Cameras and lights point -Z at it, like three.
fn void Object3D.look_at(&self, float x, float y, float z) => self.look_at_vec({ x, y, z });

fn void Object3D.look_at_vec(&self, Vector3 target)
{
    Vector3 eye = self.get_world_position();
    bool neg_z = self.kind == CAMERA || self.kind == LIGHT;
    Mat4 rot = mat4_look_at(eye, target, self.up, neg_z);
    if (self.parent)
    {
        // remove the parent's rotation (assumes uniform scale)
        Mat4 pm = self.parent.matrix_world;
        pm[3] = { 0, 0, 0, 1 };
        Vector3 s = { ((Vector3){ pm[0][0], pm[0][1], pm[0][2] }).length(), 0, 0 };
        if (s.x > 0)
        {
            for (int c = 0; c < 3; c++) for (int r = 0; r < 3; r++) pm[c][r] /= s.x;
        }
        rot = pm.transpose().mul(rot);
    }
    self.rotation = euler_from_rotation(rot);
}

fn void Object3D.rotate_x(&self, float angle) => self.rotation.x += angle;
fn void Object3D.rotate_y(&self, float angle) => self.rotation.y += angle;
fn void Object3D.rotate_z(&self, float angle) => self.rotation.z += angle;

// Moves along the object's local axis (like three's translateX/Y/Z).
fn void Object3D.translate_on_axis(&self, Vector3 axis, float distance)
{
    Mat4 r = mat4_compose({ 0, 0, 0 }, self.rotation, { 1, 1, 1 });
    self.position += r.transform_point(axis) * distance;
}
fn void Object3D.translate_x(&self, float d) => self.translate_on_axis({ 1, 0, 0 }, d);
fn void Object3D.translate_y(&self, float d) => self.translate_on_axis({ 0, 1, 0 }, d);
fn void Object3D.translate_z(&self, float d) => self.translate_on_axis({ 0, 0, 1 }, d);

fn Object3D* object3d()
{
    Object3D* o = mem::new(Object3D);
    o.init(OBJECT);
    return o;
}

// ----------------------------------------------------------------------------
// Group / Scene / Fog
// ----------------------------------------------------------------------------

struct Group
{
    inline Object3D object;
}

fn Group* group()
{
    Group* g = mem::new(Group);
    g.init(GROUP);
    return g;
}

enum FogType
{
    LINEAR,
    EXP2,
    // Extension: plain exponential falloff, exp(-density * distance), like real haze.
    EXP,
}

struct Fog
{
    FogType type;
    Color color;
    float near;
    float far;
    float density;
    // Extensions (not in three.js):
    // How much point and spot lights light up the fog around them (0 = fog has a constant color).
    // Light is also dimmed by the fog on its way to surfaces.
    float light_scatter;
    // A box without fog (e.g. the inside of a building). Enabled when has_clear_box is set.
    bool has_clear_box;
    Vector3 clear_box_min;
    Vector3 clear_box_max;
    // Fog that lies on the ground: its density falls off by exp(-height_falloff * (y -
    // height_base)) above height_base, and is `density` at it. 0 = the same at every height.
    float height_falloff;
    float height_base;
    // A sun in the fog: looking towards sun_direction (a unit vector towards the sun) the fog
    // takes sun_color instead of its own, the more sharply the higher sun_sharpness (0 = no
    // sun). Looking away from the sun's side there is only shade_density of the fog (1 = as
    // much all round).
    Vector3 sun_direction;
    Color sun_color;
    float sun_sharpness;
    float shade_density;
}

// Extension: exponential fog, exp(-density * distance).
fn Fog* fog_exp(Color color, float density = 0.01)
{
    Fog* f = mem::new(Fog);
    *f = { .type = EXP, .color = color, .density = density, .shade_density = 1 };
    return f;
}

fn void Fog.set_clear_box(&self, Vector3 min, Vector3 max)
{
    self.has_clear_box = true;
    self.clear_box_min = min;
    self.clear_box_max = max;
}

// Linear fog, like `new THREE.Fog(color, near, far)`.
fn Fog* fog(Color color, float near = 1, float far = 1000)
{
    Fog* f = mem::new(Fog);
    *f = { .type = LINEAR, .color = color, .near = near, .far = far };
    return f;
}

// Exponential squared fog, like `new THREE.FogExp2(color, density)`.
fn Fog* fog_exp2(Color color, float density = 0.00025)
{
    Fog* f = mem::new(Fog);
    *f = { .type = EXP2, .color = color, .density = density };
    return f;
}

struct Scene
{
    inline Object3D object;
    // Clear color for this scene. 0 (black) falls back to the renderer's clear color.
    Color background;
    Fog* fog;
    // Extension: bounced light for the materials inside its box (see irradiance.c3).
    IrradianceVolume* irradiance;
    // Like three's matrixWorldAutoUpdate: render() refreshes every object's world matrix first.
    // Switch it off for a second render of an unchanged scene in the same frame (update
    // whatever did move yourself).
    bool no_matrix_world_update;
}

fn Scene* scene()
{
    Scene* s = mem::new(Scene);
    s.init(SCENE);
    return s;
}

// ----------------------------------------------------------------------------
// Mesh
// ----------------------------------------------------------------------------

struct Mesh
{
    inline Object3D object;
    BufferGeometry* geometry;
    Material* material;
    // Instancing (see InstancedMesh): 12 floats per instance, the rows of a 3x4 matrix.
    List{float} instance_data;
    int instance_count;
    SgBuffer instance_buf;
    bool instances_uploaded;
    // Bounding sphere of all instances in local space, for culling.
    Vector3 instance_center;
    float instance_radius;
}

// Draws the same geometry many times with one draw call, like THREE.InstancedMesh.
// Set the transforms once with set_matrix_at; they are uploaded on first render.
struct InstancedMesh
{
    inline Mesh mesh;
}

fn InstancedMesh* instanced_mesh(BufferGeometry* geometry, Material* material, int count = 0)
{
    InstancedMesh* m = mem::new(InstancedMesh);
    m.init(MESH);
    m.instance_data.allocator = mem;
    m.geometry = geometry;
    m.material = material;
    for (int i = 0; i < count; i++) m.add_instance(mat4_identity());
    return m;
}

fn void InstancedMesh.set_matrix_at(&self, int index, Mat4 m)
{
    float* d = &self.instance_data[(sz)index * 12];
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 4; col++) d[row * 4 + col] = m[col][row];
    }
    self.instances_uploaded = false;
}

// Appends an instance and returns its index.
fn int InstancedMesh.add_instance(&self, Mat4 m)
{
    for (int i = 0; i < 12; i++) self.instance_data.push(0);
    self.instance_count++;
    self.set_matrix_at(self.instance_count - 1, m);
    return self.instance_count - 1;
}

// Level of detail: shows the one child whose distance range contains the camera, like THREE.LOD.
struct Lod
{
    inline Object3D object;
    List{float} distances;
}

fn Lod* lod()
{
    Lod* l = mem::new(Lod);
    l.init(LOD);
    l.distances.allocator = mem;
    return l;
}

// `object` is shown from `distance` on, until the next level takes over.
// Pass null to show nothing from that distance on. Add levels in increasing distance.
fn Lod* Lod.add_level(&self, Object3D* object, float distance = 0)
{
    if (!object) object = object3d();
    self.add(object);
    self.distances.push(distance);
    return self;
}

fn Mesh* mesh(BufferGeometry* geometry, Material* material = null)
{
    Mesh* m = mem::new(Mesh);
    m.init(MESH);
    m.instance_data.allocator = mem;
    m.geometry = geometry;
    m.material = material ? material : (Material*)mesh_basic_material();
    return m;
}

// ----------------------------------------------------------------------------
// Cameras
// ----------------------------------------------------------------------------

enum CameraType
{
    PERSPECTIVE,
    ORTHOGRAPHIC,
}

struct Camera
{
    inline Object3D object;
    CameraType camera_type;
    float near;
    float far;
    float zoom;
    Mat4 projection_matrix;
    Mat4 matrix_world_inverse;
}

struct PerspectiveCamera
{
    inline Camera camera;
    float fov;
    float aspect;
}

struct OrthographicCamera
{
    inline Camera camera;
    float left;
    float right;
    float top;
    float bottom;
}

fn PerspectiveCamera* perspective_camera(float fov = 50, float aspect = 1, float near = 0.1, float far = 2000)
{
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

fn OrthographicCamera* orthographic_camera(float left = -1, float right = 1, float top = 1, float bottom = -1,
                                           float near = 0.1, float far = 2000)
{
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

fn void Camera.update_projection_matrix(&self)
{
    bool z01 = depth_zero_to_one();
    switch (self.camera_type)
    {
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

// ----------------------------------------------------------------------------
// Lights
// ----------------------------------------------------------------------------

enum LightType
{
    AMBIENT,
    HEMISPHERE,
    DIRECTIONAL,
    POINT,
    SPOT,
}

struct Light
{
    inline Object3D object;
    LightType light_type;
    Color color;
    float intensity;
}

struct AmbientLight
{
    inline Light light;
}

struct HemisphereLight
{
    inline Light light;
    Color ground_color;
}

struct DirectionalLight
{
    inline Light light;
    // The light shines from its position towards target's world position.
    Object3D* target;
}

struct PointLight
{
    inline Light light;
    float distance;
    float decay;
    // Extension: size of the light source in meters. Softens the falloff close to the light
    // and widens its highlights.
    float radius;
}

// Shadow settings of a light, like three's light.shadow.
struct LightShadow
{
    int map_size;
    // World-space offset along the surface normal per meter of distance to the light.
    float normal_bias;
    // Blur radius of the shadow edge in shadow map texels.
    float radius;
    // Size of the light source in meters, for shadows whose edge is sharp where they touch
    // what casts them and widens with the distance behind it (`radius` is then the least
    // blur). 0 = the same edge everywhere, which is cheaper.
    float softness;
    float near;
    // Distance up to which shadows are cast (0 = the light's distance, or 50).
    float far;
    // Part of the light's cone covered by the shadow map, 0..1, like three's SpotLightShadow.focus.
    // A wide light that travels with the camera only needs shadows where the camera looks:
    // a smaller focus spends the map's resolution there. Outside of it nothing is shadowed.
    float focus;
    // Like three's shadow.autoUpdate and needsUpdate: with auto_update off the map is drawn
    // once, and again whenever needs_update is set. For lamps that stay where they are among
    // things that don't move. Up to four such lights share one atlas, next to the one light
    // whose map follows every frame.
    bool auto_update;
    bool needs_update;
    // Leave instanced meshes (scattered plants, usually by the thousand) out of this
    // light's map.
    bool skip_instanced;
}

struct SpotLight
{
    inline Light light;
    // The light points towards target's world position.
    Object3D* target;
    float distance;
    // Maximum cone angle in radians (up to PI/2).
    float angle;
    // Fraction of the cone that fades out at the edge, 0..1.
    float penumbra;
    float decay;
    // Extension: size of the light source in meters. Softens the falloff close to the light.
    float radius;
    // Extension: size of the bright part seen in reflections, in meters (0 = `radius`). A lamp
    // with a small bulb behind a big diffuser keeps sharp highlights on glossy surfaces.
    float specular_radius;
    // Extension: how much the light shows in fog with `light_scatter` (1 = as it is, 0 = not
    // at all, for a lamp indoors whose glow would otherwise show through the walls).
    float fog_glow;
    // Set `cast_shadow` on the light to enable.
    LightShadow shadow;
}

fn void Light.init_light(&self, LightType t, Color color, float intensity)
{
    self.init(LIGHT);
    self.light_type = t;
    self.color = color;
    self.intensity = intensity;
}

fn AmbientLight* ambient_light(Color color = 0xffffff, float intensity = 1)
{
    AmbientLight* l = mem::new(AmbientLight);
    l.init_light(AMBIENT, color, intensity);
    return l;
}

fn HemisphereLight* hemisphere_light(Color sky_color = 0xffffff, Color ground_color = 0xffffff, float intensity = 1)
{
    HemisphereLight* l = mem::new(HemisphereLight);
    l.init_light(HEMISPHERE, sky_color, intensity);
    l.ground_color = ground_color;
    l.position = { 0, 1, 0 };
    return l;
}

fn DirectionalLight* directional_light(Color color = 0xffffff, float intensity = 1)
{
    DirectionalLight* l = mem::new(DirectionalLight);
    l.init_light(DIRECTIONAL, color, intensity);
    l.position = { 0, 1, 0 };
    l.target = object3d();
    return l;
}

fn PointLight* point_light(Color color = 0xffffff, float intensity = 1, float distance = 0, float decay = 2)
{
    PointLight* l = mem::new(PointLight);
    l.init_light(POINT, color, intensity);
    l.distance = distance;
    l.decay = decay;
    return l;
}

fn SpotLight* spot_light(Color color = 0xffffff, float intensity = 1, float distance = 0, float angle = PI / 3,
                         float penumbra = 0, float decay = 2)
{
    SpotLight* l = mem::new(SpotLight);
    l.init_light(SPOT, color, intensity);
    l.position = { 0, 1, 0 };
    l.target = object3d();
    l.distance = distance;
    l.angle = angle;
    l.penumbra = penumbra;
    l.decay = decay;
    l.fog_glow = 1;
    l.shadow = { .map_size = 1024, .normal_bias = 0.006f, .radius = 2, .near = 0.05f, .focus = 1,
                 .auto_update = true, .needs_update = true };
    return l;
}
