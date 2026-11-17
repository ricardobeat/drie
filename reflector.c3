// Planar reflections, like three.js's Reflector, for a flat surface that keeps its own
// material: each frame the scene is rendered again from the camera mirrored in the plane, into a
// target of its own, with what is behind the plane clipped away by an oblique near plane (Lengyel).
// The materials that name the Reflector (Material.reflector) take their mirror image from that
// picture, moved about by their normal maps and blurred by their roughness, in place of their
// environment map's. Unlike screen-space reflections it shows what the camera can't see (the
// underside of a thing over the floor); it costs a second render of the scene.
//
//   Reflector* mirror = three::reflector({ 0, 1, 0 }, FLOOR_Y);
//   floor_material.reflector = mirror;
//   // each frame, with the renderer's target set as for the main render:
//   renderer.render_reflector(mirror, scene, camera);
//   renderer.render(scene, camera);
module three;
import std::math, std::collections::list, sokol;

struct Reflector
{
    // the plane: the points p with dot(normal, p) == constant; the side the normal points to is
    // the one mirrored
    Vector3 normal;
    float constant;
    // the size of its picture, as a part of the main render's
    float resolution;
    // how far along the ray (m) the normal map's turn of it is taken to move the picture: about
    // how far the things it mirrors are
    float distortion;
    // how far the picture is blurred for a roughness of 1 (a part of its height)
    float blur;
    // how strong the reflection is (1: as the material's Fresnel says)
    float strength;
    // what is this far behind the plane is still drawn (m): a thing standing on it keeps its foot
    float clip_bias;
    // draws with the shadow maps the last render left (as they were the frame before) instead of
    // drawing them again from its own viewpoint: much cheaper. On by default
    bool reuse_shadows;
    RenderTarget* target;
    PerspectiveCamera* camera;
    // world -> its picture's clip space, as it was rendered
    Mat4 view_proj;
    // rendered this frame (false when the camera was behind the plane)
    bool rendered;
}

fn Reflector* reflector(Vector3 normal = { 0, 1, 0 }, float constant = 0, float resolution = 0.5)
{
    Reflector* r = mem::new(Reflector);
    *r = { .normal = normal.normalize(), .constant = constant, .resolution = resolution, .distortion = 0.6,
           .blur = 0.05, .strength = 1, .clip_bias = 0.003,
           .reuse_shadows = true };
    r.camera = perspective_camera();
    r.camera.custom_projection = true;
    r.camera.matrix_auto_update = false;
    return r;
}

// The mirror image across the plane: p -> p - 2 (n.p - c) n.
fn Mat4 Reflector.mirror(&self)
{
    Vector3 n = self.normal;
    float c = self.constant;
    Mat4 m = mat4_identity();
    m[0][0] = 1 - 2 * n.x * n.x;
    m[0][1] = -2 * n.y * n.x;
    m[0][2] = -2 * n.z * n.x;
    m[1][0] = -2 * n.x * n.y;
    m[1][1] = 1 - 2 * n.y * n.y;
    m[1][2] = -2 * n.z * n.y;
    m[2][0] = -2 * n.x * n.z;
    m[2][1] = -2 * n.y * n.z;
    m[2][2] = 1 - 2 * n.z * n.z;
    m[3][0] = 2 * c * n.x;
    m[3][1] = 2 * c * n.y;
    m[3][2] = 2 * c * n.z;
    return m;
}

// Renders what `reflector`'s plane mirrors, as `camera` sees it. Call it before render() for
// the frame, with the render target set as it will be for that (its picture is a part of that
// size). Leaves the target and the reflection source as they were.
fn void Renderer.render_reflector(&self, Reflector* reflector, Scene* scene, PerspectiveCamera* camera)
{
    Reflector* r = reflector;
    r.rendered = false;
    if (!camera.parent) camera.update_matrix_world();
    Vector3 eye = camera.matrix_world.get_position();
    if (r.normal.dot(eye) - r.constant <= 0) return;

    RenderTarget* main_rt = self.render_target;
    int w = math::max((int)((main_rt ? main_rt.width : sapp::width()) * r.resolution), 1);
    int h = math::max((int)((main_rt ? main_rt.height : sapp::height()) * r.resolution), 1);
    if (!r.target) r.target = render_target(w, h, 1, RGBA16F);
    r.target.set_size(w, h, 1);

    // the camera mirrored, and turned left for right: two mirror images make a turn, so its
    // triangles keep their winding (the picture comes out the other way round, which the
    // shader's projection undoes)
    Mat4 flip = mat4_identity();
    flip[0][0] = -1;
    PerspectiveCamera* mc = r.camera;
    mc.matrix = r.mirror().mul(camera.matrix_world).mul(flip);
    mc.update_matrix_world();
    mc.matrix_world_inverse = mc.matrix_world.invert();
    mc.fov = camera.fov;
    mc.aspect = camera.aspect;
    mc.near = camera.near;
    mc.far = camera.far;
    mc.zoom = camera.zoom;
    mc.update_projection_matrix();

    // the near plane laid onto the mirror's plane (Lengyel's oblique projection): nothing
    // behind it is drawn. The plane in the mirrored camera's view space: (M^T) (n, -c).
    Mat4 mw = mc.matrix_world;
    float[<4>] pw = { r.normal.x, r.normal.y, r.normal.z, -r.constant + r.clip_bias };
    float[<4>] cv;
    for (int i = 0; i < 4; i++) cv[i] = mw[i][0] * pw[0] + mw[i][1] * pw[1] + mw[i][2] * pw[2] + mw[i][3] * pw[3];
    Mat4 p = mc.projection_matrix;
    Mat4 pinv = p.invert();
    float[<4>] corner = { cv[0] > 0 ? 1 : (cv[0] < 0 ? -1 : 0), cv[1] > 0 ? 1 : (cv[1] < 0 ? -1 : 0), 1, 1 };
    float[<4>] q;
    for (int i = 0; i < 4; i++)
    {
        q[i] = pinv[0][i] * corner[0] + pinv[1][i] * corner[1] + pinv[2][i] * corner[2] + pinv[3][i] * corner[3];
    }
    float dq = cv[0] * q[0] + cv[1] * q[1] + cv[2] * q[2] + cv[3] * q[3];
    bool z01 = depth_zero_to_one();
    float[<4>] c = cv * ((z01 ? 1.0f : 2.0f) / dq);
    // the third row: z' = c . p (0..1), or c . p - w (-1..1)
    for (int i = 0; i < 4; i++) p[i][2] = z01 ? c[i] : c[i] - p[i][3];
    mc.projection_matrix = p;
    r.view_proj = p.mul(mc.matrix_world_inverse);

    // the surfaces that show it don't take part (they are on the plane)
    hiding = r;
    hidden.clear();
    scene.traverse(&hide_reflecting);

    SgView ssr_color = self.ssr_color;
    SgView ssr_depth = self.ssr_depth;
    SgView ssr_back = self.ssr_back;
    self.clear_reflection_source();
    self.render_target = r.target;
    bool shadows = self.shadow_map_auto_update;
    // (the shadow maps as the last frame left them: a frame late in the mirror image)
    self.shadow_map_auto_update = !r.reuse_shadows;
    self.render(scene, mc);
    self.shadow_map_auto_update = shadows;
    self.render_target = main_rt;
    self.ssr_color = ssr_color;
    self.ssr_depth = ssr_depth;
    self.ssr_back = ssr_back;
    foreach (o : hidden) o.visible = true;
    r.rendered = true;
}

Reflector* hiding @local;
List{Object3D*} hidden @local;

fn void hide_reflecting(Object3D* o) @local
{
    if (o.kind != MESH || !o.visible) return;
    Mesh* m = (Mesh*)o;
    if (m.material && m.material.reflector == hiding)
    {
        o.visible = false;
        hidden.push(o);
    }
}
