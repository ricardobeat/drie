// Planar reflections (`Reflector`) rendered from a mirrored perspective camera into a dedicated
// linear `RGBA16F` `RenderTarget` each frame. Lengyel's oblique near plane (`oblique_projection`)
// clips geometry behind the mirror plane while surfaces referencing the reflector via
// `Material.reflector` keep their own shading, normal-map distortion, and roughness blur. Call
// `Renderer.render_reflector()` before `Renderer.render()` with the main render target active so
// the reflection target scales to match it.

module drie;
import std::math, sokol;

// Planar reflection target and plane definition assigned to surfaces via `Material.reflector`.
//
//   Reflector* mirror = drie::reflector({ 0, 1, 0 }, FLOOR_Y);
//   floor_material.reflector = mirror;
//   // Each frame, with the renderer's target set as for the main render:
//   renderer.render_reflector(mirror, scene, camera);
//   renderer.render(scene, camera);
struct Reflector {
    Vector3 normal; // Unit normal of the mirror plane (`dot(normal, p) == constant`).
    float constant; // Plane offset along `normal` in meters; the positive half-space is mirrored.
    float resolution; // Render target scale relative to the main render target (`0..1`).
    float distortion; // Ray travel distance (m) used to scale normal-map UV distortion.
    float blur; // Maximum blur radius at roughness 1 as a fraction of reflection height.
    float strength; // Reflection multiplier (`1` uses the material's Fresnel term directly).
    float clip_bias; // Margin behind the plane kept unclipped (m) to prevent contact gaps.
    // Optional world-space cull box (m): only objects overlapping `[box_min, box_max]` render
    // into the reflection. Ignored when `box_min == box_max`.
    Vector3 box_min;
    Vector3 box_max; // Maximum corner of the optional world-space reflection cull box (m).
    bool reuse_shadows; // When true (default), reuses the previous frame's shadow maps instead of re-rendering them.
    RenderTarget* target; // Linear `RGBA16F` target resized automatically by `render_reflector()`.
    PerspectiveCamera* camera; // Internal mirrored camera with custom oblique projection.
    Mat4 view_proj; // World-to-clip matrix of the latest reflection render.
    bool rendered; // True if the camera was on the front side and the reflection rendered this frame.
}

// Allocates a planar reflector for the plane `dot(normal, p) == constant` at `resolution` scale.
fn Reflector* reflector(Vector3 normal = { 0, 1, 0 }, float constant = 0, float resolution = 0.5) {
    Reflector* r = mem::new(Reflector);
    *r = {
        .normal = normal.normalize(),
        .constant = constant,
        .resolution = resolution,
        .distortion = 0.6,
        .blur = 0.05,
        .strength = 1,
        .clip_bias = 0.003,
        .reuse_shadows = true
    };
    r.camera = perspective_camera();
    r.camera.custom_projection = true;
    r.camera.matrix_auto_update = false;
    return r;
}

// ---- Reflection math

// Returns the Householder reflection matrix across the mirror plane: `p -> p - 2 (n.p - c) n`.
fn Mat4 Reflector.mirror(&self) {
    Vector3 n = self.normal;
    float c   = self.constant;
    Mat4 m    = mat4_identity();
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

// Tilts `cam`'s near plane onto `dot(normal, p) == constant` using Lengyel's oblique projection,
// clipping everything where `dot(normal, p) < constant - clip_bias`. Call after
// `cam.update_matrix_world()` and `cam.update_projection_matrix()`, with `cam.custom_projection`
// enabled so `Renderer.render()` preserves the modified projection matrix.
fn void oblique_projection(Camera* cam, Vector3 normal, float constant, float clip_bias = 0) {
    Vector3 n = normal.normalize();
    Mat4 mw   = cam.matrix_world;
    float[<4>] pw = { n.x, n.y, n.z, -constant + clip_bias };
    float[<4>] cv;
    for (int i = 0; i < 4; i++) cv[i] = mw[i][0] * pw[0] + mw[i][1] * pw[1] + mw[i][2] * pw[2] + mw[i][3] * pw[3];
    Mat4 p            = cam.projection_matrix;
    Mat4 pinv         = p.invert();
    float[<4>] corner = { cv[0] > 0 ? 1 : (cv[0] < 0 ? -1 : 0), cv[1] > 0 ? 1 : (cv[1] < 0 ? -1 : 0), 1, 1 };
    float[<4>] q;
    for (int i = 0; i < 4; i++) {
        q[i] = pinv[0][i] * corner[0] + pinv[1][i] * corner[1] + pinv[2][i] * corner[2] + pinv[3][i] * corner[3];
    }
    float dq = cv[0] * q[0] + cv[1] * q[1] + cv[2] * q[2] + cv[3] * q[3];
    if (math::abs(dq) < 1e-6f) return;
    bool z01     = depth_zero_to_one();
    float[<4>] c = cv * ((z01 ? 1.0f : 2.0f) / dq);
    // Replace third row so clip depth is `c . p` (`[0, 1]`) or `c . p - w` (`[-1, 1]`).
    for (int i = 0; i < 4; i++) p[i][2] = z01 ? c[i] : c[i] - p[i][3];
    cam.projection_matrix = p;
}

// ---- Rendering

// Renders `scene` into `reflector.target` from the mirrored viewpoint of `camera`. Call before
// `Renderer.render()` with the main render target already set so target sizing matches; restores
// the active render target, shadow-update flag, and SSR views on return.
fn void Renderer.render_reflector(&self, Reflector* reflector, Scene* scene, PerspectiveCamera* camera) {
    Reflector* r = reflector;
    r.rendered = false;
    if (!camera.parent) camera.update_matrix_world();
    Vector3 eye = camera.matrix_world.get_position();
    if (r.normal.dot(eye) - r.constant <= 0) return;

    RenderTarget* main_rt = self.render_target;
    int w                 = math::max((int)((main_rt ? main_rt.width : sapp::width()) * r.resolution), 1);
    int h                 = math::max((int)((main_rt ? main_rt.height : sapp::height()) * r.resolution), 1);
    if (!r.target) r.target = render_target(w, h, 1, RGBA16F);
    r.target.set_size(w, h, 1);

    // Combine plane reflection with a horizontal flip so determinant stays +1 and triangle
    // winding is preserved; the shader's projection via `view_proj` maps the flipped image back.
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

    // Align the near plane to the mirror plane so geometry behind the mirror is clipped.
    oblique_projection(mc, r.normal, r.constant, r.clip_bias);
    r.view_proj = mc.projection_matrix.mul(mc.matrix_world_inverse);


    SgView ssr_color = self.ssr_color;
    SgView ssr_depth = self.ssr_depth;
    SgView ssr_back  = self.ssr_back;
    self.clear_reflection_source();
    self.render_target = r.target;
    bool shadows = self.shadow_map_auto_update;
    // Reusing shadow maps avoids re-rendering shadows from the mirrored view (one frame latent).
    self.shadow_map_auto_update = !r.reuse_shadows;
    // Skip meshes referencing `r` during the reflection pass since they lie on the mirror plane.
    self.reflecting = r;
    self.render(scene, mc);
    self.reflecting = null;
    self.shadow_map_auto_update = shadows;
    self.render_target = main_rt;
    self.ssr_color = ssr_color;
    self.ssr_depth = ssr_depth;
    self.ssr_back = ssr_back;
    r.rendered = true;
}