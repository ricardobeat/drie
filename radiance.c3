// Radiance cascades (Sannikov 2023) in world space: light that bounces around, and the glow
// of emissive things, worked out anew every frame on the GPU. (Not in three.js.)
//
//   RadianceCascades* rc = drie::radiance_cascades(room_min, room_max);
//   rc.voxelize(scene);              // once, and again when the solid things in the box change
//   scene.radiance = rc;
//   // each frame, before render():
//   rc.update(scene);
//
// A grid of probes fills the box. Cascade 0 has one every `spacing` meters, each looking in
// 4 x 4 directions for a short way (`interval`); each cascade above has an eighth of the
// probes, four times the directions and looks four times as far, from where the one below
// stopped. A ray that finds nothing in its stretch goes on in the cascade above, so cascade 0
// ends up with all the light from around each probe: near things in detail, far things
// coarsely, which is about what their penumbras need.
//
// The rays are marched through a voxel copy of the scene (voxelize: each solid mesh's color
// and glow; see-through materials leave no voxels). A voxel that a ray hits gives off its
// glow, plus what the scene's point, spot and directional lights put on it (shadowed through
// the voxels: a directional light's shadow ray goes to the edge of the box)
// and what the probes found the frame before: bounce after bounce over the frames. Rays that
// leave the box see the sky (the scene's hemisphere and ambient lights, or sky_color and
// ground_color).
//
// Lit materials inside the box take their indirect light from the probes instead of the
// scene's ambient, hemisphere and irradiance volume. Directions are jittered and blended over
// frames (`blend`), so the result takes some frames to settle after a change.
module drie;
import std::math, std::collections::list;
import sokol;

const int RC_MAX_LEVELS = 8;
const int RC_MAX_LIGHTS = 64;

struct RadianceCascades {
    Vector3 min;
    Vector3 max;
    // between cascade 0's probes (m)
    float spacing;
    int nx;
    int ny;
    int nz;
    int levels;
    // directions across cascade 0's square (4: 16 directions)
    int base_directions;
    // how far cascade 0's rays reach (m)
    float interval;
    float voxel_size;
    int vx;
    int vy;
    int vz;
    // the share of each new frame in the probes
    float blend;
    // multiplies the light the probes give to the materials
    float intensity;
    // how much of the probes' light bounces on (1: all of it)
    float bounce;
    // the sky's radiance overhead and from below, linear; taken from the scene's hemisphere
    // and ambient lights while sky_from_scene is on
    Vector3 sky_color;
    Vector3 ground_color;
    bool sky_from_scene;
    // Surfaces look the probes up this far off along their normal (m).
    float normal_offset;
    // two numbers per voxel: rgb8 color and bit 24 = solid; the glow as rgb8 with an exponent
    uint[] voxels;
    bool voxels_changed;
    // one ray in this many is traced each frame (1: all of them; more is cheaper and slower
    // to follow a change)
    int update_every;
    StorageBuffer* voxel_buf;
    StorageBuffer* brick_buf;
    StorageBuffer* cascade_buf;
    StorageBuffer* irradiance_buf;
    StorageBuffer* light_buf;
    ComputePass* trace;
    ComputePass* gather;
    int[RC_MAX_LEVELS] offsets;
    int[RC_MAX_LEVELS] counts;
    int frame;
}

fn RadianceCascades* radiance_cascades(Vector3 min, Vector3 max, float spacing = 0.25, int levels = 4) {
    RadianceCascades* r = mem::new(RadianceCascades);
    Vector3 size        = max - min;
    *r = {
        .min = min,
        .spacing = spacing,
        .nx = math::max((int)math::ceil(size.x / spacing), 1),
        .ny = math::max((int)math::ceil(size.y / spacing), 1),
        .nz = math::max((int)math::ceil(size.z / spacing), 1),
        .levels = math::clamp(levels, 1, RC_MAX_LEVELS),
        .base_directions = 4,
        .interval = spacing,
        .voxel_size = spacing / 4,
        .blend = 0.1,
        .update_every = 1,
        .intensity = 1,
        .bounce = 1,
        .sky_from_scene = true,
        .normal_offset = spacing * 0.25f,
    };
    r.max = min + (Vector3) { r.nx, r.ny, r.nz } * spacing;
    r.vx = r.nx * 4;
    r.vy = r.ny * 4;
    r.vz = r.nz * 4;
    r.voxels = mem::new_array(uint, (sz)r.vx * r.vy * r.vz * 2);
    r.voxels_changed = true;
    int at = 0;
    for (int l = 0; l < r.levels; l++) {
        int d      = r.base_directions << l;
        int[<3>] n = r.level_dims(l);
        r.offsets[l] = at;
        r.counts[l] = n.x * n.y * n.z * d * d;
        at += r.counts[l];
    }
    r.cascade_buf = storage_buffer((sz)at * 16, label: "drie-rc-cascades");
    r.irradiance_buf = storage_buffer((sz)r.nx * r.ny * r.nz * 6 * 16, label: "drie-rc-irradiance");
    r.light_buf = storage_buffer(RC_MAX_LIGHTS * RcLight::size, cpu_write: true, label: "drie-rc-lights");
    r.voxel_buf = storage_buffer(16, label: "drie-rc-voxels");
    r.brick_buf = storage_buffer(16, label: "drie-rc-bricks");
    SgBackend backend = sg::query_backend();
    r.trace = compute_pass(shaders::rc_trace_shader_desc(backend), "drie-rc-trace");
    r.gather = compute_pass(shaders::rc_gather_shader_desc(backend), "drie-rc-gather");
    return r;
}

// The probes of a cascade along each axis.
fn int[<3>] RadianceCascades.level_dims(&self, int level) {
    int add = (1 << level) - 1;
    return {
        math::max((self.nx + add) >> level, 1),
        math::max((self.ny + add) >> level, 1),
        math::max((self.nz + add) >> level, 1)
    };
}

fn void RadianceCascades.clear_voxels(&self) {
    self.voxels[..] = 0;
    self.voxels_changed = true;
}

fn uint rc_pack_glow(Vector3 c) @local {
    float m = math::max(c.x, math::max(c.y, c.z));
    if (m < 1e-5f) return 0;
    int e = (int)math::ceil(math::log2(m));
    e = math::clamp(e, -126, 126);
    float scale = 255.0f / math::pow(2.0f, (float)e);
    uint r      = (uint)math::clamp(c.x * scale + 0.5f, 0.0f, 255.0f);
    uint g      = (uint)math::clamp(c.y * scale + 0.5f, 0.0f, 255.0f);
    uint b      = (uint)math::clamp(c.z * scale + 0.5f, 0.0f, 255.0f);
    return r | (g << 8) | (b << 16) | ((uint)(e + 128) << 24);
}

// Makes the voxel at `p` solid, with this color (linear, 0..1) and glow (linear radiance).
// Where two things share a voxel, the one that glows more keeps it.
fn void RadianceCascades.set_voxel(&self, Vector3 p, Vector3 albedo, Vector3 glow) {
    Vector3 g = (p - self.min) / self.voxel_size;
    int x     = (int)math::floor(g.x);
    int y     = (int)math::floor(g.y);
    int z     = (int)math::floor(g.z);
    if (x < 0 || y < 0 || z < 0 || x >= self.vx || y >= self.vy || z >= self.vz) return;
    sz at       = (((sz)z * self.vy + y) * self.vx + x) * 2;
    uint packed = rc_pack_glow(glow);
    if (self.voxels[at] && (self.voxels[at + 1] >> 24) > (packed >> 24)) return;
    uint r  = (uint)math::clamp(albedo.x * 255 + 0.5f, 0.0f, 255.0f);
    uint gg = (uint)math::clamp(albedo.y * 255 + 0.5f, 0.0f, 255.0f);
    uint b  = (uint)math::clamp(albedo.z * 255 + 0.5f, 0.0f, 255.0f);
    self.voxels[at] = r | (gg << 8) | (b << 16) | 0x1000000;
    self.voxels[at + 1] = packed;
    self.voxels_changed = true;
}

// Adds a triangle (world space) to the voxels.
fn void RadianceCascades.add_triangle(&self, Vector3 a, Vector3 b, Vector3 c, Vector3 albedo, Vector3 glow) {
    for (int k = 0; k < 3; k++) {
        float lo = math::min(a[k], math::min(b[k], c[k]));
        float hi = math::max(a[k], math::max(b[k], c[k]));
        if (hi < self.min[k] || lo > self.max[k]) return;
    }
    // points on it no further apart than half a voxel
    float h    = self.voxel_size * 0.5f;
    Vector3 ab = b - a;
    Vector3 ac = c - a;
    int n1     = math::min((int)math::ceil(ab.length() / h) + 1, 4096);
    int n2     = math::min((int)math::ceil(ac.length() / h) + 1, 4096);
    for (int i = 0; i <= n1; i++) {
        float u = (float)i / n1;
        for (int j = 0; j <= n2; j++) {
            float v = (float)j / n2;
            if (u + v > 1.0001f) break;
            self.set_voxel(a + ab * u + ac * v, albedo, glow);
        }
    }
}

// Whether a sphere of the mesh's geometry under `m` touches the box.
fn bool RadianceCascades.touches(&self, BufferGeometry* g, Mat4 m) {
    Vector3 c = m.transform_point(g.bounds_center);
    float scale = math::max(
        ((Vector3) { m[0][0], m[0][1], m[0][2] }).length(),
        math::max(
            ((Vector3) { m[1][0], m[1][1], m[1][2] }).length(),
            ((Vector3) { m[2][0], m[2][1], m[2][2] }).length()
        )
    );
    return self.reaches(c, g.bounds_radius * scale + 1e-3f);
}

// Adds a mesh as it stands now (its world matrix must be up to date). See-through, cut-out
// and hidden materials are left out; an unlit one (MeshBasicMaterial) glows with its color.
// The color is the material's times its map's average. `albedo_scale` scales it.
fn void RadianceCascades.add_mesh(&self, Mesh* mesh, Vector3 albedo_scale = { 1, 1, 1 }) {
    Material* mat     = mesh.material;
    BufferGeometry* g = mesh.geometry;
    if (!mat || !g || !mat.visible || mat.transparent || mat.opacity < 1 || mat.alpha_test > 0 || mat.glass) return;
    if (g.bounds_radius <= 0) g.compute_bounding_sphere();
    if (mesh.instance_count <= 0 && !self.touches(g, mesh.matrix_world)) return;
    Vector3 albedo = color_to_linear(mat.color) * albedo_scale;
    if (mat.map) albedo = albedo * mat.map.average_color();
    Vector3 glow = color_to_linear(mat.emissive) * mat.emissive_intensity;
    if (mat.type == BASIC) {
        glow = albedo;
        albedo = { };
    }
    sz count      = g.indices.len() ? g.indices.len() : (sz)g.vertex_count();
    int instances = math::max(mesh.instance_count, 1);
    for (int n = 0; n < instances; n++) {
        Mat4 m = mesh.matrix_world;
        if (mesh.instance_count > 0) {
            // 12 floats per instance: the rows of a 3x4 matrix
            float[] row = mesh.instance_data.entries[(sz)n * 12:12];
            Mat4 im     = mat4_identity();
            for (int r = 0; r < 3; r++) {
                for (int col = 0; col < 4; col++) im[col][r] = row[r * 4 + col];
            }
            m = m.mul(im);
            if (!self.touches(g, m)) continue;
        }
        for (sz i = 0; i + 2 < count; i += 3) {
            Vector3[3] p;
            for (int k = 0; k < 3; k++) {
                sz vi     = g.indices.len() ? (sz)g.indices[i + k] : i + k;
                float[] v = g.vertices.entries[vi * FLOATS_PER_VERTEX:3];
                p[k] = m.transform_point({ v[0], v[1], v[2] });
            }
            self.add_triangle(p[0], p[1], p[2], albedo, glow);
        }
    }
}

fn void rc_add_tree(RadianceCascades* self, Object3D* o) @local {
    if (!o.visible) return;
    if (o.kind == MESH) self.add_mesh((Mesh*)o);
    foreach (c : o.children) rc_add_tree(self, c);
}

// Fills the voxels from every visible mesh under `root` (after clearing them). Call it when
// the world matrices are up to date (after a render, or root.update_matrix_world()).
fn void RadianceCascades.voxelize(&self, Object3D* root) {
    self.clear_voxels();
    rc_add_tree(self, root);
}

struct RcLightList @local {
    RcLight[RC_MAX_LIGHTS] lights;
    int count;
    Vector3 sky;
    Vector3 ground;
}

fn void rc_collect_lights(RadianceCascades* self, Object3D* o, RcLightList* out) @local {
    if (!o.visible) return;
    if (o.kind == LIGHT) {
        Light* l  = (Light*)o;
        Vector3 c = color_to_linear(l.color) * l.intensity;
        switch (l.light_type) {
            case AMBIENT:
                out.sky += c;
                out.ground += c;
            case HEMISPHERE:
                out.sky += c;
                out.ground += color_to_linear(((HemisphereLight*)l).ground_color) * l.intensity;
            case DIRECTIONAL:
                // (towards the light; w = -3 marks it)
                DirectionalLight* dl = (DirectionalLight*)l;
                Vector3 from = l.matrix_world.get_position();
                Vector3 to = dl.target ? dl.target.matrix_world.get_position() : (Vector3) { };
                Vector3 d = from - to;
                if (out.count < RC_MAX_LIGHTS && l.intensity > 0 && d.length() > 1e-6f) {
                    out.lights[out.count++] = {
                        .pos = { ...d.normalize(), 0 },
                        .color = { ...c, 0 },
                        .dir = { 0, 0, 0, -3 }
                    };
                }
            case POINT:
                PointLight* pl = (PointLight*)l;
                Vector3 p = l.matrix_world.get_position();
                if (out.count < RC_MAX_LIGHTS && l.intensity > 0 && self.reaches(p, pl.distance)) {
                    out.lights[out.count++] = {
                        .pos = { ...p, pl.distance },
                        .color = { ...c, pl.decay },
                        .dir = { 0, 0, 0, -2 }
                    };
                }
            case SPOT:
                SpotLight* sl = (SpotLight*)l;
                Vector3 p = l.matrix_world.get_position();
                if (out.count < RC_MAX_LIGHTS && l.intensity > 0 && self.reaches(p, sl.distance)) {
                    float cos_outer = math::cos(clamp(sl.angle, 0, PI / 2));
                    float cos_inner = math::cos(clamp(sl.angle, 0, PI / 2) * (1 - clamp(sl.penumbra, 0, 1)));
                    out.lights[out.count++] = {
                        .pos = { ...p, sl.distance },
                        .color = { ...c, sl.decay },
                        .dir = { ...spot_direction(sl), cos_outer },
                        .params = { math::max(cos_inner, cos_outer + 1e-4f), 0, 0, 0 },
                    };
                }
            default:
        }
    }
    foreach (c : o.children) rc_collect_lights(self, c, out);
}

// Whether a light at `p` that shines `distance` far (0: without end) reaches the box.
fn bool RadianceCascades.reaches(&self, Vector3 p, float distance) {
    if (distance <= 0) return true;
    float d2 = 0;
    for (int k = 0; k < 3; k++) {
        float d = math::max(math::max(self.min[k] - p[k], p[k] - self.max[k]), 0.0f);
        d2 += d * d;
    }
    return d2 < distance * distance;
}

// Traces the cascades for this frame with the scene's lights as they are. Call it once a
// frame before render(), outside of any pass.
fn void RadianceCascades.update(&self, Scene* scene) {
    if (self.voxels_changed) {
        self.voxel_buf.write(self.voxels.ptr, (sz)self.voxels.len * uint::size);
        // the solid voxels in each block of 4 x 4 x 4, for the rays to skip the empty ones
        uint[] bricks = mem::new_array(uint, (sz)self.nx * self.ny * self.nz);
        defer free(bricks);
        for (int z = 0; z < self.vz; z++) {
            for (int y = 0; y < self.vy; y++) {
                for (int x = 0; x < self.vx; x++) {
                    if (self.voxels[(((sz)z * self.vy + y) * self.vx + x)
                        * 2]) bricks[((sz)(z / 4) * self.ny + y / 4) * self.nx + x / 4]++;
                }
            }
        }
        self.brick_buf.write(bricks.ptr, (sz)bricks.len * uint::size);
        self.voxels_changed = false;
    }
    RcLightList* list = mem::new(RcLightList);
    defer free(list);
    rc_collect_lights(self, scene, list);
    if (self.sky_from_scene) {
        // the lights give what arrives on a surface; the sky that puts that much on one
        // facing it is that over pi
        self.sky_color = list.sky / (float)PI;
        self.ground_color = list.ground / (float)PI;
    }
    self.light_buf.write(&list.lights, (sz)math::max(list.count, 1) * RcLight::size);

    int every = math::max(self.update_every, 1);
    int turn  = self.frame / every;
    RcParams p = {
        .rc_min = { ...self.min, self.spacing },
        .rc_dims = {
            self.nx,
            self.ny,
            self.nz,
            0
        }, // the first frames take all of what they find, so that it doesn't start from black
        .rc_tune = {
            self.interval,
            math::max(self.blend, 1.0f / (float)(turn + 1)),
            (float)(self.frame % 4096),
            self.voxel_size
        },
        .rc_info = { self.base_directions, self.levels, list.count, 0 },
        .rc_sky = { ...self.sky_color, self.bounce },
        .rc_ground = { ...self.ground_color, 0 },
        .rc_vox = { self.vx, self.vy, self.vz, every },
    };
    for (int l = 0; l < self.levels; l++) {
        if (l < 4) {
            p.rc_off0[l] = self.offsets[l];
        } else {
            p.rc_off1[l - 4] = self.offsets[l];
        }
    }
    ComputePass* t = self.trace;
    t.set_buffer(shaders::VIEW_RC_VOXELS, self.voxel_buf);
    t.set_buffer(shaders::VIEW_RC_CASCADES, self.cascade_buf);
    t.set_buffer(shaders::VIEW_RC_IRRADIANCE, self.irradiance_buf);
    t.set_buffer(shaders::VIEW_RC_LIGHTS, self.light_buf);
    t.set_buffer(shaders::VIEW_RC_BRICKS, self.brick_buf);
    ComputePass* g = self.gather;
    g.set_buffer(shaders::VIEW_RC_VOXELS, self.voxel_buf);
    g.set_buffer(shaders::VIEW_RC_BRICKS, self.brick_buf);
    g.set_buffer(shaders::VIEW_RC_GATHER_CASCADES, self.cascade_buf);
    g.set_buffer(shaders::VIEW_RC_GATHER_IRRADIANCE, self.irradiance_buf);
    // each cascade in a pass of its own: it reads what the one above just wrote
    for (int l = self.levels - 1; l >= 0; l--) {
        p.rc_dims[3] = l;
        p.rc_info[3] = self.counts[l];
        t.set_uniforms(shaders::UB_RC_PARAMS, &p, RcParams::size);
        begin_compute("drie-rc-trace");
        t.dispatch(((self.counts[l] + every - 1) / every + 63) / 64);
        end_compute();
    }
    g.set_uniforms(shaders::UB_RC_PARAMS, &p, RcParams::size);
    begin_compute("drie-rc-gather");
    g.dispatch((self.nx * self.ny * self.nz + 63) / 64);
    end_compute();
    self.frame++;
}

// Start over (after a cut, or when much has changed): the next frame takes all it finds.
fn void RadianceCascades.reset(&self) {
    self.frame = 0;
}

fn void RadianceCascades.dispose(&self) {
    self.voxel_buf.dispose();
    self.brick_buf.dispose();
    self.cascade_buf.dispose();
    self.irradiance_buf.dispose();
    self.light_buf.dispose();
    free(self.voxels);
}