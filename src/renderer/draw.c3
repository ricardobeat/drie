// Scene traversal, frustum culling, uniform packing, draw submission, and transmission passes.
// Collects visible meshes/lights/LODs (collect), packs IrradianceVolumes and per-frame lighting
// uniforms (use_irradiance, build_light_params), resolves MaterialKey and texture slots
// (material_slots, material_key_of), submits forward/G-Buffer/ShaderMaterial/shadow draws
// (draw_item, draw_shadow_item), and drives scene backgrounds, fog backdrop, and transmission
// mipmap generation (render_transmission, copy_for_transmission, fill_levels). Uses
// camera-relative view-projection (relative_view_proj) to keep float precision far from origin.
module drie;
import std::collections::list, std::collections::map, std::sort, std::math, std::io;
import sokol, drie::nodes;

// ---- Frustum culling and scene collection

// Extracts 5 inward-pointing clip planes (left, right, bottom, top, far) for NDC sub-box
// `[x0, x1] x [y0, y1]` from view-projection matrix `vp`.
fn Frustum frustum_from_ndc(Mat4 vp, float x0, float x1, float y0, float y1) @private {
    Vec4[4] rows;
    for (int r = 0; r < 4; r++) rows[r] = { vp[0][r], vp[1][r], vp[2][r], vp[3][r] };
    Frustum f;
    f.planes[0] = rows[0] - rows[3] * x0;
    f.planes[1] = rows[3] * x1 - rows[0];
    f.planes[2] = rows[1] - rows[3] * y0;
    f.planes[3] = rows[3] * y1 - rows[1];
    f.planes[4] = rows[3] - rows[2];
    foreach (&p : f.planes) {
        float len = ((Vector3) { (*p).x, (*p).y, (*p).z }).length();
        if (len > 0) *p /= len;
    }
    return f;
}

// Extracts 5 inward-pointing clip planes (left, right, bottom, top, far) from `vp`.
fn Frustum frustum_from(Mat4 vp) @private => frustum_from_ndc(vp, -1, 1, -1, 1);

// True when the sphere `(c, radius)` intersects or lies inside all frustum planes.
fn bool Frustum.contains_sphere(&self, Vector3 c, float radius) {
    foreach (p : self.planes) {
        if (p.x * c.x + p.y * c.y + p.z * c.z + p.w < -radius) return false;
    }
    return true;
}

fn float max_scale(Mat4 m) @private {
    float sx = ((Vector3) { m[0][0], m[0][1], m[0][2] }).length();
    float sy = ((Vector3) { m[1][0], m[1][1], m[1][2] }).length();
    float sc = ((Vector3) { m[2][0], m[2][1], m[2][2] }).length();
    return math::max(sx, math::max(sy, sc));
}

// Traverses `obj` recursively, frustum-culling meshes into `state.opaque`, `state.transparent`,
// or `state.casters`, collecting lights, and selecting LOD levels (with optional cross-fade)
// measured from `lod_from`.
fn void collect(
    Object3D* obj,
    Mat4 view,
    Frustum* frustum,
    Vector3 lod_from,
    bool shadow_pass,
    float fade = 0,
    float fade_dir = 0
) @private {
    if (!obj.visible) return;
    switch (obj.kind) {
        case MESH:
            Mesh* m = (Mesh*)obj;
            if (!m.geometry || !m.material || !m.material.visible) break;
            if (m.shadow_only && !shadow_pass) break;
            // Exclude a Reflector's own mirror meshes when rendering its reflection texture.
            if (state.renderer.reflecting && m.material.reflector == state.renderer.reflecting) break;
            if (shadow_pass
                && (!m.cast_shadow || m.material.transparent || m.is_line() || m.billboard != NO_BILLBOARD)) break;
            if (shadow_pass && state.skip_instanced && m.instance_data.len() > 0) break;
            BufferGeometry* g = m.geometry;
            if (!g.uploaded && g.indices.len() == 0) break;
            // Lines require at least two vertices to form a segment.
            if (m.is_line() && g.indices.len() < 2) break;
            g.upload();
            Vector3 center = g.bounds_center;
            float radius = g.bounds_radius;
            if (m.instance_data.len() > 0) {
                upload_instances_gpu(m);
                center = m.instance_center;
                radius = m.instance_radius;
            }
            Vector3 wc = m.matrix_world.transform_point(center);
            float r = radius * max_scale(m.matrix_world) + m.material.sway * 2;
            if (m.frustum_culled && !frustum.contains_sphere(wc, r)) break;
            // Cull meshes outside an active Reflector's bounding box.
            Reflector* refl = state.renderer.reflecting;
            if (refl && refl.box_min != refl.box_max) {
                Vector3 nearest = {
                    clamp(wc.x, refl.box_min.x, refl.box_max.x),
                    clamp(wc.y, refl.box_min.y, refl.box_max.y),
                    clamp(wc.z, refl.box_min.z, refl.box_max.z)
                };
                if ((nearest - wc).length() > r) break;
            }
            Vector3 vc = view.transform_point(wc);
            float dist = math::max(vc.length(), r);
            DrawItem item = { m, vc.z, fade, fade_dir, 0, r * r / (dist * dist) };
            if (shadow_pass) {
                state.casters.push(item);
            } else if (m.material.transparent) {
                state.transparent.push(item);
            } else {
                state.opaque.push(item);
            }
        case LIGHT:
            if (!shadow_pass) state.lights.push((Light*)obj);
        case LOD:
            Lod* l = (Lod*)obj;
            float d = (l.matrix_world.get_position() - lod_from).length() / math::max(state.renderer.lod_scale, 0.01f);
            // Bias shadow passes toward coarser LOD levels.
            if (shadow_pass) d *= math::max(state.renderer.shadow_lod_bias, 1.0f);
            sz pick = -1;
            for (sz i = 0; i < l.distances.len(); i++) {
                if (d >= l.distances[i]) pick = i;
            }
            // Cross-fade adjacent LOD levels in the transition band during main passes.
            if (l.fade > 0 && !shadow_pass) {
                float half = l.fade * 0.5f;
                for (sz i = 1; i < l.distances.len() && i < l.children.len(); i++) {
                    float t = (d - (l.distances[i] - half)) / l.fade;
                    if (t <= 0 || t >= 1) continue;
                    collect(l.children[i - 1], view, frustum, lod_from, shadow_pass, t, -1);
                    collect(l.children[i], view, frustum, lod_from, shadow_pass, t, 1);
                    return;
                }
            }
            if (pick >= 0 && pick < l.children.len()) collect(l.children[pick], view, frustum, lod_from, shadow_pass);
            return;
        default:
            break;
    }
    foreach (c : obj.children) collect(c, view, frustum, lod_from, shadow_pass, fade, fade_dir);
}

// ---- Irradiance and per-frame light uniforms

// Uploads up to MAX_IRRADIANCE_VOLUMES active volumes on `scene` and binds either the single
// volume texture or a packed 3D atlas (`state.gi_atlas`) into `state.gi_view`.
fn void use_irradiance(Scene* scene) @private {
    if (!state.no_irradiance) state.no_irradiance = irradiance_volume({ 0, 0, 0 }, { 1, 1, 1 }, 1);
    state.gi_count = 0;
    if (scene.irradiance && scene.irradiance != state.gi_skip) state.gi_volumes[state.gi_count++] = scene.irradiance;
    foreach (v : scene.irradiance_volumes) {
        if (v && v != state.gi_skip && v != state.gi_add && state.gi_count < MAX_IRRADIANCE_VOLUMES) {
            state.gi_volumes[state.gi_count++] = v;
        }
    }
    if (state.gi_add && scene.irradiance != state.gi_add && state.gi_count < MAX_IRRADIANCE_VOLUMES) {
        state.gi_volumes[state.gi_count++] = state.gi_add;
    }
    for (int i = state.gi_count; i < MAX_IRRADIANCE_VOLUMES; i++) state.gi_volumes[i] = null;
    state.irradiance = state.gi_count ? state.gi_volumes[0] : state.no_irradiance;
    bool changed = state.gi_atlas_of != state.gi_volumes;
    for (int i = 0; i < state.gi_count; i++) {
        changed |= !state.gi_volumes[i].uploaded;
        state.gi_volumes[i].upload();
    }
    state.irradiance.upload();
    if (state.gi_count <= 1) {
        state.gi_view = state.irradiance.view;
        return;
    }
    if (changed || !state.gi_atlas.id) {
        Vector3 size = gi_atlas_size();
        int w        = (int)size.x;
        int h        = (int)size.y;
        int d        = (int)size.z;
        ushort[] px  = mem::new_array(ushort, (sz)w * h * d * 4);
        defer free(px);
        int x0 = 0;
        for (int i = 0; i < state.gi_count; i++) {
            state.gi_volumes[i].fill(px, w, h, x0);
            x0 += state.gi_volumes[i].nx * 6;
        }
        if (state.gi_atlas.id) {
            sg::destroy_view(state.gi_atlas_view);
            sg::destroy_image(state.gi_atlas);
        }
        state.gi_atlas = sg::make_image(
            &&{
                .type = TYPE_3D,
                .width = w,
                .height = h,
                .num_slices = d,
                .pixel_format = RGBA16F,
                .data = { .ptr = px.ptr, .size = (usz)px.len * ushort::size },
            }
        );
        state.gi_atlas_view = sg::make_view(&&{ .texture = { .image = state.gi_atlas } });
        state.gi_atlas_of = state.gi_volumes;
    }
    state.gi_view = state.gi_atlas_view;
}

// Combined 3D texture dimensions `{ sum(nx * 6), max(ny), max(nz) }` of active irradiance volumes.
fn Vector3 gi_atlas_size() @local {
    Vector3 size = { 0, 1, 1 };
    for (int i = 0; i < state.gi_count; i++) {
        IrradianceVolume* v = state.gi_volumes[i];
        size.x += (float)(v.nx * 6);
        size.y = math::max(size.y, (float)v.ny);
        size.z = math::max(size.z, (float)v.nz);
    }
    return size;
}

// Packs collected lights, shadows, cluster grid, GI volumes, reflection probes, radiance
// cascades, and fog into `FsParams`, uploading clustered light storage buffers via `lights_gpu`.
fn FsParams build_light_params(Scene* scene, Camera* camera, SpotLight* caster) @private {
    FsParams p;
    Renderer* r  = state.renderer;
    LightData* g = &lights_gpu;
    g.clear();
    // Under VSM, `params.w` carries VSM depth bias instead of PCF filter radius.
    bool vsm       = r.shadow_type == VSM;
    float exposure = r.tone_mapping == NO_TONE_MAPPING ? 1 : math::max(r.tone_mapping_exposure, 1e-4f);
    p.camera_pos = { ...camera.matrix_world.get_position(), exposure };
    bool linear  = r.render_target && r.render_target.is_linear();
    // Window output encoding: 0 = sRGB, 1 = Display P3 SDR, 2 = Extended Linear Display P3 HDR.
    float window = linear || r.render_target ? 0
        : (r.hdr_on() ? 2 : (r.output_color_space == DISPLAY_P3_COLOR_SPACE ? 1 : 0));
    p.output_params = { linear ? 1 : 0, r.ssr_linear ? 1 : 0, (float)r.tone_mapping.ordinal, window };
    p.cull_min = { ...r.cull_box_min, r.cull_box ? 1 : 0 };
    p.cull_max = { ...r.cull_box_max, 0 };
    Mat4 cw = camera.matrix_world;
    p.dir_cascade_fwd = { ...((Vector3) { -cw[2][0], -cw[2][1], -cw[2][2] }).normalize(), 1 };
    p.camera_right = { ...((Vector3) { cw[0][0], cw[0][1], cw[0][2] }).normalize(), 0 };
    p.camera_up = { ...((Vector3) { cw[1][0], cw[1][1], cw[1][2] }).normalize(), 0 };
    rt_begin();
    foreach (l : state.lights) {
        Vector3 c = color_to_linear(l.color) * l.intensity;
        switch (l.light_type) {
            case AMBIENT:
                p.ambient += { ...c, 0 };
            case HEMISPHERE:
                HemisphereLight* h = (HemisphereLight*)l;
                p.hemi_sky += { ...c, 0 };
                p.hemi_ground += { ...color_to_linear(h.ground_color) * l.intensity, 0 };
            case DIRECTIONAL:
                if (!g.room(g.dirs.len(), r.max_lights)) break;
                DirectionalLight* d = (DirectionalLight*)l;
                Vector3 target = d.target ? d.target.get_world_position() : (Vector3) { 0, 0, 0 };
                Vector3 dir = l.matrix_world.get_position() - target;
                if (d == state.sun_caster) {
                    int n = state.sun_cascades;
                    for (int i = 0; i < math::max(n, 1); i++) {
                        p.dir_shadow_mat[i] = shadow_bias().mul(state.sun_view_proj[i]);
                        p.dir_cascade_rect[i] = state.sun_rect[i];
                        p.dir_cascade_info[i] = state.sun_info[i];
                    }
                    p.dir_shadow_params = {
                        (float)(g.dirs.len() + 1),
                        d.shadow.normal_bias,
                        1.0f / (float)state.sun_size,
                        d.shadow.radius
                    };
                    p.dir_cascade_params = {
                        (float)math::max(n, 1),
                        clamp(d.shadow.cascade_blend, 0, 1),
                        r.debug_cascades ? 1 : 0,
                        n ? 0 : 1
                    };
                    p.dir_cascade_fwd.w = clamp(d.shadow.intensity, 0, 1);
                    p.shadow_ext.x = d.shadow.bias;
                }
                GpuDirLight gd = {
                    .dir = { ...dir, 0 },
                    .color = { ...c, 0 },
                    .shadow = { -1, 1, (float)l.channels, d.shadow.contact_length }
                };
                // Secondary shadowed directional lights use a single atlas tile (dir_tile_shadow).
                AtlasMap* am = d != state.sun_caster && atlas_has_shadow(d) ? atlas_map_of(d) : null;
                if (am) {
                    GpuSpotShadow sh;
                    sh.mat = shadow_bias().mul(am.view_proj);
                    sh.params = { 3, d.shadow.normal_bias, atlas_texel(am), vsm ? d.shadow.bias : d.shadow.radius };
                    sh.rect = atlas_rect(am);
                    float fade = atlas_slot_of(d).fade;
                    gd.shadow = {
                        (float)g.shadows.len(),
                        clamp(d.shadow.intensity, 0, 1) * fade,
                        (float)l.channels,
                        d.shadow.contact_length
                    };
                    g.shadows.push(sh);
                    g.dir_tiles = true;
                }
                if (d.shadow.contact_length > 0) g.contact_shadows = true;
                g.dirs.push(gd);
            case POINT:
                if (!g.room(g.points.len(), r.max_lights)) break;
                PointLight* pl = (PointLight*)l;
                // Tube lights pack half-extent vector along local +X into `params.yzw`.
                Vector3 half;
                if (pl.length > 0) {
                    Mat4 m    = l.matrix_world;
                    Vector3 x = { m[0][0], m[0][1], m[0][2] };
                    float xl  = x.length();
                    if (xl > 0) half = x * (pl.length * 0.5f / xl);
                }
                GpuPointLight gp = {
                    .pos = { ...l.matrix_world.get_position(), pl.distance },
                    .color = { ...c, pl.decay },
                    .params = { pl.radius, half.x, half.y, half.z },
                    .shadow = { -1, 1, (float)l.channels, pl.shadow.contact_length },
                };
                cube_shadows(l, &gp.shadow, g, vsm);
                if (pl.shadow.contact_length > 0) g.contact_shadows = true;
                g.points.push(gp);
                if (pl.radius > 0 || pl.length > 0) g.sized = true;
                if (half != (Vector3) { 0, 0, 0 }) g.tubes = true;
                // Zero-intensity lights receive empty bounds so they stay out of cluster lists.
                LightBounds b = point_light_bounds(pl);
                if (l.intensity <= 0) b = { { }, 0 };
                g.point_bounds.push(b);
            case RECT_AREA:
                // Oriented half-width and half-height axes (ignoring node scale) packed into
                // GpuPointLight fields for `ltc_tables()` evaluation in src/material/material_lit.c3.
                RectAreaLight* ra = (RectAreaLight*)l;
                if (l.intensity <= 0 || !g.room(g.rects.len(), r.max_rect_area_lights)) break;
                Mat4 m = l.matrix_world;
                Vector3 x = { m[0][0], m[0][1], m[0][2] };
                Vector3 y = { m[1][0], m[1][1], m[1][2] };
                float xl = x.length();
                float yl = y.length();
                Vector3 hw = xl > 0 ? x * (ra.width * 0.5f / xl) : (Vector3) { };
                Vector3 hh = yl > 0 ? y * (ra.height * 0.5f / yl) : (Vector3) { };
                GpuPointLight gr = {
                    .pos = { ...m.get_position(), hw.x },
                    .color = { ...c, hw.y },
                    .params = { hw.z, hh.x, hh.y, hh.z },
                    .shadow = { -1, 1, (float)l.channels, 0 },
                };
                cube_shadows(l, &gr.shadow, g, vsm);
                gr.shadow.w = rt_light(r, ra, hw, hh);
                if (gr.shadow.w > 0) {
                    gr.shadow.y = clamp(ra.shadow.intensity, 0, 1);
                    g.rt_shadows = true;
                }
                g.rects.push(gr);
            case SPOT:
                SpotLight* sl = (SpotLight*)l;
                if (!g.room(g.spots.len(), r.max_lights)) break;
                float cos_outer = math::cos(clamp(sl.angle, 0, PI / 2));
                float cos_inner = math::cos(clamp(sl.angle, 0, PI / 2) * (1 - clamp(sl.penumbra, 0, 1)));
                GpuSpotLight gs = {
                    .pos = { ...l.matrix_world.get_position(), sl.distance },
                    .dir = { ...spot_direction(sl), cos_outer },
                    .color = { ...c, sl.decay },
                    .params = {
                        math::max(cos_inner, cos_outer + 1e-4f),
                        sl.radius,
                        sl.specular_radius > 0 ? sl.specular_radius : sl.radius,
                        sl.fog_glow
                    },
                    .shadow = { -1, 1, (float)l.channels, sl.shadow.contact_length },
                };
                spot_map(sl, &gs, g);
                // Clip space -> shadow map UV (V inverted on [0,1] depth backends) and [0,1] depth.
                AtlasMap* am = sl != caster && atlas_has_shadow(sl) ? atlas_map_of(sl) : null;
                if (sl == caster || am) {
                    bool zo   = depth_zero_to_one();
                    Mat4 bias = mat4_identity();
                    bias[0][0] = 0.5f;
                    bias[3][0] = 0.5f;
                    bias[1][1] = zo ? -0.5f : 0.5f;
                    bias[3][1] = 0.5f;
                    bias[2][2] = zo ? 1.0f : 0.5f;
                    bias[3][2] = zo ? 0.0f : 0.5f;
                    GpuSpotShadow sh;
                    sh.mat = bias.mul(shadow_view_proj(sl, true));
                    sh.params = {
                        sl == caster ? 1 : 2,
                        sl.shadow.normal_bias,
                        sl == caster ? 1.0f / (float)state.shadow_size : atlas_texel(am),
                        vsm ? sl.shadow.bias : sl.shadow.radius
                    };
                    float fov   = math::min(sl.angle * 2 * RAD2DEG * clamp(sl.shadow.focus, 0.05f, 1) + 4, 150.0f);
                    float width = 2 * math::tan(fov * DEG2RAD * 0.5f);
                    float far   = sl.shadow.far > 0 ? sl.shadow.far : (sl.distance > 0 ? sl.distance : 50.0f);
                    sh.soft = { sl.shadow.softness, math::max(sl.shadow.near, 0.01f), far, zo ? width : -width };
                    if (am) sh.rect = atlas_rect(am);
                    float fade = am ? atlas_slot_of(sl).fade : 1;
                    gs.shadow = {
                        (float)g.shadows.len(),
                        clamp(sl.shadow.intensity, 0, 1) * fade,
                        (float)l.channels,
                        sl.shadow.contact_length
                    };
                    g.spot_shadows = true;
                    if (sl.shadow.softness > 0) g.soft_shadows = true;
                    g.shadows.push(sh);
                }
                if (sl.shadow.contact_length > 0) g.contact_shadows = true;
                g.spots.push(gs);
                if (gs.params.z > 0) g.sized = true;
                LightBounds b = spot_light_bounds(sl);
                if (l.intensity <= 0) b = { { }, 0 };
                g.spot_bounds.push(b);
        }
    }
    int ndir         = (int)g.dirs.len();
    int npoint       = (int)g.points.len();
    int nspot        = (int)g.spots.len();
    int nrect        = (int)g.rects.len();
    ClusterGrid grid = cluster_grid(camera);
    p.cluster_view_proj = camera.projection_matrix.mul(camera.matrix_world_inverse);
    p.cluster_grid = { CLUSTER_X, CLUSTER_Y, CLUSTER_Z, 0 };
    p.cluster_depth = { grid.near, grid.log_scale, grid.far, 0 };
    g.upload(&grid, r.assign_clusters, r.check_compute);
    p.counts = { ndir, npoint, nspot, nrect };
    Vector3 atlas = gi_atlas_size();
    float x0      = 0;
    for (int i = 0; i < state.gi_count; i++) {
        IrradianceVolume* v = state.gi_volumes[i];
        Vector3 size        = v.max - v.min;
        p.gi_min[i] = { ...v.min, i == 0 && v.replaces_ambient ? 2 : 1 };
        p.gi_inv_size[i] = { 1 / size.x, 1 / size.y, 1 / size.z, v.intensity };
        p.gi_dims[i] = { v.nx, v.ny, v.nz, v.normal_offset };
        p.gi_atlas[i] = { x0 / atlas.x, (float)(v.nx * 6) / atlas.x, (float)v.ny / atlas.y, (float)v.nz / atlas.z };
        x0 += (float)(v.nx * 6);
    }
    if (scene.reflection_probes && scene.reflection_probes.count) {
        ReflectionProbes* rp = scene.reflection_probes;
        rp.upload();
        p.probe_info = { (float)rp.count, 0, 0, 0 };
        for (int i = 0; i < rp.count; i++) {
            ReflectionProbe* pr = &rp.probes[i];
            p.probe_pos[i] = { ...pr.position, pr.distance_range };
            p.probe_min[i] = { ...pr.box_min, pr.blend };
            p.probe_max[i] = { ...pr.box_max, pr.intensity };
        }
        state.probes_view = rp.view;
        state.probes_sampler = rp.sampler;
    } else {
        state.probes_view = { };
    }
    if (scene.radiance) {
        RadianceCascades* rc = scene.radiance;
        p.rc_min = { ...rc.min, rc.spacing };
        // Sparse RC signals hash mode via negative x (-hash_cap), packing nx | (ny << 9) in y.
        p.rc_size = rc.sparse ? (Vec4) {
            -(float)rc.hash_cap,
            (float)(rc.nx | (rc.ny << 9)),
            (float)rc.nz,
            rc.intensity
        }
            : (Vec4) { rc.nx, rc.ny, rc.nz, rc.intensity };
    }
    if (scene.fog) {
        Fog* f = scene.fog;
        float mode;
        switch (f.type) {
            case LINEAR: mode = 1;
            case EXP2: mode = 2;
            case EXP: mode = 3;
        }
        p.fog_color = { ...color_to_linear(f.color), state.renderer.prof_no_fog_scatter ? 0 : f.light_scatter };
        p.fog_params = { f.near, f.far, f.density, mode };
        p.fog_box_min = { ...f.clear_box_min, f.has_clear_box ? 1 : 0 };
        p.fog_box_max = { ...f.clear_box_max, clamp(f.clear_box_density, 0, 1) };
        p.fog_height = { f.height_base, f.height_falloff, state.renderer.display_fog ? 1 : 0, f.shade_density };
        Vector3 sun_dir = f.sun_direction;
        Color sun_color = f.sun_color;
        if (sun_dir == (Vector3) { 0, 0, 0 } || !sun_color) {
            // Fall back to the shadow-casting sun, or the first visible directional light.
            Light* sun = state.sun_caster;
            if (!sun) foreach (l : state.lights) if (l.light_type == DIRECTIONAL && l.visible) {
                sun = l;
                break;
            }
            if (sun) {
                DirectionalLight* d = (DirectionalLight*)sun;
                Vector3 target      = d.target ? d.target.get_world_position() : (Vector3) { 0, 0, 0 };
                if (sun_dir == (Vector3) { 0, 0, 0 }) sun_dir = (sun.matrix_world.get_position() - target).normalize();
                if (!sun_color) sun_color = sun.color;
            }
        }
        p.fog_sun_dir = { ...sun_dir, sun_dir == (Vector3) { 0, 0, 0 } ? 0 : f.sun_sharpness };
        p.fog_sun_color = { ...color_to_linear(sun_color), f.sun_shafts };
    }
    return p;
}

// ---- Material UVs, displacement, and draw helpers

// Ensures `t` is uploaded and ready to bind, falling back to the 1x1 white texture when empty.
fn Texture* ready(Texture* t) @private {
    if (!t || (!t.uploaded && !t.source && t.width <= 0)) t = state.white;
    t.upload();
    return t;
}

// Primary texture providing the base UV transform for a material (`map`, `normal_map`, or `arm_map`).
fn Texture* uv_base(Material* mat) @local => mat.map ?: (mat.normal_map ?: mat.arm_map);

// Computes a 2x3 affine UV transform matrix (repeat, rotation around `center`, offset) as two Vec4 rows.
fn Vec4[2] uv_rows(Texture* t) @local {
    if (!t) return { { 1, 0, 0, 0 }, { 0, 1, 0, 0 } };
    float c   = math::cos(t.rotation);
    float s   = math::sin(t.rotation);
    Vector2 k = t.center;
    return {
        { t.repeat.x * c, t.repeat.x * s, -t.repeat.x * (c * k.x + s * k.y) + k.x + t.offset.x, 0 },
        { -t.repeat.y * s, t.repeat.y * c, -t.repeat.y * (-s * k.x + c * k.y) + k.y + t.offset.y, 0 },
    };
}

fn Vec4[2] uv_transform(Material* mat) @local => uv_rows(uv_base(mat));

// Packs vertex displacement scale, bias, and active flag (`z == 1` when CPU heights are ready).
fn Vec4 displacement_params(Material* mat) @local {
    Texture* t = mat.displacement_map;
    bool read  = t && t.keep_pixels && t.preparing.load() == 2;
    return { mat.displacement_scale, mat.displacement_bias, read ? 1 : 0, 0 };
}

// Samples `mat.displacement_map` at each vertex UV into a cached per-mesh float vertex buffer,
// rebuilding when geometry upload version, map pointer, or UV transform changes.
fn SgBuffer displacement_heights(Mesh* m) @local {
    Texture* t        = m.material.displacement_map;
    BufferGeometry* g = m.geometry;
    uint n            = g.vertex_count();
    if (!t) return zeros(n);
    if (!t.keep_pixels) {
        Texture* image = t.source ?: t;
        if (image.preparing.load() == 2) {
            if (!image.keep_pixels) io::eprintfn(
                "drie: a displacement map was uploaded before it was one: it has no pixels to read"
            );
        }
        t.keep_pixels = true;
        image.keep_pixels = true;
    }
    Texture* image = t.source ?: t;
    if (image.preparing.load() != 2) image.prepare();
    Vec4[2] rows = uv_rows(t);
    ulong key    = (ulong)(uptr)t * 31 + g.upload_version;
    for (int i = 0; i < 2; i++) {
        for (int k = 0; k < 3; k++) key = key * 1000003 + bitcast(rows[i][k], uint);
    }
    if (m.displacement_buf.id && m.displacement_key == key) return m.displacement_buf;
    if (m.displacement_buf.id) sg::destroy_buffer(m.displacement_buf);
    float[] heights = mem::new_array(float, (sz)math::max(n, 1u));
    defer free(heights);
    for (uint v = 0; v < n; v++) {
        float u0   = g.vertices[(sz)v * FLOATS_PER_VERTEX + 6];
        float v0   = g.vertices[(sz)v * FLOATS_PER_VERTEX + 7];
        Vector2 at = { rows[0].x * u0 + rows[0].y * v0 + rows[0].z, rows[1].x * u0 + rows[1].y * v0 + rows[1].z };
        heights[v] = image.red_at(at);
    }
    m.displacement_buf = sg::make_buffer(
        &&{ .data = { .ptr = heights.ptr, .size = heights.len * float::size }, .label = "drie-displacement" }
    );
    m.displacement_key = key;
    return m.displacement_buf;
}

// Encodes per-slot wrap emulation (`slot_wrap` in shader): 0 when matching `shared`, else `1 + s + 3 * t`.
fn float wrap_code(Texture* t, Texture* shared) @local {
    if (t.wrap_s == shared.wrap_s && t.wrap_t == shared.wrap_t) return 0;
    return (float)(1 + t.wrap_s.ordinal + 3 * t.wrap_t.ordinal);
}

// Computes `t`'s UV transform relative to `base`'s UV transform (or direct UV1 transform when
// `t.channel == 1`, marked by `row0.w == 1`).
fn Vec4[2] uv_relative(Texture* base, Texture* t) @local {
    Vec4[2] m = uv_rows(t);
    if (t && t.channel == 1) {
        m[0].w = 1;
        return m;
    }
    Vec4[2] b = uv_rows(base);
    float det = b[0].x * b[1].y - b[0].y * b[1].x;
    if (math::abs(det) < 1e-12) return m;
    // Invert the 2x3 base UV transform and compose `m * inverse(b)`.
    float ia =  b[1].y / det;
    float ib = -b[0].y / det;
    float ic = -b[1].x / det;
    float id =  b[0].x / det;
    float ix = -(ia * b[0].z + ib * b[1].z);
    float iy = -(ic * b[0].z + id * b[1].z);
    return {
        { m[0].x * ia + m[0].y * ic, m[0].x * ib + m[0].y * id, m[0].x * ix + m[0].y * iy + m[0].z, 0 },
        { m[1].x * ia + m[1].y * ic, m[1].x * ib + m[1].y * id, m[1].x * ix + m[1].y * iy + m[1].z, 0 },
    };
}

// Growable zero-filled vertex attribute buffer (4 floats per vertex) for missing streams.
fn SgBuffer zeros(uint vertices) @private {
    if (state.zeros.id && state.zeros_vertices >= vertices) return state.zeros;
    if (state.zeros.id) sg::destroy_buffer(state.zeros);
    uint n       = math::max(vertices, 1024u, state.zeros_vertices * 2);
    float[] zero = mem::new_array(float, (sz)n * 4);
    defer free(zero);
    state.zeros = sg::make_buffer(
        &&{ .data = { .ptr = zero.ptr, .size = (usz)n * 4 * float::size }, .label = "drie-zeros" }
    );
    state.zeros_vertices = n;
    return state.zeros;
}

// Growable white RGB per-instance color buffer for instanced meshes without custom colors.
fn SgBuffer whites(uint instances) @private {
    if (state.whites.id && state.whites_count >= instances) return state.whites;
    if (state.whites.id) sg::destroy_buffer(state.whites);
    uint n        = math::max(instances, 1024u, state.whites_count * 2);
    float[] white = mem::new_array(float, (sz)n * 3);
    defer free(white);
    white[..] = 1;
    state.whites = sg::make_buffer(
        &&{ .data = { .ptr = white.ptr, .size = (usz)n * 3 * float::size }, .label = "drie-whites" }
    );
    state.whites_count = n;
    return state.whites;
}

// Uploads `m.instance_colors` if dirty, or returns the shared `whites` fallback buffer.
fn SgBuffer instance_colors(Mesh* m) @local {
    if (!m.instance_colors.len()) return whites((uint)math::max(m.instance_count, 1));
    if (!m.instance_colors_uploaded) {
        if (m.instance_color_buf.id) sg::destroy_buffer(m.instance_color_buf);
        m.instance_color_buf = sg::make_buffer(
            &&{
                .data = { .ptr = m.instance_colors.entries, .size = (usz)m.instance_colors.len() * float::size },
                .label = "drie-instance-colors",
            }
        );
        m.instance_colors_uploaded = true;
    }
    return m.instance_color_buf;
}

fn Vec4 sway_params(Material* mat) @local => { state.renderer.time, mat.sway, mat.sway_speed, 0 };

// True when `item` renders with line primitives (wireframe material or Line/LineSegments/LineLoop).
fn bool draws_lines(DrawItem* item) @private => item.mesh.material.wireframe || item.mesh.is_line();

// Selects the index buffer and element count for `item` (triangles, wireframe, or line topology).
fn SgBuffer index_buffer_of(DrawItem* item, bool lines, int* count) @local {
    Mesh* m           = item.mesh;
    BufferGeometry* g = m.geometry;
    switch {
        case m.draw_mode == LINE_STRIP || m.draw_mode == LINE_LOOP:
            g.upload_lines(m.draw_mode == LINE_LOOP);
            *count = (int)g.line_count;
            return g.line_ibuf;
        case m.draw_mode == LINE_SEGMENTS:
            // Drop any trailing unpaired vertex index.
            *count = (int)g.index_count & ~1;
            return g.ibuf;
        case lines:
            g.upload_wire();
            *count = (int)g.wire_count;
            return g.wire_ibuf;
        default:
            *count = (int)g.index_count;
            return g.ibuf;
    }
}

// Submits `sg::draw()` for `item` (using the GPU-culled draw count when `item.job > 0`) and
// returns the triangle count (or segment count for lines).
fn int draw_geometry(DrawItem* item, bool lines) @private {
    Mesh* m       = item.mesh;
    int instances = m.instance_data.len() > 0 ? m.instance_count : 1;
    if (item.job) instances = cull_draw_count(item.job - 1);
    int elements;
    (void)index_buffer_of(item, lines, &elements);
    int first;
    if (m.draw_count > 0 && !lines && m.draw_mode == TRIANGLES) {
        first = math::min(m.draw_start, elements);
        elements = math::min(m.draw_count, elements - first);
    }
    sg::draw(first, elements, instances);
    return elements / (m.is_line() ? 2 : 3) * instances;
}

fn float alpha_test_of(Material* mat) @local => state.renderer.prof_no_alpha_test ? 0 : mat.alpha_test;

// True when `item` requires fragment discard in the main pass (alpha test, LOD dither, or cull box).
fn bool has_holes(DrawItem* item) @private {
    Material* mat = item.mesh.material;
    if (mat.screen_target) return false;
    return alpha_test_of(mat) > 0 || item.fade_dir != 0 || state.renderer.cull_box;
}

// True when `item` should render through the depth prepass (has holes, tests/writes depth, and
// exceeds `Renderer.depth_prepass_pixels` estimated screen pixels per triangle).
fn bool prepassed(DrawItem* item) @private {
    if (draws_lines(item) || !has_holes(item)) return false;
    Material* mat = item.mesh.material;
    if (!mat.depth_test || !mat.depth_write) return false;
    if (state.prepass_focal <= 0) return true;
    Mesh* m       = item.mesh;
    int instances = m.instance_data.len() > 0 ? m.instance_count : 1;
    if (item.job) instances = cull_draw_count(item.job - 1);
    float triangles = (float)m.geometry.index_count / 3 * math::max(instances, 1);
    float pixels    = PI * item.cover * state.prepass_focal * state.prepass_focal;
    return pixels > state.renderer.depth_prepass_pixels * triangles;
}

// Builds `projection * view` translated to `origin` in double precision so vertices transform
// `(world - origin)` into clip space without float jitter far from world origin.
fn Mat4 relative_view_proj(Camera* camera, Vector3 origin) @private {
    Mat4 view = camera.matrix_world_inverse;
    for (int r = 0; r < 3; r++) {
        double t = (double)view[3][r];
        t += (double)view[0][r] * (double)origin.x + (double)view[1][r] * (double)origin.y
            + (double)view[2][r] * (double)origin.z;
        view[3][r] = (float)t;
    }
    return camera.projection_matrix.mul(view);
}

// Returns `m.matrix_world`, or a camera-oriented billboard matrix matching `m.billboard`.
fn Mat4 model_of(Mesh* m) @local {
    if (m.billboard == NO_BILLBOARD) return m.matrix_world;
    Mat4 w         = m.matrix_world;
    Vector3 at     = w.get_position();
    float sx       = ((Vector3) { w[0][0], w[0][1], w[0][2] }).length();
    float sy       = ((Vector3) { w[1][0], w[1][1], w[1][2] }).length();
    Mat4 c         = state.view_world;
    Vector3 right  = ((Vector3) { c[0][0], c[0][1], c[0][2] }).normalize();
    Vector3 up     = ((Vector3) { c[1][0], c[1][1], c[1][2] }).normalize();
    Vector3 toward = ((Vector3) { c[2][0], c[2][1], c[2][2] }).normalize();
    if (m.billboard == FACING && m.billboard_fixed_size) {
        // Scale by view depth so the sprite maintains constant screen-space size.
        float depth = math::max((at - state.view_origin).dot(-toward), 0f);
        sx *= depth;
        sy *= depth;
    }
    Vector3 ax;
    Vector3 ay;
    Vector3 az = toward;
    switch (m.billboard) {
        case UPRIGHT:
            Vector3 look = { state.view_origin.x - at.x, 0, state.view_origin.z - at.z };
            az = look.length() > 1e-6 ? look.normalize() : (Vector3) { 0, 0, 1 };
            ax = (Vector3) { az.z, 0, -az.x } * sx;
            ay = (Vector3) { 0, sy, 0 };
        case KEEP_Y:
            ax = right * sx;
            ay = { w[1][0], w[1][1], w[1][2] };
        case KEEP_X:
            ax = { w[0][0], w[0][1], w[0][2] };
            ay = up * sy;
        default:
            float s = math::sin(m.billboard_rotation);
            float k = math::cos(m.billboard_rotation);
            ax = (right * k + up * s) * sx;
            ay = (up * k - right * s) * sy;
    }
    Vector2 shift  = { m.billboard_center.x - 0.5f, m.billboard_center.y - 0.5f };
    Vector3 origin = at - ax * shift.x - ay * shift.y;
    Mat4 b         = mat4_identity();
    b[0] = { ax.x, ax.y, ax.z, 0 };
    b[1] = { ay.x, ay.y, ay.z, 0 };
    b[2] = { az.x, az.y, az.z, 0 };
    b[3] = { origin.x, origin.y, origin.z, 1 };
    return b;
}

// Populates vertex uniforms for `item` with camera-relative `view_proj` (pass `normals: false` to
// skip the normal-matrix inverse in depth-only passes) and updates per-mesh MRT motion history.
fn VsParams vs_params_of(DrawItem* item, Mat4 view_proj, bool normals = true) @private {
    Mesh* m       = item.mesh;
    Material* mat = m.material;
    Mat4 model    = model_of(m);
    VsParams vs = {
        .view_proj = view_proj,
        .view_origin = { ...state.view_origin, 0 },
        .model = model,
        .normal_mat = normals ? model.invert()
        .transpose() : model, .uv_transform = uv_transform(mat), .sway = sway_params(mat), };
    if (state.motion) {
        // Advance mesh transform history once per MRT motion frame.
        if (m.motion_frame != state.motion_frame) {
            m.motion_prev = m.motion_frame == state.motion_frame - 1 ? m.motion_cur : m.matrix_world;
            m.motion_cur = m.matrix_world;
            m.motion_frame = state.motion_frame;
        }
        vs.prev_view_proj = state.motion_view_proj;
        vs.prev_model = m.motion_prev;
        bool moved = m.motion_prev != m.motion_cur || (mat.sway > 0 && state.motion_prev_time != state.motion_time)
            || m.geometry.updated_now();
        vs.motion = { moved ? 2 : 1, state.motion_prev_time, state.motion_jitter.x, state.motion_jitter.y };
    }
    if (state.bake.on) vs.bake = { 1, depth_zero_to_one() ? 1 : 0, 0, 0 };
    Vec4 d = displacement_params(mat);
    vs.surface_vs = { mat.outline, d.x, d.y, d.z };
    return vs;
}

// ---- Material keys and texture slots

fn bool coated(Material* mat) @local => mat.clearcoat > 0 || mat.clearcoat_node.id || mat.clearcoat_roughness_node.id;

// Packs optional material maps (matching shader `MAP_*` role order) into the 6 shared `tex_ext0..5`
// texture slots, deduplicating shared images and reserving top slots for node and Reflector maps.
fn void material_slots(Material* mat, NodeBuild* nodes, Texture*[6]* slots, Vec4[6]* map_slots) @private {
    bool standard  = mat.type == STANDARD;
    bool refracts  = standard && mat.transmission > 0;
    Renderer* lens = state.renderer;
    bool defocus   = mat.defocus && lens.focus_map && lens.focus_aperture > 0;
    bool classic   = mat.type == BASIC || mat.type == LAMBERT || mat.type == PHONG;
    Texture*[22] roles = {
        standard && mat.ao_intensity != 0 ? mat.ao_map : null,
        standard ? mat.specular_color_map : null,
        standard ? mat.specular_intensity_map : null,
        refracts ? mat.transmission_map : null,
        refracts ? mat.thickness_map : null,
        standard && mat.iridescence > 0 ? mat.iridescence_map : null,
        standard && mat.iridescence > 0 ? mat.iridescence_thickness_map : null,
        standard && coated(mat) ? mat.clearcoat_map : null,
        standard && coated(mat) ? mat.clearcoat_roughness_map : null,
        standard && coated(mat) ? mat.clearcoat_normal_map : null,
        standard && mat.sheen > 0 ? mat.sheen_color_map : null,
        standard && mat.sheen > 0 ? mat.sheen_roughness_map : null,
        mat.alpha_map,
        mat.emissive_map,
        mat.type != NORMAL ? mat.light_map : null,
        standard && mat.anisotropy > 0 ? mat.anisotropy_map : null,
        mat.custom_map,
        defocus ? lens.focus_map : null,
        classic ? mat.specular_map : null,
        mat.type == MATCAP ? mat.matcap : null,
        mat.type == TOON ? mat.gradient_map : null,
        mat.type != BASIC && mat.type != MATCAP ? mat.bump_map : null,
    };
    int used;
    // Reserve upper `tex_ext` slots for node-material textures and any active Reflector texture.
    int room          = 6 - (nodes ? nodes.texture_count : 0);
    Reflector* mirror = mirror_of(mat);
    if (mirror) room--;
    foreach (i, t : roles) {
        int slot = -1;
        if (t) {
            for (int k = 0; k < used; k++) {
                if ((*slots)[k] == t) slot = k;
            }
            if (slot < 0 && used < room) {
                (*slots)[used] = t;
                slot = used++;
            }
        }
        (*map_slots)[i / 4][i % 4] = (float)slot;
    }
    if (nodes) {
        for (int k = 0; k < nodes.texture_count; k++) (*slots)[5 - k] = nodes.texture(k);
    }
    if (mirror) (*slots)[reflector_slot(nodes ? nodes.texture_count : 0)] = mirror.target.as_map();
}

// Returns `mat.reflector` when rendered and ready to bind, or null.
fn Reflector* mirror_of(Material* mat) @local {
    return mat.reflector && mat.reflector.rendered && mat.reflector.target ? mat.reflector : null;
}

// Shared `tex_ext0..5` slot index reserved for a Reflector's planar mirror texture, immediately
// below the `node_textures` slots claimed by node materials (slot 21 holds `SpotLight.map`).
fn int reflector_slot(int node_textures) => 5 - node_textures;

// Builds the `MaterialKey` for a shadow-pass shader (`plain: true` omits node graphs during warmup).
fn void shadow_key_of(DrawItem* item, MaterialKey* key, bool plain = false) @local {
    Mesh* m       = item.mesh;
    Material* mat = m.material;
    m.geometry.upload_morph();
    m.upload_morph_weights();
    NodeBuild* build = plain ? null : node_build_of(mat);
    // A node graph's color step can cut alpha holes without a texture map.
    bool node_cuts   = build && build.cuts;
    bool cut         = (mat.map || mat.alpha_map || node_cuts) && mat.alpha_test > 0
        && !state.renderer.prof_no_alpha_test;
    *key = {
        .shadow = true,
        .nodes = build != null,
        .morph = m.geometry.morph_view.id != 0 && m.morph_weights_view.id != 0,
        .foliage = build && mat.foliage,
        .gpu_cull = item.job > 0 && cull.active,
        .displacement = displacement_params(mat).z > 0.5f,
        .sway = mat.sway > 0,
        .alpha_test = cut,
        .map = cut && mat.map,
    };
    if (cut && mat.alpha_map) key.set_slot(MAP_ALPHA, 1);
}

// Builds the full `MaterialKey` describing every shader feature required for `item` in this render.
fn void material_key_of(
    DrawItem* item,
    DrawKind kind,
    FsParams* fs,
    Vec4[6]* map_slots,
    MaterialKey* key,
    bool gbuffer = false,
    bool plain = false
) @private {
    Mesh* m       = item.mesh;
    Material* mat = m.material;
    Renderer* r   = state.renderer;
    m.geometry.upload_morph();
    m.upload_morph_weights();
    MaterialType type = mat.type;
    bool lit          = type == LAMBERT || type == PHONG || type == STANDARD || type == TOON;
    bool standard     = type == STANDARD;
    float alpha_test  = alpha_test_of(mat);
    int fog           = mat.fog ? (int)fs.fog_params.w : 0;
    bool fog_volume   = fog && fs.fog_grid.w > 0;
    // Analytic sun light shafts apply in exponential fog when volumetric fog is inactive.
    bool shafts       = !fog_volume && fog == 3 && fs.fog_sun_dir.w > 0 && fs.fog_sun_color.w > 0
        && fs.dir_shadow_params.x > 0.5f;
    Texture* env_map  = mat.env_map;
    Scene* scene      = state.scene;
    if (!env_map && scene && scene.environment && standard) env_map = scene.environment;
    // Trigger upload so `has_env_sh` / `env_light` is populated before keying the shader.
    if (env_map) (void)ready(env_map);
    bool probes = !mat.env_map && mat.env_probes && state.probes_view.id;
    if (probes) env_map = null;
    bool backdrop    = m == state.background;
    bool env_combine = mat.env_map && !backdrop && (type == BASIC || type == LAMBERT || type == PHONG);
    bool env_reflect = (env_map || probes) && (mat.glass || (type != LAMBERT && !backdrop && !env_combine));

    RenderTarget* rt = r.render_target;
    bool cut         = kind == CUT;
    bool prepassed   = kind == PRETESTED || kind == DEPTH;
    bool holes       = cut || prepassed;
    bool vc          = mat.vertex_colors;
    bool shadows     = lit && m.receive_shadow && (lights_gpu.shadows.len() > 0 || fs.dir_shadow_params.x > 0.5f);
    bool linear      = fs.output_params.x > 0.5f;
    *key = {
        .type = (char)type,
        .mrt = rt && rt.mrt,
        .cut = cut,
        .cull_box = cut && r.cull_box,
        .dash = cut && m.is_line() && mat.dash_size > 0 && mat.gap_size > 0,
        .dither = holes && item.fade_dir != 0,
        .screen = mat.screen_target != null,
        .tint = vc && mat.vertex_light <= 0,
        .vertex_alpha = vc,
        .foliage = mat.foliage,
        .map = mat.map != null,
        .alpha_test = holes && alpha_test > 0,
        .prepassed = prepassed,
        .background = backdrop,
        .env_combine = env_combine,
        .vertex_occlusion = mat.vertex_occlusion && vc,
        .premultiplied = mat.premultiplied_alpha,
        .fog = (char)fog,
        .fog_box = fog && fs.fog_box_min.w >= 0.5f,
        .fog_height = fog && fs.fog_height.y > 0,
        .fog_sun = fog && fs.fog_sun_dir.w > 0,
        .display_fog = fog && !fog_volume && fs.fog_height.z > 0.5f,
        .fog_volume = fog_volume,
        .fog_glow = fog && !fog_volume && fs.fog_color.w > 0,
        .fog_shafts = shafts,
        .linear = linear,
        .tone = linear || !mat.tone_mapped ? 0 : (char)fs.output_params.z,
        .display = linear ? 0 : (char)fs.output_params.w,
        .gpu_cull = item.job > 0 && cull.active,
        .displacement = displacement_params(mat).z > 0.5f,
        .sway = mat.sway > 0,
        .outline = mat.outline > 0,
        .motion = rt && rt.mrt && state.motion,
        .dir_lights = lit && fs.counts.x > 0,
        .point_lights = lit && fs.counts.y > 0,
        .spot_lights = lit && fs.counts.z > 0,
        .channels = lit && mat.light_channels != 0,
        .map_emissive = lit
            && mat.map_emissive, // Material.vertex_light reads vertex colors even when `vertex_colors` is false.
        .baked = lit && mat.vertex_light > 0,
        .normal_map = type != BASIC
            && mat.normal_map != null, // gen_bake_pipeline() omits the tangent vertex attribute.
        .tangents = type != BASIC && !state.bake.on && uses_tangents(item),
        .depth_packing = type == DEPTH ? (char)mat.depth_packing.ordinal : 0,
        .env_light = standard && env_map && env_map.has_env_sh,
        .env_reflect = env_reflect,
        .glass = mat.glass,
        .defocus = mat.defocus && r.focus_map && r.focus_aperture > 0,
        .hex = mat.hex_tiling > 0 && mat.alpha_test <= 0,
        .probes = probes,
        .gi = lit
            && state.gi_count > 0, // Lightmap baking reuses `ssr_color` for the previous lightmap accumulation texture.
        .ssr = !state.bake.on && !mat.glass && r.ssr_color.id && r.ssr_depth.id && mat.ssr > 0,
        .reflector = !mat.glass && mat.reflector && mat.reflector.rendered && mat.reflector.target,
        .rc = lit && fs.rc_size.w > 0,
        .env_box = env_reflect && env_map && mat.env_box_min != mat.env_box_max,
        .toon_nearest = type == TOON && mat.gradient_map && mat.gradient_map.nearest,
        .bake = state.bake.on,
        .nodes = !plain && draw_build(mat) != null,
        .morph = m.geometry.morph_view.id != 0 && m.morph_weights_view.id != 0,
        .flat_shading = mat.flat_shading,
        .shadows = shadows,
        .vsm = (shadows || shafts) && vsm_on(),
        .soft_shadows = shadows && lights_gpu.soft_shadows,
        .sun_shadow = shadows && fs.dir_shadow_params.x > 0.5f,
        .dir_tiles = shadows && lights_gpu.dir_tiles,
        .point_shadows = shadows && lights_gpu.point_shadows,
        .spot_shadows = shadows && lights_gpu.spot_shadows,
        .debug_cascades = lit && fs.dir_cascade_params.z > 0.5f && fs.dir_shadow_params.x > 0.5f,
        .sized = standard && lights_gpu.sized,
        .tubes = lit && lights_gpu.tubes,
        .rect_lights = standard && fs.counts.w > 0,
        .parallax = standard && mat.parallax_map && mat.parallax_scale > 0,
        .clearcoat = standard && coated(mat),
        .sheen = standard && mat.sheen > 0
            && (mat.sheen_color.x > 0 || mat.sheen_color.y > 0 || mat.sheen_color.z > 0),
        .iridescence = standard && mat.iridescence > 0,
        .anisotropy = standard && mat.anisotropy > 0,
        .transmission = standard && !mat.glass && mat.transmission > 0 && state.transmission_mips.id != 0,
        .arm_ao = (bool)mat.arm_map,
        .arm_rough = mat.arm_map || mat.roughness_map,
        .arm_metal = mat.arm_map
            || (mat.metalness_map && (mat.roughness_map ?: mat.metalness_map) == mat.metalness_map),
    };
    for (int role = 0; role < MAP_ROLES; role++) {
        float slot = (*map_slots)[role / 4][role % 4];
        if (slot >= 0) key.set_slot(role, (int)slot + 1);
    }
    // Specular map only affects env_combine strength and Phong highlights.
    if (!key.env_combine && type != PHONG) key.set_slot(MAP_SPECULAR, 0);
    if (gbuffer) *key = gbuffer_key(*key, state.motion);
}

// Strips lighting, fog, and output flags from `full` to form the G-Buffer MaterialKey.
fn MaterialKey gbuffer_key(MaterialKey full, bool motion) @local {
    MaterialKey k = {
        .gbuffer = true,
        .type = full.type,
        .cut = full.cut,
        .cull_box = full.cull_box,
        .dither = full.dither,
        .tint = full.tint,
        .vertex_alpha = full.vertex_alpha,
        .foliage = full.foliage,
        .map = full.map,
        .alpha_test = full.alpha_test,
        .vertex_occlusion = full.vertex_occlusion,
        .gpu_cull = full.gpu_cull,
        .displacement = full.displacement,
        .sway = full.sway,
        .outline = full.outline,
        .motion = motion,
        .map_emissive = full.map_emissive,
        .baked = full.baked,
        .normal_map = full.normal_map,
        .tangents = full.tangents,
        .arm_ao = full.arm_ao,
        .arm_rough = full.arm_rough,
        .arm_metal = full.arm_metal,
        .parallax = full.parallax,
        .hex = full.hex,
        .defocus = full.defocus,
        .nodes = full.nodes,
        .morph = full.morph,
        .flat_shading = full.flat_shading,
    };
    KeyWords w = bitcast(k, KeyWords);
    w.hi = bitcast(full, KeyWords).hi;
    return bitcast(w, MaterialKey);
}

// ---- Draw submission

// Renders a custom ShaderMaterial with its own vertex/fragment shaders, texture slots, and uniforms.
fn void draw_shader_material(DrawItem* item, Mat4 view_proj, DrawKind kind) @local {
    if (kind == DEPTH || state.bake.on) return;
    Mesh* m       = item.mesh;
    Material* mat = m.material;
    if (!mat.custom_shader) abort("drie: ShaderMaterial has no custom_shader");
    BufferGeometry* g   = m.geometry;
    bool lines          = draws_lines(item);
    int elements;
    SgBuffer indices    = index_buffer_of(item, lines, &elements);
    Renderer* r         = state.renderer;
    RenderTarget* rt    = r.render_target;
    bool mrt            = rt && rt.mrt;
    bool use_mrt_shader = mrt && mat.custom_shader_mrt != null;
    SgShader* sh        = use_mrt_shader ? &mat.custom_sg_shader_mrt : &mat.custom_sg_shader;
    if (!sh.id) {
        ShaderDescFn f     = use_mrt_shader ? mat.custom_shader_mrt : mat.custom_shader;
        SgShaderDesc sdesc = f(sg::query_backend());
        *sh = make_program(sdesc);
        if (!sh.id) abort("drie: ShaderMaterial failed to create shader");
    }
    int target  = rt ? (rt.samples > 1 ? 2 : 1) + (rt.mrt ? 4 : (rt.is_linear() ? 2 : 0)) : 0;
    ulong blend = blend_key(mat);
    int depth   = (mat.depth_write ? 0 : 1) + (mat.depth_test ? 0 : 2);
    ulong key   = blend | (ulong)depth << 28 | (ulong)mat.side.ordinal << 33 | (ulong)(lines ? 1
        : 0) << 35 | (ulong)target << 36 | (ulong)(mat.transparent ? 1
            : 0) << 40 | (ulong)(use_mrt_shader ? 1 : 0) << 41;
    if (!mat.custom_pipelines.is_initialized()) mat.custom_pipelines.init(mem);
    SgPipeline pip;
    if (try found = mat.custom_pipelines[key]) {
        pip = found;
    } else {
        SgCullMode cull;
        switch (mat.side) {
            case FRONT_SIDE: cull = BACK;
            case BACK_SIDE: cull = FRONT;
            case DOUBLE_SIDE: cull = NONE;
        }
        SgPipelineDesc desc = {
            .shader = *sh,
            .index_type = UINT32,
            .primitive_type = lines ? LINES : TRIANGLES,
            .cull_mode = lines ? NONE : cull,
            .face_winding = CCW,
            .layout = {
                .buffers[0].stride = FLOATS_PER_VERTEX * float::size,
                .attrs[0] = { .format = FLOAT3, .offset = 0 },
                .attrs[1] = { .format = FLOAT3, .offset = 12 },
                .attrs[2] = { .format = FLOAT2, .offset = 24 },
                .attrs[3] = { .format = FLOAT4, .offset = 32 },
            },
            .depth = {
                .write_enabled = !mat.transparent && mat.depth_write,
                .compare = mat.depth_test ? SgCompareFunc.LESS_EQUAL : SgCompareFunc.ALWAYS,
            },
            .label = "drie-shader-material",
        };
        if (rt) {
            desc.sample_count = rt.samples;
            desc.colors[0].pixel_format = rt.pixel_format();
            desc.depth.pixel_format = DEPTH;
        }
        if (mrt) {
            desc.color_count = 1 + MRT_COUNT;
            for (int i = 0; i < MRT_COUNT; i++) {
                desc.colors[i + 1].pixel_format = mrt_format(i);
                if (!use_mrt_shader || mat.transparent) desc.colors[i + 1].write_mask = NONE;
            }
        }
        desc.colors[0].blend = blend_state(blend);
        pip = sg::make_pipeline(&desc);
        if (!pip.id) abort("drie: ShaderMaterial failed to create pipeline");
        mat.custom_pipelines[key] = pip;
    }
    sg::apply_pipeline(pip);
    SgBindings bind = {
        .vertex_buffers[0] = g.vbuf,
        .index_buffer = indices,
    };
    for (int k = 0; k < 6; k++) {
        Texture* t = mat.custom_textures[k];
        if (!t && k == 0) t = mat.map;
        if (t) {
            Texture* rt_tex = ready(t);
            bind.views[k] = rt_tex.view;
            bind.samplers[k] = rt_tex.sampler;
        }
    }
    if (mat.custom_bind) mat.custom_bind(mat, m, &bind);
    sg::apply_bindings(&bind);
    Mat4 model = m.matrix_world;
    ShaderMaterialVsParams vs = {
        .view_proj = view_proj,
        .view_origin = { ...state.view_origin, 0 },
        .model = model,
        .normal_mat = model.invert()
        .transpose(), .camera_time = { ...state.node_camera, r.time }, };
    sg::apply_uniforms(0, &&{ .ptr = &vs, .size = ShaderMaterialVsParams::size });
    if (mat.custom_vs_uniforms && mat.custom_vs_uniforms_size > 0) {
        sg::apply_uniforms(mat.custom_vs_ub, &&{ .ptr = mat.custom_vs_uniforms, .size = mat.custom_vs_uniforms_size });
    }
    if (mat.custom_fs_uniforms && mat.custom_fs_uniforms_size > 0) {
        sg::apply_uniforms(mat.custom_fs_ub, &&{ .ptr = mat.custom_fs_uniforms, .size = mat.custom_fs_uniforms_size });
    }
    int instances = m.instance_data.len() ? (int)m.instance_data.len() : 1;
    sg::draw(0, elements, instances);
    r.info.calls++;
    r.info.triangles += (lines ? elements / 2 : elements / 3) * instances;
}

// Submits a single mesh draw into the active forward, depth-prepass, G-Buffer, or bake pass.
fn void draw_item(
    DrawItem* item,
    Mat4 view_proj,
    FsParams* fs,
    DrawKind kind = CUT,
    bool deferred_gb = false
) @private {
    Mesh* m       = item.mesh;
    Material* mat = m.material;
    if (mat.type == SHADER) {
        draw_shader_material(item, view_proj, kind);
        return;
    }
    BufferGeometry* g = m.geometry;
    bool lines        = draws_lines(item);
    int elements;
    SgBuffer indices  = index_buffer_of(item, lines, &elements);
    Renderer* r       = state.renderer;

    // 1. Shader program and pipeline lookup (falling back to plain material while compiling).
    Texture* map = ready(mat.map);
    if (mat.screen_target) map = state.white;
    float alpha_test = alpha_test_of(mat);
    NodeBuild* build = draw_build(mat);
    MaterialKey key;
    Texture*[6] slots;
    Vec4[6] map_slots;
    material_slots(mat, null, &slots, &map_slots);
    material_key_of(item, kind, fs, &map_slots, &key, deferred_gb);
    MaterialProgram* gen = material_program(key, build);
    NodeBuild* gen_build = gen.key.nodes ? build : null;
    SgPipeline pip;
    for (int pick = 0; pick < 2; pick++) {
        // Retry with the plain material program when a node graph is still compiling.
        if (pick == 1) {
            material_key_of(item, kind, fs, &map_slots, &key, deferred_gb, plain: true);
            gen = material_program(key);
            gen_build = null;
            build = null;
        } else if (gen_build && material_busy(gen_build, gen)) {
            note_waiting(m, 2);
            continue;
        }
        pip = state.bake.on ? gen_bake_pipeline(gen, state.bake.format)
            : get_pipeline(
                mat.side,
                mat.transparent,
                lines,
                r.render_target,
                alpha_test > 0,
                blend_key(mat),
                kind,
                mat.depth_test,
                mat.depth_write,
                deferred_gb,
                gen
            );
        if (pip.id) break;
        note_waiting(m, 3);
    }
    if (!pip.id) return;
    ShaderProgram* own = kind == DEPTH ? gen.depth : gen.program;
    sg::apply_pipeline(pip);

    // 2. Vertex, texture, shadow, GI, probe, and storage-buffer bindings.
    // Signal already-pretested cutouts via a negative alpha_test threshold.
    if (kind == PRETESTED && alpha_test > 0) alpha_test = -1;
    SgBindings bind = {
        .vertex_buffers[0] = g.vbuf,
        .index_buffer = indices,
    };
    bind_instances(&bind, item);
    Vec4 inst_cull = cull_bind(&bind, item.job - 1);
    bind.views[shaders::VIEW_TEX_MAP] = map.view;
    bind.views[shaders::VIEW_TEX_NORMAL] = ready(mat.normal_map).view;
    // Pack separate roughness/metalness maps into the ARM slot with bitmask `arm_bits`.
    Texture* arm = mat.arm_map ?: (mat.roughness_map ?: mat.metalness_map);
    int arm_bits;
    if (mat.arm_map) {
        arm_bits = 7;
    } else if (arm) {
        arm_bits = (mat.roughness_map ? 2 : 0) | (mat.metalness_map == arm ? 4 : 0);
    }
    bind.views[shaders::VIEW_TEX_ARM] = ready(arm).view;
    bool standard  = mat.type == STANDARD;
    Renderer* lens = state.renderer;
    bool defocus   = mat.defocus && lens.focus_map && lens.focus_aperture > 0;
    bool classic   = mat.type == BASIC || mat.type == LAMBERT || mat.type == PHONG;
    for (int k = 0; k < slots.len; k++) bind.views[shaders::VIEW_TEX_EXT0 + k] = ready(slots[k]).view;
    bool backdrop = m == state.background;
    int ext_bits  = backdrop ? 32 : 0;
    if (mat.refraction_model == PLANE) ext_bits |= 64;
    bind.views[shaders::VIEW_TEX_PARALLAX] = ready(mat.parallax_map).view;
    bool vsm = vsm_on();
    if (vsm) {
        bind.views[shaders::VIEW_VSM_MAP] = state.vsm_spot.view;
        bind.views[shaders::VIEW_VSM_ATLAS] = state.vsm_atlas.view;
        bind.views[shaders::VIEW_VSM_SUN] = state.vsm_sun.view;
    } else {
        bind.views[shaders::VIEW_SHADOW_MAP] = state.shadow_view;
        bind.views[shaders::VIEW_SHADOW_ATLAS] = state.atlas_view;
        bind.views[shaders::VIEW_SUN_SHADOW] = state.sun_view;
    }
    bind.views[shaders::VIEW_FOG_VOLUME] = state.fog_view;
    bool ssr = r.ssr_color.id != 0 && r.ssr_depth.id != 0 && mat.ssr > 0;
    bind.views[shaders::VIEW_SSR_COLOR] = ssr ? r.ssr_color : ready(state.white).view;
    // Lightmap baking binds the previous accumulation target in `VIEW_SSR_COLOR`.
    if (state.bake.on) bind.views[shaders::VIEW_SSR_COLOR] = state.bake.prev;
    bind.views[shaders::VIEW_SSR_DEPTH] = ssr ? r.ssr_depth : ready(state.white).view;
    bind.views[shaders::VIEW_SSR_BACK] = ssr && r.ssr_back.id ? r.ssr_back : ready(state.white).view;
    Reflector* mirror = mirror_of(mat);
    // SpotLight.map projection texture (`VIEW_LIGHT_COOKIES`).
    bind.views[shaders::VIEW_LIGHT_COOKIES] = lights_gpu.cookie ? lights_gpu.cookie.view : ready(state.white).view;
    bind.samplers[shaders::SMP_SSR_SMP] = clamp_sampler();
    Texture* none     = ready(state.white);
    bool transmission = mat.transmission > 0 && mat.type == STANDARD && state.transmission_mips.id != 0;
    bind.views[shaders::VIEW_TEX_TRANSMISSION] = transmission ? state.transmission_mips_view : none.view;
    bind.samplers[shaders::SMP_TRANSMISSION_SMP] = transmission ? state.mip_sampler : none.sampler;
    bind.views[shaders::VIEW_TEX_GI] = state.gi_view;
    bind.views[shaders::VIEW_RC_PROBES] = state.radiance_view;
    bind.samplers[shaders::SMP_GI_SMP] = state.irradiance.sampler;
    // Scene environment applies to standard materials lacking an explicit `mat.env_map`.
    Texture* env_map    = mat.env_map;
    float env_intensity = mat.env_intensity > 0 ? mat.env_intensity : 1;
    bool env_diffuse    = false;
    Scene* scene        = state.scene;
    if (!env_map && scene && scene.environment && mat.type == STANDARD) {
        env_map = scene.environment;
        env_intensity *= scene.environment_intensity;
    }
    Texture* env = ready(env_map);
    if (env_map && env_map.has_env_sh && mat.type == STANDARD) {
        env_diffuse = true;
        // Pack extracted sun direction and radiance into `.w` of the first 6 SH coefficients.
        EnvLight* el = &env_map.env_light;
        float[6] sun = { el.sun_dir.x, el.sun_dir.y, el.sun_dir.z, el.sun.x, el.sun.y, el.sun.z };
        foreach (i, c : el.sh) fs.env_sh[i] = { ...c, i < 6 ? sun[i] : 0 };
    }
    bind.views[shaders::VIEW_TEX_ENV] = env.view;
    bind.samplers[shaders::SMP_ENV_SMP] = env.sampler;
    Texture*[NODE_TEXTURES] own_textures;
    if (build) {
        for (int k = 0; k < build.texture_count; k++) own_textures[k] = build.texture(k);
    }
    Texture* first  = slots[0] ?: own_textures[0];
    Texture* shared = mat.map ?: (mat.normal_map ?: (arm ?: (first ?: state.white)));
    // Prefer a repeating sampler when no color map is set so `slot_wrap` can clamp cleanly.
    if (!mat.map) {
        Texture*[8 + NODE_TEXTURES] maps;
        maps[:8] = (Texture*[8]) {
            mat.normal_map,
            arm,
            slots[0],
            slots[1],
            slots[2],
            slots[3],
            slots[4],
            slots[5]
        }[..];
        maps[8:NODE_TEXTURES] = own_textures[..];
        foreach (t : maps) {
            if (t && t.uploaded && t.wrap_s == REPEAT_WRAPPING && t.wrap_t == REPEAT_WRAPPING) {
                shared = t;
                break;
            }
        }
    }
    shared = ready(shared);
    bind.samplers[shaders::SMP_TEX_SMP] = shared.sampler;
    if (mat.screen_target) {
        bind.views[shaders::VIEW_TEX_MAP] = mat.screen_target.texture;
        shared = state.white;
        bind.samplers[shaders::SMP_TEX_SMP] = state.white.sampler;
    }
    if (!state.no_probes.id) {
        char[4] white = { 255, 255, 255, 255 };
        state.no_probes_image = sg::make_image(
            &&{ .type = ARRAY, .width = 1, .height = 1, .num_slices = 1, .data = { .ptr = &white, .size = 4 } }
        );
        state.no_probes = sg::make_view(&&{ .texture = { .image = state.no_probes_image } });
    }
    bool probes = !mat.env_map && mat.env_probes && state.probes_view.id;
    bind.views[shaders::VIEW_TEX_PROBES] = state.probes_view.id ? state.probes_view : state.no_probes;
    if (probes) bind.samplers[shaders::SMP_ENV_SMP] = state.probes_sampler;
    bind.samplers[shaders::SMP_SHADOW_SMP] = state.shadow_sampler;
    if (vsm) bind.samplers[shaders::SMP_VSM_SMP] = state.vsm_sampler;
    lights_gpu.bind(&bind);
    SgBindings own_bind = program_bindings(
        own,
        &bind,
        user_views(gen_build)[..],
        m.geometry.morph_view,
        m.morph_weights_view
    );
    sg::apply_bindings(&own_bind);

    // 3. Vertex uniforms.
    VsParams vs = vs_params_of(item, view_proj);
    vs.inst_cull = inst_cull;
    @pool() {
        apply_program_uniforms(own, VERTEX, &vs, fs, user: gen_build ? user_block(gen_build, m) : { });
    };

    // 4. Fragment uniforms.
    int flags = (mat.vertex_colors ? 1
        : 0) | (mat.foliage ? 2
            : 0) | (m.receive_shadow ? 4
                : 0) | (mat.glass ? 8
                    : 0) | (mat.vertex_light > 0 ? 16
                        : 0) | (mat.screen_target ? 32
                            : 0) | (mat.map_emissive ? 64
                                : 0) | (mat.vertex_occlusion && mat.vertex_colors ? 128
                                    : 0) | (mat.premultiplied_alpha ? 256
                                        : 0) | (mat.tone_mapped ? 0 : 512) | (mat.flat_shading ? 1024 : 0);
    fs.color = { ...color_to_linear(mat.color) * (mat.color_scale > 0 ? mat.color_scale : 1), mat.opacity };
    fs.emissive = { ...color_to_linear(mat.emissive) * mat.emissive_intensity, mat.vertex_light };
    if (state.bake.on) {
        fs.screen_params = { 1.0f / state.bake.width, 1.0f / state.bake.height, item.fade, item.fade_dir };
    } else if (r.render_target) {
        fs.screen_params = { 1.0f / r.render_target.width, 1.0f / r.render_target.height, item.fade, item.fade_dir };
    } else {
        fs.screen_params = { 1.0f / sapp::widthf(), 1.0f / sapp::heightf(), item.fade, item.fade_dir };
    }
    fs.specular = { ...color_to_linear(mat.specular), (float)flags };
    fs.mat_params = { (float)mat.type, mat.roughness, mat.metalness, mat.shininess };
    fs.tex_params = { mat.map ? 1 : 0, mat.normal_map ? 1 : 0, (float)arm_bits, alpha_test };
    bool parallax = mat.parallax_map && mat.type == STANDARD;
    fs.parallax_params = {
        parallax ? mat.parallax_scale : 0,
        mat.parallax_distance > 0 ? mat.parallax_distance : 30,
        mat.parallax_shadow, // Disable hex tiling on alpha-tested materials so prepass and shadows stay consistent.
        mat.alpha_test > 0 ? 0 : mat.hex_tiling
    };
    if (m.is_line() && mat.dash_size > 0 && mat.gap_size > 0) {
        fs.line_dash = { mat.dash_size, mat.dash_size + mat.gap_size, mat.dash_scale > 0 ? mat.dash_scale : 1, 1 };
    } else {
        fs.line_dash = { };
    }
    fs.env_params = { env_map ? 1 : 0, env_intensity, env_diffuse ? 1 : 0, env_map ? env_map.env_distance_range : 0 };
    if (probes) fs.env_params = { 2, mat.env_intensity > 0 ? mat.env_intensity : 1, 0, 0 };
    bool boxed = mat.env_box_min != mat.env_box_max;
    fs.env_pos = { ...mat.env_position, boxed ? 1 : 0 };
    fs.env_box_min = { ...mat.env_box_min, 0 };
    fs.env_box_max = { ...mat.env_box_max, 0 };
    fs.transmission_params = { mat.ior > 0 ? mat.ior : 1.5f, mat.thickness, mat.dispersion, mat.attenuation_distance };
    fs.transmission_color = { ...color_to_linear(mat.attenuation_color), transmission ? mat.transmission : 0 };
    fs.specular_params = { ...mat.specular_color, mat.specular_intensity };
    // Average world axis scale to convert material thickness into world meters.
    float scale = 1;
    if (transmission) {
        Mat4 w = m.matrix_world;
        scale = (((Vector3) {
            w[0][0],
            w[0][1],
            w[0][2]
        }).length() + ((Vector3) {
            w[1][0],
            w[1][1],
            w[1][2]
        }).length() + ((Vector3) { w[2][0], w[2][1], w[2][2] }).length()) / 3;
    }
    fs.ext_params = { (float)ext_bits, mat.ao_intensity, scale, (float)state.transmission_width };
    fs.map_slots = map_slots;
    fs.light_map_params = { mat.light_map_intensity, 0, 0, 0 };
    fs.surface_params = {
        mat.bump_scale,
        mat.gradient_map && mat.gradient_map.nearest ? 1 : 0,
        mat.depth_packing.ordinal,
        (float)mat.light_channels
    };
    if (classic && mat.env_map && !backdrop) {
        fs.env_combine = {
            mat.reflectivity,
            (float)mat.combine.ordinal,
            mat.env_refraction ? mat.refraction_ratio : 0,
            1
        };
    } else {
        fs.env_combine = { };
    }
    fs.focus_params = { defocus ? state.renderer.focus_aperture : 0, state.renderer.focus_maxblur, 0, 0 };
    Texture* base = uv_base(mat);
    Vec4[2] rel   = uv_relative(base, mat.normal_map);
    fs.uv_rel[0:2] = rel[..];
    rel = uv_relative(base, arm);
    fs.uv_rel[2:2] = rel[..];
    for (int k = 0; k < slots.len; k++) {
        rel = uv_relative(base, slots[k]);
        fs.uv_rel[4 + 2 * k:2] = rel[..];
    }
    // Encode per-slot wrap overrides when a slot's wrapping differs from `shared`.
    fs.uv_rel[1].w = wrap_code(ready(mat.normal_map), shared);
    fs.uv_rel[3].w = wrap_code(ready(arm), shared);
    for (int k = 0; k < slots.len; k++) {
        fs.uv_rel[5 + 2 * k].w = wrap_code(ready(slots[k]), shared);
    }
    fs.clearcoat_params = {
        standard ? mat.clearcoat : 0,
        mat.clearcoat_roughness,
        mat.clearcoat_normal_scale.x,
        mat.clearcoat_normal_scale.y
    };
    Vector3 sheen = standard ? mat.sheen_color * mat.sheen : (Vector3) { 0, 0, 0 };
    fs.sheen_params = { ...sheen, mat.sheen_roughness };
    fs.iridescence_params = {
        standard ? mat.iridescence : 0,
        mat.iridescence_ior,
        mat.iridescence_thickness_min,
        mat.iridescence_thickness_max
    };
    float anisotropy = standard && mat.anisotropy > 0 ? mat.anisotropy : 0;
    fs.tangent_params = {
        mat.normal_scale.x,
        mat.normal_scale.y,
        math::cos(mat.anisotropy_rotation) * anisotropy,
        math::sin(mat.anisotropy_rotation) * anisotropy
    };
    fs.ssr_view_proj = r.ssr_view_proj;
    fs.ssr_params = { ssr ? mat.ssr : 0, r.ssr_reach, (float)(r.ssr_frame % 64), depth_zero_to_one() ? 1 : 0 };
    fs.ssr_thickness = { ssr && r.ssr_back.id ? 1 : 0, mat.ssr_reach > 0 ? mat.ssr_reach : r.ssr_reach, 0, 0 };
    fs.quality_params = { r.shadow_sample_scale, r.ssr_steps, r.ssr_max_step, 0 };
    if (mirror) {
        fs.reflector_view_proj = mirror.view_proj;
        fs.reflector_params = { mirror.strength, mirror.distortion, mirror.blur, (float)reflector_slot(0) };
    } else {
        fs.reflector_params = { };
    }
    float fog_mode = fs.fog_params.w;
    if (!mat.fog) fs.fog_params.w = 0;
    @pool() {
        apply_program_uniforms(own, FRAGMENT, &vs, fs, user: gen_build ? user_block(gen_build, m) : { });
    };
    fs.fog_params.w = fog_mode;

    // 5. Issue draw call and update RenderInfo counters.
    r.info.calls++;
    int drawn = draw_geometry(item, lines);
    // Depth prepass draws increment call count only, not visible triangle/line counts.
    if (kind == DEPTH) return;
    if (m.is_line()) {
        r.info.lines += drawn;
    } else {
        r.info.triangles += drawn;
    }
}

// Submits a single shadow-caster draw with `light_vp` mapping `(world - origin)` into light clip space.
fn void draw_shadow_item(DrawItem* item, Mat4 light_vp, Vector3 origin = { 0, 0, 0 }) @private {
    Mesh* m          = item.mesh;
    Material* mat    = m.material;
    Texture* map     = ready(mat.map);
    NodeBuild* build = node_build_of(mat);
    MaterialKey key;
    shadow_key_of(item, &key);
    key.atlas_depth = atlas_drawing;
    MaterialProgram* gen = material_program(key, build);
    NodeBuild* gen_build = gen.key.nodes ? build : null;
    // Fall back to the plain shadow shader while a node material compiles; flag static tiles for retry.
    if (gen_build && (material_busy(gen_build, gen) || !gen_shadow_pipeline(gen).id)) {
        note_waiting(m, 2);
        shadow_key_of(item, &key, plain: true);
        key.atlas_depth = atlas_drawing;
        gen = material_program(key);
        gen_build = null;
        state.shadow_plain = true;
    }
    SgPipeline pip = gen_shadow_pipeline(gen);
    if (!pip.id) return;
    sg::apply_pipeline(pip);
    SgBindings bind = {
        .vertex_buffers[0] = m.geometry.vbuf,
        .index_buffer = m.geometry.ibuf,
    };
    bind_instances(&bind, item);
    Vec4 inst_cull = cull_bind(&bind, item.job - 1);
    bind.views[shaders::VIEW_S_MAP] = map.view;
    bind.samplers[shaders::SMP_S_SMP] = map.sampler;
    Texture* alpha_map = ready(mat.alpha_map);
    bind.views[shaders::VIEW_S_ALPHA] = alpha_map.view;
    bind.samplers[shaders::SMP_S_ALPHA_SMP] = alpha_map.sampler;
    SgBindings own = program_bindings(
        gen.program,
        &bind,
        user_views(gen_build)[..],
        m.geometry.morph_view,
        m.morph_weights_view
    );
    sg::apply_bindings(&own);
    Vec4 d = displacement_params(mat);
    VsParams pv = {
        .view_proj = light_vp,
        .view_origin = { ...origin, 0 },
        .model = m.matrix_world, // Only node shadow graphs read normals; skip matrix inverse for plain casters.
        .normal_mat = gen_build ? m.matrix_world.invert()
        .transpose() : m.matrix_world, .uv_transform = uv_transform(
            mat
        ), .sway = sway_params(mat), .inst_cull = inst_cull, .surface_vs = { 0, d.x, d.y, d.z }, };
    ShadowFsParams sfs = {
        .s_params = { key.alpha_test ? mat.alpha_test : 0, mat.opacity, atlas_near, atlas_far },
        .s_alpha_uv = uv_relative(uv_base(mat), mat.alpha_map),
    };
    @pool() {
        Vec4[] user = gen_build ? user_block(gen_build, m) : { };
        apply_program_uniforms(gen.program, VERTEX, &pv, null, user: user);
        apply_program_uniforms(gen.program, FRAGMENT, &pv, null, &sfs, user);
    };
    state.renderer.info.shadow_calls++;
    state.renderer.info.shadow_triangles += draw_geometry(item, false);
}

// Populates vertex buffers 1..6 (instances, uv1, tangents, displacement heights, instance colors,
// and previous-frame positions) on `bind` for `item`.
fn void bind_instances(SgBindings* bind, DrawItem* item) @private {
    Mesh* m           = item.mesh;
    BufferGeometry* g = m.geometry;
    bind.vertex_buffers[2] = g.uv1buf.id ? g.uv1buf : zeros(g.vertex_count());
    bind.vertex_buffers[3] = g.tangent_buf.id ? g.tangent_buf : zeros(g.vertex_count());
    bind.vertex_buffers[4] = displacement_heights(m);
    bind.vertex_buffers[5] = instance_colors(m);
    bind.vertex_buffers[6] = g.prev_vertices();
    if (!m.instance_data.len()) {
        bind.vertex_buffers[1] = state.identity_instance;
        return;
    }
    int offset;
    bind.vertex_buffers[1] = instance_vertex_buffer(m, &offset);
    bind.vertex_buffer_offsets[1] = offset;
}

fn int cmp_opaque(DrawItem a, DrawItem b) @private => a.depth < b.depth ? 1 : (a.depth > b.depth ? -1 : 0);
fn int cmp_transparent(DrawItem a, DrawItem b) @private {
    if (a.mesh.render_order != b.mesh.render_order) return a.mesh.render_order < b.mesh.render_order ? -1 : 1;
    return a.depth < b.depth ? -1 : (a.depth > b.depth ? 1 : 0);
}

// ---- Backgrounds and transmission passes

// Prepares the far-plane fog backdrop sphere around `camera` when light-scattering fog is active.
fn bool backdrop_item(Scene* scene, Camera* camera, DrawItem* item) @private {
    if (!scene.fog || scene.fog.light_scatter <= 0 || camera.camera_type != PERSPECTIVE) return false;
    if (state.renderer.prof_no_fog_scatter) return false;
    if (!state.backdrop) {
        MeshBasicMaterial* mat = mesh_basic_material();
        mat.side = BACK_SIDE;
        state.backdrop = mesh(sphere_geometry(1, 24, 12), mat);
        state.backdrop.receive_shadow = false;
    }
    Mesh* m = state.backdrop;
    m.material.color = scene.background ? scene.background : scene.fog.color;
    float r = ((PerspectiveCamera*)camera).far * 0.9f;
    m.matrix_world = mat4_compose(camera.matrix_world.get_position(), { 0, 0, 0 }, { r, r, r });
    m.geometry.upload();
    *item = { m, 0, 0, 0, 0, 1 };
    return true;
}

fn void draw_backdrop(Scene* scene, Camera* camera, Mat4 view_proj, FsParams* fs) @private {
    DrawItem item;
    if (backdrop_item(scene, camera, &item)) draw_item(&item, view_proj, fs, has_holes(&item) ? CUT : SOLID);
}

// Prepares the DrawItem for `scene.background_material` (sphere), flat `background_texture`
// (full-view quad without depth), or equirectangular `background_texture` (sky sphere).
fn bool background_item(Scene* scene, Camera* camera, DrawItem* item) @private {
    Mesh* m;
    if (scene.background_material) {
        if (!state.background_sphere) {
            state.background_sphere = mesh(sphere_geometry(1, 32, 16), scene.background_material);
            state.background_sphere.receive_shadow = false;
        }
        m = state.background_sphere;
        m.material = scene.background_material;
        float r = camera.far * 0.9f;
        m.matrix_world = mat4_compose(camera.matrix_world.get_position(), { 0, 0, 0 }, { r, r, r });
    } else if (scene.background_texture && scene.background_flat) {
        if (!state.background_quad) {
            MeshBasicMaterial* mat = mesh_basic_material(fog: false);
            mat.depth_test = false;
            mat.depth_write = false;
            state.background_quad = mesh(plane_geometry(2, 2), mat);
            state.background_quad.receive_shadow = false;
        }
        m = state.background_quad;
        Texture* t = scene.background_texture;
        m.material.map = t;
        m.material.tone_mapped = t.color_space != SRGB_COLOR_SPACE;
        float d = (camera.near + camera.far) * 0.5f;
        float cx;
        float cy;
        float hw;
        float hh;
        if (camera.camera_type == PERSPECTIVE) {
            PerspectiveCamera* pc = (PerspectiveCamera*)camera;
            hh = d * math::tan(pc.fov * (float)math::PI / 360.0f) / camera.zoom;
            hw = hh * pc.aspect;
        } else {
            OrthographicCamera* oc = (OrthographicCamera*)camera;
            cx = (oc.left + oc.right) * 0.5f;
            cy = (oc.top + oc.bottom) * 0.5f;
            hw = (oc.right - oc.left) * 0.5f / camera.zoom;
            hh = (oc.top - oc.bottom) * 0.5f / camera.zoom;
        }
        m.matrix_world = camera.matrix_world.mul(
            mat4_compose({ cx, cy, -d }, { 0, 0, 0 }, { hw * 1.01f, hh * 1.01f, 1 })
        );
    } else if (scene.background_texture && camera.camera_type == PERSPECTIVE) {
        if (!state.background) {
            MeshBasicMaterial* mat = mesh_basic_material(transparent: true, side: BACK_SIDE, fog: false);
            state.background = mesh(sphere_geometry(1, 32, 16), mat);
            state.background.receive_shadow = false;
        }
        m = state.background;
        m.material.env_map = scene.background_texture;
        m.material.env_intensity = scene.background_intensity;
        // Background blurriness maps onto material roughness in the background shader.
        m.material.roughness = math::clamp(scene.background_blurriness, 0, 1);
        float r = ((PerspectiveCamera*)camera).far * 0.9f;
        m.matrix_world = mat4_compose(camera.matrix_world.get_position(), { 0, 0, 0 }, { r, r, r });
    } else {
        return false;
    }
    m.geometry.upload();
    *item = { m, 0, 0, 0, 0, 1 };
    return true;
}

fn void draw_background(Scene* scene, Camera* camera, Mat4 view_proj, FsParams* fs) @private {
    DrawItem item;
    if (background_item(scene, camera, &item)) draw_item(&item, view_proj, fs);
}

// Allocates or resizes the mipmapped single-sampled RGBA16F transmission texture chain.
fn void transmission_levels(int width, int height) @private {
    if (state.transmission_mips.id && state.transmission_width == width && state.transmission_height == height) return;
    if (state.transmission_mips.id) {
        sg::destroy_view(state.transmission_mips_view);
        for (int i = 0; i < state.transmission_levels; i++) {
            sg::destroy_view(state.transmission_level_views[i]);
            sg::destroy_view(state.transmission_level_attachments[i]);
        }
        sg::destroy_image(state.transmission_mips);
    }
    int levels = 1;
    while (levels < 16 && (width >> levels > 0 || height >> levels > 0)) levels++;
    state.transmission_levels = levels;
    state.transmission_width = width;
    state.transmission_height = height;
    state.transmission_mips = sg::make_image(
        &&{
            .usage = { .color_attachment = true },
            .width = width,
            .height = height,
            .num_mipmaps = levels, // Force sample_count = 1 so mipmaps are valid regardless of window MSAA.
            .sample_count = 1,
            .pixel_format = RGBA16F,
            .label = "drie-transmission-mips",
        }
    );
    state.transmission_mips_view = sg::make_view(&&{ .texture = { .image = state.transmission_mips } });
    for (int i = 0; i < levels; i++) {
        state.transmission_level_views[i] = sg::make_view(
            &&{ .texture = { .image = state.transmission_mips, .mip_levels = { .base = i, .count = 1 } } }
        );
        state.transmission_level_attachments[i] = sg::make_view(
            &&{ .color_attachment = { .image = state.transmission_mips, .mip_level = i } }
        );
    }
    mip_pipeline();
}

// Lazily creates the linear-filtering blit pipeline and sampler used by fill_levels().
fn void mip_pipeline() @private {
    if (!state.mip_pipeline.id) {
        // Reuse the output pass shader with tone mapping and sRGB disabled as a bilinear blit.
        state.mip_pipeline = sg::make_pipeline(
            &&{
                .shader = make_program(postprocessing::output_shader_desc(sg::query_backend())),
                .colors[0].pixel_format = RGBA16F,
                .depth.pixel_format = NONE,
                .sample_count = 1,
                .label = "drie-transmission-mips",
            }
        );
        state.mip_sampler = sg::make_sampler(
            &&{
                .min_filter = LINEAR,
                .mag_filter = LINEAR,
                .mipmap_filter = LINEAR,
                .wrap_u = CLAMP_TO_EDGE,
                .wrap_v = CLAMP_TO_EDGE,
            }
        );
    }
}

fn void fill_transmission_levels(SgView source, int width, int height) @local {
    transmission_levels(width, height);
    fill_levels(
        source,
        state.transmission_level_views[:state.transmission_levels],
        state.transmission_level_attachments[:state.transmission_levels]
    );
}

// Generates a mipmap chain into `attachments` from `source`: level 0 copies `source`, and each
// subsequent level downsamples the previous level (`views[i - 1]`) via bilinear filtering.
fn void fill_levels(SgView source, SgView[] views, SgView[] attachments) @private {
    mip_pipeline();
    postprocessing::OutputParams p = { .params = { 0, 1, 0, 1 } };
    for (int i = 0; i < attachments.len; i++) {
        SgView level = i == 0 ? source : views[i - 1];
        profile::begin_pass(
            &&{
                .action.colors[0] = { .load_action = DONTCARE, .store_action = STORE },
                .attachments.colors[0] = attachments[i],
                .label = "drie-mip",
            }
        );
        sg::apply_pipeline(state.mip_pipeline);
        SgBindings bind;
        bind.views[postprocessing::VIEW_OUTPUT_SRC] = level;
        bind.views[postprocessing::VIEW_OUTPUT_EXPOSURE] = level;
        bind.views[postprocessing::VIEW_OUTPUT_LUT] = postprocessing::empty_lut();
        bind.samplers[postprocessing::SMP_OUTPUT_SMP] = state.mip_sampler;
        sg::apply_bindings(&bind);
        sg::apply_uniforms(
            postprocessing::UB_OUTPUT_PARAMS,
            &&{ .ptr = &p, .size = postprocessing::OutputParams::size }
        );
        sg::draw(0, 3, 1);
        sg::end_pass();
    }
}

// Renders background and non-transmissive opaque meshes into a linear RGBA16F target (scaled by
// `Renderer.transmission_resolution_scale`), builds mipmaps, and renders double-sided glass back
// faces when present. Used when the main target is non-linear (window or RGBA8).
fn void render_transmission(Renderer* r, Scene* scene, Camera* camera, Mat4 view_proj, FsParams* fs) @private {
    float scale = r.transmission_resolution_scale > 0 ? r.transmission_resolution_scale : 1;
    int width   = max((int)((r.render_target ? r.render_target.width : sapp::width()) * scale + 0.5f), 1);
    int height  = max((int)((r.render_target ? r.render_target.height : sapp::height()) * scale + 0.5f), 1);
    int samples = 1;
    if (!state.transmission) state.transmission = render_target(width, height, samples, RGBA16F);
    state.transmission.set_size(width, height, samples);

    RenderTarget* prev = r.render_target;
    r.render_target = state.transmission;
    FsParams tfs = *fs;
    tfs.output_params = { 1, fs.output_params.y, fs.output_params.z, 0 };
    Color background = scene.background ? scene.background : r.clear_color;
    Vector3 clear    = color_to_linear(background);
    // Warm up transmission-pass shaders in parallel before opening the pass.
    start_background_program(scene, camera, &tfs);
    for (int list = 0; list < 2; list++) {
        foreach (&item : list == 0 ? state.opaque : state.transparent) {
            Material* mat = item.mesh.material;
            if (list == 0 && !refracts(
                item
            )) start_item_program(item, mat.wireframe || has_holes(item) ? CUT : SOLID, &tfs);
            if (refracts(item) && mat.side == DOUBLE_SIDE) {
                mat.side = BACK_SIDE;
                start_item_program(item, list == 0 && !mat.wireframe && !has_holes(item) ? SOLID : CUT, &tfs);
                mat.side = DOUBLE_SIDE;
            }
        }
    }
    profile::begin_pass(
        &&{
            .action = {
                .colors[0] = {
                    .load_action = CLEAR,
                    .store_action = STORE,
                    .clear_value = { clear.x, clear.y, clear.z, 1 }
                },
                .depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 1 },
            },
            .attachments = state.transmission.attachments(),
            .label = "drie-transmission",
        }
    );
    draw_background(scene, camera, view_proj, &tfs);
    foreach (&item : state.opaque) {
        if (refracts(item)) continue;
        draw_item(item, view_proj, &tfs, item.mesh.material.wireframe || has_holes(item) ? CUT : SOLID);
    }
    sg::end_pass();
    fill_transmission_levels(state.transmission.texture, state.transmission.width, state.transmission.height);
    // Draw back faces of double-sided glass into the transmission buffer so front faces refract them.
    bool backs;
    for (int list = 0; list < 2 && !backs; list++) {
        foreach (&item : list == 0 ? state.opaque : state.transparent) {
            if (refracts(item) && item.mesh.material.side == DOUBLE_SIDE) backs = true;
        }
    }
    if (backs) {
        profile::begin_pass(
            &&{
                .action = {
                    .colors[0] = { .load_action = LOAD, .store_action = STORE },
                    .depth = { .load_action = LOAD, .store_action = DONTCARE },
                },
                .attachments = state.transmission.attachments(),
                .label = "drie-transmission-backs",
            }
        );
        for (int list = 0; list < 2; list++) {
            foreach (&item : list == 0 ? state.opaque : state.transparent) {
                Material* mat = item.mesh.material;
                if (!refracts(item) || mat.side != DOUBLE_SIDE) continue;
                mat.side = BACK_SIDE;
                draw_item(item, view_proj, &tfs, list == 0 && !mat.wireframe && !has_holes(item) ? SOLID : CUT);
                mat.side = DOUBLE_SIDE;
            }
        }
        sg::end_pass();
        fill_transmission_levels(state.transmission.texture, state.transmission.width, state.transmission.height);
    }
    r.render_target = prev;
}

// Closes the active linear main pass, builds transmission mipmaps from its resolved color buffer,
// and reopens the pass with LOAD actions so transmissive meshes can refract the opaque scene.
fn void copy_for_transmission(Renderer* r, SgPass* pass) @private {
    sg::end_pass();
    RenderTarget* rt = r.render_target;
    fill_transmission_levels(rt.texture, rt.width, rt.height);
    SgPass again = *pass;
    for (int i = 0; i <= (rt.mrt ? MRT_COUNT : 0); i++) again.action.colors[i].load_action = LOAD;
    again.action.depth.load_action = LOAD;
    again.label = "drie-main-glass";
    profile::begin_pass(&again);
}

// True when `item` has vertex tangents and uses a normal map, clearcoat normal map, or anisotropy.
fn bool uses_tangents(DrawItem* item) @local {
    Material* mat     = item.mesh.material;
    BufferGeometry* g = item.mesh.geometry;
    if ((usz)g.tangents.len() < (usz)g.vertex_count() * 4) return false;
    return mat.normal_map || (mat.type == STANDARD && (mat.clearcoat_normal_map || mat.anisotropy != 0));
}

// True when `item` uses physical transmission (`Material.transmission > 0` on a standard material).
fn bool refracts(DrawItem* item) @private => item.mesh.material.transmission > 0 && item.mesh.material.type == STANDARD;