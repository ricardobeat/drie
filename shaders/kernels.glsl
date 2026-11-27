// Renderer compute kernels (drie): light clustering (cluster_cs), GPU instance culling (cull_cs),
// and volumetric froxel fog (fog_cs). Regenerate generated/kernels.c3 with `just shaders`.
@header module drie::shaders;
@header import drie;

@ctype mat4 Mat4
@ctype vec4 Vec4

// Sorts the point and spot lights into the clusters (froxels) of a view, matching
// drie::cluster_lights_cpu: per cluster its first entry in the index list, its point lights, and
// its spot lights after them. The CPU finds the clusters each light's sphere touches (a box of
// them, `ranges`); this fills the grid in one group of 256: each thread counts the lights of its
// run of clusters, the runs are added up in order, and each thread writes its clusters' lists
// where they start.
@cs cluster_cs
layout(binding=0) uniform cluster_params {
    ivec4 cl_lights;  // x = point lights, y = spot lights (their ranges follow the points')
    ivec4 cl_grid;    // xyz = clusters across, up and in depth, w = all of them
};
struct light_range {
    ivec4 lo;  // xyz = the first cluster on each axis, w = 1: it reaches some
    ivec4 hi;  // xyz = the last
};
layout(binding=0) readonly buffer cluster_ranges { light_range ranges[]; };
struct gpu_cluster {
    uvec4 range;  // as gpu_light_cluster in shaders/shader_core.glsl
};
struct gpu_cluster_light {
    uint index;
};
layout(binding=1) buffer cluster_list { gpu_cluster cl_out[]; };
layout(binding=2) buffer cluster_light_list { gpu_cluster_light cl_index[]; };

layout(local_size_x=256) in;
shared uint cl_sums[256];

bool cl_inside(int i, ivec3 c) {
    light_range r = ranges[i];
    return r.lo.w != 0 && all(greaterThanEqual(c, r.lo.xyz)) && all(lessThanEqual(c, r.hi.xyz));
}

ivec3 cl_coord(int c) {
    return ivec3(c % cl_grid.x, (c / cl_grid.x) % cl_grid.y, c / (cl_grid.x * cl_grid.y));
}

void main() {
    int t = int(gl_LocalInvocationIndex);
    int per = (cl_grid.w + 255) / 256;
    int first = t * per;
    int last = min(first + per, cl_grid.w);
    int np = cl_lights.x;
    int ns = cl_lights.y;
    uint total = 0;
    for (int c = first; c < last; c++) {
        ivec3 at = cl_coord(c);
        uint points = 0;
        uint spots = 0;
        for (int i = 0; i < np; i++) if (cl_inside(i, at)) points++;
        for (int i = 0; i < ns; i++) if (cl_inside(np + i, at)) spots++;
        cl_out[c].range = uvec4(0, points, spots, spots);
        total += points + spots;
    }
    cl_sums[t] = total;
    barrier();
    for (int off = 1; off < 256; off *= 2) {
        uint v = t >= off ? cl_sums[t - off] : 0;
        barrier();
        cl_sums[t] += v;
        barrier();
    }
    uint at_entry = cl_sums[t] - total;
    for (int c = first; c < last; c++) {
        ivec3 at = cl_coord(c);
        cl_out[c].range.x = at_entry;
        for (int i = 0; i < np; i++) if (cl_inside(i, at)) cl_index[at_entry++].index = uint(i);
        for (int i = 0; i < ns; i++) if (cl_inside(np + i, at)) cl_index[at_entry++].index = uint(i);
    }
}
@end

// Instance culling: one group per job (an instanced mesh seen from one view). Tests each
// instance's bounding sphere against the view's planes and writes the ones inside, in order,
// to `visible` from the job's first entry, and how many there are to counts[job]. An instanced
// draw then reads its matrices through `visible` (shaders/shader_core.glsl, fetch_instance).
@cs cull_cs
struct cull_job {
    vec4 m0;      // the mesh's world matrix, rows
    vec4 m1;
    vec4 m2;
    vec4 sphere;  // the geometry's bounding sphere (local), w = radius
    vec4 extra;   // x = added to each world radius (sway), y = the largest scale of the world matrix
    uvec4 range;  // x = the mesh's first instance in `instances`, y = instances,
                  //   z = its first entry in `visible`, w = the view (planes from 6 * w)
};
layout(binding=0) readonly buffer cull_jobs { cull_job jobs[]; };
struct cull_plane {
    vec4 plane;
};
struct cull_index {
    uint index;
};
layout(binding=1) readonly buffer cull_planes { cull_plane planes[]; };
layout(binding=2) readonly buffer cull_instances { cull_plane rows[]; };
layout(binding=3) buffer cull_visible { cull_index visible[]; };
layout(binding=4) buffer cull_counts { cull_index counts[]; };

layout(local_size_x=256) in;
shared uint cull_scan[256];

bool cull_keep(cull_job job, uint i) {
    uint s = (job.range.x + i) * 3;
    vec4 r0 = rows[s].plane;
    vec4 r1 = rows[s + 1].plane;
    vec4 r2 = rows[s + 2].plane;
    vec4 c = vec4(job.sphere.xyz, 1.0);
    vec4 lc = vec4(dot(r0, c), dot(r1, c), dot(r2, c), 1.0);
    vec3 wc = vec3(dot(job.m0, lc), dot(job.m1, lc), dot(job.m2, lc));
    float scale = max(length(vec3(r0.x, r1.x, r2.x)),
                      max(length(vec3(r0.y, r1.y, r2.y)), length(vec3(r0.z, r1.z, r2.z))));
    float radius = job.sphere.w * scale * job.extra.y + job.extra.x;
    uint base = job.range.w * 6;
    for (uint k = 0; k < 6; k++) {
        vec4 p = planes[base + k].plane;
        if (dot(p.xyz, wc) + p.w < -radius) return false;
    }
    return true;
}

void main() {
    uint j = gl_WorkGroupID.x;
    uint t = gl_LocalInvocationIndex;
    cull_job job = jobs[j];
    uint count = job.range.y;
    uint total = 0;
    for (uint start = 0; start < count; start += 256) {
        uint i = start + t;
        bool keep = i < count && cull_keep(job, i);
        cull_scan[t] = keep ? 1 : 0;
        barrier();
        for (uint off = 1; off < 256; off *= 2) {
            uint v = t >= off ? cull_scan[t - off] : 0;
            barrier();
            cull_scan[t] += v;
            barrier();
        }
        if (keep) visible[job.range.z + total + cull_scan[t] - 1].index = job.range.x + i;
        total += cull_scan[255];
        barrier();
    }
    if (t == 0) counts[j].index = total;
}
@end


// Volumetric fog (Fog): marches every column of the froxel grid (a cell of the picture, cut into
// slices of view depth that grow exponentially) from the camera outwards, and writes per slice
// the light the fog scatters towards the camera up to the slice's far end (rgb) and how much of
// what lies there gets through (a). The slices are tiles of one 2D image, `vf_grid.w` across.
// Each slice is lit at one point, somewhere else in it every frame: by the sun through its
// cascades, by the spot lights through their shadow maps, and by the point lights, each with
// only the lights of the cluster the point is in (ClusteredLighting). The fog's density, height,
// and clear box are integrated over the slice; only the noise is evaluated at the sample point.
@cs fog_cs
layout(binding=0) uniform fog_cs_params {
    vec4 vf_cam;        // the camera's position, w = frame (for the jitter)
    vec4 vf_fwd;        // the camera's forward direction, w = noise amount
    vec4 vf_right;      // right * tan(half the horizontal view), w = noise scale (1/m)
    vec4 vf_up;         // up * tan(half the vertical view), w = time (s)
    vec4 vf_grid;       // x, y = cells across and up, z = slices, w = slices per row of tiles
    vec4 vf_depth;      // x = where slices start (m of view depth), y = slices per unit of
                        //   log(depth / x), z = where the last ends, w = 1 / tile's texel size
    vec4 vf_wind;       // xyz = noise drift (m/s)
    vec4 vf_color;      // the fog's own color (linear), w = light scatter
    vec4 vf_params;     // as fog_params in shaders/shader_core.glsl: near, far, density, mode
    vec4 vf_box_min;    // fog-free box, w = 1 when enabled
    vec4 vf_box_max;    // w = the part of the density left in the box
    vec4 vf_height;     // x = base height, y = falloff, w = density away from the sun's side
    vec4 vf_sun_dir;    // towards the sun, w = sharpness (0: no sun)
    vec4 vf_sun_color;  // w = light shafts
    vec4 vf_clusters;   // xyz = light clusters across, up, in depth; w = 1: there are lights
    vec4 vf_cluster_depth;  // as cluster_depth in shaders/shader_core.glsl
    mat4 vf_shadow_mat[4];  // as dir_shadow_mat...
    vec4 vf_shadow_params;  // x = the sun has a shadow map, z = texel size of it
    vec4 vf_cascade_rect[4];
    vec4 vf_cascade_info[4];
    vec4 vf_cascade_params; // x = cascades, w = 1: one box
    vec4 vf_cloud_shadow;   // the clouds' shadow map (Renderer.cloud_shadow): xz = its middle,
                            //   y = 1 / its span (m, 0: none), w = the altitude it lies at
};
layout(binding=0, rgba16f) uniform writeonly image2D vf_out;
layout(binding=1) uniform texture2D vf_spot_map;   // the spot light's map, drawn every frame
layout(binding=2) uniform texture2D vf_spot_atlas; // the maps drawn once
layout(binding=3) uniform texture2D vf_sun_map;    // the sun's cascades
layout(binding=9) uniform texture2D vf_cookies;    // SpotLight.map atlas (light_cookies in shaders/shader_core.glsl)
layout(binding=10) uniform texture2D vf_cloud_map; // the clouds' shadow (r: the sunlight they let through)
layout(binding=0) uniform sampler vf_shadow_smp;
layout(binding=1) uniform sampler vf_cookie_smp;

// the same layouts as in shaders/shader_core.glsl
struct vf_point_light {
    vec4 pos;
    vec4 color;
    vec4 params;
};
struct vf_spot_light {
    vec4 pos;
    vec4 dir;
    vec4 color;
    vec4 params;
    vec4 shadow;
    vec4 map;
    vec4 map_u;
    vec4 map_v;
};
struct vf_spot_shadow {
    mat4 mat;
    vec4 params;
    vec4 rect;
    vec4 soft;
};
struct vf_cluster {
    uvec4 range;
};
struct vf_index {
    uint index;
};
layout(binding=4) readonly buffer vf_points { vf_point_light points[]; };
layout(binding=5) readonly buffer vf_spots { vf_spot_light spots[]; };
layout(binding=6) readonly buffer vf_shadows { vf_spot_shadow spot_shadows[]; };
layout(binding=7) readonly buffer vf_light_clusters { vf_cluster clusters[]; };
layout(binding=8) readonly buffer vf_light_indices { vf_index cluster_lights[]; };

layout(local_size_x=8, local_size_y=8) in;

// Fraction of light that makes it through `d` meters of fog (fog_transmittance in
// shaders/shader_core.glsl).
float vf_transmittance(float d) {
    int mode = int(vf_params.w);
    if (mode == 1) return 1.0 - smoothstep(vf_params.x, vf_params.y, d);
    float dd = vf_params.z * d;
    return mode == 2 ? exp(-dd * dd) : exp(-dd);
}

// The part of the segment a-b outside the fog-free box, 0..1 (inside counts for vf_box_max.w).
float vf_box_fraction(vec3 a, vec3 b) {
    if (vf_box_min.w < 0.5) return 1.0;
    vec3 d = b - a;
    vec3 inv = 1.0 / (d + vec3(equal(d, vec3(0.0))) * 1e-6);
    vec3 t0 = (vf_box_min.xyz - a) * inv;
    vec3 t1 = (vf_box_max.xyz - a) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float enter = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    float exit = min(min(tmax.x, tmax.y), min(tmax.z, 1.0));
    return 1.0 - (1.0 - vf_box_max.w) * max(exit - enter, 0.0);
}

// The mean of the height falloff between heights ya and yb.
float vf_height_mean(float ya, float yb) {
    float k = vf_height.y;
    if (k <= 0.0) return 1.0;
    ya = max(ya - vf_height.x, 0.0);
    yb = max(yb - vf_height.x, 0.0);
    float dy = yb - ya;
    return abs(dy) > 0.01 ? (exp(-k * ya) - exp(-k * yb)) / (k * dy) : exp(-k * ya);
}

float vf_hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float vf_noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(vf_hash(i), vf_hash(i + vec3(1, 0, 0)), f.x),
                   mix(vf_hash(i + vec3(0, 1, 0)), vf_hash(i + vec3(1, 1, 0)), f.x), f.y),
               mix(mix(vf_hash(i + vec3(0, 0, 1)), vf_hash(i + vec3(1, 0, 1)), f.x),
                   mix(vf_hash(i + vec3(0, 1, 1)), vf_hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}

// The density of the noise at p, about 1 on average.
float vf_mist(vec3 p) {
    if (vf_fwd.w <= 0.0) return 1.0;
    vec3 q = (p - vf_wind.xyz * vf_up.w) * vf_right.w;
    float n = vf_noise(q) * 0.6 + vf_noise(q * 2.7 + 13.1) * 0.4;
    return max(1.0 + vf_fwd.w * (2.0 * smoothstep(0.2, 0.8, n) - 1.0), 0.0);
}

float vf_depth_tap(texture2D map, vec2 uv, float z) {
    return textureLod(sampler2DShadow(map, vf_shadow_smp), vec3(uv, z), 0.0);
}

// How much of the sun the clouds let through to p (cloud_shadow_cs in shaders/clouds.glsl).
float vf_cloud(vec3 p) {
    if (vf_cloud_shadow.y <= 0.0) return 1.0;
    vec3 l = vf_sun_dir.xyz;
    vec2 q = p.xz + l.xz / max(l.y, 0.05) * (vf_cloud_shadow.w - p.y);
    vec2 uv = (q - vf_cloud_shadow.xz) * vf_cloud_shadow.y + 0.5;
    float e = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    // (less of it up in the clouds, none above them: 2.5 km for the layer, as CloudPass's)
    float under = clamp(1.0 - (p.y - vf_cloud_shadow.w) / 2500.0, 0.0, 1.0);
    return mix(1.0, textureLod(sampler2D(vf_cloud_map, vf_cookie_smp), uv, 0.0).r, clamp(e * 12.0, 0.0, 1.0) * under);
}

// How much of the sun gets to p (1 outside of its shadow map).
float vf_sun(vec3 p, float depth) {
    if (vf_shadow_params.x < 0.5) return 1.0;
    int k = -1;
    if (vf_cascade_params.w > 0.5) {
        k = 0;
    } else {
        for (int i = 0; i < 4; i++) {
            if (i >= int(vf_cascade_params.x)) break;
            if (depth < vf_cascade_info[i].x) { k = i; break; }
        }
    }
    if (k < 0) return 1.0;
    vec3 c = (vf_shadow_mat[k] * vec4(p, 1.0)).xyz;
    if (c.x <= 0.0 || c.x >= 1.0 || c.y <= 0.0 || c.y >= 1.0 || c.z <= 0.0 || c.z >= 1.0) return 1.0;
    vec4 rect = vf_cascade_rect[k];
    vec2 t = vf_shadow_params.z / rect.zw;
    vec2 uv = rect.xy + clamp(c.xy, t * 0.5, 1.0 - t * 0.5) * rect.zw;
    float seen = vf_depth_tap(vf_sun_map, uv, c.z);
    vec2 edge = min(c.xy, 1.0 - c.xy);
    return mix(1.0, seen, clamp(min(edge.x, edge.y) * 10.0, 0.0, 1.0));
}

// How much of spot light s gets to p past what casts its shadow.
float vf_spot_lit(int s, vec3 p) {
    int si = int(spots[s].shadow.x);
    if (si < 0) return 1.0;
    vf_spot_shadow sh = spot_shadows[si];
    vec4 sp = sh.mat * vec4(p - spots[s].pos.xyz, 1.0);
    if (sp.w <= 0.0) return 1.0;
    vec3 c = sp.xyz / sp.w;
    // (a map of the atlas holds the distance from the light's plane over its reach)
    if (sh.params.x > 1.5 && sh.params.x < 2.5) c.z = (sp.w - sh.soft.y) / (sh.soft.z - sh.soft.y);
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0 || c.z > 1.0) return 1.0;
    float seen;
    if (sh.params.x > 1.5) {
        float texel = sh.params.z;
        seen = vf_depth_tap(vf_spot_atlas, sh.rect.xy + clamp(c.xy, vec2(texel), vec2(1.0 - texel)) * sh.rect.zw, c.z);
    } else {
        seen = vf_depth_tap(vf_spot_map, c.xy, c.z);
    }
    return mix(1.0, seen, spots[s].shadow.y);
}

// SpotLight.map: how much of spot light s's map gets to p (its picture's luminance; spot_cookie
// in shaders/shader_core.glsl, at a coarse mip: a froxel is large).
float vf_spot_cookie(int s, vec3 p) {
    vec4 rect = spots[s].map;
    if (rect.z <= 0.0) return 1.0;
    vec3 w = p - spots[s].pos.xyz;
    float z = max(dot(w, spots[s].dir.xyz), 1e-4);
    vec3 au = spots[s].map_u.xyz;
    vec3 av = vec3(spots[s].map_u.w, spots[s].map_v.xy);
    vec2 uv = 0.5 + 0.5 * vec2(dot(w, au), dot(w, av)) / z;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 0.0;
    vec2 at = rect.xy + vec2(uv.x, 1.0 - uv.y) * rect.zw;
    vec3 c = textureLod(sampler2D(vf_cookies, vf_cookie_smp), at, max(spots[s].map_v.z - 7.5, 0.0)).rgb;
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// The light's falloff without its 1 / d^2 part: how it fades out towards its cutoff.
float vf_window(float d, float cutoff) {
    if (cutoff <= 0.0) return 1.0;
    float r = d / cutoff;
    float f = clamp(1.0 - r * r * r * r, 0.0, 1.0);
    return f * f;
}

// The integral of 1 / distance^2 to `lp` along the ray o + dir * s, s from s0 to s1 (dir is a
// unit vector); `radius` keeps it finite through the light.
float vf_inverse_square(vec3 o, vec3 dir, float s0, float s1, vec3 lp, float radius) {
    vec3 to_light = lp - o;
    float b = dot(to_light, dir);
    float h = max(sqrt(max(dot(to_light, to_light) - b * b, 0.0)), max(radius, 0.25));
    return (atan((s1 - b) / h) - atan((s0 - b) / h)) / h;
}

void main() {
    ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
    ivec3 grid = ivec3(vf_grid.xyz);
    if (cell.x >= grid.x || cell.y >= grid.y) return;
    int per_row = int(vf_grid.w);
    vec2 ndc = (vec2(cell) + 0.5) / vec2(grid.xy) * 2.0 - 1.0;
    // The sun's shadow is taken over the cell, not at its middle: four rays through its quarters
    // (each at the slice's two points), so a shaft's edge crossing the cell gives part of the
    // light, and the cells' values run smoothly into each other, the same on every frame.
    vec2 quarter = 0.5 / vec2(grid.xy);
    vec3 sub_ray[4];
    for (int i = 0; i < 4; i++) {
        vec2 o = vec2((i & 1) == 0 ? -quarter.x : quarter.x, (i & 2) == 0 ? -quarter.y : quarter.y);
        sub_ray[i] = vf_fwd.xyz + vf_right.xyz * (ndc.x + o.x) + vf_up.xyz * (ndc.y + o.y);
    }
    // a step of 1 m in view depth along this cell's ray
    vec3 ray = vf_fwd.xyz + vf_right.xyz * ndc.x + vf_up.xyz * ndc.y;
    float ray_len = length(ray);
    vec3 look = ray / ray_len;
    vec3 cam = vf_cam.xyz;

    // the sun's side of the view
    float side_density = 1.0;
    float sun_glow = 0.0;
    if (vf_sun_dir.w > 0.0) {
        float side = dot(normalize(look.xz + vec2(1e-5)), normalize(vf_sun_dir.xz + vec2(1e-5)));
        side_density = mix(vf_height.w, 1.0, smoothstep(-0.3, 0.7, side));
        sun_glow = pow(max(dot(look, vf_sun_dir.xyz), 0.0), vf_sun_dir.w);
    }
    bool shafts = vf_sun_color.w > 0.0 && sun_glow > 0.0 && int(vf_params.w) == 3;
    float scatter = vf_color.w;
    ivec2 tile = clamp(ivec2(floor((ndc * 0.5 + 0.5) * vf_clusters.xy)), ivec2(0), ivec2(vf_clusters.xy) - 1);
    int cl_x = int(vf_clusters.x);
    int cl_y = int(vf_clusters.y);
    int cl_z = int(vf_clusters.z);

    float fd = 0.0;     // meters of fog so far, weighted by its density
    float last_t = 1.0;
    vec3 light = vec3(0.0);
    float z0 = 0.0;
    vec3 p0 = cam;
    const float qa = 0.25;
    const float qb = 0.75;
    for (int k = 0; k < grid.z; k++) {
        float z1 = k + 1 >= grid.z ? vf_depth.z : vf_depth.x * exp(float(k + 1) / vf_depth.y);
        vec3 p1 = cam + ray * z1;
        float seg = (z1 - z0) * ray_len;
        float zm = 0.5 * (z0 + z1);
        vec3 pm = cam + ray * zm;
        float za = mix(z0, z1, qa);
        float zb = mix(z0, z1, qb);
        vec3 pa = cam + ray * za;
        vec3 pb = cam + ray * zb;
        float w = vf_height_mean(p0.y, p1.y) * vf_box_fraction(p0, p1) * side_density * vf_mist(pm);
        float fd_mid = fd + seg * w * 0.5;
        fd += seg * w;
        float t = vf_transmittance(fd);
        // the fog's own light, the sun's towards it
        vec3 own = vf_color.rgb;
        if (sun_glow > 0.0) {
            float lit = sun_glow * vf_cloud(pm);
            if (shafts) {
                float seen = 0.0;
                for (int i = 0; i < 4; i++) seen += vf_sun(cam + sub_ray[i] * za, za) + vf_sun(cam + sub_ray[i] * zb, zb);
                lit *= mix(1.0, seen * 0.125, vf_sun_color.w);
            }
            own = mix(own, vf_sun_color.rgb, lit);
        }
        light += own * (last_t - t);
        // the lamps': the fog glows around them (1 / d^2, integrated over the slice)
        if (scatter > 0.0 && w > 0.0 && vf_clusters.w > 0.5) {
            int slice = zm <= vf_cluster_depth.x ? 0 : int(floor(log(zm / vf_cluster_depth.x) * vf_cluster_depth.y));
            slice = clamp(slice, 0, cl_z - 1);
            uvec4 range = clusters[tile.x + cl_x * (tile.y + cl_y * slice)].range;
            int first = int(range.x);
            int np = int(range.y);
            int ns = int(range.z);
            float s0 = z0 * ray_len;
            float sm = zm * ray_len;
            float s1 = z1 * ray_len;
            vec3 glow = vec3(0.0);
            for (int i = 0; i < np; i++) {
                vf_point_light pl = points[cluster_lights[first + i].index];
                float d = length(pl.pos.xyz - pm);
                float along = vf_inverse_square(cam, look, s0, s1, pl.pos.xyz, pl.params.x);
                glow += pl.color.rgb * along * vf_window(d, pl.pos.w) * vf_transmittance(fd_mid + d * w);
            }
            for (int i = 0; i < ns; i++) {
                int s = int(cluster_lights[first + np + i].index);
                if (spots[s].params.w <= 0.0) continue;
                vec3 sp = spots[s].pos.xyz;
                vec3 sd = spots[s].dir.xyz;
                float cos_o = spots[s].dir.w;
                float cos_i = spots[s].params.x;
                float rad = spots[s].params.y;
                float cutoff = spots[s].pos.w;
                float acc = 0.0;
                // Two stratified sub-slices per froxel keep cone edges and shadow shafts smooth
                // without per-froxel XY jitter creating checkerboard patterns.
                vec3 lv0 = pa - sp;
                float d0 = length(lv0);
                float c0 = smoothstep(cos_o, cos_i, dot(lv0 / max(d0, 1e-4), sd));
                if (c0 > 0.0) {
                    float lit0 = c0 * vf_spot_lit(s, pa) * vf_spot_cookie(s, pa);
                    if (lit0 > 0.0) {
                        acc += lit0 * vf_inverse_square(cam, look, s0, sm, sp, rad)
                             * vf_window(d0, cutoff) * vf_transmittance(fd - seg * w * (1.0 - qa) + d0 * w);
                    }
                }
                vec3 lv1 = pb - sp;
                float d1 = length(lv1);
                float c1 = smoothstep(cos_o, cos_i, dot(lv1 / max(d1, 1e-4), sd));
                if (c1 > 0.0) {
                    float lit1 = c1 * vf_spot_lit(s, pb) * vf_spot_cookie(s, pb);
                    if (lit1 > 0.0) {
                        acc += lit1 * vf_inverse_square(cam, look, sm, s1, sp, rad)
                             * vf_window(d1, cutoff) * vf_transmittance(fd - seg * w * (1.0 - qb) + d1 * w);
                    }
                }
                if (acc > 0.0) glow += spots[s].color.rgb * (spots[s].params.w * acc);
            }
            light += glow * scatter * w;
        }
        last_t = t;
        ivec2 at = ivec2((k % per_row) * grid.x, (k / per_row) * grid.y) + cell;
        imageStore(vf_out, at, vec4(light, t));
        z0 = z1;
        p0 = p1;
    }
}
@end

@program cluster cluster_cs
@program cull cull_cs
// The fog volume, blurred across its cells (3x3, 1-2-1 each way) within each slice's tile: the
// softer edges of the shafts and of the cells, as Godot's fog filter. Depth is left alone.
@cs fog_blur_cs
layout(binding=0) uniform fog_blur_params {
    vec4 fb_grid;   // as vf_grid: x, y = cells, z = slices, w = slices per row of tiles
};
layout(binding=0) uniform texture2D fb_in;
layout(binding=0) uniform sampler fb_smp;
layout(binding=1, rgba16f) uniform writeonly image2D fb_out;
layout(local_size_x=8, local_size_y=8) in;

void main() {
    ivec2 cells = ivec2(fb_grid.xy);
    int per_row = int(fb_grid.w);
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    ivec2 tile = at / cells;
    if (tile.y * per_row + tile.x >= int(fb_grid.z) || tile.x >= per_row) return;
    ivec2 cell = at - tile * cells;
    ivec2 origin = tile * cells;
    vec4 sum = vec4(0.0);
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            ivec2 c = clamp(cell + ivec2(x, y), ivec2(0), cells - 1);
            float w = (x == 0 ? 2.0 : 1.0) * (y == 0 ? 2.0 : 1.0);
            sum += texelFetch(sampler2D(fb_in, fb_smp), origin + c, 0) * w;
        }
    }
    imageStore(fb_out, at, sum / 16.0);
}
@end

@program fog fog_cs
@program fog_blur fog_blur_cs
