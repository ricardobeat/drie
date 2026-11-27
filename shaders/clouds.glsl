// Volumetric clouds (CloudPass, drie::postprocessing; after Schneider's Nubis talks on Horizon
// Zero Dawn and Forbidden West): a layer of clouds between two altitudes around a curved earth,
// shaped by a weather map (coverage, cloud type, and its distance field for skipping clear air)
// and eroded by two tiling 3D noises. cloud_shadow_cs: what the clouds let through of the sun,
// seen from straight under them, for surfaces and the fog. cloud_march_cs: each pixel's ray,
// marched from the camera, so that the camera can be in or above them; run once for what is near
// (coarser pixels, finer steps) and once for the rest. cloud_resolve_cs: the two put together and
// blended into the last frames' along the camera's motion. cloud_composite: over the picture, by
// depth, through the froxel fog in front of them. Regenerate generated/clouds.c3 with
// `just shaders`.
@header module drie::postprocessing;
@header import drie;

@ctype mat4 Mat4
@ctype vec4 Vec4

@block cloud_common
layout(binding=0) uniform cloud_params {
    vec4 cl_cam;        // the camera's position, w = frame
    vec4 cl_fwd;        // forward (with the projection's shift), w = time (s)
    vec4 cl_right;      // right * tan(half the horizontal view)
    vec4 cl_up;         // up * tan(half the vertical view), w = 1: rows run top down
    vec4 cl_layer;      // x, y = bottom and top altitude (m), z = the earth's radius (m), w = density
    vec4 cl_shape;      // x = shape noise's scale (1/m), y = detail's, z = detail erosion, w = coverage
    vec4 cl_weather;    // xy = wind offset (m), z = 1 / weather map's span (m), w = meters to the distance field's unit
    vec4 cl_sun_dir;    // towards the sun, w = forward scattering (g)
    vec4 cl_sun_color;  // the sun's light on the clouds, w = extinction (1/m at density 1)
    vec4 cl_amb_top;    // the sky's light on their tops, w = light steps
    vec4 cl_amb_bottom; // and on their bottoms (from the ground), w = most view steps
    vec4 cl_haze;       // rgb = the air's color far away, w = its density (1/m)
    vec4 cl_size;       // xy = this pass's pixels, zw = the depth texture's
    vec4 cl_range;      // x, y = distances marched (m), z, w = the camera's near and far
    vec4 cl_shadow;     // shadow map: xz = its middle, y = 1 / its span (m), w = 1: the depth runs 0..1
    vec4 cl_step;       // x = nearest step (m), y = step per sqrt(m), z = farthest distance (m)
};
layout(binding=0) uniform texture3D cl_noise;    // shape: rgba = Perlin-Worley, Worley fbm
layout(binding=1) uniform texture3D cl_detail;   // detail: rgb = Worley at 3 frequencies
layout(binding=2) uniform texture2D cl_weather_map; // r = coverage, g = type, b = distance to cloud
layout(binding=0) uniform sampler cl_smp;        // linear, repeating

float remap(float v, float a, float b, float c, float d) {
    return c + (v - a) / (b - a) * (d - c);
}

float cl_hg(float c, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (12.566371 * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// The weather at x, z: coverage, type, and how far (m) the nearest cloud is.
vec3 cl_weather_at(vec3 p) {
    vec4 w = textureLod(sampler2D(cl_weather_map, cl_smp), (p.xz + cl_weather.xy) * cl_weather.z, 0.0);
    return vec3(w.r, w.g, w.b * cl_weather.w);
}

// 0..1 between the layer's bottom and top, around the earth (whose middle is (0, -radius, 0)).
float cl_height(vec3 p) {
    float r = cl_layer.z;
    float alt = length(p + vec3(0.0, r, 0.0)) - r;
    return (alt - cl_layer.x) / (cl_layer.y - cl_layer.x);
}

// How much cloud the weather allows at height h: the vertical profile of its type (flat
// bottoms; stratus low and thin, cumulus up to the top, narrowing as they rise), by coverage.
float cl_profile(vec3 w, float h) {
    float cov = clamp(remap(w.x, 1.0 - cl_shape.w, 1.0, 0.0, 1.0), 0.0, 1.0);
    float top = mix(0.2, 0.85, w.y * w.y);
    float bottom = smoothstep(0.0, 0.06, h);
    float upper = clamp((top - h) / (top * 0.8), 0.0, 1.0);
    upper = smoothstep(0.0, 1.0, upper);
    return cov * bottom * upper * step(0.0, h) * step(h, 1.0);
}

// Cloud density at p (0..1 times cl_layer.w), the detail noise eroding the edges when `fine`.
float cl_density(vec3 p, float h, vec3 w, bool fine) {
    float dim = cl_profile(w, h);
    if (dim <= 0.0) return 0.0;
    // drifting with the wind, and rising through it: the clouds change shape as they go
    vec3 drift = vec3(cl_weather.x, -cl_fwd.w * 4.0, cl_weather.y);
    vec4 n = textureLod(sampler3D(cl_noise, cl_smp), (p + drift) * cl_shape.x, 0.0);
    float fbm = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
    float shape = remap(n.r, fbm - 1.0, 1.0, 0.0, 1.0);
    float d = clamp(remap(shape, 1.0 - dim, 1.0, 0.0, 1.0), 0.0, 1.0);
    if (fine && d > 0.0) {
        vec3 q = textureLod(sampler3D(cl_detail, cl_smp), (p + drift * 1.6 + vec3(0.0, -cl_fwd.w * 6.0, 0.0)) * cl_shape.y, 0.0).rgb;
        float det = q.r * 0.625 + q.g * 0.25 + q.b * 0.125;
        // wispy at the bottoms, billowing above
        det = mix(det, 1.0 - det, clamp(h * 4.0, 0.0, 1.0));
        d = clamp(remap(d, det * cl_shape.z, 1.0, 0.0, 1.0), 0.0, 1.0);
    }
    // sharpened (as Nubis does): thin parts denser, so that the edges are crisp
    return pow(d, 0.45) * cl_layer.w;
}

// Where the ray from o along unit dir enters and leaves the layer (t0 > t1: never).
vec2 cl_layer_span(vec3 o, vec3 dir) {
    float r = cl_layer.z;
    vec3 c = o + vec3(0.0, r, 0.0);
    float b = dot(c, dir);
    float cc = dot(c, c);
    float r0 = r + cl_layer.x;
    float r1 = r + cl_layer.y;
    float d1 = b * b - (cc - r1 * r1);
    if (d1 < 0.0) return vec2(1.0, 0.0);
    float s1 = sqrt(d1);
    float out1 = -b + s1; // leaving the top
    float in1 = -b - s1;  // coming in through the top (from above)
    float d0 = b * b - (cc - r0 * r0);
    float s0 = d0 > 0.0 ? sqrt(d0) : 0.0;
    float rad = sqrt(cc);
    if (rad < r0) {
        // under the clouds: up through the bottom, out the top
        return vec2(-b + s0, out1);
    }
    if (rad < r1) {
        // in them: out the top, or down out of the bottom
        float down = (d0 > 0.0 && -b - s0 > 0.0) ? -b - s0 : out1;
        return vec2(0.0, down);
    }
    // above them: in through the top, out the bottom (or the top again)
    if (in1 < 0.0) return vec2(1.0, 0.0);
    float leave = (d0 > 0.0 && -b - s0 > 0.0) ? -b - s0 : out1;
    return vec2(in1, leave);
}
@end

// ---- The cloud shadow map

@cs cloud_shadow_cs
@include_block cloud_common
layout(binding=5, rgba16f) uniform writeonly image2D cl_shadow_out;
layout(local_size_x=8, local_size_y=8) in;

// Each texel is a point on the plane of the layer's bottom; marched up towards the sun to the
// top. r = the share of sunlight that gets through.
void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(cl_size.xy);
    if (at.x >= size.x || at.y >= size.y) return;
    vec2 xz = cl_shadow.xz + ((vec2(at) + 0.5) / vec2(size) - 0.5) / cl_shadow.y;
    vec3 l = cl_sun_dir.xyz;
    l.y = max(l.y, 0.05);
    l = normalize(l);
    float len = (cl_layer.y - cl_layer.x) / l.y;
    const int STEPS = 12;
    float ds = len / float(STEPS);
    // four columns over the texel and around it, averaged: soft edges, not the texels' steps
    float texel = 1.0 / (cl_shadow.y * float(size.x));
    float t = 0.0;
    for (int c = 0; c < 4; c++) {
        vec2 o = vec2((c & 1) == 0 ? -0.6 : 0.6, (c & 2) == 0 ? -0.6 : 0.6) * texel;
        vec3 p0 = vec3(xz.x + o.x, cl_layer.x, xz.y + o.y);
        float sum = 0.0;
        for (int i = 0; i < STEPS; i++) {
            vec3 p = p0 + l * ((float(i) + 0.5 + 0.25 * float(c)) * ds);
            float h = (p.y - cl_layer.x) / (cl_layer.y - cl_layer.x);
            sum += cl_density(p, h, cl_weather_at(p), false);
        }
        t += exp(-sum * ds * cl_sun_color.w) * 0.25;
    }
    imageStore(cl_shadow_out, at, vec4(t, t, t, 1.0));
}
@end

// ---- The march

@cs cloud_march_cs
@include_block cloud_common
@image_sample_type cl_depth unfilterable_float
@sampler_type cl_point nonfiltering
layout(binding=3) uniform texture2D cl_depth;     // the scene's depth
layout(binding=4) uniform texture2D cl_blue;      // 64 x 64 blue noise (r)
layout(binding=1) uniform sampler cl_point;       // nearest
layout(binding=5, rgba16f) uniform writeonly image2D cl_color_out; // rgb = light, a = transmittance
layout(binding=6, rgba16f) uniform writeonly image2D cl_depth_out; // x = cloud's distance, y = where the ray ended
layout(local_size_x=8, local_size_y=8) in;

// Distance along the unit ray to what the depth buffer shows at pixel `at` (1e9: nothing).
float scene_distance(ivec2 at, float ray_len) {
    float d = texelFetch(sampler2D(cl_depth, cl_point), at, 0).r;
    if (d >= 1.0) return 1e9;
    float n = cl_range.z;
    float f = cl_range.w;
    float z = cl_shadow.w > 0.5 ? n * f / (f - d * (f - n)) : 2.0 * n * f / (f + n - (d * 2.0 - 1.0) * (f - n));
    return z * ray_len;
}

// The sun's light reaching p through the cloud towards it: a few steps, growing.
float light_march(vec3 p, vec3 l, float h0) {
    float sum = 0.0;
    float t = 0.0;
    int steps = int(cl_amb_top.w);
    float thick = cl_layer.y - cl_layer.x;
    float seg = thick * 0.012;
    for (int i = 0; i < steps; i++) {
        float ds = seg * (1.0 + float(i) * 0.9);
        t += ds;
        vec3 q = p + l * (t - ds * 0.5);
        float h = cl_height(q);
        if (h > 1.0) break;
        sum += cl_density(q, h, cl_weather_at(q), i < 1) * ds;
    }
    return sum;
}

void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(cl_size.xy);
    if (at.x >= size.x || at.y >= size.y) return;
    vec2 uv = (vec2(at) + 0.5) / vec2(size);
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, cl_up.w > 0.5 ? 1.0 - uv.y * 2.0 : uv.y * 2.0 - 1.0);
    vec3 ray = cl_fwd.xyz + cl_right.xyz * ndc.x + cl_up.xyz * ndc.y;
    float ray_len = length(ray);
    vec3 dir = ray / ray_len;
    vec3 cam = cl_cam.xyz;

    ivec2 full = ivec2(min(uv * cl_size.zw, cl_size.zw - 1.0));
    float scene = scene_distance(full, ray_len);
    vec2 span = cl_layer_span(cam, dir);
    float t0 = max(max(span.x, 0.0), cl_range.x);
    float t1 = min(min(span.y, scene), min(cl_range.y, cl_step.z));

    vec3 light = vec3(0.0);
    float trans = 1.0;
    float dist_sum = 0.0;
    float weight_sum = 0.0;
    if (t1 > t0) {
        vec3 l = cl_sun_dir.xyz;
        float cos_t = dot(dir, l);
        float g = cl_sun_dir.w;
        // forward and a little back scattering; multiple scattering as fainter, wider octaves
        float ph0 = mix(cl_hg(cos_t, g), cl_hg(cos_t, -0.25), 0.25);
        float ph1 = mix(cl_hg(cos_t, g * 0.4), cl_hg(cos_t, -0.1), 0.25);
        float ph2 = 0.0796; // the light scattered many times: from all round
        float sigma = cl_sun_color.w;
        // blue noise, moved each frame by the golden ratio, as where in its first step to start
        float bn = texelFetch(sampler2D(cl_blue, cl_point), at % 64, 0).r;
        bn = fract(bn + cl_cam.w * 0.6180339887);
        float t = t0;
        float dt = max(cl_step.x, sqrt(t) * cl_step.y);
        t += dt * bn;
        int most = int(cl_amb_bottom.w);
        bool coarse = true;
        int empty = 0;
        for (int i = 0; i < most; i++) {
            if (t >= t1 || trans < 0.01) break;
            dt = max(cl_step.x, sqrt(t) * cl_step.y);
            vec3 p = cam + dir * t;
            float h = cl_height(p);
            vec3 w = cl_weather_at(p);
            float dim = cl_profile(w, h);
            if (dim <= 0.0) {
                // clear air: as far as the weather map says no cloud is (the distance field)
                t += max(dt, w.z * 0.9);
                continue;
            }
            // cheap until it finds cloud: the shape alone, longer steps; then back a step, fine
            if (coarse) {
                if (cl_density(p, h, w, false) <= 0.0) {
                    t += dt * 2.0;
                    continue;
                }
                coarse = false;
                empty = 0;
                t = max(t - dt, t0);
                continue;
            }
            float dens = cl_density(p, h, w, true);
            if (dens <= 0.0) {
                if (++empty >= 6) coarse = true;
            } else {
                empty = 0;
            }
            if (dens > 0.0) {
                float od = light_march(p, l, h) * sigma;
                vec3 sun = cl_sun_color.rgb * (exp(-od) * ph0 + 0.6 * exp(-od * 0.3) * ph1 + 0.35 * exp(-od * 0.1) * ph2);
                // the sky's light: more on the tops, less deep in the cloud's lower part
                vec3 amb = mix(cl_amb_bottom.rgb, cl_amb_top.rgb, clamp(h * 1.2, 0.0, 1.0)) * mix(0.55, 1.0, pow(1.0 - dim, 0.5));
                // powder: the fronts of thin edges, darker than Beer alone has them
                float powder = 1.0 - exp(-dens * sigma * 2.0 * 60.0);
                vec3 s = (sun * mix(1.0, powder, 0.5) + amb) * sigma * dens;
                float ext = sigma * dens;
                float step_t = exp(-ext * dt);
                vec3 sint = (s - s * step_t) / ext;
                light += trans * sint;
                float took = trans * (1.0 - step_t);
                dist_sum += t * took;
                weight_sum += took;
                trans *= step_t;
            }
            t += dt;
        }
    }
    float cloud_dist = weight_sum > 1e-4 ? dist_sum / weight_sum : min(scene, max(t1, 1.0));
    // fading out before the farthest distance marched, so that they don't end in a line
    float fade = 1.0 - smoothstep(cl_step.z * 0.55, cl_step.z, cloud_dist);
    light *= fade;
    trans = mix(1.0, trans, fade);
    // the air in front of them: towards its far color, as far as they are
    if (weight_sum > 1e-4) {
        float haze = 1.0 - exp(-cloud_dist * cl_haze.w);
        light = mix(light, cl_haze.rgb * (1.0 - trans), haze);
    }
    imageStore(cl_color_out, at, vec4(light, trans));
    imageStore(cl_depth_out, at, vec4(cloud_dist, min(scene, 1e7), 0.0, 1.0));
}
@end

// ---- Near and far put together, and blended over time

@cs cloud_resolve_cs
layout(binding=0) uniform cloud_resolve_params {
    vec4 cr_size;      // xy = this pass's pixels, z = 1: start over, w = the new frame's share
    vec4 cr_cam;       // the camera's position, w = 1: rows run top down
    vec4 cr_fwd;
    vec4 cr_right;
    vec4 cr_up;
    mat4 cr_prev_view_proj; // last frame's camera
};
layout(binding=0) uniform texture2D cr_far;        // the far march (this pass's size)
layout(binding=1) uniform texture2D cr_far_depth;
layout(binding=2) uniform texture2D cr_near;       // the near march (smaller)
layout(binding=3) uniform texture2D cr_near_depth;
layout(binding=4) uniform texture2D cr_history;    // last frame's result
layout(binding=0) uniform sampler cr_linear;
layout(binding=1) uniform sampler cr_point;
layout(binding=5, rgba16f) uniform writeonly image2D cr_out;
layout(binding=6, rgba16f) uniform writeonly image2D cr_depth_out;
layout(local_size_x=8, local_size_y=8) in;

// The two marches at pixel `at`: near over far. Its depth (x, the cloud's weighted distance;
// y, where the ray ended) in `depth`.
vec4 combined(ivec2 at, out vec2 depth) {
    ivec2 size = ivec2(cr_size.xy);
    at = clamp(at, ivec2(0), size - 1);
    vec2 uv = (vec2(at) + 0.5) / vec2(size);
    vec4 f = texelFetch(sampler2D(cr_far, cr_point), at, 0);
    vec2 fd = texelFetch(sampler2D(cr_far_depth, cr_point), at, 0).xy;
    vec4 n = textureLod(sampler2D(cr_near, cr_linear), uv, 0.0);
    vec2 nd = textureLod(sampler2D(cr_near_depth, cr_point), uv, 0.0).xy;
    float an = 1.0 - n.a;
    float af = (1.0 - f.a) * n.a;
    depth = vec2((nd.x * an + fd.x * af) / max(an + af, 1e-4), fd.y);
    if (an + af < 1e-4) depth.x = fd.x;
    return vec4(n.rgb + n.a * f.rgb, n.a * f.a);
}

void main() {
    ivec2 at = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(cr_size.xy);
    if (at.x >= size.x || at.y >= size.y) return;
    vec2 depth;
    vec4 cur = combined(at, depth);
    vec4 lo = cur;
    vec4 hi = cur;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) continue;
            vec2 dn;
            vec4 c = combined(at + ivec2(x, y), dn);
            lo = min(lo, c);
            hi = max(hi, c);
        }
    }
    vec4 result = cur;
    if (cr_size.z < 0.5) {
        // where the cloud seen here was on the last frame
        vec2 uv = (vec2(at) + 0.5) / vec2(size);
        vec2 ndc = vec2(uv.x * 2.0 - 1.0, cr_cam.w > 0.5 ? 1.0 - uv.y * 2.0 : uv.y * 2.0 - 1.0);
        vec3 dir = normalize(cr_fwd.xyz + cr_right.xyz * ndc.x + cr_up.xyz * ndc.y);
        vec3 p = cr_cam.xyz + dir * depth.x;
        vec4 clip = cr_prev_view_proj * vec4(p, 1.0);
        if (clip.w > 0.0) {
            vec2 pn = clip.xy / clip.w;
            vec2 puv = vec2(pn.x * 0.5 + 0.5, cr_cam.w > 0.5 ? 0.5 - pn.y * 0.5 : pn.y * 0.5 + 0.5);
            if (all(greaterThanEqual(puv, vec2(0.0))) && all(lessThanEqual(puv, vec2(1.0)))) {
                vec4 hist = textureLod(sampler2D(cr_history, cr_linear), puv, 0.0);
                hist = clamp(hist, lo, hi);
                result = mix(hist, cur, cr_size.w);
            }
        }
    }
    imageStore(cr_out, at, result);
    imageStore(cr_depth_out, at, vec4(depth, 0.0, 1.0));
}
@end

// ---- Over the picture

@vs cloud_composite_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
@end

@fs cloud_composite_fs
@image_sample_type cc_depth unfilterable_float
@image_sample_type cc_cloud_depth unfilterable_float
@sampler_type cc_point nonfiltering
layout(binding=0) uniform cloud_composite_params {
    vec4 cc_size;   // xy = the picture's pixels, zw = the clouds'
    vec4 cc_clip;   // x, y = the camera's near and far, z = 1: depth runs 0..1, w = 1: fog volume
    vec4 cc_fwd;    // forward (with the projection's shift)
    vec4 cc_right;  // right * tan(half the horizontal view)
    vec4 cc_up;     // up * tan(half the vertical view), w = 1: rows run top down
    vec4 cc_fog_grid;   // the froxel fog's grid (Renderer.fog_info)
    vec4 cc_fog_slices;
};
layout(binding=0) uniform texture2D cc_depth;       // the scene's depth
layout(binding=1) uniform texture2D cc_cloud;       // the resolved clouds
layout(binding=2) uniform texture2D cc_cloud_depth;
layout(binding=3) uniform texture2D cc_fog;         // the froxel fog's atlas
layout(binding=0) uniform sampler cc_point;
layout(binding=1) uniform sampler cc_linear;
out vec4 frag_color;

// The froxel fog in front of view depth z on the ray through screen point uv (y up): rgb = its
// light, a = what gets through (fog_volume_at in src/material/layers.c3).
vec4 fog_at(vec2 uv, float z) {
    vec4 grid = cc_fog_grid;
    vec4 sl = cc_fog_slices;
    float s = z <= sl.x ? -1.0 : clamp(log(z / sl.x) * sl.y - 1.0, -1.0, grid.z - 1.0);
    float k = floor(s);
    float f = s - k;
    vec2 size = vec2(textureSize(sampler2D(cc_fog, cc_linear), 0));
    vec2 cell = clamp(uv * grid.xy, vec2(0.5), grid.xy - 0.5);
    float kb = min(k + 1.0, grid.z - 1.0);
    vec2 ab = vec2(mod(kb, grid.w), floor(kb / grid.w)) * grid.xy + cell;
    vec4 b = textureLod(sampler2D(cc_fog, cc_linear), ab / size, 0.0);
    vec4 a = vec4(0.0, 0.0, 0.0, 1.0);
    if (k >= 0.0) {
        vec2 aa = vec2(mod(k, grid.w), floor(k / grid.w)) * grid.xy + cell;
        a = textureLod(sampler2D(cc_fog, cc_linear), aa / size, 0.0);
    }
    return mix(a, b, f);
}

void main() {
    ivec2 px = ivec2(gl_FragCoord.xy);
    vec2 uv = gl_FragCoord.xy / cc_size.xy;
    float d = texelFetch(sampler2D(cc_depth, cc_point), px, 0).r;
    float n = cc_clip.x;
    float f = cc_clip.y;
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, cc_up.w > 0.5 ? 1.0 - uv.y * 2.0 : uv.y * 2.0 - 1.0);
    float ray_len = length(cc_fwd.xyz + cc_right.xyz * ndc.x + cc_up.xyz * ndc.y);
    float scene = 1e7;
    if (d < 1.0) {
        float z = cc_clip.z > 0.5 ? n * f / (f - d * (f - n)) : 2.0 * n * f / (f + n - (d * 2.0 - 1.0) * (f - n));
        scene = z * ray_len;
    }
    // the four cloud pixels around this one, weighted by how near the scene was behind each
    vec2 at = uv * cc_size.zw - 0.5;
    ivec2 base = ivec2(floor(at));
    vec2 fr = at - vec2(base);
    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    float cdist = 0.0;
    ivec2 lim = ivec2(cc_size.zw) - 1;
    vec4 nearest = vec4(0.0, 0.0, 0.0, 1.0);
    float best = 1e30;
    float nearest_dist = 0.0;
    for (int i = 0; i < 4; i++) {
        ivec2 o = ivec2(i & 1, i >> 1);
        ivec2 q = clamp(base + o, ivec2(0), lim);
        vec4 c = texelFetch(sampler2D(cc_cloud, cc_point), q, 0);
        vec2 cd = texelFetch(sampler2D(cc_cloud_depth, cc_point), q, 0).xy;
        float bw = (o.x == 1 ? fr.x : 1.0 - fr.x) * (o.y == 1 ? fr.y : 1.0 - fr.y);
        float diff = abs(cd.y - scene) / max(min(cd.y, scene), 1.0);
        float w = bw / (1e-3 + diff * 8.0);
        sum += c * w;
        cdist += cd.x * w;
        wsum += w;
        if (diff < best) {
            best = diff;
            nearest = c;
            nearest_dist = cd.x;
        }
    }
    vec4 cloud = wsum > 1e-3 ? sum / wsum : nearest;
    float dist = wsum > 1e-3 ? cdist / wsum : nearest_dist;
    if (best > 0.5) {
        cloud = nearest;
        dist = nearest_dist;
    }
    // nothing (no cloud: not even the fog in front of where one might be)
    if (cloud.a > 0.995) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    // the froxel fog in front of the cloud
    if (cc_clip.w > 0.5) {
        vec4 fog = fog_at(vec2(uv.x, cc_up.w > 0.5 ? 1.0 - uv.y : uv.y), dist / ray_len);
        cloud.rgb = cloud.rgb * fog.a + fog.rgb * (1.0 - cloud.a);
    }
    frag_color = cloud;
}
@end

@program cloud_shadow cloud_shadow_cs
@program cloud_march cloud_march_cs
@program cloud_resolve cloud_resolve_cs
@program cloud_composite cloud_composite_vs cloud_composite_fs
