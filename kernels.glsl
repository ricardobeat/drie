// Compute shaders of the renderer (see compute.c3): light clustering and instance culling.
// ./build.sh shaders turns this into kernels.c3.
@header module three;

// Sorts the point and spot lights into the clusters (froxels) of a view, as cluster_lights_cpu
// does in clusters.c3, with the same result: per cluster its first entry in the index list, its
// point lights, its spot lights after them. The CPU finds the clusters each light's sphere
// touches (a box of them, `ranges`); this fills the grid in one group of 256: each thread
// counts the lights of its run of clusters, the runs are added up in order, and each thread
// writes its clusters' lists where they start.
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
    uvec4 range;  // as gpu_light_cluster in shader.glsl
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
// to `visible` from the job's first entry, and how many there are to counts[job]. An
// instanced draw then reads its matrices through `visible` (shader.glsl, fetch_instance).
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
struct cull_row {
    vec4 row;
};
struct cull_index {
    uint index;
};
layout(binding=1) readonly buffer cull_planes { cull_plane planes[]; };
layout(binding=2) readonly buffer cull_instances { cull_row rows[]; };
layout(binding=3) buffer cull_visible { cull_index visible[]; };
layout(binding=4) buffer cull_counts { cull_index counts[]; };

layout(local_size_x=256) in;
shared uint cull_scan[256];

bool cull_keep(cull_job job, uint i) {
    uint s = (job.range.x + i) * 3;
    vec4 r0 = rows[s].row;
    vec4 r1 = rows[s + 1].row;
    vec4 r2 = rows[s + 2].row;
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

@program cluster cluster_cs
@program cull cull_cs
