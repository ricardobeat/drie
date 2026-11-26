// The kernels of GpuCloth (gpu_cloth.c3): XPBD cloth in small steps, all of it on the GPU.
// just shaders turns this into generated/gpu_cloth.c3.
//
//   gc_begin_cs       per point, at a step's start: where it starts from (the mesh is drawn
//                     between there and the step's end), a pinned point put on its pin
//   gc_forces_cs      per point: gravity, and the air on the triangles around it (wind, drag)
//   gc_integrate_cs   per point and substep: moved by its velocity
//   gc_links_cs       per link of one color (no two of a color share a point): XPBD distance
//                     constraint, one pass; one dispatch per color, so the points are solved in
//                     the same order every time (Gauss-Seidel), without atomics
//   gc_bends_cs       per bend of one run (no two of a run share a point): XPBD on a point's
//                     offset from the midpoint of its neighbours along the normal, as Cloth's
//                     (bend_model SIGNED; DISTANCE bends are links)
//   gc_tethers_cs     per point: no further from its nearest pin than along the cloth at rest
//   gc_clear_cs, gc_count_cs, gc_scan_*_cs, gc_fill_cs
//                     the spatial hash of the points: counted into cells, the counts summed
//                     (a scan), the points put in their cells
//   gc_self_cs        per point: pushed from the points nearer than the self-collision
//                     distance (from the 27 cells around it), summed in fixed point so the
//                     order the hash lists them in does not change the result
//   gc_collide_cs     per point: the self-collision push, the colliders, the floor, and the
//                     velocity from the substep's move
//   gc_write_cs       per vertex of a panel's mesh: its position and normal, written into its
//                     vertex buffer
@header module drie::shaders;
@header import drie;

@ctype vec4 Vec4

@block gc_common
layout(binding=0) uniform gc_params {
    ivec4 gc_counts;   // x = points, y = columns, z = rows, w = hash cells (a power of two)
    ivec4 gc_range;    // x = first link, y = links (of the color), z = colliders, w = 1: a floor
    vec4 gc_step;      // x = substep h, y = velocity kept (exp(-damping h) / h), z = the link's
                       //   compliance / h², w = friction
    vec4 gc_gravity;   // xyz, w = the colliders' thickness
    vec4 gc_wind;      // xyz = the air's velocity, w = drag (acceleration per m² per m/s of a point)
    vec4 gc_sweep;     // x = t, y = t a substep ago (where the colliders are), z = the floor's y,
                       //   w = self-collision distance
    vec4 gc_origin;    // xyz = where the points are kept from, w = the mesh's place from start to now
    vec4 gc_gust;      // x = gust, y = time, z = width, w = height of the cloth at rest
    vec4 gc_self;      // x = 1 / hash cell size, y = self-collision stiffness, z = tether slack,
                       //   w = self-collision friction
    vec4 gc_mode;      // w = gc_bodies_cs's mode (xy: the update and its steps), x = gc_damp_cs's share,
                       //   y = lift (gc_forces_cs: of the drag)
    ivec4 gc_grid;     // x = columns of a panel (y: the first column of gc_write_cs's, z: its
                       //   vertices to a cell's side)
};

struct gc_point {
    vec4 p;
};

// a pinned point's place (xyz) and every point's inverse mass (w)
struct gc_pin {
    vec4 p;
};

struct gc_link {
    uint a;
    uint b;
    float rest;
    float pad;
};

// m between a and b along a row or column: (a + b - 2m) along the normal at m, s (b - a) x
// (d - c), is held at `rest`; c and d are m's neighbours the other way (m itself at an edge), s
// 1 along a row and -1 down a column
struct gc_bend {
    uint a;
    uint m;
    uint b;
    uint c;
    uint d;
    float s;
    float rest;
    float pad;
};

struct gc_tether {
    uint pin;     // 0xffffffff: none
    float rest;
};

struct gc_cell {
    uint n;
};

// a point's self-collision push, in 1/GC_FIXED m, and (w) how many points pushed it
struct gc_push {
    ivec4 d;
};

// SPHERE 0, CAPSULE 1, BOX 2, PLANE 3 (info.x); from_* where it was at the update's start,
// the others where it is going; x and y: a box's first two axes (a plane's normal is x)
struct gc_collider {
    vec4 info;      // x = kind, y = radius, z = 1: active, w = its body (-1: none, kinematic)
    vec4 mass;      // x = 1 / its mass, y = 1: its body starts over (put somewhere new),
                    // z = its friction (-1: the cloth's)
    vec4 corr_off;  // what the CPU takes off its body's offset and velocity at the update's end
    vec4 corr_dv;
    vec4 from_a;
    vec4 a;
    vec4 from_b;
    vec4 b;
    vec4 from_x;
    vec4 from_y;
    vec4 x;
    vec4 y;
    vec4 half_size;
};

// A collider with a mass: how far the cloth has pushed it from where it was moved to (off)
// and how much it has changed its velocity (dv), of which the CPU has not been told yet; where
// it was at the last step's start and end (begin, end: its mesh is drawn between them, as the
// cloth's is)
struct gc_body {
    vec4 off;
    vec4 dv;
    vec4 begin;
    vec4 end;
    vec4 mark;  // the update (x) and the steps by its end (y) it is the state at the end of
};

const int GC_BODIES = 8;

// a vertex of the mesh (FLOATS_PER_VERTEX): position, normal, uv, color
struct gc_vertex {
    vec4 a;  // position, normal.x
    vec4 b;  // normal.yz, uv
    vec4 c;  // color
};

const uint GC_NONE = 0xffffffffu;
const float GC_FIXED = 16777216.0;

uint gc_hash(ivec3 c) {
    uint h = (uint(c.x) * 92837111u) ^ (uint(c.y) * 689287499u) ^ (uint(c.z) * 283923481u);
    return h & uint(gc_counts.w - 1);
}

ivec3 gc_cell_of(vec3 p) {
    return ivec3(floor(p * gc_self.x));
}
@end

// ---- a step's start

@cs gc_begin_cs
@include_block gc_common
layout(binding=0) buffer gc_begin_pos { gc_point pos[]; };
layout(binding=1) buffer gc_begin_start { gc_point start[]; };
layout(binding=2) buffer gc_begin_vel { gc_point vel[]; };
layout(binding=3) readonly buffer gc_begin_pins { gc_pin pins[]; };
layout(local_size_x=128) in;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= gc_counts.x) return;
    vec4 p = pos[i].p;
    start[i].p = p;
    vec4 pin = pins[i].p;
    if (pin.w == 0.0) {
        pos[i].p = vec4(pin.xyz, 0.0);
        vel[i].p = vec4(0.0);
    }
}
@end

@cs gc_forces_cs
@include_block gc_common
layout(binding=0) readonly buffer gc_forces_pos { gc_point pos[]; };
layout(binding=1) readonly buffer gc_forces_vel { gc_point vel[]; };
layout(binding=2) buffer gc_forces_acc { gc_point acc[]; };
layout(local_size_x=128) in;

// -1..1, smooth across the cloth and in time (as Cloth's gust_at)
float gc_gust_at(int c, int r) {
    float x = gc_gust.z * float(c) / float(gc_counts.y - 1);
    float y = gc_gust.w * float(r) / float(gc_counts.z - 1);
    float t = gc_gust.y;
    return 0.6 * sin(t * 0.73 + x * 1.9 + y * 0.8) + 0.4 * sin(t * 1.37 + x * 0.6 + y * 2.3);
}

// the force of the air on triangle a, b, d (of cell c, r) on each of its points
vec3 gc_air(int a, int b, int d, int c, int r) {
    vec3 pa = pos[a].p.xyz;
    vec3 cr = cross(pos[b].p.xyz - pa, pos[d].p.xyz - pa);
    float len2 = dot(cr, cr);
    if (len2 < 1e-12) return vec3(0.0);
    vec3 air = gc_wind.xyz * (1.0 + gc_gust.x * gc_gust_at(c, r));
    vec3 rel = air - (vel[a].p.xyz + vel[b].p.xyz + vel[d].p.xyz) * (1.0 / 3.0);
    float area2 = sqrt(len2);
    vec3 n = cr / area2;
    float into = dot(n, rel);
    vec3 f = n * (gc_wind.w * area2 * into);
    // lift: across the flow, in the plane of the flow and the normal, as sin 2θ of the angle
    // the triangle meets the flow at
    float speed2 = dot(rel, rel);
    if (gc_mode.y > 0.0 && speed2 > 1e-8) {
        vec3 side = n * speed2 - rel * into;
        float side_len = length(side);
        if (side_len > 1e-8) {
            float along = length(rel - n * into);
            f += side * (gc_mode.y * gc_wind.w * area2 * into * along / (sqrt(speed2) * side_len));
        }
    }
    return f;
}

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= gc_counts.x) return;
    vec3 f = gc_gravity.xyz;
    if (gc_wind.w > 0.0) {
        int cols = gc_counts.y;
        int col = i % cols;
        int row = i / cols;
        // the two triangles of each cell around the point (i00, i01, i11) and (i00, i11, i10)
        for (int r = row - 1; r <= row; r++) {
            for (int c = col - 1; c <= col; c++) {
                // (a cell between two panels is none)
                if (c < 0 || r < 0 || c + 1 >= cols || r + 1 >= gc_counts.z || (c + 1) % gc_grid.x == 0) continue;
                int i00 = r * cols + c;
                int i10 = i00 + 1;
                int i01 = i00 + cols;
                int i11 = i01 + 1;
                if (i == i00 || i == i01 || i == i11) f += gc_air(i00, i01, i11, c, r);
                if (i == i00 || i == i11 || i == i10) f += gc_air(i00, i11, i10, c, r);
            }
        }
    }
    acc[i].p = vec4(f, 0.0);
}
@end

// ---- a substep

@cs gc_integrate_cs
@include_block gc_common
layout(binding=0) buffer gc_int_pos { gc_point pos[]; };
layout(binding=1) buffer gc_int_prev { gc_point prev[]; };
layout(binding=2) buffer gc_int_vel { gc_point vel[]; };
layout(binding=3) readonly buffer gc_int_acc { gc_point acc[]; };
layout(binding=4) readonly buffer gc_int_pins { gc_pin pins[]; };
layout(local_size_x=128) in;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= gc_counts.x) return;
    vec4 p = pos[i].p;
    prev[i].p = p;
    if (pins[i].p.w == 0.0) return;
    float h = gc_step.x;
    vec3 v = vel[i].p.xyz + acc[i].p.xyz * h;
    vel[i].p = vec4(v, 0.0);
    pos[i].p = vec4(p.xyz + v * h, 0.0);
}
@end

@cs gc_links_cs
@include_block gc_common
layout(binding=0) buffer gc_links_pos { gc_point pos[]; };
layout(binding=1) readonly buffer gc_links_links { gc_link links[]; };
layout(binding=2) readonly buffer gc_links_pins { gc_pin pins[]; };
layout(local_size_x=128) in;
void main() {
    int k = int(gl_GlobalInvocationID.x);
    if (k >= gc_range.y) return;
    gc_link l = links[gc_range.x + k];
    float wa = pins[l.a].p.w;
    float wb = pins[l.b].p.w;
    float sum = wa + wb + gc_step.z;
    if (sum <= 0.0) return;
    vec3 pa = pos[l.a].p.xyz;
    vec3 pb = pos[l.b].p.xyz;
    vec3 d = pa - pb;
    float d2 = dot(d, d);
    if (d2 < 1e-18) return;
    vec3 c = d * (1.0 - l.rest * inversesqrt(d2));
    pos[l.a].p = vec4(pa - c * (wa / sum), 0.0);
    pos[l.b].p = vec4(pb + c * (wb / sum), 0.0);
}
@end

// As Cloth's solve_bends: the gradient takes in how the normal turns as a, b, c and d move, so
// that the correction is a force's. Every point's gradient is along the normal, so each is a
// number: a's 1 - ge, b's 1 + ge, m's -2, c's -gf and d's gf.
@cs gc_bends_cs
@include_block gc_common
layout(binding=0) buffer gc_bends_pos { gc_point pos[]; };
layout(binding=1) readonly buffer gc_bends_bends { gc_bend bends[]; };
layout(binding=2) readonly buffer gc_bends_pins { gc_pin pins[]; };
layout(local_size_x=128) in;
void main() {
    int k = int(gl_GlobalInvocationID.x);
    if (k >= gc_range.y) return;
    gc_bend b = bends[gc_range.x + k];
    vec3 pa = pos[b.a].p.xyz;
    vec3 pb = pos[b.b].p.xyz;
    vec3 pm = pos[b.m].p.xyz;
    vec3 pc = pos[b.c].p.xyz;
    vec3 pd = pos[b.d].p.xyz;
    vec3 e = pb - pa;
    vec3 f = pd - pc;
    vec3 nn = cross(e, f) * b.s;
    float n2 = dot(nn, nn);
    if (n2 < 1e-24) return;
    float inv2 = 1.0 / n2;
    vec3 n = nn * sqrt(inv2);
    vec3 dv = pa + pb - pm * 2.0;
    float c = dot(dv, n) - b.rest;
    float ef = dot(e, f);
    float de = dot(dv, e);
    float df = dot(dv, f);
    float ge = (df * ef - de * dot(f, f)) * inv2;
    float gf = (de * ef - df * dot(e, e)) * inv2;
    float ga = 1.0 - ge;
    float gb = 1.0 + ge;
    float gm = -2.0;
    float gc = -gf;
    // (at an edge c or d is m)
    bool c_edge = b.c == b.m;
    bool d_edge = b.d == b.m;
    if (c_edge) gm += gc;
    if (d_edge) gm += gf;
    float wa = pins[b.a].p.w;
    float wb = pins[b.b].p.w;
    float wm = pins[b.m].p.w;
    float wc = pins[b.c].p.w;
    float wd = pins[b.d].p.w;
    float sum = wa * ga * ga + wb * gb * gb + wm * gm * gm + gc_step.z;
    if (!c_edge) sum += wc * gc * gc;
    if (!d_edge) sum += wd * gf * gf;
    if (sum <= 0.0) return;
    vec3 l = n * (-c / sum);
    pos[b.a].p = vec4(pa + l * (ga * wa), 0.0);
    pos[b.b].p = vec4(pb + l * (gb * wb), 0.0);
    pos[b.m].p = vec4(pm + l * (gm * wm), 0.0);
    if (!c_edge) pos[b.c].p = vec4(pc + l * (gc * wc), 0.0);
    if (!d_edge) pos[b.d].p = vec4(pd + l * (gf * wd), 0.0);
}
@end

@cs gc_tethers_cs
@include_block gc_common
layout(binding=0) buffer gc_teth_pos { gc_point pos[]; };
layout(binding=1) readonly buffer gc_teth_tethers { gc_tether tethers[]; };
layout(local_size_x=128) in;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= gc_counts.x) return;
    gc_tether t = tethers[i];
    if (t.pin == GC_NONE) return;
    vec3 p = pos[i].p.xyz;
    vec3 d = p - pos[t.pin].p.xyz;
    float most = t.rest * (1.0 + gc_self.z);
    float len2 = dot(d, d);
    if (len2 <= most * most) return;
    float len = sqrt(len2);
    pos[i].p = vec4(p - d * ((len - most) / len), 0.0);
}
@end

// ---- the spatial hash: cells[h] ends as the first entry of cell h in ids, cells[h + 1] past
// its last (cells has a cell more than the table, and room for the scan's blocks of 1024)

@cs gc_clear_cs
@include_block gc_common
layout(binding=0) buffer gc_clear_cells { gc_cell cells[]; };
layout(local_size_x=128) in;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i <= gc_counts.w) cells[i].n = 0u;
}
@end

@cs gc_count_cs
@include_block gc_common
layout(binding=0) readonly buffer gc_count_pos { gc_point pos[]; };
layout(binding=1) buffer gc_count_cells { gc_cell cells[]; };
layout(local_size_x=128) in;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= gc_counts.x) return;
    atomicAdd(cells[gc_hash(gc_cell_of(pos[i].p.xyz))].n, 1u);
}
@end

// Each group sums its 1024 cells in place (inclusive: a cell gets the counts up to it) and
// writes its total to sums.
@cs gc_scan_block_cs
@include_block gc_common
layout(binding=0) buffer gc_scanb_cells { gc_cell cells[]; };
layout(binding=1) buffer gc_scanb_sums { gc_cell sums[]; };
layout(local_size_x=256) in;
shared uint gc_part[256];
void main() {
    uint t = gl_LocalInvocationIndex;
    uint base = gl_WorkGroupID.x * 1024u + t * 4u;
    uint n = uint(gc_counts.w) + 1u;
    uint v0 = base < n ? cells[base].n : 0u;
    uint v1 = base + 1u < n ? cells[base + 1u].n : 0u;
    uint v2 = base + 2u < n ? cells[base + 2u].n : 0u;
    uint v3 = base + 3u < n ? cells[base + 3u].n : 0u;
    v1 += v0;
    v2 += v1;
    v3 += v2;
    gc_part[t] = v3;
    barrier();
    for (uint off = 1u; off < 256u; off *= 2u) {
        uint add = t >= off ? gc_part[t - off] : 0u;
        barrier();
        gc_part[t] += add;
        barrier();
    }
    uint before = gc_part[t] - v3;
    if (base < n) cells[base].n = v0 + before;
    if (base + 1u < n) cells[base + 1u].n = v1 + before;
    if (base + 2u < n) cells[base + 2u].n = v2 + before;
    if (base + 3u < n) cells[base + 3u].n = v3 + before;
    if (t == 255u) sums[gl_WorkGroupID.x].n = gc_part[255];
}
@end

// One group: each block's total becomes the sum of the blocks before it (up to 1024 blocks).
@cs gc_scan_sums_cs
@include_block gc_common
layout(binding=0) buffer gc_scans_sums { gc_cell sums[]; };
layout(local_size_x=256) in;
shared uint gc_part2[256];
void main() {
    uint t = gl_LocalInvocationIndex;
    uint blocks = (uint(gc_counts.w) + 1u + 1023u) / 1024u;
    uint base = t * 4u;
    uint v0 = base < blocks ? sums[base].n : 0u;
    uint v1 = base + 1u < blocks ? sums[base + 1u].n : 0u;
    uint v2 = base + 2u < blocks ? sums[base + 2u].n : 0u;
    uint v3 = base + 3u < blocks ? sums[base + 3u].n : 0u;
    uint s1 = v0 + v1;
    uint s2 = s1 + v2;
    uint s3 = s2 + v3;
    gc_part2[t] = s3;
    barrier();
    for (uint off = 1u; off < 256u; off *= 2u) {
        uint add = t >= off ? gc_part2[t - off] : 0u;
        barrier();
        gc_part2[t] += add;
        barrier();
    }
    uint before = gc_part2[t] - s3;
    if (base < blocks) sums[base].n = before;
    if (base + 1u < blocks) sums[base + 1u].n = before + v0;
    if (base + 2u < blocks) sums[base + 2u].n = before + s1;
    if (base + 3u < blocks) sums[base + 3u].n = before + s2;
}
@end

@cs gc_scan_add_cs
@include_block gc_common
layout(binding=0) buffer gc_scana_cells { gc_cell cells[]; };
layout(binding=1) readonly buffer gc_scana_sums { gc_cell sums[]; };
layout(local_size_x=256) in;
void main() {
    uint block = gl_WorkGroupID.x;
    if (block == 0u) return;
    uint add = sums[block].n;
    uint n = uint(gc_counts.w) + 1u;
    uint base = block * 1024u + gl_LocalInvocationIndex * 4u;
    for (uint k = 0u; k < 4u; k++) {
        if (base + k < n) cells[base + k].n += add;
    }
}
@end

// Each point takes a place in its cell, from the cell's end down: after all of them, cells[h]
// is the cell's first entry. (The order in a cell changes from run to run; gc_self_cs's result
// does not depend on it.)
@cs gc_fill_cs
@include_block gc_common
layout(binding=0) readonly buffer gc_fill_pos { gc_point pos[]; };
layout(binding=1) buffer gc_fill_cells { gc_cell cells[]; };
layout(binding=2) buffer gc_fill_ids { gc_cell ids[]; };
layout(local_size_x=128) in;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= gc_counts.x) return;
    uint h = gc_hash(gc_cell_of(pos[i].p.xyz));
    uint at = atomicAdd(cells[h].n, uint(-1)) - 1u;
    ids[at].n = uint(i);
}
@end

@cs gc_self_cs
@include_block gc_common
layout(binding=0) readonly buffer gc_self_pos { gc_point pos[]; };
layout(binding=1) readonly buffer gc_self_prev { gc_point prev[]; };
layout(binding=2) readonly buffer gc_self_rest { gc_point rest[]; };
layout(binding=3) readonly buffer gc_self_cells { gc_cell cells[]; };
layout(binding=4) readonly buffer gc_self_ids { gc_cell ids[]; };
layout(binding=5) readonly buffer gc_self_pins { gc_pin pins[]; };
layout(binding=6) buffer gc_self_push { gc_push push[]; };
layout(local_size_x=128) in;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= gc_counts.x) return;
    if (pins[i].p.w == 0.0) {
        push[i].d = ivec4(0);
        return;
    }
    vec3 p = pos[i].p.xyz;
    vec3 moved = p - prev[i].p.xyz;
    vec3 rp = rest[i].p.xyz;
    float dist = gc_sweep.w;
    ivec3 home = gc_cell_of(p);
    ivec3 sum = ivec3(0);
    ivec3 rub = ivec3(0);
    int touching = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                ivec3 cell = home + ivec3(dx, dy, dz);
                uint h = gc_hash(cell);
                uint first = cells[h].n;
                uint last = cells[h + 1u].n;
                for (uint k = first; k < last; k++) {
                    int j = int(ids[k].n);
                    if (j == i) continue;
                    vec3 q = pos[j].p.xyz;
                    // (another cell that hashes the same: met in its own turn, or not near)
                    if (gc_cell_of(q) != cell) continue;
                    vec3 d = p - q;
                    float d2 = dot(d, d);
                    if (d2 >= dist * dist || d2 < 1e-16) continue;
                    // (points nearer at rest than the distance are kept apart only that far)
                    vec3 r = rp - rest[j].p.xyz;
                    float most = min(dist, sqrt(dot(r, r)));
                    if (d2 >= most * most) continue;
                    float len = sqrt(d2);
                    vec3 n = d / len;
                    // half the overlap each way
                    sum += ivec3(round(n * (0.5 * gc_self.y * (most - len) * GC_FIXED)));
                    // friction: towards the other's move over the substep, along the contact
                    vec3 rel = (q - prev[j].p.xyz) - moved;
                    rel -= n * dot(rel, n);
                    rub += ivec3(round(rel * (0.5 * gc_self.w * GC_FIXED)));
                    touching++;
                }
            }
        }
    }
    if (touching > 0) sum += rub / touching;
    push[i].d = ivec4(sum, touching);
}
@end

// What the collide kernels share: `colliders` is declared before it.
@block gc_collide_fns
// the friction of the collider being looked at (set by the collide kernels' loops)
float gc_fr;

// Friction for a point pushed out of a collider from p to out_p (normal n), on its move along
// the surface over the substep (from q, where it was a substep ago): taken off whole while it
// is under gc_fr times how far the push went (static: it stays put), and past that by that much
// or gc_fr of it, whichever is more (sliding, slowed)
vec3 gc_rub(vec3 out_p, vec3 p, vec3 q, vec3 n) {
    vec3 moved = out_p - q;
    vec3 along = moved - n * dot(moved, n);
    float slide = length(along);
    float hold = gc_fr * max(dot(out_p - p, n), 0.0);
    if (slide <= hold) return out_p - along;
    return out_p - along * max(hold / slide, gc_fr);
}

// a box's center and axes at t of its way
vec3 gc_box_at(gc_collider k, float t, out vec3 ax[3]) {
    vec3 x = normalize(mix(k.from_x.xyz, k.x.xyz, t));
    vec3 y = mix(k.from_y.xyz, k.y.xyz, t);
    vec3 z = normalize(cross(x, y));
    ax[0] = x;
    ax[1] = cross(z, x);
    ax[2] = z;
    return mix(k.from_a.xyz, k.a.xyz, t);
}

// out of a box through the face the point came in by (where it was a substep ago, in the
// box as it was then), or the nearest face for a point that was already in
vec3 gc_box(gc_collider k, vec3 p, vec3 q, float th) {
    vec3 ax[3];
    vec3 was_ax[3];
    vec3 c = gc_box_at(k, gc_sweep.x, ax);
    vec3 was_c = gc_box_at(k, gc_sweep.y, was_ax);
    vec3 h = k.half_size.xyz + vec3(th);
    vec3 d = p - c;
    vec3 l = vec3(dot(d, ax[0]), dot(d, ax[1]), dot(d, ax[2]));
    if (abs(l.x) >= h.x || abs(l.y) >= h.y || abs(l.z) >= h.z) return p;
    vec3 e = q - was_c;
    vec3 was = vec3(dot(e, was_ax[0]), dot(e, was_ax[1]), dot(e, was_ax[2]));
    int face = -1;
    float side = 1.0;
    float least = 1e30;
    for (int j = 0; j < 3; j++) {
        if (abs(was[j]) < h[j]) continue;
        float depth = h[j] - abs(l[j]);
        if (depth < least) {
            least = depth;
            face = j;
            side = was[j] < 0.0 ? -1.0 : 1.0;
        }
    }
    if (face < 0) {
        for (int j = 0; j < 3; j++) {
            float depth = h[j] - abs(l[j]);
            if (depth < least) {
                least = depth;
                face = j;
                side = l[j] < 0.0 ? -1.0 : 1.0;
            }
        }
    }
    l[face] = side * h[face];
    vec3 out_p = c + ax[0] * l.x + ax[1] * l.y + ax[2] * l.z;
    return gc_rub(out_p, p, q, ax[face] * side);
}

// out of a sphere or capsule (the segment a..b, r with the thickness; was_a..was_b a substep
// ago): a point the collider has overtaken within the substep (past the axis, on the far side
// from where it was) goes back out the side it was on
vec3 gc_round(vec3 p, vec3 q, vec3 a, vec3 b, vec3 was_a, vec3 was_b, float r) {
    vec3 ab = b - a;
    float ab2 = dot(ab, ab);
    vec3 c = a;
    if (ab2 > 1e-12) c = a + ab * clamp(dot(p - a, ab) / ab2, 0.0, 1.0);
    vec3 d = p - c;
    float d2 = dot(d, d);
    if (d2 >= r * r) return p;
    float len = sqrt(d2);
    vec3 n = len > 1e-6 ? d / len : vec3(0.0, 0.0, 1.0);
    vec3 wab = was_b - was_a;
    float wab2 = dot(wab, wab);
    vec3 qa = was_a;
    if (wab2 > 1e-12) qa = was_a + wab * clamp(dot(q - was_a, wab) / wab2, 0.0, 1.0);
    vec3 e = q - qa;
    if (dot(e, e) >= r * r * 0.25 && dot(e, n) < 0.0) n = normalize(e);
    return gc_rub(c + n * r, p, q, n);
}

vec3 gc_plane(gc_collider k, vec3 p, vec3 q, float th) {
    vec3 n = k.x.xyz;
    vec3 o = mix(k.from_a.xyz, k.a.xyz, gc_sweep.x);
    float d = dot(p - o, n) - th;
    return d < 0.0 ? gc_rub(p - n * d, p, q, n) : p;
}

vec3 gc_floor(vec3 p, vec3 q) {
    if (gc_range.w == 0) return p;
    float y = gc_sweep.z + gc_gravity.w;
    if (p.y < y) {
        p.y = y;
        vec3 moved = p - q;
        p.x -= moved.x * gc_step.w;
        p.z -= moved.z * gc_step.w;
    }
    return p;
}
@end

@cs gc_collide_cs
@include_block gc_common
layout(binding=0) buffer gc_coll_pos { gc_point pos[]; };
layout(binding=1) readonly buffer gc_coll_prev { gc_point prev[]; };
layout(binding=2) buffer gc_coll_vel { gc_point vel[]; };
layout(binding=3) readonly buffer gc_coll_pins { gc_pin pins[]; };
layout(binding=4) readonly buffer gc_coll_push { gc_push push[]; };
layout(binding=5) readonly buffer gc_coll_colliders { gc_collider colliders[]; };
@include_block gc_collide_fns
layout(local_size_x=128) in;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= gc_counts.x) return;
    if (pins[i].p.w == 0.0) return;
    vec3 p = pos[i].p.xyz;
    vec3 q = prev[i].p.xyz;
    if (gc_sweep.w > 0.0) p += vec3(push[i].d.xyz) * (1.0 / GC_FIXED);
    float th = gc_gravity.w;
    for (int c = 0; c < gc_range.z; c++) {
        gc_collider k = colliders[c];
        if (k.info.z == 0.0) continue;
        gc_fr = k.mass.z < 0.0 ? gc_step.w : k.mass.z;
        int kind = int(k.info.x);
        if (kind == 2) {
            p = gc_box(k, p, q, th);
        } else if (kind == 3) {
            p = gc_plane(k, p, q, th);
        } else {
            p = gc_round(p, q, mix(k.from_a.xyz, k.a.xyz, gc_sweep.x), mix(k.from_b.xyz, k.b.xyz, gc_sweep.x),
                         mix(k.from_a.xyz, k.a.xyz, gc_sweep.y), mix(k.from_b.xyz, k.b.xyz, gc_sweep.y), k.info.y + th);
        }
    }
    p = gc_floor(p, q);
    pos[i].p = vec4(p, 0.0);
    vel[i].p = vec4((p - q) * gc_step.y, 0.0);
}
@end

// As gc_collide_cs, with bodies: a sphere or capsule with a mass is where it was moved plus its
// body's offset, and shares each push with the point by their inverse masses. What a group's
// points push its bodies back by is summed in a fixed order (a tree in shared memory) into
// partials[group * GC_BODIES + body], for gc_bodies_cs.
@cs gc_collide_bodies_cs
@include_block gc_common
layout(binding=0) buffer gc_collb_pos { gc_point pos[]; };
layout(binding=1) readonly buffer gc_collb_prev { gc_point prev[]; };
layout(binding=2) buffer gc_collb_vel { gc_point vel[]; };
layout(binding=3) readonly buffer gc_collb_pins { gc_pin pins[]; };
layout(binding=4) readonly buffer gc_collb_push { gc_push push[]; };
layout(binding=5) readonly buffer gc_collb_colliders { gc_collider colliders[]; };
layout(binding=6) readonly buffer gc_collb_bodies { gc_body bodies[]; };
layout(binding=7) buffer gc_collb_partials { gc_point partials[]; };
@include_block gc_collide_fns
layout(local_size_x=128) in;
shared vec4 gc_backs[GC_BODIES * 128];
void main() {
    int t = int(gl_LocalInvocationIndex);
    int i = int(gl_GlobalInvocationID.x);
    for (int b = 0; b < GC_BODIES; b++) gc_backs[b * 128 + t] = vec4(0.0);
    if (i < gc_counts.x && pins[i].p.w != 0.0) {
        float w = pins[i].p.w;
        vec3 p = pos[i].p.xyz;
        vec3 q = prev[i].p.xyz;
        if (gc_sweep.w > 0.0) p += vec3(push[i].d.xyz) * (1.0 / GC_FIXED);
        float th = gc_gravity.w;
        for (int c = 0; c < gc_range.z; c++) {
            gc_collider k = colliders[c];
            if (k.info.z == 0.0) continue;
            gc_fr = k.mass.z < 0.0 ? gc_step.w : k.mass.z;
            int kind = int(k.info.x);
            if (kind == 2) {
                p = gc_box(k, p, q, th);
            } else if (kind == 3) {
                p = gc_plane(k, p, q, th);
            } else {
                int body = int(k.info.w);
                vec3 off = vec3(0.0);
                vec3 was_off = vec3(0.0);
                if (body >= 0) {
                    off = bodies[body].off.xyz;
                    was_off = off - bodies[body].dv.xyz * gc_step.x;
                }
                vec3 to = gc_round(p, q, mix(k.from_a.xyz, k.a.xyz, gc_sweep.x) + off, mix(k.from_b.xyz, k.b.xyz, gc_sweep.x) + off,
                                   mix(k.from_a.xyz, k.a.xyz, gc_sweep.y) + was_off, mix(k.from_b.xyz, k.b.xyz, gc_sweep.y) + was_off,
                                   k.info.y + th);
                float share = body >= 0 ? k.mass.x / w : 0.0;
                vec3 moved = (to - p) * (1.0 / (1.0 + share));
                if (body >= 0) gc_backs[body * 128 + t] += vec4(moved * share, 0.0);
                p += moved;
            }
        }
        p = gc_floor(p, q);
        pos[i].p = vec4(p, 0.0);
        vel[i].p = vec4((p - q) * gc_step.y, 0.0);
    }
    barrier();
    for (int s = 64; s > 0; s >>= 1) {
        if (t < s) {
            for (int b = 0; b < GC_BODIES; b++) gc_backs[b * 128 + t] += gc_backs[b * 128 + t + s];
        }
        barrier();
    }
    if (t < GC_BODIES) partials[int(gl_WorkGroupID.x) * GC_BODIES + t].p = gc_backs[t * 128];
}
@end

// One thread per collider; those with a body (info.w), by gc_mode.w:
//   0, after each substep's collide: pushed back by the sum of the groups' partials (in
//      order), its velocity changed to match, where it is now noted (end), and carried on by
//      that change
//   1, at a step's start: where it was at the last one's end (begin)
//   2, at an update's end: less the correction the CPU applies to the body (corr_off, corr_dv),
//      marked with the update (gc_mode.xy)
//   3, at an update's start: a body put somewhere new (mass.y) starts over, still
@cs gc_bodies_cs
@include_block gc_common
layout(binding=0) readonly buffer gc_bod_colliders { gc_collider colliders[]; };
layout(binding=1) buffer gc_bod_bodies { gc_body bodies[]; };
layout(binding=2) readonly buffer gc_bod_partials { gc_point partials[]; };
layout(local_size_x=32) in;
void main() {
    int c = int(gl_GlobalInvocationID.x);
    if (c >= gc_range.z) return;
    gc_collider k = colliders[c];
    int body = int(k.info.w);
    if (body < 0) return;
    int mode = int(gc_mode.w);
    gc_body s = bodies[body];
    if (mode == 0) {
        vec3 back = vec3(0.0);
        int groups = (gc_counts.x + 127) / 128;
        for (int g = 0; g < groups; g++) back += partials[g * GC_BODIES + body].p.xyz;
        float h = gc_step.x;
        s.off.xyz -= back;
        s.dv.xyz -= back / h;
        s.end = vec4(mix(k.from_a.xyz, k.a.xyz, gc_sweep.x) + s.off.xyz, 0.0);
        s.off.xyz += s.dv.xyz * h;
    } else if (mode == 1) {
        s.begin = s.end;
    } else if (mode == 2) {
        s.off.xyz -= k.corr_off.xyz;
        s.dv.xyz -= k.corr_dv.xyz;
        s.mark = gc_mode;
    } else if (k.mass.y != 0.0) {
        s.off = vec4(0.0);
        s.dv = vec4(0.0);
        s.begin = vec4(k.a.xyz, 0.0);
        s.end = s.begin;
    }
    bodies[body] = s;
}
@end

// A mesh drawn where a collider is (GpuCloth.make_collider_mesh): its shape's vertices (around
// the collider's center) moved there; a body's between where it was at the last step's start
// and end, as the cloth's mesh is drawn.
@cs gc_shape_cs
@include_block gc_common
layout(binding=0) readonly buffer gc_shape_colliders { gc_collider colliders[]; };
layout(binding=1) readonly buffer gc_shape_bodies { gc_body bodies[]; };
layout(binding=2) readonly buffer gc_shape_template { gc_vertex shape[]; };
layout(binding=3) buffer gc_shape_verts { gc_vertex verts[]; };
layout(local_size_x=128) in;
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= gc_range.y) return;
    gc_collider k = colliders[gc_range.x];
    int body = int(k.info.w);
    vec3 center = k.a.xyz;
    if (body >= 0) center = mix(bodies[body].begin.xyz, bodies[body].end.xyz, gc_origin.w);
    gc_vertex v = shape[i];
    v.a.xyz += gc_origin.xyz + center;
    verts[i] = v;
}
@end

// Internal damping, per stretch link of one color: of the difference in velocity between its
// points, `share` (gc_mode.x) is taken off, split by their inverse masses.
@cs gc_damp_cs
@include_block gc_common
layout(binding=0) buffer gc_damp_vel { gc_point vel[]; };
layout(binding=1) readonly buffer gc_damp_links { gc_link links[]; };
layout(binding=2) readonly buffer gc_damp_pins { gc_pin pins[]; };
layout(local_size_x=128) in;
void main() {
    int k = int(gl_GlobalInvocationID.x);
    if (k >= gc_range.y) return;
    gc_link l = links[gc_range.x + k];
    float wa = pins[l.a].p.w;
    float wb = pins[l.b].p.w;
    float sum = wa + wb;
    if (sum <= 0.0) return;
    vec3 va = vel[l.a].p.xyz;
    vec3 vb = vel[l.b].p.xyz;
    vec3 dv = (vb - va) * gc_mode.x;
    vel[l.a].p = vec4(va + dv * (wa / sum), 0.0);
    vel[l.b].p = vec4(vb - dv * (wb / sum), 0.0);
}
@end

// ---- the mesh

@cs gc_write_cs
@include_block gc_common
layout(binding=0) readonly buffer gc_write_pos { gc_point pos[]; };
layout(binding=1) readonly buffer gc_write_start { gc_point start[]; };
layout(binding=2) buffer gc_write_verts { gc_vertex verts[]; };
layout(local_size_x=128) in;

vec3 gc_shown(int i) {
    vec3 s = start[i].p.xyz;
    return s + (pos[i].p.xyz - s) * gc_origin.w;
}

vec4 gc_cr(float t) {
    float t2 = t * t;
    float t3 = t2 * t;
    return vec4(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}

vec4 gc_cr_slope(float t) {
    float t2 = t * t;
    return vec4(-1.5 * t2 + 2.0 * t - 0.5, 4.5 * t2 - 5.0 * t, -4.5 * t2 + 4.0 * t + 0.5, 1.5 * t2 - t);
}

// vertex i of the mesh of the panel from column gc_grid.y, gc_grid.z vertices to a cell's side:
// on the Catmull-Rom surface through the points (through each point, smooth between them), its
// normal from the surface's slopes
void main() {
    int i = int(gl_GlobalInvocationID.x);
    int cols = gc_counts.y;
    int rows = gc_counts.z;
    int w = gc_grid.x;
    int d = gc_grid.z;
    int fw = (w - 1) * d + 1;
    int fh = (rows - 1) * d + 1;
    if (i >= fw * fh) return;
    int fk = i % fw;
    int fr = i / fw;
    int k0 = min(fk / d, w - 2);
    int r0 = min(fr / d, rows - 2);
    float tu = float(fk - k0 * d) / float(d);
    float tv = float(fr - r0 * d) / float(d);
    vec4 bu = gc_cr(tu);
    vec4 bv = gc_cr(tv);
    vec4 su = gc_cr_slope(tu);
    vec4 sv = gc_cr_slope(tv);
    vec3 p = vec3(0.0);
    vec3 du = vec3(0.0);
    vec3 dv = vec3(0.0);
    for (int y = 0; y < 4; y++) {
        int r = clamp(r0 - 1 + y, 0, rows - 1);
        for (int x = 0; x < 4; x++) {
            int k = clamp(k0 - 1 + x, 0, w - 1);
            vec3 q = gc_shown(r * cols + gc_grid.y + k);
            p += q * (bu[x] * bv[y]);
            du += q * (su[x] * bv[y]);
            dv += q * (bu[x] * sv[y]);
        }
    }
    // (the side cross(across, -down) faces)
    vec3 n = cross(du, -dv);
    float len = length(n);
    n = len > 1e-12 ? n / len : vec3(0.0, 0.0, 1.0);
    verts[i].a = vec4(gc_origin.xyz + p, n.x);
    vec4 b = verts[i].b;
    verts[i].b = vec4(n.y, n.z, b.z, b.w);
}
@end

@program gc_begin gc_begin_cs
@program gc_forces gc_forces_cs
@program gc_integrate gc_integrate_cs
@program gc_links gc_links_cs
@program gc_bends gc_bends_cs
@program gc_tethers gc_tethers_cs
@program gc_clear gc_clear_cs
@program gc_count gc_count_cs
@program gc_scan_block gc_scan_block_cs
@program gc_scan_sums gc_scan_sums_cs
@program gc_scan_add gc_scan_add_cs
@program gc_fill gc_fill_cs
@program gc_self gc_self_cs
@program gc_collide gc_collide_cs
@program gc_collide_bodies gc_collide_bodies_cs
@program gc_bodies gc_bodies_cs
@program gc_shape gc_shape_cs
@program gc_damp gc_damp_cs
@program gc_write gc_write_cs
