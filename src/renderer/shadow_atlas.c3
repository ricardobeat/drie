// Dynamic quadtree shadow atlas (AtlasPacker, AtlasSlot) and static-caster caching (CasterWalk).
// Packs all shadowed lights besides the primary spot light and cascaded sun into power-of-two
// DEPTH16 blocks (minimum ATLAS_MIN_BLOCK = 128 px) ordered by Morton Z-curve near (0, 0) so
// the atlas grows only to the occupied extent. Tracks static vs moving/animated casters to cache
// static depth in a companion cache texture and composite moving casters on top; smoothly fades
// out evicted lights when over budget. Connects to prepare_and_draw_shadows() and cube_shadows().
module drie;
import std::collections::list, std::collections::map, std::math, std::io, libc;
import sokol;

// ---- Quadtree block packer

// Minimum square shadow atlas block dimension, in texels.
const int ATLAS_MIN_BLOCK     = 128;
const int ATLAS_LEVELS @local = 9;

struct AtlasPacker @private {
    int root; // Quadtree side length in pixels (power of two >= ATLAS_MIN_BLOCK).
    List{uint}[ATLAS_LEVELS] free; // Free block keys per level (level 0 = root), encoded via atlas_key().
}

// Encodes block coordinates in ATLAS_MIN_BLOCK units: x in the upper 16 bits, y in the lower 16.
fn uint atlas_key(int x, int y) @private => (uint)(x / ATLAS_MIN_BLOCK) << 16 | (uint)(y / ATLAS_MIN_BLOCK);
fn int atlas_key_x(uint key) @private => (int)(key >> 16) * ATLAS_MIN_BLOCK;
fn int atlas_key_y(uint key) @private => (int)(key & 0xffff) * ATLAS_MIN_BLOCK;

// Computes the Morton (Z-order) index of an atlas block key by interleaving its x and y bits.
fn ulong atlas_z(uint key) @local {
    ulong z;
    for (int b = 0; b < 16; b++) {
        z |= (ulong)((key >> b) & 1) << (2 * b);
        z |= (ulong)((key >> (16 + b)) & 1) << (2 * b + 1);
    }
    return z;
}

// Resets the quadtree to a single free root block of `root` pixels (rounded down to a power of two).
fn void AtlasPacker.reset(&self, int root) {
    int r = ATLAS_MIN_BLOCK;
    while (r * 2 <= root && r < ATLAS_MIN_BLOCK << (ATLAS_LEVELS - 1)) r *= 2;
    self.root = r;
    foreach (&l : self.free) {
        if (!l.allocator) l.init(mem);
        l.clear();
    }
    self.free[0].push(atlas_key(0, 0));
}

// Returns the quadtree level for blocks of `size` pixels, or -1 if invalid/larger than root.
fn int AtlasPacker.level(&self, int size) {
    int level;
    for (int s = self.root; s > size; s /= 2) level++;
    return (self.root >> level) == size && size >= ATLAS_MIN_BLOCK ? level : -1;
}

// Allocates the lowest Z-order free block at `level`, splitting a parent block when necessary.
fn bool AtlasPacker.take(&self, int level, uint* out) {
    if (level < 0 || level >= ATLAS_LEVELS) return false;
    List{uint}* l =  &self.free[level];
    sz best       = -1;
    ulong best_z;
    foreach (i, k : *l) {
        ulong z = atlas_z(k);
        if (best < 0 || z < best_z) {
            best = (sz)i;
            best_z = z;
        }
    }
    if (best >= 0) {
        *out = (*l)[best];
        l.remove_at(best);
        return true;
    }
    uint parent;
    if (!self.take(level - 1, &parent)) return false;
    int half = (self.root >> level);
    int x    = atlas_key_x(parent);
    int y    = atlas_key_y(parent);
    l.push(atlas_key(x + half, y));
    l.push(atlas_key(x, y + half));
    l.push(atlas_key(x + half, y + half));
    *out = parent;
    return true;
}

// Allocates a square power-of-two block of `size` pixels; returns false when the atlas is full.
fn bool AtlasPacker.alloc(&self, int size, uint* out) => self.take(self.level(size), out);

// Frees block `key` of `size` pixels, coalescing four free sibling quarters back into their parent.
fn void AtlasPacker.release(&self, uint key, int size) {
    int level = self.level(size);
    if (level < 0) return;
    if (level == 0) {
        self.free[0].push(key);
        return;
    }
    // Check whether all three sibling quarters are already free so the parent can be merged.
    int side = size * 2;
    int px   = atlas_key_x(key) / side * side;
    int py   = atlas_key_y(key) / side * side;
    uint[4] quarter = {
        atlas_key(px, py),
        atlas_key(px + size, py),
        atlas_key(px, py + size),
        atlas_key(px + size, py + size)
    };
    List{uint}* l = &self.free[level];
    int found;
    foreach (q : quarter) {
        if (q == key) continue;
        foreach (k : *l) {
            if (k == q) {
                found++;
                break;
            }
        }
    }
    if (found < 3) {
        l.push(key);
        return;
    }
    for (sz i = l.len(); i > 0; i--) {
        uint k = (*l)[i - 1];
        foreach (q : quarter) {
            if (k == q) {
                l.remove_at(i - 1);
                break;
            }
        }
    }
    self.release(quarter[0], side);
}

// ---- Atlas slots and placement

// Single shadow map tile in the atlas (one spot/directional map or one cube/rect face).
struct AtlasMap @private {
    int x;
    int y;
    int w;
    int h;
    Mat4 view_proj; // Projection used when last drawn (lookups match this matrix).
    bool drawn; // True once rendered into its current atlas rectangle.
    bool empty; // True when zero casters rendered into this tile (shader skips lookup).
    int block; // Index into the owning AtlasSlot's block array.
    // Static-vs-dynamic split: maps containing moving casters copy static depth from the cache
    // texture and draw moving casters on top (`composed`).
    bool cached;
    bool cache_empty;
    bool composed;
    uint cache_used; // Frame when this tile's cache block was last composited.
    bool stale; // Invalidated while outside the camera frustum; redrawn when seen again.
}

// Per-tile work item scheduled by place_and_draw_atlas().
struct AtlasJob @local {
    int slot;
    int face;
    Mat4 vp;
    bool full; // Draw all casters directly into the atlas tile.
    bool cache; // Refresh static casters into the cache tile first.
    bool compose; // Copy static cache into the atlas tile, then draw moving casters over it.
    bool restore; // Copy static cache into the atlas tile (moving casters have left).
    bool plain; // Set if any caster fell back to a plain shader while compiling.
}

// Atlas allocation record for one light (bake lights carry `bake = true` and stay separate).
struct AtlasSlot @private {
    Light* light;
    bool bake;
    int size; // Block side in pixels (0 when unallocated).
    int maps; // Map count (1 spot/dir, 5 rect area, 6 point).
    int blocks; // Quadtree block count (rect area packs 4 half-height sides into 2 blocks).
    int target; // Hysteresis-stabilized desired block size.
    int asked; // Candidate size requested in recent frames.
    int asked_frames; // Consecutive frames requesting `asked`.
    float importance; // Screen coverage weighted by brightness at the camera.
    float fade; // Smooth [0, 1] strength fade during budget eviction and reallocation.
    uint seen; // Last engine frame this light requested atlas space.
    uint[6] block;
    uint[6] cache_block;
    bool[6] cache_has;
    AtlasMap[6] map;
}

const uint ATLAS_KEEP_FRAMES @local  = 120;  // Grace frames before reclaiming an unseen light's slot.
const int ATLAS_GROW_FRAMES @local   = 3;    // Consecutive frames before growing a light's block size.
const int ATLAS_SHRINK_FRAMES @local = 30;   // Consecutive frames before shrinking an auto-updated light.
const float ATLAS_FADE_STEP @local   = 0.125f; // Per-frame shadow intensity fade step on eviction/entry.

fn int atlas_map_count(Light* l) @local => l.light_type == RECT_AREA ? 5 : l.light_type == POINT ? 6 : 1;
fn int atlas_block_count(Light* l) @local => l.light_type == RECT_AREA ? 3 : l.light_type == POINT ? 6 : 1;

// Looks up the AtlasSlot for `l` in the current mode (`state.bake.on`), or null.
fn AtlasSlot* atlas_slot_of(Light* l) @private {
    bool bake = state.bake.on;
    foreach (&s : state.atlas_slots) {
        if (s.light == l && s.bake == bake) return s;
    }
    return null;
}

// Returns face `face`'s AtlasMap for `l`, or null if unallocated.
fn AtlasMap* atlas_map_of(Light* l, int face = 0) @private {
    if (!l.cast_shadow) return null;
    AtlasSlot* s = atlas_slot_of(l);
    if (!s || !s.size || face >= s.maps) return null;
    return &s.map[face];
}

// True when `l` has an allocated slot with at least one rendered face.
fn bool atlas_has_shadow(Light* l) @private {
    if (!l.cast_shadow) return false;
    AtlasSlot* s = atlas_slot_of(l);
    if (!s || !s.size) return false;
    for (int f = 0; f < s.maps; f++) {
        if (s.map[f].drawn) return true;
    }
    return false;
}

// Normalized atlas UV rect `{ u, v, w, h }` for `m`, or all zeros when empty/unrendered.
fn Vec4 atlas_rect(AtlasMap* m) @private {
    if (!m || !m.drawn || m.empty || !m.w || !state.atlas_width) return { 0, 0, 0, 0 };
    float w = (float)state.atlas_width;
    float h = (float)state.atlas_height;
    return { (float)m.x / w, (float)m.y / h, (float)m.w / w, (float)m.h / h };
}

// Reciprocal of the smaller tile dimension, in normalized tile UVs per texel.
fn float atlas_texel(AtlasMap* m) @private => m && m.w ? 1.0f / (float)math::min(m.w, m.h) : 1.0f;

// Nearest power of two to `size`, clamped to `[ATLAS_MIN_BLOCK, state.atlas_packer.root]`.
fn int atlas_pow2(int size) @local {
    int p = ATLAS_MIN_BLOCK;
    while (p < state.atlas_packer.root && p * 3 / 2 < size) p *= 2;
    return p;
}

fn float render_height() @local {
    RenderTarget* rt = state.renderer.render_target;
    return rt ? (float)rt.height : sapp::heightf();
}

// Computes desired block size (2 texels per screen pixel spanned, capped at `shadow.map_size`)
// and priority `importance` (screen coverage * attenuated intensity at the camera).
fn int atlas_wanted(Light* l, Camera* camera, bool bake, float* importance) @local {
    int most     = atlas_pow2(shadow_map_size(light_shadow(l).map_size));
    Vector3 at   = camera.matrix_world.get_position();
    Vector3 pos  = l.matrix_world.get_position();
    float dl     = (pos - at).length();
    float bright = math::max(l.intensity, 0.0f) / math::max(dl * dl, 1.0f);
    if (bake || l.light_type == DIRECTIONAL) {
        // Bake and secondary directional lights always take priority over local lights.
        *importance = 1e9f + l.intensity;
        return most;
    }
    LightBounds b = { pos, point_shadow_far(l) };
    if (l.light_type == SPOT) {
        b = spot_light_bounds((SpotLight*)l);
        if (b.radius <= 0) b = { pos, point_shadow_far(l) };
    }
    float d   = (b.center - at).length();
    float p11 = math::abs(camera.projection_matrix[1][1]);
    // Projected bounding-sphere radius in half-screen-heights.
    float cover;
    if (camera.camera_type != PERSPECTIVE) {
        cover = b.radius * p11;
    } else if (d <= b.radius * 1.05f) {
        cover = 4;
    } else {
        cover = b.radius * p11 / math::sqrt(d * d - b.radius * b.radius);
    }
    cover = math::min(cover, 4.0f);
    *importance = cover * (bright + 1e-6f);
    // Ray-traced rect lights only need coarse fallback atlas tiles for forward-shaded items.
    if (rt_traced(l)) most = math::min(most, RT_MAP_SIZE);
    return math::min(atlas_pow2((int)(cover * render_height() * 2)), most);
}

// True when `l` should receive atlas space this render (all shadowed lights except the primary
// spot light and sun). Cached lights (`auto_update == false`) keep their tiles even at intensity 0.
fn bool atlas_wants(Light* l, bool bake, SpotLight* caster) @local {
    if (!l.cast_shadow || !l.visible) return false;
    switch (l.light_type) {
        case SPOT:
            SpotLight* sl = (SpotLight*)l;
            if (l.intensity <= 0 && (sl.shadow.auto_update || bake)) return false;
            return bake || sl != caster;
        case POINT:
        case RECT_AREA:
            LightShadow* sh = light_shadow(l);
            return l.intensity > 0 || (!sh.auto_update && !bake);
        case DIRECTIONAL:
            return l.intensity > 0 && l != state.sun_caster;
        default:
            return false;
    }
}

// Releases all atlas and static-cache blocks owned by `s`.
fn void atlas_release(AtlasSlot* s) @local {
    if (!s.size) return;
    for (int b = 0; b < s.blocks; b++) {
        state.atlas_packer.release(s.block[b], s.size);
        if (s.cache_has[b]) state.cache_packer.release(s.cache_block[b], s.size);
        s.cache_has[b] = false;
    }
    s.size = 0;
    for (int f = 0; f < 6; f++) s.map[f] = { };
}

// Allocates `s.blocks` blocks of `size` pixels in `state.atlas_packer` and lays out `s.map[]`.
// Returns false (releasing any partial allocations) if the packer lacks room.
fn bool atlas_place(AtlasSlot* s, int size) @local {
    for (int b = 0; b < s.blocks; b++) {
        if (!state.atlas_packer.alloc(size, &s.block[b])) {
            for (int k = 0; k < b; k++) state.atlas_packer.release(s.block[k], size);
            return false;
        }
    }
    s.size = size;
    if (s.blocks == s.maps) {
        for (int f = 0; f < s.maps; f++) {
            s.map[f] = { .x = atlas_key_x(s.block[f]), .y = atlas_key_y(s.block[f]), .w = size, .h = size, .block = f };
        }
        return true;
    }
    // Rect area light: front face takes block 0; four half-height side faces share blocks 1 and 2.
    s.map[0] = { .x = atlas_key_x(s.block[0]), .y = atlas_key_y(s.block[0]), .w = size, .h = size };
    for (int f = 1; f < s.maps; f++) {
        uint b = s.block[(f + 1) / 2];
        s.map[f] = {
            .x = atlas_key_x(b),
            .y = atlas_key_y(b) + (f % 2 ? 0 : size / 2),
            .w = size,
            .h = size / 2,
            .block = (f + 1) / 2
        };
    }
    return true;
}

// ---- Visibility and priority

// Unprojects the 8 corners of NDC box `[-1, 1] x [-1, 1] x [z0, z1]` through `inv`.
fn Vector3[8] clip_box_corners(Mat4 inv, float z0, float z1) @local {
    Vector3[8] c;
    for (int i = 0; i < 8; i++) {
        Vec4 p = inv.mul_vec4({ i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? z1 : z0, 1 });
        c[i] = { p.x / p.w, p.y / p.w, p.z / p.w };
    }
    return c;
}

// Separating-plane overlap test between two frusta and their 8 world-space corners.
fn bool frusta_overlap(Frustum* a, Vector3[8]* ac, Frustum* b, Vector3[8]* bc) @local {
    for (int k = 0; k < 2; k++) {
        Frustum* f    = k ? b : a;
        Vector3[8]* c = k ? ac : bc;
        foreach (p : f.planes) {
            bool inside;
            foreach (v : *c) {
                if (p.x * v.x + p.y * v.y + p.z * v.z + p.w >= 0) {
                    inside = true;
                    break;
                }
            }
            if (!inside) return false;
        }
    }
    return true;
}

struct AtlasView @local {
    Frustum frustum;
    Mat4 inv;
    Mat4 proj;
    bool perspective;
    float near;
    float far;
}

fn AtlasView atlas_view_of(Camera* camera) @local {
    Mat4 vp = camera.projection_matrix.mul(camera.matrix_world_inverse);
    return {
        .frustum = frustum_from(vp),
        .inv = vp.invert(),
        .proj = camera.projection_matrix,
        .perspective = camera.camera_type == PERSPECTIVE,
        .near = camera.near,
        .far = camera.far,
    };
}

// True when a light frustum (`light_vp`) overlaps the camera frustum truncated to light reach.
fn bool atlas_seen(AtlasView* v, Mat4 light_vp, Light* l, Vector3 eye) @local {
    bool zo  = depth_zero_to_one();
    float z0 = zo ? 0 : -1;
    float z1 = 1;
    if (v.perspective) {
        float reach = (l.matrix_world.get_position() - eye).length() + point_shadow_far(l);
        float d     = math::min(v.far, math::max(reach, v.near * 2));
        // Clip-space depth of view-space distance `d`.
        z1 = (v.proj[3][2] - v.proj[2][2] * d) / d;
        z1 = math::min(z1, 1.0f);
    }
    Vector3[8] cam = clip_box_corners(v.inv, z0, z1);
    Frustum lf     = frustum_from(light_vp);
    Vector3[8] lc  = clip_box_corners(light_vp.invert(), z0, 1);
    return frusta_overlap(&v.frustum, &cam, &lf, &lc);
}

// Insertion-sorts `order` in descending order of `key[index]`.
fn void atlas_sort(List{int}* order, float[] key) @local {
    for (sz i = 1; i < order.len(); i++) {
        int v = (*order)[i];
        sz j  = i;
        while (j > 0 && key[(*order)[j - 1]] < key[v]) {
            (*order)[j] = (*order)[j - 1];
            j--;
        }
        (*order)[j] = v;
    }
}

// ---- Static and moving caster tracking

const uint CASTER_WALK_FRAMES @local      = 60;  // Interval between full scene-graph caster walks.
const uint CASTER_STILL_FRAMES @local     = 8;   // Frames a modified caster remains classified as moving.
const uint ATLAS_CACHE_KEEP_FRAMES @local = 120; // Idle frames before freeing an unused static cache block.

// Snapshot of a shadow-casting mesh tracked across frames by CasterWalk.
struct CasterRec @local {
    Mesh* mesh;
    Mat4 world;
    BufferGeometry* geometry;
    Material* material;
    uint version;
    int instances;
    Vec4 sphere; // World-space bounding sphere `{ cx, cy, cz, radius }`.
    uint moved; // Frame when transform, geometry, material, or instances last changed.
    bool animated; // Moves on GPU every frame (sway, position_node, or morph targets).
    bool instanced;
}

// Per-scene shadow caster change tracker (updated once per frame via walk_casters()).
// Collects bounding spheres of static casters that changed, appeared, vanished, or settled
// (`changes`) alongside lists of GPU-animated, recently moved, and dynamic-geometry casters.
struct CasterWalk @private {
    Scene* scene;
    uint frame;
    List{CasterRec} recs;
    List{CasterRec} last;
    List{bool} found;
    uint last_frame;
    List{Vec4} changes;
    uint full_frame;
    uint structure;
    uint dropped;
    sz moved_seen;
    List{int} animated;
    List{int} recent;
    List{int} watched;
}

// True when `m` deforms on the GPU every frame (vertex sway, position_node, or morph targets).
fn bool caster_animated(Mesh* m) @local {
    Material* mat = m.material;
    return mat.sway > 0 || mat.position_node.id != 0 || m.morph_targets > 0;
}

fn bool rec_moving(CasterRec* r, uint frame) @local => frame - r.moved < CASTER_STILL_FRAMES;

fn void walk_object(CasterWalk* w, Object3D* obj, uint frame) @local {
    if (!obj.visible) return;
    if (obj.kind == MESH) {
        Mesh* m           = (Mesh*)obj;
        BufferGeometry* g = m.geometry;
        bool caster       = g && m.material && m.material.visible && m.cast_shadow && !m.material.transparent
            && !m.is_line()
            && m.billboard == NO_BILLBOARD
            && (g.uploaded || g.indices.len() > 0);
        if (caster) walk_caster(w, m, frame);
    }
    // Walk all LOD children; individual tiles keep whichever LOD level they rendered.
    foreach (c : obj.children) walk_object(w, c, frame);
}

fn void walk_caster(CasterWalk* w, Mesh* m, uint frame) @local {
    // Guard against shared meshes visited twice in one scene graph.
    if (m.shadow_walk == w && m.shadow_walk_frame == frame) return;
    BufferGeometry* g = m.geometry;
    int i             = m.shadow_walk_at;
    bool known        = m.shadow_walk == w && m.shadow_walk_frame == w.last_frame && i >= 0 && i < w.last.len()
        && w.last[i].mesh == m;
    CasterRec rec;
    bool changed      = true;
    if (known) {
        w.found[i] = true;
        CasterRec* was = &w.last[i];
        changed = libc::memcmp(
            &was.world,
            &m.matrix_world,
            Mat4::size
        ) != 0 || was.geometry
            != g || was.material != m.material || was.version != g.version || was.instances != m.instance_uploads;
        if (!changed) rec = *was;
    }
    if (changed) {
        Vector3 center = g.bounds_center;
        float radius   = g.bounds_radius;
        if (m.instance_data.len() > 0) {
            center = m.instance_center;
            radius = m.instance_radius;
        }
        Vector3 wc = m.matrix_world.transform_point(center);
        float r    = radius * max_scale(m.matrix_world) + m.material.sway * 2;
        rec = {
            .mesh = m,
            .world = m.matrix_world,
            .geometry = g,
            .material = m.material,
            .version = g.version,
            .instances = m.instance_uploads,
            .sphere = { wc.x, wc.y, wc.z, r },
            .animated = caster_animated(m),
            .instanced = m.instance_data.len() > 0,
        };
    }
    if (known) {
        CasterRec* was = &w.last[i];
        rec.moved = changed ? frame : was.moved;
        bool was_moving = rec_moving(was, frame - 1);
        bool now_moving = rec_moving(&rec, frame);
        // Invalidate static caches when a caster starts moving (leaving cache) or settles back in.
        if (changed && !was_moving) w.changes.push(was.sphere);
        if (was_moving && !now_moving) w.changes.push(rec.sphere);
    } else {
        // Newly discovered caster starts static and invalidates overlapping cached tiles.
        rec.moved = frame - CASTER_STILL_FRAMES;
        w.changes.push(rec.sphere);
    }
    m.shadow_moving = rec_moving(&rec, frame);
    m.shadow_animated = rec.animated;
    m.shadow_walk = w;
    m.shadow_walk_frame = frame;
    m.shadow_walk_at = (int)w.recs.len();
    w.recs.push(rec);
}

// Updates and returns the CasterWalk for `scene` (at most once per engine frame). Performs a
// full scene walk on structure changes or every CASTER_WALK_FRAMES, and incremental updates via
// `moved_meshes` and dynamic geometries in between.
fn CasterWalk* walk_casters(Scene* scene) @private {
    uint frame         = state.frame;
    CasterWalk* w;
    CasterWalk* oldest = &state.caster_walks[0];
    foreach (&cw : state.caster_walks) {
        if (cw.scene == scene) w = cw;
        if (cw.frame < oldest.frame) oldest = cw;
    }
    bool fresh = !w;
    if (!w) {
        w = oldest;
        if (!w.recs.allocator) {
            w.recs.init(mem);
            w.last.init(mem);
            w.found.init(mem);
            w.changes.init(mem);
            w.animated.init(mem);
            w.watched.init(mem);
            w.recent.init(mem);
        }
        w.recs.clear();
        w.last.clear();
        w.scene = scene;
    } else if (w.frame == frame && w.recs.len()) {
        return w;
    }
    w.changes.clear();
    // Run a full walk periodically or on scene-graph/overflow changes; otherwise check movers only.
    bool full = fresh || frame - w.full_frame >= CASTER_WALK_FRAMES || w.structure != scene_structure_changes
        || w.dropped != moved_meshes_dropped
        || w.moved_seen > moved_meshes.len();
    if (full) {
        List{CasterRec} t = w.last;
        w.last = w.recs;
        w.recs = t;
        w.last_frame = w.frame;
        w.recs.clear();
        w.found.clear();
        for (sz i = 0; i < w.last.len(); i++) w.found.push(false);
        w.frame = frame;
        w.full_frame = frame;
        w.structure = scene_structure_changes;
        w.dropped = moved_meshes_dropped;
        w.moved_seen = moved_meshes.len();
        w.animated.clear();
        w.watched.clear();
        w.recent.clear();
        walk_object(w, scene, frame);
        // Removed static casters invalidate cached tiles covering their last bounding sphere.
        foreach (i, &was : w.last) {
            if (!w.found[i] && !rec_moving(was, frame - 1)) w.changes.push(was.sphere);
        }
        foreach (k, &r : w.recs) {
            if (r.animated) w.animated.push((int)k);
            if (rec_moving(r, frame)) w.recent.push((int)k);
            if (r.geometry.dynamic) w.watched.push((int)k);
        }
    } else {
        w.last_frame = w.frame;
        w.frame = frame;
        for (sz k = w.moved_seen; k < moved_meshes.len(); k++) refresh_caster(w, moved_meshes[k], frame);
        w.moved_seen = moved_meshes.len();
        foreach (k : w.watched) refresh_caster(w, w.recs[k].mesh, frame);
        // Casters that stayed still for CASTER_STILL_FRAMES transition back into static caches.
        for (sz k = w.recent.len(); k > 0; k--) {
            CasterRec* r = &w.recs[w.recent[k - 1]];
            if (rec_moving(r, frame)) continue;
            w.changes.push(r.sphere);
            r.mesh.shadow_moving = false;
            w.recent.remove_at(k - 1);
        }
    }
    return w;
}

// Incrementally updates one tracked caster between full walks.
fn void refresh_caster(CasterWalk* w, Mesh* m, uint frame) @local {
    int i = m.shadow_walk_at;
    if (m.shadow_walk != w || i < 0 || i >= w.recs.len() || w.recs[i].mesh != m) return;
    CasterRec* r      = &w.recs[i];
    BufferGeometry* g = m.geometry;
    if (r.moved == frame) return;
    bool changed = libc::memcmp(
        &r.world,
        &m.matrix_world,
        Mat4::size
    ) != 0 || r.geometry
        != g || r.material != m.material || r.version != g.version || r.instances != m.instance_uploads;
    if (!changed) return;
    // Evict from static caches on first movement frame and track in `recent`.
    if (!rec_moving(r, frame - 1)) {
        w.changes.push(r.sphere);
        w.recent.push(i);
    }
    Vector3 center = g.bounds_center;
    float radius   = g.bounds_radius;
    if (m.instance_data.len() > 0) {
        center = m.instance_center;
        radius = m.instance_radius;
    }
    Vector3 wc = m.matrix_world.transform_point(center);
    r.world = m.matrix_world;
    r.geometry = g;
    r.material = m.material;
    r.version = g.version;
    r.instances = m.instance_uploads;
    r.sphere = { wc.x, wc.y, wc.z, radius * max_scale(m.matrix_world) + m.material.sway * 2 };
    r.moved = frame;
    m.shadow_moving = true;
}

fn bool sphere_in(Frustum* f, Vec4 s) @local => f.contains_sphere({ s.x, s.y, s.z }, s.w);

fn bool moving_in(CasterWalk* w, List{int}* near, Frustum* f) @local {
    foreach (i : *near) {
        if (sphere_in(f, w.recs[i].sphere)) return true;
    }
    return false;
}

// Gathers indices into `w.recs` of moving casters within reach of `l` (includes GPU-animated
// casters when `shadow.auto_update` is enabled).
fn void movers_near(CasterWalk* w, Light* l, List{int}* near) @local {
    LightShadow* sh = light_shadow(l);
    bool everywhere = l.light_type == DIRECTIONAL;
    Vector3 pos     = l.matrix_world.get_position();
    float reach     = l.light_type == SPOT ? spot_shadow_far((SpotLight*)l) : point_shadow_far(l);
    // Cached lights (`auto_update == false`) bake animated foliage into their static cache.
    for (int pass = 0; pass < (sh.auto_update ? 2 : 1); pass++) {
        foreach (i : pass == 0 ? w.recent : w.animated) {
            CasterRec* r = &w.recs[i];
            if (pass == 1 && w.recs[i].moved + CASTER_STILL_FRAMES > w.frame) continue;
            if (sh.skip_instanced && r.instanced) continue;
            if (!everywhere) {
                Vector3 c = { r.sphere.x, r.sphere.y, r.sphere.z };
                float d   = reach + r.sphere.w;
                if ((c - pos).length_sq() > d * d) continue;
            }
            near.push(i);
        }
    }
}

fn bool changed_in(CasterWalk* w, Frustum* f) @local {
    foreach (s : w.changes) {
        if (sphere_in(f, s)) return true;
    }
    return false;
}

fn bool caster_moves(Mesh* m, bool every) @local => m.shadow_moving || (every && m.shadow_animated);

// ---- Static depth cache and atlas update

fn bool atlas_cache_ready(AtlasSlot* s, AtlasMap* m) @local => s.cache_has[m.block];

fn bool atlas_cache_place(AtlasSlot* s, AtlasMap* m) @local {
    if (!state.cache_packer.alloc(s.size, &s.cache_block[m.block])) return false;
    s.cache_has[m.block] = true;
    for (int f = 0; f < s.maps; f++) {
        if (s.map[f].block == m.block) s.map[f].cached = false;
    }
    m.cache_used = state.frame;
    return true;
}

fn int[2] atlas_cache_at(AtlasSlot* s, AtlasMap* m) @local {
    uint c = s.cache_block[m.block];
    uint b = s.block[m.block];
    return { atlas_key_x(c) + m.x - atlas_key_x(b), atlas_key_y(c) + m.y - atlas_key_y(b) };
}

// Reclaims cache blocks whose tiles have not composited moving casters for ATLAS_CACHE_KEEP_FRAMES.
fn void atlas_cache_trim(AtlasSlot* s) @local {
    for (int b = 0; b < s.blocks; b++) {
        if (!s.cache_has[b]) continue;
        bool used;
        for (int f = 0; f < s.maps; f++) {
            AtlasMap* m = &s.map[f];
            if (m.block == b && (m.composed || state.frame - m.cache_used <= ATLAS_CACHE_KEEP_FRAMES)) used = true;
        }
        if (used) continue;
        state.cache_packer.release(s.cache_block[b], s.size);
        s.cache_has[b] = false;
        for (int f = 0; f < s.maps; f++) {
            if (s.map[f].block == b) s.map[f].cached = false;
        }
    }
}

Light* atlas_tally_light @local;
uint atlas_tally_frame @local;
int atlas_tally @local;

// Active mover list passed into draw_atlas_casters() when compositing (`which == 2`).
CasterWalk* atlas_movers @local;
List{int}* atlas_movers_near @local;

// Renders casters for `face` of `l` into the active atlas or cache viewport: `which == 0` draws
// all casters, `1` draws static casters only, `2` draws moving casters only. Returns draw count.
fn int draw_atlas_casters(Scene* scene, Camera* camera, Light* l, int face, int size, Mat4 vp, int which) @private {
    LightShadow* sh = light_shadow(l);
    Frustum fr      = frustum_from(vp);
    state.casters.clear();
    state.skip_instanced = sh.skip_instanced;
    if (which == 2 && atlas_movers) {
        // Test only the light's nearby moving casters instead of traversing the entire scene.
        Vector3 from = camera.matrix_world.get_position();
        foreach (k : *atlas_movers_near) {
            Mesh* mover = atlas_movers.recs[k].mesh;
            sz first    = state.casters.len();
            collect(mover, camera.matrix_world_inverse, &fr, from, true);
            // Exclude children walked independently as their own CasterRec entries.
            for (sz j = state.casters.len(); j > first; j--) {
                if (state.casters[j - 1].mesh != mover) state.casters.remove_at(j - 1);
            }
        }
    } else {
        collect_casters(scene, camera, &fr);
    }
    state.skip_instanced = false;
    state.shadow_plain = false;
    bool every        = sh.auto_update;
    bool from_light   = l.light_type == SPOT || casts_cube(l);
    Mat4 light_rel    = from_light ? light_view_proj(l, camera, true, face, size) : vp;
    Vector3 light_pos = from_light ? l.matrix_world.get_position() : (Vector3) { 0, 0, 0 };
    int drawn;
    RenderInfo* info = &state.renderer.info;
    int before = info.shadow_triangles;
    atlas_drawing = true;
    atlas_near = from_light ? math::max(sh.near, 0.01f) : 0;
    atlas_far = 0;
    if (from_light) atlas_far = l.light_type == SPOT ? spot_shadow_far((SpotLight*)l) : point_shadow_far(l);
    foreach (&item : state.casters) {
        if (which == 1 && caster_moves(item.mesh, every)) continue;
        if (which == 2 && !caster_moves(item.mesh, every)) continue;
        draw_shadow_item(item, light_rel, light_pos);
        drawn++;
    }
    atlas_drawing = false;
    info.atlas_casters += drawn;
    // the light that cost the most (a light's faces are drawn one after another)
    if (l != atlas_tally_light || atlas_tally_frame != state.frame) {
        atlas_tally_light = l;
        atlas_tally_frame = state.frame;
        atlas_tally = 0;
    }
    atlas_tally += info.shadow_triangles - before;
    if (atlas_tally > info.shadow_top_triangles) {
        info.shadow_top_light = l;
        info.shadow_top_triangles = atlas_tally;
    }
    return drawn;
}

// Returns `{ x, y, width, height }` in atlas pixels for face `face` of `l`'s shadow map, or all
// zeros if `l` has no rendered non-empty tile.
fn int[4] Renderer.shadow_map_rect(&self, Light* l, int face = 0) {
    AtlasMap* m = atlas_map_of(l, face);
    if (!m || !m.drawn || m.empty) return { 0, 0, 0, 0 };
    return { m.x, m.y, m.w, m.h };
}

// Allocates, resizes, and renders shadow atlas tiles for all active atlas lights (or bake lights
// when `bake` is true, keeping main-render slots untouched).
fn void update_shadow_atlas(Scene* scene, Camera* camera, bool bake = false, SpotLight* caster = null) @private {
    @pool() {
        place_and_draw_atlas(scene, camera, bake, caster);
    };
}

fn void place_and_draw_atlas(Scene* scene, Camera* camera, bool bake, SpotLight* caster) @local {
    RenderInfo* info = &state.renderer.info;
    int most         = state.renderer.shadow_atlas_size > 0 ? state.renderer.shadow_atlas_size
        : DEFAULT_SHADOW_ATLAS_SIZE;
    int limit        = sg::query_limits().max_image_size_2d;
    if (limit > 0) most = math::min(most, limit);
    int root = ATLAS_MIN_BLOCK;
    while (root * 2 <= most) root *= 2;
    if (state.atlas_packer.root != root) {
        state.atlas_slots.clear();
        state.atlas_packer.reset(root);
        state.cache_packer.reset(root);
    }

    // 1. Update active slots, hysteresis target sizes, and reclaim idle slots.
    uint frame = state.frame;
    foreach (l : state.lights) {
        if (!atlas_wants(l, bake, caster)) continue;
        AtlasSlot* s = atlas_slot_of(l);
        if (!s) {
            state.atlas_slots.push(
                { .light = l, .bake = bake, .maps = atlas_map_count(l), .blocks = atlas_block_count(l), .fade = 1 }
            );
            s = &state.atlas_slots[state.atlas_slots.len() - 1];
        }
        s.seen = frame;
        int want = atlas_wanted(l, camera, bake, &s.importance);
        if (!s.target || bake) {
            s.target = want;
        } else if (want == s.target) {
            s.asked_frames = 0;
        } else {
            if (want != s.asked) s.asked_frames = 0;
            s.asked = want;
            s.asked_frames++;
            bool cached = !light_shadow(l).auto_update;
            if ((want > s.target && s.asked_frames >= ATLAS_GROW_FRAMES)
                || (want < s.target && !cached && s.asked_frames >= ATLAS_SHRINK_FRAMES)) {
                s.target = want;
                s.asked_frames = 0;
            }
        }
    }
    for (sz i = state.atlas_slots.len(); i > 0; i--) {
        AtlasSlot* s = &state.atlas_slots[i - 1];
        if (s.bake != bake) continue;
        if (frame - s.seen > ATLAS_KEEP_FRAMES) {
            atlas_release(s);
            state.atlas_slots.remove_at(i - 1);
        }
    }

    // 2. Fit requested tiles into total pixel budget by halving/dropping lowest-importance lights.
    sz n = state.atlas_slots.len();
    List{int} order;
    order.tinit();
    float[] importance = mem::temp_array(float, n);
    int[] give         = mem::temp_array(int, n);
    long budget        = (long)root * root;
    long area;
    long others;
    foreach (i, &s : state.atlas_slots) {
        importance[i] = s.importance;
        if (s.seen == frame && s.bake == bake) {
            order.push((int)i);
            give[i] = s.target;
            area += (long)s.blocks * s.target * s.target;
        } else {
            others += (long)s.blocks * s.size * s.size;
        }
    }
    // Evict inactive lights from other renders/bakes first when over budget.
    if (area + others > budget) {
        foreach (&s : state.atlas_slots) {
            if (s.seen != frame || s.bake != bake) atlas_release(s);
        }
        others = 0;
    }
    area += others;
    atlas_sort(&order, importance);
    while (area > budget) {
        bool smaller;
        for (sz k = order.len(); k > 0 && area > budget; k--) {
            int i = order[k - 1];
            if (give[i] <= ATLAS_MIN_BLOCK) continue;
            long blocks = state.atlas_slots[i].blocks;
            area -= blocks * give[i] * give[i] * 3 / 4;
            give[i] /= 2;
            smaller = true;
        }
        if (!smaller) break;
    }
    for (sz k = order.len(); k > 0 && area > budget; k--) {
        int i = order[k - 1];
        area -= (long)state.atlas_slots[i].blocks * give[i] * give[i];
        give[i] = 0;
    }
    // Fade evicted lights out smoothly before releasing their tiles, and fade returning lights in.
    bool fading;
    foreach (i : order) {
        AtlasSlot* s = &state.atlas_slots[i];
        if (give[i]) {
            if (!s.size && s.fade <= 0) {
                s.fade = ATLAS_FADE_STEP;
            } else if (s.size) {
                s.fade = math::min(s.fade + ATLAS_FADE_STEP, 1.0f);
            }
            continue;
        }
        info.atlas_dropped++;
        if (s.size && s.fade > ATLAS_FADE_STEP) {
            s.fade -= ATLAS_FADE_STEP;
            give[i] = s.size;
            fading = true;
        } else {
            s.fade = 0;
        }
    }

    // 3. Reallocate changed slots in the quadtree (largest first); repack all on fragmentation.
    List{int} moving;
    moving.tinit();
    foreach (i : order) {
        AtlasSlot* s = &state.atlas_slots[i];
        if (give[i] == s.size) continue;
        atlas_release(s);
        if (give[i]) moving.push(i);
    }
    float[] by_size = mem::temp_array(float, n);
    for (sz i = 0; i < n; i++) by_size[i] = (float)give[i];
    atlas_sort(&moving, by_size);
    bool full;
    foreach (i : moving) {
        if (!atlas_place(&state.atlas_slots[i], give[i])) full = true;
    }
    if (full && !fading) {
        foreach (&s : state.atlas_slots) atlas_release(s);
        state.atlas_packer.reset(root);
        List{int} all_order;
        all_order.tinit();
        foreach (i : order) {
            if (give[i]) all_order.push(i);
        }
        atlas_sort(&all_order, by_size);
        foreach (i : all_order) atlas_place(&state.atlas_slots[i], give[i]);
    }

    // 4. Grow atlas texture (and VSM moments) to cover all occupied blocks.
    int width  = math::max(state.atlas_width, ATLAS_MIN_BLOCK);
    int height = math::max(state.atlas_height, ATLAS_MIN_BLOCK);
    foreach (&s : state.atlas_slots) {
        for (int f = 0; f < s.maps && s.size; f++) {
            while (width < s.map[f].x + s.map[f].w) width *= 2;
            while (height < s.map[f].y + s.map[f].h) height *= 2;
        }
    }
    ensure_shadow_atlas(width, height);
    bool all;
    bool vsm = vsm_on();
    if (vsm && ensure_vsm(&state.vsm_atlas, width, height)) all = true;
    if (all) {
        foreach (&s : state.atlas_slots) {
            for (int f = 0; f < 6; f++) s.map[f].drawn = false;
        }
    }

    // 5. Classify dirty/moving tiles into full, cache+compose, or restore jobs.
    CasterWalk* walk   = bake ? null : walk_casters(scene);
    AtlasView view     = atlas_view_of(camera);
    Vector3 eye        = camera.matrix_world.get_position();
    List{AtlasJob} jobs;
    List{int}[] movers = mem::temp_array(List{int}, state.atlas_slots.len());
    jobs.tinit();
    foreach (i : order) {
        AtlasSlot* s = &state.atlas_slots[i];
        if (!s.size) continue;
        Light* l        = s.light;
        LightShadow* sh = light_shadow(l);
        List{int}* near = &movers[i];
        near.tinit();
        if (walk) movers_near(walk, l, near);
        for (int f = 0; f < s.maps; f++) {
            AtlasMap* m = &s.map[f];
            Mat4 vp     = light_view_proj(l, camera, false, f, m.w);
            if (bake) {
                jobs.push({ .slot = i, .face = f, .vp = vp, .full = true });
                continue;
            }
            Frustum fr       = frustum_from(vp);
            bool light_moved = m.drawn && libc::memcmp(&vp, &m.view_proj, Mat4::size) != 0;
            bool dirty       = !m.drawn || m.stale || sh.needs_update || light_moved || changed_in(walk, &fr);
            bool moves       = near.len() > 0 && moving_in(walk, near, &fr);
            if (!dirty && !moves && !m.composed) continue;
            // Defer offscreen redraws for already-rendered tiles unless explicitly requested.
            if (m.drawn && !sh.needs_update && !atlas_seen(&view, vp, l, eye)) {
                if (dirty) m.stale = true;
                continue;
            }
            AtlasJob job = { .slot = i, .face = f, .vp = vp };
            if (moves && (atlas_cache_ready(s, m) || atlas_cache_place(s, m))) {
                job.cache = dirty || !m.cached;
                job.compose = true;
            } else if (dirty || moves) {
                job.full = true;
            } else {
                job.restore = true;
            }
            jobs.push(job);
        }
    }
    foreach (&s : state.atlas_slots) {
        if (s.size) atlas_cache_trim(s);
    }
    if (!jobs.len()) return;
    bool top_left = depth_zero_to_one();

    // 6. Static caster cache pass.
    int cache_width  = math::max(state.cache_width, ATLAS_MIN_BLOCK);
    int cache_height = math::max(state.cache_height, ATLAS_MIN_BLOCK);
    bool caching;
    foreach (&s : state.atlas_slots) {
        for (int b = 0; b < s.blocks && s.size; b++) {
            if (!s.cache_has[b]) continue;
            caching = true;
            while (cache_width < atlas_key_x(s.cache_block[b]) + s.size) cache_width *= 2;
            while (cache_height < atlas_key_y(s.cache_block[b]) + s.size) cache_height *= 2;
        }
    }
    if (caching) ensure_atlas_cache(cache_width, cache_height);
    bool cache_pass;
    foreach (&job : jobs) {
        if (!job.cache) continue;
        if (!cache_pass) {
            profile::begin_pass(
                &&{
                    .action = { .depth = { .load_action = LOAD, .store_action = STORE, .clear_value = 1 } },
                    .attachments = { .depth_stencil = state.cache_attachment },
                    .label = "drie-shadow-cache",
                }
            );
            cache_pass = true;
        }
        AtlasSlot* s    = &state.atlas_slots[job.slot];
        AtlasMap* m     = &s.map[job.face];
        LightShadow* sh = light_shadow(s.light);
        int[2] at       = atlas_cache_at(s, m);
        sg::apply_viewport(at[0], at[1], m.w, m.h, top_left);
        sg::apply_scissor_rect(at[0], at[1], m.w, m.h, top_left);
        clear_atlas_tile();
        int drawn = draw_atlas_casters(scene, camera, s.light, job.face, m.w, job.vp, 1);
        m.cached = true;
        m.cache_empty = drawn == 0;
        info.atlas_tiles++;
        if (state.shadow_plain) job.plain = true;
    }
    if (cache_pass) sg::end_pass();

    // 7. Atlas depth pass (full draws or cache copy + dynamic casters) and optional VSM blur.
    VsmTile[] tiles = mem::temp_array(VsmTile, jobs.len());
    int ntiles;
    profile::begin_pass(
        &&{
            .action = { .depth = { .load_action = all ? CLEAR : LOAD, .store_action = STORE, .clear_value = 1 } },
            .attachments = { .depth_stencil = state.atlas_attachment },
            .label = "drie-shadow-atlas",
        }
    );
    bool plain;
    foreach (k, &job : jobs) {
        AtlasSlot* s    = &state.atlas_slots[job.slot];
        AtlasMap* m     = &s.map[job.face];
        Light* l        = s.light;
        LightShadow* sh = light_shadow(l);
        m.view_proj = job.vp;
        m.drawn = true;
        m.stale = false;
        sg::apply_viewport(m.x, m.y, m.w, m.h, top_left);
        sg::apply_scissor_rect(m.x, m.y, m.w, m.h, top_left);
        if (job.full) {
            if (!all) clear_atlas_tile();
            int drawn = draw_atlas_casters(scene, camera, l, job.face, m.w, job.vp, 0);
            m.empty = drawn == 0;
            m.composed = false;
            // Full draw bypasses the static cache, so mark any existing cache block out of sync.
            m.cached = false;
        } else {
            int[2] at = atlas_cache_at(s, m);
            copy_depth(state.cache_view, at[0], at[1], m.x, m.y, m.w, m.h);
            int drawn;
            if (job.compose) {
                atlas_movers = walk;
                atlas_movers_near = &movers[job.slot];
                drawn = draw_atlas_casters(scene, camera, l, job.face, m.w, job.vp, 2);
                atlas_movers = null;
                m.cache_used = state.frame;
            }
            m.composed = job.compose;
            m.empty = m.cache_empty && drawn == 0;
        }
        info.atlas_tiles++;
        if (state.shadow_plain || job.plain) plain = true;
        if (!m.empty) tiles[ntiles++] = { m.x, m.y, m.w, m.h, sh.radius, sh.blur_samples };
        // Re-schedule next frame if any caster used a plain fallback while compiling.
        if (k + 1 == jobs.len() || jobs[k + 1].slot != job.slot) {
            sh.needs_update = plain;
            plain = false;
        }
    }
    sg::end_pass();
    state.casters.clear();
    if (vsm) vsm_blur(&state.vsm_atlas, state.atlas_view, tiles[:ntiles], !all);
}