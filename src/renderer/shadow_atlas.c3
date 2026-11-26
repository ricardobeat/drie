// The shadow atlas's space, handed out as a quadtree of square blocks: a block is a quarter of
// one twice its size, and four free quarters make their parent free again. A block goes to the
// free place nearest the atlas's corner (in Z order), so the part in use stays packed there and
// the texture only needs to be as large as that part.
module drie;
import std::collections::list, std::collections::map, std::math, std::io, libc;
import sokol;

// The smallest block, in pixels.
const int ATLAS_MIN_BLOCK = 128;
const int ATLAS_LEVELS @local = 9;

struct AtlasPacker @private {
    // the side of the whole quadtree (a power of two, ATLAS_MIN_BLOCK or more)
    int root;
    // the free blocks of each level (0: the root), as atlas_key()s
    List{uint}[ATLAS_LEVELS] free;
}

// A block's place, in ATLAS_MIN_BLOCK steps: x in the high half, y in the low one.
fn uint atlas_key(int x, int y) @private => (uint)(x / ATLAS_MIN_BLOCK) << 16 | (uint)(y / ATLAS_MIN_BLOCK);
fn int atlas_key_x(uint key) @private => (int)(key >> 16) * ATLAS_MIN_BLOCK;
fn int atlas_key_y(uint key) @private => (int)(key & 0xffff) * ATLAS_MIN_BLOCK;

// Z order of a key: its x and y bits interleaved.
fn ulong atlas_z(uint key) @local {
    ulong z;
    for (int b = 0; b < 16; b++) {
        z |= (ulong)((key >> b) & 1) << (2 * b);
        z |= (ulong)((key >> (16 + b)) & 1) << (2 * b + 1);
    }
    return z;
}

// Empties the packer, `root` pixels on a side (rounded down to a power of two and to the levels
// it has).
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

// The level of blocks `size` pixels on a side (-1: larger than the root).
fn int AtlasPacker.level(&self, int size) {
    int level;
    for (int s = self.root; s > size; s /= 2) level++;
    return (self.root >> level) == size && size >= ATLAS_MIN_BLOCK ? level : -1;
}

// A free block of `level`: the one first in Z order, or a quarter of one a level up.
fn bool AtlasPacker.take(&self, int level, uint* out) {
    if (level < 0 || level >= ATLAS_LEVELS) return false;
    List{uint}* l = &self.free[level];
    sz best = -1;
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
    int x = atlas_key_x(parent);
    int y = atlas_key_y(parent);
    l.push(atlas_key(x + half, y));
    l.push(atlas_key(x, y + half));
    l.push(atlas_key(x + half, y + half));
    *out = parent;
    return true;
}

// A block `size` pixels on a side (a power of two, ATLAS_MIN_BLOCK or more): false when the
// atlas has no room for it.
fn bool AtlasPacker.alloc(&self, int size, uint* out) => self.take(self.level(size), out);

// Gives back block `key`, `size` pixels on a side.
fn void AtlasPacker.release(&self, uint key, int size) {
    int level = self.level(size);
    if (level < 0) return;
    if (level == 0) {
        self.free[0].push(key);
        return;
    }
    // its three siblings free too: their parent is
    int side = size * 2;
    int px = atlas_key_x(key) / side * side;
    int py = atlas_key_y(key) / side * side;
    uint[4] quarter = { atlas_key(px, py), atlas_key(px + size, py), atlas_key(px, py + size), atlas_key(px + size, py + size) };
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

// One map of a light in the atlas: a spot or directional light's, or one face of a point or
// rect area light's cube.
struct AtlasMap @private {
    // where it is, in pixels (w 0: nowhere)
    int x;
    int y;
    int w;
    int h;
    // its view-projection when it was last drawn (a directional light's box may follow the
    // camera: the lookups use the box the map was drawn for)
    Mat4 view_proj;
    // drawn since it got its place
    bool drawn;
    // when it was last drawn, nothing cast a shadow into it: it isn't looked up (lit)
    bool empty;
    // which of its light's blocks it is in
    int block;
    // A map with casters that move in it is drawn as a copy of its cache, which holds the
    // casters that keep still, with the others drawn over it (composed). The cache is drawn
    // again only when a caster that keeps still changes in it or the light moves.
    bool cached;
    bool cache_empty;
    bool composed;
    // the frame its cache was last needed
    uint cache_used;
    // something changed in it while the camera saw nothing of it: drawn when it does
    bool stale;
}

// What a map of the atlas needs in a frame (place_and_draw_atlas): all of its casters drawn
// (full), its cache drawn (cache), its cache copied with the casters that move drawn over it
// (compose), or only the copy (restore: the casters that moved left it).
struct AtlasJob @local {
    int slot;
    int face;
    Mat4 vp;
    bool full;
    bool cache;
    bool compose;
    bool restore;
    bool plain;
}

// A light with a place in the atlas (a bake's lights have their own, see render_bake).
struct AtlasSlot @private {
    Light* light;
    bool bake;
    // the side of its maps' blocks (0: it has none), how many maps it has and how many blocks
    // they take (a rect area light's side maps are half as tall, two to a block)
    int size;
    int maps;
    int blocks;
    // the size it settles on (atlas_wanted, held steady), and the size the last frames asked
    // for instead, for how many frames
    int target;
    int asked;
    int asked_frames;
    // how much its maps matter: how large its reach looks times how bright it is at the camera
    float importance;
    // its shadow's strength, 0..1: a light that loses its maps to more important ones fades
    // out before it gives them up, and fades in when it gets maps again
    float fade;
    // the frame it last asked for its maps
    uint seen;
    // its blocks of the atlas (atlas_key), and of the cache (where it has one)
    uint[6] block;
    uint[6] cache_block;
    bool[6] cache_has;
    AtlasMap[6] map;
}

// Frames a light keeps its maps after it last asked for them (a light switched off for a
// moment, or one of a render that draws no shadows every frame).
const uint ATLAS_KEEP_FRAMES @local = 120;
// Frames a light must ask for another size before it gets it: a larger one, a smaller one.
// A light whose maps are drawn once (shadow.auto_update off) only gets smaller when the
// atlas is full.
const int ATLAS_GROW_FRAMES @local = 3;
const int ATLAS_SHRINK_FRAMES @local = 30;
// How much a shadow fades in or out per frame.
const float ATLAS_FADE_STEP @local = 0.125f;

fn int atlas_map_count(Light* l) @local => l.light_type == RECT_AREA ? 5 : l.light_type == POINT ? 6 : 1;
fn int atlas_block_count(Light* l) @local => l.light_type == RECT_AREA ? 3 : l.light_type == POINT ? 6 : 1;

// The light's place in the atlas, the current render's (a bake's or not); null: none.
fn AtlasSlot* atlas_slot_of(Light* l) @private {
    bool bake = state.bake.on;
    foreach (&s : state.atlas_slots) {
        if (s.light == l && s.bake == bake) return s;
    }
    return null;
}

// Map `face` of a light's shadow, if it has one (null: none).
fn AtlasMap* atlas_map_of(Light* l, int face = 0) @private {
    if (!l.cast_shadow) return null;
    AtlasSlot* s = atlas_slot_of(l);
    if (!s || !s.size || face >= s.maps) return null;
    return &s.map[face];
}

// Whether any of a light's maps is drawn.
fn bool atlas_has_shadow(Light* l) @private {
    if (!l.cast_shadow) return false;
    AtlasSlot* s = atlas_slot_of(l);
    if (!s || !s.size) return false;
    for (int f = 0; f < s.maps; f++) {
        if (s.map[f].drawn) return true;
    }
    return false;
}

// Where map `m` is in the atlas's uv, as spot_shadows holds it (0 wide: nowhere; the shaders
// take that as lit).
fn Vec4 atlas_rect(AtlasMap* m) @private {
    if (!m || !m.drawn || m.empty || !m.w || !state.atlas_width) return { 0, 0, 0, 0 };
    float w = (float)state.atlas_width;
    float h = (float)state.atlas_height;
    return { (float)m.x / w, (float)m.y / h, (float)m.w / w, (float)m.h / h };
}

// A texel of map `m`, as a part of its width (or height, the smaller).
fn float atlas_texel(AtlasMap* m) @private => m && m.w ? 1.0f / (float)math::min(m.w, m.h) : 1.0f;

// The power of two nearest `size`, from ATLAS_MIN_BLOCK up to the atlas's side.
fn int atlas_pow2(int size) @local {
    int p = ATLAS_MIN_BLOCK;
    while (p < state.atlas_packer.root && p * 3 / 2 < size) p *= 2;
    return p;
}

// The height of the picture being rendered, in pixels.
fn float render_height() @local {
    RenderTarget* rt = state.renderer.render_target;
    return rt ? (float)rt.height : sapp::heightf();
}

// The size of a light's maps as it looks from the camera: two texels for each pixel of the
// picture that its reach spans, up to its shadow.map_size (a directional light's and a bake's:
// that size). `importance`: how large its reach looks times how bright it is at the camera.
fn int atlas_wanted(Light* l, Camera* camera, bool bake, float* importance) @local {
    int most    = atlas_pow2(shadow_map_size(light_shadow(l).map_size));
    Vector3 at  = camera.matrix_world.get_position();
    Vector3 pos = l.matrix_world.get_position();
    float dl    = (pos - at).length();
    float bright = math::max(l.intensity, 0.0f) / math::max(dl * dl, 1.0f);
    if (bake || l.light_type == DIRECTIONAL) {
        // (before any other)
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
    // the radius it spans on screen, in halves of the picture's height
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
    // (a ray-traced light's maps only shadow what the deferred pass doesn't draw)
    if (rt_traced(l)) most = math::min(most, RT_MAP_SIZE);
    return math::min(atlas_pow2((int)(cover * render_height() * 2)), most);
}

// Whether a light gets maps in the atlas in this render: every shadowed spot, point, rect area
// and directional light besides the primary spot light (`caster`) and the sun. A light whose
// maps are drawn once keeps them at intensity 0 too: a light that flickers down to 0 would
// otherwise have them drawn again.
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

// Gives a light's blocks back.
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

// Blocks of `size` for a light's maps (false, with none taken, when they don't fit).
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
    // a rect area light's: the front in a block, the sides half a block each
    s.map[0] = { .x = atlas_key_x(s.block[0]), .y = atlas_key_y(s.block[0]), .w = size, .h = size };
    for (int f = 1; f < s.maps; f++) {
        uint b = s.block[(f + 1) / 2];
        s.map[f] = { .x = atlas_key_x(b), .y = atlas_key_y(b) + (f % 2 ? 0 : size / 2), .w = size, .h = size / 2, .block = (f + 1) / 2 };
    }
    return true;
}

// The corners of the box `inv` (an inverse view-projection) takes the clip space box with
// depths from `z0` to `z1` to.
fn Vector3[8] clip_box_corners(Mat4 inv, float z0, float z1) @local {
    Vector3[8] c;
    for (int i = 0; i < 8; i++) {
        Vec4 p = inv.mul_vec4({ i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? z1 : z0, 1 });
        c[i] = { p.x / p.w, p.y / p.w, p.z / p.w };
    }
    return c;
}

// Whether two frusta may overlap: no plane of either has all of the other's corners outside.
fn bool frusta_overlap(Frustum* a, Vector3[8]* ac, Frustum* b, Vector3[8]* bc) @local {
    for (int k = 0; k < 2; k++) {
        Frustum* f = k ? b : a;
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

// What the camera sees, as atlas_seen asks it: its frustum, and its view-projection's inverse.
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

// Whether the camera can see anything a map of a light (view-projection `light_vp`) shadows:
// their frusta overlap, the camera's cut off where the light's reach ends.
fn bool atlas_seen(AtlasView* v, Mat4 light_vp, Light* l, Vector3 eye) @local {
    bool zo = depth_zero_to_one();
    float z0 = zo ? 0 : -1;
    float z1 = 1;
    if (v.perspective) {
        float reach = (l.matrix_world.get_position() - eye).length() + point_shadow_far(l);
        float d = math::min(v.far, math::max(reach, v.near * 2));
        // (the depth of distance d in the camera's clip space)
        z1 = (v.proj[3][2] - v.proj[2][2] * d) / d;
        z1 = math::min(z1, 1.0f);
    }
    Vector3[8] cam = clip_box_corners(v.inv, z0, z1);
    Frustum lf = frustum_from(light_vp);
    Vector3[8] lc = clip_box_corners(light_vp.invert(), z0, 1);
    return frusta_overlap(&v.frustum, &cam, &lf, &lc);
}

// Sorts slot indices by `key` of each, the largest first.
fn void atlas_sort(List{int}* order, float[] key) @local {
    for (sz i = 1; i < order.len(); i++) {
        int v = (*order)[i];
        sz j = i;
        while (j > 0 && key[(*order)[j - 1]] < key[v]) {
            (*order)[j] = (*order)[j - 1];
            j--;
        }
        (*order)[j] = v;
    }
}

// Frames between walks over the whole scene (between them, only what moved is looked at:
// a matrix_world set by hand, without update_world, is seen at the next).
const uint CASTER_WALK_FRAMES @local = 60;
// Frames a caster that changed counts as moving.
const uint CASTER_STILL_FRAMES @local = 8;
// Frames a map keeps its cache of the casters that don't move after it last needed it.
const uint ATLAS_CACHE_KEEP_FRAMES @local = 120;

// A shadow caster as the last walk over its scene found it (walk_casters).
struct CasterRec @local {
    Mesh* mesh;
    Mat4 world;
    // what else it is drawn from: its geometry and material, and how often their vertices
    // and instances changed
    BufferGeometry* geometry;
    Material* material;
    uint version;
    int instances;
    // its bounding sphere in the world: xyz the centre, w the radius
    Vec4 sphere;
    // the frame it last changed
    uint moved;
    bool animated;
    bool instanced;
}

// The shadow casters of a scene, walked once a frame: what changed among those that keep still
// (the spheres of casters that started or stopped moving, came or went: the cached maps they
// are in must be drawn again) and those that move.
struct CasterWalk @private {
    Scene* scene;
    uint frame;
    List{CasterRec} recs;
    List{CasterRec} last;
    // which of `last` this walk found again (Mesh.shadow_walk_at says where)
    List{bool} found;
    uint last_frame;
    List{Vec4} changes;
    // those that move on the GPU, those that moved in the last frames, and those whose
    // geometry changes (indices into recs); and what the last full walk (every
    // CASTER_WALK_FRAMES) saw of the world's changes (see note_moved_mesh)
    uint full_frame;
    uint structure;
    uint dropped;
    sz moved_seen;
    List{int} animated;
    List{int} recent;
    List{int} watched;
}

// Whether a caster moves on the GPU every frame: its vertices sway, move by its position_node or
// its morph targets.
fn bool caster_animated(Mesh* m) @local {
    Material* mat = m.material;
    return mat.sway > 0 || mat.position_node.id != 0 || m.morph_targets > 0;
}

// Whether a caster moved in the last frames.
fn bool rec_moving(CasterRec* r, uint frame) @local => frame - r.moved < CASTER_STILL_FRAMES;

fn void walk_object(CasterWalk* w, Object3D* obj, uint frame) @local {
    if (!obj.visible) return;
    if (obj.kind == MESH) {
        Mesh* m = (Mesh*)obj;
        BufferGeometry* g = m.geometry;
        bool caster = g && m.material && m.material.visible && m.cast_shadow && !m.material.transparent && !m.is_line()
            && m.billboard == NO_BILLBOARD && (g.uploaded || g.indices.len() > 0);
        if (caster) walk_caster(w, m, frame);
    }
    // (every level of a LOD: which one a map shows is left as it was drawn)
    foreach (c : obj.children) walk_object(w, c, frame);
}

fn void walk_caster(CasterWalk* w, Mesh* m, uint frame) @local {
    // (met already in this walk: a mesh in two places of the scene)
    if (m.shadow_walk == w && m.shadow_walk_frame == frame) return;
    BufferGeometry* g = m.geometry;
    int i = m.shadow_walk_at;
    bool known = m.shadow_walk == w && m.shadow_walk_frame == w.last_frame && i >= 0 && i < w.last.len() && w.last[i].mesh == m;
    CasterRec rec;
    bool changed = true;
    if (known) {
        w.found[i] = true;
        CasterRec* was = &w.last[i];
        changed = libc::memcmp(&was.world, &m.matrix_world, Mat4::size) != 0 || was.geometry != g
            || was.material != m.material || was.version != g.version || was.instances != m.instance_uploads;
        // (as it was: its sphere too)
        if (!changed) rec = *was;
    }
    if (changed) {
        Vector3 center = g.bounds_center;
        float radius = g.bounds_radius;
        if (m.instance_data.len() > 0) {
            center = m.instance_center;
            radius = m.instance_radius;
        }
        Vector3 wc = m.matrix_world.transform_point(center);
        float r = radius * max_scale(m.matrix_world) + m.material.sway * 2;
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
        // into the maps it was cached in, or out of them; while it moves, it is drawn
        // over them
        if (changed && !was_moving) w.changes.push(was.sphere);
        if (was_moving && !now_moving) w.changes.push(rec.sphere);
    } else {
        // new (or back): still from the start
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

// Walks the shadow casters of `scene` (once a frame).
fn CasterWalk* walk_casters(Scene* scene) @private {
    uint frame = state.frame;
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
    // the whole scene now and then, and when objects came or went; otherwise only what
    // update_world says moved, and the geometries that change
    bool full = fresh || frame - w.full_frame >= CASTER_WALK_FRAMES || w.structure != scene_structure_changes
        || w.dropped != moved_meshes_dropped || w.moved_seen > moved_meshes.len();
    if (full) {
        // this frame's walk becomes the last one
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
        // the casters that went: out of the maps they were cached in
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
        // those that stopped: into the maps' caches
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

// A caster of the last full walk that may have changed (it moved, or its geometry is one that
// changes): its record brought up to date (between full walks).
fn void refresh_caster(CasterWalk* w, Mesh* m, uint frame) @local {
    int i = m.shadow_walk_at;
    if (m.shadow_walk != w || i < 0 || i >= w.recs.len() || w.recs[i].mesh != m) return;
    CasterRec* r = &w.recs[i];
    BufferGeometry* g = m.geometry;
    if (r.moved == frame) return;
    bool changed = libc::memcmp(&r.world, &m.matrix_world, Mat4::size) != 0 || r.geometry != g
        || r.material != m.material || r.version != g.version || r.instances != m.instance_uploads;
    if (!changed) return;
    // out of the maps it was cached in; drawn over them while it moves
    if (!rec_moving(r, frame - 1)) {
        w.changes.push(r.sphere);
        w.recent.push(i);
    }
    Vector3 center = g.bounds_center;
    float radius = g.bounds_radius;
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

// Whether a caster that moves is in a map of a light (whose shadow.auto_update is `every`: a
// light drawn once draws the casters that move on the GPU, as foliage in the wind, into its
// cache as they were).
fn bool moving_in(CasterWalk* w, List{int}* near, Frustum* f) @local {
    foreach (i : *near) {
        if (sphere_in(f, w.recs[i].sphere)) return true;
    }
    return false;
}

// The casters that move in reach of light `l` (indices into w.recs; see moving_in).
fn void movers_near(CasterWalk* w, Light* l, List{int}* near) @local {
    LightShadow* sh = light_shadow(l);
    bool everywhere = l.light_type == DIRECTIONAL;
    Vector3 pos = l.matrix_world.get_position();
    float reach = l.light_type == SPOT ? spot_shadow_far((SpotLight*)l) : point_shadow_far(l);
    // (those that moved; with shadow.auto_update, also those that move on the GPU: a light
    // drawn once keeps them in its cache as they were)
    for (int pass = 0; pass < (sh.auto_update ? 2 : 1); pass++) {
        foreach (i : pass == 0 ? w.recent : w.animated) {
            CasterRec* r = &w.recs[i];
            if (pass == 1 && w.recs[i].moved + CASTER_STILL_FRAMES > w.frame) continue;
            if (sh.skip_instanced && r.instanced) continue;
            if (!everywhere) {
                Vector3 c = { r.sphere.x, r.sphere.y, r.sphere.z };
                float d = reach + r.sphere.w;
                if ((c - pos).length_sq() > d * d) continue;
            }
            near.push(i);
        }
    }
}

// Whether a caster that keeps still changed in a map.
fn bool changed_in(CasterWalk* w, Frustum* f) @local {
    foreach (s : w.changes) {
        if (sphere_in(f, s)) return true;
    }
    return false;
}

// Whether a caster is drawn over a light's cache (it moves), rather than into it.
fn bool caster_moves(Mesh* m, bool every) @local => m.shadow_moving || (every && m.shadow_animated);

// Whether a map's block has a cache.
fn bool atlas_cache_ready(AtlasSlot* s, AtlasMap* m) @local => s.cache_has[m.block];

// A cache for a map's block (false: no room).
fn bool atlas_cache_place(AtlasSlot* s, AtlasMap* m) @local {
    if (!state.cache_packer.alloc(s.size, &s.cache_block[m.block])) return false;
    s.cache_has[m.block] = true;
    for (int f = 0; f < s.maps; f++) {
        if (s.map[f].block == m.block) s.map[f].cached = false;
    }
    m.cache_used = state.frame;
    return true;
}

// Where a map's cache is in the cache's texture.
fn int[2] atlas_cache_at(AtlasSlot* s, AtlasMap* m) @local {
    uint c = s.cache_block[m.block];
    uint b = s.block[m.block];
    return { atlas_key_x(c) + m.x - atlas_key_x(b), atlas_key_y(c) + m.y - atlas_key_y(b) };
}

// Gives back the caches of a light's blocks that no map has needed for a while.
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

// Draws the casters of map `face` of light `l` (its view-projection `vp`, `size` texels across)
// into the viewport: `which` 0 all of them, 1 those that keep still, 2 those that move (see
// caster_moves). Returns how many it drew.
// (draw_atlas_casters's casters that move, for `which` 2: the walk and the light's movers_near)
CasterWalk* atlas_movers @local;
List{int}* atlas_movers_near @local;

fn int draw_atlas_casters(Scene* scene, Camera* camera, Light* l, int face, int size, Mat4 vp, int which) @private {
    LightShadow* sh = light_shadow(l);
    Frustum fr = frustum_from(vp);
    state.casters.clear();
    state.skip_instanced = sh.skip_instanced;
    if (which == 2 && atlas_movers) {
        // the casters that move in reach of the light, rather than the whole scene
        Vector3 from = camera.matrix_world.get_position();
        foreach (k : *atlas_movers_near) {
            Mesh* mover = atlas_movers.recs[k].mesh;
            sz first = state.casters.len();
            collect(mover, camera.matrix_world_inverse, &fr, from, true);
            // (its children are walked on their own)
            for (sz j = state.casters.len(); j > first; j--) {
                if (state.casters[j - 1].mesh != mover) state.casters.remove_at(j - 1);
            }
        }
    } else {
        collect_casters(scene, camera, &fr);
    }
    state.skip_instanced = false;
    state.shadow_plain = false;
    bool every = sh.auto_update;
    bool from_light = l.light_type == SPOT || casts_cube(l);
    Mat4 light_rel = from_light ? light_view_proj(l, camera, true, face, size) : vp;
    Vector3 light_pos = from_light ? l.matrix_world.get_position() : (Vector3) { 0, 0, 0 };
    int drawn;
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
    state.renderer.info.atlas_casters += drawn;
    return drawn;
}

// Where map `face` of a light's shadow is in the shadow atlas, in pixels: x, y, width and
// height (all 0: it has none, or nothing casts a shadow into it).
fn int[4] Renderer.shadow_map_rect(&self, Light* l, int face = 0) {
    AtlasMap* m = atlas_map_of(l, face);
    if (!m || !m.drawn || m.empty) return { 0, 0, 0, 0 };
    return { m.x, m.y, m.w, m.h };
}

// Places the lights' maps in the atlas and draws them. Each light asking for maps (atlas_wants)
// gets a size from how it looks from the camera (atlas_wanted), held steady over a few frames;
// while the atlas can't hold them all, the least important lights' maps get smaller, then
// those lights lose their shadows. A map is drawn when it gets a new place, when its light
// asks for it (shadow.needs_update), every frame for a light with shadow.auto_update, and every
// time for a bake's light (`bake`, render_bake: its lights have places of their own, so the
// main render's maps stay as they are).
fn void update_shadow_atlas(Scene* scene, Camera* camera, bool bake = false, SpotLight* caster = null) @private {
    @pool() {
        place_and_draw_atlas(scene, camera, bake, caster);
    };
}

fn void place_and_draw_atlas(Scene* scene, Camera* camera, bool bake, SpotLight* caster) @local {
    RenderInfo* info = &state.renderer.info;
    int most = state.renderer.shadow_atlas_size > 0 ? state.renderer.shadow_atlas_size : DEFAULT_SHADOW_ATLAS_SIZE;
    int limit = sg::query_limits().max_image_size_2d;
    if (limit > 0) most = math::min(most, limit);
    int root = ATLAS_MIN_BLOCK;
    while (root * 2 <= most) root *= 2;
    if (state.atlas_packer.root != root) {
        state.atlas_slots.clear();
        state.atlas_packer.reset(root);
        state.cache_packer.reset(root);
    }
    uint frame = state.frame;
    foreach (l : state.lights) {
        if (!atlas_wants(l, bake, caster)) continue;
        AtlasSlot* s = atlas_slot_of(l);
        if (!s) {
            state.atlas_slots.push({ .light = l, .bake = bake, .maps = atlas_map_count(l), .blocks = atlas_block_count(l), .fade = 1 });
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
    // the lights not asked for a while give their places back
    for (sz i = state.atlas_slots.len(); i > 0; i--) {
        AtlasSlot* s = &state.atlas_slots[i - 1];
        if (s.bake != bake) continue;
        if (frame - s.seen > ATLAS_KEEP_FRAMES) {
            atlas_release(s);
            state.atlas_slots.remove_at(i - 1);
        }
    }

    // what the asking lights get: their targets; while the atlas can't hold them, the least
    // important ones a size smaller each round, then nothing
    sz n = state.atlas_slots.len();
    List{int} order;
    order.tinit();
    float[] importance = mem::temp_array(float, n);
    int[] give = mem::temp_array(int, n);
    long budget = (long)root * root;
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
    // (the places of lights not in this render go first)
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
    // a light left without fades out on the maps it has, then gives them up
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
    // new places for the lights whose size changed, the larger ones first; when they don't
    // fit, everything is placed anew (unless a fading light will soon make room)
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

    // the texture: as large as the part in use (it only grows)
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

    // What each map needs this frame (AtlasJob). Casters that move (CasterWalk) are drawn over a
    // copy of the map's cache of the others; a map that sees none keeps what it has until a
    // caster that keeps still changes in it or its light moves.
    CasterWalk* walk = bake ? null : walk_casters(scene);
    AtlasView view = atlas_view_of(camera);
    Vector3 eye = camera.matrix_world.get_position();
    List{AtlasJob} jobs;
    List{int}[] movers = mem::temp_array(List{int}, state.atlas_slots.len());
    jobs.tinit();
    foreach (i : order) {
        AtlasSlot* s = &state.atlas_slots[i];
        if (!s.size) continue;
        Light* l = s.light;
        LightShadow* sh = light_shadow(l);
        List{int}* near = &movers[i];
        near.tinit();
        if (walk) movers_near(walk, l, near);
        for (int f = 0; f < s.maps; f++) {
            AtlasMap* m = &s.map[f];
            Mat4 vp = light_view_proj(l, camera, false, f, m.w);
            if (bake) {
                jobs.push({ .slot = i, .face = f, .vp = vp, .full = true });
                continue;
            }
            Frustum fr = frustum_from(vp);
            bool light_moved = m.drawn && libc::memcmp(&vp, &m.view_proj, Mat4::size) != 0;
            bool dirty = !m.drawn || m.stale || sh.needs_update || light_moved || changed_in(walk, &fr);
            bool moves = near.len() > 0 && moving_in(walk, near, &fr);
            if (!dirty && !moves && !m.composed) continue;
            // a map that has something and that the camera sees nothing of keeps it for now
            // (one drawn once is drawn when asked, a reflection may see it)
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
    // caches no map has needed for a while give their blocks back
    foreach (&s : state.atlas_slots) {
        if (s.size) atlas_cache_trim(s);
    }
    if (!jobs.len()) return;
    bool top_left = depth_zero_to_one();

    // the caches: the casters that keep still
    int cache_width = math::max(state.cache_width, ATLAS_MIN_BLOCK);
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
        AtlasSlot* s = &state.atlas_slots[job.slot];
        AtlasMap* m = &s.map[job.face];
        LightShadow* sh = light_shadow(s.light);
        int[2] at = atlas_cache_at(s, m);
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

    // the atlas
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
        AtlasSlot* s = &state.atlas_slots[job.slot];
        AtlasMap* m = &s.map[job.face];
        Light* l = s.light;
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
            // (its cache no longer holds what the map does)
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
        // after a light's last map: a caster used the plain material, so the maps are drawn
        // again next frame (a light drawn once gets the real programs when they are ready)
        if (k + 1 == jobs.len() || jobs[k + 1].slot != job.slot) {
            sh.needs_update = plain;
            plain = false;
        }
    }
    sg::end_pass();
    state.casters.clear();
    if (vsm) vsm_blur(&state.vsm_atlas, state.atlas_view, tiles[:ntiles], !all);
}
