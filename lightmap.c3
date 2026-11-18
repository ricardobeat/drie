// Progressive light maps, like three's ProgressiveLightMap: light baked into the objects' uv1
// a little at a time, from lights that can move between updates (jittered across an area,
// they give soft shadows and the light's spread as the samples add up). The lights live in a
// scene of their own (`scene`), so the main scene has none; each update draws the objects in
// their uv1 into a light map, lit and shadowed by them, and blends that into the last one.
//
//   ProgressiveLightMap* plm = drie::progressive_lightmap(renderer, 1024);
//   plm.scene.add(light);                      // (or pass it to add_objects, as three's)
//   Object3D*[2] objects = { floor, box };
//   plm.add_objects(&objects);                 // uv1 = their uv, packed together
//   // each frame, before render(): move the lights about, then
//   plm.update(camera, blend_window: 100);
//
// The objects keep their own materials, with light_map set to texture(1) (channel 1: their
// uv1). As three's, the updates take turns between two maps, and the objects show the second:
// what they show changes every other update.
module drie;
import std::collections::list, std::math;
import sokol;

struct ProgressiveLightMap {
    Renderer* renderer;
    // the light map's width and height (texels)
    int res;
    // the lights that the light map is baked with
    Scene* scene;
    // the meshes baked into it
    List{Object3D*} objects;
    // the two maps, as textures: texture(0) and texture(1)
    RenderTarget*[2] targets;
    Texture*[2] maps;
    // what the objects are drawn with while baking: three's MeshPhongMaterial, white
    Material* material;
    // the next update draws into the first map (and reads the second)
    bool first_active;
    bool cleared;
    // show_debug's plane
    Mesh* debug_plane;
}

// A light map of `res` x `res` texels (RGBA32F where the GPU can draw into it and filter it,
// else RGBA16F). Must be called after the window opened (or init_headless).
fn ProgressiveLightMap* progressive_lightmap(Renderer* renderer, int res = 1024) {
    ProgressiveLightMap* p = mem::new(ProgressiveLightMap);
    p.renderer = renderer;
    p.res = math::max(res, 1);
    p.scene = scene();
    p.objects.allocator = mem;
    p.material = mesh_phong_material(color: 0xffffff, fog: false);
    SgPixelformatInfo f  = sg::query_pixelformat(RGBA32F);
    SgPixelFormat format = f.render && f.filter ? SgPixelFormat.RGBA32F : SgPixelFormat.RGBA16F;
    for (int i = 0; i < 2; i++) {
        RenderTarget* rt = render_target(p.res, p.res, 1, format, depth_buffer: false);
        // (not RenderTarget.as_map: row r of the map is at v = (r + 0.5) / res on every backend)
        Texture* t       = mem::new(Texture);
        *t = {
            .width = p.res,
            .height = p.res,
            .repeat = { 1, 1 },
            .wrap_s = CLAMP_TO_EDGE_WRAPPING,
            .wrap_t = CLAMP_TO_EDGE_WRAPPING,
            .color_space = NO_COLOR_SPACE,
            .render_target = true,
            .channel = i == 1 ? 1 : 0,
            .image = rt.texture_image,
            .view = rt.texture,
            .uploaded = true,
        };
        t.sampler = sg::make_sampler(
            &&{ .min_filter = LINEAR, .mag_filter = LINEAR, .wrap_u = CLAMP_TO_EDGE, .wrap_v = CLAMP_TO_EDGE }
        );
        p.targets[i] = rt;
        p.maps[i] = t;
    }
    return p;
}

// The light maps: 0 the one show_debug shows, 1 the one the objects show (channel 1).
fn Texture* ProgressiveLightMap.texture(&self, int i) => self.maps[i == 0 ? 0 : 1];

// Bakes `objects` into the light map: their material's light_map becomes texture(1), and they
// cast and receive shadows. The lights among them go to `scene`. With `pack` (three's) each
// mesh's uv1 is its uv, in a square of its own among the others' (potpack), with a margin of
// three texels; without, their uv1 is taken to be laid out already.
fn void ProgressiveLightMap.add_objects(&self, Object3D*[] objects, bool pack = true) {
    List{PackBox} boxes;
    boxes.allocator = mem;
    defer boxes.free();
    double padding = 3.0 / self.res;
    foreach (o : objects) {
        if (o.kind == LIGHT) {
            self.scene.add(o);
            continue;
        }
        if (o.kind != MESH) continue;
        Mesh* m = (Mesh*)o;
        if (!m.geometry || !m.material || !m.geometry.vertex_count()) continue;
        m.material.light_map = self.maps[1];
        m.cast_shadow = true;
        m.receive_shadow = true;
        self.objects.push(o);
        boxes.push({ .w = 1 + padding * 2, .h = 1 + padding * 2, .mesh = m });
    }
    if (!pack || !boxes.len()) return;
    double[2] size = potpack(boxes.array_view());
    foreach (b : boxes) {
        BufferGeometry* g = b.mesh.geometry;
        g.uv1.clear();
        for (uint i = 0; i < g.vertex_count(); i++) {
            Vector2 uv = g.get_uv(i);
            g.uv1.push((float)(((double)uv.x + b.x + padding) / size[0]));
            g.uv1.push((float)(((double)uv.y + b.y + padding) / size[1]));
        }
        // uploaded again, with the new uv1
        g.uploaded = false;
    }
}

// Draws one more sample of the light into the map: each object, lit by `scene`'s lights as
// they are now (and seen from `camera`, for the highlights), blended into what the last update
// left, 1 / blend_window of it the new sample (1: only the new one). `blur_edges` spreads the
// light past the edges of each object's part of the map first, so its seams don't show.
// Call it before render(), every frame or as often as the lights change.
fn void ProgressiveLightMap.update(&self, Camera* camera, float blend_window = 100, bool blur_edges = true) {
    if (!self.objects.len()) return;
    if (!self.cleared) {
        // (a new image holds anything)
        foreach (rt : self.targets) {
            sg::begin_pass(
                &&{
                    .action = {
                        .colors[0] = { .load_action = CLEAR, .store_action = STORE, .clear_value = { 0, 0, 0, 1 } }
                    },
                    .attachments = rt.attachments(),
                    .label = "drie-lightmap-clear",
                }
            );
            sg::end_pass();
        }
        self.cleared = true;
    }
    RenderTarget* active   = self.targets[self.first_active ? 0 : 1];
    RenderTarget* inactive = self.targets[self.first_active ? 1 : 0];
    self.first_active = !self.first_active;
    self.renderer.render_bake(
        self.scene,
        camera,
        self.objects.array_view(),
        self.material,
        active,
        inactive.texture,
        blend_window,
        blur_edges
    );
}

// Shows texture(0) on a 100 x 100 plane (both sides, unlit) next to the first object, in its
// parent, at `position` (three's showDebugLightmap).
fn void ProgressiveLightMap.show_debug(&self, bool visible, Vector3 position = { 0, 250, 0 }) {
    if (!self.objects.len()) return;
    if (!self.debug_plane) {
        Material* mat = mesh_basic_material(color: 0xffffff, side: DOUBLE_SIDE, map: self.maps[0]);
        self.debug_plane = mesh(plane_geometry(100, 100), mat);
        Object3D* parent = self.objects[0].parent;
        if (parent) parent.add(self.debug_plane);
    }
    self.debug_plane.position = position;
    self.debug_plane.visible = visible;
}

// Frees the maps, the material and the debug plane (the objects keep light_map: set it to
// null first, if they stay).
fn void ProgressiveLightMap.dispose(&self) {
    for (int i = 0; i < 2; i++) {
        sg::destroy_sampler(self.maps[i].sampler);
        free(self.maps[i]);
        self.targets[i].dispose();
        free(self.targets[i]);
    }
    if (self.debug_plane) {
        if (self.debug_plane.parent) self.debug_plane.parent.remove(self.debug_plane);
        self.debug_plane.geometry.dispose();
    }
    self.objects.free();
}

// A square of the light map for one mesh (add_objects), placed by potpack.
struct PackBox @local {
    double w;
    double h;
    double x;
    double y;
    Mesh* mesh;
}

// An empty space of potpack's.
struct PackSpace @local {
    double x;
    double y;
    double w;
    double h;
}

// potpack (Mapbox, ISC License), as three's ProgressiveLightMap uses it: places the boxes
// (tallest first) in a container about square, each in the smallest empty space that holds
// it. Returns the container's width and height.
fn double[2] potpack(PackBox[] boxes) @local {
    double area;
    double max_width;
    foreach (b : boxes) {
        area += b.w * b.h;
        max_width = math::max(max_width, b.w);
    }
    // by height, the tallest first (keeping the order of equal ones, as JavaScript's sort)
    for (usz i = 1; i < boxes.len; i++) {
        PackBox b = boxes[i];
        usz j     = i;
        for (; j > 0 && boxes[j - 1].h < b.h; j--) boxes[j] = boxes[j - 1];
        boxes[j] = b;
    }
    double start = math::max(math::ceil(math::sqrt(area / 0.95)), max_width);
    List{PackSpace} spaces;
    spaces.allocator = mem;
    defer spaces.free();
    spaces.push({ 0, 0, start, double::inf });
    double width;
    double height;
    foreach (&b : boxes) {
        // backwards: the smaller spaces first
        for (sz i = (sz)spaces.len() - 1; i >= 0; i--) {
            PackSpace* s = &spaces[i];
            if (b.w > s.w || b.h > s.h) continue;
            b.x = s.x;
            b.y = s.y;
            height = math::max(height, b.y + b.h);
            width = math::max(width, b.x + b.w);
            if (b.w == s.w && b.h == s.h) {
                PackSpace last = spaces.pop()!!;
                if (i < (sz)spaces.len()) spaces[i] = last;
            } else if (b.h == s.h) {
                s.x += b.w;
                s.w -= b.w;
            } else if (b.w == s.w) {
                s.y += b.h;
                s.h -= b.h;
            } else {
                PackSpace split = { s.x + b.w, s.y, s.w - b.w, b.h };
                s.y += b.h;
                s.h -= b.h;
                spaces.push(split);
            }
            break;
        }
    }
    return { width, height };
}
