// Progressive lightmap baker (`ProgressiveLightMap`, modeled after three.js's
// `ProgressiveLightMap`). Accumulates jittered direct light and shadows in linear light into two
// ping-pong floating-point render targets (`RGBA32F` when renderable and filterable, otherwise
// `RGBA16F`) via `Renderer.render_bake` and `shaders/lightmap.glsl`. Lights live in the baker's
// own `scene` rather than the main scene; each mesh samples `texture(1)` through its `uv1`
// channel. Create after graphics init, call `update()` before `Renderer.render()`, and call
// `dispose()` to free GPU targets and samplers.

module drie;
import std::collections::list, std::math;
import sokol;

// Progressive UV-space lightmap baker that blends samples across frames into ping-pong maps.
//
//   ProgressiveLightMap* plm = drie::progressive_lightmap(renderer, 1024);
//   plm.scene.add(light);
//   Object3D*[2] objects = { floor, box };
//   plm.add_objects(&objects);
//   // Each frame before render(): jitter or move lights, then:
//   plm.update(camera, blend_window: 100);
//
// Updates alternate between the two targets while meshes stay bound to `texture(1)`, so visible
// lighting advances every second `update()` call.
struct ProgressiveLightMap {
    Renderer* renderer; // Renderer used for `render_bake` passes.
    int res; // Lightmap width and height in texels.
    Scene* scene; // Separate scene holding the lights used during baking.
    List{Object3D*} objects; // Meshes registered for baking.
    RenderTarget*[2] targets; // Ping-pong floating-point render targets (linear light, no depth).
    Texture*[2] maps; // Texture wrappers over `targets`: index 0 for debug, index 1 (`uv1`) for meshes.
    Material* material; // White un-fogged Phong material used while baking meshes.
    bool first_active; // When true, the next `update()` writes `targets[0]` and reads `targets[1]`.
    bool cleared; // True once both targets have been cleared on the first `update()`.
    Mesh* debug_plane; // Unlit 100x100 preview plane created lazily by `show_debug()`.
}

// ---- Baking

// Allocates a `res` x `res` progressive lightmap (`RGBA32F` if renderable and filterable, else
// `RGBA16F`). Must be called after the window opens or `init_headless()` completes.
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
        // Wrap directly instead of `as_map()` so row `r` stays at `v = (r + 0.5) / res` on all backends.
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

// Returns lightmap texture `0` (shown by `show_debug()`) or `1` (bound to meshes on `uv1`).
fn Texture* ProgressiveLightMap.texture(&self, int i) => self.maps[i == 0 ? 0 : 1];

// Registers meshes and lights from `objects`: lights are added to `scene`, and each mesh gets
// `material.light_map = texture(1)` with shadows enabled. When `pack` is true, each mesh's `uv`
// is packed into `uv1` via potpack with a 3-texel margin; when false, existing `uv1` is kept.
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
        // Re-upload geometry buffers with the packed `uv1` coordinates.
        g.uploaded = false;
    }
}

// Bakes one lighting sample from `scene` (viewed from `camera`) and blends it into the inactive
// lightmap with weight `1 / blend_window` (`1` replaces previous samples). When `blur_edges` is
// true, texels are dilated past UV chart edges first to hide seams. Call before `Renderer.render()`.
fn void ProgressiveLightMap.update(&self, Camera* camera, float blend_window = 100, bool blur_edges = true) {
    if (!self.objects.len()) return;
    if (!self.cleared) {
        // Fresh GPU attachments have undefined contents; clear both ping-pong maps before first blend.
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

// Toggles an unlit double-sided 100 x 100 debug plane displaying `texture(0)` at `position`
// inside the first baked object's parent.
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

// Frees both render targets, their samplers, and the debug plane. Clear `material.light_map` on
// any surviving meshes before calling `dispose()`.
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

// ---- UV packing

// One mesh's UV square allocated by `add_objects()` and placed by `potpack()`.
struct PackBox @local {
    double w;
    double h;
    double x;
    double y;
    Mesh* mesh;
}

// Remaining free rectangle tracked during `potpack()`.
struct PackSpace @local {
    double x;
    double y;
    double w;
    double h;
}

// Packs `boxes` (tallest first) into a roughly square atlas using Mapbox's potpack algorithm
// (ISC License). Writes each box's `(x, y)` origin and returns the container `{ width, height }`.
fn double[2] potpack(PackBox[] boxes) @local {
    double area;
    double max_width;
    foreach (b : boxes) {
        area += b.w * b.h;
        max_width = math::max(max_width, b.w);
    }
    // Stable insertion sort descending by height.
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
        // Scan backwards to prefer smaller free rectangles first.
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
