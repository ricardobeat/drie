// Offscreen render targets (RenderTarget): color, optional MSAA resolve, depth, and MRT buffers.
// RGBA8 targets hold tone-mapped sRGB output; RGBA16F targets stay in linear HDR light for
// drie::postprocessing passes (OutputPass, GTAOPass, TAAPass). With `mrt`, attachments 1..3
// store world-space octahedral normals, screen-space motion vectors, and indirect lighting
// shares (requires RGBA16F). Connects to Renderer.set_render_target and get_pipeline; render
// targets have a top-left row origin on depth-[0,1] backends, so as_map() flips V accordingly.
module drie;
import sokol;

// Offscreen render surface with optional MSAA, depth buffer, and MRT auxiliary attachments.
// Sample `texture` directly in post-processing passes or call as_map() for a material Texture.
struct RenderTarget {
    int width;                      // Target width in pixels.
    int height;                     // Target height in pixels.
    int samples;                    // MSAA sample count (1 or 4 for scene rendering).
    // Pixel format (DEFAULT selects RGBA8). RGBA8 holds tone-mapped sRGB; RGBA16F holds linear
    // HDR light for post-processing passes such as drie::postprocessing OutputPass.
    SgPixelFormat format;
    bool depth_buffer;              // Whether a depth attachment and depth_texture are allocated.
    SgImage color;                  // Primary color attachment image (multisampled when samples > 1).
    SgImage resolve;                // Single-sampled resolve target when samples > 1.
    SgImage depth;                  // Depth attachment image (DEPTH format).
    SgView color_view;              // Color attachment view for `color`.
    SgView resolve_view;            // Resolve attachment view for `resolve`.
    SgView depth_view;              // Depth-stencil attachment view for `depth`.
    SgView texture;                 // Sampleable texture view of the resolved color image.
    SgImage texture_image;          // Resolved image backing `texture` (`resolve` or `color`).
    SgView depth_texture;           // Sampleable depth view (multisampled if `samples > 1`).
    SgView overlay_view;            // Color attachment view over `texture_image` for post overlays.
    Texture* map_texture;           // Cached Texture wrapper returned by as_map().
    // Enables three extra attachments on an RGBA16F target (set before the first set_size()):
    //   normal_texture    RG16SN (RG16F fallback): world-space shaded normal packed with
    //                     octahedral encoding (see unpack_octahedral_normal in shaders/gtao.glsl)
    //   velocity_texture  RG16F: UV delta (current - previous) from camera motion, mesh motion,
    //                     and dynamic geometry vertices; instances and skinned vertices report 0
    //   indirect_texture  RGBA8: r = diffuse indirect share, g = specular indirect share (used
    //                     by GTAOPass in drie::postprocessing), b = roughness, a = metalness
    // Transparent materials do not write to MRT attachments.
    bool mrt;
    uint drawn_frame;               // Engine frame counter + 1 of the last pass into this target.
    SgImage[MRT_COUNT] mrt_color;
    SgImage[MRT_COUNT] mrt_resolve;
    SgView[MRT_COUNT] mrt_color_view;
    SgView[MRT_COUNT] mrt_resolve_view;
    SgView normal_texture;          // Sampleable MRT attachment 1: world-space octahedral normals.
    SgView velocity_texture;        // Sampleable MRT attachment 2: screen-space motion vectors.
    SgView indirect_texture;        // Sampleable MRT attachment 3: indirect shares, rough, metal.
}

// Number of extra MRT color attachments after color (attachments 1..3: normal, velocity, indirect).
const int MRT_COUNT = 3;
fn SgPixelFormat mrt_format(int i) @private {
    switch (i) {
        case 0: return sg::query_pixelformat(SgPixelFormat.RG16SN).render ? SgPixelFormat.RG16SN : SgPixelFormat.RG16F;
        case 1: return RG16F;
        default: return RGBA8;
    }
}

// Exposes this render target as a material Texture (updated automatically on set_size()).
// Call after the target has a non-zero size.
//   RenderTarget* feed = drie::render_target(256, 144);
//   screen_material.map = feed.as_map();
//   renderer.set_render_target(feed); renderer.render(scene, other_camera);
fn Texture* RenderTarget.as_map(&self) {
    if (!self.map_texture) {
        Texture* t = mem::new(Texture);
        *t = {
            .repeat = { 1, 1 },
            .wrap_s = CLAMP_TO_EDGE_WRAPPING,
            .wrap_t = CLAMP_TO_EDGE_WRAPPING,
            .uploaded = true
        };
        t.sampler = sg::make_sampler(
            &&{ .min_filter = LINEAR, .mag_filter = LINEAR, .wrap_u = CLAMP_TO_EDGE, .wrap_v = CLAMP_TO_EDGE }
        );
        // Non-GL backends store row 0 at the top of render targets; flip V to match image UVs.
        if (depth_zero_to_one()) {
            t.repeat.y = -1;
            t.offset.y = 1;
        }
        self.map_texture = t;
    }
    self.map_texture.width = self.width;
    self.map_texture.height = self.height;
    self.map_texture.image = self.texture_image;
    self.map_texture.view = self.texture;
    return self.map_texture;
}

// Allocates a RenderTarget of `width` x `height` pixels. Pass `samples > 1` for MSAA with
// automatic resolve into `texture`, `format: RGBA16F` for linear HDR output (tone-mapped later
// by OutputPass in drie::postprocessing), `depth_buffer: false` for post-processing buffers, and
// `mrt: true` on RGBA16F targets to also record normals, velocity, and indirect shares.
//   drie::render_target(w, h, 4, RGBA16F, mrt: true)
// Must be called after GPU initialization.
fn RenderTarget* render_target(
    int width,
    int height,
    int samples = 1,
    SgPixelFormat format = RGBA8,
    bool depth_buffer = true,
    bool mrt = false
) {
    RenderTarget* rt = mem::new(RenderTarget);
    rt.format = format;
    rt.depth_buffer = depth_buffer;
    rt.mrt = mrt;
    rt.set_size(width, height, samples);
    return rt;
}

// Effective color pixel format, resolving DEFAULT to RGBA8.
fn SgPixelFormat RenderTarget.pixel_format(&self) => self.format == DEFAULT ? RGBA8 : self.format;

// True when scenes rendered into this target stay in linear HDR light (RGBA16F).
fn bool RenderTarget.is_linear(&self) => self.pixel_format() == RGBA16F;

// Destroys all GPU images and views while preserving format, depth_buffer, mrt, and map_texture.
fn void RenderTarget.dispose(&self) {
    if (self.texture.id) sg::destroy_view(self.texture);
    if (self.color_view.id) sg::destroy_view(self.color_view);
    if (self.resolve_view.id) sg::destroy_view(self.resolve_view);
    if (self.depth_view.id) sg::destroy_view(self.depth_view);
    if (self.depth_texture.id) sg::destroy_view(self.depth_texture);
    if (self.overlay_view.id) sg::destroy_view(self.overlay_view);
    if (self.color.id) sg::destroy_image(self.color);
    if (self.resolve.id) sg::destroy_image(self.resolve);
    if (self.depth.id) sg::destroy_image(self.depth);
    if (self.normal_texture.id) sg::destroy_view(self.normal_texture);
    if (self.velocity_texture.id) sg::destroy_view(self.velocity_texture);
    if (self.indirect_texture.id) sg::destroy_view(self.indirect_texture);
    for (int i = 0; i < MRT_COUNT; i++) {
        if (self.mrt_color_view[i].id) sg::destroy_view(self.mrt_color_view[i]);
        if (self.mrt_resolve_view[i].id) sg::destroy_view(self.mrt_resolve_view[i]);
        if (self.mrt_color[i].id) sg::destroy_image(self.mrt_color[i]);
        if (self.mrt_resolve[i].id) sg::destroy_image(self.mrt_resolve[i]);
    }
    Texture* map         = self.map_texture;
    SgPixelFormat format = self.format;
    bool depth_buffer    = self.depth_buffer;
    bool mrt             = self.mrt;
    *self = { .map_texture = map, .format = format, .depth_buffer = depth_buffer, .mrt = mrt };
}

// Recreates the target's GPU images and views when dimensions or sample count change, and updates
// any Texture returned by as_map().
fn void RenderTarget.set_size(&self, int width, int height, int samples = 1) {
    if (self.width == width && self.height == height && self.samples == samples) return;
    self.dispose();
    self.width = width;
    self.height = height;
    self.samples = samples;
    SgPixelFormat format = self.pixel_format();
    self.color = sg::make_image(
        &&{
            .usage = { .color_attachment = true },
            .width = width,
            .height = height,
            .sample_count = samples,
            .pixel_format = format,
        }
    );
    self.color_view = sg::make_view(&&{ .color_attachment = { .image = self.color } });
    if (self.depth_buffer) {
        self.depth = sg::make_image(
            &&{
                .usage = { .depth_stencil_attachment = true },
                .width = width,
                .height = height,
                .sample_count = samples,
                .pixel_format = DEPTH,
            }
        );
        self.depth_view = sg::make_view(&&{ .depth_stencil_attachment = { .image = self.depth } });
        self.depth_texture = sg::make_view(&&{ .texture = { .image = self.depth } });
    }
    self.texture_image = self.color;
    if (samples > 1) {
        self.resolve = sg::make_image(
            &&{
                .usage = { .resolve_attachment = true, .color_attachment = true },
                .width = width,
                .height = height,
                .sample_count = 1,
                .pixel_format = format,
            }
        );
        self.resolve_view = sg::make_view(&&{ .resolve_attachment = { .image = self.resolve } });
        self.texture_image = self.resolve;
    }
    self.texture = sg::make_view(&&{ .texture = { .image = self.texture_image } });
    self.overlay_view = sg::make_view(&&{ .color_attachment = { .image = self.texture_image } });
    if (self.mrt) {
        if (!self.is_linear()) abort("drie: a render target with MRT must be RGBA16F");
        for (int i = 0; i < MRT_COUNT; i++) {
            self.mrt_color[i] = sg::make_image(
                &&{
                    .usage = { .color_attachment = true },
                    .width = width,
                    .height = height,
                    .sample_count = samples,
                    .pixel_format = mrt_format(i),
                }
            );
            self.mrt_color_view[i] = sg::make_view(&&{ .color_attachment = { .image = self.mrt_color[i] } });
            SgImage shown = self.mrt_color[i];
            if (samples > 1) {
                self.mrt_resolve[i] = sg::make_image(
                    &&{
                        .usage = { .resolve_attachment = true, .color_attachment = true },
                        .width = width,
                        .height = height,
                        .sample_count = 1,
                        .pixel_format = mrt_format(i),
                    }
                );
                self.mrt_resolve_view[i] = sg::make_view(&&{ .resolve_attachment = { .image = self.mrt_resolve[i] } });
                shown = self.mrt_resolve[i];
            }
            SgView view = sg::make_view(&&{ .texture = { .image = shown } });
            switch (i) {
                case 0: self.normal_texture = view;
                case 1: self.velocity_texture = view;
                default: self.indirect_texture = view;
            }
        }
    }
    if (self.map_texture) self.as_map();
}

// Builds the sokol SgAttachments struct (color, optional MSAA resolves, depth, and MRT slots).
fn SgAttachments RenderTarget.attachments(&self) {
    SgAttachments a;
    a.colors[0] = self.color_view;
    a.depth_stencil = self.depth_view;
    if (self.samples > 1) a.resolves[0] = self.resolve_view;
    if (self.mrt) {
        for (int i = 0; i < MRT_COUNT; i++) {
            a.colors[i + 1] = self.mrt_color_view[i];
            if (self.samples > 1) a.resolves[i + 1] = self.mrt_resolve_view[i];
        }
    }
    return a;
}
