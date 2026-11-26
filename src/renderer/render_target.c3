// Offscreen render targets (RenderTarget): MSAA, linear HDR, and MRT attachments.
module drie;
import sokol;

// An offscreen image to render into, like three's WebGLRenderTarget.
// Use `texture` (a sokol texture view) to sample the result in your own passes.
struct RenderTarget {
    int width;
    int height;
    int samples;
    // its pixel format (like three's `type` and `format` options; 0 is RGBA8) and whether it
    // has a depth buffer (`depthBuffer`). A scene can be rendered into RGBA8 (tone mapped and
    // sRGB, ready to show) or RGBA16F (linear light, like three's HalfFloatType: tone map it
    // with an OutputPass at the end), either with depth.
    SgPixelFormat format;
    bool depth_buffer;
    SgImage color;
    SgImage resolve;
    SgImage depth;
    SgView color_view;
    SgView resolve_view;
    SgView depth_view;
    SgView texture;
    // The resolved image that `texture` reads from.
    SgImage texture_image;
    // For post-processing: the depth of the last render (multisampled if the target is), and
    // an attachment for drawing over the resolved picture.
    SgView depth_texture;
    SgView overlay_view;
    // see as_map
    Texture* map_texture;
    // Multiple render targets, like three's `scenePass.setMRT(mrt({ output, normal, velocity }))`:
    // a scene rendered into an RGBA16F target with `mrt` set also writes
    //   normal_texture    RG16SN (RG16F where that can't be drawn into): the surface's normal
    //                     in world space (after its normal map), packed into two numbers like
    //                     three's packNormalToOctahedral (unpack_octahedral_normal in gtao.glsl)
    //   velocity_texture  RG16F: how far it moved across the picture since the last frame
    //                     drawn with MRT, in texture coordinates (uv now - uv then), with the
    //                     camera's motion, each mesh's and a dynamic geometry's vertices'
    //                     (BufferGeometry.dynamic); instances and skinned vertices are taken
    //                     to stand still
    //   indirect_texture  RGBA8: the share of the pixel's light that is indirect, r = diffuse
    //                     (ambient, hemisphere, irradiance volume, baked vertex light), g =
    //                     specular (environment map and screen-space reflections): what ambient
    //                     occlusion darkens (GTAOPass in postprocessing.c3); b = the surface's
    //                     roughness, a = its metalness
    // Set it before the first set_size. See-through materials write none of them.
    bool mrt;
    // the frame it was last drawn into (see first_pass)
    uint drawn_frame;
    SgImage[MRT_COUNT] mrt_color;
    SgImage[MRT_COUNT] mrt_resolve;
    SgView[MRT_COUNT] mrt_color_view;
    SgView[MRT_COUNT] mrt_resolve_view;
    SgView normal_texture;
    SgView velocity_texture;
    SgView indirect_texture;
}

// RenderTarget.mrt's extra pictures (after the color, at attachments 1, 2, 3) and their pixel
// formats: normal, velocity, indirect.
const int MRT_COUNT = 3;
fn SgPixelFormat mrt_format(int i) @private {
    switch (i) {
        case 0: return sg::query_pixelformat(SgPixelFormat.RG16SN).render ? SgPixelFormat.RG16SN : SgPixelFormat.RG16F;
        case 1: return RG16F;
        default: return RGBA8;
    }
}

// The picture as a Texture, for any material's map, like three's `renderTarget.texture`: a
// screen in the scene showing what another camera sees, say. It follows set_size. Call it
// once the target has a size.
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
        // a render target's first row is at the top, except in GL
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

// samples > 1 renders with MSAA and resolves into `texture`. A target for post-processing
// passes takes a format of its own (RGBA16F, RG16F, R16F...) and no depth buffer:
//   drie::render_target(w, h, format: RGBA16F, depth_buffer: false)
// A scene rendered into an RGBA16F target keeps its light as it is, brighter than white where
// it is (see OutputPass in postprocessing.c3), and with `mrt` it also writes its normals and
// motion (see RenderTarget.mrt):
//   drie::render_target(w, h, 4, RGBA16F, mrt: true)
// Must be called after the window opened.
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

fn SgPixelFormat RenderTarget.pixel_format(&self) => self.format == DEFAULT ? RGBA8 : self.format;

// Whether a scene rendered into it is left in linear light (a half-float target).
fn bool RenderTarget.is_linear(&self) => self.pixel_format() == RGBA16F;

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
