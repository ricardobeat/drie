// Textures: images that materials sample (color maps, normal maps, ...).
//   Texture* t = three::load_texture("assets/textures/bark_diff.jpg");
//   t.repeat = { 4, 4 };
//   material.map = t;
module three;
import std::io, std::math;
import sokol, imageio;

enum Wrapping
{
    REPEAT_WRAPPING,
    CLAMP_TO_EDGE_WRAPPING,
    MIRRORED_REPEAT_WRAPPING,
}

enum ColorSpace
{
    SRGB_COLOR_SPACE,
    // For data textures (normal maps, roughness, ...): values are used as stored.
    NO_COLOR_SPACE,
}

struct Texture
{
    int width;
    int height;
    // RGBA8 pixels, top row first. Freed after the upload to the GPU.
    char[] pixels;
    ColorSpace color_space;
    Wrapping wrap_s;
    Wrapping wrap_t;
    // UV scale and offset, like three's texture.repeat / texture.offset.
    Vector2 repeat;
    Vector2 offset;
    bool generate_mipmaps;
    int anisotropy;
    // Drawn into, not loaded (see canvas_texture): black until the first drawing.
    bool render_target;
    // For an environment map captured with distances (Renderer.capture_environment): alpha 1
    // stands for this many meters from where it was captured. 0: no distances.
    float env_distance_range;
    // GPU state, created lazily on first render.
    SgImage image;
    SgView view;
    SgSampler sampler;
    bool uploaded;
}

// A texture from raw RGBA8 pixels (copied), like THREE.DataTexture.
fn Texture* data_texture(char[] rgba, int width, int height, ColorSpace color_space = SRGB_COLOR_SPACE)
{
    Texture* t = mem::new(Texture);
    *t = {
        .width = width,
        .height = height,
        .color_space = color_space,
        .repeat = { 1, 1 },
        .generate_mipmaps = true,
        .anisotropy = 8,
    };
    t.pixels = mem::new_array(char, (sz)width * height * 4);
    t.pixels[..] = rgba[:(sz)width * height * 4];
    return t;
}

// A texture that is drawn into instead of loaded, like THREE.WebGLRenderTarget's `.texture`:
// for a screen in the scene, say. Draw with sokol between begin_canvas and end_canvas (not
// inside another pass), and use `canvas.texture` as any material's map.
//   CanvasTexture* screen = three::canvas_texture(256, 144);
//   material.map = screen.texture;
//   screen.begin(0x001000); sdtx::draw(); screen.end();
struct CanvasTexture
{
    Texture* texture;
    SgView attachment;
}

fn CanvasTexture* canvas_texture(int width, int height)
{
    CanvasTexture* c = mem::new(CanvasTexture);
    Texture* t = mem::new(Texture);
    *t = { .width = width, .height = height, .repeat = { 1, 1 }, .wrap_s = CLAMP_TO_EDGE_WRAPPING,
           .wrap_t = CLAMP_TO_EDGE_WRAPPING, .render_target = true };
    c.texture = t;
    return c;
}

// Starts drawing into the texture, cleared to `clear`.
fn void CanvasTexture.begin(&self, Color clear = 0x000000)
{
    SgPass pass = { .action.colors[0] = { .load_action = CLEAR,
        .clear_value = { ((clear >> 16) & 0xff) / 255.0f, ((clear >> 8) & 0xff) / 255.0f, (clear & 0xff) / 255.0f, 1 } } };
    self.texture.upload();
    if (!self.attachment.id) self.attachment = sg::make_view(&&{ .color_attachment = { .image = self.texture.image } });
    pass.attachments.colors[0] = self.attachment;
    sg::begin_pass(&pass);
}

fn void CanvasTexture.end(&self) => sg::end_pass();

// Decodes a PNG or JPEG held in memory.
fn Texture*? decode_texture(char[] encoded, ColorSpace color_space = SRGB_COLOR_SPACE)
{
    CInt w;
    CInt h;
    char* px = imageio::decode(encoded.ptr, (usz)encoded.len, &w, &h);
    if (!px) return io::FILE_NOT_VALID~;
    defer imageio::free_pixels(px);
    return data_texture(px[:(sz)w * h * 4], w, h, color_space);
}

// Loads a PNG or JPEG file, like `new THREE.TextureLoader().load(path)`.
// A missing or broken file logs a message and gives a magenta placeholder.
fn Texture* load_texture(String path, ColorSpace color_space = SRGB_COLOR_SPACE)
{
    char[]? data = file::load(mem, path);
    if (catch err = data)
    {
        io::eprintfn("three: cannot read texture '%s'", path);
        return data_texture({ 255, 0, 255, 255 }, 1, 1, color_space);
    }
    defer mem::free(data.ptr);
    Texture*? t = decode_texture(data, color_space);
    if (catch err = t)
    {
        io::eprintfn("three: cannot decode texture '%s'", path);
        return data_texture({ 255, 0, 255, 255 }, 1, 1, color_space);
    }
    return t;
}

fn SgWrap sg_wrap(Wrapping w) @local
{
    switch (w)
    {
        case REPEAT_WRAPPING: return REPEAT;
        case CLAMP_TO_EDGE_WRAPPING: return CLAMP_TO_EDGE;
        case MIRRORED_REPEAT_WRAPPING: return MIRRORED_REPEAT;
    }
}

// Halves an RGBA8 image with a box filter. Alpha-weighted so transparent texels don't darken edges.
fn char[] downsample(char[] src, int w, int h, int nw, int nh) @local
{
    char[] dst = mem::new_array(char, (sz)nw * nh * 4);
    for (int y = 0; y < nh; y++)
    {
        for (int x = 0; x < nw; x++)
        {
            uint[4] sum;
            uint asum;
            uint[3] plain;
            for (int j = 0; j < 2; j++)
            {
                for (int i = 0; i < 2; i++)
                {
                    int sx = math::min(x * 2 + i, w - 1);
                    int sy = math::min(y * 2 + j, h - 1);
                    char* p = &src[((sz)sy * w + sx) * 4];
                    uint a = p[3];
                    for (int c = 0; c < 3; c++)
                    {
                        sum[c] += p[c] * a;
                        plain[c] += p[c];
                    }
                    asum += a;
                }
            }
            char* d = &dst[((sz)y * nw + x) * 4];
            for (int c = 0; c < 3; c++) d[c] = (char)(asum ? sum[c] / asum : plain[c] / 4);
            d[3] = (char)(asum / 4);
        }
    }
    return dst;
}

// Uploads the pixels (and mipmaps) to the GPU. Called by the renderer.
fn void Texture.upload(&self)
{
    if (self.uploaded) return;
    self.dispose();
    if (self.render_target)
    {
        self.image = sg::make_image(&&{
            .usage = { .color_attachment = true },
            .width = self.width, .height = self.height, .pixel_format = RGBA8, .sample_count = 1,
        });
        self.view = sg::make_view(&&{ .texture = { .image = self.image } });
        self.sampler = sg::make_sampler(&&{ .min_filter = LINEAR, .mag_filter = LINEAR,
                                            .wrap_u = sg_wrap(self.wrap_s), .wrap_v = sg_wrap(self.wrap_t) });
        // a render target's first row is at the top, except in GL
        if (depth_zero_to_one())
        {
            self.repeat.y = -math::abs(self.repeat.y);
            self.offset.y = 1;
        }
        self.uploaded = true;
        return;
    }
    SgImageDesc desc = {
        .width = self.width,
        .height = self.height,
        .pixel_format = self.color_space == SRGB_COLOR_SPACE ? SgPixelFormat.SRGB8A8 : SgPixelFormat.RGBA8,
    };
    char[][16] levels;
    int count = 1;
    levels[0] = self.pixels;
    if (self.generate_mipmaps)
    {
        int w = self.width;
        int h = self.height;
        while ((w > 1 || h > 1) && count < 12)
        {
            int nw = math::max(w / 2, 1);
            int nh = math::max(h / 2, 1);
            levels[count] = downsample(levels[count - 1], w, h, nw, nh);
            w = nw;
            h = nh;
            count++;
        }
    }
    desc.num_mipmaps = count;
    // sokol takes all mip levels in one block, largest first
    sz total;
    for (int i = 0; i < count; i++) total += levels[i].len;
    char[] all = mem::new_array(char, total);
    defer mem::free(all.ptr);
    sz at;
    for (int i = 0; i < count; i++)
    {
        all[at:levels[i].len] = levels[i][..];
        at += levels[i].len;
    }
    desc.data = { .ptr = all.ptr, .size = (usz)total };
    self.image = sg::make_image(&desc);
    self.view = sg::make_view(&&{ .texture = { .image = self.image } });
    self.sampler = sg::make_sampler(&&{
        .min_filter = LINEAR,
        .mag_filter = LINEAR,
        .mipmap_filter = count > 1 ? SgFilter.LINEAR : SgFilter.NEAREST,
        .wrap_u = sg_wrap(self.wrap_s),
        .wrap_v = sg_wrap(self.wrap_t),
        .max_anisotropy = count > 1 ? (uint)math::clamp(self.anisotropy, 1, 16) : 1,
    });
    for (int i = 1; i < count; i++) mem::free(levels[i].ptr);
    mem::free(self.pixels.ptr);
    self.pixels = {};
    self.uploaded = true;
}

// Frees the GPU resources.
fn void Texture.dispose(&self)
{
    if (self.view.id) sg::destroy_view(self.view);
    if (self.image.id) sg::destroy_image(self.image);
    if (self.sampler.id) sg::destroy_sampler(self.sampler);
    self.view = {};
    self.image = {};
    self.sampler = {};
    self.uploaded = false;
}
