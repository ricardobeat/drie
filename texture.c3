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
    // Display P3's wider gamut (sRGB's transfer curve): Renderer.output_color_space, for a
    // screen that shows more saturated colors than sRGB has (three's DisplayP3ColorSpace)
    DISPLAY_P3_COLOR_SPACE,
}

struct Texture
{
    int width;
    int height;
    // RGBA8 pixels, top row first. Freed after the upload to the GPU.
    char[] pixels;
    // The image's average color in linear light, once average_color() has worked it out.
    Vector3 average;
    bool has_average;
    // Or RGBA in linear light, for images brighter than 1 (an environment captured with hdr):
    // RGBA16F on the GPU, like three's DataTexture with HalfFloatType. Freed after the upload.
    float[] float_pixels;
    // Like three's texture.flipY: the image goes to the GPU bottom row first, so uv (0, 0) is
    // its bottom left corner, as drie's geometries (three's) expect. On for load_texture and
    // decode_texture (three's TextureLoader); off for data textures and glTF's images, whose
    // uv (0, 0) is the top left. Read at the upload.
    bool flip_y;
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
        .clear_value = { ((clear >> 16) & 0xff) / 255.0f, ((clear >> 8) & 0xff) / 255.0f, (clear & 0xff) / 255.0f, 1 } }, .label = "three-canvas" };
    self.texture.upload();
    if (!self.attachment.id) self.attachment = sg::make_view(&&{ .color_attachment = { .image = self.texture.image } });
    pass.attachments.colors[0] = self.attachment;
    sg::begin_pass(&pass);
}

fn void CanvasTexture.end(&self) => sg::end_pass();

// Decodes a PNG or JPEG held in memory (see Texture.flip_y).
fn Texture*? decode_texture(char[] encoded, ColorSpace color_space = SRGB_COLOR_SPACE, bool flip_y = true)
{
    CInt w;
    CInt h;
    char* px = imageio::decode(encoded.ptr, (usz)encoded.len, &w, &h);
    if (!px) return io::FILE_NOT_VALID~;
    defer imageio::free_pixels(px);
    Texture* t = data_texture(px[:(sz)w * h * 4], w, h, color_space);
    t.flip_y = flip_y;
    return t;
}

// Loads a PNG or JPEG file, like `new THREE.TextureLoader().load(path)` (see Texture.flip_y).
// A missing or broken file logs a message and gives a magenta placeholder.
fn Texture* load_texture(String path, ColorSpace color_space = SRGB_COLOR_SPACE, bool flip_y = true)
{
    char[]? data = file::load(mem, path);
    if (catch err = data)
    {
        io::eprintfn("three: cannot read texture '%s'", path);
        return data_texture({ 255, 0, 255, 255 }, 1, 1, color_space);
    }
    defer mem::free(data.ptr);
    Texture*? t = decode_texture(data, color_space, flip_y);
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

// Turns an RGBA8 image upside down, in place.
fn void flip_rows(char[] px, int width, int height) @local
{
    sz row = (sz)width * 4;
    if (px.len < row * height) return;
    char[] tmp = mem::new_array(char, row);
    defer mem::free(tmp.ptr);
    for (sz y = 0; y < height / 2; y++)
    {
        char[] a = px[y * row:row];
        char[] b = px[((sz)height - 1 - y) * row:row];
        tmp[..] = a[..];
        a[..] = b[..];
        b[..] = tmp[..];
    }
}

// Halves an RGBA float image with a box filter.
fn float[] downsample_float(float[] src, int w, int h, int nw, int nh) @local
{
    float[] dst = mem::new_array(float, (sz)nw * nh * 4);
    for (int y = 0; y < nh; y++)
    {
        for (int x = 0; x < nw; x++)
        {
            for (int c = 0; c < 4; c++)
            {
                float sum = 0;
                for (int j = 0; j < 2; j++)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        int sx = math::min(x * 2 + i, w - 1);
                        int sy = math::min(y * 2 + j, h - 1);
                        sum += src[((sz)sy * w + sx) * 4 + c];
                    }
                }
                dst[((sz)y * nw + x) * 4 + c] = sum / 4;
            }
        }
    }
    return dst;
}

// IEEE half floats, as RGBA16F stores them (rounded to nearest; too large becomes infinity,
// too small zero).
fn ushort float_to_half(float f)
{
    uint x = bitcast(f, uint);
    uint sign = (x >> 16) & 0x8000;
    int e = (int)((x >> 23) & 0xff) - 127 + 15;
    uint m = x & 0x7fffff;
    if (e >= 31) return (ushort)(sign | ((x & 0x7fffffff) > 0x7f800000 ? 0x7e00 : 0x7c00));
    if (e <= 0)
    {
        if (e < -10) return (ushort)sign;
        m |= 0x800000;
        uint shift = (uint)(14 - e);
        uint half_m = m >> shift;
        if ((m >> (shift - 1)) & 1) half_m++;
        return (ushort)(sign | half_m);
    }
    uint h = sign | ((uint)e << 10) | (m >> 13);
    if (m & 0x1000) h++;
    return (ushort)h;
}

fn float half_to_float(ushort h)
{
    uint sign = ((uint)h & 0x8000) << 16;
    uint e = ((uint)h >> 10) & 0x1f;
    uint m = (uint)h & 0x3ff;
    if (e == 0)
    {
        float v = (float)m / 16777216.0f;
        return sign ? -v : v;
    }
    if (e == 31) return bitcast(sign | 0x7f800000 | (m << 13), float);
    return bitcast(sign | ((e + 112) << 23) | (m << 13), float);
}

// Uploads float_pixels (and its mipmaps) as RGBA16F.
fn void Texture.upload_float(&self)
{
    float[][16] levels;
    int[16] widths;
    int[16] heights;
    int count = 1;
    if (self.flip_y)
    {
        sz row = (sz)self.width * 4;
        for (sz y = 0; y < self.height / 2; y++)
        {
            for (sz i = 0; i < row; i++)
            {
                @swap(self.float_pixels[y * row + i], self.float_pixels[((sz)self.height - 1 - y) * row + i]);
            }
        }
    }
    levels[0] = self.float_pixels;
    widths[0] = self.width;
    heights[0] = self.height;
    if (self.generate_mipmaps)
    {
        while ((widths[count - 1] > 1 || heights[count - 1] > 1) && count < 12)
        {
            widths[count] = math::max(widths[count - 1] / 2, 1);
            heights[count] = math::max(heights[count - 1] / 2, 1);
            levels[count] = downsample_float(levels[count - 1], widths[count - 1], heights[count - 1], widths[count], heights[count]);
            count++;
        }
    }
    sz total;
    for (int i = 0; i < count; i++) total += levels[i].len;
    ushort[] all = mem::new_array(ushort, total);
    defer mem::free(all.ptr);
    sz at;
    for (int i = 0; i < count; i++)
    {
        foreach (v : levels[i]) all[at++] = float_to_half(v);
    }
    self.image = sg::make_image(&&{
        .width = self.width,
        .height = self.height,
        .pixel_format = RGBA16F,
        .num_mipmaps = count,
        .data = { .ptr = all.ptr, .size = (usz)total * 2 },
    });
    self.view = sg::make_view(&&{ .texture = { .image = self.image } });
    self.sampler = sg::make_sampler(&&{
        .min_filter = LINEAR,
        .mag_filter = LINEAR,
        .mipmap_filter = count > 1 ? SgFilter.LINEAR : SgFilter.NEAREST,
        .wrap_u = sg_wrap(self.wrap_s),
        .wrap_v = sg_wrap(self.wrap_t),
    });
    for (int i = 1; i < count; i++) mem::free(levels[i].ptr);
    mem::free(self.float_pixels.ptr);
    self.float_pixels = {};
    self.uploaded = true;
}

// The image's average color in linear light (from every 7th pixel), for whoever needs one
// color for a whole surface (RadianceCascades' voxels). White for a texture without pixels.
fn Vector3 Texture.average_color(&self)
{
    if (self.has_average) return self.average;
    self.average = { 1, 1, 1 };
    sz count = self.pixels.len / 4;
    if (count > 0)
    {
        Vector3 sum;
        int n = 0;
        bool srgb = self.color_space == SRGB_COLOR_SPACE;
        for (sz i = 0; i < count; i += 7)
        {
            char[] px = self.pixels[i * 4 : 3];
            Vector3 c = { px[0] / 255.0f, px[1] / 255.0f, px[2] / 255.0f };
            if (srgb) c = { srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z) };
            sum += c;
            n++;
        }
        self.average = sum / (float)n;
    }
    self.has_average = true;
    return self.average;
}

// Uploads the pixels (and mipmaps) to the GPU. Called by the renderer.
fn void Texture.upload(&self)
{
    if (self.uploaded) return;
    self.dispose();
    if (self.float_pixels.len)
    {
        self.upload_float();
        return;
    }
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
    if (self.flip_y) flip_rows(self.pixels, self.width, self.height);
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
    self.average_color();
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
