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
    // As Scene.environment, the light it puts on a surface facing each way: its irradiance in
    // spherical harmonics (env_sh_irradiance), worked out at the upload of a 2:1
    // (longitude/latitude) image.
    Vector3[9] env_sh;
    bool has_env_sh;
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

// A texture from RGBA floats in linear light (copied); see Texture.float_pixels.
fn Texture* float_texture(float[] rgba, int width, int height)
{
    Texture* t = mem::new(Texture);
    *t = {
        .width = width,
        .height = height,
        .color_space = NO_COLOR_SPACE,
        .repeat = { 1, 1 },
        .generate_mipmaps = true,
        .anisotropy = 8,
    };
    t.float_pixels = mem::new_array(float, (sz)width * height * 4);
    t.float_pixels[..] = rgba[:(sz)width * height * 4];
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
    if (self.width == 2 * self.height)
    {
        int l = env_sh_level(widths[:count]);
        self.env_sh = env_sh_irradiance(levels[l], {}, widths[l], heights[l], false);
        self.has_env_sh = true;
    }
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
    if (self.width == 2 * self.height)
    {
        int[16] widths;
        for (int i = 0; i < count; i++) widths[i] = math::max(self.width >> i, 1);
        int l = env_sh_level(widths[:count]);
        self.env_sh = env_sh_irradiance({}, levels[l], widths[l], math::max(self.height >> l, 1),
                                        self.color_space == SRGB_COLOR_SPACE);
        self.has_env_sh = true;
    }
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

// The mip level to work out env_sh from: the largest at most 128 pixels wide, which still holds
// all of the image's light (each level averages the one above).
fn int env_sh_level(int[] widths) @local
{
    for (int i = 0; i < widths.len; i++)
    {
        if (widths[i] <= 128) return i;
    }
    return widths.len - 1;
}

// The 9 spherical harmonics up to band 2 at direction d, without their constants (those are in
// the coefficients). The shader's env_irradiance evaluates the same 9.
fn void sh_basis(Vector3 d, float[9]* b) @local
{
    *b = { 1, d.y, d.z, d.x, d.x * d.y, d.y * d.z, 3 * d.z * d.z - 1, d.x * d.z, d.x * d.x - d.y * d.y };
}

// Constants of the real spherical harmonics in sh_basis' order, and the cosine lobe's factor per
// band (Ramamoorthi & Hanrahan 2001, "An Efficient Representation for Irradiance Environment Maps").
const float[9] SH_K = { 0.282095f, 0.488603f, 0.488603f, 0.488603f, 1.092548f, 1.092548f, 0.315392f, 1.092548f, 0.546274f };
const float[9] SH_BAND = { 0, 1, 1, 1, 2, 2, 2, 2, 2 };
const float[3] SH_COSINE = { PI, 2 * PI / 3, PI / 4 };

// The irradiance of a longitude/latitude image of radiance (rows as uploaded: the first at the
// top, y up), as 9 coefficients c with irradiance(n) = sum c[i] * basis_i(n): the radiance
// projected onto spherical harmonics up to band 2, convolved with the cosine lobe, and windowed
// as little as keeps it from going negative where a small, bright light (a sun) would otherwise
// ring on the far side (Sloan 2017, "Deringing Spherical Harmonics"). Pixels are `floats` (RGBA,
// linear) or `bytes` (RGBA8, sRGB when `srgb`).
fn Vector3[9] env_sh_irradiance(float[] floats, char[] bytes, int w, int h, bool srgb)
{
    Vector3[9] l;
    float[9] b;
    for (int y = 0; y < h; y++)
    {
        float phi = ((float)y + 0.5f) / h * PI;
        float sin_phi = math::sin(phi);
        float d_omega = (2 * PI / w) * (PI / h) * sin_phi;
        for (int x = 0; x < w; x++)
        {
            float theta = (((float)x + 0.5f) / w - 0.5f) * 2 * PI;
            Vector3 d = { math::cos(theta) * sin_phi, math::cos(phi), math::sin(theta) * sin_phi };
            sz i = ((sz)y * w + x) * 4;
            Vector3 c;
            if (floats.len)
            {
                c = { floats[i], floats[i + 1], floats[i + 2] };
            }
            else
            {
                c = { bytes[i] / 255.0f, bytes[i + 1] / 255.0f, bytes[i + 2] / 255.0f };
                if (srgb) c = { srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z) };
            }
            sh_basis(d, &b);
            for (int k = 0; k < 9; k++) l[k] += c * (b[k] * SH_K[k] * d_omega);
        }
    }
    // irradiance, with the basis' constants folded in so the shader only sums
    Vector3[9] e;
    for (int k = 0; k < 9; k++) e[k] = l[k] * (SH_COSINE[(int)SH_BAND[k]] * SH_K[k]);
    if (sh_min(e) >= 0) return e;
    // the least windowing, 1 / (1 + lambda * l^2 (l + 1)^2) for band l, that leaves no direction negative
    float lo = 0;
    float hi = 1;
    while (sh_min(sh_window(e, hi)) < 0 && hi < 1e6f) hi *= 4;
    for (int i = 0; i < 24; i++)
    {
        float mid = (lo + hi) / 2;
        if (sh_min(sh_window(e, mid)) < 0)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    return sh_window(e, hi);
}

fn Vector3[9] sh_window(Vector3[9] e, float lambda) @local
{
    Vector3[9] out;
    for (int k = 0; k < 9; k++)
    {
        float band = SH_BAND[k];
        out[k] = e[k] / (1 + lambda * band * band * (band + 1) * (band + 1));
    }
    return out;
}

// The smallest value of any channel over 512 directions spread evenly over the sphere.
fn float sh_min(Vector3[9] e) @local
{
    float lowest = 1e30f;
    float[9] b;
    for (int i = 0; i < 512; i++)
    {
        float y = 1 - (i + 0.5f) / 256;
        float r = math::sqrt(math::max(0.0f, 1 - y * y));
        float a = i * 2.39996323f;
        sh_basis({ math::cos(a) * r, y, math::sin(a) * r }, &b);
        Vector3 v;
        for (int k = 0; k < 9; k++) v += e[k] * b[k];
        lowest = math::min(lowest, math::min(v.x, math::min(v.y, v.z)));
    }
    return lowest;
}
