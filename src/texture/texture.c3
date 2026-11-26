// 2D image textures, render-target `CanvasTexture`, cube-to-equirectangular conversion, and
// spherical-harmonic diffuse environment lighting (`EnvLight`). Supports RGBA8 (`SRGB8A8` or
// `RGBA8`) and linear RGBA float (`RGBA16F`) data with CPU mipmap generation and thread-safe
// `Texture.prepare()` ahead of GPU upload by `Renderer.render`. Constructors allocate on `mem`;
// `prepare()` and `upload()` free CPU pixel buffers unless `keep_pixels` is set, and
// `Texture.dispose()` releases GPU handles.
module drie;
import std::io, std::math, std::atomic, std::thread;
import sokol, drie::imageio;

// Texture coordinate wrapping mode beyond `[0, 1]`.
enum Wrapping {
    // Tile the texture seamlessly.
    REPEAT_WRAPPING,
    // Clamp out-of-range coordinates to the edge texel.
    CLAMP_TO_EDGE_WRAPPING,
    // Tile the texture, mirroring on every odd repeat.
    MIRRORED_REPEAT_WRAPPING,
}

// Minification and mipmap filter mode (`Texture.min_filter`) when texels are smaller than
// screen pixels. `LINEAR` and `NEAREST` sample only level 0 without mipmaps.
enum MinFilter {
    // Trilinear filtering: linear within and between the two nearest mip levels.
    LINEAR_MIPMAP_LINEAR,
    // Bilinear filtering within the nearest mip level.
    LINEAR_MIPMAP_NEAREST,
    // Bilinear filtering on level 0 only (no mipmaps).
    LINEAR,
    // Nearest texel within each of the two nearest mip levels, linearly blended between them.
    NEAREST_MIPMAP_LINEAR,
    // Nearest texel from the nearest mip level.
    NEAREST_MIPMAP_NEAREST,
    // Nearest texel on level 0 only (no mipmaps).
    NEAREST,
}

// Returns true when this minification filter samples mip levels.
fn bool MinFilter.mipmapped(self) => self != LINEAR && self != NEAREST;

// Pixel color-space interpretation for textures and `Renderer.output_color_space`.
enum ColorSpace {
    // Standard sRGB transfer curve (decoded to linear light on GPU sampling).
    SRGB_COLOR_SPACE,
    // Raw linear/data values used as stored (normal maps, roughness, metalness, height).
    NO_COLOR_SPACE,
    // Wide-gamut Display P3 primaries with the sRGB transfer curve (`Renderer.output_color_space`).
    DISPLAY_P3_COLOR_SPACE,
}

// 2D GPU image, sampler configuration, UV transform, and optional equirectangular `EnvLight`.
//
//   Texture* t = drie::load_texture("assets/textures/bark_diff.jpg");
//   t.repeat = { 4, 4 };
//   material.map = t;
struct Texture {
    // Width in texels.
    int width;
    // Height in texels.
    int height;
    // CPU RGBA8 pixels, top row first; freed by `prepare()` unless `keep_pixels` is set.
    char[] pixels;
    // Average color in linear RGB computed by `prepare()`.
    Vector3 average;
    // True once `average` has been computed.
    bool has_average;
    // CPU linear RGBA floats for HDR textures (uploaded as `RGBA16F`); freed on upload.
    float[] float_pixels;
    // Flip rows vertically at upload so UV `(0, 0)` is the bottom-left corner (default true for
    // `load_texture`/`decode_texture`, false for `data_texture` and glTF images).
    bool flip_y;
    // Color-space interpretation (`SRGB_COLOR_SPACE` vs `NO_COLOR_SPACE`).
    ColorSpace color_space;
    // Horizontal (U) wrapping mode.
    Wrapping wrap_s;
    // Vertical (V) wrapping mode.
    Wrapping wrap_t;
    // UV scale factor; the primary map's UV transform (`map`, or fallback `normal_map`/`arm_map`)
    // drives shared material UVs, while other maps apply their own.
    Vector2 repeat;
    // UV translation offset applied after `repeat`.
    Vector2 offset;
    // Counterclockwise UV rotation in radians around `center`.
    float rotation;
    // Pivot point in UV space for `rotation`.
    Vector2 center;
    // Geometry UV set index: 0 reads `uv`, 1 reads `uv1` (non-color maps only).
    char channel;
    // Generate CPU box-filtered mipmaps during `prepare()`/`upload_float()`.
    bool generate_mipmaps;
    // Treat alpha as data (height, roughness) rather than coverage so mipmaps average RGB evenly
    // instead of weighting by alpha.
    bool alpha_is_data;
    // Alpha cutout threshold (`Material.alpha_test`, 0 = disabled): scales each mip level's
    // alpha to preserve pass-rate coverage across distance (Castaño coverage-preserving mips).
    float alpha_coverage;
    // Caller-provided RGBA8 mip levels below level 0, each half the prior size; overrides
    // `generate_mipmaps` and is freed by `prepare()`.
    char[][] mipmaps;
    // Minification and mip-selection filter.
    MinFilter min_filter;
    // Retain `pixels` after `prepare()` (in GPU row order) for CPU reads like `red_at` and
    // `Material.displacement_map`.
    bool keep_pixels;
    // Max anisotropic filtering taps (clamped to 1..16; active only when all filters are linear).
    int anisotropy;
    // Use nearest-neighbor magnification (`mag_filter = NEAREST`) instead of bilinear.
    bool nearest;
    // Marks a GPU render-target texture (`CanvasTexture`) with no CPU pixel upload.
    bool render_target;
    // Distance encoding range in meters for `Renderer.capture_environment` maps (`alpha == 1`
    // maps to this distance; 0 = no distance channel).
    float env_distance_range;
    // Diffuse spherical harmonics and extracted sun direction computed on upload for 2:1
    // equirectangular maps used as `Scene.environment`.
    EnvLight env_light;
    // True once `env_light` has been computed.
    bool has_env_sh;
    // Packed RGBA8 mip chain built by `prepare()` and freed after `upload()`.
    char[] prepared;
    // Number of mip levels packed in `prepared`.
    int prepared_levels;
    // Thread-safe `prepare()` state: 0 idle, 1 running, 2 complete.
    Atomic{int} preparing;
    // GPU image handle (created lazily on first render).
    SgImage image;
    // GPU texture view handle.
    SgView view;
    // GPU sampler handle.
    SgSampler sampler;
    // True once GPU resources have been created.
    bool uploaded;
    // Source texture borrowed by `Texture.clone()`: shares GPU image/view/sampler while keeping
    // independent `repeat`, `offset`, `rotation`, `center`, and `channel`.
    Texture* source;
}

// Allocates a lightweight view sharing this texture's GPU image and sampler (uploaded once on
// the root source) with its own copy of `repeat`, `offset`, `rotation`, `center`, and `channel`.
fn Texture* Texture.clone(&self) {
    Texture* t = mem::new(Texture);
    *t = {
        .width = self.width,
        .height = self.height,
        .flip_y = self.flip_y,
        .color_space = self.color_space,
        .wrap_s = self.wrap_s,
        .wrap_t = self.wrap_t,
        .repeat = self.repeat,
        .offset = self.offset,
        .rotation = self.rotation,
        .center = self.center,
        .channel = self.channel,
        .generate_mipmaps = self.generate_mipmaps,
        .min_filter = self.min_filter,
        .anisotropy = self.anisotropy,
        .nearest = self.nearest,
        .source = self.source ?: self,
    };
    return t;
}

// Allocates a `Texture` copying raw top-down RGBA8 `rgba` (`flip_y = false`, mipmaps, 8x aniso).
fn Texture* data_texture(char[] rgba, int width, int height, ColorSpace color_space = SRGB_COLOR_SPACE) {
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

// Allocates a linear HDR `Texture` copying RGBA floats (uploaded as `RGBA16F`, `NO_COLOR_SPACE`).
fn Texture* float_texture(float[] rgba, int width, int height) {
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

// Offscreen RGBA8 render target usable as any material map (for example an in-world display).
// Draw with sokol between `begin()` and `end()` outside any other render pass:
//
//   CanvasTexture* screen = drie::canvas_texture(256, 144);
//   material.map = screen.texture;
//   screen.begin(0x001000); sdtx::draw(); screen.end();
struct CanvasTexture {
    // Backing render-target `Texture` to bind on materials.
    Texture* texture;
    // Color attachment view created lazily on the first `begin()` call.
    SgView attachment;
}

// Allocates a clamped RGBA8 `CanvasTexture` of `width` by `height` pixels on the heap (`mem`).
fn CanvasTexture* canvas_texture(int width, int height) {
    CanvasTexture* c = mem::new(CanvasTexture);
    Texture* t       = mem::new(Texture);
    *t = {
        .width = width,
        .height = height,
        .repeat = { 1, 1 },
        .wrap_s = CLAMP_TO_EDGE_WRAPPING,
        .wrap_t = CLAMP_TO_EDGE_WRAPPING,
        .render_target = true
    };
    c.texture = t;
    return c;
}

// Begins a sokol render pass targeting this canvas, clearing it to sRGB hex `clear`.
fn void CanvasTexture.begin(&self, Color clear = 0x000000) {
    SgPass pass = {
        .action.colors[0] = {
            .load_action = CLEAR,
            .clear_value = {
                ((clear >> 16) & 0xff) / 255.0f,
                ((clear >> 8) & 0xff) / 255.0f,
                (clear & 0xff) / 255.0f,
                1
            }
        },
        .label = "drie-canvas"
    };
    self.texture.upload();
    if (!self.attachment.id) self.attachment = sg::make_view(&&{ .color_attachment = { .image = self.texture.image } });
    pass.attachments.colors[0] = self.attachment;
    profile::begin_pass(&pass);
}

// Ends the sokol render pass started by `begin()`.
fn void CanvasTexture.end(&self) => sg::end_pass();

// Decodes an in-memory PNG or JPEG into a `Texture` (`flip_y = true` by default); returns
// `io::FILE_NOT_VALID` on decode failure.
fn Texture*? decode_texture(char[] encoded, ColorSpace color_space = SRGB_COLOR_SPACE, bool flip_y = true) {
    CInt w;
    CInt h;
    char* px = imageio::decode(encoded.ptr, (usz)encoded.len, &w, &h);
    if (!px) return io::FILE_NOT_VALID~;
    defer imageio::free_pixels(px);
    Texture* t = data_texture(px[:(sz)w * h * 4], w, h, color_space);
    t.flip_y = flip_y;
    return t;
}

// Loads a PNG or JPEG from `path`. Logs to stderr and returns a 1x1 magenta placeholder if
// reading or decoding fails.
fn Texture* load_texture(String path, ColorSpace color_space = SRGB_COLOR_SPACE, bool flip_y = true) {
    char[]? data = file::load(mem, path);
    if (catch err = data) {
        io::eprintfn("drie: cannot read texture '%s'", path);
        return data_texture({ 255, 0, 255, 255 }, 1, 1, color_space);
    }
    defer mem::free(data.ptr);
    Texture*? t = decode_texture(data, color_space, flip_y);
    if (catch err = t) {
        io::eprintfn("drie: cannot decode texture '%s'", path);
        return data_texture({ 255, 0, 255, 255 }, 1, 1, color_space);
    }
    return t;
}

// Loads RGB from `path` and copies the red channel of same-sized `alpha_path` into alpha
// (useful for pairing two JPEGs instead of one RGBA PNG; set `alpha_is_data` for non-coverage
// data). Missing or size-mismatched `alpha_path` logs and leaves alpha opaque.
fn Texture* load_texture_alpha(
    String path,
    String alpha_path,
    ColorSpace color_space = SRGB_COLOR_SPACE,
    bool flip_y = true
) {
    Texture* t   = load_texture(path, color_space, flip_y);
    char[]? data = file::load(mem, alpha_path);
    if (catch err = data) {
        io::eprintfn("drie: cannot read texture '%s'", alpha_path);
        return t;
    }
    defer mem::free(data.ptr);
    CInt w;
    CInt h;
    char* px = imageio::decode(data.ptr, (usz)data.len, &w, &h);
    if (!px) {
        io::eprintfn("drie: cannot decode texture '%s'", alpha_path);
        return t;
    }
    defer imageio::free_pixels(px);
    if (w != t.width || h != t.height) {
        io::eprintfn("drie: '%s' is %dx%d, not %dx%d as '%s'", alpha_path, w, h, t.width, t.height, path);
        return t;
    }
    for (sz i = 0; i < (sz)w * h; i++) t.pixels[i * 4 + 3] = px[i * 4];
    return t;
}

fn SgWrap sg_wrap(Wrapping w) @local {
    switch (w) {
        case REPEAT_WRAPPING: return REPEAT;
        case CLAMP_TO_EDGE_WRAPPING: return CLAMP_TO_EDGE;
        case MIRRORED_REPEAT_WRAPPING: return MIRRORED_REPEAT;
    }
}

// Halves an RGBA8 image with a box filter. Alpha-weighted so transparent texels don't darken edges.
fn char[] downsample(char[] src, int w, int h, int nw, int nh, bool plain_alpha = false) @local {
    char[] dst = mem::new_array(char, (sz)nw * nh * 4);
    for (int y = 0; y < nh; y++) {
        for (int x = 0; x < nw; x++) {
            uint[4] sum;
            uint asum;
            uint[3] plain;
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 2; i++) {
                    int sx  = math::min(x * 2 + i, w - 1);
                    int sy  = math::min(y * 2 + j, h - 1);
                    char* p = &src[((sz)sy * w + sx) * 4];
                    uint a  = plain_alpha ? 255 : p[3];
                    for (int c = 0; c < 3; c++) {
                        sum[c] += p[c] * a;
                        plain[c] += p[c];
                    }
                    asum += a;
                }
            }
            char* d = &dst[((sz)y * nw + x) * 4];
            for (int c = 0; c < 3; c++) d[c] = (char)(asum ? sum[c] / asum : plain[c] / 4);
            uint alpha_sum;
            if (plain_alpha) {
                for (int j = 0; j < 2; j++) {
                    for (int i = 0; i < 2; i++) alpha_sum += src[((sz)math::min(
                        y * 2 + j,
                        h - 1
                    ) * w + math::min(x * 2 + i, w - 1)) * 4 + 3];
                }
            }
            d[3] = (char)(plain_alpha ? alpha_sum / 4 : asum / 4);
        }
    }
    return dst;
}

// The share of an RGBA8 image's texels whose alpha, times `scale`, is at least `cut` (0..1).
fn float alpha_coverage_of(char[] px, int w, int h, float cut, float scale) @local {
    sz n = (sz)w * h;
    sz over;
    for (sz i = 0; i < n; i++) if ((float)px[i * 4 + 3] / 255 * scale >= cut) over++;
    return (float)over / (float)n;
}

// Bisects an alpha scale on `level` so its pass rate at `cut` matches the top mip level.
fn void keep_coverage(char[] level, int w, int h, char[] top, int tw, int th, float cut) @local {
    float want = alpha_coverage_of(top, tw, th, cut, 1);
    float lo   = 0;
    float hi   = 8;
    for (int i = 0; i < 12; i++) {
        float mid = (lo + hi) / 2;
        if (alpha_coverage_of(level, w, h, cut, mid) < want) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    float scale = (lo + hi) / 2;
    sz n        = (sz)w * h;
    for (sz i = 0; i < n; i++) level[i * 4 + 3] = (char)math::min((float)level[i * 4 + 3] * scale + 0.5f, 255.0f);
}

// Turns an RGBA8 image upside down, in place.
fn void flip_rows(char[] px, int width, int height) @local {
    sz row = (sz)width * 4;
    if (px.len < row * height) return;
    char[] tmp = mem::new_array(char, row);
    defer mem::free(tmp.ptr);
    for (sz y = 0; y < height / 2; y++) {
        char[] a = px[y * row:row];
        char[] b = px[((sz)height - 1 - y) * row:row];
        tmp[..] = a[..];
        a[..] = b[..];
        b[..] = tmp[..];
    }
}

// Halves an RGBA float image with a box filter.
fn float[] downsample_float(float[] src, int w, int h, int nw, int nh) @local {
    float[] dst = mem::new_array(float, (sz)nw * nh * 4);
    for (int y = 0; y < nh; y++) {
        for (int x = 0; x < nw; x++) {
            for (int c = 0; c < 4; c++) {
                float sum = 0;
                for (int j = 0; j < 2; j++) {
                    for (int i = 0; i < 2; i++) {
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

// ---- Half-float conversion and GPU upload

// Encodes a 32-bit float as IEEE 754 binary16 (round-to-nearest; overflows become infinity and
// underflows flush to signed zero).
fn ushort float_to_half(float f) {
    uint x    = bitcast(f, uint);
    uint sign = (x >> 16) & 0x8000;
    int e     = (int)((x >> 23) & 0xff) - 127 + 15;
    uint m    = x & 0x7fffff;
    if (e >= 31) return (ushort)(sign | ((x & 0x7fffffff) > 0x7f800000 ? 0x7e00 : 0x7c00));
    if (e <= 0) {
        if (e < -10) return (ushort)sign;
        m |= 0x800000;
        uint shift  = (uint)(14 - e);
        uint half_m = m >> shift;
        if ((m >> (shift - 1)) & 1) half_m++;
        return (ushort)(sign | half_m);
    }
    uint h = sign | ((uint)e << 10) | (m >> 13);
    if (m & 0x1000) h++;
    return (ushort)h;
}

// Decodes an IEEE 754 binary16 value into a 32-bit float.
fn float half_to_float(ushort h) {
    uint sign = ((uint)h & 0x8000) << 16;
    uint e    = ((uint)h >> 10) & 0x1f;
    uint m    = (uint)h & 0x3ff;
    if (e == 0) {
        float v = (float)m / 16777216.0f;
        return sign ? -v : v;
    }
    if (e == 31) return bitcast(sign | 0x7f800000 | (m << 13), float);
    return bitcast(sign | ((e + 112) << 23) | (m << 13), float);
}

// Generates mipmaps, converts `float_pixels` to `RGBA16F`, computes `env_light` for 2:1
// equirectangular maps, uploads to the GPU, and frees `float_pixels`.
fn void Texture.upload_float(&self) {
    float[][16] levels;
    int[16] widths;
    int[16] heights;
    int count = 1;
    if (self.flip_y) {
        sz row = (sz)self.width * 4;
        for (sz y = 0; y < self.height / 2; y++) {
            for (sz i = 0; i < row; i++) {
                @swap(self.float_pixels[y * row + i], self.float_pixels[((sz)self.height - 1 - y) * row + i]);
            }
        }
    }
    levels[0] = self.float_pixels;
    widths[0] = self.width;
    heights[0] = self.height;
    if (self.generate_mipmaps) {
        while ((widths[count - 1] > 1 || heights[count - 1] > 1) && count < 12) {
            widths[count] = math::max(widths[count - 1] / 2, 1);
            heights[count] = math::max(heights[count - 1] / 2, 1);
            levels[count] = downsample_float(
                levels[count - 1],
                widths[count - 1],
                heights[count - 1],
                widths[count],
                heights[count]
            );
            count++;
        }
    }
    sz total;
    for (int i = 0; i < count; i++) total += levels[i].len;
    ushort[] all = mem::new_array(ushort, total);
    defer mem::free(all.ptr);
    sz at;
    for (int i = 0; i < count; i++) {
        foreach (v : levels[i]) all[at++] = float_to_half(v);
    }
    self.image = sg::make_image(
        &&{
            .width = self.width,
            .height = self.height,
            .pixel_format = RGBA16F,
            .num_mipmaps = count,
            .data = { .ptr = all.ptr, .size = (usz)total * 2 },
        }
    );
    self.view = sg::make_view(&&{ .texture = { .image = self.image } });
    self.sampler = sg::make_sampler(
        &&{
            .min_filter = LINEAR,
            .mag_filter = self.nearest ? SgFilter.NEAREST : SgFilter.LINEAR,
            .mipmap_filter = count > 1 ? SgFilter.LINEAR : SgFilter.NEAREST,
            .wrap_u = sg_wrap(self.wrap_s),
            .wrap_v = sg_wrap(self.wrap_t),
        }
    );
    if (self.width == 2 * self.height) {
        int l = env_sh_level(widths[:count]);
        self.env_light = env_light(levels[l], { }, widths[l], heights[l], false);
        self.has_env_sh = true;
    }
    for (int i = 1; i < count; i++) mem::free(levels[i].ptr);
    mem::free(self.float_pixels.ptr);
    self.float_pixels = { };
    self.uploaded = true;
}

// Returns the texture's average color in linear RGB (sampled every 7th pixel, used by
// `RadianceCascades` voxelization), calling `prepare()` if needed. Safe from any thread;
// returns white for textures without RGBA8 pixels.
fn Vector3 Texture.average_color(&self) {
    if (self.source) return self.source.average_color();
    self.prepare();
    return self.has_average ? self.average : (Vector3) { 1, 1, 1 };
}

// Computes `average` in linear RGB from every 7th pixel of `pixels`.
fn void Texture.find_average(&self) {
    self.average = { 1, 1, 1 };
    sz count = self.pixels.len / 4;
    if (count > 0) {
        Vector3 sum;
        int n     = 0;
        bool srgb = self.color_space == SRGB_COLOR_SPACE;
        for (sz i = 0; i < count; i += 7) {
            char[] px = self.pixels[i * 4:3];
            Vector3 c = { px[0] / 255.0f, px[1] / 255.0f, px[2] / 255.0f };
            if (srgb) c = { srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z) };
            sum += c;
            n++;
        }
        self.average = sum / (float)n;
    }
    self.has_average = true;
}

// Creates GPU image, view, and sampler resources from prepared CPU data (or borrows them from
// `source` on a clone). Called lazily by `Renderer.render`; frees `prepared` on completion.
fn void Texture.upload(&self) {
    // Clones borrow their root source's GPU handles.
    if (self.source) {
        Texture* s = self.source;
        s.upload();
        self.image = s.image;
        self.view = s.view;
        self.sampler = s.sampler;
        self.width = s.width;
        self.height = s.height;
        self.uploaded = true;
        return;
    }
    if (self.uploaded) return;
    self.dispose();
    if (self.float_pixels.len) {
        self.upload_float();
        return;
    }
    if (self.render_target) {
        self.image = sg::make_image(
            &&{
                .usage = { .color_attachment = true },
                .width = self.width,
                .height = self.height,
                .pixel_format = RGBA8,
                .sample_count = 1,
            }
        );
        self.view = sg::make_view(&&{ .texture = { .image = self.image } });
        self.sampler = sg::make_sampler(
            &&{
                .min_filter = LINEAR,
                .mag_filter = LINEAR,
                .wrap_u = sg_wrap(self.wrap_s),
                .wrap_v = sg_wrap(self.wrap_t)
            }
        );
        // Flip V on render-target textures for top-left-origin backends (Metal/D3D).
        if (depth_zero_to_one()) {
            self.repeat.y = -math::abs(self.repeat.y);
            self.offset.y = 1;
        }
        self.uploaded = true;
        return;
    }
    self.prepare();
    SgImageDesc desc = {
        .width = self.width,
        .height = self.height,
        .pixel_format = self.color_space == SRGB_COLOR_SPACE ? SgPixelFormat.SRGB8A8 : SgPixelFormat.RGBA8,
        .num_mipmaps = self.prepared_levels,
        .data = { .ptr = self.prepared.ptr, .size = (usz)self.prepared.len },
    };
    self.image = sg::make_image(&desc);
    self.view = sg::make_view(&&{ .texture = { .image = self.image } });
    MinFilter f        = self.min_filter;
    bool nearest_min   = f == NEAREST || f == NEAREST_MIPMAP_NEAREST || f == NEAREST_MIPMAP_LINEAR;
    bool nearest_level = f == NEAREST_MIPMAP_NEAREST || f == LINEAR_MIPMAP_NEAREST || self.prepared_levels == 1;
    // Enable anisotropic filtering only when min, mag, and mip filters are all linear.
    bool anisotropic   = self.prepared_levels > 1 && !nearest_min && !nearest_level && !self.nearest;
    self.sampler = sg::make_sampler(
        &&{
            .min_filter = nearest_min ? SgFilter.NEAREST : SgFilter.LINEAR,
            .mag_filter = self.nearest ? SgFilter.NEAREST : SgFilter.LINEAR,
            .mipmap_filter = nearest_level ? SgFilter.NEAREST : SgFilter.LINEAR,
            .wrap_u = sg_wrap(self.wrap_s),
            .wrap_v = sg_wrap(self.wrap_t),
            .max_anisotropy = anisotropic ? (uint)math::clamp(self.anisotropy, 1, 16) : 1,
        }
    );
    mem::free(self.prepared.ptr);
    self.prepared = { };
    self.uploaded = true;
}

// Performs CPU-side upload work for an RGBA8 texture: averages color, flips rows when `flip_y`,
// builds and packs the mip chain, computes `env_light` for 2:1 images, and frees `pixels`
// (unless `keep_pixels` is set). Safe to call ahead of `upload()` from worker threads; runs
// once and blocks concurrent callers until done.
fn void Texture.prepare(&self) {
    int expected = 0;
    if (!self.preparing.compare_exchange(&expected, 1)) {
        while (self.preparing.load() != 2) thread::yield();
        return;
    }
    defer self.preparing.store(2);
    if (self.float_pixels.len || self.render_target || !self.pixels.len) return;
    self.find_average();
    if (self.flip_y) flip_rows(self.pixels, self.width, self.height);
    char[][16] levels;
    int count = 1;
    levels[0] = self.pixels;
    if (self.mipmaps.len && self.min_filter.mipmapped()) {
        int w = self.width;
        int h = self.height;
        foreach (level : self.mipmaps) {
            if (count == 16 || (w == 1 && h == 1)) break;
            w = math::max(w / 2, 1);
            h = math::max(h / 2, 1);
            if (level.len < (sz)w * h * 4) break;
            if (self.flip_y) flip_rows(level, w, h);
            levels[count++] = level;
        }
    } else if (self.generate_mipmaps && self.min_filter.mipmapped()) {
        int w = self.width;
        int h = self.height;
        while ((w > 1 || h > 1) && count < 12) {
            int nw = math::max(w / 2, 1);
            int nh = math::max(h / 2, 1);
            levels[count] = downsample(levels[count - 1], w, h, nw, nh, self.alpha_is_data);
            if (self.alpha_coverage > 0 && !self.alpha_is_data) keep_coverage(
                levels[count],
                nw,
                nh,
                levels[0],
                self.width,
                self.height,
                self.alpha_coverage
            );
            w = nw;
            h = nh;
            count++;
        }
    }
    // Free any trailing unused custom mip levels; used levels are freed below with generated ones.
    if (self.mipmaps.len) {
        foreach (i, level : self.mipmaps) if (i + 1 >= count) mem::free(level.ptr);
        mem::free(self.mipmaps.ptr);
        self.mipmaps = { };
    }
    // Pack all mip levels contiguously, largest first, for sokol's `SgImageDesc`.
    sz total;
    for (int i = 0; i < count; i++) total += levels[i].len;
    self.prepared = mem::new_array(char, total);
    self.prepared_levels = count;
    sz at;
    for (int i = 0; i < count; i++) {
        self.prepared[at:levels[i].len] = levels[i][..];
        at += levels[i].len;
    }
    if (self.width == 2 * self.height) {
        int[16] widths;
        for (int i = 0; i < count; i++) widths[i] = math::max(self.width >> i, 1);
        int l = env_sh_level(widths[:count]);
        self.env_light = env_light(
            { },
            levels[l],
            widths[l],
            math::max(self.height >> l, 1),
            self.color_space == SRGB_COLOR_SPACE
        );
        self.has_env_sh = true;
    }
    for (int i = 1; i < count; i++) mem::free(levels[i].ptr);
    if (self.keep_pixels) return;
    mem::free(self.pixels.ptr);
    self.pixels = { };
}

// Bilinearly samples the red channel in 0..1 (decoded to linear light for sRGB textures) at
// `uv` using `wrap_s`/`wrap_t`, for a prepared texture with `keep_pixels` set. Returns 0 when
// CPU pixels are unavailable.
fn float Texture.red_at(&self, Vector2 uv) {
    if (self.source) return self.source.red_at(uv);
    if (!self.keep_pixels || self.preparing.load() != 2 || !self.pixels.len) return 0;
    float x  = wrap_coord(uv.x, self.wrap_s) * self.width - 0.5f;
    float y  = wrap_coord(uv.y, self.wrap_t) * self.height - 0.5f;
    int x0   = (int)math::floor(x);
    int y0   = (int)math::floor(y);
    float fx = x - x0;
    float fy = y - y0;
    float r  = 0;
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            int px  = wrap_texel(x0 + i, self.width, self.wrap_s);
            int py  = wrap_texel(y0 + j, self.height, self.wrap_t);
            float c = self.pixels[((sz)py * self.width + px) * 4] / 255.0f;
            if (self.color_space == SRGB_COLOR_SPACE) c = srgb_to_linear(c);
            r += c * (i == 0 ? 1 - fx : fx) * (j == 0 ? 1 - fy : fy);
        }
    }
    return r;
}

fn float wrap_coord(float t, Wrapping w) @local {
    switch (w) {
        case REPEAT_WRAPPING: return t - math::floor(t);
        case CLAMP_TO_EDGE_WRAPPING: return math::clamp(t, 0, 1);
        case MIRRORED_REPEAT_WRAPPING:
            float f = t - 2 * math::floor(t / 2);
            return f > 1 ? 2 - f : f;
    }
}

fn int wrap_texel(int i, int n, Wrapping w) @local {
    if (w == REPEAT_WRAPPING) return ((i % n) + n) % n;
    return math::clamp(i, 0, n - 1);
}

// Releases GPU image, view, and sampler handles (clones clear only their borrowed references).
fn void Texture.dispose(&self) {
    // Clones borrow GPU handles from `source` and must not destroy them.
    if (self.source) {
        self.view = { };
        self.image = { };
        self.sampler = { };
        self.uploaded = false;
        return;
    }
    if (self.view.id) sg::destroy_view(self.view);
    if (self.image.id) sg::destroy_image(self.image);
    if (self.sampler.id) sg::destroy_sampler(self.sampler);
    self.view = { };
    self.image = { };
    self.sampler = { };
    self.uploaded = false;
}

// ---- Environment light and spherical harmonics

// Selects the largest mip level at most 128 px wide for SH integration.
fn int env_sh_level(int[] widths) @local {
    for (int i = 0; i < widths.len; i++) {
        if (widths[i] <= 128) return i;
    }
    return widths.len - 1;
}

// Evaluates the 9 unnormalized real spherical harmonic basis polynomials up to band 2 for `d`.
fn void sh_basis(Vector3 d, float[9]* b) @local {
    *b = { 1, d.y, d.z, d.x, d.x * d.y, d.y * d.z, 3 * d.z * d.z - 1, d.x * d.z, d.x * d.x - d.y * d.y };
}

// Real spherical harmonic normalization constants in `sh_basis` order (Ramamoorthi & Hanrahan 2001).
const float[9] SH_K      = {
    0.282095f,
    0.488603f,
    0.488603f,
    0.488603f,
    1.092548f,
    1.092548f,
    0.315392f,
    1.092548f,
    0.546274f
};
// Harmonic band index (0, 1, or 2) for each of the 9 SH coefficients.
const float[9] SH_BAND   = { 0, 1, 1, 1, 2, 2, 2, 2, 2 };
// Clamped-cosine irradiance convolution factors for SH bands 0, 1, and 2.
const float[3] SH_COSINE = { PI, 2 * PI / 3, PI / 4 };

// Diffuse irradiance of an equirectangular environment map split into band-2 spherical
// harmonics (`sh`) plus an optional separated directional sun (`sun_dir`, `sun`).
struct EnvLight {
    // Pre-scaled band-2 SH irradiance coefficients: `irradiance(n) = sum sh[i] * basis_i(n)`.
    Vector3[9] sh;
    // Unit direction pointing toward the extracted sun.
    Vector3 sun_dir;
    // Direct irradiance from the extracted sun on a surface facing `sun_dir` (0 = no sun).
    Vector3 sun;
}

// Evaluates diffuse irradiance on a surface with unit normal `n` (matches shader `env_irradiance`).
fn Vector3 EnvLight.irradiance(&self, Vector3 n) {
    float[9] b;
    sh_basis(n, &b);
    Vector3 e;
    for (int k = 0; k < 9; k++) e += self.sh[k] * b[k];
    e = { math::max(e.x, 0.0f), math::max(e.y, 0.0f), math::max(e.z, 0.0f) };
    return e + self.sun * math::max(n.dot(self.sun_dir), 0.0f);
}

// Folds the separated `sun` back into 9 band-2 SH irradiance coefficients (in `sh_basis` order)
// using the clamped-cosine Legendre weights (`1/4`, `1/2`, `5/16`).
fn Vector3[9] EnvLight.harmonics(&self) {
    Vector3[9] e   = self.sh;
    Vector3 l      = self.sun_dir;
    Vector3 s      = self.sun;
    // Project `s * (1/4 + 1/2 cos + 5/16 * (3 cos^2 - 1) / 2)` into the 9 basis polynomials.
    const float C2 = 0.46875f;
    float flat     = (l.x * l.x + l.y * l.y) / 2;
    float zz       = (l.z * l.z - flat) / 3;
    e[0] += s * (0.09375f + C2 * (flat + zz));
    e[1] += s * (0.5f * l.y);
    e[2] += s * (0.5f * l.z);
    e[3] += s * (0.5f * l.x);
    e[4] += s * (C2 * 2 * l.x * l.y);
    e[5] += s * (C2 * 2 * l.y * l.z);
    e[6] += s * (C2 * zz);
    e[7] += s * (C2 * 2 * l.x * l.z);
    e[8] += s * (C2 * (l.x * l.x - l.y * l.y) / 2);
    return e;
}

// Minimum peak-to-mean luminance ratio required to separate a sun disk out of the SH lobes.
const float SUN_CONTRAST = 30;
// Angular radius in radians integrated around the peak pixel when extracting a sun disk.
const float SUN_RADIUS   = 8 * DEG2RAD;

// Projects a `w x h` equirectangular radiance image (`floats` linear RGBA, or `bytes` RGBA8
// decoded from sRGB when `srgb` is true) into an `EnvLight`. Peaks exceeding `SUN_CONTRAST`
// times the spherical mean are extracted within `SUN_RADIUS` into `sun`/`sun_dir` above the
// surrounding annulus (`SUN_RADIUS..2 * SUN_RADIUS`) to avoid SH ringing.
fn EnvLight env_light(float[] floats, char[] bytes, int w, int h, bool srgb) {
    float[] px = mem::new_array(float, (sz)w * h * 4);
    defer free(px);
    float mean;
    float peak = -1;
    Vector3 peak_dir;
    for (int y = 0; y < h; y++) {
        float phi = ((float)y + 0.5f) / h * PI;
        for (int x = 0; x < w; x++) {
            sz i = ((sz)y * w + x) * 4;
            Vector3 c;
            if (floats.len) {
                c = { floats[i], floats[i + 1], floats[i + 2] };
            } else {
                c = { bytes[i] / 255.0f, bytes[i + 1] / 255.0f, bytes[i + 2] / 255.0f };
                if (srgb) c = { srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z) };
            }
            px[i] = c.x;
            px[i + 1] = c.y;
            px[i + 2] = c.z;
            float lum = luminance(c);
            mean += lum * math::sin(phi);
            if (lum > peak) {
                peak = lum;
                peak_dir = env_dir(x, y, w, h);
            }
        }
    }
    // Solid-angle weighted spherical mean (`sin(phi)` weights each latitude row).
    mean /= (float)w * h * 2 / PI;
    EnvLight env;
    if (mean > 0 && peak > SUN_CONTRAST * mean) {
        // Measure surrounding sky luminance in `[SUN_RADIUS, 2 * SUN_RADIUS]`, then extract excess.
        Vector3 around;
        float around_w;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float a = math::acos(math::clamp(env_dir(x, y, w, h).dot(peak_dir), -1.0f, 1.0f));
                if (a < SUN_RADIUS || a > 2 * SUN_RADIUS) continue;
                float o = env_omega(y, w, h);
                sz i    = ((sz)y * w + x) * 4;
                around += (Vector3) { px[i], px[i + 1], px[i + 2] } * o;
                around_w += o;
            }
        }
        if (around_w > 0) around /= around_w;
        Vector3 toward;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                Vector3 d = env_dir(x, y, w, h);
                if (d.dot(peak_dir) < math::cos(SUN_RADIUS)) continue;
                sz i = ((sz)y * w + x) * 4;
                Vector3 extra = {
                    math::max(px[i] - around.x, 0.0f),
                    math::max(px[i + 1] - around.y, 0.0f),
                    math::max(px[i + 2] - around.z, 0.0f)
                };
                float o = env_omega(y, w, h);
                env.sun += extra * o;
                toward += d * (luminance(extra) * o);
                px[i] -= extra.x;
                px[i + 1] -= extra.y;
                px[i + 2] -= extra.z;
            }
        }
        env.sun_dir = toward.length() > 0 ? toward.normalize() : peak_dir;
    }
    env.sh = env_sh_irradiance(px, { }, w, h, false);
    return env;
}

fn float luminance(Vector3 c) @local => 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;

// Direction vector of pixel `(x, y)` in a `w x h` equirectangular map (+Y at row 0).
fn Vector3 env_dir(int x, int y, int w, int h) @local {
    float phi   = ((float)y + 0.5f) / h * PI;
    float theta = (((float)x + 0.5f) / w - 0.5f) * 2 * PI;
    return { math::cos(theta) * math::sin(phi), math::cos(phi), math::sin(theta) * math::sin(phi) };
}

fn float env_omega(int y, int w, int h) @local => (2 * PI / w) * (PI / h) * math::sin(((float)y + 0.5f) / h * PI);

// Projects a `w x h` equirectangular radiance image (row 0 at +Y; `floats` linear RGBA or
// `bytes` RGBA8 decoded when `srgb` is true) onto band-2 SH irradiance coefficients with basis
// constants folded in. Applies minimal Sloan (2017) windowing so no sampled direction falls
// below -3% of the band-0 average; shaders clamp any remaining negative tail to zero.
fn Vector3[9] env_sh_irradiance(float[] floats, char[] bytes, int w, int h, bool srgb) {
    Vector3[9] l;
    float[9] b;
    for (int y = 0; y < h; y++) {
        float phi     = ((float)y + 0.5f) / h * PI;
        float sin_phi = math::sin(phi);
        float d_omega = (2 * PI / w) * (PI / h) * sin_phi;
        for (int x = 0; x < w; x++) {
            float theta = (((float)x + 0.5f) / w - 0.5f) * 2 * PI;
            Vector3 d   = { math::cos(theta) * sin_phi, math::cos(phi), math::sin(theta) * sin_phi };
            sz i        = ((sz)y * w + x) * 4;
            Vector3 c;
            if (floats.len) {
                c = { floats[i], floats[i + 1], floats[i + 2] };
            } else {
                c = { bytes[i] / 255.0f, bytes[i + 1] / 255.0f, bytes[i + 2] / 255.0f };
                if (srgb) c = { srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z) };
            }
            sh_basis(d, &b);
            for (int k = 0; k < 9; k++) l[k] += c * (b[k] * SH_K[k] * d_omega);
        }
    }
    // Fold basis and cosine-lobe constants into each coefficient so evaluation is a plain dot sum.
    Vector3[9] e;
    for (int k = 0; k < 9; k++) e[k] = l[k] * (SH_COSINE[(int)SH_BAND[k]] * SH_K[k]);
    if (sh_min(e) >= 0) return e;
    // Bisect the smallest windowing factor `1 / (1 + lambda * l^2 * (l + 1)^2)` meeting the floor.
    float lo = 0;
    float hi = 1;
    while (sh_min(sh_window(e, hi)) < 0 && hi < 1e6f) hi *= 4;
    for (int i = 0; i < 24; i++) {
        float mid = (lo + hi) / 2;
        if (sh_min(sh_window(e, mid)) < 0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return sh_window(e, hi);
}

fn Vector3[9] sh_window(Vector3[9] e, float lambda) @local {
    Vector3[9] out;
    for (int k = 0; k < 9; k++) {
        float band = SH_BAND[k];
        out[k] = e[k] / (1 + lambda * band * band * (band + 1) * (band + 1));
    }
    return out;
}

// Minimum channel value over 512 Fibonacci-sphere directions relative to the -3% band-0 floor.
fn float sh_min(Vector3[9] e) @local {
    float lowest = 1e30f;
    float[9] b;
    sh_basis({ 0, 1, 0 }, &b);
    Vector3 floor = e[0] * (b[0] * -0.03f);
    for (int i = 0; i < 512; i++) {
        float y = 1 - (i + 0.5f) / 256;
        float r = math::sqrt(math::max(0.0f, 1 - y * y));
        float a = i * 2.39996323f;
        sh_basis({ math::cos(a) * r, y, math::sin(a) * r }, &b);
        Vector3 v;
        for (int k = 0; k < 9; k++) v += e[k] * b[k];
        v -= floor;
        lowest = math::min(lowest, math::min(v.x, math::min(v.y, v.z)));
    }
    return lowest;
}

// ---- Cube maps

// Converts six `size x size` cube-map faces in GL order (`+X, -X, +Y, -Y, +Z, -Z`, top row
// first, X-mirrored to match cube environment sampling) into a linear HDR equirectangular
// `Texture` (`4 * size x 2 * size`, capped at 4096 wide) for `Scene.environment` or
// `Material.env_map`. Supply either linear RGBA `floats` or RGBA8 `bytes` (decoded from sRGB
// when `srgb` is true).
fn Texture* cube_texture(float[][6] floats, char[][6] bytes, int size, bool srgb = true) {
    float[256] lut;
    for (int i = 0; i < 256; i++) lut[i] = srgb ? srgb_to_linear(i / 255.0f) : i / 255.0f;
    bool byte_faces = bytes[0].len != 0;
    int w           = min(size * 4, 4096);
    int h           = w / 2;
    float[] px      = mem::new_array(float, (sz)w * h * 4);
    defer mem::free(px.ptr);
    for (int y = 0; y < h; y++) {
        // Equirectangular convention: v = 0 straight up (+Y), u = 0.5 toward +X (see `shading::env_uv`).
        float theta = ((float)y + 0.5f) / h * (float)math::PI;
        float st    = math::sin(theta);
        float ct    = math::cos(theta);
        for (int x = 0; x < w; x++) {
            float phi = (((float)x + 0.5f) / w - 0.5f) * 2 * (float)math::PI;
            // Mirror X to match cube environment map orientation.
            Vector3 d = { -math::cos(phi) * st, ct, math::sin(phi) * st };
            int face;
            float sc;
            float tc;
            float ma;
            Vector3 a = { math::abs(d.x), math::abs(d.y), math::abs(d.z) };
            // Select GL cube face and major-axis projection coordinates.
            if (a.x >= a.y && a.x >= a.z) {
                face = d.x > 0 ? 0 : 1;
                sc = d.x > 0 ? -d.z : d.z;
                tc = -d.y;
                ma = a.x;
            } else if (a.y >= a.z) {
                face = d.y > 0 ? 2 : 3;
                sc = d.x;
                tc = d.y > 0 ? d.z : -d.z;
                ma = a.y;
            } else {
                face = d.z > 0 ? 4 : 5;
                sc = d.z > 0 ? d.x : -d.x;
                tc = -d.y;
                ma = a.z;
            }
            float s  = ((sc / ma + 1) * 0.5f) * size - 0.5f;
            float t  = ((tc / ma + 1) * 0.5f) * size - 0.5f;
            int s0   = (int)math::floor(s);
            int t0   = (int)math::floor(t);
            float fs = s - s0;
            float ft = t - t0;
            sz o     = ((sz)y * w + x) * 4;
            for (int k = 0; k < 4; k++) {
                int si   = math::clamp(s0 + (k & 1), 0, size - 1);
                int ti   = math::clamp(t0 + (k >> 1), 0, size - 1);
                float wk = ((k & 1) != 0 ? fs : 1 - fs) * ((k >> 1) != 0 ? ft : 1 - ft);
                sz i     = ((sz)ti * size + si) * 4;
                if (byte_faces) {
                    for (int c = 0; c < 3; c++) px[o + c] += lut[bytes[face][i + c]] * wk;
                } else {
                    for (int c = 0; c < 3; c++) px[o + c] += floats[face][i + c] * wk;
                }
            }
            px[o + 3] = 1;
        }
    }
    return float_texture(px, w, h);
}

// Decodes six square same-sized PNG or JPEG cube faces (`+X, -X, +Y, -Y, +Z, -Z`) from memory
// into a linear equirectangular `Texture`; returns `io::FILE_NOT_VALID` on decode or size mismatch.
fn Texture*? decode_cube_texture(char[][6] encoded, bool srgb = true) {
    char[][6] faces;
    char*[6] decoded;
    defer for (int f = 0; f < 6; f++) if (decoded[f]) imageio::free_pixels(decoded[f]);
    int size = 0;
    for (int f = 0; f < 6; f++) {
        CInt w;
        CInt h;
        decoded[f] = imageio::decode(encoded[f].ptr, (usz)encoded[f].len, &w, &h);
        if (!decoded[f] || w != h || (f > 0 && w != size)) return io::FILE_NOT_VALID~;
        size = w;
        faces[f] = decoded[f][:(sz)w * h * 4];
    }
    float[][6] none;
    return cube_texture(none, faces, size, srgb);
}