// Scene-level box-projected reflection probes (`ReflectionProbes`) packed into a single 2D array
// texture with CPU-generated mipmaps for roughness filtering. Each probe owns an axis-aligned box
// and cross-fades over `blend` meters into neighboring probes or an unbounded fallback probe
// (`box_min == box_max`). Materials with `env_probes` set sample this array; equirectangular maps
// carrying hit distances in alpha (`Renderer.capture_environment`) apply parallax correction. Add
// maps before drawing with them so their CPU pixel buffers are still intact; `upload()` runs
// lazily from the renderer.

module drie;
import std::math;
import sokol;

// Maximum number of probes in one `ReflectionProbes` array texture.
const int MAX_REFLECTION_PROBES = 4;

// Single box-projected environment probe entry inside `ReflectionProbes`.
struct ReflectionProbe {
    Vector3 position; // World-space capture center (m).
    Vector3 box_min; // World-space influence box minimum corner (m); equal to `box_max` for fallback.
    Vector3 box_max; // World-space influence box maximum corner (m); equal to `box_min` for fallback.
    float blend; // Fade distance outside the box into neighbors or fallback (m).
    float intensity; // Scalar multiplier on reflected radiance.
    float distance_range; // Distance represented by alpha 1.0 in meters (`0` when no distances stored).
    char[] pixels; // Owned copy of the equirectangular RGBA8 pixels.
}

// Collection of up to `MAX_REFLECTION_PROBES` equirectangular probes uploaded as one 2D array
// texture and assigned via `Scene.reflection_probes`.
// 
//   ReflectionProbes* probes = drie::reflection_probes();
//   probes.add(renderer.capture_environment(scene, room_center), room_center, room_min, room_max);
//   probes.add(renderer.capture_environment(scene, outside), outside);
//   scene.reflection_probes = probes;
//   glass.env_probes = true;
struct ReflectionProbes {
    ReflectionProbe[MAX_REFLECTION_PROBES] probes; // Registered probes (`0..count-1`).
    int count; // Number of added probes.
    int width; // Shared equirectangular map width in pixels.
    int height; // Shared equirectangular map height in pixels.
    ColorSpace color_space; // Color space of the uploaded array (`SRGB_COLOR_SPACE` uses `SRGB8A8`).
    SgImage image; // Uploaded 2D array image with mip chain.
    SgView view; // Texture view bound by lit shaders.
    SgSampler sampler; // Trilinear sampler wrapping horizontally and clamping vertically.
    bool uploaded; // True when GPU array matches `probes[0..count-1]`.
}

// ---- Probe setup

// Allocates an empty `ReflectionProbes` container.
fn ReflectionProbes* reflection_probes() {
    return mem::new(ReflectionProbes);
}

// Copies `map`'s CPU pixels as a probe centered at `position` covering `box_min..box_max` (leave
// equal for a fallback sky probe) and returns the new entry for tuning `blend` and `intensity`.
// Aborts if `MAX_REFLECTION_PROBES` is exceeded, if `map.pixels` has already been released by a
// GPU upload, or if `map`'s dimensions differ from earlier probes.
fn ReflectionProbe* ReflectionProbes.add(
    &self,
    Texture* map,
    Vector3 position,
    Vector3 box_min = { },
    Vector3 box_max = { },
    float blend = 0.4
) {
    if (self.count >= MAX_REFLECTION_PROBES) abort("drie: too many reflection probes (MAX_REFLECTION_PROBES)");
    if (!map.pixels.len) abort(
        "drie: a reflection probe's map must still have its pixels (add it before it is drawn with)"
    );
    if (self.count && (map.width != self.width || map.height != self.height)) abort(
        "drie: reflection probes' maps must be of one size"
    );
    self.width = map.width;
    self.height = map.height;
    self.color_space = map.color_space;
    ReflectionProbe* p = &self.probes[self.count++];
    *p = {
        .position = position,
        .box_min = box_min,
        .box_max = box_max,
        .blend = blend,
        .intensity = 1,
        .distance_range = map.env_distance_range
    };
    p.pixels = mem::new_array(char, map.pixels.len);
    p.pixels[..] = map.pixels[..];
    self.uploaded = false;
    return p;
}

// ---- Mipmap generation & upload

// Downsamples an RGBA8 image from `w` x `h` to `nw` x `nh` with a 2x2 box filter.
fn char[] probe_downsample(char[] src, int w, int h, int nw, int nh) @local {
    char[] dst = mem::new_array(char, (sz)nw * nh * 4);
    for (int y = 0; y < nh; y++) {
        for (int x = 0; x < nw; x++) {
            for (int c = 0; c < 4; c++) {
                int sum = 0;
                for (int j = 0; j < 2; j++) {
                    for (int i = 0; i < 2; i++) {
                        sum += src[((sz)math::min(y * 2 + j, h - 1) * w + math::min(x * 2 + i, w - 1)) * 4 + c];
                    }
                }
                dst[((sz)y * nw + x) * 4 + c] = (char)((sum + 2) / 4);
            }
        }
    }
    return dst;
}

// Builds the mipmapped 2D array texture (one slice per probe, up to 12 mip levels) on the GPU
// when dirty. Called lazily by the renderer.
fn void ReflectionProbes.upload(&self) {
    if (self.uploaded || !self.count) return;
    if (self.image.id) {
        sg::destroy_view(self.view);
        sg::destroy_image(self.image);
    }
    int levels = 1;
    int lw     = self.width;
    int lh     = self.height;
    // Sokol expects mip levels packed sequentially, each containing all array slices.
    sz total   = (sz)lw * lh * 4 * self.count;
    while ((lw > 1 || lh > 1) && levels < 12) {
        lw = math::max(lw / 2, 1);
        lh = math::max(lh / 2, 1);
        total += (sz)lw * lh * 4 * self.count;
        levels++;
    }
    char[] all = mem::new_array(char, total);
    defer free(all);
    char[][MAX_REFLECTION_PROBES] level;
    for (int i = 0; i < self.count; i++) level[i] = self.probes[i].pixels;
    sz at;
    int w = self.width;
    int h = self.height;
    for (int l = 0; l < levels; l++) {
        for (int i = 0; i < self.count; i++) {
            all[at:level[i].len] = level[i][..];
            at += level[i].len;
        }
        int nw = math::max(w / 2, 1);
        int nh = math::max(h / 2, 1);
        for (int i = 0; i < self.count; i++) {
            char[] next = l + 1 < levels ? probe_downsample(level[i], w, h, nw, nh) : { };
            if (l > 0) free(level[i]);
            level[i] = next;
        }
        w = nw;
        h = nh;
    }
    self.image = sg::make_image(
        &&{
            .type = ARRAY,
            .width = self.width,
            .height = self.height,
            .num_slices = self.count,
            .num_mipmaps = levels,
            .pixel_format = self.color_space == SRGB_COLOR_SPACE ? SgPixelFormat.SRGB8A8 : SgPixelFormat.RGBA8,
            .data = { .ptr = all.ptr, .size = (usz)total },
            .label = "drie-reflection-probes",
        }
    );
    self.view = sg::make_view(&&{ .texture = { .image = self.image } });
    if (!self.sampler.id) {
        self.sampler = sg::make_sampler(
            &&{
                .min_filter = LINEAR,
                .mag_filter = LINEAR,
                .mipmap_filter = LINEAR,
                .wrap_u = REPEAT,
                .wrap_v = CLAMP_TO_EDGE,
            }
        );
    }
    self.uploaded = true;
}