// Reflection probes for a whole scene, like three's scene-level probes (box-projected, blended
// where their boxes meet, kept as the layers of one array texture): each environment map
// belongs to a box, usually a room, and a material with `env_probes` set mirrors the map of
// the box each of its points is in, whichever room the camera stands in. Where two boxes meet
// (a doorway) the two maps cross-fade over `blend` meters; a probe without a box (box_min ==
// box_max) is the one for everywhere else, the sky's.
//
//   ReflectionProbes* probes = drie::reflection_probes();
//   probes.add(renderer.capture_environment(scene, room_center), room_center, room_min, room_max);
//   probes.add(renderer.capture_environment(scene, outside), outside);
//   scene.reflection_probes = probes;
//   glass.env_probes = true;
//
// The maps are equirectangular textures of one size (capture_environment's), added before
// they are first drawn with (their pixels are copied; a texture gives its pixels up when it
// goes to the GPU). A map with distances (capture_environment's `distance`) is followed
// through the room as a material's own boxed env_map is.
module drie;
import std::math;
import sokol;

const int MAX_REFLECTION_PROBES = 4;

struct ReflectionProbe
{
    Vector3 position;
    Vector3 box_min;
    Vector3 box_max;
    // how far outside the box it fades out, into its neighbours or the probe for everywhere else (m)
    float blend;
    float intensity;
    // what its map's alpha 1 stands for in meters (0: the map holds no distances)
    float distance_range;
    char[] pixels;
}

struct ReflectionProbes
{
    ReflectionProbe[MAX_REFLECTION_PROBES] probes;
    int count;
    int width;
    int height;
    ColorSpace color_space;
    SgImage image;
    SgView view;
    SgSampler sampler;
    bool uploaded;
}

fn ReflectionProbes* reflection_probes()
{
    return mem::new(ReflectionProbes);
}

// Adds a probe: `map` as seen from `position`, for the points inside box_min..box_max (none:
// for everywhere outside the other probes' boxes). Returns it, for its blend and intensity.
fn ReflectionProbe* ReflectionProbes.add(&self, Texture* map, Vector3 position, Vector3 box_min = {}, Vector3 box_max = {},
                                         float blend = 0.4)
{
    if (self.count >= MAX_REFLECTION_PROBES) abort("drie: too many reflection probes (MAX_REFLECTION_PROBES)");
    if (!map.pixels.len) abort("drie: a reflection probe's map must still have its pixels (add it before it is drawn with)");
    if (self.count && (map.width != self.width || map.height != self.height)) abort("drie: reflection probes' maps must be of one size");
    self.width = map.width;
    self.height = map.height;
    self.color_space = map.color_space;
    ReflectionProbe* p = &self.probes[self.count++];
    *p = { .position = position, .box_min = box_min, .box_max = box_max, .blend = blend, .intensity = 1,
           .distance_range = map.env_distance_range };
    p.pixels = mem::new_array(char, map.pixels.len);
    p.pixels[..] = map.pixels[..];
    self.uploaded = false;
    return p;
}

// Halves an RGBA8 image with a box filter.
fn char[] probe_downsample(char[] src, int w, int h, int nw, int nh) @local
{
    char[] dst = mem::new_array(char, (sz)nw * nh * 4);
    for (int y = 0; y < nh; y++)
    {
        for (int x = 0; x < nw; x++)
        {
            for (int c = 0; c < 4; c++)
            {
                int sum = 0;
                for (int j = 0; j < 2; j++)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        sum += src[((sz)math::min(y * 2 + j, h - 1) * w + math::min(x * 2 + i, w - 1)) * 4 + c];
                    }
                }
                dst[((sz)y * nw + x) * 4 + c] = (char)((sum + 2) / 4);
            }
        }
    }
    return dst;
}

// Makes the array texture (a layer per probe, with its mip levels: roughness blurs through
// them). Called by the renderer.
fn void ReflectionProbes.upload(&self)
{
    if (self.uploaded || !self.count) return;
    if (self.image.id)
    {
        sg::destroy_view(self.view);
        sg::destroy_image(self.image);
    }
    int levels = 1;
    int lw = self.width;
    int lh = self.height;
    // sokol takes the levels one after the other, each with all its layers
    sz total = (sz)lw * lh * 4 * self.count;
    while ((lw > 1 || lh > 1) && levels < 12)
    {
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
    for (int l = 0; l < levels; l++)
    {
        for (int i = 0; i < self.count; i++)
        {
            all[at:level[i].len] = level[i][..];
            at += level[i].len;
        }
        int nw = math::max(w / 2, 1);
        int nh = math::max(h / 2, 1);
        for (int i = 0; i < self.count; i++)
        {
            char[] next = l + 1 < levels ? probe_downsample(level[i], w, h, nw, nh) : {};
            if (l > 0) free(level[i]);
            level[i] = next;
        }
        w = nw;
        h = nh;
    }
    self.image = sg::make_image(&&{
        .type = ARRAY,
        .width = self.width,
        .height = self.height,
        .num_slices = self.count,
        .num_mipmaps = levels,
        .pixel_format = self.color_space == SRGB_COLOR_SPACE ? SgPixelFormat.SRGB8A8 : SgPixelFormat.RGBA8,
        .data = { .ptr = all.ptr, .size = (usz)total },
        .label = "drie-reflection-probes",
    });
    self.view = sg::make_view(&&{ .texture = { .image = self.image } });
    if (!self.sampler.id)
    {
        self.sampler = sg::make_sampler(&&{
            .min_filter = LINEAR, .mag_filter = LINEAR, .mipmap_filter = LINEAR,
            .wrap_u = REPEAT, .wrap_v = CLAMP_TO_EDGE,
        });
    }
    self.uploaded = true;
}
