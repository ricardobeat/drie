// Irradiance volumes: light that has bounced around a room, stored in a grid of probes.
// (Not in three.js itself; its LightProbe is a single probe.)
//
//   IrradianceVolume* v = drie::irradiance_volume(room_min, room_max, 0.5f);
//   for every probe (x, y, z) and each of the six axis directions `dir` (+x -x +y -y +z -z):
//       v.set(x, y, z, dir, light arriving at v.probe_position(x, y, z) on a surface facing dir);
//   scene.irradiance = v;
//
// Lit materials inside the box then receive that light on top of the scene's lights, blended
// between the probes around them and between the directions by their normal. What goes into
// the probes is up to the application (see IrradianceVolume.sample for bounce after bounce).
// Values are irradiance: a white surface facing `dir` shows `value / PI`.
module drie;
import std::math;
import sokol;

// Directions of the six values of a probe.
const Vector3[6] IRRADIANCE_AXES = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };

struct IrradianceVolume {
    Vector3 min;
    Vector3 max;
    // probes along x, y and z; they sit in the middle of their cells
    int nx;
    int ny;
    int nz;
    // rgb per direction per probe
    float[] data;
    float intensity;
    // Surfaces look up the volume this far off along their normal (meters), so that a wall
    // takes its light from the room it faces and not from the probes behind it.
    float normal_offset;
    // Extension: the probes hold all the light from around them, the sky's included, instead of
    // what bounces on top of it. Inside the box they take the place of the scene's ambient and
    // hemisphere lights, which then light only what is outside it: a building's shade and its
    // dark corners, out of doors.
    bool replaces_ambient;
    SgImage image;
    SgView view;
    SgSampler sampler;
    bool uploaded;
}

// A volume over the box min..max with probes about `spacing` apart.
fn IrradianceVolume* irradiance_volume(Vector3 min, Vector3 max, float spacing = 0.5) {
    IrradianceVolume* v = mem::new(IrradianceVolume);
    Vector3 size        = max - min;
    *v = {
        .min = min,
        .max = max,
        .nx = math::max((int)math::round(size.x / spacing), 1),
        .ny = math::max((int)math::round(size.y / spacing), 1),
        .nz = math::max((int)math::round(size.z / spacing), 1),
        .intensity = 1,
        .normal_offset = spacing * 0.5f,
    };
    v.data = mem::new_array(float, (sz)v.nx * v.ny * v.nz * 18);
    return v;
}

fn Vector3 IrradianceVolume.cell(&self) {
    Vector3 size = self.max - self.min;
    return { size.x / self.nx, size.y / self.ny, size.z / self.nz };
}

fn Vector3 IrradianceVolume.probe_position(&self, int x, int y, int z) {
    Vector3 c = self.cell();
    return self.min + (Vector3) { ((float)x + 0.5f) * c.x, ((float)y + 0.5f) * c.y, ((float)z + 0.5f) * c.z };
}

fn sz IrradianceVolume.offset(&self, int x, int y, int z, int dir) {
    return ((((sz)z * self.ny + y) * self.nx + x) * 6 + dir) * 3;
}

fn void IrradianceVolume.set(&self, int x, int y, int z, int dir, Vector3 irradiance) {
    sz o = self.offset(x, y, z, dir);
    self.data[o] = irradiance.x;
    self.data[o + 1] = irradiance.y;
    self.data[o + 2] = irradiance.z;
    self.uploaded = false;
}

fn Vector3 IrradianceVolume.get(&self, int x, int y, int z, int dir) {
    sz o = self.offset(x, y, z, dir);
    return { self.data[o], self.data[o + 1], self.data[o + 2] };
}

// The light a surface at `pos` facing `normal` gets from the volume, as the shader sees it.
// Lets a bake feed one bounce into the next.
fn Vector3 IrradianceVolume.sample(&self, Vector3 pos, Vector3 normal) {
    Vector3 c  = self.cell();
    Vector3 p  = pos + normal * self.normal_offset - self.min;
    Vector3 g  = { p.x / c.x - 0.5f, p.y / c.y - 0.5f, p.z / c.z - 0.5f };
    float fade = self.edge_fade(p);
    if (fade <= 0) return { };
    int x0           = (int)math::floor(g.x);
    int y0           = (int)math::floor(g.y);
    int z0           = (int)math::floor(g.z);
    Vector3 f        = { g.x - (float)x0, g.y - (float)y0, g.z - (float)z0 };
    int[3] dirs      = { normal.x >= 0 ? 0 : 1, normal.y >= 0 ? 2 : 3, normal.z >= 0 ? 4 : 5 };
    float[3] weights = { normal.x * normal.x, normal.y * normal.y, normal.z * normal.z };
    Vector3 sum;
    for (int k = 0; k < 8; k++) {
        int x   = math::clamp(x0 + (k & 1), 0, self.nx - 1);
        int y   = math::clamp(y0 + ((k >> 1) & 1), 0, self.ny - 1);
        int z   = math::clamp(z0 + ((k >> 2) & 1), 0, self.nz - 1);
        float w = ((k & 1) ? f.x : 1 - f.x) * ((k & 2) ? f.y : 1 - f.y) * ((k & 4) ? f.z : 1 - f.z);
        for (int a = 0; a < 3; a++) sum += self.get(x, y, z, dirs[a]) * (w * weights[a]);
    }
    return sum * (fade * self.intensity);
}

// 1 inside the volume, falling to 0 over the outermost half cell. `p` is relative to min.
fn float IrradianceVolume.edge_fade(&self, Vector3 p) {
    Vector3 c    = self.cell();
    Vector3 size = self.max - self.min;
    float fade   = 1;
    for (int a = 0; a < 3; a++) {
        float edge = math::min(p[a], size[a] - p[a]) / c[a];
        fade = math::min(fade, clamp(edge * 2, 0, 1));
    }
    return fade;
}

fn ushort half_float(float f) @local {
    uint x    = bitcast(f, uint);
    uint sign = (x >> 16) & 0x8000;
    int e     = (int)((x >> 23) & 0xff) - 127 + 15;
    uint m    = x & 0x7fffff;
    if (e <= 0) return (ushort)sign;
    if (e >= 31) return (ushort)(sign | 0x7bff);
    return (ushort)(sign | ((uint)e << 10) | (m >> 13));
}

// Uploads the probes as a 3D texture: the six directions side by side along x.
fn void IrradianceVolume.upload(&self) {
    if (self.uploaded) return;
    if (self.image.id) {
        sg::destroy_view(self.view);
        sg::destroy_image(self.image);
    } else {
        self.sampler = sg::make_sampler(
            &&{
                .min_filter = LINEAR,
                .mag_filter = LINEAR,
                .wrap_u = CLAMP_TO_EDGE,
                .wrap_v = CLAMP_TO_EDGE,
                .wrap_w = CLAMP_TO_EDGE,
            }
        );
    }
    int width   = self.nx * 6;
    ushort[] px = mem::new_array(ushort, (sz)width * self.ny * self.nz * 4);
    defer free(px);
    for (int z = 0; z < self.nz; z++) {
        for (int y = 0; y < self.ny; y++) {
            for (int x = 0; x < self.nx; x++) {
                for (int d = 0; d < 6; d++) {
                    Vector3 e = self.get(x, y, z, d);
                    sz o      = (((sz)z * self.ny + y) * width + (sz)(d * self.nx + x)) * 4;
                    px[o] = half_float(math::max(e.x, 0.0f));
                    px[o + 1] = half_float(math::max(e.y, 0.0f));
                    px[o + 2] = half_float(math::max(e.z, 0.0f));
                    px[o + 3] = half_float(1);
                }
            }
        }
    }
    self.image = sg::make_image(
        &&{
            .type = TYPE_3D,
            .width = width,
            .height = self.ny,
            .num_slices = self.nz,
            .pixel_format = RGBA16F,
            .data = { .ptr = px.ptr, .size = (usz)px.len * ushort::size },
        }
    );
    self.view = sg::make_view(&&{ .texture = { .image = self.image } });
    self.uploaded = true;
}