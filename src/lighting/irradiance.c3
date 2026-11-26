// 3D grid of 6-axis diffuse irradiance probes (`IrradianceVolume`) for bounced indirect light.
// Values are linear irradiance in RGB along `+x`, `-x`, `+y`, `-y`, `+z`, `-z` (a white Lambertian
// surface facing `dir` reflects `value / PI`), and positions and `normal_offset` are in meters.
// Assign to `Scene.irradiance`; lit shaders blend the eight surrounding probes and weight axis
// faces by `normal^2`. `upload()` packs the six directional slabs along X into an `RGBA16F` 3D
// texture lazily on the GPU whenever `set()` marks the volume dirty.

module drie;
import std::math;
import sokol;

// Unit axis vectors corresponding to probe direction indices 0..5 (`+x`, `-x`, `+y`, `-y`, `+z`, `-z`).
const Vector3[6] IRRADIANCE_AXES = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };

// Axis-aligned 3D grid of 6-axis irradiance probes attached via `Scene.irradiance`.
//
//   IrradianceVolume* v = drie::irradiance_volume(room_min, room_max, 0.5f);
//   // For every probe (x, y, z) and direction dir in 0..5 (IRRADIANCE_AXES):
//   v.set(x, y, z, dir, irradiance_at_probe);
//   scene.irradiance = v;
//
// Lit materials inside `[min, max]` receive this light on top of scene lights unless
// `replaces_ambient` is set. Use `sample()` on the CPU to feed one bounce into the next.
struct IrradianceVolume {
    Vector3 min; // World-space minimum corner of the volume (m).
    Vector3 max; // World-space maximum corner of the volume (m).
    int nx; // Probe count along X; probes sit at cell centers.
    int ny; // Probe count along Y; probes sit at cell centers.
    int nz; // Probe count along Z; probes sit at cell centers.
    float[] data; // Linear RGB irradiance per direction per probe (`nx * ny * nz * 18` floats).
    float intensity; // Scalar multiplier applied to sampled irradiance.
    // Distance along the surface normal (m) applied before lookup so walls sample the room they
    // face rather than probes behind them.
    float normal_offset;
    // When true, probes hold total indirect light including sky; inside the box they replace
    // the scene's ambient and hemisphere lights instead of adding on top of them.
    bool replaces_ambient;
    SgImage image; // Uploaded `RGBA16F` 3D texture (`nx * 6` x `ny` x `nz`).
    SgView view; // Texture view bound by lit shaders.
    SgSampler sampler; // Trilinear clamp-to-edge sampler created on first upload.
    bool uploaded; // True when GPU texture matches `data`.
}

// ---- Construction & access

// Allocates a volume covering `min..max` with probes spaced approximately `spacing` meters apart.
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

// Allocates a volume covering `min..max` with an explicit `nx` x `ny` x `nz` probe resolution.
fn IrradianceVolume* irradiance_grid(Vector3 min, Vector3 max, int nx, int ny, int nz) {
    IrradianceVolume* v = mem::new(IrradianceVolume);
    *v = {
        .min = min,
        .max = max,
        .nx = math::max(nx, 1),
        .ny = math::max(ny, 1),
        .nz = math::max(nz, 1),
        .intensity = 1,
    };
    Vector3 c = v.cell();
    v.normal_offset = math::min(c.x, math::min(c.y, c.z)) * 0.5f;
    v.data = mem::new_array(float, (sz)v.nx * v.ny * v.nz * 18);
    return v;
}

// Returns the world-space dimensions of one grid cell in meters.
fn Vector3 IrradianceVolume.cell(&self) {
    Vector3 size = self.max - self.min;
    return { size.x / self.nx, size.y / self.ny, size.z / self.nz };
}

// Returns the world-space center of probe `(x, y, z)` in meters.
fn Vector3 IrradianceVolume.probe_position(&self, int x, int y, int z) {
    Vector3 c = self.cell();
    return self.min + (Vector3) { ((float)x + 0.5f) * c.x, ((float)y + 0.5f) * c.y, ((float)z + 0.5f) * c.z };
}

// Returns the starting float index in `data` for probe `(x, y, z)` and direction `dir` (0..5).
fn sz IrradianceVolume.offset(&self, int x, int y, int z, int dir) {
    return ((((sz)z * self.ny + y) * self.nx + x) * 6 + dir) * 3;
}

// Writes linear RGB `irradiance` for probe `(x, y, z)` along direction `dir` (0..5)
// and marks the GPU texture dirty.
fn void IrradianceVolume.set(&self, int x, int y, int z, int dir, Vector3 irradiance) {
    sz o = self.offset(x, y, z, dir);
    self.data[o] = irradiance.x;
    self.data[o + 1] = irradiance.y;
    self.data[o + 2] = irradiance.z;
    self.uploaded = false;
}

// Reads the stored linear RGB irradiance for probe `(x, y, z)` along direction `dir` (0..5).
fn Vector3 IrradianceVolume.get(&self, int x, int y, int z, int dir) {
    sz o = self.offset(x, y, z, dir);
    return { self.data[o], self.data[o + 1], self.data[o + 2] };
}

// ---- Sampling

// Evaluates trilinearly interpolated irradiance at `pos` for a surface facing `normal`, matching
// the GPU shader lookup so multi-bounce bakers can feed one bounce into the next.
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

// Returns 1 inside `[min, max]` and fades linearly to 0 over half a cell outside the box so a
// room-sized volume illuminates its walls, floor, and ceiling (`p` is relative to `min`).
fn float IrradianceVolume.edge_fade(&self, Vector3 p) {
    Vector3 c    = self.cell();
    Vector3 size = self.max - self.min;
    float fade   = 1;
    for (int a = 0; a < 3; a++) {
        float edge = math::min(p[a], size[a] - p[a]) / c[a];
        fade = math::min(fade, clamp(1 + edge * 2, 0, 1));
    }
    return fade;
}

// ---- GPU upload

fn ushort half_float(float f) @local {
    uint x    = bitcast(f, uint);
    uint sign = (x >> 16) & 0x8000;
    int e     = (int)((x >> 23) & 0xff) - 127 + 15;
    uint m    = x & 0x7fffff;
    if (e <= 0) return (ushort)sign;
    if (e >= 31) return (ushort)(sign | 0x7bff);
    return (ushort)(sign | ((uint)e << 10) | (m >> 13));
}

// Packs probe irradiance as half-float RGBA into `px` for a 3D texture of dimensions `width` x
// `height` x `nz`, starting at column `x0` with the six directional slabs laid out side by side
// along X (`nx` texels per direction).
fn void IrradianceVolume.fill(&self, ushort[] px, int width, int height, int x0) {
    for (int z = 0; z < self.nz; z++) {
        for (int y = 0; y < self.ny; y++) {
            for (int x = 0; x < self.nx; x++) {
                for (int d = 0; d < 6; d++) {
                    Vector3 e = self.get(x, y, z, d);
                    sz o      = (((sz)z * height + y) * width + (sz)(x0 + d * self.nx + x)) * 4;
                    px[o] = half_float(math::max(e.x, 0.0f));
                    px[o + 1] = half_float(math::max(e.y, 0.0f));
                    px[o + 2] = half_float(math::max(e.z, 0.0f));
                    px[o + 3] = half_float(1);
                }
            }
        }
    }
}

// Uploads `data` to the GPU as an `RGBA16F` 3D texture (`nx * 6` x `ny` x `nz`) when dirty,
// recreating any prior image and view. Called lazily by the renderer.
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
    self.fill(px, width, self.ny, 0);
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