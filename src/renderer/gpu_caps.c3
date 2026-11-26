// GPU capability probing beyond sokol's portable feature set, backed by platform/gpucaps.c.
// Selects between portable sokol-shdc GLSL compute kernels and runtime MSL kernels on Apple7+
// Metal devices (device float atomics, simdgroup reductions, 64-bit atomic min/max). Populated
// once during setup_gfx(); the tier depends on GPU family rather than OS version, and setting
// DRIE_COMPUTE=portable in the environment forces PORTABLE.
module drie;
import std::io, std::os::env, sokol;

// Compute kernel tier selected at GPU initialization.
enum ComputeTier : char
{
    PORTABLE,    // sokol-shdc GLSL kernels only: 32-bit int atomics, shared memory, no subgroups.
    METAL_FAST,  // Metal Apple7+ with MSL 3.0+: device atomic_float and simdgroup reductions.
}

// Queried GPU backend features and compute limits, populated by query_gpu_caps().
struct GpuCaps
{
    ComputeTier tier;                   // Active compute tier (METAL_FAST or PORTABLE).
    SgBackend backend;                  // Active sokol graphics backend.
    String name;                        // Device name reported by the backend, if available.
    int apple_family;                   // Highest MTLGPUFamilyAppleN on Metal (0 elsewhere).
    int msl_major;                      // Maximum MSL major version supported by this OS (0 elsewhere).
    int msl_minor;                      // Maximum MSL minor version supported by this OS.
    bool metal3;                        // True when the device supports Metal 3.
    bool metal4;                        // True when the device supports Metal 4.
    bool unified_memory;                // True when CPU and GPU share unified memory.
    bool float_atomics;                 // Device-memory atomic_float support (Apple7+, MSL 3.0+).
    bool threadgroup_float_atomics;     // Threadgroup-memory atomic_float support (MSL 4.1+).
    bool simd_reductions;               // simd_sum / simd_min / simd_max support (Apple7+).
    bool simd_permute;                  // simd_shuffle and quad-shuffle support (Apple6+).
    bool atomic64_minmax;               // 64-bit atomic min/max without CAS (Apple9+, Apple8 macOS).
    uint max_threadgroup_memory;        // Maximum threadgroup (shared) memory per kernel, in bytes.
    uint max_threads_per_threadgroup;   // Maximum threads per threadgroup.
    bool forced_portable;               // True when DRIE_COMPUTE=portable overrode the tier.
}

struct GpucapsMetal @local
{
    CInt apple_family;
    CInt msl_major;
    CInt msl_minor;
    bool metal3;
    bool metal4;
    bool unified_memory;
    bool macos;
    CUInt max_threadgroup_memory;
    CUInt max_threads_per_threadgroup;
    char[64] name;
}

extern fn bool gpucaps_query_metal(void* device, GpucapsMetal* out) @cname("gpucaps_query_metal") @local;

GpuCaps gpu_caps_value @local;
char[64] gpu_caps_name @local;

// Returns the queried GPU capabilities; valid after setup_gfx() (zeroed and PORTABLE before).
fn GpuCaps* gpu_caps() => &gpu_caps_value;

// True when the GPU runs the Metal fast compute tier.
fn bool GpuCaps.fast_compute(&self) => self.tier == METAL_FAST;

// Queries GPU capabilities into gpu_caps() and optionally logs a one-line summary. Called once
// during setup_gfx().
fn void query_gpu_caps(bool print = true) {
    GpuCaps c;
    c.backend = sg::query_backend();
    c.tier = PORTABLE;
    // Portable baseline limits guaranteed by sokol-shdc GLSL compute on all supported backends.
    c.max_threadgroup_memory = 32 * 1024;
    c.max_threads_per_threadgroup = 1024;
    if (c.backend == METAL_MACOS || c.backend == METAL_IOS || c.backend == METAL_SIMULATOR) {
        GpucapsMetal m;
        if (gpucaps_query_metal(sg::mtl_device(), &m)) {
            gpu_caps_name = m.name;
            c.name = ((ZString)&gpu_caps_name).str_view();
            c.apple_family = m.apple_family;
            c.msl_major = m.msl_major;
            c.msl_minor = m.msl_minor;
            c.metal3 = m.metal3;
            c.metal4 = m.metal4;
            c.unified_memory = m.unified_memory;
            c.max_threadgroup_memory = m.max_threadgroup_memory;
            c.max_threads_per_threadgroup = m.max_threads_per_threadgroup;
            int msl = m.msl_major * 100 + m.msl_minor;
            c.float_atomics = m.apple_family >= 7 && msl >= 300;
            c.threadgroup_float_atomics = c.float_atomics && msl >= 401;
            c.simd_reductions = m.apple_family >= 7;
            c.simd_permute = m.apple_family >= 6;
            c.atomic64_minmax = msl >= 300 && (m.apple_family >= 9 || (m.macos && m.apple_family >= 8));
            if (c.float_atomics && c.simd_reductions) c.tier = METAL_FAST;
        }
    }
    if (try mode = env::tget_var("DRIE_COMPUTE")) {
        if (mode == "portable") {
            c.forced_portable = true;
            c.tier = PORTABLE;
        }
    }
    gpu_caps_value = c;
    if (print) c.print();
}

fn String backend_name(SgBackend b) @local {
    switch (b) {
        case GLCORE: return "GL core";
        case GLES3: return "GLES3";
        case D3D11: return "D3D11";
        case METAL_IOS: return "Metal (iOS)";
        case METAL_MACOS: return "Metal (macOS)";
        case METAL_SIMULATOR: return "Metal (simulator)";
        case WGPU: return "WebGPU";
        case VULKAN: return "Vulkan";
        default: return "dummy";
    }
}

// Prints a single-line summary of the GPU backend, features, shared memory, and compute tier.
fn void GpuCaps.print(&self) {
    io::printf("drie: gpu %s, %s", self.name.len ? self.name : "(unnamed)", backend_name(self.backend));
    if (self.apple_family) {
        io::printf(", Apple%d, MSL %d.%d%s%s", self.apple_family, self.msl_major, self.msl_minor, self.metal3 ? ", Metal3" : "", self.metal4 ? ", Metal4" : "");
        io::printf(", float atomics %s (threadgroup %s), simd reductions %s, simd permute %s, 64-bit min/max %s",
            self.float_atomics ? "yes" : "no", self.threadgroup_float_atomics ? "yes" : "no", self.simd_reductions ? "yes" : "no",
            self.simd_permute ? "yes" : "no", self.atomic64_minmax ? "yes" : "no");
    }
    io::printfn(", %d B shared, compute tier %s%s", self.max_threadgroup_memory, self.tier, self.forced_portable ? " (DRIE_COMPUTE=portable)" : "");
}
