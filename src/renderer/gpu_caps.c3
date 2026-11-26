// GpuCaps: what the GPU can do beyond sokol's portable feature set, for choosing between a
// portable compute kernel (GLSL through sokol-shdc, every backend) and a fast one (runtime MSL on
// Metal: float atomics, simdgroup reductions, 64-bit atomic min/max).
//
// Filled once when the GPU is set up (Renderer setup) and printed then. The tier comes from
// the GPU family, never from the OS version (iOS 17 still runs on A12/A13). DRIE_COMPUTE=portable
// in the environment forces the portable tier.
module drie;
import std::io, std::os::env, sokol;

enum ComputeTier : char
{
    PORTABLE,    // shdc GLSL kernels only: 32-bit integer atomics, shared memory, no subgroups
    METAL_FAST,  // Metal Apple7+: atomic_float in device memory, simdgroup reductions (MSL 3.0+)
}

struct GpuCaps
{
    ComputeTier tier;
    SgBackend backend;
    String name;
    // Metal: the highest MTLGPUFamilyAppleN (0 elsewhere, or not an Apple GPU)
    int apple_family;
    // Metal: the newest Metal Shading Language version this OS compiles (0.0 elsewhere)
    int msl_major;
    int msl_minor;
    bool metal3;
    bool metal4;
    bool unified_memory;
    // atomic_float in device memory (Apple7+, MSL 3.0+)
    bool float_atomics;
    // atomic_float in threadgroup memory (also MSL 4.1)
    bool threadgroup_float_atomics;
    // simd_sum / simd_min / simd_max ... (Apple7+)
    bool simd_reductions;
    // simd_shuffle and quad shuffles (Apple6+)
    bool simd_permute;
    // 64-bit atomic min/max on ulong, no compare-and-swap (Apple9+, Apple8 on macOS)
    bool atomic64_minmax;
    // bytes of threadgroup (shared) memory a kernel may use
    uint max_threadgroup_memory;
    uint max_threads_per_threadgroup;
    // DRIE_COMPUTE=portable was set
    bool forced_portable;
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

// What the GPU can do: valid once the renderer has set up the GPU (zeroed and PORTABLE before).
fn GpuCaps* gpu_caps() => &gpu_caps_value;

fn bool GpuCaps.fast_compute(&self) => self.tier == METAL_FAST;

// (Renderer setup calls it.) Queries the GPU into gpu_caps(), and prints it.
fn void query_gpu_caps(bool print = true) {
    GpuCaps c;
    c.backend = sg::query_backend();
    c.tier = PORTABLE;
    // the portable tier's limits: what sokol-shdc's GLSL compute gives on every backend
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

// One line of what the GPU can do (printed once at startup).
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
