// ComputePass.dispatch_indirect and the 4 KB uniform block: one kernel writes the group counts
// into an indirect buffer, the next is dispatched from it and counts its threads into a
// storage buffer, adding a value read from the far end of a 4096-byte uniform block. Headless;
// Metal only for now (its kernels are runtime MSL), elsewhere it says so and passes.
// `just test compute_indirect`
module compute_indirect_test;
import drie, sokol;
import std::io;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

const GROUPS = 5;
const THREADS = 64;
const ADD = 7;

const ZString ARGS_MSL = `
#include <metal_stdlib>
using namespace metal;
struct Args { uint x; uint y; uint z; uint pad; };
kernel void args_main(device Args* args [[buffer(8)]], device atomic_uint* counter [[buffer(9)]],
                      uint i [[thread_position_in_grid]]) {
    if (i == 0) {
        args[0].x = 5; args[0].y = 1; args[0].z = 1;
        atomic_store_explicit(counter, 0, memory_order_relaxed);
    }
}
`;

const ZString COUNT_MSL = `
#include <metal_stdlib>
using namespace metal;
struct Params { float4 v[256]; };
kernel void count_main(constant Params& u [[buffer(0)]], device atomic_uint* counter [[buffer(8)]],
                       uint i [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(counter, 1u + (uint)u.v[255].x, memory_order_relaxed);
}
`;

fn SgShaderDesc args_desc() {
    SgShaderDesc d = {
        .compute_func = { .source = ARGS_MSL, .entry = "args_main" },
        .mtl_threads_per_threadgroup = { 1, 1, 1 },
        .label = "test-args",
    };
    d.views[0].storage_buffer = { .stage = COMPUTE, .readonly = false, .msl_buffer_n = 8 };
    d.views[1].storage_buffer = { .stage = COMPUTE, .readonly = false, .msl_buffer_n = 9 };
    return d;
}

fn SgShaderDesc count_desc() {
    SgShaderDesc d = {
        .compute_func = { .source = COUNT_MSL, .entry = "count_main" },
        .mtl_threads_per_threadgroup = { THREADS, 1, 1 },
        .label = "test-count",
    };
    d.uniform_blocks[0] = { .stage = COMPUTE, .size = 4096, .msl_buffer_n = 0 };
    d.views[0].storage_buffer = { .stage = COMPUTE, .readonly = false, .msl_buffer_n = 8 };
    return d;
}

fn int main(String[] args) {
    Renderer* renderer = drie::renderer(antialias: false, width: 64, height: 64);
    renderer.init_headless();
    SgBackend b = sg::query_backend();
    if (b != METAL_MACOS && b != METAL_IOS && b != METAL_SIMULATOR) {
        io::printfn("skip: compute_indirect's kernels are MSL (this backend: %d)", b);
        return 0;
    }
    StorageBuffer* indirect = drie::storage_buffer(16, label: "test-indirect", indirect: true);
    StorageBuffer* counter = drie::storage_buffer(16, label: "test-counter");
    ComputePass* write_args = drie::compute_pass(args_desc(), "test-args");
    write_args.set_buffer(0, indirect);
    write_args.set_buffer(1, counter);
    ComputePass* count = drie::compute_pass(count_desc(), "test-count");
    count.set_buffer(0, counter);
    float[4][256] params;
    params[255][0] = ADD;
    count.set_uniforms(0, &params, float[4][256]::size);

    int slot = drie::readback_slot();
    drie::begin_compute("test-indirect");
    write_args.dispatch(1);
    count.dispatch_indirect(indirect, 0);
    counter.track(slot);
    drie::end_compute();
    drie::commit();

    uint[4] got;
    expect(drie::read_back(slot, &got, uint[4]::size, wait: true), "the counter is read back");
    uint want = GROUPS * THREADS * (1 + ADD);
    io::printfn("     counter %d, want %d (%d groups of %d threads, +%d from uniform byte 4080)", got[0], want, GROUPS, THREADS, ADD);
    expect(got[0] == want, "dispatch_indirect ran the groups the first kernel wrote, with the 4 KB uniforms");
    io::printfn(failures ? "%d failed" : "all passed", failures);
    return failures ? 1 : 0;
}
