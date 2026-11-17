// Extension: compute shaders and storage buffers, a thin layer over sokol's compute passes in
// the spirit of three.js's WebGPU renderer (`renderer.compute(computeNode)`, storage buffer
// attributes). The renderer uses it for light clustering (clusters.c3) and for culling
// instanced meshes (culling.c3); an application can run its own kernels the same way.
//
//     StorageBuffer* particles = drie::storage_buffer(count * Particle::size);
//     ComputePass* step = drie::compute_pass(my_step_shader_desc(sg::query_backend()));
//     step.set_buffer(VIEW_PARTICLES, particles);
//     step.set_uniforms(UB_STEP_PARAMS, &params, StepParams::size);
//     renderer.compute(step, (count + 63) / 64);
//
// Kernels are written in GLSL with sokol-shdc's `@cs` (see kernels.glsl) and need storage
// buffers, which every backend the engine targets has (Metal, D3D11, GL 4.3).
module drie;
import std::math, std::io;
import sokol, readback;

// A buffer kernels read and write (`buffer` in GLSL), and that vertex shaders can read.
// `cpu_write`: written from the CPU with write(), at most once per frame, and only read by the
// GPU; otherwise the GPU writes it and it starts out zeroed.
struct StorageBuffer
{
    SgBuffer buffer;
    SgView view;
    usz size;
    bool cpu_write;
    // also bound as a vertex buffer (instance attributes)
    bool vertex;
    ZString label;
}

fn StorageBuffer* storage_buffer(sz size, bool cpu_write = false, bool vertex = false, ZString label = "drie-storage")
{
    StorageBuffer* b = mem::new(StorageBuffer);
    b.cpu_write = cpu_write;
    b.vertex = vertex;
    b.label = label;
    b.ensure(size);
    return b;
}

// Makes room for `size` bytes (rounded up to a multiple of 16). A buffer that grows starts
// over: its contents are lost (zeroed for the GPU's buffers).
fn void StorageBuffer.ensure(&self, sz bytes)
{
    usz size = math::max(((usz)bytes + 15) & ~(usz)15, (usz)16);
    if (size <= self.size && self.buffer.id) return;
    usz cap = math::max(self.size, (usz)256);
    while (cap < size) cap *= 2;
    self.create(cap, null);
}

// Replaces the contents with `bytes` of `data`, for a buffer the CPU writes (growing it if
// needed); any other buffer is made again with the data in it (for data that rarely changes).
fn void StorageBuffer.write(&self, void* data, sz bytes)
{
    if (!self.cpu_write)
    {
        self.create(math::max(((usz)bytes + 15) & ~(usz)15, (usz)16), data, (usz)bytes);
        return;
    }
    self.ensure(bytes);
    if (!bytes) return;
    sg::write_buffer_transient(&&{ .src.data = { .ptr = data, .size = (usz)bytes }, .dst.buffer = self.buffer });
}

fn void StorageBuffer.create(&self, usz size, void* data, usz bytes = 0)
{
    self.dispose();
    self.size = size;
    SgBufferDesc desc = {
        .usage = { .storage_buffer = true, .vertex_buffer = self.vertex, .write_transient = self.cpu_write },
        .size = size,
        .label = self.label,
    };
    char[] init;
    if (!self.cpu_write)
    {
        // the GPU's buffers can't be written from the CPU: they start out with their data (zeros)
        init = mem::new_array(char, (sz)size);
        if (data) mem::copy(init.ptr, data, (sz)bytes);
        desc.data = { .ptr = init.ptr, .size = size };
    }
    self.buffer = sg::make_buffer(&desc);
    self.view = sg::make_view(&&{ .storage_buffer = { .buffer = self.buffer } });
    if (init.ptr) mem::free(init.ptr);
}

fn void StorageBuffer.dispose(&self)
{
    if (self.view.id) sg::destroy_view(self.view);
    if (self.buffer.id) sg::destroy_buffer(self.buffer);
    self.view = {};
    self.buffer = {};
    self.size = 0;
}

// Asks for the buffer's contents as they will be once the GPU has run what was encoded so far.
// Call it inside the compute pass that writes the buffer, after its dispatches, and collect
// them with read_back(slot, ...) in a later frame. `slot` comes from readback_slot().
fn bool StorageBuffer.track(&self, int slot)
{
    if (!self.buffer.id || slot < 0) return false;
    return readback::track(self.buffer, slot);
}

int readback_slots @local;

// A slot for StorageBuffer.track / read_back (-1: none left; there are 64).
fn int readback_slot()
{
    if (readback_slots >= 64) return -1;
    return readback_slots++;
}

// Copies a tracked buffer's contents (as many as `bytes`) into `out`, if the GPU has got that
// far; with `wait`, waits for it (the frame must have been committed). A frame or two late,
// without waiting, as readbacks are on the GPU.
fn bool read_back(int slot, void* out, sz bytes, bool wait = false)
{
    if (slot < 0 || !bytes) return false;
    return readback::fetch(slot, out, (CInt)bytes, wait);
}

// A compute kernel with its bindings. Fill them with set_buffer and set_uniforms, then
// dispatch it with Renderer.compute, or with dispatch() between begin_compute and end_compute
// to run several in one pass.
struct ComputePass
{
    SgShader shader;
    SgPipeline pipeline;
    SgBindings bindings;
    // uniform blocks, copied at set_uniforms and applied at each dispatch
    char[256][4] uniforms;
    usz[4] uniform_size;
}

fn ComputePass* compute_pass(SgShaderDesc desc, ZString label = "drie-compute")
{
    ComputePass* p = mem::new(ComputePass);
    p.shader = sg::make_shader(&desc);
    p.pipeline = sg::make_pipeline(&&{ .compute = true, .shader = p.shader, .label = label });
    return p;
}

fn void ComputePass.set_buffer(&self, int slot, StorageBuffer* b)
{
    self.bindings.views[slot] = b.view;
}

fn void ComputePass.set_view(&self, int slot, SgView view)
{
    self.bindings.views[slot] = view;
}

fn void ComputePass.set_uniforms(&self, int slot, void* data, usz size)
{
    if (slot < 0 || slot >= 4 || size > 256) abort("drie: compute uniforms: slots 0..3, up to 256 bytes");
    mem::copy(&self.uniforms[slot], data, (sz)size);
    self.uniform_size[slot] = size;
}

// Runs the kernel over x * y * z groups. Only between begin_compute and end_compute.
fn void ComputePass.dispatch(&self, int x, int y = 1, int z = 1)
{
    if (x <= 0 || y <= 0 || z <= 0) return;
    sg::apply_pipeline(self.pipeline);
    sg::apply_bindings(&self.bindings);
    for (int i = 0; i < 4; i++)
    {
        if (self.uniform_size[i]) sg::apply_uniforms(i, &&{ .ptr = &self.uniforms[i], .size = self.uniform_size[i] });
    }
    sg::dispatch(x, y, z);
}

// Whether the GPU runs compute shaders (always, on the backends the engine targets).
fn bool compute_supported() => sg::isvalid() && sg::query_features().compute;

fn void begin_compute(ZString label = "drie-compute")
{
    sg::begin_pass(&&{ .compute = true, .label = label });
}

fn void end_compute()
{
    sg::end_pass();
}

// Runs one kernel in a pass of its own, like three's renderer.compute(). Outside of render().
fn void Renderer.compute(&self, ComputePass* pass, int x, int y = 1, int z = 1)
{
    begin_compute();
    pass.dispatch(x, y, z);
    end_compute();
}
