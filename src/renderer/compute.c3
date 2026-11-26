// Compute shader dispatch, storage buffers, and asynchronous GPU buffer readback.
// Wraps sokol compute passes and storage buffers for internal kernels (light clustering via
// cluster_lights_gpu, instance culling via cull_dispatch, and kernels in shaders/kernels.glsl)
// as well as application kernels. Uniform slots 0..3 hold up to 4096 bytes each; dispatches run
// between begin_compute() and end_compute(), or standalone via Renderer.compute(). Async readback
// uses up to 64 slots backed by drie::readback.
module drie;
import std::math, std::io;
import sokol, drie::readback;

// GPU storage buffer readable and writable by compute shaders and readable by vertex shaders.
// With `cpu_write`, the CPU updates it at most once per frame via transient writes and the GPU
// only reads it; otherwise the GPU writes it and it is zero-initialized on creation.
struct StorageBuffer {
    SgBuffer buffer; // Underlying sokol buffer handle.
    SgView view; // Storage-buffer view bound to compute or vertex slots.
    usz size; // Allocated capacity in bytes (16-byte aligned).
    bool cpu_write; // Transient CPU-written buffer, read-only on the GPU.
    bool vertex; // Also usable as a vertex buffer (e.g. instanced attributes).
    bool indirect; // Also usable as an indirect dispatch argument buffer.
    ZString label; // Debug label passed to sokol.
}

// Allocates a StorageBuffer with at least `size` bytes (rounded up to 16 bytes).
fn StorageBuffer* storage_buffer(
    sz size,
    bool cpu_write = false,
    bool vertex = false,
    ZString label = "drie-storage",
    bool indirect = false
) {
    StorageBuffer* b = mem::new(StorageBuffer);
    b.cpu_write = cpu_write;
    b.vertex = vertex;
    b.indirect = indirect;
    b.label = label;
    b.ensure(size);
    return b;
}

// Ensures capacity for at least `bytes` (16-byte aligned). Reallocating discards existing contents
// (GPU-written buffers are zeroed).
fn void StorageBuffer.ensure(&self, sz bytes) {
    usz size = math::max(((usz)bytes + 15) & ~(usz)15, (usz)16);
    if (size <= self.size && self.buffer.id) return;
    usz cap = math::max(self.size, (usz)256);
    while (cap < size) cap *= 2;
    self.create(cap, null);
}

// Uploads `bytes` from `data` into the buffer. Transient `cpu_write` buffers grow if needed and
// update in place; GPU-written buffers are recreated with the initial data.
fn void StorageBuffer.write(&self, void* data, sz bytes) {
    if (!self.cpu_write) {
        self.create(math::max(((usz)bytes + 15) & ~(usz)15, (usz)16), data, (usz)bytes);
        return;
    }
    self.ensure(bytes);
    if (!bytes) return;
    sg::write_buffer_transient(&&{ .src.data = { .ptr = data, .size = (usz)bytes }, .dst.buffer = self.buffer });
}

// Destroys any existing GPU buffer and creates a new one of `size` bytes, copying `bytes` from
// `data` into GPU-written buffers (zero-filling the rest).
fn void StorageBuffer.create(&self, usz size, void* data, usz bytes = 0) {
    self.dispose();
    self.size = size;
    SgBufferDesc desc = {
        .usage = {
            .storage_buffer = true,
            .vertex_buffer = self.vertex,
            .indirect_buffer = self.indirect,
            .write_transient = self.cpu_write,
        },
        .size = size,
        .label = self.label,
    };
    char[] init;
    if (!self.cpu_write) {
        // GPU-written buffers cannot use CPU updates; provide zeroed initial contents at creation.
        init = mem::new_array(char, (sz)size);
        if (data) mem::copy(init.ptr, data, (sz)bytes);
        desc.data = { .ptr = init.ptr, .size = size };
    }
    self.buffer = sg::make_buffer(&desc);
    self.view = sg::make_view(&&{ .storage_buffer = { .buffer = self.buffer } });
    if (init.ptr) mem::free(init.ptr);
}

// Releases the underlying sokol view and buffer handles.
fn void StorageBuffer.dispose(&self) {
    if (self.view.id) sg::destroy_view(self.view);
    if (self.buffer.id) sg::destroy_buffer(self.buffer);
    self.view = { };
    self.buffer = { };
    self.size = 0;
}

// Schedules a GPU-to-CPU readback of this buffer into `slot` (from readback_slot()). Call inside
// the compute pass after the dispatches that write the buffer, then poll with read_back().
fn bool StorageBuffer.track(&self, int slot) {
    if (!self.buffer.id || slot < 0) return false;
    return readback::track(self.buffer, slot);
}

int readback_slots @local;

// Allocates a readback slot index for StorageBuffer.track / read_back (-1 when all 64 are taken).
fn int readback_slot() {
    if (readback_slots >= 64) return -1;
    return readback_slots++;
}

// Copies up to `bytes` from a tracked buffer into `out` when the GPU has finished that pass.
// With `wait`, blocks until ready (the frame must already be committed); otherwise returns false
// until data arrives a frame or two later.
fn bool read_back(int slot, void* out, sz bytes, bool wait = false) {
    if (slot < 0 || !bytes) return false;
    return readback::fetch(slot, out, (CInt)bytes, wait);
}

// Compute shader pipeline paired with cached view bindings and uniform blocks. Bind resources
// with set_buffer/set_view/set_uniforms, then run via Renderer.compute() or dispatch() inside a
// begin_compute/end_compute block.
struct ComputePass {
    SgShader shader; // Compiled compute shader.
    SgPipeline pipeline; // Compute pipeline created from `shader`.
    SgBindings bindings; // Storage-buffer and texture views applied on dispatch.
    char[4096][4] uniforms; // Cached uniform blocks (slots 0..3), applied on dispatch.
    usz[4] uniform_size; // Active byte size per uniform slot (0 skips that slot).
}

// Creates a ComputePass from a sokol-shdc compute shader descriptor.
fn ComputePass* compute_pass(SgShaderDesc desc, ZString label = "drie-compute") {
    ComputePass* p = mem::new(ComputePass);
    p.shader = make_program(desc);
    p.pipeline = sg::make_pipeline(&&{ .compute = true, .shader = p.shader, .label = label });
    return p;
}

// Binds a StorageBuffer view to view slot `slot`.
fn void ComputePass.set_buffer(&self, int slot, StorageBuffer* b) {
    self.bindings.views[slot] = b.view;
}

// Binds an arbitrary sokol view (texture or storage buffer) to view slot `slot`.
fn void ComputePass.set_view(&self, int slot, SgView view) {
    self.bindings.views[slot] = view;
}

// Copies `size` bytes from `data` into uniform slot `slot` (slots 0..3, up to 4096 bytes).
fn void ComputePass.set_uniforms(&self, int slot, void* data, usz size) {
    if (slot < 0 || slot >= 4 || size > 4096) abort("drie: compute uniforms: slots 0..3, up to 4096 bytes");
    mem::copy(&self.uniforms[slot], data, (sz)size);
    self.uniform_size[slot] = size;
}

// Applies this pass's pipeline, view bindings, and non-empty uniform blocks to the active pass.
fn void ComputePass.apply(&self) {
    sg::apply_pipeline(self.pipeline);
    sg::apply_bindings(&self.bindings);
    for (int i = 0; i < 4; i++) {
        if (self.uniform_size[i]) sg::apply_uniforms(i, &&{ .ptr = &self.uniforms[i], .size = self.uniform_size[i] });
    }
}

// Dispatches `x * y * z` threadgroups. Must be called between begin_compute() and end_compute().
fn void ComputePass.dispatch(&self, int x, int y = 1, int z = 1) {
    if (x <= 0 || y <= 0 || z <= 0) return;
    self.apply();
    sg::dispatch(x, y, z);
}

// Dispatches threadgroups using three uint counts read from `buf` at byte `offset`. Must be
// called between begin_compute() and end_compute().
fn void ComputePass.dispatch_indirect(&self, StorageBuffer* buf, int offset = 0) {
    if (!buf || !buf.buffer.id) return;
    self.apply();
    sg::dispatch_indirect(buf.buffer, offset);
}

// True when the active sokol backend supports compute shaders.
fn bool compute_supported() => sg::isvalid() && sg::query_features().compute;

// Opens a profiled sokol compute pass for one or more ComputePass dispatches.
fn void begin_compute(ZString label = "drie-compute") {
    profile::begin_pass(&&{ .compute = true, .label = label });
}

// Closes the active compute pass opened by begin_compute().
fn void end_compute() {
    sg::end_pass();
}

// Dispatches a single ComputePass in its own compute pass. Call outside Renderer.render().
fn void Renderer.compute(&self, ComputePass* pass, int x, int y = 1, int z = 1) {
    begin_compute();
    pass.dispatch(x, y, z);
    end_compute();
}