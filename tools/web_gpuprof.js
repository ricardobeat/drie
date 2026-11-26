// GPU time per render/compute pass, from WebGPU timestamp queries (a page whose #fragment has
// "gpuprof"). Every pass gets timestamps; each queue.submit resolves them; window.gpuProf()
// gives the mean milliseconds per pass label over the frames since the last call.
(() => {
  if (!location.hash.includes('gpuprof') || !navigator.gpu) return;
  const totals = new Map();
  let frames = 0;
  const N = 512;
  const reqAdapter = GPU.prototype.requestAdapter;
  GPU.prototype.requestAdapter = async function (o) {
    const a = await reqAdapter.call(this, o);
    if (!a) return a;
    const reqDevice = a.requestDevice;
    a.requestDevice = async function (d = {}) {
      d = { ...d, requiredFeatures: [...(d.requiredFeatures || []), 'timestamp-query'] };
      const dev = await reqDevice.call(this, d);
      const qs = dev.createQuerySet({ type: 'timestamp', count: N });
      let used = 0, labels = [];
      const wrap = (proto, name) => {
        const f = proto[name];
        proto[name] = function (desc = {}) {
          if (used + 2 <= N) {
            desc = { ...desc, timestampWrites: { querySet: qs, beginningOfPassWriteIndex: used, endOfPassWriteIndex: used + 1 } };
            labels.push(desc.label || name);
            used += 2;
          }
          return f.call(this, desc);
        };
      };
      wrap(GPUCommandEncoder.prototype, 'beginRenderPass');
      wrap(GPUCommandEncoder.prototype, 'beginComputePass');
      const submit = GPUQueue.prototype.submit;
      let busy = false;
      GPUQueue.prototype.submit = function (bufs) {
        if (!used || busy) { used = 0; labels = []; return submit.call(this, bufs); }
        const n = used, ls = labels;
        used = 0; labels = [];
        const res = dev.createBuffer({ size: n * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        const rd = dev.createBuffer({ size: n * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = dev.createCommandEncoder();
        enc.resolveQuerySet(qs, 0, n, res, 0);
        enc.copyBufferToBuffer(res, 0, rd, 0, n * 8);
        submit.call(this, [...bufs, enc.finish()]);
        busy = true;
        rd.mapAsync(GPUMapMode.READ).then(() => {
          const t = new BigUint64Array(rd.getMappedRange());
          for (let i = 0; i < ls.length; i++) {
            const ms = Number(t[2 * i + 1] - t[2 * i]) / 1e6;
            totals.set(ls[i], (totals.get(ls[i]) || 0) + ms);
          }
          frames++;
          rd.unmap(); rd.destroy(); res.destroy();
          busy = false;
        });
      };
      return dev;
    };
    return a;
  };
  window.gpuProf = () => {
    const out = [...totals].map(([k, v]) => [k, +(v / frames).toFixed(2)]).sort((a, b) => b[1] - a[1]);
    const sum = out.reduce((s, [, v]) => s + v, 0);
    totals.clear(); const f = frames; frames = 0;
    return { frames: f, total_ms: +sum.toFixed(2), passes: out };
  };
})();
