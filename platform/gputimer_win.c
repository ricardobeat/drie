// Per-pass GPU timing on D3D11 for drie::gpu_timer: issues timestamp queries at each pass start
// and frame end inside a disjoint query per frame in flight, resolving without flushing a few
// frames later so each pass spans to the next pass start. Compiles empty outside Windows.
#if defined(_WIN32)
#define COBJMACROS
#include <d3d11.h>
#include <stdbool.h>
#include <stdint.h>

#define GT_SLOTS 4
#define GT_MAX_PASSES 128

static struct {
    bool ready;
    ID3D11Device* dev;
    ID3D11DeviceContext* ctx;
    ID3D11Query* disjoint[GT_SLOTS];
    ID3D11Query* ts[GT_SLOTS][GT_MAX_PASSES + 1];
    int passes[GT_SLOTS];
    bool open[GT_SLOTS];
    bool ended[GT_SLOTS];
} gt;

bool gt_init(const void* device_ptr, const void* ctx_ptr) {
    if (gt.ready) return true;
    gt.dev = (ID3D11Device*)device_ptr;
    gt.ctx = (ID3D11DeviceContext*)ctx_ptr;
    if (!gt.dev || !gt.ctx) return false;
    D3D11_QUERY_DESC dj = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
    D3D11_QUERY_DESC tq = { D3D11_QUERY_TIMESTAMP, 0 };
    for (int s = 0; s < GT_SLOTS; s++) {
        if (FAILED(ID3D11Device_CreateQuery(gt.dev, &dj, &gt.disjoint[s]))) return false;
        for (int i = 0; i <= GT_MAX_PASSES; i++) {
            if (FAILED(ID3D11Device_CreateQuery(gt.dev, &tq, &gt.ts[s][i]))) return false;
        }
    }
    gt.ready = true;
    return true;
}

void gt_pass_begin(int slot, int index) {
    if (!gt.ready || index >= GT_MAX_PASSES) return;
    if (!gt.open[slot]) {
        ID3D11DeviceContext_Begin(gt.ctx, (ID3D11Asynchronous*)gt.disjoint[slot]);
        gt.open[slot] = true;
        gt.ended[slot] = false;
    }
    ID3D11DeviceContext_End(gt.ctx, (ID3D11Asynchronous*)gt.ts[slot][index]);
    gt.passes[slot] = index + 1;
}

void gt_pass_end(int slot, int index) { (void)slot; (void)index; }

bool gt_frame_end(int slot) {
    if (!gt.ready || !gt.open[slot]) return false;
    ID3D11DeviceContext_End(gt.ctx, (ID3D11Asynchronous*)gt.ts[slot][gt.passes[slot]]);
    ID3D11DeviceContext_End(gt.ctx, (ID3D11Asynchronous*)gt.disjoint[slot]);
    gt.open[slot] = false;
    gt.ended[slot] = true;
    return true;
}

bool gt_resolve(int slot, int count, double* ms) {
    if (!gt.ready || !gt.ended[slot]) return false;
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj;
    if (ID3D11DeviceContext_GetData(gt.ctx, (ID3D11Asynchronous*)gt.disjoint[slot], &dj, sizeof(dj), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK) return false;
    if (count > gt.passes[slot]) count = gt.passes[slot];
    uint64_t t[GT_MAX_PASSES + 1];
    for (int i = 0; i <= count; i++) {
        if (ID3D11DeviceContext_GetData(gt.ctx, (ID3D11Asynchronous*)gt.ts[slot][i], &t[i], sizeof(uint64_t), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK) return false;
    }
    for (int i = 0; i < count; i++) {
        ms[i] = (dj.Disjoint || dj.Frequency == 0 || t[i + 1] < t[i]) ? -1.0 : (double)(t[i + 1] - t[i]) * 1000.0 / (double)dj.Frequency;
    }
    gt.ended[slot] = false;
    gt.passes[slot] = 0;
    return true;
}
#endif
