// GPU time per pass on desktop GL (gpu_timer.c3): a GL_TIMESTAMP query (glQueryCounter, GL 3.3)
// at the start of each pass and one at the end of the frame, per frame in flight; read back
// once available, a few frames later. A pass's time runs to the next pass's start. (Built on
// every platform from the same list of C files: empty where sokol doesn't use GL, i.e. not Linux.)
#if defined(__linux__) && !defined(__ANDROID__)
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <stdbool.h>
#include <stdint.h>

#define GT_SLOTS 4
#define GT_MAX_PASSES 128

static struct {
    bool ready;
    GLuint q[GT_SLOTS][GT_MAX_PASSES + 1];
    int passes[GT_SLOTS];
    bool ended[GT_SLOTS];
} gt;

bool gt_init(const void* a, const void* b) {
    (void)a; (void)b;
    if (gt.ready) return true;
    for (int s = 0; s < GT_SLOTS; s++) glGenQueries(GT_MAX_PASSES + 1, gt.q[s]);
    gt.ready = glGetError() == GL_NO_ERROR;
    return gt.ready;
}

void gt_pass_begin(int slot, int index) {
    if (!gt.ready || index >= GT_MAX_PASSES) return;
    glQueryCounter(gt.q[slot][index], GL_TIMESTAMP);
    gt.passes[slot] = index + 1;
}

void gt_pass_end(int slot, int index) { (void)slot; (void)index; }

bool gt_frame_end(int slot) {
    if (!gt.ready || gt.passes[slot] == 0) return false;
    glQueryCounter(gt.q[slot][gt.passes[slot]], GL_TIMESTAMP);
    gt.ended[slot] = true;
    return true;
}

bool gt_resolve(int slot, int count, double* ms) {
    if (!gt.ready || !gt.ended[slot]) return false;
    if (count > gt.passes[slot]) count = gt.passes[slot];
    GLint avail = 0;
    glGetQueryObjectiv(gt.q[slot][count], GL_QUERY_RESULT_AVAILABLE, &avail);
    if (!avail) return false;
    GLuint64 t[GT_MAX_PASSES + 1];
    for (int i = 0; i <= count; i++) glGetQueryObjectui64v(gt.q[slot][i], GL_QUERY_RESULT, &t[i]);
    for (int i = 0; i < count; i++) ms[i] = t[i + 1] < t[i] ? -1.0 : (double)(t[i + 1] - t[i]) * 1e-6;
    gt.ended[slot] = false;
    gt.passes[slot] = 0;
    return true;
}
#endif
