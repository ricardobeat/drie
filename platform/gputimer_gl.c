// Per-pass GPU timing on desktop OpenGL for drie::gpu_timer: issues GL_TIMESTAMP queries
// (glQueryCounter, GL 3.3) at each pass start and frame end, resolving elapsed ms a few frames
// later so each pass spans to the next pass start. Compiles empty outside desktop Linux.
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
