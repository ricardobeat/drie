// Optional top-down Radiance Cascades debug visualizer module (`import drie::rc_debug;`).
// Slices the scene below `cut_y` with a top-down orthographic camera, async-reads the live
// GPU cascade and irradiance buffers, and draws probe nodes, angular ray roses, cascade
// interval rings, optional solid voxels, and a compact sokol_debugtext HUD.
module drie::rc_debug;
import drie, drie::ui, drie::readback, sokol;
import std::math;

bool active;

const int IRR_SLOT = 10;
const int RAD_SLOT = 11;

struct State @local {
    bool initialized;
    OrthographicCamera* cam;
    float pan_x;
    float pan_z;
    float half_h;
    float cut_y;
    int level;          // 0..levels-1, or -1 for ALL
    int slice_y;        // cascade-0 probe Y index (-1 = auto from cut_y)
    bool show_angles;
    bool show_voxels;
    bool[512] keys;

    // Async GPU readback buffers (1-frame-late, zero stall)
    RadianceCascades* last_rc;
    drie::Vec4[] irr_cpu;
    uint[] rad_cpu;     // 2 uints (uvec2 fp16) per cascade ray
    bool irr_ready;
    bool rad_ready;
    bool tracked;

    // sgl + sdtx contexts (window + offscreen RGBA8 capture)
    SglContext win_gl;
    SglPipeline win_pip;
    SdtxContext win_text;
    SglContext img_gl;
    SglPipeline img_pip;
    SdtxContext img_text;
}

State st @local = {
    .half_h = 6.5f,
    .level = 0,
    .slice_y = -1,
    .show_angles = true,
    .show_voxels = false,
};

fn void reset(Vector3 focus, float cut_y) {
    st.pan_x = focus.x;
    st.pan_z = focus.z;
    st.half_h = 6.5f;
    st.cut_y = cut_y;
    st.slice_y = -1;
}

fn void toggle(Vector3 focus = {}, float cut_y = 2.4f) {
    active = !active;
    if (active) {
        reset(focus, cut_y);
        st.keys = {};
        if (sapp::isvalid()) sapp::lock_mouse(false);
    }
}

fn int active_slice(RadianceCascades* rc) @local {
    if (!rc) return 0;
    if (st.slice_y >= 0) return math::clamp(st.slice_y, 0, rc.ny - 1);
    float target_y = st.cut_y - 0.8f;
    int iy = (int)math::floor((target_y - rc.min.y) / rc.spacing);
    return math::clamp(iy, 0, rc.ny - 1);
}

// Handles RC debug hotkeys while active (and `toggle_key` to enter/exit). Returns true if consumed.
fn bool handle_event(Event* e, Vector3 focus = {}, float cut_y = 2.4f, KeyCode toggle_key = F3) {
    if (e.type == KEY_DOWN && !e.key_repeat && e.key_code == toggle_key) {
        toggle(focus, cut_y);
        return true;
    }
    if (!active) return false;
    switch (e.type) {
        case KEY_DOWN:
            if ((int)e.key_code < 512) st.keys[(int)e.key_code] = true;
            if (!e.key_repeat) {
                switch (e.key_code) {
                    case NUM_0: st.level = 0;
                    case NUM_1: st.level = 1;
                    case NUM_2: st.level = 2;
                    case NUM_3: st.level = 3;
                    case NUM_4:
                    case TAB: st.level = -1;
                    case R: st.show_angles = !st.show_angles;
                    case B: st.show_voxels = !st.show_voxels;
                    case SPACE: reset(focus, cut_y);
                    case W:
                        if (st.last_rc) {
                            int cur = active_slice(st.last_rc);
                            st.slice_y = math::clamp(cur + 1, 0, st.last_rc.ny - 1);
                        }
                    case S:
                        if (st.last_rc) {
                            int cur = active_slice(st.last_rc);
                            st.slice_y = math::clamp(cur - 1, 0, st.last_rc.ny - 1);
                        }
                    default:
                        break;
                }
            }
            return true;
        case KEY_UP:
            if ((int)e.key_code < 512) st.keys[(int)e.key_code] = false;
            return true;
        case MOUSE_SCROLL:
            float factor = e.scroll_y > 0 ? 0.9f : 1.11f;
            st.half_h = drie::clamp(st.half_h * factor, 1.5f, 35.0f);
            return true;
        case UNFOCUSED:
            st.keys = {};
            return false;
        default:
            return false;
    }
}

// Updates pan/zoom/cut plane and returns the top-down OrthographicCamera (sliced at cut_y).
fn OrthographicCamera* update(float dt, int width, int height, Vector3 focus = {}, float cut_y = 2.4f) {
    if (!st.initialized) {
        st.initialized = true;
        st.cam = drie::orthographic_camera(-10, 10, 10, -10, 10, 45);
        reset(focus, cut_y);
    }
    float mult = (st.keys[(int)KeyCode.LEFT_SHIFT] || st.keys[(int)KeyCode.RIGHT_SHIFT]) ? 2.5f : 1.0f;
    float speed = st.half_h * 0.9f * mult * dt;
    if (st.keys[(int)KeyCode.LEFT]) st.pan_x -= speed;
    if (st.keys[(int)KeyCode.RIGHT]) st.pan_x += speed;
    if (st.keys[(int)KeyCode.UP]) st.pan_z -= speed;
    if (st.keys[(int)KeyCode.DOWN]) st.pan_z += speed;
    if (st.keys[(int)KeyCode.EQUAL] || st.keys[(int)KeyCode.KP_ADD]) {
        st.half_h = drie::clamp(st.half_h * (1.0f - 1.5f * dt), 1.5f, 35.0f);
    }
    if (st.keys[(int)KeyCode.MINUS] || st.keys[(int)KeyCode.KP_SUBTRACT]) {
        st.half_h = drie::clamp(st.half_h * (1.0f + 1.5f * dt), 1.5f, 35.0f);
    }
    if (st.keys[(int)KeyCode.LEFT_BRACKET]) st.cut_y -= 2.0f * dt;
    if (st.keys[(int)KeyCode.RIGHT_BRACKET]) st.cut_y += 2.0f * dt;

    float aspect = height > 0 ? (float)width / (float)height : 16.0f / 9.0f;
    st.cam.left = -st.half_h * aspect;
    st.cam.right = st.half_h * aspect;
    st.cam.top = st.half_h;
    st.cam.bottom = -st.half_h;
    st.cam.near = 10.0f;
    st.cam.far = 45.0f;
    st.cam.position = { st.pan_x, st.cut_y + 10.0f, st.pan_z };
    // local +Y -> -Z (north up), local -Z -> -Y (straight down)
    st.cam.rotation = { -drie::PI / 2, 0, 0 };
    st.cam.update_matrix_world();
    st.cam.update_projection_matrix();
    return st.cam;
}

// Queues async readback of `rc`'s GPU buffers after `rc.update(scene)`.
fn void track(RadianceCascades* rc, bool wait_now = false) {
    if (!active || !rc || !rc.irradiance_buf || !rc.cascade_buf) return;
    if (st.last_rc != rc) {
        if (st.irr_cpu.len) free(st.irr_cpu);
        if (st.rad_cpu.len) free(st.rad_cpu);
        sz irr_count = (sz)rc.nx * rc.ny * rc.nz * 6;
        sz rad_rays = (sz)(rc.offsets[rc.levels - 1] + rc.counts[rc.levels - 1]);
        st.irr_cpu = mem::new_array(drie::Vec4, irr_count);
        st.rad_cpu = mem::new_array(uint, rad_rays * 2);
        st.last_rc = rc;
        st.irr_ready = false;
        st.rad_ready = false;
        st.tracked = false;
    }
    if (st.tracked) {
        CInt irr_bytes = (CInt)(st.irr_cpu.len * drie::Vec4::size);
        CInt rad_bytes = (CInt)(st.rad_cpu.len * uint::size);
        if (readback::fetch(IRR_SLOT, st.irr_cpu.ptr, irr_bytes, wait_now)) st.irr_ready = true;
        if (readback::fetch(RAD_SLOT, st.rad_cpu.ptr, rad_bytes, wait_now)) st.rad_ready = true;
    }
    readback::track(rc.irradiance_buf.buffer, IRR_SLOT);
    readback::track(rc.cascade_buf.buffer, RAD_SLOT);
    st.tracked = true;
}

fn float half_to_float(ushort h) @local {
    uint sign = ((uint)h & 0x8000u) << 16;
    uint exp = ((uint)h >> 10) & 0x1fu;
    uint mant = (uint)h & 0x3ffu;
    uint bits;
    if (exp == 0) {
        if (mant == 0) return bitcast(sign, float);
        while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
        exp++;
        mant &= ~0x400u;
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    return bitcast(bits, float);
}

fn Vector3 unpack_rad(sz ray_idx) @local {
    if (!st.rad_ready || ray_idx * 2 + 1 >= st.rad_cpu.len) return { 0, 0, 0 };
    uint p0 = st.rad_cpu[ray_idx * 2];
    uint p1 = st.rad_cpu[ray_idx * 2 + 1];
    float r = half_to_float((ushort)(p0 & 0xffffu));
    float g = half_to_float((ushort)(p0 >> 16));
    float b = half_to_float((ushort)(p1 & 0xffffu));
    return { math::max(r, 0.0f), math::max(g, 0.0f), math::max(b, 0.0f) };
}

fn Vector3 oct_dir(float ex, float ey) @local {
    float nz = 1.0f - math::abs(ex) - math::abs(ey);
    float t = math::max(-nz, 0.0f);
    float nx = ex + (ex >= 0 ? -t : t);
    float ny = ey + (ey >= 0 ? -t : t);
    return ((Vector3){ nx, ny, nz }).normalize();
}

fn float[3] tone_rgb(Vector3 c) @local {
    float[3] out;
    for (int i = 0; i < 3; i++) {
        float v = math::max(c[i], 0.0f);
        v = v / (1.0f + v);
        out[i] = math::sqrt(drie::clamp(v, 0, 1));
    }
    return out;
}

fn float[2] world_to_px(float wx, float wz, int width, int height) @local {
    float aspect = height > 0 ? (float)width / (float)height : 16.0f / 9.0f;
    float hw = st.half_h * aspect;
    float hh = st.half_h;
    float sx = ((wx - st.pan_x) / hw * 0.5f + 0.5f) * (float)width;
    float sy = ((wz - st.pan_z) / hh * 0.5f + 0.5f) * (float)height;
    return { sx, sy };
}

fn void sgl_rect(float x, float y, float w, float h, float[4] col) @local {
    sgl::begin_quads();
    sgl::c4f(col[0], col[1], col[2], col[3]);
    sgl::v2f(x, y);
    sgl::v2f(x + w, y);
    sgl::v2f(x + w, y + h);
    sgl::v2f(x, y + h);
    sgl::end();
}

fn void sgl_box(float x0, float y0, float x1, float y1, float[4] col) @local {
    sgl::begin_line_strip();
    sgl::c4f(col[0], col[1], col[2], col[3]);
    sgl::v2f(x0, y0);
    sgl::v2f(x1, y0);
    sgl::v2f(x1, y1);
    sgl::v2f(x0, y1);
    sgl::v2f(x0, y0);
    sgl::end();
}

fn void sgl_circle(float cx, float cy, float r, int segs, float[4] col) @local {
    sgl::begin_line_strip();
    sgl::c4f(col[0], col[1], col[2], col[3]);
    for (int i = 0; i <= segs; i++) {
        float a = (float)i / (float)segs * 2.0f * drie::PI;
        sgl::v2f(cx + math::cos(a) * r, cy + math::sin(a) * r);
    }
    sgl::end();
}

const float[4][5] LEVEL_COLORS @local = {
    { 0.15f, 0.85f, 1.00f, 0.90f }, // L0: cyan
    { 0.25f, 1.00f, 0.45f, 0.90f }, // L1: green
    { 1.00f, 0.78f, 0.20f, 0.90f }, // L2: amber
    { 1.00f, 0.35f, 0.85f, 0.90f }, // L3: magenta
    { 0.85f, 0.85f, 0.95f, 0.90f }, // L4+: white
};

fn void draw_level_probes(RadianceCascades* rc, int lvl, int slice0, bool draw_angles, int width, int height) @local {
    int[<3>] dims = rc.level_dims(lvl);
    float spacing = rc.spacing * (float)(1 << lvl);
    int iy = math::clamp(slice0 >> lvl, 0, dims.y - 1);
    int d = rc.base_directions << lvl;
    int dd = d * d;
    int off = rc.offsets[lvl];
    float[4] lcol = LEVEL_COLORS[math::min(lvl, 4)];
    float px_per_m = (float)height / (2.0f * st.half_h);
    float node_r = drie::clamp(spacing * px_per_m * 0.15f, 2.0f, 6.5f);
    float probe_px = spacing * px_per_m;
    int angle_stride = probe_px < 16 ? 3 : (probe_px < 28 ? 2 : 1);

    // 1. Probe nodes
    sgl::begin_quads();
    for (int iz = 0; iz < dims.z; iz++) {
        float wz = rc.min.z + ((float)iz + 0.5f) * spacing;
        for (int ix = 0; ix < dims.x; ix++) {
            float wx = rc.min.x + ((float)ix + 0.5f) * spacing;
            float[2] sp = world_to_px(wx, wz, width, height);
            if (sp[0] < -16 || sp[0] > (float)(width + 16) || sp[1] < -16 || sp[1] > (float)(height + 16)) continue;

            int vx = (int)math::floor((wx - rc.min.x) / rc.voxel_size);
            int vy = (int)math::floor(((rc.min.y + ((float)iy + 0.5f) * spacing) - rc.min.y) / rc.voxel_size);
            int vz = (int)math::floor((wz - rc.min.z) / rc.voxel_size);
            bool solid = false;
            if (vx >= 0 && vx < rc.vx && vy >= 0 && vy < rc.vy && vz >= 0 && vz < rc.vz) {
                solid = (rc.voxels[(((sz)vz * rc.vy + vy) * rc.vx + vx) * 2] & 0x1000000u) != 0;
            }

            float[3] rgb = { lcol[0] * 0.35f, lcol[1] * 0.35f, lcol[2] * 0.35f };
            float alpha = 0.88f;
            if (solid) {
                rgb = { 0.45f, 0.08f, 0.08f };
                alpha = 0.45f;
            } else if (lvl == 0 && st.irr_ready) {
                sz p_idx = ((sz)iz * dims.y + iy) * dims.x + ix;
                Vector3 avg;
                for (int a = 0; a < 6; a++) {
                    drie::Vec4 e = st.irr_cpu[p_idx * 6 + a];
                    avg += { e.x, e.y, e.z };
                }
                rgb = tone_rgb(avg * (1.0f / 6.0f));
            } else if (st.rad_ready) {
                sz p_idx = ((sz)iz * dims.y + iy) * dims.x + ix;
                Vector3 avg;
                int step = math::max(dd / 16, 1);
                int cnt = 0;
                for (int k = 0; k < dd; k += step) {
                    avg += unpack_rad((sz)off + p_idx * dd + k);
                    cnt++;
                }
                if (cnt > 0) rgb = tone_rgb(avg * (1.0f / (float)cnt));
            }

            float br = node_r + 1.0f;
            sgl::c4f(lcol[0], lcol[1], lcol[2], solid ? 0.3f : 0.85f);
            sgl::v2f(sp[0] - br, sp[1] - br);
            sgl::v2f(sp[0] + br, sp[1] - br);
            sgl::v2f(sp[0] + br, sp[1] + br);
            sgl::v2f(sp[0] - br, sp[1] + br);

            sgl::c4f(rgb[0], rgb[1], rgb[2], alpha);
            sgl::v2f(sp[0] - node_r, sp[1] - node_r);
            sgl::v2f(sp[0] + node_r, sp[1] - node_r);
            sgl::v2f(sp[0] + node_r, sp[1] + node_r);
            sgl::v2f(sp[0] - node_r, sp[1] + node_r);
        }
    }
    sgl::end();

    // 2. Angular ray spokes (projected onto XZ plane)
    if (!draw_angles || !st.rad_ready) return;
    float max_spoke = drie::clamp(spacing * px_per_m * 0.55f, 8.0f, 34.0f);
    sgl::begin_lines();
    for (int iz = 0; iz < dims.z; iz += angle_stride) {
        float wz = rc.min.z + ((float)iz + 0.5f) * spacing;
        for (int ix = 0; ix < dims.x; ix += angle_stride) {
            float wx = rc.min.x + ((float)ix + 0.5f) * spacing;
            float[2] sp = world_to_px(wx, wz, width, height);
            if (sp[0] < -36 || sp[0] > (float)(width + 36) || sp[1] < -36 || sp[1] > (float)(height + 36)) continue;

            sz p_idx = ((sz)iz * dims.y + iy) * dims.x + ix;
            int k_step = d > 16 ? 2 : 1;
            for (int ky = 0; ky < d; ky += k_step) {
                for (int kx = 0; kx < d; kx += k_step) {
                    int k = ky * d + kx;
                    float ex = ((float)kx + 0.5f) / (float)d * 2.0f - 1.0f;
                    float ey = ((float)ky + 0.5f) / (float)d * 2.0f - 1.0f;
                    Vector3 rd = oct_dir(ex, ey);
                    if (math::abs(rd.y) > 0.72f) continue;
                    Vector3 rad = unpack_rad((sz)off + p_idx * dd + k);
                    float lum = rad.x * 0.2126f + rad.y * 0.7152f + rad.z * 0.0722f;
                    float len = max_spoke * (0.35f + 0.65f * drie::clamp(lum * 2.5f / (0.25f + lum * 2.5f), 0, 1));
                    float[3] rgb = tone_rgb(rad * 3.0f);
                    float a = 0.3f + 0.65f * drie::clamp(lum * 3.0f / (0.15f + lum * 3.0f), 0, 1);
                    sgl::c4f(rgb[0], rgb[1], rgb[2], a);
                    sgl::v2f(sp[0], sp[1]);
                    sgl::v2f(sp[0] + rd.x * len, sp[1] + rd.z * len);
                }
            }
        }
    }
    sgl::end();
}

// Draws an enlarged angular radiance rose for the inspected probe in the top-right corner.
fn void draw_probe_rose(RadianceCascades* rc, int lvl, int cx, int cy, int cz, int width) @local {
    float rx = (float)width - 82.0f;
    float ry = 82.0f;
    float rad_px = 54.0f;
    sgl_rect(rx - 70, ry - 70, 140, 140, { 0.03f, 0.04f, 0.06f, 0.82f });
    sgl_box(rx - 70, ry - 70, rx + 70, ry + 70, { 0.25f, 0.55f, 0.7f, 0.6f });
    sgl_circle(rx, ry, rad_px, 32, { 0.3f, 0.4f, 0.5f, 0.45f });
    sgl_circle(rx, ry, rad_px * 0.5f, 24, { 0.3f, 0.4f, 0.5f, 0.25f });
    if (!st.rad_ready) return;

    int[<3>] dims = rc.level_dims(lvl);
    int d = rc.base_directions << lvl;
    int dd = d * d;
    int off = rc.offsets[lvl];
    sz p_idx = ((sz)cz * dims.y + cy) * dims.x + cx;

    sgl::begin_lines();
    for (int ky = 0; ky < d; ky++) {
        for (int kx = 0; kx < d; kx++) {
            int k = ky * d + kx;
            float ex = ((float)kx + 0.5f) / (float)d * 2.0f - 1.0f;
            float ey = ((float)ky + 0.5f) / (float)d * 2.0f - 1.0f;
            Vector3 rd = oct_dir(ex, ey);
            if (math::abs(rd.y) > 0.75f) continue;
            Vector3 rad = unpack_rad((sz)off + p_idx * dd + k);
            float lum = rad.x * 0.2126f + rad.y * 0.7152f + rad.z * 0.0722f;
            float len = rad_px * (0.2f + 0.8f * drie::clamp(lum * 3.0f / (0.3f + lum * 3.0f), 0, 1));
            float[3] rgb = tone_rgb(rad * 2.5f);
            sgl::c4f(rgb[0], rgb[1], rgb[2], 0.9f);
            sgl::v2f(rx, ry);
            sgl::v2f(rx + rd.x * len, ry + rd.z * len);
        }
    }
    sgl::end();
}

fn void draw_content(
    SglContext gl,
    SglPipeline pip,
    SdtxContext text,
    RadianceCascades* rc,
    String vol_name,
    int width,
    int height,
    Vector3 marker,
    Vector3 marker_dir
) @local {
    sgl::set_context(gl);
    sgl::defaults();
    sgl::load_pipeline(pip);
    sgl::matrix_mode_projection();
    sgl::ortho(0, (float)width, (float)height, 0, -1, 1);

    float px_per_m = (float)height / (2.0f * st.half_h);
    int slice0 = rc ? active_slice(rc) : 0;
    int active_lvl = rc ? (st.level < 0 ? 0 : math::clamp(st.level, 0, rc.levels - 1)) : 0;
    int cx = 0;
    int cy = 0;
    int cz = 0;

    if (rc) {
        float[2] b0 = world_to_px(rc.min.x, rc.min.z, width, height);
        float[2] b1 = world_to_px(rc.max.x, rc.max.z, width, height);
        sgl_box(b0[0], b0[1], b1[0], b1[1], { 0.4f, 0.85f, 1.0f, 0.8f });

        if (st.show_voxels) {
            int vy = math::clamp(slice0 * 4 + 2, 0, rc.vy - 1);
            float vs = rc.voxel_size;
            float vpx = math::max(vs * px_per_m, 1.0f);
            sgl::begin_quads();
            sgl::c4f(0.85f, 0.85f, 0.9f, 0.24f);
            for (int vz = 0; vz < rc.vz; vz++) {
                float wz = rc.min.z + (float)vz * vs;
                for (int vx = 0; vx < rc.vx; vx++) {
                    if ((rc.voxels[(((sz)vz * rc.vy + vy) * rc.vx + vx) * 2] & 0x1000000u) == 0) continue;
                    float wx = rc.min.x + (float)vx * vs;
                    float[2] sp = world_to_px(wx, wz, width, height);
                    if (sp[0] < -vpx || sp[0] > (float)width + vpx || sp[1] < -vpx || sp[1] > (float)height + vpx) continue;
                    sgl::v2f(sp[0], sp[1]);
                    sgl::v2f(sp[0] + vpx, sp[1]);
                    sgl::v2f(sp[0] + vpx, sp[1] + vpx);
                    sgl::v2f(sp[0], sp[1] + vpx);
                }
            }
            sgl::end();
        }

        if (st.level < 0) {
            for (int l = rc.levels - 1; l >= 0; l--) {
                draw_level_probes(rc, l, slice0, st.show_angles && l == 0, width, height);
            }
        } else {
            draw_level_probes(rc, active_lvl, slice0, st.show_angles, width, height);
        }

        int[<3>] dims = rc.level_dims(active_lvl);
        float spacing = rc.spacing * (float)(1 << active_lvl);
        cx = math::clamp((int)math::floor((st.pan_x - rc.min.x) / spacing), 0, dims.x - 1);
        cy = math::clamp(slice0 >> active_lvl, 0, dims.y - 1);
        cz = math::clamp((int)math::floor((st.pan_z - rc.min.z) / spacing), 0, dims.z - 1);
        float cwx = rc.min.x + ((float)cx + 0.5f) * spacing;
        float cwz = rc.min.z + ((float)cz + 0.5f) * spacing;
        float[2] csp = world_to_px(cwx, cwz, width, height);
        float t0 = rc.interval * (math::pow(4.0f, (float)active_lvl) - 1.0f) / 3.0f;
        float t1 = rc.interval * (math::pow(4.0f, (float)(active_lvl + 1)) - 1.0f) / 3.0f;
        float[4] lcol = LEVEL_COLORS[math::min(active_lvl, 4)];
        if (t0 > 0.001f) sgl_circle(csp[0], csp[1], t0 * px_per_m, 48, { lcol[0], lcol[1], lcol[2], 0.45f });
        sgl_circle(csp[0], csp[1], t1 * px_per_m, 64, { lcol[0], lcol[1], lcol[2], 0.85f });
        sgl_box(csp[0] - 8, csp[1] - 8, csp[0] + 8, csp[1] + 8, { 1, 1, 1, 0.95f });

        if (st.show_angles) draw_probe_rose(rc, active_lvl, cx, cy, cz, width);
    }

    // Optional player/camera marker
    if (marker_dir != (Vector3){}) {
        float[2] psp = world_to_px(marker.x, marker.z, width, height);
        sgl::begin_lines();
        sgl::c4f(1.0f, 0.95f, 0.25f, 0.95f);
        sgl::v2f(psp[0], psp[1]);
        sgl::v2f(psp[0] + marker_dir.x * 22.0f, psp[1] + marker_dir.z * 22.0f);
        sgl::end();
        sgl_rect(psp[0] - 4, psp[1] - 4, 8, 8, { 1.0f, 0.92f, 0.2f, 0.95f });
    }

    // Screen center crosshair
    sgl::begin_lines();
    sgl::c4f(1, 1, 1, 0.35f);
    sgl::v2f((float)width * 0.5f - 10, (float)height * 0.5f);
    sgl::v2f((float)width * 0.5f + 10, (float)height * 0.5f);
    sgl::v2f((float)width * 0.5f, (float)height * 0.5f - 10);
    sgl::v2f((float)width * 0.5f, (float)height * 0.5f + 10);
    sgl::end();

    // Translucent dark HUD panels
    sgl_rect(8, 8, 576, 178, { 0.03f, 0.04f, 0.06f, 0.84f });
    sgl_box(8, 8, 584, 186, { 0.25f, 0.55f, 0.7f, 0.65f });
    sgl_rect(8, (float)height - 28, (float)width - 16, 20, { 0.03f, 0.04f, 0.06f, 0.84f });

    // Small crisp ASCII sokol_debugtext HUD (FONT_ORIC, 8x8 px cells)
    sdtx::set_context(text);
    sdtx::canvas((float)width, (float)height);
    sdtx::origin(0, 0);
    sdtx::font(ui::FONT_ORIC);

    @pool() {
        float row = 2.0f;
        sdtx::color3f(0.35f, 0.92f, 1.0f);
        sdtx::pos(2.0f, row);
        sdtx::puts(string::tformat("RC DEBUG  [TOP-DOWN | VHS OFF]  VOL: %s", vol_name.len ? vol_name : "NONE").zstr_tcopy());
        row += 1.6f;

        sdtx::color3f(0.88f, 0.90f, 0.94f);
        if (rc) {
            int total_probes = rc.nx * rc.ny * rc.nz;
            int total_rays = rc.offsets[rc.levels - 1] + rc.counts[rc.levels - 1];
            float rad_mb = (float)((double)total_rays * 8.0 / (1024.0 * 1024.0));
            float irr_mb = (float)((double)total_probes * 6.0 * 16.0 / (1024.0 * 1024.0));
            sdtx::pos(2.0f, row);
            sdtx::puts(string::tformat("GRID: %dx%dx%d (%d probes)  VOX: %dx%dx%d (%.3fm)",
                rc.nx, rc.ny, rc.nz, total_probes, rc.vx, rc.vy, rc.vz, rc.voxel_size).zstr_tcopy());
            row += 1.4f;

            sdtx::pos(2.0f, row);
            sdtx::puts(string::tformat("BOX: (%.1f,%.1f,%.1f)..(%.1f,%.1f,%.1f)  LVL:%d  DIR:%d",
                rc.min.x, rc.min.y, rc.min.z, rc.max.x, rc.max.y, rc.max.z, rc.levels, rc.base_directions).zstr_tcopy());
            row += 1.4f;

            sdtx::pos(2.0f, row);
            sdtx::puts(string::tformat("CFG: every=%d  blend=%.2f  bounce=%.2f  frame=%d  VRAM=%.1fMB",
                rc.update_every, rc.blend, rc.bounce, rc.frame, rad_mb + irr_mb).zstr_tcopy());
            row += 1.6f;

            int[<3>] ld = rc.level_dims(active_lvl);
            float l_spacing = rc.spacing * (float)(1 << active_lvl);
            int d = rc.base_directions << active_lvl;
            float t0 = rc.interval * (math::pow(4.0f, (float)active_lvl) - 1.0f) / 3.0f;
            float t1 = rc.interval * (math::pow(4.0f, (float)(active_lvl + 1)) - 1.0f) / 3.0f;
            float[4] lc = LEVEL_COLORS[math::min(active_lvl, 4)];
            sdtx::color3f(lc[0], lc[1], lc[2]);
            sdtx::pos(2.0f, row);
            String ltag = st.level < 0 ? string::tformat("ALL (ring L%d)", active_lvl) : string::tformat("L%d", active_lvl);
            sdtx::puts(string::tformat("CASCADE %s: %dx%dx%d  sp=%.2fm  ang=%dx%d (%d)  [%.2f..%.2fm]",
                ltag, ld.x, ld.y, ld.z, l_spacing, d, d, d * d, t0, t1).zstr_tcopy());
            row += 1.6f;

            float slice_world_y = rc.min.y + ((float)slice0 + 0.5f) * rc.spacing;
            sdtx::color3f(0.85f, 0.88f, 0.92f);
            sdtx::pos(2.0f, row);
            sdtx::puts(string::tformat("VIEW: pan=(%.2f,%.2f)  zoom=%.1fm  cutY=%.2fm  slice=%d/%d (y=%.2fm)",
                st.pan_x, st.pan_z, st.half_h, st.cut_y, slice0, rc.ny - 1, slice_world_y).zstr_tcopy());
            row += 1.4f;

            if (active_lvl == 0 && st.irr_ready) {
                sz p_idx = ((sz)cz * ld.y + cy) * ld.x + cx;
                drie::Vec4 e0 = st.irr_cpu[p_idx * 6 + 0];
                bool valid = e0.w > 0.5f;
                Vector3 avg;
                for (int a = 0; a < 6; a++) {
                    drie::Vec4 e = st.irr_cpu[p_idx * 6 + a];
                    avg += { e.x, e.y, e.z };
                }
                avg *= (1.0f / 6.0f);
                sdtx::color3f(valid ? 0.55f : 1.0f, valid ? 1.0f : 0.45f, valid ? 0.65f : 0.45f);
                sdtx::pos(2.0f, row);
                sdtx::puts(string::tformat("PROBE (%d,%d,%d) %s  irr=(%.2f, %.2f, %.2f)  angles=%s  vox=%s",
                    cx, cy, cz, valid ? "VALID" : "SOLID", avg.x, avg.y, avg.z,
                    st.show_angles ? "ON" : "OFF", st.show_voxels ? "ON" : "OFF").zstr_tcopy());
            } else {
                sdtx::pos(2.0f, row);
                sdtx::puts(string::tformat("PROBE (%d,%d,%d)  angles=%s  vox=%s  readback=%s",
                    cx, cy, cz, st.show_angles ? "ON" : "OFF", st.show_voxels ? "ON" : "OFF",
                    st.rad_ready ? "LIVE" : "WAIT").zstr_tcopy());
            }
            row += 1.6f;

            sdtx::pos(2.0f, row);
            sdtx::color3f(0.15f, 0.85f, 1.0f); sdtx::puts("L0 ");
            sdtx::color3f(0.25f, 1.0f, 0.45f); sdtx::puts("L1 ");
            sdtx::color3f(1.0f, 0.78f, 0.2f);  sdtx::puts("L2 ");
            sdtx::color3f(1.0f, 0.35f, 0.85f); sdtx::puts("L3  ");
            sdtx::color3f(1.0f, 0.92f, 0.2f);  sdtx::puts("[PLAYER]  ");
            sdtx::color3f(0.95f, 0.35f, 0.35f); sdtx::puts("[SOLID]");

            if (st.show_angles) {
                sdtx::color3f(0.75f, 0.85f, 0.95f);
                sdtx::pos(((float)width - 146.0f) / 8.0f, 2.0f);
                sdtx::puts(string::tformat("ROSE L%d (%d,%d,%d)", active_lvl, cx, cy, cz).zstr_tcopy());
            }
        } else {
            sdtx::pos(2.0f, row);
            sdtx::puts("NO ACTIVE RADIANCE CASCADES VOLUME IN THIS SCENE");
            row += 1.4f;
            sdtx::pos(2.0f, row);
            sdtx::puts(string::tformat("VIEW: pan=(%.2f,%.2f)  zoom=%.1fm  cutY=%.2fm",
                st.pan_x, st.pan_z, st.half_h, st.cut_y).zstr_tcopy());
        }

        sdtx::color3f(0.72f, 0.78f, 0.85f);
        sdtx::pos(2.0f, ((float)height - 22.0f) / 8.0f);
        sdtx::puts("ARROWS:pan (SHIFT:fast)  +/-:zoom  [/]:cut Y  W/S:slice  0-3/4:cascade  R:angles  B:voxels  SPACE:reset  F3:exit");
    };

    sgl::context_draw(gl);
    sdtx::context_draw(text);
}

fn SgPipelineDesc blend_desc() @local => {
    .colors[0].write_mask = RGBA,
    .colors[0].blend = {
        .enabled = true,
        .src_factor_rgb = SRC_ALPHA, .dst_factor_rgb = ONE_MINUS_SRC_ALPHA,
        .src_factor_alpha = ONE, .dst_factor_alpha = ONE_MINUS_SRC_ALPHA,
    },
};

fn void draw_window(
    RadianceCascades* rc,
    String vol_name,
    int width,
    int height,
    Vector3 marker = {},
    Vector3 marker_dir = {}
) {
    if (!active || !sapp::isvalid()) return;
    if (!st.win_gl.id) {
        ui::setup();
        st.win_gl = sgl::make_context(&&{ .max_vertices = 262144, .max_commands = 64 });
        st.win_pip = sgl::context_make_pipeline(st.win_gl, &&blend_desc());
        st.win_text = sdtx::make_context(&&{ .char_buf_size = 4096, .canvas_width = (float)width, .canvas_height = (float)height });
    }
    SlbxViewport vp = slbx::letterbox(sapp::width(), sapp::height(), &&{ .content_aspect_ratio = (float)width / (float)height });
    sg::apply_viewport(vp.x, vp.y, vp.width, vp.height, true);
    draw_content(st.win_gl, st.win_pip, st.win_text, rc, vol_name, width, height, marker, marker_dir);
    sg::apply_viewport(0, 0, sapp::width(), sapp::height(), true);
}

fn void draw_target(
    RenderTarget* target,
    RadianceCascades* rc,
    String vol_name,
    int width,
    int height,
    Vector3 marker = {},
    Vector3 marker_dir = {}
) {
    if (!active || !target) return;
    if (!st.img_gl.id) {
        ui::setup();
        st.img_gl = sgl::make_context(&&{
            .max_vertices = 262144,
            .max_commands = 64,
            .color_format = RGBA8,
            .depth_format = NONE,
            .sample_count = 1,
        });
        st.img_pip = sgl::context_make_pipeline(st.img_gl, &&blend_desc());
        st.img_text = sdtx::make_context(&&{
            .char_buf_size = 4096,
            .canvas_width = (float)width,
            .canvas_height = (float)height,
            .color_format = RGBA8,
            .depth_format = NONE,
            .sample_count = 1,
        });
    }
    target.begin(LOAD, label: "drie-rc-debug");
    draw_content(st.img_gl, st.img_pip, st.img_text, rc, vol_name, width, height, marker, marker_dir);
    target.end();
}
