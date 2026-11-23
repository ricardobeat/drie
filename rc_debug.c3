// Optional top-down Radiance Cascades debug visualizer module (`import drie::rc_debug;`).
// Slices the scene below `cut_y` with a top-down orthographic camera, async-reads the live
// GPU cascade and irradiance buffers, and draws probe nodes, angular ray roses, cascade
// interval rings, optional solid voxels, and a compact sokol_debugtext HUD.
module drie::rc_debug;
import drie, drie::ui, drie::readback, sokol;
import std::math, std::io;

bool active;
bool freeze_alloc = true;

const int IRR_SLOT = 10;
const int RAD_SLOT = 11;

struct State @local {
    bool initialized;
    OrthographicCamera* cam;
    PerspectiveCamera* cam3d;
    bool view_3d;
    bool all_slices_3d;
    float pan_x;
    float pan_z;
    float half_h;
    float cut_y;
    float orbit_yaw;
    float orbit_pitch;
    float orbit_dist;
    bool dragging;
    int level; // 0..levels-1, or -1 for ALL
    int slice_y; // cascade-0 probe Y index (-1 = auto from cut_y)
    bool show_angles;
    bool show_voxels;
    bool color_by_age;
    bool[512] keys;
    drie::Mat4 last_vp; // active camera view_proj for 3D projection

    // Async GPU readback buffers (1-frame-late, zero stall)
    RadianceCascades* last_rc;
    drie::Vec4[] irr_cpu;
    uint[] rad_cpu; // 2 uints (uvec2 fp16) per cascade ray
    bool irr_ready;
    bool rad_ready;
    bool irr_inflight;
    bool rad_inflight;

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
    .orbit_yaw = -0.65f,
    .orbit_pitch = 0.62f,
    .orbit_dist = 9.5f,
    .all_slices_3d = true,
    .level = 0,
    .slice_y = -1,
    .show_angles = true,
    .show_voxels = false,
    .color_by_age = true,
};

fn void set_3d(bool on) {
    st.view_3d = on;
}

fn void reset(Vector3 focus, float cut_y) {
    st.pan_x = focus.x;
    st.pan_z = focus.z;
    st.half_h = 6.5f;
    st.cut_y = cut_y;
    st.orbit_yaw = -0.65f;
    st.orbit_pitch = 0.62f;
    st.orbit_dist = 9.5f;
    st.slice_y = -1;
}

fn void toggle(Vector3 focus = { }, float cut_y = 2.4f) {
    active = !active;
    if (active) {
        reset(focus, cut_y);
        st.keys = { };
        if (sapp::isvalid()) sapp::lock_mouse(false);
    }
}

fn int active_slice(RadianceCascades* rc) @local {
    if (!rc) return 0;
    if (st.slice_y >= 0) return math::clamp(st.slice_y, 0, rc.ny - 1);
    float target_y = st.cut_y - 0.8f;
    int iy         = (int)math::floor((target_y - rc.min.y) / rc.spacing);
    return math::clamp(iy, 0, rc.ny - 1);
}

// Handles RC debug hotkeys while active (and `toggle_key` to enter/exit). Returns true if consumed.
fn bool handle_event(Event* e, Vector3 focus = { }, float cut_y = 2.4f, KeyCode toggle_key = F3) {
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
                    case NUM_4: st.level = -1;
                    case M: st.view_3d = !st.view_3d;
                    case A: st.all_slices_3d = !st.all_slices_3d;
                    case R: st.show_angles = !st.show_angles;
                    case B: st.show_voxels = !st.show_voxels;
                    case C: st.color_by_age = !st.color_by_age;
                    case P: freeze_alloc = !freeze_alloc;
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
                    case TAB:
                    case L:
                        return false;
                    default:
                        break;
                }
            }
            return true;
        case KEY_UP:
            if ((int)e.key_code < 512) st.keys[(int)e.key_code] = false;
            return true;
        case MOUSE_DOWN:
            if (e.mouse_button == LEFT) st.dragging = true;
            return true;
        case MOUSE_UP:
            if (e.mouse_button == LEFT) st.dragging = false;
            return true;
        case MOUSE_MOVE:
            if (st.dragging && st.view_3d) {
                st.orbit_yaw -= e.mouse_dx * 0.006f;
                st.orbit_pitch = drie::clamp(st.orbit_pitch + e.mouse_dy * 0.006f, 0.08f, 1.48f);
            }
            return true;
        case MOUSE_SCROLL:
            float factor = e.scroll_y > 0 ? 0.9f : 1.11f;
            if (st.view_3d) {
                st.orbit_dist = drie::clamp(st.orbit_dist * factor, 2.0f, 45.0f);
            } else {
                st.half_h = drie::clamp(st.half_h * factor, 1.5f, 35.0f);
            }
            return true;
        case UNFOCUSED:
            st.keys = { };
            st.dragging = false;
            return false;
        default:
            return false;
    }
}

// Updates pan/zoom/orbit/cut plane and returns the active debug Camera (top-down Ortho or 3D Perspective).
fn Camera* update(float dt, int width, int height, Vector3 focus = { }, float cut_y = 2.4f) {
    if (!st.initialized) {
        st.initialized = true;
        st.cam = drie::orthographic_camera(-10, 10, 10, -10, 10, 45);
        st.cam3d = drie::perspective_camera(55, 16.0f / 9.0f, 0.1f, 150.0f);
        reset(focus, cut_y);
    }
    float mult       = (st.keys[(int)KeyCode.LEFT_SHIFT] || st.keys[(int)KeyCode.RIGHT_SHIFT]) ? 2.5f : 1.0f;
    float base_scale = st.view_3d ? st.orbit_dist * 0.55f : st.half_h;
    float speed      = base_scale * 0.9f * mult * dt;
    if (st.view_3d) {
        float sy = math::sin(st.orbit_yaw);
        float cy = math::cos(st.orbit_yaw);
        // forward on XZ is (-sy, -cy), right is (cy, -sy)
        if (st.keys[(int)KeyCode.UP]) {
            st.pan_x -= sy * speed;
            st.pan_z -= cy * speed;
        }
        if (st.keys[(int)KeyCode.DOWN]) {
            st.pan_x += sy * speed;
            st.pan_z += cy * speed;
        }
        if (st.keys[(int)KeyCode.RIGHT]) {
            st.pan_x += cy * speed;
            st.pan_z -= sy * speed;
        }
        if (st.keys[(int)KeyCode.LEFT]) {
            st.pan_x -= cy * speed;
            st.pan_z += sy * speed;
        }
        if (st.keys[(int)KeyCode.Q]) st.orbit_yaw += 1.4f * dt;
        if (st.keys[(int)KeyCode.E]) st.orbit_yaw -= 1.4f * dt;
        if (st.keys[(int)KeyCode.PAGE_UP]) st.orbit_pitch = drie::clamp(st.orbit_pitch + 1.0f * dt, 0.08f, 1.48f);
        if (st.keys[(int)KeyCode.PAGE_DOWN]) st.orbit_pitch = drie::clamp(st.orbit_pitch - 1.0f * dt, 0.08f, 1.48f);
        if (st.keys[(int)KeyCode.EQUAL] || st.keys[(int)KeyCode.KP_ADD]) {
            st.orbit_dist = drie::clamp(st.orbit_dist * (1.0f - 1.5f * dt), 2.0f, 45.0f);
        }
        if (st.keys[(int)KeyCode.MINUS] || st.keys[(int)KeyCode.KP_SUBTRACT]) {
            st.orbit_dist = drie::clamp(st.orbit_dist * (1.0f + 1.5f * dt), 2.0f, 45.0f);
        }
    } else {
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
    }
    if (st.keys[(int)KeyCode.LEFT_BRACKET]) st.cut_y -= 2.0f * dt;
    if (st.keys[(int)KeyCode.RIGHT_BRACKET]) st.cut_y += 2.0f * dt;

    float aspect = height > 0 ? (float)width / (float)height : 16.0f / 9.0f;
    if (st.view_3d) {
        st.cam3d.aspect = aspect;
        st.cam3d.near = 0.1f;
        st.cam3d.far = 150.0f;
        Vector3 target = { st.pan_x, st.cut_y - 1.2f, st.pan_z };
        float cp       = math::cos(st.orbit_pitch);
        float sp       = math::sin(st.orbit_pitch);
        Vector3 offset = {
            math::sin(st.orbit_yaw) * cp * st.orbit_dist,
            sp * st.orbit_dist,
            math::cos(st.orbit_yaw) * cp * st.orbit_dist,
        };
        st.cam3d.position = target + offset;
        st.cam3d.rotation = { };
        // update_matrix_world() BEFORE look_at_vec() so look_at_vec reads the new camera position!
        st.cam3d.update_matrix_world();
        st.cam3d.look_at_vec(target);
        st.cam3d.update_matrix_world();
        st.cam3d.custom_projection = false;
        st.cam3d.update_projection_matrix();
        st.last_vp = st.cam3d.projection_matrix.mul(st.cam3d.matrix_world.invert());
        // Slice roof/ceiling above cut_y in 3D orbit mode too when camera is above cut_y
        if (st.cam3d.position.y > st.cut_y + 0.15f) {
            drie::oblique_projection(st.cam3d, { 0, -1, 0 }, -st.cut_y, 0);
            st.cam3d.custom_projection = true;
        }
        return st.cam3d;
    }

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
    st.last_vp = st.cam.projection_matrix.mul(st.cam.matrix_world.invert());
    return st.cam;
}

// Queues async readback of `rc`'s GPU buffers after `rc.update(scene)`.
fn void track(RadianceCascades* rc, bool wait_now = false) {
    if (!active || !rc || !rc.irradiance_buf || !rc.cascade_buf) return;
    if (st.last_rc != rc) {
        if (st.irr_cpu.len) free(st.irr_cpu);
        if (st.rad_cpu.len) free(st.rad_cpu);
        sz irr_count = rc.sparse ? (sz)(rc.hash_cap + rc.level_cap[0] * 6) : (sz)rc.nx * rc.ny * rc.nz * 6;
        sz rad_rays  = (sz)(rc.offsets[rc.levels - 1] + rc.counts[rc.levels - 1]);
        st.irr_cpu = mem::new_array(drie::Vec4, irr_count);
        st.rad_cpu = mem::new_array(uint, rad_rays * 2);
        st.last_rc = rc;
        st.irr_ready = false;
        st.rad_ready = false;
        st.irr_inflight = false;
        st.rad_inflight = false;
    }
    CInt irr_bytes = (CInt)(st.irr_cpu.len * drie::Vec4::size);
    CInt rad_bytes = (CInt)(st.rad_cpu.len * uint::size);
    if (st.irr_inflight) {
        if (readback::fetch(IRR_SLOT, st.irr_cpu.ptr, irr_bytes, wait_now)) {
            st.irr_ready = true;
            st.irr_inflight = false;
        }
    }
    if (!st.irr_inflight) {
        if (readback::track(rc.irradiance_buf.buffer, IRR_SLOT)) st.irr_inflight = true;
    }
    if (st.rad_inflight) {
        if (readback::fetch(RAD_SLOT, st.rad_cpu.ptr, rad_bytes, wait_now)) {
            st.rad_ready = true;
            st.rad_inflight = false;
        }
    }
    if (!st.rad_inflight) {
        if (readback::track(rc.cascade_buf.buffer, RAD_SLOT)) st.rad_inflight = true;
    }
}

fn float half_to_float(ushort h) @local {
    uint sign = ((uint)h & 0x8000u) << 16;
    uint exp  = ((uint)h >> 10) & 0x1fu;
    uint mant = (uint)h & 0x3ffu;
    uint bits;
    if (exp == 0) {
        if (mant == 0) return bitcast(sign, float);
        while ((mant & 0x400u) == 0) {
            mant <<= 1;
            exp--;
        }
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
    float t  = math::max(-nz, 0.0f);
    float nx = ex + (ex >= 0 ? -t : t);
    float ny = ey + (ey >= 0 ? -t : t);
    return ((Vector3) { nx, ny, nz }).normalize();
}

fn float[3] age_color(uint last_frame, int cur_frame) @local {
    uint cf  = (uint)math::max(cur_frame, 1);
    uint age = cf >= last_frame ? cf - last_frame : 0;
    if (age <= 2) return { 0.15f, 1.00f, 0.55f }; // fresh (current frame): bright green-cyan
    if (age <= 15) {
        float t = (float)(age - 2) / 13.0f;
        return { 0.15f + 0.85f * t, 1.00f - 0.10f * t, 0.55f * (1.0f - t) }; // -> yellow
    }
    if (age <= 60) {
        float t = (float)(age - 15) / 45.0f;
        return { 1.00f, 0.90f - 0.50f * t, 0.10f }; // -> orange
    }
    float t = drie::clamp((float)(age - 60) / 120.0f, 0, 1);
    return { 1.00f - 0.65f * t, 0.40f - 0.12f * t, 0.15f + 0.70f * t }; // -> cool violet-blue
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
    float hw     = st.half_h * aspect;
    float hh     = st.half_h;
    float sx     = ((wx - st.pan_x) / hw * 0.5f + 0.5f) * (float)width;
    float sy     = ((wz - st.pan_z) / hh * 0.5f + 0.5f) * (float)height;
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

fn int unwrap_z(uint key, RadianceCascades* rc, int lvl) @local {
    float sp   = rc.spacing * (float)(1 << lvl);
    int z_mod  = (int)((key >> 18) & 0x1ffu);
    int base_z = (int)math::floor(rc.min.z / sp) - 16;
    return base_z + ((z_mod - base_z) & 511);
}

fn void draw_sparse_level_probes(
    RadianceCascades* rc,
    int lvl,
    int slice0,
    bool draw_angles,
    int width,
    int height
) @local {
    if (!st.irr_ready) return;
    float spacing   = rc.spacing * (float)(1 << lvl);
    int target_iy   = (int)math::floor((rc.min.y + ((float)slice0 + 0.5f) * rc.spacing) / spacing);
    int d           = rc.base_directions << lvl;
    int dd          = d * d;
    int off         = rc.offsets[lvl];
    float[4] lcol   = LEVEL_COLORS[math::min(lvl, 4)];
    float px_per_m  = (float)height / (2.0f * st.half_h);
    float node_r    = drie::clamp(spacing * px_per_m * 0.16f, 2.5f, 7.0f);
    float max_spoke = drie::clamp(spacing * px_per_m * 0.85f, 14.0f, 52.0f);

    sgl::begin_quads();
    for (int h = 0; h < rc.hash_cap; h++) {
        drie::Vec4 raw = st.irr_cpu[h];
        uint key_lo    = bitcast(raw.x, uint);
        uint key_hi    = bitcast(raw.y, uint);
        if ((key_hi & 0x80000000u) == 0) continue;
        int plvl = (int)((key_hi >> 27) & 0x7u);
        if (plvl != lvl) continue;
        int ix = (int)(key_lo & 0xffffu) - 32768;
        int iy = (int)((key_lo >> 16) & 0xffffu) - 32768;
        int iz = unwrap_z(key_hi, rc, plvl);
        if (math::abs(iy - target_iy) > 1) continue;
        float wx    = ((float)ix + 0.5f) * spacing;
        float wz    = ((float)iz + 0.5f) * spacing;
        float[2] sp = world_to_px(wx, wz, width, height);
        if (sp[0] < -16 || sp[0] > (float)(width + 16) || sp[1] < -16 || sp[1] > (float)(height + 16)) continue;

        uint slot    = bitcast(raw.z, uint);
        uint last_f  = bitcast(raw.w, uint);
        float[3] rgb = { lcol[0] * 0.4f, lcol[1] * 0.4f, lcol[2] * 0.4f };
        bool solid   = false;
        if (st.color_by_age) {
            if (lvl == 0 && (int)slot < rc.level_cap[0]) {
                solid = st.irr_cpu[(sz)rc.hash_cap + (sz)slot * 6].w < 0.5f;
            }
            rgb = solid ? (float[3]) { 0.45f, 0.08f, 0.08f } : age_color(last_f, rc.frame);
        } else if (lvl == 0 && (int)slot < rc.level_cap[0]) {
            sz base       = (sz)rc.hash_cap + (sz)slot * 6;
            drie::Vec4 e0 = st.irr_cpu[base];
            solid = e0.w < 0.5f;
            if (!solid) {
                Vector3 avg;
                for (int a = 0; a < 6; a++) {
                    drie::Vec4 e = st.irr_cpu[base + a];
                    avg += { e.x, e.y, e.z };
                }
                rgb = tone_rgb(avg * (1.0f / 6.0f));
            } else {
                rgb = { 0.45f, 0.08f, 0.08f };
            }
        } else if (st.rad_ready && (int)slot < rc.level_cap[lvl]) {
            Vector3 avg;
            int step = math::max(dd / 16, 1);
            int cnt  = 0;
            for (int k = 0; k < dd; k += step) {
                avg += unpack_rad((sz)off + (sz)slot * dd + k);
                cnt++;
            }
            if (cnt > 0) rgb = tone_rgb(avg * (1.0f / (float)cnt));
        }

        float br = node_r + 1.0f;
        sgl::c4f(lcol[0], lcol[1], lcol[2], solid ? 0.35f : 0.9f);
        sgl::v2f(sp[0] - br, sp[1] - br);
        sgl::v2f(sp[0] + br, sp[1] - br);
        sgl::v2f(sp[0] + br, sp[1] + br);
        sgl::v2f(sp[0] - br, sp[1] + br);

        sgl::c4f(rgb[0], rgb[1], rgb[2], solid ? 0.45f : 0.9f);
        sgl::v2f(sp[0] - node_r, sp[1] - node_r);
        sgl::v2f(sp[0] + node_r, sp[1] - node_r);
        sgl::v2f(sp[0] + node_r, sp[1] + node_r);
        sgl::v2f(sp[0] - node_r, sp[1] + node_r);
    }
    sgl::end();

    if (!draw_angles || !st.rad_ready) return;
    sgl::begin_lines();
    for (int h = 0; h < rc.hash_cap; h++) {
        drie::Vec4 raw = st.irr_cpu[h];
        uint key_lo    = bitcast(raw.x, uint);
        uint key_hi    = bitcast(raw.y, uint);
        if ((key_hi & 0x80000000u) == 0) continue;
        int plvl = (int)((key_hi >> 27) & 0x7u);
        if (plvl != lvl) continue;
        int ix = (int)(key_lo & 0xffffu) - 32768;
        int iy = (int)((key_lo >> 16) & 0xffffu) - 32768;
        int iz = unwrap_z(key_hi, rc, plvl);
        if (iy != target_iy || ((ix + iz) & 1) != 0) continue;
        float wx    = ((float)ix + 0.5f) * spacing;
        float wz    = ((float)iz + 0.5f) * spacing;
        float[2] sp = world_to_px(wx, wz, width, height);
        if (sp[0] < -36 || sp[0] > (float)(width + 36) || sp[1] < -36 || sp[1] > (float)(height + 36)) continue;

        uint slot = bitcast(raw.z, uint);
        if ((int)slot >= rc.level_cap[lvl]) continue;
        int k_step = d > 16 ? 2 : 1;
        for (int ky = 0; ky < d; ky += k_step) {
            for (int kx = 0; kx < d; kx += k_step) {
                int k      = ky * d + kx;
                float ex   = ((float)kx + 0.5f) / (float)d * 2.0f - 1.0f;
                float ey   = ((float)ky + 0.5f) / (float)d * 2.0f - 1.0f;
                Vector3 rd = oct_dir(ex, ey);
                if (math::abs(rd.y) > 0.72f) continue;
                Vector3 rad  = unpack_rad((sz)off + (sz)slot * dd + k);
                float lum    = rad.x * 0.2126f + rad.y * 0.7152f + rad.z * 0.0722f;
                float len    = max_spoke * (0.35f + 0.65f * drie::clamp(lum * 2.5f / (0.25f + lum * 2.5f), 0, 1));
                float[3] rgb = tone_rgb(rad * 3.0f);
                sgl::c4f(rgb[0], rgb[1], rgb[2], 0.35f + 0.6f * drie::clamp(lum * 3.0f / (0.15f + lum * 3.0f), 0, 1));
                sgl::v2f(sp[0], sp[1]);
                sgl::v2f(sp[0] + rd.x * len, sp[1] + rd.z * len);
            }
        }
    }
    sgl::end();
}

fn void draw_level_probes(RadianceCascades* rc, int lvl, int slice0, bool draw_angles, int width, int height) @local {
    if (rc.sparse) {
        draw_sparse_level_probes(rc, lvl, slice0, draw_angles, width, height);
        return;
    }
    int[<3>] dims    = rc.level_dims(lvl);
    float spacing    = rc.spacing * (float)(1 << lvl);
    int iy           = math::clamp(slice0 >> lvl, 0, dims.y - 1);
    int d            = rc.base_directions << lvl;
    int dd           = d * d;
    int off          = rc.offsets[lvl];
    float[4] lcol    = LEVEL_COLORS[math::min(lvl, 4)];
    float px_per_m   = (float)height / (2.0f * st.half_h);
    float node_r     = drie::clamp(spacing * px_per_m * 0.15f, 2.0f, 6.5f);
    float probe_px   = spacing * px_per_m;
    int angle_stride = probe_px < 16 ? 3 : (probe_px < 28 ? 2 : 1);

    // 1. Probe nodes
    sgl::begin_quads();
    for (int iz = 0; iz < dims.z; iz++) {
        float wz = rc.min.z + ((float)iz + 0.5f) * spacing;
        for (int ix = 0; ix < dims.x; ix++) {
            float wx    = rc.min.x + ((float)ix + 0.5f) * spacing;
            float[2] sp = world_to_px(wx, wz, width, height);
            if (sp[0] < -16 || sp[0] > (float)(width + 16) || sp[1] < -16 || sp[1] > (float)(height + 16)) continue;

            int vx     = (int)math::floor((wx - rc.min.x) / rc.voxel_size);
            int vy     = (int)math::floor(((rc.min.y + ((float)iy + 0.5f) * spacing) - rc.min.y) / rc.voxel_size);
            int vz     = (int)math::floor((wz - rc.min.z) / rc.voxel_size);
            bool solid = false;
            if (vx >= 0 && vx < rc.vx && vy >= 0 && vy < rc.vy && vz >= 0 && vz < rc.vz) {
                solid = (rc.voxels[(((sz)vz * rc.vy + vy) * rc.vx + vx) * 2] & 0x1000000u) != 0;
            }

            float[3] rgb = { lcol[0] * 0.35f, lcol[1] * 0.35f, lcol[2] * 0.35f };
            float alpha  = 0.88f;
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
                int cnt  = 0;
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
    float max_spoke = drie::clamp(spacing * px_per_m * 0.85f, 14.0f, 52.0f);
    sgl::begin_lines();
    for (int iz = 0; iz < dims.z; iz += angle_stride) {
        float wz = rc.min.z + ((float)iz + 0.5f) * spacing;
        for (int ix = 0; ix < dims.x; ix += angle_stride) {
            float wx    = rc.min.x + ((float)ix + 0.5f) * spacing;
            float[2] sp = world_to_px(wx, wz, width, height);
            if (sp[0] < -36 || sp[0] > (float)(width + 36) || sp[1] < -36 || sp[1] > (float)(height + 36)) continue;

            sz p_idx   = ((sz)iz * dims.y + iy) * dims.x + ix;
            int k_step = d > 16 ? 2 : 1;
            for (int ky = 0; ky < d; ky += k_step) {
                for (int kx = 0; kx < d; kx += k_step) {
                    int k      = ky * d + kx;
                    float ex   = ((float)kx + 0.5f) / (float)d * 2.0f - 1.0f;
                    float ey   = ((float)ky + 0.5f) / (float)d * 2.0f - 1.0f;
                    Vector3 rd = oct_dir(ex, ey);
                    if (math::abs(rd.y) > 0.72f) continue;
                    Vector3 rad  = unpack_rad((sz)off + p_idx * dd + k);
                    float lum    = rad.x * 0.2126f + rad.y * 0.7152f + rad.z * 0.0722f;
                    float len    = max_spoke * (0.35f + 0.65f * drie::clamp(lum * 2.5f / (0.25f + lum * 2.5f), 0, 1));
                    float[3] rgb = tone_rgb(rad * 3.0f);
                    float a      = 0.3f + 0.65f * drie::clamp(lum * 3.0f / (0.15f + lum * 3.0f), 0, 1);
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
    float rx     = (float)width - 82.0f;
    float ry     = 82.0f;
    float rad_px = 54.0f;
    sgl_rect(rx - 70, ry - 70, 140, 140, { 0.03f, 0.04f, 0.06f, 0.82f });
    sgl_box(rx - 70, ry - 70, rx + 70, ry + 70, { 0.25f, 0.55f, 0.7f, 0.6f });
    sgl_circle(rx, ry, rad_px, 32, { 0.3f, 0.4f, 0.5f, 0.45f });
    sgl_circle(rx, ry, rad_px * 0.5f, 24, { 0.3f, 0.4f, 0.5f, 0.25f });
    if (!st.rad_ready) return;

    int[<3>] dims = rc.level_dims(lvl);
    int d         = rc.base_directions << lvl;
    int dd        = d * d;
    int off       = rc.offsets[lvl];
    sz p_idx      = ((sz)cz * dims.y + cy) * dims.x + cx;

    sgl::begin_lines();
    for (int ky = 0; ky < d; ky++) {
        for (int kx = 0; kx < d; kx++) {
            int k      = ky * d + kx;
            float ex   = ((float)kx + 0.5f) / (float)d * 2.0f - 1.0f;
            float ey   = ((float)ky + 0.5f) / (float)d * 2.0f - 1.0f;
            Vector3 rd = oct_dir(ex, ey);
            if (math::abs(rd.y) > 0.75f) continue;
            Vector3 rad  = unpack_rad((sz)off + p_idx * dd + k);
            float lum    = rad.x * 0.2126f + rad.y * 0.7152f + rad.z * 0.0722f;
            float len    = rad_px * (0.2f + 0.8f * drie::clamp(lum * 3.0f / (0.3f + lum * 3.0f), 0, 1));
            float[3] rgb = tone_rgb(rad * 2.5f);
            sgl::c4f(rgb[0], rgb[1], rgb[2], 0.9f);
            sgl::v2f(rx, ry);
            sgl::v2f(rx + rd.x * len, ry + rd.z * len);
        }
    }
    sgl::end();
}

fn void sgl_wire_box_3d(Vector3 b0, Vector3 b1, float[4] col) @local {
    sgl::begin_lines();
    sgl::c4f(col[0], col[1], col[2], col[3]);
    float[2] xs = { b0.x, b1.x };
    float[2] ys = { b0.y, b1.y };
    float[2] zs = { b0.z, b1.z };
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            sgl::v3f(xs[0], ys[i], zs[j]);
            sgl::v3f(xs[1], ys[i], zs[j]);
            sgl::v3f(xs[i], ys[0], zs[j]);
            sgl::v3f(xs[i], ys[1], zs[j]);
            sgl::v3f(xs[i], ys[j], zs[0]);
            sgl::v3f(xs[i], ys[j], zs[1]);
        }
    }
    sgl::end();
}

// Appends a 6-face shaded 3D cube inside an active sgl::begin_quads() block.
fn void sgl_cube_quads(Vector3 c, float r, float[3] rgb, float a) @local {
    float x0 = c.x - r;
    float x1 = c.x + r;
    float y0 = c.y - r;
    float y1 = c.y + r;
    float z0 = c.z - r;
    float z1 = c.z + r;
    // +Y (top, brightest)
    sgl::c4f(rgb[0], rgb[1], rgb[2], a);
    sgl::v3f(x0, y1, z0);
    sgl::v3f(x1, y1, z0);
    sgl::v3f(x1, y1, z1);
    sgl::v3f(x0, y1, z1);
    // -Y (bottom, darkest)
    sgl::c4f(rgb[0] * 0.55f, rgb[1] * 0.55f, rgb[2] * 0.55f, a);
    sgl::v3f(x0, y0, z1);
    sgl::v3f(x1, y0, z1);
    sgl::v3f(x1, y0, z0);
    sgl::v3f(x0, y0, z0);
    // +Z
    sgl::c4f(rgb[0] * 0.85f, rgb[1] * 0.85f, rgb[2] * 0.85f, a);
    sgl::v3f(x0, y0, z1);
    sgl::v3f(x1, y0, z1);
    sgl::v3f(x1, y1, z1);
    sgl::v3f(x0, y1, z1);
    // -Z
    sgl::c4f(rgb[0] * 0.72f, rgb[1] * 0.72f, rgb[2] * 0.72f, a);
    sgl::v3f(x1, y0, z0);
    sgl::v3f(x0, y0, z0);
    sgl::v3f(x0, y1, z0);
    sgl::v3f(x1, y1, z0);
    // +X
    sgl::c4f(rgb[0] * 0.92f, rgb[1] * 0.92f, rgb[2] * 0.92f, a);
    sgl::v3f(x1, y0, z1);
    sgl::v3f(x1, y0, z0);
    sgl::v3f(x1, y1, z0);
    sgl::v3f(x1, y1, z1);
    // -X
    sgl::c4f(rgb[0] * 0.65f, rgb[1] * 0.65f, rgb[2] * 0.65f, a);
    sgl::v3f(x0, y0, z0);
    sgl::v3f(x0, y0, z1);
    sgl::v3f(x0, y1, z1);
    sgl::v3f(x0, y1, z0);
}

fn void draw_3d_scene_overlay(
    RadianceCascades* rc,
    int active_lvl,
    int slice0,
    Vector3 marker,
    Vector3 marker_dir
) @local {
    sgl::matrix_mode_projection();
    sgl::load_matrix((float*)&st.last_vp);
    sgl::matrix_mode_modelview();
    sgl::load_identity();

    if (rc) {
        sgl_wire_box_3d(rc.min, rc.max, { 0.35f, 0.85f, 1.0f, 0.75f });

        float[4] lcol = LEVEL_COLORS[math::min(active_lvl, 4)];
        float spacing = rc.spacing * (float)(1 << active_lvl);
        float cube_r  = spacing * 0.16f;
        int d         = rc.base_directions << active_lvl;
        int dd        = d * d;
        int off       = rc.offsets[active_lvl];

        if (rc.sparse) {
            if (st.irr_ready) {
                float max_d2 = (st.orbit_dist * 2.5f) * (st.orbit_dist * 2.5f);
                int drawn    = 0;
                sgl::begin_quads();
                for (int h = 0; h < rc.hash_cap && drawn < 4000; h++) {
                    drie::Vec4 raw = st.irr_cpu[h];
                    uint key_lo    = bitcast(raw.x, uint);
                    uint key_hi    = bitcast(raw.y, uint);
                    if ((key_hi & 0x80000000u) == 0) continue;
                    int plvl = (int)((key_hi >> 27) & 0x7u);
                    if (st.level >= 0 && plvl != active_lvl) continue;
                    int ix      = (int)(key_lo & 0xffffu) - 32768;
                    int iy      = (int)((key_lo >> 16) & 0xffffu) - 32768;
                    int iz      = unwrap_z(key_hi, rc, plvl);
                    float sp    = rc.spacing * (float)(1 << plvl);
                    Vector3 pos = { ((float)ix + 0.5f) * sp, ((float)iy + 0.5f) * sp, ((float)iz + 0.5f) * sp };
                    float dx    = pos.x - st.pan_x;
                    float dz    = pos.z - st.pan_z;
                    if (dx * dx + dz * dz > max_d2) continue;
                    drawn++;
                    uint slot    = bitcast(raw.z, uint);
                    uint last_f  = bitcast(raw.w, uint);
                    float[4] plc = LEVEL_COLORS[math::min(plvl, 4)];
                    float[3] rgb = { plc[0] * 0.5f, plc[1] * 0.5f, plc[2] * 0.5f };
                    float alpha  = 0.85f;
                    if (st.color_by_age) {
                        bool solid = plvl == 0 && (int)slot < rc.level_cap[0]
                            && st.irr_cpu[(sz)rc.hash_cap + (sz)slot * 6].w < 0.5f;
                        rgb = solid ? (float[3]) { 0.5f, 0.1f, 0.1f } : age_color(last_f, rc.frame);
                        if (solid) alpha = 0.35f;
                    } else if (plvl == 0 && (int)slot < rc.level_cap[0]) {
                        sz base = (sz)rc.hash_cap + (sz)slot * 6;
                        if (st.irr_cpu[base].w < 0.5f) {
                            rgb = { 0.5f, 0.1f, 0.1f };
                            alpha = 0.35f;
                        } else {
                            Vector3 avg;
                            for (int a = 0; a < 6; a++) {
                                drie::Vec4 e = st.irr_cpu[base + a];
                                avg += { e.x, e.y, e.z };
                            }
                            rgb = tone_rgb(avg * (1.0f / 6.0f));
                        }
                    }
                    sgl_cube_quads(pos, sp * 0.16f, rgb, alpha);
                }
                sgl::end();

                // 3D radiating angular lines on sparse cubes
                if (st.show_angles && st.rad_ready) {
                    int ray_probes = 0;
                    sgl::begin_lines();
                    for (int h = 0; h < rc.hash_cap && ray_probes < 1200; h++) {
                        drie::Vec4 raw = st.irr_cpu[h];
                        uint key_lo    = bitcast(raw.x, uint);
                        uint key_hi    = bitcast(raw.y, uint);
                        if ((key_hi & 0x80000000u) == 0) continue;
                        int plvl = (int)((key_hi >> 27) & 0x7u);
                        if (st.level >= 0 && plvl != active_lvl) continue;
                        int ix      = (int)(key_lo & 0xffffu) - 32768;
                        int iy      = (int)((key_lo >> 16) & 0xffffu) - 32768;
                        int iz      = unwrap_z(key_hi, rc, plvl);
                        float sp    = rc.spacing * (float)(1 << plvl);
                        Vector3 pos = { ((float)ix + 0.5f) * sp, ((float)iy + 0.5f) * sp, ((float)iz + 0.5f) * sp };
                        float dx    = pos.x - st.pan_x;
                        float dz    = pos.z - st.pan_z;
                        if (dx * dx + dz * dz > max_d2) continue;
                        uint slot = bitcast(raw.z, uint);
                        if ((int)slot >= rc.level_cap[plvl]) continue;
                        ray_probes++;
                        int pd        = rc.base_directions << plvl;
                        int pdd       = pd * pd;
                        int poff      = rc.offsets[plvl];
                        int k_step    = pd > 8 ? 2 : 1;
                        float ray_len = sp * 0.72f;
                        for (int ky = 0; ky < pd; ky += k_step) {
                            for (int kx = 0; kx < pd; kx += k_step) {
                                int k        = ky * pd + kx;
                                float ex     = ((float)kx + 0.5f) / (float)pd * 2.0f - 1.0f;
                                float ey     = ((float)ky + 0.5f) / (float)pd * 2.0f - 1.0f;
                                Vector3 rd   = oct_dir(ex, ey);
                                Vector3 rad  = unpack_rad((sz)poff + (sz)slot * pdd + k);
                                float lum    = rad.x * 0.2126f + rad.y * 0.7152f + rad.z * 0.0722f;
                                float len    = ray_len * (0.35f + 0.65f * drie::clamp(
                                    lum * 2.5f / (0.25f + lum * 2.5f),
                                    0,
                                    1
                                ));
                                float[3] rgb = tone_rgb(rad * 3.0f);
                                sgl::c4f(
                                    rgb[0],
                                    rgb[1],
                                    rgb[2],
                                    0.35f + 0.55f * drie::clamp(lum * 3.0f / (0.2f + lum * 3.0f), 0, 1)
                                );
                                sgl::v3f(pos.x, pos.y, pos.z);
                                sgl::v3f(pos.x + rd.x * len, pos.y + rd.y * len, pos.z + rd.z * len);
                            }
                        }
                    }
                    sgl::end();
                }
            }
        } else {
            // Dense 3D probe cubes (strided in XZ/Y when all_slices_3d is on to stay within vertex budget)
            int[<3>] dims = rc.level_dims(active_lvl);
            int y0        = st.all_slices_3d ? 0 : math::clamp(slice0 >> active_lvl, 0, dims.y - 1);
            int y1        = st.all_slices_3d ? dims.y : y0 + 1;
            int xz_step   = st.all_slices_3d && (dims.x * dims.y * dims.z > 4000) ? 2 : 1;
            int y_step    = st.all_slices_3d && dims.y > 8 ? 2 : 1;

            sgl::begin_quads();
            for (int iy = y0; iy < y1; iy += y_step) {
                float wy = rc.min.y + ((float)iy + 0.5f) * spacing;
                for (int iz = 0; iz < dims.z; iz += xz_step) {
                    float wz = rc.min.z + ((float)iz + 0.5f) * spacing;
                    for (int ix = 0; ix < dims.x; ix += xz_step) {
                        float wx = rc.min.x + ((float)ix + 0.5f) * spacing;
                        // skip solid interior cubes so the 3D view stays clean
                        int vx   = (int)math::floor((wx - rc.min.x) / rc.voxel_size);
                        int vy   = (int)math::floor((wy - rc.min.y) / rc.voxel_size);
                        int vz   = (int)math::floor((wz - rc.min.z) / rc.voxel_size);
                        if (vx >= 0 && vx < rc.vx && vy >= 0 && vy < rc.vy && vz >= 0 && vz < rc.vz) {
                            if ((rc.voxels[(((sz)vz * rc.vy + vy) * rc.vx + vx) * 2] & 0x1000000u) != 0) continue;
                        }
                        float[3] rgb = { lcol[0] * 0.5f, lcol[1] * 0.5f, lcol[2] * 0.5f };
                        if (active_lvl == 0 && st.irr_ready) {
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
                            int cnt  = 0;
                            for (int k = 0; k < dd; k += step) {
                                avg += unpack_rad((sz)off + p_idx * dd + k);
                                cnt++;
                            }
                            if (cnt > 0) rgb = tone_rgb(avg * (1.0f / (float)cnt));
                        }
                        sgl_cube_quads({ wx, wy, wz }, cube_r, rgb, 0.82f);
                    }
                }
            }
            sgl::end();

            // 3D radiating angular lines on all drawn dense cubes
            if (st.show_angles && st.rad_ready) {
                int k_step     = d > 8 ? 2 : 1;
                float ray_len  = spacing * 0.72f;
                int ray_probes = 0;
                sgl::begin_lines();
                for (int iy = y0; iy < y1 && ray_probes < 1600; iy += y_step) {
                    float wy = rc.min.y + ((float)iy + 0.5f) * spacing;
                    for (int iz = 0; iz < dims.z && ray_probes < 1600; iz += xz_step) {
                        float wz = rc.min.z + ((float)iz + 0.5f) * spacing;
                        for (int ix = 0; ix < dims.x && ray_probes < 1600; ix += xz_step) {
                            float wx = rc.min.x + ((float)ix + 0.5f) * spacing;
                            int vx   = (int)math::floor((wx - rc.min.x) / rc.voxel_size);
                            int vy   = (int)math::floor((wy - rc.min.y) / rc.voxel_size);
                            int vz   = (int)math::floor((wz - rc.min.z) / rc.voxel_size);
                            if (vx >= 0 && vx < rc.vx && vy >= 0 && vy < rc.vy && vz >= 0 && vz < rc.vz) {
                                if ((rc.voxels[(((sz)vz * rc.vy + vy) * rc.vx + vx) * 2] & 0x1000000u) != 0) continue;
                            }
                            ray_probes++;
                            sz p_idx = ((sz)iz * dims.y + iy) * dims.x + ix;
                            for (int ky = 0; ky < d; ky += k_step) {
                                for (int kx = 0; kx < d; kx += k_step) {
                                    int k        = ky * d + kx;
                                    float ex     = ((float)kx + 0.5f) / (float)d * 2.0f - 1.0f;
                                    float ey     = ((float)ky + 0.5f) / (float)d * 2.0f - 1.0f;
                                    Vector3 rd   = oct_dir(ex, ey);
                                    Vector3 rad  = unpack_rad((sz)off + p_idx * dd + k);
                                    float lum    = rad.x * 0.2126f + rad.y * 0.7152f + rad.z * 0.0722f;
                                    float len    = ray_len * (0.35f + 0.65f * drie::clamp(
                                        lum * 2.5f / (0.25f + lum * 2.5f),
                                        0,
                                        1
                                    ));
                                    float[3] rgb = tone_rgb(rad * 3.0f);
                                    sgl::c4f(
                                        rgb[0],
                                        rgb[1],
                                        rgb[2],
                                        0.35f + 0.55f * drie::clamp(lum * 3.0f / (0.2f + lum * 3.0f), 0, 1)
                                    );
                                    sgl::v3f(wx, wy, wz);
                                    sgl::v3f(wx + rd.x * len, wy + rd.y * len, wz + rd.z * len);
                                }
                            }
                        }
                    }
                }
                sgl::end();
            }

            // 3D angular rays on the inspected (center) probe
            int cx = math::clamp((int)math::floor((st.pan_x - rc.min.x) / spacing), 0, dims.x - 1);
            int cy = math::clamp(slice0 >> active_lvl, 0, dims.y - 1);
            int cz = math::clamp((int)math::floor((st.pan_z - rc.min.z) / spacing), 0, dims.z - 1);
            Vector3 cpos = {
                rc.min.x + ((float)cx + 0.5f) * spacing,
                rc.min.y + ((float)cy + 0.5f) * spacing,
                rc.min.z + ((float)cz + 0.5f) * spacing,
            };
            sgl_wire_box_3d(
                cpos - (Vector3) { spacing * 0.5f, spacing * 0.5f, spacing * 0.5f },
                cpos + (Vector3) { spacing * 0.5f, spacing * 0.5f, spacing * 0.5f },
                { 1, 1, 1, 0.95f }
            );
            if (st.show_angles && st.rad_ready) {
                sz p_idx        = ((sz)cz * dims.y + cy) * dims.x + cx;
                float t1        = rc.interval * (math::pow(4.0f, (float)(active_lvl + 1)) - 1.0f) / 3.0f;
                float ray_scale = math::min(t1, spacing * 3.0f);
                sgl::begin_lines();
                for (int ky = 0; ky < d; ky++) {
                    for (int kx = 0; kx < d; kx++) {
                        int k        = ky * d + kx;
                        float ex     = ((float)kx + 0.5f) / (float)d * 2.0f - 1.0f;
                        float ey     = ((float)ky + 0.5f) / (float)d * 2.0f - 1.0f;
                        Vector3 rd   = oct_dir(ex, ey);
                        Vector3 rad  = unpack_rad((sz)off + p_idx * dd + k);
                        float[3] rgb = tone_rgb(rad * 2.5f);
                        sgl::c4f(rgb[0], rgb[1], rgb[2], 0.9f);
                        sgl::v3f(cpos.x, cpos.y, cpos.z);
                        sgl::v3f(cpos.x + rd.x * ray_scale, cpos.y + rd.y * ray_scale, cpos.z + rd.z * ray_scale);
                    }
                }
                sgl::end();
            }
        }
    }

    // 3D yellow player column + look arrow
    if (marker_dir != (Vector3) { }) {
        sgl_wire_box_3d(
            { marker.x - 0.18f, marker.y, marker.z - 0.18f },
            { marker.x + 0.18f, marker.y + 1.65f, marker.z + 0.18f },
            { 1.0f, 0.92f, 0.2f, 0.95f }
        );
        sgl::begin_lines();
        sgl::c4f(1.0f, 0.95f, 0.25f, 0.95f);
        sgl::v3f(marker.x, marker.y + 1.4f, marker.z);
        sgl::v3f(marker.x + marker_dir.x * 1.2f, marker.y + 1.4f, marker.z + marker_dir.z * 1.2f);
        sgl::end();
    }
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
    if (rc != st.last_rc) {
        st.irr_ready = false;
        st.rad_ready = false;
    }
    sgl::set_context(gl);
    sgl::defaults();
    sgl::load_pipeline(pip);

    float px_per_m = (float)height / (2.0f * st.half_h);
    int slice0     = rc ? active_slice(rc) : 0;
    int active_lvl = rc ? (st.level < 0 ? 0 : math::clamp(st.level, 0, rc.levels - 1)) : 0;
    int cx         = 0;
    int cy         = 0;
    int cz         = 0;

    if (st.view_3d) {
        draw_3d_scene_overlay(rc, active_lvl, slice0, marker, marker_dir);
    }

    sgl::matrix_mode_projection();
    sgl::load_identity();
    sgl::ortho(0, (float)width, (float)height, 0, -1, 1);
    sgl::matrix_mode_modelview();
    sgl::load_identity();

    if (rc) {
        int[<3>] dims = rc.level_dims(active_lvl);
        float spacing = rc.spacing * (float)(1 << active_lvl);
        cx = math::clamp((int)math::floor((st.pan_x - rc.min.x) / spacing), 0, dims.x - 1);
        cy = math::clamp(slice0 >> active_lvl, 0, dims.y - 1);
        cz = math::clamp((int)math::floor((st.pan_z - rc.min.z) / spacing), 0, dims.z - 1);

        if (!st.view_3d) {
            float[2] b0 = world_to_px(rc.min.x, rc.min.z, width, height);
            float[2] b1 = world_to_px(rc.max.x, rc.max.z, width, height);
            sgl_box(b0[0], b0[1], b1[0], b1[1], { 0.4f, 0.85f, 1.0f, 0.8f });

            if (st.show_voxels) {
                int vy    = math::clamp(slice0 * 4 + 2, 0, rc.vy - 1);
                float vs  = rc.voxel_size;
                float vpx = math::max(vs * px_per_m, 1.0f);
                sgl::begin_quads();
                sgl::c4f(0.85f, 0.85f, 0.9f, 0.24f);
                for (int vz = 0; vz < rc.vz; vz++) {
                    float wz = rc.min.z + (float)vz * vs;
                    for (int vx = 0; vx < rc.vx; vx++) {
                        if ((rc.voxels[(((sz)vz * rc.vy + vy) * rc.vx + vx) * 2] & 0x1000000u) == 0) continue;
                        float wx    = rc.min.x + (float)vx * vs;
                        float[2] sp = world_to_px(wx, wz, width, height);
                        if (sp[0] < -vpx || sp[0] > (float)width + vpx || sp[1] < -vpx
                            || sp[1] > (float)height + vpx) continue;
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

            float cwx     = rc.min.x + ((float)cx + 0.5f) * spacing;
            float cwz     = rc.min.z + ((float)cz + 0.5f) * spacing;
            float[2] csp  = world_to_px(cwx, cwz, width, height);
            float t0      = rc.interval * (math::pow(4.0f, (float)active_lvl) - 1.0f) / 3.0f;
            float t1      = rc.interval * (math::pow(4.0f, (float)(active_lvl + 1)) - 1.0f) / 3.0f;
            float[4] lcol = LEVEL_COLORS[math::min(active_lvl, 4)];
            if (t0 > 0.001f) sgl_circle(csp[0], csp[1], t0 * px_per_m, 48, { lcol[0], lcol[1], lcol[2], 0.45f });
            sgl_circle(csp[0], csp[1], t1 * px_per_m, 64, { lcol[0], lcol[1], lcol[2], 0.85f });
            sgl_box(csp[0] - 8, csp[1] - 8, csp[0] + 8, csp[1] + 8, { 1, 1, 1, 0.95f });
        }

        if (st.show_angles && !rc.sparse) draw_probe_rose(rc, active_lvl, cx, cy, cz, width);
    }

    // Optional player/camera marker (2D top-down mode)
    if (!st.view_3d && marker_dir != (Vector3) { }) {
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
        sdtx::puts(
            string::tformat(
                "RC DEBUG  [%s | VHS OFF]  VOL: %s",
                st.view_3d ? "3D ORBIT" : "TOP-DOWN",
                vol_name.len ? vol_name : "NONE"
            )
            .zstr_tcopy());
        row += 1.6f;

        sdtx::color3f(0.88f, 0.90f, 0.94f);
        if (rc) {
            if (rc.sparse) {
                int occupied = 0;
                if (st.irr_ready) {
                    for (int h = 0; h < rc.hash_cap; h++) {
                        if ((bitcast(st.irr_cpu[h].y, uint) & 0x80000000u) != 0) occupied++;
                    }
                }
                int total_rays = rc.offsets[rc.levels - 1] + rc.counts[rc.levels - 1];
                float rad_mb   = (float)((double)total_rays * 8.0 / (1024.0 * 1024.0));
                float irr_mb   = (float)((double)(rc.hash_cap + rc.level_cap[0] * 6) * 16.0 / (1024.0 * 1024.0));
                sdtx::pos(2.0f, row);
                sdtx::puts(
                    string::tformat(
                        "SPARSE: %d/%d hash (%.0f%%)  L0cap=%d  VOX: %dx%dx%d (%.3fm)",
                        occupied,
                        rc.hash_cap,
                        100.0f * (float)occupied / (float)rc.hash_cap,
                        rc.level_cap[0],
                        rc.vx,
                        rc.vy,
                        rc.vz,
                        rc.voxel_size
                    )
                    .zstr_tcopy());
                row += 1.4f;

                sdtx::pos(2.0f, row);
                sdtx::puts(
                    string::tformat(
                        "BOX: (%.1f,%.1f,%.1f)..(%.1f,%.1f,%.1f)  LVL:%d  DIR:%d",
                        rc.min.x,
                        rc.min.y,
                        rc.min.z,
                        rc.max.x,
                        rc.max.y,
                        rc.max.z,
                        rc.levels,
                        rc.base_directions
                    )
                    .zstr_tcopy());
                row += 1.4f;

                sdtx::pos(2.0f, row);
                sdtx::puts(
                    string::tformat(
                        "CFG: every=%d  blend=%.2f  alloc=%s  color=%s  VRAM=%.1fMB",
                        rc.update_every,
                        rc.blend,
                        freeze_alloc ? "PLAYER" : "DEBUG",
                        st.color_by_age ? "AGE" : "RAD",
                        rad_mb + irr_mb
                    )
                    .zstr_tcopy());
                row += 1.6f;
            } else {
                int total_probes = rc.nx * rc.ny * rc.nz;
                int total_rays   = rc.offsets[rc.levels - 1] + rc.counts[rc.levels - 1];
                float rad_mb     = (float)((double)total_rays * 8.0 / (1024.0 * 1024.0));
                float irr_mb     = (float)((double)total_probes * 6.0 * 16.0 / (1024.0 * 1024.0));
                sdtx::pos(2.0f, row);
                sdtx::puts(
                    string::tformat(
                        "GRID: %dx%dx%d (%d probes)  VOX: %dx%dx%d (%.3fm)",
                        rc.nx,
                        rc.ny,
                        rc.nz,
                        total_probes,
                        rc.vx,
                        rc.vy,
                        rc.vz,
                        rc.voxel_size
                    )
                    .zstr_tcopy());
                row += 1.4f;

                sdtx::pos(2.0f, row);
                sdtx::puts(
                    string::tformat(
                        "BOX: (%.1f,%.1f,%.1f)..(%.1f,%.1f,%.1f)  LVL:%d  DIR:%d",
                        rc.min.x,
                        rc.min.y,
                        rc.min.z,
                        rc.max.x,
                        rc.max.y,
                        rc.max.z,
                        rc.levels,
                        rc.base_directions
                    )
                    .zstr_tcopy());
                row += 1.4f;

                sdtx::pos(2.0f, row);
                sdtx::puts(
                    string::tformat(
                        "CFG: every=%d  blend=%.2f  VRAM=%.1fMB",
                        rc.update_every,
                        rc.blend,
                        rad_mb + irr_mb
                    )
                    .zstr_tcopy());
                row += 1.6f;
            }

            int[<3>] ld     = rc.level_dims(active_lvl);
            float l_spacing = rc.spacing * (float)(1 << active_lvl);
            int d           = rc.base_directions << active_lvl;
            float t0        = rc.interval * (math::pow(4.0f, (float)active_lvl) - 1.0f) / 3.0f;
            float t1        = rc.interval * (math::pow(4.0f, (float)(active_lvl + 1)) - 1.0f) / 3.0f;
            float[4] lc     = LEVEL_COLORS[math::min(active_lvl, 4)];
            sdtx::color3f(lc[0], lc[1], lc[2]);
            sdtx::pos(2.0f, row);
            String ltag = st.level < 0 ? string::tformat("ALL (ring L%d)", active_lvl)
                : string::tformat("L%d", active_lvl);
            sdtx::puts(
                string::tformat(
                    "CASCADE %s: %dx%dx%d  sp=%.2fm  ang=%dx%d (%d)  [%.2f..%.2fm]",
                    ltag,
                    ld.x,
                    ld.y,
                    ld.z,
                    l_spacing,
                    d,
                    d,
                    d * d,
                    t0,
                    t1
                )
                .zstr_tcopy());
            row += 1.6f;

            float slice_world_y = rc.min.y + ((float)slice0 + 0.5f) * rc.spacing;
            sdtx::color3f(0.85f, 0.88f, 0.92f);
            sdtx::pos(2.0f, row);
            sdtx::puts(
                string::tformat(
                    "VIEW: pan=(%.2f,%.2f)  zoom=%.1fm  cutY=%.2fm  slice=%d/%d (y=%.2fm)",
                    st.pan_x,
                    st.pan_z,
                    st.half_h,
                    st.cut_y,
                    slice0,
                    rc.ny - 1,
                    slice_world_y
                )
                .zstr_tcopy());
            row += 1.4f;

            if (!rc.sparse && active_lvl == 0 && st.irr_ready) {
                sz p_idx      = ((sz)cz * ld.y + cy) * ld.x + cx;
                drie::Vec4 e0 = st.irr_cpu[p_idx * 6 + 0];
                bool valid    = e0.w > 0.5f;
                Vector3 avg;
                for (int a = 0; a < 6; a++) {
                    drie::Vec4 e = st.irr_cpu[p_idx * 6 + a];
                    avg += { e.x, e.y, e.z };
                }
                avg *= (1.0f / 6.0f);
                sdtx::color3f(valid ? 0.55f : 1.0f, valid ? 1.0f : 0.45f, valid ? 0.65f : 0.45f);
                sdtx::pos(2.0f, row);
                sdtx::puts(
                    string::tformat(
                        "PROBE (%d,%d,%d) %s  irr=(%.2f, %.2f, %.2f)  angles=%s  vox=%s",
                        cx,
                        cy,
                        cz,
                        valid ? "VALID" : "SOLID",
                        avg.x,
                        avg.y,
                        avg.z,
                        st.show_angles ? "ON" : "OFF",
                        st.show_voxels ? "ON" : "OFF"
                    )
                    .zstr_tcopy());
            } else {
                sdtx::pos(2.0f, row);
                sdtx::puts(
                    string::tformat(
                        "PROBE (%d,%d,%d)  angles=%s  vox=%s  readback=%s",
                        cx,
                        cy,
                        cz,
                        st.show_angles ? "ON" : "OFF",
                        st.show_voxels ? "ON" : "OFF",
                        st.rad_ready ? "LIVE" : "WAIT"
                    )
                    .zstr_tcopy());
            }
            row += 1.6f;

            if (rc.sparse && st.color_by_age) {
                sdtx::pos(2.0f, row);
                sdtx::color3f(0.15f, 1.00f, 0.55f);
                sdtx::puts("NEW ");
                sdtx::color3f(0.95f, 0.90f, 0.20f);
                sdtx::puts("RECENT ");
                sdtx::color3f(1.00f, 0.45f, 0.15f);
                sdtx::puts("AGING ");
                sdtx::color3f(0.45f, 0.35f, 0.90f);
                sdtx::puts("OLD  ");
                sdtx::color3f(1.00f, 0.92f, 0.20f);
                sdtx::puts("[PLAYER]  ");
                sdtx::color3f(0.95f, 0.35f, 0.35f);
                sdtx::puts("[SOLID]");
            } else {
                sdtx::pos(2.0f, row);
                sdtx::color3f(0.15f, 0.85f, 1.0f);
                sdtx::puts("L0 ");
                sdtx::color3f(0.25f, 1.0f, 0.45f);
                sdtx::puts("L1 ");
                sdtx::color3f(1.0f, 0.78f, 0.2f);
                sdtx::puts("L2 ");
                sdtx::color3f(1.0f, 0.35f, 0.85f);
                sdtx::puts("L3  ");
                sdtx::color3f(1.0f, 0.92f, 0.2f);
                sdtx::puts("[PLAYER]  ");
                sdtx::color3f(0.95f, 0.35f, 0.35f);
                sdtx::puts("[SOLID]");
            }

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
            sdtx::puts(
                string::tformat(
                    "VIEW: pan=(%.2f,%.2f)  zoom=%.1fm  cutY=%.2fm",
                    st.pan_x,
                    st.pan_z,
                    st.half_h,
                    st.cut_y
                )
                .zstr_tcopy());
        }

        sdtx::color3f(0.72f, 0.78f, 0.85f);
        sdtx::pos(2.0f, ((float)height - 22.0f) / 8.0f);
        sdtx::puts("ARROWS:pan M:2D/3D DRAG/QE:orbit +/-:zoom [/]:cutY 0-3/4:lvl R:ang C:age/rad P:alloc F3:exit");
    };

    sgl::context_draw(gl);
    sdtx::context_draw(text);
}

fn SgPipelineDesc blend_desc(
) @local => {
    .colors[0].write_mask = RGBA,
    .colors[0].blend = {
        .enabled = true,
        .src_factor_rgb = SRC_ALPHA,
        .dst_factor_rgb = ONE_MINUS_SRC_ALPHA,
        .src_factor_alpha = ONE,
        .dst_factor_alpha = ONE_MINUS_SRC_ALPHA,
    },
};

fn void draw_window(
    RadianceCascades* rc,
    String vol_name,
    int width,
    int height,
    Vector3 marker = { },
    Vector3 marker_dir = { }
) {
    if (!active || !sapp::isvalid()) return;
    if (!st.win_gl.id) {
        ui::setup();
        st.win_gl = sgl::make_context(&&{ .max_vertices = 524288, .max_commands = 64 });
        st.win_pip = sgl::context_make_pipeline(st.win_gl, &&blend_desc());
        st.win_text = sdtx::make_context(
            &&{ .char_buf_size = 4096, .canvas_width = (float)width, .canvas_height = (float)height }
        );
    }
    SlbxViewport vp = slbx::letterbox(
        sapp::width(),
        sapp::height(),
        &&{ .content_aspect_ratio = (float)width / (float)height }
    );
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
    Vector3 marker = { },
    Vector3 marker_dir = { }
) {
    if (!active || !target) return;
    if (!st.img_gl.id) {
        ui::setup();
        st.img_gl = sgl::make_context(
            &&{
                .max_vertices = 524288,
                .max_commands = 64,
                .color_format = RGBA8,
                .depth_format = NONE,
                .sample_count = 1,
            }
        );
        st.img_pip = sgl::context_make_pipeline(st.img_gl, &&blend_desc());
        st.img_text = sdtx::make_context(
            &&{
                .char_buf_size = 4096,
                .canvas_width = (float)width,
                .canvas_height = (float)height,
                .color_format = RGBA8,
                .depth_format = NONE,
                .sample_count = 1,
            }
        );
    }
    target.begin(LOAD, label: "drie-rc-debug");
    draw_content(st.img_gl, st.img_pip, st.img_text, rc, vol_name, width, height, marker, marker_dir);
    target.end();
}

// ---------------------------------------------------------------------------------------------
// stats(): what the sparse probes are doing, read back from the GPU (a stall: for tests and
// benches, or a key press, not every frame).

const int STATS_HT_SLOT       = 12;
const int STATS_CNT_SLOT      = 13;
const int STATS_FREE_SLOT     = 14;
const int STATS_META_SLOT     = 15;
const int STATS_HIST_SLOT     = 16;
// as in radiance_split_cs.glsl
const uint STATS_TOMBSTONE    = 0x7fffffffu;
const uint STATS_STALE_FRAMES = 24u;

struct RcLevelStats {
    int cap; // data slots the level has (level_cap)
    int allocated; // slots handed out so far (rc_cnt: the bump allocator's top)
    int free_listed; // of those, slots evicted and waiting on the free list for reuse
    int live; // slots holding a probe (rc_meta w >= 0)
    int fresh; // live probes not yet traced in every ray (born in the last update_every frames)
    int in_table; // hash entries of the level
    int seen; // hash entries touched in the last frame
    int stale; // hash entries older than RC_STALE_FRAMES: the next evict pass frees them
    int bins_total; // direction bins of live probes in this level (split mode)
    int bins_filled; // direction bins with sample count n > 0 (split mode)
}

struct RcStats {
    bool sparse;
    bool split;
    int frame; // updates run
    int levels;
    // dense: probes per cascade 0 (nx * ny * nz); sparse: live probes of every level
    int probes;
    int rays; // ray slots of every level (live probes' rays, for sparse)
    int traced_rays; // rays actually traced in the last frame (split mode)
    int bins_total; // total direction bins across live probes (split mode)
    int bins_filled; // total direction bins with n > 0 across live probes (split mode)
    int cache_probes; // live probes of the secondary cache (split mode, multibounce)
    int cache_traced_rays; // cache rays actually traced in the last frame
    int cache_bins_total; // direction bins of the cache's live probes
    int cache_bins_filled; // of those, bins with n > 0
    float min_beta; // min transmittance beta across filled bins (split mode)
    float max_beta; // max transmittance beta across filled bins (split mode)
    // the sparse path's hash table (irradiance_buf's front)
    int hash_cap;
    int occupied;
    int tombstones;
    float load; // occupied / hash_cap
    float load_tombstones; // (occupied + tombstones) / hash_cap: what lookups that miss walk
    int longest_run; // longest run of non-empty slots (lookups give up after 32 steps)
    int runs_over_32; // runs longer than 32: keys past their 32nd step are unreachable
    int stale; // entries the next evict pass frees
    int frames_to_evict; // updates until that pass (RC_EVICT_PERIOD)
    RcLevelStats[drie::RC_MAX_LEVELS] level;
}

fn uint stats_age(uint now, uint then) @local => now >= then ? now - then : now + 4096u - then;

fn bool stats_read(StorageBuffer* b, CInt slot, void* out, sz bytes) @local {
    if (!readback::track(b.buffer, slot)) return false;
    return readback::fetch(slot, out, (CInt)bytes, true);
}

// What `rc`'s probes are doing now. Waits for the GPU: call it after drie::commit(), outside a
// frame being encoded. A dense volume gives only its sizes.
fn RcStats stats(RadianceCascades* rc) {
    RcStats s;
    if (!rc) return s;
    s.sparse = rc.sparse;
    s.split = rc.split;
    s.frame = rc.frame;
    s.levels = rc.sparse && rc.split ? math::max(rc.split_levels, 1) : rc.levels;
    if (!rc.sparse) {
        s.probes = rc.nx * rc.ny * rc.nz;
        s.rays = rc.offsets[rc.levels - 1] + rc.counts[rc.levels - 1];
        return s;
    }
    s.hash_cap = rc.hash_cap;
    s.frames_to_evict = rc.split ? 0
        : (drie::RC_EVICT_PERIOD - rc.frame % drie::RC_EVICT_PERIOD) % drie::RC_EVICT_PERIOD;
    if (!rc.irradiance_buf) return s;
    uint now = (uint)((rc.frame - 1) % 4096) + 1u;

    if (rc.split) {
        int eff          = s.levels;
        int total_meta   = rc.split_pbase[eff - 1] + rc.split_cap[eff - 1];
        int total_bins   = rc.split_offsets[eff - 1] + rc.split_counts[eff - 1];
        int all_meta     = rc.cache_pbase[eff - 1] + rc.cache_cap[eff - 1];
        int all_bins     = rc.cache_offsets[eff - 1] + rc.cache_counts[eff - 1];
        uint[] free_list = mem::new_array(uint, (sz)(drie::RC_FREE_BASE + total_meta));
        defer free(free_list);
        drie::Vec4[] ht = mem::new_array(drie::Vec4, (sz)rc.hash_cap);
        defer free(ht);
        uint[<4>][] meta = mem::new_array(uint[<4>], (sz)all_meta * 2);
        defer free(meta);
        uint[] hist = mem::new_array(uint, (sz)all_bins * 2);
        defer free(hist);
        if (!stats_read(rc.free_buf, STATS_FREE_SLOT, free_list.ptr, free_list.len * uint::size)) return s;
        if (!stats_read(rc.irradiance_buf, STATS_HT_SLOT, ht.ptr, ht.len * drie::Vec4::size)) return s;
        if (!stats_read(rc.meta_buf, STATS_META_SLOT, meta.ptr, meta.len * uint[<4>]::size)) return s;
        if (rc.hist_buf) (void)stats_read(rc.hist_buf, STATS_HIST_SLOT, hist.ptr, hist.len * uint::size);

        s.traced_rays = (int)free_list[17];
        s.cache_traced_rays = (int)free_list[36];
        s.min_beta = 1.0f;
        s.max_beta = 0.0f;
        bool any_beta;
        int run;
        for (int h = 0; h < rc.hash_cap; h++) {
            drie::Vec4 e = ht[h];
            uint key     = bitcast(e.y, uint);
            if (key == 0) {
                if (run > 32) s.runs_over_32++;
                s.longest_run = math::max(s.longest_run, run);
                run = 0;
                continue;
            }
            run++;
            s.occupied++;
            // the secondary cache's probes share the table: they count towards its load,
            // not towards a primary level
            if ((key & 0x40000000u) != 0) continue;
            int lvl = (int)((key >> 27) & 7u);
            if (lvl >= eff) continue;
            RcLevelStats* l = &s.level[lvl];
            l.in_table++;
            uint age = stats_age(now, bitcast(e.w, uint));
            if (age == 0) l.seen++;
            if (age > STATS_STALE_FRAMES) {
                l.stale++;
                s.stale++;
            }
        }
        if (run > 32) s.runs_over_32++;
        s.longest_run = math::max(s.longest_run, run);
        s.load = (float)s.occupied / (float)rc.hash_cap;
        s.load_tombstones = s.load;

        for (int lvl = 0; lvl < eff; lvl++) {
            RcLevelStats* l = &s.level[lvl];
            l.cap = rc.split_cap[lvl];
            l.allocated = math::min((int)free_list[8 + lvl], l.cap);
            l.free_listed = math::min((int)free_list[lvl], l.cap);
            int th   = rc.base_directions << lvl;
            int dirs = 2 * th * th;
            for (int i = 0; i < l.allocated; i++) {
                int mi      = rc.split_pbase[lvl] + i;
                uint[<4>] a = meta[mi * 2];
                if (a.y == 0) continue;
                l.live++;
                uint birth = a.w >> 16;
                if (stats_age(now, birth) < 4) l.fresh++;
                int boff = rc.split_offsets[lvl] + i * dirs;
                l.bins_total += dirs;
                for (int k = 0; k < dirs; k++) {
                    uint py = hist[(boff + k) * 2 + 1];
                    uint n  = py >> 24;
                    if (n > 0) {
                        l.bins_filled++;
                        float beta = (float)((py >> 16) & 0xffu) / 255.0f;
                        s.min_beta = math::min(s.min_beta, beta);
                        s.max_beta = math::max(s.max_beta, beta);
                        any_beta = true;
                    }
                }
            }
            s.probes += l.live;
            s.rays += l.live * dirs;
            s.bins_total += l.bins_total;
            s.bins_filled += l.bins_filled;
        }
        for (int lvl = 0; lvl < eff; lvl++) {
            int cap       = rc.cache_cap[lvl];
            int allocated = math::min((int)free_list[27 + lvl], cap);
            int dirs      = 2 * (rc.base_directions << lvl) * (rc.base_directions << lvl);
            for (int i = 0; i < allocated; i++) {
                int mi = rc.cache_pbase[lvl] + i;
                if (meta[mi * 2].y == 0) continue;
                s.cache_probes++;
                int boff = rc.cache_offsets[lvl] + i * dirs;
                s.cache_bins_total += dirs;
                for (int k = 0; k < dirs; k++) {
                    if ((hist[(boff + k) * 2 + 1] >> 24) > 0) s.cache_bins_filled++;
                }
            }
        }
        if (!any_beta) s.min_beta = 0.0f;
        return s;
    }

    int total_meta   = rc.probe_base[rc.levels - 1] + rc.level_cap[rc.levels - 1];
    uint[8] cnt;
    uint[] free_list = mem::new_array(uint, (sz)rc.free_buf_len);
    defer free(free_list);
    drie::Vec4[] ht = mem::new_array(drie::Vec4, (sz)rc.hash_cap + (sz)rc.level_cap[0] * 6);
    defer free(ht);
    int[<4>][] meta = mem::new_array(int[<4>], (sz)total_meta);
    defer free(meta);
    if (!stats_read(rc.count_buf, STATS_CNT_SLOT, &cnt, uint::size * 8)) return s;
    if (!stats_read(rc.free_buf, STATS_FREE_SLOT, free_list.ptr, free_list.len * uint::size)) return s;
    if (!stats_read(rc.irradiance_buf, STATS_HT_SLOT, ht.ptr, ht.len * drie::Vec4::size)) return s;
    if (!stats_read(rc.meta_buf, STATS_META_SLOT, meta.ptr, meta.len * int[<4>]::size)) return s;

    int run;
    for (int h = 0; h < rc.hash_cap; h++) {
        drie::Vec4 e = ht[h];
        uint key     = bitcast(e.y, uint);
        if (key == 0) {
            if (run > 32) s.runs_over_32++;
            s.longest_run = math::max(s.longest_run, run);
            run = 0;
            continue;
        }
        run++;
        if (key == STATS_TOMBSTONE) {
            s.tombstones++;
            continue;
        }
        s.occupied++;
        int lvl = (int)((key >> 27) & 7u);
        if (lvl >= rc.levels) continue;
        RcLevelStats* l = &s.level[lvl];
        l.in_table++;
        uint age = stats_age(now, bitcast(e.w, uint));
        if (age == 0) l.seen++;
        if (age > STATS_STALE_FRAMES) {
            l.stale++;
            s.stale++;
        }
    }
    if (run > 32) s.runs_over_32++;
    s.longest_run = math::max(s.longest_run, run);
    s.load = (float)s.occupied / (float)rc.hash_cap;
    s.load_tombstones = (float)(s.occupied + s.tombstones) / (float)rc.hash_cap;

    for (int lvl = 0; lvl < rc.levels; lvl++) {
        RcLevelStats* l = &s.level[lvl];
        l.cap = rc.level_cap[lvl];
        l.allocated = math::min((int)cnt[lvl], l.cap);
        l.free_listed = math::min((int)free_list[lvl], l.cap);
        int d = rc.base_directions << lvl;
        for (int i = 0; i < l.allocated; i++) {
            int w = meta[rc.probe_base[lvl] + i].w;
            if (w < 0) continue;
            l.live++;
            if ((w >> 4) != 0) l.fresh++;
        }
        s.probes += l.live;
        s.rays += l.live * d * d;
    }
    return s;
}

// Prints stats() as a few lines.
fn void print_stats(RcStats* s) {
    if (!s.sparse) {
        io::printfn(
            "rc: dense, %d levels, %d probes in cascade 0, %d rays, frame %d",
            s.levels,
            s.probes,
            s.rays,
            s.frame
        );
        return;
    }
    if (s.split) {
        float fill_pct = s.bins_total > 0 ? 100.0f * (float)s.bins_filled / (float)s.bins_total : 0.0f;
        io::printfn(
            "rc: split, frame %d: %d live probes, %d traced rays, %d/%d bins filled (%.1f%%), beta [%.2f..%.2f]",
            s.frame,
            s.probes,
            s.traced_rays,
            s.bins_filled,
            s.bins_total,
            fill_pct,
            s.min_beta,
            s.max_beta
        );
        io::printfn(
            "    hash: %d of %d slots (load %.1f%%), longest run %d, %d runs over 32",
            s.occupied,
            s.hash_cap,
            s.load * 100,
            s.longest_run,
            s.runs_over_32
        );
        float cache_fill = s.cache_bins_total > 0 ? 100.0f * (float)s.cache_bins_filled / (float)s.cache_bins_total
            : 0.0f;
        io::printfn(
            "    cache: %d live probes, %d traced rays, %d/%d bins filled (%.1f%%)",
            s.cache_probes,
            s.cache_traced_rays,
            s.cache_bins_filled,
            s.cache_bins_total,
            cache_fill
        );
        io::printn("    level      cap  alloc   free   live  fresh  table   seen  bins%");
        for (int i = 0; i < s.levels; i++) {
            RcLevelStats* l = &s.level[i];
            float lp        = l.bins_total > 0 ? 100.0f * (float)l.bins_filled / (float)l.bins_total : 0.0f;
            io::printfn(
                "    L%-3d %8d %6d %6d %6d %6d %6d %6d %5.1f%%",
                i,
                l.cap,
                l.allocated,
                l.free_listed,
                l.live,
                l.fresh,
                l.in_table,
                l.seen,
                lp
            );
        }
        return;
    }
    io::printfn("rc: sparse, frame %d: %d live probes, %d rays", s.frame, s.probes, s.rays);
    io::printfn(
        "    hash: %d of %d slots (load %.1f%%, %.1f%% with %d tombstones), longest run %d, %d runs over 32",
        s.occupied,
        s.hash_cap,
        s.load * 100,
        s.load_tombstones * 100,
        s.tombstones,
        s.longest_run,
        s.runs_over_32
    );
    io::printfn("    stale %d (evicted in %d updates)", s.stale, s.frames_to_evict);
    io::printn("    level      cap  alloc   free   live  fresh  table   seen  stale");
    for (int i = 0; i < s.levels; i++) {
        RcLevelStats* l = &s.level[i];
        io::printfn(
            "    L%-3d %8d %6d %6d %6d %6d %6d %6d %6d",
            i,
            l.cap,
            l.allocated,
            l.free_listed,
            l.live,
            l.fresh,
            l.in_table,
            l.seen,
            l.stale
        );
    }
}