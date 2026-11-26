// Render pipeline creation, vertex buffer layouts, blend state packing, and shader warmup.
// Defines the 7-buffer vertex layout (0: mesh vertices, 1: instance 3x4 rows, 2: uv1, 3: tangents,
// 4: displacement heights, 5: instance colors, 6: previous-frame positions), caches forward,
// depth-prepass, deferred G-Buffer, and shadow pipelines on MaterialProgram, and coordinates
// worker-thread shader compilation with `material_program()` and `pipeline_warmup`. Connects to
// draw_item(), draw_shadow_item(), Renderer.render(), and pipeline cache recording.
module drie;
import std::collections::list, std::collections::map, std::math;
import sokol, drie::nodes;

// ---- Vertex layout and blend state

// Builds the shared 7-buffer vertex layout for attribute locations: [position, normal, uv, color,
// uv1, tangent, inst0, inst1, inst2, height, instance_color, prev_position]. Optional slots
// (tangent, instance_color, prev_position) are omitted when their attribute index is -1.
fn SgVertexLayoutState vertex_layout(int[12] attrs) @private {
    SgVertexLayoutState layout;
    layout.buffers[0].stride = FLOATS_PER_VERTEX * float::size;
    layout.buffers[1] = { .stride = 12 * float::size, .step_func = PER_INSTANCE };
    layout.buffers[2].stride = 2 * float::size;
    layout.attrs[attrs[0]] = { .format = FLOAT3, .offset = 0 };
    layout.attrs[attrs[1]] = { .format = FLOAT3, .offset = 12 };
    layout.attrs[attrs[2]] = { .format = FLOAT2, .offset = 24 };
    layout.attrs[attrs[3]] = { .format = FLOAT4, .offset = 32 };
    layout.attrs[attrs[4]] = { .format = FLOAT2, .offset = 0, .buffer_index = 2 };
    if (attrs[5] >= 0) {
        layout.buffers[3].stride = 4 * float::size;
        layout.attrs[attrs[5]] = { .format = FLOAT4, .offset = 0, .buffer_index = 3 };
    }
    layout.attrs[attrs[6]] = { .format = FLOAT4, .offset = 0, .buffer_index = 1 };
    layout.attrs[attrs[7]] = { .format = FLOAT4, .offset = 16, .buffer_index = 1 };
    layout.attrs[attrs[8]] = { .format = FLOAT4, .offset = 32, .buffer_index = 1 };
    layout.buffers[4].stride = float::size;
    layout.attrs[attrs[9]] = { .format = FLOAT, .offset = 0, .buffer_index = 4 };
    if (attrs[10] >= 0) {
        layout.buffers[5] = { .stride = 3 * float::size, .step_func = PER_INSTANCE };
        layout.attrs[attrs[10]] = { .format = FLOAT3, .offset = 0, .buffer_index = 5 };
    }
    if (attrs[11] >= 0) {
        layout.buffers[6].stride = FLOATS_PER_VERTEX * float::size;
        layout.attrs[attrs[11]] = { .format = FLOAT3, .offset = 0, .buffer_index = 6 };
    }
    return layout;
}

// Pass mode for main-pass draws (see Renderer.depth_prepass).
enum DrawKind : char @private {
    CUT,        // Shader may discard fragments (alpha test, dither, cull box, or transparent).
    SOLID,      // No fragment discard; preserves hardware early-Z.
    PRETESTED,  // Shaded after depth prepass with EQUAL depth test and depth writes disabled.
    DEPTH,      // Depth-only prepass with color writes masked off and discard enabled.
}

// Packs a material's blend configuration into an integer key (0 disables blending; normal
// blending activates only when `transparent` is set). Preset modes pack into 1..15; custom
// equations and factors pack above bit 4.
fn ulong blend_key(Material* mat) @private {
    if (mat.blending == NO_BLENDING || (mat.blending == NORMAL_BLENDING && !mat.transparent)) return 0;
    if (mat.blending != CUSTOM_BLENDING) return 1 + 2 * (ulong)mat.blending.ordinal + (mat.premultiplied_alpha ? 1UL : 0UL);
    return 16 | (ulong)mat.blend_equation.ordinal << 5 | (ulong)mat.blend_equation_alpha.ordinal << 8
        | (ulong)mat.blend_src.ordinal << 11 | (ulong)mat.blend_dst.ordinal << 15
        | (ulong)mat.blend_src_alpha.ordinal << 19 | (ulong)mat.blend_dst_alpha.ordinal << 23;
}

fn SgBlendFactor sg_factor(ulong f) @local {
    switch ((BlendFactor)f) {
        case ZERO_FACTOR: return ZERO;
        case ONE_FACTOR: return ONE;
        case SRC_COLOR_FACTOR: return SRC_COLOR;
        case ONE_MINUS_SRC_COLOR_FACTOR: return ONE_MINUS_SRC_COLOR;
        case SRC_ALPHA_FACTOR: return SRC_ALPHA;
        case ONE_MINUS_SRC_ALPHA_FACTOR: return ONE_MINUS_SRC_ALPHA;
        case DST_ALPHA_FACTOR: return DST_ALPHA;
        case ONE_MINUS_DST_ALPHA_FACTOR: return ONE_MINUS_DST_ALPHA;
        case DST_COLOR_FACTOR: return DST_COLOR;
        case ONE_MINUS_DST_COLOR_FACTOR: return ONE_MINUS_DST_COLOR;
        case SRC_ALPHA_SATURATE_FACTOR: return SRC_ALPHA_SATURATED;
    }
}

fn SgBlendOp sg_op(ulong e) @local {
    switch ((BlendEquation)e) {
        case ADD_EQUATION: return ADD;
        case SUBTRACT_EQUATION: return SUBTRACT;
        case REVERSE_SUBTRACT_EQUATION: return REVERSE_SUBTRACT;
        case MIN_EQUATION: return MIN;
        case MAX_EQUATION: return MAX;
    }
}

// Unpacks a blend_key() value into a sokol SgBlendState.
fn SgBlendState blend_state(ulong key) @private {
    if (key == 0) return { };
    if (key >= 16) {
        return {
            .enabled = true,
            .op_rgb = sg_op(key >> 5 & 7),
            .op_alpha = sg_op(key >> 8 & 7),
            .src_factor_rgb = sg_factor(key >> 11 & 15),
            .dst_factor_rgb = sg_factor(key >> 15 & 15),
            .src_factor_alpha = sg_factor(key >> 19 & 15),
            .dst_factor_alpha = sg_factor(key >> 23 & 15),
        };
    }
    bool premultiplied = (key - 1) % 2 == 1;
    switch ((Blending)((key - 1) / 2)) {
        case ADDITIVE_BLENDING:
            return {
                .enabled = true,
                .src_factor_rgb = premultiplied ? SgBlendFactor.ONE : SgBlendFactor.SRC_ALPHA,
                .dst_factor_rgb = ONE,
                .src_factor_alpha = ZERO,
                .dst_factor_alpha = ONE,
            };
        case SUBTRACTIVE_BLENDING:
            return { .enabled = true, .src_factor_rgb = ZERO, .dst_factor_rgb = ONE_MINUS_SRC_COLOR,
                     .src_factor_alpha = ZERO, .dst_factor_alpha = ONE };
        case MULTIPLY_BLENDING:
            return { .enabled = true, .src_factor_rgb = DST_COLOR, .dst_factor_rgb = ONE_MINUS_SRC_ALPHA,
                     .src_factor_alpha = ZERO, .dst_factor_alpha = ONE };
        default:
            return {
                .enabled = true,
                .src_factor_rgb = premultiplied ? SgBlendFactor.ONE : SgBlendFactor.SRC_ALPHA,
                .dst_factor_rgb = ONE_MINUS_SRC_ALPHA,
                .src_factor_alpha = ONE,
                .dst_factor_alpha = ONE_MINUS_SRC_ALPHA,
            };
    }
}

// ---- Main and G-Buffer pipelines

// True when `mat` uses physical surface layers not representable in the deferred G-Buffer
// (clearcoat, sheen, iridescence, anisotropy, transmission, or glass).
fn bool physical_layers(Material* mat) @private {
    return mat.type == STANDARD && (mat.clearcoat > 0 || mat.clearcoat_node.id || mat.clearcoat_roughness_node.id
        || mat.sheen > 0 || mat.sheen_color_map != null || mat.sheen_roughness_map != null
        || mat.iridescence > 0 || mat.iridescence_map != null
        || mat.anisotropy > 0 || mat.anisotropy_map != null
        || mat.transmission > 0 || mat.transmission_map != null
        || mat.glass);
}

// Looks up or creates a cached SgPipeline on `gen` for the requested raster, depth, target, and
// deferred G-Buffer state. Dispatches background compilation and returns `{}` during pipeline
// cache warmup.
fn SgPipeline get_pipeline(
    Side side,
    bool transparent,
    bool wireframe,
    RenderTarget* rt,
    bool alpha_test,
    ulong blend,
    DrawKind kind,
    bool depth_test,
    bool depth_write,
    bool gbuffer,
    MaterialProgram* gen
) @private {
    int target = rt ? (rt.samples > 1 ? 2 : 1) + (rt.mrt ? 4 : (rt.is_linear() ? 2 : 0)) : 0;
    if (rt && rt.samples != 1 && rt.samples != 4) abort("drie: render targets support 1 or 4 samples");
    if (rt && rt.pixel_format() != RGBA8 && !rt.is_linear()) abort("drie: a scene renders into RGBA8 or RGBA16F");
    int samples   = rt ? rt.samples : sapp::sample_count();
    // Pretested draws rely on per-sample depth written by the depth prepass instead of A2C.
    bool a2c      = alpha_test && !transparent && samples > 1 && kind != PRETESTED;
    if (kind == DEPTH) blend = 0;
    bool vsm      = kind != DEPTH && !gbuffer && state.renderer && state.renderer.shadow_type == VSM;
    if (kind == DEPTH || kind == PRETESTED) {
        depth_test = true;
        depth_write = true;
    }
    int depth = (depth_write ? 0 : 1) + (depth_test ? 0 : 2);
    if (gbuffer) {
        target = 7;
        a2c = false;
        blend = 0;
    }
    ulong key = blend | (ulong)depth << 28 | (ulong)(vsm ? 1 : 0) << 30
        | (ulong)kind.ordinal << 31 | (ulong)side.ordinal << 33 | (ulong)(wireframe ? 1 : 0) << 35
        | (ulong)target << 36 | (ulong)(a2c ? 1 : 0) << 39 | (ulong)(transparent ? 1 : 0) << 40
        | (ulong)(gbuffer ? 1 : 0) << 41;
    SgPipeline? found = gen.pipelines[key];
    if (try found) return found;
    bool window = !rt || (state.renderer && rt == state.renderer.screen_stand_in);
    record_pipeline({
        .key = gen.key,
        .shared = gen.key.nodes ? gen : null,
        .side = side,
        .transparent = transparent,
        .wireframe = wireframe,
        .alpha_test = alpha_test,
        .blend = blend,
        .kind = kind,
        .gbuffer = gbuffer,
        .depth_test = depth_test,
        .depth_write = depth_write,
        .vsm = vsm,
        .target_samples = window ? 0 : rt.samples,
        .target_format = window ? SgPixelFormat.DEFAULT : rt.pixel_format(),
        .target_mrt = !window && rt.mrt,
    });
    SgCullMode cull;
    switch (side) {
        case FRONT_SIDE: cull = BACK;
        case BACK_SIDE: cull = FRONT;
        case DOUBLE_SIDE: cull = NONE;
    }
    bool mrt = rt && rt.mrt;
    SgPipelineDesc desc = {
        .index_type = UINT32,
        .primitive_type = wireframe ? LINES : TRIANGLES,
        .cull_mode = wireframe ? NONE : cull,
        .face_winding = CCW,
        .layout = vertex_layout(
            {
                shaders::ATTR_THREE_POSITION,
                shaders::ATTR_THREE_NORMAL,
                shaders::ATTR_THREE_UV,
                shaders::ATTR_THREE_COLOR0,
                shaders::ATTR_THREE_UV1,
                shaders::ATTR_THREE_TANGENT,
                shaders::ATTR_THREE_INST0,
                shaders::ATTR_THREE_INST1,
                shaders::ATTR_THREE_INST2,
                shaders::ATTR_THREE_HEIGHT,
                shaders::ATTR_THREE_INSTANCE_COLOR,
                shaders::ATTR_THREE_PREV_POSITION
            }
        ),
        .depth = {
            .write_enabled = !transparent && kind != PRETESTED && depth_write,
            .compare = kind == PRETESTED ? SgCompareFunc.EQUAL
                : depth_test ? SgCompareFunc.LESS_EQUAL
                : SgCompareFunc.ALWAYS,
        },
        .alpha_to_coverage_enabled = a2c,
        .label = kind == DEPTH ? "drie-depth-prepass" : "drie-pipeline",
    };
    if (kind == DEPTH) desc.colors[0].write_mask = NONE;
    if (gbuffer) {
        desc.sample_count = 1;
        desc.color_count = 4;
        for (int i = 0; i < 4; i++) desc.colors[i] = { .pixel_format = RGBA16F };
        desc.depth.pixel_format = DEPTH;
        desc.alpha_to_coverage_enabled = false;
        desc.label = "drie-gbuffer";
    } else {
        if (rt) {
            desc.sample_count = rt.samples;
            desc.colors[0].pixel_format = rt.pixel_format();
            desc.depth.pixel_format = DEPTH;
        }
        if (mrt) {
            desc.color_count = 1 + MRT_COUNT;
            for (int i = 0; i < MRT_COUNT; i++) {
                desc.colors[i + 1].pixel_format = mrt_format(i);
                // Transparent and depth-only passes preserve underlying MRT normal/velocity/indirect.
                if (kind == DEPTH || transparent) desc.colors[i + 1].write_mask = NONE;
            }
        }
        desc.colors[0].blend = blend_state(blend);
    }
    ShaderProgram* own = kind == DEPTH ? gen.depth : gen.program;
    if (pipeline_warmup) {
        pipeline_warmup.start_generated(own, &desc);
        return { };
    }
    desc.shader = own.make();
    SgPipeline pip = sg::make_pipeline(&desc);
    if (!pip.id) abort("drie: sokol's pipeline pool is full (pipeline_pool_size in setup_gfx)");
    gen.pipelines[key] = pip;
    return pip;
}

// Lazily creates the built-in depth shader (`render_back_faces`) or shadow shader (`clear_atlas_tile`).
fn SgShader engine_program(bool shadow) @private {
    SgShader* s = &state.engine_programs[shadow ? 1 : 0];
    if (!s.id) {
        SgBackend backend = sg::query_backend();
        *s = make_program(shadow ? shaders::shadow_shader_desc(backend) : shaders::depth_shader_desc(backend));
    }
    return *s;
}

// ---- Background program warmup and node uniforms

// Kicks off parallel worker-thread compilation for all unbuilt material programs needed by this
// render's background, opaque, and transparent draws so they overlap before draw submission.
fn void start_material_programs(Renderer* r, Scene* scene, Camera* camera, FsParams* fs, bool prepass) @private {
    start_background_program(scene, camera, fs);
    bool defer_ok = can_defer(r, true, true);
    for (int list = 0; list < 2; list++) {
        foreach (&item : list == 0 ? state.opaque : state.transparent) {
            // Match the DrawKind that Renderer.render() will select for each list.
            DrawKind kind = list == 1 || draws_lines(item) || has_holes(item) ? CUT : SOLID;
            if (list == 0 && defer_ok && is_deferred_item(item)) {
                start_item_program(item, kind, fs, true);
                continue;
            }
            if (list == 0 && kind == CUT && prepass && prepassed(item)) kind = PRETESTED;
            start_item_program(item, kind, fs);
        }
    }
}

// Starts compiling the generated shader (and depth prepass variant) for `item`; under background
// compilation, also starts the plain fallback program for node materials.
fn void start_item_program(DrawItem* item, DrawKind kind, FsParams* fs, bool gbuffer = false) @private {
    Texture*[6] slots;
    Vec4[6] map_slots;
    material_slots(item.mesh.material, null, &slots, &map_slots);
    NodeBuild* build = draw_build(item.mesh.material);
    for (int plain = 0; plain < (build && background() ? 2 : 1); plain++) {
        MaterialKey key;
        material_key_of(item, kind, fs, &map_slots, &key, gbuffer, plain == 1);
        MaterialProgram* gen = material_program(key, plain == 1 ? null : build);
        gen.program.start();
        if (gen.depth) gen.depth.start();
    }
}

// Starts compiling programs for the scene background and fog backdrop meshes, if present.
fn void start_background_program(Scene* scene, Camera* camera, FsParams* fs) @private {
    DrawItem item;
    if (background_item(scene, camera, &item)) start_item_program(&item, CUT, fs);
    if (backdrop_item(scene, camera, &item)) start_item_program(&item, has_holes(&item) ? CUT : SOLID, fs);
}

// True when shaders should compile asynchronously (Renderer.compile_in_background or cache warmup).
fn bool background() @private => (state.renderer && state.renderer.compile_in_background) || pipeline_warmup != null;

// Returns the compiled node graph build for `mat`, or null during lightmap baking.
fn NodeBuild* draw_build(Material* mat) @private => state.bake.on ? null : node_build_of(mat);

// Packs a node material's USER uniform block in temporary memory: view/proj/model matrices,
// camera position + time (NODE_USER_HEADER), followed by graph uniform values.
fn Vec4[] user_block(NodeBuild* b, Mesh* m) @private {
    int rows = NODE_USER_HEADER;
    foreach (u : b.uniforms) rows += u.type() == MAT4 ? 4 : 1;
    Vec4[] out = mem::temp_array(Vec4, rows);
    Mat4[3] mats = { state.node_view, state.node_proj, m.matrix_world };
    foreach (i, mt : mats) mem::copy(&out[i * 4], &mt, Mat4::size);
    out[12] = { ...state.node_camera, state.renderer ? state.renderer.time : 0 };
    int at = NODE_USER_HEADER;
    foreach (u : b.uniforms) {
        float[16] v = node_data(u).value;
        int n = u.type() == MAT4 ? 4 : 1;
        for (int r = 0; r < n; r++) out[at++] = { v[r * 4], v[r * 4 + 1], v[r * 4 + 2], v[r * 4 + 3] };
    }
    return out;
}

// Returns the texture views bound to a node material's custom texture slots.
fn SgView[NODE_TEXTURES] user_views(NodeBuild* b) @private {
    SgView[NODE_TEXTURES] v;
    for (int k = 0; k < NODE_TEXTURES; k++) v[k] = ready(b && k < b.texture_count ? b.texture(k) : null).view;
    return v;
}

// True when node material `b` has programs compiling in the background and should draw with its
// plain fallback for this entire render (`node_epoch` keeps prepass and main pass consistent).
// Starts compilation of `mp` if not yet started.
fn bool material_busy(NodeBuild* b, MaterialProgram* mp) @private {
    if (!background()) return false;
    if (b.busy_epoch != node_epoch) {
        b.busy_epoch = node_epoch;
        b.busy_now = false;
        b.programs.@each(; KeyWords words, MaterialProgram* other) {
            if (other.program.compiling() || (other.depth && other.depth.compiling())) b.busy_now = true;
        };
    }
    if (!mp.program.shader.id) mp.program.start();
    if (mp.depth && !mp.depth.shader.id) mp.depth.start();
    if (mp.program.compiling() || (mp.depth && mp.depth.compiling())) b.busy_now = true;
    return b.busy_now;
}

// ---- Shadow pipelines

// Base descriptor for depth-only shadow pipelines (no color attachments, culling disabled).
fn SgPipelineDesc shadow_pipeline_desc() @local {
    return {
        .shader = { },
        .index_type = UINT32,
        .cull_mode = NONE,
        .layout = vertex_layout(
            {
                shaders::ATTR_SHADOW_POSITION,
                shaders::ATTR_SHADOW_NORMAL,
                shaders::ATTR_SHADOW_UV,
                shaders::ATTR_SHADOW_COLOR0,
                shaders::ATTR_SHADOW_UV1,
                -1,
                shaders::ATTR_SHADOW_INST0,
                shaders::ATTR_SHADOW_INST1,
                shaders::ATTR_SHADOW_INST2,
                shaders::ATTR_SHADOW_HEIGHT,
                -1,
                -1
            }
        ),
        .depth = {
            .pixel_format = DEPTH,
            .write_enabled = true,
            .compare = LESS_EQUAL,
        },
        .color_count = 0,
        .colors[0].pixel_format = NONE,
        .sample_count = 1,
        .label = "drie-shadow-pipeline",
    };
}

// Looks up or creates the shadow pipeline for `gen` (`MaterialKey.shadow`), using ATLAS_FORMAT
// for shadow-atlas tiles and DEPTH for the sun and primary spot maps. Returns `{}` during warmup.
fn SgPipeline gen_shadow_pipeline(MaterialProgram* gen) @private {
    if (try found = gen.pipelines[0]) return found;
    SgPipelineDesc desc = shadow_pipeline_desc();
    if (gen.key.atlas_depth) desc.depth.pixel_format = ATLAS_FORMAT;
    record_pipeline({ .key = gen.key, .shared = gen.key.nodes ? gen : null, .shadow = true });
    if (pipeline_warmup) {
        pipeline_warmup.start_generated(gen.program, &desc);
        return { };
    }
    desc.shader = gen.program.make();
    SgPipeline pip = sg::make_pipeline(&desc);
    gen.pipelines[0] = pip;
    return pip;
}

// 16-bit depth format for the shadow atlas and static caster cache; perspective atlas maps store
// linear distance over light reach, while sun cascades and the primary spot map use 32-bit DEPTH.
const SgPixelFormat ATLAS_FORMAT @private = DEPTH16;
