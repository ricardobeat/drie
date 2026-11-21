// Whole programs printed from node graphs (nodes_program.c3): small ones made by hand are
// compiled and drawn, and their pixels checked. The GLSL and HLSL they print too go to
// tests/build/node_programs.{glsl,hlsl} to read. `just test node_programs`
module node_programs_test;
import drie, drie::nodes, drie::readback, sokol;
import std::io, std::math, std::io::file;

const int W = 16;
const int H = 16;

int failures;

fn void expect(bool ok, String what) {
    io::printfn("%s %s", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

fn bool near(char got, float want) => math::abs((float)got - want * 255) <= 2;

fn bool rgba_near(char[] px, float r, float g, float b, float a) {
    return near(px[0], r) && near(px[1], g) && near(px[2], b) && near(px[3], a);
}

SgBuffer quad;            // two triangles over the target: position (xyz) and uv, interleaved
SgImage[2] colors;
SgView[2] color_views;
SgView[2] color_textures;
SgImage depth;
SgView depth_view;

fn void setup() {
    float[*] v = {
        -1, -1, 0, 0, 1,   1, -1, 0, 1, 1,   1, 1, 0, 1, 0,
        -1, -1, 0, 0, 1,   1, 1, 0, 1, 0,   -1, 1, 0, 0, 0,
    };
    quad = sg::make_buffer(&&{ .data = { .ptr = &v, .size = v.len * float::size } });
    for (int i = 0; i < 2; i++) {
        colors[i] = sg::make_image(&&{ .usage = { .color_attachment = true }, .width = W, .height = H, .pixel_format = RGBA8 });
        color_views[i] = sg::make_view(&&{ .color_attachment = { .image = colors[i] } });
        color_textures[i] = sg::make_view(&&{ .texture = { .image = colors[i] } });
    }
    depth = sg::make_image(&&{ .usage = { .depth_stencil_attachment = true }, .width = W, .height = H, .pixel_format = DEPTH });
    depth_view = sg::make_view(&&{ .depth_stencil_attachment = { .image = depth } });
}

// `p` drawn over the target(s), its uniform rows from `fs` (the FS source) and its views by
// source from `views`: the middle pixel of each color output.
fn char[4][2] draw(ShaderProgram* p, drie::Vec4[] fs, SgView[] views, SgSampler[] samplers, int instances = 1) {
    SgShader shader = p.make();
    SgPipelineDesc pd = {
        .shader = shader,
        .layout = {
            .buffers[0].stride = 20,
            .attrs[0] = { .format = FLOAT3, .offset = 0 },
            .attrs[1] = { .format = FLOAT3, .offset = 0 },  // (sokol reads attributes up to the first unset one)
            .attrs[2] = { .format = FLOAT2, .offset = 12 },
        },
        .color_count = p.colors,
        .depth = { .pixel_format = DEPTH },
    };
    for (int i = 0; i < p.colors; i++) pd.colors[i].pixel_format = RGBA8;
    SgPipeline pip = sg::make_pipeline(&pd);
    SgPass pass = {
        .attachments = { .depth_stencil = depth_view },
        .action = { .depth = { .load_action = CLEAR, .clear_value = 1 } },
    };
    for (int i = 0; i < p.colors; i++) {
        pass.attachments.colors[i] = color_views[i];
        pass.action.colors[i] = { .load_action = CLEAR, .clear_value = { 0, 0, 0, 0 } };
    }
    sg::begin_pass(&pass);
    sg::apply_pipeline(pip);
    SgBindings b = { .vertex_buffers[0] = quad };
    foreach (i, v : p.views) b.views[i] = views[v.source];
    foreach (i, s : p.samplers) b.samplers[i] = samplers[s.source];
    sg::apply_bindings(&b);
    void*[NodeSource::len] src;
    sz[NodeSource::len] sizes;
    src[NodeSource.FS.ordinal] = fs.ptr;
    sizes[NodeSource.FS.ordinal] = fs.len * drie::Vec4::size;
    for (int st = 0; st < 2; st++) {
        if (!p.block_rows[st]) continue;
        drie::Vec4[] block = mem::temp_array(drie::Vec4, p.block_rows[st]);
        p.gather((ProgramStage)st, src, sizes, block);
        sg::apply_uniforms(st, &&{ .ptr = block.ptr, .size = block.len * drie::Vec4::size });
    }
    sg::draw(0, 6, instances);
    sg::end_pass();
    sg::commit();
    char[4][2] out;
    char[] px = mem::temp_array(char, (usz)W * H * 4);
    for (int i = 0; i < p.colors; i++) {
        readback::read_image(colors[i], W, H, px.ptr);
        usz at = ((usz)(H / 2) * W + W / 2) * 4;
        out[i] = px[at:4];
    }
    sg::destroy_pipeline(pip);
    return out;
}

fn Node position() => nodes::vec4(drie::node_attribute(0, VEC3), 1.0f);
fn Node uv() => nodes::varying(drie::node_attribute(2, VEC2));

// the program's text in the three languages: Metal's to draw, the others to read
fn ShaderProgram* program(ProgramGraph g, String what, DString* glsl, DString* hlsl) {
    ShaderProgram* gp = drie::build_program(GLSL, &g);
    glsl.appendf("==== %s\n---- vertex\n%s---- fragment\n%s\n", what, gp.vs, gp.fs);
    ShaderProgram* hp = drie::build_program(HLSL, &g);
    hlsl.appendf("==== %s\n---- vertex\n%s---- fragment\n%s\n", what, hp.vs, hp.fs);
    return drie::build_program(drie::backend_lang(), &g);
}

fn int main(String[] args) {
    Renderer* renderer = drie::renderer(antialias: false, width: W, height: H);
    renderer.init_headless();
    setup();
    DString glsl = dstring::new(mem);
    DString hlsl = dstring::new(mem);

    // a texture, filled by drawing a flat color first
    SgView[8] views;
    SgSampler[4] samplers;
    samplers[0] = sg::make_sampler(&&{ .min_filter = NEAREST, .mag_filter = NEAREST });
    samplers[1] = sg::make_sampler(&&{ .compare = LESS_EQUAL, .min_filter = LINEAR, .mag_filter = LINEAR });

    // 1: a flat color from a uniform row, and a vec3 from another row's components
    drie::Vec4[4] fs = { { 0.25f, 0.5f, 0.75f, 1 }, { 9, 0.2f, 0.4f, 0.6f }, { 3, 0, 0, 0 }, { 0, 0, 0, 0 } };
    ShaderProgram* flat = program({ .position = position(), .colors[0] = drie::node_input(FS, 0, VEC4) }, "flat", &glsl, &hlsl);
    char[4][2] got = draw(flat, &fs, &views, &samplers);
    expect(rgba_near(&got[0], 0.25f, 0.5f, 0.75f, 1), "a uniform row");
    ShaderProgram* comps = program(
        { .position = position(), .colors[0] = nodes::vec4(drie::node_input(FS, 1, VEC3, comp: 1), 1.0f) }, "components", &glsl,
        &hlsl);
    got = draw(comps, &fs, &views, &samplers);
    expect(rgba_near(&got[0], 0.2f, 0.4f, 0.6f, 1), "a row's components");
    expect(comps.block_rows[1] == 1, "the block holds the rows read, and no others");

    // 2: the texture that drew, read at the varying uv, times a uniform: MRT's second output
    views[0] = color_textures[0];
    Node tex = drie::node_sample(SAMPLE, 0, 0, TEX_2D, uv());
    ShaderProgram* textured = program({ .position = position(), .colors[0] = tex * 0.5f, .colors[1] = tex.one_minus() }, "mrt", &glsl,
                                      &hlsl);
    // (drawn into colors[0] while reading it: draw into a copy instead)
    SgImage src = sg::make_image(&&{ .usage = { .color_attachment = true }, .width = W, .height = H, .pixel_format = RGBA8 });
    SgView src_att = sg::make_view(&&{ .color_attachment = { .image = src } });
    views[0] = sg::make_view(&&{ .texture = { .image = src } });
    sg::begin_pass(&&{ .attachments.colors[0] = src_att, .action.colors[0] = { .load_action = CLEAR, .clear_value = { 0.8f, 0.4f, 0.2f, 1 } } });
    sg::end_pass();
    got = draw(textured, &fs, &views, &samplers);
    expect(rgba_near(&got[0], 0.4f, 0.2f, 0.1f, 0.5f), "a texture at a varying's uv");
    expect(rgba_near(&got[1], 0.2f, 0.6f, 0.8f, 0), "a second output (MRT)");

    // 3: a loop over a storage buffer, its count a uniform, in a function that reads the block
    float[*] data = { 0.1f, 0, 0, 0, 0, 0.2f, 0, 0, 0, 0, 0.3f, 0, 0.5f, 0.5f, 0.5f, 0.5f };
    SgBuffer sb = sg::make_buffer(&&{ .usage = { .storage_buffer = true }, .data = { .ptr = &data, .size = data.len * float::size } });
    views[1] = sg::make_view(&&{ .storage_buffer = { .buffer = sb } });
    int sum_fn = drie::node_func("sum_items", VEC4, { INT });
    {
        Node count = drie::node_param(sum_fn, 0);
        Node acc = nodes::variable(nodes::vec4(0, 0, 0, 1));
        Node i = nodes::loop_index();
        Node body = nodes::block(acc.assign(acc + drie::node_storage(1, VEC4, i) * drie::node_input(FS, 1, FLOAT).div_f(9)));
        drie::node_func_body(sum_fn, nodes::seq(nodes::block(acc, nodes::loop(i, 0, count, body)), acc));
    }
    Node items = drie::node_call_fn(sum_fn, { nodes::convert(drie::node_input(FS, 2, FLOAT), INT) });
    ShaderProgram* looped = program({ .position = position(), .colors[0] = items }, "storage loop", &glsl, &hlsl);
    got = draw(looped, &fs, &views, &samplers);
    expect(rgba_near(&got[0], 0.1f, 0.2f, 0.3f, 1), "a loop over a storage buffer, in a function");

    // 4: a depth comparison, the depth cleared to 0.5 by a pass of its own
    SgImage shadow = sg::make_image(&&{ .usage = { .depth_stencil_attachment = true }, .width = W, .height = H, .pixel_format = DEPTH });
    SgView shadow_att = sg::make_view(&&{ .depth_stencil_attachment = { .image = shadow } });
    views[2] = sg::make_view(&&{ .texture = { .image = shadow } });
    sg::begin_pass(&&{ .attachments.depth_stencil = shadow_att, .action.depth = { .load_action = CLEAR, .store_action = STORE, .clear_value = 0.5f } });
    sg::end_pass();
    Node lit_near = drie::node_sample(COMPARE, 2, 1, TEX_DEPTH, nodes::vec2(0.5f, 0.5f), nodes::num(0.4f));
    Node lit_far = drie::node_sample(COMPARE, 2, 1, TEX_DEPTH, nodes::vec2(0.5f, 0.5f), nodes::num(0.6f));
    ShaderProgram* compare = program({ .position = position(), .colors[0] = nodes::vec4(lit_near, lit_far, 0, 1) }, "compare", &glsl,
                                     &hlsl);
    got = draw(compare, &fs, &views, &samplers);
    expect(near(got[0][0], 1) && near(got[0][1], 0), "a depth comparison");

    // 5: a 3x3 matrix, a flat integer varying, the fragment's position and a bound builtin
    Node m = drie::node_matrix({ nodes::vec3(0, 1, 0), nodes::vec3(1, 0, 0), nodes::vec3(0, 0, 0.5f) }, MAT3);
    Node turned = m * nodes::vec3(0.2f, 0.4f, 0.8f);
    Node id = nodes::varying(drie::node_system(INSTANCE_ID) + nodes::uint_(3));
    Node fc = drie::node_system(FRAG_COORD);
    Node shade = nodes::seq(nodes::block(drie::node_bind(TIME, nodes::num(0.25f))), nodes::time());
    Node c5 = nodes::vec4(turned.x(), turned.y(), nodes::convert(id, FLOAT) * 0.1f, nodes::step(1, fc.x()) * shade);
    ShaderProgram* misc = program({ .position = position(), .colors[0] = c5 }, "matrix, flat varying, bind", &glsl, &hlsl);
    got = draw(misc, &fs, &views, &samplers);
    expect(rgba_near(&got[0], 0.4f, 0.2f, 0.3f, 0.25f), "a 3x3 matrix, an integer varying, a bound builtin");

    // 6: a discard and a vertex-stage early return
    Node cut = nodes::seq(nodes::block(nodes::if_then(drie::node_input(FS, 0, FLOAT).less_than(nodes::num(1)), nodes::block(nodes::discard()))));
    Node away = nodes::seq(
        nodes::block(nodes::if_then(drie::node_input(VS, 0, FLOAT).less_than(nodes::num(1)), nodes::block(drie::node_return(nodes::vec4(2, 2, 2, 1)))))
        , position());
    ShaderProgram* cutp = program({ .position = position(), .colors[0] = nodes::vec4(1, 1, 1, 1), .fragment = cut }, "discard", &glsl, &hlsl);
    got = draw(cutp, &fs, &views, &samplers);
    expect(rgba_near(&got[0], 0, 0, 0, 0), "a discard");
    ShaderProgram* gone = program({ .position = away, .colors[0] = nodes::vec4(1, 1, 1, 1) }, "return", &glsl, &hlsl);
    got = draw(gone, &fs, &views, &samplers);
    expect(rgba_near(&got[0], 0, 0, 0, 0), "a vertex sent away (return)");

    // 7: a variable read again after it changes, and a loop that changes what was read before it
    Node x = nodes::variable(nodes::num(0.1f));
    Node twice = x * 2;
    Node acc7 = nodes::variable(nodes::num(0));
    Node k = nodes::loop_index();
    Node seq7 = nodes::seq(nodes::block(x, acc7, drie::node_eval(twice), nodes::loop(k, 0, 3, nodes::block(acc7.assign(acc7 + twice), x.assign(x + 0.05f))),
                                        x.assign(x + 0.05f)),
                           nodes::vec4(acc7 * 0.5f, twice, 0, 1));
    ShaderProgram* vars = program({ .position = position(), .colors[0] = seq7 }, "variables", &glsl, &hlsl);
    got = draw(vars, &fs, &views, &samplers);
    // acc = 0.2 + 0.3 + 0.4 = 0.9 (x * 2 each time round), x ends at 0.1 + 4 * 0.05 = 0.3
    expect(rgba_near(&got[0], 0.45f, 0.6f, 0, 1), "values read again after their variables change");

    (void)file::save("tests/build/node_programs.glsl", glsl.str_view());
    (void)file::save("tests/build/node_programs.hlsl", hlsl.str_view());
    if (failures) {
        io::printfn("%d failed", failures);
        return 1;
    }
    io::printn("all passed");
    return 0;
}
