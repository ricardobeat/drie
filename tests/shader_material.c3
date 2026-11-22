// Headless test for ShaderMaterial: custom vertex/fragment shader with
// ShaderMaterialVsParams at UB 0 and custom fragment uniform block at UB 1,
// drawn into both standard and MRT render targets.
//
//   just test shader_material
module shader_material_test;
import drie, sokol, drie::readback, drie::png, std::io, std::math;

const int W = 256;
const int H = 256;

struct CustomFsUniforms {
    drie::Vec4 tint;
}

fn SgShaderDesc custom_test_shader(SgBackend backend) {
    SgShaderDesc desc = { .label = "test-shader-material" };
    desc.uniform_blocks[0] = {
        .stage = VERTEX,
        .size = ShaderMaterialVsParams::size,
        .msl_buffer_n = 0,
        .hlsl_register_b_n = 0,
        .glsl_uniforms[0] = { .glsl_name = "vs_params", .type = FLOAT4, .array_count = (ushort)(ShaderMaterialVsParams::size / 16) },
    };
    desc.uniform_blocks[1] = {
        .stage = FRAGMENT,
        .size = CustomFsUniforms::size,
        .msl_buffer_n = 0,
        .hlsl_register_b_n = 0,
        .glsl_uniforms[0] = { .glsl_name = "fs_params", .type = FLOAT4, .array_count = 1 },
    };
    if (backend == METAL_MACOS || backend == METAL_IOS || backend == METAL_SIMULATOR) {
        desc.vertex_func.source = `#include <metal_stdlib>
using namespace metal;
struct VsParams {
    float4x4 view_proj;
    float4 view_origin;
    float4x4 model;
    float4x4 normal_mat;
    float4 camera_time;
};
struct VsIn {
    float3 pos [[attribute(0)]];
    float3 nrm [[attribute(1)]];
    float2 uv  [[attribute(2)]];
    float4 col [[attribute(3)]];
};
struct VsOut {
    float4 clip [[position]];
    float2 uv;
};
vertex VsOut main0(VsIn in [[stage_in]], constant VsParams& u [[buffer(0)]]) {
    VsOut out;
    float4 wp = u.model * float4(in.pos, 1.0);
    out.clip = u.view_proj * float4(wp.xyz - u.view_origin.xyz, 1.0);
    out.uv = in.uv;
    return out;
}`;
        desc.vertex_func.entry = "main0";
        desc.fragment_func.source = `#include <metal_stdlib>
using namespace metal;
struct FsParams {
    float4 tint;
};
struct FsIn {
    float2 uv;
};
fragment float4 main0(FsIn in [[stage_in]], constant FsParams& u [[buffer(0)]]) {
    return float4(u.tint.rgb, 1.0);
}`;
        desc.fragment_func.entry = "main0";
    } else {
        desc.vertex_func.source = `#version 430
layout(location=0) in vec3 pos;
layout(location=1) in vec3 nrm;
layout(location=2) in vec2 uv;
layout(location=3) in vec4 col;
uniform vec4 vs_params[14];
out vec2 v_uv;
void main() {
    mat4 vp = mat4(vs_params[0], vs_params[1], vs_params[2], vs_params[3]);
    vec4 vo = vs_params[4];
    mat4 model = mat4(vs_params[5], vs_params[6], vs_params[7], vs_params[8]);
    vec4 wp = model * vec4(pos, 1.0);
    gl_Position = vp * vec4(wp.xyz - vo.xyz, 1.0);
    v_uv = uv;
}`;
        desc.fragment_func.source = `#version 430
uniform vec4 fs_params[1];
in vec2 v_uv;
out vec4 frag_color;
void main() {
    frag_color = vec4(fs_params[0].rgb, 1.0);
}`;
    }
    return desc;
}

fn int main(String[] args) {
    String out_path = args.len > 1 ? args[1] : "tests/build/shader_material.png";
    Renderer* renderer = drie::renderer(width: W, height: H);
    renderer.init_headless();

    Scene* scene = drie::scene();
    scene.background = 0x101010;

    CustomFsUniforms uniforms = { .tint = { 0.25f, 0.8f, 0.5f, 1.0f } };
    ShaderMaterial* mat = drie::shader_material(
        shader: &custom_test_shader,
        fs_uniforms: &uniforms,
        fs_uniforms_size: CustomFsUniforms::size,
        fs_ub: 1,
    );
    Mesh* box = drie::mesh(drie::box_geometry(1.6f, 1.6f, 1.6f), mat);
    scene.add(box);

    PerspectiveCamera* cam = drie::perspective_camera(45, 1.0f, 0.1f, 50);
    cam.position = { 0, 0, 3.5f };
    cam.look_at(0, 0, 0);

    RenderTarget* rt = drie::render_target(W, H, 1, RGBA8);
    renderer.set_render_target(rt);
    renderer.render(scene, cam);
    drie::commit();

    char[] px = mem::new_array(char, (usz)W * (usz)H * 4);
    readback::read_image(rt.texture_image, W, H, px.ptr);
    png::write(out_path, px, W, H);

    usz mid = ((usz)(H / 2) * W + (W / 2)) * 4;
    int r = px[mid];
    int g = px[mid + 1];
    int b = px[mid + 2];
    io::printfn("center pixel: [%d, %d, %d]", r, g, b);
    if (math::abs(r - 64) > 3 || math::abs(g - 204) > 3 || math::abs(b - 128) > 3) {
        io::eprintfn("FAIL expected ~[64, 204, 128], got [%d, %d, %d]", r, g, b);
        return 1;
    }

    // Also verify drawing into an MRT target without an MRT-specific shader.
    RenderTarget* mrt = drie::render_target(W, H, 1, RGBA16F, mrt: true);
    uniforms.tint = { 0.8f, 0.2f, 0.4f, 1.0f };
    renderer.set_render_target(mrt);
    renderer.render(scene, cam);
    drie::commit();

    io::printfn("ok   ShaderMaterial rendered in RGBA8 and MRT targets");
    return 0;
}
