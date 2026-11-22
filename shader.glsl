// The uber shader, shader_core.glsl, with hooks that do nothing. drie's materials draw with
// generated programs (materials_nodes.c3); of this build drie uses the structs and binding
// slots, and the depth and shadow programs (Renderer.render_back_faces, the shadow passes).
// Regenerate generated/shader.c3 with: just shaders
//
// An application with surfaces of its own compiles its own copy of the shader: a .glsl file
// with its own @header (its module) and the two blocks below, then `@include` of
// tonemapping.glsl and shader_core.glsl, in that order (sokol-shdc finds an include from the
// file it compiles, so shader_core.glsl can't include it itself). It hands the result to the
// renderer as Renderer.shaders, and every material draws with it. The hooks run
// inside the engine's shaders and read Material.custom (`material_custom[2]`), its
// custom_map (`custom_map_texel(uv, dx, dy)`) and Renderer.custom (`frame_custom[2]`):
//
//   vertex_hook(world, local, model)  in the main, depth and shadow passes: moves a vertex.
//       `world` is its position in the world after sway, `local` the one in the instance
//       (before `model`); the vertex attributes (color0...) and the instance's matrix rows
//       (m0, m1, m2) can be read too.
//   surface_hook(geo_n, base, rough, metal)  for lit materials, after the textures: changes
//       the color (rgb, a = opacity), roughness and metalness. `geo_n` is the normal before
//       the normal map; v_world_pos, v_uv, camera_pos... are there to read. A standard
//       material's clear coat and sheen are set up by then, and it can change them too:
//       g_clearcoat (0..1), g_clearcoat_rough (0.045..1) and g_clearcoat_n (a water film over
//       a wet surface), g_sheen_color and g_sheen_rough (0: no sheen).
//   normal_hook(geo_n, n)  for lit materials, after the normal map: changes the normal the
//       surface is shaded with (and that the screen-space passes see). `geo_n` is the normal
//       before the normal map, facing the camera.
//   reflection_hook(dir, sky)  changes what an environment map (or glass's sky) reflects
//       along `dir`.
@header module drie::shaders;
@header import drie;

@ctype mat4 Mat4
@ctype vec4 Vec4

@block vertex_hooks
void vertex_hook(inout vec3 world, vec3 local, mat4 model) {}
@end

@block surface_hooks
void surface_hook(vec3 geo_n, inout vec4 base, inout float rough, inout float metal) {}
void normal_hook(vec3 geo_n, inout vec3 n) {}
void reflection_hook(vec3 dir, inout vec3 sky) {}
@end

@include tonemapping.glsl
@include shader_core.glsl

@fs fs
@include_block fs_main
@end

@fs fs_opaque
#define NO_DISCARD
@include_block fs_main
@end

@fs fs_mrt
#define MRT
@include_block fs_main
@end

@fs fs_opaque_mrt
#define NO_DISCARD
#define MRT
@include_block fs_main
@end

@fs fs_vsm
#define VSM
@include_block fs_main
@end

@fs fs_mrt_vsm
#define MRT
#define VSM
@include_block fs_main
@end

@fs depth_fs
@include_block depth_main
@end

@vs shadow_vs
@include_block shadow_vs_main
@end

@fs shadow_fs
@include_block shadow_fs_main
@end

@vs shadow_nodes_vs
@include_block shadow_nodes_vs_main
@end

@fs shadow_nodes_fs
@include_block shadow_nodes_fs_main
@end

@vs clear_vs
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 1.0, 1.0);
}
@end

@fs clear_fs
layout(binding=0) uniform clear_params {
    vec4 clear_rgba;
};
out vec4 frag_color;
void main() {
    frag_color = clear_rgba;
}
@end

@program three vs fs
@program three_opaque vs fs_opaque
@program three_mrt vs fs_mrt
@program three_opaque_mrt vs fs_opaque_mrt
@program three_vsm vs fs_vsm
@program three_mrt_vsm vs fs_mrt_vsm
@program depth vs depth_fs
@program shadow shadow_vs shadow_fs
@program clear_rect clear_vs clear_fs
@program shadow_nodes shadow_nodes_vs shadow_nodes_fs
