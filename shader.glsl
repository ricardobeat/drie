// Uber-shader for three's built-in materials, plus the depth-only shadow pass: shader_core.glsl,
// with hooks that do nothing. Regenerate shader.c3 with: ./build.sh shaders
//
// An application with surfaces of its own compiles its own copy of the shader: a .glsl file
// with its own @header (its module) and the two blocks below, then `@include` of
// shader_core.glsl. It hands the result to the renderer as Renderer.shaders. The hooks run
// inside the engine's shaders and read Material.custom (`material_custom[2]`) and
// Renderer.custom (`frame_custom[2]`):
//
//   vertex_hook(world, local, model)  in the main, depth and shadow passes: moves a vertex.
//       `world` is its position in the world after sway, `local` the one in the instance
//       (before `model`); the vertex attributes (color0...) and the instance's matrix rows
//       (m0, m1, m2) can be read too.
//   surface_hook(geo_n, base, rough, metal)  for lit materials, after the textures: changes
//       the color (rgb, a = opacity), roughness and metalness. `geo_n` is the normal before
//       the normal map; v_world_pos, v_uv, camera_pos... are there to read.
//   reflection_hook(dir, sky)  changes what an environment map (or glass's sky) reflects
//       along `dir`.
@header module three;

@ctype mat4 Mat4
@ctype vec4 Vec4

@block vertex_hooks
void vertex_hook(inout vec3 world, vec3 local, mat4 model) {}
@end

@block surface_hooks
void surface_hook(vec3 geo_n, inout vec4 base, inout float rough, inout float metal) {}
void reflection_hook(vec3 dir, inout vec3 sky) {}
@end

@include shader_core.glsl
