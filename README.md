# drie

A small three.js-style scene graph on top of [`sokol_gfx`](https://github.com/floooh/sokol-c3), written in [C3](https://c3-lang.org).

```c3
import drie;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
Mesh* cube;

fn void main()
{
    renderer = drie::renderer(antialias: true);
    scene = drie::scene();
    camera = drie::perspective_camera(75, drie::window_aspect(), 0.1, 1000);
    camera.position.z = 5;

    cube = drie::mesh(drie::box_geometry(1, 1, 1), drie::mesh_standard_material(color: 0x00ff00));
    scene.add(cube);
    scene.add(drie::ambient_light(0xffffff, 0.5));

    renderer.set_animation_loop(&animate);
}

fn void animate(double time)
{
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;
    renderer.render(scene, camera);
}
```

Add `drie` to your `project.json`:

```json
"dependencies": [
    "https://c3.ricar.do/github.com/floooh/sokol-c3:floooh/sokol",
    "https://c3.ricar.do/github.com:ricardobeat/drie"
]
```

<img width="506" height="344" alt="drie" src="https://github.com/user-attachments/assets/48b1cb90-3be4-40dd-8bb0-26cfa9bc8113" />

## three.js compatibility / JS bridge

[driejs](driejs/README.md) can run unmodified three.js code (`three.core.js`) using an embedded [boomkat](https://github.com/ricardobeat/boomkat) JS engine, with `drie` as the rendered backend. Scripts can import `three` and `three/addons/...` as they would in a browser.

It is a standalone project inside `driejs/` and is not compiled into applications that use `drie` as a library.

```sh
just driejs::build
just driejs::run demos/webgl_geometries.js
```

Here is three.js "Littlest Tokyo" example running unaltered (`demos/webgl_animation_keyframes.js`):

<img width="1392" height="860" alt="Screenshot 2026-11-17 at 19 13 26" src="https://github.com/user-attachments/assets/dacd8f64-0c9a-480a-a1e6-66f06517b23d" />

See [driejs/README.md](driejs/README.md) for more.

## Features

three.js API coverage:

| Area | Status | What's implemented |
| --- | --- | --- |
| Scene & Cameras | ✅ | `Object3D`, `Group`, `Scene` (background color, equirectangular texture with `backgroundBlurriness`, or a flat picture, `environment`), `PerspectiveCamera`, `OrthographicCamera` |
| Meshes & LOD | ✅ | `Mesh`, `InstancedMesh`, `Lod` |
| Lines | ✅ | `Line`, `LineSegments`, `LineLoop`, `LineBasic` and `LineDashed` materials |
| Geometry | ✅ | `box`, `plane`, `sphere`, `cylinder`, `cone`, `torus`, `lathe`, `buffer_geometry`, `merge` |
| Materials | ✅ | `MeshBasic`, `MeshLambert`, `MeshPhong`, `MeshStandard`, `MeshNormal`, `MeshPhysical` (transmission, volume, IOR, dispersion, specular, clearcoat, sheen, iridescence, anisotropy), tangents (a geometry's own, or from the screen derivatives), parallax mapping (`parallaxUV`), `depthTest` / `depthWrite`, shader hooks (`onBeforeCompile`) |
| Textures | ✅ | `load_texture` (PNG/JPEG), `data_texture`, `canvas_texture`, mipmaps, anisotropy, `repeat` / `offset` / `rotation` / `center` per map, `channel` (a geometry's second uv set, `uv1`), `nearest` (three's `NearestFilter` magnification), `min_filter` (three's `minFilter`, from `NEAREST` to `LINEAR_MIPMAP_LINEAR`), `mipmaps` (levels drawn by the application, three's `texture.mipmaps`) |
| Lights | ✅ | `AmbientLight`, `HemisphereLight`, `DirectionalLight`, `PointLight`, `SpotLight`, `RectAreaLight` (standard and physical materials, as three, with shadows besides: see below; up to `renderer.max_rect_area_lights`); light maps (`lightMap`, `lightMapIntensity`), `ProgressiveLightMap` |
| Lighting addons | ✅ | Clustered forward lighting (`ClusteredLighting`), irradiance volume (`LightProbeGrid`) |
| Reflections | ✅ | Box-projected env maps, `CubeCamera` capture, SSR (`SSRNode`), planar `Reflector` |
| Fog | ✅ | `Fog`, `FogExp2` |
| Controls | ✅ | `OrbitControls` (`drie::controls`) |
| Raycasting | ✅ | `Raycaster` against `Mesh`, `InstancedMesh`, `Lod` (per-triangle) |
| Post-processing | ✅ | `EffectComposer`, `RenderPass`, `ShaderPass`, `output_pass`, `gtao_pass`, `taa_pass` |
| Renderer | ✅ | `render_target`, tone mapping, frustum culling, `renderer.info`, MRT, compute passes |
| Shadows | ✅ | `DirectionalLight` (the first: CSM; every other one: a tile of the shadow atlas), `SpotLight` (1 live + cached atlas) and `PointLight` (six tiles of the atlas); PCF and VSM (`VSMShadowMap`); `alphaMap` and `alphaTest` cut holes in them |
| Skeletons | ⚠️ | `Bone`, `Skeleton`, `SkinnedMesh` (CPU skinning); animation clips ❌ |
| glTF (`drie::loaders`) | ⚠️ | `.glb` and `.gltf` on [cgltf](https://github.com/jkuhlmann/cgltf): meshes (sparse and quantized accessors, tangents, two uv sets), materials (with `KHR_materials_*` transmission, volume, ior, dispersion, specular, emissive_strength, clearcoat, sheen, iridescence, anisotropy, unlit), textures (embedded, files or `data:` URIs, their samplers, either uv set, `KHR_texture_transform`); skins, animations, morph targets, cameras, lights, Draco ❌ |
| Quaternions / Animation | ❌ | XYZ Euler rotations only; no animation clips |
| Sky | ✅ | `Sky` (`drie::objects`) |

## Differences from three.js

- Named arguments must be in declaration order (a C3 rule). For example, `color:` comes before `roughness:`.
- Colors are plain sRGB hex `uint`s, not `Color` objects.
- Rotations are XYZ Euler only (no quaternions or `rotation.order`).
- No animation clips.
- Shadow lookups are pushed off the surface along its normal (`shadow.normal_bias`, scaled by
  the slope and, for spot and point lights, the distance), with VSM too; `shadow.bias` is used by
  VSM only. A negative `normalBias` from three keeps drie's default.
- Only the first directional light gets cascades; every other shadowed one gets one box
  (`shadow_extent`) in a tile of the shadow atlas, redrawn every frame unless
  `shadow.auto_update` is off.
- `ProgressiveLightMap` keeps its lights in a scene of its own (`plm.scene`), not in the main
  scene, and bakes the objects where they are (three's moves them into a bake scene);
  call `plm.update(camera)` before `render()`.
- Lines are a pixel wide (as in WebGL), cast no shadows, and are not hit by `Raycaster`.
- `Raycaster` collects hits into a list you pass in (three's `optionalTarget`) and gives
  `normal` in world space.
- `InstancedMesh` grows: `add_instance(matrix)` appends, `set_matrix_at(index, matrix)` changes one.
  `set_color_at(index, color)` (three's `setColorAt`) tints one, times the material's color;
  instances with colors are drawn whole rather than culled on the GPU.
- A `Mesh` draws `draw_count` of its geometry's indices from `draw_start` when `draw_count` is
  above 0 (three's `drawRange`). Meshes that share a geometry and each draw a range of it with
  their own material are three's geometry groups with an array of materials.
- A mesh can be a billboard, turned toward the camera as it is drawn (`Mesh.billboard`, three's
  `Sprite`): `FACING` in the view's plane, with `billboard_center`, `billboard_rotation` and
  `billboard_fixed_size` (three's `sizeAttenuation` false); `KEEP_Y` and `KEEP_X` keep one of
  the mesh's own axes; `UPRIGHT` turns about the world's up toward the camera. A node
  material's `model_matrix()` is the mesh's `matrix_world` as it is, before it is turned.
- A material's maps share one sampler in the shader: the color map's, or else a repeating
  map's. A map that wraps otherwise has its wrapping done on its uv (`Texture.wrap_s`, `wrap_t`).
- `Lod` (three.js `LOD`) levels are added with `lod.add_level(object, distance)`, in increasing
  distance. A `null` object shows nothing from that distance on.
- Perspective cameras follow the window's aspect ratio automatically (`renderer.auto_aspect`),
  or the viewport's when one is set.
- Objects are allocated with the heap allocator and never freed. `geometry.dispose()` frees GPU buffers.
- Create objects outside of `@pool()` blocks. Their lists live on the heap, but anything you
  build with the temp allocator and hand to the library does not outlive the pool.
- Shaders are built for Metal, GL 4.3 and D3D11, but the library has only been tested on macOS/Metal. `./build.sh windows` cross-compiles a
  D3D11 build for Windows.

### Extensions

When used directly via C3, more advanced lighting features are available (see full list below).

A lot of these are in-development for three.js, see [Building the AAA Lighting Pipeline in Three.js](https://ben3d.ca/blog/building-the-aaa-lighting-pipeline-in-threejs).

*left: three.js, right: drie*

<img width="986" height="870" alt="image" src="https://github.com/user-attachments/assets/cae64b53-40e1-4fb5-97b2-e05e57f51ec5" />

Features not in three.js:

| Feature | Summary |
| --- | --- |
| **Radiance Cascades GI** | Real-time GPU global illumination and emissive bounce (`scene.radiance`) |
| **Cached Shadow Atlas** | Static spot-light shadow caching in a shared atlas, which also holds the boxes of extra shadowed directional lights |
| **Volumetric Fog** | Compute-marched froxel fog with light scatter, sun shafts, 3D noise, and clear boxes |
| **Room Reflection Probes** | Box-projected, cross-faded room probes |
| **GPU Culling** | Depth pre-pass and GPU instanced culling |
| **Light Radius & Tubes** | Point and spot lights with a size: softer close-range falloff, highlights the shape and size of the bulb; tube lights (`point_light.length`) |
| **Material Extensions** | Foliage wind sway, glass viewer silhouette, refraction with dispersion, clearcoat, sheen, iridescence |
| **UI, Input & Tooling** | Menus, macOS GameController support, deterministic timeline rewind, profiling |

### Materials & Geometry

- **Packed ARM maps:** `material.arm_map` packs ambient occlusion, roughness, and metalness in R, G, and B.
- **Parallax occlusion mapping:** `material.parallax_map` with shadowing from the first directional light.
- **Foliage & wind sway:** `material.foliage` enables two-sided wrapped lighting for alpha-tested foliage, animated with `sway`, `sway_speed`, and `renderer.time`.
- **Glass & viewer silhouette:** `material.glass` preserves reflections regardless of opacity; `material.reflect_viewer` darkens glass reflections with a soft silhouette at the camera.
- **Transmission & dispersion:** `material.transmission` shows what stands behind a surface, refracted through `thickness` meters of it at `ior` and blurred by its roughness; `dispersion` bends each color channel by its own index of refraction, fringing edges with rainbow colors, and `attenuation_color` / `attenuation_distance` tint the light as it crosses (Beer's law). `specular_color` / `specular_intensity` tint and scale what a dielectric reflects (KHR_materials_specular). Double-sided glass shows its own far side through the near one.
- **Clearcoat, sheen, iridescence & anisotropy:** `material.clearcoat` lays a thin clear varnish over the surface, with its own `clearcoat_roughness` and normal map (car paint, lacquer); `sheen` / `sheen_color` / `sheen_roughness` give cloth its soft glow at glancing angles (velvet, fabric); `iridescence` is a thin film (soap bubbles, oil) whose colors come from its thickness, `iridescence_thickness_min` to `_max` nanometers, and `iridescence_ior`; `anisotropy` draws highlights and reflections out along the surface's tangent, turned by `anisotropy_rotation` and `anisotropy_map` (brushed metal). They are three's `MeshPhysicalMaterial` layers, from its shader code.
- **Tangents:** normal and anisotropy maps are read in the geometry's own tangent frame (`geometry.tangents`, glTF's `TANGENT`), or without one in a frame from the screen derivatives of the position and uv. `normal_scale` is x and y, as three's `normalScale`: the glTF loader negates y on meshes without tangents, as three's GLTFLoader does. As three's `USE_TANGENT`, only a geometry with tangents whose material reads them is drawn with the programs that take them (the tangent costs every vertex drawn); a custom `MaterialShaders` hands them over as `main_tangent`, `main_mrt_tangent`, `main_vsm_tangent`, `main_mrt_vsm_tangent` and `depth_tangent`.
- **Matcap and toon materials:** `drie::mesh_matcap_material(matcap:)` (three's `MeshMatcapMaterial`: an image of a lit sphere read where the view-space normal points) and `mesh_toon_material(gradient_map:)` (three's `MeshToonMaterial`: light in steps, no highlights; a gradient map whose texture is `nearest` is read texel by texel).
- **Depth material:** `drie::mesh_depth_material(depth_packing:)` (three's `MeshDepthMaterial`): the window depth as a gray (white near) or packed over the channels (`RGBA_DEPTH_PACKING`, `RGB_DEPTH_PACKING`, `RG_DEPTH_PACKING`, as three's `packDepthToRGBA` and the rest).
- **Bump maps:** `material.bump_map` and `bump_scale` (three's `bumpMap`): the normal tilted by how the height changes from pixel to pixel.
- **Displacement maps:** `material.displacement_map`, `displacement_scale` and `displacement_bias` (three's `displacementMap`) move the vertices along their normals, in the main, depth and shadow passes. The map is read at each vertex on the CPU once (again when the geometry or the map's placement changes) into a vertex attribute, so changing the scale or bias costs nothing.
- **Outlines:** `material.outline` pushes the vertices out along their normals by that much of the view, so a back-side material with it, drawn over a mesh, outlines it (three's `OutlineEffect`).
- **Alpha and emissive maps:** `material.alpha_map` (green: how opaque) and `emissive_map` (tints `emissive`).
- **Depth test & write:** `material.depth_test` / `depth_write` (three.js `depthTest` / `depthWrite`); a material that isn't tested or doesn't write depth stays out of the depth pre-pass.
- **Blending & draw order:** `material.blending` is three's: `NORMAL_BLENDING` (for transparent materials), `ADDITIVE_BLENDING`, `SUBTRACTIVE_BLENDING`, `MULTIPLY_BLENDING`, `NO_BLENDING`, or `CUSTOM_BLENDING` with `blend_equation`, `blend_src`, `blend_dst` and their `_alpha` ones; `premultiplied_alpha` multiplies the colors by their opacity. `object.render_order` sets an explicit draw order, as in three.js. Each blend state is a pipeline of its own.
- **Tone mapping per material:** `material.tone_mapped = false` leaves its colors out of the renderer's tone mapping (three's `toneMapped`).
- **Flat backgrounds:** `scene.background_flat` shows `background_texture` as a picture over the whole view, placed by its `repeat` and `offset` (three's `scene.background` as a plain texture).
- **Background materials:** `scene.background_material` is drawn behind everything on a sphere around the camera, a node material that reads the view's direction from `normal_geometry` (three's `scene.backgroundNode`).
- **World-space box UVs:** `geometry.box_project_uv(tiles_per_unit)` generates box-projected UVs.
- **Node materials** (`nodes.c3`, `nodes_tsl.c3`): a material's color, opacity, normal, roughness, metalness, glow, vertex positions and final color can come from graphs of nodes built at runtime, as three's TSL and `NodeMaterial` do (`colorNode`, `positionNode`...). drie prints each graph in the backend's own language (Metal, HLSL or GLSL) into the material's generated program, where it replaces the step it stands for (see **Generated programs**), so there is no shader compiler in drie: the driver compiles the result, as the browser does with three's WGSL. Uniforms change with `Node.set` without a rebuild. driejs runs three's real TSL on top of it (`three/webgpu`, `three/tsl`). Graphs can also be written as text: `nodes::snippet("mix(material_color.rgb, tint, smoothstep(0.0, 1.0, uv.y))", { { "tint", tint } })` (`nodes_snippet.c3`) parses GLSL-style expressions with `let`s, using drie::nodes' names, into the same nodes; a mistake fails with `SNIPPET_ERROR` and `nodes::snippet_error()` gives its line and column (`just test nodes_snippet`). `just example nodes` (`examples/nodes.c3`) burns a torus away with noise and a discard, its shadow with it, twists a column with `position_node`, and swaps a ball's `color_node` while it runs (keys 1 to 4).

```c3
Node wave = nodes::sin(nodes::position_local().y() * 8 + nodes::time() * 2);
mat.color_node = nodes::mix(nodes::color(0x2266ff), nodes::color(0xffffff), wave * 0.5f + 0.5f);
mat.position_node = nodes::position_local() + nodes::normal_local() * (wave * 0.05f);
```

<details>
<summary>Material and geometry extension fields</summary>

| Feature | Key API / Fields | Notes |
| --- | --- | --- |
| Parallax | `material.parallax_map`, `parallax_scale`, `parallax_distance`, `parallax_shadow` | Shadowing uses the first directional light |
| Foliage | `material.foliage`, `material.sway`, `sway_speed`, `renderer.time`, `renderer.auto_time` | `auto_time` drives `sway` animation |
| Transmission & dispersion | `material.transmission` (+ `transmission_map`), `thickness` (m, + `thickness_map`), `ior`, `dispersion`, `attenuation_color`, `attenuation_distance`, `specular_color` / `specular_intensity` (+ maps) | Rendering into a linear (`RGBA16F`) target, the main pass's own picture of the opaque scene is copied halfway through it for the glass to refract (as engines that draw the scene once do): a copy and its mip levels (for rough glass), and a second copy for the back faces of double-sided glass. Into a tone mapped target, the opaque scene is drawn again for it first, at one sample, as three does; loaded from `KHR_materials_transmission` / `_volume` / `_ior` / `_dispersion` / `_specular`; `just test dispersion`, `just test transmission` |
| Clearcoat | `material.clearcoat`, `clearcoat_roughness` (+ `clearcoat_map`, `clearcoat_roughness_map`), `clearcoat_normal_map`, `clearcoat_normal_scale` | Loaded from `KHR_materials_clearcoat`; `just test physical` |
| Sheen | `material.sheen`, `sheen_color` (+ `sheen_color_map`), `sheen_roughness` (+ `sheen_roughness_map`) | Loaded from `KHR_materials_sheen` |
| Iridescence | `material.iridescence` (+ `iridescence_map`), `iridescence_ior`, `iridescence_thickness_min` / `_max` (nm, + `iridescence_thickness_map`) | Loaded from `KHR_materials_iridescence` |
| Anisotropy | `material.anisotropy`, `anisotropy_rotation` (radians), `anisotropy_map` (rg = direction, b = strength) | Along the tangent: the geometry's `tangents` (glTF's `TANGENT`), else the uv's u from screen derivatives; loaded from `KHR_materials_anisotropy`; `just test anisotropy` |

| Node materials | `material.color_node`, `opacity_node`, `normal_node` (world space), `roughness_node`, `metalness_node`, `emissive_node`, `position_node` (in the model; also in the depth pre-pass and the shadow maps), `output_node` (rgba after lighting, before fog; `nodes::output()`), `mask_node` (a bool: false discards, in every pass), `env_node` (rgb the environment reflects in `nodes::env_direction()` at `nodes::env_roughness()`, in place of the environment map's reflection), `specular_node` and `shininess_node` (phong), `clearcoat_node` and `clearcoat_roughness_node` (standard; either turns the clear coat on) | Constructors in `drie::nodes`, named as TSL's: values, `uniform*` (`Node.set`; one uniform can be in many materials' graphs), builtins (`uv()`, `position_world()`, `time()`, `instance_matrix()` in `position_node`...), the material's own value at the step to change it: `material_color()` (color_node, opacity_node), `material_roughness()`, `material_metalness()` (roughness_node, metalness_node), `env_color()` (env_node); `texture(t, uv, level)`, `texture_grad(t, uv, dx, dy)`, `texture_size(t)`, with `t` a Texture or `material_map(NORMAL_MAP, CUSTOM_MAP...)` for each material's own; the screen's depth behind a surface with `Material.ssr` and a reflection source: `ssr_on()`, `screen_position(p)` (uv, distance from the camera's plane), `viewport_depth(uv)`; `varying(x)`, math, swizzles, `select`; ints and uints with bitwise operators; statements: `variable`, `Node.assign`, `if_then`, `loop`, `break_`, `discard`, `block`, `seq`; functions: `nodes::function(name, ret, params...)` returns a `NodeFunction`, its body set with `.returns(value)` over `.param(i)` and called with `.call(args...)`, printed once per stage however often it is called (its body reads parameters, uniforms, textures and other functions; pass builtins in as arguments). A discard in `color_node` (or `mask_node`, or alpha under `alpha_test`) cuts the shadow too. Programs are built on first draw and shared by materials whose graphs print the same; with an application's material shader (`renderer.shaders`) the graphs go into its hooks instead, with up to 32 vec4s of uniforms, four vec4 varyings, and textures in the fragment stage only, and no functions; `just test nodes`, `just test node_inputs`, `just example nodes`, `just test render_environment` (env_node) |

A material's maps besides `map`, `normal_map` and `arm_map` share six texture slots: up to six distinct textures among them. A node material's textures take slots from the same six.

</details>

### Lighting, Shadows & Global Illumination

- **Light source radius:** `light.radius` softens close-range falloff on point and spot lights, and their highlights are the bulb's own shape (Karis' representative point with its energy normalization): a smooth surface mirrors a sharp disc as big as the bulb, a rough one gets the same light as from a point; `spot_light.specular_radius` sets the size seen in reflections. `just test lightsize`
- **Tube lights:** `point_light.length` stretches a point light into a tube along its own x axis (`radius` thick): light from the whole segment (Karis' line irradiance) and long highlights of the tube's shape. Its shadow is cast from its centre.
- **Point light shadows:** `point_light.cast_shadow` (three's `PointLightShadow`) draws six maps around the light, one atlas tile each (`shadow.map_size`, `shadow.near`, `shadow.far`: `0` for the light's `distance`, or 50), every frame or, with `shadow.auto_update` off, once. `just test pointshadows`
- **Rect area light shadows:** `rect_area_light.cast_shadow` (three's `RectAreaLight` has none) shadows its light and its highlights as a point light's six maps from its centre (`rect_area_light.shadow`, as `PointLight.shadow`; `shadow.softness` about the rectangle's size for soft edges): a doorway's glint on a glossy floor stops where a curtain hangs in front of it. `just test rect_shadows`
- **Cached static shadows:** `light.shadow.auto_update = false` (three.js `shadow.autoUpdate`) caches static shadow maps in the shared atlas (`renderer.shadow_atlas_capacity`).
- **Variance shadow maps:** `renderer.shadow_type = VSM` (three.js `VSMShadowMap`; default `PCF`, three's `PCFSoftShadowMap`) blurs each spot and directional map into its moments (`shadow.radius`, `shadow.blur_samples`, `shadow.bias`), in RG32F where the GPU can draw and filter it, else RG16F.
- **Several shadowed directional lights:** as in three.js, every directional light can cast a shadow: the first is the cascaded sun, each other one a box (`light.shadow_extent`) in a tile of the shadow atlas.
- **Light maps:** `material.light_map` and `light_map_intensity` (three.js `lightMap`, `lightMapIntensity`) add baked light to the indirect irradiance, read from the uv set of the texture's `channel` (`1`: the geometry's `uv1`).
- **Progressive light maps** (`lightmap.c3`): `drie::progressive_lightmap(renderer, res)` (three.js `ProgressiveLightMap`) bakes light into the objects' `uv1` a little at a time, from lights that may move (jittered, they give soft shadows); `renderer.render_bake` draws the samples.
- **Cascaded shadow maps:** `DirectionalLight` supports 1–4 texel-snapped cascades with blending and debug visualization.
- **Irradiance volume** (`irradiance.c3`): `scene.irradiance` (`IrradianceVolume`), a probe grid of indirect bounce light, filled by the application or by `renderer.bake_irradiance(scene, volume)` (three's `LightProbeGrid.bake`: six small faces drawn around each probe, optionally with bounces); `irradiance_grid(min, max, nx, ny, nz)` makes one by probe counts. `scene.irradiance_volumes` adds more (up to `MAX_IRRADIANCE_VOLUMES` in all), each lighting its own box in front of the ones before it and fading out over half a cell outside it, so a volume the size of a room lights its walls, floor and ceiling fully.
- **Radiance Cascades GI** (`radiance.c3`): `scene.radiance` (`RadianceCascades`) computes real-time GPU GI and emissive bounce, replacing ambient/hemisphere lights and `scene.irradiance` inside its box.

<details>
<summary>Lighting, shadow, and GI extension API</summary>

| Feature | Key API / Fields | Notes |
| --- | --- | --- |
| Light radius | `light.radius`, `spot_light.specular_radius` | `specular_radius` defaults to `radius` when `0` |
| Lighting channels | `light.channels`, `material.light_channels` | Bits: a directional, point, spot or rect area light lights the materials that take every light (`light_channels` 0, the default) and those that share a channel with it (three's `lightsNode`, Unreal's lighting channels). Ambient and hemisphere light light everything. `just test light_channels` |
| Tube lights | `point_light.length` (m, along the light's x axis) | Intensity spread along the tube; falls off as the square of the distance whatever `decay`; `distance` cuts off from its nearest point |
| Static shadow atlas | `light.shadow.auto_update = false`, `light.shadow.needs_update = true` | Set `needs_update = true` to redraw |
| Shadow filtering | `renderer.shadow_type` (`PCF`, `VSM`), `light.shadow.radius` (`2`), `shadow.blur_samples` (`8`, VSM's blur taps), `shadow.bias` (`0`, VSM only; PCF uses `shadow.normal_bias`) | |
| Shadow atlas capacity | `renderer.shadow_atlas_capacity` (`8`) | Tiles the atlas has room for at first: one per static spot light (`shadow.auto_update = false`), per extra dynamic spot/directional light besides the primary ones, and six per shadowed point light; grows to what they need, up to 64 |
| Light maps | `material.light_map` (`texture.channel = 1` for `uv1`), `material.light_map_intensity` | For `BASIC`, the map replaces the indirect light of 1 (as map / PI) |
| Progressive light map | `drie::progressive_lightmap(renderer, res)` (`1024`), `plm.scene.add(light)`, `plm.add_objects(objects, pack:)`, `plm.update(camera, blend_window:, blur_edges:)` (`100`, `true`), `plm.texture(i)`, `plm.show_debug(visible)`, `plm.dispose()` | `add_objects` sets `light_map = texture(1)`, shadows on, and packs their `uv` into `uv1` (potpack; `pack: false` keeps your `uv1`); lights passed to it go to `plm.scene`; call `update` before `render()` |
| Cascaded shadows | `light.shadow.cascades` (`4`, 1..4), `shadow.max_distance` (`150`), `shadow.split_lambda` (`0.7`), `shadow.cascade_blend` (`0.1`), `renderer.debug_cascades` | With `cascades = 1`, uses `light.shadow_extent` and optional `light.shadow_follow = true` |
| Radiance Cascades | `drie::radiance_cascades(min, max)`, `rc.voxelize(scene)`, `rc.update(scene)` | Assigned to `scene.radiance` |

</details>

### Reflections

- **Environment maps:** Equirectangular reflections with mip-based roughness blur (as wide as three's PMREM at each roughness, sampled towards the GGX lobe's dominant direction) and optional box projection.
- **Environment light:** a standard material's diffuse light from `scene.environment` (or its own map) is the map's spherical harmonics, worked out at the upload, with the map's sun (a small spot far brighter than the rest) taken out of them and lit as a light from its direction: within a few percent of the exact cosine integral for any normal, where the harmonics alone spread a low sun onto the ground's underside. A directional light coming from where the map's sun is takes its place (with its shadow). `just test envlight`
- **Cube maps:** `drie::cube_texture(...)` and `decode_cube_texture(...)` turn a cube map's six faces (floats, or 8-bit images as three's `CubeTextureLoader` gives them) into drie's equirectangular map, looked up as three looks up a `CubeTexture`.
- **Classic environment maps:** on basic, lambert and phong materials, `env_map` is three's `envMap`: mirrored or (`env_refraction`, `refraction_ratio`) seen through, sharp, and put over the light by `combine` (`MULTIPLY_OPERATION`, `MIX_OPERATION`, `ADD_OPERATION`) and `reflectivity`, times `specular_map`'s red.
- **Environment capture:** `renderer.capture_environment(...)` renders a 6-face light probe into an equirectangular `env_map` (reads back from the GPU; intended for load time). This is three.js's `CubeCamera` + `PMREMGenerator`.
- **Room reflection probes:** `scene.reflection_probes` (`ReflectionProbes`) array texture of box-projected, cross-faded room probes for materials with `env_probes`.
- **Screen-space reflections (SSR):** Previous-frame color/depth reflections with optional back-face thickness.
- **Planar reflections:** `reflector(normal, constant)` renders planar reflections using an oblique near plane.

<details>
<summary>Reflections extension API</summary>

| Feature | Key API / Fields | Notes |
| --- | --- | --- |
| Environment map | `material.env_map`, `env_intensity`, `env_box_min`, `env_box_max`, `env_position` | — |
| Classic env map | `material.env_map`, `combine`, `reflectivity`, `env_refraction`, `refraction_ratio`, `specular_map` | Basic, lambert and phong materials (three's `envMap`) |
| Probe capture | `renderer.capture_environment(scene, position, on_face:, distance:)` | Optional `distance` raycaster callback stores surface distance in alpha for parallax-corrected glossy reflections |
| Environment from a material | `renderer.render_environment(material, into, width:, diffuse:)` | A node material drawn into an equirectangular map on the GPU (with mip levels), each texel the graph in its direction (`normal_geometry`): three's `scene.environmentNode`, worked out once per texel rather than at every pixel it lights. Drawn again into `into` when the graph changes; `diffuse` reads a 64 x 32 drawing back for its light on surfaces (waits for the GPU). `just test render_environment` |
| Probe | `renderer.render_probe(scene, position, into, face_size:, near:, far:, diffuse:)` | The scene drawn in every direction from a point into an equirectangular map on the GPU: six faces, resampled by `render_environment` (three's `CubeCamera.update`). Drawn again into `into` for a moving scene; `diffuse` as `render_environment`'s. `EnvLight.harmonics()` gives the map's light on surfaces as nine coefficients, the sun in them (three's `LightProbe`). `just test render_probe` |
| Room probes | `drie::reflection_probes()`, `probes.add(map, position, box_min, box_max)` | Used by materials with `env_probes` |
| SSR | `material.ssr`, `material.ssr_reach`, `renderer.set_reflection_source(color, depth, view_proj, back:)`, `renderer.render_back_faces(target)` | — |
| Planar reflector | `reflector(normal, constant)`, `renderer.render_reflector(reflector, scene, camera)`, `material.reflector` | Uses `camera.custom_projection`, optional `reflector.box_min`/`box_max`, and `renderer.shadow_map_auto_update = false` |

</details>

### Fog

- **Exponential fog:** `drie::fog_exp(color, density)` (three.js `FogExp2`).
- **three's fog and background:** `renderer.display_fog` mixes the fog in after tone mapping, in the colors that are shown, and clears to the background color without tone mapping it, as three does. Off, both are tone mapped with the rest of the light.
- **Volumetric fog** (`fog.c3`): Compute-marched froxel grid supporting light scatter (`spot_light.fog_glow`), height falloff, sun shafts, 3D animated noise, and clear boxes.

<details>
<summary>Fog extension API</summary>

| Feature | Key API / Fields | Notes |
| --- | --- | --- |
| Volumetric fog | `drie::volumetric_fog(color, density)`, `resolution` (`0.25`), `slices` (`96`), `light_scatter`, `spot_light.fog_glow`, `height_*`, `sun_shafts`, `sun_direction`, `sun_color`, `noise_amount`, `noise_scale`, `noise_wind`, `fog.set_clear_box(min, max)`, `fog.clear_box_density` | Set `fog.volumetric = false` to disable |

</details>

### Renderer, Compute & Shaders

- **Clustered forward lighting** (`clusters.c3`): `16x9x24` view-space light clusters assigned via compute shader (requires GL 4.3+ on OpenGL).
- **Compute shaders** (`compute.c3`): Storage buffers, `@cs` compute passes, and async GPU-to-CPU readback.
- **GPU instance culling** (`culling.c3`, `renderer.gpu_culling`): Compute-based per-instance frustum culling for static `InstancedMesh`es (`>= 8` instances) across the main camera and live shadow passes.
- **Depth pre-pass:** `renderer.depth_prepass` for `alpha_test` and LOD cross-fade meshes.
- **Compute validation:** `renderer.check_compute` and `drie::print_compute_checks()` validate GPU cluster and culling results against CPU reference runs.
- **Render target & emissive extras:** `render_target.depth_texture`, `overlay_view` for post-resolve overlays, and alpha `0` for high-emissive light sources (`>= 16`) so post-processing can preserve blown-out lamps.
- **Viewports, scissor and clearing:** `renderer.viewport` (`set_viewport`), `renderer.scissor` (`set_scissor`) with `scissor_test`, `auto_clear` and `renderer.clear()`, as three's `setViewport`, `setScissor`, `setScissorTest`, `autoClear` and `clear()`: several renders can share one frame, each in its own part of the window or target, in pixels from the bottom left. A background color clears the scissor box (or all of the target) even with `auto_clear` off, as in three. Several `render()` calls to the window inside the animation loop all go into the same frame, which is shown when the loop's callback returns.
- **Headless & LOD scaling:** `renderer.init_headless()` (with `drie::commit()`) and `renderer.lod_scale` global multiplier for `Lod` distances.
- **Generated programs** (`materials_nodes.c3`, `shading_nodes.c3`, `nodes_program.c3`): every material is a node graph, as three's `NodeMaterial`s are, built from what the material and the render use (its maps, lighting model, the kinds of lights present, shadow type, fog, tone mapping, the target...) and printed into a program for the backend at runtime. A flat textured quad gets a program of a few KB, a physical material with every layer one of tens of KB; materials that print the same share one (node materials too: those whose graphs print the same text, reading their own uniforms and textures in the same places, share a program and its pipelines, and each draw hands it the material's own). A value a graph uses in several places is printed once where all of them can read it (one expression printed for two nodes is worked out once, and what both branches of an `if` read is worked out before it). Light counts are uniforms, so adding or moving lights of a kind already in the scene builds nothing. Metal is the backend that is run and checked; the HLSL and GLSL printers are written but not yet compiled by anyone, so D3D11 and OpenGL are unverified.
- **An application's material shader** (`renderer.shaders`): an application can still build its own copy of drie's uber shader, `shader_core.glsl` with its hooks (`vertex_hooks`, `surface_hooks`) filled in, compile it with sokol-shdc and hand the programs over; every material then draws with them, and node materials' graphs go into the hooks. house does this. The uber shader keeps its current features: new ones go into the generated programs only. `just test classic_shaders`.
- **Pipeline cache** (`pipeline_cache.c3`): like Unreal's PSO cache. drie lists every render pipeline it makes (the engine's and node materials', with the state and target they draw with): save the list from a run that plays the game, load it at the start of the next, and the pipelines compile on worker threads, a few at a time, while frames go on, with `renderer.compiling()` for a loading bar. The list is text and holds no compiled code (a generated program goes in as its material key; with `renderer.shaders`, a node material's as its hooks' text), so it doesn't go stale with the driver or GPU, it only gets less complete. Metal only; elsewhere loading does nothing.
- **Programs at once** (`renderer.max_programs`, `1024`): how many programs (sokol shaders) can exist at once, the generated ones and the passes'; set it before the renderer starts. A game whose node materials print many different programs raises it (house's 199 test views take about 750).
- **Compiling in the background:** a render starts compiling the programs its draws take on worker threads, all at once, and each draw waits only for its own. With `renderer.compile_in_background` (off by default) a node material's meshes are left out of the frame until their programs are ready, rather than the frame waiting.
- **Programs are made when a draw first takes one**, not at startup. A render starts compiling, all at once on worker threads (Metal), the ones its draws, background and passes will take, and a draw waits only for its own. After that the driver's cache has them. With `renderer.shaders`, its programs (main, opaque, depth, shadow, and their MRT, VSM and tangent variants: about 180 KB of Metal each, two to three seconds each for the driver the first time) are made the same way; set it before the first render. A program it doesn't give aborts when a draw needs it, except the opaque ones (main stands in), depth and shadow (drie's own).

<details>
<summary>Renderer, compute, and shader hook extension API</summary>

| Area | Key API / Fields | Notes |
| --- | --- | --- |
| Clustering | `cluster_lights_gpu`, `cluster_lights_cpu`, `renderer.assign_clusters`, `renderer.max_lights` (`1024`, `0` = unlimited) | — |
| Compute | `drie::storage_buffer(size)` (`cpu_write: true`), `drie::compute_pass(...)`, `renderer.compute(pass, groups_x)` (or `begin_compute()` / `pass.dispatch()` / `end_compute()`), `drie::readback_slot()`, `buffer.track(slot)`, `drie::read_back(slot, out, bytes)` | — |
| Depth pre-pass | `renderer.depth_prepass`, `renderer.depth_prepass_pixels` (`1.5`) | Minimum pixels per triangle |
| Application's material shader | `vertex_hooks`, `surface_hooks`, `@include shader_core.glsl`, `renderer.shaders` (`MaterialShaders`), `material.custom`, `renderer.custom` | Accessed in GLSL as `material_custom[]` / `frame_custom[]`; see `shader.glsl` for the hooks |
| Pipeline cache | `renderer.pipeline_cache(allocator)` (the text), `save_pipeline_cache(path)`, `load_pipeline_cache(text)` / `load_pipeline_cache_file(path)`, `renderer.compiling()` (`{ still compiling, all }`), `renderer.screen_stand_in` | A pipeline drawn into the window is listed as the window's and compiled for whatever window loads it; a headless run that renders for one names the target that stands for it (`screen_stand_in`) |
| Background compiles | `renderer.compile_in_background` (`false`) | Node materials' meshes skip frames until their programs are compiled |
| Programs at once | `renderer.max_programs` (`0`: `drie::DEFAULT_PROGRAMS`, 1024) | sokol's shader pool; read when the renderer starts |

</details>

### UI, Input & Utilities

- **Menus & UI** (`drie::ui`): Escape-toggled menu, screen fade, setup, and touch controls.
- **Game controllers** (`drie::input`, macOS GameController): Polled every frame; d-pad, cross/A, circle/B, and options also emit arrow, enter, backspace, and escape key events by default.
- **Rewind timeline** (`drie::timeline`): `Timeline{Input, State}` records per-frame inputs and periodic state snapshots for deterministic `seek()` replay.
- **Profiling** (`drie::profile`): Frame section timing and reporting.
- **Application hooks:** `drie::add_frame_hook`, `add_event_filter`, and `add_overlay`.

<details>
<summary>UI, input, timeline, and profiling API</summary>

| Area | Key API / Fields | Notes |
| --- | --- | --- |
| UI | `ui::menu(title, handler)`, `ui::screen_fade`, `ui::setup()`, `ui::touch_controls()` | Captures keyboard/mouse when open; sends `OPEN`, `CLOSE`, `SELECT`, `CONFIRM`, `BACK`; `overlay = true` for built-in rendering or draw manually |
| Gamepad | `input::gamepad()`, `PAD_*` (`pressed()` / `released()`), sticks, triggers | — |
| Timeline | `Timeline{Input, State}`, `seek()` | Requires a deterministic step function |
| Profiling | `profile::enabled = true`, `profile::start()`, `profile::stop(name, t)`, `profile::report(frames)` | — |

</details>

## Performance

TBD. Currently the demo, with all advanced features on, runs at 40-60FPS on a Macbook Air M3.

## Editing the shaders

The materials' programs are written in C3, as node graphs: the material's steps in
`materials_nodes.c3`, the shading functions (BRDFs, lights, shadows, fog, tone mapping) in
`shading_nodes.c3`, and the printers for Metal, HLSL and GLSL in `nodes_program.c3`. They need
no build step. `DRIE_TRACE_PROGRAMS=1` prints each program as it is built.

`shader_core.glsl` is the uber shader that an application builds with its hooks
(`renderer.shaders`); `shader.glsl` includes it with empty hooks, which checks that it still
compiles and gives drie the structs and slots its bindings use. Run `just shaders` (sokol-shdc,
from sokol-tools) to regenerate `generated/shader.c3`, and with it `kernels.c3` (the compute
shaders, from `kernels.glsl`), `output.c3`, `gtao.c3` and `taa.c3` (the post-processing passes),
`radiance_cs.c3`, `vsm.c3` (the variance shadow maps' blur) and `lightmap.c3` (the progressive
light map's seams pass); the recipe then fixes sokol-shdc's storage image slot constants with
macOS `sed -i ''`. An application with its own copy of the uber shader generates it into its
own sources, and regenerates it whenever `shader_core.glsl` changes.

## RectAreaLight's tables

`generated/ltc.c3` holds the linearly transformed cosines `RectAreaLight` is shaded with (Heitz,
Dupuy, Hill and Neubelt 2016, [ltc_code](https://github.com/selfshadow/ltc_code)), taken from
three.js's `RectAreaLightTexturesLib.js` (MIT) by `just ltc` (`third_party/ltc/ltc.py`). The shader
has no view slot left for textures of their own: a frame with rect area lights appends the tables
(192 KB) to the point lights' buffer, and the shader filters them itself.

## cgltf

The glTF loader (`gltf.c3`) reads files with [cgltf](https://github.com/jkuhlmann/cgltf) 1.15
(MIT), vendored as `third_party/cgltf/cgltf.h` and compiled by `platform/cgltf.c`. Its C3
binding, `generated/cgltf.c3`, is made with [bindgen.c3l](../bindgen.c3l) (`just bindings`;
`just bindgen=path bindings` for a checkout elsewhere), which also checks that every struct of
the binding has C's size, alignment and field offsets (`third_party/cgltf/layout.py`).

## Tests

- `just test reflections`: `tests/reflections.c3` renders a scene made for reflections from different angles
- `just test radiance`: `tests/radiance.c3` renders the same scene lit by its hemisphere light, by `RadianceCascades`,
and by them with only the glowing cubes
- `just test vsm`: `tests/vsm.c3` renders three's `webgl_shadowmap_vsm` scene (a torus for the knot) with PCF and with variance shadow maps
- `just test lightmap`: `tests/lightmap.c3` checks light maps on `BASIC` and Phong materials, and a map on `uv1`
- `just test shadows`: `tests/shadows.c3` lights a box with two shadowed directional lights from either side
- `just test bake`: `tests/bake.c3` bakes a floor and a box with `ProgressiveLightMap`, with PCF and VSM: lit floor, dark shadow, soft edges from a jittered light, the seams' pass
- `just test gltf`: `tests/gltf.c3` loads Khronos's TextureTransformTest (`tests/assets`, CC0) and renders its
`KHR_texture_transform` checks: every arrow should point at a green mark
- `just test physical`: `tests/physical.c3` renders iridescence, a clear coat, sheen and double-sided glass side by side
- `just test anisotropy`: `tests/anisotropy.c3` loads Khronos's AnisotropyRotationTest (CC-BY 4.0): every band's highlight
should be a horizontal stripe, whether its direction comes from the tangents, `anisotropy_rotation` or the map
- `just test lightsize`: `tests/lightsize.c3` lights a glossy floor and two balls with bulbs of radius 0, 0.05 and 0.3 m and tubes 0, 0.5 and 1.5 m long: each highlight on the floor should be the bulb's or the tube's shape at its size, and the rough ball should look about the same under all of them
- `just test nodes`: `tests/nodes.c3` draws five balls with node materials (a texture times a uniform, stripes, rings pushed out by `position_node` through the depth pre-pass and the shadow maps, a `normal_node` with a fresnel glow, a varying), changes a uniform between two renders, renders with and without the pre-pass, with VSM and into MRT, checks a loop, a break and integer arithmetic against a constant, and discards a ball by a uniform, its shadow with it
- `just test pipeline_cache`: `tests/pipeline_cache.c3` checks that a render lists its engine and node pipelines, that loading the list again adds none, and that a node program no graph has printed yet compiles in the background until `compiling()` says none are left
- `just test normals`: `tests/normals.c3` loads Khronos's NormalTangentTest (CC0, no tangents) and
NormalTangentMirrorTest (CC-BY 4.0, tangents): each normal-mapped cell should reflect the horizon as the real geometry beside it does

## Justfile

This repository uses [just](https://github.com/casey/just) as a task runner.

`just --list` to see all available commands.
