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
| Scene & Cameras | ✅ | `Object3D`, `Group`, `Scene`, `PerspectiveCamera`, `OrthographicCamera` |
| Meshes & LOD | ✅ | `Mesh`, `InstancedMesh`, `Lod` |
| Lines | ✅ | `Line`, `LineSegments`, `LineLoop`, `LineBasic` and `LineDashed` materials |
| Geometry | ✅ | `box`, `plane`, `sphere`, `cylinder`, `cone`, `torus`, `lathe`, `buffer_geometry`, `merge` |
| Materials | ✅ | `MeshBasic`, `MeshLambert`, `MeshPhong`, `MeshStandard`, `MeshNormal`, parallax mapping (`parallaxUV`), shader hooks (`onBeforeCompile`) |
| Textures | ✅ | `load_texture` (PNG/JPEG), `data_texture`, `canvas_texture`, mipmaps, anisotropy |
| Lights | ✅ | `AmbientLight`, `HemisphereLight`, `DirectionalLight`, `PointLight`, `SpotLight` |
| Lighting addons | ✅ | Clustered forward lighting (`ClusteredLighting`), irradiance volume (`LightProbeGrid`) |
| Reflections | ✅ | Box-projected env maps, `CubeCamera` capture, SSR (`SSRNode`), planar `Reflector` |
| Fog | ✅ | `Fog`, `FogExp2` |
| Controls | ✅ | `OrbitControls` (`drie::controls`) |
| Raycasting | ✅ | `Raycaster` against `Mesh`, `InstancedMesh`, `Lod` (per-triangle) |
| Post-processing | ✅ | `EffectComposer`, `RenderPass`, `ShaderPass`, `output_pass`, `gtao_pass`, `taa_pass` |
| Renderer | ✅ | `render_target`, tone mapping, frustum culling, `renderer.info`, MRT, compute passes |
| Shadows | ⚠️ | `DirectionalLight` (CSM) and `SpotLight` (1 live + cached atlas); `PointLight` ❌ |
| Skeletons | ⚠️ | `Bone`, `Skeleton`, `SkinnedMesh` (CPU skinning); animation clips ❌ |
| glTF (`drie::loaders`) | ⚠️ | `.glb` meshes, materials, textures; skins & animations ❌ |
| Quaternions / Animation | ❌ | XYZ Euler rotations only; no animation clips |
| Sky | ✅ | `Sky` (`drie::objects`) |

## Differences from three.js

- Named arguments must be in declaration order (a C3 rule). For example, `color:` comes before `roughness:`.
- Colors are plain sRGB hex `uint`s, not `Color` objects.
- Rotations are XYZ Euler only (no quaternions or `rotation.order`).
- No animation clips. Point lights cast no shadows.
- Lines are a pixel wide (as in WebGL), cast no shadows, and are not hit by `Raycaster`.
- `Raycaster` collects hits into a list you pass in (three's `optionalTarget`) and gives
  `normal` in world space.
- `InstancedMesh` grows: `add_instance(matrix)` appends, `set_matrix_at(index, matrix)` changes one.
- `Lod` (three.js `LOD`) levels are added with `lod.add_level(object, distance)`, in increasing
  distance. A `null` object shows nothing from that distance on.
- Perspective cameras follow the window's aspect ratio automatically (`renderer.auto_aspect`).
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
| **Cached Shadow Atlas** | Static spot-light shadow caching in a shared atlas |
| **Volumetric Fog** | Compute-marched froxel fog with light scatter, sun shafts, 3D noise, and clear boxes |
| **Room Reflection Probes** | Box-projected, cross-faded room probes |
| **GPU Culling** | Depth pre-pass and GPU instanced culling |
| **Light Radius** | Softer close-range falloff on point and spot lights |
| **Material Extensions** | Foliage wind sway, glass viewer silhouette |
| **UI, Input & Tooling** | Menus, macOS GameController support, deterministic timeline rewind, profiling |

### Materials & Geometry

- **Packed ARM maps:** `material.arm_map` packs ambient occlusion, roughness, and metalness in R, G, and B.
- **Parallax occlusion mapping:** `material.parallax_map` with shadowing from the first directional light.
- **Foliage & wind sway:** `material.foliage` enables two-sided wrapped lighting for alpha-tested foliage, animated with `sway`, `sway_speed`, and `renderer.time`.
- **Glass & viewer silhouette:** `material.glass` preserves reflections regardless of opacity; `material.reflect_viewer` darkens glass reflections with a soft silhouette at the camera.
- **Additive blending & draw order:** `material.blending = ADDITIVE_BLENDING` for transparent materials and `object.render_order` for explicit draw order, as in three.js.
- **World-space box UVs:** `geometry.box_project_uv(tiles_per_unit)` generates box-projected UVs.

<details>
<summary>Material and geometry extension fields</summary>

| Feature | Key API / Fields | Notes |
| --- | --- | --- |
| Parallax | `material.parallax_map`, `parallax_scale`, `parallax_distance`, `parallax_shadow` | Shadowing uses the first directional light |
| Foliage | `material.foliage`, `material.sway`, `sway_speed`, `renderer.time`, `renderer.auto_time` | `auto_time` drives `sway` animation |

</details>

### Lighting, Shadows & Global Illumination

- **Light source radius:** `light.radius` softens close-range falloff on point and spot lights; `spot_light.specular_radius` controls reflection highlight size.
- **Cached static shadows:** `light.shadow.auto_update = false` (three.js `shadow.autoUpdate`) caches static shadow maps in the shared atlas (`renderer.shadow_atlas_capacity`).
- **Cascaded shadow maps:** `DirectionalLight` supports 1–4 texel-snapped cascades with blending and debug visualization.
- **Irradiance volume** (`irradiance.c3`): `scene.irradiance` (`IrradianceVolume`) probe grid for user-populated indirect bounce light.
- **Radiance Cascades GI** (`radiance.c3`): `scene.radiance` (`RadianceCascades`) computes real-time GPU GI and emissive bounce, replacing ambient/hemisphere lights and `scene.irradiance` inside its box.

<details>
<summary>Lighting, shadow, and GI extension API</summary>

| Feature | Key API / Fields | Notes |
| --- | --- | --- |
| Light radius | `light.radius`, `spot_light.specular_radius` | `specular_radius` defaults to `radius` when `0` |
| Static shadow atlas | `light.shadow.auto_update = false`, `light.shadow.needs_update = true` | Set `needs_update = true` to redraw |
| Cascaded shadows | `light.shadow.cascades` (`4`, 1..4), `shadow.max_distance` (`150`), `shadow.split_lambda` (`0.7`), `shadow.cascade_blend` (`0.1`), `renderer.debug_cascades` | With `cascades = 1`, uses `light.shadow_extent` and optional `light.shadow_follow = true` |
| Radiance Cascades | `drie::radiance_cascades(min, max)`, `rc.voxelize(scene)`, `rc.update(scene)` | Assigned to `scene.radiance` |

</details>

### Reflections

- **Environment maps:** Equirectangular reflections with mip-based roughness blur and optional box projection.
- **Environment capture:** `renderer.capture_environment(...)` renders a 6-face light probe into an equirectangular `env_map` (reads back from the GPU; intended for load time). This is three.js's `CubeCamera` + `PMREMGenerator`.
- **Room reflection probes:** `scene.reflection_probes` (`ReflectionProbes`) array texture of box-projected, cross-faded room probes for materials with `env_probes`.
- **Screen-space reflections (SSR):** Previous-frame color/depth reflections with optional back-face thickness.
- **Planar reflections:** `reflector(normal, constant)` renders planar reflections using an oblique near plane.

<details>
<summary>Reflections extension API</summary>

| Feature | Key API / Fields | Notes |
| --- | --- | --- |
| Environment map | `material.env_map`, `env_intensity`, `env_box_min`, `env_box_max`, `env_position` | — |
| Probe capture | `renderer.capture_environment(scene, position, on_face:, distance:)` | Optional `distance` raycaster callback stores surface distance in alpha for parallax-corrected glossy reflections |
| Room probes | `drie::reflection_probes()`, `probes.add(map, position, box_min, box_max)` | Used by materials with `env_probes` |
| SSR | `material.ssr`, `material.ssr_reach`, `renderer.set_reflection_source(color, depth, view_proj, back:)`, `renderer.render_back_faces(target)` | — |
| Planar reflector | `reflector(normal, constant)`, `renderer.render_reflector(reflector, scene, camera)`, `material.reflector` | Uses `camera.custom_projection`, optional `reflector.box_min`/`box_max`, and `renderer.shadow_map_auto_update = false` |

</details>

### Fog

- **Exponential fog:** `drie::fog_exp(color, density)` (three.js `FogExp2`).
- **Volumetric fog** (`fog.c3`): Compute-marched froxel grid supporting light scatter (`spot_light.fog_glow`), height falloff, sun shafts, 3D animated noise, and clear boxes.

<details>
<summary>Fog extension API</summary>

| Feature | Key API / Fields | Notes |
| --- | --- | --- |
| Volumetric fog | `drie::volumetric_fog(color, density)`, `resolution` (`0.125`), `slices` (`96`), `light_scatter`, `spot_light.fog_glow`, `height_*`, `sun_shafts`, `sun_direction`, `sun_color`, `noise_amount`, `noise_scale`, `noise_wind`, `fog.set_clear_box(min, max)`, `fog.clear_box_density` | Set `fog.volumetric = false` to disable |

</details>

### Renderer, Compute & Shaders

- **Clustered forward lighting** (`clusters.c3`): `16x9x24` view-space light clusters assigned via compute shader (requires GL 4.3+ on OpenGL).
- **Compute shaders** (`compute.c3`): Storage buffers, `@cs` compute passes, and async GPU-to-CPU readback.
- **GPU instance culling** (`culling.c3`, `renderer.gpu_culling`): Compute-based per-instance frustum culling for static `InstancedMesh`es (`>= 8` instances) across the main camera and live shadow passes.
- **Depth pre-pass:** `renderer.depth_prepass` for `alpha_test` and LOD cross-fade meshes.
- **Compute validation:** `renderer.check_compute` and `drie::print_compute_checks()` validate GPU cluster and culling results against CPU reference runs.
- **Render target & emissive extras:** `render_target.depth_texture`, `overlay_view` for post-resolve overlays, and alpha `0` for high-emissive light sources (`>= 16`) so post-processing can preserve blown-out lamps.
- **Headless & LOD scaling:** `renderer.init_headless()` (with `drie::commit()`) and `renderer.lod_scale` global multiplier for `Lod` distances.
- **Shader hooks:** Custom `.glsl` files can define `vertex_hooks` and `surface_hooks` around `@include shader_core.glsl`.

<details>
<summary>Renderer, compute, and shader hook extension API</summary>

| Area | Key API / Fields | Notes |
| --- | --- | --- |
| Clustering | `cluster_lights_gpu`, `cluster_lights_cpu`, `renderer.assign_clusters`, `renderer.max_lights` (`1024`, `0` = unlimited) | — |
| Compute | `drie::storage_buffer(size)` (`cpu_write: true`), `drie::compute_pass(...)`, `renderer.compute(pass, groups_x)` (or `begin_compute()` / `pass.dispatch()` / `end_compute()`), `drie::readback_slot()`, `buffer.track(slot)`, `drie::read_back(slot, out, bytes)` | — |
| Depth pre-pass | `renderer.depth_prepass`, `renderer.depth_prepass_pixels` (`1.5`) | Minimum pixels per triangle |
| Shader hooks | `vertex_hooks`, `surface_hooks`, `@include shader_core.glsl`, `renderer.shaders`, `material.custom`, `renderer.custom` | Accessed in GLSL as `material_custom[]` / `frame_custom[]` |

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

## Editing the shader

`shader_core.glsl` is the material shader; `shader.glsl` includes it with empty hooks. Run
`just shaders` (sokol-shdc, from sokol-tools) to regenerate `generated/shader.c3`, and with it
`kernels.c3` (the compute shaders, from `kernels.glsl`), `output.c3`, `gtao.c3` and `taa.c3` (the
post-processing passes) and `radiance_cs.c3`. An application with its own copy of the material
shader generates it into its own sources, and regenerates it whenever `shader_core.glsl` changes.

## Tests

- `just test reflections`: `tests/reflections.c3` renders a scene made for reflections from different angles
- `just test radiance`: `tests/radiance.c3` renders the same scene lit by its hemisphere light, by `RadianceCascades`,
and by them with only the glowing cubes

## Justfile

This repository uses [just](https://github.com/casey/just) as a task runner.

`just --list` to see all available commands.
