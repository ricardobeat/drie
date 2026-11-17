# three.c3

A small three.js-style scene graph on top of sokol, written in C3.

```c3
import three;

Renderer* renderer;
Scene* scene;
PerspectiveCamera* camera;
Mesh* cube;

fn void main()
{
    renderer = three::webgl_renderer(antialias: true);
    scene = three::scene();
    camera = three::perspective_camera(75, three::window_aspect(), 0.1, 1000);
    camera.position.z = 5;

    cube = three::mesh(three::box_geometry(1, 1, 1), three::mesh_standard_material(color: 0x00ff00));
    scene.add(cube);
    scene.add(three::ambient_light(0xffffff, 0.5));

    renderer.set_animation_loop(&animate);
}

fn void animate(double time)
{
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;
    renderer.render(scene, camera);
}
```

## Translating from three.js

| three.js | three.c3 |
| --- | --- |
| `new THREE.Mesh(geo, mat)` | `three::mesh(geo, mat)` |
| `new THREE.BoxGeometry(1, 2, 3)` | `three::box_geometry(1, 2, 3)` |
| `new THREE.MeshStandardMaterial({ color: 0xff0000, roughness: 0.5 })` | `three::mesh_standard_material(color: 0xff0000, roughness: 0.5)` |
| `THREE.DoubleSide` | `DOUBLE_SIDE` |
| `Math.PI` | `three::PI` |
| `mesh.position.set(1, 2, 3)` | same |
| `mesh.rotation.y += 0.1` | same |
| `obj.lookAt(0, 0, 0)` | `obj.look_at(0, 0, 0)` |
| `scene.add(obj)` / `group.remove(obj)` | same |
| `scene.background = new THREE.Color(0x111111)` | `scene.background = 0x111111` |
| `scene.fog = new THREE.Fog(color, near, far)` | `scene.fog = three::fog(color, near, far)` |
| `renderer.setAnimationLoop(fn)` | `renderer.set_animation_loop(&fn)` (opens the window, blocks) |
| `new OrbitControls(camera, dom)` | `three::orbit_controls(camera)` |
| `new THREE.Clock()` / `clock.getDelta()` | `three::clock()` / `clock.get_delta()` |
| `color.setHSL(h, s, l)` | `three::hsl(h, s, l)` |
| `window.innerWidth / window.innerHeight` | `three::window_aspect()` |
| `window.addEventListener('keydown', fn)` | `three::add_event_listener(&fn)` (raw sokol events) |

Rules of thumb: `camelCase` becomes `snake_case`, `new THREE.X(...)` becomes `three::x(...)`,
and parameter objects become named arguments.

## What's there

- **Objects:** `Object3D`, `Group`, `Scene`, `Mesh`, `InstancedMesh`, `Lod`, `PerspectiveCamera`,
  `OrthographicCamera`
  `object.matrix_auto_update` and `matrix_world_auto_update` are three's `matrixAutoUpdate`
  and `matrixWorldAutoUpdate`: with the second off the scene's update leaves that object and
  everything under it alone, so a big world that stands still costs nothing per frame (place
  it once with `update_matrix_world()` first).
- **Skeletons:** `Bone`, `Skeleton`, `SkinnedMesh` (`three::bone()`, `three::skeleton(bones)`,
  `three::skinned_mesh(geometry, material)`, `mesh.bind(skeleton)`). Skin weights are set on the
  geometry with `set_skin`, `set_skin_blend` and `skin_range`. Vertices are skinned on the CPU:
  pose the bones, then call `mesh.update()` once per frame. glTF skins and animation clips are
  not loaded. See `skeleton.c3`.
- **Geometry:** `box`, `plane`, `sphere`, `cylinder`, `cone`, `torus`, `lathe` (a profile of
  (radius, height) points turned around the y axis, like `THREE.LatheGeometry`), plus custom geometry via
  `three::buffer_geometry(positions, indices)` (normals are computed for you), or built up with
  `add_vertex`, `merge` and `compute_vertex_normals`
- **Materials:** `MeshBasicMaterial`, `MeshLambertMaterial`, `MeshPhongMaterial`,
  `MeshStandardMaterial`, `MeshNormalMaterial`, with `opacity`/`transparent`, `side`, `wireframe`,
  `emissive`, `fog`, `map`, `normal_map` + `normal_scale`, `alpha_test`
- **Textures:** `three::load_texture(path)` (PNG and JPEG), `three::data_texture(rgba, w, h)`,
  `three::canvas_texture(w, h)` (a texture you draw into with sokol between `.begin()` and `.end()`,
  for screens in the scene; like a render target's `.texture`),
  with mipmaps and anisotropic filtering. Pass `NO_COLOR_SPACE` for data such as normal maps.
- **Models:** `three::load_gltf(path)` reads binary glTF (`.glb`) with its materials and
  textures; `model.get("name")` finds a node.
- **Lights:** `AmbientLight`, `HemisphereLight`, `DirectionalLight`, `PointLight`, `SpotLight`, as
  many as you like (clustered, see below). Intensities and color handling follow modern three.js (r155+): sRGB hex
  colors, linear lighting, physically based light units.
- **Shadows:** spot and directional lights cast them: `light.cast_shadow = true`, and
  `cast_shadow` / `receive_shadow` on meshes. `light.shadow` has `map_size`, `normal_bias`,
  `radius`, `intensity` (how dark, 0..1, like three r166+), `near`, `far` and `focus` (spot
  lights). One spot light's map is redrawn every
  frame; `renderer.shadow_atlas_capacity` more (default 8) can cast shadows that are drawn once (see `auto_update` below). One
  directional light (the first that casts) has a map of its own at the same time, in cascades
  (see below).
- **Fog:** `Fog` (linear) and `FogExp2`
- **Controls:** `OrbitControls` (left drag rotates, right drag or shift+drag pans, scroll zooms, damping, auto-rotate)
- **Raycasting** (`raycaster.c3`), like three's `Raycaster`: `three::raycaster(origin,
  direction)`, `set_from_camera({ x, y }, camera)` with x and y from -1 to 1, then
  `intersect_object(object, &list)` / `intersect_objects(objects, &list)`, which add
  `Intersection`s (`distance`, `point`, `object`, `face`, `face_index`, `uv`, `normal`,
  `instance_id`) to a list of yours, nearest first. `intersect_first(object, &hit)` finds only
  the nearest hit, without a list. Meshes, each instance of an instanced mesh and the level of a
  `Lod` that the ray's origin would see are tested, after their bounding spheres, following the
  material's `side`. `near`, `far` and `visible_only` limit what counts. As in three, every
  triangle of a mesh the ray comes near is tested; there is no acceleration structure.
- **Post-processing** (`three::postprocessing`, like three's addons of that name):
  `EffectComposer`, `RenderPass` and `ShaderPass`. A `ShaderPass` is a fullscreen shader of
  yours (a sokol-shdc `@vs`/`@fs` program whose vertex shader makes a triangle from
  `gl_VertexIndex`), drawn with `pass.render(target)` (null: the window), over a target's picture
  with `render_over` (with `MULTIPLY_BLEND`, say), or with `draw()` into a pass you began
  (`target.begin()`, `begin_screen(aspect: ...)` / `end_screen()`). It keeps its textures,
  samplers and uniforms (`set_texture`, `set_sampler`, `set_uniforms`) and makes a pipeline for
  each kind of target it draws into. `output_pass(renderer)` is three's `OutputPass`: the end of a
  chain in linear light, tone mapped with the renderer's settings when it draws (an
  `exposure_map`, a 1x1 texture, can scale the exposure from the GPU, for auto exposure).
  `gtao_pass(camera)` is three's `GTAOPass` (ground-truth AO, after Intel's XeGTAO): from the
  depth and normals of a target rendered with `mrt`, it darkens only the light that came in
  indirectly (the target's `indirect_texture`), as three's WebGPU renderer does with an `ao`
  context, with specular occlusion on reflections; `ao.render(scene_target)` after the scene,
  or after a `RenderPass` with `mrt` set in a composer. `taa_pass()` is temporal
  antialiasing: `taa.jitter(camera)` shifts the camera by a fraction of a pixel each frame
  (`camera.jitter`, Halton 2, 3, left out of the motion vectors) and `taa.render(scene_target)`
  blends the new frame into the history found again with the motion vectors (variance
  clipping in YCoCg, a Catmull-Rom history, Karis weighting); the result is `taa.texture()`.
  The composer's buffers are RGBA16F by default, as three's are. `ping_pong(w, h, format)` is a pair of targets for what
  builds up from frame to frame. `effect_composer(renderer)` with `add_pass()` and `render()`
  chains passes as three's does: a `ShaderPass` gets the picture of the pass before at its
  `input_view`, and the last pass draws into the window. See `postprocessing.c3`.
- **Menus** (`three::ui`, not in three.js; `menu.locked = true` leaves the toggle key to the
  game for a while): `ui::menu(title, handler)` makes a game menu that
  opens with escape and takes the keyboard and mouse while open (key releases still reach the
  game). Items are actions or pages; the handler gets `OPEN`, `CLOSE`, `SELECT`, `CONFIRM` and
  `BACK`. With `overlay = true` it is drawn with a plain built-in look over the finished frame
  (`render()` to the window and `end_screen()` do it; a game whose last pass is its own calls
  `three::draw_overlays()` in it). Otherwise draw it yourself from its fields, inside the
  picture if you like. `ui::screen_fade` fades to and from black with a callback at the end.
  `ui::setup()` sets up sokol_debugtext and sokol_gl once, for games that draw text of their
  own too. `ui::touch_controls()` are on-screen sticks and buttons for phones and tablets.
- **Game controllers** (`three::input`, macOS GameController framework, needs `--lib gamectl`):
  `input::gamepad()` has the sticks (dead zone removed), triggers and buttons (`PAD_*`, with
  `pressed()` / `released()` for this frame). The renderer polls it before every frame, and
  by default the d-pad, cross / A, circle / B and options also send arrow, enter, backspace
  and escape key events, so menus and keyboard code work with a controller unchanged.
- **Rewind** (`three::timeline`): `Timeline{Input, State}` records each frame's input and snapshots
  the state a few times a second; `seek()` goes back by loading a snapshot and replaying the
  input after it. Needs a deterministic step function.
- **Renderer:** render targets, tone mapping, frustum culling, and `renderer.info`
  (draw calls and triangles of the last frame, shadow pass counted separately).
  `three::render_target(w, h, samples, format, depth_buffer)` takes three's `type`/`format` and
  `depthBuffer` options as a sokol pixel format and a flag; passes can use any format.
  A scene is rendered into RGBA8 (tone mapped and sRGB, ready to show) or, like three's
  `HalfFloatType` targets, into RGBA16F, where it stays in linear light (a lamp brighter than
  white keeps its brightness through blurs, reflections and blending) until an `OutputPass`
  tone maps it. `renderer.tone_mapping` takes three's tone mappings (`LINEAR_`, `REINHARD_`,
  `CINEON_`, `ACES_FILMIC_`, `AGX_` and `NEUTRAL_TONE_MAPPING`) with `tone_mapping_exposure`;
  `three::tone_map(color, mode, exposure)` does the same on the CPU. Shaders of your own can
  `@include tonemapping.glsl` and use `tone_map()`, `linear_to_srgb()` and `srgb_to_linear()`.
  With `mrt: true` (`three::render_target(w, h, 4, RGBA16F, mrt: true)`, like three's
  `scenePass.setMRT(mrt({ output, normal, velocity }))`) the main pass also writes
  `normal_texture` (world-space normals after normal maps, roughness in alpha) and
  `velocity_texture` (motion vectors: uv now minus uv a frame ago, from the camera, each mesh's
  matrix and wind sway) and `indirect_texture` (the shares of each pixel's light that are
  indirect diffuse and indirect specular), for temporal filters, motion blur, ambient
  occlusion and other screen-space effects.
- **Profiling** (`three::profile`): `profile::enabled = true`, then `profile::start()` /
  `profile::stop(name, t)` around stages and `profile::report(frames)`. The renderer times its
  own stages this way.

The parts that are not in three.js are submodules: `three::ui`, `three::input`,
`three::profile`, `three::timeline`, plus `three::postprocessing` after three's addons.
`import three` imports them too. `ui` and `input` hook into the renderer when the program
starts (`three::add_frame_hook`, `add_event_filter`, `add_overlay`), which an application can
use as well.

## Extensions

These aren't in three.js:

- `material.arm_map`: one texture with ambient occlusion, roughness and metalness in R, G and B
  (the layout Poly Haven ships).
- `material.foliage`: for alpha-tested leaves and blades. Both sides are lit, and light wraps
  around them a little. `material.sway` and `sway_speed` move the vertices in the wind, more the higher
  they are above the object's origin.
- `material.glass`: a transparent material that keeps its reflections however clear it is.
- `light.radius` on point and spot lights: the size of the source. Falloff softens up close
  instead of blowing out.
- `spot_light.specular_radius`: the size of the bright core seen in reflections (0 means
  `radius`). A big soft lamp can still leave a sharp glint on glass and gloss paint.
- `material.env_map` and `env_intensity`: reflections of the surroundings from an
  equirectangular texture, like three's `envMap`. Roughness blurs them through the mip levels.
- `material.env_box_min`, `env_box_max` and `env_position`: box projection for `env_map`. Give the
  room the map was captured in and where; reflections of its walls then keep their size and
  place as the viewer moves.
- `material.reflect_viewer` (glass): the viewer's own silhouette, a soft dark figure standing at
  the camera, darkens the reflections by this much.
- `renderer.capture_environment(scene, position)`: renders the scene in six directions from a
  point, like a light probe, and returns an equirectangular texture for `env_map`. It reads
  the result back from the GPU, so call it once at load. `on_face` runs after the camera is
  aimed at each face (to aim a light with it, for example).
- `light.shadow.auto_update = false`: for a light that stays put among things that stay put. Its
  shadow map is drawn once into a shared atlas (`renderer.shadow_atlas_capacity` of them) and costs nothing
  after that. Set `light.shadow.needs_update = true` to redraw it, as in three.js: only that
  light's tile is drawn again. The soft edge takes about three taps per pixel it spans (4 to
  14), and the search for the caster of a `softness` shadow is skipped where its edge could not
  be a pixel wide, so far and small shadows cost little.
- Cascaded shadow maps for a `DirectionalLight` (like three's `CSM` addon, built in):
  `light.shadow.cascades` (1..4, default 4) cuts the camera's view up to
  `shadow.max_distance` meters (default 150) into slices, each with its own box of parallel
  rays, `map_size` square, in a 2x2 atlas. `shadow.split_lambda` places the cuts (0: evenly,
  1: logarithmic; default 0.7) and `shadow.cascade_blend` is the band where one slice fades into
  the next (a part of the slice, default 0.1); the last fades to no shadow. Each box is fitted
  around the slice's bounding sphere and moves in whole texels, so edges do not crawl as the
  camera turns or moves. The blur (`radius`) and `normal_bias` are scaled to each cascade's
  texel size, so the softness matches across the cuts. Needs a perspective camera.
  `renderer.debug_cascades = true` tints them red, green, blue and yellow, nearest first.
- `shadow.cascades = 1`: one box, `light.shadow_extent` meters from its centre to its sides
  (three's `shadow.camera.right`/`top`), on the light's target, or ahead of the camera with
  `light.shadow_follow = true`. Faded out towards its edge.
- `spot_light.fog_glow`: scales how much the light shows in `fog.light_scatter` fog (0: not at all),
  and only that: a dim lamp can still throw a beam (values far above 1 are fine). In volumetric
  fog the beam stops where the light's shadow map says it does.
- `scene.irradiance`: an `IrradianceVolume`, a grid of probes holding light that has bounced
  around a room. Lit materials inside its box receive it on top of the scene's lights. The
  engine stores and samples the probes; what goes into them is up to you (see `irradiance.c3`).
- `render_target.depth_texture` and `overlay_view`: the depth of the last render (multisampled
  if the target is) and an attachment for drawing over the resolved picture, for
  post-processing such as ambient occlusion.
- A render target's alpha is 0 where a light source was drawn (a material whose emissive is
  far over white, 16 and up), so that post-processing can keep lamps burnt out.
- `material.ssr` and `renderer.set_reflection_source(color, depth, view_proj)`: screen-space
  reflections. A material with `ssr` above 0 mirrors what was on screen around it in the frame
  before, more at a glancing angle and blurred by its roughness (one noisy ray per pixel:
  follow it with a temporal filter). The application hands over the last frame's picture and
  depth every frame, after its own post-processing; see the function's comment for their layout.
- `capture_environment(..., distance: &fn)`: with a ray caster of the application's, the map
  also stores how far away each direction's surface is (in its alpha). A box-projected glossy
  material then follows its reflected rays through the room as the map saw it, so that what
  stands in the middle of the room is mirrored where it stands, not smeared over the walls.
- Clustered forward lighting (`clusters.c3`): lights go to the GPU in storage buffers, and
  the view is cut into 16x9x24 clusters (screen tiles x exponential depth slices). Each
  cluster lists the point and spot lights whose bounding sphere reaches it, and a fragment
  only shades its own cluster's lights. `renderer.max_lights` (default 1024 per kind, 0 = no
  limit) caps a render, with a warning. The assignment runs in a compute shader
  (`cluster_lights_gpu`; `cluster_lights_cpu` gives the same result on the CPU), and
  `renderer.assign_clusters` (a `ClusterAssignFn`) can replace it. Volumetric fog uses the
  clusters too; the per-pixel fallback's glow loops over every light. Storage buffers need GL 4.3 for a GL
  backend (Metal and D3D11 have them).
- Compute shaders (`compute.c3`), after three.js's WebGPU `renderer.compute()` and storage
  buffers. `three::storage_buffer(size)` makes a buffer that kernels read and write
  (`cpu_write: true` for one the CPU fills with `write()` once a frame), and
  `three::compute_pass(my_shader_desc(sg::query_backend()))` a kernel written in GLSL with
  sokol-shdc's `@cs` (see `kernels.glsl`). Bind with `pass.set_buffer(VIEW_X, buffer)` and
  `pass.set_uniforms(UB_X, &params, Params::size)`, then `renderer.compute(pass, groups_x)`
  (or `begin_compute()`, several `pass.dispatch(x, y, z)`, `end_compute()`). Outside of
  `render()`. Results come back to the CPU without stalling: `buffer.track(slot)` (slots from
  `three::readback_slot()`) queues a copy, and `three::read_back(slot, out, bytes)` returns it
  once the GPU has got there, a frame or two later (`wait: true` waits, after the frame is
  committed).
- GPU instance culling (`culling.c3`, `renderer.gpu_culling`, on by default): an
  `InstancedMesh` in view (by the sphere around all its instances) has each instance's sphere
  tested by a compute shader, for the camera and for the shadow maps drawn every frame (the
  sun's cascades, the spot light's map), and only the instances inside are drawn: the vertex
  shader reads their matrices from a storage buffer by index. sokol has no indirect draws, so
  a draw asks for as many instances as the cull of a frame or two before kept, plus some room
  (`renderer.info` counts those), and the vertex shader drops any past this frame's count. When
  the view turns fast, a few instances can be missing for a frame. Meshes with fewer than 8
  instances, `frustum_culled` off, or whose instances keep changing after their first upload
  (`set_matrix_at` every frame) are drawn whole. Level of detail is still picked per `Lod`, not
  per instance.
- `renderer.depth_prepass` (on by default): what has holes in it (`alpha_test`, LOD
  cross-fades) is drawn twice, first only its depth with a cheap shader that cuts the holes,
  then lit where that depth is (an EQUAL depth test) with a shader that cuts nothing. Leaves
  and blades are lit once per pixel instead of once per layer, and no shader of the main pass
  can discard, which keeps the GPU's early depth test for everything. Only for what covers
  at least `renderer.depth_prepass_pixels` (default 1.5) pixels per triangle: a far-off tree
  is mostly triangles and is cheaper drawn once.
- `renderer.check_compute`: also computes on the CPU what the compute shaders do (clusters,
  instance culls) and prints where they differ; `three::print_compute_checks()` sums it up.
- `material.blending = ADDITIVE_BLENDING`, like three's `AdditiveBlending`: a see-through
  material's pixels are added to what is behind them (flames, glows).
- `object.render_order`, like three's `renderOrder`: among see-through meshes, higher ones are
  drawn later whatever their distance (a film of dirt over a pane of glass).
- `renderer.lod_scale`: multiplies every `Lod` distance, to keep detail further out.
- `three::fog_exp(color, density)`: plain exponential fog, `exp(-density * distance)`.
- `fog.light_scatter`: point and spot lights light up the fog around them, and are dimmed by it
  on their way to surfaces. The fog then has no single color, so the renderer draws a backdrop
  sphere just inside the camera's far plane to carry it where there is no geometry.
- Volumetric fog (`VolumetricFog`, after three's `webgpu_volume_lighting`): every `Fog` has its
  settings inline (`fog.light_scatter`, `height_*`, clear box, `sun_*`, `noise_*`,
  `resolution`, `slices`, `volumetric`); `three::volumetric_fog(color, density)` makes one.
  With `fog.volumetric` on (default) and something to scatter (`light_scatter`, `sun_shafts`
  or `noise_amount` above 0), a compute pass (`fog.c3`, `fog_cs` in `kernels.glsl`) marches a
  froxel grid before the main pass: `resolution` of the picture's pixels across and up
  (default 0.125), `slices` of exponential view depth (default 96) out to the far plane,
  jittered every frame. Each step is lit by the sun through its cascades, by spot lights
  through their shadow maps (atlas and live map: beams stop at walls), and by point lights,
  only those of the step's light cluster; the lamps' 1/d^2 glow is integrated exactly over
  each slice. Density is the height fog, the clear box and optional 3D noise (`noise_amount`
  0..1, `noise_scale` 1/m, `noise_wind` m/s). Surfaces read the fog in front of them from the
  grid before tone mapping (see-through ones too). `sun_direction` zero takes the scene's
  (shadow casting) directional light, `sun_color` 0 its color. `fog.volumetric = false`
  keeps the older per-pixel fog (unshadowed glow summed over every light at every pixel).
- `fog.set_clear_box(min, max)`: a box without fog, such as the inside of a building.
  `fog.clear_box_density` (0..1) leaves that part of the fog in it: a thin haze indoors.
- `geometry.box_project_uv(tiles_per_unit)`: texture coordinates from world-space position, for
  tiling a texture over walls and floors without stretching.
- `renderer.init_headless()`: sets up rendering without a window, for tests and screenshots.
  Render to a `RenderTarget` and call `three::commit()` at the end of the frame.
- `renderer.time` drives `sway`. It follows the clock unless `renderer.auto_time` is off.
- Shader hooks, compiled in (three's `onBeforeCompile` works on shader text at run time; sokol
  compiles shaders ahead). An application with surface effects of its own builds its own copy
  of the material shader: a `.glsl` file with its own `@header` (its module), blocks
  `vertex_hooks` and `surface_hooks`, and `@include` of `shader_core.glsl`; see `shader.glsl`
  for what the hooks get. It hands the four programs to `renderer.shaders` before the renderer
  starts. The hooks read `material.custom` and `renderer.custom` (two `Vec4`s each) as
  `material_custom[]` and `frame_custom[]`. The engine's own build has empty hooks, which cost
  nothing. The application's copy has to be regenerated whenever `shader_core.glsl` changes.

## Differences from three.js

- Named arguments must be in declaration order (a C3 rule). For example, `color:` comes before `roughness:`.
- Colors are plain sRGB hex `uint`s, not `Color` objects.
- Rotations are XYZ Euler only (no quaternions or `rotation.order`).
- No animation clips. Point lights cast no shadows.
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

## Editing the shader

`shader_core.glsl` is the material shader; `shader.glsl` includes it with empty hooks. Run
`./build.sh shaders` in the application (`house`) to regenerate `shader.c3` (and `kernels.c3`
from `kernels.glsl`, the compute shaders, `output.c3` and `gtao.c3` for the post-processing
passes, and `taa.c3`), and any application shaders that include `shader_core.glsl`.
