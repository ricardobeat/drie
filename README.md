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
- **Skeletons:** `Bone`, `Skeleton`, `SkinnedMesh` (`three::bone()`, `three::skeleton(bones)`,
  `three::skinned_mesh(geometry, material)`, `mesh.bind(skeleton)`). Skin weights are set on the
  geometry with `set_skin`, `set_skin_blend` and `skin_range`. Vertices are skinned on the CPU:
  pose the bones, then call `mesh.update()` once per frame. glTF skins and animation clips are
  not loaded. See `skeleton.c3`.
- **Geometry:** `box`, `plane`, `sphere`, `cylinder`, `cone`, `torus`, plus custom geometry via
  `three::buffer_geometry(positions, indices)` (normals are computed for you), or built up with
  `add_vertex`, `merge` and `compute_vertex_normals`
- **Materials:** `MeshBasicMaterial`, `MeshLambertMaterial`, `MeshPhongMaterial`,
  `MeshStandardMaterial`, `MeshNormalMaterial`, with `opacity`/`transparent`, `side`, `wireframe`,
  `emissive`, `fog`, `map`, `normal_map` + `normal_scale`, `alpha_test`
- **Textures:** `three::load_texture(path)` (PNG and JPEG), `three::data_texture(rgba, w, h)`,
  with mipmaps and anisotropic filtering. Pass `NO_COLOR_SPACE` for data such as normal maps.
- **Models:** `three::load_gltf(path)` reads binary glTF (`.glb`) with its materials and
  textures; `model.get("name")` finds a node.
- **Lights:** `AmbientLight`, `HemisphereLight`, `DirectionalLight` (up to 4), `PointLight` (up to 4),
  `SpotLight` (up to 4). Intensities and color handling follow modern three.js (r155+): sRGB hex
  colors, linear lighting, physically based light units.
- **Shadows:** spot lights cast them: `light.cast_shadow = true`, and `cast_shadow` /
  `receive_shadow` on meshes. `light.shadow` has `map_size`, `normal_bias`, `radius`, `near`,
  `far` and `focus`. One light's map is redrawn every frame; up to four more can cast shadows
  that are drawn once (see `auto_update` below).
- **Fog:** `Fog` (linear) and `FogExp2`
- **Controls:** `OrbitControls` (left drag rotates, right drag or shift+drag pans, scroll zooms, damping, auto-rotate)
- **Menus** (`menu.c3`, not in three.js): `three::menu(title, handler)` makes a game menu that
  opens with escape and takes the keyboard and mouse while open (key releases still reach the
  game). Items are actions or pages; the handler gets `OPEN`, `CLOSE`, `SELECT`, `CONFIRM` and
  `BACK`. With `overlay = true` it is drawn with a plain built-in look over the finished frame
  (`render()` to the window does it; a game whose last pass is its own calls `three::draw_ui()`
  in it). Otherwise draw it yourself from its fields, inside the picture if you like.
  `three::screen_fade` fades to and from black with a callback at the end. `three::ui_setup()`
  sets up sokol_debugtext and sokol_gl once, for games that draw text of their own too.
- **Rewind** (`timeline.c3`): `Timeline{Input, State}` records each frame's input and snapshots
  the state a few times a second; `seek()` goes back by loading a snapshot and replaying the
  input after it. Needs a deterministic step function.
- **Renderer:** render targets, ACES tone mapping, frustum culling, and `renderer.info`
  (draw calls and triangles of the last frame, shadow pass counted separately)

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
  shadow map is drawn once into a shared atlas (four such lights at most) and costs nothing
  after that. Set `light.shadow.needs_update = true` to redraw it, as in three.js.
- `spot_light.fog_glow`: scales how much the light shows in `fog.light_scatter` fog (0: not at all).
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
- Up to 5 spot lights.
- `renderer.lod_scale`: multiplies every `Lod` distance, to keep detail further out.
- `three::fog_exp(color, density)`: plain exponential fog, `exp(-density * distance)`.
- `fog.light_scatter`: point and spot lights light up the fog around them, and are dimmed by it
  on their way to surfaces. The fog then has no single color, so the renderer draws a backdrop
  sphere just inside the camera's far plane to carry it where there is no geometry.
- `fog.set_clear_box(min, max)`: a box without fog, such as the inside of a building.
- `geometry.box_project_uv(tiles_per_unit)`: texture coordinates from world-space position, for
  tiling a texture over walls and floors without stretching.
- `renderer.init_headless()`: sets up rendering without a window, for tests and screenshots.
  Render to a `RenderTarget` and call `three::commit()` at the end of the frame.
- `renderer.time` drives `sway`. It follows the clock unless `renderer.auto_time` is off.

## Differences from three.js

- Named arguments must be in declaration order (a C3 rule). For example, `color:` comes before `roughness:`.
- Colors are plain sRGB hex `uint`s, not `Color` objects.
- Rotations are XYZ Euler only (no quaternions or `rotation.order`).
- No raycasting, animation or skinning. Only spot lights cast shadows.
- `InstancedMesh` grows: `add_instance(matrix)` appends, `set_matrix_at(index, matrix)` changes one.
- `Lod` (three.js `LOD`) levels are added with `lod.add_level(object, distance)`, in increasing
  distance. A `null` object shows nothing from that distance on.
- Perspective cameras follow the window's aspect ratio automatically (`renderer.auto_aspect`).
- Objects are allocated with the heap allocator and never freed. `geometry.dispose()` frees GPU buffers.
- Create objects outside of `@pool()` blocks. Their lists live on the heap, but anything you
  build with the temp allocator and hand to the library does not outlive the pool.
- Shaders are built for Metal, GL 4.3 and D3D11, but the library has only been tested on macOS/Metal.

## Editing the shader

`three/shader.glsl` is the source. Run `./build.sh shaders` to regenerate `three/shader.c3`.
