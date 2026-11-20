# driejs

Runs three.js code on drie: three.js itself (`vendor/three.core.js`, unmodified) in
[boomkat](../../boomkat), with drie as its `WebGLRenderer` (`js/drie_renderer.js`). A script
imports `three` and `three/addons/...` as it would in a browser.

```sh
just build                                  # build out/driejs
just run demos/webgl_geometries.js          # build and run it in a window
just run --frames 3 --png shot.png demos/webgl_geometries.js   # offscreen
just bench node                             # the benchmark on one runtime (`just` lists them)
just thumbs                                 # render the demos' thumbnails into demos/thumbs/
just run demos/index.js                     # pick a demo from a page of them
```

`demos/index.js` shows every script in `demos/` as a thumbnail with its name. A click loads that
demo's script in the same process and window, in the page's place, and Escape there loads the
page again, unless the demo holds the pointer, when Escape lets go of it first. The script
that leaves loses its animation loop, timers and event listeners, and the next one draws with
the same renderer; what the old one made in drie (meshes, textures) stays in memory until
driejs quits. The
wheel and the arrow keys scroll. Thumbnails are not in git: `just thumbs` renders them (30
frames in; `just thumbs 120 webgl_geometries` for another moment of one demo), and a demo
without one shows as a grey tile.

Each thumbnail also holds the pipelines its demo drew with: drie's pipeline cache, deflated in
a `drPc` chunk of the PNG (`--png-width N` writes it, with the picture shrunk to N pixels
across). The page loads every thumbnail's list when it starts and has the pipelines compiled
in the background, with a small bar in its top right corner until they are, so a demo, a TSL
one above all, opens without its shaders' second or two. The host has the cache for scripts
of their own too: `__drie.savePipelineCache(path)`, `loadPipelineCache(path)` (a text file, or
a PNG with a `drPc` chunk) and `pipelineCacheProgress()` (0 to 1).

With `--png` (and not `--profile`), `performance.now()` runs on the frames' clock, 1/60 s a frame
from 0, as three's screenshots are taken: a script that moves things by the time passed (a
`THREE.Timer`) gives the same picture on any machine.

From drie's folder, the same recipes are `just driejs::build` and so on. boomkat's `jit` branch
is a dependency in `project.json`, fetched from `https://c3.ricar.do/github.com:ricardobeat/boomkat@jit`
into `lib/boomkat.c3l` the first time you build (delete that folder to fetch it again).

Run it from this folder: `three` and the addons are read from `js/` and `vendor/` under the
working directory.

This folder is a project of its own inside the drie library. c3c compiles only the `.c3` files
at the top of a `.c3l` folder, so nothing here is built into a program that uses drie (the
game, say).

It also measures the per-frame CPU work three.js does in JS before a renderer draws anything
(see Cases and Results).

| Path | What |
|---|---|
| `vendor/` | three.js r187 from npm (`three@0.187.0`, checked against a sha256 in the `justfile`), fetched by `just build` when missing: `three.core.js` (core without WebGLRenderer), `three.module.js`, `three.webgpu.js` and `three.tsl.js` (`three/webgpu`, `three/tsl`), and `examples/jsm` as `jsm/` for `three/addons/...`. Unmodified, not in git |
| `js/drie_nodes.js` | three's node materials and TSL on drie (see TSL) |
| `demos/` | three.js examples as scripts (`demos/src/` has their pages); their models and textures are fetched by `just build` from three.js r187, each checked against a sha256, and are not in git |
| `bench/three_bench.js` | the benchmark, for boomkat, node and bun |
| `bench/drie_bench.c3` | its transform cases in drie, natively |
| `bench/driejs_bench.js` | its transform cases in [drie-js](../../drie-js), whose three.js-shaped API pushes transforms to drie |
| `just bench` | `just bench [boomkat\|boomkat-jit\|boomkat-opt\|node\|node-jitless\|bun\|drie\|drie-js\|all]` |
| `lib/` | `draco.c3l` and `wamr.c3l`; drie and sokol come from drie's `lib/` (`../lib`, the second search path in `project.json`) |
| `lib/wamr.c3l`, `js/wasm.js` | `WebAssembly`, on [WAMR](https://github.com/bytecodealliance/wasm-micro-runtime) 2.4.5's fast interpreter, which `just build` fetches into `third_party/` and builds |
| `lib/draco.c3l` | DRACOLoader's decoder: libdraco, linked statically from Homebrew (`brew install draco` to build) |

## TSL

A script can import `three/webgpu` and `three/tsl` as three's WebGPU examples do. `WebGPURenderer`
is drie's renderer (`init()` resolves at once), and the node materials
(`MeshStandardNodeMaterial`, `MeshBasicNodeMaterial`...) are drawn with drie's node materials
(see drie's README). three's NodeBuilder does not run. `js/drie_nodes.js` walks each material's
`colorNode`, `opacityNode`, `normalNode` (view space, as in three), `roughnessNode`,
`metalnessNode`, `emissiveNode`, `positionNode`, `fragmentNode` (the color in place of the
lighting), `outputNode`, `maskNode` (which cuts the shadow too, as three's does without a
`maskShadowNode`), `envNode` (the reflection; the map it reads, if any, lights the material
as an envMap), phong's `specularNode` and `shininessNode`, and `lightsNode` (`lights([...])`:
each list of lights is a lighting channel of drie's, up to 24), and rebuilds the graph as drie
nodes. A material that isn't a NodeMaterial takes nodes in these slots too, as three/webgpu
makes every material a node material.
In `outputNode`, `output` is a variable that a `Fn` can assign. It runs TSL's `Fn` bodies itself, one call at a time, to collect their statements:
`toVar`, `assign`, `If`/`ElseIf`/`Else`, `Loop`, `Break`, `Continue` and `Discard`. drie then
prints the graph in Metal, HLSL or GLSL. A graph is rebuilt when a slot changes or a texture it
reads finishes loading. Uniforms' `.value` is read every frame, and their `onFrameUpdate` and
`onRenderUpdate` callbacks run.

The translator covers:

- constants, uniforms, textures, the geometry's attributes and varyings (including ones assigned
  in `positionNode`)
- TSL's operators and math, swizzles, conversions and `select`, with comparisons of vectors
  component by component (a vector of 0 and 1, which `select` mixes by)
- int and uint arithmetic with bitwise operators (MaterialX noise runs as it is)
- `Fn` calls with layouts and overloads
- `rotate`, of a vec2 by an angle or a vec3 about each axis
- `flipX()`, `flipY()`... (those components 1 minus themselves)
- `mat3` of three vec3s, in a product with a vec3 (TSL's `TBNViewMatrix`, which `parallaxUV`
  reads; `tangentView` and `bitangentView` are the frame from the derivatives)
- `range()`, each instance's value hashed from its index as three's is (without three's seed
  for the object: other values, as random)
- `reference()` and `materialReference()`, read from the material every frame (each
  `NodeMaterialInstance` of one node material has its own values)
- `bumpMap`, as three's `dHdxy_fwd` and `perturbNormalArb`
- `normalMap` in tangent space, on three's frame from the derivatives of the position and uv
  (not the geometry's tangents)
- `element()` of a `uniformArray` or a vector, at a constant index or one worked out in the shader
- `cubeTexture` and `pmremTexture`, read from drie's equirectangular map of the cube map
- `scene.backgroundNode`, and `scene.environmentNode` drawn into an environment map on the GPU
  (drie's `render_environment`)
- the builtins: `positionLocal`/`World`/`View`, `normalLocal`/`World`/`View` and their
  geometry's, `cameraPosition`, `time`, `modelWorldMatrix`, `cameraViewMatrix`,
  `cameraProjectionMatrix`, `screenCoordinate`, `frontFacing`, `faceDirection`, `output`,
  `instanceIndex`

Any other node class throws an error that names it. Slots drie lacks (`vertexNode`, `aoNode`, `maskShadowNode`...) are left out with a warning. Compute, post-processing nodes and
`wgslFn` are not supported. A graph that drie rejects while printing it (types that don't
combine, a texture in `positionNode`, more than 32 vec4s of uniforms or four varyings) stops the
program with drie's message.

`CubeCamera.update` draws the scene around the camera into a drie environment map on the GPU
(`render_probe`), which the render target's texture stands for as an `envMap`, `environment` or
background. `LightProbeGenerator.fromCubeRenderTarget` reads that map's light on surfaces, and
`CubeRenderTarget.fromEquirectangularTexture` uses the equirectangular map as it is. Before the
first frame there is nothing to draw with, so an update waits for the next render.

A mesh with an array of materials draws each of its geometry's groups with that group's
material: a drie mesh of its own that shares the geometry and draws the group's indices
(drie's `Mesh.draw_start` and `draw_count`). A `BatchedMesh` is an instanced drie mesh for each
of its geometries, with that geometry's visible instances. `ProgressiveLightMapGPU` is the same
drie bake as `ProgressiveLightMap`.

A `Sprite`, and a mesh with a `SpriteNodeMaterial` (an `InstancedMesh` of particles), is a drie
mesh turned toward the camera as it is drawn (drie's `Mesh.billboard`), with the sprite's
center, the material's rotation and `sizeAttenuation`. A `SpriteNodeMaterial`'s `positionNode`
and `scaleNode` move and size each quad in the view, as three's; its `rotationNode` is left out
(the material's rotation is used). A `vertexNode` that is TSL's `billboarding()` turns the mesh
the same way (about its own y, or with `horizontalRotation` about the world's up toward the
camera); its `position` is left out.

The demos `webgpu_tsl_earth.js`, `webgpu_tsl_raging_sea.js`, `webgpu_tsl_procedural_terrain.js`
and `webgpu_tsl_angular_slicing.js` are r187's examples, unchanged. Their `Inspector` is a lil-gui panel that is not drawn.

boomkat's `Object.defineProperties` defines only the first 256 keys it is given, and TSL defines
its 1052 swizzle properties with a single call. `js/dom.js` defines them one at a time when there
are more than 256.

boomkat needs commit `92c6f2ea` or later: before it, a derived class's constructor with a
non-literal default parameter threw, and three.js failed while loading. driejs builds on the
`jit` branch, with both JIT tiers on (boomkat's `--jit`); the `boomkat` column is main,
interpreted.

drie-js needs `--script` and the `now` and `updateMatrixWorld` natives in its `src/main.c3`,
and `MAX_OBJECTS` above 22000 for the 20000-mesh scene.

## Cases

The scene has N meshes in groups of 10, all in view.

- **transforms: all moving**: every mesh's rotation changes, then `scene.updateMatrixWorld()`.
- **transforms: nothing moving**: `updateMatrixWorld()` with nothing changed. three
  recomputes every matrix anyway.
- **transforms: frozen**: `matrixAutoUpdate = false` on every object. three still walks the
  tree; `matrixWorldAutoUpdate = false` only skips an object's own matrix, not its children.
- **render list**: what `WebGLRenderer.render` does before drawing: `updateMatrixWorld`
  (nothing moving), frustum test per mesh, depth sort. Cull and sort alone are this minus
  "nothing moving".
- **sync**: copying every `matrixWorld` into one `Float32Array`, as a renderer would hand them
  to native code.
- **animation**: characters of 60 bones, an `AnimationMixer` with a position and a quaternion
  track per bone, then `updateMatrixWorld` and `skeleton.update()`.
- **raycast**: `Raycaster.intersectObject` on a sphere of 8064 triangles.

In drie-js, a frame is what its `renderer.render()` does before drawing: two native calls per
object (`transform` with 10 numbers, `visible`) whether it moved or not, then drie's
`update_matrix_world` in C. Culling and sorting are in C inside drie's render and not
measured, so its "all moving" compares with three's "render list" plus "sync", minus three's
culling.

## Results

Apple Silicon, macOS, 2026-11-17 (c3c 0.8.6; boomkat main `eb5e17f1`, jit `c8ec5308`; node
24.13, bun 1.3.13). Milliseconds per frame. `driejs` is boomkat's `jit` branch with both JIT
tiers, as driejs runs a script (`BOOMKAT=<a jit build> just bench boomkat-jit`); `boomkat` is
main without a JIT.

| Case | N | drie (C3) | driejs | node --jitless | boomkat | node | bun |
|---|---:|---:|---:|---:|---:|---:|---:|
| transforms: all moving | 1000 | 0.030 | 0.315 | 1.50 | 1.43 | 0.070 | 0.059 |
|  | 5000 | 0.178 | 1.55 | 7.46 | 6.97 | 0.366 | 0.316 |
|  | 20000 | 0.698 | 7.68 | 32.7 | 28.1 | 2.79 | 1.42 |
| transforms: nothing moving | 1000 | 0.006 | 0.141 | 1.22 | 1.06 | 0.035 | 0.036 |
|  | 5000 | 0.032 | 0.708 | 6.02 | 5.07 | 0.198 | 0.193 |
|  | 20000 | 0.136 | 3.09 | 24.3 | 20.5 | 1.44 | 0.922 |
| transforms: frozen | 1000 | 0.000 | 0.019 | 0.059 | 0.057 | 0.009 | 0.006 |
|  | 5000 | 0.000 | 0.094 | 0.291 | 0.280 | 0.045 | 0.034 |
|  | 20000 | 0.001 | 0.408 | 1.16 | 1.12 | 0.182 | 0.150 |
| render list (incl. nothing moving) | 1000 |  | 0.904 | 2.81 | 2.76 | 0.159 | 0.116 |
|  | 5000 |  | 4.94 | 14.9 | 13.9 | 0.739 | 0.599 |
|  | 20000 |  | 21.2 | 59.6 | 58.8 | 3.96 | 3.19 |
| sync: copy matrixWorld out | 1000 |  | 0.058 | 0.051 | 0.069 | 0.025 | 0.016 |
|  | 5000 |  | 0.295 | 0.299 | 0.343 | 0.134 | 0.084 |
|  | 20000 |  | 1.22 | 1.08 | 1.41 | 0.585 | 0.370 |
| animation: 60-bone characters | 10 |  | 1.09 | 2.34 | 2.27 | 0.148 | 0.116 |
|  | 50 |  | 5.43 | 11.7 | 11.5 | 0.723 | 0.586 |
| raycast: 8k triangles | 1 |  | 2.45 | 8.12 | 10.5 | 0.218 | 0.268 |

drie's frozen case is a flag check per group: its `matrix_world_auto_update = false` skips the
subtree, and its transforms are only recomputed for what moved.
