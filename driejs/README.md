# driejs

Runs three.js code on drie: three.js itself (`vendor/three.core.js`, unmodified) in
[boomkat](../../boomkat), with drie as its `WebGLRenderer` (`js/drie_renderer.js`). A script
imports `three` and `three/addons/...` as it would in a browser.

```sh
just build                                  # build out/driejs
just run demos/webgl_geometries.js          # build and run it in a window
just run --frames 3 --png shot.png demos/webgl_geometries.js   # offscreen
just bench node                             # the benchmark on one runtime (`just` lists them)
```

From drie's folder, the same recipes are `just driejs::build` and so on.

Run it from this folder: `three` and the addons are read from `js/` and `vendor/` under the
working directory.

This folder is a project of its own inside the drie library. c3c compiles only the `.c3` files
at the top of a `.c3l` folder, so nothing here is built into a program that uses drie (the
game, say).

It also measures the per-frame CPU work three.js does in JS before a renderer draws anything
(see Cases and Results).

| Path | What |
|---|---|
| `vendor/` | three.js r187 from npm (`three@0.187.0`, checked against a sha256 in the `justfile`), fetched by `just build` when missing: `three.core.js` (core without WebGLRenderer), `three.module.js`, and `examples/jsm` as `jsm/` for `three/addons/...`. Unmodified, not in git |
| `demos/` | three.js examples as scripts (`demos/src/` has their pages); their models and textures are fetched by `just build` from three.js r187, each checked against a sha256, and are not in git |
| `bench/three_bench.js` | the benchmark, for boomkat, node and bun |
| `bench/drie_bench.c3` | its transform cases in drie, natively |
| `bench/driejs_bench.js` | its transform cases in [drie-js](../../drie-js), whose three.js-shaped API pushes transforms to drie |
| `just bench` | `just bench [boomkat\|boomkat-jit\|boomkat-opt\|node\|node-jitless\|bun\|drie\|drie-js\|all]` |
| `lib/` | `drie.c3l` is this repository (`../..`); sokol, readback, imageio and gamectl are symlinked from `../../house/lib` |
| `lib/wamr.c3l`, `js/wasm.js` | `WebAssembly`, on [WAMR](https://github.com/bytecodealliance/wasm-micro-runtime) 2.4.5's fast interpreter, which `just build` fetches into `third_party/` and builds |
| `lib/draco.c3l` | DRACOLoader's decoder: libdraco, linked statically from Homebrew (`brew install draco` to build) |

boomkat needs commit `92c6f2ea` or later: before it, a derived class's constructor with a
non-literal default parameter threw, and three.js failed while loading. The `jit` branch
(`d226d129`) predates it; the JIT columns are that branch with `92c6f2ea` cherry-picked.
`--jit` is the baseline tier, `--opt` adds the optimizing tier.

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

Apple Silicon, macOS, 2026-11-16. Milliseconds per frame.

| Case | N | drie (C3) | drie-js | node | bun | node --jitless | boomkat | boomkat --jit | boomkat --opt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| transforms: all moving | 1000 | 0.028 | 0.385 | 0.073 | 0.065 | 1.65 | 2.05 | 1.22 | 0.638 |
| | 5000 | 0.136 | 1.98 | 0.449 | 0.331 | 8.63 | 8.64 | 6.86 | 4.54 |
| | 20000 | 0.599 | 8.92 | 4.59 | 3.54 | 46.7 | 41.8 | 34.6 | 24.4 |
| transforms: nothing moving | 1000 | 0.007 | 0.348 | 0.037 | 0.040 | 1.30 | 1.03 | 0.688 | 0.136 |
| | 5000 | 0.036 | 1.75 | 0.229 | 0.217 | 6.68 | 5.38 | 3.60 | 0.850 |
| | 20000 | 0.155 | 7.43 | 1.53 | 2.34 | 30.9 | 24.9 | 18.1 | 6.72 |
| transforms: frozen | 1000 | 0.000 | | 0.009 | 0.007 | 0.063 | 0.071 | 0.036 | 0.019 |
| | 5000 | 0.000 | | 0.047 | 0.038 | 0.308 | 0.355 | 0.177 | 0.095 |
| | 20000 | 0.001 | | 0.205 | 0.232 | 2.06 | 2.05 | 1.04 | 0.696 |
| render list (incl. nothing moving) | 1000 | | | 0.166 | 0.128 | 3.12 | 2.91 | 1.93 | 0.946 |
| | 5000 | | | 0.827 | 0.676 | 16.7 | 15.0 | 10.7 | 6.08 |
| | 20000 | | | 5.25 | 5.70 | 84.3 | 66.1 | 49.3 | 29.8 |
| sync: copy matrixWorld out | 1000 | | | 0.025 | 0.018 | 0.055 | 0.375 | 0.370 | 0.375 |
| | 5000 | | | 0.139 | 0.095 | 0.326 | 1.92 | 1.91 | 1.99 |
| | 20000 | | | 0.756 | 1.12 | 3.02 | 10.5 | 11.1 | 10.4 |
| animation: 60-bone characters | 10 | | | 0.150 | 0.130 | 2.63 | 4.40 | 3.86 | 3.05 |
| | 50 | | | 0.813 | 0.772 | 13.3 | 23.9 | 20.6 | 16.8 |
| raycast: 8k triangles | 1 | | | 0.262 | 0.215 | 9.67 | 13.4 | 5.39 | 3.66 |

drie-js's "all moving" splits into `pushTransform` (JS, native calls) and drie's
`update_matrix_world` (C): at 20000, 6.83 ms and 0.14 ms.

drie's frozen case is a flag check per group: its `matrix_world_auto_update = false` skips the
subtree, and its transforms are only recomputed for what moved.
