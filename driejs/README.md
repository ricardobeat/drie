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
| `lib/` | `draco.c3l` and `wamr.c3l`; drie and sokol come from drie's `lib/` (`../lib`, the second search path in `project.json`) |
| `lib/wamr.c3l`, `js/wasm.js` | `WebAssembly`, on [WAMR](https://github.com/bytecodealliance/wasm-micro-runtime) 2.4.5's fast interpreter, which `just build` fetches into `third_party/` and builds |
| `lib/draco.c3l` | DRACOLoader's decoder: libdraco, linked statically from Homebrew (`brew install draco` to build) |

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
