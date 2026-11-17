#!/bin/sh
# Builds out/driejs, the host that runs three.js in boomkat with drie as its renderer.
# boomkat comes from its jit branch, ../../boomkat/.worktrees/jit.
#   ./build.sh                       build
#   ./build.sh run demos/x.js [...]  build and run (from this directory)
set -e
cd "$(dirname "$0")"
MIN_VERSION=""
if [ "$(uname)" = Darwin ]; then MIN_VERSION="--macos-min-version 14.0"; fi
# WAMR (WebAssembly's runtime, js/wasm.js): its fast interpreter as a static library
if [ ! -f third_party/wamr-build/libiwasm.a ]; then
    [ -d third_party/wamr ] || git clone -q --depth 1 --branch WAMR-2.4.5 \
        https://github.com/bytecodealliance/wasm-micro-runtime.git third_party/wamr
    cmake -S third_party/wamr/product-mini/platforms/darwin -B third_party/wamr-build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DWAMR_BUILD_TARGET=AARCH64 \
        -DWAMR_BUILD_INTERP=1 -DWAMR_BUILD_FAST_INTERP=1 -DWAMR_BUILD_AOT=0 -DWAMR_BUILD_JIT=0 \
        -DWAMR_BUILD_FAST_JIT=0 -DWAMR_BUILD_LIBC_BUILTIN=0 -DWAMR_BUILD_LIBC_WASI=0 \
        -DWAMR_BUILD_SIMD=0 -DWAMR_BUILD_BULK_MEMORY=1 -DWAMR_BUILD_REF_TYPES=1 \
        -DWAMR_BUILD_MULTI_MODULE=0 > /dev/null
    cmake --build third_party/wamr-build --target vmlib -j8 > /dev/null
fi
# fetch <url> <sha256> <file>: downloads the file unless it is there, and stops if it is not
# the one expected (sha256 is `shasum -a 256` of it).
fetch() {
    [ -f "$3" ] && return
    mkdir -p "$(dirname "$3")"
    curl -fsSL -o "$3.part" "$1"
    if [ "$(shasum -a 256 "$3.part" | cut -d' ' -f1)" != "$2" ]; then
        echo "$3: its sha256 is not the one in build.sh ($1)" >&2
        exit 1
    fi
    mv "$3.part" "$3"
}
# three.js from npm into vendor/: build/three.core.js and three.module.js, and examples/jsm as
# vendor/jsm (what `three/addons/...` reads). To update it, change the version, the tag and the
# sha256s below (the .tgz is `npm view three@<version> dist.tarball`).
THREE_VERSION=0.187.0
THREE_TAG=r187
if [ ! -f vendor/three.core.js ] || [ ! -d vendor/jsm ]; then
    tgz=third_party/three-$THREE_VERSION.tgz
    fetch "https://registry.npmjs.org/three/-/three-$THREE_VERSION.tgz" \
        ff540832ded0cfe919764d24a3d88f3e2a05f23303f876423e4f893841e90728 "$tgz"
    mkdir -p third_party/three-$THREE_VERSION vendor
    tar -xzf "$tgz" -C third_party/three-$THREE_VERSION \
        package/LICENSE package/build/three.core.js package/build/three.module.js package/examples/jsm
    src=third_party/three-$THREE_VERSION/package
    cp "$src/LICENSE" "$src/build/three.core.js" "$src/build/three.module.js" vendor/
    [ -d vendor/jsm ] || cp -R "$src/examples/jsm" vendor/jsm
fi
# The demos' models and textures: three.js's examples at the same release (not in npm's package).
while read -r sha path; do
    fetch "https://raw.githubusercontent.com/mrdoob/three.js/$THREE_TAG/examples/$path" "$sha" "demos/$path"
done <<EOF
909d9a1eb2a5d5de9d221a5e8de4e9119d409decddf522d48896bd51523d354d textures/uv_grid_opengl.jpg
8375c2aa6808e28ad1cb9c0d43e1513f57ffd7865244bf66bbbd5bb0e907e06d models/gltf/LittlestTokyo.glb
07524c7fd5829c6dc354d0b129f78943c47a8737d141496c6d611f5ee2d00864 models/gltf/collision-world.glb
EOF
c3c build driejs --libdir lib --lib sokol --lib readback --lib imageio --lib gamectl --lib drie --lib draco --lib wamr $MIN_VERSION
if [ "$1" = run ]; then shift; exec out/driejs "$@"; fi
