#!/bin/sh
# Builds and runs one of the rendering tests in this folder (headless, macOS).
#   tests/run.sh <test> <libdir> [out]
# <test> is the file's name without .c3. <libdir> holds drie.c3l and the libraries it needs
# (sokol, readback, imageio, gamectl): the game's lib/. The shaders must have been generated
# (the game's ./build.sh shaders does that).
set -e
here="$(cd "$(dirname "$0")" && pwd)"
test="$1"
libdir="$2"
out="${3:-$test.ppm}"
mkdir -p "$here/build"
c3c compile --macos-min-version 14.0 --libdir "$libdir" --lib sokol --lib readback --lib imageio --lib gamectl \
    --lib drie "$here/$test.c3" -o "$here/build/$test"
"$here/build/$test" "$out"
