# drie. `just` lists the recipes; `just driejs::<recipe>` runs driejs's (three.js on drie),
# as does `just <recipe>` inside driejs/. Nothing here is part of the library: c3c only compiles
# the .c3 files listed in manifest.json (src/** and generated/*).

mod driejs

# where drie's dependency, sokol, is (lib/sokol.c3l).
# Another: `just libdir=path test radiance`.
libdir := "lib"
libs := "--libdir " + libdir + " --lib sokol"
# project.json's target for this OS: drie's sources and C files with one program of tests/ or
# examples/ added (--sources), the way the manifest builds them for a game
target := "drie-" + os()
# where bindgen.c3l is (the folder holding bindgen.c3l/), for `just bindings`
bindgen := "../bindgen.c3l"
# sokol-shdc binary for `just shaders`, fetched on demand into build/tools/ (`just shdc=path shaders` to override)
shdc := "build/tools/sokol-shdc" + if os() == "windows" { ".exe" } else { "" }
# the macOS c3c builds for: the one it runs on, as clang does for C without a target (and as
# Homebrew's libdraco and boomkat's C are built), so that the linker finds nothing newer
macos := if os() == "macos" { "--macos-min-version " + `sw_vers -productVersion 2>/dev/null || true` } else { "" }

# builds the C sources of drie and sokol into the static libraries the manifests link on macOS
# (tools/c_libs.py; elsewhere c3c compiles them)
c_libs := if os() == "macos" { "python3 tools/c_libs.py macos-aarch64" } else { "true" }


[private]
default:
    @just --list

# type-check drie, without building a program
check:
    c3c compile-only {{libs}} --obj-out build/check $(find src -name '*.c3' | sort) generated/*.c3

# build and run a rendering test, headless: `just test radiance` writes tests/build/radiance.png
# (`just test radiance out.png --p3`: elsewhere, and in Display P3's colors)
test name *args:
    mkdir -p tests/build
    {{c_libs}}
    c3c run {{target}} {{macos}} --libdir {{libdir}} --sources tests/{{name}}.c3 -- {{args}}

# build and run an example program (examples/): `just example nodes`
example name *args:
    {{c_libs}}
    c3c run {{target}} -O2 {{macos}} --libdir {{libdir}} --sources examples/{{name}}.c3 -- {{args}}

# build an example for the web (WebGPU), one self-contained page: `just web gpu_handkerchiefs`
# writes build/web/index.html
web name:
    mkdir -p build/web
    # (the standard library with 16-byte allocations under emscripten, as drie's vectors and
    # keys need: wasm loads a misaligned value only slower)
    rm -rf build/web/std && cp -R `dirname $(realpath $(which c3c))`/../lib/c3/std build/web/std
    sed -i '' 's/env::EMSCRIPTEN ? 8 :/env::EMSCRIPTEN ? 16 :/' build/web/std/core/mem.c3
    # the textures the example names, made smaller by imageoptim
    rm -rf build/web/files && mkdir -p build/web/files/examples/textures
    for t in `grep -o 'examples/textures/[a-z0-9_]\+' examples/{{name}}.c3 | sort -u`; do cp $t* build/web/files/examples/textures/ 2>/dev/null; done
    grep -o 'surface("[a-z0-9_]*' examples/{{name}}.c3 | cut -d'"' -f2 | while read t; do cp examples/textures/${t}_* build/web/files/examples/textures/; done
    # (the cloth's knit and every AO map at 512 px: small on screen, soft by nature)
    for f in build/web/files/examples/textures/cotton_* build/web/files/examples/textures/*_ao.*; do [ -f "$f" ] && sips -Z 512 "$f" > /dev/null; done
    imageoptim build/web/files/examples/textures/ > /dev/null
    c3c build drie-web --target emscripten --stdlib build/web/std -O2 --libdir {{libdir}} --sources examples/{{name}}.c3 -z "--use-port=emdawnwebgpu" -z "-sDEFAULT_TO_CXX" -z "-sSINGLE_FILE" -z "-sALLOW_MEMORY_GROWTH" -z "-sSTACK_SIZE=8MB" -z "--profiling-funcs" -z "--embed-file=build/web/files@/"
    python3 tools/web_page.py build/drie-web.js "drie: {{name}}" build/web/index.html

# regenerate generated/cgltf.c3, the binding of cgltf (third_party/cgltf/cgltf.h), with
# bindgen.c3l: `just bindgen=path bindings` for another checkout of it
bindings:
    mkdir -p build/cgltf
    c3c compile --libdir {{bindgen}} --lib bindgen third_party/cgltf/bind.c3 -o build/cgltf/bind
    build/cgltf/bind third_party/cgltf/cgltf.h generated/cgltf.c3
    # the structs' layout in C and in the binding, compared (layout.py)
    {{bindgen}}/tools/bgdump third_party/cgltf/cgltf.h {{ if os() == "macos" { "-I$(xcrun --show-sdk-path)/usr/include" } else { "-I/usr/include" } }} > build/cgltf/dump.txt
    python3 third_party/cgltf/layout.py build/cgltf/dump.txt build/cgltf
    cc -Ithird_party/cgltf build/cgltf/layout.c -o build/cgltf/layout_c
    c3c compile -O0 build/cgltf/layout.c3 generated/cgltf.c3 -o build/cgltf/layout_c3 > /dev/null
    build/cgltf/layout_c > build/cgltf/layout_c.txt
    build/cgltf/layout_c3 > build/cgltf/layout_c3.txt
    diff build/cgltf/layout_c.txt build/cgltf/layout_c3.txt && echo "cgltf.c3: the layout of $(grep -c '^cgltf' build/cgltf/layout_c.txt) structs matches C"

# download the pinned sokol-shdc binary for this host into build/tools/
shdc:
    python3 tools/fetch_shdc.py "{{shdc}}"

# regenerate generated/*.c3 from shaders/*.glsl with sokol-shdc (sokol-tools)
shaders:
    [ -x "{{shdc}}" ] || python3 tools/fetch_shdc.py "{{shdc}}"
    for s in radiance_cs radiance_rays_cs; do \
        {{shdc}} -i shaders/$s.glsl -o generated/$s.c3 -f sokol_c3 -l glsl430:hlsl5:metal_macos:metal_ios:metal_sim || exit 1; \
    done
    # these also in WGSL, for WebGPU (the others don't translate yet)
    for s in clear_rect depth_shadow kernels output gtao taa bloom flare grade bokeh vsm atlas lightmap gpu_cloth clouds; do \
        {{shdc}} -i shaders/$s.glsl -o generated/$s.c3 -f sokol_c3 -l glsl430:hlsl5:metal_macos:metal_ios:metal_sim:wgsl || exit 1; \
    done
    # sokol-shdc's C3 output writes storage image view slots as `const int VIEW_X (0)`;
    # rewrite them to `const int VIEW_X = 0;` (BSD sed -i '' syntax, macOS)
    sed -i '' -E 's/^const int (VIEW_[A-Z0-9_]+) \(([0-9]+)\)$/const int \1 = \2;/' generated/*.c3
    # each backend's shaders only in the builds for it
    python3 tools/shdc_gate.py generated/*.c3

# regenerate generated/ltc.c3, RectAreaLight's LTC tables, from three.js's
# RectAreaLightTexturesLib.js (driejs fetches three.js into driejs/vendor: `just driejs::build`)
ltc:
    python3 third_party/ltc/ltc.py driejs/vendor/jsm/lights/RectAreaLightTexturesLib.js generated/ltc.c3

# docs.html: the API of drie's modules, with their comments (tools/docgen.py)
docs:
    c3c docgen $(find src -name '*.c3' | sort) generated/*.c3 --emit-stdlib=no {{libs}} --json | python3 tools/docgen.py -o docs.html

# format drie's sources with c3fmt (see .c3fmt)
fmt:
    c3fmt --in-place $(find src -name '*.c3' | sort)

# check that drie's sources are formatted (same files as `just fmt`)
fmt-check:
    c3fmt --check $(find src -name '*.c3' | sort)
