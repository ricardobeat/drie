# drie. `just` lists the recipes; `just driejs::<recipe>` runs driejs's (three.js on drie),
# as does `just <recipe>` inside driejs/. Nothing here is part of the library: c3c only compiles
# the .c3 files at the top of this folder.

mod driejs

# where drie's dependency, sokol, is (lib/sokol.c3l).
# Another: `just libdir=path test radiance`.
libdir := "lib"
libs := "--libdir " + libdir + " --lib sokol"
# where bindgen.c3l is (the folder holding bindgen.c3l/), for `just bindings`
bindgen := "../bindgen.c3l"
macos := if os() == "macos" { "--macos-min-version 26.0" } else { "" }

# the sources c3fmt formats: the hand written ones at the top. The sokol-shdc generated
# files are in generated/ (manifest.json's "sources" adds it to the library).
c3fmt_files := "*.c3"


[private]
default:
    @just --list

# type-check drie, without building a program
check:
    c3c compile-only {{libs}} --obj-out build/check *.c3 generated/*.c3 platform/*.c3

# build and run a rendering test, headless: `just test radiance` writes tests/build/radiance.png
# (`just test radiance out.png --p3`: elsewhere, and in Display P3's colors)
test name *args:
    mkdir -p tests/build
    c3c compile {{macos}} {{libs}} --lib drie tests/{{name}}.c3 -o tests/build/{{name}}
    tests/build/{{name}} {{args}}

# build and run an example program (examples/): `just example nodes`
example name *args:
    mkdir -p examples/build
    c3c compile {{macos}} {{libs}} --lib drie examples/{{name}}.c3 -o examples/build/{{name}}
    examples/build/{{name}} {{args}}

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

# regenerate generated/*.c3 from the .glsl files with sokol-shdc (sokol-tools)
shaders:
    for s in shader kernels output gtao taa bloom bokeh radiance_cs vsm lightmap; do \
        sokol-shdc -i $s.glsl -o generated/$s.c3 -f sokol_c3 -l glsl430:hlsl5:metal_macos:metal_ios:metal_sim || exit 1; \
    done
    # sokol-shdc's C3 output writes storage image view slots as `const int VIEW_X (0)`;
    # rewrite them to `const int VIEW_X = 0;` (BSD sed -i '' syntax, macOS)
    sed -i '' -E 's/^const int (VIEW_[A-Z0-9_]+) \(([0-9]+)\)$/const int \1 = \2;/' generated/*.c3

# regenerate generated/ltc.c3, RectAreaLight's LTC tables, from three.js's
# RectAreaLightTexturesLib.js (driejs fetches three.js into driejs/vendor: `just driejs::build`)
ltc:
    python3 third_party/ltc/ltc.py driejs/vendor/jsm/lights/RectAreaLightTexturesLib.js generated/ltc.c3

# docs.html: the API of drie's modules, with their comments (docgen.py)
docs:
    c3c docgen *.c3 generated/*.c3 platform/*.c3 --emit-stdlib=no {{libs}} --json | python3 docgen.py -o docs.html

# format drie's sources with c3fmt (see .c3fmt)
fmt:
    c3fmt --in-place {{c3fmt_files}}

# check that drie's sources are formatted (same files as `just fmt`)
fmt-check:
    c3fmt --check {{c3fmt_files}}
