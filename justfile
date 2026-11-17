# drie. `just` lists the recipes; `just driejs::<recipe>` runs driejs's (three.js on drie),
# as does `just <recipe>` inside driejs/. Nothing here is part of the library: c3c only compiles
# the .c3 files at the top of this folder.

mod driejs

# where drie's dependency, sokol, is (lib/sokol.c3l).
# Another: `just libdir=path test radiance`.
libdir := "lib"
libs := "--libdir " + libdir + " --lib sokol"
macos := if os() == "macos" { "--macos-min-version 15.0" } else { "" }

# the sources c3fmt formats: the hand written ones at the top. The sokol-shdc generated
# files are in generated/ (manifest.json's "sources" adds it to the library).
c3fmt_files := "*.c3"


[private]
default:
    @just --list

# type-check drie, without building a program
check:
    c3c compile-only {{libs}} --obj-out build/check *.c3 generated/*.c3 platform/*.c3

# build and run a rendering test, headless: `just test radiance` writes tests/build/radiance.ppm
test name out=("tests/build/" + name + ".ppm"):
    mkdir -p tests/build
    c3c compile {{macos}} {{libs}} --lib drie tests/{{name}}.c3 -o tests/build/{{name}}
    tests/build/{{name}} {{out}}

# regenerate generated/*.c3 from the .glsl files with sokol-shdc (sokol-tools)
shaders:
    for s in shader kernels output gtao taa radiance_cs; do \
        sokol-shdc -i $s.glsl -o generated/$s.c3 -f sokol_c3 -l glsl430:hlsl5:metal_macos:metal_ios:metal_sim || exit 1; \
    done

# docs.html: the API of drie's modules, with their comments (docgen.py)
docs:
    c3c docgen *.c3 generated/*.c3 platform/*.c3 --emit-stdlib=no {{libs}} --json | python3 docgen.py -o docs.html

# format drie's sources with c3fmt (see .c3fmt)
fmt:
    c3fmt --in-place {{c3fmt_files}}

# check that drie's sources are formatted (same files as `just fmt`)
fmt-check:
    c3fmt --check {{c3fmt_files}}
