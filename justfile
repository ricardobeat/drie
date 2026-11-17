# drie. `just` lists the recipes; `just driejs::<recipe>` runs driejs's (three.js on drie),
# as does `just <recipe>` inside driejs/. Nothing here is part of the library: c3c only compiles
# the .c3 files at the top of this folder.

mod driejs

# drie's dependencies (sokol, readback, imageio, gamectl) and drie itself: the game's lib/.
# Another: `just libdir=path test radiance`.
libdir := "../house/lib"
libs := "--libdir " + libdir + " --lib sokol --lib readback --lib imageio --lib gamectl"
macos := if os() == "macos" { "--macos-min-version 14.0" } else { "" }

[private]
default:
    @just --list

# type-check drie, without building a program
check:
    c3c compile-only {{libs}} --obj-out build/check *.c3

# build and run a rendering test, headless: `just test radiance` writes tests/build/radiance.ppm
test name out=("tests/build/" + name + ".ppm"):
    mkdir -p tests/build
    c3c compile {{macos}} {{libs}} --lib drie tests/{{name}}.c3 -o tests/build/{{name}}
    tests/build/{{name}} {{out}}

# docs.html: the API of drie's modules, with their comments (docgen.py)
docs:
    c3c docgen *.c3 --emit-stdlib=no {{libs}} --json | python3 docgen.py -o docs.html
