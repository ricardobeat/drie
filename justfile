# drie. `just` lists the recipes; `just driejs::<recipe>` runs driejs's (three.js on drie),
# as does `just <recipe>` inside driejs/. Nothing here is part of the library: c3c only compiles
# the .c3 files at the top of this folder.

mod driejs

# drie's dependencies (sokol, readback, imageio, gamectl) and drie itself: the game's lib/.
# Another: `just libdir=path test radiance`.
libdir := "../house/lib"
libs := "--libdir " + libdir + " --lib sokol --lib readback --lib imageio --lib gamectl"
macos := if os() == "macos" { "--macos-min-version 14.0" } else { "" }

# the sources c3fmt formats: the hand written ones. The sokol-shdc generated
# files (gtao.c3, kernels.c3, output.c3, radiance_cs.c3, shader.c3, taa.c3) are
# left out: they are machine made, and c3fmt is very slow on them.
c3fmt_files := "audio.c3 clock.c3 clusters.c3 compute.c3 controls.c3 core.c3 culling.c3 display.c3 fog.c3 gamepad.c3 geometry.c3 gltf.c3 irradiance.c3 material.c3 math.c3 menu.c3 postprocessing.c3 profile.c3 radiance.c3 raycaster.c3 reflection_probes.c3 reflector.c3 renderer.c3 skeleton.c3 sky.c3 texture.c3 timeline.c3 touch.c3"


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

# format drie's sources with c3fmt (see .c3fmt)
fmt:
    c3fmt --in-place {{c3fmt_files}}

# check that drie's sources are formatted (same files as `just fmt`)
fmt-check:
    c3fmt --check {{c3fmt_files}}
