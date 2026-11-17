#!/bin/sh
# Runs the benchmarks.
#   ./bench/run.sh [boomkat|boomkat-jit|boomkat-opt|node|node-jitless|bun|drie|drie-js|all]
# BOOMKAT=path overrides ../../boomkat/out/boomkat (boomkat-jit and boomkat-opt run it with
# --jit and --opt: point it at a build of the jit branch). drie-js runs ../../drie-js/out/drie_js.
set -e
cd "$(dirname "$0")/.."
BOOMKAT=${BOOMKAT:-../../boomkat/out/boomkat}
run() {
    echo "== $1"
    case $1 in
        boomkat) "$BOOMKAT" bench/three_bench.js ;;
        boomkat-jit) "$BOOMKAT" --jit bench/three_bench.js ;;
        boomkat-opt) "$BOOMKAT" --opt bench/three_bench.js ;;
        drie-js) (cd ../../drie-js && out/drie_js --script ../drie/driejs/bench/driejs_bench.js) ;;
        node) node bench/three_bench.js ;;
        node-jitless) node --jitless bench/three_bench.js ;;
        bun) bun bench/three_bench.js ;;
        drie)
            mkdir -p build
            c3c compile -O3 --macos-min-version 14.0 --libdir lib --lib sokol --lib readback --lib imageio \
                --lib gamectl --lib drie bench/drie_bench.c3 -o build/drie_bench >/dev/null
            build/drie_bench ;;
    esac
}
if [ "${1:-all}" = all ]; then
    for r in drie drie-js node bun node-jitless boomkat; do run $r; done
else
    run "$1"
fi
