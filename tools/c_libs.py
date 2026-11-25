#!/usr/bin/env python3
"""Builds the C sources of drie and sokol into static libraries, one per target.

    tools/c_libs.py <target> [library folder...]

The targets are macos-aarch64, ios-aarch64, ios-aarch64-sim and mingw-x64. Without folders it
builds drie's (this repository) and sokol's (lib/sokol.c3l). For each library it reads the
manifest's c-sources and the target's cflags-override and writes
<library>/build/c/<target>/lib<provides>_c.a, which the manifest names in that target's
linked-libraries (with linklib-dir build/c and c-sources-override [], so c3c compiles no C).

An object is rebuilt when its source, a header it includes (the compiler's .d file) or its
command line changed, so a run with nothing to do only reads timestamps.
"""
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

DRIE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# the oldest OS versions the games run on: house links with these
MACOS_MIN = "14.0"
IOS_MIN = "17.0"


def sdk(name):
    return subprocess.check_output(["xcrun", "--sdk", name, "--show-sdk-path"], text=True).strip()


def toolchain(target):
    """The compiler and its target flags, and the archiver, for a target."""
    if target == "macos-aarch64":
        return ["cc", "-target", "arm64-apple-macos" + MACOS_MIN], ["ar"]
    if target == "ios-aarch64":
        return ["cc", "-target", "arm64-apple-ios" + IOS_MIN, "-isysroot", sdk("iphoneos")], ["ar"]
    if target == "ios-aarch64-sim":
        return ["cc", "-target", "arm64-apple-ios" + IOS_MIN + "-simulator",
                "-isysroot", sdk("iphonesimulator")], ["ar"]
    if target == "mingw-x64":
        return ["zig", "cc", "-target", "x86_64-windows-gnu"], ["zig", "ar"]
    sys.exit("c_libs.py: no toolchain for target " + target)


def stale(obj, dep, cmd_file, cmd):
    """Whether obj must be compiled again: missing, another command, or older than a file it read."""
    try:
        t = os.path.getmtime(obj)
        with open(cmd_file) as f:
            if f.read() != cmd:
                return True
        with open(dep) as f:
            text = f.read().replace("\\\n", " ")
    except OSError:
        return True
    for path in shlex.split(text.split(":", 1)[1]) if ":" in text else []:
        try:
            if os.path.getmtime(path) > t:
                return True
        except OSError:
            return True
    return False


def build(lib_dir, target):
    lib_dir = os.path.realpath(lib_dir)
    with open(os.path.join(lib_dir, "manifest.json")) as f:
        manifest = json.load(f)
    sources = manifest.get("c-sources", [])
    flags = shlex.split(manifest.get("targets", {}).get(target, {}).get("cflags-override",
                                                                          manifest.get("cflags", "")))
    out = os.path.join(lib_dir, "build", "c", target)
    obj_dir = os.path.join(out, "obj")
    os.makedirs(obj_dir, exist_ok=True)
    archive = os.path.join(out, "lib" + manifest["provides"] + "_c.a")
    cc, ar = toolchain(target)

    jobs = []
    objs = []
    for src in sources:
        name = src.replace("/", "_").rsplit(".", 1)[0]
        obj = os.path.join(obj_dir, name + ".o")
        dep = os.path.join(obj_dir, name + ".d")
        cmd_file = os.path.join(obj_dir, name + ".cmd")
        cmd = cc + flags + ["-c", os.path.join(lib_dir, src)]
        objs.append(obj)
        if stale(obj, dep, cmd_file, shlex.join(cmd)):
            jobs.append((cmd, obj, dep, cmd_file))

    def compile_one(job):
        cmd, obj, dep, cmd_file = job
        tmp = obj + ".tmp"
        r = subprocess.run(cmd + ["-o", tmp, "-MMD", "-MF", dep, "-MT", obj])
        if r.returncode != 0:
            return False
        os.replace(tmp, obj)
        with open(cmd_file, "w") as f:
            f.write(shlex.join(cmd))
        return True

    if jobs:
        with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
            if not all(pool.map(compile_one, jobs)):
                sys.exit("c_libs.py: %s: a C source failed to compile for %s" % (manifest["provides"], target))
    if jobs or not os.path.exists(archive) or \
            any(os.path.getmtime(o) > os.path.getmtime(archive) for o in objs):
        tmp = archive + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        subprocess.run(ar + ["rcs", tmp] + objs, check=True)
        os.replace(tmp, archive)
        print("c_libs.py: %s (%d of %d C files compiled)" % (os.path.relpath(archive), len(jobs), len(objs)))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = sys.argv[1]
    libs = sys.argv[2:] or [DRIE, os.path.join(DRIE, "lib", "sokol.c3l")]
    with ThreadPoolExecutor(len(libs)) as pool:
        for _ in pool.map(lambda lib: build(lib, target), libs):
            pass


if __name__ == "__main__":
    main()
