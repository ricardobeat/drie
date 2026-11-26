#!/usr/bin/env python3
"""Downloads a pinned sokol-shdc binary from floooh/sokol-tools-bin and checks its SHA-256.

    python3 tools/fetch_shdc.py [output_path]

Writes build/tools/sokol-shdc (build/tools/sokol-shdc.exe on Windows) by default. If the
file is already there with the expected digest, nothing is downloaded.
"""
import argparse
import hashlib
import os
import platform
import stat
import sys
import urllib.request

COMMIT = "4f5b2796806875dcf0a5a5dfa9f9dab462726a5e"
BASE_URL = "https://raw.githubusercontent.com/floooh/sokol-tools-bin/" + COMMIT + "/"

# Platform key -> (path in sokol-tools-bin, expected SHA-256 hex digest).
BINARIES = {
    "osx_arm64": (
        "bin/osx_arm64/sokol-shdc",
        "54b98765886f4abd3bb86ffcf8c75c5073073a7fb009e7f693fb2f4d9ce3baa6",
    ),
    "osx": (
        "bin/osx/sokol-shdc",
        "8d8b74e0f612f556a0f57ea5a49c1c64bf990eed518001db626332a9ef887afc",
    ),
    "linux": (
        "bin/linux/sokol-shdc",
        "2554534631a7c7c6c2261dc6d20a603cdabf0cc1390e293f5ca06d345bb48c13",
    ),
    "linux_arm64": (
        "bin/linux_arm64/sokol-shdc",
        "da5e56ec9dc736b9fb7b43c881b2fe5616b2dd4beef7518dcfb5d60deb0eb50c",
    ),
    "win32": (
        "bin/win32/sokol-shdc.exe",
        "10bd643a869cbacff606c28aa44f79a55ad545981c907889b6fb28bc8977b380",
    ),
}

DRIE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def host_platform():
    system = platform.system()
    machine = platform.machine().lower()
    is_arm64 = machine in ("arm64", "aarch64")
    is_x64 = machine in ("x86_64", "amd64")
    if system == "Darwin":
        if is_arm64:
            return "osx_arm64"
        if is_x64:
            return "osx"
    elif system == "Linux":
        if is_arm64:
            return "linux_arm64"
        if is_x64:
            return "linux"
    elif system == "Windows":
        if is_x64 or is_arm64:
            return "win32"
    sys.exit("fetch_shdc.py: unsupported host %s/%s" % (system, machine))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(target, out_path):
    rel_path, expected = BINARIES[target]
    if os.path.isfile(out_path) and sha256_file(out_path) == expected:
        return

    os.makedirs(os.path.dirname(os.path.realpath(out_path)), exist_ok=True)
    url = BASE_URL + rel_path
    tmp = out_path + ".tmp"
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                h.update(chunk)
                out.write(chunk)
        got = h.hexdigest()
        if got != expected:
            os.remove(tmp)
            sys.exit(
                "fetch_shdc.py: SHA-256 mismatch for %s\n  expected: %s\n  got:      %s"
                % (url, expected, got)
            )
        if target != "win32":
            mode = os.stat(tmp).st_mode
            os.chmod(tmp, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.replace(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print("fetch_shdc.py: %s (%s)" % (os.path.relpath(out_path), target))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("output", nargs="?", help="destination path (default: build/tools/sokol-shdc)")
    ap.add_argument("--platform", choices=sorted(BINARIES), help="override host platform")
    args = ap.parse_args()

    target = args.platform or host_platform()
    default_name = "sokol-shdc.exe" if target == "win32" else "sokol-shdc"
    out_path = args.output or os.path.join(DRIE, "build", "tools", default_name)
    fetch(target, out_path)


if __name__ == "__main__":
    main()
