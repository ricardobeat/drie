# Keeps each backend's shaders out of the builds that can't use them: puts sokol-shdc's C3
# output (the shader text of every backend: `@if`, and each backend's case of the *_desc
# switches: `$if`) on the target, so a macOS build carries only Metal, the web only WGSL.
#   python3 tools/shdc_gate.py generated/*.c3
# (run after sokol-shdc; a file already gated is left as it is)
import re, sys

LANG = {
    "GLSL430": "GLCORE", "HLSL5": "D3D11", "METAL_MACOS": "METAL_MACOS",
    "METAL_IOS": "METAL_IOS", "METAL_SIM": "METAL_SIMULATOR", "WGSL": "WGPU",
}
WHEN = {
    "GLCORE": "!env::DARWIN && !env::WIN32 && !env::EMSCRIPTEN",
    "D3D11": "env::WIN32",
    "METAL_MACOS": "env::OS_TYPE == MACOS",
    "METAL_IOS": "env::OS_TYPE == IOS",
    "METAL_SIMULATOR": "env::OS_TYPE == IOS",
    "WGPU": "env::EMSCRIPTEN",
}
SOURCE = re.compile(r"^const char\[\d+\] \w+_SOURCE_(\w+?) @private = \{$")
CASE = re.compile(r"^(\s*)case SgBackend\.(\w+):\s*$")

for path in sys.argv[1:]:
    lines = open(path).read().split("\n")
    if any("@private @if(" in l for l in lines):
        continue
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = SOURCE.match(line)
        if m and m.group(1) in LANG:
            out.append(line.replace(" @private = {", f" @private @if({WHEN[LANG[m.group(1)]]}) = {{"))
            i += 1
            continue
        m = CASE.match(line)
        if m and m.group(2) in WHEN:
            indent = m.group(1)
            j = i + 1
            while j < len(lines):
                l = lines[j]
                body = len(l) - len(l.lstrip())
                if l.strip() and (body < len(indent) or (body == len(indent) and l.lstrip().startswith(("case ", "default")))):
                    break
                j += 1
            inner = indent + "    "
            out.append(line)
            out.append(f"{inner}$if {WHEN[m.group(2)]}:")
            out.extend(lines[i + 1:j])
            # (an empty case would fall through to the next backend's)
            out.append(f"{inner}$else")
            out.append(f"{inner}    break;")
            out.append(f"{inner}$endif")
            i = j
            continue
        out.append(line)
        i += 1
    open(path, "w").write("\n".join(out))
