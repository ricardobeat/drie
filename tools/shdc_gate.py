# Keeps each backend's shaders out of builds that cannot use them and stores shader text as
# raw C3 string literals instead of hex byte arrays + duplicate block comments:
#   - strips the `/* ... */` comment copy above each `const char[N] *_SOURCE_*` array
#   - decodes the hex bytes into `const ZString NAME @private @feat(...) = `...`;`
#   - rewrites `(ZString)&NAME` references to `NAME` and wraps each backend `case` in `$if`
#
#   python3 tools/shdc_gate.py generated/*.c3
# (run after sokol-shdc; a file already processed is left as it is)
import re
import sys

LANG = {
    "GLSL430": "GLCORE",
    "HLSL5": "D3D11",
    "METAL_MACOS": "METAL_MACOS",
    "METAL_IOS": "METAL_IOS",
    "METAL_SIM": "METAL_SIMULATOR",
    "WGSL": "WGPU",
}
FEAT = {
    "GLCORE": "!DARWIN & !WIN32 & !EMSCRIPTEN",
    "D3D11": "WIN32",
    "METAL_MACOS": "MACOS",
    "METAL_IOS": "IOS",
    "METAL_SIMULATOR": "IOS",
    "WGPU": "EMSCRIPTEN",
}
WHEN = {
    "GLCORE": "!env::DARWIN && !env::WIN32 && !env::EMSCRIPTEN",
    "D3D11": "env::WIN32",
    "METAL_MACOS": "env::OS_TYPE == MACOS",
    "METAL_IOS": "env::OS_TYPE == IOS",
    "METAL_SIMULATOR": "env::OS_TYPE == IOS",
    "WGPU": "env::EMSCRIPTEN",
}

SOURCE = re.compile(r"^const char\[\d+\] (\w+_SOURCE_(\w+?)) @private = \{$")
CASE = re.compile(r"^(\s*)case SgBackend\.(\w+):\s*$")
HEX = re.compile(r"0x([0-9a-fA-F]{2})")
REF = re.compile(r"\(ZString\)&(\w+_SOURCE_\w+)")

for path in sys.argv[1:]:
    lines = open(path, encoding="utf-8").read().split("\n")
    if any("@private @feat(" in l or "@private @if(" in l for l in lines):
        continue
    out = []
    converted = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = SOURCE.match(line)
        if m and m.group(2) in LANG:
            name, lang = m.group(1), m.group(2)
            # Drop the duplicate `/* ... */` shader-source comment directly above the array.
            if out and out[-1].strip() == "*/":
                k = len(out) - 1
                while k >= 0 and out[k].strip() != "/*":
                    k -= 1
                if k >= 0:
                    del out[k:]
            j = i + 1
            hex_lines = []
            while j < len(lines) and lines[j].strip() != "};":
                hex_lines.append(lines[j])
                j += 1
            raw = bytes(int(h, 16) for h in HEX.findall(" ".join(hex_lines)))
            if raw.endswith(b"\x00"):
                raw = raw[:-1]
            text = raw.decode("utf-8").replace("`", "``")
            feat = FEAT[LANG[lang]]
            out.append(f"const ZString {name} @private @feat({feat}) = `{text}`;")
            converted.add(name)
            i = j + 1
            continue

        m = CASE.match(line)
        if m and m.group(2) in WHEN:
            indent = m.group(1)
            j = i + 1
            while j < len(lines):
                l = lines[j]
                body = len(l) - len(l.lstrip())
                if l.strip() and (
                    body < len(indent)
                    or (body == len(indent) and l.lstrip().startswith(("case ", "default")))
                ):
                    break
                j += 1
            inner = indent + "    "
            out.append(line)
            out.append(f"{inner}$if {WHEN[m.group(2)]}:")
            for l in lines[i + 1 : j]:
                out.append(REF.sub(lambda rm: rm.group(1) if rm.group(1) in converted else rm.group(0), l))
            # An empty case would fall through to the next backend's.
            out.append(f"{inner}$else")
            out.append(f"{inner}    break;")
            out.append(f"{inner}$endif")
            i = j
            continue

        out.append(REF.sub(lambda rm: rm.group(1) if rm.group(1) in converted else rm.group(0), line))
        i += 1

    open(path, "w", encoding="utf-8").write("\n".join(out))
