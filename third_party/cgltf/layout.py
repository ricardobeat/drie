#!/usr/bin/env python3
# Checks that generated/cgltf.c3 lays cgltf's structs out as C does: from bindgen's dump of
# cgltf.h (tools/bgdump), it writes a C program and a C3 program that print every struct's
# size and alignment and every field's offset, for `just bindings` to build, run and compare.
#
#   python3 layout.py <dump.txt> <out dir>   (writes layout.c and layout.c3 there)
import os
import sys


def pascal(name):
    return "".join(w[:1].upper() + w[1:] for w in name.split("_"))


def main():
    dump, out = sys.argv[1], sys.argv[2]
    records = []
    current = None
    stack = []
    for line in open(dump).read().splitlines():
        f = line.split("\t")
        if f[0] in ("struct", "union"):
            current = (f[0], f[1], [])
            records.append(current)
            stack = []
        elif f[0] == "field" and current:
            # (a member of an anonymous inner record has no path to name it by)
            if f[1] and None not in stack:
                current[2].append(".".join(stack + [f[1]]))
        elif f[0] == "record" and current:
            if f[1] and None not in stack:
                current[2].append(".".join(stack + [f[1]]))
            stack.append(f[1] or None)
        elif f[0] == "rend":
            stack.pop()
        elif f[0] == "end":
            current = None
    c = ["#include <stdio.h>", "#include <stddef.h>", '#include "cgltf.h"', "int main(void) {"]
    c3 = ["module cgltf_layout;", "import drie::cgltf, std::io;", "fn void main() {"]
    for kind, tag, fields in records:
        t = pascal(tag)
        c.append(f'printf("{tag} %zu %zu\\n", sizeof({kind} {tag}), _Alignof({kind} {tag}));')
        c3.append(f'io::printfn("{tag} %d %d", {t}::size, {t}::alignment);')
        for field in fields:
            c.append(f'printf("  {field} %zu\\n", offsetof({kind} {tag}, {field}));')
            c3.append(f'{{ {t} v; io::printfn("  {field} %d", (usz)&v.{field} - (usz)&v); }}')
    c.append("return 0; }")
    c3.append("}")
    open(os.path.join(out, "layout.c"), "w").write("\n".join(c) + "\n")
    open(os.path.join(out, "layout.c3"), "w").write("\n".join(c3) + "\n")
    print(f"{len(records)} structs, {sum(len(r[2]) for r in records)} fields")


main()
