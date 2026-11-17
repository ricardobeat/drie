#!/usr/bin/env python3
"""Build docs.html from `c3c docgen --json`, keeping only our own modules.

    c3c docgen *.c3 generated/*.c3 platform/*.c3 --emit-stdlib=no --libdir lib \
        --lib sokol --json \
      | python3 docgen.py -o docs.html

c3c's JSON has no doc comments, so they are read from the sources: the `//`
lines right above a declaration, and for struct fields and enum values also a
trailing `//` on the same line. Run it from the directory the `file` paths in
the JSON are relative to (the one c3c ran in).
"""
import argparse
import html
import json
import re
import sys

KIND_ORDER = ["struct", "union", "bitstruct", "enum", "constset", "interface",
              "alias", "distinct type", "fault"]
KIND_KEYWORD = {"constset": "constdef", "distinct type": "typedef"}

_sources = {}


def source_lines(path):
    if path not in _sources:
        try:
            with open(path, encoding="utf-8") as f:
                _sources[path] = f.read().split("\n")
        except OSError:
            _sources[path] = []
    return _sources[path]


def location(item):
    path, line, _ = (item.get("file") or "::").rsplit(":", 2)
    return path, int(line) if line.isdigit() else 0


def comment_above(lines, idx):
    """The run of `//` lines directly above lines[idx], oldest first."""
    out = []
    i = idx - 1
    while i >= 0 and lines[i].strip().startswith("//"):
        out.append(lines[i].strip()[2:])
        i -= 1
    out.reverse()
    # Drop the one space after `//`, keep any further indent (marks code).
    return [l[1:] if l.startswith(" ") else l for l in out]


def doc_for(item):
    path, line = location(item)
    lines = source_lines(path)
    if not 0 < line <= len(lines):
        return []
    return comment_above(lines, line - 1)


def member_docs(item):
    """Comments for struct fields / enum values, read from the body in source."""
    path, line = location(item)
    lines = source_lines(path)
    if not 0 < line <= len(lines):
        return {}
    # Direct children of the declaration's outermost braces.
    body, depth, started = [], 0, False
    for i in range(line - 1, len(lines)):
        text = lines[i].split("//", 1)[0]
        if depth == 1 and started:
            body.append(i)
        for ch in text:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
        if started and depth == 0:
            break
    docs = {}
    for m in item.get("members", []):
        name = m.get("name")
        if not name:
            continue
        pat = re.compile(r"\b%s\b\s*(?:[;,=:(]|$)" % re.escape(name))
        for i in body:
            code, _, trailing = lines[i].partition("//")
            if pat.search(code):
                doc = comment_above(lines, i)
                if trailing.strip():
                    doc = doc + [trailing.strip()]
                if doc:
                    docs[name] = doc
                break
    return docs


# ---------------------------------------------------------------- rendering

def esc(s):
    return html.escape(str(s), quote=True)


def inline(text):
    parts = re.split(r"(`[^`]+`)", text)
    return "".join("<code>%s</code>" % esc(p[1:-1]) if p.startswith("`") and len(p) > 1
                   else esc(p) for p in parts)


def render_doc(lines):
    """Blank lines split paragraphs; lines indented further are code."""
    if not lines:
        return ""
    out, para, code = [], [], []

    def flush():
        if para:
            out.append("<p>%s</p>" % inline(" ".join(para)))
            para.clear()
        if code:
            out.append("<pre>%s</pre>" % esc("\n".join(code)))
            code.clear()

    for l in lines:
        if not l.strip():
            flush()
        elif l.startswith("  "):
            if para:
                flush()
            code.append(l[2:])
        else:
            if code:
                flush()
            para.append(l.strip())
    flush()
    return '<div class="doc">%s</div>' % "".join(out)


class Page:
    def __init__(self, anchors):
        self.anchors = anchors  # uids that have an anchor on the page

    def type_ref(self, t):
        if not t:
            return ""
        name, uid = t.get("name", ""), t.get("uid")
        if uid in self.anchors:
            short = uid.rsplit("::", 1)[-1]
            i = name.find(short)
            if i >= 0:
                return '%s<a href="#%s">%s</a>%s' % (
                    esc(name[:i]), esc(uid), esc(short), esc(name[i + len(short):]))
        return esc(name)

    def params(self, members):
        out = []
        for m in members:
            if m.get("is_ref"):
                s = "&amp;" + esc(m.get("name", "self"))
            else:
                bits = []
                if "type" in m:
                    bits.append('<span class="ty">%s</span>' % self.type_ref(m["type"]))
                if m.get("name"):
                    bits.append(esc(m["name"]))
                s = " ".join(bits)
            if "default_value" in m:
                s += ' = <span class="val">%s</span>' % esc(m["default_value"])
            out.append(s)
        return ", ".join(out)

    def fn_sig(self, f, owner=None):
        kw = "macro" if f["kind"].startswith("macro") else "fn"
        ret = self.type_ref(f.get("return_type"))
        name = esc(f["name"])
        if owner:
            name = "%s.%s" % (esc(owner), name)
        gen = ""
        if f.get("generic_parameters") and not owner:
            gen = " {%s}" % esc(", ".join(f["generic_parameters"]))
        return '<span class="kw">%s</span> <span class="ty">%s</span> <b>%s</b>%s(%s)' % (
            kw, ret, name, gen, self.params(f.get("members", [])))

    def type_sig(self, t):
        kind = t["kind"]
        kw = KIND_KEYWORD.get(kind, kind)
        name = "<b>%s</b>" % esc(t["name"])
        if t.get("generic_parameters"):
            name += " {%s}" % esc(", ".join(t["generic_parameters"]))
        if kind == "alias" and "return_type" in t:
            return '<span class="kw">alias</span> %s = <span class="kw">fn</span> <span class="ty">%s</span>(%s)' % (
                name, self.type_ref(t["return_type"]), self.params(t.get("members", [])))
        if kind == "alias":
            return '<span class="kw">alias</span> %s = <span class="ty">%s</span>' % (
                name, self.type_ref(t.get("base_type")))
        if kind == "fault":
            return '<span class="kw">faultdef</span> %s' % name
        base = ""
        if t.get("base_type"):
            sep = " =" if kind == "distinct type" else " :"
            base = '%s <span class="ty">%s</span>' % (sep, self.type_ref(t["base_type"]))
        return '<span class="kw">%s</span> %s%s' % (kw, name, base)


def vis_badge(item):
    v = item.get("visibility")
    return ' <span class="badge">@%s</span>' % v if v else ""


def src_link(item):
    return '<span class="src">%s</span>' % esc(item.get("file", ""))


def build(modules, title, show_all):
    def visible(x):
        return show_all or not x.get("visibility")

    # Collect everything first so links only point at anchors that exist.
    mods = []
    for mname in sorted(modules):
        m = modules[mname]
        types = [t for t in m.get("types", []) if visible(t)]
        methods = m.get("methods", []) + m.get("macro_methods", [])
        funcs = [f for f in m.get("functions", []) + m.get("macros", []) if visible(f)]
        consts = [v for v in m.get("variables", []) if visible(v)]
        by_owner = {}
        for f in methods:
            if visible(f):
                by_owner.setdefault(f["uid"].rsplit(".", 1)[0], []).append(f)
        mods.append((mname, m, types, by_owner, funcs, consts))

    anchors = set()
    for mname, m, types, by_owner, funcs, consts in mods:
        anchors.add(mname)
        anchors.update(t["uid"] for t in types)
        anchors.update(by_owner)
        anchors.update(f["uid"] for f in funcs)
        anchors.update(v["uid"] for v in consts)
    page = Page(anchors)

    nav, main = [], []
    for mname, m, types, by_owner, funcs, consts in mods:
        mid = esc(mname)
        gen = " {%s}" % esc(", ".join(m["generic_parameters"])) if m.get("generic_parameters") else ""
        main.append('<section class="module"><h1 id="%s"><span class="kw">module</span> %s%s</h1>' % (mid, mid, gen))
        nav.append('<details open><summary><a href="#%s">%s</a></summary>' % (mid, mid))

        types.sort(key=lambda t: (KIND_ORDER.index(t["kind"]) if t["kind"] in KIND_ORDER else 99, t["name"].lower()))
        type_uids = {t["uid"] for t in types}
        # Methods on types this module doesn't declare (or hides) still get a home.
        foreign = sorted(u for u in by_owner if u not in type_uids)

        if types or foreign:
            main.append("<h2>Types</h2>")
            nav.append('<div class="navh">Types</div><ul>')
            for t in types:
                uid = esc(t["uid"])
                nav.append('<li data-k="%s"><a href="#%s">%s</a></li>' % (esc(t["name"].lower()), uid, esc(t["name"])))
                main.append('<article class="item" id="%s"><div class="sig">%s%s</div>%s%s' % (
                    uid, page.type_sig(t), vis_badge(t), src_link(t), render_doc(doc_for(t))))
                main.append(render_members(page, t))
                main.append(render_methods(page, t["name"], by_owner.get(t["uid"], [])))
                main.append("</article>")
            for owner in foreign:
                short = owner.rsplit("::", 1)[-1]
                nav.append('<li data-k="%s"><a href="#%s">%s</a></li>' % (esc(short.lower()), esc(owner), esc(short)))
                main.append('<article class="item" id="%s"><div class="sig"><span class="kw">methods on</span> <b>%s</b></div>' % (
                    esc(owner), esc(owner)))
                main.append(render_methods(page, short, by_owner[owner]))
                main.append("</article>")
            nav.append("</ul>")

        if funcs:
            funcs.sort(key=lambda f: f["name"].lower())
            main.append("<h2>Functions</h2>")
            nav.append('<div class="navh">Functions</div><ul>')
            for f in funcs:
                nav.append('<li data-k="%s"><a href="#%s">%s</a></li>' % (esc(f["name"].lower()), esc(f["uid"]), esc(f["name"])))
                main.append('<article class="item fn" id="%s"><div class="sig">%s%s</div>%s%s</article>' % (
                    esc(f["uid"]), page.fn_sig(f), vis_badge(f), src_link(f), render_doc(doc_for(f))))
            nav.append("</ul>")

        if consts:
            consts.sort(key=lambda v: (not v.get("is_const"), v["name"].lower()))
            cid = esc(mname + ":constants")
            main.append('<h2 id="%s">Constants &amp; globals</h2><table class="consts">' % cid)
            nav.append('<a class="navh navlink" href="#%s">Constants</a>' % cid)
            for v in consts:
                kw = "const" if v.get("is_const") else "global"
                value = ' = <span class="val">%s</span>' % esc(v["value"]) if "value" in v else ""
                main.append('<tr id="%s"><td class="sig"><span class="kw">%s</span> <span class="ty">%s</span> <b>%s</b>%s%s</td><td>%s</td></tr>' % (
                    esc(v["uid"]), kw, page.type_ref(v.get("type")), esc(v["name"]), value, vis_badge(v),
                    render_doc(doc_for(v))))
            main.append("</table>")

        nav.append("</details>")
        main.append("</section>")

    return TEMPLATE.replace("{{title}}", esc(title)).replace("{{nav}}", "\n".join(nav)).replace("{{main}}", "\n".join(main))


def render_members(page, t):
    members = t.get("members")
    if not members or (t["kind"] == "alias" and "return_type" in t):
        return ""
    docs = member_docs(t)
    rows = []
    for m in members:
        if t["kind"] in ("enum", "constset"):
            left = "<b>%s</b>" % esc(m.get("name", ""))
            if "value" in m:
                left += ' = <span class="val">%s</span>' % esc(m["value"])
        else:
            left = '<span class="ty">%s</span> <b>%s</b>' % (page.type_ref(m.get("type")), esc(m.get("name", "")))
        rows.append('<tr><td class="sig">%s</td><td>%s</td></tr>' % (left, render_doc(docs.get(m.get("name"), []))))
    return '<table class="members">%s</table>' % "".join(rows)


def render_methods(page, owner, methods):
    if not methods:
        return ""
    out = ['<div class="methods">']
    for f in sorted(methods, key=lambda f: f["name"].lower()):
        out.append('<div class="method" id="%s"><div class="sig">%s</div>%s%s</div>' % (
            esc(f["uid"]), page.fn_sig(f, owner), src_link(f), render_doc(doc_for(f))))
    out.append("</div>")
    return "".join(out)


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{title}}</title>
<style>
:root {
  --bg: #fbfbf9; --panel: #f2f1ed; --fg: #1d1d1b; --muted: #6b6a64; --line: #e0ded7;
  --kw: #8a3ffc; --ty: #0f6e6e; --val: #a1531a; --link: #1f5fbf; --code: #ecebe6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #17171a; --panel: #1e1e22; --fg: #e6e4de; --muted: #9a988f; --line: #2e2e33;
    --kw: #c39bff; --ty: #5ec4c4; --val: #e8a26a; --link: #7fb0ff; --code: #26262b;
  }
}
* { box-sizing: border-box; }
html { scroll-padding-top: 16px; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
code, pre, .sig, .src { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
nav { position: fixed; top: 0; bottom: 0; left: 0; width: 270px; overflow-y: auto;
  background: var(--panel); border-right: 1px solid var(--line); padding: 14px 12px 40px; font-size: 13.5px; }
nav input { width: 100%; padding: 7px 9px; margin-bottom: 10px; border: 1px solid var(--line);
  border-radius: 6px; background: var(--bg); color: var(--fg); font: inherit; }
nav summary { font-weight: 600; cursor: pointer; padding: 4px 0; }
nav summary a { color: var(--fg); }
nav ul { list-style: none; margin: 0 0 6px; padding: 0 0 0 12px; }
nav li a { display: block; padding: 1px 0; color: var(--fg); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.navh { color: var(--muted); font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; margin: 6px 0 2px 12px; }
a.navlink { display: block; margin-bottom: 6px; }
main { margin-left: 270px; padding: 24px 40px 80px; max-width: 1060px; }
h1 { font-size: 24px; margin: 36px 0 8px; padding-bottom: 6px; border-bottom: 2px solid var(--line); }
h1:first-child { margin-top: 0; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 28px 0 10px; }
.item { border-top: 1px solid var(--line); padding: 14px 0; position: relative; }
.item:target, .method:target, tr:target { background: color-mix(in srgb, var(--link) 9%, transparent); }
.sig { font-size: 13.5px; overflow-wrap: anywhere; }
.sig b { font-weight: 650; }
.kw { color: var(--kw); }
.ty, .ty a { color: var(--ty); }
.val { color: var(--val); }
.src { position: absolute; right: 0; top: 15px; font-size: 11.5px; color: var(--muted); }
.method .src, .consts .src { position: static; }
.badge { font: 11px ui-monospace, monospace; color: var(--muted); border: 1px solid var(--line);
  border-radius: 4px; padding: 0 4px; margin-left: 6px; }
.doc { color: var(--fg); margin-top: 4px; }
.doc p { margin: 4px 0; }
.doc code { background: var(--code); border-radius: 3px; padding: 0 3px; font-size: 90%; }
.doc pre { background: var(--code); border-radius: 6px; padding: 8px 10px; margin: 6px 0; overflow-x: auto; font-size: 13px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 0; font-size: 14px; }
td { padding: 4px 10px 4px 0; vertical-align: top; border-top: 1px solid var(--line); }
td.sig { white-space: nowrap; width: 1%; padding-right: 22px; }
td .doc { margin: 0; }
.members { margin-left: 14px; width: calc(100% - 14px); }
.methods { margin: 10px 0 0 14px; }
.method { padding: 8px 0 8px 12px; border-left: 2px solid var(--line); margin-bottom: 4px; }
.method .src { font-size: 11px; }
.hidden { display: none !important; }
@media (max-width: 760px) {
  nav { position: static; width: auto; max-height: 45vh; border-right: 0; border-bottom: 1px solid var(--line); }
  main { margin-left: 0; padding: 16px; }
  .src { position: static; display: block; }
  td.sig { white-space: normal; }
}
</style>
</head>
<body>
<nav>
<input id="q" type="search" placeholder="Filter&hellip;" autocomplete="off">
{{nav}}
</nav>
<main>
{{main}}
</main>
<script>
const q = document.getElementById('q');
q.addEventListener('input', () => {
  const s = q.value.trim().toLowerCase();
  for (const li of document.querySelectorAll('nav li')) li.classList.toggle('hidden', !!s && !li.dataset.k.includes(s));
  for (const ul of document.querySelectorAll('nav ul')) {
    const empty = !ul.querySelector('li:not(.hidden)');
    ul.classList.toggle('hidden', empty);
    ul.previousElementSibling.classList.toggle('hidden', empty);
  }
});
q.addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  const a = document.querySelector('nav li:not(.hidden) a');
  if (a) location.hash = a.getAttribute('href');
});
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="?", default="-", help="c3c docgen --json output (default: stdin)")
    ap.add_argument("-o", "--output", default="docs.html")
    ap.add_argument("-m", "--module", action="append", default=None,
                    help="module prefix to keep, repeatable (default: drie)")
    ap.add_argument("-x", "--exclude", action="append", default=None,
                    help="module prefix to leave out, repeatable (default: drie::shaders, the generated shaders)")
    ap.add_argument("--all", action="store_true", help="include @private and @local items")
    ap.add_argument("--title", default="drie")
    args = ap.parse_args()

    src = sys.stdin if args.json == "-" else open(args.json, encoding="utf-8")
    data = json.load(src)
    prefixes = args.module or ["drie"]
    excluded = args.exclude if args.exclude is not None else ["drie::shaders"]

    def under(k, ps):
        return any(k == p or k.startswith(p + "::") for p in ps)

    modules = {k: v for k, v in data["modules"].items() if under(k, prefixes) and not under(k, excluded)}
    if not modules:
        sys.exit("no modules matching %s" % ", ".join(prefixes))
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(build(modules, args.title, args.all))
    print("%s: %s" % (args.output, ", ".join(sorted(modules))), file=sys.stderr)


if __name__ == "__main__":
    main()
