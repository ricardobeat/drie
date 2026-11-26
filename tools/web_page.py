# Wraps an emscripten single-file build (the .js with the wasm and files inside) in one page.
#   python3 tools/web_page.py build/drie-web.js "drie: handkerchiefs" build/web/index.html
import sys

js, title, out = sys.argv[1], sys.argv[2], sys.argv[3]
code = open(js).read().replace("</script>", "<\\/script>")
page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>html,body{{margin:0;height:100%;background:#000;overflow:hidden}}canvas{{display:block;width:100%;height:100%}}
#status{{position:fixed;left:12px;top:10px;color:#ccc;font:13px system-ui}}
#by{{position:fixed;right:12px;bottom:10px;color:#bbb;font:13px system-ui;text-decoration:none;opacity:.75}}
#by:hover{{opacity:1}}</style></head>
<body><canvas id="canvas" tabindex="-1"></canvas><div id="status">loading...</div>
<a id="by" href="https://github.com/ricardobeat" target="_blank" rel="noopener">by @ricardobeat</a>
<script>
var Module = {{
  canvas: document.getElementById('canvas'),
  print: (t) => {{ console.log(t); (window.drieLog = window.drieLog || []).push(t); }},
  printErr: (t) => console.error(t),
  onRuntimeInitialized: () => document.getElementById('status').remove(),
}};
for (const k of ['error', 'warn']) {{
  const f = console[k].bind(console);
  console[k] = (...a) => {{ f(...a); (window.drieLog = window.drieLog || []).push(k + ': ' + a.join(' ')); }};
}}
window.addEventListener('error', (e) => (window.drieLog = window.drieLog || []).push('error: ' + e.message + ' ' + (e.error && e.error.stack)));
if (!navigator.gpu) document.getElementById('status').textContent = 'this page needs WebGPU';
</script>
<script>{code}</script>
</body></html>
"""
open(out, "w").write(page)
