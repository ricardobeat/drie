# Wraps an emscripten single-file build (the .js with the wasm and files inside) in one page.
#   python3 tools/web_page.py build/drie-web.js "drie: handkerchiefs" build/web/index.html
import sys

js, title, out = sys.argv[1], sys.argv[2], sys.argv[3]
code = open(js).read().replace("</script>", "<\\/script>")
prof = open(__file__.replace("web_page.py", "web_gpuprof.js")).read()
page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>html,body{{margin:0;height:100%;background:#000;overflow:hidden}}canvas{{display:block;width:100%;height:calc(100% - 4em);margin:2em 0}}
#status{{position:fixed;left:12px;top:10px;color:#ccc;font:13px system-ui}}
#by{{position:fixed;right:12px;bottom:10px;color:#bbb;font:13px system-ui;text-decoration:none;opacity:.75}}
#by:hover{{opacity:1}}
#unsupported{{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:0 16px;color:#ddd;font:16px/1.5 system-ui}}</style></head>
<body><canvas id="canvas" tabindex="-1"></canvas><div id="status">loading...</div>
<a id="by" href="https://github.com/ricardobeat" target="_blank" rel="noopener">by @ricardobeat</a>
<script>{prof}</script>
<script>
var Module = {{
  canvas: document.getElementById('canvas'),
  // (the page's #fragment, split at spaces, as the program's arguments: #--scale 1 --off rc)
  arguments: decodeURIComponent(location.hash.slice(1)).split(' ').filter((a) => a),
  print: (t) => {{ console.log(t); (window.drieLog = window.drieLog || []).push(t); }},
  printErr: (t) => console.error(t),
  onRuntimeInitialized: () => document.getElementById('status').remove(),
}};
for (const k of ['error', 'warn']) {{
  const f = console[k].bind(console);
  console[k] = (...a) => {{ f(...a); (window.drieLog = window.drieLog || []).push(k + ': ' + a.join(' ')); }};
}}
// the right button turns the view: no context menu over the canvas
document.getElementById('canvas').addEventListener('contextmenu', (e) => e.preventDefault());
window.addEventListener('error', (e) => (window.drieLog = window.drieLog || []).push('error: ' + e.message + ' ' + (e.error && e.error.stack)));
</script>
<script type="text/x-drie" id="main">{code}</script>
<script>
// (run only where there is WebGPU; elsewhere say so instead of failing on a black canvas)
if (navigator.gpu) {{
  const s = document.createElement('script');
  s.text = document.getElementById('main').text;
  document.body.appendChild(s);
}} else {{
  document.getElementById('status').remove();
  document.getElementById('canvas').remove();
  const m = document.createElement('div');
  m.id = 'unsupported';
  m.innerHTML = 'This demo needs WebGPU, which this browser doesn\u2019t provide.<br>Try a recent Safari, Chrome or Edge (and check that Lockdown Mode is off).';
  document.body.appendChild(m);
}}
</script>
</body></html>
"""
open(out, "w").write(page)
