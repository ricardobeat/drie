// The natives behind js/wasm.js's WebAssembly: modules, instances, calls both ways and the
// exported memory as an ArrayBuffer, on WAMR's fast interpreter.
module main;

import boomkat;
import boomkat::builtins, boomkat::hobject, boomkat::heap;
import std::collections::list;
import wamr;

struct WasmInstance
{
    wamr::Instance wg;
    // the JS functions of its imports, by import index
    List{JsValue} imports;
    // memory_export's current ArrayBuffer over the linear memory, which wasmMemoryBuffer
    // replaces (detaching the old one) once the memory has moved or grown
    JsValue memory_buffer;
    bool has_buffer;
    int memory_export;
    char* memory_data;
    usz memory_size;
}

List{wamr::Module} wasm_modules;
List{WasmInstance*} wasm_instances;
// the imports wasmImport has collected for the next wasmInstantiate
List{JsValue} wasm_pending_imports;
// makes a WebAssembly.RuntimeError or CompileError: (kind, message) => error
JsValue wasm_error_factory;
// the innermost host call running wasm code, which a JS import is called through
JsCtx wasm_ctx;
bool wasm_import_threw;
char[512] wasm_err;

fn void wasm_throw(JsCtx ctx, String kind, ZString msg)
{
    JsArg? e = ctx.call(ctx.value(wasm_error_factory), { ctx.string(kind), ctx.string(msg.str_view()) });
    if (catch e) return;
    ctx.throw_value(e);
}

fn CInt wasm_on_import(void* udata, CInt index, double* args, CInt nargs, double* results, CInt nresults)
{
    WasmInstance* inst = wasm_instances[(sz)(uptr)udata];
    JsCtx ctx = wasm_ctx;
    JsArg[32] argv;
    for (int i = 0; i < nargs; i++) argv[i] = ctx.number(args[i]);
    JsArg? r = ctx.call(ctx.value(inst.imports[index]), argv[:nargs]);
    if (catch r)
    {
        wasm_import_threw = true;
        return 1;
    }
    if (nresults > 0)
    {
        double? d = r.as_number();
        if (catch d)
        {
            bool? b = r.as_bool();
            results[0] = (b ?? false) ? 1 : 0;
        }
        else
        {
            results[0] = d;
        }
    }
    return 0;
}

fn void wasm_noop_release(void* data, void* opaque)
{
}

// wasmInit(errorFactory)
fn void h_wasm_init(JsCtx ctx, void* udata)
{
    if (!wamr::init(&wasm_on_import)) { ctx.throw_error(ERROR, "wasmInit: WAMR failed to start"); return; }
    JsValue? f = ctx.persist(ctx.arg(0));
    if (catch f) { ctx.throw_error(ERROR, "wasmInit: cannot keep the error factory"); return; }
    wasm_error_factory = f;
}

// wasmCompile(bytes): a module handle
fn void h_wasm_compile(JsCtx ctx, void* udata)
{
    View v;
    if (!view_arg(ctx, 0, &v) || v.cls != ObjClass.UINT8ARRAY) { ctx.throw_error(TYPE, "wasmCompile: expected a Uint8Array"); return; }
    wamr::Module m = wamr::compile(v.data, v.len, &wasm_err, wasm_err.len);
    if (m == null) { wasm_throw(ctx, "CompileError", (ZString)&wasm_err); return; }
    wasm_modules.push(m);
    ctx.ret_number((double)(wasm_modules.len() - 1));
}

fn wamr::Module wasm_module_arg(JsCtx ctx)
{
    int i = (int)num_arg(ctx, 0, -1);
    return i >= 0 && i < wasm_modules.len() ? wasm_modules[i] : null;
}

fn WasmInstance* wasm_instance_arg(JsCtx ctx)
{
    int i = (int)num_arg(ctx, 0, -1);
    return i >= 0 && i < wasm_instances.len() ? wasm_instances[i] : null;
}

fn String kind_name(wamr::ExternKind k)
{
    switch (k)
    {
        case FUNC: return "function";
        case GLOBAL: return "global";
        case TABLE: return "table";
        case MEMORY: return "memory";
        default: return "unknown";
    }
}

// wasmImports(module): "kind\tmodule\tname" lines
fn void h_wasm_imports(JsCtx ctx, void* udata)
{
    wamr::Module m = wasm_module_arg(ctx);
    if (m == null) { ctx.throw_error(TYPE, "wasmImports: invalid module"); return; }
    DString s;
    s.tinit();
    char[512] buf;
    for (int i = 0; i < wamr::import_count(m); i++)
    {
        wamr::ExternKind k = wamr::import_info(m, i, &buf, buf.len);
        String names = ((ZString)&buf).str_view();
        sz nl = names.index_of_char('\n') ?? (sz)names.len;
        s.appendf("%s\t%s\t%s\n", kind_name(k), names[:nl], nl < names.len ? names[nl + 1..] : "");
    }
    ctx.ret_string(s.str_view());
}

// wasmExports(module): "kind\tname" lines, in export index order
fn void h_wasm_exports(JsCtx ctx, void* udata)
{
    wamr::Module m = wasm_module_arg(ctx);
    if (m == null) { ctx.throw_error(TYPE, "wasmExports: invalid module"); return; }
    DString s;
    s.tinit();
    char[512] buf;
    for (int i = 0; i < wamr::export_count(m); i++)
    {
        wamr::ExternKind k = wamr::export_info(m, i, &buf, buf.len);
        s.appendf("%s\t%s\n", kind_name(k), ((ZString)&buf).str_view());
    }
    ctx.ret_string(s.str_view());
}

// wasmImport(fn): the next import of the coming wasmInstantiate
fn void h_wasm_import(JsCtx ctx, void* udata)
{
    JsValue? f = ctx.persist(ctx.arg(0));
    if (catch f) { ctx.throw_error(ERROR, "wasmImport: cannot keep the function"); return; }
    wasm_pending_imports.push(f);
}

// wasmInstantiate(module): an instance handle, taking the imports wasmImport collected
fn void h_wasm_instantiate(JsCtx ctx, void* udata)
{
    wamr::Module m = wasm_module_arg(ctx);
    if (m == null) { ctx.throw_error(TYPE, "wasmInstantiate: invalid module"); return; }
    WasmInstance* inst = mem::new(WasmInstance);
    inst.imports.init(mem);
    foreach (f : wasm_pending_imports) inst.imports.push(f);
    wasm_pending_imports.clear();
    inst.memory_export = -1;
    sz id = wasm_instances.len();
    wasm_instances.push(inst);
    JsCtx outer = wasm_ctx;
    wasm_ctx = ctx;
    wasm_import_threw = false;
    inst.wg = wamr::instantiate(m, (void*)(uptr)id, &wasm_err, wasm_err.len);
    wasm_ctx = outer;
    if (inst.wg == null)
    {
        if (!wasm_import_threw) wasm_throw(ctx, "RuntimeError", (ZString)&wasm_err);
        return;
    }
    ctx.ret_number((double)id);
}

// wasmExportFunc(instance, export): a function handle, or -1
fn void h_wasm_export_func(JsCtx ctx, void* udata)
{
    WasmInstance* inst = wasm_instance_arg(ctx);
    if (inst == null) { ctx.throw_error(TYPE, "wasmExportFunc: invalid instance"); return; }
    wamr::Func f = wamr::export_func(inst.wg, (CInt)num_arg(ctx, 1, -1));
    ctx.ret_number(f == null ? -1 : (double)(uptr)f);
}

// wasmArity(func): its parameter count
fn void h_wasm_arity(JsCtx ctx, void* udata)
{
    CInt nresults;
    ctx.ret_number(wamr::func_arity((wamr::Func)(uptr)num_arg(ctx, 0, 0), &nresults));
}

// wasmCall(func, ...args): its first result, or undefined
fn void h_wasm_call(JsCtx ctx, void* udata)
{
    double h = num_arg(ctx, 0, 0);
    if (h <= 0) { ctx.throw_error(TYPE, "wasmCall: invalid function"); return; }
    wamr::Func f = (wamr::Func)(uptr)h;
    uint argc = ctx.argc();
    double[32] args;
    int nargs = 0;
    for (uint i = 1; i < argc && nargs < 32; i++)
    {
        JsArg a = ctx.arg(i);
        double? d = a.as_number();
        if (catch d)
        {
            bool? b = a.as_bool();
            if (catch b)
            {
                args[nargs++] = 0.0 / 0.0;
            }
            else
            {
                args[nargs++] = b ? 1 : 0;
            }
        }
        else
        {
            args[nargs++] = d;
        }
    }
    CInt nresults;
    wamr::func_arity(f, &nresults);
    double[8] results;
    JsCtx outer = wasm_ctx;
    wasm_ctx = ctx;
    wasm_import_threw = false;
    CInt rc = wamr::call(f, &args, nargs, &results, nresults, &wasm_err, wasm_err.len);
    wasm_ctx = outer;
    if (rc != 0)
    {
        // a JS import's exception is already pending on this call
        if (!wasm_import_threw) wasm_throw(ctx, "RuntimeError", (ZString)&wasm_err);
        wasm_import_threw = false;
        return;
    }
    if (nresults > 0) ctx.ret_number(results[0]);
}

// wasmMemoryBuffer(instance, export): an ArrayBuffer over the exported memory, the same one
// until the memory moves or grows
fn void h_wasm_memory_buffer(JsCtx ctx, void* udata)
{
    WasmInstance* inst = wasm_instance_arg(ctx);
    if (inst == null) { ctx.throw_error(TYPE, "wasmMemoryBuffer: invalid instance"); return; }
    int e = (int)num_arg(ctx, 1, -1);
    usz size;
    char* data = wamr::memory(inst.wg, e, &size);
    if (data == null) { ctx.throw_error(TYPE, "wasmMemoryBuffer: not a memory export"); return; }
    if (inst.has_buffer && inst.memory_export == e && inst.memory_data == data && inst.memory_size == size)
    {
        ctx.ret(ctx.value(inst.memory_buffer));
        return;
    }
    wasm_detach_buffer(ctx, inst);
    Heap* hp = ctx.ctx.heap;
    HObject*? raw = hp.alloc_object(ObjClass.ARRAYBUFFER);
    if (catch raw) { ctx.throw_error(ERROR, "wasmMemoryBuffer: out of memory"); return; }
    HObject* buf = raw;
    hp.set_prototype(buf, hp.arraybuffer_proto);
    (*buf.extra_ptr()).ab.data = null;
    (*buf.extra_ptr()).ab.byte_length = 0;
    (*buf.extra_ptr()).ab.views_head = null;
    if (!hp.make_external_buffer(buf, data, (uint)size, &wasm_noop_release, null))
    {
        ctx.throw_error(ERROR, "wasmMemoryBuffer: out of memory");
        return;
    }
    JsArg a;
    a.tv.set_object(buf);
    JsValue? kept = ctx.persist(a);
    if (catch kept) { ctx.throw_error(ERROR, "wasmMemoryBuffer: cannot keep the buffer"); return; }
    inst.memory_buffer = kept;
    inst.has_buffer = true;
    inst.memory_export = e;
    inst.memory_data = data;
    inst.memory_size = size;
    ctx.ret(a);
}

fn void wasm_detach_buffer(JsCtx ctx, WasmInstance* inst)
{
    if (!inst.has_buffer) return;
    JsArg old = ctx.value(inst.memory_buffer);
    if (old.tv.is_object()) builtins::arraybuffer_detach(ctx.ctx.heap, (HObject*)old.tv.get_heapptr());
    js.release(inst.memory_buffer);
    inst.has_buffer = false;
}

// wasmMemoryGrow(instance, export, pages): the size in pages before, or -1
fn void h_wasm_memory_grow(JsCtx ctx, void* udata)
{
    WasmInstance* inst = wasm_instance_arg(ctx);
    if (inst == null) { ctx.throw_error(TYPE, "wasmMemoryGrow: invalid instance"); return; }
    int e = (int)num_arg(ctx, 1, -1);
    CInt old = wamr::memory_grow(inst.wg, e, (uint)num_arg(ctx, 2, 0));
    // growing detaches the old buffer, even by 0 pages
    if (old >= 0 && inst.memory_export == e) wasm_detach_buffer(ctx, inst);
    ctx.ret_number(old);
}

// wasmTableSize(instance, export)
fn void h_wasm_table_size(JsCtx ctx, void* udata)
{
    WasmInstance* inst = wasm_instance_arg(ctx);
    if (inst == null) { ctx.throw_error(TYPE, "wasmTableSize: invalid instance"); return; }
    ctx.ret_number(wamr::table_size(inst.wg, (CInt)num_arg(ctx, 1, -1)));
}

// wasmTableGet(instance, export, index): a function handle, or -1 for null
fn void h_wasm_table_get(JsCtx ctx, void* udata)
{
    WasmInstance* inst = wasm_instance_arg(ctx);
    if (inst == null) { ctx.throw_error(TYPE, "wasmTableGet: invalid instance"); return; }
    wamr::Func f = wamr::table_get(inst.wg, (CInt)num_arg(ctx, 1, -1), (CInt)num_arg(ctx, 2, 0));
    ctx.ret_number(f == null ? -1 : (double)(uptr)f);
}

fn void register_wasm(JsValue natives)
{
    wasm_modules.init(mem);
    wasm_instances.init(mem);
    wasm_pending_imports.init(mem);
    reg(natives, "wasmInit", &h_wasm_init, 1);
    reg(natives, "wasmCompile", &h_wasm_compile, 1);
    reg(natives, "wasmImports", &h_wasm_imports, 1);
    reg(natives, "wasmExports", &h_wasm_exports, 1);
    reg(natives, "wasmImport", &h_wasm_import, 1);
    reg(natives, "wasmInstantiate", &h_wasm_instantiate, 1);
    reg(natives, "wasmExportFunc", &h_wasm_export_func, 2);
    reg(natives, "wasmArity", &h_wasm_arity, 1);
    reg(natives, "wasmCall", &h_wasm_call, 1);
    reg(natives, "wasmMemoryBuffer", &h_wasm_memory_buffer, 2);
    reg(natives, "wasmMemoryGrow", &h_wasm_memory_grow, 3);
    reg(natives, "wasmTableSize", &h_wasm_table_size, 2);
    reg(natives, "wasmTableGet", &h_wasm_table_get, 3);
}
