// WAMR (WebAssembly Micro Runtime, fast interpreter) through wamr_glue.c.
module wamr;

alias Module = void*;
alias Instance = void*;
alias Func = void*;

// wasm_externkind_t
constdef ExternKind : int
{
    FUNC = 0,
    GLOBAL = 1,
    TABLE = 2,
    MEMORY = 3,
}

// Calls JS function `index` of the instance made with `udata`: 0, or 1 when it threw.
alias HostCall = fn CInt(void* udata, CInt index, double* args, CInt nargs, double* results, CInt nresults);

extern fn bool init(HostCall call) @cname("wg_init");
extern fn Module compile(char* bytes, usz len, char* err, usz cap) @cname("wg_compile");
extern fn CInt import_count(Module m) @cname("wg_import_count");
extern fn CInt export_count(Module m) @cname("wg_export_count");
extern fn ExternKind import_info(Module m, CInt i, char* out, usz cap) @cname("wg_import_info");
extern fn ExternKind export_info(Module m, CInt i, char* out, usz cap) @cname("wg_export_info");
extern fn Instance instantiate(Module m, void* udata, char* err, usz cap) @cname("wg_instantiate");
extern fn Func export_func(Instance inst, CInt i) @cname("wg_export_func");
extern fn CInt func_arity(Func f, CInt* nresults) @cname("wg_func_arity");
extern fn CInt call(Func f, double* args, CInt nargs, double* results, CInt nresults, char* err, usz cap) @cname("wg_call");
extern fn char* memory(Instance inst, CInt i, usz* size) @cname("wg_memory");
extern fn CInt memory_grow(Instance inst, CInt i, uint delta) @cname("wg_memory_grow");
extern fn CInt table_size(Instance inst, CInt i) @cname("wg_table_size");
extern fn Func table_get(Instance inst, CInt i, CInt index) @cname("wg_table_get");
