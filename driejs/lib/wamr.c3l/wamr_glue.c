// WAMR's wasm-c-api reduced to what js/wasm.js needs, with every value crossing as a double:
// i32 and f32 exactly, i64 rounded to the nearest double.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wasm_c_api.h"

typedef struct WgModule {
    wasm_module_t* module;
    wasm_importtype_vec_t imports;
    wasm_exporttype_vec_t exports;
} WgModule;

typedef struct WgInstance WgInstance;

typedef struct WgImport {
    WgInstance* instance;
    int index;
} WgImport;

struct WgInstance {
    WgModule* module;
    void* udata;
    WgImport* import_envs;
    wasm_extern_t** import_externs;
    wasm_instance_t* instance;
    wasm_extern_vec_t exports;
};

// Calls the JS function of import `index`: 0 on success, 1 when it threw (the exception stays
// pending on the JS side).
typedef int (*WgHostCall)(void* udata, int index, const double* args, int nargs, double* results, int nresults);

static wasm_engine_t* engine;
static wasm_store_t* store;
static WgHostCall host_call;

static void set_error(char* err, size_t cap, const char* msg) {
    if (cap) snprintf(err, cap, "%s", msg);
}

static void trap_error(wasm_trap_t* trap, char* err, size_t cap) {
    wasm_message_t message;
    wasm_trap_message(trap, &message);
    if (cap) snprintf(err, cap, "%.*s", (int)message.size, message.data);
    wasm_byte_vec_delete(&message);
    wasm_trap_delete(trap);
}

bool wg_init(WgHostCall call) {
    host_call = call;
    if (engine) return true;
    engine = wasm_engine_new();
    if (!engine) return false;
    store = wasm_store_new(engine);
    return store != NULL;
}

// JS ToInt32
static int32_t to_i32(double d) {
    if (!isfinite(d)) return 0;
    double t = fmod(trunc(d), 4294967296.0);
    if (t < 0) t += 4294967296.0;
    return (int32_t)(uint32_t)t;
}

static wasm_val_t to_val(wasm_valkind_t kind, double d) {
    wasm_val_t v;
    memset(&v, 0, sizeof v);
    v.kind = kind;
    switch (kind) {
        case WASM_I32: v.of.i32 = to_i32(d); break;
        case WASM_I64: v.of.i64 = isfinite(d) ? (int64_t)d : 0; break;
        case WASM_F32: v.of.f32 = (float)d; break;
        case WASM_F64: v.of.f64 = d; break;
        default: v.of.ref = NULL; break;
    }
    return v;
}

static double from_val(const wasm_val_t* v) {
    switch (v->kind) {
        case WASM_I32: return v->of.i32;
        case WASM_I64: return (double)v->of.i64;
        case WASM_F32: return v->of.f32;
        case WASM_F64: return v->of.f64;
        default: return 0;
    }
}

WgModule* wg_compile(const uint8_t* bytes, size_t len, char* err, size_t cap) {
    wasm_byte_vec_t binary;
    wasm_byte_vec_new(&binary, len, (const wasm_byte_t*)bytes);
    wasm_module_t* module = wasm_module_new(store, &binary);
    wasm_byte_vec_delete(&binary);
    if (!module) {
        set_error(err, cap, "invalid or unsupported module");
        return NULL;
    }
    WgModule* m = calloc(1, sizeof *m);
    m->module = module;
    wasm_module_imports(module, &m->imports);
    wasm_module_exports(module, &m->exports);
    return m;
}

int wg_import_count(WgModule* m) { return (int)m->imports.size; }
int wg_export_count(WgModule* m) { return (int)m->exports.size; }

static int kind_of(const wasm_externtype_t* type) {
    return (int)wasm_externtype_kind(type);  // WASM_EXTERN_FUNC, _GLOBAL, _TABLE, _MEMORY
}

// Writes "module\nname" of import `i` and returns its kind.
int wg_import_info(WgModule* m, int i, char* out, size_t cap) {
    const wasm_importtype_t* t = m->imports.data[i];
    const wasm_name_t* mod = wasm_importtype_module(t);
    const wasm_name_t* name = wasm_importtype_name(t);
    snprintf(out, cap, "%.*s\n%.*s", (int)mod->size, mod->data, (int)name->size, name->data);
    return kind_of(wasm_importtype_type(t));
}

int wg_export_info(WgModule* m, int i, char* out, size_t cap) {
    const wasm_exporttype_t* t = m->exports.data[i];
    const wasm_name_t* name = wasm_exporttype_name(t);
    snprintf(out, cap, "%.*s", (int)name->size, name->data);
    return kind_of(wasm_exporttype_type(t));
}

static wasm_trap_t* on_import(void* env, const wasm_val_vec_t* args, wasm_val_vec_t* results) {
    WgImport* imp = env;
    double in[32], out[8];
    int nargs = (int)args->size, nresults = (int)results->size;
    if (nargs > 32 || nresults > 8) {
        wasm_message_t msg;
        wasm_name_new_from_string_nt(&msg, "too many arguments or results for a JS import");
        wasm_trap_t* trap = wasm_trap_new(store, &msg);
        wasm_byte_vec_delete(&msg);
        return trap;
    }
    for (int i = 0; i < nargs; i++) in[i] = from_val(&args->data[i]);
    for (int i = 0; i < nresults; i++) out[i] = 0;
    if (host_call(imp->instance->udata, imp->index, in, nargs, out, nresults)) {
        wasm_message_t msg;
        wasm_name_new_from_string_nt(&msg, "JS exception");
        wasm_trap_t* trap = wasm_trap_new(store, &msg);
        wasm_byte_vec_delete(&msg);
        return trap;
    }
    const wasm_functype_t* type = wasm_externtype_as_functype_const(wasm_importtype_type(imp->instance->module->imports.data[imp->index]));
    const wasm_valtype_vec_t* kinds = wasm_functype_results(type);
    for (int i = 0; i < nresults; i++) results->data[i] = to_val(wasm_valtype_kind(kinds->data[i]), out[i]);
    return NULL;
}

// Instantiates `m`. Function imports call host_call with `udata`; other import kinds are refused.
// Returns null with `err` set on failure, which includes a JS import throwing in the start
// function.
WgInstance* wg_instantiate(WgModule* m, void* udata, char* err, size_t cap) {
    WgInstance* inst = calloc(1, sizeof *inst);
    inst->module = m;
    inst->udata = udata;
    size_t n = m->imports.size;
    inst->import_envs = calloc(n ? n : 1, sizeof(WgImport));
    inst->import_externs = calloc(n ? n : 1, sizeof(wasm_extern_t*));
    for (size_t i = 0; i < n; i++) {
        const wasm_externtype_t* type = wasm_importtype_type(m->imports.data[i]);
        if (wasm_externtype_kind(type) != WASM_EXTERN_FUNC) {
            set_error(err, cap, "only function imports are supported");
            return NULL;
        }
        inst->import_envs[i] = (WgImport){ inst, (int)i };
        wasm_func_t* f = wasm_func_new_with_env(store, wasm_externtype_as_functype_const(type), on_import, &inst->import_envs[i], NULL);
        inst->import_externs[i] = wasm_func_as_extern(f);
    }
    wasm_extern_vec_t imports = { n, inst->import_externs, n, sizeof(wasm_extern_t*), NULL };
    wasm_trap_t* trap = NULL;
    inst->instance = wasm_instance_new(store, m->module, &imports, &trap);
    if (!inst->instance) {
        if (trap) trap_error(trap, err, cap);
        else set_error(err, cap, "instantiation failed");
        return NULL;
    }
    wasm_instance_exports(inst->instance, &inst->exports);
    return inst;
}

static wasm_extern_t* export_of(WgInstance* inst, int i) {
    return i >= 0 && (size_t)i < inst->exports.size ? inst->exports.data[i] : NULL;
}

// The export's function, also the handle wg_call takes, or null.
wasm_func_t* wg_export_func(WgInstance* inst, int i) {
    wasm_extern_t* e = export_of(inst, i);
    return e ? wasm_extern_as_func(e) : NULL;
}

int wg_func_arity(wasm_func_t* f, int* nresults) {
    *nresults = (int)wasm_func_result_arity(f);
    return (int)wasm_func_param_arity(f);
}

// 0 on success, 1 on a trap (`err` set), which includes a JS import throwing.
int wg_call(wasm_func_t* f, const double* args, int nargs, double* results, int nresults, char* err, size_t cap) {
    wasm_functype_t* type = wasm_func_type(f);
    const wasm_valtype_vec_t* params = wasm_functype_params(type);
    const wasm_valtype_vec_t* kinds = wasm_functype_results(type);
    wasm_val_t in[32], out[8];
    size_t np = params->size, nr = kinds->size;
    if (np > 32 || nr > 8) {
        wasm_functype_delete(type);
        set_error(err, cap, "too many parameters or results");
        return 1;
    }
    for (size_t i = 0; i < np; i++) in[i] = to_val(wasm_valtype_kind(params->data[i]), (int)i < nargs ? args[i] : NAN);
    for (size_t i = 0; i < nr; i++) out[i] = to_val(wasm_valtype_kind(kinds->data[i]), 0);
    wasm_functype_delete(type);
    wasm_val_vec_t argv = { np, in, np, sizeof(wasm_val_t), NULL };
    wasm_val_vec_t resv = { nr, out, nr, sizeof(wasm_val_t), NULL };
    wasm_trap_t* trap = wasm_func_call(f, &argv, &resv);
    if (trap) {
        trap_error(trap, err, cap);
        return 1;
    }
    for (size_t i = 0; i < nr && (int)i < nresults; i++) results[i] = from_val(&out[i]);
    return 0;
}

uint8_t* wg_memory(WgInstance* inst, int i, size_t* size) {
    wasm_extern_t* e = export_of(inst, i);
    wasm_memory_t* mem = e ? wasm_extern_as_memory(e) : NULL;
    if (!mem) {
        *size = 0;
        return NULL;
    }
    *size = wasm_memory_data_size(mem);
    return (uint8_t*)wasm_memory_data(mem);
}

// The size in pages before growing, or -1.
int wg_memory_grow(WgInstance* inst, int i, uint32_t delta) {
    wasm_extern_t* e = export_of(inst, i);
    wasm_memory_t* mem = e ? wasm_extern_as_memory(e) : NULL;
    if (!mem) return -1;
    int pages = (int)wasm_memory_size(mem);
    return wasm_memory_grow(mem, delta) ? pages : -1;
}

int wg_table_size(WgInstance* inst, int i) {
    wasm_extern_t* e = export_of(inst, i);
    wasm_table_t* t = e ? wasm_extern_as_table(e) : NULL;
    return t ? (int)wasm_table_size(t) : -1;
}

// The function at `index` of an exported funcref table, or null.
wasm_func_t* wg_table_get(WgInstance* inst, int i, int index) {
    wasm_extern_t* e = export_of(inst, i);
    wasm_table_t* t = e ? wasm_extern_as_table(e) : NULL;
    if (!t) return NULL;
    wasm_ref_t* ref = wasm_table_get(t, (wasm_table_size_t)index);
    return ref ? wasm_ref_as_func(ref) : NULL;
}
