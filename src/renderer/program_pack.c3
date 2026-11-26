// Ahead-of-time shader bytecode packs and source dumping for runtime-compiled shaders.
// Substitutes precompiled bytecode (currently Metal metallibs) into SgShaderDesc stages by
// matching the 64-bit FNV-1a hash of each stage's source text; unmatched stages compile from
// source. All shaders created via make_program() pass through with_bytecode() and register for
// the Metal binary pipeline archive (Renderer.use_pipeline_archive). Set DRIE_DUMP_PROGRAMS=<dir>
// or call dump_programs() to write stage sources as <dir>/<hash>.<ext> for offline packing.
module drie;
import std::io, std::io::file, std::io::path, std::os::env, std::collections::map, sokol::sg;

const String PROGRAM_PACK_HEADER @local = "drie-programs 1\n";

struct PackEntry @local {
    uint offset;
    uint size;
}

struct ProgramPack @local {
    char[] data;
    String target;
    HashMap{ulong, PackEntry} entries;
    int found; // Stages satisfied from pack bytecode.
    int missed; // Stages absent from the pack and compiled from source.
}

ProgramPack program_pack @local;
String dump_dir @local; // Target directory from DRIE_DUMP_PROGRAMS or dump_programs().
bool dump_read @local;
bool dump_only @local;  // True when dump_programs(compile: false) skips GPU compilation.

// True when shader sources are being dumped without compiling them on the GPU.
fn bool dumping_only() @private => dump_only;

extern fn CInt sg_warm_library_loads(void* ptr, usz size) @cname("sg_warm_library_loads");
extern fn void sg_warm_dry(CInt on) @cname("sg_warm_dry");

// Writes every shader stage source created from this point on to `<dir>/<hash>.<ext>`.
// With `compile: false`, background compilation jobs (sokol_warm.h) skip GPU compilation so a
// loaded pipeline cache can dump all referenced shader sources quickly.
fn void dump_programs(String dir, bool compile = true) {
    dump_read = true;
    dump_dir = dir.copy(mem);
    (void)path::mkdir(dump_dir, true);
    dump_only = !compile;
    sg_warm_dry(compile ? 0 : 1);
}

// 64-bit FNV-1a hash over `s` (excluding any null terminator) that keys program pack entries.
fn ulong program_hash(String s) {
    ulong h = 0xcbf29ce484222325;
    foreach (c : s) h = (h ^ c) * 0x100000001b3;
    return h;
}

fn uint read_u32(
    char[] d,
    usz at
) @local => (uint)d[at] | (uint)d[at + 1] << 8 | (uint)d[at + 2] << 16 | (uint)d[at + 3] << 24;

fn ulong read_u64(char[] d, usz at) @local => (ulong)read_u32(d, at) | (ulong)read_u32(d, at + 4) << 32;

// Target tag for the running backend's precompiled bytecode, or "" if unsupported.
fn String pack_target() @local {
    switch (sg::query_backend()) {
        case METAL_MACOS: return "metal-macos";
        default: return "";
    }
}

// Loads a binary program pack from `data` (owned in `mem` for the lifetime of the run) and
// returns the number of indexed entries (0 on invalid input, freeing `data`). May be called
// before or after GPU setup; target compatibility is verified on first shader lookup.
//
// Binary format (little-endian):
//   "drie-programs 1\n"                  16 bytes
//   target                               16 bytes, zero-padded (e.g. "metal-macos")
//   count, reserved                      2 x u32
//   count x { hash u64, offset u32, size u32 }  file-relative offsets
//   bytecode blobs                       one metallib per entry
fn int load_program_pack(char[] data) {
    usz head = PROGRAM_PACK_HEADER.len + 16 + 8;
    if (data.len < head || (String)data[:PROGRAM_PACK_HEADER.len] != PROGRAM_PACK_HEADER) {
        io::eprintn("drie: not a program pack");
        free(data);
        return 0;
    }
    char[] t = data[PROGRAM_PACK_HEADER.len:16];
    usz tlen;
    while (tlen < 16 && t[tlen]) tlen++;
    uint count = read_u32(data, PROGRAM_PACK_HEADER.len + 16);
    if (data.len < head + (usz)count * 16) {
        io::eprintn("drie: a program pack cut short");
        free(data);
        return 0;
    }
    ProgramPack* p = &program_pack;
    if (p.data.len) {
        free(p.data);
        p.entries.clear();
    }
    if (!p.entries.is_initialized()) p.entries.init(mem);
    for (uint i = 0; i < count; i++) {
        usz at      = head + (usz)i * 16;
        PackEntry e = { read_u32(data, at + 8), read_u32(data, at + 12) };
        if ((usz)e.offset + e.size > data.len) continue;
        p.entries[read_u64(data, at)] = e;
    }
    p.data = data;
    p.target = (String)t[:tlen];
    p.found = 0;
    p.missed = 0;
    checked = false;
    return (int)p.entries.len();
}

// Loads a program pack from `path`. Returns false if the file cannot be read or is empty/invalid.
fn bool load_program_pack_file(String path) {
    char[]? data = file::load(mem, path);
    if (catch data) return false;
    return load_program_pack(data) > 0;
}

// Returns `{ stages_from_bytecode, stages_from_source }` since the active pack was loaded.
fn int[2] program_pack_counts() => { program_pack.found, program_pack.missed };

// Set once use_pack() has validated the pack against the active backend and GPU device.
bool checked @local;

// Validates the loaded pack against the running backend and device on first use, freeing it if
// incompatible.
fn bool use_pack() @local {
    ProgramPack* p = &program_pack;
    if (!p.data.len) return false;
    if (checked) return true;
    checked = true;
    String want = pack_target();
    bool ok     = want.len && want == p.target;
    if (ok) {
        // Probe one library on the active device; bytecode built for a newer OS or GPU fails here.
        bool tried;
        p.entries.@each(; ulong h, PackEntry e) {
            if (!tried && !sg_warm_library_loads(&p.data[e.offset], e.size)) ok = false;
            tried = true;
        };
        if (!ok) io::eprintfn(
            "drie: the program pack (%s) doesn't load on this device: compiling from source",
            p.target
        );
    } else {
        io::eprintfn(
            "drie: the program pack is for %s, not this backend (%s): compiling from source",
            p.target,
            want.len ? want : "none"
        );
    }
    if (!ok) {
        free(p.data);
        p.data = { };
        p.entries.clear();
    }
    return ok;
}

// File extension for dumped shader sources on the active backend, or "" when unsupported.
fn String source_ext() @local {
    switch (sg::query_backend()) {
        case METAL_MACOS:
        case METAL_IOS:
        case METAL_SIMULATOR: return "metal";
        default: return "";
    }
}

fn void stage_bytecode(SgShaderFunction* f, bool pack) @local {
    if (!f.source || f.bytecode.ptr) return;
    String src = ((ZString)f.source).str_view();
    ulong h    = program_hash(src);
    if (dump_dir.len) {
        @pool() {
            String ext  = source_ext();
            String name = string::tformat("%s/%016x.%s", dump_dir, h, ext);
            if (ext.len && !file::exists(name)) {
                if (catch file::save(name, src)) io::eprintfn("drie: can't write %s", name);
            }
        };
    }
    if (!pack) return;
    PackEntry? e = program_pack.entries[h];
    if (catch e) {
        program_pack.missed++;
        return;
    }
    program_pack.found++;
    f.bytecode = { &program_pack.data[e.offset], e.size };
    f.source = null;
}

// Replaces stage sources in `d` with matching bytecode pointers from the active program pack, and
// dumps sources when DRIE_DUMP_PROGRAMS or dump_programs() is active. Call right before handing
// `d` to sg::make_shader or sg_warm_begin; bytecode slices point into the loaded pack buffer.
fn void with_bytecode(SgShaderDesc* d) {
    if (!dump_read) {
        dump_read = true;
        if (try dir = env::tget_var("DRIE_DUMP_PROGRAMS")) {
            dump_dir = dir.copy(mem);
            (void)path::mkdir(dump_dir, true);
        }
    }
    bool pack = use_pack();
    if (!pack && !dump_dir.len) return;
    stage_bytecode(&d.vertex_func, pack);
    stage_bytecode(&d.fragment_func, pack);
    stage_bytecode(&d.compute_func, pack);
}

extern fn void sg_warm_archivable(uint shader_id) @cname("sg_warm_archivable");

// Creates a sokol shader with program-pack bytecode substitution (with_bytecode) and marks it
// eligible for the Metal binary pipeline archive (Renderer.use_pipeline_archive).
fn SgShader make_program(SgShaderDesc desc) {
    with_bytecode(&desc);
    SgShader s = sg::make_shader(&desc);
    sg_warm_archivable(s.id);
    return s;
}