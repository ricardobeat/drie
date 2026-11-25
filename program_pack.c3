// The program pack: shaders compiled ahead of time, so that a run makes them without compiling
// their source. The shaders drie makes from source (the node programs spliced into the material
// shader, the generated programs, the engine's own) are looked up in it by a hash of their
// source text; one found is handed to sokol as bytecode, one not found is compiled from its
// source as before.
//
//   drie::load_program_pack_file("build/cache/programs.metal-macos");  // before the first frame
//
// A pack is made at build time from the sources of a run (Metal: xcrun metal, see house's
// tools/programs.py):
//
//   DRIE_DUMP_PROGRAMS=<dir>   every source made into a shader is saved as <dir>/<hash>.metal
//   drie::dump_programs(dir, compile: false)   the same, and the background compiles
//                              (sokol_warm.h) compile nothing: a loaded pipeline list then
//                              goes through every program it names in a moment
//
// The file, all numbers little-endian:
//
//   "drie-programs 1\n"                    16 bytes
//   target                                 16 bytes, zero padded: "metal-macos"
//   count, 0                               2 x u32
//   count x { hash u64, offset u32, size u32 }   by hash; offset from the file's start
//   the bytecode                           a metallib each
//
// hash: FNV-1a, 64 bits, over the source text (without its terminating zero).
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
    // how many shader stages it gave bytecode to, and how many it didn't have
    int found;
    int missed;
}

ProgramPack program_pack @local;
// DRIE_DUMP_PROGRAMS, once read ("": no dump)
String dump_dir @local;
bool dump_read @local;
// dump_programs without compiling
bool dump_only @local;

// Whether sources are being saved and nothing compiled (dump_programs).
fn bool dumping_only() @private => dump_only;

extern fn CInt sg_warm_library_loads(void* ptr, usz size) @cname("sg_warm_library_loads");
extern fn void sg_warm_dry(CInt on) @cname("sg_warm_dry");

// Saves every source made into a shader from here on as <dir>/<hash>.<ext> (as
// DRIE_DUMP_PROGRAMS does); without `compile`, the compiles in the background compile nothing
// (and are done at once).
fn void dump_programs(String dir, bool compile = true) {
    dump_read = true;
    dump_dir = dir.copy(mem);
    (void)path::mkdir(dump_dir, true);
    dump_only = !compile;
    sg_warm_dry(compile ? 0 : 1);
}

// FNV-1a, 64 bits.
fn ulong program_hash(String s) {
    ulong h = 0xcbf29ce484222325;
    foreach (c : s) h = (h ^ c) * 0x100000001b3;
    return h;
}

fn uint read_u32(char[] d, usz at) @local => (uint)d[at] | (uint)d[at + 1] << 8 | (uint)d[at + 2] << 16 | (uint)d[at + 3] << 24;

fn ulong read_u64(char[] d, usz at) @local => (ulong)read_u32(d, at) | (ulong)read_u32(d, at + 4) << 32;

// The target the running backend's bytecode is built for ("": none drie packs).
fn String pack_target() @local {
    switch (sg::query_backend()) {
        case METAL_MACOS: return "metal-macos";
        default: return "";
    }
}

// Takes `data` (in `mem`, kept for the run) as the program pack: how many programs it has for
// this backend (0, and nothing kept, if it is for another one, or not a pack). Call it after
// the GPU is set up (Renderer.init_headless or the first frame) or before: the backend is
// checked when the first program is looked up.
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
        usz at = head + (usz)i * 16;
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

// load_program_pack, from a file: false if it can't be read.
fn bool load_program_pack_file(String path) {
    char[]? data = file::load(mem, path);
    if (catch data) return false;
    return load_program_pack(data) > 0;
}

// { stages given bytecode, stages compiled from source } since the pack was loaded.
fn int[2] program_pack_counts() => { program_pack.found, program_pack.missed };

// whether the pack was checked against the backend and the device (see use_pack)
bool checked @local;

// Whether the pack's bytecode is for this backend and loads on this device; it is let go of
// otherwise.
fn bool use_pack() @local {
    ProgramPack* p = &program_pack;
    if (!p.data.len) return false;
    if (checked) return true;
    checked = true;
    String want = pack_target();
    bool ok = want.len && want == p.target;
    if (ok) {
        // (one of its libraries, made: a pack built for a newer OS or another GPU family doesn't load)
        bool tried;
        p.entries.@each(; ulong h, PackEntry e) {
            if (!tried && !sg_warm_library_loads(&p.data[e.offset], e.size)) ok = false;
            tried = true;
        };
        if (!ok) io::eprintfn("drie: the program pack (%s) doesn't load on this device: compiling from source", p.target);
    } else {
        io::eprintfn("drie: the program pack is for %s, not this backend (%s): compiling from source", p.target,
            want.len ? want : "none");
    }
    if (!ok) {
        free(p.data);
        p.data = { };
        p.entries.clear();
    }
    return ok;
}

// The file extension of the running backend's sources ("": not dumped).
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
    ulong h = program_hash(src);
    if (dump_dir.len) {
        @pool() {
            String ext = source_ext();
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

// `d` with the bytecode the program pack has for its stages, in place of their source (and
// each source saved under DRIE_DUMP_PROGRAMS). Call it on a description just before it is
// handed to sokol (sg::make_shader, sg_warm_begin): the bytecode is the pack's.
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

// sg::make_shader, with the program pack's bytecode (with_bytecode).
fn SgShader make_program(SgShaderDesc desc) {
    with_bytecode(&desc);
    return sg::make_shader(&desc);
}
