module three;
import sokol, std::io;

// CPU time per stage of a frame, for finding where it goes (`build/scene --bench --profile`).
// Off unless `profiling` is set: then prof_start/prof_stop cost a clock read each.
//
//     ulong t = three::prof_start();
//     ...
//     three::prof_stop("collect", t);
//
// prof_report prints the totals divided by a frame count, in the order the stages first ran.

bool profiling;

struct ProfSlot @local
{
    String name;
    ulong ticks;
}

ProfSlot[48] prof_slots @local;
int prof_count @local;
ulong prof_wait_base @local;

// how long sokol has waited for a frame in flight to finish (lib/sokol.c3l, Metal only: only
// used under env::DARWIN)
extern ulong sg_mtl_wait_ns @cname("sg_mtl_wait_ns");

fn ulong prof_start() => profiling ? stm::now() : 0;

fn void prof_stop(String name, ulong start)
{
    if (!profiling) return;
    ulong dt = stm::since(start);
    for (int i = 0; i < prof_count; i++)
    {
        if (prof_slots[i].name == name)
        {
            prof_slots[i].ticks += dt;
            return;
        }
    }
    if (prof_count == prof_slots.len) return;
    prof_slots[prof_count++] = { name, dt };
}

// Milliseconds the CPU has waited for the GPU since the last prof_reset (0 off Metal).
fn double prof_gpu_wait_ms()
{
    $if env::DARWIN:
        return (double)(sg_mtl_wait_ns - prof_wait_base) / 1e6;
    $else
        return 0;
    $endif
}

fn void prof_reset()
{
    for (int i = 0; i < prof_count; i++) prof_slots[i].ticks = 0;
    $if env::DARWIN:
        prof_wait_base = sg_mtl_wait_ns;
    $endif
}

fn void prof_report(int frames)
{
    for (int i = 0; i < prof_count; i++)
    {
        io::printfn("    %-22s %6.3f ms", prof_slots[i].name, stm::ms(prof_slots[i].ticks) / frames);
    }
    io::printfn("    %-22s %6.3f ms", "(waiting for the GPU)", prof_gpu_wait_ms() / frames);
}
