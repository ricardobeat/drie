// setTimeout, setInterval, clearTimeout and clearInterval, as globals.
//
// From boomkat-runtime (runtime/timers.c3): one binary heap of timers ordered by due time,
// callbacks held as registry handles. There is no event loop here to arm a native timer
// with: the frame loop calls fire_timers() before each animation callback, as a browser
// runs its due tasks before requestAnimationFrame, so a timer fires on the first frame at
// or after its due time. Before there is an animation loop, wait_for_timers() sleeps until
// each is due (a script that calls setAnimationLoop from a timeout).
module main;

import boomkat;
import std::io, std::time, std::thread;
import std::collections::list, std::collections::map;

struct PendingTimer
{
    long due; // ns since start_clock
    uint id;
    long interval; // ns, or -1 for a one-shot
    JsValue callback;
    JsValue[] args;
    sz at; // index in timer_heap
}

List{PendingTimer*} timer_heap;
HashMap{uint, PendingTimer*} timer_ids;
uint next_timer = 1;

fn void init_timers()
{
    timer_heap.init(mem, 64);
    timer_ids.init(mem);
}

fn long timer_now() => (long)start_clock.to_now();

fn bool timer_before(PendingTimer* a, PendingTimer* b) @inline => a.due < b.due || (a.due == b.due && a.id < b.id);

fn void timer_place(PendingTimer* t, sz i) @inline
{
    timer_heap[i] = t;
    t.at = i;
}

fn void timer_up(sz i)
{
    PendingTimer* t = timer_heap[i];
    while (i > 0)
    {
        sz parent = (i - 1) / 2;
        if (!timer_before(t, timer_heap[parent])) break;
        timer_place(timer_heap[parent], i);
        i = parent;
    }
    timer_place(t, i);
}

fn void timer_down(sz i)
{
    sz n = (sz)timer_heap.len();
    PendingTimer* t = timer_heap[i];
    while (true)
    {
        sz c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && timer_before(timer_heap[c + 1], timer_heap[c])) c++;
        if (!timer_before(timer_heap[c], t)) break;
        timer_place(timer_heap[c], i);
        i = c;
    }
    timer_place(t, i);
}

fn void timer_push(PendingTimer* t)
{
    timer_heap.push(t);
    timer_up((sz)timer_heap.len() - 1);
}

fn void timer_remove(PendingTimer* t)
{
    sz i = t.at;
    PendingTimer* last = timer_heap.pop()!!;
    if (last == t) return;
    timer_place(last, i);
    timer_down(i);
    timer_up(last.at);
}

fn void timer_free(PendingTimer* t)
{
    js.release(t.callback);
    foreach (a : t.args) js.release(a);
    if (t.args.len) free(t.args.ptr);
    free(t);
}

// Calls a timer's callback, then runs the promise jobs it queued (a browser's microtask
// checkpoint after each task). An error is reported and the other timers still run.
fn void timer_call(PendingTimer* t)
{
    JsValue? outcome = js.call(t.callback, t.args);
    if (catch outcome)
    {
        io::eprintfn("uncaught error in a timer: %s", js.last_error());
    }
    else
    {
        js.release(outcome);
    }
    if (catch js.get_runtime().drain_microtasks())
    {
        io::eprintfn("uncaught error in a promise job: %s", js.last_error());
    }
}

// Runs the timers that are due. The cutoff is fixed: an interval put back by this batch,
// or a timer its callbacks add, waits for a later call even if it is already due.
fn void fire_timers()
{
    long now = timer_now();
    while (timer_heap.len() && timer_heap[0].due <= now)
    {
        PendingTimer* t = timer_heap[0];
        timer_remove(t);
        if (t.interval >= 0)
        {
            t.due = timer_now() + t.interval;
            timer_push(t);
            timer_call(t);
        }
        else
        {
            timer_ids.remove(t.id);
            defer timer_free(t);
            timer_call(t);
        }
    }
}

// Until there is an animation loop or no timer is left: sleeps until the next one is due,
// then runs it.
fn void wait_for_timers()
{
    while (js_loop.is_empty() && timer_heap.len())
    {
        long wait = timer_heap[0].due - timer_now();
        if (wait > 0) (void)thread::sleep_ns((NanoDuration)wait);
        fire_timers();
    }
}

fn void add_timer(JsCtx ctx, bool repeat)
{
    JsArg f = ctx.arg(0);
    if (f.type_of() != FUNCTION)
    {
        ctx.throw_error(TYPE, "callback must be a function");
        return;
    }
    double ms = ctx.argc() > 1 ? ctx.arg(1).as_number() ?? 0 : 0;
    if (!(ms >= 0)) ms = 0;
    // as in a browser, a delay that doesn't fit in 32 bits is 1 ms
    if (ms > 2147483647) ms = 1;
    JsValue? callback = ctx.persist(f);
    if (catch callback)
    {
        ctx.throw_error(TYPE, "too many timers");
        return;
    }
    PendingTimer* t = mem::new(PendingTimer);
    t.id = next_timer++;
    t.callback = callback;
    if (ctx.argc() > 2)
    {
        t.args = mem::new_array(JsValue, (usz)ctx.argc() - 2);
        foreach (i, &a : t.args) *a = ctx.persist(ctx.arg((uint)i + 2)) ?? {};
    }
    long span = max((long)(ms * 1e6), (long)1_000_000);
    t.interval = repeat ? span : -1;
    t.due = timer_now() + span;
    timer_push(t);
    timer_ids.set(t.id, t);
    ctx.ret_number(t.id);
}

fn void h_set_timeout(JsCtx ctx, void* udata) => add_timer(ctx, false);

fn void h_set_interval(JsCtx ctx, void* udata) => add_timer(ctx, true);

// clearTimeout and clearInterval: ids are shared, as in a browser.
fn void h_clear_timer(JsCtx ctx, void* udata)
{
    double? id = ctx.arg(0).as_number();
    if (catch id) return;
    if (!(id >= 1 && id < (double)uint::max)) return;
    PendingTimer*? t = timer_ids.get((uint)id);
    if (catch t) return;
    timer_ids.remove(t.id);
    timer_remove(t);
    timer_free(t);
}

fn void register_timers(JsValue global)
{
    init_timers();
    reg(global, "setTimeout", &h_set_timeout, 2);
    reg(global, "setInterval", &h_set_interval, 2);
    reg(global, "clearTimeout", &h_clear_timer, 1);
    reg(global, "clearInterval", &h_clear_timer, 1);
}
