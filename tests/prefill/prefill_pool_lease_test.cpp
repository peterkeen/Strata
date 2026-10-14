// str-s1w.3: the CPU pool lease between prompt chunks and batch decode windows.  No GPU, no model.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

#include "strata/prefill/pool_lease.hpp"

using namespace strata::prefill;
using namespace std::chrono_literals;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static void acquire_release() {
    PoolLease l;
    CHECK(l.owner() == PoolOwner::none);
    CHECK(l.try_acquire(PoolOwner::prompt));
    CHECK(l.owner() == PoolOwner::prompt);
    CHECK(!l.try_acquire(PoolOwner::prompt));          // a second prompt stage (layer split): one at a time
    CHECK(!l.try_acquire(PoolOwner::decode));
    CHECK(!l.release(PoolOwner::decode));               // not the holder: ignored, counted
    CHECK(l.bad_releases() == 1);
    CHECK(l.owner() == PoolOwner::prompt);
    CHECK(l.release(PoolOwner::prompt));
    CHECK(l.owner() == PoolOwner::none);
    CHECK(!l.release(PoolOwner::prompt));               // double release: ignored
    CHECK(l.try_acquire(PoolOwner::decode));
    CHECK(l.release(PoolOwner::decode));
    CHECK(l.prompt_grants() == 1 && l.prompt_busy() == 1);
}

// The prompt path's fallback: while a decode window holds the pool the chunk runs on the GPU alone.
static void busy_fallback() {
    PoolLease l;
    PoolLease::Hold dec(l, PoolOwner::decode, 1000us);
    CHECK(dec.held());
    {
        PoolLease::Hold chunk(l, PoolOwner::prompt);
        CHECK(!chunk.held());                           // fallback, never blocks
    }
    CHECK(l.owner() == PoolOwner::decode);              // the failed hold released nothing
    dec.reset();
    PoolLease::Hold chunk(l, PoolOwner::prompt);
    CHECK(chunk.held());
}

// Cancellation / error: every exit path gives the pool back.
static bool chunk_that_fails(PoolLease& l, bool cancel) {
    PoolLease::Hold h(l, PoolOwner::prompt);
    if (!h.held()) return false;
    if (cancel) return false;                           // "cancelled" mid-chunk
    throw std::runtime_error("prefill: a CPU expert has no blob");
}
static void cancel_and_error_release() {
    PoolLease l;
    CHECK(!chunk_that_fails(l, true));
    CHECK(l.owner() == PoolOwner::none);
    bool threw = false;
    try { chunk_that_fails(l, false); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    CHECK(l.owner() == PoolOwner::none);
    PoolLease::Hold a(l, PoolOwner::prompt);
    PoolLease::Hold b(std::move(a));                    // moved: released exactly once
    CHECK(!a.held() && b.held());
    b.reset();
    CHECK(l.owner() == PoolOwner::none && l.bad_releases() == 0);
}

// No deadlock: a decode window on the thread that (wrongly) still holds the prompt lease times out.
static void bounded_wait_no_deadlock() {
    PoolLease l;
    PoolLease::Hold chunk(l, PoolOwner::prompt);
    const auto t0 = std::chrono::steady_clock::now();
    PoolLease::Hold dec(l, PoolOwner::decode, 20000us);
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(!dec.held());
    CHECK(ms >= 19.0 && ms < 2000.0);
    CHECK(l.timeouts() == 1 && l.waits() == 1);
    CHECK(l.owner() == PoolOwner::prompt);
}

// A decode window waits for a chunk that holds the pool on another thread, then gets it.
static void decode_waits_for_chunk() {
    PoolLease l;
    std::atomic<bool> holding{false};
    std::thread t([&] {
        PoolLease::Hold chunk(l, PoolOwner::prompt);
        holding = true;
        std::this_thread::sleep_for(15ms);
    });
    while (!holding) std::this_thread::yield();
    PoolLease::Hold dec(l, PoolOwner::decode, 2000000us);
    CHECK(dec.held());
    t.join();
}

// Two slots' decode windows and a prompt reader contend: the pool's critical section never has two owners, and a
// slot's "activation buffer" is never written by the other owner while held.
static void contention_two_slots() {
    PoolLease l;
    std::atomic<int> inside{0}, overlap{0};
    std::atomic<int> buffer_owner{0};                   // 0 free, 1 prompt, 2/3 slot 0/1
    std::atomic<bool> stop{false};
    std::atomic<int> prompt_cpu{0}, prompt_gpu{0}, windows{0}, corrupt{0};
    auto critical = [&](int tag) {
        if (inside.fetch_add(1) != 0) overlap.fetch_add(1);
        int expect = 0;
        if (!buffer_owner.compare_exchange_strong(expect, tag)) corrupt.fetch_add(1);
        std::atomic<int> spin{0};
        while (spin.fetch_add(1, std::memory_order_relaxed) < 200) {}
        if (buffer_owner.load() != tag) corrupt.fetch_add(1);
        buffer_owner.store(0);
        inside.fetch_sub(1);
    };
    std::vector<std::thread> th;
    for (int slot = 0; slot < 2; ++slot)
        th.emplace_back([&, slot] {
            while (!stop) {
                PoolLease::Hold h(l, PoolOwner::decode, 1000000us);
                if (!h.held()) { corrupt.fetch_add(1); continue; }
                critical(2 + slot);
                windows.fetch_add(1);
            }
        });
    th.emplace_back([&] {
        while (!stop) {
            PoolLease::Hold h(l, PoolOwner::prompt);
            if (h.held()) { critical(1); prompt_cpu.fetch_add(1); }
            else prompt_gpu.fetch_add(1);              // the GPU-only fallback
        }
    });
    std::this_thread::sleep_for(150ms);
    stop = true;
    for (auto& t : th) t.join();
    CHECK(overlap.load() == 0);
    CHECK(corrupt.load() == 0);
    CHECK(windows.load() > 0);
    CHECK(prompt_cpu.load() > 0);
    CHECK(prompt_gpu.load() > 0);                        // the busy fallback was exercised
    CHECK(l.owner() == PoolOwner::none);
    CHECK(l.bad_releases() == 0);
}

static void verdicts() {
    SlotShareInputs in;
    in.batch = 0; in.opt_in = true;
    CHECK(!slot_share_verdict(in).pool_to_prompt);        // solo engine: the existing rule, not this one
    in.batch = 2; in.opt_in = false;
    CHECK(!slot_share_verdict(in).pool_to_prompt);        // off path unchanged
    in.opt_in = true;
    CHECK(slot_share_verdict(in).pool_to_prompt);         // nibbler's Q4 two-slot config
    SlotShareInputs x = in; x.no_pool = true;          CHECK(!slot_share_verdict(x).pool_to_prompt);
    x = in; x.batch_groups = 2;                        CHECK(!slot_share_verdict(x).pool_to_prompt);
    x = in; x.batch_groups_auto = true;                CHECK(!slot_share_verdict(x).pool_to_prompt);
    x = in; x.layer_split = true;                      CHECK(!slot_share_verdict(x).pool_to_prompt);
    x = in; x.multi_gpu = true;                        CHECK(!slot_share_verdict(x).pool_to_prompt);
    x = in; x.hip = true;                              CHECK(!slot_share_verdict(x).pool_to_prompt);
    static_assert(slot_share_verdict(SlotShareInputs{true, 2}).pool_to_prompt, "constexpr");
}

int main() {
    acquire_release();
    busy_fallback();
    cancel_and_error_release();
    bounded_wait_no_deadlock();
    decode_waits_for_chunk();
    contention_two_slots();
    verdicts();
    if (failed) { std::fprintf(stderr, "prefill_pool_lease_test: %d failure(s)\n", failed); return 1; }
    std::printf("prefill_pool_lease_test: OK\n");
    return 0;
}
