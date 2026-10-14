#include "strata/core/cache_reclaim_gate.hpp"

#include <cstdio>
#include <cstdlib>

// Not <cassert>: the default Release build defines NDEBUG and would compile every check away.
static int checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)
#include <vector>

int main() {
    using namespace strata::core;
    std::vector<int32_t> res{0, 5, -1, 7, 3, 9};
    auto tail = cache_reclaim_tail(res, 5, 10);
    CHECK(tail.size() == 3);
    CHECK(tail[0].index == 1 && tail[0].slot == 5);
    CHECK(tail[1].index == 3 && tail[1].slot == 7);
    CHECK(tail[2].index == 5 && tail[2].slot == 9);

    CacheReclaimGate gate;
    ParallelSafePoint safe{true, true, true, 0};
    CHECK(gate.enter(safe, true, true, 4) == CacheReclaimGateResult::ready);
    // Publication precedes unmapping. On partial failure only restore the
    // entries whose addresses remain mapped; the other entries stay misses.
    for (const auto& v : tail) res[static_cast<size_t>(v.index)] = -1;
    for (const auto& v : tail)
        if (v.slot < 7) res[static_cast<size_t>(v.index)] = v.slot;
    CHECK((res == std::vector<int32_t>{0, 5, -1, -1, 3, -1}));
    CHECK(gate.complete(5) == CacheReclaimGateResult::stale_generation);
    CHECK(gate.complete(4) == CacheReclaimGateResult::ready);
    // Grow has partially mapped: only slot 7 may be refilled.
    auto plan = cache_refill_plan(tail, res, 8);
    CHECK(plan.fill.size() == 1 && plan.fill[0].index == 3);
    CHECK(plan.remaining.size() == 1 && plan.remaining[0].index == 5);
    CHECK(plan.skipped_already_resident == 1);
    // Failed copy/sync leaves all pending misses; no residency is published.
    CHECK(res[3] == -1 && res[5] == -1);
    // Successful copy publishes slot 7; slot 9 remains a coherent file miss.
    res[3] = 7;
    plan = cache_refill_plan(tail, res, 8);
    CHECK(plan.fill.empty() && plan.remaining.size() == 1);
    res[5] = 2; // adaptive tier has moved the original owner
    plan = cache_refill_plan(tail, res, 10);
    CHECK(plan.fill.empty() && plan.remaining.empty());
    CHECK(plan.skipped_already_resident == 3);
    safe.outstanding_prefill_loan_bytes = 1;
    CHECK(gate.enter(safe, true, true, 6) == CacheReclaimGateResult::unsafe_point);
    std::printf("cache reclaim plan: tail, partial unmap, refill, rollback and safe point passed (%d checks)\n", checks);
}
