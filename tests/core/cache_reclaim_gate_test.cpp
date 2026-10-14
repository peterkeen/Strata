#include "strata/core/cache_reclaim_gate.hpp"

#include <cstdio>

int main() {
    using namespace strata::core;
    CacheReclaimGate gate;
    ParallelSafePoint safe;
    safe.commits_drained = true;
    safe.dma_drained = true;
    safe.adaptive_drained = true;
    safe.outstanding_prefill_loan_bytes = 1;
    if (gate.enter(safe, false, true, 7) != CacheReclaimGateResult::unsafe_point) return 1;
    safe.outstanding_prefill_loan_bytes = 0;
    if (gate.enter(safe, true, false, 7) != CacheReclaimGateResult::incoherent_resident_complement) return 2;
    if (gate.enter(safe, true, true, 7) != CacheReclaimGateResult::ready || !gate.entered() || gate.generation() != 7)
        return 3;
    if (gate.enter(safe, false, true, 7) != CacheReclaimGateResult::already_entered) return 4;
    if (gate.complete(8) != CacheReclaimGateResult::stale_generation || !gate.entered()) return 5;
    if (gate.complete(7) != CacheReclaimGateResult::ready || gate.entered()) return 6;
    safe.commits_drained = false;
    if (gate.enter(safe, false, true, 8) != CacheReclaimGateResult::unsafe_point) return 7;
    std::puts("cache reclaim gate: safe-point drains, prefill loan, complement coherence, and generation checks passed");
    return 0;
}
