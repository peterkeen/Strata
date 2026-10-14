#pragma once

#include "strata/core/parallel_resource_policy.hpp"

#include <cstdint>

namespace strata::core {

enum class CacheReclaimGateResult {
    ready,
    unsafe_point,
    incoherent_resident_complement,
    already_entered,
    stale_generation,
};

// A boundary guard, not a second lifecycle state machine. Callers supply the
// safe-point facts and current allocation generation from the parallel resource
// policy. The ticket must be completed with the same generation after cache
// metadata and residency publication have finished.
class CacheReclaimGate {
public:
    CacheReclaimGateResult enter(const ParallelSafePoint& safe, bool resident_complement,
                                 bool complement_coherent, uint64_t generation) {
        if (entered_) return CacheReclaimGateResult::already_entered;
        if (!safe.ready()) return CacheReclaimGateResult::unsafe_point;
        if (resident_complement && !complement_coherent)
            return CacheReclaimGateResult::incoherent_resident_complement;
        entered_ = true;
        generation_ = generation;
        return CacheReclaimGateResult::ready;
    }

    CacheReclaimGateResult complete(uint64_t generation) {
        if (!entered_) return CacheReclaimGateResult::unsafe_point;
        if (generation != generation_) return CacheReclaimGateResult::stale_generation;
        entered_ = false;
        generation_ = 0;
        return CacheReclaimGateResult::ready;
    }

    bool entered() const { return entered_; }
    uint64_t generation() const { return generation_; }

private:
    bool entered_ = false;
    uint64_t generation_ = 0;
};

} // namespace strata::core
