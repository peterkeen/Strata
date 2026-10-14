#pragma once

#include "strata/core/parallel_resource_policy.hpp"

#include <cstdint>
#include <utility>
#include <vector>

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
// Pure metadata plan. The GPU cache has stable slot addresses; only the tail can
// be unmapped. A failed VMM call can unmap part of that tail, so callers publish
// misses for the entire proposed tail BEFORE calling shrink, then restore only
// entries whose slots still exist. No expert-cache internal slot_of() is used:
// host_res is the adaptive tier's authoritative (layer, expert) -> slot table.
struct CacheReclaimEntry {
    int32_t index = -1;
    int32_t slot = -1;
};

inline std::vector<CacheReclaimEntry> cache_reclaim_tail(const std::vector<int32_t>& host_res,
                                                         int64_t keep_slots, int64_t live_slots) {
    std::vector<CacheReclaimEntry> entries;
    if (keep_slots < 0 || live_slots < keep_slots) return entries;
    for (size_t i = 0; i < host_res.size(); ++i)
        if (host_res[i] >= keep_slots && host_res[i] < live_slots)
            entries.push_back({static_cast<int32_t>(i), host_res[i]});
    return entries;
}

// Refill only original owners that are still missing. Adaptive swaps may have
// moved an owner while capacity was away: never overwrite their new residency.
// `remaining` tracks slots still unmapped OR failed to refill; those are coherent
// file-tier misses, distinct from a failure to map the requested capacity.
struct CacheRefillPlan {
    std::vector<CacheReclaimEntry> fill;
    std::vector<CacheReclaimEntry> remaining;
    int64_t skipped_already_resident = 0;
};

inline CacheRefillPlan cache_refill_plan(const std::vector<CacheReclaimEntry>& evicted,
                                         const std::vector<int32_t>& host_res, int64_t live_slots) {
    CacheRefillPlan plan;
    for (const auto& entry : evicted) {
        if (entry.index < 0 || static_cast<size_t>(entry.index) >= host_res.size()) continue;
        if (host_res[static_cast<size_t>(entry.index)] >= 0) {
            ++plan.skipped_already_resident;
        } else if (entry.slot >= live_slots) {
            plan.remaining.push_back(entry);
        } else {
            plan.fill.push_back(entry);
        }
    }
    return plan;
}

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
