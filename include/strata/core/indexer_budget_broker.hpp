#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

namespace strata::core {

// CPU-only feasibility ledger for sharing one device-memory budget between
// private indexer chunks, future pooled-indexer slabs, and the expert cache. It
// never calls ExpertCache or VMM APIs: ExpertCache::admit does not evict, so the
// serving/runtime layer must provide already-safe victim candidates and later
// report which evictions and indexer commitments actually happened.
//
// The ledger keeps the reserve floor and transient allowance unavailable for
// indexer growth. A failed indexer publication after reported evictions leaves
// those experts evicted (coherent future cache misses) and does not re-advertise
// stale residency. Restore failures and uncertain outcomes poison the ledger.
// A malformed or inconsistent apply report for a current plan is treated as a
// post-mutation uncertainty and poisons the broker; callers must tear down or
// rebuild the ledger rather than retrying after such a report. Stale/replayed
// plans are rejected by generation before any report fields are trusted.
class IndexerBudgetBroker {
public:
    enum class Outcome {
        ready = 0,
        shortage,
        unsupported,
        fatal,
    };

    enum class CommitResult {
        committed = 0,
        indexer_publication_failed,
        restore_failed,
        uncertain,
    };

    struct Config {
        uint64_t device_budget_bytes = 0;
        uint64_t reserve_floor_bytes = 0;
        uint64_t transient_allowance_bytes = 0;
        uint64_t shrink_low_water_bytes = 0;
        uint64_t shrink_target_headroom_bytes = 0;
    };

    struct InitialState {
        uint64_t private_indexer_committed_bytes = 0;
        uint64_t pooled_slab_committed_bytes = 0;
        uint64_t expert_mapped_bytes = 0;
        uint64_t evicted_recoverable_bytes = 0;
    };

    struct Telemetry {
        uint64_t device_budget_bytes = 0;
        uint64_t reserve_floor_bytes = 0;
        uint64_t transient_allowance_bytes = 0;
        uint64_t private_indexer_committed_bytes = 0;
        uint64_t pooled_slab_committed_bytes = 0;
        uint64_t actual_indexer_commitment_bytes = 0;
        uint64_t expert_mapped_bytes = 0;
        uint64_t evicted_recoverable_bytes = 0;
        uint64_t unused_bytes = 0;       // Headroom above reserve/transient floors.
        uint64_t external_activation_bytes = 0;
        bool poisoned = false;
    };

    struct GrowthRequest {
        uint64_t private_indexer_growth_bytes = 0;
        uint64_t pooled_slab_growth_bytes = 0;
        // Optional/best-effort growth is deliberately not part of feasibility.
        // The broker evicts only for the minimum required request.
        uint64_t optional_growth_bytes = 0;
        // Non-indexer activation memory that must coexist with this growth.
        uint64_t external_activation_bytes = 0;
    };

    struct ExpertVictimCandidate {
        int32_t id = -1;
        uint64_t bytes = 0;
        bool in_flight_copy = false;
        bool prefill_loan = false;
        bool adaptive_publication = false;
        bool retained_pointer = false;
        bool unsupported_multi_gpu = false;
    };

    struct PlannedVictim {
        int32_t id = -1;
        uint64_t bytes = 0;
    };

    struct GrowthPlan {
        Outcome outcome = Outcome::fatal;
        uint64_t required_growth_bytes = 0;
        uint64_t optional_ignored_bytes = 0;
        uint64_t free_headroom_bytes = 0;
        uint64_t recovered_bytes = 0;
        uint64_t shortfall_bytes = 0;
        uint64_t external_activation_bytes = 0;
        std::vector<PlannedVictim> victims;
        std::string_view reason = "uninitialized";
        uint64_t generation = 0;
    };

    struct ApplyReport {
        CommitResult result = CommitResult::committed;
        uint64_t evicted_bytes = 0;
        std::vector<int32_t> evicted_ids;
        uint64_t private_indexer_committed_bytes = 0;
        uint64_t pooled_slab_committed_bytes = 0;
    };

    struct ShrinkPlan {
        Outcome outcome = Outcome::fatal;
        uint64_t retention_floor_bytes = 0;
        uint64_t chunk_bytes = 0;
        uint64_t free_headroom_bytes = 0;
        uint64_t releasable_bytes = 0;
        uint64_t release_bytes = 0;
        std::string_view reason = "uninitialized";
        uint64_t generation = 0;
    };

    static bool create(const Config& config, const InitialState& initial, IndexerBudgetBroker& out) {
        IndexerBudgetBroker tmp;
        tmp.config_ = config;
        tmp.private_indexer_committed_bytes_ = initial.private_indexer_committed_bytes;
        tmp.pooled_slab_committed_bytes_ = initial.pooled_slab_committed_bytes;
        tmp.expert_mapped_bytes_ = initial.expert_mapped_bytes;
        tmp.evicted_recoverable_bytes_ = initial.evicted_recoverable_bytes;
        if (config.shrink_target_headroom_bytes < config.shrink_low_water_bytes) return false;
        if (!tmp.state_fits_budget()) return false;
        out = std::move(tmp);
        return true;
    }

    bool poisoned() const { return poisoned_; }

    Telemetry telemetry() const {
        Telemetry t;
        t.device_budget_bytes = config_.device_budget_bytes;
        t.reserve_floor_bytes = config_.reserve_floor_bytes;
        t.transient_allowance_bytes = config_.transient_allowance_bytes;
        t.private_indexer_committed_bytes = private_indexer_committed_bytes_;
        t.pooled_slab_committed_bytes = pooled_slab_committed_bytes_;
        t.actual_indexer_commitment_bytes = private_indexer_committed_bytes_ + pooled_slab_committed_bytes_;
        t.expert_mapped_bytes = expert_mapped_bytes_;
        t.evicted_recoverable_bytes = evicted_recoverable_bytes_;
        t.external_activation_bytes = external_activation_bytes_;
        t.unused_bytes = free_headroom();
        t.poisoned = poisoned_;
        return t;
    }

    GrowthPlan plan_growth(const GrowthRequest& request, const std::vector<ExpertVictimCandidate>& candidates) const {
        GrowthPlan plan;
        plan.optional_ignored_bytes = request.optional_growth_bytes;
        plan.external_activation_bytes = request.external_activation_bytes;
        plan.generation = generation_;
        if (poisoned_) return fatal_plan(plan, "poisoned");
        uint64_t required = 0;
        if (!checked_add(request.private_indexer_growth_bytes, request.pooled_slab_growth_bytes, required))
            return fatal_plan(plan, "growth overflow");
        plan.required_growth_bytes = required;
        plan.free_headroom_bytes = free_headroom();
        uint64_t total_required = 0;
        if (!checked_add(required, request.external_activation_bytes, total_required))
            return fatal_plan(plan, "activation growth overflow");
        if (total_required <= plan.free_headroom_bytes) {
            plan.outcome = Outcome::ready;
            plan.reason = "headroom";
            return plan;
        }

        const uint64_t need = total_required - plan.free_headroom_bytes;
        bool unsupported_candidate_needed = false;
        for (const ExpertVictimCandidate& c : candidates) {
            if (c.bytes == 0) continue;
            if (c.id < 0 || contains_candidate_id(plan.victims, c.id)) return fatal_plan(plan, "invalid victim candidate");
            if (c.unsupported_multi_gpu) {
                unsupported_candidate_needed = true;
                continue;
            }
            if (c.in_flight_copy || c.prefill_loan || c.adaptive_publication || c.retained_pointer) continue;
            plan.victims.push_back({c.id, c.bytes});
            if (!checked_add(plan.recovered_bytes, c.bytes, plan.recovered_bytes)) return fatal_plan(plan, "victim overflow");
            if (plan.recovered_bytes >= need) break; // Minimum necessary victims, deterministic caller order.
        }
        if (plan.recovered_bytes >= need) {
            plan.outcome = Outcome::ready;
            plan.reason = "victims";
            return plan;
        }
        plan.shortfall_bytes = need - plan.recovered_bytes;
        plan.outcome = unsupported_candidate_needed ? Outcome::unsupported : Outcome::shortage;
        plan.reason = unsupported_candidate_needed ? "unsupported multi-gpu victim" : "insufficient safe victims";
        return plan;
    }

    bool apply_growth_result(const GrowthPlan& plan, const ApplyReport& report) {
        if (poisoned_ || plan.outcome != Outcome::ready) return false;
        if (plan.generation != generation_) return false;
        uint64_t report_evicted = 0;
        if (!validate_report_against_plan(plan, report, report_evicted)) return poison_and_fail();
        if (report.result == CommitResult::committed) {
            uint64_t committed = 0;
            if (!checked_add(report.private_indexer_committed_bytes, report.pooled_slab_committed_bytes, committed))
                return poison_and_fail();
            if (committed != plan.required_growth_bytes) return poison_and_fail();
            if (!can_apply_delta(report_evicted, report.private_indexer_committed_bytes,
                                 report.pooled_slab_committed_bytes, plan.external_activation_bytes))
                return poison_and_fail();
            apply_evictions(report_evicted);
            private_indexer_committed_bytes_ += report.private_indexer_committed_bytes;
            pooled_slab_committed_bytes_ += report.pooled_slab_committed_bytes;
            external_activation_bytes_ += plan.external_activation_bytes;
            ++generation_;
            return state_fits_budget();
        }
        if (report.private_indexer_committed_bytes != 0 || report.pooled_slab_committed_bytes != 0) return poison_and_fail();
        if (report_evicted > expert_mapped_bytes_) return poison_and_fail();
        apply_evictions(report_evicted);
        if (report.result == CommitResult::restore_failed || report.result == CommitResult::uncertain) poisoned_ = true;
        ++generation_;
        return state_fits_budget();
    }

    // Release only after private activation resources are detached and freed.
    bool release_external_activation(uint64_t bytes) {
        if (poisoned_ || bytes == 0 || bytes > external_activation_bytes_) return false;
        external_activation_bytes_ -= bytes;
        ++generation_;
        return state_fits_budget();
    }

    ShrinkPlan plan_shrink(uint64_t retention_floor_bytes, uint64_t chunk_bytes) const {
        ShrinkPlan plan;
        plan.retention_floor_bytes = retention_floor_bytes;
        plan.chunk_bytes = chunk_bytes;
        plan.generation = generation_;
        if (poisoned_) {
            plan.outcome = Outcome::fatal;
            plan.reason = "poisoned";
            return plan;
        }
        if (chunk_bytes == 0 || retention_floor_bytes > private_indexer_committed_bytes_) {
            plan.outcome = Outcome::fatal;
            plan.reason = "invalid shrink input";
            return plan;
        }
        plan.free_headroom_bytes = free_headroom();
        if (plan.free_headroom_bytes >= config_.shrink_low_water_bytes) {
            plan.outcome = Outcome::ready;
            plan.reason = "above low water";
            return plan;
        }
        plan.releasable_bytes = private_indexer_committed_bytes_ - retention_floor_bytes;
        if (plan.releasable_bytes < chunk_bytes) {
            plan.outcome = Outcome::ready;
            plan.reason = "retention floor";
            return plan;
        }
        uint64_t desired = config_.shrink_target_headroom_bytes - plan.free_headroom_bytes;
        desired = std::min(desired, plan.releasable_bytes);
        plan.release_bytes = (desired / chunk_bytes) * chunk_bytes;
        if (plan.release_bytes == 0) {
            plan.outcome = Outcome::ready;
            plan.reason = "hysteresis chunk floor";
            return plan;
        }
        plan.outcome = Outcome::ready;
        plan.reason = "shrink";
        return plan;
    }

    bool apply_shrink_result(const ShrinkPlan& plan, uint64_t released_private_indexer_bytes, bool uncertain) {
        if (poisoned_ || plan.outcome != Outcome::ready || plan.release_bytes == 0) return false;
        if (plan.generation != generation_) return false;
        if (released_private_indexer_bytes != plan.release_bytes) return poison_and_fail();
        if (released_private_indexer_bytes > private_indexer_committed_bytes_) return poison_and_fail();
        if (private_indexer_committed_bytes_ - released_private_indexer_bytes < plan.retention_floor_bytes)
            return poison_and_fail();
        if (uncertain) {
            poisoned_ = true;
            ++generation_;
            return state_fits_budget();
        }
        private_indexer_committed_bytes_ -= released_private_indexer_bytes;
        ++generation_;
        return state_fits_budget();
    }

private:
    static bool checked_add(uint64_t a, uint64_t b, uint64_t& out) {
        if (a > std::numeric_limits<uint64_t>::max() - b) return false;
        out = a + b;
        return true;
    }

    static bool contains_candidate_id(const std::vector<PlannedVictim>& victims, int32_t id) {
        for (const auto& v : victims)
            if (v.id == id) return true;
        return false;
    }

    static GrowthPlan fatal_plan(GrowthPlan plan, std::string_view reason) {
        plan.outcome = Outcome::fatal;
        plan.reason = reason;
        return plan;
    }

    uint64_t committed_and_reserved() const {
        uint64_t total = config_.reserve_floor_bytes;
        if (!checked_add(total, config_.transient_allowance_bytes, total) ||
            !checked_add(total, private_indexer_committed_bytes_, total) ||
            !checked_add(total, pooled_slab_committed_bytes_, total) ||
            !checked_add(total, expert_mapped_bytes_, total) ||
            !checked_add(total, external_activation_bytes_, total))
            return std::numeric_limits<uint64_t>::max();
        return total;
    }

    uint64_t free_headroom() const {
        const uint64_t used = committed_and_reserved();
        return used >= config_.device_budget_bytes ? 0 : config_.device_budget_bytes - used;
    }

    bool state_fits_budget() const { return committed_and_reserved() <= config_.device_budget_bytes; }

    void apply_evictions(uint64_t evicted_bytes) {
        expert_mapped_bytes_ -= evicted_bytes;
        evicted_recoverable_bytes_ += evicted_bytes;
    }

    bool can_apply_delta(uint64_t evicted_bytes, uint64_t private_growth, uint64_t pooled_growth,
                         uint64_t external_growth) const {
        if (evicted_bytes > expert_mapped_bytes_) return false;
        uint64_t expert_after = expert_mapped_bytes_ - evicted_bytes;
        uint64_t private_after = 0, pooled_after = 0;
        if (!checked_add(private_indexer_committed_bytes_, private_growth, private_after) ||
            !checked_add(pooled_slab_committed_bytes_, pooled_growth, pooled_after))
            return false;
        uint64_t used = config_.reserve_floor_bytes;
        return checked_add(used, config_.transient_allowance_bytes, used) &&
               checked_add(used, private_after, used) && checked_add(used, pooled_after, used) &&
               checked_add(used, expert_after, used) &&
               checked_add(used, external_activation_bytes_, used) &&
               checked_add(used, external_growth, used) && used <= config_.device_budget_bytes;
    }

    bool validate_report_against_plan(const GrowthPlan& plan, const ApplyReport& report,
                                      uint64_t& report_evicted) const {
        report_evicted = 0;
        std::vector<int32_t> seen;
        seen.reserve(report.evicted_ids.size());
        for (int32_t id : report.evicted_ids) {
            for (int32_t previous : seen)
                if (previous == id) return false;
            seen.push_back(id);
            bool found = false;
            for (const auto& v : plan.victims) {
                if (v.id == id) {
                    if (!checked_add(report_evicted, v.bytes, report_evicted)) return false;
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        return report.evicted_bytes == report_evicted;
    }

    bool poison_and_fail() {
        poisoned_ = true;
        ++generation_;
        return false;
    }

    Config config_{};
    uint64_t private_indexer_committed_bytes_ = 0;
    uint64_t pooled_slab_committed_bytes_ = 0;
    uint64_t expert_mapped_bytes_ = 0;
    uint64_t evicted_recoverable_bytes_ = 0;
    uint64_t external_activation_bytes_ = 0;
    bool poisoned_ = false;
    uint64_t generation_ = 1;
};

inline std::string_view indexer_budget_outcome_name(IndexerBudgetBroker::Outcome outcome) {
    using Outcome = IndexerBudgetBroker::Outcome;
    switch (outcome) {
    case Outcome::ready: return "ready";
    case Outcome::shortage: return "shortage";
    case Outcome::unsupported: return "unsupported";
    case Outcome::fatal: return "fatal";
    }
    return "unknown";
}

} // namespace strata::core
