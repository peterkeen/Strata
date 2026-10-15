#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "strata/core/indexer_budget_broker.hpp"

namespace strata::core {

// CPU-only resource ledger and lifecycle model for demand-allocated parallel
// serving resources. It intentionally does not call CUDA/HIP/SYCL, ExpertCache,
// SharedKvRuntime, Verifier or MtpDrafter APIs. Runtime integration must bind
// those systems through the explicit callbacks below and keep this model's
// publication order: plan -> shrink expert capacity -> allocate/bind private
// resources -> publish; and invalidate/detach -> free private resources -> grow
// expert capacity on reclamation.
//
// The ledger separates logical capacity from physical batch sessions. In the
// current main-admission design, parallel=2 advertises two logical streams but
// activation needs two physical batch sessions because the main session remains
// the prompt-admission owner. Shared main K/V pools, main weights and shared
// draft weights are baseline resources and are never released by this model.
// Private indexer bytes owned by str-a2y's broker are accounted separately from
// non-indexer slot-session bytes so lazy-session savings and indexer savings are
// not added twice.

enum class ParallelResourceClass {
    shared_main_pools = 0,
    shared_weights,
    private_slot_sessions,
    brokered_slot_indexer,
    batch_verifier_and_graphs,
    slot_draft_rings_and_buffers,
    transition_temporaries,
    safety_reserve,
    expert_cache_capacity,
};

inline std::string_view parallel_resource_class_name(ParallelResourceClass c) {
    switch (c) {
    case ParallelResourceClass::shared_main_pools: return "shared_main_pools";
    case ParallelResourceClass::shared_weights: return "shared_weights";
    case ParallelResourceClass::private_slot_sessions: return "private_slot_sessions";
    case ParallelResourceClass::brokered_slot_indexer: return "brokered_slot_indexer";
    case ParallelResourceClass::batch_verifier_and_graphs: return "batch_verifier_and_graphs";
    case ParallelResourceClass::slot_draft_rings_and_buffers: return "slot_draft_rings_and_buffers";
    case ParallelResourceClass::transition_temporaries: return "transition_temporaries";
    case ParallelResourceClass::safety_reserve: return "safety_reserve";
    case ParallelResourceClass::expert_cache_capacity: return "expert_cache_capacity";
    }
    return "unknown";
}

enum class ParallelState { solo = 0, activating, parallel, draining };

inline std::string_view parallel_state_name(ParallelState s) {
    switch (s) {
    case ParallelState::solo: return "solo";
    case ParallelState::activating: return "activating";
    case ParallelState::parallel: return "parallel";
    case ParallelState::draining: return "draining";
    }
    return "unknown";
}

enum class ParallelOpCode {
    ok = 0,
    invalid,
    overflow,
    safe_point_missing,
    cache_shortage,
    allocation_failed,
    map_failed,
    binding_failed,
    verifier_failed,
    graph_failed,
    draft_failed,
    publication_failed,
    policy_required,
    idle_grace,
    not_idle,
    cache_restore_failed,
    cancelled,
};

inline bool parallel_op_failed(ParallelOpCode c) { return c != ParallelOpCode::ok; }

inline std::string_view parallel_op_code_name(ParallelOpCode c) {
    switch (c) {
    case ParallelOpCode::ok: return "ok";
    case ParallelOpCode::invalid: return "invalid";
    case ParallelOpCode::overflow: return "overflow";
    case ParallelOpCode::safe_point_missing: return "safe_point_missing";
    case ParallelOpCode::cache_shortage: return "cache_shortage";
    case ParallelOpCode::allocation_failed: return "allocation_failed";
    case ParallelOpCode::map_failed: return "map_failed";
    case ParallelOpCode::binding_failed: return "binding_failed";
    case ParallelOpCode::verifier_failed: return "verifier_failed";
    case ParallelOpCode::graph_failed: return "graph_failed";
    case ParallelOpCode::draft_failed: return "draft_failed";
    case ParallelOpCode::publication_failed: return "publication_failed";
    case ParallelOpCode::policy_required: return "policy_required";
    case ParallelOpCode::idle_grace: return "idle_grace";
    case ParallelOpCode::not_idle: return "not_idle";
    case ParallelOpCode::cache_restore_failed: return "cache_restore_failed";
    case ParallelOpCode::cancelled: return "cancelled";
    }
    return "unknown";
}

struct ParallelResourceLine {
    ParallelResourceClass klass = ParallelResourceClass::private_slot_sessions;
    uint64_t bytes = 0;
    bool baseline = false;       // already paid in solo mode; never reclaimed by lazy slots.
    bool broker_owned = false;   // str-a2y / broker-owned bytes; not counted in non-indexer slot bytes.
    std::string_view note;
};

struct ParallelActivationBudget {
    uint64_t required_runtime_bytes = 0;          // Non-indexer private runtime resources to allocate for parallel mode.
    uint64_t required_brokered_indexer_bytes = 0; // Private indexer bytes delegated to the str-a2y budget broker.
    uint64_t required_total_bytes = 0;            // Physical capacity needed from headroom/expert-cache exchange.
    uint64_t baseline_shared_bytes = 0;           // Main pools/weights that are not charged again.
    uint64_t safety_reserve_bytes = 0;            // Floor that must remain unavailable to cache/session growth.
    std::vector<ParallelResourceLine> lines;
};

struct ParallelResourceLedger {
    uint32_t logical_capacity = 2;
    uint32_t required_physical_batch_sessions = 2;

    uint64_t shared_main_pool_bytes = 0;
    uint64_t shared_weight_bytes = 0;
    uint64_t safety_reserve_bytes = 0;
    uint64_t expert_cache_mapped_bytes = 0;

    // Per physical batch session. This excludes slot_indexer_brokered_bytes.
    uint64_t private_slot_session_bytes = 0;
    // Per physical batch session, owned by str-a2y's private/pooled indexer broker.
    uint64_t slot_indexer_brokered_bytes = 0;
    // One activation-wide verifier staging/graph allowance.
    uint64_t batch_verifier_graph_bytes = 0;
    // Per physical batch session. Shared MTP weights are not represented here.
    uint64_t slot_draft_private_bytes = 0;
    // Activation-only handoff images, scratch and graph-capture margin.
    uint64_t transition_temporary_bytes = 0;
    bool lazy_mtp_requested = false;

    static bool checked_add(uint64_t a, uint64_t b, uint64_t& out) {
        if (a > std::numeric_limits<uint64_t>::max() - b) return false;
        out = a + b;
        return true;
    }

    static bool checked_mul(uint64_t a, uint64_t b, uint64_t& out) {
        if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
        out = a * b;
        return true;
    }

    bool build_activation_budget(ParallelActivationBudget& out, std::string& error) const {
        out = {};
        if (logical_capacity < 2 || required_physical_batch_sessions < logical_capacity) {
            error = "parallel resource ledger: logical capacity needs matching physical batch sessions";
            return false;
        }
        uint64_t private_sessions = 0, brokered_indexer = 0, draft = 0, tmp = 0;
        if (!checked_mul(private_slot_session_bytes, required_physical_batch_sessions, private_sessions) ||
            !checked_mul(slot_indexer_brokered_bytes, required_physical_batch_sessions, brokered_indexer) ||
            !checked_mul(slot_draft_private_bytes, required_physical_batch_sessions, draft)) {
            error = "parallel resource ledger: multiplication overflow";
            return false;
        }
        if (!lazy_mtp_requested) draft = 0;
        uint64_t runtime = 0;
        if (!checked_add(private_sessions, batch_verifier_graph_bytes, runtime) ||
            !checked_add(runtime, draft, runtime) ||
            !checked_add(runtime, transition_temporary_bytes, runtime) ||
            !checked_add(runtime, brokered_indexer, tmp)) {
            error = "parallel resource ledger: activation total overflow";
            return false;
        }
        uint64_t baseline = 0;
        if (!checked_add(shared_main_pool_bytes, shared_weight_bytes, baseline)) {
            error = "parallel resource ledger: baseline overflow";
            return false;
        }
        out.required_runtime_bytes = runtime;
        out.required_brokered_indexer_bytes = brokered_indexer;
        out.required_total_bytes = tmp;
        out.baseline_shared_bytes = baseline;
        out.safety_reserve_bytes = safety_reserve_bytes;
        out.lines = {
            {ParallelResourceClass::shared_main_pools, shared_main_pool_bytes, true, false, "main K/V pools and RoPE remain shared"},
            {ParallelResourceClass::shared_weights, shared_weight_bytes, true, false, "main and shared draft weights are not duplicated"},
            {ParallelResourceClass::private_slot_sessions, private_sessions, false, false, "non-indexer private slot session state"},
            {ParallelResourceClass::brokered_slot_indexer, brokered_indexer, false, true, "delegated to str-a2y budget broker"},
            {ParallelResourceClass::batch_verifier_and_graphs, batch_verifier_graph_bytes, false, false, "batch verifier staging and graph margin"},
            {ParallelResourceClass::slot_draft_rings_and_buffers, draft, false, false, "optional lazy MTP private rings/buffers only"},
            {ParallelResourceClass::transition_temporaries, transition_temporary_bytes, false, false, "handoff images and temporary activation scratch"},
            {ParallelResourceClass::safety_reserve, safety_reserve_bytes, true, false, "driver floor; not spendable headroom"},
            {ParallelResourceClass::expert_cache_capacity, expert_cache_mapped_bytes, true, false, "elastic source/sink for physical capacity"},
        };
        error.clear();
        return true;
    }
};

// Combined planning result passed to the safe-point runtime seam. The broker
// prices private indexer growth and the remaining runtime bytes as a live
// external activation reservation. Its reserve/transient floors remain broker
// policy and are not added again to required_total_bytes.
struct ParallelBrokerActivationPlan {
    ParallelActivationBudget budget;
    IndexerBudgetBroker::GrowthPlan broker_plan;
};

inline bool build_parallel_broker_activation_plan(
    const ParallelResourceLedger& ledger,
    const IndexerBudgetBroker& broker,
    const std::vector<IndexerBudgetBroker::ExpertVictimCandidate>& candidates,
    ParallelBrokerActivationPlan& out,
    std::string& error) {
    out = {};
    if (!ledger.build_activation_budget(out.budget, error)) return false;
    out.broker_plan = broker.plan_growth(
        {out.budget.required_brokered_indexer_bytes, 0, 0, out.budget.required_runtime_bytes}, candidates);
    if (out.broker_plan.outcome != IndexerBudgetBroker::Outcome::ready) {
        error = "parallel activation: indexer broker rejected combined activation budget";
        return false;
    }
    error.clear();
    return true;
}

// Find the largest segment-aligned cache prefix that releases the activation
// budget without spending the driver's free-memory floor. `bytes_of` must
// return the mapped byte prefix for each slot count (not an average blob size).
// No state changes on failure: the caller may refuse the waiting admission.
template <class BytesOf>
inline bool parallel_activation_keep_slots(int64_t slots, uint64_t mapped_bytes,
                                           uint64_t free_bytes, uint64_t required_bytes,
                                           uint64_t free_floor_bytes, BytesOf bytes_of,
                                           int64_t& keep) {
    keep = slots;
    uint64_t target = 0;
    if (slots < 0 || !ParallelResourceLedger::checked_add(required_bytes, free_floor_bytes, target)) return false;
    for (;;) {
        const uint64_t prefix = bytes_of(keep);
        if (prefix > mapped_bytes) return false;
        const uint64_t released = mapped_bytes - prefix;
        if (free_bytes >= target || released >= target - free_bytes) return true;
        if (keep == 0) return false;
        --keep;
    }
}

struct ParallelSafePoint {
    bool commits_drained = false;
    bool dma_drained = false;
    bool adaptive_drained = false;
    uint64_t outstanding_prefill_loan_bytes = 0;

    bool ready() const {
        return commits_drained && dma_drained && adaptive_drained && outstanding_prefill_loan_bytes == 0;
    }
};

enum class ParallelWarmConversationPolicy { unspecified = 0, preserve, evict_with_replay_record };

enum class ParallelMtpMode { disabled = 0, target_only_after_optional_failure, speculative };

struct ParallelResourceConfig {
    uint32_t logical_capacity = 2;
    uint32_t required_physical_batch_sessions = 2;
    uint64_t idle_grace_ticks = 0;
    bool optional_mtp = true;
};

struct ParallelTransitionResult {
    ParallelOpCode code = ParallelOpCode::ok;
    ParallelState state = ParallelState::solo;
    bool first_stream_intact = true;
    bool second_stream_queued = false;
    uint64_t generation = 0;
    std::string_view reason;

    bool ok() const { return code == ParallelOpCode::ok; }
};

struct ParallelResourceOps {
    std::function<ParallelOpCode(const ParallelActivationBudget&)> shrink_expert_cache_for_activation;
    std::function<ParallelOpCode(uint64_t bytes)> restore_expert_cache_after_release;
    std::function<ParallelOpCode(uint32_t physical_session)> allocate_slot_session;
    std::function<ParallelOpCode(uint32_t physical_session)> bind_slot_session;
    std::function<ParallelOpCode()> allocate_batch_verifier;
    std::function<ParallelOpCode(uint64_t generation)> capture_batch_graphs;
    std::function<ParallelOpCode(uint32_t physical_session)> allocate_slot_draft;
    std::function<ParallelOpCode(uint64_t generation)> publish_parallel_bindings;
    std::function<ParallelOpCode(uint64_t generation)> invalidate_graphs;
    std::function<ParallelOpCode()> detach_bindings;
    std::function<ParallelOpCode(uint32_t physical_session)> preserve_warm_conversation;
    std::function<ParallelOpCode(uint32_t physical_session)> evict_warm_conversation;

    std::function<void(uint32_t physical_session)> free_slot_draft;
    std::function<void()> free_batch_verifier;
    std::function<void(uint32_t physical_session)> free_slot_session;
};

class ParallelResourcePolicyModel {
public:
    explicit ParallelResourcePolicyModel(ParallelResourceConfig config = {}) : config_(config) {
        if (config_.logical_capacity < 1) config_.logical_capacity = 1;
        if (config_.required_physical_batch_sessions < config_.logical_capacity)
            config_.required_physical_batch_sessions = config_.logical_capacity;
    }

    ParallelState state() const { return state_; }
    uint32_t advertised_logical_capacity() const { return config_.logical_capacity; }
    uint32_t physical_session_count() const { return allocated_sessions_; }
    bool first_stream_intact() const { return first_stream_intact_; }
    bool second_stream_queued() const { return second_stream_queued_; }
    bool bindings_published() const { return bindings_published_; }
    bool bindings_valid() const { return bindings_valid_; }
    bool verifier_allocated() const { return verifier_allocated_; }
    bool drafts_allocated() const { return drafts_allocated_ != 0; }
    bool shared_main_freed() const { return shared_main_freed_; }
    bool expert_cache_degraded() const { return expert_cache_degraded_; }
    bool warm_conversation_pending() const { return warm_conversation_pending_; }
    uint64_t generation() const { return generation_; }
    uint64_t graph_generation() const { return graph_generation_; }
    uint64_t idle_ticks() const { return idle_ticks_; }
    ParallelMtpMode mtp_mode() const { return mtp_mode_; }
    int active_streams() const { return active_streams_; }

    ParallelTransitionResult activate_on_second_arrival(const ParallelResourceLedger& ledger,
                                                        const ParallelSafePoint& safe,
                                                        ParallelResourceOps& ops) {
        second_stream_queued_ = true;
        if (state_ != ParallelState::solo) return result(ParallelOpCode::invalid, "activation requires solo state");
        if (!safe.ready()) return result(ParallelOpCode::safe_point_missing, "activation waits for commit/DMA/adaptive drain and no prefill loan");

        ParallelActivationBudget budget;
        std::string error;
        if (!ledger.build_activation_budget(budget, error)) return result(ParallelOpCode::invalid, "invalid ledger");

        state_ = ParallelState::activating;
        activation_owned_bytes_ = budget.required_total_bytes;
        const ParallelOpCode shrink = call_shrink(ops, budget);
        if (parallel_op_failed(shrink)) return rollback_activation(ops, shrink, "expert-cache shrink/broker plan failed", true);
        expert_cache_shrunk_ = budget.required_total_bytes != 0;

        for (uint32_t s = 0; s < config_.required_physical_batch_sessions; ++s) {
            const ParallelOpCode c = call_allocate_session(ops, s);
            if (parallel_op_failed(c)) return rollback_activation(ops, c, "slot session allocation failed", true);
            ++allocated_sessions_;
        }
        for (uint32_t s = 0; s < allocated_sessions_; ++s) {
            const ParallelOpCode c = call_bind_session(ops, s);
            if (parallel_op_failed(c)) return rollback_activation(ops, c, "slot binding failed", true);
        }
        bindings_valid_ = true;

        const ParallelOpCode verifier = call_allocate_verifier(ops);
        if (parallel_op_failed(verifier)) return rollback_activation(ops, verifier, "batch verifier allocation failed", true);
        verifier_allocated_ = true;

        const uint64_t next_generation = generation_ + 1;
        const ParallelOpCode graph = call_capture_graphs(ops, next_generation);
        if (parallel_op_failed(graph)) return rollback_activation(ops, graph, "batch graph capture failed", true);
        graph_generation_ = next_generation;
        graphs_valid_ = true;

        mtp_mode_ = ParallelMtpMode::disabled;
        if (ledger.lazy_mtp_requested) {
            bool draft_failure = false;
            ParallelOpCode draft_code = ParallelOpCode::ok;
            for (uint32_t s = 0; s < allocated_sessions_; ++s) {
                draft_code = call_allocate_draft(ops, s);
                if (parallel_op_failed(draft_code)) {
                    draft_failure = true;
                    break;
                }
                ++drafts_allocated_;
            }
            if (draft_failure) {
                if (!config_.optional_mtp) return rollback_activation(ops, draft_code, "required slot draft allocation failed", true);
                for (uint32_t s = drafts_allocated_; s > 0; --s) call_free_draft(ops, s - 1);
                drafts_allocated_ = 0;
                mtp_mode_ = ParallelMtpMode::target_only_after_optional_failure;
            } else {
                mtp_mode_ = ParallelMtpMode::speculative;
            }
        }

        const ParallelOpCode pub = call_publish(ops, next_generation);
        if (parallel_op_failed(pub)) return rollback_activation(ops, pub, "publication failed", true);

        bindings_published_ = true;
        state_ = ParallelState::parallel;
        generation_ = next_generation;
        second_stream_queued_ = false;
        active_streams_ = 2;
        idle_ticks_ = 0;
        return result(ParallelOpCode::ok, "parallel resources published");
    }

    ParallelTransitionResult cancel_waiting_second(ParallelResourceOps& ops) {
        if (state_ == ParallelState::activating) {
            return rollback_activation(ops, ParallelOpCode::cancelled, "queued second stream cancelled during activation", false);
        }
        if (state_ == ParallelState::solo && second_stream_queued_) {
            second_stream_queued_ = false;
            return result(ParallelOpCode::cancelled, "queued second stream cancelled");
        }
        return result(ParallelOpCode::invalid, "no queued second stream to cancel");
    }

    ParallelTransitionResult finish_one_parallel_stream() {
        if (state_ != ParallelState::parallel || active_streams_ < 2) return result(ParallelOpCode::invalid, "no parallel stream finished");
        active_streams_ = 1;
        warm_conversation_pending_ = true;
        idle_ticks_ = 0;
        return result(ParallelOpCode::ok, "one slot finished and remains warm");
    }

    ParallelTransitionResult second_arrival_during_grace() {
        if (state_ != ParallelState::parallel || active_streams_ != 1) return result(ParallelOpCode::invalid, "burst requires one warm parallel stream");
        active_streams_ = 2;
        warm_conversation_pending_ = false;
        idle_ticks_ = 0;
        second_stream_queued_ = false;
        return result(ParallelOpCode::ok, "burst reused warm parallel resources");
    }

    void tick_idle(bool waiter_present = false) {
        if (state_ == ParallelState::parallel && active_streams_ == 1 && !waiter_present) ++idle_ticks_;
        else idle_ticks_ = 0;
    }

    ParallelTransitionResult reclaim_to_solo(const ParallelResourceLedger& ledger,
                                             const ParallelSafePoint& safe,
                                             ParallelWarmConversationPolicy warm_policy,
                                             ParallelResourceOps& ops) {
        (void) ledger;
        if (state_ != ParallelState::parallel) return result(ParallelOpCode::invalid, "reclaim requires parallel state");
        if (active_streams_ != 1 || second_stream_queued_) return result(ParallelOpCode::not_idle, "parallel work or queued waiter remains");
        if (idle_ticks_ < config_.idle_grace_ticks) return result(ParallelOpCode::idle_grace, "idle grace keeps resources warm");
        if (!safe.ready()) return result(ParallelOpCode::safe_point_missing, "reclaim waits for commit/DMA/adaptive drain and no prefill loan");
        if (warm_policy == ParallelWarmConversationPolicy::unspecified)
            return result(ParallelOpCode::policy_required, "finished slot needs explicit preserve/evict policy");

        state_ = ParallelState::draining;
        const ParallelOpCode warm = apply_warm_policy(warm_policy, ops);
        if (parallel_op_failed(warm)) {
            state_ = ParallelState::parallel;
            return result(warm, "warm conversation policy failed");
        }
        warm_conversation_pending_ = false;

        if (graphs_valid_) {
            const ParallelOpCode inv = call_invalidate_graphs(ops, graph_generation_);
            if (parallel_op_failed(inv)) {
                state_ = ParallelState::parallel;
                return result(inv, "graph invalidation failed before detach");
            }
            graphs_valid_ = false;
        }
        const ParallelOpCode detach = call_detach(ops);
        if (parallel_op_failed(detach)) {
            state_ = ParallelState::parallel;
            return result(detach, "detach failed before private free");
        }
        bindings_valid_ = false;
        bindings_published_ = false;

        for (uint32_t s = drafts_allocated_; s > 0; --s) call_free_draft(ops, s - 1);
        drafts_allocated_ = 0;
        if (verifier_allocated_) {
            call_free_verifier(ops);
            verifier_allocated_ = false;
        }
        for (uint32_t s = allocated_sessions_; s > 0; --s) call_free_session(ops, s - 1);
        allocated_sessions_ = 0;
        mtp_mode_ = ParallelMtpMode::disabled;

        ParallelOpCode restore = ParallelOpCode::ok;
        if (expert_cache_shrunk_) restore = call_restore(ops, activation_owned_bytes_);
        if (parallel_op_failed(restore)) expert_cache_degraded_ = true;
        expert_cache_shrunk_ = false;
        activation_owned_bytes_ = 0;

        state_ = ParallelState::solo;
        active_streams_ = 1;
        ++generation_;
        return result(parallel_op_failed(restore) ? restore : ParallelOpCode::ok,
                      parallel_op_failed(restore) ? "private resources released; expert cache grow failed coherently" : "solo resources reclaimed");
    }

private:
    ParallelTransitionResult result(ParallelOpCode code, std::string_view reason) const {
        return {code, state_, first_stream_intact_, second_stream_queued_, generation_, reason};
    }

    ParallelTransitionResult rollback_activation(ParallelResourceOps& ops, ParallelOpCode code,
                                                 std::string_view reason, bool keep_second_queued) {
        for (uint32_t s = drafts_allocated_; s > 0; --s) call_free_draft(ops, s - 1);
        drafts_allocated_ = 0;
        if (graphs_valid_) {
            (void) call_invalidate_graphs(ops, graph_generation_);
            graphs_valid_ = false;
        }
        if (verifier_allocated_) {
            call_free_verifier(ops);
            verifier_allocated_ = false;
        }
        if (bindings_valid_) {
            (void) call_detach(ops);
            bindings_valid_ = false;
        }
        bindings_published_ = false;
        for (uint32_t s = allocated_sessions_; s > 0; --s) call_free_session(ops, s - 1);
        allocated_sessions_ = 0;
        mtp_mode_ = ParallelMtpMode::disabled;
        if (expert_cache_shrunk_) {
            const ParallelOpCode restored = call_restore(ops, activation_owned_bytes_);
            if (parallel_op_failed(restored)) expert_cache_degraded_ = true;
        }
        expert_cache_shrunk_ = false;
        activation_owned_bytes_ = 0;
        state_ = ParallelState::solo;
        active_streams_ = 1;
        second_stream_queued_ = keep_second_queued;
        return result(code, reason);
    }

    ParallelOpCode apply_warm_policy(ParallelWarmConversationPolicy policy, ParallelResourceOps& ops) {
        const uint32_t finished_slot = allocated_sessions_ == 0 ? 0 : allocated_sessions_ - 1;
        if (policy == ParallelWarmConversationPolicy::preserve) {
            if (!ops.preserve_warm_conversation) return ParallelOpCode::ok;
            return ops.preserve_warm_conversation(finished_slot);
        }
        if (policy == ParallelWarmConversationPolicy::evict_with_replay_record) {
            if (!ops.evict_warm_conversation) return ParallelOpCode::ok;
            return ops.evict_warm_conversation(finished_slot);
        }
        return ParallelOpCode::policy_required;
    }

    static ParallelOpCode normalize(ParallelOpCode c, ParallelOpCode fallback) {
        return c == ParallelOpCode::ok ? ParallelOpCode::ok : fallback;
    }

    ParallelOpCode call_shrink(ParallelResourceOps& ops, const ParallelActivationBudget& budget) {
        if (!ops.shrink_expert_cache_for_activation) return ParallelOpCode::ok;
        return ops.shrink_expert_cache_for_activation(budget);
    }
    ParallelOpCode call_restore(ParallelResourceOps& ops, uint64_t bytes) {
        if (!ops.restore_expert_cache_after_release) return ParallelOpCode::ok;
        return normalize(ops.restore_expert_cache_after_release(bytes), ParallelOpCode::cache_restore_failed);
    }
    ParallelOpCode call_allocate_session(ParallelResourceOps& ops, uint32_t s) {
        if (!ops.allocate_slot_session) return ParallelOpCode::ok;
        const ParallelOpCode c = ops.allocate_slot_session(s);
        return c == ParallelOpCode::map_failed ? c : normalize(c, ParallelOpCode::allocation_failed);
    }
    ParallelOpCode call_bind_session(ParallelResourceOps& ops, uint32_t s) {
        if (!ops.bind_slot_session) return ParallelOpCode::ok;
        return normalize(ops.bind_slot_session(s), ParallelOpCode::binding_failed);
    }
    ParallelOpCode call_allocate_verifier(ParallelResourceOps& ops) {
        if (!ops.allocate_batch_verifier) return ParallelOpCode::ok;
        return normalize(ops.allocate_batch_verifier(), ParallelOpCode::verifier_failed);
    }
    ParallelOpCode call_capture_graphs(ParallelResourceOps& ops, uint64_t gen) {
        if (!ops.capture_batch_graphs) return ParallelOpCode::ok;
        return normalize(ops.capture_batch_graphs(gen), ParallelOpCode::graph_failed);
    }
    ParallelOpCode call_allocate_draft(ParallelResourceOps& ops, uint32_t s) {
        if (!ops.allocate_slot_draft) return ParallelOpCode::ok;
        return normalize(ops.allocate_slot_draft(s), ParallelOpCode::draft_failed);
    }
    ParallelOpCode call_publish(ParallelResourceOps& ops, uint64_t gen) {
        if (!ops.publish_parallel_bindings) return ParallelOpCode::ok;
        return normalize(ops.publish_parallel_bindings(gen), ParallelOpCode::publication_failed);
    }
    ParallelOpCode call_invalidate_graphs(ParallelResourceOps& ops, uint64_t gen) {
        if (!ops.invalidate_graphs) return ParallelOpCode::ok;
        return normalize(ops.invalidate_graphs(gen), ParallelOpCode::graph_failed);
    }
    ParallelOpCode call_detach(ParallelResourceOps& ops) {
        if (!ops.detach_bindings) return ParallelOpCode::ok;
        return normalize(ops.detach_bindings(), ParallelOpCode::binding_failed);
    }
    void call_free_draft(ParallelResourceOps& ops, uint32_t s) {
        if (ops.free_slot_draft) ops.free_slot_draft(s);
    }
    void call_free_verifier(ParallelResourceOps& ops) {
        if (ops.free_batch_verifier) ops.free_batch_verifier();
    }
    void call_free_session(ParallelResourceOps& ops, uint32_t s) {
        if (ops.free_slot_session) ops.free_slot_session(s);
    }

    ParallelResourceConfig config_{};
    ParallelState state_ = ParallelState::solo;
    uint32_t allocated_sessions_ = 0;
    uint32_t drafts_allocated_ = 0;
    bool verifier_allocated_ = false;
    bool bindings_valid_ = false;
    bool bindings_published_ = false;
    bool graphs_valid_ = false;
    bool shared_main_freed_ = false;
    bool first_stream_intact_ = true;
    bool second_stream_queued_ = false;
    bool expert_cache_shrunk_ = false;
    bool expert_cache_degraded_ = false;
    bool warm_conversation_pending_ = false;
    uint64_t activation_owned_bytes_ = 0;
    uint64_t generation_ = 0;
    uint64_t graph_generation_ = 0;
    uint64_t idle_ticks_ = 0;
    int active_streams_ = 1;
    ParallelMtpMode mtp_mode_ = ParallelMtpMode::disabled;
};

} // namespace strata::core
