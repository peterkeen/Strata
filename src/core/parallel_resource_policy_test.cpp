#include "strata/core/parallel_resource_policy.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "parallel_resource_policy_test: failed: %s\n", what);
        std::exit(1);
    }
}

void check_code(ParallelOpCode got, ParallelOpCode want, const char* what) {
    ++checks;
    if (got != want) {
        std::fprintf(stderr, "parallel_resource_policy_test: failed: %s (got %.*s want %.*s)\n", what,
                     static_cast<int>(parallel_op_code_name(got).size()), parallel_op_code_name(got).data(),
                     static_cast<int>(parallel_op_code_name(want).size()), parallel_op_code_name(want).data());
        std::exit(1);
    }
}

ParallelResourceLedger ledger(bool mtp = false) {
    ParallelResourceLedger l;
    l.logical_capacity = 2;
    l.required_physical_batch_sessions = 2;
    l.shared_main_pool_bytes = 1000;
    l.shared_weight_bytes = 2000;
    l.safety_reserve_bytes = 500;
    l.expert_cache_mapped_bytes = 5000;
    l.private_slot_session_bytes = 100;
    l.slot_indexer_brokered_bytes = 300;
    l.batch_verifier_graph_bytes = 50;
    l.slot_draft_private_bytes = 10;
    l.transition_temporary_bytes = 5;
    l.lazy_mtp_requested = mtp;
    return l;
}

ParallelSafePoint safe_point() { return {true, true, true, 0}; }
ParallelSafePoint unsafe_prefill_loan() { return {true, true, true, 64}; }
ParallelSafePoint unsafe_dma() { return {true, false, true, 0}; }

enum class Action {
    none,
    shrink,
    restore,
    allocate_session,
    bind_session,
    verifier,
    graph,
    draft,
    publish,
    invalidate,
    detach,
    preserve,
    evict,
};

struct FakeOps {
    Action fail_action = Action::none;
    uint32_t fail_session = 999;
    ParallelOpCode fail_code = ParallelOpCode::allocation_failed;
    bool require_publish_after_graph = false;
    bool require_detach_after_invalidate = false;

    int shrink_calls = 0;
    int restore_calls = 0;
    int allocate_calls = 0;
    int bind_calls = 0;
    int verifier_calls = 0;
    int graph_calls = 0;
    int draft_calls = 0;
    int publish_calls = 0;
    int invalidate_calls = 0;
    int detach_calls = 0;
    int preserve_calls = 0;
    int evict_calls = 0;
    int free_session_calls = 0;
    int free_draft_calls = 0;
    int free_verifier_calls = 0;

    bool cache_shrunk = false;
    bool graph_captured = false;
    bool detached_after_invalidate = false;
    uint64_t last_budget_total = 0;
    uint64_t restored_bytes = 0;
    std::vector<std::string> events;

    bool fail(Action a, uint32_t session = 999) const {
        return fail_action == a && (fail_session == 999 || fail_session == session);
    }

    ParallelResourceOps callbacks() {
        ParallelResourceOps ops;
        ops.shrink_expert_cache_for_activation = [this](const ParallelActivationBudget& budget) {
            events.push_back("shrink");
            ++shrink_calls;
            last_budget_total = budget.required_total_bytes;
            if (fail(Action::shrink)) return fail_code;
            cache_shrunk = true;
            return ParallelOpCode::ok;
        };
        ops.restore_expert_cache_after_release = [this](uint64_t bytes) {
            events.push_back("restore");
            ++restore_calls;
            restored_bytes += bytes;
            if (fail(Action::restore)) return fail_code;
            cache_shrunk = false;
            return ParallelOpCode::ok;
        };
        ops.allocate_slot_session = [this](uint32_t s) {
            events.push_back("alloc" + std::to_string(s));
            ++allocate_calls;
            if (fail(Action::allocate_session, s)) return fail_code;
            return ParallelOpCode::ok;
        };
        ops.bind_slot_session = [this](uint32_t s) {
            events.push_back("bind" + std::to_string(s));
            ++bind_calls;
            if (fail(Action::bind_session, s)) return fail_code;
            return ParallelOpCode::ok;
        };
        ops.allocate_batch_verifier = [this]() {
            events.push_back("verifier");
            ++verifier_calls;
            if (fail(Action::verifier)) return fail_code;
            return ParallelOpCode::ok;
        };
        ops.capture_batch_graphs = [this](uint64_t generation) {
            events.push_back("graph" + std::to_string(generation));
            ++graph_calls;
            if (fail(Action::graph)) return fail_code;
            graph_captured = true;
            return ParallelOpCode::ok;
        };
        ops.allocate_slot_draft = [this](uint32_t s) {
            events.push_back("draft" + std::to_string(s));
            ++draft_calls;
            if (fail(Action::draft, s)) return fail_code;
            return ParallelOpCode::ok;
        };
        ops.publish_parallel_bindings = [this](uint64_t generation) {
            events.push_back("publish" + std::to_string(generation));
            ++publish_calls;
            if (require_publish_after_graph) check(graph_captured, "publish happens after graph capture");
            if (fail(Action::publish)) return fail_code;
            return ParallelOpCode::ok;
        };
        ops.invalidate_graphs = [this](uint64_t generation) {
            events.push_back("invalidate" + std::to_string(generation));
            ++invalidate_calls;
            if (fail(Action::invalidate)) return fail_code;
            graph_captured = false;
            return ParallelOpCode::ok;
        };
        ops.detach_bindings = [this]() {
            events.push_back("detach");
            ++detach_calls;
            if (require_detach_after_invalidate) check(invalidate_calls > 0, "detach happens after graph invalidation");
            if (fail(Action::detach)) return fail_code;
            detached_after_invalidate = invalidate_calls > 0;
            return ParallelOpCode::ok;
        };
        ops.preserve_warm_conversation = [this](uint32_t s) {
            events.push_back("preserve" + std::to_string(s));
            ++preserve_calls;
            if (fail(Action::preserve, s)) return fail_code;
            return ParallelOpCode::ok;
        };
        ops.evict_warm_conversation = [this](uint32_t s) {
            events.push_back("evict" + std::to_string(s));
            ++evict_calls;
            if (fail(Action::evict, s)) return fail_code;
            return ParallelOpCode::ok;
        };
        ops.free_slot_draft = [this](uint32_t s) {
            events.push_back("free_draft" + std::to_string(s));
            ++free_draft_calls;
        };
        ops.free_batch_verifier = [this]() {
            events.push_back("free_verifier");
            ++free_verifier_calls;
        };
        ops.free_slot_session = [this](uint32_t s) {
            events.push_back("free_session" + std::to_string(s));
            ++free_session_calls;
        };
        return ops;
    }
};

void ledger_separates_resource_classes_and_brokered_indexer() {
    ParallelActivationBudget budget;
    std::string error;
    const ParallelResourceLedger l = ledger(true);
    check(l.build_activation_budget(budget, error), "ledger builds activation budget");
    check(budget.baseline_shared_bytes == 3000, "shared main pools and weights are baseline only");
    check(budget.required_runtime_bytes == 275, "runtime bytes exclude brokered indexer but include drafts");
    check(budget.required_brokered_indexer_bytes == 600, "brokered indexer bytes are separated for str-a2y");
    check(budget.required_total_bytes == 875, "total activation capacity includes runtime and brokered indexer once");
    check(budget.safety_reserve_bytes == 500, "safety reserve is reported but not spendable");
    bool saw_indexer = false, saw_draft = false, saw_shared = false;
    for (const ParallelResourceLine& line : budget.lines) {
        if (line.klass == ParallelResourceClass::brokered_slot_indexer)
            saw_indexer = line.broker_owned && !line.baseline && line.bytes == 600;
        if (line.klass == ParallelResourceClass::slot_draft_rings_and_buffers)
            saw_draft = line.bytes == 20 && !line.baseline;
        if (line.klass == ParallelResourceClass::shared_main_pools)
            saw_shared = line.baseline && line.bytes == 1000;
    }
    check(saw_indexer && saw_draft && saw_shared, "ledger lines distinguish indexer, draft and shared pools");

    ParallelActivationBudget no_mtp;
    check(ledger(false).build_activation_budget(no_mtp, error), "ledger builds without MTP");
    check(no_mtp.required_runtime_bytes == 255 && no_mtp.required_total_bytes == 855,
          "disabled lazy MTP removes private draft bytes only");
}

void successful_cold_activation_publishes_only_after_valid_bindings() {
    ParallelResourcePolicyModel model({2, 2, 2, true});
    FakeOps fake;
    fake.require_publish_after_graph = true;
    ParallelResourceOps ops = fake.callbacks();
    const ParallelTransitionResult r = model.activate_on_second_arrival(ledger(true), safe_point(), ops);
    check_code(r.code, ParallelOpCode::ok, "cold first overlap activates");
    check(model.state() == ParallelState::parallel, "model enters parallel state");
    check(model.advertised_logical_capacity() == 2, "logical capacity stays advertised as two");
    check(model.physical_session_count() == 2, "two physical batch sessions are allocated for two clients");
    check(model.bindings_published() && model.bindings_valid(), "bindings publish only after valid allocation/binding");
    check(model.verifier_allocated() && model.drafts_allocated(), "verifier and optional draft resources allocated");
    check(model.mtp_mode() == ParallelMtpMode::speculative, "lazy MTP uses private resources after shared weights");
    check(fake.shrink_calls == 1 && fake.last_budget_total == 875, "activation uses full budget before allocation");
    check(fake.publish_calls == 1 && fake.graph_calls == 1 && fake.bind_calls == 2, "publish follows bind and graph capture");
    check(!model.shared_main_freed(), "activation never releases shared main KV");
}

void partial_mapping_failure_keeps_first_stream_and_waiter_safe() {
    ParallelResourcePolicyModel model({2, 2, 0, true});
    FakeOps fake;
    fake.fail_action = Action::allocate_session;
    fake.fail_session = 1;
    fake.fail_code = ParallelOpCode::map_failed;
    ParallelResourceOps ops = fake.callbacks();
    const ParallelTransitionResult r = model.activate_on_second_arrival(ledger(true), safe_point(), ops);
    check_code(r.code, ParallelOpCode::map_failed, "partial slot mapping failure is reported");
    check(model.state() == ParallelState::solo, "partial failure returns to solo");
    check(model.first_stream_intact() && r.first_stream_intact, "first stream remains intact");
    check(model.second_stream_queued() && r.second_stream_queued, "second stream remains queued");
    check(model.advertised_logical_capacity() == 2, "logical advertised capacity remains consistent after failure");
    check(model.physical_session_count() == 0 && !model.bindings_published(), "no broken physical slot is advertised");
    check(fake.free_session_calls == 1 && fake.restore_calls == 1, "only owned private session and cache budget are released");
    check(!model.shared_main_freed(), "rollback never frees shared main KV");
}

void publication_failure_releases_private_resources_without_advertising_slot() {
    ParallelResourcePolicyModel model({2, 2, 0, true});
    FakeOps fake;
    fake.fail_action = Action::publish;
    fake.fail_code = ParallelOpCode::publication_failed;
    ParallelResourceOps ops = fake.callbacks();
    const ParallelTransitionResult r = model.activate_on_second_arrival(ledger(true), safe_point(), ops);
    check_code(r.code, ParallelOpCode::publication_failed, "publication failure is explicit");
    check(model.state() == ParallelState::solo && !model.bindings_published(), "failed publication leaves no advertised slot");
    check(model.second_stream_queued(), "second stream remains queued after publication failure");
    check(fake.free_draft_calls == 2 && fake.free_verifier_calls == 1 && fake.free_session_calls == 2,
          "publication rollback frees only owned private resources");
    check(fake.restore_calls == 1, "publication rollback returns activation budget to cache broker");
}

void safe_points_and_prefill_loan_gate_activation_and_reclaim() {
    {
        ParallelResourcePolicyModel model({2, 2, 0, true});
        FakeOps fake;
        ParallelResourceOps ops = fake.callbacks();
        const ParallelTransitionResult r = model.activate_on_second_arrival(ledger(), unsafe_dma(), ops);
        check_code(r.code, ParallelOpCode::safe_point_missing, "activation waits for DMA/adaptive/commit drain");
        check(model.state() == ParallelState::solo && model.second_stream_queued(), "unsafe activation keeps waiter queued");
        check(fake.shrink_calls == 0 && fake.allocate_calls == 0, "unsafe activation performs no allocations");
    }
    {
        ParallelResourcePolicyModel model({2, 2, 0, true});
        FakeOps fake;
        ParallelResourceOps ops = fake.callbacks();
        check_code(model.activate_on_second_arrival(ledger(), safe_point(), ops).code, ParallelOpCode::ok,
                   "activation before prefill-loan reclaim fixture succeeds");
        check_code(model.finish_one_parallel_stream().code, ParallelOpCode::ok, "one stream finishes before reclaim");
        const ParallelTransitionResult r = model.reclaim_to_solo(ledger(), unsafe_prefill_loan(),
                                                                 ParallelWarmConversationPolicy::preserve, ops);
        check_code(r.code, ParallelOpCode::safe_point_missing, "reclaim refuses outstanding prefill loan");
        check(model.state() == ParallelState::parallel && model.physical_session_count() == 2,
              "prefill-loan reclaim does not detach or free resources");
    }
}

void graph_generation_must_invalidate_before_detach_and_free() {
    ParallelResourcePolicyModel model({2, 2, 0, false});
    FakeOps fake;
    fake.require_detach_after_invalidate = true;
    ParallelResourceOps ops = fake.callbacks();
    check_code(model.activate_on_second_arrival(ledger(), safe_point(), ops).code, ParallelOpCode::ok,
               "activation for graph invalidation fixture succeeds");
    check(model.graph_generation() == 1, "first activation graph generation is one");
    check_code(model.finish_one_parallel_stream().code, ParallelOpCode::ok, "finish one before reclaim");
    const ParallelTransitionResult r = model.reclaim_to_solo(ledger(), safe_point(),
                                                             ParallelWarmConversationPolicy::preserve, ops);
    check_code(r.code, ParallelOpCode::ok, "reclaim succeeds after graph invalidation");
    check(fake.invalidate_calls == 1 && fake.detach_calls == 1 && fake.detached_after_invalidate,
          "graphs are invalidated before detaching bindings");
    check(model.state() == ParallelState::solo && model.physical_session_count() == 0, "private resources freed after detach");
    check(model.generation() == 2, "reclaim advances allocation generation");

    ParallelResourcePolicyModel fail_model({2, 2, 0, false});
    FakeOps failing;
    ParallelResourceOps failing_ops = failing.callbacks();
    check_code(fail_model.activate_on_second_arrival(ledger(), safe_point(), failing_ops).code, ParallelOpCode::ok,
               "activation for graph failure fixture succeeds");
    check_code(fail_model.finish_one_parallel_stream().code, ParallelOpCode::ok, "finish one before failed reclaim");
    failing.fail_action = Action::invalidate;
    failing.fail_code = ParallelOpCode::graph_failed;
    const ParallelTransitionResult failed = fail_model.reclaim_to_solo(ledger(), safe_point(),
                                                                       ParallelWarmConversationPolicy::preserve,
                                                                       failing_ops);
    check_code(failed.code, ParallelOpCode::graph_failed, "graph invalidation failure blocks detach");
    check(fail_model.state() == ParallelState::parallel && failing.detach_calls == 0 && failing.free_session_calls == 0,
          "stale graph risk prevents pointer-freeing detach");
}

void warm_conversation_policy_is_required_and_tested() {
    ParallelResourcePolicyModel model({2, 2, 0, false});
    FakeOps fake;
    ParallelResourceOps ops = fake.callbacks();
    check_code(model.activate_on_second_arrival(ledger(), safe_point(), ops).code, ParallelOpCode::ok,
               "activation for warm-policy fixture succeeds");
    check_code(model.finish_one_parallel_stream().code, ParallelOpCode::ok, "finished slot becomes warm conversation");
    check(model.warm_conversation_pending(), "finished slot is retained pending explicit policy");
    const ParallelTransitionResult missing = model.reclaim_to_solo(ledger(), safe_point(),
                                                                   ParallelWarmConversationPolicy::unspecified, ops);
    check_code(missing.code, ParallelOpCode::policy_required, "active=false is not permission to discard conversation");
    check(model.state() == ParallelState::parallel && fake.free_session_calls == 0, "missing warm policy frees nothing");
    const ParallelTransitionResult evicted = model.reclaim_to_solo(ledger(), safe_point(),
                                                                   ParallelWarmConversationPolicy::evict_with_replay_record,
                                                                   ops);
    check_code(evicted.code, ParallelOpCode::ok, "explicit replay eviction policy allows reclaim");
    check(fake.evict_calls == 1 && fake.preserve_calls == 0, "eviction policy is observable and tested");
    check(model.state() == ParallelState::solo && !model.warm_conversation_pending(), "warm conversation policy resolved");
}

void idle_grace_bursts_and_reactivation_are_modeled() {
    ParallelResourcePolicyModel model({2, 2, 2, false});
    FakeOps fake;
    ParallelResourceOps ops = fake.callbacks();
    check_code(model.activate_on_second_arrival(ledger(), safe_point(), ops).code, ParallelOpCode::ok,
               "first cold activation succeeds");
    check_code(model.finish_one_parallel_stream().code, ParallelOpCode::ok, "one stream finishes");
    model.tick_idle(false);
    const ParallelTransitionResult too_soon = model.reclaim_to_solo(ledger(), safe_point(),
                                                                    ParallelWarmConversationPolicy::preserve, ops);
    check_code(too_soon.code, ParallelOpCode::idle_grace, "idle grace avoids allocation/cache churn");
    check(model.state() == ParallelState::parallel && fake.free_session_calls == 0, "grace keeps resources allocated");
    check_code(model.second_arrival_during_grace().code, ParallelOpCode::ok, "burst reuses allocated parallel resources");
    check(model.active_streams() == 2 && fake.allocate_calls == 2, "burst did not allocate a second time");
    check_code(model.finish_one_parallel_stream().code, ParallelOpCode::ok, "one stream finishes again");
    model.tick_idle(false);
    model.tick_idle(false);
    check_code(model.reclaim_to_solo(ledger(), safe_point(), ParallelWarmConversationPolicy::preserve, ops).code,
               ParallelOpCode::ok, "return to solo after idle grace succeeds");
    check(model.state() == ParallelState::solo && model.physical_session_count() == 0, "solo resources are reclaimed");
    check_code(model.activate_on_second_arrival(ledger(), safe_point(), ops).code, ParallelOpCode::ok,
               "reactivation after solo succeeds");
    check(model.state() == ParallelState::parallel && model.generation() == 3, "reactivation gets a fresh generation");
}

void cancellation_and_cache_grow_failure_are_safe() {
    {
        ParallelResourcePolicyModel model({2, 2, 0, false});
        FakeOps fake;
        ParallelResourceOps ops = fake.callbacks();
        check_code(model.activate_on_second_arrival(ledger(), unsafe_dma(), ops).code,
                   ParallelOpCode::safe_point_missing, "unsafe activation queues waiter for cancellation fixture");
        check(model.second_stream_queued(), "waiter is queued before cancellation");
        check_code(model.cancel_waiting_second(ops).code, ParallelOpCode::cancelled, "queued waiter cancellation is modeled");
        check(!model.second_stream_queued() && model.state() == ParallelState::solo, "cancellation returns to clean solo");
    }
    {
        ParallelResourcePolicyModel model({2, 2, 0, false});
        FakeOps fake;
        ParallelResourceOps ops = fake.callbacks();
        check_code(model.activate_on_second_arrival(ledger(), safe_point(), ops).code, ParallelOpCode::ok,
                   "activation for cache-grow failure fixture succeeds");
        check_code(model.finish_one_parallel_stream().code, ParallelOpCode::ok, "one stream finishes before grow failure");
        fake.fail_action = Action::restore;
        fake.fail_code = ParallelOpCode::cache_restore_failed;
        const ParallelTransitionResult r = model.reclaim_to_solo(ledger(), safe_point(),
                                                                 ParallelWarmConversationPolicy::preserve, ops);
        check_code(r.code, ParallelOpCode::cache_restore_failed, "cache grow failure is reported after private release");
        check(model.state() == ParallelState::solo && model.physical_session_count() == 0, "private slot resources are still released");
        check(model.expert_cache_degraded(), "coherent cache miss/degraded state is retained for follow-up refill");
    }
}

void optional_lazy_mtp_degrades_to_target_only_or_rolls_back_when_required() {
    {
        ParallelResourcePolicyModel model({2, 2, 0, true});
        FakeOps fake;
        fake.fail_action = Action::draft;
        fake.fail_session = 1;
        fake.fail_code = ParallelOpCode::draft_failed;
        ParallelResourceOps ops = fake.callbacks();
        const ParallelTransitionResult r = model.activate_on_second_arrival(ledger(true), safe_point(), ops);
        check_code(r.code, ParallelOpCode::ok, "optional lazy MTP draft failure still activates target-only batch");
        check(model.state() == ParallelState::parallel, "target-only fallback keeps lazy sessions usable");
        check(model.mtp_mode() == ParallelMtpMode::target_only_after_optional_failure, "MTP fallback is explicit");
        check(!model.drafts_allocated() && fake.free_draft_calls == 1, "partially allocated draft state is released");
    }
    {
        ParallelResourcePolicyModel model({2, 2, 0, false});
        FakeOps fake;
        fake.fail_action = Action::draft;
        fake.fail_session = 0;
        fake.fail_code = ParallelOpCode::draft_failed;
        ParallelResourceOps ops = fake.callbacks();
        const ParallelTransitionResult r = model.activate_on_second_arrival(ledger(true), safe_point(), ops);
        check_code(r.code, ParallelOpCode::draft_failed, "required draft failure rolls activation back");
        check(model.state() == ParallelState::solo && model.second_stream_queued(), "required MTP failure keeps first stream and waiter safe");
        check(fake.free_session_calls == 2 && fake.restore_calls == 1, "required MTP rollback releases session-owned resources");
    }
}

void cache_shortage_never_hides_solo_capacity_or_advertises_broken_slot() {
    ParallelResourcePolicyModel model({2, 2, 0, true});
    FakeOps fake;
    fake.fail_action = Action::shrink;
    fake.fail_code = ParallelOpCode::cache_shortage;
    ParallelResourceOps ops = fake.callbacks();
    const ParallelTransitionResult r = model.activate_on_second_arrival(ledger(), safe_point(), ops);
    check_code(r.code, ParallelOpCode::cache_shortage, "activation refuses hidden permanent reserve/cache shortage");
    check(model.state() == ParallelState::solo && model.physical_session_count() == 0, "cache shortage allocates no slots");
    check(model.first_stream_intact() && model.second_stream_queued(), "shortage keeps first stream and waiter semantics");
    check(fake.allocate_calls == 0 && fake.publish_calls == 0, "shortage does not publish broken capacity");
}

} // namespace

int main() {
    ledger_separates_resource_classes_and_brokered_indexer();
    successful_cold_activation_publishes_only_after_valid_bindings();
    partial_mapping_failure_keeps_first_stream_and_waiter_safe();
    publication_failure_releases_private_resources_without_advertising_slot();
    safe_points_and_prefill_loan_gate_activation_and_reclaim();
    graph_generation_must_invalidate_before_detach_and_free();
    warm_conversation_policy_is_required_and_tested();
    idle_grace_bursts_and_reactivation_are_modeled();
    cancellation_and_cache_grow_failure_are_safe();
    optional_lazy_mtp_degrades_to_target_only_or_rolls_back_when_required();
    cache_shortage_never_hides_solo_capacity_or_advertises_broken_slot();
    std::printf("parallel_resource_policy_test: %d checks passed\n", checks);
    return 0;
}
