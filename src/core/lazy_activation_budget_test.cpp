#include "strata/core/parallel_resource_policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;

void check(bool ok, const char* message) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "lazy_activation_budget_test: failed: %s\n", message);
        std::exit(1);
    }
}

ParallelResourceLedger activation_ledger() {
    ParallelResourceLedger ledger;
    ledger.shared_main_pool_bytes = 1000;
    ledger.shared_weight_bytes = 2000;
    ledger.safety_reserve_bytes = 500;
    ledger.private_slot_session_bytes = 100;
    ledger.slot_indexer_brokered_bytes = 300;
    ledger.batch_verifier_graph_bytes = 50;
    ledger.slot_draft_private_bytes = 10;
    ledger.transition_temporary_bytes = 5;
    ledger.lazy_mtp_requested = true;
    return ledger;
}

IndexerBudgetBroker make_broker(uint64_t budget) {
    IndexerBudgetBroker broker;
    check(IndexerBudgetBroker::create({budget, 100, 50, 64, 128}, {200, 0, 1500, 0}, broker),
          "broker fixture is valid");
    return broker;
}

void prices_all_classes_once_and_reserves_external_bytes() {
    auto broker = make_broker(2200); // 350 free above broker reserve/transient floors.
    ParallelBrokerActivationPlan plan;
    std::string error;
    const std::vector<IndexerBudgetBroker::ExpertVictimCandidate> victims = {
        {1, 300}, {2, 300}, {3, 900}};
    check(build_parallel_broker_activation_plan(activation_ledger(), broker, victims, plan, error),
          "combined activation budget plans with safe victims");
    check(plan.budget.required_runtime_bytes == 275 && plan.budget.required_brokered_indexer_bytes == 600,
          "runtime and broker-owned indexer charges remain separate");
    check(plan.budget.required_total_bytes == 875 && plan.budget.safety_reserve_bytes == 500,
          "activation total includes each charge once and reports rather than duplicates reserve");
    check(plan.broker_plan.required_growth_bytes == 600 &&
              plan.broker_plan.external_activation_bytes == 275 && plan.broker_plan.recovered_bytes == 600,
          "broker exchanges indexer and external activation bytes together");
    check(broker.apply_growth_result(plan.broker_plan,
              {IndexerBudgetBroker::CommitResult::committed, 600, {1, 2}, 600, 0}),
          "broker commits activation reservation");
    const auto live = broker.telemetry();
    check(live.external_activation_bytes == 275 && live.private_indexer_committed_bytes == 800,
          "broker tracks private runtime reservation without charging it as indexer");
    check(broker.release_external_activation(275), "private runtime reservation releases after teardown");
    check(broker.telemetry().external_activation_bytes == 0, "release clears activation reservation");
}

void no_hidden_reserve_is_treated_as_solo_capacity() {
    auto broker = make_broker(1850); // free headroom is zero after floor and mapped commitments.
    ParallelBrokerActivationPlan plan;
    std::string error;
    check(!build_parallel_broker_activation_plan(activation_ledger(), broker, {}, plan, error),
          "activation is refused without capacity or victims");
    check(plan.budget.required_total_bytes == 875 && plan.budget.safety_reserve_bytes == 500,
          "shortage does not reclassify reserve as recovered solo capacity");
    check(plan.broker_plan.outcome == IndexerBudgetBroker::Outcome::shortage,
          "combined plan reports broker shortage");
}

void partial_broker_result_keeps_waiter_queued_and_first_stream_safe() {
    auto broker = make_broker(2200);
    ParallelBrokerActivationPlan plan;
    std::string error;
    check(build_parallel_broker_activation_plan(activation_ledger(), broker,
              {{1, 300}, {2, 300}, {3, 900}}, plan, error), "partial-result plan is ready");
    check(broker.apply_growth_result(plan.broker_plan,
              {IndexerBudgetBroker::CommitResult::indexer_publication_failed, 300, {1}, 0, 0}),
          "broker records actual partial eviction as coherent cache miss");
    check(broker.telemetry().private_indexer_committed_bytes == 200 &&
              broker.telemetry().external_activation_bytes == 0 &&
              broker.telemetry().expert_mapped_bytes == 1200,
          "failed broker publication reserves no activation memory and does not invent commitment");

    ParallelResourcePolicyModel policy;
    ParallelResourceOps ops;
    ops.shrink_expert_cache_for_activation = [](const ParallelActivationBudget&) {
        return ParallelOpCode::cache_shortage;
    };
    const ParallelTransitionResult result = policy.activate_on_second_arrival(
        activation_ledger(), {true, true, true, 0}, ops);
    check(result.code == ParallelOpCode::cache_shortage && policy.first_stream_intact() &&
              policy.second_stream_queued() && policy.state() == ParallelState::solo,
          "rejected/partial broker activation preserves first stream and queued waiter");
}
} // namespace

int main() {
    prices_all_classes_once_and_reserves_external_bytes();
    no_hidden_reserve_is_treated_as_solo_capacity();
    partial_broker_result_keeps_waiter_queued_and_first_stream_safe();
    std::printf("lazy_activation_budget_test: %d checks passed\n", checks);
    return 0;
}
