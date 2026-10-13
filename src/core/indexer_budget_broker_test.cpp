#include "strata/core/indexer_budget_broker.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

using strata::core::IndexerBudgetBroker;

namespace {
size_t checks = 0;

void check(bool ok, const char* label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::exit(1);
    }
}

IndexerBudgetBroker make_broker(uint64_t total = 1000, uint64_t reserve = 100, uint64_t transient = 50,
                                uint64_t private_bytes = 200, uint64_t pooled = 0, uint64_t expert = 500,
                                uint64_t evicted = 0, uint64_t low = 128, uint64_t target = 256) {
    IndexerBudgetBroker broker;
    check(IndexerBudgetBroker::create({total, reserve, transient, low, target},
                                      {private_bytes, pooled, expert, evicted}, broker),
          "broker create succeeds");
    return broker;
}

void no_eviction_when_headroom_suffices() {
    auto broker = make_broker(); // headroom 150.
    const auto before = broker.telemetry();
    const auto plan = broker.plan_growth({80, 20, 500}, {{1, 400}, {2, 400}});
    check(plan.outcome == IndexerBudgetBroker::Outcome::ready, "headroom growth is ready");
    check(plan.required_growth_bytes == 100 && plan.optional_ignored_bytes == 500, "optional growth is ignored");
    check(plan.victims.empty() && plan.recovered_bytes == 0, "headroom growth selects no expert victims");
    check(broker.telemetry().expert_mapped_bytes == before.expert_mapped_bytes, "planning does not mutate experts");
    check(broker.apply_growth_result(plan, {IndexerBudgetBroker::CommitResult::committed, 0, {}, 80, 20}),
          "headroom growth confirm succeeds");
    const auto after = broker.telemetry();
    check(after.private_indexer_committed_bytes == 280 && after.pooled_slab_committed_bytes == 20,
          "confirmed headroom growth updates actual commitments");
    check(after.expert_mapped_bytes == 500 && after.evicted_recoverable_bytes == 0,
          "headroom growth leaves expert cache untouched");
    check(after.unused_bytes == 50, "telemetry reports remaining unused headroom above floors");
}

void minimal_victim_selection_and_exclusions() {
    auto broker = make_broker(); // headroom 150, need 250 more for a 400-byte request.
    std::vector<IndexerBudgetBroker::ExpertVictimCandidate> victims = {
        {10, 90, true, false, false, false, false},  // in-flight: refused
        {11, 120, false, false, false, false, false},
        {12, 80, false, true, false, false, false},  // prefill loan: refused
        {13, 140, false, false, false, false, false},
        {14, 500, false, false, false, false, false}, // must not be selected after minimum is reached
    };
    const auto plan = broker.plan_growth({400, 0, 1000}, victims);
    check(plan.outcome == IndexerBudgetBroker::Outcome::ready, "victim-backed growth is ready");
    check(plan.free_headroom_bytes == 150 && plan.recovered_bytes == 260, "plan recovers enough bytes");
    check(plan.victims.size() == 2 && plan.victims[0].id == 11 && plan.victims[1].id == 13,
          "victim selection is deterministic caller order and minimum necessary");
    check(broker.apply_growth_result(plan, {IndexerBudgetBroker::CommitResult::committed, 260, {11, 13}, 400, 0}),
          "victim-backed growth confirm succeeds");
    const auto t = broker.telemetry();
    check(t.expert_mapped_bytes == 240 && t.evicted_recoverable_bytes == 260,
          "confirmed evictions become coherent misses and recoverable capacity");
    check(t.private_indexer_committed_bytes == 600 && t.unused_bytes == 10, "actual commitment consumes recovered headroom");
}

void shortage_and_unsupported_are_explicit() {
    auto broker = make_broker();
    const auto shortage = broker.plan_growth({500, 0, 0}, {
        {1, 100, false, false, true, false, false},
        {2, 150, false, false, false, true, false},
        {3, 40, false, false, false, false, false},
    });
    check(shortage.outcome == IndexerBudgetBroker::Outcome::shortage, "excluded ordinary victims produce shortage");
    check(shortage.recovered_bytes == 40 && shortage.shortfall_bytes == 310, "shortage reports minimal shortfall");
    check(!broker.apply_growth_result(shortage, {IndexerBudgetBroker::CommitResult::committed, 40, {3}, 500, 0}),
          "shortage plan cannot be partially committed");
    check(broker.telemetry().private_indexer_committed_bytes == 200, "shortage plan does not mutate state");

    const auto unsupported = broker.plan_growth({500, 0, 0}, {
        {1, 40, false, false, false, false, false},
        {2, 500, false, false, false, false, true},
    });
    check(unsupported.outcome == IndexerBudgetBroker::Outcome::unsupported, "unsupported multi-GPU victim is explicit");
    check(unsupported.shortfall_bytes == 310, "unsupported result still reports shortfall");
}

void partial_eviction_and_publication_failure_keeps_misses_coherent() {
    auto broker = make_broker();
    const auto plan = broker.plan_growth({420, 0, 0}, {{1, 160}, {2, 160}, {3, 160}});
    check(plan.outcome == IndexerBudgetBroker::Outcome::ready && plan.victims.size() == 2,
          "partial-failure fixture selects two victims");
    check(broker.apply_growth_result(plan,
                                     {IndexerBudgetBroker::CommitResult::indexer_publication_failed, 160, {1}, 0, 0}),
          "recoverable indexer publication failure accepts actual partial eviction");
    const auto t = broker.telemetry();
    check(!t.poisoned, "recoverable publication failure does not poison broker");
    check(t.private_indexer_committed_bytes == 200 && t.expert_mapped_bytes == 340,
          "failed publication does not add indexer commitment but leaves reported victim evicted");
    check(t.evicted_recoverable_bytes == 160 && t.unused_bytes == 310,
          "failed publication yields coherent miss capacity instead of stale residency");
}

void restore_failure_and_uncertainty_poison() {
    {
        auto broker = make_broker();
        const auto plan = broker.plan_growth({420, 0, 0}, {{1, 300}});
        check(plan.outcome == IndexerBudgetBroker::Outcome::ready, "restore-failure fixture plans growth");
        check(broker.apply_growth_result(plan, {IndexerBudgetBroker::CommitResult::restore_failed, 300, {1}, 0, 0}),
              "restore failure report accepted");
        check(broker.telemetry().poisoned && broker.telemetry().expert_mapped_bytes == 200,
              "restore failure poisons after preserving coherent evicted state");
        check(broker.plan_growth({1, 0, 0}, {}).outcome == IndexerBudgetBroker::Outcome::fatal,
              "poisoned broker refuses future growth");
    }
    {
        auto broker = make_broker();
        const auto plan = broker.plan_growth({420, 0, 0}, {{1, 300}});
        check(broker.apply_growth_result(plan, {IndexerBudgetBroker::CommitResult::uncertain, 300, {1}, 0, 0}),
              "uncertain report accepted");
        check(broker.telemetry().poisoned, "uncertainty poisons broker");
    }
}

void shrink_hysteresis_and_floors() {
    auto broker = make_broker(1000, 100, 50, 500, 0, 300, 0, 200, 350); // headroom 50.
    auto plan = broker.plan_shrink(260, 64);
    check(plan.outcome == IndexerBudgetBroker::Outcome::ready && plan.release_bytes == 192,
          "low-water shrink releases whole chunks toward target");
    check(broker.apply_shrink_result(plan, 192, false), "shrink confirm succeeds");
    auto t = broker.telemetry();
    check(t.private_indexer_committed_bytes == 308 && t.unused_bytes == 242,
          "shrink updates commitment and headroom without crossing floor");
    plan = broker.plan_shrink(260, 64);
    check(plan.release_bytes == 0 && plan.reason == std::string_view("above low water"),
          "hysteresis avoids flapping above low-water mark");
    check(!broker.apply_shrink_result(plan, 64, false), "invalid shrink report changes no state");
    check(broker.telemetry().private_indexer_committed_bytes == 308, "invalid shrink report leaves state unchanged");

    auto floor = make_broker(1000, 100, 50, 300, 0, 520, 0, 200, 350); // headroom 30, only 40 above floor.
    plan = floor.plan_shrink(260, 64);
    check(plan.release_bytes == 0 && plan.reason == std::string_view("retention floor"),
          "retention floor prevents sub-chunk shrink");
    check(!floor.apply_shrink_result(plan, 0, true), "zero-release uncertain shrink is invalid/no-op");
    check(!floor.telemetry().poisoned, "invalid no-op uncertainty leaves state unchanged");

    auto uncertain = make_broker(1000, 100, 50, 500, 0, 300, 0, 200, 350);
    plan = uncertain.plan_shrink(260, 64);
    check(plan.release_bytes == 192, "uncertain shrink fixture has a real release");
    check(uncertain.apply_shrink_result(plan, 192, true), "uncertain shrink report is accepted as poison");
    const auto uncertain_after = uncertain.telemetry();
    check(uncertain_after.poisoned && uncertain_after.private_indexer_committed_bytes == 500,
          "uncertain shrink keeps conservative commitment while poisoning");
}

void invalid_inputs_and_overflow_preserve_state() {
    IndexerBudgetBroker broker;
    check(!IndexerBudgetBroker::create({100, 20, 20, 50, 40}, {70, 0, 0, 0}, broker),
          "create rejects shrink target below low-water");
    check(!IndexerBudgetBroker::create({100, 20, 20, 50, 60}, {70, 0, 0, 0}, broker),
          "create rejects over-budget initial state");

    broker = make_broker();
    const auto before = broker.telemetry();
    const auto overflow = broker.plan_growth({std::numeric_limits<uint64_t>::max(), 1, 0}, {});
    check(overflow.outcome == IndexerBudgetBroker::Outcome::fatal, "growth arithmetic overflow is fatal plan");
    check(broker.telemetry().private_indexer_committed_bytes == before.private_indexer_committed_bytes,
          "overflow plan leaves state unchanged");
    const auto duplicate = broker.plan_growth({500, 0, 0}, {{1, 200}, {1, 200}});
    check(duplicate.outcome == IndexerBudgetBroker::Outcome::fatal, "duplicate victim candidate is rejected");
}

void stale_growth_and_shrink_plans_are_rejected_before_mutation() {
    auto broker = make_broker(1000, 100, 100, 100, 0, 600); // headroom 100.
    const auto plan = broker.plan_growth({150, 0, 0}, {{1, 100}});
    check(plan.outcome == IndexerBudgetBroker::Outcome::ready, "stale growth fixture plans ready growth");
    check(broker.apply_growth_result(plan, {IndexerBudgetBroker::CommitResult::committed, 100, {1}, 150, 0}),
          "first growth application succeeds");
    const auto after_first = broker.telemetry();
    check(after_first.private_indexer_committed_bytes == 250 && after_first.expert_mapped_bytes == 500,
          "first growth application updates ledger once");
    check(!broker.apply_growth_result(plan, {IndexerBudgetBroker::CommitResult::committed, 100, {1}, 150, 0}),
          "replayed growth plan is rejected");
    const auto after_replay = broker.telemetry();
    check(!after_replay.poisoned && after_replay.private_indexer_committed_bytes == 250 &&
              after_replay.expert_mapped_bytes == 500,
          "stale growth rejection is non-poisoning and state preserving");

    auto shrinker = make_broker(1000, 100, 50, 500, 0, 300, 0, 200, 350);
    const auto shrink = shrinker.plan_shrink(260, 64);
    check(shrink.release_bytes == 192, "stale shrink fixture has releasable chunks");
    check(shrinker.apply_shrink_result(shrink, 192, false), "first shrink application succeeds");
    check(!shrinker.apply_shrink_result(shrink, 192, false), "replayed shrink plan is rejected");
    check(!shrinker.telemetry().poisoned && shrinker.telemetry().private_indexer_committed_bytes == 308,
          "stale shrink rejection is non-poisoning and state preserving");
}

void malformed_reports_poison_without_representing_evictions() {
    {
        auto broker = make_broker();
        const auto plan = broker.plan_growth({400, 0, 0}, {{1, 300}});
        check(!broker.apply_growth_result(plan, {IndexerBudgetBroker::CommitResult::indexer_publication_failed,
                                                 100, {}, 0, 0}),
              "evicted bytes without ids are rejected");
        const auto t = broker.telemetry();
        check(t.poisoned && t.expert_mapped_bytes == 500 && t.evicted_recoverable_bytes == 0,
              "malformed report poisons without inventing eviction state");
    }
    {
        auto broker = make_broker();
        const auto plan = broker.plan_growth({420, 0, 0}, {{1, 160}, {2, 160}});
        check(!broker.apply_growth_result(plan, {IndexerBudgetBroker::CommitResult::indexer_publication_failed,
                                                 320, {1, 1}, 0, 0}),
              "duplicate evicted ids are rejected");
        check(broker.telemetry().poisoned, "duplicate evicted ids poison broker");
    }
    {
        auto broker = make_broker();
        const auto plan = broker.plan_growth({420, 0, 0}, {{1, 160}, {2, 160}});
        check(!broker.apply_growth_result(plan, {IndexerBudgetBroker::CommitResult::indexer_publication_failed,
                                                 100, {1}, 0, 0}),
              "evicted bytes must equal sum of named planned victims");
        check(broker.telemetry().poisoned, "mismatched evicted bytes poison broker");
    }
}

void inconsistent_post_mutation_report_poisons() {
    auto broker = make_broker(1000, 100, 100, 100, 0, 600); // headroom 100; victim needed for 150.
    const auto before = broker.telemetry();
    const auto plan = broker.plan_growth({150, 0, 0}, {{1, 100}});
    check(plan.outcome == IndexerBudgetBroker::Outcome::ready, "inconsistent report fixture plans ready growth");
    check(!broker.apply_growth_result(plan, {IndexerBudgetBroker::CommitResult::committed, 100, {1}, 140, 0}),
          "under-reported commitment is rejected");
    const auto after = broker.telemetry();
    check(after.poisoned && after.private_indexer_committed_bytes == before.private_indexer_committed_bytes &&
              after.expert_mapped_bytes == before.expert_mapped_bytes,
          "inconsistent report poisons and does not pretend to know post-mutation state");
}

void pooled_slab_accounting_is_separate() {
    auto broker = make_broker(1000, 100, 50, 100, 120, 500);
    const auto plan = broker.plan_growth({50, 80, 5000}, {{7, 120}});
    check(plan.outcome == IndexerBudgetBroker::Outcome::ready && plan.victims.empty(),
          "combined private+pooled growth uses available headroom before evictions");
    check(broker.apply_growth_result(plan, {IndexerBudgetBroker::CommitResult::committed, 0, {}, 50, 80}),
          "combined private+pooled commitment confirms");
    const auto t = broker.telemetry();
    check(t.private_indexer_committed_bytes == 150 && t.pooled_slab_committed_bytes == 200,
          "telemetry separates private indexer and pooled slab commitment");
    check(t.actual_indexer_commitment_bytes == 350, "telemetry reports combined actual indexer commitment");
}
} // namespace

int main() {
    no_eviction_when_headroom_suffices();
    minimal_victim_selection_and_exclusions();
    shortage_and_unsupported_are_explicit();
    partial_eviction_and_publication_failure_keeps_misses_coherent();
    restore_failure_and_uncertainty_poison();
    shrink_hysteresis_and_floors();
    invalid_inputs_and_overflow_preserve_state();
    stale_growth_and_shrink_plans_are_rejected_before_mutation();
    malformed_reports_poison_without_representing_evictions();
    inconsistent_post_mutation_report_poisons();
    pooled_slab_accounting_is_separate();
    std::printf("indexer_budget_broker_test: %zu checks passed\n", checks);
}
