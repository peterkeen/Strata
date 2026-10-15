#include "strata/core/parallel_resource_policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

using namespace strata::core;

// Not <cassert>: the default Release build defines NDEBUG and would compile every check away.
static int checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)

int main() {
    const uint64_t prefix[] = {0, 20, 44, 72, 104};
    auto bytes = [&](int64_t n) { return prefix[n]; };
    int64_t keep = -1;
    CHECK(parallel_activation_keep_slots(4, 104, 16, 50, 10, bytes, keep) && keep == 2);
    CHECK(!parallel_activation_keep_slots(4, 104, 16, 120, 10, bytes, keep));
    CHECK(!parallel_activation_keep_slots(4, 104, 16, UINT64_MAX, 10, bytes, keep));
    CHECK(parallel_activation_keep_slots(4, 104, 70, 50, 10, bytes, keep) && keep == 4);
    CHECK(!parallel_activation_keep_slots(4, 103, 70, 50, 10, bytes, keep));

    ParallelResourceLedger ledger;
    ledger.private_slot_session_bytes = 512;
    ledger.batch_verifier_graph_bytes = 64;
    ParallelActivationBudget budget;
    std::string error;
    CHECK(ledger.build_activation_budget(budget, error));
    CHECK(budget.required_total_bytes == 1088);
    ParallelResourcePolicyModel model;
    int allocated = 0, released = 0, bound = 0, detached = 0;
    ParallelResourceOps ops;
    ops.shrink_expert_cache_for_activation = [](const ParallelActivationBudget&) { return ParallelOpCode::ok; };
    ops.allocate_slot_session = [&](uint32_t) { ++allocated; return ParallelOpCode::ok; };
    ops.bind_slot_session = [&](uint32_t s) {
        if (s == 1) return ParallelOpCode::binding_failed;
        ++bound;
        return ParallelOpCode::ok;
    };
    ops.free_slot_session = [&](uint32_t) { ++released; };
    ops.detach_bindings = [&] { ++detached; return ParallelOpCode::ok; };
    const ParallelSafePoint safe{true, true, true, 0};
    auto result = model.activate_on_second_arrival(ledger, safe, ops);
    CHECK(!result.ok() && result.state == ParallelState::solo && result.first_stream_intact);
    CHECK(result.second_stream_queued && model.generation() == 0 && allocated == 2 && released == 2);
    // A subsequent admission must be able to activate both slots, not only
    // the first one. Publication/generation advance only after success.
    ops.bind_slot_session = [&](uint32_t) { ++bound; return ParallelOpCode::ok; };
    result = model.activate_on_second_arrival(ledger, safe, ops);
    CHECK(result.ok() && model.physical_session_count() == 2 && model.generation() == 1);
    CHECK(model.bindings_published() && model.state() == ParallelState::parallel);
    (void) detached;
    std::printf("lazy_parallel_activation_test: %d checks passed\n", checks);
    return 0;
}
