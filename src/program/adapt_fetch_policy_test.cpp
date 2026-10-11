#include "strata/program/adapt_fetch_policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace strata::program::adapt_fetch;
namespace {
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
// The pre-change generate.cpp checks, independently retained as a regression oracle.
Reason previous_reason(int requested, const Config& c) {
    const Reason fetch_off = !c.residency ? Reason::no_residency_table : c.spec_split ? Reason::spec_split
        : c.layer_split ? Reason::layer_split : c.peer ? Reason::peer
        : c.helpers ? Reason::helper_gpus : c.async ? Reason::adapt_async
        : c.pipeline ? Reason::pipeline_windows : c.batch ? Reason::batch : Reason::none;
    return requested == 0 ? Reason::disabled : !c.adaptive_tier ? Reason::adaptive_tier_off
        : fetch_off != Reason::none ? fetch_off : !c.copy_mode ? Reason::pcie_mode
        : !c.gpu_plan ? Reason::no_gpu_plan : !c.pcie_share ? Reason::no_pcie_share
        : c.resident_ram ? Reason::resident_ram : Reason::none;
}
}

int main(int argc, char** argv) {
    Config solo{.adaptive_tier = true, .residency = true, .gpu_plan = true, .pcie_share = true};
    // CPU-only protocol fixture: Python supplies the production requested mode and
    // the batch count derived by engine_args. Other predicates use the eligible
    // single-GPU fixture, so this demonstrates the batch exclusion specifically.
    if (argc == 4 && std::strcmp(argv[1], "--info") == 0) {
        solo.batch = std::atoi(argv[3]) > 0;
        const auto s = evaluate(std::atoi(argv[2]), solo);
        std::printf("INFO adapt_fetch_requested=%d adapt_fetch_effective=%d adapt_fetch_reason=%s\n",
                    s.requested, s.effective, code(s.reason));
        return 0;
    }
    for (int requested = 0; requested <= 2; ++requested) {
        const auto s = evaluate(requested, solo);
        check(s.requested == requested && s.effective == requested, "solo retains the requested mode");
        check(s.reason == (requested ? Reason::none : Reason::disabled), "explicit off differs from excluded");
    }
    // parallel=2 maps to --batch 2. Admission is excluded by configured slots,
    // not the active-stream count; unified KV and batch MTP do not bypass it.
    Config nibbler = solo;
    nibbler.batch = true;
    const auto batch = evaluate(2, nibbler);
    check(batch.requested == 2 && batch.effective == 0 && batch.reason == Reason::batch,
          "nibbler requested mode 2 remains off in a two-slot engine");
    check(std::strcmp(code(batch.reason), "batch") == 0, "batch reason is machine-readable");
    check(std::strcmp(description(batch.reason, true), "--batch") == 0, "existing batch diagnostic retained");
    check(evaluate(0, nibbler).reason == Reason::disabled, "requested off does not claim batch exclusion");
    nibbler.resident_ram = true;
    check(evaluate(2, nibbler).reason == Reason::batch, "batch retains priority over resident-RAM exclusion");
    nibbler.adaptive_tier = false;
    check(evaluate(2, nibbler).reason == Reason::adaptive_tier_off, "adaptive tier off retains first priority");

    // Exhaust every combination of the 13 original predicates and all normalized
    // requested modes. The CLI supplies false for serve-only predicates.
    for (unsigned bits = 0; bits < (1u << 13); ++bits) {
        auto b = [&](unsigned i) { return (bits & (1u << i)) != 0; };
        const Config c{b(0), b(1), b(2), b(3), b(4), b(5), b(6), b(7), b(8), b(9), b(10), b(11), b(12)};
        for (int requested = 0; requested <= 2; ++requested) {
            const auto s = evaluate(requested, c);
            const auto old = previous_reason(requested, c);
            check(s.reason == old && s.effective == (old == Reason::none ? requested : 0),
                  "admission and exclusion priority remain identical to pre-change policy");
            const char* reason = code(s.reason);
            check(std::strcmp(reason, "unknown") != 0 && std::strpbrk(reason, " \t\r\n=") == nullptr,
                  "every reason is a single INFO value");
        }
    }
    if (!failures) std::puts("adapt_fetch_policy_test: PASS (24576 policy combinations)");
    return failures ? 1 : 0;
}
