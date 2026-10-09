#include "strata/core/batch_mtp_policy.hpp"
#include "strata/core/shared_kv_pages.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace strata::core;
namespace {
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
}

int main() {
    int reserve_calls = 0;
    int64_t reserve_end = -1;
    auto ready = [&](int64_t end) { ++reserve_calls; reserve_end = end; return SharedKvReserveResult::ready; };
    auto choice = batch_mtp_choose_window(false, true, 0, 8, 20, 64, true, true, ready);
    check(choice.decision == BatchMtpDecision::target_only && choice.fallback == BatchMtpFallback::incoherent &&
          reserve_calls == 0, "incoherent slots do not attempt optional reservation");
    choice = batch_mtp_choose_window(true, false, 0, 8, 20, 64, true, true, ready);
    check(choice.decision == BatchMtpDecision::target_only && choice.fallback == BatchMtpFallback::not_ready &&
          reserve_calls == 0, "missing proposal readiness is target-only");
    choice = batch_mtp_choose_window(true, true, 7, 8, 20, 64, true, true, ready);
    check(choice.fallback == BatchMtpFallback::limits && reserve_calls == 0,
          "last output token cannot consume optional proposal row");
    choice = batch_mtp_choose_window(true, true, 0, 8, 62, 64, true, true, ready);
    check(choice.fallback == BatchMtpFallback::limits && reserve_calls == 0,
          "context edge cannot consume optional proposal row");
    choice = batch_mtp_choose_window(true, true, 0, 8, 20, 64, true, false, ready);
    check(choice.decision == BatchMtpDecision::target_only && choice.fallback == BatchMtpFallback::capacity &&
          reserve_calls == 0, "insufficient batch rows falls back before reserving the optional extent");
    choice = batch_mtp_choose_window(true, true, 0, 8, 20, 64, true, true, ready);
    check(choice.decision == BatchMtpDecision::speculative && reserve_end == 22,
          "unified proposal reserves exclusive p+2 before verifier");
    const int calls_before_ordinary = reserve_calls;
    choice = batch_mtp_choose_window(true, true, 0, 8, 20, 64, false, true, ready);
    check(choice.decision == BatchMtpDecision::speculative && reserve_calls == calls_before_ordinary,
          "ordinary batch MTP keeps its existing no-unified reservation policy");

    auto short_reserve = [](int64_t) { return SharedKvReserveResult::shortage; };
    choice = batch_mtp_choose_window(true, true, 0, 8, 20, 64, true, true, short_reserve);
    check(choice.decision == BatchMtpDecision::target_only && choice.fallback == BatchMtpFallback::reservation,
          "optional shortage falls back without pressure escalation");
    auto fatal_reserve = [](int64_t) { return SharedKvReserveResult::fatal; };
    choice = batch_mtp_choose_window(true, true, 0, 8, 20, 64, true, true, fatal_reserve);
    check(choice.decision == BatchMtpDecision::fatal,
          "optional allocation/device failure remains fatal, not fallback pressure");

    // A shared partial tail at p=3 needs COW for the mandatory end 4. The
    // optional end 5 needs another page and must fail atomically with no reclaim.
    {
        SharedKvPages pages(2, 3);
        std::string error;
        std::vector<SharedKvPages::Copy> copies;
        check(pages.ensure(0, 0, 1, copies, error), "allocate shared-prefix source page");
        check(pages.clone_prefix(0, 1, 1, error), "clone shared page through mandatory frontier");
        auto reserve_extent = [&](int64_t end) {
            const int64_t mapped = int64_t(pages.mapping(1).size()) * 4;
            return shared_kv_reserve(end, mapped, 8, [&](int64_t desired) {
                if (pages.ensure(1, 0, size_t((desired + 3) / 4), copies, error))
                    return SharedKvReserveResult::ready;
                return error == "SharedKvPages insufficient physical pages" ? SharedKvReserveResult::shortage
                                                                            : SharedKvReserveResult::fatal;
            });
        };
        check(reserve_extent(4) == SharedKvReserveResult::ready,
              "mandatory first row COW reservation succeeds");
        const auto before_optional = pages.mapping(1);
        choice = batch_mtp_choose_window(true, true, 0, 8, 3, 64, true, true, reserve_extent);
        check(choice.decision == BatchMtpDecision::target_only &&
              choice.fallback == BatchMtpFallback::reservation,
              "p+2 optional row falls back when cross-page COW is short");
        check(pages.mapping(1) == before_optional,
              "failed optional reservation publishes no partial COW mapping");
        check(pages.mapping(1).size() * 4 >= 4,
              "mandatory [p,p+1) mapping remains available after optional shortage");
    }

    auto accepted = batch_mtp_commit_choice(true, 7, 7, false, false, 0, 8, 20, 64);
    check(accepted.keep == 2 && accepted.outcome == BatchMtpOutcome::accepted,
          "matching proposal commits two-row prefix");
    auto rejected = batch_mtp_commit_choice(true, 7, 8, false, false, 0, 8, 20, 64);
    check(rejected.keep == 1 && rejected.outcome == BatchMtpOutcome::rejected,
          "mismatch commits mandatory row only");
    auto discarded = batch_mtp_commit_choice(true, 7, 7, true, false, 0, 8, 20, 64);
    check(discarded.keep == 1 && discarded.outcome == BatchMtpOutcome::discarded,
          "matching proposal on EOS is discarded without two-token commit");
    discarded = batch_mtp_commit_choice(true, 7, 7, false, false, 6, 7, 20, 64);
    check(discarded.keep == 1 && discarded.outcome == BatchMtpOutcome::discarded,
          "matching proposal at output limit is discarded");
    check(batch_mtp_commit_choice(false, 7, 7, false, false, 0, 8, 20, 64).keep == 1,
          "one-row fallback never indexes or commits a proposal row");

    BatchMtpTestProposals proposals;
    std::string parse_error;
    check(parse_batch_mtp_test_proposals("0/1/42,1/3/7", 2, 100, proposals, parse_error) &&
          proposals.size() == 2 && proposals.at({0, 1}) == 42 && proposals.at({1, 3}) == 7,
          "test-hook proposal schedule parses exact slot/offer/token triples");
    check(!parse_batch_mtp_test_proposals("0/0/42", 2, 100, proposals, parse_error),
          "test-hook offer ordinal must be positive");
    check(!parse_batch_mtp_test_proposals("0/1/100", 2, 100, proposals, parse_error),
          "test-hook token must be inside vocabulary");
    check(!parse_batch_mtp_test_proposals("0/1/42,0/1/43", 2, 100, proposals, parse_error),
          "test-hook schedule rejects duplicate slot/offer keys");
    check(!parse_batch_mtp_test_proposals("0/1/42,", 2, 100, proposals, parse_error),
          "test-hook schedule rejects empty trailing entry");

    if (failures) return EXIT_FAILURE;
    std::cout << "batch MTP policy/reservation: PASS\n";
    return EXIT_SUCCESS;
}
