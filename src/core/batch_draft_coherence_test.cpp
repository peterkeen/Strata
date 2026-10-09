#include "strata/core/batch_draft_coherence.hpp"

#include <cstdlib>
#include <iostream>

using strata::core::BatchDraftCoherence;

namespace {
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
}

int main() {
    BatchDraftCoherence main;
    check(!strata::core::batch_draft_copy_to_slot(main, 12), "unknown main provenance refuses admission");
    main.establish(12);
    check(strata::core::batch_draft_copy_to_slot(main, 12), "exact coherent main prefix admits private copy");
    check(!strata::core::batch_draft_copy_to_slot(main, 11), "different source prefix refuses private copy");
    check(!strata::core::batch_draft_copy_to_slot(main, 12, true), "image/target-only admission refuses private copy");

    BatchDraftCoherence slot;
    slot.establish(12);
    check(strata::core::batch_draft_copy_to_main(slot, 12, true), "coherent full slot restores private draft");
    check(!strata::core::batch_draft_copy_to_main(slot, 11, true), "mismatched full slot restore fails closed");
    check(!strata::core::batch_draft_copy_to_main(slot, 12, false), "older checkpoint cannot borrow current slot draft");
    check(!strata::core::batch_draft_copy_to_main(slot, 12, true, true), "target-only/image restore suppresses private draft");

    strata::core::batch_draft_after_commit(slot, 13, true, true);
    check(slot.matches(13), "advanced drafter records exact committed prefix");
    strata::core::batch_draft_after_commit(slot, 14, false, true);
    check(!slot.matches(14), "target-only/failed catch-up invalidates prior coherence");
    slot.establish(14);
    strata::core::batch_draft_after_commit(slot, 15, false, false);
    check(!slot.coherent && slot.upto == -1, "terminal EOS/length invalidates even previously coherent ring");

    slot.establish(20);
    slot.invalidate(); // pressure park/release, cancellation, or slot reuse
    check(!strata::core::batch_draft_copy_to_main(slot, 20, true), "pressure/cancel/reuse reset prevents stale restore");
    slot.establish(30); // exact BYIELD prefix carried to its slot
    check(strata::core::batch_draft_copy_to_main(slot, 30, true), "exact yielded prefix can hand off its private ring");
    slot.invalidate(); // a target-only pressure image does not carry draft payload
    check(!strata::core::batch_draft_copy_to_main(slot, 30, true, true), "pressure target-only image cannot restore MTP");

    if (failures) return EXIT_FAILURE;
    std::cout << "batch draft coherence transitions: PASS\n";
    return EXIT_SUCCESS;
}
