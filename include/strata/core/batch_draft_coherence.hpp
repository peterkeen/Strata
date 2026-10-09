#pragma once

#include <cstdint>

namespace strata::core {

// Provenance for a private MTP K/V image. `upto` is the exclusive token
// boundary represented by the draft state; proposal readiness is separate.
struct BatchDraftCoherence {
    bool coherent = false;
    int64_t upto = -1;

    void invalidate() { coherent = false; upto = -1; }
    void establish(int64_t end) {
        coherent = end >= 0;
        upto = coherent ? end : -1;
    }
    bool matches(int64_t end) const { return coherent && end >= 0 && upto == end; }
};

// Only an exact live-prefix image may cross an ownership boundary. Images,
// older checkpoints, and target-only restores are deliberately not eligible.
inline bool batch_draft_copy_to_slot(const BatchDraftCoherence& main, int64_t source_end,
                                     bool target_only_source = false) {
    return !target_only_source && main.matches(source_end);
}

inline bool batch_draft_copy_to_main(const BatchDraftCoherence& slot, int64_t slot_end,
                                     bool full_slot, bool target_only_source = false) {
    return full_slot && !target_only_source && slot.matches(slot_end);
}

// A committed target window preserves draft provenance only if the slot's
// drafter was advanced over that exact new history. In particular, terminal
// windows cannot inherit the prior coherence bit merely because it was true.
inline void batch_draft_after_commit(BatchDraftCoherence& state, int64_t committed_end,
                                     bool drafter_advanced, bool slot_continues) {
    if (drafter_advanced && slot_continues) state.establish(committed_end);
    else state.invalidate();
}

} // namespace strata::core
