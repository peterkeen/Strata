// Bound speculative work and commit to the output actually returned to the caller.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::spec {

// Counts input positions: each verified input can yield one output. The final
// output remains the next unconsumed head, exactly as in one-token decode.
inline int bounded_window(int proposed, int64_t output_left, int64_t context_left) {
    if (proposed <= 0 || output_left <= 0 || context_left <= 0) return 0;
    return static_cast<int>(std::min<int64_t>({proposed, output_left, context_left}));
}

// Include the first EOS in the returned prefix, but never commit the inputs
// beyond it. Call only after verification, before changing persistent state.
template <class IsEos>
int usable_outputs(const int32_t* outputs, int accepted_outputs, int64_t output_left, IsEos is_eos) {
    const int count = bounded_window(accepted_outputs, output_left, accepted_outputs);
    for (int i = 0; i < count; ++i)
        if (is_eos(outputs[i])) return i + 1;
    return count;
}

}  // namespace strata::spec
