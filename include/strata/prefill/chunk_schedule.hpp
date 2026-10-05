// A capacity-bounded schedule that avoids a tiny last chunk when possible.
#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::prefill {

// Only shorten the penultimate chunk; never increase the number of chunks or
// exceed capacity. If both remaining chunks cannot reach the streaming floor,
// retain the ordinary schedule. Subtractions avoid overflow near INT64_MAX.
inline int64_t next_chunk_tokens(int64_t remaining, int64_t capacity, int64_t stream_min = 0) {
    if (remaining <= 0 || capacity <= 0) return 0;
    const int64_t ordinary = std::min(remaining, capacity);
    if (stream_min <= 0 || stream_min > capacity || remaining <= capacity) return ordinary;
    if (remaining - capacity >= stream_min) return ordinary;
    const int64_t shortened = remaining - stream_min;
    return shortened >= stream_min ? shortened : ordinary;
}

}  // namespace strata::prefill
