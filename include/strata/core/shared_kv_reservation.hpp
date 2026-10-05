// CPU-only rolling reservation policy. Neither logical context nor backing shrinks.
#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

namespace strata::core {

inline constexpr int64_t shared_kv_reserve_ahead = 256;
inline constexpr int64_t shared_kv_pressure_quantum = 256;

// Shortage is allocator preflight only, with ownership unchanged. Host allocation,
// publication and device failures are fatal, never reasons to pressure-park.
enum class SharedKvReserveResult { ready, shortage, fatal };

// required is the EXCLUSIVE end of every imminent target write (including
// speculative rows). limit is this request's logical/output bound, not its cap's
// aggregate reservation. Renew headroom only on a mapping boundary; a mapped
// write must still be checked for COW by the attempt callback.
inline int64_t shared_kv_reservation_end(int64_t required, int64_t mapped, int64_t limit) {
    if (mapped >= required) return required;
    return required + std::min(shared_kv_reserve_ahead, limit - required);
}

template<class Attempt>
SharedKvReserveResult shared_kv_reserve(int64_t required, int64_t mapped, int64_t limit, Attempt&& attempt) {
    if (required < 0 || mapped < 0 || limit < required) return SharedKvReserveResult::fatal;
    const int64_t desired = shared_kv_reservation_end(required, mapped, limit);
    if (desired != required) {
        const auto result = attempt(desired);
        if (result != SharedKvReserveResult::shortage) return result;
    }
    // Modest headroom is optional. Never invoke pressure for a write that fits.
    return attempt(required);
}

inline bool shared_kv_pressure_due(int64_t produced, int64_t at_wait) {
    return produced >= at_wait && produced - at_wait >= shared_kv_pressure_quantum;
}

// Internal HANDOFF (2) may retain a valid completed-prefill cache; actual
// cancellation (1), partial reads and disabled caches must still release.
inline bool shared_kv_handoff_cache(int stop, bool prompt_complete, bool cache_valid) {
    return stop == 2 && prompt_complete && cache_valid;
}

// A late BSTOP arriving after a slot already published its terminal BDONE must not
// fabricate a second ack: the frontend sees exactly one BDONE per owner, and a
// stray cancel can otherwise collide with the next owner's BGEN drain. Only slots
// that still own backing (active decode, protected partial read, or the live BADM
// admission currently writing) release and acknowledge; everything else is a
// no-op. The predicate is deliberately pure for CPU tests.
inline bool shared_kv_stop_ack(bool active, bool partial, bool live_admit) {
    return active || partial || live_admit;
}

} // namespace strata::core
