// CPU-only rolling reservation policy. Neither logical context nor backing shrinks.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
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

// BSTOP and BHANDOFF address one slot by number. The whole argument field must be a
// decimal inside [0, n_slots): a bare atoi("foo") is 0, so a malformed control line
// would cancel or hand off somebody else's owner. Returns false, and leaves slot
// untouched, for garbage, a negative number, or trailing junk after the number.
inline bool shared_kv_slot_arg(const std::string& line, size_t arg0, int n_slots, int& slot) {
    if (arg0 >= line.size()) return false;
    const char* begin = line.c_str() + arg0;
    char* end = nullptr;
    const long v = std::strtol(begin, &end, 10);
    if (end == begin || v < 0 || v >= static_cast<long>(n_slots)) return false;
    while (*end == ' ' || *end == '\t') ++end;   // the slot is the last field of the line
    if (*end != '\0') return false;
    slot = static_cast<int>(v);
    return true;
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
