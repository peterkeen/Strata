#pragma once

#include "strata/core/shared_kv_reservation.hpp"

#include <cstdint>
#include <charconv>
#include <map>
#include <limits>
#include <string>
#include <utility>

namespace strata::core {

enum class BatchMtpFallback { none, incoherent, not_ready, limits, capacity, reservation };
enum class BatchMtpDecision { target_only, speculative, fatal };

struct BatchMtpWindowChoice {
    BatchMtpDecision decision = BatchMtpDecision::target_only;
    BatchMtpFallback fallback = BatchMtpFallback::incoherent;
};

// Called only after the mandatory [p,p+1) target row has been reserved.
// A shortage of the optional row must never be escalated into reclaim/pressure.
template<class ReserveOptional>
BatchMtpWindowChoice batch_mtp_choose_window(bool coherent, bool ready, int64_t produced,
                                             int64_t max_new, int64_t p, int64_t max_context,
                                             bool unified, bool proposal_room, ReserveOptional&& reserve_optional) {
    if (!coherent) return {BatchMtpDecision::target_only, BatchMtpFallback::incoherent};
    if (!ready) return {BatchMtpDecision::target_only, BatchMtpFallback::not_ready};
    if (produced < 0 || max_new - produced < 2 || p < 0 || p > max_context - 3)
        return {BatchMtpDecision::target_only, BatchMtpFallback::limits};
    if (!proposal_room) return {BatchMtpDecision::target_only, BatchMtpFallback::capacity};
    if (!unified) return {BatchMtpDecision::speculative, BatchMtpFallback::none};

    const SharedKvReserveResult reserve = reserve_optional(p + 2);
    if (reserve == SharedKvReserveResult::ready)
        return {BatchMtpDecision::speculative, BatchMtpFallback::none};
    if (reserve == SharedKvReserveResult::shortage)
        return {BatchMtpDecision::target_only, BatchMtpFallback::reservation};
    return {BatchMtpDecision::fatal, BatchMtpFallback::none};
}

enum class BatchMtpOutcome { accepted, rejected, discarded };
struct BatchMtpCommitChoice {
    int keep = 1;
    BatchMtpOutcome outcome = BatchMtpOutcome::rejected;
};

inline BatchMtpCommitChoice batch_mtp_commit_choice(bool offered, int32_t target, int32_t proposal,
                                                       bool eos, bool stop, int64_t produced,
                                                       int64_t max_new, int64_t p, int64_t max_context) {
    if (!offered || target != proposal) return {1, BatchMtpOutcome::rejected};
    if (eos || stop || max_new - produced < 2 || p > max_context - 3)
        return {1, BatchMtpOutcome::discarded};
    return {2, BatchMtpOutcome::accepted};
}

using BatchMtpTestProposals = std::map<std::pair<int, int64_t>, int32_t>;

inline bool parse_batch_mtp_test_proposals(const std::string& text, int slots, int64_t vocab,
                                           BatchMtpTestProposals& out, std::string& error) {
    out.clear();
    if (slots <= 0 || vocab <= 0 || text.empty() || text.back() == ',') {
        error = "empty proposal schedule/entry or invalid bounds";
        return false;
    }
    size_t at = 0;
    auto parse_integer = [&](size_t begin, size_t end, int64_t& value) {
        if (begin == end) return false;
        const char* first = text.data() + begin;
        const char* last = text.data() + end;
        const auto parsed = std::from_chars(first, last, value);
        return parsed.ec == std::errc{} && parsed.ptr == last;
    };
    while (at < text.size()) {
        const size_t end = text.find(',', at) == std::string::npos ? text.size() : text.find(',', at);
        const size_t slash1 = text.find('/', at), slash2 = slash1 == std::string::npos ? slash1 : text.find('/', slash1 + 1);
        int64_t slot = -1, offer = -1, token = -1;
        if (slash1 == std::string::npos || slash1 >= end || slash2 == std::string::npos || slash2 >= end ||
            text.find('/', slash2 + 1) < end || !parse_integer(at, slash1, slot) ||
            !parse_integer(slash1 + 1, slash2, offer) || !parse_integer(slash2 + 1, end, token) ||
            slot < 0 || slot >= slots || offer < 1 || token < 0 || token >= vocab ||
            token > std::numeric_limits<int32_t>::max()) {
            error = "expected unique slot/positive-offer/vocabulary-token entries";
            out.clear();
            return false;
        }
        if (!out.emplace(std::make_pair(static_cast<int>(slot), offer), static_cast<int32_t>(token)).second) {
            error = "duplicate slot/offer entry";
            out.clear();
            return false;
        }
        at = end + 1;
        if (at == text.size() + 1) break;
        if (end == text.size()) break;
    }
    return true;
}

} // namespace strata::core
