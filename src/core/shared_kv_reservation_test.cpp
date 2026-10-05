#include "strata/core/shared_kv_pages.hpp"
#include "strata/core/shared_kv_reservation.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
}
int main() {
    check(shared_kv_reservation_end(1000, 0, 262144) == 1256, "prompt plus rolling headroom, not output cap");
    check(shared_kv_reservation_end(1000, 0, 1004) == 1004, "short output bound");
    check(shared_kv_reservation_end(1256, 1256, 262144) == 1256, "mapped boundary does not renew");
    check(shared_kv_reservation_end(1257, 1256, 262144) == 1513, "grow before crossing boundary");
    const auto max = std::numeric_limits<int64_t>::max();
    check(shared_kv_reservation_end(max - 2, 0, max) == max, "headroom cannot overflow");
    check(!shared_kv_pressure_due(255, 0) && shared_kv_pressure_due(256, 0), "bounded admission progress");
    check(!shared_kv_pressure_due(100, 101), "backward progress not pressure");

    using R = SharedKvReserveResult;
    std::vector<int64_t> attempts;
    auto short_then_exact = [&](int64_t end) {
        attempts.push_back(end);
        return end > 1000 ? R::shortage : R::ready;
    };
    check(shared_kv_reserve(1000, 0, 262144, short_then_exact) == R::ready, "exact write fits despite headroom shortage");
    check(attempts == std::vector<int64_t>({1256, 1000}), "desired then exact before reclaim/pressure");
    attempts.clear();
    check(shared_kv_reserve(1000, 1000, 262144, short_then_exact) == R::ready, "mapped write checks COW");
    check(attempts == std::vector<int64_t>({1000}), "no eager per-token headroom allocation");
    attempts.clear();
    auto fatal = [&](int64_t end) { attempts.push_back(end); return R::fatal; };
    check(shared_kv_reserve(1000, 0, 262144, fatal) == R::fatal, "publication/allocation failure remains fatal");
    check(attempts.size() == 1, "no exact retry after fatal device failure");
    check(shared_kv_reserve(11, 0, 10, fatal) == R::fatal, "invalid extent is not pressure");
    check(attempts.size() == 1, "invalid extent does not call allocator");

    // Real allocator integration, CPU only, four-cell pages as in the model.
    SharedKvPages pages(512, 3);
    std::string error;
    std::vector<SharedKvPages::Copy> copies;
    auto reserve = [&](size_t seq, int64_t begin, int64_t required, int64_t limit) {
        return shared_kv_reserve(required, int64_t(pages.mapping(seq).size()) * 4, limit, [&](int64_t end) {
            if (pages.ensure(seq, size_t(begin / 4), size_t((end + 3) / 4), copies, error)) return R::ready;
            return error == "SharedKvPages insufficient physical pages" ? R::shortage : R::fatal;
        });
    };
    check(reserve(0, 0, 100, 2048) == R::ready, "first full-context-cap request admits");
    check(reserve(1, 0, 200, 2048) == R::ready, "second full-context-cap request overlaps");
    check(pages.used_pages() == 203, "backing reflects prompts plus modest headroom only");
    check(reserve(0, 356, 357, 2048) == R::ready, "rolling crossing allocated before write");
    check(pages.mapping(0).size() * 4 >= 357, "crossing has complete mapping");
    check(reserve(1, 456, 480, 2048) == R::ready, "speculative lookahead mapped in entirety");
    check(pages.mapping(1).size() * 4 >= 480, "verify T extent is mapped");

    SharedKvPages cow(4, 3);
    check(cow.ensure(0, 0, 3, copies, error), "COW source");
    check(cow.clone_prefix(0, 1, 3, error), "shared cached prefix");
    auto cow_reserve = [&](int64_t required, int64_t limit) {
        return shared_kv_reserve(required, 12, limit, [&](int64_t end) {
            if (cow.ensure(1, 2, size_t((end + 3) / 4), copies, error)) return R::ready;
            return error == "SharedKvPages insufficient physical pages" ? R::shortage : R::fatal;
        });
    };
    check(cow_reserve(12, 16) == R::ready && copies.size() == 1, "mapped partial-tail writer still COWs");
    const auto before = cow.mapping(1);
    check(cow_reserve(13, 16) == R::shortage, "actual exhaustion distinguished");
    check(cow.mapping(1) == before, "shortage is atomic; never publish an incomplete map");
    cow.release(0);
    check(cow_reserve(13, 16) == R::ready, "released owner permits exact next write");

    // Late-BSTOP ack matrix. BDONE must be published exactly once per owner: a
    // stray cancel after the owner's terminal BDONE would collide with the next
    // BGEN drain and be misread as that owner's finish. Only slots that still
    // own backing (active decode, protected partial read, or the live BADM admit
    // currently writing) release+ack; everything else is a no-op. Enumerate all
    // eight combinations of (active, partial, live_admit) to pin the predicate.
    for (unsigned mask = 0; mask < 8; ++mask) {
        const bool active = (mask & 1) != 0;
        const bool partial = (mask & 2) != 0;
        const bool live_admit = (mask & 4) != 0;
        const bool expected = active || partial || live_admit;
        check(shared_kv_stop_ack(active, partial, live_admit) == expected,
              "late BSTOP acknowledges exactly when the slot still owns backing");
    }
    check(!shared_kv_stop_ack(false, false, false),
          "late BSTOP of an already-released or idle slot does not fabricate a second BDONE cancel");
    check(shared_kv_stop_ack(true, false, false), "active decoder cancels and releases");
    check(shared_kv_stop_ack(false, true, false), "protected partial read cancels and releases");
    check(shared_kv_stop_ack(false, false, true), "live admission still writing cancels and releases");

    std::printf("shared_kv_reservation: %d checks passed\n", checks);
}
