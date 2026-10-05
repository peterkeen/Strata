#include "strata/core/shared_kv_pages.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

using strata::core::SharedKvPages;

namespace {
size_t checks = 0;

void check(bool ok, const char* label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::exit(1);
    }
}

template <typename Exception, typename F>
void check_throws(F action, const char* label) {
    bool caught = false;
    try {
        action();
    } catch (const Exception&) {
        caught = true;
    }
    check(caught, label);
}

struct Snapshot {
    std::vector<std::vector<int32_t>> mappings;
    size_t free;
    size_t used;
};

Snapshot snapshot(const SharedKvPages& pool, size_t sequences) {
    Snapshot result{{}, pool.free_pages(), pool.used_pages()};
    for (size_t seq = 0; seq < sequences; ++seq) result.mappings.push_back(pool.mapping(seq));
    return result;
}

void unchanged(const SharedKvPages& pool, const Snapshot& before) {
    check(pool.free_pages() == before.free && pool.used_pages() == before.used,
          "failure preserves physical accounting");
    for (size_t seq = 0; seq < before.mappings.size(); ++seq) {
        check(pool.mapping(seq) == before.mappings[seq], "failure preserves all mappings");
    }
}

// Without exposing private refcounts, verify that used_pages counts precisely
// the union of live IDs, not the number of logical references. The COW/content
// tests below additionally detect incorrect refcounts on shared live pages.
void invariants(const SharedKvPages& pool, size_t sequences) {
    std::set<int32_t> live;
    for (size_t seq = 0; seq < sequences; ++seq) {
        std::set<int32_t> local;
        for (int32_t page : pool.mapping(seq)) {
            check(page >= 0 && static_cast<size_t>(page) < pool.capacity(), "physical ID in range");
            check(local.insert(page).second, "no duplicate physical ID within a sequence");
            live.insert(page);
        }
    }
    check(live.size() == pool.used_pages(), "distinct live IDs match ownership accounting");
    check(pool.free_pages() + pool.used_pages() == pool.capacity(), "pool accounting balances");
}

void growth_and_recycling() {
    SharedKvPages pool(9, 3);
    std::vector<SharedKvPages::Copy> copies;
    std::string error = "stale";
    check(pool.capacity() == 9 && pool.free_pages() == 9 && pool.used_pages() == 0,
          "initial pool is empty");
    check(pool.ensure(0, 0, 2, copies, error), "first sequence grows");
    check(copies.empty() && error.empty(), "success clears output state");
    check(pool.mapping(0) == std::vector<int32_t>({0, 1}), "initial allocation is ascending");
    check(pool.ensure(1, 0, 1, copies, error), "second sequence grows less");
    check(pool.ensure(0, 2, 4, copies, error), "first sequence grows again");
    check(pool.ensure(2, 0, 2, copies, error), "third sequence grows interleaved");
    check(pool.ensure(1, 1, 3, copies, error), "second sequence catches up");
    check(pool.mapping(0) == std::vector<int32_t>({0, 1, 3, 4}), "unequal growth has stable first mapping");
    check(pool.mapping(1) == std::vector<int32_t>({2, 7, 8}), "unequal growth has stable second mapping");
    check(pool.mapping(2) == std::vector<int32_t>({5, 6}), "unequal growth has stable third mapping");
    invariants(pool, 3);
    pool.release(1);
    check(pool.free_pages() == 3, "release frees only that sequence");
    pool.release(1);
    check(pool.free_pages() == 3, "repeated release is harmless");
    check(pool.ensure(2, 2, 5, copies, error), "released pages can be reused");
    check(pool.mapping(2) == std::vector<int32_t>({5, 6, 2, 7, 8}), "tail-first release gives deterministic LIFO reuse");
    check(pool.ensure(0, 0, 2, copies, error), "ensure smaller end retains tail");
    check(pool.mapping(0).size() == 4 && copies.empty(), "ensure does not truncate implicitly");
    check(pool.truncate(0, 2, error), "truncate drops only tail");
    check(pool.mapping(0) == std::vector<int32_t>({0, 1}) && pool.free_pages() == 2,
          "truncate frees unshared tail");
    check(pool.truncate(0, std::numeric_limits<size_t>::max(), error), "oversize truncate is no-op");
    check(pool.mapping(0).size() == 2, "truncate never grows");
    invariants(pool, 3);
}

void sharing_and_partial_page_cow() {
    SharedKvPages pool(8, 3);
    std::vector<SharedKvPages::Copy> copies;
    std::string error;
    check(pool.ensure(0, 0, 3, copies, error), "source grows");
    std::vector<std::array<int, 4>> physical(pool.capacity());
    for (size_t i = 0; i < physical.size(); ++i) physical[i] = {{int(i * 10), 1, 2, 3}};
    check(pool.ensure(1, 0, 1, copies, error), "destination initially owns a page");
    check(pool.clone_prefix(0, 1, 3, error), "clone replaces destination with full shared prefix");
    check(pool.mapping(1) == pool.mapping(0) && pool.used_pages() == 3,
          "clone releases old destination and shares pages");
    check(pool.clone_prefix(0, 2, 2, error), "third sequence shares shorter prefix");
    check(pool.ensure(1, 2, 4, copies, error), "partial-page write COW plus growth");
    check(copies.size() == 1 && copies[0].from == 2 && copies[0].to == 3,
          "only shared writable partial page is copied");
    for (const auto& copy : copies) physical[static_cast<size_t>(copy.to)] = physical[static_cast<size_t>(copy.from)];
    const auto destination_page = static_cast<size_t>(pool.mapping(1)[2]);
    physical[destination_page][2] = 99; // Model a write to just one token in the page.
    check(physical[2][2] == 2 && physical[destination_page][2] == 99,
          "partial write does not alter shared source page");
    check(physical[destination_page][0] == 20 && physical[destination_page][3] == 3,
          "COW preserves the untouched tokens of a partial page");
    check(pool.mapping(1)[0] == pool.mapping(0)[0] && pool.mapping(1)[1] == pool.mapping(0)[1],
          "read-only prefix remains shared");
    check(pool.ensure(1, 2, 4, copies, error) && copies.empty(), "private pages are never copied twice");
    check(pool.ensure(2, 0, 2, copies, error) && copies.size() == 2, "whole shared prefix can become private");
    check(copies[0].from == 0 && copies[1].from == 1, "copy records follow logical order");
    invariants(pool, 3);
    pool.release(0);
    check(pool.used_pages() == 6, "release frees only last-reference source page");
    pool.release(1);
    check(pool.used_pages() == 2, "release respects references after COW");
    pool.release(2);
    check(pool.free_pages() == 8, "last references recycle every page");
    invariants(pool, 3);
}

void exhaustion_and_clone_semantics() {
    SharedKvPages pool(4, 3);
    std::vector<SharedKvPages::Copy> copies;
    std::string error;
    check(pool.ensure(0, 0, 3, copies, error), "exhaustion source setup");
    check(pool.clone_prefix(0, 1, 3, error), "exhaustion shared setup");
    auto before = snapshot(pool, 3);
    copies.push_back({99, 99});
    check(!pool.ensure(1, 2, 4, copies, error), "combined COW and growth require two free pages");
    check(copies.empty() && !error.empty(), "exhaustion returns empty copies and diagnostic");
    unchanged(pool, before);
    check(pool.ensure(2, 0, 1, copies, error), "failure did not consume free-list head");
    check(pool.mapping(2) == std::vector<int32_t>({3}), "failed ensure preserves allocation order");
    before = snapshot(pool, 3);
    check(!pool.ensure(1, 0, 3, copies, error), "full pool cannot COW shared pages");
    unchanged(pool, before);
    check(pool.ensure(0, 3, 3, copies, error) && copies.empty(), "read-only ensure succeeds with no free pages");
    check(pool.clone_prefix(0, 2, 3, error), "sharing succeeds in a full pool");
    check(pool.free_pages() == 1, "clone recycles old destination");
    before = snapshot(pool, 3);
    check(!pool.clone_prefix(0, 2, 4, error), "incomplete source rejected before releasing destination");
    unchanged(pool, before);
    check(!pool.clone_prefix(0, 0, 4, error), "incomplete self-prefix rejected");
    unchanged(pool, before);
    check(pool.clone_prefix(0, 0, 2, error), "self clone truncates to complete prefix");
    check(pool.mapping(0) == std::vector<int32_t>({0, 1}) && pool.used_pages() == 3,
          "self clone leaves shared tail alive in other sequences");
    check(pool.truncate(1, 1, error), "truncate shared tail");
    check(pool.used_pages() == 3, "truncate keeps pages owned by another sequence");
    check(pool.clone_prefix(2, 2, 0, error), "empty self clone releases all its references");
    check(pool.used_pages() == 2, "last shared tail reference is freed");
    check(pool.clone_prefix(2, 0, 0, error), "empty source prefix replaces populated destination");
    check(pool.mapping(0).empty() && pool.used_pages() == 1, "empty clone respects remaining owner");
    check(pool.clone_prefix(1, 1, 1, error), "full self clone is a no-op");
    pool.release(1);
    check(pool.used_pages() == 0, "all clone references eventually released");
    invariants(pool, 3);
}

void invalid_inputs() {
    check_throws<std::invalid_argument>([] { SharedKvPages pool(0, 1); }, "zero capacity rejected explicitly");
    check_throws<std::invalid_argument>([] { SharedKvPages pool(1, 0); }, "zero sequences rejected explicitly");
    check_throws<std::invalid_argument>([] {
        SharedKvPages pool(static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1, 1);
    }, "capacity overflow rejected before allocation");
    check_throws<std::invalid_argument>([] {
        SharedKvPages pool(std::numeric_limits<size_t>::max(), 1);
    }, "maximal capacity rejected before allocation");
    SharedKvPages pool(3, 2);
    std::vector<SharedKvPages::Copy> copies;
    std::string error;
    check(pool.ensure(0, 0, 2, copies, error), "invalid-input setup");
    check(pool.clone_prefix(0, 1, 2, error), "invalid-input sharing setup");
    const auto before = snapshot(pool, 2);
    for (const auto& input : std::vector<std::array<size_t, 3>>{
             {{2, 0, 1}}, {{std::numeric_limits<size_t>::max(), 0, 0}},
             {{0, 2, 1}}, {{0, 0, 4}}, {{0, 0, std::numeric_limits<size_t>::max()}},
             {{0, std::numeric_limits<size_t>::max(), 1}}}) {
        copies.push_back({9, 9});
        check(!pool.ensure(input[0], input[1], input[2], copies, error), "invalid ensure rejected");
        check(!error.empty() && copies.empty(), "invalid ensure resets copies and sets error");
        unchanged(pool, before);
    }
    check(!pool.clone_prefix(2, 0, 0, error), "invalid clone source rejected");
    unchanged(pool, before);
    check(!pool.clone_prefix(0, 2, 0, error), "invalid clone destination rejected");
    unchanged(pool, before);
    check(!pool.clone_prefix(0, 1, std::numeric_limits<size_t>::max(), error), "huge incomplete clone rejected");
    unchanged(pool, before);
    check(!pool.truncate(2, 0, error) && !error.empty(), "invalid truncate rejected");
    unchanged(pool, before);
    check_throws<std::out_of_range>([&] { pool.release(2); }, "invalid release throws");
    check_throws<std::out_of_range>([&] { pool.release(std::numeric_limits<size_t>::max()); }, "huge release throws");
    check_throws<std::out_of_range>([&] { (void)pool.mapping(2); }, "invalid mapping throws");
    check_throws<std::out_of_range>([&] { (void)pool.mapping(std::numeric_limits<size_t>::max()); }, "huge mapping throws");
    unchanged(pool, before);
    check(pool.ensure(0, 0, 0, copies, error) && copies.empty() && error.empty(), "empty ensure clears error");
    unchanged(pool, before);
    pool.release(1);
    check(pool.ensure(0, 0, 3, copies, error) && copies.empty(), "invalid calls did not inflate refcounts");
    check(pool.mapping(0) == std::vector<int32_t>({0, 1, 2}), "invalid calls preserved free list");
    invariants(pool, 2);
}

// Deterministic model test: simulate GPU page copies and single-token writes.
// Track logical contents independently, so any premature recycling, missing
// reference, extra reference, or missing/incorrect COW corrupts an assertion.
void model_test() {
    constexpr size_t capacity = 13;
    constexpr size_t sequences = 5;
    using Page = std::array<int, 4>;
    SharedKvPages pool(capacity, sequences);
    std::vector<Page> physical(capacity, Page{{-1, -1, -1, -1}});
    std::vector<std::vector<Page>> expected(sequences);
    std::mt19937 rng(0x51A7A);
    std::vector<SharedKvPages::Copy> copies;
    std::string error;
    int value = 0;

    for (size_t step = 0; step < 4000; ++step) {
        const size_t seq = rng() % sequences;
        const auto before = snapshot(pool, sequences);
        switch (rng() % 4) {
        case 0: {
            const size_t end = rng() % (capacity + 3);
            const size_t begin = rng() % (end + 2);
            size_t cow = 0;
            for (size_t page = begin; page < std::min(end, before.mappings[seq].size()); ++page) {
                const auto id = before.mappings[seq][page];
                size_t owners = 0;
                for (const auto& mapping : before.mappings) {
                    owners += static_cast<size_t>(std::count(mapping.begin(), mapping.end(), id));
                }
                if (owners > 1) ++cow;
            }
            const size_t growth = end > expected[seq].size() ? end - expected[seq].size() : 0;
            const bool should_succeed = begin <= end && end <= capacity && cow + growth <= before.free;
            copies.push_back({-1, -1});
            check(pool.ensure(seq, begin, end, copies, error) == should_succeed, "modeled ensure admission");
            if (!should_succeed) {
                check(copies.empty() && !error.empty(), "modeled failure output");
                unchanged(pool, before);
                break;
            }
            check(error.empty() && copies.size() == cow, "modeled copy count");
            size_t record = 0;
            for (size_t page = 0; page < before.mappings[seq].size(); ++page) {
                if (pool.mapping(seq)[page] != before.mappings[seq][page]) {
                    check(page >= begin && page < end, "COW only changes writable range");
                    check(record < copies.size(), "every changed mapping has a copy");
                    const auto copy = copies[record++];
                    check(copy.from == before.mappings[seq][page] && copy.to == pool.mapping(seq)[page],
                          "copy matches old and new physical mappings");
                }
            }
            check(record == copies.size(), "no extra copy records");
            for (const auto& copy : copies) physical[static_cast<size_t>(copy.to)] = physical[static_cast<size_t>(copy.from)];
            while (expected[seq].size() < pool.mapping(seq).size()) {
                const size_t page = expected[seq].size();
                expected[seq].push_back(physical[static_cast<size_t>(pool.mapping(seq)[page])]);
            }
            for (size_t page = begin; page < end; ++page) {
                const size_t token = rng() % 4;
                physical[static_cast<size_t>(pool.mapping(seq)[page])][token] = ++value;
                expected[seq][page][token] = value;
            }
            break;
        }
        case 1: {
            const size_t from = rng() % sequences;
            const size_t end = rng() % (capacity + 3);
            const bool valid = end <= expected[from].size();
            std::vector<Page> prefix;
            if (valid) prefix.assign(expected[from].begin(), expected[from].begin() + end);
            check(pool.clone_prefix(from, seq, end, error) == valid, "modeled clone admission");
            if (valid) {
                check(error.empty(), "clone clears error");
                expected[seq] = prefix;
            } else {
                check(!error.empty(), "incomplete clone sets error");
                unchanged(pool, before);
            }
            break;
        }
        case 2: {
            const size_t end = rng() % (capacity + 3);
            check(pool.truncate(seq, end, error) && error.empty(), "modeled truncate succeeds");
            if (end < expected[seq].size()) expected[seq].resize(end);
            break;
        }
        default:
            pool.release(seq);
            expected[seq].clear();
            break;
        }
        invariants(pool, sequences);
        for (size_t owner = 0; owner < sequences; ++owner) {
            check(pool.mapping(owner).size() == expected[owner].size(), "modeled logical length");
            for (size_t page = 0; page < expected[owner].size(); ++page) {
                check(physical[static_cast<size_t>(pool.mapping(owner)[page])] == expected[owner][page],
                      "all live logical contents survive unrelated operations");
            }
        }
    }
    for (size_t seq = 0; seq < sequences; ++seq) pool.release(seq);
    check(pool.free_pages() == capacity, "model releases all references");
    invariants(pool, sequences);
}
} // namespace

int main() {
    growth_and_recycling();
    sharing_and_partial_page_cow();
    exhaustion_and_clone_semantics();
    invalid_inputs();
    model_test();
    std::printf("shared_kv_pages_test: %zu checks passed\n", checks);
}
