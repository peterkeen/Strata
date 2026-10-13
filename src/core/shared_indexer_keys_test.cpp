#include "strata/core/shared_indexer_keys.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using strata::core::SharedIndexerKeyGeometry;
using strata::core::SharedIndexerKeys;
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

std::vector<int32_t> mapping(const SharedKvPages& pages, size_t seq) {
    return pages.mapping(seq);
}

void ensure_with_keys(SharedKvPages& pages, SharedIndexerKeys& keys, size_t seq,
                      size_t begin, size_t end, std::vector<SharedKvPages::Copy>& copies,
                      std::string& error) {
    const auto before = mapping(pages, seq);
    check(pages.ensure(seq, begin, end, copies, error), "SharedKvPages ensure succeeds");
    check(keys.prepare_after_ensure(before, pages.mapping(seq), begin, end, copies, error),
          "indexer metadata observes real ensure range, mapping and copies");
    check(error.empty(), "metadata prepare clears error");
}

void clone_with_keys(SharedKvPages& pages, SharedIndexerKeys& keys, size_t from, size_t to,
                     size_t end, std::string& error) {
    std::vector<int32_t> released;
    check(pages.clone_prefix(from, to, end, error, &released), "SharedKvPages clone succeeds");
    check(keys.invalidate_released(released, error), "indexer metadata invalidates clone replacement releases");
    check(error.empty(), "clone invalidation clears error");
}

void truncate_with_keys(SharedKvPages& pages, SharedIndexerKeys& keys, size_t seq,
                        size_t end, std::string& error) {
    std::vector<int32_t> released;
    check(pages.truncate(seq, end, error, &released), "SharedKvPages truncate succeeds");
    check(keys.invalidate_released(released, error), "indexer metadata invalidates truncate releases");
    check(error.empty(), "truncate invalidation clears error");
}

void release_with_keys(SharedKvPages& pages, SharedIndexerKeys& keys, size_t seq, std::string& error) {
    std::vector<int32_t> released;
    pages.release(seq, &released);
    check(keys.invalidate_released(released, error), "indexer metadata invalidates release records");
    check(error.empty(), "release invalidation clears error");
}

void publish_all_layers(SharedIndexerKeys& keys, int32_t page, std::string& error) {
    for (size_t layer = 0; layer < keys.qsa_layers(); ++layer) {
        check(keys.publish_completed(page, layer, error), "publish completed key payload by page/layer");
        check(error.empty(), "publish completed clears error");
    }
}

void check_all_layers(const SharedIndexerKeys& keys, int32_t page, bool expected, const char* label) {
    for (size_t layer = 0; layer < keys.qsa_layers(); ++layer) check(keys.published(page, layer) == expected, label);
}

bool copies_match(const std::vector<SharedKvPages::Copy>& copies, const std::vector<SharedKvPages::Copy>& expected) {
    if (copies.size() != expected.size()) return false;
    for (size_t i = 0; i < copies.size(); ++i) {
        if (copies[i].from != expected[i].from || copies[i].to != expected[i].to) return false;
    }
    return true;
}

void unsupported_geometry_and_invalid_calls() {
    check(SharedIndexerKeys::supported_geometry(4, 4), "current page/block geometry is supported");
    check(!SharedIndexerKeys::supported_geometry(8, 4), "larger page geometry is rejected initially");
    check(!SharedIndexerKeys::supported_geometry(4, 2), "mismatched idx_block geometry is rejected initially");
    check_throws<std::invalid_argument>([] { SharedIndexerKeys keys({4, 2, 8, 4}); },
                                        "unsupported page size throws");
    check_throws<std::invalid_argument>([] { SharedIndexerKeys keys({4, 2, 4, 2}); },
                                        "unsupported idx_block throws");
    check_throws<std::invalid_argument>([] { SharedIndexerKeys keys({0, 2, 4, 4}); },
                                        "zero page metadata capacity throws");
    check_throws<std::invalid_argument>([] { SharedIndexerKeys keys({4, 0, 4, 4}); },
                                        "zero layer metadata capacity throws");

    SharedIndexerKeys keys({3, 2, 4, 4});
    std::string error;
    check(!keys.publish_completed(3, 0, error) && !error.empty(), "publish rejects out-of-range page");
    check(!keys.publish_completed(0, 2, error) && !error.empty(), "publish rejects out-of-range layer");
    check_throws<std::out_of_range>([&] { (void)keys.published(-1, 0); }, "query rejects negative page");
    check_throws<std::out_of_range>([&] { (void)keys.published(0, 2); }, "query rejects invalid layer");
    check(keys.publish_completed(0, 0, error), "setup valid metadata before invalid prepare");
    check(!keys.prepare_after_ensure(std::vector<int32_t>{0, 0}, std::vector<int32_t>{0, 1}, 0, 1, {}, error) &&
              !error.empty(),
          "prepare rejects duplicate before mapping");
    check(keys.published(0, 0), "invalid prepare leaves existing publication unchanged");
    check(!keys.prepare_after_ensure(std::vector<int32_t>{0}, std::vector<int32_t>{0}, 2, 1, {}, error) &&
              !error.empty(),
          "prepare rejects inverted writable range");
    check(keys.published(0, 0), "invalid range leaves existing publication unchanged");
    check(!keys.prepare_after_ensure(std::vector<int32_t>{0}, std::vector<int32_t>{1}, 0, 0,
                                     std::vector<SharedKvPages::Copy>{{0, 1}}, error) &&
              !error.empty(),
          "prepare rejects mapping changes outside writable range");
    check(keys.published(0, 0), "invalid copy range leaves existing publication unchanged");
    check(!keys.invalidate_released(std::vector<int32_t>{-1}, error) && !error.empty(),
          "release invalidation rejects negative ID");
    check(keys.published(0, 0), "invalid release invalidation leaves existing publication unchanged");
}

void sparse_growth_recycle_and_publication() {
    SharedKvPages pages(10, 3);
    SharedIndexerKeys keys({pages.capacity(), 3, 4, 4});
    std::vector<SharedKvPages::Copy> copies;
    std::string error;

    ensure_with_keys(pages, keys, 0, 0, 4, copies, error);
    check(pages.mapping(0) == std::vector<int32_t>({0, 1, 2, 3}), "initial real allocator mapping is dense");
    check(keys.published_count() == 0, "new pages are unpublished until payload publication");
    publish_all_layers(keys, 0, error);
    publish_all_layers(keys, 2, error);
    check(keys.published_count() == 6, "publication is tracked per physical page and per layer");

    truncate_with_keys(pages, keys, 0, 2, error);
    check_all_layers(keys, 0, true, "surviving page keeps completed metadata after truncate");
    check_all_layers(keys, 2, false, "last-reference truncated page loses completed metadata");
    check_all_layers(keys, 3, false, "last-reference truncated tail page is invalidated");

    ensure_with_keys(pages, keys, 1, 0, 2, copies, error);
    check(pages.mapping(1) == std::vector<int32_t>({2, 3}), "freed IDs are recycled before unused IDs");
    check_all_layers(keys, 2, false, "recycled ID starts unpublished despite old completed payload");
    publish_all_layers(keys, 2, error);

    ensure_with_keys(pages, keys, 0, 2, 5, copies, error);
    check(pages.mapping(0) == std::vector<int32_t>({0, 1, 4, 5, 6}), "real allocator can produce sparse mappings");
    check_all_layers(keys, 0, true, "write range starting at page two leaves earlier page zero published");
    check_all_layers(keys, 1, false, "unpublished earlier page remains unpublished");
    check_all_layers(keys, 4, false, "sparse new physical page is initially unpublished");
    check_all_layers(keys, 6, false, "sparse tail physical page is initially unpublished");
    check(keys.publish_completed(6, 1, error), "publish one sparse physical page/layer");
    check(!keys.published(6, 0) && keys.published(6, 1) && !keys.published(6, 2),
          "publication does not bleed across QSA layers");
}

void exclusive_rewrite_invalidates_in_place_page() {
    SharedKvPages pages(4, 1);
    SharedIndexerKeys keys({pages.capacity(), 2, 4, 4});
    std::vector<SharedKvPages::Copy> copies;
    std::string error;

    ensure_with_keys(pages, keys, 0, 0, 2, copies, error);
    const auto page0 = pages.mapping(0)[0];
    const auto page1 = pages.mapping(0)[1];
    publish_all_layers(keys, page0, error);
    publish_all_layers(keys, page1, error);
    ensure_with_keys(pages, keys, 0, 1, 2, copies, error);
    check(copies.empty(), "exclusive in-place rewrite emits no allocator copy records");
    check_all_layers(keys, page0, true, "exclusive rewrite leaves earlier completed page published");
    check_all_layers(keys, page1, false, "exclusive rewrite invalidates stale completed key in place");
    publish_all_layers(keys, page1, error);
    check_all_layers(keys, page1, true, "exclusive rewritten page republishes only after recomputation");
}

void shared_prefixes_divergence_and_cow_rewrite() {
    SharedKvPages pages(9, 3);
    SharedIndexerKeys keys({pages.capacity(), 3, 4, 4});
    std::vector<SharedKvPages::Copy> copies;
    std::string error;

    ensure_with_keys(pages, keys, 0, 0, 3, copies, error);
    for (int32_t page : pages.mapping(0)) publish_all_layers(keys, page, error);
    clone_with_keys(pages, keys, 0, 1, 3, error);
    check(pages.mapping(1) == pages.mapping(0), "clone shares physical prefix");
    check_all_layers(keys, pages.mapping(1)[2], true, "shared clone observes the same physical completed key");

    const auto before = mapping(pages, 1);
    check(pages.ensure(1, 2, 4, copies, error), "divergent sibling ensure performs COW and growth");
    check(copies_match(copies, {{before[2], 3}}), "real COW copy record names source and destination");
    check(keys.prepare_after_ensure(before, pages.mapping(1), 2, 4, copies, error),
          "metadata invalidates COW destination from real writable range");
    check(pages.mapping(1) == std::vector<int32_t>({0, 1, 3, 4}), "divergent sibling has private tail and growth page");
    check_all_layers(keys, before[0], true, "COW rewrite leaves earlier shared page zero published");
    check_all_layers(keys, before[1], true, "COW rewrite leaves earlier shared page one published");
    check_all_layers(keys, before[2], true, "COW rewrite does not mutate source sibling metadata");
    check_all_layers(keys, 3, false, "COW destination is unpublished before recomputation");
    check_all_layers(keys, 4, false, "new growth page has no shared completed metadata");
    publish_all_layers(keys, 3, error);
    check_all_layers(keys, 3, true, "COW destination republishes only after recomputation");
    check(keys.publish_completed(4, 0, error), "new divergent page can publish an independent completed layer");
    check(keys.published(4, 0) && !keys.published(4, 1) && !keys.published(4, 2),
          "independent publication is still per layer");
    check_all_layers(keys, before[2], true, "source remains valid after destination recomputation");

    clone_with_keys(pages, keys, 0, 2, 3, error);
    const auto failed_before = mapping(pages, 2);
    check(pages.ensure(2, 1, 3, copies, error), "second sibling COW setup succeeds");
    check(copies_match(copies, {{failed_before[1], 5}, {failed_before[2], 6}}),
          "failed-copy fixture has two real COW records");
    check(keys.prepare_after_ensure(failed_before, pages.mapping(2), 1, 3, copies, error),
          "failed-copy fixture invalidates destinations before payload copy");
    check_all_layers(keys, failed_before[0], true, "failed-copy untouched prefix remains valid");
    check_all_layers(keys, failed_before[1], true, "failed-copy source one stays valid");
    check_all_layers(keys, failed_before[2], true, "failed-copy source two stays valid");
    check_all_layers(keys, 5, false, "failed COW cannot advertise first destination");
    check_all_layers(keys, 6, false, "failed COW cannot advertise second destination");
}

void clone_replacement_truncate_release_and_reuse() {
    SharedKvPages pages(8, 3);
    SharedIndexerKeys keys({pages.capacity(), 2, 4, 4});
    std::vector<SharedKvPages::Copy> copies;
    std::string error;

    ensure_with_keys(pages, keys, 0, 0, 2, copies, error);
    publish_all_layers(keys, 0, error);
    publish_all_layers(keys, 1, error);
    ensure_with_keys(pages, keys, 1, 0, 2, copies, error);
    publish_all_layers(keys, 2, error);
    publish_all_layers(keys, 3, error);
    clone_with_keys(pages, keys, 0, 1, 2, error);
    check(pages.mapping(1) == pages.mapping(0), "clone replacement shares source prefix");
    check_all_layers(keys, 0, true, "clone replacement keeps source page metadata");
    check_all_layers(keys, 1, true, "clone replacement keeps second source page metadata");
    check_all_layers(keys, 2, false, "clone replacement invalidates released destination page");
    check_all_layers(keys, 3, false, "clone replacement invalidates released destination tail");

    release_with_keys(pages, keys, 0, error);
    check_all_layers(keys, 0, true, "releasing one shared sibling keeps page zero metadata");
    check_all_layers(keys, 1, true, "releasing one shared sibling keeps page one metadata");
    release_with_keys(pages, keys, 1, error);
    check_all_layers(keys, 0, false, "last-reference release invalidates page zero metadata");
    check_all_layers(keys, 1, false, "last-reference release invalidates page one metadata");

    ensure_with_keys(pages, keys, 2, 0, 2, copies, error);
    check(pages.mapping(2) == std::vector<int32_t>({0, 1}), "last-reference IDs are recycled deterministically");
    check_all_layers(keys, 0, false, "recycled page zero does not inherit stale key metadata");
    check_all_layers(keys, 1, false, "recycled page one does not inherit stale key metadata");
    publish_all_layers(keys, 0, error);
    truncate_with_keys(pages, keys, 2, 1, error);
    check_all_layers(keys, 0, true, "truncate preserves retained head metadata");
    check_all_layers(keys, 1, false, "truncate invalidates recycled tail metadata");
    release_with_keys(pages, keys, 2, error);
    check_all_layers(keys, 0, false, "final release invalidates retained head metadata");
    check(pages.free_pages() == pages.capacity(), "allocator fully recovers after metadata lifecycle test");
}

void unpublished_source_cow_stays_unpublished() {
    SharedKvPages pages(5, 2);
    SharedIndexerKeys keys({pages.capacity(), 2, 4, 4});
    std::vector<SharedKvPages::Copy> copies;
    std::string error;

    ensure_with_keys(pages, keys, 0, 0, 1, copies, error);
    clone_with_keys(pages, keys, 0, 1, 1, error);
    const auto before = mapping(pages, 1);
    check(pages.ensure(1, 0, 1, copies, error), "COW of unpublished source succeeds in allocator");
    check(copies_match(copies, {{before[0], 1}}), "unpublished-source fixture has real COW copy");
    check(keys.prepare_after_ensure(before, pages.mapping(1), 0, 1, copies, error),
          "metadata invalidates unpublished-source COW destination");
    check_all_layers(keys, before[0], false, "unpublished source remains unpublished");
    check_all_layers(keys, 1, false, "destination copied from unpublished source remains unpublished");
}

void large_geometry_release_is_linear_in_released_pages() {
    constexpr size_t pages_count = 65536;
    constexpr size_t layers = 12;
    SharedKvPages pages(pages_count, 1);
    SharedIndexerKeys keys({pages.capacity(), layers, 4, 4});
    std::vector<SharedKvPages::Copy> copies;
    std::string error;

    ensure_with_keys(pages, keys, 0, 0, pages_count, copies, error);
    check(pages.mapping(0).size() == pages_count, "large mapping reserves all physical pages");
    check(keys.publish_completed(0, 0, error), "large mapping publishes first page/layer");
    check(keys.publish_completed(int32_t(pages_count - 1), layers - 1, error), "large mapping publishes last page/layer");
    check(keys.published_count() == 2, "large mapping publication setup is sparse");
    release_with_keys(pages, keys, 0, error);
    check(pages.free_pages() == pages_count, "large mapping release returns every page");
    check(keys.published_count() == 0, "large release invalidates all completed-key metadata without pool-sized scans per page");
    ensure_with_keys(pages, keys, 0, 0, 1, copies, error);
    check(pages.mapping(0) == std::vector<int32_t>({0}), "large release preserves allocator reuse order");
}

struct StressStats {
    uint64_t ensure_success = 0;
    uint64_t ensure_failure = 0;
    uint64_t payload_success = 0;
    uint64_t payload_failure = 0;
    uint64_t publish = 0;
    uint64_t clone = 0;
    uint64_t self_clone = 0;
    uint64_t truncate = 0;
    uint64_t release = 0;
    std::array<uint64_t, 4> write_offsets{{0, 0, 0, 0}};
};

uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const uint64_t parsed = std::strtoull(value, &end, 0);
    return end && *end == '\0' ? parsed : fallback;
}

struct StressContext {
    uint64_t seed = 0;
    size_t op = 0;
    std::string desc;
    SharedKvPages* pages = nullptr;
    SharedIndexerKeys* keys = nullptr;
    std::vector<std::vector<int32_t>>* seq_maps = nullptr;
    std::vector<size_t>* seq_tokens = nullptr;
    std::vector<std::vector<uint8_t>>* expected = nullptr;
};

[[noreturn]] void stress_fail(const StressContext& ctx, const char* label) {
    std::fprintf(stderr, "FAIL: randomized stress: %s\nseed=%llu op=%zu desc=%s\n",
                 label, static_cast<unsigned long long>(ctx.seed), ctx.op, ctx.desc.c_str());
    if (ctx.pages && ctx.seq_maps && ctx.seq_tokens && ctx.expected) {
        std::fprintf(stderr, "free=%zu used=%zu published=%zu\n", ctx.pages->free_pages(), ctx.pages->used_pages(),
                     ctx.keys ? ctx.keys->published_count() : 0);
        for (size_t seq = 0; seq < ctx.seq_maps->size(); ++seq) {
            std::fprintf(stderr, "seq%zu tokens=%zu map=", seq, (*ctx.seq_tokens)[seq]);
            for (int32_t page : (*ctx.seq_maps)[seq]) std::fprintf(stderr, "%d,", page);
            std::fprintf(stderr, "\n");
        }
        std::fprintf(stderr, "expected_valid_pages=");
        for (size_t page = 0; page < ctx.expected->size(); ++page) {
            bool any = false;
            for (uint8_t valid : (*ctx.expected)[page]) any = any || valid != 0;
            if (!any) continue;
            std::fprintf(stderr, "%zu:", page);
            for (uint8_t valid : (*ctx.expected)[page]) std::fprintf(stderr, "%u", unsigned(valid != 0));
            std::fprintf(stderr, ",");
        }
        std::fprintf(stderr, "\n");
    }
    std::exit(1);
}

void stress_require(bool ok, const StressContext& ctx, const char* label) {
    ++checks;
    if (!ok) stress_fail(ctx, label);
}

std::vector<uint8_t> live_pages_from_maps(const std::vector<std::vector<int32_t>>& maps, size_t page_count) {
    std::vector<uint8_t> live(page_count, 0);
    for (const auto& map : maps) {
        for (int32_t page : map) live[size_t(page)] = 1;
    }
    return live;
}

void stress_invariants(const StressContext& ctx) {
    const auto& maps = *ctx.seq_maps;
    const auto& expected = *ctx.expected;
    const auto live = live_pages_from_maps(maps, expected.size());
    for (size_t seq = 0; seq < maps.size(); ++seq) {
        stress_require(ctx.pages->mapping(seq) == maps[seq], ctx, "oracle mapping mirrors SharedKvPages public mapping");
        stress_require((*ctx.seq_tokens)[seq] <= maps[seq].size() * size_t(4), ctx,
                       "logical tokens fit reserved page mapping");
    }
    for (size_t page = 0; page < expected.size(); ++page) {
        for (size_t layer = 0; layer < expected[page].size(); ++layer) {
            const bool actual = ctx.keys->published(int32_t(page), layer);
            stress_require(actual == (expected[page][layer] != 0), ctx, "metadata matches independent oracle");
            stress_require(!actual || live[page] != 0, ctx, "published page is live in some public mapping");
        }
    }
}

void stress_invalidate_page(std::vector<std::vector<uint8_t>>& expected, int32_t page) {
    std::fill(expected[size_t(page)].begin(), expected[size_t(page)].end(), uint8_t{0});
}

void stress_invalidate_released(std::vector<std::vector<uint8_t>>& expected, const std::vector<int32_t>& released) {
    for (int32_t page : released) stress_invalidate_page(expected, page);
}

void stress_prepare_oracle(std::vector<std::vector<uint8_t>>& expected,
                           const std::vector<int32_t>& before, const std::vector<int32_t>& after,
                           size_t begin_page, size_t end_page) {
    for (size_t page = begin_page; page < end_page && page < after.size(); ++page) stress_invalidate_page(expected, after[page]);
    for (size_t page = before.size(); page < after.size(); ++page) {
        if (page < begin_page || page >= end_page) stress_invalidate_page(expected, after[page]);
    }
}

void stress_publish_complete_pages(SharedIndexerKeys& keys, std::vector<std::vector<uint8_t>>& expected,
                                   const std::vector<int32_t>& map, size_t start_token, size_t end_token,
                                   std::mt19937_64& rng, StressStats& stats, StressContext& ctx,
                                   std::string& error) {
    for (size_t logical = start_token / 4; logical < (end_token + 3) / 4 && logical < map.size(); ++logical) {
        const size_t block_begin = logical * size_t(4);
        const size_t block_end = block_begin + size_t(4);
        if (start_token > block_begin || end_token < block_end) continue;
        const int32_t physical = map[logical];
        for (size_t layer = 0; layer < keys.qsa_layers(); ++layer) {
            if ((rng() & 3u) == 0u) continue; // Leave some complete blocks deliberately unpublished.
            stress_require(keys.publish_completed(physical, layer, error), ctx, "publish complete live block succeeds");
            expected[size_t(physical)][layer] = 1;
            ++stats.publish;
        }
    }
}

void randomized_stress_one_seed(uint64_t seed, size_t iterations, StressStats& stats) {
    constexpr size_t capacity = 48;
    constexpr size_t sequences = 5;
    constexpr size_t layers = 4;
    SharedKvPages pages(capacity, sequences);
    SharedIndexerKeys keys({pages.capacity(), layers, 4, 4});
    std::vector<std::vector<int32_t>> seq_maps(sequences);
    std::vector<size_t> seq_tokens(sequences, 0);
    std::vector<std::vector<uint8_t>> expected(capacity, std::vector<uint8_t>(layers, 0));
    std::vector<SharedKvPages::Copy> copies;
    std::string error;
    std::mt19937_64 rng(seed);
    StressContext ctx{seed, 0, "initial", &pages, &keys, &seq_maps, &seq_tokens, &expected};

    for (size_t op = 0; op < iterations; ++op) {
        ctx.op = op;
        copies.clear();
        error.clear();
        const size_t choice = size_t(rng() % 100);
        if (choice < 52) {
            const size_t seq = size_t(rng() % sequences);
            const size_t current_tokens = seq_tokens[seq];
            size_t start_token = 0;
            if (current_tokens == 0 || (rng() % 5) == 0) {
                start_token = size_t(rng() % (capacity * 4));
            } else if ((rng() & 1u) != 0u) {
                const size_t page = size_t(rng() % (seq_maps[seq].empty() ? 1 : seq_maps[seq].size()));
                start_token = page * size_t(4) + size_t(rng() % 4); // offsets 0/1/2/3 within an owned page.
            } else {
                start_token = size_t(rng() % (current_tokens + 1));
            }
            if ((rng() % 7) == 0) start_token = (start_token / 4) * 4; // block boundary.
            const size_t length = size_t(rng() % 13); // includes reservation/no-payload attempts.
            const size_t end_token = std::min(capacity * size_t(4), start_token + length);
            const size_t begin_page = start_token / 4;
            const size_t end_page = (end_token + 3) / 4;
            ++stats.write_offsets[start_token % 4];
            const bool payload_success = (rng() % 5) != 0 && end_token > start_token;
            ctx.desc = "ensure seq=" + std::to_string(seq) + " tokens=" + std::to_string(start_token) + ".." +
                       std::to_string(end_token) + " pages=" + std::to_string(begin_page) + ".." +
                       std::to_string(end_page) + (payload_success ? " payload=ok" : " payload=fail_or_empty");
            const auto before_map = seq_maps[seq];
            const auto before_all_maps = seq_maps;
            const auto before_tokens = seq_tokens;
            const auto before_expected = expected;
            const bool ok = pages.ensure(seq, begin_page, end_page, copies, error);
            if (!ok) {
                ++stats.ensure_failure;
                stress_require(pages.mapping(seq) == before_map, ctx, "failed ensure leaves sequence mapping unchanged");
                stress_require(seq_maps == before_all_maps && seq_tokens == before_tokens && expected == before_expected,
                               ctx, "failed ensure leaves oracle unchanged");
                stress_invariants(ctx);
                continue;
            }
            ++stats.ensure_success;
            seq_maps[seq] = pages.mapping(seq);
            stress_require(keys.prepare_after_ensure(before_map, seq_maps[seq], begin_page, end_page, copies, error),
                           ctx, "metadata prepare succeeds after successful ensure");
            stress_prepare_oracle(expected, before_map, seq_maps[seq], begin_page, end_page);
            for (const auto& copy : copies) {
                for (size_t layer = 0; layer < layers; ++layer) {
                    stress_require(expected[size_t(copy.from)][layer] == before_expected[size_t(copy.from)][layer],
                                   ctx, "COW source oracle remains unchanged after prepare");
                    stress_require(expected[size_t(copy.to)][layer] == 0, ctx,
                                   "COW destination oracle is invalid before explicit publish");
                }
            }
            if (payload_success) {
                ++stats.payload_success;
                seq_tokens[seq] = std::max(seq_tokens[seq], end_token);
                stress_publish_complete_pages(keys, expected, seq_maps[seq], start_token, end_token, rng, stats, ctx, error);
            } else {
                ++stats.payload_failure;
                for (size_t page = begin_page; page < end_page && page < seq_maps[seq].size(); ++page) {
                    for (size_t layer = 0; layer < layers; ++layer) {
                        stress_require(!keys.published(seq_maps[seq][page], layer), ctx,
                                       "failed stress payload leaves destination unpublished");
                    }
                }
            }
        } else if (choice < 70) {
            const size_t from = size_t(rng() % sequences);
            const size_t to = size_t(rng() % sequences);
            const size_t max_end = seq_maps[from].size();
            const size_t end = max_end == 0 ? 0 : size_t(rng() % (max_end + 1));
            ctx.desc = "clone from=" + std::to_string(from) + " to=" + std::to_string(to) +
                       " end=" + std::to_string(end);
            std::vector<int32_t> released;
            stress_require(pages.clone_prefix(from, to, end, error, &released), ctx, "clone_prefix succeeds");
            stress_invalidate_released(expected, released);
            stress_require(keys.invalidate_released(released, error), ctx, "clone release invalidation succeeds");
            seq_maps[to] = pages.mapping(to);
            seq_tokens[to] = std::min(seq_tokens[from], end * size_t(4));
            if (from == to) ++stats.self_clone;
            else ++stats.clone;
        } else if (choice < 85) {
            const size_t seq = size_t(rng() % sequences);
            const size_t end = size_t(rng() % (capacity + 1));
            ctx.desc = "truncate seq=" + std::to_string(seq) + " end=" + std::to_string(end);
            std::vector<int32_t> released;
            stress_require(pages.truncate(seq, end, error, &released), ctx, "truncate succeeds");
            stress_invalidate_released(expected, released);
            stress_require(keys.invalidate_released(released, error), ctx, "truncate release invalidation succeeds");
            seq_maps[seq] = pages.mapping(seq);
            seq_tokens[seq] = std::min(seq_tokens[seq], end * size_t(4));
            ++stats.truncate;
        } else {
            const size_t seq = size_t(rng() % sequences);
            ctx.desc = "release seq=" + std::to_string(seq);
            std::vector<int32_t> released;
            pages.release(seq, &released);
            stress_invalidate_released(expected, released);
            stress_require(keys.invalidate_released(released, error), ctx, "release invalidation succeeds");
            seq_maps[seq] = pages.mapping(seq);
            seq_tokens[seq] = 0;
            ++stats.release;
        }
        stress_invariants(ctx);
    }
}

void randomized_stress() {
    const size_t default_iterations = 3000;
    const uint64_t default_seeds[] = {0x51a7a2a6ULL, 0xc0ffee1234ULL, 0xbadc0de5678ULL, 0x123456789abcdef0ULL};
    const uint64_t override_seed = env_u64("STRATA_P1B_SEED", 0);
    const size_t iterations = size_t(env_u64("STRATA_P1B_ITERS", default_iterations));
    StressStats stats;
    if (override_seed != 0) {
        randomized_stress_one_seed(override_seed, iterations, stats);
        std::printf("shared_indexer_keys_randomized: seeds=%llu iterations=%zu ops=%zu ",
                    static_cast<unsigned long long>(override_seed), iterations, iterations);
    } else {
        for (uint64_t seed : default_seeds) randomized_stress_one_seed(seed, iterations, stats);
        std::printf("shared_indexer_keys_randomized: seeds=0x51a7a2a6,0xc0ffee1234,0xbadc0de5678,0x123456789abcdef0 iterations=%zu ops=%zu ",
                    iterations, iterations * (sizeof(default_seeds) / sizeof(default_seeds[0])));
    }
    std::printf("ensure_ok=%llu ensure_fail=%llu payload_ok=%llu payload_fail=%llu publish=%llu clone=%llu self_clone=%llu truncate=%llu release=%llu offsets=[%llu,%llu,%llu,%llu]\n",
                static_cast<unsigned long long>(stats.ensure_success),
                static_cast<unsigned long long>(stats.ensure_failure),
                static_cast<unsigned long long>(stats.payload_success),
                static_cast<unsigned long long>(stats.payload_failure),
                static_cast<unsigned long long>(stats.publish),
                static_cast<unsigned long long>(stats.clone),
                static_cast<unsigned long long>(stats.self_clone),
                static_cast<unsigned long long>(stats.truncate),
                static_cast<unsigned long long>(stats.release),
                static_cast<unsigned long long>(stats.write_offsets[0]),
                static_cast<unsigned long long>(stats.write_offsets[1]),
                static_cast<unsigned long long>(stats.write_offsets[2]),
                static_cast<unsigned long long>(stats.write_offsets[3]));
}
} // namespace

int main() {
    unsupported_geometry_and_invalid_calls();
    sparse_growth_recycle_and_publication();
    exclusive_rewrite_invalidates_in_place_page();
    shared_prefixes_divergence_and_cow_rewrite();
    clone_replacement_truncate_release_and_reuse();
    unpublished_source_cow_stays_unpublished();
    large_geometry_release_is_linear_in_released_pages();
    randomized_stress();
    std::printf("shared_indexer_keys_test: %zu checks passed\n", checks);
}
