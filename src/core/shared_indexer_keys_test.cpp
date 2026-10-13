#include "strata/core/shared_indexer_keys.hpp"

#include <cstdio>
#include <cstdlib>
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
} // namespace

int main() {
    unsupported_geometry_and_invalid_calls();
    sparse_growth_recycle_and_publication();
    exclusive_rewrite_invalidates_in_place_page();
    shared_prefixes_divergence_and_cow_rewrite();
    clone_replacement_truncate_release_and_reuse();
    unpublished_source_cow_stays_unpublished();
    large_geometry_release_is_linear_in_released_pages();
    std::printf("shared_indexer_keys_test: %zu checks passed\n", checks);
}
