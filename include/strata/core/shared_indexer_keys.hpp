#pragma once

#include "strata/core/shared_kv_pages.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace strata::core {

struct SharedIndexerKeyGeometry {
    size_t physical_pages = 0;
    size_t qsa_layers = 0;
    int64_t page_size = 4;
    int64_t idx_block = 4;
};

// Host-only CPU model for completed QSA indexer-key metadata associated with
// the physical pages owned by SharedKvPages. This class never allocates physical
// page IDs and never owns authoritative references: SharedKvPages remains the
// allocator/refcount source of truth. Callers feed it the real SharedKvPages
// mappings, write ranges, COW copy records, and last-reference release records.
//
// Safe event ordering while SharedKvPages has no generation counter:
//   1. Run SharedKvPages::ensure(seq, writable_begin_page, end_page, ...).
//   2. Immediately call prepare_after_ensure() with the before/after mapping,
//      the same writable range, and the returned copy records. Every physical
//      page in [writable_begin_page, end_page) of the after mapping becomes
//      unpublished, whether it was exclusive in-place, a COW destination, or a
//      newly allocated page. Newly allocated pages outside an empty write range
//      are also unpublished. Pages before writable_begin_page are untouched.
//   3. If a COW payload copy or later device write fails, do not publish a key:
//      the source page metadata is unchanged and the destination is already
//      unpublished. If the payload succeeds and the complete block has been
//      recomputed, call publish_completed() explicitly.
//   4. Pass every last-reference release record from clone_prefix(), truncate()
//      or release() to invalidate_released() before the ID can be reused or
//      advertised to readers.
//
// Because IDs can be recycled and SharedKvPages does not expose generations,
// stale reuse cannot be inferred from an ID alone. The explicit prepare and
// invalidation events above are therefore part of the correctness contract.
// publish_completed() cannot prove that a page is live: caller precondition is
// that the physical page is currently present in a live SharedKvPages mapping
// and its complete page/block payload for that QSA layer has been written.
//
// Per-call complexity:
//   * prepare_after_ensure(): O((before + after + copies) expected mapping
//     validation + (changed pages + written pages + growth pages) * layers).
//     Validation allocates scratch proportional to mapping size, not pool size.
//   * publish_completed(): O(1).
//   * invalidate_released(): O(released pages * layers).
class SharedIndexerKeys {
public:
    static constexpr int64_t kSupportedPageSize = 4;
    static constexpr int64_t kSupportedIdxBlock = 4;

    explicit SharedIndexerKeys(SharedIndexerKeyGeometry geometry) : geometry_(geometry) {
        if (!supported_geometry(geometry.page_size, geometry.idx_block)) {
            throw std::invalid_argument("shared indexer keys require page_size == idx_block == 4");
        }
        if (geometry.physical_pages == 0) {
            throw std::invalid_argument("shared indexer keys need at least one physical page");
        }
        if (geometry.qsa_layers == 0) {
            throw std::invalid_argument("shared indexer keys need at least one QSA layer");
        }
        const size_t entries = geometry.physical_pages * geometry.qsa_layers;
        if (entries / geometry.qsa_layers != geometry.physical_pages) {
            throw std::invalid_argument("shared indexer key metadata is too large");
        }
        published_.assign(entries, 0);
    }

    static bool supported_geometry(int64_t page_size, int64_t idx_block) {
        return page_size == kSupportedPageSize && idx_block == kSupportedIdxBlock;
    }

    const SharedIndexerKeyGeometry& geometry() const { return geometry_; }
    size_t physical_pages() const { return geometry_.physical_pages; }
    size_t qsa_layers() const { return geometry_.qsa_layers; }

    bool published(int32_t physical_page, size_t qsa_layer) const {
        check_page_layer(physical_page, qsa_layer);
        return published_[index(size_t(physical_page), qsa_layer)] != 0;
    }

    size_t published_count() const {
        size_t count = 0;
        for (uint8_t value : published_) count += value != 0 ? 1u : 0u;
        return count;
    }

    // Explicit publication of a freshly computed completed key payload. This is
    // intentionally per physical page and per QSA layer: sequence-local tail,
    // dead and spare state are not representable here. Caller precondition:
    // physical_page is live in a SharedKvPages mapping and the complete block's
    // payload for qsa_layer has been written successfully.
    bool publish_completed(int32_t physical_page, size_t qsa_layer, std::string& error) {
        if (!valid_page_layer(physical_page, qsa_layer, error)) return false;
        published_[index(size_t(physical_page), qsa_layer)] = 1;
        error.clear();
        return true;
    }

    // Derive invalidation from the actual before/after SharedKvPages mapping,
    // the exact writable range passed to SharedKvPages::ensure(), and its copy
    // records. Every after-mapping page whose logical page index is in the
    // writable range is unpublished before payload writes. This includes COW
    // destinations and exclusive in-place rewrites; copied metadata is never
    // inherited because a completed key covers all four tokens of the page.
    bool prepare_after_ensure(const std::vector<int32_t>& before_mapping,
                              const std::vector<int32_t>& after_mapping,
                              size_t writable_begin_page, size_t end_page,
                              const std::vector<SharedKvPages::Copy>& copies,
                              std::string& error) {
        std::vector<SharedKvPages::Copy> changed;
        if (!validate_mapping(before_mapping, error) || !validate_mapping(after_mapping, error)) return false;
        if (after_mapping.size() < before_mapping.size()) {
            error = "shared indexer keys: ensure mapping unexpectedly shrank";
            return false;
        }
        if (writable_begin_page > end_page || end_page > after_mapping.size()) {
            error = "shared indexer keys: invalid writable page range";
            return false;
        }
        changed.reserve(copies.size());
        for (size_t page = 0; page < before_mapping.size(); ++page) {
            if (before_mapping[page] != after_mapping[page]) {
                if (page < writable_begin_page || page >= end_page) {
                    error = "shared indexer keys: mapping changed outside writable range";
                    return false;
                }
                changed.push_back({before_mapping[page], after_mapping[page]});
            }
        }
        if (changed.size() != copies.size()) {
            error = "shared indexer keys: COW copy records do not match mapping changes";
            return false;
        }
        for (size_t i = 0; i < copies.size(); ++i) {
            if (copies[i].from != changed[i].from || copies[i].to != changed[i].to) {
                error = "shared indexer keys: COW copy record order does not match mapping changes";
                return false;
            }
            if (!valid_page(copies[i].from, error) || !valid_page(copies[i].to, error)) return false;
        }

        // Validate first; mutate only after every input is accepted.
        error.clear();
        const size_t write_end = end_page < after_mapping.size() ? end_page : after_mapping.size();
        for (size_t page = writable_begin_page; page < write_end; ++page) reset_page(size_t(after_mapping[page]));
        // A caller may reserve/grow pages without immediately writing them
        // (writable_begin_page == end_page). New/recycled IDs are still invalid
        // until an explicit completed-payload publication event.
        for (size_t page = before_mapping.size(); page < after_mapping.size(); ++page) {
            if (page < writable_begin_page || page >= write_end) reset_page(size_t(after_mapping[page]));
        }
        return true;
    }

    // Apply the last-reference IDs returned by SharedKvPages::clone_prefix,
    // truncate or release. Surviving shared siblings are intentionally not named
    // by SharedKvPages and therefore retain their physical-page metadata.
    bool invalidate_released(const std::vector<int32_t>& released, std::string& error) {
        for (int32_t page : released) {
            if (!valid_page(page, error)) return false;
        }
        for (int32_t page : released) reset_page(size_t(page));
        error.clear();
        return true;
    }

private:
    size_t index(size_t physical_page, size_t qsa_layer) const {
        return physical_page * geometry_.qsa_layers + qsa_layer;
    }

    bool valid_page(int32_t physical_page, std::string& error) const {
        if (physical_page < 0 || size_t(physical_page) >= geometry_.physical_pages) {
            error = "shared indexer keys: physical page out of range";
            return false;
        }
        return true;
    }

    bool valid_page_layer(int32_t physical_page, size_t qsa_layer, std::string& error) const {
        if (!valid_page(physical_page, error)) return false;
        if (qsa_layer >= geometry_.qsa_layers) {
            error = "shared indexer keys: QSA layer out of range";
            return false;
        }
        return true;
    }

    void check_page_layer(int32_t physical_page, size_t qsa_layer) const {
        std::string error;
        if (!valid_page_layer(physical_page, qsa_layer, error)) throw std::out_of_range(error);
    }

    bool validate_mapping(const std::vector<int32_t>& mapping, std::string& error) const {
        std::unordered_set<int32_t> seen;
        seen.reserve(mapping.size());
        for (int32_t page : mapping) {
            if (!valid_page(page, error)) return false;
            if (!seen.insert(page).second) {
                error = "shared indexer keys: duplicate physical page in mapping";
                return false;
            }
        }
        return true;
    }

    void reset_page(size_t physical_page) {
        for (size_t layer = 0; layer < geometry_.qsa_layers; ++layer) {
            published_[index(physical_page, layer)] = 0;
        }
    }

    SharedIndexerKeyGeometry geometry_;
    std::vector<uint8_t> published_;
};

} // namespace strata::core
