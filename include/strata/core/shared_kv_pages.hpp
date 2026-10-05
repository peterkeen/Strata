#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

// Host-only ownership of a single physical KV page pool. Not thread-safe.
// Pages are initially allocated in ascending ID order; freed pages are recycled
// LIFO. Tails are released from the last logical page toward the first.
// Mapping entries are unique within a sequence, but may be shared across sequences.
class SharedKvPages {
public:
    struct Copy {
        int32_t from;
        int32_t to;
    };

    // Empty pools and zero sequence counts are configuration errors. Validate
    // before allocating, including capacities whose IDs cannot fit in int32_t.
    SharedKvPages(size_t capacity_pages, size_t sequence_count) {
        if (capacity_pages == 0 ||
            capacity_pages > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            throw std::invalid_argument("SharedKvPages capacity must be in [1, INT32_MAX]");
        }
        if (sequence_count == 0) {
            throw std::invalid_argument("SharedKvPages needs at least one sequence");
        }
        mappings_.resize(sequence_count);
        refcounts_.resize(capacity_pages, 0);
        free_.reserve(capacity_pages);
        for (size_t page = capacity_pages; page != 0; --page) {
            free_.push_back(static_cast<int32_t>(page - 1));
        }
    }

    // Ensure [0, end_page) exists and is private wherever it overlaps the write
    // range. Existing pages below writable_begin_page stay shared. Pages beyond
    // end_page are retained; use truncate() to discard them.
    //
    // On invalid input or insufficient capacity: false, nonempty error, empty
    // copies, and unchanged mappings/refcounts/free-list order. On success the
    // ownership change is synchronous; the caller MUST execute every returned
    // physical-page copy before writing the newly private pages on the GPU.
    // Newly allocated (rather than copied) pages have unspecified contents.
    bool ensure(size_t seq, size_t writable_begin_page, size_t end_page,
                std::vector<Copy>& copies, std::string& error) {
        copies.clear();
        if (seq >= mappings_.size()) {
            error = "SharedKvPages sequence index out of range";
            return false;
        }
        if (writable_begin_page > end_page || end_page > capacity()) {
            error = "SharedKvPages invalid page range";
            return false;
        }

        auto& pages = mappings_[seq];
        const size_t existing_end = pages.size() < end_page ? pages.size() : end_page;
        size_t copy_count = 0;
        for (size_t page = writable_begin_page; page < existing_end; ++page) {
            if (refcounts_[static_cast<size_t>(pages[page])] > 1) ++copy_count;
        }
        const size_t growth = end_page > pages.size() ? end_page - pages.size() : 0;
        // Both counts describe disjoint logical pages and their sum <= end_page.
        if (growth + copy_count > free_.size()) {
            error = "SharedKvPages insufficient physical pages";
            return false;
        }

        // Do all potentially throwing allocations before changing ownership.
        pages.reserve(end_page);
        copies.reserve(copy_count);
        error.clear();
        for (size_t page = writable_begin_page; page < existing_end; ++page) {
            const int32_t old = pages[page];
            if (refcounts_[static_cast<size_t>(old)] > 1) {
                const int32_t fresh = allocate();
                --refcounts_[static_cast<size_t>(old)];
                pages[page] = fresh;
                copies.push_back({old, fresh});
            }
        }
        while (pages.size() < end_page) pages.push_back(allocate());
        return true;
    }

    // Replace the destination with the complete source prefix [0, end_page).
    // The source must already contain that prefix: validate before releasing any
    // destination pages. A self-clone is a truncate to end_page (including zero),
    // not a no-op; an incomplete self-prefix is rejected without mutation.
    bool clone_prefix(size_t from, size_t to, size_t end_page, std::string& error) {
        if (from >= mappings_.size() || to >= mappings_.size()) {
            error = "SharedKvPages sequence index out of range";
            return false;
        }
        if (end_page > mappings_[from].size()) {
            error = "SharedKvPages source prefix is incomplete";
            return false;
        }
        if (from == to) return truncate(to, end_page, error);

        // Allocate the replacement before releasing destination ownership. The
        // distinct source pins every referenced page throughout replacement.
        std::vector<int32_t> prefix(mappings_[from].begin(),
                                    mappings_[from].begin() + end_page);
        error.clear();
        release_tail(to, 0);
        for (int32_t page : prefix) ++refcounts_[static_cast<size_t>(page)];
        mappings_[to].swap(prefix);
        return true;
    }

    // Non-bool accessors throw std::out_of_range for an invalid sequence rather
    // than indexing unchecked. Releasing an already empty sequence is harmless.
    void release(size_t seq) {
        check_sequence(seq);
        release_tail(seq, 0);
    }

    // Discard pages at/after end_page; a bound beyond the current mapping is a
    // valid no-op. This never grows a sequence. Invalid sequences return false.
    bool truncate(size_t seq, size_t end_page, std::string& error) {
        if (seq >= mappings_.size()) {
            error = "SharedKvPages sequence index out of range";
            return false;
        }
        error.clear();
        release_tail(seq, end_page);
        return true;
    }

    // References remain tied to this owner; operations can invalidate iterators
    // and element references. There is no device allocation or copying here.
    const std::vector<int32_t>& mapping(size_t seq) const {
        check_sequence(seq);
        return mappings_[seq];
    }

    size_t capacity() const { return refcounts_.size(); }
    size_t free_pages() const { return free_.size(); }
    size_t used_pages() const { return capacity() - free_pages(); }

private:
    void check_sequence(size_t seq) const {
        if (seq >= mappings_.size()) {
            throw std::out_of_range("SharedKvPages sequence index out of range");
        }
    }

    int32_t allocate() {
        const int32_t page = free_.back();
        free_.pop_back();
        refcounts_[static_cast<size_t>(page)] = 1;
        return page;
    }

    void release_tail(size_t seq, size_t end_page) {
        auto& pages = mappings_[seq];
        while (pages.size() > end_page) {
            const int32_t page = pages.back();
            pages.pop_back();
            if (--refcounts_[static_cast<size_t>(page)] == 0) free_.push_back(page);
        }
    }

    std::vector<std::vector<int32_t>> mappings_;
    std::vector<size_t> refcounts_;
    std::vector<int32_t> free_;
};

} // namespace strata::core
