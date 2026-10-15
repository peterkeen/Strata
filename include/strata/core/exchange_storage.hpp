#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::core::detail {

// Non-owning views: the complement and exchange allocations retain their original
// owners until FileExpertSource::close(). Ownership changes via slot IDs; the
// physical buffer addresses, bytes and CUDA aliases stay in place. The uniform
// path is the currently wired engine path. The size-class path is the same CPU
// ownership model for compact mixed-size complements: every resident expert keeps
// its own byte range, and each supported blob size owns only its own spare queue.
class ExchangeStorage {
public:
    struct View {
        uint8_t* host = nullptr;
        const uint8_t* device = nullptr;
    };

    struct SpareClass {
        size_t bytes = 0;
        uint8_t* host = nullptr;
        const uint8_t* device = nullptr;
        size_t count = 0;
    };

    bool initialize(const std::vector<uint64_t>& offsets, uint8_t* resident_host,
                    const uint8_t* resident_device, uint64_t resident_bytes,
                    uint8_t* exchange_host, const uint8_t* exchange_device,
                    size_t count, size_t blob_bytes, std::string& error) {
        error.clear();
        if (active()) { error = "exchange storage already initialized"; return false; }
        if (!resident_host || !resident_device || !exchange_host || !exchange_device ||
            !blob_bytes || !count || !resident_bytes || resident_bytes % blob_bytes ||
            count > std::numeric_limits<size_t>::max() / blob_bytes) {
            error = "invalid uniform mapped exchange geometry"; return false;
        }
        const size_t resident_slots = (size_t)(resident_bytes / blob_bytes);
        if (resident_slots > std::numeric_limits<size_t>::max() - count) {
            error = "exchange slot count overflow"; return false;
        }
        auto experts = std::make_unique<std::atomic<size_t>[]>(offsets.size());
        std::vector<size_t> spares(count);
        std::vector<View> buffers(resident_slots + count);
        for (size_t slot = 0; slot < resident_slots; ++slot)
            buffers[slot] = {resident_host + slot * blob_bytes, resident_device + slot * blob_bytes};
        std::vector<bool> used(resident_slots, false);
        for (size_t i = 0; i < offsets.size(); ++i) {
            const uint64_t at = offsets[i];
            experts[i].store(kAbsent, std::memory_order_relaxed);
            if (at == ~uint64_t{0}) continue;
            if (at >= resident_bytes || at % blob_bytes || used[(size_t)(at / blob_bytes)]) {
                error = "invalid or duplicate resident slot"; return false;
            }
            used[(size_t)(at / blob_bytes)] = true;
            experts[i].store((size_t)(at / blob_bytes), std::memory_order_relaxed);
        }
        for (bool present : used) if (!present) {
            error = "unassigned resident slot"; return false;
        }
        for (size_t q = 0; q < count; ++q) {
            spares[q] = resident_slots + q;
            buffers[spares[q]] = {exchange_host + q * blob_bytes, exchange_device + q * blob_bytes};
        }
        experts_ = std::move(experts);
        expert_count_ = offsets.size();
        spares_.swap(spares);
        buffers_.swap(buffers);
        blob_bytes_ = blob_bytes;
        spare_bytes_ = (uint64_t) count * (uint64_t) blob_bytes;
        active_ = true;
        return true;
    }

    bool initialize_size_classes(const std::vector<uint64_t>& offsets,
                                 const std::vector<size_t>& expert_blob_bytes,
                                 uint8_t* resident_host, const uint8_t* resident_device,
                                 uint64_t resident_bytes, const std::vector<SpareClass>& spare_classes,
                                 size_t alignment, std::string& error) {
        error.clear();
        if (active()) { error = "exchange storage already initialized"; return false; }
        if (!resident_host || !resident_device || !resident_bytes || offsets.size() != expert_blob_bytes.size() ||
            offsets.empty() || spare_classes.empty() || alignment == 0) {
            error = "invalid size-class exchange geometry"; return false;
        }
        if ((alignment & (alignment - 1)) != 0) {
            error = "size-class exchange alignment must be a power of two"; return false;
        }
        if (((uintptr_t) resident_host % alignment) != 0 || ((uintptr_t) resident_device % alignment) != 0) {
            error = "resident exchange storage is not aligned"; return false;
        }

        struct ClassBuild { size_t bytes = 0, first = 0, count = 0; };
        std::vector<ClassBuild> classes;
        classes.reserve(spare_classes.size());
        uint64_t spare_bytes = 0;
        size_t spare_slots = 0;
        for (const SpareClass& cls : spare_classes) {
            if (!cls.bytes || !cls.count || !cls.host || !cls.device || cls.bytes % alignment ||
                ((uintptr_t) cls.host % alignment) != 0 || ((uintptr_t) cls.device % alignment) != 0) {
                error = "invalid size-class spare geometry"; return false;
            }
            if (find_class(classes, cls.bytes) != kAbsent) {
                error = "duplicate size-class spare geometry"; return false;
            }
            if (cls.count > std::numeric_limits<size_t>::max() - spare_slots ||
                cls.count > std::numeric_limits<uint64_t>::max() / (uint64_t) cls.bytes ||
                (uint64_t) cls.count * (uint64_t) cls.bytes > std::numeric_limits<uint64_t>::max() - spare_bytes) {
                error = "size-class spare allocation overflows"; return false;
            }
            classes.push_back({cls.bytes, spare_slots, cls.count});
            spare_slots += cls.count;
            spare_bytes += (uint64_t) cls.count * (uint64_t) cls.bytes;
        }
        if (spare_slots > std::numeric_limits<size_t>::max() - offsets.size()) {
            error = "size-class slot count overflows"; return false;
        }
        for (const size_t bytes : expert_blob_bytes) {
            if (!bytes || bytes % alignment || find_class(classes, bytes) == kAbsent) {
                error = "expert blob size has no compatible spare class"; return false;
            }
        }

        struct Range { uint64_t begin, end; };
        std::vector<Range> ranges;
        ranges.reserve(offsets.size());
        auto experts = std::make_unique<std::atomic<size_t>[]>(offsets.size());
        std::vector<View> buffers;
        buffers.reserve(offsets.size() + spare_slots);
        for (size_t i = 0; i < offsets.size(); ++i) {
            experts[i].store(kAbsent, std::memory_order_relaxed);
            const uint64_t at = offsets[i];
            if (at == ~uint64_t{0}) continue;
            const size_t bytes = expert_blob_bytes[i];
            if (at % alignment || at > resident_bytes || bytes > resident_bytes - at) {
                error = "resident expert range is outside the compact complement or is unaligned"; return false;
            }
            ranges.push_back({at, at + (uint64_t) bytes});
            const size_t slot = buffers.size();
            buffers.push_back({resident_host + (size_t) at, resident_device + (size_t) at});
            experts[i].store(slot, std::memory_order_relaxed);
        }
        std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) {
            return a.begin < b.begin || (a.begin == b.begin && a.end < b.end);
        });
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].begin < ranges[i - 1].end) {
                error = "resident expert ranges overlap"; return false;
            }
        }

        std::vector<size_t> spares(spare_slots);
        for (size_t c = 0; c < spare_classes.size(); ++c) {
            const SpareClass& cls = spare_classes[c];
            const size_t first = classes[c].first;
            for (size_t q = 0; q < cls.count; ++q) {
                const size_t slot = buffers.size();
                spares[first + q] = slot;
                buffers.push_back({cls.host + q * cls.bytes, cls.device + q * cls.bytes});
            }
        }

        std::vector<ClassState> states;
        states.reserve(classes.size());
        for (const auto& cls : classes) states.push_back({cls.bytes, cls.first, cls.count});
        experts_ = std::move(experts);
        expert_count_ = offsets.size();
        expert_blob_bytes_ = expert_blob_bytes;
        spares_.swap(spares);
        buffers_.swap(buffers);
        classes_.swap(states);
        blob_bytes_ = 0;
        spare_bytes_ = spare_bytes;
        active_ = true;
        return true;
    }

    bool active() const { return active_; }
    View resident(size_t expert) const {
        if (expert >= expert_count_) return {};
        const size_t slot = experts_[expert].load(std::memory_order_acquire);
        return slot != kAbsent ? buffers_[slot] : View{};
    }
    View spare(size_t q) const { return q < spares_.size() ? buffers_[spares_[q]] : View{}; }
    View spare_for_bytes(size_t bytes, size_t q) const {
        if (blob_bytes_ != 0) return bytes == blob_bytes_ ? spare(q) : View{};
        const size_t cls = find_class(classes_, bytes);
        if (cls == kAbsent || q >= classes_[cls].count) return {};
        return buffers_[spares_[classes_[cls].first_spare + q]];
    }
    size_t spare_capacity(size_t bytes) const {
        if (blob_bytes_ != 0) return bytes == blob_bytes_ ? spares_.size() : 0;
        const size_t cls = find_class(classes_, bytes);
        return cls == kAbsent ? 0 : classes_[cls].count;
    }
    uint64_t spare_bytes() const { return spare_bytes_; }

    // Validate without mutating metadata. The caller may choose the existing copy
    // path on rejection, but must still publish the resulting ownership here.
    bool can_commit(size_t in, size_t out, size_t q, const uint8_t* staged, size_t bytes) const {
        if (!active_ || in == out || in >= expert_count_ || out >= expert_count_ || !staged) return false;
        size_t spare_index = kAbsent;
        if (blob_bytes_ != 0) {
            if (bytes != blob_bytes_ || q >= spares_.size()) return false;
            spare_index = q;
        } else {
            if (expert_blob_bytes_[in] != bytes || expert_blob_bytes_[out] != bytes) return false;
            const size_t cls = find_class(classes_, bytes);
            if (cls == kAbsent || q >= classes_[cls].count) return false;
            spare_index = classes_[cls].first_spare + q;
        }
        return experts_[in].load(std::memory_order_acquire) != kAbsent &&
               experts_[out].load(std::memory_order_acquire) == kAbsent &&
               buffers_[spares_[spare_index]].host == staged;
    }

    // Copy fallback for a rejected rotation: bytes have already been copied into
    // the incoming slot (or will be copied before this call by the caller).
    // Only equal-size ownership can be represented by the compact complement.
    bool commit_copy(size_t in, size_t out, size_t bytes) {
        if (!active_ || in == out || in >= expert_count_ || out >= expert_count_ ||
            (blob_bytes_ ? bytes != blob_bytes_ : expert_blob_bytes_[in] != bytes || expert_blob_bytes_[out] != bytes))
            return false;
        const size_t slot = experts_[in].load(std::memory_order_acquire);
        if (slot == kAbsent || experts_[out].load(std::memory_order_acquire) != kAbsent) return false;
        experts_[out].store(slot, std::memory_order_release);
        experts_[in].store(kAbsent, std::memory_order_release);
        return true;
    }

    // Only after every CPU reader and H2D copy of `in` has completed, and `out`
    // has landed in spare(q). No allocation, memcpy, or CUDA operation here.
    bool commit(size_t in, size_t out, size_t q, const uint8_t* staged, size_t bytes) {
        if (!can_commit(in, out, q, staged, bytes)) return false;
        size_t spare_index = q;
        if (blob_bytes_ == 0) spare_index = classes_[find_class(classes_, bytes)].first_spare + q;
        const size_t incoming_slot = experts_[in].load(std::memory_order_acquire);
        // A background router lookahead can query residency to skip redundant
        // file prefetches. Publish one atomic slot ID; the pointer/alias pair is
        // immutable. Actual compute readers still obey the completion contract.
        experts_[out].store(spares_[spare_index], std::memory_order_release);
        spares_[spare_index] = incoming_slot;
        experts_[in].store(kAbsent, std::memory_order_release);
        ++exchanges_;
        avoided_bytes_ += bytes;
        return true;
    }

    uint64_t exchanges() const { return exchanges_; }
    uint64_t avoided_bytes() const { return avoided_bytes_; } // memcpy payload, not read+write total
    void clear() {
        experts_.reset(); expert_count_ = 0; buffers_.clear(); spares_.clear(); classes_.clear();
        expert_blob_bytes_.clear(); blob_bytes_ = 0; spare_bytes_ = 0; exchanges_ = avoided_bytes_ = 0;
        active_ = false;
    }

private:
    static constexpr size_t kAbsent = std::numeric_limits<size_t>::max();
    struct ClassState { size_t bytes = 0, first_spare = 0, count = 0; };

    template <class ClassList>
    static size_t find_class(const ClassList& classes, size_t bytes) {
        for (size_t i = 0; i < classes.size(); ++i)
            if (classes[i].bytes == bytes) return i;
        return kAbsent;
    }

    std::unique_ptr<std::atomic<size_t>[]> experts_;
    size_t expert_count_ = 0;
    std::vector<View> buffers_; // immutable physical addresses after initialize
    std::vector<size_t> spares_; // touched only by the serialized cache manager
    std::vector<ClassState> classes_; // size-class slices of spares_ (classed mode only)
    std::vector<size_t> expert_blob_bytes_; // per expert in classed mode
    size_t blob_bytes_ = 0;
    uint64_t spare_bytes_ = 0;
    uint64_t exchanges_ = 0, avoided_bytes_ = 0;
    bool active_ = false;
};

} // namespace strata::core::detail
