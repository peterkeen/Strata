#pragma once

#include "strata/core/indexer_commitment.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace strata::core {

// CPU-only seam between the I2a commitment ledger and a future stable-address private indexer allocator.
// The storage object reserves one virtual range for the full logical ceiling during create(); base() is then stable
// for the lifetime of the storage object and must not move when rows are grown or shrunk. I2c should implement the
// adapter with VmmRange rather than replacing it: reserve_virtual_range -> cuMemAddressReserve/VmmRange::reserve,
// acquire_chunk -> vmm_chunk_new or a borrowed cache chunk, map_chunk -> cuMemMap/VmmRange mapping, set_access ->
// cuMemSetAccess, initialize_rows -> device zero/canary init, unmap_chunk -> cuMemUnmap/VmmRange::unmap, and
// release_chunk -> cuMemRelease or returning a borrowed chunk to its owner. This header includes no CUDA types.
//
// The adapter granularity is in bytes. chunk_rows is derived from floor(granularity_bytes / row_bytes_per_layer), so
// every committed chunk fits inside one adapter granule; any leftover bytes in the granule are padding. The current
// seam models one contiguous per-layer FP32 row range. qsa_layers remains in IndexerExtentGeometry for I1 accounting,
// while a later runtime adapter can own one range per QSA layer without changing the ledger contract.
//
// If storage becomes poisoned, normal grow/shrink is refused and transaction resources may intentionally remain held.
// The owner must tear down the whole reserved range. PrivateIndexerStorage::teardown() is idempotent and delegates that
// whole-range cleanup to the adapter; the adapter must tolerate being called after partial/uncertain transactions.
//
// Ownership contract: one adapter instance owns at most one reserved range and one live PrivateIndexerStorage. create()
// refuses to replace a live output object; callers must call teardown() first. That keeps create() atomic and avoids
// tearing down a range that was just reserved through the same adapter.
enum class IndexerPrivateStorageCode {
    ok = 0,
    unsupported,
    storage_busy,
    invalid_geometry,
    negative_extent,
    ceiling_exceeded,
    overflow,
    reserve_failed,
    poisoned,
    retained_extent,
    acquisition_failed,
    map_failed,
    access_uncertain,
    initialization_failed,
    publication_failed,
    publication_uncertain,
    unmap_uncertain,
};

inline bool indexer_private_storage_failed(IndexerPrivateStorageCode c) { return c != IndexerPrivateStorageCode::ok; }

inline std::string_view indexer_private_storage_code_name(IndexerPrivateStorageCode c) {
    switch (c) {
    case IndexerPrivateStorageCode::ok: return "ok";
    case IndexerPrivateStorageCode::unsupported: return "unsupported";
    case IndexerPrivateStorageCode::storage_busy: return "storage_busy";
    case IndexerPrivateStorageCode::invalid_geometry: return "invalid_geometry";
    case IndexerPrivateStorageCode::negative_extent: return "negative_extent";
    case IndexerPrivateStorageCode::ceiling_exceeded: return "ceiling_exceeded";
    case IndexerPrivateStorageCode::overflow: return "overflow";
    case IndexerPrivateStorageCode::reserve_failed: return "reserve_failed";
    case IndexerPrivateStorageCode::poisoned: return "poisoned";
    case IndexerPrivateStorageCode::retained_extent: return "retained_extent";
    case IndexerPrivateStorageCode::acquisition_failed: return "acquisition_failed";
    case IndexerPrivateStorageCode::map_failed: return "map_failed";
    case IndexerPrivateStorageCode::access_uncertain: return "access_uncertain";
    case IndexerPrivateStorageCode::initialization_failed: return "initialization_failed";
    case IndexerPrivateStorageCode::publication_failed: return "publication_failed";
    case IndexerPrivateStorageCode::publication_uncertain: return "publication_uncertain";
    case IndexerPrivateStorageCode::unmap_uncertain: return "unmap_uncertain";
    }
    return "unknown";
}

inline IndexerPrivateStorageCode indexer_private_storage_from_commitment(IndexerCommitmentCode c) {
    switch (c) {
    case IndexerCommitmentCode::ok: return IndexerPrivateStorageCode::ok;
    case IndexerCommitmentCode::invalid_geometry: return IndexerPrivateStorageCode::invalid_geometry;
    case IndexerCommitmentCode::negative_extent: return IndexerPrivateStorageCode::negative_extent;
    case IndexerCommitmentCode::ceiling_exceeded: return IndexerPrivateStorageCode::ceiling_exceeded;
    case IndexerCommitmentCode::overflow: return IndexerPrivateStorageCode::overflow;
    case IndexerCommitmentCode::poisoned: return IndexerPrivateStorageCode::poisoned;
    case IndexerCommitmentCode::retained_extent: return IndexerPrivateStorageCode::retained_extent;
    case IndexerCommitmentCode::acquisition_failed: return IndexerPrivateStorageCode::acquisition_failed;
    case IndexerCommitmentCode::map_failed: return IndexerPrivateStorageCode::map_failed;
    case IndexerCommitmentCode::access_uncertain: return IndexerPrivateStorageCode::access_uncertain;
    case IndexerCommitmentCode::initialization_failed: return IndexerPrivateStorageCode::initialization_failed;
    case IndexerCommitmentCode::publication_failed: return IndexerPrivateStorageCode::publication_failed;
    case IndexerCommitmentCode::publication_uncertain: return IndexerPrivateStorageCode::publication_uncertain;
    case IndexerCommitmentCode::unmap_uncertain: return IndexerPrivateStorageCode::unmap_uncertain;
    }
    return IndexerPrivateStorageCode::invalid_geometry;
}

struct IndexerPrivateStorageResult {
    IndexerPrivateStorageCode code = IndexerPrivateStorageCode::ok;
    int64_t chunk = -1;
    int64_t rows = 0;

    bool ok() const { return code == IndexerPrivateStorageCode::ok; }
};

class IndexerPrivateStorageAdapter {
public:
    virtual ~IndexerPrivateStorageAdapter() = default;

    virtual bool supported() const = 0;
    virtual uint64_t granularity_bytes() const = 0;
    virtual void* base() const = 0;
    virtual IndexerPrivateStorageCode reserve_virtual_range(uint64_t bytes) = 0;
    virtual IndexerCommitmentCode acquire_chunk(int64_t chunk, IndexerChunkSource& source) = 0;
    virtual IndexerCommitmentCode map_chunk(int64_t chunk, IndexerChunkSource source) = 0;
    virtual IndexerCommitmentCode set_access(int64_t chunk, IndexerChunkSource source) = 0;
    virtual IndexerCommitmentCode initialize_rows(int64_t chunk, int64_t row_begin, int64_t row_end) = 0;
    virtual IndexerCommitmentCode unmap_chunk(int64_t chunk, IndexerChunkSource source) = 0;
    virtual void release_chunk(int64_t chunk, IndexerChunkSource source) = 0;
    virtual void teardown_reserved_range(bool poisoned) noexcept = 0;
};

class PrivateIndexerStorage {
public:
    PrivateIndexerStorage() = default;
    ~PrivateIndexerStorage() { teardown(); }
    PrivateIndexerStorage(const PrivateIndexerStorage&) = delete;
    PrivateIndexerStorage& operator=(const PrivateIndexerStorage&) = delete;
    PrivateIndexerStorage(PrivateIndexerStorage&& other) noexcept { move_from(other); }
    PrivateIndexerStorage& operator=(PrivateIndexerStorage&& other) noexcept {
        if (this != &other) {
            teardown();
            move_from(other);
        }
        return *this;
    }

    static IndexerPrivateStorageResult create(const IndexerExtentGeometry& geometry,
                                              IndexerPrivateStorageAdapter* adapter,
                                              PrivateIndexerStorage& out) {
        if (out.valid_) return {IndexerPrivateStorageCode::storage_busy, -1, 0};
        if (adapter == nullptr || !adapter->supported()) return {IndexerPrivateStorageCode::unsupported, -1, 0};
        const uint64_t granularity = adapter->granularity_bytes();
        if (granularity == 0) return {IndexerPrivateStorageCode::invalid_geometry, -1, 0};

        IndexerByteExtent one_row{};
        IndexerExtentGeometry per_layer = geometry;
        per_layer.qsa_layers = 1;
        IndexerExtentError e = indexer_byte_extent_for_rows(per_layer, 1, one_row);
        if (indexer_extent_failed(e)) {
            return {indexer_private_storage_from_commitment(indexer_commitment_from_extent_error(e)), -1, 0};
        }
        const uint64_t row_bytes = one_row.bytes_per_layer;
        if (row_bytes == 0 || row_bytes > granularity) return {IndexerPrivateStorageCode::invalid_geometry, -1, 0};
        const int64_t chunk_rows = static_cast<int64_t>(granularity / row_bytes);
        if (chunk_rows <= 0) return {IndexerPrivateStorageCode::invalid_geometry, -1, 0};

        int64_t ceiling_rows = 0;
        e = indexer_ceiling_rows(geometry, ceiling_rows);
        if (indexer_extent_failed(e)) {
            return {indexer_private_storage_from_commitment(indexer_commitment_from_extent_error(e)), -1, 0};
        }
        const int64_t reserve_chunks = ceiling_rows == 0 ? 0 : ceiling_rows / chunk_rows + (ceiling_rows % chunk_rows == 0 ? 0 : 1);
        if (reserve_chunks <= 0) return {IndexerPrivateStorageCode::invalid_geometry, -1, 0};
        if (static_cast<uint64_t>(reserve_chunks) > std::numeric_limits<uint64_t>::max() / granularity)
            return {IndexerPrivateStorageCode::overflow, -1, 0};
        const uint64_t reserve_bytes = static_cast<uint64_t>(reserve_chunks) * granularity;

        PrivateIndexerCommitmentLedger ledger;
        const IndexerCommitmentResult lr = PrivateIndexerCommitmentLedger::create(geometry, chunk_rows, ledger);
        if (!lr.ok()) return {indexer_private_storage_from_commitment(lr.code), lr.chunk, lr.rows};

        const IndexerPrivateStorageCode reserve = adapter->reserve_virtual_range(reserve_bytes);
        if (reserve != IndexerPrivateStorageCode::ok) return {reserve, -1, 0};
        if (adapter->base() == nullptr) {
            adapter->teardown_reserved_range(false);
            return {IndexerPrivateStorageCode::reserve_failed, -1, 0};
        }

        PrivateIndexerStorage tmp;
        tmp.adapter_ = adapter;
        tmp.ledger_ = ledger;
        tmp.base_ = adapter->base();
        tmp.granularity_bytes_ = granularity;
        tmp.row_bytes_ = row_bytes;
        tmp.reserve_bytes_ = reserve_bytes;
        tmp.reserve_chunks_ = reserve_chunks;
        tmp.valid_ = true;
        out = std::move(tmp);
        return {};
    }

    void* base() const { return base_; }
    uint64_t granularity_bytes() const { return granularity_bytes_; }
    uint64_t row_bytes() const { return row_bytes_; }
    uint64_t reserve_bytes() const { return reserve_bytes_; }
    int64_t reserve_chunks() const { return reserve_chunks_; }
    int64_t chunk_rows() const { return ledger_.chunk_rows(); }
    bool valid() const { return valid_; }
    bool poisoned() const { return ledger_.poisoned(); }
    const PrivateIndexerCommitmentLedger& ledger() const { return ledger_; }

    IndexerPrivateStorageResult prepare_append(int64_t start_cell, int64_t token_count) {
        if (!valid_) return {IndexerPrivateStorageCode::unsupported, -1, 0};
        IndexerCommitmentCallbacks callbacks = make_callbacks();
        const IndexerCommitmentResult r = ledger_.prepare_append(start_cell, token_count, callbacks);
        return {indexer_private_storage_from_commitment(r.code), r.chunk, r.rows};
    }

    IndexerPrivateStorageResult ensure_rows(int64_t rows) {
        if (!valid_) return {IndexerPrivateStorageCode::unsupported, -1, rows};
        IndexerCommitmentCallbacks callbacks = make_callbacks();
        const IndexerCommitmentResult r = ledger_.ensure_rows(rows, callbacks);
        return {indexer_private_storage_from_commitment(r.code), r.chunk, r.rows};
    }

    IndexerPrivateStorageResult publish_history(int64_t cells) {
        if (!valid_) return {IndexerPrivateStorageCode::unsupported, -1, cells};
        const IndexerCommitmentResult r = ledger_.publish_history(cells);
        return {indexer_private_storage_from_commitment(r.code), r.chunk, r.rows};
    }

    IndexerPrivateStorageResult shrink_to_rows(int64_t rows) {
        if (!valid_) return {IndexerPrivateStorageCode::unsupported, -1, rows};
        IndexerCommitmentCallbacks callbacks = make_callbacks();
        const IndexerCommitmentResult r = ledger_.shrink_to_rows(rows, callbacks);
        return {indexer_private_storage_from_commitment(r.code), r.chunk, r.rows};
    }

    void clear_prepared_writer_rows() { ledger_.clear_prepared_writer_rows(); }

    void teardown() noexcept {
        if (adapter_ != nullptr) adapter_->teardown_reserved_range(ledger_.poisoned());
        adapter_ = nullptr;
        base_ = nullptr;
        valid_ = false;
    }

private:
    void move_from(PrivateIndexerStorage& other) noexcept {
        adapter_ = other.adapter_;
        ledger_ = other.ledger_;
        base_ = other.base_;
        granularity_bytes_ = other.granularity_bytes_;
        row_bytes_ = other.row_bytes_;
        reserve_bytes_ = other.reserve_bytes_;
        reserve_chunks_ = other.reserve_chunks_;
        valid_ = other.valid_;
        other.adapter_ = nullptr;
        other.base_ = nullptr;
        other.valid_ = false;
    }

    IndexerCommitmentCallbacks make_callbacks() {
        IndexerCommitmentCallbacks callbacks;
        callbacks.acquire = [this](int64_t chunk, IndexerChunkSource& source) {
            return adapter_->acquire_chunk(chunk, source);
        };
        callbacks.map = [this](int64_t chunk, IndexerChunkSource source) {
            return adapter_->map_chunk(chunk, source);
        };
        callbacks.set_access = [this](int64_t chunk, IndexerChunkSource source) {
            return adapter_->set_access(chunk, source);
        };
        callbacks.initialize = [this](int64_t chunk, int64_t row_begin, int64_t row_end) {
            return adapter_->initialize_rows(chunk, row_begin, row_end);
        };
        callbacks.publish = [](int64_t, int64_t, int64_t, int64_t) { return IndexerCommitmentCode::ok; };
        callbacks.unmap = [this](int64_t chunk, IndexerChunkSource source) {
            return adapter_->unmap_chunk(chunk, source);
        };
        callbacks.release = [this](int64_t chunk, IndexerChunkSource source) {
            adapter_->release_chunk(chunk, source);
        };
        return callbacks;
    }

    IndexerPrivateStorageAdapter* adapter_ = nullptr; // non-owning; must outlive this storage object
    PrivateIndexerCommitmentLedger ledger_{};
    void* base_ = nullptr;
    uint64_t granularity_bytes_ = 0;
    uint64_t row_bytes_ = 0;
    uint64_t reserve_bytes_ = 0;
    int64_t reserve_chunks_ = 0;
    bool valid_ = false;
};

} // namespace strata::core
