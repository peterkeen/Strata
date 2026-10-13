#pragma once

#include "strata/core/indexer_extent.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace strata::core {

// Host-only ledger for future stable private QSA indexer storage. It models the
// order and visibility of chunk commitment, row initialization, writer
// preparation, accepted history validity, and retention watermarks. It never
// calls CUDA/HIP/SYCL/VMM APIs; storage adapters are represented by callbacks so
// tests can prove rollback/poisoning behavior before a device implementation
// exists. chunk_rows is the adapter-chosen physical granularity in rows; I2b is
// expected to derive it from VMM granularity divided by bytes per row.
//
// Callback exceptions are treated conservatively as uncertainty: the ledger is
// poisoned, transaction resources are intentionally not released after the
// uncertain point, and the owner must tear down the whole storage object. A
// recoverable initialization failure inside an already committed chunk may leave
// initialized-but-unpublished device contents beyond initialized_rows_; those
// bytes are don't-care and are not visible until a later successful publish.
enum class IndexerCommitmentCode {
    ok = 0,
    invalid_geometry,
    negative_extent,
    ceiling_exceeded,
    overflow,
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

inline bool indexer_commitment_failed(IndexerCommitmentCode c) { return c != IndexerCommitmentCode::ok; }

inline std::string_view indexer_commitment_code_name(IndexerCommitmentCode c) {
    switch (c) {
    case IndexerCommitmentCode::ok: return "ok";
    case IndexerCommitmentCode::invalid_geometry: return "invalid_geometry";
    case IndexerCommitmentCode::negative_extent: return "negative_extent";
    case IndexerCommitmentCode::ceiling_exceeded: return "ceiling_exceeded";
    case IndexerCommitmentCode::overflow: return "overflow";
    case IndexerCommitmentCode::poisoned: return "poisoned";
    case IndexerCommitmentCode::retained_extent: return "retained_extent";
    case IndexerCommitmentCode::acquisition_failed: return "acquisition_failed";
    case IndexerCommitmentCode::map_failed: return "map_failed";
    case IndexerCommitmentCode::access_uncertain: return "access_uncertain";
    case IndexerCommitmentCode::initialization_failed: return "initialization_failed";
    case IndexerCommitmentCode::publication_failed: return "publication_failed";
    case IndexerCommitmentCode::publication_uncertain: return "publication_uncertain";
    case IndexerCommitmentCode::unmap_uncertain: return "unmap_uncertain";
    }
    return "unknown";
}

inline IndexerCommitmentCode indexer_commitment_from_extent_error(IndexerExtentError e) {
    switch (e) {
    case IndexerExtentError::ok: return IndexerCommitmentCode::ok;
    case IndexerExtentError::invalid_geometry: return IndexerCommitmentCode::invalid_geometry;
    case IndexerExtentError::negative_extent: return IndexerCommitmentCode::negative_extent;
    case IndexerExtentError::ceiling_exceeded: return IndexerCommitmentCode::ceiling_exceeded;
    case IndexerExtentError::overflow: return IndexerCommitmentCode::overflow;
    }
    return IndexerCommitmentCode::invalid_geometry;
}

enum class IndexerChunkSource {
    fresh = 0,
    borrowed = 1,
};

struct IndexerCommitmentResult {
    IndexerCommitmentCode code = IndexerCommitmentCode::ok;
    int64_t chunk = -1;
    int64_t rows = 0;

    bool ok() const { return code == IndexerCommitmentCode::ok; }
};

struct IndexerCommitmentCallbacks {
    // acquire chooses whether the new transaction-owned chunk came from fresh
    // storage or from a borrowed/reclaimable budget. release receives the same
    // source exactly once for every recoverably rolled-back or later shrunk
    // chunk. Uncertain access/publication/unmap results poison the ledger and do
    // not pretend resources are safely reusable.
    std::function<IndexerCommitmentCode(int64_t chunk, IndexerChunkSource& source)> acquire;
    std::function<IndexerCommitmentCode(int64_t chunk, IndexerChunkSource source)> map;
    std::function<IndexerCommitmentCode(int64_t chunk, IndexerChunkSource source)> set_access;
    std::function<IndexerCommitmentCode(int64_t chunk, int64_t row_begin, int64_t row_end)> initialize;
    std::function<IndexerCommitmentCode(int64_t old_physical_rows, int64_t new_physical_rows,
                                        int64_t old_initialized_rows, int64_t new_initialized_rows)> publish;
    std::function<IndexerCommitmentCode(int64_t chunk, IndexerChunkSource source)> unmap;
    std::function<void(int64_t chunk, IndexerChunkSource source)> release;
};

class PrivateIndexerCommitmentLedger {
public:
    static IndexerCommitmentResult create(const IndexerExtentGeometry& geometry, int64_t chunk_rows,
                                          PrivateIndexerCommitmentLedger& out) {
        if (chunk_rows <= 0) return {IndexerCommitmentCode::invalid_geometry, -1, chunk_rows};
        int64_t ceiling = 0;
        const IndexerExtentError e = indexer_ceiling_rows(geometry, ceiling);
        if (indexer_extent_failed(e)) return {indexer_commitment_from_extent_error(e), -1, 0};
        PrivateIndexerCommitmentLedger tmp;
        tmp.geometry_ = geometry;
        tmp.ceiling_rows_ = ceiling;
        tmp.chunk_rows_ = chunk_rows;
        out = std::move(tmp);
        return {};
    }

    bool poisoned() const { return poisoned_; }
    const IndexerExtentGeometry& geometry() const { return geometry_; }
    int64_t ceiling_rows() const { return ceiling_rows_; }
    int64_t chunk_rows() const { return chunk_rows_; }
    int64_t committed_chunk_count() const { return static_cast<int64_t>(chunks_.size()); }
    // Real physical commitment in whole adapter chunks. Future budget brokers should price this row count or its
    // byte extent, not the logical usable row count below.
    IndexerCommitmentCode committed_chunk_rows(int64_t& rows) const {
        int64_t tmp = 0;
        if (!checked_chunk_rows(committed_chunk_count(), tmp)) return IndexerCommitmentCode::overflow;
        rows = tmp;
        return IndexerCommitmentCode::ok;
    }
    // Usable committed rows are capped at the logical ceiling; the final physical chunk may extend past this value.
    int64_t physical_committed_rows() const { return rows_for_chunks(committed_chunk_count()); }
    int64_t initialized_rows() const { return initialized_rows_; }
    int64_t prepared_writer_rows() const { return prepared_writer_rows_; }
    int64_t valid_history_rows() const { return valid_history_rows_; }
    int64_t valid_completed_rows() const { return valid_completed_rows_; }
    int64_t reader_retained_rows() const { return reader_retained_rows_; }
    int64_t checkpoint_retained_rows() const { return checkpoint_retained_rows_; }
    IndexerChunkSource chunk_source(int64_t chunk) const {
        if (chunk < 0 || chunk >= committed_chunk_count()) {
            throw std::out_of_range("PrivateIndexerCommitmentLedger chunk index out of range");
        }
        return chunks_[static_cast<size_t>(chunk)].source;
    }

    IndexerCommitmentResult prepare_append(int64_t start_cell, int64_t token_count,
                                           IndexerCommitmentCallbacks& callbacks) {
        if (poisoned_) return {IndexerCommitmentCode::poisoned, -1, 0};
        IndexerAppendExtent extent{};
        const IndexerExtentError e = indexer_append_extent(geometry_, start_cell, token_count, extent);
        if (indexer_extent_failed(e)) return {indexer_commitment_from_extent_error(e), -1, 0};
        if (extent.ceiling_rows != ceiling_rows_) return {IndexerCommitmentCode::invalid_geometry, -1, 0};
        const IndexerCommitmentResult r = ensure_rows(extent.required_rows, callbacks);
        if (!r.ok()) return r;
        prepared_writer_rows_ = extent.required_rows;
        return r;
    }

    IndexerCommitmentResult ensure_rows(int64_t rows, IndexerCommitmentCallbacks& callbacks) {
        if (poisoned_) return {IndexerCommitmentCode::poisoned, -1, rows};
        if (rows < 0) return {IndexerCommitmentCode::negative_extent, -1, rows};
        if (rows > ceiling_rows_) return {IndexerCommitmentCode::ceiling_exceeded, -1, rows};
        if (rows <= initialized_rows_) return {};

        const int64_t target_chunks = chunks_for_rows(rows);
        if (target_chunks > static_cast<int64_t>(chunks_.max_size())) return {IndexerCommitmentCode::overflow, -1, rows};
        std::vector<TxChunk> owned;
        owned.reserve(static_cast<size_t>(target_chunks - committed_chunk_count()));
        chunks_.reserve(static_cast<size_t>(target_chunks));

        for (int64_t chunk = committed_chunk_count(); chunk < target_chunks; ++chunk) {
            IndexerChunkSource source = IndexerChunkSource::fresh;
            IndexerCommitmentCode c = call_acquire(callbacks, chunk, source);
            if (poisoned_) return {c, chunk, rows};
            if (indexer_commitment_failed(c)) return rollback_owned(callbacks, owned, normalize_acquire_failure(c), chunk, rows);
            owned.push_back({chunk, source, false, false});
        }

        for (TxChunk& tx : owned) {
            IndexerCommitmentCode c = call_map(callbacks, tx.chunk, tx.source);
            if (poisoned_) return {c, tx.chunk, rows};
            if (indexer_commitment_failed(c)) return rollback_owned(callbacks, owned, normalize_map_failure(c), tx.chunk, rows);
            tx.mapped = true;
        }

        for (TxChunk& tx : owned) {
            IndexerCommitmentCode c = call_access(callbacks, tx.chunk, tx.source);
            if (indexer_commitment_failed(c)) {
                poisoned_ = true;
                return {IndexerCommitmentCode::access_uncertain, tx.chunk, rows};
            }
            tx.access_set = true;
        }

        for (int64_t chunk = initialized_rows_ / chunk_rows_; chunk < target_chunks; ++chunk) {
            const int64_t begin = std::max(initialized_rows_, chunk * chunk_rows_);
            const int64_t end = std::min(rows, (chunk + 1) * chunk_rows_);
            if (begin >= end) continue;
            IndexerCommitmentCode c = call_initialize(callbacks, chunk, begin, end);
            if (poisoned_) return {c, chunk, rows};
            if (indexer_commitment_failed(c)) {
                return rollback_owned(callbacks, owned, IndexerCommitmentCode::initialization_failed, chunk, rows);
            }
        }

        const int64_t old_physical = physical_committed_rows();
        const int64_t new_physical = rows_for_chunks(target_chunks);
        IndexerCommitmentCode c = call_publish(callbacks, old_physical, new_physical, initialized_rows_, rows);
        if (indexer_commitment_failed(c)) {
            if (c == IndexerCommitmentCode::publication_uncertain) {
                poisoned_ = true;
                return {IndexerCommitmentCode::publication_uncertain, -1, rows};
            }
            return rollback_owned(callbacks, owned, IndexerCommitmentCode::publication_failed, -1, rows);
        }

        // Commit point: all allocations were reserved before the first adapter callback, and the remaining mutations
        // append trivially movable Chunk records within existing vector capacity. After publish() succeeds this block
        // must not call adapters or allocate.
        for (const TxChunk& tx : owned) chunks_.push_back({tx.source});
        initialized_rows_ = rows;
        return {};
    }

    IndexerCommitmentResult publish_history(int64_t cells) {
        if (poisoned_) return {IndexerCommitmentCode::poisoned, -1, cells};
        IndexerHistoryExtent extent{};
        const IndexerExtentError e = indexer_history_extent(geometry_, cells, extent);
        if (indexer_extent_failed(e)) return {indexer_commitment_from_extent_error(e), -1, cells};
        return publish_history_extent(extent);
    }

    IndexerCommitmentResult publish_history_extent(const IndexerHistoryExtent& extent) {
        if (poisoned_) return {IndexerCommitmentCode::poisoned, -1, extent.cells};
        if (extent.ceiling_rows != ceiling_rows_ || extent.valid_rows < extent.completed_rows)
            return {IndexerCommitmentCode::invalid_geometry, -1, extent.valid_rows};
        if (extent.valid_rows > initialized_rows_ || extent.valid_rows > physical_committed_rows())
            return {IndexerCommitmentCode::ceiling_exceeded, -1, extent.valid_rows};
        valid_history_rows_ = extent.valid_rows;
        valid_completed_rows_ = extent.completed_rows;
        return {};
    }

    IndexerCommitmentResult set_reader_retention_rows(int64_t rows) {
        return set_retention(rows, reader_retained_rows_);
    }

    IndexerCommitmentResult set_checkpoint_retention_rows(int64_t rows) {
        return set_retention(rows, checkpoint_retained_rows_);
    }

    void clear_prepared_writer_rows() { prepared_writer_rows_ = 0; }
    void clear_reader_retention_rows() { reader_retained_rows_ = 0; }
    void clear_checkpoint_retention_rows() { checkpoint_retained_rows_ = 0; }

    int64_t shrink_floor_rows() const {
        return std::max(std::max(valid_history_rows_, prepared_writer_rows_),
                        std::max(reader_retained_rows_, checkpoint_retained_rows_));
    }

    IndexerCommitmentResult shrink_to_rows(int64_t rows, IndexerCommitmentCallbacks& callbacks) {
        if (poisoned_) return {IndexerCommitmentCode::poisoned, -1, rows};
        if (rows < 0) return {IndexerCommitmentCode::negative_extent, -1, rows};
        if (rows > ceiling_rows_) return {IndexerCommitmentCode::ceiling_exceeded, -1, rows};
        const int64_t floor = shrink_floor_rows();
        if (rows < floor) return {IndexerCommitmentCode::retained_extent, -1, floor};

        const int64_t target_chunks = chunks_for_rows(rows);
        if (target_chunks >= committed_chunk_count()) return {};

        while (committed_chunk_count() > target_chunks) {
            const int64_t chunk = committed_chunk_count() - 1;
            const IndexerChunkSource source = chunks_.back().source;
            IndexerCommitmentCode c = call_unmap(callbacks, chunk, source);
            if (indexer_commitment_failed(c)) {
                poisoned_ = true;
                return {IndexerCommitmentCode::unmap_uncertain, chunk, rows};
            }
            if (!call_release(callbacks, chunk, source)) return {IndexerCommitmentCode::unmap_uncertain, chunk, rows};
            chunks_.pop_back();
        }
        initialized_rows_ = std::min(initialized_rows_, physical_committed_rows());
        return {};
    }

private:
    struct Chunk {
        IndexerChunkSource source = IndexerChunkSource::fresh;
    };
    struct TxChunk {
        int64_t chunk = 0;
        IndexerChunkSource source = IndexerChunkSource::fresh;
        bool mapped = false;
        bool access_set = false;
    };

    int64_t chunks_for_rows(int64_t rows) const {
        if (rows <= 0) return 0;
        return rows / chunk_rows_ + (rows % chunk_rows_ == 0 ? 0 : 1);
    }

    bool checked_chunk_rows(int64_t chunks, int64_t& rows) const {
        if (chunks < 0) return false;
        if (chunks != 0 && chunk_rows_ > std::numeric_limits<int64_t>::max() / chunks) return false;
        rows = chunks * chunk_rows_;
        return true;
    }

    int64_t rows_for_chunks(int64_t chunks) const {
        int64_t rows = 0;
        if (!checked_chunk_rows(chunks, rows)) return ceiling_rows_;
        return rows > ceiling_rows_ ? ceiling_rows_ : rows;
    }

    static IndexerCommitmentCode normalize_acquire_failure(IndexerCommitmentCode c) {
        return c == IndexerCommitmentCode::ok ? IndexerCommitmentCode::ok : IndexerCommitmentCode::acquisition_failed;
    }

    static IndexerCommitmentCode normalize_map_failure(IndexerCommitmentCode c) {
        return c == IndexerCommitmentCode::ok ? IndexerCommitmentCode::ok : IndexerCommitmentCode::map_failed;
    }

    IndexerCommitmentCode call_acquire(IndexerCommitmentCallbacks& callbacks, int64_t chunk,
                                       IndexerChunkSource& source) {
        if (!callbacks.acquire) return IndexerCommitmentCode::ok;
        try {
            return callbacks.acquire(chunk, source);
        } catch (...) {
            poisoned_ = true;
            return IndexerCommitmentCode::poisoned;
        }
    }

    IndexerCommitmentCode call_map(IndexerCommitmentCallbacks& callbacks, int64_t chunk,
                                   IndexerChunkSource source) {
        if (!callbacks.map) return IndexerCommitmentCode::ok;
        try {
            return callbacks.map(chunk, source);
        } catch (...) {
            poisoned_ = true;
            return IndexerCommitmentCode::poisoned;
        }
    }

    IndexerCommitmentCode call_access(IndexerCommitmentCallbacks& callbacks, int64_t chunk,
                                      IndexerChunkSource source) {
        if (!callbacks.set_access) return IndexerCommitmentCode::ok;
        try {
            return callbacks.set_access(chunk, source);
        } catch (...) {
            poisoned_ = true;
            return IndexerCommitmentCode::access_uncertain;
        }
    }

    IndexerCommitmentCode call_initialize(IndexerCommitmentCallbacks& callbacks, int64_t chunk,
                                          int64_t begin, int64_t end) {
        if (!callbacks.initialize) return IndexerCommitmentCode::ok;
        try {
            return callbacks.initialize(chunk, begin, end);
        } catch (...) {
            poisoned_ = true;
            return IndexerCommitmentCode::poisoned;
        }
    }

    IndexerCommitmentCode call_publish(IndexerCommitmentCallbacks& callbacks, int64_t old_physical,
                                       int64_t new_physical, int64_t old_initialized, int64_t new_initialized) {
        if (!callbacks.publish) return IndexerCommitmentCode::ok;
        try {
            return callbacks.publish(old_physical, new_physical, old_initialized, new_initialized);
        } catch (...) {
            poisoned_ = true;
            return IndexerCommitmentCode::publication_uncertain;
        }
    }

    IndexerCommitmentCode call_unmap(IndexerCommitmentCallbacks& callbacks, int64_t chunk,
                                     IndexerChunkSource source) {
        if (!callbacks.unmap) return IndexerCommitmentCode::ok;
        try {
            return callbacks.unmap(chunk, source);
        } catch (...) {
            poisoned_ = true;
            return IndexerCommitmentCode::unmap_uncertain;
        }
    }

    bool call_release(IndexerCommitmentCallbacks& callbacks, int64_t chunk, IndexerChunkSource source) {
        if (!callbacks.release) return true;
        try {
            callbacks.release(chunk, source);
            return true;
        } catch (...) {
            poisoned_ = true;
            return false;
        }
    }

    IndexerCommitmentResult rollback_owned(IndexerCommitmentCallbacks& callbacks, std::vector<TxChunk>& owned,
                                           IndexerCommitmentCode original, int64_t chunk, int64_t rows) {
        for (auto it = owned.rbegin(); it != owned.rend(); ++it) {
            if (it->mapped) {
                IndexerCommitmentCode c = call_unmap(callbacks, it->chunk, it->source);
                if (indexer_commitment_failed(c)) {
                    poisoned_ = true;
                    return {IndexerCommitmentCode::unmap_uncertain, it->chunk, rows};
                }
            }
            if (!call_release(callbacks, it->chunk, it->source)) return {IndexerCommitmentCode::unmap_uncertain, it->chunk, rows};
        }
        return {original, chunk, rows};
    }

    IndexerCommitmentResult set_retention(int64_t rows, int64_t& target) {
        if (poisoned_) return {IndexerCommitmentCode::poisoned, -1, rows};
        if (rows < 0) return {IndexerCommitmentCode::negative_extent, -1, rows};
        if (rows > initialized_rows_ || rows > physical_committed_rows())
            return {IndexerCommitmentCode::ceiling_exceeded, -1, rows};
        target = rows;
        return {};
    }

    IndexerExtentGeometry geometry_{};
    int64_t ceiling_rows_ = 0;
    int64_t chunk_rows_ = 0;
    std::vector<Chunk> chunks_;
    int64_t initialized_rows_ = 0;
    int64_t prepared_writer_rows_ = 0;
    int64_t valid_history_rows_ = 0;
    int64_t valid_completed_rows_ = 0;
    int64_t reader_retained_rows_ = 0;
    int64_t checkpoint_retained_rows_ = 0;
    bool poisoned_ = false;
};

} // namespace strata::core
