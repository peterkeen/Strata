#pragma once

#include <cstdint>
#include <limits>

namespace strata::core {

// CPU-only checked arithmetic for the contiguous FP32 QSA indexer-key storage.
// These helpers intentionally know nothing about CUDA/HIP/SYCL allocation and do not imply kernel support for
// arbitrary geometry; current native kernels remain fixed at R=4, D=128. The helpers describe conservative rows
// that current append/scoring/snapshot code may touch so later storage can reserve/commit before GPU launches.
enum class IndexerExtentError {
    ok = 0,
    invalid_geometry,
    negative_extent,
    ceiling_exceeded,
    overflow,
};

struct IndexerExtentGeometry {
    int64_t ceiling_cells = 0;           // logical sequence capacity in cells/tokens
    int64_t idx_block = 0;               // cells per completed pooled key; current QSA uses 4
    int64_t idx_dim = 0;                 // floats per pooled key; current QSA uses 128
    int64_t qsa_layers = 1;              // independent per-layer indexer ranges
    uint64_t physical_granularity = 1;   // bytes, per layer/range; 1 means raw byte accounting only
};

struct IndexerByteExtent {
    int64_t rows = 0;
    uint64_t bytes_per_layer = 0;
    uint64_t bytes_all_layers = 0;
    uint64_t rounded_bytes_per_layer = 0;
    uint64_t rounded_bytes_all_layers = 0;
};

struct IndexerHistoryExtent {
    int64_t cells = 0;
    int64_t ceiling_rows = 0;        // current contiguous allocation capacity: floor(ceiling_cells / block) + 2
    int64_t completed_rows = 0;      // completed block keys committed by accepted history: floor(cells / block)
    int64_t valid_rows = 0;          // completed rows plus the sequence-local spare when cells > 0
    int64_t score_read_rows = 0;     // conservative envelope across old qsa_index and newer/optimized scorers
    IndexerByteExtent valid_bytes{};
    IndexerByteExtent ceiling_bytes{};
};

struct IndexerAppendExtent {
    int64_t start_cell = 0;
    int64_t token_count = 0;
    int64_t end_cell = 0;            // start_cell + token_count, exclusive
    int64_t ceiling_rows = 0;
    int64_t completed_rows_before = 0;
    int64_t completed_rows_after = 0;
    int64_t valid_rows_before = 0;
    int64_t valid_rows_after = 0;
    int64_t score_read_rows_after = 0;
    // Exclusive high-water row for pooled writes performed by this append range alone. It is 0 when the range
    // writes only the raw tail. Position zero writes row 0; every completed block b writes rows b and b + 1.
    int64_t pooled_write_high_row = 0;
    // Rows that must be committed before launching the operation: monotonic after-valid rows and every spare write.
    int64_t required_rows = 0;
    bool writes_tail = false;
    bool writes_dead = false;
    bool writes_completed = false;
    bool writes_spare = false;
    IndexerByteExtent required_bytes{};
    IndexerByteExtent ceiling_bytes{};
};

struct IndexerSnapshotExtent {
    int64_t tokens = 0;
    int64_t ceiling_rows = 0;
    // Canonical snapshot payload rows for pooled keys. It is zero for an empty sequence and otherwise includes
    // the moving spare row at tokens / idx_block, matching conversation_snapshot.cpp.
    int64_t snapshot_rows = 0;
    // Restore copies the saved dead/spare key to pooled[tokens / idx_block] when tokens > 0, so this is the
    // exclusive row high-water needed before restore publishes the checkpoint.
    int64_t restore_spare_write_rows = 0;
    IndexerByteExtent payload_bytes{};
    IndexerByteExtent ceiling_bytes{};
};

inline bool indexer_extent_failed(IndexerExtentError e) { return e != IndexerExtentError::ok; }

namespace indexer_extent_detail {

inline bool checked_add_u64(uint64_t a, uint64_t b, uint64_t& out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    out = a + b;
    return true;
}

inline bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

inline bool checked_round_up(uint64_t bytes, uint64_t granularity, uint64_t& out) {
    if (granularity == 0) return false;
    const uint64_t rem = bytes % granularity;
    if (rem == 0) {
        out = bytes;
        return true;
    }
    return checked_add_u64(bytes, granularity - rem, out);
}

inline IndexerExtentError validate_geometry(const IndexerExtentGeometry& g) {
    if (g.ceiling_cells < 0 || g.idx_block <= 0 || g.idx_dim <= 0 || g.qsa_layers <= 0 ||
        g.physical_granularity == 0)
        return IndexerExtentError::invalid_geometry;
    return IndexerExtentError::ok;
}

inline bool byte_extent(const IndexerExtentGeometry& g, int64_t rows, IndexerByteExtent& out) {
    if (rows < 0) return false;
    uint64_t b = 0;
    if (!checked_mul_u64((uint64_t) rows, (uint64_t) g.idx_dim, b) || !checked_mul_u64(b, sizeof(float), b))
        return false;
    uint64_t all = 0, rounded = 0, rounded_all = 0;
    if (!checked_mul_u64(b, (uint64_t) g.qsa_layers, all) ||
        !checked_round_up(b, g.physical_granularity, rounded) ||
        !checked_mul_u64(rounded, (uint64_t) g.qsa_layers, rounded_all))
        return false;
    out = {rows, b, all, rounded, rounded_all};
    return true;
}

inline int64_t valid_rows_for_cells(int64_t cells, int64_t block) {
    return cells > 0 ? cells / block + 1 : 0;
}

} // namespace indexer_extent_detail

inline IndexerExtentError indexer_ceiling_rows(const IndexerExtentGeometry& g, int64_t& rows) {
    if (const auto e = indexer_extent_detail::validate_geometry(g); indexer_extent_failed(e)) return e;
    const int64_t base = g.ceiling_cells / g.idx_block;
    if (base > std::numeric_limits<int64_t>::max() - 2) return IndexerExtentError::overflow;
    const int64_t tmp = base + 2;
    rows = tmp;
    return IndexerExtentError::ok;
}

// Raw row byte utility for allocation/slab accounting. Unlike the sequence extent helpers below, this deliberately
// permits row counts above the logical ceiling so tests and callers can price arbitrary slabs/chunks explicitly.
inline IndexerExtentError indexer_byte_extent_for_rows(const IndexerExtentGeometry& g, int64_t rows,
                                                       IndexerByteExtent& out) {
    IndexerByteExtent tmp{};
    if (const auto e = indexer_extent_detail::validate_geometry(g); indexer_extent_failed(e)) return e;
    if (rows < 0) return IndexerExtentError::negative_extent;
    if (!indexer_extent_detail::byte_extent(g, rows, tmp)) return IndexerExtentError::overflow;
    out = tmp;
    return IndexerExtentError::ok;
}

inline IndexerExtentError indexer_history_extent(const IndexerExtentGeometry& g, int64_t cells,
                                                 IndexerHistoryExtent& out) {
    IndexerHistoryExtent tmp{};
    int64_t ceiling = 0;
    if (const auto e = indexer_ceiling_rows(g, ceiling); indexer_extent_failed(e)) return e;
    if (cells < 0) return IndexerExtentError::negative_extent;
    if (cells > g.ceiling_cells) return IndexerExtentError::ceiling_exceeded;
    tmp.cells = cells;
    tmp.ceiling_rows = ceiling;
    tmp.completed_rows = cells / g.idx_block;
    tmp.valid_rows = indexer_extent_detail::valid_rows_for_cells(cells, g.idx_block);
    // Use valid_rows as a conservative read envelope: old qsa_index reads the spare row from pooled storage, while
    // newer block scorers branch to dead for b == n_bid and may not need that row physically mapped as a read.
    tmp.score_read_rows = tmp.valid_rows;
    if (tmp.valid_rows > ceiling) return IndexerExtentError::ceiling_exceeded;
    if (!indexer_extent_detail::byte_extent(g, tmp.valid_rows, tmp.valid_bytes) ||
        !indexer_extent_detail::byte_extent(g, ceiling, tmp.ceiling_bytes))
        return IndexerExtentError::overflow;
    out = tmp;
    return IndexerExtentError::ok;
}

inline IndexerExtentError indexer_append_extent(const IndexerExtentGeometry& g, int64_t start_cell,
                                                int64_t token_count, IndexerAppendExtent& out) {
    IndexerAppendExtent tmp{};
    int64_t ceiling = 0;
    if (const auto e = indexer_ceiling_rows(g, ceiling); indexer_extent_failed(e)) return e;
    if (start_cell < 0 || token_count < 0) return IndexerExtentError::negative_extent;
    if (start_cell > g.ceiling_cells || token_count > g.ceiling_cells - start_cell)
        return IndexerExtentError::ceiling_exceeded;
    const int64_t end = start_cell + token_count;
    tmp.start_cell = start_cell;
    tmp.token_count = token_count;
    tmp.end_cell = end;
    tmp.ceiling_rows = ceiling;
    tmp.completed_rows_before = start_cell / g.idx_block;
    tmp.completed_rows_after = end / g.idx_block;
    tmp.valid_rows_before = indexer_extent_detail::valid_rows_for_cells(start_cell, g.idx_block);
    tmp.valid_rows_after = indexer_extent_detail::valid_rows_for_cells(end, g.idx_block);
    // Use valid_rows as the post-append conservative read envelope. It is monotonic with the end cell even when a
    // caller rolls back accepted history after a rejected provisional verification window.
    tmp.score_read_rows_after = tmp.valid_rows_after;
    if (token_count > 0) {
        // O(1) tail detection for generic CPU geometry: block==1 has no raw tail; one token is exact by its slot;
        // any contiguous range of at least two tokens for block>1 necessarily includes a non-completing tail slot.
        tmp.writes_tail = g.idx_block > 1 && (token_count > 1 || start_cell % g.idx_block < g.idx_block - 1);
        if (start_cell == 0) {
            tmp.writes_dead = true;
            tmp.writes_spare = true;
            tmp.pooled_write_high_row = 1;
        }
        // Generic block==1 treats position zero as both initial spare creation and completed block 0, so the
        // completion logic below raises the write high-water exactly to row 2. Current GPU kernels do not support R=1.
        const int64_t first_completed_pos = g.idx_block - 1;
        if (end - 1 >= first_completed_pos) {
            int64_t first_b = 0;
            if (start_cell > first_completed_pos) {
                const int64_t numer = start_cell - first_completed_pos;
                first_b = (numer - 1) / g.idx_block + 1;
            }
            const int64_t last_b = (end - 1 - first_completed_pos) / g.idx_block;
            if (last_b >= first_b) {
                tmp.writes_completed = true;
                tmp.writes_spare = true;
                if (last_b > std::numeric_limits<int64_t>::max() - 2) return IndexerExtentError::overflow;
                const int64_t high = last_b + 2; // completed row b and spare row b + 1
                if (high > tmp.pooled_write_high_row) tmp.pooled_write_high_row = high;
            }
        }
    }
    tmp.required_rows = tmp.valid_rows_after > tmp.pooled_write_high_row ? tmp.valid_rows_after : tmp.pooled_write_high_row;
    if (tmp.required_rows > ceiling) return IndexerExtentError::ceiling_exceeded;
    if (!indexer_extent_detail::byte_extent(g, tmp.required_rows, tmp.required_bytes) ||
        !indexer_extent_detail::byte_extent(g, ceiling, tmp.ceiling_bytes))
        return IndexerExtentError::overflow;
    out = tmp;
    return IndexerExtentError::ok;
}

inline IndexerExtentError indexer_snapshot_extent(const IndexerExtentGeometry& g, int64_t tokens,
                                                  IndexerSnapshotExtent& out) {
    IndexerSnapshotExtent tmp{};
    int64_t ceiling = 0;
    if (const auto e = indexer_ceiling_rows(g, ceiling); indexer_extent_failed(e)) return e;
    if (tokens < 0) return IndexerExtentError::negative_extent;
    if (tokens > g.ceiling_cells) return IndexerExtentError::ceiling_exceeded;
    tmp.tokens = tokens;
    tmp.ceiling_rows = ceiling;
    tmp.snapshot_rows = tokens > 0 ? tokens / g.idx_block + 1 : 0;
    tmp.restore_spare_write_rows = tmp.snapshot_rows;
    if (tmp.snapshot_rows > ceiling) return IndexerExtentError::ceiling_exceeded;
    if (!indexer_extent_detail::byte_extent(g, tmp.snapshot_rows, tmp.payload_bytes) ||
        !indexer_extent_detail::byte_extent(g, ceiling, tmp.ceiling_bytes))
        return IndexerExtentError::overflow;
    out = tmp;
    return IndexerExtentError::ok;
}

} // namespace strata::core
