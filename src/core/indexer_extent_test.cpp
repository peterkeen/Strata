#include "strata/core/indexer_extent.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {
using strata::core::IndexerAppendExtent;
using strata::core::IndexerByteExtent;
using strata::core::IndexerExtentError;
using strata::core::IndexerExtentGeometry;
using strata::core::IndexerHistoryExtent;
using strata::core::IndexerSnapshotExtent;
using strata::core::indexer_append_extent;
using strata::core::indexer_byte_extent_for_rows;
using strata::core::indexer_ceiling_rows;
using strata::core::indexer_history_extent;
using strata::core::indexer_snapshot_extent;

int checks = 0;

void require(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::cerr << "indexer_extent_test: failed: " << what << '\n';
        std::exit(1);
    }
}

IndexerExtentGeometry geom(int64_t cells, uint64_t granularity = 1) {
    return {cells, 4, 128, 12, granularity};
}

void require_ok(IndexerExtentError e, const char* what) {
    require(e == IndexerExtentError::ok, what);
}

void require_error(IndexerExtentError got, IndexerExtentError want, const char* what) {
    require(got == want, what);
}

struct AppendOracle {
    bool tail = false;
    bool dead = false;
    bool completed = false;
    bool spare = false;
    int64_t high = 0;
};

AppendOracle oracle_append(int64_t block, int64_t start, int64_t count) {
    AppendOracle o{};
    for (int64_t p = start; p < start + count; ++p) {
        const int64_t slot = p % block;
        if (slot < block - 1) o.tail = true;
        if (p == 0) {
            o.dead = true;
            o.spare = true;
            if (o.high < 1) o.high = 1;
        }
        if (slot == block - 1) {
            o.completed = true;
            o.spare = true;
            const int64_t high = p / block + 2;
            if (o.high < high) o.high = high;
        }
    }
    return o;
}

void test_history_rows() {
    const auto g = geom(5);
    const struct Case { int64_t cells, completed, valid; } cases[] = {
        {0, 0, 0}, {1, 0, 1}, {3, 0, 1}, {4, 1, 2}, {5, 1, 2},
    };
    for (const auto c : cases) {
        IndexerHistoryExtent e{};
        require_ok(indexer_history_extent(g, c.cells, e), "history extent accepts in-ceiling cells");
        require(e.ceiling_rows == 3, "history ceiling rows are floor(capacity/block)+2");
        require(e.completed_rows == c.completed, "history completed rows are accepted floor(cells/block)");
        require(e.valid_rows == c.valid, "history valid rows include moving spare for non-empty histories");
        require(e.score_read_rows == c.valid, "score read rows are conservative old/new scorer envelope");
    }
}

void test_append_rows() {
    const auto g = geom(16);
    const struct Case {
        int64_t start, count, before_valid, after_valid, write_high, required;
        bool tail, dead, completed, spare;
        const char* name;
    } cases[] = {
        {0, 0, 0, 0, 0, 0, false, false, false, false, "empty append"},
        {0, 1, 0, 1, 1, 1, true,  true,  false, true,  "position zero initializes dead and spare"},
        {1, 1, 1, 1, 0, 1, true,  false, false, false, "single non-completing token writes only tail"},
        {0, 3, 0, 1, 1, 1, true,  true,  false, true,  "three-token prefix keeps one spare row"},
        {0, 4, 0, 2, 2, 2, true,  true,  true,  true,  "first full block writes completed row and next spare"},
        {0, 5, 0, 2, 2, 2, true,  true,  true,  true,  "five-token prefix keeps completed plus spare"},
        {4, 4, 2, 3, 3, 3, true,  false, true,  true,  "block-edge chunk writes block 1 and spare 2"},
        {3, 5, 1, 3, 3, 3, true,  false, true,  true,  "verification window reserves all provisional completed rows"},
    };
    for (const auto c : cases) {
        IndexerAppendExtent e{};
        require_ok(indexer_append_extent(g, c.start, c.count, e), c.name);
        require(e.valid_rows_before == c.before_valid, "append valid rows before");
        require(e.valid_rows_after == c.after_valid, "append valid rows after");
        require(e.score_read_rows_after == c.after_valid, "append score read high water after provisional append");
        require(e.pooled_write_high_row == c.write_high, "append pooled write high-water row");
        require(e.required_rows == c.required, "append required committed rows include spare writes");
        require(e.writes_tail == c.tail, "append tail write flag");
        require(e.writes_dead == c.dead, "append dead write flag");
        require(e.writes_completed == c.completed, "append completed write flag");
        require(e.writes_spare == c.spare, "append spare write flag");
    }
}

void test_append_exhaustive_against_oracle() {
    for (int64_t block = 1; block <= 8; ++block) {
        const IndexerExtentGeometry g{32, block, 8, 2, 64};
        for (int64_t start = 0; start <= g.ceiling_cells; ++start) {
            for (int64_t count = 0; count <= g.ceiling_cells - start; ++count) {
                IndexerAppendExtent e{};
                require_ok(indexer_append_extent(g, start, count, e), "exhaustive append extent accepts in-ceiling range");
                const AppendOracle o = oracle_append(block, start, count);
                require(e.writes_tail == o.tail, "exhaustive append tail flag matches oracle");
                require(e.writes_dead == o.dead, "exhaustive append dead flag matches oracle");
                require(e.writes_completed == o.completed, "exhaustive append completed flag matches oracle");
                require(e.writes_spare == o.spare, "exhaustive append spare flag matches oracle");
                require(e.pooled_write_high_row == o.high, "exhaustive append pooled high-water matches oracle");
                const int64_t valid = count == 0 && start == 0 ? 0 : (start + count > 0 ? (start + count) / block + 1 : 0);
                const int64_t required = valid > o.high ? valid : o.high;
                require(e.required_rows == required, "exhaustive append required rows are max(valid, write high-water)");
            }
        }
    }
}

void test_large_append_is_constant_time() {
    const int64_t block = 1000000000000LL;
    const IndexerExtentGeometry g{3 * block, block, 1, 1, 1};
    IndexerAppendExtent e{};
    require_ok(indexer_append_extent(g, block - 2, block + 5, e),
               "very large generic append extent completes without token/block iteration");
    require(e.writes_tail, "very large append writes at least one raw tail slot");
    require(e.writes_completed, "very large append completes provisional blocks");
    require(e.writes_spare, "very large append writes spare after completed block");
    require(!e.writes_dead, "very large append not starting at zero does not rewrite dead");
    require(e.valid_rows_before == 1, "very large append valid rows before");
    require(e.valid_rows_after == 3, "very large append valid rows after");
    require(e.pooled_write_high_row == 3, "very large append pooled write high-water");
    require(e.required_rows == 3, "very large append required rows");
}

void test_snapshot_restore_rows() {
    const auto g = geom(8);
    const struct Case { int64_t tokens, rows; } cases[] = {{0, 0}, {1, 1}, {3, 1}, {4, 2}, {5, 2}, {8, 3}};
    for (const auto c : cases) {
        IndexerSnapshotExtent e{};
        require_ok(indexer_snapshot_extent(g, c.tokens, e), "snapshot extent accepts in-ceiling tokens");
        require(e.snapshot_rows == c.rows, "snapshot payload includes spare row only for non-empty histories");
        require(e.restore_spare_write_rows == c.rows, "restore spare copy targets the snapshot spare extent");
    }
}

void test_capacity_and_bytes() {
    const uint64_t mib = 1024ull * 1024ull;
    const int64_t capacities[] = {8192, 32768, 131072, 262144};
    const uint64_t completed_payload[] = {12ull * mib, 48ull * mib, 192ull * mib, 384ull * mib};
    for (size_t i = 0; i < 4; ++i) {
        const auto g = geom(capacities[i], 2 * mib);
        int64_t rows = -1;
        require_ok(indexer_ceiling_rows(g, rows), "ceiling row formula accepts canonical capacities");
        require(rows == capacities[i] / 4 + 2, "canonical current pooled allocation is capacity/block + 2 rows");
        IndexerByteExtent ceiling{};
        require_ok(indexer_byte_extent_for_rows(g, rows, ceiling), "ceiling bytes compute");
        const uint64_t expected_raw = (uint64_t) rows * 128ull * sizeof(float) * 12ull;
        require(ceiling.bytes_all_layers == expected_raw, "current pooled raw bytes match row formula");
        const uint64_t completed = (uint64_t) (capacities[i] / 4) * 128ull * sizeof(float) * 12ull;
        require(completed == completed_payload[i], "completed-key payload table remains floor(capacity/block) rows");
        require(ceiling.rounded_bytes_per_layer % (2 * mib) == 0, "per-layer bytes round to physical granularity");
        require(ceiling.rounded_bytes_all_layers == ceiling.rounded_bytes_per_layer * 12ull,
                "granularity rounds each independent layer/range before summing");
    }

    const auto slab = geom(1, 2 * mib);
    IndexerByteExtent exact{}, plus_one{};
    require_ok(indexer_byte_extent_for_rows(slab, 4096, exact), "exact slab byte extent");
    require_ok(indexer_byte_extent_for_rows(slab, 4097, plus_one), "plus-one slab byte extent");
    require(exact.bytes_per_layer == 2 * mib && exact.rounded_bytes_per_layer == 2 * mib,
            "4096 rows at 512 bytes exactly fill a 2 MiB slab");
    require(plus_one.bytes_per_layer == 2 * mib + 512 && plus_one.rounded_bytes_per_layer == 4 * mib,
            "one row past a 2 MiB slab rounds to the next slab");
}

void test_full_ceiling_and_rejections_are_atomic() {
    const auto g = geom(262144);
    IndexerAppendExtent e{};
    require_ok(indexer_append_extent(g, 262143, 1, e), "last in-ceiling token append is accepted");
    require(e.required_rows == 262144 / 4 + 1, "full ceiling append reaches the last completed spare row");
    require(e.ceiling_rows == 262144 / 4 + 2, "full ceiling keeps one extra current allocation row");

    IndexerAppendExtent sentinel{};
    sentinel.start_cell = 123;
    sentinel.required_rows = 456;
    require_error(indexer_append_extent(g, 262143, 2, sentinel), IndexerExtentError::ceiling_exceeded,
                  "rejected verify/provisional extent beyond ceiling");
    require(sentinel.start_cell == 123 && sentinel.required_rows == 456, "rejected append does not publish partial output");
    require_error(indexer_append_extent(g, 0, 262145, sentinel), IndexerExtentError::ceiling_exceeded,
                  "rejected append from zero past ceiling");
    require(sentinel.start_cell == 123 && sentinel.required_rows == 456,
            "rejected append from zero past ceiling does not publish partial output");

    IndexerHistoryExtent hist{};
    hist.cells = 77;
    require_error(indexer_history_extent(g, -1, hist), IndexerExtentError::negative_extent,
                  "negative history is rejected");
    require(hist.cells == 77, "rejected history does not publish partial output");

    IndexerSnapshotExtent snap{};
    snap.tokens = 88;
    require_error(indexer_snapshot_extent(g, 262145, snap), IndexerExtentError::ceiling_exceeded,
                  "too-large snapshot extent is rejected");
    require(snap.tokens == 88, "rejected snapshot does not publish partial output");

    IndexerAppendExtent neg{};
    neg.token_count = 66;
    require_error(indexer_append_extent(g, 0, -1, neg), IndexerExtentError::negative_extent,
                  "negative append token count is rejected");
    require(neg.token_count == 66, "rejected negative append does not publish partial output");

    int64_t rows = 99;
    require_error(indexer_ceiling_rows({16, 0, 128, 12, 1}, rows), IndexerExtentError::invalid_geometry,
                  "invalid geometry is rejected");
    require(rows == 99, "rejected ceiling rows do not publish partial output");
    require_error(indexer_ceiling_rows({std::numeric_limits<int64_t>::max(), 1, 128, 12, 1}, rows),
                  IndexerExtentError::overflow, "ceiling row arithmetic overflow is rejected");
    require(rows == 99, "rejected overflowing ceiling rows do not publish partial output");

    IndexerByteExtent bytes{};
    bytes.rows = 101;
    require_error(indexer_byte_extent_for_rows({16, 4, std::numeric_limits<int64_t>::max(), 12, 1},
                                               std::numeric_limits<int64_t>::max(), bytes),
                  IndexerExtentError::overflow, "byte arithmetic overflow is rejected");
    require(bytes.rows == 101, "rejected byte extent does not publish partial output");
    require_error(indexer_byte_extent_for_rows({16, 4, 4611686018427387903LL, 1, 8}, 1, bytes),
                  IndexerExtentError::overflow, "granularity round-up overflow is rejected");
    require(bytes.rows == 101, "rejected round-up overflow does not publish partial output");
    require_error(indexer_byte_extent_for_rows({16, 4, 2305843009213693951LL, 2, 8}, 1, bytes),
                  IndexerExtentError::overflow, "rounded all-layer byte overflow is rejected");
    require(bytes.rows == 101, "rejected all-layer overflow does not publish partial output");
}

} // namespace

int main() {
    test_history_rows();
    test_append_rows();
    test_append_exhaustive_against_oracle();
    test_large_append_is_constant_time();
    test_snapshot_restore_rows();
    test_capacity_and_bytes();
    test_full_ceiling_and_rejections_are_atomic();
    std::cout << "indexer_extent_test: " << checks << " checks passed\n";
    return 0;
}
