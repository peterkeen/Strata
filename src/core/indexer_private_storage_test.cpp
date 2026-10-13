#include "strata/core/indexer_private_storage.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

int checks = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "indexer_private_storage_test: failed: %s\n", what);
        std::exit(1);
    }
}

void check_code(IndexerPrivateStorageCode got, IndexerPrivateStorageCode want, const char* what) {
    ++checks;
    if (got != want) {
        std::fprintf(stderr, "indexer_private_storage_test: failed: %s (got %.*s want %.*s)\n", what,
                     static_cast<int>(indexer_private_storage_code_name(got).size()),
                     indexer_private_storage_code_name(got).data(),
                     static_cast<int>(indexer_private_storage_code_name(want).size()),
                     indexer_private_storage_code_name(want).data());
        std::exit(1);
    }
}

IndexerExtentGeometry geom(int64_t cells = 32) {
    return {cells, 4, 128, 12, 1};
}

struct Event {
    std::string phase;
    int64_t chunk = -1;
    IndexerChunkSource source = IndexerChunkSource::fresh;
    int64_t begin = 0;
    int64_t end = 0;
};

struct FakeIndexerStorageAdapter final : IndexerPrivateStorageAdapter {
    struct Chunk {
        bool acquired = false;
        bool mapped = false;
        bool access = false;
        bool released = false;
        bool initialized = false;
        IndexerChunkSource source = IndexerChunkSource::fresh;
    };

    bool is_supported = true;
    uint64_t gran = 1536; // 3 rows of 128 FP32 values for one per-layer row.
    void* fake_base = reinterpret_cast<void*>(static_cast<uintptr_t>(0x12345000));
    bool fail_reserve = false;
    int64_t fail_acquire_chunk = -1;
    int64_t fail_map_chunk = -1;
    int64_t fail_access_chunk = -1;
    int64_t fail_initialize_chunk = -1;
    int64_t fail_unmap_chunk = -1;
    std::vector<int64_t> borrowed_chunks;
    std::vector<Chunk> chunks;
    std::vector<Event> events;
    uint64_t reserved_bytes = 0;
    int reserve_calls = 0;
    int teardown_calls = 0;
    int poisoned_teardown_calls = 0;
    int double_release_attempts = 0;
    int double_unmap_attempts = 0;
    int teardown_released_chunks = 0;

    bool supported() const override { return is_supported; }
    uint64_t granularity_bytes() const override { return gran; }
    void* base() const override { return reserved_bytes == 0 ? nullptr : fake_base; }

    IndexerPrivateStorageCode reserve_virtual_range(uint64_t bytes) override {
        events.push_back({"reserve", -1, IndexerChunkSource::fresh, 0, static_cast<int64_t>(bytes)});
        ++reserve_calls;
        if (fail_reserve || reserved_bytes != 0 || bytes == 0 || gran == 0 || bytes % gran != 0)
            return IndexerPrivateStorageCode::reserve_failed;
        reserved_bytes = bytes;
        chunks.assign(static_cast<size_t>(bytes / gran), {});
        return IndexerPrivateStorageCode::ok;
    }

    IndexerCommitmentCode acquire_chunk(int64_t chunk, IndexerChunkSource& source) override {
        events.push_back({"acquire", chunk, source, 0, 0});
        if (reserved_bytes == 0 || chunk == fail_acquire_chunk || chunk < 0 || chunk >= static_cast<int64_t>(chunks.size()))
            return IndexerCommitmentCode::acquisition_failed;
        Chunk& c = chunks[static_cast<size_t>(chunk)];
        if (c.acquired && !c.released) return IndexerCommitmentCode::acquisition_failed;
        source = std::find(borrowed_chunks.begin(), borrowed_chunks.end(), chunk) == borrowed_chunks.end()
                     ? IndexerChunkSource::fresh
                     : IndexerChunkSource::borrowed;
        c.acquired = true;
        c.released = false;
        c.source = source;
        events.back().source = source;
        return IndexerCommitmentCode::ok;
    }

    IndexerCommitmentCode map_chunk(int64_t chunk, IndexerChunkSource source) override {
        events.push_back({"map", chunk, source, 0, 0});
        if (reserved_bytes == 0 || chunk == fail_map_chunk || !live(chunk)) return IndexerCommitmentCode::map_failed;
        chunks[static_cast<size_t>(chunk)].mapped = true;
        return IndexerCommitmentCode::ok;
    }

    IndexerCommitmentCode set_access(int64_t chunk, IndexerChunkSource source) override {
        events.push_back({"access", chunk, source, 0, 0});
        if (reserved_bytes == 0 || chunk == fail_access_chunk || !live(chunk) || !chunks[static_cast<size_t>(chunk)].mapped)
            return IndexerCommitmentCode::access_uncertain;
        chunks[static_cast<size_t>(chunk)].access = true;
        return IndexerCommitmentCode::ok;
    }

    IndexerCommitmentCode initialize_rows(int64_t chunk, int64_t begin, int64_t end) override {
        events.push_back({"initialize", chunk, IndexerChunkSource::fresh, begin, end});
        if (reserved_bytes == 0 || chunk == fail_initialize_chunk || chunk < 0 ||
            chunk >= static_cast<int64_t>(chunks.size()))
            return IndexerCommitmentCode::initialization_failed;
        const Chunk& existing = chunks[static_cast<size_t>(chunk)];
        if (!existing.acquired || existing.released || !existing.mapped || !existing.access)
            return IndexerCommitmentCode::initialization_failed;
        chunks[static_cast<size_t>(chunk)].initialized = true;
        return IndexerCommitmentCode::ok;
    }

    IndexerCommitmentCode unmap_chunk(int64_t chunk, IndexerChunkSource source) override {
        events.push_back({"unmap", chunk, source, 0, 0});
        if (chunk == fail_unmap_chunk) return IndexerCommitmentCode::unmap_uncertain;
        if (chunk < 0 || chunk >= static_cast<int64_t>(chunks.size()) || !chunks[static_cast<size_t>(chunk)].mapped) {
            ++double_unmap_attempts;
            return IndexerCommitmentCode::ok;
        }
        Chunk& c = chunks[static_cast<size_t>(chunk)];
        c.mapped = false;
        c.access = false;
        return IndexerCommitmentCode::ok;
    }

    void release_chunk(int64_t chunk, IndexerChunkSource source) override {
        events.push_back({"release", chunk, source, 0, 0});
        if (chunk < 0 || chunk >= static_cast<int64_t>(chunks.size())) {
            ++double_release_attempts;
            return;
        }
        Chunk& c = chunks[static_cast<size_t>(chunk)];
        if (!c.acquired || c.released) {
            ++double_release_attempts;
            return;
        }
        c.acquired = false;
        c.released = true;
        c.source = source;
    }

    void teardown_reserved_range(bool poisoned) noexcept override {
        ++teardown_calls;
        if (poisoned) ++poisoned_teardown_calls;
        if (reserved_bytes == 0) return;
        for (Chunk& c : chunks) {
            if (c.acquired && !c.released) {
                c.acquired = false;
                c.released = true;
                ++teardown_released_chunks;
            }
            c.mapped = false;
            c.access = false;
        }
        reserved_bytes = 0;
    }

    bool live(int64_t chunk) const {
        if (chunk < 0 || chunk >= static_cast<int64_t>(chunks.size())) return false;
        const Chunk& c = chunks[static_cast<size_t>(chunk)];
        return c.acquired && !c.released;
    }

    int count(const char* phase) const {
        int n = 0;
        for (const Event& e : events) if (e.phase == phase) ++n;
        return n;
    }

    int count(const char* phase, int64_t chunk) const {
        int n = 0;
        for (const Event& e : events) if (e.phase == phase && e.chunk == chunk) ++n;
        return n;
    }
};

PrivateIndexerStorage make_storage(FakeIndexerStorageAdapter& adapter, int64_t cells = 32) {
    PrivateIndexerStorage storage;
    check_code(PrivateIndexerStorage::create(geom(cells), &adapter, storage).code,
               IndexerPrivateStorageCode::ok, "storage create succeeds");
    return storage;
}

void require_ledger(const PrivateIndexerStorage& storage, int64_t usable_rows, int64_t initialized,
                    const char* what) {
    check(storage.ledger().physical_committed_rows() == usable_rows, what);
    check(storage.ledger().initialized_rows() == initialized, what);
    check(storage.ledger().physical_committed_rows() <= storage.ledger().ceiling_rows(),
          "usable commitment never exceeds logical ceiling");
}

void test_unsupported_and_geometry_fail_before_reserve() {
    PrivateIndexerStorage out;
    FakeIndexerStorageAdapter adapter;
    check_code(PrivateIndexerStorage::create(geom(), nullptr, out).code, IndexerPrivateStorageCode::unsupported,
               "null adapter is unsupported without claiming CUDA support");
    adapter.is_supported = false;
    check_code(PrivateIndexerStorage::create(geom(), &adapter, out).code, IndexerPrivateStorageCode::unsupported,
               "unsupported adapter is reported explicitly");
    check(adapter.reserve_calls == 0, "unsupported adapter is not reserved");

    adapter.is_supported = true;
    adapter.gran = 0;
    check_code(PrivateIndexerStorage::create(geom(), &adapter, out).code, IndexerPrivateStorageCode::invalid_geometry,
               "zero granularity is invalid");
    adapter.gran = 128;
    check_code(PrivateIndexerStorage::create(geom(), &adapter, out).code, IndexerPrivateStorageCode::invalid_geometry,
               "granularity smaller than a row is invalid");
    adapter.gran = 1536;
    check_code(PrivateIndexerStorage::create({32, 0, 128, 12, 1}, &adapter, out).code,
               IndexerPrivateStorageCode::invalid_geometry, "invalid I1 geometry is rejected");
    check(adapter.reserve_calls == 0, "invalid geometry does not reserve the virtual range");

    adapter.fail_reserve = true;
    check_code(PrivateIndexerStorage::create(geom(), &adapter, out).code, IndexerPrivateStorageCode::reserve_failed,
               "reserve failure is reported separately");
    check(adapter.reserve_calls == 1, "reserve failure attempts exactly one virtual reserve");
}

void test_live_output_and_adapter_range_are_exclusive() {
    FakeIndexerStorageAdapter adapter;
    PrivateIndexerStorage storage = make_storage(adapter);
    const uint64_t reserved = adapter.reserved_bytes;
    const int teardown_before = adapter.teardown_calls;
    check_code(PrivateIndexerStorage::create(geom(), &adapter, storage).code, IndexerPrivateStorageCode::storage_busy,
               "create refuses to replace a live storage output");
    check(storage.valid(), "busy create leaves existing storage valid");
    check(adapter.reserved_bytes == reserved, "busy create does not disturb existing adapter range");
    check(adapter.teardown_calls == teardown_before, "busy create does not teardown the existing range");

    FakeIndexerStorageAdapter shared_adapter;
    PrivateIndexerStorage first = make_storage(shared_adapter);
    PrivateIndexerStorage second;
    check_code(PrivateIndexerStorage::create(geom(), &shared_adapter, second).code,
               IndexerPrivateStorageCode::reserve_failed,
               "adapter rejects a second reserved range while one storage is live");
    check(first.valid() && !second.valid(), "second range refusal leaves first storage live and second empty");
    check(shared_adapter.reserved_bytes != 0, "second range refusal leaves first adapter range reserved");
}

void test_create_derives_granularity_and_stable_base() {
    FakeIndexerStorageAdapter adapter;
    PrivateIndexerStorage storage = make_storage(adapter);
    check(storage.base() == adapter.fake_base, "storage exposes adapter stable base pointer");
    check(storage.base() == adapter.base(), "base pointer comes from the reserved virtual range");
    check(storage.row_bytes() == 128u * sizeof(float), "row bytes are derived from FP32 per-layer indexer rows");
    check(storage.granularity_bytes() == adapter.gran, "adapter granularity is recorded");
    check(storage.chunk_rows() == 3, "chunk rows are derived from granularity divided by row bytes");
    check(storage.reserve_chunks() == 4 && storage.reserve_bytes() == 4 * adapter.gran,
          "full ceiling is reserved once in whole adapter chunks");
    void* base = storage.base();
    check_code(storage.prepare_append(0, 24).code, IndexerPrivateStorageCode::ok,
               "growth through multiple chunks succeeds");
    check(storage.base() == base, "base pointer does not move after growth");
    check_code(storage.shrink_to_rows(3).code, IndexerPrivateStorageCode::retained_extent,
               "prepared writer extent blocks shrink until cleared");
    storage.clear_prepared_writer_rows();
    check_code(storage.shrink_to_rows(3).code, IndexerPrivateStorageCode::ok, "shrink after clearing writer succeeds");
    check(storage.base() == base, "base pointer does not move after shrink");
    storage.teardown();
    check(adapter.teardown_calls == 1, "explicit teardown releases the reserved virtual range once");
    storage.teardown();
    check(adapter.teardown_calls == 1, "storage teardown is idempotent and does not double-clean adapter");
}

void test_recoverable_acquire_shortage_does_not_publish() {
    FakeIndexerStorageAdapter adapter;
    PrivateIndexerStorage storage = make_storage(adapter);
    check_code(storage.prepare_append(0, 9).code, IndexerPrivateStorageCode::ok, "setup commits one chunk");
    storage.clear_prepared_writer_rows();
    adapter.fail_acquire_chunk = 2;
    check_code(storage.prepare_append(9, 15).code, IndexerPrivateStorageCode::acquisition_failed,
               "allocation shortage is recoverable pressure");
    require_ledger(storage, 3, 3, "acquire failure preserves prior publication");
    check(!storage.poisoned(), "recoverable acquire failure does not poison storage");
    check(adapter.count("release", 1) == 1 && adapter.count("release", 2) == 0,
          "only acquired transaction chunks are released after acquire failure");
    check(adapter.double_release_attempts == 0, "acquire rollback does not double-release");
}

void test_map_partial_success_rolls_back_like_vmm_range() {
    FakeIndexerStorageAdapter adapter;
    adapter.borrowed_chunks = {2};
    PrivateIndexerStorage storage = make_storage(adapter);
    check_code(storage.prepare_append(0, 9).code, IndexerPrivateStorageCode::ok, "setup commits one chunk");
    storage.clear_prepared_writer_rows();
    adapter.fail_map_chunk = 2;
    check_code(storage.prepare_append(9, 15).code, IndexerPrivateStorageCode::map_failed,
               "map failure after earlier chunk success is recoverable");
    require_ledger(storage, 3, 3, "map failure preserves prior publication");
    check(adapter.count("map", 1) == 1 && adapter.count("map", 2) == 1,
          "fake maps first new chunk before failing the later chunk");
    check(adapter.count("unmap", 1) == 1 && adapter.count("unmap", 2) == 0,
          "rollback unmaps only chunks whose map phase succeeded");
    check(adapter.count("release", 1) == 1 && adapter.count("release", 2) == 1,
          "rollback returns both transaction-owned physical chunks once");
    check(adapter.chunks[2].source == IndexerChunkSource::borrowed,
          "borrowed failed-map chunk disposition is preserved for release");
    check(adapter.double_release_attempts == 0 && adapter.double_unmap_attempts == 0,
          "partial map rollback has no double cleanup");
}

void test_access_failure_poisons_without_fake_cleanup() {
    FakeIndexerStorageAdapter adapter;
    PrivateIndexerStorage storage = make_storage(adapter);
    check_code(storage.prepare_append(0, 9).code, IndexerPrivateStorageCode::ok, "setup commits one chunk");
    storage.clear_prepared_writer_rows();
    adapter.fail_access_chunk = 1;
    check_code(storage.prepare_append(9, 15).code, IndexerPrivateStorageCode::access_uncertain,
               "access publication failure is uncertain and fatal");
    check(storage.poisoned(), "access failure poisons storage");
    check(adapter.count("unmap") == 0 && adapter.count("release") == 0,
          "access uncertainty does not pretend rollback is guaranteed");
    storage.teardown();
    check(adapter.poisoned_teardown_calls == 1, "poisoned storage teardown is explicitly signaled");
    check(adapter.teardown_released_chunks == 3, "whole-range teardown reclaims leaked transaction resources");
}

void test_initialization_failure_rolls_back() {
    FakeIndexerStorageAdapter adapter;
    PrivateIndexerStorage storage = make_storage(adapter);
    check_code(storage.prepare_append(0, 9).code, IndexerPrivateStorageCode::ok, "setup commits one chunk");
    storage.clear_prepared_writer_rows();
    adapter.fail_initialize_chunk = 2;
    check_code(storage.prepare_append(9, 15).code, IndexerPrivateStorageCode::initialization_failed,
               "initialization failure is recoverable before publication");
    require_ledger(storage, 3, 3, "initialization failure preserves previous initialized extent");
    check(adapter.count("initialize", 1) == 1 && adapter.count("initialize", 2) == 1,
          "partial initialization before failing chunk is exercised");
    check(adapter.count("unmap", 1) == 1 && adapter.count("unmap", 2) == 1,
          "initialization rollback unmaps transaction mappings");
    check(adapter.count("release", 1) == 1 && adapter.count("release", 2) == 1,
          "initialization rollback releases transaction chunks once");
    check(!storage.poisoned(), "recoverable initialization failure does not poison storage");
}

void test_unmap_failure_poisons_and_blocks_double_cleanup() {
    FakeIndexerStorageAdapter adapter;
    PrivateIndexerStorage storage = make_storage(adapter);
    check_code(storage.prepare_append(0, 24).code, IndexerPrivateStorageCode::ok, "setup grows through three chunks");
    storage.clear_prepared_writer_rows();
    adapter.fail_unmap_chunk = 2;
    check_code(storage.shrink_to_rows(3).code, IndexerPrivateStorageCode::unmap_uncertain,
               "unmap failure during shrink is fatal");
    check(storage.poisoned(), "unmap failure poisons storage");
    check(adapter.count("release", 2) == 0, "uncertain unmap chunk is not released normally");
    storage.teardown();
    check(adapter.poisoned_teardown_calls == 1, "poisoned shrink requires whole storage teardown");
    check(adapter.double_release_attempts == 0, "uncertain shrink cleanup does not double-release");
}

void test_normal_shrink_and_teardown_are_idempotent() {
    FakeIndexerStorageAdapter adapter;
    PrivateIndexerStorage storage = make_storage(adapter);
    check_code(storage.prepare_append(0, 24).code, IndexerPrivateStorageCode::ok, "setup grows through three chunks");
    storage.clear_prepared_writer_rows();
    check_code(storage.shrink_to_rows(0).code, IndexerPrivateStorageCode::ok, "normal shrink releases all chunks");
    check(adapter.count("release", 0) == 1 && adapter.count("release", 1) == 1 && adapter.count("release", 2) == 1,
          "normal shrink releases each committed chunk once");
    check_code(storage.shrink_to_rows(0).code, IndexerPrivateStorageCode::ok, "second shrink is a no-op");
    check(adapter.count("release") == 3, "second shrink does not release again");
    storage.teardown();
    check(adapter.teardown_released_chunks == 0, "teardown after clean shrink finds no live physical chunks");
    check(adapter.double_release_attempts == 0 && adapter.double_unmap_attempts == 0,
          "clean shrink and teardown have no double cleanup attempts");
}

void test_ops_after_teardown_are_refused_before_adapter_callbacks() {
    FakeIndexerStorageAdapter adapter;
    PrivateIndexerStorage storage = make_storage(adapter);
    storage.teardown();
    check(!storage.valid(), "storage is invalid after teardown");
    const int events = static_cast<int>(adapter.events.size());
    check_code(storage.prepare_append(0, 1).code, IndexerPrivateStorageCode::unsupported,
               "prepare after teardown is refused by storage");
    check_code(storage.ensure_rows(1).code, IndexerPrivateStorageCode::unsupported,
               "ensure after teardown is refused by storage");
    check_code(storage.shrink_to_rows(0).code, IndexerPrivateStorageCode::unsupported,
               "shrink after teardown is refused by storage");
    check_code(storage.publish_history(0).code, IndexerPrivateStorageCode::unsupported,
               "history publish after teardown is refused by storage");
    check(static_cast<int>(adapter.events.size()) == events, "teardown-refused operations do not reach adapter callbacks");

    IndexerChunkSource source = IndexerChunkSource::fresh;
    check(adapter.acquire_chunk(0, source) == IndexerCommitmentCode::acquisition_failed,
          "fake acquire fails when no range is reserved");
    check(adapter.map_chunk(0, source) == IndexerCommitmentCode::map_failed,
          "fake map fails when no range is reserved");
    check(adapter.set_access(0, source) == IndexerCommitmentCode::access_uncertain,
          "fake access fails when no range is reserved");
    check(adapter.initialize_rows(0, 0, 1) == IndexerCommitmentCode::initialization_failed,
          "fake initialize fails when no range is reserved");
}

void test_move_transfers_single_teardown_owner() {
    FakeIndexerStorageAdapter adapter;
    {
        PrivateIndexerStorage original = make_storage(adapter);
        PrivateIndexerStorage moved(std::move(original));
        check(!original.valid() && moved.valid(), "move construction invalidates source and keeps destination live");
        original.teardown();
        check(adapter.teardown_calls == 0, "moved-from explicit teardown does not touch adapter");
        PrivateIndexerStorage assigned;
        assigned = std::move(moved);
        check(!moved.valid() && assigned.valid(), "move assignment invalidates source and keeps destination live");
        moved.teardown();
        check(adapter.teardown_calls == 0, "move-assigned-from teardown does not touch adapter");
    }
    check(adapter.teardown_calls == 1, "moved-to storage destructor tears down adapter exactly once");
}

void test_no_cuda_headers_or_runtime_claims_needed() {
    // This test target includes only indexer_private_storage.hpp and CPU standard headers. The fake adapter is enough
    // to exercise the seam; a real CUDA VMM implementation is intentionally left to I2c.
    FakeIndexerStorageAdapter adapter;
    PrivateIndexerStorage storage = make_storage(adapter);
    check(storage.valid(), "CPU-only fake storage is valid for seam tests");
    check_code(storage.publish_history(0).code, IndexerPrivateStorageCode::ok,
               "logical history publication remains ledger-only and GPU-free");
}

} // namespace

int main() {
    test_unsupported_and_geometry_fail_before_reserve();
    test_live_output_and_adapter_range_are_exclusive();
    test_create_derives_granularity_and_stable_base();
    test_recoverable_acquire_shortage_does_not_publish();
    test_map_partial_success_rolls_back_like_vmm_range();
    test_access_failure_poisons_without_fake_cleanup();
    test_initialization_failure_rolls_back();
    test_unmap_failure_poisons_and_blocks_double_cleanup();
    test_normal_shrink_and_teardown_are_idempotent();
    test_ops_after_teardown_are_refused_before_adapter_callbacks();
    test_move_transfers_single_teardown_owner();
    test_no_cuda_headers_or_runtime_claims_needed();
    std::printf("indexer_private_storage_test: %d checks passed\n", checks);
    return 0;
}
