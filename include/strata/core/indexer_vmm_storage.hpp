#pragma once

// I2c (str-a2y.4): the CUDA VMM binding of the private indexer storage seam (indexer_private_storage.hpp).  The
// adapter owns one VmmRange reserved for the whole logical ceiling, so PrivateIndexerStorage::base() never moves and a
// captured graph keeps its pointer across growth and shrink.  Each ledger step maps to one driver step:
//
//   acquire_chunk   -> a borrowed chunk from the optional donor, else vmm_chunk_new on the range's device
//   map_chunk       -> VmmRange::map_at (cuMemMap, no access yet)
//   set_access      -> VmmRange::grant_access (cuMemSetAccess)
//   initialize_rows -> cudaMemsetAsync of exactly the new rows with init_byte, then a stream sync
//   unmap_chunk     -> device drain, then VmmRange::unmap (cuMemUnmap); the handle stays owned until release
//   release_chunk   -> borrowed: back to the donor; fresh: vmm_chunk_free (cuMemRelease)
//
// Rows are linear (row r at base + r * row_bytes): supported() requires the granularity to be a whole number of rows,
// which holds for the 2 MiB granule and 512-byte (128 x FP32) rows.  Contract violations by the caller (double acquire,
// release of an unowned chunk) throw, which the ledger treats as uncertainty and poisons.  An asynchronous device fault
// during initialization also throws: it is not a clean, recoverable failure.
//
// Poisoned teardown never returns a borrowed chunk to its donor (its contents and mapping history are uncertain); such
// chunks are released to the driver and counted in lost_borrowed() so the budget owner can account for them.
//
// This header names no CUDA types; the implementation (src/core/indexer_vmm_storage.cpp) is CUDA-only.

#include "strata/core/indexer_private_storage.hpp"
#include "strata/core/vmm.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace strata::core {

class VmmIndexerStorageAdapter : public IndexerPrivateStorageAdapter {
public:
    struct Options {
        uint8_t init_byte = 0;                       // every newly committed row is filled with this byte
        std::function<VmmChunk()> borrow;            // optional donor: a chunk to borrow, or 0 for none
        std::function<void(VmmChunk)> give_back;     // required with borrow: a borrowed chunk goes back to its owner
    };

    // row_bytes comes from the same geometry PrivateIndexerStorage::create sees (one layer's FP32 row).
    explicit VmmIndexerStorageAdapter(const IndexerExtentGeometry& geometry);
    VmmIndexerStorageAdapter(const IndexerExtentGeometry& geometry, Options options);
    ~VmmIndexerStorageAdapter() override;
    VmmIndexerStorageAdapter(const VmmIndexerStorageAdapter&) = delete;
    VmmIndexerStorageAdapter& operator=(const VmmIndexerStorageAdapter&) = delete;

    bool supported() const override;
    uint64_t granularity_bytes() const override;
    void* base() const override;
    IndexerPrivateStorageCode reserve_virtual_range(uint64_t bytes) override;
    IndexerCommitmentCode acquire_chunk(int64_t chunk, IndexerChunkSource& source) override;
    IndexerCommitmentCode map_chunk(int64_t chunk, IndexerChunkSource source) override;
    IndexerCommitmentCode set_access(int64_t chunk, IndexerChunkSource source) override;
    IndexerCommitmentCode initialize_rows(int64_t chunk, int64_t row_begin, int64_t row_end) override;
    IndexerCommitmentCode unmap_chunk(int64_t chunk, IndexerChunkSource source) override;
    void release_chunk(int64_t chunk, IndexerChunkSource source) override;
    void teardown_reserved_range(bool poisoned) noexcept override;

    uint64_t row_bytes() const { return row_bytes_; }
    float* row_pointer(int64_t row) const;   // nullptr when nothing is reserved or row is outside the reservation
    int device() const { return range_.device(); }
    bool reserved() const { return range_.base() != nullptr; }
    int64_t reserved_chunks() const { return range_.chunks(); }
    int64_t mapped_chunks() const { return range_.mapped_count(); }
    int64_t held_chunks() const;              // acquired and not yet released (mapped or not)
    int64_t live_fresh_chunks() const { return live_fresh_; }   // created by this adapter and not yet freed
    int64_t lost_borrowed() const { return lost_borrowed_; }    // borrowed chunks freed by a poisoned teardown
    int64_t leaked_chunks() const { return leaked_; }           // chunks whose teardown unmap failed

private:
    bool in_range(int64_t chunk) const { return chunk >= 0 && chunk < range_.chunks(); }
    void dispose(VmmChunk h, IndexerChunkSource source, bool may_return) noexcept;

    uint64_t row_bytes_ = 0;
    Options options_;
    VmmRange range_;
    std::vector<VmmChunk> held_;              // per chunk: the owned handle (acquired, maybe mapped), 0 = none
    std::vector<IndexerChunkSource> source_;  // per chunk: where held_[chunk] came from
    void* stream_ = nullptr;                  // cudaStream_t for row initialization
    int64_t live_fresh_ = 0;
    int64_t lost_borrowed_ = 0;
    int64_t leaked_ = 0;
};

}  // namespace strata::core
