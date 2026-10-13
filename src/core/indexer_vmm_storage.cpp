#include "strata/core/indexer_vmm_storage.hpp"

#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
#error "indexer_vmm_storage.cpp is CUDA-only: the HIP build has no VmmRange (vmm.cpp stubs it out)"
#endif

#include <cuda_runtime.h>

#include <stdexcept>
#include <utility>

namespace strata::core {
namespace {

// The range's device is current for every runtime call the adapter makes, and the caller's device is restored after.
class DeviceScope {
public:
    explicit DeviceScope(int dev) {
        if (cudaGetDevice(&prev_) != cudaSuccess) {
            (void) cudaGetLastError();
            prev_ = -1;
            return;
        }
        if (dev >= 0 && dev != prev_) {
            ok_ = cudaSetDevice(dev) == cudaSuccess;
            switched_ = ok_;
            if (!ok_) (void) cudaGetLastError();
        }
    }
    ~DeviceScope() {
        if (switched_) (void) cudaSetDevice(prev_);
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    bool ok() const { return ok_; }

private:
    int prev_ = -1;
    bool ok_ = true;
    bool switched_ = false;
};

uint64_t row_bytes_for(const IndexerExtentGeometry& geometry) {
    IndexerExtentGeometry per_layer = geometry;
    per_layer.qsa_layers = 1;
    IndexerByteExtent one{};
    if (indexer_extent_failed(indexer_byte_extent_for_rows(per_layer, 1, one))) return 0;
    return one.bytes_per_layer;
}

[[noreturn]] void contract(const char* what) { throw std::logic_error(what); }

}  // namespace

VmmIndexerStorageAdapter::VmmIndexerStorageAdapter(const IndexerExtentGeometry& geometry)
    : VmmIndexerStorageAdapter(geometry, Options{}) {}

VmmIndexerStorageAdapter::VmmIndexerStorageAdapter(const IndexerExtentGeometry& geometry, Options options)
    : row_bytes_(row_bytes_for(geometry)), options_(std::move(options)) {}

VmmIndexerStorageAdapter::~VmmIndexerStorageAdapter() { teardown_reserved_range(false); }

bool VmmIndexerStorageAdapter::supported() const {
    if (row_bytes_ == 0 || !vmm_available()) return false;
    if (options_.borrow && !options_.give_back) return false;
    const uint64_t g = vmm_granularity();
    return g >= row_bytes_ && g % row_bytes_ == 0;   // rows tile the granule: row r lives at base + r * row_bytes
}

uint64_t VmmIndexerStorageAdapter::granularity_bytes() const { return vmm_granularity(); }

void* VmmIndexerStorageAdapter::base() const { return range_.base(); }

float* VmmIndexerStorageAdapter::row_pointer(int64_t row) const {
    if (!reserved() || row < 0) return nullptr;
    const uint64_t total = (uint64_t) range_.chunks() * vmm_granularity();
    if ((uint64_t) row >= total / row_bytes_) return nullptr;
    return reinterpret_cast<float*>(range_.base() + (uint64_t) row * row_bytes_);
}

int64_t VmmIndexerStorageAdapter::held_chunks() const {
    int64_t n = 0;
    for (const VmmChunk h : held_) n += h != 0;
    return n;
}

IndexerPrivateStorageCode VmmIndexerStorageAdapter::reserve_virtual_range(uint64_t bytes) {
    if (!supported()) return IndexerPrivateStorageCode::unsupported;
    if (reserved()) return IndexerPrivateStorageCode::storage_busy;   // one range per adapter (the seam's contract)
    const uint64_t g = vmm_granularity();
    if (bytes == 0 || bytes % g != 0) return IndexerPrivateStorageCode::invalid_geometry;
    if (!range_.reserve(bytes)) return IndexerPrivateStorageCode::reserve_failed;
    if ((uint64_t) range_.chunks() * g != bytes) {
        range_.release();
        return IndexerPrivateStorageCode::reserve_failed;
    }
    DeviceScope scope(range_.device());
    cudaStream_t s = nullptr;
    if (!scope.ok() || cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
        (void) cudaGetLastError();
        range_.release();
        return IndexerPrivateStorageCode::reserve_failed;
    }
    stream_ = s;
    held_.assign((size_t) range_.chunks(), 0);
    source_.assign((size_t) range_.chunks(), IndexerChunkSource::fresh);
    return IndexerPrivateStorageCode::ok;
}

IndexerCommitmentCode VmmIndexerStorageAdapter::acquire_chunk(int64_t chunk, IndexerChunkSource& source) {
    if (!in_range(chunk)) contract("indexer VMM acquire: chunk outside the reservation");
    if (held_[(size_t) chunk] != 0 || range_.mapped(chunk)) contract("indexer VMM acquire: chunk already held");
    VmmChunk h = options_.borrow ? options_.borrow() : 0;
    IndexerChunkSource from = IndexerChunkSource::borrowed;
    if (h == 0) {
        h = vmm_chunk_new(range_.device());
        from = IndexerChunkSource::fresh;
        if (h == 0) return IndexerCommitmentCode::acquisition_failed;
        ++live_fresh_;
    }
    held_[(size_t) chunk] = h;
    source_[(size_t) chunk] = from;
    source = from;
    return IndexerCommitmentCode::ok;
}

IndexerCommitmentCode VmmIndexerStorageAdapter::map_chunk(int64_t chunk, IndexerChunkSource source) {
    if (!in_range(chunk) || held_[(size_t) chunk] == 0) contract("indexer VMM map: chunk not held");
    if (source_[(size_t) chunk] != source) contract("indexer VMM map: source differs from acquisition");
    if (range_.mapped(chunk)) contract("indexer VMM map: chunk already mapped");
    DeviceScope scope(range_.device());
    if (!scope.ok()) return IndexerCommitmentCode::map_failed;
    if (!range_.map_at(chunk, held_[(size_t) chunk])) {
        (void) cudaGetLastError();
        return IndexerCommitmentCode::map_failed;
    }
    return IndexerCommitmentCode::ok;
}

IndexerCommitmentCode VmmIndexerStorageAdapter::set_access(int64_t chunk, IndexerChunkSource source) {
    if (!in_range(chunk) || !range_.mapped(chunk)) contract("indexer VMM access: chunk not mapped");
    if (source_[(size_t) chunk] != source) contract("indexer VMM access: source differs from acquisition");
    DeviceScope scope(range_.device());
    if (!scope.ok() || !range_.grant_access(chunk, chunk + 1)) {
        (void) cudaGetLastError();
        return IndexerCommitmentCode::access_uncertain;
    }
    return IndexerCommitmentCode::ok;
}

IndexerCommitmentCode VmmIndexerStorageAdapter::initialize_rows(int64_t chunk, int64_t row_begin, int64_t row_end) {
    const int64_t chunk_rows = (int64_t) (vmm_granularity() / row_bytes_);
    if (!in_range(chunk) || !range_.mapped(chunk)) contract("indexer VMM init: chunk not mapped");
    if (row_begin < chunk * chunk_rows || row_end > (chunk + 1) * chunk_rows || row_begin >= row_end)
        contract("indexer VMM init: rows outside the chunk");
    DeviceScope scope(range_.device());
    if (!scope.ok()) return IndexerCommitmentCode::initialization_failed;
    auto* s = static_cast<cudaStream_t>(stream_);
    const uint64_t at = (uint64_t) row_begin * row_bytes_;
    const uint64_t n = (uint64_t) (row_end - row_begin) * row_bytes_;
    if (cudaMemsetAsync(range_.base() + at, options_.init_byte, (size_t) n, s) != cudaSuccess) {
        (void) cudaGetLastError();   // a refused launch: nothing ran, the rows stay unpublished
        return IndexerCommitmentCode::initialization_failed;
    }
    // A failure here is an asynchronous device fault (possibly sticky), not a clean refusal: the ledger poisons.
    if (cudaStreamSynchronize(s) != cudaSuccess) throw std::runtime_error("indexer VMM init: device fault");
    return IndexerCommitmentCode::ok;
}

IndexerCommitmentCode VmmIndexerStorageAdapter::unmap_chunk(int64_t chunk, IndexerChunkSource source) {
    if (!in_range(chunk) || held_[(size_t) chunk] == 0 || !range_.mapped(chunk))
        contract("indexer VMM unmap: chunk not held and mapped");
    if (source_[(size_t) chunk] != source) contract("indexer VMM unmap: source differs from acquisition");
    DeviceScope scope(range_.device());
    // Work still in flight on any stream may touch these rows; unmapping under it faults (vmm_test's note).
    if (!scope.ok() || cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        return IndexerCommitmentCode::unmap_uncertain;
    }
    const VmmChunk h = range_.unmap(chunk);
    if (h == 0) {
        (void) cudaGetLastError();
        return IndexerCommitmentCode::unmap_uncertain;
    }
    if (h != held_[(size_t) chunk]) contract("indexer VMM unmap: range held a different handle");
    return IndexerCommitmentCode::ok;
}

void VmmIndexerStorageAdapter::release_chunk(int64_t chunk, IndexerChunkSource source) {
    if (!in_range(chunk) || held_[(size_t) chunk] == 0) contract("indexer VMM release: chunk not held");
    if (range_.mapped(chunk)) contract("indexer VMM release: chunk still mapped");
    if (source_[(size_t) chunk] != source) contract("indexer VMM release: source differs from acquisition");
    const VmmChunk h = held_[(size_t) chunk];
    held_[(size_t) chunk] = 0;
    dispose(h, source, true);
}

void VmmIndexerStorageAdapter::dispose(VmmChunk h, IndexerChunkSource source, bool may_return) noexcept {
    if (h == 0) return;
    if (source == IndexerChunkSource::borrowed) {
        if (may_return && options_.give_back) {
            try {
                options_.give_back(h);
                return;
            } catch (...) {
                // the owner refused it: freeing is the only safe completion left
            }
        }
        ++lost_borrowed_;
        vmm_chunk_free(h);
        return;
    }
    --live_fresh_;
    vmm_chunk_free(h);
}

void VmmIndexerStorageAdapter::teardown_reserved_range(bool poisoned) noexcept {
    if (!reserved()) return;
    DeviceScope scope(range_.device());
    if (cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        poisoned = true;   // something faulted: do not hand borrowed chunks back
    }
    for (int64_t c = 0; c < range_.chunks(); ++c) {
        VmmChunk h = held_.empty() ? 0 : held_[(size_t) c];
        if (range_.mapped(c)) {
            const VmmChunk m = range_.unmap(c);
            if (m == 0) {   // still mapped: freeing it would only defer, so count it and let the range give up
                (void) cudaGetLastError();
                ++leaked_;
                if (!held_.empty()) held_[(size_t) c] = 0;
                continue;
            }
            if (h == 0) h = m;   // mapped but never recorded as held: not expected, but do not leak it
        }
        if (h != 0) {
            dispose(h, held_.empty() ? IndexerChunkSource::fresh : source_[(size_t) c], !poisoned);
            held_[(size_t) c] = 0;
        }
    }
    if (stream_ != nullptr) {
        (void) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }
    range_.release();
    (void) cudaGetLastError();
    held_.clear();
    source_.clear();
}

}  // namespace strata::core
