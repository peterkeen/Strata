// I2c (str-a2y.4) isolated graph canary for the CUDA VMM private indexer storage (indexer_vmm_storage.hpp).
//
// No model, no server, no expert cache: a few MiB of device memory on one card.  It proves
//   - the full logical ceiling is reserved once and base() never moves across growth, shrink and regrowth;
//   - a CUDA graph captured against that base replays correctly after every growth and shrink;
//   - only whole granules are committed, new rows carry the init canary, old rows survive growth;
//   - a guarded capacity grid (0, 1, chunk edges, ceiling-1, ceiling, ceiling+1) commits exactly ceil(rows/chunk)
//     chunks and refuses past the ceiling;
//   - the retention floor refuses an unsafe shrink without unmapping anything;
//   - injected acquire / map / init failures roll back with real cuMemUnmap/cuMemRelease and leave nothing held;
//     an access failure poisons and whole-range teardown still returns every chunk;
//   - borrowed chunks go back to their donor on shrink, and are not handed back after a poisoned teardown;
//   - (Linux, opt-in: STRATA_INDEXER_VMM_FAULT_PROBE=1) a child process writing one row past the committed prefix
//     faults: the tail is really unmapped.  Off by default because the deliberate fault logs a GPU MMU fault (Xid)
//     on the host, which is noise on a shared card.
// Exit 77 (CTest SKIP_RETURN_CODE) when there is no CUDA device or no VMM support, with the reason printed.

#include "strata/core/indexer_vmm_storage.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

using namespace strata::core;

namespace {

int checks = 0;
int fails = 0;

#define CHECK(c)                                                                         \
    do {                                                                                 \
        ++checks;                                                                        \
        if (!(c)) {                                                                      \
            std::fprintf(stderr, "indexer_vmm_canary: FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++fails;                                                                     \
        }                                                                                \
    } while (0)

#define CUDA_OK(x) CHECK((x) == cudaSuccess)

constexpr int kDim = 128;               // current QSA idx_dim: 512-byte FP32 rows
constexpr int kBlock = 4;               // current QSA idx_block
constexpr uint8_t kInit = 0xA5;         // init canary byte
constexpr uint32_t kInitWord = 0xA5A5A5A5u;

IndexerExtentGeometry geometry(int64_t ceiling_cells) { return {ceiling_cells, kBlock, kDim, 1, 1}; }

// Writes every element of rows [0, *n) with a value that encodes (epoch, row).  The row count is read from device
// memory so one captured graph serves every committed extent; the launch shape is fixed at capture.
__global__ void stamp_rows(float* base, const long long* n_rows, int dim, int epoch) {
    const long long n = *n_rows * (long long) dim;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / dim;
        base[i] = (float) ((r & 0xfff) + (long long) epoch * 4096);
    }
}

// Counts words in rows [lo, hi) that do not hold the init canary.
__global__ void count_not_init(const uint32_t* base, long long lo, long long hi, int dim, uint32_t want,
                               unsigned long long* bad) {
    const long long n = (hi - lo) * (long long) dim;
    const uint32_t* p = base + lo * dim;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
        if (p[i] != want) atomicAdd(bad, 1ull);
}

__global__ void write_one(float* p) { *p = 1.0f; }

float expected(int64_t row, int epoch) { return (float) ((row & 0xfff) + (int64_t) epoch * 4096); }

struct Graph {
    cudaStream_t stream = nullptr;
    cudaGraphExec_t exec = nullptr;
    long long* n_dev = nullptr;
    float* captured = nullptr;
    int epoch = 0;

    bool capture(float* base, int ep) {
        captured = base;
        epoch = ep;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return false;
        if (cudaMalloc(&n_dev, sizeof(long long)) != cudaSuccess) return false;
        cudaGraph_t g = nullptr;
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
        stamp_rows<<<64, 256, 0, stream>>>(base, n_dev, kDim, ep);
        if (cudaStreamEndCapture(stream, &g) != cudaSuccess) return false;
        const bool ok = cudaGraphInstantiate(&exec, g, 0) == cudaSuccess;
        cudaGraphDestroy(g);
        return ok;
    }
    bool replay(int64_t rows) {
        const long long n = rows;
        if (cudaMemcpyAsync(n_dev, &n, sizeof n, cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
        if (cudaGraphLaunch(exec, stream) != cudaSuccess) return false;
        return cudaStreamSynchronize(stream) == cudaSuccess;
    }
    ~Graph() {
        if (exec) cudaGraphExecDestroy(exec);
        if (n_dev) cudaFree(n_dev);
        if (stream) cudaStreamDestroy(stream);
    }
};

unsigned long long not_init(const float* base, int64_t lo, int64_t hi) {
    if (hi <= lo) return 0;
    unsigned long long* bad = nullptr;
    unsigned long long h = ~0ull;
    if (cudaMalloc(&bad, sizeof *bad) != cudaSuccess) return ~0ull;
    if (cudaMemset(bad, 0, sizeof *bad) == cudaSuccess) {
        count_not_init<<<64, 256>>>(reinterpret_cast<const uint32_t*>(base), lo, hi, kDim, kInitWord, bad);
        if (cudaDeviceSynchronize() == cudaSuccess) cudaMemcpy(&h, bad, sizeof h, cudaMemcpyDeviceToHost);
    }
    cudaFree(bad);
    return h;
}

// Rows [lo, hi) hold epoch's stamp (sampled: first, last and every 97th row, whole rows).
bool rows_hold(const float* base, int64_t lo, int64_t hi, int epoch) {
    std::vector<float> row(kDim);
    for (int64_t r = lo; r < hi; r = (r + 97 < hi || r == hi - 1) ? r + 97 : hi - 1) {
        if (cudaMemcpy(row.data(), base + r * kDim, sizeof(float) * kDim, cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        for (float v : row)
            if (v != expected(r, epoch)) {
                std::fprintf(stderr, "indexer_vmm_canary: row %lld holds %g, want %g\n", (long long) r, v,
                             expected(r, epoch));
                return false;
            }
        if (r == hi - 1) break;
    }
    return true;
}

int64_t chunks_for(int64_t rows, int64_t chunk_rows) { return rows <= 0 ? 0 : (rows + chunk_rows - 1) / chunk_rows; }

// Fault injection on top of the real adapter: the n-th call of a step reports failure instead of calling the driver.
struct FaultyAdapter final : VmmIndexerStorageAdapter {
    using VmmIndexerStorageAdapter::VmmIndexerStorageAdapter;
    int acquire_calls = 0, map_calls = 0, access_calls = 0, init_calls = 0;
    int fail_acquire_at = -1, fail_map_at = -1, fail_access_at = -1, fail_init_at = -1;

    IndexerCommitmentCode acquire_chunk(int64_t c, IndexerChunkSource& s) override {
        if (acquire_calls++ == fail_acquire_at) return IndexerCommitmentCode::acquisition_failed;
        return VmmIndexerStorageAdapter::acquire_chunk(c, s);
    }
    IndexerCommitmentCode map_chunk(int64_t c, IndexerChunkSource s) override {
        if (map_calls++ == fail_map_at) return IndexerCommitmentCode::map_failed;
        return VmmIndexerStorageAdapter::map_chunk(c, s);
    }
    IndexerCommitmentCode set_access(int64_t c, IndexerChunkSource s) override {
        if (access_calls++ == fail_access_at) return IndexerCommitmentCode::access_uncertain;
        return VmmIndexerStorageAdapter::set_access(c, s);
    }
    IndexerCommitmentCode initialize_rows(int64_t c, int64_t b, int64_t e) override {
        if (init_calls++ == fail_init_at) return IndexerCommitmentCode::initialization_failed;
        return VmmIndexerStorageAdapter::initialize_rows(c, b, e);
    }
    void reset() { acquire_calls = map_calls = access_calls = init_calls = 0; fail_acquire_at = fail_map_at = fail_access_at = fail_init_at = -1; }
};

// 1. Stable base, graph replay across growth, shrink and regrowth.
void graph_growth_shrink(int64_t chunk_rows) {
    const int64_t ceiling_cells = chunk_rows * kBlock * 6;   // 6 chunks of rows + 2 spare rows -> 7 chunks reserved
    VmmIndexerStorageAdapter::Options opt;
    opt.init_byte = kInit;
    VmmIndexerStorageAdapter adapter(geometry(ceiling_cells), opt);
    PrivateIndexerStorage storage;
    CHECK(PrivateIndexerStorage::create(geometry(ceiling_cells), &adapter, storage).ok());
    CHECK(storage.chunk_rows() == chunk_rows);
    CHECK(storage.reserve_chunks() == 7 && adapter.reserved_chunks() == 7);
    CHECK(adapter.mapped_chunks() == 0 && adapter.live_fresh_chunks() == 0);
    auto* const base = static_cast<float*>(storage.base());
    CHECK(base != nullptr && base == adapter.base());

    // bounded prefix: the first 4096 cells need rows [0, 1025) -> one chunk
    IndexerPrivateStorageResult r = storage.prepare_append(0, chunk_rows);
    CHECK(r.ok());
    const int64_t rows1 = storage.ledger().initialized_rows();
    CHECK(rows1 == chunk_rows / kBlock + 1);
    CHECK(adapter.mapped_chunks() == 1 && adapter.held_chunks() == 1 && adapter.live_fresh_chunks() == 1);
    CHECK(not_init(base, 0, rows1) == 0);

    Graph g;
    CHECK(g.capture(base, 1));
    CHECK(g.replay(rows1));
    CHECK(rows_hold(base, 0, rows1, 1));
    storage.clear_prepared_writer_rows();

    // growth to 3.5 chunks: same base, old rows preserved, new rows initialized, replay reaches them
    const int64_t rows2 = chunk_rows * 3 + chunk_rows / 2;
    CHECK(storage.ensure_rows(rows2).ok());
    CHECK(storage.base() == base && adapter.base() == base);
    CHECK(adapter.mapped_chunks() == 4 && adapter.live_fresh_chunks() == 4);
    CHECK(rows_hold(base, 0, rows1, 1));
    CHECK(not_init(base, rows1, rows2) == 0);
    CHECK(g.replay(rows2));
    CHECK(rows_hold(base, 0, rows2, 1));

    // retention floor: history covering rows2 refuses a shrink below it and leaves every chunk mapped
    const int64_t cells2 = (rows2 - 1) * kBlock;   // valid_rows(cells2) == rows2
    CHECK(storage.publish_history(cells2).ok());
    CHECK(storage.ledger().valid_history_rows() == rows2);
    CHECK(storage.shrink_to_rows(chunk_rows).code == IndexerPrivateStorageCode::retained_extent);
    CHECK(adapter.mapped_chunks() == 4);

    // shrink to one chunk once history allows it; the graph still replays on the remaining prefix
    CHECK(storage.publish_history((chunk_rows - 1) * kBlock).ok());
    CHECK(storage.shrink_to_rows(chunk_rows).ok());
    CHECK(storage.base() == base);
    CHECK(adapter.mapped_chunks() == 1 && adapter.held_chunks() == 1 && adapter.live_fresh_chunks() == 1);
    CHECK(storage.ledger().initialized_rows() == chunk_rows);
    CHECK(g.replay(chunk_rows));
    CHECK(rows_hold(base, 0, chunk_rows, 1));

    // regrowth to the full ceiling: fresh chunks at the same addresses, initialized again
    const int64_t ceiling = storage.ledger().ceiling_rows();
    CHECK(storage.ensure_rows(ceiling).ok());
    CHECK(storage.base() == base && adapter.mapped_chunks() == 7);
    CHECK(not_init(base, chunk_rows, ceiling) == 0);
    CHECK(g.replay(ceiling));
    CHECK(rows_hold(base, 0, ceiling, 1));
    CHECK(storage.ensure_rows(ceiling + 1).code == IndexerPrivateStorageCode::ceiling_exceeded);
    CHECK(adapter.mapped_chunks() == 7);

    CHECK(storage.publish_history(0).ok());
    CHECK(storage.shrink_to_rows(0).ok());
    CHECK(adapter.mapped_chunks() == 0 && adapter.held_chunks() == 0 && adapter.live_fresh_chunks() == 0);
    storage.teardown();
    CHECK(!adapter.reserved() && adapter.leaked_chunks() == 0);
    CUDA_OK(cudaGetLastError());
}

// 2. Guarded capacity grid: each load commits exactly ceil(rows / chunk_rows) chunks and the replay touches them all.
void capacity_grid(int64_t chunk_rows) {
    const int64_t ceiling_cells = chunk_rows * kBlock * 3 - kBlock;   // ceiling rows = 3 * chunk_rows + 1
    VmmIndexerStorageAdapter::Options opt;
    opt.init_byte = kInit;
    VmmIndexerStorageAdapter adapter(geometry(ceiling_cells), opt);
    PrivateIndexerStorage storage;
    CHECK(PrivateIndexerStorage::create(geometry(ceiling_cells), &adapter, storage).ok());
    const int64_t ceiling = storage.ledger().ceiling_rows();
    CHECK(ceiling == 3 * chunk_rows + 1 && storage.reserve_chunks() == 4);
    auto* const base = static_cast<float*>(storage.base());
    Graph g;
    CHECK(g.capture(base, 2));
    const int64_t loads[] = {0, 1, chunk_rows - 1, chunk_rows, chunk_rows + 1, 2 * chunk_rows, ceiling - 1, ceiling};
    for (const int64_t rows : loads) {
        CHECK(storage.ensure_rows(rows).ok());
        CHECK(adapter.mapped_chunks() == chunks_for(rows, chunk_rows));
        CHECK(storage.ledger().physical_committed_rows() >= rows);
        CHECK(not_init(base, 0, rows) == 0);
        CHECK(g.replay(rows));
        CHECK(rows_hold(base, 0, rows, 2));
        CHECK(storage.shrink_to_rows(0).ok());
        CHECK(adapter.mapped_chunks() == 0 && adapter.live_fresh_chunks() == 0);
        CHECK(storage.base() == base);
    }
    CHECK(storage.ensure_rows(ceiling + 1).code == IndexerPrivateStorageCode::ceiling_exceeded);
    CHECK(storage.ensure_rows(-1).code == IndexerPrivateStorageCode::negative_extent);
    CHECK(adapter.mapped_chunks() == 0);
    storage.teardown();
    CUDA_OK(cudaGetLastError());
}

// 3. Injected failures roll back through the real driver calls; an access failure poisons and teardown cleans up.
void failure_rollback(int64_t chunk_rows) {
    const int64_t ceiling_cells = chunk_rows * kBlock * 4;
    VmmIndexerStorageAdapter::Options opt;
    opt.init_byte = kInit;
    {
        FaultyAdapter adapter(geometry(ceiling_cells), opt);
        PrivateIndexerStorage storage;
        CHECK(PrivateIndexerStorage::create(geometry(ceiling_cells), &adapter, storage).ok());
        auto* const base = static_cast<float*>(storage.base());
        CHECK(storage.ensure_rows(chunk_rows / 2).ok());   // one committed, partially initialized chunk
        CHECK(adapter.mapped_chunks() == 1 && adapter.live_fresh_chunks() == 1);

        // growth by 3 chunks: acquire 3, map 3, access 3, init 4 (the partial chunk 0 and chunks 1-3)
        struct Case { const char* name; int step; int at; IndexerPrivateStorageCode want; };
        const Case cases[] = {
            {"acquire#2", 0, 1, IndexerPrivateStorageCode::acquisition_failed},
            {"map#2", 1, 1, IndexerPrivateStorageCode::map_failed},
            {"init#3", 2, 2, IndexerPrivateStorageCode::initialization_failed},
        };
        for (const Case& c : cases) {
            adapter.reset();
            if (c.step == 0) adapter.fail_acquire_at = c.at;
            if (c.step == 1) adapter.fail_map_at = c.at;
            if (c.step == 2) adapter.fail_init_at = c.at;
            const IndexerPrivateStorageResult res = storage.ensure_rows(chunk_rows * 3 + 1);   // grows by 3 chunks
            std::printf("indexer_vmm_canary: injected %s -> %s\n", c.name,
                        std::string(indexer_private_storage_code_name(res.code)).c_str());
            CHECK(res.code == c.want);
            CHECK(!storage.poisoned());
            CHECK(storage.base() == base);
            CHECK(adapter.mapped_chunks() == 1 && adapter.held_chunks() == 1 && adapter.live_fresh_chunks() == 1);
            CHECK(storage.ledger().initialized_rows() == chunk_rows / 2);
            CUDA_OK(cudaGetLastError());
        }
        adapter.reset();
        CHECK(storage.ensure_rows(chunk_rows * 3 + 1).ok());   // the same growth succeeds afterwards
        CHECK(adapter.mapped_chunks() == 4 && adapter.live_fresh_chunks() == 4);
        CHECK(not_init(base, chunk_rows / 2, chunk_rows * 3 + 1) == 0);

        // access failure on a further chunk: poisoned, normal growth refused, chunks still held until teardown
        adapter.reset();
        adapter.fail_access_at = 0;
        CHECK(storage.ensure_rows(chunk_rows * 4 + 1).code == IndexerPrivateStorageCode::access_uncertain);
        CHECK(storage.poisoned());
        CHECK(storage.ensure_rows(1).code == IndexerPrivateStorageCode::poisoned);
        CHECK(storage.shrink_to_rows(0).code == IndexerPrivateStorageCode::poisoned);
        CHECK(adapter.held_chunks() == 5 && adapter.mapped_chunks() == 5);
        storage.teardown();
        CHECK(!adapter.reserved() && adapter.live_fresh_chunks() == 0 && adapter.leaked_chunks() == 0);
        CUDA_OK(cudaGetLastError());
    }
}

// 4. Borrowed chunks: taken from the donor first, returned on shrink, not returned after a poisoned teardown.
void borrowed_chunks(int64_t chunk_rows, int dev) {
    std::vector<VmmChunk> donor;
    for (int i = 0; i < 2; ++i) {
        const VmmChunk h = vmm_chunk_new(dev);
        CHECK(h != 0);
        if (h != 0) donor.push_back(h);
    }
    int gave = 0, returned = 0;
    VmmIndexerStorageAdapter::Options opt;
    opt.init_byte = kInit;
    opt.borrow = [&] {
        if (donor.empty()) return (VmmChunk) 0;
        const VmmChunk h = donor.back();
        donor.pop_back();
        ++gave;
        return h;
    };
    opt.give_back = [&](VmmChunk h) {
        donor.push_back(h);
        ++returned;
    };
    const int64_t ceiling_cells = chunk_rows * kBlock * 4;
    {
        FaultyAdapter adapter(geometry(ceiling_cells), opt);
        PrivateIndexerStorage storage;
        CHECK(PrivateIndexerStorage::create(geometry(ceiling_cells), &adapter, storage).ok());
        auto* const base = static_cast<float*>(storage.base());
        const int64_t rows = chunk_rows * 2 + 7;   // three chunks: two borrowed, one fresh
        CHECK(storage.ensure_rows(rows).ok());
        CHECK(storage.ledger().chunk_source(0) == IndexerChunkSource::borrowed);
        CHECK(storage.ledger().chunk_source(1) == IndexerChunkSource::borrowed);
        CHECK(storage.ledger().chunk_source(2) == IndexerChunkSource::fresh);
        CHECK(gave == 2 && donor.empty() && adapter.live_fresh_chunks() == 1);
        CHECK(not_init(base, 0, rows) == 0);   // borrowed contents were overwritten by the init canary
        Graph g;
        CHECK(g.capture(base, 3));
        CHECK(g.replay(rows));
        CHECK(rows_hold(base, 0, rows, 3));
        CHECK(storage.shrink_to_rows(0).ok());
        CHECK(returned == 2 && donor.size() == 2 && adapter.live_fresh_chunks() == 0 && adapter.lost_borrowed() == 0);

        // poisoned: one borrowed chunk is mapped when access fails; teardown frees it instead of returning it
        adapter.reset();
        adapter.fail_access_at = 0;
        CHECK(storage.ensure_rows(1).code == IndexerPrivateStorageCode::access_uncertain);
        CHECK(storage.poisoned() && donor.size() == 1);
        storage.teardown();
        CHECK(adapter.lost_borrowed() == 1 && returned == 2 && donor.size() == 1);
        CHECK(adapter.live_fresh_chunks() == 0 && adapter.leaked_chunks() == 0);
    }
    for (const VmmChunk h : donor) vmm_chunk_free(h);
    CUDA_OK(cudaGetLastError());
}

// 5. Geometry whose rows do not tile the granule is reported as unsupported, not silently padded.
void unsupported_geometry() {
    const IndexerExtentGeometry odd{4096, kBlock, 3, 1, 1};   // 12-byte rows do not divide a power-of-two granule
    VmmIndexerStorageAdapter adapter(odd);
    PrivateIndexerStorage storage;
    CHECK(!adapter.supported());
    CHECK(PrivateIndexerStorage::create(odd, &adapter, storage).code == IndexerPrivateStorageCode::unsupported);
    VmmIndexerStorageAdapter::Options lopsided;
    lopsided.borrow = [] { return (VmmChunk) 0; };   // borrow without give_back
    VmmIndexerStorageAdapter adapter2(geometry(4096), lopsided);
    CHECK(!adapter2.supported());
}

// Child: commit one chunk, write to the first row past it.  Exit 0 when the write faults (the guard holds).
int probe_unmapped() {
    const int64_t chunk_rows = (int64_t) (vmm_granularity() / (kDim * sizeof(float)));
    VmmIndexerStorageAdapter adapter(geometry(chunk_rows * kBlock * 2));
    PrivateIndexerStorage storage;
    if (!PrivateIndexerStorage::create(geometry(chunk_rows * kBlock * 2), &adapter, storage).ok()) return 3;
    if (!storage.ensure_rows(chunk_rows).ok() || adapter.mapped_chunks() != 1) return 4;
    write_one<<<1, 1>>>(adapter.row_pointer(chunk_rows));
    const cudaError_t e = cudaDeviceSynchronize();
    std::printf("indexer_vmm_canary: probe write past the committed prefix -> %s\n", cudaGetErrorName(e));
    std::fflush(stdout);
    // native runs report IllegalAddress; under compute-sanitizer the same MMU fault surfaces as LaunchFailure
    const bool faulted = e == cudaErrorIllegalAddress || e == cudaErrorLaunchFailure;
    std::_Exit(faulted ? 0 : 1);   // skip destructors: the context is sticky-faulted
}

void guard_probe() {
    const char* opt = std::getenv("STRATA_INDEXER_VMM_FAULT_PROBE");
    if (opt == nullptr || std::strcmp(opt, "1") != 0) {
        std::printf("indexer_vmm_canary: unmapped-tail fault probe skipped (set STRATA_INDEXER_VMM_FAULT_PROBE=1)\n");
        return;
    }
#if defined(__linux__)
    pid_t pid = 0;
    char self[] = "/proc/self/exe";
    char arg[] = "--probe-unmapped";
    char* argv[] = {self, arg, nullptr};
    CHECK(posix_spawn(&pid, self, nullptr, nullptr, argv, environ) == 0);
    int status = 0;
    CHECK(waitpid(pid, &status, 0) == pid);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
#else
    std::printf("indexer_vmm_canary: unmapped-tail fault probe skipped (not Linux)\n");
#endif
}

}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("indexer_vmm_canary: SKIPPED: no CUDA device (%s)\n", cudaGetErrorName(cudaGetLastError()));
        return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess || !vmm_available()) {
        std::printf("indexer_vmm_canary: SKIPPED: device 0 has no CUDA virtual memory management\n");
        return 77;
    }
    if (argc > 1 && std::strcmp(argv[1], "--probe-unmapped") == 0) return probe_unmapped();

    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    const uint64_t gran = vmm_granularity();
    const int64_t chunk_rows = (int64_t) (gran / (kDim * sizeof(float)));
    std::printf("indexer_vmm_canary: device 0 %s sm_%d%d, runtime %d, driver %d, granularity %llu B, %lld rows/chunk\n",
                prop.name, prop.major, prop.minor, rt, drv, (unsigned long long) gran, (long long) chunk_rows);
    CHECK(gran >= kDim * sizeof(float) && gran % (kDim * sizeof(float)) == 0 && chunk_rows >= 2);
    if (fails) return 1;

    size_t free0 = 0, total = 0, free1 = 0;
    cudaMemGetInfo(&free0, &total);
    graph_growth_shrink(chunk_rows);
    capacity_grid(chunk_rows);
    failure_rollback(chunk_rows);
    borrowed_chunks(chunk_rows, 0);
    unsupported_geometry();
    CUDA_OK(cudaDeviceSynchronize());
    cudaMemGetInfo(&free1, &total);
    // informational only: other processes may share the card
    std::printf("indexer_vmm_canary: free memory before %zu MiB, after %zu MiB\n", free0 >> 20, free1 >> 20);
    if (fails == 0) guard_probe();

    std::printf("indexer_vmm_canary: %d checks, %s\n", checks, fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
