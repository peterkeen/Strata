// CUDA-backed borrowed-session and unified-page regression test. Link against the
// engine's session/layer and CUDA kernel targets; no model weights are required.
// Uses real device allocation/transfers (not host link wrapping). Returns 77 when
// no CUDA device is available. CMake wiring belongs to the primary integration.
#include "strata/core/mtp.hpp"
#include "strata/core/shared_kv_runtime.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::core;
namespace {
constexpr int64_t kCells = 128;
constexpr int64_t kLayerLo = 8, kLayerHi = 16; // Nonzero global QSA ordinals exercise release indexing.
constexpr size_t kGuard = 256;
size_t checks = 0;

void require(bool ok, const char* label) {
    ++checks;
    if (!ok) throw std::runtime_error(label);
}
void checked(cudaError_t status, const char* label) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
}
void sync() { checked(cudaDeviceSynchronize(), "device synchronization"); }
struct Span { void* ptr; size_t bytes; };
std::vector<uint8_t> read(Span s) {
    std::vector<uint8_t> bytes(s.bytes);
    if (s.bytes) checked(cudaMemcpy(bytes.data(), s.ptr, s.bytes, cudaMemcpyDeviceToHost), "read device bytes");
    return bytes;
}
void fill(Span s, uint8_t value) {
    checked(cudaMemsetAsync(s.ptr, value, s.bytes, nullptr), "fill device bytes");
}
void uniform(Span s, uint8_t value, const char* label) {
    const auto bytes = read(s);
    require(std::all_of(bytes.begin(), bytes.end(), [=](uint8_t b) { return b == value; }), label);
}
size_t gdn_bytes(const SessionState& s, const ModelGeometry& g) {
    return size_t(s.gdn_alloc) * (size_t(g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size) +
        size_t(g.ssm_conv_channels * (g.ssm_d_conv - 1))) * sizeof(float);
}
// One physical page per active plane; includes every quantization scale plane.
std::vector<Span> pools(QsaState& q, const ModelGeometry& g) {
    const size_t rows = size_t(strata::kernels::qsa_real_shapes().page_size * g.n_head_kv);
    const size_t codes = rows * size_t(g.head_dim);
    const size_t scales = rows * size_t(g.head_dim / strata::kernels::KV_Q8_GROUP) * sizeof(uint16_t);
    const size_t q4 = rows * strata::kernels::kv_q4_bytes_per_head(int(g.head_dim));
    if (q.kv_hybrid) return {{q.k_q, codes}, {q.v_q4, q4}, {q.k_scale, scales}};
    if (q.kv_q4) return {{q.k_q4, q4}, {q.v_q4, q4}};
    if (q.kv_int8) return {{q.k_q, codes}, {q.v_q, codes}, {q.k_scale, scales}, {q.v_scale, scales}};
    return {{q.k_pool, codes * sizeof(uint16_t)}, {q.v_pool, codes * sizeof(uint16_t)}};
}
Span page(Span plane, int32_t physical) {
    return {static_cast<uint8_t*>(plane.ptr) + size_t(physical) * plane.bytes, plane.bytes};
}
std::vector<Span> private_spans(SessionState& s, const ModelGeometry& g) {
    const auto shape = strata::kernels::qsa_real_shapes();
    std::vector<Span> spans = {{s.gdn_state, gdn_bytes(s, g)},
        {s.R, size_t(g.hc * g.n_embd) * sizeof(float)},
        {s.ple_hist, size_t(strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM) * sizeof(float)}};
    for (int64_t j = 0; j < s.qsa_alloc; ++j) {
        auto& q = s.qsa_states[s.qsa_ord0 + j];
        spans.push_back({q.idx_tail, size_t((shape.idx_block - 1) * g.idx_key_dim) * sizeof(float)});
        spans.push_back({q.idx_dead, size_t(g.idx_key_dim) * sizeof(float)});
        spans.push_back({q.idx_pooled, size_t(q.idx_pooled_rows * g.idx_key_dim) * sizeof(float)});
        spans.push_back({q.idx_block_pos, sizeof(int32_t)});
        spans.push_back({q.step, strata::kernels::qsa_step_bytes()});
        spans.push_back({q.attention_status, sizeof(int32_t)});
        spans.push_back({q.pos_dev, size_t(g.n_head) * sizeof(int32_t)});
    }
    return spans;
}

struct DeviceSession {
    SessionState state;
    void* arena = nullptr;
    uint64_t bytes = 0, used = 0;
    DeviceSession() = default;
    DeviceSession(const DeviceSession&) = delete;
    DeviceSession& operator=(const DeviceSession&) = delete;
    ~DeviceSession() {
        // The test scopes borrowers after the owner, so their arenas and staging
        // are always released first. session_release only unregisters owned RoPE.
        session_release(state);
        for (int64_t j = 0; state.qsa_states && j < state.qsa_alloc; ++j) {
            auto& q = state.qsa_states[state.qsa_ord0 + j];
            if (q.host_step) cudaFreeHost(q.host_step);
            if (q.host_pos) cudaFreeHost(q.host_pos);
            // Raw pinned payload handle exists only on owners; mapped device addresses may differ.
            if (q.kv_host_arena) cudaFreeHost(q.kv_host_arena);
        }
        delete[] state.qsa_states;
        if (arena) cudaFree(arena);
    }
    void init(const ModelGeometry& g, const SessionState* owner = nullptr, int64_t cells = kCells,
              int64_t lo = kLayerLo, int64_t hi = kLayerHi) {
        bytes = session_bytes(g, cells, 10, lo, hi, owner);
        require(bytes != 0, "nonzero session size");
        checked(cudaMalloc(&arena, size_t(bytes) + kGuard), "allocate session arena");
        fill({arena, size_t(bytes) + kGuard}, 0xa5);
        used = session_init(g, cells, 10, arena, state, lo, hi, owner);
        require(used != 0 && used <= bytes, "session init fits sized arena");
        if (owner || qsa_kv_unified()) require(used == bytes, "borrowed/unified session sizing exactly equals init consumption");
        guard();
    }
    void guard() {
        uniform({static_cast<uint8_t*>(arena) + bytes, kGuard}, 0xa5, "session arena end guard intact");
    }
    bool contains(const void* ptr, size_t count) const {
        const auto p = reinterpret_cast<uintptr_t>(ptr), b = reinterpret_cast<uintptr_t>(arena);
        return p >= b && p - b <= bytes && count <= bytes - (p - b);
    }
};

void configure(int format) {
    qsa_set_kv_unified(false);
    qsa_set_kv_resident(0);
    qsa_set_kv_int8(format == 1);
    qsa_set_kv_int8_rotate(format == 1);
    qsa_set_kv_q4(format == 2);
    qsa_set_kv_hybrid(format == 3);
}
void mapping(const SessionState& s, const std::vector<int32_t>& prefix) {
    for (int64_t j = 0; j < s.qsa_alloc; ++j) {
        const auto& q = s.qsa_states[s.qsa_ord0 + j];
        std::vector<int32_t> expected(size_t(q.n_pages), -1), device(size_t(q.n_pages));
        require(prefix.size() <= expected.size(), "runtime mapping fits logical capacity");
        std::copy(prefix.begin(), prefix.end(), expected.begin());
        checked(cudaMemcpy(device.data(), q.page_table, device.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
                "read published page table");
        require(device == expected, "device logical mapping matches runtime reservation");
        require(q.shared_page_table == expected, "host logical mapping matches device mapping");
    }
}
void identity(const SessionState& owner) {
    for (int64_t j = 0; j < owner.qsa_alloc; ++j) {
        const auto& q = owner.qsa_states[owner.qsa_ord0 + j];
        std::vector<int32_t> table(size_t(q.n_pages));
        checked(cudaMemcpy(table.data(), q.page_table, table.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
                "read owner identity page table");
        for (int64_t p = 0; p < q.n_pages; ++p) require(table[size_t(p)] == p, "legacy owner identity mapping");
        require(!q.shared_kv && q.shared_page_table.empty(), "owner shared marking is deferred to orchestrator");
    }
}
void aliases(DeviceSession& owner, DeviceSession& borrower, const ModelGeometry& g, int format) {
    auto& o = owner.state; auto& b = borrower.state;
    require(borrower.bytes < owner.bytes, "borrowed arena omits physical pool and RoPE storage");
    require(b.qsa_ord0 == 2 && b.qsa_alloc == 2, "fixture has nonzero global ordinal and multiple QSA states");
    require(b.gdn_state != o.gdn_state && b.gdn.state != o.gdn.state && b.gdn.conv_state != o.gdn.conv_state,
            "recurrence and conv allocations are private");
    require(b.R != o.R && b.ple_hist != o.ple_hist && b.qsa_buf_arena != o.qsa_buf_arena,
            "residual, PLE history and QSA scratch are private");
    const float* rope = o.qsa_states[o.qsa_primary()].cos_tab;
    for (int64_t j = 0; j < b.qsa_alloc; ++j) {
        const auto& oq = o.qsa_states[o.qsa_ord0 + j]; auto& bq = b.qsa_states[b.qsa_ord0 + j];
        require(bq.shared_kv && !bq.owns_rope && bq.kv_mode == 0, "borrower is shared resident, never a RoPE owner");
        require(bq.n_slots == oq.n_slots && oq.n_slots == kCells / 4, "owner retains full physical capacity");
        require(bq.kv_int8 == oq.kv_int8 && bq.kv_q4 == oq.kv_q4 && bq.kv_hybrid == oq.kv_hybrid &&
                bq.kv_rot == oq.kv_rot, "borrower inherits every KV format flag");
        require(oq.kv_int8 == (format == 1) && oq.kv_q4 == (format == 2) && oq.kv_hybrid == (format == 3),
                "requested owner format initialized");
        require(bq.k_pool == oq.k_pool && bq.v_pool == oq.v_pool && bq.k_q == oq.k_q && bq.v_q == oq.v_q &&
                bq.k_scale == oq.k_scale && bq.v_scale == oq.v_scale && bq.k_q4 == oq.k_q4 && bq.v_q4 == oq.v_q4,
                "all K/V and scale pointers alias owner");
        require(bq.cos_tab == oq.cos_tab && bq.cos_tab == rope && bq.sin_tab == oq.sin_tab,
                "all layers, including primary borrower, alias owner RoPE");
        require(bq.page_table != oq.page_table && bq.idx_tail != oq.idx_tail && bq.idx_dead != oq.idx_dead &&
                bq.idx_pooled != oq.idx_pooled && bq.idx_block_pos != oq.idx_block_pos && bq.step != oq.step &&
                bq.attention_status != oq.attention_status && bq.pos_dev != oq.pos_dev &&
                bq.host_step != oq.host_step && bq.host_pos != oq.host_pos, "indexer, counts, status and staging private");
        require(!bq.host.present() && !bq.map.slot_block, "borrower has no streamed host pools or residency arrays");
        for (auto p : pools(bq, g)) {
            require(owner.contains(p.ptr, p.bytes * size_t(bq.n_slots)), "physical pool lies in owner arena");
            require(!borrower.contains(p.ptr, p.bytes), "no physical pool lies in borrowed arena");
        }
        const auto qb = qsa_state_bytes(g, kCells, true, 0, &oq);
        require(qb == qsa_state_bytes(g, kCells, false, 0, &oq), "borrowed sizing never charges for RoPE");
        // A standalone exact-sized QSA carve complements the session-sized check.
        void* arena = nullptr;
        checked(cudaMalloc(&arena, size_t(qb) + kGuard), "allocate standalone borrowed QSA");
        QsaState standalone;
        fill({arena, size_t(qb) + kGuard}, 0xa5);
        const auto used = qsa_state_init(g, kCells, arena, standalone, nullptr, 0, &oq);
        require(used == qb, "borrowed QSA sizing exactly equals init consumption");
        require(standalone.cos_tab == oq.cos_tab && !standalone.owns_rope, "standalone primary borrows RoPE");
        uniform({static_cast<uint8_t*>(arena) + qb, kGuard}, 0xa5, "standalone QSA end guard intact");
        checked(cudaFreeHost(standalone.host_step), "free standalone step staging");
        checked(cudaFreeHost(standalone.host_pos), "free standalone position staging");
        checked(cudaFree(arena), "free standalone QSA arena");
    }
    for (auto p : private_spans(b, g)) require(borrower.contains(p.ptr, p.bytes), "private state lies in borrower arena");
    mapping(b, {});
}
void dirty_private(SessionState& s, const ModelGeometry& g, uint8_t value) {
    for (auto p : private_spans(s, g)) fill(p, value);
    for (int64_t j = 0; j < s.qsa_alloc; ++j) {
        auto& q = s.qsa_states[s.qsa_ord0 + j];
        std::memset(q.host_step, value, strata::kernels::qsa_step_bytes() + sizeof(int32_t));
        std::memset(q.host_pos, value, size_t(g.n_head) * sizeof(int32_t));
    }
}
void reset_private(SessionState& s, const ModelGeometry& g) {
    dirty_private(s, g, 0x4f);
    session_zero(s, g, nullptr, nullptr); sync();
    for (auto p : private_spans(s, g)) uniform(p, 0, "shared zero clears private recurrence/indexer/counts/status");
    for (int64_t j = 0; j < s.qsa_alloc; ++j) {
        const auto& q = s.qsa_states[s.qsa_ord0 + j];
        for (int i = 0; i <= strata::kernels::kStepCount; ++i) require(q.host_step[i] == 0, "host step staging reset");
        for (int64_t i = 0; i < g.n_head; ++i) require(q.host_pos[i] == 0, "host position staging reset");
    }
}
void resident_rejections(const ModelGeometry& g, const SessionState& owner) {
    const auto& oq = owner.qsa_states[owner.qsa_primary()];
    for (int mode : {1, 2}) {
        QsaState bad = oq, destination; bad.kv_mode = mode;
        require(qsa_state_bytes(g, kCells, true, 0, &bad) == 0, "streamed/ring owner sizing rejected");
        require(qsa_state_init(g, kCells, nullptr, destination, nullptr, 0, &bad) == 0 && !destination.host_step,
                "streamed/ring owner rejected before allocation");
    }
    require(qsa_state_bytes(g, kCells, true, 4, &oq) == 0, "borrowed ring request rejected");
    require(qsa_state_bytes(g, kCells + 1, true, 0, &oq) == 0, "borrowed logical capacity cannot exceed owner");
    ModelGeometry bad = g; bad.head_dim += 64;
    SessionState destination;
    require(session_bytes(bad, kCells, 10, kLayerLo, kLayerHi, &owner) == 0, "mismatched geometry sizing rejected");
    require(session_init(bad, kCells, 10, nullptr, destination, kLayerLo, kLayerHi, &owner) == 0 &&
            !destination.qsa_states, "mismatched geometry init rejected before allocation");
}

void runtime_cow(SharedKvRuntime& runtime, SessionState& owner, SessionState& borrower, const ModelGeometry& g) {
    std::string error;
    const size_t capacity = runtime.pages().capacity();
    require(runtime.ensure(0, 0, 6, error), "reserve owner partial-tail prefix");
    const auto original = runtime.pages().mapping(0);
    require(original.size() == 2 && runtime.pages().used_pages() == 2, "two physical pages reserved for six cells");
    // Distinct byte patterns by layer/plane/logical page catch missing scale or V copies.
    std::vector<std::vector<std::vector<uint8_t>>> expected;
    for (int64_t j = 0; j < owner.qsa_alloc; ++j) {
        auto planes = pools(owner.qsa_states[owner.qsa_ord0 + j], g);
        expected.emplace_back();
        for (size_t plane = 0; plane < planes.size(); ++plane) {
            std::vector<uint8_t> bytes(planes[plane].bytes);
            for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(11 + j * 31 + plane * 17 + i % 251);
            expected.back().push_back(bytes);
            const auto tail = page(planes[plane], original[1]);
            checked(cudaMemcpy(tail.ptr, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "seed owner tail page");
        }
    }
    require(runtime.clone_prefix(0, 1, 6, error), "clone partial-tail prefix without copies");
    require(runtime.pages().mapping(1) == original && runtime.pages().used_pages() == 2,
            "prefix clone shares physical pages");
    mapping(owner, original); mapping(borrower, original);
    require(runtime.ensure(1, 6, 7, error), "append to shared partial tail performs COW");
    const auto cow = runtime.pages().mapping(1);
    require(cow.size() == 2 && cow[0] == original[0] && cow[1] != original[1], "only written tail is private");
    require(runtime.pages().mapping(0) == original && runtime.pages().used_pages() == 3,
            "COW preserves owner map and consumes one physical page");
    mapping(owner, original); mapping(borrower, cow);
    for (int64_t j = 0; j < owner.qsa_alloc; ++j) {
        auto planes = pools(owner.qsa_states[owner.qsa_ord0 + j], g);
        for (size_t plane = 0; plane < planes.size(); ++plane) {
            require(read(page(planes[plane], cow[1])) == expected[size_t(j)][plane], "COW copied whole K/V/scale page");
            fill(page(planes[plane], cow[1]), 0xe7);
            require(read(page(planes[plane], original[1])) == expected[size_t(j)][plane], "COW write cannot change owner tail");
        }
    }
    require(runtime.release(1, error) && runtime.release(1, error), "borrower page release is idempotent");
    require(runtime.pages().used_pages() == 2, "release frees only borrower COW page");
    mapping(borrower, {}); mapping(owner, original);
    require(runtime.ensure(1, 0, 4, error), "reserve recycled page");
    require(runtime.pages().mapping(1) == std::vector<int32_t>{cow[1]}, "freed COW page recycled by LIFO allocator");
    mapping(borrower, runtime.pages().mapping(1));
    require(runtime.truncate(0, 4, error), "truncate owner tail");
    require(runtime.ensure(0, 4, 8, error), "regrow owner using recycled tail");
    require(runtime.pages().mapping(0) == original, "truncated owner page recycled without moving prefix");
    mapping(owner, original);
    require(runtime.release(0, error) && runtime.release(1, error), "release all reservations");
    require(runtime.pages().used_pages() == 0 && runtime.pages().free_pages() == capacity, "all physical pages recycled");
    mapping(owner, {}); mapping(borrower, {});
    require(runtime.ensure(0, 0, kCells, error), "reserve entire existing physical pool");
    const auto full = runtime.pages().mapping(0);
    require(!runtime.ensure(1, 0, 4, error) && !error.empty(), "exhaustion fails without growing physical pool");
    require(runtime.pages().mapping(0) == full && runtime.pages().mapping(1).empty() &&
            runtime.pages().free_pages() == 0, "failed reservation leaves allocator and maps unchanged");
    mapping(owner, full); mapping(borrower, {});
    require(runtime.release(0, error), "release full-capacity reservation");
}

void format_case(int format) {
    const ModelGeometry g;
    configure(format);
    DeviceSession owner; owner.init(g);
    identity(owner.state);
    const auto& primary = owner.state.qsa_states[owner.state.qsa_primary()];
    const float* cos = primary.cos_tab; const float* sin = primary.sin_tab;
    require(primary.owns_rope, "split-stage primary owns RoPE");
    require(strata::kernels::rope_table_for(strata::kernels::rope_scaling()).cos == cos, "owner RoPE registered");
    {
        DeviceSession borrower; borrower.init(g, &owner.state);
        aliases(owner, borrower, g, format);
        resident_rejections(g, owner.state);
        SharedKvRuntime runtime(g, {&owner.state, &borrower.state});
        require(runtime.pages().free_pages() == size_t(kCells / 4), "runtime uses owner's existing full-capacity pool");
        mapping(owner.state, {}); mapping(borrower.state, {});
        std::string error;
        require(runtime.ensure(0, 0, 6, error) && runtime.clone_prefix(0, 1, 6, error), "map shared pages for zero test");
        const auto shared = runtime.pages().mapping(0);
        for (int64_t j = 0; j < owner.state.qsa_alloc; ++j)
            for (auto p : pools(owner.state.qsa_states[owner.state.qsa_ord0 + j], g))
                fill({p.ptr, p.bytes * size_t(primary.n_slots)}, 0x5d);
        // Owner and borrower are now marked shared by the runtime. Both resets
        // must preserve physical bytes AND host/device logical page tables.
        dirty_private(owner.state, g, 0x37);
        reset_private(borrower.state, g);
        for (auto p : private_spans(owner.state, g))
            uniform(p, 0x37, "borrower reset cannot change owner recurrence/indexer/private state");
        for (int64_t j = 0; j < owner.state.qsa_alloc; ++j) {
            const auto& q = owner.state.qsa_states[owner.state.qsa_ord0 + j];
            require(std::all_of(q.host_step, q.host_step + strata::kernels::kStepCount + 1,
                    [](int32_t x) { return x == 0x37373737; }), "borrower reset cannot change owner host steps");
            require(std::all_of(q.host_pos, q.host_pos + g.n_head,
                    [](int32_t x) { return x == 0x37373737; }), "borrower reset cannot change owner host positions");
        }
        reset_private(owner.state, g);
        mapping(owner.state, shared); mapping(borrower.state, shared);
        for (int64_t j = 0; j < owner.state.qsa_alloc; ++j)
            for (auto p : pools(owner.state.qsa_states[owner.state.qsa_ord0 + j], g))
                uniform({p.ptr, p.bytes * size_t(primary.n_slots)}, 0x5d, "shared zero never clears owner's physical KV");
        require(runtime.release(1, error) && runtime.release(0, error), "release zero-test reservations");
        runtime_cow(runtime, owner.state, borrower.state, g);
        owner.guard(); borrower.guard();
        session_release(borrower.state);
        session_release(borrower.state);
        const auto table = strata::kernels::rope_table_for(strata::kernels::rope_scaling());
        require(table.cos == cos && table.sin == sin && primary.owns_rope,
                "borrower release never unregisters or takes ownership of owner's RoPE");
    } // Borrower arena/staging freed before owner; its destructor also calls session_release.
    require(strata::kernels::rope_table_for(strata::kernels::rope_scaling()).cos == cos,
            "borrower destruction leaves live owner RoPE registered");
    session_release(owner.state);
    require(!primary.owns_rope && strata::kernels::rope_table_for(strata::kernels::rope_scaling()).cos == nullptr,
            "owner release unregisters nonzero-global-ordinal primary RoPE");
    session_release(owner.state);
    owner.guard();
    std::printf("PASS borrowed session format %d: sizing, aliases, private state, shared zero, COW/recycling, release\n", format);
}
// Streaming fixtures keep the real 20,480-cell resident floor, but narrow model
// widths so the FP16 authoritative pinned pool is only 16 MiB per QSA layer.
constexpr int64_t kStreamCells = 32768;
ModelGeometry stream_geometry() {
    ModelGeometry g;
    g.n_layers = 16; g.n_embd = 256; g.n_head = 4; g.n_head_kv = 1; g.head_dim = 128;
    g.idx_q_heads = 1; g.idx_key_dim = 64; g.hc_lr = 32;
    g.ssm_state_size = 16; g.ssm_k_heads = 2; g.ssm_v_heads = 2;
    g.ssm_conv_channels = 96; g.ssm_value_dim = 32; g.n_ff = 256;
    return g;
}
struct DeviceQsa {
    QsaState state;
    void* arena = nullptr;
    uint64_t bytes = 0, used = 0;
    DeviceQsa() = default;
    DeviceQsa(const DeviceQsa&) = delete;
    DeviceQsa& operator=(const DeviceQsa&) = delete;
    ~DeviceQsa() {
        if (state.owns_rope) strata::kernels::rope_table_release(state.cos_tab);
        if (state.host_step) cudaFreeHost(state.host_step);
        if (state.host_pos) cudaFreeHost(state.host_pos);
        if (state.kv_host_arena) cudaFreeHost(state.kv_host_arena); // owner only, exactly once
        if (arena) cudaFree(arena);
    }
    void init(const ModelGeometry& g, int64_t cells, int64_t ring, const QsaState* rope,
              const QsaState* owner = nullptr) {
        bytes = qsa_state_bytes(g, cells, rope == nullptr, ring, owner);
        require(bytes != 0, "nonzero streamed/private QSA size");
        checked(cudaMalloc(&arena, size_t(bytes) + kGuard), "allocate streamed/private QSA arena");
        fill({arena, size_t(bytes) + kGuard}, 0xa5);
        used = qsa_state_init(g, cells, arena, state, rope, ring, owner);
        require(used != 0 && used <= bytes, "streamed/private QSA fits sized arena");
        if (owner || (state.kv_mode == 1 && state.shared_kv))
            require(used == bytes, "streamed owner/borrowed QSA has exact aligned sizing");
        uniform({static_cast<uint8_t*>(arena) + bytes, kGuard}, 0xa5, "streamed/private QSA end guard");
    }
};
std::vector<Span> host_pools(const QsaState& q, const ModelGeometry& g) {
    QsaState host = q;
    host.k_pool = q.host.k_pool; host.v_pool = q.host.v_pool;
    host.k_q = q.host.k_q; host.v_q = q.host.v_q;
    host.k_scale = q.host.k_scale; host.v_scale = q.host.v_scale;
    host.k_q4 = q.host.k_q4; host.v_q4 = q.host.v_q4;
    return pools(host, g);
}
std::vector<Span> clock_spans(const QsaState& q) {
    const size_t slots = size_t(q.map.n_slots) * sizeof(int32_t);
    return {{q.map.page_table, size_t(q.map.n_blocks) * sizeof(int32_t)},
        {q.map.slot_block, slots}, {q.map.slot_stamp, slots}, {q.map.slot_ref, slots},
        {q.map.miss_block, slots}, {q.map.miss_slot, slots},
        {q.map.ctl, size_t(strata::kernels::kKvCtlInts) * sizeof(int32_t)}};
}
void write_int(const int32_t* table, size_t at, int32_t value) {
    checked(cudaMemcpy(const_cast<int32_t*>(table) + at, &value, sizeof(value), cudaMemcpyHostToDevice),
            "write allocation-time mapping binding");
}
void ints(const int32_t* table, const std::vector<int32_t>& expected, const char* label) {
    std::vector<int32_t> actual(expected.size());
    checked(cudaMemcpy(actual.data(), table, actual.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            "read allocation-time mapping binding");
    require(actual == expected, label);
}
void stream_bindings(const QsaState& q) {
    require(q.kv_mode == 1 && q.shared_kv && q.host.present(), "allocation immediately marks unified-stream state");
    require(q.page_table && q.host.logical_pages && q.map.page_table &&
            q.page_table != q.host.logical_pages && q.page_table != q.map.page_table &&
            q.host.logical_pages != q.map.page_table, "read, backing and global residency tables are distinct");
    require(q.host.resident_pages == q.map.page_table && q.host.n_logical_pages == q.n_pages &&
            q.host.n_backing_pages == q.map.n_blocks && q.host.n_resident_slots == q.n_slots,
            "fixed writer bindings and capacities established before capture");
    const std::vector<int32_t> empty(size_t(q.n_pages), -1);
    ints(q.page_table, empty, "private GPU read view initially unmapped");
    ints(q.host.logical_pages, empty, "private backing view initially unmapped");
    require(q.shared_page_table == empty, "CPU mirror initially unmapped backing IDs");
}
void stream_rejections(const ModelGeometry& g, const QsaState& owner) {
    auto reject = [&](const QsaState& bad, const char* label) {
        QsaState destination;
        require(qsa_state_bytes(g, kStreamCells, true, 0, &bad) == 0, label);
        require(qsa_state_init(g, kStreamCells, nullptr, destination, nullptr, 0, &bad) == 0 &&
                !destination.host_step && !destination.kv_host_arena, "invalid stream borrow rejected before allocation");
    };
    { auto bad = owner; bad.shared_kv = false; reject(bad, "unmarked streamed owner rejected"); }
    { auto bad = owner; bad.page_table = nullptr; reject(bad, "missing private GPU read view rejected"); }
    { auto bad = owner; bad.host.v_pool = nullptr; bad.host.v_q = nullptr; bad.host.v_q4 = nullptr;
      reject(bad, "incomplete authoritative host payload rejected"); }
    { auto bad = owner; bad.host.logical_pages = nullptr; reject(bad, "legacy/identity streamed owner rejected"); }
    { auto bad = owner; bad.host.resident_pages = nullptr; reject(bad, "missing global writer binding rejected"); }
    { auto bad = owner; bad.map.page_table = bad.page_table; reject(bad, "aliased global/read table rejected"); }
    { auto bad = owner; bad.host.n_backing_pages = 0; reject(bad, "missing backing capacity rejected"); }
    { auto bad = owner; bad.host.n_resident_slots = 0; reject(bad, "missing resident capacity rejected"); }
    { auto bad = owner; bad.map.slot_ref = nullptr; reject(bad, "incomplete global CLOCK map rejected"); }
    { auto bad = owner; bad.kv_hybrid = true; reject(bad, "hybrid streaming borrower rejected"); }
    qsa_set_kv_resident(qsa_kv_resident_min() + 4);
    reject(owner, "different resident slot capacity rejected");
    qsa_set_kv_resident(0);
    reject(owner, "different requested residency mode rejected");
    qsa_set_kv_resident(1);
    require(qsa_state_bytes(g, kStreamCells, true, 16, &owner) == 0, "streaming borrowing never applies to MTP ring");
}
void mtp_ring_boundaries() {
    static_assert(mtp_kv_ring_cells(32768, 65536, 4) == 32848, "MTP ring helper is constexpr");
    struct Case { int64_t window, max_cells; int max_t; int64_t expected; };
    // Literal expectations test the production helper, not a copied ternary.
    constexpr Case cases[] = {
        {-1, 65536, 4, -1},
        {0, 65536, 1, -1},
        {0, 65536, 8, -1},
        {65536, 65536, 1, -1},
        {65536, 65536, 8, -1},
        {65537, 65536, 4, -1},
        {32768, 32768, 4, -1}, // default window equals context
        {32768, 16384, 4, -1}, // default window exceeds context
        {32768, 65536, 1, 32836}, // default window remains bounded
        {32768, 65536, 4, 32848},
        {32768, 65536, 8, 32864},
        {1, 65536, 1, 69},
        {65535, 65536, 8, 65631}, // retain write-ahead padding; do not clamp to context
    };
    for (const auto& c : cases)
        require(mtp_kv_ring_cells(c.window, c.max_cells, c.max_t) == c.expected,
                "production MTP ring helper preserves full-context/bounded-window boundaries and max_t padding");
}
void private_mtp(const ModelGeometry& g, const QsaState& rope) {
    for (bool ring : {false, true}) {
        qsa_set_kv_resident(ring ? 1 : 0);
        const int64_t cells = ring ? kStreamCells : kCells;
        qsa_set_kv_unified(false);
        DeviceQsa legacy; legacy.init(g, cells, 16, &rope);
        qsa_set_kv_unified(true);
        DeviceQsa opted; opted.init(g, cells, 16, &rope);
        require(legacy.bytes == opted.bytes && legacy.used == opted.used,
                "unified opt-in does not change private MTP sizing/init");
        require(read({legacy.arena, size_t(legacy.bytes)}) == read({opted.arena, size_t(opted.bytes)}),
                "private MTP arena byte-identical before zero with opt-in off/on");
        for (auto* q : {&legacy.state, &opted.state}) {
            require(!q->shared_kv && q->shared_page_table.empty() && !q->host.logical_pages &&
                    !q->host.resident_pages, "private MTP retains null legacy writer bindings");
            require(q->kv_mode == (ring ? 2 : 0) && q->map.page_table == q->page_table,
                    "private MTP retains identity/ring allocation and legacy map alias");
            std::vector<int32_t> expected(size_t(q->n_pages));
            for (int64_t i = 0; i < q->n_pages; ++i) expected[size_t(i)] = int32_t(ring ? i % q->n_slots : i);
            ints(q->page_table, expected, "private MTP table retains identity or modulo ring");
            require(q->cos_tab == rope.cos_tab && !q->owns_rope, "private MTP still borrows main RoPE only");
            require(!ring || (q->kv_host_arena && q->kv_host_arena != rope.kv_host_arena),
                    "private MTP owns independent authoritative pinned storage");
            qsa_state_zero(*q, g, nullptr);
        }
        sync();
        require(read({legacy.arena, size_t(legacy.bytes)}) == read({opted.arena, size_t(opted.bytes)}),
                "private MTP arena byte-identical after zero with opt-in off/on");
    }
    qsa_set_kv_resident(1);
}
void full_context_mtp(const ModelGeometry& g, const QsaState& main) {
    require(main.kv_mode == 1 && main.shared_kv && qsa_kv_unified() && qsa_kv_resident() > 0,
            "full-context MTP fixture retains unified-stream main allocation settings");
    const auto pinned = qsa_kv_host_bytes();
    const auto main_view = read({main.page_table, size_t(main.n_pages) * sizeof(int32_t)});
    const auto backing = read({const_cast<int32_t*>(main.host.logical_pages), size_t(main.n_pages) * sizeof(int32_t)});
    const auto mirror = main.shared_page_table;
    std::vector<std::vector<uint8_t>> clock;
    for (auto p : clock_spans(main)) clock.push_back(read(p));
    QsaState main_copy = main; // Read-only pointer view; QsaState itself does not free payloads.
    const auto main_planes = pools(main_copy, g);
    std::vector<std::vector<uint8_t>> active;
    for (auto p : main_planes) active.push_back(read(page(p, 2))); // seeded backing 17 remains active in slot 2

    // MtpDrafter::load uses -1 for window 0 or >= max_cells. Unlike ring 0,
    // -1 must force private mode 0 without disabling the main stream settings.
    DeviceQsa draft; draft.init(g, kStreamCells, -1, &main);
    auto& q = draft.state;
    require(q.kv_mode == 0 && !q.shared_kv && q.shared_page_table.empty(), "full-context MTP is private resident, not an unpublished streamed owner");
    require(q.n_slots == q.n_pages && q.max_cells == kStreamCells, "full-context MTP retains full requested physical capacity");
    require(q.kv_int8 == main.kv_int8 && q.kv_q4 == main.kv_q4 && !q.kv_hybrid,
            "full-context MTP preserves the configured whole K/V format");
    require(!q.host.present() && !q.kv_host_arena && !q.host.logical_pages && !q.host.resident_pages &&
            !q.map.slot_block && !q.map.ctl, "full-context MTP needs no pinned backing or CLOCK/resolver state");
    require(q.page_table != main.page_table && q.page_table != main.map.page_table && q.map.page_table == q.page_table,
            "full-context MTP reader table is private identity storage");
    require(q.cos_tab == main.cos_tab && q.sin_tab == main.sin_tab && !q.owns_rope,
            "full-context MTP borrows only main RoPE");
    require(q.idx_tail != main.idx_tail && q.idx_dead != main.idx_dead && q.idx_pooled != main.idx_pooled,
            "full-context MTP indexer allocation is private");
    const auto planes = pools(q, g);
    for (size_t i = 0; i < planes.size(); ++i) {
        require(planes[i].ptr != main_planes[i].ptr, "full-context MTP payload plane does not alias main GPU slots");
        fill({planes[i].ptr, planes[i].bytes * size_t(q.n_slots)}, 0x7d);
    }
    const auto shape = strata::kernels::qsa_real_shapes();
    fill({q.idx_tail, size_t((shape.idx_block - 1) * g.idx_key_dim) * sizeof(float)}, 0x7d);
    fill({q.idx_dead, size_t(g.idx_key_dim) * sizeof(float)}, 0x7d);
    fill({q.idx_pooled, size_t(q.idx_pooled_rows * g.idx_key_dim) * sizeof(float)}, 0x7d);
    std::vector<int32_t> identity(size_t(q.n_pages));
    for (int64_t i = 0; i < q.n_pages; ++i) identity[size_t(i)] = int32_t(i);
    ints(q.page_table, identity, "full-context MTP starts with identity addressing");
    qsa_state_zero(q, g, nullptr); sync();
    for (auto p : planes) uniform({p.ptr, p.bytes * size_t(q.n_slots)}, 0, "full-context MTP zero clears its private physical K/V and scales");
    uniform({q.idx_tail, size_t((shape.idx_block - 1) * g.idx_key_dim) * sizeof(float)}, 0, "full-context MTP zero clears private tail");
    uniform({q.idx_dead, size_t(g.idx_key_dim) * sizeof(float)}, 0, "full-context MTP zero clears private spare key");
    uniform({q.idx_pooled, size_t(q.idx_pooled_rows * g.idx_key_dim) * sizeof(float)}, 0, "full-context MTP zero clears private pooled indexer");
    ints(q.page_table, identity, "full-context MTP zero preserves identity addressing");
    uniform({static_cast<uint8_t*>(draft.arena) + draft.bytes, kGuard}, 0xa5, "full-context MTP zero preserves arena end guard");
    const auto spans = clock_spans(main);
    for (size_t i = 0; i < spans.size(); ++i) require(read(spans[i]) == clock[i], "full-context MTP zero preserves main global CLOCK/cache mapping");
    for (size_t i = 0; i < main_planes.size(); ++i) require(read(page(main_planes[i], 2)) == active[i], "full-context MTP zero preserves main active GPU payload");
    require(read({main.page_table, size_t(main.n_pages) * 4}) == main_view && main.shared_page_table == mirror &&
            read({const_cast<int32_t*>(main.host.logical_pages), size_t(main.n_pages) * 4}) == backing,
            "full-context MTP allocation/zero leaves main reader/backing views untouched");
    require(qsa_kv_host_bytes() == pinned && qsa_kv_unified() && qsa_kv_resident() > 0,
            "full-context MTP neither pins a stream pool nor disables main streaming globally");
}
struct DeviceSelection {
    int32_t* step = nullptr;
    DeviceSelection() {
        checked(cudaMalloc(reinterpret_cast<void**>(&step), 32), "allocate resolve selection");
        fill({step, 32}, 0);
        write_int(step, strata::kernels::kStepNKv, 1);
        write_int(step, strata::kernels::kStepWidth, 1);
        write_int(step + strata::kernels::kStepCount, 0, 0);
    }
    ~DeviceSelection() { if (step) cudaFree(step); }
};
void resolve_stream(SessionState& owner, SessionState& borrower, const ModelGeometry& g) {
    DeviceSelection selection;
    const int32_t* ids = selection.step + strata::kernels::kStepCount;
    for (int64_t j = 0; j < owner.qsa_alloc; ++j) {
        auto& o = owner.qsa_states[owner.qsa_ord0 + j];
        auto& b = borrower.qsa_states[borrower.qsa_ord0 + j];
        // Reader tables can be stale; the global backing cache, never another
        // sequence's view, decides residency. Hit 17, then miss 19 at logical 0.
        b.shared_page_table[0] = 17; write_int(b.host.logical_pages, 0, 17);
        qsa_kv_resolve(b, g, ids, selection.step, 1, 1, nullptr); sync();
        auto view = std::vector<int32_t>(size_t(b.n_pages), -1); view[0] = 2;
        ints(b.page_table, view, "resolve materializes private view from shared backing cache hit");
        b.shared_page_table[0] = 19; write_int(b.host.logical_pages, 0, 19);
        qsa_kv_resolve(b, g, ids, selection.step, 1, 1, nullptr); sync();
        std::vector<int32_t> global(size_t(o.map.n_blocks));
        checked(cudaMemcpy(global.data(), o.map.page_table, global.size() * 4, cudaMemcpyDeviceToHost), "read resolved global cache");
        require(global[17] == 2 && global[19] >= 0 && global[19] < o.n_slots && global[19] != 2,
                "resolve of borrower miss preserves other sequence's active cache page");
        view[0] = global[19]; ints(b.page_table, view, "resolve publishes borrower backing ID as GPU slot");
        std::vector<int32_t> controls(strata::kernels::kKvCtlInts);
        checked(cudaMemcpy(controls.data(), o.map.ctl, controls.size() * 4, cudaMemcpyDeviceToHost), "read resolve status");
        require(controls[3] == 0, "mapped resolve succeeds without overflow");
        for (auto p : pools(o, g)) {
            uniform(page(p, 2), 0x6b, "resolve never overwrites other sequence's active GPU payload");
            uniform(page(p, global[19]), 0x5d, "resolve copies correct authoritative backing page to GPU");
        }
        std::vector<std::vector<uint8_t>> before;
        for (auto p : clock_spans(o)) before.push_back(read(p));
        qsa_state_zero(b, g, nullptr); sync();
        const auto spans = clock_spans(o);
        for (size_t i = 0; i < spans.size(); ++i) require(read(spans[i]) == before[i], "zero preserves populated resolve CLOCK state");
        ints(b.page_table, view, "zero preserves resolved borrower GPU read view");
        ints(b.host.logical_pages, b.shared_page_table, "zero preserves resolved borrower backing mapping");
    }
}
void streaming_case(int format) {
    const auto g = stream_geometry();
    configure(format); qsa_set_kv_resident(1); qsa_set_kv_unified(true);
    const uint64_t pinned_before = qsa_kv_host_bytes();
    DeviceSession owner; owner.init(g, nullptr, kStreamCells);
    const auto& primary = owner.state.qsa_states[owner.state.qsa_primary()];
    require(primary.n_slots == qsa_kv_resident_min() / 4 && primary.n_pages == kStreamCells / 4,
            "shared-stream has one bounded GPU cache and full backing capacity");
    auto shape = strata::kernels::qsa_real_shapes();
    shape.n_head_kv = g.n_head_kv; shape.head_dim = g.head_dim;
    const uint64_t pin_per_layer = uint64_t(primary.n_pages) * strata::kernels::kv_block_bytes(shape, qsa_kv_format(primary)) + 4 * 256;
    require(qsa_kv_host_bytes() - pinned_before == uint64_t(owner.state.qsa_alloc) * pin_per_layer,
            "authoritative pinned payload charged once per owner layer");
    std::vector<std::vector<std::vector<uint8_t>>> clock;
    for (int64_t j = 0; j < owner.state.qsa_alloc; ++j) {
        auto& q = owner.state.qsa_states[owner.state.qsa_ord0 + j];
        stream_bindings(q);
        require(q.kv_host_arena, "owner retains raw pinned allocation handle for external cleanup");
        ints(q.map.page_table, std::vector<int32_t>(size_t(q.map.n_blocks), -1), "owner resets global residency once");
        ints(q.map.slot_block, std::vector<int32_t>(size_t(q.n_slots), -1), "owner resets CLOCK slot ownership once");
        ints(q.map.ctl, std::vector<int32_t>(strata::kernels::kKvCtlInts, 0), "owner resets CLOCK controls once");
        q.shared_page_table[0] = 17;
        write_int(q.host.logical_pages, 0, 17); write_int(q.page_table, 0, 2);
        write_int(q.map.page_table, 17, 2); write_int(q.map.slot_block, 2, 17);
        write_int(q.map.slot_stamp, 2, 5); write_int(q.map.slot_ref, 2, 1);
        write_int(q.map.ctl, 0, 5); write_int(q.map.ctl, 1, 3);
        fill({q.map.miss_block, size_t(q.n_slots) * 4}, 0x53);
        fill({q.map.miss_slot, size_t(q.n_slots) * 4}, 0x53);
        for (auto p : pools(q, g)) fill({p.ptr, p.bytes * size_t(q.n_slots)}, 0x6b);
        for (auto p : host_pools(q, g)) fill({p.ptr, p.bytes * size_t(q.map.n_blocks)}, 0x5d);
        clock.emplace_back();
        for (auto p : clock_spans(q)) clock.back().push_back(read(p));
    }
    const uint64_t pinned_owner = qsa_kv_host_bytes();
    {
        // Borrowing depends on the owner's explicit allocation, not today's opt-in flag.
        qsa_set_kv_unified(false);
        DeviceSession borrower; borrower.init(g, &owner.state, kStreamCells);
        qsa_set_kv_unified(true);
        require(qsa_kv_host_bytes() == pinned_owner, "borrower pins no duplicate authoritative host pool");
        require(borrower.bytes < owner.bytes, "borrower omits GPU payload/global map/RoPE storage");
        for (int64_t j = 0; j < owner.state.qsa_alloc; ++j) {
            const auto& o = owner.state.qsa_states[owner.state.qsa_ord0 + j];
            auto& b = borrower.state.qsa_states[borrower.state.qsa_ord0 + j];
            stream_bindings(b);
            require(!b.kv_host_arena && !b.owns_rope && b.cos_tab == o.cos_tab && b.sin_tab == o.sin_tab,
                    "borrower owns no payload handle or RoPE registration");
            require(b.page_table != o.page_table && b.host.logical_pages != o.host.logical_pages,
                    "each sequence has private read/backing device tables");
            require(b.map.page_table == o.map.page_table && b.map.slot_block == o.map.slot_block &&
                    b.map.slot_stamp == o.map.slot_stamp && b.map.slot_ref == o.map.slot_ref &&
                    b.map.miss_block == o.map.miss_block && b.map.miss_slot == o.map.miss_slot && b.map.ctl == o.map.ctl,
                    "borrower aliases entire global CLOCK cache");
            require(b.kv_int8 == o.kv_int8 && b.kv_q4 == o.kv_q4 && b.kv_rot == o.kv_rot && b.n_slots == o.n_slots,
                    "streaming borrower inherits format and resident capacity");
            const auto gpu_o = pools(owner.state.qsa_states[owner.state.qsa_ord0 + j], g), gpu_b = pools(b, g);
            const auto host_o = host_pools(o, g), host_b = host_pools(b, g);
            for (size_t i = 0; i < gpu_o.size(); ++i) {
                require(gpu_o[i].ptr == gpu_b[i].ptr && host_o[i].ptr == host_b[i].ptr, "all GPU/host payload planes alias owner");
                require(!borrower.contains(gpu_b[i].ptr, gpu_b[i].bytes), "borrower arena contains no GPU payload");
            }
            require(borrower.contains(b.page_table, size_t(b.n_pages) * 4) &&
                    borrower.contains(b.host.logical_pages, size_t(b.n_pages) * 4), "both private tables carved from borrower arena");
            b.shared_page_table[0] = 19; write_int(b.host.logical_pages, 0, 19);
        }
        dirty_private(owner.state, g, 0x37);
        reset_private(borrower.state, g);
        for (auto p : private_spans(owner.state, g)) uniform(p, 0x37, "stream borrower zero preserves owner private state");
        reset_private(owner.state, g);
        for (int64_t j = 0; j < owner.state.qsa_alloc; ++j) {
            auto& o = owner.state.qsa_states[owner.state.qsa_ord0 + j];
            const auto& b = borrower.state.qsa_states[borrower.state.qsa_ord0 + j];
            auto read_view = std::vector<int32_t>(size_t(o.n_pages), -1); read_view[0] = 2;
            ints(o.page_table, read_view, "shared zero preserves owner's cached read view");
            ints(o.host.logical_pages, o.shared_page_table, "shared zero preserves owner's backing ownership");
            ints(b.page_table, std::vector<int32_t>(size_t(b.n_pages), -1), "shared zero preserves borrower read view");
            ints(b.host.logical_pages, b.shared_page_table, "shared zero preserves borrower backing ownership");
            const auto spans = clock_spans(o);
            for (size_t i = 0; i < spans.size(); ++i) require(read(spans[i]) == clock[size_t(j)][i],
                    "borrow init and shared zero never reset global residency/CLOCK");
            for (auto p : pools(o, g)) uniform({p.ptr, p.bytes * size_t(o.n_slots)}, 0x6b, "shared zero preserves other sequence GPU cache payload");
            for (auto p : host_pools(o, g)) uniform({p.ptr, p.bytes * size_t(o.map.n_blocks)}, 0x5d, "shared zero preserves authoritative pinned payload");
        }
        stream_rejections(g, primary);
        resolve_stream(owner.state, borrower.state, g);
        borrower.guard(); owner.guard();
        session_release(borrower.state);
        require(strata::kernels::rope_table_for(strata::kernels::rope_scaling()).cos == primary.cos_tab,
                "stream borrower release never unregisters owner RoPE");
    }
    // Exercise non-16-byte region sizes and a shorter logical borrower while
    // retaining the owner's full backing capacity and identical resident cache.
    {
        auto odd = g; odd.n_head = 3; odd.idx_key_dim = 65;
        DeviceQsa exact; exact.init(odd, kStreamCells - 1, 0, &primary);
        const auto before = qsa_kv_host_bytes();
        DeviceQsa shorter; shorter.init(odd, qsa_kv_resident_min() + 1, 0, nullptr, &exact.state);
        require(shorter.state.host.n_logical_pages < shorter.state.host.n_backing_pages &&
                shorter.state.map.page_table == exact.state.map.page_table, "shorter logical borrower keeps global backing capacity");
        require(qsa_kv_host_bytes() == before, "odd-sized borrower still pins no payload copy");
    }
    // A non-unified mode-1 owner must remain unborrowable even with opt-in on.
    qsa_set_kv_unified(false);
    {
        DeviceQsa legacy; legacy.init(g, kStreamCells, 0, &primary);
        require(!legacy.state.shared_kv && !legacy.state.host.logical_pages && !legacy.state.host.resident_pages &&
                legacy.state.map.page_table == legacy.state.page_table, "legacy stream layout unchanged");
        qsa_set_kv_unified(true);
        require(qsa_state_bytes(g, kStreamCells, false, 0, &legacy.state) == 0, "legacy streamed storage cannot be borrowed");
    }
    private_mtp(g, primary);
    full_context_mtp(g, primary);
    owner.guard();
    std::printf("PASS shared streaming format %d: exact sizing, private bindings, one pinned/global cache, safe zero, private MTP\n", format);
}
void hybrid_stream_rejected() {
    configure(3); qsa_set_kv_resident(1); qsa_set_kv_unified(true);
    const auto g = stream_geometry(); const auto before = qsa_kv_host_bytes();
    QsaState q; SessionState s;
    require(qsa_state_bytes(g, kStreamCells) == 0 && qsa_state_init(g, kStreamCells, nullptr, q) == 0,
            "unified K8V4 streaming remains unsupported");
    require(session_bytes(g, kStreamCells, 10, kLayerLo, kLayerHi) == 0 &&
            session_init(g, kStreamCells, 10, nullptr, s, kLayerLo, kLayerHi) == 0 && !s.qsa_states,
            "unsupported unified stream session rejected before carving");
    require(qsa_kv_host_bytes() == before && !q.kv_host_arena, "hybrid rejection does not pin payload memory");
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no CUDA device: shared_kv_session_test skipped\n");
        return 77;
    }
    // Set before the first RoPE registry lookup; the toggle is cached by kernels.
#if defined(_WIN32)
    _putenv_s("STRATA_ROPE_TABLE", "1");
#else
    setenv("STRATA_ROPE_TABLE", "1", 1);
#endif
    try {
        mtp_ring_boundaries();
        for (int format = 0; format < 4; ++format) format_case(format);
        for (int format = 0; format < 3; ++format) streaming_case(format);
        hybrid_stream_rejected();
        configure(0);
        std::printf("shared_kv_session_test: %zu checks passed\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "shared_kv_session_test FAIL: %s\n", error.what());
        return 1;
    }
}
