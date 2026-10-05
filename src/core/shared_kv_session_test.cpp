// CUDA-backed borrowed-session and unified-page regression test. Link against the
// engine's session/layer and CUDA kernel targets; no model weights are required.
// Uses real device allocation/transfers (not host link wrapping). Returns 77 when
// no CUDA device is available. CMake wiring belongs to the primary integration.
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
        }
        delete[] state.qsa_states;
        if (arena) cudaFree(arena);
    }
    void init(const ModelGeometry& g, const SessionState* owner = nullptr) {
        bytes = session_bytes(g, kCells, 10, kLayerLo, kLayerHi, owner);
        require(bytes != 0, "nonzero session size");
        checked(cudaMalloc(&arena, size_t(bytes) + kGuard), "allocate session arena");
        fill({arena, size_t(bytes) + kGuard}, 0xa5);
        used = session_init(g, kCells, 10, arena, state, kLayerLo, kLayerHi, owner);
        require(used != 0 && used <= bytes, "session init fits sized arena");
        if (owner) require(used == bytes, "borrowed session sizing exactly equals init consumption");
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
        for (int format = 0; format < 4; ++format) format_case(format);
        configure(0);
        std::printf("shared_kv_session_test: %zu checks passed\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "shared_kv_session_test FAIL: %s\n", error.what());
        return 1;
    }
}
