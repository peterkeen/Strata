// Model-free shared streaming regression. CUDA-only CMake/CTest target linked to strata_prefill (which
// provides native batch append) and strata_kernels. Also run kv_stream_parity for legacy attention/ring
// coverage. Returns 77 when no CUDA device/driver is available. One-layer staging is separate workspace, NOT
// part of the shared --kv-resident cache budget; decode below only resolves selected pages.
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/prefill/kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace k = strata::kernels;
namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
void require(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
struct Memory {
    std::vector<void*> device, pinned;
    ~Memory() {
        for (void* p : device) ck(cudaFree(p), "free");
        for (void* p : pinned) ck(cudaFreeHost(p), "free host");
    }
    template<class T> T* alloc(size_t count, bool host = false) {
        void *p = nullptr, *d = nullptr;
        const size_t bytes = count * sizeof(T) + 64; // trailing scratch canary, initially 0x5a
        if (host) {
            ck(cudaHostAlloc(&p, bytes, cudaHostAllocMapped), "host alloc");
            pinned.push_back(p);
            ck(cudaHostGetDevicePointer(&d, p, 0), "mapped pointer");
            std::memset(p, 0, bytes);
        } else {
            ck(cudaMalloc(&d, bytes), "device alloc"); device.push_back(d);
            ck(cudaMemset(d, 0, bytes), "zero");
        }
        ck(cudaMemset(static_cast<unsigned char*>(d) + count * sizeof(T), 0x5a, 64), "canary");
        return static_cast<T*>(d);
    }
};
template<class T> void put(T* dst, const std::vector<T>& src) {
    ck(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template<class T> std::vector<T> get(const T* src, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
void canary(const int32_t* p, size_t count) {
    const auto bytes = get(reinterpret_cast<const uint8_t*>(p + count), 64);
    require(std::all_of(bytes.begin(), bytes.end(), [](uint8_t b) { return b == 0x5a; }), "scratch canary");
}
struct Pools {
    k::KvHostPools p;
    std::vector<uint8_t*> arrays;
    std::vector<size_t> run; // bytes per page per array
    void alloc(Memory& mem, int pages, const k::QsaShapes& s, int fmt, bool host = false) {
        const size_t rows = s.n_head_kv * s.page_size;
        auto add = [&](size_t bytes) {
            auto* a = mem.alloc<uint8_t>(pages * bytes, host);
            arrays.push_back(a); run.push_back(bytes); return a;
        };
        if (fmt == k::kKvF16) {
            p.k_pool = reinterpret_cast<uint16_t*>(add(rows * s.head_dim * 2));
            p.v_pool = reinterpret_cast<uint16_t*>(add(rows * s.head_dim * 2));
        } else if (fmt == k::kKvInt8) {
            p.k_q = reinterpret_cast<int8_t*>(add(rows * s.head_dim));
            p.v_q = reinterpret_cast<int8_t*>(add(rows * s.head_dim));
            p.k_scale = reinterpret_cast<uint16_t*>(add(rows * (s.head_dim / 64) * 2));
            p.v_scale = reinterpret_cast<uint16_t*>(add(rows * (s.head_dim / 64) * 2));
        } else {
            p.k_q4 = add(rows * k::kv_q4_bytes_per_head((int) s.head_dim));
            p.v_q4 = add(rows * k::kv_q4_bytes_per_head((int) s.head_dim));
        }
    }
    k::QsaAttnPools attn(const int32_t* table) const {
        k::QsaAttnPools a;
        a.k_pool = p.k_pool; a.v_pool = p.v_pool; a.k_q = p.k_q; a.v_q = p.v_q;
        a.k_scale = p.k_scale; a.v_scale = p.v_scale; a.k_q4 = p.k_q4; a.v_q4 = p.v_q4;
        a.page_table = table; return a;
    }
    std::vector<uint8_t> page(int b) const {
        std::vector<uint8_t> out;
        for (size_t a = 0; a < arrays.size(); ++a) {
            auto bytes = get(arrays[a] + b * run[a], run[a]);
            out.insert(out.end(), bytes.begin(), bytes.end());
        }
        return out;
    }
    void copy_page(int dst, const Pools& source, int src) {
        for (size_t a = 0; a < arrays.size(); ++a)
            ck(cudaMemcpy(arrays[a] + dst * run[a], source.arrays[a] + src * run[a], run[a], cudaMemcpyDefault), "copy page");
    }
    std::vector<uint8_t> snapshot(int pages) const {
        std::vector<uint8_t> out;
        for (size_t a = 0; a < arrays.size(); ++a) {
            auto bytes = get(arrays[a], pages * run[a]); out.insert(out.end(), bytes.begin(), bytes.end());
        }
        return out;
    }
};
void append(Pools& p, const int32_t* table, const int32_t* step, const float* key, const float* val,
            const k::QsaShapes& s, int fmt, const k::KvHostPools* host, cudaStream_t cs) {
    if (fmt == k::kKvF16) k::kv_append_step(p.p.k_pool, p.p.v_pool, table, step, key, val, s, cs, host);
    else if (fmt == k::kKvInt8)
        k::kv_append_q8_step(p.p.k_q, p.p.v_q, p.p.k_scale, p.p.v_scale, table, step, key, val, s, cs, host);
    else k::kv_append_q4_step(p.p.k_q4, p.p.v_q4, table, step, key, val, s, cs, host);
}
void batch(Pools& p, const int32_t* table, const float* key, const float* val, const k::QsaShapes& s,
           int fmt, const k::KvHostPools* host, const k::KvHostPools* stage, cudaStream_t cs) {
    if (fmt == k::kKvQ4) k::kv_append_q4(p.p.k_q4, p.p.v_q4, table, 0, 8, key, val, s, cs, host, stage);
    else strata::prefill::kv_append(key, val, 8, 0, table, s.page_size, p.p.k_pool, p.p.v_pool,
                                    p.p.k_q, p.p.v_q, p.p.k_scale, p.p.v_scale, cs, host, stage);
}
void gather(const Pools& p, const int32_t* table, const int32_t* ids, const int32_t* step,
            const k::QsaShapes& s, int fmt, uint16_t* key, uint16_t* val, cudaStream_t cs) {
    if (fmt == k::kKvF16)
        k::kv_gather_step(p.p.k_pool, p.p.v_pool, table, ids, step, 8, s, key, val, cs);
    else if (fmt == k::kKvInt8)
        k::kv_gather_q8_step(p.p.k_q, p.p.v_q, p.p.k_scale, p.p.v_scale, table, ids, step, 8, s, key, val, cs);
    else k::kv_gather_q4_step(p.p.k_q4, p.p.v_q4, table, ids, step, 8, s, key, val, cs);
}
void run(int fmt) {
    constexpr int L = 1536, B = L * 2, S = 1024, CAP = L;
    const auto s = k::qsa_real_shapes();
    Memory mem;
    cudaStream_t cs = nullptr; ck(cudaStreamCreate(&cs), "stream");
    Pools host, slots, ref_a, ref_b, stage;
    host.alloc(mem, B, s, fmt, true); slots.alloc(mem, S, s, fmt);
    ref_a.alloc(mem, L, s, fmt); ref_b.alloc(mem, L, s, fmt); stage.alloc(mem, L, s, fmt);
    k::KvStreamMap m;
    m.n_blocks = B; m.n_slots = S;
    m.page_table = mem.alloc<int32_t>(B); m.slot_block = mem.alloc<int32_t>(S);
    m.slot_stamp = mem.alloc<int32_t>(S); m.slot_ref = mem.alloc<int32_t>(S);
    m.miss_block = mem.alloc<int32_t>(S); m.miss_slot = mem.alloc<int32_t>(S); m.ctl = mem.alloc<int32_t>(k::kKvCtlInts);
    auto* logical_a = mem.alloc<int32_t>(L); auto* logical_b = mem.alloc<int32_t>(L);
    auto* view_a = mem.alloc<int32_t>(L); auto* view_b = mem.alloc<int32_t>(L); auto* identity = mem.alloc<int32_t>(L);
    std::vector<int32_t> la(L), lb(L), ident(L), minus(L, -1);
    std::iota(la.begin(), la.end(), 0); std::iota(lb.begin(), lb.end(), L); ident = la;
    put(logical_a, la); put(logical_b, lb); put(identity, ident); put(view_a, minus); put(view_b, minus);
    auto metadata = [&](int32_t* logical) {
        auto h = host.p; h.logical_pages = logical; h.resident_pages = m.page_table;
        h.n_logical_pages = L; h.n_backing_pages = B; h.n_resident_slots = S; return h;
    };
    auto ha = metadata(logical_a), hb = metadata(logical_b);
    auto* ids = mem.alloc<int32_t>(CAP * 2); auto* steps = mem.alloc<int32_t>(k::kStepCount * 2);
    auto* step = mem.alloc<int32_t>(k::kStepCount);
    auto* key = mem.alloc<float>(8 * s.n_head_kv * s.head_dim);
    auto* val = mem.alloc<float>(8 * s.n_head_kv * s.head_dim);
    auto* q = mem.alloc<float>(s.n_head * s.head_dim);
    auto* scratch = mem.alloc<float>(k::qsa_decode_attn_scratch_floats(CAP, s));
    auto* out = mem.alloc<float>(s.n_head * s.head_dim); auto* expected = mem.alloc<float>(s.n_head * s.head_dim);
    const size_t gather_values = 8 * s.n_head_kv * s.head_dim;
    auto* gather_k = mem.alloc<uint16_t>(gather_values); auto* gather_v = mem.alloc<uint16_t>(gather_values);
    auto* gather_ref_k = mem.alloc<uint16_t>(gather_values); auto* gather_ref_v = mem.alloc<uint16_t>(gather_values);
    put(q, std::vector<float>(s.n_head * s.head_dim, 0.25f));
    auto sync = [&] { ck(cudaStreamSynchronize(cs), "sync"); };
    auto values = [&](float scale) {
        std::vector<float> a(8 * s.n_head_kv * s.head_dim), b(a.size());
        for (size_t i = 0; i < a.size(); ++i) { a[i] = scale * float(int(i % 97) - 48) / 37.f; b[i] = scale * float(int(i % 53) - 26) / 19.f; }
        put(key, a); put(val, b);
    };
    auto select = [&](const std::vector<int32_t>& selected) {
        put(ids, selected);
        put(steps, std::vector<int32_t>{7, 8, 2, (int32_t) selected.size()});
    };
    auto resolve = [&](const k::KvHostPools& h, int32_t* view) {
        k::kv_stream_resolve(m, slots.attn(view), h, fmt, ids, steps, 1, CAP, s, cs); sync();
    };
    auto check_map = [&] {
        const auto pt = get(m.page_table, B), sb = get(m.slot_block, S);
        for (int b = 0; b < B; ++b) require(pt[b] >= -1 && pt[b] < S && (pt[b] < 0 || sb[pt[b]] == b), "global table inversion");
        for (int sl = 0; sl < S; ++sl) require(sb[sl] == -1 || (sb[sl] >= 0 && sb[sl] < B && pt[sb[sl]] == sl), "slot inversion");
        canary(m.miss_block, S); canary(m.miss_slot, S);
    };
    auto check_page = [&](int logical, const k::KvHostPools& h, int32_t* view, const Pools& ref) {
        const auto mapping = get(h.logical_pages, L); const auto pt = get(m.page_table, B), v = get(view, L);
        require(v[logical] >= 0 && v[logical] == pt[mapping[logical]], "materialized sequence view");
        require(slots.page(v[logical]) == ref.page(logical), "resident payload parity");
        require(host.page(mapping[logical]) == ref.page(logical), "authoritative backing payload parity");
    };
    auto attention = [&](const Pools& ref, int32_t* view) {
        k::qsa_decode_attn_batch(q, ref.attn(identity), ids, steps, CAP, s, scratch, expected, 1, cs);
        k::qsa_decode_attn_batch(q, slots.attn(view), ids, steps, CAP, s, scratch, out, 1, cs); sync();
        const auto a = get(expected, s.n_head * s.head_dim), b = get(out, a.size());
        require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, "bitwise attention parity");
    };
    k::kv_stream_reset(m, cs); sync();

    // Same logical ID, disjoint authoritative pages. Single writers must not cross-contaminate.
    put(step, std::vector<int32_t>{0, 1, 0, 1});
    values(1.f); append(ref_a, identity, step, key, val, s, fmt, nullptr, cs);
    append(slots, view_a, step, key, val, s, fmt, &ha, cs);
    values(3.f); append(ref_b, identity, step, key, val, s, fmt, nullptr, cs);
    append(slots, view_b, step, key, val, s, fmt, &hb, cs); sync();
    require(host.page(0) != host.page(L), "sequences must differ");
    select({0}); resolve(ha, view_a); check_page(0, ha, view_a, ref_a); attention(ref_a, view_a);
    require(get(view_a, L)[1] == -1, "resolve only materializes selected logical pages");
    resolve(hb, view_b); check_page(0, hb, view_b, ref_b); attention(ref_b, view_b); check_map();
    // Shared prefix/clone maps a different sequence onto the already cached authoritative backing page.
    const auto shared_before = k::kv_stream_counters(m);
    lb[0] = 0; put(logical_b, lb); resolve(hb, view_b); check_page(0, hb, view_b, ref_a);
    require(k::kv_stream_counters(m).misses == shared_before.misses &&
            get(view_a, L)[0] == get(view_b, L)[0], "shared backing reuses one global cache slot");
    lb[0] = L; put(logical_b, lb); resolve(hb, view_b);

    // Global CLOCK eviction across sequences. A's reader view deliberately remains stale.
    std::vector<int32_t> full(S); for (int i = 0; i < S; ++i) full[i] = i * 4;
    select(full); resolve(hb, view_b); check_map();
    require(get(m.page_table, B)[0] == -1, "B evicts A globally");
    const auto before = slots.snapshot(S);
    values(2.f); append(ref_a, identity, step, key, val, s, fmt, nullptr, cs);
    append(slots, view_a, step, key, val, s, fmt, &ha, cs); sync();
    require(slots.snapshot(S) == before, "evicted writer must not use stale reader slot");
    require(host.page(0) == ref_a.page(0), "evicted writer still updates backing");
    select({0}); resolve(ha, view_a); check_page(0, ha, view_a, ref_a);

    // Resident stale view: force A's reader slot to B's. Metadata-only HIP DMA writer still uses global A.
    const auto pt = get(m.page_table, B);
    // The CLOCK victim is intentionally nondeterministic (parallel miss reservation). Select an actual
    // surviving B page rather than assuming B's logical page 1 survived A's preceding resolve.
    int b_slot = -1;
    for (int b = L; b < B; ++b) if (pt[b] >= 0 && pt[b] != pt[0]) { b_slot = pt[b]; break; }
    require(b_slot >= 0, "distinct resident slots");
    auto wrong = get(view_a, L); wrong[0] = b_slot; put(view_a, wrong);
    const auto other = slots.page(b_slot), host_before = host.page(0);
    auto metadata_only = ha;
    metadata_only.k_pool = metadata_only.v_pool = metadata_only.k_scale = metadata_only.v_scale = nullptr;
    metadata_only.k_q = metadata_only.v_q = nullptr; metadata_only.k_q4 = metadata_only.v_q4 = nullptr;
    values(4.f); append(ref_a, identity, step, key, val, s, fmt, nullptr, cs);
    append(slots, view_a, step, key, val, s, fmt, &metadata_only, cs); sync();
    require(slots.page(pt[0]) == ref_a.page(0), "metadata-only writer uses global map");
    require(slots.page(b_slot) == other && host.page(0) == host_before, "metadata-only writer leaves unrelated rows alone");
    host.copy_page(0, ref_a, 0); // emulate caller's DMA publication
    resolve(ha, view_a); check_page(0, ha, view_a, ref_a);

    // All three native prefill batch formats: resident GPU target + extra logical identity stage.
    select({0, 4}); resolve(ha, view_a);
    wrong.assign(L, b_slot); put(view_a, wrong);
    values(5.f); batch(ref_a, identity, key, val, s, fmt, nullptr, nullptr, cs);
    batch(slots, view_a, key, val, s, fmt, &ha, &stage.p, cs); sync();
    resolve(ha, view_a);
    for (int i = 0; i < 2; ++i) { check_page(i, ha, view_a, ref_a); require(stage.page(i) == ref_a.page(i), "batch staging uses logical row"); }
    // Batch metadata-only must also update GPU + logical stage without writing host payload.
    const auto batch_host = host.page(0);
    values(6.f); batch(ref_a, identity, key, val, s, fmt, nullptr, nullptr, cs);
    batch(slots, view_a, key, val, s, fmt, &metadata_only, &stage.p, cs); sync();
    require(host.page(0) == batch_host, "batch metadata-only omits host writes");
    for (int i = 0; i < 2; ++i) {
        require(slots.page(get(m.page_table, B)[i]) == ref_a.page(i), "batch metadata-only global GPU slot");
        require(stage.page(i) == ref_a.page(i), "batch metadata-only logical staging");
        host.copy_page(i, stage, i);
    }
    select({0, 1, 2, 3, 4, 5, 6, 7}); resolve(ha, view_a); attention(ref_a, view_a);
    // The existing non-fused gather readers consume the very same materialized private table unchanged.
    gather(ref_a, identity, ids, steps, s, fmt, gather_ref_k, gather_ref_v, cs);
    gather(slots, view_a, ids, steps, s, fmt, gather_k, gather_v, cs); sync();
    require(get(gather_ref_k, gather_values) == get(gather_k, gather_values) &&
            get(gather_ref_v, gather_values) == get(gather_v, gather_values), "bitwise gather parity");

    // Reuse a captured writer -> resolve -> attention graph after COW/recycling publishes changed backing.
    select({0}); values(7.f);
    append(ref_a, identity, step, key, val, s, fmt, nullptr, cs); sync();
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal), "begin capture");
    append(slots, view_a, step, key, val, s, fmt, &ha, cs);
    k::kv_stream_resolve(m, slots.attn(view_a), ha, fmt, ids, steps, 1, CAP, s, cs);
    k::qsa_decode_attn_batch(q, slots.attn(view_a), ids, steps, CAP, s, scratch, out, 1, cs);
    ck(cudaStreamEndCapture(cs, &graph), "end capture");
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
    ck(cudaGraphLaunch(exec, cs), "first replay"); sync(); check_page(0, ha, view_a, ref_a);
    const auto old_backing = host.page(0);
    la[0] = B - 1; put(logical_a, la); host.copy_page(B - 1, ref_a, 0); // COW copies the whole page
    const auto counters = get(m.ctl, k::kKvCtlInts);
    k::kv_stream_invalidate(m, logical_a, 0, 1, cs); sync();
    require(get(m.ctl, k::kKvCtlInts) == counters, "range invalidation preserves counters/hand");
    values(8.f); append(ref_a, identity, step, key, val, s, fmt, nullptr, cs); sync();
    ck(cudaGraphLaunch(exec, cs), "remapped replay"); sync(); check_page(0, ha, view_a, ref_a);
    require(host.page(0) == old_backing, "COW graph writer leaves old backing unchanged");
    // Check captured attention BEFORE issuing any fresh attention launch.
    k::qsa_decode_attn_batch(q, ref_a.attn(identity), ids, steps, CAP, s, scratch, expected, 1, cs); sync();
    const auto ea = get(expected, s.n_head * s.head_dim), oa = get(out, ea.size());
    require(std::memcmp(ea.data(), oa.data(), ea.size() * sizeof(float)) == 0, "remapped captured attention parity");
    ck(cudaGraphExecDestroy(exec), "destroy exec"); ck(cudaGraphDestroy(graph), "destroy graph");

    // Device-list invalidation, duplicates, invalid IDs and no-op: last-ref release BEFORE ID reuse.
    auto* released = mem.alloc<int32_t>(5); put(released, std::vector<int32_t>{B - 1, B - 1, -1, B, 0});
    const auto ctl = get(m.ctl, k::kKvCtlInts); const int freed_slot = get(m.page_table, B)[B - 1];
    k::kv_stream_invalidate_pages(m, released, 0, cs);
    k::kv_stream_invalidate_pages(m, released, 5, cs); sync();
    require(get(m.page_table, B)[B - 1] == -1 && get(m.slot_block, S)[freed_slot] == -1 &&
            get(m.slot_stamp, S)[freed_slot] == -1 && get(m.slot_ref, S)[freed_slot] == 0, "list eviction clears metadata");
    require(get(m.ctl, k::kKvCtlInts) == ctl, "list invalidation preserves counters");
    host.copy_page(B - 1, ref_b, 0); // ID recycled, host payload replaced after release invalidation
    resolve(ha, view_a); check_page(0, ha, view_a, ref_b);
    // Snapshot-like host replacement while resident, then range invalidation must force recopy.
    host.copy_page(B - 1, ref_a, 0); k::kv_stream_invalidate(m, logical_a, 0, 1, cs);
    resolve(ha, view_a); check_page(0, ha, view_a, ref_a);

    // Stage/unstage mapped rows, including nonzero logical range and swapped backing rows.
    la[1] = 13; la[2] = 7; la[3] = 12; put(logical_a, la);
    k::kv_stage_from_host(stage.attn(identity), ha, fmt, 4, s, cs); sync();
    for (int i = 0; i < 4; ++i) require(stage.page(i) == host.page(la[i]), "mapped stage parity");
    const auto keep0 = host.page(la[0]), keep3 = host.page(la[3]);
    stage.copy_page(1, ref_b, 0); stage.copy_page(2, ref_a, 0);
    k::kv_unstage_to_host(stage.attn(identity), ha, fmt, 1, 3, s, cs); sync();
    require(host.page(la[1]) == stage.page(1) && host.page(la[2]) == stage.page(2), "mapped unstage parity");
    require(host.page(la[0]) == keep0 && host.page(la[3]) == keep3, "unstage range isolation");

    // Entire multi-query union protected: overlaps dedup, and a hit in the later query cannot be evicted.
    put(logical_a, ident); la = ident; k::kv_stream_reset(m, cs);
    select({1023 * 4}); resolve(ha, view_a); const int protected_slot = get(m.page_table, B)[1023];
    std::vector<int32_t> union_ids(2 * CAP, 0), union_steps(2 * k::kStepCount, 0);
    for (int i = 0; i < 600; ++i) { union_ids[i] = i * 4; union_ids[CAP + i] = (424 + i) * 4; }
    union_steps[k::kStepWidth] = union_steps[k::kStepCount + k::kStepWidth] = 600;
    put(ids, union_ids); put(steps, union_steps);
    k::kv_stream_resolve(m, slots.attn(view_a), ha, fmt, ids, steps, 2, CAP, s, cs); sync(); check_map();
    require(!k::kv_stream_counters(m).overflow && get(m.page_table, B)[1023] == protected_slot, "union hit protection");
    const auto union_view = get(view_a, L), union_pt = get(m.page_table, B);
    for (int i = 0; i < S; ++i)
        require(union_view[i] >= 0 && union_view[i] == union_pt[i], "union reader materialization");

    // Oversized sparse sorted union: capacity bounded on device even BEFORE overflow reaches the runtime.
    k::kv_stream_reset(m, cs);
    std::vector<int32_t> oversized(L); for (int i = 0; i < L; ++i) oversized[i] = i * 4;
    select(oversized); resolve(ha, view_a); check_map(); require(k::kv_stream_counters(m).overflow, "oversized union reports overflow");
    // Invalid positive/negative mappings, invalid selection ID and width > cap must never write OOB.
    k::kv_stream_reset(m, cs); la[0] = B; la[1] = -1; put(logical_a, la);
    select({-1, 0, 4, L * 4}); resolve(ha, view_a); check_map();
    require(k::kv_stream_counters(m).overflow, "invalid mappings report overflow");
    require(get(view_a, L)[0] == -1 && get(view_a, L)[1] == -1, "invalid mappings do not materialize slots");
    const auto invalid_before = host.snapshot(B), invalid_gpu = slots.snapshot(S);
    append(slots, view_a, step, key, val, s, fmt, &ha, cs); sync();
    require(host.snapshot(B) == invalid_before && slots.snapshot(S) == invalid_gpu, "invalid mapped append fails closed");
    const auto invalid_stage = stage.page(0);
    k::kv_stage_from_host(stage.attn(identity), ha, fmt, 1, s, cs);
    k::kv_unstage_to_host(stage.attn(identity), ha, fmt, 0, 1, s, cs); sync();
    require(stage.page(0) == invalid_stage && host.snapshot(B) == invalid_before, "invalid stage mapping fails closed");
    // Oversized positive GPU slot is rejected by writers (backing itself remains a valid host row).
    la[0] = 0; put(logical_a, la);
    ck(cudaMemcpy(m.page_table, &S, sizeof(S), cudaMemcpyHostToDevice), "invalid slot");
    append(slots, view_a, step, key, val, s, fmt, &ha, cs); sync();
    require(slots.snapshot(S) == invalid_gpu, "invalid resident slot fails closed");
    k::kv_stream_reset(m, cs);
    std::vector<int32_t> pad(CAP, -1); put(ids, pad); put(steps, std::vector<int32_t>{0, 0, 0, CAP + 1});
    resolve(ha, view_a); check_map();

    // Null-mapping identity stage/unstage, aliased legacy resolve, identity-range invalidation and MTP ring.
    k::kv_stream_reset(m, cs); select({0}); resolve(host.p, m.page_table); check_map();
    const int legacy_slot = get(m.page_table, B)[0]; require(slots.page(legacy_slot) == host.page(0), "legacy alias resolve");
    k::kv_stream_invalidate(m, nullptr, 0, 1, cs); sync(); require(get(m.page_table, B)[0] == -1, "identity range invalidation");
    k::kv_stage_from_host(stage.attn(identity), host.p, fmt, 2, s, cs); sync();
    require(stage.page(0) == host.page(0) && stage.page(1) == host.page(1), "legacy identity stage");
    stage.copy_page(1, ref_b, 0); k::kv_unstage_to_host(stage.attn(identity), host.p, fmt, 1, 2, s, cs); sync();
    require(host.page(1) == ref_b.page(0), "legacy identity unstage");
    auto* ring_table = mem.alloc<int32_t>(B); k::kv_ring_table(ring_table, B, S, cs);
    k::kv_ring_restore(slots.attn(ring_table), host.p, fmt, B - 2, B, S, s, cs); sync();
    require(slots.page((B - 1) % S) == host.page(B - 1), "legacy ring restore");
    put(step, std::vector<int32_t>{(B - 1) * 4, (B - 1) * 4 + 1, B - 1, 1});
    append(slots, ring_table, step, key, val, s, fmt, &host.p, cs); sync();
    require(slots.page((B - 1) % S) == host.page(B - 1), "legacy ring writer identity backing");
    ck(cudaStreamDestroy(cs), "destroy stream");
    std::printf("  %s: PASS (shared writers/readers, CLOCK, COW/recycle, DMA mapping, graph, overflow, legacy)\n",
                fmt == k::kKvF16 ? "fp16" : fmt == k::kKvInt8 ? "int8" : "q4_0");
}
} // namespace
int main() {
    int devices = 0;
    const cudaError_t available = cudaGetDeviceCount(&devices);
    if (available != cudaSuccess || devices <= 0) {
        std::fprintf(stderr, "SKIP: shared_kv_stream_parity: %s\n",
                     available != cudaSuccess ? cudaGetErrorString(available) : "no CUDA devices");
        return 77;
    }
    std::puts("shared_kv_stream_parity: synthetic shared KV (1024 resident slots, 1536 logical pages/sequence)");
    run(k::kKvF16); run(k::kKvInt8); run(k::kKvQ4); std::puts("PASS"); return 0;
}
