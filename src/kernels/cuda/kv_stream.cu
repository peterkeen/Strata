// src/kernels/cuda/kv_stream.cu - see include/strata/kernels/kv_stream.hpp.
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_stream: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int RT = 1024;   // the resolve block

// The per-block byte runs of the (up to four) pool arrays: block b of array i is bytes [b * len, (b + 1) * len).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvHybrid) {   // K8V4: int8 K codes, their scales, rotated q4_0 V
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        const int v = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.k_scale; r.dst[1] = (uint8_t*) slots.k_scale; r.len[1] = scales;
        r.src[2] = (const uint8_t*) host.v_q4;    r.dst[2] = (uint8_t*) slots.v_q4;    r.len[2] = v;
        r.n = 3;
    } else if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4; r.dst[0] = (uint8_t*) slots.k_q4; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.v_q;     r.dst[1] = (uint8_t*) slots.v_q;     r.len[1] = codes;
        r.src[2] = (const uint8_t*) host.k_scale; r.dst[2] = (uint8_t*) slots.k_scale; r.len[2] = scales;
        r.src[3] = (const uint8_t*) host.v_scale; r.dst[3] = (uint8_t*) slots.v_scale; r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool; r.dst[0] = (uint8_t*) slots.k_pool; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool; r.dst[1] = (uint8_t*) slots.v_pool; r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
__device__ int block_scan(int v, int* warp_sums, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    __syncthreads();
    return excl;
}

// Bounded reservation: even an oversized union can never index beyond the per-slot miss scratch.
__device__ int reserve_miss(int* count, int capacity) {
    int old = atomicAdd(count, 0);
    while (old < capacity) {
        const int seen = atomicCAS(count, old, old + 1);
        if (seen == old) return old;
        old = seen;
    }
    return -1;
}

__device__ long long selected_backing(KvStreamMap m, KvHostPools host, int cell, int page_size) {
    if (cell < 0) return -1;
    const long long logical = cell / page_size;
    if (host.logical_pages == nullptr && logical >= m.n_blocks) return -1;
    const long long backing = kv_host_backing_page(host, logical);
    return backing >= 0 && backing < m.n_blocks ? backing : -1;
}

__global__ void __launch_bounds__(RT) resolve_kernel(KvStreamMap m, KvHostPools host,
                                                     const int32_t* __restrict__ ids,
                                                     const int32_t* __restrict__ steps, int n_q, int cap, int page_size) {
    __shared__ int s_nmiss, s_lookups, s_cut, s_bad;
    __shared__ int warp_sums[32];
    const int epoch = m.ctl[0] == INT_MAX ? 1 : m.ctl[0] + 1;
    if (threadIdx.x == 0) { s_nmiss = 0; s_lookups = 0; s_bad = 0; }
    if (m.ctl[0] == INT_MAX)
        for (long long j = threadIdx.x; j < m.n_slots; j += RT) m.slot_stamp[j] = -1;
    __syncthreads();
    // 1. Protect hits across the ENTIRE query union before choosing any victim. Selections are sorted
    // ascending; adjacent cells in a logical block need only one lookup. Backing aliases dedup via CAS.
    int lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int requested = steps[(long long) q * kStepCount + kStepWidth];
        const int width = requested < 0 ? 0 : (requested > cap ? cap : requested);
        if (requested != width) atomicOr(&s_bad, 1);
        const int32_t* qi = ids + (long long) q * cap;
        for (long long i = threadIdx.x; i < width; i += RT) {
            const long long b = selected_backing(m, host, qi[i], page_size);
            if (b < 0) { atomicOr(&s_bad, 1); continue; }
            if (i > 0 && qi[i - 1] >= 0 && qi[i - 1] / page_size == qi[i] / page_size) continue;
            ++lookups;
            const int sl = m.page_table[b];
            if (sl >= 0) {
                if (sl >= m.n_slots || m.slot_block[sl] != b) { atomicOr(&s_bad, 1); continue; }
                m.slot_stamp[sl] = epoch;
                m.slot_ref[sl] = 1;
            } else if (sl == -1 && atomicCAS(&m.page_table[b], -1, -2) == -1) {
                const int k = reserve_miss(&s_nmiss, (int) m.n_slots);
                if (k >= 0) m.miss_block[k] = (int) b;
                else { atomicExch(&m.page_table[b], -1); atomicOr(&s_bad, 1); }
            } else if (sl < -2) atomicOr(&s_bad, 1);
        }
    }
    if (lookups > 0) atomicAdd(&s_lookups, lookups);   // (#783, stuchapin909) most threads see none: skip the shared atomic
    __syncthreads();
    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is never taken;
    //    a referenced one loses its bit as the hand passes it and is taken on the next pass.
    const int need = s_nmiss, n = (int) m.n_slots;
    int hand = m.ctl[1] >= 0 && m.ctl[1] < n ? m.ctl[1] : 0, got = 0;
    for (long long scanned = 0; got < need && scanned < 3LL * n; scanned += RT) {
        const int j = (int) (((long long) hand + threadIdx.x) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (threadIdx.x == 0) s_cut = RT;
        __syncthreads();
        if (cand && rank == want - 1) s_cut = threadIdx.x + 1;   // the hand stops just past the last slot taken
        __syncthreads();
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
        } else if (threadIdx.x < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int) (((long long) hand + cut) % n);
        __syncthreads();
    }
    // 3. re-point the table; the copy kernel fills the slots
    const int placed = got < need ? got : need;
    for (long long k = threadIdx.x; k < need; k += RT) {
        const int b = m.miss_block[k];
        if (k >= placed) { m.page_table[b] = -1; continue; }   // overflow: never happens with a legal n_slots
        const int sl = m.miss_slot[k];
        const int old = m.slot_block[sl];
        if (old >= 0 && old < m.n_blocks) m.page_table[old] = -1;
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
    }
    if (threadIdx.x == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need || s_bad) m.ctl[3] = 1;
        unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;
    }
}

// One block per missed block (grid-stride): copy its runs from the host copy into its slot, 16 B per thread.
__global__ void copy_kernel(KvStreamMap m, Runs r) {
    const int need = m.ctl[2];
    for (long long k = blockIdx.x; k < need; k += gridDim.x) {
        const long long b = m.miss_block[k], sl = m.miss_slot[k];
        for (int a = 0; a < r.n; ++a) {
            if (r.src[a] == nullptr || r.dst[a] == nullptr) continue;
            const uint4* src = reinterpret_cast<const uint4*>(r.src[a] + b * r.len[a]);
            uint4* dst = reinterpret_cast<uint4*>(r.dst[a] + sl * r.len[a]);
            for (int i = threadIdx.x; i < r.len[a] / 16; i += blockDim.x) dst[i] = src[i];
        }
    }
}

// Materialize only selected logical pages, immediately after resolving/copying. Nonselected entries may be
// stale and must never be used as a writer residency map. Legacy/global-table aliasing skips this kernel.
__global__ void materialize_kernel(KvStreamMap m, KvHostPools host, int32_t* view, const int32_t* ids,
                                   const int32_t* steps, int n_q, int cap, int page_size) {
    for (long long q = blockIdx.x; q < n_q; q += gridDim.x) {
        const int requested = steps[(long long) q * kStepCount + kStepWidth];
        const int width = requested < 0 ? 0 : (requested > cap ? cap : requested);
        for (long long i = threadIdx.x; i < width; i += blockDim.x) {
            const int cell = ids[(long long) q * cap + i];
            if (cell < 0) continue;
            const long long logical = cell / page_size;
            const long long limit = host.logical_pages ? host.n_logical_pages : m.n_blocks;
            if (logical >= limit) continue;
            const long long backing = selected_backing(m, host, cell, page_size);
            const int sl = backing >= 0 ? m.page_table[backing] : -1;
            // Multiple selected cells/queries can name the same page: atomic store avoids a write/write race.
            atomicExch(view + logical, sl >= 0 && sl < m.n_slots ? sl : -1);
        }
    }
}

__device__ void invalidate_backing(KvStreamMap m, long long backing) {
    if (backing < 0 || backing >= m.n_blocks) return;
    const int sl = atomicExch(m.page_table + backing, -1);
    if (sl >= 0 && sl < m.n_slots && m.slot_block[sl] == backing) {
        m.slot_block[sl] = -1;
        m.slot_stamp[sl] = -1;
        m.slot_ref[sl] = 0;
    }
}
__global__ void invalidate_kernel(KvStreamMap m, const int32_t* pages, long long begin, long long end) {
    for (long long i = begin + (long long) blockIdx.x * blockDim.x + threadIdx.x; i < end;
         i += (long long) gridDim.x * blockDim.x)
        invalidate_backing(m, pages ? pages[i] : i);
}

// Device metadata is never dereferenced on the host. One CTA per logical page copies coalesced byte runs;
// this supports arbitrary backing mappings (including a partial first/last prompt page).
__global__ void mapped_stage_kernel(Runs r, KvHostPools host, long long begin, long long end, bool unstage) {
    for (long long logical = begin + blockIdx.x; logical < end; logical += gridDim.x) {
        const long long backing = kv_host_backing_page(host, logical);
        if (backing < 0) continue;
        for (int a = 0; a < r.n; ++a) {
            if (r.src[a] == nullptr || r.dst[a] == nullptr) continue;
            const uint8_t* src = unstage ? r.dst[a] + logical * r.len[a] : r.src[a] + backing * r.len[a];
            uint8_t* dst = unstage ? const_cast<uint8_t*>(r.src[a]) + backing * r.len[a]
                                   : r.dst[a] + logical * r.len[a];
            for (int i = threadIdx.x; i < r.len[a]; i += blockDim.x) dst[i] = src[i];
        }
    }
}

__global__ void reset_kernel(KvStreamMap m) {
    const long long i0 = (long long) blockIdx.x * blockDim.x + threadIdx.x, st = (long long) gridDim.x * blockDim.x;
    for (long long i = i0; i < m.n_blocks; i += st) m.page_table[i] = -1;
    for (long long i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) m.ctl[i0] = 0;
}

__global__ void ring_kernel(int32_t* table, long long n_blocks, long long n_slots) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n_blocks;
         i += (long long) gridDim.x * blockDim.x)
        table[i] = (int32_t) (i % n_slots);
}

}  // namespace

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvHybrid)
        return rows * (uint64_t) s.head_dim + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 +
               rows * kv_q4_bytes_per_head((int) s.head_dim);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    reset_kernel<<<128, 256, 0, (cudaStream_t) stream>>>(m);
    check("reset");
}

void kv_stream_invalidate(const KvStreamMap& m, const int32_t* logical_pages, int64_t begin_page,
                          int64_t end_page, void* stream) {
    begin_page = std::max<int64_t>(0, begin_page);
    if (logical_pages == nullptr) end_page = std::min(end_page, m.n_blocks);
    if (end_page <= begin_page) return;
    invalidate_kernel<<<128, 256, 0, (cudaStream_t) stream>>>(m, logical_pages, begin_page, end_page);
    check("invalidate range");
}

void kv_stream_invalidate_pages(const KvStreamMap& m, const int32_t* backing_pages, int64_t count, void* stream) {
    if (count <= 0 || backing_pages == nullptr) return;
    invalidate_kernel<<<128, 256, 0, (cudaStream_t) stream>>>(m, backing_pages, 0, count);
    check("invalidate pages");
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    // Bound integer indices/counters before narrowing. The miss list itself is bounded on DEVICE, not by
    // assuming cap/page_size distinct pages (sparse sorted selections can name more than that).
    if (cap <= 0 || cap > INT_MAX || n_q > INT_MAX / cap || s.page_size <= 0 || s.page_size > INT_MAX ||
        m.n_blocks <= 0 || m.n_blocks > INT_MAX || m.n_slots > INT_MAX) {
        std::fprintf(stderr, "kv_stream: invalid resolve geometry/capacity\n");
        std::exit(1);
    }
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    // one sweep step looks at RT consecutive slots `(hand + thread) % n_slots`; with fewer slots than RT two
    // threads see the same slot and may both take it for two different misses.  The engine never streams with fewer
    // than qsa_kv_resident_min() / page_size = 5,120 slots, so this is a guard, not a limit.
    if (m.n_slots < RT) {
        std::fprintf(stderr, "kv_stream: %lld slots is fewer than the resolve block (%d): the clock sweep would take a "
                             "slot twice\n", (long long) m.n_slots, RT);
        std::exit(1);
    }
    resolve_kernel<<<1, RT, 0, (cudaStream_t) stream>>>(m, host, ids, steps, (int) n_q, (int) cap, (int) s.page_size);
    check("resolve");
    copy_kernel<<<96, 128, 0, (cudaStream_t) stream>>>(m, runs_of(slots, host, fmt, s));
    check("copy");
    if (slots.page_table != nullptr && slots.page_table != m.page_table) {
        materialize_kernel<<<96, 128, 0, (cudaStream_t) stream>>>(
            m, host, const_cast<int32_t*>(slots.page_table), ids, steps, (int) n_q, (int) cap, (int) s.page_size);
        check("materialize reader view");
    }
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    ring_kernel<<<64, 256, 0, (cudaStream_t) stream>>>(page_table, n_blocks, n_slots);
    check("ring table");
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);   // up to the ring's end
        for (int a = 0; a < r.n; ++a)
            if (cudaMemcpyAsync(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t) (run * r.len[a]),
                                cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
                check("ring restore");
        b += run;
    }
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    if (host.logical_pages != nullptr || host.resident_pages != nullptr) {
        const int64_t end = std::min(n_blocks, host.logical_pages ? host.n_logical_pages : host.n_backing_pages);
        if (end > 0) mapped_stage_kernel<<<96, 128, 0, (cudaStream_t) stream>>>(r, host, 0, end, false);
        check("mapped stage");
        return;
    }
    for (int a = 0; a < r.n; ++a)
        if (r.src[a] != nullptr && r.dst[a] != nullptr &&
            cudaMemcpyAsync(r.dst[a], r.src[a], (size_t) (n_blocks * r.len[a]), cudaMemcpyDefault,
                            (cudaStream_t) stream) != cudaSuccess)
            check("stage");
}

void kv_unstage_to_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                        const QsaShapes& s, void* stream) {
    if (b1 <= b0) return;
    b0 = std::max<int64_t>(0, b0);
    const Runs r = runs_of(stage, host, fmt, s);   // src: backing host copy, dst: logical identity staging
    if (host.logical_pages != nullptr || host.resident_pages != nullptr) {
        const int64_t end = std::min(b1, host.logical_pages ? host.n_logical_pages : host.n_backing_pages);
        if (end > b0) mapped_stage_kernel<<<96, 128, 0, (cudaStream_t) stream>>>(r, host, b0, end, true);
        check("mapped unstage");
        return;
    }
    if (b1 <= b0) return;
    for (int a = 0; a < r.n; ++a)
        if (r.src[a] != nullptr && r.dst[a] != nullptr &&
            cudaMemcpyAsync((void*) (r.src[a] + b0 * r.len[a]), r.dst[a] + b0 * r.len[a],
                            (size_t) ((b1 - b0) * r.len[a]), cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
            check("unstage");
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr || cudaMemcpy(c, m.ctl, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) return r;
    // A host int32_t array need not be u64-aligned; do not type-pun its counter storage.
    uint64_t u[3];
    std::memcpy(u, c + 4, sizeof(u));
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

}  // namespace strata::kernels
