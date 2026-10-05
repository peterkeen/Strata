// include/strata/kernels/kv_stream.hpp - KV streaming for the QSA layers (docs/kv-streaming-design.md).
//
// A streamed QSA layer keeps its AUTHORITATIVE K/V in pinned, device-mapped host memory, laid out exactly as a
// fully resident pool would be. Shared sequences map logical pages onto authoritative backing pages; null
// mapping pointers retain the legacy identity layout. VRAM holds a pool of `n_slots` pages (one page =
// `page_size` = 4 cells = one indexer block, all KV heads, K and V: 4,224 B in int8), and the page table becomes a
// RESIDENCY MAP: `page_table[block]` is the slot holding that block, or -1.
//
// Every reader (`qsa_decode_attn_*`, `kv_gather_*`) is unchanged: it goes through the page table as before. What
// changes is that after the selection and before the attention, `kv_stream_resolve` makes every block the
// selection names resident - on device, inside the captured graphs:
//   1. one block of 1,024 threads walks the selected ids: a resident block gets this call's epoch and its clock
//      reference bit, a missing block is claimed once (page_table -1 -> -2) into a miss list;
//   2. the same block picks one victim slot per miss with a CLOCK sweep (second chance) that never takes a slot
//      this call uses, re-points the page table (old block -> -1, new block -> slot);
//   3. a copy kernel reads the missed blocks from the host copy (zero-copy, over PCIe) into their slots.
// Writers (`kv_append_*`) write the host copy always, and the slot only if the block is resident, so the slots
// never go stale under ordinary appends. Shared page publication/recycling/COW and snapshot restores must
// invalidate the affected backing pages after synchronization, outside captured graphs.
//
// A RING (the MTP drafter, which only ever reads its last `window` cells) uses the same host copy with a static
// table `block -> block % n_slots` and no resolve: the ring is refilled from the host copy on a resume.
#pragma once

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// The host copy of one layer's K/V: device-mapped pointers (UVA) into pinned memory, backing-page layout
/// `[backing_page][kv_head][page_size][head_dim]`. Payload may be null with mapping metadata still present
/// (e.g. HIP DMA-only host writes). Staging pools always use logical identity rows.
struct KvHostPools {
    uint16_t* k_pool = nullptr;   ///< fp16 mode
    uint16_t* v_pool = nullptr;
    int8_t* k_q = nullptr;        ///< int8 mode: codes
    int8_t* v_q = nullptr;
    uint16_t* k_scale = nullptr;  ///< int8 mode: fp16 scale per 64 values
    uint16_t* v_scale = nullptr;
    uint8_t* k_q4 = nullptr;      ///< q4_0 mode (kv_q4.hpp): block_q4_0 codes, 144 B per cell and head
    uint8_t* v_q4 = nullptr;
    const int32_t* logical_pages = nullptr;   ///< device array: sequence logical page -> shared backing page
    const int32_t* resident_pages = nullptr;  ///< device array: shared backing page -> GPU slot, or -1
    // Required capacities when the corresponding mapping pointer is nonnull. Zero fails closed; legacy
    // callers with both pointers null need not set these. Keep addresses AND capacities fixed during capture.
    int64_t n_logical_pages = 0;
    int64_t n_backing_pages = 0;
    int64_t n_resident_slots = 0;
    bool present() const { return k_pool != nullptr || k_q != nullptr || k_q4 != nullptr; }
};

#if defined(__CUDACC__) || defined(__HIPCC__)
// Shared by all append paths. Mapping is independent of payload presence: metadata-only writers must still
// consult the global residency table. A mapped writer NEVER falls back to the stale sequence reader view.
__device__ inline int64_t kv_host_backing_page(const KvHostPools& host, int64_t logical) {
    if (logical < 0) return -1;
    if (host.logical_pages != nullptr) {
        if (logical >= host.n_logical_pages) return -1;
        const int64_t backing = host.logical_pages[logical];
        return backing >= 0 && backing < host.n_backing_pages ? backing : -1;
    }
    if (host.resident_pages != nullptr && logical >= host.n_backing_pages) return -1;
    return logical;
}
__device__ inline int64_t kv_host_gpu_page(const KvHostPools& host, const int32_t* table,
                                          int64_t logical, int64_t backing) {
    if (backing < 0) return -1;
    if (host.resident_pages != nullptr) {
        if (backing >= host.n_backing_pages) return -1;
        const int64_t slot = host.resident_pages[backing];
        return slot >= 0 && slot < host.n_resident_slots ? slot : -1;
    }
    if (host.logical_pages != nullptr) return -1;
    return table != nullptr ? table[logical] : -1;
}
#endif

/// The KV storage format, for the functions below that move whole blocks (`fmt`): fp16, int8 (+ scales), q4_0.
/// (A bool `int8` argument still reads as kKvF16 / kKvInt8.)
enum KvFormat : int { kKvF16 = 0, kKvInt8 = 1, kKvQ4 = 2 };

/// The residency map of a streamed layer, all device memory at fixed addresses (the graphs bake them in).
struct KvStreamMap {
    int32_t* page_table = nullptr;  ///< (n_blocks,) shared backing page -> slot, or -1
    int32_t* slot_block = nullptr;  ///< (n_slots,) slot -> block, or -1
    int32_t* slot_stamp = nullptr;  ///< (n_slots,) the resolve epoch that last used the slot
    int32_t* slot_ref = nullptr;    ///< (n_slots,) clock reference bit
    int32_t* ctl = nullptr;         ///< kKvCtlInts: epoch, hand, misses of the last call, overflow, u64 counters
    int32_t* miss_block = nullptr;  ///< (n_slots,)
    int32_t* miss_slot = nullptr;   ///< (n_slots,)
    int64_t n_blocks = 0;
    int64_t n_slots = 0;
};

inline constexpr int kKvCtlInts = 16;   ///< [0] epoch [1] hand [2] misses [3] overflow; u64 at [4] misses, [6] lookups, [8] calls
/// Bytes of the map's arrays besides the page table (which the state already has): 5 per-slot ints + ctl.
inline uint64_t kv_stream_map_bytes(int64_t n_slots) { return (uint64_t) n_slots * 4 * 5 + kKvCtlInts * 4; }

/// Every block evicted: page table all -1, slots free, counters zero.
void kv_stream_reset(const KvStreamMap& m, void* stream);

/// After synchronization, outside graphs: evict backing pages named by logical_pages[begin_page:end_page].
/// Null logical_pages means an identity backing range. Caller bounds the range to the DEVICE array's
/// allocation; invalid backing IDs are ignored. Clears table/slot metadata, not counters or CLOCK hand.
/// Publish new/recycled/COW mappings first, then invalidate; snapshot restore also invalidates its pages.
void kv_stream_invalidate(const KvStreamMap& m, const int32_t* logical_pages, int64_t begin_page,
                          int64_t end_page, void* stream);

/// Same eviction, but backing_pages is a DEVICE list of count backing IDs (duplicates/invalid IDs safe).
/// Caller bounds count to the list allocation; count <= 0 is a no-op. Invalidate last-reference releases
/// BEFORE recycling IDs becomes visible.
void kv_stream_invalidate_pages(const KvStreamMap& m, const int32_t* backing_pages, int64_t count, void* stream);

/// Make every block named by the selections of `n_q` queries resident (ids [n_q][cap], width from
/// steps[q * kStepCount + kStepWidth]). Capturable. Each selection is sorted ascending (adjacent cells in
/// a logical block dedup); `cap` bounds every row. `n_slots` must hold the distinct BACKING blocks of the
/// whole query union (usual QSA block-shaped bound: n_q x (cap / page_size + 2), not a bound for arbitrary
/// sparse selections). A call that cannot sets ctl[3] (see `kv_stream_counters`). Invalid IDs and
/// scratch overflow also set ctl[3] without out-of-bounds writes. Do not attend an overflowed call. After the
/// copies, materializes ONLY selected logical pages in slots.page_table (a writable device allocation despite
/// QsaAttnPools' reader-facing const type). Never writes a separate view when slots.page_table == m.page_table.
void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream);

/// The static ring table `block -> block % n_slots`.
void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream);

/// Copy blocks [b0, b1) of the host copy into the pool pages `page_table` names (host-side table `phys(b)`
/// computed as `b % n_slots`: the ring restore). Not capturable.
void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream);

/// Copy the first `n_blocks` logical blocks from mapped backing rows into a logical identity staging pool.
/// Legacy identity uses DMA; mapped pages use a coalesced GPU copy without host-reading device metadata.
/// Only stages the requested range, never the whole logical pool for decode.
void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream);

/// #579 #613 (HIP A/B, STRATA_KV_HOST_DMA=1): the reverse of kv_stage_from_host for blocks [b0, b1) - the staging
/// pool's logical identity blocks into mapped backing rows. Identity uses DMA; mapped pages use a GPU copy.
/// Neither direction reads mapping device pointers on the host.
void kv_unstage_to_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                        const QsaShapes& s, void* stream);

struct KvStreamCounters {
    uint64_t misses = 0, lookups = 0, calls = 0;
    bool overflow = false;
};
/// Synchronous read of the counters (debug and the end-of-request summary).
KvStreamCounters kv_stream_counters(const KvStreamMap& m);

/// Bytes of one block (page) of K and V together.
uint64_t kv_block_bytes(const QsaShapes& s, int fmt);

}  // namespace strata::kernels
