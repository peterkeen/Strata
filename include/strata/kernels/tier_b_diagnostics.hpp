// include/strata/kernels/tier_b_diagnostics.hpp - opt-in bounded kernel-selection diagnostics (Tier B).
//
// Pinned sta-osh.4: identify the *selected* first-wave layout and its source/capture shape without
// changing normal operation.  This is a dispatch/capture RECORD, not proof of execution: completion
// and actual-work counts are corroborated only by the paired native gates and bounded CLI
// correctness evidence, never by this header.
//
// Opt-in via STRATA_TIER_B_TRACE.  Enabled IFF the variable is non-empty and not the exact string
// "0"; any other value enables it.  OFF by default, and then the only cost is one cached env lookup
// plus a branch: no allocation, no readback, no synchronization, no printf, and no capture-state
// query (enabled() is checked before any CUDA call or emit; callers also guard their
// diagnostic-only selector computations with enabled()).
//
// When ON the recorder is bounded and host-only:
//   * one record per dispatch decision (a call site), NOT per launch and NOT per completion;
//   * capacity fields (cap_groups/cap_entries/grid_groups) are NOMINAL maxima from the layout and
//     the launch grid, never "actual executed groups" (that needs a device readback and is out of
//     scope here; -1 means "not measured");
//   * zero or unsupported work is labelled with path=zero / path=unsupported and emits NO positive
//     eligibility record for the same call;
//   * capture state is read once with cudaStreamIsCapturing and never switches mode inside a graph;
//     nullptr stream, Invalidated, and any CUDA error are all recorded as capture=-1 (fail closed:
//     an unmeasured/unknown state is never reported as "direct");
//   * the diagnostic never consumes the underlying CUDA error (no cudaGetLastError), so a real
//     failure is left intact for the caller's own check;
//   * records are bounded by kMaxRecords; the first overflow emits a single trunc=1 marker and all
//     further emits are dropped (explicit truncation, no unbounded logging).
//
// RECORDER ERRORS vs ORDINARY TRUNCATION.  Three different conditions are represented distinctly,
// and the consuming validator (tools/tier_b/run_pairs.py) treats them differently on purpose:
//
//   * path=truncated (trunc=1) - ORDINARY record-budget truncation: the kMaxRecords cap was hit.
//     Everything emitted before the cap is still trustworthy, so this only RESTRICTS COVERAGE; the
//     validator keeps the prefix as dispatch evidence and merely marks the verdict incomplete.
//   * path=format_error (trunc=1) - a RECORDER ERROR: snprintf failed or the record did not fit
//     buf, so a dispatch decision could NOT be written at all and the record stream has an
//     unaccounted-for hole of unknown content.  This is NOT ordinary truncation, and the validator
//     REJECTS the whole trace as evidence rather than downgrading it to a coverage limit.
//   * capture=-1 on a normal record - a CAPTURE-QUERY ERROR or otherwise unknown capture state: a
//     nullptr stream, cudaStreamCaptureStatusInvalidated, or a failed cudaStreamIsCapturing (see
//     capture_state below).  Such a record cannot establish WHERE it was dispatched, so the
//     validator REJECTS it as evidence; only capture=0 (direct) or capture=1 (capturing) is
//     admissible.  The budget marker itself is exempt: it reports capture=-1 because it describes
//     the recorder, not a dispatch.
//
// The validator additionally requires every geometry/capacity field relevant to a record's family
// to be present and STRICTLY POSITIVE before counting it as evidence: a record that claims a
// studied dispatch while reporting a zero or -1 ("not measured") geometry, or zero/-1 nominal
// capacities for the grouped-expert family, proves nothing about that dispatch and is rejected.
//
// Machine-readable schema, one record per line (tokens are `key=value`, values have no whitespace):
//   TIER_B_TRACE v=1 family=<id> caller=<id> path=<id> gu=<i> d=<i> n_embd=<i> n_ff=<i> rows=<i>
//                 head_dim=<i> n_rot=<i> ncols=<i> cap_groups=<i> cap_entries=<i>
//                 grid_groups=<i> table=<i> capture=<i> trunc=<0|1>
// All integer fields are decimal; unused/unknown are -1.  family/caller/path are identifiers
// (letters, digits, '_').  capture: 0=direct submission, 1=stream capture active, -1=unknown
// (nullptr stream / Invalidated capture / capture-query error) and therefore inadmissible as
// evidence on a normal record.
// Field meaning by family: rope/rope_norm use rows/head_dim/n_rot (+table); iq_mmvq uses
// n_embd=in_dim, rows=out_rows, ncols; iq_grouped uses gu/d/n_embd/n_ff and the capacity fields.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace tier_b {

// Hard record budget.  The counter is atomic so concurrent host threads (split streams) stay
// bounded; no heap is touched.
inline constexpr int kMaxRecords = 1024;

inline std::atomic<int> g_count{0};
inline std::atomic<int> g_truncated{0};

// Cheap read: the env var is resolved exactly once (static init), then this is an atomic load.
// Enabled iff non-empty AND not the exact string "0".
inline bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("STRATA_TIER_B_TRACE");
        return e != nullptr && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
    }();
    return on;
}

// Capture state of `stream`.  Returns 0/1/-1; never called by the recorder when the trace is off.
// A nullptr stream, a CUDA error, or an Invalidated/unknown status is -1 (unmeasured/unknown) - an
// unknown is never silently reported as direct submission.  The CUDA error is NOT cleared here
// (no cudaGetLastError), so the caller's own subsequent check still sees it.
//
// -1 is an explicit CAPTURE-QUERY ERROR / unknown-state signal, not a benign default: the Tier B
// validator refuses a normal record carrying capture=-1 as dispatch evidence (see the header
// comment above), so a failed query degrades the trace's admissibility instead of silently
// producing a record that looks positive.
inline int capture_state(void* stream) {
    if (stream == nullptr) return -1;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    const cudaError_t e = cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &st);
    if (e != cudaSuccess) {
#ifndef NDEBUG
        std::fprintf(stderr, "TIER_B_TRACE capture query error: %s\n", cudaGetErrorString(e));
#endif
        return -1;
    }
    switch (st) {
        case cudaStreamCaptureStatusNone:   return 0;
        case cudaStreamCaptureStatusActive: return 1;
        default:                            return -1;  // Invalidated / unknown
    }
}

// Emit one COMPLETE bounded marker line (never a partial record).  Used for budget overflow
// and for a format failure, so a truncated/partial line can never be misread as a positive
// dispatch record.  Always ends in a newline so later records cannot merge into it.
inline void emit_marker(const char* path) {
    std::fprintf(stderr,
        "TIER_B_TRACE v=1 family=recorder caller=emit path=%s gu=-1 d=-1 n_embd=-1 n_ff=-1 rows=-1 "
        "head_dim=-1 n_rot=-1 ncols=-1 cap_groups=-1 cap_entries=-1 grid_groups=-1 table=-1 capture=-1 trunc=1\n",
        path);
}

// Emit one record if enabled and under budget.  Thread-safe and bounded; never allocates and never
// synchronizes.  `capture` is resolved inside (after the enabled() gate) so a disabled trace never
// even queries the stream.  A failed/oversized format emits a complete error marker, never a partial
// record.
inline void emit(const char* family, const char* caller, const char* path,
                 int gu_type, int d_type,
                 int64_t n_embd, int64_t n_ff, int64_t rows, int64_t head_dim, int64_t n_rot, int64_t ncols,
                 int64_t cap_groups, int64_t cap_entries, int64_t grid_groups,
                 int table, void* stream) {
    if (!enabled()) return;
    const int n = g_count.fetch_add(1, std::memory_order_relaxed);
    if (n >= kMaxRecords) {
        // Explicit truncation: exactly one marker, then drop everything.
        int expected = 0;
        if (g_truncated.compare_exchange_strong(expected, 1, std::memory_order_relaxed)) {
            emit_marker("truncated");
        }
        return;
    }
    const int capture = capture_state(stream);
    char buf[640];
    const int w = std::snprintf(
        buf, sizeof(buf),
        "TIER_B_TRACE v=1 family=%s caller=%s path=%s gu=%d d=%d n_embd=%lld n_ff=%lld rows=%lld "
        "head_dim=%lld n_rot=%lld ncols=%lld cap_groups=%lld cap_entries=%lld grid_groups=%lld table=%d capture=%d trunc=0\n",
        family, caller, path, gu_type, d_type,
        (long long) n_embd, (long long) n_ff, (long long) rows, (long long) head_dim, (long long) n_rot, (long long) ncols,
        (long long) cap_groups, (long long) cap_entries, (long long) grid_groups, table, capture);
    if (w < 0 || static_cast<size_t>(w) >= sizeof(buf)) {
        // Format/overflow failure: complete explicit marker, never a partial positive record.
        emit_marker("format_error");
        return;
    }
    std::fwrite(buf, 1, static_cast<size_t>(w), stderr);
}

}  // namespace tier_b
}  // namespace strata::kernels
