# Incremental and pooled QSA indexer storage

Status: **design and implementation plan; not implemented or deployed**.

This plan targets `kv-unified` rebased onto upstream main `a0116b27`
(Strata 0.1.43.1). The rebased historical branch tip is `7897a98b`;
pre-rebase tip was `f3289a81`. Historical GPU results in
[MULTI_SLOT_UNIFIED_KV.md](MULTI_SLOT_UNIFIED_KV.md) and
[INCREMENTAL_UNIFIED_KV.md](INCREMENTAL_UNIFIED_KV.md) do not validate the
rebased engine or either proposed indexer mode.

## Outcome

Recover expert-cache VRAM now occupied by full-context, per-session pooled
indexer keys, without changing FP32 indexer arithmetic, selected logical cell
IDs, model context limits, sampling, or cancellation/pressure semantics.

Deliver two independently gated, opt-in changes:

1. **Incremental private indexer:** stable-address contiguous storage for each
   session, physically committed only through the required extent. Establish
   safe growth, shrink, checkpoint and graph contracts before sharing keys.
2. **Pooled completed keys:** one shared physical indexer pool associated with
   the unified KV backing-page allocator; completed keys share ownership with
   KV pages. Sequence-specific partial-block/spare state stays private.

The first change reduces overprovisioning for short histories. The second
removes duplicate keys for cloned prefixes and bounds completed-key storage by
aggregate physical KV ownership rather than the number of sessions.

Non-goals: quantizing FP32 keys, sharing independent GDN recurrence, changing
MTP ring geometry, multi-GPU unified KV, new vision modes, a new scheduler,
raising context limits, or changing deployment defaults. Do not interrupt the
current nibbler evaluation to implement or validate this work.

## Current source and blast radius

### Allocation

- `src/core/layer.cpp`: `KvPlan::pooled_rows = max_cells / idx_block + 2`;
  `kv_shared_state_bytes`, `kv_unified_owner_bytes`, `qsa_state_bytes`,
  `qsa_state_init`, and `qsa_state_zero` account for, carve and clear
  `idx_pooled`. Unified borrowers still allocate this storage privately.
- `include/strata/core/layer.hpp`: `QsaState::idx_pooled` and
  `idx_pooled_rows`; distinguish logical capacity, allocated rows and valid
  rows rather than overloading one field.
- `src/core/session.cpp`: `session_bytes` and `session_init` mirror the
  allocation sequence. `session_release` must release newly external storage
  exactly once, including partially initialized sessions.
- `include/strata/core/vmm.hpp` and its implementation: stable virtual ranges
  and physical handles already support elastic KV. Its `map_range` can leave
  earlier chunks mapped on failure. A new indexer transaction must explicitly
  track and unwind those changes; do not assume this helper is transactional.

### Writers and readers

Task I1's source-grounded row/access audit lives in
[INDEXER_ACCESS_INVENTORY.md](INDEXER_ACCESS_INVENTORY.md).

- `src/kernels/cuda/qsa.cu`: reference append/index kernels and wrappers.
- `src/kernels/cuda/native_qsa_indexer.cu` and
  `include/strata/kernels/native_qsa_indexer.hpp`: native single-step,
  multi-step and chunk append; validation currently describes the full span.
- `src/core/layer.cpp`: single-token append and scoring.
- `src/core/verify.cpp`: multi-row speculative append, scoring, commit and
  rollback. Audit both ordinary windows and batch/slot windows.
- `src/prefill/prefill.cpp`: native batched append, reference per-token debug
  path, tensor-core/SIMT score selection, and `STRATA_IDX_FP16_CHECK`.
  The debug path currently rounds `idx_pooled_rows` rows, which would touch
  uncommitted memory if capacity were mistaken for committed extent.
- Find score implementations with `rg 'qsa_block_scores|idx_pooled' src include
  sycl` before implementation. Cover every CUDA/HIP/SYCL declaration/caller
  affected by a changed interface; do not assume CUDA files are the only users.

### State transfer and persistence

- `src/program/generate.cpp`: solo/slot copies, prompt reuse, checkpoints,
  writer reservation, pressure park, restore and debug indexer hashing.
- `include/strata/core/shared_kv_runtime.hpp`: ensure, clone, release and
  logical-to-physical publication; host allocator is
  `include/strata/core/shared_kv_pages.hpp`.
- `src/core/conversation_state.cpp`: checkpoint restore writes the spare row
  at `ids.size() / idx_block`; capacity must exist before this copy.
- `src/core/conversation_snapshot.cpp`: pooled keys are currently a contiguous
  canonical payload. Keep the canonical format independent of physical layout.
- Tests: `shared_kv_session_test`, `shared_kv_pages_test`,
  `shared_kv_reservation_test`, conversation snapshot/validation/state tests,
  `tests/core/shared_kv_snapshot_test.cpp`, native indexer and streamed-KV
  parity targets, and existing incremental/handoff/batch-MTP smoke tools.

## Memory model and honest success criteria

Canonical geometry: 12 QSA layers, compression block/page size 4 tokens,
128 FP32 values per completed pooled key.

```
private payload bytes(C) = 12 * (floor(C / 4) + 2) * 128 * 4
completed payload bytes(B) = 12 * ceil(B / 4) * 128 * 4
```

The second formula counts physical backing cells, not summed logical claims.
Extra spare rows, per-sequence tails, maps, metadata, allocator granularity,
checkpoint buffers, draft state and transient staging are additional.

| Capacity/history | Completed-key payload, all 12 layers |
| --- | ---: |
| 8,192 cells | 12 MiB |
| 32,768 cells | 48 MiB |
| 131,072 cells | 192 MiB |
| 262,144 cells | 384 MiB |
| 524,288 cells | 768 MiB |

At 256K with a solo admission session and two batch slots, current private
pooled-key payload is approximately **1,152 MiB**. A fully occupied 256K shared
pool needs approximately **384 MiB**, a theoretical payload saving of
**768 MiB**. This is an estimate, not measured recovered expert-cache capacity.

Incremental mapping rounds to the device's actual allocation granularity. For
example, 12 separate ranges with 2 MiB granularity have a 24 MiB minimum physical
commitment even when the raw payload is 12 MiB. Query granularity; do not hardcode
that value or report raw payload as actual allocation.

**Unused indexer VRAM is not automatically usable by the expert cache.** If the
cache is sized against the worst-case future indexer allocation, startup savings
will remain unused. If the cache consumes all initial savings without a reclaim
path, a later long request can OOM. The cache/indexer budget broker below is a
required deliverable, not an optional performance follow-up.

The earlier ~0.25 GiB extra-slot estimate is a rough private-state floor after
removing replicated keys, not a guaranteed final footprint. GDN recurrence
(~108 MiB per session), scratch, maps and transients remain. Two distinct 256K
conversations still require at least 512K aggregate unified backing; sharing
keys does not increase physical token capacity.

## Required invariants

1. Disabled modes retain legacy allocation, arithmetic and serving behavior.
2. No kernel, async copy, graph replay, zeroing operation, diagnostic or
   snapshot transfer touches unmapped/unpublished rows.
3. Stable captured addresses remain stable for the full graph lifetime.
   Kernel grid/stride capacity may stay at its ceiling only when guards prevent
   speculative loads outside the currently committed/valid extent.
4. Reserve before **every possible write**, including the spare row,
   multi-token verification, provisional rejected proposals, complete prefill
   chunks, bootstrap admission tokens and checkpoint restore.
5. Completed keys are shared only when their associated history/position is
   identical. No token-text deduplication across unrelated position/RoPE/image
   histories. Use existing safe prefix-clone provenance.
6. A shared completed row is never overwritten by a different owner. COW of
   an already completed block copies its key with its KV payload.
7. Partial raw tail, `idx_dead`, `idx_block_pos` and sentinel/spare semantics
   remain sequence-local. Completion from a shared partial KV page first makes
   the page exclusive, then produces the completed row privately.
8. Host ownership, device maps and payload publication are consistent before
   launching a reader. Fence outstanding readers before unmap/recycle/eviction.
9. OOM before publication is recoverable capacity pressure when rollback is
   complete; failed publication, transfer, mapping-access setup or uncertain
   device state is fatal, never permission to continue with stale data.
10. KV and indexer reservations commit together or leave no externally visible
    partial ownership. Cancellation and release return each resource once.
11. Canonical snapshots stay logical and validate fully before destructive
    restore. A valid but too-large restore is pressure; malformed state is not.
12. No increased public output cap, silently smaller context, or change in
    fixed-seed position-based sampling to hide resource pressure.

## Milestone A: incremental private contiguous storage

### A1. Extents and a storage interface

Introduce a small storage abstraction separating:

- ceiling rows (the logical context),
- physically mapped rows/bytes,
- valid completed rows,
- required read/write extent for the next operation,
- physical commitment and releasable chunks.

Use FP32 contiguous addressing. Reserve the full virtual address range once,
commit an initial bounded prefix, and preserve the pointer used by graphs.
Expose checked grow/shrink/ensure functions and explicit owner lifetime. Keep
geometry and arithmetic in GPU-independent helpers for overflow/boundary tests.

Suggested initial commitment is a **policy proposal**, e.g. 8K cells, rounded to
VMM granularity. Tune only after the measured granularity/latency gate. Keep
normal cudaMalloc storage as the explicit disabled-mode path.

The first opt-in implementation should be **single-GPU CUDA VMM only**. On
unsupported devices/backends, an explicitly requested mode must refuse with a
clear diagnostic. Default HIP/SYCL paths remain unchanged. Do not claim support
by replacing VMM with an allocation that moves captured pointers.

### A2. Prepare writers, readers and lifecycle

Centralize checked extent computation; do not scatter `+1` guesses among
kernels. Derive the actual accesses from append/spare behavior:

- reference/native append writes the completed block and `b + 1` spare;
- position zero initializes `dead` and the initial spare;
- speculative windows can complete blocks later rejected by verification;
- chunk prefill appends its entire chunk before scoring earlier queries;
- checkpoint restore writes a spare at the restored boundary.

Ensure the maximum touched row before launch, not only the finally accepted
history. Trim eligibility follows committed history and any preserved
checkpoint/paused-prefill needs, not the most recent requested position.

Audit full-capacity memsets, copies, validation span checks and debug hashing.
Reset only mapped bytes and validity metadata; future chunks are initialized
before publication. A short cached history must never retain validity for an
old longer history after reset.

### A3. Broker physical memory with the expert cache

Use a stable-address, reclaimable expert-cache allocator so growing indexers
can obtain physical chunks from evicted expert capacity while retaining the
VRAM reserve. Reuse VMM primitives where appropriate, but **do not enable
elastic attention KV as a side effect**: current unified KV excludes elastic
K/V, and indexer growth is a distinct resource.

At a fenced serving boundary:

1. Compute a complete request-wide allocation plan across all attention layers.
2. Reserve required chunks, keeping the configured VRAM floor and prefill/graph
   transient allowance. Best-effort extra growth is never required for progress.
3. Select only safe expert-cache victims; respect in-flight copies, prefill loans,
   adaptive residency publication and retained pointers. Quiesce incompatible
   background work or refuse the unsupported combination explicitly.
4. Map/initialize new indexer chunks and publish valid capacity only after all
   operations succeed. Make rolled-back eviction coherent even if old experts
   cannot be restored immediately; a cache miss is valid, stale residency is not.
5. On safe shrink, return physical chunks to a cache/free-chunk broker. Refill
   experts through the normal validated residency path, not a new unsynchronized
   copy. Use hysteresis to avoid growing/shrinking at every token or turn.

The existing `VmmRange::map_range` frees failed handles and can partially map a
range. Define ownership rules for borrowed expert handles, successful earlier
mappings and failed access publication explicitly. A failed transaction must
not double-free an expert handle or leave its slot advertised resident.

Start with a narrow supported matrix (one GPU, unified streaming, no
asynchronous adaptation or prefill borrowing unless proven safe). Unsupported
explicit combinations fail at startup; do not silently disable an existing
feature. Expand compatibility in dedicated tasks after the basic broker works.

### A4. Checkpoints, parking and admission

Prepare capacity before validating destinations as writable, but validate
untrusted snapshot metadata before changing ownership. Separate logical extent
checks from physical readiness checks. Parking/capture reads only valid keys;
restoring commits enough storage before copies and preserves failure atomicity.

Growth shortage joins the existing pressure/replay path with a distinct reason.
A queued writer must not hold `ctl` while waiting for another owner to release
memory. Preserve protected BYIELD owners, rolling headroom, late STOP semantics,
coherent/target-only draft state and zero-progress retry rules.

Acceptance: an 8K history in a 256K-configured session uses a bounded, reported
commitment; growth through 256K remains safe; released capacity actually raises
expert-cache capacity or its recoverable chunk budget. No quality claim from
allocation tests alone.

## Milestone B: pool completed indexer keys

### B1. Couple keys to backing-page ownership

Current QSA page size and compression block size are both 4. Use one completed
indexer row per **physical backing page per attention layer**, with ownership
controlled by the existing KV page refcounts/COW. Guard this geometry explicitly;
unsupported page/compression relationships are rejected initially.

The pool follows aggregate KV backing capacity, **not** the 32K GPU K/V residency
window. Every attention query can score all completed indexer keys; these are
not just the sparsely selected K/V pages. This plan keeps pooled indexer keys on
the GPU, incrementally committed where safe; CPU indexer offload is out of scope.

Physical page IDs can be scattered and recycled, so prefix-length commitment
alone is insufficient for the pooled representation. Track occupancy/refcounts
per VMM slab. Commit a slab before allocating any page within it; prefer already
committed free IDs if this does not violate allocator correctness/fairness.
Unmap only when no owned or in-flight page references that slab. Include
fragmentation in accounting and measurements.

Transaction changes cover host page allocation, layer-wide key validity,
KV COW, map publication, release, last-reference invalidation and ID reuse.
A new page is not an initialized key. Publish completion only after its payload
exists; clear stale completion on recycle, including failed restore.

### B2. Separate completed rows from mutable spare semantics

Do **not** redirect existing writes of `pooled[b + 1] = dead` into the shared
pool. The next physical page may belong to another sequence or not be allocated.

Provide an explicit reader/writer view containing:

- completed physical-key pool,
- logical-block -> backing-page map,
- valid completed extent,
- sequence-local raw tail/dead/spare and block-position state.

A completed logical row resolves through that sequence's backing mapping.
The logical tail block reads the private spare/dead value as the old kernel did.
Return **logical** cell IDs to top-k and downstream attention. Never expose
physical page order as logical position or tie-breaking order.

### B3. Port all append and scoring paths

Implement layout-aware accessors or new explicit pooled entry points without
changing FP32 normalization, pooling, rotation, reduction order or tie rules.
Cover reference/native append, single-token scoring, multi-row verify and
chunk prefill, including the tensor-core and SIMT scoring alternatives.

For tensor-core prefill, choose between directly loading mapped physical rows
and bounded contiguous staging **after measurement**. If staging is needed,
allocate per active executor/tile, not a full-context duplicate per slot. Include
its peak memory and copy cost in the savings report. Do not silently switch off
the fast scoring path to claim recovered memory.

Retain a fully functioning contiguous reference path for A/B and negative
controls. Diagnostics, raw indexer comparisons and FP16-check mode must either
support the new logical view or explicitly refuse it; never read a scattered
pool as a contiguous logical array.

### B4. Speculation, transfers and canonical persistence

Provisional block keys cannot become shared committed-prefix state before
verification accepts them. Reserve writable pages for the entire verification
window; on rejection restore valid extent and private tail/dead state and retire
unused pages. Existing KV rollback alone is not sufficient evidence of correct
indexer rollback. Test all accept/reject depths crossing four-token boundaries.

Clone/handoff shares completed keys by page ownership and copies only private
mutable state. Checkpoint rewind restores the spare privately, not by writing
into a shared completed key belonging to a longer sibling. Branching from a
partial page must COW before completion. Key content depends on logical position,
RoPE/scaling/image positions and model computation, not token IDs alone.

Snapshot capture gathers keys in logical order. Restore validates canonical
payloads, allocates backing+keys transactionally, scatters by mapping and
publishes only after copies land. Keep on-disk/canonical format unchanged if
possible; otherwise require an explicit version/migration decision and tests.
Do not enlarge parking images with unused committed capacity.

## Small implementation tasks and dependency graph

Each row is one focused work unit, not authorization to implement the whole
milestone or deploy. Exact new symbols/target names below are proposals.

| ID | Deliverable and acceptance | Dependencies | Verification |
| --- | --- | --- | --- |
| I1 | Allocation/access inventory plus checked row/byte arithmetic and boundary tests; includes spare/provisional extents | None | New CPU extent test; compare calculated current allocation against real `session_bytes` |
| I2 | RAII stable private indexer range, explicit committed/valid capacity, injected rollback failures | I1 | CPU allocator double + CUDA map/unmap/canary/graph replay test |
| I3 | Every private-mode writer/read/copy/reset/debug path obeys prepared extent; legacy off unchanged | I2 | Native/reference append and scoring parity, chunk and verify boundary matrix |
| I4 | Snapshot/checkpoint/slot transfer adapters prepare private capacity safely | I3 | Conversation-state/snapshot tests and failure injection |
| I5 | Safe cache/indexer budget broker with stable cache addresses and honest telemetry | I2 | Broker CPU model; real GPU shrink/grow, residency and reserve-floor checks |
| I6 | Admission/pressure integration, flag validation and bounded shrink policy | I4, I5 | Scripted pressure/cancel fixtures and private long-growth model test |
| P1 | Physical completed-key ownership/validity/slab model coupled atomically to KV pages | I1 | Randomized CPU allocator/COW/recycle/rollback invariants |
| P2 | Separate private spare/tail from completed-key append; reference and native parity | P1, I3 | All page-tail offsets and multi-step rejection GPU parity |
| P3 | Logical-map-aware decode and prefill scoring, including optimized score paths | P2 | Exact selected logical IDs; mapped permutation/canaries; score-path A/B |
| P4 | Pooled checkpoint/clone/handoff and logical canonical gather/scatter | P3, I4 | Shared snapshot tests, sibling isolation, restore failure injection |
| P5 | Incremental physical slab commitment and budget-broker integration for pooled keys | P4, I5 | Fragmented pool growth/shrink and reused-ID GPU tests |
| V1 | Bounded hardware gate tool for allocation, memory/performance and lifecycle matrix | I1 | Tool CPU tests using malformed/missing evidence negative controls; model gates wait for I6/P5 |
| V2 | Independent source audit, complete backend builds, isolated GPU/model acceptance report | I6, P5, V1 | Gates below; every unsupported/skipped combination disclosed |

Task bodies should name owned files, non-goals and commands before being turned
into Beads. Split I3/P3 by kernel family if the diff cannot fit one focused
review. No implementation task should require live production mutation.

## Verification and acceptance gates

### Fast / CPU

- Overflow, zero/full capacity, granular rounding and maximum touched row.
- Allocation failure at every layer/chunk; failed map/access publication;
  rollback has no stale ownership and no double-free.
- KV/indexer refcount equivalence, sparse physical IDs, last-reference release,
  all COW tail offsets 1/2/3, fragmentation and ID reuse.
- Shared spare isolation, divergent sibling tails, checkpoint rewind while a
  sibling retains the longer prefix, cancellation of protected partial reads.
- Snapshot valid-but-too-large versus malformed distinction; canonical format
  regressions, no accidental capacity-sized snapshot payload.
- Startup option matrix, pressure reason/reporting, counters and legacy mode.

Existing commands (new tests are added to these workflows):

```sh
cmake -S . -B build-indexer-cpu -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DSTRATA_BUILD_ARENA_LOAD_TESTS=ON
cmake --build build-indexer-cpu -j 4
ctest --test-dir build-indexer-cpu --output-on-failure
uv run --no-project --with jinja2 --with regex --with jsonschema \
  python -m unittest discover -s serve -t .
python -m unittest discover -s tools -p 'test_*kv*.py'
```

### CUDA / model, on isolated hardware only

1. Build baseline and candidate from frozen source/config identities. Compile
   every changed backend; CUDA-only modes must explicitly reject unsupported
   backends while leaving their default path buildable. HIP/SYCL compilation
   skips are blockers to claiming cross-backend support, not silent passes.
2. Synthetic parity: reference/native append, indexer keys, raw tails/dead,
   scores under the same score implementation and selected logical cell IDs.
   Exercise 0/1/3/4/5 tokens, row/slab boundaries, full capacity, random physical
   maps, COW, recycle, graph capture before growth and replay after shrink/grow.
3. Speculation matrix: no drafts, ordinary MTP, unified batch-MTP; all accepted
   counts, rejection across a block boundary, target-only restore, optional-row
   fallback, same-slot reuse after cancellation. Preserve coherence diagnostics.
4. Existing actual-model incremental/handoff/streaming/batch-MTP pressure gates,
   both admission orders, protected BYIELD, omitted output caps, third waiter,
   early close, sampled fixed-seed continuation and healthy follow-up.
5. Grow 8K -> 32K -> 128K -> 256K in a 256K-configured process, then shrink/reuse;
   two unequal histories, shared-prefix siblings and independent histories.
   A separate 512K allocation test is not proof of model quality beyond 256K.
6. Forced low-memory negatives: reject safely before launch or use proven
   pressure continuation; no CUDA paging/stall, overwritten sibling, reduced
   context, or unreported fallback. Capture peak VRAM during prefill/capture.

### Measurements

Record explicit disabled / incremental-private / pooled modes at identical
context, KV format/residency, expert budget, reserve, model, sampling and draft
settings. Freeze expert placement for arithmetic parity; use separately labeled
normal adaptive settings for performance. Do not compare two different expert
placements as a bit-exact proof.

Measure at 8K/32K/128K/256K; parallel 2 and 4 where supported; independent and
shared-prefix histories. Report:

- ceiling, mapped and valid indexer rows/bytes;
- committed physical slabs, unique live completed keys, refs and fragmentation;
- indexer and broker peak/transient bytes, growth/shrink/copy counts and latency;
- expert-cache bytes/slots **actually recovered**, not merely unused VRAM;
- pinned host backing and optional parking separately;
- prefill, single-stream and aggregate decode rates, TTFT, pressure/replay work;
- selected-ID/token parity and explicit failure/skip counts.

Proposed performance acceptance: no more than 3% median decode or 5% prefill
regression at fixed expert-cache capacity in repeated matched trials. These are
review gates, not expected measured results; revise explicitly if hardware
variance makes them inappropriate. A recovered-cache speedup does not excuse
hidden regressions in the new scoring path.

## Rollout, compatibility and reporting

Use proposed opt-in modes such as `--indexer-storage private|incremental|pooled`
(the exact CLI needs review). Pooled requires unified KV; unsupported flag
combinations refuse rather than silently fall back. Keep logical context and
aggregate KV reporting unchanged; add physical indexer commitment and ownership
metrics so operators can distinguish capacity from allocation.

Do not deploy during the current full eval. Live build/GPU tests and any model
unload require a separately approved isolated window, normal homelab preflight,
rollback copies and verification that router/audio and unrelated services remain
untouched. Historical rollout authorization in older work logs is not renewed
permission. First rollout should retain the old binary/config and explicitly
leave both new modes off until each independent gate passes.

## Rebase verification record

The rebase preserved all 68 fork commits and used recorded resolutions for
previously reviewed unified port conflicts. New conflicts in the NVFP4 trial,
slot-MTP attribution and adaptive-fetch reporting were resolved to preserve
upstream fixes alongside fork functionality. CUDA/HIP/SYCL compilation and
actual-model parity of this rebase remain required before deployment.

Local CPU-only CMake build and **19/19 CTests passed**. Direct standalone tests
also passed: shared pages 135,708 checks, reservation 73 checks, draft coherence,
batch-MTP policy and conversation cache 4,254 checks.

An initial 417-test focused frontend run found a missing `slot_releasing` field
in a hand-built lazy test fixture, and an orphan-slot recovery regression: the
fork waits for admission room before acquiring control, so upstream's recovery
inside the control-held loop could never run with all flags leaked. The fixture
was updated and owner-tracked orphan recovery was added before admission waits,
including wait telemetry and cleanup. Three targeted regressions then passed.
The final focused rerun passed **116 serving tests**; **205 KV smoke-tool
unit tests** passed independently. The full frontend rerun completed **807
tests**, with **12 skips and one failure** in
`test_service_holds_for_the_whole_request_including_early_close`. All six
`serve.test_keepawake` tests passed in isolation immediately afterwards;
`serve/keepawake.py` and its tests are unchanged from upstream. This is an
apparently timing-sensitive result, not a clean full-suite pass. A mistaken
focused command also named seven nonexistent test modules; the corrected
116-test command is the passing result above.

A first full frontend run was terminated at its 240-second command timeout;
that is not a passing test run. `cmake` was initially absent from PATH; the
successful build used an isolated `uv tool run --from cmake` installation.
No host CUDA toolkit is available, and no live build or GPU test was performed.
