# Unified KV implementation work log

Branch: `feature/multi-slot-unified-kv`, targeting `peterkeen/strata`.

The deployed nibbler branch predates upstream batch serving. Fork main at
`6f32ec0` already implements independent active slots, multiplexed output,
batched decode, prefill interleaving and cooperative preemption. Merge
`6346b3e` combines that scheduler with the existing nibbler changes. Initial
parallel/runconfig/setup tests: 23 passed.

## Initial implementation contract

- Opt-in `--kv-unified` with existing `--batch N` / config `parallel`.
- One physical full-resident KV pool per attention layer, shared by admission
  session and active slots; total capacity is `--max-context`, not N times it.
- Per-slot recurrent, PLE and QSA indexer state remains independent.
- Logical page mappings, reference-counted prefix sharing and copy-on-write.
- Reuse upstream serving/scheduling; do not introduce a second protocol.
- Reserve output capacity before prompt execution; pressure must not overwrite
  active sequences or kill them. Idle caches are reclaimable. Admission can wait
  while active requests complete.
- Preserve default independent-slot and single-slot execution.
- First version: single session GPU, full-resident FP16/INT8/Q4/K8V4 KV.
  KV streaming (`--kv-resident`), host conversation parking and pipeline groups
  are excluded initially and must be explicitly rejected, not silently disabled.
  This restriction concerns KV streaming, **not expert streaming/caching**.
- Existing upstream batch limits (no MTP drafts or penalties in batch windows)
  must be documented; solo requests retain upstream behavior.

## Work units

1. Host page allocator and ownership/COW tests.
2. Borrowed KV session allocation, private state, shared RoPE, safe zeroing.
3. Logical-page-aware canonical conversation snapshot transfers.
4. Unified runtime and integration with admission, copy-to/from-slot, reserve,
   cancellation, eviction and reuse.
5. Build/test merged baseline and implementation on nibbler; compare solo and
   interleaved requests, boundaries, exhaustion, cancellation and cache reuse.
6. Independent adversarial review, fix findings, and controlled deployment with
   rollback artifacts. Do not claim production-ready before GPU validation.

## Using the full-resident milestone

Add `"parallel": 2` (or 4) to the Python server's model config, and add these
engine options to its `args`:

```
--kv-unified --kv-resident 0 --conversation-cache-mib 0 --max-context 32768
```

For a native engine, use `--serve --batch 2` as well. `--max-context` is both
the per-request logical ceiling and the aggregate physical cell budget. It is
not divided by the slot count; two requests can use unequal shares. Physical
capacity rounds up to four-cell pages. A request reserves its prompt and
bounded output before writing; references to cached prefixes share pages until
a writer needs copy-on-write. With insufficient space, idle caches are evicted
and active slots continue decoding while admission waits. Protected paused
prefills must be explicitly cancelled with `BSTOP` when abandoned. A request
which still cannot fit receives an error without overwriting another slot.

`/props` exposes `kv_unified` and `kv_capacity_cells`; `/slots` exposes the
engine's actual supported slot count and logical `n_ctx` for each slot.
Recurrent state, PLE history and the QSA indexer are still private allocations:
this removes replicated attention KV pools, not every per-slot allocation.

A 262144-cell full-resident pool also fits in the tested nibbler configuration,
but displaces experts from VRAM. In the four-slot native probe each extra session
was about 0.50 GiB, the expert cache was 3.52 GiB, and only 327 MiB VRAM remained.
That short probe is **not** evidence of safe long-context production headroom.
Keeping 262K logical context with only 32K GPU-resident KV requires a subsequent
shared streaming residency manager, shared authoritative host backing,
per-sequence logical-to-host addressing, safe page upload/eviction before every
attention read, and tests of residency churn across active and paused slots.
Simply aliasing the existing per-session streaming pools is incorrect.

## Validation evidence (nibbler, 2026-10-05)

Private subprocesses used the actual configured Qwen model, without replacing
the production binary or config. Evidence is under
`/data/llm/Strata-tests/multislot-20261005/`.

- Host allocator: 135298 checks, also run under ASan/UBSan.
- Full native build succeeded. 79 of 82 CTest targets passed, including allocator,
  borrowed session allocation, existing conversation snapshots and remapped
  shared-KV snapshots. Three data-dependent targets (`ple_parity`,
  `expert_parity`, `pool_test`) failed because their hardcoded Q2 GGUF/pack
  fixtures were absent; they did not reach inference.
- Two slots, 32 generated tokens each: Q4, K8V4 and FP16 each byte-identical
  to their own solo references at a 32768-cell pool. INT8 two/four-slot and
  upstream interleaving probes also passed.
- `tools/unified_kv_smoke.py` passed at aggregate capacities 512 and 1024:
  unequal overlapping requests, pressure waiting while active decode completes,
  queued `BSTOP` releasing capacity, oversized request followed by a healthy
  admission, and cached-prefix branches at tail remainders 1, 2 and 3 with
  explicit prefix reuse and token-by-token solo parity.
- Reviewed/fixed version: full Python server suite 292 tests, 7 skipped,
  no failures. Scripted-engine regressions cover close-before-capacity-error,
  immediate next request, cancelled resume waits, yields discovered by drains,
  legacy ERR-only responses and slots kept busy until stop acknowledgement.
- New model-backed paused-prefill stages passed at both 512 and 1024 cells:
  `GEN`/`BYIELD`, then pressure-blocked `BGEN` with queued `BSTOP` and no active
  decode rows. Stop acknowledgement preceded the waiter's output/admission;
  its tokens matched solo. These stages use 64-token prefill chunks without
  changing the engine's yield guard.
- Fresh upstream interleaving probe at 65536 cells passed every solo comparison:
  long-prefill interleaving, yield/resume, next-turn cache reuse, checkpoint reuse
  without prior thinking, and slot-to-solo transition with MTP drafts enabled.
- Private HTTP server on loopback port 5802 reported two slots and shared capacity
  correctly. `tools/early_close_test.py` passed both solo and concurrent early
  close followed by a clean next answer. The private server was stopped afterward.
- Independent final source review found no remaining Blocker/High defect.
  CUDA transfer failure paths are fatal by inspection; device-failure injection
  and full 262K production headroom/stress testing remain unverified.

Parity probes freeze expert placement, use `STRATA_IQ_MT_MIN=1`,
`--pcie-frac 0`, and disable prefill borrowing where appropriate. Their short
aggregate token rates are not production throughput benchmarks.

## Shared streaming continuation (validated for two-slot rollout)

Shared streaming now keeps the original logical context and bounded parking:

```
--kv-unified --batch 2 --max-context 262144 --kv-resident 32768
--conversation-cache-mib 4096 --conversation-cache-slots 4
--conversation-cache-min-free-mib 4096 --vram-reserve-mib 2048
```

The Python config uses `"parallel": 2` instead of a manual `--batch` option.
One pinned authoritative backing pool and one GPU residency/CLOCK cache exist
per attention layer. The allocator has 262144 aggregate backing cells; each
request retains a 262144 logical ceiling, and GPU residency is 32768 cells,
not that amount multiplied by the number of slots. Private logical-to-backing
maps and selected logical-to-GPU reader views are allocated before capture.
Writers consult the shared backing-to-GPU map, never stale reader views.

COW copies authoritative host rows. Last-reference releases, recycled IDs and
restored ranges invalidate the corresponding backing cache entries. Prefill
has separate, logically addressed staging workspace; it is not part of the
32768-cell persistent GPU cache budget. Private recurrence, indexer, PLE,
MTP and captured graph allocations still consume memory.

Canonical snapshot validation does not require a ready destination mapping.
Outgoing conversations are parked before destructive preparation. An optional
canonical restore is skipped, without discarding a fitting live/slot prefix,
if its fresh private pages cannot fit. Contiguous backing runs are coalesced
for transfers. The parking budget is separate from pinned authoritative KV
and does not guarantee four full-length parked histories.

The default MTP window retains its private ring. A full-context MTP window
(`0`, or at least the context) instead allocates private **fully resident**
draft KV, with an explicit allocation error if it cannot fit. It never becomes
an unmanaged shared-stream owner. Slot promotion still has upstream's stale
MTP-proposal history: target verification preserves returned-token semantics,
but draft acceptance/performance need not equal a clean solo conversation.
Shared transfers discard retained canonical-buffer provenance.

Current nibbler evidence, alongside the original milestone artifacts:

- Native engine and all test targets build; 80 of 83 CTest targets pass. The
  same three missing-data fixtures fail before inference.
- Shared allocator: 135660 checks. Updated real-CUDA allocation/private-draft
  fixture: 2857 checks, including production-used MTP ring-boundary checks.
- New shared CUDA parity passes FP16, INT8 and Q4: COW/recycling, stale-view
  writers, private reader materialization, mapped staging, CLOCK eviction,
  multi-query protection, graph replay and overflow canaries. Existing streamed
  parity and private-ring restore also pass for all three formats.
- Python server: 299 tests, seven skipped, no failures. CPU-only smoke harness
  tests: 66 passed across the original and new harnesses.
- Two-slot model probe at 262144 / 32768 / 4096: both 64-token outputs match
  solo references with 2048 MiB reserve. Startup free VRAM was 1461 MiB in
  that private probe. These frozen-placement short rates are not benchmarks.
- `streaming-short.json` passes positive canonical A→B→A parking/restore,
  unequal overlapping requests, cached partial-tail branches at offsets 1/2/3,
  cancellation and same-slot readmission. This harness disables MTP drafts.
- Private HTTP reports two 262144-context slots and the shared backing/GPU
  capacities correctly; solo and concurrent early-close tests pass, followed
  by clean next answers. That probe excluded the separate vision process.
- `streaming-long.json`: all 28 stages pass with overlapping 20000/20004-token
  histories, 40516 reserved cells, positive restore diagnostics and solo parity.
- `streaming-over-resident.json`: all 28 stages pass with overlapping
  35000/35004-token histories. Each individual history exceeds the 32768-cell
  GPU cache, with 262144 aggregate backing retained.
- `streaming-pressure.json`: 30 stages pass at 24576 backing / 20480 resident.
  BSTOP acknowledgement precedes waiting output/admission, with solo parity.
- A repeat three-slot upstream interleaving probe at 2048 MiB reserve passes
  every comparison: long prefill, BYIELD/resume, next turn, checkpoint reuse and
  slot-to-solo MTP transitions. An explicit full-context `--mtp-window 0` probe
  also passes two-slot/solo parity with drafts enabled.
- The production-settings private HTTP gate retains adaptive/fused-prefill,
  MTP and GPU vision. Early-close recovery passes; a synthetic red PNG produces
  `red`. Startup free VRAM was 1674 MiB in the first such probe. The exited-child
  broken-pipe cleanup regression is fixed; the final probe exits cleanly (0).
- Read-only adversarial review's optional-restore and full-context-MTP High
  findings are fixed. Its focused recheck found no new blocker.

The original **700 MiB reserve is not a safe three-slot deployment gate**:
that private interleaving probe had only 115 MiB free at startup. Its initial
three outputs and next-turn reuse matched solo, but a later graph instantiation
failed out of memory. Do not report that probe as passing. The larger-reserve
repeat, longer overlapping histories, backing pressure and production-settings
HTTP/vision gates pass. The tracked deployment uses two slots, not three.

Model diagnostic misses/RAM reads alone do not prove CLOCK eviction. Forced
churn is established by the synthetic CUDA regression, not by short sparse-QSA
model histories. Device-failure injection, exhaustive long-context stress and
production throughput remain outside the current evidence.

## Deployment safety

User permits interrupting nibbler evals for this work. SSH as `root@nibbler`
works; `pete@nibbler` is denied. Do not stop unrelated services. Existing
launcher is unchanged; the preceding binary and both configs are saved as
rollback artifacts. Native build uses CUDA 13.3, GCC 15, architecture 120,
Release, conversation tests enabled.

## Controlled rollout (2026-10-05)

Deployed through the existing llama-swap launcher, without restarting the router
or unrelated services. Only `qwen3.8-flash-next-iq3_s` was unloaded/reloaded;
the inactive standalone `strata.service` was not started. Live `/props` reports
`total_slots=2`, `kv_unified=true`, `kv_capacity_cells=262144`, and both
`kv_resident`/`kv_resident_capacity_cells=32768`. Both `/slots` entries retain
`n_ctx=262144`. Router warmup returned `READY`; live backend solo/concurrent
early-close recovery also passed. Shared Chat settings hash remained unchanged.

Artifacts in `/data/llm/Strata-tests/multislot-20261005/`:

- `streaming-rollout.{props,slots,warmup,running.after}.json` and rollout log.
- `streaming-rollout.strata.before`, tracked/runtime config backups, and
  `streaming-rollout.{before,after}.sha256`.
- `deploy-streaming.sh`, with automatic rollback on a failed gate, and
  `rollback-streaming.sh`, which restores both configs and the preceding binary
  but never overwrites shared Chat settings.

Binary SHA-256 before:
`ec160772e1f34d0dc5c5730f82287312c1cbb494e5924012857b99dac76143a4`.
Deployed binary:
`32e6183c3b1eb6fdc0712edfa0e9e13934e8112d85cb376acf0de3f6f8dc85c8`.
Deployed tracked/runtime config:
`e806bfeb934c1fe7350ec305d49af66ed8089bbeb8d5198b032967e236b38166`.

Rollback command:

```
ssh root@nibbler /data/llm/Strata-tests/multislot-20261005/rollback-streaming.sh
```

The pre-existing `nibbler.local.keen.land` allowed-host addition is preserved in
the local/deployed working config, not included in this feature's staged change.
PR #1 was already merged as the historical full-resident milestone; the streamed
continuation is published as a follow-up rather than rewriting that merge.
