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

## Deployment safety

User permits interrupting nibbler evals for this work. SSH as `root@nibbler`
works; `pete@nibbler` is denied. Do not stop unrelated services. Existing
production launcher/binary/config remain rollback baseline. Native build uses
CUDA 13.3, GCC 15, architecture 120, Release, conversation tests enabled.
