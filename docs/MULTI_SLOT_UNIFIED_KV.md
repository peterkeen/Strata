# Unified KV implementation work log

Status: **port in progress on `kv-unified`; the port is being validated**.
Port base: upstream main `fb58e0db` (following the 0.1.40.x lineage;
`CMakeLists.txt` identifies this base and the port as 0.1.41).

Historical branch: `feature/multi-slot-unified-kv`, targeting `peterkeen/strata`.
All measurements, test counts, source/binary hashes and rollout statements below
belong to the fork lineage and the recorded 2026-10-05 runs, not to the upstream
port. The milestone descriptions, including restrictions later lifted, remain
historical. Old source references and operator/rollback instructions are not
current port validation or authorization for live changes. See the
[ported deployment candidate](../deploy/nibbler/README.md) and its
[source/config check](../deploy/nibbler/validation.md) for current status.

The later [incremental allocation continuation](INCREMENTAL_UNIFIED_KV.md)
replaces this milestone's full-output reservation policy with known-prompt plus
rolling headroom, exact writer preflight and internal pressure/replay. Omitted
output limits no longer require nearly a whole-pool admission reservation.
Earlier measurements and rollout artifacts below remain historical evidence.

The historical deployed nibbler branch predates upstream batch serving. Fork main at
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
  must be documented; solo requests retain upstream behavior. **Partly superseded**
  for the opt-in `--batch-mtp` path (one MTP proposal per slot per window); the
  no-penalties limit still applies. See the batch-MTP milestone below.

## Unified batch-MTP milestone (opt-in, measured 2026-10-09)

Status: **opt-in, off by default, measured on isolated hardware.** `--batch-mtp`
(or `STRATA_BATCH_MTP=1`) lets each batch slot verify one MTP proposal per window
on one GPU, with `--serve --batch 2..8 --mtp --spec T >= 2` and no layer split or
helper GPU; `--kv-unified --batch-groups 1` is the measured configuration. Without
the flag the batch path and the solo path are unchanged, so this milestone changes
no default.

Measured (private subprocesses on an RTX 5060 Ti, validation binary
`e14c60c7…`, config `5dcb8e12…`, runtime unchanged from `ff48f4b7…`): dual-slot
joint overlap with exact 16/16 token parity and offered 17 / accepted 13 /
rejected 4 (including a forced accept and a forced reject); the max-new edge and
the cancel/same-slot-reuse edges; terminal target-only tails at partial-page
offsets 1/2/3; a shared-pool pressure park of 2048 tokens with an exact 2048-cell
canonical restore and zero resumed MAIN drafts; the optional-row
`fallback_reserve` witness (slot 1 `fallback_reserve=1`, one attributed park,
sibling cancel); and the coherent full-slot `BHANDOFF` transfer (3-token source
prefix, 1538/1538 unfed-prefix reuse, 30 resumed MAIN draft offers, exact 64-token
solo parity). Detail, commands and per-gate numbers are in
[INCREMENTAL_UNIFIED_KV.md](INCREMENTAL_UNIFIED_KV.md).

The private draft ring is bounded by
`mtp_kv_ring_cells(window, max_cells, max_t) = window + 4*max_t + 64` when
`0 < window < max_cells`, and `-1` (private, fully resident draft KV) otherwise.
Measured at `--max-context 4096 --mtp-window 128` with `max_t = 2`: 200 cells
allocated while the run's prompts were 1536/1540 cells and its consumed history
2048 cells. Performance is recorded separately: a three-repetition A/B measured an
aggregate throughput gain of 5.4 % to 9.8 % and a dual-decode-window gain of
7.3 % to 14.2 % (99.7 % of 956 proposals accepted), with the ~1 GiB slot-session
cost held constant in every arm and therefore not measured by that A/B.

Historical statements this milestone supersedes: the "no MTP drafts in batch
windows" limit now applies only without the flag (penalties are still not applied
in batch windows), and "slot promotion retained upstream's stale MTP-proposal
history" is superseded by the explicit target-only suppression above. The engine's
own `--help` line for `--kv-unified` still lists `--batch-mtp` among the options
it excludes; that string predates this change and no longer matches the measured
runs (it needs a source edit by the primary). Not claimed: HIP/SYCL builds,
byte-level ring internals, COW page identity, park ordering, multi-token divergent
suffix restore (UNTESTED), output quality, and no production deploy or reload.

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
an unmanaged shared-stream owner. At this streamed milestone, slot promotion
retained upstream's stale MTP-proposal history, checked by target verification.
The incremental continuation instead explicitly suppresses MTP/suffix proposals
after target-only transfers until a full residual replay rebuilds coherent draft
history, without removing the private ring. Shared transfers discard retained
canonical-buffer provenance. The measured batch-MTP milestone (one proposal per
slot per window, the optional-row fallback and the coherent full-slot transfer) is
recorded above and in [INCREMENTAL_UNIFIED_KV.md](INCREMENTAL_UNIFIED_KV.md).

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

## Production admission incident and server-only fix (2026-10-05)

After the streamed continuation merged, three overlapping client requests exposed
Python admission/cancellation defects not covered by the earlier bounded-output
HTTP gate. This was not an observed CUDA OOM: the original engine process remained
alive. A read-only Python stack dump found a request holding `ctl` while waiting
for a free slot; the request path stopped progressing and the router returned 502s.

The server-only correction:

- New admissions wait for usable slot/backing capacity **before** taking `ctl`,
  and recheck availability after acquiring it. Paused owners can resume even
  when other waiters cannot acquire a slot or fit their backing reservation.
- Yield ownership is copied into request-local cleanup state and cleared before
  handing off `ctl`. Cancellation no longer leaves a stale `YIELDED` marker for
  the next unrelated request to interpret as its own protected paused slot.
- Shared-KV preemption requires an otherwise idle pool and room for both output
  reservations. Paused reservations are published before the control handoff,
  so a late unlimited request cannot steal admission and strand their owner.
  Legacy independent-slot preemption remains supported.

Omitted or non-positive output limits still mean the remaining logical context.
With unified KV this can reserve nearly the entire aggregate backing pool, so
such requests may legitimately serialize. Bounded output caps are needed when
concurrent prompt/output reservations must fit together; no context ceiling or
memory budget was reduced by this fix.

Final validation and rollout evidence:

- Python suite: 308 tests, seven skipped, no failures. New deterministic tests
  reproduce full-slot lock starvation, cancelled-yield marker leakage, acquisition
  races, late capacity-blocked waiters, and paused-owner resumption.
- Actual production model/GPU/HTTP: three bounded requests with 36202/6202/802
  prompt tokens and 128/1024/64 output caps all complete. Sampling observes two
  slots decoding together and additional admissions waiting. This also passes
  after cancellation and subsequent reuse.
- Three requests with omitted output limits (36189/6189/789 prompt tokens) all
  complete, without unsafe preemption of the whole-pool reservation.
- A distinct 36212-token prompt demonstrably yields while two other clients
  compete; closing that stream does not strand either waiter. A later healthy
  answer and sampled zero running/waiting requests verify recovery.
- Only the Qwen backend was reloaded through llama-swap. Engine binary, tracked
  config and shared Chat settings hashes remain unchanged. Two slots,
  262144 backing/context, 32768 GPU residency, 4096 MiB parking and 2048 MiB VRAM
  reserve are retained. These synthetic checks are not a throughput benchmark.

Evidence under `/data/llm/Strata-tests/multislot-20261005/` includes
`admission-deadlock.py-stack.txt`, incident metrics/logs,
`admission-hotfix.http.json`, CPU test logs, and server-source rollback artifacts.
Two initial gated attempts automatically restored the previous source: an
idle-only assertion conflicted with real traffic, then a reused-prefix cancellation
fixture did not actually yield. Those attempts are not counted as passing; the
final gate uses a distinct prompt and observes its own yield explicitly.

Deployed Python server SHA-256:
`fc9a631650d08f37fc7bb861d0047e44770b2e41559f7769c5b11610b29a4e1a`.

Server-only rollback (restores the preceding server, including its known bugs):

```
ssh root@nibbler /data/llm/Strata-tests/multislot-20261005/rollback-admission-hotfix.sh
```
