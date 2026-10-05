# Nibbler validation and deployment — 2026-10-03/04

Branch: `nibbler/prefill-and-conversation-cache`. Upstream was fetched and verified
at `99f3dbd0b21d1401b3769e0c0d963913607f380b`. No fork or upstream push was made.
Hardware: RTX 5060 Ti 16 GB / Ryzen 9 9900X; existing CUDA 13.3/GCC 15 Release build.

## Enabled production changes

1. Native fused full-chunk prompt experts (`STRATA_PF_FUSED=1`).
2. Host conversation parking: 4096 MiB budget, four-entry cap, 4096 MiB available-RAM
   floor. Large snapshots can be rejected; this is not four guaranteed full-context
   slots or a hard OS memory reservation.
3. Speculative verification/commit is bounded by remaining output/context and first
   EOS, in both CLI and serving. Hidden accepted tokens after length/EOS no longer
   enter persistent model state.

The small-tail optimization was implemented and tested, but **not enabled**: neither
rebalancing nor routed native fused tails demonstrated a reliable end-to-end win.
Production retains the existing MMQ small-tail fallback. Both experimental knobs
remain available for controlled comparisons; `STRATA_PREFILL_FUSED_TAIL=0` is
explicit in the deployment config and balancing is off by default.

## Matched prefill timings

Four fresh alternating A/B samples per size, median engine `prompt_ms` (not just
CUDA kernel/timeline time), same model/template/context/speculation/vision. All
performance samples reported zero cached tokens. Parking and adaptive swaps were
disabled for these comparisons; decode PCIe fraction was zero and the one-row IQ
GPU path was forced with `STRATA_IQ_MT_MIN=1`. Production keeps its original adaptive
settings and does not force that diagnostic knob.

| Prompt tokens | Original upstream MMQ | Final branch defaults | Less prefill time |
|---|---:|---:|---:|
| 9,177 | 5.909 s | 5.468 s | 7.5% |
| 18,177 | 11.079 s | 10.264 s | 7.4% |

This is approximately 8% higher prefill throughput. It is not an all-workload
throughput or decode-speed claim.

The original fused-only control had a 5.451 s 9k median. Tail prototypes measured:
rebalanced chunks 6.044 s; routed per-expert fused 5.654 s; contiguous batch-8
5.503 s; batch-8 spanning inactive ids 5.718 s; pipelined batches using the already
allocated large ring 5.479 s. The final prototype was within roughly 0.5% of the
control, not a convincing improvement. No slower experiment was enabled.

## Quality and correctness evidence

- Ten matched quality fixtures cover arithmetic, Python/code, binary search, SQL,
  JSON extraction, French, Chinese, two code prompt lengths, and GPU vision.
  **9/10 correct in both upstream and final branch; no new quality failures.**
- The added 8,720-token code fixture expects 463 but returns 232 in original MMQ,
  fused-only, and tail candidates. Its expectation was not changed; it is explicitly
  reported as a pre-existing failure, not counted as a pass. The 28,560-token code
  fixture and vision fixture pass.
- Seven standalone output-limit/chunk-schedule/conversation tests pass.
- Native fused CUDA reference tests pass for all six exercised format pairs.
  Launch partitions 1/8/32/128 match the original fused per-pair outputs exactly in
  those tests, including inactive interior/end ids with null blob pointers.
- Model-level caps 1..8, natural EOS, consumed-state lengths, and immediate live-prefix
  reuse pass on the final binary.
- A/B/A parking passes byte-exact main-model-state and output checks for one-token
  windows; speculative-window-4 A/B/A passes output/reuse checks. Byte-exact state is
  not claimed for the window-4 comparison. The parity harness excludes irrelevant
  padding/stale cells and the drafter's uncomputed final cell.
- Byte-pressure (350 MiB) and oversized (1 MiB) fallback scenarios pass output and
  byte-exact main-model-state checks.
- Cold raw-engine boundary probes cover batched tails 63/64/512/978/1023/1024 with
  tail fusion off/on; output and logical consumed length agree. These truncated-data
  probes naturally emit EOS and are boundary/coherence checks, not a quality score.
- Python suite: 143 tests, five skipped, no failures, using nibbler's existing venv.
  The quality validator now rejects boolean-as-number answers and malformed schemas.

Per-run host staging/PLE telemetry now reports deltas rather than cumulative totals.
These smoke/reference gates are not proof of equivalent quality on all workloads.
Maximum-context, mixed long-running concurrent-client, broad stochastic-quality,
and other-GPU/HIP validation remain outside this run.

## Live deployment verification

llama-swap still uses its original model name/aliases and launcher path; the launcher
now delegates to this branch. Only Strata was unloaded/reloaded. The original binary
`engine/strata` and original production JSON were verified byte-identical to backups.

The first switch rolled back automatically because the service user could not create
a log in the root-owned checkout. The corrected deployment keeps mutable files in
`/data/llm/Strata-run/nibbler/`, owned by `native-inference` with mode 0700. Existing
shared Chat defaults were copied byte-identically; restarts refresh the config but
not the shared settings. Tracked source/config files do not become service-writable.

Live proxy A/B/A returned 391 / Lin / 391. A-again restored 182 parked checkpoint
tokens; its prompt time was 161.5 ms versus 987.0 ms for cold A. Native log confirms
RAM restoration in 15.6 ms. Snapshots were approximately 228 MiB. Health confirms
loaded, context 262144, and images enabled; the running command uses
`engine/strata-nibbler` and all three parking limits.

Validated deployed binary SHA-256:
`ec160772e1f34d0dc5c5730f82287312c1cbb494e5924012857b99dac76143a4`.
Native source last changed at `6cf61ad`; later commits adjust test tooling,
deployment files, and documentation.

## Evidence and rollback

Remote evidence: `nibbler:/data/llm/Strata-tests/implementation-20261003/`.
Local results/report/bundle: `/home/pete/strata-implementation-20261003/`.
Important records: `upstream-full.results.json`, `branch-final-default.results.json`,
`final-default-native-tests.log`, `final-default-fused-reference.log`,
`final-checks.log`, `final-python-tests.log`, `tail-boundaries.results.json`, the
`final-spec-state` / `final-cache-*` directories, and `production-branch-smoke.json`.
The upstream ten-fixture record combines its original nine-fixture run and the
separate targeted rerun of the newly added code fixture.

Original launcher, binary and JSON are saved as `launcher.before`, `strata.before`,
and `production-config.before.json`. Follow `README.md` for an idle-window rollback.
The router config and unrelated pocket-tts service were not changed.

To publish later, create your GitHub fork, add it as a separate remote, and push
`nibbler/prefill-and-conversation-cache`. Do not use upstream's auto-update script
blindly on this deployment; build/install the branch binary deliberately.
