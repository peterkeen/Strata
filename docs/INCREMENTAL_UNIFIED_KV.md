# Incremental unified KV work log

Status: **validated and deployed on nibbler, 2026-10-05**.
Branch: `feature/incremental-unified-kv`, based on fork main `cc82936`
(the merged admission fix `d9f737a`, PR #3).

## Reason for the change

The original full-resident milestone reserved prompt plus the entire permitted
output before admission. This simplified the guarantee that admitted sequences
would not exhaust backing later. It is a scheduling policy, not a requirement
of authoritative host backing, shared GPU residency, private logical maps or
copy-on-write.

The Python admission fix addressed lock starvation and stale yield ownership,
but full-output reservation can still serialize requests unnecessarily. The
2026-10-05 production observation had one 21598-token prompt with 29751 output
tokens progressing, and a 146614-token prompt waiting with an output allowance
of 115522. Their actual histories fit within 262144 cells; their potential
output reservations do not. Pete approved replacing that policy.

## Preserved architecture and budgets

- Two serving slots, one loaded model, existing llama-swap launcher.
- Logical context ceiling **262144 per request**; aggregate shared backing
  **262144 cells**, not divided between slots.
- Shared GPU-resident cache **32768 cells per attention layer**.
- Authoritative pinned host backing; reference-counted logical page ownership,
  COW, last-reference invalidation and shared GPU CLOCK residency.
- Private recurrence, indexer, PLE and MTP ring; stable captured-graph buffers.
- Optional canonical conversation parking **4096 MiB**, four entries, physical
  RAM floor **4096 MiB**; VRAM reserve **2048 MiB**.
- No forced HTTP output caps or changed omitted-limit/context semantics.

## Implementation contract

1. Reserve the known prompt and modest rolling output headroom, rather than the
   entire possible response. Initial headroom is at most **256 cells**.
2. Before every target write, preflight the exact exclusive extent, including
   speculative verification rows and COW. Desired headroom is best-effort;
   retry the exact required extent before treating a shortage as real pressure.
3. Trim unused tails and reclaim idle mappings first. Atomic allocator shortage
   is recoverable; allocation/publication/device failures are not pressure and
   must never permit inference from uncertain state.
4. Actual exhaustion may safely pressure-park an active decoder, release its
   backing with residency invalidation, then acknowledge it. Preserve client
   tokens and continue the same request later; do not return pressure as a
   completed response or silently drop the last unfed output token.
5. Canonical parking is optional and remains budget/RAM-gated. Capture slot
   target state only, never an unrelated main MTP ring. If an image cannot be
   stored, is evicted or cannot be restored, resume through full token replay.
6. Track main draft coherence explicitly. Slot clone-back / target-only restore
   must not silently use another conversation's MTP state. Keep the private
   ring allocated; suppress MTP/suffix/coupled proposals until a coherent full
   residual replay has rebuilt it.
7. Give an active owner bounded actual progress (nominal admission-pressure
   quantum **256 tokens**) before preempting for a pending prompt. Pressure
   retries must make progress, not oscillate indefinitely or busy-spin with no
   runnable owner. BYIELD partial owners remain protected, not asynchronously
   invalidated.
8. Fresh Python admissions must not take the control lock without a usable
   slot/backing opportunity. For incremental engines, paused footprint and
   preemption checks use known prompt plus bounded headroom, not full max_new.
   Preserve the previous cancellation / stale-YIELDED fixes and legacy mode.

## Protocol and sampling

New unified engines advertise:

```
INFO ... kv_incremental=1 kv_reserve_ahead=256
```

After safely releasing an active pressure victim:

```
BDONE <slot> <produced-in-this-segment> pressure <elapsed-ms>
```

This is an internal continuation boundary, not an HTTP finish reason. The
frontend re-submits **original prompt IDs plus all already-returned output IDs**
with the remaining output allowance. The final token is included even though
it has not yet been consumed by the model. Cancellation is honored while
parked, queued and replaying. Old engines retain their existing reservation
and preemption policy.

Sampling uses Philox(seed, **absolute logical input position**):
`Verifier::run` sets the counter from `pos0`; batched sampling uses the row's
logical position. Preserve a request's seed across continuations. Do **not**
introduce an RNG offset: reconstruction already advances the position correctly,
and an extra offset would double count. Sampled, fixed-seed replay needs an
actual model parity gate, not only greedy or scripted-engine tests.

## Ownership and verification plan

Normal subagents share this checkout; no worktrees or detached orchestrators.

| Owner | Files / responsibility |
|---|---|
| Native `0559d598-6e9c-4f97-9196-1e52012b40e6` | `generate.cpp`, runtime as needed, new reservation policy header/test |
| Frontend `ccd2038a-f3c0-423e-aec7-5cdfb975cda1` | `server.py`, lifecycle/reporting and incremental scripted tests |
| Harness `bd91373c-aca6-4541-9eb6-d4c2ea0cc9f9` | New incremental model/HTTP smoke tools and CPU harness tests |
| Audit `328d05b0-9b69-4ae5-81b6-991e0af5b312` | Independent read-only pressure/coherence/fairness review |
| Primary | CMake, docs, integration, native build, actual model/GPU/HTTP validation, deployment |

Required gates (results below distinguish completed checks from pending model gates):

- CPU rolling allocation / exact-fallback / COW / fatal-failure policy tests.
- Scripted frontend pressure re-admission, remaining-budget/token accounting,
  stable seed, third waiters and cancellation at every handoff, legacy behavior.
- Actual model: individually whole-context-sized allowances overlap when actual
  histories fit; both emit before either completes, with frozen greedy parity.
- Actual small-pool exhaustion: explicit pressure acknowledgement, continued
  streams, forward progress, solo parity, partial tails, cancellation and healthy
  reuse. Disabled/tiny optional parking must exercise replay fallback.
- MTP-enabled target-only pressure restore: explicit suppression/coherence
  diagnostic and target-token parity, not verification of only present images.
- Fixed-seed sampled re-admission parity with position-based counters.
- Production-settings HTTP with omitted output caps and overlapping large
  histories (approximately 146K and 22K where practical): bounded SSE observation,
  then early close and healthy follow-up. Keep GPU vision and adaptive settings.
- Independent final source audit, controlled model-only rollout and rollback.

## Deployment baseline

The pre-incremental production baseline was retained and hash-verified before
rollout. Only the native binary and Python server were replaced; tracked/runtime
configuration and shared Chat settings were unchanged. Rollback copies:

- Native `engine/strata-nibbler`:
  `32e6183c3b1eb6fdc0712edfa0e9e13934e8112d85cb376acf0de3f6f8dc85c8`.
- Python `serve/server.py`:
  `fc9a631650d08f37fc7bb861d0047e44770b2e41559f7769c5b11610b29a4e1a`.
- Tracked/runtime config:
  `e806bfeb934c1fe7350ec305d49af66ed8089bbeb8d5198b032967e236b38166`.
- Shared Chat settings:
  `b61e757d0d94ca1f1c7c17cf180f98e779cbd976d53658a88ae91d870d9c3e2a`.

Evidence directory: `/data/llm/Strata-tests/multislot-20261005/`, with new
`incremental-*` artifacts. Build uses existing CUDA 13.3 / GCC 15 / sm_120 Release
configuration with conversation tests enabled. SSH as `root@nibbler`; only the
Qwen backend may be unloaded/reloaded. Do not start inactive `strata.service`,
restart TTS/router or overwrite shared settings. Preserve the user's pre-existing
local `deploy/nibbler/config.json` host-allowlist modification.

## Completed checks so far (2026-10-05)

- Actual nibbler CUDA 13.3 / GCC 15 Release engine build passed, including the
  final INFO advertisement derived from the policy's headroom constant. Build
  log: `incremental-build-final.log`. Two existing unused-code warnings remain.
- **81/81 fixture-independent CTests passed** in 100.46 seconds, including the
  rolling policy, shared streaming FP16/INT8/Q4 regressions and snapshot tests.
  Log: `incremental-ctest.log`. The three excluded tests (`ple_parity`,
  `expert_parity`, `pool_test`) need model fixtures absent from this checkout;
  they are not counted as passes.
- Rolling policy: **28 checks**, also passed with ASan/UBSan locally. Native
  helper additionally reports allocator 135660, cache 4191 and memory 23 checks.
  Its CUDA-declaration/transfer doubles are host tests, not GPU evidence.
- Frontend: **329 tests, seven skipped, no failures**, using
  `uv run --no-project --with jinja2 --with regex --with jsonschema python -m unittest discover -s serve -t . -v`.
  Final rerun: **329 tests in 119.46 seconds, seven skipped**, no failures;
  `/tmp/strata-incremental-frontend-suite-final-complete.log`. 18 new incremental tests
  include stable sampling/replay, cancellation and zero-progress deferral.
- Independent final source audit: no Blocker/High findings. One Medium INFO
  constant-drift issue fixed and verified. The reviewer's 57 relevant Python
  tests and 28 policy checks passed. Nine broader failures in its different
  environment also reproduced against the reverted baseline; the prescribed
  dependency-complete frontend suite above passed.
- During private validation, production binary/Python remained unchanged. Only
  Qwen was unloaded via llama-swap for exclusive GPU tests; pocket-tts and the
  router stayed running. Final rollout is recorded below.

## Actual-model results and remaining gates

All artifacts below live in the evidence directory, with adjacent stderr,
source hashes and numeric `.status` files. These are actual configured-model
runs on nibbler, not scripted subprocesses:

- `incremental-private-all.json`: **37 stages passed**, clean QUIT, approximately
  401.8 seconds. Includes concurrent whole-context-sized output allowances,
  renewed backing after the 256-cell headroom, COW tail offsets 1/2/3, actual
  1024-cell pool exhaustion with both FIFO admission orders, token-by-token solo
  parity, abandoned pressure continuation, protected BYIELD cancellation with
  no active decoder and healthy same-slot reuse. Parking disabled.
- `incremental-sampled.json`: **passed**, temperature **0.3**, seed **12345**.
  Both logical requests pressure-stop and re-admit with byte-identical token
  continuations to their solo references in both admission orders. No RNG offset.
- `incremental-tiny.json`: **passed**, one-MiB parking budget. Both owners
  pressure-stop, make forward progress and complete with solo parity; positive
  replay work and absence of restores prove optional-cache fallback.
- `incremental-coherence.json`: **passed**, 4096-MiB parking, actual **MTP max 2**
  and positive draft offers in the original solo references. The independent
  pressure probe restores **504 target-only tokens**, logs target-only restore
  and T=1 decode suppression, reports **zero restored-main draft offers**, and
  completes with concatenated solo parity. Cold unrelated work first reclaims
  idle backing so this is a real restore, not an expectation that every
  capacity-blocked FIFO re-admission must materialize its optional image.
- Smoke harness CPU suite: **62 tests passed** on Python 3.14.7 and 3.11.2;
  existing 39 native-harness and 27 streaming-harness tests also passed. New
  real HTTP/1.0 short/multichunk Content-Length tests reproduce and fix the
  operator client's closed-socket EBADF path.

Private HTTP startup initially failed because its isolated package lacked the
existing tokenizer module; operator `PYTHONPATH` now points to the unchanged
production tools directory. The first HTTP client attempt then hit EBADF before
its initial /props result: its saved HTTP/1.0 socket was closed automatically
when read1 consumed Content-Length, but the next iteration tried settimeout on
it. This is an operator-harness defect, not a native/server pressure pass. The
failed artifacts remain preserved; the closed-FD fix/regressions passed, and
HTTP retries use fresh evidence.

`incremental-large-stream.json`: **passed**, clean QUIT, approximately 192.3
seconds. 35000/35004-cell prompts exceed GPU residency, with aggregate backing
262144, residency 32768 and prefill 1024. Both whole-context allowances overlap
and emit before either completes, cancel after at least 384 outputs, and match
their solo references. Both consumed histories grow beyond initial headroom.
Positive RAM-read counters are not claimed as direct CLOCK-eviction telemetry.

## Supplemental cancellation audit

A targeted follow-up found a latent double-ack race: unified BSTOP unconditionally
emitted BDONE even after pressure had already reset its slot. A terminal ack in
transit could let that second cancel reach a freshly admitted successor. The
initial audit did not cover this timing. Reviewer severity: Medium, High under
adversarial transport timing; rollout blocked until fixed and revalidated.

The native fix now acknowledges only **active, protected-partial or live-admit**
owners; expired/idle BSTOP is a no-op. Live-admit STOP latching remains intact,
legacy behavior unchanged. Eight flag combinations plus ownership cases raise
the policy test to **40 checks**, with normal/sanitizer and actual nibbler build
passes. No inference/reservation math changed. Candidate SHA256 is
`9439ca51d664b72470c94e1793cc91a1a92e651f11abe4068aba3ee0f9db8324`;
`incremental-ack-fix.build.log`. Earlier numerical model gates above used
`1bf8fcf3...`; final lifecycle checks use the corrected candidate.

- Final **81/81 CTests passed** in **99.93 seconds**:
  `incremental-ack-fix.ctest.log/.status`.
- Actual-model negative control reproduced the defect on the saved pre-fix
  candidate: `incremental-late-stop.negative.json` records orphan
  `BDONE 1 0 cancel 0.0` after pressure completion and ordered late BSTOP plus
  immediate same-slot BGEN.
- Corrected candidate **passed all five late-stop stages**:
  `incremental-late-stop.json`. Two solo references, actual-pressure pair,
  late stop after pressure and late stop after natural completion. Healthy
  replacements match solo output and receive exactly one terminal; expired
  owners receive no extra acknowledgement, including during cleanup.
- Independent final lifecycle source audit **approved controlled rollout**:
  no Blocker/High or outstanding actionable lifecycle finding. Reviewer
  independently checked fixed source, 40 normal/sanitized policy checks and
  44 frontend lifecycle checks. Binary/CTest/GPU conclusions are explicitly
  based on the primary's reported evidence, not reviewer-executed GPU tests.
  `incremental-final-audit.approved` records the approval; all remaining actual
  HTTP/vision gates are still required before installation.

The private small-pool HTTP attempt stopped before model requests because the
sharp tokenizer's default terse system block makes its unpadded prompt exceed
160 tokens. `incremental-http-pressure.fixture-base-failure.json` retains the
failure; it is not pressure evidence. The frozen HTTP fixture uses a private
2048-cell pool, count-endpoint targets 400/404 and finite 1632-token completions,
leaving production templates untouched and preserving the API's eight-token
context margin. Cap-1 completions calibrate actual prompt usage, with bounded
count-endpoint deltas and an independent actual-prompt + target + slack fit
check. Full solo/SSE usage and native history identities must match the
calibrated actual counts exactly; live admitted segment counts additionally
include bounded bootstrap tokens. Harness SHA256:
`a50dcc5ae58dfef7ea864b1e4b8b2c3436bcc8f075c4b8e4e698f1eb9cb2f56b`.
Both initial reservations fit, each individual history fits, and combined
completed histories exceed backing. The first resized attempt calibrated
400/404-token prompts successfully but its first solo reference naturally
stopped at **292 output tokens**, not the required 1632. The strict harness
failed before concurrent pressure work; preserved evidence is
`incremental-http-pressure.natural-eos-failure.json`. The fixture now asks for
all integers **1 through 1000**, not an unrealistic 100000-line reply. An
independent actual-model fixture-discovery probe reached **1632 outputs** with
`length` and 400 prompt tokens: `incremental-finite-fixture.json`. That probe
is fixture evidence only, not a KV/pressure pass. The final harness retains
complete attempted solo responses even on failure; EOS and pressure assertions
are unchanged. The bounded-fixture retry completed both **1632-token** SSE
streams, matching every solo text prefix and final usage exactly, but its
strict live-count matcher never submitted the required third request:
`incremental-http-pressure.live-bootstrap-failure.json`. Live segment prompt
counts were **402/404**, although immutable calibration/final usage was
**400/404**: the initial main-to-slot handoff folded two emitted tokens into
its admission prompt. The corrected live-overlap check records bootstrap
deltas and requires each to be nonnegative and no greater than both 16 and
generated output. A scripted 400/404-original, 402/404-live regression now
submits the third waiter and validates its queue proof. Exact calibration,
final SSE and completed history identities remain required. This is not reported as a complete pressure/third-waiter gate pass.
Native 1024-cell pressure evidence above remains separate and unchanged.

- `incremental-private-http.json`: **passed** on the final binary and frontend,
  with unchanged production settings (262144 backing, 32768 residency,
  4096-MiB optional parking, adaptive experts and GPU vision). No output-limit
  keys in either large request. Both histories **146002 / 22001 tokens** were
  observed decoding simultaneously; outputs had reached **15 / 1034 tokens**
  at the overlap sample. Early client close stopped both workers; cached and
  fresh healthy follow-ups succeeded. This proves API overlap/cancel/recovery,
  not full 146K-context output parity or simultaneous maximum-length completion.
- `incremental-private-vision.json`: **passed**. The production-settings private
  endpoint answered **Red** for a synthetic solid-red 64x64 PNG; live properties
  confirmed two slots, 262144 backing and vision capability. The isolated
  endpoint then shut down cleanly.

## Final HTTP pressure gate and rollout

`incremental-http-pressure.json`: **passed** on the final native/frontend,
private 2048-cell backing, resident=0, parking=1 MiB, frozen experts and default
production template. Calibrated original prompt counts were **400/404**, with
zero count-endpoint deltas. Both SSE streams completed **1632 tokens**, every
prefix and final text matching their solo reference. Completed history recorded
**three / four pressure pauses**, exact original `prompt_total`, exact aggregate
`engine_generated=1632`, and final `length`, never a public pressure finish.
The third cap-16 request was submitted before pressure while both originals
decoded, was **observed queued**, and answered **HEALTHY** like its solo reference.
Recovery matched, and all **three workers stopped**. The private endpoint then
shut down cleanly. Earlier failed fixtures/matcher artifacts remain preserved
and are not substituted for this final success.

Controlled rollout: **passed**, `incremental-rollout.status=0`. Only Qwen was
unloaded/reloaded through the already-running llama-swap router. Atomic binary
and Python replacements retained the baseline rollback copies. A request
through port **8088** answered **OK**, and live backend properties on **5801**
confirmed two slots, 262144 logical/aggregate backing, 32768 residency, 4096-MiB
optional parking, vision, and `kv_incremental=1 kv_reserve_ahead=256`.
Both Qwen and pocket-tts remained ready; `llama-swap.service` remained active and
standalone `strata.service` remained inactive. No router or TTS restart.

Final production SHA256 (`incremental-rollout.sha256`):

- `engine/strata-nibbler`:
  `9439ca51d664b72470c94e1793cc91a1a92e651f11abe4068aba3ee0f9db8324`.
- `serve/server.py`:
  `32d45cccf7a82ab9a1c303624ce581cea8eb6d1901d8717de99374e33ac4a414`.
- Both tracked and runtime config, **unchanged**:
  `e806bfeb934c1fe7350ec305d49af66ed8089bbeb8d5198b032967e236b38166`.
- Shared Chat settings, **unchanged**:
  `b61e757d0d94ca1f1c7c17cf180f98e779cbd976d53658a88ae91d870d9c3e2a`.

Rollback, if needed: run
`bash /data/llm/Strata-tests/multislot-20261005/rollback-incremental.sh` on nibbler.
It restores only `incremental-baseline.strata` and
`incremental-baseline.server.py`, unloads/reloads only Qwen and leaves configs,
shared settings, router and TTS untouched. Copies and script syntax were checked;
rollback was not executed after the successful rollout.

All required gates are complete. The continuation heartbeat can be removed.
