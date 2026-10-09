# Incremental unified KV work log

Status: **validated on isolated hardware; this branch is the fork's working main line**.

**Branch layout (fork `peterkeen/Strata`).** `main` is pinned to upstream Strata
(`Niko1221/Strata`) and is not developed in this fork: it is the reference the
fork rebases onto. All fork development happens on **`kv-unified`**, which is the
fork's main line and carries three change sets together: the unified-KV port, the
NVFP4 kernels and expert cache, and the opt-in unified batch-MTP work (below).
Feature branches under `feature/*` are historical. No pull requests are opened
from this fork; changes land on `kv-unified`.

Port base: upstream main `fb58e0db` (following the 0.1.40.x lineage;
`CMakeLists.txt` identifies this base and the port as 0.1.41).

Historical status: **fork validated and deployed on nibbler, 2026-10-05**.
Historical branch: `feature/incremental-unified-kv`, based on fork main `cc82936`
(the merged admission fix `d9f737a`, PR #3); later fork follow-ups are dated below.
All measurements, test counts, source/binary hashes and rollout statements below
belong to that fork lineage and the dates recorded, not to the upstream port.
The work log is preserved as historical evidence; old source line numbers and
operator/rollback instructions are not current port validation or authorization
for live changes. See the [ported deployment candidate](../deploy/nibbler/README.md)
and its [source/config check](../deploy/nibbler/validation.md) for current status.

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

All required gates are complete. Continuation heartbeat `d12a6ef6` was removed.
Implementation commit: `3d825e9`. Published for review in
[PR #4](https://github.com/peterkeen/Strata/pull/4); deployment does not imply the
PR has been merged. The user's local host-allowlist config change is not part
of the commit.

## Warm-cache handoff follow-up (2026-10-05)

Status: **all gates passed; deployed to nibbler (Qwen only)**.

Live inspection found full rereads of 108434, 106216 and 63513 tokens despite
successful restores between requests. The frontend used cancellation commands
for internal solo/batch transitions, while unified cancellation intentionally
released their backing. The original gates did not exercise repeated warm
continuations across those transitions.

The candidate advertises `kv_handoff=1`. Capable frontends use `HANDOFF` for
solo-to-batch and `BHANDOFF <slot>` for batch-to-solo, with an internal `handoff`
terminal. Valid completed-prefill state becomes a trimmed, reclaimable idle
cache. Actual `STOP`/`BSTOP` retains release semantics; partial prompt reads
cannot become transferred live caches. Late slot handoffs owe no duplicate ack.
Target-only slot clone-back still suppresses private MTP/suffix proposals until
full residual replay; neither the draft ring nor capacity settings change.
Legacy engines keep their old commands.

Evidence and rollback copies are under
`/data/llm/Strata-tests/handoff-20261005/`: `baseline.strata` and
`baseline.server.py`. CPU reservation/handoff policy: **55 checks passed**;
harness regression suites: **136 tests passed**; frontend suite: **340 tests,
seven skipped**, using an isolated `uv` environment with regex/jsonschema/jinja2.
The initial bare-Python run lacked regex/jsonschema and failed; that environment
failure is preserved, not counted as validation. The CUDA candidate built with
the existing unused-code warnings.

- `ctest-final-exclusive.log`: **81/81 passed**, exclusive GPU, 100.62 seconds.
  Earlier overlapping-production runs hit `cudaMalloc: out of memory` in
  `prefill_fused_iq_test`; the exclusive reruns resolve that failure. The three
  asset-dependent tests (`ple_parity`, `expert_parity`, `pool_test`) remain
  excluded, as in the previous rollout.
- `warm-small-v2.json`: **passed**, parking disabled, 36 stages. Three repeated
  handoff cycles, both admission orders, both next turns, partial-page offsets
  1/2/3, late BHANDOFF no-extra-ack and real STOP/BSTOP recovery passed.
- `warm-streamed-v2.json`: **passed**, parking disabled, 35000/35004-token
  histories, 262144 backing and 32768 residency. Every warm pair admission and
  main/batch continuation reused its entire required consumed prefix. Final
  next-turn admissions reused **35095 / 35058 cells** exactly. Each run emitted
  one main handoff and six slot handoffs with exactly matching acknowledgements;
  all logical continuation outputs matched their frozen solo references.
  Next-turn reference requests are independent but may reuse checkpoints; they
  are not claimed to be fresh full-replay oracles. Both native processes quit
  cleanly.
- The first `warm-streamed.json` failed: A's finite 64-token allowance completed
  during B's cold 35K prefill, so it could not prove overlap. The corrected gate
  prewarms B, keeps logical cache placement stable while reversing admission
  order, and returns A's resumed main prefix to its idle slot before admitting
  B. Caps and overlap/reuse assertions are not weakened. Failed evidence remains.
- Independent review by `1f9b6897-b4af-4054-9615-2f1b67a68cda`: **no correctness
  blockers**. Explicit integer stop stores/conversion address readability notes;
  two additional frontend tests cover STOP upgrading a pending HANDOFF and
  promotion never downgrading real cancellation.
- `cancellation-pressure-v2.json`: **passed**, actual small-backing pressure,
  FIFO continuation, parity, cancellation, protected BYIELD release and healthy
  recovery.

- `coherence-v2.json`: **passed**, target-only MTP coherence including
  pressure restore to main GEN; `native-v2.exit=0`.
- `private-http.json`: **passed**, actual frontend integration on an isolated
  5802 endpoint with the candidate binary and server.
- `private-vision.json`: **passed**, synthetic solid-red PNG.
- `private-http-warm.json`: **passed**, parking disabled
  (`conversation_cache_mib=0`), two ~35K-token histories over three streamed
  multi-turn rounds. Both streams decoded concurrently every round, with no
  pressure pauses. Rounds 1-2 reused all but the ~43 re-templated tail tokens
  of each prior prompt (e.g. 35083/35126, 35551/35594; native logs and history
  agree). Before this fix such turns reread the whole history.
- After each gate script, production recovery answered **OK** and production
  binary/server/config/shared-settings hashes were unchanged.

### Rollout

Controlled rollout: **passed**, `rollout.status=0` (`deploy-handoff.sh`). The
script re-checked every gate artifact, refused to run with the private
endpoint up or production busy, verified baseline hashes, froze the tested
artifacts as `candidate.strata` / `candidate.server.py`, then unloaded only Qwen
through llama-swap and atomically replaced the server, then the binary. A
request through **8088** answered **OK**; live 5801 properties confirmed two
slots, 262144 backing, 32768 residency, 4096-MiB parking, vision and
`kv_handoff=1 kv_incremental=1 kv_reserve_ahead=256`. pocket-tts stayed ready
and `llama-swap.service` stayed active.

Production SHA256 (`rollout.sha256`):

- `engine/strata-nibbler`:
  `d318cd7baf926c2b1a3d3eebfd94a615d5e9fe909fd560dcd556d61174771cac`.
- `serve/server.py`:
  `262f00a67b7056b3e2d7f2e0192dd4f923f865ea55f64cf9aa84a0fff0e8bb26`.
- Tracked and runtime config, **unchanged**:
  `e806bfeb934c1fe7350ec305d49af66ed8089bbeb8d5198b032967e236b38166`.
- Shared Chat settings, **unchanged**:
  `b61e757d0d94ca1f1c7c17cf180f98e779cbd976d53658a88ae91d870d9c3e2a`.

Rollback, if needed: `bash /data/llm/Strata-tests/handoff-20261005/rollback-handoff.sh`
on nibbler. It restores only `baseline.strata` / `baseline.server.py` (the
incremental-KV release, hashes checked), reloads only Qwen and leaves configs,
shared settings, router and TTS untouched. Not executed.

The deployed native binary was built from source that differs from the
committed `generate.cpp` only in the three review readability edits (explicit
`store(1)`/`store(0)` and `!= 0`), which are semantically identical. Heartbeat
`27e8220b` removed after rollout. The user's config edit is not committed.

## Deployment re-verification and follow-up (2026-10-06)

Every production hash from the rollout record was re-checked from a root shell on
nibbler and matched: `engine/strata-nibbler` `d318cd7b…`, `serve/server.py`
`262f00a6…` (byte-identical to `3e98524`), the runtime config `e806bfeb…`, and the
shared Chat settings `b61e757d…`. The host's `src/program/generate.cpp` still
hashed to the frozen `1339145e…`; diffing it against `3e98524` gave exactly the
three readability hunks, and a file-by-file hash of all 388 tracked source/config
files found nothing else except the config and three stale frontend test files.

Live effect of the handoff, from the running production log (11 h window, 0 `ERR`
lines): 2,777 requests; prompt reuse 180,131/181,577, 182,654/185,136 and
186,211/186,819 tokens; 169 `TARGET_ONLY slot clone` and 293 `TARGET_ONLY decode`;
drafts ran on 2,260 of 2,777 requests at 67.7% acceptance (1,382,206 of 2,041,742
offered), so target-only clone-back suppresses drafts on a minority of requests.
Parking took 22 pressure events, all parked, no budget/RAM-floor/allocation miss,
but each parked target-only image measured 2.0-2.4 GiB, so the 4096 MiB budget
held `parked=1` and a second pressure event evicted the first. `--conversation-cache-mib`
is raised to 8192 for that measurement; the four-entry cap and the 4096 MiB
available-RAM floor are unchanged.

Three deployment gaps fixed here (see [the nibbler deployment record](../deploy/nibbler/validation.md)):
the live host-allowlist entry had been committed nowhere; the host checkout sat at
`00a7289f` with the deployment as uncommitted edits, whose stale
`serve/test_unified_lifecycle.py` errored 4 of 15 against the deployed frontend
(`blocked_resume() got an unexpected keyword argument 'max_new'`) while
`serve/test_kv_handoff.py`/`serve/test_incremental_kv.py` were absent; and a dead
`strata.service` still pointed at the pre-unified-KV stack on port 8088.

Three serving-path defects from the same audit, fixed here:

- `finish_shared_slot` acknowledged `handoff` even when `keep` was false and it had
  released the backing. `serve/server.py` reads "handoff" as "this slot still holds its
  KV" - it writes `slot_held` from it and `pick_slot` steers the next turn's `BGEN` at
  that slot - so the frontend advertised a prefix that no longer existed. The terminal is
  now `cancel` whenever the cache cannot be kept (a picture slot, or prompt caching off),
  with the reason on stderr. `serve/test_kv_handoff.py`'s
  `test_engine_declining_to_keep_a_handed_off_slot_advertises_no_prefix` pins the
  frontend half against a scripted engine that declines to keep.
- `BSTOP`/`BHANDOFF` took their slot from a bare `atoi`, so `BSTOP foo` parsed to 0 and
  cancelled or handed off somebody else's owner. All three read sites - the main control
  loop, the admission retry loop, and the chunked prompt read's window boundary - go
  through `strata::core::shared_kv_slot_arg`: the whole argument field must be a decimal
  inside `[0, --batch)`. A rejected line is logged and ignored rather than answered with
  `ERR`, because an unsolicited `ERR` is consumed by whichever request owns the slot
  queue and ends its turn. `src/core/shared_kv_reservation_test.cpp` covers garbage, a
  missing field, negatives, out-of-range, trailing junk and `strtol` overflow.
- A pending internal `HANDOFF` relabelled a reservation-shortage stop as a handoff, so
  the one condition that needs the pressure disposition skipped it: no RAM snapshot
  parked, and main's mapping retained while nothing was decoding. The relabel is gated on
  the stop having come from the handoff, and a stop that releases main because the next
  window would not fit parks its image first whatever label it carries. Capacity was
  never stranded either way: `reclaim_shared` frees seq 0 once `main_required_end < 0`,
  which every request end sets.

`copy_to_slot`/`copy_from_slot` also `return`ed inside their per-stage loop, so a layer
split would transfer only stage 0's sessions. `--kv-unified` is refused with
`--layer-split` at startup, so this was latent; both now fail the transfer with an
explicit error if `stages` is ever non-empty.

Verification: `shared_kv_reservation_test` 73 checks (was 55) over the new slot-argument
matrix; `serve/test_kv_handoff.py` 12 tests (was 11), including a scripted engine that
declines to keep a handed-off slot; the frontend suites unchanged at 86 tests (18
incremental-KV, 26 unified lifecycle, 13 unified reporting, 11 parallel, 8 lifecycle,
10 monitor); `shared_kv_pages_test` (135,660) and `conversation_cache_test` (4,191) still
pass. `generate.cpp` cannot be compiled in this VM (no CUDA toolkit), so the engine half
was compile-checked on the GPU host in a throwaway worktree (`/data/llm/Strata-check`,
Ninja Release, CUDA 13.3, arch 120, g++-15 - the deployment cache's own settings): the
changed TU builds clean, with only the two pre-existing `-Wunused` warnings (1023
`argmax`, 6421 `w`). The full binary was linked and the GPU gates and deployment
followed on 2026-10-06 (below); the "still runs the audited binary" note written before
that build named `9319b4a7…`, which was a transcription error for the pre-fix
`d318cd7b…`.

### Deployment (2026-10-06, serving-path fixes)

Built on nibbler in the deployment's own build directory (`/data/llm/Strata/build`,
Ninja Release, CUDA 13.3, arch 120, g++-15, `STRATA_BUILD_CONVERSATION_TESTS=ON`,
`cmake --build build -j 4`, 33 steps, exit 0). Only the two pre-existing `-Wunused`
warnings remain (1023 `argmax`, 6421 `w`). Candidate `build/strata`:
`be37cae8df1cb73879b7aace36dc6bd4709d38bfc86f4d9d6abb0ad0c5aa4ef3` from this commit's
`generate.cpp` (`bece8f5b…`, no readability hunks - the three-hunk variant was only in
the previous build). The host checkout moved to this commit as branch `nibbler-deployed`
with a clean tree; `serve/server.py` is untouched by these commits, so the frontend did
not change.

Exclusive-GPU gates (`/data/llm/Strata-tests/servingfix-20261006/`, production unloaded
for the whole window): `ctest-final-exclusive.log` **81/81 passed** in 100.61 s
(`-E '^(ple_parity|expert_parity|pool_test)$'`, as in the previous rollout);
`warm-small-v2.json`, `warm-streamed-v2.json` (35000/35004-token histories, 3 cycles),
`cancellation-pressure-v2.json`, `coherence-v2.json` (pressure restore, 4096-MiB parking),
`private-http.json`, `private-vision.json` and `private-http-warm.json` all
`passed: true`. Every gate restored production hash-identical (the
`*.production-unchanged.log` files).

Rollout `deploy-servingfix.sh` re-checked the artifacts and installed them after
unloading only Qwen: `engine/strata-nibbler` `be37cae8…`, `serve/server.py`
`262f00a6…` (unchanged), tracked and runtime config `8178be74…` and shared Chat
settings `b61e757d…` byte-identical before and after. Live checks: two slots, 262144
backing, 32768 residency, vision, `kv_handoff=1 kv_incremental=1 kv_reserve_ahead=256`,
`conversation_cache_mib=8192`, `/health` 200, `Host: nibbler.local.keen.land` 200 and an
unknown host 403, llama-swap active, pocket-tts ready. The host's own frontend suites
pass against the deployed source: 12 `test_kv_handoff` (the deployed source now carries
the test added here), 18 incremental-KV, 26 unified lifecycle, 13 unified reporting, 11
parallel, 8 lifecycle, 10 monitor. A two-turn chat through 8088 reused 212 of 240 prompt
tokens on the second turn.

Rollback: `bash /data/llm/Strata-tests/servingfix-20261006/rollback-handoff.sh` restores
`baseline.strata` (`d318cd7b…`, the pre-fix binary) and `baseline.server.py` after
hash checks, reloads only Qwen, and leaves configs, shared settings, router and TTS
untouched. Not executed.

The shortage-parking cost of the third fix is still unmeasured: the only parked event
observed after the reload parked one 227-MiB image with zero evictions, so production
parking has still never fallen back to replay.

## Two-stream aggregate-capacity probe (2026-10-06)

Purpose: measure what the 262,144-cell aggregate backing actually does when two live
conversations together exceed it, and where the cost lands. Run against the live
production stack (llama-swap 8088 - frontend 5801) with bounded 32-token outputs.
Artifacts: `/data/llm/Strata-tests/overcap-20261006/` (both phases, engine-log byte
offsets per phase, 5-s `/metrics` timeline) and
`/data/llm/Strata-tests/overcap-clean-20261006/` (phase 1 after an unload/reload, so the
pool started with both slots at zero). This is a correctness/shape probe, not a
throughput benchmark: one seed, greedy, repetitive synthetic padding.

### Phase 1 - two concurrent 125k streams (249,996 cells <= 262,144)

Clean pool (fresh engine, both slots `held_tokens 0`, no parked entries). Both requests
were admitted, produced their 32 tokens, and no request returned an error. Two things
did not go as the arithmetic suggests:

- **The prefills never overlapped.** The first stream read 125,001 tokens in 67,510 ms
  (1,851.6 tok/s); the second waited 137 s and then read 125,001 tokens in 68,728 ms
  (1,818.8 tok/s). Both runs with an occupied pool behaved the same way, so this is not
  a pool-occupancy effect. The engine did interleave at coarse granularity instead:
  `strata batch: the prompt was read in 2 parts, the slots decoding 1075 ms between
  them`.
- **Parking ran even though 250k fits.** `conversation cache: parked 125002 tokens in
  588.3 ms; parked=1 bytes=2737017848 evictions=0`, then
  `strata batch: slot 0 takes 125002 tokens (copied in 643.7 ms)` and
  `prompt 125002 tokens = 125001 reused + 1 read in 1 ms (720.9 tok/s)`. So the handoff
  parks an image (2.74 GB) and the continuation restores the prefix by copying it back.
  Zero evictions, zero `pressure` lines, zero `ERR` lines.

End state confirms the boundary claim: both slots at `held_tokens 125032` - 250,064
cells, inside the 262,144-cell pool, with `parked=2 bytes=5474035692 evictions=0
parks=2`. Two 125k histories are simultaneously resident; the parked image is durability
for the handoff, not a capacity necessity.

### Phase 2 - both grow to 150,002 tokens (300,004 cells > 262,144)

Both growth turns were submitted together. Two 150k histories cannot co-reside, and the
engine resolved it without any error, any `pressure` event or any cancellation:

- Winner: `prompt 150002 tokens = 125029 reused + 24973 read in 15968 ms`, then
  `parked 150003 tokens ... evictions=4`, then `prompt 150003 tokens = 150002 reused +
  1 read in 1 ms`. Client wall time 17.4 s; frontend usage reports `cached_tokens`
  150002.
- Loser: `conversation cache: skip restore (private image exceeds available shared
  capacity); keep live/slot prefix or prompt replay`, then a full
  `prompt 150002 tokens = 0 reused + 150002 read in 82527 ms (1817.6 tok/s)`. Client wall
  time 100.5 s; frontend record `reused=0 prompt_read=150002 prompt_ms=82527.4`.

The whole window has 4 park/evict events, one skip-restore, 0 `pressure`, 0 `cancelled`,
0 `ERR`. The loser's penalty is a complete re-prefill of its history (~82 s for 150k,
~68 s for 125k tokens at the measured ~1.82-1.85k tok/s), against ~0.64-0.71 s to copy a
125k prefix back out of a parked image. The parking budget is what makes the winner's
image survive at all: `parked=3 bytes=8343518772` and, in phase 1,
`bytes=8568791496` - 7.77-7.98 GiB, i.e. inside the raised 8192 MiB budget and roughly
twice the previous 4096 MiB one.

### Mechanism (code, not inference)

- The skip is decided at `src/program/generate.cpp:6866` by
  `page_count(restore_end) > pages().free_pages_after_release(0)`, where
  `free_pages_after_release` (`include/strata/core/shared_kv_pages.hpp:160`) counts free
  pages plus pages **exclusively owned by that one sequence**. The check therefore
  ignores the other slot's claim, even when that slot is idle and its image is already
  parked. When it fails, `incoming.reset()` drops the canonical image and the request
  falls back to "keep live/slot prefix or prompt replay"; in this run the loser's live
  prefix had already been released to admit the winner, so the fallback was a full
  replay, and the image that could have served it was discarded.
- `reclaim_shared` (`src/program/generate.cpp:5982`) *does* release idle slots'
  mappings (and clears their `cached`/`ids`/`checks`), but it is only reached from the
  pressure path (`:6223`, `:6455`, `:6479`), never from the restore preflight.
- The comment at the skip site states the intent: a fresh full restore must not evict a
  slot source when the live/checkpoint prefix can fit instead. The assumption breaks when
  the incoming request's own prefix is already gone.

Inference, labelled as such: with both images parked (5.47 GB of an 8 GiB budget), an
idle-slot-reclaiming restore would turn an ~82 s replay into a ~1 s copy plus the
suppressed-MTP cost, so the replay here is a policy cost rather than a physical limit.
That trade is implemented and measured in *Reclaiming a busy or idle claim instead of
replaying*, below.

### Side observations

- **`held_tokens` is a logical claim, not residency.** At the end of phase 2 both slots
  reported 150,014 (sum 300,028 > `kv_capacity_cells` 262,144), and during the loser's
  replay the idle slot reported 150,014 throughout. Do not sum these as physical use.
- **The frontend's `cached_tokens` is not a cost signal after a handoff.** Phase-1 stream
  a reports `cached_tokens=124998` although its first segment read all 124,998 tokens
  cold; the number comes from the warm continuation segment.
- **Handed-off requests under-report in the per-request summary line.** Without
  competition the same request shape logs the true count (`427 tokens ... 2 generated`,
  `40003 tokens ... 32 generated, drafts accepted 23 of 23`); the handed-off streams log
  `... 1 generated, drafts accepted 0 of 0` per segment while the bulk of the decode
  appears in `strata batch: 30 windows ... 5.9 rows/s` / `31 windows ... 29.3 rows/s`,
  and the clients received 32 completion tokens each. Use client `usage` or the
  batch-window lines for token accounting.
- MTP is suppressed across the handoff (`TARGET_ONLY slot clone` /
  `TARGET_ONLY decode: T=1`), consistent with the coherence rule, so a restored turn
  drafts nothing until a full replay rebuilds history.

- KV-growing pressure is visible as a lower streaming hit rate: 85.8-86.7% of block
  reads hit VRAM during phase 2 against 97.9-98.0% in light traffic, with 326-648 MiB
  read from RAM over the window. VRAM free stayed at 1,674 MiB throughout.

## Reclaiming a claim instead of replaying (2026-10-06)

The probe above showed the cost, and this change removes most of it. Three parts:

1. `SharedKvPages::exclusive_pages(seq)` (`include/strata/core/shared_kv_pages.hpp`) reports
   the pages one sequence holds alone; `free_pages_after_release` now composes it, so a
   reclaim candidate's capacity is explicit. CPU-covered in `shared_kv_pages_test`
   (135,708 checks, was 135,660).
2. The canonical-restore preflight (`src/program/generate.cpp`, before the skip line) now
   reclaims idle slot claims whose parked image covers all but at most 4,096 tokens
   (~2.2 s of re-read at the measured 1.82k tok/s prefill), cheapest uncovered tail first.
   The victim's image stays in the parking cache, so its next turn restores a copy rather
   than re-reading its history. A picture slot, a partly-read prompt, the slot this request
   continues in, and any claim sharing pages with another live owner are never taken. Whole
   coverage is not required and never holds in practice: images are parked at segment
   boundaries, so a slot always holds a few tokens more than its image did
   (`parked 150001` against `held 150002`).
3. When the gap is held by a slot that is still *decoding*, its claim cannot be taken. The
   request is now handed back as an incremental pressure continuation with its image
   re-parked, and the retry takes the reclaim path. `serve/server.py`'s
   `_wait_pressure_progress` already retries a zero-token pressure reply only after the busy
   owner produces another token, so this neither spins nor stalls the active decode.
   Deferring is gated on the economics, because `coherence-v2` caught the unguarded version
   deferring a 1,015-token restore in its small-backing FIFO stage ("another slot holds 104
   pages") and returning a zero-token pressure continuation where the harness expects the
   engine to proceed: accept a busy claim only when the avoided replay is at least 8,192
   tokens and the busy owner's remaining output allowance R satisfies `R * 45 <=
   restore_end` (the measured ~1.82k tok/s prefill against ~40 tok/s decode), and never when
   that allowance is unbounded. Under that rule the small-pool stage keeps its old behaviour
   (0 deferrals, 0 reclaims in its log) and the 150k case still defers.
4. The skip line now carries its arithmetic, so a skip that could have been a copy is
   diagnosable from the log alone: `need N pages, have M; main K pages exclusive; slot b
   <state> held X mapped Y excl Z parked C [slot_source] [admit_slot]`.

### Measured (private endpoint, production-equivalent settings, 8192 MiB parking)

Three phases, candidate `1d71f758…`: two 125k conversations, then both grown to 150k
(300,004 cells), then both to 170k (340,003 cells).

| Phase | Allocation | Outcome |
| --- | --- | --- |
| 1 | 2 x 125,002 | prefills still serialize (73.7 s and 139.2 s wall), 250k cells co-resident |
| 2 | 2 x 150,002 | `deferring a 150002-token restore (another slot holds 37501 pages while its turn finishes; 1 so far)`, then `reclaimed idle slot 1 (37504 pages)`, then `prompt 150002 tokens = 125003 reused + 24999 read in 16137 ms` |
| 3 | 2 x 170,001 | two reclaims (`slot 0` 37504 pages, `slot 1` 42504 pages), no deferral; both streams warm (27.6 s and 14.1 s wall) |

Counts over the run: 0 `skip restore`, 1 deferral, 3 reclaims, 0 `ERR`, 0 cancellations.
The phase-2 stream that replayed 150,000 tokens in 82,527 ms before this change restored
instead: 16,137 ms of read, 34.4 s wall against the recorded 100.5 s.

### Measured (live production, nibbler, deployed binary)

Same three phases against the live stack (llama-swap 8088, frontend 5801): 0 `skip
restore`, 1 deferral, 1 reclaim, 1 `restored … (live)`, 0 `ERR`. The phase-2 loser went from
the recorded 100.5 s wall / 82.5 s read to 34.1 s wall / 16.1 s read, and the phase-3
winner stayed warm (14.4 s).

The phase-3 partner, however, replayed 169,998 tokens in 93.9 s with no skip, deferral or
reclaim line at all: its parked image had been evicted (the window's evictions reached 3
before its turn) and its idle slot claim had already been released by the other stream's
pressure path, so there was nothing to restore from. That is the limitation below, not a
fault in the reclaim path: the same turn would have replayed before this change too.

### Limitation and next step

The reclaim depends on the victim's image still being in the parking cache. In an
uncontended cache it is (the private run reclaimed the same two conversations three times),
but with a third entry present the 8,192 MiB budget's LRU can evict exactly the image the
other conversation needs next. Candidate mitigations, none measured yet: pin the image of a
conversation whose slot claim was released, refuse to release a claim whose image is at eviction
risk, prefer dropping the same conversation's superseded copy over another conversation's only
copy, or raise the parking budget. Also unmeasured: the MTP-suppression cost of a restored
turn (restored turns log `drafts accepted 0 of 0`), and how long a deferred request waits
when the busy owner's turn is long (the gate admits a wait only when that turn's remaining
allowance is small relative to the replay).

### Getting the eviction order right (2026-10-06)

`drop_superseded` only fires when the outgoing chain holds a parked copy's deepest checkpoint
**exactly**. A client re-renders its last reply, so that chain's resume points sit a few tokens
past the copy's deepest checkpoint and the test misses: both probe runs logged 0 drops across 13
parks, `make_room` then evicted oldest-first, and the live 340,003-cell phase lost a
conversation's *only* 150k image - its partner replayed 169,998 tokens in 93.9 s with no skip,
deferral or reclaim line to explain it. The page-table arithmetic was never the problem; a copy of
that conversation existed when the other stream parked, and the cache chose the wrong entry to
release.

`make_room` now takes the incoming chain (`put` passes its live ids and image keys) and, when an
entry must go, gives up one that chain **covers** first: the copy's deepest checkpoint is a proper
prefix of the incoming live ids with the same images, and the copy's own tail beyond that point is
at most 4096 tokens. A superseded copy's tail is one turn's reply; a longer one is worth more than
the eviction it avoids and stays in the oldest-first order. `drop_superseded` is unchanged, so
nothing is dropped without eviction pressure - an eager version of the same rule broke two existing
tests (a duplicate that still carries a unique checkpoint, and "an oversized put drops nothing"),
which is what pointed at the eviction order instead. `conversation_cache_test` is 4,205 checks (was
4,191).

A large prompt with no parked match now logs why:
`conversation cache: no image for a 169999-token prompt (parked=1 bytes=5610502220 evictions=1; slot -1
holds 0)`. That line is how the phase-3 replay above was attributed to eviction rather than to an
unmatched image.

Measured on the private 3-phase probe (candidate `f5f8b42a…`): 0 `skip restore`, 1 deferral, 3
reclaims, 3 restores, 0 `ERR`. Phase 2's second stream restored (`150000 tokens = 125004 reused +
24996 read in 16036 ms`) and phase 3 kept both streams warm at the grown size (14.2 s and 27.2 s
wall, the latter with 9 pressure pauses while the other stream finished). The live contention case
that this change targets - a third small parked entry tipping the LRU - has not been re-run since
the fix.

Gate note: exclusive-GPU gates must wait for the GPU to be *actually* free after the model unload
(the unload acknowledges before the engine's VRAM returns). Without that wait
`prefill_fused_iq_test` fails with `cudaMalloc: out of memory` while nothing else is running;
`native-gates.sh` waits for under 2 GiB used before `ctest`.

#### The reservation, not just the insertion (same day)

The first rollout of the eviction preference (`f5f8b42a`) did **not** fix the live case: the
re-run reproduced the replay exactly (108.5 s wall, `no image for a 169999-token prompt
(parked=2 bytes=6250821344 evictions=3; slot -1 holds 0)`). The log explains it. `park_current`
and `park_slot_target` reserve room with `make_room(estimate, held)` **before** they capture,
and those calls had no incoming chain, so they still evicted oldest-first - the older copy of the
*other* conversation - and `put`'s chain-aware choice never ran because the room was already
made. Both reservation sites now pass the chain they are about to park (main's `live`/`live_imgs`,
the slot's `ids` with no images), so the choice is consistent from the reservation through the
capture. `conversation_cache_test` is 4,208 checks; its case pins the reservation path
directly.

With `40e8305c…` deployed, the same live three-phase probe logs 0 `skip restore`, 1 deferral,
**2 reclaims, 2 restores**, 0 `ERR`, and the phase that replayed in both earlier live runs is
now: `reclaimed idle slot 0 (42503 pages) for a canonical restore` → `restored 149999 tokens
(live) in 123.8 ms` → `prompt 169999 tokens = 149999 reused + 20000 read in 12979 ms` (27.7 s
wall against 108.5 s, with the other stream warm at 14.2 s). Both slots end idle at ~170k tokens
with 2 parked entries.
### Verification

`shared_kv_pages_test` 135,708 checks (was 135,660), `shared_kv_reservation_test` 73,
`conversation_cache_test` 4,205 (was 4,191). Exclusive-GPU gates on the frozen candidate:
`ctest` 81/81, `warm-small-v2`, `warm-streamed-v2`, `cancellation-pressure-v2`,
`coherence-v2`, `private-http`, `private-vision`, `private-http-warm` all passed, with
production restored hash-identical after each window. `coherence-v2` is the gate that
caught the unguarded deferral. The same gate set was re-run for the eviction-order change
(candidate `f5f8b42a…`): `ctest` 81/81 and all seven gates passed again, and `coherence-v2`
logs no `no image` lines.

## Unified batch-MTP on the incremental path (opt-in, 2026-10-09)

Status: **opt-in and measured on isolated hardware; off by default.** This port
adds no default behaviour: without `--batch-mtp` (or `STRATA_BATCH_MTP=1`) batch
windows carry no drafts and the solo path is unchanged. It is the batch mode
that also reserves the optional proposal row, so the optional-row fallback and the
target-only suppression rules below are part of this reservation policy, not a
separate scheduler.

### What it enables, and what it needs

`--batch-mtp` lets **each batch slot verify one MTP proposal per window**: the
slot's own private drafter proposes the token after its current one, and the
verifier commits both rows when they agree. Requirements (the engine warns
`--batch-mtp is off: ...` and batches as usual when one is missing):

- `--serve` with `--batch 2..8` (a window holds at most eight rows, which is also
  what `--kv-unified` requires), `--mtp`, and `--spec T` with `T >= 2`;
- **one GPU**: no `--layer-split` and no helper GPU (the draft ring is private to
  the slot's own device);
- `--kv-unified --batch-groups 1` for the measured configuration. Without unified
  KV a slot still offers one proposal per window, but there is no shared-pool
  reservation to preflight, so the `fallback_reserve` witness below does not apply.

Restrictions that remain: **one proposal per slot** (not a chain), **one GPU**,
and no repetition / frequency / presence penalties in batch windows.

### The private ring, and when the resident fallback applies

Each slot's drafter keeps a bounded private ring sized from the MTP window:

    mtp_kv_ring_cells(window, max_cells, max_t) = window + 4*max_t + 64   if 0 < window < max_cells
                                               = -1  (private, fully resident)  otherwise

`window` is `--mtp-window` (default 32768 cells), `max_cells` the slot's logical
context, and `max_t` the MTP window (`--spec`, capped by `--mtp-max-t`). A default
window is at least the context on a small pool, so `-1` applies and the draft K/V
is fully resident; a bounded ring needs an explicit smaller window. Measured
(RTX 5060 Ti, `--max-context 4096`, `--mtp-window 128`, `max_t = 2`):
128 + 8 + 64 = **200 cells** requested and 200 cells allocated, with 1536/1540-cell
prompts and 2048 consumed cells. That the run stayed on the bounded ring is
**geometry and inference, not a byte-level witness**: the helper's size was 200
cells, both histories were far longer, no resident-fallback warning appeared, and
the slots offered real proposals. The diagnostic contract never exposes the ring's
internal indices, so byte-level ring internals remain unmeasured and need the CUDA
parity tests (the gate's own evidence note says the same). If the ring cannot get
pinned RAM the native warns and keeps the draft K/V in VRAM.

### Main versus per-slot drafters, and what they cost

Without `--batch-mtp` a slot has no drafter at all: the main owner's drafter (`--mtp`)
is used by the solo path only, which is why a request alone is not put into a slot.
With `--batch-mtp` the engine loads one additional drafter per batch slot and caps
each at **one draft per window** (`set_max_drafts(1)`), while the solo path keeps
its full `--spec` window. The per-slot drafters **share the main drafter's loaded
weights** (its dense and expert tensors, `MtpDrafter::load(..., shared)`) but keep
their own K/V, buffers and bind, so the extra memory is the ring and buffers, not a
second copy of the weights. A slot's draft K/V is always its own: it is never read
by another slot or by the main owner, and a transfer between them is allowed only
when that slot's draft state is shown to match the exact source prefix - otherwise
the owner stays target-only (above).

Memory tradeoff, measured on the RTX 5060 Ti A/B below: the two batch slot sessions
take 1.00 GiB of VRAM that the expert cache would otherwise hold (native line
`strata generate: --batch 2: the slot sessions take 1.00 GiB of VRAM on CUDA0 that
the expert cache would otherwise hold`), which is why this is opt-in and why a
smaller card should check the free-memory log first. The MTP feature's own extra
cost was about 21 MiB and 8 expert slots (expert cache 6407 MiB / 2430 slots
without it against 6386 MiB / 2422 slots with it, both arms already running
`--batch 2 --kv-unified`). A default `--mtp-window` that is at least the context
makes each slot's draft K/V fully resident; a bounded window keeps it to
`window + 4*max_t + 64` cells.

### Row reservation: the optional row falls back, and never pressures

A window first reserves its **mandatory** target row `[p, p+1)` through the
ordinary reclaim-then-retry path. Only then does the production policy helper
(`include/strata/core/batch_mtp_policy.hpp`) choose: `incoherent`, `not_ready`,
`limits`, `capacity`, or the optional row. The optional probe asks once for the
exact two-row extent `[p, p+2)`; a shortage becomes
`BatchMtpFallback::reservation` - a **target-only window with no reclaim retry and
no pressure escalation**. Only the mandatory row's shortage after its reclaim retry
reaches `finish_shared_slot(..., "pressure")` and parks an owner. The guarantee is
about that *shortage* path only: a **successful** optional reservation is an
ordinary two-row write, so it maps shared pages and can contribute to a later
mandatory shortage (in this slot or another) which the mandatory path may then
reclaim or park for.

So a per-slot `fallback_reserve` counter is a real optional-row shortage: not a
completed response, not a park, and not a token mismatch. Measured (context 4096,
4096 MiB parking, `--mtp-window 128`, two 1787/1791-cell prompts, cap 320): slot 1
reported `fallback_reserve=1` with every other fallback reason zero
(`offered=128 accepted=128`), slot 0 reported `fallback_reserve=0`
(`offered=130 accepted=130`), and the single park was the witness slot's own
mandatory row (`BDONE 1 258 pressure`, sibling `BDONE 0 261 cancel`, one park of
2048 tokens / 528687156 bytes). The gate accepts two shapes: one slot falls back
and then parks while the other cancels, or both slots fall back and cancel with no
park. The measured run observed the first; the second is accepted and not yet
observed.

Timing caveat that belongs to the contract: `batch_mtp_stats` lines and the park
line are printed at the terminal only, so the *order* of a park relative to the
optional window is not observable. The attribution above comes from the source
(only the mandatory reservation's shortage retry can reach
`finish_shared_slot(..., "pressure")`; the optional probe cannot reclaim or park)
plus the slot's own parked prefix, not from timing.

### A fallback ends speculation for that slot lifecycle (fail-closed)

The effect is not limited to the window that fell back. After any target-only
window, `batch_draft_after_commit(..., drafter_advanced=false, ...)` invalidates
the slot's draft coherence (`include/strata/core/batch_draft_coherence.hpp:31-36`,
called from `src/program/generate.cpp:8888-8894`), and nothing re-establishes it
until the slot is re-admitted against a matching prefix. Every later window in that
lifecycle is therefore target-only as well, counted as `fallback_incoherent`.
Measured in the pressure run below, slot 1: `fallback_reserve=1` together with
`fallback_incoherent=4` in the same row (`windows=257 offered=252 accepted=251
rejected=1 fallback_attempts=5`). The counter line is a per-lifecycle total and
does not timestamp the order, so the ordering is the source rule - once invalidated,
every later window is `incoherent` - and in the measured row the reservation
fallback is the transition that starts it. A slot that never had coherence reports
only `fallback_incoherent` (the task-20 tails clone: 7 of 7 windows). The output stays
exact (target-only windows still commit verified target tokens, and the tails /
limits / lifecycle stages matched their solo references exactly), but the slot's
one-proposal throughput gain is gone for the rest of that lifecycle.

### The per-slot counter line

Emitted once per slot MTP lifecycle (opt-in; terminal, pressure, cancel and handoff
paths):

    strata batch_mtp_stats slot=N windows=N offered=N accepted=N rejected=N discarded=N
        fallback_attempts=N fallback_incoherent=N fallback_not_ready=N
        fallback_limits=N fallback_capacity=N fallback_reserve=N

`offered == accepted + rejected + discarded`; a target-only window has exactly one
reason counter; `offered` counts proposal rows passed to a successful verifier
call, so it is verifier evidence rather than an intent counter.

### Target-only restore and checkpoint hand-back

A restored or cloned owner whose private draft history cannot be shown coherent
decodes target-only until a full residual replay rebuilds it. The native says so
and stops offering:

    strata serve: TARGET_ONLY restore: private MTP proposals suppressed until full replay
    strata serve: TARGET_ONLY slot clone: private MTP proposals suppressed until full replay
    strata serve: TARGET_ONLY decode: T=1, MTP/suffix proposals disabled

The ring stays allocated and is simply not used. Measured (`tails` mode, context
1024, offsets 1/2/3): each offset restored its exact shared prefix
(`reused 105/106/103`, equal to the expected prefix), printed all three
diagnostics, and its clone `DONE` reported `drafts accepted 0 of 0`. Shared-prefix
reuse, the source-slot give-back line, cap 8 and token parity all stayed hard.

One accounting gap is documented rather than papered over: the native emits **no
per-slot `batch_mtp_stats` row** for a suppressed owner that never ran a verifier
window (the restore/clone runs on the main path), so the gate witnesses the three
diagnostics plus the owner's own `DONE` draft counters and the reused-cell count.
It never requires, fabricates or synthesizes a zero-window row. When the branch
does generate enough tokens to run target-only windows, the clone reports its own
zero-offer row (task20: `windows=7 offered=0 accepted=0 rejected=0
fallback_incoherent=7`), and the gate accepts that row while still requiring zero
offers; losing any diagnostic fails closed.

Divergent-branch depth (task20, `--mode tails`, context 1024, offsets 1/2/3, the
default `--tail-suffix new-user-turn`): the branch reference generated **8 tokens
at every offset** (branch prompts 140/141/138 cells = 105/106/103 shared + 35
suffix, cap 8), with the clone still offering **0** proposals and reporting
`fallback_incoherent=7` for its seven target-only windows. The gate therefore
credits `multi_token_divergent_suffix_restore = RUN (every partial-tail branch
generated >= 2 tokens; depths [8, 8, 8])` - the earlier depth-1 result (the
`meta-instruction` wording) is superseded. Artifact
`task20-tails-new-user-turn-context1024.json`
`4dcc919d17dc1f2d6df8019481530224b051c0197ebe1ac5fdff579962202343`
(`passed=true`, reuse 105/106/103, exact branch parity). This is a terminal
target-only restore: no coherent-MTP-COW or private-ring-transfer claim follows
from it.

### Pressure park, canonical restore, resumed continuation

Measured (context 4096, 4096 MiB parking, `--mtp-window 128`, 1536/1540-cell
prompts, cap 768): `BDONE 0 513 pressure`, a positive target-only park of 2048
tokens (`parked=2 bytes=524904832`), the sibling cancelled by `BSTOP`
(`BDONE 1 509 cancel`), and per-slot rows `256 offered / 256 accepted / 0 rejected`
and `252 / 251 / 1` (507 accepted and 1 rejected across both slots). The
re-submitted MAIN continuation then reused the whole consumed prefix exactly
(`restored 2048 tokens (live) in 14.6 ms`, `actual_reused_cells == expected ==
2048`, the last returned token unfed), reported `TARGET_ONLY restore` and `T=1`
suppression, offered **0** MAIN drafts, and reproduced the same-binary solo
reference exactly. This run is also the geometry behind the bounded-ring statement
above: 200 ring cells against a 2048-cell consumed history (inference, not a
byte-level ring witness).

### Coherent full-slot hand-off (the transfer that resumes drafts)

`BHANDOFF <slot>` ends an active slot with a target-only idle cache and its
checkpoints. The engine copies the slot's sessions back and, with `--batch-mtp`,
also transfers the slot's private draft image for the exact `upto`; MAIN resumes
drafting only when that transfer is shown coherent, otherwise it stays target-only
(the clone path above). Measured (context 4096, `--mtp-window 128`, 1536-cell
prompt, cap 64): `BDONE 0 3 handoff 83.7` with a nonterminal 3-token prefix and
source slot `offered=1 accepted=1`; the MAIN continuation then reused exactly its
unfed prefix (`reused 1538 == expected 1538` of a 1539-cell prompt), generated 61
tokens that continue the same-binary solo reference exactly (3 + 61 = 64), and
resumed **30 MAIN draft offers, all accepted** (`DONE ... length 30 30 1538 ...`). The
run's only `TARGET_ONLY slot clone` line lies in the same-binary reference stage's
byte range; the continuation range was audited clean.

Not claimed by this: byte-level ring internals, COW page identity, and that every
checkpoint path resumes drafts. The checkpoint path measured in the same run went
target-only, as documented above.

### Identity of the measured gates

Every gate ran a private subprocess, one native process at a time, behind an
artifact-only guard that pinned these hashes:

- validation binary (test hook ON)
  `e14c60c7f1a892b2bbafe029a5a527261b6a9d2311063aba2a7eca8d17f29874`;
- performance binary (hook OFF)
  `ced3cedb7b5089b8f02249acc4789bbf2179e89fe9194a727b1e6a03f68bfa94`;
- config `nvfp4-trial-model.json`
  `5dcb8e1202140f114e34daae5e056f7ef2e2ac956dbfbfcada9e64b56d2f8913`;
- runtime base `ff48f4b761169aff62b1c2e3e8f4c1158f251326`: every run verified
  `git diff --quiet ff48f4b7 -- src include CMakeLists.txt` empty, and no rebuild
  was performed;
- build (CUDA 13.3, GCC 15, `CMAKE_CUDA_ARCHITECTURES=120`, Release): 10/10 host
  ctest targets, `shared_kv_stream_parity` for fp16/int8/q4_0,
  `verify_parity --selftest`, `spec_verify_parity` (1296 rows, 0 mismatches) and
  `verify_batch_parity` passed on the RTX 5060 Ti.

Correctness gates (context 1024 unless noted; `--batch-mtp` explicit):

- `correctness`: joint overlap (BADMs 8 and 15; 15 and 12 interleaved `BT` events
  before the first `BDONE`), exact 16/16 token parity, offered 17 / accepted 13 /
  rejected 4, a forced accept and a forced reject on each slot's first offer, and
  natural follow-on offers (7 and 8).
- `limits`: `windows=1 offered=0 accepted=0 fallback_attempts=1 fallback_limits=1`,
  output identical to the solo 2-token reference.
- `lifecycle`: cancel one slot (slot 0 `offered=1 accepted=1`) while its sibling
  completes, then same-slot reuse with exact parity.
- `tails`: the three target-only diagnostics and zero-offer clone counters at
  offsets 1/2/3, reuse 105/106/103 with exact branch parity, and - with the default
  `--tail-suffix new-user-turn` (task20) - measured branch depths `[8, 8, 8]` at
  every offset, crediting `multi_token_divergent_suffix_restore = RUN`.
- pressure / optional / handoff (context 4096): the three subsections above.

### What was not run, and what is not claimed

- HIP and SYCL: skipped (unavailable/not configured in the build environment).
- Byte-level ring internals and COW page identity: not claimed; the ring witnesses
  are the helper-derived size, the counters and the token parity.
- Park ordering inside a stage: not observable (terminal-only diagnostics).
- Divergent-suffix depth is RUN for the default `--tail-suffix new-user-turn`
  only; `open-fence`, `open-list-primer`, `assistant-lead-in` and the
  `meta-instruction` control were **not** swept on hardware (the sweep stops at the
  first candidate that clears depth >= 2 at every offset). The default target-only
  baseline smoke measured 13 tokens for `new-user-turn` and 16 for
  `open-list-primer`.
- `coherent_shared_tail_cow_speculation`: UNTESTED - it needs a live/coherent-source
  fixture; the tails path is target-only restore and does not claim COW. The
  coherent *full-slot* transfer above is a different witness.
- Partial-yield preservation with batch-MTP, the logical-context edge, and the
  steady batch CPU fraction: not measured.
- Output quality: not measured (the throughput work is a synthetic workload), and
  no production deployment, reload or launcher recovery is authorized or performed
  by this work.
