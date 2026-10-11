# Adaptive fetch admission in unified batch serving (str-pk5)

## Decision

Keep fetch admission disabled in every `--batch` configuration, including
`parallel=2`, unified KV, batch MTP, and a single request currently running on
that engine's solo `GEN` path. Do not remove `STRATA_ADAPT_FETCH=2` from the
production configuration to conceal the mismatch. No production settings,
model, sampling, or inference kernels are changed here.

This is an assessment, not a measured optimization. Static inspection suggests
that a single-GPU serial batch window is a plausible candidate. It does **not**
establish an inherent flaw in all batch admission. Admission was not enabled:
the completion/publication invariant below has not been demonstrated across the
unified lifecycle. The original helper VM had no native backend compiler; the
later isolated nibbler CUDA compile/link passed, but GPU validation is still
absent (see the verification section). Neither compilation nor a CPU policy test
can establish GPU cache coherence.

Ordinary adaptive swaps are a separate mechanism and remain available. Requested
mode 2 does not suppress their copies while the effective fetch mode is zero.

## Native and API facts

The shared CUDA/HIP driver reports this before serving `READY`:

```text
INFO adapt_fetch_requested=2 adapt_fetch_effective=0 adapt_fetch_reason=batch
```

`GET /metrics` exposes the same facts using the existing INFO parser:

```json
{
  "engine": {
    "adapt_fetch_requested": 2,
    "adapt_fetch_effective": 0,
    "adapt_fetch_reason": "batch"
  }
}
```

- Requested means the normalized native `STRATA_ADAPT_FETCH` setting (0..2).
  It includes the existing fork default of **2** when the variable is absent;
  this patch does not adopt upstream's default of zero.
- Effective means the admission mode allowed by the native startup policy, not
  a claim that any particular layer/window admitted an expert. A nonzero mode
  still needs pinned source bytes, a copy-kernel plan with `dst2`, a nonzero
  PCIe share in that window, and a suitable victim/usage threshold. Request-local
  PCIe shares can also differ from startup configuration.
- `none` means the startup policy allows the requested mode. `disabled` means
  requested mode zero. Other codes, in existing exclusion priority order, are
  `adaptive_tier_off`, `no_residency_table`, `spec_split`, `layer_split`, `peer`,
  `helper_gpus`, `adapt_async`, `pipeline_windows`, `batch`, `pcie_mode`,
  `no_gpu_plan`, `no_pcie_share`, and `resident_ram`.
- Only the first exclusion is reported; `batch` need not be the only unsuitable
  property. For example, an empty adaptive usage table takes precedence.
- These are native process facts, not Python guesses based on environment or
  active slots. Legacy engines omit them. A restart into an engine omitting the
  fields discards the predecessor's fetch state instead of preserving stale facts.
  An unloaded engine can still show its last loaded facts alongside the existing
  `live.state=unloaded` indicator.
- The CLI speculative branch reports the same fields when its verifier/fetch
  policy is initialized. Non-speculative CLI execution has no such admission
  setup; these fields are not invented there.
- JSON `/metrics` is covered here; no new Prometheus series or `/props` fields
  are added. The SYCL port carries its own migrated `sycl/src/program/generate.cpp`;
  this patch does not update that independently maintained driver.

## Coherence trace

Function names below are stable anchors rather than line numbers in a shared
checkout that other helpers are editing.

### Shared state and captured layouts

`generate.cpp` builds `host_res[layer * n_expert + expert]` from the cache and
uploads `d_res` with `res_put`. `VerifyHits` gives the verifier both tables and
the cache base/slot offsets. `ExpertDispatch::host_res` points at the same host
vector; enabled admission assigns `res_mut` to it as well.

Batch slots have distinct recurrent/KV sessions, not distinct expert caches.
`init_slots` binds those sessions to the main verifier. The single-GPU serial
`batch_step` collects all active rows, including proposed MTP rows, and calls
`ver.run_slot_rows`. In `verify.cpp`, `capture_batch` caches a graph by active
slot-row layout and all-resident mode (`bkey`), and `record_window` captures
shared residency/plan pointers, not a private residency table per slot. Its
expert path uses `doorbell_publish_res` (or device `resident_plan`) and the
published plan. `set_plan_slot(0)` supplies the current host plan's `dst2`.

`run_slot_rows` invokes `pre_launch_` before `stage_batch` and graph launch. It
services the CPU pool for all rows of a layer, then synchronizes the verifier
stream at window end; the copy stream is also synchronized. `refresh_ar` in
`stage_batch` adjusts the all-resident graph choice from the current host table.
Thus cached layouts are not themselves evidence of a stale expert pointer.
Nevertheless, stale `d_res` is significant: it can make the captured doorbell
skip publishing an activation while the host pool believes an expert is missing.

### Fetch admission inside the window

`expert_pool_dispatch_multi` increments shared routing usage and builds a plan
from the host residency table. `fetch_can` requires mode > 0, mutable residency,
usage, a mode-2 copy-kernel plan with `dst2`, no peer/helper tiers, a PCIe share,
and a source able to serve the layer over PCIe.

For admission the pool:

1. Selects the most-used misses for the PCIe share.
2. Marks every expert routed by **all rows of this layer-window**, including
   proposals that may later be rejected. A victim must be resident and not marked.
3. Skips incoming candidates with `in_flight[index] != 0`.
4. Sets `P.dst2[q]` to the victim's cache address, immediately marks the victim
   missing and the incoming expert resident in `res_mut`, and records both dirty
   indices.
5. Publishes the plan. Captured `fetch_blobs` and `rebase_ptrs` use `dst2` to copy
   into that slot and compute the fetched expert there.

**The host ownership change precedes the device copy.** It is safe only if no
other consumer observes that new residency until the copying window has
completed, and no consumer still references the evicted slot. Excluding every
routed row protects the current layer-window, not arbitrary other streams.
Serial `run_slot_rows` completion is encouraging evidence for a narrow candidate;
it is not a substitute for checking every route into another consumer.

### Ordinary swaps, pending copies, and evictions

`adapt()` ranks usage, submits replacement copies on the refill stream, marks
victims missing immediately, and records `(incoming residency index, slot)` in
`pending`. With admission enabled it marks incoming experts `in_flight` so the
pool cannot independently admit them. A pending destination is unavailable as an
admission victim because no host-resident expert owns it until completion.

`apply_pending` waits for or queries the refill event (and stage events), commits
resident-RAM exchanges, marks the incoming entries resident, clears `in_flight`,
and uploads residency. Dirty indices survive a failed upload for the hook.
Serial batch steps use `apply_pending(false)`; request setup and cache-sensitive
transitions also have explicit waits. A narrow candidate must preserve those
boundaries, not treat a query as a completed copy.

For lagging tiers, `evict_sync` records the victim entries as dirty before copies
land. `res_sync` patches changed entries on the verifier stream before the next
launch, falling back to a full-table upload. Admission also dirties both owners;
`STRATA_ADAPT_FETCH_CHECKRES` compares device and host tables before launch.
Pipelined groups and windows have different fences/publication rules and are
not candidates for removing the guard.

Mode 2 skips the tier's own pinned candidates only when `fetch_admit == 2` and
that layer's `fetch_avail` indicates a fetchable window since the previous round;
usage still decays. With batch excluded, `fetch_admit` stays zero and
`fetch_avail` stays empty, so the fetched-candidate shortcut is false. Neither
the policy extraction nor telemetry changes `adapt()`, `evict_sync`, pending
copies, `usage`, or these conditions.

### Unified transitions that a bounded implementation must cover

A batch-capable engine also executes solo GEN, prompt verify windows and
interleaved slot decode while an incoming prompt is being admitted. Admission
uses the process-wide dispatch and its `fetch_avail`/dirty state, not an isolated
per-request cache transaction. Changing a startup guard would enable admission
at these other call sites too, not only the selected serial decode window.

`copy_to_slot`, `copy_from_slot`, `finish_shared_slot`, target-only pressure
parking, conversation restore, and prefill `lend`/`refill` change sessions,
physical KV ownership or temporarily borrowed expert slots. Some synchronize
explicitly; that does not prove all paths preserve the newly added ownership
changes. Rejected MTP rows still used expert weights and must remain in the
victim exclusion set even though their recurrent state is not committed.
Cancellation must finish or abandon a copying window safely before another
owner is allowed to use its host-published residency.

## Precise admission safety blocker and next work

The unproven invariant is:

> Between the eager `res_mut` admission and completion of `fetch_blobs`, no
> next slot/solo/prefill/draft consumer may use the new host owner or a stale
> device owner; after completion, every captured layout must observe the patched
> table before it may skip activation publication or plan an expert as resident.

The existing process-wide guard prevents the candidate from exercising this
invariant in batch service. Deleting it is not a bounded implementation. There
is no per-window admission scope that currently limits enabling to serial unified
batch decode while excluding the other transitions above. A candidate needs
that scope (or a documented proof that every newly exposed path is safe), explicit
completion/error boundaries, and isolation of pending/dirty/fetch-availability
state across those boundaries. This assessment does **not** claim a reproduced
race in the excluded configuration, nor that such a scope cannot be implemented.

Device evidence is also missing: CHECKRES compares tables, not slot bytes, and
matching tables can both name a slot whose copy has not landed. A candidate must
check cache ownership uniqueness, valid slot bounds, residency read-back and
resident slot bytes, as well as the existing stale-activation counter. No native
GPU lifecycle gates or off/on measurements were run here, so enabling the
candidate would exceed the available correctness evidence.

Before enabling, the coordinator must pass batch-MTP accepted/rejected/discarded
proposal tests, cancellation, solo-to-batch warm handoff, target-only fallback,
parking/restore, pressure continuation and interleaved admission tests, with one
and two active streams and cold/warm caches. Exercise retained and newly captured
slot layouts, refill copies pending at transitions, and default NOWAIT/evict-sync
behavior. Check ordinary adaptation remains active when fetch is excluded.
Then run matched alternating off/on trials on one frozen binary/model, reporting
decode tok/s, TTFT, CPU/GPU phase time, PCIe bytes, cache hits and adaptation bytes.
No speedup claim or rollout follows from this patch.

## Baseline behavior contract

Relative to the pre-change driver, the requested-mode normalization, fork default
2, exclusion order, and resulting `fetch_admit` are unchanged. The CPU regression
retains the old checks as its oracle and exhausts their predicate combinations.
The shared driver still supplies the same predicates in both initialization
branches. It does not change expert selection, residency mutation, copies,
pending swaps, graph capture/replay, sampling or token emission. Batch admission
remains zero; ordinary adaptation remains independent.

Intentional observable differences are additive native INFO output and API facts,
plus discarding stale fetch facts at restart. Therefore the **whole stdout/API
response is not byte-identical**. The contract concerns inference token/logit
bytes and default execution policy, not absence of new telemetry. Native
inference-byte parity has not been measured: the coordinator should compare
pre-change and candidate decoded token IDs/logits using identical model/input,
seed, cache state and deterministic adaptation settings, separately from the
production-like asynchronous performance trials. Keep production configuration
unchanged; a test-only deterministic setting is not a rollout.

## Verification and coordinator build checks

Local checks are CPU-only. The policy regression compares all 24,576 combinations
of normalized modes and the 13 original eligibility predicates to the pre-change
checks. This establishes unchanged admission decisions/exclusion priority, not
measured inference-output byte parity. The Python test derives `--batch 2` and
mode 2 from the actual tracked `deploy/nibbler/config.json`, calls the shared
native policy in a CPU fixture, then passes its INFO line through a fake process,
the real parser and HTTP `/metrics`. Static registration checks bind that policy
to both native driver branches. No model paths from that config are opened.

Run these on the frozen review source, outside production:

```sh
c++ -std=c++20 -Wall -Wextra -Werror -Iinclude \
  src/program/adapt_fetch_policy_test.cpp -o /tmp/str-pk5-policy-test
/tmp/str-pk5-policy-test
python3 -m unittest serve.test_adapt_fetch_reporting serve.test_unified_reporting \
  serve.test_incremental_kv serve.test_kv_handoff -q
git diff --check
```

Local outcomes:

- Standalone `c++` policy build with `-Wall -Wextra -Werror`: passed all 24,576
  combinations. The primary independently repeated this with `g++` successfully.
- The combined command above: **48 tests passed in 11.374 seconds**, recorded in
  `/tmp/str-pk5-targeted-tests.log`. The primary independently ran the five new
  reporting tests successfully. Existing telemetry `/proc` ResourceWarnings
  remain; these are not test failures and were not changed here.
- A supplemental parallel/lifecycle/restart suite invocation was aborted without
  a completion summary (`/tmp/str-pk5-restart-parallel-tests.log`). It is not
  reported as passing; no rerun or unrelated failure investigation was attempted.
- `git diff --check`: passed.

Those earlier local checks had no `nvcc`, `hipcc` or `icpx`; native builds were
skipped at that stage. The later isolated nibbler **CUDA compile/link passed**
with this same fetch driver/header and the Linux loader: CUDA 13.3.73,
GCC/G++ 15.3.0, Release arch 120, `-j2`, 166 steps, 219 seconds and exit 0.
The 597-file source snapshot and linked `strata` identity are preserved in the
[native-build archive](../bench/results/2026-10-11-nibbler-native-build/README.md)
and original [report](../bench/results/2026-10-11-nibbler-native-build/REPORT.md).
Ninja's recorded dependencies confirm the fetch policy header was compiled into
`generate.cpp.o`; this is not just a standalone policy build.

Nine CPU CTests passed in 0.54 seconds. The native arena parity executable was
compiled/linked but **NEVER EXECUTED**. Host `/tmp` tmpfs cases used buffered
fallback; injected direct cases and the separate independent VM ext4 CPU direct
read audit do not establish nibbler `/data` direct I/O or GPU/model correctness.
Recorded production/dependency hashes and engine/vision process identities were
unchanged across the build. No configuration/default or admission policy changed,
and no inference-output parity, GPU lifecycle gate or speedup was established.
HIP/SYCL compilers were absent from the checked host paths; Windows remains
unbuilt. GPU validation and all four performance-audit acceptance gates remain
pending. The primary reports independent same-code reviews with no Blocker/High
findings for nibbler readiness, **not release approval**.

The coordinator should use a frozen source snapshot (`SRC`) and fresh disposable
build directories (`BUILD_ROOT`) on appropriate toolchain hosts, not the live
engine directory. CUDA 13.x must support the target architecture. The following
are build instructions, not authorization for fresh host/GPU operations:

```sh
cmake -S "$SRC" -B "$BUILD_ROOT/pk5-cuda" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_ENABLE_HIP=OFF -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build "$BUILD_ROOT/pk5-cuda" --target strata -j2

cmake -S "$SRC" -B "$BUILD_ROOT/pk5-hip" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build "$BUILD_ROOT/pk5-hip" --target strata -j2

cmake -S "$SRC" -B "$BUILD_ROOT/pk5-sycl" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_ENABLE_SYCL=ON
cmake --build "$BUILD_ROOT/pk5-sycl" --target strata -j2
```

The SYCL check is a separate-port compatibility check: its migrated driver takes
precedence over the modified shared driver, so passing it would not validate this
new telemetry on SYCL. A port-owner update is separate work if that parity is
required. No migrated files were edited by this helper.

Suggested optional CPU-test CMake registration for the CMake owner (not applied
by this helper), inside an enabled standalone CPU-testing block:

```cmake
add_executable(adapt_fetch_policy_test src/program/adapt_fetch_policy_test.cpp)
target_include_directories(adapt_fetch_policy_test PRIVATE include)
add_test(NAME adapt_fetch_policy_test COMMAND adapt_fetch_policy_test)
```

Only the designated primary/operator may perform subsequent nibbler operations.
The original fetch-assessment helper made no nibbler requests or host builds;
the later isolated native compile/link and CPU checks are recorded above.
This documentation/evidence refresh used already-retrieved local files only:
no new host/model/GPU operations, builds, Beads mutations, agents/worktrees or
commits/pushes. The primary's earlier broad Python baseline timed out after E/F
output without a classified summary; it is not a clean baseline and this change
does not attempt to repair unrelated failures.
