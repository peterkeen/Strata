# Nibbler Tier-B first-wave kernel plan

**Status:** source inventory and acceptance manifest, not a measurement report.
First-wave parity code is committed (`5ad25255` fused norm+RoPE, `8b375cc7`
sub16 gate) and harness hardening is committed (`2612b143`, `8eb80858`). The
preliminary foundation CUDA gate (sta-osh.13) **passed** on nibbler at
`8b375cc7`; the final rebuild/rerun gate (sta-osh.10), benchmark compilation
and every model reachability/timing gate remain **pending**. Offline Python
success cannot satisfy a GPU gate.

**Owned scope of sta-osh.3:** this new document only, in
`/home/pete/devel/strata-tier-b`, branch `perf/nibbler-tier-b-kernels`.
Concurrent implementation tasks own their code; this document does not accept
those implementations. Beads were inspected read-only.

## Contents

- [Pins and evidence](#pins-and-evidence)
- [First-wave manifest](#first-wave-manifest)
- [Other kernel families](#other-kernel-families)
- [Evidence and numerical contracts](#evidence-and-numerical-contracts)
- [Acceptance and dependency map](#acceptance-and-dependency-map)
- [Bounded proof and timing protocol](#bounded-proof-and-timing-protocol)
- [Host rejection gates](#host-rejection-gates)
- [Artifact manifest and disposition](#artifact-manifest-and-disposition)
- [Offline document verification](#offline-document-verification)

## Pins and evidence

| Identity | Pin | Meaning |
| --- | --- | --- |
| Original production fork | `9921eecae9d442878de5bfe2b2cd25a71fbdb4ed` (`9921eec`) | Historical comparison source, version 0.1.39; not a port target to modify. |
| Upstream engine | `e8ca9afd03d839d4f8dbbe82dffce7f8a3bafd7a` (`e8ca9af`) | Version 0.1.40.2, already compiled and tested in Tier A; first-wave kernels are present. |
| Completed Tier A | `667910e2` | Audited local stage closure, not a different compiled engine. Later harness/planning commits must not be mislabelled engine pins. |
| First-wave parity code | `5ad25255` (f), `8b375cc7` (h) | Committed rope_parity/sub16 extensions in this clone; foundation GPU gate passed at `8b375cc7`, no benchmarks timed. |
| Harness hardening | `2612b143` (sta-d76), `8eb80858` (sta-0a7) | Committed cancellation + prefill fixes; offline evidence only. |
| Foundation GPU gate | `8b375cc7` | Primary build `/data/llm/Strata-tier-b/build-foundation`; 3/3 parity ctests PASS, 0 skip. Artifacts `/data/llm/Strata-tests/tier-b-foundation-20261007` (local copies `/tmp/strata-tier-b-artifacts/foundation`). Preliminary only. |
| Tier-B build | **Foundation done, final pending** | Foundation `build-foundation` built at `8b375cc7` (CUDA 13.3.73, GCC 15.3, sm120); record final source+deps+binary hashes after later code and `.8` registration. Never attribute the instrumented binary to pristine `e8ca9af` alone. |

The fork and upstream engine have **unrelated git histories**. Tier A compared
whole pinned trees; it did not isolate kernel ports. This is focused validation
and same-binary attribution on the upstream base, **not** an eleven-PR
megabackport, an indiscriminate reapplication to the fork, or a merge of histories.

Read the full [Tier-A report](NIBBLER_TIER_A.md), especially
[build/source provenance](NIBBLER_TIER_A.md#3-binaries-under-test),
[methodology](NIBBLER_TIER_A.md#6-harness-methodology),
[eligibility](NIBBLER_TIER_A.md#10-eligibility-notes-per-comparison),
[limitations](NIBBLER_TIER_A.md#11-interpretation-guide), and
[raw artifacts](NIBBLER_TIER_A.md#12-raw-artifact-paths).
Saved artifacts are under `/data/llm/Strata-tests/tier-a-20261007/` on nibbler;
local copies used for this inventory are `/tmp/strata-tier-a-artifacts/`.
These are historical snapshots, not fresh host checks.

The saved `targeted-gates.log` records 18 targeted passes, but does **not** include
`rope_parity`, `iq_multi_parity` or `native_grouped_parity`. The saved full matrix
has 36 records (35 eligible arms and one G7 skip); canonical adapt replay has nine
arms. Those artifacts do not prove the new first-wave direct/graph gates. The
foundation GPU job (sta-osh.13) at `8b375cc7` passed `rope_parity`,
`iq_multi_parity` and `native_grouped_parity` 3/3 with 0 skips; the new section 6
and sub16 sections executed bit-identical direct+graph comparisons. This is
foundation-SHA proof only, not the final post-change build and not benchmarks.
Tier-A decode includes first-use CUDA graph capture/compilation in fresh
processes. Its aggregate differences cannot be assigned to f or h. Forced
continuation measures controlled processing cost, not free-generation quality;
matching emitted IDs does not imply matching verify-window T or capture cost.

Historical bench hardware/toolchain in the report: Ryzen 9900X with AVX-512,
RTX 5060 Ti 16-GB class, CUDA architecture `sm_120`, CUDA 13.3 and GCC 15.
These identify the intended target; fresh state and exact toolchain must be
recorded again. No new measurement or current private host fact is asserted here.

## First-wave manifest

Only **f** and the **native grouped-layout subset of h** are first-wave arms.
Both are already upstream, not missing ports. Controls are independent:

| Family | Included upstream history | Default arm A | Reference/opt-out arm B | What the pair isolates |
| --- | --- | --- | --- | --- |
| f: fused native RMSNorm + RoPE | `088e8a82`, test-enable fix `b1efb70c` | `STRATA_NO_NORM_ROPE` unset | `STRATA_NO_NORM_ROPE=1` | Fused per-head norm/rotation versus the existing split native operations, including applicable Q/gate split removal. |
| h: native grouped GU/down sub-warp layout | `2715a752` | `STRATA_NO_SUB16_GU` unset | `STRATA_NO_SUB16_GU=1` | Grouped GU **and down** layout selection only; not every change in PR-h. |

Each pair uses the **same Tier-B executable**, inputs and common settings.
The other family's control remains unset. No combined all-off arm.
`STRATA_NO_MULTI_GR`, old IQ/MMVQ kernels, codebook staging and dense MMVQ
rows2 are not experimental overrides in these pairs.

### f source eligibility and shapes

- [Native RoPE implementation](../src/kernels/cuda/native_rope.cu):
  `native_norm_rope_usable`, `native_qsa_rms_norm_rope`, `norm_rope_kernel`;
  [API and arithmetic contract](../include/strata/kernels/native_rope.hpp).
  CUDA allows head widths **128 or 256**, rotary width **64**. HIP defaults
  differ and are not evidence for this CUDA target.
- [Verifier callers](../src/core/verify.cpp): `fuse_nr` requires native QSA,
  native RoPE and usable geometry; Q uses `in_stride = 2 * head_dim` from
  interleaved q|gate rows, K uses contiguous/in-place data, indexer queries
  use width 128. Side-branch paths can retain split calls: not every Q/K
  operation is necessarily fused.
- [MTP callers](../src/core/mtp.cpp): `record_rest` fuses strided Q;
  contiguous K fusion is restricted to **T=1** because token position rows
  have different spacing for multi-token K. Do not claim every MTP K window
  is fused.
- [Real QSA geometry](../include/strata/kernels/qsa.hpp): 24 query heads,
  two KV heads at width 256; four indexer query heads at width 128;
  rotary64. Primitive gates cover **T=1..8**, translating T to the actual
  row count (for example `24*T`, `2*T`, `4*T`), not confusing heads with T.
- [Existing rope gate](../src/kernels/rope_parity.cpp), section 6, is the
  starting reference. Extend to contiguous out-of-place/in-place and
  strided q|gate, table-enabled and fallback arithmetic, changed input and
  positions on graph replay, and appropriate scaling cases. Table state
  (`STRATA_ROPE_TABLE`, registered matching scaling and table bounds) is
  fixed before capture; unsupported/disabled coverage is explicit.

This is **source-eligible**, not proven executed by a model arm. Diagnostics
must identify actual fused/split calls and shapes; completed work and numerical
comparison must corroborate them before model timing is eligible.

### h mixed-format manifest and shapes

The pack name IQ3_S is not a uniform quantization claim.
[The checked-in native expert manifest](../tests/data/native_experts/iq3_s.txt)
contains the following actual pairs (type IDs follow the native layout API):

| GU format / type | Down format / type | Layers in fixture |
| --- | --- | --- |
| IQ3_XXS / 18 | IQ4_NL / 20 | 14 |
| IQ3_XXS / 18 | Q2_0 / 42 | 3 |
| IQ2_S / 22 | IQ4_NL / 20 | 15 |
| IQ2_S / 22 | Q2_0 / 42 | 5 |
| IQ3_S / 21 | IQ4_NL / 20 | 9 |
| IQ3_S / 21 | Q2_0 / 42 | 1 |
| IQ4_XS / 23 | IQ4_NL / 20 | 1 |

These are fixture counts, not observed runtime dispatch counts. IQ4_XS/Q2_0
may be additional synthetic Cartesian coverage, but is **not** an eighth pair
present in this fixture. Reconcile and hash the actual reused host manifest
before running; do not assume fixture equality.

Geometry is **n_embd=2560, n_ff=640**: GU has `2*640` rows of width 2560;
down has 2560 rows of width 640. The [native grouped implementation](../src/kernels/cuda/iq_kernels.cu)
uses `native_expert_grouped`, `launch_gu`, `launch_down`,
`row_dot_80_sub16`, `row_dot_40_sub8`, `row_dot_20_sub16`.
GU selects 16-lane rows at the eligible 2560 geometry; down IQ4_NL uses
8-lane sub-warps and Q2_0 uses 16-lane sub-warps at n_ff=640.
Despite its name, **`STRATA_NO_SUB16_GU` changes GU AND down**.

The original fork at `9921eec` already contains `row_dot_40_sub8` and
`row_dot_20_sub16` in `src/kernels/cuda/iq_kernels.cu`, used by its grouped
down path. Thus disabling down sub-warps in upstream does not recreate the
original fork. PR-h also changes exact-N columns, small-format decode-once
handling and dense two-row MMVQ launches; those remain fixed. This pair is
not the entire historical PR delta and cannot justify a historical PR speed claim.

[Verifier native routing](../src/core/verify.cpp) constructs a per-layer
`NativeExpertLayout` from `lay.fmt` and calls `native_expert_grouped` on the
native branch, with Q8_1 activations. Alternative grouped v2 paths, old kernels
or zero-work groups can bypass the layout being studied. Selection before
capture and restoration belong to sta-osh.2's narrow selector in the
[IQ API](../include/strata/kernels/iq_kernels.hpp).

[IQ multi parity](../src/kernels/iq_multi_parity.cpp) and
[native grouped parity](../src/kernels/native_grouped_parity.cpp) supply the
existing reference/guard conventions. Required new layout coverage: identical
Q8_1 bytes and weight bytes, actual seven pairs at 2560x640, group tails
**1/2/3/4/5/8/11**, empty groups, scattered destinations and untouched rows;
direct results and changed-input graph replay. Tail11 is a stress case, not
asserted to be an observed production group size. Model timings use shapes
proved active, not nominal capacity presented as actual work.

## Other kernel families

Every listed commit is already included in the pinned upstream history.
Deferred means excluded from this first wave, **not** absent or necessarily
inactive. A control's existence and standalone parity are not model reachability
proof. Narrow controls below are inventory only, not permission to run more arms.

| Family / history | Sources and existing gate | Control boundary and disposition |
| --- | --- | --- |
| a `1690a4a3` | [qsa_select.cu](../src/kernels/cuda/qsa_select.cu), [kv_stream.cu](../src/kernels/cuda/kv_stream.cu); no dedicated block-scores parity gate: [qsa_parity.cpp](../src/kernels/qsa_parity.cpp) and scorer [qsa_select_bench.cpp](../src/kernels/qsa_select_bench.cpp) | `STRATA_QSA_EARLY_EXIT=0` isolates early query-staging exit only where multi scores dispatch is reached; it does not revert the no-op shared atomic removal in KV resolve. Potentially eligible scorer, deferred pending path/work proof and separate attribution. |
| b `ae3b249f`, fix `0c86bbec` | [fused_gdn.cu](../src/kernels/cuda/fused_gdn.cu), [verify_kernels.cu](../src/kernels/cuda/verify_kernels.cu); [gdn_parity.cpp](../src/kernels/gdn_parity.cpp) | Exact-T a/b and double-buffered/all-output recurrence are not controlled by one narrow PR-b opt-out. Keep the shuffle fix. GDN split/commit/QFUSE controls are different experiments, not substitutes. Plausible model caller, deferred; requires dedicated reference and state/output contracts. |
| c `822251be`, fixes/wiring `9d9c9ca2`, `868f2ef0` | [native_router.cu](../src/kernels/cuda/native_router.cu), [native_moe.cu](../src/kernels/cuda/native_moe.cu), [shared_expert.cu](../src/kernels/cuda/shared_expert.cu), [mtp.cpp](../src/core/mtp.cpp); [native_multi_parity.cpp](../src/kernels/native_multi_parity.cpp) | `STRATA_DEC_BATCH=0` changes multiple caller families, not just router/combine/float4 scaling. Alignment, k=10/shared-output and multi-token predicates matter. Preserve the FMA-chain correction and MTP wiring; defer until a narrow control and executed-path evidence exist. |
| d `44ffa86c`, retain `35435e3d` | [verify.cpp](../src/core/verify.cpp), [mtp.cpp](../src/core/mtp.cpp), [kv_q4.cu](../src/kernels/cuda/kv_q4.cu), [native_qsa_indexer.cu](../src/kernels/cuda/native_qsa_indexer.cu); [kv_steps_parity.cpp](../src/kernels/kv_steps_parity.cpp), [qsa_parity.cpp](../src/kernels/qsa_parity.cpp) | `STRATA_NO_BATCH_KV_STEP=1` restores per-token appends, but streamed K8V4 host-copy correctness must be retained. Deferred storage-boundary review; no Tier-C storage/serving work here. Do not reapply pre-fix append code. |
| e `fe4de5de` | [verify_kernels.cu](../src/kernels/cuda/verify_kernels.cu), [verify.cpp](../src/core/verify.cpp); [resident_plan_parity.cpp](../src/kernels/resident_plan_parity.cpp) | Parallel resident grouping/scan has no narrow old-scan control. All-resident requires all experts in the relevant stage; `STRATA_VERIFY_DEVICE_PLAN` is opt-in and off by default. Do not turn either on to manufacture relevance. Resident-plan model execution is unproved under this partial-cache protocol; deferred, no throughput claim. |
| g `104a8485`, fix `023dd015` | [fused_gr.cu](../src/kernels/cuda/fused_gr.cu), [gr.cu](../src/kernels/cuda/gr.cu), [native_bf16.cu](../src/kernels/cuda/native_bf16.cu), [native_gr_norm.cu](../src/kernels/cuda/native_gr_norm.cu), [native_gr_postops.cu](../src/kernels/cuda/native_gr_postops.cu); [gr_multi_parity.cpp](../src/kernels/gr_multi_parity.cpp), [gr_parity.cpp](../src/kernels/gr_parity.cpp) | `STRATA_NO_MULTI_GR=1` broadly changes residual writes, per-token GR and exact-T selection; not a clean first-wave control. Preserve the MTP flag fix. Deferred. `STRATA_GR_DOWN_MAX4` is shadowed by exact-T staged ct=1..6 in the selected Tier-A path: expected no-op from source plus startup diagnostic, **not** a trace of every launch or statistical proof. |
| i `3fd0460b` | [mtp.cpp](../src/core/mtp.cpp); [kv_steps_parity.cpp](../src/kernels/kv_steps_parity.cpp) for append primitives, not the full rejected-row lifecycle | `STRATA_MTP_CATCHUP_ALL=1` includes rejected rows again. Accepted/rejected counts and state catch-up need separate correctness proof; speculative execution alone is insufficient. Deferred rather than assumed inactive. |
| j `8b5dd7ad` | [verify.cpp](../src/core/verify.cpp), shared-expert stream/doorbell scheduling | `STRATA_SH_FORK_LATE=0` changes fork placement only if the shared stream is enabled, profiling is off and CPU-doorbell work is used (`!ar_on()`). Scheduling eligibility is conditional, not executed-work proof; deferred. No standalone parity is claimed to certify scheduling/lifecycle. |
| k `3281ac32`, correction `660e495e` | [quantize_act.cu](../src/kernels/cuda/quantize_act.cu), [s2_expert_grouped.cu](../src/kernels/cuda/s2_expert_grouped.cu); [s2_expert_grouped_parity.cpp](../src/kernels/s2_expert_grouped_parity.cpp) | `STRATA_S2_SWIGLU_Q8=0` splits SwiGLU and scaled Q8_0 only in `moe_grouped_s2` (fast path plus nonnull scales). Keep the correction retaining f-140's quantizer; do not assert an extra quantizer gain. IQ3_S uses `lay.native` with Q8_1 and `native_expert_grouped`, not grouped S2. A native down tensor of type Q2_0 does **not** make this S2 control active. Inapplicable to the selected native model arm unless contrary executed-path evidence is established separately. |

The CPU gather knob is separate from this inventory. On the Tier-A 9900X
AVX-512 dispatch, [native_gu_rows](../src/kernels/cpu/native_expert.cpp)
selects AVX-512 before the AVX2 gather path; `cpu_gather_fast` probes Intel
AVX-VNNI in [iq_avx2.cpp](../src/kernels/cpu/iq_avx2.cpp).
Standalone gather parity does not make `STRATA_IQ256_GATHER` a useful
end-to-end arm on this host. Do not disable AVX-512 to create an artificial gain.

## Evidence and numerical contracts

Acceptance is a ladder; no rung substitutes for the next:

1. **Source eligibility:** supported formats/shapes and caller predicates.
2. **Dispatch/capture diagnostics:** family, selected path, geometry, formats,
   direct/capture submission state, bounded records, explicit zero/unsupported
   work. Capture logs describe graph construction, not each graph replay.
3. **Completed primitive proof:** actual CUDA execution, checked synchronization,
   direct reference comparison, graph replay with changed input/positions,
   guards and nonzero executed-case counts. A successful executable with a
   skipped required CUDA section is not a pass.
4. **Completed model proof:** bounded native-window/logit export, matching natural
   inputs/windows/picks and emitted IDs; confirmed finish/status and evidence
   that the tested family did nonzero work. Dispatch records alone are insufficient.
5. **Clean timing:** diagnostics/readbacks off; only pairs passing the above
   admission criteria enter controlled statistics. Rejected pairs remain saved.

Bitwise comparisons apply **only where the operation-order contract warrants**:
section 6 fused versus split CUDA norm/RoPE on identical mode/input, and h
layouts sharing identical quantized bytes and reduction arithmetic. Do not
weaken identity expectations after seeing a failure; diagnose compiler,
mode/input or implementation differences first. Graph replay must obey the
same contract and may not change host selector state inside a captured graph.

A scalar/FP64 reference validates semantics using its documented tolerance,
not universal bitwise equality with CUDA fast math. The existing rope gate
uses row-input-magnitude relative **1e-6** for host rotation comparisons and
**3e-3** for native fast-trig versus table-path comparisons over its specified
position range. IQ multi's double-dot **relative <1e-2** gross-error guard is
separate from its old/new identity requirement; it uses dequantized Q8_1, not
unquantized activations. Preserve each gate's denominator, finite checks and
range; these tolerances are source contracts, not new measured errors or a
blanket full-model logit tolerance. Full logit comparison rules must be
predeclared by .5/.9, with exact expectations for these identity arms and
any justified tolerance explicitly named before host tests.

Nonrotary RoPE tails must not rotate; fused RMSNorm still normalizes those
head elements, so they are **not raw-input pass-through**. Assert matching
normalized tails, untouched input q|gate padding and allocation guards.
Expert output rows not addressed by the plan must retain their sentinels;
equal empty outputs without positive-work cases are vacuous.
No forced-continuation or limited prompt test certifies general output quality,
production readiness, serving parity or universal feature parity.

## Acceptance and dependency map

This table defines required evidence, not a declaration that another task is done.
`sta-osh` is the parent of .1–.13. All host/runtime rows remain pending.
Planned new paths are shown as code, not links to nonexistent files.

| Task | Required acceptance evidence | Blocking inputs / next gate |
| --- | --- | --- |
| sta-osh.1 | Committed `5ad25255` in `src/kernels/rope_parity.cpp`: section 6 T1..8, heads128/256, rotary64, contiguous/in-place/strided q|gate, table/fallback, direct and changed-input/position graph replay, tails/padding, checked CUDA errors and cleanup; preserve other sections and appropriate tolerances. | → .13/.6/.10; foundation GPU PASS: 192 cases, 576 graph launches, 384 changed-input/pos replays bit-identical, 0 errors. |
| sta-osh.2 | Committed `8b375cc7` in IQ header/CUDA/multi-parity: narrow selector initialized from existing env/default, chosen before capture and restored; identical Q8_1 bytes; mixed pairs2560x640; tails1/2/3/4/5/8/11, empty groups/scattered destinations/guards, direct and changed-input graph comparisons, fail-closed CUDA and cleanup. | → .13/.7/.10; foundation GPU PASS: all 7 manifest pairs 2560x640, tails/empty/scatter, direct+graphs bitwise, 0 failures. |
| sta-osh.13 | Primary-only foundation build of ONLY committed `rope_parity`/`iq_multi_parity`/`native_grouped_parity` at foundation SHA `8b375cc7`: cmake 61 actions CUDA 13.3.73/GCC 15.3/sm120; ctest 3/3 PASS, 0 skip, 2.92 s; unit service 40.919 s peak 1.3 G success; post state 4 prod hashes OK, source `9921eec`, guard 644, audio 1436043, GPU 2 MiB, no compute process. No benchmark/model timing. | .1+.2 + sta-d76 + sta-0a7 → .13 → .4. **PASSED at foundation SHA**; final .10 must rebuild/rerun after later code changes and .8 registration. |
| sta-osh.3 | This plan: pins/history, f/h and exclusions, source/reference links, eligibility limits, contracts, complete acceptance map, bounds, mandatory host/harness gates. | Blocks .11. Document checks only; no code/Beads/GPU authority. |
| sta-osh.4 | Opt-in bounded machine-readable diagnostics in native RoPE/IQ plus small helper; disabled mode has no readback/allocation/sync/logs. Record actual family/shape/formats/path/submission state, distinguish capacity from actual groups and unsupported/empty work, no selector switch inside capture. | .1+.2 → .4 → .5/.11. Submission record is not execution proof. |
| sta-osh.5 | Bounded completed-window export in `src/program/generate.cpp`, minimal verifier accessor only if needed; offline `tools/tier_b/test_diagnostics.py`. Window ordinal/position/T/accepted/emitted/input hashes/status; checked full native logit rows with vocab/shape/offset metadata and explicit truncation; readback/I/O errors reject. | .4 → .5 → .9/.11. Use `Verifier::copy_logits`; ordinary `--dump-logits` bypasses this native path. No change to acceptance/commit or serving ownership. |
| sta-osh.6 | `src/kernels/tier_b_rope_bench.cpp` (untracked, unregistered): parity first, AB/BA with canonical input re-upload before each timed arm outside the event window, bounded warmup/iterations/samples, direct/warmed replay/capture costs separate, changed-input replay gate, shapes/raw pairs/spread, checked cleanup/errors. | .1 → .6 → .8. Fairness fix implemented, not GPU-tested; timing pending. |
| sta-osh.7 | `src/kernels/tier_b_expert_bench.cpp` (untracked, pending parent review): selector restore, actual mixed pairs and proved representative groups2560x640, layout-only change/identical bytes, parity first, rotating >L2 working set with explicit bytes/budget, AB/BA direct/replay/capture raw evidence; old per-column pinned off. | .2 → .7 → .8. Parent review pending; no GPU perf claimed. |
| sta-osh.8 | Only [CMakeLists.txt](../CMakeLists.txt): independent benchmark targets with parity-style kernel/quant-table linkage, CUDA13.3/GCC15/sm120 compatibility, preserved HIP/platform/default targets; bounded explicit invocations, not unbounded automatic CTests. | .6+.7 → .8 → .10. Primary configure/build is mandatory. |
| sta-osh.9 | New `tools/tier_b/run_pairs.py`/`test_pairs.py`: same-binary independent f/h, full STRATA env stripping plus allowlist, fixed settings/adapt0, diagnostic versus clean runs, provenance/raw windows/capture/output/logit validation, strict rejection and cancellation, zero-launch dry-run and bounds/error fixtures. | .5 + **sta-d76 + sta-0a7** → .9 → .11. Reuse hardened Tier-A runner without owning its edits. |
| sta-osh.10 | Primary-only isolated host build in `/data/llm/Strata-tier-b` and fresh `/data/llm/Strata-tests/tier-b-*`: fresh preflight, source/deps/build/binary hashes; all chosen parity and bench targets compile, required new CUDA sections run, logs/exits, no orphan processes, production/audio/router guard preserved. | .1+.2+.8 + **sta-d76 + sta-0a7** → .10 → .11. Must rebuild/rerun after later code changes and .8 registration; .13 foundation proof is not reusable as final proof. Reject skipped required sections or any failure before timing. |
| sta-osh.11 | Primary-only predeclared interleaved native/model pairs: .10 correctness, diagnostic execution/window/logit/output proof first, then at least three paired samples/family/workload, mixed rotating experts >L2, short code/long text, raw statistics/capture provenance and postrun state. | .3+.4+.5+.9+.10 → .11 → .12. No blind continuation after failure. |
| sta-osh.12 | Only new `docs/NIBBLER_TIER_B.md`: actual pins/hardware/protocol/gates, separate warm-native and cold-model measurements with sample counts/spread, honest exclusions/rejections/capture caveats, historical-down qualification, reproducible recipes/artifacts and evidence-limited disposition. | .11 → .12; independent final audit required before epic closure. |
| sta-d76 | Committed `2612b143` in [engine_cli.py](../tools/tier_a/engine_cli.py) and cancellation tests: exceptional/KeyboardInterrupt cleanup of **owned** child/process group, bounded TERM→KILL and reap, original exception/tokenfile behavior preserved, mock and real dummy child/grandchild offline tests, no unrelated kills. Offline evidence only. | Mandatory before .13/.9/.10 and **any new GPU arm**, including parity; also blocks sta-0a7. |
| sta-0a7 | Committed `8eb80858` in the same inherited parser, separately owned task: multi-token prompt requires prefill section/count `len(prompt)-1`, positive finite time/rate and valid TTFT contract; malformed/missing/count-mismatch fixtures reject; one-token no-prefill stays valid. Offline evidence only. | sta-d76 → sta-0a7; mandatory before .13/.9/.10 and **any new GPU arm**. |

Core dependency chains: `.1 → .6`, `.2 → .7`, `.6+.7 → .8 → .10`;
`.1+.2 → .4 → .5 → .9`; preliminary foundation `.13` runs after `.1+.2`
plus both hardening fixes and gates `.4`; hardening also blocks `.9`/`.10`;
`.3+.4+.5+.9+.10 → .11 → .12`. `.13` proof is scoped to the foundation SHA and
does **not** replace `.10`, which must rebuild/rerun after later source changes
and `.8` registration. Closure/status alone is not proof: review code/test
evidence for both hardening fixes before permission to execute.

Existing targets and registration are in [CMakeLists.txt](../CMakeLists.txt).
Primary's later required correctness selection is:

```sh
ctest --test-dir "$HOST_BUILD" \
  -R '^(rope_parity|iq_multi_parity|native_grouped_parity)$' --output-on-failure
```

Run under an owned supervised control group and declared time bounds; do not
execute this from the offline helper. Also select relevant existing native
baseline/dequant/reference, GDN/GR and sampler gates explicitly in the host
manifest. Saved Tier-A passes are context, not a substitute for the new build.
Benchmark target names/options and exact bounded build/run commands must be
recorded once .6–.8 supply them; no invented runnable interface is specified here.

## Bounded proof and timing protocol

### Fixed model settings and environment

Use [Tier-A fixed paths/flags](../tools/tier_a/_config.py) as the read-only
input reference, replacing only the executable/artifact destination with the
isolated Tier-B ones. Pack `iq3_s`, both GGUF shards, expert profile and MTP
stay read-only; record actual paths/hashes. Freeze prompt IDs, cache/profile
placement, KV and speculation settings. In particular: `--expert-cache 2000`,
`--pcie-frac 0`, `--adapt-every 0`, `--prefill auto`, `--spec 4`,
`--spec-min-p 0.5`, `--max-context 262144`, `--kv-resident 32768`,
`--kv int8`, `--vram-reserve-mib 2048`, `--stats`, `--stop-eos`.
Record actual resident slots separately from requested cache slots. No
adaptation, migrated placement, new expert profile or per-arm cache tuning.
Freeze greedy sampling (no `--seed` enabling sampling), penalties and defaults
in the argv manifest; any future sampling experiment needs a separate plan.

**Allowlist:** strip **every ambient `STRATA_*` variable**, not just Tier A's
short controlled-variable list, then apply only:

| Run mode | Allowed STRATA additions |
| --- | --- |
| All model arms | `STRATA_PF_FUSED=1`, `STRATA_IQ_MT_MIN=1` |
| f reference only | `STRATA_NO_NORM_ROPE=1` |
| h reference only | `STRATA_NO_SUB16_GU=1` |
| Proof runs only | Exact bounded diagnostic controls/output paths from .4/.5, recorded before running. No undocumented inherited overrides. |
| Clean timing | **No diagnostic controls**, no logit/window readback or profiling. Both first-wave controls unset for A; exactly one family opt-out for B. |

Freeze and record non-STRATA execution variables (CUDA device visibility,
library/tool paths, CPU thread/affinity settings and relevant CUDA cache state).
Do not record secrets or unrelated environment contents. Primitive table modes
are a separate explicit gate/bench configuration, identical within a pair;
they do not silently enable `STRATA_ROPE_TABLE` in the model matrix.

### Predeclare bounds before the first host launch

The following are **planned limits, not measured results**. Save a resolved
manifest of every command, shape, order, byte cap, timeout and count before
starting. If the implemented interface cannot enforce a limit, stop and resolve
it offline rather than run an unbounded default or tune after seeing results.

- No automatic retries or adaptive sample extension. Fail fast on the first
  genuine launch, timeout, parser, CUDA, parity or reachability error; retain
  the rejected artifacts and resolve a focused task before re-verification.
- Native: sweep declared f T1..8/head128/256/rotary64, strides and both table
  modes; h actual seven pairs2560x640 plus declared active-group representatives.
  For each timing case use 10 warmup pairs, then **three recorded pairs**, each
  arm at most 32 iterations. Direct and graph-replay modes have separate
  samples; capture/instantiate costs are a separate dataset. Cap each native
  benchmark invocation at 300 seconds; no inherited `--bench` default is
  automatically accepted as this experiment.
- Expert allocation budget: at most **512 MiB total device allocation for the
  standalone expert benchmark**, including weights, scratch and outputs.
  Query/record actual device L2 capacity; choose and declare unique mixed-format
  expert weight bytes **larger than L2**, distinct expert/layer addresses and a
  fixed deterministic rotation. Report working-set bytes, bytes touched per
  sample and rotation schedule. Both arms traverse the same addresses/order.
  If >L2 cannot fit the cap, reject the timing case, do not label an L2-hot
  repeat a model-like workload. Warm kernels/graphs, not just one expert's weights.
- Model: two independent families × `short_code` and `long_doc` from the
  [fixed corpus](../tools/tier_a/prompts.py). Reuse token IDs after verifying
  their hashes/counts; proposed budgets are 256 new tokens for short code and
  512 for long text. One diagnostic A/B pair per family/workload (eight proof
  arms total), with at most 16 completed windows and 64 MiB full-logit
  export per arm (the exporter's compiled-in `TierBExport::open` bounds in
  [generate.cpp](../src/program/generate.cpp); the harness rejects a manifest
  declaring different bounds); if a row cannot fit, fail explicitly. Dispatch
  traces are capped at 1024 records: a truncated trace is prefix dispatch
  evidence only, recorded as incomplete coverage, never full coverage. Truncated exports are
  bounded prefix evidence, never whole-run logit proof. Both first-wave
  families are identity arms: exported logit rows must be bitwise equal and
  the whole-run prompt/output token IDs must match; no tolerance is fitted
  after seeing results. Timing refuses to start without a passed live proof
  summary for the same executable sha256, model inputs and prompts, and every
  arm's timeout is clipped to the remaining session budget.
- Only after proof succeeds: one excluded preparatory A/B process pair per
  family/workload, then **three interleaved recorded pairs** per family/workload
  (24 recorded model arms). Each process remains fresh; preparatory processes
  do not warm the next process's decode graphs. Alternate AB/BA/AB for the
  three recorded pairs; rotate family/workload order from the predeclared
  schedule so one family is not always first.
- Model per-arm timeout at most 600 seconds; combined .11 native/model
  measurement session cap 3600 seconds. Host parity/build timeouts are also
  declared in .10 before launch. Exhausting a cap yields incomplete/rejected
  evidence, not permission to overrun or silently drop inconvenient samples.

### Separate correctness instrumentation from clean timing

Proof runs enable .4 submission/capture diagnostics and .5 checked native
completed-window/full-logit export. Compare naturally generated matched
windows, positions, T, input hashes, accepted/emitted counts, picks and IDs;
confirm required work actually completed. Missing diagnostics, zero work,
unsupported formats, changed prompts/windows or failures reject attribution.
The bounded exporter must label truncation and completion separately.
[Verifier::copy_logits](../src/core/verify.cpp) is the native readback interface;
existing [CLI window hashes](../src/program/generate.cpp) may help, but neither
ordinary `--dump-logits` nor capture-only logs replace the completed proof.

Clean timing disables these readbacks/synchronizations/logging. Preserve normal
raw stdout/stderr/stats/IDs and require parser-valid prompt echo, decode count,
positive finite metrics and primary prefill diagnostics. Diagnostic runs are
**never timing samples**. Use separately established proof/capture signatures
and any nonintrusive existing clean-run window metadata. The native CLI's
normal stdout includes `window sizes` (whole-run T histogram) and `speculation`
(accepted/offered counts and rounded rate). Recompute these for each clean arm,
including the excluded preparatory pairs; matching output IDs does not imply
matching rates or distributions. A histogram is not an ordered sequence of
`(pos, T, input_hash)` windows. If obtaining full clean-run sequences requires
instrumentation, record that limitation rather than enable it and call the
result clean. Output mismatches are not a controlled comparison.

**Measured-time feedback:** [DraftPolicy](../include/strata/spec/draft_policy.hpp)
is not frozen by greedy sampling, fixed speculation flags or `--adapt-every 0`
(which disables expert-placement adaptation). In
[draft_policy.cpp](../src/spec/draft_policy.cpp), `observe`/`observe_chain` learn
cost per T from measured `round_ms` (first sample, then EMA weight 0.1);
`cost_ms` uses those costs or scales from measured sizes for unseen T, and
`choose` compares expected MTP/lookup tokens per millisecond with a 3% default
margin and probing/staleness rules. The native CLI loop in
[generate.cpp](../src/program/generate.cpp) starts its round clock before
proposal selection, calls `policy.choose(T, k, sfx_match)`, then after verify,
commit, optional Tier-B readback/hash/export and MTP drafting measures elapsed
round milliseconds and calls `policy.observe` (or `observe_chain`), except
for the initial native window. Diagnostic overhead, cold capture and kernel
or other round-time differences can therefore change later draft choices even
when output IDs match. Default suffix drafting permits T up to 6 with
`--spec 4` (MTP remains capped at 4). Proof-prefix shapes need not describe
clean-run work. This is a source mechanism, not attribution of a particular
observed drift to one timing component.

The [r2 report's clean-run recomputation](NIBBLER_TIER_B.md#73-clean-run-draft-statistics-and-timing-feedback)
accounts for all 32 clean arms: recorded long_doc A/B rates and histograms differ
in 5/6 pairs; all short_code pairs match in aggregate. Ordered clean-run window
sequences are absent. Preserve measured cold-model throughput, but do not
assign its small paired deltas or the native/model gap to kernel cost alone.

**Window matching (amended after host session 1, before any model timing).**
Both the `(pos, T, input_hash)` window key and the minimum **12** matched-window
criterion were chosen after seeing session 1's proof rejection; neither was
originally predeclared. The original comparison used ordinal alignment. The
amendment changes matching/admission coverage, not the bitwise row contract.
Speculative draft lengths are not deterministic run to run: two identical
default configs on `long_doc` windowed pos 8919 as T=4 vs T=3, with identical
whole-run output. Ordinal alignment is therefore not a valid contract. Windows
are matched by `(pos, T, input_hash)`; every matched window must have equal
emitted IDs and satisfy the family contract (bitwise) on its full logit rows;
at least **12** matched windows (`Bounds.min_matched_windows`) are required;
duplicate keys reject; unmatched windows are counted, never compared, and make
coverage incomplete (no full proof). Whole-run prompt/output token identity is
still required. Replay of the session-1 exports under this rule: 16/16/16/14
matched windows, all bitwise. Short output follows inherited eligibility rules and
is saved, not selectively topped up. Forced follow tokens are not a fallback
for proving natural output or quality parity.

Native CUDA-event timings distinguish direct launch and synchronized warmed
graph replay from host wall-clock capture/instantiate cost, excluding declared
input-copy/setup work consistently in both arms. Replays must include changed
inputs in correctness checks, not reuse stale outputs as evidence. Clean
fresh-process model decode **includes cold graph costs**: report it as cold CLI
throughput, not warmed-native or steady-state model throughput. Identical IDs
alone do not establish equal T distribution. Capture topology differences
intrinsic to fusion are recorded, not subtracted into an invented steady-state
speedup; timing feedback in DraftPolicy can also change future draft work.
If attribution is unresolved, report the limitation and withhold the controlled
model claim.

Save all raw paired samples/order. Report paired deltas/ratios, per-arm medians,
spread (range and the declared spread statistic), n and exclusions; do not
replace pairwise analysis with unrelated medians. Three pairs are a bounded
minimum, not automatic statistical significance. No universal noise threshold,
quality improvement or historical all-PR gain is inferred from this probe.

## Host rejection gates

Only the primary may later perform the authorized host stage. This document
task performs **no SSH, GPU, live, setup or model-download operation**.
Before any such later action, primary must read the homelab repository's
`AGENTS.md` and `docs/architecture/overview.md`, run its required
`scripts/agent-preflight.sh`, and obtain fresh target-local evidence.

A sanctioned `hl peek host nibbler summary` alone covers **CPU/RAM/disk**, not
GPU/process/service/launcher state. It cannot admit .10 or .11. Require fresh
GPU identity/driver/architecture, memory/process inventory and utilization,
RAM/disk capacity for declared allocations/artifacts, service/process and
launcher checks, plus source/dependency/build provenance. Reject if:

- Either mandatory harness-hardening gate lacks accepted evidence.
- Preflight fails/cannot run; GPU has unexpected users/work or **2 GiB or more
  used before testing**; available resources are inadequate. Never kill or
  stop unrelated jobs to manufacture availability.
- Qwen is unexpectedly loaded; the production launcher guard is changed or
  executable; unexpected inference or restoration timers exist. Historical
  `production.launcher` mode644 and saved hashes are expectations to compare,
  not assertions about today's state.
- Production source/binary/config/server/launcher hashes differ from the
  approved baseline without explicit explanation/authorization, required
  read-only model/dependencies are missing, or configuration would trigger a
  fetch/setup/download. Pin/reuse existing dependencies without network access.
- A parity target fails, required new CUDA sections skip/execute no work,
  an error is unchecked, or the supervised timeout/owned-process cleanup
  cannot be guaranteed. Do not proceed to performance arms.

Use only the isolated host checkout/build and fresh timestamped artifact root.
No edits to `/data/llm/Strata` production source, binary, config, server or
launcher; no fork replacement/reapply, deployment or promotion. Preserve the
**Qwen offline launcher guard; no timers**. Audio and router remain untouched;
verify their recorded baseline health before/after without changing them.
At exit verify owned units/process groups inactive and GPU returned to the
observed baseline, not forcibly cleared. No Tier-C KV/storage/serving/unified
KV/handoff/admission work, setup/update scripts, model downloads, GitHub
fetch/push/PR, commits/amends, branches/worktrees or extra agents are in scope.

## Artifact manifest and disposition

Before execution, write the resolved host manifest into the fresh
`/data/llm/Strata-tests/tier-b-*` artifact root (primary's later task). The
foundation job already produced `/data/llm/Strata-tests/tier-b-foundation-20261007`
(local copies `/tmp/strata-tier-b-artifacts/foundation`); the full `.10` manifest
and post-change rebuild remain future.

- Engine base/fork/Tier-A pins, exact Tier-B source+patch identity, build/source
  tree hashes and clean/dirty provenance; compiler/CUDA/dependency versions,
  cache/settings and every executable hash. No invented ggml git ref when the
  reused source has no embedded `.git`; use source hashes/version evidence.
- Read-only model pack/native manifests/shards/MTP/profile/tokenizer paths and
  checksums, prompt IDs/hash/counts, resolved argv/environment allowlist,
  placement/cache/residency/PCIe settings, sampling/speculation/KV settings.
- Fresh preflight/GPU/process/service/guard/resource snapshots and production
  content checksums, before/after; bounds/schedule/rotation/L2/allocation bytes.
- Configure/build/CTest commands, logs, exit codes, required case counts,
  skips/failures/tolerances; direct versus changed-input graph evidence.
- Separate diagnostic records/completed-window metadata/full-logit rows and
  comparison results, explicit export limits/truncation/status; rejected cases.
- Separate native direct/warm-replay/capture datasets and clean cold-model
  records with paired order, IDs/output hashes, window/capture metadata or
  stated clean-run observability limits, counts/raw samples/stats/spread.
- Owned-unit cleanup and baseline postrun state. Recompute statistics from raw
  artifacts for .12; independent final audit follows the child tasks.

Disposition now: f/h are **source-eligible and foundation-proof at `8b375cc7`**,
not production parity; their parity gates passed the sta-osh.13 foundation GPU
run, but benchmarks are not compiled/timed and no model arm has run. Harness
hardening is committed (`2612b143`/`8eb80858`) with offline evidence. Other
families are deferred or inapplicable as qualified above. No performance/
promotion recommendation is made. The future report must mark passed, failed,
skipped, rejected and pending evidence separately and retain uncertainty,
without claiming general production feature or quality certification.

## Offline document verification

For sta-osh.3 only: check all Markdown relative links/fragments, explicit
anchors, source symbol references, local commit pins and fixture counts;
crosscheck any historical quantitative statements against saved Tier-A logs/
JSON. Future filenames/commands awaiting implementation remain labelled planned.
Check this document's whitespace and run the inherited **offline** fast suite
with the current regex dependency path after every edit:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/pete/.local/share/uv/tools/nanobot-ai/lib/python3.11/site-packages \
python3 -m unittest discover -s tools/tier_a -p 'test_*.py' -q

git diff --check -- docs/NIBBLER_TIER_B_PLAN.md
```

Require zero regex-dependency skips; the current committed harness runs
**168 tests, 0 skips**, plus 59 saved-capture validations, under this exact
regex path. Report actual test counts and any failures; these offline checks
never execute CUDA.
Because a new untracked file is absent from ordinary `git diff`, also inspect
its no-index diff/whitespace without staging it. Concurrent helpers' changed
files are not part of sta-osh.3 and must not be edited or reverted. No Beads
mutation, commit/amend/push/fetch or live gate is performed by this task.
