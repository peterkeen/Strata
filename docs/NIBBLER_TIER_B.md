# Nibbler Tier-B first-wave kernel report

**Bench host:** nibbler (bare-metal AMD Ryzen 9 9900X, RTX 5060 Ti 16 GB).
**Orchestration:** paseo-nibbler VM (offline tests, harness, artifact copies).
**Dates:** 2026-10-07 (foundation and final native gates, US Eastern) to
2026-10-08 (measurement, then audit remediation and revalidation).
**Scope:** focused same-binary validation of two upstream kernel families
already present in engine `e8ca9af`. This is not a port, a merge, a production
change or a quality certification. It covers one GPU and the single-session CLI
with fresh processes. It does not cover Tier-C serving/KV work.

An independent final audit of this epic returned twelve findings; all were
fixed on this branch and a revalidation stage followed. The finding-to-commit
map, the corrected-validator replay results, the corrected host build and the
unchanged limits are in
[§13](#13-audit-remediation-and-corrected-evidence-disposition). No measured
number in §5–§7 changed, and the revalidation is **not** a performance or
production-readiness claim.

The original protocol, contracts and bounds were declared beforehand in the
[Tier-B plan](NIBBLER_TIER_B_PLAN.md). Window-key matching and its minimum of
12 matched windows were chosen **after session 1's proof rejection**, before
any model timing ran; they were not originally predeclared. See
[deviations](#8-deviations-from-the-plan).

## Contents

1. [Summary](#1-summary)
2. [Pins, hardware and toolchain](#2-pins-hardware-and-toolchain)
3. [Families under test](#3-families-under-test)
4. [Native correctness gates](#4-native-correctness-gates)
5. [Warm native kernel timings](#5-warm-native-kernel-timings)
6. [Model proof: reachability and bitwise windows](#6-model-proof-reachability-and-bitwise-windows)
7. [Cold-process model timings](#7-cold-process-model-timings)
8. [Deviations from the plan](#8-deviations-from-the-plan)
9. [Deferred families and gaps](#9-deferred-families-and-gaps)
10. [Disposition](#10-disposition)
11. [Reproduction recipes](#11-reproduction-recipes)
12. [Artifacts and provenance](#12-artifacts-and-provenance)
13. [Audit remediation and corrected-evidence disposition](#13-audit-remediation-and-corrected-evidence-disposition)

---

## 1. Summary

| Evidence rung | f: fused norm+RoPE | h: native sub16 GU/down |
| --- | --- | --- |
| Native bitwise gates (direct + changed-input graph replay) | **Passed**: 192 cases, 576 graph launches | **Passed**: all 7 manifest format pairs, tails/empty/scatter |
| Warm native kernel timing (isolated) | Direct launch: fused/split group medians 0.35–0.58× (per case 0.31–0.72×) | sub16 takes 0.72–0.81× the reference time, with a working set larger than L2 |
| Model reachability (dispatch trace) | Fused path selected on A, split path on B | `split_sub16` on A, `split` on B; GU 384 and down 384 records each |
| Model proof (bitwise logits, matched windows) | **Passed**, 2/2 workloads, 16/16 windows each | **Passed**, 2/2 workloads, 16/16 windows each |
| Cold-process model decode, paired B/A | No resolvable effect: 0.99–1.02 | long_doc: opt-out faster in 3/3 pairs (1.019–1.022); short_code: unresolved (0.94–1.05) |

- **Correctness:** both families are bitwise identical to their reference paths at
  the primitive level and in the model (matched windows, plus whole-run output IDs).
- **Model speed:** the native speedups do **not** show up as a measurable decode
  gain in cold-process model runs.
- **sub16 on long_doc:** the opt-out was faster by about 2% in all three pairs.
  With n=3 this is a lead for follow-up, not a demonstrated regression.
- **Recommendation:** no default change, no promotion and no production action
  ([disposition](#10-disposition)).

## 2. Pins, hardware and toolchain

| Item | Value |
| --- | --- |
| Engine base | upstream `e8ca9afd03d8` (0.1.40.2), already compiled in Tier A |
| Original fork (historical only) | `9921eecae9d4` (0.1.39); recorded production HEAD in gate toolchain and session post-state snapshots, not continuous source-tree monitoring |
| Final native build source | `3af1e4c8` (all `src`/`include`/`CMakeLists.txt`/`cmake` changes) |
| Measurement harness source | `222d39ea`; r2 provenance records `source_dirty=false` at capture; `git diff 3af1e4c8 222d39ea -- src include CMakeLists.txt cmake` was empty (checked in-session) |
| Build dir | `/data/llm/Strata-tier-b/build-tier-b` (isolated clone; r2 script checked out detached `222d39ea`, following the earlier `perf/nibbler-tier-b-kernels` work) |
| `strata` sha256 | `59f0d53627773b3dac02fac861421441ee11848d27a0508216e905e984c9a7f7` |
| `tier_b_rope_bench` sha256 | `82467382094a5120b95de98632e7758e6d19a79f6211c8ccf6d08f496eafa42f` |
| `tier_b_expert_bench` sha256 | `6d1bcd3a463674564d6201625a40304242d240a7e8b8332e85ae1f886ad841a2` |
| Compilers | CUDA 13.3 (V13.3.73), g++-15 15.3.0, Release, `CMAKE_CUDA_ARCHITECTURES=120` |
| ggml source | read-only reuse via `STRATA_GGML_DIR=/data/llm/Strata/third_party/llama.cpp` (no fetch); configure reports `ggml version: 0.24.0`. The tree has no embedded `.git`, so the configure-time "ggml commit 9921eeca" is the enclosing production checkout's HEAD, **not** a ggml ref. No separate content hash of that tree was captured. |
| GPU (historical pre-run snapshot, 2026-10-08) | NVIDIA GeForce RTX 5060 Ti, UUID `GPU-a29bb3c9-…`, driver 610.57.04, 16311 MiB, 36 SMs, L2 32 MiB (33554432 B) |
| Recorded GPU state before / after (r2) | 2 MiB used, 0 %, SM 180 MHz, mem 405 MHz, 31 °C, 3.95 W / 2 MiB used, SM 2842 MHz, 47 °C; final compute-process query lists no entries. These are snapshots, not enforced final GPU postconditions ([§11](#historical-safeguards-enforced-vs-observed)). |
| CPU / RAM / OS | Ryzen 9 9900X (AVX-512), about 123 GiB RAM: from the [Tier-A record](NIBBLER_TIER_A.md#2-hardware-and-software-state); not re-captured |

All benchmark/measurement binaries were verified against the hashes recorded in
the final gate run (`sha256sum -c` in-session) before use.

The later corrected rebuild (source `62bf499d`) produced **different** binary
hashes from the three above. Every timing table in §5 and §7 comes from the
historical binaries listed here, not from the corrected build; the two sets are
kept separate in [§13.3](#133-corrected-isolated-build-and-correctness-gates)
and [§13.4](#134-binary-separation).

## 3. Families under test

Both families are already upstream and **on by default**. Each pair runs the
same executable, recorded input paths and requested settings, and exactly one
opt-out variable differs. Input content identity has the limits in
[deviation 2](#8-deviations-from-the-plan):

| Family | Arm A (default) | Arm B (opt-out) | What differs |
| --- | --- | --- | --- |
| f: fused native RMSNorm+RoPE | unset | `STRATA_NO_NORM_ROPE=1` | Fused per-head norm+rotation versus split norm then RoPE |
| h: native grouped sub-warp layout | unset | `STRATA_NO_SUB16_GU=1` | 16-lane GU rows **and** down sub-warps versus the plain split layout |

The other family's control stays unset. There is no all-off arm, and
`STRATA_NO_MULTI_GR` and the older kernels were not touched.

**Historical-down qualification:** the original fork `9921eec` already contains
`row_dot_40_sub8` and `row_dot_20_sub16` (the down sub-warp kernels), but not
`row_dot_80_sub16`, `native_qsa_rms_norm_rope` or either opt-out variable
(checked with `git grep` at both pins). Arm B is therefore **not** "the
production fork". The pair isolates only the h layout selection, and its result
cannot be read as the full historical PR-h delta or as an upstream-versus-fork
comparison.

## 4. Native correctness gates

**Foundation gate** (sta-osh.13, source `8b375cc7`, 2026-10-07): CTest 3/3 passed
with 0 skips: `rope_parity`, `iq_multi_parity`, `native_grouped_parity`. This was
superseded by the final gate below.

**Final gate** (sta-osh.10, source `3af1e4c8`, unit
`strata-tier-b-final-gates-20261008`, started 2026-10-07 21:33 local, exit 0).
An earlier attempt (unit `strata-tier-b-final-gates-20261007`, source
`19067c1b`) failed at compile time, because the rope bench used
`cudaDeviceProp::clockRate`, which CUDA 13 removed. No gate ran in that attempt;
its artifacts are kept in `/data/llm/Strata-tests/tier-b-final-gates-20261007`.

- **CTest:** 15/15 passed, 0 skipped: `rope_parity`, `iq_multi_parity`,
  `native_grouped_parity`, `native_multi_parity`, and 11 `native_expert_parity_*`
  cases.
- **`rope_parity`** (section 6): 192 cases (T1..8 × head 128/256 × contiguous,
  in-place and strided q|gate × table and analytic × scaling), 576 graph launches,
  384 of them changed-input/position replays. All are bit-identical between fused
  and split, with no errors on guards, padding or normalized tails.
- **`iq_multi_parity`** (sub16): all 7 manifest pairs at 2560×640, group tails
  1/2/3/4/5/8/11, empty groups and scattered destinations. Direct and A/B/empty
  graph replays are bitwise equal.
- **Trace under `STRATA_TIER_B_TRACE=1`:** `rope_parity` emitted 511
  `norm_rope_fused` records and hit the 1024-record bound (`trunc=1`), so it is
  prefix coverage only. `iq_multi_parity` traced `split_sub16` 44+44,
  `split` 36+36 and `reference_old` 24+24 (GU+down).

The reference tolerances are the gates' own source contracts and are unchanged:
rope host rotation 1e-6 relative, fast-trig versus table 3e-3; IQ double-dot
gross-error guard below 1e-2 relative. Fused versus split and sub16 versus
reference comparisons are **bitwise** (zero tolerance).

**Corrected rebuild gate** (sta-osh.28, source `62bf499d`, 2026-10-08): after
the audit fixes the same 15 CTest cases were rebuilt and rerun on the host and
all 15 passed again, with the new CUDA-unwind and allocation-budget paths
exercised. Details, hashes and the two added live probes are in
[§13.3](#133-corrected-isolated-build-and-correctness-gates).

## 5. Warm native kernel timings

These are **isolated warm-kernel latencies**, not model throughput. Both runs
used the same measurement session, unit `strata-tier-b-measure-20261008-r2`.

### 5.1 Fused norm+RoPE

- **Command:** `tier_b_rope_bench 32 10 3 1 8`.
- **Run:** 96 cases, 96 timed, 0 gate or replay failures. Every case was
  `IDENTICAL` (bitwise) before timing.
- **Arguments:** 32 iterations per sample, 10 warmup, 3 samples per arm, AB/BA
  alternated. Canonical inputs were re-uploaded outside the event window before
  each timed arm. Each figure is CUDA-event time divided by 32 launches.

Median fused/split ratio per group of 16 cases (T1..8 × table/analytic),
recomputed from `rope-bench.log`:

| Head dim | Layout | Direct fused (µs) | Direct split (µs) | Direct F/S median [range] | Graph F/S median [range] |
| --- | --- | --- | --- | --- | --- |
| 128 | contiguous | 1.34–1.73 | 2.56–2.75 | 0.549 [0.504–0.660] | 0.515 [0.512–0.517] |
| 128 | in-place | 1.34–1.73 | 2.59–2.98 | 0.545 [0.503–0.650] | 0.515 [0.511–0.515] |
| 128 | q\|gate | 1.31–1.73 | 4.16–4.32 | 0.347 [0.315–0.413] | 0.506 [0.506–0.513] |
| 256 | contiguous | 1.38–1.95 | 2.56–2.75 | 0.576 [0.518–0.717] | 0.515 [0.512–1.002] |
| 256 | in-place | 1.38–1.95 | 2.56–2.72 | 0.572 [0.519–0.717] | 0.515 [0.512–1.000] |
| 256 | q\|gate | 1.38–1.91 | 3.78–4.22 | 0.365 [0.329–0.501] | 0.506 [0.505–0.986] |

Caveats:

- **Graph replay floor:** per-launch graph times cluster at about 2.10 µs
  (fused) and about 4.08 µs (split) across almost all shapes. *Inference:* this
  reflects a per-node graph-replay floor, so it measures how many kernels each
  graph launches rather than arithmetic cost. The six T=8/head-256 cases are the
  exception: fused graph time rises to about 4.09 µs and graph F/S is
  0.90–1.00.
- **Capture and instantiate:** these were measured separately on the host clock
  and are 0.001–0.012 ms after the first case. The first case pays one-time
  costs (capture 0.029 ms, instantiate 0.062 ms).
- **Sample count:** each figure has n=3 samples. These are latency figures, not
  a statistical result.

### 5.2 Native sub16 expert layout

- **Command:** `tier_b_expert_bench --warmup 10 --samples 3 --max-bytes 536870912`.
- **Run:** 0 parity failures, 7 pairs timed.
- **Shape:** G=8 groups × T=2 entries per call, rotating across 8 slots of
  distinct weights.
- **Cache state:** before each timed arm, an untimed prime pass touches the
  other slots, and the two arms of a pair use the same slot.
- **Working set:** larger than the 32 MiB L2. This bounds L2 residency but does
  **not** guarantee each call starts cold.
- **Layout constants:** per-column decode was off; codebook staging and rows2
  were at their defaults.

| GU / down | Working set (B) | Direct sub16 / ref (µs, median) | Direct ratio | Graph sub16 / ref (µs) | Graph ratio | Capture+instantiate sub16 / ref (µs) |
| --- | --- | --- | --- | --- | --- | --- |
| IQ3_XXS / IQ4_NL | 139264000 | 53.25 / 71.68 | 0.743 | 53.25 / 69.63 | 0.765 | 11 / 50 |
| IQ3_XXS / Q2_0 | 109772800 | 49.15 / 63.49 | 0.774 | 49.15 / 63.49 | 0.774 | 4 / 18 |
| IQ2_S / IQ4_NL | 126156800 | 53.25 / 73.73 | 0.722 | 53.25 / 73.73 | 0.722 | 4 / 15 |
| IQ2_S / Q2_0 | 96665600 | 51.20 / 65.54 | 0.781 | 51.20 / 65.54 | 0.781 | 4 / 18 |
| IQ3_S / IQ4_NL | 149094400 | 55.30 / 75.78 | 0.730 | 55.30 / 75.78 | 0.730 | 5 / 26 |
| IQ3_S / Q2_0 | 119603200 | 53.25 / 67.58 | 0.788 | 53.25 / 67.58 | 0.788 | 4 / 25 |
| IQ4_XS / IQ4_NL | 170393600 | 59.39 / 73.73 | 0.806 | 59.39 / 73.73 | 0.806 | 5 / 33 |

Caveats:

- **Timer resolution:** every single-call sample is an integer multiple of about
  2.048 µs. The timer step is therefore 3–4% of each value, and the three-sample
  ranges are mostly zero width.
- **Unproven shape:** G8/T2 is a declared representative and was **not** proved
  to be the model's actual group distribution. The model traces record only
  nominal launch capacities (`grid_groups` 4–60), not executed groups
  ([§6](#6-model-proof-reachability-and-bitwise-windows)).

## 6. Model proof: reachability and bitwise windows

- **Run:** diagnostic run, 8 arms in 214 s, `PASS`; summary in `diag/summary.json`.
- **Configuration:** same executable and fixed settings as timing
  ([§7](#7-cold-process-model-timings)), plus `STRATA_TIER_B_TRACE=1` and the
  bounded native window/logit export. These runs were **never** used as timing
  samples.

| Family / workload | Trace A | Trace B | Matched windows (pos, T, input hash) | Rows bitwise | Whole-run IDs | Full proof |
| --- | --- | --- | --- | --- | --- | --- |
| f / short_code | `norm_rope_fused` 152, `rope_inplace32` 18 | `rope128` 170 | 16/16, unmatched 0/0 | yes | equal (256 tokens, `74c3c9a95cf483bf`) | no (export cap) |
| f / long_doc | `norm_rope_fused` 153, `rope_inplace32` 9 | `rope128` 162 | 16/16, unmatched 0/0 | yes | equal (512 tokens, `0fa10a6bc706d91e`) | no (export cap) |
| h / short_code | `split_sub16` GU 384 + down 384 | `split` GU 384 + down 384 | 16/16, unmatched 0/0 | yes | equal (256 tokens, `74c3c9a95cf483bf`) | no (export cap) |
| h / long_doc | `split_sub16` GU 384 + down 384 | `split` GU 384 + down 384 | 16/16, unmatched 0/0 | yes | equal (512 tokens, `0fa10a6bc706d91e`) | no (export cap) |

What the proof covers:

- **Exported windows:** each arm exported its first 16 natural verify windows
  (bound 16 windows / 64 MiB), with full 248320-wide logit rows totalling
  46.4–55.9 MiB. The manifests carry `truncation 1` and `complete 1`.
  Everything after window 16 is covered only by the whole-run output-ID match,
  **not** by logit comparison. That is why every cell is "proof, not full proof".
- **Window shapes:** T counts per arm were short_code {1:2, 3:1, 4:12, 6:1} in
  all four arms. long_doc was {1:3, 2:1, 3:4, 4:8} in both f arms and
  {1:3, 2:1, 3:3, 4:9} in both h arms. Within each pair, A and B had identical
  T sequences in this run. The two long_doc pairs differed from each other,
  even though their A arms are the same default configuration, which is the
  draft-length variation described in [deviation 1](#8-deviations-from-the-plan).
  Logit export sizes were 55.9 MiB (short_code), 46.4 MiB (f long_doc) and
  47.4 MiB (h long_doc) per arm.
- **Trace semantics:** trace records are dispatch decisions made **during graph
  capture** (`capture=1` for all 930–938 records per arm), with zero truncation.
  They show which layout each captured graph contains, not per-replay execution.
  Execution is corroborated by the completed exported windows.
- **Formats seen:** h records cover all 7 manifest format pairs (type IDs
  18/20 224, 22/20 240, 21/20 144, 22/42 80, 18/42 48, 21/42 16, 23/20 16
  records per arm).
- **Nominal capacities only:** `grid_groups` in the traces (4, 10, 20, 30, 40,
  60) are launch capacities, not executed group counts.
- **Remaining split calls in f arm A:** 9–18 `rope_inplace32` calls at rows=2,
  head 256 remain unfused. This is consistent with the plan's note that not
  every Q/K site is fused (for example, side branches and MTP multi-token K).
  The default arm is therefore *mostly* fused, not entirely.
- **Rope table:** `STRATA_ROPE_TABLE` was not enabled, so all traced rope
  records have `table=0`.

**Verdict source.** The four verdicts in the table above were first produced by
the original (pre-audit) validators. The audit found four defects in those
validators (H1–H3, M9). The **same retained export bytes** were replayed
offline through the fully corrected validators and all four r2 cells still pass
bitwise at 16/16 matched windows, still `full_proof=false`; the session-1
exports pass at 16/16/16/14. Counts, byte hashes and limits are in
[§13.2](#132-retained-logit-replay-through-the-corrected-validators). The
surviving coverage limit is unchanged: the historical 16-window truncated
prefix.

## 7. Cold-process model timings

**Run:** proof-gated timing, unit `strata-tier-b-measure-20261008-r2`, 32
launches in 832 s.

**Protocol:**

- Every arm is a fresh process.
- Each family/workload cell had 1 excluded preparatory A/B pair, then 3
  interleaved recorded pairs.
- Order was AB/BA/AB within each cell, and cell order rotated per round. The
  full order is in `timing.log`.
- Diagnostics and readbacks were off.

**Requested settings:** argv was identical across arms apart from the temporary
token-file path. Requested cache slots are not the resolved resident count.

```
--max-new 256|512 --pack /data/llm/Strata-data/packs/iq3_s
--native …/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
--ple-gguf …/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
--expert-profile /data/llm/Strata/data/expert-profile.bin --mtp /data/llm/Strata-data/mtp/rt
--expert-cache 2000 --pcie-frac 0 --adapt-every 0 --prefill auto --spec 4 --spec-min-p 0.5
--max-context 262144 --kv-resident 32768 --kv int8 --vram-reserve-mib 2048 --stats --stop-eos
```

**Environment:** every ambient `STRATA_*` variable was stripped, then only
`STRATA_PF_FUSED=1` and `STRATA_IQ_MT_MIN=1` were set, plus the one opt-out on
B arms. Sampling was greedy.

**Observed runtime settings:** all 40 r2 model arms (8 diagnostic and 32 clean,
including preparatory arms) report the following in `diag/arms/*/stderr.txt`
and `timing/arms/*/stderr.txt`. These are engine startup/prefill observations,
not newly measured settings or content-identity checks. Concrete examples are
`timing/arms/normrope.{short_code,long_doc}.rec0.A/stderr.txt` under the r2
artifact root in §12.

| Setting | Resolved value in r2 logs |
| --- | --- |
| Expert cache | **2582 slots**, 4.96 GiB VRAM, pre-filled 2582/2582 from the profile; argv requested **2000**. PROFILE policy, ranked by routing frequency, no eviction; R4 GPU hit path on; token graph reports 2582 resident experts, decided on-device. |
| CPU pool | 36 tasks/phase (automatic), 12 participating threads: 11 workers on logical processors 1–11, host thread on 0 (also draining). |
| KV placement | 32768/262144 cells per QSA layer in VRAM, remaining K/V in 3.09 GiB pinned RAM. `--kv int8` is the recorded argv format. |
| Prompt chunk and cache loan | short_code: auto chunk 256, borrows/refills 311 slots (0.59 GiB), 41 prefill tokens in 1 chunk; long_doc: auto chunk 8192, borrows/refills 2075 slots (3.97 GiB), 8886 prefill tokens in 2 chunks. |
| Draft/verify | Draft head over 106299 tokens (212.9 MiB); verify cap T=6, 75.8 MiB device buffers. The MTP/lookup distinction remains as described in §7.3. |
| Expert arena and sampling | Buffered file-cache reads; `cudaHostRegister PORTABLE ok`; explicit huge pages unavailable (`vm.nr_hugepages=0`), THP reported always on; greedy sampling. |

No separate full non-STRATA environment/CUDA-cache-state manifest was retained.
The script records its PATH setup and the logs record CPU-pool placement, but
neither proves every ambient runtime setting was identical between arms.

**Prompts:** short_code was 42 IDs (hash `563cf85e6f7c1247`, 256 new tokens) and
long_doc was 8887 IDs (`0157820299822d1b`, 512 new tokens).

**Run validity:**

- All 32 arms exited 0 with no errors.
- Within each workload, every arm produced the same output hash and length.
- Draft acceptance was 0.931 on short_code in all arms and 0.668–0.673 on
  long_doc. Within-pair rates and T histograms were **unequal in 5/6 recorded
  long_doc pairs**; output identity does not establish identical draft work
  ([§7.3](#73-clean-run-draft-statistics-and-timing-feedback)).

**Cold-process caveat:** these figures include first-use CUDA graph capture and
compilation. They are **cold fresh-process CLI** numbers, not steady-state
or warmed-model throughput.

### 7.1 Per-arm medians (n=3 recorded arms each)

```
[short_code]
normrope.default  decode=47.51tok/s(n=3)  prefill=68.79tok/s(n=3)
normrope.no_norm_rope  decode=46.98tok/s(n=3)  prefill=67.77tok/s(n=3)
sub16.default  decode=47.01tok/s(n=3)  prefill=68.32tok/s(n=3)
sub16.no_sub16_gu  decode=47.71tok/s(n=3)  prefill=69.26tok/s(n=3)
[long_doc]
normrope.default  decode=43.26tok/s(n=3)  prefill=1476.46tok/s(n=3)
normrope.no_norm_rope  decode=43.16tok/s(n=3)  prefill=1476.34tok/s(n=3)
sub16.default  decode=41.67tok/s(n=3)  prefill=1472.97tok/s(n=3)
sub16.no_sub16_gu  decode=42.55tok/s(n=3)  prefill=1477.70tok/s(n=3)
```

`normrope.default` and `sub16.default` are the **same configuration** measured in
different cells. short_code prefill has only 41 prompt tokens, so its prefill
rate is not a meaningful throughput figure.

### 7.2 Paired statistics (recomputed from raw `arm.json`)

Ratio = B/A per pair in recorded order. A ratio above 1 means the opt-out arm was
faster.

| Cell | Decode A (tok/s) | Decode B (tok/s) | Decode B/A per pair | Median | Prefill B/A per pair |
| --- | --- | --- | --- | --- | --- |
| f / short_code | 48.41, 43.54, 47.51 | 48.84, 44.57, 46.98 | 1.009, 1.024, 0.989 | 1.009 | 0.984, 0.974, 1.004 |
| f / long_doc | 43.26, 42.68, 43.82 | 43.10, 43.16, 43.43 | 0.996, 1.011, 0.991 | 0.996 | 0.975, 1.000, 1.000 |
| h / short_code | 48.71, 47.01, 45.30 | 45.87, 47.97, 47.71 | 0.942, 1.020, 1.053 | 1.020 | 1.001, 1.056, 1.019 |
| h / long_doc | 42.43, 41.67, 40.46 | 43.25, 42.55, 41.35 | 1.019, 1.021, 1.022 | 1.021 | 1.001, 1.003, 1.006 |

Decode stdev: f/short 2.59 (A) and 2.14 (B); f/long 0.57 and 0.18; h/short 1.71
and 1.14; h/long 0.99 and 0.96 tok/s.

**Noise context:**

- **Default arms across cells:** for the same default configuration, the six
  recorded decode values span 43.54–48.71 tok/s on short_code (10.9% of the
  median) and 40.46–43.82 on long_doc (7.9%).
- **h/long_doc absolute values:** that cell's A arms (40.46–42.43) sat below the
  f/long_doc A arms (42.68–43.82). Run-to-run or position drift in this session
  is therefore larger than most paired deltas. Pairing removes some of it, but
  three pairs cannot bound it.

Reading, with no significance claimed:

- **f:** no resolvable effect on cold decode or prefill in either workload.
- **h, long_doc:** B was faster in all 3 pairs by a tight 1.9–2.2% (absolute
  +0.82, +0.88, +0.89 tok/s). The direction is consistent across AB and BA
  order, but n=3 is a bounded minimum, not significance.
- **h, short_code:** unresolved; ratios span 0.94–1.05.
- **Native vs model:** the 19–28% native sub16 latency reduction
  ([§5.2](#52-native-sub16-expert-layout))
  does **not** carry through to cold-process decode in this probe. The cause is
  unresolved, and this report does not attribute it. The differing draft work
  and timing-dependent policy below further limit kernel-only attribution.

### 7.3 Clean-run draft statistics and timing feedback

Recomputed from the immutable local r2 copies under
`/tmp/strata-tier-b-artifacts/measure-20261008-r2/timing/arms/*/stdout.txt`,
not inferred from output hashes or diagnostic windows. Each `speculation`
line gives accepted/offered draft counts and a rate rounded to three decimals;
`window sizes` gives the **whole-run** histogram. Fractions below retain the
exact counts; displayed rates use the same rounding as stdout. A is default,
B is opt-out; `rec0`–`rec2` are the recorded pairs in §7.2, while `prep` is
excluded from timing statistics.

**short_code:** all 16 clean arms (both families, prep and rec0/rec1/rec2,
A and B) have accepted/offered **188/202 = 0.931**, 71 rounds, and
`(T1,T2,T3,T4,T5,T6) = (4,0,1,65,0,1)`. Thus rates and distributions match in
all six recorded short_code pairs (and both excluded prep pairs). This does
not establish equal window order.

**long_doc:** T1=27 and T2=33 in every clean arm; the remaining histogram
counts and acceptance statistics are:

| Family | Pair | Accepted/offered A | Rate A | Accepted/offered B | Rate B | T3/T4/T5/T6 A | T3/T4/T5/T6 B | Rounds A/B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f | prep (excluded) | 304/453 | 0.671 | 304/454 | 0.670 | 30/120/0/0 | 29/121/0/0 | 210/210 |
| f | rec0 | 304/455 | 0.668 | 304/452 | 0.673 | 28/122/0/0 | 31/119/0/0 | 210/210 |
| f | rec1 | 304/455 | 0.668 | 304/454 | 0.670 | 28/122/0/0 | 29/121/0/0 | 210/210 |
| f | rec2 | 305/453 | 0.673 | 304/454 | 0.670 | 27/122/0/0 | 29/121/0/0 | 209/210 |
| h | prep (excluded) | 304/455 | 0.668 | 305/453 | 0.673 | 29/120/1/0 | 27/122/0/0 | 210/209 |
| h | rec0 | 305/453 | 0.673 | 304/454 | 0.670 | 27/122/0/0 | 29/121/0/0 | 209/210 |
| h | rec1 | 304/455 | 0.668 | 304/455 | 0.668 | 28/122/0/0 | 28/122/0/0 | 210/210 |
| h | rec2 | 304/454 | 0.670 | 305/453 | 0.673 | 29/121/0/0 | 27/122/0/0 | 210/209 |

Five of six **recorded** long_doc pairs differ in both rate and distribution;
only h/rec1 matches. Both excluded long_doc prep pairs also differ. These
histograms count windows, not an ordered sequence of `(pos, T, input_hash)`.
Clean runs did not export that sequence, so they cannot establish window-by-window
alignment, inputs or capture order even when aggregate histograms match.
Diagnostic prefix sequences in §6 are a separate dataset, not substitutes.
This stdout-statistics correction does not itself revalidate retained logit
bytes; that was done separately by the corrected-validator replay
([§13.2](#132-retained-logit-replay-through-the-corrected-validators)), which
covers the diagnostic exports only, never the clean timing arms.

**Why timing can feed back into draft work:** in the measured source
(`222d39ea`; the following source files are unchanged from that pin),
[DraftPolicy](../include/strata/spec/draft_policy.hpp) learns expected committed
tokens per millisecond. In [draft_policy.cpp](../src/spec/draft_policy.cpp),
`observe` (and `observe_chain` for chained lookup) updates the cost for T from
`round_ms`: the first sample sets it, later samples use an EMA with weight 0.1.
`cost_ms` uses that measured cost, or scales from measured sizes for an unseen
T. `choose` compares MTP and candidate lookup windows by expected tokens/cost,
with a default 3% margin and probing/staleness rules. It is not a policy based
only on acceptance or fixed `--spec` settings.

The relevant **native CLI** caller is in
[generate.cpp](../src/program/generate.cpp): it constructs `policy(o.spec)`,
starts `t0` at the top of the round, obtains a suffix proposal and calls
`policy.choose(T, k, sfx_match)` when eligible. After verification, commit,
optional Tier-B wait/readback/hash/export, emitted-token bookkeeping and MTP
drafting, it computes `round_ms = Clock::now() - t0` in milliseconds and calls
`policy.observe(from_sfx, T, a, sfx_match, round_ms)` (or `observe_chain`),
excluding the initial native window from policy observations. Thus diagnostic
overhead within the interval, cold capture costs, kernel latency or other round
time variation **can change later draft choices**, even with identical output
IDs. `--adapt-every 0` disables expert-placement adaptation, not this feedback.
The default suffix drafter remains enabled: the CLI preserves an MTP cap of 4
for `--spec 4` but permits lookup windows up to T=6, explaining the T5/T6 bins.

This source mechanism makes drift plausible; the aggregate artifacts do **not**
identify which timing component caused any particular differing window. The
cold-model timing tables remain valid observed throughput, but neither their
small deltas nor the native/model gap can be assigned to an isolated kernel
cost from these runs. No steady-state gain or regression is established.

## 8. Deviations from the plan

1. **Window-matching contract** (amended before any model timing). Session 1
   (unit `strata-tier-b-measure-20261008`, source `3437d96f`, 246 s, exit 1)
   compared windows by ordinal.
   - **What failed:** h/long_doc failed at ordinal 13, where the same position
     was windowed as T=4 in one arm and T=3 in the other.
   - **What still matched:** whole-run IDs matched, and the other three cells
     passed.
   - **Mechanism and uncertainty:** fixed flags and identical output IDs do
     not freeze draft lengths. The measured-round-time feedback in DraftPolicy
     can change subsequent choices ([§7.3](#73-clean-run-draft-statistics-and-timing-feedback));
     the retained data does not identify the cause of this particular drift.
   - **Response:** per the no-retry rule, timing did not start. The contract
     changed to matching by `(pos, T, input_hash)` with at least 12 matched
     windows, bitwise on matched rows, and whole-run IDs still required (commit
     `222d39ea`, 9 regression tests). **Both the window key and minimum 12 were
     chosen after seeing session 1's rejection, not originally predeclared.**
     Replaying the session-1 exports under the new rule gave 16/16/16/14 matched,
     all bitwise. The later corrected validators reproduced the same
     16/16/16/14 result on the same retained bytes
     ([§13.2](#132-retained-logit-replay-through-the-corrected-validators)).
   - **Rerun:** session r2 then passed. This is the only contract changed after
     seeing host data, and it relaxes window alignment only, not the bitwise
     identity requirement.
2. **Input and dependency content identity:** r2 `diag/provenance.json` and
   `timing/provenance.json` record equal paths and `size`/`mtime_ns` for the
   GGUF shards, expert profile, pack and MTP directory at those two run-level
   captures, not content sha256. The 54.8 GB and 28.8 GB shards (54817524224
   and 28800138432 B) and the 196632 B expert profile were not content-hashed
   in these records. Pack and MTP each have directory stat size 4096 B:
   **directory stat identity does not establish unchanged child contents**;
   modifying an existing child can leave the directory's stat unchanged.
   No child-file/tree content hashes were captured for the pack or MTP, and
   there is no per-arm or continuous input-content verification. The reused
   ggml dependency has a recorded path/version but no tree content hash
   ([§2](#2-pins-hardware-and-toolchain)); production HEAD is not its content
   identity. Fixed paths and matching metadata cannot certify unchanged model
   or dependency bytes.
3. **Representative expert groups:** the plan asked for "proved representative
   groups". The bench used a declared G8/T2 shape, and the model traces carry
   only nominal capacities ([§5.2](#52-native-sub16-expert-layout)).
4. **Session scripts:** neither the measurement script nor the final-gate
   heredoc is committed. The r2 measurement script was `/tmp/tier-b-measure.sh`
   (sha256 `00dea19031297350106d3322dafacb5d3e5eb7a8a102ff55fe36468dfc558a08`,
   on paseo-nibbler and on the host). Their essential steps are reproduced in
   [§11](#11-reproduction-recipes). The later revalidation script
   `/tmp/tier-b-remediation/host-remediation.sh` is likewise uncommitted; its
   evidence is in [§13.3](#133-corrected-isolated-build-and-correctness-gates)
   and [§13.5](#135-enforced-host-preconditions-and-postconditions).
5. **Validators corrected after the measured runs.** The §6 proof verdicts were
   originally computed by validators the audit found defective (H1–H3, M9).
   The retained bytes were replayed through the corrected validators and the
   verdicts held ([§13.2](#132-retained-logit-replay-through-the-corrected-validators)).
   The corrected admission protocol also makes the historical, unstamped
   diagnostic summaries **inadmissible** for any future timing run
   ([§13.6](#136-what-the-revalidation-does-not-establish)).

## 9. Deferred families and gaps

The other upstream families (a, b, c, d, e, g, i, j, k) were **not measured**. Their
deferral reasons are in the plan's
[other kernel families](NIBBLER_TIER_B_PLAN.md#other-kernel-families) table:

- **No narrow control:** b, c, e and g have no narrow, isolating control (for
  example, `STRATA_NO_MULTI_GR` changes too much).
- **Storage boundary:** d needs a storage-boundary review that belongs to Tier C.
- **Unproven reachability:** a, i and j lack executed-path or lifecycle proof.
- **Inapplicable:** k does not apply to the native IQ3_S path.

The CPU gather knob (`STRATA_IQ256_GATHER`) remains inapplicable on this AVX-512
host, as in Tier A. No claim is made for any of these families.

Open gaps for the first wave:

- **Logit proof coverage:** limited to the first 16 windows per arm. Later
  windows rely on output-ID identity.
- **Timing in clean runs:** whole-run T histograms and accepted/offered counts
  are available in raw stdout ([§7.3](#73-clean-run-draft-statistics-and-timing-feedback)).
  Rates and distributions differ in 5/6 recorded long_doc pairs; output hashes
  still match. Ordered per-window sequences are absent, so window alignment
  and equal capture order are not established even for matching histograms.
  Timing feedback in DraftPolicy limits kernel-only attribution.
- **No warm or steady-state model timing:** no profiler (nsys) or kernel-share
  breakdown of model decode was taken, so the gap between native and model
  results is unexplained.
- **Thin statistics:** n=3 pairs per cell, with a single session and a single
  day.
- **Provenance and host-state coverage:** missing model/pack/MTP/profile/ggml
  content hashes and snapshot-only state evidence limit identity claims.
  Final GPU/router state was recorded but not exit-gated in r2; post-run audio
  health is absent ([§11](#historical-safeguards-enforced-vs-observed)). The
  later revalidation stage did enforce those postconditions fail-closed
  ([§13.5](#135-enforced-host-preconditions-and-postconditions)), but that
  does **not** retroactively gate the r2 measurement runs, and no model or
  benchmark arm was re-timed.

## 10. Disposition

| Family | Correctness | Model effect (cold) | Disposition |
| --- | --- | --- | --- |
| f fused norm+RoPE | Bitwise at primitive and model level (proof, coverage-limited) | None resolved | Keep the upstream default. No action. |
| h native sub16 | Bitwise at primitive and model level (proof, coverage-limited) | long_doc: opt-out about 2% faster in 3/3 pairs; short_code unresolved | Keep the upstream default **for now**. Open a focused follow-up before any change. |

- **No promotion:** no promotion or default change is recommended. Both
  families are already part of upstream `e8ca9af`.
- **What the evidence shows:** it supports correctness identity. It neither
  supports nor refutes a model-speed benefit.
- **sub16 lead:** the long_doc result is a lead only. Confirming it would need
  more interleaved pairs, ideally a warmed or steady-state model measurement and
  a decode kernel-time breakdown, *before* touching the default.
- **Out of scope:** nothing here certifies output quality, serving parity,
  production readiness or the historical all-PR gain.
- **Post-audit status:** the twelve audit findings are fixed and the retained
  evidence was revalidated, so the two dispositions above stand on corrected
  validators. Neither disposition is strengthened: the corrected build was
  never timed, so "no action" and "follow-up before any change" are unchanged
  ([§13](#13-audit-remediation-and-corrected-evidence-disposition)).
- **Recorded production state:** all four saved session post-state files
  report the four baseline production hashes OK, launcher mode 644, production
  HEAD `9921eec` and PID 1436043 with command name `audiocpp_server`. The two
  measurement pre-check logs also record the four hashes OK. These snapshots
  do not prove the whole source tree, services or files stayed unchanged
  between checks. Measurement router snapshots list only pocket-tts, and the
  r2 final GPU snapshot is 2 MiB with no listed compute processes; those final
  observations were not enforced exit conditions. Audio health was checked
  only before the sessions, not after. See the exact
  [enforced-versus-observed mapping](#historical-safeguards-enforced-vs-observed).

## 11. Reproduction recipes

Every host step ran as root on nibbler inside a transient
`systemd-run --collect` unit with `KillMode=control-group`. The units used
`RuntimeMaxSec=900s` (build and gates) or `RuntimeMaxSec=3600s` (measurement).

### Historical safeguards: enforced vs observed

The available r2 script `/tmp/tier-b-measure.sh` (hash in §8) is the source for
**enforcement**, rather than an assumption that every item in `post-state.txt`
was asserted. Its pre-checks run under `set -euo pipefail`; `finish()` preserves
the incoming exit status and runs under `set +e`, then changes that status only
for four explicit postcondition groups (91–94). Merely printing a query result
in that function does not make it a fail-closed postcondition.

| Check | r2 precondition / intermediate enforcement | r2 exit handling and saved evidence |
| --- | --- | --- |
| Production binary/config/server/launcher contents | `sha256sum -c production.expected.sha256` must pass | Rechecked, failure sets exit 91. All four paths show OK in `session.log` and `post-state.txt`. This is four-file coverage, not a full production tree hash. |
| Offline launcher guard | Mode must be 644 | Rechecked, failure sets exit 92; `post-state.txt` prints 644. |
| Production source HEAD | Must equal `9921eecae9d442878de5bfe2b2cd25a71fbdb4ed` | Rechecked, failure sets exit 93; post snapshot prints that HEAD. No full working-tree/dirty-content check. |
| Audio process identity | PID 1436043 must have command name `audiocpp_server` | Rechecked, failure sets exit 94. PID/start time/name are also printed; start time is not an asserted condition, and PID/name is not an audio-health test. |
| Audio HTTP health | `/health` must return JSON `status=ok`, `backend=cpu` | **No post-run health request or assertion.** Only `audio-before.json` is retained. |
| Router / service | llama-swap must be active; `/running` must parse with no `qwen3.8*` model | Final `/running` response is logged but neither curl success, JSON validity, service-active state nor model absence changes the exit status. `post-state.txt` lists only pocket-tts; this is an observation, not an enforced postcondition. |
| GPU / compute processes | Used memory <2048 MiB and empty compute PID list; empty-PID assertions also follow the rope bench, expert bench and diagnostic pairs | Final memory/process queries are logged, with no final predicate or query-status assertion. `gpu-after.csv` records 2 MiB / SM 2842 MHz / 47 °C; `post-state.txt` records 2 MiB and no compute entries. Exit 0 does not enforce final GPU baseline or process absence. |
| Users / headroom | Only root/pete sessions; MemAvailable ≥33554432 KiB; ≥10737418240 B available on /data | No matching exit checks; not continuous resource monitoring. |

Exact historical local evidence roots are `/tmp/strata-tier-b-artifacts/`:

- `measure-20261008-r2/session.log`: pre-check at
  **2026-10-08T06:48:27-04:00**, four production hashes OK; script admission
  checks above. `post-state.txt`: **2026-10-08T07:05:59-04:00**; `result.txt`:
  `measure_exit=0 elapsed_s=1052`. Pre audio/router responses are in
  `audio-before.json` / `router-before.json`; the post router response is inside
  `post-state.txt`, not a separate asserted health artifact.
- `measure-20261008/session.log`: pre-check at
  **2026-10-08T06:21:52-04:00**, four hashes OK; `post-state.txt` at
  **2026-10-08T06:25:58-04:00** has hashes/mode/HEAD/PID, 2 MiB and a router
  response listing pocket-tts. `result.txt` records exit 1 from the rejected
  proof gate. The original session-1 script is not present in the inspected
  artifact copies, so the r2 script's exact assertion set is not retroactively
  certified for that run.
- `foundation/post-state.txt` at **2026-10-07T19:10:41-04:00** and
  `final-gates-20261008/post-state.txt` at **2026-10-07T21:33:12-04:00** each
  show four hashes OK, mode 644, HEAD `9921eec`, PID/name/start time and 2 MiB.
  Their `toolchain.txt` / `toolchain-source.txt` record production HEAD at
  **19:10:00** / **21:33:00** respectively; `result.txt` records gate exit 0.
  Both have pre audio/router snapshots, but neither post-state file contains a
  router response or audio-health response. The original gate heredocs are
  not present in the inspected artifact copies; gate exit 0 and query output
  alone do not establish
  the full pre/post enforcement set or a successful empty-process query.

All four post files print the audio start time as Tue Oct 6 07:28:15 2026.
That is snapshot evidence, not continuous health/liveness monitoring. No
post-run audio-health artifact is present in any of the four saved roots.
Historical session notes report supervised units and homelab preflight;
this correction performs no fresh preflight, service or host-state checks.
New enforced before/after checks and retained-logit byte revalidation belong
to the later parent-owned sta-osh.28 stage and are **not claimed here**.

The r2 script strips every ambient `STRATA_*` and sets
`PATH=/usr/local/cuda-13.3/bin:/usr/bin:/bin`; it does not retain a full
non-STRATA environment manifest. The commands below reproduce build and
measurement steps, not a claim that all plan safeguards were enforced.

**Build and native gates** (source `3af1e4c8`, isolated clone `B=/data/llm/Strata-tier-b`):

```sh
cmake -S "$B" -B "$B/build-tier-b" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-15 -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON \
  -DSTRATA_GGML_DIR=/data/llm/Strata/third_party/llama.cpp -DGGML_CUDA=OFF -DGGML_CUDA_FA=ON \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build "$B/build-tier-b" --target strata rope_parity iq_multi_parity native_grouped_parity \
  native_multi_parity native_expert_parity tier_b_rope_bench tier_b_expert_bench -j4
ctest --test-dir "$B/build-tier-b" --output-on-failure -V \
  -R '^(rope_parity|iq_multi_parity|native_grouped_parity|native_multi_parity|native_expert_parity_.*)$'
STRATA_TIER_B_TRACE=1 "$B/build-tier-b/rope_parity" --selftest
STRATA_TIER_B_TRACE=1 "$B/build-tier-b/iq_multi_parity"
```

**Measurement session** (harness source `222d39ea`; binaries checked against
the final-gate `binaries.sha256`; `PY=/data/llm/Strata/.venv/bin/python`,
`D` = fresh artifact root):

```sh
cd "$B"
timeout 300 $PY -m unittest discover -s tools/tier_b -p 'test_*.py'      # 113 tests, OK
timeout -k 10 300 build-tier-b/tier_b_rope_bench 32 10 3 1 8
timeout -k 10 300 build-tier-b/tier_b_expert_bench --warmup 10 --samples 3 --max-bytes 536870912
$PY tools/tier_b/run_pairs.py --mode diagnostic --out "$D/diag" --session-cap-s 1200
$PY tools/tier_b/run_pairs.py --mode timing --proof "$D/diag/summary.json" \
  --out "$D/timing" --session-cap-s <3480 - elapsed>
```

`run_pairs.py` defaults to both families, both workloads, 3 recorded pairs and a
600 s per-arm timeout ([source](../tools/tier_b/run_pairs.py)). Its offline tests
are in [test_pairs.py](../tools/tier_b/test_pairs.py). Timing refuses to start
without a passing diagnostic summary and matching executable hash, recorded
input paths/size/mtime and prompt identity. That metadata gate is not a
content-hash comparison of model, pack or MTP children.

## 12. Artifacts and provenance

The historical host locations and original local snapshot roots are below;
they are not fresh host availability checks. The original compact copies in
`/tmp/strata-tier-b-artifacts` omit the large logit row files. Full session-1/r2
exports are now retained separately under
`/tmp/tier-b-retained/{measure-20261008,measure-20261008-r2}`. Those are the
bytes sta-osh.28 later replayed through the corrected validators
([§13.2](#132-retained-logit-replay-through-the-corrected-validators)); the
earlier documentation-only corrections (M10, M11) revalidated nothing.

| Session | Host path | Local copy | Result |
| --- | --- | --- | --- |
| Foundation gate (.13) | `/data/llm/Strata-tests/tier-b-foundation-20261007` | `/tmp/strata-tier-b-artifacts/foundation` | Passed (superseded) |
| Final build and gates (.10) | `/data/llm/Strata-tests/tier-b-final-gates-20261008` | `/tmp/strata-tier-b-artifacts/final-gates-20261008` | `final_gates_exit=0` |
| Measurement session 1 (.11) | `/data/llm/Strata-tests/tier-b-measure-20261008` | `/tmp/strata-tier-b-artifacts/measure-20261008` | `measure_exit=1`: proof gate rejected, no timing |
| Measurement session r2 (.11) | `/data/llm/Strata-tests/tier-b-measure-20261008-r2` | `/tmp/strata-tier-b-artifacts/measure-20261008-r2` | `measure_exit=0`, 1052 s |

Key files in r2:

- `session.log` and `result.txt`: step log and exit status.
- `rope-bench.log` and `expert-bench.log`: raw native samples.
- `diag/summary.json` and `diag/arms/*/{stderr.txt,export/manifest.txt}`: proof
  verdicts, traces and exported window manifests.
- `timing/arms/*/{arm.json,argv.txt,env.json,stdout.txt,output_ids.txt}` and
  `timing.log`: raw model arms and run order; stdout retains whole-run T
  histograms and accepted/offered draft counts.
- `binaries.expected.sha256` and `harness.sha256`: binary and harness hashes.
- `diag/provenance.json` and `timing/provenance.json`: executable hash, prompt
  identity and run-level input path/stat metadata, not model/tree content hashes.
- `production.expected.sha256` and `post-state.txt`: four-file production
  baseline and timestamped post-run observations; assertion coverage is in §11.
- `audio-before.json` and `router-before.json`: pre-run service responses;
  there is no post audio-health file. The measurement post router response is
  embedded in `post-state.txt`.
- `gpu-before.csv` and `gpu-after.csv`: recorded GPU snapshots, not evidence of
  an asserted final baseline.

Final-gate files: `binaries.sha256`, `checkout.sha256`, `toolchain-source.txt`,
`CMakeCache.txt`, `configure.log`, `build.log`, `gates.log`, `rope-trace.log` and
`expert-trace.log`.

All statistics in §5–§7 were recomputed from these raw files for this report. They
agree with the harness's printed medians.

Remediation-stage artifacts are separate and additive; no historical file was
edited or deleted. Local root `/tmp/tier-b-remediation/sta-osh-28/`:
`retained-replay.json` (offline replay verdicts) and `host/` (the corrected host
build and gate run, host root
`/data/llm/Strata-tests/tier-b-remediation-20261008`).

## 13. Audit remediation and corrected-evidence disposition

An independent audit of this epic at report commit `f34a840b` (auditor
`3297118b`) returned twelve findings: six high (H1–H6), five medium (M7–M11) and
one low (L12). All twelve were fixed on branch `perf/nibbler-tier-b-kernels` and
then revalidated; none were deferred. M7 spans three executables and needed two
extra tasks (.30, .31) after its first fix proved too broad, so the twelve
findings map to sixteen tasks (sta-osh.14–.27, .30, .31).

### 13.1 Finding-to-commit map

| Finding | What was wrong | Task | Commit |
| --- | --- | --- | --- |
| H1 | Identity arms were compared as decoded floats, so a signed-zero (−0.0 versus +0.0) mismatch could pass as "bitwise" | sta-osh.14 | `cfb799c7` |
| H2 | Empty or malformed completed windows (zero supported T, zero row bytes, bad counts, overlapping row spans) counted as proof | sta-osh.15 | `5346c6dd` |
| H3 | An `export_error` or a missing/invalid completion footer was treated as ordinary truncation instead of rejecting the proof | sta-osh.16 | `d8e8dfea` |
| H4 | Timing admitted any summary carrying a `proof: true` key, without checking contracts, bounds, settings or nested verdicts | sta-osh.17 | `c40b7e09` |
| H5 | A rejected diagnostic or timing pair did not stop the session, so later pairs still launched | sta-osh.18 | `c40b7e09` |
| H6 | CLI timeout teardown could signal the wrong process group and used unbounded waits | sta-osh.19 | `3e50a2b1` |
| M7 (rope) | `tier_b_rope_bench` exited on CUDA error, leaking streams/events/graphs and leaving selector state changed | sta-osh.20 | `ae67e36c` |
| M7 (expert) | `tier_b_expert_bench` leaked partially constructed benches and capture failures | sta-osh.21 | `7b8f0234` |
| M7 (parity) | `iq_multi_parity` had the same exit-on-error unwind bypass | sta-osh.22 | `8b7217a7` |
| M7 (IQ policy) | IQ kernel launch failures could not unwind a scoped caller; a first attempt at global exception conversion was rejected as too broad and narrowed to an opt-in thread-local policy for audited standalone callers | sta-osh.30, sta-osh.31 | `6fb32311`, `a5d7fb5a` |
| M8 | The expert benchmark's device-allocation cap counted weights only, not guards/scratch/index/pointer buffers | sta-osh.23 | `a5d7fb5a` |
| M9 | Unknown/error capture states and `format_error` were accepted as trace evidence | sta-osh.24 | `c40b7e09` |
| M10 | The report's clean-run draft statistics and DraftPolicy timing-feedback caveats were wrong or missing | sta-osh.25 | `f05dd6ae` |
| M11 | The report conflated enforced postconditions with observed snapshots and overstated provenance identity | sta-osh.26 | `5443dfbd` |
| L12 | `--pair gu,dt` parsing mis-split the argument | sta-osh.27 | `a5d7fb5a` |

Every commit above is a code, test or document change. One further commit,
`590ff010`, is **Beads-only** (a single `.beads/issues.jsonl` edit): it recorded
the decision to reject the first global exception-conversion attempt as too
broad and split the remaining M7 work into sta-osh.30 and sta-osh.31. It
implements nothing and is listed here for decision provenance, not as a fix.

The two documentation findings (M10, M11) changed §7.3, §8, §9 and §10 of this
report; no measured number in §5–§7 was altered by any of these commits.

### 13.2 Retained-logit replay through the corrected validators

sta-osh.28 replayed the **original retained host export bytes** through the
fully corrected validators at code commit `62bf499d` (clean tree). This is a
replay, not a new generation run: no model, GPU or CLI arm was executed, and no
historical artifact was modified. The inputs were the full retained exports
under `/tmp/tier-b-retained/{measure-20261008,measure-20261008-r2}`
([§12](#12-artifacts-and-provenance)), read only.

Validator identity at replay time:

| File | sha256 |
| --- | --- |
| `tools/tier_b/run_pairs.py` | `2dac4521fee4a2bf2c474ef0fd85caf4c7f72b8690feccae9242dd3203149476` |
| `tools/tier_b/test_pairs.py` | `6ec941f31b3f0d688a2de07735d0495a9848fdd704777a0f4abfa109b072f2d6` |
| `include/strata/kernels/tier_b_diagnostics.hpp` | `1d76383de526be43353f09f131322ab4cce01538179a5d2a5081232d6f68a2ef` |

Verdicts on the retained bytes (`retained-replay.json`):

| Session | Family / workload | Matched windows | Rows bitwise | `proof` | `full_proof` |
| --- | --- | --- | --- | --- | --- |
| r2 | f / short_code | 16 | yes | true | false |
| r2 | f / long_doc | 16 | yes | true | false |
| r2 | h / short_code | 16 | yes | true | false |
| r2 | h / long_doc | 16 | yes | true | false |
| session 1 | f / short_code | 16 | yes | true | false |
| session 1 | f / long_doc | 16 | yes | true | false |
| session 1 | h / short_code | 16 | yes | true | false |
| session 1 | h / long_doc | 14 | yes | true | false |

Every cell reports unmatched A=0 and B=0 and the reason "controlled correctness
proven (coverage incomplete: truncated prefix/windows)". The four r2 verdicts
reproduce [§6](#6-model-proof-reachability-and-bitwise-windows); the four
session-1 verdicts reproduce the 16/16/16/14 replay reported in
[deviation 1](#8-deviations-from-the-plan). The §6 bitwise claims therefore
survive the corrected validators unchanged, including the corrected
signed-zero-sensitive byte comparison.

`retained-replay.json` also records the sha256 of each replayed `manifest.txt`
and `rows.bin`, pinning exactly which bytes produced these verdicts. Both arms
of a cell share one `rows.bin` hash wherever the exports are byte-identical,
which is the bitwise result itself; the distinct manifest hashes reflect
differing per-arm window metadata.

### 13.3 Corrected isolated build and correctness gates

sta-osh.28's host stage ran as root on nibbler in a transient
`systemd-run --collect` unit, in the isolated checkout `/data/llm/Strata-tier-b`
with a **separate** build directory `build-tier-b-fix`, writing to the fresh
artifact root `/data/llm/Strata-tests/tier-b-remediation-20261008`. The source
came from a local bundle pinned to `62bf499d`; `result.txt` records
`remediation_exit=0 elapsed_s=116` (2026-10-08T15:48:12 to 15:50:08 −04:00).

| Step | Result |
| --- | --- |
| Configure | CUDA 13.3.73 with host GNU 15.3.0, sm120, `FETCHCONTENT_FULLY_DISCONNECTED=ON`; configure 2.0 s |
| Build | 149 ninja actions, all eight targets |
| CTest (required gates) | **15/15 passed, 0 failed**, 7.52 s — the same case set as the §4 final gate |
| Offline harness suite on host | 260 tests, OK, 1 skip (the retained-artifact test, whose inputs are not on the host) |
| `tier_b_rope_bench` (bounded) | 96 cases, 96 timed, **0 gate/replay failures** |
| `tier_b_expert_bench` (default) | 7 manifest pairs, **0 parity failures**, 7 timed |
| `--pair 18,20` (L12 fix on real CUDA) | parsed to `IQ3_XXS/IQ4_NL`, 0 parity failures, 1 timed pair |
| Allocation-budget rejection (M8 fix, cap 1 MiB) | all 7 pairs rejected **before device-buffer allocation**, `budget_reject_exit=1` |

The budget log shows the corrected accounting explicitly, for example
`IQ3_XXS/IQ4_NL : explicit device buffers 139670440 B (weight working set
139264000 B, auxiliary+guards 406440 B)`. That 406440 B auxiliary term is what
M8 reported missing. The binary still prints its own limit: the budget counts
`cudaMalloc` buffers and guards and excludes opaque CUDA graph/exec/stream/event
and driver allocations, so it is not a total-VRAM cap.

Corrected binary hashes are in `host/binaries.corrected.sha256` and the
harness/tool file hashes in `host/harness.corrected.sha256`.

### 13.4 Binary separation

The corrected build produced different binaries from the measured ones. The
measured values in §5 and §7 belong to the §2 binaries only:

| Binary | Build | sha256 |
| --- | --- | --- |
| `strata` | measured, `3af1e4c8` | `59f0d53627773b3dac02fac861421441ee11848d27a0508216e905e984c9a7f7` |
| `strata` | corrected, `62bf499d` | `1d6d5c9270602e8597f872f99c2a77bf9efb922d9856d8da5bc3e392cb9cb962` |
| `tier_b_rope_bench` | measured | `82467382094a5120b95de98632e7758e6d19a79f6211c8ccf6d08f496eafa42f` |
| `tier_b_rope_bench` | corrected | `74b4451f7e76754da6a0a5e5b0c872992df57ef57c62af50fe2b41d2dc766424` |
| `tier_b_expert_bench` | measured | `6d1bcd3a463674564d6201625a40304242d240a7e8b8332e85ae1f886ad841a2` |
| `tier_b_expert_bench` | corrected | `88352ac354f107d5dd6d88cc97c6681f73059c7cb863181efcb6b9d2e0168755` |

The measured hashes repeat [§2](#2-pins-hardware-and-toolchain);
`host/binaries.corrected.sha256` additionally covers the five corrected parity
executables.

The corrected build's bench runs used small bounded settings for a liveness and
cleanup check, not the measurement protocol: 96 rope cases and 7 expert pairs in
about 4 s total, against far larger §5 sample counts. **Their printed
microsecond values are not comparable with §5 and are not quoted as results
anywhere in this report.** No corrected-build model arm was run, so §7 has no
corrected counterpart at all.

### 13.5 Enforced host preconditions and postconditions

Unlike r2 ([§11](#historical-safeguards-enforced-vs-observed)), the remediation
stage asserted its postconditions fail-closed. `post-state.txt` labels each
group `(enforced)`:

| Check | Pre | Post (enforced) |
| --- | --- | --- |
| Four production file hashes | all OK at 15:48:12 | all OK at 15:50:08 |
| Launcher guard mode | — | 644 |
| Production source HEAD | — | `9921eecae9d442878de5bfe2b2cd25a71fbdb4ed` |
| Audio process identity | — | PID 1436043, `audiocpp_server`, start Tue Oct 6 07:28:15 2026 |
| **Audio HTTP health** | `audio-before.json` | `status=ok`, `backend=cpu`, `models=2` — the r2 gap M11 identified is closed for this stage |
| Router `/running` | `router-before.json` | llama-swap `active`, only `pocket-tts` ready, no `qwen3.8*` |
| GPU | 2 MiB used, 0 % util, driver 610.57.04 on the RTX 5060 Ti | 2 MiB, 2610 MHz, 31 °C |
| Timers | — | `no strata timers` |
| Sessions | only root (tailscaled, pts/3) | — |

These remain before/after observations of two instants, now exit-gated; they
still do not establish continuously unchanged state in between, and they cover
four production files rather than a whole-tree hash. The revalidation stage
made no production, audio, router, model-download, deployment or timer change.

### 13.6 What the revalidation does not establish

- **No new timing.** No model pair and no measurement-protocol benchmark was
  rerun. Every number in §5–§7 is still the single-session, n=3, cold-process
  observation described there, with the same thin statistics.
- **No retroactive enforcement.** The enforced postconditions in §13.5 apply to
  the remediation stage only. The r2 enforced-versus-observed mapping in §11 is
  unchanged, and r2's missing post-run audio health stays missing.
- **Coverage is unchanged.** The replay inherits the historical 16-window
  truncated prefix, so every cell is still `full_proof=false`; windows past 16
  rely on whole-run output-ID identity.
- **Provenance gaps persist.** No model shard, pack, MTP, profile or ggml tree
  content hash exists for the historical runs, and a replay cannot create one
  after the fact ([deviation 2](#8-deviations-from-the-plan)).
- **Historical summaries are now inadmissible.** The H4 fix stamps
  `tier-b-proof-admission/1` into both `provenance.json` and `summary.json`,
  with no backward compatibility. The session-1 and r2 `diag/summary.json` files
  predate the stamp, so any future timing run will refuse them and require a
  fresh diagnostic run. The §13.2 verdicts came from replaying raw export bytes,
  which is a separate path from that admission gate.
- **Clean timing arms were never logit-validated.** The replay covers
  diagnostic exports only; clean runs exported no windows
  ([§7.3](#73-clean-run-draft-statistics-and-timing-feedback)).
- **Dispositions unchanged.** [§10](#10-disposition) still recommends keeping
  both upstream defaults, with a focused follow-up before any sub16 change. The
  remediation raises confidence in the correctness verdicts; it adds no
  performance evidence.
- **Audit status.** An independent final audit of this remediation is required
  before the epic closes, and was not complete when this section was written.
