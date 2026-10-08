# Nibbler Tier-B first-wave kernel report

**Bench host:** nibbler (bare-metal AMD Ryzen 9 9900X, RTX 5060 Ti 16 GB).
**Orchestration:** paseo-nibbler VM (offline tests, harness, artifact copies).
**Dates:** 2026-10-07 (foundation and final native gates, US Eastern) to
2026-10-08 (measurement).
**Scope:** focused same-binary validation of two upstream kernel families
already present in engine `e8ca9af`. This is not a port, a merge, a production
change or a quality certification. It covers one GPU and the single-session CLI
with fresh processes. It does not cover Tier-C serving/KV work.

Protocol, contracts and bounds were declared beforehand in the
[Tier-B plan](NIBBLER_TIER_B_PLAN.md). Only one contract changed after host data
was seen: window matching, which changed before any model timing ran. See
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
| Original fork (historical only) | `9921eecae9d4` (0.1.39); production source HEAD unchanged throughout |
| Final native build source | `3af1e4c8` (all `src`/`include`/`CMakeLists.txt`/`cmake` changes) |
| Measurement harness source | `222d39ea`, clean tree; `git diff 3af1e4c8 222d39ea -- src include CMakeLists.txt cmake` was empty (checked in-session) |
| Build dir | `/data/llm/Strata-tier-b/build-tier-b` (isolated clone, branch `perf/nibbler-tier-b-kernels`) |
| `strata` sha256 | `59f0d53627773b3dac02fac861421441ee11848d27a0508216e905e984c9a7f7` |
| `tier_b_rope_bench` sha256 | `82467382094a5120b95de98632e7758e6d19a79f6211c8ccf6d08f496eafa42f` |
| `tier_b_expert_bench` sha256 | `6d1bcd3a463674564d6201625a40304242d240a7e8b8332e85ae1f886ad841a2` |
| Compilers | CUDA 13.3 (V13.3.73), g++-15 15.3.0, Release, `CMAKE_CUDA_ARCHITECTURES=120` |
| ggml source | read-only reuse via `STRATA_GGML_DIR=/data/llm/Strata/third_party/llama.cpp` (no fetch); configure reports `ggml version: 0.24.0`. The tree has no embedded `.git`, so the configure-time "ggml commit 9921eeca" is the enclosing production checkout's HEAD, **not** a ggml ref. No separate content hash of that tree was captured. |
| GPU (fresh, 2026-10-08) | NVIDIA GeForce RTX 5060 Ti, UUID `GPU-a29bb3c9-…`, driver 610.57.04, 16311 MiB, 36 SMs, L2 32 MiB (33554432 B) |
| GPU state before / after (r2) | 2 MiB used, 0 %, SM 180 MHz, mem 405 MHz, 31 °C, 3.95 W / 2 MiB used, SM 2842 MHz, 47 °C, no compute processes |
| CPU / RAM / OS | Ryzen 9 9900X (AVX-512), about 123 GiB RAM: from the [Tier-A record](NIBBLER_TIER_A.md#2-hardware-and-software-state); not re-captured |

All benchmark/measurement binaries were verified against the hashes recorded in
the final gate run (`sha256sum -c` in-session) before use.

## 3. Families under test

Both families are already upstream and **on by default**. Each pair runs the
same executable, inputs and settings, and exactly one opt-out variable differs:

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

**Fixed settings:** argv was identical across arms apart from the temporary
token-file path.

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

**Prompts:** short_code was 42 IDs (hash `563cf85e6f7c1247`, 256 new tokens) and
long_doc was 8887 IDs (`0157820299822d1b`, 512 new tokens).

**Run validity:**

- All 32 arms exited 0 with no errors.
- Within each workload, every arm produced the same output hash and length.
- Draft acceptance was 0.931 on short_code in all arms and 0.668–0.673 on
  long_doc.

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
  unresolved, and this report does not attribute it.

## 8. Deviations from the plan

1. **Window-matching contract** (amended before any model timing). Session 1
   (unit `strata-tier-b-measure-20261008`, source `3437d96f`, 246 s, exit 1)
   compared windows by ordinal.
   - **What failed:** h/long_doc failed at ordinal 13, where the same position
     was windowed as T=4 in one arm and T=3 in the other.
   - **What still matched:** whole-run IDs matched, and the other three cells
     passed.
   - **Cause:** speculative draft lengths are not deterministic run to run.
   - **Response:** per the no-retry rule, timing did not start. The contract
     changed to matching by `(pos, T, input_hash)` with at least 12 matched
     windows, bitwise on matched rows, and whole-run IDs still required (commit
     `222d39ea`, 9 regression tests). Replaying the session-1 exports under the
     new rule gave 16/16/16/14 matched, all bitwise.
   - **Rerun:** session r2 then passed. This is the only contract changed after
     seeing host data, and it relaxes window alignment only, not the bitwise
     identity requirement.
2. **Model input identity:** model inputs were identified by path, size and
   mtime (`timing/provenance.json`), not by content sha256. The 54.8 GB and
   28.8 GB shards were not re-hashed. The pack is identified by its directory
   only.
3. **Representative expert groups:** the plan asked for "proved representative
   groups". The bench used a declared G8/T2 shape, and the model traces carry
   only nominal capacities ([§5.2](#52-native-sub16-expert-layout)).
4. **Session scripts:** neither the measurement script nor the final-gate
   heredoc is committed. The r2 measurement script was `/tmp/tier-b-measure.sh`
   (sha256 `00dea19031297350106d3322dafacb5d3e5eb7a8a102ff55fe36468dfc558a08`,
   on paseo-nibbler and on the host). Their essential steps are reproduced in
   [§11](#11-reproduction-recipes).

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
- **Timing in clean runs:** clean runs do not record per-window T sequences, so
  equal draft-length distributions between timed A and B arms are not
  established. Draft acceptance rates and output hashes were equal.
- **No warm or steady-state model timing:** no profiler (nsys) or kernel-share
  breakdown of model decode was taken, so the gap between native and model
  results is unexplained.
- **Thin statistics:** n=3 pairs per cell, with a single session and a single
  day.

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
- **Production state:** production source HEAD `9921eec`, the
  binary/config/server/launcher hashes, the launcher guard (mode 644) and the
  audio process (PID 1436043) were unchanged before and after every host
  session. No qwen3.8 model was loaded in the router, and the GPU was back to
  2 MiB with no compute processes.

## 11. Reproduction recipes

Every host step ran as root on nibbler inside a transient
`systemd-run --collect` unit with `KillMode=control-group`. The units used
`RuntimeMaxSec=900s` (build and gates) or `RuntimeMaxSec=3600s` (measurement).

Each unit pre-checked the following and aborted on any failure:

- production hashes and the launcher mode;
- production source HEAD `9921eec` and the audio PID;
- that no unexpected users were logged in;
- llama-swap active, audio health ok, and no qwen3.8 loaded;
- RAM and disk headroom;
- GPU below 2 GiB used, with no compute processes.

A matching post-state check ran on exit. Every ambient `STRATA_*` was unset and
`PATH=/usr/local/cuda-13.3/bin:/usr/bin:/bin`. The homelab
`scripts/agent-preflight.sh` was run from paseo-nibbler before launching host
sessions.

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
without a passing proof summary for the same executable hash, inputs and prompts.

## 12. Artifacts and provenance

All artifacts are on nibbler. The local copies omit the large logit row files.

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
- `timing/arms/*/{arm.json,argv.txt,env.json,output_ids.txt}` and `timing.log`:
  raw model arms and run order.
- `binaries.expected.sha256` and `harness.sha256`: binary and harness hashes.
- `production.expected.sha256` and `post-state.txt`: production baseline and
  post-run state.
- `gpu-before.csv` and `gpu-after.csv`: GPU state before and after.

Final-gate files: `binaries.sha256`, `checkout.sha256`, `toolchain-source.txt`,
`CMakeCache.txt`, `configure.log`, `build.log`, `gates.log`, `rope-trace.log` and
`expert-trace.log`.

All statistics in §5–§7 were recomputed from these raw files for this report. They
agree with the harness's printed medians.
